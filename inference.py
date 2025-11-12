# ==========================================================
# DLCV HW3 P2 — Final Inference Script (for hw3_2.sh)
# ----------------------------------------------------------
# ✔ interface: python3 inference.py $1 $2 $3
#    $1 = folder containing test images (e.g. images/val/)
#    $2 = output json path
#    $3 = baseline decoder_model.bin
#
# ✔ projector + LoRA are loaded from repo relative path:
#       output_p2/projector_final.pth
#       output_p2/decoder_lora_final.pth
#
# ✔ full float32 pipeline (no bf16), avoid dtype mismatch
# ✔ prompt: "Describe the image: "
# ✔ decoding: temperature=0.7, top_p=0.95, repetition_penalty=1.1
# ✔ output json: { "xxxxxx.jpg": "caption", ... }
# ==========================================================

import os
import sys
import json
from typing import Dict

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from decoder import Decoder, Config
from tokenization_qwen3 import Qwen3Tokenizer
import importlib.util


# ------------------------- 動態載入 llava 子模組 -------------------------
def _load_submodule(mod_name: str, file_path: str, package: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def build_vision_and_projector(repo_root: str):
    """固定使用訓練時的視覺設定：Vit-B/16 layer -2 + mlp2x_gelu projector"""
    mm_vision_tower = "openai/clip-vit-base-patch16"
    mm_vision_select_layer = -2
    mm_projector_type = "mlp2x_gelu"
    mm_hidden_size = 768
    hidden_size = 1024

    enc_dir = os.path.join(repo_root, "llava", "model", "multimodal_encoder")
    proj_dir = os.path.join(repo_root, "llava", "model", "multimodal_projector")

    _load_submodule(
        "llava.model.multimodal_encoder.clip_encoder",
        os.path.join(enc_dir, "clip_encoder.py"),
        "llava.model.multimodal_encoder"
    )
    enc_builder = _load_submodule(
        "llava.model.multimodal_encoder.builder",
        os.path.join(enc_dir, "builder.py"),
        "llava.model.multimodal_encoder"
    )
    proj_builder = _load_submodule(
        "llava.model.multimodal_projector.builder",
        os.path.join(proj_dir, "builder.py"),
        "llava.model.multimodal_projector"
    )

    cfg = type("Cfg", (), {
        "mm_vision_tower": mm_vision_tower,
        "mm_vision_select_layer": mm_vision_select_layer,
        "mm_projector_type": mm_projector_type,
        "mm_hidden_size": mm_hidden_size,
        "hidden_size": hidden_size
    })()

    vision_tower = enc_builder.build_vision_tower(cfg, delay_load=False).eval()
    projector = proj_builder.build_vision_projector(cfg).eval()
    image_processor = vision_tower.image_processor
    return vision_tower, projector, image_processor


# ------------------------- clean caption -------------------------
def clean_caption(text: str) -> str:
    out = []
    for ch in text:
        if 32 <= ord(ch) <= 126:
            out.append(ch)
        else:
            out.append(" ")
    s = " ".join("".join(out).split())
    return s.strip()


# ------------------------- repetition penalty -------------------------
def apply_repetition_penalty(logits, generated_ids, penalty):
    if not generated_ids or penalty == 1.0:
        return logits
    logits_view = logits[0]
    for tid in set(generated_ids):
        logits_view[tid] /= penalty
    return logits


# ------------------------- nucleus sampling -------------------------
def sample_top_p(probs, top_p):
    probs = probs.clone()
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    total = probs.sum()
    if total <= 0:
        return int(torch.argmax(probs).item())
    probs = probs / total

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    mask = cumulative > top_p
    mask[1:] = mask[:-1].clone()
    mask[0] = False
    sorted_probs[mask] = 0.0

    new_total = sorted_probs.sum()
    if new_total > 0:
        sorted_probs /= new_total
    else:
        return int(torch.argmax(probs).item())

    next_local = torch.multinomial(sorted_probs, 1)
    next_token = sorted_idx[next_local]
    return int(next_token.item())


# ==========================================================
# Main (符合 hw3_2.sh)
# ==========================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("images_root")     # $1
    parser.add_argument("output_json")     # $2
    parser.add_argument("decoder_weight")  # $3 (baseline decoder_model.bin)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🖥️ device = {device}")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    cfg = Config()

    # ------------------ Load Vision Tower + Projector ------------------
    print("📷 Building vision & projector...")
    vision_tower, projector, image_processor = build_vision_and_projector(repo_root)
    vision_tower.to(device).eval()
    projector.to(device).eval()

    # projector 權重（相對路徑）
    projector_path = os.path.join(repo_root, "output_p2", "projector_final.pth")
    projector.load_state_dict(torch.load(projector_path, map_location="cpu"), strict=True)

    # ------------------ Load Decoder + baseline + LoRA ------------------
    print("📥 Loading decoder baseline...")
    decoder = Decoder(cfg).to(device)
    decoder.load_state_dict(torch.load(args.decoder_weight, map_location="cpu"), strict=False)

    lora_path = os.path.join(repo_root, "output_p2", "decoder_lora_final.pth")
    print("📥 Loading decoder LoRA...")
    decoder.load_state_dict(torch.load(lora_path, map_location="cpu"), strict=False)

    decoder.eval()
    dtype = decoder.embed_tokens.weight.dtype

    # ------------------ Tokenizer & Prompt ------------------
    tok = Qwen3Tokenizer(
        os.path.join(repo_root, "vocab.json"),
        os.path.join(repo_root, "merges.txt")
    )

    PROMPT = "Describe the image: "
    prompt_ids = torch.tensor([tok.encode(PROMPT)], dtype=torch.long, device=device)
    prompt_emb = decoder.embed_tokens(prompt_ids).to(dtype)

    EOS_ID = 151645
    PAD_ID = 151643

    # ------------------ List input images ------------------
    img_files = sorted([f for f in os.listdir(args.images_root) if f.endswith(".jpg")])

    print(f"🪄 Generating captions for {len(img_files)} images...")
    preds: Dict[str, str] = {}

    # ------------------ Loop over all images ------------------
    for fname in tqdm(img_files):
        stem = os.path.splitext(fname)[0]
        img_path = os.path.join(args.images_root, fname)
        image = Image.open(img_path).convert("RGB")

        # vision → projector
        with torch.no_grad():
            pixel = image_processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
            feats = vision_tower(pixel)       # (1, T, 768)
            feats = feats.mean(dim=1)         # (1, 768)
            vis_emb = projector(feats).unsqueeze(1).to(dtype)  # (1,1,1024)

        # prefix embeddings
        inputs_embeds = torch.cat([vis_emb, prompt_emb], dim=1)

        generated_ids = []
        MAX_NEW_TOKENS = 60
        TEMP = 0.7
        TOP_P = 0.95
        REP = 1.1

        for _ in range(MAX_NEW_TOKENS):
            out = decoder(inputs_embeds=inputs_embeds)
            logits = out[:, -1, :]

            logits = torch.clamp(logits, -50, 50)
            logits = logits / TEMP
            logits = apply_repetition_penalty(logits, generated_ids, REP)
            logits[0, PAD_ID] -= 5

            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0, posinf=0, neginf=0)

            next_id = sample_top_p(probs[0], TOP_P)

            if next_id == EOS_ID:
                break

            generated_ids.append(next_id)

            next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            next_emb = decoder.embed_tokens(next_token).to(dtype)
            inputs_embeds = torch.cat([inputs_embeds, next_emb], dim=1)

        caption = clean_caption(tok.decode(generated_ids))
        preds[f"{stem}.jpg"] = caption

    # ------------------ Save output json ------------------
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved predictions to {args.output_json}")


if __name__ == "__main__":
    main()
