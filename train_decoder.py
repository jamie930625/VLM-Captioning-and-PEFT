# train_projector_decoder.py
# ------------------------------------------------------------
# Train projector (mlp2x_gelu) + Decoder LoRA (Q/K/V/O, r=16)
# - Vision: CLIP -> select layer (-2) -> mean-pool -> projector -> 1-token prefix
# - Text: prompt + caption; loss on caption tokens only (prompt and PAD masked)
# - Saves projector_state_dict.pth and lora_only.pth
# - Total trainable parameters < 10M (about 5.5M)
# ------------------------------------------------------------

import os
# optional speed-up; safe to skip if unavailable
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# silence transformers' torch.load version warning
import transformers
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.utils.import_utils._torch_load_is_safe = lambda *a, **k: True

import sys
import json
import math
import argparse
import importlib.util
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import loralib as lora

from decoder import Decoder, Config
from tokenization_qwen3 import Qwen3Tokenizer


# ------------------------- load only the needed llava modules -------------------------
def _load_submodule(mod_name: str, file_path: str, package: str):
    """Load only the required llava files without running llava/__init__.py."""
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[mod_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_vision_and_projector(repo_root: str,
                               mm_vision_tower: str,
                               mm_vision_select_layer: int,
                               mm_projector_type: str,
                               mm_hidden_size: int,
                               hidden_size: int):
    enc_dir = os.path.join(repo_root, "llava", "model", "multimodal_encoder")
    proj_dir = os.path.join(repo_root, "llava", "model", "multimodal_projector")

    # load clip_encoder before builder to keep relative imports working
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

    # build a config matching the provided interface
    cfg = type("Cfg", (), {
        "mm_vision_tower": mm_vision_tower,
        "mm_vision_select_layer": mm_vision_select_layer,
        "mm_projector_type": mm_projector_type,
        "mm_hidden_size": mm_hidden_size,
        "hidden_size": hidden_size
    })()

    vision_tower = enc_builder.build_vision_tower(cfg, delay_load=False).eval()
    projector = proj_builder.build_vision_projector(cfg).train()  # the projector is trained
    image_processor = vision_tower.image_processor
    return vision_tower, projector, image_processor


# ------------------------- dataset and collate -------------------------
class CocoCapDataset(Dataset):
    """
    Reads images directly (images/{split}/xxxx.jpg) instead of cached .pt files.
    JSON：{"annotations":[{"image_id": int, "caption": str}, ...]}
    Adds a fixed English prompt to keep captions in English and in one style.
    """
    def __init__(self, data_root: str, split: str, tokenizer: Qwen3Tokenizer, prompt: str):
        super().__init__()
        self.img_dir = os.path.join(data_root, "images", split)
        self.anns = json.load(open(os.path.join(data_root, f"{split}.json")))["annotations"]
        self.tok = tokenizer
        self.prompt = prompt

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx: int):
        ann = self.anns[idx]
        img_id = int(ann["image_id"])
        img = Image.open(os.path.join(self.img_dir, f"{img_id:012d}.jpg")).convert("RGB")

        # text: prompt + caption + EOS
        caption = ann["caption"].strip()
        full_text = self.prompt + caption + "<|im_end|>"

        # prompt length in tokens, used to mask the loss
        prompt_len = len(self.tok.encode(self.prompt))
        ids_full = torch.tensor(self.tok.encode(full_text), dtype=torch.long)

        return img, ids_full, img_id, prompt_len


def collate_pad(batch: List[Tuple[Image.Image, torch.Tensor, int, int]],
                pad_id: int = 151643):
    images, ids_list, img_ids, prompt_lens = zip(*batch)
    T = max(len(t) for t in ids_list)
    padded, labels = [], []

    for ids, p_len in zip(ids_list, prompt_lens):
        if len(ids) < T:
            pad = torch.full((T - len(ids),), pad_id, dtype=torch.long)
            ids = torch.cat([ids, pad], 0)

        # shift by one for teacher forcing: input=[:-1], target=[1:]
        # labels have the same length as ids and are sliced in the training loop
        lab = ids.clone()

        # mask targets in the prompt span (no loss)
        # targets are ids[1:], so the prompt covers positions before max(0, p_len-1)
        if p_len > 0:
            lab[:p_len] = -100

        # mask PAD as well
        lab = torch.where(ids == pad_id, torch.full_like(ids, -100), lab)

        padded.append(ids)
        labels.append(lab)

    return list(images), torch.stack(padded, 0), torch.stack(labels, 0), list(img_ids), list(prompt_lens)


# ------------------------- helpers -------------------------
def count_trainable_params(models: List[nn.Module]) -> int:
    total_trainable = 0
    for m in models:
        total_trainable += sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable parameters total: {total_trainable/1e6:.2f}M")
    return total_trainable


# ------------------------- training -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="root folder with train/val json files and images/train, images/val")
    parser.add_argument("--baseline_weight", type=str, required=True,
                        help="provided decoder_model.bin")
    parser.add_argument("--output_lora", type=str, required=True,
                        help="output path for LoRA weights (.pth)")
    parser.add_argument("--output_projector", type=str, required=True,
                        help="output path for projector weights (.pth state_dict)")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    # vision settings (same as inference)
    parser.add_argument("--mm_vision_tower", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")
    parser.add_argument("--mm_hidden_size", type=int, default=768)

    # training prompt (fixed English prompt)
    parser.add_argument("--prompt", type=str, default="Describe the image: ")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    cfg = Config()

    # 1) Vision / Projector / Processor
    vision_tower, projector, image_processor = build_vision_and_projector(
        repo_root=repo_root,
        mm_vision_tower=args.mm_vision_tower,
        mm_vision_select_layer=args.mm_vision_select_layer,
        mm_projector_type=args.mm_projector_type,
        mm_hidden_size=args.mm_hidden_size,
        hidden_size=cfg.hidden_size
    )
    vision_tower.to(device).eval()   # frozen
    projector.to(device).train()     # trained

    # 2) decoder with baseline weights; only LoRA is trainable (Q/K/V/O, r=16, alpha=32, dropout=0.05 in decoder.py)
    dec = Decoder(cfg).to(device)
    dec.load_state_dict(torch.load(args.baseline_weight, map_location="cpu"), strict=False)
    lora.mark_only_lora_as_trainable(dec, bias="none")  # LoRA weights only

    # count trainable parameters (LoRA + projector)
    total_trainable = count_trainable_params([dec, projector])
    if total_trainable > 10_000_000:
        raise SystemExit(f"Trainable parameters exceed 10M ({total_trainable}). Reduce the LoRA rank or projector depth.")

    # 3) Data / Tokenizer
    tok = Qwen3Tokenizer(os.path.join(repo_root, "vocab.json"),
                         os.path.join(repo_root, "merges.txt"))

    train_set = CocoCapDataset(args.data_root, "train", tok, prompt=args.prompt)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_pad
    )

    # 4) Optim / AMP / Loss
    # optimize projector and LoRA parameters only
    optim = AdamW(
        list(projector.parameters()) + [p for p in dec.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd
    )
    scaler = GradScaler(device="cuda", enabled=not args.bf16)  # no scaler with bf16
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    use_bf16 = bool(args.bf16 and torch.cuda.is_available())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"compute dtype: {'bf16' if use_bf16 else 'fp32'}")

    dec.train()
    projector.train()
    vis_prefix_len = 1  # one visual prefix token after mean pooling

    for ep in range(1, args.epochs + 1):
        running = 0.0
        for it, (images, ids_full, labels_full, _, prompt_lens) in enumerate(train_loader, 1):
            # ------- image features (no gradient) -------
            with torch.no_grad():
                pixel = image_processor(images=list(images), return_tensors="pt")["pixel_values"].to(device)
                feats = vision_tower(pixel)            # (B, T_patch, mm_hidden) from the selected layer
                feats = feats.mean(dim=1)              # (B, mm_hidden), mean-pooled to one vector
            # the projector is trained, so gradients are on
            vis_emb = projector(feats).unsqueeze(1).to(compute_dtype)  # (B,1,H)

            # ------- text (teacher forcing, prompt span masked) -------
            # ids_full: [prompt + caption + EOS + PAD...]
            input_ids = ids_full[:, :-1].to(device)    # model input
            target_ids = labels_full[:, 1:].to(device) # targets (prompt/PAD masked in collate)

            with torch.no_grad():
                txt_emb = dec.embed_tokens(input_ids).to(compute_dtype)  # (B, L-1, H)

            # ------- visual prefix + text embeddings -------
            inputs_embeds = torch.cat([vis_emb, txt_emb], dim=1)  # (B, 1 + L-1, H)

            # ------- forward and loss on target positions -------
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = dec(inputs_embeds=inputs_embeds)           # (B, 1+L-1, V)
                # drop the prefix position to align with target_ids, shape (B, L-1)
                logits_text = logits[:, vis_prefix_len:, :]         # (B, L-1, V)
                loss = loss_fn(logits_text.reshape(-1, logits_text.size(-1)),
                               target_ids.reshape(-1))

            # ------- backward and update -------
            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if it % args.accum_steps == 0:
                if use_bf16:
                    optim.step()
                else:
                    scaler.step(optim)
                    scaler.update()
                optim.zero_grad(set_to_none=True)

            running += loss.item()
            if it % 50 == 0:
                print(f"Epoch {ep} | step {it} | loss {running/it:.4f}")

        # ------- save every epoch -------
        os.makedirs(os.path.dirname(args.output_lora), exist_ok=True)
        os.makedirs(os.path.dirname(args.output_projector), exist_ok=True)

        # save LoRA weights only
        torch.save(lora.lora_state_dict(dec, bias="none"), args.output_lora)
        # save the full projector state_dict
        torch.save(projector.state_dict(), args.output_projector)

        print(f"Saved LoRA to: {args.output_lora}")
        print(f"Saved projector to: {args.output_projector}")

    print("Training finished. Projector + LoRA weights are ready for inference.")


if __name__ == "__main__":
    main()
