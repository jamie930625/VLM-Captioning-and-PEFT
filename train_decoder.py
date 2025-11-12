# train_projector_decoder.py
# ------------------------------------------------------------
# Train projector (mlp2x_gelu) + Decoder LoRA (Q/K/V/O, r=16)
# - Vision: CLIP -> select layer (-2) -> mean-pool -> projector -> 1-token prefix
# - Text: prompt + caption，loss 只計 caption（遮掉 prompt & PAD）
# - 保存：projector_state_dict.pth + lora_only.pth
# - 總可訓練參數 < 10M（~5.5M）
# ------------------------------------------------------------

import os
# 可選加速；若環境沒有也不致命
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 避免 transformers 在 torch.load 上卡版本警告（你目前 torch=2.9）
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
from torch.amp import GradScaler, autocast   # ✅ 新版 AMP 介面
import loralib as lora

from decoder import Decoder, Config
from tokenization_qwen3 import Qwen3Tokenizer


# ------------------------- 動態載入 llava 子模組 -------------------------
def _load_submodule(mod_name: str, file_path: str, package: str):
    """僅載入需要的 llava 檔案，不觸發 llava/__init__.py。"""
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

    # 先載 clip_encoder 再載 builder，保持相對匯入順序
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

    # 構建與 TA 介面一致的 cfg
    cfg = type("Cfg", (), {
        "mm_vision_tower": mm_vision_tower,
        "mm_vision_select_layer": mm_vision_select_layer,
        "mm_projector_type": mm_projector_type,
        "mm_hidden_size": mm_hidden_size,
        "hidden_size": hidden_size
    })()

    vision_tower = enc_builder.build_vision_tower(cfg, delay_load=False).eval()
    projector = proj_builder.build_vision_projector(cfg).train()  # ✅ projector 這次要訓練
    image_processor = vision_tower.image_processor
    return vision_tower, projector, image_processor


# ------------------------- Dataset 與 Collate -------------------------
class CocoCapDataset(Dataset):
    """
    直接讀圖（images/{split}/xxxx.jpg），不使用預存 .pt。
    JSON：{"annotations":[{"image_id": int, "caption": str}, ...]}
    這版加入固定英文 prompt（避免多語＆統一風格）
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

        # 文字：prompt + caption + EOS
        caption = ann["caption"].strip()
        full_text = self.prompt + caption + "<|im_end|>"

        # prompt_len（token數）用於遮蔽 loss
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

        # 右移一格作 teacher forcing：input=[:-1], target=[1:]
        # 我們先生成 labels（與 ids 等長），稍後在主程式切片
        lab = ids.clone()

        # 把 prompt 區段的 target 全部遮蔽（loss 不計）
        # target 是 ids[1:], 所以 prompt 對應 target 位置是 [max(0, p_len-1)] 以前
        if p_len > 0:
            lab[:p_len] = -100

        # PAD 也遮蔽
        lab = torch.where(ids == pad_id, torch.full_like(ids, -100), lab)

        padded.append(ids)
        labels.append(lab)

    return list(images), torch.stack(padded, 0), torch.stack(labels, 0), list(img_ids), list(prompt_lens)


# ------------------------- 工具 -------------------------
def count_trainable_params(models: List[nn.Module]) -> int:
    total_trainable = 0
    for m in models:
        total_trainable += sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"🧮 Trainable parameters total: {total_trainable/1e6:.2f}M")
    return total_trainable


# ------------------------- 訓練主程式 -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="包含 train/val json 與 images/train, images/val 的根目錄")
    parser.add_argument("--baseline_weight", type=str, required=True,
                        help="TA 提供的 decoder_model.bin")
    parser.add_argument("--output_lora", type=str, required=True,
                        help="輸出 LoRA 權重 .pth（僅 LoRA）")
    parser.add_argument("--output_projector", type=str, required=True,
                        help="輸出 projector 權重 .pth（state_dict）")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    # 視覺設定（與推論一致）
    parser.add_argument("--mm_vision_tower", type=str, default="openai/clip-vit-base-patch16")
    parser.add_argument("--mm_vision_select_layer", type=int, default=-2)
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")  # ✅ 最終設定
    parser.add_argument("--mm_hidden_size", type=int, default=768)

    # 訓練 prompt（固定英文，避免多語）
    parser.add_argument("--prompt", type=str, default="Describe the image: ")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ device = {device}")

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
    vision_tower.to(device).eval()   # 凍結
    projector.to(device).train()     # ✅ 這次要訓練

    # 2) Decoder + baseline + 僅開 LoRA（Q/K/V/O r=16, alpha=32, dropout=0.05 已在 decoder.py 中）
    dec = Decoder(cfg).to(device)
    dec.load_state_dict(torch.load(args.baseline_weight, map_location="cpu"), strict=False)
    lora.mark_only_lora_as_trainable(dec, bias="none")  # 只開 LoRA 權重

    # — 統計總訓練參數（LoRA + projector）
    total_trainable = count_trainable_params([dec, projector])
    if total_trainable > 10_000_000:
        raise SystemExit(f"❌ Trainable parameters exceed 10M ({total_trainable}). 請降低 LoRA rank 或 projector 深度。")

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
    # 只優化 projector + LoRA 參數
    optim = AdamW(
        list(projector.parameters()) + [p for p in dec.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd
    )
    scaler = GradScaler(device="cuda", enabled=not args.bf16)  # bf16 不用 scaler
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    use_bf16 = bool(args.bf16 and torch.cuda.is_available())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"🧮 compute dtype: {'bf16' if use_bf16 else 'fp32'}")

    dec.train()
    projector.train()
    vis_prefix_len = 1  # 使用 mean-pool 後 1 個視覺前綴 token

    for ep in range(1, args.epochs + 1):
        running = 0.0
        for it, (images, ids_full, labels_full, _, prompt_lens) in enumerate(train_loader, 1):
            # ------- 圖片 embedding（不需要梯度）-------
            with torch.no_grad():
                pixel = image_processor(images=list(images), return_tensors="pt")["pixel_values"].to(device)
                feats = vision_tower(pixel)            # (B, T_patch, mm_hidden) 來自指定層
                feats = feats.mean(dim=1)              # (B, mm_hidden) → mean-pool 成單向量
            # projector 需訓練 → 開梯度
            vis_emb = projector(feats).unsqueeze(1).to(compute_dtype)  # (B,1,H)

            # ------- 文字（teacher forcing, 遮蔽 prompt 區段）-------
            # ids_full: [prompt + caption + EOS + PAD...]
            input_ids = ids_full[:, :-1].to(device)    # 模型輸入
            target_ids = labels_full[:, 1:].to(device) # 模型目標（已在 collate 遮蔽 prompt/PAD）

            with torch.no_grad():
                txt_emb = dec.embed_tokens(input_ids).to(compute_dtype)  # (B, L-1, H)

            # ------- 視覺前綴 + 文字 embedding -------
            inputs_embeds = torch.cat([vis_emb, txt_emb], dim=1)  # (B, 1 + L-1, H)

            # ------- 前向與 loss（只對應 target_ids 設定位置）-------
            with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = dec(inputs_embeds=inputs_embeds)           # (B, 1+L-1, V)
                # 去掉前綴 1 個位置，對齊 target_ids（尺寸 (B, L-1)）
                logits_text = logits[:, vis_prefix_len:, :]         # (B, L-1, V)
                loss = loss_fn(logits_text.reshape(-1, logits_text.size(-1)),
                               target_ids.reshape(-1))

            # ------- 反向與更新 -------
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

        # ------- 每個 epoch 保存 -------
        os.makedirs(os.path.dirname(args.output_lora), exist_ok=True)
        os.makedirs(os.path.dirname(args.output_projector), exist_ok=True)

        # 只存 LoRA 權重（給 decoder 用）
        torch.save(lora.lora_state_dict(dec, bias="none"), args.output_lora)
        # 存 projector 全量權重（state_dict）
        torch.save(projector.state_dict(), args.output_projector)

        print(f"💾 Saved LoRA to: {args.output_lora}")
        print(f"💾 Saved projector to: {args.output_projector}")

    print("✅ Training finished. Projector + LoRA weights are ready for inference.")


if __name__ == "__main__":
    main()
