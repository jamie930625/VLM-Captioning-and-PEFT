# Vision-Language Models: Hallucination Mitigation and LoRA Captioning

Course project for NTU Deep Learning for Computer Vision (Fall 2025), HW3. The full report is in `hw3_b11103049.pdf`.

## Part 1: Reducing object hallucination in LLaVA with Visual Contrastive Decoding

Vision-language models often describe objects that are not in the image, because the language model's priors override the visual evidence.

I implemented **Visual Contrastive Decoding (VCD)** on a pre-trained **LLaVA** model and evaluated it on POPE. At each decoding step, VCD runs the model on the original image and on a noise-perturbed copy, and penalizes tokens that stay likely even when the visual evidence is corrupted. It needs no extra training.

| Decoding | POPE accuracy |
|---|---|
| LLaVA, standard decoding | 0.52 |
| LLaVA + VCD | **0.56** |

## Part 2: Image captioning with a LoRA-adapted language decoder

- **Vision encoder:** frozen CLIP ViT-B/16; patch features are mean-pooled into one visual prefix token.
- **Projector:** a 2-layer MLP (768 → 1024 → 1024) trained from scratch.
- **Decoder:** a 0.6B Qwen3-style decoder with LoRA (via `loralib`) on the attention Q/K/V/O projections (r = 16, alpha = 32, dropout = 0.05). The original weights stay frozen.
- **Training:** a fixed English prompt ("Describe the image: "); the loss is computed on caption tokens only. Projector and LoRA together have fewer than 10M trainable parameters.
- **Decoding:** nucleus sampling (temperature 0.7, top-p 0.95) with a repetition penalty of 1.1.

| Setting | CIDEr | CLIPScore |
|---|---|---|
| Best (trained projector + LoRA + English prompt) | **0.86** | **0.70** |

The report also compares two alternative LoRA settings, such as training LoRA with a frozen projector and no prompt.

## Files

```text
inference_1.py, vcd_utils/   # Part 1: VCD inference on LLaVA
decoder.py                   # Part 2: Qwen3-style decoder with LoRA and sampling
train_decoder.py             # Part 2: training the projector and LoRA
inference.py                 # Part 2: caption generation
output_p2/                   # trained projector and LoRA weights
hw3_1.sh, hw3_2.sh           # entry points
```

## Environment

```bash
conda create -n vlm_env python=3.10
conda activate vlm_env
pip install -r requirements_p2.txt
```
