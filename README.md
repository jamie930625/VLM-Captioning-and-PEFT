# Vision-Language Models: Hallucination Mitigation & PEFT

This repository explores the deployment, evaluation, and efficient fine-tuning of Large Vision-Language Models (VLMs). It consists of two main parts: mitigating object hallucinations in zero-shot settings and implementing Parameter-Efficient Fine-Tuning (PEFT) from scratch for image captioning.

## Part 1: Mitigating Object Hallucinations in LLaVA
Large Multi-modal Models (MLLMs) often suffer from object hallucination, generating text about non-existent objects due to statistical bias and language priors. 

In this section, I evaluated a pre-trained **LLaVA** model on the POPE dataset and implemented **Visual Contrastive Decoding (VCD)** to mitigate these hallucinations. 
- **Mechanism**: VCD contrasts output distributions derived from original and distorted visual inputs. 
- **Result**: By penalizing predictions that persist even when visual evidence is corrupted, the model effectively filters out hallucinated tokens driven merely by language priors, leading to more accurate and visually grounded responses.

## Part 2: Image Captioning with Custom LoRA (Low-Rank Adaptation)
The second part focuses on building and fine-tuning an image captioning pipeline using a Vision Transformer (ViT) encoder and a Qwen3-based causal language decoder.

To adapt the massive language decoder to the captioning task without incurring prohibitive computational costs, I implemented **LoRA (Low-Rank Adaptation) entirely from scratch** (without relying on high-level PEFT libraries like HuggingFace `peft`). 

### Key Highlights:
- **Custom LoRA Implementation**: Integrated trainable low-rank matrices into the self-attention layers of the decoder, keeping the original pretrained weights frozen.
- **Parameter Efficiency**: Successfully restricted the total trainable parameters to under 10M while maintaining robust generation capabilities.
- **Evaluation**: The model's generated captions were rigorously evaluated using standard natural language generation metrics, specifically **CIDEr** and **CLIPScore**, achieving highly competitive baseline scores.

## Environment Setup
To reproduce the environment and run the inference scripts:
```bash
conda create -n vlm_env python=3.10
conda activate vlm_env
pip install torch==2.0.1 torchvision==0.15.2 transformers==4.31.0 timm==0.6.13
# Check the problem-specific requirements for additional dependencies (e.g., bitsandbytes, pycocotools)
