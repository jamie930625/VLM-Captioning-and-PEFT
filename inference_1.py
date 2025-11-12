import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image

from llava.model.builder import load_pretrained_model
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise

# ===============================================================
# ✅ VCD Inference Script for DLCV HW3 (Problem 1-2)
# ===============================================================

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_file", type=str, help="path to val.json")
    parser.add_argument("images_root", type=str, help="path to images/val folder")
    parser.add_argument("llava_weight", type=str, help="path to llava-v1.5-7b checkpoint")
    parser.add_argument("pred_file", type=str, help="output json path (pred_vcd.json)")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1️⃣ Load pretrained model (LLaVA-v1.5-7b)
    # ---------------------------------------------------------------
# ---------------------------------------------------------------
# 1️⃣ Load pretrained model (LLaVA-v1.5-7b)
# ---------------------------------------------------------------
    print("🧠 Loading pretrained LLaVA model ...")
    llava_weight = args.llava_weight  # 保持名稱一致
    tokenizer, model, image_processor, _ = load_pretrained_model(
        llava_weight,
        model_base=None,
        model_name="llava-v1.5-7b",
        device_map="auto",
        device="cuda"
    )
    model.eval()
    evolve_vcd_sampling()  # enable Visual Contrastive Decoding
    print("✅ Model ready for VCD inference")

    # ---------------------------------------------------------------
    # 2️⃣ Load dataset annotations
    # ---------------------------------------------------------------
    with open(args.annotation_file, "r") as f:
        annotations = json.load(f)

    results = []

    # ---------------------------------------------------------------
    # 3️⃣ Loop over each image-question pair
    # ---------------------------------------------------------------
    for ann in tqdm(annotations, desc="Running VCD inference"):
        image_name = ann["image_source"] + ".jpg"
        image_path = os.path.join(args.images_root, image_name)
        question = ann["question"]

        # ---- Load & preprocess image ----
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].half().cuda()

        # ---- Generate noisy version (v′) ----
        noise_step = 700  # moderately strong noise
        image_tensor_cd = add_diffusion_noise(image_tensor, noise_step).half().cuda()

        # ---- Prepare text input ----
        prompt = f"You are a helpful visual assistant. Answer with a single word: Yes or No.\nQuestion: {question}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        # -----------------------------------------------------------
        # 4️⃣ Run generation with Visual Contrastive Decoding
        # -----------------------------------------------------------
        outputs = model.generate(
            input_ids=input_ids,
            images=image_tensor.unsqueeze(0),
            images_cd=image_tensor_cd.unsqueeze(0),
            cd_alpha=0.4,
            cd_beta=0.2,
            do_sample=True,
            max_new_tokens=3,
            temperature=1.0,
            return_dict_in_generate=False,
            output_scores=False,
            output_hidden_states=False,
            output_attentions=False,
        )

        # ---- Decode and normalize answer ----
        ans_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ans_text = ans_text.strip().split("\n")[-1]
        if "yes" in ans_text.lower():
            ans = "Yes"
        elif "no" in ans_text.lower():
            ans = "No"
        else:
            # fallback rule
            ans = "Yes" if "y" in ans_text.lower() else "No"

        results.append({
            "image_source": ann["image_source"],
            "question": question,
            "predict": ans
        })

        # ---- Free memory between samples ----
        del outputs, image_tensor, image_tensor_cd, input_ids
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # ---------------------------------------------------------------
    # 5️⃣ Write predictions
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.pred_file), exist_ok=True)
    with open(args.pred_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"🎯 Done! Results saved to {args.pred_file}")
    print(f"✅ Total samples: {len(results)}")


if __name__ == "__main__":
    main()
