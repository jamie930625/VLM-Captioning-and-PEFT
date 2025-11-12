#!/bin/bash
# ==========================================================
# DLCV 2025 HW3 - Problem 2 Inference Script
# 使用方式（由助教呼叫）：
#   bash hw3_2.sh $1 $2 $3
#   $1 : path to the folder containing test images (e.g. hw3/p2_data/images/val/)
#   $2 : path to the output json file           (e.g. hw3/output_p2/pred.json)
#   $3 : path to the decoder weights            (e.g. hw3/p2_data/decoder_model.bin)
#
# 注意：本腳本不硬編任何絕對路徑，只假設 inference.py 與本檔案位於同一個 repo 目錄。
# ==========================================================

set -e

IMG_DIR="$1"
OUT_JSON="$2"
DECODER_WEIGHT="$3"

# 移動到此 script 所在的資料夾（也就是 repo root）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Images folder : ${IMG_DIR}"
echo "Output json   : ${OUT_JSON}"
echo "Decoder weight: ${DECODER_WEIGHT}"

# 呼叫 inference.py（其內部會讀取相對路徑的 LoRA & projector）
python3 inference.py "${IMG_DIR}" "${OUT_JSON}" "${DECODER_WEIGHT}"
