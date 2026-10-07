#!/bin/bash
# ==========================================================
# DLCV 2025 HW3 - Problem 2 Inference Script
# Usage (run by the grader):
#   bash hw3_2.sh $1 $2 $3
#   $1 : path to the folder containing test images (e.g. hw3/p2_data/images/val/)
#   $2 : path to the output json file           (e.g. hw3/output_p2/pred.json)
#   $3 : path to the decoder weights            (e.g. hw3/p2_data/decoder_model.bin)
#
# No absolute paths: assumes inference.py is in the same folder as this script.
# ==========================================================

set -e

IMG_DIR="$1"
OUT_JSON="$2"
DECODER_WEIGHT="$3"

# move to the folder of this script (repo root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Images folder : ${IMG_DIR}"
echo "Output json   : ${OUT_JSON}"
echo "Decoder weight: ${DECODER_WEIGHT}"

# run inference.py (loads LoRA and projector weights from relative paths)
python3 inference.py "${IMG_DIR}" "${OUT_JSON}" "${DECODER_WEIGHT}"
