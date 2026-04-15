#!/bin/bash
set -e

# CTGPT: Train DCT-compressed GPT-2 on Shakespeare
# Usage: ./train-ctgpt.sh

echo "=== CTGPT: DCT-compressed GPT-2 on Shakespeare ==="
echo ""

# Train
python ctgpt_train.py \
    --epochs 50 --batch-size 8 --seq-len 256 \
    --lr 5e-4 --weight-decay 0.001 \
    --lambda-rate 1e-5 --qstep 0.1 \
    --lambda-alpha 0.5 \
    --no-train-noise --dct-dropout 0 \
    --rate-warmup-epochs 4 \
    --pixel-bit-depth 8 \
    --dct-block-size 16

# Export
echo ""
echo "--- Encoding to H.265 ---"
python ctgpt_export.py encode \
    --model ./ctgpt_checkpoints/best.pt \
    --crf 0 --bit-depth 8 --preset slower

# Decode and generate
echo ""
echo "--- Decoding and generating ---"
python ctgpt_export.py decode \
    --h265-dir ./ctgpt_h265 \
    --prompt "ROMEO:" --max-tokens 500
