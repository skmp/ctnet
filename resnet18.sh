#!/bin/bash
set -e

# CTNet-18: Train, export, and evaluate
# Usage: ./resnet18.sh [data_dir]

DATA_DIR="${1:-./imagenette2-320}"

echo "=== CTNet-18 ==="
echo "Dataset: $DATA_DIR"
echo ""

# Train
echo "--- Training ---"
python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 30 --pretrained \
    --lambda-rate 1e-4 --qstep 0.1

# Export to H.265
echo ""
echo "--- Encoding to H.265 ---"
python export_h265.py encode \
    --arch resnet18 --qstep 0.1 \
    --crf 0 --bit-depth 8 --dither 0.1 --preset slower

# Decode and evaluate
echo ""
echo "--- Decoding and Evaluating ---"
python export_h265.py decode \
    --h265-dir ./h265_out --data "$DATA_DIR" \
    --non-dct-weights ./checkpoints/best.pth
