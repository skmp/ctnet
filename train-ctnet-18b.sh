#!/bin/bash
set -e

# CTNet-18: Train, export, and evaluate
# AdamW optimizer, 256 epochs, balanced rate pressure
# Usage: ./train-ctnet-18.sh [data_dir]

DATA_DIR="${1:-./imagenette2-320}"

echo "=== CTNet-18 (AdamW, 256 epochs) ==="
echo "Dataset: $DATA_DIR"
echo ""

# Train
echo "--- Training ---"
python train_imagenet.py "$DATA_DIR" --arch resnet18 --epochs 256 --pretrained --cache-dataset

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
