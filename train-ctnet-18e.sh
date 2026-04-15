#!/bin/bash
set -e

# CTNet-18e: Single-phase, 8-bit quantization aware from start
# Usage: ./train-ctnet-18e.sh [data_dir]

DATA_DIR="${1:-./imagenette2-320}"

echo "=== CTNet-18e (8-bit aware from start) ==="
echo "Dataset: $DATA_DIR"
echo ""

python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 512 --pretrained \
    --optimizer adamw --lr 5e-4 --weight-decay 0.001 \
    --lambda-rate 1e-5 --qstep 0.1 \
    --lambda-alpha 0.5 \
    --no-train-noise --dct-dropout 0 \
    --rate-warmup-epochs 4 \
    --pixel-bit-depth 8 \
    --cache-dataset

# Export at 8-bit
echo ""
echo "--- Encoding to H.265 (8-bit) ---"
python export_h265.py encode --arch resnet18 \
    --crf 0 --bit-depth 8 --preset slower

# Decode and evaluate
echo ""
echo "--- Decoding and Evaluating ---"
python export_h265.py decode --h265-dir ./h265_out --data "$DATA_DIR"
