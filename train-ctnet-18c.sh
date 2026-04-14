#!/bin/bash
set -e

# CTNet-18c: Two-phase accuracy-focused training
# Phase 1: Pure task learning (50 epochs, no compression)
# Phase 2: Task + rate compression with fade-in (462 more epochs)
# Usage: ./train-ctnet-18c.sh [data_dir]

DATA_DIR="${1:-./imagenette2-320}"

echo "=== CTNet-18c (two-phase, accuracy-focused) ==="
echo "Dataset: $DATA_DIR"
echo ""

# Phase 1: Task-only (50 epochs, no rate compression)
echo "--- Phase 1: Task learning (50 epochs, no compression) ---"
python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 50 --pretrained \
    --optimizer adamw --lr 5e-4 --weight-decay 0.001 \
    --lambda-rate 0 --qstep 0.5 \
    --lambda-alpha 0 \
    --cache-dataset

# Phase 2: Task + rate compression (462 more epochs, auto-resumes from checkpoint)
echo ""
echo "--- Phase 2: Compression (462 epochs, rate fades in) ---"
python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 512 --pretrained \
    --optimizer adamw --lr 5e-4 --weight-decay 0.001 \
    --lambda-rate 1e-6 --qstep 0.5 \
    --lambda-alpha 0.5 \
    --rate-warmup-epochs 55 \
    --cache-dataset

# Export
echo ""
echo "--- Encoding to H.265 ---"
python export_h265.py encode --arch resnet18 \
    --crf 0 --bit-depth 8 --preset slower

# Decode and evaluate
echo ""
echo "--- Decoding and Evaluating ---"
python export_h265.py decode --h265-dir ./h265_out --data "$DATA_DIR"
