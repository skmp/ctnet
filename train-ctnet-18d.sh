#!/bin/bash
set -e

# CTNet-18d: Two-phase, accuracy-focused with 8-bit quantization awareness
# Phase 1: Task learning with gentle compression (128 epochs)
# Phase 2: 8-bit hardening — refine accuracy under pixel quantization (128 epochs)
# Usage: ./train-ctnet-18d.sh [data_dir]

DATA_DIR="${1:-./imagenette2-320}"

echo "=== CTNet-18d (two-phase, 8-bit aware) ==="
echo "Dataset: $DATA_DIR"
echo ""

# Phase 1: Task + gentle compression, no pixel quant sim
# - Lower LR preserves pretrained features
# - Rate warmup lets the model settle before compression
# - No noise/dropout for cleaner feature learning
# - pixel-bit-depth 0: no quantization simulation yet
echo "--- Phase 1: Task + gentle compression (128 epochs) ---"
python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 128 --pretrained \
    --optimizer adamw --lr 5e-4 --weight-decay 0.001 \
    --lambda-rate 1e-5 --qstep 0.1 \
    --lambda-alpha 0.5 \
    --no-train-noise --dct-dropout 0 \
    --rate-warmup-epochs 4 \
    --pixel-bit-depth 0 \
    --cache-dataset

# Save phase 1 checkpoints
cp checkpoints/best.pth checkpoints/best-phase1.pth 2>/dev/null || true
cp checkpoints/smallest.pth checkpoints/smallest-phase1.pth 2>/dev/null || true

# Phase 2: Continue with 8-bit pixel quantization simulation
# - Auto-resumes from checkpoint at epoch 128
# - pixel-bit-depth 8: straight-through estimator for 8-bit robustness
# - Lower lambda-rate: model already compressed, refine for accuracy at current size
echo ""
echo "--- Phase 2: 8-bit hardening (128 more epochs) ---"
python train_imagenet.py "$DATA_DIR" \
    --arch resnet18 --epochs 256 --pretrained \
    --optimizer adamw --lr 5e-4 --weight-decay 0.001 \
    --lambda-rate 5e-6 --qstep 0.1 \
    --lambda-alpha 0.5 \
    --no-train-noise --dct-dropout 0 \
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
