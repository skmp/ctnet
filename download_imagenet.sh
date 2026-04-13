#!/bin/bash
set -e

# Download the full ImageNet (ILSVRC2012) dataset.
#
# ImageNet requires manual registration and agreement to terms of use.
# This script downloads from the official source if you have credentials,
# or provides instructions for manual download.
#
# Usage: ./download_imagenet.sh [destination_dir]
#
# After download, the directory structure will be:
#   destination_dir/
#     train/
#       n01440764/
#         *.JPEG
#       ...
#     val/
#       n01440764/
#         *.JPEG
#       ...

DEST="${1:-./imagenet}"

if [ -d "$DEST/train" ] && [ -d "$DEST/val" ]; then
    TRAIN_CLASSES=$(ls -1 "$DEST/train" | wc -l)
    VAL_CLASSES=$(ls -1 "$DEST/val" | wc -l)
    echo "ImageNet already exists at $DEST"
    echo "  train: $TRAIN_CLASSES classes"
    echo "  val:   $VAL_CLASSES classes"
    exit 0
fi

mkdir -p "$DEST"

echo "=== ImageNet (ILSVRC2012) Download ==="
echo ""
echo "ImageNet requires registration at https://image-net.org/"
echo "You need to agree to the terms of use before downloading."
echo ""

# Check if torchvision can download it (requires credentials in env)
if command -v python3 &>/dev/null; then
    echo "Attempting download via torchvision..."
    python3 -c "
import torchvision.datasets as datasets
import sys

dest = sys.argv[1]

print('Downloading ImageNet training set...')
try:
    datasets.ImageNet(dest, split='train', download=True)
    print('Training set downloaded.')
except Exception as e:
    print(f'Auto-download failed: {e}')
    print()
    print('Manual download instructions:')
    print('  1. Go to https://image-net.org/download-images.php')
    print('  2. Download ILSVRC2012_img_train.tar (~138 GB)')
    print('  3. Download ILSVRC2012_img_val.tar (~6.3 GB)')
    print('  4. Extract them:')
    print(f'     mkdir -p {dest}/train {dest}/val')
    print(f'     tar -xf ILSVRC2012_img_train.tar -C {dest}/train')
    print(f'     tar -xf ILSVRC2012_img_val.tar -C {dest}/val')
    print('  5. For training set, extract each class tar:')
    print(f'     cd {dest}/train && for f in *.tar; do d=${{f%.tar}}; mkdir -p $d; tar -xf $f -C $d; rm $f; done')
    print('  6. For validation set, use the official devkit to sort into class folders,')
    print('     or use: https://raw.githubusercontent.com/soumith/imagenetloader/master/valprep.sh')
    sys.exit(1)
" "$DEST"
else
    echo "Python3 not found. Manual download instructions:"
    echo ""
    echo "  1. Go to https://image-net.org/download-images.php"
    echo "  2. Download ILSVRC2012_img_train.tar (~138 GB)"
    echo "  3. Download ILSVRC2012_img_val.tar (~6.3 GB)"
    echo "  4. Extract:"
    echo "     mkdir -p $DEST/train $DEST/val"
    echo "     tar -xf ILSVRC2012_img_train.tar -C $DEST/train"
    echo "     tar -xf ILSVRC2012_img_val.tar -C $DEST/val"
    echo "  5. Extract class tars in training set:"
    echo "     cd $DEST/train && for f in *.tar; do d=\${f%.tar}; mkdir -p \$d; tar -xf \$f -C \$d; rm \$f; done"
    echo "  6. Sort validation images into class folders using:"
    echo "     https://raw.githubusercontent.com/soumith/imagenetloader/master/valprep.sh"
    exit 1
fi

# Verify
if [ -d "$DEST/train" ] && [ -d "$DEST/val" ]; then
    TRAIN_CLASSES=$(ls -1 "$DEST/train" | wc -l)
    VAL_CLASSES=$(ls -1 "$DEST/val" | wc -l)
    echo ""
    echo "ImageNet downloaded to $DEST"
    echo "  train: $TRAIN_CLASSES classes"
    echo "  val:   $VAL_CLASSES classes"
else
    echo "Download incomplete. Please follow manual instructions above."
    exit 1
fi
