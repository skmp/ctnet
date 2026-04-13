# Train CTNet-50
python train_imagenet.py ./imagenette2-320 \
    --arch resnet50 --epochs 30 --pretrained \
    --lambda-rate 1e-4 --qstep 0.1

# Export to H.265
python export_h265.py encode \
    --arch resnet50 --qstep 0.1 \
    --crf 0 --bit-depth 8 --dither 0.1 --preset slower

# Decode and evaluate
python export_h265.py decode \
    --h265-dir ./h265_out --data ./imagenette2-320 \
    --non-dct-weights ./checkpoints/best.pth