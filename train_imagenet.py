"""
ImageNet training with DCT-domain convolutions and Group Lasso sparsity.

Usage (single GPU):
    python train_imagenet.py /path/to/imagenet

Usage (multi-GPU, 4 GPUs):
    torchrun --nproc_per_node=4 train_imagenet.py /path/to/imagenet

ImageNet directory layout expected:
    /path/to/imagenet/
        train/
            n01440764/
                *.JPEG
            ...
        val/
            n01440764/
                *.JPEG
            ...
"""

import argparse
import os
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dct_layers import DCTConv2d, ChannelDCTConv1x1, _is_dct_layer, replace_with_dct_convs, probe_sparsity, quantize_model, export_sparse_coefficients, dct_config
from dct_utils import calculate_hevc_rate_proxy, estimate_h265_size_bits


class CachedDataset(torch.utils.data.Dataset):
    """
    Cache dataset as pre-decoded tensors in RAM.

    For training: caches resized tensors (Resize+ToTensor), applies random
    augmentations (RandomResizedCrop, RandomHorizontalFlip) on-the-fly via GPU
    or fast tensor ops.

    For validation: caches fully transformed tensors (Resize+CenterCrop+
    ToTensor+Normalize) — __getitem__ is a pure lookup.
    """

    def __init__(self, image_folder, transform, is_train=False):
        self.targets = image_folder.targets
        self._is_train = is_train

        if is_train:
            # Cache as tensors at original size, apply augmentation on-the-fly
            self._normalize = None
            self._images = []
            to_tensor = transforms.ToTensor()
            for path, target in image_folder.imgs:
                img = image_folder.loader(path)
                self._images.append((to_tensor(img), target))
            # Extract normalize from the transform pipeline
            for t in transform.transforms:
                if isinstance(t, transforms.Normalize):
                    self._normalize = t
        else:
            # Cache fully transformed tensors — zero-cost __getitem__
            self._images = []
            for i in range(len(image_folder)):
                img, target = image_folder[i]
                self._images.append((img, target))

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img, target = self._images[idx]
        if self._is_train:
            # Fast tensor augmentation
            img = transforms.functional.resized_crop(
                img,
                *transforms.RandomResizedCrop.get_params(img, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
                size=(224, 224),
            )
            if torch.rand(1).item() < 0.5:
                img = transforms.functional.hflip(img)
            if self._normalize is not None:
                img = self._normalize(img)
        return img, target


def parse_args():
    parser = argparse.ArgumentParser(description="ImageNet DCT-Conv Training")
    parser.add_argument("data", metavar="DIR", help="path to ImageNet dataset")
    parser.add_argument(
        "--arch", default="resnet50", choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        help="model architecture (default: resnet50)",
    )
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument("-b", "--batch-size", default=256, type=int,
                        help="total batch size across all GPUs")
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"],
                        help="optimizer (default: sgd)")
    parser.add_argument("--lr", default=None, type=float,
                        help="initial learning rate (default: 0.1 for sgd, 1e-3 for adamw)")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=None, type=float,
                        help="weight decay (default: 1e-4 for sgd, 0.01 for adamw)")
    parser.add_argument("--lambda-rate", default=1e-4, type=float,
                        help="weight for HEVC rate proxy loss (rate-distortion tradeoff)")
    parser.add_argument("--lambda-l2", default=0.0, type=float,
                        help="L2 regularization on DCT coefficients (0 = off, or use --lambda-alpha)")
    parser.add_argument("--lambda-alpha", default=1.0, type=float,
                        help="coupled lambda: lambda_l2 = alpha * lambda_rate (0 = use --lambda-l2 directly)")
    parser.add_argument("--qstep", default=0.1, type=float,
                        help="quantization step size for HEVC rate proxy (larger = more sparsity)")
    parser.add_argument("--steepness", default=10.0, type=float,
                        help="sigmoid steepness for soft significance threshold")
    parser.add_argument("--no-train-noise", action="store_true",
                        help="disable training-time uniform noise (enabled by default)")
    parser.add_argument("--dct-dropout", default=0.05, type=float,
                        help="DCT coefficient dropout probability (default: 0.05, 0 = off)")
    parser.add_argument("--dct-block-size", default=16, type=int,
                        help="block size for channel-wise DCT (default: 16, 0 = full DCT)")
    parser.add_argument("--pixel-bit-depth", default=8, type=int, choices=[0, 8, 10, 12],
                        help="simulate N-bit pixel quantization during training (default: 8, 0 = off)")
    parser.add_argument("--rate-warmup-epochs", default=5, type=int,
                        help="ramp rate loss from 0 to full over N epochs (default: 5, 0 = off)")
    parser.add_argument("-j", "--workers", default=4, type=int, help="data loading workers")
    parser.add_argument("--cache-dataset", action="store_true",
                        help="cache entire dataset in RAM (fast for small datasets like ImageNette)")
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="evaluate only")
    parser.add_argument("--pretrained", action="store_true",
                        help="start from torchvision pretrained weights")
    parser.add_argument("--no-imagenet-remap", action="store_true",
                        help="disable automatic label remapping for ImageNet subsets")
    parser.add_argument("--output-dir", default="./checkpoints", type=str)
    parser.add_argument("--clip-grad", default=1.0, type=float,
                        help="max gradient norm (0 to disable)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---------- Distributed setup ----------
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    is_main = rank == 0
    device = torch.device("cuda", local_rank)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"=> Architecture: {args.arch}")
        print(f"=> World size: {world_size}")
        print(f"=> Optimizer: {args.optimizer} (lr={args.lr}, wd={args.weight_decay})")
        print(f"=> Lambda rate: {args.lambda_rate}")
        print(f"=> Quantization step: {args.qstep}")
        print(f"=> Sigmoid steepness: {args.steepness}")

    # ---------- DCT config ----------
    dct_config.qstep = args.qstep
    dct_config.train_noise = not args.no_train_noise
    dct_config.dct_dropout = args.dct_dropout
    dct_config.pixel_bit_depth = args.pixel_bit_depth
    if args.lambda_alpha > 0:
        args.lambda_l2 = args.lambda_alpha * args.lambda_rate

    # ---------- Model ----------
    model_fn = getattr(models, args.arch)
    weights = "DEFAULT" if args.pretrained else None
    model = model_fn(weights=weights)
    replace_with_dct_convs(model, block_size=args.dct_block_size)
    model = model.to(device)

    if is_main:
        n_spatial = sum(1 for m in model.modules() if isinstance(m, DCTConv2d))
        n_channel = sum(1 for m in model.modules() if isinstance(m, ChannelDCTConv1x1))
        n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        print(f"=> Replaced: {n_spatial} DCTConv2d (spatial), {n_channel} ChannelDCTConv1x1 (1x1), {n_conv} standard Conv2d remaining")

    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ---------- Loss / Optimizer / Scheduler ----------
    criterion = nn.CrossEntropyLoss().to(device)

    # Set optimizer-specific defaults
    if args.lr is None:
        args.lr = {"sgd": 0.1, "adamw": 1e-3}[args.optimizer]
    if args.weight_decay is None:
        args.weight_decay = {"sgd": 1e-4, "adamw": 0.01}[args.optimizer]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    # Cosine annealing with warmup (5 epochs linear warmup)
    warmup_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0

    # ---------- Resume ----------
    # Auto-resume from checkpoint.pth if it exists and --resume not specified
    resume_path = args.resume
    if not resume_path:
        auto_resume = os.path.join(args.output_dir, "checkpoint.pth")
        if os.path.isfile(auto_resume):
            resume_path = auto_resume
            if is_main:
                print(f"=> Found existing checkpoint at {auto_resume}, resuming...")

    if resume_path:
        if os.path.isfile(resume_path):
            if is_main:
                print(f"=> Loading checkpoint '{resume_path}'")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            start_epoch = ckpt["epoch"]
            model_state = ckpt["state_dict"]
            # Handle DDP state_dict prefix
            if not distributed and any(k.startswith("module.") for k in model_state):
                model_state = {k.removeprefix("module."): v for k, v in model_state.items()}
            model.load_state_dict(model_state)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if is_main:
                print(f"=> Resumed from epoch {start_epoch}")
        else:
            raise FileNotFoundError(f"No checkpoint at '{resume_path}'")

    # ---------- Data ----------
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    # Label remapping for subset datasets (e.g. ImageNette → ImageNet indices)
    # The 1000-class FC head expects ImageNet class indices, not sequential 0..N-1.
    # Enabled by default; disable with --no-imagenet-remap.
    num_classes = len(train_dataset.classes)
    label_map = None
    if not args.no_imagenet_remap and num_classes < 1000:
        # The dataset folder names should be WordNet IDs
        subset_wnids = sorted(train_dataset.class_to_idx.keys())
        if all(w.startswith("n") and len(w) == 9 for w in subset_wnids):
            # Build full ImageNet wnid list from torchvision
            # torchvision models expect alphabetically sorted wnid order
            try:
                from torchvision.models import ResNet18_Weights
                meta = ResNet18_Weights.DEFAULT.meta
                # meta["categories"] is in ImageNet class order
                # We need wnid ordering — get from the dataset class_to_idx
                # which ImageFolder creates from sorted folder names.
                # The ImageNet-pretrained model uses the same sorted order.
                # For a subset: the local index is the position in the subset's
                # sorted folder list, but the model expects the position in the
                # full 1000-class sorted list.

                # Since we can't easily get the full 1000-wnid list from
                # torchvision, use a known mapping for common subsets.
                pass
            except ImportError:
                pass

            # Hardcoded ImageNette wnid → ImageNet index mapping
            # (verified empirically against pretrained ResNet-18)
            IMAGENET_WNID_TO_IDX = {
                "n01440764": 0,    # tench
                "n02102040": 217,  # English springer
                "n02979186": 482,  # cassette player
                "n03000684": 491,  # chain saw
                "n03028079": 497,  # church
                "n03394916": 566,  # French horn
                "n03417042": 569,  # garbage truck
                "n03425413": 571,  # gas pump
                "n03445777": 574,  # golf ball
                "n03888257": 701,  # parachute
            }

            local_to_imagenet = {}
            for wnid, local_idx in train_dataset.class_to_idx.items():
                if wnid in IMAGENET_WNID_TO_IDX:
                    local_to_imagenet[local_idx] = IMAGENET_WNID_TO_IDX[wnid]

            if len(local_to_imagenet) == num_classes:
                label_map = local_to_imagenet
                if is_main:
                    print(f"=> Subset dataset ({num_classes} classes), "
                          f"remapping labels to ImageNet indices")

                for ds in [train_dataset, val_dataset]:
                    if hasattr(ds, "targets"):
                        ds.targets = [label_map[t] for t in ds.targets]
                    if hasattr(ds, "samples"):
                        ds.samples = [(p, label_map[t]) for p, t in ds.samples]
                    if hasattr(ds, "imgs"):
                        ds.imgs = ds.samples

    if args.cache_dataset:
        if is_main:
            print("=> Caching dataset in RAM...")
        train_dataset = CachedDataset(train_dataset, train_transform, is_train=True)
        val_dataset = CachedDataset(val_dataset, val_transform, is_train=False)
        args.workers = 0  # no multiprocessing needed, data is in RAM
        if is_main:
            print(f"=> Cached {len(train_dataset)} train + {len(val_dataset)} val images in RAM")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size // world_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size // world_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    # ---------- Evaluate-only ----------
    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    # ---------- Training loop ----------
    best_acc1 = 0.0
    smallest_est_kb = float("inf")
    # Track best accuracy at each integer compression ratio (5x, 6x, ...)
    best_acc_at_ratio: dict[int, float] = {}

    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args)
        scheduler.step()

        acc1 = validate(val_loader, model, criterion, device, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_main:
            base_model = model.module if hasattr(model, "module") else model
            total, nonzero, sparsity = probe_sparsity(base_model, qstep=args.qstep)

            state = {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            path = os.path.join(args.output_dir, "checkpoint.pth")
            torch.save(state, path)
            if is_best:
                torch.save(state, os.path.join(args.output_dir, "best.pth"))

            # Estimate H.265 compressed size at 8/10/12-bit
            est_by_depth, raw_bytes = estimate_h265_size_bits(
                base_model.named_modules(), bit_depths=(8, 10, 12)
            )
            raw_kb = raw_bytes / 1024
            est_8b_kb = est_by_depth[8] / 8 / 1024

            # Save smallest (by estimated 8-bit H.265 size)
            is_smallest = est_8b_kb < smallest_est_kb
            if is_smallest:
                smallest_est_kb = est_8b_kb
                torch.save(state, os.path.join(args.output_dir, "smallest.pth"))

            # Save best model at each integer compression ratio
            ratio_int = int(raw_kb / max(est_8b_kb, 1e-6))
            ratio_tag = ""
            if ratio_int >= 5:
                prev_best = best_acc_at_ratio.get(ratio_int, 0.0)
                if acc1 > prev_best:
                    best_acc_at_ratio[ratio_int] = acc1
                    torch.save(state, os.path.join(args.output_dir, f"best_{ratio_int}x.pth"))
                    ratio_tag = f" *best@{ratio_int}x*"

            est_parts = []
            for b in (8, 10, 12):
                kb = est_by_depth[b] / 8 / 1024
                r = raw_kb / max(kb, 1e-6)
                est_parts.append(f"{b}b:{kb:.0f}KB({int(r)}x)")
            est_str = "  ".join(est_parts)

            best_tag = " *best*" if is_best else ""
            small_tag = " *smallest*" if is_smallest else ""
            print(f"=> Epoch {epoch}: Acc@1 {acc1:.2f}%  (best: {best_acc1:.2f}%)  "
                  f"Sparsity {sparsity*100:.1f}% ({nonzero}/{total} nonzero)  "
                  f"Est[{est_str}] / {raw_kb:.0f}KB raw"
                  f"{best_tag}{small_tag}{ratio_tag}")

    # --- Post-training: quantize and export sparse coefficients ---
    if is_main:
        print("\n=> Quantizing DCT coefficients and exporting sparse model...")

        # Load best model for quantization
        best_path = os.path.join(args.output_dir, "best.pth")
        if os.path.isfile(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            base_model = model.module if hasattr(model, "module") else model
            state = ckpt["state_dict"]
            if any(k.startswith("module.") for k in state):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            base_model.load_state_dict(state)
            print(f"   Loaded best checkpoint (Acc@1 {ckpt['best_acc1']:.2f}%)")

        base_model = model.module if hasattr(model, "module") else model
        stats = quantize_model(base_model, qstep=args.qstep)
        print(f"   Overall sparsity: {stats['overall_sparsity']*100:.1f}% "
              f"({stats['nonzero_coeffs']}/{stats['total_coeffs']} coefficients retained)")
        for ls in stats["layers"]:
            print(f"     {ls['name']}: {ls['sparsity']*100:.1f}% sparse "
                  f"({ls['nonzero_coeffs']}/{ls['total_coeffs']} nonzero)")

        # Evaluate quantized model
        print("\n=> Evaluating quantized model...")
        q_acc1 = validate(val_loader, model, criterion, device, args)
        print(f"   Quantized Acc@1: {q_acc1:.2f}% (was {best_acc1:.2f}% before quantization)")

        # Save quantized model
        q_path = os.path.join(args.output_dir, "quantized.pth")
        torch.save({
            "arch": args.arch,
            "qstep": args.qstep,
            "state_dict": base_model.state_dict(),
            "acc1_before_quant": best_acc1,
            "acc1_after_quant": q_acc1,
            "sparsity_stats": stats,
        }, q_path)
        print(f"   Saved quantized model to {q_path}")

        # Export sparse coefficients
        sparse_data = export_sparse_coefficients(base_model, qstep=args.qstep)
        sparse_path = os.path.join(args.output_dir, "sparse_coefficients.pt")
        torch.save(sparse_data, sparse_path)
        print(f"   Saved sparse coefficients to {sparse_path}")

    if distributed:
        dist.destroy_process_group()


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    task_losses = AverageMeter()
    rate_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    end = time.time()

    base_model = model.module if hasattr(model, "module") else model

    # Raw weight size (float32) for compression ratio
    raw_kb = sum(
        m.weight_dct.numel() * 4 for m in base_model.modules() if _is_dct_layer(m)
    ) / 1024

    # Pre-compute size estimates for this epoch (used in batch logging)
    est_by_depth, _ = estimate_h265_size_bits(
        base_model.named_modules(), bit_depths=(8, 10, 12)
    )

    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward
        outputs = model(images)
        task_loss = criterion(outputs, targets)

        # HEVC rate proxy: differentiable estimate of compressed size
        rate_loss = torch.tensor(0.0, device=device)
        l2_loss = torch.tensor(0.0, device=device)
        for m in base_model.modules():
            if _is_dct_layer(m):
                rate_loss = rate_loss + calculate_hevc_rate_proxy(
                    m.weight_dct, qstep=args.qstep, steepness=args.steepness
                )
                if args.lambda_l2 > 0:
                    l2_loss = l2_loss + m.weight_dct.pow(2).sum()

        # Rate warmup: ramp from 0 to full over rate_warmup_epochs
        if args.rate_warmup_epochs > 0 and epoch < args.rate_warmup_epochs:
            rate_scale = (epoch + 1) / args.rate_warmup_epochs
        else:
            rate_scale = 1.0

        total_loss = (task_loss
                      + args.lambda_rate * rate_scale * rate_loss
                      + args.lambda_l2 * rate_scale * l2_loss)

        # Metrics
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        task_losses.update(task_loss.item(), images.size(0))
        rate_losses.update(rate_loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                est_parts = []
                for b in (8, 10, 12):
                    kb = est_by_depth[b] / 8 / 1024
                    r = raw_kb / max(kb, 1e-6)
                    est_parts.append(f"{b}b:{kb:.0f}KB({int(r)}x)")
                est_str = "  ".join(est_parts)
                print(
                    f"Epoch [{epoch}][{i}/{len(train_loader)}]  "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})  "
                    f"TaskLoss {task_losses.val:.4f} ({task_losses.avg:.4f})  "
                    f"RateLoss {rate_losses.val:.1f} ({rate_losses.avg:.1f})  "
                    f"Est[{est_str}]  "
                    f"Acc@1 {top1.val:.2f} ({top1.avg:.2f})  "
                    f"Acc@5 {top5.val:.2f} ({top5.avg:.2f})"
                )


def validate(val_loader, model, criterion, device, args):
    task_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            task_losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(
            f"  * Val:  Loss {task_losses.avg:.4f}  "
            f"Acc@1 {top1.avg:.2f}  Acc@5 {top5.avg:.2f}"
        )

    return top1.avg


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
