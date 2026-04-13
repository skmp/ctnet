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

from dct_layers import DCTConv2d, replace_with_dct_convs, probe_sparsity, quantize_model, export_sparse_coefficients
from dct_utils import calculate_hevc_rate_proxy


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
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--lambda-rate", default=1e-4, type=float,
                        help="weight for HEVC rate proxy loss (rate-distortion tradeoff)")
    parser.add_argument("--qstep", default=0.1, type=float,
                        help="quantization step size for HEVC rate proxy (larger = more sparsity)")
    parser.add_argument("--steepness", default=10.0, type=float,
                        help="sigmoid steepness for soft significance threshold")
    parser.add_argument("-j", "--workers", default=4, type=int, help="data loading workers")
    parser.add_argument("--print-freq", default=100, type=int)
    parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="evaluate only")
    parser.add_argument("--pretrained", action="store_true",
                        help="start from torchvision pretrained weights")
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
        print(f"=> Lambda rate: {args.lambda_rate}")
        print(f"=> Quantization step: {args.qstep}")
        print(f"=> Sigmoid steepness: {args.steepness}")

    # ---------- Model ----------
    model_fn = getattr(models, args.arch)
    weights = "DEFAULT" if args.pretrained else None
    model = model_fn(weights=weights)
    replace_with_dct_convs(model)
    model = model.to(device)

    if is_main:
        n_dct = sum(1 for m in model.modules() if isinstance(m, DCTConv2d))
        n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        print(f"=> Replaced convolutions: {n_dct} DCTConv2d, {n_conv} standard Conv2d remaining (1x1)")

    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ---------- Loss / Optimizer / Scheduler ----------
    criterion = nn.CrossEntropyLoss().to(device)

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
    if args.resume:
        if os.path.isfile(args.resume):
            if is_main:
                print(f"=> Loading checkpoint '{args.resume}'")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
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
            raise FileNotFoundError(f"No checkpoint at '{args.resume}'")

    # ---------- Data ----------
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    )

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
            # Compute estimated total network bits and compression ratio
            est_bits = 0.0
            raw_bytes = 0
            with torch.no_grad():
                for m in base_model.modules():
                    if isinstance(m, DCTConv2d):
                        est_bits += calculate_hevc_rate_proxy(
                            m.weight_dct, qstep=args.qstep, steepness=args.steepness
                        ).item()
                        raw_bytes += m.weight_dct.numel() * 4  # float32
            est_kb = est_bits / 8 / 1024
            raw_kb = raw_bytes / 1024
            ratio = raw_kb / max(est_kb, 1e-6)

            print(f"=> Epoch {epoch}: Acc@1 {acc1:.2f}%  (best: {best_acc1:.2f}%)  "
                  f"Sparsity {sparsity*100:.1f}% ({nonzero}/{total} nonzero)  "
                  f"EstSize {est_kb:.1f} KB / {raw_kb:.1f} KB raw  "
                  f"Ratio {ratio:.1f}x")

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
        m.weight_dct.numel() * 4 for m in base_model.modules() if isinstance(m, DCTConv2d)
    ) / 1024

    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward
        outputs = model(images)
        task_loss = criterion(outputs, targets)

        # HEVC rate proxy: differentiable estimate of compressed size
        rate_loss = torch.tensor(0.0, device=device)
        for m in base_model.modules():
            if isinstance(m, DCTConv2d):
                rate_loss = rate_loss + calculate_hevc_rate_proxy(
                    m.weight_dct, qstep=args.qstep, steepness=args.steepness
                )

        total_loss = task_loss + args.lambda_rate * rate_loss

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
                est_kb = rate_losses.val / 8 / 1024
                ratio = raw_kb / max(est_kb, 1e-6)
                print(
                    f"Epoch [{epoch}][{i}/{len(train_loader)}]  "
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})  "
                    f"TaskLoss {task_losses.val:.4f} ({task_losses.avg:.4f})  "
                    f"RateLoss {rate_losses.val:.1f} ({rate_losses.avg:.1f})  "
                    f"EstSize {est_kb:.1f}/{raw_kb:.0f} KB ({ratio:.1f}x)  "
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
