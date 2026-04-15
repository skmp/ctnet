"""
CTGPT training: Fine-tune GPT-2 with DCT compression on Shakespeare.

Usage:
    python ctgpt_train.py [--epochs 50] [--lambda-rate 1e-5]
"""

import argparse
import os
import math
import time
import urllib.request

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ctgpt_model import DCTLinear, _is_dct_layer, replace_linears_with_dct, probe_sparsity, dct_config
from dct_utils import calculate_hevc_rate_proxy, estimate_h265_size_bits


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args():
    p = argparse.ArgumentParser(description="CTGPT: DCT-compressed GPT-2 training")
    p.add_argument("--epochs", default=50, type=int)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--seq-len", default=256, type=int, help="sequence length")
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--weight-decay", default=0.001, type=float)
    p.add_argument("--lambda-rate", default=1e-5, type=float)
    p.add_argument("--lambda-alpha", default=0.5, type=float,
                   help="coupled L2: lambda_l2 = alpha * lambda_rate")
    p.add_argument("--qstep", default=0.1, type=float)
    p.add_argument("--steepness", default=10.0, type=float)
    p.add_argument("--dct-block-size", default=16, type=int)
    p.add_argument("--pixel-bit-depth", default=8, type=int, choices=[0, 8, 10, 12])
    p.add_argument("--rate-warmup-epochs", default=4, type=int)
    p.add_argument("--no-train-noise", action="store_true")
    p.add_argument("--dct-dropout", default=0.0, type=float)
    p.add_argument("--output-dir", default="./ctgpt_checkpoints")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--print-freq", default=50, type=int)
    p.add_argument("--eval-interval", default=1, type=int, help="evaluate every N epochs")
    p.add_argument("--generate-interval", default=5, type=int, help="generate sample every N epochs")
    return p.parse_args()


def download_shakespeare(data_dir):
    """Download Shakespeare dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.isfile(path):
        print(f"Downloading Shakespeare to {path}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path


def prepare_data(data_dir, tokenizer, seq_len):
    """Tokenize Shakespeare and split into train/val."""
    text_path = download_shakespeare(data_dir)
    with open(text_path) as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)

    # 90/10 split
    n = len(tokens)
    split = int(n * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    print(f"Shakespeare: {n} tokens ({split} train, {n - split} val)")
    return train_tokens, val_tokens


def get_batch(tokens, batch_size, seq_len, device):
    """Get a random batch of sequences."""
    n = len(tokens) - seq_len - 1
    ix = torch.randint(0, n, (batch_size,))
    x = torch.stack([tokens[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, tokens, batch_size, seq_len, device, n_batches=20):
    """Estimate loss over multiple batches."""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(tokens, batch_size, seq_len, device)
        outputs = model(x, labels=y)
        losses.append(outputs.loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="ROMEO:", max_tokens=200):
    """Generate a text sample."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids, max_new_tokens=max_tokens,
        temperature=0.8, top_k=40, do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    model.train()
    return text


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Configure DCT training
    dct_config.qstep = args.qstep
    dct_config.train_noise = not args.no_train_noise
    dct_config.dct_dropout = args.dct_dropout
    dct_config.pixel_bit_depth = args.pixel_bit_depth

    lambda_l2 = args.lambda_alpha * args.lambda_rate if args.lambda_alpha > 0 else 0.0

    # Load pretrained GPT-2
    print("Loading GPT-2 pretrained...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Replace linear layers with DCTLinear
    # Skip lm_head (tied to wte) and embeddings
    print(f"Replacing linear layers with DCTLinear (block_size={args.dct_block_size})...")
    replace_linears_with_dct(
        model.transformer, block_size=args.dct_block_size,
    )

    n_dct = sum(1 for m in model.modules() if _is_dct_layer(m))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    dct_params = sum(m.weight_dct.numel() for m in model.modules() if _is_dct_layer(m))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"=> {n_dct} DCTLinear, {n_linear} standard Linear remaining")
    print(f"=> DCT params: {dct_params / 1e6:.1f}M / {total_params / 1e6:.1f}M total")

    model = model.to(device)

    # Auto-resume
    start_epoch = 0
    best_val_loss = float("inf")
    smallest_est_kb = float("inf")
    best_acc_at_ratio = {}

    ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")
    if os.path.isfile(ckpt_path):
        print(f"=> Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"=> Resumed from epoch {start_epoch}")

    # Data
    train_tokens, val_tokens = prepare_data(args.data_dir, tokenizer, args.seq_len)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    # LR schedule: cosine annealing with warmup
    warmup_epochs = min(5, args.epochs // 10)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Raw model size for compression ratio
    raw_kb = sum(
        m.weight_dct.numel() * 4 for m in model.modules() if _is_dct_layer(m)
    ) / 1024

    print(f"\n=> Training for {args.epochs} epochs")
    print(f"=> Lambda rate: {args.lambda_rate}, L2 alpha: {args.lambda_alpha}")
    print(f"=> Qstep: {args.qstep}, Pixel bit depth: {args.pixel_bit_depth}")
    print(f"=> DCT raw size: {raw_kb:.0f} KB")
    print()

    # Steps per epoch (approximate)
    steps_per_epoch = len(train_tokens) // (args.batch_size * args.seq_len)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        total_task_loss = 0
        total_rate_loss = 0
        n_steps = 0

        for step in range(steps_per_epoch):
            x, y = get_batch(train_tokens, args.batch_size, args.seq_len, device)

            outputs = model(x, labels=y)
            task_loss = outputs.loss

            # Rate proxy on DCT layers
            rate_loss = torch.tensor(0.0, device=device)
            l2_loss = torch.tensor(0.0, device=device)
            for m in model.modules():
                if _is_dct_layer(m):
                    rate_loss = rate_loss + calculate_hevc_rate_proxy(
                        m.weight_dct, qstep=args.qstep, steepness=args.steepness
                    )
                    if lambda_l2 > 0:
                        l2_loss = l2_loss + m.weight_dct.pow(2).sum()

            # Rate warmup
            if args.rate_warmup_epochs > 0 and epoch < args.rate_warmup_epochs:
                rate_scale = (epoch + 1) / args.rate_warmup_epochs
            else:
                rate_scale = 1.0

            total_loss = (task_loss
                          + args.lambda_rate * rate_scale * rate_loss
                          + lambda_l2 * rate_scale * l2_loss)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_task_loss += task_loss.item()
            total_rate_loss += rate_loss.item()
            n_steps += 1

            if step % args.print_freq == 0 and step > 0:
                avg_task = total_task_loss / n_steps
                avg_rate = total_rate_loss / n_steps
                ppl = math.exp(min(avg_task, 20))
                print(f"  Epoch {epoch} step {step}/{steps_per_epoch}  "
                      f"TaskLoss {avg_task:.4f}  PPL {ppl:.1f}  "
                      f"RateLoss {avg_rate:.0f}")

        scheduler.step()
        elapsed = time.time() - t0
        avg_task = total_task_loss / max(n_steps, 1)
        avg_rate = total_rate_loss / max(n_steps, 1)

        # Validation
        val_loss = estimate_loss(model, val_tokens, args.batch_size, args.seq_len, device)
        val_ppl = math.exp(min(val_loss, 20))

        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)

        # Sparsity and size estimate
        total, nonzero, sparsity = probe_sparsity(model, qstep=args.qstep)
        est_by_depth, _ = estimate_h265_size_bits(
            model.named_modules(), bit_depths=(8, 10, 12)
        )
        est_8b_kb = est_by_depth[8] / 8 / 1024

        # Save checkpoints
        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "best_val_loss": best_val_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, ckpt_path)
        if is_best:
            torch.save(state, os.path.join(args.output_dir, "best.pt"))

        is_smallest = est_8b_kb < smallest_est_kb
        if is_smallest:
            smallest_est_kb = est_8b_kb
            torch.save(state, os.path.join(args.output_dir, "smallest.pt"))

        # Per-ratio best
        ratio_int = int(raw_kb / max(est_8b_kb, 1e-6))
        ratio_tag = ""
        if ratio_int >= 2:
            prev_best = best_acc_at_ratio.get(ratio_int, float("inf"))
            if val_loss < prev_best:
                best_acc_at_ratio[ratio_int] = val_loss
                torch.save(state, os.path.join(args.output_dir, f"best_{ratio_int}x.pt"))
                ratio_tag = f" *best@{ratio_int}x*"

        est_parts = []
        for b in (8, 10, 12):
            kb = est_by_depth[b] / 8 / 1024
            r = int(raw_kb / max(kb, 1e-6))
            est_parts.append(f"{b}b:{kb:.0f}KB({r}x)")
        est_str = "  ".join(est_parts)

        best_tag = " *best*" if is_best else ""
        small_tag = " *smallest*" if is_smallest else ""

        print(f"=> Epoch {epoch}: ValLoss {val_loss:.4f}  PPL {val_ppl:.1f}  "
              f"(best: {best_val_loss:.4f})  "
              f"Sparsity {sparsity*100:.1f}% ({nonzero}/{total})  "
              f"Est[{est_str}] / {raw_kb:.0f}KB raw  "
              f"[{elapsed:.0f}s]"
              f"{best_tag}{small_tag}{ratio_tag}")

        # Generate sample
        if epoch % args.generate_interval == 0:
            sample = generate_sample(model, tokenizer, device)
            print(f"  Sample: {sample[:200]}...")
            print()


if __name__ == "__main__":
    main()
