"""
Export sparse DCT coefficients as H.265-encoded video files, and decode
them back for evaluation.

Each DCT layer's coefficients (out_ch, in_ch, K_h, K_w) are reshaped into a
2D image of (out_ch * K_h, in_ch * K_w) quantized levels.  Layers whose 2D
images share the same tile dimensions are grouped into a single video (one
frame per layer).  Images larger than 128x128 are sliced into multiple
128x128 tiles (each tile becomes a separate frame).

Constraints for H.265 validity:
  - Minimum frame size: 8x8
  - Maximum tile size: 128x128
  - Padding uses circular repetition of the data

Pixel format: gray16le or gray (8-bit).  Signed levels are stored with a
per-video center and norm_factor recorded in manifest.json.

Usage:
  Encode:
    python export_h265.py encode [--quantized-model ...] [--output-dir h265_out]
                                 [--qstep 0.1] [--crf 0] [--arch resnet18]

  Decode & evaluate:
    python export_h265.py decode [--h265-dir h265_out] [--data ./imagenette2-320]
                                 [--batch-size 256] [--workers 4]
"""

import argparse
import os
import subprocess
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dct_layers import DCTConv2d, replace_with_dct_convs


def parse_args():
    p = argparse.ArgumentParser(description="Export/decode DCT coefficients as H.265 video")
    sub = p.add_subparsers(dest="command")

    # --- encode subcommand ---
    enc = sub.add_parser("encode", help="Encode DCT coefficients to H.265")
    enc.add_argument("--input", default="./checkpoints/sparse_coefficients.pt",
                     help="path to sparse_coefficients.pt")
    enc.add_argument("--quantized-model", default="./checkpoints/quantized.pth",
                     help="path to quantized.pth (for full dense coefficient export)")
    enc.add_argument("--output-dir", default="./h265_out", help="output directory")
    enc.add_argument("--qstep", default=0.1, type=float,
                     help="quantization step (must match training)")
    enc.add_argument("--arch", default="resnet18",
                     choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    enc.add_argument("--crf", default=0, type=int,
                     help="H.265 CRF (0 = lossless, higher = more compression)")
    enc.add_argument("--bit-depth", default=12, type=int, choices=[8, 10, 12],
                     help="pixel bit depth for encoding (default: 12, max supported by libx265)")
    enc.add_argument("--preset", default="medium",
                     choices=["ultrafast", "superfast", "veryfast", "faster",
                              "fast", "medium", "slow", "slower", "veryslow"],
                     help="x265 encoding preset (default: medium)")
    enc.add_argument("--profile", action="store_true",
                     help="encode with every preset from ultrafast to veryslow and compare")
    enc.add_argument("--quantize", action="store_true",
                     help="quantize to integer levels before encoding "
                          "(default: encode raw float coefficients via center+normalization)")

    # --- decode subcommand ---
    dec = sub.add_parser("decode", help="Decode H.265 back to model and evaluate")
    dec.add_argument("--h265-dir", default="./h265_out",
                     help="directory containing .hevc files and manifest.json")
    dec.add_argument("--data", default="./imagenette2-320",
                     help="path to ImageNet/ImageNette dataset for evaluation")
    dec.add_argument("-b", "--batch-size", default=256, type=int)
    dec.add_argument("-j", "--workers", default=4, type=int)
    dec.add_argument("--non-dct-weights", default="./checkpoints/quantized.pth",
                     help="checkpoint to load non-DCT weights from (fc, bn, 1x1 convs)")

    return p.parse_args()


MIN_DIM = 8
MAX_TILE = 128


def layer_to_2d(weight_dct: torch.Tensor, qstep: float,
                quantize: bool = True) -> np.ndarray:
    """
    Convert a 4D DCT weight tensor to a 2D image.
    Shape: (out_ch, in_ch, K_h, K_w) -> (out_ch * K_h, in_ch * K_w)

    If quantize=True: returns int16 quantized levels (round(w / qstep)).
    If quantize=False: returns float32 raw coefficients (no quantization loss).
    """
    out_ch, in_ch, K_h, K_w = weight_dct.shape
    if quantize:
        data = torch.round(weight_dct / qstep).to(torch.int16)
    else:
        data = weight_dct.detach().float()
    img = data.permute(0, 2, 1, 3).reshape(out_ch * K_h, in_ch * K_w)
    return img.numpy()


def circular_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad image to target size using circular (tiling) repetition."""
    h, w = img.shape
    if h >= target_h and w >= target_w:
        return img[:target_h, :target_w]

    out = np.empty((target_h, target_w), dtype=img.dtype)
    for r in range(0, target_h, h):
        for c in range(0, target_w, w):
            rr = min(h, target_h - r)
            cc = min(w, target_w - c)
            out[r:r+rr, c:c+cc] = img[:rr, :cc]
    return out


def pad_to_min(img: np.ndarray) -> np.ndarray:
    """Pad to at least 8x8 using circular repetition."""
    h, w = img.shape
    new_h = max(h, MIN_DIM)
    new_w = max(w, MIN_DIM)
    if new_h == h and new_w == w:
        return img
    return circular_pad(img, new_h, new_w)


def slice_to_tiles(img: np.ndarray, tile_h: int, tile_w: int):
    """
    Slice a 2D image into tiles of at most (tile_h, tile_w).
    Last tiles are circularly padded if smaller than tile size.
    Returns list of (tile, row_idx, col_idx) tuples.
    """
    h, w = img.shape
    tiles = []
    row_idx = 0
    for r in range(0, h, tile_h):
        col_idx = 0
        rr = min(tile_h, h - r)
        for c in range(0, w, tile_w):
            cc = min(tile_w, w - c)
            tile = img[r:r+rr, c:c+cc]
            # Pad undersized tiles circularly
            if rr < MIN_DIM or cc < MIN_DIM:
                tile = pad_to_min(tile)
            elif rr < tile_h or cc < tile_w:
                # Pad partial edge tiles to full tile size circularly
                tile = circular_pad(tile, max(rr, MIN_DIM), max(cc, MIN_DIM))
            tiles.append((tile, row_idx, col_idx))
            col_idx += 1
        row_idx += 1
    return tiles


def normalize_frame(frame: np.ndarray, bit_depth: int = 12):
    """
    Normalize a single frame to unsigned integer range [0, 2^bit_depth - 1].

    Returns:
        pixels: uint8 or uint16 array (uint16 for 10/12-bit, only low bits used)
        center: float, the center value of the original data
        norm_factor: float, scale factor
    """
    max_val = (1 << bit_depth) - 1
    half = max_val / 2.0

    fmin = float(frame.min())
    fmax = float(frame.max())
    center = (fmin + fmax) / 2.0

    span = fmax - fmin
    if span == 0:
        norm_factor = 1.0
    else:
        norm_factor = max_val / span

    dtype = np.uint8 if bit_depth == 8 else np.uint16
    p = (frame.astype(np.float64) - center) * norm_factor + half
    p = np.clip(np.round(p), 0, max_val).astype(dtype)

    return p, center, norm_factor


def encode_frames_to_h265(frames: list[np.ndarray], output_path: str,
                          crf: int = 0, preset: str = "veryslow",
                          bit_depth: int = 16):
    """
    Normalize (per-frame) and encode a list of 2D numpy arrays as an H.265 video.

    Each frame gets its own center and norm_factor so that layers with
    different value ranges don't lose precision.

    Returns:
        (success, per_frame_norms) or (False, []) on failure.
        per_frame_norms is a list of {"center": float, "norm_factor": float}.
    """
    h, w = frames[0].shape
    assert all(f.shape == (h, w) for f in frames), "All frames must have same dimensions"

    # Normalize each frame independently
    pixels = []
    per_frame_norms = []
    for frame in frames:
        p, center, norm_factor = normalize_frame(frame, bit_depth)
        pixels.append(p)
        per_frame_norms.append({"center": center, "norm_factor": norm_factor})

    pix_fmt_map = {8: "gray", 10: "gray10le", 12: "gray12le"}
    pix_fmt = pix_fmt_map[bit_depth]

    # x265 params: full range, lossless if CRF=0
    x265_params = "log-level=error:range=full"
    if crf == 0:
        x265_params += ":lossless=1"

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", pix_fmt,
        "-video_size", f"{w}x{h}",
        "-framerate", "1",
        "-i", "pipe:",
        "-c:v", "libx265",
        "-preset", preset,
        "-pix_fmt", pix_fmt,
        "-x265-params", x265_params,
        "-color_range", "pc",
    ]
    if crf > 0:
        cmd += ["-crf", str(crf)]
    cmd.append(output_path)

    raw_data = b"".join(p.tobytes() for p in pixels)

    proc = subprocess.run(cmd, input=raw_data, capture_output=True)

    if proc.returncode != 0:
        print(f"  ffmpeg error for {output_path}: {proc.stderr.decode()}")
        return False, []
    return True, per_frame_norms


def decode_h265_frames(video_path: str, n_frames: int, width: int, height: int,
                       bit_depth: int = 16) -> list[np.ndarray]:
    """
    Decode an H.265 video back to a list of 2D numpy arrays (pixels).
    """
    pix_fmt_map = {8: "gray", 10: "gray10le", 12: "gray12le"}
    pix_fmt = pix_fmt_map.get(bit_depth, "gray12le")
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    bpp = 1 if bit_depth == 8 else 2

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-f", "rawvideo",
        "-pixel_format", pix_fmt,
        "pipe:",
    ]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {video_path}: {proc.stderr.decode()}")

    raw = proc.stdout
    frame_bytes = width * height * bpp
    expected = n_frames * frame_bytes
    if len(raw) != expected:
        raise RuntimeError(
            f"Decoded size mismatch for {video_path}: "
            f"got {len(raw)} bytes, expected {expected} "
            f"({n_frames} frames x {width}x{height}x{bpp})"
        )

    frames = []
    for i in range(n_frames):
        buf = raw[i * frame_bytes:(i + 1) * frame_bytes]
        frame = np.frombuffer(buf, dtype=dtype).reshape(height, width)
        frames.append(frame)

    return frames


def denormalize_frames(pixels: list[np.ndarray],
                       per_frame_norms: list[dict],
                       bit_depth: int = 16,
                       quantized: bool = True) -> list[np.ndarray]:
    """
    Convert pixel values back to original values using per-frame center/norm.
    Inverse of normalize_frame:  value = (pixel - max_val/2) / norm_factor + center

    If quantized: rounds to int16 (integer levels).
    If not quantized: returns float64 (raw weight values).
    """
    max_val = (1 << bit_depth) - 1
    half = max_val / 2.0

    results = []
    for p, norms in zip(pixels, per_frame_norms):
        center = norms["center"]
        norm_factor = norms["norm_factor"]
        val = (p.astype(np.float64) - half) / norm_factor + center
        if quantized:
            val = np.round(val).astype(np.int16)
        results.append(val)
    return results


def reassemble_layer(tiles: list[dict], qstep: float,
                     quantized: bool = True) -> torch.Tensor:
    """
    Reassemble a list of decoded tile frames into a 4D DCT weight tensor.

    Each tile dict has: frame (np.ndarray of levels/values), orig_shape,
    img_shape, tile_row, tile_col, n_tile_rows, n_tile_cols.

    If quantized: frame values are integer levels, multiply by qstep.
    If not quantized: frame values are already float weights.

    Returns: (out_ch, in_ch, K_h, K_w) float32 tensor.
    """
    first = tiles[0]
    out_ch, in_ch, K_h, K_w = first["orig_shape"]
    img_h, img_w = first["img_shape"]

    # Reconstruct full 2D image
    dtype = np.int16 if quantized else np.float64
    img = np.zeros((img_h, img_w), dtype=dtype)

    tile_h = tiles[0]["frame"].shape[0]
    tile_w = tiles[0]["frame"].shape[1]

    for t in tiles:
        r = t["tile_row"]
        c = t["tile_col"]
        row_start = r * tile_h
        col_start = c * tile_w
        # Crop to actual image bounds (discard circular padding)
        row_end = min(row_start + tile_h, img_h)
        col_end = min(col_start + tile_w, img_w)
        src_h = row_end - row_start
        src_w = col_end - col_start
        img[row_start:row_end, col_start:col_end] = t["frame"][:src_h, :src_w]

    # Reshape back: (out_ch*K_h, in_ch*K_w) -> (out_ch, K_h, in_ch, K_w) -> (out_ch, in_ch, K_h, K_w)
    tensor_2d = torch.from_numpy(img.astype(np.float32))
    if quantized:
        tensor_2d = tensor_2d * qstep
    weight = tensor_2d.reshape(out_ch, K_h, in_ch, K_w).permute(0, 2, 1, 3).contiguous()
    return weight


def decode_main(args):
    """Decode H.265 videos back to model weights, load into model, evaluate."""
    manifest_path = os.path.join(args.h265_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    qstep = manifest["qstep"]
    arch = manifest["arch"]
    bit_depth = manifest["bit_depth"]
    was_quantized = manifest.get("quantized", True)

    mode_str = "quantized (int levels)" if was_quantized else "float (center+norm)"
    print(f"Decoding H.265 model: arch={arch}, qstep={qstep}, "
          f"bit_depth={bit_depth}, mode={mode_str}")

    # Decode all videos and denormalize back to levels/values
    layer_tiles = {}  # layer_name -> list of tile dicts

    for video_name, vinfo in manifest["videos"].items():
        video_path = os.path.join(args.h265_dir, video_name)
        w = vinfo["frame_width"]
        h = vinfo["frame_height"]
        n = vinfo["n_frames"]
        vbit = vinfo.get("bit_depth", bit_depth)

        # Build per-frame norms from manifest
        per_frame_norms = []
        for finfo in vinfo["frames"]:
            per_frame_norms.append({
                "center": finfo["center"],
                "norm_factor": finfo["norm_factor"],
            })

        print(f"  Decoding {video_name}: {n} frames @ {w}x{h}...")
        pixels = decode_h265_frames(video_path, n, w, h, vbit)
        levels = denormalize_frames(pixels, per_frame_norms, vbit,
                                    quantized=was_quantized)

        for fi, finfo in enumerate(vinfo["frames"]):
            layer_name = finfo["layer_name"]
            if layer_name not in layer_tiles:
                layer_tiles[layer_name] = []
            layer_tiles[layer_name].append({
                "frame": levels[fi],
                "orig_shape": tuple(finfo["orig_shape"]),
                "img_shape": tuple(finfo["img_shape"]),
                "tile_row": finfo["tile_row"],
                "tile_col": finfo["tile_col"],
                "n_tile_rows": finfo["n_tile_rows"],
                "n_tile_cols": finfo["n_tile_cols"],
            })

    # Build model and load non-DCT weights
    model_fn = getattr(models, arch)
    model = model_fn(weights=None)
    replace_with_dct_convs(model)

    if os.path.isfile(args.non_dct_weights):
        print(f"  Loading non-DCT weights from {args.non_dct_weights}")
        ckpt = torch.load(args.non_dct_weights, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        # Load only non-DCT weights (bn, fc, 1x1 convs)
        model_state = model.state_dict()
        dct_keys = set()
        for name, m in model.named_modules():
            if isinstance(m, DCTConv2d):
                dct_keys.add(f"{name}.weight_dct")
                if m.bias is not None:
                    dct_keys.add(f"{name}.bias")
        for k, v in state.items():
            if k not in dct_keys and k in model_state:
                model_state[k] = v
        model.load_state_dict(model_state)
    else:
        print(f"  WARNING: no non-DCT weights found at {args.non_dct_weights}")
        print(f"           BN/FC/1x1 layers will be random — accuracy will be meaningless")

    # Reassemble DCT weights from decoded tiles
    for name, m in model.named_modules():
        if isinstance(m, DCTConv2d) and name in layer_tiles:
            weight = reassemble_layer(layer_tiles[name], qstep,
                                     quantized=was_quantized)
            m.weight_dct.data.copy_(weight)
            nnz = (torch.round(weight / qstep) != 0).sum().item()
            print(f"  Loaded {name}: {list(weight.shape)}  nnz={nnz}/{weight.numel()}")
        elif isinstance(m, DCTConv2d):
            # Layer not in H.265 data — zero it out
            m.weight_dct.data.zero_()
            print(f"  Zeroed {name}: not in H.265 data")

    # Load DCT-layer biases from checkpoint if available
    if os.path.isfile(args.non_dct_weights):
        for name, m in model.named_modules():
            if isinstance(m, DCTConv2d) and m.bias is not None:
                bias_key = f"{name}.bias"
                if bias_key in state:
                    m.bias.data.copy_(state[bias_key])

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss().to(device)

    print(f"\nEvaluating decoded model on {valdir}...")
    top1_sum = 0.0
    top5_sum = 0.0
    loss_sum = 0.0
    total = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            _, pred = outputs.topk(5, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            top1_sum += correct[:1].reshape(-1).float().sum(0).item()
            top5_sum += correct[:5].reshape(-1).float().sum(0).item()
            loss_sum += loss.item() * batch_size
            total += batch_size

    acc1 = 100.0 * top1_sum / total
    acc5 = 100.0 * top5_sum / total
    avg_loss = loss_sum / total

    print(f"\n--- Decoded Model Evaluation ---")
    print(f"  Loss:  {avg_loss:.4f}")
    print(f"  Acc@1: {acc1:.2f}%")
    print(f"  Acc@5: {acc5:.2f}%")

    # Compare with pre-quantization if available
    if os.path.isfile(args.non_dct_weights):
        ckpt = torch.load(args.non_dct_weights, map_location="cpu", weights_only=False)
        if "acc1_before_quant" in ckpt:
            print(f"\n  (Before quantization: Acc@1 {ckpt['acc1_before_quant']:.2f}%)")
        if "acc1_after_quant" in ckpt:
            print(f"  (After quantization, before H.265: Acc@1 {ckpt['acc1_after_quant']:.2f}%)")
        print(f"  (After H.265 decode: Acc@1 {acc1:.2f}%)")

    # Total H.265 file size
    total_h265 = sum(
        os.path.getsize(os.path.join(args.h265_dir, vn))
        for vn in manifest["videos"]
    )
    full_model_bytes = sum(
        m.weight_dct.numel() * 4
        for m in model.modules() if isinstance(m, DCTConv2d)
    )
    print(f"\n  H.265 total size:    {total_h265/1024:.1f} KB")
    print(f"  Float32 model size:  {full_model_bytes/1024:.1f} KB")
    print(f"  Compression ratio:   {full_model_bytes/max(total_h265,1):.1f}x")


PROFILE_PRESETS = [
    "ultrafast", "superfast", "veryfast", "faster",
    "fast", "medium", "slow", "slower", "veryslow",
]


def _load_model_for_encode(args):
    """Load model and prepare tile groups. Shared by encode and profile."""
    model_fn = getattr(models, args.arch)
    model = model_fn(weights=None)
    replace_with_dct_convs(model)

    do_quantize = getattr(args, "quantize", False)

    # When not quantizing, prefer best.pth (unquantized) over quantized.pth
    if not do_quantize:
        best_path = os.path.join(os.path.dirname(args.quantized_model), "best.pth")
        if os.path.isfile(best_path):
            print(f"Loading unquantized model from {best_path} (--no-quantize)")
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            state = ckpt["state_dict"]
            if any(k.startswith("module.") for k in state):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state)
        elif os.path.isfile(args.quantized_model):
            print(f"WARNING: best.pth not found, using {args.quantized_model}")
            print(f"         Weights may already be quantized.")
            ckpt = torch.load(args.quantized_model, map_location="cpu", weights_only=False)
            state = ckpt["state_dict"]
            if any(k.startswith("module.") for k in state):
                state = {k.removeprefix("module."): v for k, v in state.items()}
            model.load_state_dict(state)
        else:
            raise FileNotFoundError(f"No model found (tried best.pth and {args.quantized_model})")
    elif os.path.isfile(args.quantized_model):
        print(f"Loading quantized model from {args.quantized_model}")
        ckpt = torch.load(args.quantized_model, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)
    elif os.path.isfile(args.input):
        print(f"Loading sparse coefficients from {args.input}")
        sparse_data = torch.load(args.input, map_location="cpu", weights_only=False)
        for name, m in model.named_modules():
            if isinstance(m, DCTConv2d) and name in sparse_data:
                m.weight_dct.data.zero_()
                indices = sparse_data[name]["indices"]
                values = sparse_data[name]["values"].to(torch.float32)
                if indices.numel() > 0:
                    for idx, val in zip(indices, values):
                        m.weight_dct.data[idx[0], idx[1], idx[2], idx[3]] = val * args.qstep
    else:
        raise FileNotFoundError(f"Neither {args.quantized_model} nor {args.input} found")

    # Convert each layer to 2D image
    layer_images = []
    for name, m in model.named_modules():
        if isinstance(m, DCTConv2d):
            img = layer_to_2d(m.weight_dct, args.qstep, quantize=do_quantize)
            out_ch, in_ch, K_h, K_w = m.weight_dct.shape
            layer_images.append({
                "name": name,
                "img": img,
                "orig_shape": (out_ch, in_ch, K_h, K_w),
                "img_shape": img.shape,
            })
            nnz = np.count_nonzero(img)
            mode = "int16 quantized" if do_quantize else "float32"
            print(f"  {name}: weight {list(m.weight_dct.shape)} -> "
                  f"2D {img.shape[0]}x{img.shape[1]}  "
                  f"nnz={nnz}/{img.size}  ({mode})")

    # Build tiles from each layer, group by tile dimensions
    tile_groups = {}
    frame_metadata = {}  # same structure but without the numpy frame data

    for li in layer_images:
        img = li["img"]
        h, w = img.shape

        if h <= MAX_TILE and w <= MAX_TILE:
            frame = pad_to_min(img)
            fh, fw = frame.shape
            key = (fh, fw)
            if key not in tile_groups:
                tile_groups[key] = []
            tile_groups[key].append({
                "frame": frame,
                "layer_name": li["name"],
                "orig_shape": li["orig_shape"],
                "img_shape": li["img_shape"],
                "tile_row": 0,
                "tile_col": 0,
                "n_tile_rows": 1,
                "n_tile_cols": 1,
            })
        else:
            n_tile_rows = (h + MAX_TILE - 1) // MAX_TILE
            n_tile_cols = (w + MAX_TILE - 1) // MAX_TILE
            tile_h = (h + n_tile_rows - 1) // n_tile_rows
            tile_w = (w + n_tile_cols - 1) // n_tile_cols
            tile_h = max(MIN_DIM, tile_h + (tile_h % 2))
            tile_w = max(MIN_DIM, tile_w + (tile_w % 2))

            tiles = slice_to_tiles(img, tile_h, tile_w)
            key = (tile_h, tile_w)
            if key not in tile_groups:
                tile_groups[key] = []
            for tile, row_idx, col_idx in tiles:
                tile = circular_pad(tile, tile_h, tile_w)
                tile_groups[key].append({
                    "frame": tile,
                    "layer_name": li["name"],
                    "orig_shape": li["orig_shape"],
                    "img_shape": li["img_shape"],
                    "tile_row": row_idx,
                    "tile_col": col_idx,
                    "n_tile_rows": n_tile_rows,
                    "n_tile_cols": n_tile_cols,
                })

    full_model_bytes = sum(
        m.weight_dct.numel() * 4
        for m in model.modules() if isinstance(m, DCTConv2d)
    )

    return model, tile_groups, full_model_bytes, do_quantize


def _encode_tile_groups(tile_groups: dict, output_dir: str, crf: int,
                        preset: str, bit_depth: int, verbose: bool = True):
    """
    Encode all tile groups to H.265 videos under output_dir.
    Returns (total_raw_bytes, total_h265_bytes, manifest_videos dict).
    """
    import time as _time

    total_raw_bytes = 0
    total_h265_bytes = 0
    manifest_videos = {}
    t0 = _time.monotonic()

    for (th, tw), entries in sorted(tile_groups.items()):
        video_name = f"dct_{th}x{tw}.hevc"
        video_path = os.path.join(output_dir, video_name)

        frames = [e["frame"] for e in entries]
        raw_bytes = sum(f.size * (bit_depth // 8) for f in frames)

        ok, per_frame_norms = encode_frames_to_h265(
            frames, video_path,
            crf=crf, preset=preset, bit_depth=bit_depth,
        )

        if ok:
            h265_bytes = os.path.getsize(video_path)
            ratio = raw_bytes / max(h265_bytes, 1)
            if verbose:
                print(f"  {video_name}: {len(frames)} frames @ {tw}x{th}  "
                      f"raw={raw_bytes/1024:.1f} KB  "
                      f"h265={h265_bytes/1024:.1f} KB  "
                      f"ratio={ratio:.1f}x")
            total_raw_bytes += raw_bytes
            total_h265_bytes += h265_bytes

            manifest_videos[video_name] = {
                "frame_width": tw,
                "frame_height": th,
                "bit_depth": bit_depth,
                "n_frames": len(frames),
                "frames": [],
            }
            for i, e in enumerate(entries):
                frame_entry = {
                    "frame_index": i,
                    "layer_name": e["layer_name"],
                    "orig_shape": list(e["orig_shape"]),
                    "img_shape": list(e["img_shape"]),
                    "tile_row": e["tile_row"],
                    "tile_col": e["tile_col"],
                    "n_tile_rows": e["n_tile_rows"],
                    "n_tile_cols": e["n_tile_cols"],
                    "center": per_frame_norms[i]["center"],
                    "norm_factor": per_frame_norms[i]["norm_factor"],
                }
                manifest_videos[video_name]["frames"].append(frame_entry)

    elapsed = _time.monotonic() - t0
    return total_raw_bytes, total_h265_bytes, manifest_videos, elapsed


def encode_main(args):
    """Encode DCT coefficients to H.265 videos."""
    os.makedirs(args.output_dir, exist_ok=True)

    model, tile_groups, full_model_bytes, do_quantize = _load_model_for_encode(args)

    if args.profile:
        # Profile mode: encode with every preset and compare
        print(f"\n{'='*70}")
        print(f"PROFILE MODE: encoding with all presets (CRF={args.crf}, {args.bit_depth}-bit)")
        print(f"{'='*70}")

        results = []
        for preset in PROFILE_PRESETS:
            preset_dir = os.path.join(args.output_dir, f"profile_{preset}")
            os.makedirs(preset_dir, exist_ok=True)

            print(f"\n--- Preset: {preset} ---")
            total_raw, total_h265, _, elapsed = _encode_tile_groups(
                tile_groups, preset_dir, args.crf, preset, args.bit_depth,
                verbose=False,
            )
            ratio_f32 = full_model_bytes / max(total_h265, 1)
            ratio_raw = total_raw / max(total_h265, 1)
            results.append({
                "preset": preset,
                "h265_bytes": total_h265,
                "time_s": elapsed,
                "ratio_vs_f32": ratio_f32,
                "ratio_vs_raw": ratio_raw,
            })
            print(f"  H.265: {total_h265/1024:.1f} KB  "
                  f"vs f32: {ratio_f32:.1f}x  "
                  f"vs raw: {ratio_raw:.1f}x  "
                  f"time: {elapsed:.1f}s")

        # Summary table
        print(f"\n{'='*70}")
        print(f"  {'Preset':<12} {'Size':>10} {'vs f32':>10} {'vs raw':>10} {'Time':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for r in results:
            print(f"  {r['preset']:<12} "
                  f"{r['h265_bytes']/1024:>8.1f}KB "
                  f"{r['ratio_vs_f32']:>9.1f}x "
                  f"{r['ratio_vs_raw']:>9.1f}x "
                  f"{r['time_s']:>8.1f}s")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  Float32 raw: {full_model_bytes/1024:.1f} KB")
        return

    # Normal single-preset encode
    print(f"\nEncoding {len(tile_groups)} video(s) at {args.bit_depth}-bit "
          f"(preset={args.preset})...")
    total_raw_bytes, total_h265_bytes, manifest_videos, elapsed = _encode_tile_groups(
        tile_groups, args.output_dir, args.crf, args.preset, args.bit_depth,
    )

    manifest = {
        "qstep": args.qstep,
        "arch": args.arch,
        "bit_depth": args.bit_depth,
        "quantized": do_quantize,
        "reconstruction": (
            "level = (pixel - max_val/2) / norm_factor + center; weight = level * qstep"
            if do_quantize else
            "weight = (pixel - max_val/2) / norm_factor + center"
        ),
        "videos": manifest_videos,
    }

    print(f"\n--- Summary (preset={args.preset}, {elapsed:.1f}s) ---")
    print(f"Full model DCT weights (float32): {full_model_bytes/1024:.1f} KB")
    print(f"Quantized frames (int16 raw):     {total_raw_bytes/1024:.1f} KB")
    print(f"H.265 encoded:                    {total_h265_bytes/1024:.1f} KB")
    if total_h265_bytes > 0:
        print(f"Compression vs float32:           {full_model_bytes/total_h265_bytes:.1f}x")
        print(f"Compression vs int16 raw:         {total_raw_bytes/total_h265_bytes:.1f}x")

    manifest["summary"] = {
        "full_model_float32_bytes": full_model_bytes,
        "quantized_int16_raw_bytes": total_raw_bytes,
        "h265_encoded_bytes": total_h265_bytes,
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.command == "encode":
        encode_main(args)
    elif args.command == "decode":
        decode_main(args)
    else:
        print("Usage: python export_h265.py {encode|decode} [options]")
        print("  python export_h265.py encode --help")
        print("  python export_h265.py decode --help")
