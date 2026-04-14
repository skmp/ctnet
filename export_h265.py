"""
Export DCT coefficients as H.265-encoded video files, and decode them back
for evaluation.

Each DCT layer's coefficients (out_ch, in_ch, K_h, K_w) are reshaped into a
2D image of (out_ch * K_h, in_ch * K_w).  Layers whose 2D images share the
same tile dimensions are grouped into a single video (one frame per layer).
Images larger than 128x128 are sliced into tiles; smaller than 8x8 are
circularly padded.

Float coefficients are stored via per-frame center+normalization to N-bit
pixel range.  The center and norm_factor are recorded in manifest.json for
exact reconstruction.

Usage:
  Encode:
    python export_h265.py encode [--model best.pth] [--output-dir h265_out]
                                 [--crf 0] [--arch resnet18]

  Decode & evaluate:
    python export_h265.py decode [--h265-dir h265_out] [--data ./imagenette2-320]
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

from dct_layers import DCTConv2d, ChannelDCTConv1x1, _is_dct_layer, replace_with_dct_convs


def parse_args():
    p = argparse.ArgumentParser(description="Export/decode DCT coefficients as H.265 video")
    sub = p.add_subparsers(dest="command")

    # --- encode subcommand ---
    enc = sub.add_parser("encode", help="Encode DCT coefficients to H.265")
    enc.add_argument("--model", default="./checkpoints/best.pth",
                     help="path to model checkpoint (default: best.pth)")
    enc.add_argument("--output-dir", default="./h265_out", help="output directory")
    enc.add_argument("--arch", default="resnet18",
                     choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    enc.add_argument("--crf", default=0, type=int,
                     help="H.265 CRF (0 = lossless, higher = more compression)")
    enc.add_argument("--bit-depth", default=8, type=int, choices=[8, 10, 12],
                     help="pixel bit depth for encoding (default: 8)")
    enc.add_argument("--preset", default="medium",
                     choices=["ultrafast", "superfast", "veryfast", "faster",
                              "fast", "medium", "slow", "slower", "veryslow"],
                     help="x265 encoding preset (default: medium)")
    enc.add_argument("--profile", action="store_true",
                     help="encode with every preset from ultrafast to veryslow and compare")
    enc.add_argument("--dither", default=0.0, type=float,
                     help="subtractive dither amplitude (0 = off, 0.5 = standard)")
    enc.add_argument("--yuv", action="store_true",
                     help="encode as YUV 4:2:0 (Main/Main10 profile) for hardware "
                          "decoder compatibility on phones, Mac, Windows, etc.")
    enc.add_argument("--bn-crf", default=0, type=int,
                     help="CRF for BN layers (default: 0 = lossless). BN running_var "
                          "needs high precision; use 0 unless you know what you're doing.")
    enc.add_argument("--bn-bit-depth", default=12, type=int, choices=[8, 10, 12],
                     help="bit depth for BN layers (default: 12). Higher prevents "
                          "near-zero running_var from going negative after roundtrip.")

    # --- decode subcommand ---
    dec = sub.add_parser("decode", help="Decode H.265 back to model and evaluate")
    dec.add_argument("--h265-dir", default="./h265_out",
                     help="directory containing .hevc files and manifest.json")
    dec.add_argument("--data", default="./imagenette2-320",
                     help="path to ImageNet/ImageNette dataset for evaluation")
    dec.add_argument("-b", "--batch-size", default=256, type=int)
    dec.add_argument("-j", "--workers", default=4, type=int)

    return p.parse_args()


MIN_DIM = 16  # libx265 minimum frame dimension
MAX_TILE = 128


def layer_to_2d(weight_dct: torch.Tensor) -> np.ndarray:
    """
    Convert a 4D DCT weight tensor to a 2D float32 image.
    Shape: (out_ch, in_ch, K_h, K_w) -> (out_ch * K_h, in_ch * K_w)
    """
    out_ch, in_ch, K_h, K_w = weight_dct.shape
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
            if rr < MIN_DIM or cc < MIN_DIM:
                tile = pad_to_min(tile)
            elif rr < tile_h or cc < tile_w:
                tile = circular_pad(tile, max(rr, MIN_DIM), max(cc, MIN_DIM))
            tiles.append((tile, row_idx, col_idx))
            col_idx += 1
        row_idx += 1
    return tiles


DITHER_BASE_SEED = 0xDC7C0EFF


def _dither_noise(shape: tuple, frame_index: int, amplitude: float) -> np.ndarray:
    """Deterministic white noise in [-amplitude, +amplitude) for subtractive dither."""
    rng = np.random.default_rng(DITHER_BASE_SEED + frame_index)
    return rng.uniform(-amplitude, amplitude, size=shape)


def normalize_frame(frame: np.ndarray, bit_depth: int = 12,
                    frame_index: int = 0, dither: float = 0.0):
    """
    Normalize a single frame to unsigned integer range [0, 2^bit_depth - 1]
    using center+scale.  Optionally adds subtractive dither before rounding.

    Returns: (pixels, center, norm_factor)
    """
    max_val = (1 << bit_depth) - 1
    half = max_val / 2.0

    fmin = float(frame.min())
    fmax = float(frame.max())
    center = (fmin + fmax) / 2.0

    span = fmax - fmin
    norm_factor = max_val / span if span > 0 else 1.0

    p = (frame.astype(np.float64) - center) * norm_factor + half

    if dither > 0:
        p = p + _dither_noise(p.shape, frame_index, amplitude=dither)

    dtype = np.uint8 if bit_depth == 8 else np.uint16
    p = np.clip(np.round(p), 0, max_val).astype(dtype)

    return p, center, norm_factor


def _pad_to_even(frames_list: list[np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    """Pad frames to even dimensions (required for YUV 4:2:0)."""
    h, w = frames_list[0].shape
    new_h = h + (h % 2)
    new_w = w + (w % 2)
    if new_h == h and new_w == w:
        return frames_list, h, w
    padded = []
    for f in frames_list:
        p = np.zeros((new_h, new_w), dtype=f.dtype)
        p[:h, :w] = f
        if new_h > h:
            p[h, :w] = f[0, :]
            if new_w > w:
                p[h, w] = f[0, 0]
        if new_w > w:
            p[:h, w] = f[:, 0]
        padded.append(p)
    return padded, new_h, new_w


def _gray_to_yuv420(frame: np.ndarray, bit_depth: int) -> bytes:
    """Convert grayscale frame to YUV 4:2:0 planar. Luma = data, chroma = neutral."""
    h, w = frame.shape
    chroma_h, chroma_w = h // 2, w // 2
    neutral = 1 << (bit_depth - 1)

    if bit_depth == 8:
        y_plane = frame.astype(np.uint8).tobytes()
        uv_val = np.full((chroma_h, chroma_w), neutral, dtype=np.uint8)
    else:
        y_plane = frame.astype(np.uint16).tobytes()
        uv_val = np.full((chroma_h, chroma_w), neutral, dtype=np.uint16)

    return y_plane + uv_val.tobytes() + uv_val.tobytes()


def encode_frames_to_h265(frames: list[np.ndarray], output_path: str,
                          crf: int = 0, preset: str = "medium",
                          bit_depth: int = 12, dither: float = 0.0,
                          yuv: bool = False):
    """
    Normalize (per-frame center+scale) and encode as H.265 video.

    Returns: (success, per_frame_norms, actual_bit_depth) or (False, [], 0).
    """
    h, w = frames[0].shape
    assert all(f.shape == (h, w) for f in frames), "All frames must have same dimensions"

    # YUV 4:2:0 only supports up to 10-bit
    if yuv and bit_depth not in (8, 10):
        bit_depth = 10

    # Normalize each frame independently
    pixels = []
    per_frame_norms = []
    for i, frame in enumerate(frames):
        p, center, norm_factor = normalize_frame(frame, bit_depth,
                                                 frame_index=i, dither=dither)
        pixels.append(p)
        per_frame_norms.append({"center": center, "norm_factor": norm_factor})

    if yuv:
        pixels, enc_h, enc_w = _pad_to_even(pixels)
        if bit_depth == 8:
            in_pix_fmt, out_pix_fmt, h265_profile = "yuv420p", "yuv420p", "main"
        else:
            in_pix_fmt, out_pix_fmt, h265_profile = "yuv420p10le", "yuv420p10le", "main10"

        if crf == 0:
            x265_params = ("log-level=error:range=full:lossless=1"
                           ":keyint=-1:bframes=16:ref=16:b-adapt=2:rc-lookahead=250")
        else:
            x265_params = "log-level=error:range=full"

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", in_pix_fmt,
            "-video_size", f"{enc_w}x{enc_h}", "-framerate", "1",
            "-i", "pipe:",
            "-c:v", "libx265", "-preset", preset,
            "-pix_fmt", out_pix_fmt, "-profile:v", h265_profile,
            "-x265-params", x265_params, "-color_range", "pc",
        ]
        if crf > 0:
            cmd += ["-crf", str(crf)]
        cmd.append(output_path)
        raw_data = b"".join(_gray_to_yuv420(p, bit_depth) for p in pixels)
    else:
        enc_h, enc_w = h, w
        pix_fmt_map = {8: "gray", 10: "gray10le", 12: "gray12le"}
        pix_fmt = pix_fmt_map[bit_depth]

        if crf == 0:
            x265_params = ("log-level=error:range=full:lossless=1"
                           ":keyint=-1:bframes=16:ref=16:b-adapt=2:rc-lookahead=250")
        else:
            x265_params = "log-level=error:range=full"

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pixel_format", pix_fmt,
            "-video_size", f"{enc_w}x{enc_h}", "-framerate", "1",
            "-i", "pipe:",
            "-c:v", "libx265", "-preset", preset,
            "-pix_fmt", pix_fmt,
            "-x265-params", x265_params, "-color_range", "pc",
        ]
        if crf > 0:
            cmd += ["-crf", str(crf)]
        cmd.append(output_path)
        raw_data = b"".join(p.tobytes() for p in pixels)

    proc = subprocess.run(cmd, input=raw_data, capture_output=True)

    if proc.returncode != 0:
        print(f"  ffmpeg error for {output_path}: {proc.stderr.decode()}")
        return False, [], 0
    return True, per_frame_norms, bit_depth


def decode_h265_frames(video_path: str, n_frames: int, width: int, height: int,
                       bit_depth: int = 12, yuv: bool = False) -> list[np.ndarray]:
    """Decode H.265 video back to a list of 2D numpy arrays (luma pixels)."""
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    bpp = 1 if bit_depth == 8 else 2

    if yuv:
        pix_fmt = "yuv420p" if bit_depth == 8 else "yuv420p10le"
        y_samples = width * height
        uv_samples = (width // 2) * (height // 2)
        frame_bytes = (y_samples + 2 * uv_samples) * bpp
    else:
        pix_fmt_map = {8: "gray", 10: "gray10le", 12: "gray12le"}
        pix_fmt = pix_fmt_map.get(bit_depth, "gray12le")
        frame_bytes = width * height * bpp

    cmd = ["ffmpeg", "-y", "-i", video_path,
           "-f", "rawvideo", "-pixel_format", pix_fmt, "pipe:"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {video_path}: {proc.stderr.decode()}")

    raw = proc.stdout
    expected = n_frames * frame_bytes
    if len(raw) != expected:
        raise RuntimeError(
            f"Decoded size mismatch for {video_path}: "
            f"got {len(raw)} bytes, expected {expected}")

    frames = []
    y_bytes = width * height * bpp
    for i in range(n_frames):
        buf = raw[i * frame_bytes:i * frame_bytes + y_bytes]
        frame = np.frombuffer(buf, dtype=dtype).reshape(height, width)
        frames.append(frame)

    return frames


def denormalize_frames(pixels: list[np.ndarray],
                       per_frame_norms: list[dict],
                       bit_depth: int = 12,
                       dither: float = 0.0) -> list[np.ndarray]:
    """
    Convert pixel values back to float weights using per-frame center/norm.
    Inverse: weight = (pixel - max_val/2) / norm_factor + center
    """
    max_val = (1 << bit_depth) - 1
    half = max_val / 2.0

    results = []
    for i, (p, norms) in enumerate(zip(pixels, per_frame_norms)):
        center = norms["center"]
        norm_factor = norms["norm_factor"]
        p_f = p.astype(np.float64)
        if dither > 0:
            p_f = p_f - _dither_noise(p.shape, frame_index=i, amplitude=dither)
        val = (p_f - half) / norm_factor + center
        results.append(val)
    return results


def _reassemble_2d(tiles: list[dict]) -> torch.Tensor:
    """Reassemble decoded tiles into a 2D tensor at orig_shape dimensions."""
    first = tiles[0]
    img_h, img_w = first["img_shape"]

    img = np.zeros((img_h, img_w), dtype=np.float64)

    tile_h = tiles[0]["frame"].shape[0]
    tile_w = tiles[0]["frame"].shape[1]

    for t in tiles:
        r, c = t["tile_row"], t["tile_col"]
        row_start, col_start = r * tile_h, c * tile_w
        row_end = min(row_start + tile_h, img_h)
        col_end = min(col_start + tile_w, img_w)
        src_h, src_w = row_end - row_start, col_end - col_start
        img[row_start:row_end, col_start:col_end] = t["frame"][:src_h, :src_w]

    return torch.from_numpy(img.astype(np.float32))


def reassemble_spatial_dct(tiles: list[dict]) -> torch.Tensor:
    """Reassemble tiles into a 4D spatial DCT weight (out, in, K_h, K_w)."""
    first = tiles[0]
    out_ch, in_ch, K_h, K_w = first["orig_shape"]
    tensor_2d = _reassemble_2d(tiles)
    return tensor_2d.reshape(out_ch, K_h, in_ch, K_w).permute(0, 2, 1, 3).contiguous()


def reassemble_2d(tiles: list[dict]) -> torch.Tensor:
    """Reassemble tiles into a 2D tensor (channel_dct, fc, fc_bias, bn)."""
    return _reassemble_2d(tiles)


def _sort_by_similarity(entries: list[dict]) -> list[dict]:
    """
    Sort frames by similarity using greedy nearest-neighbor ordering.
    Places similar frames adjacent for better H.265 inter-frame prediction.
    """
    if len(entries) <= 2:
        return entries

    flat = [e["frame"].astype(np.float32).ravel() for e in entries]
    n = len(entries)
    visited = [False] * n
    order = [0]
    visited[0] = True

    for _ in range(n - 1):
        last = order[-1]
        best_idx, best_dist = -1, float("inf")
        last_flat = flat[last]
        for j in range(n):
            if visited[j]:
                continue
            diff = last_flat - flat[j]
            dist = np.dot(diff, diff) / len(diff)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        order.append(best_idx)
        visited[best_idx] = True

    return [entries[i] for i in order]


# ---------- Decode ----------

def decode_main(args):
    """Decode H.265 videos back to model weights, load into model, evaluate."""
    manifest_path = os.path.join(args.h265_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    arch = manifest["arch"]
    bit_depth = manifest["bit_depth"]
    dither = manifest.get("dither", 0.0)
    use_yuv = manifest.get("yuv", False)

    dither_str = f", dither={dither}" if dither > 0 else ""
    yuv_str = ", yuv420" if use_yuv else ""
    print(f"Decoding H.265 model: arch={arch}, bit_depth={bit_depth}{dither_str}{yuv_str}")

    # Decode all videos and denormalize back to float weights
    layer_tiles = {}

    for video_name, vinfo in manifest["videos"].items():
        video_path = os.path.join(args.h265_dir, video_name)
        w, h, n = vinfo["frame_width"], vinfo["frame_height"], vinfo["n_frames"]
        vbit = vinfo.get("bit_depth", bit_depth)

        per_frame_norms = [
            {"center": f["center"], "norm_factor": f["norm_factor"]}
            for f in vinfo["frames"]
        ]

        print(f"  Decoding {video_name}: {n} frames @ {w}x{h}...")
        pixels = decode_h265_frames(video_path, n, w, h, vbit, yuv=use_yuv)
        values = denormalize_frames(pixels, per_frame_norms, vbit, dither=dither)

        for fi, finfo in enumerate(vinfo["frames"]):
            layer_name = finfo["layer_name"]
            if layer_name not in layer_tiles:
                layer_tiles[layer_name] = {"tiles": [], "layer_type": finfo.get("layer_type", "spatial_dct")}
            layer_tiles[layer_name]["tiles"].append({
                "frame": values[fi],
                "orig_shape": tuple(finfo["orig_shape"]),
                "img_shape": tuple(finfo["img_shape"]),
                "tile_row": finfo["tile_row"],
                "tile_col": finfo["tile_col"],
                "n_tile_rows": finfo["n_tile_rows"],
                "n_tile_cols": finfo["n_tile_cols"],
            })

    # Build model — all weights will come from H.265 data
    model_fn = getattr(models, arch)
    model = model_fn(weights=None)
    replace_with_dct_convs(model)

    # Reassemble ALL weights from decoded tiles
    for name, m in model.named_modules():
        if isinstance(m, DCTConv2d) and name in layer_tiles:
            weight = reassemble_spatial_dct(layer_tiles[name]["tiles"])
            m.weight_dct.data.copy_(weight)
            print(f"  Loaded {name} (spatial_dct): {list(weight.shape)}")

        elif isinstance(m, ChannelDCTConv1x1) and name in layer_tiles:
            weight = reassemble_2d(layer_tiles[name]["tiles"])
            m.weight_dct.data.copy_(weight)
            print(f"  Loaded {name} (channel_dct): {list(weight.shape)}")

        elif isinstance(m, nn.BatchNorm2d) and name in layer_tiles:
            img = reassemble_2d(layer_tiles[name]["tiles"])
            m.weight.data.copy_(img[0])
            m.bias.data.copy_(img[1])
            m.running_mean.copy_(img[2])
            # Clamp running_var to non-negative: 8-bit roundtrip can push
            # near-zero variances slightly negative, causing nan in sqrt
            m.running_var.copy_(img[3].clamp(min=0.0))
            print(f"  Loaded {name} (bn): {list(img.shape)}")

        elif isinstance(m, nn.Linear):
            wkey = name + ".weight"
            bkey = name + ".bias"
            if wkey in layer_tiles:
                weight = reassemble_2d(layer_tiles[wkey]["tiles"])
                m.weight.data.copy_(weight)
                print(f"  Loaded {wkey} (fc): {list(weight.shape)}")
            if bkey in layer_tiles and m.bias is not None:
                bias = reassemble_2d(layer_tiles[bkey]["tiles"])
                m.bias.data.copy_(bias.squeeze(0))
                print(f"  Loaded {bkey} (fc_bias): {list(bias.shape)}")

    # Evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    valdir = os.path.join(args.data, "val")
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    )

    # Remap subset dataset labels to ImageNet indices (same as training)
    IMAGENET_WNID_TO_IDX = {
        "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
        "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
        "n03445777": 574, "n03888257": 701,
    }
    num_classes = len(val_dataset.classes)
    if num_classes < 1000:
        label_map = {}
        for wnid, local_idx in val_dataset.class_to_idx.items():
            if wnid in IMAGENET_WNID_TO_IDX:
                label_map[local_idx] = IMAGENET_WNID_TO_IDX[wnid]
        if len(label_map) == num_classes:
            print(f"  Remapping {num_classes} labels to ImageNet indices")
            val_dataset.targets = [label_map[t] for t in val_dataset.targets]
            val_dataset.samples = [(p, label_map[t]) for p, t in val_dataset.samples]
            val_dataset.imgs = val_dataset.samples

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss().to(device)

    print(f"\nEvaluating decoded model on {valdir}...")
    top1_sum = top5_sum = loss_sum = 0.0
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

    print(f"  (After H.265 decode: Acc@1 {acc1:.2f}%)")

    # Sizes
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


# ---------- Encode ----------

PROFILE_PRESETS = [
    "ultrafast", "superfast", "veryfast", "faster",
    "fast", "medium", "slow", "slower", "veryslow",
]


def _module_to_2d(name, m):
    """
    Convert any module's parameters to a list of (2D image, metadata) entries.
    Returns list of dicts with keys: name, img, orig_shape, img_shape, layer_type.
    """
    entries = []

    if isinstance(m, DCTConv2d):
        img = layer_to_2d(m.weight_dct)
        entries.append({
            "name": name, "img": img,
            "orig_shape": list(m.weight_dct.shape),
            "img_shape": list(img.shape),
            "layer_type": "spatial_dct",
        })

    elif isinstance(m, ChannelDCTConv1x1):
        img = m.weight_dct.detach().float().numpy()  # already 2D
        entries.append({
            "name": name, "img": img,
            "orig_shape": list(m.weight_dct.shape),
            "img_shape": list(img.shape),
            "layer_type": "channel_dct",
        })

    elif isinstance(m, nn.BatchNorm2d):
        # Stack [weight, bias, running_mean, running_var] as [4, features]
        img = torch.stack([
            m.weight.data, m.bias.data, m.running_mean, m.running_var
        ], dim=0).float().numpy()
        entries.append({
            "name": name, "img": img,
            "orig_shape": [4, m.num_features],
            "img_shape": list(img.shape),
            "layer_type": "bn",
        })

    elif isinstance(m, nn.Linear):
        img = m.weight.detach().float().numpy()
        entries.append({
            "name": name + ".weight", "img": img,
            "orig_shape": list(m.weight.shape),
            "img_shape": list(img.shape),
            "layer_type": "fc",
        })
        if m.bias is not None:
            bias_img = m.bias.detach().float().unsqueeze(0).numpy()
            entries.append({
                "name": name + ".bias", "img": bias_img,
                "orig_shape": [1, m.bias.shape[0]],
                "img_shape": list(bias_img.shape),
                "layer_type": "fc_bias",
            })

    return entries


def _load_model_for_encode(args):
    """Load model and prepare tile groups for ALL layers."""
    model_fn = getattr(models, args.arch)
    model = model_fn(weights=None)
    replace_with_dct_convs(model)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"No model found at {args.model}")

    print(f"Loading model from {args.model}")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state)

    # Convert ALL layers to 2D float images
    layer_images = []
    full_model_bytes = 0

    for name, m in model.named_modules():
        entries = _module_to_2d(name, m)
        for e in entries:
            layer_images.append(e)
            full_model_bytes += e["img"].size * 4  # float32
            print(f"  {e['name']}: {e['orig_shape']} -> "
                  f"2D {e['img_shape'][0]}x{e['img_shape'][1]}  "
                  f"({e['layer_type']})")

    # Build tiles, group by (height, width, is_bn)
    # BN layers get their own groups so they can be encoded with different settings
    tile_groups = {}
    for li in layer_images:
        img = li["img"]
        h, w = img.shape
        is_bn = li["layer_type"] == "bn"

        if h <= MAX_TILE and w <= MAX_TILE:
            frame = pad_to_min(img)
            fh, fw = frame.shape
            key = (fh, fw, is_bn)
            if key not in tile_groups:
                tile_groups[key] = []
            tile_groups[key].append({
                "frame": frame,
                "layer_name": li["name"],
                "orig_shape": li["orig_shape"],
                "img_shape": li["img_shape"],
                "layer_type": li["layer_type"],
                "tile_row": 0, "tile_col": 0,
                "n_tile_rows": 1, "n_tile_cols": 1,
            })
        else:
            n_tile_rows = (h + MAX_TILE - 1) // MAX_TILE
            n_tile_cols = (w + MAX_TILE - 1) // MAX_TILE
            tile_h = (h + n_tile_rows - 1) // n_tile_rows
            tile_w = (w + n_tile_cols - 1) // n_tile_cols
            tile_h = max(MIN_DIM, tile_h + (tile_h % 2))
            tile_w = max(MIN_DIM, tile_w + (tile_w % 2))

            tiles = slice_to_tiles(img, tile_h, tile_w)
            key = (tile_h, tile_w, is_bn)
            if key not in tile_groups:
                tile_groups[key] = []
            for tile, row_idx, col_idx in tiles:
                tile = circular_pad(tile, tile_h, tile_w)
                tile_groups[key].append({
                    "frame": tile,
                    "layer_name": li["name"],
                    "orig_shape": li["orig_shape"],
                    "img_shape": li["img_shape"],
                    "layer_type": li["layer_type"],
                    "tile_row": row_idx, "tile_col": col_idx,
                    "n_tile_rows": n_tile_rows, "n_tile_cols": n_tile_cols,
                })

    return model, tile_groups, full_model_bytes


def _encode_tile_groups(tile_groups: dict, output_dir: str, crf: int,
                        preset: str, bit_depth: int, dither: float = 0.0,
                        yuv: bool = False, verbose: bool = True,
                        bn_crf: int = 0, bn_bit_depth: int = 12):
    """Encode all tile groups to H.265 videos.
    BN groups (keyed with is_bn=True) use bn_crf and bn_bit_depth."""
    import time as _time

    total_raw_bytes = 0
    total_h265_bytes = 0
    manifest_videos = {}
    t0 = _time.monotonic()

    for (th, tw, is_bn), entries in sorted(tile_groups.items()):
        tag = "bn" if is_bn else "dct"
        video_name = f"{tag}_{th}x{tw}.hevc"
        use_crf = bn_crf if is_bn else crf
        use_bit_depth = bn_bit_depth if is_bn else bit_depth
        video_path = os.path.join(output_dir, video_name)

        # Sort frames by similarity for better inter-frame prediction
        entries = _sort_by_similarity(entries)

        frames = [e["frame"] for e in entries]
        raw_bytes = sum(f.size * (use_bit_depth // 8) for f in frames)

        ok, per_frame_norms, actual_bit_depth = encode_frames_to_h265(
            frames, video_path,
            crf=use_crf, preset=preset, bit_depth=use_bit_depth,
            dither=dither, yuv=yuv,
        )

        if ok:
            h265_bytes = os.path.getsize(video_path)
            raw_bytes = sum(f.size * (actual_bit_depth // 8) for f in frames)
            ratio = raw_bytes / max(h265_bytes, 1)
            if verbose:
                print(f"  {video_name}: {len(frames)} frames @ {tw}x{th}  "
                      f"raw={raw_bytes/1024:.1f} KB  "
                      f"h265={h265_bytes/1024:.1f} KB  "
                      f"ratio={ratio:.1f}x ({actual_bit_depth}bit)")
            total_raw_bytes += raw_bytes
            total_h265_bytes += h265_bytes

            manifest_videos[video_name] = {
                "frame_width": tw, "frame_height": th,
                "bit_depth": actual_bit_depth, "n_frames": len(frames),
                "frames": [],
            }
            for i, e in enumerate(entries):
                manifest_videos[video_name]["frames"].append({
                    "frame_index": i,
                    "layer_name": e["layer_name"],
                    "layer_type": e["layer_type"],
                    "orig_shape": list(e["orig_shape"]),
                    "img_shape": list(e["img_shape"]),
                    "tile_row": e["tile_row"],
                    "tile_col": e["tile_col"],
                    "n_tile_rows": e["n_tile_rows"],
                    "n_tile_cols": e["n_tile_cols"],
                    "center": per_frame_norms[i]["center"],
                    "norm_factor": per_frame_norms[i]["norm_factor"],
                })

    elapsed = _time.monotonic() - t0
    return total_raw_bytes, total_h265_bytes, manifest_videos, elapsed


def encode_main(args):
    """Encode DCT coefficients to H.265 videos."""
    os.makedirs(args.output_dir, exist_ok=True)

    model, tile_groups, full_model_bytes = _load_model_for_encode(args)

    if args.profile:
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
                dither=args.dither, yuv=args.yuv, verbose=False,
                bn_crf=args.bn_crf, bn_bit_depth=args.bn_bit_depth,
            )
            ratio_f32 = full_model_bytes / max(total_h265, 1)
            ratio_raw = total_raw / max(total_h265, 1)
            results.append({
                "preset": preset, "h265_bytes": total_h265,
                "time_s": elapsed, "ratio_vs_f32": ratio_f32,
                "ratio_vs_raw": ratio_raw,
            })
            print(f"  H.265: {total_h265/1024:.1f} KB  "
                  f"vs f32: {ratio_f32:.1f}x  "
                  f"vs raw: {ratio_raw:.1f}x  "
                  f"time: {elapsed:.1f}s")

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
        dither=args.dither, yuv=args.yuv,
        bn_crf=args.bn_crf, bn_bit_depth=args.bn_bit_depth,
    )

    manifest = {
        "arch": args.arch,
        "bit_depth": args.bit_depth,
        "dither": args.dither,
        "yuv": args.yuv,
        "reconstruction": "weight = (pixel - max_val/2) / norm_factor + center",
        "videos": manifest_videos,
    }

    print(f"\n--- Summary (preset={args.preset}, {elapsed:.1f}s) ---")
    print(f"Full model DCT weights (float32): {full_model_bytes/1024:.1f} KB")
    print(f"Pixel frames raw:                 {total_raw_bytes/1024:.1f} KB")
    print(f"H.265 encoded:                    {total_h265_bytes/1024:.1f} KB")
    if total_h265_bytes > 0:
        print(f"Compression vs float32:           {full_model_bytes/total_h265_bytes:.1f}x")
        print(f"Compression vs pixel raw:         {total_raw_bytes/total_h265_bytes:.1f}x")

    manifest["summary"] = {
        "full_model_float32_bytes": full_model_bytes,
        "pixel_raw_bytes": total_raw_bytes,
        "h265_encoded_bytes": total_h265_bytes,
    }

    print(f"\nTotal compressed model (H.265 only): {total_h265_bytes/1024:.1f} KB")
    print(f"Compression ratio:                   {full_model_bytes/max(total_h265_bytes,1):.1f}x")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


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
