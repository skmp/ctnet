"""
CTGPT export: Encode/decode GPT-2 DCT weights as H.265 video.

Usage:
    python ctgpt_export.py encode [--model ctgpt_checkpoints/best.pt]
    python ctgpt_export.py decode [--h265-dir ctgpt_h265]
"""

import argparse
import os
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from ctgpt_model import DCTLinear, _is_dct_layer, replace_linears_with_dct
from export_h265 import (
    encode_frames_to_h265, decode_h265_frames, denormalize_frames,
    normalize_frame, _sort_by_similarity, _reassemble_2d,
    pad_to_min, circular_pad, slice_to_tiles,
    MIN_DIM, MAX_TILE,
)


def parse_args():
    p = argparse.ArgumentParser(description="CTGPT H.265 export/decode")
    sub = p.add_subparsers(dest="command")

    enc = sub.add_parser("encode")
    enc.add_argument("--model", default="./ctgpt_checkpoints/best.pt")
    enc.add_argument("--output-dir", default="./ctgpt_h265")
    enc.add_argument("--dct-block-size", default=16, type=int)
    enc.add_argument("--crf", default=0, type=int)
    enc.add_argument("--bit-depth", default=8, type=int, choices=[8, 10, 12])
    enc.add_argument("--preset", default="slower")
    enc.add_argument("--dither", default=0.0, type=float)

    dec = sub.add_parser("decode")
    dec.add_argument("--h265-dir", default="./ctgpt_h265")
    dec.add_argument("--prompt", default="ROMEO:", type=str)
    dec.add_argument("--max-tokens", default=500, type=int)

    return p.parse_args()


def _module_to_2d(name, m):
    """Convert module parameters to 2D images for H.265 encoding."""
    entries = []

    if isinstance(m, DCTLinear):
        img = m.weight_dct.detach().float().numpy()
        entries.append({
            "name": name, "img": img,
            "orig_shape": list(m.weight_dct.shape),
            "img_shape": list(img.shape),
            "layer_type": "dct_linear",
        })

    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        # Stack [weight, bias] as [2, normalized_shape]
        tensors = [m.weight.data, m.bias.data]
        img = torch.stack(tensors, dim=0).float().numpy()
        entries.append({
            "name": name, "img": img,
            "orig_shape": [2, m.normalized_shape[0]],
            "img_shape": list(img.shape),
            "layer_type": "ln",
        })

    elif isinstance(m, nn.Embedding):
        img = m.weight.detach().float().numpy()
        entries.append({
            "name": name, "img": img,
            "orig_shape": list(m.weight.shape),
            "img_shape": list(img.shape),
            "layer_type": "embedding",
        })

    elif isinstance(m, nn.Linear):
        # Standard linear (e.g., lm_head)
        img = m.weight.detach().float().numpy()
        entries.append({
            "name": name + ".weight", "img": img,
            "orig_shape": list(m.weight.shape),
            "img_shape": list(img.shape),
            "layer_type": "linear",
        })
        if m.bias is not None:
            bias_img = m.bias.detach().float().unsqueeze(0).numpy()
            entries.append({
                "name": name + ".bias", "img": bias_img,
                "orig_shape": [1, m.bias.shape[0]],
                "img_shape": list(bias_img.shape),
                "layer_type": "linear_bias",
            })

    return entries


def encode_main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading GPT-2 and checkpoint...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    replace_linears_with_dct(model.transformer, block_size=args.dct_block_size)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Collect all layers as 2D images
    layer_images = []
    ln_state = {}  # LayerNorm stored as raw tensors
    full_model_bytes = 0

    for name, m in model.named_modules():
        entries = _module_to_2d(name, m)
        for e in entries:
            if e["layer_type"] == "ln":
                # Store LayerNorm as raw (like BN in CTNet)
                ln_state[e["name"]] = torch.from_numpy(e["img"])
                full_model_bytes += e["img"].size * 4
                print(f"  {e['name']}: {e['orig_shape']} (ln, raw)")
            else:
                layer_images.append(e)
                full_model_bytes += e["img"].size * 4
                print(f"  {e['name']}: {e['orig_shape']} -> "
                      f"2D {e['img_shape'][0]}x{e['img_shape'][1]} ({e['layer_type']})")

    # Build tiles
    tile_groups = {}
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
                "frame": frame, "layer_name": li["name"],
                "orig_shape": li["orig_shape"], "img_shape": li["img_shape"],
                "layer_type": li["layer_type"],
                "tile_row": 0, "tile_col": 0,
                "n_tile_rows": 1, "n_tile_cols": 1,
            })
        else:
            n_tr = (h + MAX_TILE - 1) // MAX_TILE
            n_tc = (w + MAX_TILE - 1) // MAX_TILE
            th = (h + n_tr - 1) // n_tr
            tw = (w + n_tc - 1) // n_tc
            th = max(MIN_DIM, th + (th % 2))
            tw = max(MIN_DIM, tw + (tw % 2))
            tiles = slice_to_tiles(img, th, tw)
            key = (th, tw)
            if key not in tile_groups:
                tile_groups[key] = []
            for tile, row_idx, col_idx in tiles:
                tile = circular_pad(tile, th, tw)
                tile_groups[key].append({
                    "frame": tile, "layer_name": li["name"],
                    "orig_shape": li["orig_shape"], "img_shape": li["img_shape"],
                    "layer_type": li["layer_type"],
                    "tile_row": row_idx, "tile_col": col_idx,
                    "n_tile_rows": n_tr, "n_tile_cols": n_tc,
                })

    # Encode
    import time as _time
    total_h265 = 0
    manifest_videos = {}
    t0 = _time.monotonic()

    print(f"\nEncoding {len(tile_groups)} video(s) at {args.bit_depth}-bit...")
    for (th, tw), entries in sorted(tile_groups.items()):
        video_name = f"gpt_{th}x{tw}.hevc"
        video_path = os.path.join(args.output_dir, video_name)

        entries = _sort_by_similarity(entries)
        frames = [e["frame"] for e in entries]

        if args.crf == 0:
            x265_extra = ":keyint=-1:bframes=16:ref=16:b-adapt=2:rc-lookahead=250"
        else:
            x265_extra = ""

        ok, per_frame_norms, actual_bd = encode_frames_to_h265(
            frames, video_path,
            crf=args.crf, preset=args.preset, bit_depth=args.bit_depth,
            dither=args.dither,
        )

        if ok:
            h265_bytes = os.path.getsize(video_path)
            total_h265 += h265_bytes
            print(f"  {video_name}: {len(frames)} frames @ {tw}x{th}  "
                  f"h265={h265_bytes/1024:.1f} KB ({actual_bd}bit)")

            manifest_videos[video_name] = {
                "frame_width": tw, "frame_height": th,
                "bit_depth": actual_bd, "n_frames": len(frames),
                "frames": [],
            }
            for i, e in enumerate(entries):
                manifest_videos[video_name]["frames"].append({
                    "frame_index": i, "layer_name": e["layer_name"],
                    "layer_type": e["layer_type"],
                    "orig_shape": list(e["orig_shape"]),
                    "img_shape": list(e["img_shape"]),
                    "tile_row": e["tile_row"], "tile_col": e["tile_col"],
                    "n_tile_rows": e["n_tile_rows"], "n_tile_cols": e["n_tile_cols"],
                    "center": per_frame_norms[i]["center"],
                    "norm_factor": per_frame_norms[i]["norm_factor"],
                })

    elapsed = _time.monotonic() - t0

    # Save LN state
    ln_path = os.path.join(args.output_dir, "ln_state.pt")
    torch.save(ln_state, ln_path)
    ln_bytes = os.path.getsize(ln_path)

    total_bytes = total_h265 + ln_bytes

    manifest = {
        "model": "gpt2",
        "dct_block_size": args.dct_block_size,
        "bit_depth": args.bit_depth,
        "dither": args.dither,
        "videos": manifest_videos,
        "summary": {
            "full_model_float32_bytes": full_model_bytes,
            "h265_encoded_bytes": total_h265,
            "ln_raw_bytes": ln_bytes,
            "total_compressed_bytes": total_bytes,
        },
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n--- Summary ({elapsed:.1f}s) ---")
    print(f"Full model (float32):  {full_model_bytes/1024:.1f} KB ({full_model_bytes/1024/1024:.1f} MB)")
    print(f"H.265 encoded:         {total_h265/1024:.1f} KB")
    print(f"LN raw:                {ln_bytes/1024:.1f} KB")
    print(f"Total compressed:      {total_bytes/1024:.1f} KB ({total_bytes/1024/1024:.1f} MB)")
    print(f"Compression ratio:     {full_model_bytes/max(total_bytes,1):.1f}x")
    print(f"Manifest: {manifest_path}")


def decode_main(args):
    manifest_path = os.path.join(args.h265_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    dct_block_size = manifest.get("dct_block_size", 16)
    bit_depth = manifest["bit_depth"]
    dither = manifest.get("dither", 0.0)

    print(f"Decoding CTGPT model: block_size={dct_block_size}, bit_depth={bit_depth}")

    # Decode H.265 videos
    layer_tiles = {}
    for vname, vinfo in manifest["videos"].items():
        video_path = os.path.join(args.h265_dir, vname)
        w, h, n = vinfo["frame_width"], vinfo["frame_height"], vinfo["n_frames"]
        vbit = vinfo.get("bit_depth", bit_depth)

        per_frame_norms = [
            {"center": f["center"], "norm_factor": f["norm_factor"]}
            for f in vinfo["frames"]
        ]

        print(f"  Decoding {vname}: {n} frames @ {w}x{h}...")
        pixels = decode_h265_frames(video_path, n, w, h, vbit)
        values = denormalize_frames(pixels, per_frame_norms, vbit, dither=dither)

        for fi, finfo in enumerate(vinfo["frames"]):
            ln = finfo["layer_name"]
            if ln not in layer_tiles:
                layer_tiles[ln] = {"tiles": [], "layer_type": finfo.get("layer_type", "dct_linear")}
            layer_tiles[ln]["tiles"].append({
                "frame": values[fi],
                "orig_shape": tuple(finfo["orig_shape"]),
                "img_shape": tuple(finfo["img_shape"]),
                "tile_row": finfo["tile_row"], "tile_col": finfo["tile_col"],
                "n_tile_rows": finfo["n_tile_rows"], "n_tile_cols": finfo["n_tile_cols"],
            })

    # Load LN state
    ln_path = os.path.join(args.h265_dir, "ln_state.pt")
    ln_state = {}
    if os.path.isfile(ln_path):
        ln_state = torch.load(ln_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(ln_state)} LayerNorm layers")

    # Build model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    replace_linears_with_dct(model.transformer, block_size=dct_block_size)

    # Load weights
    for name, m in model.named_modules():
        if isinstance(m, DCTLinear) and name in layer_tiles:
            weight = _reassemble_2d(layer_tiles[name]["tiles"])
            m.weight_dct.data.copy_(weight)
            print(f"  Loaded {name} (dct_linear)")

        elif isinstance(m, nn.LayerNorm) and name in ln_state:
            img = ln_state[name]
            m.weight.data.copy_(img[0])
            m.bias.data.copy_(img[1])
            print(f"  Loaded {name} (ln)")

        elif isinstance(m, nn.Embedding) and name in layer_tiles:
            weight = _reassemble_2d(layer_tiles[name]["tiles"])
            m.weight.data.copy_(weight)
            print(f"  Loaded {name} (embedding)")

        elif isinstance(m, nn.Linear) and not isinstance(m, DCTLinear):
            wkey = name + ".weight"
            bkey = name + ".bias"
            if wkey in layer_tiles:
                weight = _reassemble_2d(layer_tiles[wkey]["tiles"])
                m.weight.data.copy_(weight)
                print(f"  Loaded {wkey} (linear)")
            if bkey in layer_tiles and m.bias is not None:
                bias = _reassemble_2d(layer_tiles[bkey]["tiles"])
                m.bias.data.copy_(bias.squeeze(0))
                print(f"  Loaded {bkey} (linear_bias)")

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate sample
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"\nGenerating from prompt: '{args.prompt}'")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=args.max_tokens,
            temperature=0.8, top_k=40, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{text}")

    # File sizes
    total_h265 = sum(
        os.path.getsize(os.path.join(args.h265_dir, vn))
        for vn in manifest["videos"]
    )
    ln_bytes = os.path.getsize(ln_path) if os.path.isfile(ln_path) else 0
    print(f"\n  H.265: {total_h265/1024:.1f} KB")
    print(f"  LN raw: {ln_bytes/1024:.1f} KB")
    print(f"  Total: {(total_h265 + ln_bytes)/1024:.1f} KB")


if __name__ == "__main__":
    args = parse_args()
    if args.command == "encode":
        encode_main(args)
    elif args.command == "decode":
        decode_main(args)
    else:
        print("Usage: python ctgpt_export.py {encode|decode} [options]")
