"""
Tests for DCT infrastructure: roundtrip correctness, block DCT, rate proxy, etc.
"""

import torch
import torch.nn as nn
from dct_utils import get_1d_dct_matrix, get_dct_matrix, calculate_hevc_rate_proxy, _build_channel_freq_weight
from dct_layers import DCTConv2d, ChannelDCTConv1x1, replace_with_dct_convs, _is_dct_layer


def test_dct_matrix_orthogonal():
    """DCT matrix should be orthogonal: C @ C^T = I."""
    for N in [3, 7, 16, 64, 128]:
        C = get_1d_dct_matrix(N)
        eye = C @ C.t()
        err = (eye - torch.eye(N)).abs().max().item()
        assert err < 1e-5, f"DCT({N}) not orthogonal: err={err}"
    print("PASS: DCT matrices are orthogonal")


def test_dct_matrix_cache():
    """Cached DCT matrix should be identical to freshly computed."""
    C1 = get_dct_matrix(64, device=None, dtype=None)
    C2 = get_dct_matrix(64, device=None, dtype=None)
    assert C1 is C2, "Cache miss: same key returned different objects"
    print("PASS: DCT matrix cache works")


def test_full_dct_roundtrip_2d():
    """Full DCT forward+inverse on a 2D matrix should be identity."""
    for H, W in [(3, 3), (7, 7), (16, 32), (768, 768), (768, 2304)]:
        C_h = get_1d_dct_matrix(H)
        C_w = get_1d_dct_matrix(W)
        w = torch.randn(H, W)
        # Forward: C @ w @ C^T ... wait, need to match the IDCT
        # IDCT (full): C^T @ dct @ C (for _idct_full in ChannelDCTConv1x1)
        # So forward: dct = C @ w @ C^T
        # Check: C^T @ (C @ w @ C^T) @ C = (C^T @ C) @ w @ (C^T @ C) = w
        # Wait, _idct_full does: C_out.t() @ w @ C_in
        # So: C_out^T @ dct @ C_in = w => dct = C_out @ w @ C_in^T
        dct = C_h @ w @ C_w.t()
        recon = C_h.t() @ dct @ C_w
        err = (w - recon).abs().max().item()
        assert err < 1e-4, f"Full DCT roundtrip ({H},{W}) failed: err={err}"
    print("PASS: Full 2D DCT roundtrip")


def test_block_dct_roundtrip():
    """Block DCT forward+inverse should be identity for multiples of block size."""
    for B in [4, 8, 16, 32]:
        for H, W in [(B, B), (2*B, 3*B), (4*B, 6*B), (768, 2304)]:
            C_B = get_1d_dct_matrix(B)
            w = torch.randn(H, W)

            # Forward: per-block C @ block @ C
            pad_h = (B - H % B) % B
            pad_w = (B - W % B) % B
            wp = torch.nn.functional.pad(w, (0, pad_w, 0, pad_h))
            Hp, Wp = wp.shape
            blocks = wp.reshape(Hp // B, B, Wp // B, B)
            dct_blocks = torch.einsum("ab, hbwc, cd -> hawd", C_B, blocks, C_B)
            dct_w = dct_blocks.reshape(Hp, Wp)[:H, :W]

            # Inverse: per-block C^T @ dct @ C^T
            pad_h2 = (B - H % B) % B
            pad_w2 = (B - W % B) % B
            dct_p = torch.nn.functional.pad(dct_w, (0, pad_w2, 0, pad_h2))
            Hp2, Wp2 = dct_p.shape
            blocks2 = dct_p.reshape(Hp2 // B, B, Wp2 // B, B)
            inv = torch.einsum("ab, hbwc, cd -> hawd", C_B.t(), blocks2, C_B.t())
            recon = inv.reshape(Hp2, Wp2)[:H, :W]

            err = (w - recon).abs().max().item()
            assert err < 1e-4, f"Block DCT roundtrip B={B} ({H},{W}) failed: err={err}"
    print("PASS: Block DCT roundtrip (all sizes)")


def test_spatial_dct_conv2d_roundtrip():
    """DCTConv2d with pretrained init should produce same output as Conv2d."""
    for K in [3, 5, 7]:
        conv = nn.Conv2d(16, 32, K, padding=K//2, bias=True)
        nn.init.normal_(conv.weight)
        nn.init.normal_(conv.bias)

        x = torch.randn(1, 16, 8, 8)
        y_orig = conv(x)

        # Replace
        module = nn.Sequential()
        module.conv = conv
        replace_with_dct_convs(module, block_size=0)

        y_dct = module.conv(x)
        err = (y_orig - y_dct).abs().max().item()
        assert err < 1e-3, f"DCTConv2d K={K} roundtrip failed: err={err}"
    print("PASS: DCTConv2d pretrained init roundtrip")


def test_channel_dct_1x1_roundtrip():
    """ChannelDCTConv1x1 with pretrained init should produce same output as Conv2d 1x1."""
    for block_size in [0, 8, 16]:
        conv = nn.Conv2d(64, 128, 1, bias=True)
        nn.init.normal_(conv.weight)
        nn.init.normal_(conv.bias)

        x = torch.randn(1, 64, 4, 4)
        y_orig = conv(x)

        module = nn.Sequential()
        module.conv = conv
        replace_with_dct_convs(module, block_size=block_size)

        y_dct = module.conv(x)
        err = (y_orig - y_dct).abs().max().item()
        assert err < 1e-3, f"ChannelDCTConv1x1 block={block_size} roundtrip failed: err={err}"
    print("PASS: ChannelDCTConv1x1 pretrained init roundtrip (all block sizes)")


def test_dct_linear_roundtrip():
    """DCTLinear with pretrained init should produce same output as Linear."""
    from ctgpt_model import DCTLinear, replace_linears_with_dct

    for block_size in [0, 16]:
        linear = nn.Linear(256, 512, bias=True)
        nn.init.normal_(linear.weight)
        nn.init.normal_(linear.bias)

        x = torch.randn(2, 256)
        y_orig = linear(x)

        module = nn.Sequential()
        module.linear = linear
        replace_linears_with_dct(module, block_size=block_size)

        y_dct = module.linear(x)
        err = (y_orig - y_dct).abs().max().item()
        assert err < 1e-3, f"DCTLinear block={block_size} roundtrip failed: err={err}"
    print("PASS: DCTLinear pretrained init roundtrip (all block sizes)")


def test_rate_proxy_2d():
    """Rate proxy should work on 2D tensors."""
    w = torch.randn(64, 128)
    rate = calculate_hevc_rate_proxy(w, qstep=0.1, steepness=10.0)
    assert rate.dim() == 0, "Rate proxy should return scalar"
    assert rate.item() > 0, "Rate should be positive"
    assert rate.requires_grad is False or w.requires_grad, "Grad flow check"
    print(f"PASS: Rate proxy on 2D tensor (rate={rate.item():.0f})")


def test_rate_proxy_4d():
    """Rate proxy should work on 4D tensors."""
    w = torch.randn(32, 16, 3, 3)
    rate = calculate_hevc_rate_proxy(w, qstep=0.1, steepness=10.0)
    assert rate.dim() == 0, "Rate proxy should return scalar"
    assert rate.item() > 0, "Rate should be positive"
    print(f"PASS: Rate proxy on 4D tensor (rate={rate.item():.0f})")


def test_channel_freq_weight():
    """Channel frequency weights should be in [1, 3] range."""
    for H, W in [(8, 8), (64, 128), (768, 2304)]:
        fw = _build_channel_freq_weight(H, W)
        assert fw.shape == (H, W)
        assert fw.min() >= 1.0 - 1e-6
        assert fw.max() <= 3.0 + 1e-6
        assert fw[0, 0] == 1.0, "DC should have weight 1.0"
    print("PASS: Channel frequency weights")


if __name__ == "__main__":
    test_dct_matrix_orthogonal()
    test_dct_matrix_cache()
    test_full_dct_roundtrip_2d()
    test_block_dct_roundtrip()
    test_spatial_dct_conv2d_roundtrip()
    test_channel_dct_1x1_roundtrip()
    test_dct_linear_roundtrip()
    test_rate_proxy_2d()
    test_rate_proxy_4d()
    test_channel_freq_weight()
    print("\n=== ALL TESTS PASSED ===")
