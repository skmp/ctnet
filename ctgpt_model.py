"""
CTGPT: DCT-compressed GPT-2 model.

Replaces nn.Linear layers with DCTLinear (channel-wise block DCT),
reusing the same compression infrastructure as CTNet for convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dct_utils import get_1d_dct_matrix, get_dct_matrix
from dct_layers import (
    DCTConfig, dct_config,
    _apply_train_noise_and_dropout, _simulate_pixel_quantization,
)


class DCTLinear(nn.Module):
    """
    Linear layer with channel-wise DCT reparameterization.

    Learnable parameters are DCT coefficients of shape (out_features, in_features).
    Forward: 2D IDCT → F.linear. Supports block DCT for large matrices.

    Directly analogous to ChannelDCTConv1x1 from CTNet but for nn.Linear.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, block_size: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.weight_dct = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight_dct, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def _idct_full(self, w, device, dtype):
        C_out = get_dct_matrix(self.out_features, device, dtype)
        C_in = get_dct_matrix(self.in_features, device, dtype)
        return C_out.t() @ w @ C_in

    def _idct_block(self, w, device, dtype):
        B = self.block_size
        H, W = w.shape
        pad_h = (B - H % B) % B
        pad_w = (B - W % B) % B
        if pad_h > 0 or pad_w > 0:
            w = F.pad(w, (0, pad_w, 0, pad_h))
        Hp, Wp = w.shape
        C_B = get_dct_matrix(B, device, dtype)
        blocks = w.reshape(Hp // B, B, Wp // B, B)
        # IDCT per block: C^T @ dct_block @ C^T
        spatial = torch.einsum("ab, hbwc, cd -> hawd", C_B.t(), blocks, C_B.t())
        return spatial.reshape(Hp, Wp)[:H, :W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _apply_train_noise_and_dropout(self.weight_dct, self.training)

        if self.block_size > 0:
            spatial_weight = self._idct_block(w, x.device, x.dtype)
        else:
            spatial_weight = self._idct_full(w, x.device, x.dtype)

        return F.linear(x, spatial_weight, self.bias)

    def extra_repr(self) -> str:
        bs = f", block_size={self.block_size}" if self.block_size > 0 else ""
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}{bs}")


def _is_dct_layer(m):
    return isinstance(m, DCTLinear)


def _block_forward_dct(w: torch.Tensor, block_size: int) -> torch.Tensor:
    """Compute block forward DCT matching _idct_block's inverse."""
    B = block_size
    H, W = w.shape
    pad_h = (B - H % B) % B
    pad_w = (B - W % B) % B
    if pad_h > 0 or pad_w > 0:
        w = F.pad(w, (0, pad_w, 0, pad_h))
    Hp, Wp = w.shape
    C_B = get_1d_dct_matrix(B).to(w)
    blocks = w.reshape(Hp // B, B, Wp // B, B)
    # Forward DCT per block: C @ block @ C
    dct_blocks = torch.einsum("ab, hbwc, cd -> hawd", C_B, blocks, C_B)
    result = dct_blocks.reshape(Hp, Wp)
    return result[:H, :W]


def _get_linear_weight_and_bias(child):
    """
    Extract weight and bias from nn.Linear or HuggingFace Conv1D.
    Conv1D stores weight as (in_features, out_features) — transposed vs nn.Linear.
    Returns (weight, bias) where weight is (out_features, in_features).
    """
    try:
        from transformers.pytorch_utils import Conv1D
        if isinstance(child, Conv1D):
            # Conv1D weight is (in_features, out_features), need to transpose
            w = child.weight.data.t()  # → (out_features, in_features)
            out_features, in_features = w.shape
            bias = child.bias.data if child.bias is not None else None
            return w, bias, in_features, out_features, child.bias is not None
    except ImportError:
        pass

    if isinstance(child, nn.Linear):
        w = child.weight.data  # (out_features, in_features)
        return w, child.bias.data if child.bias is not None else None, \
               child.in_features, child.out_features, child.bias is not None

    return None, None, None, None, False


def _is_replaceable_linear(child):
    """Check if module is nn.Linear or HuggingFace Conv1D."""
    if isinstance(child, nn.Linear):
        return True
    try:
        from transformers.pytorch_utils import Conv1D
        return isinstance(child, Conv1D)
    except ImportError:
        return False


def replace_linears_with_dct(module: nn.Module, block_size: int = 16,
                              skip_names: set = None) -> None:
    """
    Recursively replace nn.Linear / HuggingFace Conv1D layers with DCTLinear.

    Args:
        module: model to modify in-place
        block_size: DCT block size (0 = full DCT, 16 = recommended)
        skip_names: set of child names to skip
    """
    if skip_names is None:
        skip_names = set()

    for name, child in module.named_children():
        if name in skip_names:
            continue

        if _is_replaceable_linear(child):
            w, bias_data, in_f, out_f, has_bias = _get_linear_weight_and_bias(child)
            if w is None:
                continue

            dct_linear = DCTLinear(
                in_features=in_f,
                out_features=out_f,
                bias=has_bias,
                block_size=block_size,
            )
            # Initialize from pretrained weights via forward DCT
            with torch.no_grad():
                if block_size > 0:
                    # Block forward DCT: must match block IDCT in forward()
                    dct_linear.weight_dct.data.copy_(
                        _block_forward_dct(w, block_size)
                    )
                else:
                    C_out = get_1d_dct_matrix(out_f).to(w)
                    C_in = get_1d_dct_matrix(in_f).to(w)
                    dct_linear.weight_dct.data.copy_(C_out @ w @ C_in.t())
            if bias_data is not None:
                dct_linear.bias.data.copy_(bias_data)
            setattr(module, name, dct_linear)
        else:
            replace_linears_with_dct(child, block_size=block_size,
                                     skip_names=skip_names)


def probe_sparsity(model: nn.Module, qstep: float = 1.0):
    """Non-destructive: count DCT coefficients that would survive quantization."""
    total = 0
    nonzero = 0
    with torch.no_grad():
        for m in model.modules():
            if _is_dct_layer(m):
                levels = torch.round(m.weight_dct.data / qstep)
                total += levels.numel()
                nonzero += (levels != 0).sum().item()
    sparsity = 1.0 - nonzero / max(total, 1)
    return total, nonzero, sparsity
