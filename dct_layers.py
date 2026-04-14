import torch
import torch.nn as nn
import torch.nn.functional as F
from dct_utils import get_1d_dct_matrix, get_dct_matrix


# Module-level training config, set by the training script
class DCTConfig:
    qstep: float = 0.1
    dct_dropout: float = 0.0
    train_noise: bool = False


dct_config = DCTConfig()


def _apply_train_noise_and_dropout(weight_dct: torch.Tensor, training: bool) -> torch.Tensor:
    """Apply training-time uniform noise and DCT dropout to DCT coefficients."""
    if not training:
        return weight_dct

    w = weight_dct

    # Training-time uniform noise (ECRF): Φ(θ) = θ + u * qstep, u ~ U(-½, ½)
    if dct_config.train_noise and dct_config.qstep > 0:
        noise = torch.empty_like(w).uniform_(-0.5, 0.5) * dct_config.qstep
        w = w + noise

    # DCT dropout (DCT-Conv): randomly zero coefficients, scale survivors
    if dct_config.dct_dropout > 0:
        mask = torch.bernoulli(torch.full_like(w, 1.0 - dct_config.dct_dropout))
        w = w * mask / (1.0 - dct_config.dct_dropout)

    return w


class DCTConv2d(nn.Module):
    """
    Convolutional layer whose learnable parameters live in the DCT (frequency)
    domain.  The forward pass materializes spatial weights via 2D IDCT before
    running a standard conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        K_h, K_w = kernel_size

        self.weight_dct = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, K_h, K_w)
        )
        nn.init.kaiming_uniform_(self.weight_dct, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        C_h = get_1d_dct_matrix(K_h)
        C_w = get_1d_dct_matrix(K_w)
        self.register_buffer("C_h_T", C_h.t())
        self.register_buffer("C_w", C_w)

        self._quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _apply_train_noise_and_dropout(self.weight_dct, self.training)

        spatial_weight = torch.einsum(
            "ab, oibd, cd -> oiac", self.C_h_T, w, self.C_w
        )

        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                spatial_weight, self.bias, self.stride,
                (0, 0), self.dilation, self.groups,
            )

        return F.conv2d(
            x, spatial_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )

    @torch.no_grad()
    def quantize(self, qstep: float = 1.0):
        levels = torch.round(self.weight_dct.data / qstep)
        total = levels.numel()
        nonzero = (levels != 0).sum().item()
        self.weight_dct.data = levels * qstep
        self._quantized = True
        return {"total_coeffs": total, "nonzero_coeffs": nonzero,
                "sparsity": 1.0 - nonzero / total}

    def get_sparse_coefficients(self, qstep: float = 1.0):
        with torch.no_grad():
            levels = torch.round(self.weight_dct.data / qstep).to(torch.int16)
            nonzero_mask = levels != 0
            indices = nonzero_mask.nonzero(as_tuple=False)
            values = levels[nonzero_mask]
        return indices, values

    def extra_repr(self) -> str:
        q_str = ", quantized" if self._quantized else ""
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}{q_str}"
        )


class ChannelDCTConv1x1(nn.Module):
    """
    1x1 convolution with channel-wise DCT reparameterization.

    Supports optional block DCT: instead of one large N×N DCT, applies
    B×B DCT in blocks. Set block_size=0 for full (non-block) DCT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride=1,
        padding=0,
        groups: int = 1,
        bias: bool = True,
        block_size: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_ch_per_group = in_channels // groups
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.groups = groups
        self.block_size = block_size

        self.weight_dct = nn.Parameter(
            torch.empty(out_channels, self.in_ch_per_group)
        )
        nn.init.kaiming_uniform_(self.weight_dct, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def _idct_full(self, w, device, dtype):
        C_out = get_dct_matrix(self.out_channels, device, dtype)
        C_in = get_dct_matrix(self.in_ch_per_group, device, dtype)
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
        spatial = torch.einsum("ab, hbwc, cd -> hawc", C_B.t(), blocks, C_B)
        return spatial.reshape(Hp, Wp)[:H, :W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _apply_train_noise_and_dropout(self.weight_dct, self.training)

        if self.block_size > 0:
            spatial_2d = self._idct_block(w, x.device, x.dtype)
        else:
            spatial_2d = self._idct_full(w, x.device, x.dtype)

        spatial_weight = spatial_2d.unsqueeze(-1).unsqueeze(-1)

        return F.conv2d(
            x, spatial_weight, self.bias, self.stride,
            self.padding, (1, 1), self.groups,
        )

    @torch.no_grad()
    def quantize(self, qstep: float = 1.0):
        levels = torch.round(self.weight_dct.data / qstep)
        total = levels.numel()
        nonzero = (levels != 0).sum().item()
        self.weight_dct.data = levels * qstep
        return {"total_coeffs": total, "nonzero_coeffs": nonzero,
                "sparsity": 1.0 - nonzero / total}

    def get_sparse_coefficients(self, qstep: float = 1.0):
        with torch.no_grad():
            levels = torch.round(self.weight_dct.data / qstep).to(torch.int16)
            nonzero_mask = levels != 0
            indices = nonzero_mask.nonzero(as_tuple=False)
            values = levels[nonzero_mask]
        return indices, values

    def extra_repr(self) -> str:
        bs = f", block_size={self.block_size}" if self.block_size > 0 else ""
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"stride={self.stride}, padding={self.padding}, "
            f"groups={self.groups}, bias={self.bias is not None}{bs}"
        )


def replace_with_dct_convs(module: nn.Module, block_size: int = 0) -> None:
    """
    Recursively replace all nn.Conv2d layers with DCT variants:
    - kernel > 1x1 → DCTConv2d (spatial DCT)
    - kernel == 1x1 → ChannelDCTConv1x1 (channel-wise DCT)
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size == (1, 1):
                ch_dct = ChannelDCTConv1x1(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    stride=child.stride,
                    padding=child.padding,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    block_size=block_size,
                )
                # Initialize from pretrained weights via forward DCT
                if child.weight is not None:
                    with torch.no_grad():
                        w_2d = child.weight.data.squeeze(-1).squeeze(-1)
                        C_out = get_1d_dct_matrix(child.out_channels).to(w_2d)
                        C_in = get_1d_dct_matrix(child.in_channels // child.groups).to(w_2d)
                        ch_dct.weight_dct.data.copy_(C_out @ w_2d @ C_in.t())
                if child.bias is not None:
                    ch_dct.bias.data.copy_(child.bias.data)
                setattr(module, name, ch_dct)
            else:
                dct_conv = DCTConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode,
                )
                # Initialize from pretrained weights via forward DCT
                # forward() computes C_h^T @ weight_dct @ C^T (IDCT)
                # so weight_dct = C_h @ weight @ C_w
                if child.weight is not None:
                    with torch.no_grad():
                        w = child.weight.data  # (out, in, K_h, K_w)
                        C_h = dct_conv.C_h_T.t()  # C_h
                        C_w = dct_conv.C_w          # C_w
                        dct_conv.weight_dct.data.copy_(
                            torch.einsum("ab, oibc, cd -> oiad", C_h, w, C_w)
                        )
                if child.bias is not None:
                    dct_conv.bias.data.copy_(child.bias.data)
                setattr(module, name, dct_conv)
        else:
            replace_with_dct_convs(child, block_size=block_size)


def _is_dct_layer(m):
    """Check if a module is any DCT-reparameterized layer."""
    return isinstance(m, (DCTConv2d, ChannelDCTConv1x1))


def probe_sparsity(model: nn.Module, qstep: float = 1.0):
    """Non-destructive: count coefficients that would survive quantization."""
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


def quantize_model(model: nn.Module, qstep: float = 1.0):
    """Quantize all DCT layers in-place. Returns aggregate stats."""
    total_coeffs = 0
    total_nonzero = 0
    layer_stats = []
    for name, m in model.named_modules():
        if _is_dct_layer(m):
            stats = m.quantize(qstep)
            stats["name"] = name
            layer_stats.append(stats)
            total_coeffs += stats["total_coeffs"]
            total_nonzero += stats["nonzero_coeffs"]
    return {
        "total_coeffs": total_coeffs,
        "nonzero_coeffs": total_nonzero,
        "overall_sparsity": 1.0 - total_nonzero / max(total_coeffs, 1),
        "layers": layer_stats,
    }


def export_sparse_coefficients(model: nn.Module, qstep: float = 1.0):
    """Export all DCT layers as sparse coefficient maps."""
    result = {}
    for name, m in model.named_modules():
        if _is_dct_layer(m):
            indices, values = m.get_sparse_coefficients(qstep)
            result[name] = {"indices": indices, "values": values}
    return result
