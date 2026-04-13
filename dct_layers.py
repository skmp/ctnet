import torch
import torch.nn as nn
import torch.nn.functional as F
from dct_utils import get_1d_dct_matrix


class DCTConv2d(nn.Module):
    """
    Convolutional layer whose learnable parameters live in the DCT (frequency)
    domain.  The forward pass materializes spatial weights via 2D IDCT before
    running a standard conv2d.

    Supports two modes:
      - Training: uses full-precision DCT coefficients.
      - Quantized (after calling quantize()): hard-thresholds and quantizes
        DCT coefficients, keeping only those that survive rounding — the same
        coefficients that an H.265 encoder would retain.
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

        # Learnable DCT coefficients (W_hat)
        self.weight_dct = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, K_h, K_w)
        )
        nn.init.kaiming_uniform_(self.weight_dct, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        # Precompute IDCT matrices: spatial_W = C_h^T @ W_hat @ C_w
        C_h = get_1d_dct_matrix(K_h)
        C_w = get_1d_dct_matrix(K_w)
        self.register_buffer("C_h_T", C_h.t())
        self.register_buffer("C_w", C_w)

        self._quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Materialize spatial weights via 2D IDCT
        spatial_weight = torch.einsum(
            "ab, oibd, cd -> oiac", self.C_h_T, self.weight_dct, self.C_w
        )

        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                spatial_weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )

        return F.conv2d(
            x, spatial_weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )

    @torch.no_grad()
    def quantize(self, qstep: float = 1.0):
        """
        Hard-quantize DCT coefficients in-place: divide by qstep, round,
        zero out coefficients that round to zero. This is the export path —
        after calling this, only the surviving integer levels are stored
        (multiplied back by qstep for the IDCT).

        Returns a dict with sparsity stats for this layer.
        """
        levels = torch.round(self.weight_dct.data / qstep)
        total = levels.numel()
        nonzero = (levels != 0).sum().item()

        # Store quantized coefficients (de-quantized back to continuous scale)
        self.weight_dct.data = levels * qstep
        self._quantized = True

        return {
            "total_coeffs": total,
            "nonzero_coeffs": nonzero,
            "sparsity": 1.0 - nonzero / total,
        }

    def get_sparse_coefficients(self, qstep: float = 1.0):
        """
        Extract only the non-zero quantized DCT coefficients as a sparse
        representation: list of (out_ch, in_ch, freq_h, freq_w, level).

        This is what you'd feed to an H.265-compatible entropy coder.
        """
        with torch.no_grad():
            levels = torch.round(self.weight_dct.data / qstep).to(torch.int16)
            nonzero_mask = levels != 0
            indices = nonzero_mask.nonzero(as_tuple=False)  # (N, 4)
            values = levels[nonzero_mask]                    # (N,)
        return indices, values

    def extra_repr(self) -> str:
        q_str = ", quantized" if self._quantized else ""
        return (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}{q_str}"
        )


def replace_with_dct_convs(module: nn.Module) -> None:
    """
    Recursively replace all nn.Conv2d layers (kernel > 1x1) with DCTConv2d.
    1x1 pointwise convolutions are left unchanged — they have no spatial
    frequencies to compress.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.kernel_size == (1, 1):
                continue

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
            setattr(module, name, dct_conv)
        else:
            replace_with_dct_convs(child)


def probe_sparsity(model: nn.Module, qstep: float = 1.0):
    """
    Non-destructive: count how many DCT coefficients would survive
    quantization without modifying the weights. Use this during training
    to monitor sparsity progress.
    """
    total = 0
    nonzero = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, DCTConv2d):
                levels = torch.round(m.weight_dct.data / qstep)
                total += levels.numel()
                nonzero += (levels != 0).sum().item()
    sparsity = 1.0 - nonzero / max(total, 1)
    return total, nonzero, sparsity


def quantize_model(model: nn.Module, qstep: float = 1.0):
    """
    Quantize all DCTConv2d layers in the model, zeroing out coefficients
    that don't survive rounding. Returns aggregate sparsity stats.
    """
    total_coeffs = 0
    total_nonzero = 0
    layer_stats = []

    for name, m in model.named_modules():
        if isinstance(m, DCTConv2d):
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
    """
    Export all DCT layers as sparse coefficient maps.
    Returns a dict mapping layer name to (indices, values) tuples,
    suitable for H.265-style entropy coding.
    """
    result = {}
    for name, m in model.named_modules():
        if isinstance(m, DCTConv2d):
            indices, values = m.get_sparse_coefficients(qstep)
            result[name] = {"indices": indices, "values": values}
    return result
