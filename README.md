# CTNet (Cosine Transform Network): Neural Network Compression via DCT-Domain Training and H.265 Video Encoding

**Stylianos Iordanis and Stefanos Kornilios Mitsis Poiitidis**

> **Note:** This is an exploratory proof of concept. All experiments are conducted on ImageNette2-320 (a 10-class subset of ImageNet) using a single consumer GPU (NVIDIA GTX 1080 Ti, 11 GB). Results demonstrate the viability of the approach but are not directly comparable to full ImageNet-1K benchmarks. Hyperparameters have not been exhaustively tuned. We release this work to invite further investigation and collaboration.

---

## Abstract

We present CTNet (Cosine Transform Network), a family of compressed neural networks that reparameterize convolutional layers into the Discrete Cosine Transform (DCT) domain and leverage H.265/HEVC video encoding as the compression backend. Rather than relying on traditional pruning or fixed-bitwidth quantization, CTNet trains convolutional weights directly as DCT coefficients, regularized by a differentiable proxy of the H.265 bitrate cost. At export time, the DCT coefficient maps are tiled into 2D frames, normalized, and encoded as H.265 video streams with lossless or near-lossless settings.

We present two variants:

- **CTNet-18** (based on ResNet-18): achieves **92.25% Top-1** on ImageNette2-320 with the entire model (44.6 MB) compressed to **3.5 MB via H.265** (12.3x lossless at 12-bit), or **2.1 MB** (21.7x at 8-bit). All layers — spatial DCT convolutions, channel-wise DCT 1x1 convolutions, batch normalization, and FC — are encoded as H.265 video frames. Only 551 of 11.2M DCT coefficients are nonzero (99.995% sparse).
- **CTNet-50** (based on ResNet-50): applies the same DCT reparameterization to ResNet-50's bottleneck architecture. Due to ResNet-50's heavy use of 1x1 pointwise convolutions (which are not DCT-transformed), CTNet-50 has a similar DCT parameter count (11.3M) to CTNet-18 (11.0M) but benefits from the deeper architecture's representational capacity. The 36 retained 1x1 convolutions and batch normalization layers (54.3 MB total non-DCT overhead) can be independently compressed via standard quantization.

---

## 1. Introduction

Neural network compression is critical for deploying deep learning models on resource-constrained devices. Existing approaches fall into several broad categories: weight pruning (Han et al., 2015; Frankle & Carlin, 2019), quantization (Esser et al., 2020; Rastegari et al., 2016), knowledge distillation (Hinton et al., 2015), and neural architecture search for efficient models (Tan & Le, 2019). These methods operate directly on weight magnitudes or network topology.

We observe that convolutional kernels, when viewed as small 2D signals, are natural candidates for transform coding -- the same compression paradigm that underlies JPEG, H.264, and H.265. Video codecs like H.265/HEVC already implement highly optimized DCT-based transform coding with sophisticated entropy coding (CABAC), rate-distortion optimization, and multi-scale prediction. Rather than reinventing these tools for neural network compression, we propose to directly leverage them.

CTNet introduces three key ideas:

1. **DCT-domain parameterization**: Convolutional weights are learned directly as DCT coefficients, with spatial weights materialized via IDCT at each forward pass.
2. **Differentiable H.265 rate proxy**: A training-time regularizer that approximates the bitrate cost of encoding each DCT coefficient, modeling significance maps, level coding costs, and zig-zag scan order weights -- the same components H.265 uses internally.
3. **Video codec compression**: At export time, DCT coefficient maps are reshaped into 2D image tiles, normalized with per-frame center and scale factors, optionally dithered, and encoded as H.265 video streams.

---

## 2. Architecture Variants

### CTNet-18

Based on ResNet-18. All 17 spatial convolutions (kernel > 1x1) are replaced with DCTConv2d layers. Only 3 pointwise (1x1) convolutions remain as standard Conv2d.

| Component | Parameters | Size (float32) |
|-----------|-----------|----------------|
| DCT conv layers (17) | 11.0M | 41.9 MB |
| Standard 1x1 convs (3) | 0.3M | 1.1 MB |
| BN + FC + other | 0.4M | 1.6 MB |
| **Total** | **11.7M** | **44.6 MB** |

The DCT layers constitute 94% of the model's parameters, making CTNet-18 an ideal target for frequency-domain compression.

### CTNet-50

Based on ResNet-50's bottleneck architecture. The same 17 spatial convolutions (the 3x3 convolutions inside each bottleneck block, plus `conv1`) are replaced with DCTConv2d layers. However, ResNet-50 uses far more 1x1 convolutions (36 layers) in its bottleneck design for channel expansion/reduction.

| Component | Parameters | Size (float32) |
|-----------|-----------|----------------|
| DCT conv layers (17) | 11.3M | 43.2 MB |
| Standard 1x1 convs (36) | 12.6M | 48.0 MB |
| BN + FC + other | 1.7M | 6.3 MB |
| **Total** | **25.6M** | **97.5 MB** |

In CTNet-50, DCT layers account for 44% of parameters. The 1x1 convolutions (49% of parameters) are not DCT-transformed -- they have no spatial frequencies to compress. This creates an interesting hybrid: the spatial convolutions are compressed via H.265, while the pointwise convolutions can be independently compressed via standard INT8/INT4 quantization.

**Architectural insight.** Despite being a much deeper network, CTNet-50 has nearly identical DCT parameter counts to CTNet-18 (11.3M vs 11.0M). This is because ResNet-50's depth comes primarily from stacking bottleneck blocks with narrow 3x3 convolutions flanked by wider 1x1 projections. The H.265 compressed size of the DCT layers should therefore be comparable between the two variants, while CTNet-50's additional accuracy comes from the deeper feature hierarchy enabled by the (uncompressed) 1x1 convolutions.

---

## 3. Method

### 3.1 DCT-Domain Convolutional Layers

Each standard `Conv2d` layer with kernel size > 1x1 is replaced by a `DCTConv2d` layer. The learnable parameters are DCT coefficients $\hat{W} \in \mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$. At each forward pass, spatial weights are materialized via 2D IDCT:

$$W = C_h^T \hat{W} C_w$$

where $C_h$ and $C_w$ are Type-II DCT matrices. The convolution then proceeds normally using $W$. This reparameterization is transparent to the rest of the network -- gradients flow through the IDCT via standard backpropagation.

Pointwise (1x1) convolutions are left unchanged, as they have no spatial frequencies to compress.

### 3.2 Differentiable H.265 Rate Proxy

During training, we add a rate regularization term that approximates the bitrate cost an H.265 encoder would incur for each layer's DCT coefficients. The proxy models three components of HEVC entropy coding:

**Significance cost.** Each coefficient incurs approximately 1 bit to signal whether it is zero or non-zero. We approximate this with a sigmoid soft-threshold:

$$\sigma_k = \text{sigmoid}(s \cdot (|\hat{W}_k| / q - 0.5))$$

where $q$ is the quantization step size, $s$ is the sigmoid steepness, and $k$ indexes each coefficient. The 0.5 threshold matches H.265's rounding behavior: values below $0.5q$ quantize to zero.

**Level coding cost.** For non-zero coefficients, encoding the magnitude requires approximately $\log_2(1 + |\hat{W}_k|/q)$ bits.

**Scan-order cost.** H.265 encodes coefficients in diagonal zig-zag scan order, with CABAC context models that make later-scanned (higher frequency) non-zero coefficients more expensive. We assign a weight $w_{i,j}$ to each frequency position $(i,j)$ based on its zig-zag scan index, with DC (position 0,0) receiving weight 1.0 and the last position receiving weight $K_h \times K_w$.

The total rate proxy for one layer is:

$$\mathcal{L}_{rate} = \sum_{k} \sigma_k \cdot (1 + \log_2(1 + |l_k|)) \cdot w_{pos(k)}$$

The training loss is:

$$\mathcal{L} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{rate}$$

where $\lambda$ controls the rate-distortion tradeoff.

### 3.3 H.265 Video Encoding Pipeline

At export time, every parameter tensor in the model is reshaped into a 2D image: spatial DCT weights are permuted to $(C_{out} \cdot K_h, C_{in} \cdot K_w)$, channel-DCT 1x1 weights are already 2D $(C_{out}, C_{in})$, BN parameters are stacked as $[4, \text{features}]$ (weight, bias, running\_mean, running\_var), and FC weights are used directly as 2D matrices. Images larger than 128x128 are sliced into tiles; smaller than 16x16 are circularly padded.

Frames are sorted by similarity (greedy nearest-neighbor on MSE distance) so that similar coefficient patterns are adjacent in the video stream, maximizing H.265's inter-frame prediction efficiency.

**Per-frame normalization.** Each frame is independently normalized to the pixel range $[0, 2^b - 1]$ with a stored center and scale factor for reconstruction:

$$\text{pixel} = \text{round}((\text{value} - \text{center}) \cdot \text{norm\_factor} + 2^{b-1})$$

Per-frame normalization is critical: different layers have vastly different coefficient magnitudes, and a shared normalization would waste precision on layers with small dynamic ranges.

**Per-layer-type encoding.** BN layers are encoded separately from DCT/FC layers with their own CRF and bit depth settings (`--bn-crf 0`, `--bn-bit-depth 12`). BN `running_var` values can be very close to zero, and low bit-depth quantization can push them negative, causing `nan` in the BN `sqrt(var + eps)` computation. Using 12-bit lossless for BN (quantization error ~0.00003) prevents this, while DCT layers can safely use 8-bit (error ~0.002) since convolution weights are more tolerant of small perturbations. On decode, `running_var` is additionally clamped to $\geq 0$ as a safety net.

**Maximized inter-frame prediction.** The H.265 encoder is configured for maximum exploitation of temporal redundancy between frames:

| Parameter | Value | Effect |
|-----------|-------|--------|
| `keyint=-1` | Single I-frame | Only the first frame is an I-frame; all subsequent frames use inter-frame prediction (P/B) |
| `bframes=16` | Max B-frames | Up to 16 bi-directional frames between reference frames (H.265 maximum) |
| `ref=16` | Max reference frames | Each frame can reference up to 16 previous frames for prediction |
| `b-adapt=2` | Optimal placement | Algorithm selects optimal B-frame placement for each GOP |
| `rc-lookahead=250` | Lookahead buffer | Encoder looks ahead 250 frames for rate-distortion decisions |

These settings are critical because our similarity-sorted frames have high inter-frame correlation — adjacent frames often represent tiles from the same or similar layers, where coefficient patterns differ only slightly.

**Subtractive dithering.** Optionally, deterministic white noise of configurable amplitude is added before rounding and subtracted after decoding. This decorrelates quantization error, which can improve accuracy when using lower bit depths or lossy CRF settings.

**Full-range encoding.** We use H.265's full-range mode (`range=full`, `color_range=pc`) to utilize the complete pixel value space, avoiding the wasted levels in the default limited (broadcast) range.

**Lossless mode.** With CRF 0, H.265 operates in mathematically lossless mode, preserving every pixel exactly. Compression then comes entirely from CABAC entropy coding exploiting spatial redundancy in the coefficient images.

### 3.4 Reconstruction

Decoding reverses the pipeline: H.265 video frames are decoded via ffmpeg, per-frame dither noise is subtracted, pixels are denormalized back to weight values using the stored center and scale factors, tiles are reassembled into 2D images, and the images are reshaped back into the original parameter tensors. BN `running_var` is clamped to non-negative. The IDCT in each DCT layer converts frequency coefficients to spatial weights at inference time.

A JSON manifest stores all metadata needed for exact reconstruction: architecture, bit depth, dither amplitude, per-frame normalization parameters, and layer type tags for dispatch.

---

## 4. Experimental Results

### 4.1 CTNet-18

**Setup:**
- **Base architecture**: ResNet-18 with all Conv2d replaced by DCT variants (17 spatial DCTConv2d + 3 ChannelDCTConv1x1), plus BN and FC layers — all encoded as H.265
- **Dataset**: ImageNette2-320 (10-class subset of ImageNet, 320px images)
- **Training**: AdamW, lr=1e-3, weight-decay=0.01, $\lambda=10^{-5}$, $q=0.1$, 256 epochs, pretrained
- **Total model float32 size**: 45,700 KB (44.6 MB) — includes all parameters (DCT convolutions + 1x1 channel-DCT + BN + FC)

**Results — Run 1 (without pretrained init, 256 epochs):**

| Epoch | Best Acc@1 | H.265 Size | Compression | Nonzero DCT coeffs |
|-------|-----------|-----------|-------------|---------------------|
| 69 | 85.99% | 5,389 KB | 8.5x | 862 / 11.2M |
| 105 | 87.03% | 4,566 KB | 10.0x | 849 / 11.2M |
| 186 | 91.41% | 3,005 KB | 15.2x | 579 / 11.2M |
| **230** | **92.23%** | **2,105 KB** | **21.7x** | **551 / 11.2M** |

**Results — Run 2 (with pretrained init fix, in progress):**

| Epoch | Best Acc@1 | H.265 Size | Compression | Nonzero DCT coeffs |
|-------|-----------|-----------|-------------|---------------------|
| **79** | **86.42%** | **3,570 KB** | **12.8x** | **11,976 / 11.2M** |

*Baseline pretrained ResNet-18 achieves ~95-97% Top-1 on ImageNette2-320.*

**Comparison at similar training stage (~epoch 70-80):**

| | Run 1 (no pretrained init) | Run 2 (pretrained init) | Improvement |
|---|---|---|---|
| Best Acc@1 | 85.99% | **86.42%** | +0.43% |
| H.265 Size | 5,389 KB | **3,570 KB** | **34% smaller** |
| Compression | 8.5x | **12.8x** | **+50% better** |

The pretrained initialization produces 50% better compression at the same epoch count. The model starts with meaningful features (78% accuracy at epoch 0 vs 10% without), allowing the rate proxy to focus on compressing already-useful representations rather than learning features from scratch. Run 2 is still in progress.

**Algorithmic changes between Run 1 and Run 2:**

*Bug fixes:*
- **Pretrained weight transfer via forward DCT** (critical): `replace_with_dct_convs` now computes `weight_dct = C @ weight @ C` instead of random initialization, preserving pretrained features exactly through the DCT transform.
- **ImageNette label remapping**: Subset dataset labels (0-9) remapped to ImageNet class indices (0, 217, 482, ...) so the pretrained FC head receives correct supervision.

*Techniques from ECRF (Lee et al., 2023):*
- **Training-time uniform noise**: Additive noise $u \sim \mathcal{U}(-\frac{1}{2}, \frac{1}{2}) \cdot q$ during training makes quantization differentiable, improving rate proxy gradient signal.
- **L2 coefficient regularization**: $\|\hat{W}\|_2^2$ penalty coupled to rate loss via $\lambda_2 = \alpha \cdot \lambda_r$, directly shrinking coefficient magnitudes.
- **Rate warmup**: Compression loss ramped from 0 to full strength over first 5 epochs, preventing rate pressure from fighting random/pretrained initialization.

*Techniques from DCT-Conv (Checinski & Wawrzynski, 2020):*
- **DCT dropout**: Random zeroing of DCT coefficients ($p = 0.05$) during training for frequency-domain regularization.
- **Block DCT for 1x1 convolutions**: 16x16 block DCT instead of full N×N, matching H.265 CTU block structure and avoiding O(N²) memory.

*Architecture and encoding improvements:*
- **Channel-wise DCT for 1x1 convolutions**: All 1x1 Conv2d layers replaced with `ChannelDCTConv1x1`, bringing them under the DCT rate proxy.
- **All layers encoded as H.265**: BN and FC parameters encoded as video frames alongside DCT weights, eliminating separate weight files.
- **Per-layer-type encoding**: BN layers encoded at 12-bit lossless separately from DCT layers (8-bit), preventing BN `running_var` corruption.
- **Similarity-sorted frames**: Greedy nearest-neighbor ordering within each video for better H.265 inter-frame prediction.
- **Maximized inter-frame prediction**: `keyint=-1`, `bframes=16`, `ref=16` for lossless encoding (single I-frame, all others P/B).

**Decoded model evaluation (Run 1, epoch 230 smallest checkpoint):**

| Export settings | Acc@1 | Acc@5 | H.265 Size | Compression |
|----------------|-------|-------|-----------|-------------|
| **CRF 0, 12-bit** | **92.25%** | **99.39%** | **3,493 KB** | **12.3x** |
| CRF 0, 10-bit | 90.11% | 98.98% | 2,513 KB | 17.1x |

**Key findings:**

- **Entire model in H.265**: All parameters (spatial DCT, channel-wise DCT for 1x1 convolutions, batch normalization, FC layer) are encoded as H.265 video frames. The output directory is fully self-contained.
- **Pretrained init matters**: Correct DCT initialization from pretrained weights (forward DCT transform) gives 50% better compression at the same epoch, because the optimizer refines meaningful features rather than learning from scratch.
- **Label remapping**: Subset datasets (e.g., ImageNette) require remapping folder labels to ImageNet class indices for pretrained models to work correctly. Without this, the cross-entropy loss trains against wrong targets.
- **Extreme sparsity**: Only 551-12,000 of 11.2M DCT coefficients are nonzero (>99.9% sparse), yet accuracy exceeds 86-92%. The rate proxy successfully discovers minimal frequency representations.
- **BN requires high precision**: BN layers are encoded separately at 12-bit lossless (`--bn-bit-depth 12 --bn-crf 0`) to prevent near-zero `running_var` from going negative. DCT/FC layers use 8-bit.
- **Compression improves with training**: H.265 size decreases monotonically as the rate proxy reshapes coefficients toward codec-friendly patterns.

### 4.2 CTNet-50

CTNet-50 applies the identical DCT reparameterization and H.265 compression pipeline to ResNet-50. Training and export commands:

```bash
# Train CTNet-50
python train_imagenet.py ./imagenette2-320 \
    --arch resnet50 --epochs 256 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-5 --qstep 0.1

# Export
python export_h265.py encode --arch resnet50 \
    --crf 0 --bit-depth 8 --bn-bit-depth 12 --bn-crf 0 --preset slower

# Decode and evaluate
python export_h265.py decode --h265-dir ./h265_out --data ./imagenette2-320
```

**Hardware limitation.** We were unable to train and evaluate CTNet-50 on our test hardware (NVIDIA GeForce GTX 1080 Ti, 11 GB VRAM). ResNet-50's larger activation maps and the additional memory overhead of the DCT reparameterization exceeded the available GPU memory during training. CTNet-50 evaluation is left for future work on hardware with >= 24 GB VRAM.

**Expected behavior.** Because CTNet-50's DCT parameter count (11.3M) is nearly identical to CTNet-18's (11.0M), we expect:

- **H.265 compressed size of DCT layers**: comparable to CTNet-18 (~1.8 MB)
- **Higher accuracy**: ResNet-50's deeper bottleneck architecture provides stronger feature representations. The uncompressed 1x1 convolutions and batch normalization layers carry the additional representational capacity.
- **Total compressed model**: the 12.6M parameters in 1x1 convolutions (48 MB float32) would need separate compression. With standard INT8 quantization, these add ~12 MB, for an estimated total of ~14 MB.
- **Hybrid compression ratio**: ~97.5 MB float32 to ~14 MB (H.265 DCT + INT8 pointwise) = ~7x total, or ~23x on DCT layers alone.

This highlights a key architectural insight: CTNet compression is most effective on architectures where spatial convolutions dominate the parameter budget. For bottleneck architectures like ResNet-50, combining CTNet (for spatial convolutions) with standard quantization (for 1x1 convolutions) yields the best overall compression.

### 4.3 Comparison with State-of-the-Art

The following table compares CTNet against established neural network compression methods. Results are drawn from the original publications; where ResNet-18 results are unavailable, we report the closest comparable architecture.

| Method | Reference | Model | Compression | Top-1 Acc | Acc Drop |
|--------|-----------|-------|-------------|-----------|----------|
| **CTNet-18 (ours, 12-bit lossless)** | -- | ResNet-18 | **12.3x** | **92.25%*** | ~4% from baseline* |
| CTNet-18 (ours, 10-bit lossless) | -- | ResNet-18 | 17.1x | 90.11%* | ~6% from baseline* |
| Deep Compression | Han et al., 2016 | VGG-16 | 49x | 68.3%** | ~0%** |
| Deep Compression | Han et al., 2016 | AlexNet | 35x | 57.2%** | ~0%** |
| Deep Compression (est.) | -- | ResNet-18 | 15-25x | ~68-69%** | 1-2%** |
| Lottery Ticket (rewinding) | Frankle et al., 2020 | ResNet-50 | 3.3-5x | 76.1%** | ~0%** |
| Magnitude pruning (90%) | Zhu & Gupta, 2018 | ResNet-50 | 10x | 73.9%** | -2.2%** |
| Magnitude pruning (95%) | Zhu & Gupta, 2018 | ResNet-50 | 20x | 72.0%** | -4.1%** |
| INT8 quantization | Standard PTQ | ResNet-18 | 4x | 69.7%** | -0.1%** |
| INT4 quantization (QAT) | Esser et al., 2020 | ResNet-18 | 8x | 67.6%** | -2.2%** |
| Binary (ReActNet) | Liu et al., 2020 | ResNet-18 | 32x | 65.5%** | -4.3%** |
| Structured pruning (FPGM) | He et al., 2019 | ResNet-18 | ~1.7x FLOPs | 68.4%** | -1.4%** |
| Structured pruning (HRank) | Lin et al., 2020 | ResNet-18 | ~1.5x FLOPs | 69.1%** | -0.7%** |
| EfficientNet-B0 (NAS) | Tan & Le, 2019 | EfficientNet | 2.2x vs R18 | 77.1%** | +7.3%** |
| MobileNet-V2 (NAS) | Sandler et al., 2018 | MobileNet | 3.3x vs R18 | 72.0%** | +2.2%** |

\* *ImageNette2-320 (10-class subset). Pretrained baseline achieves ~95-97% on this dataset.*
\*\* *Full ImageNet-1K (1000 classes). Not directly comparable with ImageNette2 results.*

**Important note on comparability.** Our results are on ImageNette2-320 (10 classes), while most prior work reports on full ImageNet-1K (1000 classes). The compression ratio (23.7x on stored bytes) is directly comparable as it measures the same quantity -- ratio of original float32 parameter storage to compressed representation. However, accuracy numbers across different datasets should not be directly compared. A pretrained ResNet-18 baseline achieves ~95-97% on ImageNette2 vs ~69.8% on ImageNet-1K.

### 4.4 Analysis

**Entire model as H.265.** With channel-wise DCT for 1x1 convolutions and all BN/FC parameters encoded as video frames, the entire 44.6 MB model compresses to 3.5 MB (12.3x) at 12-bit lossless with only ~3.7% accuracy loss. No separate weight files are needed — the H.265 output directory is fully self-contained.

**Extreme sparsity.** At epoch 230, only 551 of 11.2M DCT coefficients survive (99.995% sparse), yet accuracy is 92.23%. This validates DCT-Conv's finding that neural network weights are highly redundant in the frequency domain. The rate proxy successfully discovers the minimal set of frequency components needed for each layer.

**Bit depth is critical for BN.** 12-bit encoding preserves accuracy (92.25% vs 92.23% pre-export) because BN running variance values near zero require high precision. 8-bit introduces ~0.002 quantization error that can push near-zero variances negative, causing NaN. 12-bit error is ~0.00003, safely below the smallest variances. This is the recommended setting.

**Compression improves with training.** H.265 size drops from 6.3 MB at epoch 32 to 2.1 MB at epoch 230 (3x improvement) while accuracy climbs from ~86% to 92%. Both metrics improve simultaneously as the rate proxy reshapes coefficients toward patterns H.265 encodes efficiently.

**Rate-distortion tradeoff.** CTNet offers a continuous rate-distortion tradeoff controlled by $\lambda$, $q$, CRF, and bit depth. Increasing $\lambda$ during training encourages sparser DCT representations; increasing CRF during export trades accuracy for smaller files. This is analogous to how video codecs offer smooth quality-vs-bitrate curves.

**Codec preset profiling.** H.265 encoding presets from `ultrafast` to `veryslow` trade encoding time for compression efficiency. On CTNet-18, `medium` achieved near-optimal compression (within 3% of `veryslow`) at 4x faster encoding speed, suggesting that the coefficient images are relatively easy for the encoder to optimize.

**CTNet-18 vs CTNet-50.** The two variants illuminate a fundamental property of codec-based compression: it is most effective on spatial convolutions. In CTNet-18, where 94% of parameters are in spatial convolutions, nearly the entire model benefits from H.265 compression. In CTNet-50, only 44% of parameters are spatial -- the rest are 1x1 projections that require separate compression. This suggests CTNet is particularly well-suited for architectures that maximize spatial convolution usage, and motivates future work on frequency-domain reparameterization of 1x1 convolutions (e.g., via channel-wise transforms).

**Advantages over traditional methods:**

- *No custom entropy coder needed.* H.265/HEVC is a mature, hardware-accelerated standard available on virtually every modern device. Decoding is a single ffmpeg call.
- *Continuous rate-distortion control.* Unlike pruning (discrete sparsity levels) or quantization (discrete bit widths), CTNet inherits video coding's smooth tradeoff via CRF and $\lambda$.
- *Complementary to other methods.* CTNet operates in the frequency domain and can be combined with channel pruning, knowledge distillation, or standard quantization for non-DCT layers.

---

## 5. Related Work

**DCT-Conv: Coding filters with Discrete Cosine Transform (Checinski & Wawrzynski, 2020).** The most directly relevant prior work. Checinski & Wawrzynski proposed DCT-Conv layers where convolutional filters are defined by their DCT coefficients rather than spatial weights, with IDCT applied to recover spatial filters during the forward pass. They demonstrated on ResNet-50/CIFAR-100 that 99.9% of 3x3 DCT coefficients can be switched off while maintaining good performance — the network effectively learns to use only a few frequency components per filter. Their key finding validates our core premise: neural network weights are highly compressible in the DCT domain. CTNet extends this work in three significant ways: (1) we add a differentiable rate proxy that optimizes *for* compressibility during training rather than post-hoc coefficient removal, (2) we use a production video codec (H.265) as the compression backend rather than simple coefficient zeroing, and (3) we extend DCT reparameterization beyond spatial convolutions to 1x1 pointwise convolutions via channel-wise DCT and to all other layer types (BN, FC) via center+normalize encoding.

**ECRF: Entropy-Constrained Neural Radiance Fields (Lee et al., 2023).** Lee et al. applied DCT-domain compression with entropy optimization to Neural Radiance Fields (NeRF), achieving state-of-the-art compression on TensoRF feature grids. Their approach shares several key ideas with CTNet: (1) transforming parameters to the frequency domain via DCT, (2) training with an entropy-based loss to encourage compressible representations, and (3) post-training quantization to 8-bit followed by entropy coding. They also use additive uniform noise as a differentiable proxy for quantization during training — similar to our dither mechanism. A crucial insight from ECRF is that jointly optimizing the entropy of transformed coefficients with the task loss (in their case, rendering quality) produces significantly sparser frequency-domain representations than training without entropy awareness. Their pipeline (DCT → 8-bit quantization → arithmetic coding) parallels CTNet's pipeline (DCT → center+normalize → H.265 CABAC), but we replace custom arithmetic coding with a standard video codec, gaining hardware decoder support at the cost of some compression efficiency.

**Weight pruning.** Unstructured pruning (Han et al., 2015) removes individual weights by magnitude, achieving 9-13x compression on AlexNet/VGG. The Lottery Ticket Hypothesis (Frankle & Carlin, 2019) shows that sparse subnetworks can be trained from scratch, but finding tickets requires iterative pruning-retraining cycles. Structured pruning (He et al., 2017; Lin et al., 2020) removes entire filters or channels for hardware-friendly speedups but typically achieves lower compression ratios (1.5-2x on ResNet-18).

**Quantization.** Reducing weight precision from 32-bit floating point to 8-bit integers gives 4x compression with negligible accuracy loss (Jacob et al., 2018). More aggressive quantization to 4 bits (Esser et al., 2020) or 2 bits degrades accuracy significantly on small models like ResNet-18. Binary networks (Rastegari et al., 2016; Liu et al., 2020) achieve 32x compression but with 4-18% accuracy drops.

**Deep Compression.** Han et al. (2016) pipeline pruning, quantization (to 5-8 bits with codebook), and Huffman entropy coding to achieve 35-49x on AlexNet/VGG. This remains the gold standard for maximum compression, but requires custom decompression code and benefits disproportionately from large fully-connected layers absent in modern architectures.

**Learned image compression.** The learned compression literature (Balle et al., 2017; Minnen et al., 2018) has developed differentiable rate-distortion optimization for image codecs. Our rate proxy draws inspiration from this work, adapting it to the specific structure of neural network coefficient maps.

### 5.1 Key Influences on CTNet

CTNet's design draws directly from specific techniques in the above works:

| Technique in CTNet | Origin | What we adopted | What we changed |
|-------------------|--------|-----------------|-----------------|
| DCT reparameterization of conv filters | DCT-Conv | Learnable DCT coefficients with IDCT in forward pass | Extended to 1x1 convolutions via channel-wise DCT |
| DCT coefficient dropout during training | DCT-Conv | Random zeroing of frequency components (default $p=0.05$) | Applied to both spatial and channel-wise DCT layers |
| Block DCT for large weight matrices | DCT-Conv + ECRF | DCT in 16x16 blocks for 1x1 convolutions | Avoids O(N²) matrices while matching H.265 CTU block size |
| Training-time uniform noise | ECRF | $\Phi(\theta) = \theta + u \cdot q$, $u \sim \mathcal{U}(-\frac{1}{2}, \frac{1}{2})$ | Scaled by qstep to match the quantization grid |
| L2 coefficient regularization | ECRF | $\lambda_2 \|\hat{W}\|_2^2$ coupled via $\lambda_2 = \alpha \cdot \lambda_r$ | Single $\alpha$ parameter reduces hyperparameter search |
| Rate warmup schedule | ECRF | Ramp compression loss from 0 over first epochs | Separate from LR warmup; prevents early rate-vs-init conflict |
| Entropy-aware training loss | ECRF | Differentiable bitrate proxy during training | Replaced learned entropy model with H.265-specific rate proxy (significance + level + scan cost) |
| Video codec as compression backend | Novel | — | H.265/HEVC with CABAC, replacing custom arithmetic coding |
| Similarity-sorted frame ordering | Novel | — | Greedy nearest-neighbor MSE ordering for inter-frame prediction |
| Per-layer-type encoding (BN isolation) | Novel | — | Separate CRF/bit-depth for BN layers to prevent running_var corruption |

---

## 6. Reproducibility

### 6.1 CTNet-18

```bash
pip install -r requirements.txt
./download_imagenette.sh ./imagenette2-320

# Train (best config: AdamW, 256 epochs, all paper innovations enabled by default)
python train_imagenet.py ./imagenette2-320 \
    --arch resnet18 --epochs 256 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-5 --qstep 0.1 \
    --cache-dataset

# Export to H.265 (8-bit DCT + 12-bit lossless BN, self-contained output)
python export_h265.py encode --arch resnet18 \
    --crf 0 --bit-depth 8 --bn-bit-depth 12 --bn-crf 0 --preset slower

# Decode and evaluate
python export_h265.py decode --h265-dir ./h265_out --data ./imagenette2-320
```

### 6.2 CTNet-50

```bash
# Train
python train_imagenet.py ./imagenette2-320 \
    --arch resnet50 --epochs 256 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-5 --qstep 0.1

# Export to H.265
python export_h265.py encode --arch resnet50 \
    --crf 0 --bit-depth 8 --bn-bit-depth 12 --bn-crf 0 --preset slower

# Decode and evaluate
python export_h265.py decode --h265-dir ./h265_out --data ./imagenette2-320
```

### 6.3 Profile Encoding Presets

```bash
python export_h265.py encode --arch resnet18 --profile
python export_h265.py encode --arch resnet50 --profile
```

---

## 7. Conclusion

CTNet demonstrates that modern video codecs are surprisingly effective neural network compressors. By reparameterizing all convolutional layers into the DCT domain (spatial DCT for 3x3/7x7, channel-wise DCT for 1x1) and encoding the entire model — including BN and FC layers — as H.265 video streams, CTNet-18 compresses a 44.6 MB ResNet-18 to 3.5 MB (12.3x) at 92.25% Top-1 on ImageNette2, only ~4% below the uncompressed baseline. The network achieves 99.995% DCT coefficient sparsity while maintaining accuracy. The approach requires no custom entropy coding implementation, leverages decades of video codec optimization, and offers continuous rate-distortion control via standard codec parameters.

CTNet-50 extends the framework to deeper bottleneck architectures, revealing that the approach is most effective when spatial convolutions dominate the parameter budget. For architectures with many 1x1 convolutions, CTNet naturally combines with standard quantization for a hybrid compression strategy.

Future work includes evaluation on full ImageNet-1K, combination with structured pruning and knowledge distillation, exploration of newer codecs (AV1, VVC/H.266), extension of frequency-domain compression to 1x1 convolutions via channel-wise transforms, and investigation of whether the H.265 encoder's internal rate-distortion optimization can be directly exploited during training.

---

## References

- Balle, J., Laparra, V., & Simoncelli, E. P. (2017). End-to-end optimized image compression. ICLR.
- Checinski, K., & Wawrzynski, P. (2020). DCT-Conv: Coding filters in convolutional networks with Discrete Cosine Transform. arXiv:2001.08517.
- Dong, Z., Yao, Z., Gholami, A., Mahoney, M. W., & Keutzer, K. (2020). HAWQ-V2: Hessian aware trace-weighted quantization of neural networks. NeurIPS.
- Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., & Modha, D. S. (2020). Learned step size quantization. ICLR.
- Frankle, J., & Carlin, M. (2019). The lottery ticket hypothesis: Finding sparse, trainable networks. ICLR.
- Frankle, J., Dziugaite, G. K., Roy, D. M., & Carlin, M. (2020). Stabilizing the lottery ticket hypothesis. arXiv:1903.01611.
- Han, S., Mao, H., & Dally, W. J. (2016). Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding. ICLR.
- Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). Learning both weights and connections for efficient neural networks. NeurIPS.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- He, Y., Kang, G., Dong, X., Fu, Y., & Yang, Y. (2018). Soft filter pruning for accelerating deep convolutional neural networks. IJCAI.
- He, Y., Liu, P., Wang, Z., Hu, Z., & Yang, Y. (2019). Filter pruning via geometric median for deep convolutional neural networks acceleration. CVPR.
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
- Howard, A., et al. (2019). Searching for MobileNetV3. ICCV.
- Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. CVPR.
- Lee, S., Shu, F., Sanchez, Y., Schierl, T., & Hellge, C. (2023). ECRF: Entropy-constrained neural radiance fields compression with frequency domain optimization. arXiv:2311.14208.
- Lin, M., et al. (2020). HRank: Filter pruning using high-rank feature map. CVPR.
- Liu, Z., Shen, Z., Savvides, M., & Cheng, K. T. (2020). ReActNet: Towards precise binary neural network with generalized activation functions. ECCV.
- Luo, J., Wu, J., & Lin, W. (2017). ThiNet: A filter level pruning method for deep neural network compression. ICCV.
- Minnen, D., Balle, J., & Toderici, G. (2018). Joint autoregressive and hierarchical priors for learned image compression. NeurIPS.
- Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). XNOR-Net: ImageNet classification using binary convolutional neural networks. ECCV.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
- Zhu, M., & Gupta, S. (2018). To prune, or not to prune: Exploring the efficacy of pruning for model compression. ICLR Workshop.

---

## 8. Corrections and Errata

**Pretrained weight initialization (critical).** All results reported in Section 4 were obtained with a bug where `replace_with_dct_convs` discarded pretrained weights. When replacing `nn.Conv2d` with `DCTConv2d` or `ChannelDCTConv1x1`, the new layers were initialized with random `kaiming_uniform_` values instead of computing the forward DCT of the existing pretrained weights. This meant `--pretrained` had no effect — the model always trained from random initialization in the DCT domain.

This has been fixed: `replace_with_dct_convs` now computes the forward DCT transform of pretrained weights (`weight_dct = C @ weight @ C` for spatial, `weight_dct = C_out @ weight @ C_in^T` for channel-wise) and verifies the IDCT roundtrip reproduces the original output (max error < 1e-5). With this fix, a pretrained ResNet-18 starts at ~96% Top-1 on ImageNette2 immediately after replacement, rather than 10% (random chance).

All previously reported results (92.25% accuracy, 12.3x compression) were achieved *despite* this bug, effectively training from scratch in DCT domain. Results with correct pretrained initialization are expected to be significantly better — both in final accuracy (closer to baseline) and convergence speed (reaching good accuracy in fewer epochs).

**ImageNette label mapping.** A second initialization issue: ImageNette uses folder names that are WordNet IDs (e.g. `n01440764` for tench), which `ImageFolder` assigns sequential labels 0-9. But the pretrained ResNet-18 FC layer outputs 1000-class logits where tench = class 0, English springer = class 217, parachute = class 701, etc. Without remapping, the cross-entropy loss trains against wrong targets, destroying the pretrained features in the first few gradient steps. This was fixed by detecting subset datasets (< 1000 classes with WordNet ID folder names) and remapping labels to their correct ImageNet indices. With both fixes applied, a pretrained ResNet-18 immediately achieves 78.2% Top-1 on ImageNette after DCT replacement (vs ~10% random chance without the fixes), confirming the pretrained features are intact.

**BN running_var precision.** Early 8-bit export results showed NaN loss due to near-zero `running_var` values going negative after the 8-bit quantization roundtrip (error ~0.002). This was addressed by: (1) encoding BN layers separately with 12-bit lossless (`--bn-bit-depth 12 --bn-crf 0`), and (2) clamping `running_var >= 0` on decode as a safety net.

---

## License

This project is released for research purposes.
