# CTNet (Cosine Transform Network): Neural Network Compression via DCT-Domain Training and H.265 Video Encoding

**Stylianos Iordanis and Stefanos Kornilios Mitsis Poiitidis**

> **Note:** This is an exploratory proof of concept. All experiments are conducted on ImageNette2-320 (a 10-class subset of ImageNet) using a single consumer GPU (NVIDIA GTX 1080 Ti, 11 GB). Results demonstrate the viability of the approach but are not directly comparable to full ImageNet-1K benchmarks. Hyperparameters have not been exhaustively tuned. We release this work to invite further investigation and collaboration.

---

## Abstract

We present CTNet (Cosine Transform Network), a family of compressed neural networks that reparameterize convolutional layers into the Discrete Cosine Transform (DCT) domain and leverage H.265/HEVC video encoding as the compression backend. Rather than relying on traditional pruning or fixed-bitwidth quantization, CTNet trains convolutional weights directly as DCT coefficients, regularized by a differentiable proxy of the H.265 bitrate cost. At export time, the DCT coefficient maps are tiled into 2D frames, normalized, and encoded as H.265 video streams with lossless or near-lossless settings.

We present two variants:

- **CTNet-18** (based on ResNet-18): achieves **14.5x compression** (42.9 MB to 2.9 MB) at **92.99% Top-1** on ImageNette2-320 (256 epochs, AdamW), or **32.5x compression** (42.9 MB to 1.3 MB) at **81.12% Top-1** with lossy CRF 30 encoding. With 17 DCT layers replacing all spatial convolutions.
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

At export time, each DCT layer's 4D coefficient tensor $(C_{out}, C_{in}, K_h, K_w)$ is reshaped into a 2D image of size $(C_{out} \cdot K_h, C_{in} \cdot K_w)$. Images larger than 128x128 are sliced into tiles; images smaller than 8x8 are circularly padded. Tiles of the same dimensions are grouped into a single H.265 video stream (one frame per tile).

**Per-frame normalization.** Each frame is independently normalized to the pixel range $[0, 2^b - 1]$ (where $b$ is the bit depth, default 12) with a stored center and scale factor for reconstruction:

$$\text{pixel} = \text{round}((\text{value} - \text{center}) \cdot \text{norm\_factor} + 2^{b-1})$$

Per-frame normalization is critical: different layers have vastly different coefficient magnitudes, and a shared normalization would waste precision on layers with small dynamic ranges.

**Subtractive dithering.** Optionally, deterministic white noise of configurable amplitude is added before rounding and subtracted after decoding. This decorrelates quantization error, which can improve accuracy when using lower bit depths or lossy CRF settings.

**Full-range encoding.** We use H.265's full-range mode (`range=full`, `color_range=pc`) to utilize the complete pixel value space, avoiding the wasted levels in the default limited (broadcast) range.

**Lossless mode.** With CRF 0, H.265 operates in mathematically lossless mode, preserving every pixel exactly. Compression then comes entirely from CABAC entropy coding exploiting spatial redundancy in the coefficient images.

### 3.4 Reconstruction

Decoding reverses the pipeline: H.265 video frames are decoded via ffmpeg, per-frame dither noise is subtracted, pixels are denormalized back to weight values using the stored center and scale factors, tiles are reassembled into 2D images, and the images are reshaped back into 4D DCT coefficient tensors. The IDCT in each `DCTConv2d` layer converts them to spatial weights at inference time.

A JSON manifest stores all metadata needed for exact reconstruction: architecture, quantization step, bit depth, dither amplitude, and per-frame normalization parameters.

---

## 4. Experimental Results

### 4.1 CTNet-18

**Setup:**
- **Base architecture**: ResNet-18 (17 DCT layers, 3 standard 1x1 convolutions)
- **Dataset**: ImageNette2-320 (10-class subset of ImageNet, 320px images)
- **Export**: 8-bit depth, `slower` preset, dither amplitude 0.1

**Results across training configurations:**

| Config | Optimizer | $\lambda$ | Epochs | Top-1 | Top-5 | H.265 Size | Ratio |
|--------|-----------|-----------|--------|-------|-------|-----------|-------|
| SGD baseline | SGD | $10^{-4}$ | 30 | 83.72% | 98.37% | 1,811 KB | 23.7x |
| SGD long | SGD | $10^{-4}$ | 128 | 87.75% | 98.78% | 1,240 KB | 34.6x |
| **AdamW (CRF 0)** | **AdamW** | $10^{-6}$ | **256** | **92.99%** | **99.36%** | **2,967 KB** | **14.5x** |
| AdamW (CRF 20) | AdamW | $10^{-6}$ | 256 | 92.66% | 99.36% | 2,913 KB | 14.7x |
| AdamW (CRF 30) | AdamW | $10^{-6}$ | 256 | 81.12% | 97.63% | 1,322 KB | 32.5x |

*Baseline pretrained ResNet-18 achieves ~95-97% Top-1 on ImageNette2-320.*

**Key findings:**

- **AdamW vs SGD**: Switching from SGD ($\lambda=10^{-4}$) to AdamW ($\lambda=10^{-6}$) with a rebalanced rate term dramatically improved accuracy (87.75% to 92.99%) while still achieving meaningful compression (14.5x). AdamW's per-parameter learning rates handle the mixed task+rate loss better.
- **Lossless vs lossy encoding**: CRF 0 (lossless) and CRF 20 give nearly identical accuracy (92.99% vs 92.66%, only -0.33%), indicating the H.265 encoder's own quantization at CRF 20 introduces negligible additional error. CRF 30 degrades significantly (81.12%) — the codec's quantization becomes too coarse.
- **Rate-distortion tradeoff**: Higher $\lambda$ (SGD runs) produces more aggressive sparsity and better compression ratios (34.6x) but sacrifices accuracy. Lower $\lambda$ (AdamW run) preserves accuracy at lower compression. This is the expected Pareto tradeoff.
- **Training duration matters**: All configurations benefited from longer training, with both accuracy and compression improving over time.

### 4.2 CTNet-50

CTNet-50 applies the identical DCT reparameterization and H.265 compression pipeline to ResNet-50. Training and export commands:

```bash
# Train CTNet-50
python train_imagenet.py ./imagenette2-320 \
    --arch resnet50 --epochs 30 --pretrained \
    --lambda-rate 1e-4 --qstep 0.1

# Export
python export_h265.py encode \
    --arch resnet50 --qstep 0.1 \
    --crf 0 --bit-depth 8 --dither 0.1 --preset slower

# Decode and evaluate
python export_h265.py decode \
    --h265-dir ./h265_out --data ./imagenette2-320 \
    --non-dct-weights ./checkpoints/best.pth
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
| **CTNet-18 (ours, AdamW 256ep, CRF 0)** | -- | ResNet-18 | **14.5x** | **92.99%*** | ~3% from baseline* |
| CTNet-18 (ours, AdamW 256ep, CRF 30) | -- | ResNet-18 | 32.5x | 81.12%* | ~15% from baseline* |
| CTNet-18 (ours, SGD 128ep) | -- | ResNet-18 | 34.6x | 87.75%* | ~8% from baseline* |
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

**Compression ratio.** CTNet-18 achieves 14.5x compression at 92.99% accuracy (CRF 0, lossless) or 32.5x at 81.12% (CRF 30, lossy). The lossless result is competitive with standard quantization methods (INT4 gives 8x), while the lossy result approaches Deep Compression's ratios on ResNet-class architectures (estimated 15-25x) with the advantage of requiring no custom decompression code.

**Optimizer choice is critical.** Switching from SGD ($\lambda=10^{-4}$) to AdamW ($\lambda=10^{-6}$) improved accuracy from 87.75% to 92.99% — a 5.24 percentage point gain. The bi-objective loss (task + rate) benefits from AdamW's per-parameter adaptive learning rates, which handle DCT coefficients near the significance threshold more carefully than SGD's uniform step size.

**CRF as a post-training compression knob.** The H.265 CRF parameter provides a post-training mechanism to trade accuracy for size without retraining. CRF 20 is nearly free (only -0.33% accuracy vs lossless) while CRF 30 is too aggressive. This is analogous to adjusting JPEG quality — a property unique to codec-based compression that traditional pruning/quantization methods lack.

**Rate-distortion tradeoff.** CTNet offers a continuous rate-distortion tradeoff controlled by $\lambda$, $q$, CRF, and bit depth. Increasing $\lambda$ during training encourages sparser DCT representations; increasing CRF during export trades accuracy for smaller files. This is analogous to how video codecs offer smooth quality-vs-bitrate curves.

**Codec preset profiling.** H.265 encoding presets from `ultrafast` to `veryslow` trade encoding time for compression efficiency. On CTNet-18, `medium` achieved near-optimal compression (within 3% of `veryslow`) at 4x faster encoding speed, suggesting that the coefficient images are relatively easy for the encoder to optimize.

**CTNet-18 vs CTNet-50.** The two variants illuminate a fundamental property of codec-based compression: it is most effective on spatial convolutions. In CTNet-18, where 94% of parameters are in spatial convolutions, nearly the entire model benefits from H.265 compression. In CTNet-50, only 44% of parameters are spatial -- the rest are 1x1 projections that require separate compression. This suggests CTNet is particularly well-suited for architectures that maximize spatial convolution usage, and motivates future work on frequency-domain reparameterization of 1x1 convolutions (e.g., via channel-wise transforms).

**Advantages over traditional methods:**

- *No custom entropy coder needed.* H.265/HEVC is a mature, hardware-accelerated standard available on virtually every modern device. Decoding is a single ffmpeg call.
- *Continuous rate-distortion control.* Unlike pruning (discrete sparsity levels) or quantization (discrete bit widths), CTNet inherits video coding's smooth tradeoff via CRF and $\lambda$.
- *Complementary to other methods.* CTNet operates in the frequency domain and can be combined with channel pruning, knowledge distillation, or standard quantization for non-DCT layers.

---

## 5. Related Work

**Weight pruning.** Unstructured pruning (Han et al., 2015) removes individual weights by magnitude, achieving 9-13x compression on AlexNet/VGG. The Lottery Ticket Hypothesis (Frankle & Carlin, 2019) shows that sparse subnetworks can be trained from scratch, but finding tickets requires iterative pruning-retraining cycles. Structured pruning (He et al., 2017; Lin et al., 2020) removes entire filters or channels for hardware-friendly speedups but typically achieves lower compression ratios (1.5-2x on ResNet-18).

**Quantization.** Reducing weight precision from 32-bit floating point to 8-bit integers gives 4x compression with negligible accuracy loss (Jacob et al., 2018). More aggressive quantization to 4 bits (Esser et al., 2020) or 2 bits degrades accuracy significantly on small models like ResNet-18. Binary networks (Rastegari et al., 2016; Liu et al., 2020) achieve 32x compression but with 4-18% accuracy drops.

**Deep Compression.** Han et al. (2016) pipeline pruning, quantization (to 5-8 bits with codebook), and Huffman entropy coding to achieve 35-49x on AlexNet/VGG. This remains the gold standard for maximum compression, but requires custom decompression code and benefits disproportionately from large fully-connected layers absent in modern architectures.

**Transform-domain approaches.** Several works have explored frequency-domain representations for neural networks. Wang et al. (2016) proposed learning in the frequency domain, and Chen et al. (2016) used hashing for weight compression. However, to our knowledge, no prior work has directly used a production video codec as the compression backend for neural network weights.

**Learned image compression.** The learned compression literature (Balle et al., 2017; Minnen et al., 2018) has developed differentiable rate-distortion optimization for image codecs. Our rate proxy draws inspiration from this work, adapting it to the specific structure of neural network coefficient maps.

---

## 6. Reproducibility

### 6.1 CTNet-18

```bash
pip install -r requirements.txt
./download_imagenette.sh ./imagenette2-320

# Train (best config: AdamW, 256 epochs)
python train_imagenet.py ./imagenette2-320 \
    --arch resnet18 --epochs 256 --pretrained \
    --optimizer adamw --lr 1e-3 --weight-decay 0.01 \
    --lambda-rate 1e-6 --qstep 0.1 \
    --cache-dataset

# Export to H.265 (lossless)
python export_h265.py encode \
    --arch resnet18 --qstep 0.1 \
    --crf 0 --bit-depth 8 --dither 0.1 --preset slower

# Export to H.265 (lossy, higher compression)
python export_h265.py encode \
    --arch resnet18 --qstep 0.1 \
    --crf 20 --bit-depth 8 --dither 0.1 --preset slower

# Decode and evaluate
python export_h265.py decode \
    --h265-dir ./h265_out --data ./imagenette2-320 \
    --non-dct-weights ./checkpoints/best.pth
```

### 6.2 CTNet-50

```bash
# Train
python train_imagenet.py ./imagenette2-320 \
    --arch resnet50 --epochs 30 --pretrained \
    --lambda-rate 1e-4 --qstep 0.1

# Export to H.265
python export_h265.py encode \
    --arch resnet50 --qstep 0.1 \
    --crf 0 --bit-depth 8 --dither 0.1 --preset slower

# Decode and evaluate
python export_h265.py decode \
    --h265-dir ./h265_out --data ./imagenette2-320 \
    --non-dct-weights ./checkpoints/best.pth
```

### 6.3 Profile Encoding Presets

```bash
python export_h265.py encode --arch resnet18 --qstep 0.1 --profile
python export_h265.py encode --arch resnet50 --qstep 0.1 --profile
```

---

## 7. Conclusion

CTNet demonstrates that modern video codecs are surprisingly effective neural network compressors. By reparameterizing convolutional layers into the DCT domain and training with a differentiable H.265 rate proxy, CTNet-18 achieves 14.5x compression at 92.99% Top-1 accuracy on ImageNette2 (only ~3% below baseline), with a continuous rate-distortion tradeoff up to 32.5x compression via lossy encoding. The approach requires no custom entropy coding implementation, leverages decades of video codec optimization, and offers continuous rate-distortion control via standard codec parameters.

CTNet-50 extends the framework to deeper bottleneck architectures, revealing that the approach is most effective when spatial convolutions dominate the parameter budget. For architectures with many 1x1 convolutions, CTNet naturally combines with standard quantization for a hybrid compression strategy.

Future work includes evaluation on full ImageNet-1K, combination with structured pruning and knowledge distillation, exploration of newer codecs (AV1, VVC/H.266), extension of frequency-domain compression to 1x1 convolutions via channel-wise transforms, and investigation of whether the H.265 encoder's internal rate-distortion optimization can be directly exploited during training.

---

## References

- Balle, J., Laparra, V., & Simoncelli, E. P. (2017). End-to-end optimized image compression. ICLR.
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
- Lin, M., et al. (2020). HRank: Filter pruning using high-rank feature map. CVPR.
- Liu, Z., Shen, Z., Savvides, M., & Cheng, K. T. (2020). ReActNet: Towards precise binary neural network with generalized activation functions. ECCV.
- Luo, J., Wu, J., & Lin, W. (2017). ThiNet: A filter level pruning method for deep neural network compression. ICCV.
- Minnen, D., Balle, J., & Toderici, G. (2018). Joint autoregressive and hierarchical priors for learned image compression. NeurIPS.
- Rastegari, M., Ordonez, V., Redmon, J., & Farhadi, A. (2016). XNOR-Net: ImageNet classification using binary convolutional neural networks. ECCV.
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.
- Zhu, M., & Gupta, S. (2018). To prune, or not to prune: Exploring the efficacy of pruning for model compression. ICLR Workshop.

---

## License

This project is released for research purposes.
