# Quantization: Math to Code

Decoding a large language model is limited by memory bandwidth: every generated token requires reading all of the model's weights, and a growing KV cache, out of GPU memory. Quantization attacks that bottleneck directly. Store each weight in 4 or 8 bits instead of 16, and there are fewer bytes to move, so decode gets faster and more memory is left for the KV cache. The compute is usually still done in FP16 or BF16 after dequantization (weight-only quantization, written "WxA16"), or in INT8 or FP8 when activations are quantized too ("WxAx").

This section develops quantization **math first, then the code that implements it**. Every quantizer you will meet is "just" a way of mapping high-precision numbers to a few bits while controlling the distortion that mapping introduces, and the methods differ entirely in *which math* they use to pick the mapping. We start with the ten-line core, affine and symmetric quantization, then work through granularity, distortion, and the kernels that turn fewer bits into real speed, before walking the history of methods from LLM.int8() to TurboQuant and closing with practical guidance. Examples are in C, CUDA, and Rust ([Burn](https://burn.dev/)), targeting the 2025/2026 inference stack.

This section builds on the chapter opening, [LLM Systems & Inference](../README.md), which explains why decode is bandwidth-bound and why the KV cache dominates memory. The math draws on [Linear Algebra](../../../mathematics-for-computing/03-linear-algebra/README.md) (Hessians, Cholesky factors, orthogonal and Hadamard matrices) and [Information Theory](../../../mathematics-for-computing/11-information-theory/README.md) (rate-distortion) from *Mathematics for Computing*, and the code assumes basic C and CUDA. Standalone, compilable examples live in [`code/`](code/).

> **Learning objectives**
> - Derive and implement affine and symmetric quantization, and explain how granularity trades metadata for accuracy.
> - Write down the distortion of a uniform quantizer and explain how calibration chooses the clipping threshold.
> - Explain why a quantizer only pays off with a dequant-fused GEMM kernel.
> - For each major method (GPTQ, AWQ, SmoothQuant, NF4, GGUF k-quants, FP8, QuIP#, TurboQuant), state the distribution it assumes, the error it minimizes, and the extra state it stores.
> - Choose a quantizer for a given goal and hardware, including KV-cache quantization.

## Quantization as lossy compression

A quantizer is a function $Q: \mathbb{R} \to \{l_0, \ldots, l_{2^b - 1}\}$ mapping reals to $2^b$ levels, plus a decoder $D$ mapping levels back to reals. Its **distortion** is $\mathbb{E}[(x - D(Q(x)))^2]$, and rate-distortion theory from [Information Theory](../../../mathematics-for-computing/11-information-theory/README.md) says the achievable distortion at $b$ bits depends on the *entropy* of the source.

Quantization is therefore **lossy compression of a known distribution**. Every method is a bet about what the weights or activations look like (uniform, Gaussian, or sphere-uniform after a rotation), paired with a quantizer that is optimal for *that* assumption. The whole game has three moves: **make the source easy** (rotate it to be Gaussian or uniform), **spend your levels where the mass is** (Lloyd–Max quantizers, quantile codebooks), and **protect what matters** (salient weights, inner products). Keep those three moves in mind; every method later in the section is some combination of them.

A useful habit as you read is to write down three things for each quantizer: what distribution it assumes, what error metric it minimizes (mean squared error? inner-product error?), and what extra state it stores (scales, zero-points, codebooks, rotations).

## Affine and symmetric quantization

The core math maps a real range $[\beta, \alpha]$ onto integer codes $[q_\text{min}, q_\text{max}]$, for example $[-128, 127]$ for INT8.

**Affine (asymmetric)** quantization handles skewed ranges, such as post-ReLU activations, by storing both a scale and an integer zero-point:

```
s = (α − β) / (q_max − q_min)          # scale (Δ, the step size)
z = round(q_min − β / s)                # zero-point (integer)
quantize:    q = clamp(round(x / s) + z, q_min, q_max)
dequantize:  x̂ = s · (q − z)
```

**Symmetric** quantization sets $z = 0$. It is used for weights, which are roughly zero-mean, because it makes the GEMM cheaper: with no zero-point there are no cross-terms to correct.

```
s = max(|x|) / q_max          # q_max = 127 for INT8
q = clamp(round(x / s), −127, 127)
x̂ = s · q
```

To see the numbers, take the tiny tensor $x = [0.5, -1.27, 0.02]$. Its absmax is 1.27, so $s = 1.27/127 = 0.01$, the codes are $q = [50, -127, 2]$, and dequantizing gives back $[0.5, -1.27, 0.02]$ exactly. Real values rarely land on multiples of $s$, so in general each $\hat{x}$ is off by at most $s/2$; the rest of the section is about keeping that error small where it matters.

The translation into C is direct. This is symmetric per-tensor quantization, with the scale computed once over the tensor:

```c
// scale = absmax(x) / 127  (computed once over the tensor)
int8_t quantize(float x, float scale)   { 
    int32_t q = (int32_t)lrintf(x / scale);   // round-to-nearest-even
    return (int8_t)(q < -127 ? -127 : q > 127 ? 127 : q);
}
float   dequantize(int8_t q, float scale) { return scale * (float)q; }
```

The full per-tensor kernel, an absmax reduction followed by quantize and dequantize plus an INT8 dot product using `dp4a`, is in [`code/symmetric_quant.cu`](code/symmetric_quant.cu). Implement this yourself from the formula before touching a library. It is ten lines, and everything else in the section is a refinement of it.

## Granularity: per-tensor, channel, group, and block

The first refinement is to stop sharing one scale across an entire tensor. Across all the granularities, the *only* thing that changes is **how many $(s, z)$ pairs you keep**, trading metadata for accuracy:

| Granularity | One scale per | Metadata | Use |
|-------------|---------------|----------|-----|
| Per-tensor | whole tensor | tiny | activations, fast path |
| Per-channel (per-row) | output channel | one vector | weights (W8A8) |
| Per-group | block of K (e.g. 128) along in-dim | K⁻¹ of weights | INT4 weights (GPTQ/AWQ default `group_size=128`) |
| Per-block | block of 32–256 | a few % | GGUF k-quants, NF4 (block 64) |

Finer granularity means more scales and less distortion, because each block clips to its *own* range. This is why INT4 group quantization works at all: a 128-wide group has a far tighter dynamic range than a 4096-wide row, so one outlier spoils only its own group.

## Distortion, clipping, and calibration

Granularity decides how many ranges you have. Calibration decides what each range should be. For a uniform quantizer with step $\Delta$, the **granular noise** (the rounding error) is uniform on $[-\Delta/2, \Delta/2]$, so its variance is $\Delta^2/12$. If you clip at threshold $c$, the step is $\Delta = 2c / 2^b$, and the total distortion has two parts:

$$D(c) = \underbrace{\text{(overload/clipping error for } |x| > c)}_{\text{clipping}} + \underbrace{\frac{\Delta^2}{12}}_{\text{granular}}$$

A smaller $c$ shrinks the granular noise but raises the clipping error. **Calibration** picks $c$, and there are three common strategies. **Min-max** sets $c = \max|x|$: there is no clipping, but a single outlier blows up $\Delta$ for everything else. **Percentile** sets $c$ to, say, the 99.9th percentile, deliberately clipping the tail. **MSE-optimal** calibration runs a grid or golden-section search over $c$ to minimize the measured $D(c)$, which is what AWQ and GPTQ calibration and `compressed-tensors` do.

Stated abstractly, this is a [rate-distortion](../../../mathematics-for-computing/11-information-theory/README.md) problem: at a fixed rate $b$, choose the quantizer that minimizes expected distortion for the source distribution.

## Where the speed comes from: dequant-fused GEMM

All of this saves memory, but memory saved is not time saved unless the **kernel** exploits it. The pattern for weight-only INT4 (`W4A16`) has three steps. First, load the 4-bit weights from HBM, using a quarter of the bandwidth FP16 would need. Second, **dequantize in registers or shared memory** to FP16 just before the tensor-core matmul. Third, accumulate in FP32.

This is what [Marlin](https://github.com/vllm-project/vllm) (W4A16 on Ampere and Ada) and **Machete** (Hopper) do in vLLM, and what CUTLASS's mixed-input GEMM provides. For `W8A8` the matmul itself runs in INT8 (`mma.sync` or `dp4a`), and you **dequantize in the epilogue**: $\text{out} = (s_x \cdot s_w) \cdot \text{int32\_acc}$. The `dp4a` dot product in [`code/symmetric_quant.cu`](code/symmetric_quant.cu) shows this.

> **Warning:** A quantizer is only as good as its kernel. INT4 with a slow Python dequantization loop is *slower* than FP16. Always check that a fused kernel (Marlin, Machete, CUTLASS, tinygemm) exists for your format on your GPU, and profile the dequant-plus-GEMM path, not just the quantizer.

## How quantization evolved

With the core math in place, we can look at the methods themselves. They are not a flat list but a *sequence of discoveries*, each fixing the failure of the one before, and reading them in order is the fastest way to understand why each one exists. The table summarizes the arc; the subsections that follow derive each method.

| When | Method (paper) | The problem it hit | What changed |
|------|----------------|--------------------|--------------|
| 2018 | **Integer-only PTQ/QAT** ([Jacob et al., 1712.05877](https://arxiv.org/abs/1712.05877)) | Floats are expensive on edge HW | Affine INT8, per-tensor/per-channel scales: the baseline everything refines |
| Aug 2022 | **LLM.int8()** ([2208.07339](https://arxiv.org/abs/2208.07339)) | Naive INT8 *collapses* on >6.7B models | Found **activation outlier channels**; keep those few in FP16, rest INT8 (mixed-precision decomposition) |
| Oct 2022 | **GPTQ** ([2210.17323](https://arxiv.org/abs/2210.17323)) | Want 3–4-bit *weights*, not just 8 | Second-order, **error-compensating** PTQ (minimize output error, not weight error), so INT4 weights become usable |
| Nov 2022 | **SmoothQuant** ([2211.10438](https://arxiv.org/abs/2211.10438)) | Outliers still block full **W8A8** | **Migrate** outlier difficulty from activations into weights, so both fit INT8 |
| May 2023 | **QLoRA / NF4** ([2305.14314](https://arxiv.org/abs/2305.14314)) | Can't *fine-tune* a quantized model cheaply | **Quantile codebook** (NormalFloat) + double-quant: fine-tune a 4-bit base on one GPU |
| Jun 2023 | **AWQ** ([2306.00978](https://arxiv.org/abs/2306.00978)) | GPTQ needs the Hessian; want simpler W4 | **Protect ~1% salient channels** (found via *activation* magnitude) by scaling: best W4A16 quality/speed |
| 2023–24 | **FP8 (E4M3/E5M2)** | INT8 wastes range on near-zero weights | A float's **exponent** handles dynamic range; native Hopper/Ada tensor cores give W8A8 at near-INT8 speed, tiny loss |
| Feb 2024 | **QuIP#** ([2402.04396](https://arxiv.org/abs/2402.04396)) | 2-bit is too lossy: outliers again | **Rotate outliers away** (Hadamard incoherence) + E8-lattice vector quant: usable 2-bit |
| Jun 2024 | **QJL** ([2406.03482](https://arxiv.org/abs/2406.03482)) | KV cache can't be calibrated (unseen tokens) | **Data-oblivious** JL sign-sketch for keys: low-bit attention with unbiased scores |
| Apr 2025 | **TurboQuant** ([2504.19874](https://arxiv.org/abs/2504.19874)) | Want optimal, online, no-calibration | Rotation gives **known sphere marginals**, then per-coordinate Lloyd–Max; provably near-optimal at 2.5–3.5 bit KV |

Two through-lines run through the table. The first is that **outliers are the enemy**: activation outlier channels are what make naive INT8 fail, and successive methods *isolate* them (LLM.int8()), *migrate* them (SmoothQuant), *protect* them (AWQ), and finally *rotate them away* (QuIP#, TurboQuant). The second is that the field moved from **data-dependent calibration** toward **data-oblivious, provable** schemes, because the KV cache, the new bottleneck, cannot be calibrated on tokens you have not generated yet.

### GPTQ: second-order, error-compensating

GPTQ's assumption is that what matters is not the error in each weight but the perturbation to the layer's output. For a linear layer with calibration inputs $X$, it minimizes $\lVert WX - \hat{W}X \rVert^2$, whose Hessian is $H = 2XX^\top$. When it quantizes weight column $q$, it greedily compensates the *not-yet-quantized* columns by pushing the error into them:

```
err  = (w_q − quant(w_q)) / [H⁻¹]_qq
W   −= err · H⁻¹[:, q]          # push the error into remaining weights
```

To make this practical, GPTQ fixes the column order, uses the **Cholesky factor of $H^{-1}$** for numerical stability, and applies updates in lazy batches. In pseudocode:

```python
H = 2 * X @ X.T + damp * I            # damp ~ 1% of mean(diag) for stability
Hinv = cholesky_inverse(H)            # upper-triangular Cholesky of H^-1
for j in range(n_cols):               # left-to-right
    w = W[:, j]
    q = quant(w, scale[j])            # 3/4-bit symmetric, per-group scale
    err = (w - dequant(q)) / Hinv[j, j]
    W[:, j+1:] -= err[:, None] * Hinv[j, j+1:][None, :]   # compensate
    Q[:, j] = q
```

In the current stack, GPTQModel and AutoGPTQ produce `compressed-tensors` or GPTQ checkpoints, which vLLM serves with Marlin.

### AWQ: protect the salient channels

AWQ starts from an empirical observation: about 1% of weight channels matter disproportionately, and you find them by **activation** magnitude, not weight magnitude. Its move is to scale those channels up before quantizing, so they get more effective resolution, and to fold the inverse scale into the activations so the product is mathematically unchanged:

$$\hat{W} = W \operatorname{diag}(s), \quad \hat{X} = \operatorname{diag}(s)^{-1} X \quad\Longrightarrow\quad \hat{X}\hat{W} = XW$$

The per-input-channel scale is $s_j = (\text{activation\_scale}_j)^\alpha$, with $\alpha \in [0, 1]$ chosen by minimizing the layer's output MSE on calibration data; then $\hat{W}$ is quantized to INT4. The search is short:

```python
act_scale = X.abs().mean(dim=0)                  # per-channel activation magnitude
def loss(alpha):
    s  = act_scale.pow(alpha).clamp(min=1e-4)
    Wq = quant_dequant(W * s, n_bits=4, group=128) / s
    return ((X @ W.T) - (X @ Wq.T)).pow(2).mean()
alpha = golden_section_search(loss, 0.0, 1.0)    # ~20 evals
```

AutoAWQ produces these checkpoints, and AWQ Marlin kernels serve them in vLLM and TensorRT-LLM. AWQ is often the best quality-for-speed choice for W4A16.

### SmoothQuant: migrate the difficulty for W8A8

SmoothQuant targets full W8A8, where both operands are quantized. Its assumption is that activations have hard-to-quantize outlier channels while weights are easy, so it **migrates** the outliers from activations into weights with a per-channel smoothing factor $s$:

$$Y = \left(X \operatorname{diag}(s)^{-1}\right)\left(\operatorname{diag}(s) W\right), \qquad s_j = \frac{\max|X_j|^\alpha}{\max|W_j|^{1-\alpha}},\ \alpha \approx 0.5$$

The first equation is an identity; you then quantize both factors. After smoothing, $\hat{X} = X \operatorname{diag}(s)^{-1}$ and $\hat{W} = \operatorname{diag}(s) W$ both fit comfortably in INT8, enabling a full W8A8 matmul. SmoothQuant ships in TensorRT-LLM and in `compressed-tensors` (`w8a8`).

### NF4: a quantile codebook for Gaussian weights

NF4, introduced with QLoRA, assumes the weights are approximately $\mathcal{N}(0, \sigma)$. Under that assumption, the information-optimal 4-bit code with *equiprobable bins* puts its levels at the **quantiles** of a normal distribution, which gives the format its name, NormalFloat. The 16 levels are 8 negative, a zero, and 7 positive, each a fixed $\mathcal{N}(0, 1)$ quantile, and each block of weights is scaled by its absmax:

```
codebook = [ Φ⁻¹( (i + 0.5)/16 ) normalized to [−1, 1] ]   # 16 fixed levels
block (64 weights):  s = absmax(block);  q = argmin_i |w/s − codebook[i]|
dequant:             ŵ = s · codebook[q]
```

NF4 adds **double quantization**: the FP32 block scales $s$ are themselves quantized to 8 bits (in blocks of 256) with a second-level scale, saving about 0.37 bits per parameter. A full C implementation is in [`code/nf4_quant.c`](code/nf4_quant.c). In practice NF4 comes from bitsandbytes (`load_in_4bit`, `bnb_4bit_quant_type="nf4"`), the default for QLoRA fine-tuning.

### GGUF k-quants: hierarchical block scales

llama.cpp's k-quants assume that per-block scales themselves compress well, and build a hierarchy. **`Q4_K`** packs a **super-block of 256** weights as 8 sub-blocks of 32. Each sub-block has a 6-bit scale $d$ and a 6-bit min $m$, and the eight $d$s and eight $m$s share two FP16 super-scales:

```
d_sub = d_super · scale6[b]          # b = sub-block index
m_sub = m_super · min6[b]
ŵ = d_sub · q4 − m_sub               # q4 ∈ [0, 15]
```

The result is an effective ~4.5 bits per weight with no calibration. `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q2_K`, and the newer **IQ** (i-quant) codebook variants such as `IQ4_NL` and `IQ2_XS` all follow this hierarchical-block idea. A faithful `Q4_K`-style dequantizer in C is the `dequant_q4k_block` function in [`code/nf4_quant.c`](code/nf4_quant.c). These formats run in llama.cpp, Ollama, and LM Studio, on CPU, Metal, and CUDA.

### FP8: native low-precision floats

FP8 changes the representation rather than the codebook. Its assumption is that a float's exponent handles dynamic range better than a fixed-point integer, which wastes much of its range on near-zero weights. **E4M3** has 1 sign bit, 4 exponent bits, and 3 mantissa bits with bias 7, so a normal value is $(-1)^s \cdot 2^{e-7} \cdot (1 + m/8)$; its maximum is about 448 and it has no infinities. **E5M2** has more range and less precision and is used for gradients. To use E4M3, apply a per-tensor scale and cast:

```
x_scaled = x / amax * 448.0          # fit E4M3 dynamic range
fp8 = cast_e4m3(x_scaled)            # hardware instruction on Hopper/Ada
```

Hopper (H100) and Ada (L40S) tensor cores do FP8 matmuls natively, giving near-INT8 throughput with far less accuracy loss. TensorRT-LLM, vLLM (`--quantization fp8`), and torchao (`float8`) all support it, and it is the default for serving the latest large models when an H100 is available.

FP8 is one member of a family of float formats, in which exponent bits buy range and mantissa bits buy precision. The table places it among its relatives:

| Format | Bits (S/E/M) | Used for |
|--------|--------------|----------|
| **FP32** | 1/8/23 | Master weights, accumulators |
| **TF32** | 1/8/10 | Ampere+ matmul mode, internal only |
| **BF16** | 1/8/7 | Default training/serving dtype; FP32's range |
| **FP16** | 1/5/10 | Older default; narrow range, needs loss scaling |
| **FP8 E4M3** | 1/4/3 | Weights + activations (DeepSeek-V3: `fmt: e4m3`, 128×128 blocks) |
| **FP8 E5M2** | 1/5/2 | Gradients (range over precision) |
| **MXFP8 / MXFP6 / MXFP4** | E4M3·E5M2 / E3M2·E2M3 / E2M1 | OCP microscaling: 32-element blocks share a power-of-two scale (gpt-oss ships MXFP4) |
| **NVFP4** | 1/2/1 | Blackwell; 16-element blocks with an FP8 E4M3 scale, finer than MXFP4 |
| **NF4** | 4-bit codebook | Not a true float: 16 fixed Gaussian-quantile levels (see [NF4](#nf4-a-quantile-codebook-for-gaussian-weights)) |

### QuIP#: incoherence plus a lattice codebook

At 2 bits, outliers defeat even careful per-group scaling. QuIP#'s assumption is that outliers can be *rotated away*. It multiplies the weights and the Hessian by **random orthogonal (Hadamard) matrices** to make them *incoherent*, with mass spread evenly and no spikes, and then vector-quantizes with the **E8 lattice** codebook, the densest lattice in 8 dimensions, followed by light fine-tuning. The result is usable 2-bit weights. QuIP# is the conceptual bridge to TurboQuant: **rotation makes the distribution predictable, and then you quantize optimally.**

### TurboQuant: provably near-optimal and data-oblivious

[TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, and Mirrokni, Google and NYU, 2025) is the current high-water mark for *online, calibration-free* quantization, especially of the KV cache. Its derivation takes four steps, and each builds on the last.

**Step 1: random rotation.** Apply a randomized Hadamard transform $R = \frac{1}{\sqrt{d}} H D$, where $D$ is a random $\pm 1$ diagonal and $H$ is the Hadamard matrix. Computing $Rx$ costs $O(d \log d)$ with a Fast Walsh–Hadamard Transform, *not* $O(d^2)$. For any input vector, $Rx$ is distributed like a uniform random vector on the sphere of the same norm.

**Step 2: known marginals.** On the sphere in $\mathbb{R}^d$, each coordinate follows a **concentrated Beta distribution**, with density proportional to $(1 - t^2)^{(d-3)/2}$, which is approximately $\mathcal{N}(0, 1/d)$ for large $d$, and distinct coordinates are *nearly independent* in high dimension. Crucially, this marginal is **the same for every vector and known in advance**, so no calibration data is needed.

**Step 3: an optimal scalar quantizer per coordinate.** Because the marginal is known, you can apply the **Lloyd–Max optimal scalar quantizer** for that distribution to every coordinate independently. This provably reaches the MSE distortion-rate bound within a constant factor of about **2.7**, across *all* bit-widths and dimensions.

**Step 4: unbiased inner products.** A pure MSE quantizer is *biased* for inner products: it systematically shrinks $\langle q, k \rangle$, which is exactly what attention computes. TurboQuant fixes this with a **two-stage** scheme. It MSE-quantizes, then takes the **residual** and applies a **1-bit QJL** ([Quantized Johnson–Lindenstrauss](https://arxiv.org/abs/2406.03482)) transform, storing $\operatorname{sign}(\langle \text{residual}, g_i \rangle)$ for random Gaussian projections $g_i$. Summing the two stages gives an **unbiased** inner-product estimate.

The results for the KV cache are that **3.5 bits per channel** is quality-neutral and **2.5 bits per channel** is marginal, with no calibration and computed online as tokens stream. That data-oblivious property is exactly what KV caching needs, because you cannot calibrate on tokens you have not seen.

A minimal end-to-end TurboQuant encoder and decoder in C is in [`code/turboquant.c`](code/turboquant.c). It implements the random-sign plus iterative FWHT rotation, a per-coordinate uniform quantizer (a stand-in for the Lloyd–Max levels, with a table of the Gaussian-optimal points), and the 1-bit QJL residual sign for unbiased inner products. The rotation is the whole trick, and it fits in a dozen lines:

```c
// The rotation is the whole trick. In-place Fast Walsh–Hadamard Transform, O(d log d):
void fwht(float* a, int n) {            // n must be a power of two
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float u = a[j], v = a[j + len];
                a[j] = u + v;  a[j + len] = u - v;
            }
    float inv = 1.0f / sqrtf((float)n);  // orthonormal scaling
    for (int i = 0; i < n; i++) a[i] *= inv;
}
```

## Putting it together: KV-cache quantization

TurboQuant's focus on the KV cache is no accident. The KV cache is the prime quantization target because it grows without bound and is read on every decode step, as the [chapter opening](../README.md) explains. Several techniques apply. KVQuant and KIVI use **per-token and per-channel scales**, having found that keys quantize better per channel and values per token. **QJL** applies a data-oblivious sign sketch to the keys, giving low-bit attention with unbiased scores. **TurboQuant** is the online, no-calibration sweet spot at 2.5 to 3.5 bits per channel. Whichever you use, storing the cache in INT4, INT8, or FP8 frees memory for longer contexts and bigger batches.

## Choosing a quantizer

The history and the math reduce to practical guidance. For the 2025/2026 stack, this table matches each goal to a method, its bit-width, and the reason it fits:

| Goal | Use | Bits | Why |
|------|-----|------|-----|
| Best W4 quality, GPU serving | **AWQ** or **GPTQ** + Marlin | 4 (W4A16) | Salient-channel / second-order; fast fused kernel |
| Max throughput on H100 | **FP8** (E4M3) | 8 (W8A8) | Native tensor cores, tiny accuracy loss |
| Big-model W8A8 without retrain | **SmoothQuant** | 8 | Migrates activation outliers |
| Fine-tune a quantized model | **NF4** (QLoRA) | 4 | Quantile-optimal for Gaussian weights |
| CPU / laptop / edge | **GGUF k-quants** (`Q4_K_M`) | ~4.5 | No GPU, hierarchical block scales |
| Aggressive 2-bit weights | **QuIP#** | 2 | Incoherence + E8 lattice |
| KV cache, online, no calibration | **TurboQuant** / QJL | 2.5–3.5 | Data-oblivious, near-optimal distortion |

## Quantization in Rust with Burn

Everything so far has been C, CUDA, or Python pseudocode. In Rust, [Burn](https://burn.dev/) has a backend-agnostic quantization API in `burn::tensor::quantization`. Burn is pre-1.0 and evolving fast, so pin a version. Its core types are `QuantScheme` (the configuration), `QuantValue` (`Q8S`, `Q4S`, and `Q2S` for fixed-point symmetric; `E4M3`, `E5M2`, and `E2M1` for floats), `QuantLevel` (per-tensor or per-block granularity), `Calibration` (min-max), and `QuantMode` (symmetric). The simplest path is **dynamic quantization**, which calibrates on the fly:

```rust
use burn::tensor::quantization::{QuantScheme, QuantValue};
use burn::tensor::Tensor;

// 8-bit symmetric, dynamic (min-max calibration computed from the tensor):
let scheme = QuantScheme::default().with_value(QuantValue::Q8S);
let q = tensor.clone().quantize_dynamic(&scheme);   // -> quantized Tensor
let back = q.dequantize();                           // -> f32 Tensor

// 4-bit, per-block granularity for weight-only quantization:
let scheme = QuantScheme::default()
    .with_value(QuantValue::Q4S)
    .with_level(QuantLevel::block([128]));           // group_size = 128
```

A runnable example that quantizes, dequantizes, and measures the MSE on the NdArray backend with autodiff is in [`code/burn_quantize.rs`](code/burn_quantize.rs). For production Rust inference, GGUF k-quant models also run through llama.cpp bindings (`llama-cpp-2`) and candle's `quantized` module. The full cross-language runtime landscape that consumes these formats is mapped in [Inference Frameworks](../frameworks/README.md), including a pure-Zig SIMD INT8 dot product, [`frameworks/code/quant_dot.zig`](../frameworks/code/quant_dot.zig), which is the CPU twin of [`code/symmetric_quant.cu`](code/symmetric_quant.cu).

## Where this leads

The theory behind this section comes from *Mathematics for Computing*: rate-distortion, entropy, and quantile coding in [Information Theory](../../../mathematics-for-computing/11-information-theory/README.md) explain why NF4 and Lloyd–Max quantizers are optimal; Hessians, Cholesky factors, and orthogonal and Hadamard matrices in [Linear Algebra](../../../mathematics-for-computing/03-linear-algebra/README.md) underlie GPTQ, QuIP#, and TurboQuant's rotations; and the Beta and Gaussian marginals and concentration results in [Probability and Statistics](../../../mathematics-for-computing/07-probability-statistics/README.md) describe TurboQuant's coordinate distribution. On the systems side, what quantization buys you in KV-cache capacity and throughput is the subject of [LLM Systems & Inference](../README.md) and [Serving & Load](../serving-and-load/README.md), and how quantized checkpoints are recognized and loaded is covered in [Model Loading: Hub to GPU](../model-loading/README.md).

The same ideas appear across the industry: NVIDIA builds FP8 and INT4 kernels in CUTLASS and TensorRT-LLM on tensor-core `mma`; Google and DeepMind published TurboQuant, QJL, and related rotation-based research; Anthropic and OpenAI cut serving cost through weight and KV-cache quantization; Neural Magic, now part of the vLLM effort, built `compressed-tensors` and the Marlin and Machete kernels; and Meta's models anchor the GGUF and llama.cpp edge-quantization ecosystem.

### Further reading

- Foundational papers, all free on arXiv: [LLM.int8()](https://arxiv.org/abs/2208.07339) (2208.07339), [GPTQ](https://arxiv.org/abs/2210.17323) (2210.17323), [SmoothQuant](https://arxiv.org/abs/2211.10438) (2211.10438), [AWQ](https://arxiv.org/abs/2306.00978) (2306.00978), [QLoRA/NF4](https://arxiv.org/abs/2305.14314) (2305.14314), [QuIP#](https://arxiv.org/abs/2402.04396) (2402.04396), [QJL](https://arxiv.org/abs/2406.03482) (2406.03482), [TurboQuant](https://arxiv.org/abs/2504.19874) (2504.19874)
- The current stack: [llama.cpp / GGUF](https://github.com/ggml-org/llama.cpp) k-quants, [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (NF4, LLM.int8()), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) and [GPTQModel](https://github.com/ModelCloud/GPTQModel), [`compressed-tensors`](https://github.com/neuralmagic/compressed-tensors) with the [Marlin/Machete](https://github.com/vllm-project/vllm) kernels in vLLM, [torchao](https://github.com/pytorch/ao), [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) FP8, and [Burn](https://burn.dev/) for Rust

## Key terms

- **scale and zero-point**: the parameters $s$ and $z$ of an affine quantizer, mapping a real range onto integer codes.
- **symmetric quantization**: affine quantization with $z = 0$, used for roughly zero-mean weights to avoid GEMM cross-terms.
- **granularity**: how many scales are kept (per tensor, channel, group, or block), trading metadata for distortion.
- **granular noise**: the rounding error of a uniform quantizer, with variance $\Delta^2/12$.
- **calibration**: choosing a clipping threshold or other quantizer parameters from sample data.
- **W4A16 / W8A8**: notation for weight and activation bit-widths, here 4-bit weights with 16-bit activations and 8-bit weights with 8-bit activations.
- **dequant-fused GEMM**: a kernel that dequantizes low-bit weights in registers or shared memory immediately before the matrix multiply.
- **incoherence**: the property, produced by random rotation, that a vector's mass is spread evenly with no outlier coordinates.
- **Lloyd–Max quantizer**: the MSE-optimal scalar quantizer for a known distribution.

## Summary

- Quantization is lossy compression of a known distribution; every method bets on what the data looks like and quantizes optimally for that bet.
- The win is memory bandwidth: fewer bits per weight mean fewer bytes per decode step, but only if a fused dequant-plus-GEMM kernel exists for the format.
- Finer granularity and better calibration reduce distortion by giving each block its own, well-chosen range.
- Outliers are the enemy: methods have successively isolated, migrated, protected, and rotated them away.
- The frontier (QuIP#, TurboQuant) is incoherence via random rotation, which makes coordinate marginals known so a near-optimal scalar quantizer needs no calibration data, exactly what online KV-cache quantization requires.

## Review questions

1. Quantize $x = [0.5, -1.27, 0.02]$ to symmetric per-tensor INT8. Then explain what would happen to the small values if one entry were 127.0 instead.
   <details><summary>Answer</summary>Absmax is 1.27, so $s = 0.01$ and $q = [50, -127, 2]$, which dequantizes exactly. If one entry were 127.0, absmax min-max calibration would give $s = 1.0$, and 0.5 and 0.02 would round to codes 0 or 1 (with round-to-nearest-even, 0.5 rounds to 0), destroying them. This is the outlier problem, which finer granularity, clipping, or rotation addresses.</details>
2. Why does INT4 group quantization with `group_size=128` work when per-row INT4 does not?
   <details><summary>Answer</summary>Each group gets its own scale and therefore clips to its own range. A 128-wide group has a far tighter dynamic range than a 4096-wide row, so the step size $\Delta$, and with it the granular noise $\Delta^2/12$, is much smaller, and an outlier only degrades its own group.</details>
3. Why can INT4 weights be slower than FP16, and what makes them fast?
   <details><summary>Answer</summary>The savings are in memory traffic, and they are only realized if the kernel loads the packed 4-bit weights and dequantizes them in registers or shared memory right before the tensor-core matmul. A separate Python dequantization loop adds work and memory traffic, making it slower than FP16. Fused kernels such as Marlin, Machete, CUTLASS mixed-input GEMM, or tinygemm deliver the speedup.</details>
4. Contrast how SmoothQuant and AWQ use per-channel scaling.
   <details><summary>Answer</summary>Both rely on the identity $(X\operatorname{diag}(s)^{-1})(\operatorname{diag}(s)W) = XW$. SmoothQuant uses it to move outlier difficulty from activations into weights, with $s_j = \max|X_j|^\alpha / \max|W_j|^{1-\alpha}$, so that both fit INT8 for W8A8. AWQ uses it to give roughly 1% of salient channels, identified by activation magnitude, more effective resolution before quantizing weights to INT4, choosing the exponent $\alpha$ by output MSE.</details>
5. Why is the KV cache hard to quantize with calibration-based methods, and how does TurboQuant avoid needing calibration?
   <details><summary>Answer</summary>The KV cache holds keys and values of tokens generated at run time, so you cannot calibrate on them in advance. TurboQuant first applies a random Hadamard rotation, after which any vector looks uniform on the sphere and each coordinate's marginal is a known concentrated Beta distribution (about $\mathcal{N}(0, 1/d)$). With the distribution known a priori, a Lloyd–Max quantizer per coordinate is near-optimal, and a 1-bit QJL sketch of the residual makes inner-product estimates unbiased.</details>
6. What extra state does each of NF4 and `Q4_K` store, and roughly how many bits per weight does `Q4_K` cost?
   <details><summary>Answer</summary>NF4 stores a fixed 16-level codebook and one absmax scale per block of 64, with the scales themselves double-quantized to 8 bits in blocks of 256, saving about 0.37 bits per parameter. `Q4_K` stores, per 256-weight super-block, a 6-bit scale and 6-bit min for each of 8 sub-blocks of 32, plus two FP16 super-scales, for about 4.5 bits per weight.</details>
