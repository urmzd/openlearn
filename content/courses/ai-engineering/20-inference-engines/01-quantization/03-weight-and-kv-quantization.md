# Weight and KV quantization methods

This part of [Quantization: Math to Code](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Distinguish weight compression from cache compression and their error sources.

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

## Summary

Distinguish weight compression from cache compression and their error sources. State the assumptions before applying the result to a different workload.

## Review questions

1. Does weight-only quantization reduce cache bytes per token?

   <details><summary>Answer</summary>

   No. KV dtype, head structure, retained context, and cache layout determine those bytes.

   </details>

[Chapter contents](README.md) · [Next: Choosing and validating quantization](04-choosing-and-validating-quantization.md)
