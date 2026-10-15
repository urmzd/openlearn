# Quantization: Math → Code

A deep dive into LLM quantization, organized as **math first, then the code that implements it**. Every quantizer below is "just" a way of mapping high-precision numbers to a few bits while controlling the distortion it introduces -- the differences are entirely in *which math* they use to pick the mapping. Examples are in C, CUDA, and Rust ([Burn](https://burn.dev/)), targeting the 2025/2026 inference stack.

> Parent topic: [LLM Systems & Inference](../). Standalone, compilable examples live in [`code/`](code/).

## Overview

- **Foundational papers** (all free on arXiv): [LLM.int8()](https://arxiv.org/abs/2208.07339) (2208.07339), [GPTQ](https://arxiv.org/abs/2210.17323) (2210.17323), [SmoothQuant](https://arxiv.org/abs/2211.10438) (2211.10438), [AWQ](https://arxiv.org/abs/2306.00978) (2306.00978), [QLoRA/NF4](https://arxiv.org/abs/2305.14314) (2305.14314), [QuIP#](https://arxiv.org/abs/2402.04396) (2402.04396), [QJL](https://arxiv.org/abs/2406.03482) (2406.03482), [**TurboQuant**](https://arxiv.org/abs/2504.19874) (2504.19874)
- **The current stack**: [llama.cpp / GGUF](https://github.com/ggml-org/llama.cpp) k-quants, [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) (NF4, LLM.int8()), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) / [GPTQModel](https://github.com/ModelCloud/GPTQModel), [`compressed-tensors`](https://github.com/neuralmagic/compressed-tensors) + [Marlin/Machete](https://github.com/vllm-project/vllm) kernels in vLLM, [torchao](https://github.com/pytorch/ao), [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) FP8, and [Burn](https://burn.dev/) for Rust
- **Prerequisites**: [LLM Systems & Inference](../), [Linear Algebra](../../../math/03-linear-algebra/), [Information Theory](../../../math/11-information-theory/) (rate-distortion), C/CUDA basics
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- Quantization is **lossy compression of a known distribution**. Every method is a bet about what the weights/activations look like -- uniform, Gaussian, sphere-uniform after rotation -- and an optimal quantizer for *that* assumption
- The bottleneck quantization attacks is **memory bandwidth**: fewer bits per weight → fewer bytes to move → faster decode. The compute is usually still done in FP16/BF16 after dequant (weight-only, "WxA16"), or in INT8/FP8 (WxAx)
- **Outliers are the enemy.** Activation outlier channels are what make naive INT8 fail; SmoothQuant migrates them, AWQ protects them, QuIP#/TurboQuant *rotate them away*
- The frontier (QuIP#, TurboQuant) is **incoherence via random rotation**: a rotation makes any vector look sphere-uniform, whose coordinate marginals are *known*, so you can apply a provably near-optimal scalar quantizer with no calibration data

## How to Study

- Implement symmetric per-tensor INT8 quant/dequant from the formula before touching a library -- it's ten lines and everything else is a refinement of it
- For each quantizer, write down: (1) what distribution it assumes, (2) what error metric it minimizes (MSE? inner product?), (3) what extra state it stores (scales, zero-points, codebooks, rotations)
- Profile the dequant+GEMM path, not just the quantizer -- the win is realized in the kernel ([Marlin](https://github.com/vllm-project/vllm), CUTLASS), not in Python

---

# Concepts & Techniques

## Core Insight

A quantizer is a function `Q: ℝ → {l₀, …, l_{2^b−1}}` mapping reals to `2^b` levels, plus a decoder `D` mapping levels back to reals. The **distortion** is `E[(x − D(Q(x)))²]`, and rate-distortion theory ([Information Theory](../../../math/11-information-theory/)) says the achievable distortion at `b` bits depends on the *entropy* of the source. So the whole game is: **make the source easy** (rotate it to be Gaussian/uniform), then **spend your levels where the mass is** (Lloyd–Max, quantile codebooks), while **protecting what matters** (salient weights, inner products).

## 1. The Core Math: Affine and Symmetric Quantization

Map a real range `[β, α]` onto integer codes `[q_min, q_max]` (e.g. `[-128, 127]` for INT8).

**Affine (asymmetric)** — handles skewed ranges (e.g. post-ReLU activations):

```
s = (α − β) / (q_max − q_min)          # scale (Δ, the step size)
z = round(q_min − β / s)                # zero-point (integer)
quantize:    q = clamp(round(x / s) + z, q_min, q_max)
dequantize:  x̂ = s · (q − z)
```

**Symmetric** — `z = 0`, used for weights (roughly zero-mean) because it makes the GEMM cheaper (no cross-terms):

```
s = max(|x|) / q_max          # q_max = 127 for INT8
q = clamp(round(x / s), −127, 127)
x̂ = s · q
```

Math → code (C, symmetric per-tensor):

```c
// scale = absmax(x) / 127  (computed once over the tensor)
int8_t quantize(float x, float scale)   { 
    int32_t q = (int32_t)lrintf(x / scale);   // round-to-nearest-even
    return (int8_t)(q < -127 ? -127 : q > 127 ? 127 : q);
}
float   dequantize(int8_t q, float scale) { return scale * (float)q; }
```

The full per-tensor kernel (absmax reduction → quantize → dequant, plus an INT8 dot product with `dp4a`) is in [`code/symmetric_quant.cu`](code/symmetric_quant.cu).

## 2. Granularity: Per-Tensor / Channel / Group / Block

The *only* thing that changes is **how many `(s, z)` pairs you keep**, trading metadata for accuracy:

| Granularity | One scale per | Metadata | Use |
|-------------|---------------|----------|-----|
| Per-tensor | whole tensor | tiny | activations, fast path |
| Per-channel (per-row) | output channel | one vector | weights (W8A8) |
| Per-group | block of K (e.g. 128) along in-dim | K⁻¹ of weights | INT4 weights (GPTQ/AWQ default `group_size=128`) |
| Per-block | block of 32–256 | a few % | GGUF k-quants, NF4 (block 64) |

Finer granularity = more scales = less distortion, because each block clips to its *own* range. This is why INT4 group quantization works at all: a 128-wide group has a far tighter dynamic range than a 4096-wide row.

## 3. Distortion, Clipping, and Calibration

For a uniform quantizer with step `Δ`, the **granular noise** (the rounding error) is uniform on `[−Δ/2, Δ/2]`, so its variance is `Δ²/12`. With `Δ = 2c / 2^b` for clip threshold `c`:

```
D(c) = (overload/clipping error for |x| > c)  +  Δ²/12
                                                  ─────────  granular
```

Smaller `c` shrinks granular noise but raises clipping error. **Calibration** picks `c`:
- **Min-max**: `c = max(|x|)` — no clipping, but one outlier blows up `Δ`
- **Percentile**: `c = 99.9th percentile` — clip the tail
- **MSE-optimal**: grid/golden-section search `c` minimizing measured `D(c)` (what AWQ/GPTQ calibration and `compressed-tensors` do)

This is a [rate-distortion](../../../math/11-information-theory/) problem: at fixed rate `b`, choose the quantizer minimizing expected distortion for the source distribution.

## 4. Where the Speed Comes From: Dequant-Fused GEMM

Quantization only pays off if the **kernel** exploits it. The pattern for weight-only INT4 (`W4A16`):

1. Load 4-bit weights from HBM (4× less bandwidth than FP16)
2. **Dequantize in registers/shared memory** to FP16 just before the tensor-core matmul
3. Accumulate in FP32

This is what [Marlin](https://github.com/vllm-project/vllm) (Ampere/Ada W4A16) and **Machete** (Hopper) do in vLLM, and what CUTLASS mixed-input GEMM provides. For `W8A8`, the matmul itself is INT8 (`mma.sync` / `dp4a`), and you **dequantize in the epilogue**: `out = (s_x · s_w) · int32_acc`. See the `dp4a` dot product in [`code/symmetric_quant.cu`](code/symmetric_quant.cu).

> Rule: a quantizer is only as good as its kernel. INT4 with a slow Python dequant loop is *slower* than FP16. Always check there's a fused kernel (Marlin/Machete/CUTLASS/tinygemm) for your format on your GPU.

## The Arc: How Quantization Evolved (and What Changed Each Time)

The zoo below isn't a flat list — it's a *sequence of discoveries*, each fixing the failure of the last. Reading it chronologically is the fastest way to understand *why* each method exists.

| When | Method (paper) | The problem it hit | What changed |
|------|----------------|--------------------|--------------|
| 2018 | **Integer-only PTQ/QAT** ([Jacob et al., 1712.05877](https://arxiv.org/abs/1712.05877)) | Floats are expensive on edge HW | Affine INT8, per-tensor/per-channel scales — the baseline everything refines |
| Aug 2022 | **LLM.int8()** ([2208.07339](https://arxiv.org/abs/2208.07339)) | Naive INT8 *collapses* on >6.7B models | Found **activation outlier channels**; keep those few in FP16, rest INT8 (mixed-precision decomposition) |
| Oct 2022 | **GPTQ** ([2210.17323](https://arxiv.org/abs/2210.17323)) | Want 3–4-bit *weights*, not just 8 | Second-order, **error-compensating** PTQ (minimize output error, not weight error) → INT4 weights become usable |
| Nov 2022 | **SmoothQuant** ([2211.10438](https://arxiv.org/abs/2211.10438)) | Outliers still block full **W8A8** | **Migrate** outlier difficulty from activations into weights → both fit INT8 |
| May 2023 | **QLoRA / NF4** ([2305.14314](https://arxiv.org/abs/2305.14314)) | Can't *fine-tune* a quantized model cheaply | **Quantile codebook** (NormalFloat) + double-quant → fine-tune a 4-bit base on one GPU |
| Jun 2023 | **AWQ** ([2306.00978](https://arxiv.org/abs/2306.00978)) | GPTQ needs the Hessian; want simpler W4 | **Protect ~1% salient channels** (found via *activation* magnitude) by scaling — best W4A16 quality/speed |
| 2023–24 | **FP8 (E4M3/E5M2)** | INT8 wastes range on near-zero weights | A float's **exponent** handles dynamic range; native Hopper/Ada tensor cores → W8A8 at near-INT8 speed, tiny loss |
| Feb 2024 | **QuIP#** ([2402.04396](https://arxiv.org/abs/2402.04396)) | 2-bit is too lossy — outliers again | **Rotate outliers away** (Hadamard incoherence) + E8-lattice vector quant → usable 2-bit |
| Jun 2024 | **QJL** ([2406.03482](https://arxiv.org/abs/2406.03482)) | KV cache can't be calibrated (unseen tokens) | **Data-oblivious** JL sign-sketch for keys → low-bit attention with unbiased scores |
| Apr 2025 | **TurboQuant** ([2504.19874](https://arxiv.org/abs/2504.19874)) | Want optimal, online, no-calibration | Rotation → **known sphere marginals** → per-coordinate Lloyd–Max; provably near-optimal at 2.5–3.5 bit KV |

**The two through-lines**: (1) the enemy is always **outliers** — successive methods *isolate* (LLM.int8), *migrate* (SmoothQuant), *protect* (AWQ), then *rotate away* (QuIP#/TurboQuant) them; (2) the field moved from **data-dependent calibration** toward **data-oblivious, provable** schemes, because the KV cache (the new bottleneck) can't be calibrated on tokens you haven't generated yet.

## 5. The Quantizer Zoo

Each entry: the assumption, the math, the code translation.

### 5.1 GPTQ — second-order, error-compensating

**Assumption**: quantizing a weight perturbs the layer output; minimize that, not the weight error. **Math**: for a linear layer with calibration inputs `X`, minimize `‖WX − ŴX‖²`. The Hessian is `H = 2XXᵀ`. Quantizing weight column `q` and greedily compensating the *not-yet-quantized* columns:

```
err  = (w_q − quant(w_q)) / [H⁻¹]_qq
W   −= err · H⁻¹[:, q]          # push the error into remaining weights
```

GPTQ fixes the column order, uses the **Cholesky factor of `H⁻¹`** for numerical stability, and updates in lazy batches. Pseudocode:

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

**Stack**: GPTQModel / AutoGPTQ produce `compressed-tensors`/GPTQ checkpoints; served by Marlin in vLLM.

### 5.2 AWQ — protect the salient channels

**Assumption**: ~1% of weight channels matter disproportionately, identifiable by **activation** magnitude (not weight magnitude). **Math**: scale those channels up before quantizing (so they get more effective resolution), and fold the inverse into activations so the product is unchanged:

```
Ŵ = W · diag(s),   X̂ = diag(s)⁻¹ · X      ⟹   X̂ Ŵ = X W   (math-equivalent)
s_j = (activation_scale_j)^α              # per-input-channel, grid-search α∈[0,1]
```

Pick `α` by minimizing the layer's output MSE on calibration data. Quantize `Ŵ` to INT4. Code:

```python
act_scale = X.abs().mean(dim=0)                  # per-channel activation magnitude
def loss(alpha):
    s  = act_scale.pow(alpha).clamp(min=1e-4)
    Wq = quant_dequant(W * s, n_bits=4, group=128) / s
    return ((X @ W.T) - (X @ Wq.T)).pow(2).mean()
alpha = golden_section_search(loss, 0.0, 1.0)    # ~20 evals
```

**Stack**: AutoAWQ; AWQ Marlin kernels in vLLM/TensorRT-LLM. Often the best quality/speed for W4A16.

### 5.3 SmoothQuant — migrate the difficulty (W8A8)

**Assumption**: activations have hard-to-quantize outlier channels; weights are easy. **Migrate** outliers from activations into weights via a per-channel smoothing factor `s`:

```
Y = (X · diag(s)⁻¹) · (diag(s) · W)            # identity, then quantize both
s_j = max(|X_j|)^α / max(|W_j|)^(1−α),   α ≈ 0.5
```

After smoothing, both `X̂ = X·diag(s)⁻¹` and `Ŵ = diag(s)·W` fit comfortably in INT8 → full **W8A8** matmul. **Stack**: SmoothQuant in TensorRT-LLM and `compressed-tensors` (`w8a8`).

### 5.4 NF4 — quantile codebook for Gaussian weights (QLoRA)

**Assumption**: weights are ~`N(0, σ)`. The information-optimal 4-bit code under *equiprobable bins* puts levels at the **quantiles** of a normal — "NormalFloat". 16 levels: 8 negative, a zero, 7 positive, each a fixed `N(0,1)` quantile, scaled per block by `absmax`:

```
codebook = [ Φ⁻¹( (i + 0.5)/16 ) normalized to [−1, 1] ]   # 16 fixed levels
block (64 weights):  s = absmax(block);  q = argmin_i |w/s − codebook[i]|
dequant:             ŵ = s · codebook[q]
```

**Double quantization**: the FP32 block scales `s` are themselves quantized to 8-bit (block 256) with a second scale → saves ~0.37 bits/param. Full C implementation: [`code/nf4_quant.c`](code/nf4_quant.c). **Stack**: bitsandbytes (`load_in_4bit`, `bnb_4bit_quant_type="nf4"`) — the default for QLoRA fine-tuning.

### 5.5 GGUF k-quants — hierarchical block scales (llama.cpp)

**Assumption**: per-block scales themselves compress well. **`Q4_K`** packs a **super-block of 256** weights = 8 sub-blocks of 32. Each sub-block has a 6-bit scale `d` and 6-bit min `m`; the eight `d`s and eight `m`s share two FP16 super-scales:

```
d_sub = d_super · scale6[b]          # b = sub-block index
m_sub = m_super · min6[b]
ŵ = d_sub · q4 − m_sub               # q4 ∈ [0, 15]
```

Effective ~4.5 bits/weight, no calibration. `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q2_K`, and the newer **IQ** (i-quant) codebook variants (`IQ4_NL`, `IQ2_XS`) all follow this hierarchical-block idea. A faithful `Q4_K`-style dequant in C is in [`code/nf4_quant.c`](code/nf4_quant.c) (`dequant_q4k_block`). **Stack**: llama.cpp / Ollama / LM Studio; runs on CPU + Metal + CUDA.

### 5.6 FP8 — native low-precision float (E4M3 / E5M2)

**Assumption**: a float's exponent handles dynamic range better than a fixed-point int. **E4M3** (1 sign, 4 exp, 3 mantissa, bias 7): `value = (−1)^s · 2^(e−7) · (1 + m/8)` for normals; max ≈ 448, no infinities. **E5M2** has more range, less precision (used for gradients). Apply a per-tensor scale, then cast:

```
x_scaled = x / amax * 448.0          # fit E4M3 dynamic range
fp8 = cast_e4m3(x_scaled)            # hardware instruction on Hopper/Ada
```

Hopper (H100) and Ada (L40S) tensor cores do FP8 matmul natively → near-INT8 throughput with far less accuracy loss. **Stack**: TensorRT-LLM, vLLM (`--quantization fp8`), torchao `float8`. The default for serving the latest large models when an H100 is available.

**The float family.** E = exponent bits (range), M = mantissa bits (precision):

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
| **NF4** | 4-bit codebook | Not a true float: 16 fixed Gaussian-quantile levels (§5.4) |

### 5.7 QuIP# — incoherence + lattice codebook

**Assumption**: outliers can be *rotated away*. Multiply weights and Hessian by **random orthogonal (Hadamard) matrices** to make them *incoherent* (mass spread evenly, no spikes), then vector-quantize with the **E8 lattice** codebook (the densest 8-D lattice) + light fine-tuning. Achieves usable 2-bit. This is the conceptual bridge to TurboQuant: **rotation makes the distribution predictable, then quantize optimally.**

### 5.8 TurboQuant — provably near-optimal, data-oblivious

[**TurboQuant**](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — Google/NYU, 2025) is the current high-water mark for *online, calibration-free* quantization, especially KV cache. The math:

**Step 1 — random rotation.** Apply a randomized Hadamard transform `R = (1/√d) H D`, where `D` is a random `±1` diagonal and `H` is the Hadamard matrix (so `Rx` costs `O(d log d)` via a Fast Walsh–Hadamard Transform, *not* `O(d²)`). For any input vector, `Rx` is distributed like a uniform random vector on the sphere of the same norm.

**Step 2 — known marginals.** On the sphere in `ℝ^d`, each coordinate has a **concentrated Beta distribution** (density `∝ (1 − t²)^((d−3)/2)`, ≈ `N(0, 1/d)` for large `d`), and distinct coordinates are *near-independent* in high dimension. Crucially this marginal is **the same for every vector and known a priori** — no calibration data needed.

**Step 3 — optimal scalar quantizer per coordinate.** Because the marginal is known, apply the **Lloyd–Max optimal scalar quantizer** for that distribution to every coordinate independently. This provably hits the MSE distortion-rate bound within a constant factor ≈ **2.7** across *all* bit-widths and dimensions.

**Step 4 — unbiased inner products (the KV-cache trick).** A pure MSE quantizer is *biased* for inner products (it systematically shrinks `⟨q, k⟩`, which attention needs). TurboQuant fixes this with a **two-stage** scheme: MSE-quantize, then take the **residual** and apply a **1-bit QJL** ([Quantized Johnson–Lindenstrauss](https://arxiv.org/abs/2406.03482)) transform — store `sign(⟨residual, gᵢ⟩)` for random Gaussian projections `gᵢ`. Summing the two stages gives an **unbiased** inner-product estimate.

**Results**: for KV cache, **3.5 bits/channel** is quality-neutral and **2.5 bits/channel** is marginal — with no calibration, computed online as tokens stream. That data-oblivious property is exactly what KV caching needs (you can't calibrate on tokens you haven't seen).

A minimal end-to-end TurboQuant encoder/decoder in C — random-sign + iterative FWHT rotation, per-coordinate uniform quantizer (a stand-in for the Lloyd–Max levels, with a table for the Gaussian-optimal points), and the 1-bit QJL residual sign for unbiased inner products — is in [`code/turboquant.c`](code/turboquant.c).

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

## 6. Putting It Together: KV-Cache Quantization

The KV cache is the prime target because it grows unbounded and is read every decode step ([see the parent topic](../)):
- **Per-token / per-channel scales** (KVQuant, KIVI): keys quantize better per-channel, values per-token
- **QJL**: data-oblivious sign sketch for the keys → low-bit attention with unbiased scores
- **TurboQuant**: the online, no-calibration sweet spot — 2.5–3.5 bits/channel
- Stores the cache in INT4/INT8/FP8, freeing memory for longer context and bigger batches

---

## Choosing a Quantizer (2025/2026)

| Goal | Use | Bits | Why |
|------|-----|------|-----|
| Best W4 quality, GPU serving | **AWQ** or **GPTQ** + Marlin | 4 (W4A16) | Salient-channel / second-order; fast fused kernel |
| Max throughput on H100 | **FP8** (E4M3) | 8 (W8A8) | Native tensor cores, tiny accuracy loss |
| Big-model W8A8 without retrain | **SmoothQuant** | 8 | Migrates activation outliers |
| Fine-tune a quantized model | **NF4** (QLoRA) | 4 | Quantile-optimal for Gaussian weights |
| CPU / laptop / edge | **GGUF k-quants** (`Q4_K_M`) | ~4.5 | No GPU, hierarchical block scales |
| Aggressive 2-bit weights | **QuIP#** | 2 | Incoherence + E8 lattice |
| KV cache, online, no calibration | **TurboQuant** / QJL | 2.5–3.5 | Data-oblivious, near-optimal distortion |

## 7. Rust / Burn

[Burn](https://burn.dev/) (pre-1.0, evolving fast — pin a version) has a backend-agnostic quantization API in `burn::tensor::quantization`. Core types: `QuantScheme` (config), `QuantValue` (`Q8S`, `Q4S`, `Q2S` fixed-point symmetric; `E4M3`/`E5M2`/`E2M1` float), `QuantLevel` (per-tensor / per-block granularity), `Calibration` (min-max), `QuantMode` (symmetric). The simplest path is **dynamic quantization** (calibrate on the fly):

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

A runnable example (autodiff/NdArray backend, quantize → dequantize → measure MSE) is in [`code/burn_quantize.rs`](code/burn_quantize.rs). For production Rust inference, GGUF k-quant models also run through `llama.cpp` bindings (`llama-cpp-2`) and `candle`'s `quantized` module.

> See [Inference Frameworks](../frameworks/) for the full cross-language runtime landscape (C/Python/Rust/Zig) that consumes these formats -- including a pure-Zig SIMD INT8 dot product ([`frameworks/code/quant_dot.zig`](../frameworks/code/quant_dot.zig)), the CPU twin of [`code/symmetric_quant.cu`](code/symmetric_quant.cu).

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Rate-distortion, entropy, quantile coding | [Information Theory](../../../math/11-information-theory/) | Why NF4/Lloyd–Max are optimal |
| Hessian, Cholesky, orthogonal/Hadamard matrices | [Linear Algebra](../../../math/03-linear-algebra/) | GPTQ, QuIP#, TurboQuant rotations |
| Beta/Gaussian marginals, concentration | [Probability & Statistics](../../../math/07-probability-statistics/) | TurboQuant coordinate distribution |
| KV cache, serving throughput | [LLM Systems & Inference](../) | What quantization buys you |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| NVIDIA | FP8/INT4 kernels (CUTLASS, TensorRT-LLM), tensor-core mma | Expert |
| Google/DeepMind | TurboQuant, QJL, rotation-based quantization research | PhD-level |
| Anthropic / OpenAI | Serving-cost reduction via weight + KV-cache quantization | PhD-level |
| Neural Magic (vLLM) | compressed-tensors, Marlin/Machete kernels | Expert |
| Meta | GGUF/llama.cpp ecosystem, edge quantization | Expert |
