# Choosing and validating quantization

This part of [Quantization: Math to Code](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Choose a supported format and validate it on the deployment workload.

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

A runnable example that quantizes, dequantizes, and measures the MSE on the NdArray backend with autodiff is in [`code/burn_quantize.rs`](code/burn_quantize.rs). For production Rust inference, GGUF k-quant models also run through llama.cpp bindings (`llama-cpp-2`) and candle's `quantized` module. The full cross-language runtime landscape that consumes these formats is mapped in [Inference Frameworks](../02-frameworks/README.md), including a pure-Zig SIMD INT8 dot product, [`frameworks/code/quant_dot.zig`](../02-frameworks/code/quant_dot.zig), which is the CPU twin of [`code/symmetric_quant.cu`](code/symmetric_quant.cu).

## Where this leads

The theory behind this section comes from *Mathematics*: rate-distortion, entropy, and quantile coding in [Information Theory](../../../mathematics/17-information-theory/README.md) explain why NF4 and Lloyd–Max quantizers are optimal; Hessians, Cholesky factors, and orthogonal and Hadamard matrices in [Linear Algebra](../../../mathematics/09-linear-algebra/README.md) underlie GPTQ, QuIP#, and TurboQuant's rotations; and the Beta and Gaussian marginals and concentration results in [Probability and Statistics](../../../mathematics/13-probability-statistics/README.md) describe TurboQuant's coordinate distribution. On the systems side, what quantization buys you in KV-cache capacity and throughput is the subject of [LLM Systems & Inference](../README.md) and [Serving & Load](../04-serving-and-load/README.md), and how quantized checkpoints are recognized and loaded is covered in [Model Loading: Hub to GPU](../03-model-loading/README.md).

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

[Chapter contents](README.md)
