# Quantization: Math to Code

Decoding a large language model is limited by memory bandwidth: every generated token requires reading all of the model's weights, and a growing KV cache, out of GPU memory. Quantization attacks that bottleneck directly. Store each weight in 4 or 8 bits instead of 16, and there are fewer bytes to move, so decode gets faster and more memory is left for the KV cache. The compute is usually still done in FP16 or BF16 after dequantization (weight-only quantization, written "WxA16"), or in INT8 or FP8 when activations are quantized too ("WxAx").

This section develops quantization **math first, then the code that implements it**. Every quantizer you will meet is "just" a way of mapping high-precision numbers to a few bits while controlling the distortion that mapping introduces, and the methods differ entirely in *which math* they use to pick the mapping. We start with the ten-line core, affine and symmetric quantization, then work through granularity, distortion, and the kernels that turn fewer bits into real speed, before walking the history of methods from LLM.int8() to TurboQuant and closing with practical guidance. Examples are in C, CUDA, and Rust ([Burn](https://burn.dev/)), targeting the 2025/2026 inference stack.

This section builds on the chapter opening, [LLM Systems & Inference](../README.md), which explains why decode is bandwidth-bound and why the KV cache dominates memory. The math draws on [Linear Algebra](../../../mathematics/09-linear-algebra/README.md) (Hessians, Cholesky factors, orthogonal and Hadamard matrices) and [Information Theory](../../../mathematics/17-information-theory/README.md) (rate-distortion) from *Mathematics*, and the code assumes basic C and CUDA. Standalone, compilable examples live in [`code/`](code/).

> **Learning objectives**
> - Derive and implement affine and symmetric quantization, and explain how granularity trades metadata for accuracy.
> - Write down the distortion of a uniform quantizer and explain how calibration chooses the clipping threshold.
> - Explain why a quantizer only pays off with a dequant-fused GEMM kernel.
> - For each major method (GPTQ, AWQ, SmoothQuant, NF4, GGUF k-quants, FP8, QuIP#, TurboQuant), state the distribution it assumes, the error it minimizes, and the extra state it stores.
> - Choose a quantizer for a given goal and hardware, including KV-cache quantization.

## In this chapter

1. [Quantization error and granularity](01-quantization-error-and-granularity.md): Derive quantization error for a stated scale and grouping rule.
2. [Calibration and kernel execution](02-calibration-and-kernel-execution.md): Explain how calibration and dequantization kernels affect quality and speed.
3. [Weight and KV quantization methods](03-weight-and-kv-quantization.md): Distinguish weight compression from cache compression and their error sources.
4. [Choosing and validating quantization](04-choosing-and-validating-quantization.md): Choose a supported format and validate it on the deployment workload.
