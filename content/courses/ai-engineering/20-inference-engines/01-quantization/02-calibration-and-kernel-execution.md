# Calibration and kernel execution

This part of [Quantization: Math to Code](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Explain how calibration and dequantization kernels affect quality and speed.

## Distortion, clipping, and calibration

Granularity decides how many ranges you have. Calibration decides what each range should be. For a uniform quantizer with step $\Delta$, the **granular noise** (the rounding error) is uniform on $[-\Delta/2, \Delta/2]$, so its variance is $\Delta^2/12$. If you clip at threshold $c$, the step is $\Delta = 2c / 2^b$, and the total distortion has two parts:

$$D(c) = \underbrace{\text{(overload/clipping error for } |x| > c)}_{\text{clipping}} + \underbrace{\frac{\Delta^2}{12}}_{\text{granular}}$$

A smaller $c$ shrinks the granular noise but raises the clipping error. **Calibration** picks $c$, and there are three common strategies. **Min-max** sets $c = \max|x|$: there is no clipping, but a single outlier blows up $\Delta$ for everything else. **Percentile** sets $c$ to, say, the 99.9th percentile, deliberately clipping the tail. **MSE-optimal** calibration runs a grid or golden-section search over $c$ to minimize the measured $D(c)$, which is what AWQ and GPTQ calibration and `compressed-tensors` do.

Stated abstractly, this is a [rate-distortion](../../../mathematics/17-information-theory/README.md) problem: at a fixed rate $b$, choose the quantizer that minimizes expected distortion for the source distribution.

## Where the speed comes from: dequant-fused GEMM

All of this saves memory, but memory saved is not time saved unless the **kernel** exploits it. The pattern for weight-only INT4 (`W4A16`) has three steps. First, load the 4-bit weights from HBM, using a quarter of the bandwidth FP16 would need. Second, **dequantize in registers or shared memory** to FP16 just before the tensor-core matmul. Third, accumulate in FP32.

This is what [Marlin](https://github.com/vllm-project/vllm) (W4A16 on Ampere and Ada) and **Machete** (Hopper) do in vLLM, and what CUTLASS's mixed-input GEMM provides. For `W8A8` the matmul itself runs in INT8 (`mma.sync` or `dp4a`), and you **dequantize in the epilogue**: $\text{out} = (s_x \cdot s_w) \cdot \text{int32\_acc}$. The `dp4a` dot product in [`code/symmetric_quant.cu`](code/symmetric_quant.cu) shows this.

> **Warning:** A quantizer is only as good as its kernel. INT4 with a slow Python dequantization loop is *slower* than FP16. Always check that a fused kernel (Marlin, Machete, CUTLASS, tinygemm) exists for your format on your GPU, and profile the dequant-plus-GEMM path, not just the quantizer.

## Summary

Explain how calibration and dequantization kernels affect quality and speed. State the assumptions before applying the result to a different workload.

## Review questions

1. Can a smaller checkpoint execute more slowly?

   <details><summary>Answer</summary>

   Yes. Missing fused kernels, unpacking overhead, shape mismatch, and device support can erase the bandwidth saving.

   </details>

[Chapter contents](README.md) · [Next: Weight and KV quantization methods](03-weight-and-kv-quantization.md)
