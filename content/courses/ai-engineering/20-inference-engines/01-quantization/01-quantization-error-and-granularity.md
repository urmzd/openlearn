# Quantization error and granularity

This part of [Quantization: Math to Code](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Derive quantization error for a stated scale and grouping rule.

## Quantization as lossy compression

A quantizer is a function $Q: \mathbb{R} \to \{l_0, \ldots, l_{2^b - 1}\}$ mapping reals to $2^b$ levels, plus a decoder $D$ mapping levels back to reals. Its **distortion** is $\mathbb{E}[(x - D(Q(x)))^2]$, and rate-distortion theory from [Information Theory](../../../mathematics/17-information-theory/README.md) says the achievable distortion at $b$ bits depends on the *entropy* of the source.

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

## Summary

Derive quantization error for a stated scale and grouping rule. State the assumptions before applying the result to a different workload.

## Review questions

1. Why is a bits-per-weight count insufficient to size a quantized model?

   <details><summary>Answer</summary>

   Scales, zero points, padding, metadata, and unquantized layers also consume memory.

   </details>

[Chapter contents](README.md) · [Next: Calibration and kernel execution](02-calibration-and-kernel-execution.md)
