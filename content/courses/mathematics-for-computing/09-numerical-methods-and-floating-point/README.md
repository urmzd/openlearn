# Numerical Methods and Floating Point

Real numbers do not fit in 32 bits, so every computation a model performs is an approximation, and the job of the engineer is to know how far off it is. Floating point fails in a few predictable ways: overflow, underflow, cancellation, and the accumulation of rounding error. Each failure has a known algorithmic fix and a provable error bound, and the same bound that explains why a kernel is accurate becomes the tolerance its test asserts. This chapter teaches you to see those failures coming, to fix them, and to prove that the fix works.

The chapter moves from representation to algorithms to low-level kernels. It opens by taking float32 apart bit by bit and rebuilding bfloat16 and float16 from it, because a float is just a sign, an exponent, and a mantissa, and every operation rounds to the nearest representable value. It then writes the stable reductions that every later chapter imports (log-sum-exp, softmax, and compensated sums), because most numerical bugs are cancellation or overflow, and both have standard cures. With the rounding model in hand, it derives rigorous error bounds and condition numbers and turns them into tolerance budgets, so that tests assert a derived number instead of `1e-5`. The second half goes below 16 bits and into C: fp8 and the block-scaled microscaling formats, where low precision is an encoding problem; a fast inverse square root built from Newton's method and a clever first guess; `expf` as range reduction plus a polynomial; and bit-exact low-precision conversions in standalone C. Two problem sets let you work every idea by hand.

> **Learning objectives**
> - Decode and construct IEEE 754 bit patterns, compute ulps, and round to bfloat16, float16, fp8, and FP4 with round to nearest, ties to even.
> - Compute softmax, log-softmax, and log-sum-exp without overflow or underflow, and sum long sequences accurately with pairwise and Kahan summation.
> - Derive rounding-error bounds for sums, dot products, and matrix products, and build tolerance budgets that include deliberate perturbations such as quantization.
> - Distinguish an ill-conditioned problem from an unstable algorithm using condition numbers.
> - Implement transcendental functions in C as range reduction plus a polynomial or a Newton iteration, with accuracy contracts stated in ulps.

## Before you begin

You need powers of two and logarithms from [Precalculus](../00-precalculus/README.md), Newton's method and finite differences from [Calculus 1](../01-calculus-1/README.md), Taylor series and their remainder bounds from [Calculus 2](../02-calculus-2/README.md), and the SVD from [Linear Algebra](../03-linear-algebra/README.md). The stable softmax here is what the vector-Jacobian products of [Matrix Calculus and Autodiff](../08-matrix-calculus-and-autodiff/README.md) need in practice. A good way to read the chapter is with a Python session open: reproduce every cancellation example with `numpy.float32`, and decompose floats by hand with `struct.pack` until reading bits is easy.

The first two sections are needed early, as soon as you build autograd and need a stable softmax. The rest pairs naturally with the inference and kernel chapters of *Build Your Own LLM System*, [Inference](../../build-an-llm/p08-inference/README.md) and [Kernels in C](../../build-an-llm/p09-kernels/README.md), where quantization and the C kernels that call `tl_expf` and the fast `rsqrt` live. The C sections (5 to 7) are optional.

## In this chapter

1. [IEEE 754 anatomy, ulps, and round to nearest even](01-ieee-754.md). The float32 format bit by bit, the spacing of floats, and bfloat16 and float16 rounding as exact integer arithmetic on the bits, including the NaN case that the obvious trick gets wrong.
2. [Stable numerics: logsumexp, softmax, and compensated sums](02-stable-numerics.md). The shift identity that keeps softmax and log-sum-exp finite at any scale, defined behavior for masked rows, and pairwise and Kahan summation with their error bounds.
3. [Error analysis, condition numbers, and tolerance budgets](03-error-analysis-condition-numbers-and-tolerance-budgets.md). The standard rounding model, $\gamma_k$ bounds for dot products and matmuls, tolerance budgets that include quantization, condition numbers, and the optimal finite-difference step.
4. [FP8 and microscaling formats](04-fp8-and-microscaling-formats.md). E4M3, E5M2, FP4, and E8M0 block scales, with one generic rounding routine and the edge rules layered on top.
5. [Fixed-point iteration and fast inverse square root in C](05-fixed-point-iteration-and-rsqrt.md). Newton's method as a fixed-point iteration, the bit-pattern first guess, and a third step in double that brings the result within 0.5004 ulp. Optional.
6. [Polynomial approximation and range reduction: expf in C](06-polynomial-approximation-and-expf.md). Range reduction with the Cody and Waite split, a degree-7 Taylor polynomial by Horner's rule, and exact scaling at the overflow and underflow edges. Optional.
7. [Low-precision conversions in C](07-low-precision-c-conversions.md). One integer rounding routine shared by E4M3, E5M2, bfloat16, and binary16, bit-exact with the Python version. Optional.
8. [Problem set A: representation, rounding, and cancellation](90-problem-set-a.md). Pen-and-paper problems on float fields, gaps, ties, cancellation, stable rewrites, and the max shift, with worked answers.
9. [Problem set B: condition numbers, error bounds, Newton, polynomial approximation, and low precision](91-problem-set-b.md). Pen-and-paper problems on conditioning, $\gamma_k$, convergence rates, polynomial degree and Chebyshev nodes, and fp8 and MX rounding, with worked answers.

## Related chapters

Taylor series in [Calculus 2](../02-calculus-2/README.md) are behind every polynomial approximation here. [Matrix Calculus and Autodiff](../08-matrix-calculus-and-autodiff/README.md) uses the stable softmax and normalization in its VJPs. In *Build Your Own LLM System*, [Inference](../../build-an-llm/p08-inference/README.md) applies the quantization formats and their error budget, and [Kernels in C](../../build-an-llm/p09-kernels/README.md) contains the C kernels that call `tl_expf` and the fast `rsqrt`. For production quantization formats in more depth, see [Quantization: Math to Code](../../machine-learning/04-llm-systems/quantization/README.md) in *Machine Learning and AI*.

## Further reading

- Goldberg, [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (free). Read it once quickly before starting.
- Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM, 2nd ed.). Chapters 1 to 4 pair with the first three sections.
- Trefethen and Bau, *Numerical Linear Algebra* (SIAM).
- Muller et al., *Handbook of Floating-Point Arithmetic*.
- The [OCP Microscaling Formats (MX) v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (free).
- Micikevicius et al., [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433) (free).
