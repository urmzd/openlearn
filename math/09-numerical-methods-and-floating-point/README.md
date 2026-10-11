# Numerical Methods and Floating Point

## Overview

- **Primary references**: Goldberg, [*What Every Computer Scientist Should Know About Floating-Point Arithmetic*](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) (free); Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM, 2nd ed.)
- **Supplementary**: Trefethen and Bau, *Numerical Linear Algebra* (SIAM); Muller et al., *Handbook of Floating-Point Arithmetic*; the [OCP Microscaling Formats (MX) v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (free); Micikevicius et al., [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433) (free)
- **Prerequisites**: [Precalculus](../00-precalculus/), [Calculus 1](../01-calculus-1/) and [Calculus 2](../02-calculus-2/) (Taylor series), [Linear Algebra](../03-linear-algebra/)
- **Estimated time**: 3 to 4 weeks at 10 to 12 h/week; in the course, Pass 2 (M09.1, M09.2) and Pass 6 (M09.3 to M09.6)

## Key Takeaways

- **A float is a sign, an exponent, and a mantissa**, and every operation rounds to the nearest representable value (round to nearest, ties to even). The gap between neighbours, one ULP, grows with magnitude.
- **Most numerical bugs are cancellation or overflow**, and both have standard cures: subtract the maximum before `exp` (log-sum-exp), sum in pairs or with compensation (Kahan).
- **Tolerances are derived, not guessed.** A dot product of length $k$ in a format with unit roundoff $u$ is off by at most about $k u \sum |a_i b_i|$; tests should assert that bound, not `1e-5`.
- **Low precision is an encoding problem.** bf16, fp16, fp8, and MX formats differ in how many bits go to range vs precision; quantization picks a scale so values land where the format is dense.
- **Transcendentals in C are polynomials plus range reduction**, and `rsqrt` is Newton's method with a good first guess.

## How to Study

Read Goldberg once quickly, then Higham chapters 1 to 4 with a Python session: reproduce every cancellation example with `numpy.float32`. Decompose floats by hand (`struct.pack`) until reading bits is easy. In the course, `M09.1` and `M09.2` come in Pass 2 because autograd needs stable softmax; `M09.3` to `M09.6` come in Pass 6 with the inference engine and the C kernels.

---

# Concepts & Techniques

## Core Insight

Real numbers do not fit in 32 bits, so every computation is an approximation, and the job is to know how far off it is. Floating point fails in a few predictable ways (overflow, underflow, cancellation, accumulation of rounding error), each with a known algorithmic fix and a provable error bound. The same bound that explains why a kernel is accurate becomes the tolerance its test asserts.

## 1. IEEE-754 anatomy and rounding

**Key ideas**:
- **Layout**: f32 is 1 sign, 8 exponent, 23 mantissa bits; bf16 keeps f32's exponent with 7 mantissa bits; fp16 has 5 and 10.
- **Rounding** to nearest even through a `uint32` view is how bf16 is emulated in numpy (`M09.1`, used by `L7.9` weight loading and `L11.1` mixed precision).

## 2. Stable numerics

**Key ideas**:
- **Log-sum-exp**: $\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$ with $m = \max_i x_i$; softmax and log-softmax follow, finite at $\pm 10^4$, and a fully masked row gives zeros.
- **Summation**: Kahan compensation and pairwise summation keep long reductions (perplexity over $10^7$ tokens) accurate (`M09.2`).

## 3. Error analysis and tolerance budgets

**Key ideas**:
- **Unit roundoff** $u$ and $\gamma_k = k u / (1 - k u)$ bound the error of $k$ accumulated operations.
- **Condition number** $\kappa(A) = \|A\| \|A^{-1}\|$ separates a bad problem from a bad algorithm.
- **Call sites**: `M09.3` gives the learner's own differential tests their bounds, `L8.5` the quantization error budget, and optional `L9.1` the standalone C matmul parity bound.

## 4. Low-precision formats and kernels in C

**Key ideas**:
- **fp8** E4M3 and E5M2, and **MX** block formats with a shared E8M0 scale per 32 values (`M09.4`, used by `L8.5`, `L9.4`, and the KV format v2 migration).
- **Fast inverse square root**: a bit-level first guess plus two Newton steps (`M09.5`, used by `tl_rmsnorm_f32`).
- **`expf`**: range reduction $x = k \ln 2 + r$, a minimax polynomial for $e^r$, and a scale by $2^k$ (`M09.6`, used by softmax, attention, and SiLU kernels).

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `S-M09a` | Floating point and stable numerics problem set | solve | 2 |
| `S-M09b` | Error analysis and low-precision problem set | solve | 6 |
| `M09.1` | IEEE-754 anatomy, ULP, RNE rounding, bf16/fp16 emulation | build | 2 |
| `M09.2` | Stable numerics: LSE, softmax, Kahan and pairwise sums | build | 2 |
| `M09.3` | Error analysis, condition numbers, **tolerance budgets** | build | 6 |
| `M09.4` | FP8 E4M3/E5M2, MXFP4/MXFP8 with E8M0 block scales; C conversions | build | 6 |
| `M09.5` | Fixed-point iteration, fast inverse sqrt in C | build | 6 |
| `M09.6` | Polynomial approximation and range reduction: `expf` in C | build | 6 |
| `M09.7` | Iterative solvers: conjugate gradient | build | optional |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M09.1` | [IEEE 754 anatomy, ulp, round to nearest even, and bf16 and fp16 emulation](01-ieee-754.md) | build | 2 |
| 2 | `M09.2` | [Stable numerics: logsumexp, softmax, compensated sums](02-stable-numerics.md) | build | 2 |
| 3 | `M09.3` | [Error analysis, condition numbers, tolerance budgets](03-error-analysis-condition-numbers-and-tolerance-budgets.md) | build | 6 |
| 4 | `M09.4` | [FP8 E4M3/E5M2, MXFP4/MXFP8 with E8M0 scales](04-fp8-and-microscaling-formats.md) | build | 6 |
| 5 | `M09.5` | [Fast inverse square root in standalone C (optional)](05-fixed-point-iteration-and-rsqrt.md) | side | 6 |
| 6 | `M09.6` | [Polynomial approximation and range reduction: expf in C (optional)](06-polynomial-approximation-and-expf.md) | side | 6 |
| 7 | `M09.7` | [Low-precision conversions in standalone C (optional)](07-low-precision-c-conversions.md) | side | 6 |
| 8 | `S-M09a` | [Floating point problem set, part a: representation, rounding, cancellation](90-problem-set-a.md) | solve | 2 |
| 9 | `S-M09b` | [Floating point problem set, part b: condition numbers, error bounds, Newton, polynomial approximation, low precision](91-problem-set-b.md) | solve | 6 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Calculus 2](../02-calculus-2/) | Taylor series behind polynomial approximation |
| [Matrix Calculus and Autodiff](../08-matrix-calculus-and-autodiff/) | VJPs that need stable softmax and normalization |
| [tinyllm Part 8](../../ml/08-tinyllm/p08-inference/) | quantization and its error budget |
| [tinyllm Part 9](../../ml/08-tinyllm/p09-kernels/) | the C kernels that call `tl_expf` and the fast `rsqrt` |
| [LLM Systems: quantization](../../ml/04-llm-systems/quantization/) | production quantization formats as depth |
