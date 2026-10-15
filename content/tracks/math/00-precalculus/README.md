# Precalculus

## Overview

- **Primary reference**: OpenStax, [*Precalculus 2e*](https://openstax.org/details/books/precalculus-2e) (free, CC BY)
- **Supplementary**: OpenStax, [*College Algebra 2e*](https://openstax.org/details/books/college-algebra-2e) (free) for the algebra review; Paul's Online Notes, [Algebra and Trig Review](https://tutorial.math.lamar.edu/Extras/AlgebraTrigReview/AlgebraTrigReview.aspx) (free); 3Blue1Brown, [Euler's formula with introductory group theory](https://www.youtube.com/watch?v=mvmuCPvRoWQ) (free)
- **Prerequisites**: high-school algebra (solving linear equations, manipulating fractions and powers)
- **Estimated time**: 2 to 3 weeks at 10 to 12 h/week; in the course, the first stages of Pass 2

## Key Takeaways

- **Logarithms turn products into sums and change the unit of information.** A model's loss in nats becomes bits by dividing by $\ln 2$; bits per byte is the comparison every tokenizer and model in the course is judged by.
- **A rotation is a multiplication by a unit complex number.** Rotating a 2D pair by $\theta$ is multiplying $x + iy$ by $e^{i\theta}$; RoPE applies that to every pair of a query and a key.
- **Geometric sequences are frequency ladders.** RoPE's inverse frequencies $b^{-2i/d}$ and ALiBi's head slopes are geometric sequences; their sums and ratios decide how far a position signal reaches.
- **How you evaluate a formula matters as much as the formula.** Horner's rule is the cheapest and most stable way to evaluate a polynomial, and the textbook quadratic formula can lose most of its digits for the small root when $b^2 \gg 4ac$.

## How to Study

Read each OpenStax chapter with a notebook open and check every identity numerically before you trust it (`math.log`, `cmath.exp`, `numpy`). In the course, take the solve set `S-M00` first: it is the pen-and-paper check of the four build modules, then build `M00.1` to `M00.4` in order with `ol start <ID>` and `ol check <ID>`.

---

# Concepts & Techniques

## Core Insight

Precalculus is the toolbox the rest of the math is written in: exponents and logarithms, trigonometry and complex numbers, sequences and series, and polynomials. Each one has a call site in an LLM system: logarithms measure information, rotations encode position, geometric ladders set frequencies, and polynomials approximate the functions a C kernel cannot call.

## 1. Exponents, logarithms, and units of information

**Key ideas**:
- **Laws**: $a^{m}a^{n} = a^{m+n}$, $\log_b(xy) = \log_b x + \log_b y$, change of base $\log_b x = \ln x / \ln b$.
- **Units**: a negative log-likelihood summed in nats over a text of $n$ bytes is $\text{bpb} = \text{nll} / (n \ln 2)$ bits per byte, a number comparable across tokenizers.
- **Call sites**: `M11.2` perplexity and bits per byte, `L1.6` tokenizer metrics, `L6.7` the model-zoo table, `C1`.

## 2. Trigonometry, rotations, and complex numbers

**Key ideas**:
- **Unit circle**: $(\cos\theta, \sin\theta)$; a 2D rotation is the matrix $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$, which preserves length and composes by adding angles: $R(a)R(b) = R(a+b)$.
- **Euler's formula**: $e^{i\theta} = \cos\theta + i\sin\theta$, so the same rotation is one complex multiplication.
- **Call sites**: `L5.4` sinusoidal positional encoding, `L7.3` RoPE.

## 3. Sequences, series, and frequency ladders

**Key ideas**:
- **Geometric sequence**: $a, ar, ar^2, \dots$ with sum $a(1 - r^n)/(1 - r)$ for $r \ne 1$.
- **Ladders**: RoPE `inv_freq` is $b^{-2i/d}$ for $i = 0, \dots, d/2 - 1$; ALiBi slopes are a geometric sequence per head, with a rule for head counts that are not powers of two.
- **Call sites**: `L7.3` RoPE, `L7.4` ALiBi and YaRN, `M02.2`, `M10.4` schedules.

## 4. Polynomials, Horner's rule, and stable roots

**Key ideas**:
- **Horner**: $a_0 + x(a_1 + x(a_2 + \dots))$ uses $n$ multiplications and $n$ additions and is the form every polynomial kernel uses.
- **Cancellation**: when $b^2 \gg 4ac$ one root of $-b \pm \sqrt{b^2 - 4ac}$ subtracts two nearly equal numbers; compute the large-magnitude root first and get the other from $x_1 x_2 = c/a$.
- **Call sites**: `M02.1` Taylor polynomials, `M09.6` the polynomial core of `tl_expf`.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `S-M00` | Precalculus problem set (checked by SymPy) | solve | 2 |
| `M00.1` | Exponents, logs, change of base, units of information | build | 2 |
| `M00.2` | Trig, unit circle, 2D rotations, complex numbers, Euler's formula | build | 2 |
| `M00.3` | Sequences, geometric series, frequency ladders | build | 2 |
| `M00.4` | Polynomials, Horner, stable quadratic roots | build | 2 |
| `M00.5` | Functions, inverses, monotonicity, inequalities | solve | solve set |

`M00.5` (functions, inverses, monotonicity, inequalities) is solve-only: it appears as items of `S-M00` and is used by `M01.1` and `M05.2`.

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `S-M00` | [Solve set: functions, exp/log, trig and Euler, series, polynomials, inequalities](00-solve-set.md) | solve | 2 |
| 2 | `M00.1` | [Exponents, logs, change of base, units of information](01-exponents-logs-and-units-of-information.md) | build | 2 |
| 3 | `M00.2` | [Trig, unit circle, 2D rotations, complex numbers, Euler's formula](02-trig-rotations-and-eulers-formula.md) | build | 2 |
| 4 | `M00.3` | [Sequences, geometric series, frequency ladders](03-sequences-and-frequency-ladders.md) | build | 2 |
| 5 | `M00.4` | [Polynomials, Horner, stable quadratic roots](04-polynomials-horner-and-stable-roots.md) | build | 2 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Calculus 1](../01-calculus-1/) | limits and derivatives build on functions, exponentials, and logarithms |
| [Numerical Methods and Floating Point](../09-numerical-methods-and-floating-point/) | range reduction and polynomial approximation of `exp` (M09.6) start from Horner's rule |
| [Information Theory](../11-information-theory/) | entropy and perplexity are logarithms with a unit |
| [tinyllm](../../../courses/build-an-llm/) | RoPE, ALiBi, and bits per byte are the first places this math runs |
