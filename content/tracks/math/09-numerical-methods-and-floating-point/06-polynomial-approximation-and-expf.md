<!-- ol:module M09.6 -->
# Polynomial approximation and range reduction: expf in C (optional)

## Overview

| | |
|---|---|
| **Module** | `M09.6` · side · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/numerics/expf.c`: `tl_expf` and `tl_exp_f32` (and the helper `pow2i`) |
| **Contract** | [`course/contracts/c/include/tinyllm/numerics.h`](../../../../course/contracts/c/include/tinyllm/numerics.h) (the M09.6 section) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/M09.6/`: `test_expf.c` (C, under ASan and UBSan) (what they check: section 4) |
| **Needs** | [`rt.02` runtime support](../../../courses/build-an-llm/p09-kernels/02-arena-allocator-with-marks.md), [`M02.1` Taylor series and range reduction](../02-calculus-2/01-taylor-series.md) (`exp_range_reduced`, the Python twin) (or `--ref-deps`). Reading: [`M00.4` Horner's rule](../00-precalculus/04-polynomials-horner-and-stable-roots.md), [`M09.1` IEEE 754](01-ieee-754.md) |
| **Used by** | `L9.2` softmax · `L9.3` FlashAttention · `L9.4` paged attention · `L9.6` SiLU, each calling `tl_expf` per element |
| **Milestone** | `MS-L9`, the optional standalone C module group |
| **Optional depth** | Muller, *Elementary Functions: Algorithms and Implementation* (3rd ed.), ch. 2 and 11; Cody and Waite, *Software Manual for the Elementary Functions* (1980); Trefethen, *Approximation Theory and Approximation Practice*, ch. 10 |

## Key Takeaways

- **Range reduction turns an infinite domain into a tiny one exactly**: $e^x = 2^k e^r$ with $k = \mathrm{round}(x/\ln 2)$ and $|r| \le \ln(2)/2$, and $2^k$ costs nothing because it is an exponent field (`reduction_boundaries`).
- **The reduction must not round away $r$**: $k \ln 2$ is subtracted in two parts (Cody and Waite), a 9-bit high part whose product with $k$ is exact and a low part for the rest (`million_samples_within_2_ulp`).
- **On $|r| \le 0.347$ a Taylor polynomial of degree 7 is accurate to about $10^{-8}$**, a tenth of a float32 ulp, so the result is within 1.21 ulp everywhere; degree 6 is not enough (`million_samples_within_2_ulp`, `sweep_of_minus_one_to_one`).
- **$k$ runs from $-150$ to $128$**, past both ends of the float exponent range, so $2^k$ is applied in two halves; the edges overflow to $+\infty$ and underflow to $0$ exactly where the contract says (`overflow_edge`, `underflow_edge`).
- **The C kernel is your M02.1 algorithm in float32**, within 2 ulp of `exp_range_reduced` on 50000 inputs (`test_within_2_ulp_of_exp_range_reduced`).

## How to work this chapter

```bash
ol start M09.6              # stubs expf.c into your repo
ol tests M09.6              # read the test catalog first: rung R0, you write no tests here
ol check M09.6              # exit code is the verdict
ol check M09.6 --ref-deps   # only if you skipped rt.02 or M02.1
ol diff  M09.6              # after passing: your code against the reference
```

---

## 1. Why now

Every softmax in your model exponentiates its logits: the sampler (L8.1) once per token, attention once per query and key. In Part 9 those loops move into C (`L9.2` online softmax, `L9.3` FlashAttention, and SiLU in `L9.6`, which is $x/(1 + e^{-x})$), and each of them calls an exponential per element. The C library's `expf` is correct, but it is opaque, its speed and accuracy differ between macOS and Linux (which breaks bitwise reproducibility between the two CI runners), and it cannot be inlined into a vector loop. You already derived the algorithm in M02.1 (`exp_range_reduced`, in float64). This module ports it to float32 C with an accuracy contract in ulps, and the edges (overflow at 88.72, underflow below $-87.34$, $\pm\infty$, NaN) that a float64 prototype never meets.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the input | `float` |
| $k$ | the integer nearest $x / \ln 2$ (ties to even) | `int`, $-150 \le k \le 128$ in range |
| $r$ | the reduced argument $x - k \ln 2$ | `float`, $\lvert r \rvert \le \ln(2)/2 \approx 0.3466$ |
| $p(r)$ | the polynomial approximating $e^r$ | `float` |
| $L_{hi}, L_{lo}$ | $\ln 2$ split: $L_{hi} = 355/512 = 0.693359375$, $L_{lo} = \ln 2 - L_{hi} \approx -2.1219444 \times 10^{-4}$ | `float` constants |
| $R_7(r)$ | the remainder of the degree-7 Taylor polynomial | |
| $\mathrm{ulp}(y)$ | the spacing of float32 values at $y$ | |

### 2.1 Range reduction

$e^x$ is defined everywhere but a polynomial is accurate only near its center. The identity $e^{a + b} = e^a e^b$ moves any input into a small interval: write $x = k \ln 2 + r$, so

$$e^x = e^{k \ln 2} e^r = 2^k e^r.$$

Choosing $k = \mathrm{round}(x / \ln 2)$ puts $r$ in $[-\ln(2)/2, \ln(2)/2]$. Multiplying by $2^k$ is free in binary floating point: $2^k$ (for $-126 \le k \le 127$) is the float with exponent field $k + 127$ and mantissa 0, and multiplying by it changes only the exponent of the result, so it is exact whenever the result stays normal.

### 2.2 Computing $r$ without losing it

$r = x - k \ln 2$ subtracts two nearly equal numbers when $x$ is large: at $x = 88$, $k = 127$ and $k \ln 2 \approx 88.03$. In float32, $k \cdot \mathrm{fl}(\ln 2)$ has an error of about $127 \times 2^{-25} \approx 4 \times 10^{-6}$, which is relative error $10^{-4}$ in $r \approx -0.03$ and therefore in the result: a hundred ulps. **Cody and Waite's** fix: split $\ln 2 = L_{hi} + L_{lo}$ with $L_{hi}$ having few significant bits. $L_{hi} = 0.693359375 = 355/512$ has 9 significant bits and $|k| \le 150$ has 8, so $k L_{hi}$ fits in 17 bits and is **exact** in float32; $x - k L_{hi}$ is then exact too (the two are within a factor of 2 of each other, Sterbenz's lemma). The remaining $k L_{lo}$ is tiny, so its rounding error is tiny:

$$r = (x - k L_{hi}) - k L_{lo}.$$

### 2.3 The polynomial

Taylor's theorem (M02.1) for $e^r$ at 0 with degree $n$: $e^r = \sum_{j=0}^{n} r^j/j! + R_n(r)$ with $|R_n(r)| \le e^{|r|} |r|^{n+1}/(n+1)!$. On $|r| \le 0.3466$ and relative to $e^r \ge e^{-0.3466}$:

| Degree $n$ | Relative remainder bound $e^{2\lvert r \rvert}\lvert r\rvert^{n+1}/(n+1)!$ | In float32 ulps of the result |
|---|---|---|
| 6 | $2.4 \times 10^{-7}$ | about 3: too many |
| 7 | $1.1 \times 10^{-8}$ | about 0.1: rounding dominates |

Evaluate it with **Horner's rule** (M00.4): $p = 1 + r(1 + r(\tfrac12 + r(\tfrac16 + \dots + r \tfrac{1}{5040})))$, seven multiply-adds from the innermost coefficient out. Each step rounds once, and the sum of those roundings plus the reduction is what the tests measure: 1.21 ulp at worst for the reference. (A **minimax** polynomial, fitted to minimize the worst error on the interval instead of matching derivatives at 0, gets the same accuracy at degree 5. That is what production libraries ship; see Going further.)

### 2.4 Scaling by $2^k$ at the edges

The largest float is just below $2^{128}$, so $e^x$ overflows for $x > \ln(\mathrm{FLT\_MAX}) = 88.7228391$; the largest float below that is `0x1.62e42ep+6` $= 88.7228317$, where $k = 128$. One past the largest exponent: $2^{128}$ is not a float. Below $x = \ln(2^{-126}) = -87.3365$ the result is subnormal, and below $\ln(2^{-150}) = -103.972$ it rounds to 0, where $k$ reaches $-150$. Both ends fall outside $[-126, 127]$, so apply $2^k$ as $2^{k_1} \cdot 2^{k_2}$ with $k_1 = k/2$ (C rounds toward zero) and $k_2 = k - k_1$, both in $[-75, 64]$: the first product is exact and the second rounds once, even into the subnormal range. The specials come first: NaN returns NaN (converting NaN to `int` is undefined behavior in C, and UBSan stops the test), $x > 88.7228317$ returns $+\infty$, $x < -103.97208$ (and $-\infty$) returns $+0$.

## 3. Worked example by hand

$x = 1$.

| Step | Computation | Value |
|---|---|---|
| $k$ | $1 \times 1.442695 = 1.442695$, nearest integer | 1 |
| $x - k L_{hi}$ | $1 - 0.693359375$ (exact) | 0.306640625 |
| $r$ | $0.306640625 - 1 \times (-2.1219444 \times 10^{-4})$ | 0.30685282 |
| Horner | $\tfrac{1}{5040} = 0.00019841$; $\times r + \tfrac{1}{720} = 0.00144977$; $\times r + \tfrac{1}{120} = 0.00877820$; $\times r + \tfrac1{24} = 0.04436028$; $\times r + \tfrac16 = 0.18027875$; $\times r + \tfrac12 = 0.55531910$; $\times r + 1 = 1.17040120$; $\times r + 1$ | $p = 1.3591409$ |
| scale | $k_1 = 0$, $k_2 = 1$: $p \cdot 2^0 \cdot 2^1$ | 2.7182817 |

$e = 2.718281828\ldots$ and the float32 nearest to it is `2.7182817`: the kernel is exact here. With $x = -1$: $k = -1$, $r = -0.30685282$, $p = 0.7357589$, result `0.36787945`. With $x = 10$: $k = 14$, $r = 0.29593948$, $p = 1.3443887$, result `22026.465`. These are the first assertions of `hand_example`.

## 4. The interface

```c
/* tinyllm/numerics.h, M09.6 section */
float tl_expf(float x);                                /* within 2 ulp where e^x is normal; > 88.7228317 -> +inf; -inf -> +0; NaN -> NaN */
void  tl_exp_f32(const float *x, float *y, int64_t n); /* elementwise; y == x allowed */
```

Write `pow2i(k)` (the float $2^k$ from its bit pattern, for $-126 \le k \le 127$) as a `static` helper. Keep each multiply-add of the Horner loop a separate statement or keep `#pragma STDC FP_CONTRACT OFF`, so the compiler does not fuse them (a fused multiply-add is more accurate, but differs between compilers and machines).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3: $e^1, e^{-1}, e^{10}$ give the bits above; $e^{\pm 0} = 1$ | you and the test agree on the algorithm |
| `million_samples_within_2_ulp` | differential | $10^6$ seeded $x \in [-87.34, 88.72]$ within 2 ulp of `exp` in double | the softmax's accuracy budget (M09.3) |
| `sweep_of_minus_one_to_one` | property | every 512th float of $[-1, 1]$ (4 million inputs) within 2 ulp | small logits after max subtraction |
| `reduction_boundaries` | boundary | 64 floats each side of $(j + \tfrac12)\ln 2$ for every $j$: the largest $\lvert r \rvert$ | the worst case of the polynomial |
| `overflow_edge` | boundary | $e^{88.7228317}$ finite and within 2 ulp; the next float, `FLT_MAX`, and $+\infty$ give $+\infty$ | $k = 128$ needs the split scale |
| `underflow_edge` | boundary | $-87.3$ within 2 ulp; below the normal range $0 \le y \le$ `FLT_MIN` and within one subnormal step (or 0); $e^{-\infty} = +0$ | masked attention scores are $-\infty$ |
| `nan_in_nan_out` | boundary | NaN stays NaN, with no undefined behavior | a NaN logit stays visible |
| `array_matches_scalar_and_aliases` | unit | the array form gives the scalar's bits, writes exactly $n$, works in place | the softmax row in place |
| `hand_example` | unit, smoke | the standalone C implementation evaluates $x = 1$ within $10^{-8}$ of $e$ | checks the worked reduction |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. stopping the Taylor polynomial at $r^6$ | about 3 ulp near $\lvert r \rvert = 0.35$ | `million_samples_within_2_ulp` (mutant `s01`) |
| 2. reducing with one product, `x - k * 0.69314718f` | tens of ulps for large $\lvert x \rvert$ | `million_samples_within_2_ulp` (mutant `s02`) |
| 3. $k$ by truncation, `(int)(x / ln 2)` | $\lvert r \rvert$ up to $\ln 2$, where degree 7 is 20 ulp off | `million_samples_within_2_ulp` (mutant `s03`) |
| 4. $2^k$ built in one piece | $e^{88.72} = \infty$ (exponent field 255) and garbage below $-87.3$ | `overflow_edge`, `underflow_edge` (mutant `s04`) |
| 5. the overflow test written `x >= 88.7228317f` | the largest finite result becomes $\infty$ | `overflow_edge` (mutant `s05`) |
| 6. no NaN check before `(int)` | undefined behavior: UBSan aborts | `nan_in_nan_out` (mutant `s06`) |
| 7. no underflow cutoff | $(int)(-\infty)$ is undefined; large negative $k$ builds a garbage exponent | `underflow_edge` (mutant `s07`) |
| 8. the array loop off by one | the last element unwritten, or one past the end overwritten | `array_matches_scalar_and_aliases` (mutants `s08`, `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot of every stub, and the status and error support used by `tl_exp_f32` |
| Back | `M02.1` | `exp_range_reduced` is this algorithm in float64; the differential test compares the two |
| Back | `M00.4` | Horner's rule for the polynomial (reading) |
| Back | `M09.1` | exponent fields, ulps, subnormals (reading) |
| Forward | `L9.2` | the 3-pass and online softmax kernels exponentiate each logit minus the row maximum |
| Forward | `L9.3` | FlashAttention's running softmax rescales with $e^{m_{old} - m_{new}}$ |
| Forward | `L9.4` | paged decode attention's softmax over the keys of each block table |
| Forward | `L9.6` | SiLU $x / (1 + e^{-x})$ in `tl_silu_mul_f32` |

If you skip this module, `ol check L9.2` stops with `BLOCKED ... needs M09.6`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| degree-7 Taylor | SLEEF `expf`, Cephes `expf` | a degree-5 minimax polynomial (Remez algorithm) with the same accuracy and two fewer multiplies | SLEEF `src/libm/sleefsp.c` (`xexpf`); Cephes `expf.c` |
| scalar loop | SLEEF and XNNPACK vector `exp` | 8 or 16 lanes per instruction; the reduction rounds with the "add $1.5 \times 2^{23}$" trick instead of `rintf` | XNNPACK `src/f32-vscaleexpminusmax/` |
| $\le 2$ ulp | CORE-MATH `expf` | correctly rounded for every input, with a proof | the CORE-MATH project (Inria) |
| `tl_expf` in softmax | FlashAttention-3 | `exp2` with the $\log_2 e$ factor folded into the score scale, so the reduction is a plain integer split | Shah et al., *FlashAttention-3* (2024), section 3 |
