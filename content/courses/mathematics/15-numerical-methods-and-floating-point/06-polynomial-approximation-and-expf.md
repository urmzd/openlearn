# Polynomial approximation and range reduction: expf in C

Every softmax in a language model exponentiates its logits: the sampler once per token, attention once per query and key. In *AI Engineering* those loops move into C, in [Softmax in C: three-pass and online two-pass](../../ai-engineering/22-kernels/04-softmax-three-pass-and-online.md), [FlashAttention forward in C](../../ai-engineering/22-kernels/05-flash-attention-forward.md), and the SiLU of [Elementwise kernels in C](../../ai-engineering/22-kernels/08-elementwise-kernels.md), which is $x/(1 + e^{-x})$. Each of them calls an exponential per element. The C library's `expf` is correct, but it is opaque, its speed and accuracy differ between macOS and Linux (which breaks bitwise reproducibility between two CI runners), and it cannot be inlined into a vector loop.

The algorithm is already derived: `exp_range_reduced` in [Taylor series, remainder bounds, and range reduction](../08-calculus-2/01-taylor-series.md) computes it in float64. This optional section ports it to float32 C with an accuracy contract in ulps, and handles the edges that a float64 prototype never meets: overflow at 88.72, underflow below $-87.34$, $\pm\infty$, and NaN. You will see how range reduction shrinks an infinite domain to a tiny interval exactly, why the subtraction that performs it must be split in two, how to choose a polynomial degree from a remainder bound, and how to scale the result by $2^k$ when $k$ runs past both ends of the exponent range.

Besides the Taylor chapter, you need Horner's rule from [Polynomials, Horner's rule, and stable quadratic roots](../06-precalculus/04-polynomials-horner-and-stable-roots.md), exponent fields and subnormals from [IEEE 754 anatomy, ulps, and round to nearest even](01-ieee-754.md), and the runtime support of [C runtime support and an arena allocator with marks](../../ai-engineering/22-kernels/01-arena-allocator-with-marks.md). The solution, [`expf.c`](../../../../course/solutions/c/src/numerics/expf.c) (`tl_expf`, `tl_exp_f32`, and the helper `pow2i`), is a standalone C file built under ASan and UBSan. Its interface is in [`numerics.h`](../../../../course/contracts/c/include/tinyllm/numerics.h), under the rules of [`c/ABI.md`](../../../../course/contracts/c/ABI.md).

> **Learning objectives**
> - Reduce any float32 input to $e^x = 2^k e^r$ with $|r| \le \ln(2)/2$, and explain why multiplying by $2^k$ is free.
> - Use the Cody and Waite split of $\ln 2$ to compute $r$ without cancellation error.
> - Choose a Taylor polynomial degree from its remainder bound and evaluate it with Horner's rule.
> - Apply $2^k$ for $k \in [-150, 128]$ without overflow or undefined behavior, and handle NaN and the overflow and underflow edges exactly.

## The notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the input | `float` |
| $k$ | the integer nearest $x / \ln 2$ (ties to even) | `int`, $-150 \le k \le 128$ in range |
| $r$ | the reduced argument $x - k \ln 2$ | `float`, $\lvert r \rvert \le \ln(2)/2 \approx 0.3466$ |
| $p(r)$ | the polynomial approximating $e^r$ | `float` |
| $L_{hi}, L_{lo}$ | $\ln 2$ split: $L_{hi} = 355/512 = 0.693359375$, $L_{lo} = \ln 2 - L_{hi} \approx -2.1219444 \times 10^{-4}$ | `float` constants |
| $R_7(r)$ | the remainder of the degree-7 Taylor polynomial | |
| $\mathrm{ulp}(y)$ | the spacing of float32 values at $y$ | |

## Range reduction

$e^x$ is defined everywhere, but a polynomial is accurate only near its center. The identity $e^{a + b} = e^a e^b$ moves any input into a small interval: write $x = k \ln 2 + r$, so

$$e^x = e^{k \ln 2} e^r = 2^k e^r.$$

Choosing $k = \mathrm{round}(x / \ln 2)$ puts $r$ in $[-\ln(2)/2, \ln(2)/2]$. Multiplying by $2^k$ is free in binary floating point: $2^k$ (for $-126 \le k \le 127$) is the float with exponent field $k + 127$ and mantissa 0, and multiplying by it changes only the exponent of the result, so it is exact whenever the result stays normal.

## Computing $r$ without losing it

The reduction itself hides a trap. $r = x - k \ln 2$ subtracts two nearly equal numbers when $x$ is large: at $x = 88$, $k = 127$ and $k \ln 2 \approx 88.03$. In float32, $k \cdot \mathrm{fl}(\ln 2)$ has an error of about $127 \times 2^{-25} \approx 4 \times 10^{-6}$, which is a relative error of $10^{-4}$ in $r \approx -0.03$ and therefore in the result: a hundred ulps.

**Cody and Waite's** fix is to split $\ln 2 = L_{hi} + L_{lo}$ with $L_{hi}$ having few significant bits. $L_{hi} = 0.693359375 = 355/512$ has 9 significant bits and $|k| \le 150$ has 8, so $k L_{hi}$ fits in 17 bits and is **exact** in float32. $x - k L_{hi}$ is then exact too, because the two are within a factor of 2 of each other (Sterbenz's lemma). The remaining $k L_{lo}$ is tiny, so its rounding error is tiny:

$$r = (x - k L_{hi}) - k L_{lo}.$$

## The polynomial

With $r$ confined to a small interval, a polynomial can do the rest. Taylor's theorem for $e^r$ at 0 with degree $n$ gives $e^r = \sum_{j=0}^{n} r^j/j! + R_n(r)$ with $|R_n(r)| \le e^{|r|} |r|^{n+1}/(n+1)!$. On $|r| \le 0.3466$ and relative to $e^r \ge e^{-0.3466}$, the two candidate degrees compare like this:

| Degree $n$ | Relative remainder bound $e^{2\lvert r \rvert}\lvert r\rvert^{n+1}/(n+1)!$ | In float32 ulps of the result |
|---|---|---|
| 6 | $2.4 \times 10^{-7}$ | about 3: too many |
| 7 | $1.1 \times 10^{-8}$ | about 0.1: rounding dominates |

Evaluate it with **Horner's rule**: $p = 1 + r(1 + r(\tfrac12 + r(\tfrac16 + \dots + r \tfrac{1}{5040})))$, seven multiply-adds from the innermost coefficient out. Each step rounds once, and the sum of those roundings plus the reduction is what an accuracy check measures: 1.21 ulp at worst for the solution.

> **Note:** A **minimax** polynomial, fitted to minimize the worst error on the interval instead of matching derivatives at 0, gets the same accuracy at degree 5. That is what production libraries ship; see the end of the section.

## Scaling by $2^k$ at the edges

The last step, multiplying by $2^k$, is only free in the middle of the range. The largest float is just below $2^{128}$, so $e^x$ overflows for $x > \ln(\mathrm{FLT\_MAX}) = 88.7228391$. The largest float below that is `0x1.62e42ep+6` $= 88.7228317$, where $k = 128$: one past the largest exponent, since $2^{128}$ is not a float. Below $x = \ln(2^{-126}) = -87.3365$ the result is subnormal, and below $\ln(2^{-150}) = -103.972$ it rounds to 0, where $k$ reaches $-150$.

Both ends fall outside $[-126, 127]$, so apply $2^k$ as $2^{k_1} \cdot 2^{k_2}$ with $k_1 = k/2$ (C rounds toward zero) and $k_2 = k - k_1$, both in $[-75, 64]$. The first product is exact and the second rounds once, even into the subnormal range.

The special values come first. NaN returns NaN (converting NaN to `int` is undefined behavior in C, and UBSan aborts on it), $x > 88.7228317$ returns $+\infty$, and $x < -103.97208$ (and $-\infty$) returns $+0$.

## Working it by hand

Run $x = 1$ through every stage:

| Step | Computation | Value |
|---|---|---|
| $k$ | $1 \times 1.442695 = 1.442695$, nearest integer | 1 |
| $x - k L_{hi}$ | $1 - 0.693359375$ (exact) | 0.306640625 |
| $r$ | $0.306640625 - 1 \times (-2.1219444 \times 10^{-4})$ | 0.30685282 |
| Horner | $\tfrac{1}{5040} = 0.00019841$; $\times r + \tfrac{1}{720} = 0.00144977$; $\times r + \tfrac{1}{120} = 0.00877820$; $\times r + \tfrac1{24} = 0.04436028$; $\times r + \tfrac16 = 0.18027875$; $\times r + \tfrac12 = 0.55531910$; $\times r + 1 = 1.17040120$; $\times r + 1$ | $p = 1.3591409$ |
| scale | $k_1 = 0$, $k_2 = 1$: $p \cdot 2^0 \cdot 2^1$ | 2.7182817 |

$e = 2.718281828\ldots$, and the float32 nearest to it is `2.7182817`: the kernel is exact here. With $x = -1$: $k = -1$, $r = -0.30685282$, $p = 0.7357589$, result `0.36787945`. With $x = 10$: $k = 14$, $r = 0.29593948$, $p = 1.3443887$, result `22026.465`. These become the hand-worked test cases.

## Designing the API

```c
/* tinyllm/numerics.h, expf section */
float tl_expf(float x);                                /* within 2 ulp where e^x is normal; > 88.7228317 -> +inf; -inf -> +0; NaN -> NaN */
void  tl_exp_f32(const float *x, float *y, int64_t n); /* elementwise; y == x allowed */
```

The solution writes `pow2i(k)` (the float $2^k$ from its bit pattern, for $-126 \le k \le 127$) as a `static` helper. It keeps `#pragma STDC FP_CONTRACT OFF` over the Horner steps, so the compiler does not fuse them: a fused multiply-add is more accurate, but whether one is used differs between compilers and machines.

## Building it step by step

Six decisions shape the code. First, answer the specials before any integer conversion: NaN, overflow, and underflow are decided by comparisons on $x$, and doing them first keeps `(int)` away from NaN and $\pm\infty$, where the conversion is undefined behavior. Second, reduce with the nearest integer: $k = \mathrm{rint}(x \log_2 e)$ puts $r$ in $[-\ln(2)/2, \ln(2)/2]$, while truncation would let $\lvert r\rvert$ reach $\ln 2$, where the degree-7 polynomial is 20 ulp off. Third, subtract $k \ln 2$ in two parts, so that $k L_{hi}$ and $x - kL_{hi}$ are exact and only the small $kL_{lo}$ rounds. Fourth, pick the degree from the remainder table: on $\lvert r\rvert \le 0.3466$, degree 7 leaves about 0.1 ulp of truncation and degree 6 about 3 ulp. Taylor coefficients $1/j!$ are used because the remainder bound is provable by hand. Fifth, evaluate with Horner, unfused: seven multiply-adds, innermost first, each rounding once, with `FP_CONTRACT OFF` keeping the result identical across compilers. Sixth, scale by $2^k$ in two exact halves, since $k \in [-150, 128]$ overflows the normal exponent range at both ends; $2^k = 2^{k_1}2^{k_2}$ with $k_1, k_2 \in [-75, 64]$, each a valid normal float built from its bit pattern.

The constants and the power-of-two helper come first:

```c
#pragma STDC FP_CONTRACT OFF

#define TL_EXP_HI 0x1.62e42ep+6f   /* 88.7228317: largest x with e^x finite */
#define TL_EXP_LO -0x1.9fe368p+6f  /* -103.972076: below this e^x rounds to 0 */
#define TL_LOG2E 0x1.715476p+0f    /* 1 / ln 2 */
#define TL_LN2_HI 0.693359375f     /* 355 / 512: 9 significant bits */
#define TL_LN2_LO -2.12194440e-4f  /* ln 2 - TL_LN2_HI */

/* 2^k for -126 <= k <= 127: exponent field k + 127, mantissa 0. */
static float pow2i(int k) {
    uint32_t u = (uint32_t)(k + 127) << 23;
    float f;
    memcpy(&f, &u, sizeof f);      /* the defined way to reinterpret bits */
    return f;
}
```

The constants are written in hex-float form where the exact bit pattern matters. `TL_EXP_HI` is the largest float whose exponential is finite, so the overflow test is `x > TL_EXP_HI`, not `>=`. `TL_LN2_HI` is $355/512$, exactly representable with 9 significant bits; `TL_LN2_LO` is negative because $355/512$ is slightly *larger* than $\ln 2$. `pow2i` only ever receives $k \in [-75, 64]$, so its exponent field is always between 52 and 191, a normal float.

The kernel itself follows the six decisions in order:

```c
float tl_expf(float x) {
    if (isnan(x)) return x + x;            /* NaN in, quiet NaN out */
    if (x > TL_EXP_HI) return INFINITY;    /* also +inf */
    if (x < TL_EXP_LO) return 0.0f; /* also -inf */

    float kf = rintf(x * TL_LOG2E); /* nearest integer, ties to even */
    int k = (int)kf;                /* safe: |kf| <= 150 after the cutoffs */
    float r = x - kf * TL_LN2_HI;   /* exact: kf * LN2_HI has <= 17 bits, Sterbenz */
    r = r - kf * TL_LN2_LO;         /* the small correction, one small rounding */

    float p = 1.0f / 5040.0f;       /* 1/7!: Horner from the innermost coefficient */
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;               /* p ~ e^r on |r| <= 0.3466 */

    int k1 = k / 2, k2 = k - k1; /* both in [-75, 64] */
    return p * pow2i(k1) * pow2i(k2);   /* first product exact, second rounds once */
}

void tl_exp_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = tl_expf(x[i]);   /* read then write: y == x is safe */
}
```

Trace the hand example for $x = 1$: `kf = rintf(1.442695) = 1`; $r = 1 - 0.693359375 = 0.306640625$ exactly, then $0.306640625 + 2.1219444 \times 10^{-4} = 0.30685282$; the Horner chain gives $p = 1.3591409$; $k_1 = 0$, $k_2 = 1$, and the result is $1.3591409 \cdot 1 \cdot 2 = 2.7182817$, the float32 nearest $e$.

The edges are where the split scale earns its keep. At the overflow edge $x = 88.7228317$, $k = 128$: a single `pow2i(128)` would be exponent field 255 (infinity), but $k_1 = 64$ and $k_2 = 64$ are both fine, and $p \cdot 2^{64} \cdot 2^{64}$ is finite because $p < 1$ there. At $x = -100$, $k = -144$: $k_1 = -72$, $k_2 = -72$, the first product is a normal float and the second rounds once into the subnormal range. The left-to-right evaluation of `p * pow2i(k1) * pow2i(k2)` is what makes the first product the exact one. The full solution is in [`expf.c`](../../../../course/solutions/c/src/numerics/expf.c).

### Why this design

The kernel costs three comparisons, one `rintf`, one multiply for $k$, two multiply-subtracts for $r$, seven multiply-adds, two integer shifts, and two multiplies: about 25 operations, no table, no division. The array form is $O(n)$ and simple enough for a compiler to vectorize.

The C library's `expf` is correctly rounded or close, but its speed and last-bit results differ between platforms, and it cannot be traced step by step against the Python twin. This version is within 1.21 ulp everywhere and identical on any IEEE platform with `FP_CONTRACT OFF`.

A degree-5 minimax polynomial reaches the same accuracy with two fewer multiply-adds. Taylor is kept because its remainder bound is derivable by hand and it matches the Taylor chapter's coefficients exactly. Doing the reduction in `double` would also avoid the cancellation, but it doubles the width of a hot path and does not vectorize as well in float32 lanes; the Cody and Waite split costs one extra multiply-subtract. And the split scale beats a special subnormal path: two multiplies by normal powers of two handle both edges with one rounding, while branching on $k$ and building subnormal patterns by hand adds branches and bugs.

## What can go wrong

The tests begin with the hand-worked inputs ($e^1$, $e^{-1}$, $e^{10}$ give the bits above, and $e^{\pm 0} = 1$) so everyone agrees on the algorithm. A differential test draws $10^6$ seeded $x \in [-87.34, 88.72]$ and requires each to be within 2 ulp of `exp` in double, which is the accuracy the softmax's tolerance budget from [Error analysis, condition numbers, and tolerance budgets](03-error-analysis-condition-numbers-and-tolerance-budgets.md) relies on. A sweep of every 512th float of $[-1, 1]$ (4 million inputs) covers the small logits that appear after max subtraction. The reduction boundaries get their own test: 64 floats on each side of $(j + \tfrac12)\ln 2$ for every $j$, which is where $\lvert r \rvert$ is largest and the polynomial is at its worst.

At the overflow edge, $e^{88.7228317}$ must be finite and within 2 ulp, while the next float, `FLT_MAX`, and $+\infty$ give $+\infty$; this is the $k = 128$ case that needs the split scale. At the underflow edge, $-87.3$ must be within 2 ulp; below the normal range the result must satisfy $0 \le y \le$ `FLT_MIN` and be within one subnormal step (or 0); and $e^{-\infty} = +0$, since masked attention scores are $-\infty$. NaN must stay NaN, with no undefined behavior, so that a NaN logit stays visible. The array form must give the scalar's bits, write exactly $n$ elements, and work in place, because the softmax row is updated in place.

The mistakes these tests catch:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| stopping the Taylor polynomial at $r^6$ | about 3 ulp near $\lvert r \rvert = 0.35$ | the Horner chain starts at $1/7!$ |
| reducing with one product, `x - k * 0.69314718f` | tens of ulps for large $\lvert x \rvert$ | `TL_LN2_HI` and `TL_LN2_LO`, two subtractions |
| $k$ by truncation, `(int)(x / ln 2)` | $\lvert r \rvert$ up to $\ln 2$, where degree 7 is 20 ulp off | `rintf`, nearest with ties to even |
| $2^k$ built in one piece | $e^{88.72} = \infty$ (exponent field 255) and garbage below $-87.3$ | `pow2i(k1) * pow2i(k2)` with both halves in $[-75, 64]$ |
| the overflow test written `x >= 88.7228317f` | the largest finite result becomes $\infty$ | `x > TL_EXP_HI` |
| no NaN check before `(int)` | undefined behavior: UBSan aborts | `isnan` is the first line |
| no underflow cutoff | $(int)(-\infty)$ is undefined; large negative $k$ builds a garbage exponent | `x < TL_EXP_LO` returns 0 before $k$ is formed |
| the array loop off by one | the last element unwritten, or one past the end overwritten | `for (int64_t i = 0; i < n; i++)` |

## Where this leads

`exp_range_reduced` in [Taylor series, remainder bounds, and range reduction](../08-calculus-2/01-taylor-series.md) is this algorithm in float64, and the differential test compares the two. In *AI Engineering*, the three-pass and online softmax kernels of [Softmax in C: three-pass and online two-pass](../../ai-engineering/22-kernels/04-softmax-three-pass-and-online.md) exponentiate each logit minus the row maximum; [FlashAttention forward in C](../../ai-engineering/22-kernels/05-flash-attention-forward.md) rescales its running softmax with $e^{m_{old} - m_{new}}$; [Paged attention for decode in C](../../ai-engineering/22-kernels/06-paged-attention-decode.md) computes the softmax over the keys of each block table; and [Elementwise kernels in C](../../ai-engineering/22-kernels/08-elementwise-kernels.md) computes SiLU $x / (1 + e^{-x})$ in `tl_silu_mul_f32`. All of them call `tl_expf` per element.

Production implementations push each piece further. SLEEF's `expf` (`src/libm/sleefsp.c`, `xexpf`) and Cephes's `expf.c` use a degree-5 minimax polynomial (Remez algorithm) with the same accuracy and two fewer multiplies. SLEEF and XNNPACK vector `exp` routines process 8 or 16 lanes per instruction and round the reduction with the "add $1.5 \times 2^{23}$" trick instead of `rintf` (XNNPACK `src/f32-vscaleexpminusmax/`). CORE-MATH's `expf` is correctly rounded for every input, with a proof. And FlashAttention-3 uses `exp2` with the $\log_2 e$ factor folded into the score scale, so the reduction is a plain integer split.

### Further reading

- Muller, *Elementary Functions: Algorithms and Implementation* (3rd ed.), chapters 2 and 11.
- Cody and Waite, *Software Manual for the Elementary Functions* (1980).
- Trefethen, *Approximation Theory and Approximation Practice*, chapter 10.
- Shah et al., *FlashAttention-3* (2024), section 3.
- The CORE-MATH project (Inria).

## Key terms

- **Range reduction**: rewriting $e^x = 2^k e^r$ so that the polynomial only needs to be accurate on a small interval around 0.
- **Cody and Waite split**: representing $\ln 2$ as a short high part, whose products with $k$ are exact, plus a small low part.
- **Sterbenz's lemma**: if two floats are within a factor of 2 of each other, their difference is exact.
- **Horner's rule**: evaluating a polynomial as nested multiply-adds from the innermost coefficient out.
- **Minimax polynomial**: the polynomial of a given degree that minimizes the worst-case error on an interval.
- **Floating-point contraction**: the compiler fusing `a * b + c` into a single fused multiply-add with one rounding.

## Summary

- Range reduction turns an infinite domain into a tiny one exactly: $e^x = 2^k e^r$ with $k = \mathrm{round}(x/\ln 2)$ and $|r| \le \ln(2)/2$, and $2^k$ costs nothing because it is an exponent field.
- The reduction must not round away $r$: $k \ln 2$ is subtracted in two parts (Cody and Waite), a 9-bit high part whose product with $k$ is exact and a low part for the rest.
- On $|r| \le 0.347$ a Taylor polynomial of degree 7 is accurate to about $10^{-8}$, a tenth of a float32 ulp, so the result is within 1.21 ulp everywhere; degree 6 is not enough.
- $k$ runs from $-150$ to $128$, past both ends of the float exponent range, so $2^k$ is applied in two halves; the edges overflow to $+\infty$ and underflow to $0$ exactly where the contract says.
- The C kernel is the Taylor chapter's algorithm in float32, within 2 ulp of `exp_range_reduced` on the float32 range.

## Review questions

1. Why is $L_{hi}$ chosen with only 9 significant bits?

   <details><summary>Answer</summary>

   So that $k L_{hi}$ is exact in float32: $\lvert k\rvert \le 150$ needs 8 bits and $L_{hi}$ has 9, so the product has at most 17 significant bits, well under 24. Then $x - kL_{hi}$ subtracts two floats within a factor of 2 of each other, which Sterbenz's lemma makes exact. All the rounding is pushed into $kL_{lo}$, which is about $3 \times 10^{-4}$ times smaller.

   </details>

2. Why does the overflow test use `>` and not `>=`?

   <details><summary>Answer</summary>

   `TL_EXP_HI` $= 88.7228317$ is the *largest* float whose exponential is still below `FLT_MAX`. With `>=`, that input would return $+\infty$ although its true result is finite and representable; the contract requires a finite value within 2 ulp there.

   </details>

3. Why is $2^k$ applied as two factors?

   <details><summary>Answer</summary>

   $k$ ranges from $-150$ to $128$, but a normal float's exponent only covers $2^{-126}$ to $2^{127}$. A single `pow2i(128)` would set the exponent field to 255 (infinity) and `pow2i(-150)` would underflow the field. Splitting into $k_1 = k/2$ and $k_2 = k - k_1$, both in $[-75, 64]$, keeps each factor a normal power of two; the first product is exact and the second rounds once, even when the result is subnormal.

   </details>

4. Why must the NaN and range checks come before `(int)kf`?

   <details><summary>Answer</summary>

   Converting NaN or an out-of-range float to `int` is undefined behavior in C, and under UBSan it aborts. After the three checks, $x \in [-103.97, 88.72]$, so $\lvert kf\rvert \le 150$ and the conversion is defined. The checks also give the exact contract values ($+\infty$, $+0$) instead of whatever the arithmetic would produce.

   </details>

5. What does `#pragma STDC FP_CONTRACT OFF` buy, if a fused multiply-add is more accurate?

   <details><summary>Answer</summary>

   Reproducibility. With contraction allowed, each compiler and target decides independently whether `p * r + c` becomes one FMA, so results differ in the last bit between machines and builds. Turning it off fixes the rounding sequence; the 1.21 ulp worst case is already inside the contract, so the extra accuracy is not needed.

   </details>
