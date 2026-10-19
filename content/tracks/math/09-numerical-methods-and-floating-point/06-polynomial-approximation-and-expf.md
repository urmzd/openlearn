# Polynomial approximation and range reduction: expf in C (optional)

## Overview

| | |
|---|---|
| **Module** | `M09.6` · side · C · Pass 6 · 1 to 2 h reading |
| **Solution** | [`course/solutions/c/src/numerics/expf.c`](../../../../course/solutions/c/src/numerics/expf.c): `tl_expf` and `tl_exp_f32` (and the helper `pow2i`) |
| **Contract** | [`course/contracts/c/include/tinyllm/numerics.h`](../../../../course/contracts/c/include/tinyllm/numerics.h) (the M09.6 section) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Needs** | [`rt.02` runtime support](../../../courses/build-an-llm/p09-kernels/02-arena-allocator-with-marks.md), [`M02.1` Taylor series and range reduction](../02-calculus-2/01-taylor-series.md) (`exp_range_reduced`, the Python twin). Reading: [`M00.4` Horner's rule](../00-precalculus/04-polynomials-horner-and-stable-roots.md), [`M09.1` IEEE 754](01-ieee-754.md) |
| **Used by** | `L9.2` softmax · `L9.3` FlashAttention · `L9.4` paged attention · `L9.6` SiLU, each calling `tl_expf` per element |
| **Optional depth** | Muller, *Elementary Functions: Algorithms and Implementation* (3rd ed.), ch. 2 and 11; Cody and Waite, *Software Manual for the Elementary Functions* (1980); Trefethen, *Approximation Theory and Approximation Practice*, ch. 10 |

## Key Takeaways

- **Range reduction turns an infinite domain into a tiny one exactly**: $e^x = 2^k e^r$ with $k = \mathrm{round}(x/\ln 2)$ and $|r| \le \ln(2)/2$, and $2^k$ costs nothing because it is an exponent field.
- **The reduction must not round away $r$**: $k \ln 2$ is subtracted in two parts (Cody and Waite), a 9-bit high part whose product with $k$ is exact and a low part for the rest.
- **On $|r| \le 0.347$ a Taylor polynomial of degree 7 is accurate to about $10^{-8}$**, a tenth of a float32 ulp, so the result is within 1.21 ulp everywhere; degree 6 is not enough.
- **$k$ runs from $-150$ to $128$**, past both ends of the float exponent range, so $2^k$ is applied in two halves; the edges overflow to $+\infty$ and underflow to $0$ exactly where the contract says.
- **The C kernel is M02.1's algorithm in float32**, within 2 ulp of `exp_range_reduced` on the float32 range.

## How to read this chapter

Sections 1 to 3 derive range reduction, the Cody and Waite split, the polynomial degree, and the edge scaling, and run $e^1$ by hand. Section 4 fixes the interface. Section 5 walks through `expf.c` statement by statement. The solution is a standalone C file built under ASan and UBSan; try to predict each constant from section 2 before reading it.

---

## 1. Why now

Every softmax in the model exponentiates its logits: the sampler (L8.1) once per token, attention once per query and key. In Part 9 those loops move into C (`L9.2` online softmax, `L9.3` FlashAttention, and SiLU in `L9.6`, which is $x/(1 + e^{-x})$), and each of them calls an exponential per element. The C library's `expf` is correct, but it is opaque, its speed and accuracy differ between macOS and Linux (which breaks bitwise reproducibility between the two CI runners), and it cannot be inlined into a vector loop. The algorithm is already derived in M02.1 (`exp_range_reduced`, in float64). This module ports it to float32 C with an accuracy contract in ulps, and the edges (overflow at 88.72, underflow below $-87.34$, $\pm\infty$, NaN) that a float64 prototype never meets.

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

Evaluate it with **Horner's rule** (M00.4): $p = 1 + r(1 + r(\tfrac12 + r(\tfrac16 + \dots + r \tfrac{1}{5040})))$, seven multiply-adds from the innermost coefficient out. Each step rounds once, and the sum of those roundings plus the reduction is what an accuracy check measures: 1.21 ulp at worst for the solution. (A **minimax** polynomial, fitted to minimize the worst error on the interval instead of matching derivatives at 0, gets the same accuracy at degree 5. That is what production libraries ship; see Going further.)

### 2.4 Scaling by $2^k$ at the edges

The largest float is just below $2^{128}$, so $e^x$ overflows for $x > \ln(\mathrm{FLT\_MAX}) = 88.7228391$; the largest float below that is `0x1.62e42ep+6` $= 88.7228317$, where $k = 128$. One past the largest exponent: $2^{128}$ is not a float. Below $x = \ln(2^{-126}) = -87.3365$ the result is subnormal, and below $\ln(2^{-150}) = -103.972$ it rounds to 0, where $k$ reaches $-150$. Both ends fall outside $[-126, 127]$, so apply $2^k$ as $2^{k_1} \cdot 2^{k_2}$ with $k_1 = k/2$ (C rounds toward zero) and $k_2 = k - k_1$, both in $[-75, 64]$: the first product is exact and the second rounds once, even into the subnormal range. The specials come first: NaN returns NaN (converting NaN to `int` is undefined behavior in C, and UBSan aborts on it), $x > 88.7228317$ returns $+\infty$, $x < -103.97208$ (and $-\infty$) returns $+0$.

## 3. Worked example by hand

$x = 1$.

| Step | Computation | Value |
|---|---|---|
| $k$ | $1 \times 1.442695 = 1.442695$, nearest integer | 1 |
| $x - k L_{hi}$ | $1 - 0.693359375$ (exact) | 0.306640625 |
| $r$ | $0.306640625 - 1 \times (-2.1219444 \times 10^{-4})$ | 0.30685282 |
| Horner | $\tfrac{1}{5040} = 0.00019841$; $\times r + \tfrac{1}{720} = 0.00144977$; $\times r + \tfrac{1}{120} = 0.00877820$; $\times r + \tfrac1{24} = 0.04436028$; $\times r + \tfrac16 = 0.18027875$; $\times r + \tfrac12 = 0.55531910$; $\times r + 1 = 1.17040120$; $\times r + 1$ | $p = 1.3591409$ |
| scale | $k_1 = 0$, $k_2 = 1$: $p \cdot 2^0 \cdot 2^1$ | 2.7182817 |

$e = 2.718281828\ldots$ and the float32 nearest to it is `2.7182817`: the kernel is exact here. With $x = -1$: $k = -1$, $r = -0.30685282$, $p = 0.7357589$, result `0.36787945`. With $x = 10$: $k = 14$, $r = 0.29593948$, $p = 1.3443887$, result `22026.465`. These are the hand-worked cases of section 4.

## 4. The interface

```c
/* tinyllm/numerics.h, M09.6 section */
float tl_expf(float x);                                /* within 2 ulp where e^x is normal; > 88.7228317 -> +inf; -inf -> +0; NaN -> NaN */
void  tl_exp_f32(const float *x, float *y, int64_t n); /* elementwise; y == x allowed */
```

The solution writes `pow2i(k)` (the float $2^k$ from its bit pattern, for $-126 \le k \le 127$) as a `static` helper. It keeps `#pragma STDC FP_CONTRACT OFF` over the Horner steps, so the compiler does not fuse them (a fused multiply-add is more accurate, but differs between compilers and machines).

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked inputs | unit, smoke | section 3: $e^1, e^{-1}, e^{10}$ give the bits above; $e^{\pm 0} = 1$ | everyone agrees on the algorithm |
| a million samples within 2 ulp | differential | $10^6$ seeded $x \in [-87.34, 88.72]$ within 2 ulp of `exp` in double | the softmax's accuracy budget (M09.3) |
| a sweep of $[-1, 1]$ | property | every 512th float of $[-1, 1]$ (4 million inputs) within 2 ulp | small logits after max subtraction |
| reduction boundaries | boundary | 64 floats each side of $(j + \tfrac12)\ln 2$ for every $j$: the largest $\lvert r \rvert$ | the worst case of the polynomial |
| the overflow edge | boundary | $e^{88.7228317}$ finite and within 2 ulp; the next float, `FLT_MAX`, and $+\infty$ give $+\infty$ | $k = 128$ needs the split scale |
| the underflow edge | boundary | $-87.3$ within 2 ulp; below the normal range $0 \le y \le$ `FLT_MIN` and within one subnormal step (or 0); $e^{-\infty} = +0$ | masked attention scores are $-\infty$ |
| NaN in, NaN out | boundary | NaN stays NaN, with no undefined behavior | a NaN logit stays visible |
| the array form matches the scalar and may alias | unit | the array form gives the scalar's bits, writes exactly $n$, works in place | the softmax row in place |

## 5. The solution, step by step

### From first principles

1. **Answer the specials before any integer conversion.** NaN, overflow, and underflow are decided by comparisons on $x$ (section 2.4). Doing them first keeps `(int)` away from NaN and $\pm\infty$, where the conversion is undefined behavior.
2. **Reduce with the nearest integer.** $k = \mathrm{rint}(x \log_2 e)$ puts $r$ in $[-\ln(2)/2, \ln(2)/2]$; truncation would let $\lvert r\rvert$ reach $\ln 2$ and the degree-7 polynomial would be 20 ulp off there.
3. **Subtract $k \ln 2$ in two parts.** $L_{hi}$ has 9 significant bits and $\lvert k\rvert \le 150$ has 8, so $k L_{hi}$ is exact and $x - kL_{hi}$ is exact (Sterbenz); $kL_{lo}$ is small, so its rounding is small (section 2.2).
4. **Pick the degree from the remainder table.** On $\lvert r\rvert \le 0.3466$ degree 7 leaves about 0.1 ulp of truncation, degree 6 about 3 ulp (section 2.3). Taylor coefficients $1/j!$ are used because the remainder bound is provable by hand.
5. **Evaluate with Horner, unfused.** Seven multiply-adds, innermost first, each rounding once; `FP_CONTRACT OFF` keeps the result identical across compilers.
6. **Scale by $2^k$ in two exact halves.** $k \in [-150, 128]$ overflows the normal exponent range at both ends, so $2^k = 2^{k_1}2^{k_2}$ with $k_1, k_2 \in [-75, 64]$, each a valid normal float built from its bit pattern.

### The code, annotated

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

The constants are written in hex-float form where the exact bit pattern matters: `TL_EXP_HI` is the largest float whose exponential is finite, so the overflow test is `x > TL_EXP_HI`, not `>=`. `TL_LN2_HI` is $355/512$, exactly representable with 9 significant bits; `TL_LN2_LO` is negative because $355/512$ is slightly *larger* than $\ln 2$. `pow2i` only ever receives $k \in [-75, 64]$, so its exponent field is always between 52 and 191, a normal float.

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

Trace section 3 for $x = 1$: `kf = rintf(1.442695) = 1`; $r = 1 - 0.693359375 = 0.306640625$ exactly, then $0.306640625 + 2.1219444 \times 10^{-4} = 0.30685282$; the Horner chain gives $p = 1.3591409$; $k_1 = 0$, $k_2 = 1$, and the result is $1.3591409 \cdot 1 \cdot 2 = 2.7182817$, the float32 nearest $e$. At the overflow edge $x = 88.7228317$, $k = 128$: one `pow2i(128)` would be exponent field 255 (infinity), but $k_1 = 64$, $k_2 = 64$ are both fine, and $p \cdot 2^{64} \cdot 2^{64}$ is finite because $p < 1$ there. At $x = -100$, $k = -144$: $k_1 = -72$, $k_2 = -72$, the first product is a normal float and the second rounds once into the subnormal range. The left-to-right evaluation of `p * pow2i(k1) * pow2i(k2)` is what makes the first product the exact one.

Full solution: [`expf.c`](../../../../course/solutions/c/src/numerics/expf.c).

### Why this solution

- **Cost.** Three comparisons, one `rintf`, one multiply for $k$, two multiply-subtracts for $r$, seven multiply-adds, two integer shifts, and two multiplies: about 25 operations, no table, no division. $O(n)$ for the array form, and simple enough for a compiler to vectorize.
- **Over the C library's `expf`.** It is correctly rounded or close, but its speed and last-bit results differ between platforms, and it cannot be traced step by step against the Python twin. This version is within 1.21 ulp everywhere and identical on any IEEE platform with `FP_CONTRACT OFF`.
- **Taylor over minimax.** A degree-5 minimax polynomial reaches the same accuracy with two fewer multiply-adds (Going further). Taylor is kept because its remainder bound is derivable by hand and it matches M02.1's coefficients exactly.
- **Two-part constant over a double reduction.** Doing the reduction in `double` would also avoid the cancellation, but it doubles the width of a hot path and does not vectorize as well in float32 lanes; the Cody and Waite split costs one extra multiply-subtract.
- **Split scale over a special subnormal path.** Two multiplies by normal powers of two handle both edges with one rounding; the alternative, branching on $k$ and building subnormal patterns by hand, adds branches and bugs.

### Explain it back

**1. Why is $L_{hi}$ chosen with only 9 significant bits?**

<details><summary>Answer</summary>

So that $k L_{hi}$ is exact in float32: $\lvert k\rvert \le 150$ needs 8 bits and $L_{hi}$ has 9, so the product has at most 17 significant bits, well under 24. Then $x - kL_{hi}$ subtracts two floats within a factor of 2 of each other, which Sterbenz's lemma makes exact. All the rounding is pushed into $kL_{lo}$, which is about $3 \times 10^{-4}$ times smaller.

</details>

**2. Why does the overflow test use `>` and not `>=`?**

<details><summary>Answer</summary>

`TL_EXP_HI` $= 88.7228317$ is the *largest* float whose exponential is still below `FLT_MAX`. With `>=`, that input would return $+\infty$ although its true result is finite and representable; the contract requires a finite value within 2 ulp there.

</details>

**3. Why is $2^k$ applied as two factors?**

<details><summary>Answer</summary>

$k$ ranges from $-150$ to $128$, but a normal float's exponent only covers $2^{-126}$ to $2^{127}$. A single `pow2i(128)` would set the exponent field to 255 (infinity) and `pow2i(-150)` would underflow the field. Splitting into $k_1 = k/2$ and $k_2 = k - k_1$, both in $[-75, 64]$, keeps each factor a normal power of two; the first product is exact and the second rounds once, even when the result is subnormal.

</details>

**4. Why must the NaN and range checks come before `(int)kf`?**

<details><summary>Answer</summary>

Converting NaN or an out-of-range float to `int` is undefined behavior in C, and under UBSan it aborts. After the three checks, $x \in [-103.97, 88.72]$, so $\lvert kf\rvert \le 150$ and the conversion is defined. The checks also give the exact contract values ($+\infty$, $+0$) instead of whatever the arithmetic would produce.

</details>

**5. What does `#pragma STDC FP_CONTRACT OFF` buy, if a fused multiply-add is more accurate?**

<details><summary>Answer</summary>

Reproducibility. With contraction allowed, each compiler and target decides independently whether `p * r + c` becomes one FMA, so results differ in the last bit between machines and builds. Turning it off fixes the rounding sequence; the 1.21 ulp worst case is already inside the contract, so the extra accuracy is not needed.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. stopping the Taylor polynomial at $r^6$ | about 3 ulp near $\lvert r \rvert = 0.35$ | the Horner chain starts at $1/7!$; step 4 |
| 2. reducing with one product, `x - k * 0.69314718f` | tens of ulps for large $\lvert x \rvert$ | `TL_LN2_HI` and `TL_LN2_LO`, two subtractions; step 3 |
| 3. $k$ by truncation, `(int)(x / ln 2)` | $\lvert r \rvert$ up to $\ln 2$, where degree 7 is 20 ulp off | `rintf`, nearest with ties to even; step 2 |
| 4. $2^k$ built in one piece | $e^{88.72} = \infty$ (exponent field 255) and garbage below $-87.3$ | `pow2i(k1) * pow2i(k2)` with both halves in $[-75, 64]$; step 6 |
| 5. the overflow test written `x >= 88.7228317f` | the largest finite result becomes $\infty$ | `x > TL_EXP_HI` |
| 6. no NaN check before `(int)` | undefined behavior: UBSan aborts | `isnan` is the first line; step 1 |
| 7. no underflow cutoff | $(int)(-\infty)$ is undefined; large negative $k$ builds a garbage exponent | `x < TL_EXP_LO` returns 0 before $k$ is formed; step 1 |
| 8. the array loop off by one | the last element unwritten, or one past the end overwritten | `for (int64_t i = 0; i < n; i++)` |

## 7. Where it's used next

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

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| degree-7 Taylor | SLEEF `expf`, Cephes `expf` | a degree-5 minimax polynomial (Remez algorithm) with the same accuracy and two fewer multiplies | SLEEF `src/libm/sleefsp.c` (`xexpf`); Cephes `expf.c` |
| scalar loop | SLEEF and XNNPACK vector `exp` | 8 or 16 lanes per instruction; the reduction rounds with the "add $1.5 \times 2^{23}$" trick instead of `rintf` | XNNPACK `src/f32-vscaleexpminusmax/` |
| $\le 2$ ulp | CORE-MATH `expf` | correctly rounded for every input, with a proof | the CORE-MATH project (Inria) |
| `tl_expf` in softmax | FlashAttention-3 | `exp2` with the $\log_2 e$ factor folded into the score scale, so the reduction is a plain integer split | Shah et al., *FlashAttention-3* (2024), section 3 |
