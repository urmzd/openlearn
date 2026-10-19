# Fast inverse square root in standalone C (optional)

## Overview

| | |
|---|---|
| **Module** | `M09.5` · side · C · Pass 6 · 1 to 2 h reading |
| **Solution** | [`course/solutions/c/src/numerics/rsqrt.c`](../../../../course/solutions/c/src/numerics/rsqrt.c): `tl_rsqrtf` and `tl_rsqrt_f32` (and the helper `rsqrt_normal`) |
| **Contract** | [`course/contracts/c/include/tinyllm/numerics.h`](../../../../course/contracts/c/include/tinyllm/numerics.h) (the M09.5 section) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Needs** | [`M01.2` Newton's method](../01-calculus-1/02-newtons-method.md) for the iteration; `M09.1` for float32 bit patterns and subnormals |
| **Used by** | Optional `L9.6` includes this module in its standalone C build |
| **Optional depth** | Lomont, *Fast Inverse Square Root* (2003, free); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 1; Süli and Mayers, *An Introduction to Numerical Analysis*, ch. 1 (fixed-point iteration) |

## Key Takeaways

- **Newton's method is a fixed-point iteration** $y \leftarrow g(y)$ whose map has $g'(r) = 0$ at the root, so the relative error squares every step: $e_{n+1} = \tfrac32 e_n^2 - \tfrac12 e_n^3$.
- **A float's bit pattern is a piecewise-linear $\log_2$**, so one integer subtraction, `0x5F3759DF - (bits >> 1)`, computes a first guess for $x^{-1/2}$ within 3.5%.
- **Two float32 Newton steps reach 4.7e-6 relative error, which is still about 73 ulp**; one more step in float64 lands within 0.5004 ulp of the true value, checked on every input.
- **Two binades decide every normal input**, because $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ holds bit for bit for this algorithm; subnormals need a rescale first.
- **This remains a standalone C kernel**: its numerical contract is derived from M01.2 and holds without any Python at run time.

## How to read this chapter

Sections 1 to 3 derive the iteration and the bit-pattern guess and run them by hand on $x = 4$. Section 4 fixes the interface. Section 5 walks through `rsqrt.c` statement by statement and explains why each one is written the way it is. The solution is a standalone C file built under ASan and UBSan; Python and C do not call into each other.

---

## 1. Why now

Every layer of the Llama-family model (L7.1) normalizes its activations with RMSNorm: $y_i = x_i \, w_i / \sqrt{\frac1d \sum_j x_j^2 + \epsilon}$. In Python that is one `np.sqrt` per row. In Part 9 the same normalization moves into C (`L9.6` `tl_rmsnorm_f32`), and the Rust engine calls it once per row, per layer, per token. The C library's `1.0f / sqrtf(s)` would work, but it is two of the slowest float instructions in a hot loop, it is a black box, and the course rule is that every kernel is something you can derive and test against the Python version (P6). The derivation already exists: M01.2's `rsqrt_newton` is Newton's method for $1/\sqrt{x}$ without a division. This module turns it into a C function with a contract stated in ulps, and proves the contract on every float32 input that matters.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the input, a positive float32 | `float` |
| $r = x^{-1/2}$ | the exact answer (a real number) | |
| $y_n$ | the $n$-th iterate, an approximation of $r$ | `float` or `double` |
| $e_n = (r - y_n)/r$ | the relative error of $y_n$ | |
| $g$ | the iteration map: $y_{n+1} = g(y_n)$ | |
| $I(x)$ | the 32-bit pattern of $x$ read as an unsigned integer | `uint32_t` |
| $E, M$ | the exponent field (8 bits) and mantissa field (23 bits) of $x$ | integers |
| $\mathrm{ulp}(r)$ | the spacing of float32 values at $r$: $2^{\lfloor \log_2 r \rfloor - 23}$ for normal $r$ | |
| $\sigma$ | the tuning offset in the bit-pattern logarithm, about 0.045 | |

### 2.1 Fixed-point iteration

A **fixed point** of a map $g$ is a value $r$ with $g(r) = r$. **Fixed-point iteration** starts from a guess $y_0$ and repeats $y_{n+1} = g(y_n)$. If $g$ is differentiable near $r$, a Taylor expansion (M02.1) gives

$$y_{n+1} - r = g(y_n) - g(r) = g'(r)(y_n - r) + \tfrac12 g''(\xi)(y_n - r)^2.$$

So the distance to $r$ shrinks by the factor $|g'(r)|$ each step when $|g'(r)| < 1$ (a **contraction**: linear convergence), and when $g'(r) = 0$ the first-order term vanishes and the distance is squared each step (**quadratic convergence**: the number of correct digits doubles).

### 2.2 Newton's method for $1/\sqrt{x}$ is a fixed-point iteration with $g'(r) = 0$

M01.2 applied Newton's method to $f(y) = 1/y^2 - x$, whose positive root is $r = x^{-1/2}$. With $f'(y) = -2/y^3$:

$$g(y) = y - \frac{f(y)}{f'(y)} = y + \frac{y^3}{2}\left(\frac{1}{y^2} - x\right) = y\left(\frac32 - \frac{x}{2}\, y^2\right).$$

No division, no square root: two multiplies, a subtract, a multiply. Its derivative is $g'(y) = \frac32 - \frac32 x y^2$, which is $0$ at $y = r$ because $x r^2 = 1$. Substituting $y_n = r(1 - e_n)$ and $x r^2 = 1$:

$$y_{n+1} = r(1 - e_n)\left(\tfrac32 - \tfrac12(1 - e_n)^2\right) = r\left(1 - \tfrac32 e_n^2 + \tfrac12 e_n^3\right),$$

so $e_{n+1} = \frac32 e_n^2 - \frac12 e_n^3$ exactly, in real arithmetic. From $e_0 = 0.0344$: $e_1 \le 1.8 \times 10^{-3}$, $e_2 \le 4.7 \times 10^{-6}$, $e_3 \le 3.3 \times 10^{-11}$. In float32 each step also adds a few units of $2^{-24}$ of rounding, which is why the error cannot go below about $10^{-7}$ in float32 no matter how many steps you take.

### 2.3 A first guess from the bits

A positive normal float is $x = (1 + m)\, 2^{E - 127}$ with $m = M/2^{23} \in [0, 1)$. Its bit pattern is $I(x) = 2^{23}(E + m)$ (the exponent field sits above the 23 mantissa bits). Since $\log_2(1 + m) \approx m + \sigma$ for $m \in [0, 1)$ with a small constant $\sigma$:

$$\log_2 x = E - 127 + \log_2(1 + m) \approx \frac{I(x)}{2^{23}} - 127 + \sigma.$$

The bit pattern is a scaled, shifted logarithm. Taking logs of $r = x^{-1/2}$ gives $\log_2 r = -\frac12 \log_2 x$; replacing both logs by the bit-pattern formula and solving for $I(r)$:

$$I(r) \approx \tfrac32\, 2^{23}(127 - \sigma) - \tfrac12 I(x).$$

With $\sigma \approx 0.0450466$ the constant is `0x5F3759DF`, and `I(x) >> 1` is $\frac12 I(x)$ rounded down. The resulting $y_0$ is never more than 3.44% from $r$. In C the only defined way to read a float's bits is `memcpy` into a `uint32_t` (a pointer cast breaks the aliasing rule); C++20 has `std::bit_cast`, Rust `f32::to_bits`, numpy `.view(np.uint32)`.

### 2.4 The algorithm, and why two binades check everything

The contract asks for 2 ulp. Two float32 steps give $4.7 \times 10^{-6}$, which is 40 to 73 ulp; a third float32 step gets to about 2.2 ulp because of its own rounding. So the third step runs in float64, where rounding is $2^{-53}$, and the result is rounded to float32 once:

1. `y = bits(0x5F3759DF - (bits(x) >> 1))`;
2. twice, in float32, `t = hx * y; t = t * y; t = 1.5f - t; y = y * t` with `hx = 0.5f * x`;
3. once, in float64, the same four statements on `(double)x` and `(double)y`; return `(float)y`.

Each statement is separate on purpose: C allows the compiler to fuse `a * b + c` inside one expression into a fused multiply-add with one rounding instead of two (`#pragma STDC FP_CONTRACT OFF` forbids it too), and M01.2 computes in exactly this order.

**Scaling.** For normal $x$, replace $x$ by $4x$: the pattern grows by $2^{24}$, so the guess's pattern shrinks by $2^{23}$, which halves $y_0$ exactly. Every product in steps 2 and 3 then scales by a power of two, which is exact, so $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ **bit for bit**. Every normal float is $4^j x$ for one $x \in [1, 4)$, and $[1, 4)$ holds $2^{24}$ floats: an exhaustive check simply tries all of them. The solution's worst case is 0.5004 ulp.

**Subnormals** have $E = 0$, so $I(x)$ is no longer a logarithm and the guess is far off. Multiply by $2^{24}$ (exact, the result is normal), compute, and multiply the result by $2^{12}$ (also exact).

**Specials.** $+0 \mapsto +\infty$ and $-0 \mapsto -\infty$ (as $1/\sqrt{\pm 0}$ in IEEE arithmetic), $+\infty \mapsto +0$, negative numbers and NaN give NaN. Check them before the bit trick, which would turn each of them into a finite number.

## 3. Worked example by hand

$x = 4$, so $r = 0.5$.

| Step | Computation | Value | Relative error |
|---|---|---|---|
| bits | $I(4) = $ `0x40800000`; `>> 1` is `0x20400000` | | |
| guess | `0x5F3759DF - 0x20400000 = 0x3EF759DF`: $E = 125$, $m = $ `0x7759DF` $/2^{23} = 0.93243$, $y_0 = 1.93243 \times 2^{-2}$ | 0.48310754 | 0.0338 |
| step 1 (f32) | $hx = 2$; $t = 2 \cdot y_0 \cdot y_0 = 0.4667856$; $1.5 - t = 1.0332142$; $y_1 = y_0 \cdot 1.0332142$ | 0.49915358 | 1.69e-3 ($\approx \frac32 \cdot 0.0338^2$) |
| step 2 (f32) | $1.5 - 2 y_1^2 = 1.0016913$; $y_2 = y_1 \cdot 1.0016913$ | 0.49999782 | 4.35e-6 ($\approx \frac32 \cdot 0.00169^2$) |
| step 3 (f64) | $1.5 - 2 y_2^2 = 1.0000043$; $y_3 = 0.4999999999858$ | 0.5 after rounding to float32 | 2.8e-11 |

The same steps with $x = 2$ give the float32 nearest $1/\sqrt2$, `0.70710677`, and with $x = 0.15625$ give `2.529822`. These are the hand-worked cases of section 4.

## 4. The interface

```c
/* tinyllm/numerics.h, M09.5 section */
float tl_rsqrtf(float x);                                 /* within 2 ulp; +-0 -> +-inf, +inf -> +0, x < 0 or NaN -> NaN */
void  tl_rsqrt_f32(const float *x, float *y, int64_t n);  /* elementwise; y == x (exact aliasing) allowed */
```

The solution writes `rsqrt_normal` (steps 1 to 3 for a positive normal input) as a `static` helper, then `tl_rsqrtf` as the dispatcher over the special cases and the subnormal rescale. Neither function can fail, so neither touches the error slot.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked inputs | unit, smoke | section 3: 4, 0.25, 1, 2, 0.15625 give the bits above | everyone agrees on the algorithm |
| every input of two binades | property | all $2^{24}$ floats of $[1, 4)$ within 2 ulp | checks worst-case error over two complete binades |
| scaling by four halves the result | property | `rsqrt(4x) == rsqrt(x)/2` bitwise on 2000 seeded normals | why two binades are enough |
| subnormal inputs | boundary | all $2^{23} - 1$ positive subnormals within 2 ulp | RMSNorm of a nearly-zero row |
| special values | boundary | $\pm 0$, $\pm\infty$, negatives, NaN, `FLT_MAX`, `FLT_TRUE_MIN` | a zero row with $\epsilon = 0$ gives $\infty$, visibly |
| the array form matches the scalar and may alias | unit | the array form gives the scalar's bits, writes exactly $n$, works in place, $n = 0$ is a no-op | L9.6 runs it on a whole row in place |

## 5. The solution, step by step

### From first principles

1. **Avoid the division and the square root.** Newton on $f(y) = 1/y^2 - x$ gives $g(y) = y(\frac32 - \frac{x}{2}y^2)$ (section 2.2): multiplies and one subtraction only, with $g'(r) = 0$, so the relative error squares each step.
2. **Quadratic convergence needs a good start.** From 3.44% the errors go $1.8 \times 10^{-3}$, $4.7 \times 10^{-6}$, $3.3 \times 10^{-11}$. The bit pattern is a scaled logarithm (section 2.3), so one integer subtraction gives that start for any normal input.
3. **Count the steps against the contract.** Two steps leave $4.7 \times 10^{-6}$, about 73 ulp. A third step in float32 is limited by its own rounding (2.18 ulp). A third step in float64 has error $\approx 3 \times 10^{-11}$ plus $2^{-53}$ roundings, so rounding the result to float32 once gives a correctly rounded or nearly correctly rounded answer (0.5004 ulp worst case).
4. **Keep the arithmetic order fixed.** Each multiply and subtract is a separate statement with `FP_CONTRACT OFF`, so no compiler fuses them and the C result matches M01.2's Python bit for bit.
5. **Prove accuracy on two binades.** Every step commutes with $x \mapsto 4x$ exactly (section 2.4), so checking all of $[1, 4)$ checks every normal float.
6. **Bring every other input into that case or handle it directly.** Specials are answered before the bit trick; subnormals are scaled by $2^{24}$ into the normal range and the answer by $2^{12}$ back, both exact.

### The code, annotated

```c
#pragma STDC FP_CONTRACT OFF      /* no fused multiply-add: two roundings, as in Python */

#define TL_RSQRT_MAGIC 0x5F3759DFu

static uint32_t f32_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u); /* the defined way to view a float's bits */
    return u;
}

static float f32_from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}
```

`memcpy` is the only bit reinterpretation the C standard defines; a pointer cast (`*(uint32_t *)&x`) violates strict aliasing and UBSan or an optimizer can make it misbehave. Compilers turn this `memcpy` into a single register move.

```c
/* Steps 1 to 3 for a positive normal x. */
static float rsqrt_normal(float x) {
    float hx = 0.5f * x;                                         /* exact: a power of two */
    float y = f32_from_bits(TL_RSQRT_MAGIC - (f32_bits(x) >> 1)); /* y0 within 3.44% */
    for (int i = 0; i < 2; i++) {        /* two float32 Newton steps: 3.44% -> 4.7e-6 */
        float t = hx * y;
        t = t * y;                       /* (0.5 x) y^2, in M01.2's order */
        t = 1.5f - t;
        y = y * t;                       /* y (1.5 - 0.5 x y^2) */
    }
    double xd = (double)x, yd = (double)y;   /* exact widening */
    double hd = 0.5 * xd;
    double t = hd * yd;
    t = t * yd;
    t = 1.5 - t;
    yd = yd * t;                         /* error ~3e-11, far below half a float32 ulp */
    return (float)yd;                    /* the only rounding to float32 of the last step */
}
```

Trace section 3: for $x = 4$ the pattern `0x40800000` shifts to `0x20400000`, the subtraction gives `0x3EF759DF` $= 0.48310754$, the two float32 passes give $0.49915358$ and $0.49999782$, and the float64 pass gives $0.4999999999858$, which rounds to exactly `0.5f`. The shift happens on the *bits* (`f32_bits(x) >> 1`); shifting the value converted to an integer is pitfall 2.

```c
float tl_rsqrtf(float x) {
    if (isnan(x)) return x + x; /* quiet NaN in, NaN out */
    if (x == 0.0f) return signbit(x) ? -INFINITY : INFINITY;   /* 1/sqrt(+-0) */
    if (x < 0.0f) return NAN;
    if (isinf(x)) return 0.0f;                                  /* only +inf is left here */
    if (x < 0x1p-126f) {
        /* Subnormal: 2^24 x is normal and exact; rsqrt(2^24 x) = rsqrt(x) / 2^12. */
        return rsqrt_normal(x * 0x1p24f) * 0x1p12f;
    }
    return rsqrt_normal(x);
}

void tl_rsqrt_f32(const float *x, float *y, int64_t n) {
    /* One read and one write per index, in order, so y == x (exact
     * aliasing) is safe. */
    for (int64_t i = 0; i < n; i++) y[i] = tl_rsqrtf(x[i]);
}
```

The order of the checks matters. NaN goes first because every comparison with NaN is false, so it would otherwise fall through to the bit trick. Zero is tested before the sign, since $-0.0 < 0$ is false and $-0$ must map to $-\infty$, not NaN. After NaN, zero, and negatives are gone, `isinf` can only see $+\infty$. `0x1p-126f` is `FLT_MIN`, the smallest normal. The subnormal rescale uses $2^{24}$ because the smallest subnormal is $2^{-149}$, and $2^{-149} \cdot 2^{24} = 2^{-125}$ is normal; $\sqrt{2^{24}} = 2^{12}$ undoes it. The array form uses `int64_t` for $n$, so $n \le 0$ simply runs no iterations, and reading `x[i]` before writing `y[i]` makes in-place use safe.

Full solution: [`rsqrt.c`](../../../../course/solutions/c/src/numerics/rsqrt.c).

### Why this solution

- **Cost.** One integer subtract, nine float32 operations, five float64 operations, two conversions per call, and no division, square root, table, or branch in the normal path. $O(n)$ for the array form.
- **Over `1.0f / sqrtf(x)`.** The library pair is correctly rounded and simple, but it uses the two slowest float instructions and cannot be derived and bit-matched against the course's Python twin. This version is within 0.5004 ulp, inside the 2 ulp contract, and reproducible on any IEEE platform.
- **Over the Quake III version (two float32 steps).** 73 ulp is too far for a normalization that feeds every layer; the third step costs a handful of float64 operations.
- **Over three float32 steps.** Cheaper on hardware without fast doubles, but its own rounding leaves 2.18 ulp, outside the contract. Finishing in a wider format is what makes the last step's rounding irrelevant.
- **Tradeoffs accepted.** The float64 step makes the code harder to vectorize with float32 SIMD lanes; a production kernel would use a hardware estimate plus one float32 refinement (Going further) and accept a looser bound.

### Explain it back

**1. Why is Newton's iteration for $1/\sqrt{x}$ written on $f(y) = 1/y^2 - x$ rather than on $f(y) = y^2 - 1/x$?**

<details><summary>Answer</summary>

$f(y) = 1/y^2 - x$ gives the update $y(\frac32 - \frac{x}{2}y^2)$, which needs only multiplies and a subtraction. $y^2 - 1/x$ gives $y - (y^2 - 1/x)/(2y)$, which divides twice: the expensive operation the routine exists to avoid.

</details>

**2. Why is the third Newton step done in double instead of float?**

<details><summary>Answer</summary>

After two steps the mathematical error is $4.7 \times 10^{-6}$, and one more step reduces it to about $3 \times 10^{-11}$. In float32, the step's own roundings (each up to $2^{-24}$ relative) dominate and leave 2.18 ulp. In float64 the roundings are $2^{-53}$, so the only significant error is the final conversion to float32, at most half an ulp plus a tiny margin.

</details>

**3. Why does testing every float in $[1, 4)$ cover every normal float32 input?**

<details><summary>Answer</summary>

Multiplying $x$ by 4 adds $2^{24}$ to its bit pattern, so `>> 1` adds $2^{23}$ and the guess's pattern drops by $2^{23}$: exactly half the guess. Every later product is then scaled by a power of two, which is exact in binary floating point, so $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ bit for bit and the error in ulps is the same. Every normal float is $4^j x$ for one $x \in [1, 4)$.

</details>

**4. Why must NaN and $\pm 0$ be checked before anything else in `tl_rsqrtf`?**

<details><summary>Answer</summary>

Comparisons with NaN are false, so a NaN would pass every later test and reach the bit trick, which turns its pattern into a finite guess. $-0.0 < 0.0$ is false, so $-0$ would skip the negative check, and the bit trick on a zero pattern returns a finite number; IEEE says $1/\sqrt{\pm 0} = \pm\infty$.

</details>

**5. Why are subnormals multiplied by $2^{24}$ and the result by $2^{12}$, and why is that exact?**

<details><summary>Answer</summary>

A subnormal has exponent field 0, so its pattern is not a logarithm and the guess is far off. Multiplying by $2^{24}$ lifts even the smallest subnormal $2^{-149}$ to $2^{-125}$, a normal number, without rounding (only the exponent changes). Since $\mathrm{rsqrt}(2^{24}x) = 2^{-12}\,\mathrm{rsqrt}(x)$, multiplying the result by $2^{12}$ recovers the answer, again exactly.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. stopping after the two float32 steps (the Quake III version) | 73 ulp worst case | a third step in float64; step 3 |
| 2. guessing from the value: `(float)(MAGIC - ((uint32_t)x >> 1))` | converts the number 4 to the integer 4; the guess is $10^9$ and Newton diverges | reads the bits with `memcpy` (`f32_bits`) before shifting |
| 3. no rescale for subnormal inputs | thousands of ulps below `FLT_MIN` | `x < 0x1p-126f` scales by $2^{24}$ and back by $2^{12}$; step 6 |
| 4. the third step in float32 | 2.18 ulp worst case, over the contract | the third step widens to `double`; step 3 |
| 5. treating $\pm 0$ or $+\infty$ like any other input | a finite number where the contract says $\pm\infty$ or 0 | NaN, zero, negative, and infinity are answered before the bit trick |
| 6. the array loop off by one | the last element unwritten, or one past the end overwritten | `for (int64_t i = 0; i < n; i++)`, one read and one write per index |
| 7. a different constant shift or step count | caught as accuracy, not as a crash | fixed `TL_RSQRT_MAGIC`, `>> 1`, and exactly two plus one steps |

A fused multiply-add in the float32 steps is worth trying once: it changes $y_2$ in its last bit, yet the float64 step absorbs the difference and the result keeps the same bits, so no accuracy check can see it. That robustness is the point of finishing in a wider format.

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.2` | derives the same Newton iteration in Python |
| Back | `M09.1` | bit patterns, exponent and mantissa fields, ulps, subnormals (reading) |
| Forward | Optional `L9.6` | its standalone C normalization kernel uses `tl_rsqrtf(mean(x^2) + eps)` |
| Forward | `M09.6` | the same pattern, a reduction then a cheap approximation, for $e^x$ |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| bit-pattern guess + Newton | x86 `rsqrtps` / ARM `frsqrte` | a hardware 12-bit table estimate, then one Newton step (`frsqrts` computes $\frac{3 - ab}{2}$ in one instruction) | Arm A64 ISA reference, `FRSQRTE`, `FRSQRTS` |
| `tl_rsqrtf` | glibc and CORE-MATH `rsqrtf` | correctly rounded for every input, proved | the CORE-MATH project (Inria) |
| scalar loop | llama.cpp `ggml_vec_*` RMSNorm, PyTorch fused RMSNorm | vectorized rows, the norm fused with the scale | `ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_rms_norm`) |
| magic constant `0x5F3759DF` | Lomont's `0x5F375A86`, Moroz et al.'s constants with modified Newton coefficients | smaller first-step error, so one Newton step is enough | Moroz, Walczyk, et al., *Fast calculation of inverse square root with the use of magic constant* (2018) |
