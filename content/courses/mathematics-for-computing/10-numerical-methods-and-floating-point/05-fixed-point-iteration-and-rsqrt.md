# Fixed-point iteration and fast inverse square root in C

Every layer of a Llama-family model normalizes its activations with RMSNorm, introduced in [Pre-LN and RMSNorm](../../build-an-llm/08-modern-block/01-pre-ln-rmsnorm.md) in *Build Your Own LLM System*: $y_i = x_i \, w_i / \sqrt{\frac1d \sum_j x_j^2 + \epsilon}$. In Python that is one `np.sqrt` per row. When the same normalization moves into C as `tl_rmsnorm_f32` in [Elementwise kernels in C](../../build-an-llm/10-kernels/08-elementwise-kernels.md), the Rust engine calls it once per row, per layer, per token. The C library's `1.0f / sqrtf(s)` would work, but it is two of the slowest float instructions in a hot loop, it is a black box, and it breaks the rule that every kernel in these books is something you can derive and test against its Python version.

The derivation already exists. `rsqrt_newton` in [Newton's Method](../02-calculus-1/02-newtons-method.md) is Newton's method for $1/\sqrt{x}$ without a division. This optional section turns it into a C function whose contract is stated in ulps, and proves the contract on every float32 input that matters. Along the way you will see Newton's method as a special case of fixed-point iteration, understand why the famous `0x5F3759DF` constant works, and learn why a third step in double precision is what turns a 73-ulp approximation into a nearly correctly rounded one.

You need the iteration from [Newton's Method](../02-calculus-1/02-newtons-method.md) and the float32 bit patterns and subnormals of [IEEE 754 anatomy, ulps, and round to nearest even](01-ieee-754.md). The solution, [`rsqrt.c`](../../../../course/solutions/c/src/numerics/rsqrt.c), is a standalone C file built under ASan and UBSan; Python and C do not call into each other, and the numerical contract holds without any Python at run time. The interface lives in [`numerics.h`](../../../../course/contracts/c/include/tinyllm/numerics.h), under the rules of [`c/ABI.md`](../../../../course/contracts/c/ABI.md).

> **Learning objectives**
> - Explain the difference between linear and quadratic convergence of a fixed-point iteration in terms of $g'(r)$.
> - Derive the division-free Newton update for $x^{-1/2}$ and its exact error recurrence $e_{n+1} = \tfrac32 e_n^2 - \tfrac12 e_n^3$.
> - Explain why a float's bit pattern is a piecewise-linear $\log_2$, and derive the magic-constant first guess from it.
> - Justify why checking two binades exhaustively proves the accuracy contract for every normal input, and handle subnormals and special values.

## The notation

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

## Fixed-point iteration

A **fixed point** of a map $g$ is a value $r$ with $g(r) = r$. **Fixed-point iteration** starts from a guess $y_0$ and repeats $y_{n+1} = g(y_n)$. If $g$ is differentiable near $r$, a Taylor expansion (see [Taylor series, remainder bounds, and range reduction](../03-calculus-2/01-taylor-series.md)) gives

$$y_{n+1} - r = g(y_n) - g(r) = g'(r)(y_n - r) + \tfrac12 g''(\xi)(y_n - r)^2.$$

So the distance to $r$ shrinks by the factor $|g'(r)|$ each step when $|g'(r)| < 1$ (a **contraction**: linear convergence). When $g'(r) = 0$ the first-order term vanishes and the distance is squared each step (**quadratic convergence**: the number of correct digits doubles).

## Newton's method as a fixed-point iteration with $g'(r) = 0$

Newton's method is the fixed-point iteration you get by choosing $g$ so that $g'(r) = 0$. The Newton chapter applied it to $f(y) = 1/y^2 - x$, whose positive root is $r = x^{-1/2}$. With $f'(y) = -2/y^3$:

$$g(y) = y - \frac{f(y)}{f'(y)} = y + \frac{y^3}{2}\left(\frac{1}{y^2} - x\right) = y\left(\frac32 - \frac{x}{2}\, y^2\right).$$

There is no division and no square root: two multiplies, a subtract, a multiply. Its derivative is $g'(y) = \frac32 - \frac32 x y^2$, which is $0$ at $y = r$ because $x r^2 = 1$. Substituting $y_n = r(1 - e_n)$ and $x r^2 = 1$:

$$y_{n+1} = r(1 - e_n)\left(\tfrac32 - \tfrac12(1 - e_n)^2\right) = r\left(1 - \tfrac32 e_n^2 + \tfrac12 e_n^3\right),$$

so $e_{n+1} = \frac32 e_n^2 - \frac12 e_n^3$ exactly, in real arithmetic. From $e_0 = 0.0344$ you get $e_1 \le 1.8 \times 10^{-3}$, $e_2 \le 4.7 \times 10^{-6}$, and $e_3 \le 3.3 \times 10^{-11}$. In float32 each step also adds a few units of $2^{-24}$ of rounding, which is why the error cannot go below about $10^{-7}$ in float32 no matter how many steps you take.

## A first guess from the bits

Quadratic convergence is only as good as its starting point, and the starting point here comes from a beautiful observation about the float format. A positive normal float is $x = (1 + m)\, 2^{E - 127}$ with $m = M/2^{23} \in [0, 1)$. Its bit pattern is $I(x) = 2^{23}(E + m)$, because the exponent field sits above the 23 mantissa bits. Since $\log_2(1 + m) \approx m + \sigma$ for $m \in [0, 1)$ with a small constant $\sigma$,

$$\log_2 x = E - 127 + \log_2(1 + m) \approx \frac{I(x)}{2^{23}} - 127 + \sigma.$$

The bit pattern is a scaled, shifted logarithm. Taking logs of $r = x^{-1/2}$ gives $\log_2 r = -\frac12 \log_2 x$; replacing both logs by the bit-pattern formula and solving for $I(r)$:

$$I(r) \approx \tfrac32\, 2^{23}(127 - \sigma) - \tfrac12 I(x).$$

With $\sigma \approx 0.0450466$ the constant is `0x5F3759DF`, and `I(x) >> 1` is $\frac12 I(x)$ rounded down. The resulting $y_0$ is never more than 3.44% from $r$. In C the only defined way to read a float's bits is `memcpy` into a `uint32_t` (a pointer cast breaks the aliasing rule); C++20 has `std::bit_cast`, Rust `f32::to_bits`, and numpy `.view(np.uint32)`.

## The algorithm, and why two binades check everything

The contract asks for 2 ulp. Two float32 steps give $4.7 \times 10^{-6}$, which is 40 to 73 ulp, and a third float32 step gets only to about 2.2 ulp because of its own rounding. So the third step runs in float64, where rounding is $2^{-53}$, and the result is rounded to float32 once. The full algorithm is:

1. `y = bits(0x5F3759DF - (bits(x) >> 1))`;
2. twice, in float32, `t = hx * y; t = t * y; t = 1.5f - t; y = y * t` with `hx = 0.5f * x`;
3. once, in float64, the same four statements on `(double)x` and `(double)y`; return `(float)y`.

Each statement is separate on purpose. C allows the compiler to fuse `a * b + c` inside one expression into a fused multiply-add with one rounding instead of two (`#pragma STDC FP_CONTRACT OFF` forbids it too), and the Python version computes in exactly this order.

The accuracy proof rests on scaling. For normal $x$, replace $x$ by $4x$: the pattern grows by $2^{24}$, so the guess's pattern shrinks by $2^{23}$, which halves $y_0$ exactly. Every product in steps 2 and 3 then scales by a power of two, which is exact, so $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ **bit for bit**. Every normal float is $4^j x$ for one $x \in [1, 4)$, and $[1, 4)$ holds $2^{24}$ floats, so an exhaustive check simply tries all of them. The solution's worst case is 0.5004 ulp.

Two kinds of input fall outside that argument. **Subnormals** have $E = 0$, so $I(x)$ is no longer a logarithm and the guess is far off; multiply by $2^{24}$ (exact, and the result is normal), compute, and multiply the result by $2^{12}$ (also exact). **Special values** have fixed answers: $+0 \mapsto +\infty$ and $-0 \mapsto -\infty$ (as $1/\sqrt{\pm 0}$ in IEEE arithmetic), $+\infty \mapsto +0$, and negative numbers and NaN give NaN. Check them before the bit trick, which would turn each of them into a finite number.

## Working it by hand

Take $x = 4$, so $r = 0.5$. The table follows each stage of the algorithm:

| Step | Computation | Value | Relative error |
|---|---|---|---|
| bits | $I(4) = $ `0x40800000`; `>> 1` is `0x20400000` | | |
| guess | `0x5F3759DF - 0x20400000 = 0x3EF759DF`: $E = 125$, $m = $ `0x7759DF` $/2^{23} = 0.93243$, $y_0 = 1.93243 \times 2^{-2}$ | 0.48310754 | 0.0338 |
| step 1 (f32) | $hx = 2$; $t = 2 \cdot y_0 \cdot y_0 = 0.4667856$; $1.5 - t = 1.0332142$; $y_1 = y_0 \cdot 1.0332142$ | 0.49915358 | 1.69e-3 ($\approx \frac32 \cdot 0.0338^2$) |
| step 2 (f32) | $1.5 - 2 y_1^2 = 1.0016913$; $y_2 = y_1 \cdot 1.0016913$ | 0.49999782 | 4.35e-6 ($\approx \frac32 \cdot 0.00169^2$) |
| step 3 (f64) | $1.5 - 2 y_2^2 = 1.0000043$; $y_3 = 0.4999999999858$ | 0.5 after rounding to float32 | 2.8e-11 |

You can watch the error square at each step, exactly as the recurrence predicts. The same steps with $x = 2$ give the float32 nearest $1/\sqrt2$, `0.70710677`, and with $x = 0.15625$ give `2.529822`. These become the hand-worked test cases.

## Designing the API

The public interface is a scalar function and an array form:

```c
/* tinyllm/numerics.h, rsqrt section */
float tl_rsqrtf(float x);                                 /* within 2 ulp; +-0 -> +-inf, +inf -> +0, x < 0 or NaN -> NaN */
void  tl_rsqrt_f32(const float *x, float *y, int64_t n);  /* elementwise; y == x (exact aliasing) allowed */
```

The solution writes `rsqrt_normal` (steps 1 to 3 for a positive normal input) as a `static` helper, then `tl_rsqrtf` as the dispatcher over the special cases and the subnormal rescale. Neither function can fail, so neither touches the error slot.

## Building it step by step

The reasoning chain has six links. First, avoid the division and the square root: Newton on $f(y) = 1/y^2 - x$ gives $g(y) = y(\frac32 - \frac{x}{2}y^2)$, with multiplies and one subtraction only, and with $g'(r) = 0$, so the relative error squares each step. Second, quadratic convergence needs a good start: from 3.44% the errors go $1.8 \times 10^{-3}$, $4.7 \times 10^{-6}$, $3.3 \times 10^{-11}$, and because the bit pattern is a scaled logarithm, one integer subtraction gives that start for any normal input. Third, count the steps against the contract: two steps leave $4.7 \times 10^{-6}$, about 73 ulp; a third step in float32 is limited by its own rounding (2.18 ulp); a third step in float64 has error $\approx 3 \times 10^{-11}$ plus $2^{-53}$ roundings, so rounding the result to float32 once gives a correctly rounded or nearly correctly rounded answer (0.5004 ulp worst case). Fourth, keep the arithmetic order fixed: each multiply and subtract is a separate statement with `FP_CONTRACT OFF`, so no compiler fuses them and the C result matches the Python version bit for bit. Fifth, prove accuracy on two binades: every step commutes with $x \mapsto 4x$ exactly, so checking all of $[1, 4)$ checks every normal float. Sixth, bring every other input into that case or handle it directly: specials are answered before the bit trick, and subnormals are scaled by $2^{24}$ into the normal range and the answer by $2^{12}$ back, both exact.

The file begins with the bit-view helpers:

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

`memcpy` is the only bit reinterpretation the C standard defines; a pointer cast (`*(uint32_t *)&x`) violates strict aliasing, and UBSan or an optimizer can make it misbehave. Compilers turn this `memcpy` into a single register move.

The core is the normal-input path:

```c
/* Steps 1 to 3 for a positive normal x. */
static float rsqrt_normal(float x) {
    float hx = 0.5f * x;                                         /* exact: a power of two */
    float y = f32_from_bits(TL_RSQRT_MAGIC - (f32_bits(x) >> 1)); /* y0 within 3.44% */
    for (int i = 0; i < 2; i++) {        /* two float32 Newton steps: 3.44% -> 4.7e-6 */
        float t = hx * y;
        t = t * y;                       /* (0.5 x) y^2, in the Python version's order */
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

Trace the hand example through it. For $x = 4$ the pattern `0x40800000` shifts to `0x20400000`, the subtraction gives `0x3EF759DF` $= 0.48310754$, the two float32 passes give $0.49915358$ and $0.49999782$, and the float64 pass gives $0.4999999999858$, which rounds to exactly `0.5f`. Note that the shift happens on the *bits* (`f32_bits(x) >> 1`); shifting the value converted to an integer is one of the classic mistakes listed below.

The dispatcher and array form complete the file:

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

The order of the checks matters. NaN goes first because every comparison with NaN is false, so it would otherwise fall through to the bit trick. Zero is tested before the sign, since $-0.0 < 0$ is false and $-0$ must map to $-\infty$, not NaN. After NaN, zero, and negatives are gone, `isinf` can only see $+\infty$. `0x1p-126f` is `FLT_MIN`, the smallest normal. The subnormal rescale uses $2^{24}$ because the smallest subnormal is $2^{-149}$, and $2^{-149} \cdot 2^{24} = 2^{-125}$ is normal; $\sqrt{2^{24}} = 2^{12}$ undoes it. The array form uses `int64_t` for $n$, so $n \le 0$ simply runs no iterations, and reading `x[i]` before writing `y[i]` makes in-place use safe. The full solution is in [`rsqrt.c`](../../../../course/solutions/c/src/numerics/rsqrt.c).

### Why this design

The normal path costs one integer subtract, nine float32 operations, five float64 operations, and two conversions per call, with no division, square root, table, or branch; the array form is $O(n)$.

Compare it with the alternatives. The library pair `1.0f / sqrtf(x)` is correctly rounded and simple, but it uses the two slowest float instructions and cannot be derived and bit-matched against a Python twin; this version is within 0.5004 ulp, inside the 2 ulp contract, and reproducible on any IEEE platform. The Quake III version (two float32 steps) reaches only 73 ulp, too far for a normalization that feeds every layer, and the third step costs only a handful of float64 operations. Three float32 steps would be cheaper on hardware without fast doubles, but their own rounding leaves 2.18 ulp, outside the contract. Finishing in a wider format is what makes the last step's rounding irrelevant.

The tradeoff accepted is that the float64 step makes the code harder to vectorize with float32 SIMD lanes. A production kernel would use a hardware estimate plus one float32 refinement, as described at the end of the section, and accept a looser bound.

## What can go wrong

The tests start from the hand-worked inputs (4, 0.25, 1, 2, 0.15625, with the bits above), so everyone agrees on the algorithm. The central property test runs all $2^{24}$ floats of $[1, 4)$ and requires each to be within 2 ulp, which checks the worst-case error over two complete binades. A second property checks that `rsqrt(4x) == rsqrt(x)/2` bitwise on 2000 seeded normals, which is the reason two binades are enough. All $2^{23} - 1$ positive subnormals must be within 2 ulp, since RMSNorm of a nearly-zero row produces them. Special values ($\pm 0$, $\pm\infty$, negatives, NaN, `FLT_MAX`, `FLT_TRUE_MIN`) must follow the contract, so that a zero row with $\epsilon = 0$ visibly gives $\infty$. Finally, the array form must give the scalar's bits, write exactly $n$ elements, work in place, and treat $n = 0$ as a no-op, because the RMSNorm kernel runs it on a whole row in place.

The mistakes these tests catch:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| stopping after the two float32 steps (the Quake III version) | 73 ulp worst case | a third step in float64 |
| guessing from the value: `(float)(MAGIC - ((uint32_t)x >> 1))` | converts the number 4 to the integer 4; the guess is $10^9$ and Newton diverges | reads the bits with `memcpy` (`f32_bits`) before shifting |
| no rescale for subnormal inputs | thousands of ulps below `FLT_MIN` | `x < 0x1p-126f` scales by $2^{24}$ and back by $2^{12}$ |
| the third step in float32 | 2.18 ulp worst case, over the contract | the third step widens to `double` |
| treating $\pm 0$ or $+\infty$ like any other input | a finite number where the contract says $\pm\infty$ or 0 | NaN, zero, negative, and infinity are answered before the bit trick |
| the array loop off by one | the last element unwritten, or one past the end overwritten | `for (int64_t i = 0; i < n; i++)`, one read and one write per index |
| a different constant shift or step count | caught as accuracy, not as a crash | fixed `TL_RSQRT_MAGIC`, `>> 1`, and exactly two plus one steps |

> **Tip:** A fused multiply-add in the float32 steps is worth trying once. It changes $y_2$ in its last bit, yet the float64 step absorbs the difference and the result keeps the same bits, so no accuracy check can see it. That robustness is the point of finishing in a wider format.

## Where this leads

The optional standalone C normalization kernel of [Elementwise kernels in C](../../build-an-llm/10-kernels/08-elementwise-kernels.md) in *Build Your Own LLM System* uses `tl_rsqrtf(mean(x^2) + eps)` and includes this file in its build. The next section, [Polynomial approximation and range reduction: expf in C](06-polynomial-approximation-and-expf.md), uses the same pattern, a reduction followed by a cheap approximation, for $e^x$.

Production code takes the idea in several directions. x86 `rsqrtps` and ARM `frsqrte` give a hardware 12-bit table estimate, followed by one Newton step (ARM's `frsqrts` computes $\frac{3 - ab}{2}$ in one instruction). glibc and CORE-MATH provide an `rsqrtf` that is correctly rounded for every input, with a proof. llama.cpp's `ggml_vec_*` RMSNorm and PyTorch's fused RMSNorm vectorize rows and fuse the norm with the scale (`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_rms_norm`). And Lomont's `0x5F375A86`, or Moroz et al.'s constants with modified Newton coefficients, give a smaller first-step error, so that one Newton step is enough.

### Further reading

- Lomont, *Fast Inverse Square Root* (2003, free).
- Higham, *Accuracy and Stability of Numerical Algorithms*, chapter 1.
- Süli and Mayers, *An Introduction to Numerical Analysis*, chapter 1 (fixed-point iteration).
- Arm A64 ISA reference, `FRSQRTE` and `FRSQRTS`.
- The CORE-MATH project (Inria).
- Moroz, Walczyk, et al., *Fast calculation of inverse square root with the use of magic constant* (2018).

## Key terms

- **Fixed point**: a value $r$ with $g(r) = r$.
- **Contraction**: a map with $|g'(r)| < 1$ near its fixed point, whose iteration converges linearly.
- **Quadratic convergence**: convergence in which the error is squared each step, so correct digits double; it happens when $g'(r) = 0$.
- **Magic constant**: `0x5F3759DF`, the offset that turns the bit-pattern logarithm into a first guess for $x^{-1/2}$ within 3.44%.
- **Strict aliasing**: the C rule that forbids reading an object through a pointer of an unrelated type; `memcpy` is the defined way to view a float's bits.

## Summary

- Newton's method is a fixed-point iteration $y \leftarrow g(y)$ whose map has $g'(r) = 0$ at the root, so the relative error squares every step: $e_{n+1} = \tfrac32 e_n^2 - \tfrac12 e_n^3$.
- A float's bit pattern is a piecewise-linear $\log_2$, so one integer subtraction, `0x5F3759DF - (bits >> 1)`, computes a first guess for $x^{-1/2}$ within 3.5%.
- Two float32 Newton steps reach $4.7 \times 10^{-6}$ relative error, which is still about 73 ulp; one more step in float64 lands within 0.5004 ulp of the true value, checked on every input.
- Two binades decide every normal input, because $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ holds bit for bit for this algorithm; subnormals need a rescale first.
- The result is a standalone C kernel whose numerical contract is derived from the Newton chapter and holds without any Python at run time.

## Review questions

1. Why is Newton's iteration for $1/\sqrt{x}$ written on $f(y) = 1/y^2 - x$ rather than on $f(y) = y^2 - 1/x$?

   <details><summary>Answer</summary>

   $f(y) = 1/y^2 - x$ gives the update $y(\frac32 - \frac{x}{2}y^2)$, which needs only multiplies and a subtraction. $y^2 - 1/x$ gives $y - (y^2 - 1/x)/(2y)$, which divides twice: the expensive operation the routine exists to avoid.

   </details>

2. Why is the third Newton step done in double instead of float?

   <details><summary>Answer</summary>

   After two steps the mathematical error is $4.7 \times 10^{-6}$, and one more step reduces it to about $3 \times 10^{-11}$. In float32, the step's own roundings (each up to $2^{-24}$ relative) dominate and leave 2.18 ulp. In float64 the roundings are $2^{-53}$, so the only significant error is the final conversion to float32, at most half an ulp plus a tiny margin.

   </details>

3. Why does testing every float in $[1, 4)$ cover every normal float32 input?

   <details><summary>Answer</summary>

   Multiplying $x$ by 4 adds $2^{24}$ to its bit pattern, so `>> 1` adds $2^{23}$ and the guess's pattern drops by $2^{23}$: exactly half the guess. Every later product is then scaled by a power of two, which is exact in binary floating point, so $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ bit for bit and the error in ulps is the same. Every normal float is $4^j x$ for one $x \in [1, 4)$.

   </details>

4. Why must NaN and $\pm 0$ be checked before anything else in `tl_rsqrtf`?

   <details><summary>Answer</summary>

   Comparisons with NaN are false, so a NaN would pass every later test and reach the bit trick, which turns its pattern into a finite guess. $-0.0 < 0.0$ is false, so $-0$ would skip the negative check, and the bit trick on a zero pattern returns a finite number; IEEE says $1/\sqrt{\pm 0} = \pm\infty$.

   </details>

5. Why are subnormals multiplied by $2^{24}$ and the result by $2^{12}$, and why is that exact?

   <details><summary>Answer</summary>

   A subnormal has exponent field 0, so its pattern is not a logarithm and the guess is far off. Multiplying by $2^{24}$ lifts even the smallest subnormal $2^{-149}$ to $2^{-125}$, a normal number, without rounding (only the exponent changes). Since $\mathrm{rsqrt}(2^{24}x) = 2^{-12}\,\mathrm{rsqrt}(x)$, multiplying the result by $2^{12}$ recovers the answer, again exactly.

   </details>
