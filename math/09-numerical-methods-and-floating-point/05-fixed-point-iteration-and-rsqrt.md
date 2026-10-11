<!-- ol:module M09.5 -->
# Fast inverse square root in standalone C (optional)

## Overview

| | |
|---|---|
| **Module** | `M09.5` · side · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/numerics/rsqrt.c`: `tl_rsqrtf` and `tl_rsqrt_f32` (and the helper `rsqrt_normal`) |
| **Contract** | [`course/contracts/c/include/tinyllm/numerics.h`](../../course/contracts/c/include/tinyllm/numerics.h) (the M09.5 section) · rules: [`c/ABI.md`](../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/M09.5/test_rsqrt.c`, a standalone C suite under sanitizers |
| **Needs** | [`M01.2` Newton's method](../01-calculus-1/02-newtons-method.md) for the iteration; `M09.1` for float32 bit patterns and subnormals |
| **Used by** | Optional `L9.6` includes this module in its standalone C build |
| **Milestone** | `MS-L9`, the optional C module group |
| **Optional depth** | Lomont, *Fast Inverse Square Root* (2003, free); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 1; Süli and Mayers, *An Introduction to Numerical Analysis*, ch. 1 (fixed-point iteration) |

## Key Takeaways

- **Newton's method is a fixed-point iteration** $y \leftarrow g(y)$ whose map has $g'(r) = 0$ at the root, so the relative error squares every step: $e_{n+1} = \tfrac32 e_n^2 - \tfrac12 e_n^3$ (`every_input_of_two_binades`).
- **A float's bit pattern is a piecewise-linear $\log_2$**, so one integer subtraction, `0x5F3759DF - (bits >> 1)`, computes a first guess for $x^{-1/2}$ within 3.5% (`hand_example`).
- **Two float32 Newton steps reach 4.7e-6 relative error, which is still about 73 ulp**; one more step in float64 lands within 0.5004 ulp of the true value, checked on every input (`every_input_of_two_binades`).
- **Two binades decide every normal input**, because $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ holds bit for bit for this algorithm (`scaling_by_four_halves_the_result`); subnormals need a rescale first (`subnormal_inputs`).
- **This remains a standalone C kernel**: its numerical contract is derived from M01.2 and checked in the C test process (`hand_example`, `special_values`).

## How to work this chapter

```bash
ol start M09.5              # stubs rsqrt.c into your repo
ol tests M09.5              # read the test catalog first: rung R0, you write no tests here
ol check M09.5              # exit code is the verdict
ol check M09.5 --ref-deps   # only if you skipped M01.2
ol diff  M09.5              # after passing: your code against the reference
```

The check compiles `rsqrt.c` into a standalone ASan and UBSan test binary. Python and C do not call into each other.

---

## 1. Why now

Every layer of your Llama-family model (L7.1) normalizes its activations with RMSNorm: $y_i = x_i \, w_i / \sqrt{\frac1d \sum_j x_j^2 + \epsilon}$. In Python that is one `np.sqrt` per row. In Part 9 the same normalization moves into C (`L9.6` `tl_rmsnorm_f32`), and the Rust engine calls it once per row, per layer, per token. The C library's `1.0f / sqrtf(s)` would work, but it is two of the slowest float instructions in a hot loop, it is a black box, and the course rule is that every kernel is something you can derive and test against your own Python (P6). You already wrote the derivation: M01.2's `rsqrt_newton` is Newton's method for $1/\sqrt{x}$ without a division. This module turns it into a C function with a contract stated in ulps, and proves the contract on every float32 input that matters.

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

With $\sigma \approx 0.0450466$ the constant is `0x5F3759DF`, and `I(x) >> 1` is $\frac12 I(x)$ rounded down. The resulting $y_0$ is never more than 3.44% from $r$ (`test_two_float_steps_are_not_enough` measures it). In C the only defined way to read a float's bits is `memcpy` into a `uint32_t` (a pointer cast breaks the aliasing rule); C++20 has `std::bit_cast`, Rust `f32::to_bits`, numpy `.view(np.uint32)`.

### 2.4 The algorithm, and why two binades check everything

The contract asks for 2 ulp. Two float32 steps give $4.7 \times 10^{-6}$, which is 40 to 73 ulp; a third float32 step gets to about 2.2 ulp because of its own rounding. So the third step runs in float64, where rounding is $2^{-53}$, and the result is rounded to float32 once:

1. `y = bits(0x5F3759DF - (bits(x) >> 1))`;
2. twice, in float32, `t = hx * y; t = t * y; t = 1.5f - t; y = y * t` with `hx = 0.5f * x`;
3. once, in float64, the same four statements on `(double)x` and `(double)y`; return `(float)y`.

Each statement is separate on purpose: C allows the compiler to fuse `a * b + c` inside one expression into a fused multiply-add with one rounding instead of two (`#pragma STDC FP_CONTRACT OFF` forbids it too), and M01.2 computes in exactly this order.

**Scaling.** For normal $x$, replace $x$ by $4x$: the pattern grows by $2^{24}$, so the guess's pattern shrinks by $2^{23}$, which halves $y_0$ exactly. Every product in steps 2 and 3 then scales by a power of two, which is exact, so $\mathrm{rsqrt}(4x) = \mathrm{rsqrt}(x)/2$ **bit for bit**. Every normal float is $4^j x$ for one $x \in [1, 4)$, and $[1, 4)$ holds $2^{24}$ floats: the test simply tries all of them. The reference's worst case is 0.5004 ulp.

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

The same steps with $x = 2$ give the float32 nearest $1/\sqrt2$, `0.70710677`, and with $x = 0.15625$ give `2.529822`. All three are the first assertions of `hand_example`.

## 4. The interface

```c
/* tinyllm/numerics.h, M09.5 section */
float tl_rsqrtf(float x);                                 /* within 2 ulp; +-0 -> +-inf, +inf -> +0, x < 0 or NaN -> NaN */
void  tl_rsqrt_f32(const float *x, float *y, int64_t n);  /* elementwise; y == x (exact aliasing) allowed */
```

Write `rsqrt_normal` (steps 1 to 3 for a positive normal input) as a `static` helper, then `tl_rsqrtf` as the dispatcher over the special cases and the subnormal rescale. Neither function can fail, so neither touches the error slot.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3: 4, 0.25, 1, 2, 0.15625 give the bits above | you and the test agree on the algorithm |
| `every_input_of_two_binades` | property | all $2^{24}$ floats of $[1, 4)$ within 2 ulp | checks worst-case error over two complete binades |
| `scaling_by_four_halves_the_result` | property | `rsqrt(4x) == rsqrt(x)/2` bitwise on 2000 seeded normals | why two binades are enough |
| `subnormal_inputs` | boundary | all $2^{23} - 1$ positive subnormals within 2 ulp | RMSNorm of a nearly-zero row |
| `special_values` | boundary | $\pm 0$, $\pm\infty$, negatives, NaN, `FLT_MAX`, `FLT_TRUE_MIN` | a zero row with $\epsilon = 0$ gives $\infty$, visibly |
| `array_matches_scalar_and_aliases` | unit | the array form gives the scalar's bits, writes exactly $n$, works in place, $n = 0$ is a no-op | L9.6 runs it on a whole row in place |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. stopping after the two float32 steps (the Quake III version) | 73 ulp worst case | `every_input_of_two_binades` (mutant `s01`) |
| 2. guessing from the value: `(float)(MAGIC - ((uint32_t)x >> 1))` | converts the number 4 to the integer 4; the guess is $10^9$ and Newton diverges | `hand_example` (mutant `s02`) |
| 3. no rescale for subnormal inputs | thousands of ulps below `FLT_MIN` | `subnormal_inputs` (mutants `s03`, `m003`) |
| 4. the third step in float32 | 2.18 ulp worst case, over the contract | `every_input_of_two_binades` (mutant `s04`) |
| 5. treating $\pm 0$ or $+\infty$ like any other input | a finite number where the contract says $\pm\infty$ or 0 | `special_values` (mutants `s05`, `s06`, `s07`) |
| 6. the array loop off by one | the last element unwritten, or one past the end overwritten | `array_matches_scalar_and_aliases` (mutants `s08`, `s09`) |
| 7. a different constant shift or step count | caught as accuracy, not as a crash | `hand_example` (mutants `m001`, `m002`) |

A fused multiply-add in the float32 steps is worth trying once: it changes $y_2$ in its last bit, yet the float64 step absorbs the difference and the result keeps the same bits, so no test can see it. That robustness is the point of finishing in a wider format.

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.2` | derives the same Newton iteration in Python |
| Back | `M09.1` | bit patterns, exponent and mantissa fields, ulps, subnormals (reading) |
| Forward | Optional `L9.6` | its standalone C normalization kernel uses `tl_rsqrtf(mean(x^2) + eps)` |
| Forward | `M09.6` | the same pattern, a reduction then a cheap approximation, for $e^x$ |

If you skip this module, `ol check L9.6` stops with `BLOCKED ... needs M09.5`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| bit-pattern guess + Newton | x86 `rsqrtps` / ARM `frsqrte` | a hardware 12-bit table estimate, then one Newton step (`frsqrts` computes $\frac{3 - ab}{2}$ in one instruction) | Arm A64 ISA reference, `FRSQRTE`, `FRSQRTS` |
| `tl_rsqrtf` | glibc and CORE-MATH `rsqrtf` | correctly rounded for every input, proved | the CORE-MATH project (Inria) |
| scalar loop | llama.cpp `ggml_vec_*` RMSNorm, PyTorch fused RMSNorm | vectorized rows, the norm fused with the scale | `ggml/src/ggml-cpu/ops.cpp` (`ggml_compute_forward_rms_norm`) |
| magic constant `0x5F3759DF` | Lomont's `0x5F375A86`, Moroz et al.'s constants with modified Newton coefficients | smaller first-step error, so one Newton step is enough | Moroz, Walczyk, et al., *Fast calculation of inverse square root with the use of magic constant* (2018) |
