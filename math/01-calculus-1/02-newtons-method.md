<!-- ol:module M01.2 -->
# Newton's method

## Overview

| | |
|---|---|
| **Module** | `M01.2` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/num/newton.py`: `newton` (roots of a scalar function, with a convergence report) and `rsqrt_newton` (the division-free iteration for $1/\sqrt{x}$) |
| **Contract** | [`course/contracts/py/tinyllm/num/newton.pyi`](../../course/contracts/py/tinyllm/num/newton.pyi) |
| **Tests** | `course/tests/M01.2/test_newton.py` (what they check: section 4) |
| **Needs** | `M01.1` `central_diff`, the slope Newton uses when you pass no derivative (or `--ref-deps`) |
| **Used by** | `M07.7`'s `fit_temperature` (temperature scaling) solves for $1/T$ with `newton` · optional `M09.5` ports `rsqrt_newton` to C as the reciprocal square root inside `L9.6`'s RMSNorm kernel and checks it against your Python · optional `M10.6` uses Newton-Schulz iterations for Muon (section 6) |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), section 4.9 (Newton's method); Sauer, *Numerical Analysis*, sections 1.4 and 1.5 (convergence order, when Newton fails); Lomont, "Fast inverse square root" (2003) for the magic constant |

## Key Takeaways

- **Newton's method** replaces $f$ by its tangent line at $x_n$ and jumps to the tangent's zero: $x_{n+1} = x_n - f(x_n)/f'(x_n)$ (`test_hand_example`).
- Near a simple root it converges **quadratically**: $e_{n+1} \approx C e_n^2$ with $C = |f''/(2f')|$ at the root, so the number of correct digits doubles every step (`test_digits_double_per_step`).
- It is only **locally** convergent: it can cycle, diverge, or hit a flat tangent. A correct implementation reports each of these instead of returning a wrong number (`test_cycle_raises_after_max_iter`, `test_divergence_raises`, `test_zero_derivative_raises`).
- The stopping test must be **relative**, $|x_{n+1} - x_n| \le \mathrm{tol} \cdot \max(1, |x_{n+1}|)$, because floats near a large root are far apart (`test_relative_tolerance_for_large_roots`).
- Applied to $g(y) = 1/y^2 - x$, Newton computes $1/\sqrt{x}$ with only multiplications, $y \leftarrow y(\frac{3}{2} - \frac{1}{2} x y^2)$; two steps from a 3.5 percent guess reach float32 accuracy (`test_rsqrt_error_squares_each_step`).

## How to work this chapter

```bash
ol start M01.2              # stubs python/tinyllm/num/newton.py into your repo
ol tests M01.2              # read the test catalog first: rung R0, you write no tests here
ol check M01.2              # exit code is the verdict
ol check M01.2 --ref-deps   # only if your M01.1 is not passing yet
ol diff  M01.2              # after passing: your code against the reference
```

---

## 1. Why now

Every transformer you build normalizes its activations, and from `L7.1` on the normalization is RMSNorm: divide a vector by $\sqrt{\text{mean of squares}}$. In Pass 6 your C kernel (`L9.6`) computes it for every token of every layer, and a division plus a square root per element is the slow way. Hardware and libraries compute $1/\sqrt{x}$ directly, by a cheap guess refined with Newton's method, and `M09.5` does the same in C. This module teaches the method on paper and in Python, where you can see the digits double, so that the C port has a trusted reference: your own Python, step for step. The general `newton` you write here is also the first algorithm in the course that can fail in ways a test must catch: a flat tangent, a cycle, a runaway. You build it to say so.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f$ | a differentiable function of one real variable | `Callable[[float], float]` |
| $f'$ | its derivative (`df`), or `central_diff` from `M01.1` when `df` is `None` | `Callable[[float], float]` |
| $x^\star$ | a **root**: $f(x^\star) = 0$ | `float` |
| $x_n$ | the $n$-th iterate, $x_0$ the starting guess | `float` |
| $e_n = x_n - x^\star$ | the error of the $n$-th iterate | `float` |
| tol | the relative step tolerance | `float`, default $10^{-12}$ |
| $x$ (in 2.4) | a positive number whose $1/\sqrt{x}$ we want | float32 or float64 array |
| $y_k$ | the $k$-th estimate of $1/\sqrt{x}$ | same dtype as $x$ |

### 2.1 Roots

A **root** of $f$ is a number $x^\star$ with $f(x^\star) = 0$. Many quantities are roots in disguise: $\sqrt{2}$ is the positive root of $x^2 - 2$, $\ln 3$ is the root of $e^x - 3$, and $1/\sqrt{a}$ is a root of $1/y^2 - a$. A root is **simple** when $f'(x^\star) \neq 0$: the graph crosses zero at an angle instead of touching it.

### 2.2 The tangent-line step

Near $x_n$ the tangent line approximates $f$ (that is what the derivative means, `M01.1`):

$$f(x) \approx f(x_n) + f'(x_n)(x - x_n) .$$

The tangent line is zero at $x = x_n - f(x_n)/f'(x_n)$, and Newton's method takes that point as the next iterate:

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} .$$

For $f(x) = x^2 - 2$ the step is $x_{n+1} = x_n - \frac{x_n^2 - 2}{2x_n} = \frac{1}{2}\left(x_n + \frac{2}{x_n}\right)$: average your guess with 2 divided by it, the method the Babylonians used for square roots. If $f'(x_n) = 0$ the tangent is flat and never crosses zero, so there is no next iterate.

### 2.3 Quadratic convergence

How fast does the error shrink? Expand $f$ around $x_n$ and evaluate at the root (Taylor's formula with the exact remainder, `M02.1`): for some $\xi$ between $x_n$ and $x^\star$,

$$0 = f(x^\star) = f(x_n) + f'(x_n)(x^\star - x_n) + \tfrac{1}{2} f''(\xi)(x^\star - x_n)^2 .$$

Divide by $f'(x_n)$ and use the definition of $x_{n+1}$: $x_{n+1} - x^\star = \frac{f''(\xi)}{2 f'(x_n)} (x_n - x^\star)^2$, that is

$$e_{n+1} = \frac{f''(\xi)}{2 f'(x_n)}\, e_n^2 \approx C\, e_n^2, \qquad C = \left|\frac{f''(x^\star)}{2 f'(x^\star)}\right| .$$

The new error is proportional to the **square** of the old one. If $|e_n| = 10^{-3}$ and $C \approx 1$, then $|e_{n+1}| \approx 10^{-6}$ and $|e_{n+2}| \approx 10^{-12}$: the number of correct digits doubles each step. For $x^2 - 2$, $C = \frac{2}{2 \cdot 2\sqrt{2}} = \frac{1}{2\sqrt 2} \approx 0.354$, and the ratio $e_{n+1}/e_n^2$ approaches exactly that (`test_digits_double_per_step`).

The proof needs $f'(x_n) \neq 0$ and $x_n$ already close to the root. Far from it nothing is promised. $f(x) = x^3 - 2x + 2$ started at 0 jumps to 1 and back to 0 forever. $f(x) = \sqrt[3]{x}$ has its tangent at $x$ cross zero at $-2x$, so every step doubles the distance. A method that is only **locally convergent** must be given an iteration budget, `max_iter`, and must report when it runs out.

**The derivative must move with the iterate.** Computing $f'(x_0)$ once and reusing it (the "chord method") still converges, but only linearly: the error shrinks by a constant factor $|1 - f'(x^\star)/f'(x_0)|$ per step, and if that factor exceeds 1 it diverges. When you pass no derivative, `newton` calls `central_diff` at each $x_n$; its relative error near $10^{-10}$ keeps the convergence effectively quadratic.

**Stopping.** Stop when the step is negligible relative to the iterate, $|x_{n+1} - x_n| \le \mathrm{tol} \cdot \max(1, |x_{n+1}|)$. An absolute test fails for large roots: floats near 12649 are $1.8 \times 10^{-12}$ apart, so near $\sqrt{1.6 \times 10^8}$ the last steps hop between two neighbouring floats and an absolute $10^{-12}$ is never met. The $\max(1, \cdot)$ keeps the test absolute near zero, where a relative one would demand impossible precision. And if $f(x_n)$ is exactly 0, $x_n$ is a root and no step is needed.

### 2.4 Newton for $1/\sqrt{x}$

Apply the step to $g(y) = \frac{1}{y^2} - x$, whose positive root is $y = 1/\sqrt{x}$. With $g'(y) = -2/y^3$:

$$y_{k+1} = y_k - \frac{1/y_k^2 - x}{-2/y_k^3} = y_k + \frac{y_k - x y_k^3}{2} = y_k \left(\frac{3}{2} - \frac{1}{2}\, x\, y_k^2\right) .$$

No division and no square root: three multiplications and a subtraction. (The more obvious $g(y) = y^2 - 1/x$ needs $1/x$ first, a division.) Write $y_k = (1 + \delta_k)/\sqrt{x}$, a relative error $\delta_k$. Substituting,

$$\delta_{k+1} = -\tfrac{3}{2}\delta_k^2 - \tfrac{1}{2}\delta_k^3 ,$$

so the relative error squares (times $\frac{3}{2}$) each step. A good first guess comes from the bits of a float32: the integer $\mathtt{0x5f3759df} - (\text{bits}(x) \gg 1)$, read back as a float, is within 3.5 percent of $1/\sqrt{x}$ for every positive normal $x$, because halving the bits roughly halves the exponent. Then $\delta_1 \le \frac{3}{2}(0.035)^2 \approx 1.8 \times 10^{-3}$ and $\delta_2 \approx 5 \times 10^{-6}$: float32 accuracy (its spacing is $6 \times 10^{-8}$ relative) after two steps. `M09.5` uses exactly this, so `rsqrt_newton` computes in the dtype it is given: float32 in, float32 arithmetic, float32 out.

## 3. Worked example by hand

**$\sqrt{2}$ from $x_0 = 1$**, with $f(x) = x^2 - 2$ and the step $x_{n+1} = \frac{1}{2}(x_n + 2/x_n)$:

| $n$ | $x_n$ | as a fraction | $e_n = x_n - \sqrt 2$ | $e_n / e_{n-1}^2$ |
|---|---|---|---|---|
| 0 | 1 | $1$ | $-4.1 \times 10^{-1}$ | |
| 1 | 1.5 | $\frac{1}{2}(1 + 2) = \frac{3}{2}$ | $8.6 \times 10^{-2}$ | 0.50 |
| 2 | 1.416666... | $\frac{1}{2}(\frac{3}{2} + \frac{4}{3}) = \frac{17}{12}$ | $2.5 \times 10^{-3}$ | 0.33 |
| 3 | 1.4142157 | $\frac{577}{408}$ | $2.1 \times 10^{-6}$ | 0.353 |
| 4 | 1.41421356237469 | $\frac{665857}{470832}$ | $1.6 \times 10^{-12}$ | 0.354 |
| 5 | 1.4142135623730951 | | below one float spacing | |

The correct digits go 0, 1, 2, 5, 11, 16. Update 5 moves by $1.6 \times 10^{-12}$, more than $\mathrm{tol} \cdot |x| = 1.4 \times 10^{-12}$, so the method takes update 6, which moves by one unit in the last place, and stops: `newton` returns $(x_6, 6)$. `test_hand_example` checks the fractions, the count, and the five points where $f$ was evaluated.

**$1/\sqrt{4} = 0.5$ from $y_0 = 0.4$** ($\delta_0 = -0.2$):

$$y_1 = 0.4 \left(1.5 - 0.5 \cdot 4 \cdot 0.16\right) = 0.4 \cdot 1.18 = 0.472, \qquad \delta_1 = -0.056 ,$$

$$y_2 = 0.472 \left(1.5 - 2 \cdot 0.222784\right) = 0.472 \cdot 1.054432 = 0.497691904, \qquad \delta_2 = -0.0046 .$$

Check with the error formula: $-\frac{3}{2}(0.2)^2 - \frac{1}{2}(-0.2)^3 = -0.06 + 0.004 = -0.056$. These are `test_rsqrt_hand_example`.

## 4. The interface

```python
def newton(f, df, x0: float, tol: float = 1e-12, max_iter: int = 50) -> tuple[float, int]: ...
def rsqrt_newton(x: ArrayLike, y0: ArrayLike, iters: int) -> NDArray: ...
```

`newton` returns `(root, updates)`. It raises `ValueError` for a non-finite `x0`, `tol <= 0`, or `max_iter < 1`, and `RuntimeError` naming the cause when the derivative is zero or not finite ("derivative"), an iterate is not finite ("diverged"), or `max_iter` updates do not converge ("no convergence"). `rsqrt_newton` broadcasts `x` against `y0` and computes in their floating dtype (float64 for integers).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3: the fractions, 6 updates, 5 evaluations | you and the tests agree on the iteration and the count |
| `test_digits_double_per_step` | property | $e_{n+1}/e_n^2 \to 0.354$ | quadratic convergence, the reason Newton is used at all |
| `test_returns_on_exact_root` | boundary | $f(x_0) = 0$ returns $(x_0, 0)$ | no needless step or miscount |
| `test_relative_tolerance_for_large_roots` | boundary | $\sqrt{1.6 \times 10^8}$ and $\sqrt[3]{3 \times 10^{20}}$ converge | roots of every scale |
| `test_converges_on_many_roots` | golden | 30 seeded square and cube roots to full precision | the closed forms are the oracle |
| `test_numeric_derivative_when_df_is_none` | unit | $e^x = 3$ with `df=None` gives $\ln 3$ | your `M01.1` slope inside Newton |
| `test_zero_derivative_raises` | boundary | a flat or nan tangent raises `RuntimeError` ("derivative") | failures are reported, not divided by |
| `test_cycle_raises_after_max_iter` | boundary | the 0, 1, 0, 1 cycle raises `RuntimeError` | local convergence has a budget |
| `test_max_iter_counts_updates` | boundary | 6 updates fit `max_iter = 6`, not 5 | the budget is exact |
| `test_divergence_raises` | boundary | $\sqrt[3]{x}$ doubles away until inf; `RuntimeError` ("diverged") | runaway iterates are caught |
| `test_rejects_bad_arguments` | boundary | nan or inf start, `tol <= 0`, `max_iter = 0` | caller bugs surface at once |
| `test_rsqrt_hand_example` | unit, smoke | 0.472, then 0.4977 | the update formula |
| `test_rsqrt_error_squares_each_step` | golden | from the bit-trick guess, float32 errors 3.5e-2, 1.8e-3, 5e-6 | `M09.5`'s accuracy target |
| `test_rsqrt_float64_reaches_full_precision` | golden | four float64 steps reach rounding level | the error keeps squaring |
| `test_rsqrt_keeps_dtype_and_broadcasts` | unit | float32 step for step, bit for bit; ints in float64; `iters = 0`; `iters < 0` raises | the C port is compared bit for bit |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. not checking for a zero or non-finite derivative | `ZeroDivisionError`, or a nan that later looks like "no convergence" | `test_zero_derivative_raises` (mutant `s07`) |
| 2. returning the last iterate when the method did not converge | the cycle's last point (0.0) reported as a root of $x^3 - 2x + 2$ | `test_cycle_raises_after_max_iter` (mutant `s08`) |
| 2b. letting an inf iterate continue | the failure is blamed on a flat tangent one step later | `test_divergence_raises` (mutant `s09`) |
| 3. an absolute stopping test | no convergence at the right answer near $1.3 \times 10^4$ | `test_relative_tolerance_for_large_roots` (mutant `s05`) |
| 4. computing the derivative once at $x_0$ (chord method) | linear convergence, or divergence when $f'$ changes a lot | `test_digits_double_per_step` (mutant `s03`), `test_numeric_derivative_when_df_is_none` (mutant `s06`) |
| 5. dropping a factor of $y$ in $y(\frac{3}{2} - \frac{1}{2} x y^2)$ | converges to the wrong value, $1/x$-like | `test_rsqrt_hand_example` (mutant `s10`) |
| 6. computing float32 inputs in float64 | results that differ from the C port in the last bits, and a float64 array where float32 was expected | `test_rsqrt_keeps_dtype_and_broadcasts` (mutant `s12`) |
| stepping uphill, $x + f/f'$, or multiplying by $f'$ | the hand example's iterates go wrong at step 1 | `test_hand_example` (mutants `s01`, `s02`) |
| always stepping once, even at an exact root | $(3.0, 1)$ instead of $(3.0, 0)$ | `test_returns_on_exact_root` (mutant `s04`) |
| one Newton step fewer than asked | the error after two steps is $1.8 \times 10^{-3}$, not $5 \times 10^{-6}$ | `test_rsqrt_error_squares_each_step` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.1` | `newton(f, None, x0)` calls `central_diff` at every iterate |
| Forward | `M09.5` | the same $y(\frac{3}{2} - \frac{1}{2}xy^2)$ in C, float32, from the bit-trick guess; its differential test compares against your `rsqrt_newton` (Pass 6) |
| Forward | `L9.6` | `tl_rmsnorm_f32` multiplies by that reciprocal square root for every token |
| Forward | `M10.6` | optional: Muon orthogonalizes momentum with Newton-Schulz, Newton's method for a matrix function |
| Forward | `M09.3` | convergence order and error propagation, generalized |
| Forward | `M07.7` | `fit_temperature` calls `newton(g, dg, 0.0)` on the derivative of the NLL in $1/T$, with the variance of the logits as `dg` |

`M07.7` is the registered call site; `M09.5` is an optional C side quest and `M10.6` is not authored yet.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rsqrt_newton` with the bit trick | x86 `rsqrtps` / ARM `frsqrte` plus one Newton step; CUDA `rsqrtf` | a hardware table lookup for the first 12 bits, then the same refinement | Intel intrinsics guide `_mm_rsqrt_ps`; ggml's `ggml_vec_*` RMSNorm paths |
| `newton` | SciPy `scipy.optimize.newton`, `brentq` | secant and Halley variants; Brent's method brackets the root so it can never diverge | `scipy/optimize/_zeros_py.py` |
| a Newton step that can fail | safeguarded Newton (`rtsafe`) | falls back to bisection whenever the Newton step leaves the bracket | Press et al., *Numerical Recipes*, section 9.4 |
| scalar Newton | Newton-Schulz in Muon | Newton's method for the matrix sign function, $X \leftarrow \frac{3}{2}X - \frac{1}{2}XX^\top X$, the same polynomial as section 2.4 | Jordan et al., Muon (2024); `M10.6` |
