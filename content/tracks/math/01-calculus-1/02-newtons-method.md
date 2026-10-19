# Newton's method

## Overview

| | |
|---|---|
| **Module** | `M01.2` · build · Python · Pass 2 · 1 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/newton.py`](../../../../course/solutions/python/tinyllm/num/newton.py): `newton` (roots of a scalar function, with a convergence report) and `rsqrt_newton` (the division-free iteration for $1/\sqrt{x}$) |
| **Contract** | [`course/contracts/py/tinyllm/num/newton.pyi`](../../../../course/contracts/py/tinyllm/num/newton.pyi) |
| **Needs** | `M01.1` `central_diff`, the slope Newton uses when no derivative is passed |
| **Used by** | `M07.7`'s `fit_temperature` (temperature scaling) solves for $1/T$ with `newton` · optional `M09.5` ports `rsqrt_newton` to C as the reciprocal square root inside `L9.6`'s RMSNorm kernel and checks it against this Python · optional `M10.6` uses Newton-Schulz iterations for Muon (section 7) |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), section 4.9 (Newton's method); Sauer, *Numerical Analysis*, sections 1.4 and 1.5 (convergence order, when Newton fails); Lomont, "Fast inverse square root" (2003) for the magic constant |

## Key Takeaways

- **Newton's method** replaces $f$ by its tangent line at $x_n$ and jumps to the tangent's zero: $x_{n+1} = x_n - f(x_n)/f'(x_n)$.
- Near a simple root it converges **quadratically**: $e_{n+1} \approx C e_n^2$ with $C = |f''/(2f')|$ at the root, so the number of correct digits doubles every step.
- It is only **locally** convergent: it can cycle, diverge, or hit a flat tangent. A correct implementation reports each of these instead of returning a wrong number.
- The stopping test must be **relative**, $|x_{n+1} - x_n| \le \mathrm{tol} \cdot \max(1, |x_{n+1}|)$, because floats near a large root are far apart.
- Applied to $g(y) = 1/y^2 - x$, Newton computes $1/\sqrt{x}$ with only multiplications, $y \leftarrow y(\frac{3}{2} - \frac{1}{2} x y^2)$; two steps from a 3.5 percent guess reach float32 accuracy.

## How to read this chapter

Sections 1 and 2 derive Newton's step, its quadratic convergence, its failure modes, and the division-free iteration for $1/\sqrt{x}$ from first principles. Section 3 works both by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Every transformer in the course normalizes its activations, and from `L7.1` on the normalization is RMSNorm: divide a vector by $\sqrt{\text{mean of squares}}$. In Pass 6 the C kernel (`L9.6`) computes it for every token of every layer, and a division plus a square root per element is the slow way. Hardware and libraries compute $1/\sqrt{x}$ directly, by a cheap guess refined with Newton's method, and `M09.5` does the same in C. This module teaches the method on paper and in Python, where you can see the digits double, so that the C port has a trusted reference: this Python, step for step. The general `newton` here is also the first algorithm in the course that can fail in ways a caller must hear about: a flat tangent, a cycle, a runaway. The solution is built to say so.

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

The new error is proportional to the **square** of the old one. If $|e_n| = 10^{-3}$ and $C \approx 1$, then $|e_{n+1}| \approx 10^{-6}$ and $|e_{n+2}| \approx 10^{-12}$: the number of correct digits doubles each step. For $x^2 - 2$, $C = \frac{2}{2 \cdot 2\sqrt{2}} = \frac{1}{2\sqrt 2} \approx 0.354$, and the ratio $e_{n+1}/e_n^2$ approaches exactly that.

The proof needs $f'(x_n) \neq 0$ and $x_n$ already close to the root. Far from it nothing is promised. $f(x) = x^3 - 2x + 2$ started at 0 jumps to 1 and back to 0 forever. $f(x) = \sqrt[3]{x}$ has its tangent at $x$ cross zero at $-2x$, so every step doubles the distance. A method that is only **locally convergent** must be given an iteration budget, `max_iter`, and must report when it runs out.

**The derivative must move with the iterate.** Computing $f'(x_0)$ once and reusing it (the "chord method") still converges, but only linearly: the error shrinks by a constant factor $|1 - f'(x^\star)/f'(x_0)|$ per step, and if that factor exceeds 1 it diverges. When no derivative is passed, `newton` calls `central_diff` at each $x_n$; its relative error near $10^{-10}$ keeps the convergence effectively quadratic.

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

The correct digits go 0, 1, 2, 5, 11, 16. Update 5 moves by $1.6 \times 10^{-12}$, more than $\mathrm{tol} \cdot |x| = 1.4 \times 10^{-12}$, so the method takes update 6, which moves by one unit in the last place, and stops: `newton` returns $(x_6, 6)$, having evaluated $f$ at the six points $x_0, \ldots, x_5$.

**$1/\sqrt{4} = 0.5$ from $y_0 = 0.4$** ($\delta_0 = -0.2$):

$$y_1 = 0.4 \left(1.5 - 0.5 \cdot 4 \cdot 0.16\right) = 0.4 \cdot 1.18 = 0.472, \qquad \delta_1 = -0.056 ,$$

$$y_2 = 0.472 \left(1.5 - 2 \cdot 0.222784\right) = 0.472 \cdot 1.054432 = 0.497691904, \qquad \delta_2 = -0.0046 .$$

Check with the error formula: $-\frac{3}{2}(0.2)^2 - \frac{1}{2}(-0.2)^3 = -0.06 + 0.004 = -0.056$.

## 4. The interface

```python
def newton(f, df, x0: float, tol: float = 1e-12, max_iter: int = 50) -> tuple[float, int]: ...
def rsqrt_newton(x: ArrayLike, y0: ArrayLike, iters: int) -> NDArray: ...
```

`newton` returns `(root, updates)`. It raises `ValueError` for a non-finite `x0`, `tol <= 0`, or `max_iter < 1`, and `RuntimeError` naming the cause when the derivative is zero or not finite ("derivative"), an iterate is not finite ("diverged"), or `max_iter` updates do not converge ("no convergence"). `rsqrt_newton` broadcasts `x` against `y0` and computes in their floating dtype (float64 for integers).

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked $\sqrt 2$ | unit | section 3: the fractions, 6 updates, $f$ evaluated at $x_0, \ldots, x_5$ | everyone agrees on the iteration and the count |
| digits double per step | property | $e_{n+1}/e_n^2 \to 0.354$ | quadratic convergence, the reason Newton is used at all |
| an exact root at the start | boundary | $f(x_0) = 0$ returns $(x_0, 0)$ | no needless step or miscount |
| large roots | boundary | $\sqrt{1.6 \times 10^8}$ and $\sqrt[3]{3 \times 10^{20}}$ converge | roots of every scale |
| many roots | golden | 30 seeded square and cube roots to full precision | the closed forms are the oracle |
| no derivative given | unit | $e^x = 3$ with `df=None` gives $\ln 3$ | the `M01.1` slope inside Newton |
| a flat tangent | boundary | a flat or nan tangent raises `RuntimeError` ("derivative") | failures are reported, not divided by |
| a cycle | boundary | the 0, 1, 0, 1 cycle raises `RuntimeError` | local convergence has a budget |
| the budget counts updates | boundary | 6 updates fit `max_iter = 6`, not 5 | the budget is exact |
| divergence | boundary | $\sqrt[3]{x}$ doubles away until inf; `RuntimeError` ("diverged") | runaway iterates are caught |
| bad arguments | boundary | nan or inf start, `tol <= 0`, `max_iter = 0` | caller bugs surface at once |
| the hand-worked $1/\sqrt 4$ | unit | 0.472, then 0.4977 | the update formula |
| rsqrt error squares each step | golden | from the bit-trick guess, float32 errors 3.5e-2, 1.8e-3, 5e-6 | `M09.5`'s accuracy target |
| rsqrt in float64 | golden | four float64 steps reach rounding level | the error keeps squaring |
| rsqrt dtype and broadcasting | unit | float32 step for step, bit for bit; ints in float64; `iters = 0`; `iters < 0` raises | the C port is compared bit for bit |

## 5. The solution, step by step

### From first principles

1. **One loop iteration is one tangent step.** Section 2.2: evaluate $f(x_n)$ and $f'(x_n)$, then $x_{n+1} = x_n - f(x_n)/f'(x_n)$. Count *updates*, the number of times $x$ moved, because that is what the convergence analysis is about and what `max_iter` budgets.
2. **Check for an exact root before computing a slope.** If $f(x_n) = 0$ there is nothing to do; return $x_n$ with the updates so far (0 if $x_0$ was already a root).
3. **Refuse to divide by a useless slope.** A zero, NaN, or infinite derivative has no tangent root; dividing would produce inf or NaN that surfaces later as a misleading failure. Raise with the word "derivative" now.
4. **Re-evaluate the derivative at every iterate.** Section 2.3: a frozen derivative (the chord method) converges only linearly. When `df` is `None`, the slope is `central_diff(f, x_n)` at the current point, accurate to about $10^{-10}$, which keeps convergence effectively quadratic.
5. **Catch runaways immediately.** If $x_{n+1}$ is not finite, the method diverged; report it at that step, not one step later as a flat tangent at infinity.
6. **Stop on a relative step.** Section 2.3: stop when $|x_{n+1} - x_n| \le \text{tol} \cdot \max(1, |x_{n+1}|)$. Relative for large roots (float spacing grows with $|x|$), absolute near 0 (a relative test would demand impossible precision there).
7. **Running out of budget is a failure, not an answer.** If `max_iter` updates pass without meeting the test, raise; the last iterate of a cycle is not a root.
8. **The rsqrt iteration is the same idea, specialised.** Section 2.4: $y \leftarrow y(\tfrac32 - \tfrac12 x y^2)$, run a fixed number of times on whole arrays. It must compute in the inputs' float dtype so the float32 version matches the C port operation for operation. The factor $\tfrac12 x$ does not change between steps, so it is computed once.

### The code, annotated

```python
def newton(
    f: Callable[[float], float],
    df: Optional[Callable[[float], float]],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> tuple[float, int]:
    x = float(x0)
    if not math.isfinite(x):
        raise ValueError(f"x0 must be finite, got {x0}")
    if not tol > 0:                       # also rejects a nan tol
        raise ValueError(f"tol must be > 0, got {tol}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    # No derivative given: a fresh central difference at each iterate (step 4).
    deriv = df if df is not None else (lambda t: central_diff(f, t))
    for n in range(max_iter):             # n = updates made so far
        fx = float(f(x))
        if fx == 0.0:
            return x, n                   # x is a root: no step needed (step 2)
        d = float(deriv(x))
        if d == 0.0 or not math.isfinite(d):
            raise RuntimeError(
                f"derivative is {d} at x = {x!r} (step {n}): the tangent has no root"
            )
        x_new = x - fx / d                # the tangent's zero
        if not math.isfinite(x_new):
            raise RuntimeError(f"iterate {n + 1} is {x_new}: Newton diverged")
        # Relative step test: near a root of size 1e10 an absolute 1e-12 is
        # below the float spacing and could never be met.
        if abs(x_new - x) <= tol * max(1.0, abs(x_new)):
            return x_new, n + 1           # this was update n + 1
        x = x_new
    raise RuntimeError(f"no convergence after {max_iter} updates (last iterate {x!r})")
```

The count works out exactly: in the iteration with index `n`, `n` updates have already happened, and computing `x_new` is update `n + 1`. On the $\sqrt 2$ example the loop runs with `n = 0, ..., 5`, evaluates $f$ at $x_0, \ldots, x_5$, and returns at `n = 5` with 6 updates, which fits `max_iter = 6` exactly. On the cycle $x^3 - 2x + 2$ from 0 every update is a full step of size 1, the test is never met, and after 50 updates the function raises "no convergence" instead of returning 0. On $\sqrt[3]{x}$ from 1 each step doubles $|x|$, so with a large enough budget iterate 1024 overflows to inf and the "diverged" branch fires.

```python
def rsqrt_newton(x: ArrayLike, y0: ArrayLike, iters: int) -> NDArray:
    if iters < 0:
        raise ValueError(f"iters must be >= 0, got {iters}")
    xa, ya = np.asarray(x), np.asarray(y0)
    dt = np.result_type(xa, ya)                  # float32 + float32 stays float32
    if not np.issubdtype(dt, np.floating):
        dt = np.dtype(np.float64)                # integers compute in float64
    xa, y = np.broadcast_arrays(xa.astype(dt), ya.astype(dt))
    y = y.copy()                                 # broadcast views are read-only and may alias y0
    half, three_halves = dt.type(0.5), dt.type(1.5)   # constants in the working dtype
    hx = half * xa                               # 0.5 x, hoisted out of the loop
    for _ in range(int(iters)):
        y = y * (three_halves - hx * y * y)      # y (3/2 - x y^2 / 2): no division, no sqrt
    return y
```

Every operand in the loop has dtype `dt`, so a float32 call does float32 arithmetic throughout and returns float32, which is what lets `M09.5` compare its C output bit for bit; the C port must then group the products the same way, `((hx * y) * y)`. `iters = 0` returns a fresh copy of the broadcast guess.

The function takes the first guess `y0` from the caller. The bit-trick guess of section 2.4 is one line of numpy (shown here for reference, not part of the module):

```python
x = np.array([4.0, 2.0, 1e-3, 7e5], dtype=np.float32)
y0 = (np.uint32(0x5f3759df) - (x.view(np.uint32) >> np.uint32(1))).view(np.float32)
# relative error of rsqrt_newton(x, y0, k) for k = 0, 1, 2: about 3.4e-2, 1.7e-3, 4.4e-6
```

Full solution: [`newton.py`](../../../../course/solutions/python/tinyllm/num/newton.py).

### Why this solution

- **Cost.** Each update costs one evaluation of $f$ and one of $f'$ (or two more of $f$ for the central difference). Quadratic convergence means the number of updates grows only like $\log_2$ of the digits wanted: the $\sqrt 2$ example goes from 0 to 16 correct digits in 6 updates. `rsqrt_newton` is 3 multiplications and 1 subtraction per element per step, with no division or square root.
- **Exceptions over sentinel values.** Returning the last iterate or a NaN on failure invites callers like `fit_temperature` (`M07.7`) to use a wrong temperature. Distinct messages ("derivative", "diverged", "no convergence") tell the caller which assumption broke.
- **Alternatives.** Bisection or Brent's method (Going further) cannot diverge but converge linearly or superlinearly, and need a bracket the caller must supply. A safeguarded Newton combines both; this module keeps plain Newton so its behaviour, including its failures, is visible. For $1/\sqrt{x}$, computing `1 / np.sqrt(x)` is simpler in Python but is not the operation the C kernel performs, so it cannot serve as a step-for-step reference.
- **Tradeoffs accepted.** The stopping test looks at the step, not at $|f(x)|$, so on a function that is very flat near its root it can stop with a small step but a modest residual; and a multiple root (where $f'(x^\star) = 0$) converges only linearly and may exhaust the budget. Both are documented properties of Newton's method that the contract leaves to the caller.

### Explain it back

**1. Why does `newton` return `(x, n)` when $f(x) = 0$, but `(x_new, n + 1)` when the step test passes?**

<details><summary>Answer</summary>

`n` counts updates already made. An exact root needs no further update, so the count stays `n` (0 if $x_0$ is a root). When the step test passes, the update that produced `x_new` has just happened, so it is counted: `n + 1`. Getting this wrong makes `max_iter = 6` fail on a problem that needs exactly 6 updates.

</details>

**2. Why is the stopping test $|x_{n+1} - x_n| \le \text{tol} \cdot \max(1, |x_{n+1}|)$ rather than $|x_{n+1} - x_n| \le \text{tol}$?**

<details><summary>Answer</summary>

Float spacing near $x$ is about $2.2 \times 10^{-16}|x|$. Near $12649 \approx \sqrt{1.6 \times 10^8}$ that is $1.8 \times 10^{-12}$, so the last steps hop between neighbouring floats and never get below an absolute $10^{-12}$: the method would report "no convergence" at the right answer. Scaling by $|x_{n+1}|$ makes the test relative; the $\max(1, \cdot)$ keeps it absolute near 0, where relative precision is unreachable.

</details>

**3. Why check `math.isfinite(x_new)` separately when the derivative check already rejects non-finite slopes?**

<details><summary>Answer</summary>

A finite $f$ and a finite, nonzero, but tiny slope can still produce a step that overflows, and a diverging iteration like $\sqrt[3]{x}$ reaches inf through finite steps. Without the check the next iteration would evaluate $f(\infty)$ and fail with a misleading message about the derivative; catching it at the step that produced it reports "diverged" accurately.

</details>

**4. Why does `rsqrt_newton` build its constants with `dt.type(0.5)` and hoist `hx = half * xa`?**

<details><summary>Answer</summary>

Typed constants keep every operation in the working dtype, so float32 inputs give float32 arithmetic, the same rounding as the C port, rather than being promoted to float64 under older numpy casting rules. Hoisting $\tfrac12 x$ saves one multiplication per step and fixes the operation order the C code reproduces.

</details>

**5. With $g(y) = 1/y^2 - x$ the step needs no division. Why not use the simpler-looking $g(y) = y^2 - 1/x$?**

<details><summary>Answer</summary>

Its Newton step is $y - (y^2 - 1/x)/(2y) = \tfrac12(y + 1/(xy))$, which needs $1/x$ and a division by $y$ at each step: exactly the expensive operations the iteration is meant to avoid. $1/y^2 - x$ puts the unknown in the denominator of $g$, and the division cancels algebraically in the step.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. not checking for a zero or non-finite derivative | `ZeroDivisionError`, or a nan that later looks like "no convergence" | raises "derivative" before dividing; step 3 |
| 2. returning the last iterate when the method did not converge | the cycle's last point (0.0) reported as a root of $x^3 - 2x + 2$ | raises "no convergence" after the loop; step 7 |
| 2b. letting an inf iterate continue | the failure is blamed on a flat tangent one step later | checks `math.isfinite(x_new)` right after the step; step 5 |
| 3. an absolute stopping test | no convergence at the right answer near $1.3 \times 10^4$ | `tol * max(1.0, abs(x_new))`; step 6 |
| 4. computing the derivative once at $x_0$ (chord method) | linear convergence, or divergence when $f'$ changes a lot | calls `deriv(x)` inside the loop, including `central_diff` at each iterate; step 4 |
| 5. dropping a factor of $y$ in $y(\frac{3}{2} - \frac{1}{2} x y^2)$ | converges to the wrong value, $1/x$-like | `y * (three_halves - hx * y * y)`: two factors of $y$ inside, one outside |
| 6. computing float32 inputs in float64 | results that differ from the C port in the last bits, and a float64 array where float32 was expected | `dt = np.result_type(...)` and constants of type `dt`; step 8 |
| stepping uphill, $x + f/f'$, or multiplying by $f'$ | the hand example's iterates go wrong at step 1 | `x_new = x - fx / d` |
| always stepping once, even at an exact root | $(3.0, 1)$ instead of $(3.0, 0)$ | the `fx == 0.0` check comes before any step; step 2 |
| one Newton step fewer than asked | the error after two steps is $1.8 \times 10^{-3}$, not $5 \times 10^{-6}$ | `range(int(iters))` runs exactly `iters` steps |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.1` | `newton(f, None, x0)` calls `central_diff` at every iterate |
| Forward | `M09.5` | the same $y(\frac{3}{2} - \frac{1}{2}xy^2)$ in C, float32, from the bit-trick guess; its differential test compares against `rsqrt_newton` (Pass 6) |
| Forward | `L9.6` | `tl_rmsnorm_f32` multiplies by that reciprocal square root for every token |
| Forward | `M10.6` | optional: Muon orthogonalizes momentum with Newton-Schulz, Newton's method for a matrix function |
| Forward | `M09.3` | convergence order and error propagation, generalized |
| Forward | `M07.7` | `fit_temperature` calls `newton(g, dg, 0.0)` on the derivative of the NLL in $1/T$, with the variance of the logits as `dg` |

`M07.7` is the main call site; `M09.5` is an optional C side quest and `M10.6` is not authored yet.

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `rsqrt_newton` with the bit trick | x86 `rsqrtps` / ARM `frsqrte` plus one Newton step; CUDA `rsqrtf` | a hardware table lookup for the first 12 bits, then the same refinement | Intel intrinsics guide `_mm_rsqrt_ps`; ggml's `ggml_vec_*` RMSNorm paths |
| `newton` | SciPy `scipy.optimize.newton`, `brentq` | secant and Halley variants; Brent's method brackets the root so it can never diverge | `scipy/optimize/_zeros_py.py` |
| a Newton step that can fail | safeguarded Newton (`rtsafe`) | falls back to bisection whenever the Newton step leaves the bracket | Press et al., *Numerical Recipes*, section 9.4 |
| scalar Newton | Newton-Schulz in Muon | Newton's method for the matrix sign function, $X \leftarrow \frac{3}{2}X - \frac{1}{2}XX^\top X$, the same polynomial as section 2.4 | Jordan et al., Muon (2024); `M10.6` |
