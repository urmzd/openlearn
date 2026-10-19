# Convexity, smoothness, gradient descent, and Armijo line search

## Overview

| | |
|---|---|
| **Module** | `M10.1` · build · Python · Pass 2 · 2 to 3 h |
| **Solution** | [`course/solutions/python/tinyllm/optim/gd.py`](../../../../course/solutions/python/tinyllm/optim/gd.py): `gradient_descent`, `armijo_step` |
| **Contract** | [`course/contracts/py/tinyllm/optim/gd.pyi`](../../../../course/contracts/py/tinyllm/optim/gd.pyi) |
| **Needs** | no code dependency · reading: `M04.1` gradients, `M03.4` eigenvalues of a symmetric matrix |
| **Used by** | `M10.2` (SGD with momentum 0 is this gradient descent) · later: `L0.5` reads its loss curves with $\kappa$ and $2/L$, `M10.5` measures $\lambda_{\max}$ during training, `M09.3` generalizes $\kappa$ to matrices · together with the rest of Pass 2's math, it is what lets the autograd bigram of `L0.5` train |
| **Optional depth** | Boyd and Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), sections 3.1 and 9.2 to 9.3; Nocedal and Wright, *Numerical Optimization* (2nd ed.), ch. 3 |

## Key Takeaways

- On an $L$-smooth, $\mu$-strongly convex function, gradient descent with step $1/L$ shrinks the optimality gap by at least $1 - 1/\kappa$ per step, where $\kappa = L/\mu$ is the condition number.
- On a quadratic, each eigen-direction is multiplied by $1 - \eta\lambda_i$ per step, so the iteration converges exactly when $0 < \eta < 2/L$ and blows up beyond it.
- Backtracking with the Armijo condition finds a step that decreases $f$ enough without knowing $L$, and returns the first acceptable step of $\alpha_0, \rho\alpha_0, \rho^2\alpha_0, \dots$.
- Numerical edge cases are part of the algorithm: a NaN trial value must be rejected, and a direction that does not descend must be refused.

## How to read this chapter

Sections 1 to 3 derive the step-size limit $2/L$, the $1 - 1/\kappa$ rate, and the Armijo condition, and run both algorithms by hand. Section 4 fixes the interface. Section 5 walks through the solution line by line and explains why each check is there. Try to predict each line from section 2 before reading it.

---

## 1. Why now

In `L0.5` the bigram stops being a table of counts and becomes 65536 weights trained by the course's own autograd, and the first decision is the learning rate. Too large and the loss jumps to `inf`, then NaN, within a few steps; too small and it crawls for thousands of steps toward the count model's NLL that `L0.0` reached instantly. Both symptoms have exact explanations on a quadratic, the local model of every smooth loss: the step must stay under $2/L$, and the number of steps scales with the condition number $\kappa$. This module builds plain gradient descent and a line search, proves their rates on functions where the theory is exact, and gives `M10.2`'s optimizer protocol the update it generalizes.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f: \mathbb{R}^n \to \mathbb{R}$ | the function to minimize (a loss) | callable |
| $\nabla f(x)$ | its gradient at $x$ | `float64[n]` |
| $x^\star$, $f^\star$ | a minimizer and the minimum value | `float64[n]`, float |
| $\eta$ (`lr`) | the learning rate (step size) | float $> 0$ |
| $x_t$ | the iterate after $t$ steps | `float64[n]` |
| $L$ | smoothness constant: $\lVert\nabla f(x) - \nabla f(y)\rVert \le L \lVert x - y\rVert$ | float |
| $\mu$ | strong convexity constant | float $> 0$ |
| $\kappa = L/\mu$ | the condition number | float $\ge 1$ |
| $H$ | a symmetric positive definite matrix; for a quadratic, the Hessian | `float64[n, n]` |
| $\lambda_i$ | eigenvalues of $H$, with $\mu = \lambda_{\min}$ and $L = \lambda_{\max}$ | floats |
| $d$ | a search direction | `float64[n]` |
| $\alpha$, $\alpha_0$, $\rho$, $c$ | trial step, first trial, shrink factor, sufficient-decrease constant | floats |

**Convexity.** $f$ is convex if the chord between any two points of its graph lies above the graph: $f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y)$ for $\theta \in [0, 1]$. For differentiable $f$ this is equivalent to every tangent plane lying below the graph:

$$f(y) \ge f(x) + \nabla f(x)^\top (y - x).$$

Then $\nabla f(x^\star) = 0$ implies $f(y) \ge f(x^\star)$ for all $y$: every stationary point is a global minimum. Neural network losses are not convex, but near a minimum they look like convex quadratics, and that is where the rates below describe what you see.

**Smoothness bounds the function from above.** If the gradient is $L$-Lipschitz, the function never curves up faster than a parabola of curvature $L$ (the descent lemma):

$$f(y) \le f(x) + \nabla f(x)^\top (y - x) + \tfrac L2 \lVert y - x\rVert^2.$$

**Strong convexity bounds it from below.** $f$ is $\mu$-strongly convex if $f(y) \ge f(x) + \nabla f(x)^\top (y - x) + \tfrac\mu2 \lVert y - x\rVert^2$: it curves up at least as fast as a parabola of curvature $\mu$. For a quadratic $f(x) = \tfrac12 x^\top H x$, the gradient is $Hx$, and the best constants are the extreme eigenvalues of $H$ (`M03.4`): $L = \lambda_{\max}$, $\mu = \lambda_{\min}$.

**The condition number.** $\kappa = L/\mu$ measures how stretched the level sets are: $\kappa = 1$ is a round bowl, $\kappa = 100$ is a long narrow valley. `M09.3` later defines the condition number of a matrix, $\lVert A\rVert \lVert A^{-1}\rVert$; for a symmetric positive definite $H$ it is the same $\lambda_{\max}/\lambda_{\min}$.

**Gradient descent and its rate.** The update is $x_{t+1} = x_t - \eta \nabla f(x_t)$. Put $y = x_{t+1}$ in the descent lemma with $\eta = 1/L$:

$$f(x_{t+1}) \le f(x_t) - \tfrac{1}{2L} \lVert\nabla f(x_t)\rVert^2.$$

Strong convexity implies $\lVert\nabla f(x)\rVert^2 \ge 2\mu\,(f(x) - f^\star)$ (minimize both sides of its defining inequality over $y$). Combining,

$$f(x_{t+1}) - f^\star \le \left(1 - \frac{\mu}{L}\right)\left(f(x_t) - f^\star\right) \quad\Rightarrow\quad f(x_t) - f^\star \le \left(1 - \frac1\kappa\right)^t \left(f(x_0) - f^\star\right).$$

To shrink the gap by a factor $\varepsilon$ takes about $\kappa \ln(1/\varepsilon)$ steps: ten times the condition number, ten times the steps.

**The quadratic, exactly.** For $f = \tfrac12 x^\top H x$, write $x$ in the eigenvectors of $H$. Gradient descent is then independent in each coordinate:

$$x_{t+1, i} = (1 - \eta \lambda_i)\, x_{t, i}.$$

The coordinate shrinks when $\lvert 1 - \eta\lambda_i\rvert < 1$, that is $0 < \eta < 2/\lambda_i$. For all coordinates at once: $0 < \eta < 2/L$. Above $2/L$ the stiffest direction grows by $\lvert 1 - \eta L\rvert > 1$ per step, which is the loss that jumps to `inf`. At $\eta = 1/L$ the stiffest direction is solved in one step and the flattest shrinks by $1 - 1/\kappa$: the bound above is attained.

**Line search: a step without knowing $L$.** Real losses do not announce $L$. Given a direction $d$ with $\nabla f(x)^\top d < 0$ (a descent direction, for example $d = -\nabla f(x)$), try $\alpha = \alpha_0$ and accept it if the **Armijo condition** holds:

$$f(x + \alpha d) \le f(x) + c\,\alpha\, \nabla f(x)^\top d, \qquad 0 < c < 1.$$

The right side is a line through $f(x)$ with a fraction $c$ of the initial slope; the step must land below it. Otherwise set $\alpha \leftarrow \rho\alpha$ and try again. For an $L$-smooth $f$ and $d = -\nabla f$, the descent lemma shows every $\alpha \le 2(1-c)/L$ is accepted, so backtracking stops after a bounded number of halvings with a step of at least $2\rho(1-c)/L$. If $d$ is not a descent direction, no small step helps, and the search must refuse instead of halving forever.

**NaN is a rejection.** A trial point outside the function's domain (a log of a negative number, an overflow) gives `nan`, and every comparison with NaN is false. Written as `if f(x + a d) <= bound: return a`, a NaN fails the test and the step shrinks. Written as `while f(x + a d) > bound: a *= rho`, the NaN makes the loop condition false and the bad step is accepted. Same math, opposite behavior.

## 3. Worked example by hand

**Gradient descent on $f(x) = \tfrac12(x_1^2 + 10 x_2^2)$.** $H = \mathrm{diag}(1, 10)$, so $\mu = 1$, $L = 10$, $\kappa = 10$, and $\nabla f = (x_1, 10 x_2)$. Start at $x_0 = (10, 1)$ with $\eta = 1/L = 0.1$:

| $t$ | $x_t$ | $\nabla f(x_t)$ | $f(x_t)$ | $f(x_t)/f(x_{t-1})$ |
|---|---|---|---|---|
| 0 | $(10, 1)$ | $(10, 10)$ | 55 | |
| 1 | $(9, 0)$ | $(9, 0)$ | 40.5 | 0.736 |
| 2 | $(8.1, 0)$ | $(8.1, 0)$ | 32.805 | 0.81 |

The stiff coordinate is multiplied by $1 - 0.1 \cdot 10 = 0$ and is done in one step; the flat one by $1 - 0.1 \cdot 1 = 0.9$, so its share of $f$ shrinks by $0.81$ per step, inside the bound $1 - 1/\kappa = 0.9$. With $\eta = 0.21$, just over $2/L = 0.2$, the stiff coordinate is multiplied by $1 - 2.1 = -1.1$ per step: it alternates sign and grows without bound.

**One Armijo search** on $f(x) = x^2$ at $x = 1$ along $d = -f'(1) = -2$, with $\alpha_0 = 1$, $c = 10^{-4}$, $\rho = 0.5$. The slope is $f'(1)\,d = -4$.

| $\alpha$ | $x + \alpha d$ | $f$ | bound $1 + c\,\alpha\,(-4)$ | accept? |
|---|---|---|---|---|
| 1 | $-1$ | 1 | 0.9996 | no |
| 0.5 | 0 | 0 | 0.9998 | yes |

The full step overshoots to the mirror point with the same value; half of it lands on the minimum. The search returns 0.5.

These numbers are the first case in section 4's table.

## 4. The interface

```python
# python/tinyllm/optim/gd.py
def gradient_descent(f, grad, x0, lr: float, steps: int) -> list[NDArray]   # [x_0, ..., x_steps]
def armijo_step(f, grad, x, d, alpha0: float = 1.0, c: float = 1e-4, rho: float = 0.5) -> float
```

`gradient_descent` returns `steps + 1` independent float64 arrays and never modifies `x0`. It evaluates `f` at each iterate only to detect divergence: a non-finite value raises `FloatingPointError` naming the step. `armijo_step` raises `ValueError` for a non-descent direction or parameters outside $0 < c < 1$, $0 < \rho < 1$, $\alpha_0 > 0$, and `RuntimeError` after 60 reductions without success.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked trajectory and search | unit | the section 3 trajectory and the Armijo search | everyone agrees on the update |
| every iterate returned as a copy | unit | `steps + 1` independent arrays, `x0` untouched | trajectories are data you plot and compare |
| the rate on quadratics | property | gap $\le (1 - 1/\kappa)^t$ at every step for $\kappa$ = 2, 10, 100 | the theorem behind learning-rate choices |
| an exact rational trajectory | differential | equals an exact `fractions.Fraction` recomputation | the float trajectory is the math, to rounding |
| divergence raises | boundary | $\eta = 3 > 2/L$ raises `FloatingPointError`; $\eta = 0.19$ converges | a diverging run fails loudly |
| bad arguments | boundary | $\eta \le 0$ and negative steps are `ValueError` | caller bugs |
| Armijo returns the first acceptable step | property | the step satisfies the condition and twice it does not, on 50 Rosenbrock points | the largest acceptable step on the grid |
| Armijo accepts $\alpha_0$ | unit | a full step that works is returned unchanged | no needless shrinking |
| Armijo keeps halving | unit | $f = x^2$ at 1 along $-4$ needs two halvings and gets $\alpha = 0.25$ | each trial shrinks the last one |
| Armijo rejects NaN steps | boundary | NaN trial values are rejected; the search returns 0.125 | functions with a restricted domain |
| Armijo needs a descent direction | boundary | ascent and orthogonal directions, bad $c$, $\rho$, $\alpha_0$ raise | no infinite halving |
| Armijo gives up | boundary | a lying gradient ends in `RuntimeError`; the 60th halving, $2^{-60}$, is still tried | no infinite loop, and no reduction skipped |
| line search descends Rosenbrock | property | 300 steepest-descent steps with Armijo decrease $f$ monotonically | a step size without knowing $L$ |

## 5. The solution, step by step

### From first principles

1. **The update is one line.** $x_{t+1} = x_t - \eta\nabla f(x_t)$ (section 2). Everything else in `gradient_descent` exists to make the trajectory trustworthy as data.
2. **A trajectory is a list of independent snapshots.** Callers plot and compare iterates, so each must be a separate array, and the caller's `x0` must not change. Writing `x = x - lr * g` (a new array each step) rather than `x -= lr * g` gives that for free.
3. **Divergence must be loud.** Above $2/L$ the stiff coordinate grows by $\lvert 1 - \eta L\rvert > 1$ per step (section 2). A list of `inf` and NaN returned as a "trajectory" hides the bug, so every iterate, including the last, is checked and the first non-finite value raises.
4. **Armijo needs the slope once.** The bound $f(x) + c\alpha\,\nabla f(x)^\top d$ uses $f(x)$ and the slope at $\alpha = 0$; compute both once, before the loop.
5. **Refuse a non-descent direction.** If the slope is $\ge 0$, no positive step can satisfy the bound for small $\alpha$, so backtracking would halve forever; raise instead.
6. **Write the test so NaN fails it.** `if f(new) <= bound: return` treats NaN as a rejection; the `while f(new) > bound` form accepts it (section 2).
7. **Bound the loop.** For an $L$-smooth $f$ the search stops after a bounded number of halvings; if it does not, the gradient is lying, and after 60 reductions (a factor $2^{-60} \approx 10^{-18}$) it raises rather than returning a meaningless step.

### The code, annotated

```python
MAX_REDUCTIONS = 60


def gradient_descent(f, grad, x0, lr: float, steps: int) -> list[NDArray]:
    if not lr > 0:                        # also rejects lr = nan
        raise ValueError(f"lr must be > 0, got {lr}")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    x = np.array(x0, dtype=np.float64, copy=True)   # never alias the caller's x0
    out = [x]                                        # x_0 is part of the trajectory
    for t in range(steps + 1):
        fx = float(f(x))
        if not (math.isfinite(fx) and np.isfinite(x).all()):
            raise FloatingPointError(f"gradient descent diverged at step {t}: f(x_t) = {fx}")
        if t == steps:
            break                          # x_steps is checked, but not stepped from
        x = x - lr * np.asarray(grad(x), dtype=np.float64)   # a new array, not in place
        out.append(x)
    return out
```

The loop runs `steps + 1` times so that the divergence check sees every iterate, $x_0$ through $x_{\text{steps}}$, while only `steps` updates are made. `not lr > 0` is used instead of `lr <= 0` because a NaN learning rate fails both comparisons, and the first form rejects it. In the hand example, `lr=0.1` from $(10, 1)$ produces `[(10, 1), (9, 0), (8.1, 0)]` for `steps=2`; `lr=3` on the same quadratic grows by $\lvert 1 - 30\rvert = 29$ per step in the stiff coordinate and raises at the first iterate whose $f$ is no longer finite.

```python
def armijo_step(f, grad, x, d, alpha0=1.0, c=1e-4, rho=0.5) -> float:
    if not (0 < c < 1 and 0 < rho < 1 and alpha0 > 0):
        raise ValueError(f"need 0 < c < 1, 0 < rho < 1, alpha0 > 0; got c={c}, rho={rho}, alpha0={alpha0}")
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    fx = float(f(x))
    slope = float(np.dot(np.ravel(grad(x)), np.ravel(d)))   # grad f(x)^T d, any shape
    if not slope < 0:                       # ascent, orthogonal, or nan: refuse
        raise ValueError(f"d is not a descent direction: grad(x) . d = {slope}")
    alpha = alpha0
    for _ in range(MAX_REDUCTIONS + 1):     # alpha0 and 60 reductions: 61 trials
        # Written so that a nan f(x + alpha d) fails the test and shrinks alpha.
        if f(x + alpha * d) <= fx + c * alpha * slope:
            return alpha                    # the first acceptable step, unshrunk
        alpha *= rho                        # shrink the previous trial, not alpha0
    raise RuntimeError(f"no step satisfies the Armijo condition after {MAX_REDUCTIONS} reductions")
```

Trace section 3's search: $f(1) = 1$, slope $= 2 \cdot (-2) = -4$. Trial $\alpha = 1$: $f(-1) = 1 > 1 - 4 \times 10^{-4}$, reject, $\alpha = 0.5$. Trial $\alpha = 0.5$: $f(0) = 0 \le 0.9998$, return 0.5. The loop count is `MAX_REDUCTIONS + 1` so that the 60th reduction, $\alpha_0 \cdot 2^{-60}$, is actually tried before giving up. `np.ravel` lets $x$ be a matrix (a weight array) as well as a vector.

Full solution: [`gd.py`](../../../../course/solutions/python/tinyllm/optim/gd.py).

### Why this solution

- **Cost.** `gradient_descent` makes `steps` gradient calls and `steps + 1` function calls, and stores `steps + 1` arrays: $O(\text{steps} \cdot n)$ memory, which is the point (it returns the trajectory). `armijo_step` costs one gradient, one $f(x)$, and one $f$ per trial; on an $L$-smooth function the number of trials is at most $1 + \log_{1/\rho}(\alpha_0 L / (2(1 - c)))$.
- **Copies over in-place updates.** In-place `x -= lr * g` saves an allocation per step, but appending the same array object every step makes every entry the final iterate. For a reference whose output is a trajectory, a fresh array per step is the correct choice; `M10.2`'s optimizers update in place because they return nothing.
- **Raise over clamp or skip.** Silently stopping at the last finite iterate would hide a learning-rate bug; raising names the step where it happened.
- **Backtracking Armijo over exact line search.** Minimizing $f(x + \alpha d)$ exactly costs many evaluations and buys little for gradient descent; Armijo needs only $f$ values, guarantees sufficient decrease, and is the building block of Wolfe searches (Going further). Halving instead of interpolating is simpler and loses at most a factor $\rho$ in step length.

### Explain it back

**1. Why does the gradient-descent loop run `steps + 1` times when it makes only `steps` updates?**

<details><summary>Answer</summary>

So that every iterate, including the last one, passes the divergence check. The final pass checks $x_{\text{steps}}$ and then breaks before computing another update. Running only `steps` times would return a final iterate that might already be `inf`.

</details>

**2. Why `x = x - lr * g` and not `x -= lr * g`?**

<details><summary>Answer</summary>

The trajectory list holds references. With an in-place update, every appended entry is the same array, so the whole list shows the final iterate; and on the first step it would also modify the caller's `x0` if `x` were not a copy. Creating a new array per step makes each entry an independent snapshot.

</details>

**3. What goes wrong if the acceptance test is written `while f(x + alpha * d) > bound: alpha *= rho`?**

<details><summary>Answer</summary>

When the trial point is outside the function's domain, $f$ returns NaN, and `nan > bound` is false, so the loop exits and accepts the bad step. In the solution's form, `nan <= bound` is also false, so the step is rejected and shrunk. The two forms are equal in exact arithmetic and opposite on NaN.

</details>

**4. Why does `armijo_step` raise for a direction with $\nabla f(x)^\top d \ge 0$ instead of searching anyway?**

<details><summary>Answer</summary>

If the slope is $\ge 0$, $f$ does not decrease to first order along $d$. With a positive slope the bound $f(x) + c\alpha\,\nabla f(x)^\top d$ lies above $f(x)$ and small steps increase $f$; with a zero slope the bound asks for no decrease at all. Either way backtracking cannot deliver sufficient decrease: it halves until it gives up, or accepts a step that makes no progress. Refusing up front reports the caller's bug where it happened.

</details>

**5. With $\eta = 1/L$ on $f = \tfrac12(x_1^2 + 10x_2^2)$, why is the stiff coordinate finished after one step?**

<details><summary>Answer</summary>

Gradient descent multiplies each eigen-coordinate by $1 - \eta\lambda_i$. With $\eta = 1/L = 1/10$ and $\lambda_2 = 10$, that factor is exactly 0, so $x_2$ becomes 0 in one step; the flat coordinate is multiplied by $1 - 1/10 = 0.9$ per step, which is what limits the rate to $1 - 1/\kappa$.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a learning rate above $2/L$ with no divergence check | a list of `inf` and NaN returned as a trajectory | checks every iterate and raises `FloatingPointError`; step 3 |
| 2. `while f(new) > bound` | a NaN trial point is accepted | `if f(new) <= bound: return`, so NaN is a rejection; step 6 |
| 3. searching along a direction that does not descend | halving forever, or accepting an ascent | `if not slope < 0: raise ValueError`; step 5 |
| 4. updating `x` in place and appending it | every entry of the trajectory is the final iterate; `x0` changes | copies `x0` once and builds a new array each step; step 2 |
| 5. the sign of the step | gradient ascent | `x - lr * grad(x)` |
| 6. the sign of the Armijo bound | steps that increase $f$ are accepted | the bound is `fx + c * alpha * slope` with `slope < 0` |
| 7. shrinking once more after success | steps half as long as they could be | returns `alpha` before `alpha *= rho` |
| 8. returning a vanishing step instead of giving up | silent stalls at $\alpha = 10^{-18}$ | raises `RuntimeError` after `MAX_REDUCTIONS`; step 7 |
| 9. leaving $x_0$ out of the trajectory | every index is off by one | `out = [x]` before the loop |
| 10. restarting each trial from $\alpha_0$, `alpha = alpha0 * rho` | only $\alpha_0$ and $\rho\alpha_0$ are ever tried; a search that needs two halvings gives up | `alpha *= rho` shrinks the previous trial |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M04.1` | gradients, and how to check them |
| Back | `M03.4` | the eigenvalues that give $L$, $\mu$, and $\kappa$ |
| Forward | `M10.2` | SGD with momentum 0 performs exactly this update, in place, behind the `Optimizer` protocol |
| Forward | `L0.5` | the bigram's learning rate and loss curve, read with $2/L$ and $\kappa$ |
| Forward | `M10.5` | the top Hessian eigenvalue during training, against the $2/\eta$ edge of stability |
| Forward | `M09.3` | the condition number of a matrix and of a problem |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `armijo_step` | `scipy.optimize.line_search` | the strong Wolfe conditions (a curvature test as well as Armijo) with interpolation instead of halving | `scipy/optimize/_linesearch.py` |
| line search in a training loop | PyTorch `torch.optim.LBFGS(line_search_fn="strong_wolfe")` | quasi-Newton directions that undo much of $\kappa$ | `torch/optim/lbfgs.py` |
| gradient descent on quadratics | conjugate gradient | $\sqrt\kappa$ instead of $\kappa$ steps on quadratics, by choosing $H$-orthogonal directions | Nocedal and Wright, ch. 5 |
