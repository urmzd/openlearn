# Partial derivatives, gradients, gradcheck

## Overview

| | |
|---|---|
| **Module** | `M04.1` · build · Python · Pass 2 · about 1 to 2 h reading |
| **Solution** | [`gradcheck.py`](../../../../course/solutions/python/tinyllm/num/gradcheck.py): `numerical_grad`, `gradcheck`, and its `GradcheckReport` |
| **Contract** | [`course/contracts/py/tinyllm/num/gradcheck.pyi`](../../../../course/contracts/py/tinyllm/num/gradcheck.pyi) |
| **Needs** | `M01.1` `central_diff` ([`diff.py`](../../../../course/solutions/python/tinyllm/num/diff.py)), one call per coordinate |
| **Used by** | `M04.2` builds `vjp_numeric` on `numerical_grad` · later `L0.2`'s `F.gradcheck_all()` runs every op of the autograd library through `gradcheck`, behind the system's `{tinyllm} gradcheck --suite all` command |
| **Optional depth** | OpenStax, *Calculus Volume 3* (free), sections 4.3 and 4.6 (partial derivatives, the gradient); Nocedal and Wright, *Numerical Optimization*, section 8.1 (finite-difference gradients); the PyTorch autograd notes on `gradcheck` |

## Key Takeaways

- A **partial derivative** $\partial f / \partial x_i$ is an ordinary derivative along one coordinate with every other coordinate frozen; the **gradient** $\nabla f$ collects one per input element, in the input's shape.
- A numerical gradient is one central difference per element, $2n$ evaluations of $f$ for $n$ elements, exact on quadratics.
- An element passes when $|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n|$: absolute near zero, relative for large gradients.
- `gradcheck` must reject the bugs backward passes really have (a transposed gradient, a factor of 2, one forgotten element) and point at the element furthest past its tolerance.
- It perturbs float64 **copies**, so the caller's parameters are untouched and integer or float32 inputs still get exact steps.

## How to read this chapter

Sections 1 to 3 build partial derivatives, gradients, and the numerical check from first principles and work one by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L0.1` and `L0.2` are next: an autograd engine and a library of about 40 differentiable ops, each with a hand-written backward rule. A backward rule is a claim, "this array is the gradient of the loss with respect to that input", and claims need a referee. The referee is the definition of the derivative, applied one coordinate at a time: perturb one weight by $\pm\epsilon$, rerun the forward pass, divide. `M01.1` gives the one-coordinate version; this chapter turns it into the gradient of a scalar function of several arrays and into the check that compares it with an analytic gradient. The system's `{tinyllm} gradcheck --suite all` command runs the whole op library through it, so every backward rule written in Pass 2 passes through this file.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f(X_1, \ldots, X_m)$ | a scalar function of $m$ arrays (a loss) | `Callable[..., float]` |
| $X_k$ | the $k$-th input array; $X_k[i]$ one element, $i$ a multi-index | `float64[...]` |
| $\frac{\partial f}{\partial X_k[i]}$ | the partial derivative with respect to that element | `float` |
| $\nabla_{X_k} f$ | the gradient with respect to $X_k$, same shape as $X_k$ | `float64[...]` |
| $\epsilon$ | the finite-difference step (`eps`), default $10^{-6}$ | `float` |
| $a$, $n$ | one element of the analytic and of the numerical gradient | `float` |
| rtol, atol | relative and absolute tolerances, defaults $10^{-5}$ and $10^{-7}$ | `float` |
| $e_i$ | the array that is 1 at position $i$ and 0 elsewhere | same shape as $X_k$ |

### 2.1 Partial derivatives

For a function of two numbers, $f(x, y) = x^2 y + 3y$, freeze $y$ and differentiate in $x$: $\frac{\partial f}{\partial x} = 2xy$. Freeze $x$ and differentiate in $y$: $\frac{\partial f}{\partial y} = x^2 + 3$. Formally,

$$\frac{\partial f}{\partial x}(x, y) = \lim_{h \to 0}\frac{f(x + h, y) - f(x, y)}{h},$$

the one-variable derivative of `M01.1` along the $x$ direction. A partial derivative measures how $f$ responds to one input when nothing else moves, which is exactly the question a training step asks of each weight.

### 2.2 The gradient

The **gradient** lists every partial derivative: $\nabla f(x, y) = (2xy,\; x^2 + 3)$. For a function of arrays there is one partial derivative per element, and the gradient with respect to $X_k$ is arranged in $X_k$'s shape, so that a training step is simply $X_k \leftarrow X_k - \eta\,\nabla_{X_k} f$. Two facts make the gradient the right object. First, for a small change $\Delta$ in the inputs, $f$ changes by about $\sum_{k,i} \frac{\partial f}{\partial X_k[i]}\,\Delta_k[i]$, the dot product of the gradient with $\Delta$ (a first-order Taylor expansion, `M02.1`). Second, among all directions of length 1 that dot product is largest along the gradient, so $-\nabla f$ is the direction of steepest descent (`M10.1`). A gradient is only defined for a **scalar** $f$; the derivative of a vector-valued function is a matrix, the Jacobian of `M04.2`.

### 2.3 Central differences per coordinate

For each input $k$ and each element $i$:

$$n_k[i] = \frac{f(\ldots, X_k + \epsilon e_i, \ldots) - f(\ldots, X_k - \epsilon e_i, \ldots)}{2\epsilon},$$

which is `central_diff` applied to the one-variable function $t \mapsto f(\ldots, X_k \text{ with element } i \text{ set to } t, \ldots)$ at $t = X_k[i]$, with step $\epsilon$. That is how `numerical_grad` is written: it never re-derives the formula, it calls `M01.1`'s `central_diff`. The error analysis carries over: truncation $\frac{1}{6}|f'''|\epsilon^2$, rounding about $\varepsilon |f| / \epsilon$. With $\epsilon = 10^{-6}$ and values of size 1, both are near $10^{-10}$, comfortably under the tolerances below. On a quadratic, $f(x) = \frac12 x^\top A x + b^\top x$ with gradient $Ax + b$, the third derivative is zero and the numerical gradient is exact up to rounding.

Three details decide whether this is trustworthy. **Copies**: perturb float64 copies of the inputs, never the caller's arrays (they are the model's parameters). **Restore**: put each element back before moving to the next, or every later partial derivative is taken at the wrong point. **Promote**: an integer array cannot hold $x + 10^{-6}$ (it rounds back to $x$, and the gradient comes out 0), and float32 moves $x$ only in steps of about $6 \times 10^{-8}|x|$, so the step actually taken is not $\epsilon$. The cost is $2n$ evaluations of $f$ for $n$ input elements: fine for tests on small inputs, hopeless for training, which is why backpropagation exists.

### 2.4 When is a gradient right?

Compare elementwise, with a mixed test:

$$|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n| .$$

Near zero the absolute term dominates ($10^{-7}$: a numerical gradient of $10^{-9}$ is indistinguishable from 0); for large gradients the relative term does ($10^{-5}|n|$: at $n = 100$ the allowance is $10^{-3}$). A pure relative test would reject a correct analytic 0 against a numerical $10^{-11}$; a pure absolute test would accept a gradient of 100 that is off by 0.5 percent at a loose atol, or reject a correct one at a tight one. A **nan** in either gradient must fail: every comparison with nan is false, so `diff > tol` would let it pass; test `not (diff <= tol)` instead.

The report says `ok`, the largest absolute error, the largest relative error $|a - n| / \max(|n|, \mathrm{atol})$, and **where** to look: the input and multi-index of the element whose error is the largest multiple of its own allowance. That is not the largest absolute error: an error of $5 \times 10^{-3}$ on a gradient of 1000 is within tolerance, an error of $2 \times 10^{-4}$ on a gradient of $10^{-3}$ is a bug.

## 3. Worked example by hand

$f(x) = x_0^2 x_1 + 3 x_1$ at $x = (1, 2)$, with the large step $\epsilon = 0.1$ so the arithmetic is easy by hand. The gradient is $(2 x_0 x_1, x_0^2 + 3) = (4, 4)$.

| Element | $f(x + \epsilon e_i)$ | $f(x - \epsilon e_i)$ | quotient | analytic |
|---|---|---|---|---|
| $x_0$ | $f(1.1, 2) = 1.21 \cdot 2 + 6 = 8.42$ | $f(0.9, 2) = 0.81 \cdot 2 + 6 = 7.62$ | $0.8 / 0.2 = 4$ | 4 |
| $x_1$ | $f(1, 2.1) = 2.1 + 6.3 = 8.4$ | $f(1, 1.9) = 1.9 + 5.7 = 7.6$ | $0.8 / 0.2 = 4$ | 4 |

Both are exact even at $\epsilon = 0.1$: $f$ is quadratic in $x_0$ and linear in $x_1$, so the central difference has no truncation error (section 2.3). Note the restore: the $x_1$ row is evaluated at $x_0 = 1$, not at the 0.9 left over from the row before. A backward pass that returned $(2, 4)$ (a factor of 2 lost on $x_0$) fails at element 0 with $|2 - 4| = 2 > 10^{-7} + 10^{-5} \cdot 4$.

## 4. The interface

```python
@dataclass
class GradcheckReport: ok: bool; max_abs_err: float; max_rel_err: float; worst_input: int; worst_index: tuple
def numerical_grad(f, inputs: list[ArrayLike], eps: float = 1e-6) -> list[NDArray]: ...
def gradcheck(f, inputs: list[ArrayLike], analytic: list[ArrayLike],
              eps: float = 1e-6, rtol: float = 1e-5, atol: float = 1e-7) -> GradcheckReport: ...
```

`numerical_grad` returns one float64 array per input, in its shape. `gradcheck` raises `ValueError` for mismatched lists or shapes (a bug in the caller, not a numeric disagreement), and reports everything else. `f` must return a single number.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example | unit | section 3: $(4, 4)$ at $\epsilon = 0.1$, `ok` | pins the definition |
| Quadratics | property | numerical gradient of $\frac12 x^\top A x + b^\top x$ equals $Ax + b$ to $10^{-8}$ | central differences are exact there |
| Matrix and several inputs | golden | $\nabla$ of $\mathrm{sum}(\tanh(WX))$ is $G X^\top$ and $W^\top G$ | layers have weights and inputs |
| Transposed gradient | unit | a transposed weight gradient is not ok, worst input 0 | the most common backward bug |
| Gradient off by 2 | unit | $x$ instead of $2x$ is rejected, worst at the largest element, relative error 0.5 | a lost factor keeps every sign right |
| One zeroed element | unit | one zeroed element of the second input is found at $(2, 1)$ | the report says where to look |
| Mixed tolerance | boundary | the mixed test at gradients $10^{-3}$ and 100 | both regimes of section 2.4 |
| Worst element | unit | the worst element is the one furthest past its allowance | not the largest absolute error |
| NaN gradient | boundary | a nan analytic gradient is not ok | nan comparisons are always false |
| Caller's arrays | unit | they come back bit for bit | they are the model's parameters |
| Integer and float32 inputs | boundary | exact gradients; $f$ sees float64 | steps survive the dtype |
| Defaults | unit | step $10^{-6}$, rtol $10^{-5}$, atol $10^{-7}$ | what `L0.2` relies on |
| Mismatched arguments | boundary | wrong counts and shapes raise `ValueError` | caller bugs are not numeric errors |
| Non-scalar $f$ | boundary | a vector-valued $f$ raises; a 1-element array is fine | gradients are for scalars |

## 5. The solution, step by step

### From first principles

1. A partial derivative is a one-variable derivative with everything else frozen (2.1). So the numerical gradient is `central_diff` applied to a one-variable function that writes $t$ into one element and calls $f$: no new difference formula, just a loop over elements.
2. The loop must not touch the caller's arrays and must take steps that are representable, which forces float64 copies made once at the start (2.3).
3. Each element is perturbed in place in the copy, and must be restored before the next one, even if $f$ raises; a `try/finally` guarantees it.
4. Iterating over a flat view (`reshape(-1)` of a contiguous copy) visits every element of any shape, and writing into a flat view of a zero array of the same shape produces the gradient already in the input's shape.
5. The comparison is $|a - n| \le \mathrm{atol} + \mathrm{rtol}|n|$ elementwise (2.4). To make NaN fail, the code tests the negation of "passes" rather than "exceeds".
6. The location to report is the element with the largest ratio of error to allowance, across all inputs, with ties going to the first in row-major order and the first input.

### The code, annotated

The scalar guard and the numerical gradient:

```python
def _scalar(v) -> float:
    a = np.asarray(v)
    if a.size != 1:   # a 1-element array is accepted; a vector is a caller bug
        raise ValueError(f"f must return one number, got an array of shape {a.shape}")
    return float(a.reshape(()))


def numerical_grad(f, inputs, eps=1e-6):
    if len(inputs) == 0:
        raise ValueError("inputs is empty: there is nothing to differentiate")
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    # Private float64 copies: an int array would round x + eps back to x, and
    # float32 would round it to the nearest 6e-8 * |x|.
    xs = [np.array(x, dtype=np.float64, copy=True) for x in inputs]
    grads = []
    for k, x in enumerate(xs):          # every input, not just the first
        g = np.zeros_like(x)
        flat, gflat = x.reshape(-1), g.reshape(-1)  # views into x and g
        for i in range(flat.size):
            orig = flat[i]

            def along(t: float, i: int = i) -> float:
                flat[i] = t             # writes through the view into xs[k]
                return _scalar(f(*xs))  # f sees all inputs, one element moved

            try:
                gflat[i] = central_diff(along, orig, eps)
            finally:
                flat[i] = orig  # restore before the next coordinate
        grads.append(g)
    return grads
```

`central_diff` (from `M01.1`) evaluates `along(x + eps)` and `along(x - eps)` and divides by `(x + eps) - (x - eps)`, the step actually representable in float64, rather than by $2\epsilon$. On the worked example it returns $0.8/0.2 = 4$ for both elements.

`gradcheck` validates the shapes, computes the numerical gradient once, and then folds each input's comparison into the report:

```python
    num = numerical_grad(f, inputs, eps)
    ok, max_abs, max_rel = True, 0.0, 0.0
    worst = (-1.0, 0, tuple(0 for _ in num[0].shape))
    for k, (a, n) in enumerate(zip(ana, num)):
        if a.size == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            diff = np.abs(a - n)
            # A nan never passes: compare the negation, since nan <= x is False.
            # One failing element anywhere makes the whole check fail.
            ok = ok and not (~(diff <= atol + rtol * np.abs(n))).any()
            diff = np.where(np.isnan(diff), np.inf, diff)  # a nan is the worst error
            max_abs = max(max_abs, float(diff.max()))
            rel = diff / np.maximum(
                np.abs(n), atol if atol > 0 else np.finfo(float).tiny
            )
            max_rel = max(max_rel, float(rel.max()))
            # How far each element is past its own allowance (> 1 fails).
            ratio = np.where(diff == 0, 0.0, diff / (atol + rtol * np.abs(n)))
        i = int(np.argmax(ratio))  # first maximum in row-major order
        if ratio.reshape(-1)[i] > worst[0]:   # strict: earlier inputs win ties
            worst = (
                float(ratio.reshape(-1)[i]),
                k,
                tuple(int(j) for j in np.unravel_index(i, a.shape)),
            )
```

The report then carries `ok`, the two maxima, and `worst_input`, `worst_index`.

Full solution: [`gradcheck.py`](../../../../course/solutions/python/tinyllm/num/gradcheck.py).

### Why this solution

- **Cost.** $2n$ evaluations of $f$ for $n$ input elements, $O(n)$ extra memory for the copies and gradients. That is the price of checking every element; it is acceptable because checks run on small inputs.
- **Why central and not forward differences.** A forward difference has error $O(\epsilon)$, about $10^{-6}$ at the default step, which is above `atol` and would make correct gradients fail. Central differences have error $O(\epsilon^2)$ and are exact on quadratics.
- **Why a ratio for the worst element.** The ratio to the allowance is what decides pass or fail, so the element with the largest ratio is the one a person should look at. The largest absolute error is often a correct large gradient.
- **Why not random directional checks.** Checking $u^\top J v$ for random $u, v$ costs 2 evaluations instead of $2n$, but cannot say which element is wrong. Element-by-element is the right default for debugging backward rules; the directional form is in "Going further".
- **Tradeoff accepted.** The `atol` floor makes a gradient of $10^{-9}$ indistinguishable from 0. That is deliberate: below the finite-difference noise the comparison has no information.

### Explain it back

1. Why does `numerical_grad` copy the inputs to float64 instead of perturbing them in their own dtype?

   <details><summary>Answer</summary>

   Two reasons. The caller's arrays are model parameters and must not change. And an integer array cannot hold $x + 10^{-6}$, while float32 moves only in steps of about $6 \times 10^{-8}|x|$, so the step actually taken would be 0 or far from $\epsilon$.

   </details>

2. What goes wrong without the `finally: flat[i] = orig`?

   <details><summary>Answer</summary>

   After element $i$, the copy would hold $x_i - \epsilon$ (the last value `central_diff` wrote). Every later partial derivative would then be taken at a shifted point; in the worked example the $x_1$ row would be evaluated at $x_0 = 0.9$.

   </details>

3. Why is `ok` computed as `not (~(diff <= tol)).any()` and not `not (diff > tol).any()`?

   <details><summary>Answer</summary>

   Every comparison with NaN is false. `diff > tol` is false for a NaN, so a NaN gradient would count as passing. `diff <= tol` is also false for a NaN, so its negation is true and the check fails, as it should.

   </details>

4. An element has $n = 1000$, $a = 1000.005$; another has $n = 10^{-3}$, $a = 1.2 \times 10^{-3}$. Which is reported as worst?

   <details><summary>Answer</summary>

   The second. The first has allowance $10^{-7} + 10^{-5} \cdot 1000 \approx 10^{-2}$ and error $5 \times 10^{-3}$, ratio about 0.5 (passes). The second has allowance about $1.1 \times 10^{-7}$ and error $2 \times 10^{-4}$, ratio about 1800.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. combining the per-element verdicts wrongly (`all` instead of `any`) | a transposed or partly zeroed gradient is reported ok | `ok` is false as soon as any element of any input fails |
| 2. not restoring an element before the next one | every later partial derivative is taken at a shifted point | restores `flat[i] = orig` in a `finally`; see step 3 |
| 2b. perturbing the caller's arrays in place | the model's weights change during a check | perturbs private copies `xs` |
| 3. perturbing in the caller's dtype | gradient 0 for int inputs; two digits for float32 | copies with `dtype=np.float64`; see step 2 |
| 4. `diff > tol` as the failure test | a nan gradient passes | tests `~(diff <= tol)`; see step 5 |
| a one-sided difference per coordinate | errors near $10^{-6}$ instead of $10^{-10}$; quadratics no longer exact | calls `central_diff` |
| dividing by $4\epsilon$ (the $\epsilon$ versus $2\epsilon$ confusion) | every numerical gradient halved | lets `central_diff` divide by the actual step `xp - xm` |
| differentiating or comparing only the first input | the input gradient is never checked | loops over every input in both functions |
| reporting the largest absolute error as the worst | the report points at a correct large gradient | ranks by `diff / (atol + rtol * abs(n))`; see step 6 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.1` | `numerical_grad` calls `central_diff(along, x_i, eps)` for every element |
| Forward | `M04.2` | `vjp_numeric(f, x, u)` is `numerical_grad` of the scalar $u \cdot f(x)$ |
| Forward | `L0.2` | `F.gradcheck_all()` checks every op's backward with this `gradcheck`, behind the system's `{tinyllm} gradcheck --suite all` command |
| Forward | `M10.1` | gradient descent steps along $-\nabla f$ |
| Forward | `L4.1` to `L7.9` | the model test suites gradcheck every new backward rule |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `gradcheck` | PyTorch `torch.autograd.gradcheck` and `gradgradcheck` | complex inputs, sparse and batched layouts, second derivatives, a fast mode that checks random projections $u^\top J v$ instead of every element | `torch/autograd/gradcheck.py` |
| `numerical_grad` | JAX `jax.test_util.check_grads` | checks forward and reverse mode up to a chosen order | `jax/_src/public_test_util.py` |
| `atol + rtol * abs(n)` | `numpy.isclose` | the same mixed test (and the same asymmetry: relative to the second argument) | `numpy/_core/numeric.py` |
| element-by-element checks | directional checks | one $u^\top J v$ per random pair costs 2 evaluations instead of $2n$; `M04.2` builds the pieces | PyTorch `fast_mode=True` |
