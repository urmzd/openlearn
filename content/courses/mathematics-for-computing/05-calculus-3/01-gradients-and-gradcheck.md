# Partial derivatives, gradients, and gradcheck

Every neural network you train is a function of many arrays, and every training step needs the derivative of a scalar loss with respect to every element of every one of them. Later, in *Build Your Own LLM System*, you will write an autograd engine and a library of about 40 differentiable ops, each with a hand-written backward rule ([Tensor, broadcasting backward, and no_grad](../../build-an-llm/01-foundations/02-tensor-and-broadcasting-backward.md) and [The op library and gradcheck_all](../../build-an-llm/01-foundations/03-op-library-and-gradcheck.md)). A backward rule is a claim: "this array is the gradient of the loss with respect to that input." Claims need a referee.

The referee is the definition of the derivative, applied one coordinate at a time: perturb one weight by $\pm\epsilon$, rerun the forward pass, divide. [Derivatives and Finite Differences](../02-calculus-1/01-derivatives-and-finite-differences.md) gave you the one-coordinate version as `central_diff`. This section turns it into the gradient of a scalar function of several arrays, and into the check that compares that numerical gradient with an analytic one. The system's `{tinyllm} gradcheck --suite all` command runs the whole op library through this check, so every backward rule you write later passes through the file you build here.

The solution lives in [`gradcheck.py`](../../../../course/solutions/python/tinyllm/num/gradcheck.py) (`numerical_grad`, `gradcheck`, and its `GradcheckReport`), with the typed interface in [`course/contracts/py/tinyllm/num/gradcheck.pyi`](../../../../course/contracts/py/tinyllm/num/gradcheck.pyi). It makes one call to `central_diff` from [`diff.py`](../../../../course/solutions/python/tinyllm/num/diff.py) per coordinate, and nothing else.

> **Learning objectives**
> - Compute partial derivatives and the gradient of a scalar function of several arrays, arranged in each input's shape.
> - Build a numerical gradient from one central difference per element, and explain its cost ($2n$ evaluations) and why it is exact on quadratics.
> - Apply the mixed tolerance $|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n|$ and explain why neither a pure absolute nor a pure relative test works.
> - Write a gradient check that rejects real backward-pass bugs (a transposed gradient, a factor of 2, one forgotten element, a NaN) and reports the element furthest past its tolerance.
> - Explain why the check perturbs float64 copies rather than the caller's arrays.

## Notation

The section uses a small set of symbols throughout; this table pairs each one with the Python type it becomes.

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

## Partial derivatives

Start with a function of two numbers, $f(x, y) = x^2 y + 3y$. Freeze $y$ and differentiate in $x$: $\frac{\partial f}{\partial x} = 2xy$. Freeze $x$ and differentiate in $y$: $\frac{\partial f}{\partial y} = x^2 + 3$. Formally,

$$\frac{\partial f}{\partial x}(x, y) = \lim_{h \to 0}\frac{f(x + h, y) - f(x, y)}{h},$$

which is just the one-variable derivative from [Calculus 1](../02-calculus-1/01-derivatives-and-finite-differences.md), taken along the $x$ direction. A partial derivative measures how $f$ responds to one input when nothing else moves, and that is exactly the question a training step asks of each weight.

## The gradient

The **gradient** lists every partial derivative: $\nabla f(x, y) = (2xy,\; x^2 + 3)$. For a function of arrays there is one partial derivative per element, and the gradient with respect to $X_k$ is arranged in $X_k$'s shape, so that a training step is simply $X_k \leftarrow X_k - \eta\,\nabla_{X_k} f$.

Two facts make the gradient the right object. First, for a small change $\Delta$ in the inputs, $f$ changes by about $\sum_{k,i} \frac{\partial f}{\partial X_k[i]}\,\Delta_k[i]$, the dot product of the gradient with $\Delta$. That is a first-order Taylor expansion, as in [Taylor series, remainder bounds, and range reduction](../03-calculus-2/01-taylor-series.md). Second, among all directions of length 1 that dot product is largest along the gradient, so $-\nabla f$ is the direction of steepest descent, which [gradient descent](../11-optimization/01-gradient-descent-and-line-search.md) exploits. Note that a gradient is only defined for a **scalar** $f$. The derivative of a vector-valued function is a matrix, the Jacobian, which is the subject of [the next section](02-jacobians-and-the-chain-rule.md).

## Central differences, one coordinate at a time

The numerical gradient follows directly from the definition. For each input $k$ and each element $i$,

$$n_k[i] = \frac{f(\ldots, X_k + \epsilon e_i, \ldots) - f(\ldots, X_k - \epsilon e_i, \ldots)}{2\epsilon}.$$

This is `central_diff` applied to the one-variable function $t \mapsto f(\ldots, X_k \text{ with element } i \text{ set to } t, \ldots)$ at $t = X_k[i]$, with step $\epsilon$. That is exactly how `numerical_grad` is written: it never re-derives the formula, it calls `central_diff`. The error analysis carries over too. Truncation error is $\frac{1}{6}|f'''|\epsilon^2$ and rounding error about $\varepsilon |f| / \epsilon$; with $\epsilon = 10^{-6}$ and values of size 1, both are near $10^{-10}$, comfortably under the tolerances you will see shortly. On a quadratic, $f(x) = \frac12 x^\top A x + b^\top x$ with gradient $Ax + b$, the third derivative is zero and the numerical gradient is exact up to rounding.

Three details decide whether this is trustworthy. **Copies**: perturb float64 copies of the inputs, never the caller's arrays, because those are the model's parameters. **Restore**: put each element back before moving to the next, or every later partial derivative is taken at the wrong point. **Promote**: an integer array cannot hold $x + 10^{-6}$ (it rounds back to $x$, and the gradient comes out 0), and float32 moves $x$ only in steps of about $6 \times 10^{-8}|x|$, so the step actually taken is not $\epsilon$.

The cost is $2n$ evaluations of $f$ for $n$ input elements. That is fine for tests on small inputs and hopeless for training, which is why backpropagation exists.

## When is a gradient right?

Having a numerical gradient, you need a rule for when an analytic gradient agrees with it. Compare elementwise, with a mixed test:

$$|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n| .$$

Near zero the absolute term dominates: with atol $= 10^{-7}$, a numerical gradient of $10^{-9}$ is indistinguishable from 0. For large gradients the relative term does: $10^{-5}|n|$ gives an allowance of $10^{-3}$ at $n = 100$. Each pure test fails somewhere. A pure relative test would reject a correct analytic 0 against a numerical $10^{-11}$. A pure absolute test would accept a gradient of 100 that is off by 0.5 percent at a loose atol, or reject a correct one at a tight atol.

A **NaN** in either gradient must fail. Every comparison with NaN is false, so the natural `diff > tol` would let it pass; test `not (diff <= tol)` instead.

The report says `ok`, the largest absolute error, the largest relative error $|a - n| / \max(|n|, \mathrm{atol})$, and **where** to look: the input and multi-index of the element whose error is the largest multiple of its own allowance. That is not the same as the largest absolute error. An error of $5 \times 10^{-3}$ on a gradient of 1000 is within tolerance; an error of $2 \times 10^{-4}$ on a gradient of $10^{-3}$ is a bug.

## Working it by hand

Take $f(x) = x_0^2 x_1 + 3 x_1$ at $x = (1, 2)$, with the large step $\epsilon = 0.1$ so the arithmetic is easy by hand. The gradient is $(2 x_0 x_1, x_0^2 + 3) = (4, 4)$. The table shows both perturbations for each element:

| Element | $f(x + \epsilon e_i)$ | $f(x - \epsilon e_i)$ | quotient | analytic |
|---|---|---|---|---|
| $x_0$ | $f(1.1, 2) = 1.21 \cdot 2 + 6 = 8.42$ | $f(0.9, 2) = 0.81 \cdot 2 + 6 = 7.62$ | $0.8 / 0.2 = 4$ | 4 |
| $x_1$ | $f(1, 2.1) = 2.1 + 6.3 = 8.4$ | $f(1, 1.9) = 1.9 + 5.7 = 7.6$ | $0.8 / 0.2 = 4$ | 4 |

Both are exact even at $\epsilon = 0.1$: $f$ is quadratic in $x_0$ and linear in $x_1$, so the central difference has no truncation error. Notice the restore at work: the $x_1$ row is evaluated at $x_0 = 1$, not at the 0.9 left over from the row before. Now suppose a backward pass returned $(2, 4)$, losing a factor of 2 on $x_0$. It fails at element 0, because $|2 - 4| = 2 > 10^{-7} + 10^{-5} \cdot 4$.

## Designing the interface

The whole module is one dataclass and two functions:

```python
@dataclass
class GradcheckReport: ok: bool; max_abs_err: float; max_rel_err: float; worst_input: int; worst_index: tuple
def numerical_grad(f, inputs: list[ArrayLike], eps: float = 1e-6) -> list[NDArray]: ...
def gradcheck(f, inputs: list[ArrayLike], analytic: list[ArrayLike],
              eps: float = 1e-6, rtol: float = 1e-5, atol: float = 1e-7) -> GradcheckReport: ...
```

`numerical_grad` returns one float64 array per input, in that input's shape. `gradcheck` raises `ValueError` for mismatched lists or shapes, because those are bugs in the caller rather than numeric disagreements, and reports everything else. `f` must return a single number.

A correct solution has to handle a specific set of cases, each tied to something that matters downstream. They are compact enough to list in one table:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example | unit | the worked example: $(4, 4)$ at $\epsilon = 0.1$, `ok` | pins the definition |
| Quadratics | property | numerical gradient of $\frac12 x^\top A x + b^\top x$ equals $Ax + b$ to $10^{-8}$ | central differences are exact there |
| Matrix and several inputs | golden | $\nabla$ of $\mathrm{sum}(\tanh(WX))$ is $G X^\top$ and $W^\top G$ | layers have weights and inputs |
| Transposed gradient | unit | a transposed weight gradient is not ok, worst input 0 | the most common backward bug |
| Gradient off by 2 | unit | $x$ instead of $2x$ is rejected, worst at the largest element, relative error 0.5 | a lost factor keeps every sign right |
| One zeroed element | unit | one zeroed element of the second input is found at $(2, 1)$ | the report says where to look |
| Mixed tolerance | boundary | the mixed test at gradients $10^{-3}$ and 100 | both regimes of the tolerance |
| Worst element | unit | the worst element is the one furthest past its allowance | not the largest absolute error |
| NaN gradient | boundary | a NaN analytic gradient is not ok | NaN comparisons are always false |
| Caller's arrays | unit | they come back bit for bit | they are the model's parameters |
| Integer and float32 inputs | boundary | exact gradients; $f$ sees float64 | steps survive the dtype |
| Defaults | unit | step $10^{-6}$, rtol $10^{-5}$, atol $10^{-7}$ | what the op library relies on |
| Mismatched arguments | boundary | wrong counts and shapes raise `ValueError` | caller bugs are not numeric errors |
| Non-scalar $f$ | boundary | a vector-valued $f$ raises; a 1-element array is fine | gradients are for scalars |

## Building it step by step

The design falls out of the principles above in six steps, and it is worth predicting each one before reading it.

1. A partial derivative is a one-variable derivative with everything else frozen. So the numerical gradient is `central_diff` applied to a one-variable function that writes $t$ into one element and calls $f$: no new difference formula, just a loop over elements.
2. The loop must not touch the caller's arrays and must take steps that are representable, which forces float64 copies made once at the start.
3. Each element is perturbed in place in the copy, and must be restored before the next one, even if $f$ raises. A `try/finally` guarantees that.
4. Iterating over a flat view (`reshape(-1)` of a contiguous copy) visits every element of any shape, and writing into a flat view of a zero array of the same shape produces the gradient already in the input's shape.
5. The comparison is $|a - n| \le \mathrm{atol} + \mathrm{rtol}|n|$ elementwise. To make NaN fail, the code tests the negation of "passes" rather than "exceeds".
6. The location to report is the element with the largest ratio of error to allowance, across all inputs, with ties going to the first in row-major order and the first input.

Here are the scalar guard and the numerical gradient:

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

`central_diff` evaluates `along(x + eps)` and `along(x - eps)` and divides by `(x + eps) - (x - eps)`, the step actually representable in float64, rather than by $2\epsilon$. On the worked example it returns $0.8/0.2 = 4$ for both elements.

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

The report then carries `ok`, the two maxima, and `worst_input`, `worst_index`. The full solution is in [`gradcheck.py`](../../../../course/solutions/python/tinyllm/num/gradcheck.py).

### Why this design

The cost is $2n$ evaluations of $f$ for $n$ input elements and $O(n)$ extra memory for the copies and gradients. That is the price of checking every element, and it is acceptable because checks run on small inputs.

Central rather than forward differences is not a matter of taste. A forward difference has error $O(\epsilon)$, about $10^{-6}$ at the default step, which is above `atol` and would make correct gradients fail. Central differences have error $O(\epsilon^2)$ and are exact on quadratics.

The worst element is ranked by its ratio to the allowance because that ratio is what decides pass or fail, so the element with the largest ratio is the one a person should look at. The largest absolute error is often a correct large gradient.

You could instead check $u^\top J v$ for random $u, v$, which costs 2 evaluations instead of $2n$, but it cannot say which element is wrong. Element-by-element is the right default for debugging backward rules; the directional form appears under further reading below.

One tradeoff is accepted deliberately: the `atol` floor makes a gradient of $10^{-9}$ indistinguishable from 0. Below the finite-difference noise, the comparison has no information.

## What can go wrong

Most bugs in a gradient checker make it too lenient, which is the worst failure mode a referee can have. The common ones, and how the solution avoids each, are these:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| combining the per-element verdicts wrongly (`all` instead of `any`) | a transposed or partly zeroed gradient is reported ok | `ok` is false as soon as any element of any input fails |
| not restoring an element before the next one | every later partial derivative is taken at a shifted point | restores `flat[i] = orig` in a `finally` (step 3) |
| perturbing the caller's arrays in place | the model's weights change during a check | perturbs private copies `xs` |
| perturbing in the caller's dtype | gradient 0 for int inputs; two digits for float32 | copies with `dtype=np.float64` (step 2) |
| `diff > tol` as the failure test | a NaN gradient passes | tests `~(diff <= tol)` (step 5) |
| a one-sided difference per coordinate | errors near $10^{-6}$ instead of $10^{-10}$; quadratics no longer exact | calls `central_diff` |
| dividing by $4\epsilon$ (the $\epsilon$ versus $2\epsilon$ confusion) | every numerical gradient halved | lets `central_diff` divide by the actual step `xp - xm` |
| differentiating or comparing only the first input | the input gradient is never checked | loops over every input in both functions |
| reporting the largest absolute error as the worst | the report points at a correct large gradient | ranks by `diff / (atol + rtol * abs(n))` (step 6) |

> **Warning:** The NaN case is the easiest to miss because the buggy test looks correct. `diff > tol` reads as "fails when too large", but a NaN is never larger than anything, so a backward pass that produces NaN would sail through.

## Where this leads

This section looks back to [Derivatives and Finite Differences](../02-calculus-1/01-derivatives-and-finite-differences.md): `numerical_grad` calls `central_diff(along, x_i, eps)` for every element. Looking forward, [Jacobians, the multivariable chain rule, and numeric JVP/VJP](02-jacobians-and-the-chain-rule.md) builds `vjp_numeric(f, x, u)` as `numerical_grad` of the scalar $u \cdot f(x)$. In *Build Your Own LLM System*, [The op library and gradcheck_all](../../build-an-llm/01-foundations/03-op-library-and-gradcheck.md) uses `F.gradcheck_all()` to check every op's backward with this `gradcheck`, behind the system's `{tinyllm} gradcheck --suite all` command, and the model test suites from [Encoder-decoder with teacher forcing](../../build-an-llm/05-attention-origins/01-encoder-decoder-teacher-forcing.md) through [Llama-family model](../../build-an-llm/08-modern-block/09-llama-family-model.md) gradcheck every new backward rule. Back in this book, [gradient descent](../11-optimization/01-gradient-descent-and-line-search.md) steps along $-\nabla f$.

### Further reading

- PyTorch `torch.autograd.gradcheck` and `gradgradcheck` (`torch/autograd/gradcheck.py`): the production equivalent of `gradcheck`, adding complex inputs, sparse and batched layouts, second derivatives, and a fast mode (`fast_mode=True`) that checks random projections $u^\top J v$ instead of every element, at 2 evaluations per random pair instead of $2n$. The next section builds the pieces.
- JAX `jax.test_util.check_grads` (`jax/_src/public_test_util.py`): checks forward and reverse mode up to a chosen order.
- `numpy.isclose` (`numpy/_core/numeric.py`): the same mixed test `atol + rtol * abs(n)`, with the same asymmetry (relative to the second argument).
- OpenStax, *Calculus Volume 3* (free), sections 4.3 and 4.6 on partial derivatives and the gradient.
- Nocedal and Wright, *Numerical Optimization*, section 8.1, on finite-difference gradients.
- The PyTorch autograd notes on `gradcheck`.

## Key terms

- **Partial derivative**: the ordinary derivative of $f$ along one coordinate with every other coordinate frozen.
- **Gradient**: the collection of all partial derivatives of a scalar function, one per input element, arranged in the input's shape.
- **Numerical gradient**: an approximation of the gradient by one central difference per element, costing $2n$ evaluations of $f$.
- **Mixed tolerance**: the pass rule $|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n|$, absolute near zero and relative for large values.
- **Gradient check**: a comparison of an analytic gradient against the numerical one that reports pass or fail and the element furthest past its allowance.

## Summary

- A **partial derivative** $\partial f / \partial x_i$ is an ordinary derivative along one coordinate with every other coordinate frozen; the **gradient** $\nabla f$ collects one per input element, in the input's shape.
- A numerical gradient is one central difference per element, $2n$ evaluations of $f$ for $n$ elements, exact on quadratics.
- An element passes when $|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n|$: absolute near zero, relative for large gradients.
- `gradcheck` must reject the bugs backward passes really have (a transposed gradient, a factor of 2, one forgotten element, a NaN) and point at the element furthest past its tolerance.
- It perturbs float64 **copies**, so the caller's parameters are untouched and integer or float32 inputs still get exact steps.

## Review questions

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

5. Why would forward differences make correct gradients fail at the default tolerances?

   <details><summary>Answer</summary>

   A forward difference has truncation error $O(\epsilon)$, about $10^{-6}$ at $\epsilon = 10^{-6}$, which exceeds atol $= 10^{-7}$ for small gradients. Central differences have error $O(\epsilon^2)$, near $10^{-10}$ after rounding, well inside the tolerance.

   </details>
