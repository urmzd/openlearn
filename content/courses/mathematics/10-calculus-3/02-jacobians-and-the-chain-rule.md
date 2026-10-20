# Jacobians, the multivariable chain rule, and numeric JVP/VJP

[The previous section](01-gradients-and-gradcheck.md) checked the gradient of a scalar loss. But an autodiff engine never handles the loss as one formula. It handles a chain of layers, each mapping arrays to arrays, and its backward pass multiplies through them one at a time. The objects being multiplied are Jacobians, or rather products with Jacobians, because the matrices themselves are far too large to form: a layer from 4096 to 4096 numbers has a Jacobian of 16 million entries per token.

Chapter 8 of this book builds automatic differentiation three ways: [dual numbers](../14-matrix-calculus-and-autodiff/01-dual-numbers-forward-mode.md) (forward mode), [a scalar `Value` graph](../14-matrix-calculus-and-autodiff/02-scalar-reverse-mode.md) (reverse mode), and [closed-form VJPs of matrix expressions](../14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md). Each needs an independent numeric oracle for the products $Jv$ and $u^\top J$. This section defines the Jacobian and the chain rule from partial derivatives and builds those oracles.

You will need `numerical_grad` from the previous section, on which `vjp_numeric` is built, and it helps to have [Vectors, matrices, row-major layout, and matmul](../09-linear-algebra/01-vectors-matrices-and-matmul.md) fresh for matrices and their product. The solution is [`jacobian.py`](../../../../course/solutions/python/tinyllm/num/jacobian.py), with `jacobian`, `jvp_numeric` ($Jv$), and `vjp_numeric` ($u^\top J$); the typed interface is [`course/contracts/py/tinyllm/num/jacobian.pyi`](../../../../course/contracts/py/tinyllm/num/jacobian.pyi).

> **Learning objectives**
> - Define the Jacobian of $f: \mathbb{R}^n \to \mathbb{R}^m$, compute it by hand and numerically, and read its rows and columns.
> - State the multivariable chain rule as a matrix product and explain the order of the factors.
> - Compute a JVP and a VJP without forming $J$, and explain which one forward and reverse mode compute.
> - Use the identity $u^\top(Jv) = (u^\top J)v$ as a consistency check.
> - Explain why the JVP step must scale with the size of $v$.

## Notation

These are the symbols the section uses, with the Python shapes they correspond to:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f: \mathbb{R}^n \to \mathbb{R}^m$ | a function from $n$ numbers to $m$ numbers | `Callable[[NDArray], ArrayLike]` |
| $x$ | the input; any shape, flattened row-major to $n$ numbers | `float64[...]` |
| $f_i$ | output $i$ of $f$ (row-major order of $f(x)$) | `float` |
| $J = J_f(x)$ | the Jacobian, $J_{ij} = \partial f_i / \partial x_j$ | `float64[m, n]` |
| $v \in \mathbb{R}^n$ | a **tangent**: a direction to move the input | shape of $x$ |
| $u \in \mathbb{R}^m$ | a **cotangent**: a weighting of the outputs | shape of $f(x)$ |
| $Jv$ | the Jacobian-vector product (JVP) | shape of $f(x)$ |
| $u^\top J$ | the vector-Jacobian product (VJP) | shape of $x$ |
| $\epsilon$, $h$ | finite-difference steps | `float` |

## The Jacobian

Each output $f_i$ is a scalar function of $x$, so it has a gradient. Stack those gradients as rows and you get the **Jacobian**:

$$J = \begin{pmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & & \vdots \\ \partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n \end{pmatrix} .$$

It is the best **linear approximation** of $f$ near $x$: $f(x + \Delta) \approx f(x) + J\Delta$ for small $\Delta$, the several-variable version of the tangent line. Row $i$ is the gradient of output $i$. Column $j$ is how the whole output moves per unit move of input $j$, which is also how to compute it numerically:

$$J_{:, j} \approx \frac{f(x + \epsilon e_j) - f(x - \epsilon e_j)}{(x_j + \epsilon) - (x_j - \epsilon)} ,$$

which costs $2n$ evaluations of $f$ and divides by the step actually taken, as in [Derivatives and Finite Differences](../07-calculus-1/01-derivatives-and-finite-differences.md).

Three examples recur throughout the book. A linear map $f(x) = Ax$ has $J = A$ everywhere. An elementwise map like $\tanh$ has a diagonal Jacobian, $\operatorname{diag}(1 - \tanh^2 x)$, because output $i$ depends only on input $i$. Softmax, $p_i = e^{z_i}/\sum_k e^{z_k}$, couples everything: $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$, so $J = \operatorname{diag}(p) - pp^\top$, symmetric with columns summing to 0 (the probabilities always add to 1). [Matrix differentials, the trace trick, and closed-form VJPs](../14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) derives that one by hand.

When $x$ or $f(x)$ is a matrix, flatten both in row-major order, as in [the linear algebra chapter](../09-linear-algebra/01-vectors-matrices-and-matmul.md): the derivative with respect to $X[1, 2]$ of a $2 \times 3$ input is column $1 \cdot 3 + 2 = 5$.

## The chain rule

With the Jacobian in hand, composition becomes multiplication. If $y = f(x)$ and $z = g(y)$, then a small change $\Delta$ in $x$ moves $y$ by about $J_f \Delta$, which moves $z$ by about $J_g (J_f \Delta)$. So

$$J_{g \circ f}(x) = J_g\bigl(f(x)\bigr)\, J_f(x) ,$$

a product of a $p \times m$ and an $m \times n$ matrix, with the outer function's Jacobian on the left, evaluated at the inner function's output. Entry by entry this is $\frac{\partial z_k}{\partial x_j} = \sum_i \frac{\partial z_k}{\partial y_i}\frac{\partial y_i}{\partial x_j}$: add up every path from $x_j$ to $z_k$. The one-variable chain rule is the $1 \times 1$ case. A deep network is a long composition, and its Jacobian is a long product.

## Products without the matrix

Forming $J$ costs $n$ forward passes and $mn$ memory. Two products avoid it.

The **JVP**, $Jv$, is the derivative of $t \mapsto f(x + tv)$ at $t = 0$: the rate at which the output moves when the input moves in direction $v$. One central difference computes it, $\frac{f(x + hv) - f(x - hv)}{2h}$. Forward-mode autodiff ([Dual numbers and forward-mode autodiff](../14-matrix-calculus-and-autodiff/01-dual-numbers-forward-mode.md)) computes it exactly by carrying a tangent alongside every value. Along a chain, $J_g J_f v = J_g (J_f v)$: you push $v$ through one layer at a time.

The **VJP**, $u^\top J$, is the gradient of the scalar $u \cdot f(x) = \sum_i u_i f_i(x)$, since $\frac{\partial}{\partial x_j}\sum_i u_i f_i = \sum_i u_i J_{ij}$. `vjp_numeric` computes it with the previous section's `numerical_grad`. Reverse-mode autodiff, which is backpropagation ([Scalar reverse mode: the Value graph](../14-matrix-calculus-and-autodiff/02-scalar-reverse-mode.md), and [Tensor, broadcasting backward, and no_grad](../../ai-engineering/06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md) in *AI Engineering*), computes it exactly by pulling $u$ back one layer at a time: $u^\top J_g J_f = (u^\top J_g) J_f$. With $u = 1$ and a scalar loss, the VJP is the gradient, and that is why training uses reverse mode.

Both products describe the same $J$, so for every $u$ and $v$, $u^\top(Jv) = (u^\top J)v$: one number computed two ways, which makes a good check at random $u$, $v$, $x$. Mixing them up (returning $Ju$ for a VJP) gives the right shape whenever $J$ is square and the wrong numbers unless it is symmetric.

One practical detail matters for the JVP: **scale the step to $v$.** $h$ multiplies $v$, so the input moves by $h\max|v|$. A fixed $h = 10^{-6}$ with $|v| = 10^6$ moves $x$ by 1 and measures a secant over a whole unit. Choosing $h = 10^{-6}/\max|v|$ keeps the largest coordinate's move at $10^{-6}$; then $J(cv) = c\,Jv$ to rounding, and $v = 0$ returns 0 without computing $0/0$.

## Working it by hand

Take $f(x, y) = (x^2 y,\; 5x + \sin y)$ at $(1, 2)$. The partial derivatives give

$$J = \begin{pmatrix} \partial_x(x^2 y) & \partial_y(x^2 y) \\ \partial_x(5x + \sin y) & \partial_y(5x + \sin y) \end{pmatrix} = \begin{pmatrix} 2xy & x^2 \\ 5 & \cos y \end{pmatrix} = \begin{pmatrix} 4 & 1 \\ 5 & -0.4161468 \end{pmatrix} .$$

With $v = (1, 0)$, $Jv = (4, 5)$, the first column: moving $x$ moves the outputs at rates 4 and 5. With $u = (1, 0)$, $u^\top J = (4, 1)$, the first row: the gradient of the first output. $J$ is not symmetric, so the two differ, and a transposed Jacobian would swap them.

Now check the identity with $u = (1, 1)$ and $v = (0, 1)$. On one side, $Jv = (1, -0.4161468)$ and $u \cdot Jv = 0.5838532$. On the other, $u^\top J = (9, 0.5838532)$ and $(u^\top J) \cdot v = 0.5838532$. They agree.

## Designing the interface

Three functions cover it:

```python
def jacobian(f, x: ArrayLike, eps: float = 1e-6) -> NDArray: ...   # [f(x).size, x.size]
def jvp_numeric(f, x: ArrayLike, v: ArrayLike) -> NDArray: ...      # J v, shape f(x).shape
def vjp_numeric(f, x: ArrayLike, u: ArrayLike) -> NDArray: ...      # u^T J, shape x.shape
```

All arithmetic is float64 on a private copy of `x`. `jacobian` raises `ValueError` for a bad `eps` or when `f`'s output shape changes; the products raise it when `v` does not have `x`'s shape or `u` does not have `f(x)`'s.

The cases a correct solution must handle, and why each matters later, are these:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example, $J$ | unit | the worked example's $J$ | pins rows and columns |
| Hand example, products | unit | $Jv = (4, 5)$ and $u^\top J = (4, 1)$ | forward and reverse mode are different products |
| Matrix-shaped $x$ | unit | $x$ of shape $(2, 3)$ gives $J$ of shape $(4, 6)$ in row-major columns; product shapes; shape errors | backward passes return gradients in the input's shape |
| Linear map | golden | $J_{Ax} = A$ | matmul is every layer's first step |
| Softmax | golden | $\operatorname{diag}(p) - pp^\top$, columns sum to 0 | chapter 8 derives this VJP by hand |
| Elementwise map | golden | $J_{\tanh} = \operatorname{diag}(1 - \tanh^2)$; its VJP is elementwise | activation backward passes ([Activation Functions and Their Derivatives](../07-calculus-1/03-activation-functions.md)) |
| Composition | differential | $J_{g \circ f} = J_g(f(x)) J_f(x)$ | the rule backpropagation applies |
| Random $u$, $v$, $x$ | property | $Jv$, $u^\top J$, and $u^\top(Jv) = (u^\top J)v$ | forward and reverse oracles agree |
| Tangent size | boundary | $J(10^6 v) = 10^6 Jv$, $J(10^{-6}v) = 10^{-6}Jv$, $J0 = 0$ | tangents of any size |
| Caller's $x$ and dtype | unit | $x$ is untouched; $f$ sees float64 even for int $x$ | parameters are not perturbed in place |
| Bad `eps`, changing shapes | boundary | `eps = 0` and an $f$ whose output shape changes raise | no ragged Jacobians |

## Building it step by step

The implementation follows from the principles in five steps.

1. Column $j$ of $J$ is the rate of change of every output with input $j$. So `jacobian` loops over input elements, perturbs one at a time by $\pm\epsilon$, and stores the central difference of the whole output as column $j$. Stacking columns along `axis=1` gives the $m \times n$ orientation; stacking them as rows would give $J^\top$.
2. Inputs of any shape are flattened row-major. The flattening must be a **view** of a private float64 copy, so that writing `flat[j]` actually changes what $f$ sees, and the caller's array is never touched. Each coordinate is restored before the next.
3. The divisor is the step actually taken, `(x_j + eps) - (x_j - eps)`.
4. A JVP is the derivative of $t \mapsto f(x + tv)$ at 0: one central difference along $v$, with $h = 10^{-6}/\max|v|$ so the input moves by $10^{-6}$ whatever $v$'s size. $v = 0$ returns 0 directly.
5. A VJP is the gradient of the scalar $u \cdot f(x)$, so `vjp_numeric` is one call to `numerical_grad` on that scalar: no new differencing code, and the result is already in $x$'s shape.

The Jacobian is built one column per input element:

```python
def jacobian(f, x, eps=1e-6):
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    xw = np.array(x, dtype=np.float64, copy=True)   # private float64 copy
    flat = xw.reshape(-1)  # a view: writing flat[j] moves xw
    shape = None
    cols = []
    for j in range(flat.size):
        orig = flat[j]
        hi_x, lo_x = orig + eps, orig - eps
        flat[j] = hi_x
        hi = _eval(f, xw)
        flat[j] = lo_x
        lo = _eval(f, xw)
        flat[j] = orig                               # restore before the next column
        if shape is None:
            shape = hi.shape
        if hi.shape != shape or lo.shape != shape:  # no ragged Jacobians
            raise ValueError(
                f"f returned shapes {hi.shape} and {lo.shape}, earlier {shape}"
            )
        # Column j: how every output moves when input j moves (hi minus lo,
        # divided by the step actually taken).
        cols.append((hi - lo).reshape(-1) / (hi_x - lo_x))
    if not cols:
        m = _eval(f, xw).size
        return np.zeros((m, 0))                      # an empty input still has m rows
    return np.stack(cols, axis=1)                    # columns, so J is [m, n]
```

On the worked example, column 0 is the change in $(x^2 y, 5x + \sin y)$ when $x$ moves, $(4, 5)$, and column 1 is $(1, \cos 2)$.

The JVP scales its step to $v$:

```python
JVP_STEP = 1e-6

def jvp_numeric(f, x, v):
    xw = np.array(x, dtype=np.float64, copy=True)
    va = np.asarray(v, dtype=np.float64)
    if va.shape != xw.shape:
        raise ValueError(f"v has shape {va.shape}, x has {xw.shape}")
    scale = float(np.max(np.abs(va))) if va.size else 0.0
    if scale == 0.0:
        return np.zeros_like(_eval(f, xw))           # J 0 = 0, no 0/0
    # The largest coordinate of x moves by JVP_STEP whatever the size of v.
    h = JVP_STEP / scale
    return (_eval(f, xw + h * va) - _eval(f, xw - h * va)) / (2.0 * h)
```

And the VJP is a gradient of a scalar:

```python
def vjp_numeric(f, x, u):
    xw = np.array(x, dtype=np.float64, copy=True)
    ua = np.asarray(u, dtype=np.float64)
    fx = _eval(f, xw)
    if ua.shape != fx.shape:
        raise ValueError(f"u has shape {ua.shape}, f(x) has {fx.shape}")
    # u^T J is the gradient of the scalar u . f(x): one gradient, not m rows.
    return numerical_grad(lambda z: float(np.sum(ua * _eval(f, z))), [xw])[0]
```

With $u = (1, 0)$ on the worked example, the scalar is $x^2 y$ and its gradient is $(4, 1)$, the first row of $J$. The full solution is in [`jacobian.py`](../../../../course/solutions/python/tinyllm/num/jacobian.py).

### Why this design

On cost: `jacobian` takes $2n$ evaluations and $O(mn)$ memory. `jvp_numeric` takes 2 evaluations whatever $n$ and $m$. `vjp_numeric` takes $2n + 1$ (one to check `u`'s shape, then `numerical_grad`'s $2n$), but stores only $O(n)$.

You could compute the VJP as `u @ jacobian(f, x)`. It gives the same numbers but stores the whole $m \times n$ matrix. Building it on `numerical_grad` also makes it the same code that `gradcheck` already trusts.

Central differences are needed because one-sided differences have $O(\epsilon)$ error, about $10^{-6}$, too coarse to confirm closed forms such as softmax's Jacobian. And the JVP step is scaled because a finite difference approximates the derivative only if the input moves a small distance; a fixed $h$ makes the distance proportional to $\lvert v\rvert$.

The tradeoff accepted is that these are oracles, not autodiff: their accuracy is about $10^{-10}$ on well-scaled functions. That is enough to confirm the exact forward- and reverse-mode results of chapter 8.

## What can go wrong

Most of the bugs here produce arrays of the right shape with the wrong numbers, which is why they survive casual testing:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the Jacobian transposed (columns stacked as rows) | $J^\top$: right shape for square $J$, wrong numbers, and a chain rule product in the wrong order | stacks columns with `axis=1` (step 1) |
| perturbing the caller's $x$ and not restoring it | the input array changes during a check | works on `np.array(x, dtype=np.float64, copy=True)` |
| leaving coordinate $j$ at $x_j - \epsilon$ | every later column is taken at a shifted point | sets `flat[j] = orig` after both evaluations |
| a fixed JVP step whatever the size of $v$ | $J(10^6 v)$ is a secant over a unit step | `h = JVP_STEP / max|v|` (step 4) |
| a VJP that computes $Ju$ instead of $u^\top J$ | the column $(4, 5)$ where the row $(4, 1)$ was wanted | differentiates the scalar $u \cdot f(x)$ (step 5) |
| one-sided differences for the columns | errors near $10^{-6}$; softmax's closed form missed | evaluates both $x_j \pm \epsilon$ |
| dividing the JVP by $h$ instead of $2h$ | every JVP doubled | divides by `2.0 * h` |
| flattening $x$ through a copy | the perturbations never reach $f$: $J = 0$ | uses `xw.reshape(-1)`, a view of the copy |
| the column difference reversed | $-J$ | computes `hi - lo` |
| a VJP that ignores $u$ | the gradient of $\sum_i f_i$ for every $u$ | multiplies by `ua` inside the scalar |

> **Warning:** `xw.flatten()` and `xw.reshape(-1)` look interchangeable, but `flatten` always copies. Perturbing a copy never reaches $f$, and the Jacobian silently comes out all zeros.

## Where this leads

Looking back, `vjp_numeric` is `numerical_grad(lambda z: sum(u * f(z)), [x])` from [the previous section](01-gradients-and-gradcheck.md). Looking forward, the oracles built here referee everything in chapter 8: [Dual numbers and forward-mode autodiff](../14-matrix-calculus-and-autodiff/01-dual-numbers-forward-mode.md) computes $Jv$ exactly and its tests compare against `jvp_numeric`; [Scalar reverse mode: the Value graph](../14-matrix-calculus-and-autodiff/02-scalar-reverse-mode.md) computes $u^\top J$ by backpropagation, checked against `vjp_numeric`; and [Matrix differentials, the trace trick, and closed-form VJPs](../14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) derives closed-form VJPs of matmul, softmax, LayerNorm, RMSNorm, and cross-entropy, checked against `jacobian` and `vjp_numeric`. In *AI Engineering*, [Vanilla RNN with manual BPTT and truncation](../../ai-engineering/09-recurrent-networks/01-vanilla-rnn-manual-bptt.md) does manual backpropagation through time, a chain of Jacobians over time steps.

### Further reading

- JAX `jax.jvp` and `jax.vjp` (`jax/_src/api.py`): exact products by forward and reverse mode, composable with `jit` and `vmap`.
- PyTorch `torch.autograd.functional.jacobian` with `vectorize=True` (`torch/autograd/functional.py`), and JAX `jacfwd` and `jacrev`: the full Jacobian from $n$ JVPs or $m$ VJPs, batched. Pick forward when $n < m$, reverse when $m < n$.
- Reverse-mode autodiff in production does one VJP per layer, in reverse order, storing the forward values it needs; see [Tensor, broadcasting backward, and no_grad](../../ai-engineering/06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md) and Baydin et al. below.
- PyTorch `gradcheck(fast_mode=True)` (`torch/autograd/gradcheck.py`) uses $u^\top(Jv) = (u^\top J)v$ to check a whole backward with one random $u$ and $v$ instead of $mn$ entries.
- OpenStax, *Calculus Volume 3* (free), section 4.5, on the chain rule for several variables.
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2018), sections 3.1 and 3.2, on forward and reverse mode.

## Key terms

- **Jacobian**: the $m \times n$ matrix of partial derivatives $J_{ij} = \partial f_i / \partial x_j$ of a function $f: \mathbb{R}^n \to \mathbb{R}^m$.
- **Tangent**: a direction $v$ in input space along which the input is moved.
- **Cotangent**: a weighting $u$ of the outputs, pulled back through a function by a VJP.
- **JVP (Jacobian-vector product)**: $Jv$, the rate of change of the output along $v$; what forward mode computes.
- **VJP (vector-Jacobian product)**: $u^\top J$, the gradient of $u \cdot f(x)$; what reverse mode computes.
- **Chain rule**: $J_{g \circ f}(x) = J_g(f(x))\,J_f(x)$.

## Summary

- For $f: \mathbb{R}^n \to \mathbb{R}^m$ the **Jacobian** $J$ is the $m \times n$ matrix of partial derivatives, $J_{ij} = \partial f_i / \partial x_j$: row $i$ is the gradient of output $i$, column $j$ is how every output moves with input $j$.
- The **chain rule** in several variables is a matrix product, outer function on the left: $J_{g \circ f}(x) = J_g(f(x))\,J_f(x)$.
- Autodiff never forms $J$. **Forward mode** pushes a direction through, $Jv$; **reverse mode** pulls a cotangent back, $u^\top J$, which is the gradient of the scalar $u \cdot f(x)$.
- The two products are consistent: $u^\top(Jv) = (u^\top J)v$ for every $u$ and $v$.
- The JVP step must scale with $v$: $h = 10^{-6}/\max|v|$, so the input moves by $10^{-6}$ whatever the size of $v$.

## Review questions

1. Why does `jacobian` stack the columns with `axis=1` and not `axis=0`?

   <details><summary>Answer</summary>

   Each computed vector is how all $m$ outputs move with one input: a column of $J$. Stacking along `axis=0` would place it as a row and return $J^\top$, which has the right shape whenever $m = n$ and wrong numbers unless $J$ is symmetric. On the worked example it would give rows $(4, 5)$ and $(1, -0.416)$.

   </details>

2. What happens if `flat` is `xw.flatten()` instead of `xw.reshape(-1)`?

   <details><summary>Answer</summary>

   `flatten` returns a copy, so writing `flat[j]` never changes `xw`, $f$ always sees the unperturbed input, `hi == lo`, and every column is 0.

   </details>

3. Why is $h = 10^{-6}/\max|v|$ and not a fixed $10^{-6}$?

   <details><summary>Answer</summary>

   The input moves by $h v$. With $|v| = 10^6$ a fixed $h$ moves $x$ by a whole unit, and the quotient is a secant over that unit, not a derivative. Dividing by $\max|v|$ keeps the largest coordinate's move at $10^{-6}$, so $J(cv) = c\,Jv$ to rounding.

   </details>

4. Why is the VJP the gradient of $u \cdot f(x)$?

   <details><summary>Answer</summary>

   $\partial(\sum_i u_i f_i)/\partial x_j = \sum_i u_i \,\partial f_i/\partial x_j = (u^\top J)_j$. So the row vector $u^\top J$ is exactly the gradient of that scalar, and one gradient computation replaces $m$ rows.

   </details>

5. What does $u^\top(Jv) = (u^\top J)v$ check that neither product checks alone?

   <details><summary>Answer</summary>

   That the forward and reverse products describe the same matrix. A VJP that returned $Ju$ or a JVP that used $J^\top$ could each look plausible, but would break the identity for generic $u$ and $v$.

   </details>
