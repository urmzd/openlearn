# Matrix differentials, the trace trick, and closed-form VJPs

## Overview

| | |
|---|---|
| **Module** | `M08.3` · build · Python · Pass 2 · about 1.5 to 2 h reading |
| **Solution** | [`vjp.py`](../../../../course/solutions/python/tinyllm/autograd/vjp.py): `unbroadcast`, `matmul_vjp`, `softmax_vjp`, `log_softmax_vjp`, `layernorm_vjp`, `rmsnorm_vjp`, `cross_entropy_vjp` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/vjp.pyi`](../../../../course/contracts/py/tinyllm/autograd/vjp.pyi) |
| **Needs** | `M09.2` stable softmax ([`stable.py`](../../../../course/solutions/python/tinyllm/num/stable.py)) · `M11.1` cross-entropy (the loss whose gradient this is) · `M04.2` numeric VJP · reading: `M03.1` matrices and matmul shapes |
| **Used by** | `L0.1` `unbroadcast` in the Tensor's backward · `L0.2` op VJPs · `L0.3` fused cross-entropy, on the way to the Pass 2 result, a bigram trained by the course's own autograd · later: `L3.1` backpropagation through time by hand, `L7.1` RMSNorm |
| **Optional depth** | Parr and Howard, [*The Matrix Calculus You Need for Deep Learning*](https://arxiv.org/abs/1802.01528); Minka, "Old and New Matrix Algebra Useful for Statistics" (the differential method); Petersen and Pedersen, *The Matrix Cookbook*, sections 2 and 4 |

## Key Takeaways

- For a scalar loss, $dL = \operatorname{tr}(G^\top dY)$ with $G = \partial L / \partial Y$; write $dY$ in terms of $dX$, move $dX$ to the right with the cyclic property of the trace, and the matrix in front of it is $\partial L / \partial X$.
- That gives every rule in a few lines: $\bar A = G B^\top$ and $\bar B = A^\top G$ for a matmul, $y \odot (g - \langle g, y\rangle)$ for softmax, $(\mathrm{softmax} - \mathrm{onehot}) / n$ for cross-entropy.
- LayerNorm and RMSNorm divide by a statistic of every input, so their VJPs carry correction terms that a "treat the statistic as a constant" derivation drops.
- A broadcast input receives the sum of its copies' gradients, and padded positions receive exactly zero.

## How to read this chapter

Sections 1 to 3 derive every rule from the differential and the trace trick and check each on small numbers by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`M08.2` gave reverse mode one scalar at a time. The bigram's forward pass is a $[T, 256] \times [256, 256]$ matmul followed by a softmax over 256 entries per row, about 16 million multiply-adds per step; spelled out as Python `Value` objects that is minutes per step instead of milliseconds. `L0.2` builds a tensor op library instead, where each op (matmul, softmax, log-softmax, LayerNorm, RMSNorm, cross-entropy) has one backward function written in numpy. Each needs a closed-form vector-Jacobian product, derived once on paper and proven by gradcheck. This module derives and implements them, so that `L0.2` only wires them into the graph, `L0.3` fuses softmax with cross-entropy, `L3.1` reuses them to backpropagate through time by hand, and `L7.1`'s RMSNorm has its backward ready.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $L$ | a scalar loss | float |
| $X, Y = f(X)$ | an op's input and output | arrays |
| $G = \bar Y = \partial L / \partial Y$ | the upstream gradient: same shape as $Y$ | array |
| $\bar X = \partial L / \partial X$ | the VJP's result: same shape as $X$ | array |
| $dX$ | a differential: an arbitrary small change of $X$ | same shape as $X$ |
| $\langle A, B\rangle = \sum_{ij} A_{ij} B_{ij} = \operatorname{tr}(A^\top B)$ | the inner product of two same-shape arrays | float |
| $\odot$ | elementwise product | |
| $\mathbf{1}$ | the vector of ones | `float[D]` |
| $D$ | the size of the normalized (last) axis | `int` |
| $\mu, \sigma^2$ | mean and variance of a row over its $D$ entries | float per row |
| $\epsilon$ | small constant added to the variance ($10^{-5}$ for LayerNorm, $10^{-6}$ for RMSNorm in this course's models) | float |
| $r$ (`rstd`) | reciprocal standard deviation, $1/\sqrt{\sigma^2 + \epsilon}$ (LayerNorm) or $1/\sqrt{\mathrm{mean}(x^2) + \epsilon}$ (RMSNorm) | float per row |
| $\hat x = (x - \mu)\, r$ | the normalized row | `float[D]` |
| $\gamma, \beta, w$ | learned scale and shift (LayerNorm), learned scale (RMSNorm) | `float[D]` |
| $y = \mathrm{softmax}(x)$, $\ell = \mathrm{logsoftmax}(x)$ | probabilities and log-probabilities of a row | `float[V]` |
| $t$, $n$ | a row's target class; the number of rows whose target is not `ignore_index` | `int` |

**Gradients have the shape of their variable.** Whatever layout convention a textbook uses, here $\bar X$ is stored with exactly the shape of $X$, and the entry $\bar X_{ij}$ is $\partial L / \partial X_{ij}$. Then the first-order change of the loss is

$$dL = \sum_{ij} \bar X_{ij}\, dX_{ij} = \langle \bar X, dX\rangle.$$

**The recipe.** For $Y = f(X)$, the chain rule says $dL = \langle G, dY\rangle$. Write $dY$ as a linear expression in $dX$, then rearrange $\langle G, dY\rangle$ into the form $\langle \text{something}, dX\rangle$. Since this holds for every $dX$, the something is $\bar X$. Two tools do the rearranging:

- the **cyclic property** of the trace: $\operatorname{tr}(ABC) = \operatorname{tr}(CAB) = \operatorname{tr}(BCA)$, and $\operatorname{tr}(A^\top) = \operatorname{tr}(A)$;
- moving factors across the inner product: $\langle A, BC\rangle = \langle B^\top A, C\rangle = \langle A C^\top, B\rangle$.

**Matmul.** $Y = AB$ with $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$. The product rule holds for matrices (keep the order): $dY = dA\, B + A\, dB$. Then

$$\langle G, dA\, B\rangle = \operatorname{tr}(G^\top dA\, B) = \operatorname{tr}(B G^\top dA) = \langle G B^\top, dA\rangle, \qquad \langle G, A\, dB\rangle = \langle A^\top G, dB\rangle,$$

so $\bar A = G B^\top$ and $\bar B = A^\top G$. Check the shapes: $G$ is $m \times n$ and $B^\top$ is $n \times k$, so $\bar A$ is $m \times k$ like $A$. Shape-checking catches many mistakes but not all: if $B$ is square, $G B$ has the right shape and the wrong values.

**Broadcasting.** numpy's matmul broadcasts batch axes: a weight $B$ of shape $[k, n]$ times activations $[T, m, k]$ is the same $B$ used $T$ times. Each use contributes a gradient, so $\bar B$ is the sum over the batch axis. In general, if the forward pass broadcast $X$ from shape $s$ to a larger shape, `unbroadcast` sums the gradient over every axis that was added in front and over every axis where $s$ has size 1 (keeping it as size 1).

**Softmax.** For one row, $y_i = e^{x_i} / \sum_j e^{x_j}$. Differentiating the quotient gives $dy_i = y_i\,(dx_i - \sum_j y_j\, dx_j)$, that is, $dy = y \odot (dx - \langle y, dx\rangle \mathbf{1})$; the Jacobian is $\mathrm{diag}(y) - y y^\top$. Then

$$\langle g, dy\rangle = \sum_i g_i y_i\, dx_i - \Big(\sum_i g_i y_i\Big)\Big(\sum_j y_j\, dx_j\Big) = \big\langle y \odot (g - \langle g, y\rangle \mathbf{1}),\; dx\big\rangle,$$

so $\bar x = y \odot (g - \langle g, y\rangle)$. The VJP needs only the saved output $y$; the diagonal term alone, $y \odot g$, is a common half-derivation.

**Log-softmax.** $\ell = x - \mathrm{LSE}(x)\mathbf{1}$ and $d\,\mathrm{LSE} = \langle y, dx\rangle$ (the gradient of log-sum-exp is softmax), so $d\ell = dx - \langle y, dx\rangle \mathbf{1}$ and

$$\bar x = g - y \sum_i g_i, \qquad y = e^{\ell}.$$

The function receives the saved log-probabilities $\ell$ and exponentiates them.

**Cross-entropy, fused.** For logits $z$ (one row per position) and targets $t$, the loss is the mean over the $n$ valid rows of $-\ell_{t}$. For a valid row the upstream gradient of $\ell$ is $g = -\tfrac1n e_t$ (a one-hot vector scaled), with $\sum_i g_i = -\tfrac1n$. Plug into the log-softmax VJP:

$$\bar z = -\tfrac1n e_t + \tfrac1n y = \frac{\mathrm{softmax}(z) - \mathrm{onehot}(t)}{n}.$$

Rows whose target is `ignore_index` (padding, $-100$ as in PyTorch) are not in the loss, so their gradient is exactly 0 and they do not count in $n$. Use `M09.2`'s softmax, so logits near $10^4$ give a finite gradient. A target outside $[0, V)$ that is not `ignore_index` is a data bug, not a class: numpy would quietly wrap $-1$ to the last column.

**LayerNorm, defined.** LayerNorm (Ba, Kiros, and Hinton, 2016) standardizes each row of features, then applies a learned scale and shift:

$$\mu = \tfrac1D \textstyle\sum_i x_i,\quad \sigma^2 = \tfrac1D \sum_i (x_i - \mu)^2,\quad r = \frac{1}{\sqrt{\sigma^2 + \epsilon}},\quad \hat x = (x - \mu)\,r,\quad y = \gamma \odot \hat x + \beta.$$

Every token's features come out with mean 0 and variance about 1 before the scale, which keeps activations in a range where training is stable (the 2017 transformer of `L5` uses it). The forward pass saves $\hat x$ and $r$.

**LayerNorm's VJP.** The parameter gradients are immediate from $y = \gamma \odot \hat x + \beta$: summing over every row, $\bar\gamma = \sum_{\text{rows}} g \odot \hat x$ and $\bar\beta = \sum_{\text{rows}} g$. For $x$, let $d = g \odot \gamma$ be the gradient reaching $\hat x$. Both $\mu$ and $r$ depend on every $x_i$:

$$d\mu = \tfrac1D \langle \mathbf{1}, dx\rangle, \qquad d\sigma^2 = \tfrac2D \langle x - \mu, dx\rangle \;(\text{because } \textstyle\sum_i (x_i - \mu) = 0), \qquad dr = -\tfrac12 r^3\, d\sigma^2.$$

So $d\hat x = r\,(dx - d\mu\,\mathbf{1}) + (x - \mu)\,dr = r\,(dx - \tfrac1D\langle\mathbf{1}, dx\rangle\mathbf{1}) - \tfrac rD\, \hat x\, \langle \hat x, dx\rangle$. Taking $\langle d, \cdot\rangle$ and moving $dx$ to the right:

$$\bar x = r\left(d - \mathrm{mean}(d)\,\mathbf{1} - \hat x\; \mathrm{mean}(d \odot \hat x)\right).$$

Three terms: the direct path, the path through the mean, and the path through the variance. Dropping either correction is "treating $\mu$ (or $r$) as a constant", and gradcheck catches it immediately. A consequence worth noticing: $\langle \mathbf{1}, \bar x\rangle = 0$, because adding a constant to $x$ does not change $\hat x$.

**RMSNorm, defined.** RMSNorm (Zhang and Sennrich, 2019) drops the mean and the shift: it divides by the root mean square,

$$r = \frac{1}{\sqrt{\tfrac1D \sum_i x_i^2 + \epsilon}}, \qquad y = w \odot x\, r.$$

It is cheaper and works as well in practice; Llama-family models (`L7.1`, SmolLM2) use it before every attention and MLP block.

**RMSNorm's VJP.** With $d = g \odot w$ and $dr = -\tfrac12 r^3 \cdot \tfrac2D \langle x, dx\rangle = -\tfrac{r^3}{D} \langle x, dx\rangle$:

$$\langle d, r\,dx + x\,dr\rangle = r\langle d, dx\rangle - \tfrac{r^3}{D}\langle d, x\rangle \langle x, dx\rangle \;\Rightarrow\; \bar x = r\left(d - x\, r^2\, \mathrm{mean}(d \odot x)\right),$$

and $\bar w = \sum_{\text{rows}} g \odot x\, r$. Here $\langle x, \bar x\rangle = 0$ when $\epsilon = 0$: scaling $x$ does not change the output.

**Pure functions.** A VJP reads the saved values and the upstream gradient and returns new arrays. Writing into its arguments (subtracting the one-hot from a softmax buffer passed in, updating `g` in place) corrupts values the graph still needs.

## 3. Worked example by hand

**Matmul.** $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$, $B = \begin{bmatrix}1&0\\2&1\end{bmatrix}$, and $G = \begin{bmatrix}1&0\\0&0\end{bmatrix}$, so $L = Y_{11} = a_{11}b_{11} + a_{12}b_{21}$. Directly: $\partial L/\partial a_{11} = b_{11} = 1$, $\partial L/\partial a_{12} = b_{21} = 2$, $\partial L/\partial b_{11} = a_{11} = 1$, $\partial L/\partial b_{21} = a_{12} = 2$, everything else 0. The formulas agree: $G B^\top = \begin{bmatrix}1&2\\0&0\end{bmatrix}$ and $A^\top G = \begin{bmatrix}1&0\\2&0\end{bmatrix}$. Using $GB$ instead gives $\begin{bmatrix}1&0\\0&0\end{bmatrix}$: the right shape, the wrong gradient.

**Softmax.** $x = [0, \ln 3]$: $e^x = [1, 3]$, $y = [1/4, 3/4]$. With $g = [1, 0]$, $\langle g, y\rangle = 1/4$, so $\bar x = [\tfrac14(1 - \tfrac14), \tfrac34(0 - \tfrac14)] = [3/16, -3/16]$. The entries sum to 0, as they must: adding a constant to $x$ does not change $y$.

**Cross-entropy.** The same logits with target 1 and $n = 1$: $\bar z = y - e_1 = [1/4, -1/4]$. The loss is $-\ln(3/4) = 0.2877$.

**LayerNorm** of $x = [1, 2, 6]$ with $\gamma = \mathbf{1}$, $\epsilon = 0$, $g = [1, 0, 0]$:

| quantity | value |
|---|---|
| $\mu$, $x - \mu$ | 3, $[-2, -1, 3]$ |
| $\sigma^2$, $r$ | $14/3$, $\sqrt{3/14} = 0.462910$ |
| $\hat x$ | $[-0.925820, -0.462910, 1.388730]$ |
| $d = g \odot \gamma$, $\mathrm{mean}(d)$ | $[1, 0, 0]$, $1/3$ |
| $\mathrm{mean}(d \odot \hat x)$ | $-0.925820 / 3 = -0.308607$ |
| $\hat x \cdot \mathrm{mean}(d \odot \hat x)$ | $[2/7, 1/7, -3/7]$ |
| $d - \mathrm{mean}(d) - $ that | $[8/21, -10/21, 2/21]$ |
| $\bar x = r \cdot$ that | $[0.176347, -0.220433, 0.044087]$ |

The entries of $\bar x$ sum to 0. The parameter gradients are $\bar\gamma = g \odot \hat x = [-0.925820, 0, 0]$ and $\bar\beta = [1, 0, 0]$.

**RMSNorm** of $x = [3, 4]$ with $w = \mathbf{1}$, $\epsilon = 0$, $g = [1, 0]$: $\mathrm{mean}(x^2) = 12.5$, $r = 0.282843$, $y = [0.848528, 1.131371]$. Then $d = [1, 0]$, $\mathrm{mean}(d \odot x) = 1.5$, $r^2 = 0.08$, $x r^2 \cdot 1.5 = [0.36, 0.48]$, and $\bar x = r\,[0.64, -0.48] = [0.181019, -0.135765]$. Check: $\langle x, \bar x\rangle \propto 3(0.64) + 4(-0.48) = 0$. And $\bar w = g \odot x\,r = [0.848528, 0]$.

All of these are the first case in section 4.

## 4. The interface

```python
# python/tinyllm/autograd/vjp.py
def unbroadcast(g, shape: tuple[int, ...]) -> NDArray
def matmul_vjp(g, A, B) -> tuple[NDArray, NDArray]               # A [..., m, k], B [..., k, n]
def softmax_vjp(g, y, axis: int = -1) -> NDArray                  # y = softmax(x)
def log_softmax_vjp(g, y, axis: int = -1) -> NDArray              # y = log_softmax(x)
def layernorm_vjp(g, xhat, rstd, gamma) -> tuple[NDArray, NDArray, NDArray]   # dx, dgamma, dbeta
def rmsnorm_vjp(g, x, rstd, w) -> tuple[NDArray, NDArray]                     # dx, dw
def cross_entropy_vjp(logits, targets, ignore_index: int = -100) -> NDArray
```

LayerNorm and RMSNorm normalize the last axis; `rstd` may be passed with shape `x.shape[:-1]` or `x.shape[:-1] + (1,)`, and the parameter gradients sum over every leading axis. `matmul_vjp` needs at least 2-D operands and unbroadcasts both gradients. The forward passes are not part of this module: `L0.2` writes them and saves `y`, `xhat`, and `rstd`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example | unit | every section 3 number | pins each rule |
| PyTorch golden values | golden | torch.autograd's gradients for all seven rules, in float64 | the framework `L0.2` is compared against |
| Unbroadcast shapes | unit | seven broadcast patterns, counted with a gradient of ones | biases and shared weights |
| Impossible shapes | boundary | shapes that could not have broadcast raise `ValueError` | a wrong-shaped gradient is a bug |
| Matmul gradcheck | gradcheck | 2-D, square, and batched products with either operand broadcast | every linear layer |
| Matmul rejects vectors | boundary | 1-D operands raise `ValueError` | vectors must be reshaped explicitly |
| Softmax and log-softmax gradcheck | gradcheck | along the last axis and axis 0 | attention weights, the loss |
| Softmax against numeric VJP | differential | against `M04.2`'s `vjp_numeric` on one row | the same $u^\top J$ two ways |
| LayerNorm gradcheck | gradcheck | $x$, $\gamma$, $\beta$ on `[2, 3, 8]`, `rstd` in both shapes | the 2017 transformer (`L5`) |
| RMSNorm gradcheck | gradcheck | $x$ and $w$, `rstd` in both shapes | `L7.1` |
| Cross-entropy gradcheck | gradcheck | the gradient of `M11.1`'s cross-entropy, with ignored rows | `L0.3`'s fused loss |
| Ignore index | boundary | ignored rows get 0, the mean counts valid rows, out-of-range targets raise | padded batches |
| Large logits | boundary | logits near $10^4$ give a finite gradient | a confident model |
| Trace identity | property | $\operatorname{tr}(G^\top dY) = \langle \bar A, dA\rangle + \langle \bar B, dB\rangle$ on 20 directions | the definition of a VJP |
| No mutation | unit | saved values and upstream gradients are unchanged | the graph reuses them |

## 5. The solution, step by step

### From first principles

1. A VJP returns $\bar X$ with the shape of $X$ such that $\langle G, dY\rangle = \langle \bar X, dX\rangle$ for every $dX$ (section 2). Each function is therefore the closed form derived there, written with whole-array numpy operations, never a Jacobian.
2. Most ops act on the last axis of a batch of rows. Reductions over that axis use `keepdims=True`, so a per-row scalar ($\langle g, y\rangle$, $\mathrm{mean}(d)$) broadcasts back over the row without reshaping.
3. Parameters ($\gamma$, $\beta$, $w$) are shared by every row, so their gradients sum over all leading axes. Batched matmul operands may be broadcast too, so both matmul gradients pass through `unbroadcast`.
4. `unbroadcast` inverts numpy's broadcasting rule: first sum away the axes added in front, then sum (keeping size 1) every axis where the target shape has size 1. Anything that still does not match could never have broadcast, and is an error.
5. The fused cross-entropy is the log-softmax VJP with $g = -e_t/n$, which simplifies to $(\mathrm{softmax}(z) - e_t)/n$. Computing the softmax with `M09.2`'s max-subtracted version keeps it finite; ignored rows are zeroed and excluded from $n$.
6. Every function builds new arrays from its inputs and never writes into them.

### The code, annotated

A shared conversion first: float inputs keep their dtype (float32 stays float32), anything else becomes float64.

```python
def _f64(x):
    a = np.asarray(x)
    return a if np.issubdtype(a.dtype, np.floating) else a.astype(np.float64)
```

`unbroadcast` and the matmul rule:

```python
def unbroadcast(g, shape):
    out = _f64(g)
    shape = tuple(shape)
    extra = out.ndim - len(shape)
    if extra < 0:                        # a gradient cannot have fewer axes
        raise ValueError(...)
    if extra:                            # axes numpy added in front: sum away
        out = out.sum(axis=tuple(range(extra)))
    for i, n in enumerate(shape):
        if n == 1 and out.shape[i] != 1: # a size-1 axis was stretched: sum, keep it
            out = out.sum(axis=i, keepdims=True)
    if out.shape != shape:               # e.g. (3,) -> (2,): never a broadcast
        raise ValueError(...)
    return out


def matmul_vjp(g, A, B):
    g, A, B = _f64(g), _f64(A), _f64(B)
    if A.ndim < 2 or B.ndim < 2:         # 1-D matmul drops axes; make callers reshape
        raise ValueError(...)
    dA = g @ np.swapaxes(B, -1, -2)      # G B^T, transposing only the matrix axes
    dB = np.swapaxes(A, -1, -2) @ g      # A^T G
    # a [k, n] weight used across a [T, m, k] batch gets the sum over T
    return unbroadcast(dA, A.shape), unbroadcast(dB, B.shape)
```

`swapaxes(B, -1, -2)` and not `B.T`: for a batched `B` of shape `[T, k, n]`, `.T` would reverse every axis.

The two softmax rules use only the saved output:

```python
def softmax_vjp(g, y, axis=-1):
    g, y = _f64(g), _f64(y)
    # y * (g - <g, y>): the second term is the off-diagonal -y y^T part
    return y * (g - np.sum(g * y, axis=axis, keepdims=True))


def log_softmax_vjp(g, y, axis=-1):
    g, y = _f64(g), _f64(y)
    # y here is the saved log-softmax; exp(y) is the softmax
    return g - np.exp(y) * np.sum(g, axis=axis, keepdims=True)
```

The normalizations, with `rstd` accepted in either per-row shape:

```python
def _rows(rstd, like):
    """rstd reshaped to like.shape[:-1] + (1,), so it broadcasts over the last axis."""
    return _f64(rstd).reshape(like.shape[:-1] + (1,))


def layernorm_vjp(g, xhat, rstd, gamma):
    g, xhat, gamma = _f64(g), _f64(xhat), _f64(gamma)
    r = _rows(rstd, xhat)
    d = g * gamma                        # gradient reaching xhat
    # direct path, minus the path through the mean, minus the path through r
    dx = r * (d - d.mean(axis=-1, keepdims=True) - xhat * (d * xhat).mean(axis=-1, keepdims=True))
    lead = tuple(range(g.ndim - 1))      # every axis but the feature axis
    return dx, (g * xhat).sum(axis=lead), g.sum(axis=lead)   # dx, dgamma, dbeta


def rmsnorm_vjp(g, x, rstd, w):
    g, x, w = _f64(g), _f64(x), _f64(w)
    r = _rows(rstd, x)
    d = g * w
    # direct path minus the path through r; no mean term, RMSNorm does not center
    dx = r * (d - x * (r * r) * (d * x).mean(axis=-1, keepdims=True))
    lead = tuple(range(g.ndim - 1))
    return dx, (g * x * r).sum(axis=lead)                    # dx, dw
```

The fused cross-entropy:

```python
def cross_entropy_vjp(logits, targets, ignore_index=-100):
    z = _f64(logits)
    t = np.asarray(targets)
    if t.shape != z.shape[:-1]:          # one target per row
        raise ValueError(...)
    v = z.shape[-1]
    valid = t != ignore_index
    # -1 would silently index the last class; a bad target is a data bug
    if ((t[valid] < 0) | (t[valid] >= v)).any():
        raise ValueError(...)
    n = int(valid.sum())                 # the mean is over kept rows only
    if n == 0:
        return np.zeros_like(z)          # nothing in the loss: no gradient
    grad = softmax(z, axis=-1)           # M09.2: max-subtracted, finite at 1e4; a new array
    rows = np.nonzero(valid)
    grad[rows + (t[valid],)] -= 1.0      # minus the one-hot, on kept rows
    grad[~valid] = 0.0                   # padding gets exactly 0
    return grad / n
```

The in-place `-=` is safe because `softmax` returned a fresh array, not one of the caller's. On the section 3 example, `softmax([0, ln 3])` is $[1/4, 3/4]$, subtracting 1 at target 1 gives $[1/4, -1/4]$, and $n = 1$.

Full solution: [`vjp.py`](../../../../course/solutions/python/tinyllm/autograd/vjp.py).

### Why this solution

- **Cost.** Every rule is $O(\text{size of the input})$ apart from matmul, whose two products cost $2mkn$ FLOPs each, twice the forward product. The softmax rules need one reduction per row instead of a $V \times V$ Jacobian: for $V = 256$ that is 256 multiply-adds per row instead of 65 536.
- **Why save outputs, not inputs.** Softmax and log-softmax VJPs are cheapest in terms of $y$ (or $\ell$), and LayerNorm's in terms of $\hat x$ and $r$. Saving those avoids recomputing exponentials and statistics in the backward pass.
- **Why fuse cross-entropy.** Chaining log-softmax and a one-hot pick would compute the same thing in two steps with an intermediate gradient; the fused form is one subtraction, and computing it from logits with the stable softmax avoids ever taking $\log$ of a probability that underflowed.
- **Why raise on bad shapes and targets.** A gradient with the wrong shape, or a target of $-1$ that numpy wraps to the last class, does not crash later: it trains the wrong thing. Raising at the boundary turns a silent error into a loud one.
- **Tradeoff accepted.** Inputs keep their float dtype rather than being promoted to float64, so a float32 graph stays float32 and its gradients carry float32 rounding. Gradcheck therefore runs in float64.

### Explain it back

1. Why does `matmul_vjp` call `unbroadcast` on `dB` even though `X^T G` already has two axes?

   <details><summary>Answer</summary>

   With a batched `A` of shape `[T, m, k]` and a weight `B` of shape `[k, n]`, `swapaxes(A) @ g` has shape `[T, k, n]`: one gradient per use of `B`. The weight's gradient is the sum over the $T$ uses, which `unbroadcast` performs.

   </details>

2. In `softmax_vjp`, what does the `np.sum(g * y, ..., keepdims=True)` term correspond to in the Jacobian, and what is wrong without it?

   <details><summary>Answer</summary>

   It is the $-yy^\top g$ part of $J^\top g = (\operatorname{diag}(y) - yy^\top)g$. Without it the result is $y \odot g$, only the diagonal. In the section 3 example that gives $[1/4, 0]$ instead of $[3/16, -3/16]$, and the gradients no longer sum to zero.

   </details>

3. LayerNorm's `dx` has three terms and RMSNorm's has two. Where does the missing one go?

   <details><summary>Answer</summary>

   LayerNorm subtracts the mean, so a change in $x$ also changes $\mu$; that path is the `- d.mean(...)` term. RMSNorm never centers, so there is no path through a mean. Both divide by a statistic of $x$, which gives the last term in each.

   </details>

4. Why does `cross_entropy_vjp` divide by `n = valid.sum()` and not by the number of rows?

   <details><summary>Answer</summary>

   The loss is the mean over the kept rows, so its derivative carries $1/n$ for those rows. Dividing by all rows would scale every gradient by the fraction of non-padding tokens, so batches with more padding would train with a smaller effective learning rate.

   </details>

5. Why compute the softmax with `M09.2`'s `softmax` instead of `np.exp(z) / np.exp(z).sum()`?

   <details><summary>Answer</summary>

   `np.exp(z)` overflows to `inf` in float64 once a logit passes about 709, and `inf / inf` is NaN. The stable softmax subtracts the row maximum first, so every exponent is at most 0 and the result is finite for logits near $10^4$.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. $G B$ instead of $G B^\top$ | correct shape for square $B$, wrong values | `g @ np.swapaxes(B, -1, -2)` |
| 2. a transposed gradient | $\bar B^\top$ where $\bar B$ belongs | `dB = np.swapaxes(A, -1, -2) @ g`, the shape of `B` |
| 3. not summing over broadcast axes | a bias or shared weight gets a batch of gradients | both matmul gradients go through `unbroadcast`; parameter gradients sum over `lead` |
| 4. the softmax Jacobian's diagonal only | $\bar x = y \odot g$ | subtracts $\langle g, y\rangle$ per row before multiplying by $y$ |
| 5. log-softmax VJP with $\ell$ where $e^\ell$ belongs | log-probabilities used as probabilities | `np.exp(y)` turns the saved log-softmax back into probabilities |
| 6. treating $\mu$ or $r$ as a constant | missing correction terms in LayerNorm or RMSNorm | the mean and variance terms in LayerNorm, the $x r^2\,\mathrm{mean}(d \odot x)$ term in RMSNorm |
| 7. $\bar\gamma = \sum g$ | the scale learns like a shift | `dgamma = (g * xhat).sum(...)`; only `dbeta` is `g.sum(...)` |
| 8. averaging over every row, or letting padding through | gradients scaled by the padding ratio; padding tokens trained | divides by the kept count `n`; sets ignored rows to 0 |
| 9. `exp(z) / sum(exp(z))` inside the loss gradient | NaN once a logit passes 709 | `M09.2`'s max-subtracted `softmax` |
| 10. updating the upstream gradient in place | the caller's `g` changes under it | every rule builds new arrays; the one `-=` writes into the fresh softmax |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.2` | `softmax` inside `cross_entropy_vjp` |
| Back | `M11.1` | its `cross_entropy` is the loss whose gradient `cross_entropy_vjp` is |
| Back | `M04.2` | `vjp_numeric` checks the softmax VJP numerically |
| Back | `M03.1` | row-major matrices and matmul shapes |
| Forward | `L0.1` | the Tensor's backward sums broadcast gradients back to each input's shape with `unbroadcast` |
| Forward | `L0.2` | each op of the tensor library registers one of these VJPs |
| Forward | `L0.3` | the fused softmax cross-entropy, $(\mathrm{softmax} - \mathrm{onehot})/n$ |
| Forward | `L3.1` | backpropagation through time by hand chains `matmul_vjp` over steps |
| Forward | `L7.1` | `rmsnorm_vjp` is RMSNorm's backward |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the closed-form rules | PyTorch's derivative table | one backward formula per op, from which the autograd code is generated | `tools/autograd/derivatives.yaml` |
| `layernorm_vjp` | llm.c | the same three-term backward in plain C, fused over a batch | `train_gpt2.c` (`layernorm_backward`) |
| `cross_entropy_vjp` | Liger Kernel | the linear layer, softmax, and cross-entropy fused in chunks, so the $[T, V]$ logits never exist in memory | `src/liger_kernel/ops/fused_linear_cross_entropy.py` |
| `rmsnorm_vjp` | PyTorch `F.rms_norm` | fused kernels with the saved `rstd`, the same formula | `aten/src/ATen/native/layer_norm.cpp` |
