# Hessian-vector products and the recompute-versus-memory schedule

Reverse mode buys its speed with memory. Consider the capstone of *AI Engineering*: an 8-layer Llama with a 512-token context, trained on a laptop. Its backward pass needs every layer's saved activations at once. At batch 16 that is several hundred megabytes in float32 before the logits, and it is the activations, not the 10M weights, that run the process out of memory. [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../ai-engineering/17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) fixes this with activation checkpointing: keep only some layer inputs and recompute the rest during backward. Which layers to keep is a counting problem with an exact answer, and you need that answer before the mechanism can be written.

The same stage of training is where curvature questions come up. Is the learning rate past $2/\lambda_{\max}$, the stability limit of gradient descent? The tool for such questions is the Hessian-vector product, which gives you curvature along a direction without ever forming the Hessian. This section provides both pieces: `hvp_fd`, and the schedule functions the checkpointing code calls. The [Optimization](../16-optimization/README.md) chapter can then estimate the top Hessian eigenvalue with `hvp_fd` as an optional extension.

Nothing here depends on earlier code, but the reading builds on [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md) (its cross-entropy gradient is the one whose $Hv$ is compared with PyTorch), on the forward-versus-reverse cost counts of the [chapter problem set](05-problem-set.md), and on the Hessian and second-derivative test of [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../10-calculus-3/02-jacobians-and-the-chain-rule.md). The complete implementation is [`hvp.py`](../../../../course/solutions/python/tinyllm/autograd/hvp.py), with its interface in the contract [`course/contracts/py/tinyllm/autograd/hvp.pyi`](../../../../course/contracts/py/tinyllm/autograd/hvp.pyi).

> **Learning objectives**
> - Derive the central-difference Hessian-vector product and its $O(\epsilon^2)$ error from a Taylor expansion of the gradient.
> - Explain why the central difference is exact on a quadratic and why a finite-difference Hessian must be symmetrized.
> - Compute the peak memory and recomputation of any checkpoint schedule, and the least achievable peak $P(P+1)/2 \ge n$.
> - Construct the optimal schedule under a memory budget and explain why the greedy construction is correct.

## Notation

The section uses two families of symbols, one for curvature and one for schedules.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f: \mathbb{R}^n \to \mathbb{R}$ | a scalar loss of $n$ parameters | function |
| $x \in \mathbb{R}^n$ | the point (parameters) | `float64[...]` |
| $g(x) = \nabla f(x)$ | the gradient (`grad_fn`) | same shape as $x$ |
| $H(x) = \nabla^2 f(x)$ | the Hessian: $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ | $n \times n$, symmetric |
| $v$ | a direction | same shape as $x$ |
| $\epsilon$ | the finite-difference step along $v$ (`eps`) | float |
| $D^k g(x)[v, \dots, v]$ | the $k$-th directional derivative of $g$ along $v$ (a derivative of order $k + 1$ of $f$) | same shape as $x$ |
| $e_j$ | the $j$-th unit vector | `float64[n]` |
| $n$ (`n_layers`) | layers in a stack | `int` |
| $k$ | segments in a schedule | `int` |
| $s_i$ | layers in segment $i$ ($i = 0, \dots, k-1$), $\sum_i s_i = n$ | `int` |
| `starts` | first layer of each segment: $0, s_0, s_0 + s_1, \dots$ | `list[int]` |
| $B$ (`mem_budget_layers`) | the memory budget, in saved layer inputs | `int` |
| $P$ | the peak number of saved layer inputs held at once | `int` |

## The Hessian-vector product as a derivative of the gradient

The key observation is that $Hv$ is a directional derivative of the gradient. Taylor-expand the gradient around $x$ along $v$:

$$g(x + \epsilon v) = g(x) + \epsilon\, H v + \tfrac{\epsilon^2}{2} D^2 g(x)[v, v] + \tfrac{\epsilon^3}{6} D^3 g(x)[v, v, v] + O(\epsilon^4).$$

The first-order term is $\epsilon H v$, so $Hv = \lim_{\epsilon \to 0} (g(x + \epsilon v) - g(x))/\epsilon$. That limit is a derivative, and derivatives can be taken numerically, as in [Derivatives and Finite Differences](../07-calculus-1/01-derivatives-and-finite-differences.md).

A central difference does much better than a one-sided one. Subtract the expansion at $-\epsilon$ from the one at $+\epsilon$: $g(x)$ and the $\epsilon^2$ term cancel, and

$$\frac{g(x + \epsilon v) - g(x - \epsilon v)}{2\epsilon} = Hv + \frac{\epsilon^2}{6} D^3 g(x)[v, v, v] + O(\epsilon^4).$$

The one-sided quotient keeps the $\tfrac{\epsilon}{2} D^2 g(x)[v,v]$ term: halving $\epsilon$ halves its error, while halving it in the central quotient quarters the error. The step cannot be made arbitrarily small either. Each gradient carries rounding error of about $u\,|g|$ ($u = 2^{-53} \approx 1.1 \times 10^{-16}$ in float64), and dividing by $2\epsilon$ magnifies it to about $u\,|g|/\epsilon$. Balancing $\epsilon^2$ against $u/\epsilon$ puts the best step near $u^{1/3} \approx 5 \times 10^{-6}$ times the scale of the problem; the default $10^{-4}$ gives about $10^{-8}$ relative accuracy on smooth losses and is far from the rounding floor.

On a quadratic the central difference is exact. For $f(x) = \tfrac12 x^\top A x + b^\top x$, the gradient $g(x) = \tfrac12(A + A^\top) x + b$ is affine, $D^3 g = 0$, and the central difference returns $\tfrac12 (A + A^\top) v$ for every $\epsilon$ (up to rounding). Using a non-symmetric $A$ makes the point that the Hessian is the symmetric part.

## What curvature costs

One $Hv$ costs two gradient calls, each about one forward and one backward pass. Forming $H$ would take $n$ of them and $n^2$ numbers of memory: for a 10M-parameter model, $10^{14}$ entries. So everything that needs curvature uses products instead. Newton-CG solves with $Hv$, and the power iteration $v \leftarrow Hv / \lVert Hv \rVert$ finds the top eigenvalue $\lambda_{\max}$.

For small $n$, `hessian_fd` builds $H$ column by column from $H e_j$ and returns $(H + H^\top)/2$. The symmetrization matters: the columns carry $O(\epsilon^2)$ errors that differ between $H_{ij}$ and $H_{ji}$, and code that assumes symmetry (eigenvalues, Cholesky) must get it exactly.

Frameworks compute $Hv$ exactly at the same cost with Pearlmutter's $\mathcal{R}$-operator, forward mode over reverse mode. That needs an autograd that differentiates its own backward pass, which the book's engine does not, so finite differences are the right tool here.

## Why reverse mode stores activations

Now the memory side. A stack of layers $y_{l+1} = f_l(y_l)$, $l = 0, \dots, n-1$, computes the VJP of layer $l$ from its saved input $y_l$ (a matmul's $\bar W = \bar y^\top x$ needs $x$). Count memory in units of one saved layer input. A plain forward pass keeps all $n$, and backward frees them from the top down, so the peak is $n$.

Recomputation breaks that. Split the stack into $k$ consecutive segments of $s_0, \dots, s_{k-1}$ layers. Every segment but the last runs forward keeping only its input (one unit) and discards its internal activations; the last segment runs normally and keeps its $s_{k-1}$. During backward, segment $i$ is rerun from its saved input to rebuild its $s_i$ activations, then backpropagated and freed. This is what `torch.utils.checkpoint.checkpoint_sequential` does.

While segment $i$ is being backpropagated, memory holds the inputs of segments $0, \dots, i-1$ (still needed later) plus segment $i$'s $s_i$ activations (the first of which is its saved input). The end of the forward pass is the case $i = k - 1$. So

$$P = \max_{0 \le i < k} (i + s_i), \qquad \text{recomputed layers} = n - s_{k-1}.$$

Both extremes cost $n$: one segment ($k = 1$) is no checkpointing, and one layer per segment holds $n$ inputs while recomputing $n - 1$ layers for nothing. The interesting schedules lie between.

## From $\sqrt n$ to $\sqrt{2n}$

The classic choice is uniform segments. With $k$ segments of $s$ layers ($ks = n$), $P = (k - 1) + s = n/s + s - 1$. Setting the derivative $-n/s^2 + 1$ to zero gives $s = \sqrt n$ and $P \approx 2\sqrt n - 1$: the sublinear-memory result of Chen et al.

You can do better by noticing that the term $i + s_i$ grows with $i$, so later segments should be smaller. If every $i + s_i \le P$, then $s_i \le P - i$, and $k$ segments hold at most

$$C(k, P) = \sum_{i=0}^{k-1} (P - i) = kP - \frac{k(k-1)}{2}$$

layers. That is largest at $k = P$ (the segment sizes $P, P-1, \dots, 1$), where it is the triangular number $P(P+1)/2$. So the least peak of any schedule is the least $P$ with $P(P+1)/2 \ge n$, about $\sqrt{2n}$: for $n = 16$, $P = 6$ against 7 for uniform segments of 4. Compute it with integers (`math.isqrt`); a float square root is wrong by one for large $n$.

In practice you are given a budget rather than asked for the minimum. Given $B \ge P_{\min}$, the goal is to minimize recomputation, that is, maximize the last segment. With $k$ segments the last holds at most $\min(B - k + 1,\, n - k + 1)$ layers (its $i + s_i \le B$, and every other segment needs at least one), and the other $k-1$ segments must cover the rest within $C(k-1, B)$. That bound on the last segment shrinks as $k$ grows, so the fewest feasible segments is optimal. Fill the first $k - 1$ segments greedily, largest first, $s_i = \min(B - i,\ \text{rest} - (k - 2 - i))$, leaving at least one layer for each later segment. The order fixes one answer among ties, so every caller checkpoints the same layers. A brute force over all $2^{n-1}$ schedules agrees for every $n \le 10$.

## Working it by hand

Both halves of the section deserve a hand check; these numbers become the first two tests of the implementation.

For the Hessian-vector product, take the quartic $f(x) = x^4/24$ in one dimension: $g(x) = x^3/6$, $H(x) = g'(x) = x^2/2$, $g''(x) = x$, and $g'''(x) = 1$. At $x = 1$, $v = 1$, $\epsilon = 0.1$:

| quantity | value |
|---|---|
| $g(1.1) = 1.331/6$, $g(0.9) = 0.729/6$ | $0.2218333$, $0.1215$ |
| central: $(1.331 - 0.729)/(6 \cdot 0.2)$ | $0.602/1.2 = 0.5016667$ |
| exact $H v$ | $0.5$ |
| error, predicted $\epsilon^2 v^3 g'''/6$ | $0.01/6 = 0.0016667$ (exact here: $g$ is cubic, so the series stops) |
| one-sided: $(1.331 - 1)/(6 \cdot 0.1)$ | $0.5516667$, error $0.0517 \approx \tfrac{\epsilon}{2} g''(1)$, thirty times larger |
| forgetting the 2: $0.602/0.6$ | $1.0033$ |

For checkpointing, take a six-layer stack, $n = 6$, and compare five schedules:

| schedule (`starts`) | sizes $s_i$ | $i + s_i$ | peak $P$ | recomputed |
|---|---|---|---|---|
| `[0]` | 6 | 6 | 6 | 0 |
| `[0, 1, 2, 3, 4, 5]` | 1, 1, 1, 1, 1, 1 | 1, 2, 3, 4, 5, 6 | 6 | 5 |
| `[0, 2, 4]` (uniform, $s \approx \sqrt 6$) | 2, 2, 2 | 2, 3, 4 | 4 | 4 |
| `[0, 3, 5]` | 3, 2, 1 | 3, 3, 3 | 3 | 5 |
| `[0, 3]` | 3, 3 | 3, 4 | 4 | 3 |

The least peak is 3, because $3 \cdot 4/2 = 6 \ge 6 > 2 \cdot 3/2$, and `[0, 3, 5]` reaches it. With a budget of 4: one segment needs 6 units, too many; two segments can keep a last segment of $\min(4 - 1, 6 - 1) = 3$ layers if the first covers the other 3 within $C(1, 4) = 4$, which it does. So `checkpoint_schedule(6, 4) == [0, 3]`: the same peak as uniform segments and one recomputed layer fewer.

## Designing the interface

Five functions cover both halves:

```python
# python/tinyllm/autograd/hvp.py
def hvp_fd(grad_fn, x, v, eps: float = 1e-4) -> NDArray          # (g(x + eps v) - g(x - eps v)) / (2 eps)
def hessian_fd(grad_fn, x, eps: float = 1e-4) -> NDArray          # 1-D x: columns H e_j, then (H + H^T) / 2
def checkpoint_cost(n_layers: int, starts) -> tuple[int, int]     # (peak, recomputed)
def min_checkpoint_memory(n_layers: int) -> int                   # least P with P (P + 1) / 2 >= n
def checkpoint_schedule(n_layers: int, mem_budget_layers: int) -> list[int]   # segment starts
```

`hvp_fd` calls `grad_fn` exactly twice, each time with a new float64 array, and copies each result, so a gradient that returns its own argument is safe. The step is $\epsilon v$, not normalized: the caller scales $v$ if its norm is far from 1. Shape mismatches, a non-positive or non-finite step, and an infeasible budget raise `ValueError`.

## Building it step by step

The reasoning behind the code runs in six steps.

1. $Hv$ is the derivative of the gradient along $v$, and the central difference has error $O(\epsilon^2)$ where the one-sided one has $O(\epsilon)$. So `hvp_fd` is two gradient calls at $x \pm \epsilon v$ divided by $2\epsilon$, with $v$ used as given.
2. A cheap `grad_fn` may return its own argument or a buffer it reuses. The two results must therefore be private copies, taken before the second call.
3. A dense Hessian is $n$ products with the unit vectors, one per column. The columns carry different $O(\epsilon^2)$ errors, so the result is symmetrized, which makes $H = H^\top$ exact in floating point.
4. The cost model needs only the segment sizes: peak $\max_i (i + s_i)$ and recompute $n - s_{k-1}$. Validating the schedule first (starts at 0, strictly increasing, below $n$) means every size is at least 1.
5. The least peak is the least $P$ with $P(P+1)/2 \ge n$. `math.isqrt(2n)` is within one of it, so start there and adjust with integer comparisons, never a float square root.
6. Under a budget $B$, fewer segments means a larger last segment means less recompute. So try $k = 1, 2, \dots$ and stop at the first $k$ whose largest allowed last segment leaves a remainder the other $k - 1$ segments can cover; then fill those greedily, largest first.

The Hessian-vector product comes first:

```python
def _grad_at(grad_fn, x):
    """grad_fn(x) as a new float64 array of x's shape (a copy, never a view)."""
    g = np.array(grad_fn(x), dtype=np.float64)   # np.array copies; asarray would not
    if g.shape != x.shape:
        raise ValueError(...)
    return g


def hvp_fd(grad_fn, x, v, eps=1e-4):
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if x.shape != v.shape:
        raise ValueError(...)
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(...)
    # x + eps * v and x - eps * v are two NEW arrays; each gradient is copied
    # before the next call, so an aliasing grad_fn cannot change gp under us
    gp = _grad_at(grad_fn, x + eps * v)
    gm = _grad_at(grad_fn, x - eps * v)
    return (gp - gm) / (2.0 * eps)               # central: the eps^2 term cancels
```

The dense Hessian, for small $n$, is a loop of products:

```python
def hessian_fd(grad_fn, x, eps=1e-4):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(...)
    n = x.size
    H = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        H[:, j] = hvp_fd(grad_fn, x, e, eps)     # column j is H e_j
    return (H + H.T) / 2.0                       # exactly symmetric in floating point
```

The cost model validates the schedule and applies the two formulas:

```python
def _sizes(n_layers, starts):
    """Segment sizes of a valid schedule; ValueError otherwise."""
    if n_layers < 1:
        raise ValueError(...)
    s = [int(b) for b in starts]
    if not s or s[0] != 0:
        raise ValueError(...)
    if any(b <= a for a, b in zip(s, s[1:])) or s[-1] >= n_layers:
        raise ValueError(...)                    # so every segment has >= 1 layer
    return [b - a for a, b in zip(s, s[1:] + [n_layers])]


def checkpoint_cost(n_layers, starts):
    sizes = _sizes(n_layers, starts)
    # while segment i is backpropagated: i saved inputs of earlier segments
    # plus its own s_i activations; i = k - 1 is the end of the forward pass
    peak = max(i + s for i, s in enumerate(sizes))
    return peak, n_layers - sizes[-1]            # the last segment is never rerun
```

The least peak is computed entirely in integers:

```python
def min_checkpoint_memory(n_layers):
    if n_layers < 1:
        raise ValueError(...)
    p = math.isqrt(2 * n_layers)                 # floor(sqrt(2n)): close to the answer
    while p * (p + 1) // 2 < n_layers:           # too small: grow
        p += 1
    while p > 1 and (p - 1) * p // 2 >= n_layers:  # one smaller still fits: shrink
        p -= 1
    return p
```

For $n = 6$, `isqrt(12)` is 3 and $3 \cdot 4/2 = 6 \ge 6$, while $2 \cdot 3/2 = 3 < 6$, so it returns 3. A float `ceil(sqrt(12))` would give 4.

Finally, the budgeted schedule:

```python
def _cover(k, budget):
    """Most layers k segments can hold when segment i has at most budget - i."""
    return k * budget - k * (k - 1) // 2


def checkpoint_schedule(n_layers, mem_budget_layers):
    n, b = n_layers, mem_budget_layers
    if b < min_checkpoint_memory(n):             # no schedule fits: refuse loudly
        raise ValueError(...)
    for k in range(1, n + 1):                    # fewest segments first
        # the last segment (index k - 1) may hold b - (k - 1) layers, and must
        # leave at least one layer for each of the k - 1 earlier segments
        last = min(b - k + 1, n - k + 1)
        if last >= 1 and n - last <= _cover(k - 1, b):
            break                                # the first feasible k is optimal
    rest = n - last
    sizes = []
    for i in range(k - 1):
        # as large as the budget allows, keeping one layer for each later segment
        s = min(b - i, rest - (k - 2 - i))
        sizes.append(s)
        rest -= s
    sizes.append(last)
    starts, at = [], 0
    for s in sizes:                              # sizes to starting layers
        starts.append(at)
        at += s
    return starts
```

Trace it on $n = 6$, $B = 4$. $k = 1$ gives `last = 4` but leaves 2 layers for zero segments; $k = 2$ gives `last = 3`, and 3 layers fit in one segment of at most 4. The fill gives `sizes = [3, 3]`, so `starts = [0, 3]`, peak 4, recompute 3, matching the hand table. The full solution is [`hvp.py`](../../../../course/solutions/python/tinyllm/autograd/hvp.py).

## Why this design

`hvp_fd` costs two gradient calls and $O(n)$ memory; `hessian_fd` costs $2n$ gradient calls and $n^2$ memory, for small $n$ only. The schedule functions are $O(n)$ integer work, negligible next to one training step.

Finite differences stand in for Pearlmutter's exact $Hv$ because the exact product needs forward mode over reverse mode, an autograd that can differentiate its own backward. The book's engine cannot. Two gradient calls give $Hv$ to about $10^{-8}$ relative accuracy, plenty for estimating $\lambda_{\max}$.

Uniform $\sqrt n$ segments are not good enough because they peak at about $2\sqrt n$ while shrinking segments reach $\sqrt{2n}$, and under a loose budget the uniform rule recomputes layers it did not need to. The greedy schedule is optimal (checked against brute force for $n \le 10$) and costs nothing extra.

Integer arithmetic for the least peak is a correctness issue, not a style choice. `math.sqrt` of a large integer rounds, and `ceil` of a rounded value can be off by one; for $n$ near $2^{61}$ the error is large. `isqrt` and integer comparisons are exact for every $n$.

The tradeoff accepted is that the cost model counts saved layer inputs as equal units and ignores the logits and optimizer state. That is a good approximation for a uniform stack of transformer blocks, which is what the checkpointing code applies it to.

## What a correct implementation must handle

On the curvature side, the tests start from the hand-computed quartic, with error $\epsilon^2/6$, which pins the formula. A quadratic must give $\tfrac12 (A + A^\top) v$ at three step sizes with $v$ not a unit vector, confirming that the step is $\epsilon v$. Halving $\epsilon$ must divide the error by 4, confirming a central rather than one-sided difference. Two golden tests compare with PyTorch: the exact double-backward $Hv$ of a cross-entropy loss whose gradient is the rule from the previous section (the curvature an optimizer would estimate), and a dense $6 \times 6$ Hessian, exactly symmetric (small-model Newton steps). Rosenbrock at $(-1.2, 1)$ must give $H = H^\top$ bit for bit and stay close to the analytic Hessian, because eigenvalue code assumes symmetry. A gradient that returns its own input must be called exactly twice, with `x` untouched, since cheap gradients alias. Shape and step errors must raise, and the default step must be accurate.

On the schedule side, the hand table above must come out exactly, since the checkpointing code checkpoints exactly these layers. No checkpointing and all checkpoints must both peak at $n$, and invalid schedules must raise. The least peak must satisfy $P(P+1)/2 \ge n > (P-1)P/2$ for $n$ up to $2^{61} - 1$, which tests integer arithmetic at scale. The budgeted schedule must stay within budget and recompute as few layers as brute force for $n \le 10$; ties must go to the largest first segment, on four cases, so every caller gets the same schedule; and a budget below the least peak must raise rather than overrun silently.

The pitfalls these tests catch:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| A one-sided difference | error $O(\epsilon)$: 0.5517 instead of 0.5017 in the hand example | evaluates at both $x + \epsilon v$ and $x - \epsilon v$ |
| Dividing by $\epsilon$ instead of $2\epsilon$ | every $Hv$ twice too large | divides by `2.0 * eps`, the distance between the two points |
| Reusing one buffer for $x \pm \epsilon v$ | a gradient that aliases its input changes under the caller; $Hv = 0$ | two new arrays for the points; `_grad_at` copies each result |
| Not symmetrizing the finite-difference Hessian | $H_{ij} \ne H_{ji}$ by $O(\epsilon^2)$ | returns `(H + H.T) / 2.0` |
| Peak as (segments $-$ 1) + (largest segment) | overcounts when the largest segment comes first; the optimizer picks worse schedules | `max(i + s for i, s in enumerate(sizes))` |
| The uniform $\sqrt n$ rule whatever the budget | over budget when it is tight, needless recomputation when it is loose | searches for the fewest feasible segments under the given budget |
| A float `ceil(sqrt(2n))` for the least peak | one too many at $n = 6$, wrong at large $n$ | `math.isqrt` and integer adjustment |
| Counting the last segment as recomputed | recomputation off by $s_{k-1}$ | recompute is `n_layers - sizes[-1]` |
| A budget check off by one | peaks one unit over the budget | last segment `b - k + 1`, so its $i + s_i = b$ exactly; earlier segments at most `b - i` |
| Normalizing $v$ inside `hvp_fd` | returns $Hv/\lVert v \rVert$ | uses $v$ as given; the step is $\epsilon v$ |

## Where this leads

Looking back, the cross-entropy VJP of [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md), $X^\top(\mathrm{softmax} - \mathrm{onehot})/n$ for a linear model, is the gradient whose $Hv$ is compared with PyTorch; the forward-versus-reverse cost counts of the [chapter problem set](05-problem-set.md) are behind "two gradients per $Hv$"; and [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../10-calculus-3/02-jacobians-and-the-chain-rule.md) supplies the Hessian and the second-derivative test.

Looking forward, `checkpoint_sequential` in [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../ai-engineering/17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) runs the segments with `checkpoint_schedule` and recomputes them in backward, which is what lets the capstone train with checkpointing on a laptop. In the [Optimization](../16-optimization/README.md) chapter, an optional extension runs power iteration on `hvp_fd` to estimate $\lambda_{\max}$ and compare the learning rate with $2/\lambda_{\max}$.

Production tools extend each piece. `torch.autograd.functional.hvp` and `torch.func.jvp(grad(f))` give exact forward-over-reverse products (Pearlmutter's $\mathcal{R}$-operator) at the cost of two backward passes (`torch/autograd/functional.py`). PyHessian computes Hessian spectra of real networks by power iteration and stochastic Lanczos, from $Hv$ only (`pyhessian/hessian.py`). `torch.utils.checkpoint.checkpoint_sequential` uses uniform segments, RNG state replay, and non-reentrant saved-tensor hooks (`torch/utils/checkpoint.py`). Checkmate (Jain et al., 2020) solves optimal rematerialization for arbitrary graphs as an integer program (`checkmate/core/solvers/` in the `checkmate` repository).

Further reading:

- Pearlmutter, ["Fast Exact Multiplication by the Hessian"](https://doi.org/10.1162/neco.1994.6.1.147) (Neural Computation, 1994).
- Chen, Xu, Zhang, and Guestrin, [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174) (2016).
- Griewank and Walther, *Evaluating Derivatives*, chapter 12 (checkpointing).

## Key terms

- **Hessian-vector product**: $Hv$, the derivative of the gradient along $v$, computed without forming $H$.
- **Central difference**: $(g(x + \epsilon v) - g(x - \epsilon v))/2\epsilon$, whose even-order error terms cancel, leaving $O(\epsilon^2)$.
- **Activation checkpointing**: saving only some layer inputs in the forward pass and recomputing the rest during backward.
- **Segment**: a run of consecutive layers that is recomputed from its saved input during backward.
- **Peak memory**: $P = \max_i (i + s_i)$, the most saved layer inputs held at once under a schedule.
- **Pearlmutter's $\mathcal{R}$-operator**: forward mode applied over reverse mode, giving an exact $Hv$ at the cost of about two gradients.

## Summary

- The Hessian-vector product $Hv$ is the derivative of the gradient along $v$, so two gradient calls give it to $O(\epsilon^2)$ without forming the $n \times n$ Hessian.
- On a quadratic the central difference is exact at every $\epsilon$, because the gradient is affine; a dense finite-difference Hessian must be symmetrized.
- Reverse mode keeps one saved input per layer. Recomputing segments trades forward time for memory: the peak is $\max_i (i + s_i)$ and the extra work is every layer outside the last segment.
- Uniform segments of $\sqrt n$ layers peak near $2\sqrt n$; shrinking segments (sizes $P, P-1, \dots, 1$) reach the optimum $P \approx \sqrt{2n}$, and under a budget the best schedule is found greedily.
- Exact integer arithmetic (`math.isqrt`) is required for the least peak at large $n$.

## Review questions

1. Why does `_grad_at` use `np.array(...)` rather than `np.asarray(...)`?

   <details><summary>Answer</summary>

   `np.asarray` returns the same object when it is already a float64 array. If `grad_fn` returns its argument (or a buffer it reuses), `gp` would be that array, and the second call could overwrite it, or `gp` and `gm` would be the same object and the difference 0. `np.array` copies.

   </details>

2. On the quartic of the hand example, how does the error change if $\epsilon$ goes from 0.1 to 0.05, and what would it do with a one-sided difference?

   <details><summary>Answer</summary>

   Central: the error is $\epsilon^2/6$, so it goes from $0.00167$ to $0.00042$, a factor 4. One-sided: the error is about $\frac{\epsilon}{2}g''(1) = \epsilon/2$, from about $0.05$ to $0.025$, a factor 2.

   </details>

3. Why is the peak $\max_i(i + s_i)$ and not $(k - 1) + \max_i s_i$?

   <details><summary>Answer</summary>

   While segment $i$ is rebuilt, only the inputs of the earlier segments $0, \dots, i - 1$ are still held: $i$ units, not $k - 1$. The inputs of later segments have already been freed after their own backward. So a large segment early costs less than the same segment late; the second formula overcounts whenever the largest segment comes first, and an optimizer using it would prefer worse schedules.

   </details>

4. Why does `checkpoint_schedule` stop at the first feasible $k$?

   <details><summary>Answer</summary>

   Recompute is $n - s_{k-1}$, and the largest possible last segment, $\min(B - k + 1, n - k + 1)$, shrinks as $k$ grows. So the fewest segments that can still cover the remaining layers gives the largest last segment and the least recompute.

   </details>

5. Why does `hvp_fd` not normalize $v$?

   <details><summary>Answer</summary>

   $Hv$ is linear in $v$; a caller doing power iteration or Newton-CG expects exactly $Hv$. Normalizing inside would return $Hv/\lVert v\rVert$, which is a different quantity. The step size is $\epsilon\lVert v\rVert$, so a caller with a huge or tiny $v$ scales it before calling.

   </details>

6. What is the least peak for a 16-layer stack, and how does it compare with uniform segments?

   <details><summary>Answer</summary>

   The least $P$ with $P(P+1)/2 \ge 16$ is 6 ($6 \cdot 7/2 = 21 \ge 16 > 15$), reached by shrinking segments. Uniform segments of 4 give $P = (4 - 1) + 4 = 7$.

   </details>
