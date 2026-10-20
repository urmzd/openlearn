# Error analysis, condition numbers, and tolerance budgets

Sooner or later every numerical codebase has to compare two numbers that are not equal and must not be. In *AI Engineering*, the C matmul of [Cache-blocked, packed, batch-invariant matmul in C](../../ai-engineering/22-kernels/03-tiled-batch-invariant-matmul.md) sums in tiles while numpy sums pairwise in blocks, and the two agree only to rounding. The int8 and int4 kernels of [Fused int4 and int8 dequantize-matmul in C](../../ai-engineering/22-kernels/07-fused-quantized-matmul.md) start from weights that [Quantization: int8, int4 groups, fp8, and the KV cache](../../ai-engineering/21-inference/05-quantization.md) moved on purpose. A tolerance chosen by eye fails in one of two ways. Too tight, and a correct kernel fails on a long row, because the error of a dot product grows with its length. Too loose, and a kernel that drops one term of 257 passes.

The usual way out is a $\sqrt{K}$ rule, which is a statistical rule of thumb. This section derives the rigorous version before the kernels arrive: where the error comes from, how big it can be for a given input, and how much of it is the problem's fault rather than the algorithm's. You will build one model of rounding, use it to bound sums, dot products, and matrix products, add deliberate perturbations such as quantization to the budget, separate conditioning from stability, and finally use the same reasoning to choose the step of a finite difference.

The section builds on the unit roundoff of [IEEE 754 anatomy, ulps, and round to nearest even](01-ieee-754.md), the summation orders of [Stable numerics](02-stable-numerics.md), and the cancellation problems of [Problem set A](08-problem-set-a.md). In code it uses `svd` from [SVD, Eckart-Young low rank, and least squares](../09-linear-algebra/05-svd-low-rank-and-least-squares.md) (its singular values feed `cond`) and `central_diff` from [Derivatives and Finite Differences](../07-calculus-1/01-derivatives-and-finite-differences.md) (the real difference the optimal step is measured against). The reference implementation is [`tolerance.py`](../../../../course/solutions/python/tinyllm/num/tolerance.py), with its interface in [`tolerance.pyi`](../../../../course/contracts/py/tinyllm/num/tolerance.pyi).

> **Learning objectives**
> - Derive $\gamma_k = ku/(1 - ku)$ from the standard rounding model and use it to bound the error of a sum, a dot product, and a matrix product in any summation order.
> - Explain why rounding bounds scale with $\lvert x\rvert^\top \lvert y\rvert$ rather than $\lvert x^\top y\rvert$.
> - Build a tolerance budget that includes a deliberate perturbation such as quantization.
> - Compute the condition number of a scalar function and of a linear system, and explain what it says that an algorithm cannot fix.
> - Derive the optimal step for forward and central finite differences.

## The notation

The symbols used throughout the section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$ | significand bits including the implicit 1: 53 (f64), 24 (f32), 11 (f16), 8 (bf16), 4 (e4m3), 3 (e5m2) | integer |
| $u = 2^{-p}$ | unit roundoff: the largest relative error of one rounding | float |
| $\mathrm{fl}(\cdot)$ | the floating-point result of an expression | float |
| $\delta_i$ | the relative error of one rounding, $\lvert\delta_i\rvert \le u$ | float |
| $\gamma_k = \dfrac{ku}{1 - ku}$ | the accumulated relative error of $k$ roundings, for $ku < 1$ | float |
| $x, y \in \mathbb{R}^k$ | the vectors of a dot product | `float[k]` |
| $\lvert x\rvert$ | elementwise absolute value | `float[k]` |
| $A, B$ | matrices, $A$ is $m \times k$, $B$ is $k \times n$ | `float[m, k]`, `float[k, n]` |
| $\sigma_{\max}, \sigma_{\min}$ | the largest and smallest singular values | float |
| $\kappa_2(A) = \sigma_{\max} / \sigma_{\min}$ | the 2-norm condition number | float $\ge 1$ |
| $\kappa_f(x) = \lvert x f'(x) / f(x)\rvert$ | the relative condition number of a scalar function | float $\ge 0$ |
| $h$ | a finite-difference step | float $> 0$ |

## The rounding model

IEEE arithmetic rounds the exact result of each operation to the nearest float, so for $\circ \in \{+, -, \times, /\}$

$$\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta), \qquad \lvert\delta\rvert \le u = 2^{-p}.$$

$u$ is half the gap between 1 and the next float. That gap is $2^{1-p}$ (machine epsilon, `np.finfo(np.float32).eps` $= 2^{-23}$), and rounding to the nearest float errs by at most half of it, $2^{-24}$ for float32. Counting the implicit bit matters: bf16 stores 7 fraction bits but keeps 8 significant bits, so its $u$ is $2^{-8}$.

## Products of $(1 + \delta)$ and $\gamma_k$

One rounding is easy. The interesting question is what happens when a value passes through many. After $k$ roundings it carries a factor $\prod_{i=1}^k (1 + \delta_i)$, and if $ku < 1$,

$$\prod_{i=1}^{k} (1 + \delta_i) = 1 + \theta_k, \qquad \lvert\theta_k\rvert \le \gamma_k = \frac{ku}{1 - ku}.$$

The proof is an induction: $(1 + \theta_{k-1})(1 + \delta_k) = 1 + \theta_{k-1} + \delta_k + \theta_{k-1}\delta_k$, and $\gamma_{k-1} + u + \gamma_{k-1} u \le \gamma_k$. To first order $\gamma_k \approx ku$; the denominator makes the bound rigorous and says it is void once $ku \ge 1$, which for bf16 happens at $k = 256$.

Now apply it to a dot product summed left to right: $s_1 = \mathrm{fl}(x_1 y_1)$, $s_i = \mathrm{fl}(s_{i-1} + \mathrm{fl}(x_i y_i))$. The term $x_1 y_1$ goes through one multiplication and $k - 1$ additions, and $x_i y_i$ for $i \ge 2$ through one multiplication and $k - i + 1$ additions. Every term goes through at most $k$ roundings, so

$$\lvert \mathrm{fl}(x^\top y) - x^\top y\rvert \le \gamma_k \sum_i \lvert x_i y_i\rvert = \gamma_k\, \lvert x\rvert^\top \lvert y\rvert.$$

The same count holds for any order of the additions (pairwise, blocked, tiled), because each term meets at most $k - 1$ additions. That is why one bound covers numpy's BLAS, the tiled C kernel, and a plain loop. A sum without products ($y = 1$, exact multiplications) needs only $\gamma_{k-1}$, since one term alone is exact.

## Why $\lvert x\rvert \cdot \lvert y\rvert$ and not $\lvert x \cdot y\rvert$

The bound has an absolute value inside the sum, and that is not a technicality. The roundings are relative to the partial sums, and the partial sums can be large even when the result is small. $[1, 10^{-8}, -1] \cdot [1, 1, 1]$ in float32 computes $1 + 10^{-8} = 1$, then $1 - 1 = 0$: the exact result $10^{-8}$ is lost completely, a 100% relative error, and that is correct float32 arithmetic. The bound $\gamma_3 (1 + 10^{-8} + 1) \approx 3.6 \times 10^{-7}$ allows it. A bound proportional to $\lvert x^\top y\rvert = 10^{-8}$ would call a correct kernel broken. For a matrix product the elementwise bound is $\gamma_k (\lvert A\rvert\, \lvert B\rvert)_{ij}$.

## Tolerance budgets

With a bound for rounding in hand, you can write a test allowance that is derived rather than guessed. A test compares an implementation against a reference, and every source of difference must be in the allowance, and nothing else:

$$\text{allowed}_{ij} = \underbrace{(dA\, \lvert B\rvert)_{ij}}_{\text{weights moved by } \le dA} + \underbrace{\gamma_k \big((\lvert A\rvert + dA)\, \lvert B\rvert\big)_{ij}}_{\text{rounding of the moved product}}.$$

With quantization to a grid of step $s$, each weight moves by at most $dA = s/2$. With $dA = 0$ it is the plain rounding bound that the standalone C matmul is checked against. `bound_ratio` reports $\max_{ij} \lvert\text{error}\rvert / \text{allowed}$: at most 1 passes, and how close to 1 it is tells you whether the test can still catch a bug. `assert_close_bounded` multiplies the dot bound by a `slack` (default 4) because the "expected" side is itself computed in floating point (in float64, or in float32 by a different order).

The worst case is rarely reached. Rounding errors have random signs and partially cancel, so the typical error grows like $\sqrt{k}\, u$, not $k u$ (Higham and Mary 2019). A statistical rule uses that: tolerances times $\sqrt{K}$. It is tighter and almost always right; the bound here is looser and always right. Use the bound when a false failure would be expensive to debug.

## Condition numbers

So far the question has been how wrong an algorithm can be. Error analysis separates that from a second question. The **backward error** of an algorithm asks: for which nearby input is my computed output the exact answer? The **condition number** of the problem asks: how much does the exact answer move when the input moves? The forward error is at most their product.

For a scalar function, a relative change $\epsilon$ in $x$ changes $f(x)$ by $f'(x)\, x \epsilon$, a relative change of

$$\kappa_f(x) = \left\lvert \frac{x f'(x)}{f(x)} \right\rvert.$$

$\sqrt{x}$ has $\kappa = 1/2$: it halves relative errors. $x - 1$ near $x = 1$ has $\kappa = \lvert x/(x - 1)\rvert$, which is $10^8$ at $1 + 10^{-8}$. The cancellation of [Problem set A](08-problem-set-a.md) is ill-conditioning, a property of subtraction near equal numbers, not of any algorithm.

For a linear system $Ax = b$ and a perturbation $A(x + \delta x) = b + \delta b$, you have $A\, \delta x = \delta b$, so $\lVert\delta x\rVert \le \lVert A^{-1}\rVert\, \lVert\delta b\rVert$, and $\lVert b\rVert \le \lVert A\rVert\, \lVert x\rVert$. Multiplying,

$$\frac{\lVert\delta x\rVert}{\lVert x\rVert} \le \lVert A\rVert\, \lVert A^{-1}\rVert\, \frac{\lVert\delta b\rVert}{\lVert b\rVert} = \kappa_2(A)\, \frac{\lVert\delta b\rVert}{\lVert b\rVert}$$

in the 2-norm, where $\lVert A\rVert_2 = \sigma_{\max}$ and $\lVert A^{-1}\rVert_2 = 1/\sigma_{\min}$. Equality holds when $b$ lies along the top left singular vector and $\delta b$ along the bottom one. Eigenvalues are not a substitute: $\begin{pmatrix} 1 & 10^3 \\ 0 & 1 \end{pmatrix}$ has both eigenvalues 1 and $\kappa_2 \approx 10^6$. A matrix whose $\sigma_{\min}$ is below $\max(m, n) \cdot 2^{-52} \sigma_{\max}$ is singular to working precision, and `cond` returns $\infty$ rather than a meaningless $10^{17}$.

## The optimal finite-difference step

The same two-sided accounting answers a question from the calculus chapters: how small should a finite-difference step be? The forward difference $(f(x + h) - f(x))/h$ of [Derivatives and Finite Differences](../07-calculus-1/01-derivatives-and-finite-differences.md) has truncation error $\tfrac{h}{2}\lvert f''\rvert$ (Taylor) and rounding error up to $2u\lvert f\rvert / h$ (two evaluations, each off by $u\lvert f\rvert$, divided by $h$). The total is

$$E_1(h) = \frac{h}{2} D + \frac{2uF}{h}, \qquad \frac{dE_1}{dh} = \frac{D}{2} - \frac{2uF}{h^2} = 0 \;\Rightarrow\; h^* = 2\sqrt{uF/D},$$

with $F = \lvert f\rvert$ and $D = \lvert f''\rvert$. The central difference $(f(x + h) - f(x - h))/(2h)$ has truncation $\tfrac{h^2}{6}\lvert f'''\rvert$ and rounding $uF/h$:

$$E_2(h) = \frac{h^2}{6} D + \frac{uF}{h}, \qquad \frac{D h}{3} - \frac{uF}{h^2} = 0 \;\Rightarrow\; h^* = (3uF/D)^{1/3}.$$

In float64 with unit scales that is about $2.1 \times 10^{-8}$ and $6.9 \times 10^{-6}$, and the best achievable errors are about $10^{-8}$ and $10^{-11}$. The central difference wins by three orders of magnitude, which is why gradient checks use it.

## Working it by hand

Two small examples make the bounds concrete. First, a dot product that loses two terms: $x = [1, 10^{-8}, 10^{-8}]$, $y = [1, 1, 1]$, all float32. In recursive summation $1 + 10^{-8}$ rounds to 1 (the gap at 1 is $2^{-23} \approx 1.19 \times 10^{-7}$, and $10^{-8}$ is less than half of it), and so does the next addition. The computed result is 1, the exact result is $1.00000002$, and the error is $2 \times 10^{-8}$.

For the bound, $u = 2^{-24} \approx 5.96 \times 10^{-8}$, $\gamma_3 = 3u/(1 - 3u) \approx 1.7881 \times 10^{-7}$, and $\lvert x\rvert \cdot \lvert y\rvert = 1.00000002$, so the allowance is $1.7881 \times 10^{-7}$. The error is about 0.11 of the allowance: inside, as it must be.

Second, an ill-conditioned system. $A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix}$ is symmetric, so its singular values are its eigenvalues, $\lambda = \tfrac12\big(2.0001 \pm \sqrt{4 + 10^{-8}}\big)$: about $2.00005$ and $4.99988 \times 10^{-5}$. So $\kappa_2(A) \approx 40002$. With $b = [2, 2.0001]$ the solution is $x = [1, 1]$. Move $b$ to $[2, 2.0002]$, a relative change of $10^{-4} / \lVert b\rVert = 10^{-4}/2.8285 = 3.54 \times 10^{-5}$, and the solution becomes $[0, 2]$, a relative change of $\lVert[-1, 1]\rVert / \lVert[1, 1]\rVert = 1$. The amplification is $1 / 3.54 \times 10^{-5} \approx 28284$, below 40002 as the theory says. No algorithm can do better than this problem allows. Both examples become the first test cases below.

## Designing the API

The helpers are for tests that compare two floating-point computations: they turn a tolerance into a derived number instead of a guess.

```python
# python/tinyllm/num/tolerance.py
def unit_roundoff(dtype: str) -> float                                   # "f32" -> 2^-24
def gamma(k: int, dtype: str) -> float                                   # k u / (1 - k u)
def sum_error_bound(k: int, dtype: str, abs_sum) -> NDArray              # gamma_(k-1) * sum|x|
def dot_error_bound(k: int, dtype: str, abs_dot) -> NDArray              # gamma_k * |x|.|y|
def matmul_error_bound(A, B, dtype: str, dA=0.0) -> NDArray              # dA|B| + gamma_k (|A| + dA)|B|
def assert_close_bounded(actual, expected, k: int, dtype: str, abs_dot, slack: float = 4.0) -> None
def bound_ratio(actual, expected, bound) -> float                        # max |error| / bound
def cond(A) -> float                                                     # sigma_max / sigma_min via svd
def relative_condition(f, df, x: float) -> float                         # |x f'(x) / f(x)|
def fd_error_model(h, order, dtype, f_scale=1.0, deriv_scale=1.0) -> float
def optimal_fd_step(order, dtype, f_scale=1.0, deriv_scale=1.0) -> float
```

`dtype` is one of `"f64"`, `"f32"`, `"f16"`, `"bf16"`, `"e4m3"`, `"e5m2"` or the numpy names.

## Building it step by step

Each function falls out of one principle. Everything starts from $u$: every bound above is a multiple of $u = 2^{-p}$, so the first piece is a table of $p$ (significand bits *including* the implicit 1) per format. Storing $p$ rather than $u$ keeps the table exact and readable, and $u$ is one power of two away. Next, $\gamma_k$ must refuse where it is void. The induction proves $\lvert\theta_k\rvert \le ku/(1 - ku)$ only for $ku < 1$; past that point no finite number is a bound, so the honest output is an error, not a large float.

```python
# significand bits including the implicit leading 1
_PRECISION = {
    "f64": 53, "float64": 53,
    "f32": 24, "float32": 24,
    "f16": 11, "float16": 11,
    "bf16": 8, "bfloat16": 8,     # 7 stored fraction bits + the implicit 1
    "e4m3": 4,
    "e5m2": 3,
}


def unit_roundoff(dtype: str) -> float:
    if dtype not in _PRECISION:
        raise ValueError(f"unknown dtype {dtype!r}; one of {sorted(_PRECISION)}")
    return 2.0 ** -_PRECISION[dtype]          # u = 2^-p, half of machine epsilon


def gamma(k: int, dtype: str) -> float:
    u = unit_roundoff(dtype)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    ku = k * u
    if ku >= 1.0:                             # the induction needs ku < 1
        raise ValueError(f"k u = {ku} >= 1: the bound is void for k = {k} in {dtype}")
    return ku / (1.0 - ku)                    # gamma_0 = 0: no rounding, no error
```

An unknown format raises rather than defaulting to float32, because a silently wrong $u$ scales every tolerance by a power of two. The bf16 entry is the one most often written as 7; the rounding model above explains why it is 8.

Then come the counts. A sum of $k$ terms has $k - 1$ roundings, a dot product $k$: the first term of a sum is exact, and a dot product adds one multiplication to every term. So `sum_error_bound` uses $\gamma_{k-1}$ and `dot_error_bound` uses $\gamma_k$, each times the caller's $\sum\lvert\cdot\rvert$ (the absolute values, not the result). A matmul bound is a matrix of dot bounds: element $(i, j)$ is a dot product of length $k$, so the bound is $\gamma_k (\lvert A\rvert\,\lvert B\rvert)_{ij}$, computed in float64 with one matrix product of absolute values. A deliberate perturbation enters twice: moving $A$ by at most $dA$ moves the exact product by at most $dA\,\lvert B\rvert$, and the perturbed $A'$ has $\lvert A'\rvert \le \lvert A\rvert + dA$, so its rounding is at most $\gamma_k(\lvert A\rvert + dA)\,\lvert B\rvert$.

```python
def sum_error_bound(k: int, dtype: str, abs_sum: ArrayLike) -> NDArray:
    if k < 1:
        raise ValueError(f"a sum has k >= 1 terms, got {k}")
    return gamma(k - 1, dtype) * np.asarray(abs_sum, dtype=np.float64)   # first term is exact


def dot_error_bound(k: int, dtype: str, abs_dot: ArrayLike) -> NDArray:
    if k < 1:
        raise ValueError(f"a dot product has k >= 1 terms, got {k}")
    return gamma(k, dtype) * np.asarray(abs_dot, dtype=np.float64)       # + one multiply per term


def matmul_error_bound(A: ArrayLike, B: ArrayLike, dtype: str, dA: ArrayLike = 0.0) -> NDArray:
    a = np.abs(np.asarray(A, dtype=np.float64))
    b = np.abs(np.asarray(B, dtype=np.float64))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.shape[1] < 1:
        raise ValueError(f"need A [m, k] and B [k, n] with k >= 1, got {a.shape} and {b.shape}")
    d = np.broadcast_to(np.asarray(dA, dtype=np.float64), a.shape)   # scalar or per-weight
    if (d < 0).any():
        raise ValueError("dA must be >= 0")
    k = a.shape[1]
    # The perturbation moves the exact product by at most dA @ |B|; rounding
    # the perturbed product adds gamma_k |A'| @ |B| <= gamma_k (|A| + dA) @ |B|.
    return d @ b + gamma(k, dtype) * ((a + d) @ b)
```

The bounds take the absolute sum as an argument rather than the vectors: the caller already has $\lvert x\rvert^\top\lvert y\rvert$ (or computes it once in float64), and the bound then works for any shape by broadcasting. Absolute values are taken *before* the product (`np.abs(A) @ np.abs(B)`), never after, because $\lvert AB\rvert$ is the cancelling quantity. For the quantization chapter, `dA = abs(W - dequantize(q))` is a full matrix, which is why `dA` is broadcast to $A$'s shape rather than required to be a scalar.

A check needs slack for its own reference. The expected side was also computed in floating point, so the comparison allows `slack` times the bound, and it treats NaN-against-NaN and identical infinities as equal (their difference is NaN):

```python
def assert_close_bounded(actual, expected, k, dtype, abs_dot, slack=4.0) -> None:
    # ... argument checks: slack > 0, shapes equal ...
    allowed = slack * np.broadcast_to(dot_error_bound(k, dtype, abs_dot), a.shape)
    with np.errstate(invalid="ignore"):
        err = np.abs(a - e)                    # inf - inf = nan, which fails <=
        ok = (err <= allowed) | (np.isnan(a) & np.isnan(e)) | (a == e)
    if ok.all():
        return
    i = tuple(int(j) for j in np.argwhere(~ok)[0])     # report the first failure
    ratio = err[i] / allowed[i] if allowed[i] > 0 else math.inf
    raise AssertionError(...)  # count, index, actual, expected, |error|, allowed, ratio, dtype


def bound_ratio(actual, expected, bound) -> float:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    b = np.broadcast_to(np.asarray(bound, dtype=np.float64), a.shape)
    if a.size == 0:
        return 0.0
    err = np.abs(a - e)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(err == 0, 0.0, err / b)   # 0/0 is "exact", x/0 is inf
    return float(np.max(r))
```

The three-way `ok` is what makes the check usable on overflowed kernels: two NaNs agree, two equal infinities agree (`a == e`), and everything else must be within the allowance. `bound_ratio` answers the second question a budget raises: not just pass or fail, but how close to the edge, so a ratio of $10^{-4}$ tells you the test could not catch a dropped term.

`cond` takes singular values and declares numerical singularity. $\kappa_2 = \sigma_{\max}/\sigma_{\min}$, and below $\max(m, n)\,2^{-52}\sigma_{\max}$ the smallest singular value is rounding noise, so the answer is $\infty$:

```python
def cond(A: ArrayLike) -> float:
    a = np.asarray(A, dtype=np.float64)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(f"cond needs a finite, non-empty 2-D array, got shape {a.shape}")
    s = svd(a)[1]                              # singular values, descending
    smax, smin = float(s[0]), float(s[-1])
    if smax == 0.0 or smin <= max(a.shape) * 2.0**-52 * smax:
        return math.inf                        # singular to working precision
    return smax / smin


def relative_condition(f, df, x: float) -> float:
    num = abs(x * df(x))
    if num == 0.0:
        return 0.0                             # nothing moves, even if f(x) = 0
    fx = abs(f(x))
    return math.inf if fx == 0.0 else num / fx
```

`cond` uses singular values, not eigenvalues, because the 2-norms of $A$ and $A^{-1}$ are $\sigma_{\max}$ and $1/\sigma_{\min}$; the non-normal example above has eigenvalues $1, 1$ and $\kappa_2 \approx 10^6$. The threshold is the same rank rule numpy's `matrix_rank` uses. `relative_condition` handles $f(x) = 0$ explicitly: $x - 1$ at $x = 1$ is infinitely ill-conditioned, while $f'(x) x = 0$ means a zero response whatever $f$ is.

Finally, the finite-difference step is the minimum of a two-term model: write truncation plus rounding, differentiate in $h$, solve. The code is those two closed forms:

```python
def fd_error_model(h, order, dtype, f_scale=1.0, deriv_scale=1.0) -> float:
    if not h > 0:
        raise ValueError(f"h must be > 0, got {h}")
    u = unit_roundoff(dtype)
    if order == 1:
        return deriv_scale * h / 2.0 + 2.0 * u * f_scale / h      # E_1(h): D h/2 + 2uF/h
    if order == 2:
        return deriv_scale * h * h / 6.0 + u * f_scale / h        # E_2(h): D h^2/6 + uF/h
    raise ValueError(f"order must be 1 (forward) or 2 (central), got {order!r}")


def optimal_fd_step(order, dtype, f_scale=1.0, deriv_scale=1.0) -> float:
    # ... f_scale > 0 and deriv_scale > 0 ...
    u = unit_roundoff(dtype)
    if order == 1:
        # d/dh (D h / 2 + 2 u F / h) = D / 2 - 2 u F / h^2 = 0
        return 2.0 * math.sqrt(u * f_scale / deriv_scale)
    if order == 2:
        # d/dh (D h^2 / 6 + u F / h) = D h / 3 - u F / h^2 = 0
        return (3.0 * u * f_scale / deriv_scale) ** (1.0 / 3.0)
    raise ValueError(f"order must be 1 (forward) or 2 (central), got {order!r}")
```

The model and the optimum are written as a pair so that one can be checked against the other: `fd_error_model(optimal_fd_step(o, d), o, d)` must be no larger than the model at any other step. In float64 with unit scales the two steps are $2\sqrt{2^{-53}} \approx 2.1 \times 10^{-8}$ and $(3 \cdot 2^{-53})^{1/3} \approx 6.9 \times 10^{-6}$, the numbers derived above. The full solution is in [`tolerance.py`](../../../../course/solutions/python/tinyllm/num/tolerance.py).

### Why this design

Every bound is $O(1)$ on top of the caller's absolute sums, except `matmul_error_bound`, which is one or two extra $O(mkn)$ float64 matmuls: the same order as the product it checks. `cond` is an SVD, $O(mn \min(m, n))$, which is fine for test-sized matrices.

The design prefers a worst-case bound to a statistical one. The $\sqrt{k}\,u$ rule is tighter and nearly always right; this bound is looser and always right. For tests on kernels you cannot easily debug (tiled C, quantized), a false failure is expensive, and a bound with a median bound/error ratio under 100 still catches a dropped term. It is elementwise, not normwise: a normwise bound $\gamma_k\lVert A\rVert\lVert B\rVert$ is simpler but lets a single bad element hide behind large ones, while the elementwise $\gamma_k(\lvert A\rvert\lvert B\rvert)_{ij}$ checks every output. And it uses an SVD rather than a condition estimator. LAPACK's `xGECON` estimates $\kappa_1$ in $O(n^2)$ after an LU factorization, which is faster but approximate and in a different norm; the exact $\kappa_2$ matches the theorem above and reuses the existing `svd`.

One tradeoff is accepted: the bounds ignore underflow (a subnormal partial sum has an absolute, not relative, error). For the matrices in these books that term is negligible, and adding it would make every formula carry an extra $k\eta$.

## What can go wrong

The tests check each claim of the section. The two hand examples pin the definitions ($u$, $\gamma_3$, the dot bound) and the meaning of a condition number ($\kappa_2 \approx 40002$, amplification 28284). The unit roundoff table covers six formats and the numpy names, with $u = \epsilon/2$, since every bound scales with it. $\gamma_k$ is checked at its edges: $\gamma_0 = 0$, the exact formula, and void at $ku \ge 1$, which is why bf16 sums of 256 terms have no bound. The dot bound must hold on 1000 float32 dots and must not be loose, with a median bound/error ratio under 100, so the tolerance never false-fails and still catches errors. The sum bound must hold for float16 running sums within $\gamma_{k-1}\sum\lvert x\rvert$, with $k = 1$ exact. Bounds are elementwise arrays (array in, float64 array out), because kernel tests pass whole matrices.

The matmul bound must cover any order: BLAS, long sequential sums, and cancellation all stay inside, since the tiled C kernel sums in its own order. The budget must include the perturbation: quantized weights pass with $dA = s/2$ and fail without it. `assert_close_bounded` must pass a correct dot, fail one with a dropped term, and scale with slack, and it must handle NaN, infinities, zero bounds, and shape mismatches so that overflowed kernels compare sanely. `bound_ratio` must report the worst element, with $0/0 = 0$ and $x/0 = \infty$, since the C parity check reports one number per operation. `cond` must match LAPACK on square, tall, wide, Hilbert, and non-normal matrices, must handle the identity, scale invariance, orthogonal matrices, and singular matrices (giving $\infty$, never a meaningless $10^{17}$), and must bound the amplification, with the bound reached at the singular directions. `relative_condition` is checked on $\sqrt{x}$, $\log$, $x - 1$ near 1, and zeros. The finite-difference model and both optimal steps are checked for minimality, and on real differences the central difference of the calculus chapter must be best near $h^*$, showing that the model predicts real behavior.

The common mistakes:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| using $\epsilon = 2^{1-p}$ for $u$, or forgetting the implicit bit | every tolerance off by 2 | stores $p$ with the implicit bit and returns $2^{-p}$ |
| $\gamma_k = ku$ without the denominator | a bound that stays finite where it is void | `gamma` returns $ku/(1 - ku)$ and raises at $ku \ge 1$ |
| the wrong count of roundings, or $\lvert A B\rvert$ for $\lvert A\rvert\lvert B\rvert$ | a correct kernel fails on long or cancelling rows | $\gamma_{k-1}$ for sums, $\gamma_k$ for dots, and absolute values before the product |
| a budget without the deliberate perturbation | a correct int4 kernel fails its test | `matmul_error_bound` adds `d @ b` and uses `(a + d) @ b` |
| ignoring the slack on the expected side | flaky failures at long $k$ | `assert_close_bounded` multiplies by `slack` (default 4) |
| treating a numerically singular matrix as finite | $\kappa = 10^{17}$ reported as a number | `cond` returns $\infty$ below the rank threshold |
| eigenvalues instead of singular values | non-normal matrices look well conditioned | `cond` reads `svd(a)[1]` |
| the absolute condition $\lvert f'/f\rvert$ | a scale-dependent number | `relative_condition` multiplies by $x$ |
| $\sqrt{u}$ for the central difference | a step 100 times too small, ten times the error | `optimal_fd_step` returns $(3uF/D)^{1/3}$ for order 2 |

## Where this leads

The main consumer is quantization. In *AI Engineering*, `output_error_bound(w, q, x)` in [Quantization: int8, int4 groups, fp8, and the KV cache](../../ai-engineering/21-inference/05-quantization.md) is `matmul_error_bound(W, x^T, "f32", dA=abs(W - dequantize(q)))`: the quantization step (at most `scale/2`) plus the f32 rounding. The optional [Cache-blocked, packed, batch-invariant matmul in C](../../ai-engineering/22-kernels/03-tiled-batch-invariant-matmul.md) checks its standalone C matmul against fixture values within `matmul_error_bound`, and states its own differential test with these bounds. Any differential test that compares two orders of summation can use them too. The next section, [FP8 and microscaling formats](04-fp8-and-microscaling-formats.md), supplies the $u$ values for the 8-bit formats in the precision table.

Production tools take different points on the same spectrum. PyTorch's `torch.testing.assert_close` uses per-dtype default tolerances with NaN and device handling: a fixed table, not a bound (`torch/testing/_comparison.py`). Probabilistic error analysis gives $\lambda\sqrt{k}\,u$ bounds that hold with probability $1 - \delta$. `numpy.linalg.cond` and LAPACK's `xGECON` estimate $\kappa_1$ in $O(n^2)$ from an LU factorization instead of an SVD (`dgecon.f`). JAX and PyTorch gradcheck use central differences at a fixed $\epsilon \approx 10^{-6}$ in float64 (`torch/autograd/gradcheck.py`), close to the optimal step derived here.

### Further reading

- Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM, 2nd ed.), chapters 2 to 4 and 7.
- Trefethen and Bau, *Numerical Linear Algebra*, lectures 12 to 15.
- Higham and Mary, "A New Approach to Probabilistic Rounding Error Analysis" (SIAM J. Sci. Comput., 2019).

## Key terms

- **Unit roundoff**: $u = 2^{-p}$, the largest relative error of one rounding; half of machine epsilon.
- **$\gamma_k$**: $ku/(1 - ku)$, a rigorous bound on the relative error accumulated over $k$ roundings, valid for $ku < 1$.
- **Tolerance budget**: a test allowance that sums every deliberate perturbation and the rounding bound, and nothing else.
- **Backward error**: how far the input must move for the computed output to be the exact answer.
- **Condition number**: how much the exact answer of a problem moves relative to a change in its input.
- **Optimal finite-difference step**: the step that balances truncation error against rounding error.

## Summary

- One model of rounding, $\mathrm{fl}(a \circ b) = (a \circ b)(1 + \delta)$ with $\lvert\delta\rvert \le u$, chained through $k$ operations, gives $\gamma_k = ku/(1 - ku)$, and with it a bound $\gamma_k \lvert x\rvert \cdot \lvert y\rvert$ on a dot product's error that holds for every input and every summation order.
- The bound scales with $\lvert x\rvert \cdot \lvert y\rvert$, not with $\lvert x \cdot y\rvert$: when terms cancel, the relative error of the result can be huge while the algorithm is perfectly fine.
- The condition number measures the problem, not the algorithm: $\kappa_2(A) = \sigma_{\max}/\sigma_{\min}$ bounds how much a relative change in $b$ moves the solution of $Ax = b$, and the bound is reached.
- A tolerance budget adds what you introduce on purpose (quantization moves each weight by up to half a step) to what rounding adds; a test that only allows rounding fails a correct quantized kernel.
- Finite differences trade truncation (falls with $h$) against rounding (grows as $1/h$); the optimal step is $\approx 2\sqrt{u}$ forward and $(3u)^{1/3}$ central.

## Review questions

1. Why does `sum_error_bound` use $\gamma_{k-1}$ while `dot_error_bound` uses $\gamma_k$?

   <details><summary>Answer</summary>

   A sum of $k$ exact inputs performs $k - 1$ additions, and the first term never passes through a rounding of its own, so each term meets at most $k - 1$ roundings. A dot product first rounds each product $x_i y_i$, one more rounding per term, so the count is $k$.

   </details>

2. Why is the matmul bound computed from $\lvert A\rvert\,\lvert B\rvert$ and not from $\lvert AB\rvert$?

   <details><summary>Answer</summary>

   Each rounding is relative to a partial sum, and partial sums can be large when the result is small. In $[1, 10^{-8}, -1] \cdot [1, 1, 1]$ the exact result is $10^{-8}$ but the partial sums are of size 1, and float32 loses the whole result. $\lvert AB\rvert$ would give an allowance near $10^{-15}$ and fail correct arithmetic; $\lvert A\rvert\lvert B\rvert = 2$ gives $3.6 \times 10^{-7}$.

   </details>

3. In `matmul_error_bound` with a perturbation, why does `dA` appear inside the $\gamma_k$ term as well as on its own?

   <details><summary>Answer</summary>

   Two errors are being added. The deliberate move changes the exact product by at most $dA\,\lvert B\rvert$. The kernel then multiplies the moved matrix $A'$ in floating point, and its rounding error is bounded by $\gamma_k\lvert A'\rvert\lvert B\rvert$; since $\lvert A'\rvert \le \lvert A\rvert + dA$, the rounding term uses $\lvert A\rvert + dA$. Leaving `dA` out of the second term undercounts the rounding of large moved weights.

   </details>

4. Why does `cond` return $\infty$ instead of $\sigma_{\max}/\sigma_{\min}$ when $\sigma_{\min}$ is tiny?

   <details><summary>Answer</summary>

   The SVD itself is computed with backward error about $u\lVert A\rVert$, so any singular value below roughly $\max(m, n)\,u\,\sigma_{\max}$ is indistinguishable from 0. Dividing by it produces a number like $10^{17}$ whose digits mean nothing; $\infty$ states the true conclusion, that the matrix is singular to working precision.

   </details>

5. Why is the optimal central-difference step $(3u)^{1/3}$ and not $\sqrt{u}$?

   <details><summary>Answer</summary>

   The truncation error of the central difference is quadratic in $h$ ($h^2 D/6$), while rounding still grows as $uF/h$. Balancing $h^2$ against $1/h$ gives $h^3 \propto u$, so $h^* \propto u^{1/3}$. $\sqrt{u}$ is the forward-difference answer, where truncation is linear; used for the central difference it is about 100 times too small in float64, and rounding dominates.

   </details>
