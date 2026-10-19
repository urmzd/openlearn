# Stable numerics: logsumexp, softmax, compensated sums

## Overview

| | |
|---|---|
| **Module** | `M09.2` · build · Python · Pass 2 · about 1 to 1.5 h reading |
| **Solution** | [`stable.py`](../../../../course/solutions/python/tinyllm/num/stable.py): `logsumexp`, `softmax`, `log_softmax`, `kahan_sum`, `pairwise_sum` |
| **Contract** | [`course/contracts/py/tinyllm/num/stable.pyi`](../../../../course/contracts/py/tinyllm/num/stable.pyi) |
| **Needs** | no code dependency · reading: `M09.1` IEEE 754 (why `exp` overflows float32 at 88.7), `M02.1` Taylor series |
| **Used by** | `M11.1` entropy from logits · `M08.3` the cross-entropy VJP · `L0.2` op library · `L0.3` fused cross-entropy · later: `L8.1` sampler, `L6.7` perplexity over $10^7$ tokens, `L9.2` the same math in C · later: `L4.4`, `L5.1`, `L7.7`; all on the way to the Pass 2 result, a bigram trained by the course's own autograd |
| **Optional depth** | Higham, *Accuracy and Stability of Numerical Algorithms* (SIAM, 2nd ed.), ch. 1 and 4; Blanchard, Higham, and Higham, "Accurately computing the log-sum-exp and softmax functions" (IMA J. Numer. Anal., 2021) |

## Key Takeaways

- $\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$ with $m = \max_i x_i$: after the shift the largest exponential is $e^0 = 1$, so nothing overflows, and the answer is finite for logits of any size.
- `log_softmax` is `x - logsumexp(x)`, never `log(softmax(x))`: the second takes the log of an underflowed 0 and returns $-\infty$ where the answer is $-10^4$.
- A fully masked row (all $-\infty$) is defined, not NaN: softmax gives zeros, log-sum-exp and log-softmax give $-\infty$.
- Plain summation of $n$ floats can be off by about $n u \sum|x_i|$; pairwise summation cuts that to $\lceil\log_2 n\rceil u \sum|x_i|$ and Kahan's compensation to about $2u|S|$, independent of $n$.

## How to read this chapter

Sections 1 to 3 derive the shift identity and the summation error bounds from first principles and work them by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Your tracer bigram (`L0.0`) already hides a numerically careful softmax: its `nll` subtracts each row's maximum before `exp`. In this pass that one line becomes a dozen call sites. Your autograd op library (`L0.2`) needs softmax and log-softmax as ops, the fused cross-entropy (`L0.3`) needs log-softmax at the target, the sampler (`L8.1`) turns temperature-scaled logits into probabilities, and the model zoo (`L6.7`) sums ten million per-token losses into one perplexity. Written the obvious way, each of them fails the first time training succeeds: a confident model produces a logit above 88.7, `np.exp` overflows float32 to `inf`, and the loss becomes `inf / inf = nan`; a confidently wrong token gets probability $e^{-200} = 0$ in float32, and `log(0)` is an infinite loss with a NaN gradient. This module writes the stable versions once, proves them, and makes every later module import them.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^n$ | one row of logits (scores) | `float[n]` |
| $m = \max_i x_i$ | the shift that prevents overflow | scalar |
| $\mathrm{LSE}(x) = \log\sum_i e^{x_i}$ | log-sum-exp | scalar |
| $\mathrm{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$ | probabilities from logits | `float[n]`, sums to 1 |
| $\mathrm{logsoftmax}(x)_i = x_i - \mathrm{LSE}(x)$ | log-probabilities from logits | `float[n]` |
| $u$ | unit roundoff: $2^{-24} \approx 6.0 \times 10^{-8}$ for float32, $2^{-53} \approx 1.1 \times 10^{-16}$ for float64 | scalar |
| $\mathrm{fl}(a \circ b)$ | the floating-point result of an operation $\circ$ | scalar |
| $S = \sum_{i=1}^n x_i$ | the exact sum | scalar |
| $\hat S$ | the computed sum | scalar |
| $c$ | Kahan's compensation: the low-order part lost so far | scalar |

**Where exp breaks.** `M09.1` showed that the largest float32 is about $3.4 \times 10^{38} = e^{88.72}$ and the largest float64 about $e^{709.78}$. So `np.exp(89.0)` in float32 is `inf`. At the other end, $e^x$ underflows to exactly 0 below about $-103.9$ in float32 (counting subnormals) and $-745.1$ in float64. Logits after training routinely exceed 100 in magnitude, and masks write $-\infty$ on purpose.

**The shift identity.** For any constant $c$, $\sum_i e^{x_i} = e^{c} \sum_i e^{x_i - c}$, so

$$\mathrm{LSE}(x) = c + \log \sum_i e^{x_i - c}.$$

Choose $c = m = \max_i x_i$. Every shifted exponent $x_i - m$ is at most 0, so every term is at most 1, and the largest term is exactly $e^0 = 1$. The sum lies in $[1, n]$: it cannot overflow, it is never 0, and its log lies in $[0, \log n]$. Nothing is lost by the shift, because the identity is exact.

**Softmax is shift invariant.** Dividing by the sum cancels the factor $e^{c}$:

$$\mathrm{softmax}(x + c)_i = \frac{e^{x_i + c}}{\sum_j e^{x_j + c}} = \frac{e^{c} e^{x_i}}{e^{c} \sum_j e^{x_j}} = \mathrm{softmax}(x)_i.$$

So compute $e^{x_i - m} / \sum_j e^{x_j - m}$. The same identity lets the C kernel in `L9.2` process a row in one pass: when a new maximum $m'$ appears, it rescales the running sum by $e^{m - m'}$ instead of starting over.

**Log-softmax from the identity, not from softmax.** $\log \mathrm{softmax}(x)_i = x_i - \mathrm{LSE}(x)$. Computed this way it is a subtraction of two moderate numbers. Computed as `log(softmax(x))`, it first rounds $e^{x_i - m}$, which underflows to 0 once $x_i - m < -104$ in float32, and then takes `log(0) = -inf`. The cross-entropy loss is $-\mathrm{logsoftmax}(x)_t$ at the target $t$, so this is the difference between a large finite loss (with a useful gradient) and an infinite one.

**Masks.** A masked entry is $x_i = -\infty$, so $e^{x_i - m} = 0$ and it gets probability exactly 0. If every entry is masked, $m = -\infty$ and $x_i - m = -\infty - (-\infty)$ is NaN. The contract defines that row instead of propagating NaN: replace a non-finite $m$ by 0, so the shifted entries stay $-\infty$, the sum is 0, softmax returns zeros (divide by 1 when the sum is 0), and log-sum-exp returns $\log 0 = -\infty$. A padded position in a batch is exactly such a row.

**Rounding in a sum.** Every floating addition rounds: $\mathrm{fl}(a + b) = (a + b)(1 + \delta)$ with $|\delta| \le u$. Summing left to right, the $k$-th partial sum carries the rounding errors of all earlier additions, and the first terms pass through $n - 1$ additions. The standard bound is

$$|\hat S - S| \le (n - 1)\, u \sum_i |x_i| + O(u^2).$$

For $n = 10^7$ float32 values that is a relative error up to about $0.6$: useless. In practice errors partly cancel, but when the addends have the same sign (losses, counts, probabilities) they accumulate.

**Pairwise summation.** Split the array in halves, sum each half the same way, add the two results. The recursion is $\lceil \log_2 n \rceil$ levels deep, and each element takes part in only one addition per level, so

$$|\hat S - S| \le \lceil \log_2 n \rceil\, u \sum_i |x_i| + O(u^2).$$

For $n = 10^7$ that is 24 instead of $10^7$. numpy's `np.sum` does this internally (with blocks of 8 at the leaves), which is why `np.cumsum`, a strictly sequential sum, is the one that shows the problem.

**Kahan's compensated summation.** Keep a second variable $c$ holding the part the running sum $s$ could not absorb. For each $x$:

$$y = x - c, \qquad t = s + y, \qquad c = (t - s) - y, \qquad s = t.$$

When $|s| \ge |y|$, $t - s$ is computed exactly, and it is the part of $y$ that actually made it into $t$; subtracting $y$ leaves minus the part that was rounded away. Algebraically $c = 0$, which is the point: in floating point it is the rounding error, recovered exactly and fed back into the next addend. The error bound becomes

$$|\hat S - S| \le 2u|S| + O(n u^2) \sum_i |x_i|,$$

which no longer grows with $n$ to first order. It costs four operations per element and a sequential loop, so it is used where the count is huge and the budget is not: running totals of losses, token counts, and perplexity.

**Work in the input's dtype.** The point of compensation is to get extra precision without a wider type. Both sums here keep every operation in the dtype of $x$ (float32 stays float32) and return a Python float.

## 3. Worked example by hand

**Log-sum-exp and softmax of $x = [1, 2, 3]$.**

| step | values |
|---|---|
| $m = \max x$ | 3 |
| $x - m$ | $[-2, -1, 0]$ |
| $e^{x - m}$ | $[0.135335, 0.367879, 1]$ |
| sum | $1.503215$ |
| $\log$ sum | $0.407606$ |
| $\mathrm{LSE} = m + \log$ sum | $3.407606$ |
| softmax $= e^{x-m} / $ sum | $[0.090031, 0.244728, 0.665241]$ |
| log-softmax $= x - \mathrm{LSE}$ | $[-2.407606, -1.407606, -0.407606]$ |

The softmax row sums to 1, and $e^{-0.407606} = 0.665241$ confirms the last entry. Adding 1000 to every entry changes $m$ to 1003 and leaves every shifted value, and so the softmax, unchanged.

**A masked row** $[0, -\infty, 0]$: $m = 0$, shifted $[0, -\infty, 0]$, exponentials $[1, 0, 1]$, softmax $[0.5, 0, 0.5]$, log-softmax $[-\ln 2, -\infty, -\ln 2]$. **A fully masked row** $[-\infty, -\infty, -\infty]$: $m$ becomes 0, every exponential is 0, softmax is $[0, 0, 0]$, and log-sum-exp is $-\infty$.

**Summing $[2^{24}, 1, 1, -2^{24}]$ in float32.** At $2^{24} = 16777216$ consecutive float32 values are 2 apart, so $2^{24} + 1 = 16777217$ is exactly halfway between two floats and rounds to the even one, $16777216$. The exact sum is 2.

- Left to right: $2^{24} + 1 \to 2^{24}$, $+ 1 \to 2^{24}$, $- 2^{24} \to 0$. Both ones are lost.
- Pairwise: $(2^{24} + 1) + (1 - 2^{24}) = 16777216 + (-16777215) = 1$. The left pair loses its 1; $1 - 2^{24} = -16777215$ is representable, so the right pair keeps its 1. Better, not exact.
- Kahan, one row per element:

| $x$ | $y = x - c$ | $t = s + y$ | $c = (t - s) - y$ | $s$ |
|---|---|---|---|---|
| $2^{24}$ | $2^{24}$ | $2^{24}$ | 0 | $2^{24}$ |
| 1 | 1 | $2^{24}$ (rounded) | $(2^{24} - 2^{24}) - 1 = -1$ | $2^{24}$ |
| 1 | $1 - (-1) = 2$ | $2^{24} + 2$ (exact) | $2 - 2 = 0$ | $2^{24} + 2$ |
| $-2^{24}$ | $-2^{24}$ | 2 | $(2 - (2^{24} + 2)) + 2^{24} = 0$ | 2 |

The second row is the whole idea: the 1 that rounding threw away reappears as $c = -1$, and the next step adds it back. Kahan returns the exact 2.

In float64 the same thing happens one scale up: at $10^{16}$ floats are 2 apart, so plain summation of $[10^{16}, 1, 1, -10^{16}]$ gives 0. Pairwise also gives 0 there, because $1 - 10^{16}$ is a tie too and rounds back to $-10^{16}$; Kahan gives 2.

These numbers are the first cases in section 4.

## 4. The interface

```python
# python/tinyllm/num/stable.py
def logsumexp(x: ArrayLike, axis: int = -1, keepdims: bool = False) -> NDArray
def softmax(x: ArrayLike, axis: int = -1) -> NDArray
def log_softmax(x: ArrayLike, axis: int = -1) -> NDArray
def kahan_sum(x: ArrayLike) -> float
def pairwise_sum(x: ArrayLike) -> float
```

Float inputs keep their dtype; integer inputs are computed in float64. `axis` and `keepdims` mean what they mean in numpy. The sums flatten their input in C order. `pairwise_sum` splits at `h = n // 2` down to single elements; pairing neighbours bottom up builds the same tree when $n$ is a power of two, so a vectorized version is fine. The contract has the full rules.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example | unit | the section 3 table for $[1, 2, 3]$ | pins the definitions |
| Hand example, sums | unit | float32 $[2^{24}, 1, 1, -2^{24}]$: plain 0, Kahan 2, pairwise 1 | the three sums differ exactly as derived |
| scipy golden values | golden | scipy's three functions on seven shapes, axes, and scales | an independent implementation agrees |
| Shift invariance | property | adding $c \in [-1000, 1000]$ changes nothing but LSE by $c$ | the online softmax of `L9.2` rescales a running sum |
| No overflow | boundary | $[1000, 1000]$ gives $[0.5, 0.5]$; $\pm 10^4$ stays finite | logits after training |
| No underflow in log-softmax | boundary | $[0, -10^4]$ gives $[0, -10^4]$, not $-\infty$ | the cross-entropy of `L0.3` |
| Fully masked row | boundary | all $-\infty$ gives zeros and $-\infty$, no NaN, no warning | padded rows in a batch |
| Axis and keepdims | unit | every axis of a 3-D array, keepdims shapes | attention over `[B, H, T, T]` |
| dtype preserved | unit | float32 in, float32 out; integers become float64 | kernels compare in float32 |
| Rows sum to one | property | rows of scale 0.1 to 700 sum to 1 | the sampler's CDF ends at 1 |
| Kahan error bound | property | 65536 float32 values within $2u\lvert S\rvert$; plain summation is not | `L6.7` sums $10^7$ losses |
| Pairwise error bound | property | within $\lceil\log_2 n\rceil u \sum \lvert x\rvert$ for $n = 65536$ and $50001$ | the cheap accurate sum |
| Every element covered | boundary | lengths 0, 1, odd, a 2-D array, integers | no dropped or invented term |
| Input dtype does the work | unit | the float64 version of the hand example | compensation, not a wider type, does the work |

## 5. The solution, step by step

### From first principles

1. The shift identity is exact for any constant (section 2), and the maximum makes every term at most 1. So every function starts with the row maximum, kept as a size-1 axis so it broadcasts back.
2. A fully masked row has maximum $-\infty$, and $-\infty - (-\infty)$ is NaN. Replacing a non-finite maximum by 0 keeps the shifted entries at $-\infty$, their exponentials at 0, and the arithmetic defined.
3. Log-sum-exp is then $\log \sum e^{x - m} + m$; softmax is $e^{x - m}$ divided by its sum, with a zero sum (only possible for a fully masked row) replaced by 1.
4. Log-softmax is $x - \mathrm{LSE}(x)$, computed from the log-sum-exp, never from the softmax. A fully masked row has $\mathrm{LSE} = -\infty$, and the result there is $-\infty$ everywhere.
5. The sums flatten the input and keep its float dtype. Kahan is the four-line loop of section 2 with $s$ and $c$ in that dtype. Pairwise is a recursion that halves the range, so its depth is $\lceil\log_2 n\rceil$.

### The code, annotated

The two helpers:

```python
def _float(x):
    """x as a float array: float dtypes kept, everything else float64."""
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)       # integers: exp of an int array is float64 anyway
    return a                           # float32 stays float32: no silent upcast


def _safe_max(a, axis):
    """max along axis with keepdims, replaced by 0 where it is not finite, so
    a fully masked row (max = -inf) subtracts 0 instead of making -inf - -inf."""
    m = np.max(a, axis=axis, keepdims=True)
    return np.where(np.isfinite(m), m, np.zeros_like(m))
```

The three row functions:

```python
def logsumexp(x, axis=-1, keepdims=False):
    a = _float(x)
    m = _safe_max(a, axis)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = np.sum(np.exp(a - m), axis=axis, keepdims=True)  # in [1, n], or 0 if masked
        out = np.log(s) + m            # add the shift back: LSE = m + log sum
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(x, axis=-1):
    a = _float(x)
    m = _safe_max(a, axis)
    with np.errstate(over="ignore", invalid="ignore"):
        e = np.exp(a - m)              # every entry <= 1, the largest exactly 1
        s = np.sum(e, axis=axis, keepdims=True)
        # A fully masked row has s = 0 and e = 0: divide by 1, keep the zeros.
        return e / np.where(s == 0, np.ones_like(s), s)


def log_softmax(x, axis=-1):
    a = _float(x)
    lse = logsumexp(a, axis=axis, keepdims=True)  # never log(softmax(x))
    with np.errstate(invalid="ignore"):
        out = a - lse
    # -inf - (-inf) is nan; a fully masked row is log 0 = -inf everywhere.
    return np.where(np.isneginf(lse), np.full_like(out, -np.inf), out)
```

The `errstate` blocks silence the warnings numpy would print for the masked cases (`log(0)`, `inf - inf`), which the code then handles explicitly; the row of all $-\infty$ is a defined input, not an error. On $[1, 2, 3]$: `m = 3`, `exp([-2, -1, 0])` sums to $1.503215$, its log $0.407606$ plus 3 gives $3.407606$, and `log_softmax` subtracts that from each entry.

The two sums:

```python
def kahan_sum(x):
    a = _float(x).ravel()              # C order, any shape
    zero = a.dtype.type(0)             # s and c in the input's dtype
    s, c = zero, zero
    for v in a:
        y = v - c  # the addend, corrected by what the last step lost
        t = s + y  # the low bits of y are lost here...
        c = (t - s) - y  # ...and recovered here (algebraically 0)
        s = t
    return float(s)


def pairwise_sum(x):
    a = _float(x).ravel()

    def rec(lo, hi):
        n = hi - lo
        if n == 1:
            return a[lo]               # a numpy scalar: additions stay in a's dtype
        h = lo + n // 2                # split in the middle: depth ceil(log2 n)
        return rec(lo, h) + rec(h, hi) # [lo, h) and [h, hi) cover every element once

    if a.size == 0:
        return 0.0                     # the empty sum; rec would index a[0]
    return float(rec(0, a.size))
```

`zero = a.dtype.type(0)` matters: starting from the Python float `0.0` would leave the dtype of `s` and `c` to numpy's promotion rules for Python scalars, which changed between versions (NumPy 1 promoted `0.0 + np.float32(v)` to float64). A float64 accumulator would silently remove the float32 rounding that Kahan is meant to correct. On the section 3 row, the loop reproduces the Kahan table: after the second element `c = -1`, and the third step adds it back.

Full solution: [`stable.py`](../../../../course/solutions/python/tinyllm/num/stable.py).

### Why this solution

- **Cost.** The row functions are one max, one exp, and one sum per row: $O(n)$, with a few temporaries the size of the input. `kahan_sum` is a Python loop, $O(n)$ but slow per element; `pairwise_sum` is $O(n)$ additions with $O(\log n)$ recursion depth.
- **Why the max and not another shift.** Any shift keeps the identity exact, but only the max guarantees the largest term is 1 (no overflow) and the sum at least 1 (no $\log 0$ for an unmasked row).
- **Why not compute in float64.** Upcasting hides the problem in Python but not in the C kernels of `L9.2`, which run in float32; the Python reference must round where they round. Compensation gets the accuracy without the wider type.
- **Why not `math.fsum`.** It is exactly rounded but keeps a growing list of partials and works only in float64. Kahan's bound $2u\lvert S\rvert$ is enough for loss totals and costs four operations per element.
- **Tradeoff accepted.** `kahan_sum` is an interpreted loop, fine for running totals of a few million numbers but not for inner loops. `pairwise_sum` recurses to single elements rather than using numpy's blocked leaves; it is a readable reference for the error bound, not the fastest sum.

### Explain it back

1. Why does `_safe_max` replace a non-finite maximum by 0, and why does that not change any finite result?

   <details><summary>Answer</summary>

   A finite maximum is left alone, so finite rows are unaffected. A row whose maximum is $-\infty$ is fully masked; subtracting it would give $-\infty - (-\infty) = $ NaN. Subtracting 0 instead leaves every entry $-\infty$, every exponential 0, and the shift identity still holds because any constant is a valid shift.

   </details>

2. Why does `log_softmax` call `logsumexp` instead of taking `np.log(softmax(x))`?

   <details><summary>Answer</summary>

   For $x = [0, -10^4]$, `softmax` returns $[1, 0]$ in float32 or float64, because $e^{-10^4}$ underflows to 0, and $\log 0 = -\infty$. $x - \mathrm{LSE}(x) = [0, -10^4] - 0$ is exact. The loss at that target is $10^4$, large but finite, with a useful gradient; the other route gives an infinite loss and NaN gradients.

   </details>

3. In `kahan_sum`, what would happen if `c = (t - s) - y` were computed but the next step used `y = v` instead of `y = v - c`?

   <details><summary>Answer</summary>

   The lost low-order bits would be measured and thrown away: the loop would be plain left-to-right summation with extra work, and on $[2^{24}, 1, 1, -2^{24}]$ it would return 0 instead of 2.

   </details>

4. Why does `pairwise_sum` split at `n // 2` instead of adding `a[lo]` to the sum of the rest?

   <details><summary>Answer</summary>

   Peeling one element at a time builds a tree of depth $n$: each early element passes through about $n$ additions, which is the sequential error bound again, and the recursion hits Python's limit at about 1000 elements. Halving gives depth $\lceil\log_2 n\rceil$, so each element passes through only that many roundings.

   </details>

5. Why does `kahan_sum` start from `a.dtype.type(0)` rather than `0.0`?

   <details><summary>Answer</summary>

   With a Python `0.0`, whether the loop runs in float32 or float64 depends on numpy's promotion rules for Python scalars (NumPy 1 gave float64). If it runs in float64, a wider type is doing the work, not compensation, and the result no longer matches the float32 C kernels. A zero of the input's dtype makes every operation stay in that dtype on any numpy version.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. exponentiating before subtracting the max | `inf / inf = nan` at a float32 logit of 89 | `np.exp(a - m)` with `m` from `_safe_max` |
| 2. `log(softmax(x))` | $-\infty$ log-probability, infinite loss, NaN gradient | `log_softmax` is `a - logsumexp(a)` |
| 3. subtracting a max of $-\infty$, or dividing by a zero sum | a padded row of NaN that spreads through the batch | `_safe_max` replaces a non-finite max by 0; softmax divides by 1 where the sum is 0 |
| 4. computing the compensation and never using it | Kahan silently becomes plain summation | `y = v - c` feeds the compensation into every addend |
| 5. a "pairwise" recursion that peels off one element at a time | a depth-$n$ tree, the sequential error, and a recursion limit | splits at `lo + n // 2` |
| 6. forgetting to add $m$ back in log-sum-exp | off by exactly the row maximum | `np.log(s) + m` |
| 7. dropping the middle element of an odd split, or indexing an empty array | off by a whole term | ranges `[lo, h)` and `[h, hi)`; an empty input returns 0.0 before recursing |
| 8. reducing the wrong axis, ignoring `keepdims` | attention normalized over the batch | every reduction uses the given `axis` with `keepdims=True`, squeezed only when asked |
| 9. upcasting float32 to float64 | the C kernel's float32 reference no longer matches | `_float` keeps float dtypes; Kahan starts from `a.dtype.type(0)` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.1` | the overflow and underflow thresholds of `exp`, and the unit roundoff $u$ in every bound |
| Back | `M02.1` | Taylor's view of $e^x$ near 0, behind why small shifted exponents are accurate |
| Forward | `M11.1` | `entropy_from_logits` uses `log_softmax` |
| Forward | `M08.3` | `cross_entropy_vjp` uses `softmax` |
| Forward | `L0.2` | softmax and log-softmax become autograd ops |
| Forward | `L0.3` | the fused cross-entropy is $-\mathrm{logsoftmax}$ at the target |
| Forward | `L8.1` | temperature-scaled logits to sampling probabilities |
| Forward | `L6.7` | perplexity over $10^7$ tokens accumulates with `kahan_sum` (through `M11.2`) |
| Forward | `L9.2` | the same math in C, in one pass with a running maximum |
| Forward | `L4.4` | beam search scores candidates with `log_softmax` |
| Forward | `L5.1` | scaled dot-product attention normalizes scores with `softmax` |
| Forward | `L7.7` | sliding-window attention with sinks combines scores with `logsumexp` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `softmax` | PyTorch `torch.softmax` | fused vectorized kernels, autocast to float32 inside half-precision models | `aten/src/ATen/native/SoftMax.cpp` |
| `logsumexp` | `scipy.special.logsumexp` | weights `b` (log of a weighted sum), sign handling for negative weights | `scipy/special/_logsumexp.py` |
| the shift identity | FlashAttention's online softmax | one pass over keys with a running max and a rescaled running sum | Dao et al., *FlashAttention* (2022), section 3.1 |
| `pairwise_sum` | numpy `np.add.reduce` | pairwise with 8-way unrolled leaf blocks, chosen for speed and accuracy | `numpy/_core/src/umath/loops_utils.h.src` |
| `kahan_sum` | Python `math.fsum` | Shewchuk's algorithm: the exactly rounded sum, at the cost of a list of partials | `Modules/mathmodule.c` (`math_fsum`) |
