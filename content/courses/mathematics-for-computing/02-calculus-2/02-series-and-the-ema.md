# Series convergence, the EMA as a geometric series, and bias correction

The loss a training loop prints jumps around from batch to batch, and what you care about is its trend. The standard tool for extracting that trend is an exponential moving average, and the first thing it does is lie: started from zero, it reports a loss of 0.55 when every batch says 5.5, and climbs toward the truth over hundreds of steps. Adam, the optimizer every model in *Build Your Own LLM System* trains with, keeps two such averages per parameter and would take wildly wrong first steps without the correction this section derives.

Both the lie and the fix are a fact about one infinite sum, the geometric series. This section defines what it means for a series to converge, sums the geometric series exactly, and builds the EMA and its bias correction on top of the `geometric` and `geometric_sum` helpers from [Sequences, geometric series, and frequency ladders](../00-precalculus/03-sequences-and-frequency-ladders.md). The notion of a limit comes from [Derivatives and finite differences](../01-calculus-1/01-derivatives-and-finite-differences.md).

The finished code is [`course/solutions/python/tinyllm/num/ema.py`](../../../../course/solutions/python/tinyllm/num/ema.py), which provides `ema_weights` and the class `EMA` (with `update`, `value`, `t`, and `value_debiased`); its interface is fixed by the contract [`course/contracts/py/tinyllm/num/ema.pyi`](../../../../course/contracts/py/tinyllm/num/ema.pyi).

> **Learning objectives**
> - Define convergence of a series through its partial sums and show why the harmonic series diverges.
> - Derive the partial sum $\frac{1 - r^t}{1 - r}$ of a geometric series and its limit for $|r| < 1$.
> - Unroll the EMA recursion into a weighted sum with geometric weights, and compute its effective window and half-life.
> - Derive the bias correction $\hat m_t = m_t / (1 - \beta^t)$ and explain why it requires the zero start.
> - Implement an `EMA` class that rejects non-finite input without corrupting its state and never leaks internal arrays.

## Series and convergence

The symbols used throughout are collected here first:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $a_0, a_1, a_2, \ldots$ | a sequence of numbers | |
| $S_t = \sum_{j=0}^{t-1} a_j$ | the $t$-th partial sum | `float` |
| $r$ | the ratio of a geometric sequence, $a_j = a r^j$ | `float` |
| $x_t$ | the input at step $t = 1, 2, \ldots$ | `float` or array |
| $\beta$ | the decay, $0 \le \beta < 1$ | `float` |
| $m_t$ | the EMA after $t$ updates, $m_0 = 0$ | `float` or array |
| $w_i^{(t)}$ | the weight of $x_i$ in $m_t$ | `float` |
| $\hat m_t$ | the bias-corrected average $m_t / (1 - \beta^t)$ | `float` or array |

A **series** $\sum_{j \ge 0} a_j$ is the sequence of its **partial sums** $S_t = a_0 + \cdots + a_{t-1}$. It **converges** to $S$ when the partial sums settle on $S$ in the sense of a limit: for every tolerance there is a $t$ beyond which every $S_t$ is within it. Otherwise it **diverges**.

Small terms are necessary but not enough. The harmonic series $1 + \frac12 + \frac13 + \frac14 + \cdots$ diverges, because the terms from $\frac{1}{2^k + 1}$ to $\frac{1}{2^{k+1}}$ are $2^k$ numbers each at least $\frac{1}{2^{k+1}}$, adding to at least $\frac12$ for every $k$, so the sum passes any bound.

A useful sufficient condition is the **ratio test**: if $|a_{j+1}/a_j|$ eventually stays below some $q < 1$, the series converges, because from there on its terms are smaller than those of a geometric series with ratio $q$. The previous section, [Taylor series, remainder bounds, and range reduction](01-taylor-series.md), used exactly this argument to bound the tail of the erf series. Everything therefore rests on the geometric series, which is where we turn next.

## The geometric series

For $a_j = a r^j$, multiply the partial sum by $r$ and subtract. Everything but the first and last terms cancels:

$$S_t - r S_t = a - a r^t \quad\Longrightarrow\quad S_t = a\,\frac{1 - r^t}{1 - r} \quad (r \neq 1), \qquad S_t = a t \quad (r = 1) .$$

If $|r| < 1$, $r^t \to 0$ and the series converges to $\frac{a}{1 - r}$; if $|r| \ge 1$ the terms do not shrink and it diverges.

There is one numerical catch. When $r$ is close to 1, $1 - r^t$ subtracts two nearly equal numbers. The `geometric_sum` helper evaluates it as $\frac{\mathrm{expm1}(t \ln r)}{r - 1}$ to keep full precision, and the code here calls it rather than re-deriving it.

## The EMA unrolled

The update $m_t = \beta m_{t-1} + (1 - \beta)x_t$ keeps a fraction $\beta$ of the old average and mixes in $1 - \beta$ of the new input. Unroll it from $m_0 = 0$ and the geometric series appears:

$$m_t = (1 - \beta)x_t + \beta(1 - \beta)x_{t-1} + \beta^2(1 - \beta)x_{t-2} + \cdots + \beta^{t-1}(1 - \beta)x_1 = \sum_{i=1}^{t} \underbrace{(1 - \beta)\beta^{t-i}}_{w_i^{(t)}}\, x_i .$$

The newest input has weight $1 - \beta$, and each step back in time multiplies the weight by $\beta$: a geometric sequence. `ema_weights(beta, t)` returns $w_1^{(t)}, \ldots, w_t^{(t)}$ oldest first, which is `geometric(1 - beta, beta, t)` reversed.

Two numbers summarize how far back the average looks. The **effective window** is $\frac{1}{1 - \beta}$ (the sum of the infinite weight sequence divided by its largest weight): 10 steps for $\beta = 0.9$, 1000 for $\beta = 0.999$. The **half-life**, after which an input's weight has halved, is $\frac{\ln \frac12}{\ln \beta}$: 6.6 steps for $\beta = 0.9$.

## Bias correction

Now the lie becomes visible. The weights add up to a geometric partial sum:

$$\sum_{i=1}^{t} w_i^{(t)} = (1 - \beta)\frac{1 - \beta^t}{1 - \beta} = 1 - \beta^t .$$

An average whose weights add to less than 1 is pulled toward 0, which is where it started. If every $x_i = c$, then $m_t = c(1 - \beta^t)$: after one step with $\beta = 0.9$, only $0.1c$. Dividing by the total turns the weights back into an average,

$$\hat m_t = \frac{m_t}{1 - \beta^t} ,$$

and for a constant input $\hat m_t = c$ at every $t$, exactly in real arithmetic (and to a few $\varepsilon/(1 - \beta)$ in float64). If the inputs are random with the same mean $\mu$, the expected value of $m_t$ is $(1 - \beta^t)\mu$, so $\hat m_t$ is an unbiased estimate of $\mu$. That is why Adam's update has the right size from its first step.

The correction matters only early. $\beta^t$ falls below 1 percent after $t \approx \frac{4.6}{1 - \beta}$ steps, and from then on $\hat m_t \approx m_t$.

Two consequences shape the code. First, the zero start is part of the definition: initializing $m$ with $x_1$ and then dividing by $1 - \beta^t$ corrects a bias that is not there ($\hat m_1 = 10\,x_1$ for $\beta = 0.9$). Second, at $t = 0$ the correction is $0/0$: nothing has been averaged, so `value_debiased` raises instead of returning a number.

## Working it by hand

Take $\beta = 0.9$ and inputs $x_1 = 1$, $x_2 = 2$, $x_3 = 3$. The recursion, the weight total, and the corrected value evolve like this:

| $t$ | $m_t = 0.9\,m_{t-1} + 0.1\,x_t$ | $1 - 0.9^t$ | $\hat m_t$ |
|---|---|---|---|
| 1 | $0.9 \cdot 0 + 0.1 \cdot 1 = 0.1$ | 0.1 | $0.1/0.1 = 1$ |
| 2 | $0.9 \cdot 0.1 + 0.1 \cdot 2 = 0.29$ | 0.19 | $0.29/0.19 = 1.5263158$ |
| 3 | $0.9 \cdot 0.29 + 0.1 \cdot 3 = 0.561$ | 0.271 | $0.561/0.271 = 2.0701107$ |

Unrolled, $m_3 = 0.081 \cdot 1 + 0.09 \cdot 2 + 0.1 \cdot 3 = 0.081 + 0.18 + 0.3 = 0.561$: the same number, with weights $[0.081, 0.09, 0.1]$ adding to $0.271 = 1 - 0.9^3$. The biased $m_3 = 0.561$ is below even the smallest input; the debiased 2.07 is a weighted average of 1, 2, 3 that leans toward the newest.

## Designing the interface

The module exposes one function and one class:

```python
def ema_weights(beta: float, t: int) -> NDArray: ...     # oldest first, sum 1 - beta**t
class EMA:
    def __init__(self, beta: float) -> None: ...         # 0 <= beta < 1, m_0 = 0
    t: int                                               # property: updates so far
    value: float | NDArray                               # property: m_t (a copy)
    def update(self, x: ArrayLike) -> float | NDArray: ...
    def value_debiased(self) -> float | NDArray: ...     # m_t / (1 - beta**t)
```

A number in gives a Python `float` out; an array in gives a new float64 array, averaged elementwise, and later updates must keep its shape. Non-finite input raises `ValueError` and leaves the state unchanged; `value_debiased` before any update raises `RuntimeError`.

## Building it step by step

Each design decision follows from the derivation above.

1. **The state is two numbers: $m_t$ and $t$.** The recursion $m_t = \beta m_{t-1} + (1-\beta)x_t$ needs only the previous average, and the correction $1 - \beta^t$ needs only the count. Keeping the whole history is unnecessary, since the unrolled form shows the recursion *is* the weighted sum.
2. **Start at $m_0 = 0$.** The bias correction is derived for that start. Seeding with $x_1$ and also dividing by $1 - \beta^t$ would correct a bias that is not there.
3. **The correction total comes from `geometric_sum`.** $\sum_i w_i^{(t)} = (1-\beta)\frac{1-\beta^t}{1-\beta}$ is exactly `geometric_sum(1 - beta, beta, t)`, which stays accurate for $\beta$ near 1 where $1 - \beta^t$ cancels.
4. **$t = 0$ has no average.** The debiased value would be $0/0$; raising is the only honest answer.
5. **Validate before mutating.** A non-finite input or a shape change raises *before* the state changes, so one bad batch leaves the curve intact. $\beta$ is checked once, at construction: $0 \le \beta < 1$ (with $\beta = 1$ the average never moves and the correction divides by 0).
6. **Never hand out internal state.** Returning the internal array would let a caller's in-place edit change the average, so every array returned is a copy, and scalars come back as Python `float`.
7. **The weights are the geometric sequence, reversed.** `geometric(1 - beta, beta, t)` lists $(1-\beta), (1-\beta)\beta, \ldots$, newest first; the contract lists $x_1$ (the oldest) first, so reverse it.

### Validating $\beta$ and listing the weights

```python
def _check_beta(beta: float) -> float:
    b = float(beta)
    # isfinite first, so nan fails; beta = 0 is allowed (the EMA is the last value)
    if not (math.isfinite(b) and 0.0 <= b < 1.0):
        raise ValueError(f"beta must satisfy 0 <= beta < 1, got {beta}")
    return b


def ema_weights(beta: float, t: int) -> NDArray:
    b = _check_beta(beta)
    if t < 0:
        raise ValueError(f"t must be >= 0, got {t}")
    # (1 - b), (1 - b) b, (1 - b) b^2, ... is newest first; x_1 is the oldest.
    return geometric(1.0 - b, b, int(t))[::-1].copy()
```

The `.copy()` turns the reversed view into a fresh, contiguous array. For $\beta = 0.9$, $t = 3$: `geometric(0.1, 0.9, 3)` is $[0.1, 0.09, 0.081]$, reversed to $[0.081, 0.09, 0.1]$, the weights of the hand example.

### The EMA class

```python
class EMA:
    def __init__(self, beta: float) -> None:
        self.beta = _check_beta(beta)
        self._t = 0
        self._m: float | NDArray = 0.0          # m_0 = 0, the start the correction assumes
        self._shape: tuple | None = None        # fixed by the first update

    @property
    def t(self) -> int:
        return self._t

    @property
    def value(self) -> float | NDArray:
        m = self._m
        return m.copy() if isinstance(m, np.ndarray) else float(m)

    def update(self, x: ArrayLike) -> float | NDArray:
        a = np.asarray(x, dtype=np.float64)
        if not np.isfinite(a).all():           # checked before any state changes
            raise ValueError(
                "x must be finite: one nan or inf would poison every later average"
            )
        if self._shape is not None and a.shape != self._shape:
            raise ValueError(
                f"x has shape {a.shape}, earlier updates had {self._shape}"
            )
        # Start from m_0 = 0, never from x_1: the bias correction below
        # assumes the zero start, and the two together would double-correct.
        m = self.beta * self._m + (1.0 - self.beta) * (a if a.ndim else float(a))
        self._shape = a.shape
        self._m = m
        self._t += 1
        return self.value                       # a copy, or a Python float

    def value_debiased(self) -> float | NDArray:
        if self._t == 0:
            raise RuntimeError(
                "value_debiased before any update: the average of nothing is 0 / 0"
            )
        # The weights of m_t add up to (1 - b)(1 + b + ... + b^(t-1)) = 1 - b^t.
        # geometric_sum (M00.3) gets that total right even for b near 1, where
        # 1 - b**t subtracts two nearly equal numbers.
        out = self._m / geometric_sum(1.0 - self.beta, self.beta, self._t)
        return out.copy() if isinstance(out, np.ndarray) else float(out)
```

The expression `(a if a.ndim else float(a))` keeps scalars as Python floats, so a scalar EMA stays a `float` all the way through, while an array input makes `m` a new float64 array (the expression allocates a new array; nothing is updated in place). Run the hand example through it: after `update(1)`, `update(2)`, `update(3)` with $\beta = 0.9$, `value` is $0.561$, `t` is 3, and `value_debiased()` divides by `geometric_sum(0.1, 0.9, 3)` $= 0.271$, giving $2.0701107$. With $\beta = 0$, `geometric_sum(1.0, 0.0, t)` is 1 and $m_t = x_t$, so both values are the latest input.

The full solution is [`ema.py`](../../../../course/solutions/python/tinyllm/num/ema.py).

### Why this design

`update` is $O(1)$ per scalar ($O(d)$ for an array of $d$ elements) and the state is $O(1)$ (or $O(d)$), however many steps have run. `value_debiased` adds one `geometric_sum`, which is $O(1)$. Recomputing $\sum_i w_i x_i$ from a stored history would cost $O(t)$ time and memory per step.

The recursion is preferred over the explicit sum for the same reason. Both give the same number, as the hand example shows, but the recursion needs no history and rounds only a couple of times per step, so its rounding error stays at a few hundred ulps after many steps.

Computing the total beats tracking it. You could keep a running $\beta^t$ by multiplying by $\beta$ each step and then form $1 - \beta^t$, but that rounds once per step and then cancels for $\beta$ near 1. `geometric_sum` evaluates the total from $t$ and $\beta$ directly, accurate at any $t$.

Raising beats skipping bad inputs. Silently dropping a NaN would hide a real failure upstream (a diverged batch); raising with the state unchanged lets the caller decide, and the curve stays valid either way.

One tradeoff is accepted: the EMA keeps one decay for its whole life. Schedules that change $\beta$ over time (annealing) need a new object or a different correction, because $1 - \beta^t$ assumes a constant $\beta$.

## What can go wrong

A correct implementation must pass the following checks, each tied to something that depends on it:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked EMA | unit | the hand example's $m_t$ and $\hat m_t$, the counter, Python floats | everyone agrees on the recursion and the correction |
| the hand-worked weights | unit | $[0.081, 0.09, 0.1]$, total 0.271 | the unrolled form |
| weights total $1 - \beta^t$ | property | totals $1 - \beta^t$ and ratio $\beta$ for five $\beta$ and five $t$ | the geometric partial sum |
| recursion equals weighted sum | differential | the recursion equals the dot product with `ema_weights` on 200 seeded inputs | two definitions, one number |
| a debiased constant is exact | property | $\hat m_t = c$ for $\beta$ up to 0.9999, $t$ up to 3000 | the central property of the EMA |
| exact rational arithmetic | golden | the float64 $m_t$ and $\hat m_t$ against the same recursion in exact rational arithmetic (`fractions.Fraction`), rounded once, after 200 seeded steps | rounding in the recursion stays at a few hundred ulps |
| the bias fades | property | $m_{50} = 10(1 - 0.9^{50})$ | how long the bias lasts |
| $\beta = 0$ | boundary | $\beta = 0$ returns the latest input | annealed schedules |
| arrays averaged elementwise | differential | an array EMA equals one scalar EMA per element; returned arrays are copies; shape changes raise | EMA weights over whole tensors in the TinyStories capstone |
| non-finite input | boundary | NaN and inf raise, state unchanged | one bad batch must not poison the curve |
| debiasing before any update | boundary | $t = 0$ raises `RuntimeError` | 0/0 is not a number |
| bad $\beta$ | boundary | $\beta < 0$, $\beta \ge 1$, NaN raise | $\beta = 1$ never moves |

The common mistakes each leave a recognizable fingerprint.

> **Warning:** Swapping $\beta$ and $1 - \beta$ in the update makes the average follow the newest input with weight 0.9 and forget at once. The update must read `self.beta * self._m + (1.0 - self.beta) * x`.

Starting from $m_0 = x_1$ and still dividing by $1 - \beta^t$ gives $\hat m_1 = 10\,x_1$ for $\beta = 0.9$; `_m` must start at 0.0. An off-by-one power, $1 - \beta^{t+1}$ or $1 - \beta^{t-1}$, gives $\hat m_1 = 0.526$ for $x_1 = 1$, or a division by zero at $t = 1$; the total is `geometric_sum(..., self._t)` with `_t` incremented once per update. Accepting NaN or inf makes every later average NaN; `np.isfinite(a).all()` is checked before any state changes. Forgetting the correction entirely reports the hand example's 0.1 as the average of 1; `value_debiased` divides by the weight total.

The remaining mistakes are subtler. Listing the weights newest first produces a weighted sum with the wrong inputs emphasized; `ema_weights` reverses `geometric(...)`. Rejecting $\beta = 0$ refuses a valid, if trivial, average; `_check_beta` accepts `0.0 <= b`. Returning the internal array from `update` lets a caller's in-place edit change the average; `update` returns `self.value`, a copy. Having no check at $t = 0$ yields a `ZeroDivisionError` instead of a clear error; `value_debiased` raises `RuntimeError` when `_t == 0`.

## Where this leads

The EMA shows up as soon as you train anything. In *Build Your Own LLM System*, [Training loop and the autograd bigram](../../build-an-llm/p00-foundations/05-training-loop-and-the-autograd-bigram.md) passes `ema=EMA(beta)` to `train_step`, which feeds each step's loss to `update` and reports `value_debiased()` as `loss_ema`, the smoothed loss curve. The [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) keeps an EMA of the model weights and evaluates it alongside the raw weights. And in [Adam and AdamW: bias correction and decoupled weight decay](../10-optimization/03-adam-and-adamw.md), Adam's first and second moments are EMAs of the gradient and its square, divided by $1 - \beta_1^t$ and $1 - \beta_2^t$: exactly this correction, applied twice.

### Further reading

- OpenStax, *Calculus Volume 2* (free), sections 5.2 and 5.3: infinite series, the geometric series, and divergence.
- Kingma and Ba, "Adam" (2015), section 3: initialization bias correction. Adam and AdamW in `torch/optim/adam.py` apply the same correction to two moments, often folded into the step size $\alpha \sqrt{1 - \beta_2^t}/(1 - \beta_1^t)$.
- PyTorch `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn` (`torch/optim/swa_utils.py`): an EMA of every parameter tensor and buffer, updated in place on the device.
- TensorBoard's smoothing slider (`tensorboard/plugins/scalar`, the `smoothing` weight): a debiased EMA of the plotted scalar, exactly the one built here.
- `numpy.expm1` and `numpy.log1p`, the functions that make $1 - \beta^t$ accurate for $\beta = 0.9999$; see [Sequences, geometric series, and frequency ladders](../00-precalculus/03-sequences-and-frequency-ladders.md).

## Key terms

- **Series**: the sequence of partial sums $S_t = a_0 + \cdots + a_{t-1}$ of a sequence.
- **Convergence**: the partial sums settle on a finite limit; otherwise the series diverges.
- **Ratio test**: if successive term ratios eventually stay below some $q < 1$, the series converges.
- **Geometric series**: $\sum a r^j$, with partial sum $a\frac{1 - r^t}{1 - r}$ and limit $\frac{a}{1-r}$ for $|r| < 1$.
- **Exponential moving average (EMA)**: the recursion $m_t = \beta m_{t-1} + (1 - \beta)x_t$, a weighted sum with geometric weights.
- **Effective window**: $\frac{1}{1 - \beta}$, roughly how many recent steps the EMA reflects.
- **Bias correction**: dividing $m_t$ by its weight total $1 - \beta^t$ so the zero start no longer pulls the average toward 0.

## Summary

- A series converges when its partial sums settle on a number; the geometric series $\sum_{j \ge 0} r^j$ converges to $\frac{1}{1 - r}$ exactly when $|r| < 1$, and its partial sum is $\frac{1 - r^t}{1 - r}$.
- The EMA $m_t = \beta m_{t-1} + (1 - \beta) x_t$ from $m_0 = 0$ is a weighted sum of all inputs with geometric weights $(1 - \beta)\beta^{t-i}$.
- Those weights add up to $1 - \beta^t$, not 1, so early values are biased toward 0; dividing by $1 - \beta^t$ fixes it, and the debiased average of a constant is that constant at every step.
- The bias fades on its own as $\beta^t \to 0$, after roughly $\frac{1}{1 - \beta}$ steps; with $\beta = 0.999$ that is a thousand steps of a misleading curve.
- One NaN in the input poisons every later average, so `update` rejects non-finite values and keeps its state.

## Review questions

1. Why must the EMA start at $m_0 = 0$ rather than at the first input?

   <details><summary>Answer</summary>

   The correction $\hat m_t = m_t / (1 - \beta^t)$ is derived from the unrolled sum starting at $m_0 = 0$, whose weights total $1 - \beta^t$. Starting at $m_0 = x_1$ already gives an unbiased average (its weights total 1), so dividing by $1 - \beta^t$ inflates it: with $\beta = 0.9$, $\hat m_1 = x_1 / 0.1 = 10x_1$. One can use either the zero start with the correction or the $x_1$ start without it, never both.

   </details>

2. Why does `value_debiased` divide by `geometric_sum(1 - beta, beta, t)` instead of computing `1 - beta**t`?

   <details><summary>Answer</summary>

   They are equal in exact arithmetic, but for $\beta$ near 1 and small $t$, $\beta^t$ is close to 1 and $1 - \beta^t$ subtracts two nearly equal numbers, keeping only part of its digits. `geometric_sum` evaluates the same total with `expm1` and `log1p`, so it stays accurate at $\beta = 0.9999$; that is what keeps the debiased average of a constant equal to the constant.

   </details>

3. What happens to the EMA if `update` accepts a single NaN, and why does the solution check before changing state?

   <details><summary>Answer</summary>

   $m_t = \beta m_{t-1} + (1-\beta)\,\text{nan}$ is NaN, and every later $m$ is $\beta \cdot \text{nan} + \ldots = $ NaN: the whole curve is lost from that step on, since nothing ever removes the NaN. Checking first and raising before assigning `_m`, `_shape`, or `_t` means a rejected input leaves the EMA exactly as it was, so the caller can skip the bad batch and continue.

   </details>

4. Why does `update` return `self.value` (a copy) rather than `self._m`?

   <details><summary>Answer</summary>

   For array inputs `self._m` is a numpy array; returning it would hand the caller a reference to the internal state, and an in-place operation on the result (`r *= 2`, `r[0] = 0`) would silently change the average. `value` returns a copy, so the EMA can only be changed through `update`.

   </details>

5. With $\beta = 0.999$, roughly how long is the uncorrected curve misleading, and how does that follow from the geometric series?

   <details><summary>Answer</summary>

   The uncorrected $m_t$ of a constant $c$ is $c(1 - \beta^t)$. $\beta^t = e^{t \ln \beta} \approx e^{-t(1-\beta)}$, which falls to 1 percent at $t \approx 4.6/(1-\beta) = 4600$ steps, and to about $1/e$ after $1/(1-\beta) = 1000$ steps, the effective window. For the first thousand or so steps the raw curve sits visibly below the true loss, which is why the training loop reports the debiased value.

   </details>
