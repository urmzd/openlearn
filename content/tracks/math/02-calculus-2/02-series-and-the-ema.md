# Series convergence, EMA as a geometric series, bias correction

## Overview

| | |
|---|---|
| **Module** | `M02.2` · build · Python · Pass 2 · 1 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/ema.py`](../../../../course/solutions/python/tinyllm/num/ema.py): `ema_weights` and the class `EMA` (`update`, `value`, `t`, `value_debiased`) |
| **Contract** | [`course/contracts/py/tinyllm/num/ema.pyi`](../../../../course/contracts/py/tinyllm/num/ema.pyi) |
| **Needs** | `M00.3` `geometric` and `geometric_sum`, which give the weights and their total |
| **Used by** | `L0.5`'s `train_step` smooths the training-loss curve with `EMA` (`ema=`, reported as `loss_ema`), `C1` keeps an EMA of the weights, and `M10.3` (Adam) applies the same bias correction to its moments (section 7) |
| **Optional depth** | OpenStax, *Calculus Volume 2* (free), sections 5.2 and 5.3 (infinite series, the geometric series, divergence); Kingma and Ba, "Adam" (2015), section 3 (initialization bias correction) |

## Key Takeaways

- A **series** converges when its partial sums settle on a number; the **geometric series** $\sum_{j \ge 0} r^j$ converges to $\frac{1}{1 - r}$ exactly when $|r| < 1$, and its partial sum is $\frac{1 - r^t}{1 - r}$.
- The **exponential moving average** $m_t = \beta m_{t-1} + (1 - \beta) x_t$ from $m_0 = 0$ is a weighted sum of all inputs with geometric weights $(1 - \beta)\beta^{t-i}$.
- Those weights add up to $1 - \beta^t$, not 1, so early values are **biased toward 0**; dividing by $1 - \beta^t$ fixes it, and the debiased average of a constant is that constant at every step.
- The bias fades on its own as $\beta^t \to 0$, after roughly $\frac{1}{1 - \beta}$ steps; with $\beta = 0.999$ that is a thousand steps of a misleading curve.
- One nan in the input poisons every later average, so `update` rejects non-finite values and keeps its state.

## How to read this chapter

Sections 1 and 2 define convergence of series, sum the geometric series, and derive the EMA's weights and its bias correction from first principles. Section 3 works an EMA by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The loss the training loop prints (`L0.5`) jumps around from batch to batch, and what matters is its trend. The standard tool is an exponential moving average, and the first thing it does is lie: started from zero, it reports a loss of 0.55 when every batch says 5.5, and climbs toward the truth over hundreds of steps. Adam (`M10.3`), the optimizer every model in the course trains with, keeps two such averages per parameter and would take wildly wrong first steps without the correction this module derives. Both the lie and the fix are a fact about one infinite sum, the geometric series. This module defines convergence of series, sums the geometric one exactly, and builds the EMA with its bias correction on top of the sums of `M00.3`.

## 2. Principles

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

### 2.1 Series and convergence

A **series** $\sum_{j \ge 0} a_j$ is the sequence of its **partial sums** $S_t = a_0 + \cdots + a_{t-1}$. It **converges** to $S$ when the partial sums settle on $S$ in the sense of a limit (`M01.1`): for every tolerance there is a $t$ beyond which every $S_t$ is within it. Otherwise it **diverges**. Small terms are necessary but not enough: $1 + \frac12 + \frac13 + \frac14 + \cdots$ diverges, because the terms from $\frac{1}{2^k + 1}$ to $\frac{1}{2^{k+1}}$ are $2^k$ numbers each at least $\frac{1}{2^{k+1}}$, adding to at least $\frac12$ for every $k$, so the sum passes any bound. A useful test, the **ratio test**: if $|a_{j+1}/a_j|$ eventually stays below some $q < 1$, the series converges, because from there on its terms are smaller than those of a geometric series with ratio $q$ (`M02.1` used exactly this to bound the tail of the erf series).

### 2.2 The geometric series

For $a_j = a r^j$, multiply the partial sum by $r$ and subtract:

$$S_t - r S_t = a - a r^t \quad\Longrightarrow\quad S_t = a\,\frac{1 - r^t}{1 - r} \quad (r \neq 1), \qquad S_t = a t \quad (r = 1) .$$

If $|r| < 1$, $r^t \to 0$ and the series converges to $\frac{a}{1 - r}$; if $|r| \ge 1$ the terms do not shrink and it diverges. When $r$ is close to 1, $1 - r^t$ subtracts two nearly equal numbers, and `M00.3`'s `geometric_sum` evaluates it as $\frac{\mathrm{expm1}(t \ln r)}{r - 1}$ to keep full precision; this module calls it rather than re-deriving it.

### 2.3 The EMA unrolled

The update $m_t = \beta m_{t-1} + (1 - \beta)x_t$ keeps a fraction $\beta$ of the old average and mixes in $1 - \beta$ of the new input. Unroll it from $m_0 = 0$:

$$m_t = (1 - \beta)x_t + \beta(1 - \beta)x_{t-1} + \beta^2(1 - \beta)x_{t-2} + \cdots + \beta^{t-1}(1 - \beta)x_1 = \sum_{i=1}^{t} \underbrace{(1 - \beta)\beta^{t-i}}_{w_i^{(t)}}\, x_i .$$

The newest input has weight $1 - \beta$, and each step back in time multiplies the weight by $\beta$: a geometric sequence. `ema_weights(beta, t)` returns $w_1^{(t)}, \ldots, w_t^{(t)}$ oldest first, which is `geometric(1 - beta, beta, t)` reversed. The **effective window** is $\frac{1}{1 - \beta}$ (the sum of the infinite weight sequence divided by its largest weight): 10 steps for $\beta = 0.9$, 1000 for $\beta = 0.999$. The **half-life**, after which an input's weight has halved, is $\frac{\ln \frac12}{\ln \beta}$: 6.6 steps for $\beta = 0.9$.

### 2.4 Bias correction

The weights add up to a geometric partial sum:

$$\sum_{i=1}^{t} w_i^{(t)} = (1 - \beta)\frac{1 - \beta^t}{1 - \beta} = 1 - \beta^t .$$

An average whose weights add to less than 1 is pulled toward 0, which is where it started. If every $x_i = c$, then $m_t = c(1 - \beta^t)$: after one step with $\beta = 0.9$, only $0.1c$. Dividing by the total turns the weights back into an average,

$$\hat m_t = \frac{m_t}{1 - \beta^t} ,$$

and for a constant input $\hat m_t = c$ at every $t$, exactly in real arithmetic (and to a few $\varepsilon/(1 - \beta)$ in float64). If the inputs are random with the same mean $\mu$, the expected value of $m_t$ is $(1 - \beta^t)\mu$, so $\hat m_t$ is an unbiased estimate of $\mu$: the reason Adam's update has the right size from its first step. The correction matters only early: $\beta^t$ falls below 1 percent after $t \approx \frac{4.6}{1 - \beta}$ steps, and from then on $\hat m_t \approx m_t$.

Two consequences shape the code. The zero start is part of the definition: initializing $m$ with $x_1$ and then dividing by $1 - \beta^t$ corrects a bias that is not there ($\hat m_1 = 10\,x_1$ for $\beta = 0.9$). And at $t = 0$ the correction is $0/0$: nothing has been averaged, so `value_debiased` raises instead of returning a number.

## 3. Worked example by hand

$\beta = 0.9$, inputs $x_1 = 1$, $x_2 = 2$, $x_3 = 3$:

| $t$ | $m_t = 0.9\,m_{t-1} + 0.1\,x_t$ | $1 - 0.9^t$ | $\hat m_t$ |
|---|---|---|---|
| 1 | $0.9 \cdot 0 + 0.1 \cdot 1 = 0.1$ | 0.1 | $0.1/0.1 = 1$ |
| 2 | $0.9 \cdot 0.1 + 0.1 \cdot 2 = 0.29$ | 0.19 | $0.29/0.19 = 1.5263158$ |
| 3 | $0.9 \cdot 0.29 + 0.1 \cdot 3 = 0.561$ | 0.271 | $0.561/0.271 = 2.0701107$ |

Unrolled, $m_3 = 0.081 \cdot 1 + 0.09 \cdot 2 + 0.1 \cdot 3 = 0.081 + 0.18 + 0.3 = 0.561$: the same number, with weights $[0.081, 0.09, 0.1]$ adding to $0.271 = 1 - 0.9^3$. The biased $m_3 = 0.561$ is below even the smallest input; the debiased 2.07 is a weighted average of 1, 2, 3 that leans toward the newest.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked EMA | unit | section 3's $m_t$ and $\hat m_t$, the counter, Python floats | everyone agrees on the recursion and the correction |
| the hand-worked weights | unit | $[0.081, 0.09, 0.1]$, total 0.271 | the unrolled form |
| weights total $1 - \beta^t$ | property | totals $1 - \beta^t$ and ratio $\beta$ for five $\beta$ and five $t$ | the geometric partial sum of section 2.2 |
| recursion equals weighted sum | differential | the recursion equals the dot product with `ema_weights` on 200 seeded inputs | two definitions, one number |
| a debiased constant is exact | property | $\hat m_t = c$ for $\beta$ up to 0.9999, $t$ up to 3000 | the central property of this module |
| exact rational arithmetic | golden | the float64 $m_t$ and $\hat m_t$ against the same recursion in exact rational arithmetic (`fractions.Fraction`), rounded once, after 200 seeded steps | rounding in the recursion stays at a few hundred ulps |
| the bias fades | property | $m_{50} = 10(1 - 0.9^{50})$ | how long the bias lasts |
| $\beta = 0$ | boundary | $\beta = 0$ returns the latest input | annealed schedules |
| arrays averaged elementwise | differential | an array EMA equals one scalar EMA per element; returned arrays are copies; shape changes raise | EMA weights over whole tensors in `C1` |
| non-finite input | boundary | nan and inf raise, state unchanged | one bad batch must not poison the curve |
| debiasing before any update | boundary | $t = 0$ raises `RuntimeError` | 0/0 is not a number |
| bad $\beta$ | boundary | $\beta < 0$, $\beta \ge 1$, nan raise | $\beta = 1$ never moves |

## 5. The solution, step by step

### From first principles

1. **The state is two numbers: $m_t$ and $t$.** The recursion $m_t = \beta m_{t-1} + (1-\beta)x_t$ needs only the previous average, and the correction $1 - \beta^t$ needs only the count. Keeping the whole history is unnecessary (section 2.3 shows the recursion *is* the weighted sum).
2. **Start at $m_0 = 0$.** The bias correction of section 2.4 is derived for that start. Seeding with $x_1$ and also dividing by $1 - \beta^t$ would correct a bias that is not there.
3. **The correction total comes from `geometric_sum`.** $\sum_i w_i^{(t)} = (1-\beta)\frac{1-\beta^t}{1-\beta}$ is exactly `geometric_sum(1 - beta, beta, t)` from `M00.3`, which stays accurate for $\beta$ near 1 where $1 - \beta^t$ cancels.
4. **$t = 0$ has no average.** The debiased value would be $0/0$; raising is the only honest answer.
5. **Validate before mutating.** A non-finite input or a shape change raises *before* the state changes, so one bad batch leaves the curve intact. $\beta$ is checked once, at construction: $0 \le \beta < 1$ (with $\beta = 1$ the average never moves and the correction divides by 0).
6. **Never hand out internal state.** Returning the internal array would let a caller's in-place edit change the average, so every array returned is a copy, and scalars come back as Python `float`.
7. **The weights are the geometric sequence, reversed.** `geometric(1 - beta, beta, t)` lists $(1-\beta), (1-\beta)\beta, \ldots$, newest first; the contract lists $x_1$ (the oldest) first, so reverse it.

### The code, annotated

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

The `.copy()` turns the reversed view into a fresh, contiguous array. For $\beta = 0.9$, $t = 3$: `geometric(0.1, 0.9, 3)` is $[0.1, 0.09, 0.081]$, reversed to $[0.081, 0.09, 0.1]$, the weights of section 3.

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

`(a if a.ndim else float(a))` keeps scalars as Python floats, so a scalar EMA stays a `float` all the way through, while an array input makes `m` a new float64 array (the expression allocates a new array; nothing is updated in place). Running section 3: after `update(1)`, `update(2)`, `update(3)` with $\beta = 0.9$, `value` is $0.561$, `t` is 3, and `value_debiased()` divides by `geometric_sum(0.1, 0.9, 3)` $= 0.271$, giving $2.0701107$. With $\beta = 0$, `geometric_sum(1.0, 0.0, t)` is 1 and $m_t = x_t$, so both values are the latest input.

Full solution: [`ema.py`](../../../../course/solutions/python/tinyllm/num/ema.py).

### Why this solution

- **Cost.** `update` is $O(1)$ per scalar ($O(d)$ for an array of $d$ elements) and the state is $O(1)$ (or $O(d)$), however many steps have run. `value_debiased` adds one `geometric_sum`, $O(1)$. Recomputing $\sum_i w_i x_i$ from a stored history would cost $O(t)$ time and memory per step.
- **Recursion over the explicit sum.** Both give the same number (section 3), but the recursion needs no history and rounds only a couple of times per step, so its rounding error stays at a few hundred ulps after many steps.
- **Computing the total instead of tracking it.** One could keep a running $\beta^t$ by multiplying by $\beta$ each step and form $1 - \beta^t$. That rounds once per step and then cancels for $\beta$ near 1; `geometric_sum` evaluates the total from $t$ and $\beta$ directly, accurate at any $t$.
- **Raising over skipping bad inputs.** Silently dropping a NaN would hide a real failure upstream (a diverged batch); raising with the state unchanged lets the caller decide, and the curve stays valid either way.
- **Tradeoffs accepted.** The EMA keeps one decay for its whole life; schedules that change $\beta$ over time (annealing) need a new object or a different correction, because $1 - \beta^t$ assumes a constant $\beta$.

### Explain it back

**1. Why must the EMA start at $m_0 = 0$ rather than at the first input?**

<details><summary>Answer</summary>

The correction $\hat m_t = m_t / (1 - \beta^t)$ is derived from the unrolled sum starting at $m_0 = 0$, whose weights total $1 - \beta^t$. Starting at $m_0 = x_1$ already gives an unbiased average (its weights total 1), so dividing by $1 - \beta^t$ inflates it: with $\beta = 0.9$, $\hat m_1 = x_1 / 0.1 = 10x_1$. One can use either the zero start with the correction or the $x_1$ start without it, never both.

</details>

**2. Why does `value_debiased` divide by `geometric_sum(1 - beta, beta, t)` instead of computing `1 - beta**t`?**

<details><summary>Answer</summary>

They are equal in exact arithmetic, but for $\beta$ near 1 and small $t$, $\beta^t$ is close to 1 and $1 - \beta^t$ subtracts two nearly equal numbers, keeping only part of its digits. `geometric_sum` evaluates the same total with `expm1` and `log1p` (`M00.3`), so it stays accurate at $\beta = 0.9999$; that is what keeps the debiased average of a constant equal to the constant.

</details>

**3. What happens to the EMA if `update` accepts a single NaN, and why does the solution check before changing state?**

<details><summary>Answer</summary>

$m_t = \beta m_{t-1} + (1-\beta)\,\text{nan}$ is NaN, and every later $m$ is $\beta \cdot \text{nan} + \ldots = $ NaN: the whole curve is lost from that step on, since nothing ever removes the NaN. Checking first and raising before assigning `_m`, `_shape`, or `_t` means a rejected input leaves the EMA exactly as it was, so the caller can skip the bad batch and continue.

</details>

**4. Why does `update` return `self.value` (a copy) rather than `self._m`?**

<details><summary>Answer</summary>

For array inputs `self._m` is a numpy array; returning it would hand the caller a reference to the internal state, and an in-place operation on the result (`r *= 2`, `r[0] = 0`) would silently change the average. `value` returns a copy, so the EMA can only be changed through `update`.

</details>

**5. With $\beta = 0.999$, roughly how long is the uncorrected curve misleading, and how does that follow from the geometric series?**

<details><summary>Answer</summary>

The uncorrected $m_t$ of a constant $c$ is $c(1 - \beta^t)$. $\beta^t = e^{t \ln \beta} \approx e^{-t(1-\beta)}$, which falls to 1 percent at $t \approx 4.6/(1-\beta) = 4600$ steps, and to about $1/e$ after $1/(1-\beta) = 1000$ steps, the effective window. For the first thousand or so steps the raw curve sits visibly below the true loss, which is why the training loop reports the debiased value.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. $\beta$ and $1 - \beta$ swapped in the update | the average follows the newest input with weight 0.9 and forgets at once | `self.beta * self._m + (1.0 - self.beta) * x` |
| 2. starting from $m_0 = x_1$ and still dividing by $1 - \beta^t$ | $\hat m_1 = 10\,x_1$ for $\beta = 0.9$ | `_m` starts at 0.0; step 2 |
| 3. an off-by-one power: $1 - \beta^{t+1}$ or $1 - \beta^{t-1}$ | $\hat m_1 = 0.526$ for $x_1 = 1$, or a division by zero at $t = 1$ | the total is `geometric_sum(..., self._t)` with `_t` incremented once per update; step 3 |
| 4. accepting nan or inf | every later average is nan | `np.isfinite(a).all()` checked before any state changes; step 5 |
| forgetting the correction entirely | the hand example's 0.1 reported as the average of 1 | `value_debiased` divides by the weight total |
| weights listed newest first | a weighted sum with the wrong inputs emphasized | `ema_weights` reverses `geometric(...)`; step 7 |
| rejecting $\beta = 0$ | a valid, if trivial, average refused | `_check_beta` accepts `0.0 <= b` |
| returning the internal array from `update` | a caller's in-place edit changes the average | `update` returns `self.value`, a copy; step 6 |
| no check at $t = 0$ | `ZeroDivisionError` instead of a clear error | `value_debiased` raises `RuntimeError` when `_t == 0`; step 4 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.3` | `geometric(1 - beta, beta, t)` gives the weights and `geometric_sum(1 - beta, beta, t)` the correction total $1 - \beta^t$ |
| Forward | `L0.5` | `train_step(..., ema=EMA(beta))` feeds each step's loss to `update` and reports `value_debiased()` as `loss_ema`, the smoothed loss curve |
| Forward | `M10.3` | Adam's first and second moments are EMAs of the gradient and its square, divided by $1 - \beta_1^t$ and $1 - \beta_2^t$ |
| Forward | `C1` | an EMA of the model weights, evaluated alongside the raw weights |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `EMA` over weights | PyTorch `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn` | EMA of every parameter tensor and buffer, updated in place on the device | `torch/optim/swa_utils.py` |
| bias correction | Adam and AdamW | the same correction on two moments, often folded into the step size $\alpha \sqrt{1 - \beta_2^t}/(1 - \beta_1^t)$ | `torch/optim/adam.py`; `M10.3` |
| smoothed loss curve | TensorBoard's smoothing slider | a debiased EMA of the plotted scalar, exactly this module | `tensorboard/plugins/scalar` (the `smoothing` weight) |
| `geometric_sum` near $r = 1$ | `numpy.expm1`, `numpy.log1p` | the functions that make $1 - \beta^t$ accurate for $\beta = 0.9999$ | `M00.3` section 2 |
