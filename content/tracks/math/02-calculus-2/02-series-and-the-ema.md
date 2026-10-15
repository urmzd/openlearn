<!-- ol:module M02.2 -->
# Series convergence, EMA as a geometric series, bias correction

## Overview

| | |
|---|---|
| **Module** | `M02.2` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/num/ema.py`: `ema_weights` and the class `EMA` (`update`, `value`, `t`, `value_debiased`) |
| **Contract** | [`course/contracts/py/tinyllm/num/ema.pyi`](../../../../course/contracts/py/tinyllm/num/ema.pyi) |
| **Tests** | `course/tests/M02.2/test_ema.py` (what they check: section 4) |
| **Needs** | `M00.3` `geometric` and `geometric_sum`, which give the weights and their total (or `--ref-deps`) |
| **Used by** | `L0.5`'s `train_step` smooths the training-loss curve with `EMA` (`ema=`, reported as `loss_ema`), `C1` keeps an EMA of the weights, and `M10.3` (Adam) applies the same bias correction to its moments (section 6) |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Calculus Volume 2* (free), sections 5.2 and 5.3 (infinite series, the geometric series, divergence); Kingma and Ba, "Adam" (2015), section 3 (initialization bias correction) |

## Key Takeaways

- A **series** converges when its partial sums settle on a number; the **geometric series** $\sum_{j \ge 0} r^j$ converges to $\frac{1}{1 - r}$ exactly when $|r| < 1$, and its partial sum is $\frac{1 - r^t}{1 - r}$ (`test_weights_sum_to_one_minus_beta_power`).
- The **exponential moving average** $m_t = \beta m_{t-1} + (1 - \beta) x_t$ from $m_0 = 0$ is a weighted sum of all inputs with geometric weights $(1 - \beta)\beta^{t-i}$ (`test_weights_hand_example`, `test_update_equals_weighted_sum`).
- Those weights add up to $1 - \beta^t$, not 1, so early values are **biased toward 0**; dividing by $1 - \beta^t$ fixes it, and the debiased average of a constant is that constant at every step (`test_hand_example`, `test_debiased_constant_is_exact`).
- The bias fades on its own as $\beta^t \to 0$, after roughly $\frac{1}{1 - \beta}$ steps; with $\beta = 0.999$ that is a thousand steps of a misleading curve (`test_bias_fades_without_correction`).
- One nan in the input poisons every later average, so `update` rejects non-finite values and keeps its state (`test_rejects_nonfinite_input_and_keeps_state`).

## How to work this chapter

```bash
ol start M02.2              # stubs python/tinyllm/num/ema.py into your repo
ol tests M02.2              # read the test catalog first: rung R0, you write no tests here
ol check M02.2              # exit code is the verdict
ol check M02.2 --ref-deps   # only if your M00.3 is not passing yet
ol diff  M02.2              # after passing: your code against the reference
```

---

## 1. Why now

The loss your training loop prints (`L0.5`) jumps around from batch to batch, and you want to see its trend. The standard tool is an exponential moving average, and the first thing it does is lie: started from zero, it reports a loss of 0.55 when every batch says 5.5, and climbs toward the truth over hundreds of steps. Adam (`M10.3`), the optimizer you will train every model with, keeps two such averages per parameter and would take wildly wrong first steps without the correction this module derives. Both the lie and the fix are a fact about one infinite sum, the geometric series. This module defines convergence of series, sums the geometric one exactly, and builds the EMA with its bias correction on top of the sums you wrote in `M00.3`.

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

Unrolled, $m_3 = 0.081 \cdot 1 + 0.09 \cdot 2 + 0.1 \cdot 3 = 0.081 + 0.18 + 0.3 = 0.561$: the same number, with weights $[0.081, 0.09, 0.1]$ adding to $0.271 = 1 - 0.9^3$. The biased $m_3 = 0.561$ is below even the smallest input; the debiased 2.07 is a weighted average of 1, 2, 3 that leans toward the newest. These are `test_hand_example` and `test_weights_hand_example`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3's $m_t$ and $\hat m_t$, the counter, Python floats | you and the tests agree on the recursion and the correction |
| `test_weights_hand_example` | unit | $[0.081, 0.09, 0.1]$, total 0.271 | the unrolled form |
| `test_weights_sum_to_one_minus_beta_power` | property | totals $1 - \beta^t$ and ratio $\beta$ for five $\beta$ and five $t$ | the geometric partial sum of section 2.2 |
| `test_update_equals_weighted_sum` | differential | the recursion equals the dot product with `ema_weights` on 200 seeded inputs | two definitions, one number |
| `test_debiased_constant_is_exact` | property, smoke | $\hat m_t = c$ for $\beta$ up to 0.9999, $t$ up to 3000 | the design's property for this module |
| `test_matches_exact_rational_arithmetic` | golden | the float64 $m_t$ and $\hat m_t$ against the same recursion in exact rational arithmetic (`fractions.Fraction`), rounded once, after 200 seeded steps | rounding in the recursion stays at a few hundred ulps |
| `test_bias_fades_without_correction` | property | $m_{50} = 10(1 - 0.9^{50})$ | how long the bias lasts |
| `test_beta_zero_is_the_last_value` | boundary | $\beta = 0$ returns the latest input | annealed schedules |
| `test_arrays_are_averaged_elementwise` | differential | an array EMA equals one scalar EMA per element; returned arrays are copies; shape changes raise | EMA weights over whole tensors in `C1` |
| `test_rejects_nonfinite_input_and_keeps_state` | boundary | nan and inf raise, state unchanged | one bad batch must not poison the curve |
| `test_debiased_before_update_raises` | boundary | $t = 0$ raises `RuntimeError` | 0/0 is not a number |
| `test_rejects_bad_beta` | boundary | $\beta < 0$, $\beta \ge 1$, nan raise | $\beta = 1$ never moves |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. $\beta$ and $1 - \beta$ swapped in the update | the average follows the newest input with weight 0.9 and forgets at once | `test_hand_example`, `test_update_equals_weighted_sum` (mutant `s01`) |
| 2. starting from $m_0 = x_1$ and still dividing by $1 - \beta^t$ | $\hat m_1 = 10\,x_1$ for $\beta = 0.9$ | `test_debiased_constant_is_exact` (mutant `s02`) |
| 3. an off-by-one power: $1 - \beta^{t+1}$ or $1 - \beta^{t-1}$ | $\hat m_1 = 0.526$ for $x_1 = 1$, or a division by zero at $t = 1$ | `test_debiased_constant_is_exact` (mutants `s03`, `s06`) |
| 4. accepting nan or inf | every later average is nan | `test_rejects_nonfinite_input_and_keeps_state` (mutant `s09`) |
| forgetting the correction entirely | the hand example's 0.1 reported as the average of 1 | `test_hand_example` (mutant `s04`) |
| weights listed newest first | a weighted sum with the wrong inputs emphasized | `test_weights_hand_example` (mutant `s05`) |
| rejecting $\beta = 0$ | a valid, if trivial, average refused | `test_beta_zero_is_the_last_value` (mutant `s07`) |
| returning the internal array from `update` | a caller's in-place edit changes the average | `test_arrays_are_averaged_elementwise` (mutant `s08`) |
| no check at $t = 0$ | `ZeroDivisionError` instead of a clear error | `test_debiased_before_update_raises` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.3` | `geometric(1 - beta, beta, t)` gives the weights and `geometric_sum(1 - beta, beta, t)` the correction total $1 - \beta^t$ |
| Forward | `L0.5` | `train_step(..., ema=EMA(beta))` feeds each step's loss to `update` and reports `value_debiased()` as `loss_ema`, the smoothed loss curve |
| Forward | `M10.3` | Adam's first and second moments are EMAs of the gradient and its square, divided by $1 - \beta_1^t$ and $1 - \beta_2^t$ |
| Forward | `C1` | an EMA of the model weights, evaluated alongside the raw weights |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `EMA` over weights | PyTorch `torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn` | EMA of every parameter tensor and buffer, updated in place on the device | `torch/optim/swa_utils.py` |
| bias correction | Adam and AdamW | the same correction on two moments, often folded into the step size $\alpha \sqrt{1 - \beta_2^t}/(1 - \beta_1^t)$ | `torch/optim/adam.py`; `M10.3` |
| smoothed loss curve | TensorBoard's smoothing slider | a debiased EMA of the plotted scalar, exactly this module | `tensorboard/plugins/scalar` (the `smoothing` weight) |
| `geometric_sum` near $r = 1$ | `numpy.expm1`, `numpy.log1p` | the functions that make $1 - \beta^t$ accurate for $\beta = 0.9999$ | `M00.3` section 2 |
