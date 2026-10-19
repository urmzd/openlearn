# Adam and AdamW: bias correction and decoupled weight decay

## Overview

| | |
|---|---|
| **Module** | `M10.3` · build · Python · Pass 2 · 3 to 4 h |
| **Solution** | [`course/solutions/python/tinyllm/optim/adamw.py`](../../../../course/solutions/python/tinyllm/optim/adamw.py): `Adam` and `AdamW`, with `step`, `zero_grad`, `state_dict`, `load_state_dict` |
| **Contract** | [`course/contracts/py/tinyllm/optim/adamw.pyi`](../../../../course/contracts/py/tinyllm/optim/adamw.pyi) · checkpoint keys: [`formats/checkpoint.md`](../../../../course/contracts/formats/checkpoint.md) |
| **Needs** | no code dependency. Reading: `M10.2` (the `Optimizer` protocol and SGD with momentum), `M02.2` (the exponential moving average and its bias correction), `S-M10a` (the Adam derivation problems) |
| **Used by** | `L0.5` trains every model with it from Part 0 on · `L0.6` saves and restores its state in checkpoints · later `L4.1`, `L5.5`, `L6.1`, `L7.9`, `C1`, and `L12.1` · later: `L11.1`, `L3.6`, `L6.5` |
| **Optional depth** | Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2015), sections 2 and 3; Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (2019), sections 2 and 3; the torch source `torch/optim/adam.py`, `_single_tensor_adam` |

## Key Takeaways

- **Adam** keeps two exponential moving averages per coordinate, $m$ of the gradient and $v$ of its square, and moves each coordinate by $\eta\, \hat m / (\sqrt{\hat v} + \epsilon)$.
- **Bias correction** divides $m_t$ by $1 - \beta_1^t$ and $v_t$ by $1 - \beta_2^t$, which undoes the pull toward the zero they start from; the first step is then exactly $\eta$ per coordinate.
- Dividing by $\sqrt{\hat v}$ makes the update **invariant to the scale of the gradient**, so one learning rate serves layers whose gradients differ by orders of magnitude.
- **AdamW decouples weight decay**: it multiplies the weights by $1 - \eta\lambda$ instead of adding $\lambda w$ to the gradient, where Adam's normalization would wash the penalty out.
- The optimizer **state is part of the model**: a save and load between steps must reproduce the uninterrupted run bit for bit.

## How to read this chapter

Sections 1 to 3 derive the two moving averages, the normalization, bias correction, and decoupled decay, and run two steps by hand. Section 4 fixes the interface. Section 5 walks through `step` line by line and explains each placement: where the decay goes, where $\epsilon$ goes, and why the state is copied. Try to predict each line from section 2 before reading it.

---

## 1. Why now

The training loop does not exist yet. `L0.5` builds it next, and it needs an optimizer that trains every model in the course without retuning: a bigram table, a 64-unit MLP on digits, an LSTM, a Transformer, and the 10M-parameter Llama-style model of `C1`. Plain SGD with momentum (`M10.2`) needs a learning rate tuned per model, and inside one model it is too slow for layers with small gradients and unstable for layers with large ones: an embedding row that sees a rare token gets a gradient a thousand times smaller than a layer norm gain. Adam fixes this by rescaling each coordinate by its own gradient history, and AdamW is the version every modern language model trains with. The `C1` training spec defaults to `adamw` with betas `(0.9, 0.95)`, and the checkpoint format already reserves `exp_avg` and `exp_avg_sq` for its state, so `dur.11` can kill a training worker and resume it. This module builds that optimizer and proves it equals `torch.optim.AdamW` step for step.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $w$ | one parameter tensor (the optimizer treats each coordinate the same way) | `float32[...]`, `p.data` |
| $g_t$ | the gradient of the loss with respect to $w$ at step $t$ | same shape as $w$, `p.grad` |
| $t$ | the step counter: 1 for the first update | `int` |
| $\eta$ | the learning rate, read on every step | `float`, `opt.lr` |
| $\beta_1, \beta_2$ | decay rates of the two moving averages, in $[0, 1)$ | `float`, `opt.betas` |
| $m_t$ | first moment: moving average of $g$ | same shape as $w$, `exp_avg` |
| $v_t$ | second moment: moving average of $g^2$ (elementwise) | same shape as $w$, `exp_avg_sq` |
| $\hat m_t, \hat v_t$ | bias-corrected moments, $m_t / (1 - \beta_1^t)$ and $v_t / (1 - \beta_2^t)$ | same shape as $w$ |
| $\epsilon$ | a small constant that keeps the denominator away from 0 | `float`, default $10^{-8}$ |
| $\lambda$ | the weight decay coefficient | `float`, `opt.weight_decay` |

All operations on vectors are **elementwise**: $g^2$ squares each coordinate, $\sqrt{v}$ takes each square root, and $m / \sqrt{v}$ divides coordinate by coordinate.

### 2.1 Moving averages of the gradient

Gradient descent moves $w$ against the gradient: $w \leftarrow w - \eta\, g_t$. A minibatch gradient is noisy, and its direction changes from step to step. Momentum (`M10.2`) smooths it with an **exponential moving average** (EMA, `M02.2`):

$$m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t, \qquad m_0 = 0 .$$

Unrolling the recursion shows what $m_t$ is: a weighted sum of all past gradients with geometrically decaying weights, $m_t = (1 - \beta_1) \sum_{s=1}^{t} \beta_1^{t-s} g_s$. With $\beta_1 = 0.9$ the last 10 or so steps carry most of the weight. Adam keeps a second EMA, of the squared gradient:

$$v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2, \qquad v_0 = 0 .$$

With $\beta_2 = 0.999$ it remembers about 1000 steps. $v_t$ estimates $\mathbb{E}[g^2]$, the mean squared size of this coordinate's gradient.

### 2.2 Normalizing by the second moment

Adam's update divides the first moment by the square root of the second:

$$w_t = w_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} .$$

$\sqrt{\hat v_t}$ has the units of the gradient, so the ratio $\hat m_t / \sqrt{\hat v_t}$ has no units: multiplying every gradient by a constant $c > 0$ multiplies both $\hat m_t$ and $\sqrt{\hat v_t}$ by $c$ and leaves the update unchanged (when $\epsilon$ is negligible). Each coordinate's step is therefore about $\eta$ in size when its gradient keeps a consistent sign, and smaller when the sign flips (then $|\hat m| \ll \sqrt{\hat v}$, because positive and negative gradients cancel in $m$ but not in $v$). The learning rate becomes a step length in parameter space, which is why one value such as $3 \times 10^{-4}$ works across many models. $\epsilon$ only matters when $\sqrt{\hat v}$ is near or below it, for coordinates whose gradients are almost always zero.

### 2.3 Bias correction

Both averages start at 0, so early on they are too small. Take every $g_s$ equal to the same value $g$. Then $m_t = (1 - \beta_1)\, g \sum_{s=0}^{t-1} \beta_1^{s} = (1 - \beta_1^t)\, g$ by the geometric sum (`M00.3`). The average is short of $g$ by exactly the factor $1 - \beta_1^t$, and dividing by it removes the bias:

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t} .$$

For random gradients with constant mean the same argument holds in expectation. The factor matters most for $v$: with $\beta_2 = 0.999$, $1 - \beta_2^{1} = 0.001$, so the uncorrected $v_1$ is 1000 times too small and the first step would be $\sqrt{1000} \approx 32$ times too large in the ratio $m / \sqrt{v}$ (partly offset by the uncorrected $m$, which is 10 times too small: net $3.16$ times too large). After a few thousand steps both factors are 1 and the correction disappears.

At $t = 1$ the corrections give $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the first update is $\eta\, g_1 / (|g_1| + \epsilon)$: exactly $\eta$ times the sign of $g_1$ when $\epsilon = 0$. torch folds the corrections into the step for speed, and the solution uses the same arrangement, because it decides where $\epsilon$ goes:

$$w_t = w_{t-1} - \frac{\eta}{1 - \beta_1^t}\cdot \frac{m_t}{\sqrt{v_t} / \sqrt{1 - \beta_2^t} + \epsilon} .$$

### 2.4 Decoupled weight decay

**Weight decay** shrinks the weights toward 0 a little on every step, which regularizes the model. For SGD it is the same as adding the penalty $\frac{\lambda}{2}\lVert w \rVert^2$ to the loss, whose gradient is $\lambda w$: the step $w - \eta(g + \lambda w) = (1 - \eta\lambda)\, w - \eta g$ decays $w$ by the factor $1 - \eta\lambda$.

For Adam the two are not the same. **Adam with L2** (`torch.optim.Adam(weight_decay=...)`) adds $\lambda w$ to $g$ before both moments see it, and then the normalization divides it by $\sqrt{\hat v}$: a coordinate with large gradients gets almost no decay, and with $g = 0$ the penalty itself becomes a normalized step of size $\eta$, whatever the weight's size. **AdamW** applies the decay directly to the weights, before the Adam update, and leaves the gradient alone:

$$w \leftarrow (1 - \eta\lambda)\, w, \quad \text{then the Adam step on } g_t .$$

Every coordinate shrinks by the same factor, as intended. The decay is scaled by $\eta$, so a schedule (`M10.4`) that lowers $\eta$ lowers the decay with it. Parameters whose gradient is `None` (the loss did not reach them this step) are skipped entirely: no decay, no moment update.

### 2.5 Saving and restoring the state

The optimizer state is $t$, $m$, and $v$ for every parameter, plus the hyperparameters. Training that stops and resumes (`dur.11` kills workers on purpose) must continue as if it never stopped, so `state_dict` returns **copies** of everything, and `load_state_dict` restores all of it, including $t$: with $t$ reset to 0 the bias correction would restart and the next step would be far too large. `formats/checkpoint.md` writes the moments as `<name>.exp_avg` and `<name>.exp_avg_sq`, in the parameter's dtype.

## 3. Worked example by hand

One scalar parameter $x_0 = 1$, gradients $g_1 = 2$ then $g_2 = -1$ (given, not computed from a loss), $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 0$, no weight decay.

| Step | $m_t$ | $v_t$ | $1 - \beta_1^t$ | $1 - \beta_2^t$ | $\hat m_t$ | $\hat v_t$ | update $\eta \hat m / \sqrt{\hat v}$ | $x_t$ |
|---|---|---|---|---|---|---|---|---|
| 1 | $0.1 \cdot 2 = 0.2$ | $0.001 \cdot 4 = 0.004$ | $0.1$ | $0.001$ | $2$ | $4$ | $0.1 \cdot 2 / 2 = 0.1$ | $0.9$ |
| 2 | $0.9 \cdot 0.2 + 0.1 \cdot (-1) = 0.08$ | $0.999 \cdot 0.004 + 0.001 \cdot 1 = 0.004996$ | $0.19$ | $0.001999$ | $8/19 \approx 0.42105$ | $4996/1999 \approx 2.49925$ | $0.1 \cdot 0.42105 / 1.58090 \approx 0.026634$ | $\approx 0.873366$ |

Two things to see. Step 1 moves by exactly $\eta = 0.1$: that is bias correction at work (without it, $0.1 \cdot 0.2 / \sqrt{0.004} \approx 0.316$). Step 2 has a negative gradient, yet $x$ keeps decreasing: $m$ still points the old way, and the step is smaller because $\hat v$ remembers the large first gradient. This is the first case in section 4's table.

**With weight decay $\lambda = 0.1$.** AdamW first multiplies $x$ by $1 - \eta\lambda = 0.99$: step 1 gives $0.99 - 0.1 = 0.89$; step 2 gives $0.89 \cdot 0.99 - 0.026634 \approx 0.854466$. Adam with L2 instead feeds $g_1 + \lambda x_0 = 2.1$ and $g_2 + \lambda x_1 = -1 + 0.09 = -0.91$ to the moments: step 1 still moves by exactly $0.1$ (the first step is $\eta$ times a sign, whatever the gradient), to $0.9$; step 2 lands at $\approx 0.868123$. The penalty changed $x_2$ by $0.0052$ under L2 and by $0.0189$ under AdamW.

## 4. The interface

```python
class Adam:
    lr: float; betas: tuple[float, float]; eps: float; weight_decay: float
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0) -> None
    def step(self) -> None             # one update of every p whose grad is not None, in place
    def zero_grad(self) -> None        # every p.grad = None
    def state_dict(self) -> dict       # {"step", "lr", "betas", "eps", "weight_decay", "exp_avg": [...], "exp_avg_sq": [...]}
    def load_state_dict(self, sd: dict) -> None

class AdamW(Adam):                     # decoupled decay; weight_decay defaults to 0.01
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01) -> None
```

A parameter is any object with `data` (a float ndarray) and `grad` (an ndarray or `None`); a two-attribute class is enough, and `L0.5` passes autograd tensors (`L0.1`). The step counter $t$ is one global count of `step()` calls, as `formats/checkpoint.md` stores one step. torch keeps a counter per parameter instead; the two agree whenever every parameter has a gradient on every step, which is how the course trains. A schedule changes the learning rate by assigning `opt.lr` before `step()`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked two steps | unit | section 3, both steps, to float64 precision | everyone agrees on the update before any code |
| AdamW and Adam with L2 differ | unit | section 3 with $\lambda = 0.1$ for both kinds of decay | the one difference between the two classes |
| torch AdamW trajectory | golden | 20 steps of torch AdamW on two tensors, and the final moments | every training run assumes torch's update |
| torch Adam-with-L2 trajectory | golden | 20 steps of torch Adam with `weight_decay = 0.1` | L2 coupling feeds both moments |
| a learning rate set between steps is used | golden | lr changes on every step (betas `(0.9, 0.95)`) | `M10.4` schedules write `opt.lr` |
| $\epsilon$ is added after bias correction | golden | gradients of size $10^{-6}$ with $\epsilon = 10^{-6}$ | where $\epsilon$ goes is visible for rare features |
| float32 parameters stay float32 | golden | torch's float32 trajectory; moments stay float32 | model parameters are float32 (`L0.4`) |
| the first step moves each coordinate by lr | property | with $\epsilon = 0$ the first step is $\eta \cdot \mathrm{sign}(g)$ at every scale | bias correction, stated as a law |
| the update ignores gradient scale | property | gradients times $10^{-3}$ or $250$ give the same trajectory | one lr for every layer |
| decoupled decay with zero gradient | unit | $g = 0$: AdamW gives $3.8$, Adam with L2 gives $3.9$ | decoupling, in one line |
| a parameter without grad is untouched | boundary | `grad = None`: no decay, no moment change, $t$ still advances | unused embeddings and frozen branches |
| the update is in place | unit | `p.data` keeps its identity, a view writes through to its buffer | the model and `L11` flat buffers hold the arrays |
| `zero_grad` sets `None` | unit | every grad becomes `None` | `L0.1` accumulates into `grad` |
| resume is bitwise | property | 5 steps, save, load into a fresh optimizer, 5 steps equals 10 steps bit for bit | `dur.11` kill and resume |
| `state_dict` is a snapshot | unit | later steps do not change a saved dict | `L0.6` may write it after training continues |
| `state_dict` layout | unit | keys, shapes, hyperparameters restored, mismatches rejected | `formats/checkpoint.md` |
| bad hyperparameters | boundary | negative lr, eps, or decay, a beta outside $[0, 1)$, no parameters, integer data | errors at construction, not NaN at step 9000 |

## 5. The solution, step by step

### From first principles

1. **Two EMAs per coordinate.** $m$ tracks the gradient's direction, $v$ its squared size (section 2.1). They have the parameter's shape and dtype and start at 0.
2. **Normalize to make the step a length.** Dividing $\hat m$ by $\sqrt{\hat v}$ removes the gradient's scale (section 2.2), so $\eta$ is a step size in parameter space.
3. **Correct the zero start.** Dividing by $1 - \beta_1^t$ and $1 - \beta_2^t$ makes the first step exactly $\eta$ (section 2.3), which requires $t$ to count from 1 at the first `step()`.
4. **Put $\epsilon$ where torch does.** Folding both corrections into one scalar `step_size = lr / (1 - beta1^t)` and dividing $\sqrt{v}$ by $\sqrt{1 - \beta_2^t}$ before adding $\epsilon$ reproduces torch to the last bit; adding $\epsilon$ elsewhere changes small-gradient coordinates.
5. **Two kinds of decay, one flag.** Adam folds $\lambda w$ into $g$ (so both moments see it); AdamW multiplies $w$ by $1 - \eta\lambda$ before the update and leaves $g$ alone (section 2.4). Everything else is shared, so `AdamW` is `Adam` with a class flag and a different default.
6. **Read hyperparameters on every step.** Schedules assign `opt.lr` between steps, so `step` reads `self.lr` each call.
7. **Update in place, skip `None` gradients.** The arrays are shared with the model; a parameter that got no gradient must not decay or change its moments, while the global $t$ still advances.
8. **State is $t$, $m$, $v$, and the hyperparameters, copied.** Restoring $t$ keeps the bias correction where it was (section 2.5).

### The code, annotated

```python
class Adam:
    _decoupled = False  # AdamW decays the weights directly instead

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0) -> None:
        _check_hyper(lr, betas, eps, weight_decay)       # lr, eps, decay >= 0; betas in [0, 1)
        self.params = list(params)
        if not self.params:
            raise ValueError("params is empty: nothing to optimize")
        for i, p in enumerate(self.params):
            d = getattr(p, "data", None)
            if not isinstance(d, np.ndarray) or not np.issubdtype(d.dtype, np.floating):
                raise ValueError(f"parameter {i}: data must be a floating-point ndarray")
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0                                       # becomes 1 on the first step
        # Moments in the parameter's own dtype, allocated up front so every
        # parameter has an entry in the checkpoint even before its first grad.
        self.exp_avg = [np.zeros_like(p.data) for p in self.params]
        self.exp_avg_sq = [np.zeros_like(p.data) for p in self.params]
```

Integer `data` is rejected because `p.data -= ...` on an integer array would truncate every update (or raise), a failure that shows up thousands of steps later as a model that never trains. The `_check_hyper` conditions are written `not lr >= 0.0`, so NaN is rejected too. Moments are allocated eagerly, unlike `M10.2`'s lazy buffers, because the checkpoint format stores one `exp_avg` and `exp_avg_sq` per parameter.

```python
    def step(self) -> None:
        self.t += 1                                      # t = 1 on the first call
        lr, (beta1, beta2), eps, wd = self.lr, self.betas, self.eps, self.weight_decay   # read now
        bias_correction1 = 1.0 - beta1**self.t
        bias_correction2_sqrt = math.sqrt(1.0 - beta2**self.t)
        step_size = lr / bias_correction1                # torch's arrangement: one scalar
        for p, m, v in zip(self.params, self.exp_avg, self.exp_avg_sq):
            g = p.grad
            if g is None:
                continue                                 # no decay, no moment update
            g = np.asarray(g, dtype=p.data.dtype)
            if g.shape != p.data.shape:
                raise ValueError(f"grad shape {g.shape} != parameter shape {p.data.shape}")
            if self._decoupled:
                p.data *= 1.0 - lr * wd                  # AdamW: shrink w, leave g alone
            elif wd != 0.0:
                g = g + wd * p.data                      # Adam: L2 gradient, seen by m and v
            m *= beta1
            m += (1.0 - beta1) * g                       # m_t = b1 m + (1 - b1) g
            v *= beta2
            v += (1.0 - beta2) * (g * g)                 # v_t = b2 v + (1 - b2) g^2
            denom = np.sqrt(v) / bias_correction2_sqrt + eps   # eps after the correction
            p.data -= step_size * (m / denom)            # in place, in p's dtype
```

Trace section 3's step 1: $t = 1$, `bias_correction1` $= 0.1$, `bias_correction2_sqrt` $= \sqrt{0.001}$, `step_size` $= 1$. With $g = 2$: $m = 0.2$, $v = 0.004$, `denom` $= \sqrt{0.004}/\sqrt{0.001} = 2$, and $x$ moves by $1 \cdot 0.2/2 = 0.1$ to $0.9$. Step 2 with $g = -1$: $m = 0.08$, $v = 0.004996$, `step_size` $= 0.1/0.19$, `denom` $= \sqrt{0.004996/0.001999}$, giving $x_2 \approx 0.873366$. With AdamW and $\lambda = 0.1$, the decay line first multiplies $x$ by $0.99$, and the same arithmetic lands at $0.854466$; with Adam and L2 the gradients become $2.1$ and $-0.91$, landing at $0.868123$. The moments are updated with `*=` and `+=`, so they stay the same arrays and keep the parameter's dtype; a float32 model never silently gets float64 moments.

```python
    def state_dict(self) -> dict:
        return {
            "step": self.t,
            "lr": self.lr,
            "betas": [self.betas[0], self.betas[1]],
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "exp_avg": [m.copy() for m in self.exp_avg],         # snapshots, not live arrays
            "exp_avg_sq": [v.copy() for v in self.exp_avg_sq],
        }

    def load_state_dict(self, sd: dict) -> None:
        ms, vs = list(sd["exp_avg"]), list(sd["exp_avg_sq"])
        # ... same number of moments as parameters, each with the parameter's shape ...
        betas = (float(sd["betas"][0]), float(sd["betas"][1]))
        _check_hyper(float(sd["lr"]), betas, float(sd["eps"]), float(sd["weight_decay"]))
        self.t = int(sd["step"])                         # restore t: the correction continues
        # ... lr, betas, eps, weight_decay ...
        self.exp_avg = [np.array(m, dtype=p.data.dtype, copy=True) for p, m in zip(self.params, ms)]
        self.exp_avg_sq = [np.array(v, dtype=p.data.dtype, copy=True) for p, v in zip(self.params, vs)]


class AdamW(Adam):
    _decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
```

As in `M10.2`, validation happens before assignment so a bad checkpoint leaves the optimizer unchanged, and loaded moments are copied and cast to the parameter's dtype. `AdamW` changes only the flag and the default `weight_decay` (0.01, torch's default).

Full solution: [`adamw.py`](../../../../course/solutions/python/tinyllm/optim/adamw.py).

### Why this solution

- **Cost.** Per step and parameter: about ten elementwise array operations and one square root; memory is two moment arrays per parameter, twice the model size, the main reason production systems shard or quantize optimizer state (Going further).
- **torch's arithmetic, exactly.** Writing the textbook form $\hat m / (\sqrt{\hat v} + \epsilon)$ is mathematically the same but places $\epsilon$ differently relative to the correction and rounds differently; matching `_single_tensor_adam` statement for statement is what makes the golden trajectories agree.
- **One global step counter.** torch keeps one per parameter; with one counter the checkpoint stores one step, and the two agree whenever every parameter gets a gradient every step. The tradeoff: a parameter that starts receiving gradients late gets a slightly wrong correction (Going further).
- **Inheritance over duplication.** The two optimizers differ in one branch; a class flag keeps the shared update in one place so a fix to one is a fix to both.

### Explain it back

**1. Why is the first Adam step exactly $\eta$ per coordinate when $\epsilon = 0$?**

<details><summary>Answer</summary>

At $t = 1$, $m_1 = (1 - \beta_1)g_1$ and $v_1 = (1 - \beta_2)g_1^2$. Dividing by $1 - \beta_1$ and $1 - \beta_2$ gives $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the update is $\eta\,g_1/\lvert g_1\rvert = \eta\,\mathrm{sign}(g_1)$, whatever the gradient's scale.

</details>

**2. Why does AdamW multiply the weights by $1 - \eta\lambda$ instead of adding $\lambda w$ to the gradient?**

<details><summary>Answer</summary>

Anything added to the gradient is divided by $\sqrt{\hat v}$. For a coordinate with large gradients the penalty becomes negligible; for one with zero gradient it becomes a full normalized step of size $\eta$ regardless of the weight's size. Shrinking the weights directly applies the same relative decay to every coordinate, which is what weight decay is meant to do. In the zero-gradient case of section 4, AdamW gives 3.8 and Adam with L2 gives 3.9.

</details>

**3. Where exactly is $\epsilon$ added, and why does it matter?**

<details><summary>Answer</summary>

To $\sqrt{v_t}/\sqrt{1 - \beta_2^t}$, that is, after the second moment is bias-corrected and square-rooted. This is torch's placement. Adding it inside the square root, or before the correction, changes the denominator by an amount comparable to $\sqrt{v}$ for coordinates whose gradients are near $\epsilon$ (rare features), so their steps would differ from torch's.

</details>

**4. What goes wrong after a resume if `load_state_dict` does not restore `step`?**

<details><summary>Answer</summary>

$t$ restarts at 0, so the next step uses $1 - \beta^1$ in both corrections while the moments already hold thousands of steps of history. $\hat v$ becomes about 1000 times too large and $\hat m$ about 10 times too large; the result is a step about $10/\sqrt{1000} \approx 0.32$ times the right size at first, and the trajectory no longer matches the uninterrupted run.

</details>

**5. Why does the optimizer read `self.lr` at the start of every `step` instead of storing it once?**

<details><summary>Answer</summary>

Schedules (`M10.4`) change the learning rate by assigning `opt.lr` before each step. A value cached at construction would silently ignore the schedule, and both the Adam step and AdamW's $1 - \eta\lambda$ decay must use the current rate.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| no bias correction, or correcting only one moment | the first steps are about 3 times too large (or too small); early loss spikes | `step_size = lr / (1 - beta1**t)` and `sqrt(v) / sqrt(1 - beta2**t)`; step 3 |
| counting $t$ from 0 inside the correction, or using $t + 1$ | $1 - \beta^0 = 0$ divides by zero, or every step is mis-scaled | `self.t += 1` is the first line of `step`, so $t = 1$ on the first update |
| AdamW adding $\lambda w$ to the gradient | AdamW behaves like Adam with L2: decay is normalized away for large-gradient weights | the `_decoupled` branch multiplies `p.data`; only Adam adds to `g`; step 5 |
| decaying by $1 - \lambda$ instead of $1 - \eta\lambda$ | with $\lambda = 0.1$ the weights lose 10% per step and collapse toward 0 | `p.data *= 1.0 - lr * wd` |
| $\epsilon$ inside the square root, or before the correction | rare-feature coordinates take steps of the wrong size; only visible when gradients are near $\epsilon$ | `np.sqrt(v) / bias_correction2_sqrt + eps`; step 4 |
| caching the learning rate at construction | the schedule (`M10.4`) is silently ignored | `step` reads `self.lr` on every call; step 6 |
| `p.data = p.data - ...` instead of `p.data -= ...` | the optimizer updates a copy; the model keeps its old weights | `p.data -= ...` and `p.data *= ...`; step 7 |
| returning live arrays from `state_dict`, or not restoring $t$ | a resumed run differs from the uninterrupted one; bias correction restarts after every resume | copies out and in, and `self.t = int(sd["step"])`; step 8 |
| decaying parameters whose grad is `None` | embeddings of tokens absent from the batch shrink anyway | the `None` check comes before the decay line; step 7 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M10.2` | the `Optimizer` protocol and SGD with momentum; Adam's $m$ is momentum's EMA form (reading) |
| Back | `M02.2` | the EMA as a geometric series and its bias correction, here applied per coordinate (reading) |
| Back | `S-M10a` | the Adam derivation problems check section 2.3 by hand (reading) |
| Forward | `L0.5` | `train_step` calls `opt.zero_grad()`, the backward pass, then `opt.step()` on every model in Part 0 |
| Forward | `L0.6` | the checkpoint writer stores `state_dict()` as `<name>.exp_avg`, `<name>.exp_avg_sq`, and the step, and resume calls `load_state_dict` |
| Forward | `M10.4` | schedules set `opt.lr` between steps; clipping runs before `step()` |
| Forward | `L4.1` | trains the encoder-decoder with teacher forcing |
| Forward | `L5.5` | the 2017 Transformer with the Noam schedule |
| Forward | `L6.1` | pretraining objectives |
| Forward | `L7.9` and `C1` | the modern decoder and the TinyStories capstone, betas $(0.9, 0.95)$, $\lambda = 0.1$, checkpoints with `exp_avg` and `exp_avg_sq` |
| Forward | `L12.1` | supervised fine-tuning |
| Forward | `L3.6` | the recurrent language models train with it |
| Forward | `L6.5` | trains with it |
| Forward | `L11.1` | trains with it |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `AdamW.step` loop over parameters | `torch.optim.AdamW` `foreach` and `fused` paths | one kernel launch for all parameters at once; the fused CUDA kernel also handles AMP grad scaling | `torch/optim/adam.py`, `_multi_tensor_adam`, `_fused_adam` |
| one global step counter | torch per-parameter `state["step"]` | correct bias correction for parameters that start receiving gradients late | `torch/optim/adam.py` |
| float32 moments | 8-bit optimizers (bitsandbytes) | block-wise quantized $m$ and $v$: optimizer memory drops by 4 times | Dettmers et al., "8-bit Optimizers via Block-wise Quantization" (2022) |
| state on one process | ZeRO stage 1 (DeepSpeed, FSDP) | each data-parallel rank owns a shard of $m$ and $v$; the optional `L11.3` | Rajbhandari et al., "ZeRO" (2020) |
| AdamW for every matrix | Muon | orthogonalized momentum for 2-D weights, AdamW for the rest; the optional `M10.6` | Jordan et al., "Muon" (2024) |
