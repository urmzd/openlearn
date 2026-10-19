# Adam and AdamW: bias correction and decoupled weight decay

The training loop you build in [Training loop and the autograd bigram](../../build-an-llm/01-foundations/06-training-loop-and-the-autograd-bigram.md) from *Build Your Own LLM System* needs an optimizer that trains every model in that book without retuning: a bigram table, a 64-unit MLP on digits, an LSTM, a Transformer, and the 10M-parameter Llama-style model of [Capstone: TinyStories, owned end to end](../../build-an-llm/capstones/01-tinystories.md). Plain SGD with momentum, from [The Optimizer protocol: SGD, momentum, Nesterov, weight decay](02-optimizer-protocol-and-sgd.md), needs a learning rate tuned per model. Worse, inside one model it is too slow for layers with small gradients and unstable for layers with large ones: an embedding row that sees a rare token gets a gradient a thousand times smaller than a layer norm gain.

Adam fixes this by rescaling each coordinate by its own gradient history, and AdamW is the version every modern language model trains with. The capstone's training spec defaults to `adamw` with betas `(0.9, 0.95)`, and the checkpoint format already reserves `exp_avg` and `exp_avg_sq` for its state, so that a platform workflow such as [Platform workflows: TrainRun and EvalSuite](../../ai-platform-engineering/05-durable-orchestration-and-workers/11-trainrun-and-evalsuite.md) in *AI Platform Engineering* can kill a training worker and resume it. This section builds that optimizer and checks that it equals `torch.optim.AdamW` step for step.

The section builds on the `Optimizer` protocol of the previous section and on the exponential moving average and its bias correction from [Series convergence, the EMA as a geometric series, and bias correction](../03-calculus-2/02-series-and-the-ema.md). The Adam derivation problems in [Optimization problem set, part a](05-problem-set-a.md) check the algebra by hand. The reference solution is [`course/solutions/python/tinyllm/optim/adamw.py`](../../../../course/solutions/python/tinyllm/optim/adamw.py), which provides `Adam` and `AdamW` with `step`, `zero_grad`, `state_dict`, and `load_state_dict`; the contract is [`course/contracts/py/tinyllm/optim/adamw.pyi`](../../../../course/contracts/py/tinyllm/optim/adamw.pyi), and the checkpoint keys are fixed by [`formats/checkpoint.md`](../../../../course/contracts/formats/checkpoint.md).

> **Learning objectives**
> - Write Adam's two moving averages and its normalized update, and explain why the update is invariant to the gradient's scale.
> - Derive the bias correction from the geometric sum and show that the first step is exactly $\eta$ per coordinate.
> - Explain why weight decay as an L2 gradient fails under Adam and how AdamW's decoupled decay fixes it.
> - Implement `Adam` and `AdamW` that match torch's arithmetic, including where $\epsilon$ goes, and resume bit for bit.

## Notation

The symbols below follow torch's names where they appear in code. All operations on vectors are elementwise: $g^2$ squares each coordinate, $\sqrt{v}$ takes each square root, and $m / \sqrt{v}$ divides coordinate by coordinate.

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

## Moving averages of the gradient

Gradient descent moves $w$ against the gradient: $w \leftarrow w - \eta\, g_t$. A minibatch gradient is noisy, and its direction changes from step to step. Momentum smooths it with an **exponential moving average** (EMA):

$$m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t, \qquad m_0 = 0 .$$

Unrolling the recursion shows what $m_t$ is: a weighted sum of all past gradients with geometrically decaying weights, $m_t = (1 - \beta_1) \sum_{s=1}^{t} \beta_1^{t-s} g_s$. With $\beta_1 = 0.9$ the last 10 or so steps carry most of the weight. Adam keeps a second EMA, of the squared gradient:

$$v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2, \qquad v_0 = 0 .$$

With $\beta_2 = 0.999$ it remembers about 1000 steps. $v_t$ estimates $\mathbb{E}[g^2]$, the mean squared size of this coordinate's gradient.

## Normalizing by the second moment

The second average exists so that Adam can divide by it. The update divides the first moment by the square root of the second:

$$w_t = w_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} .$$

$\sqrt{\hat v_t}$ has the units of the gradient, so the ratio $\hat m_t / \sqrt{\hat v_t}$ has no units. Multiplying every gradient by a constant $c > 0$ multiplies both $\hat m_t$ and $\sqrt{\hat v_t}$ by $c$ and leaves the update unchanged (when $\epsilon$ is negligible). Each coordinate's step is therefore about $\eta$ in size when its gradient keeps a consistent sign, and smaller when the sign flips, because positive and negative gradients cancel in $m$ but not in $v$, so $|\hat m| \ll \sqrt{\hat v}$. The learning rate becomes a step length in parameter space, which is why one value such as $3 \times 10^{-4}$ works across many models. $\epsilon$ only matters when $\sqrt{\hat v}$ is near or below it, for coordinates whose gradients are almost always zero.

## Bias correction

The hats in that formula need explaining. Both averages start at 0, so early on they are too small. Take every $g_s$ equal to the same value $g$. Then $m_t = (1 - \beta_1)\, g \sum_{s=0}^{t-1} \beta_1^{s} = (1 - \beta_1^t)\, g$ by the geometric sum from [Sequences, geometric series, and frequency ladders](../01-precalculus/03-sequences-and-frequency-ladders.md). The average is short of $g$ by exactly the factor $1 - \beta_1^t$, and dividing by it removes the bias:

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t} .$$

For random gradients with constant mean the same argument holds in expectation. The factor matters most for $v$: with $\beta_2 = 0.999$, $1 - \beta_2^{1} = 0.001$, so the uncorrected $v_1$ is 1000 times too small and the first step would be $\sqrt{1000} \approx 32$ times too large in the ratio $m / \sqrt{v}$. That is partly offset by the uncorrected $m$, which is 10 times too small, for a net of $3.16$ times too large. After a few thousand steps both factors are 1 and the correction disappears.

At $t = 1$ the corrections give $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the first update is $\eta\, g_1 / (|g_1| + \epsilon)$: exactly $\eta$ times the sign of $g_1$ when $\epsilon = 0$. torch folds the corrections into the step for speed, and the solution uses the same arrangement, because it decides where $\epsilon$ goes:

$$w_t = w_{t-1} - \frac{\eta}{1 - \beta_1^t}\cdot \frac{m_t}{\sqrt{v_t} / \sqrt{1 - \beta_2^t} + \epsilon} .$$

## Decoupled weight decay

**Weight decay** shrinks the weights toward 0 a little on every step, which regularizes the model. For SGD it is the same as adding the penalty $\frac{\lambda}{2}\lVert w \rVert^2$ to the loss, whose gradient is $\lambda w$: the step $w - \eta(g + \lambda w) = (1 - \eta\lambda)\, w - \eta g$ decays $w$ by the factor $1 - \eta\lambda$.

For Adam the two are not the same, and the normalization you just saw is the reason. **Adam with L2** (`torch.optim.Adam(weight_decay=...)`) adds $\lambda w$ to $g$ before both moments see it, and then the normalization divides it by $\sqrt{\hat v}$. A coordinate with large gradients gets almost no decay, and with $g = 0$ the penalty itself becomes a normalized step of size $\eta$, whatever the weight's size. **AdamW** applies the decay directly to the weights, before the Adam update, and leaves the gradient alone:

$$w \leftarrow (1 - \eta\lambda)\, w, \quad \text{then the Adam step on } g_t .$$

Every coordinate shrinks by the same factor, as intended. The decay is scaled by $\eta$, so a schedule that lowers $\eta$ (the subject of [the next section](04-schedules-and-gradient-clipping.md)) lowers the decay with it. Parameters whose gradient is `None` (the loss did not reach them this step) are skipped entirely: no decay, no moment update.

## Saving and restoring the state

The optimizer state is $t$, $m$, and $v$ for every parameter, plus the hyperparameters. Training that stops and resumes must continue as if it never stopped, so `state_dict` returns **copies** of everything, and `load_state_dict` restores all of it, including $t$: with $t$ reset to 0 the bias correction would restart and the next step would be badly mis-scaled. `formats/checkpoint.md` writes the moments as `<name>.exp_avg` and `<name>.exp_avg_sq`, in the parameter's dtype.

## Working it by hand

Take one scalar parameter $x_0 = 1$, gradients $g_1 = 2$ then $g_2 = -1$ (given, not computed from a loss), $\eta = 0.1$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 0$, and no weight decay. The two steps work out as follows:

| Step | $m_t$ | $v_t$ | $1 - \beta_1^t$ | $1 - \beta_2^t$ | $\hat m_t$ | $\hat v_t$ | update $\eta \hat m / \sqrt{\hat v}$ | $x_t$ |
|---|---|---|---|---|---|---|---|---|
| 1 | $0.1 \cdot 2 = 0.2$ | $0.001 \cdot 4 = 0.004$ | $0.1$ | $0.001$ | $2$ | $4$ | $0.1 \cdot 2 / 2 = 0.1$ | $0.9$ |
| 2 | $0.9 \cdot 0.2 + 0.1 \cdot (-1) = 0.08$ | $0.999 \cdot 0.004 + 0.001 \cdot 1 = 0.004996$ | $0.19$ | $0.001999$ | $8/19 \approx 0.42105$ | $4996/1999 \approx 2.49925$ | $0.1 \cdot 0.42105 / 1.58090 \approx 0.026634$ | $\approx 0.873366$ |

Two things are worth seeing. Step 1 moves by exactly $\eta = 0.1$: that is bias correction at work (without it, $0.1 \cdot 0.2 / \sqrt{0.004} \approx 0.316$). Step 2 has a negative gradient, yet $x$ keeps decreasing: $m$ still points the old way, and the step is smaller because $\hat v$ remembers the large first gradient.

Now add weight decay $\lambda = 0.1$. AdamW first multiplies $x$ by $1 - \eta\lambda = 0.99$: step 1 gives $0.99 - 0.1 = 0.89$; step 2 gives $0.89 \cdot 0.99 - 0.026634 \approx 0.854466$. Adam with L2 instead feeds $g_1 + \lambda x_0 = 2.1$ and $g_2 + \lambda x_1 = -1 + 0.09 = -0.91$ to the moments. Step 1 still moves by exactly $0.1$ (the first step is $\eta$ times a sign, whatever the gradient), to $0.9$; step 2 lands at $\approx 0.868123$. The penalty changed $x_2$ by $0.0052$ under L2 and by $0.0189$ under AdamW.

## Designing the interface

The two classes share one interface:

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

A parameter is any object with `data` (a float ndarray) and `grad` (an ndarray or `None`). A two-attribute class is enough, and the training loop passes the autograd tensors of [Tensor, broadcasting backward, and no_grad](../../build-an-llm/01-foundations/02-tensor-and-broadcasting-backward.md). The step counter $t$ is one global count of `step()` calls, as `formats/checkpoint.md` stores one step. torch keeps a counter per parameter instead; the two agree whenever every parameter has a gradient on every step, which is how the models in this series train. A schedule changes the learning rate by assigning `opt.lr` before `step()`.

The tests hold the implementation to a long list of behaviors, each tied to a downstream need. It must reproduce both hand-worked steps to float64 precision, and the $\lambda = 0.1$ variants for both kinds of decay, since that is the one difference between the classes. It must match 20 steps of torch AdamW on two tensors, including the final moments, and 20 steps of torch Adam with `weight_decay = 0.1`, where L2 coupling feeds both moments. A learning rate changed on every step (with betas `(0.9, 0.95)`) must be used, because schedules write `opt.lr`. Gradients of size $10^{-6}$ with $\epsilon = 10^{-6}$ must match torch, which is where the placement of $\epsilon$ becomes visible. float32 parameters must follow torch's float32 trajectory, with float32 moments, because model parameters are float32 (see [Module system and basic layers](../../build-an-llm/01-foundations/05-module-system-and-layers.md)).

Two laws are checked as properties: with $\epsilon = 0$ the first step is $\eta \cdot \mathrm{sign}(g)$ at every scale, and gradients multiplied by $10^{-3}$ or $250$ give the same trajectory. With zero gradient, starting from the test's weights, AdamW gives $3.8$ and Adam with L2 gives $3.9$: decoupling, in one line. A parameter with `grad = None` gets no decay and no moment change, while $t$ still advances. The update is in place: `p.data` keeps its identity and a view writes through to its buffer, because the model and the flat buffers of large-scale training hold the arrays. `zero_grad` sets every grad to `None`, since the autograd accumulates into `grad`. Five steps, save, load into a fresh optimizer, and five more equal ten steps bit for bit; a saved dict does not change after later steps; the layout's keys and shapes are checked, hyperparameters restored, and mismatches rejected. Finally, bad hyperparameters (negative lr, eps, or decay, a beta outside $[0, 1)$, no parameters, integer data) are errors at construction, not NaN at step 9000.

## Building it step by step

The design follows from the principles in eight steps.

1. **Two EMAs per coordinate.** $m$ tracks the gradient's direction, $v$ its squared size. They have the parameter's shape and dtype and start at 0.
2. **Normalize to make the step a length.** Dividing $\hat m$ by $\sqrt{\hat v}$ removes the gradient's scale, so $\eta$ is a step size in parameter space.
3. **Correct the zero start.** Dividing by $1 - \beta_1^t$ and $1 - \beta_2^t$ makes the first step exactly $\eta$, which requires $t$ to count from 1 at the first `step()`.
4. **Put $\epsilon$ where torch does.** Folding both corrections into one scalar `step_size = lr / (1 - beta1^t)` and dividing $\sqrt{v}$ by $\sqrt{1 - \beta_2^t}$ before adding $\epsilon$ reproduces torch to the last bit; adding $\epsilon$ elsewhere changes small-gradient coordinates.
5. **Two kinds of decay, one flag.** Adam folds $\lambda w$ into $g$ (so both moments see it); AdamW multiplies $w$ by $1 - \eta\lambda$ before the update and leaves $g$ alone. Everything else is shared, so `AdamW` is `Adam` with a class flag and a different default.
6. **Read hyperparameters on every step.** Schedules assign `opt.lr` between steps, so `step` reads `self.lr` each call.
7. **Update in place, skip `None` gradients.** The arrays are shared with the model; a parameter that got no gradient must not decay or change its moments, while the global $t$ still advances.
8. **State is $t$, $m$, $v$, and the hyperparameters, copied.** Restoring $t$ keeps the bias correction where it was.

The constructor validates and allocates:

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

Integer `data` is rejected because `p.data -= ...` on an integer array would truncate every update (or raise), a failure that shows up thousands of steps later as a model that never trains. The `_check_hyper` conditions are written `not lr >= 0.0`, so NaN is rejected too. Moments are allocated eagerly, unlike SGD's lazy buffers, because the checkpoint format stores one `exp_avg` and `exp_avg_sq` per parameter.

The step is a direct transcription of torch's single-tensor path:

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

Trace the hand example's step 1 through it. With $t = 1$, `bias_correction1` $= 0.1$, `bias_correction2_sqrt` $= \sqrt{0.001}$, and `step_size` $= 1$. With $g = 2$: $m = 0.2$, $v = 0.004$, `denom` $= \sqrt{0.004}/\sqrt{0.001} = 2$, and $x$ moves by $1 \cdot 0.2/2 = 0.1$ to $0.9$. Step 2 with $g = -1$ gives $m = 0.08$, $v = 0.004996$, `step_size` $= 0.1/0.19$, `denom` $= \sqrt{0.004996/0.001999}$, and $x_2 \approx 0.873366$. With AdamW and $\lambda = 0.1$, the decay line first multiplies $x$ by $0.99$, and the same arithmetic lands at $0.854466$; with Adam and L2 the gradients become $2.1$ and $-0.91$, landing at $0.868123$. The moments are updated with `*=` and `+=`, so they stay the same arrays and keep the parameter's dtype; a float32 model never silently gets float64 moments.

Saving, loading, and the AdamW subclass complete the code:

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

As with SGD, validation happens before assignment so a bad checkpoint leaves the optimizer unchanged, and loaded moments are copied and cast to the parameter's dtype. `AdamW` changes only the flag and the default `weight_decay` (0.01, torch's default).

The full solution is in [`adamw.py`](../../../../course/solutions/python/tinyllm/optim/adamw.py).

### Why this design

Per step and parameter, the cost is about ten elementwise array operations and one square root. Memory is two moment arrays per parameter, twice the model size, which is the main reason production systems shard or quantize optimizer state (see Further reading).

The code matches torch's arithmetic exactly. Writing the textbook form $\hat m / (\sqrt{\hat v} + \epsilon)$ is mathematically the same but places $\epsilon$ differently relative to the correction and rounds differently; matching `_single_tensor_adam` statement for statement is what makes the golden trajectories agree.

There is one global step counter. torch keeps one per parameter; with one counter the checkpoint stores one step, and the two agree whenever every parameter gets a gradient every step. The tradeoff is that a parameter that starts receiving gradients late gets a slightly wrong correction.

Inheritance beats duplication here. The two optimizers differ in one branch, and a class flag keeps the shared update in one place, so a fix to one is a fix to both.

## What can go wrong

Most Adam bugs are scaling bugs that train, just badly. These are the ones to look for:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| no bias correction, or correcting only one moment | the first steps are about 3 times too large (or too small); early loss spikes | `step_size = lr / (1 - beta1**t)` and `sqrt(v) / sqrt(1 - beta2**t)` |
| counting $t$ from 0 inside the correction, or using $t + 1$ | $1 - \beta^0 = 0$ divides by zero, or every step is mis-scaled | `self.t += 1` is the first line of `step`, so $t = 1$ on the first update |
| AdamW adding $\lambda w$ to the gradient | AdamW behaves like Adam with L2: decay is normalized away for large-gradient weights | the `_decoupled` branch multiplies `p.data`; only Adam adds to `g` |
| decaying by $1 - \lambda$ instead of $1 - \eta\lambda$ | with $\lambda = 0.1$ the weights lose 10% per step and collapse toward 0 | `p.data *= 1.0 - lr * wd` |
| $\epsilon$ inside the square root, or before the correction | rare-feature coordinates take steps of the wrong size; only visible when gradients are near $\epsilon$ | `np.sqrt(v) / bias_correction2_sqrt + eps` |
| caching the learning rate at construction | the schedule is silently ignored | `step` reads `self.lr` on every call |
| `p.data = p.data - ...` instead of `p.data -= ...` | the optimizer updates a copy; the model keeps its old weights | `p.data -= ...` and `p.data *= ...` |
| returning live arrays from `state_dict`, or not restoring $t$ | a resumed run differs from the uninterrupted one; bias correction restarts after every resume | copies out and in, and `self.t = int(sd["step"])` |
| decaying parameters whose grad is `None` | embeddings of tokens absent from the batch shrink anyway | the `None` check comes before the decay line |

## Where this leads

From here on, nearly every model in *Build Your Own LLM System* trains with this optimizer. In [Training loop and the autograd bigram](../../build-an-llm/01-foundations/06-training-loop-and-the-autograd-bigram.md), `train_step` calls `opt.zero_grad()`, the backward pass, then `opt.step()` on every model of its part. [Safetensors for every dtype, atomic checkpoints, and the token stream](../../build-an-llm/01-foundations/07-safetensors-checkpoints-and-token-streams.md) stores `state_dict()` as `<name>.exp_avg`, `<name>.exp_avg_sq`, and the step, and resume calls `load_state_dict`. The next section, [Learning-rate schedules (cosine, WSD, Noam) and gradient clipping](04-schedules-and-gradient-clipping.md), sets `opt.lr` between steps and clips before `step()`.

Later, AdamW trains the encoder-decoder of [Encoder-decoder with teacher forcing](../../build-an-llm/05-attention-origins/01-encoder-decoder-teacher-forcing.md), the 2017 Transformer with the Noam schedule in [Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing](../../build-an-llm/06-transformer-2017/05-encoder-decoder-transformer.md), the pretraining objectives starting with [GPT: the decoder-only language model](../../build-an-llm/07-objectives/01-gpt-decoder-only.md), the fine-tuning heads of [Fine-tuning heads and the linear-head export](../../build-an-llm/07-objectives/05-fine-tuning-heads-and-the-linear-head-export.md), and the recurrent models of [RNN language model with stateful TBPTT](../../build-an-llm/04-recurrent/06-rnn-language-model-stateful-tbptt.md). The modern decoder of [The Llama-Family Model: Config, Weight Loading, and Download](../../build-an-llm/08-modern-block/09-llama-family-model.md) and the TinyStories capstone use betas $(0.9, 0.95)$ and $\lambda = 0.1$, with checkpoints holding `exp_avg` and `exp_avg_sq`. [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](../../build-an-llm/12-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) and [Supervised fine-tuning: chat rendering, assistant-only loss, packing](../../build-an-llm/13-post-training/01-supervised-fine-tuning.md) train with it too.

### Further reading

- Kingma and Ba, ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) (2015), sections 2 and 3; Loshchilov and Hutter, ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) (2019), sections 2 and 3; the torch source `torch/optim/adam.py`, `_single_tensor_adam`.
- `torch.optim.AdamW`'s `foreach` and `fused` paths (`torch/optim/adam.py`, `_multi_tensor_adam`, `_fused_adam`) update all parameters in one kernel launch; the fused CUDA kernel also handles AMP grad scaling.
- torch's per-parameter `state["step"]` gives correct bias correction for parameters that start receiving gradients late.
- Dettmers et al., "8-bit Optimizers via Block-wise Quantization" (2022): bitsandbytes quantizes $m$ and $v$ block-wise, cutting optimizer memory by 4 times.
- Rajbhandari et al., "ZeRO" (2020): with ZeRO stage 1 (DeepSpeed, FSDP) each data-parallel rank owns a shard of $m$ and $v$; see [Data parallelism and ZeRO stages 1 to 3](../../build-an-llm/12-training-at-scale/03-ddp-and-zero.md).
- Jordan et al., ["Muon"](https://kellerjordan.github.io/posts/muon/) (2024): orthogonalized momentum for 2-D weights, AdamW for the rest.

## Key terms

- **Exponential moving average (EMA)**: a recursive average $m_t = \beta m_{t-1} + (1 - \beta) x_t$ whose weights decay geometrically with age.
- **First moment ($m$, `exp_avg`)**: the EMA of the gradient.
- **Second moment ($v$, `exp_avg_sq`)**: the EMA of the elementwise squared gradient.
- **Bias correction**: dividing $m_t$ and $v_t$ by $1 - \beta_1^t$ and $1 - \beta_2^t$ to undo the zero initialization.
- **Adam with L2**: weight decay added to the gradient, so it passes through the normalization.
- **Decoupled weight decay (AdamW)**: multiplying the weights by $1 - \eta\lambda$ outside the gradient path.

## Summary

- Adam keeps two exponential moving averages per coordinate, $m$ of the gradient and $v$ of its square, and moves each coordinate by $\eta\, \hat m / (\sqrt{\hat v} + \epsilon)$.
- Bias correction divides $m_t$ by $1 - \beta_1^t$ and $v_t$ by $1 - \beta_2^t$, which undoes the pull toward the zero they start from; the first step is then exactly $\eta$ per coordinate.
- Dividing by $\sqrt{\hat v}$ makes the update invariant to the scale of the gradient, so one learning rate serves layers whose gradients differ by orders of magnitude.
- AdamW decouples weight decay: it multiplies the weights by $1 - \eta\lambda$ instead of adding $\lambda w$ to the gradient, where Adam's normalization would wash the penalty out.
- The optimizer state is part of the model: a save and load between steps must reproduce the uninterrupted run bit for bit.

## Review questions

1. Why is the first Adam step exactly $\eta$ per coordinate when $\epsilon = 0$?

   <details><summary>Answer</summary>

   At $t = 1$, $m_1 = (1 - \beta_1)g_1$ and $v_1 = (1 - \beta_2)g_1^2$. Dividing by $1 - \beta_1$ and $1 - \beta_2$ gives $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the update is $\eta\,g_1/\lvert g_1\rvert = \eta\,\mathrm{sign}(g_1)$, whatever the gradient's scale.

   </details>

2. Why does AdamW multiply the weights by $1 - \eta\lambda$ instead of adding $\lambda w$ to the gradient?

   <details><summary>Answer</summary>

   Anything added to the gradient is divided by $\sqrt{\hat v}$. For a coordinate with large gradients the penalty becomes negligible; for one with zero gradient it becomes a full normalized step of size $\eta$ regardless of the weight's size. Shrinking the weights directly applies the same relative decay to every coordinate, which is what weight decay is meant to do. In the zero-gradient test case, AdamW gives 3.8 and Adam with L2 gives 3.9.

   </details>

3. Where exactly is $\epsilon$ added, and why does it matter?

   <details><summary>Answer</summary>

   To $\sqrt{v_t}/\sqrt{1 - \beta_2^t}$, that is, after the second moment is bias-corrected and square-rooted. This is torch's placement. Adding it inside the square root, or before the correction, changes the denominator by an amount comparable to $\sqrt{v}$ for coordinates whose gradients are near $\epsilon$ (rare features), so their steps would differ from torch's.

   </details>

4. What goes wrong after a resume if `load_state_dict` does not restore `step`?

   <details><summary>Answer</summary>

   $t$ restarts at 0, so the next step uses $1 - \beta^1$ in both corrections while the moments already hold thousands of steps of history. $\hat v$ becomes about 1000 times too large and $\hat m$ about 10 times too large; the result is a step about $10/\sqrt{1000} \approx 0.32$ times the right size at first, and the trajectory no longer matches the uninterrupted run.

   </details>

5. Why does the optimizer read `self.lr` at the start of every `step` instead of storing it once?

   <details><summary>Answer</summary>

   Schedules change the learning rate by assigning `opt.lr` before each step. A value cached at construction would silently ignore the schedule, and both the Adam step and AdamW's $1 - \eta\lambda$ decay must use the current rate.

   </details>
