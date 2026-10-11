<!-- ol:module M10.3 -->
# Adam and AdamW: bias correction and decoupled weight decay

## Overview

| | |
|---|---|
| **Module** | `M10.3` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/optim/adamw.py`: `Adam` and `AdamW`, with `step`, `zero_grad`, `state_dict`, `load_state_dict` |
| **Contract** | [`course/contracts/py/tinyllm/optim/adamw.pyi`](../../course/contracts/py/tinyllm/optim/adamw.pyi) · checkpoint keys: [`formats/checkpoint.md`](../../course/contracts/formats/checkpoint.md) |
| **Tests** | `course/tests/M10.3/test_adamw.py` (what they check: section 4); golden trajectories from torch 2.14 in `course/fixtures/M10.3/adam_torch.json` |
| **Needs** | no code dependency. Reading: `M10.2` (the `Optimizer` protocol and SGD with momentum), `M02.2` (the exponential moving average and its bias correction), `S-M10a` (the Adam derivation problems) |
| **Used by** | `L0.5` trains every model with it from Part 0 on · `L0.6` saves and restores its state in checkpoints · later `L4.1`, `L5.5`, `L6.1`, `L7.9`, `C1`, and `L12.1` · later: `L11.1`, `L3.6`, `L6.5` |
| **Milestone** | `MS-P2` (Pass 2 closes with every math module it teaches passing) |
| **Optional depth** | Kingma and Ba, "Adam: A Method for Stochastic Optimization" (2015), sections 2 and 3; Loshchilov and Hutter, "Decoupled Weight Decay Regularization" (2019), sections 2 and 3; the torch source `torch/optim/adam.py`, `_single_tensor_adam` |

## Key Takeaways

- **Adam** keeps two exponential moving averages per coordinate, $m$ of the gradient and $v$ of its square, and moves each coordinate by $\eta\, \hat m / (\sqrt{\hat v} + \epsilon)$ (`test_hand_example_two_steps`, `test_matches_torch_adamw_trajectory`).
- **Bias correction** divides $m_t$ by $1 - \beta_1^t$ and $v_t$ by $1 - \beta_2^t$, which undoes the pull toward the zero they start from; the first step is then exactly $\eta$ per coordinate (`test_first_step_moves_each_coordinate_by_lr`).
- Dividing by $\sqrt{\hat v}$ makes the update **invariant to the scale of the gradient**, so one learning rate serves layers whose gradients differ by orders of magnitude (`test_update_ignores_gradient_scale`).
- **AdamW decouples weight decay**: it multiplies the weights by $1 - \eta\lambda$ instead of adding $\lambda w$ to the gradient, where Adam's normalization would wash the penalty out (`test_decoupled_decay_with_zero_gradient`).
- The optimizer **state is part of the model**: a save and load between steps must reproduce the uninterrupted run bit for bit (`test_resume_is_bitwise`).

## How to work this chapter

```bash
ol start M10.3              # stubs python/tinyllm/optim/adamw.py, contract alongside
ol tests M10.3              # read the test catalog first: rung R0, you write no tests here
ol check M10.3              # exit code is the verdict
ol diff  M10.3              # after passing: your code against the reference
```

---

## 1. Why now

Your training loop does not exist yet. `L0.5` builds it next, and it needs an optimizer that trains every model in the course without retuning: a bigram table, a 64-unit MLP on digits, an LSTM, a Transformer, and the 10M-parameter Llama-style model of `C1`. Plain SGD with momentum (`M10.2`) needs a learning rate tuned per model, and inside one model it is too slow for layers with small gradients and unstable for layers with large ones: an embedding row that sees a rare token gets a gradient a thousand times smaller than a layer norm gain. Adam fixes this by rescaling each coordinate by its own gradient history, and AdamW is the version every modern language model trains with. The `C1` training spec defaults to `adamw` with betas `(0.9, 0.95)`, and the checkpoint format already reserves `exp_avg` and `exp_avg_sq` for its state, so `dur.11` can kill a training worker and resume it. This module builds that optimizer and proves it equals `torch.optim.AdamW` step for step.

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

At $t = 1$ the corrections give $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the first update is $\eta\, g_1 / (|g_1| + \epsilon)$: exactly $\eta$ times the sign of $g_1$ when $\epsilon = 0$. torch folds the corrections into the step for speed, and you implement the same arrangement, because it decides where $\epsilon$ goes:

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

Two things to see. Step 1 moves by exactly $\eta = 0.1$: that is bias correction at work (without it, $0.1 \cdot 0.2 / \sqrt{0.004} \approx 0.316$). Step 2 has a negative gradient, yet $x$ keeps decreasing: $m$ still points the old way, and the step is smaller because $\hat v$ remembers the large first gradient. This is the first test, `test_hand_example_two_steps`.

**With weight decay $\lambda = 0.1$.** AdamW first multiplies $x$ by $1 - \eta\lambda = 0.99$: step 1 gives $0.99 - 0.1 = 0.89$; step 2 gives $0.89 \cdot 0.99 - 0.026634 \approx 0.854466$. Adam with L2 instead feeds $g_1 + \lambda x_0 = 2.1$ and $g_2 + \lambda x_1 = -1 + 0.09 = -0.91$ to the moments: step 1 still moves by exactly $0.1$ (the first step is $\eta$ times a sign, whatever the gradient), to $0.9$; step 2 lands at $\approx 0.868123$. The penalty changed $x_2$ by $0.0052$ under L2 and by $0.0189$ under AdamW (`test_hand_example_adamw_and_adam_l2_differ`).

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

A parameter is any object with `data` (a float ndarray) and `grad` (an ndarray or `None`); the tests use a two-attribute class, and `L0.5` passes autograd tensors (`L0.1`). The step counter $t$ is one global count of `step()` calls, as `formats/checkpoint.md` stores one step. torch keeps a counter per parameter instead; the two agree whenever every parameter has a gradient on every step, which is how the course trains. A schedule changes the learning rate by assigning `opt.lr` before `step()`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_two_steps` | unit | section 3, both steps, to float64 precision | you and the test agree on the update before any code |
| `test_hand_example_adamw_and_adam_l2_differ` | unit | section 3 with $\lambda = 0.1$ for both kinds of decay | the one difference between the two classes |
| `test_matches_torch_adamw_trajectory` | golden | 20 steps of torch AdamW on two tensors, and the final moments | every training run assumes torch's update |
| `test_matches_torch_adam_l2_trajectory` | golden | 20 steps of torch Adam with `weight_decay = 0.1` | L2 coupling feeds both moments |
| `test_lr_set_between_steps_is_used` | golden | lr changes on every step (betas `(0.9, 0.95)`) | `M10.4` schedules write `opt.lr` |
| `test_eps_is_added_after_bias_correction` | golden | gradients of size $10^{-6}$ with $\epsilon = 10^{-6}$ | where $\epsilon$ goes is visible for rare features |
| `test_float32_parameters_stay_float32` | golden | torch's float32 trajectory; moments stay float32 | model parameters are float32 (`L0.4`) |
| `test_first_step_moves_each_coordinate_by_lr` | property | with $\epsilon = 0$ the first step is $\eta \cdot \mathrm{sign}(g)$ at every scale | bias correction, stated as a law |
| `test_update_ignores_gradient_scale` | property | gradients times $10^{-3}$ or $250$ give the same trajectory | one lr for every layer |
| `test_decoupled_decay_with_zero_gradient` | unit | $g = 0$: AdamW gives $3.8$, Adam with L2 gives $3.9$ | decoupling, in one line |
| `test_parameter_without_grad_is_untouched` | boundary | `grad = None`: no decay, no moment change, $t$ still advances | unused embeddings and frozen branches |
| `test_update_is_in_place` | unit | `p.data` keeps its identity, a view writes through to its buffer | the model and `L11` flat buffers hold the arrays |
| `test_zero_grad_sets_none` | unit | every grad becomes `None` | `L0.1` accumulates into `grad` |
| `test_resume_is_bitwise` | property | 5 steps, save, load into a fresh optimizer, 5 steps equals 10 steps bit for bit | `dur.11` kill and resume |
| `test_state_dict_is_a_snapshot` | unit | later steps do not change a saved dict | `L0.6` may write it after training continues |
| `test_state_dict_layout` | unit | keys, shapes, hyperparameters restored, mismatches rejected | `formats/checkpoint.md` |
| `test_rejects_bad_hyperparameters` | boundary | negative lr, eps, or decay, a beta outside $[0, 1)$, no parameters, integer data | errors at construction, not NaN at step 9000 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| no bias correction, or correcting only one moment | the first steps are about 3 times too large (or too small); early loss spikes | `test_first_step_moves_each_coordinate_by_lr`, `test_hand_example_two_steps` (mutants `s01`, `s03`) |
| counting $t$ from 0 inside the correction, or using $t + 1$ | $1 - \beta^0 = 0$ divides by zero, or every step is mis-scaled | `test_hand_example_two_steps` (mutant `s02`) |
| AdamW adding $\lambda w$ to the gradient | AdamW behaves like Adam with L2: decay is normalized away for large-gradient weights | `test_decoupled_decay_with_zero_gradient`, `test_hand_example_adamw_and_adam_l2_differ` (mutant `s05`) |
| decaying by $1 - \lambda$ instead of $1 - \eta\lambda$ | with $\lambda = 0.1$ the weights lose 10% per step and collapse toward 0 | `test_matches_torch_adamw_trajectory` (mutant `s06`) |
| $\epsilon$ inside the square root, or before the correction | rare-feature coordinates take steps of the wrong size; only visible when gradients are near $\epsilon$ | `test_eps_is_added_after_bias_correction` (mutants `s08`, `s14`) |
| caching the learning rate at construction | the schedule (`M10.4`) is silently ignored | `test_lr_set_between_steps_is_used` (mutant `s13`) |
| `p.data = p.data - ...` instead of `p.data -= ...` | the optimizer updates a copy; the model keeps its old weights | `test_update_is_in_place` (mutant `s16`) |
| returning live arrays from `state_dict`, or not restoring $t$ | a resumed run differs from the uninterrupted one; bias correction restarts after every resume | `test_state_dict_is_a_snapshot`, `test_resume_is_bitwise` (mutants `s19`, `s22`) |
| decaying parameters whose grad is `None` | embeddings of tokens absent from the batch shrink anyway | `test_parameter_without_grad_is_untouched` (mutant `s07`) |

## 6. Where it's used next
| Forward | `L11.1` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L6.5` | Registered call site uses this module. |

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

If you skip this module, `ol check L0.5` stops with `BLOCKED ... needs M10.3`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `AdamW.step` loop over parameters | `torch.optim.AdamW` `foreach` and `fused` paths | one kernel launch for all parameters at once; the fused CUDA kernel also handles AMP grad scaling | `torch/optim/adam.py`, `_multi_tensor_adam`, `_fused_adam` |
| one global step counter | torch per-parameter `state["step"]` | correct bias correction for parameters that start receiving gradients late | `torch/optim/adam.py` |
| float32 moments | 8-bit optimizers (bitsandbytes) | block-wise quantized $m$ and $v$: optimizer memory drops by 4 times | Dettmers et al., "8-bit Optimizers via Block-wise Quantization" (2022) |
| state on one process | ZeRO stage 1 (DeepSpeed, FSDP) | each data-parallel rank owns a shard of $m$ and $v$; your optional `L11.3` | Rajbhandari et al., "ZeRO" (2020) |
| AdamW for every matrix | Muon | orthogonalized momentum for 2-D weights, AdamW for the rest; your optional `M10.6` | Jordan et al., "Muon" (2024) |
