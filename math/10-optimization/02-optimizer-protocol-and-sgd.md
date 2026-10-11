<!-- ol:module M10.2 -->
# The Optimizer protocol: SGD, momentum, Nesterov, weight decay

## Overview

| | |
|---|---|
| **Module** | `M10.2` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/optim/sgd.py`: the `Param` and `Optimizer` protocols, `SGD` (`step`, `zero_grad`, `state_dict`, `load_state_dict`) |
| **Contract** | [`course/contracts/py/tinyllm/optim/sgd.pyi`](../../course/contracts/py/tinyllm/optim/sgd.pyi) |
| **Tests** | `course/tests/M10.2/` (what they check: section 4) |
| **Needs** | `M10.1` gradient descent (the tests compare, or `--ref-deps`) |
| **Used by** | `L0.5` trains the bigram · later: `L2.2` the neural n-gram model, `L3.6` the recurrent models; `M10.3` (AdamW) and `M10.4` (schedules, clipping) implement and drive the same protocol · later: `L0.6`, `L11.1` |
| **Milestone** | `MS-P2` (Pass 2 gate: every math module of the pass checks green, then your autograd bigram trains) |
| **Optional depth** | Goh, [*Why Momentum Really Works*](https://distill.pub/2017/momentum/) (Distill, 2017); Sutskever, Martens, Dahl, and Hinton, "On the importance of initialization and momentum in deep learning" (ICML 2013) |

## Key Takeaways

- An optimizer is an object: it holds references to the parameters and the state it carries between steps, and every optimizer of the course exposes the same four methods (`test_sgd_is_an_optimizer`).
- Momentum accumulates past gradients, $v_t = \beta v_{t-1} + g_t$, which on an ill-conditioned problem turns a rate of $1 - 1/\kappa$ into roughly $1 - 2/\sqrt\kappa$ (`test_hand_example`, `test_momentum_beats_plain_on_ill_conditioned`).
- Coupled weight decay adds $\lambda\theta$ to the gradient before the momentum buffer; for plain SGD that is the same as shrinking $\theta$ by $1 - \eta\lambda$, a coincidence Adam breaks (`test_weight_decay_shrinks_toward_zero`, `test_matches_torch_golden`).
- Updates happen in place, parameters without a gradient are skipped, and `state_dict` makes a stopped run resume bit for bit (`test_updates_in_place`, `test_state_dict_resume_bitwise`).

## How to work this chapter

```bash
ol start M10.2              # stubs sgd.py into your repo, contract alongside
ol tests M10.2              # read the test catalog first: rung R0, you write no tests here
ol check M10.2              # exit code is the verdict
ol check M10.2 --ref-deps   # only if your M10.1 is not passing yet
ol diff  M10.2              # after passing: your code against the reference
```

---

## 1. Why now

`M10.1`'s `gradient_descent` takes a function and returns a trajectory: fine for a two-dimensional quadratic, wrong for training. In `L0.5` your bigram's weights live inside a model, gradients arrive from your autograd one minibatch at a time, the loop runs for thousands of steps, and a run that stops (a closed laptop, a killed worker in Pass 8) must resume exactly where it was. That needs an object that holds the parameters by reference, updates them in place, keeps per-parameter state such as a momentum buffer, and can save and restore that state. The loop then reads `opt.zero_grad(); loss.backward(); opt.step()` whatever the optimizer is: SGD here, AdamW in `M10.3`, Muon in `M10.6`.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\theta$ | one parameter tensor (`p.data`) | float array |
| $g_t$ | its gradient at step $t$ (`p.grad`), or None | same shape |
| $\eta$ | learning rate (`lr`) | float $\ge 0$ |
| $\beta$ | momentum coefficient (`momentum`) | float in $[0, 1)$ |
| $v_t$ | momentum buffer of one parameter | same shape as $\theta$ |
| $\lambda$ | weight decay coefficient (`weight_decay`) | float $\ge 0$ |
| $L$, $\mu$, $\kappa = L/\mu$ | smoothness, strong convexity, condition number (`M10.1`) | floats |

**Stochastic gradients.** A training loss is an average over examples, and its gradient is an average of per-example gradients. A minibatch gives an unbiased estimate of it, so SGD is gradient descent with a noisy gradient. Everything below works per parameter tensor, with whatever gradient the backward pass produced.

**The protocol.** A parameter is anything with a float array `data` and a `grad` that is an array of the same shape or `None`. An optimizer has four methods:

| Method | Does |
|---|---|
| `step()` | update every parameter's `data` in place from its `grad` |
| `zero_grad()` | set every `grad` to `None` |
| `state_dict()` | return a deep copy of everything needed to resume |
| `load_state_dict(sd)` | restore it, copying arrays in |

`zero_grad` exists because backward accumulates (`M08.2`): without it the second step would use the sum of two steps' gradients. Setting `None` instead of zeros (PyTorch's default) lets `step` skip parameters that received no gradient at all, such as a frozen layer: no update, no decay, no buffer change.

**Momentum (the heavy ball).** Keep a buffer per parameter:

$$v_t = \beta v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t,$$

with $v_1 = g_1$ on the first step (PyTorch's convention: no dampening, and the learning rate multiplies the buffer). Unrolled, $v_t = \sum_{k \ge 0} \beta^k g_{t-k}$: an exponentially weighted sum of past gradients. When gradients agree from step to step (a long shallow valley) they add up, to $\eta/(1 - \beta)$ times one gradient in the limit; when they flip sign (bouncing across a steep valley) they cancel. On a quadratic with condition number $\kappa$, choosing $\eta = 4/(\sqrt L + \sqrt\mu)^2$ and $\beta = \left(\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1}\right)^2$ gives a per-step rate of $\frac{\sqrt\kappa - 1}{\sqrt\kappa + 1} \approx 1 - 2/\sqrt\kappa$, against $1 - 1/\kappa$ for plain gradient descent: for $\kappa = 100$, about 0.82 instead of 0.99.

**The buffer holds gradients, not steps.** $\eta$ multiplies $v_t$ when the step is taken; it never enters the buffer. With a constant $\eta$ that is only bookkeeping: a buffer $u_t = \beta u_{t-1} + \eta g_t$ with update $\theta_{t+1} = \theta_t - u_t$ produces the same trajectory. It stops being the same the moment $\eta$ changes, and the schedules of `M10.4` change it before every step. With $\eta$ outside, a new learning rate scales the whole next step at once; with $\eta$ inside, the old rate lingers in the buffer for about $1/(1 - \beta)$ steps.

**Nesterov momentum.** Nesterov's method evaluates the gradient at a look-ahead point. Rewritten in the variables PyTorch stores, it becomes one extra term:

$$v_t = \beta v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\,(g_t + \beta v_t).$$

It needs momentum to mean anything, so `nesterov=True` with $\beta = 0$ is an error.

**Weight decay, coupled.** The L2 penalty $\tfrac\lambda2 \lVert\theta\rVert^2$ adds $\lambda\theta$ to the gradient. SGD applies it first, so it flows through the momentum buffer like any other gradient:

$$g_t \leftarrow g_t + \lambda\theta_t \quad \text{(before momentum)}.$$

With plain SGD this is the same as shrinking the weights, $\theta \leftarrow (1 - \eta\lambda)\theta - \eta g$; with a zero gradient, $\theta_t = (1 - \eta\lambda)^t \theta_0$. With momentum the decay accumulates in the buffer, so adding $\lambda\theta$ after the buffer instead ("decoupled" decay, SGDW) is a different optimizer with a different trajectory. AdamW (`M10.3`) is built on exactly that difference.

**In place, in the parameter's dtype.** The model, its tied weights, and the checkpoint writer hold references to each `data` array. `p.data -= lr * g` changes that array; `p.data = p.data - lr * g` builds a new one and leaves every other reference pointing at the old weights. In-place arithmetic also keeps a float32 parameter float32.

**State, and resuming bit for bit.** SGD's state is one momentum buffer per parameter, keyed by the parameter's position in the list. `state_dict` copies the buffers and the hyperparameters (PyTorch's layout: `{"state": {i: {"momentum_buffer": ...}}, "param_groups": [{...}]}`); a copy, because a checkpoint must not change when training continues. Since every operation is deterministic, 10 steps, save, load into a fresh optimizer, and 10 more steps give exactly the bits of 20 uninterrupted steps.

## 3. Worked example by hand

Minimize $f(x) = x^2/2$, whose gradient is $x$, from $x_0 = 1$ with $\eta = 0.1$.

**Momentum $\beta = 0.9$:**

| $t$ | $g_t = x_{t-1}$ | $v_t = 0.9\,v_{t-1} + g_t$ | $x_t = x_{t-1} - 0.1\,v_t$ |
|---|---|---|---|
| 1 | 1 | 1 | 0.9 |
| 2 | 0.9 | $0.9 + 0.9 = 1.8$ | $0.9 - 0.18 = 0.72$ |
| 3 | 0.72 | $1.62 + 0.72 = 2.34$ | $0.72 - 0.234 = 0.486$ |

Plain gradient descent would give $0.9, 0.81, 0.729$: the buffer has already doubled the effective step.

**Nesterov:**

| $t$ | $g_t$ | $v_t$ | step direction $g_t + 0.9\,v_t$ | $x_t$ |
|---|---|---|---|---|
| 1 | 1 | 1 | 1.9 | 0.81 |
| 2 | 0.81 | $0.9 + 0.81 = 1.71$ | $0.81 + 1.539 = 2.349$ | $0.81 - 0.2349 = 0.5751$ |

**Weight decay $\lambda = 0.1$, no momentum:** the gradient becomes $x + 0.1x = 1.1$, so $x_1 = 1 - 0.11 = 0.89$, which is $(1 - \eta\lambda)\cdot 1 - \eta \cdot 1 = 0.99 - 0.1$.

These numbers are the first test case in section 4, `test_hand_example`.

**Changing the learning rate, momentum $\beta = 0.9$:** take the first momentum step above with $\eta = 0.1$ ($v_1 = 1$, $x_1 = 0.9$), then set $\eta = 0.01$. The buffer is $v_2 = 0.9 + 0.9 = 1.8$ as before, and the step is $0.01 \cdot 1.8 = 0.018$, so $x_2 = 0.882$. A buffer that had absorbed $\eta$ would hold $u_1 = 0.1$, then $u_2 = 0.09 + 0.009 = 0.099$, and land on $x_2 = 0.801$: ten times the step the schedule asked for. This is `test_lr_change_takes_effect_on_the_next_step`.

## 4. The interface

```python
# python/tinyllm/optim/sgd.py
@runtime_checkable
class Param(Protocol):     data: NDArray; grad: NDArray | None
@runtime_checkable
class Optimizer(Protocol):
    def step(self) -> None; def zero_grad(self) -> None
    def state_dict(self) -> dict; def load_state_dict(self, sd: dict) -> None
class SGD:
    def __init__(self, params, lr: float, momentum: float = 0.0, nesterov: bool = False, weight_decay: float = 0.0)
```

`SGD` follows `torch.optim.SGD` with dampening 0, so the contract's update order is PyTorch's: weight decay, then the buffer, then Nesterov, then the in-place update. `ValueError` for a negative `lr`, `momentum`, or `weight_decay`, and for Nesterov without momentum. `load_state_dict` refuses a state for a different number or shape of parameters.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the section 3 tables for momentum, Nesterov, and decay | you and the test agree on the update order |
| `test_lr_change_takes_effect_on_the_next_step` | unit | $\eta$ from 0.1 to 0.01 after one momentum step gives $x_2 = 0.882$ and $v_2 = 1.8$ | schedules (`M10.4`) set `opt.lr` before every step |
| `test_matches_torch_golden` | golden | 20-step torch.optim.SGD trajectories, five settings, two parameters | recipes from PyTorch work on your optimizer |
| `test_plain_sgd_equals_gradient_descent` | differential | bit for bit equal to `M10.1`'s `gradient_descent` | the protocol wraps the same update |
| `test_updates_in_place` | unit | `p.data` is the same array after `step`; float32 stays float32 | the model's references see the update |
| `test_none_grad_is_skipped_and_zero_grad_clears` | boundary | a `None` grad means no move, no decay, no buffer; `zero_grad` sets `None` | frozen and unused parameters |
| `test_state_dict_resume_bitwise` | property | 10 + save + load + 10 steps equals 20 steps exactly | resumable training (`L0.5`, `C1`) |
| `test_state_dict_is_a_copy` | unit | saved buffers do not change when training continues; loading copies | checkpoints are snapshots |
| `test_rejects_bad_hyperparameters` | boundary | negative values and Nesterov without momentum | PyTorch's rules |
| `test_momentum_beats_plain_on_ill_conditioned` | property | $\kappa = 100$: plain leaves a gap above 0.02, tuned heavy ball below $10^{-8}$ | why momentum exists |
| `test_weight_decay_shrinks_toward_zero` | property | zero gradient: $\theta_t = (1 - \eta\lambda)^t \theta_0$ | decay as shrinkage, before `M10.3` |
| `test_buffers_are_per_parameter` | unit | two parameters with opposite gradients keep separate buffers | no velocity leaks between tensors |
| `test_sgd_is_an_optimizer` | unit | `isinstance` against both runtime-checkable protocols | loops accept any optimizer |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. adding weight decay after the momentum buffer | a different optimizer (SGDW); matches PyTorch only without momentum | `test_matches_torch_golden` (mutant `s04`) |
| 2. one momentum buffer for every parameter | velocity leaks between tensors, or shapes fail to broadcast | `test_buffers_are_per_parameter` (mutant `s10`) |
| 3. `p.data = p.data - lr * g` | the model keeps training on stale weights it still references | `test_updates_in_place` (mutant `s05`) |
| 4. a `state_dict` of live buffers, or a load that drops them | a checkpoint that changes after saving; a resumed run that diverges from the original | `test_state_dict_is_a_copy` (mutant `s08`), `test_state_dict_resume_bitwise` (mutant `s09`) |
| 5. momentum as an average, $v = \beta v + (1 - \beta) g$ | steps $1 - \beta$ times smaller than PyTorch's | `test_hand_example` (mutant `s01`) |
| 6. Nesterov ignored or with its terms swapped | plain momentum where look-ahead was asked for | `test_hand_example` (mutants `s02`, `s03`) |
| 7. zeros instead of `None`, or decaying parameters without a gradient | frozen layers shrink every step | `test_none_grad_is_skipped_and_zero_grad_clears` (mutants `s06`, `s07`) |
| 8. the sign of the decay term | weights pushed away from zero | `test_weight_decay_shrinks_toward_zero` (mutant `s12`) |
| 9. folding $\eta$ into the momentum buffer | identical while $\eta$ is constant; under a schedule the old rate lingers for about $1/(1-\beta)$ steps | `test_lr_change_takes_effect_on_the_next_step` (mutant `s11`) |

## 6. Where it's used next
| Forward | `L0.6` | Registered call site uses this module. |
| Forward | `L11.1` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M10.1` | the gradient descent step this generalizes; plain SGD reproduces it bit for bit |
| Forward | `L0.5` | the bigram's training loop: `zero_grad`, backward, `step`, and `state_dict` in the checkpoint |
| Forward | `L2.2` | the neural n-gram model trains with SGD and momentum |
| Forward | `L3.6` | the recurrent language models train through the same protocol |
| Forward | `M10.3` | AdamW implements the same four methods, with decoupled decay |
| Forward | `M10.4` | schedules set `lr` between steps; clipping rescales `grad` before `step` |

If you skip this module, `ol check L0.5` stops with `L0.5 needs M10.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SGD.step` | PyTorch `torch.optim.SGD` | parameter groups with their own hyperparameters, dampening, `foreach` and fused multi-tensor kernels | `torch/optim/sgd.py` (`_single_tensor_sgd`) |
| the `Optimizer` protocol | optax | optimizers as composable pure gradient transformations (`trace` for momentum, `add_decayed_weights`, `scale`) | `optax/_src/alias.py` (`sgd`) |
| `state_dict` | PyTorch distributed checkpointing | sharded optimizer state saved and resharded across ranks | `torch/distributed/checkpoint/` |
