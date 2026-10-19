# The Optimizer protocol: SGD, momentum, Nesterov, weight decay

## Overview

| | |
|---|---|
| **Module** | `M10.2` · build · Python · Pass 2 · 2 to 3 h |
| **Solution** | [`course/solutions/python/tinyllm/optim/sgd.py`](../../../../course/solutions/python/tinyllm/optim/sgd.py): the `Param` and `Optimizer` protocols, `SGD` (`step`, `zero_grad`, `state_dict`, `load_state_dict`) |
| **Contract** | [`course/contracts/py/tinyllm/optim/sgd.pyi`](../../../../course/contracts/py/tinyllm/optim/sgd.pyi) |
| **Needs** | `M10.1` gradient descent (plain SGD reproduces it bit for bit) |
| **Used by** | `L0.5` trains the bigram · later: `L2.2` the neural n-gram model, `L3.6` the recurrent models; `M10.3` (AdamW) and `M10.4` (schedules, clipping) implement and drive the same protocol · later: `L0.6`, `L11.1` |
| **Optional depth** | Goh, [*Why Momentum Really Works*](https://distill.pub/2017/momentum/) (Distill, 2017); Sutskever, Martens, Dahl, and Hinton, "On the importance of initialization and momentum in deep learning" (ICML 2013) |

## Key Takeaways

- An optimizer is an object: it holds references to the parameters and the state it carries between steps, and every optimizer of the course exposes the same four methods.
- Momentum accumulates past gradients, $v_t = \beta v_{t-1} + g_t$, which on an ill-conditioned problem turns a rate of $1 - 1/\kappa$ into roughly $1 - 2/\sqrt\kappa$.
- Coupled weight decay adds $\lambda\theta$ to the gradient before the momentum buffer; for plain SGD that is the same as shrinking $\theta$ by $1 - \eta\lambda$, a coincidence Adam breaks.
- Updates happen in place, parameters without a gradient are skipped, and `state_dict` makes a stopped run resume bit for bit.

## How to read this chapter

Sections 1 to 3 define the protocol, derive momentum, Nesterov, and coupled weight decay, and run each by hand. Section 4 fixes the interface. Section 5 walks through `SGD` method by method and explains the order of the update and why the state is copied. Try to predict each line of `step` from section 2 before reading it.

---

## 1. Why now

`M10.1`'s `gradient_descent` takes a function and returns a trajectory: fine for a two-dimensional quadratic, wrong for training. In `L0.5` the bigram's weights live inside a model, gradients arrive from the autograd one minibatch at a time, the loop runs for thousands of steps, and a run that stops (a closed laptop, a killed worker in Pass 8) must resume exactly where it was. That needs an object that holds the parameters by reference, updates them in place, keeps per-parameter state such as a momentum buffer, and can save and restore that state. The loop then reads `opt.zero_grad(); loss.backward(); opt.step()` whatever the optimizer is: SGD here, AdamW in `M10.3`, Muon in `M10.6`.

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

These numbers are the first case in section 4's table.

**Changing the learning rate, momentum $\beta = 0.9$:** take the first momentum step above with $\eta = 0.1$ ($v_1 = 1$, $x_1 = 0.9$), then set $\eta = 0.01$. The buffer is $v_2 = 0.9 + 0.9 = 1.8$ as before, and the step is $0.01 \cdot 1.8 = 0.018$, so $x_2 = 0.882$. A buffer that had absorbed $\eta$ would hold $u_1 = 0.1$, then $u_2 = 0.09 + 0.009 = 0.099$, and land on $x_2 = 0.801$: ten times the step the schedule asked for. This is the second case in section 4's table.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked tables | unit | the section 3 tables for momentum, Nesterov, and decay | everyone agrees on the update order |
| a learning-rate change takes effect on the next step | unit | $\eta$ from 0.1 to 0.01 after one momentum step gives $x_2 = 0.882$ and $v_2 = 1.8$ | schedules (`M10.4`) set `opt.lr` before every step |
| PyTorch trajectories | golden | 20-step torch.optim.SGD trajectories, five settings, two parameters | recipes from PyTorch work on this optimizer |
| plain SGD equals gradient descent | differential | bit for bit equal to `M10.1`'s `gradient_descent` | the protocol wraps the same update |
| updates in place | unit | `p.data` is the same array after `step`; float32 stays float32 | the model's references see the update |
| a `None` grad is skipped; `zero_grad` clears | boundary | a `None` grad means no move, no decay, no buffer; `zero_grad` sets `None` | frozen and unused parameters |
| resume bit for bit | property | 10 + save + load + 10 steps equals 20 steps exactly | resumable training (`L0.5`, `C1`) |
| `state_dict` is a copy | unit | saved buffers do not change when training continues; loading copies | checkpoints are snapshots |
| bad hyperparameters | boundary | negative values and Nesterov without momentum | PyTorch's rules |
| momentum beats plain SGD on an ill-conditioned problem | property | $\kappa = 100$: plain leaves a gap above 0.02, tuned heavy ball below $10^{-8}$ | why momentum exists |
| weight decay shrinks toward zero | property | zero gradient: $\theta_t = (1 - \eta\lambda)^t \theta_0$ | decay as shrinkage, before `M10.3` |
| buffers are per parameter | unit | two parameters with opposite gradients keep separate buffers | no velocity leaks between tensors |
| `SGD` is an `Optimizer` | unit | `isinstance` against both runtime-checkable protocols | loops accept any optimizer |

## 5. The solution, step by step

### From first principles

1. **The protocol is structural.** A training loop needs only `step`, `zero_grad`, `state_dict`, and `load_state_dict`, and a parameter only `data` and `grad`. Declaring them as `runtime_checkable` `Protocol`s lets any class with those members qualify, without inheritance, and lets `isinstance` check it.
2. **Validate at construction.** PyTorch rejects negative `lr`, `momentum`, and `weight_decay`, and Nesterov without momentum. Checking in `__init__` turns a typo into an immediate error instead of a silently odd run.
3. **Skip what has no gradient.** A `None` grad means the parameter took no part in the loss (a frozen layer). It must not move, decay, or create a buffer, so `step` skips it before anything else.
4. **Follow PyTorch's order.** Decay is added to the gradient, then the buffer is updated with that decayed gradient, then Nesterov's look-ahead term is formed, then the parameter moves (section 2). Any other order is a different optimizer.
5. **The buffer holds gradients; $\eta$ multiplies at the end.** So a schedule's new `lr` scales the very next step (section 3's 0.882 example).
6. **One buffer per parameter, keyed by position.** Parameters have different shapes and different gradients; a dict from list index to array keeps them apart and matches PyTorch's `state` layout.
7. **Mutate `data` in place.** `p.data -= ...` keeps every other reference to the array (the model, tied weights, the checkpoint writer) current and keeps the dtype.
8. **State out and in are copies.** A checkpoint must not change when training continues, and a loaded state must not alias the caller's dict; with deterministic arithmetic, that makes save and resume bit-exact.

### The code, annotated

```python
@runtime_checkable
class Param(Protocol):
    data: NDArray
    grad: NDArray | None


@runtime_checkable
class Optimizer(Protocol):
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, sd: dict) -> None: ...


class SGD:
    """torch.optim.SGD semantics with dampening 0 (see the contract)."""

    def __init__(self, params, lr, momentum=0.0, nesterov=False, weight_decay=0.0) -> None:
        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if momentum < 0:
            raise ValueError(f"momentum must be >= 0, got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if nesterov and momentum == 0:
            raise ValueError("nesterov needs momentum > 0")
        self.params = list(params)              # references, not copies: step mutates them
        self.lr = float(lr)                     # public: schedules assign opt.lr
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.weight_decay = float(weight_decay)
        self.buffers: dict[int, NDArray] = {}   # created lazily, one per parameter index
```

`list(params)` materializes a generator once, so the same parameters are seen on every step, while keeping references to the caller's objects. Buffers are created on first use rather than as zeros up front, which is how a parameter that never gets a gradient never gets a buffer.

```python
    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue                                     # no move, no decay, no buffer
            g = np.asarray(p.grad, dtype=p.data.dtype)       # compute in the parameter's dtype
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data           # coupled decay: before the buffer
            if self.momentum != 0:
                buf = self.buffers.get(i)
                if buf is None:
                    buf = np.array(g, copy=True)             # v_1 = g_1, a private copy
                else:
                    buf *= self.momentum                     # v_t = beta v_{t-1} + g_t, in place
                    buf += g
                self.buffers[i] = buf
                g = g + self.momentum * buf if self.nesterov else buf   # look-ahead: g + beta v
            p.data -= self.lr * g                            # eta outside the buffer, in place

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None                                    # None, not zeros: see step's skip
```

Trace section 3's momentum table: step 1, no buffer, so `buf = [1]` and $x$ moves by $0.1 \cdot 1$ to $0.9$; step 2, `buf = 0.9 * 1 + 0.9 = 1.8`, $x = 0.9 - 0.18 = 0.72$. With `nesterov=True`, step 1 moves by $0.1 \cdot (1 + 0.9 \cdot 1) = 0.19$ to $0.81$. `g = g + ...` (not `g += ...`) matters in the decay line: `np.asarray` may return `p.grad` itself, and an in-place add would change the caller's gradient. The first buffer is a copy for the same reason: later `buf *= ...` must not modify the gradient array it came from.

```python
    def state_dict(self) -> dict:
        return {
            "state": {i: {"momentum_buffer": b.copy()} for i, b in self.buffers.items()},
            "param_groups": [{
                "lr": self.lr, "momentum": self.momentum,
                "nesterov": self.nesterov, "weight_decay": self.weight_decay,
                "params": list(range(len(self.params))),
            }],
        }

    def load_state_dict(self, sd: dict) -> None:
        (group,) = sd["param_groups"]                        # exactly one group, or it raises
        if len(group["params"]) != len(self.params):
            raise ValueError(...)
        buffers = {}
        for i, st in sd["state"].items():
            i = int(i)                                       # keys may come back as strings
            b = np.array(st["momentum_buffer"], copy=True)   # never alias the caller's arrays
            if not 0 <= i < len(self.params) or b.shape != self.params[i].data.shape:
                raise ValueError(f"momentum buffer {i} does not match parameter shapes")
            buffers[i] = b
        self.lr = float(group["lr"])
        # ... momentum, nesterov, weight_decay likewise ...
        self.buffers = buffers                               # assigned only after every check
```

`load_state_dict` validates everything before assigning anything, so a bad checkpoint leaves the optimizer unchanged. `int(i)` accepts keys that a JSON round trip turned into strings. Restoring `lr` matters: a resumed run continues with the learning rate it had, not the one passed to the constructor.

Full solution: [`sgd.py`](../../../../course/solutions/python/tinyllm/optim/sgd.py).

### Why this solution

- **Cost.** One pass over the parameters per step with $O(1)$ array operations each; memory is one buffer per parameter with momentum, none without.
- **PyTorch's conventions over textbook ones.** The "average" form $v = \beta v + (1 - \beta)g$ and folding $\eta$ into the buffer are both common in texts. The solution follows `torch.optim.SGD` (sum form, $\eta$ outside) so that published recipes transfer unchanged and schedules behave as expected.
- **Coupled decay here, decoupled in AdamW.** SGD's decay goes through the buffer because that is the L2 penalty's gradient and PyTorch's definition; `M10.3` shows why decoupling matters once the update is normalized.
- **A protocol over a base class.** Structural typing lets `SGD`, `AdamW`, and a test double all satisfy `Optimizer` with no shared code, and keeps the loop independent of the optimizer.
- **Tradeoffs accepted.** One parameter group only, no dampening, and a Python loop over tensors instead of fused multi-tensor kernels (Going further). These keep the update readable and bit-checkable against PyTorch's single-tensor path.

### Explain it back

**1. Why does `step` add weight decay before the momentum update and not after?**

<details><summary>Answer</summary>

Coupled decay is the gradient of the L2 penalty $\frac\lambda2\lVert\theta\rVert^2$, so it belongs to the gradient and must flow through the buffer like any other gradient; that is `torch.optim.SGD`'s definition. Adding it after the buffer is decoupled decay (SGDW), a different optimizer whose trajectory differs whenever momentum is non-zero.

</details>

**2. Why is the learning rate applied in `p.data -= self.lr * g` rather than stored in the buffer?**

<details><summary>Answer</summary>

With a constant rate the two are equivalent, but schedules change `lr` before every step. With $\eta$ outside, a new rate scales the next whole step immediately; with $\eta$ inside, the buffer still carries steps scaled by the old rate for about $1/(1 - \beta)$ steps. Section 3: going from 0.1 to 0.01 gives $x_2 = 0.882$ with the solution and $0.801$ with a folded buffer.

</details>

**3. What would break if `step` wrote `p.data = p.data - self.lr * g`?**

<details><summary>Answer</summary>

It would bind `p.data` to a new array. The model, tied weights, and any checkpoint writer that hold the old array would keep the old weights, so the forward pass would not see the update. The in-place `-=` modifies the shared array and keeps its dtype.

</details>

**4. Why does `zero_grad` set `None` instead of zero arrays?**

<details><summary>Answer</summary>

`step` skips parameters whose `grad` is `None`: no update, no decay, no buffer. A zero array would still be decayed (the weights would shrink every step) and would get a momentum buffer that keeps moving the parameter from old velocity. `None` is how a frozen or unused parameter says "I did not take part".

</details>

**5. Why must `state_dict` copy the buffers?**

<details><summary>Answer</summary>

`step` updates the buffers in place (`buf *= self.momentum; buf += g`). A state dict holding the live arrays would change as training continued, so a checkpoint taken at step 10 would contain step 20's momentum. Copying makes the checkpoint a snapshot, and resuming from it reproduces the uninterrupted run bit for bit.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. adding weight decay after the momentum buffer | a different optimizer (SGDW); matches PyTorch only without momentum | `g = g + weight_decay * p.data` comes before the buffer; step 4 |
| 2. one momentum buffer for every parameter | velocity leaks between tensors, or shapes fail to broadcast | `self.buffers` is keyed by parameter index; step 6 |
| 3. `p.data = p.data - lr * g` | the model keeps training on stale weights it still references | `p.data -= self.lr * g`; step 7 |
| 4. a `state_dict` of live buffers, or a load that drops them | a checkpoint that changes after saving; a resumed run that diverges from the original | `b.copy()` out, `np.array(..., copy=True)` in, and `lr` and buffers restored; step 8 |
| 5. momentum as an average, $v = \beta v + (1 - \beta) g$ | steps $1 - \beta$ times smaller than PyTorch's | `buf *= momentum; buf += g`, the sum form |
| 6. Nesterov ignored or with its terms swapped | plain momentum where look-ahead was asked for | `g + self.momentum * buf` when `nesterov` |
| 7. zeros instead of `None`, or decaying parameters without a gradient | frozen layers shrink every step | `zero_grad` sets `None` and `step` skips it first; step 3 |
| 8. the sign of the decay term | weights pushed away from zero | decay is added to the gradient, which is then subtracted |
| 9. folding $\eta$ into the momentum buffer | identical while $\eta$ is constant; under a schedule the old rate lingers for about $1/(1-\beta)$ steps | `lr` multiplies only at the final update; step 5 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M10.1` | the gradient descent step this generalizes; plain SGD reproduces it bit for bit |
| Forward | `L0.5` | the bigram's training loop: `zero_grad`, backward, `step`, and `state_dict` in the checkpoint |
| Forward | `L2.2` | the neural n-gram model trains with SGD and momentum |
| Forward | `L3.6` | the recurrent language models train through the same protocol |
| Forward | `M10.3` | AdamW implements the same four methods, with decoupled decay |
| Forward | `M10.4` | schedules set `lr` between steps; clipping rescales `grad` before `step` |
| Forward | `L0.6` | trains through the same `Optimizer` protocol |
| Forward | `L11.1` | trains through the same `Optimizer` protocol |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SGD.step` | PyTorch `torch.optim.SGD` | parameter groups with their own hyperparameters, dampening, `foreach` and fused multi-tensor kernels | `torch/optim/sgd.py` (`_single_tensor_sgd`) |
| the `Optimizer` protocol | optax | optimizers as composable pure gradient transformations (`trace` for momentum, `add_decayed_weights`, `scale`) | `optax/_src/alias.py` (`sgd`) |
| `state_dict` | PyTorch distributed checkpointing | sharded optimizer state saved and resharded across ranks | `torch/distributed/checkpoint/` |
