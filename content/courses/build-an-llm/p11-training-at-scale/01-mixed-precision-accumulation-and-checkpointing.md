# Mixed precision, loss scaling, gradient accumulation, and activation checkpointing

## Overview

| | |
|---|---|
| **Module** | `L11.1` · build · Python · Pass 9 · 5 to 6 h |
| **Solution** | [`precision.py`](../../../../course/solutions/python/tinyllm/train/precision.py): `cast`, `autocast`, `autocast_bf16`, `autocast_dtype`, `DynamicLossScaler`, `grad_accumulate`, `train_step_mixed` · [`recompute.py`](../../../../course/solutions/python/tinyllm/train/recompute.py): `checkpoint`, `checkpoint_sequential` |
| **Contract** | [`course/contracts/py/tinyllm/train/precision.pyi`](../../../../course/contracts/py/tinyllm/train/precision.pyi) · [`course/contracts/py/tinyllm/train/recompute.pyi`](../../../../course/contracts/py/tinyllm/train/recompute.pyi) |
| **Needs** | `L0.1` Tensor, `from_op`, grad mode · `L0.3` `mse` (the examples' loss) · `L0.4` Module and layers · `M09.1` `round_to_bf16`, `round_to_fp16` · `M10.2` SGD and `M10.3` AdamW (the examples' optimizers) · `M10.4` `clip_grad_norm_` · `M08.4` `checkpoint_schedule` · reading: `M05.1` the memory plan, `L0.5` `train_step` |
| **Used by** | the capstone trainer: `C1` trains with `--bf16 --micro-batch 8 --accum 4 --checkpoint-activations`; with all three on, a Llama trains to within 2% of the float32 loss at a lower peak memory |
| **Optional depth** | Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740) (2018); Kalamkar et al., [*A Study of BFLOAT16 for Deep Learning Training*](https://arxiv.org/abs/1905.12322) (2019); Chen et al., [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174) (2016); Korthikanti et al., [*Reducing Activation Recomputation in Large Transformer Models*](https://arxiv.org/abs/2205.05198) (2022) |

## Key Takeaways

- Mixed precision rounds what a matmul reads and writes, and nothing else: the weights stay float32 masters, so an update smaller than bf16's spacing still lands.
- fp16 gradients below $2^{-24}$ flush to zero. Multiplying the loss by $S$ multiplies every gradient by $S$ (backprop is linear), which lifts them into range; dividing by $S$ before the update undoes it, and a step whose gradients overflowed is skipped and $S$ halved.
- Weighting micro-batch $i$'s mean loss by $n_i/N$ makes $k$ micro-batches give exactly the gradient of one batch of $N$ rows.
- A checkpointed segment reruns the same operations on the same values during backward, so its gradients are equal bit for bit, provided the rerun replays the random draws and its parameters are parents of the node.

## How to read this chapter

Sections 1 to 3 build the four tools from first principles and work each one by hand. Section 4 fixes the interface. Section 5 walks through the solution, `precision.py` and `recompute.py`, and explains why each choice is the right one. Try to predict each step of the solution from the principles before reading it.

---

## 1. Why now

The capstone (`C1`) trains a 10.4M-parameter Llama with a 512-token context. The `L0.5` loop runs one batch per step, all in float32, and keeps every layer's activations until `backward()` returns. Three things break at that size. The batch the recipe calls for (32 sequences) does not fit: `M05.1`'s `memory_plan` puts the float32 activations of one 512-token sequence through the 8 layers at 161 MiB, so a batch of 32 needs 5.0 GiB of activations next to 159 MiB of weights, gradients, and AdamW moments. Real hardware trains in bf16 or fp16 for speed and memory, and you need to know that the run you design here still converges when every matmul reads 8-bit mantissas, before you trust a recipe that depends on it. And when a gradient underflows or overflows in fp16, a float32 loop never sees it. This module adds the four tools every large training run uses, each behind one call: `autocast` (emulated low precision), `DynamicLossScaler` (fp16 range), `grad_accumulate` (a big batch from small ones), and `checkpoint_sequential` (activations kept on `M08.4`'s schedule). The emulation runs on numpy and is not faster; what it gives you is the exact numerics, so the claims that the capstone converges within 2% of float32 at a lower peak memory are measured, not assumed.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\mathrm{rd}_{16}(x)$ | $x$ rounded to nearest, ties to even, into bf16 or fp16 (`M09.1`) | elementwise |
| $p$ | the precision of a format, in significant bits: 24 (fp32), 11 (fp16), 8 (bf16) | `int` |
| $u = 2^{-p}$ | the unit roundoff: the largest relative error of one rounding | float |
| $w$ | a parameter (the float32 master copy) | `float32[...]` |
| $\eta$ | the learning rate | float |
| $L$, $g = \partial L/\partial w$ | the loss and a gradient | float, like $w$ |
| $S$ | the loss scale (`loss_scale`) | float, a power of 2 |
| $N$, $n_i$, $k$ | rows in the full batch, rows in micro-batch $i$, number of micro-batches | `int` |
| $L_i$ | the mean loss over micro-batch $i$'s rows | float |
| $s_j$, $P$ | segment sizes and peak memory of a checkpoint schedule (`M08.4`) | `int` |

**Two 16-bit formats.** Both keep a sign, an exponent, and a mantissa in 16 bits. bf16 keeps float32's 8 exponent bits and 7 mantissa bits: the same range (up to about $3.4 \times 10^{38}$) with $p = 8$, so $u = 2^{-8} \approx 0.4\%$. fp16 has 5 exponent bits and 10 mantissa bits: $p = 11$, $u \approx 0.05\%$, but its largest finite value is 65504, its smallest normal $2^{-14} \approx 6.1 \times 10^{-5}$, and its smallest subnormal $2^{-24} \approx 6 \times 10^{-8}$. bf16 trades precision for range; fp16 trades range for precision. numpy has neither, so `M09.1`'s `round_to_bf16` and `round_to_fp16` emulate them: a float array whose entries are all values of the format.

**What a bf16 matmul does.** A tensor core reads two bf16 matrices, multiplies, accumulates the sums in float32, and writes a bf16 result. So $Y = AB$ becomes

$$Y = \mathrm{rd}_{16}\big(\mathrm{rd}_{16}(A)\, \mathrm{rd}_{16}(B)\big), \qquad \text{the product computed in float32}.$$

Each operand entry has relative error up to $u$, and the output one more rounding. Errors of different entries are independent-looking, so a dot product of length $K$ drifts by about $\sqrt K\, u$ relative, not $K u$; for training this is noise of a few tenths of a percent, smaller than minibatch noise.

**The cast is an op with a gradient.** Write the emulated storage as an op `cast`: forward $y = \mathrm{rd}_{16}(x)$. Its true derivative is zero almost everywhere (a step function), which would stop all learning; what hardware does is pass the gradient through the conversion, and a gradient that arrives in bf16 is a bf16 value. So the backward of a cast is the cast of the gradient: $\bar x = \mathrm{rd}_{16}(\bar y)$. Built from `from_op` (`L0.1`), the autocast matmul is `cast(cast(a) @ cast(b))`, and its backward computes $\bar A = \mathrm{rd}_{16}(\mathrm{rd}_{16}(\bar Y)\, \mathrm{rd}_{16}(B)^\top)$: a bf16 backward matmul, exactly what a GPU runs. Because every matmul in your models goes through `Tensor.__matmul__` (`F.matmul`, `Linear`, attention), `autocast` swaps in the rounding version for the length of a `with` block and puts the original back on the way out, also when the block raises. That is what `torch.autocast` does with its dispatch keys.

**Why the weights stay float32.** The update $w \leftarrow w - \eta g$ is where precision matters most. The spacing of bf16 numbers near 1 is $2^{-7} \approx 0.0078$: a weight of 1 updated by $\eta g = 10^{-3}$ rounds straight back to 1, and the update is lost. Training in pure bf16 stalls once updates become small. So the parameters, the optimizer state (AdamW's moments), and the update stay float32: the "master weights". Only the matmuls, which dominate the time and the activation memory, run on rounded operands. `M05.1`'s memory plan counts this: bf16 AdamW is 2 (weights) + 2 (grads) + 4 (master) + 8 (moments) = 16 bytes per parameter, the same as float32, and the saving is all in activations.

**Loss scaling.** In fp16 the problem is range, not precision. Gradients of a mean loss over many tokens are small: a cross-entropy gradient on the logits is $(\mathrm{softmax} - \mathrm{onehot})/N$, and backpropagated through a few layers many entries fall below $2^{-24}$ and flush to zero. Backprop is linear in the upstream gradient: every VJP is linear in $\bar y$, so starting backward from $S \cdot L$ instead of $L$ multiplies every gradient by exactly $S$. Choose $S$ so the small gradients land in fp16's normal range, and divide the float32 master gradients by $S$ before the update. With $S$ a power of two, scaling and unscaling are exact (they only change the exponent).

**Dynamic scaling.** Too large an $S$ overflows the large gradients to $\infty$ (and $\infty - \infty$ to NaN). The fix is to detect, not to predict: after backward, if any gradient is non-finite, skip the whole step (one poisoned entry would corrupt every weight it touches through the optimizer), drop the gradients, and multiply $S$ by the backoff $\tfrac12$. After `interval` consecutive good steps, multiply $S$ by the growth factor 2 to probe upward again. The count of good steps restarts after every overflow and every growth. A skipped step costs one batch; a run without scaling silently loses its small gradients. Clipping (`M10.4`) comes after unscaling, because the clip bound is a norm in true gradient units. bf16 has float32's range and needs no scaler.

**Gradient accumulation.** The loss of a batch of $N$ rows is the mean of per-row losses $\ell_r$. Split the rows into micro-batches $B_1, \dots, B_k$ with $n_i$ rows each, and let $L_i$ be micro-batch $i$'s own mean. Then

$$L = \frac1N \sum_{r} \ell_r = \sum_{i=1}^k \frac{n_i}{N} \cdot \frac{1}{n_i}\sum_{r \in B_i} \ell_r = \sum_{i=1}^k \frac{n_i}{N} L_i,$$

and since the gradient is linear, $\nabla L = \sum_i \tfrac{n_i}{N} \nabla L_i$. Backpropagating $\tfrac{n_i}{N} L_i$ for each micro-batch, without zeroing in between (the Tensor adds into `.grad`, `L0.1`), accumulates exactly $\nabla L$. Only one micro-batch's activations exist at a time. Weighting by $\tfrac1k$ is the same only when all micro-batches have equal size; the last one of an epoch usually does not. The results agree with one big batch to float32 rounding, not bit for bit, because the sums are added in a different order.

**Activation checkpointing.** `M08.4` counted the trade: keep only some layers' inputs and recompute the rest during backward. The mechanism is one graph node. `checkpoint(fn, x)` runs `fn` under `no_grad`, so none of its intermediate values are recorded, and returns a node (`from_op`) whose parents are $x$ and the parameters `fn` reads. When backward reaches the node, its VJP reruns `fn` with grad mode on, from fresh leaf copies of its inputs, backpropagates the upstream gradient through that rerun (which adds the parameter gradients into their `.grad` directly), and returns the inputs' gradients. `checkpoint_sequential` applies `M08.4`'s `checkpoint_schedule`: every segment but the last through `checkpoint`, the last one normally.

Three details make the gradients equal bit for bit. The rerun performs the same floating-point operations on the same values, so it produces the same activations. A dropout layer draws a new mask from its generator on every call, so the node records the generator's state before the forward pass, sets it back for the rerun, and afterwards restores the state the stream had reached, as if nothing had been rerun. And the segment's parameters must be parents of the node: the first segment's input is data, which does not require grad, and `from_op` records a node only when some parent requires grad, so without the parameters the first layers would silently get no gradient (the classic bug of reentrant checkpointing in PyTorch).

One difference from `M08.4`'s model: the `L0.1` `backward()` keeps the whole outer graph alive until it returns, where PyTorch frees each saved tensor as soon as backward has used it. So the activation memory held between forward and backward falls as `M08.4` predicts, while the peak during backward falls less, because the last segment stays alive while earlier segments are recomputed.

## 3. Worked example by hand

**A bf16 matmul.** $A = [1, 1]$, $B = [1, 2^{-8}]^\top$. Both are bf16 values already. The float32 product is $1 + 2^{-8} = 1.00390625$. Its bf16 neighbours are 1 and $1 + 2^{-7}$, with $1 + 2^{-8}$ exactly halfway; ties go to the even mantissa, so autocast returns $1.0$. An input of $1.01$ is stored as $1.0078125 = 1 + 2^{-7}$, the nearest bf16 value. For $L = \sum Y$, $\bar Y = 1$ and $\bar A = \mathrm{rd}(\bar Y)\,\mathrm{rd}(B)^\top = [1, 2^{-8}]$, $\bar B = \mathrm{rd}(A)^\top \mathrm{rd}(\bar Y) = [1, 1]^\top$. The same product outside autocast is the float32 $1.00390625$.

**Accumulation.** One weight $w = 1$, inputs $x = [1, 2, 3, 4]$, targets $2x$, loss the mean of $(wx - 2x)^2$. With $w = 1$ the residuals are $-x$:

| | rows | $L_i$ | $\partial L_i/\partial w = \mathrm{mean}(2(wx - 2x)x)$ | weight |
|---|---|---|---|---|
| one batch | 1, 2, 3, 4 | $(1 + 4 + 9 + 16)/4 = 7.5$ | $2(-1 - 4 - 9 - 16)/4 = -15$ | 1 |
| micro-batch 1 | 1 | 1 | $-2$ | $1/4$ |
| micro-batch 2 | 2, 3, 4 | $29/3$ | $2(-4 - 9 - 16)/3 = -58/3$ | $3/4$ |

Weighted: $\tfrac14(-2) + \tfrac34(-\tfrac{58}{3}) = -0.5 - 14.5 = -15$, and the loss $\tfrac14 \cdot 1 + \tfrac34 \cdot \tfrac{29}{3} = 7.5$. Weighting each by $\tfrac12$ gives $-10.67$; not weighting at all gives $-21.3$.

**Loss scaling.** $S = 8$, growth 2, backoff $\tfrac12$, interval 2, SGD with $\eta = 0.5$ on one parameter starting at 0. The scaled gradients arrive as below:

| step | scaled $g$ | finite? | unscaled $g/S$ | $w$ after | $S$ after | good steps |
|---|---|---|---|---|---|---|
| 1 | 8 | yes | 1 | $-0.5$ | 8 | 1 |
| 2 | $\infty$ | no: skip | | $-0.5$ | 4 | 0 |
| 3 | 4 | yes | 1 | $-1.0$ | 4 | 1 |
| 4 | 8 | yes | 2 | $-2.0$ | 8 (grown) | 0 |
| 5 | 16 | yes | 2 | $-3.0$ | 8 | 1 |

If the count did not restart at the overflow, step 3 would already be the second good step and grow $S$ one step early.

**Checkpointing six layers.** With a budget of 4 units, `checkpoint_schedule(6, 4) == [0, 3]` (`M08.4`, section 3). The forward pass runs layers 0 to 2 under `no_grad` and keeps only their input, then layers 3 to 5 normally. Backward runs layers 5, 4, 3, reaches the checkpoint node, reruns layers 0 to 2 forward, and backpropagates through them. Each of layers 0 to 2 has run forward twice, layers 3 to 5 once, and every gradient equals the plain run's. These four examples are the first four cases in section 4's table.

## 4. The interface

```python
# python/tinyllm/train/precision.py
def cast(x: Tensor, dtype: Literal["bf16", "fp16"]) -> Tensor        # rounds forward and backward
@contextmanager
def autocast(dtype) -> Iterator[None]                                 # every Tensor matmul: cast(cast(a) @ cast(b))
def autocast_bf16(); def autocast_dtype() -> Optional[str]
class DynamicLossScaler:
    def __init__(self, init=2.0**16, growth=2.0, backoff=0.5, interval=2000)
    def scale(self, loss: Tensor) -> Tensor                           # loss * S
    def step(self, opt, params, clip=None) -> bool                    # check, unscale, clip, step, update S
    def state_dict(self) -> dict; def load_state_dict(self, sd) -> None
def grad_accumulate(model, micro_batches, loss_fn, scaler=None) -> float          # weights n_i / N
def train_step_mixed(model, micro_batches, loss_fn, opt, precision="fp32", scaler=None, clip=None) -> dict

# python/tinyllm/train/recompute.py
def checkpoint(fn, *args, params=(), rngs=()) -> Tensor
def checkpoint_sequential(layers, x, mem_budget_layers, rngs=()) -> Tensor        # M08.4's schedule
```

`train_step_mixed` is `L0.5`'s `train_step` for the capstone: zero the gradients, accumulate the micro-batches under `autocast(precision)` (none for `"fp32"`), then either the scaler's step or a finite-loss check, clipping, and `opt.step()`. It returns `{"loss", "skipped", "scale"}` plus `"grad_norm"` when it clipped. `autocast` patches `Tensor.__matmul__` and `__rmatmul__` process-wide while the outermost block is open; blocks nest and the innermost dtype applies.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: bf16 matmul | unit | section 3: $1 + 2^{-8} \to 1$, $1.01 \to 1.0078125$, bf16 gradients | the emulation is pinned to exact values |
| hand example: accumulation | unit | section 3: loss 7.5, gradient $-15$ from micro-batches of 1 and 3 | the capstone's `--accum 4` |
| hand example: loss scaler | unit | section 3's five-step trace of $S$, the count, and $w$ | fp16 runs |
| hand example: checkpoint schedule | unit | section 3: forward counts $[2, 2, 2, 1, 1, 1]$, equal gradients | `--checkpoint-activations` |
| cast rounds both directions | unit | both formats, forward and gradient, dtype kept | the op every autocast matmul is built from |
| autocast matmul matches a rounded reference | property | random products, reflected operands, `Linear` | attention and MLP matmuls |
| gradients are bf16, masters stay float32 | property | weight gradients are bf16 values; a $10^{-3}$ update survives in the master | why mixed precision converges |
| autocast restores the matmul | boundary | an exception, nesting, an unknown dtype | no bf16 leaking into the next eval |
| fp16 underflow rescued by loss scaling | differential | tiny gradients vanish in fp16 and match float32 once scaled | fp16 hardware |
| scaler skips inf steps | property | an overflow changes nothing, drops gradients, halves $S$ until steps succeed | a long run survives a spike |
| scaler unscales before clipping | differential | clip to 1 of a gradient of norm 5 scaled by 1024 | the clip bound means what it says |
| scaler state round trip | unit | resume keeps $S$ and the count; bad arguments raise | `TrainRun` resumes from checkpoints |
| accumulation equals one big batch | differential | micro-batches of 5, 11, 16 rows equal one of 32 to $10^{-6}$ | the capstone's loss within 2% |
| bf16 training tracks float32 | differential | 60 AdamW steps in bf16 end within 2% of float32 | the capstone in miniature |
| NaN loss rejected | boundary | a NaN loss raises before any update | `L0.5`'s guarantee kept |
| checkpoint gradients bitwise equal | differential | 7 layers, three budgets, float64 and float32, bit for bit | checkpointing never changes training |
| checkpoint replays the dropout generator | differential | same masks, and the stream ends where the plain run's does | dropout in the capstone |
| parameters learn from a data input | unit | the first segment's parameters learn from a data input | the embedding and first block |
| `checkpoint` called directly | boundary | a tensor passed twice, a number argument, `no_grad` | `checkpoint` used directly |
| checkpointing lowers peak memory | property | tracemalloc: forward under half, overall peak lower | the reason to checkpoint |

## 5. The solution, step by step

### From first principles

1. **Rounding has to happen where hardware rounds, and only there.** A tensor core reads 16-bit operands and writes a 16-bit result, so the emulation must round both operands and the output of every matmul. Rounding anything else (the weights, the update) would model a machine that does not exist and would lose the small updates of section 2.
2. **So rounding must be a graph op, not a numpy call.** If the forward pass rounded `a.data` in place, the backward pass would see float32 gradients. Making `cast` a `from_op` node with VJP $\bar x = \mathrm{rd}_{16}(\bar y)$ gives the backward matmuls the same rounding for free, because the autograd engine composes VJPs.
3. **Every matmul goes through `Tensor.__matmul__`, so patching that one method covers every model.** The patch must be scoped (a context manager), must survive exceptions (`finally`), and must nest: only the outermost block installs and removes the patch, and a stack of dtypes says which format applies right now.
4. **Loss scaling follows from linearity.** Every VJP is linear in its upstream gradient, so backpropagating $S L$ produces $S g$. Scaling the loss is one multiplication; unscaling is one multiplication of each master gradient by $1/S$. Everything after unscaling (clipping, the optimizer) then works in true gradient units.
5. **Overflow is detected, not predicted.** A single non-finite entry would poison every weight through AdamW's moments, so the step is all or nothing: skip, drop the gradients, halve $S$, restart the count. Growth is the mirror image: after `interval` good steps double $S$ and restart the count.
6. **Accumulation is a reweighting of the loss.** By the identity in section 2, backpropagating $\tfrac{n_i}{N} L_i$ for each micro-batch without zeroing sums to $\nabla L$. The weight must use row counts, which means the solution must know each micro-batch's size before it starts.
7. **Checkpointing is one graph node whose VJP reruns the forward.** The forward runs under `no_grad`, so nothing inside is saved. The node's parents must include the parameters (otherwise a data input gives a node that is never recorded), and the rerun must start from fresh leaves (otherwise its backward would flow into the outer graph a second time) and replay the random generators.

### The code, annotated

The cast and the patched matmul. `_round` keeps the array's own float dtype, so a float64 test stays float64 (pitfall 4).

```python
def _round(a: np.ndarray, dtype: str) -> np.ndarray:
    """a rounded to dtype, in a's own float dtype."""
    return np.asarray(_ROUND[dtype](a)).astype(a.dtype, copy=False)


def cast(x: Tensor, dtype: str) -> Tensor:
    _check(dtype)
    # forward: round the values; backward: round the gradient (straight-through, but in 16 bits)
    return from_op(_round(x.data, dtype), (x,), lambda g: (_round(np.asarray(g), dtype),), f"cast_{dtype}")


def _matmul(a: Tensor, b: Any) -> Tensor:
    """a @ b with rounded operands and output (the patched __matmul__)."""
    dt = _stack[-1]                 # the innermost open block decides the format
    b = _as_tensor(b, a)
    # both operands rounded, the float32 product rounded once more on the way out
    return cast(_orig["mm"](cast(a, dt), cast(b, dt)), dt)
```

`_rmatmul` is the same with the operands swapped, for `ndarray @ Tensor`. The context manager installs the patch only when the stack is empty and removes it only when the stack empties again:

```python
@contextmanager
def autocast(dtype: str) -> Iterator[None]:
    _check(dtype)                   # an unknown dtype raises before anything is patched
    if not _stack:                  # outermost block: save the originals and patch
        _orig["mm"], _orig["rmm"] = Tensor.__matmul__, Tensor.__rmatmul__
        Tensor.__matmul__, Tensor.__rmatmul__ = _matmul, _rmatmul
    _stack.append(dtype)
    try:
        yield
    finally:                        # runs on an exception too (pitfall 5)
        _stack.pop()
        if not _stack:              # an inner block leaving does not unpatch the outer one (pitfall 6)
            Tensor.__matmul__, Tensor.__rmatmul__ = _orig.pop("mm"), _orig.pop("rmm")
```

The scaler's `step` is the whole of section 2's dynamic scaling in order: check, skip or unscale, clip, step, grow.

```python
def step(self, opt: Any, params: Iterable[Any], clip: Optional[float] = None) -> bool:
    ps = [p for p in params if p.grad is not None]
    if not all(np.isfinite(p.grad).all() for p in ps):
        for p in ps:
            p.grad = None               # drop the poisoned gradients (pitfall 13)
        self.loss_scale *= self.backoff
        self.good_steps = 0             # restart the count (pitfall 14)
        return False
    inv = 1.0 / self.loss_scale
    for p in ps:
        p.grad *= inv                   # unscale first: the clip bound is in true units (pitfall 15)
    self.last_grad_norm = float(clip_grad_norm_(ps, clip)) if clip is not None else None
    opt.step()
    self.good_steps += 1
    if self.good_steps == self.interval:
        self.loss_scale *= self.growth
        self.good_steps = 0
    return True
```

`state_dict` saves `good_steps` along with $S$, so a resumed run grows $S$ on the same step the uninterrupted run would (pitfall 16).

Accumulation computes all sizes first, because each weight needs the total $N$:

```python
def grad_accumulate(model, micro_batches, loss_fn, scaler=None) -> float:
    mbs = list(micro_batches)
    # ... empty list or an empty micro-batch raises
    sizes = [_size(mb) for mb in mbs]
    total = sum(sizes)
    out = 0.0
    for mb, n in zip(mbs, sizes):
        loss = _split(loss_fn(model, mb))
        w = n / total  # a mean over n rows times n / N is that rows' share of the big mean
        part = loss * w
        # no zero_grad here: .grad accumulates across micro-batches (pitfall 9)
        (scaler.scale(part) if scaler is not None else part).backward()
        out += w * float(np.asarray(loss.data).reshape(()))   # the reported loss is weighted too (pitfall 10)
    return out
```

`train_step_mixed` composes these: `opt.zero_grad()`, then `grad_accumulate` inside `autocast(precision)` (no block for `"fp32"`), then either `scaler.step(...)`, which may skip, or a check that raises `FloatingPointError` on a non-finite loss before clipping and `opt.step()`.

The checkpoint node:

```python
def checkpoint(fn, *args, params=(), rngs=()) -> Tensor:
    # each distinct Tensor once: a tensor passed twice gets one summed gradient (pitfall 20)
    tensors = list({id(a): a for a in args if isinstance(a, Tensor)}.values())
    parents = tensors + list(params)     # parameters are parents, so a data input still records a node
    if not is_grad_enabled():
        return fn(*args)
    before = [r.state() for r in rngs]   # where each generator stood before the forward pass
    with no_grad():
        out = fn(*args)                  # no graph: the activations inside fn are freed now

    def vjp(g):
        after = [r.state() for r in rngs]
        for r, s in zip(rngs, before):
            r.set_state(s)               # replay the same dropout masks (pitfall 21)
        # fresh leaves: the rerun's backward stops here instead of entering the outer graph (pitfall 24)
        leaves = {id(a): Tensor(a.data, requires_grad=a.requires_grad, dtype=a.data.dtype) for a in tensors}
        try:
            y = fn(*[leaves[id(a)] if isinstance(a, Tensor) else a for a in args])
            if y.requires_grad:
                y.backward(g)            # parameters inside fn get their .grad here, directly
        finally:
            for r, s in zip(rngs, after):
                r.set_state(s)           # the stream continues where the forward pass left it (pitfall 22)
        grads = [leaves[id(a)].grad for a in tensors]
        return grads + [None] * len(params)   # parameter grads were already added by y.backward

    return from_op(out.data, parents, vjp, "checkpoint")
```

The VJP returns `None` for the parameters because the rerun's `backward` has already added their gradients into `.grad`; returning them again would double them. `checkpoint_sequential` applies `checkpoint_schedule` and runs the last segment plainly:

```python
starts = checkpoint_schedule(len(layers), mem_budget_layers)
ends = starts[1:] + [len(layers)]
for i, (a, b) in enumerate(zip(starts, ends)):
    seg = layers[a:b]
    if i == len(starts) - 1:
        x = _segment(seg)(x)  # the last segment keeps its activations: no rerun (pitfall 18)
    else:
        params = [p for m in seg for p in m.parameters()]
        x = checkpoint(_segment(seg), x, params=params, rngs=rngs)
```

Full solution: [`precision.py`](../../../../course/solutions/python/tinyllm/train/precision.py) · [`recompute.py`](../../../../course/solutions/python/tinyllm/train/recompute.py)

### Why this solution

- **Cost.** Autocast adds three elementwise roundings per matmul, $O(\text{size})$ next to the matmul's $O(MNK)$. Accumulation costs nothing extra in compute and divides activation memory by $k$. Checkpointing with segments $s_j$ runs every checkpointed layer forward twice (about one extra forward pass, a third more compute) and keeps the activations of one segment plus the segment inputs, as `M08.4` counted.
- **Patching `__matmul__` versus a dtype on every tensor.** A real per-tensor dtype would mean touching every op in the library. Patching the one method every matmul goes through gives exact hardware numerics for matmuls, which are where the error lives, with ten lines and no change to any model. The tradeoff accepted: it is process-wide state, hence the stack and the `finally`.
- **Dynamic versus static scaling.** A fixed $S$ must be tuned per model and fails when the gradient scale drifts during training. Dynamic scaling costs a skipped step on each overflow and nothing else.
- **Weighting by $n_i/N$ versus $1/k$.** The two agree only for equal sizes; the row-count weight is exact for every split, at the cost of computing sizes up front.
- **A reentrant node versus saved-tensor hooks.** PyTorch's non-reentrant checkpoint frees memory more aggressively, but needs hooks the `L0.1` engine does not have. One `from_op` node needs only the engine as it is, and the price is the explicit `params` argument.

### Explain it back

**Why is the backward of `cast` the cast of the gradient, and not the true derivative?**

<details><summary>Answer</summary>

The true derivative of rounding is zero almost everywhere, which would stop learning. Hardware passes the gradient through the conversion, and that gradient is itself stored in 16 bits, so the faithful emulation rounds it. This also makes the backward matmuls bf16 matmuls without any extra code.

</details>

**Why is it safe to multiply the loss by $S$ and divide the gradients by $S$, and why a power of two?**

<details><summary>Answer</summary>

Backprop is linear in the upstream gradient, so every gradient is exactly $S$ times the true one, and dividing restores it. A power of two changes only the exponent, so neither the scaling nor the unscaling rounds.

</details>

**Why does the scaler set every gradient to `None` when it skips a step?**

<details><summary>Answer</summary>

A loop that forgets `zero_grad` would otherwise add the next step's gradients onto $\infty$ or NaN and skip forever. Dropping them makes the skip leave no trace except the smaller $S$.

</details>

**What goes wrong if `checkpoint` leaves the parameters out of the node's parents?**

<details><summary>Answer</summary>

When the segment's input is data (no `requires_grad`), `from_op` sees no parent that requires grad and records no node, so backward never reaches the rerun and the segment's parameters get no gradient. With the parameters as parents, the node is recorded and its VJP runs.

</details>

**Why does the VJP restore the generators to the *after* state when it is done?**

<details><summary>Answer</summary>

Setting them to the *before* state replays the forward masks, but leaves the stream rewound. Later draws would then repeat earlier ones, and a run with checkpointing would diverge from one without it. Restoring the state the stream had reached makes the rerun invisible.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. not rounding the matmul output | bf16 runs look more accurate than hardware will be | `_matmul` wraps the product in a final `cast` |
| 2. passing the gradient through the cast unrounded | float32 gradients from a "bf16" backward | `cast`'s VJP rounds the gradient with `_round` |
| 3. rounding only one operand | half the error budget, unnoticed | `_matmul` casts both `a` and `b` |
| 4. a cast that changes float64 to float32 | dtype mixing downstream | `_round` converts back with `.astype(a.dtype)` |
| 5. no `try`/`finally` around the patched matmul | an exception leaves every later matmul in bf16 | `autocast` restores the methods in `finally` |
| 6. leaving an inner block ends the outer one | the rest of the outer block runs in float32 | only the block that empties `_stack` unpatches |
| 7. weighting micro-batches $1/k$ | wrong gradient whenever the sizes differ | `grad_accumulate` weights by `n / total` |
| 8. summing micro-batch losses | gradients $k$ times too large | each part is `loss * w` with weights summing to 1 |
| 9. zeroing gradients between micro-batches | only the last micro-batch trains | `grad_accumulate` never zeroes; `train_step_mixed` zeroes once, before |
| 10. reporting the unweighted mean loss | the logged loss disagrees with the gradient's | the returned loss is `sum(w * loss_i)` |
| 11. never unscaling | the update is $S$ times too large | `step` multiplies every gradient by `1 / loss_scale` before `opt.step()` |
| 12. stepping on non-finite gradients | one overflow makes every weight NaN | `step` checks `np.isfinite` on every gradient and returns `False` |
| 13. keeping the bad gradients after a skip | a loop without `zero_grad` adds onto $\infty$ | a skipped step sets every `p.grad = None` |
| 14. not restarting the count after an overflow | $S$ grows right back into overflow | `good_steps = 0` on overflow and on growth |
| 15. clipping before unscaling | the clip acts on $S \cdot g$: tiny updates | unscaling comes before `clip_grad_norm_` in `step` |
| 16. losing the count on resume | the resumed run grows $S$ late | `state_dict` saves `good_steps` |
| 17. stepping on a NaN loss without a scaler | the run is poisoned silently | `train_step_mixed` raises `FloatingPointError` before `opt.step()` |
| 18. checkpointing the last segment too | every layer runs twice for nothing | `checkpoint_sequential` runs the last segment through `_segment` directly |
| 19. parameters not parents of the node | the first segment stops learning on data input | `params` are part of the node's `parents` |
| 20. a tensor passed twice gets its gradient twice | doubled gradients for shared inputs | `tensors` deduplicates the arguments by `id` |
| 21. not replaying the generator | the rerun's dropout mask differs from the forward pass's | the VJP sets each generator to its `before` state |
| 22. not restoring the generator after the rerun | later draws repeat; runs with and without checkpointing diverge | the VJP's `finally` sets each generator to its `after` state |
| 23. running the forward pass with the graph on | correct gradients, no memory saved | the forward pass runs under `no_grad()` |
| 24. rerunning on the original tensors | the rerun's backward runs into the outer graph | the rerun reads fresh leaf copies of the inputs |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | `from_op` builds `cast` and the checkpoint node; grad mode decides what the forward pass records |
| Back | `L0.3` | `mse` is the examples' loss |
| Back | `L0.4` | `Module.parameters()` lists a segment's parents |
| Back | `M09.1` | `round_to_bf16` and `round_to_fp16` are the emulated formats |
| Back | `M10.2` | `SGD` in the worked examples |
| Back | `M10.3` | `AdamW` in the capstone |
| Back | `M10.4` | `clip_grad_norm_` after unscaling |
| Back | `M08.4` | `checkpoint_schedule` plans the segments |
| Forward | `C1` | the capstone trainer: bf16, accumulation of 4 micro-batches of 8, checkpointed blocks, compared with a float32 run |

Without this module, the capstone's `--bf16 --accum --checkpoint-activations` flags have nothing to call.

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `autocast` | `torch.autocast` | per-op cast policies (softmax and norms kept in float32), real bf16 kernels | `torch/amp/autocast_mode.py`, `aten/src/ATen/autocast_mode.cpp` |
| `DynamicLossScaler` | `torch.amp.GradScaler` | per-device inf checks fused into one kernel, optimizer-step skipping | `torch/amp/grad_scaler.py` |
| bf16 and fp16 | NVIDIA Transformer Engine | fp8 matmuls with per-tensor delayed scaling, the same idea one format lower | `transformer_engine/pytorch/fp8.py` |
| `checkpoint_sequential` | `torch.utils.checkpoint` | non-reentrant checkpointing with saved-tensor hooks, RNG state for every device | `torch/utils/checkpoint.py` |
| uniform segments | Megatron-LM selective recomputation | recompute only attention's cheap, large softmax and dropout, keep the matmul outputs | `megatron/core/tensor_parallel/random.py` |
| `grad_accumulate` | DeepSpeed `gradient_accumulation_steps` | accumulation fused with ZeRO's gradient partitioning (`L11.3`) | `deepspeed/runtime/engine.py` |
