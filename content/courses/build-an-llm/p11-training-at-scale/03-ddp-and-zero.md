# Data parallelism and ZeRO stages 1 to 3

[Collectives over processes](02-collectives-over-processes.md) built an all-reduce. The first thing anyone builds with it is data parallelism: $p$ workers, each with a full copy of the model, each computing the gradient of its own slice of the batch, then averaging. That turns $p$ processes into one large-batch trainer, and it is exactly the gradient accumulation of [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](01-mixed-precision-accumulation-and-checkpointing.md), spread over processes.

But every replica also holds a full copy of the optimizer state: for AdamW, two moments per parameter, twice the size of the model. At scale that redundancy is what runs out of memory, and ZeRO removes it in three stages. On a laptop the [capstone](../capstones/01-tinystories.md) needs neither, so this section is optional. It is, however, how every large model you will serve was trained, and the capstone's `--world 4 --zero 2` run matches the single-process run, which is the evidence that the implementation is right.

You need `Comm` and `chunk_bounds` from the previous section, the Tensor from [Tensor, broadcasting backward, and no_grad](../p00-foundations/01-tensor-and-broadcasting-backward.md), and the Module from [Module system and basic layers](../p00-foundations/04-module-system-and-layers.md). The single-process references and wrapped optimizers are SGD from [The Optimizer protocol](../../mathematics-for-computing/10-optimization/02-optimizer-protocol-and-sgd.md) and AdamW from [Adam and AdamW](../../mathematics-for-computing/10-optimization/03-adam-and-adamw.md) in *Mathematics for Computing*; the memory plan of [Counting: params, FLOPs, KV bytes, memory plans](../../mathematics-for-computing/05-discrete-math-1/01-counting-params-flops-kv-bytes.md) is useful background. The solution is [`zero.py`](../../../../course/solutions/python/tinyllm/dist/zero.py) (`DDP` with `forward` and `sync_grads`; `ZeroOptimizer` with `gather`, `step`, `zero_grad`, and `memory_bytes`), and the contract is [`zero.pyi`](../../../../course/contracts/py/tinyllm/dist/zero.pyi).

> **Learning objectives**
> - Explain why data parallelism with averaged gradients equals single-process training on the whole batch.
> - State what each ZeRO stage shards and compute the per-rank memory of each stage.
> - Explain why an elementwise optimizer stepped on a chunk gives exactly the single-process update.
> - Implement DDP and ZeRO stages 1 to 3 on top of ring collectives.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$, $r$ | number of ranks, this rank | `int` |
| $w \in \mathbb{R}^n$ | all parameters flattened in `parameters()` order | `float64[n]` |
| $g_r$ | the gradient of rank $r$'s local mean loss | `float64[n]` |
| $\bar g = \frac1p \sum_r g_r$ | the averaged gradient | `float64[n]` |
| $c_r = [a_r, b_r)$ | rank $r$'s chunk of $w$ (from `chunk_bounds`) | slice |
| $\beta$ | bytes per value (8 for float64) | `int` |
| $K$ | optimizer state values per parameter (2 for AdamW's moments, plus a master copy where kept) | `int` |

## Data parallelism

Split a batch of $N$ rows into $p$ equal slices. Each rank computes $g_r = \nabla L_r$ of its slice's mean loss. By the accumulation argument with equal sizes, the full batch's gradient is $\frac1p \sum_r g_r = \bar g$. One `all_reduce(op="mean")` over all gradients, flattened into a single bucket, gives every rank $\bar g$ with identical bits, as the ring guarantees. If every replica starts from the same weights and applies the same optimizer to the same $\bar g$, they stay identical forever, and they equal one process training on the whole batch up to the rounding of the ring's sum.

So DDP is three things: broadcast rank 0's parameters at construction, average gradients after `backward()`, and nothing else. The optimizer is untouched.

## The redundancy, and three stages that remove it

Per rank, DDP with AdamW in float64 holds $n$ parameters, $n$ gradients, and $2n$ moments: $4n\beta$ bytes, the same as the single-process memory plan, on every one of the $p$ ranks. Yet each rank uses only the result of the update, which is identical everywhere. ZeRO removes one redundant copy per stage.

**Stage 1 shards the optimizer state.** Rank $r$ updates only its chunk $c_r$. It keeps a master copy $w[c_r]$ and the optimizer state for those entries only, and wraps them in one flat "parameter" that the unchanged optimizer can step. A step all-reduces the gradients (as DDP does), takes $\bar g[c_r]$, steps, then `all_gather`s the updated chunks so every rank's model has the full new $w$. AdamW and SGD are elementwise (each entry's update depends only on that entry's gradient and state), so the chunked update equals the single-process update entry by entry. Optimizer memory per rank: $K n / p$.

**Stage 2 shards the gradients.** After the all-reduce each rank still holds all $n$ averaged gradients but uses only $c_r$. Replace the all-reduce by a `reduce_scatter`, which leaves rank $r$ with just $\sum_r g_r[c_r]$; divide by $p$; drop the full `.grad` arrays. Gradient memory per rank is $n/p$. (In a real system the full gradients are reduce-scattered bucket by bucket during backward, so they never all exist at once.) The traffic is unchanged, because a reduce-scatter plus an all-gather is exactly an all-reduce.

**Stage 3 shards the parameters.** Between steps each rank keeps only $w[c_r]$ and releases the rest. Before a forward pass, `gather()` all-gathers the chunks into full parameters; after the step they are released again. Memory per rank is everything over $p$, at the cost of one more all-gather per step. (Real systems gather layer by layer just before use and free right after.)

The table summarizes how many values of each kind a rank holds:

| state per rank | DDP | stage 1 | stage 2 | stage 3 |
|---|---|---|---|---|
| parameters | $n$ | $n$ | $n$ | $n/p$ |
| gradients | $n$ | $n$ | $n/p$ | $n/p$ |
| optimizer (moments, master) | $Kn$ | $Kn/p$ | $Kn/p$ | $Kn/p$ |

To make those numbers measurable, `memory_bytes()` reports what the rank holds after a step: the parameters' `.data` (in stage 3 between steps, only the chunk), the `.grad` arrays plus the chunk's gradient, and every array in the wrapped optimizer's `state_dict()` plus, in stages 1 and 2, the chunk's master copy. In stage 3 the chunk is the parameter storage, so it is counted once. Stage 1 keeps the averaged full gradients in `.grad` until `zero_grad()`, as a training loop that logs gradient norms would.

## Working it by hand

Take one parameter $w = [0, 1, \dots, 9]$ ($n = 10$) on $p = 4$ ranks. `chunk_bounds(10, 4)` gives chunks of 3, 3, 2, 2: rank 0 owns entries 0 to 2, and rank 3 owns entries 8 and 9. Rank $r$'s loss is $\sum_i (w_i - r)^2/2$, so $g_r = w - r$ and the averaged gradient is $\bar g = w - 1.5$ (the mean of 0, 1, 2, 3).

**The step.** Use AdamW with $\eta = 0.5$ and no weight decay. On the first step, $m = (1 - \beta_1)\bar g$ and $v = (1 - \beta_2)\bar g^2$; after bias correction $\hat m = \bar g$ and $\sqrt{\hat v} = |\bar g|$, so each entry moves by $\eta\, \bar g / |\bar g| = \pm 0.5$ (the $\epsilon = 10^{-8}$ changes the eighth digit). Entries 0 and 1 have $\bar g < 0$ and move up; the others move down:

$$w = [0.5,\ 1.5,\ 1.5,\ 2.5,\ 3.5,\ 4.5,\ 5.5,\ 6.5,\ 7.5,\ 8.5].$$

Every stage gives this on every rank: rank 0 computes entries 0 to 2, rank 3 entries 8 and 9, and the all-gather (or stage 3's `gather()`) assembles the rest.

**The memory.** After the step, in bytes (float64, $\beta = 8$), the ranks hold:

| | stage 1, rank 0 | stage 1, rank 3 | stage 2, rank 0 | stage 2, rank 3 | stage 3, rank 0 | stage 3, rank 3 | one process |
|---|---|---|---|---|---|---|---|
| params | 80 | 80 | 80 | 80 | 24 | 16 | 80 |
| grads | 80 + 24 = 104 | 80 + 16 = 96 | 24 | 16 | 24 | 16 | 80 |
| optimizer | 2·24 + 24 = 72 | 2·16 + 16 = 48 | 72 | 48 | 48 | 32 | 160 |

Rank 0's stage 3 total is 96 bytes against 320 for one process: about $1/p$, up to the unequal chunks.

## Designing the API

```python
# python/tinyllm/dist/zero.py
class DDP(Module):
    def __init__(self, model: Module, comm: Comm)       # broadcasts rank 0's parameters
    def forward(self, *args, **kwargs)                  # model(*args, **kwargs)
    def sync_grads(self) -> None                        # one bucket, all_reduce(op="mean")
class ZeroOptimizer:
    def __init__(self, opt_cls, params, comm: Comm, stage: Literal[1, 2, 3], **kw)
    def gather(self) -> None                            # stage 3: full parameters before forward
    def step(self) -> None                              # reduce, update the chunk, gather or release
    def zero_grad(self) -> None
    def memory_bytes(self) -> dict[str, int]            # params, grads, optimizer
```

A ZeRO training step on each rank is `zo.gather()`, `zo.zero_grad()`, forward and `backward()` on the rank's slice, then `zo.step()`. With DDP it is `opt.zero_grad()`, forward and `backward()`, `ddp.sync_grads()`, then `opt.step()`.

## Building it step by step

Five observations determine the code.

1. **Replicas must compute the same step.** Two things make every rank's update identical: identical starting weights, and identical gradients. The first needs one broadcast at construction; the second needs an all-reduce whose result has the same bits everywhere, which the ring provides.
2. **The gradient to apply is the mean, not the sum.** Each rank's loss is a mean over its slice, so by the accumulation identity with equal slices the full-batch gradient is $\frac1p\sum_r g_r$. Summing would scale the step by $p$.
3. **One flat bucket is simpler and as fast.** Flattening every gradient in `parameters()` order into one array means one collective per step instead of one per tensor, and the same flat order defines ZeRO's chunks. Writing results back needs the shapes and dtypes recorded once.
4. **Sharding works because the optimizer is elementwise.** AdamW's update of entry $i$ reads only $g_i$, $m_i$, $v_i$, $w_i$. So rank $r$ can hold a flat "parameter" containing $w[c_r]$, give it the gradient $\bar g[c_r]$, and call the *unchanged* optimizer on it; the result is the single-process update for those entries.
5. **Each stage removes one redundant copy, using the matching collective.** Stage 1 still all-reduces (every rank gets all of $\bar g$) but only uses its chunk. Stage 2 uses reduce-scatter instead, so only the chunk's sum ever arrives, and drops the full gradients. Stage 3 keeps only $w[c_r]$ between steps and all-gathers before the forward pass. Since reduce-scatter plus all-gather is an all-reduce, the traffic does not grow until stage 3's extra gather.

### The flat layout

Two helpers carry the flat layout. Flattening uses zeros for a parameter without a gradient, so every rank sends arrays of the same length even if some parameter was unused:

```python
def _flat_grads(params: list) -> np.ndarray:
    """Every parameter's gradient (zeros when None), flattened in order, float64."""
    return np.concatenate(
        [
            np.zeros(p.data.size) if p.grad is None else np.asarray(p.grad, dtype=np.float64).reshape(-1)
            for p in params
        ]
    )


def _write_back(params, shapes, dtypes, flat, attr) -> None:
    at = 0
    for p, shape, dt in zip(params, shapes, dtypes):   # the same order as _flat_grads
        size = int(np.prod(shape)) if shape else 1
        piece = flat[at : at + size].reshape(shape).astype(dt)
        if attr == "data" and p.data.shape == shape:
            p.data[...] = piece  # in place: the arrays other code holds stay the same objects
        else:
            setattr(p, attr, piece)   # a gradient, or stage 3 restoring a released parameter
        at += size
```

### DDP

DDP is the three things described earlier, and nothing else:

```python
class DDP(Module):
    def __init__(self, model: Module, comm: Comm) -> None:
        super().__init__()
        self.module = model
        self._comm = comm
        for p in model.parameters():
            p.data[...] = comm.broadcast(p.data, 0)   # every replica starts from rank 0's weights

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_grads(self) -> None:
        ps = list(self.module.parameters())
        mean = self._comm.all_reduce(_flat_grads(ps), op="mean")   # one bucket, averaged
        _write_back(ps, [p.data.shape for p in ps], [p.data.dtype for p in ps], mean, "grad")
```

### The ZeRO optimizer

`ZeroOptimizer` builds its chunk once. `_Chunk` is a two-field object (`data`, `grad`) that looks enough like a parameter for `SGD` or `AdamW` to step it:

```python
n = sum(p.data.size for p in self.params)
self.bounds = chunk_bounds(n, comm.world)          # the same bounds the collectives use
a, b = self.bounds[comm.rank]
full = np.concatenate([np.asarray(p.data, dtype=np.float64).reshape(-1) for p in self.params])
self.chunk = _Chunk(full[a:b].copy())              # this rank's master copy
self.opt = opt_cls([self.chunk], **kw)             # the unchanged optimizer, with the caller's hyperparameters
if stage == 3:
    self._release()                                # stage 3: drop the full parameters right away
```

The step is where the stages differ:

```python
def step(self) -> None:
    a, b = self.bounds[self.comm.rank]
    g = _flat_grads(self.params)
    if self.stage == 1:
        # all of the mean arrives; only the chunk is used. .grad keeps the full averaged gradient
        self.chunk.grad = self.comm.all_reduce(g, op="mean")[a:b].copy()
    else:
        # only the chunk's sum arrives; divide by p to make it a mean
        self.chunk.grad = self.comm.reduce_scatter(g) / self.comm.world
        for p in self.params:
            p.grad = None  # the full gradient is not kept: only the chunk's
    self.opt.step()                                # elementwise: equals the single-process update here
    if self.stage == 3:
        self._release()                            # keep only the chunk until the next gather()
        return
    full = self.comm.all_gather(self.chunk.data)   # stages 1 and 2: every rank gets the new w
    _write_back(self.params, self.shapes, self.dtypes, full, "data")

def gather(self) -> None:
    if self.stage != 3:
        return                                     # stages 1 and 2 already hold full parameters
    full = self.comm.all_gather(self.chunk.data)
    _write_back(self.params, self.shapes, self.dtypes, full, "data")
```

`memory_bytes` counts exactly the bookkeeping described above: the parameters' `.data` (plus the chunk in stage 3, where it is the storage), the `.grad` arrays plus the chunk's gradient, and the optimizer's `state_dict()` arrays plus the master chunk in stages 1 and 2.

The full solution is [`zero.py`](../../../../course/solutions/python/tinyllm/dist/zero.py), and a model test suite is [`test_l11_3_mine.py`](../../../../course/solutions/learner-tests/L11.3/python/tests/l11-3-zero/test_l11_3_mine.py).

### Why this design

DDP and stages 1 and 2 move one all-reduce's worth of bytes per step, $2(p-1)n\beta/p$ per rank. Stage 3 adds one all-gather, $(p-1)n\beta/p$. Memory per rank falls from $(2 + K)n$ values to about $(2 + K)n/p$ at stage 3.

Wrapping the optimizer instead of rewriting it is possible because the update is elementwise: giving the unchanged `SGD` or `AdamW` a flat chunk reuses every line of the optimizers in *Mathematics for Computing* and guarantees the same arithmetic. A non-elementwise optimizer (one that needs a global norm, or Shampoo-style matrix statistics) would break this, and global-norm clipping needs an extra all-reduce of squared chunk norms.

One bucket beats per-tensor collectives because per-tensor all-reduces pay the ring's $2(p-1)$-step latency once per tensor, while one bucket pays it once. Production DDP uses several buckets so it can overlap communication with the rest of backward; this emulation has nothing to overlap with. The accepted tradeoff is that the flat float64 copy in `_flat_grads` briefly holds a full gradient even in stage 2. A real system reduce-scatters bucket by bucket during backward, so the full gradient never exists at once.

## What can go wrong

A correct solution must pass these checks:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: one ZeRO step | unit | the worked step, every stage, every rank | the chunking is pinned to exact values |
| hand example: memory by stage | unit | the worked byte table | the stage definitions |
| DDP matches one process | differential | 2 and 4 ranks, 5 momentum-SGD steps, $10^{-10}$, bitwise-equal gradients | DDP is just a bigger batch |
| DDP broadcasts rank 0's weights | unit | ranks initialized differently end equal to rank 0's run | replicas never drift |
| ZeRO matches one process | differential | stages 1 to 3 on 2 and 4 ranks, 4 AdamW steps with decay | sharding never changes training |
| ZeRO memory is $1/p$ | property | a 44-parameter model on 4 ranks: exact shares per stage | the reason to use ZeRO |
| bad stage rejected | boundary | stage 0 or 4, no parameters | configuration errors surface early |

The DDP pitfalls are about keeping replicas identical. Without broadcasting the initial weights, replicas with different seeds never agree; `DDP.__init__` broadcasts every parameter from rank 0. Summing gradients instead of averaging makes the learning rate effectively $p$ times larger; `sync_grads` uses `all_reduce(op="mean")`. Synchronizing only some parameters lets replicas silently drift apart; `_flat_grads` concatenates every parameter, with zeros for `None`.

The ZeRO pitfalls are about the chunk. A rank starting from the wrong chunk holds a master copy that belongs to its neighbour; the chunk is `self.bounds[comm.rank]`. A reduce-scatter without dividing by $p$ makes stages 2 and 3 step on the summed gradient; those stages divide by `world`. Without an all-gather after the update, other ranks' chunks go stale in your model; `step` all-gathers in stages 1 and 2, and stage 3 gathers in `gather()`. If stage 2 keeps the full gradients, no gradient memory is saved; stages 2 and 3 set every `p.grad = None` after reducing. If stage 3 keeps the full parameters, no parameter memory is saved; stage 3 calls `_release()` after the step. Dropping the optimizer's hyperparameters silently gives the default learning rate and decay instead of the configured ones; `ZeroOptimizer` passes `**kw` to `opt_cls`.

> **Warning:** Unflattening in a different order from flattening swaps values between tensors. `_write_back` must walk the parameters in the same order `_flat_grads` did.

## Where this leads

This is the end of the training-at-scale chapter. The optional `--world 4 --zero 2` mode of the [capstone trainer](../capstones/01-tinystories.md) trains on 4 local processes and must match the single-process loss. The production systems below take the same ideas much further:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `DDP` | `torch.nn.parallel.DistributedDataParallel` | gradient buckets all-reduced during backward (overlapping communication with compute), unused-parameter detection | `torch/nn/parallel/distributed.py`, `torch/csrc/distributed/c10d/reducer.cpp` |
| `ZeroOptimizer` stages 1 and 2 | DeepSpeed ZeRO | bucketed reduce-scatter during backward, CPU and NVMe offload (ZeRO-Offload, ZeRO-Infinity) | `deepspeed/runtime/zero/stage_1_and_2.py` |
| stage 3 | PyTorch FSDP2 | per-layer all-gather just before use, prefetching the next layer, mixed-precision shards | `torch/distributed/fsdp/` |
| clipping | sharded global-norm clipping | an all-reduce of squared chunk norms before the step | `FullyShardedDataParallel.clip_grad_norm_` in `torch/distributed/fsdp/` |

### Further reading

- Li et al., [*PyTorch Distributed: Experiences on Accelerating Data Parallel Training*](https://arxiv.org/abs/2006.15704) (VLDB 2020).
- Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054) (SC 2020).
- Zhao et al., [*PyTorch FSDP*](https://arxiv.org/abs/2304.11277) (VLDB 2023).

## Key terms

- **data parallelism (DDP)**: each rank holds a full model replica, computes the gradient of its batch slice, and all ranks average gradients before stepping.
- **replica**: one rank's full copy of the model.
- **ZeRO**: sharding training state across ranks so that each piece is stored once instead of $p$ times.
- **stage 1, 2, 3**: ZeRO sharding of the optimizer state; also the gradients; also the parameters.
- **chunk**: the contiguous range of the flat parameter vector a rank owns, given by `chunk_bounds`.
- **elementwise optimizer**: an optimizer whose update of each entry depends only on that entry's gradient, state, and value.

## Summary

- Data parallelism is gradient accumulation across processes: equal slices, mean losses, an averaged gradient, and every replica takes the same step.
- Replicas must start equal, so DDP broadcasts rank 0's weights before the first step.
- ZeRO keeps one copy of each piece of training state across the ranks instead of $p$: stage 1 the optimizer state, stage 2 also the gradients, stage 3 also the parameters, so per-rank memory falls toward $1/p$.
- Sharding changes who stores what, never the arithmetic: an elementwise optimizer on a chunk computes exactly the single process's update for those entries.

## Review questions

1. Why must DDP broadcast the weights if every rank seeds the same generator?

   <details><summary>Answer</summary>

   Nothing forces every rank to seed the same way, and any difference (a seed, a different init order) means the replicas apply identical updates to different weights and never agree. Broadcasting from rank 0 makes equality a property of the code, not of the caller.

   </details>

2. Why does sharding the optimizer give exactly the single-process update?

   <details><summary>Answer</summary>

   AdamW and SGD update each entry from that entry's gradient and state only. Rank $r$ gives the optimizer $w[c_r]$ and $\bar g[c_r]$, so it computes the same numbers for those entries that one process would; the all-gather then assembles all entries.

   </details>

3. Why does stage 2 divide the reduce-scatter result by $p$ while stage 1 does not?

   <details><summary>Answer</summary>

   Stage 1 calls `all_reduce(op="mean")`, which divides. `reduce_scatter` returns the sum of the chunk, so stage 2 divides by `world` itself to turn it into the mean gradient.

   </details>

4. What would go wrong if `_write_back` unflattened in a different order from `_flat_grads`?

   <details><summary>Answer</summary>

   The sizes would line up only by accident, and values would land in the wrong tensors: one layer's gradient written into another's. Both helpers walk `self.params` in the same `parameters()` order, which is the only thing tying a flat index to a tensor entry.

   </details>
