# Data parallelism and ZeRO stages 1 to 3

## Overview

| | |
|---|---|
| **Module** | `L11.3` · side · Python · Pass 9 (optional) |
| **Solution** | [`zero.py`](../../../../course/solutions/python/tinyllm/dist/zero.py): `DDP` (`forward`, `sync_grads`), `ZeroOptimizer` (`gather`, `step`, `zero_grad`, `memory_bytes`) |
| **Contract** | [`course/contracts/py/tinyllm/dist/zero.pyi`](../../../../course/contracts/py/tinyllm/dist/zero.pyi) |
| **Needs** | `L11.2` `Comm` and `chunk_bounds` · `L0.1` Tensor · `L0.4` Module and layers · `M10.2` SGD and `M10.3` AdamW (single-process references and wrapped optimizers) · reading: `M05.1` the memory plan, `L11.1` accumulation |
| **Used by** | the capstone trainer's `--world 4 --zero 2` option (`C1`) |
| **Optional depth** | Li et al., [*PyTorch Distributed: Experiences on Accelerating Data Parallel Training*](https://arxiv.org/abs/2006.15704) (VLDB 2020); Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054) (SC 2020); Zhao et al., [*PyTorch FSDP*](https://arxiv.org/abs/2304.11277) (VLDB 2023) |

## Key Takeaways

- Data parallelism is gradient accumulation across processes: equal slices, mean losses, an averaged gradient, and every replica takes the same step.
- Replicas must start equal, so DDP broadcasts rank 0's weights before the first step.
- ZeRO keeps one copy of each piece of training state across the ranks instead of $p$: stage 1 the optimizer state, stage 2 also the gradients, stage 3 also the parameters, so per-rank memory falls toward $1/p$.
- Sharding changes who stores what, never the arithmetic: an elementwise optimizer on a chunk computes exactly the single process's update for those entries.

## How to read this chapter

Sections 1 to 3 derive data parallelism and the three ZeRO stages from first principles and work one step and its memory by hand. Section 4 fixes the interface. Section 5 walks through `zero.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L11.2` built an all-reduce. The first thing anyone builds with it is data parallelism: $p$ workers, each with a full copy of the model, each computing the gradient of its own slice of the batch, then averaging. That turns $p$ processes into one large-batch trainer, and it is exactly `L11.1`'s accumulation spread over processes. But every replica also holds a full copy of the optimizer state: for AdamW, two moments per parameter, twice the size of the model. At scale that redundancy is what runs out of memory, and ZeRO removes it in three stages. On a laptop the capstone does not need either, so this module is optional; it is how every large model you will serve was trained, and the capstone's `--world 4 --zero 2` run matches the single-process run, which is the evidence that it is right.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$, $r$ | number of ranks, this rank | `int` |
| $w \in \mathbb{R}^n$ | all parameters flattened in `parameters()` order | `float64[n]` |
| $g_r$ | the gradient of rank $r$'s local mean loss | `float64[n]` |
| $\bar g = \frac1p \sum_r g_r$ | the averaged gradient | `float64[n]` |
| $c_r = [a_r, b_r)$ | rank $r$'s chunk of $w$ (`L11.2`'s `chunk_bounds`) | slice |
| $\beta$ | bytes per value (8 for float64) | `int` |
| $K$ | optimizer state values per parameter (2 for AdamW's moments, plus a master copy where kept) | `int` |

**DDP.** Split a batch of $N$ rows into $p$ equal slices. Each rank computes $g_r = \nabla L_r$ of its slice's mean loss. By `L11.1`'s argument with equal sizes, the full batch's gradient is $\frac1p \sum_r g_r = \bar g$. One `all_reduce(op="mean")` over all gradients, flattened into a single bucket, gives every rank $\bar g$ with identical bits (`L11.2`). If every replica starts from the same weights and applies the same optimizer to the same $\bar g$, they stay identical forever, and they equal one process training on the whole batch up to the rounding of the ring's sum. So DDP is three things: broadcast rank 0's parameters at construction, average gradients after `backward()`, and nothing else (the optimizer is untouched).

**The redundancy.** Per rank, DDP with AdamW in float64 holds $n$ parameters, $n$ gradients, and $2n$ moments: $4n\beta$ bytes, the same as `M05.1`'s single-process plan, on every one of the $p$ ranks. Yet each rank uses only the result of the update, which is identical everywhere.

**Stage 1: shard the optimizer state.** Rank $r$ updates only its chunk $c_r$. It keeps a master copy $w[c_r]$ and the optimizer state for those entries only, and wraps them in one flat "parameter" the unchanged optimizer can step. A step: all-reduce the gradients (as DDP), take $\bar g[c_r]$, step, then `all_gather` the updated chunks so every rank's model has the full new $w$. AdamW and SGD are elementwise (each entry's update depends only on that entry's gradient and state), so the chunked update equals the single-process update entry by entry. Optimizer memory per rank: $K n / p$.

**Stage 2: shard the gradients.** After the all-reduce each rank still holds all $n$ averaged gradients but uses only $c_r$. Replace the all-reduce by a `reduce_scatter`, which leaves rank $r$ with just $\sum_r g_r[c_r]$; divide by $p$; drop the full `.grad` arrays. Gradient memory per rank: $n/p$ (in a real system the full gradients are reduce-scattered bucket by bucket during backward, so they never all exist at once). The traffic is the same: a reduce-scatter plus an all-gather is exactly an all-reduce.

**Stage 3: shard the parameters.** Between steps each rank keeps only $w[c_r]$ and releases the rest. Before a forward pass, `gather()` all-gathers the chunks into full parameters; after the step they are released again. Memory per rank: everything over $p$, at the cost of one more all-gather per step (real systems gather layer by layer just before use and free right after).

| state per rank | DDP | stage 1 | stage 2 | stage 3 |
|---|---|---|---|---|
| parameters | $n$ | $n$ | $n$ | $n/p$ |
| gradients | $n$ | $n$ | $n/p$ | $n/p$ |
| optimizer (moments, master) | $Kn$ | $Kn/p$ | $Kn/p$ | $Kn/p$ |

**The emulation's bookkeeping.** `memory_bytes()` reports what the rank holds after a step: the parameters' `.data` (in stage 3 between steps, only the chunk), the `.grad` arrays plus the chunk's gradient, and every array in the wrapped optimizer's `state_dict()` plus, in stages 1 and 2, the chunk's master copy (in stage 3 the chunk is the parameter storage, counted once). Stage 1 keeps the averaged full gradients in `.grad` until `zero_grad()`, as a training loop that logs gradient norms would.

## 3. Worked example by hand

One parameter $w = [0, 1, \dots, 9]$ ($n = 10$) on $p = 4$ ranks. `chunk_bounds(10, 4)` gives chunks of 3, 3, 2, 2: rank 0 owns entries 0 to 2, rank 3 entries 8 and 9. Rank $r$'s loss is $\sum_i (w_i - r)^2/2$, so $g_r = w - r$ and the averaged gradient is $\bar g = w - 1.5$ (the mean of 0, 1, 2, 3).

**The step.** AdamW with $\eta = 0.5$, no weight decay. On the first step, $m = (1 - \beta_1)\bar g$ and $v = (1 - \beta_2)\bar g^2$; after bias correction $\hat m = \bar g$ and $\sqrt{\hat v} = |\bar g|$, so each entry moves by $\eta\, \bar g / |\bar g| = \pm 0.5$ (the $\epsilon = 10^{-8}$ changes the eighth digit). Entries 0 and 1 have $\bar g < 0$ and move up; the others move down:

$$w = [0.5,\ 1.5,\ 1.5,\ 2.5,\ 3.5,\ 4.5,\ 5.5,\ 6.5,\ 7.5,\ 8.5].$$

Every stage gives this on every rank: rank 0 computes entries 0 to 2, rank 3 entries 8 and 9, and the all-gather (or stage 3's `gather()`) assembles the rest.

**The memory** after the step, in bytes (float64, $\beta = 8$):

| | stage 1, rank 0 | stage 1, rank 3 | stage 2, rank 0 | stage 2, rank 3 | stage 3, rank 0 | stage 3, rank 3 | one process |
|---|---|---|---|---|---|---|---|
| params | 80 | 80 | 80 | 80 | 24 | 16 | 80 |
| grads | 80 + 24 = 104 | 80 + 16 = 96 | 24 | 16 | 24 | 16 | 80 |
| optimizer | 2·24 + 24 = 72 | 2·16 + 16 = 48 | 72 | 48 | 48 | 32 | 160 |

Rank 0's stage 3 total is 96 bytes against 320 for one process: about $1/p$, up to the unequal chunks. These are the first two cases in section 4's table.

## 4. The interface

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

A ZeRO training step on each rank: `zo.gather()`, `zo.zero_grad()`, forward and `backward()` on the rank's slice, `zo.step()`. With DDP: `opt.zero_grad()`, forward and `backward()`, `ddp.sync_grads()`, `opt.step()`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: one ZeRO step | unit | section 3's step, every stage, every rank | the chunking is pinned to exact values |
| hand example: memory by stage | unit | section 3's byte table | the stage definitions |
| DDP matches one process | differential | 2 and 4 ranks, 5 momentum-SGD steps, $10^{-10}$, bitwise-equal gradients | DDP is just a bigger batch |
| DDP broadcasts rank 0's weights | unit | ranks initialized differently end equal to rank 0's run | replicas never drift |
| ZeRO matches one process | differential | stages 1 to 3 on 2 and 4 ranks, 4 AdamW steps with decay | sharding never changes training |
| ZeRO memory is $1/p$ | property | a 44-parameter model on 4 ranks: exact shares per stage | the reason to use ZeRO |
| bad stage rejected | boundary | stage 0 or 4, no parameters | configuration errors surface early |

## 5. The solution, step by step

### From first principles

1. **Replicas must compute the same step.** Two things make every rank's update identical: identical starting weights, and identical gradients. The first needs one broadcast at construction; the second needs an all-reduce whose result has the same bits everywhere, which `L11.2`'s ring provides.
2. **The gradient to apply is the mean, not the sum.** Each rank's loss is a mean over its slice, so by `L11.1`'s accumulation identity with equal slices the full-batch gradient is $\frac1p\sum_r g_r$. Summing would scale the step by $p$.
3. **One flat bucket is simpler and as fast.** Flattening every gradient in `parameters()` order into one array means one collective per step instead of one per tensor, and the same flat order defines ZeRO's chunks. Writing results back needs the shapes and dtypes recorded once.
4. **Sharding works because the optimizer is elementwise.** AdamW's update of entry $i$ reads only $g_i$, $m_i$, $v_i$, $w_i$. So rank $r$ can hold a flat "parameter" containing $w[c_r]$, give it the gradient $\bar g[c_r]$, and call the *unchanged* optimizer on it; the result is the single-process update for those entries.
5. **Each stage removes one redundant copy, using the matching collective.** Stage 1 still all-reduces (every rank gets all of $\bar g$) but only uses its chunk. Stage 2 uses reduce-scatter instead, so only the chunk's sum ever arrives, and drops the full gradients. Stage 3 keeps only $w[c_r]$ between steps and all-gathers before the forward pass. Since reduce-scatter plus all-gather is an all-reduce, the traffic does not grow until stage 3's extra gather.

### The code, annotated

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

DDP is the three things of section 2, and nothing else:

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

`ZeroOptimizer` builds its chunk once. `_Chunk` is a two-field object (`data`, `grad`) that looks enough like a parameter for `SGD` or `AdamW` to step it:

```python
n = sum(p.data.size for p in self.params)
self.bounds = chunk_bounds(n, comm.world)          # the same bounds L11.2's collectives use
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

`memory_bytes` counts exactly section 2's bookkeeping: the parameters' `.data` (plus the chunk in stage 3, where it is the storage), the `.grad` arrays plus the chunk's gradient, and the optimizer's `state_dict()` arrays plus the master chunk in stages 1 and 2.

Full solution: [`zero.py`](../../../../course/solutions/python/tinyllm/dist/zero.py)

### Why this solution

- **Cost.** DDP and stages 1 and 2 move one all-reduce's worth of bytes per step, $2(p-1)n\beta/p$ per rank. Stage 3 adds one all-gather, $(p-1)n\beta/p$. Memory per rank falls from $(2 + K)n$ values to about $(2 + K)n/p$ at stage 3.
- **Wrapping the optimizer versus rewriting it.** Because the update is elementwise, giving the unchanged `SGD` or `AdamW` a flat chunk reuses every line of `M10.x` and guarantees the same arithmetic. A non-elementwise optimizer (one that needs a global norm, or Shampoo-style matrix statistics) would break this, and global-norm clipping needs an extra all-reduce of squared chunk norms.
- **One bucket versus per-tensor collectives.** Per-tensor all-reduces pay the ring's $2(p-1)$-step latency once per tensor. One bucket pays it once. Production DDP uses several buckets so it can overlap communication with the rest of backward; this emulation has nothing to overlap with.
- **Accepted tradeoff.** The flat float64 copy in `_flat_grads` briefly holds a full gradient even in stage 2. A real system reduce-scatters bucket by bucket during backward so the full gradient never exists at once.

### Explain it back

**Why must DDP broadcast the weights if every rank seeds the same generator?**

<details><summary>Answer</summary>

Nothing forces every rank to seed the same way, and any difference (a seed, a different init order) means the replicas apply identical updates to different weights and never agree. Broadcasting from rank 0 makes equality a property of the code, not of the caller.

</details>

**Why does sharding the optimizer give exactly the single-process update?**

<details><summary>Answer</summary>

AdamW and SGD update each entry from that entry's gradient and state only. Rank $r$ gives the optimizer $w[c_r]$ and $\bar g[c_r]$, so it computes the same numbers for those entries that one process would; the all-gather then assembles all entries.

</details>

**Why does stage 2 divide the reduce-scatter result by $p$ while stage 1 does not?**

<details><summary>Answer</summary>

Stage 1 calls `all_reduce(op="mean")`, which divides. `reduce_scatter` returns the sum of the chunk, so stage 2 divides by `world` itself to turn it into the mean gradient.

</details>

**What would go wrong if `_write_back` unflattened in a different order from `_flat_grads`?**

<details><summary>Answer</summary>

The sizes would line up only by accident, and values would land in the wrong tensors: one layer's gradient written into another's. Both helpers walk `self.params` in the same `parameters()` order, which is the only thing tying a flat index to a tensor entry.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. not broadcasting the initial weights | replicas with different seeds never agree | `DDP.__init__` broadcasts every parameter from rank 0 |
| 2. summing gradients instead of averaging | the learning rate is effectively $p$ times larger | `sync_grads` uses `all_reduce(op="mean")` |
| 3. synchronizing only some parameters | replicas silently drift apart | `_flat_grads` concatenates every parameter, zeros for `None` |
| 4. a rank starting from the wrong chunk | its master copy belongs to its neighbour | the chunk is `self.bounds[comm.rank]` |
| 5. reduce-scatter without dividing by $p$ | stage 2 and 3 steps on the summed gradient | stages 2 and 3 divide the reduce-scatter by `world` |
| 6. no all-gather after the update | other ranks' chunks go stale in your model | `step` all-gathers the chunks in stages 1 and 2; stage 3 gathers in `gather()` |
| 7. stage 2 keeping the full gradients | no gradient memory saved | stages 2 and 3 set every `p.grad = None` after reducing |
| 8. stage 3 keeping the full parameters | no parameter memory saved | stage 3 calls `_release()` after the step |
| 9. unflattening in the wrong order | parameters swapped between tensors | `_write_back` walks the parameters in the same order `_flat_grads` did |
| 10. dropping the optimizer's hyperparameters | the default learning rate and decay instead of the configured ones | `ZeroOptimizer` passes `**kw` to `opt_cls` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L11.2` | `broadcast`, `all_reduce`, `reduce_scatter`, `all_gather`, and `chunk_bounds` |
| Back | `L0.1` | parameters are Tensors; `.grad` is what gets reduced |
| Back | `L0.4` | `DDP` is a Module wrapping the model; `parameters()` fixes the flat order |
| Back | `M10.2` | SGD in the DDP comparison |
| Back | `M10.3` | AdamW wrapped by `ZeroOptimizer` |
| Forward | `C1` | (optional) `--world 4 --zero 2` trains the capstone on 4 local processes and must match the single-process loss |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `DDP` | `torch.nn.parallel.DistributedDataParallel` | gradient buckets all-reduced during backward (overlapping communication with compute), unused-parameter detection | `torch/nn/parallel/distributed.py`, `torch/csrc/distributed/c10d/reducer.cpp` |
| `ZeroOptimizer` stages 1 and 2 | DeepSpeed ZeRO | bucketed reduce-scatter during backward, CPU and NVMe offload (ZeRO-Offload, ZeRO-Infinity) | `deepspeed/runtime/zero/stage_1_and_2.py` |
| stage 3 | PyTorch FSDP2 | per-layer all-gather just before use, prefetching the next layer, mixed-precision shards | `torch/distributed/fsdp/` |
| clipping | sharded global-norm clipping | an all-reduce of squared chunk norms before the step | `FullyShardedDataParallel.clip_grad_norm_` in `torch/distributed/fsdp/` |
