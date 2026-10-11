<!-- ol:module L11.3 -->
# Data parallelism and ZeRO stages 1 to 3

## Overview

| | |
|---|---|
| **Module** | `L11.3` · side · Python · Pass 9 (optional) · 4 to 5 h |
| **You build** | `python/tinyllm/dist/zero.py`: `DDP` (`forward`, `sync_grads`), `ZeroOptimizer` (`gather`, `step`, `zero_grad`, `memory_bytes`) |
| **Contract** | [`course/contracts/py/tinyllm/dist/zero.pyi`](../../../course/contracts/py/tinyllm/dist/zero.pyi) |
| **Tests** | `course/tests/L11.3/` (what they check: section 4) · your own tests in `python/tests/l11-3-zero/`, rung R5, graded by mutation (threshold 0.80, every pitfall mutant required) |
| **Needs** | `L11.2` `Comm` and `chunk_bounds` · `L0.1` Tensor · `L0.4` Module and layers · `M10.2` SGD and `M10.3` AdamW (single-process references and wrapped optimizers) · reading: `M05.1` the memory plan, `L11.1` accumulation |
| **Used by** | the capstone trainer's `--world 4 --zero 2` option (`C1`, joins the registry with the capstone) |
| **Milestone** | none; this optional side module is not part of `MS-L11` |
| **Optional depth** | Li et al., [*PyTorch Distributed: Experiences on Accelerating Data Parallel Training*](https://arxiv.org/abs/2006.15704) (VLDB 2020); Rajbhandari et al., [*ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*](https://arxiv.org/abs/1910.02054) (SC 2020); Zhao et al., [*PyTorch FSDP*](https://arxiv.org/abs/2304.11277) (VLDB 2023) |

## Key Takeaways

- Data parallelism is gradient accumulation across processes: equal slices, mean losses, an averaged gradient, and every replica takes the same step (`test_ddp_matches_single_process`).
- Replicas must start equal, so DDP broadcasts rank 0's weights before the first step (`test_ddp_broadcasts_rank0_weights`).
- ZeRO keeps one copy of each piece of training state across the ranks instead of $p$: stage 1 the optimizer state, stage 2 also the gradients, stage 3 also the parameters, so per-rank memory falls toward $1/p$ (`test_hand_example_memory_by_stage`, `test_zero_memory_is_one_over_p`).
- Sharding changes who stores what, never the arithmetic: an elementwise optimizer on a chunk computes exactly the single process's update for those entries (`test_hand_example_zero_step`, `test_zero_matches_single_process`).

## How to work this chapter

```bash
ol start L11.3              # stubs zero.py into your repo
ol tests L11.3              # read the test catalog first
ol check L11.3              # course tests, then your tests graded by mutation
ol mutate L11.3             # the full mutation grade of your tests
ol check L11.3 --ref-deps   # only if your L11.2 is not passing yet
ol diff  L11.3              # after passing: your code against the reference
```

---

## 1. Why now

`L11.2` gave you an all-reduce. The first thing anyone builds with it is data parallelism: $p$ workers, each with a full copy of the model, each computing the gradient of its own slice of the batch, then averaging. That turns $p$ processes into one large-batch trainer, and it is exactly `L11.1`'s accumulation spread over processes. But every replica also holds a full copy of the optimizer state: for AdamW, two moments per parameter, twice the size of the model. At scale that redundancy is what runs out of memory, and ZeRO removes it in three stages. On a laptop the capstone does not need either, so this module is optional; it is how every large model you will serve was trained, and the capstone's `--world 4 --zero 2` run must match the single-process run to show you have it right.

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

**Your emulation's bookkeeping.** `memory_bytes()` reports what the rank holds after a step: the parameters' `.data` (in stage 3 between steps, only the chunk), the `.grad` arrays plus the chunk's gradient, and every array in the wrapped optimizer's `state_dict()` plus, in stages 1 and 2, the chunk's master copy (in stage 3 the chunk is the parameter storage, counted once). Stage 1 keeps the averaged full gradients in `.grad` until `zero_grad()`, as a training loop that logs gradient norms would.

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

Rank 0's stage 3 total is 96 bytes against 320 for one process: about $1/p$, up to the unequal chunks. These are `test_hand_example_zero_step` and `test_hand_example_memory_by_stage`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_zero_step` | unit | section 3's step, every stage, every rank | you and the test agree on the chunking |
| `test_hand_example_memory_by_stage` | unit | section 3's byte table | the stage definitions |
| `test_ddp_matches_single_process` | differential | 2 and 4 ranks, 5 momentum-SGD steps, $10^{-10}$, bitwise-equal gradients | DDP is just a bigger batch |
| `test_ddp_broadcasts_rank0_weights` | unit | ranks initialized differently end equal to rank 0's run | replicas never drift |
| `test_zero_matches_single_process` | differential | stages 1 to 3 on 2 and 4 ranks, 4 AdamW steps with decay | sharding never changes training |
| `test_zero_memory_is_one_over_p` | property | a 44-parameter model on 4 ranks: exact shares per stage | the reason to use ZeRO |
| `test_zero_rejects_bad_stage` | boundary | stage 0 or 4, no parameters | configuration errors surface early |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. not broadcasting the initial weights | replicas with different seeds never agree | `test_ddp_broadcasts_rank0_weights` (mutant `s01`) |
| 2. summing gradients instead of averaging | the learning rate is effectively $p$ times larger | `test_ddp_matches_single_process` (mutant `s02`) |
| 3. synchronizing only some parameters | replicas silently drift apart | `test_ddp_matches_single_process` (mutant `s03`) |
| 4. a rank starting from the wrong chunk | its master copy belongs to its neighbour | `test_hand_example_zero_step` (mutant `s04`) |
| 5. reduce-scatter without dividing by $p$ | stage 2 and 3 steps on the summed gradient | `test_zero_matches_single_process` (mutant `s05`) |
| 6. no all-gather after the update | other ranks' chunks go stale in your model | `test_hand_example_zero_step` (mutant `s06`) |
| 7. stage 2 keeping the full gradients | no gradient memory saved | `test_hand_example_memory_by_stage` (mutant `s07`) |
| 8. stage 3 keeping the full parameters | no parameter memory saved | `test_zero_memory_is_one_over_p` (mutant `s08`) |
| 9. unflattening in the wrong order | parameters swapped between tensors | `test_zero_matches_single_process` (mutant `s09`) |
| 10. dropping the optimizer's hyperparameters | the default learning rate and decay instead of yours | `test_zero_matches_single_process` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L11.2` | `broadcast`, `all_reduce`, `reduce_scatter`, `all_gather`, and `chunk_bounds` |
| Back | `L0.1` | parameters are Tensors; `.grad` is what gets reduced |
| Back | `L0.4` | `DDP` is a Module wrapping yours; `parameters()` fixes the flat order |
| Back | `M10.2` | SGD in the DDP tests |
| Back | `M10.3` | AdamW wrapped by `ZeroOptimizer` |
| Forward | `C1` | (optional) `--world 4 --zero 2` trains the capstone on 4 local processes and must match the single-process loss |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `DDP` | `torch.nn.parallel.DistributedDataParallel` | gradient buckets all-reduced during backward (overlapping communication with compute), unused-parameter detection | `torch/nn/parallel/distributed.py`, `torch/csrc/distributed/c10d/reducer.cpp` |
| `ZeroOptimizer` stages 1 and 2 | DeepSpeed ZeRO | bucketed reduce-scatter during backward, CPU and NVMe offload (ZeRO-Offload, ZeRO-Infinity) | `deepspeed/runtime/zero/stage_1_and_2.py` |
| stage 3 | PyTorch FSDP2 | per-layer all-gather just before use, prefetching the next layer, mixed-precision shards | `torch/distributed/fsdp/` |
| clipping | sharded global-norm clipping | an all-reduce of squared chunk norms before the step | `FullyShardedDataParallel.clip_grad_norm_` in `torch/distributed/fsdp/` |
