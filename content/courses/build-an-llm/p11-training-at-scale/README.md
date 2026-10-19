# Part 11: Training at Scale

What it takes to train the capstone on a laptop, and what changes on a cluster. The core module adds bf16 mixed precision (emulated), fp16 loss scaling, gradient accumulation, and activation checkpointing. Collectives (ring all-reduce over shared memory) and data parallelism with ZeRO stages are optional, with tensor and pipeline parallelism as side quests.

**Course passes**: L11.1 belongs to [Pass 9](../paths/course-p09-capstone-training/milestone.md); L11.2 and L11.3 are optional

**Before you start**: the just-in-time math `M08.4` (Hessian-vector products and checkpoint schedules); bf16 emulation from [Numerical Methods and Floating Point](../../mathematics-for-computing/09-numerical-methods-and-floating-point/); the optimizers of [Optimization](../../mathematics-for-computing/10-optimization/).

## Key ideas

- **Mixed precision** keeps a float32 master copy of the weights and computes in bf16; fp16 needs a loss scale so small gradients do not underflow.
- **Accumulation** sums gradients over micro-batches so a large effective batch fits in memory, with the loss scaled by the number of micro-batches.
- **Checkpointing** stores activations at segment boundaries and recomputes the rest during backward, trading compute for memory.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L11.1` | bf16 mixed precision, fp16 loss scaling, gradient accumulation, activation checkpointing | build | 9 |
| `L11.2` | Collectives over processes (ring all-reduce over pipes) | side | 9, optional |
| `L11.3` | DDP and ZeRO-1/2/3 | side | 9, optional |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L11.1` | [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](01-mixed-precision-accumulation-and-checkpointing.md) | build | 9 |
| 2 | `L11.2` | [Collectives over processes: ring all-reduce](02-collectives-over-processes.md) | side | 9 |
| 3 | `L11.3` | [Data parallelism and ZeRO stages 1 to 3](03-ddp-and-zero.md) | side | 9 |

## Going further

- Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740); Chen et al., [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174).
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054); the [Training and Post-Training](../../machine-learning/07-training-and-post-training/) topic for production practice.
