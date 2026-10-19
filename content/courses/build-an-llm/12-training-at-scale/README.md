# Training at Scale

By this point in the book you have every piece of a modern language model and a training loop that can fit it. What you do not yet have is a way to train it at the size the [capstone](../capstones/01-tinystories.md) asks for. The foundations loop runs one float32 batch per step and keeps every activation until backward finishes, and at 10.4M parameters and a 512-token context that stops fitting in memory. This chapter is about what it takes to train the capstone on a laptop, and about what changes when the same run moves to a cluster.

The arc has two halves. The first section is the one the capstone depends on: it adds bf16 mixed precision (emulated exactly on numpy), fp16 loss scaling, gradient accumulation, and activation checkpointing, each behind one call, and shows that together they train to within 2% of the float32 loss at a lower peak memory. The remaining two sections are optional and look past the laptop. They build collectives between processes, with the bandwidth-optimal ring all-reduce, and then data parallelism and the three ZeRO stages on top of them. Tensor and pipeline parallelism are left for further reading.

Three ideas run through the whole chapter. **Mixed precision** keeps a float32 master copy of the weights and computes in bf16; fp16 needs a loss scale so that small gradients do not underflow. **Accumulation** sums gradients over micro-batches so that a large effective batch fits in memory, with each micro-batch's loss weighted by its share of the rows; data parallelism is the same idea spread over processes. **Checkpointing** stores activations at segment boundaries and recomputes the rest during backward, trading compute for memory.

> **Learning objectives**
> - Train with emulated bf16 or fp16 matmuls while keeping float32 master weights, and explain why that converges.
> - Derive and implement dynamic loss scaling, exact gradient accumulation, and bit-exact activation checkpointing.
> - Implement ring reduce-scatter, all-gather, and all-reduce over processes, and count their bytes.
> - Build data parallelism and ZeRO stages 1 to 3, and compute per-rank memory for each.

## Before you begin

You should be comfortable with the Tensor and training loop of the [Foundations](../01-foundations/README.md) chapter. From *Mathematics for Computing*, the chapter leans on the bf16 and fp16 emulation in [Numerical Methods and Floating Point](../../mathematics-for-computing/10-numerical-methods-and-floating-point/README.md), the optimizers and gradient clipping in [Optimization](../../mathematics-for-computing/11-optimization/README.md), and the checkpoint schedule in [Hessian-vector products and the recompute-versus-memory schedule](../../mathematics-for-computing/09-matrix-calculus-and-autodiff/04-hessian-vector-products-and-recompute.md).

## In this chapter

1. [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](01-mixed-precision-accumulation-and-checkpointing.md): the four tools every large training run uses, built from first principles and checked against float32, as used by the capstone's `--bf16 --accum --checkpoint-activations` flags.
2. [Collectives over processes: ring all-reduce](02-collectives-over-processes.md) (optional): point-to-point messages between processes, the ring reduce-scatter and all-gather, why the ring's traffic per link does not grow with the number of ranks, and how to avoid deadlock.
3. [Data parallelism and ZeRO stages 1 to 3](03-ddp-and-zero.md) (optional): data parallelism as accumulation across processes, and sharding the optimizer state, gradients, and parameters so that per-rank memory falls toward $1/p$.

## Further reading

- Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740); Chen et al., [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174).
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054); the [Training and Post-Training](../../machine-learning/07-training-and-post-training/README.md) chapter of *Machine Learning and AI* for production practice.
