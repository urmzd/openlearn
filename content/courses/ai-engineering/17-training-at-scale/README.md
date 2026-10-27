# Training at Scale

A model and a correct training loop still need a memory, compute, and recovery plan. The foundations loop uses float32 and retains intermediate activations; a larger batch, longer context, or larger model can make that state dominate memory. This chapter develops the precision, accumulation, recomputation, communication, and checkpoint mechanisms needed to move the [capstone](../capstones/01-tinystories.md) from a local experiment to a larger training system.

The first section supplies the capstone's precision, accumulation, and recomputation techniques. The next two build process collectives, data parallelism, and ZeRO from small implementations. The final parts connect those mechanisms to tensor, pipeline, context, and expert layouts, then to profiling, numerical failures, checkpointing, and recovery. Implementation checks and production measurements have different scopes: use the small checks to understand the algorithm, then validate memory, quality, and throughput on the actual training stack.

Three ideas run through the whole chapter. **Mixed precision** keeps a float32 master copy of the weights and computes in bf16; fp16 needs a loss scale so that small gradients do not underflow. **Accumulation** sums gradients over micro-batches so that a large effective batch fits in memory, with each micro-batch's loss weighted by its share of the objective's units, such as valid target tokens; data parallelism is the same idea spread over processes. **Checkpointing** stores activations at segment boundaries and recomputes the rest during backward, trading compute for memory.

> **Learning objectives**
> - Train with emulated bf16 or fp16 matmuls while keeping float32 master weights, and explain why that converges.
> - Derive and implement dynamic loss scaling, exact gradient accumulation, and bit-exact activation checkpointing.
> - Implement ring reduce-scatter, all-gather, and all-reduce over processes, and count their bytes.
> - Build data parallelism and ZeRO stages 1 to 3, and compute per-rank memory for each.

## Before you begin

You should be comfortable with the Tensor and training loop of the [Foundations](../06-numerics-and-autograd/README.md) chapter. From *Mathematics*, the chapter leans on the bf16 and fp16 emulation in [Numerical Methods and Floating Point](../../mathematics/15-numerical-methods-and-floating-point/README.md), the optimizers and gradient clipping in [Optimization](../../mathematics/16-optimization/README.md), and the checkpoint schedule in [Hessian-vector products and the recompute-versus-memory schedule](../../mathematics/14-matrix-calculus-and-autodiff/04-hessian-vector-products-and-recompute.md).

## In this chapter

1. [Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](01-mixed-precision-accumulation-and-checkpointing.md): the four tools every large training run uses, built from first principles and checked against float32, as used by the capstone's `--bf16 --accum --checkpoint-activations` flags.
2. [Collectives over processes: ring all-reduce](02-collectives-over-processes.md) (optional): point-to-point messages between processes, the ring reduce-scatter and all-gather, why the ring's traffic per link does not grow with the number of ranks, and how to avoid deadlock.
3. [Data parallelism and ZeRO stages 1 to 3](03-ddp-and-zero.md) (optional): data parallelism as accumulation across processes, and sharding the optimizer state, gradients, and parameters so that per-rank memory falls toward $1/p$.
4. [Parallelism, topology, and the global batch](04-parallelism-and-topology.md): choose the dimension to split, count independent replicas, and validate distributed equivalence.
5. [Training failures, profiling, and recovery](05-training-failures-and-recovery.md): diagnose data/numerical/optimization failures, save complete state, and test interrupted-run continuation.

## Further reading

- Micikevicius et al., [*Mixed Precision Training*](https://arxiv.org/abs/1710.03740); Chen et al., [*Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/abs/1604.06174).
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054); the [Training and Post-Training](../19-training-methods/README.md) chapter of *AI Engineering* for production practice.
