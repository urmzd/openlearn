# PyTorch and JAX execution

This part of [Training and Frameworks](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Relate eager, compiled, and transformed execution to the training loop.

## PyTorch

PyTorch is the default framework for both research and production, and the reason is debuggability. It uses **eager execution with dynamic graphs**: each operation runs immediately when Python reaches it, control flow is ordinary Python, and autograd records a tape during every forward pass. When something goes wrong you can set a breakpoint and inspect a tensor, which is exactly what you cannot easily do inside a compiled graph.

The model itself is a tree of `nn.Module` objects. Calling `loss.backward()` walks the autograd tape in reverse to compute gradients, and an optimizer such as `AdamW` then steps the parameters. Speed comes from **`torch.compile`** (TorchDynamo plus Inductor), which traces the eager program, captures graphs, and JIT-compiles fused kernels. That gives you eager ergonomics while you develop and much of the speed of a static graph when you run.

For scale, PyTorch provides `DistributedDataParallel` (DDP) and `FullyShardedDataParallel` (FSDP), built on the `torch.distributed` collectives (all-reduce, all-gather, reduce-scatter) that you will meet again under distributed training below. Around the core sits the largest ecosystem in the field: TorchData, TorchEval, the HuggingFace stack, and `torchao` for quantization. That ecosystem is a gravity well; most new models, papers, and vendor integrations arrive in PyTorch first.

## JAX

JAX takes the opposite starting point. It is functional and compiler-first: you write **pure functions** and apply **transformations** to them. `jit` compiles, `grad` differentiates, `vmap` batches automatically, and `pmap` or `shard_map` parallelize. The transformations compose, so `jit(vmap(grad(f)))` is a single fused, batched, differentiated, compiled function. That composability is JAX's central advantage.

The price is **functional state**. There is no in-place mutation; parameters are pytrees passed explicitly into and out of every function. Optimizers live in libraries such as Optax, and models in Flax, Equinox, or Haiku. JAX was built for the XLA compiler, so it is first-class on TPU and strong on GPU, and its mental model is "stage a whole computation, then compile it." For large models, `pjit` and the sharding API let you describe device meshes and tensor sharding declaratively; the compiler inserts the collectives. This is how large models train on TPU pods.

The two frameworks differ along a handful of axes, summarized here:

| | PyTorch | JAX |
|---|---------|-----|
| Style | Imperative, eager | Functional, staged |
| Debugging | Native Python, step-through | Trace abstractions; print inside `jit` is tricky |
| Parallelism | DDP/FSDP, explicit | `pmap`/`pjit`/sharding, compiler-inserted |
| Hardware | GPU-first, TPU via XLA | TPU-first, strong GPU |
| Ecosystem | Largest (HF, vendors) | Smaller, research-heavy (Flax, Optax) |

In practice they are converging: both increasingly lower to XLA-style compilers, and the eager-for-development, compiled-for-production split appears on both sides. Pick PyTorch when ecosystem and debuggability on GPUs matter most; pick JAX when you are on TPUs or want `vmap` and `pmap` composability.

## Summary

Relate eager, compiled, and transformed execution to the training loop. State the assumptions before applying the result to a different workload.

## Review questions

1. Why separate compilation time from steady-state throughput?

   <details><summary>Answer</summary>

   Compilation can dominate short jobs while disappearing in long runs; both costs belong in the decision.

   </details>

[Chapter contents](README.md) · [Next: Distributed and serving boundaries](02-distributed-and-serving-boundaries.md)
