# Memory, precision, and parallelism

This part of [Choosing Training and Adaptation Methods](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Account for model states and select the dimension to shard.

## Memory and parallelism

Compute tells you how long a run takes; memory tells you how many GPUs it needs before it can start. This section is the arithmetic and the current APIs, drawing on [ZeRO](https://arxiv.org/abs/1910.02054), [Megatron-LM](https://arxiv.org/abs/1909.08053), [Narayanan et al.](https://arxiv.org/abs/2104.04473), [Korthikanti et al.](https://arxiv.org/abs/2205.05198), and the [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277).

### Sixteen bytes per parameter

A common mixed-precision AdamW accounting model ([Micikevicius et al.](https://arxiv.org/abs/1710.03740)) keeps the copies below. Actual frameworks can choose different gradient, master-weight, and optimizer-state representations, so inspect the installed implementation:

| Item | Bytes/param |
|------|-------------|
| BF16 weights (used in forward/backward) | 2 |
| BF16 gradients | 2 (4 if accumulated in FP32) |
| FP32 master weights | 4 |
| Adam first moment $m$ (FP32) | 4 |
| Adam second moment $v$ (FP32) | 4 |
| **Total** | **16** (18 with FP32 grads) |

So an 8B model needs about 128 GB of state and a 70B about 1.1 TB, **before any activation**. That single fact is why full fine-tuning of a 70B is a multi-node job, and why LoRA, later in this chapter, exists.

Activations come on top. A simplified transformer-layer model in BF16 with memory-efficient attention estimates roughly $34 \cdot s \cdot b \cdot d$ bytes ([Korthikanti et al.](https://arxiv.org/abs/2205.05198)) for sequence length $s$, micro-batch $b$, and width $d$. For an 8B at $s = 4096$ and $b = 1$ that is about 0.57 GB per layer, or about 18 GB across 32 layers. **Activation checkpointing** stores only each layer's input ($2 \cdot s \cdot b \cdot d$ bytes) and recomputes the rest during the backward pass, buying an order of magnitude less activation memory for about 33% more compute.

> **Warning:** Do not forget the logits. They take $s \cdot V \cdot 4$ bytes in FP32, which is 2.1 GB per 4K-token sequence at a vocabulary of $V = 128\text{K}$. Fused or chunked cross entropy (Liger, Cut Cross-Entropy) avoids materializing them at all.

### Sharding the states: ZeRO and FSDP2

The first and cheapest way to fit a model is to stop replicating its states on every data-parallel rank. ZeRO does this in three stages, each sharding one more kind of state across the $N_d$ ranks. The table compares them for a model of $\Psi$ parameters:

| Stage | Shards across $N_d$ data-parallel ranks | Memory per GPU | Comms vs DDP |
|-------|----------------------------------------|----------------|--------------|
| DDP | nothing | $16\Psi$ | all-reduce grads ($2\Psi$) |
| **ZeRO-1** | optimizer states | $4\Psi + 12\Psi/N_d$ | same |
| **ZeRO-2** | + gradients | $2\Psi + 14\Psi/N_d$ | same (reduce-scatter instead of all-reduce) |
| **ZeRO-3 / FSDP** | + parameters | $16\Psi/N_d$ | 1.5x (all-gather params in forward and backward) |

These communication comparisons use an idealized bandwidth-volume model with matched dtypes and a particular gather/reduce schedule. Similar total bytes do not mean communication is free: launches, group size, overlap, offload, and topology affect wall time. ZeRO-3 can have extra parameter gathers, with resharding policy changing the schedule.

One concrete implementation is PyTorch **FSDP2**, documented in the [2.9 `fully_shard` reference](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html). It uses per-parameter **DTensor** sharding rather than FSDP1's flat-parameter wrapper. Sharding groups and precision policies must be chosen for the model and topology. A common pattern applies sharding block by block, then to the root:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:          # shard each block: its all-gather overlaps the previous block's compute
    fully_shard(block, mp_policy=mp)
fully_shard(model, mp_policy=mp)    # then the root (embeddings, head)
model = torch.compile(model)        # Dynamo + Inductor: fused kernels, same eager semantics
```

Sharding per block matters: each block's all-gather can overlap the previous block's compute, so the communication hides behind useful work.

### Splitting the model itself

Sharding states only helps until a single layer's working set no longer fits, or until the batch stops being large enough to keep data parallelism efficient. Then you split the model. The useful way to think about this is that parallelism is a choice of *what to split*: the optimizer state (ZeRO/FSDP), each matrix (TP), the layers (PP), the sequence (CP), or the experts (EP). Name the dimension that does not fit, then pick the tool. The Megatron-LM family of axes compares as follows:

| Axis | Splits | Communication | Placement |
|------|--------|---------------|-----------|
| **TP** (tensor) | each matmul: column-parallel then row-parallel, so an MLP needs one all-reduce forward and one backward | 2 all-reduces per layer per pass, on the critical path | inside a fast interconnect domain where feasible |
| **SP** (sequence, with TP) | the LayerNorm/dropout regions along the sequence | turns all-reduce into reduce-scatter + all-gather, same volume, less activation memory | with TP |
| **PP** (pipeline) | layers into stages; micro-batches fill the pipe (1F1B, interleaved, zero-bubble) | point-to-point activations | across nodes; bubble ≈ $(p-1)/(m+p-1)$ |
| **CP** (context) | the sequence across GPUs for attention (ring attention: pass K/V blocks around a ring) | K/V exchange per layer | long-context (64K-1M) training |
| **EP** (expert) | MoE experts across GPUs | all-to-all token dispatch and combine | MoE models |

The pipeline bubble formula, with $p$ stages and $m$ micro-batches, explains why pipelines want many micro-batches. Tensor parallelism benefits from a fast low-latency interconnect because collectives sit on its critical path. The suitable degree depends on the model and topology, not a universal eight-GPU limit.

Combining axes gives **3D parallelism**, $\text{DP} \times \text{TP} \times \text{PP}$, and **4D or 5D** when CP and EP are added. The canonical layout puts TP inside the node, PP across nodes, and FSDP or DP over everything else. The reference implementations are **Megatron-LM / Megatron-Core** from NVIDIA; **torchtitan**, the clean PyTorch-native reference with FSDP2, TP, PP, CP, EP, and Float8/MXFP8; and **DeepSpeed**, the home of ZeRO and offload.

### The same ideas in JAX

JAX expresses all of the above as **sharding annotations** on arrays, and the XLA partitioner (GSPMD, now Shardy) inserts the collectives for you:

```python
import jax
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh((8, 4), ("data", "model"))              # 32 devices: FSDP/DP × TP
w = jax.device_put(w, NamedSharding(mesh, P("data", "model")))  # FSDP on rows, TP on cols
x = jax.device_put(x, NamedSharding(mesh, P("data", None)))     # batch sharded on "data"
step = jax.jit(train_step)                                       # compiler inserts all-gathers / reduce-scatters

# drop to per-device code with explicit collectives when the compiler's choice is wrong
@jax.shard_map(mesh=mesh, in_specs=(P("data", "model"), P("data", None)), out_specs=P("data", "model"))
def local_matmul(w_blk, x_blk): ...
```

A few API facts are worth knowing. `jax.shard_map` is now top-level, having graduated from `jax.experimental`; `pjit` has been merged into `jax.jit`, and `pmap` is legacy. Recent JAX defaults `make_mesh` to **Explicit** axis types, where shardings are part of each array's type; pass `axis_types=(AxisType.Auto, ...)` to get classic GSPMD propagation, as [`train_step_jax.py`](code/train_step_jax.py) does. **MaxText** is the reference JAX LLM trainer, on TPU and GPU, covering pretraining, SFT, and GRPO.

### Mixed precision: BF16 and FP8

**BF16** has FP32's 8-bit exponent, so loss scaling is usually unnecessary for its exponent range. It is common in training, but precision choices remain model-, operation-, and device-dependent. **FP8** goes further ([Micikevicius et al. 2022](https://arxiv.org/abs/2209.05433)): E4M3 for the forward pass and E5M2 or E4M3 for gradients doubles the H100 matmul peak to about 1979 TFLOP/s. With so few bits it needs **scaling factors**, either per-tensor delayed or current scaling, or **MXFP8** block scaling over 32-element blocks, which is native on Blackwell. DeepSeek-V3 trained in FP8 with fine-grained block scaling.

Two libraries supply FP8 training. In **NVIDIA Transformer Engine** you use `te.Linear` and friends under `te.autocast(recipe=...)`; in TE 2.x, `fp8_autocast` is deprecated in favor of the recipe-agnostic `autocast`. In **torchao**, `convert_to_float8_training(model)` takes a `"tensorwise"`, `"rowwise"`, or `"rowwise_with_gw_hp"` recipe and composes with FSDP2 (including FP8 all-gather) and `torch.compile`.

Note what FP8 does not change: master weights and optimizer states stay in FP32. FP8 changes the matmuls, not the 16 bytes.

## Summary

Account for model states and select the dimension to shard. State the assumptions before applying the result to a different workload.

## Review questions

1. Why can a model still run out of memory after its state fits?

   <details><summary>Answer</summary>

   Activations, logits, temporary collective buffers, workspaces, and allocator headroom also occupy memory.

   </details>

[Chapter contents](README.md) · [Next: Choosing post-training signals](04-choosing-post-training-signals.md)
