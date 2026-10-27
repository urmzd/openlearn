# Parameter-efficient fine-tuning

This part of [Choosing Training and Adaptation Methods](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Explain the memory and computation saved by freezing weights.

## Parameter-efficient fine-tuning

Every post-training method above can be run on a full copy of the weights, but the 16-bytes-per-parameter rule makes that expensive. Parameter-efficient fine-tuning freezes the base and trains a small number of new parameters instead. The key papers are [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), and [DoRA](https://arxiv.org/abs/2402.09353).

### LoRA

LoRA freezes a weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and learns a low-rank update:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r},\ r \ll \min(d, k)$$

$A$ is initialized Gaussian (or Kaiming) and $B$ is initialized to zero, so $\Delta W = 0$ and training starts exactly at the base model. Each adapted matrix has $r(d + k)$ trainable parameters instead of $dk$.

The hypothesis behind LoRA is that the fine-tuning update has low intrinsic rank. That makes it best for teaching format, style, or a narrow skill, and worst for injecting large amounts of new knowledge. The $\alpha/r$ scaling keeps the update's magnitude roughly constant as you change $r$, so a learning rate transfers across ranks; **rsLoRA** uses $\alpha/\sqrt{r}$ instead ([2312.03732](https://arxiv.org/abs/2312.03732)) and does better at high rank.

Which modules to adapt matters more than the original paper suggested. Apply LoRA to all linear layers (`q`, `k`, `v`, `o`, `gate`, `up`, `down`), not just `q` and `v`. QLoRA showed this is needed to match full fine-tuning, and Thinking Machines' [*LoRA Without Regret*](https://thinkingmachines.ai/blog/lora/) (free) found that LoRA on all layers matches full fine-tuning on small-to-medium datasets with about 10 times the full-FT learning rate, and is notably efficient for RL, where each episode carries few bits of information.

For zero-latency serving you can **merge** the adapter, $W = W_0 + (\alpha/r)BA$, or keep adapters separate for multi-LoRA serving. On forgetting, [*LoRA Learns Less and Forgets Less*](https://arxiv.org/abs/2405.09673) (free) found lower target-domain gains than full fine-tuning on code and math continued pretraining, but better retention of base capabilities. [`lora_from_scratch.py`](code/lora_from_scratch.py) implements LoRA on a frozen two-layer MLP with a merge check and a finite-difference gradient check.

### QLoRA

QLoRA stores the base weights in 4 bits, keeps the adapters in BF16, and lets gradients flow through the dequantized base. Its 4-bit format, **NF4**, places 16 levels at the quantiles of $\mathcal{N}(0,1)$, which is information-theoretically optimal for normally distributed weights. Blocks of 64 weights each carry an FP32 absmax scale, an overhead of 0.5 bits per parameter. **Double quantization** then quantizes those absmax constants to FP8 in blocks of 256, cutting the overhead to $8/64 + 32/(64 \cdot 256) \approx 0.127$ bits per parameter, for a total of about **4.127 bits, or 0.516 bytes, per parameter**. **Paged optimizers** keep optimizer states in NVIDIA unified memory and page them to CPU during memory spikes from long sequences.

The compute dtype is still BF16, so QLoRA saves memory, not FLOPs, and runs slower than LoRA because of the dequantization. The paper's headline result was a 65B fine-tune on one 48 GB GPU. The quantization details are in [Quantization](../20-inference-engines/01-quantization/README.md).

### DoRA

DoRA decomposes each weight into a magnitude vector and a column-normalized direction, $W = m \cdot V/\|V\|_c$. It trains $m$ directly and updates the direction with LoRA:

$$W' = m \cdot \frac{W_0 + BA}{\|W_0 + BA\|_c}$$

This brings learning dynamics closer to full fine-tuning at small rank, for a small compute overhead, and it merges just like LoRA.

### Why LoRA fits on one GPU

Now the payoff. The table below comes from [`memory_calc.py`](code/memory_calc.py) for Llama 3 shapes, rank 16 on all linear layers, sequence length 4096, micro-batch 1, with activation checkpointing:

| Model | Method | Trainable | States | + activations | Min H100-80GB |
|-------|--------|-----------|--------|---------------|---------------|
| 8B | full FT (16 B/param) | 8.03B | 128.5 GB | 132 GB | 2 (FSDP), 8 in practice for throughput |
| 8B | LoRA (2 B/param base) | 41.9M (0.5%) | 16.7 GB | 20.5 GB | 1 (fits a 24 GB card) |
| 8B | QLoRA (0.516 B/param) | 41.9M | 4.8 GB | 8.6 GB | 1 |
| 70B | full FT | 70.6B | 1,129 GB | 1,138 GB | 18, i.e. 3 nodes |
| 70B | LoRA | 207M (0.3%) | 144.4 GB | 153 GB | 3 (shard the frozen base) |
| 70B | QLoRA | 207M | 39.7 GB | 48.3 GB | 1 |

The mechanism is simple. Frozen parameters need no gradient, no master copy, and no Adam moments, so 14 of their 16 bytes vanish, and QLoRA removes most of the remaining 2. Compute drops too: with no weight gradients for frozen matrices, the backward pass costs about $2N$ instead of $4N$, so LoRA training costs about **$4ND$** instead of $6ND$. Training compute for a typical customer job is single-digit GPU-hours at most. The real cost is data and evaluation.

### Serving many adapters

Adapters are megabytes, so one base model can serve hundreds of fine-tunes. The server batches requests for different adapters together and applies each adapter's $BA$ with segmented gather matmul kernels ([Punica](https://arxiv.org/abs/2310.18547), [S-LoRA](https://arxiv.org/abs/2311.03285)). vLLM (`--enable-lora`), SGLang, and LoRAX implement it, and it is the economic basis of per-customer fine-tunes at inference clouds: a LoRA deployment is billed like the base model. The serving side is covered in [Serving and Load](../20-inference-engines/04-serving-and-load/README.md).

## Summary

Explain the memory and computation saved by freezing weights. State the assumptions before applying the result to a different workload.

## Review questions

1. Does freezing the base remove its forward and backward compute?

   <details><summary>Answer</summary>

   The forward still runs and gradients must usually propagate through frozen layers to adapters. Only base weight-gradient computation and associated optimizer state disappear.

   </details>

[Chapter contents](README.md) · [Next: Evaluation and fine-tune scope](06-evaluation-and-fine-tune-scope.md)
