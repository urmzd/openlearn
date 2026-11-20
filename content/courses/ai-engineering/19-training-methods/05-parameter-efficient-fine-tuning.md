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

### What trains and what does not

Frozen means "not updated". It does not mean "skipped". The backward pass still runs through the whole network, frozen layers included, because that is the only route for the error signal from the loss at the output to an adapter in an early block.

| | Frozen $W_0$ | Adapter $A$, $B$ |
|---|---|---|
| The error signal passes through it | Yes | Yes |
| A gradient for its own values is computed | No | Yes |
| The optimizer updates it | No | Yes |

The base weights never improve. The **effective** weight does, because the layer behaves as $W_0 + (\alpha/r)BA$. With $g = \partial L / \partial h$ at an adapted layer, the chain rule gives each matrix its direction:

$$\frac{\partial L}{\partial B} = \frac{\alpha}{r}\, g\,(Ax)^\top, \qquad \frac{\partial L}{\partial A} = \frac{\alpha}{r}\, B^\top g\, x^\top.$$

The second formula contains $B$, which starts at zero. On the first step every $A$ receives a zero gradient and only $B$ moves. $A$ starts to learn on the second step.

The loss is one scalar at the output, as in any training: cross-entropy for supervised fine-tuning, and [reward-weighted log probabilities](../04-reinforcement-learning/03-policy-gradients-and-actor-critic.md) for RL. The adapter mechanics are the same in both.

Whether a layer trains depends on whether it has an adapter:

- **A module without an adapter is not trained at all.** LoRA is opt-in for each module. Untargeted projections, norms, embeddings, and the output head stay exactly as they are in the base model.
- **A module with an adapter always trains.** Every $A$ and $B$ receives a gradient on every step after the first.

Trained does not mean changed much. Where a layer matters for the task, $BA$ grows into a real correction. Where it does not, the gradient is small, $B$ stays near zero, and the layer behaves almost like the base. [AdaLoRA](https://arxiv.org/abs/2303.10512) uses this unevenness: it scores the adapters and moves rank from the idle ones to the useful ones.

### One pattern, one adapter for each layer

No formal standard defines an adapter. The Hugging Face PEFT format is the de facto one: an `adapter_config.json` that holds the definition and an `adapter_model.safetensors` that holds only the $A$ and $B$ tensors. [Downloads, loaders, and sharding](../20-inference-engines/03-model-loading/02-downloads-loaders-and-sharding.md) covers the files and how a server loads them.

The config is a pattern, and the weight file is the definition for each layer. `"target_modules": ["q_proj"]` matches by name suffix, so it selects the `q_proj` in every block. A 32-block model gets 32 adapters from that one entry. Each has its own $A$ and $B$, and nothing is shared between them:

```text
layers.0.self_attn.q_proj.lora_A.weight     shape (r, in_features)
layers.0.self_attn.q_proj.lora_B.weight     shape (out_features, r)
layers.1.self_attn.q_proj.lora_A.weight
...
layers.31.self_attn.q_proj.lora_B.weight
```

The default is uniform: the same rank and the same $\alpha$ on every match. The [PEFT `LoraConfig`](https://huggingface.co/docs/peft/package_reference/lora) has finer controls:

| Field | Effect |
|---|---|
| `layers_to_transform` | Adapt only the listed block indices |
| `target_modules` as one string | Treat the string as a regular expression over the full module path |
| `rank_pattern` | Override the rank for matching layers |
| `alpha_pattern` | Override $\alpha$ for matching layers |
| `exclude_modules` | Remove matches from the selection |

An adapter fits only the exact base model it was trained on. The architecture and the base weights must both match, or the correction is meaningless.

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

Adapters are megabytes, so one base model can serve hundreds of fine-tunes. The server batches requests for different adapters together and applies each adapter's $BA$ with segmented gather matmul kernels ([Punica](https://arxiv.org/abs/2310.18547), [S-LoRA](https://arxiv.org/abs/2311.03285)). vLLM (`--enable-lora`), SGLang, and LoRAX implement it, and it is the economic basis of per-customer fine-tunes at inference clouds: a LoRA deployment is billed like the base model.

The merged and unmerged paths are not numerically identical. $W_0x + (\alpha/r)BAx$ computed in two parts does not round the same way as $(W_0 + (\alpha/r)BA)x$ computed once. If RL rollouts come from one path and the trainer scores tokens on the other, that difference is one more source of [trainer and inference engine drift](../20-inference-engines/13-batch-invariance-and-reproducibility.md). The serving side is covered in [Serving and Load](../20-inference-engines/04-serving-and-load/README.md).

## Further reading

All of these are free on arXiv.

| Topic | Paper | Read it for |
|---|---|---|
| The method | Hu et al., [LoRA](https://arxiv.org/abs/2106.09685), 2021 | The low-rank update, the zero start, and which modules to adapt |
| Why low rank works | Aghajanyan et al., [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255), 2020 | The evidence that a fine-tune needs few free directions |
| Rank allocation | Zhang et al., [AdaLoRA](https://arxiv.org/abs/2303.10512), 2023 | Moving rank from idle adapters to useful ones |
| Initialization | Meng et al., [PiSSA](https://arxiv.org/abs/2404.02948), 2024 | Starting the adapter from the top singular directions of the weight |
| Learning rates | Hayou et al., [LoRA+](https://arxiv.org/abs/2402.12354), 2024 | Separate learning rates for $A$ and $B$ |
| Fewer parameters | Kopiczko et al., [VeRA](https://arxiv.org/abs/2310.11454), 2023 | Shared random matrices with small trained scaling vectors |
| Capacity and forgetting | Biderman et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673), 2024 | Measured gaps to full fine-tuning and retention of base ability |
| Multi-LoRA serving | Chen et al., [Punica](https://arxiv.org/abs/2310.18547), 2023 | Batching requests for different adapters on one base model |
| Multi-LoRA serving cost | Sheng et al., [S-LoRA](https://arxiv.org/abs/2311.03285), 2023 | Adapter memory management and the throughput cost of serving thousands of adapters |
| The engine underneath | Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180), 2023 | The cache and batching design that multi-adapter servers build on |

## Summary

Explain the memory and computation saved by freezing weights. State the assumptions before applying the result to a different workload.

## Review questions

1. Does freezing the base remove its forward and backward compute?

   <details><summary>Answer</summary>

   The forward still runs and gradients must usually propagate through frozen layers to adapters. Only base weight-gradient computation and associated optimizer state disappear.

   </details>

2. The base weights are frozen. How does the model's behavior improve?

   <details><summary>Answer</summary>

   The layer computes with the effective weight $W_0 + (\alpha/r)BA$. Backpropagation gives gradients for $A$ and $B$, the optimizer moves them, and the product grows into a learned correction on top of an unchanged $W_0$.

   </details>

3. On the first training step, which adapter matrices change, and why?

   <details><summary>Answer</summary>

   Only $B$. The gradient for $A$ is proportional to $B^\top$, and $B$ starts at zero, so every $A$ receives a zero gradient until $B$ has moved.

   </details>

4. A config sets `"target_modules": ["q_proj", "v_proj"]` on a 32-block model. How many adapters exist, and do they share weights?

   <details><summary>Answer</summary>

   Sixty-four: one for each matching module in each block. Each adapter has its own $A$ and $B$. The config is a pattern, and the keys in the weight file identify each layer's tensors.

   </details>

5. Are any layers left untrained in a LoRA fine-tune?

   <details><summary>Answer</summary>

   Yes. Every module without an adapter keeps its base values. Modules with an adapter all receive gradients, but those that matter little for the task stay close to a zero correction.

   </details>

6. Why can a merged adapter and an unmerged adapter give different logits for the same input?

   <details><summary>Answer</summary>

   The two paths group the floating-point operations differently, so they round differently. The functions are equal in exact arithmetic and only close in finite precision.

   </details>

[Chapter contents](README.md) · [Next: Evaluation and fine-tune scope](06-evaluation-and-fine-tune-scope.md)
