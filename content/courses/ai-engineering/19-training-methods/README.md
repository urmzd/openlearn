# Choosing Training and Adaptation Methods

Every model you serve was made by a training pipeline, and most of the decisions that determine its quality, its cost, and its behavior were taken there rather than at inference time. This chapter is about that pipeline: how a base model is pretrained, how its memory and compute are budgeted and split across many GPUs, and how post-training (supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards) turns a next-token predictor into an assistant or a specialist. It ends where most practitioners actually work: scoping a fine-tune for a real task, deciding whether one is needed at all, and estimating what it will cost.

The organizing idea is that every training job is three budgets. **FLOPs**, roughly $6ND$, are paid in GPU-hours. **Memory**, bytes per parameter plus activations, is paid in GPU count. **Data**, tokens of the right kind, is paid in people. Pretraining is FLOP-bound and data-hungry; post-training is data-bound and FLOP-cheap. Distributed training techniques trade memory for communication; parameter-efficient fine-tuning trades capacity for memory. If you can do the arithmetic for all three budgets in your head during a meeting, and you know which kind of data each post-training method consumes, you can scope any fine-tuning engagement and hold a credible conversation with the researchers who run the large runs.

The chapter is a single long reading, built up in the order the pipeline runs: the stages and the people who own them, the mathematics of pretraining, memory and parallelism, post-training, parameter-efficient fine-tuning, scoping, and finally the same training step written in three frameworks. A companion [code directory](code/README.md) holds small runnable files that implement the formulas and assert that they hold, so you can check every number in the chapter yourself.

> **Learning objectives**
> - Estimate the training FLOPs, GPU-hours, and dollar cost of a run from $N$, $D$, the hardware peak, and an assumed MFU, and explain why real runs cost about three times the ideal.
> - Account for the 16 bytes per parameter of mixed-precision AdamW, estimate activation memory, and choose among ZeRO/FSDP, tensor, pipeline, context, and expert parallelism by naming the dimension that does not fit.
> - Derive the DPO loss from the KL-constrained RLHF objective, and explain how GRPO removes the critic and why it needs prompts the model sometimes solves.
> - Explain why LoRA and QLoRA make fine-tuning a one-GPU problem, with the memory and FLOP arithmetic that shows it.
> - Scope a fine-tune eval-first: decide whether to fine-tune, pick the method that matches the available data, and estimate the cost.

## Before you begin

This chapter assumes [Deep Learning](../02-deep-learning/README.md) (backpropagation, Adam, transformers), the policy-gradient material in [Reinforcement Learning](../04-reinforcement-learning/README.md), and the lifecycle overview in [Foundation Models and Architectures](../05-foundation-models/README.md). The framework basics, PyTorch versus JAX, DDP, the FSDP concept, and `jit`, are covered in [Training and Frameworks](../16-training-frameworks/README.md) in *AI Engineering*, and this chapter does not repeat them. The objectives lean on cross entropy, KL divergence, and the Bradley-Terry likelihood, which you can review in [Information Theory](../../mathematics/17-information-theory/README.md) and [Probability and Statistics](../../mathematics/13-probability-statistics/README.md) in *Mathematics*.

The best single references are free. Stanford's [*CS336: Language Modeling from Scratch*](https://stanford-cs336.github.io/) has lectures and assignments that build a tokenizer, a transformer, parallel training, scaling laws, data filtering, SFT, and RL. Hugging Face's [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) is the best single document on DP, TP, PP, CP, EP, and ZeRO with measured numbers. The full list of papers and supplementary material is under "Further reading" at the end of the chapter.

## The pipeline and who does what

A modern model is made in a fixed sequence of stages, as written down in the [Llama 3 report](https://arxiv.org/abs/2407.21783) and taught in [CS336](https://stanford-cs336.github.io/):

```text
data → pretraining → mid-training → SFT → preference / RL → evals → release
```

The table below summarizes what each stage does, how big it is, and what learning signal it uses.

| Stage | What happens | Typical scale | Signal |
|-------|-------------|---------------|--------|
| **Data** | Crawl (Common Crawl), text extraction, language ID, **dedup** (exact hash + MinHash LSH near-dup), quality filtering (heuristics + classifier, e.g. FineWeb-Edu), PII and toxicity filters, **decontamination** against evals, tokenizer training (BPE), **mixture** weights across web/code/math/books/multilingual | 10s of trillions of raw tokens down to ~15T kept | none (curation) |
| **Pretraining** | Next-token cross entropy on the mixture, cosine or WSD learning-rate schedule | $10^{23}$ to $10^{25+}$ FLOPs | self-supervised |
| **Mid-training** | Anneal on high-quality data, up-weight code/math, **long-context extension** (raise RoPE base $\theta$, train on long documents, 8K to 128K+) | 1-10% of pretraining tokens | self-supervised |
| **SFT** | Instruction→response demonstrations in a chat template | $10^4$ to $10^6$ examples | supervised |
| **Preference / RL** | Reward model + PPO, or DPO on pairs, or GRPO on verifiable tasks; often several rounds | $10^4$ to $10^6$ prompts | preference / reward |
| **Evals** | Capability (MMLU-Pro, GPQA, SWE-bench, AIME), safety, regression vs previous checkpoint | every checkpoint | measurement |
| **Release** | Quantize, convert, write model card, ship weights and serving config | | |

Two of these rows deserve emphasis. First, **data is the source code**: most of the quality differences between open models trace to filtering and mixtures, not to architecture ([FineWeb](https://arxiv.org/abs/2406.17557), free). Second, **decontamination** is a correctness requirement, not a nicety. A benchmark that leaked into pretraining makes every later evaluation on it meaningless.

Different people own different stages, and it helps to know what each of them will ask you. The research scientist owns objectives, data recipes, ablations, and the post-training algorithm, and asks questions like "What is your KL coefficient and how did you pick it?" The ML engineer owns training and evaluation pipelines, data tooling, checkpoints, and reproducibility: "How do you version data and resume a failed run?" The member of technical staff on performance owns MFU, kernels (FlashAttention, fused optimizers), the parallelism layout, communication overlap, and FP8: "What MFU do you get on H100 for a 70B with TP=8?" And the forward deployed engineer owns a customer's adaptation, choosing among prompting, RAG, LoRA, and full fine-tuning, building their dataset and evaluation, running the job, and getting it served. Their question is the one the last section of this chapter answers: "Will fine-tuning fix this, how much data, what will it cost, and how do we know it worked?"

At a fine-tuning cloud, customers almost always start from an open base or instruct model, so the deployed engineer's pipeline is the last three rows of the table. But you cannot reason about those rows without the arithmetic of the first ones, so that is where we start.

## The mathematics of pretraining

The pretraining objective is the cross entropy between the data and the model, in nats per token:

$$L(\theta) = -\frac{1}{T}\sum_t \log p_\theta(x_{t+1} \mid x_{\le t})$$

Its exponential, $e^{L}$, is the perplexity. In code the whole objective is a shift by one: `inputs = x[:, :-1]` and `targets = x[:, 1:]`, as in [`train_step_torch.py`](code/train_step_torch.py).

### Why training costs 6ND

The number to carry everywhere is that training costs about $6ND$ FLOPs for a model with $N$ parameters trained on $D$ tokens. The forward pass costs $2N$ FLOPs per token, one multiply-add per parameter. The backward pass costs $4N$, because it computes two sets of gradients, with respect to the activations and with respect to the weights, at $2N$ each. Attention adds about $12 \cdot L \cdot d \cdot s$ FLOPs per token for $L$ layers of width $d$ at sequence length $s$, which stays small until the context gets long.

### Chinchilla, and why everyone overtrains anyway

Given a fixed compute budget $C = 6ND$, how should you split it between model size and data? [Kaplan et al.](https://arxiv.org/abs/2001.08361) gave the first answer; [Chinchilla](https://arxiv.org/abs/2203.15556) gave the one that stuck. Loss is minimized when $N$ and $D$ are scaled equally, $N_{\text{opt}} \propto C^{0.5}$, which works out to **$D \approx 20N$**. The fitted loss is

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \qquad \alpha \approx 0.34,\ \beta \approx 0.28.$$

Yet modern models are trained far past that point on purpose. Chinchilla minimizes *training* cost, but a model's lifetime cost is $6ND + 2N \cdot D_{\text{inference}}$, and for a popular model $D_{\text{inference}}$ dwarfs $D$ ([*Beyond Chinchilla-Optimal*](https://arxiv.org/abs/2401.00448)). A smaller model trained well past $20N$ reaches the same quality at a much lower serving cost. Llama 3 8B saw about 15T tokens, roughly 93 times its Chinchilla-optimal budget.

### Utilization: MFU and HFU

Hardware never runs at its peak. **Model FLOPs utilization** measures how close you get:

$$\text{MFU} = \frac{6N \cdot \text{tokens/sec}}{n_{\text{gpus}} \cdot \text{peak FLOP/s}}$$

Good large dense runs reach 35-50% BF16 MFU on H100; MoE and long-context runs are lower. **Hardware FLOPs utilization** (HFU) also counts FLOPs that are recomputed, so under activation checkpointing HFU is greater than MFU.

### Working it by hand: an 8B model on 15T tokens

These formulas are enough to cost a real pretraining run. Take the Llama 3 8B shape on H100s, as computed by [`memory_calc.py`](code/memory_calc.py):

```text
N = 8.03e9 (Llama 3 8B shape)       D = 15e12
C = 6 N D = 6 × 8.03e9 × 15e12      = 7.23e23 FLOPs
H100 dense BF16 peak                = 989 TFLOP/s
at 40% MFU                          = 3.96e14 FLOP/s per GPU
GPU-seconds = 7.23e23 / 3.96e14     = 1.83e9 s  →  ~507,000 GPU-hours
on 1,024 GPUs                       ≈ 20.6 days
at $2-3 per H100-hour               ≈ $1.0-1.5M of pure compute
```

Meta's Llama 3.1 model card reports about 1.46M H100-hours for the 8B. That roughly threefold gap to the ideal estimate is the lesson. Real runs pay for lower MFU at small model size, for restarts after hardware failures, for evaluations, and for the ablations that found the recipe in the first place.

## Memory and parallelism

Compute tells you how long a run takes; memory tells you how many GPUs it needs before it can start. This section is the arithmetic and the current APIs, drawing on [ZeRO](https://arxiv.org/abs/1910.02054), [Megatron-LM](https://arxiv.org/abs/1909.08053), [Narayanan et al.](https://arxiv.org/abs/2104.04473), [Korthikanti et al.](https://arxiv.org/abs/2205.05198), and the [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277).

### Sixteen bytes per parameter

Mixed-precision AdamW ([Micikevicius et al.](https://arxiv.org/abs/1710.03740)) keeps several copies of every parameter. The breakdown is:

| Item | Bytes/param |
|------|-------------|
| BF16 weights (used in forward/backward) | 2 |
| BF16 gradients | 2 (4 if accumulated in FP32) |
| FP32 master weights | 4 |
| Adam first moment $m$ (FP32) | 4 |
| Adam second moment $v$ (FP32) | 4 |
| **Total** | **16** (18 with FP32 grads) |

So an 8B model needs about 128 GB of state and a 70B about 1.1 TB, **before any activation**. That single fact is why full fine-tuning of a 70B is a multi-node job, and why LoRA, later in this chapter, exists.

Activations come on top. Per layer, in BF16, with FlashAttention so there is no $s^2$ term, they take about $34 \cdot s \cdot b \cdot d$ bytes ([Korthikanti et al.](https://arxiv.org/abs/2205.05198)) for sequence length $s$, micro-batch $b$, and width $d$. For an 8B at $s = 4096$ and $b = 1$ that is about 0.57 GB per layer, or about 18 GB across 32 layers. **Activation checkpointing** stores only each layer's input ($2 \cdot s \cdot b \cdot d$ bytes) and recomputes the rest during the backward pass, buying an order of magnitude less activation memory for about 33% more compute.

> **Warning:** Do not forget the logits. They take $s \cdot V \cdot 4$ bytes in FP32, which is 2.1 GB per 4K-token sequence at a vocabulary of $V = 128\text{K}$. Fused or chunked cross entropy (Liger, Cut Cross-Entropy) avoids materializing them at all.

### Sharding the states: ZeRO and FSDP2

The first and cheapest way to fit a model is to stop replicating its states on every data-parallel rank. ZeRO does this in three stages, each sharding one more kind of state across the $N_d$ ranks. The table compares them for a model of $\Psi$ parameters:

| Stage | Shards across $N_d$ data-parallel ranks | Memory per GPU | Comms vs DDP |
|-------|----------------------------------------|----------------|--------------|
| DDP | nothing | $16\Psi$ | all-reduce grads ($2\Psi$) |
| **ZeRO-1** | optimizer states | $4\Psi + 12\Psi/N_d$ | same |
| **ZeRO-2** | + gradients | $2\Psi + 14\Psi/N_d$ | same (reduce-scatter instead of all-reduce) |
| **ZeRO-3 / FSDP** | + parameters | $16\Psi/N_d$ | 1.5x (all-gather params in forward and backward) |

ZeRO-1 and ZeRO-2 are free in communication; ZeRO-3 pays 1.5 times the DDP volume because parameters must be gathered before each use.

In PyTorch the current API is **FSDP2**. The older FSDP1 `FullyShardedDataParallel` wrapper class is deprecated. FSDP2 shards each parameter as a **DTensor** on dimension 0, which is what lets it compose with tensor parallelism and FP8. You apply it block by block and then to the root:

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
| **TP** (tensor) | each matmul: column-parallel then row-parallel, so an MLP needs one all-reduce forward and one backward | 2 all-reduces per layer per pass, on the critical path | inside a node (NVLink), TP ≤ 8 |
| **SP** (sequence, with TP) | the LayerNorm/dropout regions along the sequence | turns all-reduce into reduce-scatter + all-gather, same volume, less activation memory | with TP |
| **PP** (pipeline) | layers into stages; micro-batches fill the pipe (1F1B, interleaved, zero-bubble) | point-to-point activations | across nodes; bubble ≈ $(p-1)/(m+p-1)$ |
| **CP** (context) | the sequence across GPUs for attention (ring attention: pass K/V blocks around a ring) | K/V exchange per layer | long-context (64K-1M) training |
| **EP** (expert) | MoE experts across GPUs | all-to-all token dispatch and combine | MoE models |

The pipeline bubble formula, with $p$ stages and $m$ micro-batches, explains why pipelines want many micro-batches. Tensor parallelism stays inside a node because its all-reduces sit on the critical path and need NVLink bandwidth.

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

**BF16** has FP32's 8-bit exponent, so unlike FP16 it needs no loss scaling, and it is the default for all training. **FP8** goes further ([Micikevicius et al. 2022](https://arxiv.org/abs/2209.05433)): E4M3 for the forward pass and E5M2 or E4M3 for gradients doubles the H100 matmul peak to about 1979 TFLOP/s. With so few bits it needs **scaling factors**, either per-tensor delayed or current scaling, or **MXFP8** block scaling over 32-element blocks, which is native on Blackwell. DeepSeek-V3 trained in FP8 with fine-grained block scaling.

Two libraries supply FP8 training. In **NVIDIA Transformer Engine** you use `te.Linear` and friends under `te.autocast(recipe=...)`; in TE 2.x, `fp8_autocast` is deprecated in favor of the recipe-agnostic `autocast`. In **torchao**, `convert_to_float8_training(model)` takes a `"tensorwise"`, `"rowwise"`, or `"rowwise_with_gw_hp"` recipe and composes with FSDP2 (including FP8 all-gather) and `torch.compile`.

Note what FP8 does not change: master weights and optimizer states stay in FP32. FP8 changes the matmuls, not the 16 bytes.

## Post-training

Once you have a base model, everything a customer cares about happens in post-training. The three main tools need three different kinds of data: SFT teaches format from demonstrations, preference optimization such as DPO teaches taste from comparisons, and RL with verifiable rewards such as GRPO teaches skills that a grader can check. The key papers are [InstructGPT](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), and [Tulu 3](https://arxiv.org/abs/2411.15124).

### Supervised fine-tuning

SFT is next-token training on demonstrations, with one important change: **loss masking**. You train only on the response tokens $R$, labeling prompt and system positions `-100` (the `ignore_index`), so the loss is

$$L = -\frac{1}{|R|}\sum_{t \in R} \log p(y_t \mid y_{<t}, x).$$

Training on prompts would waste capacity on imitating users. [`train_step_torch.py`](code/train_step_torch.py) implements the masking.

The model also learns the exact special-token format of its **chat template** (`<|start_header_id|>`, `<|im_start|>`, and so on). Train and serve with the *same* template, taken from the tokenizer via `apply_chat_template`.

> **Warning:** Template mismatch between training and serving is the most common silent bug in customer fine-tunes. Nothing crashes; the model just behaves worse.

To avoid wasting compute on padding, **packing** concatenates short examples into full-length sequences. Pair it with **document masking**, a position-ID reset plus block-diagonal attention (for example FlashAttention's varlen kernels), so packed examples do not attend to each other.

Typical hyperparameters are 1-3 epochs, a learning rate around 1e-5 for full fine-tuning and 1e-4 to 2e-4 for LoRA, with cosine decay and warmup. Quality beats quantity: [LIMA](https://arxiv.org/abs/2305.11206) got strong results from 1,000 curated examples.

### RLHF with PPO

Classic RLHF, as in InstructGPT, has three parts. First, train a **reward model** on preference pairs with the **Bradley-Terry** loss, where $y_w$ is the preferred response and $y_l$ the rejected one:

$$L_{RM} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

Second, use RL to maximize that reward with a KL penalty that keeps the policy near the SFT reference:

$$\max_\pi\ \mathbb{E}_{x,\, y \sim \pi}\big[r_\phi(x, y)\big] - \beta \cdot \mathrm{KL}\big(\pi(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\big)$$

In practice the KL term is applied per token as $r_t = -\beta\big(\log \pi(y_t \mid \cdot) - \log \pi_{\text{ref}}(y_t \mid \cdot)\big)$, with $r_\phi$ added at the final token.

Third, optimize with **PPO** ([Schulman et al.](https://arxiv.org/abs/1707.06347)), whose clipped surrogate uses the ratio $\rho_t = \pi_\theta / \pi_{\text{old}}$:

$$L = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\, A_t\big)\Big]$$

with advantages $A_t$ computed by GAE from a learned **value model**, the critic.

Count the models: policy, reference, reward model, and critic all live in memory at once, and generation sits inside the training loop. That cost is why the field moved on.

### Deriving DPO

DPO is RLHF with the reward model solved away. Start from the same KL-constrained objective. Its optimum has a closed form, a Gibbs distribution:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\, \exp\big(r(x,y)/\beta\big)}{Z(x)}, \qquad Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, \exp\big(r(x,y)/\beta\big)$$

Solve it for the reward:

$$r(x,y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Now substitute this into the Bradley-Terry model, $p(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$. The intractable $\beta \log Z(x)$ appears in both terms and **cancels**. Parametrize $\pi^*$ by the policy $\pi_\theta$ and maximize the likelihood of the observed preferences:

$$L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

Call the bracketed margin $h$. The gradient is

$$\nabla L = -\beta\, \sigma(-h)\,\big[\nabla \log \pi_\theta(y_w \mid x) - \nabla \log \pi_\theta(y_l \mid x)\big].$$

The weight $\sigma(-h)$ is what makes DPO sensible: it focuses learning on pairs that the implicit reward $\beta \log \pi_\theta / \pi_{\text{ref}}$ currently ranks wrongly. And at initialization, when $\pi_\theta = \pi_{\text{ref}}$, the margin is zero and the loss is exactly $\log 2$. Both facts are asserted in [`dpo_loss.py`](code/dpo_loss.py), which checks the gradient against central finite differences.

The practical consequences follow from the derivation. DPO is offline: no reward model, no sampling, and only two models, the policy and a frozen reference whose log-probs can be precomputed. Its memory cost is that of SFT plus one forward pass. The coefficient $\beta$, typically 0.05-0.5, sets how far the policy may drift. Its known failure modes are likelihood displacement, where both the chosen and the rejected log-probs fall, and verbosity bias. Variants include IPO, KTO (which works on unpaired thumbs up/down), ORPO, and SimPO (reference-free). And on-policy beats off-policy: pairs sampled from the current model and labeled by a judge or reward model (online or iterative DPO) outperform static datasets.

### GRPO and RL with verifiable rewards

DPO removes the reward model; **GRPO** ([DeepSeekMath](https://arxiv.org/abs/2402.03300)) keeps the RL loop but removes the critic. For each prompt, sample a **group** of $G$ outputs (8-64), score each, and use the group itself as the baseline. Every token of output $i$ gets the same advantage:

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)}$$

The objective is the PPO surrogate averaged per output and per token, with a KL penalty:

$$J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t \Big(\min\big(\rho_{i,t}\hat{A}_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\,\hat{A}_i\big) - \beta\, \mathrm{KL}_t(\pi_\theta \,\|\, \pi_{\text{ref}})\Big)\right]$$

The per-token KL uses the k3 estimator, which is unbiased and always nonnegative:

$$\mathrm{KL}_t = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$

The group mean is a Monte Carlo baseline. It removes a policy-sized value network along with its training instability, so memory and complexity drop to roughly those of DPO plus a generation engine.

GRPO pairs naturally with **RLVR**, RL with verifiable rewards (named in Tulu 3), where the reward is a program rather than a model: the math answer matches, the unit tests pass, the JSON validates against a schema, the format regex matches. With no learned reward model there is no reward hacking of a learned RM, though models still exploit weak graders. **DeepSeek-R1** ([2501.12948](https://arxiv.org/abs/2501.12948)) is the landmark result. R1-Zero ran GRPO with rule-based accuracy and format rewards directly on the base model, and long chain-of-thought and self-verification emerged. R1 itself added a cold-start SFT, then RL, then rejection-sampling SFT, then a final RL stage, and was distilled into small dense models.

Three follow-ups are worth naming: DAPO ([2503.14476](https://arxiv.org/abs/2503.14476)), with clip-higher, dynamic sampling, and a token-level loss; Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783)), which removes the length and std normalization biases; and GSPO ([2507.18071](https://arxiv.org/abs/2507.18071)), which uses sequence-level ratios and is stabler for MoE. Fireworks and others sell this as **RFT**, reinforcement fine-tuning, where the customer supplies the grader.

The hard part of RL is a systems problem: RL is mostly **generation**. Rollouts run in vLLM or SGLang, training runs in FSDP or Megatron, and weights sync between them every step. Asynchronous, off-policy rollouts trade staleness for throughput.

### Distillation

The last post-training tool trains a student on a teacher's outputs. **Sequence-level** distillation is SFT on teacher generations, which is how R1's distilled models were built, and needs only text. **Logit-level** distillation minimizes KL to the teacher's distribution ([Hinton et al.](https://arxiv.org/abs/1503.02531)) and needs a shared tokenizer and the teacher's logits. **On-policy distillation** (GKD) scores the student's own samples with the teacher. For a customer, "distill a frontier model's behavior on your task into an 8B" is often the best cost play, subject to the teacher's license terms.

### The library landscape

You will rarely write these trainers yourself. The main libraries, with their status as checked in October 2026, are:

| Library | Framework | Scope | Status |
|---------|-----------|-------|--------|
| **TRL** | PyTorch / HF | SFT, DPO, KTO, Reward, GRPO, RLOO, distillation; vLLM-backed generation | v1.x, active. `PPOTrainer` moved to `trl.experimental` and was then removed in v1.13 |
| **OpenRLHF** | PyTorch, Ray + vLLM + DeepSpeed | PPO with critic, REINFORCE++, GRPO, RLOO, async RL, LoRA | active |
| **verl** (Volcano Engine) | PyTorch, FSDP/FSDP2 or Megatron + vLLM/SGLang | PPO, GRPO, GSPO, DAPO, RLOO, multi-turn agentic RL; scales to 671B MoE | active, the common choice for large RL |
| **torchtune** | PyTorch | recipes for SFT, LoRA/QLoRA, DPO | **no longer actively maintained**: development wound down in 2025 (README notice). Use torchtitan for pretraining-scale PyTorch, TRL/Axolotl for fine-tuning |
| **torchtitan** | PyTorch | pretraining reference: FSDP2, TP, PP, CP, EP, Float8/MXFP8 | active |
| **Axolotl** | PyTorch (YAML over HF/TRL) | full FT, LoRA/QLoRA, DPO/KTO/ORPO, GRPO, reward models; FSDP2, DeepSpeed, sequence parallel, multi-node | active |
| **Unsloth** | PyTorch, custom Triton kernels | LoRA/QLoRA/full FT, DPO, GRPO, FP8; memory-lean single-GPU focus, multi-GPU support | active |
| **MaxText** | JAX | pretraining reference + SFT + GRPO/GSPO (rollouts in vLLM, RL via Tunix) on TPU and GPU | active |
| **Tunix** | JAX (Flax NNX) | SFT, LoRA, DPO, ORPO, PPO, GRPO, agentic RL | active, V2 |

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

## Scoping a fine-tune

Everything so far equips you for the practitioner's core loop: decide, measure, train, measure again. Scoping starts by asking whether the problem needs fine-tuning at all.

### Does this need fine-tuning?

Most requests for fine-tuning are really requests for something else. The table maps symptoms to the first fix to try and to the case where fine-tuning does win:

| Symptom | First fix | Fine-tune when |
|---------|-----------|----------------|
| Model lacks facts or private, changing knowledge | **RAG** ([Retrieval and RAG](../27-retrieval-and-rag/README.md)) | never for fast-changing facts; fine-tuning is poor at reliable knowledge injection |
| Wrong format, style, tone, schema | prompting + few-shot + structured output | prompt is long, brittle, or eats latency/cost: **SFT/LoRA** bakes it in |
| Narrow task, high volume, latency or cost bound | prompt a large model to get a baseline | **SFT or distill** into a small model; often the biggest ROI |
| "Better" is a preference, not a label | prompt iteration | **DPO** on chosen/rejected pairs |
| Correctness is checkable (code runs, answer matches, schema validates) | prompting + verification loop | **RFT / GRPO** with the checker as reward |
| Domain language the base barely knows (new language, proprietary DSL) | none | **continued pretraining** (billions of tokens), then SFT |

### Matching the method to the data

Each method consumes a different shape of data, in different amounts:

| Method | Data shape | Practical minimum | Typical |
|--------|-----------|-------------------|---------|
| SFT (LoRA) | prompt → ideal response | a few hundred high-quality | 1K-50K |
| DPO | prompt, chosen, rejected | ~1K pairs | 5K-100K |
| RFT / GRPO | prompts + grader (no answers needed) | 100s of prompts the model sometimes solves | 1K-20K |
| Distillation | prompts (teacher writes responses) | 1K | 10K-1M |

> **Warning:** GRPO needs a base pass rate strictly between 0 and 1 on the training prompts. If every sample in a group gets the same reward, then $\hat{A}_i = 0$ for all of them and nothing is learned. Prompts the model always solves or never solves are wasted.

### Evaluate first

If you cannot measure base versus tuned on a held-out set the customer agrees with, you are not ready to train. The procedure has five steps. First, build the held-out evaluation set **before training**, with the customer, from real traffic; freeze it and deduplicate it against the training set. Second, define the metric (exact match, pass@k, or a rubric scored by an LLM judge calibrated against human labels) and the threshold for shipping. Third, measure the base model, the base model with the best prompt you can write, and the tuned model on the same set; the prompt baseline is the honest comparison. Fourth, also run a general-capability regression, such as a slice of MMLU-Pro, IFEval, or the customer's other use cases, to catch **catastrophic forgetting**. Fifth, watch train and eval loss per epoch, because overfitting on small SFT sets shows up within 2-3 epochs.

The mitigations for catastrophic forgetting are LoRA instead of full fine-tuning, fewer epochs and a lower learning rate, **replay** (mixing in 5-20% general instruction data), a KL penalty to the base (built into DPO and GRPO), and evaluating the general set at every checkpoint. Evaluation design is covered in depth in [LLM Evaluation](../29-evaluation/README.md).

### Estimating the cost

The cost estimate reuses the pretraining arithmetic with a different constant:

```text
D_train   = examples × avg_tokens_per_example × epochs
FLOPs     = k · N · D_train            k = 6 (full FT), ≈ 4 (LoRA / QLoRA)
GPU-hours = FLOPs / (peak_FLOP/s × MFU) / 3600
$         = GPU-hours × $/GPU-hour  +  data creation  +  eval runs  +  (RL: rollout GPU-hours, often > training)
```

Work an example. Five thousand examples of 800 tokens for 3 epochs is 12M tokens. On an 8B with LoRA that is $4 \times 8\times10^9 \times 1.2\times10^7 = 3.8\times10^{17}$ FLOPs, which at 35% MFU on one H100 is about **0.3 GPU-hours**. The compute is a rounding error; the bill is in the 5,000 examples, the evaluation set, and the iterations. Say this early in every scoping call. Running `python code/memory_calc.py --tokens 12e6` prints the full table.

### A decision table

Putting the pieces together, the common customer situations map to recommendations like this:

| Customer situation | Recommendation |
|--------------------|----------------|
| < 100 examples, no eval set | prompt engineering + build the eval set; no training yet |
| 500-5K labeled examples, format/task adherence | LoRA SFT on an 8B-class instruct model, rank 16-64, all linear layers |
| Wants frontier quality at small-model cost | distill: frontier model generates, filter by grader, SFT a small model |
| Has thumbs up/down or A/B preference logs | SFT first, then DPO (or KTO for unpaired feedback) |
| Has a reliable automatic grader | SFT warm start, then GRPO/RFT |
| Needs > 1 task or per-tenant variants | one base, many LoRA adapters, multi-LoRA serving |
| Regulated, needs weights on-prem | LoRA or full FT on open weights; export merged weights |
| Model must learn a large new corpus | continued pretraining (full FT, multi-node), then SFT; budget weeks |

And the methods themselves, side by side, compare on data, memory, and sampling:

| Method | Data | Models in memory | Online sampling | Typical use |
|--------|------|------------------|-----------------|-------------|
| SFT | prompt → response | 1 | no | format, task, style |
| Reward model | preference pairs | 1 | no | scoring for PPO / rejection sampling |
| PPO (RLHF) | prompts + RM | 4 (policy, ref, RM, critic) | yes | classic RLHF, mostly superseded |
| DPO / KTO / SimPO | pairs / thumbs | 2 (1 for SimPO) | no (online variants: yes) | preference alignment |
| GRPO / RLOO / DAPO | prompts + verifier | 2 (policy, ref) + rollout engine | yes | reasoning, code, agents, RFT |
| Distillation | prompts + teacher | student (+ teacher logits) | optional | cheap small models |
| LoRA / QLoRA / DoRA | any of the above | base frozen + adapter | n/a | memory-bound fine-tuning |

## The same step in three frameworks

The last piece is the code you will actually read and write. Here is the same next-token training step in three front ends; the full runnable versions are [`train_step_torch.py`](code/train_step_torch.py) and [`train_step_jax.py`](code/train_step_jax.py).

In **PyTorch**, modules are mutable, autograd records an eager tape, and the optimizer mutates parameters in place:

```python
logits = model(inputs)                                        # (B, T, V)
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
```

In **JAX with Optax**, parameters are a pytree, the step is a pure function, and `jit` compiles all of it, the optimizer update included:

```python
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-3, weight_decay=0.1))

@jax.jit
def train_step(params, opt_state, inputs, targets, mask):
    def loss_fn(p):
        nll = optax.softmax_cross_entropy_with_integer_labels(forward(p, inputs), targets)
        return (nll * mask).sum() / mask.sum()
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
```

In **Keras 3**, one model definition runs on three backends, selected with `KERAS_BACKEND=jax|torch|tensorflow` before import, and `fit()` compiles to the backend's step (XLA under JAX):

```python
import os; os.environ["KERAS_BACKEND"] = "jax"
import keras

model.compile(optimizer=keras.optimizers.AdamW(3e-3, weight_decay=0.1, clipnorm=1.0),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(inputs, targets, batch_size=32, epochs=1)    # targets = inputs shifted by one
# KerasHub models expose LoRA directly: gemma_lm.backbone.enable_lora(rank=8)
```

The differences in programming model are summarized here:

| | PyTorch | JAX | Keras 3 |
|---|---------|-----|---------|
| State | mutable `nn.Module` | explicit pytree | `keras.Model` variables, backend-held |
| Compile | opt-in `torch.compile` | `jax.jit` is the model | backend's (XLA by default via `jit_compile="auto"`) |
| Scale-out | FSDP2 + DTensor TP/PP/CP | `Mesh` + `NamedSharding`, `shard_map` | `keras.distribution` (DataParallel, ModelParallel) on JAX |
| LLM training stacks | TRL, torchtitan, Megatron, verl, Axolotl, Unsloth | MaxText, Tunix | KerasHub (Gemma etc.), fine for SFT/LoRA on JAX/TPU |

**TensorFlow** as a training framework is effectively legacy for LLM work: new training stacks target PyTorch or JAX. TF survives as a Keras backend and in existing production serving (TF Serving, LiteRT on device).

## Working through the material

The chapter rewards doing the arithmetic yourself. Run [`code/memory_calc.py`](code/memory_calc.py) first and predict each column before you read it, then reproduce the 8B pretraining estimate by hand. Derive DPO on paper before you read the paper's appendix, and check your gradient against [`code/dpo_loss.py`](code/dpo_loss.py). Then fine-tune a 1-8B model with LoRA on a public task using TRL or Axolotl, with a held-out evaluation set built **before** training; report base versus tuned, plus a general-capability evaluation to catch forgetting. Read the Llama 3 and DeepSeek-R1 reports as scoping documents, listing every stage, its data, and its compute as you would for a customer. Finally, run the toy step in both [`train_step_torch.py`](code/train_step_torch.py) and [`train_step_jax.py`](code/train_step_jax.py) and list every difference in the programming model.

## Where this leads

This chapter sits at the junction of several others. It builds on the framework layer in [Training and Frameworks](../16-training-frameworks/README.md), on the model and optimizer from [Deep Learning](../02-deep-learning/README.md), and on the policy-gradient methods of [Reinforcement Learning](../04-reinforcement-learning/README.md), since RLHF and GRPO are policy-gradient methods with baselines and advantages. What the pipeline produces, the architectures, MoE, and long context, is the subject of [Foundation Models and Architectures](../05-foundation-models/README.md). NF4, FP8, and GPTQ/AWQ, which matter for QLoRA bases, FP8 training, and post-training quantization, are developed in [Quantization](../20-inference-engines/01-quantization/README.md), and shipping the fine-tune, including multi-LoRA serving, batching, the KV cache, and the rollout engines RL depends on, is covered in [Serving and Load](../20-inference-engines/04-serving-and-load/README.md) and the rest of [LLM Systems and Inference](../20-inference-engines/README.md).

In *AI Engineering*, [Model Routing and Cascades](../24-model-routing/README.md) uses the SFT-versus-DPO-versus-RFT ladder to decide when a tuned small model can replace a large one, and [LLM Evaluation](../29-evaluation/README.md) develops the held-out sets, LLM judges, and regression evaluations that eval-first scoping depends on. If you want to build these methods yourself, *AI Engineering* implements them from scratch: [Training at Scale](../17-training-at-scale/README.md) covers mixed precision, collectives, DDP, and ZeRO, and [Post-Training](../18-post-training/README.md) covers SFT, DPO, GRPO with verifiable rewards, and distillation.

The skills in this chapter show up across the industry, at very different depths, as the table below shows:

| Company | How this appears | Difficulty |
|---------|-----------------|------------|
| Fireworks AI | Managed SFT, DPO, and RFT on LoRA adapters deployed onto shared bases; FDEs scope datasets, graders, and evals with customers | Expert |
| Together AI | Full and LoRA fine-tuning, DPO, dedicated training clusters; FDEs size jobs and debug customer runs | Expert |
| Baseten | Training + serving of custom and fine-tuned models; multi-LoRA and dedicated deployments | Advanced |
| OpenAI / Anthropic / Google DeepMind | Pretraining at $10^{25+}$ FLOPs, RLHF/RLVR research, FP8 and 4D parallelism | PhD-level |
| Meta | Llama pipeline, PyTorch FSDP2 / torchtitan | PhD-level |
| DeepSeek / Qwen / Moonshot | GRPO and RLVR at scale, MoE + FP8 training, distillation into small models | PhD-level |
| NVIDIA | Megatron-Core, NeMo, Transformer Engine, MFU engineering | Expert |
| Hugging Face / Unsloth / Axolotl | The open post-training stack (TRL, PEFT, Unsloth kernels, Axolotl configs) | Advanced |

### Further reading

- Stanford [*CS336: Language Modeling from Scratch*](https://stanford-cs336.github.io/) (free): lectures and assignments that build a tokenizer, transformer, parallel training, scaling laws, data filtering, SFT, and RL.
- Hugging Face, [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (free): DP, TP, PP, CP, EP, and ZeRO with measured numbers.
- Nathan Lambert, [*RLHF Book*](https://rlhfbook.com/) (free): reward models, PPO, DPO, RLVR.
- [TRL docs](https://huggingface.co/docs/trl), the [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html), and the [JAX sharded computation guide](https://docs.jax.dev/en/latest/sharded-computation.html) (all free).
- Google DeepMind, [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/) (free): scaling on TPUs with JAX.
- [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) (a complete pipeline written down) and [*Tulu 3*](https://arxiv.org/abs/2411.15124) (a fully open post-training recipe with RLVR).
- Primary papers, all free: [Chinchilla](https://arxiv.org/abs/2203.15556), [ZeRO](https://arxiv.org/abs/1910.02054), [Megatron-LM](https://arxiv.org/abs/1909.08053), [3D parallelism](https://arxiv.org/abs/2104.04473), [activation recomputation](https://arxiv.org/abs/2205.05198), [InstructGPT](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [DoRA](https://arxiv.org/abs/2402.09353).

## Key terms

- **6ND**: the approximate training cost in FLOPs of a model with $N$ parameters on $D$ tokens: $2N$ per token forward, $4N$ backward.
- **Chinchilla-optimal**: the compute-optimal split $D \approx 20N$ that minimizes loss for a fixed training budget.
- **MFU**: model FLOPs utilization, useful model FLOPs per second divided by the hardware's peak.
- **ZeRO / FSDP**: sharding optimizer states, gradients, and (at stage 3) parameters across data-parallel ranks instead of replicating them.
- **Tensor, pipeline, context, and expert parallelism**: splitting each matmul, the layers, the sequence, or the MoE experts across devices.
- **Loss masking**: computing the SFT loss only on response tokens by labeling prompt positions with `ignore_index`.
- **DPO**: preference optimization that uses the closed-form optimum of KL-constrained RLHF so the reward model and partition function drop out.
- **GRPO**: a PPO-style RL method that replaces the critic with the group-normalized reward of several samples of the same prompt.
- **RLVR**: reinforcement learning whose reward is computed by a program (answer match, tests, schema) rather than a learned model.
- **LoRA**: fine-tuning a frozen matrix through a trainable low-rank update $(\alpha/r)BA$ initialized to zero.
- **QLoRA**: LoRA on a base stored in 4-bit NF4 with double-quantized scales, about 0.516 bytes per parameter.
- **Catastrophic forgetting**: loss of general capabilities caused by fine-tuning on a narrow task.

## Summary

- Compute is about $6ND$ and memory is 16 bytes per parameter before activations; those two numbers let you sanity-check any training plan, quote, or research claim in your head. Real runs cost about three times the ideal estimate.
- Modern models overtrain on purpose: Chinchilla's $D \approx 20N$ minimizes training cost, but inference dominates lifetime cost, so Llama 3 8B saw about 15T tokens, roughly 90 times that.
- Parallelism is a choice of what to split: optimizer state (ZeRO/FSDP), the matrix (TP), the layers (PP), the sequence (CP), or the experts (EP). Name the dimension that does not fit, then pick the tool.
- Post-training is where customers live. SFT teaches format, DPO teaches taste, and GRPO with verifiable rewards teaches skills a grader can check; each needs a different kind of data. DPO is RLHF with the reward model solved away, because $\beta \log Z(x)$ cancels in pairwise comparisons.
- LoRA makes fine-tuning a one-GPU problem: freezing the base removes 14 of 16 bytes per parameter and QLoRA removes most of the rest, so the real cost of a customer job is data and evaluation, not compute.
- Scope eval-first: without a held-out set the customer agrees with, measured on base, prompted base, and tuned models, you are not ready to train.

## Review questions

1. Why does training cost about $6ND$ FLOPs, and where does attention fit in?
   <details><summary>Answer</summary>The forward pass does one multiply-add per parameter per token, $2N$ FLOPs. The backward pass computes gradients with respect to both activations and weights, $2N$ each, for $4N$. Together that is $6N$ per token, times $D$ tokens. Attention adds about $12 \cdot L \cdot d \cdot s$ per token, which is small until the context gets long.</details>

2. If Chinchilla says $D \approx 20N$ is optimal, why was Llama 3 8B trained on about 15T tokens?
   <details><summary>Answer</summary>Chinchilla minimizes training cost for a fixed compute budget. A model's lifetime cost is $6ND + 2N \cdot D_{\text{inference}}$, and for a widely used model the inference tokens dwarf the training tokens. A smaller model trained far past $20N$ reaches the same quality and is much cheaper to serve.</details>

3. The ideal estimate for the 8B pretraining run is about 507,000 H100-hours, but Meta reports about 1.46M. Account for the gap.
   <details><summary>Answer</summary>Real runs get lower MFU than 40% at small model size, lose time to restarts after hardware failures, spend compute on evaluations, and include the ablations that found the recipe. The roughly threefold gap is normal.</details>

4. Where do the 16 bytes per parameter of mixed-precision AdamW come from, and which of them does LoRA remove for the frozen base?
   <details><summary>Answer</summary>2 bytes of BF16 weights, 2 of BF16 gradients, 4 of FP32 master weights, and 4 each for Adam's first and second moments. A frozen parameter needs no gradient, master copy, or moments, so only the 2-byte BF16 weight remains: 14 of the 16 bytes vanish. QLoRA then stores that weight in about 0.516 bytes.</details>

5. Compare ZeRO-1, ZeRO-2, and ZeRO-3 in memory per GPU and communication.
   <details><summary>Answer</summary>ZeRO-1 shards optimizer states: $4\Psi + 12\Psi/N_d$, same communication as DDP. ZeRO-2 also shards gradients: $2\Psi + 14\Psi/N_d$, same volume using reduce-scatter. ZeRO-3/FSDP also shards parameters: $16\Psi/N_d$, at 1.5 times DDP's communication because parameters are all-gathered in forward and backward.</details>

6. Why is tensor parallelism kept inside a node, while pipeline parallelism goes across nodes?
   <details><summary>Answer</summary>TP needs two all-reduces per layer per pass on the critical path, which only NVLink bandwidth makes affordable, so TP stays at 8 or below inside a node. PP sends only point-to-point activations between stages, which tolerates slower inter-node links; its cost is the bubble, about $(p-1)/(m+p-1)$, reduced by using many micro-batches.</details>

7. Derive why the partition function $Z(x)$ does not appear in the DPO loss.
   <details><summary>Answer</summary>The KL-constrained optimum is $\pi^* = \pi_{\text{ref}} \exp(r/\beta)/Z(x)$, so $r(x,y) = \beta \log(\pi^*/\pi_{\text{ref}}) + \beta \log Z(x)$. Bradley-Terry depends only on the difference $r(x,y_w) - r(x,y_l)$, and since both responses share the same prompt $x$, the $\beta \log Z(x)$ terms cancel.</details>

8. What is the DPO loss at initialization, and what does the gradient weight $\sigma(-h)$ do?
   <details><summary>Answer</summary>At $\pi_\theta = \pi_{\text{ref}}$ both log-ratios are zero, so the margin is zero and the loss is $-\log \sigma(0) = \log 2$. The weight $\sigma(-h)$ is large when the implicit reward ranks the pair wrongly and small when it ranks it correctly, so learning focuses on the mistakes.</details>

9. How does GRPO avoid a critic, and why does it need prompts the model sometimes solves?
   <details><summary>Answer</summary>It samples a group of $G$ outputs per prompt and uses the group's mean and standard deviation of rewards as the baseline, $\hat{A}_i = (r_i - \text{mean})/\text{std}$. If every output gets the same reward, all advantages are zero and the prompt contributes no learning signal, so the base pass rate must be strictly between 0 and 1.</details>

10. Why does LoRA training cost about $4ND$ rather than $6ND$?
    <details><summary>Answer</summary>The backward pass normally computes gradients with respect to activations ($2N$) and weights ($2N$). For frozen matrices there are no weight gradients, so backward costs about $2N$, and the total is about $2N + 2N = 4N$ per token.</details>

11. A customer wants to fine-tune a model so it knows their constantly changing product catalog. What do you recommend?
    <details><summary>Answer</summary>RAG, not fine-tuning. Fine-tuning is poor at reliable knowledge injection and is never the right tool for fast-changing facts. Build an evaluation set first and measure retrieval against it.</details>

12. Estimate the compute for a LoRA fine-tune of an 8B on 5,000 examples of 800 tokens for 3 epochs, and say where the real cost lies.
    <details><summary>Answer</summary>12M tokens; $4 \times 8\times10^9 \times 1.2\times10^7 = 3.8\times10^{17}$ FLOPs; at 35% MFU on one H100 that is about 0.3 GPU-hours. The compute is a rounding error; the cost is creating the 5,000 examples, building the evaluation set, and iterating.</details>

13. Why must the evaluation include a prompted-base baseline and a general-capability regression?
    <details><summary>Answer</summary>The prompted base is the honest comparison: if a good prompt gets the same result, training is not worth it. The general-capability regression (a slice of MMLU-Pro, IFEval, or the customer's other use cases) catches catastrophic forgetting that a task-only eval would miss.</details>
