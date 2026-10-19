# Training & Post-Training

## Overview

- **Primary reference**: Stanford [*CS336: Language Modeling from Scratch*](https://stanford-cs336.github.io/) (free, lectures + assignments that build tokenizer, transformer, parallel training, scaling laws, data filtering, SFT, and RL), paired with Hugging Face's [*The Ultra-Scale Playbook*](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (free, the best single document on DP/TP/PP/CP/EP and ZeRO with measured numbers)
- **Supplementary**: Nathan Lambert's [*RLHF Book*](https://rlhfbook.com/) (free, reward models, PPO, DPO, RLVR), [TRL docs](https://huggingface.co/docs/trl) (free), [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) (free), [JAX sharded computation guide](https://docs.jax.dev/en/latest/sharded-computation.html) (free), [*How to Scale Your Model*](https://jax-ml.github.io/scaling-book/) (free, Google DeepMind's TPU/JAX scaling book), the [*Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) report (free, a complete pipeline written down), [*Tulu 3*](https://arxiv.org/abs/2411.15124) (free, a fully open post-training recipe with RLVR)
- **Primary papers** (all free): [Chinchilla](https://arxiv.org/abs/2203.15556), [ZeRO](https://arxiv.org/abs/1910.02054), [Megatron-LM](https://arxiv.org/abs/1909.08053) + [3D parallelism](https://arxiv.org/abs/2104.04473) + [activation recomputation](https://arxiv.org/abs/2205.05198), [InstructGPT](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [DoRA](https://arxiv.org/abs/2402.09353)
- **Code**: [`code/`](code/) has a stdlib memory/FLOPs/GPU-hours calculator, numpy LoRA and DPO with gradient checks, and the same next-token step in PyTorch and JAX
- **Prerequisites**: [Deep Learning](../02-deep-learning/) (backprop, Adam, transformers), [Reinforcement Learning](../03-reinforcement-learning/) (policy gradients), [Foundation Models §7](../05-foundation-models/) (the lifecycle at a glance), [Training & Frameworks](../../ai-platform-engineering/01-training-and-frameworks/) (PyTorch vs JAX basics, which this topic does not repeat)
- **Estimated time**: 5-6 weeks at 10-12 hrs/week
- **Last reviewed**: October 2026

## Key Takeaways

- **Compute is `6ND`**, memory is **16 bytes per parameter** before activations. Those two numbers let you sanity-check any training plan, quote, or research claim in your head.
- **Modern models overtrain on purpose.** Chinchilla says `D ≈ 20N` minimizes loss for a fixed training budget; Llama 3 8B saw ~15T tokens (~90x that) because inference cost, not training cost, dominates a model's lifetime.
- **Parallelism is a choice of what to split**: the optimizer state (ZeRO/FSDP), the matrix (TP), the layers (PP), the sequence (CP), or the experts (EP). Name the dimension that does not fit, then pick the tool.
- **Post-training is where customers live.** SFT teaches format, preference optimization (DPO) teaches taste, and RL with verifiable rewards (GRPO) teaches skills that a grader can check. Each needs a different kind of data.
- **DPO is RLHF with the reward model solved away**: the KL-constrained optimum gives reward as `β log π/π_ref`, and the Bradley-Terry partition function cancels in pairwise comparisons.
- **LoRA makes fine-tuning a one-GPU problem.** Freezing the base removes 14 of the 16 bytes per parameter; QLoRA removes most of the remaining 2. Training compute for a typical customer job is single-digit GPU-hours: the real cost is data and evals.
- **Scope eval-first.** If you cannot measure base vs tuned on a held-out set the customer agrees with, you are not ready to train.

## How to Study

- Run [`code/memory_calc.py`](code/memory_calc.py) first and predict each column before reading it. Then reproduce the 8B pretraining estimate (§2) by hand.
- Derive DPO on paper (§4) before reading the paper's appendix, then check your gradient with [`code/dpo_loss.py`](code/dpo_loss.py).
- Fine-tune a 1-8B model with LoRA on a public task using TRL or Axolotl, with a held-out eval set built **before** training. Report base vs tuned, and a general-capability eval to catch forgetting.
- Read the Llama 3 and DeepSeek-R1 reports as scoping documents: list every stage, its data, and its compute, as you would for a customer.
- Train the toy step in both [`train_step_torch.py`](code/train_step_torch.py) and [`train_step_jax.py`](code/train_step_jax.py) and list every difference in the programming model.

---

# Concepts & Techniques

## Core Insight

Every training job is three budgets: **FLOPs** (`6ND`, paid in GPU-hours), **memory** (bytes per parameter plus activations, paid in GPU count), and **data** (tokens of the right kind, paid in people). Pretraining is FLOP-bound and data-hungry; post-training is data-bound and FLOP-cheap. Distributed training techniques trade memory for communication; PEFT trades capacity for memory. An FDE who can do the arithmetic for all three in a customer meeting, and who knows which kind of data each post-training method consumes, can scope any fine-tuning engagement and hold a credible conversation with the researchers who run the big runs.

## 1. The Pipeline and Who Does What

**[Llama 3 report](https://arxiv.org/abs/2407.21783) + [CS336](https://stanford-cs336.github.io/)**

`data → pretraining → mid-training → SFT → preference / RL → evals → release`

| Stage | What happens | Typical scale | Signal |
|-------|-------------|---------------|--------|
| **Data** | Crawl (Common Crawl), text extraction, language ID, **dedup** (exact hash + MinHash LSH near-dup), quality filtering (heuristics + classifier, e.g. FineWeb-Edu), PII and toxicity filters, **decontamination** against evals, tokenizer training (BPE), **mixture** weights across web/code/math/books/multilingual | 10s of trillions of raw tokens down to ~15T kept | none (curation) |
| **Pretraining** | Next-token cross entropy on the mixture, cosine or WSD learning-rate schedule | 10^23 to 10^25+ FLOPs | self-supervised |
| **Mid-training** | Anneal on high-quality data, up-weight code/math, **long-context extension** (raise RoPE base `θ`, train on long documents, 8K to 128K+) | 1-10% of pretraining tokens | self-supervised |
| **SFT** | Instruction→response demonstrations in a chat template | 10^4 to 10^6 examples | supervised |
| **Preference / RL** | Reward model + PPO, or DPO on pairs, or GRPO on verifiable tasks; often several rounds | 10^4 to 10^6 prompts | preference / reward |
| **Evals** | Capability (MMLU-Pro, GPQA, SWE-bench, AIME), safety, regression vs previous checkpoint | every checkpoint | measurement |
| **Release** | Quantize, convert, write model card, ship weights and serving config | | |

**Who does what**:

| Role | Owns | What they ask you |
|------|------|-------------------|
| **Research scientist** | Objectives, data recipes, ablations, the post-training algorithm | "What is your KL coefficient and how did you pick it?" |
| **ML engineer** | Training and eval pipelines, data tooling, checkpoints, reproducibility | "How do you version data and resume a failed run?" |
| **MTS / performance engineer** | MFU, kernels (FlashAttention, fused optimizers), parallelism layout, comms overlap, FP8 | "What MFU do you get on H100 for a 70B with TP=8?" |
| **Forward Deployed Engineer** | Scoping a customer's adaptation (prompting vs RAG vs LoRA vs full FT), building their dataset and eval, running the job, getting it served | "Will fine-tuning fix this, how much data, what will it cost, and how do we know it worked?" |

**Key ideas**:
- **Data is the source code.** Most quality differences between open models trace to filtering and mixtures, not architecture ([FineWeb](https://arxiv.org/abs/2406.17557), free).
- **Decontamination** is a correctness requirement: a benchmark leaked into pretraining makes every later eval meaningless.
- At a fine-tuning cloud, customers almost always start from an open base or instruct model. The FDE's pipeline is the last three rows.

## 2. Pretraining Math

**[Kaplan](https://arxiv.org/abs/2001.08361), [Chinchilla](https://arxiv.org/abs/2203.15556), [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)**

**Key ideas**:
- **Objective**: `L(θ) = -(1/T) Σ_t log p_θ(x_{t+1} | x_{≤t})`, the cross entropy between the data and the model, in nats per token. `exp(L)` is perplexity. In code: `inputs = x[:, :-1]`, `targets = x[:, 1:]` ([`train_step_torch.py`](code/train_step_torch.py)).
- **Training FLOPs ≈ 6ND**: a forward pass costs `2N` FLOPs per token (one multiply-add per parameter), the backward pass `4N` (gradients w.r.t. activations and w.r.t. weights, `2N` each). Attention adds `≈ 12·L·d·s` per token, small until context gets long.
- **Chinchilla**: for a fixed compute budget `C = 6ND`, loss is minimized with `N` and `D` scaled equally, `N_opt ∝ C^0.5`, giving **`D ≈ 20N`**. Their fitted loss is `L(N, D) = E + A/N^α + B/D^β` with `α ≈ 0.34`, `β ≈ 0.28`.
- **Why everyone overtrains**: Chinchilla minimizes training cost. Lifetime cost is `6ND + 2N·D_inference`, and `D_inference` for a popular model dwarfs `D`. A smaller model trained far past `20N` is cheaper to serve at the same quality. Llama 3 8B on ~15T tokens is ~93x Chinchilla-optimal.
- **MFU** (model FLOPs utilization) = `(6N · tokens/sec) / (n_gpus · peak FLOP/s)`. Good large dense runs reach **35-50%** BF16 MFU on H100; MoE and long-context runs are lower. **HFU** counts recomputed FLOPs too, so HFU > MFU under activation checkpointing.

**Worked example: an 8B model on 15T tokens on H100s** ([`memory_calc.py`](code/memory_calc.py)):

```
N = 8.03e9 (Llama 3 8B shape)       D = 15e12
C = 6 N D = 6 × 8.03e9 × 15e12      = 7.23e23 FLOPs
H100 dense BF16 peak                = 989 TFLOP/s
at 40% MFU                          = 3.96e14 FLOP/s per GPU
GPU-seconds = 7.23e23 / 3.96e14     = 1.83e9 s  →  ~507,000 GPU-hours
on 1,024 GPUs                       ≈ 20.6 days
at $2-3 per H100-hour               ≈ $1.0-1.5M of pure compute
```

Meta's Llama 3.1 model card reports ~1.46M H100-hours for the 8B. The ~3x gap to the ideal estimate is the lesson: real runs pay for lower MFU at small model size, restarts after hardware failures, evals, and the ablations that found the recipe.

## 3. Memory and Parallelism

**[ZeRO](https://arxiv.org/abs/1910.02054), [Megatron-LM](https://arxiv.org/abs/1909.08053), [Narayanan et al.](https://arxiv.org/abs/2104.04473), [Korthikanti et al.](https://arxiv.org/abs/2205.05198), [PyTorch FSDP](https://arxiv.org/abs/2304.11277)**

Framework basics (DDP, FSDP concept, `jit`) are in [Training & Frameworks §1-4](../../ai-platform-engineering/01-training-and-frameworks/). This section is the arithmetic and the current APIs.

### Bytes per parameter

Mixed-precision AdamW ([Micikevicius et al.](https://arxiv.org/abs/1710.03740)):

| Item | Bytes/param |
|------|-------------|
| BF16 weights (used in forward/backward) | 2 |
| BF16 gradients | 2 (4 if accumulated in FP32) |
| FP32 master weights | 4 |
| Adam first moment `m` (FP32) | 4 |
| Adam second moment `v` (FP32) | 4 |
| **Total** | **16** (18 with FP32 grads) |

An 8B model needs ~128 GB of states and a 70B ~1.1 TB **before any activation**. That is why full fine-tuning of a 70B is a multi-node job and LoRA (§5) exists.

**Activations** (per layer, BF16, with FlashAttention so no `s²` term): `≈ 34·s·b·d` bytes ([Korthikanti et al.](https://arxiv.org/abs/2205.05198)). For an 8B at `s = 4096`, `b = 1`, that is ~0.57 GB/layer, ~18 GB across 32 layers. **Activation checkpointing** stores only each layer's input (`2·s·b·d`) and recomputes the rest in backward: ~33% more compute for an order of magnitude less activation memory. Do not forget the **logits**: `s · V · 4` bytes in FP32, which is 2.1 GB per 4K sequence at `V = 128K`; fused or chunked cross entropy (Liger, Cut Cross-Entropy) avoids materializing them.

### Sharding the states: ZeRO and FSDP2

| Stage | Shards across `N_d` data-parallel ranks | Memory per GPU (Ψ params) | Comms vs DDP |
|-------|----------------------------------------|---------------------------|--------------|
| DDP | nothing | `16Ψ` | all-reduce grads (`2Ψ`) |
| **ZeRO-1** | optimizer states | `4Ψ + 12Ψ/N_d` | same |
| **ZeRO-2** | + gradients | `2Ψ + 14Ψ/N_d` | same (reduce-scatter instead of all-reduce) |
| **ZeRO-3 / FSDP** | + parameters | `16Ψ/N_d` | 1.5x (all-gather params in forward and backward) |

**FSDP2** is PyTorch's current API; FSDP1 (`FullyShardedDataParallel` wrapper class) is deprecated. FSDP2 shards each parameter as a **DTensor** on dim 0, so it composes with TP and FP8:

```python
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
for block in model.layers:          # shard each block: its all-gather overlaps the previous block's compute
    fully_shard(block, mp_policy=mp)
fully_shard(model, mp_policy=mp)    # then the root (embeddings, head)
model = torch.compile(model)        # Dynamo + Inductor: fused kernels, same eager semantics
```

### Model parallelism (Megatron-LM)

| Axis | Splits | Communication | Placement |
|------|--------|---------------|-----------|
| **TP** (tensor) | each matmul: column-parallel then row-parallel, so an MLP needs one all-reduce forward and one backward | 2 all-reduces per layer per pass, on the critical path | inside a node (NVLink), TP ≤ 8 |
| **SP** (sequence, with TP) | the LayerNorm/dropout regions along the sequence | turns all-reduce into reduce-scatter + all-gather, same volume, less activation memory | with TP |
| **PP** (pipeline) | layers into stages; micro-batches fill the pipe (1F1B, interleaved, zero-bubble) | point-to-point activations | across nodes; bubble ≈ `(p-1)/(m+p-1)` |
| **CP** (context) | the sequence across GPUs for attention (ring attention: pass K/V blocks around a ring) | K/V exchange per layer | long-context (64K-1M) training |
| **EP** (expert) | MoE experts across GPUs | all-to-all token dispatch and combine | MoE models |

**3D parallelism** is `DP × TP × PP`; **4D/5D** adds CP and EP. The canonical layout: TP inside the node, PP across nodes, FSDP/DP over the rest. Reference implementations: **Megatron-LM / Megatron-Core** (NVIDIA), **torchtitan** (PyTorch-native FSDP2 + TP + PP + CP + EP + Float8/MXFP8, the clean reference), **DeepSpeed** (ZeRO, offload).

### JAX equivalents

JAX expresses all of the above as **sharding annotations** on arrays, and XLA's partitioner (GSPMD / Shardy) inserts the collectives.

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

- `jax.shard_map` is now top-level (it graduated from `jax.experimental`). `pjit` is merged into `jax.jit`; `pmap` is legacy.
- Recent JAX defaults `make_mesh` to **Explicit** axis types, where shardings are part of each array's type; pass `axis_types=(AxisType.Auto, ...)` for classic GSPMD propagation (see [`train_step_jax.py`](code/train_step_jax.py)).
- **MaxText** is the reference JAX LLM trainer (TPU and GPU, pretraining + SFT + GRPO).

### Mixed precision: BF16 and FP8

- **BF16** has FP32's 8-bit exponent, so no loss scaling (unlike FP16). Default for all training.
- **FP8** (E4M3 forward, E5M2 or E4M3 for gradients, [Micikevicius et al. 2022](https://arxiv.org/abs/2209.05433)) doubles H100 matmul peak (~1979 TFLOP/s). Needs **scaling factors**: per-tensor delayed or current scaling, or **MXFP8** block scaling (32-element blocks, native on Blackwell). DeepSeek-V3 trained in FP8 with fine-grained block scaling.
- **NVIDIA Transformer Engine**: `te.Linear` etc. under `te.autocast(recipe=...)`. In TE 2.x `fp8_autocast` is deprecated in favor of the recipe-agnostic `autocast`.
- **torchao**: `convert_to_float8_training(model)` with `"tensorwise"`, `"rowwise"`, or `"rowwise_with_gw_hp"` recipes; composes with FSDP2 (FP8 all-gather) and `torch.compile`.
- Master weights and optimizer states stay FP32; FP8 changes the matmuls, not the 16 bytes.

## 4. Post-Training

**[InstructGPT](https://arxiv.org/abs/2203.02155), [DPO](https://arxiv.org/abs/2305.18290), [DeepSeekMath](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [Tulu 3](https://arxiv.org/abs/2411.15124)**

### SFT

**Key ideas**:
- **Loss masking**: train only on response tokens. Label prompt and system positions `-100` (`ignore_index`), so `L = -(1/|R|) Σ_{t∈R} log p(y_t | y_<t, x)`. Training on prompts wastes capacity on imitating users. Implemented in [`train_step_torch.py`](code/train_step_torch.py).
- **Chat templates**: the model learns the exact special-token format (`<|start_header_id|>`, `<|im_start|>`). Train and serve with the **same** template from the tokenizer (`apply_chat_template`). Template mismatch is the most common silent bug in customer fine-tunes.
- **Packing**: concatenate short examples into full-length sequences to avoid padding waste. Use **document masking** (position-ID reset + block-diagonal attention, e.g. FlashAttention varlen) so examples do not attend to each other.
- **Hyperparameters**: 1-3 epochs, LR ~1e-5 for full FT, ~1e-4 to 2e-4 for LoRA, cosine decay with warmup. **Quality beats quantity** ([LIMA](https://arxiv.org/abs/2305.11206): 1,000 curated examples).

### RLHF with PPO

1. **Reward model** on preference pairs with the **Bradley-Terry** loss: `L_RM = -log σ(r_φ(x, y_w) - r_φ(x, y_l))`.
2. **RL** maximizes reward with a KL penalty to the SFT reference:
   `max_π E_{x, y~π}[ r_φ(x, y) ] - β · KL(π(·|x) ‖ π_ref(·|x))`, implemented per token as `r_t = -β (log π(y_t|·) - log π_ref(y_t|·))` plus `r_φ` at the final token.
3. **PPO** ([Schulman et al.](https://arxiv.org/abs/1707.06347)) clipped surrogate with ratio `ρ_t = π_θ / π_old`:
   `L = E_t[ min(ρ_t A_t, clip(ρ_t, 1-ε, 1+ε) A_t) ]`, advantages `A_t` from GAE using a learned **value model** (critic).

Four models live in memory (policy, reference, reward, critic) and generation sits inside the training loop. That cost is why the field moved on.

### DPO: deriving the loss

Start from the same KL-constrained objective. Its optimum has a closed form (Gibbs distribution):

```
π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x),     Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β)
```

Solve for the reward: `r(x,y) = β log [π*(y|x) / π_ref(y|x)] + β log Z(x)`. Substitute into Bradley-Terry, `p(y_w ≻ y_l | x) = σ(r(x,y_w) - r(x,y_l))`. The intractable `β log Z(x)` appears in both terms and **cancels**. Parametrize `π*` by the policy `π_θ` and maximize likelihood of the observed preferences:

```
L_DPO(θ) = -E_{(x,y_w,y_l)} [ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
```

Gradient, with `h` the bracketed margin: `∇L = -β σ(-h) [∇ log π_θ(y_w|x) - ∇ log π_θ(y_l|x)]`. The weight `σ(-h)` focuses learning on pairs the implicit reward `β log π_θ/π_ref` currently ranks wrongly. At `π_θ = π_ref`, `L = log 2`. Both facts are asserted in [`dpo_loss.py`](code/dpo_loss.py).

**Key ideas**:
- Offline, no reward model, no sampling, two models (policy + frozen reference; the reference log-probs can be precomputed). Same memory as SFT plus one forward pass.
- `β` (typ. 0.05-0.5) sets how far the policy may drift. Failure modes: both chosen and rejected log-probs fall (likelihood displacement), verbosity bias. Variants: IPO, KTO (unpaired thumbs up/down), ORPO, SimPO (reference-free).
- **On-policy beats off-policy**: pairs sampled from the current model, labeled by a judge or reward model (online / iterative DPO), outperform static datasets.

### GRPO and RL with verifiable rewards

**GRPO** ([DeepSeekMath](https://arxiv.org/abs/2402.03300)): for each prompt, sample a **group** of `G` outputs (8-64), score each, and use the group as the baseline:

```
Â_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G)          (same advantage for every token of output i)
J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t min(ρ_{i,t} Â_i, clip(ρ_{i,t}, 1-ε, 1+ε) Â_i) - β KL_t(π_θ ‖ π_ref) ]
KL_t estimator (k3): π_ref/π_θ - log(π_ref/π_θ) - 1     (unbiased, always ≥ 0)
```

**Key ideas**:
- **No critic**: the group mean is a Monte Carlo baseline, removing a policy-sized value network and its training instability. Memory and complexity drop to roughly DPO plus a generation engine.
- **RLVR** (RL with verifiable rewards, named in Tulu 3): the reward is a program, not a model. Math answer match, unit tests pass, JSON schema validates, format regex. No reward model means no **reward hacking of a learned RM**, though models still exploit weak graders.
- **DeepSeek-R1** ([2501.12948](https://arxiv.org/abs/2501.12948)): R1-Zero ran GRPO with rule-based accuracy + format rewards directly on the base model; long chain-of-thought and self-verification emerged. R1 added a cold-start SFT, RL, rejection-sampling SFT, then a final RL stage, and distilled into small dense models.
- **Follow-ups worth naming**: DAPO ([2503.14476](https://arxiv.org/abs/2503.14476): clip-higher, dynamic sampling, token-level loss), Dr. GRPO ([2503.20783](https://arxiv.org/abs/2503.20783): removes length and std normalization biases), GSPO ([2507.18071](https://arxiv.org/abs/2507.18071): sequence-level ratios, stabler for MoE). Fireworks and others sell this as **RFT** (reinforcement fine-tuning): the customer supplies a grader.
- **The systems problem**: RL is mostly **generation**. Rollouts run in vLLM or SGLang, training in FSDP/Megatron, and weights sync between them each step. Async/off-policy rollouts trade staleness for throughput.

### Distillation

Train a student on a teacher's outputs. **Sequence-level** (SFT on teacher generations: how R1's distilled models were built) needs only text. **Logit-level** (KL to teacher distribution, [Hinton et al.](https://arxiv.org/abs/1503.02531)) needs a shared tokenizer and teacher logits. **On-policy distillation** (GKD) scores the student's own samples with the teacher. For a customer, "distill a frontier model's behavior on your task into an 8B" is often the best cost play, subject to the teacher's license terms.

### Libraries (status checked October 2026)

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

## 5. Parameter-Efficient Fine-Tuning

**[LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), [DoRA](https://arxiv.org/abs/2402.09353)**

### LoRA

Freeze `W₀ ∈ ℝ^{d×k}` and learn a low-rank update:

```
h = W₀x + ΔW x = W₀x + (α/r) · B A x,     A ∈ ℝ^{r×k},  B ∈ ℝ^{d×r},  r ≪ min(d, k)
init: A ~ Gaussian/Kaiming, B = 0   →   ΔW = 0, training starts exactly at the base model
trainable params per matrix: r(d + k) instead of dk
```

**Key ideas**:
- **Hypothesis**: the fine-tuning update has low intrinsic rank. Works best for "teach format / style / a narrow skill", worst for injecting large amounts of new knowledge.
- **α/r scaling** keeps update magnitude roughly constant as you change `r`, so LR transfers. **rsLoRA** uses `α/√r` ([2312.03732](https://arxiv.org/abs/2312.03732)), better at high rank.
- **Which modules**: all linear layers (`q,k,v,o,gate,up,down`), not just `q,v` as in the original paper. QLoRA showed this is needed to match full FT; Thinking Machines' [*LoRA Without Regret*](https://thinkingmachines.ai/blog/lora/) (free) found LoRA on all layers matches full FT on small-to-medium datasets with ~10x the full-FT learning rate, and is notably efficient for RL, where each episode carries few bits.
- **Merge** for zero-latency serving: `W = W₀ + (α/r)BA`. Or keep adapters separate for multi-LoRA serving.
- **Forgetting**: [*LoRA Learns Less and Forgets Less*](https://arxiv.org/abs/2405.09673) (free): lower target-domain gains than full FT on code/math continued pretraining, but better retention of base capabilities.
- Implemented with a merge check and finite-difference gradient check in [`lora_from_scratch.py`](code/lora_from_scratch.py).

### QLoRA

Base weights stored in 4 bits, adapters in BF16, gradients flow through the dequantized base:

- **NF4**: 16 levels at quantiles of `N(0,1)`, information-theoretically optimal for normally distributed weights. Block size 64 with an FP32 absmax per block = 0.5 bits/param overhead.
- **Double quantization**: quantize the absmax constants to FP8 in blocks of 256: overhead `8/64 + 32/(64·256) ≈ 0.127` bits/param. Total ≈ **4.127 bits ≈ 0.516 bytes/param**.
- **Paged optimizers**: optimizer states in NVIDIA unified memory, paged to CPU on memory spikes (long sequences).
- Compute dtype is BF16: QLoRA saves memory, not FLOPs, and runs slower than LoRA due to dequantization. Paper headline: a 65B fine-tune on one 48 GB GPU. Quantization detail: [Quantization](../04-llm-systems/quantization/).

### DoRA

Decompose `W = m · V/‖V‖_c` (magnitude vector `m`, column-normalized direction `V`). Train `m` directly and update the direction with LoRA: `W' = m · (W₀ + BA)/‖W₀ + BA‖_c`. Closer to full-FT learning dynamics at small rank, with a small compute overhead; merges like LoRA.

### Why LoRA fits on one GPU

From [`memory_calc.py`](code/memory_calc.py), Llama 3 shapes, rank 16 on all linear layers, seq 4096, micro-batch 1, activation checkpointing:

| Model | Method | Trainable | States | + activations | Min H100-80GB |
|-------|--------|-----------|--------|---------------|---------------|
| 8B | full FT (16 B/param) | 8.03B | 128.5 GB | 132 GB | 2 (FSDP), 8 in practice for throughput |
| 8B | LoRA (2 B/param base) | 41.9M (0.5%) | 16.7 GB | 20.5 GB | 1 (fits a 24 GB card) |
| 8B | QLoRA (0.516 B/param) | 41.9M | 4.8 GB | 8.6 GB | 1 |
| 70B | full FT | 70.6B | 1,129 GB | 1,138 GB | 18, i.e. 3 nodes |
| 70B | LoRA | 207M (0.3%) | 144.4 GB | 153 GB | 3 (shard the frozen base) |
| 70B | QLoRA | 207M | 39.7 GB | 48.3 GB | 1 |

The mechanism: frozen parameters need no gradient, no master copy, and no Adam moments, so 14 of 16 bytes vanish. Compute also drops: with no weight gradients for frozen matrices, backward is `~2N` instead of `4N`, so LoRA costs **`~4ND`** instead of `6ND`.

### Multi-LoRA serving

Adapters are megabytes, so one base model serves hundreds of fine-tunes: batch requests for different adapters together and apply each adapter's `BA` with segmented gather matmul kernels ([Punica](https://arxiv.org/abs/2310.18547), [S-LoRA](https://arxiv.org/abs/2311.03285)). vLLM (`--enable-lora`), SGLang, and LoRAX implement it, and it is the economic basis of per-customer fine-tunes at inference clouds: a LoRA deployment is billed like the base model. Serving detail: [Serving & Load](../04-llm-systems/serving-and-load/).

## 6. Scoping a Customer Fine-Tune

**The FDE's core loop: decide, measure, train, measure again**

### Does this need fine-tuning at all?

| Symptom | First fix | Fine-tune when |
|---------|-----------|----------------|
| Model lacks facts or private, changing knowledge | **RAG** ([Retrieval & RAG](../../ai-platform-engineering/07-retrieval-and-rag/)) | never for fast-changing facts; fine-tuning is poor at reliable knowledge injection |
| Wrong format, style, tone, schema | prompting + few-shot + structured output | prompt is long, brittle, or eats latency/cost: **SFT/LoRA** bakes it in |
| Narrow task, high volume, latency or cost bound | prompt a large model to get a baseline | **SFT or distill** into a small model; often the biggest ROI |
| "Better" is a preference, not a label | prompt iteration | **DPO** on chosen/rejected pairs |
| Correctness is checkable (code runs, answer matches, schema validates) | prompting + verification loop | **RFT / GRPO** with the checker as reward |
| Domain language the base barely knows (new language, proprietary DSL) | none | **continued pretraining** (billions of tokens), then SFT |

### Data requirements

| Method | Data shape | Practical minimum | Typical |
|--------|-----------|-------------------|---------|
| SFT (LoRA) | prompt → ideal response | a few hundred high-quality | 1K-50K |
| DPO | prompt, chosen, rejected | ~1K pairs | 5K-100K |
| RFT / GRPO | prompts + grader (no answers needed) | 100s of prompts the model sometimes solves | 1K-20K |
| Distillation | prompts (teacher writes responses) | 1K | 10K-1M |

GRPO needs a base pass rate strictly between 0 and 1 on the training prompts: if every sample in a group gets the same reward, `Â_i = 0` and nothing is learned.

### Eval-first

1. Build the **held-out eval set before training**, with the customer, from real traffic. Freeze it. Dedup it against the training set.
2. Define the metric (exact match, pass@k, rubric with an LLM judge calibrated against human labels) and the ship threshold.
3. Measure the **base model**, the **base with the best prompt**, and the **tuned model** on the same set. The prompt baseline is the honest comparison.
4. Also run a **general-capability regression** (a slice of MMLU-Pro, IFEval, or the customer's other use cases) to catch **catastrophic forgetting**.
5. Watch train vs eval loss per epoch; overfitting on small SFT sets shows up within 2-3 epochs.

**Catastrophic forgetting mitigations**: LoRA instead of full FT, fewer epochs and lower LR, **replay** (mix 5-20% general instruction data), a KL penalty to the base (built into DPO/GRPO), and evaluating the general set every checkpoint. See [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/).

### Cost estimate

```
D_train   = examples × avg_tokens_per_example × epochs
FLOPs     = k · N · D_train            k = 6 (full FT), ≈ 4 (LoRA / QLoRA)
GPU-hours = FLOPs / (peak_FLOP/s × MFU) / 3600
$         = GPU-hours × $/GPU-hour  +  data creation  +  eval runs  +  (RL: rollout GPU-hours, often > training)
```

Example: 5,000 examples × 800 tokens × 3 epochs = 12M tokens on an 8B with LoRA: `4 × 8e9 × 1.2e7 = 3.8e17` FLOPs, at 35% MFU on one H100 ≈ **0.3 GPU-hours**. The compute is a rounding error; the bill is in the 5,000 examples, the eval set, and the iterations. Say this early in every scoping call. Run `python code/memory_calc.py --tokens 12e6` for the full table.

### Decision table

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

## 7. Frameworks Side by Side

Same step, three front ends. Full runnable versions: [`train_step_torch.py`](code/train_step_torch.py), [`train_step_jax.py`](code/train_step_jax.py).

**PyTorch**: mutable modules, eager autograd tape, optimizer mutates parameters in place.

```python
logits = model(inputs)                                        # (B, T, V)
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
```

**JAX + Optax**: parameters are a pytree, the step is a pure function, and `jit` compiles all of it, including the optimizer update.

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

**Keras 3**: one model definition, three backends (`KERAS_BACKEND=jax|torch|tensorflow`, set before import). `fit()` compiles to the backend's step (XLA under JAX).

```python
import os; os.environ["KERAS_BACKEND"] = "jax"
import keras

model.compile(optimizer=keras.optimizers.AdamW(3e-3, weight_decay=0.1, clipnorm=1.0),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(inputs, targets, batch_size=32, epochs=1)    # targets = inputs shifted by one
# KerasHub models expose LoRA directly: gemma_lm.backbone.enable_lora(rank=8)
```

| | PyTorch | JAX | Keras 3 |
|---|---------|-----|---------|
| State | mutable `nn.Module` | explicit pytree | `keras.Model` variables, backend-held |
| Compile | opt-in `torch.compile` | `jax.jit` is the model | backend's (XLA by default via `jit_compile="auto"`) |
| Scale-out | FSDP2 + DTensor TP/PP/CP | `Mesh` + `NamedSharding`, `shard_map` | `keras.distribution` (DataParallel, ModelParallel) on JAX |
| LLM training stacks | TRL, torchtitan, Megatron, verl, Axolotl, Unsloth | MaxText, Tunix | KerasHub (Gemma etc.), fine for SFT/LoRA on JAX/TPU |

**TensorFlow** as a training framework is effectively legacy for LLM work: new training stacks target PyTorch or JAX. TF survives as a Keras backend and in existing production serving (TF Serving, LiteRT on device).

---

## Method Catalog

| Method | Data | Models in memory | Online sampling | Typical use |
|--------|------|------------------|-----------------|-------------|
| SFT | prompt → response | 1 | no | format, task, style |
| Reward model | preference pairs | 1 | no | scoring for PPO / rejection sampling |
| PPO (RLHF) | prompts + RM | 4 (policy, ref, RM, critic) | yes | classic RLHF, mostly superseded |
| DPO / KTO / SimPO | pairs / thumbs | 2 (1 for SimPO) | no (online variants: yes) | preference alignment |
| GRPO / RLOO / DAPO | prompts + verifier | 2 (policy, ref) + rollout engine | yes | reasoning, code, agents, RFT |
| Distillation | prompts + teacher | student (+ teacher logits) | optional | cheap small models |
| LoRA / QLoRA / DoRA | any of the above | base frozen + adapter | n/a | memory-bound fine-tuning |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| PyTorch vs JAX, DDP/FSDP basics, embedding fine-tuning | [Training & Frameworks](../../ai-platform-engineering/01-training-and-frameworks/) | The framework layer this topic builds on |
| Backprop, Adam, transformers, scaling laws | [Deep Learning](../02-deep-learning/) | The model and optimizer being trained |
| Policy gradients, PPO, baselines, advantages | [Reinforcement Learning](../03-reinforcement-learning/) | RLHF and GRPO are policy-gradient methods |
| Lifecycle, architectures, MoE, long context | [Foundation Models](../05-foundation-models/) | What the pipeline produces |
| NF4, FP8, GPTQ/AWQ | [Quantization](../04-llm-systems/quantization/) | QLoRA bases, FP8 training, post-training quantization |
| Multi-LoRA serving, batching, KV cache | [Serving & Load](../04-llm-systems/serving-and-load/), [LLM Systems](../04-llm-systems/) | Shipping the fine-tune; RL rollout engines |
| SFT vs DPO vs RFT ladder, cascades | [Model Routing & Cascades §9](../../ai-platform-engineering/11-model-routing-and-cascades/) | When a tuned small model replaces a large one |
| Held-out sets, LLM judges, regression evals | [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) | Eval-first scoping (§6) |
| KL divergence, cross entropy, Bradley-Terry likelihood | [Information Theory](../../mathematics-for-computing/11-information-theory/), [Probability](../../mathematics-for-computing/07-probability-statistics/) | The objectives in §2 and §4 |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Fireworks AI | Managed SFT, DPO, and RFT on LoRA adapters deployed onto shared bases; FDEs scope datasets, graders, and evals with customers | Expert |
| Together AI | Full and LoRA fine-tuning, DPO, dedicated training clusters; FDEs size jobs and debug customer runs | Expert |
| Baseten | Training + serving of custom and fine-tuned models; multi-LoRA and dedicated deployments | Advanced |
| OpenAI / Anthropic / Google DeepMind | Pretraining at 10^25+ FLOPs, RLHF/RLVR research, FP8 and 4D parallelism | PhD-level |
| Meta | Llama pipeline, PyTorch FSDP2 / torchtitan | PhD-level |
| DeepSeek / Qwen / Moonshot | GRPO and RLVR at scale, MoE + FP8 training, distillation into small models | PhD-level |
| NVIDIA | Megatron-Core, NeMo, Transformer Engine, MFU engineering | Expert |
| Hugging Face / Unsloth / Axolotl | The open post-training stack (TRL, PEFT, Unsloth kernels, Axolotl configs) | Advanced |
