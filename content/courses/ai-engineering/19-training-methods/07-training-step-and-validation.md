# The training step and validation

This part of [Choosing Training and Adaptation Methods](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Connect framework operations to the same optimization step and validate the result.

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

[Chapter contents](README.md)
