# Model landscape and lifecycle

This part of [Foundation Models and Architectures](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Read model examples as architecture and training decisions rather than a fixed leaderboard.

## The model landscape in 2026

With the architectures in hand, you can read the current models as combinations of these choices. These are the main open-weight families; for each, read the tech report and model card:

| Family | Notable variants | Architectural notes |
|--------|------------------|---------------------|
| **Gemma** (Google) | 3 (270M/1B/4B/12B/27B), **3n** (E2B/E4B, MatFormer + per-layer embeddings, edge), **4** (E2B/E4B/26B-MoE/31B-dense, native audio + video), PaliGemma, MedGemma, CodeGemma, ShieldGemma, RecurrentGemma | multimodal, local+global attention, single-GPU focus |
| **Llama** (Meta) | 3.x, 4 (MoE) | the reference open decoder-only recipe |
| **Qwen** (Alibaba) | dense + MoE, VL, audio | strong multilingual + multimodal |
| **DeepSeek** | V3/R1 (MLA + fine-grained MoE) | MLA attention, reasoning (RL) |
| **Mistral** | 7B, Mixtral (MoE), sliding-window | efficiency-focused |
| **Phi** (Microsoft) | small, data-curated | "textbooks" data quality over scale |

The closed, frontier models, GPT (OpenAI), Claude (Anthropic), and Gemini (Google), are API-only, larger, and multimodal.

To read any model, check its size and active parameters (dense or MoE), its context length, its attention type (GQA, MLA, or sliding-window), its modality support, its license, and the availability of quantized GGUF or other checkpoints, which decides whether you can actually run it; see [Quantization: Math to Code](../20-inference-engines/01-quantization/README.md).

## The LLM development lifecycle

The architecture tells you what a model *is*. The final question is how one comes to exist, and here the right mental model departs sharply from ordinary software. In traditional software you *write* the behavior: a human encodes rules as deterministic code. In an LLM you *grow* the behavior: a pipeline of **learning paradigms** turns a corpus into weights, and the behavior is whatever statistically emerged. Behavior is emergent rather than specified, which is why "testing" becomes *evals* and "bug fixes" become *more data and more alignment*. This is the deepest shift in the chapter, so treat the two lifecycles as genuinely different engineering disciplines, not the same one with a neural network swapped in. The account here draws on Karpathy's *State of GPT*, Huyen's *AI Engineering*, and the Llama 3 tech report.

### The four learning paradigms

The paradigms differ only in *where the training signal comes from*, and each owns a stage of the lifecycle:

| Paradigm | Where the label comes from | Lifecycle stage | Deeper treatment |
|----------|----------------------------|-----------------|------------------|
| **Unsupervised** | no labels: structure in the data itself (clustering, dedup, topic balance) | Corpus curation & data engineering | [Unsupervised learning](../01-statistical-learning/README.md#unsupervised-learning) |
| **Self-supervised** | labels *derived* from the data (predict the next token / a masked span / a held-out view) | **Pretraining** (the expensive 99%) | [Representation learning](../02-deep-learning/README.md#representation-learning) |
| **Supervised** | curated human input→output pairs | **SFT / instruction tuning** | [Statistical Learning](../01-statistical-learning/README.md) |
| **RL from feedback** (RLHF / RLAIF / DPO / GRPO) | a *preference* or *reward* signal over whole outputs | **Preference optimization / alignment** | [RLHF and alignment](../02-deep-learning/README.md#rlhf-and-alignment), [Reinforcement Learning](../04-reinforcement-learning/README.md) |

**Self-supervision is the engine.** It is what let pretraining escape the labeling bottleneck and consume the internet. Two families are worth naming specifically, because they show that self-supervision is broader than next-token prediction.

**Siamese or joint-embedding networks** use two weight-sharing encoder towers to map two *views* of the same datum to embeddings; the loss pulls matching pairs together and pushes non-matching pairs apart. This is the backbone of **contrastive** learning (SimCLR, MoCo, CLIP's dual encoder) and of **metric learning** (face verification, retrieval). The classic failure mode is **representation collapse**, in which everything maps to one point, and the fixes define the field: negative pairs (SimCLR), a momentum target encoder (BYOL, MoCo), stop-gradient plus a predictor (SimSiam), and redundancy-reduction objectives (Barlow Twins, VICReg).

**JEPA** (Joint-Embedding Predictive Architecture) is LeCun's *non-generative* self-supervision. Instead of reconstructing pixels or tokens, which is generative and wastes capacity on noise, it predicts the *representation* of a masked target from a context, **in embedding space**. **I-JEPA** (images) and **V-JEPA** and **V-JEPA 2** (video, a step toward world models) avoid collapse with an asymmetric context and target encoder, an EMA target, and a predictor. The thesis is to model the world in an abstract latent space, not pixel by pixel. Contrast this with generative pretraining and with diffusion, both of which *do* reconstruct the input.

### The lifecycle, stage by stage

The paradigms are applied in a fixed order, looping back on every iteration: data curation, pretraining, mid-training, SFT, preference optimization, evaluation, deployment, and monitoring.

1. **Data curation** (unsupervised): crawl, deduplicate, filter, decontaminate against evals, and balance the mixture. *Data is the source code now*; most of the quality lives here.
2. **Pretraining** (self-supervised): next-token loss over trillions of tokens. This stage dominates compute and is governed by [scaling laws](../02-deep-learning/README.md#transformers-and-modern-architectures), such as Chinchilla-optimal data-to-parameter ratios.
3. **Mid-training or continued pretraining**: long-context extension, up-weighting of domain or code data, and annealing on high-quality data.
4. **SFT** (supervised): a smaller, curated set of instruction-response demonstrations teaches the *format* of being helpful.
5. **Preference optimization** (RL from feedback): RLHF with a reward model and PPO; or **DPO**, which skips the reward model; or **RLAIF** and Constitutional AI, with AI-generated preferences; or **GRPO** for reasoning. This is where helpfulness, harmlessness, and style are dialed in.
6. **Evaluation**: not pass/fail tests but *distributional* evals (MMLU, GPQA, SWE-bench, LMArena) and behavioral red-teaming, where a regression means a benchmark score *dropping*. The [Statistical Learning](../01-statistical-learning/README.md) chapter covers honest evaluation.
7. **Deployment**: *the artifact is the weights*. Quantize them ([Quantization: Math to Code](../20-inference-engines/01-quantization/README.md)), pick a runtime ([Inference Frameworks](../20-inference-engines/02-frameworks/README.md)), and serve with a KV cache ([LLM Systems & Inference](../20-inference-engines/README.md)).
8. **Monitoring and iteration**: watch for drift, jailbreaks, and regressions; collect production preferences; and feed them back into the next SFT or preference round.

### The LLM lifecycle versus the software SDLC

Set side by side, the two lifecycles share stage names but differ in almost every stage's content:

| Stage | Traditional software SDLC | LLM development lifecycle |
|-------|---------------------------|---------------------------|
| Requirements | specs, user stories | capability target + **evals defined first** |
| Design | architecture, modules, interfaces | data mixture, tokenizer, model architecture |
| Implementation | humans **write** deterministic code | **pretraining** grows weights from data (self-supervised) |
| Refinement | (none) | SFT (supervised) + preference optimization (RLHF/DPO) |
| Testing | unit/integration/e2e, **deterministic** pass/fail | **evals**: statistical, behavioral, red-team; "regression" = score drop |
| Build artifact | a versioned binary | a set of **weights** (+ tokenizer + config) |
| Deployment | CI/CD, blue-green | quantize → serve; same weights, many quantizations |
| Maintenance | patch the buggy line | **can't patch a line**: mitigate with more data, more alignment, guardrails, or a retrain |
| Version control | git over source | git over source **+ data + weights + eval provenance** |

Four differences matter most. First, behavior is **specified versus emergent**: code does exactly what is written, while a model does what the data made statistically likely, so you debug a distribution, not a line. Second, fixes are **deterministic versus probabilistic**: a software bug has a root-cause fix, while a model failure is *reduced*, by more SFT data, a preference pass, or a guardrail, and rarely eliminated. Third, **"it compiles" has no analog**: the closest signal is a loss curve plus eval scores, and success is graded, not binary. Fourth, **the source of truth moved**: in software the code is canonical, while for an LLM the **data and the eval set** are canonical and the weights are a build output. Reproducibility means versioning data and experiments, not just code, and the *Pragmatic Programmer*'s principles of DRY and reversibility from [*Software Engineering*](../../software-engineering/README.md) still apply, to your *data and pipeline*.

The engineering practices of the right-hand column, review, testing as evals, reproducible builds, and deprecation, are exactly *Software Engineering* applied to ML. The two lifecycles diverge in *what* you build but converge on *how to build it responsibly*. For a contrast with the classic software lifecycle, read [*Software Engineering at Google*](https://abseil.io/resources/swe-book).

## Where this leads

This chapter sits between the foundations and the systems. The architecture foundations, transformers, attention, scaling laws, and VAE and diffusion basics, come from [Deep Learning](../02-deep-learning/README.md), which also covers the self-supervised, Siamese, contrastive, and JEPA objectives behind pretraining and the RLHF and DPO methods for post-training. Running these models, with KV caches, GQA and MLA, FlashAttention, and serving, is the subject of [LLM Systems & Inference](../20-inference-engines/README.md); making them fit with GGUF, AWQ, and FP8 is [Quantization: Math to Code](../20-inference-engines/01-quantization/README.md); and the candle, diffusers, and transformers runtimes where they execute are mapped in [Inference Frameworks](../20-inference-engines/02-frameworks/README.md). The pipeline from pretraining through post-training is worked out in detail in [Training and Post-Training](../19-training-methods/README.md). The KL divergence, score, and entropy in the diffusion and VAE objectives come from [Information Theory](../../mathematics/17-information-theory/README.md) in *Mathematics*, and building responsibly across the LLM lifecycle connects to [*Software Engineering*](../../software-engineering/README.md).

This material appears throughout the industry: in Gemma and Gemini architecture, multimodality, and diffusion (Imagen, Veo) at Google DeepMind; in GPT architecture, multimodality, and scaling at OpenAI; in transformer internals, interpretability, and post-training at Anthropic; in the Llama recipe and open-weight strategy at Meta; in the Transformers and Diffusers open ecosystem at Hugging Face; and in Flux and Stable Diffusion, with flow matching and MMDiT, at Black Forest Labs and Stability.

### Further reading

- Stanford CRFM, [*On the Opportunities and Risks of Foundation Models*](https://arxiv.org/abs/2108.07258), the paper that named the field
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) and [Diffusers](https://huggingface.co/docs/diffusers) documentation
- Jay Alammar, [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/)
- Andrej Karpathy, [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Lilian Weng, [*What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- On the lifecycle and training paradigms: Karpathy's [*State of GPT*](https://www.youtube.com/watch?v=bZQun8Y4L2A) (the canonical pretraining → SFT → reward modeling → RLHF walkthrough) and [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI); the Hugging Face [*TRL*](https://huggingface.co/docs/trl) docs (SFT, reward, PPO, DPO, and GRPO trainers) and [*Alignment Handbook*](https://github.com/huggingface/alignment-handbook); the [*Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) tech report (an end-to-end real pipeline); and the [*InstructGPT*](https://arxiv.org/abs/2203.02155), [*DPO*](https://arxiv.org/abs/2305.18290), and [*Constitutional AI*](https://arxiv.org/abs/2212.08073) papers
- [*Software Engineering at Google*](https://abseil.io/resources/swe-book), for the classic software SDLC

## Key terms

- **foundation model**: a model pretrained on broad data at scale and then adapted to many downstream tasks.
- **adaptation ladder**: the sequence of increasingly expensive ways to specialize a model, from prompting through RAG and fine-tuning to continued pretraining.
- **RoPE**: rotary position embedding, which rotates queries and keys by position to encode relative position.
- **SwiGLU**: a gated MLP, $(\text{Swish}(xW) \odot xV)W_2$, that replaced the ReLU feed-forward block.
- **GQA / MLA**: attention variants that shrink the KV cache by sharing K/V heads across groups of query heads, or by compressing K/V into a low-rank latent.
- **patchify**: cutting an image into a grid of patches, each embedded as one token.
- **classifier-free guidance**: combining conditional and unconditional denoising predictions with weight $w$ to trade diversity for prompt adherence.
- **flow matching**: training a velocity field along straight paths between noise and data, giving fewer, straighter sampling steps.
- **JEPA**: a non-generative self-supervised architecture that predicts the embedding of a masked target rather than reconstructing it.
- **representation collapse**: the failure of joint-embedding training in which all inputs map to the same point.

## Summary

- A foundation model is pretrained once and adapted many times; most practitioners now build *with* models, with latency and cost as first-class constraints.
- The transformer has converged to a recipe, decoder-only with pre-norm, RMSNorm, RoPE, SwiGLU, and GQA, optionally MoE, and the major open families differ mostly at the margins.
- Attention is the architectural battleground: MHA, MQA, GQA, and MLA trade quality for KV-cache size, while sliding-window and sparse variants trade global context for linear cost.
- Everything is tokens: text subwords, image patches, and audio codec codes or spectrogram frames all feed one architecture, which is how models went multimodal.
- Diffusion generates by denoising, and modern diffusion is a transformer (DiT) trained with flow matching, converging with the LLM stack.
- Building an LLM is not the software SDLC: behavior is grown from data through self-supervised pretraining, SFT, and preference optimization, so tests become evals and fixes become more data and alignment.

## Review questions

1. Why did foundation models shift most practitioners from training models to adapting them, and what is the cheapest-to-most-expensive order of adaptation methods?
   <details><summary>Answer</summary>Pretraining on internet-scale data yields general representations and emergent capabilities, such as in-context learning, that a per-task model cannot match, and pretraining is expensive enough that few teams repeat it. Adaptation, in order of cost, is prompting, few-shot prompting, RAG, PEFT such as LoRA, full fine-tuning, and continued pretraining.</details>
2. List the components of the modern decoder-only recipe and say what each replaced and why.
   <details><summary>Answer</summary>Pre-norm replaced post-norm for stable deep training; RMSNorm replaced LayerNorm because it is cheaper; RoPE replaced learned or sinusoidal positions because it encodes relative position and extrapolates; SwiGLU replaced the ReLU FFN for better quality per parameter; GQA replaced MHA for a smaller KV cache; and MoE optionally replaces the dense MLP to add parameters without adding per-token FLOPs.</details>
3. Why is attention called a fight over the KV cache, and how do MQA, GQA, and MLA differ?
   <details><summary>Answer</summary>During decoding every token reads all previous keys and values, so the number and size of K/V heads set the cache size and therefore how many concurrent users fit in memory. MQA shares a single K/V head across all query heads (tiny cache, some quality loss), GQA uses a few shared groups (the standard compromise), and MLA compresses K/V into a low-rank latent (DeepSeek's large cache savings).</details>
4. Why does audio need a codec or spectrogram front end while text does not?
   <details><summary>Answer</summary>Text is discrete and low-rate, so subword tokens are already a manageable sequence. Audio is continuous and high-rate, at 16,000 or more samples per second, so it must be converted to spectrogram frames or quantized into discrete codec codes (for example EnCodec or SoundStream with RVQ) to reach a token rate a transformer can handle, and streaming latency matters more.</details>
5. State the DDPM training loss and the flow-matching loss, and explain the practical advantage of flow matching.
   <details><summary>Answer</summary>DDPM trains $\epsilon_\theta(x_t, t)$ to predict the added noise, minimizing $\mathbb{E}\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$. Flow matching learns a velocity field along the straight path $x_t = (1 - t)x_0 + t x_1$, minimizing $\mathbb{E}\lVert v_\theta - (x_1 - x_0) \rVert^2$. Straighter paths need fewer sampling steps.</details>
6. How does JEPA differ from contrastive Siamese learning and from generative pretraining?
   <details><summary>Answer</summary>Contrastive Siamese networks pull embeddings of two views of the same datum together and push other pairs apart, guarding against collapse with negatives, momentum encoders, or stop-gradients. Generative pretraining reconstructs tokens or pixels. JEPA predicts the embedding of a masked target from a context, entirely in representation space, avoiding collapse with asymmetric encoders, an EMA target, and a predictor, so it spends no capacity modeling unpredictable detail.</details>
7. Give the four differences between the LLM lifecycle and the software SDLC that matter most.
   <details><summary>Answer</summary>Behavior is emergent rather than specified, so you debug a distribution; failures are reduced probabilistically rather than fixed at a root cause; there is no "it compiles", only graded loss curves and eval scores; and the source of truth is the data and eval set, with weights as a build output, so reproducibility requires versioning data, weights, and eval provenance as well as code.</details>

[Chapter contents](README.md)
