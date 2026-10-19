# Foundation Models & Architectures

## Overview

- **Optional paid reference**: *AI Engineering: Building Applications with Foundation Models* by Chip Huyen (O'Reilly, 2025). The [official companion repository](https://github.com/chiphuyen/aie-book) has free supporting materials, not the full book. Follow the free papers and documentation below alongside it; see the [source register](../../../references/sources.md).
- **Supplementary**: Stanford CRFM [*On the Opportunities and Risks of Foundation Models*](https://arxiv.org/abs/2108.07258) (free, the paper that named the field), [HuggingFace Transformers](https://huggingface.co/docs/transformers) + [Diffusers](https://huggingface.co/docs/diffusers) docs (free), Jay Alammar's [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) (free), Karpathy's [*Let's build GPT*](https://www.youtube.com/watch?v=kCc8FmEb1nY) (free), Lilian Weng's [*What are Diffusion Models?*](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (free)
- **Lifecycle & training paradigms** (for §7): Karpathy's [*State of GPT*](https://www.youtube.com/watch?v=bZQun8Y4L2A) (free, the canonical pretraining → SFT → reward modeling → RLHF walkthrough) and [*Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) (free), HuggingFace [*TRL*](https://huggingface.co/docs/trl) docs (free, SFT/reward/PPO/DPO/GRPO trainers) and the [*Alignment Handbook*](https://github.com/huggingface/alignment-handbook) (free), the [*Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) tech report (free, an end-to-end real pipeline), [*InstructGPT*](https://arxiv.org/abs/2203.02155) / [*DPO*](https://arxiv.org/abs/2305.18290) / [*Constitutional AI*](https://arxiv.org/abs/2212.08073) (free); contrast with the classic software SDLC in [*SWE at Google*](https://abseil.io/resources/swe-book) (free) -- see [Software Craftsmanship](../../software-craftsmanship/)
- **Prerequisites**: [Deep Learning](../02-deep-learning/) (transformers, attention, generative models), [Linear Algebra](../../mathematics-for-computing/03-linear-algebra/), [Probability](../../mathematics-for-computing/07-probability-statistics/)
- **Estimated time**: 4-5 weeks at 10-12 hrs/week

## Key Takeaways

- A **foundation model** is one pretrained on broad data at scale, then *adapted* to many downstream tasks -- the shift Chip Huyen calls **AI engineering** (building *with* models) vs ML engineering (building models)
- The 2018 transformer has **converged to a recipe**: decoder-only, pre-norm, RMSNorm, rotary positions, SwiGLU MLPs, grouped-query attention -- Llama, Gemma, Qwen, Mistral, and DeepSeek differ mostly at the margins
- **Attention is the architectural battleground**: MHA → MQA → GQA → MLA trades quality for KV-cache size; sliding-window and sparse variants trade global context for linear cost
- **Everything is tokens**: text (BPE), images (patches), and audio (neural-codec or spectrogram frames) all become token sequences a transformer consumes -- which is why one architecture went multimodal
- **Diffusion** is the other foundation-model family: instead of autoregressive next-token prediction, it learns to *denoise*. Modern diffusion is a transformer too (DiT), trained with flow matching
- **Building an LLM is not the software SDLC**: you don't *write* the behavior, you *grow* it from data through a pipeline of learning paradigms (self-supervised pretraining → supervised fine-tuning → preference optimization). Behavior is statistical and emergent, not specified -- which is why "testing" becomes *evals* and "bug fixes" become *more data and more alignment* (§7)

## How to Study

- Read Chip Huyen for the *engineering* view (adaptation, evaluation, serving cost), the original papers for the *architecture* view
- Build a tiny GPT from scratch (Karpathy) so the recipe in §2 is concrete, not vocabulary
- For each model family, read its **model card and tech report** -- that's where the real architectural choices (and their justifications) live
- Treat text, audio, vision, and diffusion as four instantiations of one idea (sequence modeling), not four separate fields

---

# Concepts & Techniques

## Core Insight

Before ~2020 you trained a model *per task*. Foundation models invert this: pretrain once on internet-scale data to learn general representations, then *adapt* (prompt, RAG, fine-tune) to thousands of tasks. The transformer is the substrate because it scales predictably and is modality-agnostic -- give it a sequence of tokens, any tokens, and it learns the structure. The whole landscape below is variations on "how do we tokenize the input, how do we attend over it, and do we generate autoregressively or by denoising."

## 1. What Is a Foundation Model

**Stanford CRFM 2108.07258 + Chip Huyen *AI Engineering***

**Key ideas**:
- **Pretraining + adaptation**: self-supervised pretraining (next-token, masked, or denoising) learns general features; adaptation specializes them
- **Emergence & scaling**: capabilities appear with scale (in-context learning, chain-of-thought) -- ties to [Deep Learning §11](../02-deep-learning/) scaling laws
- **The AI-engineering shift**: most practitioners now *adapt* existing models rather than train them. The new stack is **prompt engineering → RAG → fine-tuning → agents → dataset engineering**, with **latency and cost** as first-class constraints (Huyen's framing)
- **Adaptation ladder** (cheap → expensive): prompting → few-shot → RAG → PEFT/LoRA → full fine-tune → continued pretraining

## 2. Transformer Architecture Variants

**The 2018 paper, and what 7 years of iteration changed**

**Key ideas**:
- **Three shapes**: *encoder-only* (BERT -- understanding), *decoder-only* (GPT/Llama/Gemma -- generation, now dominant), *encoder-decoder* (T5, Whisper -- seq2seq, translation/ASR)
- **Pre-norm vs post-norm**: modern models put LayerNorm *before* the sublayer (pre-norm) for stable deep training
- **RMSNorm** replaces LayerNorm (cheaper, no mean subtraction) -- Llama, Gemma
- **Rotary Position Embedding (RoPE)** replaces learned/sinusoidal positions -- rotates Q/K by position, extrapolates to longer context; ALiBi is the linear-bias alternative
- **Gated MLPs**: SwiGLU / GeGLU replace the ReLU FFN -- `(Swish(xW) ⊙ xV)W₂`, better quality per parameter
- **Tokenization**: BPE (GPT), SentencePiece/Unigram (Llama, Gemma) -- subword units; vocab size trades sequence length for embedding size
- **Mixture-of-Experts (MoE)**: replace the dense MLP with routed experts -- more parameters, same per-token FLOPs (Mixtral, DeepSeek, Gemma 4 MoE)

**The modern decoder-only recipe** (Llama/Gemma/Qwen converged here): decoder-only + pre-norm + RMSNorm + RoPE + SwiGLU + GQA, optionally MoE.

## 3. Attention Mechanisms

**The architectural battleground -- mostly a fight over the KV cache**

| Mechanism | KV heads | Tradeoff |
|-----------|----------|----------|
| **MHA** (multi-head) | = query heads | Best quality, biggest KV cache |
| **MQA** (multi-query) | 1 (shared) | Tiny KV cache, some quality loss |
| **GQA** (grouped-query) | a few groups | The standard compromise (Llama 2+, Gemma) |
| **MLA** (multi-head latent) | low-rank latent | DeepSeek: compress KV into a latent, big cache savings |

**Other axes**:
- **Scaled dot-product** (the core): `softmax(QKᵀ/√d)V`, O(n²) in sequence length
- **Sliding-window / local attention**: each token attends to a fixed window (Mistral, Gemma alternate local/global layers) -- linear in length
- **Sparse / block attention**: attend to a structured subset (Longformer, BigBird)
- **Cross-attention**: queries from one stream attend to keys/values from another -- decoder→encoder (T5), text→image (diffusion conditioning)
- **Linear / sub-quadratic**: Performer, and the SSM cousins (Mamba) that drop attention entirely for linear recurrence
- **FlashAttention**: not a new *mechanism* but the systems realization that makes exact attention memory-efficient (see [LLM Systems §5](../04-llm-systems/))

## 4. Modalities: Text vs Audio vs Vision

**One architecture, four tokenizers**

**Key ideas**:
- **Text**: BPE/SentencePiece subwords → embedding table. The original setting
- **Vision**: **patchify** the image into a grid of patches, linearly embed each as a token (ViT). Pixels → patch tokens
- **Audio**: two routes --
  - *Spectrogram frames*: mel-spectrogram → frame tokens (Whisper's encoder)
  - *Neural audio codecs*: EnCodec / SoundStream quantize audio into discrete codes (RVQ), so audio becomes a token stream a transformer generates (audio LMs, TTS)
- **The unification**: once every modality is tokens, a single decoder can ingest interleaved text+image+audio tokens -- this is how models went **multimodal**
- **Fusion patterns**: *dual-encoder* contrastive (CLIP -- align image and text embeddings); *vision-language* (project image features into the LLM's token space -- LLaVA, PaliGemma); *natively multimodal* (Gemma 3/4, Gemini -- trained on mixed modalities from the start, incl. native audio in Gemma 4 E2B/E4B)
- **Text vs audio, concretely**: text is discrete and low-rate; audio is continuous and high-rate (16k+ samples/sec), so audio needs a codec/spectrogram front-end to get to a manageable token rate, and latency/streaming matter far more

## 5. Diffusion Models

**The non-autoregressive foundation-model family -- HF [Diffusers](https://huggingface.co/docs/diffusers)**

**Key ideas**:
- **Forward process**: gradually add Gaussian noise over T steps -- `q(x_t | x_0) = N(√ᾱ_t · x_0, (1−ᾱ_t)I)`
- **Reverse process**: learn to denoise. Train a network `ε_θ(x_t, t)` to predict the noise; loss = `E‖ε − ε_θ(x_t, t)‖²` (DDPM)
- **Score view**: `ε_θ` is (up to scale) the score `∇ log p(x_t)`; sampling is following the score back to data (score-based / SDE)
- **Latent diffusion** (Stable Diffusion): diffuse in a VAE's compressed latent space, not pixels -- far cheaper. Decode the final latent to an image
- **Classifier-free guidance (CFG)**: `ε̂ = ε_uncond + w·(ε_cond − ε_uncond)` -- trade diversity for prompt adherence
- **DiT (Diffusion Transformer)**: replace the U-Net with a transformer over latent patch tokens; condition on timestep/class via **adaptive LayerNorm (adaLN-zero)**. Scales like LLMs
- **Flow matching / rectified flow**: instead of the noisy DDPM chain, learn a **velocity field** `v_θ(x_t,t)` along the straight path `x_t = (1−t)x_0 + t·x_1`; loss = `E‖v_θ − (x_1 − x_0)‖²`. Fewer, straighter sampling steps. Used by **Stable Diffusion 3**, **Flux.1**, Lumina
- **MMDiT (multimodal DiT)**: separate text and image token streams with joint attention (SD3, Flux)
- **Samplers**: DDIM, DPM++ -- ODE solvers that cut sampling from ~1000 steps to ~20

**Why it matters**: diffusion dominates image/video/audio generation; the architecture (transformer + flow matching) is converging with the LLM stack, so the two families increasingly share infrastructure.

## 6. The Current Model Landscape (2026)

**Open-weight families** (read the tech report + model card):

| Family | Notable variants | Architectural notes |
|--------|------------------|---------------------|
| **Gemma** (Google) | 3 (270M/1B/4B/12B/27B), **3n** (E2B/E4B, MatFormer + per-layer embeddings, edge), **4** (E2B/E4B/26B-MoE/31B-dense, native audio + video), PaliGemma, MedGemma, CodeGemma, ShieldGemma, RecurrentGemma | multimodal, local+global attention, single-GPU focus |
| **Llama** (Meta) | 3.x, 4 (MoE) | the reference open decoder-only recipe |
| **Qwen** (Alibaba) | dense + MoE, VL, audio | strong multilingual + multimodal |
| **DeepSeek** | V3/R1 (MLA + fine-grained MoE) | MLA attention, reasoning (RL) |
| **Mistral** | 7B, Mixtral (MoE), sliding-window | efficiency-focused |
| **Phi** (Microsoft) | small, data-curated | "textbooks" data quality over scale |

**Closed/frontier**: GPT (OpenAI), Claude (Anthropic), Gemini (Google) -- API-only, larger, multimodal.

**How to read a model**: size + active params (dense vs MoE), context length, attention type (GQA/MLA/sliding), modality support, license, and the quantized GGUF/checkpoint availability ([Quantization](../04-llm-systems/quantization/)) that decides whether you can actually run it.

## 7. The LLM Development Lifecycle -- and How It Differs from Software's SDLC

**Karpathy *State of GPT* + Huyen *AI Engineering* + the Llama 3 tech report**

In traditional software you *write* the behavior: a human encodes rules as deterministic code. In an LLM you *grow* the behavior: a pipeline of **learning paradigms** turns a corpus into weights, and the behavior is whatever statistically emerged. This is the deepest mental-model shift in the track -- treat the two lifecycles as genuinely different engineering disciplines, not the same one with a neural net swapped in.

### The four learning paradigms

The paradigms differ only in *where the training signal comes from*. Each owns a stage of the lifecycle:

| Paradigm | Where the label comes from | Lifecycle stage | Deeper treatment |
|----------|----------------------------|-----------------|------------------|
| **Unsupervised** | no labels -- structure in the data itself (clustering, dedup, topic balance) | Corpus curation & data engineering | [Statistical Learning §8](../01-statistical-learning/) |
| **Self-supervised** | labels *derived* from the data (predict the next token / a masked span / a held-out view) | **Pretraining** (the expensive 99%) | [Deep Learning §10](../02-deep-learning/) |
| **Supervised** | curated human input→output pairs | **SFT / instruction tuning** | [Statistical Learning](../01-statistical-learning/) |
| **RL from feedback** (RLHF / RLAIF / DPO / GRPO) | a *preference* or *reward* signal over whole outputs | **Preference optimization / alignment** | [Deep Learning §12](../02-deep-learning/), [RL](../03-reinforcement-learning/) |

**Self-supervision is the engine** -- it's what let pretraining escape the labeling bottleneck and consume the internet. Two families worth naming specifically, because they show self-supervision is broader than "next-token prediction":

- **Siamese / joint-embedding networks**: two (weight-sharing) encoder towers map two *views* of the same datum to embeddings; the loss pulls matching pairs together and pushes non-matching apart. This is the backbone of **contrastive** learning (SimCLR, MoCo, CLIP's dual-encoder) and **metric learning** (face verification, retrieval). The classic failure mode is **representation collapse** (everything maps to one point); the fixes define the field -- negative pairs (SimCLR), a momentum target encoder (BYOL/MoCo), or stop-gradient + predictor (SimSiam), and redundancy-reduction objectives (Barlow Twins, VICReg).
- **JEPA (Joint-Embedding Predictive Architecture)**: LeCun's *non-generative* self-supervision -- instead of reconstructing pixels/tokens (generative, wastes capacity on noise), predict the *representation* of a masked target from a context, **in embedding space**. **I-JEPA** (images) and **V-JEPA / V-JEPA 2** (video, a step toward world models) avoid collapse with an asymmetric context/target encoder + EMA target and a predictor. The thesis: model the world in an abstract latent, not pixel-by-pixel -- contrast this with generative pretraining (§1) and diffusion (§5), which *do* reconstruct the input.

### The lifecycle, stage by stage

`data curation → pretraining → mid-training → SFT → preference optimization → evaluation → deployment → monitoring`, looping back on every iteration:

1. **Data curation** (unsupervised): crawl, dedup, filter, decontaminate against evals, balance the mixture. *Data is the source code now* -- most quality lives here.
2. **Pretraining** (self-supervised): next-token loss over trillions of tokens. Compute-dominant; governed by [scaling laws](../02-deep-learning/) (Chinchilla-optimal data:param ratios).
3. **Mid-training / continued pretraining**: long-context extension, domain or code up-weighting, annealing on high-quality data.
4. **SFT** (supervised): a smaller, curated set of instruction→response demonstrations teaches the *format* of being helpful.
5. **Preference optimization** (RL from feedback): RLHF (reward model + PPO), or skip the reward model with **DPO**, or **RLAIF / Constitutional AI** (AI-generated preferences), or **GRPO** for reasoning. This is where helpfulness, harmlessness, and style are dialed in.
6. **Evaluation**: not pass/fail tests -- *distributional* evals (MMLU, GPQA, SWE-bench, LMArena), behavioral red-teaming, and regression = a benchmark score *dropping*. See [Statistical Learning](../01-statistical-learning/) for honest evaluation.
7. **Deployment**: the *artifact is the weights* -- quantize ([Quantization](../04-llm-systems/quantization/)), pick a runtime ([Frameworks](../04-llm-systems/frameworks/)), serve with a KV cache ([LLM Systems](../04-llm-systems/)).
8. **Monitoring & iteration**: watch for drift, jailbreaks, and regressions; collect production preferences; feed them back into the next SFT/preference round.

### LLM lifecycle vs the software SDLC

| Stage | Traditional software SDLC | LLM development lifecycle |
|-------|---------------------------|---------------------------|
| Requirements | specs, user stories | capability target + **evals defined first** |
| Design | architecture, modules, interfaces | data mixture, tokenizer, model architecture (§2) |
| Implementation | humans **write** deterministic code | **pretraining** grows weights from data (self-supervised) |
| Refinement | -- | SFT (supervised) + preference optimization (RLHF/DPO) |
| Testing | unit/integration/e2e, **deterministic** pass/fail | **evals**: statistical, behavioral, red-team; "regression" = score drop |
| Build artifact | a versioned binary | a set of **weights** (+ tokenizer + config) |
| Deployment | CI/CD, blue-green | quantize → serve; same weights, many quantizations |
| Maintenance | patch the buggy line | **can't patch a line** -- mitigate with more data, more alignment, guardrails, or a retrain |
| Version control | git over source | git over source **+ data + weights + eval provenance** |

**The four differences that matter**:
- **Specified vs emergent**: code does exactly what's written; a model does what the data made statistically likely. You debug a distribution, not a line.
- **Deterministic vs probabilistic fixes**: a software bug has a root-cause fix; a model failure is *reduced* (more SFT data, a preference pass, a guardrail), rarely eliminated.
- **"It compiles" has no analog**: the closest signal is a loss curve plus eval scores -- success is graded, not binary.
- **The source of truth moved**: in software the code is canonical; in an LLM the **data + the eval set** are canonical and the weights are a build output. Reproducibility means versioning data and experiments, not just code ([Pragmatic Programmer DRY/reversibility](../../software-craftsmanship/) still apply -- to your *data and pipeline*).

> **Where this connects**: the *engineering practices* of the right column (review, testing-as-evals, reproducible builds, deprecation) are exactly the [Software Craftsmanship](../../software-craftsmanship/) track applied to ML. The two lifecycles diverge in *what* you build but converge on *how to build it responsibly*.

---

## Architecture Component Catalog

| Component | Old | Modern | Why changed |
|-----------|-----|--------|-------------|
| Normalization | LayerNorm (post) | RMSNorm (pre) | cheaper, more stable deep |
| Positions | learned/sinusoidal | RoPE / ALiBi | extrapolates, relative |
| MLP | ReLU FFN | SwiGLU/GeGLU | better quality/param |
| Attention | MHA | GQA / MLA | smaller KV cache |
| Capacity | dense | MoE | params without FLOPs |
| Long context | full attention | sliding-window + global | linear cost |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Transformers, attention, scaling laws, VAE/diffusion basics | [Deep Learning](../02-deep-learning/) | The architecture foundations |
| KV cache, GQA/MLA, FlashAttention, serving | [LLM Systems & Inference](../04-llm-systems/) | Running these models |
| GGUF/AWQ/FP8 quantization | [Quantization](../04-llm-systems/quantization/) | Making models fit |
| candle/diffusers/transformers runtimes | [Inference Frameworks](../04-llm-systems/frameworks/) | Where they execute |
| KL divergence, score, entropy | [Information Theory](../../mathematics-for-computing/11-information-theory/) | Diffusion + VAE objectives |
| RLHF/DPO alignment | [Deep Learning §12](../02-deep-learning/) | Post-training foundation models |
| Self-supervision, Siamese/contrastive, JEPA | [Deep Learning §10](../02-deep-learning/) | The pretraining objective (§7) |
| LLM lifecycle vs software SDLC | [Software Craftsmanship](../../software-craftsmanship/) | Building responsibly (§7) |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google/DeepMind | Gemma/Gemini architecture, multimodal, diffusion (Imagen/Veo) | PhD-level |
| OpenAI | GPT architecture, multimodal, scaling | PhD-level |
| Anthropic | Transformer internals, interpretability, post-training | PhD-level |
| Meta | Llama recipe, open-weight strategy | Expert |
| HuggingFace | Transformers + Diffusers, the open ecosystem | Expert |
| Black Forest Labs / Stability | Flux, Stable Diffusion -- flow matching, MMDiT | Expert |
