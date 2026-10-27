# Foundation Models and Architectures

Before about 2020, you trained a model *per task*: one for sentiment, another for translation, another for image classification. Foundation models invert this. A **foundation model** is pretrained once on broad, internet-scale data to learn general representations, and then *adapted*, by prompting, retrieval, or fine-tuning, to thousands of downstream tasks. Chip Huyen calls the resulting discipline **AI engineering**, building *with* models, in contrast to ML engineering, building models. The transformer is the substrate because it scales predictably and is modality-agnostic: give it a sequence of tokens, any tokens, and it learns their structure.

This chapter surveys that landscape, and nearly all of it is variations on three questions: how do we tokenize the input, how do we attend over it, and do we generate autoregressively or by denoising? We start with what makes a model a foundation model and how practitioners adapt one. We then trace how the 2018 transformer converged to a single recipe, why attention became an architectural battleground over the KV cache, and how text, images, and audio all became tokens. Diffusion models, the other foundation-model family, get their own treatment, followed by a tour of the 2026 model landscape. The chapter closes with the deepest mental-model shift of all: building an LLM is not the software development lifecycle. You do not *write* the behavior; you *grow* it from data.

> **Learning objectives**
> - Define a foundation model and place prompting, RAG, parameter-efficient fine-tuning, full fine-tuning, and continued pretraining on the adaptation ladder.
> - Describe the modern decoder-only recipe (pre-norm, RMSNorm, RoPE, SwiGLU, GQA, optionally MoE) and why each component replaced its predecessor.
> - Compare MHA, MQA, GQA, and MLA by KV-cache size and quality, and explain how text, images, and audio are tokenized.
> - Explain the DDPM, latent diffusion, classifier-free guidance, DiT, and flow-matching objectives.
> - Map the four learning paradigms onto the stages of the LLM lifecycle, and contrast that lifecycle with the software SDLC.

## Before you begin

This chapter builds on [Deep Learning](../02-deep-learning/README.md), especially its treatment of transformers, attention, and generative models, and on [Linear Algebra](../../mathematics/09-linear-algebra/README.md) and [Probability and Statistics](../../mathematics/13-probability-statistics/README.md) from *Mathematics*.

The optional paid reference is Chip Huyen's *AI Engineering: Building Applications with Foundation Models* (O'Reilly, 2025). Its [official companion repository](https://github.com/chiphuyen/aie-book) has free supporting materials, but not the full book, so follow the free papers and documentation listed under Further reading alongside it; see the [source register](../../../references/sources.md). Read Huyen for the *engineering* view (adaptation, evaluation, serving cost) and the original papers for the *architecture* view. Build a tiny GPT from scratch with Karpathy so the transformer recipe is concrete rather than vocabulary. For each model family, read its **model card and tech report**, which is where the real architectural choices and their justifications live. And treat text, audio, vision, and diffusion as four instantiations of one idea, sequence modeling, rather than four separate fields.

## In this chapter

1. [Foundation models and transformer families](01-foundation-models-and-transformer-families.md): Distinguish model architecture from training signal and deployment role.
2. [Modalities and diffusion](02-modalities-and-diffusion.md): Explain why modality and generation objective change the serving workload.
3. [Model landscape and lifecycle](03-model-landscape-and-lifecycle.md): Read model examples as architecture and training decisions rather than a fixed leaderboard.
