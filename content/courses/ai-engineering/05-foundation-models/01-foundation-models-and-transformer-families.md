# Foundation models and transformer families

This part of [Foundation Models and Architectures](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Distinguish model architecture from training signal and deployment role.

## What is a foundation model

The term comes from Stanford CRFM's [*On the Opportunities and Risks of Foundation Models*](https://arxiv.org/abs/2108.07258), the paper that named the field, and Huyen's *AI Engineering* develops its practical consequences. The core idea is **pretraining plus adaptation**: self-supervised pretraining, by next-token prediction, masking, or denoising, learns general features, and adaptation specializes them.

What makes this pay off is **emergence with scale**. Capabilities such as in-context learning and chain-of-thought reasoning appear as models grow, which ties directly to the scaling laws discussed under [Transformers and modern architectures](../02-deep-learning/README.md#transformers-and-modern-architectures) in the Deep Learning chapter.

The practical result is the **AI-engineering shift**: most practitioners now *adapt* existing models rather than train them. In Huyen's framing, the new stack is prompt engineering, then RAG, then fine-tuning, then agents, then dataset engineering, with **latency and cost** as first-class constraints throughout. The adaptation options form a ladder from cheap to expensive: prompting, few-shot prompting, retrieval-augmented generation, parameter-efficient fine-tuning such as LoRA, full fine-tuning, and finally continued pretraining. You climb only as far as the task requires.

## Transformer architecture variants

The model you adapt is almost always a transformer, but not the transformer of the 2018 paper. Seven years of iteration changed most of its components, and the field has largely **converged to a recipe**.

Transformers come in **three shapes**. *Encoder-only* models (BERT) are built for understanding. *Decoder-only* models (GPT, Llama, Gemma) are built for generation and now dominate. *Encoder-decoder* models (T5, Whisper) handle sequence-to-sequence tasks such as translation and speech recognition.

Within the dominant decoder-only shape, each component has been replaced by something better. Modern models put the normalization *before* each sublayer (**pre-norm**) rather than after it, which keeps deep training stable. **RMSNorm** replaces LayerNorm, since it is cheaper, skipping the mean subtraction; Llama and Gemma both use it. **Rotary Position Embedding** (RoPE) replaces learned or sinusoidal positions: it rotates Q and K by position, encoding relative position and extrapolating to longer contexts, with ALiBi's linear attention bias as the alternative. **Gated MLPs** such as SwiGLU and GeGLU replace the ReLU feed-forward block, computing $(\text{Swish}(xW) \odot xV)W_2$ for better quality per parameter.

Two further choices shape capacity and input. **Tokenization** uses subword units, BPE in GPT and SentencePiece or Unigram in Llama and Gemma, and vocabulary size trades sequence length against embedding size. **Mixture-of-Experts** (MoE) replaces the dense MLP with routed experts, adding parameters without adding per-token FLOPs (Mixtral, DeepSeek, Gemma 4 MoE).

Put together, this is **the modern decoder-only recipe** on which Llama, Gemma, Qwen, Mistral, and DeepSeek converged, differing mostly at the margins: decoder-only, pre-norm, RMSNorm, RoPE, SwiGLU, and grouped-query attention, optionally with MoE. This table shows each component's old and modern form and why it changed:

| Component | Old | Modern | Why changed |
|-----------|-----|--------|-------------|
| Normalization | LayerNorm (post) | RMSNorm (pre) | cheaper, more stable deep |
| Positions | learned/sinusoidal | RoPE / ALiBi | extrapolates, relative |
| MLP | ReLU FFN | SwiGLU/GeGLU | better quality/param |
| Attention | MHA | GQA / MLA | smaller KV cache |
| Capacity | dense | MoE | params without FLOPs |
| Long context | full attention | sliding-window + global | linear cost |

## Attention mechanisms

Of all those components, attention has changed the most, because it is the architectural battleground, and the battle is mostly over the KV cache. The core operation is scaled dot-product attention, $\text{softmax}(QK^\top / \sqrt{d})V$, which is $O(n^2)$ in sequence length. Every decoded token must read the keys and values of all previous tokens, so the number of K/V heads directly sets the cache size and thus how many users a GPU can serve. The variants trade quality against that size:

| Mechanism | KV heads | Tradeoff |
|-----------|----------|----------|
| **MHA** (multi-head) | = query heads | Best quality, biggest KV cache |
| **MQA** (multi-query) | 1 (shared) | Tiny KV cache, some quality loss |
| **GQA** (grouped-query) | a few groups | The standard compromise (Llama 2+, Gemma) |
| **MLA** (multi-head latent) | low-rank latent | DeepSeek: compress KV into a latent, big cache savings |

Other variants attack the $O(n^2)$ cost instead, trading global context for linear cost. **Sliding-window or local attention** has each token attend to a fixed window, making cost linear in length; Mistral uses it, and Gemma alternates local and global layers. **Sparse or block attention** attends to a structured subset of positions (Longformer, BigBird). **Linear and sub-quadratic** methods such as Performer approximate attention, and their state-space cousins such as Mamba drop attention entirely for a linear recurrence.

Two more entries complete the picture. **Cross-attention** takes queries from one stream and keys and values from another, as when a decoder attends to an encoder (T5) or image generation attends to text (diffusion conditioning). And **FlashAttention** is not a new mechanism at all but the systems realization that makes exact attention memory-efficient; it is covered in [LLM Systems & Inference](../20-inference-engines/README.md#attention-and-kernel-optimization).

## Summary

Distinguish model architecture from training signal and deployment role. State the assumptions before applying the result to a different workload.

## Review questions

1. Does a decoder-only architecture imply one fixed learning method?

   <details><summary>Answer</summary>

   No. The same architecture can be pretrained, fine-tuned on demonstrations, optimized from preferences or rewards, and distilled.

   </details>

[Chapter contents](README.md) · [Next: Modalities and diffusion](02-modalities-and-diffusion.md)
