# The Modern Decoder Block

Between 2017 and a Llama-family model, almost every part of the transformer block changed. The previous two chapters gave you the 2017 architecture and the objectives built on it; none of that code can load a single checkpoint published today. This chapter rebuilds the block piece by piece until it loads Hugging Face configs and weights and matches SmolLM2-135M's logits.

The arc follows the block from the inside out. You start with the two changes that make deep stacks train without warmup: moving the norm in front of each sublayer and replacing LayerNorm with RMSNorm. Next comes the gated MLP that now holds most of a block's parameters, then rotary position embeddings, which put position inside every attention layer by rotating queries and keys so a score depends only on the offset between them, and the context-extension rules (position interpolation, NTK, YaRN, Llama 3 scaling, and the optional ALiBi) that stretch a model past its training length.

Attention itself is next, and its theme is that KV bytes, not FLOPs, are the serving cost. Grouped-query attention shares key and value heads across query heads; multi-head latent attention compresses them into a latent and absorbs the up-projection into the query; sliding windows and attention sinks bound the cache for streams of any length. Mixture of experts then routes each token to $k$ of $E$ experts, so parameters grow without growing per-token compute, and shows that balancing the load is the hard part. The last section assembles everything into one model, loads real checkpoints by name, and downloads them with a resumable, verified downloader.

> **Learning objectives**
> - Implement RMSNorm and the Pre-LN residual step, and explain why Pre-LN trains without warmup.
> - Build SwiGLU and GeGLU MLPs whose shapes and widths match published checkpoints.
> - Implement RoPE in both layouts and every common context-extension rule, matching Hugging Face to long positions.
> - Build grouped-query and multi-head latent attention with a cache hook, sliding windows, and sinks, and reason about KV bytes per token.
> - Route tokens through a mixture of experts with sorted dispatch and both load-balancing schemes.
> - Assemble a Llama-family model that loads Hugging Face configs and weights and matches SmolLM2-135M's logits.

## Before you begin

You should have [The Transformer (2017)](../p05-transformer-2017/README.md) and [Objectives](../p06-objectives/README.md) behind you, along with the `Tensor`, op library, and module system of the [foundations chapter](../p00-foundations/README.md). From *Mathematics for Computing*, [Counting: params, FLOPs, KV bytes, memory plans](../../mathematics-for-computing/05-discrete-math-1/01-counting-params-flops-kv-bytes.md) supplies the parameter and KV-byte counts used throughout, and the rotations and frequency ladders of [Precalculus](../../mathematics-for-computing/00-precalculus/README.md) are the mathematics behind RoPE.

## In this chapter

1. [Pre-LN and RMSNorm](01-pre-ln-rmsnorm.md). Normalizing by the root mean square with eps inside the root, Gemma's stored-gain offset, and the Pre-LN residual step that gives the gradient an identity path.
2. [Gated MLPs: SwiGLU and GeGLU](02-gated-mlps.md). Two projections, one gating the other, the activation on the gate, and why three matrices at width $\frac83 d$ explain Llama's odd hidden sizes.
3. [Rotary Position Embeddings](03-rope.md). Rotating query and key pairs by position-proportional angles so scores depend on relative position, in the half and interleaved layouts, with partial rotary and float32 angles.
4. [Context Extension: PI, NTK, YaRN, Llama 3 Scaling, and ALiBi](04-context-extension.md). Why slow RoPE pairs break past the trained context and how each scaling rule fixes the frequencies; ALiBi is an optional final part.
5. [Grouped-Query Attention with a Cache Hook, Windows, and Sinks](05-gqa-attention.md). Sharing kv heads across query heads, the visibility rule for cached calls, the hook through which every KV cache plugs in, and learned sinks.
6. [Multi-Head Latent Attention and Weight Absorption](06-multi-head-latent-attention.md). DeepSeek's low-rank latent cache, the decoupled rope key, and the absorbed decode form that never rebuilds per-head keys.
7. [Sliding Windows and Attention Sinks](07-sliding-window-and-sinks.md). One visibility rule for windows and sink tokens, a cache that holds at most $S + W - 1$ keys, and a float64 reference attention with learned sinks.
8. [Mixture of Experts](08-mixture-of-experts.md). Three routers, sorted dispatch with an inverse permutation, the Switch load-balancing loss, and DeepSeek-V3's aux-free bias.
9. [The Llama-Family Model: Config, Weight Loading, and Download](09-llama-family-model.md). Assembling the block, reading every `config.json` dialect, loading strictly by name, counting SmolLM2's 134,515,008 parameters, and a resumable, verified downloader.

## Further reading

- Su et al., [RoFormer](https://arxiv.org/abs/2104.09864); Peng et al., [YaRN](https://arxiv.org/abs/2309.00071); Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245); DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (MLA).
- Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202); Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961); Xiao et al., [StreamingLLM](https://arxiv.org/abs/2309.17453).
