# Part 7: The Modern Decoder Block

What changed between 2017 and a Llama-family model. Pre-LN with RMSNorm, gated MLPs (SwiGLU, GeGLU), rotary position embeddings and context extension (PI, NTK, YaRN, Llama-3 scaling), grouped-query and multi-head latent attention, sliding windows and attention sinks, mixture of experts, and the Llama-family model that loads Hugging Face configs and weights and matches SmolLM2-135M.

**Course passes**: 5 (L7.1 to L7.9, milestone MS-L7, part of gate MS-P5)

**Before you start**: the just-in-time math `M05.1` (parameter and FLOP counting); [Part 5](../p05-transformer-2017/) and [Part 6](../p06-objectives/); rotations from [Precalculus](../../../tracks/math/00-precalculus/).

## Key ideas

- **RoPE** rotates each pair of query and key dimensions by an angle proportional to position, so their dot product depends only on the relative offset.
- **KV bytes are the serving cost**: GQA shares key and value heads across query heads; MLA compresses them into a latent and absorbs the up-projection into the query.
- **MoE** routes each token to $k$ of $E$ experts, so parameters grow without growing per-token compute; balancing the load is the hard part.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L7.1` | Pre-LN, RMSNorm | build | 5 |
| `L7.2` | Gated MLPs: SwiGLU, GeGLU | build | 5 |
| `L7.3` | RoPE (half and interleaved layouts, partial rotary) | build | 5 |
| `L7.4` | Context extension: PI, NTK, YaRN, Llama-3 scaling; ALiBi | build | 5 |
| `L7.5` | MQA/GQA attention with cache hook, window, learned sinks | build | 5 |
| `L7.6` | Multi-head latent attention (DeepSeek-V2/V3), weight absorption | build | 5 |
| `L7.7` | Sliding window, StreamingLLM sinks, learned sinks | build | 5 |
| `L7.8` | Mixture of Experts: routing, sorted dispatch, Switch aux loss, aux-free bias | build | 5 |
| `L7.9` | Llama-family model, HF config, weight loading, downloader | build | 5 |

The ALiBi part of `L7.4` is optional (course decision D31).

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L7.1` | [Pre-LN, RMSNorm](01-pre-ln-rmsnorm.md) | build | 5 |
| 2 | `L7.2` | [Gated MLPs: SwiGLU, GeGLU](02-gated-mlps.md) | build | 5 |
| 3 | `L7.3` | [RoPE (half and interleaved layouts, partial rotary)](03-rope.md) | build | 5 |
| 4 | `L7.4` | [Context extension: PI, NTK, YaRN, Llama-3 scaling; ALiBi (optional part)](04-context-extension.md) | build | 5 |
| 5 | `L7.5` | [MQA/GQA attention with cache hook, window, learned sinks](05-gqa-attention.md) | build | 5 |
| 6 | `L7.6` | [Multi-head latent attention (DeepSeek-V2/V3), weight absorption](06-multi-head-latent-attention.md) | build | 5 |
| 7 | `L7.7` | [Sliding window, StreamingLLM sinks, learned sinks](07-sliding-window-and-sinks.md) | build | 5 |
| 8 | `L7.8` | [Mixture of Experts: routing, sorted dispatch, Switch aux loss, aux-free bias](08-mixture-of-experts.md) | build | 5 |
| 9 | `L7.9` | [Llama-family model, HF config, weight loading, downloader](09-llama-family-model.md) | build | 5 |

## Going further

- Su et al., [RoFormer](https://arxiv.org/abs/2104.09864); Peng et al., [YaRN](https://arxiv.org/abs/2309.00071); Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245); DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (MLA).
- Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202); Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961); Xiao et al., [StreamingLLM](https://arxiv.org/abs/2309.17453).
