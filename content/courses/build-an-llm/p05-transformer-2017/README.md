# Part 5: The Transformer (2017)

Attention is all you need, built from its parts. Scaled dot-product attention with its backward pass, the masks (causal, padding, sliding window, additive), multi-head attention, sinusoidal and learned positions, and the full encoder-decoder transformer with both LayerNorm placements and the Noam schedule.

**Course passes**: L5.1 to L5.5 belong to [Pass 5](../paths/course-p05-transformer/milestone.md)

**Before you start**: the solve parts `S-M05` (logic) and `S-M08` (VJP derivations); rotations and frequency ladders from [Precalculus](../../mathematics-for-computing/00-precalculus/); attention from [Part 4](../p04-attention-origins/).

## Key ideas

- **Scaled dot product**: $\text{softmax}(QK^{\top}/\sqrt{d_k} + M)V$, where the mask $M$ adds $-\infty$ to forbidden positions.
- **Heads** split the model dimension so each head attends with its own projections; the outputs are concatenated and projected back.
- **LayerNorm placement** decides trainability: post-LN as in 2017 needs warmup, pre-LN trains without it.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L5.1` | Scaled dot-product attention (forward and backward) | build | 5 |
| `L5.2` | Masks: causal, padding, sliding window, additive | build | 5 |
| `L5.3` | Multi-head attention | build | 5 |
| `L5.4` | Positional encodings: sinusoidal, learned | build | 5 |
| `L5.5` | Encoder-decoder Transformer with both LayerNorm placements (post-LN as in 2017, pre-LN as `norm='pre'`), Noam schedule, label smoothing | build | 5 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L5.1` | [Scaled dot-product attention (forward and backward)](01-scaled-dot-product-attention.md) | build | 5 |
| 2 | `L5.2` | [Masks: causal, padding, sliding window, additive](02-masks.md) | build | 5 |
| 3 | `L5.3` | [Multi-head attention](03-multi-head-attention.md) | build | 5 |
| 4 | `L5.4` | [Positional encodings: sinusoidal, learned](04-positional-encodings.md) | build | 5 |
| 5 | `L5.5` | [Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing](05-encoder-decoder-transformer.md) | build | 5 |

## Going further

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762); Rush et al., [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/).
- Xiong et al., [*On Layer Normalization in the Transformer Architecture*](https://arxiv.org/abs/2002.04745).
