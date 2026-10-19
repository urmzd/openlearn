# Part 4: Attention Origins

Where attention came from: sequence-to-sequence translation. An encoder-decoder trained with teacher forcing, then Bahdanau's additive attention and Luong's dot, general, and concat scores with input feeding, a generic beam search over any step function, and the metrics that grade generated sequences (exact match, BLEU, chrF).

**Course passes**: 4 (L4.1 to L4.5, milestone MS-L4, part of gate MS-P4)

**Before you start**: the just-in-time math `M07.4` and the solve part `S-M07c`; the recurrent layers of [Part 3](../p03-recurrent/).

## Key ideas

- **The bottleneck**: a fixed-size encoder state cannot hold a long sentence; attention lets each decoder step read a weighted average of all encoder states.
- **Scores to weights**: a score per source position, a softmax, a weighted sum; the 2017 transformer keeps exactly this and drops the recurrence.
- **Search is separate from the model**: beam search keeps the $k$ best prefixes by summed log-probability, with length normalization.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L4.1` | Encoder-decoder with teacher forcing | build | 4 |
| `L4.2` | Bahdanau additive attention | build | 4 |
| `L4.3` | Luong attention (dot, general, concat) + input feeding | build | 4 |
| `L4.4` | Beam search (generic over `step_fn`) | build | 4 |
| `L4.5` | Sequence metrics: exact match, BLEU, chrF | build | 4 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L4.1` | [Encoder-decoder with teacher forcing](01-encoder-decoder-teacher-forcing.md) | build | 4 |
| 2 | `L4.2` | [Bahdanau additive attention](02-bahdanau-additive-attention.md) | build | 4 |
| 3 | `L4.3` | [Luong attention and input feeding](03-luong-attention-input-feeding.md) | build | 4 |
| 4 | `L4.4` | [Beam search, generic over a step function](04-beam-search.md) | build | 4 |
| 5 | `L4.5` | [Sequence metrics: exact match, BLEU, chrF](05-sequence-metrics.md) | build | 4 |

## Going further

- Bahdanau, Cho, and Bengio, [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473); Luong, Pham, and Manning, [*Effective Approaches to Attention-based NMT*](https://arxiv.org/abs/1508.04025).
- Papineni et al., *BLEU* (ACL 2002); Post, [*A Call for Clarity in Reporting BLEU Scores*](https://arxiv.org/abs/1804.08771) (sacrebleu).
