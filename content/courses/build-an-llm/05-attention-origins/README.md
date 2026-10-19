# Attention Origins

Attention was not invented for language models. It was invented for translation, as a fix for a specific failure: an encoder-decoder that squeezes a whole source sentence into one fixed-size vector forgets the start of a long input by the time it writes the end. This chapter retraces that history in code. You build a sequence-to-sequence model trained with teacher forcing, then Bahdanau's additive attention and Luong's dot, general, and concat scores with input feeding, then a beam search that works over any step function, and finally the metrics that grade generated sequences: exact match, BLEU, and chrF.

The arc matters for the rest of the book. Attention here is already the operation the 2017 transformer keeps: a score per source position, a softmax, a weighted sum. The transformer simply drops the recurrence around it. And the separation you build at the end, a model that is only a step function and a search that knows nothing else about it, is how production decoders are structured too.

Three ideas carry the chapter:

- **The bottleneck.** A fixed-size encoder state cannot hold a long sentence; attention lets each decoder step read a weighted average of all encoder states.
- **Scores to weights.** A score per source position, a softmax, a weighted sum; the 2017 transformer keeps exactly this and drops the recurrence.
- **Search is separate from the model.** Beam search keeps the $k$ best prefixes by summed log-probability, with length normalization.

> **Learning objectives**
> - Build an encoder-decoder with a bridge, train it with teacher forcing and scheduled sampling, and show that attention beats the bottleneck on the same budget.
> - Implement Bahdanau and Luong attention with exact padding masks, and explain where each reads the source.
> - Implement beam search generic over a step function, and show by hand a case where it beats greedy decoding.
> - Compute exact match, BLEU, and chrF exactly as sacreBLEU does, with a bootstrap interval.

## Before you begin

You should have built the recurrent layers of [Recurrent networks](../04-recurrent/README.md), in particular the [GRU](../04-recurrent/03-gru.md), the [LSTM](../04-recurrent/02-lstm.md), and the [bidirectional RNN](../04-recurrent/04-bidirectional-rnn.md), on top of the tensor, op, and module foundations of [Foundations](../01-foundations/README.md). The metrics section uses the bootstrap from [LLN, CLT, confidence intervals, bootstrap](../../mathematics-for-computing/08-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md) in *Mathematics for Computing*; its [problem set](../../mathematics-for-computing/08-probability-statistics/11-problem-set-c.md) is good practice before you start.

## In this chapter

1. [Encoder-decoder with teacher forcing](01-encoder-decoder-teacher-forcing.md): an encoder, a bridge, and a decoder written as one step function, trained with teacher forcing or scheduled sampling, with or without attention. It ends with the experiment that motivates the chapter: on the same budget, attention reverses digit strings almost perfectly and the bottleneck model does not.
2. [Bahdanau additive attention](02-bahdanau-additive-attention.md): the first attention mechanism, a one-hidden-layer score per source position, with padding masked to $-\infty$ on the scores and the key projection computed once per sentence.
3. [Luong attention and input feeding](03-luong-attention-input-feeding.md): the `dot`, `general`, and `concat` scores, attending after the recurrent step, and feeding the attentional state back into the next input. The unscaled dot score is one step from the transformer's.
4. [Beam search, generic over a step function](04-beam-search.md): keeping the $K$ best prefixes, ranking all extensions together, gathering any model state by parent, and normalizing by length.
5. [Sequence metrics: exact match, BLEU, chrF](05-sequence-metrics.md): the metrics that grade generated text, built from sufficient statistics to match sacreBLEU exactly, each with a bootstrap interval.

## Further reading

- Bahdanau, Cho, and Bengio, [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473); Luong, Pham, and Manning, [*Effective Approaches to Attention-based NMT*](https://arxiv.org/abs/1508.04025).
- Papineni et al., *BLEU* (ACL 2002); Post, [*A Call for Clarity in Reporting BLEU Scores*](https://arxiv.org/abs/1804.08771) (sacrebleu).
