# Pass 4: Sequence Models

The models of Pass 3 look at a fixed window of context. Recurrent networks carry state across a whole sequence instead, and this pass builds them: an Elman RNN with backpropagation through time written by hand, the LSTM and GRU, a bidirectional encoder, and a recurrent language model trained with stateful truncated BPTT. The gap between the plain RNN and the gated cells is the vanishing-gradient argument made measurable.

The second half of the pass puts recurrent networks into encoder-decoders and translates dates. Without attention, a fixed-size encoder state is a bottleneck that long inputs overflow; with Bahdanau's or Luong's attention, every decoder step can look at every encoder position and the gap closes. That idea, attention, is what the transformer of Pass 5 is built from. Beam search and the sequence metrics that grade the translations complete the pass.

> **Learning objectives**
> - Explain how bootstrap confidence intervals are computed for the metrics this pass reports.
> - Derive BPTT for a vanilla RNN and explain why gated cells train where it struggles.
> - Explain how stateful truncated BPTT trains on long text with bounded memory.
> - Explain why attention removes the encoder bottleneck, and how beam search and the sequence metrics work.

## Before you begin

This pass needs [Pass 3](../course-p03-tokens-and-data/README.md): its story corpus and NPLM are the baseline the recurrent model must beat.

## In this pass

1. [LLN, CLT, confidence intervals, bootstrap](../../../mathematics-for-computing/08-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md). Afterwards you can explain why a sample mean concentrates and how a bootstrap interval is computed.
2. [Probability problem set, part c: limit theorems and confidence intervals](../../../mathematics-for-computing/08-probability-statistics/11-problem-set-c.md). Afterwards you can work every problem in the set and explain each answer.
3. [Vanilla RNN with manual BPTT and truncation](../../04-recurrent/01-vanilla-rnn-manual-bptt.md). Afterwards you can derive backpropagation through time for a vanilla RNN and explain what truncation trades away.
4. [LSTM (torch gate order)](../../04-recurrent/02-lstm.md). Afterwards you can explain each LSTM gate in torch's order and why the cell state eases vanishing gradients.
5. [GRU (torch gate order)](../../04-recurrent/03-gru.md). Afterwards you can explain how the GRU merges the LSTM's gates and how torch orders them.
6. [Bidirectional RNN with length-aware reversal](../../04-recurrent/04-bidirectional-rnn.md). Afterwards you can explain how the bidirectional RNN reverses padded sequences by length.
7. [RNN language model with stateful TBPTT](../../04-recurrent/06-rnn-language-model-stateful-tbptt.md). Afterwards you can explain how stateful TBPTT carries hidden state across batches and where it is detached.
8. [Bahdanau additive attention](../../05-attention-origins/02-bahdanau-additive-attention.md). Afterwards you can derive Bahdanau's additive attention scores and context vector.
9. [Luong attention and input feeding](../../05-attention-origins/03-luong-attention-input-feeding.md). Afterwards you can explain Luong's score functions and why input feeding helps.
10. [Encoder-decoder with teacher forcing](../../05-attention-origins/01-encoder-decoder-teacher-forcing.md). Afterwards you can explain teacher forcing and how the encoder-decoder is trained and decoded.
11. [Beam search, generic over a step function](../../05-attention-origins/04-beam-search.md). Afterwards you can explain how beam search over a generic step function ranks and prunes hypotheses.
12. [Sequence metrics: exact match, BLEU, chrF](../../05-attention-origins/05-sequence-metrics.md). Afterwards you can compute exact match, BLEU and chrF by hand on a small example.
13. [Mutation testing in depth: equivalent mutants, semantic mutants from pitfalls, reading survivors](../../../software-craftsmanship/03-testing-mentality/05-mutation-testing-in-depth.md). Afterwards you can explain equivalent and semantic mutants and what a surviving mutant says about a test suite.
14. [What the System Does After Pass 4](milestone.md). Afterwards you can explain why the LSTM beats the RNN and the NPLM, and why attention closes the long-date gap that beam search alone cannot.

## Going deeper

One optional section follows the pass for readers who want to see how recurrent language models were used for contextual embeddings before the transformer:

15. [ELMo: biLM, ScalarMix, linear probes](../../04-recurrent/05-elmo-bilm-scalarmix-probes.md). Afterwards you can explain ScalarMix and what a linear probe measures about a biLM's layers.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 5](../course-p05-transformer/README.md) builds the transformer from the attention introduced here.
