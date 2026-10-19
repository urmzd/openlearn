# Recurrent Networks

Every model in the book so far predicts from a fixed window of context. This chapter gives the model a state: a vector it updates once per token, so in principle it can carry information across any distance. Making that work in practice is a story about gradients. You start by writing backpropagation through time by hand for a vanilla RNN, and you see exactly why the gradient into an early state vanishes or explodes: it is a product of Jacobians, one per step, and its size is governed by the spectral radius of the recurrent matrix. Clipping and gating are the cures.

The gated cells come next. The LSTM adds a memory cell updated by addition, so along it the gradient is multiplied only by a learned forget gate; the GRU gets most of the same effect with one state and two gates. Both are built in torch's exact gate order and parameter layout, so weights load from torch checkpoints, and a golden comparison catches the gate-order and reset-placement bugs that otherwise fail silently. Then you compose two single-direction cells into a bidirectional layer, which looks trivial until padded batches force a length-aware reversal. The chapter closes by turning all three cells into a language model trained with stateful truncated BPTT, where the state flows across training windows but the gradient is cut at each boundary. An optional section on ELMo shows how the states of a bidirectional language model become contextual word vectors, and how a linear probe measures what each layer has learned.

> **Learning objectives**
> - Derive and implement backpropagation through time by hand, including truncation, and predict exploding and vanishing gradients from the spectral radius.
> - Build LSTM and GRU cells, stacked and masked for padded batches, that match torch's outputs, gradients, and parameter names.
> - Compose a bidirectional layer with a length-aware reversal that keeps padding out of both directions.
> - Train an RNN language model with stateful truncated BPTT: contiguous lanes, detached carried state, epoch resets, and clipping.
> - (Optional) Build a biLM, mix its layers with ScalarMix, and probe them with a linear classifier.

## Before you begin

You should be comfortable with vector-Jacobian products, from [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics-for-computing/08-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md), and with gradient clipping, from [Learning-rate schedules and gradient clipping](../../mathematics-for-computing/10-optimization/04-schedules-and-gradient-clipping.md), both in *Mathematics for Computing*. The spectral radius comes from [Eigenvalues, power iteration, and the spectral radius](../../mathematics-for-computing/03-linear-algebra/04-eigenvalues-and-power-iteration.md). Everything here runs on the autograd engine, op library, and module system of [Part 0](../p00-foundations/), and the language model is compared against the models of [Part 2](../p02-statistical-lm/). For a C implementation of the LSTM cell worked through by hand, see `lstm_cell.c` in [Neural Architectures](../../machine-learning/06-neural-architectures/), whose RNN and LSTM sections this chapter absorbs.

## In this chapter

1. [Vanilla RNN with manual BPTT and truncation](01-vanilla-rnn-manual-bptt.md): backpropagation through time written by hand as reverse mode on the unrolled loop, truncated BPTT as bookkeeping on windows, and a gradient-flow diagnostic that predicts explosion and vanishing from $\rho(W_{hh})$.
2. [LSTM (torch gate order)](02-lstm.md): the additive cell state that keeps gradients alive, built in torch's $i, f, g, o$ layout with a forget bias of 1, orthogonal recurrence, stacking, dropout, and padding masks.
3. [GRU (torch gate order)](03-gru.md): one state and two gates, with the reset gate placed where torch and cuDNN put it, and the test that tells the two placements apart.
4. [Bidirectional RNN with length-aware reversal](04-bidirectional-rnn.md): two directions joined per position, with a per-sequence reversal that is its own inverse and keeps padding at the end.
5. [ELMo: biLM, ScalarMix, linear probes](05-elmo-bilm-scalarmix-probes.md) (optional): a bidirectional language model whose layers become contextual word vectors, a learned mix of them, and a deterministic linear probe.
6. [RNN language model with stateful TBPTT](06-rnn-language-model-stateful-tbptt.md): the three cells as language models, trained on one long stream with a carried, detached state, evaluated in bits per character.

## Further reading

- Olah, [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/); Karpathy, [*The Unreasonable Effectiveness of Recurrent Neural Networks*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- Hochreiter and Schmidhuber, *Long Short-Term Memory* (1997); Cho et al., [GRU](https://arxiv.org/abs/1406.1078); Peters et al., [ELMo](https://arxiv.org/abs/1802.05365).
