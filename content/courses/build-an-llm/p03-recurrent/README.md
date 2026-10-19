# Part 3: Recurrent Networks

Sequence models with state. A vanilla RNN with backpropagation through time written by hand, then the LSTM and GRU in torch's gate order (so weights load from torch checkpoints), bidirectional RNNs with length-aware reversal, and an RNN language model trained with stateful truncated BPTT. ELMo's bidirectional LM and scalar mix is optional.

**Course passes**: L3.1 to L3.6 belong to [Pass 4](../paths/course-p04-sequence-models/milestone.md)

**Before you start**: the matrix-calculus VJPs ([M08.3](../../mathematics-for-computing/08-matrix-calculus-and-autodiff/)) and gradient clipping ([M10.4](../../mathematics-for-computing/10-optimization/)); the autograd engine of [Part 0](../p00-foundations/).

## Key ideas

- **BPTT** is reverse mode through time: the gradient of an early state is a product of Jacobians, which vanishes or explodes; clipping and gating are the cures.
- **Gates are learned multiplexers**: the LSTM's forget gate lets gradient flow through the cell state almost unchanged.
- **Truncation is a memory budget**: stateful TBPTT carries the hidden state across batches but cuts the gradient every $k$ steps.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L3.1` | Vanilla RNN with **manual** BPTT and truncation | build | 4 |
| `L3.2` | LSTM (torch gate order i, f, g, o) | build | 4 |
| `L3.3` | GRU (torch gate order r, z, n) | build | 4 |
| `L3.4` | Bidirectional RNN with length-aware reversal | build | 4 |
| `L3.5` | ELMo: biLM, ScalarMix, linear probes | build | 4, optional |
| `L3.6` | RNN language model with stateful TBPTT | build | 4 |

The worked example for `L3.2` is `lstm_cell.c` from [Neural Architectures](../../machine-learning/06-neural-architectures/), whose RNN and LSTM sections these chapters absorb.

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L3.1` | [Vanilla RNN with manual BPTT and truncation](01-vanilla-rnn-manual-bptt.md) | build | 4 |
| 2 | `L3.2` | [LSTM (torch gate order)](02-lstm.md) | build | 4 |
| 3 | `L3.3` | [GRU (torch gate order)](03-gru.md) | build | 4 |
| 4 | `L3.4` | [Bidirectional RNN with length-aware reversal](04-bidirectional-rnn.md) | build | 4 |
| 5 | `L3.5` | [ELMo: biLM, ScalarMix, linear probes](05-elmo-bilm-scalarmix-probes.md) | side | 4 |
| 6 | `L3.6` | [RNN language model with stateful TBPTT](06-rnn-language-model-stateful-tbptt.md) | build | 4 |

## Going further

- Olah, [*Understanding LSTM Networks*](https://colah.github.io/posts/2015-08-Understanding-LSTMs/); Karpathy, [*The Unreasonable Effectiveness of Recurrent Neural Networks*](https://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- Hochreiter and Schmidhuber, *Long Short-Term Memory* (1997); Cho et al., [GRU](https://arxiv.org/abs/1406.1078); Peters et al., [ELMo](https://arxiv.org/abs/1802.05365).
