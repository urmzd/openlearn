<!-- ol:module L3.2 -->
# LSTM (torch gate order)

## Overview

| | |
|---|---|
| **Module** | `L3.2` · build · Python · Pass 4 · 4 to 5 h, plus your graded tests (rung R3) |
| **You build** | `python/tinyllm/rnn/lstm.py`: `lstm_cell`, `LSTMCell`, `LSTM` (stacked layers, inter-layer dropout, `lengths` for padded batches); and your own tests in `python/tests/l3-2-lstm/`, written first |
| **Contract** | [`course/contracts/py/tinyllm/rnn/lstm.pyi`](../../../../course/contracts/py/tinyllm/rnn/lstm.pyi) |
| **Tests** | `course/tests/L3.2/` (what they check: section 4), golden values from torch 2.14 in `course/fixtures/L3.2/lstm_torch.npz`; your tests are graded by mutation, threshold 0.70 plus one required fault, with a red-then-green journal |
| **Needs** | `L0.1` `Tensor` · `L0.2` the ops the cell is written in · `L0.4` `Module`, `load_state_dict`, `Dropout` · `M07.3` `xavier_uniform` · `M03.3` `orthogonal_init` · `M06.3` the default PCG32 stream · reading: `M01.3` sigmoid and tanh, `L3.1` BPTT, `craft.03` (or `--ref-deps`) |
| **Used by** | `L3.4` two LSTMs make a bidirectional layer · later: `L3.6` `cell='lstm'`, `L4.1` encoder option |
| **Milestone** | `MS-L3` (the LSTM language model must beat the vanilla RNN) |
| **Optional depth** | Hochreiter and Schmidhuber, "Long Short-Term Memory" (1997); Gers, Schmidhuber, Cummins, "Learning to Forget" (2000); Jozefowicz, Zaremba, Sutskever, "An Empirical Exploration of Recurrent Network Architectures" (2015); Olah, *Understanding LSTM Networks* (2015) |

## Key Takeaways

- An LSTM keeps a second state, the cell $c_t$, updated **additively**: $c_t = f_t \odot c_{t-1} + i_t \odot g_t$. Along it the gradient is multiplied only by the forget gate, so with $f$ near 1 it survives many steps (`test_cell_state_is_a_gradient_highway`).
- The four gate blocks are stacked in torch's order $i, f, g, o$, with two biases, so torch weights load by name and give torch's outputs and gradients (`test_hand_example_lstm_cell`, `test_matches_torch_cell`, `test_matches_torch_sequence`).
- Initialization matters at step 0: a forget bias of 1 makes the cell remember by default, and orthogonal recurrent blocks keep the state's scale (`test_init_forget_bias_and_orthogonal_recurrence`).
- In a padded batch each sequence stops at its own length: its outputs past it are 0 and its final state is the state after its last real token (`test_lengths_keep_padding_out`).

## How to work this chapter

```bash
ol start L3.2              # stubs lstm.py; prints your test path and rung (R3)
ol tests L3.2              # the course tests
# write ONE test in python/tests/l3-2-lstm/, then:
ol tdd red L3.2            # must FAIL against your current code
# make it pass, then:
ol tdd green L3.2          # must PASS with the same test files
# repeat for each test; then:
ol check L3.2              # course tests, the journal, the mutation grade
```

---

## 1. Why now

`L3.1` showed the vanilla RNN's problem in numbers: the gradient into a state 30 steps back is the late gradient times $W_{hh}^\top$ thirty times, and with $\rho(W_{hh}) = 0.8$ that is a factor of about $10^{-3}$. Your RNN language model (`L3.6`) will learn which character comes next but not that a quote opened a line ago must close. The LSTM fixes this with one structural change, a memory cell updated by addition instead of by a squashing matmul, plus three gates that decide what to write, what to keep, and what to show. It is also the first module whose weights must match a framework's layout exactly: torch-trained LSTMs are everywhere, and the tests load torch's weights by name. With autograd from Part 0 you write only the forward; the backward through time comes for free.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t$ | input at step $t$ | `float32[B, D]` |
| $h_t$, $c_t$ | hidden state (what the next layer sees) and cell state (the memory) | `float32[B, H]` each |
| $W_{ih}$, $W_{hh}$ | input and recurrent weights, four gate blocks stacked | `[4H, D]`, `[4H, H]` |
| $b_{ih}$, $b_{hh}$ | the two biases (torch keeps both) | `[4H]` each |
| $z_t$ | all four pre-activations, $x_t W_{ih}^\top + b_{ih} + h_{t-1} W_{hh}^\top + b_{hh}$ | `[B, 4H]` |
| $i_t, f_t, g_t, o_t$ | input gate, forget gate, candidate, output gate | `[B, H]` each |
| $\sigma(v) = 1/(1 + e^{-v})$ | the logistic sigmoid, values in $(0, 1)$ | function |
| $\odot$ | elementwise product | |
| $\ell_b$ | the length of sequence $b$ in a padded batch | integer in $1..T$ |

**The cell.** One matmul per weight computes every gate at once, and the result is split into four blocks of $H$ columns, **in torch's order**:

$$i = \sigma(z_{[0:H]}),\quad f = \sigma(z_{[H:2H]}),\quad g = \tanh(z_{[2H:3H]}),\quad o = \sigma(z_{[3H:4H]})$$
$$c_t = f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t).$$

The gates are sigmoids because they are fractions: how much of the old memory to keep ($f$), how much of the candidate to write ($i$), how much of the memory to show ($o$). The candidate $g$ is a tanh because it is content, centered at 0 and able to subtract. The output goes through one more tanh so $h$ stays in $(-1, 1)$ however large $c$ grows.

**Why it does not vanish.** Differentiate the cell update: $\partial c_t/\partial c_{t-1} = \operatorname{diag}(f_t)$, plus terms through the gates' dependence on $h_{t-1}$. Along the cell path the gradient from $c_T$ to $c_0$ is $\prod_t f_t$, elementwise, with no weight matrix and no tanh slope in it. If the network wants to remember, it sets $f \approx 1$ and the gradient flows back almost unchanged (Hochreiter's "constant error carousel"). With $W_{hh} = 0$ the cell path is the only path, and the test measures exactly $\prod_t f_t$.

**torch's layout.** `torch.nn.LSTMCell` and `torch.nn.LSTM` store `weight_ih` $[4H, D]$ and `weight_hh` $[4H, H]$ (one row per gate unit, so the products use the transpose, like `L0.4`'s `Linear`), and two biases `bias_ih` and `bias_hh`. The two biases only ever appear as a sum, so a single bias is mathematically enough, but the checkpoint has both keys and both receive the same gradient. A stacked `LSTM` names its parameters per layer: `weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, weight_ih_l1, ...`, registered in that order (`L0.4`: assignment order is `state_dict` order). Layer $k > 0$ reads layer $k - 1$'s output sequence, so its `weight_ih` is $[4H, H]$, and each layer starts from its own slice of the initial state.

**Initialization.** The contract draws, per layer, each of the four $[H, D]$ input blocks with `xavier_uniform` (`M07.3`, gain 1) and each of the four $[H, H]$ recurrent blocks with `orthogonal_init` (`M03.3`): an orthogonal matrix has every singular value 1, so the recurrence neither inflates nor shrinks the state at step 0 (the `L3.1` analysis with $\rho = 1$). The forget-gate block of $b_{ih}$ is 1 and every other bias is 0, so $f$ starts near $\sigma(1) = 0.73$: the cell remembers by default (Gers 2000, Jozefowicz 2015). torch's own default draws everything from $U(-1/\sqrt{H}, 1/\sqrt{H})$; the tests that compare with torch load torch's weights, so the initializations need not match.

**Stacking and dropout.** torch applies dropout to each layer's output sequence **except the last layer's**, and only in training mode; the dropout here is `L0.4`'s `Dropout`, so `eval()` turns it off.

**Padded batches.** Batching sequences of different lengths pads them to $T$ steps. torch's `pack_padded_sequence` makes the RNN skip padding; the equivalent here is a mask: at a step $t \ge \ell_b$, sequence $b$ keeps its $(h, c)$ unchanged (`F.where(mask, new, old)`) and outputs 0. Then the returned final state $(h_n, c_n)$ is each sequence's state after its own last real step, and nothing the padding contains reaches any output or receives any gradient.

## 3. Worked example by hand

One unit ($D = H = B = 1$), $x = 1$, $h_{t-1} = 0$, $c_{t-1} = 0.5$. Weights: $W_{ih} = [0, 0, \ln 2, 0]^\top$ (only the candidate reads $x$), $W_{hh} = 0$, $b_{ih} = [0, \ln 3, 0, \ln 3]$, $b_{hh} = 0$. So $z = [0, \ln 3, \ln 2, \ln 3]$.

1. $i = \sigma(0) = 0.5$; $f = \sigma(\ln 3) = 3/4 = 0.75$; $g = \tanh(\ln 2) = (4 - 1)/(4 + 1) = 0.6$; $o = \sigma(\ln 3) = 0.75$.
2. $c_t = 0.75 \times 0.5 + 0.5 \times 0.6 = 0.375 + 0.3 = 0.675$.
3. $h_t = 0.75 \times \tanh(0.675) = 0.75 \times 0.588259 = 0.441194$.
4. Backward from $\partial L/\partial h_t = 1$: $\partial h_t/\partial c_t = o\,(1 - \tanh^2 c_t) = 0.75 \times 0.653952 = 0.490463$, and through the cell path $\partial c_t/\partial c_{t-1} = f = 0.75$, so $\partial L/\partial c_{t-1} = 0.367847$.

With the gate order $i, f, o, g$ the same weights give $g = \tanh(\ln 3) = 0.8$ and $o = \sigma(\ln 2) = 2/3$, so $c_t = 0.775$ and $h_t = 0.433$: one swapped block, every number different.

This is `test_hand_example_lstm_cell`.

## 4. The interface

```python
# python/tinyllm/rnn/lstm.py
def lstm_cell(x, h, c, w_ih, w_hh, b_ih, b_hh) -> tuple[Tensor, Tensor]       # (h', c')
class LSTMCell(Module):   # weight_ih [4H, D], weight_hh [4H, H], bias_ih, bias_hh [4H]
    def __init__(self, d_in, d_h, rng=None); def forward(self, x, state=None)
class LSTM(Module):       # weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}
    def __init__(self, d_in, d_h, num_layers=1, dropout=0.0, rng=None)
    def forward(self, x, state=None, lengths=None) -> tuple[Tensor, tuple[Tensor, Tensor]]
    # x [T, B, d_in] -> out [T, B, H], (h_n, c_n) [num_layers, B, H]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_lstm_cell` | unit | section 3: $c' = 0.675$, $h' = 0.441194$, $\partial L/\partial c = 0.367847$ | you and the test agree on the gates |
| `test_matches_torch_cell` | golden | `torch.nn.LSTMCell` outputs and every gradient | torch weights load and run |
| `test_matches_torch_sequence` | golden | one layer with a state, two layers, a packed padded batch: outputs, $(h_n, c_n)$, all gradients | `L3.6` and `L4.1` load torch checkpoints |
| `test_state_dict_names_match_torch` | golden | torch's parameter names, order, and shapes | the safetensors key contract (`L0.6`) |
| `test_gradcheck_lstm_cell` | gradcheck | autograd through the cell against central differences, float64 | your backward is right without a torch |
| `test_cell_state_is_a_gradient_highway` | property | with $W_{hh} = 0$, $\partial c_T/\partial c_0 = \prod_t f_t$ | the reason the LSTM remembers |
| `test_init_forget_bias_and_orthogonal_recurrence` | unit | forget bias 1, others 0; orthogonal $W_{hh}$ blocks; Xavier bound; seeding | trainable from step 0 (`L3.6`) |
| `test_lengths_keep_padding_out` | boundary | padded outputs 0; final state equals running the sequence alone; padding gets no gradient | batches of sentences (`L4.1`) |
| `test_dropout_between_layers_only` | unit | dropout between layers, in training mode only | evaluation is deterministic |
| `test_default_state_and_bad_arguments` | boundary | no state means zeros; bad shapes, lengths, sizes raise | errors at the call site |

### Your graded tests (rung R3)

The given test:

```python
# python/tests/l3-2-lstm/test_lstm.py
import math
import numpy as np
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.lstm import lstm_cell

def test_hand_example_lstm_cell():
    """i = 0.5, f = 0.75, g = 0.6, o = 0.75: c' = 0.675, h' = 0.75 tanh(0.675)."""
    w = [Tensor(a, dtype=np.float64) for a in ([[0.0], [0.0], [math.log(2)], [0.0]], [[0.0]] * 4,
                                              [0, math.log(3), 0, math.log(3)], [0.0] * 4)]
    h2, c2 = lstm_cell(Tensor([[1.0]], dtype=np.float64), np.zeros((1, 1)), np.array([[0.5]]), *w)
    assert np.allclose(c2.data, [[0.675]])
    assert np.allclose(h2.data, [[0.75 * np.tanh(0.675)]])
```

Then, one at a time, red then green: the cell against your own numpy formula with random weights and both biases nonzero; the `state_dict` keys of a 2-layer LSTM; the forget bias and orthogonal $W_{hh}$ blocks; a padded batch (zeros past each length, the final state of a short sequence equal to running it alone); a 2-layer LSTM against stacking your numpy cell by hand from a nonzero $(h_0, c_0)$; dropout ignored by a 1-layer LSTM; bad arguments and the default state. Import only `tinyllm.rnn.lstm`, `tinyllm.autograd.tensor`, and `tinyllm.autograd.functional` (as `import tinyllm.autograd.functional as F`). The required fault is pitfall 1.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the gates in another order ($i, f, o, g$ is common in papers and other frameworks) | trains fine from scratch; torch weights load and produce garbage | `test_hand_example_lstm_cell`, `test_matches_torch_cell` (mutant `s01`) |
| 2. the wrong squashing: $g$ through a sigmoid, $h = o \odot c$ without the tanh, the input gate dropped, or the old cell squashed ($f \odot \tanh c$) | the cell can only add, $h$ grows without bound, or the gradient highway gains a tanh slope per step and vanishes again | `test_hand_example_lstm_cell`, `test_cell_state_is_a_gradient_highway` (mutants `s02`, `s03`, `s04`, `s15`) |
| 3. forget bias 0, or the $+1$ on the wrong block | the cell forgets half its memory per step at the start; long dependencies learned late or never | `test_init_forget_bias_and_orthogonal_recurrence` (mutants `s05`, `s06`) |
| 4. padding leaking into the state or the outputs | the final state of a short sentence depends on how long its batch-mates are | `test_lengths_keep_padding_out` (mutants `s07`, `s08`) |
| 5. one bias instead of two | the model computes the same function but a torch checkpoint has an extra key, and loading it drops half the bias | `test_matches_torch_cell` (mutant `s09`) |
| 6. stacking mistakes: dropout after the last layer, layer $k$ starting from layer $k - 1$'s final state | training and eval outputs differ in a 1-layer model; 2-layer outputs differ from torch | `test_dropout_between_layers_only`, `test_matches_torch_sequence` (mutants `s10`, `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | parameters are Tensors; `x[t]` and `h0[k]` are getitem |
| Back | `L0.2` | `F.matmul`, `F.transpose`, `F.sigmoid`, `F.tanh`, `F.where`, `F.stack` |
| Back | `L0.4` | `Module` registration order is the key order; `Dropout` between layers |
| Back | `M07.3` | `xavier_uniform` for the input blocks |
| Back | `M03.3` | `orthogonal_init` for the recurrent blocks |
| Back | `M06.3` | `PCG32(0).substream("init")` when no `rng` is given |
| Forward | `L3.4` | `bidirectional(LSTM, LSTM, x, lengths)` |
| Forward | `L3.6` | `RNNLM(cell='lstm')`, trained with stateful TBPTT; must beat the vanilla RNN at MS-L3 |
| Forward | `L4.1` | the seq2seq encoder's LSTM option |

If you skip this module, `ol check L3.4` stops with `L3.4 needs L3.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `lstm_cell` | cuDNN's fused LSTM | the input projection for all $T$ steps as one GEMM, the four gates fused into one kernel | `aten/src/ATen/native/cudnn/RNN.cpp` |
| `lengths` masking | `torch.nn.utils.rnn.pack_padded_sequence` | sorts by length and shrinks the batch as sequences end, so no padded step is computed | `torch/nn/utils/rnn.py` |
| forget bias 1 | Keras `unit_forget_bias=True` | the same default, on by default | `keras/layers/rnn/lstm.py` |
| the LSTM itself | xLSTM (2024) | exponential gating and a matrix memory, parallelizable over time | `NX-AI/xlstm` |
