<!-- ol:module L3.3 -->
# GRU (torch gate order)

## Overview

| | |
|---|---|
| **Module** | `L3.3` · build · Python · Pass 4 · 3 to 4 h, plus your graded tests (rung R3) |
| **You build** | `python/tinyllm/rnn/gru.py`: `gru_cell`, `GRUCell`, `GRU` (stacked layers, inter-layer dropout, `lengths` for padded batches); and your own tests in `python/tests/l3-3-gru/`, written first |
| **Contract** | [`course/contracts/py/tinyllm/rnn/gru.pyi`](../../../course/contracts/py/tinyllm/rnn/gru.pyi) |
| **Tests** | `course/tests/L3.3/` (what they check: section 4), golden values from torch 2.14 in `course/fixtures/L3.3/gru_torch.npz`; your tests are graded by mutation, threshold 0.70 plus one required fault, with a red-then-green journal |
| **Needs** | `L0.1` `Tensor` · `L0.2` the op library · `L0.4` `Module` and `Dropout` · `M07.3` `xavier_uniform` · `M03.3` `orthogonal_init` · `M06.3` the default PCG32 stream · reading: `L3.2` the LSTM (same layout, same masking), `M01.3`, `craft.03` (or `--ref-deps`) |
| **Used by** | `L3.4` two GRUs make a bidirectional layer · later: `L3.6` `cell='gru'`, `L4.1` the default encoder and decoder cell |
| **Milestone** | `MS-L3` (the GRU language model reaches the calibrated threshold) |
| **Optional depth** | Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014); Chung et al., "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling" (2014); the cuDNN developer guide, RNN formulas |

## Key Takeaways

- A GRU has one state and two gates: the update gate $z$ interpolates, $h' = (1 - z) \odot n + z \odot h$, so $z \to 1$ copies the state and its gradient unchanged (`test_update_gate_near_one_keeps_the_state`).
- The reset gate multiplies the recurrent part **after** its bias, $n = \tanh(W_{in}x + b_{in} + r \odot (W_{hn}h + b_{hn}))$: torch's and cuDNN's form, not Cho's original, and weights trained one way do not load the other (`test_hand_example_gru_cell`, `test_matches_torch_cell`).
- Gate blocks are stacked $r, z, n$ with torch's names, so torch GRUs load by name and give torch's outputs and gradients, padded batches included (`test_matches_torch_sequence`).
- Three gate blocks instead of four: three quarters of the LSTM's parameters for a similar memory, which is why `L4.1` defaults to it.

## How to work this chapter

```bash
ol start L3.3              # stubs gru.py; prints your test path and rung (R3)
ol tests L3.3              # the course tests
# write ONE test in python/tests/l3-3-gru/, then:
ol tdd red L3.3            # must FAIL against your current code
# make it pass, then:
ol tdd green L3.3          # must PASS with the same test files
# repeat for each test; then:
ol check L3.3              # course tests, the journal, the mutation grade
```

---

## 1. Why now

The LSTM (`L3.2`) solved vanishing gradients with two states and four gates. Cho et al. found in 2014 that one state and two gates do about as well: merge the cell and hidden state, and let a single update gate decide how much of the old state to keep, with the complement going to the new candidate. Your seq2seq models of Part 4 (`L4.1`) use the GRU by default because it is a quarter smaller and a little faster. It also brings the course's most common weight-loading bug: the reset gate's position. Papers write it one way, cuDNN and torch compute it another, and a checkpoint only works with the form it was trained in. This module builds torch's form and proves it against torch.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t$, $h_t$ | input and state | `float32[B, D]`, `float32[B, H]` |
| $W_{ih}$, $W_{hh}$ | input and recurrent weights, three gate blocks stacked | `[3H, D]`, `[3H, H]` |
| $b_{ih}$, $b_{hh}$ | the two biases | `[3H]` each |
| $a = x W_{ih}^\top + b_{ih}$ | the input part of all three pre-activations | `[B, 3H]` |
| $u = h W_{hh}^\top + b_{hh}$ | the recurrent part, bias included | `[B, 3H]` |
| $r, z, n$ | reset gate, update gate, candidate | `[B, H]` each |
| $\sigma$, $\tanh$, $\odot$ | sigmoid, tanh, elementwise product | |
| $\ell_b$ | length of sequence $b$ in a padded batch | integer in $1..T$ |

**The cell.** Split $a$ and $u$ into blocks of $H$ columns, in torch's order $r, z, n$:

$$r = \sigma(a_{[0:H]} + u_{[0:H]}),\qquad z = \sigma(a_{[H:2H]} + u_{[H:2H]}),$$
$$n = \tanh\bigl(a_{[2H:3H]} + r \odot u_{[2H:3H]}\bigr),\qquad h' = (1 - z) \odot n + z \odot h.$$

$r$ decides how much of the old state the candidate may read: $r = 0$ makes $n$ a function of the input alone, a reset. $z$ decides how much of the old state survives: $h'$ is a convex combination of $n$ and $h$, so it stays in $(-1, 1)$ without an extra tanh.

**Where the reset gate goes.** Cho's paper computes $n = \tanh(W_{in}x + W_{hn}(r \odot h) + b_n)$: reset first, then the matmul. cuDNN computes $r \odot (W_{hn}h + b_{hn})$: matmul first (so the recurrent matmul for all three gates is one GEMM before any gate is known), then the reset, which also scales the recurrent bias $b_{hn}$. torch follows cuDNN. The two forms are different functions with the same parameter shapes, so nothing fails at load time; the outputs are just wrong. The worked example shows the gap: 0.635 against 0.848.

**Why it does not vanish.** $\partial h'/\partial h = \operatorname{diag}(z) + (\text{terms through } n, r, z)$. With $z \to 1$ the first term is the identity and the others vanish ($1 - z \to 0$ and $\sigma'(\cdot) \to 0$), so the state and its gradient pass through a step unchanged, the GRU's version of the LSTM's forget gate. The test saturates $z$ with a large bias and measures $\partial h_T/\partial h_0 = 1$ over 15 steps.

**Layout, initialization, stacking, padding.** Everything else follows `L3.2` with three blocks instead of four: torch's names `weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}` in that order; per layer, three Xavier input blocks (`M07.3`) then three orthogonal recurrent blocks (`M03.3`), both biases 0 (no gate here plays the forget gate's role at initialization: $z$ starts at $\sigma(0) = 0.5$); dropout between layers in training mode only; and the `lengths` mask that freezes a finished sequence's state and zeroes its outputs.

## 3. Worked example by hand

One unit, $x = 1$, $h = 0.5$. Weights: $W_{ih} = 0$, $b_{ih} = [0, \ln 3, 0]$, $W_{hh} = [0, 0, 1]^\top$ (only the candidate's recurrent part reads $h$), $b_{hh} = [0, 0, 1]$.

1. $a = [0, \ln 3, 0]$; $u = [0, 0, 0.5 \times 1 + 1] = [0, 0, 1.5]$.
2. $r = \sigma(0) = 0.5$; $z = \sigma(\ln 3) = 0.75$.
3. $n = \tanh(0 + 0.5 \times 1.5) = \tanh(0.75) = 0.635149$.
4. $h' = 0.25 \times 0.635149 + 0.75 \times 0.5 = 0.158787 + 0.375 = 0.533787$.
5. $\partial h'/\partial h = z + (1 - z)(1 - n^2)\,r\,W_{hn} = 0.75 + 0.25 \times 0.596586 \times 0.5 = 0.824573$ (the gates do not read $h$ here, so no other terms).

Cho's form would compute $n = \tanh(0 + (0.5 \times 0.5) \times 1 + 1) = \tanh(1.25) = 0.848284$; swapping the interpolation would give $h' = 0.75n + 0.25h = 0.601362$.

This is `test_hand_example_gru_cell`.

## 4. The interface

```python
# python/tinyllm/rnn/gru.py
def gru_cell(x, h, w_ih, w_hh, b_ih, b_hh) -> Tensor                     # h'
class GRUCell(Module):   # weight_ih [3H, D], weight_hh [3H, H], bias_ih, bias_hh [3H]
    def __init__(self, d_in, d_h, rng=None); def forward(self, x, h=None)
class GRU(Module):       # weight_ih_l{k}, weight_hh_l{k}, bias_ih_l{k}, bias_hh_l{k}
    def __init__(self, d_in, d_h, num_layers=1, dropout=0.0, rng=None)
    def forward(self, x, state=None, lengths=None) -> tuple[Tensor, Tensor]
    # x [T, B, d_in] -> out [T, B, H], h_n [num_layers, B, H]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_gru_cell` | unit | section 3: $h' = 0.533787$, $\partial h'/\partial h = 0.824573$ | you and the test agree on the gates |
| `test_matches_torch_cell` | golden | `torch.nn.GRUCell` outputs and every gradient | torch weights load and run |
| `test_matches_torch_sequence` | golden | one layer with a state, two layers, a packed padded batch | `L4.1` loads and trains GRUs |
| `test_state_dict_names_match_torch` | golden | torch's names, order, and shapes | the safetensors key contract |
| `test_gradcheck_gru_cell` | gradcheck | autograd through the cell against central differences | your backward is right without a torch |
| `test_update_gate_near_one_keeps_the_state` | property | saturated $z$: $h_T = h_0$ and $\partial h_T/\partial h_0 = 1$ | the reason the GRU remembers |
| `test_init_orthogonal_recurrence_and_zero_bias` | unit | zero biases, orthogonal $W_{hh}$ blocks, Xavier bound, seeding | trainable from step 0 |
| `test_lengths_keep_padding_out` | boundary | padded outputs 0; final state equals running alone; no gradient into padding | batches of sentences (`L4.1`) |
| `test_dropout_between_layers_only` | unit | dropout between layers, in training mode only | evaluation is deterministic |
| `test_default_state_and_bad_arguments` | boundary | no state means zeros; bad shapes, lengths, sizes raise | errors at the call site |

### Your graded tests (rung R3)

The given test:

```python
# python/tests/l3-3-gru/test_gru.py
import math
import numpy as np
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.gru import gru_cell

def test_hand_example_gru_cell():
    """r = 0.5, z = 0.75, n = tanh(0.5 * 1.5): h' = 0.25 n + 0.75 * 0.5."""
    w = [Tensor(a, dtype=np.float64) for a in ([[0.0]] * 3, [[0.0], [0.0], [1.0]], [0, math.log(3), 0], [0, 0, 1.0])]
    h2 = gru_cell(Tensor([[1.0]], dtype=np.float64), np.array([[0.5]]), *w)
    assert np.allclose(h2.data, [[0.25 * np.tanh(0.75) + 0.375]])
```

Then, one at a time, red then green: the cell against your own numpy formula with random weights and $b_{hn} \ne 0$ (this is the test that tells the two reset placements apart); the `state_dict` keys; orthogonal $W_{hh}$ blocks; a padded batch; a 2-layer GRU against stacking your numpy cell by hand from a nonzero $h_0$; dropout ignored by a 1-layer GRU; bad arguments and the default state. Import only `tinyllm.rnn.gru` and `tinyllm.autograd.tensor`. The required fault is pitfall 2 (Cho's placement).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the gates in the order $z, r$ | trains from scratch; torch weights give garbage | `test_hand_example_gru_cell`, `test_matches_torch_cell` (mutant `s01`) |
| 2. the reset gate in Cho's position, or applied to $W_{hn}h$ but not to $b_{hn}$ | close to torch, never equal; a loaded checkpoint loses accuracy for no visible reason | `test_hand_example_gru_cell`, `test_matches_torch_cell` (mutants `s02`, `s04`) |
| 3. the interpolation swapped, $z \odot n + (1 - z) \odot h$ | the update gate's meaning inverted: a torch checkpoint forgets what it should keep | `test_hand_example_gru_cell`, `test_update_gate_near_one_keeps_the_state` (mutant `s03`) |
| 4. padding leaking into the state or the outputs | a short sentence's encoding depends on its batch-mates | `test_lengths_keep_padding_out` (mutants `s07`, `s08`) |
| 5. a sigmoid candidate | $n$ cannot be negative; the state drifts to positive values | `test_hand_example_gru_cell` (mutant `s05`) |
| 6. stacking mistakes: dropout after the last layer, layer $k$ starting from layer $k - 1$'s state | 1-layer train and eval differ; 2-layer outputs differ from torch | `test_dropout_between_layers_only`, `test_matches_torch_sequence` (mutants `s09`, `s11`) |
| 7. part of the step computed on raw arrays (`h.data`), so $z \odot h$ carries no gradient | the forward is right and training is quietly worse: the state's own path back through time is cut | `test_gradcheck_gru_cell`, `test_hand_example_gru_cell` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | parameters are Tensors; `x[t]` and `h0[k]` are getitem |
| Back | `L0.2` | `F.matmul`, `F.transpose`, `F.sigmoid`, `F.tanh`, `F.where`, `F.stack` |
| Back | `L0.4` | `Module` registration order is the key order; `Dropout` between layers |
| Back | `M07.3` | `xavier_uniform` for the input blocks |
| Back | `M03.3` | `orthogonal_init` for the recurrent blocks |
| Back | `M06.3` | `PCG32(0).substream("init")` when no `rng` is given |
| Forward | `L3.4` | `bidirectional(GRU, GRU, x, lengths)`, the encoder of `L4.1` |
| Forward | `L3.6` | `RNNLM(cell='gru')` at MS-L3 |
| Forward | `L4.1` | the default encoder and decoder cell of the seq2seq model |

If you skip this module, `ol check L3.4` stops with `L3.4 needs L3.3`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `gru_cell` | cuDNN's GRU (`CUDNN_GRU`) | the "linear before reset" form you implemented, chosen so the recurrent GEMM runs before the gates | NVIDIA cuDNN API reference, `cudnnRNNMode_t` |
| `GRU` | `torch.nn.GRU` | packed sequences, bidirectional layers, projection sizes | `torch/nn/modules/rnn.py` |
| the original form | Keras `GRU(reset_after=False)` | Cho's placement, kept for old checkpoints; `reset_after=True` is the cuDNN form | `keras/layers/rnn/gru.py` |
| gating | minGRU (2024) | gates that read only the input, so the recurrence becomes a parallel scan | Feng et al., "Were RNNs All We Needed?" |
