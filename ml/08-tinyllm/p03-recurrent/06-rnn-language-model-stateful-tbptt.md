<!-- ol:module L3.6 -->
# RNN language model with stateful TBPTT

## Overview

| | |
|---|---|
| **Module** | `L3.6` · build · Python · Pass 4 · 4 to 5 h, plus your graded property tests (rung R4) |
| **You build** | `python/tinyllm/rnn/rnnlm.py`: `ElmanRNN` (L3.1's forward and backward as one autograd op), `RNNLM` (`forward`, `init_state`, `detach_state`, `nll`, `generate`), `tbptt_batches`, `train_tbptt`, `save_rnnlm`, `load_rnnlm`; and your own property tests in `python/tests/l3-6-rnnlm/` |
| **Contract** | [`course/contracts/py/tinyllm/rnn/rnnlm.pyi`](../../../course/contracts/py/tinyllm/rnn/rnnlm.pyi) |
| **Tests** | `course/tests/L3.6/test_rnnlm.py` (what they check: section 4); the learning test trains on `course/fixtures/L3.6/corpus.txt` (an original synthetic text, `course/oracle/L3.6/corpus.py`) against a bar in `course/fixtures/ref-thresholds.tsv`; your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L3.1` `rnn_forward`, `rnn_backward` · `L3.2` `LSTM` · `L3.3` `GRU` · `L0.5` `train_step` · `L0.3` `cross_entropy` · `L0.4` `Embedding`, `Linear` · `L0.2` ops · `L0.1` `Tensor`, `from_op`, `no_grad` · `L0.6` safetensors I/O · `M07.3` `xavier_uniform` · `M03.3` `orthogonal_init` · `M06.3` `PCG32` · tests: `L1.1` `CharTokenizer`, `M10.3` `AdamW` · reading: `M10.4` (clipping), `M11.2` (bits per character) (or `--ref-deps`) |
| **Used by** | later: `L6.7` the zoo's `rnnlm` bits-per-byte rows (joins the registry with B7) |
| **Milestone** | `MS-L3` (`{tinyllm} train rnnlm --cell {rnn,lstm,gru}`: LSTM and GRU at the calibrated bar, RNN worse than LSTM) |
| **Optional depth** | Mikolov et al., "Recurrent neural network based language model" (Interspeech 2010); Zaremba, Sutskever, and Vinyals, "Recurrent Neural Network Regularization" (2014), section 4; Williams and Peng, "An efficient gradient-based algorithm for on-line training of recurrent network trajectories" (1990) |

## Key Takeaways

- An RNN language model is an embedding, a recurrent layer, and a linear read-out; the three cells share one interface, and the Elman cell is `L3.1`'s hand-written BPTT wrapped as a single autograd op (`test_elman_is_one_fused_op_over_l3_1`).
- Stateful training cuts the stream into contiguous lanes and walks them window by window: the hidden state flows from one window to the next (`test_hand_example_windows`, `test_state_carries_across_calls`).
- The gradient stops at the window boundary: the carried state is **detached**, not reset (`test_gradient_is_cut_at_the_window_boundary`, `test_state_flows_into_the_next_window`).
- Each new epoch starts from zeros, and clipping guards every step (`test_epoch_wraps_and_resets_the_state`, `test_clip_is_applied`).
- A character LSTM trained this way for 150 steps beats the unigram entropy of the text by more than half a bit per character (`test_lstm_learns_characters`).

## How to work this chapter

```bash
ol start L3.6              # stubs rnnlm.py; prints your test path and rung (R4)
ol tests L3.6              # the course tests
# write the properties of section 4 as tests in python/tests/l3-6-rnnlm/, then:
ol check L3.6              # course tests and the mutation grade of your tests
ol diff  L3.6              # after passing: your code against the reference
```

---

## 1. Why now

The n-gram model (`L2.1`) and the NPLM (`L2.2`) predict the next token from a fixed window: whatever happened 20 characters ago is invisible to them. You now have recurrent layers (`L3.1` to `L3.3`) whose state can, in principle, carry information across any distance. This module turns them into a language model and trains it the way recurrent LMs were trained in practice: on one long stream, with a state that persists across training windows. The result is the first model in the zoo (`L6.7`) whose context is not bounded by its architecture, and `MS-L3` compares its bits per byte with the NPLM's on the same text.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$, $d_e$, $d_h$ | vocabulary size, embedding width, hidden width | `int` |
| $x_t$ | token id at position $t$ | `int` |
| $E$ | embedding table | `float32[V, d_e]` |
| $h_t$ | recurrent state after reading $x_t$ ($h$ and $c$ for an LSTM) | `float32[n_layers, B, d_h]` |
| $W_o$, $b_o$ | read-out | `float32[V, d_h]`, `[V]` |
| $N$ | stream length | `int` |
| $B$ | number of lanes (`batch`) | `int` |
| $L = \lfloor N / B \rfloor$ | lane length | `int` |
| $k$ | window length (truncation) | `int` |
| $\mathrm{bpc}$ | bits per character: mean NLL in nats divided by $\ln 2$ | `float` |

### 2.1 Three cells, one interface

$$h_t = \mathrm{cell}(E[x_t], h_{t-1}), \qquad \text{logits}_t = W_o h_t + b_o, \qquad \mathcal{L} = \frac{1}{T}\sum_t -\log \operatorname{softmax}(\text{logits}_t)[x_{t+1}] .$$

`cell` is `L3.2`'s `LSTM`, `L3.3`'s `GRU`, or `ElmanRNN`, all with the same `forward(x, state) -> (out, state)`. `ElmanRNN` is the vanilla $h_t = \tanh(x_t W_{ih}^\top + h_{t-1} W_{hh}^\top + b_{ih} + b_{hh})$ in torch's parameter names; its forward over a whole sequence is **one** autograd node built with `from_op`: forward calls `L3.1`'s `rnn_forward`, backward calls `rnn_backward` on the saved cache. That is how cuDNN runs recurrent layers (one fused kernel per layer, not one graph node per time step), and it makes your hand-written BPTT the engine of a real model. Two details: L3.1 uses $W_{xh} = W_{ih}^\top$, so the weight gradients are transposed back; the two biases are summed in the forward, so each receives the full bias gradient.

### 2.2 Stateful training

A long text could be backpropagated through in one piece only with memory proportional to its length. **Truncated BPTT** runs windows of $k$ tokens. The key choice is what state starts each window:

- **Stateless**: zeros every time. Simple, but the model never sees context from before the window, so it cannot learn dependencies longer than $k$.
- **Stateful**: the state the previous window ended with. The forward pass sees unbounded context; only the **gradient** is truncated at the window start.

Stateful training needs the state to be a **value** at the boundary: `detach_state` keeps the arrays and drops the graph, so backward on window $j + 1$ stops there instead of running on into window $j$'s graph (whose gradients were already applied).

### 2.3 Lanes and windows

For the carried state to make sense, row $b$ of window $j + 1$ must continue row $b$ of window $j$. So the stream is cut into $B$ **contiguous lanes**, lane $b$ = `stream[b L : (b + 1) L]` (the tail $N - BL$ is dropped), and window $j$ is columns $[jk, jk + k)$ of every lane as inputs, shifted by one as targets. A window needs $k + 1$ tokens, so the windows start at $p = 0, k, 2k, \dots$ while $p + k + 1 \le L$. After the last window the lanes start over and the state is reset to zeros: carrying the end of a lane into its own beginning would condition text on what never precedes it.

### 2.4 Clipping

The exploding gradients `L3.1` diagnosed are real in training: one bad window can produce a gradient that throws the weights far away. `train_tbptt` passes `clip` to `L0.5`'s `train_step`, which rescales the global gradient norm to at most `clip` (`M10.4`) before every step.

### 2.5 Evaluation and sampling

`nll(ids)` scores a held-out stream in one lane, chunk by chunk under `no_grad`, carrying the state, so its result does not depend on the chunk size; the mean divided by $\ln 2$ is bits per character, and per byte with `M11.2`'s accumulator. `generate` feeds each sampled token back in, drawing from `PCG32(seed)` exactly as `L0.5`'s bigram sampler does.

## 3. Worked example by hand

The stream $0, 1, \dots, 19$, $B = 2$ lanes, $k = 3$. $L = 10$: lane 0 is $0 \dots 9$, lane 1 is $10 \dots 19$. A window needs $k + 1 = 4$ tokens, so it starts at $p = 0, 3, 6$ ($p = 9$ would need token 12 of a 10-token lane):

| window | inputs | targets |
|---|---|---|
| 0 | `[[0 1 2] [10 11 12]]` | `[[1 2 3] [11 12 13]]` |
| 1 | `[[3 4 5] [13 14 15]]` | `[[4 5 6] [14 15 16]]` |
| 2 | `[[6 7 8] [16 17 18]]` | `[[7 8 9] [17 18 19]]` |

The state after reading `0 1 2` starts the read of `3 4 5`: lane 0 continues itself. Tokens 9 and 19 are only targets. With interleaved lanes (`reshape(L, B).T`), lane 0 would be $0, 2, 4, \dots$ and the state would carry across text that is not contiguous. This is `test_hand_example_windows`.

## 4. The interface

```python
class ElmanRNN(Module):
    def __init__(self, d_in, d_h, num_layers=1, rng=None): ...
    def forward(self, x, state=None, lengths=None) -> tuple[Tensor, Tensor]: ...
class RNNLM(Module):
    def __init__(self, vocab, d_emb, d_h, cell, n_layers=1, rng=None): ...
    def forward(self, ids, state=None) -> tuple[Tensor, Any]: ...    # ids [B, T] -> logits [B, T, V]
    def init_state(self, batch): ...; def detach_state(self, state): ...
    def nll(self, ids, chunk=256) -> NDArray: ...; def generate(self, prefix, n, temperature, seed) -> list[int]: ...
def tbptt_batches(stream, k, batch) -> list[tuple[NDArray, NDArray]]: ...
def train_tbptt(model, stream, k, batch, opt, clip, steps) -> list[float]: ...
def save_rnnlm(model, dir, tokenizer="bytes") -> None: ...; def load_rnnlm(dir) -> RNNLM: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_windows` | unit | section 3's three windows | the lanes the state flows along |
| `test_windows_drop_the_tail_and_validate` | boundary | the tail is dropped; too-short lanes raise | no empty epochs |
| `test_elman_is_one_fused_op_over_l3_1` | differential | output and every gradient equal `rnn_forward` and `rnn_backward` | your BPTT drives a real model |
| `test_gradcheck_rnnlm` | gradcheck | every parameter of a 2-layer model, Elman and LSTM cells, float64 | the model trains through all its layers |
| `test_state_carries_across_calls` | property | two calls with the carried state equal one call | the definition of stateful |
| `test_gradient_is_cut_at_the_window_boundary` | differential | step 2's gradients equal window 2 alone from a constant state | truncation is where you put it |
| `test_state_flows_into_the_next_window` | differential | step 2's loss equals the run from window 1's state, not from zeros | context crosses windows |
| `test_epoch_wraps_and_resets_the_state` | unit | step `len(windows)` repeats step 0 exactly | each epoch starts clean |
| `test_clip_is_applied` | unit | the optimizer sees a gradient norm of at most `clip` | the guard against explosions |
| `test_nll_does_not_depend_on_the_chunk` | property | chunk 1, 7, 256 agree and equal one pass | evaluation of long held-out text |
| `test_generate_is_seeded_and_greedy_is_argmax` | unit | same seed, same ids; greedy feeds back the argmax | `{tinyllm} generate` in `MS-L3` |
| `test_save_load_roundtrip` | unit | `config.json`, torch key names, identical logits | the zoo's checkpoint contract |
| `test_validation` | boundary | unknown cell, bad ids, lengths for the Elman layer, negative steps | caller bugs fail loudly |
| `test_lstm_learns_characters` | learning | 150 steps: held-out bpc at the reference bar and 0.5 below the unigram entropy | the model learns |

### Your graded tests (rung R4)

Rung R4 grades **properties**: write them as tests that hold for any seed. The ones this module rests on: two calls with a carried state equal one call; `nll` does not depend on the chunk; with an optimizer that only records gradients (a small class with `zero_grad` and `step`), step 2's gradients equal those of window 2 from a constant state, the step `len(windows)` loss equals step 0's, and every recorded gradient norm is at most `clip`. Add the section 3 windows, the Elman layer against `rnn_forward`, greedy generation, and a save and load. Import only contract modules.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the carried state not detached | backward runs into the previous window's graph again; gradients double-count | `test_gradient_is_cut_at_the_window_boundary` (mutant `s06`) |
| 2. "detaching" by resetting to zeros every window | stateless training: no context beyond $k$ | `test_state_flows_into_the_next_window` (mutant `s07`) |
| 3. interleaved lanes | the state carries across non-contiguous text | `test_hand_example_windows` (mutant `s01`) |
| 4. no clipping | one exploding window ruins the run | `test_clip_is_applied` (mutant `s09`) |
| bias gradient to `bias_ih` only | `bias_hh` never trains | `test_elman_is_one_fused_op_over_l3_1` (mutant `s03`) |
| weight gradients not transposed back | wrong updates (or a crash when $d_e \ne d_h$) | `test_elman_is_one_fused_op_over_l3_1` (mutant `s04`) |
| `forward` ignores its state | no statefulness at all | `test_state_carries_across_calls` (mutant `s05`) |
| no reset at the epoch start | the first window is conditioned on the lane's end | `test_epoch_wraps_and_resets_the_state` (mutant `s08`) |
| `nll` restarting the state per chunk | the score depends on the chunk size | `test_nll_does_not_depend_on_the_chunk` (mutant `s10`) |
| `generate` feeding back the wrong token | greedy text differs from the argmax loop | `test_generate_is_seeded_and_greedy_is_argmax` (mutant `s11`) |
| `tokenizer` ignored by `save_rnnlm` | the zoo loads the wrong tokenizer | `test_save_load_roundtrip` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L3.1` | `rnn_forward` and `rnn_backward` are the Elman layer's forward and backward |
| Back | `L3.2` | `LSTM`, `cell = "lstm"` |
| Back | `L3.3` | `GRU`, `cell = "gru"` |
| Back | `L0.5` | `train_step` runs each window (zero_grad, backward, clip, step) |
| Back | `L0.3` | `cross_entropy` over the window |
| Back | `L0.4` | `Embedding` and `Linear` |
| Back | `L0.2` | `F.transpose`, `F.stack`, `F.reshape` |
| Back | `L0.1` | `Tensor`, `from_op`, `no_grad` |
| Back | `L0.6` | `save_safetensors` and `load_safetensors` for the model directory |
| Back | `M07.3` | `xavier_uniform` for the Elman input weights |
| Back | `M03.3` | `orthogonal_init` for the Elman recurrent weights |
| Back | `M06.3` | `PCG32` for initialization and sampling |
| Back | `L1.1` | `CharTokenizer` encodes the corpus in the learning test |
| Back | `M10.3` | `AdamW` trains the learning test |
| Forward | `L6.7` | the zoo trains `rnnlm` checkpoints and reports their bits per byte next to the n-gram and the transformer |
| Forward | `L3.5` | ELMo's biLM is two of these language models, one per direction |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `train_tbptt` | PyTorch's word language model example | the same `batchify` lanes and `repackage_hidden` detach | `pytorch/examples`, `word_language_model/main.py` |
| `ElmanRNN` fused op | cuDNN RNN kernels | one kernel per layer, weights packed for both directions | `torch/nn/modules/rnn.py`, `_VF.rnn_tanh` |
| LSTM LM | AWD-LSTM | weight-dropped hidden matrices, variational dropout, averaged SGD | Merity, Keskar, and Socher (2017), `salesforce/awd-lstm-lm` |
| truncation | RWKV, Mamba | recurrent models trained in parallel over the whole sequence, run as RNNs at inference | Peng et al. 2023; Gu and Dao 2023 |
