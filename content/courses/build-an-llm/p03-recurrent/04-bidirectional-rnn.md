<!-- ol:module L3.4 -->
# Bidirectional RNN with length-aware reversal

## Overview

| | |
|---|---|
| **Module** | `L3.4` · build · Python · Pass 4 · 2 to 3 h, plus your graded property tests (rung R4) |
| **You build** | `python/tinyllm/rnn/bi.py`: `reversal_index`, `reverse_padded`, `bidirectional`; and your own property tests in `python/tests/l3-4-bi/` |
| **Contract** | [`course/contracts/py/tinyllm/rnn/bi.pyi`](../../../../course/contracts/py/tinyllm/rnn/bi.pyi) |
| **Tests** | `course/tests/L3.4/` (what they check: section 4), golden values from torch 2.14 in `course/fixtures/L3.4/bi_torch.npz`; your tests are graded by mutation, threshold 0.80 with every semantic fault required |
| **Needs** | `L3.2` `LSTM` and `L3.3` `GRU` (the two directions) · `L0.1` `Tensor` (the gather is getitem) · `L0.2` `F.concat` · reading: `L0.4` `Module`, `craft.04` property tests (or `--ref-deps`) |
| **Used by** | later: `L4.1` the seq2seq encoder reads each source sentence both ways |
| **Milestone** | `MS-L3` |
| **Optional depth** | Schuster and Paliwal, "Bidirectional Recurrent Neural Networks" (1997); Graves and Schmidhuber, "Framewise phoneme classification with bidirectional LSTM" (2005); the `torch.nn.utils.rnn` documentation |

## Key Takeaways

- A bidirectional layer is two recurrent modules: one reads left to right, the other right to left, and their outputs are concatenated, forward half first, so position $t$ sees the whole sentence (`test_hand_example_running_sums`, `test_matches_torch_bidirectional`).
- With padding, "right to left" means from each sequence's **own** last real token: reverse inside each length with the index $\ell_b - 1 - t$ and leave the padding where it is; a plain `x[::-1]` makes the backward RNN read padding first (`test_hand_example_reversal`, `test_padding_never_reaches_either_direction`).
- The reversal is an involution, so the same gather puts the backward outputs back in time order, and on a Tensor its gradient is the same permutation (`test_reversal_is_an_involution_that_keeps_padding`).
- The batched computation equals running every sequence alone, unpadded, which is the property that makes it trustworthy (`test_matches_per_sequence_loop`).

## How to work this chapter

```bash
ol start L3.4              # stubs bi.py; prints your test path and rung (R4)
ol tests L3.4              # the course tests
# write the properties of section 4 as tests in python/tests/l3-4-bi/, then:
ol check L3.4              # course tests and the mutation grade of your tests
ol mutate L3.4             # the full grade, cached by your test files' hash
```

---

## 1. Why now

Your LSTM and GRU read a sentence left to right, so the state at word 3 knows words 0 to 3 and nothing after. For a language model that is the point: it must not see the future. For an **encoder** it is a handicap: when `L4.1` encodes a source sentence to translate it, the representation of "bank" should already know whether "river" or "account" follows. Reading the sentence in both directions and concatenating gives every position both contexts. The idea is one line; the bug is in the batching. Sentences in a batch have different lengths and are padded at the end, and the backward direction must start at each sentence's last real word, not at the padding. This module builds the length-aware reversal once, proves it against torch's packed sequences and against a per-sentence loop, and gives `L4.1` its encoder.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T, B$ | padded length, batch size | integers |
| $x$ | a padded batch, time first | `[T, B, D]` |
| $\ell_b$ | the real length of sequence $b$; steps $t \ge \ell_b$ are padding | integer in $1..T$ |
| $\pi_b(t)$ | the reversal index: $\ell_b - 1 - t$ for $t < \ell_b$, else $t$ | integer in $0..T-1$ |
| $R(x)$ | the length-aware reversal, $R(x)[t, b] = x[\pi_b(t), b]$ | `[T, B, ...]` |
| $\overrightarrow{h}_t$, $\overleftarrow{h}_t$ | forward and backward states at position $t$ | `[B, H_f]`, `[B, H_b]` |

**Reversal inside each length.** For sequence $b$ the real tokens are $x[0..\ell_b - 1, b]$. Reading them backward means position $t$ takes token $\ell_b - 1 - t$ for $t < \ell_b$; positions past the length keep their padding. That map $\pi_b$ is a permutation of $0..T-1$ that mirrors the first $\ell_b$ positions and fixes the rest, so applying it twice gives the identity: $R(R(x)) = x$. One integer index array of shape $[T, B]$ holds all $B$ permutations, and the gather `x[idx, arange(B)]` applies them at once; on a `Tensor` it is `L0.1`'s getitem, whose backward scatters each gradient back to where its value came from, which is the same permutation again.

**Why not `x[::-1]`.** Flipping the padded batch reverses every sequence over all $T$ steps: a sequence of length 2 in a batch of length 5 becomes three padding steps followed by its two real tokens. The backward RNN would start from padding, and even though `L3.2`'s and `L3.3`'s `lengths` mask makes them ignore steps past the length, after a full flip the padding is no longer past the length, it is at the front. Its state would be garbage before the first real token.

**The layer.** With `fwd` and `bwd` single-direction modules that honor `lengths` (output 0 and carry the state past each length):

$$\text{out}_f = \texttt{fwd}(x), \qquad \text{out}_r = \texttt{bwd}(R(x)), \qquad \text{out} = [\,\text{out}_f \,;\, R(\text{out}_r)\,] \text{ on the last axis.}$$

$\text{out}_r$ is in reversed time (its row $t$ is the backward state after reading $\ell_b - 1, \dots, \ell_b - 1 - t$), so it is reversed back before the concatenation; padded positions are 0 in both halves because $R$ fixes them. The forward half comes first, as in torch's `bidirectional=True`, whose `*_l0` weights are the forward module and `*_l0_reverse` weights the backward one. Each direction's final state is in `out`: the forward one at $[\ell_b - 1, b, :H_f]$ (its last real step) and the backward one at $[0, b, H_f:]$ (it ends on the first token).

## 3. Worked example by hand

$T = 3$, three sequences with lengths $\ell = [3, 1, 2]$: "abc", "d", "ef" (a dot is padding).

| $t$ | seq 0 ($\ell = 3$) | seq 1 ($\ell = 1$) | seq 2 ($\ell = 2$) |
|---|---|---|---|
| 0 | a | d | e |
| 1 | b | . | f |
| 2 | c | . | . |

$\pi_0 = [2, 1, 0]$, $\pi_1 = [0, 1, 2]$ (a one-token sequence is its own reverse, and both padding steps stay), $\pi_2 = [1, 0, 2]$. As a $[T, B]$ array, `reversal_index([3, 1, 2], 3)` $= [[2, 0, 1], [1, 1, 0], [0, 2, 2]]$, and the reversed batch reads "cba", "d..", "fe.". `x[::-1]` would read "cba", "..d", ".fe".

**A bidirectional layer you can compute.** Take as both modules a running sum: output $t$ is the sum of the inputs up to $t$, 0 past the length. For the sequence $[1, 2, 3]$ the forward half is the prefix sums $[1, 3, 6]$. The backward module reads $[3, 2, 1]$, outputs $[3, 5, 6]$, and reversing that back gives $[6, 5, 3]$: the suffix sums, so position $t$ holds the sum from $t$ to the end. For a second sequence $[4]$ padded with two 99s, both halves are $[4, 0, 0]$: the 99s never enter.

These are `test_hand_example_reversal` and `test_hand_example_running_sums`.

## 4. The interface

```python
# python/tinyllm/rnn/bi.py
def reversal_index(lengths, T: int) -> NDArray                  # int64 [T, B]
def reverse_padded(x, lengths)                                  # Tensor or ndarray [T, B, ...], same kind back
def bidirectional(fwd: Module, bwd: Module, x: Tensor, lengths=None) -> Tensor   # [T, B, H_f + H_b]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_reversal` | unit | section 3: the index array and "cba", "d..", "fe." | you and the test agree on the reversal |
| `test_hand_example_running_sums` | unit | prefix sums forward, suffix sums backward, padding 0, forward half first | you and the test agree on the layer |
| `test_reversal_is_an_involution_that_keeps_padding` | property | $R(R(x)) = x$; padding fixed; the Tensor gradient is the same permutation | the backward outputs come back in order; gradients reach the right tokens |
| `test_no_lengths_is_a_plain_flip` | unit | without padding, `x[::-1]` | the unpadded special case |
| `test_matches_torch_bidirectional` | golden | `bidirectional=True` GRU and LSTM over packed batches: outputs and every gradient | `L4.1` loads torch encoders |
| `test_matches_per_sequence_loop` | differential | the batch equals each sequence run alone, both directions | batching changes nothing |
| `test_padding_never_reaches_either_direction` | boundary | garbage padding changes no output, gets no gradient; final states at $\ell_b - 1$ and 0 | encoder states for the decoder (`L4.1`) |
| `test_bad_lengths` | boundary | length 0, longer than $T$, wrong count, floats | errors at the batcher |

### Your graded tests (rung R4)

At rung R4 the course gives you properties in prose, and you write them as property tests (Hypothesis, `craft.04`) plus the examples they need. Every semantic fault is required, so each property matters:

1. **Involution.** For random $T$ and lengths in $1..T$, `reverse_padded(reverse_padded(x, n), n)` equals `x`.
2. **Mirror and fix.** For each $b$, the first $\ell_b$ rows of `reverse_padded(x, n)[:, b]` are `x[:ℓ_b, b]` reversed and the rest equal `x[ℓ_b:, b]`.
3. **Plain flip.** With `lengths=None`, the result is `x[::-1]`.
4. **Per-sequence equivalence.** With two `GRU`s, `bidirectional` equals running each sequence alone: the forward half from `fwd(x[:ℓ_b, b])`, the backward half from `bwd` on the reversed sequence, reversed back; padded rows are 0.
5. **Padding is inert.** Changing the values in the padding changes no output.
6. **Bad lengths raise.**

Import only `tinyllm.rnn.bi`, `tinyllm.rnn.gru` (or `lstm`), and `tinyllm.autograd.tensor`.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. reversing the padded batch with `x[::-1]`, or mirroring the padding too | the backward direction reads padding before the sentence; short sentences get worse encodings than long ones | `test_hand_example_reversal`, `test_reversal_is_an_involution_that_keeps_padding` (mutants `s01`, `s04`) |
| 2. not reversing the backward outputs back | position $t$'s backward half describes position $\ell_b - 1 - t$ | `test_hand_example_running_sums`, `test_matches_torch_bidirectional` (mutant `s02`) |
| 3. concatenating backward first | torch encoders load but their output halves are swapped for the decoder | `test_hand_example_running_sums`, `test_matches_torch_bidirectional` (mutant `s03`) |
| 4. running a direction without `lengths` | padded outputs are no longer 0, and the backward state at the end of a short sequence has read padding | `test_padding_never_reaches_either_direction`, `test_matches_per_sequence_loop` (mutants `s05`, `s06`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L3.2` | `LSTM` with `lengths` is one direction |
| Back | `L3.3` | `GRU` with `lengths` is one direction |
| Back | `L0.1` | the reversal is a `Tensor` getitem with two integer index arrays |
| Back | `L0.2` | `F.concat` joins the halves |
| Forward | `L4.1` | the seq2seq encoder: a bidirectional GRU over the source sentence, whose final states initialize the decoder |
| Forward | `L4.2` | Bahdanau attention reads the encoder's per-position outputs, both halves |

If you skip this module, `ol check L4.1` stops with `L4.1 needs L3.4`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `reverse_padded` | `torch.nn.utils.rnn.pack_padded_sequence` | no padded step is computed at all: sequences are sorted by length and the batch shrinks as they end | `torch/nn/utils/rnn.py` |
| `bidirectional` | `torch.nn.LSTM(bidirectional=True)` and cuDNN | both directions in one call, sharing the input GEMM | `aten/src/ATen/native/RNN.cpp` |
| a bidirectional encoder | BERT (`L6.2`) | bidirectional context by attention instead of recurrence, every position at once | `transformers/models/bert/modeling_bert.py` |
| reversal index | Keras `go_backwards` with masking | the same mask-aware reversal for masked sequences | `keras/layers/rnn/bidirectional.py` |
