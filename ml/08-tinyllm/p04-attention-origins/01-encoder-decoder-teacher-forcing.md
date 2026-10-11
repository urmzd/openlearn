<!-- ol:module L4.1 -->
# Encoder-decoder with teacher forcing

## Overview

| | |
|---|---|
| **Module** | `L4.1` · build · Python · Pass 4 · 4 to 6 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/seq2seq/model.py`: `Seq2Seq` (`encode`, `init_state`, `decode_step`, `forward`, `greedy`), `EncoderState`, `DecoderState`, `save_seq2seq`, `load_seq2seq`; and your own oracle tests in `python/tests/l4-1-seq2seq/` |
| **Contract** | [`course/contracts/py/tinyllm/seq2seq/model.pyi`](../../../course/contracts/py/tinyllm/seq2seq/model.pyi) |
| **Tests** | `course/tests/L4.1/test_seq2seq.py` (what they check: section 4), golden values from torch 2.14.1 in `course/fixtures/L4.1/seq2seq_torch.npz` (`course/oracle/L4.1/seq2seq_torch.py`), learning bars in `course/fixtures/ref-thresholds.tsv`; your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L3.4` `bidirectional` · `L3.3` `GRU`, `GRUCell` · `L3.2` `LSTMCell` · `L4.2` `AdditiveAttention`, `length_mask` · `L4.3` `LuongAttention` · `L0.4` `Embedding`, `Linear` · `L0.2` ops · `L0.1` `Tensor`, `no_grad` · `L0.6` safetensors I/O · `M06.3` `PCG32` · tests: `L0.3` `cross_entropy`, `M10.3` `AdamW` · reading: `L3.1` (or `--ref-deps`) |
| **Used by** | `L4.4` beam-searches a `Seq2Seq` in its tests · later `L6.7` the zoo's seq2seq rows on the dates task |
| **Milestone** | `MS-L4` (`{tinyllm} train seq2seq`, then `{tinyllm} translate --beam 5`) |
| **Optional depth** | Sutskever, Vinyals, and Le, "Sequence to Sequence Learning with Neural Networks" (2014); Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014); Bengio et al., "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (2015) |

## Key Takeaways

- An encoder reads the source, a bridge turns its two final states into the decoder's first state, and the decoder writes the target one token at a time (`test_hand_example_zero_weights`, `test_golden_torch`).
- Teacher forcing feeds the true previous token, so training is `decode_step` over the target, all steps in one graph; scheduled sampling sometimes feeds the model's own guess instead (`test_forward_is_the_step_loop`, `test_teacher_forcing_ratio`).
- Bahdanau attention reads with the state from before the step, Luong's with the state after it, plus input feeding (`test_bahdanau_reads_with_the_previous_state`, `test_luong_reads_with_the_new_state_and_feeds_input`).
- Each row's final encoder states are read at its own length, so padding never changes a sentence (`test_padding_never_changes_a_sentence`).
- With the same 150 steps, attention reverses digit strings almost perfectly and the bottleneck model does not (`test_attention_learns_to_reverse`, `test_attention_beats_the_bottleneck`).

## How to work this chapter

```bash
ol start L4.1              # stubs model.py; prints your test path and rung (R5)
ol tests L4.1              # the course tests
# write your oracle tests in python/tests/l4-1-seq2seq/, then:
ol check L4.1              # course tests and the mutation grade of your tests
ol check L4.1 --ref-deps   # if you skipped L3.2 to L3.4, L4.2, or L4.3
ol diff  L4.1              # after passing: your code against the reference
```

---

## 1. Why now

Every model so far maps a sequence to the next token of the **same** sequence. Translation, summarization, and the dates task of `MS-L4` (`"3 March 2021"` to `"2021-03-03"`) map one sequence to a **different** one, of a different length, in a different vocabulary. The encoder-decoder is the architecture that does that, and it is the setting where attention was invented: this module builds the model with and without attention, and its learning test measures the gap. Without attention the decoder sees the source only through one vector; the bottleneck test shows what that costs on a task as simple as reversing six digits.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_{1:S}$, $\ell_b$ | source ids, padded to $S$, real length $\ell_b$ | `int[B, S]`, `int[B]` |
| $y_{0:T-1}$ | decoder inputs `tgt_in`: `bos` then the target without its last token | `int[B, T]` |
| $d_h$ | decoder width; each encoder direction has $d_h / 2$ | `int` (even) |
| $o_t = [\overrightarrow{h}_t ; \overleftarrow{h}_t]$ | encoder output at source position $t$ (the keys) | `float32[B, S, d_h]` |
| $s_0 = \tanh(W_b [\overrightarrow{h}_{\ell - 1} ; \overleftarrow{h}_0] + b_b)$ | the bridge: the decoder's first state | `float32[B, d_h]` |
| $s_t$ | decoder state after step $t$ | `float32[B, d_h]` |
| $c_t$, $\tilde h_t$ | context and Luong's attentional state | `float32[B, d_h]` |
| $\rho$ | `teacher_forcing`, the probability of feeding the true token | `float` in $[0, 1]$ |

### 2.1 Two networks and a bridge

The **encoder** is `L3.4`'s `bidirectional` over two `L3.3` GRUs of width $d_h/2$: position $t$'s output $o_t$ sees the whole source, left context from the forward GRU and right context from the backward one. The **bridge** summarizes the source for the decoder: the forward GRU's state after the last **real** token (position $\ell_b - 1$) and the backward GRU's state after reading back to position 0, concatenated, through a `Linear` and a `tanh`. The **decoder** is a `GRUCell` (or `LSTMCell`, with $c_0 = 0$) over target embeddings, and an output `Linear` gives logits over the target vocabulary.

### 2.2 The model

One decoder step from token $y$ and state $s$, by attention type (`attention.query_from`):

| Attention | Step |
|---|---|
| none | $s' = \mathrm{cell}(E[y], s)$, logits $= W_o s' + b_o$ |
| `"previous"` (Bahdanau, `L4.2`) | $c, a = \mathrm{att}(s, o)$; $s' = \mathrm{cell}([E[y] ; c], s)$; logits $= W_o [s' ; c] + b_o$ |
| `"current"` (Luong, `L4.3`) | $s' = \mathrm{cell}([E[y] ; \tilde h], s)$; $c, a = \mathrm{att}(s', o)$; $\tilde h' = \tanh(W_c [c ; s'])$; logits $= W_o \tilde h' + b_o$ |

The parameters are registered in the order of the contract (`src_emb`, `enc_fwd`, `enc_bwd`, `bridge`, `tgt_emb`, `cell`, `attention`, `out`), and that order is the safetensors key list of a checkpoint.

### 2.3 Teacher forcing

Training maximizes $\sum_t \log p(y_{t+1} \mid y_{\le t}, x)$. **Teacher forcing** feeds the true $y_t$ at step $t$ regardless of what the model predicted, so all $T$ losses come from one forward pass and the gradient at step $t$ does not depend on the model's own mistakes. At inference the model must feed its own guesses, a mismatch called exposure bias. **Scheduled sampling** (Bengio et al. 2015) trains with $\rho < 1$: before each step after the first, one uniform $u$ is drawn for the batch; if $u < \rho$ the true token is fed, otherwise the argmax of the previous step's logits, with no gradient through the choice. $\rho = 1$ draws nothing, so a seeded run is unchanged by adding the option.

Targets are shifted by one: `tgt_in` = `bos` $y_1 \dots y_{T-1}$, and the loss compares step $t$'s logits with $y_{t+1}$ (`eos` at the end). Padding in the target is ignored with `ignore_index` (`L0.3`).

### 2.4 Where attention plugs in

Attention is a module passed to the constructor; the decoder reads its `query_from` attribute to choose the order of section 2.2. The encoder computes `attention.project_keys(o)` once per sentence and stores it in the state, so the keys are projected once, not at every step (`L4.2` section 2.4). `decode_step` returns the attention weights too, which is what an alignment plot draws.

### 2.5 Decoding

`greedy` decodes all rows at once under `no_grad`, feeding back each row's argmax and stopping a row at its first `eos`. `L4.4` replaces it with beam search over `decode_step`.

## 3. Worked example by hand

Take a GRU encoder-decoder with **every** GRU weight and bias 0. One GRU step gives $r = z = \sigma(0) = \tfrac12$ and $n = \tanh(0 + r \cdot 0) = 0$, so

$$h' = (1 - z)\, n + z\, h = \tfrac12 h .$$

From $h_0 = 0$ the encoder outputs 0 everywhere, whatever the source tokens. Let `bridge.weight` be 0 and `bridge.bias` $= (0.5, -0.5, 1, 0)$: $s_0 = \tanh(0.5, -0.5, 1, 0) = (0.462117, -0.462117, 0.761594, 0)$. Each decoder step halves it: $s_1 = (0.231059, -0.231059, 0.380797, 0)$, $s_2 = (0.115529, -0.115529, 0.190399, 0)$. With `out.weight` the first two rows of the identity and no bias, logits$_1 = (0.231059, -0.231059)$ and logits$_2 = (0.115529, -0.115529)$, whatever the target tokens. The example fixes the data path: encoder, bridge, decoder cell, output layer; dropping the bridge gives zero logits. This is `test_hand_example_zero_weights`.

## 4. The interface

```python
class EncoderState(NamedTuple): keys; mask; proj; init
class DecoderState(NamedTuple): h; c; feed; keys; mask; proj
class Seq2Seq(Module):
    def __init__(self, src_vocab, tgt_vocab, d_emb, d_h, cell="gru", attention=None, rng=None): ...
    def encode(self, src, src_lens) -> EncoderState: ...
    def init_state(self, enc: EncoderState) -> DecoderState: ...
    def decode_step(self, y_prev, state) -> tuple[Tensor, DecoderState, Optional[Tensor]]: ...
    def forward(self, src, src_lens, tgt_in, teacher_forcing=1.0, rng=None) -> Tensor: ...  # [B, T, Vt]
    def greedy(self, src, src_lens, bos, eos, max_len) -> list[list[int]]: ...
def save_seq2seq(model, dir) -> None: ...
def load_seq2seq(dir) -> Seq2Seq: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_zero_weights` | unit | section 3: logits $s_0/2$, $s_0/4$ | the data path through the bridge |
| `test_golden_torch` | golden | logits and every gradient against torch's bidirectional `nn.GRU`, `GRUCell`, `LSTMCell`, four attention configurations | your model computes what the papers describe |
| `test_forward_is_the_step_loop` | differential | `forward` equals `decode_step` over `tgt_in` | beam search and the zoo drive the same steps |
| `test_teacher_forcing_ratio` | unit | $\rho = 0$ feeds the model's argmax, one draw per step; $\rho = 1$ draws nothing | scheduled sampling, reproducibly |
| `test_bahdanau_reads_with_the_previous_state` | unit | the query is $s_0$, then $s_1$ | Bahdanau's order |
| `test_luong_reads_with_the_new_state_and_feeds_input` | unit | the query is the new state; $\tilde h$ reaches the next input; logits read $\tilde h$ | Luong's order and input feeding |
| `test_padding_never_changes_a_sentence` | property | garbage padding (even out-of-vocabulary ids) changes nothing; a row alone equals it in a batch | batches of mixed lengths |
| `test_gradient_reaches_the_encoder` | unit | every parameter gets a nonzero gradient | the encoder learns from the decoder's loss |
| `test_greedy_matches_step_loop_and_stops_at_eos` | unit | greedy equals its own step loop; `eos` ends a row | the decoder `MS-L4` runs |
| `test_save_load_roundtrip` | unit | config, key order, identical logits for every attention | the zoo's checkpoint contract |
| `test_validation` | boundary | odd $d_h$, unknown cell, non-attention module, bad lengths and ids | caller bugs fail loudly |
| `test_attention_learns_to_reverse` | learning | 150 steps of AdamW on reversal: exact match at the reference within 3 sd | the model trains |
| `test_attention_beats_the_bottleneck` | learning | same budget without attention: at least 0.2 lower exact match | the lesson of this part |

### Your graded tests (rung R5)

The oracles for your tests are the model's own pieces recombined by hand: run `enc_fwd` and `enc_bwd` on one unpadded sentence at a time and check `init` against $\tanh(W_b [\dots])$; recompute one decoder step from `cell`, `attention`, and `out` and compare with `decode_step`; check that `forward` equals the step loop and that free running feeds back the argmax. Add the zero-weights example, padding with out-of-vocabulary ids, a greedy decode that must stop at once (a huge `eos` bias), and a save and load. Import only contract modules.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the decoder starts from zeros (bridge state dropped) | the source reaches the decoder only through attention, or not at all | `test_hand_example_zero_weights` (mutant `s01`) |
| 2. teacher forcing fed `tgt_in[:, t - 1]` at step $t$ | the model is asked to copy its input; training loss falls, translation fails | `test_forward_is_the_step_loop` (mutant `s02`) |
| 3. final encoder states read at the padded end or the wrong position | short sentences are summarized from padding | `test_padding_never_changes_a_sentence` (mutant `s06`), `test_gradient_reaches_the_encoder` (mutant `s07`) |
| 4. attention in the wrong place: Luong reading with the old state, Bahdanau's query not $s_{t-1}$, or its context left out of the output | a different model from the paper's; golden values disagree | `test_luong_reads_with_the_new_state_and_feeds_input` (mutant `s04`), `test_bahdanau_reads_with_the_previous_state` (mutant `s09`), `test_golden_torch` (mutant `s03`) |
| 5. no input feeding | the next step never sees $\tilde h$ | `test_luong_reads_with_the_new_state_and_feeds_input` (mutant `s05`) |
| teacher forcing when $u \ge \rho$ | $\rho = 0$ still feeds the truth | `test_teacher_forcing_ratio` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L3.4` | `bidirectional` runs the encoder both ways inside each length |
| Back | `L3.3` | the encoder GRUs and the default decoder `GRUCell` |
| Back | `L3.2` | `LSTMCell`, the other decoder cell |
| Back | `L4.2` | `AdditiveAttention` and `length_mask` |
| Back | `L4.3` | `LuongAttention` and `attentional` |
| Back | `L0.4` | `Embedding` and `Linear` |
| Back | `L0.2` | the op library |
| Back | `L0.1` | `Tensor` and `no_grad` |
| Back | `L0.6` | `save_safetensors` and `load_safetensors` |
| Back | `M06.3` | `PCG32` for the default initialization |
| Back | `L0.3` | `cross_entropy` with `ignore_index`, in the tests |
| Back | `M10.3` | `AdamW`, in the tests |
| Forward | `L4.4` | beam search over `decode_step`, gathering `DecoderState` by parent |
| Forward | `L4.5` | exact match, BLEU, and chrF grade its outputs |
| Forward | `L6.7` | the zoo trains and evaluates seq2seq checkpoints on the dates task |
| Forward | `L5.3` | the 2017 transformer keeps the encoder-decoder and replaces both RNNs with attention |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Seq2Seq` | OpenNMT-py, fairseq LSTM models | multi-layer stacks, dropout, copy attention, batched beam search | OpenNMT-py `onmt/models/model.py`; fairseq `models/lstm.py` |
| the bridge | `bridge` in OpenNMT | a learned map from encoder to decoder state per layer | OpenNMT-py `onmt/encoders/rnn_encoder.py` |
| teacher forcing ratio | scheduled sampling, professor forcing | curricula for $\rho$, adversarial matching of free-running dynamics | Bengio et al. 2015; Lamb et al. 2016 |
| encoder-decoder RNN | T5, BART | the same split with transformer blocks and span corruption | `L6.4`; Hugging Face `T5ForConditionalGeneration` |
