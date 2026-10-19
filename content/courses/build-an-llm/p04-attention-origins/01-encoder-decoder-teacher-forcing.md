# Encoder-decoder with teacher forcing

## Overview

| | |
|---|---|
| **Module** | `L4.1` · build · Python · Pass 4 · 2 to 3 h reading |
| **Solution** | [`python/tinyllm/seq2seq/model.py`](../../../../course/solutions/python/tinyllm/seq2seq/model.py): `Seq2Seq` (`encode`, `init_state`, `decode_step`, `forward`, `greedy`), `EncoderState`, `DecoderState`, `save_seq2seq`, `load_seq2seq`; model oracle tests in [`test_seq2seq.py`](../../../../course/solutions/learner-tests/L4.1/python/tests/l4-1-seq2seq/test_seq2seq.py) |
| **Contract** | [`course/contracts/py/tinyllm/seq2seq/model.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/model.pyi) |
| **Needs** | `L3.4` `bidirectional` · `L3.3` `GRU`, `GRUCell` · `L3.2` `LSTMCell` · `L4.2` `AdditiveAttention`, `length_mask` · `L4.3` `LuongAttention` · `L0.4` `Embedding`, `Linear` · `L0.2` ops · `L0.1` `Tensor`, `no_grad` · `L0.6` safetensors I/O · `M06.3` `PCG32` · training: `L0.3` `cross_entropy`, `M10.3` `AdamW` · reading: `L3.1` |
| **Used by** | `L4.4` beam-searches a `Seq2Seq` · later `L6.7` the zoo's seq2seq rows on the dates task · end of the attention-origins stage: `{tinyllm} train seq2seq`, then `{tinyllm} translate --beam 5`, runs this model end to end |
| **Optional depth** | Sutskever, Vinyals, and Le, "Sequence to Sequence Learning with Neural Networks" (2014); Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014); Bengio et al., "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (2015) |

## Key Takeaways

- An encoder reads the source, a bridge turns its two final states into the decoder's first state, and the decoder writes the target one token at a time.
- Teacher forcing feeds the true previous token, so training is `decode_step` over the target, all steps in one graph; scheduled sampling sometimes feeds the model's own guess instead.
- Bahdanau attention reads with the state from before the step, Luong's with the state after it, plus input feeding.
- Each row's final encoder states are read at its own length, so padding never changes a sentence.
- With the same 150 steps, attention reverses digit strings almost perfectly and the bottleneck model does not.

## How to read this chapter

Sections 1 to 3 build the encoder-decoder from first principles and work a zero-weights example by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Every model so far maps a sequence to the next token of the **same** sequence. Translation, summarization, and the dates task (`"3 March 2021"` to `"2021-03-03"`) map one sequence to a **different** one, of a different length, in a different vocabulary. The encoder-decoder is the architecture that does that, and it is the setting where attention was invented: this module builds the model with and without attention, and its learning test measures the gap. Without attention the decoder sees the source only through one vector; the bottleneck test shows what that costs on a task as simple as reversing six digits.

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

From $h_0 = 0$ the encoder outputs 0 everywhere, whatever the source tokens. Let `bridge.weight` be 0 and `bridge.bias` $= (0.5, -0.5, 1, 0)$: $s_0 = \tanh(0.5, -0.5, 1, 0) = (0.462117, -0.462117, 0.761594, 0)$. Each decoder step halves it: $s_1 = (0.231059, -0.231059, 0.380797, 0)$, $s_2 = (0.115529, -0.115529, 0.190399, 0)$. With `out.weight` the first two rows of the identity and no bias, logits$_1 = (0.231059, -0.231059)$ and logits$_2 = (0.115529, -0.115529)$, whatever the target tokens. The example fixes the data path: encoder, bridge, decoder cell, output layer; dropping the bridge gives zero logits. A correct solution reproduces these numbers exactly.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example zero weights | unit | section 3: logits $s_0/2$, $s_0/4$ | the data path through the bridge |
| matches torch | golden | logits and every gradient against torch's bidirectional `nn.GRU`, `GRUCell`, `LSTMCell`, four attention configurations | the model computes what the papers describe |
| forward is the step loop | differential | `forward` equals `decode_step` over `tgt_in` | beam search and the zoo drive the same steps |
| teacher forcing ratio | unit | $\rho = 0$ feeds the model's argmax, one draw per step; $\rho = 1$ draws nothing | scheduled sampling, reproducibly |
| bahdanau reads with the previous state | unit | the query is $s_0$, then $s_1$ | Bahdanau's order |
| luong reads with the new state and feeds input | unit | the query is the new state; $\tilde h$ reaches the next input; logits read $\tilde h$ | Luong's order and input feeding |
| padding never changes a sentence | property | garbage padding (even out-of-vocabulary ids) changes nothing; a row alone equals it in a batch | batches of mixed lengths |
| gradient reaches the encoder | unit | every parameter gets a nonzero gradient | the encoder learns from the decoder's loss |
| greedy matches step loop and stops at eos | unit | greedy equals its own step loop; `eos` ends a row | the decoder `{tinyllm} translate` runs |
| save and load roundtrip | unit | config, key order, identical logits for every attention | the zoo's checkpoint contract |
| validation | boundary | odd $d_h$, unknown cell, non-attention module, bad lengths and ids | caller bugs fail loudly |
| attention learns to reverse | learning | 150 steps of AdamW on reversal: exact match at the reference within 3 sd | the model trains |
| attention beats the bottleneck | learning | same budget without attention: at least 0.2 lower exact match | the lesson of this part |

### The oracle tests

The model suite [`test_seq2seq.py`](../../../../course/solutions/learner-tests/L4.1/python/tests/l4-1-seq2seq/test_seq2seq.py) uses the model's own pieces, recombined by hand, as oracles: it runs `enc_fwd` and `enc_bwd` on one unpadded sentence at a time and checks `init` against $\tanh(W_b [\dots])$; recomputes one decoder step from `cell`, `attention`, and `out` and compares with `decode_step`; checks that `forward` equals the step loop and that free running feeds back the argmax. It adds the zero-weights example, padding with out-of-vocabulary ids, a greedy decode that must stop at once (a huge `eos` bias), and a save and load. It imports only contract modules.

## 5. The solution, step by step

### From first principles

1. **Different lengths and vocabularies force two networks.** The source is read once and in full (encoder), the target is written one token at a time conditioned on what was written so far (decoder). The encoder can look both ways; the decoder cannot look ahead.
2. **The encoder output at $t$ should know the whole sentence**, so it is `L3.4`'s bidirectional GRU with $d_h/2$ per direction, which makes the concatenated output exactly $d_h$ wide, the decoder's width. That is why $d_h$ must be even.
3. **The decoder needs a starting state that summarizes the source.** The two states that have read the whole sentence are the forward state at the last *real* token ($\ell_b - 1$) and the backward state at position 0. Reading either at the padded end $S - 1$ would summarize padding (pitfall 3). A `Linear` plus `tanh` (the bridge) maps them into the decoder's state space; the worked example shows that without it the logits would be 0.
4. **Training should not depend on the model's own mistakes.** Feeding the true previous token (teacher forcing) makes every step's input known in advance, so the whole target is one forward pass. Scheduled sampling relaxes this with one draw per step, and with $\rho = 1$ it draws nothing so seeded runs are unchanged.
5. **One step function serves training, greedy and beam search.** If `forward` is literally a loop over `decode_step`, then beam search (`L4.4`) and greedy decoding drive exactly the computation that was trained.
6. **Attention is a pluggable module with one switch.** The only difference between Bahdanau and Luong in the step is *when* the attention reads (with $s_{t-1}$ or $s_t$) and what the output layer sees, so `query_from` selects the order. Keys are projected once per sentence in `encode`.

### The code, annotated

The encoder and the bridge:

```python
def encode(self, src, src_lens):
    s = _ints(src, "src", 2)
    n = _ints(src_lens, "src_lens", 1)
    B, S = s.shape
    # ... (lengths in 1..S)
    mask = length_mask(n, S)                    # [B, S], True on real tokens
    # Padding may hold any id; read it as 0 so the embedding lookup is valid.
    s = np.where(mask, s, 0)
    # ... (ids in range)
    out = bidirectional(self.enc_fwd, self.enc_bwd, self.src_emb(s.T), n)  # [S, B, d_h]
    H2 = self.d_h // 2
    rows = np.arange(B)
    # Forward state after each sequence's own last token; backward state
    # after reading it back to the first token (position 0).
    last_f = out[n - 1, rows][:, :H2]
    first_b = out[0][:, H2:]
    init = F.tanh(self.bridge(F.concat([last_f, first_b], axis=-1)))
    keys = F.transpose(out, 0, 1)               # batch first for attention: [B, S, d_h]
    proj = self.attention.project_keys(keys) if self.attention is not None else None
    return EncoderState(keys, mask, proj, init)
```

Replacing padding ids with 0 before the embedding is what lets callers pad with anything, even out-of-vocabulary ids: the padded embeddings never reach a real position because `bidirectional` honors `lengths`, and attention masks them with `mask`.

`init_state` adds what the decoder carries beyond $h$: a zero $c$ for an LSTM cell, and a zero $\tilde h_0$ (`feed`) for Luong's input feeding. Then the step, one branch per row of the table in section 2.2:

```python
def decode_step(self, y_prev, state):
    y = _ints(y_prev, "y_prev", 1)
    # ... (ids in range)
    e = self.tgt_emb(y)
    att = self.attention
    if att is None:
        h, c = self._cell(e, state)                         # bottleneck: the source only via s_0
        return self.out(h), state._replace(h=h, c=c), None
    if att.query_from == "previous":
        # Bahdanau: read the source with the state from BEFORE this step.
        ctx, a = att(state.h, state.keys, state.mask, state.proj)
        h, c = self._cell(F.concat([e, ctx], axis=-1), state)
        logits = self.out(F.concat([h, ctx], axis=-1))      # the output sees the context too
        return logits, state._replace(h=h, c=c), a
    # Luong: step first (input feeding), then read with the NEW state.
    h, c = self._cell(F.concat([e, state.feed], axis=-1), state)
    ctx, a = att(h, state.keys, state.mask, state.proj)
    feed = att.attentional(h, ctx)                          # h~ = tanh(W_c [c ; h])
    return self.out(feed), state._replace(h=h, c=c, feed=feed), a
```

The state is an immutable `NamedTuple` updated with `_replace`, so beam search can gather it by parent index without aliasing. The shapes in `__init__` follow from this code: the cell input is `d_emb + d_h` with attention (the context or the fed $\tilde h$), and the output layer is `2 * d_h` wide only for Bahdanau.

Training is the step loop, with teacher forcing or scheduled sampling:

```python
def forward(self, src, src_lens, tgt_in, teacher_forcing=1.0, rng=None):
    tgt = _ints(tgt_in, "tgt_in", 2)
    # ... (shape and id checks; rho in [0, 1]; an rng is required when rho < 1)
    state = self.init_state(self.encode(src, src_lens))
    steps = []
    y = tgt[:, 0]                                   # bos
    for t in range(tgt.shape[1]):
        if t > 0:
            # rho = 1 short-circuits, so no draw is consumed.
            if teacher_forcing >= 1.0 or rng.uniform() < teacher_forcing:
                y = tgt[:, t]                       # the true previous token
            else:
                # The model's own guess: a choice, so no gradient through it.
                y = np.argmax(steps[-1].data, axis=-1)
        logits, state, _ = self.decode_step(y, state)
        steps.append(logits)
    return F.stack(steps, axis=1)                   # [B, T, Vt]
```

One `rng.uniform()` per step decides for the whole batch. Using `.data` for the argmax makes the fed token a plain integer array: the gradient never flows through the choice. The loss is not part of the model; the caller computes `cross_entropy` against `tgt_out` with `ignore_index` for padding.

`greedy` runs the same steps under `no_grad`, feeds back each row's argmax, appends to a row until it emits `eos`, and stops early once every row is done. `save_seq2seq` writes `config.json` (sizes, cell, and the attention kind: `none`, `bahdanau` with `tl_d_attn`, or `luong-<score>`) and the `state_dict` in registration order; `load_seq2seq` rebuilds the matching attention module and loads strictly. Full solution: [`model.py`](../../../../course/solutions/python/tinyllm/seq2seq/model.py).

### Why this solution

- **Cost.** Encoding is $O(S B d_h^2)$; each decoder step is $O(B d_h^2)$ for the cell plus $O(B S d_h)$ for attention, so a target of length $T$ costs $O(T B (d_h^2 + S d_h + d_h V_t))$. Projecting the keys once in `encode` removes an $O(T S d_h d_a)$ term.
- **Why not a unidirectional encoder?** Its last state would know the whole sentence but each $o_t$ only the left context; attention reads $o_t$, so the bidirectional encoder gives every key the full sentence.
- **Why the bridge and not zeros or a copy?** Zeros make the decoder depend on attention alone (or on nothing, without attention). Copying the encoder state forces the decoder to live in the encoder's state space; the learned `Linear` + `tanh` decouples them at a cost of $d_h^2$ parameters.
- **Why teacher forcing at all?** Free running would make early training feed garbage tokens and make the gradient depend on discrete choices. Teacher forcing gives a stable, parallel-in-time loss; the price is exposure bias, which scheduled sampling partly addresses.
- **Tradeoff accepted.** The decoder loop is in Python, one step at a time, which is slow for long targets; that is what the transformer of `L5.3` removes by attending over the whole target in parallel during training.

### Explain it back

1. Why is the forward half of the bridge read at `n - 1` while the backward half is read at `0`?
   <details><summary>Answer</summary>The forward GRU has read the whole sentence after its last real token, position $\ell_b - 1$. The backward GRU reads from $\ell_b - 1$ down to 0, so its complete state lands at position 0 once `bidirectional` puts its outputs back in time order.</details>
2. What would the zero-weights example give if `init_state` started the decoder from zeros?
   <details><summary>Answer</summary>Every GRU step halves the state, and half of zero is zero, so every logit would be 0 regardless of the source. The example's nonzero logits $s_0/2$, $s_0/4$ exist only because $s_0$ comes from the bridge.</details>
3. Why does Bahdanau's output layer take `[h ; ctx]` while Luong's takes only $\tilde h$?
   <details><summary>Answer</summary>In Bahdanau's model the context is computed before the step and joins both the cell input and the read-out. In Luong's model the context has already been folded into $\tilde h = \tanh(W_c [c ; h])$, so $\tilde h$ alone carries it.</details>
4. Why is the argmax in scheduled sampling taken on `steps[-1].data` rather than on the Tensor?
   <details><summary>Answer</summary>Choosing a token is discrete and has no gradient. Taking it from `.data` makes that explicit: the next input is a plain id array, and the graph only connects through the decoder state.</details>
5. Why must `forward` be built from `decode_step` rather than a separate batched implementation?
   <details><summary>Answer</summary>Greedy and beam search can only call `decode_step`. If training used different code, any difference (order of attention, input feeding, the output layer) would mean decoding runs a model that was never trained.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the decoder starts from zeros (bridge state dropped) | the source reaches the decoder only through attention, or not at all | `init_state` starts the decoder from `enc.init`, the bridge output |
| 2. teacher forcing fed `tgt_in[:, t - 1]` at step $t$ | the model is asked to copy its input; training loss falls, translation fails | `forward` feeds `tgt[:, t]` at step $t$ |
| 3. final encoder states read at the padded end or the wrong position | short sentences are summarized from padding | `encode` reads `out[n - 1, rows]` forward and `out[0]` backward, inside each length |
| 4. attention in the wrong place: Luong reading with the old state, Bahdanau's query not $s_{t-1}$, or its context left out of the output | a different model from the paper's; golden values disagree | `decode_step` branches on `query_from`: `att(state.h, ...)` before the cell for Bahdanau, `att(h, ...)` after it for Luong; Bahdanau's logits read `[h ; ctx]` |
| 5. no input feeding | the next step never sees $\tilde h$ | the cell input is `[e ; state.feed]`, and `feed` is replaced by $\tilde h$ each step |
| teacher forcing when $u \ge \rho$ | $\rho = 0$ still feeds the truth | `rng.uniform() < teacher_forcing` feeds the truth |

## 7. Where it's used next

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

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Seq2Seq` | OpenNMT-py, fairseq LSTM models | multi-layer stacks, dropout, copy attention, batched beam search | OpenNMT-py `onmt/models/model.py`; fairseq `models/lstm.py` |
| the bridge | `bridge` in OpenNMT | a learned map from encoder to decoder state per layer | OpenNMT-py `onmt/encoders/rnn_encoder.py` |
| teacher forcing ratio | scheduled sampling, professor forcing | curricula for $\rho$, adversarial matching of free-running dynamics | Bengio et al. 2015; Lamb et al. 2016 |
| encoder-decoder RNN | T5, BART | the same split with transformer blocks and span corruption | `L6.4`; Hugging Face `T5ForConditionalGeneration` |
