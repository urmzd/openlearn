# Encoder-decoder with teacher forcing

Every model in the book so far maps a sequence to the next token of the **same** sequence. Translation, summarization, and the dates task (`"3 March 2021"` to `"2021-03-03"`) map one sequence to a **different** one, of a different length, in a different vocabulary. The encoder-decoder is the architecture that does that, and it is the setting where attention was invented. In this section you build the model both with and without attention, and its learning test measures the gap. Without attention the decoder sees the source only through one vector, and the bottleneck test shows what that costs on a task as simple as reversing six digits.

The pieces come from the previous chapter and the foundations. The encoder is the length-aware [bidirectional RNN](../09-recurrent-networks/04-bidirectional-rnn.md) over two [GRUs](../09-recurrent-networks/03-gru.md); the decoder cell is a `GRUCell` or the [LSTM](../09-recurrent-networks/02-lstm.md)'s `LSTMCell`; the attention modules are [Bahdanau additive attention](02-bahdanau-additive-attention.md) (which also supplies `length_mask`) and [Luong attention](03-luong-attention-input-feeding.md), the next two sections. Underneath sit `Embedding` and `Linear` from [the module system](../06-numerics-and-autograd/05-module-system-and-layers.md), [the op library](../06-numerics-and-autograd/03-op-library-and-gradcheck.md), `Tensor` and `no_grad` from [the tensor section](../06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md), [safetensors checkpoints](../06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md), and the `PCG32` generator from [Modular arithmetic, hashing, and PCG32](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics*. Training uses `cross_entropy` from [Losses with a fused backward](../06-numerics-and-autograd/04-losses-with-fused-backward.md) and [AdamW](../../mathematics/16-optimization/03-adam-and-adamw.md). The [vanilla RNN](../09-recurrent-networks/01-vanilla-rnn-manual-bptt.md) is useful background reading.

The solution lives in [`python/tinyllm/seq2seq/model.py`](../../../../course/solutions/python/tinyllm/seq2seq/model.py): `Seq2Seq` (with `encode`, `init_state`, `decode_step`, `forward`, `greedy`), `EncoderState`, `DecoderState`, `save_seq2seq`, and `load_seq2seq`. The interface it implements is the contract [`course/contracts/py/tinyllm/seq2seq/model.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/model.pyi), and the oracle tests are in [`test_seq2seq.py`](../../../../course/solutions/learner-tests/L4.1/python/tests/l4-1-seq2seq/test_seq2seq.py).

> **Learning objectives**
> - Build an encoder, a bridge, and a decoder, and explain why the bridge reads the forward state at the last real token and the backward state at position 0.
> - Train with teacher forcing as a loop over `decode_step`, and add scheduled sampling with one random draw per step.
> - Plug in Bahdanau or Luong attention with one switch, and say exactly where each reads the source.
> - Reproduce the zero-weights example by hand and show that padding never changes a sentence.

## Notation

The section uses the following symbols, with the shapes the code gives them.

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

## Two networks and a bridge

Different lengths and different vocabularies force two networks. The source is read once and in full, by the **encoder**; the target is written one token at a time, conditioned on what was written so far, by the **decoder**. The encoder can look both ways; the decoder cannot look ahead.

The encoder output at position $t$ should know the whole sentence, so the encoder is `bidirectional` over two GRUs of width $d_h/2$: $o_t$ sees left context from the forward GRU and right context from the backward one. Two halves of $d_h/2$ make the concatenated output exactly $d_h$ wide, the decoder's width, which is why $d_h$ must be even.

The decoder needs a starting state that summarizes the source. The two states that have read the whole sentence are the forward GRU's state after the last **real** token (position $\ell_b - 1$) and the backward GRU's state after reading back to position 0. The **bridge** concatenates them and passes them through a `Linear` and a `tanh`, mapping them into the decoder's state space. The **decoder** itself is a `GRUCell` (or an `LSTMCell`, with $c_0 = 0$) over target embeddings, and an output `Linear` gives logits over the target vocabulary.

## One decoder step, three ways

With the pieces named, a single decoder step from token $y$ and state $s$ depends on which attention, if any, is plugged in. The attention module's `query_from` attribute picks the order:

| Attention | Step |
|---|---|
| none | $s' = \mathrm{cell}(E[y], s)$, logits $= W_o s' + b_o$ |
| `"previous"` (Bahdanau) | $c, a = \mathrm{att}(s, o)$; $s' = \mathrm{cell}([E[y] ; c], s)$; logits $= W_o [s' ; c] + b_o$ |
| `"current"` (Luong) | $s' = \mathrm{cell}([E[y] ; \tilde h], s)$; $c, a = \mathrm{att}(s', o)$; $\tilde h' = \tanh(W_c [c ; s'])$; logits $= W_o \tilde h' + b_o$ |

The only difference between Bahdanau and Luong in the step is *when* attention reads (with $s_{t-1}$ or with $s_t$) and what the output layer sees, so one switch is enough. Attention is a module passed to the constructor. The encoder computes `attention.project_keys(o)` once per sentence and stores it in the state, so the keys are projected once rather than at every step (the reason is worked out in [Bahdanau additive attention](02-bahdanau-additive-attention.md)). `decode_step` returns the attention weights too, which is what an alignment plot draws.

The parameters are registered in the order of the contract (`src_emb`, `enc_fwd`, `enc_bwd`, `bridge`, `tgt_emb`, `cell`, `attention`, `out`), and that order is the safetensors key list of a checkpoint.

## Teacher forcing and scheduled sampling

Training maximizes $\sum_t \log p(y_{t+1} \mid y_{\le t}, x)$, and it should not depend on the model's own mistakes. **Teacher forcing** feeds the true $y_t$ at step $t$ regardless of what the model predicted, so every step's input is known in advance: all $T$ losses come from one forward pass, and the gradient at step $t$ does not depend on the model's earlier choices. At inference the model must feed its own guesses, a mismatch called **exposure bias**.

**Scheduled sampling** (Bengio et al. 2015) trains with $\rho < 1$. Before each step after the first, one uniform $u$ is drawn for the whole batch; if $u < \rho$ the true token is fed, otherwise the argmax of the previous step's logits, with no gradient through the choice. With $\rho = 1$ nothing is drawn, so adding the option leaves a seeded run unchanged.

Targets are shifted by one: `tgt_in` is `bos` $y_1 \dots y_{T-1}$, and the loss compares step $t$'s logits with $y_{t+1}$ (`eos` at the end). Padding in the target is ignored with `ignore_index`.

There is one more design principle hidden here: one step function should serve training, greedy decoding, and beam search. If `forward` is literally a loop over `decode_step`, then greedy decoding and [beam search](04-beam-search.md) drive exactly the computation that was trained. `greedy` decodes all rows at once under `no_grad`, feeding back each row's argmax and stopping a row at its first `eos`; beam search later replaces it, still calling `decode_step`.

## Working it by hand

A zero-weights example pins down the data path. Take a GRU encoder-decoder with **every** GRU weight and bias 0. One GRU step gives $r = z = \sigma(0) = \tfrac12$ and $n = \tanh(0 + r \cdot 0) = 0$, so

$$h' = (1 - z)\, n + z\, h = \tfrac12 h .$$

From $h_0 = 0$ the encoder outputs 0 everywhere, whatever the source tokens. Let `bridge.weight` be 0 and `bridge.bias` $= (0.5, -0.5, 1, 0)$. Then $s_0 = \tanh(0.5, -0.5, 1, 0) = (0.462117, -0.462117, 0.761594, 0)$. Each decoder step halves it: $s_1 = (0.231059, -0.231059, 0.380797, 0)$, $s_2 = (0.115529, -0.115529, 0.190399, 0)$. With `out.weight` the first two rows of the identity and no bias, logits$_1 = (0.231059, -0.231059)$ and logits$_2 = (0.115529, -0.115529)$, whatever the target tokens.

The example fixes the path encoder, bridge, decoder cell, output layer. Drop the bridge and the logits are zero. A correct solution reproduces these numbers exactly.

## Designing the API

The contract fixes two immutable state records and a model class:

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

`EncoderState` holds the keys, the padding mask, the projected keys, and the bridge output. `DecoderState` adds the decoder's `h`, an LSTM's `c`, and Luong's fed $\tilde h$. Because both are `NamedTuple`s updated with `_replace`, beam search can gather a state by parent index without aliasing. The loss is not part of the model: the caller computes `cross_entropy` against `tgt_out` with `ignore_index` for padding.

## Building it step by step

### The encoder and the bridge

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

Replacing padding ids with 0 before the embedding is what lets callers pad with anything, even out-of-vocabulary ids. The padded embeddings never reach a real position, because `bidirectional` honors `lengths` and attention masks them with `mask`. Reading the final states at each row's own length, rather than at the padded end $S - 1$, is what keeps short sentences from being summarized from padding.

### The decoder step

`init_state` adds what the decoder carries beyond $h$: a zero $c$ for an LSTM cell, and a zero $\tilde h_0$ (`feed`) for Luong's input feeding. Its $h$ is `enc.init`, the bridge output. The step then has one branch per row of the table above:

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

The shapes in `__init__` follow from this code: the cell input is `d_emb + d_h` with attention (the context or the fed $\tilde h$), and the output layer is `2 * d_h` wide only for Bahdanau.

### Training as the step loop

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

One `rng.uniform()` per step decides for the whole batch. Using `.data` for the argmax makes the fed token a plain integer array, so the gradient never flows through the choice.

### Greedy decoding and checkpoints

`greedy` runs the same steps under `no_grad`, feeds back each row's argmax, appends to a row until it emits `eos`, and stops early once every row is done. `save_seq2seq` writes `config.json` (sizes, cell, and the attention kind: `none`, `bahdanau` with `tl_d_attn`, or `luong-<score>`) and the `state_dict` in registration order; `load_seq2seq` rebuilds the matching attention module and loads strictly. The full solution is [`model.py`](../../../../course/solutions/python/tinyllm/seq2seq/model.py).

### Why this design

Encoding costs $O(S B d_h^2)$. Each decoder step is $O(B d_h^2)$ for the cell plus $O(B S d_h)$ for attention, so a target of length $T$ costs $O(T B (d_h^2 + S d_h + d_h V_t))$. Projecting the keys once in `encode` removes an $O(T S d_h d_a)$ term.

A unidirectional encoder would not do. Its last state would know the whole sentence, but each $o_t$ would know only the left context, and attention reads the $o_t$. The bidirectional encoder gives every key the full sentence.

Why a bridge, and not zeros or a copy? Zeros make the decoder depend on attention alone, or on nothing at all without attention. Copying the encoder state forces the decoder to live in the encoder's state space. The learned `Linear` plus `tanh` decouples them at a cost of $d_h^2$ parameters.

Why teacher forcing at all? Free running would feed garbage tokens early in training and make the gradient depend on discrete choices. Teacher forcing gives a stable, parallel-in-time loss; the price is exposure bias, which scheduled sampling partly addresses.

The tradeoff accepted is that the decoder loop runs in Python, one step at a time, which is slow for long targets. That is what the transformer's [multi-head attention](../11-transformer/03-multi-head-attention.md) removes, by attending over the whole target in parallel during training.

## Testing it

The model suite [`test_seq2seq.py`](../../../../course/solutions/learner-tests/L4.1/python/tests/l4-1-seq2seq/test_seq2seq.py) uses the model's own pieces, recombined by hand, as oracles. It runs `enc_fwd` and `enc_bwd` on one unpadded sentence at a time and checks `init` against $\tanh(W_b [\dots])$; recomputes one decoder step from `cell`, `attention`, and `out` and compares it with `decode_step`; and checks that `forward` equals the step loop and that free running feeds back the argmax. It adds the zero-weights example, padding with out-of-vocabulary ids, a greedy decode that must stop at once (a huge `eos` bias), and a save and load. It imports only contract modules.

The full set of behaviors a correct solution must have is easiest to see side by side:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example zero weights | unit | the logits $s_0/2$, $s_0/4$ | the data path through the bridge |
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
| attention beats the bottleneck | learning | same budget without attention: at least 0.2 lower exact match | the lesson of this chapter |

The last two rows are the point of the chapter: with the same 150 steps, attention reverses digit strings almost perfectly and the bottleneck model does not.

## What can go wrong

Most bugs in an encoder-decoder produce a model that trains, just not the model the papers describe. These are the ones to watch for:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the decoder starts from zeros (bridge state dropped) | the source reaches the decoder only through attention, or not at all | `init_state` starts the decoder from `enc.init`, the bridge output |
| teacher forcing fed `tgt_in[:, t - 1]` at step $t$ | the model is asked to copy its input; training loss falls, translation fails | `forward` feeds `tgt[:, t]` at step $t$ |
| final encoder states read at the padded end or the wrong position | short sentences are summarized from padding | `encode` reads `out[n - 1, rows]` forward and `out[0]` backward, inside each length |
| attention in the wrong place: Luong reading with the old state, Bahdanau's query not $s_{t-1}$, or its context left out of the output | a different model from the paper's; golden values disagree | `decode_step` branches on `query_from`: `att(state.h, ...)` before the cell for Bahdanau, `att(h, ...)` after it for Luong; Bahdanau's logits read `[h ; ctx]` |
| no input feeding | the next step never sees $\tilde h$ | the cell input is `[e ; state.feed]`, and `feed` is replaced by $\tilde h$ each step |
| teacher forcing when $u \ge \rho$ | $\rho = 0$ still feeds the truth | `rng.uniform() < teacher_forcing` feeds the truth |

> **Warning:** The off-by-one in teacher forcing is the most insidious. Feeding `tgt_in[:, t - 1]` at step $t$ asks the model to copy its input, so training loss falls nicely and translation fails completely.

## Where this leads

[Beam search](04-beam-search.md) searches over `decode_step`, gathering `DecoderState` by parent, and [sequence metrics](05-sequence-metrics.md) (exact match, BLEU, chrF) grade its outputs. In the next part, [the encoder-decoder Transformer](../11-transformer/05-encoder-decoder-transformer.md) keeps the encoder-decoder split and replaces both RNNs with attention. Later, [the LM evaluation harness and the model zoo](../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) trains and evaluates seq2seq checkpoints on the dates task. At the end of this chapter, `{tinyllm} train seq2seq` followed by `{tinyllm} translate --beam 5` runs this model end to end.

The production equivalents add a good deal on top of this model:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Seq2Seq` | OpenNMT-py, fairseq LSTM models | multi-layer stacks, dropout, copy attention, batched beam search | OpenNMT-py `onmt/models/model.py`; fairseq `models/lstm.py` |
| the bridge | `bridge` in OpenNMT | a learned map from encoder to decoder state per layer | OpenNMT-py `onmt/encoders/rnn_encoder.py` |
| teacher forcing ratio | scheduled sampling, professor forcing | curricula for $\rho$, adversarial matching of free-running dynamics | Bengio et al. 2015; Lamb et al. 2016 |
| encoder-decoder RNN | T5, BART | the same split with transformer blocks and span corruption | [T5 span corruption](../12-objectives-and-adaptation/04-t5-span-corruption-and-relative-buckets.md); Hugging Face `T5ForConditionalGeneration` |

### Further reading

- Sutskever, Vinyals, and Le, "Sequence to Sequence Learning with Neural Networks" (2014).
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder" (2014).
- Bengio et al., "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks" (2015).
- Lamb et al., "Professor Forcing" (2016).

## Key terms

- **encoder**: the network that reads the whole source once; here a bidirectional GRU whose output at each position sees the full sentence.
- **decoder**: the network that writes the target one token at a time from its previous token and state.
- **bridge**: the `Linear` plus `tanh` that maps the encoder's two final states to the decoder's first state.
- **teacher forcing**: feeding the true previous target token at every training step.
- **exposure bias**: the mismatch between training on true tokens and decoding on the model's own guesses.
- **scheduled sampling**: training with probability $\rho$ of feeding the true token and $1 - \rho$ of feeding the model's argmax.
- **input feeding**: Luong's practice of feeding the previous attentional state $\tilde h$ into the next decoder input.

## Summary

- An encoder reads the source, a bridge turns its two final states into the decoder's first state, and the decoder writes the target one token at a time.
- Teacher forcing feeds the true previous token, so training is `decode_step` over the target, all steps in one graph; scheduled sampling sometimes feeds the model's own guess instead.
- Bahdanau attention reads with the state from before the step, Luong's with the state after it, plus input feeding.
- Each row's final encoder states are read at its own length, so padding never changes a sentence.
- With the same 150 steps, attention reverses digit strings almost perfectly and the bottleneck model does not.

## Review questions

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
6. Why can callers pad the source with out-of-vocabulary ids?
   <details><summary>Answer</summary>`encode` replaces padded ids with 0 before the embedding lookup, `bidirectional` honors the lengths so padded positions never reach a real one, and attention masks them with `mask`.</details>
