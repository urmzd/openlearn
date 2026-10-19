# Bidirectional RNN with length-aware reversal

Your [LSTM](02-lstm.md) and [GRU](03-gru.md) read a sentence left to right, so the state at word 3 knows words 0 to 3 and nothing after. For a language model that is the point: it must not see the future. For an **encoder** it is a handicap. When [Encoder-decoder with teacher forcing](../05-attention-origins/01-encoder-decoder-teacher-forcing.md) encodes a source sentence to translate it, the representation of "bank" should already know whether "river" or "account" follows. Reading the sentence in both directions and concatenating gives every position both contexts.

The idea is one line; the bug is in the batching. Sentences in a batch have different lengths and are padded at the end, and the backward direction must start at each sentence's last real word, not at the padding. In this section you build the length-aware reversal once, check it against torch's packed sequences and against a per-sentence loop, and so give the seq2seq model its encoder. Together with the LSTM, the GRU, and the [RNN language model](06-rnn-language-model-stateful-tbptt.md), this completes the recurrent part of the book.

The two directions are the `LSTM` and `GRU` modules of the previous sections. The reversal is a [`Tensor`](../01-foundations/02-tensor-and-broadcasting-backward.md) getitem, and the halves are joined with `F.concat` from the [op library](../01-foundations/03-op-library-and-gradcheck.md). The tests are property tests in the style of [Property-based tests with Hypothesis](../../software-craftsmanship/03-testing-mentality/02-property-based-tests.md) in *Software Craftsmanship*. The solution is [`bi.py`](../../../../course/solutions/python/tinyllm/rnn/bi.py), with `reversal_index`, `reverse_padded`, and `bidirectional`; its contract is [`bi.pyi`](../../../../course/contracts/py/tinyllm/rnn/bi.pyi) and its property tests are [`test_bi.py`](../../../../course/solutions/learner-tests/L3.4/python/tests/l3-4-bi/test_bi.py).

> **Learning objectives**
> - Build the per-sequence reversal index $\pi_b(t)$ that mirrors each sequence inside its own length and leaves padding in place.
> - Explain why `x[::-1]` is wrong for a padded batch even when the RNN masks steps past the length.
> - Compose two single-direction modules into a bidirectional layer whose halves match torch's `bidirectional=True`.
> - State and test the properties (involution, mirror-and-fix, per-sequence equivalence, inert padding) that make the batched computation trustworthy.

## Reversal inside each length

The notation:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T, B$ | padded length, batch size | integers |
| $x$ | a padded batch, time first | `[T, B, D]` |
| $\ell_b$ | the real length of sequence $b$; steps $t \ge \ell_b$ are padding | integer in $1..T$ |
| $\pi_b(t)$ | the reversal index: $\ell_b - 1 - t$ for $t < \ell_b$, else $t$ | integer in $0..T-1$ |
| $R(x)$ | the length-aware reversal, $R(x)[t, b] = x[\pi_b(t), b]$ | `[T, B, ...]` |
| $\overrightarrow{h}_t$, $\overleftarrow{h}_t$ | forward and backward states at position $t$ | `[B, H_f]`, `[B, H_b]` |

For sequence $b$ the real tokens are $x[0..\ell_b - 1, b]$. Reading them backward means position $t$ takes token $\ell_b - 1 - t$ for $t < \ell_b$; positions past the length keep their padding. That map $\pi_b$ is a permutation of $0..T-1$ that mirrors the first $\ell_b$ positions and fixes the rest, so applying it twice gives the identity: $R(R(x)) = x$.

One integer index array of shape $[T, B]$ holds all $B$ permutations, and the gather `x[idx, arange(B)]` applies them at once. On a `Tensor` it is getitem, whose backward scatters each gradient back to where its value came from, which is the same permutation again.

## Why not `x[::-1]`

It is tempting to flip the padded batch. Flipping reverses every sequence over all $T$ steps: a sequence of length 2 in a batch of length 5 becomes three padding steps followed by its two real tokens. The backward RNN would start from padding. The LSTM's and GRU's `lengths` mask does make them ignore steps past the length, but after a full flip the padding is no longer past the length; it is at the front. The state would be garbage before the first real token.

## The layer

With `fwd` and `bwd` single-direction modules that honor `lengths` (output 0 and carry the state past each length), the layer is

$$\text{out}_f = \texttt{fwd}(x), \qquad \text{out}_r = \texttt{bwd}(R(x)), \qquad \text{out} = [\,\text{out}_f \,;\, R(\text{out}_r)\,] \text{ on the last axis.}$$

$\text{out}_r$ is in reversed time (its row $t$ is the backward state after reading $\ell_b - 1, \dots, \ell_b - 1 - t$), so it is reversed back before the concatenation; padded positions are 0 in both halves because $R$ fixes them. The forward half comes first, as in torch's `bidirectional=True`, whose `*_l0` weights are the forward module and `*_l0_reverse` weights the backward one. Each direction's final state is in `out`: the forward one at $[\ell_b - 1, b, :H_f]$ (its last real step) and the backward one at $[0, b, H_f:]$ (it ends on the first token).

## Working it by hand

Take $T = 3$ and three sequences with lengths $\ell = [3, 1, 2]$: "abc", "d", "ef". The padded batch, with a dot for padding:

| $t$ | seq 0 ($\ell = 3$) | seq 1 ($\ell = 1$) | seq 2 ($\ell = 2$) |
|---|---|---|---|
| 0 | a | d | e |
| 1 | b | . | f |
| 2 | c | . | . |

The permutations are $\pi_0 = [2, 1, 0]$, $\pi_1 = [0, 1, 2]$ (a one-token sequence is its own reverse, and both padding steps stay), and $\pi_2 = [1, 0, 2]$. As a $[T, B]$ array, `reversal_index([3, 1, 2], 3)` $= [[2, 0, 1], [1, 1, 0], [0, 2, 2]]$, and the reversed batch reads "cba", "d..", "fe.". `x[::-1]` would read "cba", "..d", ".fe".

To see the whole layer at work, take as both modules a running sum: output $t$ is the sum of the inputs up to $t$, 0 past the length. For the sequence $[1, 2, 3]$ the forward half is the prefix sums $[1, 3, 6]$. The backward module reads $[3, 2, 1]$, outputs $[3, 5, 6]$, and reversing that back gives $[6, 5, 3]$: the suffix sums, so position $t$ holds the sum from $t$ to the end. For a second sequence $[4]$ padded with two 99s, both halves are $[4, 0, 0]$: the 99s never enter.

A correct solution must reproduce both of these exactly.

## Designing the interface

Three functions, from the index to the layer:

```python
# python/tinyllm/rnn/bi.py
def reversal_index(lengths, T: int) -> NDArray                  # int64 [T, B]
def reverse_padded(x, lengths)                                  # Tensor or ndarray [T, B, ...], same kind back
def bidirectional(fwd: Module, bwd: Module, x: Tensor, lengths=None) -> Tensor   # [T, B, H_f + H_b]
```

Six properties pin the solution down. The model suite writes them as Hypothesis property tests in [`test_bi.py`](../../../../course/solutions/learner-tests/L3.4/python/tests/l3-4-bi/test_bi.py):

1. **Involution.** For random $T$ and lengths in $1..T$, `reverse_padded(reverse_padded(x, n), n)` equals `x`.
2. **Mirror and fix.** For each $b$, the first $\ell_b$ rows of `reverse_padded(x, n)[:, b]` are `x[:ℓ_b, b]` reversed and the rest equal `x[ℓ_b:, b]`.
3. **Plain flip.** With `lengths=None`, the result is `x[::-1]`.
4. **Per-sequence equivalence.** With two `GRU`s, `bidirectional` equals running each sequence alone: the forward half from `fwd(x[:ℓ_b, b])`, the backward half from `bwd` on the reversed sequence, reversed back; padded rows are 0.
5. **Padding is inert.** Changing the values in the padding changes no output.
6. **Bad lengths raise.**

The model tests import only `tinyllm.rnn.bi`, `tinyllm.rnn.gru` and `tinyllm.autograd.tensor`, so they test the contract, not the implementation.

## Building it step by step

The reasoning comes first.

1. **The backward direction must see real tokens in reverse order and nothing else.** The single-direction LSTM and GRU already handle padding correctly *if it is at the end*: past $\ell_b$ they output 0 and carry the state unchanged. So the job is to produce, for each sequence, an input where the real tokens are reversed and the padding is still at the end. Then `bwd` can be called exactly like `fwd`, with the same `lengths`.
2. **That input is a per-column permutation of time.** Sequence $b$ needs $t \mapsto \ell_b - 1 - t$ on $0..\ell_b - 1$ and $t \mapsto t$ beyond. Different columns need different permutations, so no slice can do it; a gather with an index array of shape $[T, B]$ can, in one vectorized operation.
3. **The permutation is its own inverse.** Mirroring $0..\ell_b-1$ twice is the identity, and fixed points stay fixed. So the *same* function that builds the backward input also restores the backward outputs to time order. In the worked example, `bwd` produced $[3, 5, 6]$ in reversed time and one more reversal gave the suffix sums $[6, 5, 3]$.
4. **Gradients come for free.** If the gather is `Tensor.__getitem__` with integer arrays, its backward scatters each upstream gradient to the source position. Because every position is used exactly once (it is a permutation), the gradient is just the same permutation, and padding receives gradient only from padded outputs, which are constant 0.
5. **Order of the halves is a convention fixed by torch.** Concatenating `[forward ; backward]` on the last axis matches `bidirectional=True`, so weights loaded from torch line up with the halves the decoder reads.
6. **Validate lengths once, at the boundary.** Lengths outside $1..T$ or non-integers would silently produce a wrong permutation (negative indices wrap around in NumPy). Rejecting them up front turns a silent wrong answer into an error.

### The reversal index

`reversal_index` builds all $B$ permutations at once by broadcasting a time column against a lengths row:

```python
def reversal_index(lengths: ArrayLike, T: int) -> NDArray:
    n = np.asarray(lengths)
    # Integer 1-D only: a float length like 2.0 would "work" and hide a batcher bug.
    if n.ndim != 1 or not np.issubdtype(n.dtype, np.integer):
        raise ValueError(f"lengths must be a 1-D integer array, got {n.dtype} {n.shape}")
    # 0 would mean "mirror nothing" and index -1; > T would index past the end.
    if (n < 1).any() or (n > T).any():
        raise ValueError(f"lengths must lie in 1..{T}, got {n.tolist()}")
    t = np.arange(T, dtype=np.int64)[:, None]     # [T, 1]
    n = n.astype(np.int64)[None, :]               # [1, B]
    # Inside the length: mirror around the sequence's own middle. Padding: stay.
    return np.where(t < n, n - 1 - t, t)          # [T, B]
```

For lengths $[3, 1, 2]$ and $T = 3$ this returns exactly the array of the hand example, $[[2, 0, 1], [1, 1, 0], [0, 2, 2]]$.

### Applying the permutation

`reverse_padded` applies the permutation. It works on both `ndarray` and `Tensor` because it only uses `.ndim`, `.shape` and getitem:

```python
def reverse_padded(x: Any, lengths: Optional[ArrayLike]) -> Any:
    if x.ndim < 2:
        raise ValueError(f"x must be [T, B, ...], got shape {x.shape}")
    T, B = x.shape[0], x.shape[1]
    if lengths is None:
        # No padding: every sequence has length T, and the index is a plain flip.
        lengths = np.full(B, T, dtype=np.int64)
    if np.asarray(lengths).shape != (B,):
        raise ValueError(f"lengths must have {B} entries, got shape {np.asarray(lengths).shape}")
    idx = reversal_index(lengths, T)
    # x[idx[t, b], b]: a gather with two integer index arrays; for a Tensor
    # the getitem backward scatters the gradient back to the same places.
    return x[idx, np.arange(B)[None, :]]
```

The second index `np.arange(B)[None, :]` broadcasts against `idx`, so output `[t, b]` reads `x[idx[t, b], b]`: each column is permuted only within itself. Trailing feature axes are carried along untouched.

### The layer

`bidirectional` is then three lines, one per equation of the layer:

```python
def bidirectional(fwd, bwd, x, lengths=None):
    if not isinstance(x, Tensor):
        x = Tensor(np.asarray(x))
    out_f, _ = fwd(x, lengths=lengths)                           # forward, padding -> 0
    out_r, _ = bwd(reverse_padded(x, lengths), lengths=lengths)  # backward, in reversed time
    # Reverse back to time order (the involution), then forward half first, like torch.
    return F.concat([out_f, reverse_padded(out_r, lengths)], axis=-1)
```

Both directions receive `lengths`; that is what keeps padded outputs at 0 and stops each direction's state from reading padding. The final states the modules return are discarded because they are recoverable from `out` (at $[\ell_b - 1, b, :H_f]$ and $[0, b, H_f:]$).

The full solution is [`bi.py`](../../../../course/solutions/python/tinyllm/rnn/bi.py).

### Why this design

The reversal is $O(TB)$ index work plus one gather of $O(TBD)$; the two recurrences dominate at $O(T B H^2)$ each. Running both directions doubles the RNN cost, nothing more.

Consider the alternatives. A Python loop over sequences, slicing `x[:ℓ_b, b][::-1]` per sequence, is obviously correct but serializes the batch into $B$ separate RNN runs. The gather keeps one batched call per direction, and the per-sequence loop survives as the differential check. Packing (`pack_padded_sequence`) skips padded steps entirely, saving compute when lengths vary a lot, but needs sorting and a variable-batch RNN kernel. Here padding costs some wasted steps, which the `lengths` mask makes harmless; simplicity wins at this scale. And `x[::-1]` is cheaper to write and correct only when there is no padding; with padding it puts the padding first. The `lengths=None` path shows that the plain flip is just the special case $\ell_b = T$.

## What can go wrong

The checks a correct solution passes:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: reversal | unit | the index array and "cba", "d..", "fe." | the reversal is pinned to a case you can compute |
| hand example: running sums | unit | prefix sums forward, suffix sums backward, padding 0, forward half first | the layer is pinned to a case you can compute |
| involution that keeps padding | property | $R(R(x)) = x$; padding fixed; the Tensor gradient is the same permutation | the backward outputs come back in order; gradients reach the right tokens |
| no lengths is a plain flip | unit | without padding, `x[::-1]` | the unpadded special case |
| matches torch `bidirectional=True` | golden | GRU and LSTM over packed batches: outputs and every gradient | the seq2seq model loads torch encoders |
| matches a per-sequence loop | differential | the batch equals each sequence run alone, both directions | batching changes nothing |
| padding never reaches either direction | boundary | garbage padding changes no output, gets no gradient; final states at $\ell_b - 1$ and 0 | encoder states for the decoder |
| bad lengths | boundary | length 0, longer than $T$, wrong count, floats | errors at the batcher |

Four mistakes are common.

> **Warning:** Reversing the padded batch with `x[::-1]`, or mirroring the padding too, makes the backward direction read padding before the sentence, so short sentences get worse encodings than long ones. `reversal_index` mirrors only $t < \ell_b$ and leaves padding fixed (`np.where(t < n, n - 1 - t, t)`).

> **Warning:** Not reversing the backward outputs back means position $t$'s backward half describes position $\ell_b - 1 - t$. `bidirectional` applies `reverse_padded` to `out_r` before concatenating.

> **Warning:** Concatenating the backward half first lets torch encoders load, but their output halves are swapped for the decoder. The solution writes `F.concat([out_f, ...])`: forward half first, as in torch.

> **Warning:** Running a direction without `lengths` makes padded outputs nonzero, and the backward state at the end of a short sequence has read padding. Both `fwd` and `bwd` are called with `lengths=lengths`.

## Where this leads

This section composed the [LSTM](02-lstm.md) and [GRU](03-gru.md), each with `lengths`, as the two directions, with the reversal as a [`Tensor`](../01-foundations/02-tensor-and-broadcasting-backward.md) getitem on two integer index arrays and `F.concat` from the [op library](../01-foundations/03-op-library-and-gradcheck.md) joining the halves. Next, the seq2seq encoder in [Encoder-decoder with teacher forcing](../05-attention-origins/01-encoder-decoder-teacher-forcing.md) is a bidirectional GRU over the source sentence, whose final states initialize the decoder, and [Bahdanau additive attention](../05-attention-origins/02-bahdanau-additive-attention.md) reads the encoder's per-position outputs, both halves.

Production equivalents:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `reverse_padded` | `torch.nn.utils.rnn.pack_padded_sequence` | no padded step is computed at all: sequences are sorted by length and the batch shrinks as they end | `torch/nn/utils/rnn.py` |
| `bidirectional` | `torch.nn.LSTM(bidirectional=True)` and cuDNN | both directions in one call, sharing the input GEMM | `aten/src/ATen/native/RNN.cpp` |
| a bidirectional encoder | BERT ([BERT: the bidirectional encoder and masked language modeling](../07-objectives/02-bert-masked-lm.md)) | bidirectional context by attention instead of recurrence, every position at once | `transformers/models/bert/modeling_bert.py` |
| reversal index | Keras `go_backwards` with masking | the same mask-aware reversal for masked sequences | `keras/layers/rnn/bidirectional.py` |

### Further reading

- Schuster and Paliwal, "Bidirectional Recurrent Neural Networks" (1997).
- Graves and Schmidhuber, "Framewise phoneme classification with bidirectional LSTM" (2005).
- The `torch.nn.utils.rnn` documentation.

## Key terms

- **Bidirectional layer**: two recurrent modules, one reading left to right and one right to left, with outputs concatenated per position.
- **Length-aware reversal $R$**: the per-sequence permutation that mirrors positions $0..\ell_b - 1$ and fixes the padding.
- **Involution**: a map that is its own inverse, $R(R(x)) = x$.
- **Gather**: indexing with integer arrays, here `x[idx, arange(B)]`, whose backward scatters gradients to the source positions.

## Summary

- A bidirectional layer is two recurrent modules: one reads left to right, the other right to left, and their outputs are concatenated, forward half first, so position $t$ sees the whole sentence.
- With padding, "right to left" means from each sequence's **own** last real token: reverse inside each length with the index $\ell_b - 1 - t$ and leave the padding where it is; a plain `x[::-1]` makes the backward RNN read padding first.
- The reversal is an involution, so the same gather puts the backward outputs back in time order, and on a Tensor its gradient is the same permutation.
- The batched computation equals running every sequence alone, unpadded, which is the property that makes it trustworthy.

## Review questions

1. Why can the same function both prepare the backward input and restore the backward output?
   <details><summary>Answer</summary>The per-sequence map mirrors positions $0..\ell_b-1$ and fixes the rest, so applying it twice is the identity. Reversing the output of an RNN that read reversed input puts row $t$ back at time $t$.</details>
2. The LSTM and GRU modules already ignore steps past the length. Why is that not enough to make `bwd(x[::-1])` correct?
   <details><summary>Answer</summary>The mask ignores steps at indices $\ge \ell_b$. After a full flip, the padding sits at indices $0..T-\ell_b-1$, which are *before* the length, so the mask treats it as real input and the backward state is corrupted before the first real token.</details>
3. Where in `out` is the backward direction's final state, and why there?
   <details><summary>Answer</summary>At $[0, b, H_f:]$. The backward RNN ends on the first token, and after reversing its outputs back to time order its last real step lands at position 0.</details>
4. Why does the gather give correct gradients without any custom backward?
   <details><summary>Answer</summary>Getitem's backward scatters each output gradient to the position it was read from. A permutation reads each position exactly once, so the scatter is the inverse permutation, which here is the same permutation, and padded positions receive only the zero gradient of their zero outputs.</details>
5. What goes wrong if a length of 0 is allowed through?
   <details><summary>Answer</summary>The mirror index $\ell_b - 1 - t$ is $-1$ at $t = 0$, which NumPy reads as the last row: a padded value would be silently copied in. The solution rejects lengths outside $1..T$ instead.</details>
