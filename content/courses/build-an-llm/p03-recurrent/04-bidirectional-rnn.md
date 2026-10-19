# Bidirectional RNN with length-aware reversal

## Overview

| | |
|---|---|
| **Module** | `L3.4` · build · Python · Pass 4 · 1 to 1.5 h reading |
| **Solution** | [`python/tinyllm/rnn/bi.py`](../../../../course/solutions/python/tinyllm/rnn/bi.py): `reversal_index`, `reverse_padded`, `bidirectional`; model property tests in [`test_bi.py`](../../../../course/solutions/learner-tests/L3.4/python/tests/l3-4-bi/test_bi.py) |
| **Contract** | [`course/contracts/py/tinyllm/rnn/bi.pyi`](../../../../course/contracts/py/tinyllm/rnn/bi.pyi) |
| **Needs** | `L3.2` `LSTM` and `L3.3` `GRU` (the two directions) · `L0.1` `Tensor` (the gather is getitem) · `L0.2` `F.concat` · reading: `L0.4` `Module`, `craft.04` property tests |
| **Used by** | later: `L4.1` the seq2seq encoder reads each source sentence both ways; with `L3.2`, `L3.3` and `L3.6` it completes the recurrent stage |
| **Optional depth** | Schuster and Paliwal, "Bidirectional Recurrent Neural Networks" (1997); Graves and Schmidhuber, "Framewise phoneme classification with bidirectional LSTM" (2005); the `torch.nn.utils.rnn` documentation |

## Key Takeaways

- A bidirectional layer is two recurrent modules: one reads left to right, the other right to left, and their outputs are concatenated, forward half first, so position $t$ sees the whole sentence.
- With padding, "right to left" means from each sequence's **own** last real token: reverse inside each length with the index $\ell_b - 1 - t$ and leave the padding where it is; a plain `x[::-1]` makes the backward RNN read padding first.
- The reversal is an involution, so the same gather puts the backward outputs back in time order, and on a Tensor its gradient is the same permutation.
- The batched computation equals running every sequence alone, unpadded, which is the property that makes it trustworthy.

## How to read this chapter

Sections 1 to 3 build the length-aware reversal from first principles and work it by hand. Section 4 fixes the interface and the properties a correct solution must satisfy. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Your LSTM and GRU read a sentence left to right, so the state at word 3 knows words 0 to 3 and nothing after. For a language model that is the point: it must not see the future. For an **encoder** it is a handicap: when `L4.1` encodes a source sentence to translate it, the representation of "bank" should already know whether "river" or "account" follows. Reading the sentence in both directions and concatenating gives every position both contexts. The idea is one line; the bug is in the batching. Sentences in a batch have different lengths and are padded at the end, and the backward direction must start at each sentence's last real word, not at the padding. This module builds the length-aware reversal once, checks it against torch's packed sequences and against a per-sentence loop, and gives `L4.1` its encoder.

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

A correct solution must reproduce both of these exactly.

## 4. The interface

```python
# python/tinyllm/rnn/bi.py
def reversal_index(lengths, T: int) -> NDArray                  # int64 [T, B]
def reverse_padded(x, lengths)                                  # Tensor or ndarray [T, B, ...], same kind back
def bidirectional(fwd: Module, bwd: Module, x: Tensor, lengths=None) -> Tensor   # [T, B, H_f + H_b]
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: reversal | unit | section 3: the index array and "cba", "d..", "fe." | the reversal is pinned to a case you can compute |
| hand example: running sums | unit | prefix sums forward, suffix sums backward, padding 0, forward half first | the layer is pinned to a case you can compute |
| involution that keeps padding | property | $R(R(x)) = x$; padding fixed; the Tensor gradient is the same permutation | the backward outputs come back in order; gradients reach the right tokens |
| no lengths is a plain flip | unit | without padding, `x[::-1]` | the unpadded special case |
| matches torch `bidirectional=True` | golden | GRU and LSTM over packed batches: outputs and every gradient | `L4.1` loads torch encoders |
| matches a per-sequence loop | differential | the batch equals each sequence run alone, both directions | batching changes nothing |
| padding never reaches either direction | boundary | garbage padding changes no output, gets no gradient; final states at $\ell_b - 1$ and 0 | encoder states for the decoder (`L4.1`) |
| bad lengths | boundary | length 0, longer than $T$, wrong count, floats | errors at the batcher |

### The properties, as tests

The six properties below pin the solution down; the model suite writes them as Hypothesis property tests (`craft.04`) in [`test_bi.py`](../../../../course/solutions/learner-tests/L3.4/python/tests/l3-4-bi/test_bi.py):

1. **Involution.** For random $T$ and lengths in $1..T$, `reverse_padded(reverse_padded(x, n), n)` equals `x`.
2. **Mirror and fix.** For each $b$, the first $\ell_b$ rows of `reverse_padded(x, n)[:, b]` are `x[:ℓ_b, b]` reversed and the rest equal `x[ℓ_b:, b]`.
3. **Plain flip.** With `lengths=None`, the result is `x[::-1]`.
4. **Per-sequence equivalence.** With two `GRU`s, `bidirectional` equals running each sequence alone: the forward half from `fwd(x[:ℓ_b, b])`, the backward half from `bwd` on the reversed sequence, reversed back; padded rows are 0.
5. **Padding is inert.** Changing the values in the padding changes no output.
6. **Bad lengths raise.**

The model tests import only `tinyllm.rnn.bi`, `tinyllm.rnn.gru` and `tinyllm.autograd.tensor`, so they test the contract, not the implementation.

## 5. The solution, step by step

### From first principles

1. **The backward direction must see real tokens in reverse order and nothing else.** The single-direction modules of `L3.2` and `L3.3` already handle padding correctly *if it is at the end*: past $\ell_b$ they output 0 and carry the state unchanged. So the job is to produce, for each sequence, an input where the real tokens are reversed and the padding is still at the end. Then `bwd` can be called exactly like `fwd`, with the same `lengths`.
2. **That input is a per-column permutation of time.** Sequence $b$ needs $t \mapsto \ell_b - 1 - t$ on $0..\ell_b - 1$ and $t \mapsto t$ beyond. Different columns need different permutations, so no slice can do it; a gather with an index array of shape $[T, B]$ can, in one vectorized operation.
3. **The permutation is its own inverse.** Mirroring $0..\ell_b-1$ twice is the identity, and fixed points stay fixed. So the *same* function that builds the backward input also restores the backward outputs to time order. In the worked example, `bwd` produced $[3, 5, 6]$ in reversed time and one more reversal gave the suffix sums $[6, 5, 3]$.
4. **Gradients come for free.** If the gather is `Tensor.__getitem__` with integer arrays (`L0.1`), its backward scatters each upstream gradient to the source position. Because every position is used exactly once (it is a permutation), the gradient is just the same permutation, and padding receives gradient only from padded outputs, which are constant 0.
5. **Order of the halves is a convention fixed by torch.** Concatenating `[forward ; backward]` on the last axis matches `bidirectional=True`, so weights loaded from torch line up with the halves the decoder reads.
6. **Validate lengths once, at the boundary.** Lengths outside $1..T$ or non-integers would silently produce a wrong permutation (negative indices wrap around in NumPy). Rejecting them up front turns a silent wrong answer into an error.

### The code, annotated

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

For lengths $[3, 1, 2]$ and $T = 3$ this returns exactly the array of section 3, $[[2, 0, 1], [1, 1, 0], [0, 2, 2]]$.

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

`bidirectional` is then three lines, one per equation in section 2:

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

Full solution: [`bi.py`](../../../../course/solutions/python/tinyllm/rnn/bi.py).

### Why this solution

- **Cost.** The reversal is $O(TB)$ index work plus one gather of $O(TBD)$; the two recurrences dominate at $O(T B H^2)$ each. Running both directions doubles the RNN cost, nothing more.
- **Versus a Python loop over sequences.** Slicing `x[:ℓ_b, b][::-1]` per sequence is obviously correct but serializes the batch into $B$ separate RNN runs. The gather keeps one batched call per direction, and the per-sequence loop survives as the differential check.
- **Versus packing (`pack_padded_sequence`).** Packing skips padded steps entirely, saving compute when lengths vary a lot, but needs sorting and a variable-batch RNN kernel. Here padding costs some wasted steps, which the `lengths` mask makes harmless; simplicity wins at this scale.
- **Versus `x[::-1]`.** Cheaper to write and correct only when there is no padding; with padding it puts the padding first (section 2). The `lengths=None` path shows that the plain flip is just the special case $\ell_b = T$.

### Explain it back

1. Why can the same function both prepare the backward input and restore the backward output?
   <details><summary>Answer</summary>The per-sequence map mirrors positions $0..\ell_b-1$ and fixes the rest, so applying it twice is the identity. Reversing the output of an RNN that read reversed input puts row $t$ back at time $t$.</details>
2. The `L3.2`/`L3.3` modules already ignore steps past the length. Why is that not enough to make `bwd(x[::-1])` correct?
   <details><summary>Answer</summary>The mask ignores steps at indices $\ge \ell_b$. After a full flip, the padding sits at indices $0..T-\ell_b-1$, which are *before* the length, so the mask treats it as real input and the backward state is corrupted before the first real token.</details>
3. Where in `out` is the backward direction's final state, and why there?
   <details><summary>Answer</summary>At $[0, b, H_f:]$. The backward RNN ends on the first token, and after reversing its outputs back to time order its last real step lands at position 0.</details>
4. Why does the gather give correct gradients without any custom backward?
   <details><summary>Answer</summary>Getitem's backward scatters each output gradient to the position it was read from. A permutation reads each position exactly once, so the scatter is the inverse permutation, which here is the same permutation, and padded positions receive only the zero gradient of their zero outputs.</details>
5. What goes wrong if a length of 0 is allowed through?
   <details><summary>Answer</summary>The mirror index $\ell_b - 1 - t$ is $-1$ at $t = 0$, which NumPy reads as the last row: a padded value would be silently copied in. The solution rejects lengths outside $1..T$ instead.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. reversing the padded batch with `x[::-1]`, or mirroring the padding too | the backward direction reads padding before the sentence; short sentences get worse encodings than long ones | `reversal_index` mirrors only $t < \ell_b$ and leaves padding fixed (`np.where(t < n, n - 1 - t, t)`) |
| 2. not reversing the backward outputs back | position $t$'s backward half describes position $\ell_b - 1 - t$ | `bidirectional` applies `reverse_padded` to `out_r` before concatenating |
| 3. concatenating backward first | torch encoders load but their output halves are swapped for the decoder | `F.concat([out_f, ...])`: forward half first, as in torch |
| 4. running a direction without `lengths` | padded outputs are no longer 0, and the backward state at the end of a short sequence has read padding | both `fwd` and `bwd` are called with `lengths=lengths` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L3.2` | `LSTM` with `lengths` is one direction |
| Back | `L3.3` | `GRU` with `lengths` is one direction |
| Back | `L0.1` | the reversal is a `Tensor` getitem with two integer index arrays |
| Back | `L0.2` | `F.concat` joins the halves |
| Forward | `L4.1` | the seq2seq encoder: a bidirectional GRU over the source sentence, whose final states initialize the decoder |
| Forward | `L4.2` | Bahdanau attention reads the encoder's per-position outputs, both halves |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `reverse_padded` | `torch.nn.utils.rnn.pack_padded_sequence` | no padded step is computed at all: sequences are sorted by length and the batch shrinks as they end | `torch/nn/utils/rnn.py` |
| `bidirectional` | `torch.nn.LSTM(bidirectional=True)` and cuDNN | both directions in one call, sharing the input GEMM | `aten/src/ATen/native/RNN.cpp` |
| a bidirectional encoder | BERT (`L6.2`) | bidirectional context by attention instead of recurrence, every position at once | `transformers/models/bert/modeling_bert.py` |
| reversal index | Keras `go_backwards` with masking | the same mask-aware reversal for masked sequences | `keras/layers/rnn/bidirectional.py` |
