# Bahdanau additive attention

The [encoder-decoder](01-encoder-decoder-teacher-forcing.md) of the previous section, without attention, squeezes the whole source sentence into one vector, the decoder's first state. On short inputs that works. On a 30-character date string the decoder has forgotten the start of the input by the time it writes the end, and exact match drops sharply with length. Bahdanau's fix, the first attention mechanism in neural translation, lets every decoder step look back at **all** encoder outputs and choose which to read. It is the idea the 2017 transformer's [scaled dot-product attention](../06-transformer-2017/01-scaled-dot-product-attention.md) keeps while dropping the recurrence, so this is where the core operation of every model in the rest of the book is born.

The section builds on `Linear` and `Module` from [the module system](../01-foundations/05-module-system-and-layers.md), the `tanh`, `masked_fill`, `softmax`, and `matmul` ops of [the op library](../01-foundations/03-op-library-and-gradcheck.md), the `Tensor` of [the tensor section](../01-foundations/02-tensor-and-broadcasting-backward.md), and `PCG32` from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing* for the default initialization. Two more sections of that book are useful reading: [Stable numerics](../../mathematics-for-computing/10-numerical-methods-and-floating-point/02-stable-numerics.md) defines the masked softmax used here, and the [matrix calculus problem set](../../mathematics-for-computing/09-matrix-calculus-and-autodiff/05-problem-set.md) works its vector-Jacobian product by hand.

The solution is [`python/tinyllm/seq2seq/additive.py`](../../../../course/solutions/python/tinyllm/seq2seq/additive.py), with `length_mask` and `AdditiveAttention` (`project_keys`, `scores`, `forward`). It implements the contract [`course/contracts/py/tinyllm/seq2seq/additive.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/additive.pyi), and the oracle tests are in [`test_additive.py`](../../../../course/solutions/learner-tests/L4.2/python/tests/l4-2-additive/test_additive.py). Plugged into the encoder-decoder, a Bahdanau seq2seq reaches an exact match of at least 0.97 on the dates task.

> **Learning objectives**
> - Write the additive score $e_s = v^\top \tanh(W q + U k_s + b)$ and explain why it must be nonlinear in the pair $(q, k_s)$.
> - Mask padding on the scores, to $-\infty$, and show that padded weights are exactly 0 and padding content never matters.
> - Handle an empty source without producing NaN.
> - Precompute the key projection once per sentence and say what it saves per step.

## Notation

The symbols and shapes used throughout:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $B$, $S$ | batch size, source length (padded) | `int` |
| $q$ | the query: the decoder state that asks | `float32[B, Dq]` |
| $k_s$ | the key at source position $s$: an encoder output, also the value read | `float32[B, S, Dk]` |
| $\ell_b$ | the real length of source $b$ | `int` |
| $m_{bs}$ | mask: true when $s < \ell_b$ | `bool[B, S]` |
| $W$, $U$, $b$, $v$ | parameters `query.weight` $[A, Dq]$, `key.weight` $[A, Dk]$, `key.bias` $[A]$, `v.weight` $[1, A]$ | `float32` |
| $A$ | `d_attn`, the width of the scoring network | `int` |
| $e_s$ | the score of position $s$ | `float` |
| $a_s$ | the attention weight of position $s$ | `float` in $[0, 1]$ |
| $c$ | the context: $\sum_s a_s k_s$ | `float32[B, Dk]` |

## Reading by weighted average

A decoder that must write the next target word needs some of the source, not all of it equally. A **weighted average** of the encoder outputs, with weights that are nonnegative and sum to 1, reads exactly that. A weight near 1 on one position copies that position; spread weights blend several. The read is a convex combination, so the context is a point "between" the encoder outputs. The weights must depend on what the decoder is doing now, so they are computed from the decoder state $q$, and a softmax over per-position scores is the standard map from real scores to such weights.

## Scores, weights, context

What should the score be? It must depend on both the query and the key, and nonlinearly. A linear score $w^\top q + u^\top k_s$ would rank positions the same way for every query, because the $q$ term is constant across $s$, so the decoder could never change where it looks. Adding $W q$ and $U k_s$ inside a $\tanh$ and reading out with $v$ is the smallest network that couples them. Bahdanau scores each position with exactly this one-hidden-layer network:

$$e_s = v^\top \tanh(W q + U k_s + b), \qquad a = \operatorname{softmax}(e), \qquad c = \sum_s a_s k_s .$$

The softmax turns arbitrary real scores into positive weights that sum to 1. Every operation is an op-library call, so backward reaches $q$ (the decoder learns where to look), the keys (the encoder learns what to offer), and $W, U, b, v$, with no hand-written backward.

Notice two things about the formula. The context has the keys' width $D_k$, not $A$: the scoring network decides **how much** of each key to take, and the keys themselves are what is read. And nothing in the formula looks at the position $s$ itself. Permute the keys and the weights permute with them, so attention by itself ignores order; order has to come from the encoder.

## Masking padding

A batch pads every source to the longest length $S$, and padded positions must never be read. The only way to give a position weight 0 while keeping the rest summing to 1 is to remove it *before* normalizing: set its **score** to $-\infty$. Since $e^{-\infty} = 0$, its weight is exactly 0 and the real weights still sum to 1. `length_mask(lengths, S)` builds $m_{bs} = (s < \ell_b)$.

Masking after the softmax, by multiplying the weights by the mask, gets this wrong: the real weights then sum to less than 1, and padding still influences the normalization.

An empty source raises one more question. A row with no real position is all $-\infty$, and a naive softmax computes $0/0 = $ NaN. The masked softmax of [Stable numerics](../../mathematics-for-computing/10-numerical-methods-and-floating-point/02-stable-numerics.md) defines an all-masked row as zeros, so the context is 0 and no gradient flows back. That is the right meaning of "nothing to read".

## Precomputing the keys

The score splits by what changes when. $U k_s + b$ involves only the keys, which are fixed for the whole sentence, while $W q$ changes at every decoder step. Computing $U k_s + b$ once (`project_keys`) and passing it to every step as `proj` saves $S \cdot A \cdot D_k$ multiply-adds per step; Bahdanau's appendix points this out. The result must be identical to recomputing.

The same split decides where the biases go. $U$ carries the bias $b$; $W$ and $v$ have none. A bias in $W q$ would duplicate $b$, and a bias in $v$ would shift every score equally and cancel in the softmax.

## Working it by hand

Make every size 1 ($D_q = D_k = A = 1$), with $W = U = v = 1$ and $b = 0$. Take the query $q = 0$, keys $k = (0, 1, 2)$, and let the third position be padding (mask `[True, True, False]`).

1. Scores before the mask: $\tanh(0 + 0) = 0$, $\tanh(1) = 0.761594$, $\tanh(2) = 0.964028$. The padded one would be the largest.
2. After the mask: $(0, 0.761594, -\infty)$.
3. Softmax: $e^0 = 1$, $e^{0.761594} = 2.141688$, $e^{-\infty} = 0$; the sum is $3.141688$, so $a = (0.318300, 0.681700, 0)$.
4. Context: $c = 0.318300 \cdot 0 + 0.681700 \cdot 1 + 0 \cdot 2 = 0.681700$.

The padded key, the biggest value in the row, contributes nothing. A correct solution reproduces these numbers to float32 tolerance.

## Designing the API

```python
def length_mask(lengths: ArrayLike, S: int) -> NDArray: ...          # bool [B, S]
class AdditiveAttention(Module):
    query_from = "previous"                                           # read by the seq2seq decoder
    def __init__(self, d_query: int, d_key: int, d_attn: int, rng=None) -> None: ...
    def project_keys(self, keys: Tensor) -> Tensor: ...               # [B, S, A]
    def scores(self, query: Tensor, keys: Tensor, proj=None) -> Tensor: ...   # [B, S], unmasked
    def forward(self, query, keys, mask, proj=None) -> tuple[Tensor, Tensor]: ...  # context, weights
```

`query_from = "previous"` tells the encoder-decoder to read with $s_{t-1}$, before the cell. `scores` returns unmasked scores; `forward` applies the mask and returns both the context and the weights, so the decoder can draw alignments.

## Building it step by step

### The mask

The mask is one broadcast comparison:

```python
def length_mask(lengths, S):
    n = np.asarray(lengths)
    # ... (1-D integers in [0, S]; 0 is allowed: an empty source)
    return np.arange(S)[None, :] < n[:, None]      # [B, S], True where s < l_b
```

Lengths of 0 are allowed here, unlike in the encoder, because attention has a well-defined answer for an empty source. The comparison is strict: `<=` would read one padding position per row.

### The parameters

The parameters follow the split above: only `key` has a bias.

```python
class AdditiveAttention(Module):
    query_from = "previous"            # the decoder reads with s_{t-1}, before the cell

    def __init__(self, d_query, d_key, d_attn, rng=None):
        # ... (positive sizes; default rng PCG32(0).substream("init"))
        self.query = Linear(d_query, d_attn, bias=False, rng=r)   # W
        self.key = Linear(d_key, d_attn, rng=r)                   # U, b
        self.v = Linear(d_attn, 1, bias=False, rng=r)             # v

    def project_keys(self, keys):
        # ... (keys must be [B, S, d_key])
        return self.key(keys)          # U k + b, the step-independent part; no tanh yet
```

`project_keys` stops before the $\tanh$ on purpose. The query term must be added *inside* the nonlinearity; caching $\tanh(U k + b)$ would compute $\tanh(U k + b) + W q$, a different function.

### The scores

The scores broadcast one $W q$ per batch row over all $S$ positions:

```python
def scores(self, query, keys, proj=None):
    # ... (query must be [B, d_query])
    B, S = keys.shape[0], keys.shape[1]
    if proj is None:
        proj = self.project_keys(keys)
    elif proj.shape != (B, S, self.d_attn):
        raise ValueError(...)
    # W q is one vector per batch row, added to every source position.
    wq = F.reshape(self.query(query), (B, 1, self.d_attn))
    e = self.v(F.tanh(proj + wq))  # [B, S, 1]
    return F.reshape(e, (B, S))    # unmasked scores
```

### The read

Finally, mask, softmax, and the weighted sum as a batched matmul:

```python
def forward(self, query, keys, mask, proj=None):
    # ... (shape checks)
    B, S = keys.shape[0], keys.shape[1]
    m = _check_mask(mask, B, S)                    # bool [B, S]
    e = F.masked_fill(self.scores(query, keys, proj), ~m, -np.inf)
    # Mask BEFORE the softmax: a masked score of -inf has weight exactly 0
    # and the rest still sum to 1.
    a = F.softmax(e, axis=-1)
    ctx = F.matmul(F.reshape(a, (B, 1, S)), keys)  # [B, 1, S] @ [B, S, Dk] = [B, 1, Dk]
    return F.reshape(ctx, (B, self.d_key)), a
```

Because `masked_fill` writes a constant, padded keys receive no gradient through the scores, and because their weight is exactly 0 they receive none through the context either: garbage in padding cannot change anything. The context is computed from the live `keys` Tensor, so the encoder gets its gradient through attention. The full solution is [`additive.py`](../../../../course/solutions/python/tinyllm/seq2seq/additive.py).

### Why this design

Per decoder step, with `proj` precomputed, the cost is $O(B A D_q)$ for $W q$, $O(B S A)$ for the $\tanh$ and $v$, and $O(B S D_k)$ for the context. Without precomputation each step would add $O(B S A D_k)$, the dominant term for long sources.

Why not dot-product scores $q^\top k_s$? They need $D_q = D_k$ (or one projection) and no hidden layer, so they are cheaper; [Luong attention](03-luong-attention-input-feeding.md) and the transformer use them. Additive scores were the first to work and handle mismatched widths naturally; the price is the extra $A$-wide layer per position.

Why $-\infty$ and not a large negative number? $-10^9$ works when at least one position is real, but on an empty row it gives uniform weights over padding instead of zeros. And why mask the scores rather than the weights? Multiplying weights by the mask after the softmax leaves the real weights summing to less than 1 and lets padded scores shape the normalizer.

## Testing it

The oracle in [`test_additive.py`](../../../../course/solutions/learner-tests/L4.2/python/tests/l4-2-additive/test_additive.py) is the formula itself, written out in numpy with the module's own weights read from `state_dict()`. It computes $e$, the mask, the softmax, and the context in float64 and compares them with `forward`. It adds the hand-worked numbers, a finite-difference check of the gradients of the query and the keys in float64, the masking properties, and the parameter names, and it imports only the contract modules.

Taken together, the checks a correct solution must pass are these:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example weights and context | unit | the hand-worked numbers to float32 tolerance | pins down the formula |
| length mask | unit | `s < lengths[b]`, bad lengths raise | the only place lengths enter attention |
| matches torch | golden | context, weights, and gradients of the query, keys, and all four parameters against torch, lengths 5, 3, 1 | the encoder and decoder train through this |
| gradcheck of every input and parameter | gradcheck | float64 central differences with a padded row | both sides of the model learn |
| masked weights are exactly zero and rows sum to one | property | exact zeros, positive real weights, rows sum to 1 | padding is never read |
| padding content never matters | property | garbage in padded keys changes nothing bitwise; their gradient is 0 | a batch padded differently translates the same |
| fully masked row reads nothing | boundary | zeros, no NaN, no gradient | empty inputs in a batch |
| precomputed keys equal recomputed | property | `proj` passed or recomputed, bitwise equal | the decoder passes it at every step |
| permuting the source permutes the weights | property | weights permute, context unchanged | order comes from the encoder, not from attention |
| parameter names and shapes | unit | keys, shapes, order, `query_from`, seeded init | the safetensors keys of seq2seq checkpoints |
| validation | boundary | wrong widths, missing source axis, wrong or float mask, wrong `proj` | wiring bugs fail loudly |

## What can go wrong

Most attention bugs still produce weights that look plausible, which is why the tests check exact properties. The common ones:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| masking after the softmax | real weights no longer sum to 1; padding still shapes the normalization | `F.masked_fill(..., ~m, -np.inf)` on the scores, then `F.softmax` |
| masking with $-10^9$ instead of $-\infty$ | an empty source gets uniform weights and reads padding | fills with `-np.inf`; the masked softmax returns zeros for an all-$-\infty$ row |
| softmax over the wrong axis | weights sum to 1 over the batch, not the source | `F.softmax(e, axis=-1)`: the source axis of `[B, S]` |
| reading detached keys | the encoder gets no gradient through attention | the context is `F.matmul(a, keys)` on the live `keys` Tensor |
| mask polarity inverted | only padding is read | fills where `~m`, the padding |
| no `tanh` | a linear score: the golden values disagree | `self.v(F.tanh(proj + wq))` |
| `<=` in `length_mask` | one padding position is read per row | `np.arange(S)[None, :] < n[:, None]`, strict |
| caching $\tanh(U k + b)$ instead of $U k + b$ | the query is added after the nonlinearity | `project_keys` returns `self.key(keys)`, before any `tanh` |

> **Warning:** A large negative fill such as $-10^9$ passes every test with at least one real position and silently reads padding on an empty source. Use $-\infty$ and a softmax that defines the all-masked row.

## Where this leads

[Luong attention](03-luong-attention-input-feeding.md) keeps the same masked read and builds its masks with `length_mask`. The [encoder-decoder](01-encoder-decoder-teacher-forcing.md) calls `attention(s_{t-1}, keys, mask, proj)` before each step. In the next part, [scaled dot-product attention](../06-transformer-2017/01-scaled-dot-product-attention.md) is this read with a cheaper score, and [the model zoo](../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) includes the attention variants of the seq2seq rows.

Production systems generalize each piece:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `AdditiveAttention` | `torch.nn.MultiheadAttention`, `F.scaled_dot_product_attention` | dot-product scores (no hidden layer), many heads, fused kernels | `torch/nn/functional.py` (`multi_head_attention_forward`) |
| masking to $-\infty$ | `attn_mask` and `key_padding_mask` in torch | boolean and additive masks, broadcast over heads | PyTorch SDPA documentation |
| `project_keys` | the KV cache | keys and values projected once and reused at every step of generation | [KV cache and generate](../09-inference/02-kv-cache-and-generate.md); vLLM's paged KV cache |
| additive scores | location-sensitive attention (Tacotron 2) | adds the previous weights as a feature so speech alignment moves forward | Chorowski et al. 2015 |

### Further reading

- Bahdanau, Cho, and Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (ICLR 2015), section 3.1 and appendix A.1.2.
- Graves, "Generating Sequences With Recurrent Neural Networks" (2013), section 5.
- Chorowski et al., "Attention-Based Models for Speech Recognition" (2015).

## Key terms

- **attention**: a weighted average of encoder outputs whose weights the decoder computes from its state.
- **query**: the decoder state that asks where to look.
- **key**: an encoder output, scored against the query and also read as the value.
- **additive score**: $v^\top \tanh(W q + U k_s + b)$, a one-hidden-layer network on the pair $(q, k_s)$.
- **context**: the weighted sum $\sum_s a_s k_s$ that the decoder reads.
- **length mask**: the boolean array $m_{bs} = (s < \ell_b)$ marking real source positions.

## Summary

- Attention is a weighted average of the encoder outputs whose weights the decoder computes: a score per source position, a softmax, a sum.
- Padding is masked on the **scores**, to $-\infty$, before the softmax: padded weights are exactly 0, the rest sum to 1, and padding content never matters.
- An empty source reads nothing: weights and context 0, not NaN.
- The key projection $U k_s + b$ does not depend on the decoder step, so it is computed once per sentence.
- Attention by itself ignores order: permute the source and the weights permute with it.

## Review questions

1. Why does `query` have no bias while `key` does?
   <details><summary>Answer</summary>Both are added inside the same $\tanh$, so one bias is enough. Putting it on the key side makes it part of the precomputed `proj`, so it costs nothing per step.</details>
2. Why is `project_keys` $U k + b$ and not $\tanh(U k + b)$?
   <details><summary>Answer</summary>The query term must be added *inside* the nonlinearity: $\tanh(W q + U k + b)$. Caching the $\tanh$ would compute $\tanh(U k + b) + W q$, a different function.</details>
3. In the worked example the padded key had the highest raw score. Why does it get weight exactly 0?
   <details><summary>Answer</summary>Its score is replaced by $-\infty$ before the softmax and $e^{-\infty} = 0$. The remaining two weights are normalized among themselves: $1/3.1417$ and $2.1417/3.1417$.</details>
4. What do `forward`'s outputs look like for a row with length 0, and why is that correct?
   <details><summary>Answer</summary>Weights and context are all zeros, with no NaN and no gradient. There is nothing to read, and the masked softmax defines an all-$-\infty$ row as zeros instead of $0/0$.</details>
5. Why does permuting the source permute the weights but leave the context unchanged?
   <details><summary>Answer</summary>Each score depends only on $(q, k_s)$, not on $s$, so permuting keys permutes scores and weights with them. The context is a sum over positions, and sums do not depend on order.</details>
6. Why would a linear score $w^\top q + u^\top k_s$ be useless?
   <details><summary>Answer</summary>The $w^\top q$ term is the same for every position $s$, so it cancels in the softmax and the ranking of positions never depends on the query. The decoder could not change where it looks.</details>
