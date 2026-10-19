# Bahdanau additive attention

## Overview

| | |
|---|---|
| **Module** | `L4.2` · build · Python · Pass 4 · 1.5 h reading |
| **Solution** | [`python/tinyllm/seq2seq/additive.py`](../../../../course/solutions/python/tinyllm/seq2seq/additive.py): `length_mask` and `AdditiveAttention` (`project_keys`, `scores`, `forward`); model oracle tests in [`test_additive.py`](../../../../course/solutions/learner-tests/L4.2/python/tests/l4-2-additive/test_additive.py) |
| **Contract** | [`course/contracts/py/tinyllm/seq2seq/additive.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/additive.pyi) |
| **Needs** | `L0.4` `Linear`, `Module` · `L0.2` `F.tanh`, `F.masked_fill`, `F.softmax`, `F.matmul` · `L0.1` `Tensor` · `M06.3` `PCG32` (default initialization) · reading: `M09.2` (the masked softmax), `S-M08` (its VJP by hand) |
| **Used by** | `L4.3` builds masks with `length_mask` · `L4.1` plugs `AdditiveAttention` into the decoder · later `L6.7` the zoo's attention rows · on the dates task, a Bahdanau seq2seq reaches an exact match of at least 0.97 |
| **Optional depth** | Bahdanau, Cho, and Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (ICLR 2015), section 3.1 and appendix A.1.2; Graves, "Generating Sequences With Recurrent Neural Networks" (2013), section 5 |

## Key Takeaways

- Attention is a weighted average of the encoder outputs whose weights the decoder computes: a score per source position, a softmax, a sum.
- Padding is masked on the **scores**, to $-\infty$, before the softmax: padded weights are exactly 0, the rest sum to 1, and padding content never matters.
- An empty source reads nothing: weights and context 0, not NaN.
- The key projection $U k_s + b$ does not depend on the decoder step, so it is computed once per sentence.
- Attention by itself ignores order: permute the source and the weights permute with it.

## How to read this chapter

Sections 1 to 3 build additive attention from first principles and work a masked example by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The encoder-decoder (`L4.1`, next in this pass) squeezes the whole source sentence into one vector, the decoder's first state. On short inputs that works; on a 30-character date string the decoder has forgotten the start of the input by the time it writes the end, and exact match drops sharply with length. Bahdanau's fix, the first attention mechanism in neural translation, lets every decoder step look back at **all** encoder outputs and choose which to read. It is the idea the 2017 transformer (`L5.1`) keeps while dropping the recurrence, so this is where the core operation of every model in the rest of the course is born.

## 2. Principles

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

### 2.1 Reading by weighted average

A decoder that must write the next target word needs some of the source, not all of it equally. A **weighted average** of the encoder outputs, with weights that sum to 1, reads exactly that: a weight near 1 on one position copies that position; spread weights blend several. The weights must depend on what the decoder is doing now, so they are computed from the decoder state $q$.

### 2.2 Scores, weights, context

Bahdanau scores each position with a one-hidden-layer network on the pair $(q, k_s)$:

$$e_s = v^\top \tanh(W q + U k_s + b), \qquad a = \operatorname{softmax}(e), \qquad c = \sum_s a_s k_s .$$

The softmax (`M09.2`) turns arbitrary real scores into positive weights that sum to 1. Every operation is an op-library call (`L0.2`), so backward reaches $q$ (the decoder learns where to look), the keys (the encoder learns what to offer), and $W, U, b, v$. The context has the keys' width $D_k$, not $A$: the scoring network decides **how much** of each key to take, the keys themselves are what is read. Nothing in the formula looks at the position $s$ itself: permute the keys and the weights permute with them. Order has to come from the encoder.

### 2.3 Masking padding

A batch pads every source to the longest length $S$. Padded positions must never be read. The rule is to set their **scores** to $-\infty$ before the softmax: $e^{-\infty} = 0$, so their weights are exactly 0 and the real weights still sum to 1. `length_mask(lengths, S)` builds $m_{bs} = (s < \ell_b)$. Masking after the softmax (multiplying the weights by the mask) leaves the real weights summing to less than 1 and lets padding influence the normalization. A row with no real position at all (an empty source) is all $-\infty$; `M09.2` defines its softmax as zeros, so the context is 0 and no gradient flows back, where a naive softmax would give $0/0 = $ NaN.

### 2.4 Precomputing the keys

$U k_s + b$ involves only the keys, which are fixed for the whole sentence, while $W q$ changes at every decoder step. Computing $U k_s + b$ once (`project_keys`) and passing it to every step as `proj` saves $S \cdot A \cdot D_k$ multiply-adds per step; Bahdanau's appendix points this out. The result must be identical to recomputing.

## 3. Worked example by hand

Every size 1 ($D_q = D_k = A = 1$), $W = U = v = 1$, $b = 0$. Query $q = 0$, keys $k = (0, 1, 2)$, and the third position is padding (mask `[True, True, False]`).

1. Scores before the mask: $\tanh(0 + 0) = 0$, $\tanh(1) = 0.761594$, $\tanh(2) = 0.964028$. The padded one would be the largest.
2. After the mask: $(0, 0.761594, -\infty)$.
3. Softmax: $e^0 = 1$, $e^{0.761594} = 2.141688$, $e^{-\infty} = 0$; the sum is $3.141688$, so $a = (0.318300, 0.681700, 0)$.
4. Context: $c = 0.318300 \cdot 0 + 0.681700 \cdot 1 + 0 \cdot 2 = 0.681700$.

The padded key, the biggest value in the row, contributes nothing. A correct solution reproduces these numbers to float32 tolerance.

## 4. The interface

```python
def length_mask(lengths: ArrayLike, S: int) -> NDArray: ...          # bool [B, S]
class AdditiveAttention(Module):
    query_from = "previous"                                           # read by L4.1's decoder
    def __init__(self, d_query: int, d_key: int, d_attn: int, rng=None) -> None: ...
    def project_keys(self, keys: Tensor) -> Tensor: ...               # [B, S, A]
    def scores(self, query: Tensor, keys: Tensor, proj=None) -> Tensor: ...   # [B, S], unmasked
    def forward(self, query, keys, mask, proj=None) -> tuple[Tensor, Tensor]: ...  # context, weights
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example weights and context | unit | section 3 to float32 tolerance | pins down the formula |
| length mask | unit | `s < lengths[b]`, bad lengths raise | the only place lengths enter attention |
| matches torch | golden | context, weights, and gradients of the query, keys, and all four parameters against torch, lengths 5, 3, 1 | the encoder and decoder train through this |
| gradcheck of every input and parameter | gradcheck | float64 central differences with a padded row | both sides of the model learn |
| masked weights are exactly zero and rows sum to one | property | exact zeros, positive real weights, rows sum to 1 | padding is never read |
| padding content never matters | property | garbage in padded keys changes nothing bitwise; their gradient is 0 | a batch padded differently translates the same |
| fully masked row reads nothing | boundary | zeros, no NaN, no gradient | empty inputs in a batch |
| precomputed keys equal recomputed | property | `proj` passed or recomputed, bitwise equal | `L4.1` passes it at every step |
| permuting the source permutes the weights | property | weights permute, context unchanged | order comes from the encoder, not from attention |
| parameter names and shapes | unit | keys, shapes, order, `query_from`, seeded init | the safetensors keys of seq2seq checkpoints |
| validation | boundary | wrong widths, missing source axis, wrong or float mask, wrong `proj` | wiring bugs fail loudly |

### The oracle tests

The oracle in the model suite [`test_additive.py`](../../../../course/solutions/learner-tests/L4.2/python/tests/l4-2-additive/test_additive.py) is the formula itself, written out in numpy with the module's own weights read from `state_dict()`: it computes $e$, the mask, the softmax and the context in float64 and compares with `forward`. It adds the section 3 numbers, a finite-difference check of the gradients of the query and the keys in float64, the masking properties, and the parameter names, and imports only the contract modules.

## 5. The solution, step by step

### From first principles

1. **The read must be a convex combination of the keys.** Weights $\ge 0$ summing to 1 (section 2.1) make the context a point "between" the encoder outputs; a softmax over per-position scores is the standard map from real scores to such weights.
2. **The score must depend on both the query and the key, nonlinearly.** A linear score $w^\top q + u^\top k_s$ would rank positions the same way for every query (the $q$ term is constant across $s$), so the decoder could never change where it looks. Adding $W q$ and $U k_s$ inside a $\tanh$ and reading out with $v$ is the smallest network that couples them.
3. **Split the score by what changes when.** $U k_s + b$ depends only on the sentence; $W q$ changes every step. So $U$ carries the bias, $W$ and $v$ have none (a bias in $W q$ would duplicate $b$; a bias in $v$ would shift every score equally and cancel in the softmax), and $U k + b$ is computed once (`project_keys`).
4. **Padding must vanish exactly.** The only way to give a position weight 0 while keeping the rest summing to 1 is to remove it *before* normalizing: score $-\infty$. In the worked example the padded key had the largest raw score (0.964) and still got weight 0.
5. **An empty row must not produce NaN.** With every score $-\infty$, $\operatorname{softmax}$ is $0/0$. Relying on `M09.2`'s convention (zeros for an all-masked row) gives a zero context and zero gradients, which is the right meaning of "nothing to read".
6. **Everything is built from autograd ops**, so gradients reach the query, the keys and every parameter without a hand-written backward.

### The code, annotated

The mask is one broadcast comparison:

```python
def length_mask(lengths, S):
    n = np.asarray(lengths)
    # ... (1-D integers in [0, S]; 0 is allowed: an empty source)
    return np.arange(S)[None, :] < n[:, None]      # [B, S], True where s < l_b
```

Lengths of 0 are allowed here (unlike the encoder) because attention has a well-defined answer for an empty source.

The parameters follow step 3: only `key` has a bias.

```python
class AdditiveAttention(Module):
    query_from = "previous"            # L4.1 reads with s_{t-1}, before the cell

    def __init__(self, d_query, d_key, d_attn, rng=None):
        # ... (positive sizes; default rng PCG32(0).substream("init"))
        self.query = Linear(d_query, d_attn, bias=False, rng=r)   # W
        self.key = Linear(d_key, d_attn, rng=r)                   # U, b
        self.v = Linear(d_attn, 1, bias=False, rng=r)             # v

    def project_keys(self, keys):
        # ... (keys must be [B, S, d_key])
        return self.key(keys)          # U k + b, the step-independent part; no tanh yet
```

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

The read: mask, softmax, weighted sum as a batched matmul.

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

Because `masked_fill` writes a constant, padded keys receive no gradient through the scores, and their weight is exactly 0, so they receive none through the context either: garbage in padding cannot change anything. Full solution: [`additive.py`](../../../../course/solutions/python/tinyllm/seq2seq/additive.py).

### Why this solution

- **Cost per decoder step.** With `proj` precomputed: $O(B A D_q)$ for $W q$, $O(B S A)$ for the $\tanh$ and $v$, $O(B S D_k)$ for the context. Without precomputation each step would add $O(B S A D_k)$, the dominant term for long sources.
- **Why not dot-product scores $q^\top k_s$?** They need $D_q = D_k$ (or one projection) and no hidden layer, so they are cheaper; Luong (`L4.3`) and the transformer (`L5.1`) use them. Additive scores were the first to work and handle mismatched widths naturally; the price is the extra $A$-wide layer per position.
- **Why $-\infty$ and not a large negative number?** $-10^9$ works when at least one position is real, but on an empty row it gives uniform weights over padding instead of zeros (pitfall 2).
- **Why mask the scores and not the weights?** Multiplying weights by the mask after the softmax leaves the real weights summing to less than 1 and lets padded scores shape the normalizer.

### Explain it back

1. Why does `query` have no bias while `key` does?
   <details><summary>Answer</summary>Both are added inside the same $\tanh$, so one bias is enough. Putting it on the key side makes it part of the precomputed `proj`, so it costs nothing per step.</details>
2. Why is `project_keys` $U k + b$ and not $\tanh(U k + b)$?
   <details><summary>Answer</summary>The query term must be added *inside* the nonlinearity: $\tanh(W q + U k + b)$. Caching the $\tanh$ would compute $\tanh(U k + b) + W q$, a different function.</details>
3. In the worked example the padded key had the highest raw score. Why does it get weight exactly 0?
   <details><summary>Answer</summary>Its score is replaced by $-\infty$ before the softmax and $e^{-\infty} = 0$. The remaining two weights are normalized among themselves: $1/3.1417$ and $2.1417/3.1417$.</details>
4. What do `forward`'s outputs look like for a row with length 0, and why is that correct?
   <details><summary>Answer</summary>Weights and context are all zeros, with no NaN and no gradient. There is nothing to read, and `M09.2`'s softmax defines an all-$-\infty$ row as zeros instead of $0/0$.</details>
5. Why does permuting the source permute the weights but leave the context unchanged?
   <details><summary>Answer</summary>Each score depends only on $(q, k_s)$, not on $s$, so permuting keys permutes scores and weights with them. The context is a sum over positions, and sums do not depend on order.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. masking after the softmax | real weights no longer sum to 1; padding still shapes the normalization | `F.masked_fill(..., ~m, -np.inf)` on the scores, then `F.softmax` |
| 2. masking with $-10^9$ instead of $-\infty$ | an empty source gets uniform weights and reads padding | fills with `-np.inf`; `M09.2`'s softmax returns zeros for an all-$-\infty$ row |
| 3. softmax over the wrong axis | weights sum to 1 over the batch, not the source | `F.softmax(e, axis=-1)`: the source axis of `[B, S]` |
| 4. reading detached keys | the encoder gets no gradient through attention | the context is `F.matmul(a, keys)` on the live `keys` Tensor |
| mask polarity inverted | only padding is read | fills where `~m`, the padding |
| no `tanh` | a linear score: the golden values disagree | `self.v(F.tanh(proj + wq))` |
| `<=` in `length_mask` | one padding position is read per row | `np.arange(S)[None, :] < n[:, None]`, strict |
| caching $\tanh(U k + b)$ instead of $U k + b$ | the query is added after the nonlinearity | `project_keys` returns `self.key(keys)`, before any `tanh` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.4` | `Linear` holds $W$, $U$, $b$, $v$ |
| Back | `L0.2` | `masked_fill`, `softmax`, `tanh`, `matmul` give the backward for free |
| Back | `L0.1` | `Tensor`, the type of queries, keys, and weights |
| Back | `M06.3` | `PCG32` initializes the layers when no rng is given |
| Forward | `L4.3` | Luong attention keeps the same masked read and builds its masks with `length_mask` |
| Forward | `L4.1` | the decoder calls `attention(s_{t-1}, keys, mask, proj)` before each step |
| Forward | `L5.1` | scaled dot-product attention is this read with a cheaper score |
| Forward | `L6.7` | the zoo's attention variants of the seq2seq rows |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `AdditiveAttention` | `torch.nn.MultiheadAttention`, `F.scaled_dot_product_attention` | dot-product scores (no hidden layer), many heads, fused kernels | `torch/nn/functional.py` (`multi_head_attention_forward`) |
| masking to $-\infty$ | `attn_mask` and `key_padding_mask` in torch | boolean and additive masks, broadcast over heads | PyTorch SDPA documentation |
| `project_keys` | the KV cache | keys and values projected once and reused at every step of generation | `L8.2` in this course; vLLM's paged KV cache |
| additive scores | location-sensitive attention (Tacotron 2) | adds the previous weights as a feature so speech alignment moves forward | Chorowski et al. 2015 |
