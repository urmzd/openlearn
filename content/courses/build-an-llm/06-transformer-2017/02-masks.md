# Masks: causal, padding, sliding window, additive

[Scaled dot-product attention](01-scaled-dot-product-attention.md) lets every position read every other position, which is exactly wrong for the language models built next. A language model is trained to predict token $t + 1$ from tokens up to $t$. If position $t$ can read position $t + 1$, training loss falls to nearly zero by copying, and generation, which has no future to copy, produces noise. In a batch, short sequences are padded, and a query that reads padding learns from garbage. Long contexts in the modern block read only a recent window.

All three problems have the same solution: a boolean matrix of allowed (query, key) pairs, turned into $0$ or $-\infty$ added to the scores. Getting the off-by-ones right here is the difference between a model that trains and one that cheats, and the decode path of the inference chapter reuses the same function with an offset. In this section you will build causal, padding, and sliding-window masks from one convention, combine them, and convert them to the additive form the softmax consumes.

Nothing here calls earlier code. For background it helps to know AND and implication from the [discrete math 1 problem set](../../mathematics-for-computing/06-discrete-math-1/03-problem-set.md) and the fact, from [Stable numerics: logsumexp, softmax, and compensated sums](../../mathematics-for-computing/10-numerical-methods-and-floating-point/02-stable-numerics.md) in *Mathematics for Computing*, that the stable softmax maps an all-$-\infty$ row to zeros. The full solution is [`python/tinyllm/xfmr/masks.py`](../../../../course/solutions/python/tinyllm/xfmr/masks.py) (`causal_mask`, `padding_mask`, `sliding_window_mask`, `combine`, `to_additive`), its contract is [`course/contracts/py/tinyllm/xfmr/masks.pyi`](../../../../course/contracts/py/tinyllm/xfmr/masks.pyi), and the model oracle tests are in [`test_masks.py`](../../../../course/solutions/learner-tests/L5.2/python/tests/l5-2-masks/test_masks.py).

> **Learning objectives**
> - Write the causal, padding, and sliding-window rules as predicates on absolute positions, and build each with one broadcast comparison.
> - Use `q_offset` so that a decode chunk's mask is exactly the last rows of the full training mask.
> - Combine masks by AND with broadcasting into one `[B, 1, T, T]` mask for every head.
> - Explain why the additive form must use $-\infty$ and not a large negative number.

## The notation

These symbols recur through the section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T_q$, $T_k$ | number of queries and keys | `int` |
| $i$, $j$ | query row and key column | `int` |
| $q_{\text{off}}$ (`q_offset`) | absolute position of query 0; query $i$ is at $q_{\text{off}} + i$ | `int` $\ge 0$ |
| $M$ | a boolean mask, True = may attend | `bool[Tq, Tk]` (or broadcastable) |
| $\ell_b$ | real length of sequence $b$ in a padded batch | `int` |
| $w$ (`window`) | sliding-window size | `int` $\ge 1$ |
| $A$ | the additive mask: 0 where $M$ is True, $-\infty$ where False | `float[Tq, Tk]` |
| $S$ | attention scores $QK^\top/\sqrt{d}$ | `float[..., Tq, Tk]` |

## One convention

Every mask in the system means the same thing: entry $(i, j)$ True means "query $i$ may read key $j$". This is the convention of `torch.nn.functional.scaled_dot_product_attention`.

> **Warning:** `torch.nn.MultiheadAttention` uses the opposite convention (True = blocked). Booleans are the interface here, so the functions reject float or integer masks rather than guessing which convention a 0/1 array meant.

## Causal masks and q_offset

With the convention fixed, the first rule is causality. Position $p$ may read positions $\le p$, itself included, since it needs its own token to predict the next one. With queries at absolute positions $q_{\text{off}} + i$ and keys at $j$:

$$M_{ij} = [\,j \le q_{\text{off}} + i\,].$$

During training $T_q = T_k = T$ and $q_{\text{off}} = 0$, which gives the lower triangle with the diagonal. During generation with a [KV cache](../09-inference/02-kv-cache-and-generate.md), the keys of all $T_k$ tokens so far are cached, and a step computes only the last $T_q$ queries, at positions $T_k - T_q, \ldots, T_k - 1$. Then $q_{\text{off}} = T_k - T_q$, and the triangle is aligned to the **bottom right**, so the last query sees every key. Torch ships both alignments (`causal_upper_left`, `causal_lower_right`); here one parameter covers both. The property to test is that the chunk's mask is exactly the last $T_q$ rows of the full mask, so cached and recomputed attention agree.

Because $-\infty$ contributes exact zeros (more on this below), causality holds to the bit: changing future tokens must leave past outputs **bitwise** unchanged.

## Padding and combining

A batch stacks sequences of lengths $\ell_b$ padded to $T$. The key-padding mask is $P_{bt} = [t < \ell_b]$, shape `[B, T]`. To use it on scores of shape `[B, H, Tq, Tk]`, index it as `P[:, None, None, :]`: one row per sequence, the same for every head and every query.

Rules compose by **AND**, with numpy broadcasting. `combine(causal_mask(T), P[:, None, None, :])` has shape `[B, 1, T, T]` and opens $(i, j)$ only if $j \le i$ and $j < \ell_b$. Padded *queries* still produce outputs, but the loss ignores them, so they need no mask.

## Sliding windows

Attention cost grows with $T_q T_k$. A sliding window keeps only the last $w$ keys:

$$M_{ij} = [\,j \le q_{\text{off}} + i\,] \wedge [\,(q_{\text{off}} + i) - j < w\,].$$

Each query sees $\min(w, p + 1)$ keys, its own included. A stack of $L$ windowed layers still has a receptive field of about $L w$ tokens, which is how Mistral reads long contexts with a 4096-token window. A window of at least $T$ is the causal mask; a window without the causal half lets queries read $w - 1$ future tokens.

## Additive masks

The softmax takes scores, not booleans, so the last step converts. Add $A_{ij} = 0$ where allowed and $-\infty$ where blocked. Since $e^{-\infty} = 0$ exactly, blocked keys get weight exactly 0, and allowed scores are untouched (adding $0.0$ changes nothing, not even the sign of a zero). If every key of a row is blocked, the row is all $-\infty$, and the stable softmax defines that as zeros.

A "large negative" such as $-10^9$ looks equivalent but is not. In an all-blocked row every entry is $-10^9$, the softmax subtracts the maximum, and the row becomes uniform: the query averages over keys it was forbidden to read. In float16, $-10^9$ is not even representable (it rounds to $-\infty$ anyway). Exactness is also what makes causality bitwise: perturbing a future token changes its score, but the score is masked to $-\infty$ and its value vector is multiplied by an exact 0.

## Working it by hand

Here is every rule on small sizes. First causal with $T = 3$, and a decode chunk of 2 queries over 5 keys ($q_{\text{off}} = 3$: queries at positions 3 and 4):

$$\text{causal\_mask}(3) = \begin{pmatrix} 1&0&0\\1&1&0\\1&1&1 \end{pmatrix}, \qquad \text{causal\_mask}(2, 5, 3) = \begin{pmatrix} 1&1&1&1&0\\1&1&1&1&1 \end{pmatrix}.$$

The chunk is rows 3 and 4 of `causal_mask(5)`. A window of 2 over 4 positions is a band, where row $p$ opens $p - 1$ and $p$:

$$\text{sliding\_window\_mask}(4, 4, 2) = \begin{pmatrix} 1&0&0&0\\1&1&0&0\\0&1&1&0\\0&0&1&1 \end{pmatrix}.$$

Padding with lengths $(3, 1)$ and $T = 3$ gives rows $(1, 1, 1)$ and $(1, 0, 0)$. Combined with the causal mask (padding broadcast over queries), sequence 0 keeps the triangle and sequence 1 opens only key 0 in every row.

Now the additive form on one row: scores $(2, 1, 5)$ with the third key blocked. $A = (0, 0, -\infty)$, the sum is $(2, 1, -\infty)$, and the softmax is $(e^2, e^1, 0)/(e^2 + e^1) = (e/(e + 1), 1/(e + 1), 0) = (0.731059, 0.268941, 0)$. The blocked key had the largest score and still gets exactly 0. A correct solution reproduces every matrix and this softmax row exactly.

## Designing the API

Five functions cover every case:

```python
def causal_mask(Tq: int, Tk: int | None = None, q_offset: int = 0) -> NDArray: ...      # bool [Tq, Tk]
def padding_mask(lengths: ArrayLike, T: int) -> NDArray: ...                           # bool [B, T]
def sliding_window_mask(Tq: int, Tk: int, window: int, q_offset: int = 0) -> NDArray: ...
def combine(*masks: ArrayLike) -> NDArray: ...                                         # AND, broadcast
def to_additive(mask: ArrayLike, dtype=np.float32) -> NDArray: ...                     # 0 / -inf
```

`Tk` defaults to `q_offset + Tq`. Everything raises `ValueError` on sizes below 1, negative offsets, lengths outside $[0, T]$, non-bool masks, shapes that do not broadcast, or a non-float dtype.

## Building it step by step

The reasoning chain comes first; the code then follows it directly.

1. **Every rule is a predicate on two absolute positions.** Query $i$ is at $q_{\text{off}} + i$, key $j$ at $j$. Writing each rule as a comparison of a column of query positions against a row of key positions makes NumPy broadcasting produce the whole `[Tq, Tk]` matrix in one expression, with no loops.
2. **Causal is $\le$, not $<$.** A position must read its own token. The worked example's `causal_mask(3)` has ones on the diagonal.
3. **One offset covers prefill and decode.** Using absolute query positions means the chunk mask is literally the last $T_q$ rows of the full mask (`causal_mask(2, 5, 3)` equals rows 3 and 4 of `causal_mask(5)`). Defaulting `Tk` to `q_offset + Tq` means "every key up to the last query", so cached keys are never cut off.
4. **A window is the causal mask AND a distance bound.** Distance $p - j$ ranges over $0..w-1$ for exactly $w$ keys, so the bound is strict, $< w$. Building it on top of `causal_mask` guarantees no future key leaks.
5. **Rules compose by AND.** A key is readable only if every rule allows it. Broadcasting lets a `[T, T]` causal mask and a `[B, 1, 1, T]` padding mask combine into `[B, 1, T, T]`.
6. **The additive form must use $-\infty$.** It is the only value whose exponential is exactly 0 and that keeps an all-blocked row all-blocked after the max is subtracted.
7. **Fail loudly on ambiguous inputs.** A 0/1 integer array could mean either convention, so only bool masks are accepted; sizes, offsets, and lengths are validated.

### Causal and padding masks

```python
def causal_mask(Tq, Tk=None, q_offset=0):
    Tq = _count(Tq, "Tq", 1)                       # integer >= 1 (bools rejected)
    q_offset = _count(q_offset, "q_offset", 0)
    Tk = q_offset + Tq if Tk is None else _count(Tk, "Tk", 1)   # default: up to the last query
    # Query i sits at absolute position q_offset + i; key j at position j.
    pos_q = q_offset + np.arange(Tq)[:, None]      # [Tq, 1]
    pos_k = np.arange(Tk)[None, :]                 # [1, Tk]
    return pos_k <= pos_q                          # [Tq, Tk], the diagonal included
```

```python
def padding_mask(lengths, T):
    T = _count(T, "T", 1)
    n = np.asarray(lengths)
    # ... (1-D integers in [0, T]; 0 is an empty slot, allowed)
    return np.arange(T)[None, :] < n[:, None]      # [B, T], True on the l_b real tokens
```

### The sliding window

The window reuses the causal mask and adds the distance rule:

```python
def sliding_window_mask(Tq, Tk, window, q_offset=0):
    window = _count(window, "window", 1)
    Tk = _count(Tk, "Tk", 1)
    causal = causal_mask(Tq, Tk, q_offset)         # never the future
    pos_q = q_offset + np.arange(Tq)[:, None]
    pos_k = np.arange(Tk)[None, :]
    # At most `window` keys, the query's own position included.
    return causal & (pos_q - pos_k < window)
```

For `sliding_window_mask(4, 4, 2)`, row 2 opens keys with $0 \le 2 - j < 2$, that is $j \in \{1, 2\}$: the band from the worked example.

### Combining and the additive form

`combine` checks that the shapes broadcast, then ANDs into an all-True array of the broadcast shape:

```python
def combine(*masks):
    if not masks:
        raise ValueError("combine needs at least one mask")
    arrs = [_bool(m) for m in masks]               # bool only: no guessing the convention
    try:
        shape = np.broadcast_shapes(*(a.shape for a in arrs))
    except ValueError as e:
        raise ValueError(f"masks do not broadcast: {[a.shape for a in arrs]}") from e
    out = np.ones(shape, dtype=bool)               # the identity of AND
    for a in arrs:
        out &= a
    return out
```

And the additive form:

```python
def to_additive(mask, dtype=np.float32):
    m = _bool(mask)
    dt = np.dtype(dtype)
    if dt.kind != "f":
        raise ValueError(f"dtype must be a float type, got {dt}")
    # -inf, not a large negative number: exp(-inf) is exactly 0, and a row
    # with every key blocked stays all -inf, which the stable softmax maps
    # to zeros instead of a uniform average over forbidden keys.
    return np.where(m, dt.type(0.0), dt.type(-np.inf)).astype(dt)
```

Building the constants in the target dtype means float16 masks hold a true $-\infty$, and allowed entries are an exact $+0.0$. The full solution is [`masks.py`](../../../../course/solutions/python/tinyllm/xfmr/masks.py).

### Why this design

Each mask is $O(T_q T_k)$ booleans built by one broadcast comparison; nothing is computed per element in Python.

Why absolute positions and one `q_offset` instead of two functions (upper-left, lower-right)? One formula serves training, prefill, and every decode step, and the row-slice property (chunk = last rows of the full mask) falls out of it. That property is what makes cached and recomputed attention agree in the KV cache.

Why boolean masks as the interface, not additive ones? Booleans compose by AND without arithmetic surprises ($-\infty + -\infty$ is fine, but $0 \cdot -\infty$ is NaN), and kernels can take them as flags. The additive form is produced once, at the end.

Why materialize masks at all? At the sizes in this book a `[T, T]` bool array is cheap and easy to test. Production kernels (FlashAttention, FlexAttention) never build it and skip whole blocked tiles instead; the rules they implement are exactly these.

## What can go wrong

The checks a correct solution must pass mirror the rules. The worked matrices pin down the convention, and the additive softmax row $(0.731059, 0.268941, 0)$ confirms that blocked means weight exactly 0. Against torch, `causal_mask` must reproduce the upper-left and lower-right causal biases (prefill and decode alignment), and attention through causal, decode-chunk, and padded masks must equal torch's sdpa. The sliding window must match torch flex_attention's window of 3, the same rule as the Mistral window in [sliding window and sinks](../08-modern-block/07-sliding-window-and-sinks.md).

The properties are the ones argued above: future perturbations leave past outputs bitwise equal; chunk masks are the last rows of the full masks; each window row has $\min(w, p + 1)$ keys and a wide window is causal; and `combine` is AND, order-free, and produces `[B, 1, T, T]` from `[T, T]` and `[B, 1, 1, T]`. At the boundaries, padding must handle lengths 0 and $T$ (empty slots in a batch), and an all-blocked row must be all $-\infty$ and attend to nothing. `to_additive` must produce exact 0.0 and $-\infty$ in float16, float32, and float64. `causal_mask(2, q_offset=3)` must be `[2, 5]`, so cached keys are not dropped. Bad sizes, offsets, lengths, float masks, and int dtypes must raise, so a convention mix-up fails loudly.

The model suite [`test_masks.py`](../../../../course/solutions/learner-tests/L5.2/python/tests/l5-2-masks/test_masks.py) uses the definition written as two nested loops over $(i, j)$ as its oracle. `causal_mask` and `sliding_window_mask` are compared with it on several shapes and offsets (including a decode step); padding is checked on lengths 0 and $T$, `combine` against `&`, the exact $0$ and $-\infty$ of `to_additive` (including an all-blocked row), and causality through a few lines of numpy attention. It imports only `tinyllm.xfmr.masks`.

The common mistakes are almost all off-by-ones or sign flips:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| a strict $j < i$ causal mask | position 0 sees nothing; each token cannot see itself | `pos_k <= pos_q`, the diagonal included |
| ignoring `q_offset` (top-left alignment for a decode chunk), or a default `Tk = Tq` | cached decoding differs from recomputation | `pos_q = q_offset + arange(Tq)`; `Tk` defaults to `q_offset + Tq` |
| a window of $w + 1$ keys ($\le$ for $<$) | disagrees with Mistral checkpoints | `pos_q - pos_k < window`, strict |
| a window without the causal half | the model reads $w - 1$ future tokens | `sliding_window_mask` starts from `causal_mask(...)` and ANDs the window |
| padding open at $t = \ell_b$ | every sequence reads one pad token | `arange(T) < n`, strict |
| a finite "minus infinity" | an all-blocked row averages forbidden keys | `to_additive` writes `dt.type(-np.inf)` |
| the inverted convention ($-\infty$ where True) | attention reads only what it should not | `np.where(m, 0.0, -inf)`: 0 where True |
| combining by OR | padding re-opened by the causal rule | `out &= a` for every mask |

## Where this leads

`scaled_dot_product_attention(q, k, v, mask)` takes these masks directly (True = attend). [Multi-head attention](03-multi-head-attention.md) broadcasts one mask over every head, and the [encoder-decoder Transformer](05-encoder-decoder-transformer.md) builds its decoder mask as `combine(causal_mask(T), padding)`. Later, [GPT](../07-objectives/01-gpt-decoder-only.md) trains under `causal_mask(T)`, the modern block uses [sliding-window attention](../08-modern-block/07-sliding-window-and-sinks.md), the [KV cache](../09-inference/02-kv-cache-and-generate.md) step is `causal_mask(Tq, Tk, q_offset=Tk - Tq)`, and the C kernels for [FlashAttention](../10-kernels/05-flash-attention-forward.md) and [paged attention](../10-kernels/06-paged-attention-decode.md) take `causal`, `q_offset`, and `window` flags with exactly these meanings. [Whisper decoding](../15-speech/03-whisper-decoder-and-weights.md) calls this code through its declared contract.

In production, the same rules appear without materialized masks:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `causal_mask` with `q_offset` | `torch.nn.attention.bias.causal_lower_right` | a lazy bias the fused kernels recognize, never materialized | `torch/nn/attention/bias.py` |
| `sliding_window_mask` | FlexAttention `mask_mod` + `BlockMask` | arbitrary rules compiled into block-sparse kernels that skip fully blocked tiles | `torch/nn/attention/flex_attention.py` |
| `to_additive` | FlashAttention's `causal` and `window_size` arguments | the mask is never a tensor; tiles above the diagonal are skipped | `flash_attn/flash_attn_interface.py` |
| `combine` with padding | HF `transformers` `masking_utils` | one function builds causal, sliding, chunked, and padding masks per layer type | `src/transformers/masking_utils.py` |

### Further reading

- Vaswani et al., "Attention Is All You Need" (2017), section 3.2.3 (masking in the decoder).
- Beltagy, Peters, and Cohan, "Longformer" (2020), and Jiang et al., "Mistral 7B" (2023), section 2 (sliding-window attention).

## Key terms

- **mask**: a boolean matrix over (query, key) pairs where True means the query may attend to the key.
- **causal mask**: the rule $j \le q_{\text{off}} + i$; each position reads itself and the past.
- **q_offset**: the absolute position of the first query, which aligns a decode chunk to the bottom right of the full mask.
- **key-padding mask**: $[t < \ell_b]$ per sequence, blocking padded keys.
- **sliding window**: the causal mask AND a distance bound $< w$, at most $w$ keys per query.
- **additive mask**: the float form, 0 where allowed and $-\infty$ where blocked, added to scores before the softmax.

## Summary

- A mask is a boolean matrix over (query, key) with **True = may attend**; rules compose by AND.
- **Causal**: query $i$ at absolute position $q_{\text{off}} + i$ sees keys $j \le q_{\text{off}} + i$. One offset turns the training triangle into a decode chunk aligned to the bottom right.
- Causality is checkable to the bit: changing future tokens must leave past outputs **bitwise** unchanged.
- A **sliding window** is causal AND local: at most `window` keys, the query's own included.
- The additive form is $0$ and $-\infty$, never a big negative number: a fully blocked row must give zero weights, not a uniform average over forbidden keys.

## Review questions

1. Why is `causal_mask(2, 5, 3)` equal to the last two rows of `causal_mask(5)`?
   <details><summary>Answer</summary>Both compute $[j \le p]$ for the same absolute query positions $p = 3, 4$ and the same keys $0..4$. Using `q_offset + arange(Tq)` as the query positions is what makes a chunk a row slice of the full mask.</details>
2. Why does the window bound use `pos_q - pos_k < window` and not `<=`?
   <details><summary>Answer</summary>Together with causality the distance $p - j$ is in $\{0, \dots, w-1\}$, exactly $w$ keys including the query's own. `<=` would open $w + 1$ keys and disagree with models trained with a window of $w$.</details>
3. What goes wrong if `combine` uses OR?
   <details><summary>Answer</summary>Any rule could re-open what another blocked: the causal rule would open padded keys before the query, and padding would open future keys. A key must satisfy every rule, which is AND.</details>
4. Why does `to_additive` refuse an integer dtype?
   <details><summary>Answer</summary>Integers cannot represent $-\infty$, so the blocked entries would become some finite value and an all-blocked row would turn into a uniform average over forbidden keys.</details>
5. Why do padded *queries* not need a mask?
   <details><summary>Answer</summary>Their outputs are never used: the loss ignores padded target positions. Blocking them would only risk all-blocked rows; what matters is that real queries never read padded keys.</details>
