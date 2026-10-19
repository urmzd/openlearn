# Masks: causal, padding, sliding window, additive

## Overview

| | |
|---|---|
| **Module** | `L5.2` · build · Python · Pass 5 · 1 h reading |
| **Solution** | [`python/tinyllm/xfmr/masks.py`](../../../../course/solutions/python/tinyllm/xfmr/masks.py): `causal_mask`, `padding_mask`, `sliding_window_mask`, `combine`, `to_additive`; model oracle tests in [`test_masks.py`](../../../../course/solutions/learner-tests/L5.2/python/tests/l5-2-masks/test_masks.py) |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/masks.pyi`](../../../../course/contracts/py/tinyllm/xfmr/masks.pyi) |
| **Needs** | nothing to call. Reading: `S-M05` (logic: AND, implication), [`M09.2` stable softmax](../../../tracks/math/09-numerical-methods-and-floating-point/02-stable-numerics.md) (an all $-\infty$ row gives zeros) |
| **Used by** | `L5.3` and `L5.5` build every attention mask here · later `L6.1` (GPT's causal mask), `L7.7` (sliding windows), `L8.2` (`q_offset` while decoding with a KV cache), `L9.3` and `L9.4` (the same rules as kernel flags) · `L14.3` Whisper decoding · part of the 2017 transformer stage |
| **Optional depth** | Vaswani et al., "Attention Is All You Need" (2017), section 3.2.3 (masking in the decoder); Beltagy, Peters, Cohan, "Longformer" (2020) and Jiang et al., "Mistral 7B" (2023), section 2 (sliding-window attention) |

## Key Takeaways

- A mask is a boolean matrix over (query, key) with **True = may attend**; rules compose by AND.
- **Causal**: query $i$ at absolute position $q_{\text{off}} + i$ sees keys $j \le q_{\text{off}} + i$. One offset turns the training triangle into a decode chunk aligned to the bottom right.
- Causality is checkable to the bit: changing future tokens must leave past outputs **bitwise** unchanged.
- A **sliding window** is causal AND local: at most `window` keys, the query's own included.
- The additive form is $0$ and $-\infty$, never a big negative number: a fully blocked row must give zero weights, not a uniform average over forbidden keys.

## How to read this chapter

Sections 1 to 3 build every mask from one convention and work them by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Scaled dot-product attention (`L5.1`) lets every position read every other position, which is exactly wrong for the language models built next. A language model is trained to predict token $t + 1$ from tokens up to $t$; if position $t$ can read position $t + 1$, training loss falls to nearly zero by copying, and generation, which has no future to copy, produces noise. In a batch, short sequences are padded, and a query that reads padding learns from garbage. Long contexts (Part 7) read only a recent window. All three are the same mechanism: a boolean matrix of allowed (query, key) pairs, turned into $0$ or $-\infty$ added to the scores. Getting the off-by-ones right here is the difference between a model that trains and one that cheats, and the decode path of Part 8 reuses the same function with an offset.

## 2. Principles

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

### 2.1 One convention

Every mask in the system means the same thing: entry $(i, j)$ True means "query $i$ may read key $j$". This is the convention of `torch.nn.functional.scaled_dot_product_attention`. Beware: `torch.nn.MultiheadAttention` uses the opposite (True = blocked). Booleans are the interface, so the functions reject float or integer masks rather than guessing which convention a 0/1 array meant.

### 2.2 Causal masks and q_offset

Position $p$ may read positions $\le p$, itself included (it needs its own token to predict the next one). With queries at absolute positions $q_{\text{off}} + i$ and keys at $j$:

$$M_{ij} = [\,j \le q_{\text{off}} + i\,].$$

During training $T_q = T_k = T$ and $q_{\text{off}} = 0$: the lower triangle with the diagonal. During generation with a KV cache (`L8.2`), the keys of all $T_k$ tokens so far are cached, and a step computes only the last $T_q$ queries, at positions $T_k - T_q, \ldots, T_k - 1$: $q_{\text{off}} = T_k - T_q$, and the triangle is aligned to the **bottom right**, so the last query sees every key. Torch ships both alignments (`causal_upper_left`, `causal_lower_right`); one parameter covers both here. The key test: the chunk's mask is exactly the last $T_q$ rows of the full mask, so cached and recomputed attention agree.

### 2.3 Padding and combining

A batch stacks sequences of lengths $\ell_b$ padded to $T$. The key-padding mask is $P_{bt} = [t < \ell_b]$, shape `[B, T]`. To use it on scores of shape `[B, H, Tq, Tk]`, index it as `P[:, None, None, :]`: one row per sequence, the same for every head and every query. Rules compose by **AND**, with numpy broadcasting: `combine(causal_mask(T), P[:, None, None, :])` has shape `[B, 1, T, T]` and opens $(i, j)$ only if $j \le i$ and $j < \ell_b$. Padded *queries* still produce outputs; the loss ignores them, so they need no mask.

### 2.4 Sliding windows

Attention cost grows with $T_q T_k$. A sliding window keeps only the last $w$ keys:

$$M_{ij} = [\,j \le q_{\text{off}} + i\,] \wedge [\,(q_{\text{off}} + i) - j < w\,].$$

Each query sees $\min(w, p + 1)$ keys. A stack of $L$ windowed layers still has a receptive field of about $L w$ tokens, which is how Mistral reads long contexts with a 4096-token window. A window of at least $T$ is the causal mask; a window without the causal half lets queries read $w - 1$ future tokens.

### 2.5 Additive masks

The softmax takes scores, not booleans. Add $A_{ij} = 0$ where allowed and $-\infty$ where blocked: $e^{-\infty} = 0$ exactly, so blocked keys get weight exactly 0, and allowed scores are untouched (adding $0.0$ changes nothing, not even the sign of a zero). If every key of a row is blocked, the row is all $-\infty$; the stable softmax of `M09.2` defines that as zeros. A "large negative" such as $-10^9$ looks equivalent but is not: in an all-blocked row every entry is $-10^9$, the softmax subtracts the maximum, and the row becomes uniform: the query averages over keys it was forbidden to read. In float16, $-10^9$ is not even representable (it rounds to $-\infty$ anyway). Because $-\infty$ contributes exact zeros, causality holds to the bit: perturbing a future token changes its score, but the score is masked to $-\infty$ and its value vector is multiplied by an exact 0.

## 3. Worked example by hand

Causal, $T = 3$, and a decode chunk of 2 queries over 5 keys ($q_{\text{off}} = 3$: queries at positions 3 and 4):

$$\text{causal\_mask}(3) = \begin{pmatrix} 1&0&0\\1&1&0\\1&1&1 \end{pmatrix}, \qquad \text{causal\_mask}(2, 5, 3) = \begin{pmatrix} 1&1&1&1&0\\1&1&1&1&1 \end{pmatrix}.$$

The chunk is rows 3 and 4 of `causal_mask(5)`. A window of 2 over 4 positions is a band: row $p$ opens $p - 1$ and $p$:

$$\text{sliding\_window\_mask}(4, 4, 2) = \begin{pmatrix} 1&0&0&0\\1&1&0&0\\0&1&1&0\\0&0&1&1 \end{pmatrix}.$$

Padding with lengths $(3, 1)$ and $T = 3$ gives rows $(1, 1, 1)$ and $(1, 0, 0)$. Combined with the causal mask (padding broadcast over queries), sequence 0 keeps the triangle and sequence 1 opens only key 0 in every row.

Now the additive form on one row: scores $(2, 1, 5)$ with the third key blocked. $A = (0, 0, -\infty)$, the sum is $(2, 1, -\infty)$, and the softmax is $(e^2, e^1, 0)/(e^2 + e^1) = (e/(e + 1), 1/(e + 1), 0) = (0.731059, 0.268941, 0)$. The blocked key had the largest score and still gets exactly 0. A correct solution reproduces every matrix and this softmax row exactly.

## 4. The interface

```python
def causal_mask(Tq: int, Tk: int | None = None, q_offset: int = 0) -> NDArray: ...      # bool [Tq, Tk]
def padding_mask(lengths: ArrayLike, T: int) -> NDArray: ...                           # bool [B, T]
def sliding_window_mask(Tq: int, Tk: int, window: int, q_offset: int = 0) -> NDArray: ...
def combine(*masks: ArrayLike) -> NDArray: ...                                         # AND, broadcast
def to_additive(mask: ArrayLike, dtype=np.float32) -> NDArray: ...                     # 0 / -inf
```

`Tk` defaults to `q_offset + Tq`. Everything raises `ValueError` on sizes below 1, negative offsets, lengths outside $[0, T]$, non-bool masks, shapes that do not broadcast, or a non-float dtype.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: masks | unit, smoke | the section 3 matrices | pins down the convention |
| hand example: additive softmax | unit, smoke | $(0.731059, 0.268941, 0)$ | blocked means weight exactly 0 |
| causal matches torch biases | golden | torch's upper-left and lower-right causal biases | prefill and decode alignment |
| attention with masks matches torch sdpa | golden | attention through causal, decode-chunk, and padded masks equals torch | the masks do what attention needs |
| sliding window matches flex attention | golden | torch flex_attention's window of 3 | `L7.7`'s Mistral window |
| causality is bitwise | property | future perturbations leave past outputs bitwise equal | no leak of the answer in training |
| chunked decode masks are rows of the full mask | property | chunk masks are the last rows of the full masks | `L8.2`'s cache vs recompute test |
| window counts | property | $\min(w, p + 1)$ keys per row; a wide window is causal | the window rule exactly |
| combine is and with broadcasting | property | AND, order-free, `[B, 1, T, T]` from `[T, T]` and `[B, 1, 1, T]` | one mask per batch for every head |
| padding mask edges | boundary | lengths 0 and $T$ | empty slots in a batch |
| fully masked row is all minus inf | boundary | an all-blocked row is all $-\infty$ and attends to nothing | padded queries never read padding |
| additive dtypes and values | unit | float16/32/64 with exact 0.0 and $-\infty$ | masks join scores in their dtype |
| default `Tk` follows `q_offset` | unit | `causal_mask(2, q_offset=3)` is `[2, 5]` | cached keys are not dropped |
| rejects bad arguments | boundary | bad sizes, offsets, lengths, float masks, int dtypes | the convention mix-up fails loudly |

### The oracle tests

In the model suite [`test_masks.py`](../../../../course/solutions/learner-tests/L5.2/python/tests/l5-2-masks/test_masks.py) the oracle is the definition written as two nested loops over $(i, j)$: `causal_mask` and `sliding_window_mask` are compared with it on several shapes and offsets (including a decode step); padding is checked on lengths 0 and $T$, `combine` against `&`, the exact $0$ and $-\infty$ of `to_additive` (including an all-blocked row), and causality through a few lines of numpy attention. It imports only `tinyllm.xfmr.masks`.

## 5. The solution, step by step

### From first principles

1. **Every rule is a predicate on two absolute positions.** Query $i$ is at $q_{\text{off}} + i$, key $j$ at $j$. Writing each rule as a comparison of a column of query positions against a row of key positions makes NumPy broadcasting produce the whole `[Tq, Tk]` matrix in one expression, with no loops.
2. **Causal is $\le$, not $<$.** A position must read its own token. The worked example's `causal_mask(3)` has ones on the diagonal.
3. **One offset covers prefill and decode.** Using absolute query positions means the chunk mask is literally the last $T_q$ rows of the full mask (`causal_mask(2, 5, 3)` equals rows 3 and 4 of `causal_mask(5)`). Defaulting `Tk` to `q_offset + Tq` means "every key up to the last query", so cached keys are never cut off.
4. **A window is the causal mask AND a distance bound.** Distance $p - j$ ranges over $0..w-1$ for exactly $w$ keys, so the bound is strict, $< w$. Building it on top of `causal_mask` guarantees no future key leaks.
5. **Rules compose by AND.** A key is readable only if every rule allows it. Broadcasting lets a `[T, T]` causal mask and a `[B, 1, 1, T]` padding mask combine into `[B, 1, T, T]`.
6. **The additive form must use $-\infty$.** It is the only value whose exponential is exactly 0 and that keeps an all-blocked row all-blocked after the max is subtracted (section 2.5).
7. **Fail loudly on ambiguous inputs.** A 0/1 integer array could mean either convention, so only bool masks are accepted; sizes, offsets and lengths are validated.

### The code, annotated

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

For `sliding_window_mask(4, 4, 2)`, row 2 opens keys with $0 \le 2 - j < 2$, that is $j \in \{1, 2\}$: the band of section 3.

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

Building the constants in the target dtype means float16 masks hold a true $-\infty$, and allowed entries are an exact $+0.0$. Full solution: [`masks.py`](../../../../course/solutions/python/tinyllm/xfmr/masks.py).

### Why this solution

- **Cost.** Each mask is $O(T_q T_k)$ booleans built by one broadcast comparison; nothing is computed per element in Python.
- **Why absolute positions and one `q_offset` instead of two functions (upper-left, lower-right)?** One formula serves training, prefill and every decode step, and the row-slice property (chunk = last rows of the full mask) falls out of it, which is what makes cached and recomputed attention agree in `L8.2`.
- **Why boolean masks as the interface, not additive ones?** Booleans compose by AND without arithmetic surprises ($-\infty + -\infty$ is fine, but $0 \cdot -\infty$ is NaN), and kernels can take them as flags. The additive form is produced once, at the end.
- **Why materialize masks at all?** At course sizes a `[T, T]` bool array is cheap and easy to test. Production kernels (FlashAttention, FlexAttention) never build it and skip whole blocked tiles instead; the rules they implement are exactly these.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a strict $j < i$ causal mask | position 0 sees nothing; each token cannot see itself | `pos_k <= pos_q`, the diagonal included |
| 2. ignoring `q_offset` (top-left alignment for a decode chunk), or a default `Tk = Tq` | cached decoding differs from recomputation | `pos_q = q_offset + arange(Tq)`; `Tk` defaults to `q_offset + Tq` |
| 3. a window of $w + 1$ keys ($\le$ for $<$) | disagrees with Mistral checkpoints | `pos_q - pos_k < window`, strict |
| 4. a window without the causal half | the model reads $w - 1$ future tokens | `sliding_window_mask` starts from `causal_mask(...)` and ANDs the window |
| 5. padding open at $t = \ell_b$ | every sequence reads one pad token | `arange(T) < n`, strict |
| 6. a finite "minus infinity" | an all-blocked row averages forbidden keys | `to_additive` writes `dt.type(-np.inf)` |
| 7. the inverted convention ($-\infty$ where True) | attention reads only what it should not | `np.where(m, 0.0, -inf)`: 0 where True |
| 8. combining by OR | padding re-opened by the causal rule | `out &= a` for every mask |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | AND of conditions, and implication as a matrix (reading) |
| Back | `M09.2` | the softmax that maps an all $-\infty$ row to zeros (reading) |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract |
| Forward | `L5.1` | `scaled_dot_product_attention(q, k, v, mask)` takes these masks (True = attend) |
| Forward | `L5.3` | multi-head attention broadcasts one mask over every head |
| Forward | `L5.5` | the decoder's mask is `combine(causal_mask(T), padding)` |
| Forward | `L6.1` | GPT trains under `causal_mask(T)` |
| Forward | `L7.7` | sliding-window attention in the modern block |
| Forward | `L8.2` | the KV-cache step is `causal_mask(Tq, Tk, q_offset=Tk - Tq)` |
| Forward | `L9.3` | the C attention kernel takes `causal`, `q_offset`, and `window` flags with these meanings |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `causal_mask` with `q_offset` | `torch.nn.attention.bias.causal_lower_right` | a lazy bias the fused kernels recognize, never materialized | `torch/nn/attention/bias.py` |
| `sliding_window_mask` | FlexAttention `mask_mod` + `BlockMask` | arbitrary rules compiled into block-sparse kernels that skip fully blocked tiles | `torch/nn/attention/flex_attention.py` |
| `to_additive` | FlashAttention's `causal` and `window_size` arguments | the mask is never a tensor; tiles above the diagonal are skipped | `flash_attn/flash_attn_interface.py` |
| `combine` with padding | HF `transformers` `masking_utils` | one function builds causal, sliding, chunked, and padding masks per layer type | `src/transformers/masking_utils.py` |
