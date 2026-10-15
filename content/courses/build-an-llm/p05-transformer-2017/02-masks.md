<!-- ol:module L5.2 -->
# Masks: causal, padding, sliding window, additive

## Overview

| | |
|---|---|
| **Module** | `L5.2` · build · Python · Pass 5 · 1 to 2 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/xfmr/masks.py`: `causal_mask`, `padding_mask`, `sliding_window_mask`, `combine`, `to_additive`; and your own oracle tests in `python/tests/l5-2-masks/` |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/masks.pyi`](../../../../course/contracts/py/tinyllm/xfmr/masks.pyi) |
| **Tests** | `course/tests/L5.2/test_masks.py` (what they check: section 4), golden values from torch 2.14.1 in `course/fixtures/L5.2/masks_torch.npz` (`course/oracle/L5.2/masks_torch.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | nothing to call. Reading: `S-M05` (logic: AND, implication), [`M09.2` stable softmax](../../../tracks/math/09-numerical-methods-and-floating-point/02-stable-numerics.md) (an all $-\infty$ row gives zeros) |
| **Used by** | `L5.3` and `L5.5` build every attention mask here · later `L6.1` (GPT's causal mask), `L7.7` (sliding windows), `L8.2` (`q_offset` while decoding with a KV cache), `L9.3` and `L9.4` (the same rules as kernel flags)  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L5` |
| **Optional depth** | Vaswani et al., "Attention Is All You Need" (2017), section 3.2.3 (masking in the decoder); Beltagy, Peters, Cohan, "Longformer" (2020) and Jiang et al., "Mistral 7B" (2023), section 2 (sliding-window attention) |

## Key Takeaways

- A mask is a boolean matrix over (query, key) with **True = may attend**; rules compose by AND (`test_hand_example_masks`, `test_combine_is_and_with_broadcasting`).
- **Causal**: query $i$ at absolute position $q_{\text{off}} + i$ sees keys $j \le q_{\text{off}} + i$. One offset turns the training triangle into a decode chunk aligned to the bottom right (`test_causal_matches_torch_biases`, `test_chunked_decode_masks_are_rows_of_the_full_mask`).
- Causality is checkable to the bit: changing future tokens must leave past outputs **bitwise** unchanged (`test_causality_is_bitwise`).
- A **sliding window** is causal AND local: at most `window` keys, the query's own included (`test_sliding_window_matches_flex_attention`, `test_window_counts`).
- The additive form is $0$ and $-\infty$, never a big negative number: a fully blocked row must give zero weights, not a uniform average over forbidden keys (`test_fully_masked_row_is_all_minus_inf`).

## How to work this chapter

```bash
ol start L5.2              # stubs masks.py; prints your test path and rung (R5)
ol tests L5.2              # the course tests
# write your oracle tests in python/tests/l5-2-masks/, then:
ol check L5.2              # course tests and the mutation grade of your tests
ol diff  L5.2              # after passing: your code against the reference
```

---

## 1. Why now

Scaled dot-product attention (`L5.1`) lets every position read every other position, which is exactly wrong for the model you are about to build. A language model is trained to predict token $t + 1$ from tokens up to $t$; if position $t$ can read position $t + 1$, training loss falls to nearly zero by copying, and generation, which has no future to copy, produces noise. In a batch, short sequences are padded, and a query that reads padding learns from garbage. Long contexts (Part 7) read only a recent window. All three are the same mechanism: a boolean matrix of allowed (query, key) pairs, turned into $0$ or $-\infty$ added to the scores. Getting the off-by-ones right here is the difference between a model that trains and one that cheats, and the decode path of Part 8 reuses the same function with an offset.

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

Now the additive form on one row: scores $(2, 1, 5)$ with the third key blocked. $A = (0, 0, -\infty)$, the sum is $(2, 1, -\infty)$, and the softmax is $(e^2, e^1, 0)/(e^2 + e^1) = (e/(e + 1), 1/(e + 1), 0) = (0.731059, 0.268941, 0)$. The blocked key had the largest score and still gets exactly 0. These are the first two tests.

## 4. The interface

```python
def causal_mask(Tq: int, Tk: int | None = None, q_offset: int = 0) -> NDArray: ...      # bool [Tq, Tk]
def padding_mask(lengths: ArrayLike, T: int) -> NDArray: ...                           # bool [B, T]
def sliding_window_mask(Tq: int, Tk: int, window: int, q_offset: int = 0) -> NDArray: ...
def combine(*masks: ArrayLike) -> NDArray: ...                                         # AND, broadcast
def to_additive(mask: ArrayLike, dtype=np.float32) -> NDArray: ...                     # 0 / -inf
```

`Tk` defaults to `q_offset + Tq`. Everything raises `ValueError` on sizes below 1, negative offsets, lengths outside $[0, T]$, non-bool masks, shapes that do not broadcast, or a non-float dtype.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_masks` | unit, smoke | the section 3 matrices | you and the tests agree on the convention |
| `test_hand_example_additive_softmax` | unit, smoke | $(0.731059, 0.268941, 0)$ | blocked means weight exactly 0 |
| `test_causal_matches_torch_biases` | golden | torch's upper-left and lower-right causal biases | prefill and decode alignment |
| `test_attention_with_masks_matches_torch_sdpa` | golden | attention through causal, decode-chunk, and padded masks equals torch | the masks do what attention needs |
| `test_sliding_window_matches_flex_attention` | golden | torch flex_attention's window of 3 | `L7.7`'s Mistral window |
| `test_causality_is_bitwise` | property | future perturbations leave past outputs bitwise equal | no leak of the answer in training |
| `test_chunked_decode_masks_are_rows_of_the_full_mask` | property | chunk masks are the last rows of the full masks | `L8.2`'s cache vs recompute test |
| `test_window_counts` | property | $\min(w, p + 1)$ keys per row; a wide window is causal | the window rule exactly |
| `test_combine_is_and_with_broadcasting` | property | AND, order-free, `[B, 1, T, T]` from `[T, T]` and `[B, 1, 1, T]` | one mask per batch for every head |
| `test_padding_mask_edges` | boundary | lengths 0 and $T$ | empty slots in a batch |
| `test_fully_masked_row_is_all_minus_inf` | boundary | an all-blocked row is all $-\infty$ and attends to nothing | padded queries never read padding |
| `test_additive_dtypes_and_values` | unit | float16/32/64 with exact 0.0 and $-\infty$ | masks join scores in their dtype |
| `test_default_tk_follows_q_offset` | unit | `causal_mask(2, q_offset=3)` is `[2, 5]` | cached keys are not dropped |
| `test_rejects_bad_arguments` | boundary | bad sizes, offsets, lengths, float masks, int dtypes | the convention mix-up fails loudly |

### Your graded tests (rung R5)

The oracle is the definition written as two nested loops over $(i, j)$: compare `causal_mask` and `sliding_window_mask` with it on several shapes and offsets (including a decode step), check padding on lengths 0 and $T$, check `combine` against `&`, check the exact $0$ and $-\infty$ of `to_additive` (including an all-blocked row), and check causality through a few lines of numpy attention. Import only `tinyllm.xfmr.masks`.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a strict $j < i$ causal mask | position 0 sees nothing; each token cannot see itself | `test_hand_example_masks` (mutant `s01`) |
| 2. ignoring `q_offset` (top-left alignment for a decode chunk), or a default `Tk = Tq` | cached decoding differs from recomputation | `test_chunked_decode_masks_are_rows_of_the_full_mask` (mutant `s02`), `test_default_tk_follows_q_offset` (mutant `s09`) |
| 3. a window of $w + 1$ keys ($\le$ for $<$) | disagrees with Mistral checkpoints | `test_window_counts` (mutant `s03`) |
| 4. a window without the causal half | the model reads $w - 1$ future tokens | `test_causality_is_bitwise` (mutant `s04`) |
| 5. padding open at $t = \ell_b$ | every sequence reads one pad token | `test_padding_mask_edges` (mutant `s05`) |
| 6. a finite "minus infinity" | an all-blocked row averages forbidden keys | `test_fully_masked_row_is_all_minus_inf` (mutant `s06`) |
| 7. the inverted convention ($-\infty$ where True) | attention reads only what it should not | `test_hand_example_additive_softmax` (mutant `s07`) |
| 8. combining by OR | padding re-opened by the causal rule | `test_combine_is_and_with_broadcasting` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | AND of conditions, and implication as a matrix (reading) |
| Back | `M09.2` | the softmax that maps an all $-\infty$ row to zeros (reading) |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L5.1` | `scaled_dot_product_attention(q, k, v, mask)` takes these masks (True = attend) |
| Forward | `L5.3` | multi-head attention broadcasts one mask over every head |
| Forward | `L5.5` | the decoder's mask is `combine(causal_mask(T), padding)` |
| Forward | `L6.1` | GPT trains under `causal_mask(T)` |
| Forward | `L7.7` | sliding-window attention in the modern block |
| Forward | `L8.2` | the KV-cache step is `causal_mask(Tq, Tk, q_offset=Tk - Tq)` |
| Forward | `L9.3` | the C attention kernel takes `causal`, `q_offset`, and `window` flags with these meanings |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `causal_mask` with `q_offset` | `torch.nn.attention.bias.causal_lower_right` | a lazy bias the fused kernels recognize, never materialized | `torch/nn/attention/bias.py` |
| `sliding_window_mask` | FlexAttention `mask_mod` + `BlockMask` | arbitrary rules compiled into block-sparse kernels that skip fully blocked tiles | `torch/nn/attention/flex_attention.py` |
| `to_additive` | FlashAttention's `causal` and `window_size` arguments | the mask is never a tensor; tiles above the diagonal are skipped | `flash_attn/flash_attn_interface.py` |
| `combine` with padding | HF `transformers` `masking_utils` | one function builds causal, sliding, chunked, and padding masks per layer type | `src/transformers/masking_utils.py` |
