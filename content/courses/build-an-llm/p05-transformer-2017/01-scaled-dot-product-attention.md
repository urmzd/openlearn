<!-- ol:module L5.1 -->
# Scaled dot-product attention (forward and backward)

## Overview

| | |
|---|---|
| **Module** | `L5.1` · build · Python · Pass 5 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/xfmr/sdpa.py`: `sdpa_forward`, `sdpa_backward` (by hand), and `scaled_dot_product_attention`, one autograd node; and your own oracle tests in `python/tests/l5-1-sdpa/` |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/sdpa.pyi`](../../../../course/contracts/py/tinyllm/xfmr/sdpa.pyi) |
| **Tests** | `course/tests/L5.1/test_sdpa.py` (what they check: section 4), golden values from torch 2.14.1 in `course/fixtures/L5.1/sdpa_torch.npz` (`course/oracle/L5.1/sdpa_torch.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | [`L0.1` `Tensor` and `from_op`](../p00-foundations/01-tensor-and-broadcasting-backward.md) · [`M09.2` stable softmax](../../../tracks/math/09-numerical-methods-and-floating-point/02-stable-numerics.md) · reading: [`L4.3` Luong's dot score](../p04-attention-origins/03-luong-attention-input-feeding.md), [`M08.3`](../../../tracks/math/08-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) and `S-M08` (the softmax VJP), `L0.2` (the dropout draw rule) (or `--ref-deps`) |
| **Used by** | `L5.3` multi-head attention calls it once per layer · later `L7.5` and `L7.6` (grouped-query and latent attention), `L8.2` (attention over a KV cache), `L9.3` (the C kernel is proven against `sdpa_forward`) · later: `L13.3`  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L5` |
| **Optional depth** | Vaswani et al., "Attention Is All You Need" (2017), section 3.2.1; Dao et al., "FlashAttention" (2022), section 3.1 and appendix B.2 (the backward with $D = \operatorname{rowsum}(dO \circ O)$) |

## Key Takeaways

- Attention is a soft dictionary lookup: $O = \operatorname{softmax}(QK^\top/\sqrt{d} + \text{mask})\,V$, a weighted average of values with weights from query-key similarity (`test_hand_example_forward`).
- The $1/\sqrt{d}$ keeps the scores' variance at 1 whatever the width, so the softmax neither saturates nor flattens (`test_score_variance_is_one`, `test_default_scale_is_inverse_sqrt_d`).
- The backward is four matrix products and the softmax VJP $dS = P \circ (dP - \operatorname{rowsum}(dP \circ P))$, and that row sum equals $dO \cdot O$ (`test_hand_example_backward`, `test_gradcheck_float64`).
- Attention is equivariant to permuting keys with their values: it has no notion of order (`test_key_permutation_equivariance`).
- Blocked keys get weight and gradient exactly 0, and a fully blocked row is zeros, never NaN (`test_masked_keys_get_no_weight_or_gradient`, `test_fully_masked_row_gives_zeros`).

## How to work this chapter

```bash
ol start L5.1              # stubs sdpa.py; prints your test path and rung (R5)
ol tests L5.1              # the course tests
# write your oracle tests in python/tests/l5-1-sdpa/, then:
ol check L5.1              # course tests and the mutation grade of your tests
ol diff  L5.1              # after passing: your code against the reference
```

---

## 1. Why now

The sequence models of Part 4 read the source through an RNN: one step per token, and everything the decoder knows about token 3 has passed through every later state. Luong attention (`L4.3`) already reads the encoder states directly with a dot product. The transformer keeps only that read: every position attends to every position in one matrix product, so a sequence of length $T$ costs $T^2$ dot products but no sequential steps, and the gradient from position 50 to position 3 is one hop instead of 47. Every model you build from here on (the 2017 Transformer, GPT, BERT, the modern Llama block, the inference engine's kernels) is built on this one function. Writing its backward by hand, rather than composing ops, makes it one node of your autograd graph instead of seven, and it is the exact computation the C kernel of `L9.3` and FlashAttention reorganize.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $Q$ | queries, one row per position asking | `[..., Tq, d]` |
| $K$ | keys, one row per position offering | `[..., Tk, d]` |
| $V$ | values, what each key position contributes | `[..., Tk, dv]` |
| $d$ | query and key width | `int` |
| $\alpha$ (`scale`) | score scale, default $1/\sqrt{d}$ | `float` |
| $S = \alpha QK^\top$ | scores | `[..., Tq, Tk]` |
| $M$ | mask, True = may attend (`L5.2`) | `bool`, broadcast to $S$ |
| $P = \operatorname{softmax}(S)$ | weights, each row sums to 1 (over keys) | `[..., Tq, Tk]` |
| $O = PV$ | output | `[..., Tq, dv]` |
| $dO, dP, dS, dQ, dK, dV$ | gradients of a scalar loss with respect to each | same shapes as their arrays |
| $D_i = \sum_j P_{ij}\, dP_{ij}$ | the row sum in the softmax VJP | `[..., Tq, 1]` |

### 2.1 Attention as a soft lookup

A Python dict returns the value whose key equals the query. Attention returns a blend: score query $q_i$ against every key $k_j$ by the dot product (large when they point the same way), turn the scores into positive weights that sum to 1 with a softmax over the keys, and average the values with those weights: $o_i = \sum_j P_{ij} v_j$. Because the output is a weighted sum over a *set* of key-value pairs, reordering the pairs changes nothing: attention has no idea of position, which is why the transformer adds positional encodings (`L5.4`). Each query is computed independently of the others.

Why scale? If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k = \sum_{t=1}^{d} q_t k_t$ is a sum of $d$ terms of variance 1: variance $d$. At $d = 64$ the scores have standard deviation 8, the softmax is nearly one-hot, and its gradient (which is $P(1-P)$-shaped) nearly vanishes. Multiplying by $1/\sqrt{d}$ brings the variance back to 1 for every width. `test_score_variance_is_one` measures exactly this.

### 2.2 Masking

A mask blocks query-key pairs (`L5.2`): set the blocked scores to $-\infty$ **before** the softmax. Then $e^{-\infty} = 0$, the blocked weights are exactly 0, and the open weights still sum to 1. Masking after the softmax (multiplying weights by 0/1) leaves rows that no longer sum to 1. A row with every key blocked is all $-\infty$; the stable softmax of `M09.2` defines its weights as zeros, so its output and gradients are zero instead of NaN. The softmax also subtracts each row's maximum first, so scores of $10^4$ never overflow.

### 2.3 The backward pass

Given $dO$, the gradient of the loss with respect to the output, work backward through $O = PV$, then $P = \operatorname{softmax}(S)$, then $S = \alpha QK^\top$. For a matrix product $C = AB$, the vector-Jacobian products are $dA = dC\,B^\top$ and $dB = A^\top dC$ (`M08.3`). So

$$dV = P^\top dO, \qquad dP = dO\,V^\top .$$

For one row $p = \operatorname{softmax}(s)$, $\partial p_j / \partial s_k = p_j(\delta_{jk} - p_k)$, so $ds_k = \sum_j dp_j\, p_j(\delta_{jk} - p_k) = p_k(dp_k - \sum_j p_j dp_j)$ (`S-M08`):

$$dS = P \circ (dP - D), \qquad D_i = \sum_j P_{ij}\, dP_{ij} .$$

Finally $S = \alpha QK^\top$ gives $dQ = \alpha\, dS\,K$ and $dK = \alpha\, dS^\top Q$. Note the transpose in $dK$: $dS$ is `[Tq, Tk]` and $dK$ must be `[Tk, d]`. A useful identity: $D_i = \sum_j P_{ij}(dO_i \cdot v_j) = dO_i \cdot o_i$, so the row sum can be computed from the output without storing $dP$. FlashAttention's backward uses exactly this. Masked positions have $P = 0$, so every gradient through them is 0: padding never learns.

### 2.4 Batches and heads

Every leading dimension is an independent problem: `[B, H, T, d]` is $B \cdot H$ separate attentions, and numpy's `@` broadcasts over them. `L5.3` reshapes `[B, T, d_model]` into heads and calls this function once.

### 2.5 Attention dropout

During training, the 2017 Transformer drops attention weights: draw one uniform $u$ per weight (in C order, from the PCG32 the caller passes, the same rule as `L0.2`'s dropout), keep where $u \ge p$, and scale the kept ones by $1/(1-p)$ so the expected weight is unchanged: $O = (P \circ m)V$ with $m = \text{keep}/(1-p)$. The backward must use the same $m$: $dV = (P \circ m)^\top dO$ and $dP = (dO\,V^\top) \circ m$, and the softmax VJP is unchanged. The function returns $P$ before dropout, as a constant: the weights are for inspection (heat maps, the KV cache), and gradients flow through $O$ only.

## 3. Worked example by hand

One query, three keys, $d = 4$ so $\alpha = 1/2$. $q = (2, 0, 0, 0)$; keys $(1, 0, 0, 0)$, $(0, 1, 0, 0)$, $(2, 0, 0, 0)$; values $(1, 0)$, $(0, 1)$, $(1, 1)$.

| Step | Computation | Value |
|---|---|---|
| $S$ | $\frac12(2, 0, 4)$ | $(1, 0, 2)$ |
| $P$ | $(e, 1, e^2)/(1 + e + e^2)$, $1 + e + e^2 = 11.107$ | $(0.244728, 0.090031, 0.665241)$ |
| $O$ | $0.244728(1, 0) + 0.090031(0, 1) + 0.665241(1, 1)$ | $(0.909969, 0.755272)$ |

Backward with $dO = (1, 0)$:

| Step | Computation | Value |
|---|---|---|
| $dV$ | $P^\top dO$: row $j$ is $(P_j, 0)$ | $(0.244728, 0)$, $(0.090031, 0)$, $(0.665241, 0)$ |
| $dP$ | $dO \cdot v_j$ | $(1, 0, 1)$ |
| $D$ | $0.244728 + 0.665241$, which is $dO \cdot O$ | $0.909969$ |
| $dS$ | $P \circ (dP - D)$ | $(0.022033, -0.081925, 0.059892)$ |
| $dQ$ | $\frac12 \sum_j dS_j k_j = \frac12(0.022033 + 2 \cdot 0.059892, -0.081925, 0, 0)$ | $(0.070909, -0.040963, 0, 0)$ |
| $dK$ | $\frac12 dS_j\, q$: row $j$ is $(dS_j, 0, 0, 0)$ | $(0.022033, 0, 0, 0)$, $(-0.081925, 0, 0, 0)$, $(0.059892, 0, 0, 0)$ |

The third key scored highest and gets most of the weight; raising its score further (moving $q$ toward it) raises $O_1$, which is why $dS_3 > 0$. These numbers are the first two tests.

## 4. The interface

```python
def sdpa_forward(q, k, v, mask=None, scale=None) -> tuple[NDArray, NDArray]: ...           # (O, P)
def sdpa_backward(q, k, v, p, dout, scale=None, dropout_mult=None) -> tuple[NDArray, NDArray, NDArray]: ...
def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask=None, dropout_p=0.0,
                                 scale=None, rng=None) -> tuple[Tensor, Tensor]: ...      # (out, weights)
```

`scaled_dot_product_attention` computes the forward with `sdpa_forward`, applies dropout if asked, and returns `from_op(out, [q, k, v], vjp)` whose `vjp` is `sdpa_backward`: one node of the graph. Masks are bool (True = may attend) and must broadcast to the scores without changing their shape. float32 stays float32.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_forward` | unit, smoke | section 3's $P$ and $O$ | you and the tests agree on the definition |
| `test_hand_example_backward` | unit, smoke | section 3's $dV$, $dS$, $dQ$, $dK$ | the backward by hand |
| `test_golden_torch` | golden | torch's output, weights, and float64 gradients on five cases (cross attention, causal, padding, custom scale, 3-D) | `L5.3`'s torch comparison builds on it |
| `test_golden_torch_float32` | golden | float32 in, float32 out, within the reduction bound | models train in float32 |
| `test_gradcheck_float64` | gradcheck | every gradient element against central differences, with masks | the hand backward is the derivative |
| `test_gradcheck_through_dropout` | gradcheck | the same with a replayed dropout mask | training-mode gradients |
| `test_key_permutation_equivariance` | property, smoke | permuting keys and values permutes only the weights' columns | why positions are added in `L5.4` |
| `test_queries_are_independent` | property | permuted or single queries give the same rows | `L8.2` decodes one query at a time |
| `test_masked_keys_get_no_weight_or_gradient` | boundary | blocked weights and gradients exactly 0, rows sum to 1 | padding never learns |
| `test_fully_masked_row_gives_zeros` | boundary | zeros, never NaN | empty sequences in a batch |
| `test_large_scores_do_not_overflow` | boundary | scores of $10^4$ give a clean one-hot | sharp attention in trained models |
| `test_default_scale_is_inverse_sqrt_d` | unit | default equals $1/\sqrt{d}$ explicitly | head width changes do not change the math |
| `test_score_variance_is_one` | statistical | scaled score differences have variance 2 at $d = 4$ and 64 | section 2.1's argument |
| `test_dropout_draws_and_scaling` | unit | one uniform per weight, keep $u \ge p$, scale $1/(1-p)$; $p = 0$ draws nothing | the same stream as `L0.2`'s dropout |
| `test_weights_are_a_constant` | unit | `out` requires grad, `weights` does not | no gradient through inspection |
| `test_batched_equals_per_item` | property | `[B, H, T, d]` equals $B \cdot H$ 2-D calls, gradients too | heads are independent |
| `test_rejects_bad_arguments` | boundary | mismatched shapes, float masks, bad scale, bad dropout rate, dropout without rng | wiring bugs fail loudly |

### Your graded tests (rung R5)

The oracle is the formula written out in numpy (scores, masked softmax, weighted sum, in float64); check the section 3 numbers, every mask kind, and a custom scale against it. For the backward, write your own central differences of `sum(oracle(q, k, v) * g)` and compare them with the gradients of the Tensor op and of `sdpa_backward`, with and without dropout (replay the same uniforms for the oracle's mask). Import only contract modules (`tinyllm.xfmr.sdpa`, `tinyllm.autograd.tensor`).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. scaling by $1/d$ instead of $1/\sqrt{d}$ | flat attention that sharpens with width; torch disagrees | `test_hand_example_forward`, `test_score_variance_is_one` (mutant `s01`) |
| 2. the softmax over the query axis | columns sum to 1 instead of rows | `test_hand_example_forward` (mutant `s02`) |
| 3. masking after the softmax | rows sum to less than 1 | `test_masked_keys_get_no_weight_or_gradient` (mutant `s03`) |
| 4. the softmax VJP without the $-D$ term | gradients of a sum-to-one output that do not sum to 0 | `test_hand_example_backward`, `test_gradcheck_float64` (mutant `s04`) |
| 5. $dK = dS\,Q$ without the transpose | a shape error for $T_q \ne T_k$, wrong values otherwise | `test_batched_equals_per_item` (mutant `s05`) |
| 6. forgetting the scale in $dQ$ | gradients $\sqrt{d}$ times too large | `test_gradcheck_float64` (mutant `s06`) |
| 7. a backward that ignores the dropout mask | training gradients of a different function | `test_gradcheck_through_dropout` (mutants `s07`, `s08`) |
| 8. keeping $u < p$ | drops the wrong 1 - p of the weights | `test_dropout_draws_and_scaling` (mutant `s09`) |
| 9. masking with $-10^9$ | a fully masked row attends uniformly to padding | `test_fully_masked_row_gives_zeros` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | `from_op(out, [q, k, v], vjp)`: the whole attention is one node |
| Back | `M09.2` | `softmax` with the max subtracted and an all $-\infty$ row mapped to zeros |
| Back | `L4.3` | the unscaled dot score this module scales (reading) |
| Back | `M08.3` | the matrix-product and softmax VJPs of section 2.3 (reading) |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L5.2` | the masks passed as `mask` |
| Forward | `L5.3` | multi-head attention: `[B, H, T, d_head]` in one call |
| Forward | `L7.5` | grouped-query attention shares keys and values across heads |
| Forward | `L7.6` | latent attention reconstructs keys and values from a small cache |
| Forward | `L8.2` | queries of one step against the cached keys, with `q_offset` masks |
| Forward | `L9.3` | the C kernel (online softmax, tiled) is tested against `sdpa_forward` |
| Forward | `L13.3` | rope2d blocks call `scaled_dot_product_attention` on rotated heads |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sdpa_forward` | `torch.nn.functional.scaled_dot_product_attention` | dispatch to fused kernels (Flash, memory-efficient, math) by shape and device | `aten/src/ATen/native/transformers/attention.cpp` |
| `sdpa_backward` | FlashAttention-2 backward | recomputes $P$ tile by tile from saved row statistics instead of storing it; $D = \operatorname{rowsum}(dO \circ O)$ | `flash_attn/flash_attn_triton.py`, Dao (2023) |
| the softmax over all keys | the online softmax | one pass over the keys with a running max and sum, the basis of `L9.3` | Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018) |
| `scaled_dot_product_attention` | vLLM PagedAttention | keys and values in fixed-size blocks of a shared pool (`L8.3`) | `vllm/attention/` |
