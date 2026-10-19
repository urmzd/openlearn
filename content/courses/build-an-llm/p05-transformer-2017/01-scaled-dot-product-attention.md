# Scaled dot-product attention (forward and backward)

## Overview

| | |
|---|---|
| **Module** | `L5.1` · build · Python · Pass 5 · 1.5 to 2 h reading |
| **Solution** | [`python/tinyllm/xfmr/sdpa.py`](../../../../course/solutions/python/tinyllm/xfmr/sdpa.py): `sdpa_forward`, `sdpa_backward` (by hand), and `scaled_dot_product_attention`, one autograd node; model oracle tests in [`test_sdpa.py`](../../../../course/solutions/learner-tests/L5.1/python/tests/l5-1-sdpa/test_sdpa.py) |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/sdpa.pyi`](../../../../course/contracts/py/tinyllm/xfmr/sdpa.pyi) |
| **Needs** | [`L0.1` `Tensor` and `from_op`](../p00-foundations/01-tensor-and-broadcasting-backward.md) · [`M09.2` stable softmax](../../../tracks/math/09-numerical-methods-and-floating-point/02-stable-numerics.md) · reading: [`L4.3` Luong's dot score](../p04-attention-origins/03-luong-attention-input-feeding.md), [`M08.3`](../../../tracks/math/08-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) and `S-M08` (the softmax VJP), `L0.2` (the dropout draw rule) |
| **Used by** | `L5.3` multi-head attention calls it once per layer · later `L7.5` and `L7.6` (grouped-query and latent attention), `L8.2` (attention over a KV cache), `L9.3` (the C kernel is proven against `sdpa_forward`) · later: `L13.3` · `L14.3` Whisper decoding · the first piece of the 2017 transformer stage |
| **Optional depth** | Vaswani et al., "Attention Is All You Need" (2017), section 3.2.1; Dao et al., "FlashAttention" (2022), section 3.1 and appendix B.2 (the backward with $D = \operatorname{rowsum}(dO \circ O)$) |

## Key Takeaways

- Attention is a soft dictionary lookup: $O = \operatorname{softmax}(QK^\top/\sqrt{d} + \text{mask})\,V$, a weighted average of values with weights from query-key similarity.
- The $1/\sqrt{d}$ keeps the scores' variance at 1 whatever the width, so the softmax neither saturates nor flattens.
- The backward is four matrix products and the softmax VJP $dS = P \circ (dP - \operatorname{rowsum}(dP \circ P))$, and that row sum equals $dO \cdot O$.
- Attention is equivariant to permuting keys with their values: it has no notion of order.
- Blocked keys get weight and gradient exactly 0, and a fully blocked row is zeros, never NaN.

## How to read this chapter

Sections 1 to 3 derive scaled dot-product attention and its backward from first principles and work both by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The sequence models of Part 4 read the source through an RNN: one step per token, and everything the decoder knows about token 3 has passed through every later state. Luong attention (`L4.3`) already reads the encoder states directly with a dot product. The transformer keeps only that read: every position attends to every position in one matrix product, so a sequence of length $T$ costs $T^2$ dot products but no sequential steps, and the gradient from position 50 to position 3 is one hop instead of 47. Every model from here on (the 2017 Transformer, GPT, BERT, the modern Llama block, the inference engine's kernels) is built on this one function. Writing its backward by hand, rather than composing ops, makes it one node of the autograd graph instead of seven, and it is the exact computation the C kernel of `L9.3` and FlashAttention reorganize.

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

Why scale? If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k = \sum_{t=1}^{d} q_t k_t$ is a sum of $d$ terms of variance 1: variance $d$. At $d = 64$ the scores have standard deviation 8, the softmax is nearly one-hot, and its gradient (which is $P(1-P)$-shaped) nearly vanishes. Multiplying by $1/\sqrt{d}$ brings the variance back to 1 for every width. A correct solution shows exactly this: the difference of two scaled scores has variance 2 (each has variance 1) at $d = 4$ and at $d = 64$.

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

The third key scored highest and gets most of the weight; raising its score further (moving $q$ toward it) raises $O_1$, which is why $dS_3 > 0$. A correct solution reproduces both tables.

## 4. The interface

```python
def sdpa_forward(q, k, v, mask=None, scale=None) -> tuple[NDArray, NDArray]: ...           # (O, P)
def sdpa_backward(q, k, v, p, dout, scale=None, dropout_mult=None) -> tuple[NDArray, NDArray, NDArray]: ...
def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask=None, dropout_p=0.0,
                                 scale=None, rng=None) -> tuple[Tensor, Tensor]: ...      # (out, weights)
```

`scaled_dot_product_attention` computes the forward with `sdpa_forward`, applies dropout if asked, and returns `from_op(out, [q, k, v], vjp)` whose `vjp` is `sdpa_backward`: one node of the graph. Masks are bool (True = may attend) and must broadcast to the scores without changing their shape. float32 stays float32.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: forward | unit, smoke | section 3's $P$ and $O$ | pins down the definition |
| hand example: backward | unit, smoke | section 3's $dV$, $dS$, $dQ$, $dK$ | the backward by hand |
| matches torch (float64) | golden | torch's output, weights, and float64 gradients on five cases (cross attention, causal, padding, custom scale, 3-D) | `L5.3`'s torch comparison builds on it |
| matches torch (float32) | golden | float32 in, float32 out, within the reduction bound | models train in float32 |
| gradcheck in float64 | gradcheck | every gradient element against central differences, with masks | the hand backward is the derivative |
| gradcheck through dropout | gradcheck | the same with a replayed dropout mask | training-mode gradients |
| key permutation equivariance | property, smoke | permuting keys and values permutes only the weights' columns | why positions are added in `L5.4` |
| queries are independent | property | permuted or single queries give the same rows | `L8.2` decodes one query at a time |
| masked keys get no weight or gradient | boundary | blocked weights and gradients exactly 0, rows sum to 1 | padding never learns |
| fully masked row gives zeros | boundary | zeros, never NaN | empty sequences in a batch |
| large scores do not overflow | boundary | scores of $10^4$ give a clean one-hot | sharp attention in trained models |
| default scale is inverse sqrt d | unit | default equals $1/\sqrt{d}$ explicitly | head width changes do not change the math |
| score variance is one | statistical | scaled score differences have variance 2 at $d = 4$ and 64 | section 2.1's argument |
| dropout draws and scaling | unit | one uniform per weight, keep $u \ge p$, scale $1/(1-p)$; $p = 0$ draws nothing | the same stream as `L0.2`'s dropout |
| weights are a constant | unit | `out` requires grad, `weights` does not | no gradient through inspection |
| batched equals per item | property | `[B, H, T, d]` equals $B \cdot H$ 2-D calls, gradients too | heads are independent |
| rejects bad arguments | boundary | mismatched shapes, float masks, bad scale, bad dropout rate, dropout without rng | wiring bugs fail loudly |

### The oracle tests

In the model suite [`test_sdpa.py`](../../../../course/solutions/learner-tests/L5.1/python/tests/l5-1-sdpa/test_sdpa.py) the oracle is the formula written out in numpy (scores, masked softmax, weighted sum, in float64); the section 3 numbers, every mask kind, and a custom scale are checked against it. For the backward, central differences of `sum(oracle(q, k, v) * g)` are compared with the gradients of the Tensor op and of `sdpa_backward`, with and without dropout (the same uniforms are replayed for the oracle's mask). It imports only contract modules.

## 5. The solution, step by step

### From first principles

1. **A soft lookup is a softmax over keys followed by a weighted sum of values.** Similarity is a dot product, so all scores are one matrix product $QK^\top$, and all outputs another, $PV$ (section 2.1). Every leading axis is a batch axis.
2. **The scale is forced by variance.** A dot product of $d$ unit-variance terms has variance $d$; multiplying by $1/\sqrt{d}$ restores variance 1 at every width. A caller may override it (some models fold it into $Q$), but it must be positive and finite.
3. **Masks act on scores, with $-\infty$.** Only then are blocked weights exactly 0 and open weights still sum to 1; a fully blocked row is handled by `M09.2`'s convention (zeros). A mask may broadcast (one `[Tq, Tk]` causal mask for every batch and head), but it must not *grow* the scores' shape, which would silently change the output shape.
4. **The backward follows the forward in reverse, one VJP per line.** $dV = P^\top dO$, $dP = dO\,V^\top$, $dS = P \circ (dP - \operatorname{rowsum}(P \circ dP))$, $dQ = \alpha\, dS\,K$, $dK = \alpha\, dS^\top Q$ (section 2.3). The worked example confirms it: $D = 0.909969 = dO \cdot O$.
5. **One autograd node, not seven.** Writing the backward by hand and registering it with `from_op` stores only $Q, K, V, P$ and gives the exact computation fused kernels reorganize.
6. **Dropout is a fixed multiplier on $P$.** The forward uses $P \circ m$; the backward must use the same $m$ on both $dV$ and $dP$. The softmax VJP is applied to the *undropped* $P$, because $m$ sits after the softmax.
7. **The returned weights are for inspection.** They are $P$ before dropout and are wrapped as a constant Tensor, so no gradient path runs through them.

### The code, annotated

The forward, in plain numpy so `L9.3`'s C kernel and `L8.2`'s cache can call it directly:

```python
def sdpa_forward(q, k, v, mask=None, scale=None):
    qa, ka, va = _float(q, "q"), _float(k, "k"), _float(v, "v")   # ints -> float64; float32 stays
    _check(qa, ka, va)                      # same leading dims, q/k widths, k/v lengths
    # S = alpha Q K^T, the scale cast to q's dtype so float32 stays float32.
    s = (qa @ _t(ka)) * qa.dtype.type(_scale(scale, qa.shape[-1]))
    if mask is not None:
        m = np.asarray(mask)
        if m.dtype != np.bool_:
            raise ValueError(f"mask must be bool (True = may attend), got {m.dtype}")
        # ... (the mask must broadcast to s without changing s's shape)
        s = np.where(m, s, s.dtype.type(-np.inf))
    # Softmax over the KEYS (the last axis): each query's weights sum to 1;
    # a row with every key masked is all zeros (M09.2), never NaN.
    p = softmax(s, axis=-1)
    return p @ va, p
```

`_t` swaps the last two axes, so `qa @ _t(ka)` is $QK^\top$ for any number of leading axes. `_scale(None, d)` is `1.0 / math.sqrt(d)`; an explicit scale must be finite and positive.

The backward, line for line with section 2.3:

```python
def sdpa_backward(q, k, v, p, dout, scale=None, dropout_mult=None):
    qa, ka, va = _float(q, "q"), _float(k, "k"), _float(v, "v")
    _check(qa, ka, va)
    pa, g = np.asarray(p), np.asarray(dout)
    # ... (p must be [..., Tq, Tk] and dout [..., Tq, dv])
    s = _scale(scale, qa.shape[-1])
    pd = pa if dropout_mult is None else pa * dropout_mult   # the P the forward actually used
    dv = _t(pd) @ g                                          # dV = (P o m)^T dO
    dp = g @ _t(va)                                          # dP = dO V^T ...
    if dropout_mult is not None:
        dp = dp * dropout_mult                               # ... o m, back through the dropout
    # Softmax VJP: dS = P * (dP - sum_j P_j dP_j), row by row.
    ds = pa * (dp - np.sum(dp * pa, axis=-1, keepdims=True))
    dq = (ds @ ka) * s                                       # dQ = alpha dS K
    dk = (_t(ds) @ qa) * s                                   # dK = alpha dS^T Q (note the transpose)
    return dq.astype(qa.dtype), dk.astype(ka.dtype), dv.astype(va.dtype)
```

Masked positions have $P = 0$, so `ds` is 0 there and `dq`, `dk` receive nothing from them; `dv` rows of masked keys are 0 for the same reason.

The autograd wrapper adds dropout and registers the node:

```python
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, scale=None, rng=None):
    qt, kt, vt = (x if isinstance(x, Tensor) else Tensor(x, dtype=np.asarray(x).dtype) for x in (q, k, v))
    # ... (dropout_p in [0, 1]; an rng is required when dropout_p > 0)
    out, p = sdpa_forward(qt.data, kt.data, vt.data, mask, scale)
    mult = None
    if dropout_p > 0.0:
        u = np.asarray(rng.uniforms(p.size)).reshape(p.shape)   # one uniform per weight, C order
        keep = u >= dropout_p
        mult = (np.zeros_like(p) if dropout_p == 1.0
                else (keep / (1.0 - dropout_p)).astype(p.dtype))  # inverted dropout
        out = (p * mult) @ vt.data
    # dropout_p == 0 draws nothing, so adding the option never shifts the rng stream.

    def vjp(g):
        return sdpa_backward(qt.data, kt.data, vt.data, p, g, scale, mult)

    res = from_op(out.astype(qt.dtype), [qt, kt, vt], vjp, op="scaled_dot_product_attention")
    return res, Tensor(p, dtype=p.dtype)    # weights: a constant, before dropout
```

The closure captures `p` and `mult`, which is exactly the state the backward needs; nothing else from the forward is kept. Full solution: [`sdpa.py`](../../../../course/solutions/python/tinyllm/xfmr/sdpa.py).

### Why this solution

- **Cost.** Forward and backward are each $O(T_q T_k (d + d_v))$ time per batch-head; memory is $O(T_q T_k)$ for $P$, kept for the backward. That quadratic memory is what FlashAttention removes by recomputing $P$ in tiles from saved row statistics.
- **Why a hand-written backward instead of composing `matmul`, `masked_fill`, `softmax`?** The composed graph would be correct but store every intermediate and create seven nodes per call. The hand backward stores only $P$ and is the reference that `L9.3`'s kernel is tested against.
- **Why compute $D$ as `sum(dp * pa)` and not $dO \cdot O$?** They are equal, with or without dropout (with dropout $O = (P \circ m)V$ and $dP$ carries the same $m$). Here $dP$ is already materialized, so the direct row sum is the simplest; FlashAttention uses $dO \cdot O$ because it never holds a full row of $dP$ at once.
- **Why $-\infty$ rather than a large negative number?** A finite fill makes a fully masked row uniform over padding instead of zero (pitfall 9).
- **Why return the weights before dropout and detached?** They are for heat maps and caches; making them a gradient path would let an inspection call change training.

### Explain it back

1. Why is the default scale $1/\sqrt{d}$ and not $1/d$?
   <details><summary>Answer</summary>The variance of $q \cdot k$ grows as $d$, so its standard deviation grows as $\sqrt{d}$. Dividing by $\sqrt{d}$ restores unit variance; dividing by $d$ would shrink the scores toward 0 as width grows and flatten the softmax.</details>
2. In the worked example, why is $dS$'s sum zero, and which line of the code guarantees it?
   <details><summary>Answer</summary>$\sum_k dS_k = \sum_k P_k (dP_k - D) = D - D \cdot 1 = 0$, since the weights sum to 1: moving all scores by a constant cannot change a softmax. The `- np.sum(dp * pa, ...)` term in `ds` is what makes it hold.</details>
3. Why does `dk` use `_t(ds) @ qa` while `dq` uses `ds @ ka`?
   <details><summary>Answer</summary>$S = \alpha Q K^\top$ is `[Tq, Tk]`. $dQ$ must be `[Tq, d]`, which is $dS\,K$; $dK$ must be `[Tk, d]`, which needs $dS^\top Q$. Without the transpose the shapes only match when $T_q = T_k$, and the values are wrong.</details>
4. With dropout, why is the softmax VJP applied to `pa` and not `pd`?
   <details><summary>Answer</summary>The dropout multiplier comes after the softmax: $O = (P \circ m)V$. The chain rule goes through $m$ first (`dp * dropout_mult`), then through the softmax, whose Jacobian depends on the undropped $P$.</details>
5. Why must a mask be rejected if it would change the scores' shape?
   <details><summary>Answer</summary>Broadcasting a larger mask would make `np.where` return a larger array, so the output would gain axes the caller never asked for. Rejecting it turns a silent shape bug into an error.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. scaling by $1/d$ instead of $1/\sqrt{d}$ | flat attention that sharpens with width; torch disagrees | `_scale` returns `1.0 / math.sqrt(d)` by default |
| 2. the softmax over the query axis | columns sum to 1 instead of rows | `softmax(s, axis=-1)`: the key axis |
| 3. masking after the softmax | rows sum to less than 1 | `np.where(m, s, -inf)` on the scores, then the softmax |
| 4. the softmax VJP without the $-D$ term | gradients of a sum-to-one output that do not sum to 0 | `ds = pa * (dp - np.sum(dp * pa, axis=-1, keepdims=True))` |
| 5. $dK = dS\,Q$ without the transpose | a shape error for $T_q \ne T_k$, wrong values otherwise | `dk = (_t(ds) @ qa) * s` |
| 6. forgetting the scale in $dQ$ | gradients $\sqrt{d}$ times too large | both `dq` and `dk` are multiplied by `s` |
| 7. a backward that ignores the dropout mask | training gradients of a different function | the vjp passes `mult` to `sdpa_backward`, which applies it to both `dv` and `dp` |
| 8. keeping $u < p$ | drops the wrong 1 - p of the weights | `keep = u >= dropout_p` |
| 9. masking with $-10^9$ | a fully masked row attends uniformly to padding | masks with `-np.inf`; `M09.2`'s softmax maps an all-$-\infty$ row to zeros |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | `from_op(out, [q, k, v], vjp)`: the whole attention is one node |
| Back | `M09.2` | `softmax` with the max subtracted and an all $-\infty$ row mapped to zeros |
| Back | `L4.3` | the unscaled dot score this module scales (reading) |
| Back | `M08.3` | the matrix-product and softmax VJPs of section 2.3 (reading) |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract |
| Forward | `L5.2` | the masks passed as `mask` |
| Forward | `L5.3` | multi-head attention: `[B, H, T, d_head]` in one call |
| Forward | `L7.5` | grouped-query attention shares keys and values across heads |
| Forward | `L7.6` | latent attention reconstructs keys and values from a small cache |
| Forward | `L8.2` | queries of one step against the cached keys, with `q_offset` masks |
| Forward | `L9.3` | the C kernel (online softmax, tiled) is tested against `sdpa_forward` |
| Forward | `L13.3` | rope2d blocks call `scaled_dot_product_attention` on rotated heads |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sdpa_forward` | `torch.nn.functional.scaled_dot_product_attention` | dispatch to fused kernels (Flash, memory-efficient, math) by shape and device | `aten/src/ATen/native/transformers/attention.cpp` |
| `sdpa_backward` | FlashAttention-2 backward | recomputes $P$ tile by tile from saved row statistics instead of storing it; $D = \operatorname{rowsum}(dO \circ O)$ | `flash_attn/flash_attn_triton.py`, Dao (2023) |
| the softmax over all keys | the online softmax | one pass over the keys with a running max and sum, the basis of `L9.3` | Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018) |
| `scaled_dot_product_attention` | vLLM PagedAttention | keys and values in fixed-size blocks of a shared pool (`L8.3`) | `vllm/attention/` |
