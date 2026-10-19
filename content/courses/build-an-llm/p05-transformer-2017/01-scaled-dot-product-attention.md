# Scaled dot-product attention (forward and backward)

The sequence models of the previous chapter read the source through an RNN: one step per token, and everything the decoder knows about token 3 has passed through every later state. [Luong attention](../p04-attention-origins/03-luong-attention-input-feeding.md) already reads the encoder states directly with a dot product. The transformer keeps only that read. Every position attends to every position in one matrix product, so a sequence of length $T$ costs $T^2$ dot products but no sequential steps, and the gradient from position 50 to position 3 is one hop instead of 47.

Every model from here on (the 2017 Transformer, GPT, BERT, the modern Llama block, the inference engine's kernels) is built on this one function. In this section you will derive it, derive its backward pass by hand, and register the whole thing as a single node of the autograd graph instead of seven composed ops. That hand-written backward is also the exact computation that the [FlashAttention forward in C](../p09-kernels/06-flash-attention-forward.md) and FlashAttention itself reorganize.

This section builds on [Tensor, broadcasting backward, and no_grad](../p00-foundations/01-tensor-and-broadcasting-backward.md) (the `Tensor` class and `from_op`) and on the stable softmax of [Stable numerics: logsumexp, softmax, and compensated sums](../../mathematics-for-computing/09-numerical-methods-and-floating-point/02-stable-numerics.md) in *Mathematics for Computing*. For the backward it helps to have read [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics-for-computing/08-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) and the softmax VJP derivation in the [matrix calculus problem set](../../mathematics-for-computing/08-matrix-calculus-and-autodiff/90-problem-set.md); the dropout draw rule comes from [The op library and gradcheck_all](../p00-foundations/02-op-library-and-gradcheck.md). The full solution is [`python/tinyllm/xfmr/sdpa.py`](../../../../course/solutions/python/tinyllm/xfmr/sdpa.py) (`sdpa_forward`, `sdpa_backward`, and `scaled_dot_product_attention`), its contract is [`course/contracts/py/tinyllm/xfmr/sdpa.pyi`](../../../../course/contracts/py/tinyllm/xfmr/sdpa.pyi), and the model oracle tests are in [`test_sdpa.py`](../../../../course/solutions/learner-tests/L5.1/python/tests/l5-1-sdpa/test_sdpa.py).

> **Learning objectives**
> - Compute $O = \operatorname{softmax}(QK^\top/\sqrt{d} + \text{mask})\,V$ by hand and explain why the scale is $1/\sqrt{d}$.
> - Derive the backward pass, including the softmax VJP $dS = P \circ (dP - \operatorname{rowsum}(dP \circ P))$, and show that the row sum equals $dO \cdot O$.
> - Implement forward, backward, and attention dropout as one autograd node with the same multiplier in both directions.
> - Explain why masking uses $-\infty$ before the softmax, and why a fully blocked row gives zeros rather than NaN.

## The notation

The whole section uses one set of symbols; the table lists each with its shape.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $Q$ | queries, one row per position asking | `[..., Tq, d]` |
| $K$ | keys, one row per position offering | `[..., Tk, d]` |
| $V$ | values, what each key position contributes | `[..., Tk, dv]` |
| $d$ | query and key width | `int` |
| $\alpha$ (`scale`) | score scale, default $1/\sqrt{d}$ | `float` |
| $S = \alpha QK^\top$ | scores | `[..., Tq, Tk]` |
| $M$ | mask, True = may attend (see [Masks](02-masks.md)) | `bool`, broadcast to $S$ |
| $P = \operatorname{softmax}(S)$ | weights, each row sums to 1 (over keys) | `[..., Tq, Tk]` |
| $O = PV$ | output | `[..., Tq, dv]` |
| $dO, dP, dS, dQ, dK, dV$ | gradients of a scalar loss with respect to each | same shapes as their arrays |
| $D_i = \sum_j P_{ij}\, dP_{ij}$ | the row sum in the softmax VJP | `[..., Tq, 1]` |

## Attention as a soft lookup

A Python dict returns the value whose key equals the query. Attention returns a blend. It scores query $q_i$ against every key $k_j$ by the dot product (large when they point the same way), turns the scores into positive weights that sum to 1 with a softmax over the keys, and averages the values with those weights: $o_i = \sum_j P_{ij} v_j$. Because the output is a weighted sum over a *set* of key-value pairs, reordering the pairs changes nothing. Attention has no idea of position, which is why the transformer adds [positional encodings](04-positional-encodings.md). Each query is also computed independently of the others.

Why scale? If the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k = \sum_{t=1}^{d} q_t k_t$ is a sum of $d$ terms of variance 1, so it has variance $d$. At $d = 64$ the scores have standard deviation 8, the softmax is nearly one-hot, and its gradient (which is $P(1-P)$-shaped) nearly vanishes. Multiplying by $1/\sqrt{d}$ brings the variance back to 1 for every width. You can check this empirically: the difference of two scaled scores has variance 2 (each has variance 1) at $d = 4$ and at $d = 64$ alike.

## Masking

A mask blocks query-key pairs, and the next section builds every kind you will need. The mechanism belongs here: set the blocked scores to $-\infty$ **before** the softmax. Then $e^{-\infty} = 0$, the blocked weights are exactly 0, and the open weights still sum to 1. Masking after the softmax (multiplying weights by 0/1) leaves rows that no longer sum to 1.

A row with every key blocked is all $-\infty$. The stable softmax defines its weights as zeros, so its output and gradients are zero instead of NaN. The same softmax subtracts each row's maximum first, so scores of $10^4$ never overflow.

## The backward pass

With the forward fixed, the backward follows it in reverse. Given $dO$, the gradient of the loss with respect to the output, you work back through $O = PV$, then $P = \operatorname{softmax}(S)$, then $S = \alpha QK^\top$. For a matrix product $C = AB$, the vector-Jacobian products are $dA = dC\,B^\top$ and $dB = A^\top dC$. So

$$dV = P^\top dO, \qquad dP = dO\,V^\top .$$

For one row $p = \operatorname{softmax}(s)$, $\partial p_j / \partial s_k = p_j(\delta_{jk} - p_k)$, so $ds_k = \sum_j dp_j\, p_j(\delta_{jk} - p_k) = p_k(dp_k - \sum_j p_j dp_j)$. Stacked over rows:

$$dS = P \circ (dP - D), \qquad D_i = \sum_j P_{ij}\, dP_{ij} .$$

Finally $S = \alpha QK^\top$ gives $dQ = \alpha\, dS\,K$ and $dK = \alpha\, dS^\top Q$. Note the transpose in $dK$: $dS$ is `[Tq, Tk]` and $dK$ must be `[Tk, d]`.

One identity is worth knowing: $D_i = \sum_j P_{ij}(dO_i \cdot v_j) = dO_i \cdot o_i$, so the row sum can be computed from the output without storing $dP$. FlashAttention's backward uses exactly this. And because masked positions have $P = 0$, every gradient through them is 0: padding never learns.

## Batches and heads

Every leading dimension is an independent problem. An input of shape `[B, H, T, d]` is $B \cdot H$ separate attentions, and numpy's `@` broadcasts over them. [Multi-head attention](03-multi-head-attention.md) reshapes `[B, T, d_model]` into heads and calls this function once.

## Attention dropout

During training, the 2017 Transformer drops attention weights. Draw one uniform $u$ per weight (in C order, from the PCG32 the caller passes, the same rule as the op library's dropout), keep where $u \ge p$, and scale the kept ones by $1/(1-p)$ so the expected weight is unchanged: $O = (P \circ m)V$ with $m = \text{keep}/(1-p)$.

The backward must use the same $m$: $dV = (P \circ m)^\top dO$ and $dP = (dO\,V^\top) \circ m$, and the softmax VJP is unchanged. The function returns $P$ before dropout, as a constant. The weights are for inspection (heat maps, the KV cache), and gradients flow through $O$ only.

## Working it by hand

Now put numbers through every formula. Take one query, three keys, and $d = 4$, so $\alpha = 1/2$. The query is $q = (2, 0, 0, 0)$; the keys are $(1, 0, 0, 0)$, $(0, 1, 0, 0)$, $(2, 0, 0, 0)$; the values are $(1, 0)$, $(0, 1)$, $(1, 1)$. The forward pass is three steps:

| Step | Computation | Value |
|---|---|---|
| $S$ | $\frac12(2, 0, 4)$ | $(1, 0, 2)$ |
| $P$ | $(e, 1, e^2)/(1 + e + e^2)$, $1 + e + e^2 = 11.107$ | $(0.244728, 0.090031, 0.665241)$ |
| $O$ | $0.244728(1, 0) + 0.090031(0, 1) + 0.665241(1, 1)$ | $(0.909969, 0.755272)$ |

The backward with $dO = (1, 0)$ follows the derivation line by line:

| Step | Computation | Value |
|---|---|---|
| $dV$ | $P^\top dO$: row $j$ is $(P_j, 0)$ | $(0.244728, 0)$, $(0.090031, 0)$, $(0.665241, 0)$ |
| $dP$ | $dO \cdot v_j$ | $(1, 0, 1)$ |
| $D$ | $0.244728 + 0.665241$, which is $dO \cdot O$ | $0.909969$ |
| $dS$ | $P \circ (dP - D)$ | $(0.022033, -0.081925, 0.059892)$ |
| $dQ$ | $\frac12 \sum_j dS_j k_j = \frac12(0.022033 + 2 \cdot 0.059892, -0.081925, 0, 0)$ | $(0.070909, -0.040963, 0, 0)$ |
| $dK$ | $\frac12 dS_j\, q$: row $j$ is $(dS_j, 0, 0, 0)$ | $(0.022033, 0, 0, 0)$, $(-0.081925, 0, 0, 0)$, $(0.059892, 0, 0, 0)$ |

The third key scored highest and gets most of the weight. Raising its score further (moving $q$ toward it) raises $O_1$, which is why $dS_3 > 0$. Notice also that $D = 0.909969$ is exactly $O_1 = dO \cdot O$, the identity from the derivation. A correct implementation reproduces both tables.

## Designing the API

The design has three functions: a numpy forward and backward that other code (the C kernel's tests, the KV cache) can call directly, and a Tensor-level wrapper that adds dropout and registers the node.

```python
def sdpa_forward(q, k, v, mask=None, scale=None) -> tuple[NDArray, NDArray]: ...           # (O, P)
def sdpa_backward(q, k, v, p, dout, scale=None, dropout_mult=None) -> tuple[NDArray, NDArray, NDArray]: ...
def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask=None, dropout_p=0.0,
                                 scale=None, rng=None) -> tuple[Tensor, Tensor]: ...      # (out, weights)
```

`scaled_dot_product_attention` computes the forward with `sdpa_forward`, applies dropout if asked, and returns `from_op(out, [q, k, v], vjp)` whose `vjp` is `sdpa_backward`: one node of the graph. Masks are bool (True = may attend) and must broadcast to the scores without changing their shape. float32 inputs stay float32.

## Building it step by step

Before the code, here is the reasoning chain that forces each choice.

1. **A soft lookup is a softmax over keys followed by a weighted sum of values.** Similarity is a dot product, so all scores are one matrix product $QK^\top$, and all outputs another, $PV$. Every leading axis is a batch axis.
2. **The scale is forced by variance.** A dot product of $d$ unit-variance terms has variance $d$; multiplying by $1/\sqrt{d}$ restores variance 1 at every width. A caller may override it (some models fold it into $Q$), but it must be positive and finite.
3. **Masks act on scores, with $-\infty$.** Only then are blocked weights exactly 0 and open weights still sum to 1; a fully blocked row follows the stable softmax's convention (zeros). A mask may broadcast (one `[Tq, Tk]` causal mask for every batch and head), but it must not *grow* the scores' shape, which would silently change the output shape.
4. **The backward follows the forward in reverse, one VJP per line.** $dV = P^\top dO$, $dP = dO\,V^\top$, $dS = P \circ (dP - \operatorname{rowsum}(P \circ dP))$, $dQ = \alpha\, dS\,K$, $dK = \alpha\, dS^\top Q$. The worked example confirms it: $D = 0.909969 = dO \cdot O$.
5. **One autograd node, not seven.** Writing the backward by hand and registering it with `from_op` stores only $Q, K, V, P$ and gives the exact computation that fused kernels reorganize.
6. **Dropout is a fixed multiplier on $P$.** The forward uses $P \circ m$; the backward must use the same $m$ on both $dV$ and $dP$. The softmax VJP is applied to the *undropped* $P$, because $m$ sits after the softmax.
7. **The returned weights are for inspection.** They are $P$ before dropout and are wrapped as a constant Tensor, so no gradient path runs through them.

### The forward

The forward is plain numpy, so the C kernel's tests and the KV cache can call it directly:

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
    # a row with every key masked is all zeros (stable softmax), never NaN.
    p = softmax(s, axis=-1)
    return p @ va, p
```

`_t` swaps the last two axes, so `qa @ _t(ka)` is $QK^\top$ for any number of leading axes. `_scale(None, d)` is `1.0 / math.sqrt(d)`; an explicit scale must be finite and positive.

### The backward

The backward matches the derivation line for line:

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

Masked positions have $P = 0$, so `ds` is 0 there and `dq`, `dk` receive nothing from them; the `dv` rows of masked keys are 0 for the same reason.

### The autograd wrapper

The wrapper adds dropout and registers the node:

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

The closure captures `p` and `mult`, which is exactly the state the backward needs; nothing else from the forward is kept. The full solution is [`sdpa.py`](../../../../course/solutions/python/tinyllm/xfmr/sdpa.py).

### Why this design

Forward and backward each cost $O(T_q T_k (d + d_v))$ time per batch-head, and memory is $O(T_q T_k)$ for $P$, kept for the backward. That quadratic memory is what FlashAttention removes by recomputing $P$ in tiles from saved row statistics.

You could instead compose `matmul`, `masked_fill`, and `softmax` from the op library. The composed graph would be correct, but it would store every intermediate and create seven nodes per call. The hand backward stores only $P$, and it is the reference the C kernel is tested against.

Why compute $D$ as `sum(dp * pa)` and not as $dO \cdot O$? They are equal, with or without dropout (with dropout $O = (P \circ m)V$ and $dP$ carries the same $m$). Here $dP$ is already materialized, so the direct row sum is simplest. FlashAttention uses $dO \cdot O$ because it never holds a full row of $dP$ at once.

Why $-\infty$ rather than a large negative number? A finite fill makes a fully masked row uniform over padding instead of zero. And why return the weights before dropout and detached? They are for heat maps and caches; making them a gradient path would let an inspection call change training.

## What can go wrong

A correct implementation has to survive a specific list of checks, and each one guards something later chapters rely on. The hand example pins down the definition (forward) and the hand backward ($dV$, $dS$, $dQ$, $dK$). Against torch, the output, weights, and float64 gradients must match on five cases (cross attention, causal, padding, custom scale, 3-D); [multi-head attention](03-multi-head-attention.md) builds its own torch comparison on this. In float32, inputs stay float32 and agree within the reduction bound, since models train in float32. A float64 gradcheck compares every gradient element against central differences, with masks, and a second gradcheck replays a fixed dropout mask so training-mode gradients are covered too.

Several properties follow from the math. Permuting keys and values together permutes only the weights' columns (key permutation equivariance, the reason positions must be added). Permuted or single queries give the same rows, which is what lets the [KV cache](../p08-inference/02-kv-cache-and-generate.md) decode one query at a time. A batched `[B, H, T, d]` call equals $B \cdot H$ separate 2-D calls, gradients included.

At the boundaries, blocked keys get weights and gradients of exactly 0 while rows still sum to 1; a fully masked row gives zeros, never NaN (empty sequences in a batch); and scores of $10^4$ give a clean one-hot rather than overflowing. The default scale must equal $1/\sqrt{d}$ explicitly, and scaled score differences must have variance 2 at both $d = 4$ and $d = 64$. Dropout must draw one uniform per weight, keep $u \ge p$, scale by $1/(1-p)$, and draw nothing at $p = 0$, so it shares a stream with the op library's dropout. The returned `out` requires grad and `weights` does not. Finally, mismatched shapes, float masks, a bad scale, a bad dropout rate, and dropout without an rng must all raise, so wiring bugs fail loudly.

The model suite [`test_sdpa.py`](../../../../course/solutions/learner-tests/L5.1/python/tests/l5-1-sdpa/test_sdpa.py) uses the formula written out in numpy as its oracle (scores, masked softmax, weighted sum, in float64); the worked numbers, every mask kind, and a custom scale are checked against it. For the backward, central differences of `sum(oracle(q, k, v) * g)` are compared with the gradients of the Tensor op and of `sdpa_backward`, with and without dropout (the same uniforms are replayed for the oracle's mask). It imports only contract modules.

The common mistakes are worth listing with their symptoms, because each produces a model that runs but is subtly wrong:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| scaling by $1/d$ instead of $1/\sqrt{d}$ | flat attention that sharpens with width; torch disagrees | `_scale` returns `1.0 / math.sqrt(d)` by default |
| the softmax over the query axis | columns sum to 1 instead of rows | `softmax(s, axis=-1)`: the key axis |
| masking after the softmax | rows sum to less than 1 | `np.where(m, s, -inf)` on the scores, then the softmax |
| the softmax VJP without the $-D$ term | gradients of a sum-to-one output that do not sum to 0 | `ds = pa * (dp - np.sum(dp * pa, axis=-1, keepdims=True))` |
| $dK = dS\,Q$ without the transpose | a shape error for $T_q \ne T_k$, wrong values otherwise | `dk = (_t(ds) @ qa) * s` |
| forgetting the scale in $dQ$ | gradients $\sqrt{d}$ times too large | both `dq` and `dk` are multiplied by `s` |
| a backward that ignores the dropout mask | training gradients of a different function | the vjp passes `mult` to `sdpa_backward`, which applies it to both `dv` and `dp` |
| keeping $u < p$ | drops the wrong $1 - p$ of the weights | `keep = u >= dropout_p` |
| masking with $-10^9$ | a fully masked row attends uniformly to padding | masks with `-np.inf`; the stable softmax maps an all-$-\infty$ row to zeros |

> **Warning:** the last pitfall is the hardest to notice. A finite fill such as $-10^9$ looks equivalent to $-\infty$ on every row that has at least one open key; it only differs on fully masked rows, where it silently averages over padding.

## Where this leads

This function is the first piece of the 2017 transformer and is called everywhere after it. The [masks](02-masks.md) of the next section are what you pass as `mask`. [Multi-head attention](03-multi-head-attention.md) calls it once per layer on `[B, H, T, d_head]`. In the modern block, [MQA/GQA attention](../p07-modern-block/05-gqa-attention.md) shares keys and values across heads and [multi-head latent attention](../p07-modern-block/06-multi-head-latent-attention.md) reconstructs them from a small cache. The [KV cache](../p08-inference/02-kv-cache-and-generate.md) runs one step's queries against cached keys with `q_offset` masks, and the [FlashAttention forward in C](../p09-kernels/06-flash-attention-forward.md) (online softmax, tiled) is tested against `sdpa_forward`. The [vision transformer](../p13-vision/03-vision-transformer.md) calls `scaled_dot_product_attention` on 2D-rotated heads, and [Whisper decoding](../p14-speech/03-whisper-decoder-and-weights.md) calls it through its declared contract.

Each piece also has a production equivalent worth reading:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `sdpa_forward` | `torch.nn.functional.scaled_dot_product_attention` | dispatch to fused kernels (Flash, memory-efficient, math) by shape and device | `aten/src/ATen/native/transformers/attention.cpp` |
| `sdpa_backward` | FlashAttention-2 backward | recomputes $P$ tile by tile from saved row statistics instead of storing it; $D = \operatorname{rowsum}(dO \circ O)$ | `flash_attn/flash_attn_triton.py`, Dao (2023) |
| the softmax over all keys | the online softmax | one pass over the keys with a running max and sum, the basis of the C FlashAttention kernel | Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018) |
| `scaled_dot_product_attention` | vLLM PagedAttention | keys and values in fixed-size blocks of a shared pool (see the [paged KV cache](../p08-inference/03-paged-kv-cache.md)) | `vllm/attention/` |

### Further reading

- Vaswani et al., "Attention Is All You Need" (2017), section 3.2.1.
- Dao et al., "FlashAttention" (2022), section 3.1 and appendix B.2 (the backward with $D = \operatorname{rowsum}(dO \circ O)$).

## Key terms

- **scaled dot-product attention**: $\operatorname{softmax}(\alpha QK^\top + \text{mask})\,V$, a weighted average of values with weights from query-key similarity.
- **score scale**: the factor $\alpha$, by default $1/\sqrt{d}$, that keeps score variance at 1 whatever the width.
- **softmax VJP**: $dS = P \circ (dP - D)$ with $D_i = \sum_j P_{ij}\,dP_{ij}$, the backward of a row-wise softmax.
- **attention dropout**: an inverted-dropout multiplier $m$ applied to $P$ after the softmax, reused unchanged in the backward.
- **permutation equivariance**: permuting key-value pairs permutes only the weight columns and leaves the output unchanged.

## Summary

- Attention is a soft dictionary lookup: $O = \operatorname{softmax}(QK^\top/\sqrt{d} + \text{mask})\,V$, a weighted average of values with weights from query-key similarity.
- The $1/\sqrt{d}$ keeps the scores' variance at 1 whatever the width, so the softmax neither saturates nor flattens.
- The backward is four matrix products and the softmax VJP $dS = P \circ (dP - \operatorname{rowsum}(dP \circ P))$, and that row sum equals $dO \cdot O$.
- Attention is equivariant to permuting keys with their values: it has no notion of order.
- Blocked keys get weight and gradient exactly 0, and a fully blocked row is zeros, never NaN.

## Review questions

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
