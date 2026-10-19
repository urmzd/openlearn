# Multi-Head Latent Attention and Weight Absorption

The grouped-query layer of [Grouped-Query Attention](05-gqa-attention.md) already shrank SmolLM2's cache by three: 3 kv heads instead of 9. The cache still grows by $2 H_{kv} d_h$ numbers per token per layer, and at serving time, where [Continuous batching](../p10-serving/02-continuous-batching.md) admits requests by KV bytes, that is what limits how many conversations fit in memory.

DeepSeek-V2 cut it much further with a different idea: cache a low-rank summary of the token and rebuild keys and values from it. Built naively, that costs compute at every step; built with weight absorption, it costs almost nothing. This section builds both forms of multi-head latent attention (MLA) and proves they agree, so the [TinyStories capstone](../capstones/01-tinystories.md) can compare MLA and GQA at equal KV bytes.

You will reuse [Rotary Position Embeddings](03-rope.md) (`apply_rope`, `rope_cos_sin`, `RopeSpec`), `RMSNorm` from [Pre-LN and RMSNorm](01-pre-ln-rmsnorm.md), `Linear` and `Module` from [Module system and basic layers](../p00-foundations/04-module-system-and-layers.md), the op library from [The op library and gradcheck_all](../p00-foundations/02-op-library-and-gradcheck.md), and the `Tensor` from [Tensor, broadcasting backward, and no_grad](../p00-foundations/01-tensor-and-broadcasting-backward.md). From *Mathematics for Computing* you need `PCG32` ([Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md)) and `kv_bytes_per_token` ([Counting: params, FLOPs, KV bytes, memory plans](../../mathematics-for-computing/05-discrete-math-1/01-counting-params-flops-kv-bytes.md)); [SVD, Eckart-Young low rank, and least squares](../../mathematics-for-computing/03-linear-algebra/05-svd-low-rank-and-least-squares.md) and [Stable numerics](../../mathematics-for-computing/09-numerical-methods-and-floating-point/02-stable-numerics.md) are useful background, as is the cache hook from the previous section. The reference solution is [`mla.py`](../../../../course/solutions/python/tinyllm/modern/mla.py), with `MLAttention`, `ConcatLatentCache`, and `AbsorbedMLA`, against the contract [`course/contracts/py/tinyllm/modern/mla.pyi`](../../../../course/contracts/py/tinyllm/modern/mla.pyi).

> **Learning objectives**
> - Explain what MLA caches per token ($r + d_r$ numbers) and why that is independent of the number of heads.
> - Derive weight absorption for the nope score and the output, and explain why the rope part cannot be absorbed.
> - Work a one-head MLA layer by hand in both the naive and absorbed forms.
> - Lay out every projection as Hugging Face does so DeepSeek checkpoints load by name.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_t \in \mathbb{R}^d$ | the input of token $t$ | `float32[d]` |
| $H$ | query heads | `int` |
| $r$ | `kv_lora_rank`: width of the latent | `int` |
| $r_q$ | `q_lora_rank`: width of the query bottleneck (`None`: none) | `int` |
| $d_n$, $d_r$, $d_v$ | `qk_nope_dim`, `qk_rope_dim`, `v_dim` per head | `int` |
| $c_t \in \mathbb{R}^r$ | the latent of token $t$: what the cache holds | `float32[r]` |
| $k^R_t \in \mathbb{R}^{d_r}$ | the rotated rope key, shared by every head | `float32[d_r]` |
| $W_{UK}^h \in \mathbb{R}^{d_n \times r}$, $W_{UV}^h \in \mathbb{R}^{d_v \times r}$ | head $h$'s rows of `kv_b_proj` | matrices |
| $W_O^h \in \mathbb{R}^{d \times d_v}$ | head $h$'s columns of `o_proj` | matrix |
| $R_p$ | the RoPE rotation at position $p$ | $d_r \times d_r$ |
| $s$ | `softmax_scale` $= (d_n + d_r)^{-1/2}$ | `float` |

## Low rank, again

A matrix of rank $r$ factors as a product of an $m \times r$ and an $r \times n$ matrix, and truncating the SVD keeps the best rank-$r$ approximation. MLA bets that the keys and values of all heads, stacked as one $H(d_n + d_v)$-wide vector per token, live near an $r$-dimensional subspace. It learns the factorization directly: a down projection to $c_t$ (`kv_a_proj_with_mqa`, the first $r$ outputs), an RMSNorm (`kv_a_layernorm`), and an up projection `kv_b_proj` of shape $[H(d_n + d_v), r]$. SmolLM2-sized MHA caches $2 H d_h$ numbers per token; DeepSeek-V3 uses $r = 512$ and $d_r = 64$ for 128 heads.

## The latent and the decoupled rope key

Here is the naive form, for each token:

$$c_t = \mathrm{RMSNorm}(W_{DKV} x_t), \qquad k^{N,h}_t = W_{UK}^h c_t, \qquad v^h_t = W_{UV}^h c_t,$$
$$k^R_t = R_t\, W_{KR} x_t, \qquad q^h_t = [\,q^{N,h}_t \mid R_t\, q^{R,h}_t\,],$$
$$\mathrm{score}^h(t, j) = s\,\big(q^{N,h}_t \cdot k^{N,h}_j + (R_t q^{R,h}_t) \cdot k^R_j\big).$$

Each head's query is `[nope | rope]` with the nope part first, and `kv_b_proj` splits per head into `[k_nope | v]`, nope first. Positions enter only through $R$, so a shift of every position changes nothing, the same law RoPE always obeys. The rope key is one vector shared by all heads, like multi-query attention, which keeps it to $d_r$ cached numbers.

## What the cache holds

The cache receives $c_t$ after the norm and $k^R_t$ after the rotation: $r + d_r$ numbers per token per layer. `ConcatLatentCache.update(layer, c, k_rope)` appends a chunk and returns everything held. A query of a chunk of $T$ new tokens sits at key index $T_k - T + t$, exactly as in grouped-query attention, so prefilling in chunks equals one full forward.

## Weight absorption

The naive form rebuilds every head's keys and values from every cached latent at every step. Linearity lets you avoid that. Rewrite the nope score and the output with the matrix products moved around:

$$q^{N,h}_t \cdot (W_{UK}^h c_j) = \big((W_{UK}^h)^\top q^{N,h}_t\big) \cdot c_j, \qquad W_O^h \sum_j p_j W_{UV}^h c_j = (W_O^h W_{UV}^h) \sum_j p_j c_j .$$

The query is mapped into latent space once ($q^{lat,h}_t = (W_{UK}^h)^\top q^{N,h}_t$, an $r$-vector), the scores are dot products with the cached $c_j$, the weighted latent $\sum_j p_j c_j$ is formed once per head, and $W_O^h W_{UV}^h$ (precomputed, $d \times r$) maps it out. No per-head key or value is ever built: decoding reads $r + d_r$ numbers per cached token. RoPE's part cannot be absorbed, because it depends on $t - j$, and that is the reason for the decoupled rope key.

Absorption is an inference rewrite. Training uses the naive form built from the op library, so gradients reach every projection through the latent, its norm, and the rope key.

## Working it by hand

Take $d = 2$, one head, $r = 1$, $d_n = 1$, $d_r = 2$, $d_v = 1$, and `inv_freq` $= (\pi/2)$, so position 1 rotates the rope pair by 90 degrees: $(a, b) \mapsto (-b, a)$. Choose weights so that the query is $q = [x_0 \mid x_0, x_1]$; $c_{raw} = x_0 + x_1$ and $k^R = (x_1, x_0)$; `kv_b_proj` gives $k^N = 2c$ and $v = 3c$; and `o_proj` outputs $(o, 0)$. The latent norm of a single number is $c / |c| = \pm 1$ (the $\epsilon = 10^{-6}$ changes the 7th digit).

- Token 0 is $x = (1, 1)$ at position 0. Then $c_{raw} = 2$, so $c = 1$; $k^R = (1, 1)$, not rotated; and $v = 3$. It sees only itself, so its output is $(3, 0)$.
- Token 1 is $x = (0, -1)$ at position 1. Then $c_{raw} = -1$, so $c = -1$ and $v = -3$; $k^R = (-1, 0)$ rotates to $(0, -1)$. The query has $q^N = 0$ and $q^R = (0, -1)$, which rotates to $(1, 0)$.
- Token 1's score on key 0 is $0 \cdot 2 + (1, 0)\cdot(1, 1) = 1$, and on key 1 it is $0 \cdot (-2) + (1, 0) \cdot (0, -1) = 0$. Scaled by $s = 3^{-1/2} = 0.577350$, the softmax gives $p = (0.640457, 0.359543)$.
- The output is $3 \cdot 0.640457 - 3 \cdot 0.359543 = 0.842745$, so $(0.842745, 0)$.

Now the absorbed form. $W_{UK} = 2$, so $q^{lat} = 0$ and the scores are the rope terms alone, the same two numbers. The weighted latent is $0.640457 - 0.359543 = 0.280915$, and $W_O W_{UV} = 1 \cdot 3 = 3$, which again gives $0.842745$. The cache holds latents $(1, -1)$ and rope keys $(1, 1)$, $(0, -1)$: three numbers per token.

## The interface

```python
class LatentCacheHook(Protocol):
    def update(self, layer: int, c_new: NDArray, k_rope_new: NDArray) -> tuple[NDArray, NDArray]: ...
class ConcatLatentCache:                                    # update, seq_len
class MLAttention(Module):
    def __init__(self, d, n_heads, q_lora_rank, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_dim, rope: RopeSpec,
                 attention_bias=False, softmax_scale=None, rng=None)
    def forward(self, x, positions, mask=None, cache=None, layer=0) -> Tensor   # [B, T, d] -> [B, T, d]
    def absorb_weights(self) -> AbsorbedMLA
class AbsorbedMLA:                                          # w_uk [H, dn, r], w_ov [H, d, r], o_bias
    def forward(self, x, positions, cache, layer=0, mask=None) -> NDArray
```

`MLAttention` is the trainable naive form; `absorb_weights` folds its trained weights into an `AbsorbedMLA`, a numpy-only decoder that always runs against a cache. The latent cache hook mirrors the KV cache hook but carries a latent and a rope key instead of keys and values.

## Building it step by step

The cache must hold only what cannot be recomputed cheaply and is shared by all heads: the normalized latent $c_t$ ($r$ numbers) and the rotated rope key $k^R_t$ ($d_r$ numbers). Everything per head is a linear function of $c_t$ and can be rebuilt.

One projection, `kv_a_proj_with_mqa`, produces both: its first $r$ outputs are the raw latent and the last $d_r$ the rope key. The norm is applied to the latent only and the rotation to the rope key only, both **before** caching. The query is `[nope | rope]` per head, nope first, from either one projection or a low-rank bottleneck `q_a_proj -> q_a_layernorm -> q_b_proj`. Its rope part rotates with the same tables and layout as the rope key, so their dot product depends only on the offset.

The naive form follows directly: rebuild `[k_nope | v]` per head from all cached latents with `kv_b_proj`, add the nope score and the rope score (one rope key broadcast to every head), scale by $(d_n + d_r)^{-1/2}$, mask, softmax, average the values, and project. Built from the op library, it trains. Linearity then gives the absorbed form: fold $W_{UK}^h$ into the query and $W_O^h W_{UV}^h$ into a per-head output matrix, computed once from the trained weights, and decoding never builds per-head keys or values.

Both forms must share every piece that is not part of the absorption (query, latent, tables, visibility), so the absorbed class calls the naive layer's helpers rather than copying them. Then a difference between the two forms can only come from the absorption itself.

### The latent cache

The latent cache is the `ConcatKVCache` of the previous section with two arrays per layer, `c [B, T, r]` and `k_rope [B, T, dr]`, appended along time (axis 1) and returned whole. It has no head axis, and that is the saving.

### Registration order

```python
        dqk = qk_nope_dim + qk_rope_dim
        if softmax_scale is None:
            softmax_scale = dqk**-0.5  # the full query width, nope AND rope
        # ...
        # Registration order = HF DeepseekV3Attention = the state_dict keys.
        if q_lora_rank is None:
            self.q_proj = Linear(d, n_heads * dqk, bias=False, rng=r)
            self.q_a_proj = self.q_a_layernorm = self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = Linear(d, q_lora_rank, bias=attention_bias, rng=r)
            self.q_a_layernorm = RMSNorm(q_lora_rank, eps=LATENT_EPS)
            self.q_b_proj = Linear(q_lora_rank, n_heads * dqk, bias=False, rng=r)
        # first r outputs: raw latent; last dr: the shared rope key (the "mqa" part)
        self.kv_a_proj_with_mqa = Linear(d, kv_lora_rank + qk_rope_dim, bias=attention_bias, rng=r)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=LATENT_EPS)
        # per head: dn rows of k_nope, then dv rows of v
        self.kv_b_proj = Linear(kv_lora_rank, n_heads * (qk_nope_dim + v_dim), bias=False, rng=r)
        self.o_proj = Linear(n_heads * v_dim, d, bias=attention_bias, rng=r)
```

The constructor also insists `rope.rotary_dim == qk_rope_dim`: only the rope part rotates, and all of it does. `LATENT_EPS = 1e-6` is Hugging Face's epsilon for both latent norms.

### Query and latent

```python
    def _query(self, x):
        """[B, T, d] -> [B, H, T, dn + dr]."""
        B, T, _ = x.shape
        if self.q_proj is not None:
            q = self.q_proj(x)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))   # low-rank query
        return F.transpose(F.reshape(q, (B, T, self.n_heads, self.qk_nope_dim + self.qk_rope_dim)), 1, 2)

    def _latent(self, x, cos, sin):
        """The normalized latent c [B, T, r] and the rotated rope key [B, T, dr]."""
        ckv = self.kv_a_proj_with_mqa(x)
        r = self.kv_lora_rank
        c = self.kv_a_layernorm(ckv[..., :r])          # norm BEFORE caching
        k_rope = apply_rope(ckv[..., r:], cos, sin, self.rope.layout, self.qk_rope_dim)  # rotate BEFORE caching
        return c, k_rope
```

`_tables` builds one set of cos and sin tables and returns it twice: `[B, T, r/2]` for the head-less rope key and, for per-row positions, `[B, 1, T, r/2]` for the `[B, H, T, dr]` query.

### The naive forward

```python
        kcos, ksin, qcos, qsin = self._tables(pos)
        q = self._query(x)
        q_nope = q[..., :dn]                                         # nope first
        q_rope = apply_rope(q[..., dn:], qcos, qsin, self.rope.layout, self.qk_rope_dim)
        c, k_rope = self._latent(x, kcos, ksin)
        if cache is not None:
            # The cache holds the latent and the rotated rope key: r + dr numbers
            # per token, whatever the number of heads.
            ca, ka = cache.update(layer, c.data, k_rope.data)
            c, k_rope = Tensor(ca, dtype=x.dtype), Tensor(ka, dtype=x.dtype)
        Tk = c.shape[1]
        # rebuild every head's [k_nope | v] from every cached latent
        kv = F.transpose(F.reshape(self.kv_b_proj(c), (B, Tk, H, dn + dv)), 1, 2)  # [B, H, Tk, dn + dv]
        k_nope, v = kv[..., :dn], kv[..., dn:]
        k_r = F.reshape(k_rope, (B, 1, Tk, self.qk_rope_dim))  # one rope key, every head
        scores = F.matmul(q_nope, F.transpose(k_nope, 2, 3)) + F.matmul(q_rope, F.transpose(k_r, 2, 3))
        scores = F.masked_fill(scores * self.softmax_scale, ~_visible(T, Tk, mask, B, H), -np.inf)
        o = F.matmul(F.softmax(scores, axis=-1), v)  # [B, H, T, dv]
        return self.o_proj(F.reshape(F.transpose(o, 1, 2), (B, T, H * dv)))
```

Trace the hand example. For token 1 the nope query is 0, so the nope scores vanish; the rope scores are $(1, 0)\cdot(1,1) = 1$ and $(1,0)\cdot(0,-1) = 0$, scaled by $3^{-1/2}$ and softmaxed to $(0.640457, 0.359543)$; and the values $3c = (3, -3)$ average to $0.842745$. The two score terms are summed, not concatenated, which is the same as one dot product of `[q_nope | q_rope]` with `[k_nope | k_rope]`. That is why the scale uses the full width $d_n + d_r$.

### Absorbing the weights

Absorption is done once, in float64, after training:

```python
class AbsorbedMLA:
    def __init__(self, mla):
        H, dn, dv, r = mla.n_heads, mla.qk_nope_dim, mla.v_dim, mla.kv_lora_rank
        self.mla = mla
        # kv_b_proj rows are [head 0: dn k_nope rows, dv v rows][head 1: ...]
        w = mla.kv_b_proj.weight.data.astype(np.float64).reshape(H, dn + dv, r)
        self.w_uk = np.array(w[:, :dn, :])  # [H, dn, r]: k_nope_h = W_UK_h c
        w_uv = w[:, dn:, :]  # [H, dv, r]:  v_h = W_UV_h c
        # o_proj columns are [head 0: dv][head 1: dv]...: split them per head
        w_o = mla.o_proj.weight.data.astype(np.float64).reshape(mla.d, H, dv)  # [d, H, dv]
        self.w_ov = np.einsum("dhv,hvr->hdr", w_o, w_uv)  # [H, d, r] = W_O_h W_UV_h
        b = mla.o_proj.bias
        self.o_bias = None if b is None else np.array(b.data, dtype=np.float64)
```

The `einsum` contracts over $d_v$ separately for each head `h`, so head $h$'s output columns are paired with head $h$'s value rows; contracting over the head axis too would mix heads.

### The absorbed forward

```python
        kcos, ksin, qcos, qsin = m._tables(pos)       # shared helpers
        q = m._query(xt)
        q_nope = q.data[..., : m.qk_nope_dim]
        q_rope = apply_rope(q[..., m.qk_nope_dim :], qcos, qsin, m.rope.layout, m.qk_rope_dim).data
        c, k_rope = m._latent(xt, kcos, ksin)
        ca, ka = cache.update(layer, c.data, k_rope.data)
        ca, ka = np.asarray(ca, dtype=dt), np.asarray(ka, dtype=dt)
        Tk = ca.shape[1]
        # Query absorption: q_nope . (W_UK c) = (q_nope W_UK) . c, one latent per key.
        q_lat = np.einsum("bhtn,hnr->bhtr", q_nope, self.w_uk.astype(dt))
        # nope scores against cached latents + rope scores against cached rope keys
        scores = np.einsum("bhtr,bkr->bhtk", q_lat, ca) + np.einsum("bhtn,bkn->bhtk", q_rope, ka)
        scores = scores * dt.type(m.softmax_scale)
        vis = _visible(T, Tk, mask, B, m.n_heads)     # the padding mask applies here too
        scores = np.where(vis, scores, -np.inf)
        # a masked softmax by hand: subtract the row max, zero for an empty row
        mx = np.max(scores, axis=-1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0)
        e = np.where(vis, np.exp(scores - mx), 0)
        s = e.sum(axis=-1, keepdims=True)
        p = np.divide(e, s, out=np.zeros_like(e), where=s > 0)
        # Output absorption: W_O_h W_UV_h applied once to the weighted latent.
        o_lat = np.einsum("bhtk,bkr->bhtr", p, ca)
        out = np.einsum("bhtr,hdr->btd", o_lat, self.w_ov.astype(dt))   # sums over heads
        if self.o_bias is not None:
            out = out + self.o_bias.astype(dt)
        return out.astype(dt)
```

On the hand example, $W_{UK} = 2$ and $q^N = 0$ give $q^{lat} = 0$, so the scores are the rope terms alone, as before. The weighted latent is $0.640457 \cdot 1 + 0.359543 \cdot (-1) = 0.280915$, and $W_O W_{UV} = 3$ maps it to $0.842745$, the naive answer.

The absorbed form is plain numpy, not autograd: it is an inference path, and a training step through it would need gradients for `w_ov`, which is not a parameter. The full solution is in [`mla.py`](../../../../course/solutions/python/tinyllm/modern/mla.py).

### Why this design

The cache holds $r + d_r$ numbers per token per layer, independent of $H$. For DeepSeek-V3 that is $512 + 64 = 576$, against $2 \cdot 128 \cdot 128 = 32768$ for full MHA at $d_h = 128$.

Decode cost is where absorption pays. The naive form rebuilds $H(d_n + d_v)$ numbers for every cached token at every step, $O(T_k \cdot r \cdot H(d_n + d_v))$. The absorbed form costs $O(H \cdot r \cdot (d_n + T_k + d))$ per new token for the query map, the latent scores, and the output map: the per-key work is one $r$-wide dot product per head.

The naive form stays because training needs gradients through `kv_b_proj`, while the absorbed matrices are products of parameters. Prefill is also matmul-friendly in the naive form. So the solution trains and prefills naively and decodes absorbed, which is what DeepSeek's serving stacks do.

Why a decoupled rope key rather than RoPE on the latent? Rotating $k^N$ would make $W_{UK}^h$ position-dependent and break absorption. A separate small shared rope key keeps absorption exact at a cost of $d_r$ cached numbers. And `w_ov` is a product of two trained matrices, so forming it in float64 keeps the absorbed decode within float tolerance of the naive one.

The tradeoff you accept is that the absorbed forward reimplements the masked softmax in numpy instead of reusing `F.softmax`, because it runs outside autograd. The visibility mask is still the shared `_visible`, so masking cannot diverge.

## What can go wrong

The checks start with the hand example: outputs, weights, cache contents, and the absorbed form must all match, so you and the solution agree on every split. The state-dict keys, order, and shapes must be Hugging Face's so DeepSeek checkpoints load by name. Goldens from `DeepseekV3Attention` compare outputs and cached latents for plain and low-rank queries. The absorbed decode must equal the naive one in float64 after a prefill, since cached decoding uses the cheap form, and chunked prefill through the cache must equal the full forward, which [Chunked prefill](../p10-serving/03-chunked-prefill.md) relies on. The cached arrays must be exactly `kv_bytes_per_token` for `mla`, which admission by KV bytes and the capstone's equal-bytes ablation depend on. Shifting every position must change nothing, which confirms the decoupled rope key is real RoPE. Padded keys must be hidden in both forms and an empty row must give 0, for batched prompts of unequal length. A float64 central-difference gradcheck through the latent covers training. `ConcatLatentCache` must append, copy, report `seq_len`, and check chunks, and bad rope widths, sizes, scales, positions, or mask shapes must raise.

[`test_mla.py`](../../../../course/solutions/learner-tests/L7.6/python/tests/l7-6-mla/test_mla.py) writes MLA out in numpy float64 from the equations above (RoPE as complex multiplication) and compares it with `MLAttention` on random weights for both query forms and both layouts. It also checks the absorbed form against the naive one while decoding, a padding mask in the absorbed form, a central-difference gradient through `kv_a_proj_with_mqa`, the cache's copy, the rope-width check, and the state-dict names. The naive-versus-absorbed comparison is the strongest single check: the two forms share no score or output arithmetic, so agreement means both the factorization and the absorption are right.

> **Warning:** Reading `kv_b_proj` per head as v first, then k_nope, lets a DeepSeek checkpoint load and attend to garbage. The layout is nope first in both the query and `kv_b_proj`; the solution reshapes to `[.., H, dn + dv]` and slices `[:dn]` then `[dn:]` in both forms.

The full list:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| reading `kv_b_proj` per head as v first, then k_nope | a DeepSeek checkpoint loads and attends to garbage | reshapes to `[.., H, dn + dv]` and slices `[:dn]` then `[dn:]`, in both forms |
| splitting the query rope-first | scores mix position-free and rotated entries | `q[..., :dn]` is nope, `q[..., dn:]` rope |
| forgetting to rotate the shared rope key | relative position is lost; outputs drift with absolute position | `_latent` rotates `ckv[..., r:]` before returning it |
| caching the latent before `kv_a_layernorm` | the cached keys and values are scaled wrong | `_latent` normalizes before `cache.update` |
| scaling scores by $d_n^{-1/2}$ instead of $(d_n + d_r)^{-1/2}$ | softmax too sharp; Hugging Face disagrees | default `softmax_scale = (dn + dr) ** -0.5`; a non-positive scale raises |
| the rope key in a different layout from the query | relative scores break only for interleaved checkpoints | both rotations use `self.rope.layout` |
| absorbing with the value rows of `kv_b_proj` | the absorbed decode diverges from the naive one | `w_uk = w[:, :dn, :]`, `w_uv = w[:, dn:, :]` |
| a cache that returns only the new chunk | each decode step forgets the past | `ConcatLatentCache.update` returns the concatenation of everything held |
| the absorbed form dropping the padding mask | padded batches decode wrong only in production | both forms call `_visible(T, Tk, mask, ...)` |
| pairing `o_proj` columns with the wrong head in $W_O W_{UV}$ | absorbed outputs wrong for $H > 1$ only | `einsum("dhv,hvr->hdr")` contracts over $d_v$ per head |
| a detached latent | `kv_a_proj_with_mqa` never learns | the naive form uses the `c` tensor from `_latent` directly; only a cache turns it into a constant |

## Where this leads

In [the Llama-family model](09-llama-family-model.md), `tl_attention = "mla"` builds this layer in every block, so the decoder loads and matches Hugging Face checkpoints. The `LatentCache` of [KV cache and generate](../p08-inference/02-kv-cache-and-generate.md) implements the hook and decodes with `AbsorbedMLA`, and the [TinyStories capstone](../capstones/01-tinystories.md) runs the MLA-versus-GQA ablation at equal KV bytes.

### Further reading

- DeepSeek-AI, ["DeepSeek-V2"](https://arxiv.org/abs/2405.04434) (2024), section 2.1.
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024), section 2.1.1.
- Hugging Face `DeepseekV3Attention` in `transformers/models/deepseek_v3/modeling_deepseek_v3.py`: YaRN `mscale` folded into the softmax scale, and interleaved rope weights.
- vLLM's MLA backend (`vllm/attention/backends/mla/`) and DeepSeek FlashMLA: absorbed decode kernels over a paged latent cache.
- SGLang `MLATokenToKVPool` in `mem_cache/memory_pool.py`: one latent pool per layer with radix-tree prefix sharing.
- TransMLA and MHA2MLA (Meng et al. 2025, Ji et al. 2025): converting a trained GQA model to MLA with an SVD and fine-tuning.

## Key terms

- **Multi-head latent attention (MLA)**: attention whose keys and values are rebuilt per head from a cached low-rank latent of each token.
- **Latent** ($c_t$): the normalized $r$-dimensional summary of a token that the cache holds.
- **Decoupled rope key**: a small rotated key shared by all heads, kept separate so the latent part stays position-free.
- **Weight absorption**: folding $W_{UK}^h$ into the query and $W_O^h W_{UV}^h$ into the output so decoding never builds per-head keys or values.
- **Nope part**: the position-free part of each query and key, the one that can be absorbed.

## Summary

- MLA caches one latent vector $c$ of width $r$ plus one shared rope key of width $d_r$ per token, whatever the number of heads: $r + d_r$ numbers, which is `kv_bytes_per_token` for `mla`.
- Every head's key and value are linear in $c$, so a decode step folds $W_{UK}$ into the query and $W_{UV}$ into `o_proj` and never rebuilds per-head keys.
- RoPE cannot be absorbed (it depends on position), so MLA splits each query and key into a position-free nope part and a small rotated part.
- The cached latent is the normalized one, and every projection uses Hugging Face's layout, so DeepSeek checkpoints load by name.
- The naive form trains and prefills; the absorbed form decodes; they share every helper so they can only differ in the absorption.

## Review questions

1. Why must the latent be cached after `kv_a_layernorm`, not before?
   <details><summary>Answer</summary>Keys and values are `kv_b_proj` applied to the normalized latent. Caching the raw latent would either force re-normalizing at every step or, if forgotten, produce keys and values scaled wrongly. Caching the normalized one makes the cached vector exactly what `kv_b_proj` consumes.</details>
2. Why can the nope part be absorbed but the rope part cannot?
   <details><summary>Answer</summary>The nope score is $q^\top W_{UK} c$, bilinear with a fixed matrix, so $W_{UK}$ can move onto the query. The rope score involves $R_t^\top R_j$, which depends on the offset $j - t$, so no single fixed matrix can be folded in.</details>
3. Why is the softmax scale $(d_n + d_r)^{-1/2}$?
   <details><summary>Answer</summary>The score is one dot product over the concatenated `[nope | rope]` query and key, of width $d_n + d_r$; the variance argument for $1/\sqrt{d}$ applies to the full width, and it is what Hugging Face uses.</details>
4. What would go wrong if `w_ov` were computed by contracting over heads as well?
   <details><summary>Answer</summary>Head $h$'s output columns of `o_proj` would be multiplied with other heads' value rows, mixing heads. The absorbed output would agree with the naive one only when $H = 1$.</details>
5. Why do `MLAttention` and `AbsorbedMLA` share `_query`, `_latent`, `_tables`, and `_visible`?
   <details><summary>Answer</summary>So the only difference between the two forms is the absorption. Any shared bug shows in both, and any disagreement between them points straight at the folded matrices.</details>
6. How many numbers per token per layer does DeepSeek-V3's MLA cache, and how does that compare with full MHA?
   <details><summary>Answer</summary>$r + d_r = 512 + 64 = 576$, against $2 \cdot 128 \cdot 128 = 32768$ for full MHA with 128 heads at $d_h = 128$, roughly 57 times less.</details>
