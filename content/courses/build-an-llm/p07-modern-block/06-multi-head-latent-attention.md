# Multi-head latent attention (DeepSeek-V2/V3), weight absorption

## Overview

| | |
|---|---|
| **Module** | `L7.6` · build · Python · Pass 5 · 2 h reading |
| **Solution** | [`mla.py`](../../../../course/solutions/python/tinyllm/modern/mla.py): `MLAttention`, `ConcatLatentCache`, `AbsorbedMLA`; model oracle tests in [`test_mla.py`](../../../../course/solutions/learner-tests/L7.6/python/tests/l7-6-mla/test_mla.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/mla.pyi`](../../../../course/contracts/py/tinyllm/modern/mla.pyi) |
| **Needs** | `L7.3` `apply_rope`, `rope_cos_sin`, `RopeSpec` · `L7.1` `RMSNorm` · `L0.4` `Linear`, `Module` · `L0.2` the op library · `L0.1` `Tensor` · `M06.3` `PCG32` · `M05.1` `kv_bytes_per_token` (what the cache must hold) · reading: `M03.5` low rank, `L7.5` the cache hook, `M09.2` |
| **Used by** | `L7.9` builds it for `tl_attention = "mla"`, so the decoder loads and matches Hugging Face checkpoints · later: `L8.2` `LatentCache`, C1's MLA-vs-GQA ablation at equal KV bytes |
| **Optional depth** | DeepSeek-AI, "DeepSeek-V2" (2024), section 2.1; "DeepSeek-V3 Technical Report" (2024), section 2.1.1 |

## Key Takeaways

- MLA caches one latent vector $c$ of width $r$ plus one shared rope key of width $d_r$ per token, whatever the number of heads: $r + d_r$ numbers, M05.1's `kv_bytes_per_token` for `mla`.
- Every head's key and value are linear in $c$, so a decode step folds $W_{UK}$ into the query and $W_{UV}$ into `o_proj` and never rebuilds per-head keys.
- RoPE cannot be absorbed (it depends on position), so MLA splits each query and key into a position-free "nope" part and a small rotated part.
- The cached latent is the normalized one, and the layout of every projection is HF's, so DeepSeek checkpoints load by name.

## How to read this chapter

Sections 1 to 3 build the latent factorization, the decoupled rope key, and weight absorption from first principles and work a one-head layer by hand in both forms. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution, the naive form first and then the absorbed one, and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The GQA layer (`L7.5`) already shrank SmolLM2's cache by three: 3 kv heads instead of 9. The cache still grows by $2 H_{kv} d_h$ numbers per token per layer, and at serving time (`L10.2` admits requests by KV bytes) that is what limits how many conversations fit in memory. DeepSeek-V2 cut it much further with a different idea: cache a low-rank summary of the token and rebuild keys and values from it. Built naively that costs compute at every step; built with weight absorption it costs almost nothing. This module builds both forms and proves they agree, so C1 can compare MLA and GQA at equal KV bytes.

## 2. Principles

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
| $R_p$ | the RoPE rotation at position $p$ (`L7.3`) | $d_r \times d_r$ |
| $s$ | `softmax_scale` $= (d_n + d_r)^{-1/2}$ | `float` |

### 2.1 Low rank, again

`M03.5` showed that a matrix of rank $r$ factors as a product of an $m \times r$ and an $r \times n$ matrix, and that truncating the SVD keeps the best rank-$r$ approximation. MLA bets that the keys and values of all heads, stacked as one $H(d_n + d_v)$-wide vector per token, live near an $r$-dimensional subspace. It learns the factorization directly: a down projection to $c_t$ (`kv_a_proj_with_mqa`, the first $r$ outputs), an RMSNorm (`kv_a_layernorm`, `L7.1`), and an up projection `kv_b_proj` of shape $[H(d_n + d_v), r]$. SmolLM2-sized MHA caches $2 H d_h$ numbers per token; DeepSeek-V3 uses $r = 512$ and $d_r = 64$ for 128 heads.

### 2.2 The latent and the decoupled rope key

Steps of the naive form, for each token:

$$c_t = \mathrm{RMSNorm}(W_{DKV} x_t), \qquad k^{N,h}_t = W_{UK}^h c_t, \qquad v^h_t = W_{UV}^h c_t,$$
$$k^R_t = R_t\, W_{KR} x_t, \qquad q^h_t = [\,q^{N,h}_t \mid R_t\, q^{R,h}_t\,],$$
$$\mathrm{score}^h(t, j) = s\,\big(q^{N,h}_t \cdot k^{N,h}_j + (R_t q^{R,h}_t) \cdot k^R_j\big).$$

Each head's query is `[nope | rope]` with the nope part first, and `kv_b_proj` splits per head into `[k_nope | v]`, nope first. Positions enter only through $R$, so a shift of every position changes nothing (`L7.3`'s law). The rope key is one vector shared by all heads, like MQA, which keeps it to $d_r$ cached numbers.

### 2.3 What the cache holds

The cache receives $c_t$ after the norm and $k^R_t$ after the rotation, $r + d_r$ numbers per token per layer: `ConcatLatentCache.update(layer, c, k_rope)` appends a chunk and returns everything held. A query of a chunk of $T$ new tokens sits at key index $T_k - T + t$, exactly as in `L7.5`, so prefilling in chunks equals one full forward.

### 2.4 Weight absorption

Rewrite the nope score and the output with matrix products moved around:

$$q^{N,h}_t \cdot (W_{UK}^h c_j) = \big((W_{UK}^h)^\top q^{N,h}_t\big) \cdot c_j, \qquad W_O^h \sum_j p_j W_{UV}^h c_j = (W_O^h W_{UV}^h) \sum_j p_j c_j .$$

The query is mapped into latent space once ($q^{lat,h}_t = (W_{UK}^h)^\top q^{N,h}_t$, an $r$-vector), the scores are dot products with the cached $c_j$, the weighted latent $\sum_j p_j c_j$ is formed once per head, and $W_O^h W_{UV}^h$ (precomputed, $d \times r$) maps it out. No per-head key or value is ever built: decoding reads $r + d_r$ numbers per cached token. RoPE's part cannot be absorbed (it depends on $t - j$), which is the reason for the decoupled rope key.

### 2.5 Training the naive form

Absorption is an inference rewrite. Training uses the naive form built from the op library, so gradients reach every projection through the latent, its norm, and the rope key.

## 3. Worked example by hand

$d = 2$, one head, $r = 1$, $d_n = 1$, $d_r = 2$, $d_v = 1$, `inv_freq` $= (\pi/2)$, so position 1 rotates the rope pair by 90 degrees: $(a, b) \mapsto (-b, a)$. Weights: $q = [x_0 \mid x_0, x_1]$; $c_{raw} = x_0 + x_1$ and $k^R = (x_1, x_0)$; `kv_b_proj` gives $k^N = 2c$, $v = 3c$; `o_proj` outputs $(o, 0)$. The latent norm of a single number is $c / |c| = \pm 1$ (the $\epsilon = 10^{-6}$ changes the 7th digit).

- Token 0, $x = (1, 1)$, position 0: $c_{raw} = 2$, so $c = 1$; $k^R = (1, 1)$, not rotated; $v = 3$. It sees only itself: output $(3, 0)$.
- Token 1, $x = (0, -1)$, position 1: $c_{raw} = -1$, so $c = -1$, $v = -3$; $k^R = (-1, 0)$ rotated to $(0, -1)$. Query $q^N = 0$, $q^R = (0, -1)$ rotated to $(1, 0)$.
- Scores of token 1: key 0 is $0 \cdot 2 + (1, 0)\cdot(1, 1) = 1$; key 1 is $0 \cdot (-2) + (1, 0) \cdot (0, -1) = 0$. Scaled by $s = 3^{-1/2} = 0.577350$: softmax gives $p = (0.640457, 0.359543)$.
- Output: $3 \cdot 0.640457 - 3 \cdot 0.359543 = 0.842745$, so $(0.842745, 0)$.

Absorbed: $W_{UK} = 2$, so $q^{lat} = 0$ and the scores are the rope terms alone, the same two numbers. The weighted latent is $0.640457 - 0.359543 = 0.280915$ and $W_O W_{UV} = 1 \cdot 3 = 3$: again $0.842745$. The cache holds latents $(1, -1)$ and rope keys $(1, 1)$, $(0, -1)$: three numbers per token.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: outputs, weights, cache contents, absorbed form | the solution and the reader agree on every split |
| state-dict keys and shapes are HF's | unit | HF's parameter names, order, shapes | DeepSeek checkpoints load by name in `L7.9` |
| `DeepseekV3Attention` goldens | golden | outputs and cached latents, plain and low-rank queries | the `L7.9` model and C1 |
| absorbed decode equals naive | differential | prefill then decode, absorbed vs naive, float64 | `L8.2` decodes with the cheap form |
| cached chunks equal the full forward | differential | chunked prefill through the cache | chunked prefill in `L10.3` |
| cache bytes match `M05.1` | property | cached arrays are `kv_bytes_per_token` for `mla` | admission by KV bytes, C1's equal-bytes ablation |
| scores see relative position only | property | shifting every position changes nothing | the decoupled rope key is RoPE |
| padding mask and fully masked rows | boundary | padded keys hidden in both forms; an empty row gives 0 | batched prompts of unequal length |
| gradients through the latent | gradcheck | float64 central differences | C1 trains MLA |
| `ConcatLatentCache` | unit | append, copy, `seq_len`, chunk checks | the hook `L8.2` implements |
| validation | boundary | rope width, sizes, scale, positions, mask shape | wiring bugs fail loudly |

### A model oracle test suite

[`test_mla.py`](../../../../course/solutions/learner-tests/L7.6/python/tests/l7-6-mla/test_mla.py) writes MLA out in numpy float64 from the equations of section 2 (RoPE as complex multiplication) and compares it with `MLAttention` on random weights for both query forms and both layouts; it also checks the absorbed form against the naive one while decoding, a padding mask in the absorbed form, a central-difference gradient through `kv_a_proj_with_mqa`, the cache's copy, the rope-width check, and the state-dict names. The naive-vs-absorbed comparison is the strongest single check: the two forms share no score or output arithmetic, so agreement means both the factorization and the absorption are right.

## 5. The solution, step by step

### 5.1 From first principles

1. The cache must hold only what cannot be recomputed cheaply and is shared by all heads: the normalized latent $c_t$ ($r$ numbers) and the rotated rope key $k^R_t$ ($d_r$ numbers). Everything per head is a linear function of $c_t$ and can be rebuilt.
2. One projection, `kv_a_proj_with_mqa`, produces both: its first $r$ outputs are the raw latent, the last $d_r$ the rope key. The norm is applied to the latent only, and the rotation to the rope key only, both **before** caching (section 2.3).
3. The query is `[nope | rope]` per head, nope first, from either one projection or a low-rank bottleneck `q_a_proj -> q_a_layernorm -> q_b_proj`. The rope part rotates with the same tables and layout as the rope key, so their dot product depends only on the offset.
4. The naive form follows directly: rebuild `[k_nope | v]` per head from all cached latents with `kv_b_proj`, add the nope score and the rope score (one rope key broadcast to every head), scale by $(d_n + d_r)^{-1/2}$, mask, softmax, average the values, project. Built from `L0.2` ops, it trains.
5. Linearity gives the absorbed form (section 2.4): fold $W_{UK}^h$ into the query and $W_O^h W_{UV}^h$ into a per-head output matrix, computed once from the trained weights. Decoding then never builds per-head keys or values.
6. Both forms must share every piece that is not part of the absorption (query, latent, tables, visibility), so the absorbed class calls the naive layer's helpers rather than copying them. Then a difference between the two forms can only come from the absorption itself.

### 5.2 The code, annotated

**The latent cache** is `L7.5`'s `ConcatKVCache` with two arrays per layer, `c [B, T, r]` and `k_rope [B, T, dr]`, appended along time (axis 1) and returned whole. It has no head axis: that is the saving.

**Registration order.**

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

The constructor also insists `rope.rotary_dim == qk_rope_dim`: only the rope part rotates, and all of it does. `LATENT_EPS = 1e-6` is HF's epsilon for both latent norms.

**Query and latent.**

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

`_tables` builds one set of cos/sin and returns it twice: `[B, T, r/2]` for the head-less rope key and, for per-row positions, `[B, 1, T, r/2]` for the `[B, H, T, dr]` query.

**The naive forward.**

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

On section 3: for token 1 the nope query is 0, so the nope scores vanish; the rope scores are $(1, 0)\cdot(1,1) = 1$ and $(1,0)\cdot(0,-1) = 0$, scaled by $3^{-1/2}$ and softmaxed to $(0.640457, 0.359543)$; the values $3c = (3, -3)$ average to $0.842745$. The two score terms are summed, not concatenated, which is the same as one dot product of `[q_nope | q_rope]` with `[k_nope | k_rope]`: that is why the scale uses the full width $d_n + d_r$.

**Absorbing the weights.** Done once, in float64, after training.

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

**The absorbed forward.**

```python
        kcos, ksin, qcos, qsin = m._tables(pos)       # shared helpers: step 6
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

On section 3: $W_{UK} = 2$ and $q^N = 0$ give $q^{lat} = 0$, so the scores are the rope terms alone, as before. The weighted latent is $0.640457 \cdot 1 + 0.359543 \cdot (-1) = 0.280915$, and $W_O W_{UV} = 3$ maps it to $0.842745$, the naive answer.

The absorbed form is plain numpy, not autograd: it is an inference path, and a training step through it would need gradients for `w_ov`, which is not a parameter.

Full solution: [`mla.py`](../../../../course/solutions/python/tinyllm/modern/mla.py).

### 5.3 Why this solution

- **Cache size.** $r + d_r$ numbers per token per layer, independent of $H$: for DeepSeek-V3, $512 + 64 = 576$ against $2 \cdot 128 \cdot 128 = 32768$ for full MHA at $d_h = 128$.
- **Decode cost.** The naive form rebuilds $H(d_n + d_v)$ numbers for every cached token at every step: $O(T_k \cdot r \cdot H(d_n + d_v))$. The absorbed form costs $O(H \cdot r \cdot (d_n + T_k + d))$ per new token for the query map, the latent scores, and the output map: the per-key work is one $r$-wide dot product per head.
- **Why keep the naive form.** Training needs gradients through `kv_b_proj`; the absorbed matrices are products of parameters. Prefill is also matmul-friendly in the naive form. So the solution trains and prefills naively and decodes absorbed, which is what DeepSeek's serving stacks do.
- **Decoupled rope key, not RoPE on the latent.** Rotating $k^N$ would make $W_{UK}^h$ position-dependent and break absorption. A separate small shared rope key keeps absorption exact at a cost of $d_r$ cached numbers.
- **float64 absorption.** `w_ov` is a product of two trained matrices; forming it in float64 keeps the absorbed decode within float tolerance of the naive one.
- **Accepted tradeoff.** The absorbed forward reimplements the masked softmax in numpy instead of reusing `F.softmax`, because it runs outside autograd; the visibility mask is still the shared `_visible`, so masking cannot diverge.

### 5.4 Explain it back

1. Why must the latent be cached after `kv_a_layernorm`, not before?
   <details><summary>Answer</summary>Keys and values are `kv_b_proj` applied to the normalized latent. Caching the raw latent would either force re-normalizing at every step or, if forgotten, produce keys and values scaled wrongly. Caching the normalized one makes the cached vector exactly what `kv_b_proj` consumes.</details>
2. Why can the nope part be absorbed but the rope part cannot?
   <details><summary>Answer</summary>The nope score is $q^\top W_{UK} c$, bilinear with a fixed matrix, so $W_{UK}$ can move onto the query. The rope score involves $R_t^\top R_j$, which depends on the offset $j - t$, so no single fixed matrix can be folded in.</details>
3. Why is the softmax scale $(d_n + d_r)^{-1/2}$?
   <details><summary>Answer</summary>The score is one dot product over the concatenated `[nope | rope]` query and key, of width $d_n + d_r$; the variance argument for $1/\sqrt{d}$ applies to the full width, and it is what HF uses.</details>
4. What would go wrong if `w_ov` were computed by contracting over heads as well?
   <details><summary>Answer</summary>Head $h$'s output columns of `o_proj` would be multiplied with other heads' value rows, mixing heads. The absorbed output would agree with the naive one only when $H = 1$.</details>
5. Why do `MLAttention` and `AbsorbedMLA` share `_query`, `_latent`, `_tables`, and `_visible`?
   <details><summary>Answer</summary>So the only difference between the two forms is the absorption. Any shared bug shows in both, and any disagreement between them points straight at the folded matrices.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. reading `kv_b_proj` per head as v first, then k_nope | a DeepSeek checkpoint loads and attends to garbage | reshapes to `[.., H, dn + dv]` and slices `[:dn]` then `[dn:]`, in both forms |
| 2. splitting the query rope-first | scores mix position-free and rotated entries | `q[..., :dn]` is nope, `q[..., dn:]` rope (step 3) |
| 3. forgetting to rotate the shared rope key | relative position is lost; outputs drift with absolute position | `_latent` rotates `ckv[..., r:]` before returning it |
| 4. caching the latent before `kv_a_layernorm` | the cached keys and values are scaled wrong | `_latent` normalizes before `cache.update` (step 2) |
| 5. scaling scores by $d_n^{-1/2}$ instead of $(d_n + d_r)^{-1/2}$ | softmax too sharp; HF disagrees | default `softmax_scale = (dn + dr) ** -0.5`; a non-positive scale raises |
| the rope key in a different layout from the query | relative scores break only for interleaved checkpoints | both rotations use `self.rope.layout` |
| absorbing with the value rows of `kv_b_proj` | the absorbed decode diverges from the naive one | `w_uk = w[:, :dn, :]`, `w_uv = w[:, dn:, :]` |
| a cache that returns only the new chunk | each decode step forgets the past | `ConcatLatentCache.update` returns the concatenation of everything held |
| the absorbed form dropping the padding mask | padded batches decode wrong only in production | both forms call `_visible(T, Tk, mask, ...)` |
| pairing `o_proj` columns with the wrong head in $W_O W_{UV}$ | absorbed outputs wrong for $H > 1$ only | `einsum("dhv,hvr->hdr")` contracts over $d_v$ per head |
| a detached latent | `kv_a_proj_with_mqa` never learns | the naive form uses the `c` tensor from `_latent` directly; only a cache turns it into a constant |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L7.3` | rotates the decoupled rope parts with a `RopeSpec` |
| Back | `L7.1` | `RMSNorm` on the latent and the query bottleneck |
| Back | `L0.4` | `Linear` projections, `Module` registration |
| Back | `L0.2` | the ops that make the naive form trainable |
| Back | `L0.1` | `Tensor` slicing and products |
| Back | `M06.3` | the default initialization stream |
| Back | `M05.1` | `kv_bytes_per_token` for `mla`, checked against the cache |
| Forward | `L7.9` | `tl_attention = "mla"` builds this layer in every block |
| Forward | `L8.2` | `LatentCache` implements the hook and decodes with `AbsorbedMLA` |
| Forward | C1 | the MLA-vs-GQA ablation at equal KV bytes |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `MLAttention` | HF `DeepseekV3Attention` | YaRN `mscale` folded into the softmax scale, interleaved rope weights | `transformers/models/deepseek_v3/modeling_deepseek_v3.py` |
| `AbsorbedMLA` | vLLM MLA backend, FlashMLA | absorbed decode kernels over a paged latent cache | vLLM `vllm/attention/backends/mla/`, DeepSeek FlashMLA |
| `ConcatLatentCache` | SGLang `MLATokenToKVPool` | one latent pool per layer, radix-tree prefix sharing | SGLang `mem_cache/memory_pool.py` |
| the factorization | TransMLA, MHA2MLA | converting a trained GQA model to MLA with an SVD and fine-tuning | Meng et al. 2025, Ji et al. 2025 |
