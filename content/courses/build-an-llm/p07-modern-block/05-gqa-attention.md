# MQA/GQA attention with cache hook, window, learned sinks

## Overview

| | |
|---|---|
| **Module** | `L7.5` · build · Python · Pass 5 · 2 to 3 h reading |
| **Solution** | [`gqa.py`](../../../../course/solutions/python/tinyllm/modern/gqa.py): `GQAttention`, `repeat_kv`, `_visible`, the `KVCacheHook` protocol and its smallest implementation `ConcatKVCache`; model oracle tests in [`test_gqa.py`](../../../../course/solutions/learner-tests/L7.5/python/tests/l7-5-gqa/test_gqa.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/gqa.pyi`](../../../../course/contracts/py/tinyllm/modern/gqa.pyi) |
| **Needs** | `L7.3` `rope_cos_sin`, `apply_rope`, `RopeSpec` · `L0.4` `Linear`, `Module` · `L0.2` `F.matmul`, `F.masked_fill`, `F.softmax`, `F.concat` · `L0.1` `Tensor` · `M06.3` `PCG32` (default initialization) · `M05.1` `kv_bytes_per_token` (what the cache must hold) · reading: `M09.2` (the masked softmax), Part 5's scaled dot-product attention |
| **Used by** | `L7.7` sliding-window and sink attention builds on it · `L7.9` every Llama attention layer (SmolLM2: 9 query heads over 3 kv heads), on the way to SmolLM2-135M logits that match Hugging Face · `L8.2` the `KVCache` behind the hook · later: `L9.3` and `L9.4` the C attention kernels take `Hkv` |
| **Optional depth** | Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019); Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023); Beltagy et al., "Longformer" (2020), section 3.1; Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023); OpenAI, "gpt-oss-120b and gpt-oss-20b model card" (2025) |

## Key Takeaways

- $H$ query heads share $H_{kv}$ key/value heads; query head $h$ reads kv head $\lfloor h / n_\text{rep} \rfloor$, each kv head repeated $n_\text{rep}$ times **in a row**.
- The cache hook receives only the $H_{kv}$ heads, so the KV cache shrinks by $H / H_{kv}$: exactly `M05.1`'s `kv_bytes_per_token`.
- With a cache, the new queries are the **last** $T$ of $T_k$ keys: the causal mask uses that offset, and keys are rotated with their own positions before they are cached.
- A sliding window of $W$ lets query $t$ read keys $t - W + 1 .. t$.
- A learned sink is one extra logit per head in every softmax row, dropped afterwards: a head may attend to nothing.

## How to read this chapter

Sections 1 to 3 build grouped-query attention, the visibility rule, the cache hook, and learned sinks from first principles and work a two-head layer by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The 2017 multi-head attention of Part 5 gives every query head its own key and value head. At inference the engine caches keys and values for every past token (`L8.2`, and the Rust engine in Pass 7), so cache memory, not compute, limits how many requests run at once. Open SmolLM2-135M's checkpoint: `k_proj.weight` is $192 \times 576$, a third of `q_proj`'s $576 \times 576$. It has 9 query heads but only 3 key/value heads, and plain MHA cannot load it. Newer models add two more things to the same layer: a **sliding window** (Mistral, gpt-oss) so the cache stops growing, and **learned sinks** (gpt-oss) so a head can decline to attend. This module builds that attention layer, with the hook through which the KV cache will plug in.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the layer input | `float32[B, T, d]` |
| $H$, $H_{kv}$ | query heads, key/value heads | `int`, $H_{kv} \mid H$ |
| $n_\text{rep}$ | $H / H_{kv}$, query heads per kv head | `int` |
| $d_h$ | head width (`d_head`; not necessarily $d / H$) | `int` |
| $q$ | queries after RoPE | `float32[B, H, T, d_h]` |
| $k$, $v$ | keys (after RoPE) and values | `float32[B, H_{kv}, T_k, d_h]` |
| $T$, $T_k$ | new tokens in this call, keys visible (cached plus new) | `int` |
| $W$ | `window` | `int` or none |
| $\sigma_h$ | learned sink logit of head $h$ (`sinks`) | `float32[H]` |
| $a_{tj}$ | attention weight of query $t$ on key $j$ | `float` |

### 2.1 Grouped-query attention

Each head computes scaled dot-product scores $q_t \cdot k_j / \sqrt{d_h}$, a softmax over the visible keys, and a weighted average of the values; the heads are concatenated and projected back to $d$ by `o_proj`. RoPE (`L7.3`) rotates $q$ and $k$ (with `RopeSpec`'s frequencies, layout, rotary width, and attention scaling) before the scores, so a score depends on the relative position of query and key only. The scale is $1/\sqrt{d_h}$, the head width: a dot product of $d_h$ unit-variance products has variance $d_h$.

### 2.2 Sharing kv heads

Multi-query attention (Shazeer 2019) keeps one key/value head for all query heads ($H_{kv} = 1$); grouped-query attention (Ainslie et al.) keeps $H_{kv}$ of them, each serving a contiguous group of $n_\text{rep}$ query heads. `repeat_kv` expands $[B, H_{kv}, T, d_h]$ to $[B, H, T, d_h]$ with output head $h$ reading input head $\lfloor h / n_\text{rep} \rfloor$: $(a, b) \to (a, a, a, b, b, b)$, the order the checkpoints were trained with. Tiling, $(a, b, a, b, a, b)$, loads without error and is wrong. GQA is MHA whose kv heads come in identical groups, so an MHA layer with repeated kv weights gives the same output. Because indexing accumulates gradients, a kv head's gradient is the sum over its copies.

### 2.3 Who may see whom

Query $t$ of this call sits at key index $i = T_k - T + t$ (with no cache, $T_k = T$ and $i = t$). Key $j$ is visible when:

- $j \le i$: causal, always;
- $i - j < W$ when a window is set: the query and the $W - 1$ keys before it;
- the optional `mask` allows it (`True` = may attend, broadcast to $[B, H, T, T_k]$: padding).

Hidden scores are $-\infty$, so their weights are exactly 0, and a row with nothing visible gives zeros, not NaN (`M09.2`).

### 2.4 The cache hook

Decoding one token at a time, recomputing every past key and value is wasted work. With `cache` given, after RoPE the layer calls `cache.update(layer, k, v)` with this call's $[B, H_{kv}, T, d_h]$ keys and values (numpy arrays, after rotation, before repetition) and attends over everything the cache returns. Two consequences: the cache stores $H_{kv}$ heads, which is where GQA's saving happens (`M05.1`: $2 H_{kv} d_h$ numbers per token per layer); and keys must be rotated with **their own** positions before they are stored, because a key cached at position 5 must stay rotated by 5. `ConcatKVCache` is the smallest implementation (it concatenates along time); `L8.2`'s `KVCache` and `L8.3`'s paged cache implement the same `update`.

### 2.5 Learned sinks

A softmax must put weight 1 somewhere, so a head with nothing useful to read dumps it on some token (often the first: StreamingLLM's "attention sink"). gpt-oss gives each head a learned logit $\sigma_h$ appended to every row before the softmax and dropped afterwards:

$$a_{tj} = \frac{e^{s_{tj}}}{e^{\sigma_h} + \sum_{j'} e^{s_{tj'}}}.$$

The weights now sum to $1 - p_\text{sink} \le 1$: a large $\sigma_h$ means the head attends to almost nothing, a very negative one recovers plain attention. Adding $\sigma_h$ to every score instead does nothing at all, because softmax ignores a constant shift.

### 2.6 Backward

Every step is an op of `L0.2` or `L7.3`, so autograd reaches $x$, the four projections, their biases, and the sinks. With a cache, the cached keys and values are constants: the cache is for inference, and no gradient flows into it.

## 3. Worked example by hand

$d = 2$, $H = 2$, $H_{kv} = 1$, $d_h = 2$, RoPE off (`inv_freq = [0]`, every angle 0). Weights: `q_proj` makes head 0's query $x$ and head 1's $-x$; $k = x$; $v = (x_0, 2 x_1)$; `o_proj` keeps entry 0 of head 0 and entry 1 of head 1. Tokens $x^{(0)} = (1, 0)$, $x^{(1)} = (0, 1)$; both heads read the single kv head: $k^{(0)} = (1, 0)$, $k^{(1)} = (0, 1)$, $v^{(0)} = (1, 0)$, $v^{(1)} = (0, 2)$.

1. Token 0 sees only key 0: both heads output $v^{(0)} = (1, 0)$; the output is $(1, 0)$.
2. Token 1, head 0: $q = (0, 1)$, scores $(0, 1) / \sqrt 2 = (0, 0.707107)$; $e^{0.707107} = 2.028115$, weights $(1, 2.028115) / 3.028115 = (0.330237, 0.669763)$; head output $0.330237\, v^{(0)} + 0.669763\, v^{(1)} = (0.330237, 1.339527)$.
3. Token 1, head 1: $q = (0, -1)$, scores $(0, -0.707107)$, weights $(0.669763, 0.330237)$; head output $(0.669763, 0.660474)$.
4. `o_proj` keeps entry 0 of head 0 and entry 1 of head 1: $(0.330237, 0.660474)$.

One kv head served two query heads that attend in opposite directions.

## 4. The interface

```python
class KVCacheHook(Protocol):
    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]: ...
class ConcatKVCache:                     # update(...), seq_len(layer=0)
def repeat_kv(x: Tensor, n_rep: int) -> Tensor: ...                      # [B, Hkv, T, dh] -> [B, Hkv n_rep, T, dh]
class GQAttention(Module):
    def __init__(self, d, n_heads, n_kv_heads, d_head, rope: RopeSpec, qkv_bias=False, window=None, sinks=False, rng=None): ...
    def forward(self, x, positions, mask=None, cache=None, layer=0) -> Tensor: ...   # [B, T, d] -> [B, T, d]
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3 | the solution and the reader agree on the formula |
| Hugging Face goldens | golden | Llama GQA, an extra mask, Qwen2 biases, gpt-oss sinks and window with gapped per-row positions: outputs and every gradient | SmolLM2 in `L7.9` |
| gradients of every parameter | gradcheck | float64, biases, window, sinks | every weight trains |
| `repeat_kv` repeats each head in a row | unit | $(a, b) \to (a, a, a, b, b, b)$, summed gradients | checkpoints' head grouping |
| GQA equals MHA with repeated kv weights | differential | GQA = MHA with grouped kv weights | what GQA is |
| cached chunks equal the full forward | differential | prefill then decode through the cache, with and without window and sinks | `L8.2`'s cached generation |
| the cache holds kv heads only | property | the hook gets $H_{kv}$ heads, bytes = `kv_bytes_per_token` | `L10.2` admission by KV bytes |
| `ConcatKVCache` | unit | appends, copies, rejects bad chunks | the reference hook |
| causality, bit for bit | property | future tokens change nothing | a decoder |
| window reach | boundary | key $t - W$ invisible, $t - W + 1$ visible | Mistral-style windows, `L7.7` |
| sinks take weight from every key | property | large sink: output 0; very negative: plain attention | gpt-oss heads |
| a fully masked row is zero | boundary | zeros, no NaN, other rows intact | padded batches |
| shifted and per-row positions | property | shift invariance; per-row positions equal separate calls | left padding, continued caches |
| attention scaling squares into the scores | property | YaRN scaling = $s^2$ on the scores | `rope_scaling` in `L7.9` |
| parameter names, shapes, and draw order | unit | HF keys, no `o_proj` bias, `d_head` default, draw order | the safetensors keys |
| validation | boundary | heads, `d_head`, window, rotary width, input, positions, mask | wiring bugs fail loudly |

### A model oracle test suite

[`test_gqa.py`](../../../../course/solutions/learner-tests/L7.5/python/tests/l7-5-gqa/test_gqa.py) writes the whole layer out in numpy float64 with the module's own `state_dict()`: the projections, RoPE as complex multiplication, per head the scores of kv head $\lfloor h / n_\text{rep} \rfloor$, the causal and window mask, the optional sink column, the softmax, the values, and `o_proj`. It compares `forward` with and without window and sinks, decodes through `ConcatKVCache` against the full forward, and checks the head order of `repeat_kv`, the mask, per-row positions, the parameter names, the attention scaling, the cache's copies, and the validation. The oracle indexes kv head `h // n_rep` per query head directly instead of repeating, so it is an independent statement of the grouping.

## 5. The solution, step by step

### 5.1 From first principles

1. The checkpoint fixes the parameter shapes: `q_proj` is $d \to H d_h$, `k_proj` and `v_proj` are $d \to H_{kv} d_h$, `o_proj` is $H d_h \to d$ with no bias (section 2.2). $d_h$ is a separate argument because some models set it independently of $d / H$.
2. Attention needs one key per query head. The cheapest correct way to get there is to compute $H_{kv}$ keys and values and **index** them: query head $h$ reads kv head $\lfloor h / n_\text{rep} \rfloor$. Indexing gives the backward for free: copies of a kv head sum their gradients.
3. The cache must see the smallest thing that can be stored: keys after RoPE (so each keeps its own position, section 2.4) and before `repeat_kv` (so it holds $H_{kv}$ heads). That fixes where `cache.update` sits in the forward: after `apply_rope`, before `repeat_kv`.
4. With a cache, this call's $T$ queries are the last $T$ of $T_k$ keys. Every visibility rule (causal, window) is a comparison between absolute key indices $i = T_k - T + t$ and $j$, so one boolean `[T, Tk]` matrix captures them, and the user `mask` is ANDed in.
5. Hidden scores become $-\infty$ before the softmax, never after: then their weights are exactly zero and the rows still sum to one over what is visible. `F.softmax` returns zeros for an all-$-\infty$ row (`M09.2`).
6. A learned sink is one extra column of logits. Concatenate it, softmax over $T_k + 1$ entries, then slice off the sink: the remaining weights sum to $1 - p_\text{sink}$. Because the sink column is built from the `sinks` tensor, it trains.
7. RoPE's attention scaling is applied inside `rope_cos_sin` to both $q$ and $k$, so the layer needs no special case for YaRN: the $s^2$ arrives in the scores automatically.

### 5.2 The code, annotated

**The cache hook and its smallest implementation.**

```python
class KVCacheHook(Protocol):
    """The seam L8.2's KVCache (and L8.3's paged cache) implements."""
    def update(self, layer, k_new, v_new) -> tuple[NDArray, NDArray]: ...

class ConcatKVCache:
    def update(self, layer, k_new, v_new):
        # copy: the caller's arrays may be reused or mutated after this call
        k_new, v_new = np.array(k_new, copy=True), np.array(v_new, copy=True)
        if k_new.ndim != 4 or k_new.shape != v_new.shape:
            raise ValueError(...)
        if layer in self._k:
            old = self._k[layer]
            # a chunk must extend the same batch, heads and head width; only time grows
            if old.shape[:2] != k_new.shape[:2] or old.shape[3] != k_new.shape[3]:
                raise ValueError(...)
            self._k[layer] = np.concatenate([old, k_new], axis=2)
            self._v[layer] = np.concatenate([self._v[layer], v_new], axis=2)
        else:
            self._k[layer], self._v[layer] = k_new, v_new
        # return everything cached so far: the layer attends over all of it
        return self._k[layer], self._v[layer]
```

A `Protocol` rather than a base class keeps the attention layer ignorant of how a cache stores things: `L8.2` preallocates, `L8.3` pages, and both just implement `update`. The cache works on numpy arrays because it is an inference-only object; no gradient flows into cached keys.

**Repeating kv heads.**

```python
def repeat_kv(x, n_rep):
    if n_rep < 1 or x.ndim != 4:
        raise ValueError(...)
    if n_rep == 1:
        return x                         # MHA: nothing to do
    # head h reads kv head h // n_rep: [0, 0, 1, 1, ...], not [0, 1, 0, 1, ...]
    idx = np.repeat(np.arange(x.shape[1]), n_rep)
    return x[:, idx]                     # fancy indexing: gradients of copies add up
```

`np.repeat` (each element $n_\text{rep}$ times in a row) and not `np.tile` (the whole array $n_\text{rep}$ times) is the entire difference between the checkpoint's grouping and a silently wrong model.

**Visibility.**

```python
def _visible(T, Tk, window):
    """bool [T, Tk]: query t (key index Tk - T + t) may read key j."""
    i = np.arange(T)[:, None] + (Tk - T)   # absolute index of each new query
    j = np.arange(Tk)[None, :]
    vis = j <= i                           # causal
    if window is not None:
        vis &= (i - j) < window            # the query and the W - 1 keys before it
    return vis
```

With no cache $T_k = T$ and this is the ordinary lower triangle. In one-token decoding $T = 1$ and the single row is "all keys" (or the last $W$).

**The constructor.**

```python
        if min(d, n_heads, n_kv_heads) < 1 or n_heads % n_kv_heads:
            raise ValueError(...)            # groups must divide evenly
        if d_head is None:
            if d % n_heads:
                raise ValueError(...)
            d_head = d // n_heads
        # ... validate d_head, rope.rotary_dim (even, 2..d_head) and window >= 1
        r = rng if rng is not None else PCG32(0).substream("init")
        # ...
        # draw order q, k, v, o: fixed, so a seeded init is reproducible
        self.q_proj = Linear(d, n_heads * d_head, bias=qkv_bias, rng=r)
        self.k_proj = Linear(d, n_kv_heads * d_head, bias=qkv_bias, rng=r)
        self.v_proj = Linear(d, n_kv_heads * d_head, bias=qkv_bias, rng=r)
        self.o_proj = Linear(n_heads * d_head, d, bias=False, rng=r)  # never a bias, even for Qwen2
        self.sinks = Tensor(np.zeros(n_heads, dtype=np.float32), requires_grad=True) if sinks else None
```

The attribute names `q_proj`, `k_proj`, `v_proj`, `o_proj`, `sinks` are the Hugging Face state-dict keys, so `L7.9` loads safetensors by name with no renaming table.

**The forward.**

```python
    def forward(self, x, positions, mask=None, cache=None, layer=0):
        # ... validate x is [B, T, d]
        B, T, _ = x.shape
        H, Hkv, dh = self.n_heads, self.n_kv_heads, self.d_head
        pos = np.asarray(positions)
        if pos.shape not in ((T,), (B, T)):           # shared or per-row positions
            raise ValueError(...)
        q = self._heads(self.q_proj(x), B, T, H)      # [B, H, T, dh]
        k = self._heads(self.k_proj(x), B, T, Hkv)    # [B, Hkv, T, dh]
        v = self._heads(self.v_proj(x), B, T, Hkv)
        cos, sin = rope_cos_sin(pos, self.rope.inv_freq, self.rope.attention_scaling)
        if pos.ndim == 2:  # per-row positions: [B, 1, T, r/2] broadcasts over heads
            cos, sin = cos[:, None], sin[:, None]
        q = apply_rope(q, cos, sin, self.rope.layout, self.rope.rotary_dim)
        k = apply_rope(k, cos, sin, self.rope.layout, self.rope.rotary_dim)
        if cache is not None:
            # The cache keeps Hkv heads, not H: that is the point of GQA.
            # Keys are already rotated by their own positions (step 3).
            ka, va = cache.update(layer, k.data, v.data)
            k, v = Tensor(ka, dtype=x.dtype), Tensor(va, dtype=x.dtype)  # constants
        Tk = k.shape[2]
        vis = _visible(T, Tk, self.window)
        if mask is not None:
            m = np.asarray(mask)
            # ... check m broadcasts to [B, H, T, Tk]
            vis = vis & m.astype(bool)                # True = may attend
        n_rep = H // Hkv
        # scale by the HEAD width: q . k sums dh products
        scores = F.matmul(q, F.transpose(repeat_kv(k, n_rep), 2, 3)) * (dh**-0.5)
        scores = F.masked_fill(scores, ~vis, -np.inf)
        if self.sinks is not None:
            # one extra logit per head in every row: weight it may give to nothing
            col = F.reshape(self.sinks, (1, H, 1, 1)) * np.ones((B, 1, T, 1))
            p = F.softmax(F.concat([scores, col], axis=-1), axis=-1)[..., :Tk]
        else:
            p = F.softmax(scores, axis=-1)            # over keys, the last axis
        o = F.matmul(p, repeat_kv(v, n_rep))          # [B, H, T, dh]
        return self.o_proj(F.reshape(F.transpose(o, 1, 2), (B, T, H * dh)))
```

Walk section 3 through it: $H = 2$, $H_{kv} = 1$, so $n_\text{rep} = 2$ and `repeat_kv` gives both query heads kv head 0. For token 1, head 0's scores are $(0, 1)/\sqrt 2$, `_visible(2, 2, None)` keeps both keys, the softmax gives $(0.330237, 0.669763)$, and `o_proj` picks entry 0 of head 0 and entry 1 of head 1: $(0.330237, 0.660474)$.

The sink column is built by multiplying the `[1, H, 1, 1]` sinks by a `[B, 1, T, 1]` array of ones: broadcasting turns it into one logit per (batch, head, query), and because the product involves the `sinks` tensor its gradient is the sum over every row the sink appeared in.

Full solution: [`gqa.py`](../../../../course/solutions/python/tinyllm/modern/gqa.py).

### 5.3 Why this solution

- **Cost.** Projections are $O(B T d (H + 2H_{kv}) d_h)$; scores and values are $O(B H T T_k d_h)$, the same as MHA. The cache, and the `k_proj`/`v_proj` weights, shrink by $H / H_{kv}$: 3 times for SmolLM2, 8 for Llama-2-70B.
- **Repeat in memory, not in the kernel.** `repeat_kv` materialises $H$ copies of each key for the matmul. Production kernels index kv head `h / n_rep` directly and never copy (see Going further). The copy here is temporary and lets the layer reuse `F.matmul` and its backward unchanged; only the cache, which lives for the whole generation, must be small.
- **One boolean mask.** Combining causal, window, and padding into `vis` and filling once with $-\infty$ is simpler and safer than adding separate bias tensors: there is one polarity (True = may attend) and one place where scores are hidden.
- **Sinks as a concatenated column.** Writing the denominator $e^{\sigma_h} + \sum e^{s}$ by hand would need a custom, numerically careful softmax. Concatenating a column reuses `F.softmax`'s max-subtraction and backward.
- **Accepted tradeoff.** The cache stores float arrays and the layer re-wraps them as fresh `Tensor`s each call, so the cache is never part of the graph. That rules out training through a cache, which nobody needs.

### 5.4 Explain it back

1. Why is `cache.update` called after `apply_rope` and before `repeat_kv`?
   <details><summary>Answer</summary>After RoPE, so each key is stored rotated by its own position; rotating later would rotate every cached key to the newest position. Before `repeat_kv`, so the cache holds $H_{kv}$ heads and realises GQA's memory saving, matching `kv_bytes_per_token`.</details>
2. Why does `_visible` add `Tk - T` to the query index?
   <details><summary>Answer</summary>With a cache the new queries are the last $T$ of $T_k$ positions. Without the offset, a decoded token would be treated as position 0 and could see only key 0.</details>
3. Why is adding $\sigma_h$ to every score not a sink?
   <details><summary>Answer</summary>Softmax is invariant to adding a constant to a whole row, so the weights would not change. A sink must be a separate entry in the denominator that takes weight away from every key, which is what the concatenated column does.</details>
4. Why is the scale $1/\sqrt{d_h}$ and not $1/\sqrt{d}$?
   <details><summary>Answer</summary>Each score is a dot product of two $d_h$-dimensional vectors, so with unit-variance entries its variance is $d_h$. Dividing by $\sqrt{d_h}$ keeps scores at unit scale regardless of the model width; $\sqrt d$ makes attention too flat whenever $d \ne d_h$.</details>
5. What does a fully masked row produce, and why does that matter?
   <details><summary>Answer</summary>All scores are $-\infty$, and the masked softmax returns zeros instead of NaN, so the output for that row is zero and other rows are unaffected. Padded positions in a batch hit this case constantly.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. tiling kv heads instead of repeating them | query head 1 reads kv head 1, not 0: loads fine, wrong logits | `np.repeat(np.arange(Hkv), n_rep)` in `repeat_kv` |
| 2. scaling by $1/\sqrt{d}$ instead of $1/\sqrt{d_h}$ | attention too flat whenever $d \ne d_h$ | multiplies by `dh**-0.5` |
| 3. a causal mask that ignores the cached prefix | decoded tokens see only the first keys | `_visible` offsets queries by `Tk - T` (step 4) |
| 4. adding the sink to every score | no effect at all (softmax is shift invariant) | concatenates a sink column and slices it off after the softmax (step 6) |
| 5. a window one key too wide ($\le W$) | one extra key per query; outputs drift from the reference | `(i - j) < window`, strictly |
| 6. rotating keys after the cache | every cached key re-rotated to the newest position: generation degrades after the prompt | `apply_rope` runs before `cache.update` (step 3) |
| mask polarity inverted | only padding is read | `vis & mask`, then `masked_fill(scores, ~vis, -inf)`: True means may attend |
| per-row positions ignored | left-padded rows rotated wrongly | accepts `[B, T]` positions and inserts a head axis into the tables |
| a bias on `o_proj` with `qkv_bias` | Qwen2 checkpoints fail to load | `o_proj` is always `bias=False` |
| caching the repeated heads | the cache is $n_\text{rep}$ times too large | `repeat_kv` is applied only after `cache.update` |
| softmax over the wrong axis | weights sum to 1 over queries | `F.softmax(..., axis=-1)`, the key axis |
| sinks as constants | the sinks never train | `sinks` is a `Tensor` with `requires_grad=True` and enters the graph through the concatenated column |
| ignoring the attention scaling | YaRN models lose their temperature | passes `rope.attention_scaling` to `rope_cos_sin` (step 7) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L7.3` | `rope_cos_sin`, `apply_rope`, `RopeSpec` rotate $q$ and $k$ |
| Back | `L0.4` | the four `Linear` projections |
| Back | `L0.2` | `F.matmul`, `F.masked_fill`, `F.softmax`, `F.concat` give the backward |
| Back | `L0.1` | `Tensor` and its indexing (`repeat_kv`) |
| Back | `M06.3` | `PCG32` initializes the layers when no rng is given |
| Back | `M05.1` | `kv_bytes_per_token` is what the cache hook must receive |
| Forward | `L7.9` | `self_attn` of every Llama layer |
| Forward | `L7.7` | sliding-window caches and StreamingLLM sinks build on `window` and `sinks` |
| Forward | `L8.2` | `KVCache.update` is the hook |
| Forward | `L9.3` | the C attention kernel takes `Hkv` and reads kv head `h / n_rep` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `GQAttention` | HF `LlamaAttention`, `GptOssAttention` | interchangeable attention backends (eager, SDPA, FlashAttention) | `transformers/models/llama/modeling_llama.py` |
| `repeat_kv` | FlashAttention, vLLM paged attention | no repetition in memory: the kernel indexes kv head `h / n_rep` directly | FlashAttention-2 `num_heads_k`; vLLM `paged_attention` |
| the cache hook | vLLM PagedAttention, SGLang RadixAttention | block tables, prefix sharing, eviction | `L8.3`, `L8.4` in this course |
| learned sinks | StreamingLLM | keeps the first tokens' KV forever so a sliding window stays stable | `L7.7` in this course; Xiao et al. 2023 |
