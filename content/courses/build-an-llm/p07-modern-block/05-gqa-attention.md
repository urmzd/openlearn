# Grouped-Query Attention with a Cache Hook, Windows, and Sinks

The multi-head attention of [Multi-head attention](../p05-transformer-2017/03-multi-head-attention.md) gives every query head its own key and value head. At inference the engine caches keys and values for every past token (you will build that cache in [KV cache and generate](../p08-inference/02-kv-cache-and-generate.md), and the Rust engine in [Serving](../p10-serving/) keeps one too), so cache memory, not compute, limits how many requests run at once.

Open SmolLM2-135M's checkpoint and you will see the response to that pressure: `k_proj.weight` is $192 \times 576$, a third of `q_proj`'s $576 \times 576$. The model has 9 query heads but only 3 key/value heads, and plain multi-head attention cannot load it. Newer models add two more things to the same layer: a **sliding window** (Mistral, gpt-oss) so the cache stops growing, and **learned sinks** (gpt-oss) so a head can decline to attend. This section builds that attention layer, together with the hook through which the KV cache will later plug in.

The layer builds on [Rotary Position Embeddings](03-rope.md) (`rope_cos_sin`, `apply_rope`, `RopeSpec`), `Linear` and `Module` from [Module system and basic layers](../p00-foundations/04-module-system-and-layers.md), `F.matmul`, `F.masked_fill`, `F.softmax`, and `F.concat` from [The op library and gradcheck_all](../p00-foundations/02-op-library-and-gradcheck.md), and the `Tensor` of [Tensor, broadcasting backward, and no_grad](../p00-foundations/01-tensor-and-broadcasting-backward.md). From *Mathematics for Computing* it uses `PCG32` from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) for default initialization and `kv_bytes_per_token` from [Counting: params, FLOPs, KV bytes, memory plans](../../mathematics-for-computing/05-discrete-math-1/01-counting-params-flops-kv-bytes.md) for what the cache must hold; [Stable numerics](../../mathematics-for-computing/09-numerical-methods-and-floating-point/02-stable-numerics.md) covers the masked softmax, and [Scaled dot-product attention](../p05-transformer-2017/01-scaled-dot-product-attention.md) is the attention it extends. The reference solution is [`gqa.py`](../../../../course/solutions/python/tinyllm/modern/gqa.py), with `GQAttention`, `repeat_kv`, `_visible`, the `KVCacheHook` protocol, and its smallest implementation `ConcatKVCache`, against the contract [`course/contracts/py/tinyllm/modern/gqa.pyi`](../../../../course/contracts/py/tinyllm/modern/gqa.pyi).

> **Learning objectives**
> - Explain how $H$ query heads share $H_{kv}$ key/value heads and why the grouping must repeat each kv head in a row.
> - State the visibility rule for a query in a cached call, including a sliding window and a padding mask.
> - Place the cache hook so the cache stores only $H_{kv}$ rotated heads, and explain both constraints.
> - Implement learned sinks as an extra softmax column and explain why adding a constant to every score does nothing.
> - Work a two-head, one-kv-head layer by hand.

## Notation

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

## Attention with rotated heads

Start from what you know. Each head computes scaled dot-product scores $q_t \cdot k_j / \sqrt{d_h}$, a softmax over the visible keys, and a weighted average of the values; the heads are concatenated and projected back to $d$ by `o_proj`. RoPE rotates $q$ and $k$ (with `RopeSpec`'s frequencies, layout, rotary width, and attention scaling) before the scores, so a score depends on the relative position of query and key only. The scale is $1/\sqrt{d_h}$, the head width, because a dot product of $d_h$ unit-variance products has variance $d_h$.

## Sharing kv heads

Multi-query attention (Shazeer 2019) keeps one key/value head for all query heads ($H_{kv} = 1$). Grouped-query attention (Ainslie et al.) keeps $H_{kv}$ of them, each serving a contiguous group of $n_\text{rep}$ query heads.

`repeat_kv` expands $[B, H_{kv}, T, d_h]$ to $[B, H, T, d_h]$ with output head $h$ reading input head $\lfloor h / n_\text{rep} \rfloor$: $(a, b) \to (a, a, a, b, b, b)$, the order the checkpoints were trained with. Tiling, $(a, b, a, b, a, b)$, loads without error and is wrong. Seen another way, GQA is MHA whose kv heads come in identical groups, so an MHA layer with repeated kv weights gives the same output. Because indexing accumulates gradients, a kv head's gradient is the sum over its copies.

## Who may see whom

Once a cache is involved, positions within a call and positions in the sequence differ, so the visibility rule has to be stated carefully. Query $t$ of this call sits at key index $i = T_k - T + t$ (with no cache, $T_k = T$ and $i = t$). Key $j$ is visible when all of these hold:

- $j \le i$: causal, always;
- $i - j < W$ when a window is set: the query and the $W - 1$ keys before it;
- the optional `mask` allows it (`True` means may attend, broadcast to $[B, H, T, T_k]$; this is how padding is expressed).

Hidden scores are set to $-\infty$, so their weights are exactly 0, and a row with nothing visible gives zeros, not NaN.

## The cache hook

Decoding one token at a time, recomputing every past key and value is wasted work. With `cache` given, after RoPE the layer calls `cache.update(layer, k, v)` with this call's $[B, H_{kv}, T, d_h]$ keys and values (numpy arrays, after rotation, before repetition) and attends over everything the cache returns.

Two consequences follow from where that call sits. First, the cache stores $H_{kv}$ heads, which is where GQA's saving happens: $2 H_{kv} d_h$ numbers per token per layer, exactly `kv_bytes_per_token`. Second, keys must be rotated with **their own** positions before they are stored, because a key cached at position 5 must stay rotated by 5. `ConcatKVCache` is the smallest implementation (it concatenates along time); the `KVCache` of [KV cache and generate](../p08-inference/02-kv-cache-and-generate.md) and the paged cache of [Paged KV cache in Python](../p08-inference/03-paged-kv-cache.md) implement the same `update`.

## Learned sinks

A softmax must put weight 1 somewhere, so a head with nothing useful to read dumps it on some token, often the first; StreamingLLM calls this an "attention sink". gpt-oss gives each head a learned logit $\sigma_h$ appended to every row before the softmax and dropped afterwards:

$$a_{tj} = \frac{e^{s_{tj}}}{e^{\sigma_h} + \sum_{j'} e^{s_{tj'}}}.$$

The weights now sum to $1 - p_\text{sink} \le 1$. A large $\sigma_h$ means the head attends to almost nothing, and a very negative one recovers plain attention. Adding $\sigma_h$ to every score instead does nothing at all, because softmax ignores a constant shift.

As for the backward, every step is an op from the op library or from the RoPE section, so autograd reaches $x$, the four projections, their biases, and the sinks. With a cache, the cached keys and values are constants: the cache is for inference, and no gradient flows into it.

## Working it by hand

Take $d = 2$, $H = 2$, $H_{kv} = 1$, $d_h = 2$, with RoPE off (`inv_freq = [0]`, so every angle is 0). Choose weights so that `q_proj` makes head 0's query $x$ and head 1's $-x$; $k = x$; $v = (x_0, 2 x_1)$; and `o_proj` keeps entry 0 of head 0 and entry 1 of head 1. The tokens are $x^{(0)} = (1, 0)$ and $x^{(1)} = (0, 1)$. Both heads read the single kv head: $k^{(0)} = (1, 0)$, $k^{(1)} = (0, 1)$, $v^{(0)} = (1, 0)$, $v^{(1)} = (0, 2)$.

1. Token 0 sees only key 0, so both heads output $v^{(0)} = (1, 0)$ and the output is $(1, 0)$.
2. For token 1, head 0 has $q = (0, 1)$ and scores $(0, 1) / \sqrt 2 = (0, 0.707107)$. Since $e^{0.707107} = 2.028115$, the weights are $(1, 2.028115) / 3.028115 = (0.330237, 0.669763)$, and the head output is $0.330237\, v^{(0)} + 0.669763\, v^{(1)} = (0.330237, 1.339527)$.
3. For token 1, head 1 has $q = (0, -1)$, scores $(0, -0.707107)$, weights $(0.669763, 0.330237)$, and head output $(0.669763, 0.660474)$.
4. `o_proj` keeps entry 0 of head 0 and entry 1 of head 1: $(0.330237, 0.660474)$.

One kv head served two query heads that attend in opposite directions.

## The interface

```python
class KVCacheHook(Protocol):
    def update(self, layer: int, k_new: NDArray, v_new: NDArray) -> tuple[NDArray, NDArray]: ...
class ConcatKVCache:                     # update(...), seq_len(layer=0)
def repeat_kv(x: Tensor, n_rep: int) -> Tensor: ...                      # [B, Hkv, T, dh] -> [B, Hkv n_rep, T, dh]
class GQAttention(Module):
    def __init__(self, d, n_heads, n_kv_heads, d_head, rope: RopeSpec, qkv_bias=False, window=None, sinks=False, rng=None): ...
    def forward(self, x, positions, mask=None, cache=None, layer=0) -> Tensor: ...   # [B, T, d] -> [B, T, d]
```

`GQAttention.forward` takes positions of shape `[T]` (shared) or `[B, T]` (per row), an optional boolean mask, and an optional cache with a layer index. `repeat_kv` and the cache hook are public because later chapters reuse them.

## Building it step by step

The checkpoint fixes the parameter shapes: `q_proj` is $d \to H d_h$, `k_proj` and `v_proj` are $d \to H_{kv} d_h$, and `o_proj` is $H d_h \to d$ with no bias. $d_h$ is a separate argument because some models set it independently of $d / H$.

Attention needs one key per query head. The cheapest correct way to get there is to compute $H_{kv}$ keys and values and **index** them, so query head $h$ reads kv head $\lfloor h / n_\text{rep} \rfloor$. Indexing gives the backward for free: copies of a kv head sum their gradients.

The cache must see the smallest thing that can be stored: keys after RoPE (so each keeps its own position) and before `repeat_kv` (so it holds $H_{kv}$ heads). That fixes where `cache.update` sits in the forward, after `apply_rope` and before `repeat_kv`.

With a cache, this call's $T$ queries are the last $T$ of $T_k$ keys. Every visibility rule (causal, window) is a comparison between absolute key indices $i = T_k - T + t$ and $j$, so one boolean `[T, Tk]` matrix captures them all, and the user `mask` is ANDed in. Hidden scores become $-\infty$ before the softmax, never after; then their weights are exactly zero and the rows still sum to one over what is visible. `F.softmax` returns zeros for an all-$-\infty$ row.

A learned sink is one extra column of logits. Concatenate it, softmax over $T_k + 1$ entries, then slice off the sink, and the remaining weights sum to $1 - p_\text{sink}$. Because the sink column is built from the `sinks` tensor, it trains. Finally, RoPE's attention scaling is applied inside `rope_cos_sin` to both $q$ and $k$, so the layer needs no special case for YaRN: the $s^2$ arrives in the scores automatically.

### The cache hook and its smallest implementation

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

A `Protocol` rather than a base class keeps the attention layer ignorant of how a cache stores things: the KV cache of the inference chapter preallocates, the paged cache pages, and both just implement `update`. The cache works on numpy arrays because it is an inference-only object; no gradient flows into cached keys.

### Repeating kv heads

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

### Visibility

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

### The constructor

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

The attribute names `q_proj`, `k_proj`, `v_proj`, `o_proj`, and `sinks` are the Hugging Face state-dict keys, so the Llama loader reads safetensors by name with no renaming table.

### The forward

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
            # Keys are already rotated by their own positions.
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

Walk the hand example through it. With $H = 2$ and $H_{kv} = 1$, $n_\text{rep} = 2$ and `repeat_kv` gives both query heads kv head 0. For token 1, head 0's scores are $(0, 1)/\sqrt 2$, `_visible(2, 2, None)` keeps both keys, the softmax gives $(0.330237, 0.669763)$, and `o_proj` picks entry 0 of head 0 and entry 1 of head 1: $(0.330237, 0.660474)$.

The sink column is built by multiplying the `[1, H, 1, 1]` sinks by a `[B, 1, T, 1]` array of ones. Broadcasting turns it into one logit per (batch, head, query), and because the product involves the `sinks` tensor, its gradient is the sum over every row the sink appeared in. The full solution is in [`gqa.py`](../../../../course/solutions/python/tinyllm/modern/gqa.py).

### Why this design

Projections cost $O(B T d (H + 2H_{kv}) d_h)$, and scores and values cost $O(B H T T_k d_h)$, the same as MHA. The cache, and the `k_proj` and `v_proj` weights, shrink by $H / H_{kv}$: 3 times for SmolLM2, 8 for Llama-2-70B.

The solution repeats kv heads in memory, not in the kernel. `repeat_kv` materialises $H$ copies of each key for the matmul, whereas production kernels index kv head `h / n_rep` directly and never copy. The copy here is temporary and lets the layer reuse `F.matmul` and its backward unchanged; only the cache, which lives for the whole generation, must be small.

Combining causal, window, and padding into one boolean `vis` and filling once with $-\infty$ is simpler and safer than adding separate bias tensors: there is one polarity (True means may attend) and one place where scores are hidden. Writing the sink denominator $e^{\sigma_h} + \sum e^{s}$ by hand would need a custom, numerically careful softmax; concatenating a column reuses `F.softmax`'s max-subtraction and backward.

The tradeoff you accept is that the cache stores float arrays and the layer re-wraps them as fresh `Tensor`s each call, so the cache is never part of the graph. That rules out training through a cache, which nobody needs.

## What can go wrong

The checks cover numerics, structure, and the cache. The hand example must come out exactly. Hugging Face goldens cover Llama GQA, an extra mask, Qwen2 biases, and gpt-oss sinks and window with gapped per-row positions, comparing outputs and every gradient; SmolLM2 depends on them. A float64 gradcheck covers every parameter, with biases, window, and sinks. `repeat_kv` must map $(a, b) \to (a, a, a, b, b, b)$ and sum the gradients of copies, and GQA must equal MHA with grouped kv weights, which is what GQA is.

For the cache, prefill followed by decoding through the cache must equal the full forward, with and without window and sinks, since cached generation depends on it. The hook must receive only $H_{kv}$ heads, with bytes equal to `kv_bytes_per_token`, which the scheduler in [Continuous batching](../p10-serving/02-continuous-batching.md) uses for admission. `ConcatKVCache` must append, copy, and reject bad chunks.

Structural properties come next. Future tokens must change nothing, bit for bit. With a window, key $t - W$ must be invisible and $t - W + 1$ visible. A large sink must drive the output to 0 and a very negative one must recover plain attention. A fully masked row must give zeros, no NaN, and leave other rows intact, as padded batches require. Shifting all positions must leave outputs unchanged, and per-row positions must equal separate calls, which is what left padding and continued caches need. YaRN's attention scaling must appear as $s^2$ on the scores. The parameter names and shapes must be Hugging Face's, with no `o_proj` bias, the `d_head` default, and the fixed draw order. Finally, bad heads, `d_head`, window, rotary width, input, positions, or mask must fail loudly.

[`test_gqa.py`](../../../../course/solutions/learner-tests/L7.5/python/tests/l7-5-gqa/test_gqa.py) writes the whole layer out in numpy float64 with the module's own `state_dict()`: the projections, RoPE as complex multiplication, per head the scores of kv head $\lfloor h / n_\text{rep} \rfloor$, the causal and window mask, the optional sink column, the softmax, the values, and `o_proj`. It compares `forward` with and without window and sinks, decodes through `ConcatKVCache` against the full forward, and checks the head order of `repeat_kv`, the mask, per-row positions, the parameter names, the attention scaling, the cache's copies, and the validation. The oracle indexes kv head `h // n_rep` per query head directly instead of repeating, so it is an independent statement of the grouping.

> **Warning:** Rotating keys after the cache, rather than before, passes every prefill test. It then re-rotates every cached key to the newest position, and generation degrades right after the prompt. `apply_rope` must run before `cache.update`.

The full list of pitfalls:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| tiling kv heads instead of repeating them | query head 1 reads kv head 1, not 0: loads fine, wrong logits | `np.repeat(np.arange(Hkv), n_rep)` in `repeat_kv` |
| scaling by $1/\sqrt{d}$ instead of $1/\sqrt{d_h}$ | attention too flat whenever $d \ne d_h$ | multiplies by `dh**-0.5` |
| a causal mask that ignores the cached prefix | decoded tokens see only the first keys | `_visible` offsets queries by `Tk - T` |
| adding the sink to every score | no effect at all (softmax is shift invariant) | concatenates a sink column and slices it off after the softmax |
| a window one key too wide ($\le W$) | one extra key per query; outputs drift from the reference | `(i - j) < window`, strictly |
| rotating keys after the cache | every cached key re-rotated to the newest position: generation degrades after the prompt | `apply_rope` runs before `cache.update` |
| mask polarity inverted | only padding is read | `vis & mask`, then `masked_fill(scores, ~vis, -inf)`: True means may attend |
| per-row positions ignored | left-padded rows rotated wrongly | accepts `[B, T]` positions and inserts a head axis into the tables |
| a bias on `o_proj` with `qkv_bias` | Qwen2 checkpoints fail to load | `o_proj` is always `bias=False` |
| caching the repeated heads | the cache is $n_\text{rep}$ times too large | `repeat_kv` is applied only after `cache.update` |
| softmax over the wrong axis | weights sum to 1 over queries | `F.softmax(..., axis=-1)`, the key axis |
| sinks as constants | the sinks never train | `sinks` is a `Tensor` with `requires_grad=True` and enters the graph through the concatenated column |
| ignoring the attention scaling | YaRN models lose their temperature | passes `rope.attention_scaling` to `rope_cos_sin` |

## Where this leads

[Sliding windows and attention sinks](07-sliding-window-and-sinks.md) builds sliding-window caches and StreamingLLM sinks on `window` and `sinks`. In [the Llama-family model](09-llama-family-model.md), this layer is the `self_attn` of every decoder layer (SmolLM2: 9 query heads over 3 kv heads), on the way to SmolLM2-135M logits that match Hugging Face. The `KVCache.update` of [KV cache and generate](../p08-inference/02-kv-cache-and-generate.md) is the hook, and later the C kernels in [FlashAttention forward in C](../p09-kernels/06-flash-attention-forward.md) and [Paged attention for decode in C](../p09-kernels/07-paged-attention-decode.md) take `Hkv` and read kv head `h / n_rep` directly.

### Further reading

- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019).
- Ainslie et al., ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245) (2023).
- Beltagy et al., "Longformer" (2020), section 3.1.
- Xiao et al., ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453) (2023).
- OpenAI, "gpt-oss-120b and gpt-oss-20b model card" (2025).
- Hugging Face `LlamaAttention` and `GptOssAttention` in `transformers/models/llama/modeling_llama.py`: interchangeable attention backends (eager, SDPA, FlashAttention).
- FlashAttention-2 (`num_heads_k`) and vLLM `paged_attention`: no repetition in memory; the kernel indexes kv head `h / n_rep` directly.
- vLLM PagedAttention and SGLang RadixAttention: block tables, prefix sharing, and eviction, which this book builds in [Paged KV cache in Python](../p08-inference/03-paged-kv-cache.md) and [Radix prefix cache](../p08-inference/04-radix-prefix-cache.md).

## Key terms

- **Multi-query attention (MQA)**: all query heads share one key/value head.
- **Grouped-query attention (GQA)**: $H$ query heads share $H_{kv}$ key/value heads, each serving a contiguous group of $n_\text{rep}$ query heads.
- **`repeat_kv`**: the expansion that lets query head $h$ read kv head $\lfloor h / n_\text{rep} \rfloor$.
- **Cache hook**: the `update(layer, k, v)` protocol through which a KV cache receives rotated, unrepeated keys and values.
- **Sliding window**: a limit $W$ so query $t$ reads only keys $t - W + 1 .. t$.
- **Learned sink**: a per-head extra logit in every softmax row, dropped afterwards, letting a head attend to nothing.

## Summary

- $H$ query heads share $H_{kv}$ key/value heads; query head $h$ reads kv head $\lfloor h / n_\text{rep} \rfloor$, each kv head repeated $n_\text{rep}$ times in a row.
- The cache hook receives only the $H_{kv}$ heads, so the KV cache shrinks by $H / H_{kv}$: exactly `kv_bytes_per_token`.
- With a cache, the new queries are the last $T$ of $T_k$ keys: the causal mask uses that offset, and keys are rotated with their own positions before they are cached.
- A sliding window of $W$ lets query $t$ read keys $t - W + 1 .. t$.
- A learned sink is one extra logit per head in every softmax row, dropped afterwards: a head may attend to nothing.

## Review questions

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
6. Why does tiling kv heads, $(a, b, a, b, a, b)$, load without error?
   <details><summary>Answer</summary>Tiling and repeating produce tensors of the same shape, so nothing in the layer can detect the difference. Only the outputs reveal it: query head 1 reads kv head 1 instead of 0, so every logit is wrong.</details>
