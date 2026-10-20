# Sliding Windows and Attention Sinks

The grouped-query layer of [Grouped-Query Attention](05-gqa-attention.md) reads every earlier token, so a chat that runs for hours grows its KV cache without bound until the engine runs out of memory, and the attention cost per token grows with it. Mistral reads only the last $W$ tokens. That bounds memory, but, as StreamingLLM found, a model trained with full attention falls apart the moment its first tokens leave the window: it had learned to park unwanted attention on them. Keeping a few "sink" tokens visible forever fixes it. gpt-oss goes one step further and learns a sink logit per head.

Three later pieces of the book need what this section builds. [The Llama-family model](09-llama-family-model.md) loads configs that use windows and sinks, [KV cache and generate](../21-inference/02-kv-cache-and-generate.md) needs an eviction policy, and the FlashAttention kernel in [FlashAttention forward in C](../22-kernels/05-flash-attention-forward.md) needs one reference to be checked against. Here you will build all three: the visibility rule, a cache that forgets, and a float64 reference attention with learned sinks.

The section extends `causal_mask` and `sliding_window_mask` from [Masks: causal, padding, sliding window, additive](../11-transformer/02-masks.md), uses `logsumexp` from [Stable numerics](../../mathematics/15-numerical-methods-and-floating-point/02-stable-numerics.md) in *Mathematics*, and streams through `GQAttention` and `ConcatKVCache` from the previous section, with keys rotated by the `RopeSpec` of [Rotary Position Embeddings](03-rope.md). The reference solution is [`window.py`](../../../../course/solutions/python/tinyllm/modern/window.py), with `sink_window_mask`, `windowed_attention`, `SinkWindowCache`, and `attention_mask`, against the contract [`course/contracts/py/tinyllm/modern/window.pyi`](../../../../course/contracts/py/tinyllm/modern/window.pyi).

> **Learning objectives**
> - State the single visibility rule that combines causality, a sliding window, and sink tokens, in absolute positions.
> - Build a cache that holds at most $S + W - 1$ keys per layer and explain why it must evict after the chunk reads.
> - Compute a softmax row with a learned sink by hand, including its log-sum-exp.
> - Give any attention layer the right mask for whatever cache it was handed.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p_i$ | absolute position of query $i$: `q_offset + i` | `int` |
| $j$ | absolute position of a key | `int` |
| $W$ | `window`: the query and $W - 1$ keys before it | `int` $\ge 1$ |
| $S$ | `n_sink`: tokens $0 .. S-1$ stay visible | `int` $\ge 0$ |
| $s_j$ | score $q \cdot k_j \cdot d_h^{-1/2}$ | `float` |
| $\sigma_h$ | head $h$'s learned sink logit | `float` |
| $\mathrm{lse}$ | $\log\big(\sum_{j\ \mathrm{visible}} e^{s_j} + e^{\sigma_h}\big)$ | `float` |

## The visibility rule

One rule decides every key a query reads:

$$\mathrm{visible}(i, j) = \big(j \le p_i\big) \wedge \big(j < S \ \vee\ p_i - j < W\big).$$

With $S = 0$ it is the `sliding_window_mask` you already have; with no window it is the causal mask. The window counts the query itself, so $W = 1$ sees only itself (plus the sinks). The rule uses absolute positions, never indices into a cache, because once a cache evicts, index and position stop agreeing.

## A cache that forgets

`SinkWindowCache(S, W)` implements the same cache hook as `ConcatKVCache`. On `update(layer, k, v)` it appends the chunk at absolute positions `seen .. seen + T - 1`, returns the held keys followed by the chunk (what the chunk's queries may need), and only then evicts down to positions $< S$ and the last $W - 1$ positions: what the next query needs besides itself. `positions(layer)` reports the absolute positions it returned, and `chunk_mask(layer, T)` builds the rule above for the next chunk against them.

Because the keys arrive already rotated at their absolute positions, attention through this cache is exactly full attention under the rule. StreamingLLM's paper goes further and re-rotates keys at their cache slots; that variant is listed under further reading.

## Learned sinks

A softmax must give weight 1 in total, so a head with nothing useful to read still has to put that weight somewhere. A learned sink gives it a place that is no token:

$$p_j = \frac{e^{s_j}}{\sum_{k\ \mathrm{visible}} e^{s_k} + e^{\sigma_h}}, \qquad \sum_j p_j = 1 - p_{sink}, \quad p_{sink} = e^{\sigma_h - \mathrm{lse}}.$$

`windowed_attention` is the numpy reference for all of this: grouped kv heads (head $h$ reads kv head $h / n_{rep}$), the rule, the sink in the denominator, and the log-sum-exp of every row. FlashAttention keeps the lse for the backward and for merging chunks, which is why the reference returns it. A row that sees nothing outputs 0, with lse $-\infty$, or lse equal to the sink logit when there is one.

## One mask function for every layer

Attention layers are handed different caches at different times, and each needs a different mask. `attention_mask(cache, layer, T, window, n_sink)` hides that. It returns no mask when there is no window; the full-sequence rule when there is no cache; the bounded cache's own `chunk_mask` for a `SinkWindowCache`; and, for a cache that keeps everything (`ConcatKVCache`, or the `KVCache` of the inference chapter), the rule placed at the chunk's offset `seq_len(layer)`.

## Working it by hand

Take window $W = 2$, one sink, and five positions. Query 3 reads key 0 (the sink), key 2 ($3 - 2 = 1 < 2$), and key 3; key 1 is out ($3 - 1 = 2$). The full mask is:

```
       k0 k1 k2 k3 k4
q0      1  .  .  .  .
q1      1  1  .  .  .
q2      1  1  1  .  .
q3      1  .  1  1  .
q4      1  .  .  1  1
```

Fed one token at a time, `SinkWindowCache(1, 2)` returns the keys at positions `[0]`, `[0, 1]`, `[0, 1, 2]`, `[0, 2, 3]`, `[0, 3, 4]`, each row of the mask, and then holds `[0, 4]`: $S + W - 1 = 2$ keys.

Now a learned-sink row: one query at position 1, scores $(0, \ln 2)$, values $(4, 8)$, and sink logit 0. The denominator is $e^0 + e^{\ln 2} + e^0 = 4$, so the weights are $(1/4, 1/2)$, the sink takes $1/4$, and the output is $4/4 + 8/2 = 5$, with $\mathrm{lse} = \ln 4$.

## The interface

```python
def sink_window_mask(Tq, Tk, window, n_sink=0, q_offset=0) -> NDArray          # bool [Tq, Tk]
def windowed_attention(q, k, v, window=None, n_sink=0, sink_logits=None, q_offset=0, scale=None)
    -> tuple[NDArray, NDArray]                                                 # out [B, H, Tq, dv], lse [B, H, Tq]
class SinkWindowCache:                                                         # the KVCacheHook protocol
    def __init__(self, n_sink, window); update, seq_len, held, positions, chunk_mask
def attention_mask(cache, layer, T, window, n_sink=0) -> Optional[NDArray]
```

`sink_window_mask` is the rule as a boolean matrix; `windowed_attention` is the float64 reference returning outputs and per-row lse; `SinkWindowCache` is the bounded cache; and `attention_mask` is the dispatcher.

## Building it step by step

The rule depends only on absolute positions, so it is one boolean matrix. Its two pieces already exist: the sliding window (which is causal) and the causal mask. Sinks add "causal and $j < S$", ORed in.

The reference attention must handle everything a kernel will: grouped kv heads, the rule, an optional sink logit, and the log-sum-exp of every row. Computing the lse with the sink column included makes the sink part of the denominator automatically, and $p_j = e^{s_j - \mathrm{lse}}$ then sums to $1 - p_{sink}$ with no extra code.

A bounded cache must return, for a chunk of $T$ queries, every key any of them may need: what it already holds plus the chunk. Only after that can it evict, and it must keep exactly what the **next** query needs besides itself, the sinks and the last $W - 1$ positions. After eviction, cache index and position disagree, so the cache records the absolute position of everything it holds and builds the next chunk's mask from those positions, not from indices. Finally, because different caches need different masks, one dispatch function lets every attention layer ask for its mask without knowing which cache it has.

### The rule

```python
def sink_window_mask(Tq, Tk, window, n_sink=0, q_offset=0):
    # ... validate integers (bool rejected; Tq, Tk >= 1, n_sink, q_offset >= 0)
    if window is None:
        return causal_mask(Tq, Tk, q_offset)
    window = _int("window", window, 1)
    recent = sliding_window_mask(Tq, Tk, window, q_offset)   # j <= p and p - j < W
    # Sink keys are exempt from the window, never from causality.
    sinks = causal_mask(Tq, Tk, q_offset) & (np.arange(Tk) < n_sink)[None, :]
    return recent | sinks
```

On the hand example ($W = 2$, $S = 1$), `recent` is the diagonal and the subdiagonal, and `sinks` adds column 0 for every query. That is the printed mask.

### The reference attention

```python
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))   # a float64 reference
    # ... shape checks: 4-D, matching B, Tk and dh, H divisible by Hkv
    scale = dh**-0.5 if scale is None else float(scale)
    rep = np.arange(H) // (H // Hkv)  # query head h reads kv head h // n_rep
    s = np.einsum("bhtd,bhkd->bhtk", q, k[:, rep]) * scale
    vis = sink_window_mask(Tq, Tk, window, n_sink, q_offset)
    s = np.where(vis, s, -np.inf)
    if sink_logits is not None:
        sl = np.asarray(sink_logits, dtype=np.float64)
        # ... must be [H]
        col = np.broadcast_to(sl[None, :, None, None], (B, H, Tq, 1))
        lse = logsumexp(np.concatenate([s, col], axis=-1), axis=-1)   # sink in the denominator
    else:
        lse = logsumexp(s, axis=-1)
    # p_j = exp(s_j - lse): the sink takes its share of the denominator and
    # gives back no value.
    with np.errstate(invalid="ignore"):
        p = np.where(np.isfinite(lse)[..., None] & vis, np.exp(s - lse[..., None]), 0.0)
    out = np.einsum("bhtk,bhkd->bhtd", p, v[:, rep])
    return out, lse
```

The `np.where` handles empty rows. With no visible key and no sink, lse is $-\infty$ and $s - \mathrm{lse}$ is $-\infty - (-\infty) = $ NaN; the guard replaces it by 0, so the output is 0. With a sink, an empty row has finite lse $= \sigma_h$ and every $p_j$ is 0. On the hand sink row, $\mathrm{lse} = \ln(1 + 2 + 1) = \ln 4$, $p = (1/4, 2/4)$, and the output is $4 \cdot 1/4 + 8 \cdot 1/2 = 5$.

Returning the lse is what lets the tiled C kernel be checked against this function row by row, since FlashAttention merges tiles through their lse.

### The bounded cache

```python
    def update(self, layer, k_new, v_new):
        k_new, v_new = np.array(k_new, copy=True), np.array(v_new, copy=True)
        # ... shape checks
        seen, T = self._seen.get(layer, 0), k_new.shape[2]
        new_pos = np.arange(seen, seen + T, dtype=np.int64)    # absolute positions of the chunk
        if layer in self._k:
            # ... check the chunk extends what is held
            k = np.concatenate([old, k_new], axis=2)
            v = np.concatenate([self._v[layer], v_new], axis=2)
            pos = np.concatenate([self._pos[layer], new_pos])
        else:
            k, v, pos = k_new, v_new, new_pos
        self._seen[layer] = seen + T
        self._last[layer] = pos           # what positions() reports: the keys just returned
        # Evict AFTER returning what this chunk needs: keep the sinks and the
        # window - 1 most recent positions, which is what the next query needs
        # besides itself.
        keep = (pos < self.n_sink) | (pos >= seen + T - (self.window - 1))
        self._k[layer], self._v[layer], self._pos[layer] = k[:, :, keep], v[:, :, keep], pos[keep]
        return k, v                       # held + chunk, before eviction
```

Trace the hand example one token at a time with $S = 1$, $W = 2$. At step 3 (position 3), held is `[0, 2]`, so `update` returns `[0, 2, 3]`; `keep` is `pos < 1` or `pos >= 4 - 1 = 3`, so it then holds `[0, 3]`. Step 4 returns `[0, 3, 4]` and holds `[0, 4]`. The held count never exceeds $S + W - 1 = 2$.

`seq_len` returns how many tokens have been **seen**, not held, because that is the position of the next token, which RoPE and the mask need.

### The chunk mask

```python
    def chunk_mask(self, layer, T):
        T = _int("T", T, 1)
        seen = self._seen.get(layer, 0)
        held = self._pos.get(layer, np.zeros(0, dtype=np.int64))
        # the keys update() will return: held positions, then the chunk's
        keys = np.concatenate([held, np.arange(seen, seen + T, dtype=np.int64)])
        p = seen + np.arange(T)[:, None]          # absolute position of each query
        j = keys[None, :]
        return (j <= p) & ((j < self.n_sink) | (p - j < self.window))
```

This is the visibility rule evaluated at positions, column for column aligned with what the next `update` returns. It is called before `update`, so it describes the cache's state at the moment the chunk arrives.

### Dispatch

```python
def attention_mask(cache, layer, T, window, n_sink=0):
    if window is None:
        return None                                   # plain causal: the layer's own rule
    if cache is None:
        return sink_window_mask(T, T, window, n_sink) # a fresh sequence
    if isinstance(cache, SinkWindowCache):
        return cache.chunk_mask(layer, T)             # columns are held positions
    if not hasattr(cache, "seq_len"):
        raise ValueError(...)
    s = int(cache.seq_len(layer))                     # keep-everything cache: offset by its length
    return sink_window_mask(T, s + T, window, n_sink, s)
```

The full solution is in [`window.py`](../../../../course/solutions/python/tinyllm/modern/window.py).

### Why this design

A stream of $N$ tokens holds at most $S + W - 1$ keys per layer, against $N$ for full attention, and per-step attention cost drops from $O(N)$ to $O(S + W)$.

Recording absolute positions costs one int per held key and is what makes the mask right after eviction; index-based masks look correct until the first sink is held. The cache concatenates and then evicts rather than using a ring buffer. Mistral's rolling buffer writes into $W$ fixed slots, which avoids copies but scrambles key order and cannot hold sinks without a second buffer. The concatenating cache costs a copy per update and keeps keys in position order, which keeps the mask simple and the reference easy to check.

The solution does not re-rotate. Keys are cached already rotated at their absolute positions, so attention through the cache equals full attention under the mask exactly. StreamingLLM re-rotates keys at their cache slots to keep distances within the trained range, which changes the outputs. Finally, `windowed_attention` is the oracle for the C kernel, so it computes in float64 and returns the lse; speed is not its job.

## What can go wrong

The checks follow the design. The hand example's mask, cache steps, and sink row must match. With no sinks the rule must equal `sliding_window_mask`, and with no window it must equal `causal_mask`, so there is one rule at many call sites. Learned-sink goldens from gpt-oss eager attention cover a window, a decode chunk, and the causal case, comparing both outputs and lse, because the C kernel is checked against this function. Hugging Face Mistral attention, run in one pass and token by token through the cache, covers Mistral checkpoints. Decoding and chunked prefill through the bounded cache must equal full attention, which StreamingLLM-style serving relies on, and a keep-everything cache must get the mask at the chunk's offset. Over 200 tokens the cache must never hold more than $S + W - 1$ keys, and the keys it returns must be the visible ones. `attention_mask` must dispatch all four cases correctly. Weights must sum to $1 - p_{sink}$, and empty rows must give 0, never NaN. A zero window, negative sinks, bad head counts, and a wrong-length sink vector must all raise.

[`test_window.py`](../../../../course/solutions/learner-tests/L7.7/python/tests/l7-7-window/test_window.py) uses three oracles: the visibility rule written as a double loop; attention with a sink computed row by row with `math.exp`; and full attention under the mask as the oracle for streaming through `SinkWindowCache` (decoding and chunks, with `GQAttention`). It adds the hand sink row, `positions()` after eviction, a keep-everything cache with `attention_mask`, and the validation errors. The streaming oracle is the important one: a bounded cache is correct exactly when it is indistinguishable from an unbounded cache under the mask.

> **Warning:** Evicting before the chunk's queries have read is the classic bug. A chunk's earliest queries may still need keys its last query does not, so evicting first silently drops keys from chunked prefill. Evict after returning, never before.

The full list:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| a window of $W$ keys before the query instead of $W$ including it | one key too many; Mistral disagrees after position $W$ | `p - j < window`, the same rule as the 2017 masks |
| sink tokens falling out of the window like any other key | the stream degrades exactly when token 0 leaves the window | the mask ORs in `j < n_sink` |
| evicting before the chunk's queries read | chunked prefill and decode miss keys | `update` returns the pre-eviction keys |
| evicting the sinks | memory is bounded and the model is broken | `keep` includes `pos < n_sink` |
| leaving the learned sink out of the denominator | weights sum to 1; gpt-oss outputs and lse disagree | the sink column is concatenated before `logsumexp` |
| query heads tiled over kv heads instead of repeated in blocks | GQA reads the wrong kv head | `rep = np.arange(H) // (H // Hkv)` |
| `positions()` reporting the held keys after eviction | the cache and its mask disagree | `positions()` returns `_last`, the keys `update` returned |
| placing chunk queries by cache index, not position | wrong once sinks are held | `chunk_mask` uses `seen + arange(T)` and stored positions |
| a keep-everything cache masked as if the chunk started at 0 | decode reads the wrong window | `attention_mask` offsets by `cache.seq_len(layer)` |
| scaling by $1/d_h$ instead of $1/\sqrt{d_h}$ | every softmax too flat | `scale = dh**-0.5` |
| a bounded cache given the full-history mask | shapes or windows disagree | `attention_mask` dispatches `SinkWindowCache` to its own `chunk_mask` |

## Where this leads

[The Llama-family model](09-llama-family-model.md) reads `tl_sliding_window`, `tl_sink_tokens`, and `tl_learned_sinks` from `config.json` and places them, so the decoder loads and matches Hugging Face checkpoints. [KV cache and generate](../21-inference/02-kv-cache-and-generate.md) uses this eviction policy for its decode cache. The C kernel's `window` and `sink_logits` in [FlashAttention forward in C](../22-kernels/05-flash-attention-forward.md) are checked against `windowed_attention`, and the decode kernel in [Paged attention for decode in C](../22-kernels/06-paged-attention-decode.md) applies the same window and sink rule.

### Further reading

- Xiao et al., ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453) (2023). Their cache (`mit-han-lab/streaming-llm`, `enable_streaming_llm`) re-rotates cached keys at their cache slots, so relative distances stay inside the trained range.
- Jiang et al., "Mistral 7B" (2023), section 2: the rolling buffer cache, a fixed ring of $W$ slots indexed by position modulo $W$.
- OpenAI, "gpt-oss-120b and gpt-oss-20b model card" (2025), attention sinks; Hugging Face `modeling_gpt_oss.py` alternates sliding and full layers with a sink per head.
- FlashAttention-2 `window_size` (`flash_attn_func(window_size=...)`), and sinks in FlashAttention-3 and vLLM `attention/ops`: the window and the sink inside the tiled kernel.

## Key terms

- **Sliding window**: a limit $W$ on how far back a query reads, counting the query itself.
- **Attention sink**: an early token that a trained model parks unwanted attention on; StreamingLLM keeps the first $S$ tokens visible forever.
- **Learned sink**: a per-head logit added to the softmax denominator and attached to no value.
- **Bounded cache**: a KV cache that evicts to the sinks and the last $W - 1$ positions, holding at most $S + W - 1$ keys.
- **Log-sum-exp (lse)**: the log of a row's softmax denominator, kept so tiled kernels can merge partial results.

## Summary

- One rule decides every key a query reads: causal, and inside the window of $W$ or among the first $S$ sink tokens.
- A cache that keeps only the sinks and the last $W - 1$ keys holds at most $S + W - 1$ keys per layer for a stream of any length, and attention through it equals full attention under the mask.
- Evict after the chunk's queries have read, never before, and track absolute positions so the mask stays right after eviction.
- A learned sink is one extra logit per head in the softmax denominator, attached to no value: the weights then sum to $1 - p_{sink}$.
- One dispatch function gives every layer the correct mask for whatever cache it holds.

## Review questions

1. Why does the cache evict after returning the keys, not before?
   <details><summary>Answer</summary>The chunk's earliest queries may need keys that the chunk's last query no longer does. Evicting first would drop keys still inside the window of the early queries, so chunked prefill and decoding would miss them.</details>
2. Why keep $W - 1$ recent positions and not $W$?
   <details><summary>Answer</summary>The window counts the query itself. The next query brings its own key in the next chunk, so it needs only the $W - 1$ before it, plus the sinks.</details>
3. Why does including the sink logit in the lse make the weights sum to $1 - p_{sink}$?
   <details><summary>Answer</summary>The lse is the log of the full denominator, sink included. Each visible key's weight is $e^{s_j}$ over that denominator, so together they get everything except the sink's share $e^{\sigma_h - \mathrm{lse}}$. The sink has no value, so that share contributes nothing to the output.</details>
4. Why does `chunk_mask` use stored positions instead of `np.arange` over the cache?
   <details><summary>Answer</summary>Once tokens are evicted, the $k$-th held key is not at position $k$. With sinks held at position 0 and recent keys at large positions, an index-based mask would apply the window to the wrong keys.</details>
5. Why does `seq_len` count seen tokens rather than held ones?
   <details><summary>Answer</summary>Callers use it as the position of the next token, for RoPE and for the offset of the mask. The held count is bounded and would make positions wrap back.</details>
6. In the hand example, which keys does query 4 read and why is key 2 excluded?
   <details><summary>Answer</summary>Keys 0, 3, and 4. Key 0 is a sink; keys 3 and 4 are within the window ($4 - 3 = 1 < 2$, $4 - 4 = 0 < 2$). Key 2 has $4 - 2 = 2$, which is not less than $W = 2$.</details>
