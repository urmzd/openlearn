# Sliding window, StreamingLLM sinks, learned sinks

## Overview

| | |
|---|---|
| **Module** | `L7.7` · build · Python · Pass 5 · 2 h reading |
| **Solution** | [`window.py`](../../../../course/solutions/python/tinyllm/modern/window.py): `sink_window_mask`, `windowed_attention`, `SinkWindowCache`, `attention_mask`; model oracle tests in [`test_window.py`](../../../../course/solutions/learner-tests/L7.7/python/tests/l7-7-window/test_window.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/window.pyi`](../../../../course/contracts/py/tinyllm/modern/window.pyi) |
| **Needs** | `L5.2` `causal_mask`, `sliding_window_mask` · `M09.2` `logsumexp` · `L7.5` `GQAttention`, `ConcatKVCache` (streaming runs through them) · `L7.3` `RopeSpec` · `L0.1` `Tensor` |
| **Used by** | `L7.9` places `tl_sliding_window` and `tl_sink_tokens`, so the decoder loads and matches Hugging Face checkpoints · later: `L8.2` eviction policy, `L9.3` the C kernel's `window` and `sink_logits` · later: `L9.4` |
| **Optional depth** | Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023); Jiang et al., "Mistral 7B" (2023), section 2; OpenAI, "gpt-oss-120b and gpt-oss-20b model card" (2025), attention sinks |

## Key Takeaways

- One rule decides every key a query reads: causal, and inside the window of $W$ or among the first $S$ sink tokens.
- A cache that keeps only the sinks and the last $W - 1$ keys holds at most $S + W - 1$ keys per layer for a stream of any length, and attention through it equals full attention under the mask.
- Evict after the chunk's queries have read, never before.
- A learned sink is one extra logit per head in the softmax denominator, attached to no value: the weights then sum to $1 - p_{sink}$.

## How to read this chapter

Sections 1 to 3 build the visibility rule, the forgetting cache, and learned sinks from first principles and work a five-token stream by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The GQA layer (`L7.5`) reads every earlier token, so a chat that runs for hours grows its KV cache without bound until the engine runs out of memory, and the attention cost per token grows with it. Mistral reads only the last $W$ tokens; that bounds memory but, as StreamingLLM found, a model trained with full attention falls apart the moment its first tokens leave the window: it had learned to park unwanted attention on them. Keeping a few "sink" tokens fixes it. gpt-oss goes one step further and learns a sink logit per head. `L7.9` loads configs that use these, `L8.2` needs the eviction policy, and `L9.3`'s FlashAttention kernel needs one reference to be checked against. This module is all three.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p_i$ | absolute position of query $i$: `q_offset + i` | `int` |
| $j$ | absolute position of a key | `int` |
| $W$ | `window`: the query and $W - 1$ keys before it | `int` $\ge 1$ |
| $S$ | `n_sink`: tokens $0 .. S-1$ stay visible | `int` $\ge 0$ |
| $s_j$ | score $q \cdot k_j \cdot d_h^{-1/2}$ | `float` |
| $\sigma_h$ | head $h$'s learned sink logit | `float` |
| $\mathrm{lse}$ | $\log\big(\sum_{j\ \mathrm{visible}} e^{s_j} + e^{\sigma_h}\big)$ | `float` |

### 2.1 The visibility rule

$$\mathrm{visible}(i, j) = \big(j \le p_i\big) \wedge \big(j < S \ \vee\ p_i - j < W\big).$$

With $S = 0$ it is `L5.2`'s `sliding_window_mask`; with no window it is the causal mask. The window counts the query itself, so $W = 1$ sees only itself (plus the sinks). The rule uses absolute positions, never indices into a cache, because once a cache evicts, index and position stop agreeing.

### 2.2 A cache that forgets

`SinkWindowCache(S, W)` implements `L7.5`'s hook. On `update(layer, k, v)` it appends the chunk at absolute positions `seen .. seen + T - 1`, returns the held keys followed by the chunk (what the chunk's queries may need), and only then evicts down to positions $< S$ and the last $W - 1$ positions: what the next query needs besides itself. `positions(layer)` reports the absolute positions it returned, and `chunk_mask(layer, T)` builds the rule above for the next chunk against them. Because the keys arrive already rotated at their absolute positions (`L7.3`), attention through this cache is exactly full attention under the rule; StreamingLLM's paper goes further and re-rotates keys at their cache slots (see Going further).

### 2.3 Learned sinks

A softmax must give weight 1 in total; a head with nothing useful to read still has to put that weight somewhere. A learned sink gives it a place that is no token:

$$p_j = \frac{e^{s_j}}{\sum_{k\ \mathrm{visible}} e^{s_k} + e^{\sigma_h}}, \qquad \sum_j p_j = 1 - p_{sink}, \quad p_{sink} = e^{\sigma_h - \mathrm{lse}}.$$

`windowed_attention` is the numpy reference: grouped kv heads (head $h$ reads kv head $h / n_{rep}$), the rule, the sink in the denominator, and the log-sum-exp of every row (FlashAttention keeps it for the backward and for merging chunks, `L9.3`). A row that sees nothing outputs 0, with lse $-\infty$, or the sink logit when there is one.

### 2.4 One function for every layer

`attention_mask(cache, layer, T, window, n_sink)` gives an attention layer its mask whatever cache it was handed: none without a window, the full-sequence rule without a cache, the bounded cache's own `chunk_mask`, or, for a cache that keeps everything (`L7.5`'s `ConcatKVCache`, `L8.2`'s `KVCache`), the rule placed at the chunk's offset `seq_len(layer)`.

## 3. Worked example by hand

Window $W = 2$, one sink, five positions. Query 3 reads key 0 (the sink), key 2 ($3 - 2 = 1 < 2$), and key 3; key 1 is out ($3 - 1 = 2$). The mask:

```
       k0 k1 k2 k3 k4
q0      1  .  .  .  .
q1      1  1  .  .  .
q2      1  1  1  .  .
q3      1  .  1  1  .
q4      1  .  .  1  1
```

Fed one token at a time, `SinkWindowCache(1, 2)` returns the keys at positions `[0]`, `[0, 1]`, `[0, 1, 2]`, `[0, 2, 3]`, `[0, 3, 4]`, each row of the mask, and then holds `[0, 4]`: $S + W - 1 = 2$ keys.

A learned-sink row: one query at position 1, scores $(0, \ln 2)$, values $(4, 8)$, sink logit 0. The denominator is $e^0 + e^{\ln 2} + e^0 = 4$, so the weights are $(1/4, 1/2)$, the sink takes $1/4$, and the output is $4/4 + 8/2 = 5$; $\mathrm{lse} = \ln 4$.

## 4. The interface

```python
def sink_window_mask(Tq, Tk, window, n_sink=0, q_offset=0) -> NDArray          # bool [Tq, Tk]
def windowed_attention(q, k, v, window=None, n_sink=0, sink_logits=None, q_offset=0, scale=None)
    -> tuple[NDArray, NDArray]                                                 # out [B, H, Tq, dv], lse [B, H, Tq]
class SinkWindowCache:                                                         # L7.5's KVCacheHook
    def __init__(self, n_sink, window); update, seq_len, held, positions, chunk_mask
def attention_mask(cache, layer, T, window, n_sink=0) -> Optional[NDArray]
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | section 3: the mask, the cache's steps, one sink row | the solution and the reader agree on the rule |
| agrees with `L5.2` | differential | no sinks: `sliding_window_mask`; no window: `causal_mask` | one rule, many call sites |
| learned-sink goldens | golden | gpt-oss eager attention with sinks: window, a decode chunk, causal; out and lse | `L9.3` is checked against this function |
| Mistral sliding window, streamed | golden | HF Mistral attention, one pass and token by token through the cache | Mistral checkpoints in `L7.9` |
| streaming with sinks equals full attention | differential | decode and chunked prefill through the bounded cache vs full attention | StreamingLLM serving in `L8.2` |
| a plain cache uses absolute positions | differential | a keep-everything cache gets the mask at the chunk's offset | `L8.2`'s `KVCache` |
| the cache is bounded | property | 200 tokens never hold more than $S + W - 1$; returned keys are the visible ones | memory that does not grow |
| `attention_mask` dispatch | unit | the four cases of section 2.4 | every layer gets the right mask |
| sink mass and empty rows | property | weights sum to $1 - p_{sink}$; empty rows give 0, never NaN | learned sinks and long gaps |
| validation | boundary | zero window, negative sinks, head counts, sink vector length | config bugs fail loudly |

### A model oracle test suite

[`test_window.py`](../../../../course/solutions/learner-tests/L7.7/python/tests/l7-7-window/test_window.py) uses three oracles: the visibility rule written as a double loop; attention with a sink computed row by row with `math.exp`; and full attention under the mask as the oracle for streaming through `SinkWindowCache` (decode and chunks, with `L7.5`'s `GQAttention`). It adds the hand sink row, `positions()` after eviction, a keep-everything cache with `attention_mask`, and the validation errors. The streaming oracle is the important one: a bounded cache is correct exactly when it is indistinguishable from an unbounded cache under the mask.

## 5. The solution, step by step

### 5.1 From first principles

1. Section 2.1's rule depends only on absolute positions, so it is one boolean matrix. Its two pieces already exist in `L5.2`: the sliding window (which is causal) and the causal mask. Sinks add "causal and $j < S$", ORed in.
2. The reference attention must handle everything a kernel will: grouped kv heads, the rule, an optional sink logit, and the log-sum-exp of every row. Computing the lse with the sink column included makes the sink part of the denominator automatically, and $p_j = e^{s_j - \mathrm{lse}}$ then sums to $1 - p_{sink}$ with no extra code.
3. A bounded cache must return, for a chunk of $T$ queries, every key any of them may need: what it already holds plus the chunk. Only after that can it evict, and it must keep exactly what the **next** query needs besides itself: the sinks and the last $W - 1$ positions.
4. After eviction cache index and position disagree, so the cache records the absolute position of everything it holds, and builds the next chunk's mask from those positions, not from indices.
5. Different caches need different masks (none, a fresh sequence, the bounded cache, a keep-everything cache at an offset). One dispatch function lets every attention layer ask for its mask without knowing which cache it has.

### 5.2 The code, annotated

**The rule.**

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

On section 3 ($W = 2$, $S = 1$): `recent` is the diagonal and the subdiagonal; `sinks` adds column 0 for every query. That is the printed mask.

**The reference attention.**

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

The `np.where` handles empty rows: with no visible key and no sink, lse is $-\infty$ and $s - \mathrm{lse}$ is $-\infty - (-\infty) = $ NaN; the guard replaces it by 0 so the output is 0. With a sink, an empty row has finite lse $= \sigma_h$ and every $p_j$ is 0. On section 3's sink row, $\mathrm{lse} = \ln(1 + 2 + 1) = \ln 4$, $p = (1/4, 2/4)$, output $4 \cdot 1/4 + 8 \cdot 1/2 = 5$.

Returning the lse is what lets `L9.3`'s tiled kernel be checked against this function row by row: FlashAttention merges tiles through their lse.

**The bounded cache.**

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

Trace section 3, one token at a time with $S = 1$, $W = 2$. Step 3 (position 3): held is `[0, 2]`, so it returns `[0, 2, 3]`; `keep` is `pos < 1` or `pos >= 4 - 1 = 3`, holding `[0, 3]`. Step 4 returns `[0, 3, 4]` and holds `[0, 4]`. The held count never exceeds $S + W - 1 = 2$.

`seq_len` returns how many tokens have been **seen**, not held: that is the position of the next token, which RoPE and the mask need.

**The chunk mask.**

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

This is section 2.1's rule evaluated at positions, column for column aligned with what the next `update` returns. It is called before `update`, so it describes the cache's state at the moment the chunk arrives.

**Dispatch.**

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

Full solution: [`window.py`](../../../../course/solutions/python/tinyllm/modern/window.py).

### 5.3 Why this solution

- **Memory.** A stream of $N$ tokens holds at most $S + W - 1$ keys per layer, against $N$ for full attention; per-step attention cost drops from $O(N)$ to $O(S + W)$.
- **Positions, not indices.** Recording absolute positions costs one int per held key and is what makes the mask right after eviction. Index-based masks look correct until the first sink is held.
- **Concatenate-then-evict, not a ring buffer.** Mistral's rolling buffer writes into $W$ fixed slots, which avoids copies but scrambles key order and cannot hold sinks without a second buffer. The concatenating cache costs a copy per update and keeps keys in position order, which keeps the mask simple and the reference easy to check.
- **No re-rotation.** Keys are cached already rotated at their absolute positions, so attention through the cache equals full attention under the mask exactly. StreamingLLM re-rotates keys at their cache slots to keep distances within the trained range; that changes the outputs and is listed in Going further.
- **float64 reference.** `windowed_attention` is the oracle for the C kernel, so it computes in float64 and returns the lse; speed is not its job.

### 5.4 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a window of $W$ keys before the query instead of $W$ including it | one key too many; Mistral disagrees after position $W$ | `p - j < window`, the same rule as `L5.2` |
| 2. sink tokens falling out of the window like any other key | the stream degrades exactly when token 0 leaves the window | the mask ORs in `j < n_sink` (step 1) |
| 3. evicting before the chunk's queries read | chunked prefill and decode miss keys | `update` returns the pre-eviction keys (step 3) |
| 4. evicting the sinks | memory is bounded and the model is broken | `keep` includes `pos < n_sink` |
| 5. leaving the learned sink out of the denominator | weights sum to 1; gpt-oss outputs and lse disagree | the sink column is concatenated before `logsumexp` (step 2) |
| query heads tiled over kv heads instead of repeated in blocks | GQA reads the wrong kv head | `rep = np.arange(H) // (H // Hkv)` |
| `positions()` reporting the held keys after eviction | the cache and its mask disagree | `positions()` returns `_last`, the keys `update` returned |
| placing chunk queries by cache index, not position | wrong once sinks are held | `chunk_mask` uses `seen + arange(T)` and stored positions (step 4) |
| a keep-everything cache masked as if the chunk started at 0 | decode reads the wrong window | `attention_mask` offsets by `cache.seq_len(layer)` |
| scaling by $1/d_h$ instead of $1/\sqrt{d_h}$ | every softmax too flat | `scale = dh**-0.5` |
| a bounded cache given the full-history mask | shapes or windows disagree | `attention_mask` dispatches `SinkWindowCache` to its own `chunk_mask` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.2` | the causal and sliding-window masks this rule extends |
| Back | `M09.2` | `logsumexp` for each row's lse |
| Back | `L7.5` | `GQAttention` takes the mask and the cache hook; `ConcatKVCache` keeps everything |
| Back | `L7.3` | keys arrive rotated at absolute positions |
| Back | `L0.1` | `Tensor` inputs in the streaming runs |
| Forward | `L7.9` | `tl_sliding_window`, `tl_sink_tokens`, and `tl_learned_sinks` in config.json |
| Forward | `L8.2` | the eviction policy of the decode cache |
| Forward | `L9.3` | the C kernel's `window` and `sink_logits`, checked against `windowed_attention` |
| Forward | `L9.4` | the C paged-attention decode kernel applies the same window and sink rule |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SinkWindowCache` | StreamingLLM's cache | re-rotates cached keys at their cache slots, so relative distances stay inside the trained range | `mit-han-lab/streaming-llm`, `enable_streaming_llm` |
| sliding window | Mistral's rolling buffer cache | a fixed ring buffer of $W$ slots indexed by position modulo $W$ | Mistral 7B paper, section 2 |
| `windowed_attention` | FlashAttention-2 `window_size`, sinks in FlashAttention-3 and vLLM | the window and the sink inside the tiled kernel | `flash_attn_func(window_size=...)`, vLLM `attention/ops` |
| learned sinks | gpt-oss | alternating sliding and full layers, a sink per head | HF `modeling_gpt_oss.py` |
