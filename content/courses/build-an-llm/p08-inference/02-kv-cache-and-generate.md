# KV cache, incremental decode, incremental UTF-8 detokenizer, generate

## Overview

| | |
|---|---|
| **Module** | `L8.2` · build · Python · Pass 6 · 2 to 3 h reading |
| **Solution** | [`kvcache.py`](../../../../course/solutions/python/tinyllm/infer/kvcache.py): `KVCache` and `LatentCache` (`update`, `seq_len`, `positions`, `mask`, `truncate`, `nbytes`), the shared `_append` · [`generate.py`](../../../../course/solutions/python/tinyllm/infer/generate.py): `cache_dims`, `IncrementalDecoder` (`push`, `flush`), `Generation`, `_holdback`, `generate`; model oracle tests in [`test_generate_oracle.py`](../../../../course/solutions/learner-tests/L8.2/python/tests/l8-2-generate/test_generate_oracle.py) |
| **Contract** | [`course/contracts/py/tinyllm/infer/kvcache.pyi`](../../../../course/contracts/py/tinyllm/infer/kvcache.pyi) · [`course/contracts/py/tinyllm/infer/generate.pyi`](../../../../course/contracts/py/tinyllm/infer/generate.pyi) |
| **Needs** | `L8.1` (`sample`, `request_rng`, `sampled_entropy`) · `L5.2` (`causal_mask` with `q_offset`) · `L1.2` (the byte-level BPE the detokenizer streams) · `L7.9` (`LlamaForCausalLM`, the model `generate` drives) · `L7.5` (`GQAttention`, whose cache hook this cache implements) · `L7.3` (`RopeSpec`, to build that attention) · `L0.1` (`Tensor`, its input) · reading: `M05.1` KV bytes per token |
| **Used by** | `L8.3` (the paged cache gives the same greedy text as this contiguous one and as no cache at all) · `L8.4`, `L8.5`, `L8.6` (speculative decoding rolls the cache back with `truncate`), and `L10.5` ports the detokenizer to Rust · `L14.3` Whisper decoding |
| **Optional depth** | Pope et al., "Efficiently Scaling Transformer Inference" (2022); Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023), section 2; the Unicode Standard, ch. 3.9 (UTF-8 and the replacement of ill-formed sequences) |

## Key Takeaways

- Attention at position $t$ needs the keys and values of positions $0..t$; caching them turns each decode step into one token's work against the stored keys, and the cached logits equal a full recompute to $10^{-5}$ at every step.
- A decode chunk sits at absolute positions $s, s + 1, \dots$ with $s$ the committed length, and its mask is the causal mask aligned to the bottom right, `causal_mask(T, s + T, q_offset=s)`; both come from the cache before any layer appends.
- The cache returns what it stores: a float16 cache hands back the float16-rounded keys, the new chunk included, so a paged cache in float16 (`L8.3`) can be compared with it exactly.
- Byte-level tokens split characters; the incremental detokenizer emits text only when it no longer ends in U+FFFD, and the pieces always concatenate to `decode(all ids)`.
- Streaming must hold back any tail that could still become a stop string, so a client never receives part of the text that is later cut.

## How to read this chapter

Sections 1 to 3 derive why a KV cache is exact, how a chunk finds its positions and mask, how to roll back, and how to stream byte-level text without half characters, then work the cache and the detokenizer by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution: the cache, the detokenizer, and the generate loop. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L7.9` gives a Llama-family model that reproduces SmolLM2's logits, and `L8.1` a sampler that picks a token from them. Put the two in a loop and generation works, slowly: to produce token 200 the model re-runs over all 200 positions, so 256 new tokens cost about $256^2/2$ token-forwards instead of 256, and decode speed falls as the text grows. On the course's test model the cache is expected to make decoding at least 5 times faster. The fix is the KV cache, and it has to be exact: cached generation must produce the same tokens as recomputation, or every later optimization (paging, prefix sharing, speculative decoding) is measured against a moving target. Text must also stream out as it is generated, and byte-level BPE splits "é" and "🙂" across tokens, so naive per-token decoding prints U+FFFD to the user. This module builds the cache, the loop, and the streaming detokenizer, each checked against its slow, obviously correct counterpart.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $L$ | number of layers | integer |
| $B$ | batch size | integer |
| $H_{kv}$, $d_h$ | key/value heads and head width (GQA, `L7.5`) | integers |
| $T$ | positions in the current chunk (the prompt at prefill, 1 at decode) | integer |
| $s$ | committed length: positions every layer holds | integer |
| $K^{(\ell)}, V^{(\ell)}$ | layer $\ell$'s cached keys and values, $[B, H_{kv}, s, d_h]$ | arrays |
| $N_{\max}$ | preallocated capacity (`max_len`) | integer |
| $q_{\text{off}}$ | absolute position of a chunk's first query, $= s$ | integer |
| ids, $P$ | generated ids and prompt ids | lists |

### 2.1 The generate loop

Encode the prompt to $P$, run the model once over all of it (**prefill**: $T = \lvert P\rvert$, positions $0..\lvert P\rvert - 1$), take the last position's logits. Then repeat: sample with `L8.1` (`sample(logits, p, history=ids, rng, prompt=P)`, `rng = request_rng(seed)`), stop on EOS, a stop string, or the budget, and otherwise run the model on that one token (**decode**: $T = 1$, position $\lvert P\rvert + \lvert\text{ids}\rvert - 1$). The model only ever sees the new tokens; the cache supplies the past. Without a seed the request uses seed 0, so runs are reproducible.

### 2.2 Why caching is exact

In layer $\ell$ the key and value of position $j$ depend only on tokens $0..j$ (causal attention, `L5.2`), so they never change once computed. Caching them and appending new ones produces exactly the same $K$ and $V$ a recompute would build; the query of the new token attends to the same keys with the same mask. The only difference is floating-point order: the recompute evaluates $QK^\top$ for all rows at once, the decode step for one row, and BLAS may sum differently, which is why the comparison is to $10^{-5}$ and not bitwise. Feeding the whole context back into a cache that already holds it double-counts every key and overflows the storage.

### 2.3 Committed length, positions, and the mask

A forward pass appends layer by layer. If $s$ moved as soon as layer 0 appended, layer 1 would place the same chunk one chunk later. So each layer keeps its own fill count and `seq_len()` returns $s = \min_\ell \text{fill}_\ell$: it advances only after the last layer. `seq_len(layer)` returns one layer's count, the convention of `L7.5`'s `ConcatKVCache` that `L7.7`'s mask and `L7.9`'s forward read; between forward passes all counts are equal. A model reads `positions(T)` $= [s, s + T)$ (what RoPE rotates by) and `mask(T)` once, at the start of the pass. Query $i$ of the chunk sits at absolute position $s + i$ and may see keys $j \le s + i$:

$$\text{mask}(T)_{ij} = [\, j \le s + i \,], \qquad 0 \le i < T,\; 0 \le j < s + T,$$

`L5.2`'s `causal_mask(T, s + T, q_offset=s)`. With $q_{\text{off}} = 0$ instead, a decode query sees only key 0.

### 2.4 Rollback

Speculative decoding (`L8.6`) appends draft positions, asks the target to verify them, and keeps only an accepted prefix. `truncate(n)` sets every layer's fill count to $n$; the next append overwrites from $n$, and the result is the cache that never saw the rejected positions. Preallocated storage makes this free: nothing is freed or copied.

### 2.5 Storage dtype

The cache preallocates $2 L B H_{kv} N_{\max} d_h$ elements (`nbytes`), so a 135M-parameter model at 2048 tokens holds 94 MB of float32 per sequence. A float16 cache halves it. Values are converted on the way in, and `update` returns the stored values as float32: the past and the new chunk both rounded. Returning the raw new chunk next to a rounded past mixes two precisions in one attention row, and the paged cache of `L8.3`, which stores float16 blocks, would no longer match this one.

### 2.6 The latent cache

MLA (`L7.6`) caches one compressed latent $c \in \mathbb{R}^{r}$ and one shared RoPE key $k^R \in \mathbb{R}^{d_R}$ per position instead of per-head $K$ and $V$. `LatentCache` follows the same rules (per-layer fill, committed length, rollback) on $[B, T, r]$ and $[B, T, d_R]$ chunks; its `nbytes` is $L B N_{\max} (r + d_R)$ times the item size.

### 2.7 The incremental detokenizer

A byte-level BPE token is a byte string; "é" is `C3 A9`, and a tokenizer may put `C3` and `A9` in different tokens. Decoding a prefix that ends inside a character produces U+FFFD. The decoder keeps the ids and a window: `prefix` (context already emitted) and `read` (end of emitted text). On each push it decodes `ids[prefix:read]` and `ids[prefix:]`; if the second is longer and does not end in U+FFFD, the new text is final: emit the difference and slide the window. Otherwise wait. `flush` emits whatever is left, so a sequence cut off at the end becomes one U+FFFD, and resets. Invariant: the pieces plus the flush equal `decode(all ids)`. A lone continuation byte is never completed; it is emitted as U+FFFD as soon as the next character shows it is invalid.

### 2.8 Stopping

An id in `eos_ids` ends generation and is in neither the ids nor the text. A stop string ends it too, and the text is cut before its first occurrence, even when it spans tokens. When streaming, the last characters of the text may be the beginning of a stop string ("here" while waiting for "here!"): they are held back until the next token proves otherwise, so a client never receives text that is later removed. The budget (`max_tokens`) and the model's context (`max_len`) end generation with `finish_reason = "length"`.

## 3. Worked example by hand

**The cache.** One layer, one kv head of width 2, `max_len` 4. Prefill a chunk of two positions with keys $[[1, 2], [3, 4]]$: storage positions 0 and 1 are filled, the fill count is 2, `update` returns both rows, and $s = 2$; the next token's position is 2. Decode one position with key $[5, 6]$: it is written at position 2, `update` returns all three rows $[[1, 2], [3, 4], [5, 6]]$, and $s = 3$. The next query sits at position 3 and its mask row over keys $0..3$ is $[1, 1, 1, 1]$. Storage is $2 \times 1 \times 1 \times 1 \times 4 \times 2 \times 4 = 64$ bytes.

**The detokenizer.** Push the byte token `C3`: `decode([C3])` is U+FFFD, so nothing is emitted. Push `A9`: `decode([C3, A9])` is "é", longer than the empty window and not ending in U+FFFD, so "é" is emitted and the window moves past both ids. Push "x": "x" is emitted at once. `flush` returns "".

## 4. The interface

```python
# python/tinyllm/infer/kvcache.py
class KVCache:
    def __init__(self, n_layers, n_kv_heads, d_head, max_len, batch=1, dtype=np.float32)
    def seq_len(self, layer=None) -> int                          # committed (min); or one layer's count
    def update(self, layer, k_new, v_new) -> tuple[NDArray, NDArray]   # [B, Hkv, T, dh] in, all stored out
    def positions(self, t_new) -> NDArray                         # [seq_len, seq_len + t_new)
    def mask(self, t_new) -> NDArray                              # causal_mask(t, s + t, q_offset=s)
    def truncate(self, n) -> None
    def nbytes(self) -> int
class LatentCache: ...                                            # the same for MLA's (c, k_rope)

# python/tinyllm/infer/generate.py
def cache_dims(model) -> tuple[int, int, int, int]                # n_layers, n_kv_heads, d_head, max_len
class IncrementalDecoder:
    def __init__(self, tok); def push(self, token_id) -> str; def flush(self) -> str
@dataclass
class Generation: text; ids; logprobs; timings; stats; finish_reason = "length"
def generate(model, tok, prompt, p, cache="contiguous", kv_dtype=np.float32,
             on_text=None, eos_ids=()) -> Generation
```

`model.forward(ids, positions, cache)` returns logits $[B, T, V]$ and calls `cache.update(layer, k, v)` once per layer: the `CausalLM` protocol in the contract, which `L7.9`'s model follows. `cache` is `"none"` (recompute), `"contiguous"` (this cache), or a cache object (the paged cache of `L8.3`).

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example: cache | unit | section 3's cache: returned rows, $s$, positions, mask, bytes | the solution and the reader agree on the definitions |
| the hand example: detokenizer | unit | section 3's "é" from two byte tokens | streaming text never shows U+FFFD |
| cached logits equal a full recompute | differential | 16 decode steps, logits within $10^{-5}$, same ids, one prefill then one token per call | the cache is exact |
| chunked prefill equals whole | differential | chunks of 3, 1, 5, 2 positions equal one forward | chunked prefill (`L10.2`) |
| `L7.5` attention through the cache | differential | `L7.5`'s attention with this cache as its hook equals the full forward | the seam `L7.9` uses |
| the Llama generates the same with and without cache | differential | `L7.9`'s tiny Llama: same greedy ids, logits within $10^{-5}$ every step | the model the system serves |
| `update` copies and checks | boundary | inputs copied, shapes and capacity checked, nothing changed on error | a caller's buffer reuse is safe |
| `seq_len` commits after every layer | boundary | $s$ moves only after the last layer | every layer places a chunk at the same position |
| the mask is `L5.2`'s causal mask with offset | unit | `mask` equals `L5.2`'s offset causal mask | decode queries see every cached key |
| truncate rolls back | property | truncate then append equals never having appended | `L8.6` rollback |
| a float16 cache rounds what it stores | unit | half the bytes; stored values returned, new chunk too | `L8.3` compares in float16 |
| float16 generation stays close | differential | float16 cache keeps logits within $2 \times 10^{-2}$ and greedy ids | the memory saving is safe |
| the latent cache | unit | MLA's latent cache follows the same rules | `L7.6`'s hook |
| incremental pieces concatenate to `decode` | property | pieces plus flush equal `decode(all)`; every piece is final | the streaming invariant |
| the detokenizer waits for whole characters | boundary | a 4-byte emoji, a lone continuation byte, a cut-off sequence | ill-formed UTF-8 handled like `decode` |
| greedy generate follows the script | unit | ids, text, logprobs, counts, `finish_reason` | the loop end to end |
| EOS stops and is not emitted | unit | EOS stops and is not in ids or text | OpenAI semantics |
| stop strings cut and stream safely | boundary | stops across tokens are cut; streamed pieces never contain a stop's beginning | clients see only final text |
| `max_len` limits generation | boundary | the context limit stops with "length"; bad prompts raise | no position past the model's range |
| seeded sampling uses the request stream | differential | replaying `L8.1` by hand gives the same ids, logprobs, entropy | seeded requests reproduce |
| cache modes and dims | boundary | "paged" and unknown modes raise; a cache object is used; `cache_dims` reads HF names | `L8.3` and `L7.9` plug in |

### A model oracle test suite

[`test_generate_oracle.py`](../../../../course/solutions/learner-tests/L8.2/python/tests/l8-2-generate/test_generate_oracle.py) tests against the contract only. Its oracles are the slow versions: the same model without a cache, `decode(all ids)` for the detokenizer, and `L8.1`'s `sample` replayed by hand for seeded generation. It uses a small numpy attention model with rotary positions, so that a wrong position or mask changes the logits, and a scripted model whose greedy output is chosen in advance, for stops and EOS. Each fast component has an obviously correct slow twin, and comparing the two is the whole test strategy.

## 5. The solution, step by step

### 5.1 From first principles

1. Keys and values of past positions never change (section 2.2), so the cache is append-only storage per layer, and `update` must return everything stored so far: that is what the new query attends to.
2. Decoding must not allocate per token, and rollback must be free. So preallocate `[B, H_kv, max_len, d_h]` per layer and track a **fill count**; appending writes at the fill count, truncation just lowers it.
3. One forward pass appends to each layer in turn. The committed length that positions and masks are computed from must not move mid-pass, so it is the minimum fill count over layers (section 2.3), and the model reads it once before the first layer.
4. What a layer attends over must be what is stored, so the new chunk is converted to the storage dtype on the way in and the returned view is the stored prefix, rounded the same way for old and new rows.
5. Validation has to finish before any write, so a failed update leaves the cache unchanged.
6. MLA's latent cache differs only in shapes, so both caches share one append routine parameterised by shapes with `None` marking the time axis.
7. The detokenizer's invariant is "pieces concatenate to `decode(all ids)`". The safe way to emit is: decode a window with left context, compare with what was already emitted, and emit only when the new text is longer and not ending in U+FFFD.
8. Streamed text cannot be taken back, so the loop emits only text that can no longer become part of a stop string: the longest suffix that is a proper prefix of a stop is held back.

### 5.2 The code, annotated

**Appending.**

```python
def _append(cache, layer, stores, chunks, shapes):
    arrs = [np.asarray(c) for c in chunks]
    T = None
    for a, want in zip(arrs, shapes):
        ax = want.index(None)                              # the time axis
        if a.ndim != len(want) or any(w is not None and s != w for s, w in zip(a.shape, want)):
            raise ValueError(...)
        if T is None:
            T = a.shape[ax]
        elif a.shape[ax] != T:
            raise ValueError("chunks disagree on the number of new positions")
    if T < 1:
        raise ValueError("a chunk needs at least one position")
    start = cache._fill[layer]                             # this layer's own count
    if start + T > cache.max_len:
        raise ValueError(...)                              # all checks happen before any write
    out = []
    for store, a, want in zip(stores, arrs, shapes):
        ax = want.index(None)
        idx = [slice(None)] * a.ndim
        idx[ax] = slice(start, start + T)
        store[tuple(idx)] = a.astype(cache.dtype)  # a copy, converted on the way in
        idx[ax] = slice(0, start + T)
        held = store[tuple(idx)]                           # the stored prefix, new rows included
        out.append(held if held.dtype == np.float32 else held.astype(np.float32))
    cache._fill[layer] = start + T
    return tuple(out)
```

Section 3's cache: prefill keys $[[1, 2], [3, 4]]$ go to slots 0 and 1, the fill becomes 2, and the returned prefix is both rows. The decode key $[5, 6]$ goes to slot 2 and the returned prefix is all three rows. Assigning into the preallocated store copies the caller's array, so a caller that reuses its buffer cannot corrupt the cache.

**The cache object.**

```python
class KVCache:
    def __init__(self, n_layers, n_kv_heads, d_head, max_len, batch=1, dtype=np.float32):
        # ... validate sizes >= 1 and a floating dtype
        self._fill = [0] * self.n_layers
        shape = (self.batch, self.n_kv_heads, self.max_len, self.d_head)
        self._k = [np.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]
        self._v = [np.zeros(shape, dtype=self.dtype) for _ in range(self.n_layers)]

    def update(self, layer, k_new, v_new):
        want = (self.batch, self.n_kv_heads, None, self.d_head)
        # ... layer in range
        return _append(self, layer, [self._k[layer], self._v[layer]], [k_new, v_new], [want, want])

    def seq_len(self, layer=None):
        # committed length = the slowest layer; or one layer's own count (L7.5's convention)
        return min(self._fill) if layer is None else self._fill[layer]

    def positions(self, t_new):
        return _positions(self.seq_len(), t_new)          # arange(s, s + t_new)

    def mask(self, t_new):
        return _mask(self.seq_len(), t_new)               # causal_mask(t_new, s + t_new, q_offset=s)

    def truncate(self, n):
        if not 0 <= n <= self.seq_len():
            raise ValueError(...)
        self._fill = [int(n)] * self.n_layers             # every layer, nothing freed or copied

    def nbytes(self):
        return 2 * self.n_layers * self.batch * self.n_kv_heads * self.max_len * self.d_head * self.dtype.itemsize
```

After section 3's two updates $s = 3$: `positions(1)` is `[3]` and `mask(1)` is the row $[1, 1, 1, 1]$ over keys $0..3$. `nbytes` is $2 \cdot 1 \cdot 1 \cdot 1 \cdot 4 \cdot 2 \cdot 4 = 64$. `LatentCache` is the same class with stores `[B, max_len, r]` and `[B, max_len, d_R]` and shapes `(B, None, r)`, `(B, None, d_R)`.

**The detokenizer.**

```python
class IncrementalDecoder:
    def __init__(self, tok):
        self.tok = tok
        self.ids = []
        self.prefix = 0  # start of the context window (already emitted)
        self.read = 0  # end of what has been emitted

    def push(self, token_id):
        self.ids.append(int(token_id))
        before = self.tok.decode(self.ids[self.prefix : self.read])
        now = self.tok.decode(self.ids[self.prefix :])
        if len(now) > len(before) and not now.endswith("�"):
            # The window [prefix, read) gives the decoder its left context;
            # emit only what lies past it, then slide the window forward.
            self.prefix, self.read = self.read, len(self.ids)
            return now[len(before) :]
        return ""                                         # wait: a character is still incomplete

    def flush(self):
        before = self.tok.decode(self.ids[self.prefix : self.read])
        now = self.tok.decode(self.ids[self.prefix :])
        self.ids, self.prefix, self.read = [], 0, 0       # reset for the next request
        return now[len(before) :]                          # a cut-off sequence becomes one U+FFFD
```

Section 3: after `C3`, `before = ""` and `now = "�"`, which ends in U+FFFD, so nothing is emitted. After `A9`, `now = "é"`, longer and complete, so "é" is emitted and the window becomes `prefix = 0, read = 2`. After "x", `before = "é"` (decoded from the window `ids[0:2]`), `now = "éx"`, so "x" is emitted. Decoding a window with left context, rather than only the new ids, is what lets tokenizers that depend on the previous token (leading-space handling) produce the same text as `decode(all ids)`; decoding only the window keeps each push $O(\text{window})$ rather than $O(\text{all ids})$.

**Holding back partial stops.**

```python
def _holdback(text, stops):
    """Length of the longest suffix of text that is a proper prefix of a
    stop string: text that may still become a stop must not be streamed."""
    best = 0
    for s in stops:
        for n in range(min(len(s) - 1, len(text)), 0, -1):   # longest first; proper prefixes only
            if text.endswith(s[:n]):
                best = max(best, n)
                break
    return best
```

Waiting for "here!" with text ending in "here", the hold-back is 4, so "here" is not streamed until the next token shows whether "!" follows.

**The loop.** Setup chooses the cache and the generator:

```python
    p.validate()
    prompt_ids = [int(t) for t in tok.encode(prompt)]
    # ... reject an empty prompt or one longer than the model's max_len
    n_layers, n_kv, d_head, model_max = cache_dims(model)        # from model attrs or an HF config
    limit = min(model_max, len(prompt_ids) + p.max_tokens)        # no slot past what can be used
    if isinstance(cache, str):
        # "none" -> recompute; "contiguous" -> KVCache(n_layers, n_kv, d_head, limit, dtype=kv_dtype)
        # "paged" -> ValueError: pass the L8.3 object; anything else -> ValueError
        ...
    else:
        kv = cache
        if kv.seq_len() != 0:
            raise ValueError("a cache passed to generate must be empty")
    rng = request_rng(p.seed if p.seed is not None else 0)       # seed 0 by default: reproducible
```

Prefill, then one token per step:

```python
    if kv is None:
        logits = _logits_last(model, prompt_ids, np.arange(len(prompt_ids)), None)
    else:
        logits = _logits_last(model, prompt_ids, kv.positions(len(prompt_ids)), kv)   # positions 0..|P|-1
    # ...
    while len(ids) < p.max_tokens:
        tok_id, lp = sample(logits, p, ids, rng, prompt=prompt_ids)   # history = generated, prompt separate
        ents.append(sampled_entropy(logits, p, ids, prompt_ids))
        if tok_id in eos:
            finish = "stop"
            break                                     # EOS is in neither ids nor text
        ids.append(tok_id)
        lps.append(lp)
        ctx.append(tok_id)
        text += dec.push(tok_id)
        hit = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
        if hit >= 0:
            text, finish = text[:hit], "stop"         # cut before the earliest stop
            break
        if on_text is not None:
            ready = len(text) - _holdback(text, stops)
            if ready > sent:
                on_text(text[sent:ready])             # only text that can never be removed
                sent = ready
        if len(ids) >= p.max_tokens or len(ctx) >= limit:
            break
        if kv is None:
            logits = _logits_last(model, ctx, np.arange(len(ctx)), None)   # recompute everything
        else:
            logits = _logits_last(model, [tok_id], kv.positions(1), kv)    # one token, position s
```

The cached branch feeds only the new token, at `kv.positions(1)`, read before the forward appends. The uncached branch feeds the whole context. These are the two sides of section 2.2's equivalence. After the loop, unless a stop was hit, the decoder is flushed (and checked once more for a stop), the remaining text is streamed, and a `Generation` records ids, logprobs, prefill and decode timings, token counts, mean entropy, and tokens per second.

Full solution: [`kvcache.py`](../../../../course/solutions/python/tinyllm/infer/kvcache.py), [`generate.py`](../../../../course/solutions/python/tinyllm/infer/generate.py).

### 5.3 Why this solution

- **Cost.** Without a cache, $n$ new tokens after a prompt of $P$ cost about $\sum_{t} (P + t) \approx nP + n^2/2$ token-forwards; with it, $P + n$. Attention per step is still $O(s)$ against the cache, but the projections and MLP run on one token. On the course's test model the cached loop is several times faster, and the gap grows with length.
- **Preallocated vs growing storage.** Concatenating per step (like `L7.5`'s `ConcatKVCache`) copies the whole cache every token, $O(n^2)$ bytes moved. Preallocation writes $O(1)$ rows per step and makes `truncate` a counter reset. The price is reserving `limit` slots up front, which is why `limit` is capped at prompt plus `max_tokens`.
- **Min fill count, not a single counter.** One counter advanced by layer 0 would misplace layer 1's chunk; a counter advanced by the last layer would need the cache to know which layer is last in every model. The minimum works for any number of layers and any call order.
- **Returning stored values.** A float16 cache that returned the raw float32 chunk would be slightly more accurate for one step, but would make contiguous and paged float16 caches disagree. Agreement between implementations matters more.
- **Hold-back over retraction.** SSE clients cannot un-receive text, so the only correct streaming policy is to delay ambiguous suffixes; the delay is at most the longest stop string minus one character.

### 5.4 Explain it back

1. Why is `seq_len()` the minimum over layers' fill counts?
   <details><summary>Answer</summary>A forward pass appends to layer 0, then 1, and so on. If the committed length moved after layer 0, layer 1 would compute positions and masks one chunk too late. The minimum moves only after the last layer has appended, and between passes all counts are equal.</details>
2. Why does a decode step use `causal_mask(1, s + 1, q_offset=s)` and not `causal_mask(1, s + 1)`?
   <details><summary>Answer</summary>The new query sits at absolute position $s$, so it may see keys $0..s$. Without the offset the mask treats it as position 0 and lets it see only key 0, ignoring the cache.</details>
3. Why can `truncate` be just a counter reset?
   <details><summary>Answer</summary>Storage is preallocated and later appends overwrite from the fill count. Slots past $n$ are never read (returned prefixes end at the fill count), so the cache after truncate-then-append is identical to one that never saw the rejected positions.</details>
4. Why does `push` wait when the decoded text ends in U+FFFD?
   <details><summary>Answer</summary>Byte-level tokens can end in the middle of a UTF-8 character, and a partial character decodes to U+FFFD. Emitting it would show a replacement character that the next token would have completed; waiting keeps every emitted piece final and the concatenation equal to `decode(all ids)`.</details>
5. Why are prompt ids passed to `sample` separately from the generated ids?
   <details><summary>Answer</summary>`L8.1`'s repetition penalty covers both prompt and output, while presence and frequency count only generated tokens. Passing the prompt as history would apply the additive penalties to prompt tokens; omitting it would drop them from the repetition penalty.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. decode tokens at the wrong position (0, or one too far) | text degrades after the prompt; RoPE angles wrong | decodes at `kv.positions(1)`, read before the forward appends |
| 2. committing the length after the first layer | layer 1 places the chunk one chunk late | `seq_len()` is `min(self._fill)` (step 3) |
| 3. a decode mask without the offset | the new token sees only the first key | `mask` is `causal_mask(t, s + t, q_offset=s)` |
| 4. `update` returning only the new chunk | attention over one key: the cache is ignored | `_append` returns `store[0 : start + T]` |
| 5. feeding the whole context into the cache each step | keys counted twice, then a capacity error | the cached branch feeds `[tok_id]` only |
| 6. truncating one layer only | rejected drafts survive in deeper layers | `truncate` resets every layer's count |
| 7. emitting text that ends in U+FFFD, decoding tokens one by one, or a flush that keeps the window | "Ã©" or U+FFFD on screen | `push` decodes a window with left context and waits on U+FFFD; `flush` resets the state |
| 8. a float16 cache returning the raw new chunk | float16 paged and contiguous caches disagree | the chunk is stored first, then the stored prefix is returned (step 4) |
| 9. EOS in the output | an end-of-text marker printed to the user | EOS breaks before `ids.append` and `dec.push` |
| 10. a stop string left in, streamed early, or held back one character short | the client sees text that is then removed | cuts at the earliest stop; streams only up to `len(text) - _holdback(text, stops)` |
| 11. the prompt missing from the repetition penalty, or counted as generated | seeded output differs from the engine's | `sample(logits, p, ids, rng, prompt=prompt_ids)` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.1` | `sample`, `request_rng`, and `sampled_entropy` every step |
| Back | `L5.2` | `causal_mask(T, s + T, q_offset=s)` is the decode mask |
| Back | `L1.2` | the byte-level BPE whose tokens split characters |
| Back | `L7.9` | `LlamaForCausalLM.forward(ids, positions, cache)` is the model `generate` drives; `cache_dims` reads its `config` |
| Back | `L7.5` | `GQAttention` calls `cache.update(layer, k, v)`: this cache is its hook |
| Back | `L7.3` | `RopeSpec` configures that attention's rotary positions |
| Back | `L0.1` | `Tensor` wraps the attention's input |
| Forward | `L14.3` | Whisper decoding runs through this module's contract |
| Forward | `L8.3` | optional: the pure-Python paged cache, compared with this one in float16 |
| Forward | `L8.6` | speculative decoding verifies drafts and rolls back with `truncate` |
| Forward | `L8.5` | quantized models generate through the same loop |
| Forward | `L10.5` | the Rust server ports the incremental detokenizer for SSE |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `KVCache` | Hugging Face `StaticCache` | preallocated per-layer buffers for `torch.compile`; `DynamicCache` grows instead | `transformers/cache_utils.py` |
| `IncrementalDecoder` | Hugging Face `tokenizers` `DecodeStream`, vLLM `detokenize_incrementally` | the same prefix/read window, in Rust, with special-token handling | `tokenizers/src/tokenizer/mod.rs`, `vllm/transformers_utils/detokenizer_utils.py` |
| `generate` | vLLM `LLMEngine` | continuous batching: many requests share one forward pass, each with its own cache | `vllm/v1/engine/` |
| stop strings | vLLM `StopChecker` | the same hold-back of partial stop strings in streamed output | `vllm/v1/engine/detokenizer.py` |
