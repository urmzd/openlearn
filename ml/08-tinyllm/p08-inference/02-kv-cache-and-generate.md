<!-- ol:module L8.2 -->
# KV cache, incremental decode, incremental UTF-8 detokenizer, generate

## Overview

| | |
|---|---|
| **Module** | `L8.2` · build · Python · Pass 6 · 5 to 7 h |
| **You build** | `python/tinyllm/infer/kvcache.py`: `KVCache` and `LatentCache` (`update`, `seq_len`, `positions`, `mask`, `truncate`, `nbytes`) · `python/tinyllm/infer/generate.py`: `cache_dims`, `IncrementalDecoder` (`push`, `flush`), `Generation`, `generate` |
| **Contract** | [`course/contracts/py/tinyllm/infer/kvcache.pyi`](../../../course/contracts/py/tinyllm/infer/kvcache.pyi) · [`course/contracts/py/tinyllm/infer/generate.pyi`](../../../course/contracts/py/tinyllm/infer/generate.pyi) |
| **Tests** | `course/tests/L8.2/` (what they check: section 4; the test models are in `_tinylm.py`) · your own tests in `python/tests/l8-2-generate/`, rung R5, graded by mutation (threshold 0.80, every pitfall mutant required) |
| **Needs** | `L8.1` (`sample`, `request_rng`, `sampled_entropy`) · `L5.2` (`causal_mask` with `q_offset`) · `L1.2` (the byte-level BPE the detokenizer streams) · `L7.9` (`LlamaForCausalLM`, the model `generate` drives) · `L7.5` (`GQAttention`, whose cache hook this cache implements) · `L7.3` (`RopeSpec`, to build that attention) · `L0.1` (`Tensor`, its input) · reading: `M05.1` KV bytes per token |
| **Used by** | `L8.4`, `L8.5`, `L8.6` (speculative decoding rolls the cache back with `truncate`), and `L10.5` ports the detokenizer to Rust  · `L14.3` Whisper decoding |
| **Milestone** | `MS-L8` (step 1: `generate --cache none,contiguous,paged` give the same greedy text) |
| **Optional depth** | Pope et al., "Efficiently Scaling Transformer Inference" (2022); Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023), section 2; the Unicode Standard, ch. 3.9 (UTF-8 and the replacement of ill-formed sequences) |

## Key Takeaways

- Attention at position $t$ needs the keys and values of positions $0..t$; caching them turns each decode step into one token's work against the stored keys, and the cached logits equal a full recompute to $10^{-5}$ at every step (`test_cached_logits_equal_full_recompute`).
- A decode chunk sits at absolute positions $s, s + 1, \dots$ with $s$ the committed length, and its mask is the causal mask aligned to the bottom right, `causal_mask(T, s + T, q_offset=s)`; both come from the cache before any layer appends (`test_mask_is_l52_causal_with_offset`, `test_seq_len_commits_after_every_layer`).
- The cache returns what it stores: a float16 cache hands back the float16-rounded keys, the new chunk included, so a paged cache in float16 (`L8.3`) can be compared with it exactly (`test_float16_cache_rounds_what_it_stores`).
- Byte-level tokens split characters; the incremental detokenizer emits text only when it no longer ends in U+FFFD, and the pieces always concatenate to `decode(all ids)` (`test_incremental_decode_concat_equals_decode`).
- Streaming must hold back any tail that could still become a stop string, so a client never receives part of the text that is later cut (`test_stop_strings_cut_and_stream_safely`).

## How to work this chapter

```bash
ol start L8.2               # stubs kvcache.py and generate.py into your repo
ol tests L8.2               # read the test catalog first
ol check L8.2               # course tests, then your tests graded by mutation
ol mutate L8.2              # the full mutation grade of your tests
ol check L8.2 --ref-deps    # only if you skipped a dependency
ol diff  L8.2               # after passing: your code against the reference
```

---

## 1. Why now

`L7.9` gives you a Llama-family model that reproduces SmolLM2's logits, and `L8.1` a sampler that picks a token from them. Put the two in a loop and generation works, slowly: to produce token 200 the model re-runs over all 200 positions, so 256 new tokens cost about $256^2/2$ token-forwards instead of 256, and decode speed falls as the text grows. The milestone measures it (`MS-L8` step 4 wants a speedup of at least 5). The fix is the KV cache, and it has to be exact: cached generation must produce the same tokens as recomputation, or every later optimization (paging, prefix sharing, speculative decoding) is measured against a moving target. Text must also stream out as it is generated, and byte-level BPE splits "é" and "🙂" across tokens, so naive per-token decoding prints U+FFFD to the user. This module writes the cache, the loop, and the streaming detokenizer, each checked against its slow, obviously correct counterpart.

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

These are the first cases in section 4: `test_hand_example` and `test_hand_example_detokenizer`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3's cache: returned rows, $s$, positions, mask, bytes | you and the test agree on the definitions |
| `test_hand_example_detokenizer` | unit | section 3's "é" from two byte tokens | streaming text never shows U+FFFD |
| `test_cached_logits_equal_full_recompute` | differential | 16 decode steps, logits within $10^{-5}$, same ids, one prefill then one token per call | the cache is exact |
| `test_chunked_prefill_equals_whole` | differential | chunks of 3, 1, 5, 2 positions equal one forward | chunked prefill (`L10.2`) |
| `test_l75_attention_through_the_cache` | differential | `L7.5`'s attention with this cache as its hook equals the full forward | the seam `L7.9` uses |
| `test_llama_generates_the_same_with_and_without_cache` | differential | `L7.9`'s tiny Llama: same greedy ids, logits within $10^{-5}$ every step | the model of `MS-L8` |
| `test_update_copies_and_checks` | boundary | inputs copied, shapes and capacity checked, nothing changed on error | a caller's buffer reuse is safe |
| `test_seq_len_commits_after_every_layer` | boundary | $s$ moves only after the last layer | every layer places a chunk at the same position |
| `test_mask_is_l52_causal_with_offset` | unit | `mask` equals `L5.2`'s offset causal mask | decode queries see every cached key |
| `test_truncate_rolls_back` | property | truncate then append equals never having appended | `L8.6` rollback |
| `test_float16_cache_rounds_what_it_stores` | unit | half the bytes; stored values returned, new chunk too | `L8.3` compares in float16 |
| `test_float16_generation_stays_close` | differential | float16 cache keeps logits within $2 \times 10^{-2}$ and greedy ids | the memory saving is safe |
| `test_latent_cache` | unit | MLA's latent cache follows the same rules | `L7.6`'s hook |
| `test_incremental_decode_concat_equals_decode` | property | pieces plus flush equal `decode(all)`; every piece is final | the streaming invariant |
| `test_detokenizer_waits_for_whole_characters` | boundary | a 4-byte emoji, a lone continuation byte, a cut-off sequence | ill-formed UTF-8 handled like `decode` |
| `test_generate_greedy_follows_the_script` | unit | ids, text, logprobs, counts, `finish_reason` | the loop end to end |
| `test_eos_stops_and_is_not_emitted` | unit | EOS stops and is not in ids or text | OpenAI semantics |
| `test_stop_strings_cut_and_stream_safely` | boundary | stops across tokens are cut; streamed pieces never contain a stop's beginning | clients see only final text |
| `test_max_len_limits_generation` | boundary | the context limit stops with "length"; bad prompts raise | no position past the model's range |
| `test_seeded_sampling_uses_the_request_stream` | differential | replaying `L8.1` by hand gives the same ids, logprobs, entropy | seeded requests reproduce |
| `test_cache_modes_and_dims` | boundary | "paged" and unknown modes raise; a cache object is used; `cache_dims` reads HF names | `L8.3` and `L7.9` plug in |

### Your tests (rung R5)

Write `python/tests/l8-2-generate/` against the contract only. The oracles are the slow versions: the same model without a cache, `decode(all ids)` for the detokenizer, and `L8.1`'s `sample` replayed by hand for seeded generation. Write a small numpy attention model with rotary positions so that a wrong position or mask changes the logits, and a scripted model whose greedy output you choose, for stops and EOS.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. decode tokens at the wrong position (0, or one too far) | text degrades after the prompt; RoPE angles wrong | `test_cached_logits_equal_full_recompute` (mutants `s01`, `s13`) |
| 2. committing the length after the first layer | layer 1 places the chunk one chunk late | `test_seq_len_commits_after_every_layer` (mutants `s02`, `s03`) |
| 3. a decode mask without the offset | the new token sees only the first key | `test_mask_is_l52_causal_with_offset` (mutant `s04`) |
| 4. `update` returning only the new chunk | attention over one key: the cache is ignored | `test_hand_example` (mutant `s05`) |
| 5. feeding the whole context into the cache each step | keys counted twice, then a capacity error | `test_cached_logits_equal_full_recompute` (mutant `s06`) |
| 6. truncating one layer only | rejected drafts survive in deeper layers | `test_truncate_rolls_back` (mutant `s09`) |
| 7. emitting text that ends in U+FFFD, decoding tokens one by one, or a flush that keeps the window | "Ã©" or U+FFFD on screen | `test_detokenizer_waits_for_whole_characters` (mutants `s10`, `s11`, `s14`) |
| 8. a float16 cache returning the raw new chunk | float16 paged and contiguous caches disagree | `test_float16_cache_rounds_what_it_stores` (mutant `s12`) |
| 9. EOS in the output | an end-of-text marker printed to the user | `test_eos_stops_and_is_not_emitted` (mutant `s16`) |
| 10. a stop string left in, streamed early, or held back one character short | the client sees text that is then removed | `test_stop_strings_cut_and_stream_safely` (mutants `s17`, `s18`, `s19`) |
| 11. the prompt missing from the repetition penalty, or counted as generated | seeded output differs from the engine's | `test_seeded_sampling_uses_the_request_stream` (mutants `s21`, `s22`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.1` | `sample`, `request_rng`, and `sampled_entropy` every step |
| Back | `L5.2` | `causal_mask(T, s + T, q_offset=s)` is the decode mask |
| Back | `L1.2` | the byte-level BPE whose tokens split characters |
| Back | `L7.9` | `LlamaForCausalLM.forward(ids, positions, cache)` is the model `generate` drives; `cache_dims` reads its `config` |
| Back | `L7.5` | `GQAttention` calls `cache.update(layer, k, v)`: this cache is its hook |
| Back | `L7.3` | `RopeSpec` configures that attention's rotary positions |
| Back | `L0.1` | `Tensor` wraps the attention's input |
| Forward | `L14.3` | Whisper decoding calls this module through its declared contract. |
| Forward | `L8.3` | optional: the pure-Python paged cache, compared with this one in float16 |
| Forward | `L8.6` | speculative decoding verifies drafts and rolls back with `truncate` |
| Forward | `L8.5` | quantized models generate through the same loop |
| Forward | `L10.5` | the Rust server ports the incremental detokenizer for SSE |

If you skip this module, `ol check L8.3` stops with `L8.3 needs L8.2`; `--ref-deps` substitutes the reference.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `KVCache` | Hugging Face `StaticCache` | preallocated per-layer buffers for `torch.compile`; `DynamicCache` grows instead | `transformers/cache_utils.py` |
| `IncrementalDecoder` | Hugging Face `tokenizers` `DecodeStream`, vLLM `detokenize_incrementally` | the same prefix/read window, in Rust, with special-token handling | `tokenizers/src/tokenizer/mod.rs`, `vllm/transformers_utils/detokenizer_utils.py` |
| `generate` | vLLM `LLMEngine` | continuous batching: many requests share one forward pass, each with its own cache | `vllm/v1/engine/` |
| stop strings | vLLM `StopChecker` | the same hold-back of partial stop strings in streamed output | `vllm/v1/engine/detokenizer.py` |
