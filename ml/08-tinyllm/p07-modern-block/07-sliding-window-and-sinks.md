<!-- ol:module L7.7 -->
# Sliding window, StreamingLLM sinks, learned sinks

## Overview

| | |
|---|---|
| **Module** | `L7.7` · build · Python · Pass 5 · 3 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/window.py`: `sink_window_mask`, `windowed_attention`, `SinkWindowCache`, `attention_mask`; and your own oracle tests in `python/tests/l7-7-window/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/window.pyi`](../../../course/contracts/py/tinyllm/modern/window.pyi) |
| **Tests** | `course/tests/L7.7/test_window.py` (what they check: section 4), golden values from transformers 5.19.0 `MistralAttention` (sliding window) and gpt-oss's eager attention with learned sinks in `course/fixtures/L7.7/window_hf.npz` (`course/oracle/L7.7/window_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L5.2` `causal_mask`, `sliding_window_mask` · `M09.2` `logsumexp` · `L7.5` `GQAttention`, `ConcatKVCache` (the tests stream through them) · `L7.3` `RopeSpec` · `L0.1` `Tensor` (or `--ref-deps`) |
| **Used by** | `L7.9` places `tl_sliding_window` and `tl_sink_tokens` · later: `L8.2` eviction policy, `L9.3` the C kernel's `window` and `sink_logits` · later: `L9.4` |
| **Milestone** | `MS-L7` (your decoder loads and matches Hugging Face checkpoints) |
| **Optional depth** | Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023); Jiang et al., "Mistral 7B" (2023), section 2; OpenAI, "gpt-oss-120b and gpt-oss-20b model card" (2025), attention sinks |

## Key Takeaways

- One rule decides every key a query reads: causal, and inside the window of $W$ or among the first $S$ sink tokens (`test_hand_example`, `test_mask_agrees_with_l5_2`).
- A cache that keeps only the sinks and the last $W - 1$ keys holds at most $S + W - 1$ keys per layer for a stream of any length (`test_cache_is_bounded`), and attention through it equals full attention under the mask (`test_streaming_with_sinks_equals_full_attention`).
- Evict after the chunk's queries have read, never before (`test_streaming_with_sinks_equals_full_attention`).
- A learned sink is one extra logit per head in the softmax denominator, attached to no value: the weights then sum to $1 - p_{sink}$ (`test_sink_mass_and_empty_rows`).

## How to work this chapter

```bash
ol start L7.7              # stubs window.py; prints your test path and rung (R5)
ol tests L7.7              # the course tests
# write your oracle tests in python/tests/l7-7-window/, then:
ol check L7.7              # course tests and the mutation grade of your tests
ol diff  L7.7              # after passing: your code against the reference
```

---

## 1. Why now

Your GQA layer (`L7.5`) reads every earlier token, so a chat that runs for hours grows its KV cache without bound until the engine runs out of memory, and the attention cost per token grows with it. Mistral reads only the last $W$ tokens; that bounds memory but, as StreamingLLM found, a model trained with full attention falls apart the moment its first tokens leave the window: it had learned to park unwanted attention on them. Keeping a few "sink" tokens fixes it. gpt-oss goes one step further and learns a sink logit per head. `L7.9` loads configs that use these, `L8.2` needs the eviction policy, and `L9.3`'s FlashAttention kernel needs one reference to be checked against. This module is all three.

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

A learned-sink row: one query at position 1, scores $(0, \ln 2)$, values $(4, 8)$, sink logit 0. The denominator is $e^0 + e^{\ln 2} + e^0 = 4$, so the weights are $(1/4, 1/2)$, the sink takes $1/4$, and the output is $4/4 + 8/2 = 5$; $\mathrm{lse} = \ln 4$. This is `test_hand_example`.

## 4. The interface

```python
def sink_window_mask(Tq, Tk, window, n_sink=0, q_offset=0) -> NDArray          # bool [Tq, Tk]
def windowed_attention(q, k, v, window=None, n_sink=0, sink_logits=None, q_offset=0, scale=None)
    -> tuple[NDArray, NDArray]                                                 # out [B, H, Tq, dv], lse [B, H, Tq]
class SinkWindowCache:                                                         # L7.5's KVCacheHook
    def __init__(self, n_sink, window); update, seq_len, held, positions, chunk_mask
def attention_mask(cache, layer, T, window, n_sink=0) -> Optional[NDArray]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3: the mask, the cache's steps, one sink row | you and the test agree on the rule |
| `test_mask_agrees_with_l5_2` | differential | no sinks: `sliding_window_mask`; no window: `causal_mask` | one rule, many call sites |
| `test_learned_sinks_golden` | golden | gpt-oss eager attention with sinks: window, a decode chunk, causal; out and lse | `L9.3` is checked against this function |
| `test_mistral_sliding_window_golden_streamed` | golden | HF Mistral attention, one pass and token by token through the cache | Mistral checkpoints in `L7.9` |
| `test_streaming_with_sinks_equals_full_attention` | differential | decode and chunked prefill through the bounded cache vs full attention | StreamingLLM serving in `L8.2` |
| `test_plain_cache_uses_absolute_positions` | differential | a keep-everything cache gets the mask at the chunk's offset | `L8.2`'s `KVCache` |
| `test_cache_is_bounded` | property | 200 tokens never hold more than $S + W - 1$; returned keys are the visible ones | memory that does not grow |
| `test_attention_mask_dispatch` | unit | the four cases of section 2.4 | every layer gets the right mask |
| `test_sink_mass_and_empty_rows` | property | weights sum to $1 - p_{sink}$; empty rows give 0, never NaN | learned sinks and long gaps |
| `test_validation` | boundary | zero window, negative sinks, head counts, sink vector length | config bugs fail loudly |

### Your graded tests (rung R5)

Your oracles: the visibility rule written as a double loop; attention with a sink computed row by row with `math.exp`; and full attention under the mask as the oracle for streaming through `SinkWindowCache` (decode and chunks, with `L7.5`'s `GQAttention`). Add the hand sink row, `positions()` after eviction, a keep-everything cache with `attention_mask`, and the validation errors. Import only `tinyllm.modern.window`, `tinyllm.modern.gqa`, `tinyllm.modern.rope`, and `tinyllm.autograd.tensor`. `ol check L7.7` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a window of $W$ keys before the query instead of $W$ including it | one key too many; Mistral disagrees after position $W$ | `test_mask_agrees_with_l5_2`, `test_learned_sinks_golden` (mutant `s01`) |
| 2. sink tokens falling out of the window like any other key | the stream degrades exactly when token 0 leaves the window | `test_streaming_with_sinks_equals_full_attention` (mutant `s02`) |
| 3. evicting before the chunk's queries read | chunked prefill and decode miss keys | `test_mistral_sliding_window_golden_streamed`, `test_streaming_with_sinks_equals_full_attention` (mutant `s03`) |
| 4. evicting the sinks | memory is bounded and the model is broken | `test_cache_is_bounded`, `test_streaming_with_sinks_equals_full_attention` (mutant `s04`) |
| 5. leaving the learned sink out of the denominator | weights sum to 1; gpt-oss outputs and lse disagree | `test_learned_sinks_golden`, `test_sink_mass_and_empty_rows` (mutant `s05`) |
| query heads tiled over kv heads instead of repeated in blocks | GQA reads the wrong kv head | `test_learned_sinks_golden` (mutant `s06`) |
| `positions()` reporting the held keys after eviction | the cache and its mask disagree | `test_cache_is_bounded`, `test_hand_example` (mutant `s07`) |
| placing chunk queries by cache index, not position | wrong once sinks are held | `test_streaming_with_sinks_equals_full_attention` (mutant `s08`) |
| a keep-everything cache masked as if the chunk started at 0 | decode reads the wrong window | `test_plain_cache_uses_absolute_positions` (mutant `s09`) |
| scaling by $1/d_h$ instead of $1/\sqrt{d_h}$ | every softmax too flat | `test_learned_sinks_golden` (mutant `s10`) |
| a bounded cache given the full-history mask | shapes or windows disagree | `test_attention_mask_dispatch` (mutant `s11`) |

## 6. Where it's used next
| Forward | `L9.4` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.2` | the causal and sliding-window masks this rule extends |
| Back | `M09.2` | `logsumexp` for each row's lse |
| Back | `L7.5` | `GQAttention` takes the mask and the cache hook; `ConcatKVCache` keeps everything |
| Back | `L7.3` | keys arrive rotated at absolute positions |
| Back | `L0.1` | `Tensor` inputs in the streaming tests |
| Forward | `L7.9` | `tl_sliding_window`, `tl_sink_tokens`, and `tl_learned_sinks` in config.json |
| Forward | `L8.2` | the eviction policy of the decode cache |
| Forward | `L9.3` | the C kernel's `window` and `sink_logits`, checked against `windowed_attention` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SinkWindowCache` | StreamingLLM's cache | re-rotates cached keys at their cache slots, so relative distances stay inside the trained range | `mit-han-lab/streaming-llm`, `enable_streaming_llm` |
| sliding window | Mistral's rolling buffer cache | a fixed ring buffer of $W$ slots indexed by position modulo $W$ | Mistral 7B paper, section 2 |
| `windowed_attention` | FlashAttention-2 `window_size`, sinks in FlashAttention-3 and vLLM | the window and the sink inside the tiled kernel | `flash_attn_func(window_size=...)`, vLLM `attention/ops` |
| learned sinks | gpt-oss | alternating sliding and full layers, a sink per head | HF `modeling_gpt_oss.py` |
