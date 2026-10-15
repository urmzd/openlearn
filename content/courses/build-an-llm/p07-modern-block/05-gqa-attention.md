<!-- ol:module L7.5 -->
# MQA/GQA attention with cache hook, window, learned sinks

## Overview

| | |
|---|---|
| **Module** | `L7.5` · build · Python · Pass 5 · 4 to 5 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/gqa.py`: `GQAttention`, `repeat_kv`, the `KVCacheHook` protocol and its smallest implementation `ConcatKVCache`; and your own oracle tests in `python/tests/l7-5-gqa/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/gqa.pyi`](../../../../course/contracts/py/tinyllm/modern/gqa.pyi) |
| **Tests** | `course/tests/L7.5/test_gqa.py` (what they check: section 4), golden values from transformers 5.19.0 `LlamaAttention`, `Qwen2Attention`, and `GptOssAttention` in `course/fixtures/L7.5/gqa_hf.npz` (`course/oracle/L7.5/gqa_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L7.3` `rope_cos_sin`, `apply_rope`, `RopeSpec` · `L0.4` `Linear`, `Module` · `L0.2` `F.matmul`, `F.masked_fill`, `F.softmax`, `F.concat` · `L0.1` `Tensor` · `M06.3` `PCG32` (default initialization) · `M05.1` `kv_bytes_per_token` (the tests check the cache against it) · reading: `M09.2` (the masked softmax), Part 5's scaled dot-product attention (or `--ref-deps`) |
| **Used by** | `L7.7` sliding-window and sink attention builds on it · `L7.9` every Llama attention layer (SmolLM2: 9 query heads over 3 kv heads) · `L8.2` the `KVCache` behind the hook · later: `L9.3` and `L9.4` the C attention kernels take `Hkv` |
| **Milestone** | `MS-L7` (SmolLM2-135M logits match Hugging Face) |
| **Optional depth** | Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019); Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023); Beltagy et al., "Longformer" (2020), section 3.1; Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023); OpenAI, "gpt-oss-120b and gpt-oss-20b model card" (2025) |

## Key Takeaways

- $H$ query heads share $H_{kv}$ key/value heads; query head $h$ reads kv head $\lfloor h / n_\text{rep} \rfloor$, each kv head repeated $n_\text{rep}$ times **in a row** (`test_repeat_kv_repeats_each_head_in_a_row`, `test_gqa_equals_mha_with_repeated_kv_weights`).
- The cache hook receives only the $H_{kv}$ heads, so the KV cache shrinks by $H / H_{kv}$: exactly `M05.1`'s `kv_bytes_per_token` (`test_cache_holds_kv_heads_only`).
- With a cache, the new queries are the **last** $T$ of $T_k$ keys: the causal mask uses that offset, and keys are rotated with their own positions before they are cached (`test_cache_chunks_equal_full_forward`).
- A sliding window of $W$ lets query $t$ read keys $t - W + 1 .. t$ (`test_window_reach`).
- A learned sink is one extra logit per head in every softmax row, dropped afterwards: a head may attend to nothing (`test_sinks_take_weight_from_every_key`).

## How to work this chapter

```bash
ol start L7.5              # stubs gqa.py; prints your test path and rung (R5)
ol tests L7.5              # the course tests
# write your oracle tests in python/tests/l7-5-gqa/, then:
ol check L7.5              # course tests and the mutation grade of your tests
ol diff  L7.5              # after passing: your code against the reference
```

---

## 1. Why now

Your 2017 multi-head attention gives every query head its own key and value head. At inference the engine caches keys and values for every past token (`L8.2`, and the Rust engine in Pass 7), so cache memory, not compute, limits how many requests run at once. Open SmolLM2-135M's checkpoint: `k_proj.weight` is $192 \times 576$, a third of `q_proj`'s $576 \times 576$. It has 9 query heads but only 3 key/value heads, and your MHA cannot load it. Newer models add two more things to the same layer: a **sliding window** (Mistral, gpt-oss) so the cache stops growing, and **learned sinks** (gpt-oss) so a head can decline to attend. This module builds that attention layer, with the hook through which the KV cache will plug in.

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

One kv head served two query heads that attend in opposite directions. This is `test_hand_example`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3 | you and the test agree on the formula |
| `test_golden_hf` | golden | Llama GQA, an extra mask, Qwen2 biases, gpt-oss sinks and window with gapped per-row positions: outputs and every gradient | SmolLM2 in MS-L7 |
| `test_gradcheck_every_parameter` | gradcheck | float64, biases, window, sinks | every weight trains |
| `test_repeat_kv_repeats_each_head_in_a_row` | unit | $(a, b) \to (a, a, a, b, b, b)$, summed gradients | checkpoints' head grouping |
| `test_gqa_equals_mha_with_repeated_kv_weights` | differential | GQA = MHA with grouped kv weights | what GQA is |
| `test_cache_chunks_equal_full_forward` | differential | prefill then decode through the cache, with and without window and sinks | `L8.2`'s cached generation |
| `test_cache_holds_kv_heads_only` | property | the hook gets $H_{kv}$ heads, bytes = `kv_bytes_per_token` | `L10.2` admission by KV bytes |
| `test_concat_cache` | unit | appends, copies, rejects bad chunks | the reference hook |
| `test_causality_bitwise` | property | future tokens change nothing, bit for bit | a decoder |
| `test_window_reach` | boundary | key $t - W$ invisible, $t - W + 1$ visible | Mistral-style windows, `L7.7` |
| `test_sinks_take_weight_from_every_key` | property | large sink: output 0; very negative: plain attention | gpt-oss heads |
| `test_fully_masked_row_is_zero` | boundary | zeros, no NaN, other rows intact | padded batches |
| `test_positions_shift_and_per_row` | property | shift invariance; per-row positions equal separate calls | left padding, continued caches |
| `test_attention_scaling_squares_into_the_scores` | property | YaRN scaling = $s^2$ on the scores | `rope_scaling` in `L7.9` |
| `test_parameter_names_shapes_and_draw_order` | unit | HF keys, no `o_proj` bias, `d_head` default, draw order | the safetensors keys |
| `test_validation` | boundary | heads, `d_head`, window, rotary width, input, positions, mask | wiring bugs fail loudly |

### Your graded tests (rung R5)

Your oracle is the whole layer written out in numpy float64 with the module's own `state_dict()`: the projections, RoPE as complex multiplication, per head the scores of kv head $\lfloor h / n_\text{rep} \rfloor$, the causal and window mask, the optional sink column, the softmax, the values, and `o_proj`. Compare `forward` with and without window and sinks, decode through `ConcatKVCache` against the full forward, and check the head order of `repeat_kv`, the mask, per-row positions, the parameter names, the attention scaling, the cache's copies, and the validation. Import only `tinyllm.modern.gqa`, `tinyllm.modern.rope`, `tinyllm.autograd.tensor`, and `tinyllm.autograd.functional`. `ol check L7.5` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. tiling kv heads instead of repeating them | query head 1 reads kv head 1, not 0: loads fine, wrong logits | `test_repeat_kv_repeats_each_head_in_a_row`, `test_golden_hf` (mutant `s01`) |
| 2. scaling by $1/\sqrt{d}$ instead of $1/\sqrt{d_h}$ | attention too flat whenever $d \ne d_h$ | `test_golden_hf` (mutant `s02`) |
| 3. a causal mask that ignores the cached prefix | decoded tokens see only the first keys | `test_cache_chunks_equal_full_forward` (mutant `s03`) |
| 4. adding the sink to every score | no effect at all (softmax is shift invariant) | `test_sinks_take_weight_from_every_key` (mutant `s04`) |
| 5. a window one key too wide ($\le W$) | one extra key per query; outputs drift from the reference | `test_window_reach` (mutant `s05`) |
| 6. rotating keys after the cache | every cached key re-rotated to the newest position: generation degrades after the prompt | `test_cache_chunks_equal_full_forward` (mutant `s06`) |
| mask polarity inverted | only padding is read | `test_fully_masked_row_is_zero` (mutant `s07`) |
| per-row positions ignored | left-padded rows rotated wrongly | `test_positions_shift_and_per_row` (mutant `s08`) |
| a bias on `o_proj` with `qkv_bias` | Qwen2 checkpoints fail to load | `test_parameter_names_shapes_and_draw_order` (mutant `s09`) |
| caching the repeated heads | the cache is $n_\text{rep}$ times too large | `test_cache_holds_kv_heads_only` (mutant `s10`) |
| softmax over the wrong axis | weights sum to 1 over queries | `test_hand_example` (mutant `s11`) |
| sinks as constants | the sinks never train | `test_gradcheck_every_parameter` (mutant `s12`) |
| ignoring the attention scaling | YaRN models lose their temperature | `test_attention_scaling_squares_into_the_scores` (mutant `s13`) |

## 6. Where it's used next

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

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `GQAttention` | HF `LlamaAttention`, `GptOssAttention` | interchangeable attention backends (eager, SDPA, FlashAttention) | `transformers/models/llama/modeling_llama.py` |
| `repeat_kv` | FlashAttention, vLLM paged attention | no repetition in memory: the kernel indexes kv head `h / n_rep` directly | FlashAttention-2 `num_heads_k`; vLLM `paged_attention` |
| the cache hook | vLLM PagedAttention, SGLang RadixAttention | block tables, prefix sharing, eviction | `L8.3`, `L8.4` in this course |
| learned sinks | StreamingLLM | keeps the first tokens' KV forever so a sliding window stays stable | `L7.7` in this course; Xiao et al. 2023 |
