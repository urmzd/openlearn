<!-- ol:module L9.3 -->
# FlashAttention forward in C

## Overview

| | |
|---|---|
| **Module** | `L9.3` · side · C · Pass 6 · 6 to 8 h |
| **You build** | `c/src/kernels/flash_attn.c`: `tl_flash_attn_fwd_f32`, attention for a whole prefill (or a chunk of one) that never stores the score matrix: GQA head sharing, causal masking at absolute positions (`q_offset`), a sliding window, learned sinks, and the log-sum-exp of every row, with scratch from your arena and threads from your pool |
| **Contract** | [`course/contracts/c/include/tinyllm/attention.h`](../../../../course/contracts/c/include/tinyllm/attention.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) (rule 10, invariance) |
| **Tests** | `course/tests/L9.3/`: `test_flash_attn.c` (C, under ASan and UBSan, and ThreadSanitizer for the pool case, against a naive oracle built on your `L9.2` softmax) and shared file fixtures (what they check: section 4) |
| **Needs** | `rt.02` the loader · `rt.02` the arena · `rt.03` the pool · `M09.6` `tl_expf` · `L9.2` `tl_softmax_f32` (the C oracle) · `L7.7` `windowed_attention` · `L5.1` `sdpa_forward` (or `--ref-deps`). Reading: `L5.2` (mask flags), `L7.5` (GQA) |
| **Used by** | `L9.4` checks paged decode against this kernel, bit for bit · `L10.3` relies on the chunk invariance proved here |
| **Milestone** | `MS-L9` |
| **Optional depth** | Dao et al., "FlashAttention" (2022) and "FlashAttention-2" (2023); Milakov and Gimelshein, "Online normalizer calculation for softmax" (2018); Rabe and Staats, "Self-attention Does Not Need $O(n^2)$ Memory" (2021) |

## Key Takeaways

- **Attention can be computed one tile of keys at a time** with the online softmax of `L9.2`, extended to the output: each row keeps a running maximum $m$, denominator $\ell$, and output $a$, and rescales $\ell$ and $a$ by $e^{m_{\text{old}} - m_{\text{new}}}$ when the maximum grows (`gqa_causal_window_match_naive`).
- **Memory stops growing with the context:** scratch is $B_r$ rows of state and one tile of $B_c$ scores per worker, so the arena's high-water mark is the same for 16 keys and 1000 (`arena_rewound_and_scratch_independent_of_tk`).
- **Visibility is per row and per absolute position:** query $i$ sits at $p = q_{\text{offset}} + i$ and sees key $j$ when $j \le p$ (causal) and $j > p - w$ (window). Rows of one query tile may see different keys (`no_visible_key_gives_zeros_and_minus_inf`).
- **Key tiles are aligned to absolute positions** ($[tB_c, (t+1)B_c)$), so a row's arithmetic is the same whether the prompt arrives in one call or in chunks, in any batch, with any query tile size: bitwise (`chunk_invariant_with_q_offset`, `query_tile_and_batch_invariant`).
- **A sink is one extra score with no value:** it joins the denominator and the lse, and the weights then sum to $1 - p_{\text{sink}}$ (`sinks_join_the_denominator`).

## How to work this chapter

```bash
ol start L9.3              # stubs c/src/kernels/flash_attn.c into your repo
ol tests L9.3              # read the test catalog first
ol check L9.3              # exit code is the verdict
ol check L9.3 --ref-deps   # only if a runtime piece, L9.2, L7.7, or L5.1 is not passing yet
ol parity flash.fwd        # against the float64 golden
ol diff  L9.3              # after passing: your code against the reference
```

---

## 1. Why now

Your numpy attention (`L5.1`, `L7.5`, `L7.7`) forms the score matrix $S = QK^\top$ for every head: $T_q \times T_k$ floats, then a softmax over each row, then $PV$. For a 2048-token prompt that is 4 million floats (16 MB) per head per layer, written once and read twice, and most of the prefill time goes to moving it rather than computing it. This standalone C exercise makes memory use independent of prompt length. `L10.3` additionally splits long prompts into chunks so prefill does not stall decode; chunk invariance ensures every query row's output is stable across split choices.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $q_i, k_j, v_j \in \mathbb{R}^D$ | query row $i$, key and value at position $j$ | `float[D]` |
| $B, H, H_{kv}$ | batch, query heads, KV heads ($H \bmod H_{kv} = 0$) | `int64_t` |
| $T_q, T_k$ | query rows in this call, keys in the cache | `int64_t` |
| $p_i = q_{\text{offset}} + i$ | the absolute position of query row $i$ | `int64_t` |
| $\sigma$ | `scale`, usually $D^{-1/2}$ | `float` |
| $s_{ij} = \sigma\, q_i \cdot k_j$ | a score | `float` |
| $V_i$ | the keys row $i$ may see | set |
| $w$ | the window ($w \le 0$: none) | `int64_t` |
| $z_h$ | the sink logit of head $h$ | `float` |
| $B_r, B_c$ | query and key tile sizes | `int64_t` |
| $m_i, \ell_i, a_i$ | running maximum, denominator, and output of row $i$ | `float`, `float`, `float[D]` |
| $\mathrm{lse}_i$ | $\log\big(\sum_{j \in V_i} e^{s_{ij}} + e^{z_h}\big)$ | `float` |

### 2.1 What is computed

For every sequence $b$, head $h$, and query row $i$:

$$o_i = \frac{\sum_{j \in V_i} e^{s_{ij}}\, v_j}{\sum_{j \in V_i} e^{s_{ij}} + e^{z_h}}, \qquad V_i = \{\, j < T_k : (\text{causal} = 0 \text{ or } j \le p_i) \text{ and } (w \le 0 \text{ or } j > p_i - w) \,\}.$$

The $e^{z_h}$ term is present only with sinks. **GQA**: head $h$ reads KV head $\lfloor h / (H / H_{kv}) \rfloor$, so the $H/H_{kv}$ query heads of a group share one KV head (`L7.5`). **Positions are absolute**: row $i$ of a chunk that starts at token 24 is at $p_i = 24 + i$, and key $j$ is at position $j$, so the same flags serve a prefill ($q_{\text{offset}} = 0$, $T_q = T_k$), a chunk of a prefill, and a decode step ($T_q = 1$, $q_{\text{offset}} = T_k - 1$). A row with no visible key and no sink has no distribution: the contract says $o_i = 0$ and $\mathrm{lse}_i = -\infty$.

### 2.2 The tiled online softmax

`L9.2` showed that a running maximum $m$ and a running sum $\ell = \sum e^{s - m}$ can absorb a row's entries one block at a time: when a block raises the maximum to $m'$, multiply the old sum by $e^{m - m'}$. The same factor fixes the running output $a = \sum e^{s_j - m} v_j$, because each of its terms was weighted against the old maximum. For one key tile with visible scores $s_j$:

$$m' = \max\big(m, \max_j s_j\big), \quad \alpha = e^{m - m'}, \quad \ell' = \alpha\,\ell + \sum_j e^{s_j - m'}, \quad a' = \alpha\, a + \sum_j e^{s_j - m'}\, v_j .$$

After the last tile, $o = a / \ell$ and $\mathrm{lse} = m + \log \ell$. A sink is the cheapest possible tile: start each row at $m = z_h$, $\ell = e^0 = 1$, $a = 0$. Without a sink, start at $m = -\infty$, $\ell = 0$; the first visible tile then has $\alpha = e^{-\infty} = 0$, which multiplies a zero accumulator. A tile in which the row sees nothing must be skipped for that row, not processed: with $m = -\infty$ and no finite score, $m - m'$ is $-\infty - (-\infty) = \mathrm{NaN}$.

### 2.3 Tiles, scratch, and why memory stops growing

The kernel loops over work items (sequence, head, query tile of $B_r$ rows). For each item it keeps $m, \ell, a$ for $B_r$ rows, $B_r (D + 2)$ floats, and one tile of $B_c$ scores, and walks the key tiles that any of its rows can see. Each key tile is read from memory once per query tile and used by all $B_r$ rows while it is in cache, which is FlashAttention's whole point on a GPU (where the tiles live in shared memory) and helps on a CPU too. The scratch per worker is $B_r(D + 2) + B_c$ floats, from the caller's `rt.02` arena (one slice per worker, taken before the threads start, because the arena is not thread-safe). None of it depends on $T_k$. The kernel takes an arena mark on entry and rewinds to it before returning, so the caller's arena is exactly as it was; with `scratch == NULL` it uses a private arena and destroys it.

### 2.4 Invariance: key tiles at absolute positions

Key tile $t$ always covers positions $[tB_c, (t + 1)B_c)$, whatever the query tile, the chunk, or the batch. Within the walk, each row:

1. computes its own $V_i$ (its own `lo` and `hi`),
2. skips every tile that holds none of its keys,
3. in a tile that holds some, adds its visible keys in increasing $j$.

So the sequence of floating-point operations that produces row $i$ depends only on $q_i$, the keys and values, $p_i$, the flags, and $B_c$. Not on which other rows share its query tile ($B_r$), not on how the prompt was chunked ($q_{\text{offset}}$ moves $p_i$ and the row together), not on the batch, and not on the worker. That is the chunk invariance `L10.3` needs: query rows computed in one call or in chunks with `q_offset` are bitwise equal. Two tempting shortcuts break it: starting the tiles at a query tile's first visible key (the tile boundaries then move with the chunk), and deciding visibility once per query tile instead of per row.

### 2.5 Threads

Work items are independent: `tl_parallel_for` runs them on `rt.03` workers, each using the scratch slice of its worker index. Every row is computed by one worker with the arithmetic above, so 4 threads give the serial bits.

## 3. Worked example by hand

One query $q = [1, 0]$, three keys and values, no mask, $\sigma = 1$, $B_c = 2$:

| $j$ | $k_j$ | $v_j$ | $s_j = q \cdot k_j$ |
|---|---|---|---|
| 0 | $[1, 0]$ | $[1, 2]$ | 1 |
| 1 | $[0, 1]$ | $[3, 4]$ | 0 |
| 2 | $[1, 1]$ | $[5, 6]$ | 1 |

**Tile 0** (keys 0, 1). Start $m = -\infty$, $\ell = 0$, $a = [0, 0]$. Tile maximum 1, so $m' = 1$ and $\alpha = e^{-\infty} = 0$. Weights $e^{1-1} = 1$ and $e^{0-1} = 0.3678794$:

$$\ell = 0 + 1 + 0.3678794 = 1.3678794, \qquad a = 1 \cdot [1, 2] + 0.3678794 \cdot [3, 4] = [2.1036383, 3.4715177].$$

**Tile 1** (key 2). Tile maximum 1, $m' = 1$, $\alpha = e^0 = 1$ (no rescale). Weight $e^0 = 1$:

$$\ell = 1.3678794 + 1 = 2.3678794, \qquad a = [2.1036383 + 5, 3.4715177 + 6] = [7.1036383, 9.4715177].$$

**Normalize**: $o = a / \ell = [3, 4]$ exactly (because $(6 + 3e^{-1})/(2 + e^{-1}) = 3$ and $(8 + 4e^{-1})/(2 + e^{-1}) = 4$), and $\mathrm{lse} = m + \log \ell = 1 + \log 2.3678794 = 1.8619948$.

This is `hand_example` and `test_hand_example`. With a causal mask and $q_{\text{offset}} = 1$ the row would see only keys 0 and 1 (tile 0), giving $o = [2.1036383, 3.4715177]/1.3678794 = [1.5378828, 2.5378828]$.

## 4. The interface

```c
/* tinyllm/attention.h: q, o [B, H, Tq, D]; k, v [B, Hkv, Tk, D]; lse [B, H, Tq] or NULL */
tl_status tl_flash_attn_fwd_f32(const float *q, const float *k, const float *v, float *o, float *lse,
                                int64_t B, int64_t H, int64_t Hkv, int64_t Tq, int64_t Tk, int64_t D,
                                float scale, int64_t q_offset, int causal, int64_t window,
                                const float *sink_logits /* [H] or NULL */, int64_t Br, int64_t Bc,
                                tl_arena *scratch /* or NULL */, tl_pool *tp /* or NULL */);
/* Br, Bc = 0 pick 64. TL_EINVAL: negative sizes or q_offset, NULL tensors.
   TL_ESHAPE: H % Hkv != 0 or Hkv > H. TL_ENOMEM: the arena cannot grow. */
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3: $o = [3, 4]$, lse $= 1.8619948$ | the definition and the lse |
| `gqa_causal_window_match_naive` | differential | GQA 6:2 and MQA, causal prefill, a chunk at `q_offset` 24, window 6, window without causal, plain; odd tiles and the default 64 | every flag of a Llama-family model |
| `sinks_join_the_denominator` | unit | sinks against the oracle; all-ones values give $1 - p_{\text{sink}} = 2/3$ and lse $= \ln 3$ | gpt-oss style learned sinks |
| `no_visible_key_gives_zeros_and_minus_inf` | boundary | a query whose window excludes every key: zeros and $-\infty$; with a sink, zeros and lse = the sink | no NaN in the residual stream |
| `chunk_invariant_with_q_offset` | property | 37 rows in one call vs chunks of 1, 3, 5, 16 with `q_offset`, bitwise | chunked prefill (`L10.3`) |
| `query_tile_and_batch_invariant` | property | $B_r = 1, 3, 64$ bitwise; a sequence alone vs in a batch of 3 | batching (`L10.2`) |
| `arena_rewound_and_scratch_independent_of_tk` | property | the arena is back at its mark; high water equal for $T_k = 16$ and 1000 | memory flat in the context length |
| `pool_result_equals_serial_bitwise` | property | 4 threads vs serial (also under TSan) | threads never change bits |
| `shape_and_argument_errors` | boundary | `TL_ESHAPE` for 6 heads on 4 KV heads and $H_{kv} > H$; `TL_EINVAL` cases leave $o$ alone; $T_k = 0$ gives zeros | errors before any write |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| not rescaling the output accumulator when the maximum grows | earlier tiles are overweighted by $e^{m' - m}$ | `gqa_causal_window_match_naive` (mutant `s01`) |
| not rescaling the denominator | outputs do not average the values | `gqa_causal_window_match_naive` (mutant `s02`) |
| mapping query heads to KV heads round-robin ($h \bmod H_{kv}$) | GQA models attend with the wrong keys; MHA still passes | `gqa_causal_window_match_naive` (mutant `s03`) |
| causal as $j < p$ | a token cannot see itself | `gqa_causal_window_match_naive` (mutant `s04`) |
| a window of $w + 1$ keys | drifts from the model's training | `gqa_causal_window_match_naive` (mutant `s05`) |
| ignoring `q_offset` | a chunk at position 24 behaves like the start of a prompt | `chunk_invariant_with_q_offset` (mutant `s06`) |
| leaving the sink out of the denominator | weights sum to 1; the sink does nothing | `sinks_join_the_denominator` (mutant `s07`) |
| lse without the maximum ($\log \ell$ instead of $m + \log \ell$) | lse off by $m$; any later merge of partial results breaks | `hand_example` (mutant `s08`) |
| dividing by $\ell = 0$ | a row that sees nothing becomes NaN | `no_visible_key_gives_zeros_and_minus_inf` (mutant `s09`) |
| key tiles starting at a query tile's first visible key | the bits depend on the chunking | `chunk_invariant_with_q_offset` (mutant `s10`) |
| not rewinding the caller's arena | the engine's step arena grows every call | `arena_rewound_and_scratch_independent_of_tk` (mutant `s11`) |
| a score buffer sized by $T_k$ | scratch grows with the context: the memory FlashAttention exists to save | `arena_rewound_and_scratch_independent_of_tk` (mutant `s12`) |
| a pooled partition that drops a work item | wrong only with threads | `pool_result_equals_serial_bitwise` (mutant `s13`) |
| deciding visibility once per query tile | rows in one tile see the wrong keys; results depend on $B_r$ | `query_tile_and_batch_invariant` (mutant `s14`) |
| accepting $H \bmod H_{kv} \ne 0$ | head groups read past the KV heads | `shape_and_argument_errors` (mutant `s15`) |
| pointer arithmetic on a NULL `k` when $T_k = 0$ | UBSan: "applying zero offset to null pointer" | `shape_and_argument_errors` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot and allocator support |
| Back | `rt.02` | the arena that holds the tiles, with a mark taken and rewound around each call |
| Back | `rt.03` | `tl_parallel_for` over (sequence, head, query tile) |
| Back | `M09.6` | `tl_expf` for every weight and rescale factor |
| Back | `L9.2` | the online softmax update this kernel applies to tiles; `tl_softmax_f32` is the C tests' oracle |
| Back | `L7.7` | `windowed_attention`, the specification of every flag |
| Back | `L5.1` | `sdpa_forward`, the specification without a causal mask |
| Forward | `L9.4` | paged attention for decode performs the same tile update on blocks; its tests hold it to this kernel bit for bit |
| Forward | the standalone Rust engine | (Pass 7) the Rust forward's prefill; `L10.3`'s chunked prefill relies on the chunk invariance |

If you use the optional C paged-attention exercise, its `L9.3` prerequisite supplies this standalone C kernel.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| CPU tiles in an arena | FlashAttention-2 (CUDA) | tiles in shared memory, warps split over queries, the backward pass by recomputation | `Dao-AILab/flash-attention`, `csrc/flash_attn/src/` |
| one kernel for all flags | FlashInfer | variants generated per mask, layout, and positional encoding; paged KV inputs | `flashinfer/include/flashinfer/attention/` |
| float32 scores | FlashAttention-3 | fp8 inputs with incoherent processing, asynchronous tiles on Hopper | Shah et al. (2024) |
| fixed tile order for invariance | split-KV (FlashDecoding) | splits long contexts across thread blocks and merges partial (m, l, a) with the same rescale, trading invariance for parallelism | Dao et al., "Flash-Decoding for long-context inference" (2023) |
| float32 KV only | llama.cpp `ggml_flash_attn_ext` | CPU and GPU flash attention over quantized KV | `ggml/src/ggml-cpu/ops.cpp` |
