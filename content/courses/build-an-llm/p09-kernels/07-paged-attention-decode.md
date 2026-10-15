<!-- ol:module L9.4 -->
# Paged attention for decode in C

## Overview

| | |
|---|---|
| **Module** | `L9.4` · side · C · Pass 6 · 4 to 6 h |
| **You build** | `c/src/kernels/paged_attn.c`: `tl_paged_attn_decode_f32`, one decode step of attention for a batch of sequences whose K and V live in blocks of your `rt.04` pool, reached through block tables, read as f16, with GQA and a sliding window, allocating nothing |
| **Contract** | [`course/contracts/c/include/tinyllm/attention.h`](../../../../course/contracts/c/include/tinyllm/attention.h) · the pool: [`kv_pool.h`](../../../../course/contracts/c/include/tinyllm/kv_pool.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/L9.4/`: `test_paged_attn.c` (C, under ASan and UBSan, and TSan for the pool case, against a naive oracle and, bitwise, against your `L9.3`) and shared file fixtures (what they check: section 4) |
| **Needs** | `rt.02` the loader · `rt.03` the pool · `rt.04` the block pool · `M09.7` `tl_f16_to_f32` · `M09.6` `tl_expf` · `L9.2` `tl_softmax_f32` (the C oracle) · `L9.3` FlashAttention (the bitwise oracle) · `L8.3` `PagedKVCache` · `L7.7` `windowed_attention` (or `--ref-deps`). Reading: `L7.5` (GQA) |
| **Used by** | None: this optional C exercise is tested as a standalone binary; the Rust engine implements decode independently |
| **Milestone** | `MS-L9` |
| **Optional depth** | Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023); Dao et al., "Flash-Decoding" (2023) |

## Key Takeaways

- **A sequence's cache is a list of blocks,** not one buffer: position $j$ lives in slot $j \bmod b$ of block $\mathit{table}[\lfloor j / b \rfloor]$, so the kernel reads keys in position order through the table, whatever the block ids (`hand_example`).
- **Decode is FlashAttention with one query at the newest position:** the same online softmax over the blocks, with the running output kept in the caller's `out` row, so the kernel allocates nothing (`matches_naive_over_the_gathered_cache`).
- **Blocks are tiles aligned to absolute positions,** so a paged decode gives the same bits as your `L9.3` kernel on the gathered cache with $B_c$ = the block size: prefill with one kernel and decode with the other, and the tokens agree (`equals_flash_attention_bitwise`).
- **Shared prefix blocks just work:** after a fork two tables name the same blocks, each read independently, and a sequence's output has the same bits alone or in a batch of 16 (`shared_blocks_and_batch_invariance`).
- **The whole table is validated before the first write:** an empty context, a table too short, a block id outside the pool, or a layer out of range is `TL_EINVAL` (`bad_tables_and_shapes`).

## How to work this chapter

```bash
ol start L9.4              # stubs c/src/kernels/paged_attn.c into your repo
ol tests L9.4              # read the test catalog first
ol check L9.4              # exit code is the verdict
ol check L9.4 --ref-deps   # only if rt.04, L8.3, L9.3, or another dependency is not passing yet
ol diff  L9.4              # after passing: your code against the reference
```

---

## 1. Why now

`L8.3` defines the logical paged-cache behavior in Python: each sequence owns a block table, prefix blocks are shared after a fork, and a block is copied only when someone writes into a shared one. This optional C exercise implements the same layout for a standalone decode kernel. Its tests compare the C result with a naive oracle and the C prefill kernel, so the implementation can be checked without binding C into Python or Rust. The Rust engine owns an independent Rust KV block manager and attention implementation.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $b$ | `block_tokens`: positions per block (16 in the engine) | `uint32_t` |
| $\mathit{table}_s$ | sequence $s$'s block ids in position order, row $s$ of `block_tables` | `uint32_t[max_blocks]` |
| $n_s$ | `ctx_lens[s]`: positions in the cache, the newest included | `int32_t` |
| $p_s = n_s - 1$ | the query's position (the newest token) | |
| $q_{s,h} \in \mathbb{R}^D$ | the query of sequence $s$, head $h$ | `float[D]` |
| $k_j, v_j$ | key and value at position $j$, stored as f16 | `uint16_t[D]` |
| $w$ | the window ($w \le 0$: none) | `int64_t` |
| $m, \ell, a$ | running maximum, denominator, output (as in `L9.3`) | `float`, `float`, `float[D]` |

### 2.1 Where position $j$ lives

`kv_pool.h` lays out one block as, for each layer, a K slab and a V slab, each `[n_kv_heads][block_tokens][head_dim]` elements of the pool's dtype (f16 in format 1). So the key of KV head $g$ at position $j$ of sequence $s$ is at

$$\texttt{tl\_kv\_block\_ptr}(\mathit{kv}, \mathit{table}_s[\lfloor j/b \rfloor], \mathit{layer}, 0) + \big(g \cdot b + (j \bmod b)\big) \cdot D ,$$

and the value is the same with `is_v = 1`. The table maps logical blocks (positions $0..b-1$, $b..2b-1$, ...) to physical blocks anywhere in the pool. That indirection is the whole point of paging: memory is allocated a block at a time, no sequence needs a contiguous region, and two sequences can name the same physical block.

### 2.2 One decode step

For each sequence $s$ and head $h$ (KV head $g = \lfloor h / (H/H_{kv}) \rfloor$), the query is the newest token, so it is causal by construction and sees positions

$$V = \{\, j : \max(0, p_s - w + 1) \le j \le p_s \,\} \quad (\text{all of } 0..p_s \text{ when } w \le 0).$$

The kernel walks the logical blocks that intersect $V$ in increasing order and applies `L9.3`'s update to each block's visible positions: scores $\sigma\, q \cdot k_j$ (the f16 key decoded by your `tl_f16_to_f32`), the block's maximum, $m' = \max(m, \cdot)$, $\alpha = e^{m - m'}$, $\ell \leftarrow \alpha \ell + \sum e^{s_j - m'}$, $a \leftarrow \alpha a + \sum e^{s_j - m'} v_j$. At the end $o = a \cdot (1/\ell)$. The query sees at least itself, so $\ell > 0$.

### 2.3 No scratch

The contract gives this kernel no arena: a decode step is small and frequent, and every byte of state fits elsewhere. The running output $a$ lives in the caller's `out` row (it is overwritten anyway); $m$ and $\ell$ are two locals; the scores of a block go into a fixed stack buffer of 256 floats (a block larger than that is scored in aligned pieces of 256). Nothing depends on $n_s$.

### 2.4 Equal to FlashAttention, bit for bit

`L9.3` with one query row at `q_offset` $= p_s$, causal, the same window, and $B_c = b$ visits key tiles $[tb, (t+1)b)$ in increasing $t$ and performs, for each, exactly the operations above on the same float32 values (the f16 values decoded). The two kernels therefore produce the same bits; `equals_flash_attention_bitwise` checks it. Anything that changes the grouping of the sums (a piece size that depends on the batch, a different shift, dividing by $\ell$ instead of multiplying by $1/\ell$) keeps the answer close but breaks the bits, and with them the engine's guarantee that prefill and decode agree.

### 2.5 Batch invariance and threads

Each $(s, h)$ row reads only its own query, its own table, and the pool, and writes only its own output: a sequence's result is the same alone or in a test batch of 16, and the same whichever `rt.03` worker computes it.

### 2.6 Validate the tables first

The engine builds the tables, and a bug there would make the kernel read another sequence's memory or past the pool. So before the first write the kernel checks the shapes against the pool (`TL_ESHAPE` when $H_{kv}$ or $D$ differ, or $H \bmod H_{kv} \ne 0$), the layer, every context length ($\ge 1$), that each table holds $\lceil n_s / b \rceil \le$ `max_blocks` entries, and that every block id it will read is below `n_blocks` (`TL_EINVAL`). Pools in format 2 (fp8, `craft.13`) are `TL_EUNSUPPORTED` until that migration.

## 3. Worked example by hand

`L9.3`'s example in a paged cache: $D = 2$, $b = 2$, one sequence of $n = 3$ positions in a pool of 4 blocks, table $[3, 1]$:

| Position $j$ | block $\lfloor j/2 \rfloor$ → id | slot | $k_j$ | $v_j$ |
|---|---|---|---|---|
| 0 | 0 → 3 | 0 | $[1, 0]$ | $[1, 2]$ |
| 1 | 0 → 3 | 1 | $[0, 1]$ | $[3, 4]$ |
| 2 | 1 → 1 | 0 | $[1, 1]$ | $[5, 6]$ |

All values are exact in f16. The query $q = [1, 0]$ is position $p = 2$ and sees positions 0, 1, 2.

**Block id 3** (positions 0, 1): scores 1 and 0, $m' = 1$, $\alpha = e^{-\infty} = 0$, $\ell = 1 + e^{-1} = 1.3678794$, $a = [1, 2] + 0.3678794\,[3, 4] = [2.1036383, 3.4715177]$.

**Block id 1** (position 2; slot 1 is empty and not visible): score 1, $m' = 1$, $\alpha = 1$, $\ell = 2.3678794$, $a = [7.1036383, 9.4715177]$.

$o = a / \ell = [3, 4]$: the first test, `hand_example`, computes this exact result through the C interface.

## 4. The interface

```c
/* tinyllm/attention.h */
tl_status tl_paged_attn_decode_f32(const float *q, const tl_kv_pool *kv, uint32_t layer,
                                   const uint32_t *block_tables, int32_t max_blocks,
                                   const int32_t *ctx_lens, float *out,
                                   int64_t B, int64_t H, int64_t Hkv, int64_t D,
                                   float scale, int64_t window, tl_pool *tp);
/* q, out [B, H, D]. Sequence b's blocks: block_tables[b * max_blocks + i].
   TL_EINVAL: NULL pointers, a block id or layer out of range, ctx_lens[b] < 1,
   a sequence needing more than max_blocks blocks. TL_ESHAPE: Hkv or D differ
   from the pool, or H % Hkv != 0. */
```

From Python, `PagedKVCache.pool` (`L8.3`) is the `tl_kv_pool *` and `block_table(seq)` the row of ids.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3 through table $[3, 1]$ | the indirection and the slot arithmetic |
| `matches_naive_over_the_gathered_cache` | differential | GQA 6:2, $D = 32$, two layers, lengths 1, 37, 70 in shuffled blocks, window 0 and 20 | every flag against plain attention |
| `equals_flash_attention_bitwise` | differential, property | your `L9.3` on the gathered values with $B_c = 16$, bitwise | prefill and decode agree |
| `shared_blocks_and_batch_invariance` | property | 16 sequences sharing two prefix blocks; each alone vs in the test batch, bitwise | forked prefix blocks and sequence isolation |
| `pool_result_equals_serial_bitwise` | property | 4 threads vs serial (also under TSan) | threads never change bits |
| `bad_tables_and_shapes` | boundary | empty context, short table, id 9 in a 4-block pool, layer out of range (`TL_EINVAL`); head or width mismatch (`TL_ESHAPE`); `out` untouched | a table bug fails loudly |
| `hand_example` | unit, smoke | section 3 written through your `PagedKVCache` | the cache and the kernel share one layout |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| not rescaling the running output | blocks before a larger score are overweighted | `matches_naive_over_the_gathered_cache` (mutant `s01`) |
| not rescaling the denominator | outputs do not average the values | `matches_naive_over_the_gathered_cache` (mutant `s02`) |
| using the logical block index as the block id | reads whatever block happens to have that id | `hand_example` (mutant `s03`) |
| forgetting the KV head's offset inside the slab | every head reads KV head 0 | `matches_naive_over_the_gathered_cache` (mutant `s04`) |
| mapping heads round-robin ($h \bmod H_{kv}$) | GQA models attend with the wrong keys | `matches_naive_over_the_gathered_cache` (mutant `s05`) |
| a window of $w + 1$ positions | drifts from the model's training | `matches_naive_over_the_gathered_cache` (mutant `s06`) |
| placing the query at $n$ instead of $n - 1$ | reads an unwritten slot | `hand_example` (mutant `s07`) |
| reading V from the K slab | outputs are averages of keys | `hand_example` (mutant `s08`) |
| shifting by a maximum that includes positions outside the window | the answer is close, but not FlashAttention's bits | `equals_flash_attention_bitwise` (mutant `s09`) |
| dividing by $\ell$ instead of multiplying by $1/\ell$ | close, but not FlashAttention's bits: prefill and decode disagree on near-ties | `equals_flash_attention_bitwise` (mutant `s10`) |
| a smaller piece size for a lone sequence | its bits change when it joins a batch | `shared_blocks_and_batch_invariance` (mutant `s11`) |
| a pooled partition that drops a row | wrong only with threads | `pool_result_equals_serial_bitwise` (mutant `s12`) |
| accepting an empty context | $1/\ell = \infty$: NaN | `bad_tables_and_shapes` (mutant `s13`) |
| trusting `max_blocks` | reads past the table | `bad_tables_and_shapes` (mutant `s14`) |
| trusting the block ids | `tl_kv_block_ptr` returns NULL, then a crash | `bad_tables_and_shapes` (mutant `s15`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | the error slot and allocator support |
| Back | `rt.03` | `tl_parallel_for` over (sequence, head) |
| Back | `rt.04` | the block pool: `tl_kv_pool_cfg` and `tl_kv_block_ptr` |
| Back | `M09.7` | `tl_f16_to_f32` decodes every key and value in this standalone C module |
| Back | `M09.6` | `tl_expf` |
| Back | `L9.2` | the online update; `tl_softmax_f32` is the naive oracle's softmax |
| Back | `L9.3` | the bitwise oracle: decode must equal FlashAttention with one query |
| Back | `L8.3` | `PagedKVCache` writes the blocks the Python tests read |
| Back | `L7.7` | `windowed_attention`, the specification |
| Forward | None | the production Rust engine uses its own Rust implementation; this optional C exercise has no production caller |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one query per (sequence, head) | vLLM PagedAttention v1 and v2 | one thread block per (sequence, head); v2 splits long contexts across blocks and merges partial (m, l, a) | vLLM `csrc/attention/` |
| a serial walk over a long table | Flash-Decoding | split the context across workers for one query and merge with the same rescale: parallel at batch size 1, at the price of a fixed split order | Dao et al. (2023) |
| f16 KV | fp8 KV caches | half the bytes per token; scales per head or per block (your `craft.13` migration) | vLLM `kv_cache_dtype="fp8"` |
| block tables from the engine | SGLang RadixAttention | the tables come from a radix tree over token ids, so shared prefixes are found automatically | SGLang `radix_cache.py` |
