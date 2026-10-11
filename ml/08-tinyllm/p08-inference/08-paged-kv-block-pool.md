<!-- ol:module rt.04 -->
# Paged KV block pool (format v1, optional C)

## Overview

| | |
|---|---|
| **Module** | `rt.04` · side · C · Pass 6 · 6 to 8 h |
| **You build** | `c/src/runtime/kv_pool.c`: the pool (`tl_kv_pool_create`, `_destroy`, `_cfg`, `tl_kv_alloc`, `tl_kv_ref`, `tl_kv_unref`, `tl_kv_cow`, `tl_kv_set_fill`, `tl_kv_fill`, `tl_kv_block_ptr`, `tl_kv_block_bytes`, `tl_kv_stats_get`), the prefix index (`tl_kv_block_hash`, `tl_kv_register`, `tl_kv_lookup`), and the wire format (`tl_kv_export_bytes`, `tl_kv_export`, `tl_kv_import`, `tl_crc32c`) |
| **Contract** | [`tinyllm/kv_pool.h`](../../../course/contracts/c/include/tinyllm/kv_pool.h); the hash and the bytes in [`formats/kv-block.md`](../../../course/contracts/formats/kv-block.md); rules in [`c/ABI.md`](../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/rt.04/test_kv_pool.c`, 20 tests under ASan and UBSan with the counting allocator (what they check: section 4) · your own tests in `c/tests/rt04-kv-pool/`, rung R3, graded by mutation (threshold 0.70) · parity suite `ol parity kv.wire.v1` |
| **Needs** | `rt.02` (C allocation and error support) · `ds.01` [`tl_vec`](../../../algorithms/16-systems-data-structures/01-growable-array.md) (the free list) · `ds.02` [the Swiss table](../../../algorithms/16-systems-data-structures/02-swiss-table.md) (the prefix index) · `ds.03` [the intrusive LRU](../../../algorithms/16-systems-data-structures/03-intrusive-list-and-lru.md) (cached blocks) · reading: `M06.3` [FNV-1a](../../../math/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) and [`L8.2`'s contiguous cache](README.md) |
| **Used by** | `L9.4` uses the pool in its standalone C paged-attention exercise; cross-language comparisons use fixture files. |
| **Milestone** | `MS-L9`, the optional standalone C module group |
| **Optional depth** | Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (SOSP 2023), sections 4.1 to 4.4 |

## Key Takeaways

- The KV cache becomes fixed-size blocks in one slab; a sequence is a list of block ids, so memory is reserved a block at a time and never fragments (`alloc_is_all_or_nothing`).
- A block is in exactly one of three states, free, used, or cached, and `free + used + cached == n_blocks` after every call (`random_ops_keep_the_invariants`).
- A full block is named by a chained FNV-1a hash of its parent's hash and its token ids, so equal hashes mean equal prefixes; only full blocks are hashed, registered, or deduplicated (`hand_example_block_hash_chain`, `only_full_blocks_are_registered`).
- Shared blocks are copied only when someone writes: refcount 1 writes in place, refcount above 1 copies (`cow_copies_only_when_shared`).
- The export envelope is a pure function of the blocks' contents, checksummed with CRC-32C, so one flipped bit is refused and two implementations produce the same bytes (`hand_example_export_envelope`, `every_bit_flip_is_eformat`).

## How to work this chapter

```bash
ol start rt.04              # stubs c/src/runtime/kv_pool.c
ol tests rt.04              # read the test catalog first
ol check rt.04              # exit code is the verdict; then grades your tests by mutation
ol parity kv.wire.v1        # your export bytes against the golden envelopes
ol diff  rt.04              # after passing: your code against the reference
```

Write it in three passes, each green before the next: (1) create, alloc, ref, unref, cow, block_ptr, stats, with no hashing at all; (2) `tl_crc32c`, `tl_kv_block_hash`, register, lookup, and the LRU of cached blocks; (3) export and import against the 60-byte worked envelope.

---

## 1. Why now

`L8.2` gave every sequence a contiguous K and V array of `max_len` positions per layer. That wastes memory twice: a sequence that stops after 40 tokens still holds `max_len` positions, and two requests that start with the same 500-token system prompt each store it. A serving engine runs hundreds of sequences of unknown length, forks sequences for beam search and parallel sampling, and sees the same prompt prefixes over and over. The fix is the one operating systems use for processes: fixed-size pages and a table per sequence. This optional C module studies a block pool as a standalone exercise. The Python paged cache and Rust Candle engine implement storage independently; shared fixtures compare serialized behavior.

## 2. Principles

### 2.1 Blocks and states

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | `n_blocks`, blocks in the pool | `uint32_t` |
| $B$ | `block_tokens`, token positions per block (the engine uses 16) | `uint32_t` |
| $L$, $H$, $D$ | layers, KV heads, head dimension | `uint32_t` |
| $P$ | payload bytes of one block, $L \cdot 2 \cdot H \cdot B \cdot D \cdot 2$ (f16) | `size_t` |
| ref | holders of a block (sequences, a lookup hit) | `uint32_t` |
| fill | positions of the block that hold data, $0..B$ | `uint32_t` |
| $h_i$ | the hash of a sequence's $i$-th full block | `uint64_t` |

The pool allocates one slab of $N \cdot P$ bytes at create and never allocates again (until destroy). Inside block `id`, the K or V values of one layer are a slab `[H][B][D]` of f16, at offset `id * P + (layer * 2 + is_v) * H * B * D * 2`: exactly the order of the export payload, so `tl_kv_block_ptr` is arithmetic and export copies rows.

Every block is in exactly one state:

| State | ref | Registered | Where | Next state |
|---|---|---|---|---|
| free | 0 | no | the free list (a `tl_vec` stack of ids) | used (alloc) |
| used | at least 1 | maybe | held by sequences | free or cached when ref reaches 0 |
| cached | 0 | yes | the prefix index and the LRU list | used (lookup hit) or free (evicted by alloc) |

`tl_kv_alloc(n)` takes $n$ blocks, all or nothing: free ones first; when the free list runs dry it evicts cached blocks oldest first (unregistering each and counting `evictions`); if free + cached $< n$ it takes nothing and returns `TL_EFULL`. `tl_kv_unref` at ref 0 caches a registered block and frees any other. A second unref of a block at ref 0 is `TL_EINVAL` and changes nothing: a refcount below zero would put one block on the free list twice.

### 2.2 Copy on write

A fork shares every block of its parent: `tl_kv_ref` on each. Before a sequence writes into a block it calls `tl_kv_cow(id, &out)`: with ref 1 it is the only holder and writes in place (`out == id`); with ref above 1 the pool allocates a block, copies the contents and the fill, drops the writer's reference to the original (which the others keep), and returns the copy. Only the partially filled last block of a fork is ever copied; full prefix blocks stay shared forever.

### 2.3 The chained block hash and the prefix index

A full block's hash is FNV-1a 64 (`M06.3`) over 8 little-endian bytes of the previous block's hash (0 for block 0) followed by 4 little-endian bytes per token id:

$$h_i = \text{fnv1a64}\big(\text{le}_{64}(h_{i-1}) \,\|\, \text{le}_{32}(t_{iB}) \,\|\, \cdots \,\|\, \text{le}_{32}(t_{iB+B-1})\big), \qquad h_{-1} = 0,$$

with a result of 0 replaced by 1, because 0 means "no hash" (the partial tail). Chaining makes $h_i$ name the whole prefix up to block $i$, not just its $B$ tokens. `tl_kv_register(id, h)` puts $h \to$ id in the prefix index (a `tl_map`, created for $N$ keys so it never grows); `tl_kv_lookup(h)` returns the id and takes a reference, reviving a cached block. Only full blocks are registered: a partial block's contents still change. A second block with an already registered hash gets `TL_EBUSY` and stays a private copy. A miss is `TL_ENOTFOUND` and leaves the error slot alone: it is the normal answer for a new prompt.

### 2.4 The export envelope

`tl_kv_export` writes the blocks of a sequence for another engine: a 28-byte header (`TLKV`, version 1, dtype 1 = f16, $n$, $B$, $L$, $H$, $D$, all little-endian), then per block its hash (0 if unregistered), its fill as `n_tokens`, and its payload with every position at or past the fill written as zeros, then the CRC-32C (Castagnoli, reflected polynomial `0x82F63B78`, initial value and final XOR `0xFFFFFFFF`) of everything before it. Zeroing past the fill makes the envelope a pure function of the block's data, so two implementations write the same bytes. `tl_kv_import` refuses, allocating nothing, any envelope that is too short, has a bad magic, a version other than 1 or a dtype other than f16 (`TL_EFORMAT`), a length that does not match its header, a CRC mismatch, an `n_tokens` outside $1..B$ or a hashed block that is partial (`TL_EFORMAT`), or dimensions that differ from the pool's (`TL_ESHAPE`). It registers each imported full block unless the hash is already held, in which case the caller keeps a private copy.

## 3. Worked example by hand

The example of `formats/kv-block.md`: $B = 2$, $L = H = 1$, $D = 2$, one sequence with tokens 1 2 3 4 5.

**Hashes.** Block 0 hashes the 12 bytes `00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00`: $h_0 =$ `0xA91AB0C1027B9366`. Block 1 hashes $h_0$'s 8 bytes, least significant first (`66 93 7b 02 c1 b0 1a a9`), then `03 00 00 00 04 00 00 00`: $h_1 =$ `0x03DF829571605C60`. Token 5 is a partial block: no hash. This is `hand_example_block_hash_chain`.

**One block on the wire.** Block 0 holds K = `[[1, 2], [3, 4]]` and V = `[[0.5, -1], [0, 0.25]]` (rows are positions). In f16, 1.0 is `0x3C00` (sign 0, exponent 15 = bias, mantissa 0), stored least significant byte first as `00 3c`. $P = 1 \cdot 2 \cdot 1 \cdot 2 \cdot 2 \cdot 2 = 16$, so the envelope is $28 + 12 + 16 + 4 = 60$ bytes:

```
54 4c 4b 56 01 00 01 00 01 00 00 00 02 00 00 00     magic, version 1, dtype 1 (f16), n_blocks 1, block_tokens 2
01 00 00 00 01 00 00 00 02 00 00 00                 n_layers 1, n_kv_heads 1, head_dim 2
66 93 7b 02 c1 b0 1a a9 02 00 00 00                 block_hash 0xA91AB0C1027B9366, n_tokens 2
00 3c 00 40 00 42 00 44                             K: 1.0 2.0 3.0 4.0
00 38 00 bc 00 00 00 34                             V: 0.5 -1.0 0.0 0.25
1c 54 0b 4f                                         crc32c 0x4F0B541C
```

This is `hand_example_export_envelope`, byte for byte.

**States through a fork.** A pool of 3 blocks: alloc 2 for the sequence (free 1, used 2), register block 0 under $h_0$, fork (both blocks at ref 2), the child writes position 3 into block 1: `tl_kv_cow` copies it (free 0, used 3). The parent finishes: block 0 drops to ref 1, block 1 (its own now) to 0 and, unregistered, back to free. The child finishes: block 0 reaches ref 0 and, registered, becomes cached (free 2, used 0, cached 1). A new prompt starting with 1 2 looks up $h_0$ and gets block 0 back without recomputing it.

## 4. The interface

```c
/* tinyllm/kv_pool.h (rt.04), abridged; the header documents every status */
typedef struct { uint32_t n_blocks, block_tokens, n_layers, n_kv_heads, head_dim;
                 int32_t dtype; uint32_t format; } tl_kv_cfg;        /* TL_F16, format 1 */
typedef struct { uint32_t free, used, cached, evictions; } tl_kv_stats;
tl_status tl_kv_pool_create(const tl_kv_cfg *cfg, tl_kv_pool **out);
tl_status tl_kv_alloc(tl_kv_pool *p, uint32_t n, uint32_t *ids);     /* all or nothing; TL_EFULL */
void      tl_kv_ref(tl_kv_pool *p, uint32_t id);
tl_status tl_kv_unref(tl_kv_pool *p, uint32_t id);                   /* double unref: TL_EINVAL */
tl_status tl_kv_cow(tl_kv_pool *p, uint32_t id, uint32_t *out);
tl_status tl_kv_set_fill(tl_kv_pool *p, uint32_t id, uint32_t n_tokens);
uint64_t  tl_kv_block_hash(uint64_t parent, const uint32_t *toks, uint32_t n);
tl_status tl_kv_register(tl_kv_pool *p, uint32_t id, uint64_t hash); /* full blocks; TL_EBUSY */
tl_status tl_kv_lookup(tl_kv_pool *p, uint64_t hash, uint32_t *id);  /* TL_ENOTFOUND, slot untouched */
void     *tl_kv_block_ptr(const tl_kv_pool *p, uint32_t id, uint32_t layer, int is_v);
tl_status tl_kv_export(const tl_kv_pool *p, const uint32_t *ids, uint32_t n,
                       void *buf, size_t cap, size_t *written);       /* TL_EFULL sets *written */
tl_status tl_kv_import(tl_kv_pool *p, const void *buf, size_t len, uint32_t *ids_out);
uint32_t  tl_crc32c(const void *data, size_t n, uint32_t crc);
```

Two choices the header leaves open are fixed here: export refuses a block with fill 0 (`TL_EINVAL`), because reader rule 5 would refuse the envelope; and the free list is a `tl_vec` of ids reserved for $N$ at create, popped by reading the last id and shortening `len`, so pushing a freed id never reallocates.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_block_hash_chain` | unit, golden | $h_0$, $h_1$ of section 3; the same tokens under another parent hash differently | L10.4 and L10.6 find each other's blocks only through these values |
| `hand_example_export_envelope` | unit, golden | the 60-byte envelope of section 3, nothing written past it | the PushKv payload and the parity golden |
| `crc32c_check_value_and_continuation` | unit | `crc32c("123456789") = 0xE3069283`; continuing from a previous result | the envelope trailer and the gRPC chunk checksum |
| `alloc_is_all_or_nothing` | unit, boundary | asking for more than free + cached takes nothing | admission by free blocks (L10.2) |
| `unref_frees_or_caches` | unit | ref 0 caches a registered block, frees others; the three counts add up | prefix reuse without leaking memory |
| `double_unref_is_einval_and_changes_nothing` | fault | a second unref and a bad id are `TL_EINVAL`; ref on a free block sets the error slot | one block on the free list twice is shared memory nobody knows about |
| `lookup_revives_a_cached_block_and_misses_quietly` | unit | a hit returns the block used again; a miss is `TL_ENOTFOUND` with the error slot unchanged; the contents are intact | prefix hits for every new prompt |
| `eviction_takes_the_least_recently_used` | unit | a hit moves a cached block to the newest end; two allocations evict the two oldest; `evictions` counts | the cache keeps the prefixes in use |
| `a_revived_block_is_never_evicted` | unit | after a lookup hit, the next allocation reclaims another cached block | a running request's prefix stays put |
| `cow_copies_only_when_shared` | unit | ref 1 writes in place; ref 2 copies contents and fill; the original is unchanged | forks for beam search and parallel sampling |
| `cow_in_a_full_pool_is_efull` | boundary | a copy that cannot allocate changes nothing | preemption instead of corruption |
| `only_full_blocks_are_registered` | unit, boundary | partial, hash 0, and unheld blocks refused; a duplicate hash is `TL_EBUSY`; a registered block is immutable | a partial block's hash would name contents that still change |
| `block_ptr_layout` | unit, boundary | every (block, layer, K/V) slab distinct and exactly $H B D$ f16 long; out of range gives NULL | L9.4 and L8.3 index slabs `[H][B][D]` |
| `export_zeroes_past_the_fill_and_reports_its_size` | unit, boundary | stale data past the fill exported as zeros; a short buffer gets `TL_EFULL` and the size needed | envelopes compare byte for byte |
| `export_import_roundtrip` | unit | export, import into another pool, export again: same bytes; full blocks arrive registered, the tail does not | disaggregated prefill and decode (L10.6) |
| `import_keeps_the_first_holder_of_a_hash` | unit | an imported block whose hash is held stays a private copy | KV-transfer dedup |
| `every_bit_flip_is_eformat` | fault | all 480 single-bit flips of the worked envelope are refused; nothing allocated | a corrupted transfer never decodes |
| `import_reader_rules` | boundary | version 2, a wrong dtype, bad `n_tokens`, a hashed partial block (each with a valid CRC) are `TL_EFORMAT`; another shape is `TL_ESHAPE` | format v2 arrives only through craft.13 |
| `create_rules_and_alloc_failure` | boundary, fault | format 2 and other dtypes `TL_EUNSUPPORTED`; zero dimensions `TL_EINVAL`; every failing allocation of create `TL_ENOMEM` with no leak | the engine starts or fails cleanly |
| `random_ops_keep_the_invariants` | property | 40 seeded runs of alloc, ref, unref, cow, register, lookup, export, and import against a model of the refcounts; held blocks keep their contents | the three promises under any interleaving |

**Your tests (rung R3).** First write C files under `c/tests/rt04-kv-pool/` and run `ol tdd red rt.04` against your stub; then implement until `ol tdd green rt.04`. Name them `hash_chain_by_hand`, `alloc_all_or_nothing`, `unref_caches_registered_blocks`, `double_unref_refused`, `lookup_hit_is_not_evicted`, `cow_copies_shared_blocks`, `partial_blocks_not_registered`, `roundtrip_and_bit_flip`. The planted bug "a revived block stays on the eviction list" (`s11`) is required; overall 70%.

**Parity.** `ol parity kv.wire.v1` feeds six sequences (the worked example, partial tails, several layers and heads) through a driver that builds the blocks with your pool and compares `tl_kv_export` with envelopes written by an independent Python transcription of `formats/kv-block.md`; the Rust writer of `L10.6` joins the same suite.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Hashing a block without its parent | two prompts that differ early share later blocks | `hand_example_block_hash_chain` (mutant `s01`) |
| Hashing token ids with the wrong width | hashes differ from the Rust engine's | `hand_example_block_hash_chain` (mutant `s02`) |
| Starting FNV from 0 instead of the offset basis | every hash differs from the format's | `hand_example_block_hash_chain` (mutant `m01`) |
| Header fields out of order | another engine reads a nonsense shape | `hand_example_export_envelope` (mutant `s03`) |
| Exporting hash 0 for registered blocks | the receiver cannot deduplicate | `hand_example_export_envelope`, `export_import_roundtrip` (mutant `s04`) |
| Checksumming only the payload | header corruption passes | `hand_example_export_envelope` (mutant `s05`) |
| CRC without the initial inversion | the check value is wrong, envelopes disagree | `crc32c_check_value_and_continuation` (mutant `s06`) |
| Taking blocks one at a time and failing halfway | blocks leak on every refused admission | `alloc_is_all_or_nothing` (mutant `s07`) |
| Freeing registered blocks at ref 0 | no prefix is ever reused | `unref_frees_or_caches` (mutant `s08`) |
| Letting a refcount go below zero | a block handed to two sequences | `double_unref_is_einval_and_changes_nothing` (mutant `s09`) |
| `tl_kv_ref` reviving a block nobody holds | a free block counts as used and leaks | `double_unref_is_einval_and_changes_nothing` (mutant `s10`) |
| Forgetting to unlink a revived block from the LRU | the next allocation evicts a block a request is reading | `a_revived_block_is_never_evicted` (mutant `s11`) |
| Setting the error slot on a miss | the caller's earlier error message is lost | `lookup_revives_a_cached_block_and_misses_quietly` (mutant `s12`) |
| Evicting the newest cached block | the hot system prompt is recomputed | `eviction_takes_the_least_recently_used` (mutant `s13`) |
| Not counting evictions | `/metrics` shows no pressure | `eviction_takes_the_least_recently_used` (mutant `s14`) |
| Copy on write keeping the writer's reference | the original never returns to the pool | `cow_copies_only_when_shared` (mutant `s15`) |
| Writing a shared block in place | the parent sees the child's tokens | `cow_copies_only_when_shared`, `cow_in_a_full_pool_is_efull` (mutant `s16`) |
| Registering a partial block | a prefix hit returns half-written KV | `only_full_blocks_are_registered` (mutant `s17`) |
| A duplicate hash replacing the first holder | the first block stays cached but unreachable | `only_full_blocks_are_registered` (mutant `s18`) |
| K and V sharing one slab | every V overwrites a K | `block_ptr_layout` (mutant `s19`) |
| Exporting stale data past the fill | two engines write different bytes for the same block | `export_zeroes_past_the_fill_and_reports_its_size` (mutant `s20`) |
| Not reporting the size needed | the caller cannot size its buffer | `export_zeroes_past_the_fill_and_reports_its_size` (mutant `s21`) |
| Import dropping the fill | the decode engine sees empty blocks | `export_import_roundtrip` (mutant `s22`) |
| Import never registering | transferred prefixes are never hit again | `export_import_roundtrip` (mutant `s23`) |
| Import fighting the first holder of a hash | a valid envelope fails with `TL_EBUSY` | `import_keeps_the_first_holder_of_a_hash` (mutant `s24`) |
| Skipping the CRC | corrupted KV decodes | `every_bit_flip_is_eformat` (mutant `s25`) |
| Reading any version | a v2 envelope's fp8 bytes read as f16 | `import_reader_rules` (mutant `s27`) |
| Ignoring the receiving pool's shape | a payload laid out for other dimensions is copied in | `import_reader_rules` (mutant `s28`) |
| Accepting format 2 before its migration | a pool that cannot read what it was asked to hold | `create_rules_and_alloc_failure` (mutant `s29`) |
| Leaking on a failed create | the counting allocator reports live blocks | `create_rules_and_alloc_failure` (mutant `s30`) |
| Envelope version field 0 | every reader refuses it | `hand_example_export_envelope` (mutant `m02`) |
| Losing a freed id | the pool shrinks by one block per request | `random_ops_keep_the_invariants` (mutant `m03`) |

## 6. Where it's used next
| Forward | `L10.1` | Registered call site uses this module. |
| Forward | `L10.8` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | C allocation and error support for the standalone pool |
| Back | `M06.3` | `tl_fnv1a64`, continued across the parent and the tokens |
| Back | `ds.01` | the free list is a `tl_vec` of ids |
| Back | `ds.02` | the prefix index from block hash to block id |
| Back | `ds.03` | the LRU of cached blocks, embedded in each block's record |
| Forward | `L8.3` | the Python paged cache implements its own allocation and block operations; fixture files can compare behavior |
| Forward | `L9.4` | the paged attention kernel reads K and V slabs through `tl_kv_block_ptr` |
| Forward | `L10.4` | the engine's block manager with `--prefix-cache=hash` uses register and lookup |
| Forward | `L10.6` | prefill and decode engines move blocks as export envelopes |

If you skip this module, `ol check L8.3` stops with `needs rt.04: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| block pool with refcounts and CoW | vLLM's `BlockPool` and `KVCacheManager` | block hashes with extra keys (LoRA, multimodal), per-request block tables on the GPU | [`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py) |
| chained block hash | vLLM's `hash_block_tokens` | the same parent-chained hash, in Python over tuples | [`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py) |
| export envelope | NVIDIA Dynamo / LMCache KV transfer | NIXL/RDMA transfer of KV blocks between GPUs and nodes | [LMCache](https://github.com/LMCache/LMCache) |
| f16 blocks | llama.cpp KV cache | quantized KV cache types (`q8_0`, `q4_0`) selected per run | `src/llama-kv-cache.cpp` |
