<!-- ol:module L10.4 -->
# Block manager with prefix cache (none, hash, radix)

## Overview

| | |
|---|---|
| **Module** | `L10.4` · build · Rust · Pass 7 · 8 to 12 h |
| **You build** | `rust/crates/tl-engine/src/block_manager.rs`: `BlockManager`, the scheduler's `BlockSpace` over the C KV pool, with three prefix-cache modes: `none`, `hash` (the pool's own index of chained block hashes), and `radix` (your `L8.4` radix cache) |
| **Contract** | the pool API in [`tinyllm/kv_pool.h`](../../../course/contracts/c/include/tinyllm/kv_pool.h) and the block hash in [`formats/kv-block.md`](../../../course/contracts/formats/kv-block.md); `prefix_cache` of `[engine]` in [`config/runtime.schema.json`](../../../course/contracts/config/runtime.schema.json) |
| **Tests** | `course/tests/rust/l10_4.rs`, 10 tests plus one B test (what they check: section 4); `ol bench L10.4 --assert` runs `bench_shared_prefix_ttft` against the budget TTFT none / TTFT hash >= 2 |
| **Needs** | `L10.2` the `BlockSpace` seam ([chapter](02-continuous-batching.md)) · `L10.1` `KvPool` and `kv_block_hash` ([chapter](01-model-runner-and-sampler.md)) · `L8.4` the radix prefix cache ([chapter](../p08-inference/04-radix-prefix-cache.md)) · `L10.1` the pool's refcounts, prefix index, and LRU · reading: `ds.02` the pool's hash table, `ds.07` the radix tree, `L8.3` paged KV · or `--ref-deps` |
| **Used by** | `L10.5` (the engine creates one `BlockManager` over the runner's pool, in the mode runtime.toml names) |
| **Milestone** | `MS-L10` |
| **Optional depth** | [Zheng et al. 2024, SGLang and RadixAttention](https://arxiv.org/abs/2312.07104) (free); [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html) (free); [Kwon et al. 2023, PagedAttention](https://arxiv.org/abs/2309.06180) (free), section 4.4 on sharing |

## Key Takeaways

- A position's K and V depend only on the tokens up to it, so requests that share a prefix can share the blocks of that prefix; a cache of finished requests' full blocks turns repeated system prompts into skipped prefill (`hand_example_prefix_hit`, `prefix_hit_skips_prefill_tokens`).
- A block is named by its tokens **and** everything before it: the chained hash folds in the parent's hash, the radix tree the path from the root (`same_tokens_at_another_position_do_not_hit`).
- At most `len - 1` prompt tokens are reused, rounded down to a block: the model must still run on the last prompt token to produce the first new one (`last_token_is_always_computed`).
- Sharing is by reference count; a request gives back every reference it took, including after a failed allocation, or cached blocks can never be evicted (`shared_blocks_return_to_free_after_eviction`, `failed_allocation_returns_its_cache_hits`).
- The cache changes how much is computed, never what: outputs are identical in all three modes (`identical_outputs_with_and_without_cache`).

## How to work this chapter

```bash
ol start L10.4               # stubs tl-engine/src/block_manager.rs
ol tests L10.4
ol check L10.4
ol bench L10.4 --assert      # the TTFT budget (local; never part of ol check)
```

---

## 1. Why now

Chat traffic repeats itself. Every request to an assistant starts with the same system prompt; a conversation resends its whole history each turn; a few-shot template sends the same examples every time. Without a cache, the engine recomputes K and V for those shared tokens on every request: prefill work, KV memory, and time to first token, spent again on identical inputs. `L8.4` built the radix cache in isolation; `L10.1` built a pool that can register and look up full blocks by hash. This module puts them behind the scheduler's `BlockSpace`, so the engine of `L10.5` reuses prefixes across requests.

## 2. Principles

### 2.1 What may be shared

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | token positions per block | integer |
| $x_0 \dots x_{n-1}$ | a prompt's token ids | `u32` |
| $h_i$ | hash naming block $i$ (tokens $iB$ to $(i+1)B - 1$) | `u64` |
| $\mathrm{FNV}(\cdot)$ | FNV-1a 64 over bytes | function |
| $\rho_b$ | the reference count of pool block $b$ | integer |

The K and V of position $j$ are a function of $x_0, \dots, x_j$ only (causal attention, absolute RoPE). So two sequences with the same first $m$ tokens have the same K and V for positions $0$ to $m - 1$, bit for bit (the kernels are batch-invariant, `L10.1`). Three rules turn that into a cache:

1. **Whole blocks only.** A block is shared only when full; a request's partial last block is its own, and writes always go to blocks no one else holds, so no copy on write is ever needed.
2. **The prefix, not just the tokens.** Block $i$ is reusable only when every block before it matches too. The hash chains: $h_i = \mathrm{FNV}(h_{i-1} \,\|\, x_{iB} \,\|\, \dots \,\|\, x_{(i+1)B - 1})$ with $h_{-1} = 0$, each id as 4 little-endian bytes and the parent as 8, and 0 replaced by 1 (`formats/kv-block.md`, `tl_kv_block_hash`). The radix tree gets the same effect by matching paths from the root.
3. **The last token is always computed.** A match covers at most $\lfloor (n - 1) / B \rfloor \cdot B$ tokens: the model must run on the last prompt token to produce the logits of the first new one, so even a fully cached prompt recomputes its last block.

### 2.2 Three modes over one pool

- **none**: `allocate` takes $\lceil n / B \rceil$ fresh blocks; `release` drops every reference, so blocks go back to free.
- **hash**: on `release`, each full computed block $i$ is registered in the pool's prefix index under $h_i$ (`tl_kv_register`; its fill must be $B$, which the block manager sets, so a block is full by the time it is named). When its last reference goes, a registered block becomes **cached**, not free; `tl_kv_alloc` evicts cached blocks least recently used first when free ones run out. `allocate` walks the prompt's full blocks front to back, `tl_kv_lookup`-ing each $h_i$ (a hit takes a reference: a cached block becomes used again) and stops at the first miss.
- **radix**: the radix cache of `L8.4` maps token runs to block ids. On `release` the full computed blocks are inserted; the cache keeps the request's reference to each block it newly stores, so a cached block has $\rho = 1$ (the cache's). `allocate` matches the longest cached prefix, takes a reference on each matched block for the request, and locks the matched node so eviction cannot take it while the request runs. When the pool's free list is short, the block manager evicts least recently used unlocked leaves and drops the cache's reference on each.

In both caches a reused block has $\rho \ge 2$ while a request holds it (the cache, the request), and $\rho$ returns to the cache's count when the request ends. The invariant `free + used + cached == n_blocks` (`L10.1`) holds at every step; at quiet times, `used` is 0 in hash mode and equals the cache's blocks in radix mode (`refcounts_balance_under_random_workload`).

### 2.3 Hash or radix

The hash index is flat and O(1) per block, lives in C, and needs nothing more than the pool; a lookup walks blocks one hash at a time, and eviction is the pool's own LRU over blocks. The radix tree matches a whole prefix in one walk, keeps the structure of shared prefixes visible (a conversation tree), and evicts whole leaves (the newest suffixes) first. Both give the same hits on the same traffic in this engine; `bench_shared_prefix_ttft` measures both, and `gw.05` routes requests to the engine whose cache is warm using the hit ratio exported here.

## 3. Worked example by hand

Blocks of $B = 4$, an 8-block pool, either cache.

**A** = tokens $1, 2, \dots, 10$ arrives to an empty cache: `allocate` finds nothing and takes $\lceil 10 / 4 \rceil = 3$ fresh blocks. A finishes; `release` with its 10 computed tokens has $\lfloor 10 / 4 \rfloor = 2$ full blocks: $[1, 2, 3, 4]$ and $[5, 6, 7, 8]$. Both are cached; the third block (tokens 9, 10) is partial and goes back to free.

**B** = $1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97$ (11 tokens). Usable: $\lfloor (11 - 1) / 4 \rfloor = 2$ blocks. Block 0's hash $h_0 = \mathrm{FNV}(0 \,\|\, 1, 2, 3, 4)$ hits, block 1's $h_1 = \mathrm{FNV}(h_0 \,\|\, 5, 6, 7, 8)$ hits: 8 cached tokens. B needs $\lceil 11/4 \rceil = 3$ blocks, so one fresh one. Its table starts with A's two blocks, and its prefill starts at position 8: 3 tokens computed instead of 11. The hit rate so far is $8 / (10 + 11) = 0.381$ (`hand_example_prefix_hit`).

**C** = $1, \dots, 8$ (exactly A's two full blocks). Usable: $\lfloor 7 / 4 \rfloor = 1$ block, so 4 cached tokens, and the last block is recomputed to produce C's first logits (`last_token_is_always_computed`).

**D** = $5, 6, 7, 8, \dots$: its first block holds the same tokens as A's second, but $\mathrm{FNV}(0 \,\|\, 5, 6, 7, 8) \ne h_1$, and the radix tree has no root edge starting with 5: no hit, correctly (`same_tokens_at_another_position_do_not_hit`).

## 4. The interface

```rust
// rust/crates/tl-engine/src/block_manager.rs
pub struct BlockStats { pub lookups: u64, pub query_tokens: u64, pub hit_tokens: u64, pub evictions: u64 }
pub struct BlockManager { /* SharedPool, mode, tables, Option<RadixCache>, stats */ }
impl BlockManager {
    pub fn new(pool: SharedPool, mode: PrefixCache) -> BlockManager;   // PrefixCache::{None, Hash, Radix} (L10.1's EngineConfig)
    pub fn mode(&self) -> PrefixCache;  pub fn stats(&self) -> BlockStats;
    pub fn hit_rate(&self) -> f64;      // hit_tokens / query_tokens
    pub fn cached_blocks(&self) -> usize;
}
impl BlockSpace for BlockManager { /* allocate (with prefix hits), append_slot, release (caches full blocks), block_table, ... */ }
```

The scheduler starts a request's prefill at `Allocation::cached_tokens`; the runner writes only the new positions, into the fresh blocks at the end of the table.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_prefix_hit` | unit | section 3 for A and B in both caches: 8 cached tokens, shared block ids, the hit rate | the worked example |
| `none_mode_never_shares` | unit | no hits; every block free after each request | the baseline mode |
| `last_token_is_always_computed` | boundary | C reuses 4 tokens, not 8 | the first new token always has logits |
| `same_tokens_at_another_position_do_not_hit` | boundary | D gets no hit | stale K and V would be served otherwise |
| `refcounts_balance_under_random_workload` | property | three shared prefixes, a small pool, all modes: exact outputs, the pool invariant, nothing held at the end, hits counted | no leak over weeks of traffic |
| `eviction_reclaims_cached_blocks` | fault | a pool full of cached prefixes admits new unrelated requests | the cache is spare capacity, not a leak |
| `shared_blocks_return_to_free_after_eviction` | property | after B reuses A's blocks, a request needing the whole pool fits | references taken on hits are given back |
| `failed_allocation_returns_its_cache_hits` | fault | a request too big for the pool does not pin its hits | all-or-nothing includes the hits |
| `prefix_hit_skips_prefill_tokens` | unit | the scheduler's first chunk starts at position 8 | the saving is real work skipped |
| `identical_outputs_with_and_without_cache` | differential | four requests with a shared system prompt: the same greedy tokens in all modes, at least 60 fewer prefill tokens with a cache | caching never changes answers |
| `bench_shared_prefix_ttft` | bench | TTFT with a 224-byte shared system prompt, none vs hash vs radix (budget: none / hash >= 2) | the reason to cache, measured |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Hashing a block without its parent | equal token runs at different positions share K and V: wrong attention | `hand_example_prefix_hit` (mutant `s01`) |
| Matching up to the last prompt token | no logits for the first new token, or a write into a shared full block | `last_token_is_always_computed` (mutant `s02`) |
| Keeping the request's reference on reused radix blocks | evicted blocks never become free; the pool shrinks | `shared_blocks_return_to_free_after_eviction` (mutant `s03`) |
| Not dropping hit references when allocation fails | cached blocks pinned by a request that never ran | `failed_allocation_returns_its_cache_hits` (mutant `s04`) |
| Never evicting radix leaves | a full cache refuses new prompts | `eviction_reclaims_cached_blocks` (mutant `s05`) |
| Not counting hits | `tl_engine_prefix_cache_hit_ratio` stays 0; `gw.05` routes blindly | `hand_example_prefix_hit` (mutant `s06`) |
| Registering blocks whose fill was never set | `tl_kv_register` refuses them: hash mode never hits | `hand_example_prefix_hit` (mutant `s07`) |
| Rounding the slot count down | a decode position with no block | `identical_outputs_with_and_without_cache` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.2` | `BlockSpace`: the scheduler admits, grows, and releases through it |
| Back | `L10.1` | `KvPool` (alloc, retain, release, register, lookup, set_fill) and `kv_block_hash` |
| Back | `L8.4` | `RadixCache`: match, insert, lock, evict |
| Back | `L10.1` | the Rust pool: refcounts, the cached state, LRU eviction, the prefix index |
| Forward | `L10.5` | the engine's block space; `/metrics` reports `tl_engine_prefix_cache_hit_ratio` and `tl_engine_kv_blocks` |

`L10.6` dedups KV transfers with the same block hashes (`HasBlocks`), and `gw.05` routes on the hit ratio.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| block hashes in a C index | vLLM automatic prefix caching | the same chained hashes, with extra keys (LoRA id, multimodal inputs) | [vLLM prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html) |
| a radix tree per engine | SGLang RadixAttention | cache-aware scheduling: requests sharing a prefix are scheduled together | [SGLang](https://github.com/sgl-project/sglang), `radix_cache.py` |
| LRU leaves | frequency-aware eviction | keeps hot system prompts under churn | the SGLang and vLLM eviction policies |
| one engine's cache | cluster-wide KV reuse | routing on cache contents, KV moved between engines | `gw.05`, `L10.6`, [Mooncake](https://arxiv.org/abs/2407.00079) |
