<!-- ol:module L8.4 -->
# Radix prefix cache (block-granular, LRU leaf eviction, locks)

## Overview

| | |
|---|---|
| **Module** | `L8.4` · build · Rust · Pass 6 · 3 to 4 h |
| **You build** | `rust/crates/tl-engine/src/prefix.rs`: `RadixCache` (`new`, `block_size`, `match_prefix`, `insert`, `lock`, `unlock`, `evict`, `cached_tokens`, `cached_blocks`, `evictable_blocks`, `stats`, `hit_rate`, `tree`), `PrefixMatch`, `Inserted`, `PrefixStats`, `BlockId` · the crate root `rust/crates/tl-engine/src/lib.rs` (`pub mod prefix;`) and its `Cargo.toml` (a path dependency on `tl-ds`) |
| **Contract** | the interface in section 4 (no Rust trait file yet; the tests pin it) |
| **Tests** | `course/tests/rust/l8_4.rs`, 8 tests (what they check: section 4) · fixture `course/fixtures/L8.4/radix_trace.txt` · your own tests in `rust/crates/tl-engine/tests/l84_prefix.rs`, rung R4, graded by mutation (threshold 0.80) |
| **Needs** | `ds.07` [the radix tree](../../../tracks/algorithms/16-systems-data-structures/07-radix-tree.md) · reading: `M06.2` [tries and longest-prefix match](../../../tracks/math/06-discrete-math-2/02-trees-and-tries-longest-prefix-match.md) |
| **Used by** | later: `L10.4` (the block manager's `--prefix-cache=radix`), `gw.05` (routes on the hit metrics over gRPC) |
| **Milestone** | `MS-L8` (step 5: these tests pass from `cargo test`) |
| **Optional depth** | Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104), section 3 (RadixAttention) |

## Key Takeaways

- A prefix cache maps the token prefix of a new prompt to KV blocks already computed; block granularity means only whole blocks are matched or stored (`hand_example_shared_system_prompt`, `insert_takes_exactly_the_full_blocks`).
- A match never covers the last prompt token: prefill must run on at least one token to produce the first output's logits (`the_last_prompt_token_is_never_covered`).
- Two requests can compute the same prefix at once; the cache keeps the first one's blocks and hands the other's back for the caller to free (`a_prefix_computed_twice_is_kept_once`).
- A running request locks the node its match ended at, which pins the whole prefix; eviction frees the least recently used unlocked leaves (`a_locked_prefix_survives_eviction`, `eviction_frees_the_least_recently_used_first`).
- Every block id is in exactly one place, the free list, a running request, or the cache, under any interleaving (`blocks_are_conserved_and_locked_prefixes_stay`).

## How to work this chapter

```bash
ol start L8.4               # stubs prefix.rs and the tl-engine crate root; writes Cargo.toml if absent
ol tests L8.4               # read the test catalog first
ol check L8.4               # exit code is the verdict; then grades your tests by mutation
ol check L8.4 --ref-deps    # only if ds.07 is not passing yet
ol diff  L8.4               # after passing: your code against the reference
```

Add `crates/tl-engine` to your workspace `rust/Cargo.toml` if `ol start` did not. The whole module is about 120 lines over `ds.07`: the arithmetic of section 2, the duplicates, and the counters.

---

## 1. Why now

`rt.04` can name a full KV block by its chained hash and find it again, block by block. An engine admitting a request wants more: the longest cached prefix of the prompt in one walk, eviction that respects which prefixes are hot, and a guarantee that the blocks a running request reads are never reclaimed under it. Your radix tree (`ds.07`) does exactly that at any granularity. This module fixes the rules a serving engine needs on top of it, so that `L10.4`'s block manager can choose between the hash index of `rt.04` (`--prefix-cache=hash`) and this tree (`--prefix-cache=radix`) and measure both.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | `block_size`, tokens per KV block (16 in the engine) | `usize` |
| $n$ | prompt length in tokens | `usize` |
| $u$ | usable prefix: $\lfloor (n - 1) / B \rfloor \cdot B$ | `usize` |
| matched | tokens covered by cached blocks, a multiple of $B$, at most $u$ | `usize` |
| hit rate | hit tokens over query tokens, over every lookup | `f64` |

**Block granular.** The cache is `RadixTree<BlockId>` at granule $B$: one block id per $B$ tokens of every label, matches and splits only at block boundaries. A partial last block is the request's own and is never inserted.

**The last token is computed.** The model must run on at least one prompt position to produce the logits of the first new token. So `match_prefix` searches only the first $u = \lfloor (n-1)/B \rfloor \cdot B$ tokens: with $B = 2$, a fully cached 4-token prompt matches 2 tokens, and the request prefills tokens 3 and 4.

**One owner per stored block.** `insert(tokens, blocks)` takes `blocks[i]` for tokens $[iB, (i+1)B)$, exactly $\lfloor n / B \rfloor$ of them. For positions already cached, the tree keeps its block; if the caller passed a different id there (it prefilled the same prefix itself, racing another request), that id comes back in `duplicates` and the caller frees it. A caller that matched first passes the cached ids themselves, and nothing comes back.

**Locks and eviction.** `lock(node)` pins the prefix ending at `node` (the node and its ancestors) while a request reads it; `unlock` releases it when the request finishes. `evict(n)` frees at least $n$ blocks if it can, least recently used unlocked leaves first, whole leaves at a time, and returns their ids for the block manager to give back to the pool. A match or insert counts as a use of every node on its path.

**Counters.** `stats()` reports lookups, query tokens, hit tokens, inserted blocks, and evicted blocks; `hit_rate()` is hit tokens over query tokens. The gateway's affinity routing (`gw.05`) reads them over gRPC.

## 3. Worked example by hand

$B = 2$. Request A ran prompt `1 2 3 4 5`: blocks 10 (tokens 1 2) and 11 (tokens 3 4) are full; token 5's block is partial and stays A's.

| Step | Call | Result | Why |
|---|---|---|---|
| 1 | `insert([1,2,3,4,5], [10, 11])` | cached tokens 4 | 5 tokens hold 2 full blocks |
| 2 | B: `match_prefix([1,2,3,4,9,9])` | 4 tokens, blocks `[10, 11]` | $u = \lfloor 5/2 \rfloor \cdot 2 = 4$ |
| 3 | C: `match_prefix([1,2,3,4])` | 2 tokens, blocks `[10]` | $u = \lfloor 3/2 \rfloor \cdot 2 = 2$: C prefills tokens 3 and 4 |
| 4 | `match_prefix([7,7,7])` | 0 tokens | a miss |

After the three lookups: query tokens $6 + 4 + 3 = 13$, hit tokens $4 + 2 + 0 = 6$, hit rate $6/13$. This is `hand_example_shared_system_prompt`.

## 4. The interface

```rust
// rust/crates/tl-engine/src/prefix.rs, over tl_ds::radix (ds.07)
pub type BlockId = u32;
pub use tl_ds::radix::NodeId;
pub struct PrefixMatch { pub matched_tokens: usize, pub blocks: Vec<BlockId>, pub node: NodeId }
pub struct Inserted { pub node: NodeId, pub duplicates: Vec<BlockId> }
pub struct PrefixStats { pub lookups: u64, pub query_tokens: u64, pub hit_tokens: u64,
                         pub inserted_blocks: u64, pub evicted_blocks: u64 }
impl RadixCache {
    pub fn new(block_size: usize) -> Self;                         // panics on 0
    pub fn block_size(&self) -> usize;
    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch; // at most (n - 1) / B * B tokens
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId]) -> Inserted;  // blocks.len() == n / B
    pub fn lock(&mut self, node: NodeId);
    pub fn unlock(&mut self, node: NodeId);
    pub fn evict(&mut self, n_blocks: usize) -> Vec<BlockId>;
    pub fn cached_tokens(&self) -> usize;
    pub fn cached_blocks(&self) -> usize;
    pub fn evictable_blocks(&self) -> usize;
    pub fn stats(&self) -> PrefixStats;
    pub fn hit_rate(&self) -> f64;
    pub fn tree(&self) -> &tl_ds::radix::RadixTree<BlockId>;
}
```

The design sketch (DESIGN 4.3) had `insert` return a `NodeId`; it returns `Inserted` so the duplicates have an owner. When nothing matched, `PrefixMatch.node` is `ROOT`, and locking it does nothing.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_shared_system_prompt` | unit | section 3: matches 4, 2, and 0 tokens; the counters and hit rate | you and the tests agree on the rules |
| `the_last_prompt_token_is_never_covered` | boundary | prompts of 0 to 7 tokens over 6 cached tokens match 0, 0, 0, 2, 2, 4, 4, 6 | prefill always has a token to run |
| `insert_takes_exactly_the_full_blocks` | boundary | a partial block among the blocks panics | a growing block is never shared |
| `a_prefix_computed_twice_is_kept_once` | unit | a racing request's blocks come back as duplicates; the cached ids themselves do not | the block manager frees the duplicates |
| `a_locked_prefix_survives_eviction` | unit | eviction frees only unlocked blocks until the request unlocks; the counts follow | running requests keep their KV |
| `eviction_frees_the_least_recently_used_first` | unit | three prompts, one re-used: the other two go first, oldest first | hot prompts stay cached |
| `golden_reference_trace` | golden | 300 operations of a recorded workload replayed against an independent Python model: every match, duplicate, eviction order, and cached count | your cache and the oracle agree on every rule at once |
| `blocks_are_conserved_and_locked_prefixes_stay` | property | 3,000 seeded engine steps over 64 block ids: each id in exactly one place, no locked block evicted, longest prefix found right after an insert, all 64 back at the end | the accounting L10.4 builds on |

**Your tests (rung R4).** Write `rust/crates/tl-engine/tests/l84_prefix.rs` (an integration test: `use tl_engine::prefix::...` only), with properties in prose turned into code (`proptest` is available as a dev-dependency): "a fully cached prompt matches all but its last block", "inserting what a match returned hands nothing back", "a locked prefix is never evicted", "blocks in equal blocks out". The planted bugs "a fully cached prompt is matched whole" (`s01`) and "lock pins nothing" (`s07`) are required; overall 80%.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Matching a fully cached prompt whole | prefill has no token to run; the first output has no logits | `the_last_prompt_token_is_never_covered`, `hand_example_shared_system_prompt` (mutant `s01`) |
| Returning only the last node's blocks | a prefix spanning several nodes loses its first blocks | `a_prefix_computed_twice_is_kept_once`, `golden_reference_trace` (mutant `s02`) |
| Counting the searched length as a hit | the hit rate the gateway routes on is inflated | `hand_example_shared_system_prompt` (mutant `s03`) |
| Accepting a block for the partial tail | a block whose contents still grow becomes shared | `insert_takes_exactly_the_full_blocks` (mutant `s04`) |
| Handing back the cached ids themselves | the caller frees blocks the cache still holds | `a_prefix_computed_twice_is_kept_once`, `golden_reference_trace` (mutant `s05`) |
| Inserting the tokens of the partial tail | the tree asserts, or caches a half-written block | `hand_example_shared_system_prompt`, `golden_reference_trace` (mutant `s06`) |
| A lock that pins nothing | a running request's blocks are evicted and reused | `a_locked_prefix_survives_eviction`, `blocks_are_conserved_and_locked_prefixes_stay` (mutant `s07`) |
| An unlock that releases nothing | the cache fills with pinned blocks and admission stalls | `a_locked_prefix_survives_eviction` (mutant `s08`) |
| Truncating an evicted leaf's blocks to the request | the rest of the leaf's blocks leak | `eviction_frees_the_least_recently_used_first`, `a_locked_prefix_survives_eviction` (mutant `s09`) |
| Counting locked blocks as evictable | the scheduler admits a request that cannot get blocks | `a_locked_prefix_survives_eviction` (mutant `s10`) |
| Not counting evicted blocks | `/metrics` shows no pressure | `a_locked_prefix_survives_eviction` (mutant `m01`) |
| Reporting cached tokens in blocks | every capacity log is off by a factor of $B$ | `hand_example_shared_system_prompt`, `golden_reference_trace` (mutant `m02`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ds.07` | `RadixTree<BlockId>` at granule $B$: matching, splitting, locks, LRU eviction |
| Back | `M06.2` | longest-prefix match, from the tokenizer's trie |
| Forward | `L10.4` | the block manager matches each new request, locks its prefix, allocates the rest from `rt.04`, inserts after prefill, and evicts under pressure; `--prefix-cache=radix` |
| Forward | `gw.05` | prefix-affinity routing reads `hit_rate` and the cached token counts over `tl.control.v1` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `RadixCache` | SGLang `RadixCache` | page-granular matching (`page_size`), a priority heap for eviction, host-memory offload tiers (`HiRadixCache`) | [`python/sglang/srt/mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) |
| hash-or-radix choice | vLLM automatic prefix caching | hash-per-block lookups instead of a tree, with the same "full blocks only" rule | [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) |
| hit-rate counters | llm-d KV-cache-aware routing | a cluster-wide index of which replica holds which prefix blocks | [llm-d](https://github.com/llm-d/llm-d) |
