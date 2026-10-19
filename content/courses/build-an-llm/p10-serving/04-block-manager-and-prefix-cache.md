# Block manager with prefix cache (none, hash, radix)

## Overview

| | |
|---|---|
| **Module** | `L10.4` · build · Rust · Pass 7 · 8 to 12 h |
| **Solution** | [`rust/crates/tl-engine/src/block_manager.rs`](../../../../course/solutions/rust/crates/tl-engine/src/block_manager.rs): `BlockManager` (`match_prefix`, `take`, `drop_refs`, and the `BlockSpace` methods), the scheduler's `BlockSpace` over the Rust KV pool of `L10.1`, with three prefix-cache modes: `none`, `hash` (the pool's own index of chained block hashes), and `radix` (the `L8.4` radix cache, [`prefix.rs`](../../../../course/solutions/rust/crates/tl-engine/src/prefix.rs)) |
| **Contract** | the pool semantics of [`tinyllm/kv_pool.h`](../../../../course/contracts/c/include/tinyllm/kv_pool.h) (which the Rust `KvPool` mirrors) and the block hash in [`formats/kv-block.md`](../../../../course/contracts/formats/kv-block.md); `prefix_cache` of `[engine]` in [`config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) |
| **Needs** | `L10.2` the `BlockSpace` seam ([chapter](02-continuous-batching.md)) · `L10.1` `KvPool` and `kv_block_hash` ([chapter](01-model-runner-and-sampler.md)) · `L8.4` the radix prefix cache ([chapter](../p08-inference/04-radix-prefix-cache.md)) · `L10.1` the pool's refcounts, prefix index, and LRU · reading: `ds.02` the pool's hash table, `ds.07` the radix tree, `L8.3` paged KV |
| **Used by** | `L10.5` (the engine creates one `BlockManager` over the runner's pool, in the mode runtime.toml names) |
| **Optional depth** | [Zheng et al. 2024, SGLang and RadixAttention](https://arxiv.org/abs/2312.07104) (free); [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching.html) (free); [Kwon et al. 2023, PagedAttention](https://arxiv.org/abs/2309.06180) (free), section 4.4 on sharing |

## Key Takeaways

- A position's K and V depend only on the tokens up to it, so requests that share a prefix can share the blocks of that prefix; a cache of finished requests' full blocks turns repeated system prompts into skipped prefill.
- A block is named by its tokens **and** everything before it: the chained hash folds in the parent's hash, the radix tree the path from the root.
- At most `len - 1` prompt tokens are reused, rounded down to a block: the model must still run on the last prompt token to produce the first new one.
- Sharing is by reference count; a request gives back every reference it took, including after a failed allocation, or cached blocks can never be evicted.
- The cache changes how much is computed, never what: outputs are identical in all three modes.

## How to read this chapter

Sections 1 to 3 derive what may be shared and why, describe the three modes over one pool, and run four requests through the cache by hand. Section 4 fixes the Rust interface and lists what a correct block manager must handle. Section 5 walks through the solution, [`block_manager.rs`](../../../../course/solutions/rust/crates/tl-engine/src/block_manager.rs), with a close eye on where every reference is taken and given back. Try to predict each step of the solution before reading it.

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
2. **The prefix, not just the tokens.** Block $i$ is reusable only when every block before it matches too. The hash chains: $h_i = \mathrm{FNV}(h_{i-1} \,\|\, x_{iB} \,\|\, \dots \,\|\, x_{(i+1)B - 1})$ with $h_{-1} = 0$, each id as 4 little-endian bytes and the parent as 8, and 0 replaced by 1 (`formats/kv-block.md`, `kv_block_hash`). The radix tree gets the same effect by matching paths from the root.
3. **The last token is always computed.** A match covers at most $\lfloor (n - 1) / B \rfloor \cdot B$ tokens: the model must run on the last prompt token to produce the logits of the first new one, so even a fully cached prompt recomputes its last block.

### 2.2 Three modes over one pool

- **none**: `allocate` takes $\lceil n / B \rceil$ fresh blocks; `release` drops every reference, so blocks go back to free.
- **hash**: on `release`, each full computed block $i$ is registered in the pool's prefix index under $h_i$ (`KvPool::register`; its fill must be $B$, which the block manager sets, so a block is full by the time it is named). When its last reference goes, a registered block becomes **cached**, not free; `KvPool::alloc` evicts cached blocks least recently used first when free ones run out. `allocate` walks the prompt's full blocks front to back, looking up each $h_i$ with `KvPool::lookup` (a hit takes a reference: a cached block becomes used again) and stops at the first miss.
- **radix**: the radix cache of `L8.4` maps token runs to block ids. On `release` the full computed blocks are inserted; the cache keeps the request's reference to each block it newly stores, so a cached block has $\rho = 1$ (the cache's). `allocate` matches the longest cached prefix, takes a reference on each matched block for the request, and locks the matched node so eviction cannot take it while the request runs. When the pool's free list is short, the block manager evicts least recently used unlocked leaves and drops the cache's reference on each.

In both caches a reused block has $\rho \ge 2$ while a request holds it (the cache, the request), and $\rho$ returns to the cache's count when the request ends. The invariant `free + used + cached == n_blocks` (`L10.1`) holds at every step; at quiet times, `used` is 0 in hash mode and equals the cache's blocks in radix mode.

### 2.3 Hash or radix

The hash index is flat and O(1) per block, lives in the pool, and needs nothing more; a lookup walks blocks one hash at a time, and eviction is the pool's own LRU over blocks. The radix tree matches a whole prefix in one walk, keeps the structure of shared prefixes visible (a conversation tree), and evicts whole leaves (the newest suffixes) first. Both give the same hits on the same traffic in this engine; a shared-prefix TTFT benchmark compares both, and `gw.05` routes requests to the engine whose cache is warm using the hit ratio exported here.

## 3. Worked example by hand

Blocks of $B = 4$, an 8-block pool, either cache.

**A** = tokens $1, 2, \dots, 10$ arrives to an empty cache: `allocate` finds nothing and takes $\lceil 10 / 4 \rceil = 3$ fresh blocks. A finishes; `release` with its 10 computed tokens has $\lfloor 10 / 4 \rfloor = 2$ full blocks: $[1, 2, 3, 4]$ and $[5, 6, 7, 8]$. Both are cached; the third block (tokens 9, 10) is partial and goes back to free.

**B** = $1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97$ (11 tokens). Usable: $\lfloor (11 - 1) / 4 \rfloor = 2$ blocks. Block 0's hash $h_0 = \mathrm{FNV}(0 \,\|\, 1, 2, 3, 4)$ hits, block 1's $h_1 = \mathrm{FNV}(h_0 \,\|\, 5, 6, 7, 8)$ hits: 8 cached tokens. B needs $\lceil 11/4 \rceil = 3$ blocks, so one fresh one. Its table starts with A's two blocks, and its prefill starts at position 8: 3 tokens computed instead of 11. The hit rate so far is $8 / (10 + 11) = 0.381$.

**C** = $1, \dots, 8$ (exactly A's two full blocks). Usable: $\lfloor 7 / 4 \rfloor = 1$ block, so 4 cached tokens, and the last block is recomputed to produce C's first logits.

**D** = $5, 6, 7, 8, \dots$: its first block holds the same tokens as A's second, but $\mathrm{FNV}(0 \,\|\, 5, 6, 7, 8) \ne h_1$, and the radix tree has no root edge starting with 5: no hit, correctly.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the prefix-hit hand example | unit | section 3 for A and B in both caches: 8 cached tokens, shared block ids, the hit rate | the worked example |
| `none` never shares | unit | no hits; every block free after each request | the baseline mode |
| the last token is always computed | boundary | C reuses 4 tokens, not 8 | the first new token always has logits |
| same tokens at another position | boundary | D gets no hit | stale K and V would be served otherwise |
| refcounts balance under a random workload | property | three shared prefixes, a small pool, all modes: exact outputs, the pool invariant, nothing held at the end, hits counted | no leak over weeks of traffic |
| eviction reclaims cached blocks | fault | a pool full of cached prefixes admits new unrelated requests | the cache is spare capacity, not a leak |
| shared blocks return to free | property | after B reuses A's blocks, a request needing the whole pool fits | references taken on hits are given back |
| a failed allocation returns its hits | fault | a request too big for the pool does not pin its hits | all-or-nothing includes the hits |
| a hit skips prefill tokens | unit | the scheduler's first chunk starts at position 8 | the saving is real work skipped |
| identical outputs in every mode | differential | four requests with a shared system prompt: the same greedy tokens in all modes, at least 60 fewer prefill tokens with a cache | caching never changes answers |
| time to first token with a shared prefix | benchmark | TTFT with a 224-byte shared system prompt, none vs hash vs radix: none / hash should be at least 2 | the reason to cache, measured |

## 5. The solution, step by step

### 5.1 From first principles

1. **K and V at position $j$ depend only on $x_0..x_j$** (section 2.1). So a cached block is valid for a new request exactly when the new prompt agrees with the cached one on every token up to the block's end. That forces naming a block by its whole prefix: a chained hash, or a path in a radix tree.
2. **Only full blocks are shared, and a full block is never written again**, so sharing needs only reference counts, no copy on write. A request's fresh blocks are always appended after the matched ones, so all its writes go to blocks it owns alone.
3. **At most $\lfloor (n-1)/B \rfloor$ blocks may be matched**, because the last prompt token must be fed to get the first new token's logits. In the example, C (8 tokens) matches 1 block, not 2.
4. **Every reference taken must be given back on every path:** on release, and on a failed allocation. Otherwise a cached block is pinned and can never be evicted, and the pool shrinks.
5. **Caching happens at release time**, when the request's computed tokens are known: register (hash) or insert (radix) its full computed blocks, then drop the request's references. Blocks the cache now owns stay out of the free list until evicted.
6. **The cache is spare capacity**, so allocation must be able to reclaim it: the pool's LRU does this in hash mode; in radix mode the block manager evicts leaves before asking the pool.

### 5.2 The code, annotated

**Matching a prefix.** `usable` encodes rule 3. In hash mode each block's hash chains the previous one (`parent`), and the walk stops at the first miss: a later block cannot be valid if an earlier one is not. Each `lookup` hit takes a reference for the request. In radix mode one `match_prefix` walks the tree, the end node is locked against eviction, and each matched block is retained for the request. Hits are counted in both modes.

```rust
    fn match_prefix(&mut self, tokens: &[u32]) -> (Vec<u32>, Option<NodeId>) {
        let bt = self.block_tokens;
        let usable = tokens.len().saturating_sub(1) / bt;   // never the block holding the last token
        let mut blocks = Vec::new();
        let mut node = None;
        match self.mode {
            PrefixCache::None => return (blocks, None),
            PrefixCache::Hash => {
                let mut pool = lock(&self.pool).expect("pool lock");
                let mut parent = 0u64;                        // h_{-1} = 0
                for i in 0..usable {
                    let h = kv_block_hash(parent, &tokens[i * bt..(i + 1) * bt]); // chained: prefix-aware
                    match pool.lookup(h) {
                        Some(b) => blocks.push(b),            // lookup takes a reference
                        None => break,                        // first miss ends the prefix
                    }
                    parent = h;
                }
            }
            PrefixCache::Radix => {
                let r = self.radix.as_mut().expect("radix mode has a tree");
                let m = r.match_prefix(tokens);
                r.lock(m.node);                               // not evictable while we run
                node = Some(m.node);
                let mut pool = lock(&self.pool).expect("pool lock");
                for &b in &m.blocks {
                    pool.retain(b).expect("a cached block is used by the cache"); // our reference
                }
                blocks = m.blocks;
            }
        }
        self.stats.lookups += 1;
        self.stats.query_tokens += tokens.len() as u64;
        self.stats.hit_tokens += (blocks.len() * bt) as u64;  // feeds the hit-ratio metric
        (blocks, node)
    }
```

For B in the example, `usable` is $\lfloor 10/4 \rfloor = 2$, both hashes hit, and `hit_tokens` becomes 8.

**Allocating is all or nothing, hits included.** The fresh blocks are what is left after the match; if they cannot be taken, the references the match took are dropped (and the radix node unlocked) before returning `NoCapacity`.

```rust
    fn allocate(&mut self, id: RequestId, tokens: &[u32]) -> Result<Allocation, NoCapacity> {
        if self.tables.contains_key(&id) { return Err(NoCapacity); }
        let (matched, node) = self.match_prefix(tokens);
        let need = tokens.len().div_ceil(self.block_tokens) - matched.len(); // round UP: every position has a slot
        match self.take(need) {
            Ok(fresh) => {
                let n = matched.len();
                let mut blocks = matched;
                blocks.extend(fresh);                          // shared prefix first, own blocks after
                self.tables.insert(id, Table { blocks, matched: n, node });
                Ok(Allocation { cached_tokens: n * self.block_tokens }) // prefill starts here
            }
            Err(e) => {
                self.drop_refs(&matched, node);                // do not pin hits for a request that never runs
                Err(e)
            }
        }
    }
```

**Taking fresh blocks** evicts radix leaves first when the pool's free list is short (each evicted block loses the cache's reference and so becomes free), after checking that free plus evictable suffices. In hash mode `pool.alloc` does the eviction itself from its LRU, and the manager just counts it.

```rust
    fn take(&mut self, n: usize) -> Result<Vec<u32>, NoCapacity> {
        if n == 0 { return Ok(Vec::new()); }
        let mut pool = lock(&self.pool).map_err(|_| NoCapacity)?;
        if let Some(r) = self.radix.as_mut() {
            let free = pool.stats().free as usize;
            if free < n {
                if free + r.evictable_blocks() < n { return Err(NoCapacity); } // check before evicting
                for b in r.evict(n - free) {                  // least recently used unlocked leaves
                    pool.release(b).expect("the radix cache holds one reference to each cached block");
                    self.stats.evictions += 1;
                }
            }
        }
        let before = pool.stats().evictions;
        let ids = pool.alloc(n).map_err(|_| NoCapacity)?;     // hash mode: the pool's LRU evicts here
        self.stats.evictions += (pool.stats().evictions - before) as u64;
        Ok(ids)
    }
```

**Releasing caches the full computed blocks.** `full` is the number of complete blocks of computed K and V. In hash mode, each of the request's *own* full blocks (index at least `matched`) gets its fill set to $B$ (the pool registers only full blocks) and is registered under its chained hash; a duplicate registration returns `Ok(false)` and the block simply returns to free. Then every reference is dropped: registered blocks become cached, the rest free.

```rust
            PrefixCache::Hash => {
                {
                    let mut pool = lock(&self.pool).expect("pool lock");
                    let mut parent = 0u64;
                    for i in 0..full {
                        let h = kv_block_hash(parent, &computed[i * bt..(i + 1) * bt]);
                        let b = t.blocks[i];
                        if i >= t.matched {                   // matched blocks are already registered
                            if pool.fill(b) < bt as u32 {
                                pool.set_fill(b, bt as u32).expect("an unregistered block of ours");
                            }
                            let _ = pool.register(b, h);      // Ok(false): someone else holds this prefix
                        }
                        parent = h;                           // the chain continues through matched blocks
                    }
                }
                self.drop_refs(&t.blocks, None);              // refcount 0 + hash = cached, else free
            }
```

In radix mode, the node is unlocked first (an insert may split it), the full computed blocks are inserted, and the cache *keeps* the request's reference on each block it newly stored. Every other reference goes back: matched blocks (the cache already had its own), partial blocks past `full`, and duplicates the tree already held.

```rust
            PrefixCache::Radix => {
                let r = self.radix.as_mut().expect("radix mode");
                if let Some(n) = t.node { r.unlock(n); }      // before insert: the node may split
                let ins = r.insert(&computed[..full * bt], &t.blocks[..full]);
                let give_back: Vec<u32> = t.blocks.iter().enumerate()
                    .filter(|&(i, b)| i < t.matched || i >= full || ins.duplicates.contains(b))
                    .map(|(_, &b)| b)
                    .collect();
                self.drop_refs(&give_back, None);             // the cache's refs stay; ours go
            }
```

After A in the example: blocks 0 and 1 hold refcount 1 (the cache), and A's partial third block is free. After B reuses them and ends: back to 1 each, so a later request needing the whole pool can evict them.

**`free_blocks`** reports free plus cached (hash) plus evictable (radix), because all of those can be handed out; the scheduler's admission and `NeverFits` logic then treat the cache as spare capacity.

Full solution: [`block_manager.rs`](../../../../course/solutions/rust/crates/tl-engine/src/block_manager.rs). The radix tree is [`prefix.rs`](../../../../course/solutions/rust/crates/tl-engine/src/prefix.rs) (`L8.4`); the pool's `lookup`, `register`, and LRU are in [`kv.rs`](../../../../course/solutions/rust/crates/tl-engine/src/kv.rs) (`L10.1`).

### 5.3 Why this solution

- **Cost.** Hash mode: one FNV over $B$ ids and one hash-map lookup per block, $O(n)$ per allocation; radix mode: one tree walk, $O(n)$ token comparisons. Both are tiny next to the prefill they save ($O(n \cdot d^2 \cdot L)$ per skipped token block).
- **Why two caches?** They give the same hits here, but differ operationally: the hash index lives in the pool, is flat, and evicts per block LRU; the radix tree shows the shape of shared prefixes and evicts whole leaves (newest suffixes) first. Having both behind one `BlockSpace` lets the engine choose by configuration.
- **Why share only full blocks?** Sharing partial blocks would need copy on write when the next token is appended. Full-block sharing gives up at most $B - 1$ cached tokens per prefix and removes the copy entirely.
- **Why cache at release rather than as soon as a block fills?** At release the computed tokens are final, and the request's own references are known, so ownership can be handed to the cache in one place. Caching earlier would let concurrent requests share sooner, at the cost of more bookkeeping (vLLM does this).
- **Tradeoff accepted:** the last block of a fully cached prompt is recomputed. It costs up to $B$ tokens of prefill per request, and buys the guarantee that the first new token always has logits and no shared block is ever written.

### 5.4 Explain it back

1. Why does the hash of block $i$ include $h_{i-1}$?
   <details><summary>Answer</summary>K and V depend on every earlier token, not just the block's own. Without the parent, D's block $[5, 6, 7, 8]$ at position 0 would match A's block at position 4, and D would attend to keys computed for a different context.</details>
2. Why does `allocate` drop the matched references when `take` fails?
   <details><summary>Answer</summary>The match took a reference on each hit. If the request is not admitted, those references would never be released, so the blocks could never return to cached or free: a permanent leak of exactly the most useful blocks.</details>
3. Why does `release` in hash mode register only blocks with index at least `matched`?
   <details><summary>Answer</summary>Matched blocks came from the index and are already registered under the same hashes. Registering them again would be a no-op at best; the request's own new full blocks are the only ones to add.</details>
4. Why does `free_blocks` count cached blocks as free?
   <details><summary>Answer</summary>A cached block has no running request on it and can be evicted whenever an allocation needs it. Counting it as unavailable would make the scheduler refuse admissions the pool can actually serve.</details>
5. In radix mode, why is the matched node unlocked before `insert`?
   <details><summary>Answer</summary>Inserting a longer sequence can split the node where the earlier match ended. Unlocking first means the lock count is released on the node that exists now, not on a node the split has replaced.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Hashing a block without its parent | equal token runs at different positions share K and V: wrong attention | `kv_block_hash(parent, ...)`, `parent = h` each step |
| Matching up to the last prompt token | no logits for the first new token, or a write into a shared full block | `usable = (len - 1) / bt` |
| Keeping the request's reference on reused radix blocks | evicted blocks never become free; the pool shrinks | `give_back` includes every `i < matched` |
| Not dropping hit references when allocation fails | cached blocks pinned by a request that never ran | `drop_refs(&matched, node)` on `Err` |
| Never evicting radix leaves | a full cache refuses new prompts | `take` calls `r.evict(n - free)` |
| Not counting hits | `tl_engine_prefix_cache_hit_ratio` stays 0; `gw.05` routes blindly | `hit_tokens` and `query_tokens` updated in `match_prefix` |
| Registering blocks whose fill was never set | the pool refuses them: hash mode never hits | `set_fill(b, bt)` before `register` |
| Rounding the slot count down | a decode position with no block | `div_ceil` in `allocate` and `append_slot` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.2` | `BlockSpace`: the scheduler admits, grows, and releases through it |
| Back | `L10.1` | `KvPool` (alloc, retain, release, register, lookup, set_fill) and `kv_block_hash` |
| Back | `L8.4` | `RadixCache`: match, insert, lock, evict |
| Back | `L10.1` | the Rust pool: refcounts, the cached state, LRU eviction, the prefix index |
| Forward | `L10.5` | the engine's block space; `/metrics` reports `tl_engine_prefix_cache_hit_ratio` and `tl_engine_kv_blocks` |

`L10.6` dedups KV transfers with the same block hashes (`HasBlocks`), and `gw.05` routes on the hit ratio.

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| block hashes in the pool's index | vLLM automatic prefix caching | the same chained hashes, with extra keys (LoRA id, multimodal inputs) | [vLLM prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching.html) |
| a radix tree per engine | SGLang RadixAttention | cache-aware scheduling: requests sharing a prefix are scheduled together | [SGLang](https://github.com/sgl-project/sglang), `radix_cache.py` |
| LRU leaves | frequency-aware eviction | keeps hot system prompts under churn | the SGLang and vLLM eviction policies |
| one engine's cache | cluster-wide KV reuse | routing on cache contents, KV moved between engines | `gw.05`, `L10.6`, [Mooncake](https://arxiv.org/abs/2407.00079) |
