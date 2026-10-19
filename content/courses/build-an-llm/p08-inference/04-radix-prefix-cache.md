# A radix prefix cache

Many requests begin the same way: a shared system prompt, a few-shot preamble, the earlier turns of a conversation. Their KV blocks are identical, so computing them again for every request is pure waste. [The paged KV block pool](08-paged-kv-block-pool.md) can name a full KV block by its chained hash and find it again, block by block. An engine admitting a request wants more: the longest cached prefix of the prompt in one walk, eviction that respects which prefixes are hot, and a guarantee that the blocks a running request reads are never reclaimed under it.

[The radix tree](../../algorithms/16-systems-data-structures/07-radix-tree.md) of *Algorithms* does exactly that at any granularity. This section fixes the rules a serving engine needs on top of it, so that [the engine's block manager](../p10-serving/04-block-manager-and-prefix-cache.md) can choose between the block pool's hash index (`--prefix-cache=hash`) and this tree (`--prefix-cache=radix`) and measure both. The idea is SGLang's RadixAttention. As background, [trees and tries, longest-prefix match](../../mathematics-for-computing/06-discrete-math-2/02-trees-and-tries-longest-prefix-match.md) in *Mathematics for Computing* covers the underlying lookup.

The solution is Rust: [`prefix.rs`](../../../../course/solutions/rust/crates/tl-engine/src/prefix.rs) in the `tl-engine` crate (exported from [`lib.rs`](../../../../course/solutions/rust/crates/tl-engine/src/lib.rs) as `pub mod prefix;`), about 120 lines over the radix tree in [`radix.rs`](../../../../course/solutions/rust/crates/tl-ds/src/radix.rs). There is no separate trait file; the interface given below is the contract.

> **Learning objectives**
> - Compute the usable prefix $u = \lfloor (n-1)/B \rfloor \cdot B$ and explain why a match never covers the last prompt token.
> - Explain why the cache stores block ids, at block granularity, and never KV data.
> - Identify which handed-back ids on insert are true duplicates from a race.
> - Explain how locks pin a prefix path and why eviction takes only whole LRU leaves.
> - Keep every block id in exactly one place under any interleaving.

## Block granularity

These quantities appear throughout.

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | `block_size`, tokens per KV block (16 in the engine) | `usize` |
| $n$ | prompt length in tokens | `usize` |
| $u$ | usable prefix: $\lfloor (n - 1) / B \rfloor \cdot B$ | `usize` |
| matched | tokens covered by cached blocks, a multiple of $B$, at most $u$ | `usize` |
| hit rate | hit tokens over query tokens, over every lookup | `f64` |

The cache is `RadixTree<BlockId>` at granule $B$: one block id per $B$ tokens of every label, with matches and splits only at block boundaries. A partial last block is the request's own and is never inserted.

## The last token is computed

The model must run on at least one prompt position to produce the logits of the first new token. So `match_prefix` searches only the first $u = \lfloor (n-1)/B \rfloor \cdot B$ tokens. With $B = 2$, a fully cached 4-token prompt matches 2 tokens, and the request prefills tokens 3 and 4.

## One owner per stored block

Two requests can compute the same prefix at once, and the cache has to settle who owns the result. `insert(tokens, blocks)` takes `blocks[i]` for tokens $[iB, (i+1)B)$, exactly $\lfloor n / B \rfloor$ of them. For positions already cached, the tree keeps its block. If the caller passed a different id there (it prefilled the same prefix itself, racing another request), that id comes back in `duplicates` and the caller frees it. A caller that matched first passes the cached ids themselves, and nothing comes back.

## Locks and eviction

A running request must not lose the blocks it reads. `lock(node)` pins the prefix ending at `node` (the node and its ancestors) while a request reads it; `unlock` releases it when the request finishes. `evict(n)` frees at least $n$ blocks if it can, least recently used unlocked leaves first, whole leaves at a time, and returns their ids for the block manager to give back to the pool. A match or insert counts as a use of every node on its path.

Finally, the cache keeps counters. `stats()` reports lookups, query tokens, hit tokens, inserted blocks, and evicted blocks; `hit_rate()` is hit tokens over query tokens. The gateway's [prefix-affinity routing](../../ai-platform-engineering/12-gateway/05-routing.md) in *AI Platform Engineering* reads them over gRPC.

## Working it by hand

Take $B = 2$. Request A ran prompt `1 2 3 4 5`: blocks 10 (tokens 1 2) and 11 (tokens 3 4) are full; token 5's block is partial and stays A's. Then three more lookups arrive.

| Step | Call | Result | Why |
|---|---|---|---|
| 1 | `insert([1,2,3,4,5], [10, 11])` | cached tokens 4 | 5 tokens hold 2 full blocks |
| 2 | B: `match_prefix([1,2,3,4,9,9])` | 4 tokens, blocks `[10, 11]` | $u = \lfloor 5/2 \rfloor \cdot 2 = 4$ |
| 3 | C: `match_prefix([1,2,3,4])` | 2 tokens, blocks `[10]` | $u = \lfloor 3/2 \rfloor \cdot 2 = 2$: C prefills tokens 3 and 4 |
| 4 | `match_prefix([7,7,7])` | 0 tokens | a miss |

After the three lookups: query tokens $6 + 4 + 3 = 13$, hit tokens $4 + 2 + 0 = 6$, hit rate $6/13$.

## Designing the API

```rust
// rust/crates/tl-engine/src/prefix.rs, over tl_ds::radix
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

An earlier design sketch had `insert` return a `NodeId`; it returns `Inserted` so the duplicates have an owner. When nothing matched, `PrefixMatch.node` is `ROOT`, and locking it does nothing.

A few properties are worth keeping in mind, and testing if you write a variant: a fully cached prompt matches all but its last block; inserting what a match returned hands nothing back; a locked prefix is never evicted; blocks in equal blocks out.

## Building it step by step

### The reasoning

1. **What is cached is a mapping from token prefixes to block ids, not KV data.** The KV tensors live in the pool ([the model runner](../p10-serving/01-model-runner-and-sampler.md)); the block manager moves references. So the value type is `BlockId = u32`, and the cache never allocates or frees memory itself: every id that leaves it (as a duplicate or an eviction) is returned to the caller.
2. **Prefix lookup over shared prefixes is exactly what a radix tree does.** A trie over token ids would cost one node per token; compressing chains into labelled edges (the radix tree) makes a 2,000-token system prompt one node. The tree from *Algorithms* already provides an LRU list of leaves and lock counts.
3. **Blocks are the unit of reuse, so the tree's granule is $B$.** A KV block is filled $B$ tokens at a time; a half-filled block is still being written. Setting the tree's granule to $B$ forces every match length, split point, and stored label to a multiple of $B$, and stores exactly one value (a block id) per granule.
4. **Prefill needs at least one token, so cap the search at $u = \lfloor (n-1)/B \rfloor \cdot B$.** Truncating the query (rather than the result) means the tree itself never returns more; the hit counter then counts what was really reused. Step 3 of the worked example: $n = 4$ gives $u = 2$.
5. **Duplicates come from races, and must be identified by comparing ids.** The tree hands back the caller's values for the part already stored. Those are not all duplicates: a caller that matched first passes the cached ids themselves. Only ids that differ from what the tree keeps are the caller's own redundant blocks.
6. **Locks belong to the path, not the leaf.** A request reads every block from the root down to its match node, so locking must pin the ancestors too. The tree's `lock` already walks up; the cache delegates.
7. **Eviction must be whole leaves, LRU first.** Removing an interior node would orphan its children's prefixes, and a prefix's KV is only useful together with everything before it. Leaves are the only nodes whose removal keeps every remaining path valid.

### Matching

`match_prefix` truncates first, then gathers the blocks of every node on the path:

```rust
pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
    let b = self.block_size;
    // Step 4: never cover the last token. saturating_sub makes n = 0 safe
    // (usable = 0), and integer division rounds down to a whole block.
    let usable = tokens.len().saturating_sub(1) / b * b;
    // The tree marks the path as just used (LRU) and splits an edge if the
    // match ends inside it, so the last node ends exactly at `matched`.
    let (matched, path) = self.tree.match_prefix(&tokens[..usable]);
    let mut blocks = Vec::with_capacity(matched / b);
    // A prefix can span several nodes: concatenate every node's blocks in
    // root-to-leaf order, not just the last one's.
    for &id in &path {
        blocks.extend_from_slice(self.tree.values(id));
    }
    self.stats.lookups += 1;
    self.stats.query_tokens += tokens.len() as u64; // the whole prompt
    self.stats.hit_tokens += matched as u64;        // what was really found
    PrefixMatch {
        matched_tokens: matched,
        blocks,
        // The node to lock; ROOT on a miss, and locking ROOT is a no-op.
        node: path.last().copied().unwrap_or(ROOT),
    }
}
```

### Inserting

`insert` stores only full blocks and works out which handed-back ids are true duplicates:

```rust
pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId]) -> Inserted {
    let b = self.block_size;
    let full = tokens.len() / b;
    // One id per full block, no more: a block for the partial tail would
    // become shared while its contents still grow.
    assert_eq!(blocks.len(), full, /* ... */);
    let before = self.tree.len();
    // Insert only the tokens of full blocks; the tail stays the request's.
    let (node, handed_back) = self.tree.insert(&tokens[..full * b], blocks.to_vec());
    // tree.len() counts stored values (blocks); the difference is what is new.
    self.stats.inserted_blocks += (self.tree.len() - before) as u64;
    // Collect the ids the tree keeps along the path to `node`, root first.
    let mut kept = Vec::with_capacity(handed_back.len());
    let mut cur = Some(node);
    let mut path = Vec::new();
    while let Some(id) = cur {
        if id == ROOT { break; }
        path.push(id);
        cur = self.tree.parent(id);
    }
    for &id in path.iter().rev() {
        kept.extend_from_slice(self.tree.values(id));
    }
    // Step 5: a handed-back id equal to the kept one is the cache's own block
    // (the caller matched first); only different ids are the caller's to free.
    let duplicates = handed_back
        .iter()
        .zip(kept.iter())
        .filter(|(a, k)| a != k)
        .map(|(a, _)| *a)
        .collect();
    Inserted { node, duplicates }
}
```

The handed-back values are the prefix of `blocks` that covered already-stored positions, so zipping them against the first `handed_back.len()` kept ids lines them up position by position.

### Locks, eviction, and counters

These are thin wrappers that add accounting:

```rust
pub fn lock(&mut self, node: NodeId) { self.tree.lock(node); }     // pins node + ancestors
pub fn unlock(&mut self, node: NodeId) { self.tree.unlock(node); } // releases one lock

pub fn evict(&mut self, n_blocks: usize) -> Vec<BlockId> {
    let mut out = Vec::new();
    // The tree removes whole LRU unlocked leaves until at least n_blocks
    // values are gone; every value of a removed leaf comes back, never a
    // truncated subset (the rest would leak).
    self.tree.evict(n_blocks, |blk| out.push(blk));
    self.stats.evicted_blocks += out.len() as u64;
    out
}

pub fn cached_tokens(&self) -> usize {
    self.tree.len() * self.block_size // len() is in blocks; report tokens
}

pub fn evictable_blocks(&self) -> usize {
    // Walk every node; count blocks of nodes nobody locks. A lock pins the
    // ancestors too, so a locked leaf makes its whole path unevictable.
    let mut n = 0;
    let mut stack = self.tree.children(ROOT);
    while let Some(id) = stack.pop() {
        if self.tree.lock_count(id) == 0 {
            n += self.tree.values(id).len();
        }
        stack.extend(self.tree.children(id));
    }
    n
}
```

Underneath, the tree's `lock` increments the count on the node and every ancestor and refreshes their place on the LRU list, so a locked node is never at its head. `evict` repeatedly takes the list head, unlinks it from its parent, hands every value to the callback, and re-checks the parent, which may now be a childless leaf and go next.

The full solution is [`prefix.rs`](../../../../course/solutions/rust/crates/tl-engine/src/prefix.rs); the tree it delegates to is [`radix.rs`](../../../../course/solutions/rust/crates/tl-ds/src/radix.rs).

### Why this design

The costs are linear where they need to be. `match_prefix` and `insert` are $O(n)$ in the prompt length (one edge comparison per token, child lookup by the first granule in a hash map), plus $O(\text{depth})$ for the parent walk in `insert`. `evict` is $O(1)$ per leaf removed thanks to the tree's LRU list. `evictable_blocks` is $O(\text{nodes})$; it is called by the scheduler at admission, not per token.

Compare a hash per block, as in the block pool. A chained hash per block finds a prefix in $n/B$ lookups and needs no tree, but it has no notion of "leaf", so LRU eviction can drop a block in the middle of a hot prefix, and pinning a prefix means pinning every block. The tree gets structure-aware eviction and one-node locks. The engine's block manager keeps both so the tradeoff can be measured.

Token granularity would match longer prefixes, but a match ending inside a block cannot reuse that block's KV without copying it; the extra precision buys nothing the engine can use. Storing KV in the cache would also be a mistake: keeping only ids makes the cache cheap to copy, test, and replay against an independent model, and keeps one owner (the pool) for memory.

One tradeoff is accepted knowingly. Up to $B$ tokens of a fully cached prompt are recomputed (the last-token rule plus rounding). With $B = 16$ that is at most 16 tokens of prefill, against a correctness requirement that cannot be avoided.

## What can go wrong

A correct solution must handle these cases.

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand example (shared system prompt) | matches 4, 2, and 0 tokens; the counters and hit rate | the rules are pinned by numbers you can check by hand |
| The last prompt token is never covered | prompts of 0 to 7 tokens over 6 cached tokens match 0, 0, 0, 2, 2, 4, 4, 6 | prefill always has a token to run |
| Insert takes exactly the full blocks | a partial block among the blocks panics | a growing block is never shared |
| A prefix computed twice is kept once | a racing request's blocks come back as duplicates; the cached ids themselves do not | the block manager frees the duplicates |
| A locked prefix survives eviction | eviction frees only unlocked blocks until the request unlocks; the counts follow | running requests keep their KV |
| Eviction frees the least recently used first | three prompts, one re-used: the other two go first, oldest first | hot prompts stay cached |
| A recorded 300-operation workload | every match, duplicate, eviction order, and cached count agrees with an independent Python model | every rule holds at once |
| Block conservation under 3,000 seeded engine steps over 64 block ids | each id in exactly one place, no locked block evicted, longest prefix found right after an insert, all 64 back at the end | the accounting the block manager builds on |

The pitfalls below are the ways those cases fail.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Matching a fully cached prompt whole | prefill has no token to run; the first output has no logits | truncates the query to $u$ before the walk |
| Returning only the last node's blocks | a prefix spanning several nodes loses its first blocks | concatenates the blocks of every node on the path |
| Counting the searched length as a hit | the hit rate the gateway routes on is inflated | adds `matched`, the tree's result, to `hit_tokens` |
| Accepting a block for the partial tail | a block whose contents still grow becomes shared | asserts `blocks.len() == tokens.len() / B` |
| Handing back the cached ids themselves | the caller frees blocks the cache still holds | filters handed-back ids against the kept ids |
| Inserting the tokens of the partial tail | the tree asserts, or caches a half-written block | inserts `tokens[..full * b]` only |
| A lock that pins nothing | a running request's blocks are evicted and reused | delegates to the tree's `lock`, which pins the node and its ancestors |
| An unlock that releases nothing | the cache fills with pinned blocks and admission stalls | delegates to the tree's `unlock`, which walks the same path |
| Truncating an evicted leaf's blocks to the request | the rest of the leaf's blocks leak | collects every value the tree's `evict` hands to the callback |
| Counting locked blocks as evictable | the scheduler admits a request that cannot get blocks | `evictable_blocks` skips nodes with a nonzero lock count |
| Not counting evicted blocks | `/metrics` shows no pressure | adds `out.len()` to `evicted_blocks` |
| Reporting cached tokens in blocks | every capacity log is off by a factor of $B$ | multiplies `tree.len()` by `block_size` |

## Where this leads

[The engine's block manager](../p10-serving/04-block-manager-and-prefix-cache.md) matches each new request, locks its prefix, allocates the rest from [the block pool](08-paged-kv-block-pool.md), inserts after prefill, and evicts under pressure; that is `--prefix-cache=radix`. In *AI Platform Engineering*, [prefix-affinity routing](../../ai-platform-engineering/12-gateway/05-routing.md) reads `hit_rate` and the cached token counts over `tl.control.v1`.

Production systems take the same idea further:

- SGLang's `RadixCache` ([`python/sglang/srt/mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)) adds page-granular matching (`page_size`), a priority heap for eviction, and host-memory offload tiers (`HiRadixCache`).
- vLLM's automatic prefix caching ([`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)) makes the other choice: hash-per-block lookups instead of a tree, with the same "full blocks only" rule.
- [llm-d](https://github.com/llm-d/llm-d)'s KV-cache-aware routing extends the hit-rate counters to a cluster-wide index of which replica holds which prefix blocks.

### Further reading

- Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104), section 3 (RadixAttention).

## Key terms

- **Prefix cache**: a map from token prefixes of past prompts to the KV blocks already computed for them.
- **Radix tree**: a trie whose chains are compressed into labelled edges.
- **Granule**: the unit of matching and splitting, here one block of $B$ tokens.
- **Usable prefix**: $u = \lfloor (n-1)/B \rfloor \cdot B$, the longest part of a prompt a match may cover.
- **Duplicate**: a caller's block id for a position the tree already stores with a different id, returned for the caller to free.
- **Lock**: a count on a node and its ancestors that keeps them from eviction while a request reads them.

## Summary

- A prefix cache maps the token prefix of a new prompt to KV blocks already computed; block granularity means only whole blocks are matched or stored.
- A match never covers the last prompt token: prefill must run on at least one token to produce the first output's logits.
- Two requests can compute the same prefix at once; the cache keeps the first one's blocks and hands the other's back for the caller to free.
- A running request locks the node its match ended at, which pins the whole prefix; eviction frees the least recently used unlocked leaves.
- Every block id is in exactly one place, the free list, a running request, or the cache, under any interleaving.

## Review questions

1. Why is the query truncated to $\lfloor (n-1)/B \rfloor \cdot B$ before the walk, rather than trimming the match afterwards?
   <details><summary>Answer</summary>Truncating first means the tree only ever sees the usable part, so it never splits an edge or refreshes LRU stamps for tokens the request will not reuse, and the hit counter counts exactly the reused tokens. Trimming afterwards would also need to drop blocks from the result and pick a different node to lock.</details>
2. A caller matched 4 tokens, prefilled the rest, and inserts the whole sequence with the matched ids in the first two slots. Why does nothing come back in `duplicates`?
   <details><summary>Answer</summary>The tree hands back the caller's ids for the already-stored positions, which are the cached ids themselves. The filter keeps only ids that differ from the kept ones, so none survive. Returning them would make the caller free blocks the cache still holds.</details>
3. Why does `lock` pin the ancestors and not only the node?
   <details><summary>Answer</summary>The request reads every block from the root to its match node. If an ancestor could be evicted, its blocks would return to the pool and be overwritten while the request still attends over them. Since eviction only takes leaves, pinning the path also stops a parent from becoming an evictable leaf.</details>
4. Why does `evict` free whole leaves even if that exceeds `n_blocks`?
   <details><summary>Answer</summary>A leaf's blocks are a contiguous continuation of its parent's prefix. Removing only some of them would need a split and leave a shorter leaf; returning only some ids while removing the node would leak the rest. Over-freeing by part of a leaf is harmless; the block manager just has more free blocks.</details>
5. Why is `query_tokens` incremented by the full prompt length and not by $u$?
   <details><summary>Answer</summary>The hit rate the gateway routes on is "what fraction of prompt tokens did we not have to compute". The denominator is the work requested, the whole prompt; using $u$ would inflate the rate.</details>
