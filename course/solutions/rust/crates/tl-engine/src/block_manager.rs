//! The block manager (L10.4): each request's KV blocks in the Rust pool, with
//! an optional prefix cache, behind the scheduler's `BlockSpace` (L10.2).
//!
//! Many requests start with the same tokens (a system prompt, a few-shot
//! preamble, the earlier turns of a chat). Their K and V for that prefix
//! are identical, because a position's K and V depend only on the tokens
//! up to it. A prefix cache keeps the full blocks of finished requests and
//! hands them to new requests whose prompt starts the same way, so prefill
//! skips those tokens: shorter time to first token, fewer blocks in use.
//!
//! Three modes (`--prefix-cache`, runtime.toml `prefix_cache`):
//!
//! - `none`: allocate fresh blocks; free them when the request ends.
//! - `hash`: each full block is named by the chained FNV-1a hash of its
//!   tokens and its parent's hash (formats/kv-block.md). On release a full
//!   block is registered in the pool's index (L10.1); when its refcount
//!   reaches 0 it stays **cached** and `tl_kv_alloc` evicts it LRU-first.
//!   A new request looks its blocks up one by one, front to back.
//! - `radix`: a radix tree over token ids at block granularity (L8.4)
//!   holds one reference to each cached block; matching walks the tree,
//!   and eviction takes least recently used unlocked leaves.
//!
//! Rules shared by both caches: only full blocks are shared; a match covers
//! at most `len - 1` prompt tokens, so the model always runs on at least the
//! last one (it needs those logits); a block is shared by reference count
//! and never written once full, so no copy on write is needed.

use std::collections::HashMap;

use crate::prefix::{NodeId, RadixCache};
use crate::kv::kv_block_hash;
use crate::runner::{lock, PrefixCache, SharedPool};
use crate::sched::{Allocation, BlockSpace, NoCapacity, RequestId};

/// Prefix-cache counters (`gw.05` routes on the hit rate).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockStats {
    pub lookups: u64,
    pub query_tokens: u64,
    pub hit_tokens: u64,
    /// Cached blocks reclaimed to make room (pool evictions or radix leaves).
    pub evictions: u64,
}

/// One request's blocks.
#[derive(Clone, Debug, Default)]
struct Table {
    blocks: Vec<u32>,
    /// Leading blocks that came from the cache (the request holds one
    /// reference to each, the cache another).
    matched: usize,
    /// The radix node the match ended at, locked while the request runs.
    node: Option<NodeId>,
}

/// KV blocks for the scheduler, over the shared Rust pool.
pub struct BlockManager {
    pool: SharedPool,
    mode: PrefixCache,
    block_tokens: usize,
    total: usize,
    tables: HashMap<RequestId, Table>,
    radix: Option<RadixCache>,
    stats: BlockStats,
}

impl BlockManager {
    /// A manager over `pool` (the runner's) in `mode`.
    pub fn new(pool: SharedPool, mode: PrefixCache) -> BlockManager {
        let cfg = lock(&pool).expect("pool lock").cfg();
        let block_tokens = cfg.block_tokens as usize;
        BlockManager {
            pool,
            mode,
            block_tokens,
            total: cfg.n_blocks as usize,
            tables: HashMap::new(),
            radix: (mode == PrefixCache::Radix).then(|| RadixCache::new(block_tokens)),
            stats: BlockStats::default(),
        }
    }

    /// The mode.
    pub fn mode(&self) -> PrefixCache {
        self.mode
    }

    /// Counters.
    pub fn stats(&self) -> BlockStats {
        self.stats
    }

    /// hit_tokens / query_tokens (0 before any lookup).
    pub fn hit_rate(&self) -> f64 {
        if self.stats.query_tokens == 0 {
            0.0
        } else {
            self.stats.hit_tokens as f64 / self.stats.query_tokens as f64
        }
    }

    /// Blocks the cache holds with no running request on them.
    pub fn cached_blocks(&self) -> usize {
        match &self.radix {
            Some(r) => r.evictable_blocks(),
            None => lock(&self.pool).map(|p| p.stats().cached as usize).unwrap_or(0),
        }
    }

    /// `n` fresh blocks from the pool; in radix mode, evicts cached leaves
    /// first when the pool's free list is short. All or nothing.
    fn take(&mut self, n: usize) -> Result<Vec<u32>, NoCapacity> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut pool = lock(&self.pool).map_err(|_| NoCapacity)?;
        if let Some(r) = self.radix.as_mut() {
            let free = pool.stats().free as usize;
            if free < n {
                if free + r.evictable_blocks() < n {
                    return Err(NoCapacity);
                }
                for b in r.evict(n - free) {
                    pool.release(b).expect("the radix cache holds one reference to each cached block");
                    self.stats.evictions += 1;
                }
            }
        }
        let before = pool.stats().evictions;
        let ids = pool.alloc(n).map_err(|_| NoCapacity)?;
        self.stats.evictions += (pool.stats().evictions - before) as u64;
        Ok(ids)
    }

    /// The cached blocks of `tokens`' prefix, each with a reference taken
    /// for the request: (blocks, radix node to lock).
    fn match_prefix(&mut self, tokens: &[u32]) -> (Vec<u32>, Option<NodeId>) {
        let bt = self.block_tokens;
        let usable = tokens.len().saturating_sub(1) / bt;
        let mut blocks = Vec::new();
        let mut node = None;
        match self.mode {
            PrefixCache::None => return (blocks, None),
            PrefixCache::Hash => {
                let mut pool = lock(&self.pool).expect("pool lock");
                let mut parent = 0u64;
                for i in 0..usable {
                    let h = kv_block_hash(parent, &tokens[i * bt..(i + 1) * bt]);
                    match pool.lookup(h) {
                        Some(b) => blocks.push(b),
                        None => break,
                    }
                    parent = h;
                }
            }
            PrefixCache::Radix => {
                let r = self.radix.as_mut().expect("radix mode has a tree");
                let m = r.match_prefix(tokens);
                r.lock(m.node);
                node = Some(m.node);
                let mut pool = lock(&self.pool).expect("pool lock");
                for &b in &m.blocks {
                    pool.retain(b).expect("a cached block is used by the cache");
                }
                blocks = m.blocks;
            }
        }
        self.stats.lookups += 1;
        self.stats.query_tokens += tokens.len() as u64;
        self.stats.hit_tokens += (blocks.len() * bt) as u64;
        (blocks, node)
    }

    /// Drops the request's references to `blocks` (and the radix lock).
    fn drop_refs(&mut self, blocks: &[u32], node: Option<NodeId>) {
        if let Some(n) = node {
            self.radix.as_mut().expect("radix mode").unlock(n);
        }
        let mut pool = lock(&self.pool).expect("pool lock");
        for &b in blocks {
            pool.release(b).expect("the request holds a reference");
        }
    }
}

impl BlockSpace for BlockManager {
    fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    fn total_blocks(&self) -> usize {
        self.total
    }

    fn free_blocks(&self) -> usize {
        let s = lock(&self.pool).map(|p| p.stats()).unwrap_or_default();
        let radix = self.radix.as_ref().map_or(0, |r| r.evictable_blocks());
        (s.free + s.cached) as usize + radix
    }

    fn allocate(&mut self, id: RequestId, tokens: &[u32]) -> Result<Allocation, NoCapacity> {
        if self.tables.contains_key(&id) {
            return Err(NoCapacity);
        }
        let (matched, node) = self.match_prefix(tokens);
        let need = tokens.len().div_ceil(self.block_tokens) - matched.len();
        match self.take(need) {
            Ok(fresh) => {
                let n = matched.len();
                let mut blocks = matched;
                blocks.extend(fresh);
                self.tables.insert(id, Table { blocks, matched: n, node });
                Ok(Allocation { cached_tokens: n * self.block_tokens })
            }
            Err(e) => {
                self.drop_refs(&matched, node);
                Err(e)
            }
        }
    }

    fn append_slot(&mut self, id: RequestId, total: usize) -> Result<(), NoCapacity> {
        let have = self.tables.get(&id).map(|t| t.blocks.len()).ok_or(NoCapacity)?;
        let need = total.div_ceil(self.block_tokens).saturating_sub(have);
        let fresh = self.take(need)?;
        self.tables.get_mut(&id).expect("checked above").blocks.extend(fresh);
        Ok(())
    }

    fn release(&mut self, id: RequestId, computed: &[u32]) {
        let Some(t) = self.tables.remove(&id) else { return };
        let bt = self.block_tokens;
        let full = (computed.len() / bt).min(t.blocks.len());
        match self.mode {
            PrefixCache::None => self.drop_refs(&t.blocks, None),
            PrefixCache::Hash => {
                {
                    let mut pool = lock(&self.pool).expect("pool lock");
                    let mut parent = 0u64;
                    for i in 0..full {
                        let h = kv_block_hash(parent, &computed[i * bt..(i + 1) * bt]);
                        let b = t.blocks[i];
                        if i >= t.matched {
                            // the block is full of computed K and V; only a
                            // full block may be registered (L10.1)
                            if pool.fill(b) < bt as u32 {
                                pool.set_fill(b, bt as u32).expect("an unregistered block of ours");
                            }
                            // Ok(false): another block already holds this
                            // prefix; ours just goes back to the free list
                            let _ = pool.register(b, h);
                        }
                        parent = h;
                    }
                }
                self.drop_refs(&t.blocks, None);
            }
            PrefixCache::Radix => {
                let r = self.radix.as_mut().expect("radix mode");
                // unlock first: an insert may split the node the match ended at
                if let Some(n) = t.node {
                    r.unlock(n);
                }
                let ins = r.insert(&computed[..full * bt], &t.blocks[..full]);
                // The cache keeps the request's reference to each block it
                // newly stored; every other reference goes back.
                let give_back: Vec<u32> = t
                    .blocks
                    .iter()
                    .enumerate()
                    .filter(|&(i, b)| i < t.matched || i >= full || ins.duplicates.contains(b))
                    .map(|(_, &b)| b)
                    .collect();
                self.drop_refs(&give_back, None);
            }
        }
    }

    fn block_table(&self, id: RequestId) -> &[u32] {
        self.tables.get(&id).map_or(&[], |t| t.blocks.as_slice())
    }
}
