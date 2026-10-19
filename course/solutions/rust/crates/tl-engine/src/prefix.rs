//! The radix prefix cache (L8.4): which KV blocks of a new prompt are
//! already computed, kept in a radix tree over token ids at block
//! granularity (tl-ds `radix`, ds.07).
//!
//! The cache stores block ids, never KV data: the blocks live in the Rust pool
//! (L10.1) and the block manager (L10.4) moves references between the pool,
//! running requests, and this cache. The rules the cache adds to the tree:
//!
//! - Block granular. Only whole blocks of `block_size` tokens are matched or
//!   stored; a prompt's partial last block is the request's own.
//! - The last token is always computed. A match covers at most
//!   `len - 1` tokens of the prompt (rounded down to a block), because the
//!   model must run on at least one prompt token to produce the logits of
//!   the first new token.
//! - One owner per stored block. `insert` stores the blocks of the new part
//!   of a sequence and hands back the ones whose prefix was already stored
//!   by another request (the caller releases those).
//! - Locked prefixes stay. A running request locks the node its match ended
//!   at; `evict` takes least recently used unlocked leaves only.

use tl_ds::radix::{RadixTree, ROOT};

pub use tl_ds::radix::NodeId;

/// A KV block id of the Rust pool (L10.1).
pub type BlockId = u32;

/// The longest cached prefix of a prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixMatch {
    /// Tokens covered, a multiple of the block size, at most `len - 1`.
    pub matched_tokens: usize,
    /// The cached blocks of those tokens, in order.
    pub blocks: Vec<BlockId>,
    /// The node the match ended at (`ROOT` when nothing matched): what the
    /// request locks while it runs.
    pub node: NodeId,
}

/// What `insert` did with the blocks it was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inserted {
    /// The node that ends at the inserted tokens.
    pub node: NodeId,
    /// Blocks the caller passed for an already-cached position whose cached
    /// block is a different one: not stored, so the caller releases them.
    pub duplicates: Vec<BlockId>,
}

/// Counters for metrics (`gw.05` routes on the hit rate over gRPC).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixStats {
    pub lookups: u64,
    pub query_tokens: u64,
    pub hit_tokens: u64,
    pub inserted_blocks: u64,
    pub evicted_blocks: u64,
}

pub struct RadixCache {
    tree: RadixTree<BlockId>,
    block_size: usize,
    stats: PrefixStats,
}

impl RadixCache {
    /// An empty cache for blocks of `block_size` tokens. Panics on 0.
    pub fn new(block_size: usize) -> Self {
        assert!(
            block_size >= 1,
            "RadixCache::new: block_size must be at least 1"
        );
        RadixCache {
            tree: RadixTree::new(block_size),
            block_size,
            stats: PrefixStats::default(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// The longest cached prefix of `tokens` in whole blocks, leaving at
    /// least one token uncovered. Marks the matched blocks as just used and
    /// counts the lookup in `stats`.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        let b = self.block_size;
        let usable = tokens.len().saturating_sub(1) / b * b;
        let (matched, path) = self.tree.match_prefix(&tokens[..usable]);
        let mut blocks = Vec::with_capacity(matched / b);
        for &id in &path {
            blocks.extend_from_slice(self.tree.values(id));
        }
        self.stats.lookups += 1;
        self.stats.query_tokens += tokens.len() as u64;
        self.stats.hit_tokens += matched as u64;
        PrefixMatch {
            matched_tokens: matched,
            blocks,
            node: path.last().copied().unwrap_or(ROOT),
        }
    }

    /// Stores the full blocks of a sequence: `blocks[i]` holds tokens
    /// `[i * B, (i + 1) * B)`, and `blocks.len()` must equal
    /// `tokens.len() / B` (tokens past the last full block are ignored;
    /// a panic otherwise). Positions already cached keep their block; a
    /// different block passed for one of them comes back in `duplicates`.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId]) -> Inserted {
        let b = self.block_size;
        let full = tokens.len() / b;
        assert_eq!(
            blocks.len(),
            full,
            "RadixCache::insert: {} blocks for {} full blocks of tokens",
            blocks.len(),
            full
        );
        let before = self.tree.len();
        let (node, handed_back) = self.tree.insert(&tokens[..full * b], blocks.to_vec());
        self.stats.inserted_blocks += (self.tree.len() - before) as u64;
        // The handed-back values are the caller's own ids for the cached
        // part; compare them with the ids the tree keeps for those positions.
        let mut kept = Vec::with_capacity(handed_back.len());
        let mut cur = Some(node);
        let mut path = Vec::new();
        while let Some(id) = cur {
            if id == ROOT {
                break;
            }
            path.push(id);
            cur = self.tree.parent(id);
        }
        for &id in path.iter().rev() {
            kept.extend_from_slice(self.tree.values(id));
        }
        let duplicates = handed_back
            .iter()
            .zip(kept.iter())
            .filter(|(a, k)| a != k)
            .map(|(a, _)| *a)
            .collect();
        Inserted { node, duplicates }
    }

    /// Pins the prefix that ends at `node` while a request reads it.
    pub fn lock(&mut self, node: NodeId) {
        self.tree.lock(node);
    }

    /// Releases one `lock(node)`.
    pub fn unlock(&mut self, node: NodeId) {
        self.tree.unlock(node);
    }

    /// Frees at least `n_blocks` cached blocks (fewer when the rest are
    /// locked), least recently used leaves first; returns them for the
    /// caller to give back to the pool.
    pub fn evict(&mut self, n_blocks: usize) -> Vec<BlockId> {
        let mut out = Vec::new();
        self.tree.evict(n_blocks, |blk| out.push(blk));
        self.stats.evicted_blocks += out.len() as u64;
        out
    }

    /// Tokens covered by the cached blocks.
    pub fn cached_tokens(&self) -> usize {
        self.tree.len() * self.block_size
    }

    pub fn cached_blocks(&self) -> usize {
        self.tree.len()
    }

    /// Blocks `evict` could free right now: those of nodes nobody locks.
    pub fn evictable_blocks(&self) -> usize {
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

    pub fn stats(&self) -> PrefixStats {
        self.stats
    }

    /// hit_tokens / query_tokens over every lookup so far (0 before any).
    pub fn hit_rate(&self) -> f64 {
        if self.stats.query_tokens == 0 {
            0.0
        } else {
            self.stats.hit_tokens as f64 / self.stats.query_tokens as f64
        }
    }

    /// The tree underneath, for tests and diagrams.
    pub fn tree(&self) -> &RadixTree<BlockId> {
        &self.tree
    }
}
