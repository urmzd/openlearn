<!-- ol:module ds.07 -->
# Radix tree over token ids with index-linked LRU leaf list

## Overview

| | |
|---|---|
| **Module** | `ds.07` · build · Rust · Pass 6 · 6 to 8 h |
| **You build** | `rust/crates/tl-ds/src/radix.rs`: `RadixTree<V>` (`new`, `granule`, `len`, `is_empty`, `node_count`, `key`, `values`, `parent`, `children`, `child_count`, `lock_count`, `lru_order`, `match_prefix`, `insert`, `lock`, `unlock`, `evict`), `NodeId`, `ROOT` · one line in the crate root: `pub mod radix;` |
| **Contract** | the interface in section 4 (no Rust trait file yet; the tests pin it) |
| **Tests** | `course/tests/rust/ds_07.rs`, 11 tests (what they check: section 4) · your own tests in `rust/crates/tl-ds/tests/ds07_radix.rs`, rung R3, graded by mutation (threshold 0.70) |
| **Needs** | `ds.05` [the Robin Hood map](05-robin-hood-hash-map.md) (children are keyed in a `RobinHoodMap`) · reading: `M06.2` [tries and longest-prefix match](../../math/06-discrete-math-2/02-trees-and-tries-longest-prefix-match.md) |
| **Used by** | `L8.4` the radix prefix cache is this tree with one KV block per `block_size` tokens |
| **Milestone** | `MS-L8` (step 5: the L8.4 course tests pass) |
| **Optional depth** | Morrison, *PATRICIA: Practical Algorithm To Retrieve Information Coded in Alphanumeric* (JACM 1968); Zheng et al., [*SGLang*](https://arxiv.org/abs/2312.07104), section 3 (RadixAttention) |

## Key Takeaways

- A radix tree is a trie whose chains of single children are merged into one edge with a label, so a 2,000-token prompt is one node until another prompt diverges from it (`hand_example_split_and_match`).
- Matching ends inside an edge whenever two prompts diverge there; the tree splits the edge at that point, so every match ends exactly at a node (`hand_example_split_and_match`).
- With a granule $g$ the tree compares and splits only at multiples of $g$ tokens: at $g = 16$ it never shares part of a KV block (`granule_matches_whole_runs_only`).
- Nodes live in a `Vec` and point at each other by index; the LRU list of evictable leaves is a doubly linked list through those indices, with no `unsafe` and no `Rc` (`freed_slots_are_reused`, `lru_order_follows_last_use`).
- A lock pins a node and every ancestor; eviction takes the least recently used unlocked leaf, and a parent left childless becomes a leaf in its turn (`locks_pin_the_path`, `a_childless_parent_becomes_evictable`).

## How to work this chapter

```bash
ol start ds.07              # stubs rust/crates/tl-ds/src/radix.rs; add `pub mod radix;` to your tl-ds lib.rs
ol tests ds.07              # read the test catalog first
ol check ds.07              # exit code is the verdict; then grades your tests by mutation
ol diff  ds.07              # after passing: your code against the reference
```

`ol start` never rewrites your crate root (it belongs to `ds.05`), so add the one `pub mod radix;` line yourself. Build in this order: `new`, `insert` without splits, `match_prefix` with splits, `lock`/`unlock`, then the LRU list and `evict`. Get the model test passing at granule 1 before granule 2.

---

## 1. Why now

Chat traffic repeats itself: every request of a deployment starts with the same system prompt, every turn of a conversation starts with all the previous turns, and an agent loop resends its tool descriptions on every call. A serving engine that remembers the KV blocks of prefixes it has computed can skip their prefill entirely. To find them it needs the longest stored prefix of a new token sequence, in time proportional to the prompt, and it must forget the prefixes nobody has used for a while, without ever forgetting one a running request is reading. The tokenizer's trie (`M06.2`) finds longest prefixes one character per node; that is one node per token here, 2,000 nodes for one system prompt. This chapter compresses the chains, keys children with the hash map of `ds.05`, and adds the recency order the cache needs.

## 2. Principles

### 2.1 Labels, granules, and splits

| Symbol | Meaning | Type |
|---|---|---|
| $g$ | the granule: labels, matches, and splits come in runs of $g$ tokens | `usize` |
| label | a node's edge: the tokens from its parent to it, a multiple of $g$ long | `Vec<u32>` |
| value | one per $g$ tokens of a label (for L8.4, one KV block id) | `V` |
| stamp | the clock value of the last match or insert that walked the node | `u64` |
| lock | how many locks pin the node: its own and its descendants' | `u32` |

Every node except the root holds a label and one value per granule of it. The path from the root to a node spells the prefix that node ends. Children are keyed by the first $g$ tokens of their labels in a `RobinHoodMap<Vec<u32>, NodeId>`, and no two children start alike, so at each node the next step of a walk is one hash lookup (by slice: `Vec<u32>` borrows as `[u32]`, so no key is allocated).

**Match.** Walk from the root while at least $g$ tokens remain: look up the child that starts with the next $g$ tokens, then compare its label run by run. If the whole label matches, step into the child and continue. If only $m$ runs match, the match ends inside the edge: **split** it, inserting a new node with the first $m$ runs of the label and values between the parent and the child, and stop there. The walk returns the tokens matched (a multiple of $g$) and the nodes on the path.

**Split.** The new middle node takes the upper part of the label and values; the old child keeps its id and the lower part. Every lock on the child passes through the new node, so the middle node starts with the child's lock count.

**Insert.** Walk as a match does; the part of the key already stored keeps its stored values, and the caller's values for that part are **handed back** (the caller decides what to do with duplicates); the rest of the key becomes a new leaf under the node the walk ended at.

### 2.2 Indices instead of pointers

Nodes live in `Vec<Node<V>>` and refer to each other by index (`NodeId = usize`): the parent link, the children map's values, and the LRU links `prev` and `next`. A removed node's slot goes on a free list and is reused by the next insert. In safe Rust, a doubly linked list of `Box` or `&mut` links does not exist (each node would have two owners); with indices, the borrow checker sees one owner, the `Vec`.

### 2.3 Recency, locks, and eviction

A clock advances by one on every `match_prefix` and `insert`, and every node on the walked path is stamped with it. The **LRU list** holds exactly the leaves that are not locked, ordered by stamp, oldest first. A node touched just now joins at the newest end in O(1); a parent that becomes a leaf when its last child is evicted joins at its own stamp, walking in from the newest end (it is usually old, so this walk is short in practice and O(leaves) at worst).

`lock(node)` adds one to the node and every ancestor and takes them off the list; `unlock` undoes it and an unlocked childless node rejoins the list. Unlocking a node that is not locked panics: a count below zero would unpin a prefix someone else holds. `evict(n, f)` pops leaves from the oldest end, calls `f` on every value of each (in label order), removes the leaf from its parent, and repeats until at least `n` values are gone or the list is empty. Leaves go whole, so `evict` may free more than `n`.

## 3. Worked example by hand

Granule 1, values are letters.

| Step | Call | Tree afterwards (label: values) | Returns |
|---|---|---|---|
| 1 | `insert([1,2,3], [a,b,c])` | root → `1 2 3: a b c` | node of `1 2 3`, nothing handed back |
| 2 | `insert([1,2,4,5], [x,y,d,e])` | root → `1 2: a b` → {`3: c`, `4 5: d e`} | the new leaf; `x y` handed back (the stored `a b` stay) |
| 3 | `match_prefix([1,2,3,9])` | unchanged | 3 tokens, path [`1 2`, `3`] (values `a b c`) |
| 4 | `match_prefix([1,2,7])` | unchanged | 2 tokens, path [`1 2`] |
| 5 | `match_prefix([1,9])` | root → `1: a` → `2: b` → {`3: c`, `4 5: d e`} | 1 token: the match ended inside `1 2`, so it split |
| 6 | `match_prefix([7,1])` | unchanged | 0 tokens, empty path |

In step 2, the walk matches `1 2` of the edge `1 2 3` and stops inside it: the edge splits into a new node `1 2` (values `a b`) whose children are the old node, relabelled `3` (value `c`, same id), and the new leaf `4 5`. The tree holds 5 values in all steps after 2; splits move values, never copy or drop them. This is `hand_example_split_and_match`.

**Granule 2.** Store `1 2 3 4` (values 10, 11). A match of `1 2 3 9` compares the runs `1 2` (equal) and `3 9` against `3 4` (not equal): 2 tokens, and the split falls between the runs. A match of `1` matches nothing: less than one run. This is `granule_matches_whole_runs_only`.

## 4. The interface

```rust
// rust/crates/tl-ds/src/radix.rs
pub type NodeId = usize;
pub const ROOT: NodeId = 0;
pub struct RadixTree<V> { /* nodes: Vec<Node<V>>, free list, LRU head and tail, clock */ }
impl<V> RadixTree<V> {
    pub fn new(granule: usize) -> Self;                    // panics on 0
    pub fn granule(&self) -> usize;
    pub fn len(&self) -> usize;                            // values stored
    pub fn is_empty(&self) -> bool;
    pub fn node_count(&self) -> usize;                     // root excluded
    pub fn key(&self, id: NodeId) -> &[u32];               // the edge label
    pub fn values(&self, id: NodeId) -> &[V];
    pub fn parent(&self, id: NodeId) -> Option<NodeId>;
    pub fn children(&self, id: NodeId) -> Vec<NodeId>;     // increasing id order
    pub fn child_count(&self, id: NodeId) -> usize;
    pub fn lock_count(&self, id: NodeId) -> u32;
    pub fn lru_order(&self) -> Vec<NodeId>;                // unlocked leaves, oldest first
    pub fn match_prefix(&mut self, key: &[u32]) -> (usize, Vec<NodeId>);
    pub fn insert(&mut self, key: &[u32], values: Vec<V>) -> (NodeId, Vec<V>);  // values.len() * g == key.len()
    pub fn lock(&mut self, id: NodeId);
    pub fn unlock(&mut self, id: NodeId);                  // panics when not locked
    pub fn evict<F: FnMut(V)>(&mut self, n: usize, f: F) -> usize;
}
```

`match_prefix` takes `&mut self` because a match splits edges and stamps the path. `insert` of an empty key returns `(ROOT, vec![])`. Locking or unlocking `ROOT` does nothing. Accessors panic on an id that is not a live node.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_split_and_match` | unit | section 3 step by step: the split, the handed-back values, the paths, the split on a match | you and the tests agree on every rule |
| `granule_matches_whole_runs_only` | boundary | granule 2: runs compared whole; a key shorter than a run matches nothing | L8.4 never shares part of a block |
| `reinserting_hands_every_value_back` | unit | a stored key returns all of the caller's values and the same node | a prefix computed twice is kept once |
| `lru_order_follows_last_use` | unit | a match moves a leaf to the newest end; `evict(1)` takes the oldest leaf whole | prefixes in use stay cached |
| `a_childless_parent_becomes_evictable` | unit | evicting the last child puts the parent on the list at its own stamp | eviction can empty an unused subtree |
| `locks_pin_the_path` | unit | a lock pins the node and its ancestors; locks nest; unlocking frees them again | a running request's prefix is never evicted |
| `a_split_inherits_the_locks_below_it` | unit | a split under a locked node gives the new node the same count | a split never unpins a running request |
| `unlock_without_a_lock_panics` | boundary | an unmatched unlock panics | bookkeeping bugs surface at once |
| `insert_needs_one_value_per_granule` | boundary | a wrong number of values panics | values never shift against their tokens |
| `freed_slots_are_reused` | regression | 1,000 insert-and-evict rounds keep node ids below 5 | the arena does not grow over an engine's life |
| `model_based_against_a_naive_map` | property | 2,500 seeded inserts, matches, locks, unlocks, evictions at granules 1 and 2 against a map of every stored prefix: longest match, stored and handed-back values, no locked prefix evicted, every eviction the oldest unlocked leaf, values conserved | every invariant under any interleaving |

**Your tests (rung R3).** First write `rust/crates/tl-ds/tests/ds07_radix.rs` (an integration test: `use tl_ds::radix::...` only) and run `ol tdd red ds.07` against your stub; then implement until `ol tdd green ds.07`. Name them `split_by_hand`, `match_splits_inside_an_edge`, `insert_hands_back_the_stored_part`, `oldest_leaf_goes_first`, `locks_pin_ancestors`, `parent_becomes_evictable`. The planted bug "a locked leaf stays on the eviction list" (`s11`) is required; overall 70%.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Giving the split halves each other's values | stored values move to the wrong tokens | `hand_example_split_and_match` (mutant `s01`) |
| Not splitting when a match ends inside an edge | the returned node covers more than the match | `hand_example_split_and_match`, `model_based_against_a_naive_map` (mutant `s02`) |
| Forgetting to link the old child under the new middle node | the lower half of the edge disappears | `hand_example_split_and_match` (mutant `s03`) |
| Comparing a run by its first token only | `1 2 3 9` matches `1 2 3 4` whole at granule 2 | `granule_matches_whole_runs_only` (mutant `s04`) |
| Storing values for a prefix already stored | the tree holds two copies; the caller's duplicates leak | `reinserting_hands_every_value_back` (mutant `s05`) |
| Always linking at the newest end | a parent that becomes a leaf jumps ahead of older leaves | `a_childless_parent_becomes_evictable` (mutant `s06`) |
| Not stamping a match | a hot prefix is evicted as if unused | `lru_order_follows_last_use` (mutant `s07`) |
| Not putting a childless parent on the list | unused subtrees never go | `a_childless_parent_becomes_evictable` (mutant `s08`) |
| Leaving an evicted leaf in its parent's map | the parent never becomes childless; later walks reach a dead node | `a_childless_parent_becomes_evictable` (mutant `s09`) |
| Locking only the node itself | an ancestor's tokens can be evicted under a running request | `locks_pin_the_path` (mutant `s10`) |
| Leaving locked leaves on the list | `evict` frees a block a request is reading | `locks_pin_the_path` (mutant `s11`) |
| Not relisting a node on unlock | a finished request's prefix can never be evicted | `locks_pin_the_path` (mutant `s12`) |
| A split forgetting the locks below it | the middle node is evictable under a running request | `a_split_inherits_the_locks_below_it` (mutant `s13`) |
| Clamping an unmatched unlock at zero | someone else's lock is silently released | `unlock_without_a_lock_panics` (mutant `s14`) |
| Accepting a wrong number of values | every value after the gap belongs to the wrong tokens | `insert_needs_one_value_per_granule` (mutant `s15`) |
| Not counting inserted values | `len` and every capacity decision built on it are wrong | `model_based_against_a_naive_map` (mutant `m01`) |
| Not reusing freed slots | the node array grows for the life of the engine | `freed_slots_are_reused` (mutant `m02`) |
| Evicting while `freed <= n` | one leaf too many goes | `a_childless_parent_becomes_evictable` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ds.05` | the children of each node in a `RobinHoodMap<Vec<u32>, NodeId>`, looked up by slice |
| Back | `M06.2` | the trie and longest-prefix match this tree compresses |
| Forward | `L8.4` | `RadixTree<BlockId>` at granule `block_size` is the prefix cache; locks are running requests |

If you skip this module, `ol check L8.4` stops with `needs ds.07: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `RadixTree` | SGLang's `RadixCache` | the same tree over GPU KV pages, a heap for eviction, paged matching (`page_size`) | [`python/sglang/srt/mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) |
| index-linked list | the `slab` and `generational-arena` crates | generation counters so a stale id is detected instead of reaching a reused slot | [`generational-arena`](https://github.com/fitzgen/generational-arena) |
| label compression | Linux's radix tree (now the XArray) | fixed fan-out over integer keys, RCU-safe lookups | `lib/xarray.c` |
