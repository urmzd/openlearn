# A radix tree over token ids

Chat traffic repeats itself. Every request of a deployment starts with the same system prompt, every turn of a conversation starts with all the previous turns, and an agent loop resends its tool descriptions on every call. A serving engine that remembers the KV blocks of prefixes it has computed can skip their prefill entirely. To find them it needs the longest stored prefix of a new token sequence, in time proportional to the prompt, and it must forget the prefixes nobody has used for a while, without ever forgetting one a running request is reading.

The tokenizer's trie in [Trees and tries, longest-prefix match](../../mathematics-for-computing/06-discrete-math-2/02-trees-and-tries-longest-prefix-match.md) in *Mathematics for Computing* finds longest prefixes one character per node. Here that would be one node per token: 2,000 nodes for one system prompt. This section compresses those chains into a **radix tree**, keys children with [the Robin Hood map](05-robin-hood-hash-map.md), and adds the recency order and locks the cache needs. The result is the data structure behind the [Radix prefix cache](../../build-an-llm/p08-inference/04-radix-prefix-cache.md) in *Build an LLM*, and behind SGLang's RadixAttention.

There is no separate contract file: the public API of [`radix.rs`](../../../../course/solutions/rust/crates/tl-ds/src/radix.rs) is the interface, declared in the crate root [`lib.rs`](../../../../course/solutions/rust/crates/tl-ds/src/lib.rs) as `pub mod radix;`. When you reach the code, follow the worked example at granule 1 through it first; granule 2 changes only the run comparisons.

> **Learning objectives**
> - Explain how a radix tree compresses a trie, and why a match that ends inside an edge must split it.
> - Work in granules of $g$ tokens so that sharing happens only in whole KV blocks.
> - Link nodes by index in a `Vec` arena so that parent links and a doubly linked LRU list need no `unsafe` and no `Rc`.
> - Maintain an LRU list of exactly the unlocked leaves, ordered by stamp, and evict from its oldest end.
> - Pin a running request's prefix with locks on the node and every ancestor, and keep the counts right across splits.

## Labels, granules, and splits

The vocabulary for this section:

| Symbol | Meaning | Type |
|---|---|---|
| $g$ | the granule: labels, matches, and splits come in runs of $g$ tokens | `usize` |
| label | a node's edge: the tokens from its parent to it, a multiple of $g$ long | `Vec<u32>` |
| value | one per $g$ tokens of a label (for the prefix cache, one KV block id) | `V` |
| stamp | the clock value of the last match or insert that walked the node | `u64` |
| lock | how many locks pin the node: its own and its descendants' | `u32` |

Every node except the root holds a label and one value per granule of it. The path from the root to a node spells the prefix that node ends. Children are keyed by the first $g$ tokens of their labels in a `RobinHoodMap<Vec<u32>, NodeId>`, and no two children start alike, so at each node the next step of a walk is one hash lookup. The lookup is by slice: `Vec<u32>` borrows as `[u32]`, so no key is allocated.

### Match

Walk from the root while at least $g$ tokens remain: look up the child that starts with the next $g$ tokens, then compare its label run by run. If the whole label matches, step into the child and continue. If only $m$ runs match, the match ends inside the edge: **split** it, inserting a new node with the first $m$ runs of the label and values between the parent and the child, and stop there. The walk returns the tokens matched (a multiple of $g$) and the nodes on the path.

### Split

The new middle node takes the upper part of the label and values; the old child keeps its id and the lower part. Every lock on the child passes through the new node, so the middle node starts with the child's lock count.

### Insert

Insert walks as a match does. The part of the key already stored keeps its stored values, and the caller's values for that part are **handed back** (the caller decides what to do with duplicates); the rest of the key becomes a new leaf under the node the walk ended at.

## Indices instead of pointers

The tree needs parent links, child links, and a doubly linked LRU list through its leaves. In safe Rust, a doubly linked list of `Box` or `&mut` links does not exist, because each node would have two owners. So nodes live in `Vec<Node<V>>` and refer to each other by index (`NodeId = usize`): the parent link, the children map's values, and the LRU links `prev` and `next`. With indices, the borrow checker sees one owner, the `Vec`. A removed node's slot goes on a free list and is reused by the next insert, so the arena stays bounded.

## Recency, locks, and eviction

A clock advances by one on every `match_prefix` and `insert`, and every node on the walked path is stamped with it. The **LRU list** holds exactly the leaves that are not locked, ordered by stamp, oldest first. A node touched just now joins at the newest end in $O(1)$. A parent that becomes a leaf when its last child is evicted joins at its own stamp, walking in from the newest end; it is usually old, so this walk is short in practice, and $O(\text{leaves})$ at worst.

`lock(node)` adds one to the node and every ancestor and takes them off the list; `unlock` undoes it, and an unlocked childless node rejoins the list. Unlocking a node that is not locked panics: a count below zero would unpin a prefix someone else holds.

`evict(n, f)` pops leaves from the oldest end, calls `f` on every value of each (in label order), removes the leaf from its parent, and repeats until at least `n` values are gone or the list is empty. Leaves go whole, so `evict` may free more than `n`.

## Working it by hand

Use granule 1, with letters as values. The table traces six calls.

| Step | Call | Tree afterwards (label: values) | Returns |
|---|---|---|---|
| 1 | `insert([1,2,3], [a,b,c])` | root → `1 2 3: a b c` | node of `1 2 3`, nothing handed back |
| 2 | `insert([1,2,4,5], [x,y,d,e])` | root → `1 2: a b` → {`3: c`, `4 5: d e`} | the new leaf; `x y` handed back (the stored `a b` stay) |
| 3 | `match_prefix([1,2,3,9])` | unchanged | 3 tokens, path [`1 2`, `3`] (values `a b c`) |
| 4 | `match_prefix([1,2,7])` | unchanged | 2 tokens, path [`1 2`] |
| 5 | `match_prefix([1,9])` | root → `1: a` → `2: b` → {`3: c`, `4 5: d e`} | 1 token: the match ended inside `1 2`, so it split |
| 6 | `match_prefix([7,1])` | unchanged | 0 tokens, empty path |

In step 2, the walk matches `1 2` of the edge `1 2 3` and stops inside it: the edge splits into a new node `1 2` (values `a b`) whose children are the old node, relabelled `3` (value `c`, same id), and the new leaf `4 5`. The tree holds 5 values in all steps after 2; splits move values, never copy or drop them.

Now granule 2. Store `1 2 3 4` (values 10, 11). A match of `1 2 3 9` compares the runs `1 2` (equal) and `3 9` against `3 4` (not equal): 2 tokens, and the split falls between the runs. A match of `1` matches nothing: it is less than one run.

## The interface

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

## Building it step by step

The solution was designed in this order: the node arena, `common` and `split`, `walk` (shared by `match_prefix` and `insert`), then `lock` and `unlock`, the LRU list, and `evict`. Seven observations lead there.

1. **The question is "longest stored prefix of this key".** A trie answers it in time proportional to the key, but a 2,000-token system prompt would be 2,000 nodes. Merging single-child chains into labelled edges (a radix tree) keeps one node per point where stored prefixes diverge.
2. **A KV block covers $g$ tokens, so sharing must happen in whole blocks.** Labels, comparisons, and splits therefore work in runs of $g$, and a node stores one value per run. With runs, "no two children start alike" means children are keyed by their first run, and the next step of a walk is one hash lookup (`RobinHoodMap<Vec<u32>, NodeId>`, looked up by slice).
3. **A match that ends inside an edge must still end at a node,** because the caller locks and later references "the node where my prefix ends". So the walk splits the edge there. Splitting moves the first $m$ runs and their values into a new middle node; the old child keeps its id (other holders of that id remain correct) and the remainder.
4. **Insert is a walk plus one new leaf.** The walked part is already stored; its stored values win, and the caller's duplicates are handed back so the caller can free them (in the prefix cache, KV blocks computed twice). In the worked example, step 2 walks `1 2`, splits `1 2 3`, hands back `x y`, and hangs `4 5: d e` under the new node.
5. **The cache needs eviction by recency,** and only leaves can go (removing an inner node would orphan its subtree). So recency is a stamp per node, set on every walk, and the eviction candidates are a list of unlocked leaves ordered by stamp.
6. **A running request reads its whole path,** so a lock must pin the node and every ancestor: a per-node lock count incremented along the parent chain. A split must copy the child's count to the middle node, because every lock that pinned the child passes through it.
7. **Rust forbids two owners of a node,** but a doubly linked list and parent links need exactly that. Storing nodes in one `Vec` and linking by index gives one owner; a free list reuses evicted slots so the arena stays bounded.

### The arena

`in_lru` makes list membership an $O(1)$ question; `live` lets accessors reject ids of evicted nodes.

```rust
struct Node<V> {
    key: Vec<u32>,                           // edge label, a multiple of g tokens
    values: Vec<V>,                          // one per g tokens of key
    parent: Option<NodeId>,
    children: RobinHoodMap<Vec<u32>, NodeId>, // keyed by each child's first run
    lock: u32,                               // locks on this node and its descendants
    stamp: u64,                              // clock value of the last walk through it
    prev: Option<NodeId>,                    // LRU links, by index
    next: Option<NodeId>,
    in_lru: bool,
    live: bool,                              // false once evicted (slot on the free list)
}

pub struct RadixTree<V> {
    granule: usize,
    nodes: Vec<Node<V>>,                     // nodes[0] is ROOT
    free: Vec<NodeId>,
    head: Option<NodeId>,                    // oldest unlocked leaf
    tail: Option<NodeId>,                    // newest unlocked leaf
    n_values: usize,
    clock: u64,
}
```

### Comparing and splitting

`common` compares whole runs, so at $g = 2$, `3 9` against `3 4` fails as a run even though the first tokens agree.

```rust
/// Granules shared by `label` and the start of `rest`.
fn common(&self, label: &[u32], rest: &[u32]) -> usize {
    let g = self.granule;
    let mut m = 0;
    while (m + 1) * g <= label.len()
        && (m + 1) * g <= rest.len()
        && label[m * g..(m + 1) * g] == rest[m * g..(m + 1) * g]  // a whole run, never one token
    {
        m += 1;
    }
    m
}

/// Split `child` after `m` granules; returns the new middle node.
fn split(&mut self, child: NodeId, m: usize, stamp: u64) -> NodeId {
    let g = self.granule;
    let parent = self.nodes[child].parent.expect("the root is never split");
    // child keeps its id and the LOWER part; the new node gets the UPPER part
    let lower_key = self.nodes[child].key.split_off(m * g);
    let upper_key = std::mem::replace(&mut self.nodes[child].key, lower_key);
    let lower_vals = self.nodes[child].values.split_off(m);
    let upper_vals = std::mem::replace(&mut self.nodes[child].values, lower_vals);
    let mut mid = Node::new(upper_key, upper_vals, Some(parent), stamp);
    mid.lock = self.nodes[child].lock;       // every lock below passes through mid
    let first = mid.key[..g].to_vec();
    let mid_id = self.alloc(mid);
    let lower_first = self.nodes[child].key[..g].to_vec();
    self.nodes[mid_id].children.insert(lower_first, child); // relink the old child under mid
    self.nodes[child].parent = Some(mid_id);
    self.nodes[parent].children.insert(first, mid_id);      // same first run: replaces child
    mid_id
}
```

### The walk

The walk is shared by `match_prefix` and `insert`, so both split and stamp in exactly the same way. A split always ends the walk: the key diverged inside that edge.

```rust
fn walk(&mut self, key: &[u32], stamp: u64) -> (usize, Vec<NodeId>) {
    let g = self.granule;
    let mut cur = ROOT;
    let mut done = 0;                         // tokens matched
    let mut path = Vec::new();
    while key.len() - done >= g {             // fewer than g tokens left: nothing to match
        let rest = &key[done..];
        let child = match self.nodes[cur].children.get(&rest[..g]) {  // by slice: no allocation
            Some(&c) => c,
            None => break,
        };
        let m = self.common(&self.nodes[child].key, rest);  // >= 1: the first run matched
        let node = if m * g < self.nodes[child].key.len() {
            self.split(child, m, stamp)       // match ends inside the edge
        } else {
            child
        };
        self.nodes[node].stamp = stamp;
        path.push(node);
        done += m * g;
        cur = node;
        if node != child {
            break; // the match ended inside the old edge
        }
    }
    for &id in &path {
        self.lru_refresh(id);                 // new stamps: move leaves to their new place
    }
    (done, path)
}

pub fn match_prefix(&mut self, key: &[u32]) -> (usize, Vec<NodeId>) {
    self.clock += 1;
    let stamp = self.clock;
    self.walk(key, stamp)
}
```

### Insert

The walk returns how many runs are already stored; those values go back to the caller, and the rest become one new leaf.

```rust
pub fn insert(&mut self, key: &[u32], values: Vec<V>) -> (NodeId, Vec<V>) {
    let g = self.granule;
    assert!(values.len() * g == key.len(), /* ... */);  // values never shift against tokens
    self.clock += 1;
    let stamp = self.clock;
    let (done, path) = self.walk(key, stamp);
    let mut values = values;
    let fresh = values.split_off(done / g);   // `values` now holds the duplicates to hand back
    let last = path.last().copied().unwrap_or(ROOT);
    if fresh.is_empty() {
        return (last, values);                // whole key already stored
    }
    self.n_values += fresh.len();
    let leaf = Node::new(key[done..].to_vec(), fresh, Some(last), stamp);
    let first = leaf.key[..g].to_vec();
    let id = self.alloc(leaf);                // reuses a freed slot when there is one
    self.nodes[last].children.insert(first, id);
    self.lru_refresh(last); // no longer a leaf
    self.lru_refresh(id);
    (id, values)
}
```

### The LRU list

`lru_refresh` is the single rule for membership: a live, unlocked, childless non-root node is on the list; anything else is not. Every structural change calls it rather than editing the list directly.

```rust
fn lru_refresh(&mut self, id: NodeId) {
    let n = &self.nodes[id];
    let want = id != ROOT && n.live && n.lock == 0 && n.children.is_empty();
    self.lru_unlink(id);                      // no-op when not on the list
    if want {
        self.lru_link(id);
    }
}

/// Link `id` after every node whose stamp is not newer, searching back from
/// the tail: O(1) for a node stamped just now, a short walk for an old parent.
fn lru_link(&mut self, id: NodeId) {
    let stamp = self.nodes[id].stamp;
    let mut after = self.tail;
    while let Some(a) = after {
        if self.nodes[a].stamp <= stamp {
            break;
        }
        after = self.nodes[a].prev;
    }
    let before = match after {
        Some(a) => self.nodes[a].next,
        None => self.head,
    };
    // ... four link writes, updating head/tail when `after`/`before` is None ...
    self.nodes[id].in_lru = true;
}
```

### Locks

Both lock and unlock walk the parent chain to the root and refresh each node, so a locked leaf leaves the list and an unlocked childless one rejoins it. The assertion in `unlock` refuses to drive a count below zero.

```rust
pub fn lock(&mut self, id: NodeId) {
    self.node(id);                            // panics on a dead id
    let mut cur = Some(id);
    while let Some(n) = cur {
        if n == ROOT {
            break;
        }
        self.nodes[n].lock += 1;              // the node and every ancestor
        self.lru_refresh(n);
        cur = self.nodes[n].parent;
    }
}

pub fn unlock(&mut self, id: NodeId) {
    self.node(id);
    if id != ROOT {
        assert!(self.nodes[id].lock > 0, "RadixTree::unlock: node {id} is not locked");
    }
    // ... the same walk with `lock -= 1` ...
}
```

### Eviction

Pop the oldest unlocked leaf, detach it from its parent's map, hand its values to `f`, free its slot, and refresh the parent, which may now be a childless leaf and so a candidate itself.

```rust
pub fn evict<F: FnMut(V)>(&mut self, n: usize, mut f: F) -> usize {
    let g = self.granule;
    let mut freed = 0;
    while freed < n {                         // strictly less: stop as soon as n values went
        let Some(id) = self.head else { break };
        self.lru_unlink(id);
        let parent = self.nodes[id].parent.expect("the root is never on the list");
        let first = self.nodes[id].key[..g].to_vec();
        self.nodes[parent].children.remove(&first[..]); // or the parent never becomes a leaf
        let vals = std::mem::take(&mut self.nodes[id].values);
        freed += vals.len();
        self.n_values -= vals.len();
        for v in vals {
            f(v);                             // a leaf goes whole, values in label order
        }
        let node = &mut self.nodes[id];
        node.live = false;
        node.key = Vec::new();
        node.children = RobinHoodMap::new();
        self.free.push(id);
        self.lru_refresh(parent);             // a childless parent joins at its own stamp
    }
    freed
}
```

The full solution is [`radix.rs`](../../../../course/solutions/rust/crates/tl-ds/src/radix.rs).

### Why this design

`match_prefix` and `insert` are $O(|key|)$ token comparisons plus one hash lookup per edge on the path and at most one split. `lock` and `unlock` are $O(\text{depth})$. `evict` is $O(1)$ per leaf, plus the parent's relink, which is $O(1)$ for a recently stamped parent and at worst $O(\text{leaves})$. Space is one node per divergence point, not per token.

A plain trie would need one node per token: 2,000 nodes and 2,000 hash lookups for one system prompt, and an LRU over that many leaves. A hash map from chained block hashes to blocks, the [Swiss table](02-swiss-table.md) design behind the [Paged KV block pool](../../build-an-llm/p08-inference/08-paged-kv-block-pool.md), finds the longest match with one lookup per block and is a good fit for a C pool, but it has no tree: "a lock pins every ancestor" and "only leaves are evicted" become per-block reference counts kept by the caller. The radix tree states those rules structurally, which is what the prefix cache wants.

On the Rust side, `Rc<RefCell<Node>>` adds runtime borrow checks and leaks on cycles (parent and LRU links form them), and raw pointers need `unsafe`. Indices are safe and compact, at the cost of stale ids being possible, which `node()` turns into a panic via `live`. SGLang uses a heap for eviction; a heap gives $O(\log n)$ eviction without the occasional ordered-insert walk, but must be rebuilt or lazily cleaned when stamps change. The stamp-ordered list makes the common case (touch to newest) $O(1)$ and keeps the order exact.

Three tradeoffs are accepted. `match_prefix` mutates (splits and stamps), so it needs `&mut self`. A split on a pure read can add nodes that are never extended. Leaves go whole, so `evict(n)` may free more than `n` values.

## Testing it and what can go wrong

A correct tree handles these cases:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example's split and match | unit | the worked example step by step: the split, the handed-back values, the paths, the split on a match | every rule of the tree is pinned |
| Granule matches whole runs only | boundary | granule 2: runs compared whole; a key shorter than a run matches nothing | the prefix cache never shares part of a block |
| Re-inserting hands every value back | unit | a stored key returns all of the caller's values and the same node | a prefix computed twice is kept once |
| LRU order follows last use | unit | a match moves a leaf to the newest end; `evict(1)` takes the oldest leaf whole | prefixes in use stay cached |
| A childless parent becomes evictable | unit | evicting the last child puts the parent on the list at its own stamp | eviction can empty an unused subtree |
| Locks pin the path | unit | a lock pins the node and its ancestors; locks nest; unlocking frees them again | a running request's prefix is never evicted |
| A split inherits the locks below it | unit | a split under a locked node gives the new node the same count | a split never unpins a running request |
| Unlock without a lock panics | boundary | an unmatched unlock panics | bookkeeping bugs surface at once |
| Insert needs one value per granule | boundary | a wrong number of values panics | values never shift against their tokens |
| Freed slots are reused | regression | 1,000 insert-and-evict rounds keep node ids below 5 | the arena does not grow over an engine's life |
| Agrees with a naive map | property | 2,500 seeded inserts, matches, locks, unlocks, evictions at granules 1 and 2 against a map of every stored prefix: longest match, stored and handed-back values, no locked prefix evicted, every eviction the oldest unlocked leaf, values conserved | every invariant under any interleaving |

A model integration test suite, using only `tl_ds::radix::...` (split by hand, a match splits inside an edge, insert hands back the stored part, the oldest leaf goes first, locks pin ancestors, a parent becomes evictable), is in [`ds07_radix.rs`](../../../../course/solutions/learner-tests/ds.07/rust/crates/tl-ds/tests/ds07_radix.rs).

> **Warning:** The case most worth testing is "a locked leaf is not on the eviction list." Getting it wrong frees a KV block that a running request is still reading.

The full list of pitfalls is long because the tree has three interlocking pieces of bookkeeping: structure, recency, and locks.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Giving the split halves each other's values | stored values move to the wrong tokens | `split` splits `key` at `m * g` and `values` at `m` the same way: upper to `mid`, lower stays |
| Not splitting when a match ends inside an edge | the returned node covers more than the match | `walk` splits whenever `m * g` is shorter than the label |
| Forgetting to link the old child under the new middle node | the lower half of the edge disappears | `split` inserts the child into `mid`'s map and sets its parent |
| Comparing a run by its first token only | `1 2 3 9` matches `1 2 3 4` whole at granule 2 | `common` compares whole slices of `g` tokens |
| Storing values for a prefix already stored | the tree holds two copies; the caller's duplicates leak | `insert` hands back `values[..done / g]` and stores only `fresh` |
| Always linking at the newest end | a parent that becomes a leaf jumps ahead of older leaves | `lru_link` places by stamp |
| Not stamping a match | a hot prefix is evicted as if unused | `walk` stamps every node on the path, for matches too |
| Not putting a childless parent on the list | unused subtrees never go | `evict` calls `lru_refresh(parent)` |
| Leaving an evicted leaf in its parent's map | the parent never becomes childless; later walks reach a dead node | `evict` removes the leaf's first run from the parent's map |
| Locking only the node itself | an ancestor's tokens can be evicted under a running request | `lock` walks the parent chain |
| Leaving locked leaves on the list | `evict` frees a block a request is reading | `lru_refresh` requires `lock == 0`, and `lock` refreshes every node it pins |
| Not relisting a node on unlock | a finished request's prefix can never be evicted | `unlock` refreshes every node it unpins |
| A split forgetting the locks below it | the middle node is evictable under a running request | `split` copies `lock` into `mid` |
| Clamping an unmatched unlock at zero | someone else's lock is silently released | `unlock` asserts `lock > 0` |
| Accepting a wrong number of values | every value after the gap belongs to the wrong tokens | `insert` asserts `values.len() * g == key.len()` |
| Not counting inserted values | `len` and every capacity decision built on it are wrong | `insert` adds `fresh.len()`, `evict` subtracts each leaf's values |
| Not reusing freed slots | the node array grows for the life of the engine | `alloc` pops the free list first |
| Evicting while `freed <= n` | one leaf too many goes | the loop runs while `freed < n` |

## Where this leads

The [Radix prefix cache](../../build-an-llm/p08-inference/04-radix-prefix-cache.md) is this tree as `RadixTree<BlockId>` at granule `block_size`, with locks held by running requests. In the engine, [Block manager with prefix cache](../../build-an-llm/p10-serving/04-block-manager-and-prefix-cache.md) benchmarks it against the hash-indexed alternative.

Production structures that share its ideas:

- SGLang's `RadixCache` is the same tree over GPU KV pages, with a heap for eviction and paged matching (`page_size`); see [`python/sglang/srt/mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py).
- The `slab` and [`generational-arena`](https://github.com/fitzgen/generational-arena) crates add generation counters to index-linked storage, so a stale id is detected instead of reaching a reused slot.
- Linux's radix tree (now the XArray) uses fixed fan-out over integer keys with RCU-safe lookups; see `lib/xarray.c`.

### Further reading

- Morrison, *PATRICIA: Practical Algorithm To Retrieve Information Coded in Alphanumeric* (JACM 1968).
- Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104), section 3 (RadixAttention).

## Key terms

- **Radix tree**: a trie whose single-child chains are merged into labelled edges.
- **Granule ($g$)**: the run length in tokens at which labels are compared and split; one KV block.
- **Split**: inserting a middle node so that a match ending inside an edge ends at a node.
- **Arena**: a `Vec` that owns every node, with nodes linked by index.
- **Stamp**: the clock value of the last walk through a node, used to order the LRU list.
- **Lock count**: the number of locks pinning a node, counting locks on its descendants.

## Summary

- A radix tree is a trie whose chains of single children are merged into one edge with a label, so a 2,000-token prompt is one node until another prompt diverges from it.
- Matching ends inside an edge whenever two prompts diverge there; the tree splits the edge at that point, so every match ends exactly at a node.
- With a granule $g$ the tree compares and splits only at multiples of $g$ tokens: at $g = 16$ it never shares part of a KV block.
- Nodes live in a `Vec` and point at each other by index; the LRU list of evictable leaves is a doubly linked list through those indices, with no `unsafe` and no `Rc`.
- A lock pins a node and every ancestor; eviction takes the least recently used unlocked leaf, and a parent left childless becomes a leaf in its turn.

## Review questions

1. Why does a split keep the old child's id for the lower half instead of giving it to the new upper node?
   <details><summary>Answer</summary>Callers hold node ids: a request that locked the child, or a path returned earlier, refers to "the node where this prefix ends". The lower half still ends where the old child ended, so keeping its id keeps those references meaningful. The upper half is a new point in the tree and gets a new id.</details>
2. Why does the middle node created by a split inherit the child's lock count?
   <details><summary>Answer</summary>A lock pins a node and every ancestor. Every lock counted on the child was also counted on every ancestor of the child, and the middle node is now one of those ancestors. Starting it at 0 would leave it unpinned, so once its other children went it could become an evictable leaf, and evicting it would remove tokens a running request reads.</details>
3. Why does `evict` remove the leaf from its parent's `children` map before refreshing the parent?
   <details><summary>Answer</summary>`lru_refresh` decides membership from `children.is_empty()`. If the dead leaf stayed in the map, the parent would never look childless, so an unused subtree would never be evictable, and later walks could step into a freed (and possibly reused) slot.</details>
4. Why does `lru_link` search back from the tail instead of always appending at the tail?
   <details><summary>Answer</summary>The list must be in stamp order so that the head is truly the least recently used leaf. A node stamped just now does belong at the tail and the loop stops at once. But a parent that becomes a leaf after its last child is evicted keeps its old stamp; appending it at the tail would make it look recently used and let it outlive leaves that were used after it.</details>
5. Why is the loop condition `freed < n` and not `freed <= n`?
   <details><summary>Answer</summary>The contract is "until at least `n` values are gone". Once `freed >= n` the request is satisfied; with `<=`, reaching exactly `n` would evict one more leaf and drop cached prefixes nobody asked to drop.</details>
