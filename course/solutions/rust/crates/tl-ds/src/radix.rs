//! A radix tree over token ids with an index-linked LRU list of its leaves
//! (ds.07). The L8.4 prefix cache is this tree with one KV block id stored
//! per `block_size` tokens.
//!
//! Shape. Every node but the root holds an edge label `key`: a run of token
//! ids whose length is a multiple of the tree's `granule` g, plus one value
//! per g tokens of it. A node's children are keyed by the first g tokens of
//! their labels (a `RobinHoodMap<Vec<u32>, NodeId>`, ds.05), so no two
//! children start alike. The path from the root to a node spells a prefix;
//! with g = 1 this is a plain radix tree, and with g = 16 matches and splits
//! happen only at multiples of 16 tokens (whole KV blocks).
//!
//! Nodes live in a `Vec` and refer to each other by index (`NodeId`): the
//! parent link, and the `prev`/`next` links of the LRU list. Indices instead
//! of pointers are the safe-Rust answer to a doubly linked list: no `Rc`,
//! no `unsafe`, and a freed slot is reused through a free list.
//!
//! Recency. Every `match_prefix` and `insert` advances a clock by one and
//! stamps every node on the walked path with it. The LRU list holds exactly
//! the leaves that are not locked, oldest stamp first; `evict` takes from
//! its front. `lock` pins a node and its ancestors (a running request reads
//! that prefix); a pinned node is never evicted.

use crate::robin::RobinHoodMap;

/// Index of a node; `ROOT` is the root, which holds no tokens.
pub type NodeId = usize;
pub const ROOT: NodeId = 0;

struct Node<V> {
    key: Vec<u32>,
    values: Vec<V>,
    parent: Option<NodeId>,
    children: RobinHoodMap<Vec<u32>, NodeId>,
    lock: u32,
    stamp: u64,
    prev: Option<NodeId>,
    next: Option<NodeId>,
    in_lru: bool,
    live: bool,
}

impl<V> Node<V> {
    fn new(key: Vec<u32>, values: Vec<V>, parent: Option<NodeId>, stamp: u64) -> Self {
        Node {
            key,
            values,
            parent,
            children: RobinHoodMap::new(),
            lock: 0,
            stamp,
            prev: None,
            next: None,
            in_lru: false,
            live: true,
        }
    }
}

pub struct RadixTree<V> {
    granule: usize,
    nodes: Vec<Node<V>>,
    free: Vec<NodeId>,
    head: Option<NodeId>, // oldest unlocked leaf
    tail: Option<NodeId>, // newest unlocked leaf
    n_values: usize,
    clock: u64,
}

impl<V> RadixTree<V> {
    /// An empty tree whose labels come in runs of `granule` tokens.
    /// Panics when granule is 0.
    pub fn new(granule: usize) -> Self {
        assert!(granule >= 1, "RadixTree::new: granule must be at least 1");
        RadixTree {
            granule,
            nodes: vec![Node::new(Vec::new(), Vec::new(), None, 0)],
            free: Vec::new(),
            head: None,
            tail: None,
            n_values: 0,
            clock: 0,
        }
    }

    pub fn granule(&self) -> usize {
        self.granule
    }

    /// Values stored in the whole tree (one per granule of every label).
    pub fn len(&self) -> usize {
        self.n_values
    }

    pub fn is_empty(&self) -> bool {
        self.n_values == 0
    }

    /// Live nodes, the root excluded.
    pub fn node_count(&self) -> usize {
        self.nodes.len() - 1 - self.free.len()
    }

    /// The edge label of a node (empty for the root).
    pub fn key(&self, id: NodeId) -> &[u32] {
        &self.node(id).key
    }

    /// The values of a node's label, one per granule, in order.
    pub fn values(&self, id: NodeId) -> &[V] {
        &self.node(id).values
    }

    pub fn parent(&self, id: NodeId) -> Option<NodeId> {
        self.node(id).parent
    }

    pub fn child_count(&self, id: NodeId) -> usize {
        self.node(id).children.len()
    }

    /// The node's children, in increasing id order.
    pub fn children(&self, id: NodeId) -> Vec<NodeId> {
        let mut c: Vec<NodeId> = self.node(id).children.values().copied().collect();
        c.sort_unstable();
        c
    }

    /// How many locks pin this node (its own and its descendants').
    pub fn lock_count(&self, id: NodeId) -> u32 {
        self.node(id).lock
    }

    /// The unlocked leaves, least recently used first: the eviction order.
    pub fn lru_order(&self) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut cur = self.head;
        while let Some(id) = cur {
            out.push(id);
            cur = self.nodes[id].next;
        }
        out
    }

    fn node(&self, id: NodeId) -> &Node<V> {
        let n = self.nodes.get(id).filter(|n| n.live);
        n.unwrap_or_else(|| panic!("RadixTree: node {id} does not exist"))
    }

    // -- the LRU list of unlocked leaves --------------------------------------

    fn lru_unlink(&mut self, id: NodeId) {
        if !self.nodes[id].in_lru {
            return;
        }
        let (p, n) = (self.nodes[id].prev, self.nodes[id].next);
        match p {
            Some(p) => self.nodes[p].next = n,
            None => self.head = n,
        }
        match n {
            Some(n) => self.nodes[n].prev = p,
            None => self.tail = p,
        }
        let node = &mut self.nodes[id];
        node.prev = None;
        node.next = None;
        node.in_lru = false;
    }

    /// Link `id` in stamp order: after every node whose stamp is not newer.
    /// A node touched just now goes to the tail in O(1); a parent that
    /// becomes a leaf after an eviction walks in from the old end.
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
        self.nodes[id].prev = after;
        self.nodes[id].next = before;
        match after {
            Some(a) => self.nodes[a].next = Some(id),
            None => self.head = Some(id),
        }
        match before {
            Some(b) => self.nodes[b].prev = Some(id),
            None => self.tail = Some(id),
        }
        self.nodes[id].in_lru = true;
    }

    /// Put `id` on the list or take it off, by what it is now.
    fn lru_refresh(&mut self, id: NodeId) {
        let n = &self.nodes[id];
        let want = id != ROOT && n.live && n.lock == 0 && n.children.is_empty();
        self.lru_unlink(id);
        if want {
            self.lru_link(id);
        }
    }

    // -- structure ------------------------------------------------------------

    fn alloc(&mut self, node: Node<V>) -> NodeId {
        match self.free.pop() {
            Some(id) => {
                self.nodes[id] = node;
                id
            }
            None => {
                self.nodes.push(node);
                self.nodes.len() - 1
            }
        }
    }

    /// Granules shared by `label` and the start of `rest`.
    fn common(&self, label: &[u32], rest: &[u32]) -> usize {
        let g = self.granule;
        let mut m = 0;
        while (m + 1) * g <= label.len()
            && (m + 1) * g <= rest.len()
            && label[m * g..(m + 1) * g] == rest[m * g..(m + 1) * g]
        {
            m += 1;
        }
        m
    }

    /// Split `child` after `m` granules: a new node takes the first m
    /// granules of its label and values and becomes child's parent. The
    /// new node inherits child's lock count (every lock below passes
    /// through it). Returns the new node.
    fn split(&mut self, child: NodeId, m: usize, stamp: u64) -> NodeId {
        let g = self.granule;
        let parent = self.nodes[child].parent.expect("the root is never split");
        let lower_key = self.nodes[child].key.split_off(m * g);
        let upper_key = std::mem::replace(&mut self.nodes[child].key, lower_key);
        let lower_vals = self.nodes[child].values.split_off(m);
        let upper_vals = std::mem::replace(&mut self.nodes[child].values, lower_vals);
        let mut mid = Node::new(upper_key, upper_vals, Some(parent), stamp);
        mid.lock = self.nodes[child].lock;
        let first = mid.key[..g].to_vec();
        let mid_id = self.alloc(mid);
        let lower_first = self.nodes[child].key[..g].to_vec();
        self.nodes[mid_id].children.insert(lower_first, child);
        self.nodes[child].parent = Some(mid_id);
        self.nodes[parent].children.insert(first, mid_id);
        mid_id
    }

    /// Walk from the root along `key`, splitting the last edge when the
    /// match ends inside it. Returns the granules matched and the path
    /// (root excluded); stamps every node on the path with `stamp`.
    fn walk(&mut self, key: &[u32], stamp: u64) -> (usize, Vec<NodeId>) {
        let g = self.granule;
        let mut cur = ROOT;
        let mut done = 0; // tokens matched
        let mut path = Vec::new();
        while key.len() - done >= g {
            let rest = &key[done..];
            let child = match self.nodes[cur].children.get(&rest[..g]) {
                Some(&c) => c,
                None => break,
            };
            let m = self.common(&self.nodes[child].key, rest);
            let node = if m * g < self.nodes[child].key.len() {
                self.split(child, m, stamp)
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
            self.lru_refresh(id);
        }
        (done, path)
    }

    /// The longest prefix of `key` stored in the tree, as a multiple of the
    /// granule: (tokens matched, the nodes on the path in order, root
    /// excluded). The last node ends exactly at the matched length (its
    /// edge is split when the match ends inside it). Marks the path as just
    /// used.
    pub fn match_prefix(&mut self, key: &[u32]) -> (usize, Vec<NodeId>) {
        self.clock += 1;
        let stamp = self.clock;
        self.walk(key, stamp)
    }

    /// Stores `key` with one value per granule (`values.len() * granule ==
    /// key.len()`, else a panic). The part of `key` already in the tree keeps
    /// its stored values, and the caller's values for that part are handed
    /// back in order; the rest becomes a new leaf. Returns the node that ends
    /// at `key` (ROOT for an empty key) and the values handed back.
    pub fn insert(&mut self, key: &[u32], values: Vec<V>) -> (NodeId, Vec<V>) {
        let g = self.granule;
        assert!(
            values.len() * g == key.len(),
            "RadixTree::insert: {} values for {} tokens at granule {g}",
            values.len(),
            key.len()
        );
        self.clock += 1;
        let stamp = self.clock;
        let (done, path) = self.walk(key, stamp);
        let mut values = values;
        let fresh = values.split_off(done / g);
        let last = path.last().copied().unwrap_or(ROOT);
        if fresh.is_empty() {
            return (last, values);
        }
        self.n_values += fresh.len();
        let leaf = Node::new(key[done..].to_vec(), fresh, Some(last), stamp);
        let first = leaf.key[..g].to_vec();
        let id = self.alloc(leaf);
        self.nodes[last].children.insert(first, id);
        self.lru_refresh(last); // no longer a leaf
        self.lru_refresh(id);
        (id, values)
    }

    /// Pins `id` and every ancestor: none of them is evicted until a
    /// matching `unlock`. Locking the root does nothing.
    pub fn lock(&mut self, id: NodeId) {
        self.node(id);
        let mut cur = Some(id);
        while let Some(n) = cur {
            if n == ROOT {
                break;
            }
            self.nodes[n].lock += 1;
            self.lru_refresh(n);
            cur = self.nodes[n].parent;
        }
    }

    /// Undoes one `lock(id)`. Panics when `id` is not locked.
    pub fn unlock(&mut self, id: NodeId) {
        self.node(id);
        if id != ROOT {
            assert!(
                self.nodes[id].lock > 0,
                "RadixTree::unlock: node {id} is not locked"
            );
        }
        let mut cur = Some(id);
        while let Some(n) = cur {
            if n == ROOT {
                break;
            }
            self.nodes[n].lock -= 1;
            self.lru_refresh(n);
            cur = self.nodes[n].parent;
        }
    }

    /// Removes least recently used unlocked leaves until at least `n` values
    /// are gone or nothing more can go, calling `f` on each value removed
    /// (a leaf's values in order). A parent left childless becomes a leaf
    /// and may go next. Returns the number of values removed.
    pub fn evict<F: FnMut(V)>(&mut self, n: usize, mut f: F) -> usize {
        let g = self.granule;
        let mut freed = 0;
        while freed < n {
            let Some(id) = self.head else { break };
            self.lru_unlink(id);
            let parent = self.nodes[id]
                .parent
                .expect("the root is never on the list");
            let first = self.nodes[id].key[..g].to_vec();
            self.nodes[parent].children.remove(&first[..]);
            let vals = std::mem::take(&mut self.nodes[id].values);
            freed += vals.len();
            self.n_values -= vals.len();
            for v in vals {
                f(v);
            }
            let node = &mut self.nodes[id];
            node.live = false;
            node.key = Vec::new();
            node.children = RobinHoodMap::new();
            self.free.push(id);
            self.lru_refresh(parent);
        }
        freed
    }
}
