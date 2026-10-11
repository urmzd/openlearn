//! ds.07 course tests: the radix tree over token ids (tl-ds `radix`).
//!
//! Annotated exemplars (DESIGN 5.12). The hand examples use granule 1 (a
//! plain radix tree) and granule 2 (matches in whole runs of two tokens, as
//! the L8.4 prefix cache uses whole KV blocks). The property test checks the
//! tree against a naive model: a map from every stored granule-aligned
//! prefix to its value, last use, and lock count. Random operations come
//! from the PCG32 transcribed below from contracts/spec/pcg32.md (the frozen
//! generator, D35), never from a crate.

use std::collections::HashMap;

use tl_ds::radix::{NodeId, RadixTree, ROOT};

// ---------------------------------------------------------------------------
// helpers

/// The frozen PCG32 (spec/pcg32.md): `pcg32_srandom_r(seed, seq)`.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut g = Pcg32 {
            state: 0,
            inc: (seq << 1) | 1,
        };
        g.next_u32();
        g.state = g.state.wrapping_add(seed);
        g.next_u32();
        g
    }
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        xs.rotate_right((old >> 59) as u32)
    }
    fn below(&mut self, n: u32) -> u32 {
        let t = n.wrapping_neg() % n;
        loop {
            let r = self.next_u32();
            if r >= t {
                return r % n;
            }
        }
    }
}

fn ol_seed() -> u64 {
    std::env::var("OL_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// The values on a matched path, in order.
fn path_values<V: Clone>(t: &RadixTree<V>, path: &[NodeId]) -> Vec<V> {
    path.iter().flat_map(|&id| t.values(id).to_vec()).collect()
}

fn evict_all<V>(t: &mut RadixTree<V>, n: usize) -> Vec<V> {
    let mut out = Vec::new();
    t.evict(n, |v| out.push(v));
    out
}

// ---------------------------------------------------------------------------
// the worked example

#[test]
fn hand_example_split_and_match() {
    // WHY: section 3 by hand. Insert 1 2 3 (values a b c), then 1 2 4 5:
    //      the edge 1 2 3 splits after two tokens into a node 1 2 (a b)
    //      with children 3 (c) and 4 5 (d e), and the caller's values x y
    //      for the shared 1 2 come back. A match that ends inside an edge
    //      splits it too, so the returned last node ends exactly there.
    // KIND: unit
    // CATCHES: s01, s02, s03
    // CHAPTER: ds.07 section 3, Worked example by hand
    let mut t = RadixTree::new(1);
    let (abc, back) = t.insert(&[1, 2, 3], vec!['a', 'b', 'c']);
    assert!(back.is_empty());
    assert_eq!(t.key(abc), &[1, 2, 3]);
    let (de, back) = t.insert(&[1, 2, 4, 5], vec!['x', 'y', 'd', 'e']);
    assert_eq!(
        back,
        vec!['x', 'y'],
        "the stored prefix keeps a b; the caller's x y come back"
    );
    let mid = t.parent(de).unwrap();
    assert_eq!(t.key(mid), &[1, 2]);
    assert_eq!(t.values(mid), &['a', 'b']);
    assert_eq!(
        t.key(abc),
        &[3],
        "the old leaf keeps its id and the rest of its label"
    );
    assert_eq!(t.parent(abc), Some(mid));
    assert_eq!(t.key(de), &[4, 5]);
    assert_eq!(t.children(mid).len(), 2);
    assert_eq!((t.len(), t.node_count()), (5, 3));

    let (n, path) = t.match_prefix(&[1, 2, 3, 9]);
    assert_eq!((n, path.clone()), (3, vec![mid, abc]));
    assert_eq!(path_values(&t, &path), vec!['a', 'b', 'c']);
    let (n, path) = t.match_prefix(&[1, 2, 7]);
    assert_eq!((n, path), (2, vec![mid]));
    let (n, path) = t.match_prefix(&[1, 9]);
    assert_eq!(n, 1);
    let one = path[0];
    assert_eq!((t.key(one), t.values(one)), (&[1u32][..], &['a'][..]));
    assert_eq!(t.parent(mid), Some(one));
    assert_eq!(t.key(mid), &[2]);
    assert_eq!(t.match_prefix(&[7, 1]), (0, vec![]));
    assert_eq!(
        t.len(),
        5,
        "splits move values, they never copy or drop them"
    );
}

#[test]
fn granule_matches_whole_runs_only() {
    // WHY: at granule 2 the tree compares runs of two tokens. 1 2 3 9
    //      against a stored 1 2 3 4 matches 2 tokens, not 3, and the split
    //      falls between the runs; a key shorter than one run matches
    //      nothing. This is what keeps L8.4 from sharing half a KV block.
    // KIND: boundary
    // CATCHES: s04
    // CHAPTER: ds.07 section 2
    let mut t = RadixTree::new(2);
    t.insert(&[1, 2, 3, 4], vec![10u32, 11]);
    let (n, path) = t.match_prefix(&[1, 2, 3, 9]);
    assert_eq!(n, 2);
    assert_eq!(t.key(path[0]), &[1, 2]);
    assert_eq!(t.values(path[0]), &[10]);
    assert_eq!(t.match_prefix(&[1]), (0, vec![]));
    assert_eq!(t.match_prefix(&[1, 3, 3, 4]), (0, vec![]));
    let (n, _) = t.match_prefix(&[1, 2, 3, 4, 5]);
    assert_eq!(n, 4);
    let (leaf, back) = t.insert(&[1, 2, 5, 6], vec![20, 21]);
    assert_eq!(back, vec![20]);
    assert_eq!(t.values(leaf), &[21]);
    assert_eq!(t.len(), 3);
}

#[test]
fn reinserting_hands_every_value_back() {
    // WHY: inserting a key that is already stored stores nothing: all of the
    //      caller's values come back and the node is the existing one, so a
    //      prefix computed twice by two requests is kept once.
    // KIND: unit
    // CATCHES: s05
    // CHAPTER: ds.07 section 4
    let mut t = RadixTree::new(1);
    let (a, _) = t.insert(&[5, 6, 7], vec![1u32, 2, 3]);
    let (b, back) = t.insert(&[5, 6, 7], vec![7, 8, 9]);
    assert_eq!(a, b);
    assert_eq!(back, vec![7, 8, 9]);
    assert_eq!(t.values(a), &[1, 2, 3]);
    assert_eq!((t.len(), t.node_count()), (3, 1));
    assert_eq!(t.insert(&[], Vec::new()), (ROOT, vec![]));
}

#[test]
fn lru_order_follows_last_use() {
    // WHY: the eviction order is the order of last use among unlocked
    //      leaves. A, B, C inserted in that order; a match touches A, so B
    //      is now the oldest and evict(1) removes exactly B's values.
    // KIND: unit
    // CATCHES: s07
    // CHAPTER: ds.07 section 2
    let mut t = RadixTree::new(1);
    let (a, _) = t.insert(&[1, 1], vec![10u32, 11]);
    let (b, _) = t.insert(&[2, 2], vec![20, 21]);
    let (c, _) = t.insert(&[3, 3], vec![30, 31]);
    assert_eq!(t.lru_order(), vec![a, b, c]);
    t.match_prefix(&[1, 1, 5]);
    assert_eq!(t.lru_order(), vec![b, c, a]);
    assert_eq!(evict_all(&mut t, 1), vec![20, 21], "a leaf goes whole");
    assert_eq!(t.lru_order(), vec![c, a]);
    assert_eq!(t.len(), 4);
}

#[test]
fn a_childless_parent_becomes_evictable() {
    // WHY: evicting a parent's last child turns the parent into a leaf; it
    //      joins the list at its own last use and can go next, so eviction
    //      can empty a whole unlocked subtree, oldest leaves first.
    // KIND: unit
    // CATCHES: s06, s08, s09, m03
    // CHAPTER: ds.07 section 3
    let mut t = RadixTree::new(1);
    t.insert(&[1, 2, 3], vec![1u32, 2, 3]);
    t.insert(&[1, 2, 4], vec![1, 2, 4]); // splits: 1 2 -> {3, 4}
    let (z, _) = t.insert(&[9], vec![9]);
    assert_eq!(t.lru_order().len(), 3);
    assert_eq!(evict_all(&mut t, 2), vec![3, 4]);
    let order = t.lru_order();
    assert_eq!(order.len(), 2);
    assert_eq!(t.key(order[0]), &[1, 2], "the parent last used before z");
    assert_eq!(order[1], z);
    assert_eq!(evict_all(&mut t, 1), vec![1, 2]);
    assert_eq!(evict_all(&mut t, 10), vec![9]);
    assert!(t.is_empty());
    assert_eq!(t.node_count(), 0);
    assert_eq!(evict_all(&mut t, 1), Vec::<u32>::new());
}

#[test]
fn locks_pin_the_path() {
    // WHY: a running request locks the node its prefix ends at, which pins
    //      that node and every ancestor. Eviction then frees only what is
    //      not on a locked path; unlocking makes it evictable again. Locks
    //      nest: two locks need two unlocks.
    // KIND: unit
    // CATCHES: s10, s11, s12
    // CHAPTER: ds.07 section 2
    let mut t = RadixTree::new(1);
    let (long, _) = t.insert(&[1, 2, 3, 4], vec![1u32, 2, 3, 4]);
    let (other, _) = t.insert(&[1, 2, 7], vec![0, 0, 7]);
    let mid = t.parent(other).unwrap();
    t.lock(long);
    t.lock(long);
    assert_eq!(
        (t.lock_count(long), t.lock_count(mid), t.lock_count(other)),
        (2, 2, 0)
    );
    assert_eq!(t.lru_order(), vec![other]);
    assert_eq!(evict_all(&mut t, 100), vec![7]);
    assert_eq!(t.len(), 4);
    t.unlock(long);
    assert_eq!(evict_all(&mut t, 100), Vec::<u32>::new());
    t.unlock(long);
    assert_eq!(t.lock_count(mid), 0);
    assert_eq!(evict_all(&mut t, 100), vec![3, 4, 1, 2]);
    t.lock(ROOT);
    t.unlock(ROOT);
}

#[test]
fn a_split_inherits_the_locks_below_it() {
    // WHY: splitting a locked node's edge puts a new node between it and
    //      its parent. Every lock on the lower node passes through the new
    //      one, so the new node starts with the same count; otherwise it
    //      could be evicted from under a running request.
    // KIND: unit
    // CATCHES: s13
    // CHAPTER: ds.07 section 5, Pitfalls
    let mut t = RadixTree::new(1);
    let (leaf, _) = t.insert(&[1, 2, 3, 4], vec![1u32, 2, 3, 4]);
    t.lock(leaf);
    let (n, path) = t.match_prefix(&[1, 2, 9]);
    assert_eq!(n, 2);
    let mid = path[0];
    assert_eq!(t.lock_count(mid), 1);
    let (side, _) = t.insert(&[1, 2, 8], vec![0, 0, 8]);
    assert_eq!(evict_all(&mut t, 100), vec![8]);
    assert_eq!(t.len(), 4);
    t.unlock(leaf);
    assert_eq!(t.lock_count(mid), 0);
    let _ = side;
    assert_eq!(evict_all(&mut t, 100), vec![3, 4, 1, 2]);
}

#[test]
#[should_panic]
fn unlock_without_a_lock_panics() {
    // WHY: an unmatched unlock is a bug in the caller's bookkeeping; letting
    //      a count go below zero would unpin a prefix another request holds.
    // KIND: boundary
    // CATCHES: s14
    // CHAPTER: ds.07 section 4
    let mut t = RadixTree::new(1);
    let (leaf, _) = t.insert(&[1], vec![1u32]);
    t.unlock(leaf);
}

#[test]
#[should_panic]
fn insert_needs_one_value_per_granule() {
    // WHY: every granule of a label carries exactly one value (one KV block
    //      for L8.4); a mismatch would shift every value after it.
    // KIND: boundary
    // CATCHES: s15
    // CHAPTER: ds.07 section 4
    let mut t = RadixTree::new(2);
    t.insert(&[1, 2, 3, 4], vec![1u32]);
}

#[test]
fn freed_slots_are_reused() {
    // WHY: the engine inserts and evicts for its whole life. Node slots of
    //      evicted leaves go on a free list and are reused, so 1,000 insert
    //      and evict rounds never grow the node array past what one round
    //      needs (ids stay small).
    // KIND: regression
    // CATCHES: m02
    // CHAPTER: ds.07 section 2
    let mut t = RadixTree::new(1);
    let mut max_id = 0;
    for r in 0..1000u32 {
        let (a, _) = t.insert(&[r % 7, 1, 2], vec![r, r, r]);
        let (b, _) = t.insert(&[r % 7, 1, 3], vec![r, r, r]);
        max_id = max_id.max(a).max(b);
        evict_all(&mut t, usize::MAX);
        assert!(t.is_empty());
    }
    assert!(max_id <= 4, "max node id {max_id}");
}

// ---------------------------------------------------------------------------
// model-based property test

#[derive(Clone, Copy)]
struct Pos {
    value: u32,
    last_use: u64,
    locks: u32,
}

/// Every stored granule-aligned prefix of the tree, keyed by its tokens.
struct Model {
    g: usize,
    pos: HashMap<Vec<u32>, Pos>,
}

impl Model {
    fn longest(&self, key: &[u32]) -> usize {
        let mut n = 0;
        while n + self.g <= key.len() && self.pos.contains_key(&key[..n + self.g]) {
            n += self.g;
        }
        n
    }
    fn touch(&mut self, key: &[u32], upto: usize, now: u64) {
        let mut n = self.g;
        while n <= upto {
            self.pos.get_mut(&key[..n]).unwrap().last_use = now;
            n += self.g;
        }
    }
    /// No stored prefix extends k by one granule (the alphabet is 0..3).
    fn is_leaf(&self, k: &[u32]) -> bool {
        let combos = 3u32.pow(self.g as u32);
        !(0..combos).any(|c| {
            let mut e = k.to_vec();
            let mut c = c;
            for _ in 0..self.g {
                e.push(c % 3);
                c /= 3;
            }
            self.pos.contains_key(&e)
        })
    }
}

fn model_run(seed: u64, g: usize) {
    let mut r = Pcg32::new(seed, g as u64);
    let mut t: RadixTree<u32> = RadixTree::new(g);
    let mut m = Model {
        g,
        pos: HashMap::new(),
    };
    let mut locks: Vec<(NodeId, Vec<u32>)> = Vec::new(); // node and the prefix it ends at
    let mut next_val = 1u32;
    let (mut inserted, mut evicted) = (0usize, 0usize);
    for op in 1..=2500u64 {
        let len = g * (1 + r.below(5) as usize);
        let key: Vec<u32> = (0..len).map(|_| r.below(3)).collect();
        match r.below(10) {
            0..=3 => {
                let vals: Vec<u32> = (0..len / g)
                    .map(|_| {
                        next_val += 1;
                        next_val
                    })
                    .collect();
                let (node, back) = t.insert(&key, vals.clone());
                let had = m.longest(&key);
                assert_eq!(
                    back,
                    vals[..had / g].to_vec(),
                    "values handed back for the stored part"
                );
                for i in had / g..len / g {
                    m.pos.insert(
                        key[..(i + 1) * g].to_vec(),
                        Pos {
                            value: vals[i],
                            last_use: op,
                            locks: 0,
                        },
                    );
                }
                m.touch(&key, len, op);
                assert_eq!(t.key(node).last(), key.last());
                inserted += vals.len() - back.len();
            }
            4..=6 => {
                let (n, path) = t.match_prefix(&key);
                assert_eq!(n, m.longest(&key), "longest stored prefix");
                let want: Vec<u32> = (1..=n / g).map(|i| m.pos[&key[..i * g]].value).collect();
                assert_eq!(path_values(&t, &path), want);
                m.touch(&key, n, op);
                if n > 0 && r.below(2) == 0 {
                    let node = *path.last().unwrap();
                    t.lock(node);
                    let mut k = g;
                    while k <= n {
                        m.pos.get_mut(&key[..k]).unwrap().locks += 1;
                        k += g;
                    }
                    locks.push((node, key[..n].to_vec()));
                }
            }
            7 => {
                if !locks.is_empty() {
                    let (node, prefix) = locks.swap_remove(r.below(locks.len() as u32) as usize);
                    t.unlock(node);
                    let mut k = g;
                    while k <= prefix.len() {
                        m.pos.get_mut(&prefix[..k]).unwrap().locks -= 1;
                        k += g;
                    }
                }
            }
            _ => {
                // evict one leaf: the oldest unlocked one, whole
                let oldest = m
                    .pos
                    .iter()
                    .filter(|(k, p)| p.locks == 0 && m.is_leaf(k))
                    .map(|(_, p)| p.last_use)
                    .min();
                let gone = evict_all(&mut t, 1);
                match oldest {
                    None => assert!(gone.is_empty(), "nothing unlocked to evict"),
                    Some(stamp) => {
                        assert!(!gone.is_empty());
                        let keys: Vec<Vec<u32>> = gone
                            .iter()
                            .map(|v| m.pos.iter().find(|(_, p)| p.value == *v).unwrap().0.clone())
                            .collect();
                        for k in &keys {
                            let p = m.pos[k];
                            assert_eq!(p.locks, 0, "a locked prefix was evicted");
                            assert_eq!(p.last_use, stamp, "not the least recently used leaf");
                        }
                        for w in keys.windows(2) {
                            assert!(
                                w[1].len() == w[0].len() + g && w[1].starts_with(&w[0]),
                                "a leaf is one chain"
                            );
                        }
                        assert!(m.is_leaf(keys.last().unwrap()));
                        for k in &keys {
                            m.pos.remove(k);
                        }
                        evicted += gone.len();
                    }
                }
            }
        }
        assert_eq!(t.len(), m.pos.len());
        assert_eq!(
            inserted,
            t.len() + evicted,
            "every stored value is in the tree or was evicted"
        );
    }
}

#[test]
fn model_based_against_a_naive_map() {
    // WHY: 2,500 seeded inserts, matches, locks, unlocks, and one-leaf
    //      evictions over a 3-token alphabet, at granules 1 and 2, checked
    //      after every step against a naive map of stored prefixes: the
    //      longest match, the stored values, values handed back, a locked
    //      prefix never evicted, every eviction the least recently used
    //      unlocked leaf, and values conserved.
    // KIND: property
    // CATCHES: s01, s02, s04, s05, s06, s08, s10, s11, s13, m01
    // CHAPTER: ds.07 section 4
    for g in [1usize, 2] {
        model_run(ol_seed(), g);
    }
}
