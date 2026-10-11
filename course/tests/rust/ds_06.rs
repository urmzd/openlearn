//! ds.06 course tests: the stable binary heap and the lazy-deletion heap
//! (tl-ds `heap`).
//!
//! Annotated exemplars (DESIGN 5.12). The hand examples spell out the array
//! after every push and pop; the models are a stably sorted Vec (for `Heap`)
//! and a list of live (key, push order, handle) triples (for `LazyHeap`).
//! Random operations come from the PCG32 transcribed below from
//! contracts/spec/pcg32.md (the frozen generator, D35).

use std::cmp::Ordering;

use tl_ds::heap::{Handle, Heap, LazyHeap};

// ---------------------------------------------------------------------------
// helpers

/// The frozen PCG32 (spec/pcg32.md): `pcg32_srandom_r(seed, seq)`.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut g = Pcg32 { state: 0, inc: (seq << 1) | 1 };
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
    std::env::var("OL_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn arr(h: &Heap<i32, fn(&i32, &i32) -> Ordering>) -> Vec<i32> {
    h.array().into_iter().copied().collect()
}

/// Order pairs by their first field only: equal keys are ties.
fn by_key(a: &(u32, u32), b: &(u32, u32)) -> Ordering {
    a.0.cmp(&b.0)
}

// ---------------------------------------------------------------------------
// the worked example

#[test]
fn hand_example_sift() {
    // WHY: section 3 by hand: pushing 5, 3, 8, 1, 9, 2 into a min-heap. Each
    //      push appends and swaps with its parent (i - 1) / 2 while it is
    //      smaller: 1 climbs two levels, 2 climbs one.
    // KIND: unit
    // CATCHES: s01, m03
    // CHAPTER: ds.06 section 3, Worked example by hand
    let mut h: Heap<i32, fn(&i32, &i32) -> Ordering> = Heap::new_min();
    let want: [&[i32]; 6] = [&[5], &[3, 5], &[3, 5, 8], &[1, 3, 8, 5], &[1, 3, 8, 5, 9], &[1, 3, 2, 5, 9, 8]];
    for (x, w) in [5, 3, 8, 1, 9, 2].into_iter().zip(want) {
        h.push(x);
        assert_eq!(arr(&h), w, "after push {x}");
    }
    assert_eq!(h.peek(), Some(&1));
    assert_eq!(h.len(), 6);
}

#[test]
fn pop_sifts_down_through_the_smaller_child() {
    // WHY: pop moves the last item (8) to the root and swaps it with the
    //      SMALLER child (2, not 3) while that child is smaller: the array
    //      becomes [2, 3, 8, 5, 9]. Swapping with the left child would put 3
    //      above 2 and break the heap.
    // KIND: unit
    // CATCHES: s05, m01
    // CHAPTER: ds.06 section 3, Worked example by hand
    let mut h: Heap<i32, fn(&i32, &i32) -> Ordering> = Heap::new_min();
    for x in [5, 3, 8, 1, 9, 2] {
        h.push(x);
    }
    assert_eq!(h.pop(), Some(1));
    assert_eq!(arr(&h), vec![2, 3, 8, 5, 9]);
    assert_eq!(h.pop(), Some(2));
    assert_eq!(arr(&h), vec![3, 5, 8, 9]);
    assert_eq!(h.into_sorted_vec(), vec![3, 5, 8, 9]);
}

#[test]
fn equal_items_come_out_in_push_order() {
    // WHY: stability. Items that compare Equal leave first-in, first-out,
    //      whatever the array looked like. L10.2 queues requests by priority
    //      and relies on arrival order among equals; L1.5 relies on it for
    //      nothing, because its keys are unique, but a heap that is not
    //      stable gives a different order on every input permutation.
    // KIND: unit
    // CATCHES: s02
    // CHAPTER: ds.06 section 2.3
    let mut h = Heap::new(by_key);
    for (i, k) in [1, 1, 0, 1, 0, 2, 1].into_iter().enumerate() {
        h.push((k, i as u32));
    }
    let out: Vec<u32> = std::iter::from_fn(|| h.pop()).map(|p| p.1).collect();
    assert_eq!(out, vec![2, 4, 0, 1, 3, 6, 5]);
}

#[test]
fn lazy_remove_hand_example() {
    // WHY: section 3, second half. Removing 3 by its handle only bumps its
    //      slot's generation: the entry stays in the array (stale_len 1)
    //      until it reaches the root, where peek discards it. Pushing 4 then
    //      reuses the slot with a new generation, so the old handle can never
    //      remove 4.
    // KIND: unit
    // CATCHES: s03, s04
    // CHAPTER: ds.06 section 3, Worked example by hand
    let mut h = LazyHeap::new(|a: &i32, b: &i32| a.cmp(b));
    let h5 = h.push(5);
    let h3 = h.push(3);
    let h8 = h.push(8);
    assert_eq!((h.len(), h.stale_len()), (3, 0));
    assert!(h.remove(h3));
    assert_eq!((h.len(), h.stale_len()), (2, 1));
    assert!(!h.contains(h3) && h.contains(h5) && h.contains(h8));
    assert_eq!(h.peek(), Some(&5), "the stale 3 is skipped");
    assert_eq!(h.stale_len(), 0, "peek discarded it");
    let h4 = h.push(4);
    assert_ne!(h4, h3, "the reused slot has a new generation");
    assert!(!h.remove(h3), "a stale handle never removes a newer item");
    assert_eq!(h.len(), 3);
    assert_eq!(h.pop_with_handle(), Some((h4, 4)));
    assert_eq!(h.pop(), Some(5));
    assert!(!h.remove(h5), "popped items cannot be removed");
    assert!(h.remove(h8));
    assert_eq!((h.pop(), h.len(), h.is_empty()), (None, 0, true));
}

// ---------------------------------------------------------------------------
// boundaries

#[test]
fn empty_and_single_item() {
    // WHY: the edges every caller hits: pop and peek of an empty heap are
    //      None, one item comes straight back, and clear empties the heap.
    // KIND: boundary
    // CATCHES: m04
    // CHAPTER: ds.06 section 4
    let mut h: Heap<i32, fn(&i32, &i32) -> Ordering> = Heap::new_min();
    assert_eq!((h.pop(), h.peek(), h.len(), h.is_empty()), (None, None, 0, true));
    h.push(7);
    assert_eq!((h.peek(), h.len()), (Some(&7), 1));
    assert_eq!(h.pop(), Some(7));
    assert_eq!(h.pop(), None);
    for x in [3, 1, 2] {
        h.push(x);
    }
    h.clear();
    assert!(h.is_empty() && h.pop().is_none());
    let mut l = LazyHeap::new(|a: &u8, b: &u8| a.cmp(b));
    assert_eq!((l.pop(), l.peek().copied(), l.len()), (None, None, 0));
    let x = l.push(1);
    assert!(l.remove(x));
    assert_eq!((l.pop(), l.peek().copied(), l.is_empty()), (None, None, true));
}

#[test]
fn comparator_decides_the_order() {
    // WHY: the order is the comparator, not T's Ord: a reversed comparator
    //      makes a max-heap. L10.2 orders requests by (priority, arrival).
    // KIND: unit
    // CATCHES: s01, m03
    // CHAPTER: ds.06 section 4
    let mut h = Heap::new(|a: &i32, b: &i32| b.cmp(a));
    for x in [4, -1, 9, 9, 0, 7] {
        h.push(x);
    }
    assert_eq!(h.into_sorted_vec(), vec![9, 9, 7, 4, 0, -1]);
}

#[test]
fn from_vec_heapifies_in_place() {
    // WHY: from_vec builds a heap in O(n) by sifting down every parent from
    //      the last one (index n/2 - 1) to the root. Skipping the root, or
    //      sifting up instead, leaves a heap that pops out of order. Equal
    //      items keep their order in the input.
    // KIND: unit
    // CATCHES: s07
    // CHAPTER: ds.06 section 2.2
    let h = Heap::from_vec(vec![9, 4, 7, 1, 8, 2, 6, 3, 5, 0], |a: &i32, b: &i32| a.cmp(b));
    assert_eq!(h.peek(), Some(&0));
    assert_eq!(h.into_sorted_vec(), (0..10).collect::<Vec<_>>());
    let h = Heap::from_vec(vec![(2, 0), (1, 1), (2, 2), (1, 3), (0, 4)], by_key);
    let order: Vec<u32> = h.into_sorted_vec().into_iter().map(|p| p.1).collect();
    assert_eq!(order, vec![4, 1, 3, 0, 2]);
    let mut h = Heap::from_vec(vec![3, 1, 2], |a: &i32, b: &i32| a.cmp(b));
    h.push(0);
    assert_eq!(h.into_sorted_vec(), vec![0, 1, 2, 3]);
}

#[test]
fn bpe_merge_queue_pops_lowest_rank_then_leftmost() {
    // WHY: L1.5's merge queue keys pairs by (merge rank, position of the left
    //      symbol). The lowest rank goes first; among pairs of one rank, the
    //      leftmost; a merge removes its neighbours' pairs by handle.
    // KIND: unit
    // CATCHES: s04
    // CHAPTER: ds.06 section 6
    let mut q = LazyHeap::new(|a: &(u32, usize), b: &(u32, usize)| a.cmp(b));
    let hs: Vec<Handle> = [(5, 0), (2, 1), (2, 3), (7, 2), (2, 2)].iter().map(|&p| q.push(p)).collect();
    assert_eq!(q.pop(), Some((2, 1)));
    // Merging at position 1 destroys the pairs at 0 and 2.
    assert!(q.remove(hs[0]));
    assert!(q.remove(hs[4]));
    assert_eq!(q.pop(), Some((2, 3)));
    assert_eq!(q.pop(), Some((7, 2)));
    assert_eq!(q.pop(), None);
}

// ---------------------------------------------------------------------------
// models

#[test]
fn differential_against_sorted_vec() {
    // WHY: 20,000 random pushes and pops agree with a model that keeps the
    //      items in a Vec sorted stably by key: pop takes the first item. Few
    //      distinct keys, so ties are common and stability is tested too.
    // KIND: differential
    // CATCHES: s01, s02, s05, m01, m02
    // CHAPTER: ds.06 section 4
    let mut g = Pcg32::new(ol_seed(), 5);
    let mut h = Heap::new(by_key);
    let mut model: Vec<(u32, u32)> = Vec::new();
    for step in 0..20_000u32 {
        if g.below(5) < 3 {
            let item = (g.below(16), step);
            h.push(item);
            let at = model.partition_point(|x| x.0 <= item.0);
            model.insert(at, item);
        } else {
            let want = if model.is_empty() { None } else { Some(model.remove(0)) };
            assert_eq!(h.pop(), want, "pop at step {step}");
        }
        assert_eq!(h.len(), model.len());
        assert_eq!(h.peek(), model.first());
    }
}

#[test]
fn lazy_differential_against_model() {
    // WHY: 20,000 random pushes, removes (by live AND by stale handles), and
    //      pops agree with a model of the live items, through every
    //      compaction. A removed item never comes out; a stale handle never
    //      removes anything; ties leave in push order.
    // KIND: differential
    // CATCHES: s02, s03, s04, s08
    // CHAPTER: ds.06 section 4
    let mut g = Pcg32::new(ol_seed(), 5);
    let mut h = LazyHeap::new(by_key);
    let mut live: Vec<((u32, u32), Handle)> = Vec::new(); // in push order
    let mut dead: Vec<Handle> = Vec::new();
    for step in 0..20_000u32 {
        match g.below(10) {
            0..=3 => {
                let item = (g.below(32), step);
                live.push((item, h.push(item)));
            }
            4..=6 if !live.is_empty() => {
                let i = g.below(live.len() as u32) as usize;
                let (_, hd) = live.remove(i);
                assert!(h.remove(hd), "live handle at step {step}");
                dead.push(hd);
            }
            7 if !dead.is_empty() => {
                let hd = dead[g.below(dead.len() as u32) as usize];
                assert!(!h.remove(hd), "a stale handle removed something at step {step}");
            }
            _ => {
                let best = live.iter().enumerate().min_by(|a, b| by_key(&a.1 .0, &b.1 .0).then(a.0.cmp(&b.0)));
                let want = best.map(|(i, _)| i);
                let got = h.pop_with_handle();
                match want {
                    None => assert_eq!(got, None, "pop of an empty heap at step {step}"),
                    Some(i) => {
                        let (item, hd) = live.remove(i);
                        assert_eq!(got, Some((hd, item)), "pop at step {step}");
                        dead.push(hd);
                    }
                }
            }
        }
        assert_eq!(h.len(), live.len(), "len at step {step}");
    }
}

#[test]
fn stale_entries_stay_bounded() {
    // WHY: lazy deletion trades memory for speed; compaction keeps the trade
    //      bounded. After 10,000 pushes and 9,900 removes the array holds at
    //      most 2 x live + 32 entries, and every survivor still pops in order.
    // KIND: boundary
    // CATCHES: s06, s08
    // CHAPTER: ds.06 section 2.4
    let mut h = LazyHeap::new(|a: &u32, b: &u32| a.cmp(b));
    let hs: Vec<Handle> = (0..10_000u32).rev().map(|x| h.push(x)).collect();
    for (i, hd) in hs.iter().enumerate() {
        if i % 100 != 0 {
            assert!(h.remove(*hd));
        }
    }
    assert_eq!(h.len(), 100);
    assert!(h.len() + h.stale_len() <= 2 * h.len() + 32, "{} stale entries for {} live", h.stale_len(), h.len());
    let out: Vec<u32> = std::iter::from_fn(|| h.pop()).collect();
    let want: Vec<u32> = (0..100u32).map(|i| 9_999 - 100 * (99 - i)).collect();
    assert_eq!(out, want);
}

#[test]
fn clear_reuses_the_heap_and_stales_every_handle() {
    // WHY: L1.5 merges one pre-token after another from ONE LazyHeap,
    //      cleared between pieces so it allocates once per text. A handle
    //      from before clear must never remove an item pushed after it, even
    //      though the slot is reused.
    // KIND: boundary
    // CATCHES: s09
    // CHAPTER: ds.06 section 2.4
    let mut h = LazyHeap::new(|a: &u32, b: &u32| a.cmp(b));
    let old: Vec<Handle> = (0..4u32).map(|x| h.push(x)).collect();
    h.clear();
    assert_eq!((h.len(), h.stale_len(), h.pop()), (0, 0, None));
    let new: Vec<Handle> = (10..14u32).map(|x| h.push(x)).collect();
    for hd in &old {
        assert!(!h.remove(*hd), "a handle from before clear removed an item");
        assert!(!h.contains(*hd));
    }
    assert_eq!(h.len(), 4);
    assert!(h.remove(new[2]));
    let out: Vec<u32> = std::iter::from_fn(|| h.pop()).collect();
    assert_eq!(out, vec![10, 11, 13]);
}
