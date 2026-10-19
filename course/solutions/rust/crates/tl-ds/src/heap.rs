//! Binary heaps: a stable heap and a heap with lazy deletion (ds.06).
//!
//! A binary heap is a complete binary tree stored in an array: the children
//! of index `i` are `2i + 1` and `2i + 2`, its parent is `(i - 1) / 2`. The
//! heap property says no child comes out before its parent, so the root is
//! the next item out. Push appends and sifts up; pop moves the last item to
//! the root and sifts down. Both are O(log n).
//!
//! The order is a comparator `cmp(a, b)`: `Less` means `a` comes out first.
//! Ties are broken by push order (an internal sequence number), so items
//! that compare `Equal` come out first-in, first-out: the heap is **stable**
//! and its output never depends on how the array happened to be arranged.
//!
//! [`LazyHeap`] adds O(1) removal of any pushed item through a [`Handle`]:
//! removal only bumps the generation counter of the item's slot; the stale
//! array entry stays until it reaches the root, where `pop` and `peek` skip
//! it. When stale entries outnumber live ones the array is rebuilt.
//!
//! Callers: `tl-tok` (L1.5) runs BPE merges from a `LazyHeap` keyed by
//! (merge rank, position); `L10.2` keeps the engine's waiting queue in one.
//! Chapter: content/courses/algorithms/16-systems-data-structures/06-binary-heap-lazy-deletion.md

use std::cmp::Ordering;

/// One array entry: the item and its push sequence number.
#[derive(Clone, Debug)]
struct Node<T> {
    seq: u64,
    item: T,
}

/// Does `a` come out before `b`? The comparator first, then push order.
#[inline]
fn before<T, F: Fn(&T, &T) -> Ordering>(cmp: &F, a: &Node<T>, b: &Node<T>) -> bool {
    match cmp(&a.item, &b.item) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.seq < b.seq,
    }
}

/// Restores the heap property upward from `i`.
fn sift_up<T, F: Fn(&T, &T) -> Ordering>(data: &mut [Node<T>], cmp: &F, mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if !before(cmp, &data[i], &data[parent]) {
            break;
        }
        data.swap(i, parent);
        i = parent;
    }
}

/// Restores the heap property downward from `i`: swap with the child that
/// comes out first while that child comes out before `i`.
fn sift_down<T, F: Fn(&T, &T) -> Ordering>(data: &mut [Node<T>], cmp: &F, mut i: usize) {
    let n = data.len();
    loop {
        let l = 2 * i + 1;
        if l >= n {
            break;
        }
        let r = l + 1;
        let first = if r < n && before(cmp, &data[r], &data[l]) { r } else { l };
        if !before(cmp, &data[first], &data[i]) {
            break;
        }
        data.swap(i, first);
        i = first;
    }
}

/// Removes the root: the last entry moves to the root and sifts down.
fn pop_root<T, F: Fn(&T, &T) -> Ordering>(data: &mut Vec<Node<T>>, cmp: &F) -> Option<Node<T>> {
    if data.is_empty() {
        return None;
    }
    let last = data.len() - 1;
    data.swap(0, last);
    let out = data.pop();
    if !data.is_empty() {
        sift_down(data, cmp, 0);
    }
    out
}

/// Makes any array a heap in O(n): sift down every parent, last first.
fn heapify<T, F: Fn(&T, &T) -> Ordering>(data: &mut [Node<T>], cmp: &F) {
    for i in (0..data.len() / 2).rev() {
        sift_down(data, cmp, i);
    }
}

/// A stable binary heap ordered by `cmp`: `pop` returns the item that
/// compares least, the earliest pushed among equals.
#[derive(Clone)]
pub struct Heap<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    data: Vec<Node<T>>,
    seq: u64,
    cmp: F,
}

impl<T: Ord> Heap<T, fn(&T, &T) -> Ordering> {
    /// A min-heap over `T`'s own order.
    pub fn new_min() -> Self {
        Heap::new(<T as Ord>::cmp as fn(&T, &T) -> Ordering)
    }
}

impl<T, F> Heap<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    pub fn new(cmp: F) -> Self {
        Heap { data: Vec::new(), seq: 0, cmp }
    }

    pub fn with_capacity(n: usize, cmp: F) -> Self {
        Heap { data: Vec::with_capacity(n), seq: 0, cmp }
    }

    /// A heap holding `items`, built in O(n); equal items keep their order
    /// in `items`.
    pub fn from_vec(items: Vec<T>, cmp: F) -> Self {
        let mut data: Vec<Node<T>> = items
            .into_iter()
            .enumerate()
            .map(|(i, item)| Node { seq: i as u64, item })
            .collect();
        heapify(&mut data, &cmp);
        let seq = data.len() as u64;
        Heap { data, seq, cmp }
    }

    pub fn push(&mut self, item: T) {
        let seq = self.seq;
        self.seq += 1;
        self.data.push(Node { seq, item });
        let last = self.data.len() - 1;
        sift_up(&mut self.data, &self.cmp, last);
    }

    pub fn pop(&mut self) -> Option<T> {
        pop_root(&mut self.data, &self.cmp).map(|n| n.item)
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.first().map(|n| &n.item)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// The items in array order: index 0 is the root, the children of `i`
    /// are `2i + 1` and `2i + 2`. For tests and the chapter's diagrams.
    pub fn array(&self) -> Vec<&T> {
        self.data.iter().map(|n| &n.item).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Every item in pop order (consumes the heap).
    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.data.len());
        while let Some(x) = self.pop() {
            out.push(x);
        }
        out
    }
}

/// Names one item pushed into a [`LazyHeap`]: its slot and the slot's
/// generation at push time. A handle stays valid until its item is popped or
/// removed; after that it never matches again, even when the slot is reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Handle {
    slot: u32,
    gen: u32,
}

/// An array entry of a lazy heap: the node plus the handle it was pushed as.
#[derive(Clone, Debug)]
struct LazyItem<T> {
    handle: Handle,
    item: T,
}

/// A stable binary heap with O(1) removal by [`Handle`] (lazy deletion with
/// generation counters).
pub struct LazyHeap<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    data: Vec<Node<LazyItem<T>>>,
    gens: Vec<u32>,  // the current generation of each slot
    free: Vec<u32>,  // slots with no live item
    live: usize,
    seq: u64,
    cmp: F,
}

impl<T, F> LazyHeap<T, F>
where
    F: Fn(&T, &T) -> Ordering,
{
    pub fn new(cmp: F) -> Self {
        LazyHeap { data: Vec::new(), gens: Vec::new(), free: Vec::new(), live: 0, seq: 0, cmp }
    }

    #[inline]
    fn is_live(&self, h: Handle) -> bool {
        (h.slot as usize) < self.gens.len() && self.gens[h.slot as usize] == h.gen
    }

    /// Ends the life of `h`: its slot's generation moves on and the slot is free.
    fn retire(&mut self, h: Handle) {
        let g = &mut self.gens[h.slot as usize];
        *g = g.wrapping_add(1);
        self.free.push(h.slot);
        self.live -= 1;
    }

    /// Pushes `item`; the handle removes it later.
    pub fn push(&mut self, item: T) -> Handle {
        let slot = match self.free.pop() {
            Some(s) => s,
            None => {
                self.gens.push(0);
                (self.gens.len() - 1) as u32
            }
        };
        let handle = Handle { slot, gen: self.gens[slot as usize] };
        let seq = self.seq;
        self.seq += 1;
        self.data.push(Node { seq, item: LazyItem { handle, item } });
        self.live += 1;
        let last = self.data.len() - 1;
        let cmp = &self.cmp;
        sift_up(&mut self.data, &|a: &LazyItem<T>, b: &LazyItem<T>| cmp(&a.item, &b.item), last);
        handle
    }

    /// Removes the item `h` names in O(1). False when it was already popped
    /// or removed (a stale handle never touches a newer item in its slot).
    pub fn remove(&mut self, h: Handle) -> bool {
        if !self.is_live(h) {
            return false;
        }
        self.retire(h);
        if self.data.len() > 2 * self.live + 32 {
            self.compact();
        }
        true
    }

    /// True while the item `h` names is in the heap.
    pub fn contains(&self, h: Handle) -> bool {
        self.is_live(h)
    }

    /// Drops stale entries from the root until a live one is there.
    fn purge_top(&mut self) {
        while let Some(top) = self.data.first() {
            if self.is_live(top.item.handle) {
                return;
            }
            let cmp = &self.cmp;
            pop_root(&mut self.data, &|a: &LazyItem<T>, b: &LazyItem<T>| cmp(&a.item, &b.item));
        }
    }

    /// The live item that comes out first, skipping removed ones.
    pub fn pop(&mut self) -> Option<T> {
        self.pop_with_handle().map(|(_, item)| item)
    }

    /// Like `pop`, with the handle the item was pushed as.
    pub fn pop_with_handle(&mut self) -> Option<(Handle, T)> {
        self.purge_top();
        let cmp = &self.cmp;
        let node = pop_root(&mut self.data, &|a: &LazyItem<T>, b: &LazyItem<T>| cmp(&a.item, &b.item))?;
        self.retire(node.item.handle);
        Some((node.item.handle, node.item.item))
    }

    /// The live item that would pop next. Takes `&mut self` because it
    /// discards the stale entries above it.
    pub fn peek(&mut self) -> Option<&T> {
        self.purge_top();
        self.data.first().map(|n| &n.item.item)
    }

    /// Live items (removed ones are not counted).
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Array entries whose item was removed but not yet discarded.
    pub fn stale_len(&self) -> usize {
        self.data.len() - self.live
    }

    /// Empties the heap and keeps its allocations, so a caller that runs many
    /// small jobs (one BPE piece after another) allocates once. Every handle
    /// issued so far becomes stale.
    pub fn clear(&mut self) {
        self.data.clear();
        for g in self.gens.iter_mut() {
            *g = g.wrapping_add(1);
        }
        self.free.clear();
        self.free.extend((0..self.gens.len() as u32).rev());
        self.live = 0;
    }

    /// Drops every stale entry and rebuilds the heap in O(n). Pop order is
    /// unchanged: it depends only on the comparator and push order.
    pub fn compact(&mut self) {
        let gens = &self.gens;
        self.data.retain(|n| gens[n.item.handle.slot as usize] == n.item.handle.gen);
        let cmp = &self.cmp;
        heapify(&mut self.data, &|a: &LazyItem<T>, b: &LazyItem<T>| cmp(&a.item, &b.item));
    }
}
