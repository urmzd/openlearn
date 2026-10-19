# Binary heap with lazy deletion

## Overview

| | |
|---|---|
| **Module** | `ds.06` · build · Rust · Pass 3 · 2 to 3 h reading |
| **Solution** | [`rust/crates/tl-ds/src/heap.rs`](../../../../course/solutions/rust/crates/tl-ds/src/heap.rs): the shared free functions `before`, `sift_up`, `sift_down`, `pop_root`, `heapify`; `Heap<T, F>` (`new`, `new_min`, `with_capacity`, `from_vec`, `push`, `pop`, `peek`, `len`, `is_empty`, `clear`, `array`, `into_sorted_vec`) and `LazyHeap<T, F>` with `Handle` (`push`, `remove`, `contains`, `pop`, `pop_with_handle`, `peek`, `len`, `is_empty`, `stale_len`, `compact`, `clear`) · model tests: [`ds06_heap.rs`](../../../../course/solutions/learner-tests/ds.06/rust/crates/tl-ds/tests/ds06_heap.rs) |
| **Contract** | the interface in section 4 (there is no separate contract file; the public API of `heap.rs` is the interface) |
| **Needs** | reading: `lang.04` the [Rust primer](../../programming-languages/02-language-and-tool-primers/04-rust.md) (closures, generics) · the crate root is `ds.05`'s |
| **Used by** | `L1.5` runs every BPE merge from a `LazyHeap` keyed by (rank, position) · later: `L10.2` keeps the engine's waiting queue in one; `ds.04` (top-k in C) builds on this chapter |
| **Optional depth** | Cormen et al., *Introduction to Algorithms*, chapter 6 (heapsort and priority queues); Williams, *Algorithm 232: Heapsort* (1964); Floyd, *Algorithm 245: Treesort 3* (1964, the O(n) build) |

## Key Takeaways

- A binary heap is a complete binary tree stored in an array: the children of index $i$ are $2i + 1$ and $2i + 2$, so push and pop are $O(\log n)$ swaps along one path.
- Ties broken by push order make the heap stable: equal items come out first in, first out, whatever the array looks like.
- Lazy deletion removes any item in $O(1)$: bump its slot's generation and let `pop` skip the stale entry when it reaches the root; the generation stops an old handle from removing a newer item.
- Stale entries cost memory, so the heap compacts when they outnumber live ones; pop order never changes.
- Building a heap from $n$ items costs $O(n)$, not $O(n \log n)$: sift down every parent from the last one to the root.

## How to read this chapter

Sections 1 to 3 build the heap, its stable order, and lazy deletion from first principles, and trace pushes, a pop, and a lazy removal by hand. Section 4 fixes the interface and lists what a correct solution must handle. Section 5 walks through [`heap.rs`](../../../../course/solutions/rust/crates/tl-ds/src/heap.rs). The solution writes `before`, `sift_up`, `sift_down`, `pop_root`, and `heapify` once, as free functions over a slice of `Node<T>`, and both `Heap` and `LazyHeap` call them; read those first. Try to predict each step before reading it.

---

## 1. Why now

The Rust tokenizer (`L1.5`) has to apply BPE merges in rank order: in each pre-token, repeatedly merge the adjacent pair with the lowest rank, the leftmost on a tie. Rescanning every pair after every merge, as the definition says and as a straightforward Python version does, costs $O(n^2)$ per pre-token; a 300-byte run of one letter (they occur in real text) makes it visible. A priority queue gives the next pair in $O(\log n)$. But each merge also **destroys** the two pairs that touched the merged symbols, and a binary heap cannot delete from the middle. This module builds the heap, then the lazy deletion that makes the merge queue work. The engine's scheduler (`L10.2`) needs the same thing: a queue by priority from which a cancelled request disappears.

## 2. Principles

### 2.1 The heap property

| Symbol | Meaning | Type |
|---|---|---|
| $n$ | number of items in the array | `usize` |
| $i$ | an array index, $0 \le i < n$ | `usize` |
| $\text{parent}(i) = \lfloor (i - 1)/2 \rfloor$, $\text{left}(i) = 2i + 1$, $\text{right}(i) = 2i + 2$ | the tree links, computed, never stored | `usize` |
| $a \prec b$ | $a$ comes out before $b$: the comparator says `Less`, or `Equal` and $a$ was pushed first | relation |
| $g_s$ | the generation of slot $s$ in a lazy heap | `u32` |

A **complete binary tree** fills each level left to right, so it fits in an array with no gaps and no pointers. The **heap property**: no child comes out before its parent ($\text{parent}(i) \preceq i$). The root, index 0, is then the next item out. The height is $\lfloor \log_2 n \rfloor$.

**Push** appends at index $n$ and **sifts up**: while the new item comes out before its parent, swap them. **Pop** takes the root, moves the last item to index 0, and **sifts down**: while one of its children comes out before it, swap it with the child that comes out **first** (swapping with the other would put the larger child above the smaller one). Each touches one root-to-leaf path: $O(\log n)$.

### 2.2 Building in O(n)

Pushing $n$ items one by one costs $O(n \log n)$. Floyd's construction treats the array as a tree whose leaves (the last $\lceil n/2 \rceil$ indices) are already heaps, then sifts down each parent from index $\lfloor n/2 \rfloor - 1$ back to 0. A node at height $h$ sifts at most $h$ levels and there are about $n / 2^{h+1}$ of them, so the total is $\sum_h h\, n / 2^{h+1} \le n$.

### 2.3 Order and stability

The order is a comparator `cmp(a, b) -> Ordering`, so one heap type serves a min-heap (`Heap::new_min`), a max-heap (`|a, b| b.cmp(a)`), and the merge queue (`(rank, position)` as a tuple). A binary heap is not stable by itself: two equal items can come out in either order depending on the array's history. Each pushed item gets a sequence number, and ties compare by it, so equal items leave first in, first out. Determinism (P11) needs this: the engine's queue must not reorder two requests of the same priority differently on two runs.

### 2.4 Lazy deletion with generation counters

To remove an item from the middle, `LazyHeap::push` returns a `Handle { slot, gen }`. The heap keeps one generation counter $g_s$ per slot and a free list of slots. `remove(h)` checks that $g_{h.slot} = h.gen$ (the item is still in the heap), then increments $g_{h.slot}$ and frees the slot, in $O(1)$; the array entry stays where it is, now **stale**. `pop` and `peek` first discard stale entries from the root ("purge"), then pop as usual. A reused slot gets the new generation, so a stale handle never matches again: without the generation, an old handle would remove whatever item reused its slot.

Stale entries take memory and slow the sifts, so when the array holds more than $2 \cdot \text{live} + 32$ entries, `remove` **compacts**: drop every stale entry and rebuild the heap in $O(n)$. Pop order depends only on the comparator and the push order, so compaction never changes it. `clear` empties the heap but keeps the arrays, bumping every generation so that no handle from before survives: `L1.5` clears one heap between pre-tokens and allocates once per text.

## 3. Worked example by hand

**Pushes.** Push 5, 3, 8, 1, 9, 2 into an empty min-heap and write the array after each push:

| Push | Sift-up path | Array |
|---|---|---|
| 5 | root | [5] |
| 3 | index 1, parent 0 holds 5: swap | [3, 5] |
| 8 | index 2, parent 0 holds 3: stay | [3, 5, 8] |
| 1 | index 3, parent 1 holds 5: swap; index 1, parent 0 holds 3: swap | [1, 3, 8, 5] |
| 9 | index 4, parent 1 holds 3: stay | [1, 3, 8, 5, 9] |
| 2 | index 5, parent 2 holds 8: swap; index 2, parent 0 holds 1: stay | [1, 3, 2, 5, 9, 8] |

**Pop.** Take 1 from the root, move the last item (8) to index 0: [8, 3, 2, 5, 9]. Its children are 3 and 2; the one that comes out first is 2 (index 2), and 2 comes out before 8: swap, [2, 3, 8, 5, 9]. Index 2 has no children: done.

**Lazy removal.** Push 5, 3, 8 into a `LazyHeap`: handles $h_5 = (0, 0)$, $h_3 = (1, 0)$, $h_8 = (2, 0)$ as (slot, generation). `remove(h_3)`: $g_1$ becomes 1 and slot 1 is free; the array still holds 3 at the root, so `stale_len` is 1. `peek` finds the root's handle $(1, 0)$ stale ($g_1 = 1$), pops it, and returns 5. Push 4: it reuses slot 1 with handle $(1, 1)$. `remove(h_3)` now compares $h_3.gen = 0$ with $g_1 = 1$: stale, returns false, and 4 stays.

## 4. The interface

```rust
// rust/crates/tl-ds/src/heap.rs
pub struct Heap<T, F: Fn(&T, &T) -> Ordering> { /* data: Vec<Node<T>>, seq, cmp */ }
impl<T: Ord> Heap<T, fn(&T, &T) -> Ordering> { pub fn new_min() -> Self; }
impl<T, F: Fn(&T, &T) -> Ordering> Heap<T, F> {
    pub fn new(cmp: F) -> Self;  pub fn with_capacity(n: usize, cmp: F) -> Self;
    pub fn from_vec(items: Vec<T>, cmp: F) -> Self;          // O(n); equal items keep their order
    pub fn push(&mut self, item: T);  pub fn pop(&mut self) -> Option<T>;  pub fn peek(&self) -> Option<&T>;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;  pub fn clear(&mut self);
    pub fn array(&self) -> Vec<&T>;                           // array order, root first
    pub fn into_sorted_vec(self) -> Vec<T>;                   // pop order
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub struct Handle { /* slot: u32, gen: u32 */ }
pub struct LazyHeap<T, F: Fn(&T, &T) -> Ordering> { /* data, gens, free, live, seq, cmp */ }
impl<T, F: Fn(&T, &T) -> Ordering> LazyHeap<T, F> {
    pub fn new(cmp: F) -> Self;
    pub fn push(&mut self, item: T) -> Handle;
    pub fn remove(&mut self, h: Handle) -> bool;              // O(1); false if already popped or removed
    pub fn contains(&self, h: Handle) -> bool;
    pub fn pop(&mut self) -> Option<T>;  pub fn pop_with_handle(&mut self) -> Option<(Handle, T)>;
    pub fn peek(&mut self) -> Option<&T>;                     // &mut: discards stale entries above it
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;  pub fn stale_len(&self) -> usize;
    pub fn compact(&mut self);  pub fn clear(&mut self);
}
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example's sift-up | unit | the section 3 push table, array after every push | the layout is pinned exactly |
| Pop sifts down through the smaller child | unit | pop gives [2, 3, 8, 5, 9], then [3, 5, 8, 9] | the swap goes to the child that comes out first |
| Equal items come out in push order | unit | seven items with three keys leave stably | `L10.2` serves equal priorities in arrival order |
| The hand example's lazy removal | unit | the section 3 handles: stale count, purge on peek, slot reuse, stale handles refused | `L1.5` removes destroyed pairs by handle |
| Empty and single item | boundary | empty pop and peek, one item, `clear`, an emptied lazy heap | the queue of a short piece |
| The comparator decides the order | unit | a reversed comparator is a max-heap | one type for every queue |
| `from_vec` heapifies in place | unit | O(n) build pops sorted; equal items keep input order; push after build | building a queue from a batch |
| A BPE merge queue pops lowest rank, then leftmost | unit | (rank, position) keys with two removals by handle | the exact use in `L1.5` |
| Agrees with a sorted Vec | differential | 20,000 random pushes and pops agree with a stably sorted Vec | the heap is a priority queue |
| Lazy heap agrees with a model | differential | 20,000 pushes, removes by live and stale handles, and pops agree with a model of the live items | lazy deletion never leaks a removed item |
| Stale entries stay bounded | boundary | after 10,000 pushes and 9,900 removes at most 2 x live + 32 entries; survivors in order | memory stays proportional to the live queue |
| `clear` reuses the heap and stales every handle | boundary | after `clear`, no old handle removes or contains a new item | `L1.5` clears one heap between pieces |

A model test suite, written test-first (pops in order, array after pushes, ties are FIFO, `from_vec` sorts, removed items never pop, many removes stay bounded), is in [`ds06_heap.rs`](../../../../course/solutions/learner-tests/ds.06/rust/crates/tl-ds/tests/ds06_heap.rs). The tie-breaking case is the one most worth having: a heap without it passes almost every other check.

## 5. The solution, step by step

### 5.1 From first principles

1. **A priority queue needs the next item in O(1) and updates in $O(\log n)$.** A complete binary tree has height $\lfloor \log_2 n \rfloor$ and fits in an array with computed links, so the solution stores nothing but the array.
2. **"Comes out before" must be a strict total order, or output depends on array history.** The comparator alone is not (ties), so every entry carries a push sequence number and ties compare by it. One function, `before`, encodes this, and every sift calls it.
3. **Push and pop each break the heap property at one position,** so each needs one repair: sift up from the new leaf, sift down from the root. Sift down must swap with the child that comes out first, or the other child ends up above a smaller one.
4. **Building from $n$ items should not cost $n$ pushes.** Leaves are already heaps; sifting down each parent from the last one back to the root repairs everything in $O(n)$ (section 2.2).
5. **Removing from the middle cannot be done in place cheaply** without knowing each item's array index, which every swap would have to update. Instead, removal marks the item dead in a side table and leaves its array entry; the root is the only place a dead entry can do harm, so `pop` and `peek` drop dead roots first.
6. **"Dead" must be per item, not per slot,** because slots are reused. A generation counter per slot, copied into the handle at push and bumped at retirement, makes an old handle mismatch forever. In the worked example, $h_3 = (1, 0)$ stops matching once $g_1 = 1$, so it cannot remove 4 at $(1, 1)$.
7. **Dead entries must not grow without bound,** so `remove` compacts when the array exceeds $2 \cdot \text{live} + 32$: filter, then heapify. Because order is "comparator, then sequence number", rebuilding cannot change what pops next.

### 5.2 The code, annotated

The shared core. `before` is the strict total order; the two sifts and `pop_root` are written once over `&mut [Node<T>]` and used by both heaps.

```rust
struct Node<T> {
    seq: u64, // push order: the tie-breaker that makes the heap stable
    item: T,
}

/// Does `a` come out before `b`? The comparator first, then push order.
fn before<T, F: Fn(&T, &T) -> Ordering>(cmp: &F, a: &Node<T>, b: &Node<T>) -> bool {
    match cmp(&a.item, &b.item) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => a.seq < b.seq, // earlier push wins: FIFO among equals
    }
}

fn sift_up<T, F: Fn(&T, &T) -> Ordering>(data: &mut [Node<T>], cmp: &F, mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if !before(cmp, &data[i], &data[parent]) {
            break;                    // parent comes out first: property holds
        }
        data.swap(i, parent);
        i = parent;
    }
}

fn sift_down<T, F: Fn(&T, &T) -> Ordering>(data: &mut [Node<T>], cmp: &F, mut i: usize) {
    let n = data.len();
    loop {
        let l = 2 * i + 1;
        if l >= n {
            break;                    // a leaf
        }
        let r = l + 1;
        // the child that comes out first: the only one that may become the parent
        let first = if r < n && before(cmp, &data[r], &data[l]) { r } else { l };
        if !before(cmp, &data[first], &data[i]) {
            break;
        }
        data.swap(i, first);
        i = first;
    }
}

fn pop_root<T, F: Fn(&T, &T) -> Ordering>(data: &mut Vec<Node<T>>, cmp: &F) -> Option<Node<T>> {
    if data.is_empty() {
        return None;
    }
    let last = data.len() - 1;
    data.swap(0, last);               // the root goes to the end, where Vec::pop takes it
    let out = data.pop();
    if !data.is_empty() {
        sift_down(data, cmp, 0);      // the former last leaf sinks to its place
    }
    out
}

/// Floyd: sift down every parent, last first. O(n).
fn heapify<T, F: Fn(&T, &T) -> Ordering>(data: &mut [Node<T>], cmp: &F) {
    for i in (0..data.len() / 2).rev() {  // includes index 0: the root is sifted too
        sift_down(data, cmp, i);
    }
}
```

`Heap` is a thin wrapper. `from_vec` numbers items by input position before heapifying, so equal items keep input order, and continues `seq` from there so later pushes come after them.

```rust
pub fn from_vec(items: Vec<T>, cmp: F) -> Self {
    let mut data: Vec<Node<T>> = items
        .into_iter()
        .enumerate()
        .map(|(i, item)| Node { seq: i as u64, item })
        .collect();
    heapify(&mut data, &cmp);
    let seq = data.len() as u64;      // next push is later than every input item
    Heap { data, seq, cmp }
}

pub fn push(&mut self, item: T) {
    let seq = self.seq;
    self.seq += 1;
    self.data.push(Node { seq, item });
    let last = self.data.len() - 1;
    sift_up(&mut self.data, &self.cmp, last);
}
```

`LazyHeap` state. Each array entry carries the handle it was pushed with; `gens` says which handles are still alive.

```rust
pub struct Handle { slot: u32, gen: u32 }

pub struct LazyHeap<T, F: Fn(&T, &T) -> Ordering> {
    data: Vec<Node<LazyItem<T>>>, // heap array; may hold stale entries
    gens: Vec<u32>,               // the current generation of each slot
    free: Vec<u32>,               // slots with no live item
    live: usize,                  // live items: data.len() - live are stale
    seq: u64,
    cmp: F,
}

fn is_live(&self, h: Handle) -> bool {
    (h.slot as usize) < self.gens.len() && self.gens[h.slot as usize] == h.gen
}

/// Ends the life of `h`: its slot's generation moves on and the slot is free.
fn retire(&mut self, h: Handle) {
    let g = &mut self.gens[h.slot as usize];
    *g = g.wrapping_add(1);           // every outstanding copy of h is now stale
    self.free.push(h.slot);
    self.live -= 1;
}
```

Push takes a free slot (or a new one) and stamps the handle with its current generation. The comparator is wrapped so the shared sifts compare the user's items, not the handles.

```rust
pub fn push(&mut self, item: T) -> Handle {
    let slot = match self.free.pop() {
        Some(s) => s,                                     // reuse: gens[s] already bumped
        None => { self.gens.push(0); (self.gens.len() - 1) as u32 }
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
```

Remove is O(1) apart from the occasional compaction; pop purges dead roots before popping. `peek` takes `&mut self` for the same purge.

```rust
pub fn remove(&mut self, h: Handle) -> bool {
    if !self.is_live(h) {
        return false;                 // already popped or removed: touch nothing
    }
    self.retire(h);                   // the array entry stays, now stale
    if self.data.len() > 2 * self.live + 32 {
        self.compact();               // amortised: at least live + 32 removes between compactions
    }
    true
}

fn purge_top(&mut self) {
    while let Some(top) = self.data.first() {
        if self.is_live(top.item.handle) {
            return;
        }
        let cmp = &self.cmp;
        pop_root(&mut self.data, &|a: &LazyItem<T>, b: &LazyItem<T>| cmp(&a.item, &b.item));
    }
}

pub fn pop_with_handle(&mut self) -> Option<(Handle, T)> {
    self.purge_top();                 // the root is now live, or the heap is empty
    let cmp = &self.cmp;
    let node = pop_root(&mut self.data, &|a: &LazyItem<T>, b: &LazyItem<T>| cmp(&a.item, &b.item))?;
    self.retire(node.item.handle);    // a popped item's handle must stop matching too
    Some((node.item.handle, node.item.item))
}
```

Compaction and clear. Compaction must rebuild the heap after filtering, because `retain` keeps array order, not heap order. `clear` keeps the allocations and bumps every generation so no handle from before survives.

```rust
pub fn compact(&mut self) {
    let gens = &self.gens;
    self.data.retain(|n| gens[n.item.handle.slot as usize] == n.item.handle.gen);
    let cmp = &self.cmp;
    heapify(&mut self.data, &|a: &LazyItem<T>, b: &LazyItem<T>| cmp(&a.item, &b.item));
}

pub fn clear(&mut self) {
    self.data.clear();
    for g in self.gens.iter_mut() {
        *g = g.wrapping_add(1);       // stale every handle issued so far
    }
    self.free.clear();
    self.free.extend((0..self.gens.len() as u32).rev()); // all slots free; slot 0 is reused first
    self.live = 0;
}
```

Full solution: [`heap.rs`](../../../../course/solutions/rust/crates/tl-ds/src/heap.rs).

### 5.3 Why this solution

- **Cost.** `push` and `pop` are $O(\log n)$ in the array length; `remove`, `contains`, and `len` are $O(1)$ plus compaction, which costs $O(n)$ but runs at most once per $\text{live} + 32$ removals, so it is amortised $O(1)$ per remove. Because the array stays within $2 \cdot \text{live} + 32$, $\log n$ is within a constant of $\log \text{live}$. Space: one `Node` per array entry plus 4 bytes of generation per slot.
- **Against an indexed heap with `decrease_key`/delete in place.** It needs a position map updated on every swap, which doubles the bookkeeping of every sift and is easy to get subtly wrong; lazy deletion keeps the sifts unchanged.
- **Against a balanced tree (`BTreeSet<(key, seq)>`).** It removes from the middle in $O(\log n)$ with no stale entries, but allocates nodes and has worse constants than an array heap for the tokenizer's millions of small queues.
- **Against `std::collections::BinaryHeap`.** It is a max-heap over `Ord`, not stable, and has no removal by handle.
- **Tradeoffs accepted.** Stale entries cost memory up to about twice the live size, and `peek` needs `&mut self`. Generations are `u32` and wrap, so a handle could in principle match again after $2^{32}$ reuses of one slot, far beyond any piece or queue here.

### 5.4 Explain it back

**Why does sift down swap with the child that comes out first, and not just the left child?**

<details><summary>Answer</summary>

After the swap, the promoted child becomes the parent of the other child. If it were the one that comes out later, the heap property would be violated at the other child immediately. In the worked pop, 8 must swap with 2 (the earlier child), not with 3; swapping with 3 would leave 3 above 2.

</details>

**What breaks if `retire` frees a slot without bumping its generation?**

<details><summary>Answer</summary>

The next push reuses the slot and stamps the same generation, so an old handle (for an item already popped or removed) matches the new item. In `L1.5` a merge removes the handles of the two destroyed pairs; with this bug one of those removes could delete a live pair that happens to reuse the slot, and the tokenizer would skip a merge.

</details>

**Why is the sequence number needed at all, if equal items are "equal"?**

<details><summary>Answer</summary>

Equal under the comparator does not mean indistinguishable: two requests with the same priority are different requests. Without a tie-breaker, which one pops first depends on the array's history (which swaps happened), so two runs with different interleavings serve them in different orders. The sequence number makes the order a function of the push order only, which also guarantees that compaction cannot change it.

</details>

**Why must `compact` call `heapify` after `retain`?**

<details><summary>Answer</summary>

`retain` removes entries and shifts the rest down, which changes every index after the first removed one. Parent and child relationships are by index, so the result is in general not a heap; pops would return items in roughly array order. Heapify restores the property in $O(n)$.

</details>

**Why does `pop_with_handle` retire the handle of the popped item?**

<details><summary>Answer</summary>

Once an item has left the heap, its handle must stop matching: `contains` must say false and `remove` must refuse it, and its slot must be free for reuse. Retiring bumps the generation, frees the slot, and decrements `live` in one place, the same as for a removal.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Sifting up when the child does NOT come out before its parent | the root is not the minimum; everything pops out of order | `sift_up` swaps only when `before(child, parent)` |
| Breaking ties by reversed push order | equal priorities leave last in, first out | `before` uses `a.seq < b.seq` on `Equal` |
| Freeing a slot without bumping its generation | an old handle removes the new item in its slot: a BPE merge deletes a live pair | `retire` bumps the generation and frees the slot together |
| Popping without purging stale entries first | removed items come back out; the live count underflows | `pop_with_handle` and `peek` call `purge_top` first |
| Sifting down into the left child always | a larger child rises above a smaller one | `sift_down` picks `first` of the two children |
| Never compacting | the array grows with every removal: the merge queue of a long word holds every pair it ever saw | `remove` compacts above `2 * live + 32` entries |
| Starting the heapify loop at index 1 | the root is never sifted; `from_vec` returns a non-heap | `heapify` runs `(0..len / 2).rev()`, root included |
| Compacting without rebuilding the heap | after a compaction pops come out in array order | `compact` heapifies after `retain` |
| Clearing without bumping generations | a handle from the previous piece removes a pair of the next | `clear` bumps every generation |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.04` | closures as comparators, generic structs, `Option` |
| Forward | `L1.5` | `LazyHeap<(u32, usize), _>` holds every mergeable pair of a pre-token; a merge pops one pair, removes two by handle, and pushes up to two |
| Forward | `L10.2` | the continuous-batching scheduler's waiting queue by (priority, arrival), with cancellation by handle |
| Forward | `ds.04` | the C top-k keeps a size-k min-heap over logits with the same sift operations |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Heap` | `std::collections::BinaryHeap` | a max-heap over `Ord` with `into_sorted_vec`, `peek_mut`, and sift with a "hole" instead of swaps | `library/alloc/src/collections/binary_heap/mod.rs` |
| `LazyHeap` for BPE | [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) word merging | the same lazy queue: stale merges are skipped when popped | `tokenizers/src/models/bpe/word.rs` (`merge_all`) |
| lazy deletion | indexed (addressable) heaps, pairing heaps | `decrease_key` in place by keeping each item's array index | Dijkstra's algorithm in any graph library |
| the waiting queue | [vLLM's scheduler](https://github.com/vllm-project/vllm) | priority plus preemption policies over waiting and running queues | `vllm/core/scheduler.py` |
