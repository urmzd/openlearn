<!-- ol:module ds.06 -->
# Binary heap with lazy deletion

## Overview

| | |
|---|---|
| **Module** | `ds.06` · build · Rust · Pass 3 · 4 to 6 h |
| **You build** | `rust/crates/tl-ds/src/heap.rs`: `Heap<T, F>` (`new`, `new_min`, `with_capacity`, `from_vec`, `push`, `pop`, `peek`, `len`, `is_empty`, `clear`, `array`, `into_sorted_vec`) and `LazyHeap<T, F>` with `Handle` (`push`, `remove`, `contains`, `pop`, `pop_with_handle`, `peek`, `len`, `is_empty`, `stale_len`, `compact`, `clear`) |
| **Contract** | the interface in section 4 (no Rust trait file yet; the tests pin it) |
| **Tests** | `course/tests/rust/ds_06.rs`, 12 tests (what they check: section 4) · your own tests in `rust/crates/tl-ds/tests/ds06_heap.rs`, rung R3 (tests first, `ol tdd red`), graded by mutation (threshold 0.70, mutant `s02` required) |
| **Needs** | reading: `lang.04` the [Rust primer](../../software-craftsmanship/12-language-and-tool-primers/04-rust.md) (closures, generics) · the crate root is `ds.05`'s |
| **Used by** | `L1.5` runs every BPE merge from a `LazyHeap` keyed by (rank, position) · later: `L10.2` keeps the engine's waiting queue in one; `ds.04` (top-k in C) builds on this chapter |
| **Milestone** | `MS-L1` (the Rust tokenizer's merges come out of your heap) |
| **Optional depth** | Cormen et al., *Introduction to Algorithms*, chapter 6 (heapsort and priority queues); Williams, *Algorithm 232: Heapsort* (1964); Floyd, *Algorithm 245: Treesort 3* (1964, the O(n) build) |

## Key Takeaways

- A binary heap is a complete binary tree stored in an array: the children of index $i$ are $2i + 1$ and $2i + 2$, so push and pop are $O(\log n)$ swaps along one path (`hand_example_sift`, `pop_sifts_down_through_the_smaller_child`).
- Ties broken by push order make the heap stable: equal items come out first in, first out, whatever the array looks like (`equal_items_come_out_in_push_order`).
- Lazy deletion removes any item in $O(1)$: bump its slot's generation and let `pop` skip the stale entry when it reaches the root; the generation stops an old handle from removing a newer item (`lazy_remove_hand_example`).
- Stale entries cost memory, so the heap compacts when they outnumber live ones; pop order never changes (`stale_entries_stay_bounded`, `lazy_differential_against_model`).
- Building a heap from $n$ items costs $O(n)$, not $O(n \log n)$: sift down every parent from the last one to the root (`from_vec_heapifies_in_place`).

## How to work this chapter

```bash
ol start ds.06              # stubs heap.rs (the tl-ds crate root comes with ds.05)
ol tdd red ds.06            # rung R3: your tests first, and they must fail on the stub
ol tests ds.06              # read the test catalog
ol check ds.06              # exit code is the verdict; then grades your tests by mutation
ol diff  ds.06              # after passing: your code against the reference
```

Write `before`, `sift_up`, `sift_down`, and `pop_root` as free functions over a slice of `Node<T>`: both heaps share them, and the hand example checks them through `Heap::array`.

---

## 1. Why now

Your Rust tokenizer (`L1.5`) has to apply BPE merges in rank order: in each pre-token, repeatedly merge the adjacent pair with the lowest rank, the leftmost on a tie. Rescanning every pair after every merge, as the definition says and as your Python may do, costs $O(n^2)$ per pre-token; a 300-byte run of one letter (they occur in real text) makes it visible. A priority queue gives the next pair in $O(\log n)$. But each merge also **destroys** the two pairs that touched the merged symbols, and a binary heap cannot delete from the middle. This module builds the heap, then the lazy deletion that makes the merge queue work. The engine's scheduler (`L10.2`) needs the same thing: a queue by priority from which a cancelled request disappears.

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

**Pop.** Take 1 from the root, move the last item (8) to index 0: [8, 3, 2, 5, 9]. Its children are 3 and 2; the one that comes out first is 2 (index 2), and 2 comes out before 8: swap, [2, 3, 8, 5, 9]. Index 2 has no children: done. These two tables are `hand_example_sift` and `pop_sifts_down_through_the_smaller_child`.

**Lazy removal.** Push 5, 3, 8 into a `LazyHeap`: handles $h_5 = (0, 0)$, $h_3 = (1, 0)$, $h_8 = (2, 0)$ as (slot, generation). `remove(h_3)`: $g_1$ becomes 1 and slot 1 is free; the array still holds 3 at the root, so `stale_len` is 1. `peek` finds the root's handle $(1, 0)$ stale ($g_1 = 1$), pops it, and returns 5. Push 4: it reuses slot 1 with handle $(1, 1)$. `remove(h_3)` now compares $h_3.gen = 0$ with $g_1 = 1$: stale, returns false, and 4 stays. This is `lazy_remove_hand_example`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_sift` | unit | the section 3 push table, array after every push | you and the tests agree on the layout |
| `pop_sifts_down_through_the_smaller_child` | unit | pop gives [2, 3, 8, 5, 9], then [3, 5, 8, 9] | the swap goes to the child that comes out first |
| `equal_items_come_out_in_push_order` | unit | seven items with three keys leave stably | `L10.2` serves equal priorities in arrival order |
| `lazy_remove_hand_example` | unit | the section 3 handles: stale count, purge on peek, slot reuse, stale handles refused | `L1.5` removes destroyed pairs by handle |
| `empty_and_single_item` | boundary | empty pop and peek, one item, `clear`, an emptied lazy heap | the queue of a short piece |
| `comparator_decides_the_order` | unit | a reversed comparator is a max-heap | one type for every queue |
| `from_vec_heapifies_in_place` | unit | O(n) build pops sorted; equal items keep input order; push after build | building a queue from a batch |
| `bpe_merge_queue_pops_lowest_rank_then_leftmost` | unit | (rank, position) keys with two removals by handle | the exact use in `L1.5` |
| `differential_against_sorted_vec` | differential | 20,000 random pushes and pops agree with a stably sorted Vec | the heap is a priority queue |
| `lazy_differential_against_model` | differential | 20,000 pushes, removes by live and stale handles, and pops agree with a model of the live items | lazy deletion never leaks a removed item |
| `stale_entries_stay_bounded` | boundary | after 10,000 pushes and 9,900 removes at most 2 x live + 32 entries; survivors in order | memory stays proportional to the live queue |
| `clear_reuses_the_heap_and_stales_every_handle` | boundary | after `clear`, no old handle removes or contains a new item | `L1.5` clears one heap between pieces |

**Your tests (rung R3).** Write `rust/crates/tl-ds/tests/ds06_heap.rs` first and run `ol tdd red ds.06` (they must fail on the stub), then write the code and `ol tdd green ds.06`. Suggested tests: `pops_in_order`, `array_after_pushes`, `ties_are_fifo`, `from_vec_sorts`, `removed_items_never_pop`, `many_removes_stay_bounded`. At least 70% of the planted bugs, and the tie-breaking bug `s02`, must make one of them fail.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Sifting up when the child does NOT come out before its parent | the root is not the minimum; everything pops out of order | `hand_example_sift`, `differential_against_sorted_vec` (mutant `s01`) |
| Breaking ties by reversed push order | equal priorities leave last in, first out | `equal_items_come_out_in_push_order` (mutant `s02`) |
| Freeing a slot without bumping its generation | an old handle removes the new item in its slot: a BPE merge deletes a live pair | `lazy_remove_hand_example` (mutant `s03`) |
| Popping without purging stale entries first | removed items come back out; the live count underflows | `lazy_remove_hand_example`, `bpe_merge_queue_pops_lowest_rank_then_leftmost` (mutant `s04`) |
| Sifting down into the left child always | a larger child rises above a smaller one | `pop_sifts_down_through_the_smaller_child` (mutant `s05`) |
| Never compacting | the array grows with every removal: the merge queue of a long word holds every pair it ever saw | `stale_entries_stay_bounded` (mutant `s06`) |
| Starting the heapify loop at index 1 | the root is never sifted; `from_vec` returns a non-heap | `from_vec_heapifies_in_place` (mutant `s07`) |
| Compacting without rebuilding the heap | after a compaction pops come out in array order | `lazy_differential_against_model`, `stale_entries_stay_bounded` (mutant `s08`) |
| Clearing without bumping generations | a handle from the previous piece removes a pair of the next | `clear_reuses_the_heap_and_stales_every_handle` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.04` | closures as comparators, generic structs, `Option` |
| Forward | `L1.5` | `LazyHeap<(u32, usize), _>` holds every mergeable pair of a pre-token; a merge pops one pair, removes two by handle, and pushes up to two |
| Forward | `L10.2` | the continuous-batching scheduler's waiting queue by (priority, arrival), with cancellation by handle |
| Forward | `ds.04` | the C top-k keeps a size-k min-heap over logits with the same sift operations |

If you skip this module, `ol check L1.5` stops with `needs ds.06: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Heap` | `std::collections::BinaryHeap` | a max-heap over `Ord` with `into_sorted_vec`, `peek_mut`, and sift with a "hole" instead of swaps | `library/alloc/src/collections/binary_heap/mod.rs` |
| `LazyHeap` for BPE | [Hugging Face tokenizers](https://github.com/huggingface/tokenizers) word merging | the same lazy queue: stale merges are skipped when popped | `tokenizers/src/models/bpe/word.rs` (`merge_all`) |
| lazy deletion | indexed (addressable) heaps, pairing heaps | `decrease_key` in place by keeping each item's array index | Dijkstra's algorithm in any graph library |
| the waiting queue | [vLLM's scheduler](https://github.com/vllm-project/vllm) | priority plus preemption policies over waiting and running queues | `vllm/core/scheduler.py` |
