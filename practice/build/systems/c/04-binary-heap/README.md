# C 04: Binary heap

**Concepts:** array-backed trees, `sift_up`/`sift_down`, heapsort
**Difficulty:** ⭐⭐

A min-heap where the entire data structure is one flat array and the tree shape
exists only in the index arithmetic. Then heapsort, which is the same machinery
run in place with no extra memory at all.

## The problem

[`heap.h`](heap.h) declares the interface, [`heap.c`](heap.c) is the solution,
and [`main.c`](main.c) is the test suite.

| Function | Does | Cost |
|----------|------|------|
| `heap_push(h, p, v)` | Insert | O(log n) |
| `heap_pop(h, &out)` | Remove the smallest | O(log n) |
| `heap_peek(h, &out)` | Look at the smallest | O(1) |
| `heap_build(h, items, n)` | Heapify an unordered array | O(n), not O(n log n) |
| `heap_sort(items, n)` | Sort ascending in place | O(n log n), no allocation |

Items are `HeapItem { int priority; int payload; }`. The payload rides along so
the tests can check that whole items move together. Everything is built from
two private helpers, `sift_up` and `sift_down`.

## Build and run

From this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c heap.c -o heap && ./heap
# ok  c/04-binary-heap  38041 checks passed
```

## What to notice

**There are no pointers, and that is the point.** With a 0-based array the
children of `i` are at `2i+1` and `2i+2` and the parent is at `(i-1)/2`. A
pointer-based tree of the same shape costs two pointers per node, an allocation
per insert, and a cache miss per level. The array version costs nothing per
element and walks contiguous memory, which is why heaps back priority queues
everywhere from schedulers to Dijkstra.

**`sift_down` must descend toward the smaller child.** Taking the left child
unconditionally is the classic bug: it can swap a large value past a smaller
sibling and leave the invariant broken one level down. It survives simple
ordering tests, because it only shows once the heap has been reshaped from both
ends; the random push/pop interleaving test (`test_random_interleaving`) is what
catches it.

**Building is O(n), and the intuition that says O(n log n) is wrong.** Pushing n
items one at a time is O(n log n). Floyd's method, sifting down from the last
internal node backwards, is O(n): most nodes are near the bottom of the tree and
sift down by almost nothing, and the sum of heights over all nodes is less than
n rather than n log n.

**A min-heap sorts descending, which is not what `heap_sort` promises.** In-place
heapsort parks each extracted element at the shrinking end, so a min-heap
produces a descending array. Getting ascending order needs a max-heap. Rather
than write `sift_down` twice with the comparison flipped, the solution flips
the *data*: an order-reversing involution applied before and after.

**And that flip cannot be plain negation.** `-INT_MIN` overflows, which is
undefined behaviour and in practice leaves `INT_MIN` unchanged, silently
misordering. The map `x → -x-1` covers every `int` with no hole, and it is its
own inverse. `test_heap_sort` includes `INT_MIN` and `INT_MAX` specifically to
catch the naive version.

**Heapsort is not stable, and the array shows you why.** `heap_pop` moves the
*last* element to the root, which throws away any relationship between an
element's position and its arrival order. Equal priorities come out in whatever
order the sifting happened to leave them, which is why the tests carry a payload
but never assert on tie order.

## The solution, step by step

### From first principles

1. **A priority queue needs only the minimum, not a full order.** The heap
   property asks for less than sorting: every parent is `<=` its children. Then
   the root is the minimum, and nothing else is promised.
2. **A complete binary tree fits in an array with no pointers.** Fill levels left
   to right; node `i`'s children are `2i+1` and `2i+2`, its parent `(i-1)/2`.
   The tree has height `floor(log2 n)`, so any root-to-leaf walk is O(log n).
3. **Push appends, then repairs upward.** Appending at index `len` keeps the tree
   complete. Only the edge to its parent can violate the property, so swap with
   the parent while the parent is larger (`sift_up`).
4. **Pop replaces the root with the last element, then repairs downward.** Moving
   the last item to index 0 keeps the tree complete. Now the root may be too
   large; swap it with the *smaller* child until both children are no smaller
   (`sift_down`). Swapping with the smaller child is what keeps the new parent
   `<=` the other child.
5. **Heapify bottom-up.** Leaves (indices `>= n/2`) are already heaps. Sifting
   down each internal node from `n/2 - 1` back to 0 makes every subtree a heap
   once its children are. A node at height `h` costs O(h), and about `n/2^(h+1)`
   nodes have height `h`, so the total is O(n).
6. **Heapsort is heapify plus `n - 1` pops in place.** Swap the root with the
   last element of the heap region, shrink the region by one, and sift down. The
   extracted elements accumulate at the end. With a min-heap that end fills with
   the smallest first, giving descending order, so the solution flips priorities
   to turn the min-heap into a max-heap without a second `sift_down`.

### The code, annotated

The index arithmetic and the two repairs:

```c
static size_t parent_of(size_t i) { return (i - 1) / 2; }
static size_t left_of(size_t i) { return 2 * i + 1; }

static void sift_up(HeapItem *items, size_t i) {
  while (i > 0) {                       /* the root has no parent */
    size_t p = parent_of(i);
    if (items[p].priority <= items[i].priority) break;   /* property holds */
    swap_items(&items[p], &items[i]);
    i = p;
  }
}

/* `n` is the size of the heap region, which heap_sort shrinks as it goes. */
static void sift_down(HeapItem *items, size_t n, size_t i) {
  for (;;) {
    size_t l = left_of(i);
    if (l >= n) break;                  /* no children: i is a leaf */

    /* Descend toward the SMALLER child. Picking the left unconditionally can
     * put a larger value above a smaller sibling. */
    size_t smallest = l;
    size_t r = l + 1;
    if (r < n && items[r].priority < items[l].priority) smallest = r;

    if (items[i].priority <= items[smallest].priority) break;
    swap_items(&items[i], &items[smallest]);
    i = smallest;
  }
}
```

The `<=` in both break conditions stops at equal priorities, which avoids
pointless swaps. The `l >= n` test comes first because `r = l + 1` could
otherwise read past the region.

Push and pop are the two repairs applied at the two ends:

```c
int heap_push(Heap *h, int priority, int payload) {
  if (heap_reserve(h, h->len + 1) != 0) return -1;   /* doubling, like C 01 */
  h->items[h->len].priority = priority;
  h->items[h->len].payload = payload;
  h->len++;
  sift_up(h->items, h->len - 1);
  return 0;
}

int heap_pop(Heap *h, HeapItem *out) {
  if (h->len == 0) return -1;
  *out = h->items[0];

  /* Move the last item to the root and sift it down. Shifting everything
   * left instead would be O(n) and would scramble the parent/child pairs. */
  h->items[0] = h->items[h->len - 1];
  h->len--;
  sift_down(h->items, h->len, 0);
  return 0;
}
```

Floyd's linear-time build:

```c
int heap_build(Heap *h, const HeapItem *items, size_t n) {
  if (heap_reserve(h, n) != 0) return -1;
  for (size_t i = 0; i < n; i++) h->items[i] = items[i];
  h->len = n;

  if (n < 2) return 0;
  /* From the last internal node (n/2 - 1) down to the root. `i-- > 0` is the
   * size_t-safe way to count down to and including 0. */
  for (size_t i = n / 2; i-- > 0;) sift_down(h->items, n, i);
  return 0;
}
```

Heapsort with the order-reversing flip:

```c
/* Total over every int: computed in long long so -INT_MIN cannot overflow,
 * and -x-1 maps INT_MIN to INT_MAX and back. Its own inverse. */
static int flip(int x) { return (int)(-(long long)x - 1); }

void heap_sort(HeapItem *items, size_t n) {
  if (n < 2) return;

  /* Flip, so the min-heap machinery behaves as a max-heap on real values. */
  for (size_t i = 0; i < n; i++) items[i].priority = flip(items[i].priority);

  for (size_t i = n / 2; i-- > 0;) sift_down(items, n, i);   /* heapify */

  /* Each pass parks the smallest flipped value (the largest real one) at the
   * end of the region, then re-heapifies the shrinking prefix. */
  for (size_t end = n; end > 1; end--) {
    swap_items(&items[0], &items[end - 1]);
    sift_down(items, end - 1, 0);
  }

  for (size_t i = 0; i < n; i++) items[i].priority = flip(items[i].priority);
}
```

Full solution: [`heap.c`](heap.c).

### Why this solution

- **Cost.** Push and pop O(log n), peek O(1), build O(n), sort O(n log n) time
  and O(1) extra space. Storage is one array of `HeapItem`, grown by doubling.
- **Versus a sorted array.** Pop-min is O(1) but insert is O(n) because of the
  shift. The heap balances both at O(log n).
- **Versus a balanced BST.** A BST also gives O(log n) and supports ordered
  iteration and arbitrary deletion, but costs pointers per node and allocation
  per insert. If you only need the minimum, the heap is smaller and faster.
- **Flip versus a comparator parameter.** Passing a comparison function would
  make one `sift_down` serve both orders too, at the cost of an indirect call per
  comparison. The flip keeps the hot loop a plain `<` and costs two linear
  passes over the data.
- **Accepted trade.** Heapsort is not stable and has poor cache behaviour on
  large arrays (sift_down jumps between distant indices). Merge sort
  ([C 05](../05-merge-sort/)) is stable; heapsort's advantage is O(1) space with
  a guaranteed O(n log n).

### Explain it back

1. Why does `sift_down` compare against the smaller child rather than either
   child?

   <details><summary>Answer</summary>

   After the swap, the child moved up becomes the parent of its former sibling.
   Only the smaller of the two children is guaranteed to be `<=` the other, so
   only promoting the smaller keeps the property on both edges.

   </details>

2. Why is building bottom-up O(n) when n pushes is O(n log n)?

   <details><summary>Answer</summary>

   A push can sift up the full height, and half the nodes are leaves at the
   bottom, so pushes cost about `log n` each. Sifting down from a node costs at
   most its height, and half the nodes have height 0, a quarter height 1, and so
   on: the sum `n * (0/2 + 1/4 + 2/8 + ...)` is less than `n`.

   </details>

3. Why does `flip` use `-x - 1` computed in `long long` rather than `-x`?

   <details><summary>Answer</summary>

   `-INT_MIN` does not fit in an `int`, so `-x` is undefined for that value and
   typically yields `INT_MIN` again, which misorders it. `-x - 1` maps the range
   `[INT_MIN, INT_MAX]` onto itself in reverse order with no gap, and the wider
   type keeps the intermediate in range.

   </details>

4. Why is heapsort unstable even though `sift_down` uses `<=` to stop at ties?

   <details><summary>Answer</summary>

   Instability comes from the big moves, not the comparisons. Every extraction
   swaps the root with the last element of the region, carrying an item across
   the whole array, regardless of which equal items came first in the input.

   </details>

5. In `heap_build`, why does the loop start at `n / 2` rather than `n - 1`?

   <details><summary>Answer</summary>

   Nodes from `n/2` to `n-1` have no children (their left child index `2i+1` is
   at least `n`), so they are already one-element heaps. Sifting them down would
   do nothing. `i-- > 0` starts the body at `n/2 - 1`, the last internal node.

   </details>

## Extending it

Add `heap_decrease_key`, which is what Dijkstra actually needs and what makes a
heap awkward: you must find the element first, and the array gives you no way to
do that in less than O(n). The usual fix is a side table from element identity
to current index, updated on every swap, which is a good illustration of how a
clean data structure grows a wart the moment a real algorithm uses it.
