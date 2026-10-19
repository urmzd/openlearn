/* heap.c - the reference implementation of the heap. */
#include "heap.h"

#include <stdint.h>
#include <stdlib.h>

/* The index arithmetic that replaces pointers. With a 0-based array the
 * children of i are 2i+1 and 2i+2, and the parent of i is (i-1)/2. */
static size_t parent_of(size_t i) { return (i - 1) / 2; }
static size_t left_of(size_t i) { return 2 * i + 1; }

static void swap_items(HeapItem *a, HeapItem *b) {
  HeapItem t = *a;
  *a = *b;
  *b = t;
}

/* Order-reversing map used by heap_sort, computed in a wider type so it is
 * total over every int. Plain negation is not: -INT_MIN overflows, which is
 * undefined behaviour and in practice returns INT_MIN unchanged. Mapping
 * x to -x-1 has no such hole, and it is its own inverse. */
static int flip(int x) { return (int)(-(long long)x - 1); }

/* Move items[i] up until its parent is no larger than it. */
static void sift_up(HeapItem *items, size_t i) {
  while (i > 0) {
    size_t p = parent_of(i);
    if (items[p].priority <= items[i].priority) break;
    swap_items(&items[p], &items[i]);
    i = p;
  }
}

/* Move items[i] down until both children are at least as large as it. `n` is
 * the size of the region being treated as a heap, which is not always the
 * whole array: heap_sort shrinks it as it goes. */
static void sift_down(HeapItem *items, size_t n, size_t i) {
  for (;;) {
    size_t l = left_of(i);
    if (l >= n) break; /* no children: i is a leaf */

    /* Descend toward the SMALLER child. Picking the left child unconditionally
     * is the classic bug: it can swap a large value past a smaller sibling and
     * leave the heap property broken one level down. */
    size_t smallest = l;
    size_t r = l + 1;
    if (r < n && items[r].priority < items[l].priority) smallest = r;

    if (items[i].priority <= items[smallest].priority) break;
    swap_items(&items[i], &items[smallest]);
    i = smallest;
  }
}

static int heap_reserve(Heap *h, size_t n) {
  if (n <= h->cap) return 0;
  size_t cap = h->cap ? h->cap : 8;
  while (cap < n) {
    if (cap > SIZE_MAX / 2) return -1;
    cap *= 2;
  }
  if (cap > SIZE_MAX / sizeof(HeapItem)) return -1;
  HeapItem *grown = realloc(h->items, cap * sizeof(HeapItem));
  if (!grown) return -1;
  h->items = grown;
  h->cap = cap;
  return 0;
}

void heap_init(Heap *h) {
  h->items = NULL;
  h->len = 0;
  h->cap = 0;
}

int heap_push(Heap *h, int priority, int payload) {
  if (heap_reserve(h, h->len + 1) != 0) return -1;
  h->items[h->len].priority = priority;
  h->items[h->len].payload = payload;
  h->len++;
  sift_up(h->items, h->len - 1);
  return 0;
}

int heap_pop(Heap *h, HeapItem *out) {
  if (h->len == 0) return -1;
  *out = h->items[0];

  /* Move the last item to the root and sift it down. Shifting everything left
   * instead would be O(n) and would also destroy the heap shape. */
  h->items[0] = h->items[h->len - 1];
  h->len--;
  sift_down(h->items, h->len, 0);
  return 0;
}

int heap_peek(const Heap *h, HeapItem *out) {
  if (h->len == 0) return -1;
  *out = h->items[0];
  return 0;
}

size_t heap_len(const Heap *h) {
  return h->len;
}

int heap_build(Heap *h, const HeapItem *items, size_t n) {
  if (heap_reserve(h, n) != 0) return -1;
  for (size_t i = 0; i < n; i++) h->items[i] = items[i];
  h->len = n;

  /* Floyd's method: sift down from the last internal node backwards. This is
   * O(n), not O(n log n), because most nodes are near the bottom and sift down
   * by almost nothing. Pushing one at a time would be the slower bound. */
  if (n < 2) return 0;
  for (size_t i = n / 2; i-- > 0;) sift_down(h->items, n, i);
  return 0;
}

void heap_free(Heap *h) {
  free(h->items);
  heap_init(h);
}

void heap_sort(HeapItem *items, size_t n) {
  if (n < 2) return;

  /* Sorting ascending in place wants a MAX-heap: you repeatedly move the
   * largest to the shrinking end. But sift_down above implements a MIN-heap,
   * and duplicating it with the comparison flipped is how the two copies drift
   * apart later.
   *
   * So flip the data instead of the code. `flip` reverses the order and is its
   * own inverse, so applying it, running the min-heap machinery, and applying
   * it again sorts ascending with one sift_down implementation and no extra
   * memory. */
  for (size_t i = 0; i < n; i++) items[i].priority = flip(items[i].priority);

  for (size_t i = n / 2; i-- > 0;) sift_down(items, n, i);

  /* Each pass parks the smallest flipped value (the largest real one) at the
   * end, then re-heapifies the shrinking prefix. */
  for (size_t end = n; end > 1; end--) {
    swap_items(&items[0], &items[end - 1]);
    sift_down(items, end - 1, 0);
  }

  for (size_t i = 0; i < n; i++) items[i].priority = flip(items[i].priority);
}
