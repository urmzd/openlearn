/* heap.h - an array-backed binary min-heap.
 *
 * The interface. heap.c implements it and main.c tests it.
 *
 * The whole data structure is one array. A tree shape is implied by the
 * indices, which is why a heap has no pointers, no allocation per element, and
 * perfect cache locality on a sift.
 */
#ifndef HEAP_H
#define HEAP_H

#include <stddef.h>

typedef struct {
  int priority;
  int payload; /* carried along so the tests can tell equal priorities apart */
} HeapItem;

typedef struct {
  HeapItem *items;
  size_t len;
  size_t cap;
} Heap;

/* Zero state, no allocation. */
void heap_init(Heap *h);

/* Insert. 0 on success, -1 on allocation failure. O(log n). */
int heap_push(Heap *h, int priority, int payload);

/* Remove the smallest priority into *out. -1 when empty. O(log n). */
int heap_pop(Heap *h, HeapItem *out);

/* Look at the smallest without removing it. -1 when empty. O(1). */
int heap_peek(const Heap *h, HeapItem *out);

size_t heap_len(const Heap *h);

/* Build a heap from an unordered array in O(n), not O(n log n). Takes
 * ownership of nothing: the items are copied in. */
int heap_build(Heap *h, const HeapItem *items, size_t n);

void heap_free(Heap *h);

/* Sort `items` ascending by priority, using a heap. In place, O(n log n), no
 * allocation. Ties may end up in any order: heapsort is not stable, and
 * the README explains why. */
void heap_sort(HeapItem *items, size_t n);

#endif /* HEAP_H */
