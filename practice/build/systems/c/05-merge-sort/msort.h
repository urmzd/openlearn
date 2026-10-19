/* msort.h - merge sort over an array of records.
 *
 * The interface. msort.c implements it and main.c tests it.
 *
 * Records carry a key and a tag. The tag is never compared, so it is what the
 * tests use to prove stability: equal keys must come out in their original
 * relative order.
 */
#ifndef MSORT_H
#define MSORT_H

#include <stddef.h>

typedef struct {
  int key;
  int tag;
} Record;

/* Sort ascending by key, stably.
 *
 * Allocates one scratch buffer of n records, once, for the whole sort. A
 * version that allocates inside the recursion is O(n log n) allocations and
 * defeats the purpose even though it produces the right answer.
 *
 * Returns 0 on success, -1 if the single scratch allocation fails. */
int msort(Record *a, size_t n);

/* Merge two already-sorted adjacent runs, a[0..left) and a[left..n), into
 * ascending order using `scratch` as workspace. Exposed so the tests can drive
 * the merge step directly, which is where the interesting bugs live. */
void msort_merge(Record *a, size_t left, size_t n, Record *scratch);

/* Sort in place with NO scratch buffer at all, by rotating instead of copying.
 * O(n log^2 n) and much slower in practice, but O(1) extra space.
 *
 * This is the honest version of "in-place merge sort": the phrase usually
 * means the merge is done by rotation rather than into a buffer, and the cost
 * is a log factor. */
void msort_inplace(Record *a, size_t n);

#endif /* MSORT_H */
