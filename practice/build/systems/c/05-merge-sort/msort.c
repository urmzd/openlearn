/* msort.c - the part you write. */
#include "msort.h"

#include <stdlib.h>
#include <string.h>

void msort_merge(Record *a, size_t left, size_t n, Record *scratch) {
  size_t i = 0;      /* cursor into the left run,  a[0..left)  */
  size_t j = left;   /* cursor into the right run, a[left..n)  */
  size_t k = 0;      /* cursor into scratch                    */

  while (i < left && j < n) {
    /* <= and not <. This single character is the whole of stability: when the
     * keys are equal the LEFT element must win, because it came first in the
     * original array. Flipping it to < still sorts correctly and silently
     * destroys the ordering of equal elements. */
    if (a[i].key <= a[j].key) {
      scratch[k++] = a[i++];
    } else {
      scratch[k++] = a[j++];
    }
  }

  /* Exactly one of these runs still has elements. */
  while (i < left) scratch[k++] = a[i++];
  while (j < n) scratch[k++] = a[j++];

  memcpy(a, scratch, n * sizeof(Record));
}

/* Sort a[0..n) using `scratch`, which is guaranteed to hold at least n
 * records. */
static void msort_rec(Record *a, size_t n, Record *scratch) {
  if (n < 2) return;

  /* Halving with n/2 rather than (n+1)/2 keeps both halves non-empty for every
   * n >= 2, which is what guarantees the recursion terminates. */
  size_t mid = n / 2;
  msort_rec(a, mid, scratch);
  msort_rec(a + mid, n - mid, scratch);

  /* Skip the merge entirely when the halves are already in order. On sorted or
   * nearly sorted input this turns the whole sort into a linear scan, and it
   * costs one comparison otherwise. */
  if (a[mid - 1].key <= a[mid].key) return;

  msort_merge(a, mid, n, scratch);
}

int msort(Record *a, size_t n) {
  if (n < 2) return 0;

  /* One allocation for the entire sort. Allocating inside the recursion would
   * mean O(n log n) calls to malloc, which dominates the runtime and can fail
   * partway through a sort that has already permuted the array. */
  Record *scratch = malloc(n * sizeof(Record));
  if (!scratch) return -1;

  msort_rec(a, n, scratch);
  free(scratch);
  return 0;
}

/* ------------------------------------------------------------------------- */
/* The O(1)-space variant.                                                    */
/* ------------------------------------------------------------------------- */

/* Reverse a[lo..hi), hi exclusive. Written so that neither index can underflow
 * when the range is empty, which is easy to get wrong with size_t. */
static void reverse(Record *a, size_t lo, size_t hi) {
  while (lo + 1 < hi) {
    Record t = a[lo];
    a[lo] = a[hi - 1];
    a[hi - 1] = t;
    lo++;
    hi--;
  }
}

/* Rotate a[lo..hi) left by k, using three reversals and no extra memory.
 * (AB)^R reversed piecewise gives BA, which is the trick that makes an
 * in-place merge possible at all. */
static void rotate(Record *a, size_t lo, size_t hi, size_t k) {
  if (k == 0 || lo + k >= hi) return;
  reverse(a, lo, lo + k);
  reverse(a, lo + k, hi);
  reverse(a, lo, hi);
}

/* First index in a[lo..hi) whose key is >= key. Stable-safe lower bound. */
static size_t lower_bound(const Record *a, size_t lo, size_t hi, int key) {
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (a[mid].key < key) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/* First index in a[lo..hi) whose key is > key. */
static size_t upper_bound(const Record *a, size_t lo, size_t hi, int key) {
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    if (a[mid].key <= key) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/* Merge the sorted runs a[lo..mid) and a[mid..hi) in place. */
static void merge_inplace(Record *a, size_t lo, size_t mid, size_t hi) {
  if (lo >= mid || mid >= hi) return;
  if (a[mid - 1].key <= a[mid].key) return; /* already ordered */

  /* Split the larger run in half, find where that midpoint belongs in the
   * other run, rotate the block between them into place, and recurse on both
   * sides. Each rotation places at least one element correctly and the
   * recursion depth is logarithmic, giving O(n log^2 n) overall. */
  size_t l1, l2;
  if (mid - lo >= hi - mid) {
    /* Split the left run. a[l1] came from the left, so equal keys in the right
     * run must end up AFTER it: search for the first right-run element that is
     * not less than it, and leave the equals behind. That is lower_bound. */
    l1 = lo + (mid - lo) / 2;
    l2 = lower_bound(a, mid, hi, a[l1].key);
  } else {
    /* Split the right run. a[l2] came from the right, so equal keys in the
     * left run must end up BEFORE it: search past them. That is upper_bound.
     *
     * Using the same bound in both branches is the stability bug here, and it
     * only shows on input with duplicates spanning the seam. */
    l2 = mid + (hi - mid) / 2;
    l1 = upper_bound(a, lo, mid, a[l2].key);
  }

  /* Bring a[l1..mid) and a[mid..l2) into the order they belong in. */
  rotate(a, l1, l2, mid - l1);
  size_t new_mid = l1 + (l2 - mid);

  merge_inplace(a, lo, l1, new_mid);
  merge_inplace(a, new_mid, l2, hi);
}

void msort_inplace(Record *a, size_t n) {
  if (n < 2) return;
  size_t mid = n / 2;
  msort_inplace(a, mid);
  msort_inplace(a + mid, n - mid);
  merge_inplace(a, 0, mid, n);
}
