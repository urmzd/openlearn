# C 05: Merge sort

**Concepts:** pointer arithmetic, recursion, stability, rotation
**Difficulty:** ⭐⭐

A stable merge sort with one scratch allocation for the whole sort, and then the
genuinely in-place variant that uses rotations instead of a buffer.

## The problem

[`msort.h`](msort.h) declares the interface, [`msort.c`](msort.c) is the
solution, and [`main.c`](main.c) is the test suite.

| Function | Does | Space |
|----------|------|-------|
| `msort(a, n)` | Stable sort ascending by key | One buffer of `n`, allocated once |
| `msort_merge(a, left, n, scratch)` | Merge two adjacent sorted runs | Caller's scratch |
| `msort_inplace(a, n)` | Stable sort with no buffer at all | O(1) heap, at O(n log²n) time |

Records carry a `key` and a `tag`. The tag is never compared, so it is how the
tests detect an unstable sort: tags start in input order, so equal keys that come
out with descending tags prove elements were reordered.

## Build and run

From this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c msort.c -o msort && ./msort
# ok  c/05-merge-sort  26111 checks passed
```

## What to notice

**Stability is one character.** In the merge loop, `a[i].key <= a[j].key` takes
from the left run on a tie and `<` takes from the right. Both produce a
correctly sorted array. Only one of them is stable, and the difference is
invisible until you sort records by one field having already sorted by another,
at which point the first sort's work is silently destroyed. This is why
`std::sort` and `std::stable_sort` are different functions.

**One allocation, not O(n) of them.** Allocating scratch inside the recursion is
correct and slow: `malloc` then dominates the runtime, and a failure partway
through leaves the array half-permuted with no way to report which half.
Allocating once up front means the sort either starts or does not.

**The `a[mid-1] <= a[mid]` check is nearly free and sometimes everything.** One
comparison per merge, and on already-sorted input the entire sort collapses to
a linear number of comparisons. Real data is very often nearly sorted, which is
the observation Timsort is built on.

**Comparison by subtraction is a bug waiting for the right input.** Writing
`a[i].key - a[j].key < 0` overflows when the keys straddle zero at the extremes,
and the sign flips. `test_extreme_keys` uses `INT_MIN` and `INT_MAX` for exactly
this reason. Compare with `<`, never with subtraction.

**"In-place merge sort" usually means rotation, and it costs a log factor.** The
buffer is what makes merging linear. Remove it and the standard approach is:
split the larger run in half, binary search for where that midpoint belongs in
the other run, rotate the block between them, and recurse. Rotation is three
reversals and no extra memory. The result is O(n log²n), meaningfully slower in
practice, and the honest answer to "can you merge sort without extra space."

**Stability in the rotating merge depends on which binary search you use, and
the two cases need different ones.** When you split the *left* run, the pivot
element came from the left, so equal keys on the right must stay after it:
`lower_bound`. When you split the *right* run, the pivot came from the right, so
equal keys on the left must come before it: `upper_bound`. Using the same bound
in both branches sorts correctly and is unstable, and it only shows on input
with duplicates spanning the seam. That is why
`test_stability_under_heavy_duplication` uses a key range of five over four
thousand elements.

## The solution, step by step

### From first principles

1. **Two sorted runs merge in linear time.** Compare the heads, take the smaller,
   advance. Each step places one element, so merging `n` elements takes at most
   `n - 1` comparisons. Writing into a separate buffer is what keeps it linear:
   no element has to be shifted.
2. **Divide and conquer turns merging into sorting.** Split in half, sort each
   half recursively, merge. The recursion has `log2 n` levels and each level
   merges `n` elements in total, so the sort is O(n log n).
3. **Every merge at every level can share one buffer.** A merge of `n` elements
   needs `n` scratch slots and nothing survives in scratch after the merge
   copies back. So one `n`-element buffer, allocated before the recursion,
   serves all of them, and allocation failure is reported before any element
   moves.
4. **Stability is decided at ties.** An element of the left run was before every
   element of the right run in the input. Taking the left one on equal keys
   (`<=`) keeps that order; that is the whole of stability.
5. **Without a buffer, merge by rotation.** Rotating `[A B]` into `[B A]` is three
   reversals and O(1) memory. Split the larger run at its midpoint, find the
   matching cut in the other run by binary search, rotate the middle block so
   both pieces left of the cut are smaller than both right of it, then merge the
   two halves recursively. Each rotation is linear, the recursion is
   logarithmic deep per merge, giving O(n log n) per merge and O(n log²n)
   overall.
6. **The binary search must respect stability.** The cut must keep equal keys
   on the side they came from, which fixes `lower_bound` when the pivot is from
   the left run and `upper_bound` when it is from the right.

### The code, annotated

The buffered merge. Note that the cursors are indices into one array, since the
two runs are adjacent:

```c
void msort_merge(Record *a, size_t left, size_t n, Record *scratch) {
  size_t i = 0;      /* cursor into the left run,  a[0..left)  */
  size_t j = left;   /* cursor into the right run, a[left..n)  */
  size_t k = 0;      /* cursor into scratch                    */

  while (i < left && j < n) {
    /* <= and not <: on equal keys the LEFT element wins, because it came
     * first in the original array. This character is the whole of stability. */
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
```

The recursion, with the already-ordered shortcut:

```c
static void msort_rec(Record *a, size_t n, Record *scratch) {
  if (n < 2) return;

  /* n/2 keeps both halves non-empty for every n >= 2, so the recursion
   * always shrinks and terminates. */
  size_t mid = n / 2;
  msort_rec(a, mid, scratch);
  msort_rec(a + mid, n - mid, scratch);   /* pointer arithmetic: the right half */

  /* Halves already in order: skip the merge. One comparison when it fails,
   * a whole merge saved when it succeeds. */
  if (a[mid - 1].key <= a[mid].key) return;

  msort_merge(a, mid, n, scratch);
}

int msort(Record *a, size_t n) {
  if (n < 2) return 0;

  /* One allocation for the entire sort. If it fails, nothing has moved. */
  Record *scratch = malloc(n * sizeof(Record));
  if (!scratch) return -1;

  msort_rec(a, n, scratch);
  free(scratch);
  return 0;
}
```

Both recursive calls receive the same `scratch` pointer. That is safe because
the calls run one after the other and each finishes with its merge copied back.

The in-place variant's building block, rotation by three reversals:

```c
/* Reverse a[lo..hi). lo + 1 < hi avoids computing hi - 1 on an empty range,
 * which would underflow a size_t. */
static void reverse(Record *a, size_t lo, size_t hi) {
  while (lo + 1 < hi) {
    Record t = a[lo];
    a[lo] = a[hi - 1];
    a[hi - 1] = t;
    lo++;
    hi--;
  }
}

/* Rotate a[lo..hi) left by k: reverse A, reverse B, reverse the whole. */
static void rotate(Record *a, size_t lo, size_t hi, size_t k) {
  if (k == 0 || lo + k >= hi) return;
  reverse(a, lo, lo + k);
  reverse(a, lo + k, hi);
  reverse(a, lo, hi);
}
```

The rotating merge, where the choice of bound carries stability:

```c
static void merge_inplace(Record *a, size_t lo, size_t mid, size_t hi) {
  if (lo >= mid || mid >= hi) return;          /* one run is empty */
  if (a[mid - 1].key <= a[mid].key) return;    /* already ordered */

  size_t l1, l2;
  if (mid - lo >= hi - mid) {
    /* Split the left run. a[l1] came from the left, so equal right-run keys
     * must stay AFTER it: lower_bound stops before them. */
    l1 = lo + (mid - lo) / 2;
    l2 = lower_bound(a, mid, hi, a[l1].key);
  } else {
    /* Split the right run. a[l2] came from the right, so equal left-run keys
     * must stay BEFORE it: upper_bound searches past them. */
    l2 = mid + (hi - mid) / 2;
    l1 = upper_bound(a, lo, mid, a[l2].key);
  }

  /* [l1..mid) are left elements bigger than the cut, [mid..l2) right
   * elements smaller than it: swap the two blocks. */
  rotate(a, l1, l2, mid - l1);
  size_t new_mid = l1 + (l2 - mid);

  merge_inplace(a, lo, l1, new_mid);
  merge_inplace(a, new_mid, l2, hi);
}
```

After the rotation, everything in `[lo, new_mid)` is no larger than everything
in `[new_mid, hi)`, so the two recursive merges are independent. Splitting the
*larger* run guarantees each recursive merge gets at most three quarters of the
elements, which bounds the depth.

Full solution: [`msort.c`](msort.c).

### Why this solution

- **Cost.** `msort`: O(n log n) time, O(n) scratch, O(log n) stack, best case
  O(n) comparisons on sorted input thanks to the skip. `msort_inplace`:
  O(n log²n) time, no heap memory, O(log n) stack (the sort's recursion plus
  one merge's recursion, each logarithmic).
- **Versus quicksort.** Quicksort is in place and usually faster, but it is not
  stable and is O(n²) on adversarial input without care. Merge sort guarantees
  O(n log n) and stability, which is why it backs `std::stable_sort` and
  Python's `sorted`.
- **Versus heapsort** ([C 04](../04-binary-heap/)). Also O(1) space with a
  guaranteed O(n log n), but unstable. The rotating merge is the price of having
  both stability and no buffer.
- **Top-down versus bottom-up.** Bottom-up merge sort (merge runs of 1, 2, 4,
  ...) has no recursion at all. Top-down is easier to read and its depth is only
  `log2 n`, so stack depth is not a concern here.
- **Accepted trade.** `msort` copies the merged run back with `memcpy` every
  time. Alternating the roles of `a` and `scratch` between levels would save
  that copy, at the cost of noticeably more intricate code.

### Explain it back

1. If the merge used `<` instead of `<=`, would the output still be sorted? What
   would change?

   <details><summary>Answer</summary>

   It would still be sorted by key, but on equal keys the right run's element
   would be taken first, reversing the input order of equal records. The sort
   becomes unstable, and the tag checks fail.

   </details>

2. Why is it safe for both recursive calls in `msort_rec` to use the same
   `scratch` buffer?

   <details><summary>Answer</summary>

   The calls are sequential, and each merge copies its result back into `a`
   before returning. Scratch never holds data that must survive past the merge
   that wrote it, so the next merge can overwrite it freely.

   </details>

3. Why does `reverse` loop on `lo + 1 < hi` instead of `lo < hi - 1`?

   <details><summary>Answer</summary>

   The indices are `size_t`. If `hi` is 0, `hi - 1` wraps to `SIZE_MAX` and the
   loop runs off the array. `lo + 1 < hi` expresses the same condition without
   ever subtracting from `hi` when it could be zero.

   </details>

4. In `merge_inplace`, why `lower_bound` when splitting the left run and
   `upper_bound` when splitting the right?

   <details><summary>Answer</summary>

   The pivot's equals in the other run must end up on the side that preserves
   input order. A left-run pivot must precede right-run equals, so the cut goes
   before them (lower bound). A right-run pivot must follow left-run equals, so
   the cut goes after them (upper bound).

   </details>

5. Why does the in-place merge split the larger of the two runs?

   <details><summary>Answer</summary>

   Halving the larger run guarantees that each recursive sub-merge contains at
   most three quarters of the elements, so the recursion depth is O(log n).
   Halving a tiny run next to a huge one could leave nearly all the elements in
   one sub-merge and make the depth linear.

   </details>

## Extending it

The natural next step is a bottom-up merge sort with no recursion at all, which
is what you want when stack depth is a real constraint. After that, run
detection: scan for already-ordered runs, reverse the descending ones, and merge
only what is left. That is most of Timsort, and it turns nearly-sorted input
from O(n log n) into something close to O(n).
