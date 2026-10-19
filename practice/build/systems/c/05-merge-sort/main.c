/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "msort.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

static int checks = 0;
#define CHECK(cond, what)                                                      \
  do {                                                                         \
    checks++;                                                                  \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static unsigned long rng_state = 0x9E3779B97F4A7C15UL;
static int next_int(int lo, int hi) {
  rng_state = rng_state * 6364136223846793005UL + 1442695040888963407UL;
  return lo + (int)((rng_state >> 33) % (unsigned long)(hi - lo + 1));
}

/* Ascending by key, and among equal keys ascending by tag. Since tags are
 * assigned in input order, the second half is exactly the stability check:
 * a sort that reorders equal elements produces a descending tag somewhere. */
static void assert_sorted_and_stable(const Record *a, size_t n, const char *what) {
  for (size_t i = 1; i < n; i++) {
    checks++;
    if (a[i - 1].key > a[i].key) {
      fprintf(stderr, "FAIL not sorted at %zu (%s)\n", i, what);
      exit(1);
    }
    if (a[i - 1].key == a[i].key && a[i - 1].tag > a[i].tag) {
      fprintf(stderr, "FAIL not stable at %zu: equal keys %d with tags %d then %d (%s)\n",
              i, a[i].key, a[i - 1].tag, a[i].tag, what);
      exit(1);
    }
  }
}

/* Every input element still present, none invented. Catches a merge that
 * duplicates one run's tail over the other's. */
static void assert_is_a_permutation(const Record *sorted, const Record *orig,
                                    size_t n, const char *what) {
  long long sum_k = 0, sum_t = 0, xor_t = 0;
  for (size_t i = 0; i < n; i++) {
    sum_k += sorted[i].key - orig[i].key;
    sum_t += sorted[i].tag - orig[i].tag;
    xor_t ^= sorted[i].tag ^ orig[i].tag;
  }
  checks += 3;
  if (sum_k != 0 || sum_t != 0 || xor_t != 0) {
    fprintf(stderr, "FAIL output is not a permutation of the input (%s)\n", what);
    exit(1);
  }
}

static void fill_random(Record *a, size_t n, int lo, int hi) {
  for (size_t i = 0; i < n; i++) {
    a[i].key = next_int(lo, hi);
    a[i].tag = (int)i;
  }
}

static void test_trivial_sizes(void) {
  Record a[2] = {{5, 0}, {3, 1}};
  CHECK(msort(a, 0) == 0, "sorting nothing succeeds");
  CHECK(a[0].key == 5, "sorting nothing changes nothing");
  CHECK(msort(a, 1) == 0, "sorting one succeeds");
  CHECK(a[0].key == 5, "sorting one changes nothing");
  CHECK(msort(a, 2) == 0, "sorting two succeeds");
  CHECK(a[0].key == 3 && a[1].key == 5, "two elements swap into order");
}

static void test_merge_step_directly(void) {
  /* Two sorted runs with interleaving keys, and equal keys across the seam. */
  Record a[8] = {{1, 0}, {4, 1}, {4, 2}, {9, 3},   /* left run  */
                 {2, 4}, {4, 5}, {7, 6}, {9, 7}};  /* right run */
  Record scratch[8];
  Record orig[8];
  for (int i = 0; i < 8; i++) orig[i] = a[i];

  msort_merge(a, 4, 8, scratch);
  assert_sorted_and_stable(a, 8, "direct merge");
  assert_is_a_permutation(a, orig, 8, "direct merge");

  /* The three 4s must come out left-run-first: tags 1, 2, then 5. */
  int seen[3], k = 0;
  for (int i = 0; i < 8; i++) {
    if (a[i].key == 4) seen[k++] = a[i].tag;
  }
  CHECK(k == 3, "all three equal keys survived the merge");
  CHECK(seen[0] == 1 && seen[1] == 2 && seen[2] == 5,
        "equal keys keep left-run-before-right-run order");

  /* An empty left run and an empty right run are the boundaries where a merge
   * loop with the wrong bound reads past the end. */
  Record b[3] = {{7, 0}, {8, 1}, {9, 2}};
  Record s3[3];
  msort_merge(b, 0, 3, s3);
  CHECK(b[0].key == 7 && b[2].key == 9, "empty left run is a no-op");
  msort_merge(b, 3, 3, s3);
  CHECK(b[0].key == 7 && b[2].key == 9, "empty right run is a no-op");
}

static void test_sorts_random_input(void) {
  enum { N = 4000 };
  Record *a = malloc(N * sizeof *a);
  Record *orig = malloc(N * sizeof *orig);
  fill_random(a, N, -10000, 10000);
  for (int i = 0; i < N; i++) orig[i] = a[i];

  CHECK(msort(a, N) == 0, "sort succeeds");
  assert_sorted_and_stable(a, N, "random input");
  assert_is_a_permutation(a, orig, N, "random input");
  free(a);
  free(orig);
}

/* Heavy duplication is where stability actually gets tested: with a key range
 * of 5 over 4000 elements, almost every comparison is between equals. */
static void test_stability_under_heavy_duplication(void) {
  enum { N = 4000 };
  Record *a = malloc(N * sizeof *a);
  Record *orig = malloc(N * sizeof *orig);
  fill_random(a, N, 0, 4);
  for (int i = 0; i < N; i++) orig[i] = a[i];

  CHECK(msort(a, N) == 0, "sort succeeds");
  assert_sorted_and_stable(a, N, "heavy duplication");
  assert_is_a_permutation(a, orig, N, "heavy duplication");

  /* All keys identical: the output must be the input, untouched. */
  for (int i = 0; i < N; i++) {
    a[i].key = 42;
    a[i].tag = i;
  }
  CHECK(msort(a, N) == 0, "sort succeeds");
  for (int i = 0; i < N; i++) {
    CHECK(a[i].tag == i, "all-equal input is left in its original order");
  }
  free(a);
  free(orig);
}

static void test_adversarial_orders(void) {
  enum { N = 2000 };
  Record *a = malloc(N * sizeof *a);

  for (int i = 0; i < N; i++) { a[i].key = i; a[i].tag = i; }
  CHECK(msort(a, N) == 0, "sort succeeds");
  assert_sorted_and_stable(a, N, "already sorted");

  for (int i = 0; i < N; i++) { a[i].key = N - i; a[i].tag = i; }
  CHECK(msort(a, N) == 0, "sort succeeds");
  assert_sorted_and_stable(a, N, "reverse sorted");

  /* Organ pipe: ascending then descending. Defeats the "already ordered" fast
   * path at the top level but hits it in the sub-sorts. */
  for (int i = 0; i < N; i++) {
    a[i].key = i < N / 2 ? i : N - i;
    a[i].tag = i;
  }
  CHECK(msort(a, N) == 0, "sort succeeds");
  assert_sorted_and_stable(a, N, "organ pipe");

  /* Odd sizes exercise the uneven split. */
  for (size_t n = 1; n <= 65; n++) {
    for (size_t i = 0; i < n; i++) {
      a[i].key = next_int(-50, 50);
      a[i].tag = (int)i;
    }
    CHECK(msort(a, n) == 0, "sort succeeds");
    assert_sorted_and_stable(a, n, "every small size");
  }
  free(a);
}

/* Keys at the extremes catch a comparison written as subtraction, where
 * INT_MAX - INT_MIN overflows and flips the sign. */
static void test_extreme_keys(void) {
  Record a[6] = {{INT_MAX, 0}, {INT_MIN, 1}, {0, 2},
                 {INT_MIN, 3}, {INT_MAX, 4}, {-1, 5}};
  CHECK(msort(a, 6) == 0, "sort succeeds");
  CHECK(a[0].key == INT_MIN && a[1].key == INT_MIN, "INT_MIN sorts first");
  CHECK(a[0].tag == 1 && a[1].tag == 3, "equal INT_MINs stay in order");
  CHECK(a[2].key == -1, "-1 sorts next");
  CHECK(a[3].key == 0, "0 sorts next");
  CHECK(a[4].key == INT_MAX && a[5].key == INT_MAX, "INT_MAX sorts last");
  CHECK(a[4].tag == 0 && a[5].tag == 4, "equal INT_MAXs stay in order");
}

static void test_inplace_variant(void) {
  enum { N = 1500 };
  Record *a = malloc(N * sizeof *a);
  Record *orig = malloc(N * sizeof *orig);

  fill_random(a, N, -500, 500);
  for (int i = 0; i < N; i++) orig[i] = a[i];
  msort_inplace(a, N);
  assert_sorted_and_stable(a, N, "in-place, random");
  assert_is_a_permutation(a, orig, N, "in-place, random");

  fill_random(a, N, 0, 3);
  for (int i = 0; i < N; i++) orig[i] = a[i];
  msort_inplace(a, N);
  assert_sorted_and_stable(a, N, "in-place, heavy duplication");
  assert_is_a_permutation(a, orig, N, "in-place, heavy duplication");

  for (int i = 0; i < N; i++) { a[i].key = N - i; a[i].tag = i; }
  msort_inplace(a, N);
  assert_sorted_and_stable(a, N, "in-place, reverse sorted");

  for (size_t n = 0; n <= 33; n++) {
    for (size_t i = 0; i < n; i++) {
      a[i].key = next_int(-20, 20);
      a[i].tag = (int)i;
    }
    msort_inplace(a, n);
    assert_sorted_and_stable(a, n, "in-place, every small size");
  }

  free(a);
  free(orig);
}

/* Both entry points must agree exactly, including on tie order. */
static void test_both_agree(void) {
  enum { N = 900 };
  Record *a = malloc(N * sizeof *a);
  Record *b = malloc(N * sizeof *b);
  fill_random(a, N, -30, 30);
  for (int i = 0; i < N; i++) b[i] = a[i];

  CHECK(msort(a, N) == 0, "sort succeeds");
  msort_inplace(b, N);
  for (int i = 0; i < N; i++) {
    CHECK(a[i].key == b[i].key && a[i].tag == b[i].tag,
          "buffered and in-place sorts produce identical output");
  }
  free(a);
  free(b);
}

int main(void) {
  test_trivial_sizes();
  test_merge_step_directly();
  test_sorts_random_input();
  test_stability_under_heavy_duplication();
  test_adversarial_orders();
  test_extreme_keys();
  test_inplace_variant();
  test_both_agree();
  printf("ok  c/05-merge-sort  %d checks passed\n", checks);
  return 0;
}
