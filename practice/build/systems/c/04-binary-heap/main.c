/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "heap.h"

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

/* A deterministic pseudo-random stream. rand() would work, but pinning the
 * generator here means a failure reproduces exactly on every platform. */
static unsigned long rng_state = 0x2545F4914F6CDD1DUL;
static int next_int(int lo, int hi) {
  rng_state = rng_state * 6364136223846793005UL + 1442695040888963407UL;
  return lo + (int)((rng_state >> 33) % (unsigned long)(hi - lo + 1));
}

/* The heap property, checked directly against the array rather than inferred
 * from pop order: every node is no larger than either of its children. */
static void assert_heap_property(const Heap *h, const char *where) {
  for (size_t i = 0; i < h->len; i++) {
    size_t l = 2 * i + 1, r = 2 * i + 2;
    if (l < h->len) {
      checks++;
      if (h->items[i].priority > h->items[l].priority) {
        fprintf(stderr, "FAIL heap property broken at %zu/left (%s)\n", i, where);
        exit(1);
      }
    }
    if (r < h->len) {
      checks++;
      if (h->items[i].priority > h->items[r].priority) {
        fprintf(stderr, "FAIL heap property broken at %zu/right (%s)\n", i, where);
        exit(1);
      }
    }
  }
}

static void test_empty(void) {
  Heap h;
  HeapItem out;
  heap_init(&h);
  CHECK(heap_len(&h) == 0, "fresh heap is empty");
  CHECK(h.items == NULL, "fresh heap has not allocated");
  CHECK(heap_pop(&h, &out) == -1, "pop on empty fails");
  CHECK(heap_peek(&h, &out) == -1, "peek on empty fails");
  heap_free(&h);
}

static void test_pops_in_priority_order(void) {
  Heap h;
  HeapItem out;
  heap_init(&h);
  int input[] = {5, 3, 8, 1, 9, 2, 7};
  for (size_t i = 0; i < sizeof input / sizeof *input; i++) {
    CHECK(heap_push(&h, input[i], (int)i) == 0, "push succeeds");
    assert_heap_property(&h, "after push");
  }
  CHECK(heap_len(&h) == 7, "every push counted");
  CHECK(heap_peek(&h, &out) == 0 && out.priority == 1, "peek sees the minimum");
  CHECK(heap_len(&h) == 7, "peek does not remove");

  int expect[] = {1, 2, 3, 5, 7, 8, 9};
  for (int i = 0; i < 7; i++) {
    CHECK(heap_pop(&h, &out) == 0, "pop succeeds");
    CHECK(out.priority == expect[i], "pop returns the smallest remaining");
    assert_heap_property(&h, "after pop");
  }
  CHECK(heap_pop(&h, &out) == -1, "drained heap is empty");
  heap_free(&h);
}

/* The payload rides along with its priority. An implementation that sifts the
 * priorities but forgets to move the payloads passes the ordering test above
 * and fails this one. */
static void test_payload_travels_with_priority(void) {
  Heap h;
  HeapItem out;
  heap_init(&h);
  for (int i = 0; i < 50; i++) heap_push(&h, 100 - i, i * 1000);

  for (int i = 0; i < 50; i++) {
    CHECK(heap_pop(&h, &out) == 0, "pop succeeds");
    /* priority p was pushed with payload (100 - p) * 1000 */
    CHECK(out.payload == (100 - out.priority) * 1000, "payload stayed with its priority");
  }
  heap_free(&h);
}

/* Descending input is the worst case for a naive push loop and the case where
 * a sift_up that stops one level early is most visible. */
static void test_adversarial_orders(void) {
  const int N = 500;
  Heap h;
  HeapItem out;

  heap_init(&h);
  for (int i = N; i > 0; i--) heap_push(&h, i, i);
  assert_heap_property(&h, "descending input");
  for (int i = 1; i <= N; i++) {
    heap_pop(&h, &out);
    CHECK(out.priority == i, "descending input still pops ascending");
  }
  heap_free(&h);

  heap_init(&h);
  for (int i = 0; i < N; i++) heap_push(&h, 42, i);
  assert_heap_property(&h, "all equal");
  CHECK(heap_len(&h) == (size_t)N, "equal priorities all fit");
  for (int i = 0; i < N; i++) {
    CHECK(heap_pop(&h, &out) == 0 && out.priority == 42, "equal priorities pop cleanly");
  }
  heap_free(&h);
}

/* Interleaving pushes and pops is what catches a sift_down that descends into
 * the left child unconditionally: the damage only shows after the heap has
 * been reshaped from both ends. */
static void test_random_interleaving(void) {
  Heap h;
  HeapItem out;
  heap_init(&h);
  int live = 0;

  for (int step = 0; step < 20000; step++) {
    if (live == 0 || next_int(0, 2) > 0) {
      CHECK(heap_push(&h, next_int(-1000, 1000), step) == 0, "push succeeds");
      live++;
    } else {
      CHECK(heap_pop(&h, &out) == 0, "pop succeeds");
      live--;
    }
  }
  CHECK(heap_len(&h) == (size_t)live, "length tracks the interleaving");
  assert_heap_property(&h, "after random interleaving");

  /* Draining must produce a non-decreasing sequence. */
  int prev = INT_MIN;
  while (heap_pop(&h, &out) == 0) {
    CHECK(out.priority >= prev, "drain is non-decreasing");
    prev = out.priority;
  }
  heap_free(&h);
}

static void test_build_is_a_heap(void) {
  Heap h;
  HeapItem out;
  HeapItem raw[1000];
  for (int i = 0; i < 1000; i++) {
    raw[i].priority = next_int(-5000, 5000);
    raw[i].payload = i;
  }

  heap_init(&h);
  CHECK(heap_build(&h, raw, 1000) == 0, "build succeeds");
  CHECK(heap_len(&h) == 1000, "build takes every item");
  assert_heap_property(&h, "after build");

  int prev = INT_MIN;
  while (heap_pop(&h, &out) == 0) {
    CHECK(out.priority >= prev, "a built heap drains in order");
    prev = out.priority;
  }
  heap_free(&h);

  /* Degenerate sizes must not walk off the array. */
  heap_init(&h);
  CHECK(heap_build(&h, raw, 0) == 0, "build of nothing succeeds");
  CHECK(heap_len(&h) == 0, "build of nothing is empty");
  heap_free(&h);

  heap_init(&h);
  CHECK(heap_build(&h, raw, 1) == 0, "build of one succeeds");
  CHECK(heap_peek(&h, &out) == 0 && out.payload == 0, "build of one holds it");
  heap_free(&h);
}

static void test_heap_sort(void) {
  HeapItem a[777];
  for (int i = 0; i < 777; i++) {
    a[i].priority = next_int(-100, 100);
    a[i].payload = i;
  }
  heap_sort(a, 777);
  for (int i = 1; i < 777; i++) {
    CHECK(a[i - 1].priority <= a[i].priority, "heap_sort produces ascending order");
  }

  /* Already sorted and reverse sorted are the inputs that expose an off-by-one
   * in the shrinking loop. */
  HeapItem asc[64], desc[64];
  for (int i = 0; i < 64; i++) {
    asc[i].priority = i;
    asc[i].payload = i;
    desc[i].priority = 64 - i;
    desc[i].payload = i;
  }
  heap_sort(asc, 64);
  heap_sort(desc, 64);
  for (int i = 1; i < 64; i++) {
    CHECK(asc[i - 1].priority <= asc[i].priority, "sorted input stays sorted");
    CHECK(desc[i - 1].priority <= desc[i].priority, "reversed input becomes sorted");
  }

  /* The extremes of int, where a naive negate-to-flip-the-comparison overflows
   * and silently misorders. */
  HeapItem ext[5] = {
      {INT_MAX, 0}, {INT_MIN, 1}, {0, 2}, {INT_MIN, 3}, {INT_MAX, 4},
  };
  heap_sort(ext, 5);
  CHECK(ext[0].priority == INT_MIN, "INT_MIN sorts first");
  CHECK(ext[1].priority == INT_MIN, "both INT_MINs sort first");
  CHECK(ext[2].priority == 0, "zero sorts in the middle");
  CHECK(ext[3].priority == INT_MAX, "INT_MAX sorts last");
  CHECK(ext[4].priority == INT_MAX, "both INT_MAXs sort last");

  /* Sorting nothing, or one thing, must not read out of bounds. */
  HeapItem tiny[1] = {{5, 0}};
  heap_sort(tiny, 0);
  heap_sort(tiny, 1);
  CHECK(tiny[0].priority == 5, "trivial sizes are left alone");
}

int main(void) {
  test_empty();
  test_pops_in_priority_order();
  test_payload_travels_with_priority();
  test_adversarial_orders();
  test_random_interleaving();
  test_build_is_a_heap();
  test_heap_sort();
  printf("ok  c/04-binary-heap  %d checks passed\n", checks);
  return 0;
}
