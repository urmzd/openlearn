/* main.c - the test suite for the reference implementation.
 *
 * Exits 0 when everything holds, nonzero on the first failure.
 */
#include "vec.h"

#include <assert.h>
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

static void test_init_allocates_nothing(void) {
  Vec v;
  vec_init(&v);
  CHECK(v.len == 0, "fresh vector has len 0");
  CHECK(v.cap == 0, "fresh vector has cap 0");
  CHECK(v.data == NULL, "fresh vector has not allocated");
  vec_free(&v);
}

static void test_push_and_get(void) {
  Vec v;
  vec_init(&v);
  CHECK(vec_push(&v, 10) == 0, "push succeeds");
  CHECK(vec_push(&v, 20) == 0, "push succeeds");
  CHECK(vec_push(&v, 30) == 0, "push succeeds");
  CHECK(v.len == 3, "three pushes give len 3");
  CHECK(vec_get(&v, 0) == 10, "element 0");
  CHECK(vec_get(&v, 1) == 20, "element 1");
  CHECK(vec_get(&v, 2) == 30, "element 2");
  vec_free(&v);
}

static void test_pop_is_lifo(void) {
  Vec v;
  int out = 0;
  vec_init(&v);
  CHECK(vec_pop(&v, &out) == -1, "pop on empty fails");
  vec_push(&v, 1);
  vec_push(&v, 2);
  CHECK(vec_pop(&v, &out) == 0 && out == 2, "pop returns the last element");
  CHECK(vec_pop(&v, &out) == 0 && out == 1, "pop returns the first element");
  CHECK(vec_pop(&v, &out) == -1, "pop on drained vector fails");
  CHECK(v.len == 0, "drained vector has len 0");
  vec_free(&v);
}

static void test_reserve_does_not_change_len(void) {
  Vec v;
  vec_init(&v);
  CHECK(vec_reserve(&v, 100) == 0, "reserve succeeds");
  CHECK(v.cap >= 100, "reserve gives at least the asked-for capacity");
  CHECK(v.len == 0, "reserve does not change len");
  CHECK(vec_reserve(&v, 10) == 0, "reserving less than cap is a no-op");
  CHECK(v.cap >= 100, "reserve never shrinks");
  vec_free(&v);
}

/* The one that matters. realloc is free to move the buffer, so a correct
 * implementation must copy the elements across; an implementation that tracks
 * capacity wrong will lose or corrupt them here. Pushing well past the initial
 * capacity guarantees several moves. */
static void test_growth_preserves_every_element(void) {
  enum { N = 10000 };
  Vec v;
  vec_init(&v);
  for (int i = 0; i < N; i++) {
    CHECK(vec_push(&v, i * 3) == 0, "push during growth succeeds");
  }
  CHECK(v.len == (size_t)N, "len counts every push");
  CHECK(v.cap >= v.len, "capacity is never less than length");
  for (int i = 0; i < N; i++) {
    CHECK(vec_get(&v, (size_t)i) == i * 3, "element survived reallocation");
  }
  vec_free(&v);
}

/* Geometric growth, not arithmetic. A +1-per-push implementation passes every
 * test above and is quadratic; this is what catches it. After N pushes a
 * doubling vector has done about log2(N) allocations and holds cap < 2N. */
static void test_growth_is_geometric(void) {
  enum { N = 4096 };
  Vec v;
  vec_init(&v);
  for (int i = 0; i < N; i++) vec_push(&v, i);
  CHECK(v.cap < (size_t)N * 4, "capacity stays within a constant factor of length");
  vec_free(&v);
}

static void test_free_is_idempotent(void) {
  Vec v;
  vec_init(&v);
  vec_push(&v, 1);
  vec_free(&v);
  CHECK(v.data == NULL, "free clears the pointer");
  CHECK(v.len == 0 && v.cap == 0, "free returns the zero state");
  vec_free(&v); /* must not double-free */
  vec_init(&v);
  CHECK(vec_push(&v, 7) == 0, "a freed vector can be reused");
  CHECK(vec_get(&v, 0) == 7, "reuse works");
  vec_free(&v);
}

int main(void) {
  test_init_allocates_nothing();
  test_push_and_get();
  test_pop_is_lifo();
  test_reserve_does_not_change_len();
  test_growth_preserves_every_element();
  test_growth_is_geometric();
  test_free_is_idempotent();
  printf("ok  c/01-dynamic-array  %d checks passed\n", checks);
  return 0;
}
