/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "arena.h"

#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0;
#define CHECK(cond, what)                                                      \
  do {                                                                         \
    checks++;                                                                  \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int is_aligned(const void *p, size_t align) {
  return ((uintptr_t)p % align) == 0;
}

static void test_init_allocates_nothing(void) {
  Arena a;
  arena_init(&a, 0);
  CHECK(arena_block_count(&a) == 0, "a fresh arena holds no blocks");
  CHECK(a.used_total == 0, "and has handed out nothing");
  arena_free(&a);
}

static void test_basic_allocation(void) {
  Arena a;
  arena_init(&a, 1024);

  int *x = arena_alloc(&a, sizeof(int));
  CHECK(x != NULL, "allocation succeeds");
  *x = 42;
  CHECK(*x == 42, "the memory is writable");

  double *d = arena_alloc(&a, sizeof(double));
  CHECK(d != NULL, "second allocation succeeds");
  *d = 3.5;
  CHECK(*d == 3.5 && *x == 42, "allocations do not overlap");

  CHECK((void *)d != (void *)x, "distinct allocations have distinct addresses");
  arena_free(&a);
}

/* Every allocation must satisfy the alignment it asked for. An implementation
 * that bumps by size alone passes for ints and produces misaligned doubles the
 * moment an odd-sized allocation comes first. */
static void test_alignment(void) {
  Arena a;
  arena_init(&a, 4096);

  /* Deliberately start with a 1-byte allocation so the offset is odd. */
  char *c = arena_alloc_aligned(&a, 1, 1);
  CHECK(c != NULL, "byte allocation succeeds");

  for (size_t align = 1; align <= 64; align *= 2) {
    void *p = arena_alloc_aligned(&a, 1, align);
    CHECK(p != NULL, "aligned allocation succeeds");
    CHECK(is_aligned(p, align), "the pointer honours the requested alignment");
  }

  /* The default alignment must suit any scalar. */
  for (int i = 0; i < 20; i++) {
    arena_alloc_aligned(&a, 1, 1); /* push the offset out of alignment */
    void *p = arena_alloc(&a, sizeof(long double));
    CHECK(is_aligned(p, alignof(max_align_t)), "arena_alloc is suitably aligned for any scalar");
  }

  CHECK(arena_alloc_aligned(&a, 8, 3) == NULL, "a non-power-of-two alignment is rejected");
  CHECK(arena_alloc_aligned(&a, 8, 0) == NULL, "a zero alignment is rejected");
  arena_free(&a);
}

/* Writing to every byte of every allocation catches overlapping regions, which
 * a wrong bump can produce while every individual pointer still looks fine. */
static void test_allocations_never_overlap(void) {
  enum { N = 500 };
  Arena a;
  arena_init(&a, 256); /* small blocks, so this spills across many */
  unsigned char *ptrs[N];
  size_t sizes[N];

  for (int i = 0; i < N; i++) {
    sizes[i] = (size_t)(i % 37) + 1;
    ptrs[i] = arena_alloc(&a, sizes[i]);
    CHECK(ptrs[i] != NULL, "allocation succeeds");
    memset(ptrs[i], i & 0xFF, sizes[i]);
  }

  for (int i = 0; i < N; i++) {
    for (size_t j = 0; j < sizes[i]; j++) {
      CHECK(ptrs[i][j] == (unsigned char)(i & 0xFF), "no allocation was overwritten by another");
    }
  }
  CHECK(arena_block_count(&a) > 1, "the test really did spill across blocks");
  arena_free(&a);
}

/* An allocation larger than the default block must still work: the arena
 * grows a block sized to fit rather than failing. */
static void test_oversized_allocation(void) {
  Arena a;
  arena_init(&a, 64);
  unsigned char *big = arena_alloc(&a, 100000);
  CHECK(big != NULL, "an allocation larger than the block size succeeds");
  memset(big, 0xAB, 100000);
  CHECK(big[0] == 0xAB && big[99999] == 0xAB, "the whole region is usable");

  int *after = arena_alloc(&a, sizeof(int));
  CHECK(after != NULL, "the arena still works afterwards");
  *after = 7;
  CHECK(*after == 7 && big[50000] == 0xAB, "and the big block was not clobbered");
  arena_free(&a);
}

static void test_strdup(void) {
  Arena a;
  arena_init(&a, 1024);
  char buf[32];
  snprintf(buf, sizeof buf, "%s", "hello arena");

  char *copy = arena_strdup(&a, buf);
  CHECK(copy != NULL, "strdup succeeds");
  CHECK(strcmp(copy, "hello arena") == 0, "the copy matches");
  CHECK(copy != buf, "it really is a copy");

  snprintf(buf, sizeof buf, "%s", "clobbered");
  CHECK(strcmp(copy, "hello arena") == 0, "the copy is independent of the source");

  char *empty = arena_strdup(&a, "");
  CHECK(empty != NULL && empty[0] == '\0', "the empty string round-trips");
  arena_free(&a);
}

static void test_mark_and_rewind(void) {
  Arena a;
  arena_init(&a, 1024);

  int *keep = arena_alloc(&a, sizeof(int));
  *keep = 111;

  ArenaMark m = arena_mark(&a);

  /* Allocate a lot, enough to add blocks past the mark. */
  for (int i = 0; i < 2000; i++) {
    void *p = arena_alloc(&a, 64);
    CHECK(p != NULL, "scratch allocation succeeds");
  }
  CHECK(arena_block_count(&a) > 1, "the scratch work added blocks");

  arena_reset_to(&a, m);
  CHECK(*keep == 111, "data allocated before the mark survives the rewind");

  /* The next allocation should reuse the space the scratch work occupied. */
  void *reused = arena_alloc(&a, 64);
  CHECK(reused != NULL, "allocation after rewind succeeds");
  CHECK(*keep == 111, "and still does not disturb the kept data");
  arena_free(&a);
}

/* Rewinding to a mark taken on a fresh arena, before any block exists, is the
 * boundary case that a naive implementation dereferences NULL on. */
static void test_rewind_to_the_very_beginning(void) {
  Arena a;
  arena_init(&a, 128);
  ArenaMark start = arena_mark(&a);
  CHECK(arena_block_count(&a) == 0, "no block exists yet");

  for (int i = 0; i < 100; i++) arena_alloc(&a, 32);
  CHECK(arena_block_count(&a) > 0, "allocation created blocks");

  arena_reset_to(&a, start);
  CHECK(arena_block_count(&a) == 0, "rewinding to the start drops every block");

  void *p = arena_alloc(&a, 8);
  CHECK(p != NULL, "the arena is still usable afterwards");
  arena_free(&a);
}

/* reset must keep memory for reuse. If it freed everything, the second round
 * of allocations would need fresh mallocs, which is exactly the cost an arena
 * exists to avoid in a per-frame or per-request loop. */
static void test_reset_reuses_memory(void) {
  Arena a;
  arena_init(&a, 4096);

  for (int i = 0; i < 10; i++) arena_alloc(&a, 128);
  void *first_round = arena_alloc(&a, 128);
  size_t blocks_before = arena_block_count(&a);

  arena_reset(&a);
  CHECK(a.used_total == 0, "reset zeroes the usage counter");
  CHECK(arena_block_count(&a) >= 1, "reset keeps memory rather than returning it");

  for (int i = 0; i < 10; i++) arena_alloc(&a, 128);
  void *second_round = arena_alloc(&a, 128);
  CHECK(second_round == first_round, "the same addresses are handed out again");
  CHECK(arena_block_count(&a) <= blocks_before, "and no extra blocks were needed");

  arena_free(&a);
  CHECK(arena_block_count(&a) == 0, "free returns everything");
  arena_free(&a); /* must not double-free */
}

static void test_zero_size_allocation(void) {
  Arena a;
  arena_init(&a, 256);
  void *p = arena_alloc(&a, 0);
  CHECK(p != NULL, "a zero-size allocation returns a usable pointer");
  CHECK(is_aligned(p, alignof(max_align_t)), "and it is still aligned");

  int *q = arena_alloc(&a, sizeof(int));
  *q = 5;
  CHECK(*q == 5, "the arena continues to work");
  arena_free(&a);
}

int main(void) {
  test_init_allocates_nothing();
  test_basic_allocation();
  test_alignment();
  test_allocations_never_overlap();
  test_oversized_allocation();
  test_strdup();
  test_mark_and_rewind();
  test_rewind_to_the_very_beginning();
  test_reset_reuses_memory();
  test_zero_size_allocation();
  printf("ok  c/08-arena-allocator  %d checks passed\n", checks);
  return 0;
}
