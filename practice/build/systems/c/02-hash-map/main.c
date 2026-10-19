/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "map.h"

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

static void test_init_allocates_nothing(void) {
  Map m;
  map_init(&m);
  CHECK(map_len(&m) == 0, "fresh map is empty");
  CHECK(m.cap == 0, "fresh map has no slots");
  CHECK(m.slots == NULL, "fresh map has not allocated");
  map_free(&m);
}

static void test_put_get_overwrite(void) {
  Map m;
  int v = 0;
  map_init(&m);
  CHECK(map_put(&m, "alpha", 1) == 0, "put succeeds");
  CHECK(map_put(&m, "beta", 2) == 0, "put succeeds");
  CHECK(map_get(&m, "alpha", &v) == 0 && v == 1, "get returns what was put");
  CHECK(map_get(&m, "beta", &v) == 0 && v == 2, "get returns what was put");
  CHECK(map_get(&m, "gamma", &v) == -1, "absent key is absent");
  CHECK(map_len(&m) == 2, "two distinct keys give len 2");
  CHECK(map_put(&m, "alpha", 99) == 0, "overwrite succeeds");
  CHECK(map_get(&m, "alpha", &v) == 0 && v == 99, "overwrite replaces the value");
  CHECK(map_len(&m) == 2, "overwrite does not change len");
  map_free(&m);
}

/* The map must copy keys, not alias the caller's buffer. */
static void test_keys_are_copied(void) {
  Map m;
  int v = 0;
  char buf[16];
  map_init(&m);
  strcpy(buf, "transient");
  CHECK(map_put(&m, buf, 42) == 0, "put succeeds");
  strcpy(buf, "clobbered"); /* the caller reuses their buffer */
  CHECK(map_get(&m, "transient", &v) == 0 && v == 42, "map kept its own copy of the key");
  CHECK(map_get(&m, "clobbered", &v) == -1, "the clobbered text was never a key");
  map_free(&m);
}

static void test_many_entries_survive_rehashing(void) {
  enum { N = 5000 };
  Map m;
  char key[32];
  int v = 0;
  map_init(&m);
  for (int i = 0; i < N; i++) {
    snprintf(key, sizeof key, "key-%d", i);
    CHECK(map_put(&m, key, i * 7) == 0, "put during growth succeeds");
  }
  CHECK(map_len(&m) == (size_t)N, "every distinct key counted once");
  for (int i = 0; i < N; i++) {
    snprintf(key, sizeof key, "key-%d", i);
    CHECK(map_get(&m, key, &v) == 0 && v == i * 7, "entry survived rehashing");
  }
  map_free(&m);
}

/* Find `n` distinct short keys that all land in the same bucket for a table of
 * `cap` slots, so the probing tests are deliberate rather than lucky. */
static void colliding_keys(size_t cap, char out[][8], int n) {
  int found = 0;
  size_t target = 0;
  for (int seed = 0; seed < 100000 && found < n; seed++) {
    char cand[8];
    snprintf(cand, sizeof cand, "c%d", seed);
    size_t bucket = map_hash(cand) & (cap - 1);
    if (found == 0) {
      target = bucket;
      strcpy(out[found++], cand);
    } else if (bucket == target) {
      strcpy(out[found++], cand);
    }
  }
  if (found < n) {
    fprintf(stderr, "FAIL could not find %d colliding keys\n", n);
    exit(1);
  }
}

/* The test that separates a working open-addressed map from a broken one.
 *
 * Three keys collide, so they occupy a probe run of three consecutive slots.
 * Deleting the middle one must leave a tombstone: if it is cleared to EMPTY
 * instead, the probe for the third key stops at that hole and reports the key
 * as missing even though it is still sitting in the table. */
static void test_delete_leaves_a_tombstone(void) {
  Map m;
  int v = 0;
  char k[3][8];
  map_init(&m);
  map_put(&m, "seed", 0); /* force the initial allocation so cap is known */
  colliding_keys(m.cap, k, 3);

  CHECK(map_put(&m, k[0], 10) == 0, "put first colliding key");
  CHECK(map_put(&m, k[1], 20) == 0, "put second colliding key");
  CHECK(map_put(&m, k[2], 30) == 0, "put third colliding key");

  CHECK(map_del(&m, k[1]) == 0, "delete the middle of the probe run");
  CHECK(map_get(&m, k[1], &v) == -1, "deleted key is gone");
  CHECK(map_get(&m, k[0], &v) == 0 && v == 10, "key before the hole is still found");
  CHECK(map_get(&m, k[2], &v) == 0 && v == 30, "key after the hole is still found");
  CHECK(map_len(&m) == 3, "len dropped by exactly one");

  CHECK(map_del(&m, k[1]) == -1, "deleting twice reports absent");
  CHECK(map_put(&m, k[1], 21) == 0, "the slot can be reused");
  CHECK(map_get(&m, k[1], &v) == 0 && v == 21, "reinserted key is found");
  map_free(&m);
}

/* Tombstones cost probe steps even though they hold nothing, so a table that
 * grows on live count alone degrades to a linear scan under churn. Growing on
 * live-plus-tombstones keeps capacity bounded. */
static void test_churn_does_not_explode(void) {
  enum { ROUNDS = 20000 };
  Map m;
  char key[32];
  map_init(&m);
  for (int i = 0; i < ROUNDS; i++) {
    snprintf(key, sizeof key, "churn-%d", i);
    CHECK(map_put(&m, key, i) == 0, "put during churn succeeds");
    CHECK(map_del(&m, key) == 0, "delete during churn succeeds");
  }
  CHECK(map_len(&m) == 0, "churn leaves nothing behind");
  CHECK(m.cap < 4096, "capacity stayed bounded under insert/delete churn");
  map_free(&m);
}

static void test_free_is_idempotent(void) {
  Map m;
  map_init(&m);
  map_put(&m, "a", 1);
  map_put(&m, "b", 2);
  map_free(&m);
  CHECK(m.slots == NULL && m.cap == 0 && m.len == 0, "free returns the zero state");
  map_free(&m); /* must not double-free */
  map_init(&m);
  CHECK(map_put(&m, "reuse", 5) == 0, "a freed map can be reused");
  map_free(&m);
}

int main(void) {
  test_init_allocates_nothing();
  test_put_get_overwrite();
  test_keys_are_copied();
  test_many_entries_survive_rehashing();
  test_delete_leaves_a_tombstone();
  test_churn_does_not_explode();
  test_free_is_idempotent();
  printf("ok  c/02-hash-map  %d checks passed\n", checks);
  return 0;
}
