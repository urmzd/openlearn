/* main.c - the test suite for the reference implementation.
 *
 * Nothing here asserts on the ORDER tasks run in: that is genuinely
 * nondeterministic and asserting on it would produce a flaky exercise. What is
 * deterministic is that every task runs exactly once, and that is what is
 * checked, mostly by having tasks write to disjoint slots so a lost or
 * duplicated run is visible without any locking in the test itself.
 */
#include "pool.h"

#include <pthread.h>
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

/* Each task owns one slot, so no synchronisation is needed to record that it
 * ran. A pool that runs a task twice leaves a 2 here; one that drops a task
 * leaves a 0. */
typedef struct {
  int *slot;
  int value;
} Marker;

static void mark(void *arg) {
  Marker *m = arg;
  (*m->slot) += m->value;
}

/* A shared counter guarded by its own mutex, to check that concurrent
 * increments from many workers all land. */
static pthread_mutex_t counter_lock = PTHREAD_MUTEX_INITIALIZER;
static long shared_counter = 0;

static void bump(void *arg) {
  long n = (long)(intptr_t)arg;
  /* Do a little real work so tasks actually overlap in time. */
  volatile long spin = 0;
  for (long i = 0; i < 1000; i++) spin += i;

  pthread_mutex_lock(&counter_lock);
  shared_counter += n;
  pthread_mutex_unlock(&counter_lock);
}

static void test_every_task_runs_exactly_once(void) {
  enum { N = 2000 };
  ThreadPool *p = pool_new(4, 16);
  CHECK(p != NULL, "pool created");

  int *slots = calloc(N, sizeof(int));
  Marker *markers = malloc(N * sizeof(Marker));
  for (int i = 0; i < N; i++) {
    markers[i].slot = &slots[i];
    markers[i].value = 1;
    CHECK(pool_submit(p, mark, &markers[i]) == 0, "submit succeeds");
  }

  pool_wait(p);

  for (int i = 0; i < N; i++) {
    CHECK(slots[i] == 1, "each task ran exactly once");
  }
  CHECK(pool_completed(p) == N, "the pool counted every task");

  pool_destroy(p);
  free(slots);
  free(markers);
}

/* The queue is far smaller than the submission count, so producers must block
 * and be woken as workers drain. A pool that mishandles the full condition
 * either deadlocks here or silently drops work. */
static void test_bounded_queue_blocks_the_producer(void) {
  enum { N = 3000 };
  ThreadPool *p = pool_new(3, 4); /* deliberately tiny queue */
  CHECK(p != NULL, "pool created");

  shared_counter = 0;
  for (int i = 0; i < N; i++) {
    CHECK(pool_submit(p, bump, (void *)(intptr_t)1) == 0, "submit succeeds under backpressure");
  }
  pool_wait(p);

  CHECK(shared_counter == N, "every increment landed despite a tiny queue");
  CHECK(pool_completed(p) == N, "and the pool agrees on the count");
  pool_destroy(p);
}

/* pool_wait must not return while a task is still executing. Tasks here take
 * a noticeable amount of time, so a wait that only checks the queue rather
 * than in-flight work will return early and see an incomplete total. */
static void test_wait_covers_in_flight_work(void) {
  enum { N = 64 };
  ThreadPool *p = pool_new(8, 64);
  CHECK(p != NULL, "pool created");

  shared_counter = 0;
  for (int i = 0; i < N; i++) {
    CHECK(pool_submit(p, bump, (void *)(intptr_t)7) == 0, "submit succeeds");
  }
  pool_wait(p);
  CHECK(shared_counter == N * 7, "wait returned only after every task finished");

  /* The pool must still be usable after a wait. */
  shared_counter = 0;
  for (int i = 0; i < N; i++) {
    CHECK(pool_submit(p, bump, (void *)(intptr_t)2) == 0, "submit after wait succeeds");
  }
  pool_wait(p);
  CHECK(shared_counter == N * 2, "a second round works too");
  CHECK(pool_completed(p) == N * 2, "completions accumulate across rounds");
  pool_destroy(p);
}

/* Destroy must drain what was accepted rather than abandoning it: a submit
 * that returned 0 is a promise that the task will run. */
static void test_destroy_drains_accepted_work(void) {
  enum { N = 500 };
  ThreadPool *p = pool_new(2, 256);
  CHECK(p != NULL, "pool created");

  int *slots = calloc(N, sizeof(int));
  Marker *markers = malloc(N * sizeof(Marker));
  for (int i = 0; i < N; i++) {
    markers[i].slot = &slots[i];
    markers[i].value = 1;
    CHECK(pool_submit(p, mark, &markers[i]) == 0, "submit succeeds");
  }

  /* No pool_wait: destroy alone must finish the backlog and join cleanly. */
  pool_destroy(p);

  for (int i = 0; i < N; i++) {
    CHECK(slots[i] == 1, "destroy ran every accepted task exactly once");
  }
  free(slots);
  free(markers);
}

/* Many workers and a single-slot queue is the configuration most likely to
 * expose a lost wakeup: nearly every worker is parked nearly all the time. */
static void test_many_workers_tiny_queue(void) {
  enum { N = 2000 };
  ThreadPool *p = pool_new(16, 1);
  CHECK(p != NULL, "pool created");

  shared_counter = 0;
  for (int i = 0; i < N; i++) {
    CHECK(pool_submit(p, bump, (void *)(intptr_t)1) == 0, "submit succeeds");
  }
  pool_wait(p);
  CHECK(shared_counter == N, "no task was lost to a missed wakeup");
  pool_destroy(p);
}

static void test_degenerate_configurations(void) {
  CHECK(pool_new(0, 8) == NULL, "a pool with no threads is rejected");
  CHECK(pool_new(4, 0) == NULL, "a pool with no queue is rejected");

  /* One worker, one slot: everything is serialised, which must still work. */
  ThreadPool *p = pool_new(1, 1);
  CHECK(p != NULL, "the minimal pool is valid");
  shared_counter = 0;
  for (int i = 0; i < 100; i++) {
    CHECK(pool_submit(p, bump, (void *)(intptr_t)3) == 0, "submit succeeds");
  }
  pool_wait(p);
  CHECK(shared_counter == 300, "the minimal pool runs everything");
  pool_destroy(p);

  /* A pool that is created and destroyed with no work at all must not hang. */
  ThreadPool *q = pool_new(4, 8);
  CHECK(q != NULL, "pool created");
  pool_wait(q); /* waiting on an idle pool returns immediately */
  pool_destroy(q);
  CHECK(1, "an unused pool tears down cleanly");
}

int main(void) {
  /* Repeat the whole suite: a race that shows up one run in ten is still a
   * bug, and a single pass proves very little about concurrent code. */
  for (int round = 0; round < 3; round++) {
    test_every_task_runs_exactly_once();
    test_bounded_queue_blocks_the_producer();
    test_wait_covers_in_flight_work();
    test_destroy_drains_accepted_work();
    test_many_workers_tiny_queue();
    test_degenerate_configurations();
  }
  printf("ok  c/07-thread-pool  %d checks passed\n", checks);
  return 0;
}
