/* pool.h - a fixed-size thread pool with a bounded work queue.
 *
 * The interface. pool.c implements it and main.c tests it.
 *
 * Determinism note: the ORDER in which tasks run is genuinely
 * nondeterministic, and nothing here asserts on it. What is deterministic, and
 * what the tests do assert on, is that every submitted task runs exactly once
 * and that every result is accounted for.
 */
#ifndef POOL_H
#define POOL_H

#include <stddef.h>

typedef struct ThreadPool ThreadPool;

/* A unit of work. The pool never looks inside `arg`. */
typedef void (*TaskFn)(void *arg);

/* Start `n_threads` workers with a queue holding at most `queue_cap` pending
 * tasks. NULL on failure. */
ThreadPool *pool_new(size_t n_threads, size_t queue_cap);

/* Submit work. Blocks while the queue is full, so a fast producer is throttled
 * by the workers rather than growing an unbounded backlog.
 *
 * Returns 0 on success, -1 if the pool is shutting down. */
int pool_submit(ThreadPool *p, TaskFn fn, void *arg);

/* Block until every task submitted so far has finished. The pool stays usable
 * afterwards. */
void pool_wait(ThreadPool *p);

/* Stop accepting work, finish what is already queued, join every worker, and
 * release the pool. */
void pool_destroy(ThreadPool *p);

/* Tasks completed so far. Only meaningful after pool_wait. */
size_t pool_completed(ThreadPool *p);

#endif /* POOL_H */
