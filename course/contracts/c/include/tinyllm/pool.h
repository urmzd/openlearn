/* tinyllm/pool.h (rt.03): a fixed pthread pool and a blocking parallel for.
 * chapter: content/courses/build-an-llm/10-kernels/02-thread-pool-and-parallel-for.md
 * Rules in c/ABI.md.
 *
 * The partition is deterministic: tl_parallel_for(p, n, grain, fn, ctx)
 * calls fn exactly once for each range
 *     [k * grain, min((k + 1) * grain, n))   for k = 0 .. ceil(n / grain) - 1
 * whatever the number of threads. Which worker runs a range, and in which
 * order ranges run, is unspecified. A kernel that reduces within one range
 * and never across ranges is therefore bitwise identical with 1 or 8
 * threads (the rt.03 equivalence test and batch invariance, c/ABI.md rule
 * 10).
 *
 * module: rt.03 (c/src/runtime/pool.c) */
#ifndef TINYLLM_POOL_H
#define TINYLLM_POOL_H

#include <stdint.h>

#include "tinyllm/abi.h"


/* Also forward-declared by matmul.h and attention.h; C11 allows the
 * identical typedef more than once. */
typedef struct tl_pool tl_pool;

/* The range function. worker is in [0, tl_pool_threads(p)) and is stable
 * for one call, so it can index per-worker scratch. The only callback that
 * the ABI allows; it never crosses into another language. */
typedef void (*tl_range_fn)(void *ctx, int64_t lo, int64_t hi, int worker);

/* n_threads workers in total, counting the calling thread, which also runs
 * ranges (so n_threads == 1 starts no pthread). 0 means the hardware
 * concurrency. TL_EINVAL for n_threads < 0 or out == NULL; TL_ENOMEM when
 * the hook fails; TL_EIO when pthread_create fails (nothing leaks). */
tl_status tl_pool_create(int n_threads, tl_pool **out);

/* Runs the partition above and returns when every range has finished.
 * p == NULL runs the ranges serially, in order, on the calling thread with
 * worker 0. n == 0 calls nothing. TL_EINVAL for n < 0, grain < 1, or
 * fn == NULL. Calls from several threads on one pool are serialized: the
 * second waits for the first. A call on p from inside one of p's range
 * functions returns TL_EBUSY (no nested parallelism). */
tl_status tl_parallel_for(tl_pool *p, int64_t n, int64_t grain, tl_range_fn fn, void *ctx);

/* The number of workers: n_threads as resolved by create; 1 for NULL. */
int tl_pool_threads(const tl_pool *p);

/* Joins every thread and frees the pool. Waits for a running
 * tl_parallel_for to finish first. NULL does nothing. */
void tl_pool_destroy(tl_pool *p);


#endif /* TINYLLM_POOL_H */
