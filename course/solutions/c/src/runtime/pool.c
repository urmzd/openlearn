/* c/src/runtime/pool.c (rt.03): a fixed pthread pool and a blocking
 * parallel for. Contract: tinyllm/pool.h. Rules: c/ABI.md (pthreads only:
 * <threads.h> is missing on macOS). Callers: L9.1's matmul (row blocks),
 * L9.3's attention (heads). This pool is never linked into the Rust engine.
 *
 * The pool has n workers: the calling thread is worker 0 and n - 1 pthreads
 * are workers 1 .. n - 1. tl_parallel_for publishes one job (n, grain, fn,
 * ctx) under the mutex, wakes the threads, and joins in. Every participant
 * takes range indices from one atomic counter, k = next++, and runs range k,
 * [k * grain, min((k + 1) * grain, n)), until the counter passes the number
 * of ranges. The ranges are therefore always the same, whoever runs them:
 * that is what makes a kernel that never reduces across ranges bitwise
 * identical with 1 or 8 threads.
 *
 * Completion: each thread that woke for a job decrements `active` when it
 * finds no range left; the caller waits until active == 0, so no thread
 * still reads the job when tl_parallel_for returns and the next job may
 * overwrite it. `call_lock` serializes callers from different threads, and
 * a thread-local pointer to the pool a thread is running ranges for turns a
 * nested call on the same pool into TL_EBUSY instead of a deadlock.
 *
 * The struct below is fixed; the functions that follow maintain it.
 */
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "tinyllm/pool.h"

struct tl_pool {
    int n_threads;            /* workers, counting the caller */
    int n_started;            /* pthreads actually running (n_threads - 1 once created) */
    pthread_t *threads;       /* [n_threads - 1] */
    struct tl_pool_arg *args; /* [n_threads - 1]: what each pthread is started with */
    pthread_mutex_t mu;       /* guards everything below except `next` */
    pthread_cond_t wake;      /* workers wait here for a new job or stop */
    pthread_cond_t idle;      /* the caller waits here for active == 0 */
    pthread_mutex_t call_lock;/* one tl_parallel_for at a time */
    uint64_t generation;      /* bumped once per job */
    int stop;                 /* set by destroy */
    int active;               /* threads that woke for the current job and are not done */
    /* the current job */
    int64_t n, grain, n_ranges;
    tl_range_fn fn;
    void *ctx;
    _Atomic int64_t next;     /* the next range index to hand out */
};

typedef struct tl_pool_arg {
    tl_pool *pool;
    int worker;
} tl_pool_arg;

/* The pool whose ranges this thread is running, or NULL. */
static _Thread_local tl_pool *running_for = NULL;

/* Takes range indices until none is left. */
static void run_ranges(tl_pool *p, int worker) {
    for (;;) {
        int64_t k = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
        if (k >= p->n_ranges) return;
        int64_t lo = k * p->grain;
        int64_t hi = lo + p->grain < p->n ? lo + p->grain : p->n;
        p->fn(p->ctx, lo, hi, worker);
    }
}

static void *worker_main(void *arg) {
    tl_pool_arg a = *(tl_pool_arg *)arg;
    tl_pool *p = a.pool;
    running_for = p;
    uint64_t seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->stop && p->generation == seen) pthread_cond_wait(&p->wake, &p->mu);
        if (p->stop) break;
        seen = p->generation;
        pthread_mutex_unlock(&p->mu);
        run_ranges(p, a.worker);
        pthread_mutex_lock(&p->mu);
        if (--p->active == 0) pthread_cond_signal(&p->idle);
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

/* Stops and joins the first n_started threads, then frees everything. */
static void shutdown_pool(tl_pool *p) {
    pthread_mutex_lock(&p->mu);
    p->stop = 1;
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->n_started; i++) pthread_join(p->threads[i], NULL);
    pthread_cond_destroy(&p->wake);
    pthread_cond_destroy(&p->idle);
    pthread_mutex_destroy(&p->mu);
    pthread_mutex_destroy(&p->call_lock);
    tl_free(p->threads);
    tl_free(p->args);
    tl_free(p);
}

tl_status tl_pool_create(int n_threads, tl_pool **out) {
    if (out == NULL || n_threads < 0) {
        tl_set_last_error(out == NULL ? "tl_pool_create: out is NULL" : "tl_pool_create: n_threads < 0");
        return TL_EINVAL;
    }
    *out = NULL;
    if (n_threads == 0) {
        long hw = 1;
#if defined(_SC_NPROCESSORS_ONLN)
        hw = sysconf(_SC_NPROCESSORS_ONLN);
#endif
        n_threads = hw > 0 ? (int)(hw < 1024 ? hw : 1024) : 1;
    }
    tl_pool *p = tl_alloc(sizeof *p, _Alignof(tl_pool));
    if (p == NULL) {
        tl_set_last_error("tl_pool_create: the allocator hook returned NULL");
        return TL_ENOMEM;
    }
    memset(p, 0, sizeof *p);
    p->n_threads = n_threads;
    atomic_init(&p->next, 0);
    if (n_threads > 1) {
        /* Every allocation happens here, on the calling thread: the hook
         * need not be thread-safe. */
        p->threads = tl_alloc((size_t)(n_threads - 1) * sizeof *p->threads, _Alignof(pthread_t));
        p->args = p->threads ? tl_alloc((size_t)(n_threads - 1) * sizeof *p->args, _Alignof(tl_pool_arg)) : NULL;
        if (p->args == NULL) {
            tl_free(p->threads);
            tl_free(p);
            tl_set_last_error("tl_pool_create: the allocator hook returned NULL");
            return TL_ENOMEM;
        }
    }
    pthread_mutex_init(&p->mu, NULL);
    pthread_mutex_init(&p->call_lock, NULL);
    pthread_cond_init(&p->wake, NULL);
    pthread_cond_init(&p->idle, NULL);
    for (int i = 1; i < n_threads; i++) {
        p->args[i - 1] = (tl_pool_arg){p, i};
        if (pthread_create(&p->threads[i - 1], NULL, worker_main, &p->args[i - 1]) != 0) {
            shutdown_pool(p);
            tl_set_last_error("tl_pool_create: pthread_create failed");
            return TL_EIO;
        }
        p->n_started++;
    }
    *out = p;
    return TL_OK;
}

tl_status tl_parallel_for(tl_pool *p, int64_t n, int64_t grain, tl_range_fn fn, void *ctx) {
    if (n < 0 || grain < 1 || fn == NULL) {
        tl_set_last_error(fn == NULL ? "tl_parallel_for: fn is NULL"
                                     : "tl_parallel_for: need n >= 0 and grain >= 1");
        return TL_EINVAL;
    }
    if (n == 0) return TL_OK;
    int64_t n_ranges = (n - 1) / grain + 1; /* ceil(n / grain) without overflow */
    if (p == NULL) {
        for (int64_t k = 0; k < n_ranges; k++) {
            int64_t lo = k * grain;
            fn(ctx, lo, lo + grain < n ? lo + grain : n, 0);
        }
        return TL_OK;
    }
    if (running_for == p) {
        tl_set_last_error("tl_parallel_for: nested call on the same pool");
        return TL_EBUSY;
    }
    pthread_mutex_lock(&p->call_lock);
    tl_pool *outer = running_for; /* a range of another pool may call us */
    if (p->n_started == 0) {
        running_for = p;
        for (int64_t k = 0; k < n_ranges; k++) {
            int64_t lo = k * grain;
            fn(ctx, lo, lo + grain < n ? lo + grain : n, 0);
        }
        running_for = outer;
        pthread_mutex_unlock(&p->call_lock);
        return TL_OK;
    }
    pthread_mutex_lock(&p->mu);
    p->n = n;
    p->grain = grain;
    p->n_ranges = n_ranges;
    p->fn = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    p->active = p->n_started;
    p->generation++;
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mu);

    running_for = p;
    run_ranges(p, 0);
    running_for = outer;

    pthread_mutex_lock(&p->mu);
    while (p->active > 0) pthread_cond_wait(&p->idle, &p->mu);
    pthread_mutex_unlock(&p->mu);
    pthread_mutex_unlock(&p->call_lock);
    return TL_OK;
}

int tl_pool_threads(const tl_pool *p) {
    return p == NULL ? 1 : p->n_threads;
}

void tl_pool_destroy(tl_pool *p) {
    if (p == NULL) return;
    /* A tl_parallel_for running on another thread holds call_lock until it
     * has finished: taking it here waits for that call. */
    pthread_mutex_lock(&p->call_lock);
    pthread_mutex_unlock(&p->call_lock);
    shutdown_pool(p);
}
