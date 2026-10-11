/* Course tests for rt.03: c/src/runtime/pool.c against tinyllm/pool.h.
 *
 * Built twice by `ol check rt.03`: with ASan and UBSan (and the counting
 * allocator), then with ThreadSanitizer ([tests].sanitize = ["thread"]),
 * which reports any two threads touching the same memory without ordering
 * between them. Thread counts are fixed in every test (DESIGN 5.11).
 */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

/* -- a watchdog ---------------------------------------------------------------
 * A pool bug is often a deadlock, which would hang the run until the
 * harness's timeout. Every test starts WATCH(): if the test has not finished
 * 10 s later (the reference takes milliseconds), the binary reports it and
 * exits with a failure. */

static const char *watched = "";

static void on_alarm(int sig) {
    (void)sig;
    static const char msg[] = "FAIL (no progress in 10 s: a deadlock?) ";
    write(1, msg, sizeof msg - 1);
    write(1, watched, strlen(watched));
    write(1, "\n", 1);
    _exit(1);
}

#define WATCH()                                                                                     \
    do {                                                                                            \
        watched = __func__;                                                                         \
        signal(SIGALRM, on_alarm);                                                                  \
        alarm(10);                                                                                  \
    } while (0)

/* -- recording ranges -------------------------------------------------------- */

enum { MAXR = 4096 };
typedef struct {
    _Atomic int n;
    int64_t lo[MAXR], hi[MAXR];
    int worker[MAXR];
    int n_threads;
    _Atomic int bad_worker;
} record_t;

static void record(void *ctx, int64_t lo, int64_t hi, int worker) {
    record_t *r = ctx;
    int i = atomic_fetch_add(&r->n, 1);
    if (i < MAXR) {
        r->lo[i] = lo;
        r->hi[i] = hi;
        r->worker[i] = worker;
    }
    if (worker < 0 || worker >= r->n_threads) atomic_store(&r->bad_worker, 1);
}

static int cmp_lo(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

/* The ranges seen, sorted by lo, must be exactly [k g, min((k+1) g, n)). */
static int exact_partition(record_t *r, int64_t n, int64_t grain) {
    int m = atomic_load(&r->n);
    int64_t want = n == 0 ? 0 : (n - 1) / grain + 1;
    if (m != want || m > MAXR) return 0;
    int64_t pairs[MAXR][2];
    for (int i = 0; i < m; i++) {
        pairs[i][0] = r->lo[i];
        pairs[i][1] = r->hi[i];
    }
    qsort(pairs, (size_t)m, sizeof pairs[0], cmp_lo);
    for (int k = 0; k < m; k++) {
        int64_t lo = k * grain, hi = lo + grain < n ? lo + grain : n;
        if (pairs[k][0] != lo || pairs[k][1] != hi) return 0;
    }
    return 1;
}

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example: n = 10, grain = 4 is three ranges,
     *      [0, 4), [4, 8), [8, 10), whatever the number of threads; with
     *      p = NULL they run in order on the caller as worker 0.
     * KIND: unit, smoke
     * CATCHES: s02, s03, m001, m002, m003
     * CHAPTER: rt.03 section 3 */
    WATCH();
    static record_t r;
    memset(&r, 0, sizeof r);
    r.n_threads = 1;
    OL_EQ(tl_parallel_for(NULL, 10, 4, record, &r), TL_OK);
    OL_EQ(atomic_load(&r.n), 3);
    OL_EQ(r.lo[0], 0);
    OL_EQ(r.hi[0], 4);
    OL_EQ(r.lo[1], 4);
    OL_EQ(r.hi[1], 8);
    OL_EQ(r.lo[2], 8);
    OL_EQ(r.hi[2], 10);
    for (int i = 0; i < 3; i++) OL_EQ(r.worker[i], 0);

    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(4, &p), TL_OK);
    OL_EQ(tl_pool_threads(p), 4);
    memset(&r, 0, sizeof r);
    r.n_threads = 4;
    OL_EQ(tl_parallel_for(p, 10, 4, record, &r), TL_OK);
    OL_TRUE(exact_partition(&r, 10, 4));
    OL_EQ(atomic_load(&r.bad_worker), 0);
    tl_pool_destroy(p);
}

OL_TEST(grain_edges_and_empty_range) {
    /* WHY: the partition at its edges: grain 1 (one index per range),
     *      grain = n (one range), grain > n (one short range), n not a
     *      multiple of grain (a short last range), n = 0 (no call at all),
     *      on 1 and 3 threads.
     * KIND: boundary
     * CATCHES: s02, s03, s06, m001, m002
     * CHAPTER: rt.03 section 2.2 */
    WATCH();
    static record_t r;
    static const int64_t cases[][2] = {{7, 1}, {7, 7}, {7, 100}, {1000, 7}, {1, 1}, {0, 5}};
    for (int t = 1; t <= 3; t += 2) {
        tl_pool *p = NULL;
        OL_EQ(tl_pool_create(t, &p), TL_OK);
        for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            memset(&r, 0, sizeof r);
            r.n_threads = t;
            OL_EQ(tl_parallel_for(p, cases[c][0], cases[c][1], record, &r), TL_OK);
            if (!exact_partition(&r, cases[c][0], cases[c][1])) {
                tl_pool_destroy(p);
                OL_FAIL("ranges differ from [k*grain, min((k+1)*grain, n))");
            }
        }
        tl_pool_destroy(p);
    }
}

OL_TEST(bad_arguments) {
    /* WHY: the contract's errors, before any range runs: n < 0, grain < 1,
     *      fn == NULL, n_threads < 0, out == NULL. Each sets the error slot.
     * KIND: boundary
     * CATCHES: s04, m004
     * CHAPTER: rt.03 section 4 */
    WATCH();
    static record_t r;
    memset(&r, 0, sizeof r);
    r.n_threads = 2;
    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(2, &p), TL_OK);
    OL_EQ(tl_parallel_for(p, -1, 1, record, &r), TL_EINVAL);
    OL_EQ(tl_parallel_for(p, 5, 0, record, &r), TL_EINVAL);
    OL_EQ(tl_parallel_for(NULL, 5, 0, record, &r), TL_EINVAL);
    tl_set_last_error("");
    OL_EQ(tl_parallel_for(p, 5, 1, NULL, &r), TL_EINVAL);
    OL_TRUE(strstr(tl_last_error(), "tl_parallel_for") != NULL);
    OL_EQ(atomic_load(&r.n), 0);
    tl_pool_destroy(p);
    tl_pool *q = (tl_pool *)&r;
    OL_EQ(tl_pool_create(-1, &q), TL_EINVAL);
    OL_EQ(tl_pool_create(2, NULL), TL_EINVAL);
    OL_EQ(tl_pool_threads(NULL), 1);
    tl_pool_destroy(NULL);
}

/* -- every index exactly once ----------------------------------------------- */

typedef struct {
    _Atomic uint32_t *hits;
} bitmap_t;

static void mark_hits(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)worker;
    bitmap_t *b = ctx;
    for (int64_t i = lo; i < hi; i++) atomic_fetch_add(&b->hits[i], 1u);
}

static tl_pool *shared_pool = NULL;

static int each_index_once(ol_gen *g, int size) {
    int64_t n = ol_gen_int(g, 0, 50 * (int64_t)size + 1);
    int64_t grain = ol_gen_int(g, 1, 64);
    _Atomic uint32_t *hits = calloc((size_t)n + 1, sizeof *hits);
    bitmap_t b = {hits};
    int ok = tl_parallel_for(shared_pool, n, grain, mark_hits, &b) == TL_OK;
    for (int64_t i = 0; i < n && ok; i++) ok = atomic_load(&hits[i]) == 1u;
    free(hits);
    return ok;
}

OL_TEST(each_index_runs_exactly_once) {
    /* WHY: the property every kernel relies on: across 200 seeded (n, grain)
     *      pairs on an 8-thread pool, each index is visited exactly once (an
     *      atomic counter per index), never 0 times and never twice, which is
     *      what a racy "next range" counter gets wrong.
     * KIND: property
     * CATCHES: s01, s03, s05, s06
     * CHAPTER: rt.03 section 2.3 */
    WATCH();
    OL_EQ(tl_pool_create(8, &shared_pool), TL_OK);
    OL_CHECK_PROP(each_index_once, 200, 200);
    tl_pool_destroy(shared_pool);
    shared_pool = NULL;
}

/* -- a pooled kernel is bitwise independent of the thread count ------------- */

enum { M = 37, N = 29, K = 211 };
typedef struct {
    const float *A, *B;
    float *C;
} mm_t;

/* Rows [lo, hi) of C = A @ B, each element summed over k in one fixed order:
 * nothing is reduced across ranges. */
static void mm_rows(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)worker;
    mm_t *m = ctx;
    for (int64_t i = lo; i < hi; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += m->A[i * K + k] * m->B[k * N + j];
            m->C[i * N + j] = s;
        }
}

OL_TEST(pooled_matmul_is_bitwise_serial) {
    /* WHY: a float32 sum depends on its order. Because the partition is
     *      fixed and each output element is reduced inside one range, the
     *      pooled product equals the serial one bit for bit on 1, 2, 3, and
     *      8 threads and with grain 1 or 5. This is how L9.1 stays
     *      batch-invariant (c/ABI.md rule 10).
     * KIND: differential
     * CATCHES: s03
     * CHAPTER: rt.03 section 2.4 */
    WATCH();
    static float A[M * K], B[K * N], C0[M * N], C1[M * N];
    ol_gen g = {ol_prop_base_seed() + 31};
    for (int i = 0; i < M * K; i++) A[i] = ol_gen_f32(&g, -1.0f, 1.0f);
    for (int i = 0; i < K * N; i++) B[i] = ol_gen_f32(&g, -1.0f, 1.0f);
    mm_t ser = {A, B, C0};
    OL_EQ(tl_parallel_for(NULL, M, 1, mm_rows, &ser), TL_OK);
    static const int threads[] = {1, 2, 3, 8};
    for (int t = 0; t < 4; t++)
        for (int64_t grain = 1; grain <= 5; grain += 4) {
            tl_pool *p = NULL;
            OL_EQ(tl_pool_create(threads[t], &p), TL_OK);
            memset(C1, 0xFF, sizeof C1);
            mm_t par = {A, B, C1};
            OL_EQ(tl_parallel_for(p, M, grain, mm_rows, &par), TL_OK);
            tl_pool_destroy(p);
            OL_EQ(memcmp(C0, C1, sizeof C0), 0);
        }
}

/* -- workers really run in parallel, and worker ids are per thread ---------- */

typedef struct {
    _Atomic int arrived;
    int n_threads;
    _Atomic int timed_out;
    _Atomic int seen[16];
} barrier_t;

static void wait_for_all(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)lo;
    (void)hi;
    barrier_t *b = ctx;
    atomic_fetch_add(&b->seen[worker], 1);
    atomic_fetch_add(&b->arrived, 1);
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (atomic_load(&b->arrived) < b->n_threads) { /* spin until every range started */
        clock_gettime(CLOCK_MONOTONIC, &t);
        if (t.tv_sec - t0.tv_sec > 5) {
            atomic_store(&b->timed_out, 1);
            return;
        }
    }
}

OL_TEST(all_workers_run_concurrently) {
    /* WHY: a "pool" that runs every range on the caller passes the partition
     *      tests but gives no speedup. Here n = 4 ranges on 4 threads each
     *      wait until all 4 have started: that finishes only if 4 threads run
     *      at the same time, and then each worker id 0..3 is seen once.
     * KIND: unit
     * CATCHES: s07, s08, m002
     * CHAPTER: rt.03 section 2.3 */
    WATCH();
    static barrier_t b;
    memset(&b, 0, sizeof b);
    b.n_threads = 4;
    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(4, &p), TL_OK);
    OL_EQ(tl_parallel_for(p, 4, 1, wait_for_all, &b), TL_OK);
    tl_pool_destroy(p);
    OL_EQ(atomic_load(&b.timed_out), 0);
    for (int w = 0; w < 4; w++) OL_EQ(atomic_load(&b.seen[w]), 1);
}

/* -- nesting, serialization, destroy ---------------------------------------- */

typedef struct {
    tl_pool *p;
    _Atomic int busy, other;
} nest_t;

static void noop(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)ctx;
    (void)lo;
    (void)hi;
    (void)worker;
}

static void call_inside(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)lo;
    (void)hi;
    (void)worker;
    nest_t *s = ctx;
    if (tl_parallel_for(s->p, 4, 1, noop, NULL) == TL_EBUSY) atomic_fetch_add(&s->busy, 1);
    if (tl_parallel_for(NULL, 4, 1, noop, NULL) == TL_OK) atomic_fetch_add(&s->other, 1);
}

OL_TEST(nested_call_is_busy_not_a_deadlock) {
    /* WHY: a range function that calls tl_parallel_for on its own pool would
     *      wait for workers that are busy running it: a deadlock. The
     *      contract returns TL_EBUSY instead, from every worker, including
     *      a 1-thread pool; a serial call (p = NULL) inside is fine.
     * KIND: fault
     * CATCHES: s02, s09, s10
     * CHAPTER: rt.03 section 5, Pitfalls */
    WATCH();
    static const int threads[] = {1, 3};
    for (int t = 0; t < 2; t++) {
        nest_t s;
        memset(&s, 0, sizeof s);
        OL_EQ(tl_pool_create(threads[t], &s.p), TL_OK);
        OL_EQ(tl_parallel_for(s.p, 6, 1, call_inside, &s), TL_OK);
        tl_pool_destroy(s.p);
        OL_EQ(atomic_load(&s.busy), 6);
        OL_EQ(atomic_load(&s.other), 6);
    }
}

typedef struct {
    tl_pool *p;
    _Atomic long *total;
    _Atomic int started;
} caller_t;

static void add_one(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)worker;
    _Atomic long *total = ctx;
    atomic_fetch_add(total, (long)(hi - lo));
}

static void *call_many(void *arg) {
    caller_t *c = arg;
    atomic_store(&c->started, 1);
    for (int i = 0; i < 50; i++)
        if (tl_parallel_for(c->p, 1000, 7, add_one, c->total) != TL_OK) return (void *)1;
    return NULL;
}

OL_TEST(two_callers_are_serialized) {
    /* WHY: the Rust engine's step loop and a benchmark thread may share one
     *      pool. Two threads each make 50 calls of 1000 indices; every call
     *      must complete in full (the counter ends at 100000) and
     *      ThreadSanitizer must see no race on the pool's job.
     * KIND: fault
     * CATCHES: s01, s06, s11
     * CHAPTER: rt.03 section 2.5 */
    WATCH();
    _Atomic long total = 0;
    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(4, &p), TL_OK);
    caller_t a = {p, &total, 0}, b = {p, &total, 0};
    pthread_t ta, tb;
    OL_EQ(pthread_create(&ta, NULL, call_many, &a), 0);
    OL_EQ(pthread_create(&tb, NULL, call_many, &b), 0);
    void *ra = NULL, *rb = NULL;
    pthread_join(ta, &ra);
    pthread_join(tb, &rb);
    tl_pool_destroy(p);
    OL_TRUE(ra == NULL && rb == NULL);
    OL_EQ(atomic_load(&total), 100000);
}

typedef struct {
    tl_pool *p;
    _Atomic int inside, done;
} slow_t;

static void slow_range(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)lo;
    (void)hi;
    slow_t *s = ctx;
    /* The calling thread (worker 0) is still inside its range long after
     * the other workers have run out of ranges. */
    struct timespec d = {0, worker == 0 ? 100000000 : 2000000}; /* 100 ms : 2 ms */
    if (worker == 0) atomic_store(&s->inside, 1);
    nanosleep(&d, NULL);
    atomic_fetch_add(&s->done, 1);
}

static void *run_slow(void *arg) {
    slow_t *s = arg;
    return (void *)(intptr_t)tl_parallel_for(s->p, 16, 1, slow_range, s);
}

OL_TEST(destroy_waits_for_running_work) {
    /* WHY: shutting the engine down while a step is still running must not
     *      free the pool under the call. destroy, called while another
     *      thread's tl_parallel_for is in flight (its calling thread still
     *      inside a 100 ms range after the workers ran out of ranges),
     *      returns only after all 16 ranges have finished. Joining the
     *      workers is not enough: the caller still holds the pool.
     * KIND: fault
     * CATCHES: s12, s13
     * CHAPTER: rt.03 section 2.5 */
    WATCH();
    slow_t s;
    memset(&s, 0, sizeof s);
    OL_EQ(tl_pool_create(4, &s.p), TL_OK);
    pthread_t t;
    OL_EQ(pthread_create(&t, NULL, run_slow, &s), 0);
    while (!atomic_load(&s.inside)) { /* until the call is in flight */
        struct timespec d = {0, 100000};
        nanosleep(&d, NULL);
    }
    tl_pool_destroy(s.p);
    OL_EQ(atomic_load(&s.done), 16);
    void *rv = NULL;
    pthread_join(t, &rv);
    OL_EQ((intptr_t)rv, TL_OK);
}

OL_TEST(create_failure_leaks_nothing) {
    /* WHY: tl_pool_create allocates the pool, the thread table, and the
     *      table of thread arguments, all on the calling thread. Failing the
     *      n-th allocation for every n gives TL_ENOMEM, out = NULL, and no
     *      leaked block (the counting allocator); a success still works.
     * KIND: fault
     * CATCHES: s14, m005
     * CHAPTER: rt.03 section 4 */
    WATCH();
    for (long n = 0; n < 12; n++) {
        ol_alloc_fail_after(n);
        tl_pool *p = (tl_pool *)&n;
        tl_status st = tl_pool_create(6, &p);
        ol_alloc_fail_after(-1);
        if (st == TL_OK) {
            OL_EQ(tl_parallel_for(p, 100, 3, noop, NULL), TL_OK);
            tl_pool_destroy(p);
            continue;
        }
        OL_EQ(st, TL_ENOMEM);
        OL_TRUE(p == NULL);
    }
}

OL_TEST(hardware_concurrency_default) {
    /* WHY: n_threads = 0 asks for the machine's core count; the pool must
     *      report what it resolved (at least 1) and work. The engine uses it
     *      when [engine].threads is unset.
     * KIND: unit
     * CATCHES: s15
     * CHAPTER: rt.03 section 4 */
    WATCH();
    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(0, &p), TL_OK);
    OL_TRUE(tl_pool_threads(p) >= 1);
    static record_t r;
    memset(&r, 0, sizeof r);
    r.n_threads = tl_pool_threads(p);
    OL_EQ(tl_parallel_for(p, 100, 9, record, &r), TL_OK);
    OL_TRUE(exact_partition(&r, 100, 9));
    OL_EQ(atomic_load(&r.bad_worker), 0);
    tl_pool_destroy(p);
}

int main(void) { return OL_RUN_ALL(); }
