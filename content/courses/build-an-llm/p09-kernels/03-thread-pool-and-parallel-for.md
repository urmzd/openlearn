# Thread pool and tl_parallel_for (optional C)

## Overview

| | |
|---|---|
| **Module** | `rt.03` · side · C · Pass 6 |
| **Solution** | [`pool.c`](../../../../course/solutions/c/src/runtime/pool.c): `tl_pool_create`, `tl_parallel_for`, `tl_pool_threads`, `tl_pool_destroy`, and the helpers `run_ranges`, `worker_main`, `shutdown_pool` |
| **Contract** | [`course/contracts/c/include/tinyllm/pool.h`](../../../../course/contracts/c/include/tinyllm/pool.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) (pthreads only; batch invariance, rule 10) |
| **Needs** | [`rt.02` runtime support](02-arena-allocator-with-marks.md): `tl_alloc`, `tl_free`, `tl_set_last_error`. Reading: [`lang.03` C](../../programming-languages/02-language-and-tool-primers/03-c.md) |
| **Used by** | `L9.1`, `L9.3`, `L9.4`, and `L9.5` use the pool in their optional standalone C implementations, which together form the optional C module group of the kernels pass |
| **Optional depth** | Herlihy and Shavit, *The Art of Multiprocessor Programming*, ch. 16 (work distribution); Butenhof, *Programming with POSIX Threads*, ch. 3 and 7; the OpenMP specification, `schedule(dynamic, chunk)` |

## Key Takeaways

- **The partition is part of the contract**: ranges are always $[kg, \min((k+1)g, n))$, whoever runs them, so a kernel that reduces only inside a range gives the same bits with 1 or 8 threads.
- **One atomic counter hands out ranges**: `atomic_fetch_add` gives each range to exactly one thread; a read followed by a write gives some ranges twice.
- **"Done" means every thread that woke has finished**, not that the counter ran out; returning early lets the next job overwrite one still being read.
- **A nested call on the same pool is `TL_EBUSY`, not a deadlock**, and calls from two threads are serialized.
- **Destroy waits for a running call** and joins every thread; a failed create leaks nothing.

## How to read this chapter

Sections 1 to 3 build the pool from first principles (threads and their synchronisation, the fixed partition, handing out ranges, finishing a job safely) and work one partition by hand. Section 4 fixes the interface. Section 5 walks through the solution, and explains why each choice is the right one. Concurrency bugs rarely show in a single run, so the cases in section 4 are meant to be run under ThreadSanitizer as well as AddressSanitizer, with a watchdog that turns a deadlock into a failure instead of a hang.

---

## 1. Why now

The C matmul of M03.1 runs on one core. The optional tiled kernel of L9.1 can split independent columns across workers; attention (L9.3) splits heads, and other optional C exercises parallelize rows or sequences. Each implementation still owns a fixed reduction order, so scheduling changes which worker computes a result but does not change the result's bits. Starting threads per call costs tens of microseconds, so these standalone C exercises keep a pool that sleeps between calls. The production Rust engine owns its own Rust thread pool and independent kernels.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n$ | the number of indices to process, $n \ge 0$ | `int64_t` |
| $g$ | the grain: indices per range, $g \ge 1$ | `int64_t` |
| $R = \lceil n / g \rceil$ | the number of ranges | `int64_t` |
| $k$ | a range index, $0 \le k < R$ | `int64_t` |
| $T$ | the number of workers, counting the caller | `int` |
| $w$ | a worker id, $0 \le w < T$; the caller is 0 | `int` |

### 2.1 Threads, mutexes, condition variables, atomics

A **thread** runs a function concurrently with the others in the same address space (`pthread_create`, `pthread_join`; c/ABI.md rule 8: pthreads only, since `<threads.h>` is missing on macOS). Two threads touching the same memory, at least one writing, with nothing ordering the two accesses, is a **data race**, which is undefined behavior in C11. Three tools order accesses: a **mutex** (`pthread_mutex_lock`/`unlock`: one holder at a time, and everything written before an unlock is visible after the next lock), a **condition variable** (`pthread_cond_wait` releases the mutex and sleeps until another thread signals; always re-check the condition in a loop, because wakeups can be spurious), and an **atomic** (`<stdatomic.h>`: `atomic_fetch_add` reads, adds, and writes as one indivisible step). ThreadSanitizer (`-fsanitize=thread`) instruments every access and reports any pair not ordered by these tools.

### 2.2 The partition

`tl_parallel_for(p, n, g, fn, ctx)` calls `fn(ctx, lo, hi, w)` once for each range

$$[\,k g,\ \min((k + 1) g,\ n)\,) \quad \text{for } k = 0, 1, \dots, R - 1,\quad R = \lceil n/g \rceil = \lfloor (n - 1)/g \rfloor + 1 \ (n > 0).$$

The second form of $R$ avoids overflow in $n + g - 1$. The last range is short when $g$ does not divide $n$. Nothing about the ranges depends on $T$: only which worker runs which range, and in which order, does. A kernel that writes each output element inside one range, from a reduction in a fixed order, is therefore **bitwise identical** for every $T$. The obvious alternative, "split $n$ into $T$ equal chunks", changes the ranges with $T$ and breaks this.

### 2.3 Handing out ranges

The pool has $T$ workers: the caller is worker 0 and $T - 1$ pthreads are workers $1 \dots T - 1$. A job publishes $(n, g, R, \mathit{fn}, \mathit{ctx})$ and resets an atomic counter `next` to 0. Every participant then loops:

```c
for (;;) { int64_t k = atomic_fetch_add(&next, 1); if (k >= R) break; run range k; }
```

`atomic_fetch_add` returns each value of the counter to exactly one caller, so each range runs exactly once, and fast workers simply take more ranges (dynamic scheduling, which balances uneven ranges). Writing it as `k = next; next = k + 1;` lets two threads read the same $k$: a race, and a range run twice (ThreadSanitizer reports it, and a per-index counter sees a 2).

### 2.4 Waking, finishing, and reusing the job

Workers sleep on a condition variable `wake` and wait for a **generation** number to change. `tl_parallel_for` locks the mutex, writes the job, sets `active` to the number of pthreads, bumps the generation, broadcasts, unlocks, and runs ranges itself. Each pthread that woke runs ranges until the counter passes $R$, then, under the mutex, decrements `active`; the one that reaches 0 signals `idle`. The caller returns only after waiting for `active == 0`.

Why wait for `active`, when the counter alone says no range is left? Because "no range left to hand out" is not "no range still running", and a worker that took the last range may still be inside `fn`. Worse, a worker that is between waking and reading the job would read the **next** job's fields if the caller returned and was called again. Waiting for every woken thread to check in is what makes the job struct safe to overwrite.

### 2.5 Nested calls, concurrent callers, destroy

**Nested.** If `fn` calls `tl_parallel_for` on the same pool, the inner call would wait for workers that are busy running the outer call's ranges, forever. A thread-local pointer records which pool's ranges the current thread is running; a call on that pool returns `TL_EBUSY`. A 1-thread pool has no pthreads but must refuse too, because the contract does not depend on $T$. A call with `p = NULL` (serial) inside a range is fine.

**Concurrent callers.** Two threads calling on one pool would overwrite each other's job. A second mutex, `call_lock`, held for the whole call, makes the second wait for the first.

**Destroy.** `tl_pool_destroy` first takes and releases `call_lock`, which waits for a call running on another thread to return; then it sets `stop`, broadcasts, joins every pthread, destroys the mutexes and condition variables, and frees through the hook. Joining the workers alone is not enough: the caller of the running call (worker 0) still reads the pool after the workers have run out of ranges.

**Create.** All allocations (the pool, the thread table, the thread arguments) happen on the creating thread, so the allocator hook never needs to be thread-safe. If a `pthread_create` fails, the threads already started are stopped and joined, everything is freed, and the result is `TL_EIO`; a failed allocation gives `TL_ENOMEM`. `n_threads == 0` means the hardware concurrency, `sysconf(_SC_NPROCESSORS_ONLN)`.

## 3. Worked example by hand

$n = 10$, $g = 4$: $R = \lfloor 9 / 4 \rfloor + 1 = 3$.

| $k$ | range | length |
|---|---|---|
| 0 | $[0, 4)$ | 4 |
| 1 | $[4, 8)$ | 4 |
| 2 | $[8, \min(12, 10)) = [8, 10)$ | 2 (the short last range) |

With `p = NULL` the three calls run in this order on the caller with `worker = 0`. With a 4-thread pool, one possible run: the caller takes $k = 0$, worker 2 takes $k = 1$, worker 1 takes $k = 2$, worker 3 finds the counter at 3 $\ge R$ and takes nothing. Another run gives other workers other ranges, but the set of three ranges is always the table above (sorted by `lo`).

For the bitwise claim: a $37 \times 211$ by $211 \times 29$ product where range $[i_0, i_1)$ computes rows $i_0 \dots i_1 - 1$, each element summed over $k$ from 0 to 210 in order. Whether row 5 is computed by worker 0 or worker 3, the same 211 products are added in the same order, so the results compare equal under `memcmp`.

## 4. The interface

```c
typedef struct tl_pool tl_pool;
typedef void (*tl_range_fn)(void *ctx, int64_t lo, int64_t hi, int worker);
tl_status tl_pool_create(int n_threads, tl_pool **out);   /* 0 = hardware concurrency; TL_EINVAL, TL_ENOMEM, TL_EIO */
tl_status tl_parallel_for(tl_pool *p, int64_t n, int64_t grain, tl_range_fn fn, void *ctx);
                                                           /* blocks; p == NULL is serial; TL_EINVAL, TL_EBUSY (nested) */
int       tl_pool_threads(const tl_pool *p);               /* T; 1 for NULL */
void      tl_pool_destroy(tl_pool *p);                     /* waits for a running call; NULL is a no-op */
```

`worker` is stable during one call of `fn` and is below `tl_pool_threads(p)`, so a kernel can index per-worker scratch with it (one rt.02 arena per worker, for instance).

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit, smoke | section 3: serial order and worker 0 with `p = NULL`; the same three ranges on 4 threads | the partition is pinned by a table you can check by hand |
| Grain edges and the empty range | boundary | $g = 1$, $g = n$, $g > n$, a short last range, $n = 0$ (no call), on 1 and 3 threads | every kernel's edge shapes |
| Bad arguments | boundary | $n < 0$, $g < 1$, `fn == NULL`, $T < 0$, `out == NULL`; no range runs | defined behavior at the boundary |
| Each index runs exactly once | property | 200 seeded $(n, g)$ on 8 threads: an atomic counter per index ends at exactly 1 | no lost or doubled rows |
| A pooled matmul is bitwise serial | differential | a row-range matmul on 1, 2, 3, 8 threads and grains 1 and 5 equals the serial one under `memcmp` | batch invariance (c/ABI.md rule 10) |
| All workers run concurrently | unit | 4 ranges on 4 threads each wait for all 4 to start; worker ids 0 to 3 each seen once | real parallelism, correct worker ids |
| A nested call is busy, not a deadlock | fault | a range calling on its own pool gets `TL_EBUSY` (1 and 3 threads); `p = NULL` inside works | no deadlock in composed kernels |
| Two callers are serialized | fault | two threads, 50 calls each of 1000 indices: the total is 100000; TSan clean | the engine and a benchmark sharing a pool |
| Destroy waits for running work | fault | destroy during another thread's call (whose caller is still inside a 100 ms range) returns after all 16 ranges | clean engine shutdown |
| A failed create leaks nothing | fault | the n-th allocation fails for each n: `TL_ENOMEM`, `out == NULL`, no leak; success still works | every constructor's failure path |
| The hardware-concurrency default | unit | $T = 0$ resolves to at least 1 and runs the exact partition | `[engine].threads` unset |

## 5. The solution, step by step

### From first principles

1. **Bitwise results for any $T$ force the partition to be independent of $T$.** So the ranges are fixed by $(n, g)$ alone (section 2.2), and the only thing threads change is who runs which range.
2. **Threads must be reused**, because creating them per call costs more than many kernels. So workers sleep on a condition variable between jobs, and a job is published in shared fields.
3. **Each range must run exactly once, with uneven speeds balanced.** A shared atomic counter is the simplest correct dispenser: `fetch_add` is one indivisible read-modify-write, and fast threads take more.
4. **The caller must not return while anyone can still read the job.** So every woken thread checks in (`active--`) and the caller waits for `active == 0`, not for the counter.
5. **Workers must tell a new job from a spurious wakeup.** A generation number, bumped once per job, is what they wait on; each remembers the last one it served.
6. **The pool must not deadlock or corrupt itself under composition**: refuse a nested call on the same pool (thread-local marker), serialise callers from different threads (`call_lock`), and make destroy wait for a running call (take `call_lock`).
7. **Allocation happens only on the creating thread**, so the rt.02 hook need not be thread-safe, and every failure path frees what it took.

### The code, annotated

The state, all guarded by `mu` except the counter:

```c
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
    int64_t n, grain, n_ranges;  /* the current job */
    tl_range_fn fn;
    void *ctx;
    _Atomic int64_t next;     /* the next range index to hand out */
};

/* The pool whose ranges this thread is running, or NULL. */
static _Thread_local tl_pool *running_for = NULL;
```

Handing out ranges, step 3:

```c
static void run_ranges(tl_pool *p, int worker) {
    for (;;) {
        /* Relaxed is enough: the job fields were published under `mu`, which
         * orders them before this thread's reads; the counter only needs
         * atomicity, so each k goes to exactly one thread. */
        int64_t k = atomic_fetch_add_explicit(&p->next, 1, memory_order_relaxed);
        if (k >= p->n_ranges) return;
        int64_t lo = k * p->grain;
        int64_t hi = lo + p->grain < p->n ? lo + p->grain : p->n;   /* short last range */
        p->fn(p->ctx, lo, hi, worker);
    }
}
```

A worker's life, steps 4 and 5:

```c
static void *worker_main(void *arg) {
    tl_pool_arg a = *(tl_pool_arg *)arg;
    tl_pool *p = a.pool;
    running_for = p;          /* a range on this thread calling this pool is nested */
    uint64_t seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        /* A loop, not an if: wakeups can be spurious, and a broadcast for
         * stop must not be mistaken for a job. */
        while (!p->stop && p->generation == seen) pthread_cond_wait(&p->wake, &p->mu);
        if (p->stop) break;
        seen = p->generation;
        pthread_mutex_unlock(&p->mu);     /* run ranges without the lock */
        run_ranges(p, a.worker);
        pthread_mutex_lock(&p->mu);
        if (--p->active == 0) pthread_cond_signal(&p->idle);   /* the last one tells the caller */
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}
```

The call itself:

```c
tl_status tl_parallel_for(tl_pool *p, int64_t n, int64_t grain, tl_range_fn fn, void *ctx) {
    // ... TL_EINVAL for n < 0, grain < 1, fn == NULL ...
    if (n == 0) return TL_OK;                      /* no call at all */
    int64_t n_ranges = (n - 1) / grain + 1;        /* ceil(n / grain) without overflow */
    if (p == NULL) { /* serial: ranges in order on the caller, worker 0 */ ... return TL_OK; }
    if (running_for == p) {                        /* nested: refuse, do not deadlock */
        tl_set_last_error("tl_parallel_for: nested call on the same pool");
        return TL_EBUSY;
    }
    pthread_mutex_lock(&p->call_lock);             /* one caller at a time */
    tl_pool *outer = running_for; /* a range of another pool may call us */
    if (p->n_started == 0) {
        /* A 1-thread pool runs serially, but still marks running_for so a
         * nested call is refused: the contract does not depend on T. */
        running_for = p;
        // ... ranges in order, worker 0 ...
        running_for = outer;
        pthread_mutex_unlock(&p->call_lock);
        return TL_OK;
    }
    pthread_mutex_lock(&p->mu);                    /* publish the job */
    p->n = n; p->grain = grain; p->n_ranges = n_ranges; p->fn = fn; p->ctx = ctx;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);   /* reset per job */
    p->active = p->n_started;
    p->generation++;
    pthread_cond_broadcast(&p->wake);
    pthread_mutex_unlock(&p->mu);

    running_for = p;
    run_ranges(p, 0);                              /* the caller is worker 0 */
    running_for = outer;

    pthread_mutex_lock(&p->mu);                    /* wait for every woken thread */
    while (p->active > 0) pthread_cond_wait(&p->idle, &p->mu);
    pthread_mutex_unlock(&p->mu);
    pthread_mutex_unlock(&p->call_lock);
    return TL_OK;
}
```

On the worked example with 4 threads: $R = 3$; the counter hands out 0, 1, 2 to whichever three threads reach `fetch_add` first and 3 to the fourth, which returns at once. `active` starts at 3 (the pthreads); the caller returns only when all three have decremented it.

Create and destroy, steps 6 and 7. Create resolves $T = 0$ through `sysconf(_SC_NPROCESSORS_ONLN)` (capped at 1024), allocates the pool, the thread table, and the argument table on the calling thread (any failure frees what was taken and returns `TL_ENOMEM`), initialises the mutexes and condition variables, and starts workers $1 \dots T - 1$, counting `n_started`. If a `pthread_create` fails, `shutdown_pool` stops and joins exactly the `n_started` threads already running and frees everything (`TL_EIO`). Destroy waits for a running call first:

```c
void tl_pool_destroy(tl_pool *p) {
    if (p == NULL) return;
    /* A tl_parallel_for running on another thread holds call_lock until it
     * has finished: taking it here waits for that call. */
    pthread_mutex_lock(&p->call_lock);
    pthread_mutex_unlock(&p->call_lock);
    shutdown_pool(p);   /* stop = 1, broadcast, join n_started threads, destroy, free */
}
```

Full solution: [`pool.c`](../../../../course/solutions/c/src/runtime/pool.c); contract: [`pool.h`](../../../../course/contracts/c/include/tinyllm/pool.h).

### Why this solution

- **Cost.** Per call: one lock and broadcast to publish, one `fetch_add` per range plus one that finds the end per thread, one lock per thread to check in. With a sensible grain this is a few microseconds, versus tens for creating threads.
- **Against static chunks of $n/T$.** No counter traffic, but the ranges change with $T$, so any kernel whose reduction order follows the ranges stops being bitwise reproducible, and uneven ranges leave threads idle.
- **Against work stealing.** Per-thread deques with stealing (TBB, rayon) handle nested parallelism and very uneven work better, at much higher complexity. A blocking parallel for over independent output regions does not need it.
- **Against a barrier per job.** A `pthread_barrier` would also wait for everyone, but it is optional on macOS and does not handle the "woken but not yet started" case without the same generation logic.
- **Tradeoff accepted.** Nested calls on the same pool are refused rather than run; a kernel that needs inner parallelism passes `NULL` (serial) inside a range.

### Explain it back

1. Why can the counter use `memory_order_relaxed`?
   <details><summary>Answer</summary>The job's fields (`n`, `grain`, `fn`, `ctx`, `n_ranges`) are written under `mu` and read by workers only after they took `mu` and saw the new generation, so the mutex orders them. The counter itself only has to hand each value to one thread, which atomicity alone guarantees.</details>
2. Why does the caller wait for `active == 0` instead of returning when `run_ranges` returns?
   <details><summary>Answer</summary>`run_ranges` returning means no range is left to hand out, not that the others have finished theirs; one may still be inside `fn` writing output. And a worker that woke late could read the job fields after the next call overwrote them. Waiting for every woken thread to check in rules out both.</details>
3. What would happen without the `running_for == p` check when a range calls `tl_parallel_for` on its own pool?
   <details><summary>Answer</summary>The inner call would block on `call_lock`, which the outer call holds, or (with a different lock design) wait for workers that are busy running the outer call's ranges. Either way it would wait forever. The check turns that deadlock into `TL_EBUSY`.</details>
4. Why does destroy lock and immediately unlock `call_lock`?
   <details><summary>Answer</summary>A call running on another thread holds `call_lock` until it returns, and its caller (worker 0) still reads the pool after the pthreads ran out of ranges. Acquiring the lock waits for that call to finish; only then is it safe to stop, join, and free.</details>
5. Why are ranges $[kg, \min((k+1)g, n))$ enough to make a row-range matmul bitwise identical across thread counts?
   <details><summary>Answer</summary>Each output element is computed entirely inside one range by one thread, summing over the inner dimension in the same fixed order. Which thread computes it, and when, changes nothing about the sequence of floating-point operations, so the bits are the same.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. taking a range with a plain read and write of the counter | some ranges run twice, others never; TSan reports the race | `atomic_fetch_add_explicit` on `next` |
| 2. splitting $n$ into $T$ chunks instead of by grain | different sums with different thread counts; the edge partitions fail | ranges are $[kg, \min((k+1)g, n))$, independent of $T$ |
| 3. $R = \lfloor n/g \rfloor$ | the short last range is dropped | `n_ranges = (n - 1) / grain + 1` |
| 4. returning when the counter runs out, not when the workers finish | a range still running after return; the next job is read half-written | the caller waits on `idle` until `active == 0` |
| 5. not resetting the counter for each job | the second call runs nothing | `atomic_store(&next, 0)` while publishing the job |
| 6. the caller doing all the work, or every worker reporting id 0 | no speedup; per-worker scratch shared by all | pthreads are woken before the caller joins in; each passes its own `a.worker` |
| 7. no nested-call check, or none for the 1-thread pool | the inner call waits forever | `running_for == p` returns `TL_EBUSY`; the 1-thread path sets `running_for` too |
| 8. no lock between two callers | one caller's job overwrites the other's | `call_lock` held for the whole call |
| 9. destroy that only joins the workers, or detaches them | the pool freed under a running call (ASan, TSan) | destroy takes `call_lock` first, then joins |
| 10. a failed create that frees only part of what it allocated | a leak per failed create | every failure path frees the tables and the pool; `shutdown_pool` joins exactly `n_started` threads |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | `tl_alloc`/`tl_free` for the pool and its tables (counted in tests), `tl_set_last_error` for failures |
| Forward | `L9.1` | `tl_matmul_f32(..., tp)` splits rows of C into ranges; each output element is reduced in one range |
| Forward | `L9.3` | FlashAttention gives each (batch, head) pair to a range; worker ids index per-worker scratch arenas (rt.02) |
| Forward | `L9.4` | paged decode attention splits the batch's sequences and heads into ranges |
| Forward | `L9.5` | the int4 and int8 matmuls split output rows into ranges |
| Forward | `L9.1`, `L9.3`, `L9.4`, `L9.5` | Optional standalone C kernels use the pool to schedule independent output regions |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_parallel_for` | OpenMP `parallel for schedule(dynamic, g)`, TBB `parallel_for` | work stealing between per-thread deques, nested parallelism without deadlock | oneTBB `include/oneapi/tbb/parallel_for.h` |
| fixed partition | ggml's threadpool and `ggml_compute_forward_mul_mat` | splits rows by `ith`/`nth` per thread, with a barrier between graph nodes | `ggml/src/ggml-cpu/ggml-cpu.c` |
| sleeping workers | Rust `rayon` | work-stealing scheduler, `join` and parallel iterators, a global pool | the rayon `rayon-core` crate |
| the counter | PyTorch `at::parallel_for` | a grain size to avoid overhead on small loops, thread-local intra-op pools | `aten/src/ATen/Parallel.h` |
