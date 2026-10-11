<!-- ol:module rt.03 -->
# Thread pool and tl_parallel_for (optional C)

## Overview

| | |
|---|---|
| **Module** | `rt.03` · side · C · Pass 6 · 3 to 4 h |
| **You build** | `c/src/runtime/pool.c`: `tl_pool_create`, `tl_parallel_for`, `tl_pool_threads`, `tl_pool_destroy`, and the helpers `run_ranges`, `worker_main`, `shutdown_pool` (the struct is given) |
| **Contract** | [`course/contracts/c/include/tinyllm/pool.h`](../../../course/contracts/c/include/tinyllm/pool.h) · rules: [`c/ABI.md`](../../../course/contracts/c/ABI.md) (pthreads only; batch invariance, rule 10) |
| **Tests** | `course/tests/rt.03/test_pool.c` (C, built twice: ASan and UBSan with the counting allocator, then ThreadSanitizer) (what they check: section 4) |
| **Needs** | [`rt.02` runtime support](02-arena-allocator-with-marks.md): `tl_alloc`, `tl_free`, `tl_set_last_error` (or `--ref-deps`). Reading: [`lang.03` C](../../../software-craftsmanship/12-language-and-tool-primers/03-c.md) |
| **Used by** | `L9.1`, `L9.3`, `L9.4`, and `L9.5` use the pool in their optional standalone C implementations |
| **Milestone** | `MS-L9`, the optional C module group |
| **Optional depth** | Herlihy and Shavit, *The Art of Multiprocessor Programming*, ch. 16 (work distribution); Butenhof, *Programming with POSIX Threads*, ch. 3 and 7; the OpenMP specification, `schedule(dynamic, chunk)` |

## Key Takeaways

- **The partition is part of the contract**: ranges are always $[kg, \min((k+1)g, n))$, whoever runs them, so a kernel that reduces only inside a range gives the same bits with 1 or 8 threads (`pooled_matmul_is_bitwise_serial`).
- **One atomic counter hands out ranges**: `atomic_fetch_add` gives each range to exactly one thread; a read followed by a write gives some ranges twice (`each_index_runs_exactly_once`, under ThreadSanitizer).
- **"Done" means every thread that woke has finished**, not that the counter ran out; returning early lets the next job overwrite one still being read (`each_index_runs_exactly_once`).
- **A nested call on the same pool is `TL_EBUSY`, not a deadlock**, and calls from two threads are serialized (`nested_call_is_busy_not_a_deadlock`, `two_callers_are_serialized`).
- **Destroy waits for a running call** and joins every thread; a failed create leaks nothing (`destroy_waits_for_running_work`, `create_failure_leaks_nothing`).

## How to work this chapter

```bash
ol start rt.03              # stubs pool.c into your repo (the struct is given)
ol tests rt.03              # read the test catalog first
ol check rt.03              # exit code is the verdict (ASan build, then TSan build)
OL_TSAN=0 ol check rt.03    # skip the ThreadSanitizer build while iterating
ol diff  rt.03              # after passing: your code against the reference
```

Every test starts a 10-second watchdog: a deadlock reports `FAIL (no progress in 10 s: a deadlock?)` and the test's name instead of hanging until the harness timeout.

---

## 1. Why now

Your C matmul (M03.1) runs on one core. The optional tiled kernel of L9.1 can split independent columns across workers; attention (L9.3) splits heads, and other optional C exercises parallelize rows or sequences. Each implementation still owns a fixed reduction order, so scheduling changes which worker computes a result but does not change the result's bits. Starting threads per call costs tens of microseconds, so these standalone C exercises keep a pool that sleeps between calls. The production Rust engine owns its own Rust thread pool and independent kernels.

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

`atomic_fetch_add` returns each value of the counter to exactly one caller, so each range runs exactly once, and fast workers simply take more ranges (dynamic scheduling, which balances uneven ranges). Writing it as `k = next; next = k + 1;` lets two threads read the same $k$: a race, and a range run twice (ThreadSanitizer reports it; the per-index counters of `each_index_runs_exactly_once` see a 2).

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

With `p = NULL` the three calls run in this order on the caller with `worker = 0`. With a 4-thread pool, one possible run: the caller takes $k = 0$, worker 2 takes $k = 1$, worker 1 takes $k = 2$, worker 3 finds the counter at 3 $\ge R$ and takes nothing. Another run gives other workers other ranges, but the set of three ranges is always the table above, which is what `hand_example` checks (sorted by `lo`) on 4 threads after checking the serial order exactly.

For the bitwise claim: a $37 \times 211$ by $211 \times 29$ product where range $[i_0, i_1)$ computes rows $i_0 \dots i_1 - 1$, each element summed over $k$ from 0 to 210 in order. Whether row 5 is computed by worker 0 or worker 3, the same 211 products are added in the same order, so `pooled_matmul_is_bitwise_serial` compares with `memcmp`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3: serial order and worker 0 with `p = NULL`; the same three ranges on 4 threads | you and the test agree on the partition |
| `grain_edges_and_empty_range` | boundary | $g = 1$, $g = n$, $g > n$, a short last range, $n = 0$ (no call), on 1 and 3 threads | every kernel's edge shapes |
| `bad_arguments` | boundary | $n < 0$, $g < 1$, `fn == NULL`, $T < 0$, `out == NULL`; no range runs | defined behavior at the boundary |
| `each_index_runs_exactly_once` | property | 200 seeded $(n, g)$ on 8 threads: an atomic counter per index ends at exactly 1 | no lost or doubled rows |
| `pooled_matmul_is_bitwise_serial` | differential | a row-range matmul on 1, 2, 3, 8 threads and grains 1 and 5 equals the serial one under `memcmp` | batch invariance (c/ABI.md rule 10) |
| `all_workers_run_concurrently` | unit | 4 ranges on 4 threads each wait for all 4 to start; worker ids 0 to 3 each seen once | real parallelism, correct worker ids |
| `nested_call_is_busy_not_a_deadlock` | fault | a range calling on its own pool gets `TL_EBUSY` (1 and 3 threads); `p = NULL` inside works | no deadlock in composed kernels |
| `two_callers_are_serialized` | fault | two threads, 50 calls each of 1000 indices: the total is 100000; TSan clean | the engine and a benchmark sharing a pool |
| `destroy_waits_for_running_work` | fault | destroy during another thread's call (whose caller is still inside a 100 ms range) returns after all 16 ranges | clean engine shutdown |
| `create_failure_leaks_nothing` | fault | the n-th allocation fails for each n: `TL_ENOMEM`, `out == NULL`, no leak; success still works | every constructor's failure path |
| `hardware_concurrency_default` | unit | $T = 0$ resolves to at least 1 and runs the exact partition | `[engine].threads` unset |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. taking a range with a plain read and write of the counter | some ranges run twice, others never; TSan reports the race | `each_index_runs_exactly_once` (mutant `s01`) |
| 2. splitting $n$ into $T$ chunks instead of by grain | different sums with different thread counts; the edge partitions fail | `hand_example`, `grain_edges_and_empty_range` (mutant `s02`) |
| 3. $R = \lfloor n/g \rfloor$ | the short last range is dropped | `grain_edges_and_empty_range` (mutant `s03`) |
| 4. returning when the counter runs out, not when the workers finish | a range still running after return; the next job is read half-written | `each_index_runs_exactly_once` (mutant `s05`) |
| 5. not resetting the counter for each job | the second call runs nothing | `grain_edges_and_empty_range` (mutant `s06`) |
| 6. the caller doing all the work, or every worker reporting id 0 | no speedup; per-worker scratch shared by all | `all_workers_run_concurrently` (mutants `s07`, `s08`) |
| 7. no nested-call check, or none for the 1-thread pool | the inner call waits forever | `nested_call_is_busy_not_a_deadlock` (mutants `s09`, `s10`) |
| 8. no lock between two callers | one caller's job overwrites the other's | `two_callers_are_serialized` (mutant `s11`) |
| 9. destroy that only joins the workers, or detaches them | the pool freed under a running call (ASan, TSan) | `destroy_waits_for_running_work` (mutants `s12`, `s13`) |
| 10. a failed create that frees only part of what it allocated | a leak per failed create | `create_failure_leaks_nothing` (mutant `s14`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | `tl_alloc`/`tl_free` for the pool and its tables (counted in tests), `tl_set_last_error` for failures |
| Forward | `L9.1` | `tl_matmul_f32(..., tp)` splits rows of C into ranges; each output element is reduced in one range |
| Forward | `L9.3` | FlashAttention gives each (batch, head) pair to a range; worker ids index per-worker scratch arenas (rt.02) |
| Forward | `L9.4` | paged decode attention splits the batch's sequences and heads into ranges |
| Forward | `L9.5` | the int4 and int8 matmuls split output rows into ranges |
| Forward | `L9.1`, `L9.3`, `L9.4`, `L9.5` | Optional standalone C kernels use the pool to schedule independent output regions |

If you skip this module, `ol check L9.1` stops with `BLOCKED ... needs rt.03`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_parallel_for` | OpenMP `parallel for schedule(dynamic, g)`, TBB `parallel_for` | work stealing between per-thread deques, nested parallelism without deadlock | oneTBB `include/oneapi/tbb/parallel_for.h` |
| fixed partition | ggml's threadpool and `ggml_compute_forward_mul_mat` | splits rows by `ith`/`nth` per thread, with a barrier between graph nodes | `ggml/src/ggml-cpu/ggml-cpu.c` |
| sleeping workers | Rust `rayon` | work-stealing scheduler, `join` and parallel iterators, a global pool | the rayon `rayon-core` crate |
| the counter | PyTorch `at::parallel_for` | a grain size to avoid overhead on small loops, thread-local intra-op pools | `aten/src/ATen/Parallel.h` |
