# Thread pool

Creating a thread per task is simple and slow: thread creation costs tens of microseconds, and ten thousand simultaneous threads exhaust memory long before they finish. A thread pool fixes both by starting a fixed set of worker threads once and feeding them tasks through a queue. This section builds one with pthreads: a bounded queue that pushes back on the producer when it is full, workers that sleep on condition variables when it is empty, a `pool_wait` that knows the difference between "nothing queued" and "nothing running", and a clean shutdown that finishes everything it accepted.

Concurrent code is where "it passed the tests" means least. Every rule in this section (wait in a loop, run the task outside the lock, broadcast on shutdown) describes a bug that works on a lightly loaded laptop and fails in production. You should be comfortable with mutexes and condition variables as concepts; the code shows exactly how to use them. The same design, extended with a `parallel_for` over index ranges, is [Thread pool and tl_parallel_for](../../../ai-engineering/22-kernels/02-thread-pool-and-parallel-for.md) in *AI Engineering*.

> **Learning objectives**
> - Guard a pool's shared state with one mutex and use one condition variable per reason to sleep.
> - Explain why every condition-variable wait must re-test its predicate in a `while` loop.
> - Track in-flight work so that `pool_wait` cannot return while a task is still running.
> - Shut a pool down without deadlock and without dropping accepted work.
> - Explain backpressure and why a bounded queue turns overload into a slowdown instead of a crash.

## The interface

[`pool.h`](../../../../../practice/build/systems/c/07-thread-pool/pool.h) declares the interface, [`pool.c`](../../../../../practice/build/systems/c/07-thread-pool/pool.c) is the solution, and [`main.c`](../../../../../practice/build/systems/c/07-thread-pool/main.c) is the test suite.

| Function | Does |
|----------|------|
| `pool_new(n, cap)` | Start `n` workers with room for `cap` pending tasks |
| `pool_submit(p, fn, arg)` | Enqueue, blocking while the queue is full. -1 once shutting down |
| `pool_wait(p)` | Block until everything submitted so far has finished |
| `pool_destroy(p)` | Stop accepting, drain the backlog, join, free |
| `pool_completed(p)` | Tasks finished so far |

The order tasks run in is genuinely nondeterministic and nothing asserts on it. What is deterministic, and what the tests check, is that every submitted task runs *exactly once*. Most tests give each task its own slot to write, so a lost or duplicated run is visible with no locking in the test itself. The whole suite runs three times, because a race that appears one run in ten is still a bug.

This exercise needs `-pthread`, so it ships a [`run.sh`](../../../../../practice/build/systems/c/07-thread-pool/run.sh) that builds into a temporary directory and runs the tests:

```bash
./run.sh
# ok  c/07-thread-pool  ... checks passed
```

Or by hand, from this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 -pthread main.c pool.c -o pool && ./pool
```

Passing once means little for concurrent code. ThreadSanitizer finds data races the tests cannot:

```bash
cc -std=c11 -Wall -Wextra -pthread -fsanitize=thread -O1 main.c pool.c -o pool && ./pool
```

## Condition variables and the lock

Wait in a `while`, never an `if`. `pthread_cond_wait` can return without the condition holding: spuriously, or because several workers were woken for one signal and only the first to reacquire the lock got the task. Re-testing the predicate after waking is not defensive programming; it is the required usage, and an `if` here produces a pool that works until it is loaded.

Run the task outside the lock. Holding the mutex while calling `t.fn(t.arg)` serialises every worker down to one, which is the most common way to write a thread pool that is slower than a plain loop and still passes all its tests. The lock protects the queue, not the work.

## Knowing when the pool is quiet

`pool_wait` has to account for in-flight tasks, not just an empty queue. A task that has been dequeued is no longer counted by the queue and has not finished either. That is what the `active` counter is for, and why the idle signal fires only when `count == 0 && active == 0`. `test_wait_covers_in_flight_work` uses tasks slow enough that a queue-only wait returns early and sees an incomplete total.

## Shutting down and pushing back

Broadcast on shutdown; do not signal. Every worker is parked on `not_empty`, and every one of them must wake to notice the shutdown flag and exit. A signal wakes exactly one, leaving the rest blocked forever and `pthread_join` hanging. This is the classic thread-pool deadlock, and it only shows up with more than one worker.

Shutdown means "no new work", not "abandon the backlog". A `pool_submit` that returned 0 is a promise the task will run. `pool_destroy` must therefore drain the queue before workers exit, which is why the worker's exit test is `count == 0 && shutting_down` rather than `shutting_down` alone.

The bounded queue is a design choice with a name. Blocking the producer when the queue is full is backpressure: the system slows the source rather than growing an unbounded backlog until memory runs out. An unbounded queue makes `submit` never block and turns a sustained overload into an OOM kill instead of a slowdown.

## Deriving the design

Six observations fix the structure.

Shared state needs one lock. The queue (a ring buffer with `head`, `tail`, `count`), the `active` and `completed` counters, and the `shutting_down` flag are all read and written by several threads. One mutex guards all of them, so every invariant between them is checked atomically.

Each reason to sleep gets its own condition variable. Workers sleep until the queue is non-empty (`not_empty`), producers until it is non-full (`not_full`), and `pool_wait` until the pool is quiet (`idle`). Separate variables mean a wake-up goes to a thread that cares about it.

Every wait re-tests its predicate in a loop. A condition variable is only a hint that something changed. The predicate, read under the lock, is the truth.

A ring buffer gives O(1) enqueue and dequeue in fixed memory. `tail` advances on push, `head` on pop, both modulo `cap`; `count` disambiguates full from empty.

"Quiet" is a property of two numbers. A task in the queue and a task being run both count as unfinished. `active` is incremented under the lock in the same critical section that dequeues, so there is no instant when a task is in neither count.

Shutdown is a flag plus a broadcast. Set `shutting_down` under the lock, wake *every* sleeper, and let each one decide from the predicate: a worker exits only when the queue is also empty, and a producer gives up with -1.

## Building it step by step

The worker loop is the heart of the pool:

```c
static void *worker_main(void *arg) {
  ThreadPool *p = arg;

  for (;;) {
    pthread_mutex_lock(&p->lock);

    /* while, not if: wake-ups can be spurious, and another worker may have
     * taken the task first. */
    while (p->count == 0 && !p->shutting_down) {
      pthread_cond_wait(&p->not_empty, &p->lock);
    }

    /* Exit only when shutting down AND drained: accepted work still runs. */
    if (p->count == 0 && p->shutting_down) {
      pthread_mutex_unlock(&p->lock);
      return NULL;
    }

    Task t = p->queue[p->head];             /* copy out under the lock */
    p->head = (p->head + 1) % p->cap;
    p->count--;
    p->active++;                            /* same critical section as the dequeue */

    pthread_cond_signal(&p->not_full);      /* one slot freed: one producer */
    pthread_mutex_unlock(&p->lock);

    /* Run OUTSIDE the lock, or the pool has one effective worker. */
    t.fn(t.arg);

    pthread_mutex_lock(&p->lock);
    p->active--;
    p->completed++;
    /* Quiet means nothing queued AND nothing running. */
    if (p->count == 0 && p->active == 0) pthread_cond_broadcast(&p->idle);
    pthread_mutex_unlock(&p->lock);
  }
}
```

`idle` uses broadcast because several threads may be inside `pool_wait`.
`not_full` uses signal because exactly one slot opened, so waking one producer
is enough.

Submit is the mirror image, with shutdown as a second way out of the wait:

```c
int pool_submit(ThreadPool *p, TaskFn fn, void *arg) {
  pthread_mutex_lock(&p->lock);

  /* Backpressure: block while full. Shutdown must also end the wait, or a
   * producer blocked here would never return. */
  while (p->count == p->cap && !p->shutting_down) {
    pthread_cond_wait(&p->not_full, &p->lock);
  }

  if (p->shutting_down) {
    pthread_mutex_unlock(&p->lock);
    return -1;                              /* refused, so no promise made */
  }

  p->queue[p->tail] = (Task){fn, arg};
  p->tail = (p->tail + 1) % p->cap;
  p->count++;

  pthread_cond_signal(&p->not_empty);       /* one task: one worker */
  pthread_mutex_unlock(&p->lock);
  return 0;
}
```

Waiting and shutting down:

```c
void pool_wait(ThreadPool *p) {
  pthread_mutex_lock(&p->lock);
  while (p->count > 0 || p->active > 0) {
    pthread_cond_wait(&p->idle, &p->lock);
  }
  pthread_mutex_unlock(&p->lock);
}

void pool_destroy(ThreadPool *p) {
  if (!p) return;

  pthread_mutex_lock(&p->lock);
  p->shutting_down = 1;
  /* Broadcast, not signal: every parked worker must wake to see the flag,
   * or pthread_join below waits forever. Blocked producers too. */
  pthread_cond_broadcast(&p->not_empty);
  pthread_cond_broadcast(&p->not_full);
  pthread_mutex_unlock(&p->lock);

  for (size_t i = 0; i < p->n_threads; i++) pthread_join(p->threads[i], NULL);

  /* Only now, with every worker joined, is it safe to destroy what they use. */
  pthread_mutex_destroy(&p->lock);
  pthread_cond_destroy(&p->not_empty);
  pthread_cond_destroy(&p->not_full);
  pthread_cond_destroy(&p->idle);
  free(p->queue);
  free(p->threads);
  free(p);
}
```

`pool_new` (not quoted) allocates the queue and thread array, initialises the
lock and condition variables, and starts the workers. If some `pthread_create`
calls fail it keeps the workers that did start (`n_threads = i`), and only if
none started does it tear everything down and return `NULL`.

The full solution is in [`pool.c`](../../../../../practice/build/systems/c/07-thread-pool/pool.c).

## Why this design

Submit and dequeue are O(1) under one lock. Memory is fixed at `cap` task slots plus `n` threads, whatever the submission rate.

One queue with one lock is simple and correct, and fine for coarse tasks. For very short tasks the shared lock becomes the bottleneck, because every worker contends on it for every task; work-stealing deques, described at the end of this section, fix that.

A bounded queue is chosen over an unbounded one. Unbounded makes `submit` never block, which is convenient until a fast producer outruns the workers and memory runs out. Bounded turns overload into a slowdown of the producer.

Three condition variables are chosen over one. One shared variable would need `broadcast` everywhere (any wake might be for anyone), waking threads that then go back to sleep. Separate variables let `signal` wake exactly the right kind of thread.

The accepted trade is that `pool_destroy` blocks until the whole backlog has run. A pool that needed fast cancellation would drop queued tasks and say so in its contract.

## What can go wrong

Every one of these bugs passes a single-threaded or lightly loaded test.

| Mistake | Consequence |
|---------|-------------|
| `if` instead of `while` around `pthread_cond_wait` | A woken worker dequeues from an empty queue under load |
| Running the task while holding the lock | The pool has one effective worker |
| Waiting only for an empty queue | `pool_wait` returns while dequeued tasks are still running |
| Incrementing `active` after releasing the lock | A window in which a task is counted nowhere |
| `signal` instead of `broadcast` on shutdown | Only one worker exits; `pthread_join` hangs |
| Exiting workers on `shutting_down` alone | Accepted tasks in the queue are silently dropped |
| Not testing `shutting_down` in `pool_submit`'s wait | A producer blocked on a full queue never returns |
| Running `t.fn` from the queue slot instead of a copy | The slot is overwritten by the next `submit` after unlocking |

> **Tip:** Run concurrent tests many times and under ThreadSanitizer. A race that fails one run in ten passes most CI runs and is still a bug.

## Where this leads

The next step is work stealing: give each worker its own deque and let idle workers steal from the back of a busy one's. Go's scheduler, Rayon, and Java's `ForkJoinPool` all do this, and the reason is contention: one shared queue means every worker touches the same cache line on every task, which stops scaling long before the cores run out. In *AI Engineering*, [Thread pool and tl_parallel_for](../../../ai-engineering/22-kernels/02-thread-pool-and-parallel-for.md) uses a pool like this one to parallelise kernel loops.

## Key terms

- **Thread pool**: a fixed set of long-lived worker threads that execute tasks taken from a shared queue.
- **Condition variable**: a primitive that lets a thread sleep until another thread signals that shared state may have changed.
- **Spurious wake-up**: a return from a condition-variable wait with no corresponding signal, or after another thread consumed the change.
- **Predicate loop**: re-testing the awaited condition under the lock after every wake-up.
- **Backpressure**: slowing the producer when the consumer cannot keep up, here by blocking `submit` on a full queue.
- **Ring buffer**: a fixed array used as a FIFO queue with head and tail indices that wrap.

## Summary

- One mutex guards the ring-buffer queue, the counters, and the shutdown flag; three condition variables separate the three reasons to sleep.
- Every wait re-tests its predicate in a `while` loop, and tasks run with the lock released.
- `active` is incremented in the same critical section as the dequeue, so `pool_wait` waits for `count == 0 && active == 0`.
- Shutdown sets a flag and broadcasts to every sleeper; workers drain the queue before exiting, and blocked producers return -1.
- A bounded queue gives backpressure: overload slows the producer instead of exhausting memory.

## Review questions

1. Why is `active` incremented in the same critical section that dequeues,
   rather than when the task starts running?

   <details><summary>Answer</summary>

   If the lock were released between `count--` and `active++`, `pool_wait`
   could acquire it in that gap, see `count == 0 && active == 0`, and return
   while a task is about to run. Doing both under one lock means the task is
   always counted somewhere.

   </details>

2. What goes wrong if `pool_destroy` uses `pthread_cond_signal(&p->not_empty)`?

   <details><summary>Answer</summary>

   Only one worker wakes, sees the shutdown, and exits. The others stay parked
   on `not_empty` with nothing to wake them, and `pthread_join` on them never
   returns: the program hangs.

   </details>

3. Why does the worker check `count == 0 && shutting_down` rather than just
   `shutting_down` before exiting?

   <details><summary>Answer</summary>

   Tasks still in the queue were accepted with `pool_submit` returning 0, which
   promises they will run. Exiting on the flag alone would drop them silently.
   Workers keep dequeuing until the queue is empty, and only then exit.

   </details>

4. Why must `pool_submit`'s wait loop also test `shutting_down`?

   <details><summary>Answer</summary>

   A producer blocked on a full queue waits on `not_full`. If shutdown did not
   end that wait, and the workers drained the queue and exited, nothing would
   ever signal `not_full` again for it. With the flag in the predicate, the
   broadcast in `pool_destroy` wakes it and it returns -1.

   </details>

5. Why does `t.fn(t.arg)` run with the lock released, and what makes copying
   `t` out first necessary?

   <details><summary>Answer</summary>

   Holding the lock during the task would block every other worker and every
   producer for its whole duration, serialising the pool. Once the lock is
   released, the ring-buffer slot may be overwritten by the next `submit`, so the
   task must be copied into a local before unlocking.

   </details>
