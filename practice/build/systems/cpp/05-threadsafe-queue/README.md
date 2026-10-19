# C++ 05: Thread-safe queue

**Concepts:** `std::mutex`, `std::condition_variable`, move semantics
**Difficulty:** ⭐⭐⭐

A bounded blocking queue: producers wait when it is full, consumers wait when it
is empty, and closing it wakes everyone. The solution is the header
[`queue.hpp`](queue.hpp); [`main.cpp`](main.cpp) is the test suite.

## The problem

| Member | Does |
|--------|------|
| `push(v)` | Block until there is room. False if closed |
| `try_push(v)` | Never blocks. False if full or closed |
| `pop()` | Block until an element exists. `nullopt` once closed *and* drained |
| `try_pop()` | Never blocks |
| `close()` | Wake every waiter, refuse new pushes, keep delivering the backlog |

Also `size()`, `empty()`, `closed()`, all `const`. Payloads may be move-only.
Nothing asserts on cross-thread ordering; what is checked is that every item is
consumed exactly once.

## Build and run

This exercise needs `-pthread`, so it ships a [`run.sh`](run.sh):

```bash
./run.sh
# ok  cpp/05-threadsafe-queue  ... checks passed
```

Or by hand, from this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 -pthread main.cpp -o queue && ./queue
```

Passing once means little for concurrent code, which is why the suite runs
three times. ThreadSanitizer finds races the tests cannot:

```bash
c++ -std=c++20 -Wall -Wextra -pthread -fsanitize=thread -O1 main.cpp -o queue && ./queue
```

## What to notice

**RAII is the whole difference from the C version.** There is no path through
this file where a lock is acquired and not released, including the paths where
an exception is thrown, because `unique_lock`'s destructor runs during stack
unwinding. Compare with the [C thread pool](../../c/07-thread-pool/), where
every early return has to remember to unlock.

**Use the predicate form of `wait`.** `cv.wait(lock, pred)` is exactly
`while (!pred()) cv.wait(lock);` and the loop is mandatory: a bare `wait` can
return spuriously, and several waiters can be woken by one `notify` with only
the first to reacquire the mutex finding the condition true.

**Unlock before notifying.** If you notify while still holding the mutex, the
woken thread immediately blocks trying to acquire it, costing a second context
switch for nothing. It is a performance bug, not a correctness one, which is
exactly why it survives code review.

**`notify_all` on close, never `notify_one`.** Every blocked thread must wake to
observe the closure. Waking one leaves the others parked forever, and any
`join` on them deadlocks. `test_close_wakes_blocked_consumers` blocks four
consumers and joins them, so a `notify_one` implementation hangs the test.

**Closed does not mean empty.** A producer whose `push` returned `true` is
entitled to have that element consumed. So `pop` must drain the backlog before
it starts reporting `nullopt`.

**`mutable std::mutex` is not a hack.** `size()` and `closed()` are logically
const but must lock, and `mutable` is the standard way to say "this member is
not part of the object's observable state".

**The queue is neither copyable nor movable, and it should not be.** A mutex is
neither, and a queue that could be moved out from under a waiting thread would
be a data race by construction.

## The solution, step by step

### From first principles

1. **One mutex guards all shared state**: the `std::queue`, the capacity, and
   the `closed_` flag. Every member function locks it first.
2. **Two reasons to sleep, two condition variables.** Producers wait for
   "room or closed" on `not_full_`; consumers wait for "an element or closed" on
   `not_empty_`. Each successful operation notifies the other side.
3. **Closure is part of every wait predicate.** Otherwise a thread waiting for
   room or data would never learn the queue has closed.
4. **After waking, decide by state, in the right order.** `push`: closed means
   refuse. `pop`: an element present means deliver it, *even if closed*; only
   empty-and-closed means `nullopt`.
5. **Values move, never copy.** `push` takes `T` by value and moves it in; `pop`
   moves the front out before popping. That supports move-only payloads.
6. **Notify after unlocking**, so the woken thread can take the lock at once.

### The code, annotated

```cpp
bool push(T value) {
  std::unique_lock lock(mutex_);

  // Predicate form: re-checked after every wake-up, spurious or not.
  not_full_.wait(lock, [this] { return q_.size() < cap_ || closed_; });

  if (closed_) return false;          // the lock is released by ~unique_lock

  q_.push(std::move(value));

  // Unlock BEFORE notifying, so the consumer we wake can take the mutex
  // immediately instead of blocking on it.
  lock.unlock();
  not_empty_.notify_one();            // one element: one consumer
  return true;
}

std::optional<T> pop() {
  std::unique_lock lock(mutex_);
  not_empty_.wait(lock, [this] { return !q_.empty() || closed_; });

  // Drain before reporting closure: an accepted element must be delivered.
  if (q_.empty()) return std::nullopt;   // here, empty implies closed

  T value = std::move(q_.front());       // move out, then remove the husk
  q_.pop();
  lock.unlock();
  not_full_.notify_one();                // one slot freed: one producer
  return value;
}
```

The non-blocking variants check the same conditions without waiting:

```cpp
bool try_push(T value) {
  std::unique_lock lock(mutex_);
  if (closed_ || q_.size() >= cap_) return false;
  q_.push(std::move(value));
  lock.unlock();
  not_empty_.notify_one();
  return true;
}
```

Closing:

```cpp
void close() {
  {
    std::lock_guard lock(mutex_);   // the flag is read under the lock by waiters
    closed_ = true;
  }
  // Every waiter must re-check its predicate and see closed_.
  not_empty_.notify_all();
  not_full_.notify_all();
}
```

Setting `closed_` under the lock matters even though it is a single `bool`: a
waiter evaluates its predicate under the mutex, so changing the flag outside it
could slip between a waiter's check and its sleep, and the notification would
be lost.

The class is completed by `const` observers that lock a `mutable std::mutex`,
and deleted copy operations (which also suppress the implicit moves).

Full solution: [`queue.hpp`](queue.hpp).

### Why this solution

- **Cost.** Every operation is O(1) under one lock (amortised, since
  `std::queue` over `std::deque` allocates in blocks).
- **Versus a lock-free queue.** Lock-free ring buffers avoid the mutex but cannot
  block: a consumer would spin or need a separate wake-up mechanism. For a
  blocking queue with backpressure, mutex plus condition variables is simpler
  and correct.
- **Two condition variables versus one.** One shared variable would force
  `notify_all` on every operation, since a wake-up could be meant for either
  side.
- **Accepted trade.** `try_push` takes its argument by value, so a failed
  `try_push(std::move(x))` has already moved `x` into the parameter and destroys
  it. A caller that must keep the value on failure needs an overload taking
  `T&&` that only moves on success.

### Explain it back

1. Why does `pop` check `q_.empty()` after the wait instead of `closed_`?

   <details><summary>Answer</summary>

   The queue may be closed and still hold elements that producers were promised
   would be consumed. Checking empty first delivers them; only when the queue is
   both closed and empty does `pop` return `nullopt`.

   </details>

2. What happens if `close()` uses `notify_one`?

   <details><summary>Answer</summary>

   One waiter wakes and sees `closed_`; every other waiter stays asleep, because
   nothing will ever notify them again. Threads joining those waiters hang.

   </details>

3. Why set `closed_` while holding the mutex?

   <details><summary>Answer</summary>

   A waiter checks its predicate and goes to sleep atomically with respect to
   the mutex. If `closed_` were set and notified without the lock, the change
   could land after a waiter evaluated the predicate as false but before it
   started waiting, and that waiter would miss the notification forever.

   </details>

4. Why is notifying after `lock.unlock()` still correct?

   <details><summary>Answer</summary>

   The state change happened under the lock, so any waiter that checks its
   predicate afterwards sees it. A waiter that was already sleeping gets the
   notification. The only thing moving the notify outside the lock changes is
   that the woken thread does not immediately block on a mutex still held.

   </details>

## Extending it

Add `pop_for(timeout)` using `wait_for`, and notice that the predicate overload
handles the spurious-wakeup-plus-deadline interaction that hand-rolled timeout
loops almost always get wrong. Then try replacing the `std::queue` with a fixed
ring buffer to remove the per-element allocation.
