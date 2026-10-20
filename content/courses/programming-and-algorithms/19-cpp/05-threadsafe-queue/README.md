# Thread-safe queue

A blocking queue is the simplest correct way to hand work from one thread to another. Producers wait when it is full, consumers wait when it is empty, and closing it wakes everyone so the program can shut down. This section builds a bounded one in C++20 with `std::mutex` and `std::condition_variable`, holding payloads that may be move-only.

If you have read the C [thread pool](../../18-c/07-thread-pool/README.md), the concurrency rules will be familiar: one lock, one condition variable per reason to sleep, predicates re-checked after every wake-up, and a broadcast on shutdown. What C++ adds is RAII, and it is the whole difference. There is no path through this file where a lock is acquired and not released, including the paths where an exception is thrown, because `unique_lock`'s destructor runs during stack unwinding. In the C version, every early return has to remember to unlock. The solution is the header [`queue.hpp`](../../../../../practice/build/systems/cpp/05-threadsafe-queue/queue.hpp); [`main.cpp`](../../../../../practice/build/systems/cpp/05-threadsafe-queue/main.cpp) is the test suite.

> **Learning objectives**
> - Build a bounded blocking queue from one mutex and two condition variables, with RAII locking.
> - Use the predicate form of `wait` and explain why the loop it hides is mandatory.
> - Implement `close` so that every waiter wakes and accepted elements are still delivered.
> - Explain why the closed flag must change under the lock, and why notifying after unlocking is both correct and faster.

## The interface

| Member | Does |
|--------|------|
| `push(v)` | Block until there is room. False if closed |
| `try_push(v)` | Never blocks. False if full or closed |
| `pop()` | Block until an element exists. `nullopt` once closed *and* drained |
| `try_pop()` | Never blocks |
| `close()` | Wake every waiter, refuse new pushes, keep delivering the backlog |

There are also `size()`, `empty()`, and `closed()`, all `const`. Payloads may be move-only. Nothing asserts on cross-thread ordering; what is checked is that every item is consumed exactly once.

This exercise needs `-pthread`, so it ships a [`run.sh`](../../../../../practice/build/systems/cpp/05-threadsafe-queue/run.sh):

```bash
./run.sh
# ok  cpp/05-threadsafe-queue  ... checks passed
```

Or by hand, from this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 -pthread main.cpp -o queue && ./queue
```

Passing once means little for concurrent code, which is why the suite runs three times. ThreadSanitizer finds races the tests cannot:

```bash
c++ -std=c++20 -Wall -Wextra -pthread -fsanitize=thread -O1 main.cpp -o queue && ./queue
```

## Waiting and waking

Use the predicate form of `wait`. `cv.wait(lock, pred)` is exactly `while (!pred()) cv.wait(lock);`, and the loop is mandatory: a bare `wait` can return spuriously, and several waiters can be woken by one `notify` with only the first to reacquire the mutex finding the condition true.

Unlock before notifying. If you notify while still holding the mutex, the woken thread immediately blocks trying to acquire it, costing a second context switch for nothing. It is a performance bug, not a correctness one, which is exactly why it survives code review.

## Closing the queue

Use `notify_all` on close, never `notify_one`. Every blocked thread must wake to observe the closure. Waking one leaves the others parked forever, and any `join` on them deadlocks. `test_close_wakes_blocked_consumers` blocks four consumers and joins them, so a `notify_one` implementation hangs the test.

Closed does not mean empty. A producer whose `push` returned `true` is entitled to have that element consumed. So `pop` must drain the backlog before it starts reporting `nullopt`.

## Const members and identity

`mutable std::mutex` is not a hack. `size()` and `closed()` are logically const but must lock, and `mutable` is the standard way to say "this member is not part of the object's observable state".

The queue is neither copyable nor movable, and it should not be. A mutex is neither, and a queue that could be moved out from under a waiting thread would be a data race by construction.

## Deriving the design

Six observations build the class.

One mutex guards all shared state: the `std::queue`, the capacity, and the `closed_` flag. Every member function locks it first.

There are two reasons to sleep, so there are two condition variables. Producers wait for "room or closed" on `not_full_`; consumers wait for "an element or closed" on `not_empty_`. Each successful operation notifies the other side.

Closure is part of every wait predicate. Otherwise a thread waiting for room or data would never learn the queue has closed.

After waking, decide by state, in the right order. In `push`, closed means refuse. In `pop`, an element present means deliver it, *even if closed*; only empty-and-closed means `nullopt`.

Values move, never copy. `push` takes `T` by value and moves it in; `pop` moves the front out before popping. That supports move-only payloads.

Notify after unlocking, so the woken thread can take the lock at once.

## Building it step by step

The blocking operations come first:

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

The full solution is in [`queue.hpp`](../../../../../practice/build/systems/cpp/05-threadsafe-queue/queue.hpp).

## Why this design

Every operation is O(1) under one lock (amortised, since `std::queue` over `std::deque` allocates in blocks).

A lock-free queue is the obvious alternative. Lock-free ring buffers avoid the mutex but cannot block: a consumer would spin or need a separate wake-up mechanism. For a blocking queue with backpressure, a mutex plus condition variables is simpler and correct.

Two condition variables are chosen over one, because one shared variable would force `notify_all` on every operation, since a wake-up could be meant for either side.

## What can go wrong

| Mistake | Consequence |
|---------|-------------|
| Bare `wait` without a predicate | Spurious or stolen wake-ups pop from an empty queue |
| `notify_one` in `close` | Other waiters sleep forever; joins hang |
| Returning `nullopt` as soon as the queue is closed | Accepted elements are never delivered |
| Setting `closed_` without the lock | A waiter can miss the notification between its check and its sleep |
| Notifying while holding the lock | Correct, but costs a second context switch per wake-up |
| Copying payloads | Move-only types cannot be queued |

> **Warning:** `try_push` takes its argument by value, so a failed `try_push(std::move(x))` has already moved `x` into the parameter and destroys it. A caller that must keep the value on failure needs an overload taking `T&&` that only moves on success. This is a trade the reference accepts for simplicity.

## Where this leads

Add `pop_for(timeout)` using `wait_for`, and notice that the predicate overload handles the spurious-wake-up-plus-deadline interaction that hand-rolled timeout loops almost always get wrong. Then try replacing the `std::queue` with a fixed ring buffer, as the C [thread pool](../../18-c/07-thread-pool/README.md) uses, to remove the per-element allocation. In a larger system this queue is the hand-off point between a network reader and a pool of workers.

## Key terms

- **Blocking queue**: a queue whose operations wait, rather than fail, when it is full or empty.
- **`std::unique_lock`**: an RAII lock that can be unlocked early and is required by `condition_variable::wait`.
- **Predicate wait**: `cv.wait(lock, pred)`, which loops until `pred()` is true.
- **Lost wake-up**: a notification that arrives between a waiter's predicate check and its sleep, and is therefore missed.
- **`mutable`**: a member that may be modified inside `const` member functions because it is not part of observable state.

## Summary

- One mutex guards the queue, capacity, and closed flag; `not_full_` and `not_empty_` separate the two reasons to sleep.
- Every wait uses the predicate form, and every predicate includes `closed_`.
- `pop` delivers remaining elements after close and returns `nullopt` only when closed and empty.
- `close` sets the flag under the lock and calls `notify_all` on both condition variables.
- RAII locking means no exception or early return can leave the mutex held.

## Review questions

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
