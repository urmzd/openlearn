// queue.hpp - a bounded, blocking, thread-safe queue.
//
// You write the marked regions. main.cpp tests them.
//
// This is the C++ counterpart to the C thread pool, and the interesting
// difference is that RAII does the lock management. There is no path through
// this file where a lock is taken and not released, including the paths where
// an exception is thrown, because unique_lock's destructor runs during
// unwinding.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(std::size_t capacity) : cap_(capacity) {}

  // Not copyable or movable: a mutex is neither, and a queue that could be
  // moved out from under a waiting thread would be a data race by design.
  BlockingQueue(const BlockingQueue&) = delete;
  BlockingQueue& operator=(const BlockingQueue&) = delete;

  // Block until there is room, then enqueue. Returns false if the queue was
  // closed while waiting.
  bool push(T value) {
    std::unique_lock lock(mutex_);

    // The predicate form of wait is the same as `while (!pred) wait(lock);`
    // and exists because a bare wait can return spuriously, and because
    // several waiters may be woken for one notify with only the first to
    // reacquire the lock finding the condition true.
    not_full_.wait(lock, [this] { return q_.size() < cap_ || closed_; });

    if (closed_) return false;

    q_.push(std::move(value));

    // Unlock BEFORE notifying. If the waiter wakes while this thread still
    // holds the mutex, it immediately blocks again on acquiring it, which
    // costs a second context switch for nothing. This is the "hurry up and
    // wait" pessimisation.
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Enqueue only if there is room right now. Never blocks.
  bool try_push(T value) {
    std::unique_lock lock(mutex_);
    if (closed_ || q_.size() >= cap_) return false;
    q_.push(std::move(value));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Block until an element is available. Returns nullopt once the queue is
  // both closed and drained, which is how consumers learn to stop.
  std::optional<T> pop() {
    std::unique_lock lock(mutex_);
    not_empty_.wait(lock, [this] { return !q_.empty() || closed_; });

    // Drain before reporting closure. A producer whose push returned true is
    // entitled to have that element consumed, so closed-and-non-empty must
    // still yield elements.
    if (q_.empty()) return std::nullopt;

    T value = std::move(q_.front());
    q_.pop();
    lock.unlock();
    not_full_.notify_one();
    return value;
  }

  // Take an element only if one is available right now.
  std::optional<T> try_pop() {
    std::unique_lock lock(mutex_);
    if (q_.empty()) return std::nullopt;
    T value = std::move(q_.front());
    q_.pop();
    lock.unlock();
    not_full_.notify_one();
    return value;
  }

  // Wake every waiter and refuse further pushes. Elements already queued are
  // still delivered.
  void close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
    }
    // notify_all, not notify_one: every blocked thread has to wake up to
    // observe the closure. Waking one leaves the rest parked forever, and any
    // join on them deadlocks.
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  bool closed() const {
    std::lock_guard lock(mutex_);
    return closed_;
  }

  std::size_t size() const {
    std::lock_guard lock(mutex_);
    return q_.size();
  }

  bool empty() const { return size() == 0; }

 private:
  mutable std::mutex mutex_;  // mutable, so the const observers can lock it
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::queue<T> q_;
  std::size_t cap_;
  bool closed_ = false;
};
