// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
//
// Nothing asserts on the ORDER items are consumed across threads: that is
// genuinely nondeterministic. What is deterministic is that every produced
// item is consumed exactly once, and that is what is checked.
#include "queue.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

static int checks = 0;
#define CHECK(cond, what)                                                     \
  do {                                                                        \
    ++checks;                                                                 \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static void test_single_threaded_basics() {
  BlockingQueue<int> q(4);
  CHECK(q.empty(), "a fresh queue is empty");
  CHECK(q.size() == 0, "size 0");
  CHECK(!q.closed(), "and open");

  CHECK(q.push(1), "push succeeds");
  CHECK(q.push(2), "push succeeds");
  CHECK(q.size() == 2, "size tracks pushes");

  auto a = q.pop();
  CHECK(a.has_value() && *a == 1, "pop returns the first element pushed");
  auto b = q.pop();
  CHECK(b.has_value() && *b == 2, "and then the second: FIFO order");
  CHECK(q.empty(), "the queue is drained");

  CHECK(!q.try_pop().has_value(), "try_pop on an empty queue yields nothing");
}

static void test_capacity_is_respected() {
  BlockingQueue<int> q(3);
  CHECK(q.try_push(1), "try_push succeeds with room");
  CHECK(q.try_push(2), "try_push succeeds with room");
  CHECK(q.try_push(3), "try_push fills the queue");
  CHECK(!q.try_push(4), "try_push fails when full rather than growing");
  CHECK(q.size() == 3, "the queue held its capacity");

  CHECK(q.try_pop().has_value(), "making room");
  CHECK(q.try_push(4), "try_push succeeds again once there is room");
}

// close() must wake blocked consumers, or a join on them hangs forever.
static void test_close_wakes_blocked_consumers() {
  BlockingQueue<int> q(4);
  std::atomic<int> woke{0};
  std::vector<std::thread> consumers;

  for (int i = 0; i < 4; ++i) {
    consumers.emplace_back([&] {
      auto v = q.pop();  // blocks: the queue is empty
      if (!v.has_value()) ++woke;
    });
  }

  // Give them a moment to actually block, then close.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  q.close();

  for (auto& t : consumers) t.join();  // hangs if close only notified one
  CHECK(woke == 4, "every blocked consumer woke and saw the closure");
}

// A closed queue must still deliver what was already accepted.
static void test_close_drains_before_reporting_empty() {
  BlockingQueue<int> q(10);
  for (int i = 0; i < 5; ++i) CHECK(q.push(i), "push succeeds");
  q.close();

  CHECK(!q.push(99), "pushing to a closed queue fails");
  CHECK(!q.try_push(99), "try_push too");

  for (int i = 0; i < 5; ++i) {
    auto v = q.pop();
    CHECK(v.has_value(), "a closed queue still yields its backlog");
    CHECK(*v == i, "in order");
  }
  CHECK(!q.pop().has_value(), "and only then reports exhaustion");
}

// The real test: many producers, many consumers, a queue far too small to
// hold the traffic, so producers block constantly and consumers starve
// constantly. Every item must be consumed exactly once.
static void test_many_producers_and_consumers() {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kPerProducer = 5000;
  constexpr int kTotal = kProducers * kPerProducer;

  BlockingQueue<int> q(8);  // deliberately tiny
  std::vector<std::atomic<int>> seen(kTotal);
  for (auto& s : seen) s.store(0);

  std::vector<std::thread> threads;
  std::atomic<int> consumed{0};

  for (int p = 0; p < kProducers; ++p) {
    threads.emplace_back([&, p] {
      for (int i = 0; i < kPerProducer; ++i) {
        bool ok = q.push(p * kPerProducer + i);
        if (!ok) std::abort();  // nothing should be refused before close()
      }
    });
  }

  for (int c = 0; c < kConsumers; ++c) {
    threads.emplace_back([&] {
      while (auto v = q.pop()) {
        seen[*v].fetch_add(1);
        consumed.fetch_add(1);
      }
    });
  }

  // Join producers, then close so consumers can finish.
  for (int i = 0; i < kProducers; ++i) threads[i].join();
  q.close();
  for (std::size_t i = kProducers; i < threads.size(); ++i) threads[i].join();

  CHECK(consumed == kTotal, "every item was consumed");
  for (int i = 0; i < kTotal; ++i) {
    CHECK(seen[i].load() == 1, "each item was consumed exactly once");
  }
  CHECK(q.empty(), "the queue is empty at the end");
}

// Backpressure: with capacity 1 the queue can never run ahead, so the
// consumer's progress bounds the producer's.
static void test_capacity_one_serialises() {
  BlockingQueue<int> q(1);
  std::atomic<int> max_seen{0};
  constexpr int kN = 2000;

  std::thread producer([&] {
    for (int i = 0; i < kN; ++i) q.push(i);
    q.close();
  });

  long long sum = 0;
  int count = 0;
  while (auto v = q.pop()) {
    sum += *v;
    ++count;
    int s = static_cast<int>(q.size());
    if (s > max_seen) max_seen = s;
  }
  producer.join();

  CHECK(count == kN, "every item arrived");
  CHECK(sum == static_cast<long long>(kN) * (kN - 1) / 2, "and their sum is right");
  CHECK(max_seen <= 1, "the queue never exceeded its capacity");
}

// Move-only payloads must work: the queue must move elements, not copy them.
static void test_move_only_payload() {
  BlockingQueue<std::unique_ptr<int>> q(4);
  CHECK(q.push(std::make_unique<int>(42)), "a move-only value can be pushed");

  auto v = q.pop();
  CHECK(v.has_value(), "and popped");
  CHECK(**v == 42, "with its value intact");

  std::thread producer([&] {
    for (int i = 0; i < 500; ++i) q.push(std::make_unique<int>(i));
    q.close();
  });

  int expect = 0;
  while (auto p = q.pop()) {
    CHECK(**p == expect, "move-only values arrive in order from one producer");
    ++expect;
  }
  producer.join();
  CHECK(expect == 500, "all of them arrived");
}

static void test_closing_twice_is_harmless() {
  BlockingQueue<int> q(2);
  q.close();
  q.close();
  CHECK(q.closed(), "the queue is closed");
  CHECK(!q.pop().has_value(), "and yields nothing");
}

int main() {
  // Concurrency bugs are probabilistic, so the whole suite runs several times.
  for (int round = 0; round < 3; ++round) {
    test_single_threaded_basics();
    test_capacity_is_respected();
    test_close_wakes_blocked_consumers();
    test_close_drains_before_reporting_empty();
    test_many_producers_and_consumers();
    test_capacity_one_serialises();
    test_move_only_payload();
    test_closing_twice_is_harmless();
  }
  std::printf("ok  cpp/05-threadsafe-queue  %d checks passed\n", checks);
  return 0;
}
