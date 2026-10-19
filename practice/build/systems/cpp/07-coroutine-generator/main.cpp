// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
#include "generator.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
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

static Generator<int> range(int from, int to) {
  for (int i = from; i < to; ++i) co_yield i;
}

static Generator<int> empty_gen() { co_return; }

static Generator<int> naturals() {
  int i = 0;
  while (true) co_yield i++;  // infinite, and safe because of initial_suspend
}

static Generator<std::string> words() {
  co_yield "alpha";
  co_yield "beta";
  co_yield "gamma";
}

static void test_basic_sequence() {
  auto g = range(0, 5);
  std::vector<int> got;
  while (g.next()) got.push_back(g.value());

  CHECK(got.size() == 5, "five values produced");
  for (int i = 0; i < 5; ++i) CHECK(got[static_cast<std::size_t>(i)] == i, "in order");
  CHECK(g.done(), "the generator reports done");
  CHECK(!g.next(), "and stays done when asked again");
}

static void test_empty_generator() {
  auto g = empty_gen();
  CHECK(!g.next(), "an empty generator yields nothing");
  CHECK(g.done(), "and is immediately done");

  int n = 0;
  for (int v : empty_gen()) {
    (void)v;
    ++n;
  }
  CHECK(n == 0, "range-for over an empty generator runs zero times");
}

// The body must not begin executing until the first next(). If
// initial_suspend returned suspend_never, constructing `naturals()` would run
// straight into an infinite loop and hang before this line finished.
static void test_laziness_makes_infinite_safe() {
  auto g = naturals();
  CHECK(!g.done(), "an unstarted generator is not done");

  // Merely constructing it did not run the body. Take ten of infinitely many.
  std::vector<int> first_ten;
  for (int i = 0; i < 10; ++i) {
    CHECK(g.next(), "an infinite generator always has a next value");
    first_ten.push_back(g.value());
  }
  CHECK(first_ten.size() == 10, "took ten values from an infinite sequence");
  CHECK(first_ten[0] == 0 && first_ten[9] == 9, "and they are the first ten");
  // g is abandoned here with the coroutine suspended mid-loop; the destructor
  // must free the frame anyway.
}

static void test_interleaving_is_real() {
  // The generator's execution genuinely interleaves with the consumer's,
  // rather than the body running to completion and buffering. A side effect
  // ordered between two next() calls proves it.
  static std::vector<std::string> log;
  log.clear();

  auto tracer = []() -> Generator<int> {
    log.push_back("gen: before 1");
    co_yield 1;
    log.push_back("gen: before 2");
    co_yield 2;
    log.push_back("gen: done");
  };

  auto g = tracer();
  CHECK(log.empty(), "nothing ran before the first next()");

  g.next();
  log.push_back("main: got 1");
  g.next();
  log.push_back("main: got 2");
  g.next();

  const std::vector<std::string> expected = {
      "gen: before 1", "main: got 1", "gen: before 2", "main: got 2", "gen: done"};
  CHECK(log == expected, "generator and consumer interleave, they do not batch");
}

static void test_range_for() {
  int sum = 0;
  int count = 0;
  for (int v : range(1, 101)) {
    sum += v;
    ++count;
  }
  CHECK(count == 100, "range-for visited every element");
  CHECK(sum == 5050, "and their sum is right");

  std::vector<std::string> collected;
  for (const std::string& w : words()) collected.push_back(w);
  CHECK(collected.size() == 3, "non-trivial value types work");
  CHECK(collected[0] == "alpha" && collected[2] == "gamma", "with the right values");
}

// An exception thrown inside the body must surface on the consumer's side,
// not call std::terminate.
static void test_exception_propagates_to_the_consumer() {
  auto thrower = []() -> Generator<int> {
    co_yield 1;
    co_yield 2;
    throw std::runtime_error("generator failed");
  };

  auto g = thrower();
  CHECK(g.next() && g.value() == 1, "first value arrives");
  CHECK(g.next() && g.value() == 2, "second value arrives");

  bool caught = false;
  try {
    g.next();
  } catch (const std::runtime_error& e) {
    caught = true;
    CHECK(std::string(e.what()) == "generator failed", "the original exception is rethrown");
  }
  CHECK(caught, "the exception reached the consumer");
}

// Generators compose: one can consume another, which is where laziness earns
// its keep. Nothing here builds an intermediate container.
static Generator<int> take(Generator<int> src, int n) {
  int taken = 0;
  while (taken < n && src.next()) {
    co_yield src.value();
    ++taken;
  }
}

static Generator<int> squares_of(Generator<int> src) {
  while (src.next()) co_yield src.value() * src.value();
}

static void test_composition() {
  std::vector<int> got;
  for (int v : take(squares_of(naturals()), 5)) got.push_back(v);

  CHECK(got.size() == 5, "composed generators produce the right count");
  const std::vector<int> want = {0, 1, 4, 9, 16};
  CHECK(got == want, "squares of the first five naturals");
}

// The frame is a unique resource. Moving must transfer it exactly once.
static void test_move_semantics() {
  auto a = range(0, 3);
  CHECK(a.next() && a.value() == 0, "first value from the original");

  auto b = std::move(a);
  CHECK(a.done(), "the moved-from generator is empty");
  CHECK(!a.next(), "and yields nothing");
  CHECK(b.next() && b.value() == 1, "the target resumes where the source left off");

  Generator<int> c;
  CHECK(c.done(), "a default-constructed generator is done");
  c = std::move(b);
  CHECK(c.next() && c.value() == 2, "move assignment transfers the frame");
}

// Abandoning a generator mid-sequence must still free its frame. Under
// AddressSanitizer a missing handle.destroy() shows up here as a leak.
static void test_abandonment_frees_the_frame() {
  for (int i = 0; i < 1000; ++i) {
    auto g = naturals();
    g.next();
    g.next();
    // dropped while suspended, 1000 times over
  }
  CHECK(true, "abandoning suspended generators did not leak");
}

static void test_values_are_not_shared_between_generators() {
  auto g1 = range(0, 3);
  auto g2 = range(100, 103);

  CHECK(g1.next() && g1.value() == 0, "first generator");
  CHECK(g2.next() && g2.value() == 100, "second generator is independent");
  CHECK(g1.next() && g1.value() == 1, "and they do not share state");
  CHECK(g2.value() == 100, "an untouched generator keeps its current value");
}

int main() {
  test_basic_sequence();
  test_empty_generator();
  test_laziness_makes_infinite_safe();
  test_interleaving_is_real();
  test_range_for();
  test_exception_propagates_to_the_consumer();
  test_composition();
  test_move_semantics();
  test_abandonment_frees_the_frame();
  test_values_are_not_shared_between_generators();
  std::printf("ok  cpp/07-coroutine-generator  %d checks passed\n", checks);
  return 0;
}
