// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
#include "function.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
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

static int free_function(int x) { return x * 2; }

struct Functor {
  int offset;
  int operator()(int x) const { return x + offset; }
};

// Counts lifetime events, so the tests can prove the callable is destroyed
// exactly once however it was stored.
struct Tracked {
  static int live;
  std::array<char, 64> padding{};  // deliberately too big for the buffer
  int value;

  explicit Tracked(int v) : value(v) { ++live; }
  Tracked(const Tracked& o) : value(o.value) { ++live; }
  Tracked(Tracked&& o) noexcept : value(o.value) { ++live; }
  ~Tracked() { --live; }
  int operator()(int x) const { return x + value; }
};
int Tracked::live = 0;

static void test_calls_every_kind_of_callable() {
  Function<int(int)> f = free_function;
  CHECK(f(21) == 42, "a free function pointer");

  Function<int(int)> g = [](int x) { return x + 1; };
  CHECK(g(41) == 42, "a captureless lambda");

  int captured = 10;
  Function<int(int)> h = [captured](int x) { return x + captured; };
  CHECK(h(32) == 42, "a capturing lambda");

  Function<int(int)> i = Functor{5};
  CHECK(i(37) == 42, "a functor with state");

  Function<int(int)> j = [&captured](int x) { return x + captured; };
  captured = 100;
  CHECK(j(0) == 100, "a lambda capturing by reference sees the update");
}

static void test_empty_and_bool() {
  Function<int(int)> f;
  CHECK(!f, "a default-constructed Function is falsy");

  bool threw = false;
  try {
    (void)f(1);
  } catch (const std::bad_function_call&) {
    threw = true;
  }
  CHECK(threw, "calling an empty Function throws bad_function_call");

  f = [](int x) { return x; };
  CHECK(static_cast<bool>(f), "assigning a callable makes it truthy");

  f = nullptr;
  CHECK(!f, "assigning nullptr empties it");
}

static void test_various_signatures() {
  Function<void()> v = [] {};
  v();  // must compile and run
  CHECK(true, "void() works");

  Function<std::string(const std::string&, int)> cat =
      [](const std::string& s, int n) {
        std::string out;
        for (int i = 0; i < n; ++i) out += s;
        return out;
      };
  CHECK(cat("ab", 3) == "ababab", "multiple arguments and a class return type");

  int side_effect = 0;
  Function<void(int)> setter = [&side_effect](int x) { side_effect = x; };
  setter(7);
  CHECK(side_effect == 7, "void return with arguments");

  // A reference return: the Function must not decay it to a value.
  static int global = 0;
  Function<int&()> ref = []() -> int& { return global; };
  ref() = 5;
  CHECK(global == 5, "a reference return type is preserved");
}

// The small buffer must actually be used, or every callback allocates.
static void test_small_buffer_optimisation() {
  Function<int(int)> small = [](int x) { return x; };
  CHECK(small.uses_inline_storage(), "a captureless lambda is stored inline");

  int a = 1;
  Function<int(int)> one_capture = [a](int x) { return x + a; };
  CHECK(one_capture.uses_inline_storage(), "a small capture is stored inline");

  Function<int(int)> fn_ptr = free_function;
  CHECK(fn_ptr.uses_inline_storage(), "a function pointer is stored inline");

  // Too big to fit: must spill to the heap rather than corrupt the buffer.
  std::array<char, 128> big{};
  big[0] = 3;
  Function<int(int)> large = [big](int x) { return x + big[0]; };
  CHECK(!large.uses_inline_storage(), "a large capture spills to the heap");
  CHECK(large(39) == 42, "and still calls correctly");
}

static void test_lifetime_is_exact() {
  Tracked::live = 0;
  {
    Function<int(int)> f = Tracked{10};
    CHECK(Tracked::live == 1, "exactly one callable is alive");
    CHECK(!f.uses_inline_storage(), "this one is too big for the buffer");
    CHECK(f(32) == 42, "and it calls correctly");
  }
  CHECK(Tracked::live == 0, "the callable was destroyed exactly once");

  // Reassignment must destroy the previous callable.
  Tracked::live = 0;
  {
    Function<int(int)> f = Tracked{1};
    CHECK(Tracked::live == 1, "one alive");
    f = Tracked{2};
    CHECK(Tracked::live == 1, "reassignment destroyed the first");
    CHECK(f(0) == 2, "and installed the second");
    f = nullptr;
    CHECK(Tracked::live == 0, "clearing destroyed it");
  }
}

// Moving must transfer ownership without duplicating or dropping the callable,
// for both storage strategies.
static void test_move_semantics() {
  // Inline case.
  int captured = 5;
  Function<int(int)> a = [captured](int x) { return x + captured; };
  CHECK(a.uses_inline_storage(), "stored inline");
  Function<int(int)> b = std::move(a);
  CHECK(!a, "the moved-from Function is empty");
  CHECK(b(37) == 42, "the target works");

  // Heap case, with lifetime counting.
  Tracked::live = 0;
  {
    Function<int(int)> c = Tracked{10};
    CHECK(Tracked::live == 1, "one alive");
    Function<int(int)> d = std::move(c);
    CHECK(Tracked::live == 1, "moving did not duplicate the callable");
    CHECK(!c, "the source is empty");
    CHECK(d(32) == 42, "the target works");

    Function<int(int)> e;
    e = std::move(d);
    CHECK(Tracked::live == 1, "move assignment did not duplicate it either");
    CHECK(e(0) == 10, "and it still works");
  }
  CHECK(Tracked::live == 0, "exactly one destruction overall");
}

static void test_self_move_assignment() {
  Tracked::live = 0;
  {
    Function<int(int)> f = Tracked{7};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
    f = std::move(f);
#pragma GCC diagnostic pop
    CHECK(Tracked::live == 1, "self move-assignment did not destroy the callable");
    CHECK(f && f(0) == 7, "and it still works");
  }
  CHECK(Tracked::live == 0, "nothing leaked");
}

static void test_swap() {
  Function<int(int)> a = [](int x) { return x + 1; };
  Function<int(int)> b = [](int x) { return x + 2; };
  a.swap(b);
  CHECK(a(0) == 2, "swap exchanged the callables");
  CHECK(b(0) == 1, "in both directions");
}

// Move-only captures must work, which they cannot if the implementation ever
// requires the callable to be copyable.
static void test_move_only_capture() {
  auto owned = std::make_unique<int>(42);
  Function<int()> f = [p = std::move(owned)]() { return *p; };
  CHECK(f() == 42, "a lambda capturing a unique_ptr works");

  Function<int()> g = std::move(f);
  CHECK(g() == 42, "and survives being moved");
}

// A vector of Functions exercises move-on-reallocation heavily.
static void test_in_a_container() {
  Tracked::live = 0;
  {
    std::vector<Function<int(int)>> fns;
    for (int i = 0; i < 200; ++i) {
      if (i % 2 == 0) {
        fns.emplace_back([i](int x) { return x + i; });  // inline
      } else {
        fns.emplace_back(Tracked{i});  // heap
      }
    }
    CHECK(fns.size() == 200, "all stored");
    CHECK(Tracked::live == 100, "exactly the heap ones are alive");

    int sum = 0;
    for (auto& f : fns) sum += f(0);
    CHECK(sum == 199 * 200 / 2, "every callable produced its value");
  }
  CHECK(Tracked::live == 0, "the container destroyed them all exactly once");
}

// Recursion through a Function: the classic case where the callable holds a
// reference to the Function holding it.
static void test_recursive_use() {
  Function<int(int)> fact;
  fact = [&fact](int n) { return n <= 1 ? 1 : n * fact(n - 1); };
  CHECK(fact(5) == 120, "a Function can call itself through a captured reference");
  CHECK(fact(10) == 3628800, "for larger inputs too");
}

int main() {
  test_calls_every_kind_of_callable();
  test_empty_and_bool();
  test_various_signatures();
  test_small_buffer_optimisation();
  test_lifetime_is_exact();
  test_move_semantics();
  test_self_move_assignment();
  test_swap();
  test_move_only_capture();
  test_in_a_container();
  test_recursive_use();
  std::printf("ok  cpp/08-function  %d checks passed\n", checks);
  return 0;
}
