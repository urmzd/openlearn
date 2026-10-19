// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
#include "vector.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

static int checks = 0;
#define CHECK(cond, what)                                                     \
  do {                                                                        \
    ++checks;                                                                 \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// A type that counts every operation, so the tests can assert on how many
// objects exist rather than merely on the values they hold. Most Vector bugs
// are invisible to a Vector<int> and obvious to this.
struct Tracked {
  static int constructed, destroyed, copied, moved;
  static void reset() { constructed = destroyed = copied = moved = 0; }
  static int live() { return constructed - destroyed; }

  int value;
  bool moved_from = false;

  explicit Tracked(int v = 0) : value(v) { ++constructed; }
  Tracked(const Tracked& o) : value(o.value) { ++constructed; ++copied; }
  Tracked(Tracked&& o) noexcept : value(o.value) {
    ++constructed;
    ++moved;
    o.moved_from = true;
  }
  Tracked& operator=(const Tracked& o) {
    value = o.value;
    ++copied;
    return *this;
  }
  Tracked& operator=(Tracked&& o) noexcept {
    value = o.value;
    ++moved;
    o.moved_from = true;
    return *this;
  }
  ~Tracked() { ++destroyed; }
};
int Tracked::constructed = 0;
int Tracked::destroyed = 0;
int Tracked::copied = 0;
int Tracked::moved = 0;

static void test_basics() {
  Vector<int> v;
  CHECK(v.empty(), "a fresh vector is empty");
  CHECK(v.size() == 0, "size 0");
  CHECK(v.capacity() == 0, "a fresh vector has not allocated");

  for (int i = 0; i < 100; ++i) v.push_back(i);
  CHECK(v.size() == 100, "every push counted");
  CHECK(v.capacity() >= 100, "capacity covers the size");
  for (int i = 0; i < 100; ++i) CHECK(v[i] == i, "elements survived growth");

  CHECK(v.front() == 0, "front");
  CHECK(v.back() == 99, "back");
  CHECK(v.at(50) == 50, "at");

  bool threw = false;
  try {
    (void)v.at(100);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw, "at() throws past the end");

  v.pop_back();
  CHECK(v.size() == 99 && v.back() == 98, "pop_back removes the last element");
}

// Growth must be geometric, or push_back is not amortised O(1).
static void test_growth_is_geometric() {
  Vector<int> v;
  for (int i = 0; i < 10000; ++i) v.push_back(i);
  CHECK(v.capacity() < 40000, "capacity stays within a constant factor of size");
  CHECK(v.capacity() >= v.size(), "capacity is never less than size");

  Vector<int> r;
  r.reserve(1000);
  CHECK(r.capacity() >= 1000, "reserve raises capacity");
  CHECK(r.size() == 0, "reserve does not change size");
  std::size_t before = r.capacity();
  r.reserve(10);
  CHECK(r.capacity() == before, "reserve never shrinks");
}

// The test a Vector<int> cannot fail: spare capacity must hold NO objects.
// An implementation built on new[] default-constructs the whole buffer, so
// `live` here would be the capacity rather than the size.
static void test_spare_capacity_holds_no_objects() {
  Tracked::reset();
  {
    Vector<Tracked> v;
    v.reserve(100);
    CHECK(Tracked::live() == 0, "reserving capacity constructs nothing");

    v.emplace_back(1);
    v.emplace_back(2);
    v.emplace_back(3);
    CHECK(Tracked::live() == 3, "only the pushed elements exist");
    CHECK(v.capacity() >= 100, "the spare capacity is still there");
  }
  CHECK(Tracked::live() == 0, "the destructor destroyed exactly the live elements");
}

static void test_destruction_is_exact() {
  Tracked::reset();
  {
    Vector<Tracked> v;
    for (int i = 0; i < 50; ++i) v.emplace_back(i);
    CHECK(Tracked::live() == 50, "fifty live objects");

    v.pop_back();
    CHECK(Tracked::live() == 49, "pop_back destroys exactly one");

    v.clear();
    CHECK(Tracked::live() == 0, "clear destroys them all");
    CHECK(v.size() == 0, "and size is zero");
    CHECK(v.capacity() > 0, "but capacity is retained");

    v.emplace_back(7);
    CHECK(Tracked::live() == 1, "the vector is reusable after clear");
    CHECK(v[0].value == 7, "and holds the right value");
  }
  CHECK(Tracked::live() == 0, "nothing leaked");
}

static void test_copy_is_deep() {
  Tracked::reset();
  {
    Vector<Tracked> a;
    for (int i = 0; i < 10; ++i) a.emplace_back(i);

    Vector<Tracked> b = a;  // copy construct
    CHECK(b.size() == 10, "the copy has the same size");
    CHECK(Tracked::live() == 20, "the copy really copied every element");
    CHECK(b.data() != a.data(), "and owns separate storage");

    b[0].value = 999;
    CHECK(a[0].value == 0, "mutating the copy does not touch the original");

    Vector<Tracked> c;
    c.emplace_back(42);
    c = a;  // copy assign, over an existing element
    CHECK(c.size() == 10, "copy assignment replaces the contents");
    CHECK(c[3].value == 3, "with the source's values");
    CHECK(Tracked::live() == 30, "and destroys what it replaced");
  }
  CHECK(Tracked::live() == 0, "nothing leaked");
}

static void test_move_steals_rather_than_copies() {
  Tracked::reset();
  {
    Vector<Tracked> a;
    for (int i = 0; i < 10; ++i) a.emplace_back(i);
    const Tracked* original = a.data();
    int copies_before = Tracked::copied;

    Vector<Tracked> b = std::move(a);
    CHECK(b.size() == 10, "the move target has the elements");
    CHECK(b.data() == original, "and the very same buffer: nothing was copied");
    CHECK(Tracked::copied == copies_before, "no element was copied by the move");
    CHECK(a.size() == 0, "the moved-from vector is empty");
    CHECK(a.data() == nullptr, "and owns nothing");

    // A moved-from vector must remain usable, not merely destructible.
    a.emplace_back(5);
    CHECK(a.size() == 1 && a[0].value == 5, "a moved-from vector still works");

    Vector<Tracked> c;
    c.emplace_back(1);
    c = std::move(b);
    CHECK(c.size() == 10, "move assignment transfers the elements");
    CHECK(b.size() == 0, "and empties the source");
  }
  CHECK(Tracked::live() == 0, "nothing leaked");
}

static void test_self_assignment() {
  Tracked::reset();
  {
    Vector<Tracked> v;
    for (int i = 0; i < 5; ++i) v.emplace_back(i);

    // Assigning a variable to itself is exactly what is under test here, so
    // the warnings that normally flag it as a typo are turned off rather than
    // worked around: routing through a reference would test a different thing.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-assign-overloaded"
#pragma GCC diagnostic ignored "-Wself-move"

    // The naive copy assignment destroys the contents before copying, which
    // on self-assignment reads objects it has already destroyed.
    v = v;
    CHECK(v.size() == 5, "self copy-assignment preserves the contents");
    for (int i = 0; i < 5; ++i) CHECK(v[i].value == i, "and the values");

    v = std::move(v);
    CHECK(v.size() == 5, "self move-assignment does not destroy the vector");

#pragma GCC diagnostic pop
  }
  CHECK(Tracked::live() == 0, "nothing leaked");
}

// emplace_back must construct in place from the arguments, without building a
// temporary and copying it.
static void test_emplace_constructs_in_place() {
  Tracked::reset();
  {
    Vector<Tracked> v;
    v.reserve(4);
    v.emplace_back(7);
    CHECK(Tracked::copied == 0, "emplace_back copied nothing");
    CHECK(Tracked::moved == 0, "and moved nothing");
    CHECK(v[0].value == 7, "the element was built from the arguments");

    Tracked& ref = v.emplace_back(8);
    CHECK(ref.value == 8, "emplace_back returns a reference to the new element");
    ref.value = 9;
    CHECK(v[1].value == 9, "and it refers into the vector");
  }

  // A type that cannot be copied at all must still work via emplace.
  Vector<std::unique_ptr<int>> ptrs;
  ptrs.emplace_back(std::make_unique<int>(42));
  ptrs.push_back(std::make_unique<int>(43));
  CHECK(ptrs.size() == 2, "move-only types can be stored");
  CHECK(*ptrs[0] == 42 && *ptrs[1] == 43, "and hold their values");
}

// Reallocation must preserve every element. With a throwing move constructor
// the implementation must fall back to copying, or a failed reallocation
// destroys data that cannot be recovered.
struct ThrowingMove {
  static int live;
  static int throw_after;  // -1 disables
  int value;

  explicit ThrowingMove(int v) : value(v) { ++live; }
  ThrowingMove(const ThrowingMove& o) : value(o.value) { ++live; }
  // Deliberately NOT noexcept.
  ThrowingMove(ThrowingMove&& o) : value(o.value) {
    if (throw_after == 0) throw std::runtime_error("move failed");
    if (throw_after > 0) --throw_after;
    ++live;
  }
  ~ThrowingMove() { --live; }
};
int ThrowingMove::live = 0;
int ThrowingMove::throw_after = -1;

static void test_realloc_preserves_elements_when_move_may_throw() {
  ThrowingMove::live = 0;
  ThrowingMove::throw_after = -1;
  {
    Vector<ThrowingMove> v;
    for (int i = 0; i < 20; ++i) v.push_back(ThrowingMove(i));
    CHECK(v.size() == 20, "twenty elements");
    for (int i = 0; i < 20; ++i) CHECK(v[i].value == i, "values survived reallocation");
  }
  CHECK(ThrowingMove::live == 0, "no leak with a throwing move type");
}

static void test_strings_and_iteration() {
  Vector<std::string> v;
  v.push_back("alpha");
  v.push_back("beta");
  v.emplace_back(3, 'x');
  CHECK(v.size() == 3, "three strings");
  CHECK(v[2] == "xxx", "emplace forwarded both arguments to the string constructor");

  std::string joined;
  for (const std::string& s : v) joined += s;
  CHECK(joined == "alphabetaxxx", "range-for iterates in order");

  int n = 0;
  for (auto it = v.begin(); it != v.end(); ++it) ++n;
  CHECK(n == 3, "explicit iteration visits every element");

  Vector<std::string> init{"a", "b", "c"};
  CHECK(init.size() == 3 && init[1] == "b", "initializer_list construction works");
}

static void test_swap() {
  Vector<int> a{1, 2, 3};
  Vector<int> b{9};
  const int* a_data = a.data();
  const int* b_data = b.data();

  a.swap(b);
  CHECK(a.size() == 1 && a[0] == 9, "swap exchanged the contents");
  CHECK(b.size() == 3 && b[0] == 1, "in both directions");
  CHECK(a.data() == b_data && b.data() == a_data, "by exchanging buffers, not copying");
}

int main() {
  test_basics();
  test_growth_is_geometric();
  test_spare_capacity_holds_no_objects();
  test_destruction_is_exact();
  test_copy_is_deep();
  test_move_steals_rather_than_copies();
  test_self_assignment();
  test_emplace_constructs_in_place();
  test_realloc_preserves_elements_when_move_may_throw();
  test_strings_and_iteration();
  test_swap();
  std::printf("ok  cpp/01-vector  %d checks passed\n", checks);
  return 0;
}
