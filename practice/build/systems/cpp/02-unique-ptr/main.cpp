// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
#include "unique_ptr.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

struct Counted {
  static int live;
  static int destroyed;
  int value;
  explicit Counted(int v = 0) : value(v) { ++live; }
  ~Counted() {
    --live;
    ++destroyed;
  }
};
int Counted::live = 0;
int Counted::destroyed = 0;

// ---- compile-time properties ----------------------------------------------
//
// These are static_asserts rather than runtime checks because the whole value
// of the type is that the compiler rejects the misuse. A UniquePtr that is
// copyable would still pass every runtime test right up until it double-frees.

static_assert(!std::is_copy_constructible_v<UniquePtr<Counted>>,
              "UniquePtr must not be copy constructible");
static_assert(!std::is_copy_assignable_v<UniquePtr<Counted>>,
              "UniquePtr must not be copy assignable");
static_assert(std::is_move_constructible_v<UniquePtr<Counted>>,
              "UniquePtr must be move constructible");
static_assert(std::is_move_assignable_v<UniquePtr<Counted>>,
              "UniquePtr must be move assignable");
static_assert(std::is_nothrow_move_constructible_v<UniquePtr<Counted>>,
              "moving must be noexcept, or containers will copy instead");
// Not convertible to bool implicitly: `int x = p;` must not compile.
static_assert(!std::is_convertible_v<UniquePtr<Counted>, bool>,
              "operator bool must be explicit");

static void test_basic_ownership() {
  Counted::live = 0;
  {
    UniquePtr<Counted> p(new Counted(7));
    CHECK(Counted::live == 1, "the object exists");
    CHECK(static_cast<bool>(p), "a non-null pointer is truthy");
    CHECK(p->value == 7, "operator-> reaches the object");
    CHECK((*p).value == 7, "operator* dereferences it");
    CHECK(p.get() != nullptr, "get returns the raw pointer");
  }
  CHECK(Counted::live == 0, "leaving scope destroyed it");
}

static void test_default_and_null() {
  UniquePtr<Counted> p;
  CHECK(!p, "a default-constructed pointer is falsy");
  CHECK(p.get() == nullptr, "and holds nullptr");
  CHECK(p == nullptr, "and compares equal to nullptr");

  UniquePtr<Counted> q(nullptr);
  CHECK(!q, "explicitly null is falsy");

  // Destroying a null pointer must not call the deleter on nullptr.
  Counted::live = 0;
  { UniquePtr<Counted> r; }
  CHECK(Counted::live == 0, "destroying an empty pointer is harmless");
}

static void test_move_transfers_ownership() {
  Counted::live = 0;
  Counted::destroyed = 0;
  {
    UniquePtr<Counted> a(new Counted(1));
    Counted* raw = a.get();

    UniquePtr<Counted> b = std::move(a);
    CHECK(b.get() == raw, "the move target holds the original pointer");
    CHECK(a.get() == nullptr, "and the source is emptied");
    CHECK(!a, "so the source is falsy");
    CHECK(Counted::live == 1, "the object was not duplicated");
    CHECK(Counted::destroyed == 0, "nor destroyed by the move");
  }
  CHECK(Counted::live == 0, "one destruction at the end, not two");
  CHECK(Counted::destroyed == 1, "exactly one destruction total");
}

// Move-assigning over a live pointer must destroy what was there. Forgetting
// is a silent leak that no value-based test would see.
static void test_move_assignment_destroys_the_old_target() {
  Counted::live = 0;
  {
    UniquePtr<Counted> a(new Counted(1));
    UniquePtr<Counted> b(new Counted(2));
    CHECK(Counted::live == 2, "two objects");

    b = std::move(a);
    CHECK(Counted::live == 1, "assigning over b destroyed b's old object");
    CHECK(b->value == 1, "and b now holds a's");
    CHECK(!a, "and a is empty");
  }
  CHECK(Counted::live == 0, "nothing leaked");
}

static void test_self_move_assignment() {
  Counted::live = 0;
  {
    UniquePtr<Counted> p(new Counted(5));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
    // The naive implementation resets before taking the source's pointer,
    // which on self-assignment deletes the object and then adopts the
    // dangling pointer.
    p = std::move(p);
#pragma GCC diagnostic pop
    CHECK(Counted::live == 1, "self move-assignment did not destroy the object");
    CHECK(p && p->value == 5, "and the pointer still works");
  }
  CHECK(Counted::live == 0, "nothing leaked");
}

static void test_reset() {
  Counted::live = 0;
  {
    UniquePtr<Counted> p(new Counted(1));
    p.reset(new Counted(2));
    CHECK(Counted::live == 1, "reset destroyed the old object");
    CHECK(p->value == 2, "and adopted the new one");

    p.reset();
    CHECK(Counted::live == 0, "reset with no argument just destroys");
    CHECK(!p, "leaving the pointer empty");

    p.reset();
    CHECK(Counted::live == 0, "resetting an empty pointer is harmless");
  }
}

// Resetting to the pointer already held must not delete it and then keep it.
static void test_self_reset() {
  Counted::live = 0;
  {
    UniquePtr<Counted> p(new Counted(3));
    Counted* raw = p.get();
    p.reset(raw);
    CHECK(Counted::live == 1, "self-reset did not destroy the object");
    CHECK(p.get() == raw && p->value == 3, "and the pointer is unchanged");
  }
  CHECK(Counted::live == 0, "nothing leaked");
}

static void test_release_gives_up_without_deleting() {
  Counted::live = 0;
  {
    UniquePtr<Counted> p(new Counted(9));
    Counted* raw = p.release();
    CHECK(Counted::live == 1, "release did NOT destroy the object");
    CHECK(!p, "but the pointer gave up ownership");
    CHECK(p.get() == nullptr, "and holds nullptr");
    CHECK(raw->value == 9, "the caller now owns a usable object");
    delete raw;  // the caller's responsibility now
  }
  CHECK(Counted::live == 0, "the manual delete cleaned up");
}

static void test_swap() {
  Counted::live = 0;
  {
    UniquePtr<Counted> a(new Counted(1));
    UniquePtr<Counted> b(new Counted(2));
    a.swap(b);
    CHECK(a->value == 2 && b->value == 1, "swap exchanged the pointers");
    CHECK(Counted::live == 2, "and destroyed nothing");
  }
  CHECK(Counted::live == 0, "both destroyed at scope exit");
}

// The point of RAII: an exception unwinding the stack still runs destructors,
// so the resource cannot leak on an error path. A raw `new` here would.
static void test_exception_safety() {
  Counted::live = 0;
  bool caught = false;
  try {
    UniquePtr<Counted> p(new Counted(1));
    CHECK(Counted::live == 1, "object created");
    throw std::runtime_error("boom");
  } catch (const std::runtime_error&) {
    caught = true;
  }
  CHECK(caught, "the exception propagated");
  CHECK(Counted::live == 0, "stack unwinding destroyed the object");
}

// A custom deleter is why the deleter is a template parameter rather than a
// hardcoded `delete`: the resource need not even be heap memory.
static int freed_with_custom = 0;
struct CountingDelete {
  void operator()(Counted* p) const {
    ++freed_with_custom;
    delete p;
  }
};

static void test_custom_deleter() {
  Counted::live = 0;
  freed_with_custom = 0;
  {
    UniquePtr<Counted, CountingDelete> p(new Counted(1));
    CHECK(Counted::live == 1, "object created");
  }
  CHECK(freed_with_custom == 1, "the custom deleter ran");
  CHECK(Counted::live == 0, "and it actually freed the object");

  // A stateful deleter must move with the pointer.
  int calls = 0;
  struct Stateful {
    int* counter;
    void operator()(Counted* p) const {
      ++(*counter);
      delete p;
    }
  };
  {
    UniquePtr<Counted, Stateful> a(new Counted(2), Stateful{&calls});
    UniquePtr<Counted, Stateful> b = std::move(a);
    CHECK(b.get_deleter().counter == &calls, "the deleter moved with the pointer");
  }
  CHECK(calls == 1, "and the moved-to pointer used it exactly once");
}

static void test_array_specialisation() {
  Counted::live = 0;
  {
    UniquePtr<Counted[]> arr(new Counted[5]);
    CHECK(Counted::live == 5, "five objects created");
    arr[0].value = 10;
    arr[4].value = 40;
    CHECK(arr[0].value == 10 && arr[4].value == 40, "operator[] indexes the array");
  }
  // If this used `delete` rather than `delete[]`, only the first element's
  // destructor would run and live would be 4.
  CHECK(Counted::live == 0, "delete[] destroyed every element");
}

static void test_make_unique_ptr() {
  Counted::live = 0;
  {
    auto p = make_unique_ptr<Counted>(42);
    CHECK(p->value == 42, "arguments were forwarded to the constructor");
    CHECK(Counted::live == 1, "one object");

    auto s = make_unique_ptr<std::string>(3, 'z');
    CHECK(*s == "zzz", "multiple arguments forward correctly");
  }
  CHECK(Counted::live == 0, "nothing leaked");
}

// Move-only types must work in containers, which is the practical payoff:
// a vector of UniquePtr is how you hold polymorphic owned objects.
static void test_works_in_a_container() {
  Counted::live = 0;
  {
    std::vector<UniquePtr<Counted>> v;
    for (int i = 0; i < 100; ++i) v.push_back(make_unique_ptr<Counted>(i));
    CHECK(Counted::live == 100, "the container holds them all");
    CHECK(v[50]->value == 50, "and they are intact after the vector reallocated");

    v.erase(v.begin() + 10);
    CHECK(Counted::live == 99, "erasing destroyed exactly one");
  }
  CHECK(Counted::live == 0, "the container destroyed the rest");
}

int main() {
  test_basic_ownership();
  test_default_and_null();
  test_move_transfers_ownership();
  test_move_assignment_destroys_the_old_target();
  test_self_move_assignment();
  test_reset();
  test_self_reset();
  test_release_gives_up_without_deleting();
  test_swap();
  test_exception_safety();
  test_custom_deleter();
  test_array_specialisation();
  test_make_unique_ptr();
  test_works_in_a_container();
  std::printf("ok  cpp/02-unique-ptr  %d checks passed\n", checks);
  return 0;
}
