// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
#include "expr.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

static int checks = 0;
#define CHECK(cond, what)                                                     \
  do {                                                                        \
    ++checks;                                                                 \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static bool close_to(double a, double b) { return std::fabs(a - b) < 1e-12; }

// ---- laziness, proved at compile time --------------------------------------
//
// The defining property is that `a + b` is NOT a Vec. If an implementation
// evaluates eagerly, these fail to build, which is the earliest possible
// moment to find out.

using AddNode = decltype(std::declval<Vec>() + std::declval<Vec>());
static_assert(!std::is_same_v<AddNode, Vec>,
              "a + b must not produce a Vec: that would be eager evaluation");
static_assert(std::is_base_of_v<Expr<AddNode>, AddNode>,
              "the result of + must itself be an expression");

using ChainNode = decltype(std::declval<Vec>() + std::declval<Vec>() *
                           std::declval<Vec>() - std::declval<Vec>());
static_assert(!std::is_same_v<ChainNode, Vec>, "a whole chain stays lazy");

// An expression node holds references, so it is pointer-sized, not array-sized.
// A node that had somehow captured a Vec by value would be far larger.
static_assert(sizeof(AddNode) <= 4 * sizeof(void*),
              "an expression node must be tiny: it holds references, not data");

// ---- a leaf that counts how often each element is read ----------------------
//
// This is how the tests prove that evaluating `a + b + c + d` walks each input
// exactly once rather than materialising intermediates. Defining it here, in
// the test file, also demonstrates that the Expr interface is open: any type
// that satisfies it joins the algebra.

static int reads = 0;

class CountingVec : public Expr<CountingVec> {
 public:
  explicit CountingVec(std::vector<double> d) : data_(std::move(d)) {}
  double operator[](std::size_t i) const {
    ++reads;
    return data_[i];
  }
  std::size_t size() const { return data_.size(); }

 private:
  std::vector<double> data_;
};

static void test_arithmetic_is_correct() {
  Vec a{1.0, 2.0, 3.0, 4.0};
  Vec b{10.0, 20.0, 30.0, 40.0};
  Vec c{100.0, 200.0, 300.0, 400.0};

  Vec sum = a + b;
  CHECK(sum.size() == 4, "the result has the right length");
  CHECK(close_to(sum[0], 11.0), "addition, element 0");
  CHECK(close_to(sum[3], 44.0), "addition, element 3");

  Vec diff = b - a;
  CHECK(close_to(diff[1], 18.0), "subtraction");

  Vec prod = a * b;
  CHECK(close_to(prod[2], 90.0), "elementwise multiplication");

  Vec quot = b / a;
  CHECK(close_to(quot[0], 10.0), "elementwise division");

  Vec neg = -a;
  CHECK(close_to(neg[0], -1.0), "negation");
  CHECK(close_to(neg[3], -4.0), "across every element");

  // A three-deep tree, which is where a naive implementation would build two
  // temporaries.
  Vec chain = a + b + c;
  CHECK(close_to(chain[0], 111.0), "a three-term chain");
  CHECK(close_to(chain[3], 444.0), "at the far end too");

  // Mixed operators, exercising precedence through the template machinery.
  Vec mixed = a + b * c - a;
  CHECK(close_to(mixed[0], 1.0 + 10.0 * 100.0 - 1.0), "mixed operators respect precedence");
  CHECK(close_to(mixed[1], 2.0 + 20.0 * 200.0 - 2.0), "for every element");
}

static void test_scalars() {
  Vec v{1.0, 2.0, 3.0};

  Vec doubled = v * Scalar(2.0);
  CHECK(doubled.size() == 3, "a scalar does not determine the length");
  CHECK(close_to(doubled[0], 2.0), "scaling on the right");
  CHECK(close_to(doubled[2], 6.0), "every element");

  // Scalar on the LEFT is the case that breaks an implementation which takes
  // the left operand's size unconditionally.
  Vec scaled_left = Scalar(3.0) * v;
  CHECK(scaled_left.size() == 3, "a left-hand scalar still gives the vector's length");
  CHECK(close_to(scaled_left[1], 6.0), "and the right values");

  Vec shifted = v + Scalar(10.0);
  CHECK(close_to(shifted[0], 11.0), "adding a scalar broadcasts it");
  CHECK(shifted.size() == 3, "without changing the length");

  Vec both = Scalar(1.0) + (v * Scalar(2.0));
  CHECK(close_to(both[2], 7.0), "scalars compose with vector expressions");
}

// The payoff: evaluating a chain reads each input array exactly once per
// element. An eager implementation reads them once per intermediate, and the
// count would be higher.
static void test_no_intermediate_materialisation() {
  CountingVec a({1.0, 2.0, 3.0, 4.0});
  CountingVec b({1.0, 1.0, 1.0, 1.0});
  CountingVec c({2.0, 2.0, 2.0, 2.0});
  CountingVec d({3.0, 3.0, 3.0, 3.0});

  reads = 0;
  Vec result = a + b + c + d;

  // Four inputs, four elements each: 16 reads if and only if the tree was
  // walked once per output element with nothing stored in between.
  CHECK(reads == 16, "each input element is read exactly once");
  CHECK(result.size() == 4, "the result is the right length");
  CHECK(close_to(result[0], 7.0), "and holds the right values");
  CHECK(close_to(result[3], 10.0), "for every element");

  // Building the expression WITHOUT assigning it must read nothing at all:
  // that is what "lazy" means.
  reads = 0;
  auto lazy = a + b + c + d;
  CHECK(reads == 0, "building an expression evaluates nothing");

  // Only now does anything happen.
  Vec forced = lazy;
  CHECK(reads == 16, "assignment is what triggers evaluation");
  CHECK(close_to(forced[1], 8.0), "with the right result");
}

// Reading one element of an expression must not evaluate the others.
static void test_single_element_access_is_cheap() {
  CountingVec a({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
  CountingVec b({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  auto e = a + b;
  reads = 0;
  double one = e[3];
  CHECK(close_to(one, 5.0), "the element is correct");
  CHECK(reads == 2, "reading one output element read exactly one input from each side");
}

static void test_assignment_over_existing() {
  Vec a{1.0, 2.0, 3.0};
  Vec b{10.0, 20.0, 30.0};
  Vec target{0.0, 0.0};  // deliberately the wrong length

  target = a + b;
  CHECK(target.size() == 3, "assignment resizes to the expression's length");
  CHECK(close_to(target[2], 33.0), "and holds the result");

  target = a;  // a plain Vec, not an expression
  CHECK(target.size() == 3, "assigning a Vec still works");
  CHECK(close_to(target[0], 1.0), "with the right values");
}

static void test_large_vectors() {
  const std::size_t N = 100000;
  Vec a(N, 2.0);
  Vec b(N, 3.0);
  Vec c(N, 4.0);

  Vec result = a * b + c;
  CHECK(result.size() == N, "large expressions produce the right length");
  CHECK(close_to(result[0], 10.0), "and the right value at the start");
  CHECK(close_to(result[N - 1], 10.0), "and at the end");

  // Deeply nested, to be sure the recursion is genuinely compile-time and
  // does not blow up at runtime.
  Vec deep = a + b + c + a + b + c + a + b + c;
  CHECK(close_to(deep[0], 27.0), "a nine-term expression evaluates correctly");
}

static void test_self_referential_expression() {
  Vec a{1.0, 2.0, 3.0};
  Vec doubled = a + a;
  CHECK(close_to(doubled[0], 2.0), "an expression may use the same operand twice");
  CHECK(close_to(doubled[2], 6.0), "for every element");

  Vec squared = a * a;
  CHECK(close_to(squared[2], 9.0), "including under multiplication");
}

int main() {
  test_arithmetic_is_correct();
  test_scalars();
  test_no_intermediate_materialisation();
  test_single_element_access_is_cheap();
  test_assignment_over_existing();
  test_large_vectors();
  test_self_referential_expression();
  std::printf("ok  cpp/06-expression-template  %d checks passed\n", checks);
  return 0;
}
