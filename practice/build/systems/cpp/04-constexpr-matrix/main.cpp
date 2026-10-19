// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
//
// Most of these are static_assert, so a wrong implementation fails to COMPILE.
// That is the point of the exercise: the errors arrive before the program does.
#include "matrix.hpp"

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

using M22 = Matrix<int, 2, 2>;
using M23 = Matrix<int, 2, 3>;
using M32 = Matrix<int, 3, 2>;

constexpr M23 a{{1, 2, 3, 4, 5, 6}};
constexpr M23 b{{10, 20, 30, 40, 50, 60}};
constexpr M32 c{{7, 8, 9, 10, 11, 12}};

// ---- compile-time behaviour -----------------------------------------------
//
// Every assertion below is evaluated by the compiler. If any is false, this
// file does not build, and there is no runtime to reach.

static_assert(a(0, 0) == 1, "element access at compile time");
static_assert(a(1, 2) == 6, "row-major indexing");
static_assert(M23::rows == 2 && M23::cols == 3, "dimensions are compile-time constants");

// Addition and subtraction.
static_assert((a + b)(0, 0) == 11, "addition");
static_assert((a + b)(1, 2) == 66, "addition across the whole matrix");
static_assert((b - a)(0, 1) == 18, "subtraction");
static_assert((a + b) - b == a, "adding then subtracting is the identity");

// Scalar multiplication.
static_assert((a * 3)(1, 1) == 15, "scalar multiplication");
static_assert(a * 0 == M23::zero(), "scaling by zero gives the zero matrix");
static_assert(a * 1 == a, "scaling by one changes nothing");

// Matrix product. 2x3 times 3x2 gives 2x2:
//   [1 2 3]   [ 7  8]   [ 58  64]
//   [4 5 6] * [ 9 10] = [139 154]
//             [11 12]
static_assert((a * c)(0, 0) == 58, "matrix product, element (0,0)");
static_assert((a * c)(0, 1) == 64, "element (0,1)");
static_assert((a * c)(1, 0) == 139, "element (1,0)");
static_assert((a * c)(1, 1) == 154, "element (1,1)");

// The product's TYPE is derived from the operands' shapes.
static_assert(std::is_same_v<decltype(a * c), M22>, "2x3 times 3x2 is a 2x2");
static_assert(std::is_same_v<decltype(c * a), Matrix<int, 3, 3>>, "3x2 times 2x3 is a 3x3");

// Transpose swaps the dimensions in the type, not just in the data.
static_assert(std::is_same_v<decltype(a.transpose()), M32>, "transposing a 2x3 gives a 3x2");
static_assert(a.transpose()(2, 1) == 6, "transpose moves elements correctly");
static_assert(a.transpose().transpose() == a, "transposing twice is the identity");

// Identity and trace exist only for square matrices.
static_assert(M22::identity()(0, 0) == 1, "identity has ones on the diagonal");
static_assert(M22::identity()(0, 1) == 0, "and zeros elsewhere");
static_assert(M22::identity().trace() == 2, "the trace of I(2) is 2");

constexpr M22 sq{{1, 2, 3, 4}};
static_assert(sq * M22::identity() == sq, "multiplying by I changes nothing");
static_assert(M22::identity() * sq == sq, "on either side");
static_assert(sq.trace() == 5, "trace sums the diagonal");

// Matrix exponentiation, entirely at compile time. Powers of the Fibonacci
// matrix [[1,1],[1,0]] have Fibonacci numbers as their entries, which makes
// this a self-checking example: F(10) = 55, F(11) = 89.
constexpr M22 fib{{1, 1, 1, 0}};
static_assert(matrix_pow(fib, 10)(0, 1) == 55, "the Fibonacci matrix computes F(10)");
static_assert(matrix_pow(fib, 11)(0, 1) == 89, "and F(11)");
static_assert(matrix_pow(fib, 1) == fib, "raising to the first power changes nothing");
static_assert(matrix_pow(fib, 0) == M22::identity(), "raising to the zeroth gives I");

// Associativity, checked by the compiler over real values.
constexpr M22 p{{2, 0, 1, 3}};
constexpr M22 q{{1, 4, 2, 5}};
constexpr M22 r{{3, 1, 0, 2}};
static_assert((p * q) * r == p * (q * r), "matrix multiplication is associative");
static_assert(p * q != q * p, "but not commutative");

// Distributivity.
static_assert(p * (q + r) == p * q + p * r, "multiplication distributes over addition");

// Transpose of a product reverses the order.
static_assert((a * c).transpose() == c.transpose() * a.transpose(),
              "(AB)^T equals B^T A^T");

// ---- shape mismatches must not compile ------------------------------------
//
// These use SFINAE to assert that the ill-formed expression is REJECTED. If a
// mismatched product compiled, the check below would fail to build, which is
// how you test that something does not compile.

template <typename A, typename B, typename = void>
struct can_multiply : std::false_type {};

template <typename A, typename B>
struct can_multiply<A, B, std::void_t<decltype(std::declval<A>() * std::declval<B>())>>
    : std::true_type {};

static_assert(can_multiply<M23, M32>::value, "2x3 times 3x2 is allowed");
static_assert(can_multiply<M32, M23>::value, "3x2 times 2x3 is allowed");
static_assert(can_multiply<M22, M23>::value, "2x2 times 2x3 is allowed: inner dims agree");
static_assert(!can_multiply<M23, M23>::value, "2x3 times 2x3 must NOT compile");
static_assert(!can_multiply<M22, M32>::value, "2x2 times 3x2 must NOT compile");
static_assert(!can_multiply<M32, M32>::value, "3x2 times 3x2 must NOT compile");

template <typename M, typename = void>
struct has_identity : std::false_type {};

template <typename M>
struct has_identity<M, std::void_t<decltype(M::identity())>> : std::true_type {};

static_assert(has_identity<M22>::value, "a square matrix has an identity");
static_assert(!has_identity<M23>::value, "a non-square matrix must NOT have one");

template <typename M, typename = void>
struct has_trace : std::false_type {};

template <typename M>
struct has_trace<M, std::void_t<decltype(std::declval<M>().trace())>> : std::true_type {};

static_assert(has_trace<M22>::value, "a square matrix has a trace");
static_assert(!has_trace<M23>::value, "a non-square matrix must NOT have one");

// ---- the same code at runtime ---------------------------------------------
//
// A constexpr function is not compile-time only. The whole point is that one
// implementation serves both, so these confirm the runtime path works with
// values the compiler could not have known.

static void test_runtime_use() {
  int seed = 0;
  std::sscanf("3", "%d", &seed);  // opaque to the compiler

  Matrix<int, 2, 2> m{{seed, 1, 1, 0}};
  CHECK(m(0, 0) == 3, "runtime construction");

  Matrix<int, 2, 2> doubled = m + m;
  CHECK(doubled(0, 0) == 6, "runtime addition");
  CHECK(doubled(1, 1) == 0, "across every element");

  Matrix<int, 2, 2> product = m * Matrix<int, 2, 2>::identity();
  CHECK(product == m, "runtime multiplication by the identity");

  Matrix<int, 2, 2> powered = matrix_pow(Matrix<int, 2, 2>{{1, 1, 1, 0}}, 20);
  CHECK(powered(0, 1) == 6765, "runtime exponentiation gives F(20)");

  m(0, 0) = 42;
  CHECK(m(0, 0) == 42, "elements are mutable at runtime");
}

static void test_larger_matrices() {
  Matrix<long, 8, 8> big{};
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) big(i, j) = static_cast<long>(i * 8 + j);
  }
  auto t = big.transpose();
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      CHECK(t(j, i) == big(i, j), "transpose is correct for every element");
    }
  }

  auto squared = big * big;
  // Row 0 of big is 0..7; column 0 of big is 0,8,16,...,56.
  long expect = 0;
  for (std::size_t k = 0; k < 8; ++k) expect += big(0, k) * big(k, 0);
  CHECK(squared(0, 0) == expect, "the product matches a hand-computed element");

  // The extra parens matter: the preprocessor does not understand angle
  // brackets, so the commas in Matrix<long, 8, 8> would otherwise be read as
  // macro argument separators.
  CHECK((Matrix<long, 8, 8>::identity().trace() == 8), "trace of I(8)");
}

// A non-arithmetic element type must work too: the class never assumes int.
static void test_other_element_types() {
  Matrix<double, 2, 2> d{{1.5, 2.5, 3.5, 4.5}};
  CHECK(d(0, 0) == 1.5, "doubles are stored exactly");
  CHECK((d + d)(1, 1) == 9.0, "and added");
  CHECK((d * 2.0)(0, 1) == 5.0, "and scaled");
  CHECK(d.trace() == 6.0, "and traced");
}

int main() {
  test_runtime_use();
  test_larger_matrices();
  test_other_element_types();
  // Every static_assert above already passed, or this would not have built.
  std::printf("ok  cpp/04-constexpr-matrix  %d runtime checks passed"
              " (plus every static_assert at compile time)\n",
              checks);
  return 0;
}
