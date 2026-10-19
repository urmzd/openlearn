# C++ 04: Compile-time matrix

**Concepts:** `constexpr`, dimensions as template parameters, `requires`
**Difficulty:** ⭐⭐⭐

A fixed-size matrix whose dimensions live in the type and whose arithmetic the
compiler can evaluate. The solution is the header [`matrix.hpp`](matrix.hpp).
Most of the test suite in [`main.cpp`](main.cpp) is `static_assert`, so a wrong
implementation **fails to build** rather than failing at runtime.

## The problem

| Member | Notes |
|--------|-------|
| `operator()(r, c)` | Element access. Not `operator[]`, see below |
| `operator+` `-` | Same-shape only, enforced by the type |
| `operator*(scalar)` | Scale |
| `operator*(Matrix<T, Cols, N>)` | Product. Shape checked at compile time |
| `transpose()` | Returns a *different type* with the dimensions swapped |
| `identity()`, `trace()` | Square matrices only, via `requires` |
| `matrix_pow(m, n)` | Repeated squaring, O(log n) multiplications |

Every operation must be usable in a constant expression, and the non-square
`Matrix<int,2,3>` must not have `identity()` or `trace()` at all.

## Build and run

From this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o matrix && ./matrix
# ok  cpp/04-constexpr-matrix  76 runtime checks passed (plus every static_assert at compile time)
```

If the build succeeds, every compile-time assertion already holds.

## What to notice

**Two separate ideas are doing the work, and they are worth keeping apart.**
Dimensions as template parameters means a shape mismatch is a *type* error:
multiplying a 2×3 by a 4×5 is not a bug you can have at runtime, because the
program will not build. `constexpr` is the orthogonal claim that the same code
can run during compilation. You can have either without the other.

**`transpose()` returning a different type is the payoff.** `Matrix<T,2,3>` and
`Matrix<T,3,2>` are unrelated classes, so the compiler tracks shapes through
every operation for free. That is why the tests can assert
`decltype(a * c) == Matrix<int,2,2>` and why `(AB)^T == B^T A^T` is checkable at
compile time.

**`requires(Rows == Cols)` beats a `static_assert` in the body.** Constraining
the member means `Matrix<int,2,3>::identity()` is *not a member at all* rather
than a member that fails when instantiated. The tests use a detection idiom to
assert exactly that: `has_identity<M23>::value` must be `false`. A
`static_assert` inside the function would make that test impossible to write,
because the mere attempt would be a hard error.

**Testing that something does *not* compile takes a detection idiom.** You
cannot write "assert this is a compile error" directly. `std::void_t` plus a
partial specialisation gives you a trait that is `true` only when the
expression is well formed, which turns "must not compile" into an ordinary
`static_assert`.

**`operator()` rather than `operator[]`, and there is a historical reason.**
Before C++23, the subscript operator took exactly one argument, so `m[1,2]` was
impossible. Every C++ matrix library, Eigen included, settled on `m(i,j)`.

**Watch for macro commas.** `CHECK(Matrix<long, 8, 8>::identity()..., "...")`
does not compile, because the preprocessor knows nothing about angle brackets
and reads those commas as argument separators. An extra pair of parentheses
fixes it.

**The Fibonacci matrix makes the tests self-checking.** Powers of `[[1,1],[1,0]]`
have Fibonacci numbers as entries, so `matrix_pow(fib, 10)(0,1) == 55` is an
assertion whose expected value comes from somewhere other than the
implementation being tested.

## The solution, step by step

### From first principles

1. **If the shape is in the type, mismatches cannot compile.** Make `Rows` and
   `Cols` template parameters. Then `+` can take `const Matrix&` (the *same*
   type), and only same-shape matrices can be added.
2. **A product's shape is derived from its operands.** An `R×C` matrix times a
   `C×K` matrix is `R×K`. Write `operator*` as a template over `K` that accepts
   only `Matrix<T, Cols, K>`: the shared `Cols` is the inner-dimension check.
3. **Fixed-size storage needs no allocation**, so `std::array<T, Rows*Cols>` in
   row-major order: element `(r, c)` is at `r * Cols + c`. Everything about it is
   usable in constant expressions.
4. **Constant evaluation needs every function `constexpr`** and the storage
   initialised (`data_{}` zero-initialises), since reading an uninitialised
   value is not allowed at compile time.
5. **Square-only operations are constrained, not asserted.** A `requires` clause
   removes the member from the class when the condition fails, which is both a
   clearer error and detectable by traits.
6. **Powers by repeated squaring.** `m^n` uses the binary digits of `n`:
   square the base each step and multiply it into the result when the current
   bit is 1. That is O(log n) multiplications instead of n.

### The code, annotated

Storage, construction, and access:

```cpp
template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix {
  static_assert(Rows > 0 && Cols > 0, "a matrix needs at least one row and column");

 public:
  // Zero-initialised, so it is valid in a constant expression.
  constexpr Matrix() : data_{} {}
  constexpr explicit Matrix(const std::array<T, Rows * Cols>& flat) : data_(flat) {}

  // Row-major indexing. Two arguments, hence operator() before C++23.
  constexpr T& operator()(std::size_t r, std::size_t c) { return data_[r * Cols + c]; }
  constexpr const T& operator()(std::size_t r, std::size_t c) const {
    return data_[r * Cols + c];
  }
  // ...
 private:
  std::array<T, Rows * Cols> data_;
};
```

The product, where the type system does the shape check:

```cpp
// Only Matrix<T, Cols, OtherCols> is accepted: the shared Cols IS the
// inner-dimension check. The return type is computed from the shapes.
template <std::size_t OtherCols>
constexpr Matrix<T, Rows, OtherCols> operator*(
    const Matrix<T, Cols, OtherCols>& rhs) const {
  Matrix<T, Rows, OtherCols> out{};
  for (std::size_t i = 0; i < Rows; ++i) {
    for (std::size_t j = 0; j < OtherCols; ++j) {
      T sum{};
      for (std::size_t k = 0; k < Cols; ++k) sum = sum + (*this)(i, k) * rhs(k, j);
      out(i, j) = sum;
    }
  }
  return out;
}

// A different type: Rows and Cols swap.
constexpr Matrix<T, Cols, Rows> transpose() const {
  Matrix<T, Cols, Rows> out{};
  for (std::size_t i = 0; i < Rows; ++i) {
    for (std::size_t j = 0; j < Cols; ++j) out(j, i) = (*this)(i, j);
  }
  return out;
}
```

`sum = sum + ...` rather than `+=` keeps the element-type requirements to `+`
and `*` only.

Constrained members:

```cpp
// Not a member at all unless the matrix is square.
static constexpr Matrix identity()
  requires(Rows == Cols)
{
  Matrix m{};
  for (std::size_t i = 0; i < Rows; ++i) m(i, i) = T{1};
  return m;
}

constexpr T trace() const
  requires(Rows == Cols)
{
  T sum{};
  for (std::size_t i = 0; i < Rows; ++i) sum = sum + (*this)(i, i);
  return sum;
}
```

Every instantiation is a different class, so `Matrix<T,2,3>` cannot touch
`Matrix<T,3,2>`'s private members without `template <typename, std::size_t,
std::size_t> friend class Matrix;`, which the class declares.

Repeated squaring:

```cpp
template <typename T, std::size_t N>
constexpr Matrix<T, N, N> matrix_pow(const Matrix<T, N, N>& m, std::size_t exp) {
  Matrix<T, N, N> result = Matrix<T, N, N>::identity();   // m^0
  Matrix<T, N, N> base = m;                               // m^(2^k)
  while (exp > 0) {
    if (exp % 2 == 1) result = result * base;   // this bit of exp is set
    base = base * base;
    exp /= 2;
  }
  return result;
}
```

The invariant is `result * base^exp == m^original_exp` at the top of every
iteration; when `exp` reaches 0, `result` is the answer.

Full solution: [`matrix.hpp`](matrix.hpp).

### Why this solution

- **Cost.** Addition O(R·C), product O(R·C·K), power O(N³ log n). All of it can
  be done by the compiler, costing nothing at runtime.
- **Versus runtime dimensions** (`std::vector` plus `rows`, `cols`): flexible,
  but shape errors become runtime checks or silent bugs, and nothing is
  `constexpr`-friendly before C++20's limited dynamic allocation.
- **`requires` versus `static_assert` versus SFINAE.** `static_assert` gives a
  hard error on use and cannot be detected; SFINAE (`enable_if`) works but is
  unreadable; `requires` is both detectable and readable.
- **Accepted trade.** Each shape is a separate instantiation, so many shapes
  mean more compile time and code size, and dimensions must be known at compile
  time.

### Explain it back

1. Why does `operator+` take `const Matrix&` rather than a template?

   <details><summary>Answer</summary>

   Inside the class, `Matrix` means this exact shape. Accepting only that type
   makes adding a different shape fail overload resolution, so the shape check
   costs nothing and happens at compile time.

   </details>

2. How does the product's signature reject a 2×3 times a 4×5?

   <details><summary>Answer</summary>

   It only accepts `Matrix<T, Cols, OtherCols>`, where `Cols` is the left
   operand's column count (3). A `Matrix<T,4,5>` cannot be deduced into
   `Matrix<T,3,OtherCols>`, so no `operator*` matches and the program does not
   compile.

   </details>

3. Why would `static_assert(Rows == Cols)` inside `identity()` break the
   `has_identity` test?

   <details><summary>Answer</summary>

   The member would exist for every shape, so the detection trait would see a
   well-formed expression and report `true`, and actually instantiating it would
   be a hard error rather than a substitution failure. `requires` removes the
   member, so the trait sees it is missing.

   </details>

4. Why does `matrix_pow` start `result` at the identity?

   <details><summary>Answer</summary>

   The identity is `m^0`, and multiplying by it changes nothing. That makes
   `matrix_pow(m, 0)` correct and lets each set bit simply multiply its power of
   `m` into the result.

   </details>

## Extending it

Add a `constexpr` determinant and inverse for small fixed sizes, which forces
you to decide what an integer matrix's inverse even means and pushes you toward
a rational element type. Then try making the element type a `concept` rather
than an unconstrained `typename`, so misuse produces a readable error instead of
a page of template instantiation backtrace.
