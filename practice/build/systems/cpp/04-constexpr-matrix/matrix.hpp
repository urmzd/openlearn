// matrix.hpp - a fixed-size matrix whose arithmetic runs at compile time.
//
// The reference implementation. main.cpp tests it, mostly with
// static_assert, which means a wrong implementation fails to BUILD rather
// than failing at runtime.
//
// Two separate ideas are in play and it is worth keeping them apart:
//
//   * Dimensions are template parameters, so a shape mismatch is a compile
//     error rather than a runtime check. Multiplying a 2x3 by a 4x5 is not a
//     bug you can have.
//   * The operations are constexpr, so the same code that runs at runtime can
//     also produce a value the compiler bakes into the binary.
#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

template <typename T, std::size_t Rows, std::size_t Cols>
class Matrix {
  static_assert(Rows > 0 && Cols > 0, "a matrix needs at least one row and column");

 public:
  using value_type = T;
  static constexpr std::size_t rows = Rows;
  static constexpr std::size_t cols = Cols;

  // Zero-initialised. constexpr, so `constexpr Matrix<int,2,2> m;` is legal.
  constexpr Matrix() : data_{} {}

  // Row-major, from a flat initializer list of exactly Rows*Cols entries.
  constexpr explicit Matrix(const std::array<T, Rows * Cols>& flat) : data_(flat) {}

  // ---- element access ---------------------------------------------------
  //
  // operator() rather than operator[], because before C++23 the subscript
  // operator took exactly one argument. m(1,2) is the idiomatic spelling and
  // the reason Eigen and every other C++ matrix library looks the way it does.

  constexpr T& operator()(std::size_t r, std::size_t c) { return data_[r * Cols + c]; }
  constexpr const T& operator()(std::size_t r, std::size_t c) const {
    return data_[r * Cols + c];
  }

  // ---- construction helpers ---------------------------------------------

  static constexpr Matrix zero() { return Matrix{}; }

  // The identity, which only exists for a square matrix. Requiring it here
  // means Matrix<int,2,3>::identity() is a compile error rather than a
  // runtime surprise.
  static constexpr Matrix identity()
    requires(Rows == Cols)
  {
    Matrix m{};
    for (std::size_t i = 0; i < Rows; ++i) m(i, i) = T{1};
    return m;
  }

  // ---- arithmetic -------------------------------------------------------

  constexpr Matrix operator+(const Matrix& rhs) const {
    Matrix out{};
    for (std::size_t i = 0; i < Rows * Cols; ++i) out.data_[i] = data_[i] + rhs.data_[i];
    return out;
  }

  constexpr Matrix operator-(const Matrix& rhs) const {
    Matrix out{};
    for (std::size_t i = 0; i < Rows * Cols; ++i) out.data_[i] = data_[i] - rhs.data_[i];
    return out;
  }

  constexpr Matrix operator*(const T& scalar) const {
    Matrix out{};
    for (std::size_t i = 0; i < Rows * Cols; ++i) out.data_[i] = data_[i] * scalar;
    return out;
  }

  // Matrix product. The inner dimension must agree, and it is checked by the
  // type system: this only participates in overload resolution when the
  // shapes line up, so a mismatch is a compile error naming the shapes.
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

  // Transpose returns a DIFFERENT type: Rows and Cols are swapped. This is
  // what makes dimension-as-type worth the trouble.
  constexpr Matrix<T, Cols, Rows> transpose() const {
    Matrix<T, Cols, Rows> out{};
    for (std::size_t i = 0; i < Rows; ++i) {
      for (std::size_t j = 0; j < Cols; ++j) out(j, i) = (*this)(i, j);
    }
    return out;
  }

  constexpr T trace() const
    requires(Rows == Cols)
  {
    T sum{};
    for (std::size_t i = 0; i < Rows; ++i) sum = sum + (*this)(i, i);
    return sum;
  }

  constexpr bool operator==(const Matrix& rhs) const {
    for (std::size_t i = 0; i < Rows * Cols; ++i) {
      if (!(data_[i] == rhs.data_[i])) return false;
    }
    return true;
  }

 private:
  std::array<T, Rows * Cols> data_;

  // Every instantiation is a distinct class, so Matrix<T,2,3> cannot reach
  // into Matrix<T,3,2>'s privates without this.
  template <typename, std::size_t, std::size_t>
  friend class Matrix;
};

// Raise a square matrix to a power by repeated squaring, entirely at compile
// time. O(log n) multiplications rather than n.
template <typename T, std::size_t N>
constexpr Matrix<T, N, N> matrix_pow(const Matrix<T, N, N>& m, std::size_t exp) {
  Matrix<T, N, N> result = Matrix<T, N, N>::identity();
  Matrix<T, N, N> base = m;
  while (exp > 0) {
    if (exp % 2 == 1) result = result * base;
    base = base * base;
    exp /= 2;
  }
  return result;
}
