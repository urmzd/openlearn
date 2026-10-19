// expr.hpp - expression templates: lazy evaluation with no runtime cost.
//
// The reference implementation. main.cpp tests it.
//
// The problem being solved: with ordinary operator overloading,
//
//     Vec d = a + b + c;
//
// builds a temporary for (a + b), allocates it, fills it, then builds another
// for the outer sum. Three vectors' worth of traffic to compute one. For large
// arrays that is memory bandwidth, which is the actual bottleneck.
//
// Expression templates make `a + b` return a tiny object that DESCRIBES the
// addition instead of performing it. Nothing happens until the result is
// assigned, and then one loop evaluates the entire tree element by element.
// The temporaries exist only in the type system.
#pragma once

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

// Forward declarations, needed by the storage trait below.
template <typename L, typename R, typename Op>
class BinaryExpr;
template <typename E, typename Op>
class UnaryExpr;
class Scalar;

// How a node holds each operand.
//
// The obvious choice is "always by reference", since copying arrays is exactly
// what this technique exists to avoid. It is also wrong, and produces the most
// famous bug in expression templates:
//
//     auto e = a + b + c;      // (a + b) is a TEMPORARY
//     Vec  v = e;              // ...which died at the semicolon above
//
// The outer node holds a reference to an intermediate that no longer exists.
// Assigning directly (`Vec v = a + b + c;`) is fine because everything lives
// until the end of that full expression, which is why the bug hides until
// somebody reaches for `auto`.
//
// The fix real libraries use: hold intermediate NODES by value, and only the
// user's containers by reference. Nodes are a couple of pointers, so copying
// them is free; containers are the things that must not be copied.
template <typename T>
struct ExprStorage {
  using type = const T&;  // leaves: Vec and anything else the user defines
};

template <typename L, typename R, typename Op>
struct ExprStorage<BinaryExpr<L, R, Op>> {
  using type = BinaryExpr<L, R, Op>;  // by value
};

template <typename E, typename Op>
struct ExprStorage<UnaryExpr<E, Op>> {
  using type = UnaryExpr<E, Op>;  // by value
};

template <>
struct ExprStorage<Scalar> {
  using type = Scalar;  // by value: Scalar(2.0) is almost always a temporary
};

// CRTP base. Every expression node inherits from Expr<Itself>, so the base can
// downcast to the derived type with no virtual call. That is the whole trick:
// static polymorphism, resolved at compile time, inlinable.
template <typename Derived>
struct Expr {
  // These are what make an Expr usable without knowing its concrete type.
  double operator[](std::size_t i) const {
    return static_cast<const Derived&>(*this)[i];
  }
  std::size_t size() const { return static_cast<const Derived&>(*this).size(); }

  const Derived& self() const { return static_cast<const Derived&>(*this); }
};

// A real, owning array. The only thing in this file that holds memory.
class Vec : public Expr<Vec> {
 public:
  Vec() = default;
  explicit Vec(std::size_t n, double fill = 0.0) : data_(n, fill) {}
  Vec(std::initializer_list<double> init) : data_(init) {}

  // Build from ANY expression. This is where lazy evaluation cashes in: one
  // loop over the whole tree, no temporaries.
  template <typename E>
  Vec(const Expr<E>& e) {
    const E& expr = e.self();
    data_.resize(expr.size());
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] = expr[i];
  }

  template <typename E>
  Vec& operator=(const Expr<E>& e) {
    const E& expr = e.self();
    data_.resize(expr.size());
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] = expr[i];
    return *this;
  }

  double operator[](std::size_t i) const { return data_[i]; }
  double& operator[](std::size_t i) { return data_[i]; }
  std::size_t size() const { return data_.size(); }

 private:
  std::vector<double> data_;
};

// A scalar, dressed up as an expression so that `v * 2.0` type-checks the same
// way `v * w` does. size() is 0 because a scalar has no length of its own; the
// binary node takes the length from its other operand.
class Scalar : public Expr<Scalar> {
 public:
  explicit Scalar(double v) : value_(v) {}
  double operator[](std::size_t) const { return value_; }
  std::size_t size() const { return 0; }

 private:
  double value_;
};

// ---- expression nodes ------------------------------------------------------
//
// Operand storage is chosen by ExprStorage above: containers by reference,
// intermediate nodes and scalars by value. That combination is what keeps
// these free AND keeps `auto e = a + b + c;` from dangling.

template <typename L, typename R, typename Op>
class BinaryExpr : public Expr<BinaryExpr<L, R, Op>> {
 public:
  BinaryExpr(const L& l, const R& r) : l_(l), r_(r) {}

  double operator[](std::size_t i) const {
    // The recursion happens here, at compile time. For (a+b)*c this expands
    // into a[i] + b[i] then times c[i], all inlined into one expression with
    // no function calls and no intermediate storage.
    return Op::apply(l_[i], r_[i]);
  }

  std::size_t size() const {
    // A Scalar reports 0, so the length comes from whichever operand is a
    // real array. Taking the left operand's size unconditionally breaks
    // `2.0 * v`.
    return l_.size() != 0 ? l_.size() : r_.size();
  }

 private:
  typename ExprStorage<L>::type l_;
  typename ExprStorage<R>::type r_;
};

template <typename E, typename Op>
class UnaryExpr : public Expr<UnaryExpr<E, Op>> {
 public:
  explicit UnaryExpr(const E& e) : e_(e) {}
  double operator[](std::size_t i) const { return Op::apply(e_[i]); }
  std::size_t size() const { return e_.size(); }

 private:
  typename ExprStorage<E>::type e_;
};

struct AddOp {
  static double apply(double a, double b) { return a + b; }
};
struct SubOp {
  static double apply(double a, double b) { return a - b; }
};
struct MulOp {
  static double apply(double a, double b) { return a * b; }
};
struct DivOp {
  static double apply(double a, double b) { return a / b; }
};
struct NegOp {
  static double apply(double a) { return -a; }
};

// ---- operators -------------------------------------------------------------
//
// Each returns a node rather than a Vec, which is what makes the whole thing
// lazy. Note the return type is spelled out: `auto` would work, but the
// explicit form makes it obvious that no array is being produced.

template <typename L, typename R>
BinaryExpr<L, R, AddOp> operator+(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, AddOp>(l.self(), r.self());
}

template <typename L, typename R>
BinaryExpr<L, R, SubOp> operator-(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, SubOp>(l.self(), r.self());
}

template <typename L, typename R>
BinaryExpr<L, R, MulOp> operator*(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, MulOp>(l.self(), r.self());
}

template <typename L, typename R>
BinaryExpr<L, R, DivOp> operator/(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, DivOp>(l.self(), r.self());
}

template <typename E>
UnaryExpr<E, NegOp> operator-(const Expr<E>& e) {
  return UnaryExpr<E, NegOp>(e.self());
}
