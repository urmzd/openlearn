# Expression templates

Write `d = a + b + c` for vectors with ordinary operator overloading and you get two temporary arrays, two extra passes over memory, and code several times slower than the hand-written loop. Expression templates fix this by having `+` return a description of the work rather than the work's result. The description is a tree of types; assigning it to a vector walks the tree once per element, so the whole expression computes in a single pass with no temporaries.

The technique powers Eigen, Blaze, and xtensor, and it rests on two ideas you will use elsewhere: the curiously recurring template pattern (CRTP), which gives you an interface without a vtable, and a careful decision about which operands to store by reference and which by value. Getting that decision wrong produces the most famous bug in expression templates, which the reference implementation itself had until AddressSanitizer caught it. This section assumes comfort with templates and operator overloading at the level of the [compile-time matrix](../04-constexpr-matrix/README.md). The solution is the header [`expr.hpp`](../../../../../practice/build/systems/cpp/06-expression-template/expr.hpp); [`main.cpp`](../../../../../practice/build/systems/cpp/06-expression-template/main.cpp) is the test suite.

> **Learning objectives**
> - Explain why naive operator overloading on arrays is bound by memory bandwidth.
> - Use CRTP to give expression nodes a shared interface with static dispatch.
> - Build a lazy expression tree that evaluates in one loop at assignment.
> - Choose by-reference or by-value storage per node type so `auto` expressions do not dangle.

## The interface

| Type | Role |
|------|------|
| `Expr<Derived>` | CRTP base. Anything satisfying it joins the algebra |
| `Vec` | The only type that owns memory |
| `Scalar` | A broadcast constant, reporting `size() == 0` |
| `BinaryExpr` / `UnaryExpr` | Nodes holding operands and an operation |
| `operator+ - * /`, unary `-` | Return nodes, never a `Vec` |

The constraints are: no intermediate arrays, every element of every input read exactly once per assignment, `Scalar` usable on either side, and `auto e = a + b + c;` must not dangle.

To build and run the tests from this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o expr && ./expr
# ok  cpp/06-expression-template  40 checks passed
```

Build once with `-g -fsanitize=address` as well: the dangling-temporary bug described below is exactly what AddressSanitizer reports.

## The cost of temporaries

The naive version's cost is memory bandwidth, not arithmetic. `a + b + c` with ordinary overloading allocates a temporary for `a + b`, fills it, then allocates another for the outer sum. On anything large, that traffic *is* the run time, which is why the major numerical libraries are all built on this technique.

## Static polymorphism with CRTP

CRTP is static polymorphism: the base downcasts to the derived type. `static_cast<const Derived&>(*this)` costs nothing and inlines, where a virtual call would defeat the whole purpose by preventing the compiler from collapsing the tree. You get an interface without a vtable.

The recursion happens at compile time, and the loop happens once. By the time `Vec::operator=` runs its single loop, `expr[i]` has expanded into `a[i] + b[i] + c[i]` as one inlined expression. `test_no_intermediate_materialisation` proves it by counting reads: four inputs of four elements must be exactly sixteen reads.

The interface is open. The tests define their own leaf type, `CountingVec`, which lives in `main.cpp`, inherits `Expr<CountingVec>`, and immediately works with every operator. Any type providing `operator[]` and `size()` participates.

## Deriving the design

Six observations build the library.

To avoid temporaries, delay the work. If `a + b` returns an object that *describes* the sum (references to `a` and `b`, and the operation), then `(a + b) + c` is a tree of descriptions. Only assignment to a `Vec` needs numbers, and it can compute element `i` of the whole tree directly.

Element access is the only operation the tree needs. A node's `[i]` is `op(left[i], right[i])`. Recursing through the tree is just nested calls, which the compiler inlines into one expression per element.

The node types must be known at compile time to inline. So every node is a distinct template type, and the shared interface is a CRTP base `Expr<Derived>` that casts to `Derived` statically, never virtually.

Operators accept any `Expr<L>` and `Expr<R>`. That one template overload per operator covers vectors, scalars, nodes, and user types alike.

Ownership decides storage. User containers outlive the expression and must not be copied: hold them by reference. Intermediate nodes are temporaries that may die before evaluation: hold them by value (they are just a couple of references, so copying is free). A trait, `ExprStorage<T>`, picks per type.

Length comes from a real array. A scalar has no length, so it reports 0 and a binary node takes the non-zero side.

## Building it step by step

The CRTP base:

```cpp
template <typename Derived>
struct Expr {
  // No virtual functions: the cast is resolved at compile time and inlines.
  double operator[](std::size_t i) const {
    return static_cast<const Derived&>(*this)[i];
  }
  std::size_t size() const { return static_cast<const Derived&>(*this).size(); }

  const Derived& self() const { return static_cast<const Derived&>(*this); }
};
```

The storage trait that prevents the dangling-reference bug:

```cpp
template <typename T>
struct ExprStorage {
  using type = const T&;            // leaves (Vec, user types): by reference
};

template <typename L, typename R, typename Op>
struct ExprStorage<BinaryExpr<L, R, Op>> {
  using type = BinaryExpr<L, R, Op>;   // intermediate nodes: by value
};

template <typename E, typename Op>
struct ExprStorage<UnaryExpr<E, Op>> {
  using type = UnaryExpr<E, Op>;       // by value
};

template <>
struct ExprStorage<Scalar> {
  using type = Scalar;                 // Scalar(2.0) is almost always a temporary
};
```

The node:

```cpp
template <typename L, typename R, typename Op>
class BinaryExpr : public Expr<BinaryExpr<L, R, Op>> {
 public:
  BinaryExpr(const L& l, const R& r) : l_(l), r_(r) {}

  double operator[](std::size_t i) const {
    // Recurses into l_ and r_; after inlining, (a+b)*c[i] is one expression.
    return Op::apply(l_[i], r_[i]);
  }

  std::size_t size() const {
    // Scalar reports 0, so take the side that is a real array.
    return l_.size() != 0 ? l_.size() : r_.size();
  }

 private:
  typename ExprStorage<L>::type l_;   // reference or value, chosen per type
  typename ExprStorage<R>::type r_;
};
```

The operator just builds a node:

```cpp
template <typename L, typename R>
BinaryExpr<L, R, AddOp> operator+(const Expr<L>& l, const Expr<R>& r) {
  return BinaryExpr<L, R, AddOp>(l.self(), r.self());   // no arithmetic here
}
```

And `Vec` is where evaluation happens, in one loop:

```cpp
template <typename E>
Vec(const Expr<E>& e) {
  const E& expr = e.self();
  data_.resize(expr.size());
  for (std::size_t i = 0; i < data_.size(); ++i) data_[i] = expr[i];
}
```

`operator=` has the same body. Because each element is read and written at the
same index, `a = a + b` is safe: `a[i]` is read before `a[i]` is overwritten.

The full solution is in [`expr.hpp`](../../../../../practice/build/systems/cpp/06-expression-template/expr.hpp).

## Why this design

Evaluating an expression with $k$ leaves over $n$ elements reads $k \cdot n$ values and writes $n$, with no allocation beyond the destination. Ordinary overloading would write $(k - 1) \cdot n$ extra values to temporaries.

Virtual nodes would also be lazy, but a virtual `operator[]` per node costs an indirect call per element per node and blocks vectorisation. All-by-reference storage is correct only when the whole expression is consumed within one full expression; `auto` makes it dangle.

The accepted trade is that error messages and types are huge, and aliasing beyond the same index (shifts, slices) would need an explicit evaluation step.

## What can go wrong

Storing operands by reference is the obvious choice, and it is wrong:

```cpp
auto e = a + b + c;   // (a + b) is a temporary...
Vec  v = e;           // ...that died at the semicolon above
```

Assigning directly is safe because every temporary lives to the end of that full expression. Reach for `auto` and the outer node holds a reference to an intermediate that no longer exists. That is why `ExprStorage` stores nodes by value.

`size()` cannot just take the left operand's. `Scalar` reports 0 so that a binary node can take its length from whichever side is a real array. Taking the left unconditionally works for `v * Scalar(2)` and breaks on `Scalar(2) * v`, which is why the tests do both.

> **Warning:** A temporary `Vec` used as a leaf, as in `auto e = Vec{1,2} + a;`, is still held by reference and still dangles. Eigen documents the same caveat. Name your vectors, or evaluate the expression in the same statement.

## Where this leads

Add a `Slice` node so `v[range(2,5)] = expr` works and see how quickly aliasing becomes a real problem: `v = v + shift(v, 1)` reads elements it has already overwritten. Eigen's answer is an explicit `.eval()` and a documented aliasing rule, and understanding why they could not simply detect it is the interesting part. The type-erasure section, [Function](../08-function/README.md), is the mirror image of this one: it hides types behind a uniform interface at run time, where expression templates expose them all at compile time.

## Key terms

- **Expression template**: a type that represents an unevaluated expression, evaluated later in one pass.
- **CRTP**: the curiously recurring template pattern, `class D : Base<D>`, letting the base call the derived type statically.
- **Lazy evaluation**: deferring computation until its result is needed.
- **Dangling reference**: a reference to an object whose lifetime has ended.
- **Full expression**: the outermost expression of a statement; temporaries live until it ends.

## Summary

- Operators return node objects describing the computation, and only assignment to `Vec` runs a loop.
- CRTP gives nodes a common interface with static dispatch, so the tree inlines into one expression per element.
- Leaves are stored by reference and intermediate nodes by value, chosen by the `ExprStorage` trait, so `auto` expressions stay valid.
- `Scalar` reports size 0 and binary nodes take the non-zero side, so scalars work on either side.
- Same-index aliasing (`a = a + b`) is safe; shifted aliasing and temporary leaves are not.

## Review questions

1. Why does `auto e = a + b + c; Vec v = e;` dangle if nodes store operands by
   reference?

   <details><summary>Answer</summary>

   The outer node refers to the inner node `a + b`, which is a temporary
   destroyed at the end of the `auto` declaration. When `v` is built, the outer
   node reads through a reference to a dead object.

   </details>

2. Why is it fine to store `Vec` operands by reference?

   <details><summary>Answer</summary>

   They are the user's named containers, which outlive the expression, and
   copying them would reintroduce exactly the memory traffic the technique
   removes. Nodes are cheap to copy; containers are not.

   </details>

3. Why does `Scalar::size()` return 0, and what would break if `BinaryExpr`
   always used the left size?

   <details><summary>Answer</summary>

   A scalar has no length of its own. With "left size always", `Scalar(2) * v`
   would report length 0 and the destination `Vec` would be resized to empty.
   Taking the non-zero side gives the vector's length either way.

   </details>

4. Why CRTP rather than a virtual base class?

   <details><summary>Answer</summary>

   With CRTP the concrete node type is known at compile time, so the whole tree
   inlines into one loop body. Virtual dispatch would call through a vtable for
   every element of every node and prevent that collapse.

   </details>
