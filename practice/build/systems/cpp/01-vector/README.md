# Vector

`std::vector` is the container you use most in C++ and the one whose implementation teaches the most about the language. Underneath its friendly interface are the rules that make C++ hard: allocation is not construction, an object exists only once its constructor returns, a move can throw, and a moved-from object must still be usable. This section builds a `std::vector` work-alike, `Vector<T>`, that obeys all of them.

The C [dynamic array](../../c/01-dynamic-array/README.md) solved growth, overflow, and allocation failure for `int`, where an element is just bytes. Here elements are objects with constructors and destructors, so every slot is either raw storage or a live `T`, and the code must always know which. By the end you will be able to explain why `new T[n]` is wrong for a vector, why marking a move constructor `noexcept` makes vectors of that type faster, and how copy-and-swap gives the strong exception guarantee. The whole solution is the header [`vector.hpp`](vector.hpp); [`main.cpp`](main.cpp) is the test suite.

> **Learning objectives**
> - Separate raw storage from object lifetime using `operator new`, `std::construct_at`, `std::destroy_at`, and `operator delete`.
> - Keep the invariant that exactly the first `size_` slots hold live objects, even when a constructor throws.
> - Implement growth with `std::move_if_noexcept` so a throwing move cannot lose elements.
> - Implement the rule of five, including copy-and-swap assignment and a move that leaves a valid empty source.

## The interface

The members to implement are:

| Member | Does |
|--------|------|
| `reserve(n)` | Raise capacity, never lower it |
| `emplace_back(args...)` | Construct in place from the arguments |
| `push_back(v)` | Copy or move one element in |
| `pop_back()` | Destroy the last element, keep the capacity |
| `clear()` | Destroy all elements, keep the capacity |
| The rule of five | Copy/move construct, copy/move assign, destructor |

On top of these come `size`, `capacity`, `empty`, `operator[]`, a bounds-checked `at`, `front`, `back`, `data`, `swap`, and pointer iterators. The constraints are what make it hard: `T` need not be default-constructible, spare capacity must hold no objects, every constructed object is destroyed exactly once, and growth must not lose elements even when `T`'s move constructor can throw.

To build and run the tests from this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o vector && ./vector
# ok  cpp/01-vector  189 checks passed
```

GCC prints one `-Wpragmas` warning for a Clang-only warning name in the self-assignment test; it is harmless. Add `-g -fsanitize=address,undefined` to check that nothing is destroyed twice or leaked.

The tests count objects, not values. A `Tracked` type records every construction, destruction, copy, and move, which is how a test can assert that a move copied *nothing* and that a copy really copied ten elements. Most vector bugs are invisible to value-based assertions and obvious to lifetime-based ones.

## Allocation is not construction

This distinction is the whole exercise. A vector holds `capacity` worth of *raw storage* and constructs objects only in the first `size` slots. Reach for `new T[n]` and you have default-constructed every spare slot: `Vector<T>` then requires `T` to be default-constructible, wastes work, and destroys objects nobody created. `test_spare_capacity_holds_no_objects` is what catches it, and a `Vector<int>` never could.

The same reasoning fixes the order of two lines. Increment `size_` *after* the constructor returns, not before. If `T`'s constructor throws, that element never existed, and a size already bumped means `clear()` and the destructor will run a destructor on raw memory.

## Growing when moves can throw

`std::move_if_noexcept` in `reserve` is not a micro-optimisation. If `T`'s move constructor can throw, moving elements into the new buffer can fail halfway, with the originals already moved-from and no way back. Copying instead keeps the source intact until the new buffer is complete. This is precisely why marking a move constructor `noexcept` makes vectors of that type measurably faster: without it, every reallocation silently degrades to copying.

## Copying, moving, and assignment

Copy-and-swap gets self-assignment and exception safety in one move. The naive copy assignment destroys the contents and then copies from the source, which on `v = v` reads objects it has already destroyed. Building the copy first means the operation either fully succeeds or leaves `*this` untouched.

A moved-from object must be valid, not merely destructible. The standard requires "valid but unspecified", and the test pushes onto a moved-from vector to prove it. Leaving stale pointers in the source and relying on nobody touching it works right up until someone reuses the variable.

## Deriving the design

Six observations build the class.

Three numbers describe the state. `data_` points at raw storage for `cap_` objects, of which the first `size_` are alive. The invariant every member keeps is that slots `[0, size_)` hold constructed `T`s and slots `[size_, cap_)` hold nothing.

So storage and objects need separate tools. `operator new(bytes, align)` gets raw memory and constructs nothing; `std::construct_at` builds one object in a slot; `std::destroy_at` ends one; `operator delete` frees raw memory. Using `new T[n]` would merge the first two and break the invariant.

Count an object only once it exists. Construct, *then* `++size_`. If the constructor throws, `size_` still describes exactly the live objects, and the destructor cleans up correctly.

Growth must not lose anything. Build the new buffer completely before touching the old one. If `T`'s move can throw, use copies (`move_if_noexcept`), so a failure leaves the old buffer fully intact; then destroy the partial new buffer and rethrow.

Doubling makes `push_back` amortised O(1), for the same reason as in the C [dynamic array](../../c/01-dynamic-array/README.md).

Ownership of the buffer is unique. A copy must build its own buffer (a deep copy). A move just steals the three numbers and leaves the source as an empty vector, so it allocates nothing and cannot throw.

## Building it step by step

`reserve` is where every hard rule meets:

```cpp
void reserve(size_type n) {
  if (n <= cap_) return;  // reserve never shrinks

  // Raw storage: operator new, not new[], which would construct n objects.
  void* raw = operator new(n * sizeof(T), std::align_val_t{alignof(T)});
  T* fresh = static_cast<T*>(raw);

  // Move across only if moving cannot throw; otherwise copy, so that a
  // failure leaves the originals untouched.
  size_type built = 0;
  try {
    for (; built < size_; ++built) {
      std::construct_at(fresh + built, std::move_if_noexcept(data_[built]));
    }
  } catch (...) {
    // Undo exactly what was built, free the new block, and let the caller
    // see the exception with *this unchanged.
    for (size_type i = 0; i < built; ++i) std::destroy_at(fresh + i);
    operator delete(raw, std::align_val_t{alignof(T)});
    throw;
  }

  // Only now is the old buffer expendable.
  for (size_type i = 0; i < size_; ++i) std::destroy_at(data_ + i);
  operator delete(static_cast<void*>(data_), std::align_val_t{alignof(T)});

  data_ = fresh;
  cap_ = n;
}
```

The aligned `operator new`/`operator delete` pair is used everywhere, so
over-aligned types work and every allocation is freed by its matching function.
`operator delete(nullptr, ...)` is a no-op, so the first `reserve` on an empty
vector needs no special case.

Insertion and removal around the `size_` invariant:

```cpp
template <typename... Args>
T& emplace_back(Args&&... args) {
  if (size_ == cap_) grow();                          // doubling
  std::construct_at(data_ + size_, std::forward<Args>(args)...);
  ++size_;   // only AFTER the constructor returned: if it threw, no object
  return data_[size_ - 1];
}

void pop_back() {
  --size_;
  std::destroy_at(data_ + size_);   // the slot becomes raw storage again
}

void clear() noexcept {
  for (size_type i = 0; i < size_; ++i) std::destroy_at(data_ + i);
  size_ = 0;                        // capacity is kept
}

void grow() { reserve(cap_ == 0 ? 1 : cap_ * 2); }
```

`std::forward` passes each argument with its original value category, so
`emplace_back(std::string("x"))` moves the string into the slot and
`emplace_back(3, 'a')` calls `T(3, 'a')` directly, with no temporary `T`.

The rule of five:

```cpp
Vector(const Vector& other) {
  reserve(other.size_);
  for (size_type i = 0; i < other.size_; ++i) {
    std::construct_at(data_ + i, other.data_[i]);   // construct, not assign
    ++size_;
  }
}

Vector(Vector&& other) noexcept {
  // Steal the buffer and leave the source a valid empty vector. No allocation.
  data_ = other.data_;
  size_ = other.size_;
  cap_ = other.cap_;
  other.data_ = nullptr;
  other.size_ = 0;
  other.cap_ = 0;
}

Vector& operator=(const Vector& other) {
  // Copy-and-swap: build the copy first, so a throw leaves *this untouched.
  if (this != &other) {
    Vector tmp(other);
    swap(tmp);          // tmp's destructor frees our old contents
  }
  return *this;
}

~Vector() {
  clear();   // destroy the live objects...
  operator delete(static_cast<void*>(data_), std::align_val_t{alignof(T)});
  // ...then free the raw storage. Two steps, because they are two things.
}
```

Move assignment (not quoted) is the destructor's two steps followed by the move
constructor's steal, guarded by `this != &other`.

The full solution is in [`vector.hpp`](vector.hpp).

## Why this design

`push_back` and `emplace_back` are amortised O(1), `operator[]` is O(1), `reserve` and copy are O(size), and move is O(1) and `noexcept`.

Raw storage beats `new T[n]`, which is shorter and wrong: it constructs objects in spare slots, requires a default constructor, and cannot destroy elements individually.

Calling `operator new` directly is chosen over `std::allocator<T>`. Using an allocator object is what `std::vector` does and what makes custom allocators possible, as the [memory pool allocator](../10-pool-allocator/README.md) section shows. Calling `operator new` directly keeps the lifetime rules visible without the allocator-traits machinery.

Copy-and-swap is chosen over element-wise assignment. Reusing existing elements with `=` can avoid reallocation, which is why `std::vector` does it, but it gives only the basic exception guarantee. Copy-and-swap always allocates and gives the strong guarantee.

## What can go wrong

The bugs the design prevents are summarised here.

| Mistake | Consequence |
|---------|-------------|
| `new T[n]` for the buffer | Spare slots hold default-constructed objects; `T` must be default-constructible |
| `++size_` before the constructor | A throwing constructor leaves a raw slot that the destructor will "destroy" |
| `std::move` instead of `move_if_noexcept` in `reserve` | A throwing move loses elements with no way back |
| Destroying old elements before the new buffer is complete | Nothing to fall back to if a later construction throws |
| Naive copy assignment | `v = v` reads objects it already destroyed |
| A move that leaves the source's pointers in place | Two owners; a double free |

This implementation also has two known limitations that a production vector closes.

> **Warning:** `v.push_back(v[0])` when `size_ == cap_` is undefined behaviour here. `grow()` destroys the old buffer before `construct_at` reads the argument, which still refers into it. `std::vector` constructs the new element in the new buffer before releasing the old one.

The second limitation is in the constructors. If a copy throws inside the copy constructor (or the `(n, value)` and initializer-list constructors), the elements already built and the buffer leak, because a constructor that did not finish never runs `~Vector`. Delegating to `Vector()` first, or wrapping the loop in `try`/`catch`, makes the object complete so cleanup happens.

## Where this leads

Two extensions are worth doing. Add `insert` and `erase` in the middle, which force you to think about iterator invalidation and about shifting a range of objects using move-assignment rather than construction. Then add an `Allocator` template parameter and discover why `std::vector`'s real signature is so much more complicated than this one; the [memory pool allocator](../10-pool-allocator/README.md) is a good allocator to plug in. The move-only ownership ideas carry straight into the next section, [UniquePtr](../02-unique-ptr/README.md).

## Key terms

- **Raw storage**: allocated memory in which no object has yet been constructed.
- **`std::construct_at` / `std::destroy_at`**: start and end the lifetime of one object at a given address.
- **Strong exception guarantee**: if an operation throws, the object is left exactly as it was before.
- **`std::move_if_noexcept`**: yields an rvalue if the move constructor cannot throw (or the type is not copyable), and a const lvalue otherwise.
- **Copy-and-swap**: implementing assignment by copying into a temporary and swapping, so failure cannot damage the target.
- **Rule of five**: a type that manages a resource defines (or deletes) the copy and move constructors, copy and move assignment, and destructor together.

## Summary

- Slots `[0, size_)` hold live objects and `[size_, cap_)` hold raw storage; separate tools allocate memory and construct objects.
- `size_` is incremented only after a constructor returns, so a throw never leaves an uncounted or overcounted slot.
- `reserve` builds the new buffer completely, moving only if moves are `noexcept`, before destroying the old one.
- Copy is a deep copy, move steals the buffer and leaves a valid empty source, and assignment uses copy-and-swap.
- Self-referencing `push_back` on a full vector and throwing copies in constructors are known gaps.

## Review questions

1. Why is `++size_` after `construct_at` and not before?

   <details><summary>Answer</summary>

   `size_` must count only live objects. If the constructor throws with `size_`
   already incremented, the destructor would call `~T()` on a slot that holds
   raw memory, which is undefined behaviour.

   </details>

2. What does `std::move_if_noexcept` choose for a type whose move constructor is
   not `noexcept` but which is copyable, and why?

   <details><summary>Answer</summary>

   It yields a const lvalue reference, so the copy constructor is used. If a move
   threw halfway, some originals would already be moved-from and the vector could
   not be restored. Copies leave the originals intact, so `reserve` can roll back
   and give the strong guarantee.

   </details>

3. Why must the old elements be destroyed only after the whole new buffer is
   built?

   <details><summary>Answer</summary>

   Until the last element is copied or moved successfully, the old buffer is the
   only complete copy of the data. Destroying it earlier would leave nothing to
   fall back to if a later construction threw.

   </details>

4. Why does the move constructor leave `other` with `nullptr, 0, 0` instead of
   leaving it alone?

   <details><summary>Answer</summary>

   Two objects would own one buffer, and both destructors would free it. Resetting
   the source also makes it a genuinely valid empty vector, which is what the
   test relies on when it pushes onto a moved-from vector.

   </details>

5. Why does `v.push_back(v[0])` misbehave when the vector is full, and how would
   you fix it?

   <details><summary>Answer</summary>

   `push_back(const T&)` receives a reference into `data_`. `grow()` moves the
   elements to a new buffer and destroys the old ones, so the reference now
   dangles when `construct_at` copies from it. The fix is to allocate the new
   buffer, construct the new element there first (from the still-valid
   argument), then relocate the old elements and free the old buffer.

   </details>
