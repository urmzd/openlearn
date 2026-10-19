# UniquePtr

The single most important idea in C++ resource management is that a resource belongs to an object, and the object's destructor releases it. `std::unique_ptr` is that idea applied to heap memory: exactly one owner, no copying, and deletion only in the destructor, or in `reset`, which is the same act on demand. This section builds a work-alike, `UniquePtr<T>`, with custom deleters, an array specialisation, and a factory function.

The class is short, and nearly every line of it encodes a rule. Deleting the copy operations is the design. `release` and `reset` differ in one way that decides between a leak and a double free. The order of two statements in `reset` decides whether self-reset is safe. And the move operations must be `noexcept`, which connects this section to the growth logic in [Vector](../01-vector/README.md). The solution is the header [`unique_ptr.hpp`](unique_ptr.hpp); [`main.cpp`](main.cpp) is the test suite.

> **Learning objectives**
> - Explain RAII and why the destructor is the right place to release a resource.
> - Make a type move-only and verify the property at compile time with `static_assert`.
> - Implement `release`, `reset`, and move assignment so that self-reset and self-move are safe.
> - Parameterise deletion with a deleter type and specialise for arrays so `delete[]` matches `new[]`.

## The interface

| Member | Does |
|--------|------|
| `release()` | Give up ownership *without* deleting |
| `reset(p)` | Destroy what is owned, adopt `p` |
| `swap(other)` | Exchange pointers and deleters |
| Move ctor / move assign | Transfer ownership, empty the source |
| Copy ctor / copy assign | Deleted, and that is the point |
| `~UniquePtr()` | Destroys the owned object with the deleter |

Beyond these, the class takes a `Deleter` template parameter (default `DefaultDelete<T>`), and there is an array specialisation `UniquePtr<T[]>` that uses `delete[]` and offers `operator[]`, a factory `make_unique_ptr`, an explicit `operator bool`, and comparison with `nullptr`. Moves must be `noexcept`; the tests check that with `static_assert`.

To build and run the tests from this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o unique_ptr && ./unique_ptr
# ok  cpp/02-unique-ptr  61 checks passed
```

## Ownership as a compile-time rule

Deleting the copy operations is the design. A copyable owning pointer is a double free waiting for a second scope exit. Writing `= delete` moves the invariant out of the documentation and into the compiler, and the tests assert it with `static_assert` rather than at run time, because a violation must fail to *build*.

The move operations must be `noexcept`, and there is a `static_assert` for that too. Containers check the trait: a `std::vector<UniquePtr<T>>` that cannot move without risk of throwing will try to *copy* on reallocation instead, and copying is deleted. That is the same `move_if_noexcept` machinery as in [Vector](../01-vector/README.md), seen from the other side.

## Release, reset, and the order of operations

`release` and `reset` differ in exactly one thing, and it is the important one. `release` hands the raw pointer back and deletes nothing, transferring responsibility to the caller. `reset` destroys what is held. Confusing them gives you either a leak or a double free, and both compile perfectly.

Order matters in `reset`: overwrite the member, then delete the old value. Deleting first means that if `~T()` reaches back into this `UniquePtr`, it observes a pointer to memory that is already gone. Doing it in the safe order also makes self-reset fall out for free, which `test_self_reset` pins.

## Arrays, conversions, and factories

The array specialisation exists because `delete` and `delete[]` are not interchangeable. Calling the wrong one is undefined behaviour that usually appears to work. `test_array_specialisation` catches it by counting destructors: plain `delete` on an array of five runs one destructor instead of five.

`operator bool` is explicit for a reason. Without `explicit`, `if (p)` still works, and so do `int x = p;`, `p + 1`, and comparing two unrelated `UniquePtr`s for equality by accident. The explicit form permits exactly the contextual conversions you want and rejects the rest.

`make_unique_ptr` is not just convenience. In `f(UniquePtr<T>(new T), g())`, compilers were once permitted to evaluate `new T`, then `g()`, then the constructor. If `g()` threw, the `T` leaked. C++17 tightened the rules, but the factory function remains the habit worth having.

## Deriving the design

Six observations build the class.

Ownership is an invariant: every owned pointer is deleted exactly once. Once leaks and double frees are both bugs, "exactly once" follows.

The destructor is the one place that runs for sure. It runs on normal scope exit and during exception unwinding, so putting the delete there makes cleanup automatic. That is RAII.

Two owners would mean two deletes, so copying must not exist. Deleting the copy constructor and copy assignment makes the compiler enforce uniqueness.

Transfer must be possible, so moving must exist. A move hands the pointer over and leaves the source holding `nullptr`. `release()` is exactly "hand over and forget", so moves are written in terms of it.

Replacing the held pointer must delete the old one, last. `reset(p)` stores `p` first and deletes the old pointer afterwards, so the object is never observed half-destroyed and `reset(get())` is harmless.

What "delete" means is a policy. A `Deleter` type parameter, called as `del_(ptr)`, lets the same class own a `FILE*` (with `fclose`), a pool block, or an array (with `delete[]`).

## Building it step by step

The moves, both built on `release()`:

```cpp
UniquePtr(UniquePtr&& other) noexcept {
  // release() leaves the source holding nullptr: a transfer, not a share.
  ptr_ = other.release();
  del_ = std::move(other.del_);
}

UniquePtr& operator=(UniquePtr&& other) noexcept {
  if (this != &other) {
    // Destroy what we own (with OUR current deleter), adopt theirs.
    reset(other.release());
    del_ = std::move(other.del_);
  }
  return *this;
}
```

The order in move assignment matters for custom deleters: `reset` destroys the
old object with the deleter it was created with, and only then is the new
deleter moved in.

`release` and `reset`:

```cpp
pointer release() noexcept {
  pointer old = ptr_;
  ptr_ = nullptr;    // we no longer own it; the caller does
  return old;
}

void reset(pointer p = nullptr) noexcept {
  // Overwrite first, delete second: if ~T() looks back at this UniquePtr,
  // it sees the new value, not a dangling one.
  pointer old = ptr_;
  ptr_ = p;
  if (old && old != p) del_(old);   // old != p makes reset(get()) a no-op
}
```

The destructor and the observers:

```cpp
~UniquePtr() {
  // Runs on scope exit and during unwinding: exception safety for free.
  if (ptr_) del_(ptr_);
}

// explicit: `if (p)` works, `int x = p;` does not compile.
explicit operator bool() const noexcept { return ptr_ != nullptr; }

T& operator*() const { return *ptr_; }
pointer operator->() const noexcept { return ptr_; }
```

The deleter and the array specialisation that needs it:

```cpp
template <typename T>
struct DefaultDelete {
  void operator()(T* p) const { delete p; }
};

template <typename T>
struct DefaultDelete<T[]> {
  void operator()(T* p) const { delete[] p; }   // the only correct match for new[]
};

template <typename T, typename Deleter>
class UniquePtr<T[], Deleter> {
  // ... same ownership logic ...
  T& operator[](std::size_t i) const { return ptr_[i]; }   // no * or ->
  // ...
};
```

And the factory, which keeps `new` out of user code:

```cpp
template <typename T, typename... Args>
UniquePtr<T> make_unique_ptr(Args&&... args) {
  return UniquePtr<T>(new T(std::forward<Args>(args)...));
}
```

The full solution is in [`unique_ptr.hpp`](unique_ptr.hpp).

## Why this design

Every operation is O(1). With a stateless deleter the object could be one pointer; the reference stores `Deleter del_{}` directly for clarity, so it is larger than a raw pointer, while `std::unique_ptr` uses `[[no_unique_address]]` or the empty-base optimisation to make the deleter cost zero bytes.

Against raw pointers, the gain is coverage: every early return and exception path would need its own `delete`, and the destructor covers them all. Against `shared_ptr`, the gain is cost and clarity: shared ownership needs a reference count, usually atomic, and a control block. When there is one owner, that is pure overhead, and it hides who is responsible.

The deleter is a template parameter rather than a function pointer member. A type parameter lets the compiler inline the call and costs nothing when stateless; a function pointer would add 8 bytes and an indirect call to every object.

## What can go wrong

Self-move-assignment is the bug most implementations have. The obvious `reset(other.release())` is correct, but `reset(nullptr); ptr_ = other.ptr_;` is not: on `p = std::move(p)` it destroys the object and then adopts the dangling pointer. The standard only promises "valid but unspecified" for moved-from objects, but destroying the thing you are about to keep is never acceptable.

| Mistake | Consequence |
|---------|-------------|
| Leaving copy operations defined | Two owners and a double free at the second scope exit |
| Calling `release` where `reset` was meant | A leak |
| Calling `reset` where `release` was meant | A double free when the caller also deletes |
| Deleting before overwriting in `reset` | `~T()` can observe a dangling pointer; `reset(get())` frees the live object |
| `reset(nullptr); ptr_ = other.ptr_;` in move assignment | Self-move destroys the object and keeps a dangling pointer |
| Moving in the new deleter before `reset` | The old object is destroyed with the wrong deleter |
| Non-`noexcept` moves | `std::vector<UniquePtr<T>>` cannot reallocate |
| `delete` on memory from `new[]` | Undefined behaviour; one destructor instead of five |

## Where this leads

The natural next step is `SharedPtr` with a control block, and building it shows how much you gave up: an atomic reference count on every copy, a second allocation unless you use the `make_shared` trick, and cycles that never free. Then implement `WeakPtr` and see why the control block needs two counts rather than one. Within this chapter, the type-erasure section, [Function](../08-function/README.md), applies the same ownership rules to a stored callable.

## Key terms

- **RAII**: Resource Acquisition Is Initialization; tying a resource's release to an object's destructor.
- **Move-only type**: a type that can be moved but not copied, so ownership can transfer but never duplicate.
- **Deleter**: the callable a smart pointer uses to destroy what it owns.
- **`release` / `reset`**: give up ownership without deleting, versus delete the owned object and adopt another.
- **Contextual conversion**: an implicit use of `explicit operator bool` permitted only in conditions such as `if (p)` and `!p`.

## Summary

- `UniquePtr` owns one pointer, deletes it exactly once in its destructor, and cannot be copied.
- Moves transfer ownership through `release()` and are `noexcept`, so containers can relocate them.
- `reset` overwrites the member before deleting the old pointer, making `reset(get())` and self-move safe.
- A deleter type parameter makes destruction a policy; the array specialisation uses `delete[]` and offers `operator[]`.
- `operator bool` is explicit, and `make_unique_ptr` keeps `new` out of user code.

## Review questions

1. Why does `reset` assign `ptr_ = p` before deleting the old pointer?

   <details><summary>Answer</summary>

   If the old object's destructor reaches back into this `UniquePtr` (for
   example through a parent pointer), it must not see a pointer to the object
   being destroyed. Overwriting first means it sees the new value. It also makes
   `reset(get())` safe, because `old == p` skips the delete.

   </details>

2. Is `p = std::move(p)` safe with this implementation even without the
   `this != &other` check?

   <details><summary>Answer</summary>

   Yes. `other.release()` sets `ptr_` to null and returns the pointer, then
   `reset(ptr)` sees `old == nullptr`, stores the pointer back, and deletes
   nothing. The object survives. The check only saves the work.

   </details>

3. Why must the move operations be `noexcept`?

   <details><summary>Answer</summary>

   Containers like `std::vector` use `move_if_noexcept` when they reallocate. If
   the move might throw, they fall back to copying, and `UniquePtr` cannot be
   copied, so `std::vector<UniquePtr<T>>` would not compile, or would lose the
   strong exception guarantee.

   </details>

4. What goes wrong if `UniquePtr<int[]>` used `DefaultDelete<int>`?

   <details><summary>Answer</summary>

   Memory from `new[]` would be freed with `delete`, which is undefined
   behaviour. In practice, for types with destructors only the first element's
   destructor runs and the allocator may be handed the wrong address. The test
   counts destructors to catch it.

   </details>

5. Why is the conversion to `bool` marked `explicit`?

   <details><summary>Answer</summary>

   An implicit conversion would let a `UniquePtr` silently become a `bool`, and
   from there an integer, in arithmetic, assignments, and comparisons between
   unrelated pointers. `explicit` still allows contextual uses like `if (p)` and
   `!p`, and rejects the rest at compile time.

   </details>
