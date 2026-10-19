# C++ 08: `std::function` clone

**Concepts:** type erasure, small buffer optimisation, manual vtables
**Difficulty:** ⭐⭐⭐⭐

A `Function<R(Args...)>` that stores any compatible callable behind one type,
keeping small ones inline and spilling large ones to the heap. The solution is
the header [`function.hpp`](function.hpp); [`main.cpp`](main.cpp) is the test
suite.

## The problem

| Member | Does |
|--------|------|
| `Function(F&& f)` | Erase any compatible callable |
| `operator()(args...)` | Invoke, or throw `bad_function_call` if empty |
| `operator bool()` | Whether it holds anything |
| `uses_inline_storage()` | Exposed so the tests can prove the SBO works |
| Move ctor / assign, `swap` | Transfer, handling both storage strategies |

Copy operations are deliberately deleted, so move-only callables (a lambda
capturing a `unique_ptr`) can be stored. Every stored callable must be
destroyed exactly once.

## Build and run

From this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o function && ./function
# ok  cpp/08-function  49 checks passed
```

## What to notice

**Two independent ideas share one class.** Type erasure is about storing
unrelated types behind one interface. The small buffer is purely a performance
trick. You could have either without the other.

**The vtable is written by hand, and it has to be.** Inheritance would work and
would force every callable onto the heap: a base subobject cannot live in a
buffer whose type the base does not know. A struct of function pointers, built
per callable type by `make_vtable<Callable, Inline>()`, gives the same dispatch
with the storage decision left free.

**The `enable_if` on the converting constructor is not optional.** Without it,
`Function(F&&)` is a *better* match for a non-const `Function&` than any
special member, so "copying" a Function would wrap it inside itself. The guard
excludes `std::decay_t<F> == Function`.

**A callable goes inline only if it fits *and* its move cannot throw.** Moving
the `Function` has to relocate an inline callable, and if that move can throw
there is no valid state to leave behind. The heap path has no such problem,
because moving is then just copying a pointer, which is also why the move
constructor can be `noexcept` for every callable.

**Relocating an inline callable means calling its move constructor, not
copying bytes.** A callable may hold a pointer into itself, and even when it
does not, `memcpy` on a non-trivially-copyable type is undefined. That is what
`move_to` in the vtable is for.

**Destruction must mirror construction exactly.** An inline callable was
placement-new'd, so only its destructor runs. A heap one came from a
new-expression and needs the matching delete-expression, which also selects the
correctly aligned `operator delete` for over-aligned types.

**Copying is deleted here, and `std::function` pays a real price for
supporting it.** A copyable erasure needs a fourth vtable entry and forces
every stored callable to be copyable, which is why
`std::function<void()> f = [p = std::make_unique<int>(1)]{};` does not compile.
C++23 added `std::move_only_function` for precisely this reason.
`test_move_only_capture` shows what that buys.

## The solution, step by step

### From first principles

1. **One type must hold many types, so behaviour moves into data.** Everything
   the `Function` will ever do to its callable is: call it, relocate it, destroy
   it. Capture those three operations as function pointers, generated per
   callable type, and store a pointer to that table.
2. **The table is shared and immutable**, so one `static const VTable` per
   callable type suffices; each `Function` holds just a pointer to it.
3. **Storage is either a buffer inside the object or a heap pointer.** A union
   of a 32-byte, max-aligned buffer and a `void*` holds one or the other, and the
   vtable records which.
4. **Inline is allowed only when relocation is safe:** the callable fits in
   size and alignment and its move constructor is `noexcept`.
5. **Moving a `Function` depends on storage.** Heap: steal the pointer. Inline:
   move-construct into the destination buffer, destroy the source's callable.
   Either way the source is left empty (`vtable_ == nullptr`).
6. **Empty is `vtable_ == nullptr`.** Calling an empty `Function` throws
   `std::bad_function_call`, like the standard one.

### The code, annotated

The table and storage:

```cpp
struct VTable {
  R (*invoke)(void* self, Args&&... args);
  void (*move_to)(void* from, void* to);  // relocate between storages
  void (*destroy)(void* self);
  bool inline_storage;
};

const VTable* vtable_ = nullptr;          // nullptr means empty
union {
  alignas(kBufferAlign) unsigned char buffer_[kBufferSize];  // 32 bytes
  void* heap_;
};
```

The converting constructor makes the storage decision once, at compile time:

```cpp
template <typename F,
          typename = std::enable_if_t<
              !std::is_same_v<std::decay_t<F>, Function> &&            // not ourselves
              std::is_invocable_r_v<R, std::decay_t<F>&, Args...>>>     // callable as R(Args...)
Function(F&& f) {
  using Callable = std::decay_t<F>;
  constexpr bool fits = sizeof(Callable) <= kBufferSize &&
                        alignof(Callable) <= kBufferAlign &&
                        std::is_nothrow_move_constructible_v<Callable>;

  static const VTable vt = make_vtable<Callable, fits>();  // one per type
  vtable_ = &vt;

  if constexpr (fits) {
    ::new (static_cast<void*>(&buffer_)) Callable(std::forward<F>(f));  // placement new
  } else {
    heap_ = new Callable(std::forward<F>(f));
  }
}
```

The generated operations. Each lambda without captures converts to a plain
function pointer, and knows the concrete `Callable` type:

```cpp
template <typename Callable, bool Inline>
static constexpr VTable make_vtable() {
  return VTable{
      [](void* self, Args&&... args) -> R {
        return (*static_cast<Callable*>(self))(std::forward<Args>(args)...);
      },
      [](void* from, void* to) {
        // Relocation by move construction, never memcpy.
        ::new (to) Callable(std::move(*static_cast<Callable*>(from)));
      },
      [](void* self) {
        // Mirror construction: destructor only for placement new,
        // full delete-expression for new.
        if constexpr (Inline) {
          static_cast<Callable*>(self)->~Callable();
        } else {
          delete static_cast<Callable*>(self);
        }
      },
      Inline,
  };
}
```

Moving and calling:

```cpp
Function(Function&& other) noexcept {
  if (other.vtable_) {
    vtable_ = other.vtable_;
    if (vtable_->inline_storage) {
      vtable_->move_to(&other.buffer_, &buffer_);   // construct ours
      vtable_->destroy(&other.buffer_);             // end theirs
    } else {
      heap_ = other.heap_;                          // steal the pointer
      other.heap_ = nullptr;
    }
    other.vtable_ = nullptr;                        // source is now empty
  }
}

R operator()(Args... args) const {
  if (!vtable_) throw std::bad_function_call();
  void* self = vtable_->inline_storage
                   ? const_cast<void*>(static_cast<const void*>(&buffer_))
                   : heap_;
  return vtable_->invoke(self, std::forward<Args>(args)...);
}
```

Move assignment is `reset()` followed by the same logic. `reset()` calls
`vtable_->destroy` on whichever storage is active and nulls `vtable_`; the
destructor is just `reset()`. `swap` is three moves through a temporary.

Full solution: [`function.hpp`](function.hpp).

### Why this solution

- **Cost.** A call is one indirect call through the vtable. Construction of a
  small callable allocates nothing; a large one allocates once. Each `Function`
  is 48 bytes on x86-64 (an 8-byte vtable pointer, padding to the buffer's
  16-byte alignment, and the 32-byte buffer).
- **Versus a virtual base class.** Same dispatch cost, but every callable must
  be heap-allocated as a derived object, so the small buffer becomes impossible.
- **Versus templates all the way** (`template <class F> void run(F f)`): zero
  overhead and inlinable, but every caller and container must know the callable
  type. Erasure is for when one type must hold many.
- **Accepted trade.** `operator()` is `const` but calls the callable through a
  non-const pointer (the `const_cast`), so a mutable lambda can change state
  through a `const Function&`. `std::function` has the same wart.

### Explain it back

1. Why does the converting constructor exclude `std::decay_t<F> == Function`?

   <details><summary>Answer</summary>

   For a non-const `Function&` argument, the forwarding template binds exactly
   and beats the deleted copy constructor (which needs a const conversion). It
   would then try to store a `Function` inside a `Function`. The guard removes the
   template from overload resolution for that case.

   </details>

2. Why is a callable with a throwing move constructor forced onto the heap?

   <details><summary>Answer</summary>

   Moving an inline callable means move-constructing it into the destination.
   If that threw partway, the `Function` move could not complete or roll back,
   and the move constructor is promised `noexcept`. On the heap, a move is a
   pointer copy that cannot throw.

   </details>

3. Why can't the move constructor `memcpy` the inline buffer?

   <details><summary>Answer</summary>

   The callable may contain pointers into itself or members with non-trivial
   moves (a `std::string` with a small buffer, for instance). Copying bytes
   would leave those pointing into the old object, and is undefined for
   non-trivially-copyable types. Only its move constructor may relocate it.

   </details>

4. Why does `destroy` call `delete` for heap callables instead of
   `~Callable()` plus `operator delete`?

   <details><summary>Answer</summary>

   The heap object was created with a new-expression, so the matching
   delete-expression both runs the destructor and selects the correct
   deallocation function, including the aligned one for over-aligned types.

   </details>

## Extending it

Add copy support and watch the constraint propagate: the converting constructor
now needs `is_copy_constructible`, and every test involving a move-only capture
stops compiling. Then measure the buffer size against real code, because 32
bytes is a guess and the standard library implementations disagree with each
other about the right answer.
