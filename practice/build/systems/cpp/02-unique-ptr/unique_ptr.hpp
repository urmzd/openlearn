// unique_ptr.hpp - a std::unique_ptr work-alike.
//
// You write the marked regions. main.cpp tests them.
//
// The exercise is ownership expressed in the type system: exactly one
// UniquePtr owns a given pointer, the compiler refuses to let you copy it, and
// the destructor is the only place deletion happens. Get it right and the
// resource cannot leak even when an exception unwinds the stack.
#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

// The default deleter. A separate type rather than a hardcoded `delete` so
// that the deleter can be swapped for fclose, a pool release, or anything
// else, without UniquePtr knowing what it means to destroy the thing.
template <typename T>
struct DefaultDelete {
  void operator()(T* p) const { delete p; }
};

template <typename T>
struct DefaultDelete<T[]> {
  void operator()(T* p) const { delete[] p; }
};

template <typename T, typename Deleter = DefaultDelete<T>>
class UniquePtr {
 public:
  using pointer = T*;
  using element_type = T;
  using deleter_type = Deleter;

  constexpr UniquePtr() noexcept = default;
  constexpr UniquePtr(std::nullptr_t) noexcept {}

  explicit UniquePtr(pointer p) noexcept : ptr_(p) {}
  UniquePtr(pointer p, Deleter d) noexcept : ptr_(p), del_(std::move(d)) {}

  // Copying is what makes double-free possible, so it is deleted rather than
  // left to the reader's discipline. This is the difference between a type
  // that documents an invariant and one that enforces it.
  UniquePtr(const UniquePtr&) = delete;
  UniquePtr& operator=(const UniquePtr&) = delete;

  UniquePtr(UniquePtr&& other) noexcept {
    // release() leaves the source holding nullptr, which is what makes this a
    // transfer rather than a share. Forgetting it is a double free.
    ptr_ = other.release();
    del_ = std::move(other.del_);
  }

  UniquePtr& operator=(UniquePtr&& other) noexcept {
    if (this != &other) {
      // reset() first: whatever we currently own has to be destroyed, or
      // assigning over a live pointer leaks it.
      reset(other.release());
      del_ = std::move(other.del_);
    }
    return *this;
  }

  UniquePtr& operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  ~UniquePtr() {
    // The only place deletion happens. Because destructors run during stack
    // unwinding, this is also what makes the type exception-safe for free.
    if (ptr_) del_(ptr_);
  }

  // ---- observers --------------------------------------------------------

  pointer get() const noexcept { return ptr_; }
  Deleter& get_deleter() noexcept { return del_; }
  const Deleter& get_deleter() const noexcept { return del_; }

  // explicit, so that `if (p)` works but `int x = p;` and accidental
  // comparisons between unrelated pointers do not compile.
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  T& operator*() const { return *ptr_; }
  pointer operator->() const noexcept { return ptr_; }

  // ---- modifiers --------------------------------------------------------

  // Give up ownership WITHOUT deleting, returning the raw pointer. The caller
  // becomes responsible for it.
  pointer release() noexcept {
    pointer old = ptr_;
    ptr_ = nullptr;
    return old;
  }

  // Destroy what is owned and adopt `p`.
  void reset(pointer p = nullptr) noexcept {
    // Swap first, delete second. Deleting before overwriting ptr_ means that
    // if ~T() reaches back into this UniquePtr, it sees a dangling pointer.
    // Self-reset (p == ptr_) is also handled correctly this way.
    pointer old = ptr_;
    ptr_ = p;
    if (old && old != p) del_(old);
  }

  void swap(UniquePtr& other) noexcept {
    std::swap(ptr_, other.ptr_);
    std::swap(del_, other.del_);
  }

 private:
  pointer ptr_ = nullptr;
  // No [[no_unique_address]] here for clarity, though a real implementation
  // uses it (or inherits from the deleter) so a stateless deleter costs zero
  // bytes and UniquePtr stays the same size as a raw pointer.
  Deleter del_{};
};

// ---- array specialisation -------------------------------------------------
//
// T[] needs delete[] rather than delete, and offers operator[] instead of
// operator* and operator->. Mixing them up is undefined behaviour that
// usually appears to work, which is exactly why the two are separate types.

template <typename T, typename Deleter>
class UniquePtr<T[], Deleter> {
 public:
  using pointer = T*;

  constexpr UniquePtr() noexcept = default;
  constexpr UniquePtr(std::nullptr_t) noexcept {}
  explicit UniquePtr(pointer p) noexcept : ptr_(p) {}

  UniquePtr(const UniquePtr&) = delete;
  UniquePtr& operator=(const UniquePtr&) = delete;

  UniquePtr(UniquePtr&& other) noexcept : ptr_(other.release()) {}

  UniquePtr& operator=(UniquePtr&& other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }

  ~UniquePtr() {
    if (ptr_) del_(ptr_);
  }

  T& operator[](std::size_t i) const { return ptr_[i]; }
  pointer get() const noexcept { return ptr_; }
  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  pointer release() noexcept {
    pointer old = ptr_;
    ptr_ = nullptr;
    return old;
  }

  void reset(pointer p = nullptr) noexcept {
    pointer old = ptr_;
    ptr_ = p;
    if (old && old != p) del_(old);
  }

 private:
  pointer ptr_ = nullptr;
  Deleter del_{};
};

// Construct in one allocation-free-of-leaks step. The reason this exists at
// all rather than `UniquePtr<T>(new T(...))`: in an expression like
// f(UniquePtr<T>(new T), g()), the compiler was once permitted to run `new T`,
// then g(), then the UniquePtr constructor. If g() threw, the T leaked.
template <typename T, typename... Args>
UniquePtr<T> make_unique_ptr(Args&&... args) {
  return UniquePtr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename D>
bool operator==(const UniquePtr<T, D>& p, std::nullptr_t) noexcept {
  return p.get() == nullptr;
}
template <typename T, typename D>
bool operator!=(const UniquePtr<T, D>& p, std::nullptr_t) noexcept {
  return p.get() != nullptr;
}
