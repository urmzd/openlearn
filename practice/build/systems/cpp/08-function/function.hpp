// function.hpp - a std::function clone: type erasure plus a small buffer.
//
// The reference implementation. main.cpp tests it.
//
// Two ideas, and they are independent:
//
//   TYPE ERASURE. Function<int(int)> can hold a lambda, a function pointer, or
//   a functor with members, all of which are different types with different
//   sizes. Storing them behind one interface means capturing their behaviour
//   in a table of operations rather than in the type.
//
//   SMALL BUFFER OPTIMISATION. Most callables are tiny (a lambda capturing one
//   pointer is 8 bytes). Heap-allocating those would make every callback cost
//   an allocation, so small ones are stored inline and only large ones spill.
#pragma once

#include <cstddef>
#include <functional>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

template <typename Signature>
class Function;

template <typename R, typename... Args>
class Function<R(Args...)> {
  // Big enough for a lambda capturing two or three pointers, which covers the
  // overwhelming majority in practice. Bigger buffers cost size in every
  // Function, including the empty ones.
  static constexpr std::size_t kBufferSize = 32;
  static constexpr std::size_t kBufferAlign = alignof(std::max_align_t);

  // The vtable, written by hand. A virtual base class would work and would
  // force every callable onto the heap, because a base subobject cannot live
  // in a buffer whose type the base does not know.
  struct VTable {
    R (*invoke)(void* self, Args&&... args);
    void (*move_to)(void* from, void* to);  // relocate between storages
    void (*destroy)(void* self);
    bool inline_storage;
  };

 public:
  Function() noexcept = default;
  Function(std::nullptr_t) noexcept {}

  // Construct from any compatible callable.
  //
  // The decay + enable_if guard is what stops this template from hijacking the
  // copy constructor: without it, Function(Function&) is a better match than
  // Function(const Function&) and you get infinite recursion.
  template <typename F,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<F>, Function> &&
                std::is_invocable_r_v<R, std::decay_t<F>&, Args...>>>
  Function(F&& f) {
    using Callable = std::decay_t<F>;
    // Inline only if it fits AND relocating it cannot throw. A callable whose
    // move can throw must live on the heap, because moving the Function would
    // otherwise have to move the callable, and a throw partway through leaves
    // no valid state to return to.
    constexpr bool fits = sizeof(Callable) <= kBufferSize &&
                          alignof(Callable) <= kBufferAlign &&
                          std::is_nothrow_move_constructible_v<Callable>;

    static const VTable vt = make_vtable<Callable, fits>();
    vtable_ = &vt;

    if constexpr (fits) {
      ::new (static_cast<void*>(&buffer_)) Callable(std::forward<F>(f));
    } else {
      heap_ = new Callable(std::forward<F>(f));
    }
  }

  Function(const Function&) = delete;  // see the README: copying needs a fourth op
  Function& operator=(const Function&) = delete;

  Function(Function&& other) noexcept {
    if (other.vtable_) {
      vtable_ = other.vtable_;
      if (vtable_->inline_storage) {
        // An inline callable cannot simply have its bytes copied: it may hold
        // a pointer to itself, and in any case its move constructor is the
        // only thing entitled to relocate it.
        vtable_->move_to(&other.buffer_, &buffer_);
        vtable_->destroy(&other.buffer_);
      } else {
        heap_ = other.heap_;
        other.heap_ = nullptr;
      }
      other.vtable_ = nullptr;
    }
  }

  Function& operator=(Function&& other) noexcept {
    if (this != &other) {
      reset();
      if (other.vtable_) {
        vtable_ = other.vtable_;
        if (vtable_->inline_storage) {
          vtable_->move_to(&other.buffer_, &buffer_);
          vtable_->destroy(&other.buffer_);
        } else {
          heap_ = other.heap_;
          other.heap_ = nullptr;
        }
        other.vtable_ = nullptr;
      }
    }
    return *this;
  }

  Function& operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  ~Function() { reset(); }

  // Call it. Throws if empty, matching std::bad_function_call.
  R operator()(Args... args) const {
    if (!vtable_) throw std::bad_function_call();
    void* self = vtable_->inline_storage
                     ? const_cast<void*>(static_cast<const void*>(&buffer_))
                     : heap_;
    return vtable_->invoke(self, std::forward<Args>(args)...);
  }

  explicit operator bool() const noexcept { return vtable_ != nullptr; }

  // Exposed so the tests can prove the small buffer is actually used.
  bool uses_inline_storage() const noexcept {
    return vtable_ != nullptr && vtable_->inline_storage;
  }

  void swap(Function& other) noexcept {
    Function tmp = std::move(*this);
    *this = std::move(other);
    other = std::move(tmp);
  }

 private:
  void reset() noexcept {
    if (!vtable_) return;
    // destroy() knows which storage it was built for, so it either runs the
    // destructor in place or does a full delete. Splitting that decision
    // across the vtable and here is how you get a mismatched operator delete
    // for over-aligned types.
    vtable_->destroy(vtable_->inline_storage ? static_cast<void*>(&buffer_) : heap_);
    vtable_ = nullptr;
  }

  // Build the operation table for one concrete callable type. Every entry is
  // a plain function pointer to a template instantiation, which is what makes
  // the erasure work without inheritance.
  template <typename Callable, bool Inline>
  static constexpr VTable make_vtable() {
    return VTable{
        [](void* self, Args&&... args) -> R {
          return (*static_cast<Callable*>(self))(std::forward<Args>(args)...);
        },
        [](void* from, void* to) {
          ::new (to) Callable(std::move(*static_cast<Callable*>(from)));
        },
        [](void* self) {
          // Destruction has to mirror construction exactly. An inline callable
          // was placement-new'd, so only its destructor runs; a heap one came
          // from a new-expression, so it needs the matching delete-expression,
          // which also picks the correctly aligned operator delete.
          if constexpr (Inline) {
            static_cast<Callable*>(self)->~Callable();
          } else {
            delete static_cast<Callable*>(self);
          }
        },
        Inline,
    };
  }

  const VTable* vtable_ = nullptr;
  union {
    alignas(kBufferAlign) unsigned char buffer_[kBufferSize];
    void* heap_;
  };
};
