// generator.hpp - a lazy sequence built on C++20 coroutines.
//
// The reference implementation. main.cpp tests it.
//
// C++20 coroutines are unusual: the language provides no coroutine TYPE at
// all. It provides a protocol. Write a function containing co_yield, and the
// compiler looks at the declared return type, finds its nested promise_type,
// and calls the members below at fixed points. Supplying those members is all
// this file does.
//
// The transformation the compiler performs, roughly:
//
//     Generator<int> counter() { co_yield 1; co_yield 2; }
//
//   becomes a state machine whose locals live in a heap-allocated frame, with
//   the promise embedded in it. Each co_yield stores a value and suspends,
//   returning control to whoever resumed it.
#pragma once

#include <coroutine>
#include <exception>
#include <utility>

template <typename T>
class Generator {
 public:
  // The compiler looks for exactly this name. Everything the coroutine
  // machinery needs is found through it.
  struct promise_type {
    T current;
    std::exception_ptr error;

    // Called once, to build the object the caller receives. from_promise
    // recovers the handle from the promise the compiler already allocated.
    Generator get_return_object() {
      return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    // Run eagerly up to the first co_yield, or suspend immediately?
    //
    // suspend_always means the body does not start until the caller asks for
    // the first element, which is what makes an infinite generator safe to
    // construct. suspend_never would run to the first yield during
    // construction, which for `while(true)` is a hang.
    std::suspend_always initial_suspend() noexcept {
      return {};
    }

    // Suspend at the end rather than destroying the frame, so the caller can
    // still ask "are you done?" after the body returns. Returning
    // suspend_never here frees the frame the moment the coroutine finishes,
    // and every subsequent handle access is a use-after-free.
    std::suspend_always final_suspend() noexcept {
      return {};
    }

    // co_yield v is rewritten into co_await promise.yield_value(v).
    std::suspend_always yield_value(T value) {
      current = std::move(value);
      return {};
    }

    // A generator produces values through co_yield and never through co_return
    // with an operand, so returning void is the whole implementation.
    void return_void() noexcept {}

    // Called if the body throws. Stashing it lets the exception surface on the
    // consumer's thread at the point they asked for the next element, rather
    // than calling std::terminate here.
    void unhandled_exception() {
      error = std::current_exception();
    }
  };

  using handle_type = std::coroutine_handle<promise_type>;

  Generator() noexcept = default;
  explicit Generator(handle_type h) noexcept : handle_(h) {}

  // A coroutine frame is a unique resource, exactly like a unique_ptr's
  // pointee. Copying would give two owners of one frame.
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  Generator(Generator&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}

  Generator& operator=(Generator&& other) noexcept {
    if (this != &other) {
      if (handle_) handle_.destroy();
      handle_ = std::exchange(other.handle_, {});
    }
    return *this;
  }

  ~Generator() {
    // The frame was heap-allocated by the compiler; destroying the handle is
    // what frees it. Because final_suspend is suspend_always, a finished
    // coroutine still owns its frame and this is the only thing that releases
    // it.
    if (handle_) handle_.destroy();
  }

  // Advance to the next value. False once the body has finished.
  bool next() {
    if (!handle_ || handle_.done()) return false;

    handle_.resume();

    // Re-throw on the consumer's side, so a throwing generator body behaves
    // like a throwing loop.
    if (handle_.promise().error) std::rethrow_exception(handle_.promise().error);

    return !handle_.done();
  }

  const T& value() const { return handle_.promise().current; }

  bool done() const { return !handle_ || handle_.done(); }

  // ---- range-for support ------------------------------------------------
  //
  // An input iterator, which is all a single-pass lazy sequence can honestly
  // provide: there is no way to revisit an element or to copy the position.

  class iterator {
   public:
    iterator() noexcept = default;
    explicit iterator(Generator* g) : gen_(g) {
      if (gen_ && !gen_->next()) gen_ = nullptr;
    }

    const T& operator*() const { return gen_->value(); }

    iterator& operator++() {
      if (gen_ && !gen_->next()) gen_ = nullptr;
      return *this;
    }

    bool operator==(const iterator& o) const { return gen_ == o.gen_; }
    bool operator!=(const iterator& o) const { return gen_ != o.gen_; }

   private:
    Generator* gen_ = nullptr;
  };

  iterator begin() { return iterator(this); }
  iterator end() { return iterator(); }

 private:
  handle_type handle_{};
};
