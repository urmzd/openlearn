# Coroutine generator

A generator is a function that produces a sequence one value at a time, pausing between values and resuming exactly where it left off. Python has had them for decades; C++20 added the machinery but not the type. The language provides no coroutine *type* at all. It provides a protocol: write a function containing `co_yield`, and the compiler looks at the declared return type, finds its nested `promise_type`, and calls specific members at fixed points. This section builds `Generator<T>`, which turns any function containing `co_yield` into a lazy, composable sequence that works in a range-`for`.

Two choices in the promise decide whether infinite generators are safe and whether the handle stays usable after the body ends. Ownership of the coroutine frame follows exactly the rules of [UniquePtr](../02-unique-ptr/README.md), and exceptions have to be smuggled from inside the coroutine to the consumer's stack. By the end you will be able to read the coroutine protocol as a short list of hooks and say what each choice does. The solution is the header [`generator.hpp`](../../../../../practice/build/systems/cpp/07-coroutine-generator/generator.hpp); [`main.cpp`](../../../../../practice/build/systems/cpp/07-coroutine-generator/main.cpp) is the test suite.

> **Learning objectives**
> - Describe the C++20 coroutine protocol: the frame, the promise, the handle, and when each promise member is called.
> - Choose `initial_suspend` and `final_suspend` so a generator is lazy and its handle stays valid after the body ends.
> - Own a coroutine frame correctly in a move-only type.
> - Propagate exceptions from a coroutine body to the consumer, and expose the sequence as an input iterator.

## The protocol

These are the members the compiler calls, and when:

| Member | Called |
|--------|--------|
| `get_return_object()` | Once, to build what the caller receives |
| `initial_suspend()` | Before the body's first statement |
| `yield_value(v)` | By each `co_yield` |
| `return_void()` | When the body finishes normally |
| `final_suspend()` | After the body finishes |
| `unhandled_exception()` | If the body throws |
| `~Generator()` | Must `destroy()` the frame |

The generator must be lazy (so infinite generators are safe), move-only, free its frame when dropped, rethrow body exceptions to the consumer, and work in a range-`for`.

To build and run the tests from this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o generator && ./generator
# ok  cpp/07-coroutine-generator  47 checks passed
```

Add `-g -fsanitize=address` to turn the abandonment test into a leak check.

## Suspending at the start and at the end

`initial_suspend` returning `suspend_always` is what makes infinite generators possible. With `suspend_never` the body runs to its first `co_yield` during construction. Suspending immediately means nothing executes until the consumer asks.

`final_suspend` returning `suspend_always` is what keeps the handle usable. Return `suspend_never` and the frame is destroyed the instant the body finishes, so every later `handle_.done()` is a use-after-free.

## Owning the frame

The frame is heap-allocated and uniquely owned. That is why `Generator` is move-only with a `destroy()` in the destructor: it is a `unique_ptr` wearing a different name. `test_abandonment_frees_the_frame` drops a thousand suspended generators.

## Exceptions and iteration

Stash the exception; do not let it escape. `unhandled_exception` is called *inside* the coroutine, where there is no consumer stack to unwind into. Capturing it with `std::current_exception()` and rethrowing on the consumer's next `next()` makes a throwing generator behave like a throwing loop.

Only an input iterator is honest here. A single-pass lazy sequence cannot provide a forward iterator: you cannot revisit an element, and copying the position does not copy the state.

Composition is where laziness pays. `take(squares_of(naturals()), 5)` evaluates exactly five elements and builds no intermediate container.

## Deriving the design

Six observations build the type.

A coroutine's locals live in a heap frame the compiler allocates. The promise object is inside that frame, and a `coroutine_handle` is a pointer to it. Whoever holds the handle must eventually call `destroy()`, exactly once: the `Generator` owns the handle as a `unique_ptr` owns its pointer.

Laziness means not running until asked. Suspend at the start (`initial_suspend` returns `suspend_always`); each `next()` resumes the body until the next `co_yield` suspends it again.

A yielded value must outlive the suspension. `yield_value` moves it into the promise (`current`), and the consumer reads it from there.

The end must be observable. Suspend at the end too (`final_suspend` returns `suspend_always`), so `done()` can be asked safely and the destructor remains the only place the frame is freed.

Exceptions belong to the consumer. Record them in the promise and rethrow from `next()`, on the consumer's stack.

Iteration is `next()` in disguise. `begin()` advances once; `++` advances again; an iterator becomes "end" (a null generator pointer) when `next()` returns false.

## Building it step by step

The promise, the compiler's interface:

```cpp
struct promise_type {
  T current;                    // the last yielded value
  std::exception_ptr error;     // set if the body threw

  Generator get_return_object() {
    // Recover the handle from the promise the compiler already placed in
    // the frame, and give it to the caller's Generator.
    return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
  }

  // Lazy start: nothing runs until the first next().
  std::suspend_always initial_suspend() noexcept { return {}; }

  // Suspend at the end so the frame (and done()) stays valid until
  // ~Generator destroys it.
  std::suspend_always final_suspend() noexcept { return {}; }

  // co_yield v is rewritten into co_await promise.yield_value(v).
  std::suspend_always yield_value(T value) {
    current = std::move(value);
    return {};
  }

  void return_void() noexcept {}

  // Called inside the coroutine; stash it for the consumer.
  void unhandled_exception() { error = std::current_exception(); }
};
```

Ownership of the frame:

```cpp
Generator(Generator&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}

Generator& operator=(Generator&& other) noexcept {
  if (this != &other) {
    if (handle_) handle_.destroy();          // free the frame we owned
    handle_ = std::exchange(other.handle_, {});
  }
  return *this;
}

~Generator() {
  // Works whether the coroutine is suspended mid-body or at final suspend.
  if (handle_) handle_.destroy();
}
```

Copying is deleted: two `Generator`s sharing one handle would destroy the frame
twice.

Driving the coroutine:

```cpp
bool next() {
  if (!handle_ || handle_.done()) return false;   // never resume a finished one

  handle_.resume();   // run the body until the next co_yield or the end

  // Surface a body exception on the consumer's side.
  if (handle_.promise().error) std::rethrow_exception(handle_.promise().error);

  return !handle_.done();   // false: the body returned, no new value
}

const T& value() const { return handle_.promise().current; }
```

Resuming a coroutine that is at its final suspend point is undefined
behaviour, which is why the `done()` check comes first. After an exception the
coroutine is also at its final suspend, so later `next()` calls return `false`
instead of rethrowing again.

The input iterator:

```cpp
class iterator {
 public:
  iterator() noexcept = default;                         // end
  explicit iterator(Generator* g) : gen_(g) {
    if (gen_ && !gen_->next()) gen_ = nullptr;          // advance to the first value
  }
  const T& operator*() const { return gen_->value(); }
  iterator& operator++() {
    if (gen_ && !gen_->next()) gen_ = nullptr;          // exhausted: become end
    return *this;
  }
  bool operator==(const iterator& o) const { return gen_ == o.gen_; }
  // ...
 private:
  Generator* gen_ = nullptr;
};
```

The full solution is in [`generator.hpp`](../../../../../practice/build/systems/cpp/07-coroutine-generator/generator.hpp).

## Why this design

Each generator costs one heap allocation (the frame; compilers may elide it when the lifetime is visible), and each element costs a resume/suspend pair, roughly the cost of an indirect call.

Returning a `std::vector` instead is eager: it computes everything up front, needs memory for all of it, and cannot express infinite sequences. A hand-written iterator class has the same laziness, but the loop state must be turned into member variables by hand; the coroutine keeps it as ordinary locals. C++23's `std::generator` adds recursive yielding and allocator support; this one is the minimal core of the same protocol.

The accepted trade is that `T current` requires `T` to be default-constructible, and values are copied or moved into the promise. Storing a pointer to the yielded value (it lives in the suspended frame) would avoid both.

## What can go wrong

Interleaving is real, not batching, and a values-only test cannot tell the difference. `test_interleaving_is_real` records a log from both sides and asserts the exact alternation; a generator that ran the body to completion and buffered the results would pass a test that only checked the values.

| Mistake | Consequence |
|---------|-------------|
| `initial_suspend` returns `suspend_never` | The first value is computed during construction and then skipped |
| `final_suspend` returns `suspend_never` | The frame is freed when the body ends; `done()` is a use-after-free |
| Resuming without checking `done()` | Undefined behaviour at the final suspend point |
| Letting the exception escape `unhandled_exception` | There is no consumer stack to unwind into |
| Copyable `Generator` | Two destructors destroy one frame |
| Claiming a forward iterator | Copies of the iterator share, not copy, the sequence state |

After an exception the coroutine is at its final suspend, so later `next()` calls return `false` rather than rethrowing again.

## Where this leads

Try `co_await` on a generator to flatten nested sequences, which needs an awaiter type and is how recursive generators avoid O(depth) resumption cost. Then compare with `std::generator` in C++23, which standardises this exercise and adds exactly that recursive-yield support. The same promise-and-handle protocol underlies asynchronous C++ code, where `co_await` suspends on I/O instead of on a yielded value.

## Key terms

- **Coroutine**: a function that can suspend and later resume, keeping its locals in a heap-allocated frame.
- **Promise type**: the nested `promise_type` whose members the compiler calls at fixed points in a coroutine's life.
- **Coroutine handle**: a non-owning pointer to a coroutine frame, used to resume, query, and destroy it.
- **Suspend point**: a place where the coroutine pauses; initial and final suspend are the two implicit ones.
- **Input iterator**: a single-pass iterator whose copies do not independently preserve position.

## Summary

- C++20 provides a protocol, not a type: `Generator<T>` supplies the `promise_type` the compiler calls.
- Suspending at the start makes the generator lazy; suspending at the end keeps the handle valid until the destructor frees the frame.
- `Generator` owns its frame like a `unique_ptr`: move-only, with `destroy()` in the destructor.
- Body exceptions are captured in the promise and rethrown from `next()` on the consumer's stack.
- Iteration is an input iterator over `next()`, and composition evaluates only the elements consumed.

## Review questions

1. What would `initial_suspend` returning `suspend_never` do to
   `Generator<int> naturals()` with a `while (true)` body?

   <details><summary>Answer</summary>

   It does not hang: the body runs to its first `co_yield`, which suspends. The damage is that the first value is computed eagerly
   during construction, and `next()` would then skip it by resuming past it.
   Bodies with side effects before the first yield would run at construction.
   With `suspend_always`, nothing runs until the consumer asks.

   </details>

2. Why must `final_suspend` return `suspend_always` here?

   <details><summary>Answer</summary>

   With `suspend_never` the frame is destroyed as soon as the body finishes.
   `next()` would then call `handle_.done()` and `promise()` on freed memory, and
   `~Generator` would destroy it a second time.

   </details>

3. Why does `next()` check `done()` before calling `resume()`?

   <details><summary>Answer</summary>

   Resuming a coroutine suspended at its final suspend point is undefined
   behaviour. Once the body has ended, `next()` must just report `false`.

   </details>

4. Why is `Generator` move-only?

   <details><summary>Answer</summary>

   It owns a coroutine frame through its handle. A copy would share the handle,
   two destructors would call `destroy()` on the same frame, and two consumers
   would resume the same state.

   </details>
