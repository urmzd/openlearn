# C++ 07: Coroutine generator

**Concepts:** C++20 coroutines, `co_yield`, lazy sequences
**Difficulty:** ⭐⭐⭐⭐

A `Generator<T>` that turns any function containing `co_yield` into a lazy,
composable sequence. The solution is the header
[`generator.hpp`](generator.hpp); [`main.cpp`](main.cpp) is the test suite.

## The problem

C++20 provides no coroutine *type*. It provides a protocol: write a function
containing `co_yield`, and the compiler looks at the declared return type, finds
its nested `promise_type`, and calls specific members at fixed points.

| Member | Called |
|--------|--------|
| `get_return_object()` | Once, to build what the caller receives |
| `initial_suspend()` | Before the body's first statement |
| `yield_value(v)` | By each `co_yield` |
| `return_void()` | When the body finishes normally |
| `final_suspend()` | After the body finishes |
| `unhandled_exception()` | If the body throws |
| `~Generator()` | Must `destroy()` the frame |

The generator must be lazy (infinite generators are safe), move-only, free its
frame when dropped, rethrow body exceptions to the consumer, and work in a
range-`for`.

## Build and run

From this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o generator && ./generator
# ok  cpp/07-coroutine-generator  47 checks passed
```

Add `-g -fsanitize=address` to turn the abandonment test into a leak check.

## What to notice

**`initial_suspend` returning `suspend_always` is what makes infinite
generators possible.** With `suspend_never` the body runs to its first
`co_yield` during construction. Suspending immediately means nothing executes
until the consumer asks.

**`final_suspend` returning `suspend_always` is what keeps the handle usable.**
Return `suspend_never` and the frame is destroyed the instant the body finishes,
so every later `handle_.done()` is a use-after-free.

**The frame is heap-allocated and uniquely owned.** That is why `Generator` is
move-only with a `destroy()` in the destructor: it is a `unique_ptr` wearing a
different name. `test_abandonment_frees_the_frame` drops a thousand suspended
generators.

**Stash the exception, do not let it escape.** `unhandled_exception` is called
*inside* the coroutine, where there is no consumer stack to unwind into.
Capturing it with `std::current_exception()` and rethrowing on the consumer's
next `next()` makes a throwing generator behave like a throwing loop.

**Interleaving is real, not batching.** `test_interleaving_is_real` records a
log from both sides and asserts the exact alternation. A generator that ran the
body to completion and buffered the results would pass a values-only test.

**Only an input iterator is honest here.** A single-pass lazy sequence cannot
provide a forward iterator: you cannot revisit an element, and copying the
position does not copy the state.

**Composition is where laziness pays.** `take(squares_of(naturals()), 5)`
evaluates exactly five elements and builds no intermediate container.

## The solution, step by step

### From first principles

1. **A coroutine's locals live in a heap frame the compiler allocates.** The
   promise object is inside that frame, and a `coroutine_handle` is a pointer to
   it. Whoever holds the handle must eventually call `destroy()`, exactly once:
   the `Generator` owns the handle like a `unique_ptr` owns its pointer.
2. **Laziness means not running until asked.** Suspend at the start
   (`initial_suspend` returns `suspend_always`); each `next()` resumes the body
   until the next `co_yield` suspends it again.
3. **A yielded value must outlive the suspension.** `yield_value` moves it into
   the promise (`current`), and the consumer reads it from there.
4. **The end must be observable.** Suspend at the end too (`final_suspend`
   returns `suspend_always`), so `done()` can be asked safely and the destructor
   remains the only place the frame is freed.
5. **Exceptions belong to the consumer.** Record them in the promise and rethrow
   from `next()`, on the consumer's stack.
6. **Iteration is `next()` in disguise.** `begin()` advances once; `++` advances
   again; an iterator becomes "end" (a null generator pointer) when `next()`
   returns false.

### The code, annotated

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

Full solution: [`generator.hpp`](generator.hpp).

### Why this solution

- **Cost.** One heap allocation per generator (the frame; compilers may elide
  it when the lifetime is visible), and a resume/suspend pair per element,
  roughly the cost of an indirect call.
- **Versus returning a `std::vector`.** Eager: computes everything up front,
  needs memory for all of it, and cannot express infinite sequences.
- **Versus a hand-written iterator class.** Same laziness, but the loop state
  must be turned into member variables by hand; the coroutine keeps it as
  ordinary locals.
- **Versus C++23 `std::generator`.** The standard one adds recursive yielding
  and allocator support; this one is the minimal core of the same protocol.
- **Accepted trade.** `T current` requires `T` to be default-constructible, and
  values are copied/moved into the promise. Storing a pointer to the yielded
  value (it lives in the suspended frame) would avoid both.

### Explain it back

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

## Extending it

Try `co_await` on a generator to flatten nested sequences, which needs an
awaiter type and is how recursive generators avoid O(depth) resumption cost.
Then compare with `std::generator` in C++23, which standardises this exercise
and adds exactly that recursive-yield support.
