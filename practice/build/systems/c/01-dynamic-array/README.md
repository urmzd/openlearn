# Dynamic array

A growable array is the first data structure most programs need and the first one that C refuses to give you. The language has fixed-size arrays and `malloc`; everything between those two you build yourself. This section builds a growable array of `int`, and along the way it teaches the lesson that makes C memory management hard: `realloc` is dangerous in a way `malloc` is not, because it can move your buffer, and moving the buffer invalidates every pointer anyone still holds into it.

The design questions are small but real. How much should the array grow when it fills? What happens when allocation fails halfway through a push? How do you stop a size calculation from silently wrapping around? By the end you will have answers to all three, backed by an amortised cost argument you can reproduce on paper. A longer treatment of the same structure, generalised to any element type, is [A growable array](../../../../../content/courses/algorithms/16-systems-data-structures/01-growable-array.md) in *Algorithms*; the C background it assumes is in [C: memory, pointers, structs, headers, and the C11 toolchain](../../../../../content/courses/programming-languages/02-language-and-tool-primers/03-c.md) from *Programming Languages*.

> **Learning objectives**
> - Explain why geometric growth gives amortised O(1) push and constant-step growth does not, and compute the total copy cost of each.
> - Write a `realloc` call that cannot leak or lose data when allocation fails.
> - Guard `size_t` arithmetic against overflow before it reaches the allocator.
> - Explain why pointers into a growable array dangle after a push, and why the bug often hides on small inputs.

## The interface

[`vec.h`](vec.h) declares the interface, [`vec.c`](vec.c) is the solution, and [`main.c`](main.c) is the test suite. The six functions are:

| Function | Does |
|----------|------|
| `vec_init(v)` | Zero-capacity vector, no allocation yet |
| `vec_push(v, x)` | Append, growing if needed. Returns 0 on success, -1 on allocation failure |
| `vec_pop(v, out)` | Remove and return the last element. -1 if empty |
| `vec_get(v, i)` | Element at `i`. Undefined behaviour past `len`, deliberately |
| `vec_reserve(v, n)` | Ensure capacity for at least `n` without changing `len` |
| `vec_free(v)` | Release the buffer and return to the zero state |

Three constraints shape the solution. It may use only the standard library and must not leak. `vec_free` must leave the struct safe to `vec_free` again. And a failed push must leave the vector unchanged and still valid, so a caller who sees -1 can carry on with the elements it already has.

To build and run the tests from this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c vec.c -o vec && ./vec
# ok  c/01-dynamic-array  20027 checks passed
```

The program exits 0 when every check holds and stops with `FAIL file:line` on the first one that does not. To check memory safety as well, build with the sanitizers:

```bash
cc -std=c11 -Wall -Wextra -g -fsanitize=address,undefined main.c vec.c -o vec && ./vec
```

## Three things that make this harder than it looks

The first is that the growth factor is a real decision, not a detail. Doubling gives amortised O(1) push. Growing by a constant gives O(n) per push and quadratic total cost, which you can feel on a million elements. Growing by 1.5 reuses freed blocks better, because the sum of previous allocations eventually exceeds the next request; with doubling it never does. glibc, MSVC, and Rust all pick differently here, and all three have a defensible reason.

The second is that `realloc` can move the buffer. Any pointer, or pointer derived from an index, handed out before a `push` may now dangle. The bug is invisible until the one push that happens to cross a page boundary, which is why the test suite pushes 10,000 elements and re-reads every one.

The third is a trap in the standard itself: `realloc(p, 0)` is not `free(p)`. In C17 its behaviour is implementation-defined, and in C23 it is undefined. Do not rely on it.

## Deriving the design

With those hazards in view, the design follows from six observations, each forced by the one before.

A contiguous array is the only representation that gives O(1) `get`. `vec_get` is `data[i]`, one address computation. So the elements must live in one block, and that block must be replaced when it fills.

Replacing the block costs O(len), so it must happen rarely. If capacity grows by a constant $k$, pushing $n$ elements copies roughly $k + 2k + \dots + n$ elements, which is $O(n^2/k)$. If it grows by a factor (here 2), the copies sum to at most $n + n/2 + n/4 + \dots < 2n$, so each push costs O(1) amortised. That forces geometric growth.

Length and capacity are therefore separate numbers. `len` is how many elements are live, `cap` how many fit. `len <= cap` is the invariant every function keeps, and `cap` only grows until `vec_free`.

Allocation can fail, and the vector must survive it. `realloc` returns `NULL` on failure and leaves the old block untouched. So the result goes into a temporary and replaces `v->data` only on success. Then a failed push leaves `len`, `cap`, and the data exactly as they were.

Size arithmetic can overflow before `realloc` sees it. `cap * 2` and `cap * sizeof(int)` are `size_t` multiplications that wrap silently. A wrapped size makes `realloc` hand back a small block that the code then writes past. Both products are checked against `SIZE_MAX` before they are computed.

Finally, an empty vector should cost nothing. `vec_init` sets everything to zero and allocates nothing, and `vec_free` returns to that same state, which makes a second `vec_free` a harmless `free(NULL)`.

## Building it step by step

All growth goes through `vec_reserve`, so the growth policy lives in one place:

```c
int vec_reserve(Vec *v, size_t n) {
  if (n <= v->cap) return 0;          /* reserve never shrinks */

  /* Double, but never less than the caller asked for, and start at 4 rather
   * than 1 so the first few pushes do not each cause a realloc. */
  size_t cap = v->cap ? v->cap : 4;
  while (cap < n) {
    /* Overflow check before doubling. Unreachable on a 64-bit host in
     * practice, reachable on a 32-bit build. */
    if (cap > SIZE_MAX / 2) { cap = n; break; }
    cap *= 2;
  }
  /* cap * sizeof(int) must not wrap either. */
  if (cap > SIZE_MAX / sizeof(int)) return -1;

  int *grown = realloc(v->data, cap * sizeof(int));
  /* Assign only after realloc succeeds. v->data = realloc(v->data, ...)
   * would leak the old buffer on failure and lose every element. */
  if (!grown) return -1;
  v->data = grown;
  v->cap = cap;
  return 0;
}
```

`realloc(NULL, size)` behaves like `malloc(size)`, which is why the first call needs no special case: a fresh vector has `data == NULL`.

Push and pop are then one line each around the invariant `len <= cap`:

```c
int vec_push(Vec *v, int x) {
  /* Grow only when full. Asking for len + 1 lets reserve pick the doubled
   * size; the && short-circuits so a vector with room never calls it. */
  if (v->len == v->cap && vec_reserve(v, v->len + 1) != 0) return -1;
  v->data[v->len++] = x;   /* write first, then count it */
  return 0;
}

int vec_pop(Vec *v, int *out) {
  if (v->len == 0) return -1;
  *out = v->data[--v->len];  /* capacity is kept for the next push */
  return 0;
}

int vec_get(const Vec *v, size_t i) {
  return v->data[i];          /* no bounds check, by contract */
}
```

`vec_free` restores the state `vec_init` produces, so it is idempotent:

```c
void vec_free(Vec *v) {
  free(v->data);       /* free(NULL) is a no-op, so a second call is safe */
  v->data = NULL;
  v->len = 0;
  v->cap = 0;
}
```

The full solution is in [`vec.c`](vec.c).

## Why this design

The costs come out as the derivation promised. `vec_get` and `vec_pop` are O(1). `vec_push` is O(1) amortised and O(n) on the push that grows. After $n$ pushes, `cap < 2n` (or 4 for tiny $n$), so wasted space is under half the buffer. The growth test asserts `cap < 4n` to reject arithmetic growth.

The choice of factor 2 over 1.5 is a tradeoff, not a correctness question. A factor of 1.5 lets the allocator reuse earlier freed blocks for later requests; 2 does fewer reallocations and keeps the arithmetic trivial. Both are amortised O(1), and the reference picks 2 for simplicity.

Starting at 4 is a small optimisation with no downside. Starting at 1 does three reallocations to reach 4 elements, for no benefit, and starting at 0 is impossible with pure doubling.

`vec_get` has no bounds check because a checked accessor would need an error channel, and the return value is already the element. The interface puts the check on the caller, like `std::vector::operator[]`.

The real alternative is a linked list of chunks. Growth never moves elements, so pointers stay valid, but `get` becomes O(n / chunk) and loses cache locality. That trade is worth it only when pointer stability matters more than indexing.

## What can go wrong

Most of the bugs this design prevents are one-character mistakes. The table collects them.

| Mistake | Consequence |
|---------|-------------|
| `v->data = realloc(v->data, ...)` | On failure the only pointer to the old block is overwritten: a leak, and every element lost |
| Growing by a constant | Quadratic total copying; a million pushes becomes visibly slow |
| Computing `cap * sizeof(int)` without a check | On overflow, `realloc` returns a small block and the push writes past it |
| `vec_free` that only calls `free` | A second `vec_free` is a double free and heap corruption |
| Holding `&v.data[i]` across a push | A dangling pointer after any growth that moves the buffer |
| Using `realloc(p, 0)` to free | Implementation-defined in C17, undefined in C23 |

> **Warning:** The dangling-pointer bug passes small tests. `realloc` often extends a small block in place and returns the same address, so the stale pointer keeps working until the one growth that actually moves. Test with thousands of elements and AddressSanitizer, and store indices rather than pointers when you need to refer to an element across pushes.

## Where this leads

The reference stops at `int`. Making it generic is the real lesson about C: the language has no generics, so you either write the same code per type, cast through `void *` and an element size (losing type safety), or generate per-type code with a macro such as `VEC_DEFINE(name, T)` (losing debuggability). Each choice gives something up; pick one deliberately and know which. [A growable array](../../../../../content/courses/algorithms/16-systems-data-structures/01-growable-array.md) in *Algorithms* takes the type-erased route. Within this chapter, the same growth policy reappears in the [graph](../06-graph/README.md)'s per-vertex edge arrays and in the [binary heap](../04-binary-heap/README.md)'s backing store.

## Key terms

- **Capacity**: the number of elements the current buffer can hold, as distinct from the number it does hold (the length).
- **Geometric growth**: growing capacity by a constant factor, which makes the total copying cost linear in the number of pushes.
- **Amortised cost**: the average cost per operation over a worst-case sequence of operations, here O(1) per push even though one push in many is O(n).
- **Dangling pointer**: a pointer to memory that has been freed or moved, which is undefined behaviour to dereference.
- **Idempotent free**: a release function that is safe to call twice because it returns the object to its initial empty state.

## Summary

- Elements live in one contiguous block so that `get` is a single address computation; the block is replaced when it fills.
- Doubling makes the total copy cost less than $2n$, so push is amortised O(1); constant-step growth is quadratic.
- `realloc`'s result goes into a temporary so that failure leaves the vector unchanged.
- Both `cap * 2` and `cap * sizeof(int)` are checked against `SIZE_MAX` by dividing the limit, never by multiplying the value.
- Any push may move the buffer, so pointers into it are invalid afterwards; `vec_free` resets to the zero state so a second call is harmless.

## Review questions

1. Why does `vec_reserve` store `realloc`'s result in `grown` before assigning it to `v->data`?

   <details><summary>Answer</summary>

   On failure `realloc` returns `NULL` and leaves the old block allocated.
   Assigning directly would overwrite the only pointer to that block, leaking it
   and losing every element. With the temporary, a failed push leaves the vector
   exactly as it was, which is what the header promises.

   </details>

2. Why is doubling O(1) amortised, while growing by 16 each time is not?

   <details><summary>Answer</summary>

   With doubling, the copies made by all reallocations up to `n` elements sum to
   less than `2n`, so the average per push is a constant. With `+16`, there are
   `n/16` reallocations copying 16, 32, 48, ... elements, which sums to about
   `n²/32`: linear per push.

   </details>

3. A caller stores `int *p = &v.data[0];`, then pushes 100 more elements and
   reads `*p`. What can happen, and why does no test failure appear for small
   vectors?

   <details><summary>Answer</summary>

   Any growth may move the buffer, so `p` can point into freed memory: reading it
   is undefined behaviour. With few elements `realloc` often extends the block in
   place and returns the same address, so the bug stays hidden until a growth
   that actually moves. Storing indices instead of pointers avoids it.

   </details>

4. Why can `vec_free` be called twice, and what would break if it only called
   `free(v->data)`?

   <details><summary>Answer</summary>

   It sets `data` back to `NULL`, and `free(NULL)` does nothing. If it left the
   old pointer in place, the second call would free the same block twice, which
   is undefined behaviour and a classic heap-corruption bug.

   </details>

5. Why check `cap > SIZE_MAX / sizeof(int)` instead of
   `cap * sizeof(int) > SIZE_MAX`?

   <details><summary>Answer</summary>

   The second expression can never be true: the multiplication wraps modulo
   `SIZE_MAX + 1` before the comparison happens. Dividing the limit instead of
   multiplying the value keeps every intermediate in range.

   </details>
