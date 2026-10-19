# C 01: Dynamic array

**Concepts:** `realloc`, growth factor, amortised cost
**Difficulty:** ⭐⭐

A growable array of `int`. This is the exercise that teaches why `realloc` is
dangerous in a way `malloc` is not: it can move your buffer, which invalidates
every pointer anyone still holds into it.

## The problem

[`vec.h`](vec.h) declares the interface, [`vec.c`](vec.c) is the solution, and
[`main.c`](main.c) is the test suite.

| Function | Does |
|----------|------|
| `vec_init(v)` | Zero-capacity vector, no allocation yet |
| `vec_push(v, x)` | Append, growing if needed. Returns 0 on success, -1 on allocation failure |
| `vec_pop(v, out)` | Remove and return the last element. -1 if empty |
| `vec_get(v, i)` | Element at `i`. Undefined behaviour past `len`, deliberately |
| `vec_reserve(v, n)` | Ensure capacity for at least `n` without changing `len` |
| `vec_free(v)` | Release the buffer and return to the zero state |

Constraints: standard library only, no leaks, and `vec_free` must leave the
struct safe to `vec_free` again. A failed push must leave the vector unchanged
and still valid.

## Build and run

From this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c vec.c -o vec && ./vec
# ok  c/01-dynamic-array  20027 checks passed
```

The program exits 0 when every check holds and stops with `FAIL file:line` on
the first one that does not. To check memory safety as well:

```bash
cc -std=c11 -Wall -Wextra -g -fsanitize=address,undefined main.c vec.c -o vec && ./vec
```

## What to notice

**The growth factor is a real decision, not a detail.** Doubling gives
amortised O(1) push. Growing by a constant gives O(n) per push and quadratic
total cost, which you can feel on a million elements. Growing by 1.5 reuses
freed blocks better because the sum of previous allocations eventually exceeds
the next request; doubling never does. glibc, MSVC, and Rust all pick
differently here, and all three have a defensible reason.

**`realloc` can move the buffer.** Any pointer or index-derived pointer handed
out before a `push` may now dangle. The bug is invisible until the one push that
happens to cross a page boundary, which is why the test suite pushes 10,000
elements and re-reads every one.

**`realloc(p, 0)` is not `free(p)`.** In C17 it is implementation-defined, and
in C23 it is undefined. Do not rely on it.

## The solution, step by step

### From first principles

1. **A contiguous array is the only representation that gives O(1) `get`.**
   `vec_get` is `data[i]`, one address computation. So the elements must live in
   one block, and that block must be replaced when it fills.
2. **Replacing the block costs O(len), so it must happen rarely.** If capacity
   grows by a constant `k`, pushing `n` elements copies roughly
   `k + 2k + ... + n` elements, which is O(n²/k). If it grows by a factor (here
   2), the copies sum to at most `n + n/2 + n/4 + ... < 2n`, so each push costs
   O(1) amortised. That forces geometric growth.
3. **Length and capacity are separate numbers.** `len` is how many elements are
   live, `cap` how many fit. `len <= cap` is the invariant every function keeps,
   and `cap` only grows until `vec_free`.
4. **Allocation can fail, and the vector must survive it.** `realloc` returns
   `NULL` on failure and leaves the old block untouched. So the result goes into
   a temporary and replaces `v->data` only on success. Then a failed push leaves
   `len`, `cap`, and the data exactly as they were.
5. **Size arithmetic can overflow before `realloc` sees it.** `cap * 2` and
   `cap * sizeof(int)` are `size_t` multiplications that wrap silently. A wrapped
   size makes `realloc` hand back a small block that the code then writes past.
   Both products are checked against `SIZE_MAX` before they are computed.
6. **An empty vector should cost nothing.** `vec_init` sets everything to
   zero and allocates nothing, and `vec_free` returns to that same state, which
   makes a second `vec_free` a harmless `free(NULL)`.

### The code, annotated

All growth goes through `vec_reserve`, so the policy lives in one place:

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

`realloc(NULL, size)` behaves like `malloc(size)`, which is why the first call
needs no special case: a fresh vector has `data == NULL`.

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

Full solution: [`vec.c`](vec.c).

### Why this solution

- **Cost.** `vec_get`, `vec_pop`: O(1). `vec_push`: O(1) amortised, O(n) on the
  push that grows. After `n` pushes, `cap < 2n` (or 4 for tiny `n`), so wasted
  space is under half the buffer. The growth test asserts `cap < 4n` to reject
  arithmetic growth.
- **Factor 2 versus 1.5.** 1.5 lets the allocator reuse earlier freed blocks
  for later requests; 2 does fewer reallocations and keeps the arithmetic
  trivial. Both are amortised O(1). The reference picks 2 for simplicity.
- **Starting at 4.** Starting at 1 does three reallocations to reach 4
  elements, for no benefit. Starting at 0 is impossible with pure doubling.
- **No bounds check in `vec_get`.** A checked accessor would need an error
  channel (the return value is already the element). The interface puts the
  check on the caller, like `std::vector::operator[]`.
- **Alternative: a linked list of chunks.** Growth never moves elements, so
  pointers stay valid, but `get` becomes O(n / chunk) and loses cache locality.
  That trade is worth it only when pointer stability matters more than indexing.

### Explain it back

1. Why does `vec_reserve` store `realloc`'s result in `grown` before assigning
   it to `v->data`?

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

## Extending it

The reference stops at `int`. Making it generic is the real lesson about C: it
has no generics, so you either write the same code per type, cast through
`void *` and an element size (losing type safety), or generate per-type code
with a macro such as `VEC_DEFINE(name, T)` (losing debuggability). Each choice
gives something up; pick one deliberately and know which.
