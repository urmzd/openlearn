# A growable array

Every serious C program eventually needs `list.append`, and C does not have one. The first place this book's system feels the absence is the paged KV pool built in [Paged KV block pool](../../build-an-llm/p08-inference/08-paged-kv-block-pool.md) in *Build Your Own LLM System*. That pool gives each sequence a **block table**: the list of block ids that hold its keys and values, one per 16 tokens. A sequence's table grows by one id every 16 generated tokens, and nobody knows in advance how long a request will run. You can either preallocate the worst case for every sequence, which wastes memory on short requests, or grow on demand.

Growing on demand has two classic failures. Grow by a fixed amount and a long generation reallocates and copies its table over and over, a "realloc storm" that `--kv-stats` shows as a spike in allocator calls. Grow carelessly and an allocation failure under memory pressure leaves the table half updated, which corrupts a running request. This section writes the array once, generic over the element size, with the growth rule that makes appends cheap and the failure rule that makes them safe.

The C you need is in [the C primer](../../programming-languages/02-language-and-tool-primers/03-c.md) from *Programming Languages* (pointers, `sizeof`, `memcpy`), and the allocation hooks and error reporting come from [C runtime support and arena allocator with marks](../../build-an-llm/p09-kernels/02-arena-allocator-with-marks.md). The interface is fixed by the contract header [`tinyllm/ds.h`](../../../../course/contracts/c/include/tinyllm/ds.h) (its `tl_vec` section), under the rules every C unit follows in [`c/ABI.md`](../../../../course/contracts/c/ABI.md). The full solution is [`vec.c`](../../../../course/solutions/c/src/ds/vec.c).

> **Learning objectives**
> - Explain why doubling makes $n$ appends cost fewer than $2n$ element copies, and why a constant step costs $O(n^2)$.
> - Lay out a type-erased array where every offset is `i * elem` and every copy is `n * elem` bytes.
> - Order the steps of a growth (allocate, copy, free, publish) so that an allocation failure leaves the array exactly as it was.
> - Guard byte-count arithmetic against `size_t` overflow.
> - Test a container against a plain-array model, plus the fault and boundary cases a random model cannot reach.

## The state of a vec

A growable array is a block of `cap` slots of `elem` bytes each, `len` of them in use. Appending to a full block allocates one at least twice as large, copies `len * elem` bytes, and frees the old one. The notation below is used throughout the section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| `data` | pointer to the block, `NULL` before the first allocation | `void *` |
| `len` | elements in use | `size_t` |
| `cap` | elements the block can hold, `cap >= len` | `size_t` |
| `elem` | bytes per element, fixed at `tl_vec_init` | `size_t > 0` |
| $n$ | number of pushes | integer |
| $c_0$ | the first capacity (4 here) | integer |
| $g$ | the growth factor, `cap` becomes $g \cdot$ `cap` | $g \ge 2$ |

### Type erasure

`tl_vec` stores bytes, not a type. Element $i$ lives at `(char *)data + i * elem`, and a push copies `elem` bytes from the caller's pointer. One implementation serves the `uint32_t` block ids of the KV pool, 16-byte structs, or anything else. The price is that every size computation must multiply by `elem`, and the compiler cannot check that you did. An element size of 3 is the cheapest way to catch code that silently assumed 4 or 8.

## Why the array must grow geometrically

When `len == cap`, a push must move everything to a larger block. With growth factor $g$ starting from $c_0$, the moves happen at sizes $c_0, g c_0, g^2 c_0, \dots$, each copying the current length. After $n$ pushes the total number of element copies is

$$c_0 + g c_0 + \dots + g^{j} c_0 \;<\; \frac{g}{g - 1}\, n \;\le\; 2n \quad (g = 2),$$

because the last move copied fewer than $n$ elements and each earlier one at most $1/g$ of the next. So $n$ pushes cost $O(n)$ copies in total, $O(1)$ amortized per push, and the number of moves is about $\log_g(n / c_0)$.

Growing by a constant $a$ instead moves at sizes $a, 2a, 3a, \dots$, a total of $a(1 + 2 + \dots + n/a) \approx n^2 / (2a)$ copies: quadratic. Geometric growth has one more pleasant property: the capacity stays within a factor $g$ of the length, so at most half the block is wasted.

### Reserving ahead

Sometimes the caller knows the final size. A request's `max_tokens` fixes its block table's maximum length, for instance. `tl_vec_reserve(v, n)` then allocates once for exactly $n$ elements, and the next $n - $ `len` pushes never move. Reserve never shrinks: asking for less than the capacity is a no-op, because a caller holding a pointer into the block must be able to keep it.

## Failing safely

Cheap growth is half the story. The other half is what happens when the allocator says no. Every allocation goes through `tl_alloc`, which calls the installed allocator hook from the C runtime support, and a test can install a counting allocator that fails on demand.

A growth is three steps: allocate the new block, copy, free the old one. Doing them in that order makes a failure harmless. If the allocation fails, nothing has been touched yet, so `push` returns `TL_ENOMEM` with `data`, `len`, `cap`, and the contents unchanged. Any other order loses data: freeing first destroys the contents, and updating `cap` or `len` first leaves the vec claiming room or elements it does not have.

There is one more way a growth can go wrong before it starts. The byte count `cap * elem` can overflow `size_t`; checking `cap > SIZE_MAX / elem` before multiplying turns a would-be tiny allocation into `TL_EINVAL`.

### The invariants

Put together, these rules give a set of invariants that hold after every operation:

- `len <= cap`;
- `data == NULL` exactly when `cap == 0`;
- bytes `[0, len * elem)` hold the pushed elements in order;
- `tl_vec_at(v, i)` is non-`NULL` exactly for `i < len`;
- every block the vec allocated is either its current `data` or freed.

A property test can check all of them after each of a million random operations against a plain array, for an 8-byte and a 3-byte element.

## Working it by hand

Push the `uint32_t` values 10, 20, 30, 40, 50 into a fresh vec (`elem = 4`). The table traces `len` and `cap` through each push.

| push | before: `len`, `cap` | what happens | after: `len`, `cap` |
|---|---|---|---|
| 10 | 0, 0 | `len == cap`: allocate $c_0 = 4$ slots (16 bytes); nothing to copy | 1, 4 |
| 20 | 1, 4 | room: copy 4 bytes to `data + 1 * 4` | 2, 4 |
| 30 | 2, 4 | room | 3, 4 |
| 40 | 3, 4 | room | 4, 4 |
| 50 | 4, 4 | full: allocate 8 slots (32 bytes), copy `4 * 4 = 16` bytes, free the old block, then write at `data + 4 * 4` | 5, 8 |

Total copies: 4 elements for 5 pushes. `tl_vec_at(v, 4)` points at 50; `tl_vec_at(v, 5)` is `NULL`. If the allocation of the 8-slot block had failed, `push` would return `TL_ENOMEM` with `len = 4`, `cap = 4`, the same `data`, and 10 to 40 still in place.

## The interface

The contract header pins down five functions:

```c
/* tinyllm/ds.h (ds.01) */
typedef struct { void *data; size_t len, cap, elem; } tl_vec;
tl_status tl_vec_init(tl_vec *v, size_t elem);           /* TL_EINVAL for NULL or elem 0 */
tl_status tl_vec_reserve(tl_vec *v, size_t cap_min);     /* never shrinks; unchanged on failure */
tl_status tl_vec_push(tl_vec *v, const void *x);         /* at least doubles; unchanged on failure */
void     *tl_vec_at(const tl_vec *v, size_t i);          /* NULL (error slot set) for i >= len */
void      tl_vec_free(tl_vec *v);                         /* back to the init state, elem kept */
```

The caller may read `data[0 .. len * elem)` directly. The vec is not thread-safe; as `c/ABI.md` says, the caller serializes. Blocks are 16-byte aligned, enough for any scalar element.

## Building it step by step

The design follows from six observations, each a direct consequence of the principles above. Try to predict each step before reading it.

1. **The vec knows only sizes.** Without a type, the element at index $i$ can only be found by arithmetic: `(char *)data + i * elem`. Every copy is `n * elem` bytes. The `char *` cast is needed because C does not do arithmetic on `void *`.
2. **Appends must be $O(1)$ amortized, so capacity must grow geometrically.** Doubling copies fewer than $2n$ elements for $n$ pushes; a constant step copies $\Theta(n^2)$. So a full vec doubles, and an empty one starts at $c_0 = 4$.
3. **Growth must be failure-atomic, which fixes the order.** Allocate the new block first (the only step that can fail), then copy, then free the old block, then update `data` and `cap`. If the allocation fails, nothing has been written.
4. **Byte counts can overflow.** `cap * elem` wraps around for a large `cap`, so the multiplication is guarded by `cap > SIZE_MAX / elem`; doubling is guarded by `cap > SIZE_MAX / 2`.
5. **Reserve only grows.** A caller may hold a pointer into the block; shrinking would invalidate it. Asking for less than the current capacity is a no-op.
6. **Reads past `len` are errors, not spare capacity.** Bytes in `[len, cap)` are uninitialized, so `tl_vec_at` refuses `i >= len`.

### The one function that moves memory

Steps 3 and 4 live in a single helper, used by both `push` and `reserve`:

```c
/* Replace the block with one of exactly `cap` elements (cap >= len). On
 * failure nothing changes. */
static tl_status vec_regrow(tl_vec *v, size_t cap) {
    if (cap > SIZE_MAX / v->elem) {               /* cap * elem would wrap around */
        tl_set_last_error("tl_vec: capacity * elem overflows size_t");
        return TL_EINVAL;
    }
    void *p = tl_alloc(cap * v->elem, TL_VEC_ALIGN); /* the only step that can fail */
    if (p == NULL) return TL_ENOMEM;              /* v untouched: data, len, cap, contents */
    if (v->len) memcpy(p, v->data, v->len * v->elem); /* bytes, not elements */
    tl_free(v->data);                             /* only after the copy succeeded */
    v->data = p;                                  /* publish the new block last */
    v->cap = cap;
    return TL_OK;
}
```

`TL_VEC_ALIGN` is 16: the vec does not know what it holds, so it aligns for any scalar type. `tl_alloc` and `tl_free` are the shared allocation hooks, which is what lets a test make a specific allocation fail. `memcpy` is skipped when `len` is 0 because `v->data` may then be `NULL` (a fresh vec), and passing `NULL` to `memcpy` is undefined behavior even for zero bytes.

### Push, reserve, at, and free

`push` grows only when full, and by doubling (step 2):

```c
tl_status tl_vec_push(tl_vec *v, const void *x) {
    if (v == NULL || x == NULL || v->elem == 0) { /* elem 0: never initialized */
        tl_set_last_error("tl_vec_push: v or x is NULL, or v is not initialized");
        return TL_EINVAL;
    }
    if (v->len == v->cap) {
        if (v->cap > SIZE_MAX / 2) {              /* 2 * cap would wrap around */
            tl_set_last_error("tl_vec_push: capacity overflow");
            return TL_ENOMEM;
        }
        size_t cap = v->cap ? 2 * v->cap : TL_VEC_MIN_CAP; /* 0 -> 4 -> 8 -> 16 ... */
        tl_status s = vec_regrow(v, cap);
        if (s != TL_OK) return s;                 /* failure: v exactly as before */
    }
    memcpy((char *)v->data + v->len * v->elem, x, v->elem); /* slot len, not cap - 1 */
    v->len++;                                     /* only after the element is in place */
    return TL_OK;
}
```

`reserve` is step 5 in one line before the shared growth:

```c
tl_status tl_vec_reserve(tl_vec *v, size_t cap_min) {
    // ... TL_EINVAL for NULL or an uninitialized vec
    if (cap_min <= v->cap) return TL_OK; /* never shrinks */
    return vec_regrow(v, cap_min);       /* exactly cap_min: the caller knows the final size */
}
```

`at` is step 6, and `free` returns to the init state while keeping `elem`, so the vec can be reused and a second free is harmless (`tl_free(NULL)` does nothing):

```c
void *tl_vec_at(const tl_vec *v, size_t i) {
    if (v == NULL || i >= v->len) {       /* >=, not >: index len is spare capacity */
        tl_set_last_error("tl_vec_at: index out of range");
        return NULL;
    }
    return (char *)v->data + i * v->elem;
}

void tl_vec_free(tl_vec *v) {
    if (v == NULL) return;
    tl_free(v->data);
    v->data = NULL;                       /* data == NULL exactly when cap == 0 */
    v->len = 0;
    v->cap = 0;                           /* elem kept: push works again without init */
}
```

The full solution is [`vec.c`](../../../../course/solutions/c/src/ds/vec.c).

### Why this design

The costs come out as promised. `push` is $O(1)$ amortized (fewer than $2n$ element copies for $n$ pushes) and $O(\text{len})$ in the worst case, on the push that grows. `at` is $O(1)$. Space is at most $2 \cdot$ `len` elements once the vec has grown past $c_0$.

The explicit allocate, copy, free sequence is deliberate, and you may wonder why it is not simply `realloc`. `realloc` can extend in place and avoid the copy, but on failure it returns `NULL` while the old block stays valid, so code that writes `v->data = realloc(v->data, ...)` loses the block on failure. It also does not take an alignment. The explicit sequence makes the failure rule visible and works through the allocator hook.

The factor is 2 rather than 1.5: fewer moves and simpler arithmetic. A factor below the golden ratio (about 1.618) lets a later growth reuse the memory freed by earlier ones, which is why folly and MSVC pick 1.5; for per-sequence block tables of a few hundred ids, the difference does not matter.

Type erasure wins over a macro-generated vec per type because it means one implementation and one set of tests. The cost is that `elem` must be multiplied in every offset, which the 3-byte element in the property test checks. The one tradeoff the design accepts is that the vec never shrinks on its own. A sequence's block table is freed when the sequence ends, so shrinking would only add moves.

## Testing it and what can go wrong

A correct vec has to handle the cases below. Each row says what kind of check exercises it and why it matters to the callers.

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | capacities 4 then 8, contents through `tl_vec_at` and `data` | the growth rule is pinned down |
| 100000 pushes | unit | the block moves at most 18 times; `cap` stays within $[len, 2\,len]$ | no realloc storm in block tables |
| reserve | unit | exact reservation, no move within it, no allocation for a smaller request | the KV pool reserves a request's table up front |
| `tl_vec_at` at and past `len` | boundary | `NULL`, error slot set | no reads of spare capacity |
| an allocation failure during growth | fault | `TL_ENOMEM`; same `data`, `len`, `cap`, contents; no leak; the next push works | a failed request does not corrupt a sequence |
| free, then reuse | unit | init state after free, `elem` kept, a double free harmless | sequences reuse their table |
| bad arguments | boundary | `NULL`, `elem` 0, an overflowing reservation give `TL_EINVAL` | no crash, no tiny allocation |
| a million random operations | property | against a plain array, for 8- and 3-byte elements | every invariant listed earlier |

The solution ships with a model test suite that checks these properties: [`ds01_vec_test.c`](../../../../course/solutions/learner-tests/ds.01/c/tests/ds01_vec_test.c). It shows how to test a container by comparing it, operation by operation, with the simplest correct thing (a flat array), plus the fault and boundary cases a random model cannot reach: a forced allocation failure, `NULL` arguments, an overflowing reservation.

Most bugs in a hand-written vec are one of a small set of mistakes. Each has a recognizable symptom, and the solution above avoids each with a specific line.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| growing by a constant | quadratic copying; `--kv-stats` shows allocator spikes | `push` doubles: `2 * v->cap` |
| copying `len` bytes instead of `len * elem` | every element after the first is garbage after a move | `memcpy(p, v->data, v->len * v->elem)` |
| touching the vec before the allocation succeeded, or freeing the old block first | a failed push corrupts the table | `vec_regrow` allocates first, frees after the copy, and updates `data` and `cap` last |
| `i > len` in the bounds check | a pointer into uninitialized spare capacity | `i >= v->len` |
| writing at `cap - 1` instead of `len` | elements overwrite each other, ASan reports overflow on growth | the new element goes to `data + len * elem` |
| reserve shrinking the block | a caller's pointer dangles | `cap_min <= v->cap` returns without moving |
| not freeing the old block | the counting allocator reports a leak on every growth | `tl_free(v->data)` after the copy |
| free leaving `len` and `cap` behind | a reused vec pushes into a NULL block | `tl_vec_free` zeroes `data`, `len`, and `cap` |
| no overflow check on `cap * elem` | a huge reservation allocates a few bytes | `cap > SIZE_MAX / v->elem` returns `TL_EINVAL` |

> **Warning:** `v->data = realloc(v->data, n)` is the most common version of the failure-atomicity bug. When `realloc` fails it returns `NULL` and leaves the old block alive, so the assignment leaks the block and loses every element.

## Where this leads

The first caller is the [Paged KV block pool](../../build-an-llm/p08-inference/08-paged-kv-block-pool.md), where each sequence's block table is a `tl_vec` of `uint32_t` block ids. The next section, [the Swiss table](02-swiss-table.md), grows by the same allocate, copy, free order and obeys the same failure rule, and [heap top-k in C](04-binary-heap-top-k.md) keeps its working set in the same kind of block. Together these form the optional standalone C group of the chapter.

Production containers make the same choices with different weights:

- Rust `Vec<T>` is typed, uses `realloc` in place when the allocator can extend the block, and grows by a factor of 2; see `library/alloc/src/raw_vec.rs`.
- C++ `std::vector::push_back` gives the strong exception guarantee, the C++ name for failure atomicity, with factor 1.5 in MSVC and 2 in libstdc++; see `bits/vector.tcc` (`_M_realloc_insert`).
- Facebook `folly::fbvector` uses factor 1.5 so freed blocks can be reused by later growth, together with `jemalloc` size classes; see `folly/FBVector.h`.
- vLLM's `BlockTable` preallocates per-request tables on the GPU and appends block by block; see `vllm/v1/worker/block_table.py`.

### Further reading

- Cormen et al., *Introduction to Algorithms* (3rd ed.), section 17.4 (dynamic tables, amortized analysis).
- Sedgewick and Wayne, *Algorithms* (4th ed.), section 1.3 (resizing arrays).
- The dynamic-array warm-up exercise in [the C primer](../../programming-languages/02-language-and-tool-primers/03-c.md).

## Key terms

- **Block table**: the per-sequence list of KV block ids, one per 16 tokens.
- **Type erasure**: storing elements as raw bytes of a fixed size `elem`, so one implementation serves every element type.
- **Geometric growth**: multiplying the capacity by a constant factor $g \ge 2$ when full, which makes appends $O(1)$ amortized.
- **Amortized cost**: the total cost of a sequence of operations divided by their number.
- **Failure atomicity**: the property that a failed operation leaves the structure exactly as it was.
- **Reserve**: a one-time allocation of a known final capacity, which never shrinks.

## Summary

- A growable array is `cap` slots of `elem` bytes, `len` in use; a full push allocates a block twice as large, copies `len * elem` bytes, and frees the old one.
- Doubling makes $n$ appends cost fewer than $2n$ element copies, $O(1)$ amortized; growing by a constant costs $O(n^2)$.
- Allocate, copy, free, publish, in that order, so that `TL_ENOMEM` leaves the vec unchanged and nothing leaks.
- Type erasure means every offset is `i * elem`; a 3-byte element in tests catches code that assumed 4 or 8.
- Guard `cap * elem` with `cap > SIZE_MAX / elem`, and refuse reads at `i >= len`.

## Review questions

1. In the hand example, why does the push of 50 copy 16 bytes and not 4?
   <details><summary>Answer</summary>The copy moves all `len = 4` existing elements of `elem = 4` bytes: `len * elem = 16`. Copying `len` bytes (4) would move only the first element correctly and leave 20, 30, 40 as garbage in the new block.</details>
2. What would go wrong if `vec_regrow` freed the old block before calling `tl_alloc`?
   <details><summary>Answer</summary>If the allocation then failed, the contents would already be gone and `v->data` would point to freed memory. Allocating first means the only failure point comes before any change, so `TL_ENOMEM` leaves the vec exactly as it was.</details>
3. Why is the overflow check `cap > SIZE_MAX / elem` and not `cap * elem > SIZE_MAX`?
   <details><summary>Answer</summary>`cap * elem` is computed in `size_t` and wraps around modulo $2^{64}$, so it can never be greater than `SIZE_MAX`; the wrapped value may be small, giving a tiny allocation for a huge request. Dividing first asks the same question without overflowing.</details>
4. Why must `reserve` never shrink, even when asked for less?
   <details><summary>Answer</summary>Shrinking moves the block, which invalidates any pointer the caller holds into it (from `tl_vec_at` or `data`), and it could drop elements if asked for less than `len`. Reserve is a promise about future pushes, not a request to reallocate.</details>
5. How many times does the block move during 100000 pushes, and why is that at most 18?
   <details><summary>Answer</summary>The capacities are $4, 8, 16, \dots, 4 \cdot 2^{k}$; the block moves once to allocate 4 and then once per doubling until the capacity reaches at least 100000. $4 \cdot 2^{15} = 131072 \ge 100000$, so there are 16 moves (the first allocation plus 15 doublings), within the bound of 18.</details>
6. Why does `vec_regrow` skip `memcpy` when `len` is 0?
   <details><summary>Answer</summary>A fresh vec has `data == NULL`, and passing `NULL` to `memcpy` is undefined behavior even for a zero-byte copy.</details>
