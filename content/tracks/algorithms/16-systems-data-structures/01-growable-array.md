# Growable array tl_vec (type-erased, optional C)

## Overview

| | |
|---|---|
| **Module** | `ds.01` · side · C · Pass 6 · 2 to 3 h |
| **Solution** | [`vec.c`](../../../../course/solutions/c/src/ds/vec.c): `vec_regrow` (the one place memory moves), `tl_vec_init`, `tl_vec_reserve`, `tl_vec_push`, `tl_vec_at`, `tl_vec_free` · a model property-test suite: [`ds01_vec_test.c`](../../../../course/solutions/learner-tests/ds.01/c/tests/ds01_vec_test.c) |
| **Contract** | [`tinyllm/ds.h`](../../../../course/contracts/c/include/tinyllm/ds.h) (the ds.01 section) · the rules every unit follows: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Needs** | `rt.02` (C allocation and error support) · reading: `lang.03` the C primer (pointers, `sizeof`, `memcpy`) |
| **Used by** | `rt.04` keeps one `tl_vec` of block ids per sequence (its block table) · `ds.02` and `ds.04` build on this chapter's growth and failure rules · part of the optional standalone C module group |
| **Optional depth** | Cormen et al., *Introduction to Algorithms* (3rd ed.), section 17.4 (dynamic tables, amortized analysis); Sedgewick and Wayne, *Algorithms* (4th ed.), section 1.3 (resizing arrays); the dynamic-array warm-up exercise of `lang.03` |

## Key Takeaways

- A growable array is a block of `cap` slots of `elem` bytes, `len` of them in use; appending to a full block allocates one at least twice as large, copies `len * elem` bytes, and frees the old one.
- Doubling makes $n$ appends cost fewer than $2n$ element copies in total, $O(1)$ amortized; growing by a constant makes them cost $O(n^2)$.
- Every byte comes through the allocator hook, so a test can make the next allocation fail; a failed growth must leave the vec exactly as it was, with nothing leaked.
- Type erasure means the vec only knows sizes: every offset is `i * elem`, every copy `n * elem` bytes, and an element size of 3 catches code that silently assumed 4 or 8.

## How to read this chapter

Sections 1 to 3 derive the growth and failure rules from first principles and push five values by hand. Section 4 fixes the interface (the contract header). Section 5 walks through `vec.c` line by line and explains why each choice is the right one; try to predict each step before reading it.

---

## 1. Why now

The paged KV pool built next (`rt.04`) gives each sequence a **block table**: the list of block ids that hold its keys and values, one per 16 tokens. A sequence's table grows by one id every 16 generated tokens, and nobody knows in advance how long a request will run. In C there is no `list.append`: you either preallocate the worst case for every sequence (wasting memory on short requests) or grow on demand. Growing on demand has two classic failures. Grow by a fixed amount and a long generation reallocates and copies its table over and over, a "realloc storm" that `--kv-stats` shows as a spike in allocator calls. Grow carelessly and an allocation failure under memory pressure leaves the table half updated, which corrupts a running request. This module writes the array once, generic over the element size, with the growth rule that makes appends cheap and the failure rule that makes them safe.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| `data` | pointer to the block, `NULL` before the first allocation | `void *` |
| `len` | elements in use | `size_t` |
| `cap` | elements the block can hold, `cap >= len` | `size_t` |
| `elem` | bytes per element, fixed at `tl_vec_init` | `size_t > 0` |
| $n$ | number of pushes | integer |
| $c_0$ | the first capacity (4 here) | integer |
| $g$ | the growth factor, `cap` becomes $g \cdot$ `cap` | $g \ge 2$ |

### 2.1 Type erasure

`tl_vec` stores bytes, not a type: element $i$ lives at `(char *)data + i * elem`, and a push copies `elem` bytes from the caller's pointer. One implementation serves `uint32_t` block ids (`rt.04`), 16-byte structs, or anything else. The price is that every size computation must multiply by `elem`, and the compiler cannot check it.

### 2.2 Geometric growth and the amortized cost

When `len == cap`, a push must move everything to a larger block. With growth factor $g$ starting from $c_0$, the moves happen at sizes $c_0, g c_0, g^2 c_0, \dots$, each copying the current length. After $n$ pushes the total number of element copies is

$$c_0 + g c_0 + \dots + g^{j} c_0 \;<\; \frac{g}{g - 1}\, n \;\le\; 2n \quad (g = 2),$$

because the last move copied fewer than $n$ elements and each earlier one at most $1/g$ of the next. So $n$ pushes cost $O(n)$ copies in total, $O(1)$ amortized per push, and the number of moves is about $\log_g(n / c_0)$. Growing by a constant $a$ instead moves at sizes $a, 2a, 3a, \dots$, a total of $a(1 + 2 + \dots + n/a) \approx n^2 / (2a)$ copies: quadratic. The capacity also stays within a factor $g$ of the length, so at most half the block is wasted.

### 2.3 Reserve

If the caller knows the final size (a request's `max_tokens` fixes its block table's maximum), `tl_vec_reserve(v, n)` allocates once for exactly $n$ elements, and the next $n - $ `len` pushes never move. Reserve never shrinks: asking for less than the capacity is a no-op, because a caller holding a pointer into the block must be able to keep it.

### 2.4 Failure atomicity

Every allocation goes through `tl_alloc`, which calls the installed allocator hook (`rt.01`); a test installs a counting allocator that can fail on demand. A growth is three steps: allocate the new block, copy, free the old one. Doing them in that order makes a failure harmless: if the allocation fails, nothing has been touched yet, so `push` returns `TL_ENOMEM` with `data`, `len`, `cap`, and the contents unchanged. Any other order loses data: freeing first destroys the contents, and updating `cap` or `len` first leaves the vec claiming room or elements it does not have. The byte count `cap * elem` can overflow `size_t`; checking `cap > SIZE_MAX / elem` before multiplying turns a would-be tiny allocation into `TL_EINVAL`.

### 2.5 Invariants

After every operation: `len <= cap`; `data == NULL` exactly when `cap == 0`; bytes `[0, len * elem)` hold the pushed elements in order; `tl_vec_at(v, i)` is non-`NULL` exactly for `i < len`; and every block the vec allocated is either its current `data` or freed. A property test can check all of them after each of a million random operations against a plain array, for an 8-byte and a 3-byte element.

## 3. Worked example by hand

Push the `uint32_t` values 10, 20, 30, 40, 50 into a fresh vec (`elem = 4`).

| push | before: `len`, `cap` | what happens | after: `len`, `cap` |
|---|---|---|---|
| 10 | 0, 0 | `len == cap`: allocate $c_0 = 4$ slots (16 bytes); nothing to copy | 1, 4 |
| 20 | 1, 4 | room: copy 4 bytes to `data + 1 * 4` | 2, 4 |
| 30 | 2, 4 | room | 3, 4 |
| 40 | 3, 4 | room | 4, 4 |
| 50 | 4, 4 | full: allocate 8 slots (32 bytes), copy `4 * 4 = 16` bytes, free the old block, then write at `data + 4 * 4` | 5, 8 |

Total copies: 4 elements for 5 pushes. `tl_vec_at(v, 4)` points at 50; `tl_vec_at(v, 5)` is `NULL`. If the allocation of the 8-slot block had failed, `push` would return `TL_ENOMEM` with `len = 4`, `cap = 4`, the same `data`, and 10 to 40 still in place.

## 4. The interface

```c
/* tinyllm/ds.h (ds.01) */
typedef struct { void *data; size_t len, cap, elem; } tl_vec;
tl_status tl_vec_init(tl_vec *v, size_t elem);           /* TL_EINVAL for NULL or elem 0 */
tl_status tl_vec_reserve(tl_vec *v, size_t cap_min);     /* never shrinks; unchanged on failure */
tl_status tl_vec_push(tl_vec *v, const void *x);         /* at least doubles; unchanged on failure */
void     *tl_vec_at(const tl_vec *v, size_t i);          /* NULL (error slot set) for i >= len */
void      tl_vec_free(tl_vec *v);                         /* back to the init state, elem kept */
```

The caller may read `data[0 .. len * elem)` directly. Not thread-safe: the caller serializes (`c/ABI.md`). Blocks are 16-byte aligned, enough for any scalar element.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example (section 3) | unit | capacities 4 then 8, contents through `tl_vec_at` and `data` | the growth rule is pinned down |
| 100000 pushes | unit | the block moves at most 18 times; `cap` stays within $[len, 2\,len]$ | no realloc storm in block tables |
| reserve | unit | exact reservation, no move within it, no allocation for a smaller request | `rt.04` reserves a request's table up front |
| `tl_vec_at` at and past `len` | boundary | `NULL`, error slot set | no reads of spare capacity |
| an allocation failure during growth | fault | `TL_ENOMEM`; same `data`, `len`, `cap`, contents; no leak; the next push works | a failed request does not corrupt a sequence |
| free, then reuse | unit | init state after free, `elem` kept, a double free harmless | sequences reuse their table |
| bad arguments | boundary | `NULL`, `elem` 0, an overflowing reservation give `TL_EINVAL` | no crash, no tiny allocation |
| a million random operations | property | against a plain array, for 8- and 3-byte elements | every invariant of section 2.5 |

The solution ships with a model test suite that checks these properties against a plain-array model: [`ds01_vec_test.c`](../../../../course/solutions/learner-tests/ds.01/c/tests/ds01_vec_test.c). It shows how to test a container by comparing it, operation by operation, with the simplest correct thing (a flat array), plus the fault and boundary cases a random model cannot reach (a forced allocation failure, `NULL` arguments, an overflowing reservation).

## 5. The solution, step by step

### 5.1 From first principles

1. **The vec knows only sizes.** Without a type, the element at index $i$ can only be found by arithmetic: `(char *)data + i * elem`. Every copy is `n * elem` bytes. The `char *` cast is needed because C does not do arithmetic on `void *`.
2. **Appends must be $O(1)$ amortized, so capacity must grow geometrically.** Section 2.2 shows doubling copies fewer than $2n$ elements for $n$ pushes; a constant step copies $\Theta(n^2)$. So a full vec doubles, and an empty one starts at $c_0 = 4$.
3. **Growth must be failure-atomic, which fixes the order.** Allocate the new block first (the only step that can fail), then copy, then free the old block, then update `data` and `cap`. If the allocation fails, nothing has been written.
4. **Byte counts can overflow.** `cap * elem` wraps around for a large `cap`, so the multiplication is guarded by `cap > SIZE_MAX / elem`; doubling is guarded by `cap > SIZE_MAX / 2`.
5. **Reserve only grows.** A caller may hold a pointer into the block; shrinking would invalidate it. Asking for less than the current capacity is a no-op.
6. **Reads past `len` are errors, not spare capacity.** Bytes in `[len, cap)` are uninitialized, so `tl_vec_at` refuses `i >= len`.

### 5.2 The code, annotated

The one function that moves memory, used by both `push` and `reserve`, is step 3 and step 4:

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

Full solution: [`vec.c`](../../../../course/solutions/c/src/ds/vec.c).

### 5.3 Why this solution

- **Cost.** `push` is $O(1)$ amortized (fewer than $2n$ element copies for $n$ pushes) and $O(\text{len})$ in the worst case, on the push that grows. `at` is $O(1)$. Space is at most $2 \cdot$ `len` elements once the vec has grown past $c_0$.
- **Allocate, copy, free** rather than `realloc`: `realloc` can extend in place and avoid the copy, but on failure it returns `NULL` while the old block stays valid, so code that writes `v->data = realloc(v->data, ...)` loses the block on failure. It also does not take an alignment. The explicit sequence makes the failure rule visible and works through the allocator hook.
- **Factor 2** rather than 1.5: fewer moves and simpler arithmetic. A factor below the golden ratio (about 1.618) lets a later growth reuse the memory freed by earlier ones, which is why folly and MSVC pick 1.5; for per-sequence block tables of a few hundred ids, the difference does not matter.
- **Type erasure** rather than a macro-generated vec per type: one implementation, one set of tests; the cost is that `elem` must be multiplied in every offset, which the 3-byte element in the property test checks.
- **Tradeoff accepted:** the vec never shrinks on its own. A sequence's block table is freed when the sequence ends, so shrinking would only add moves.

### 5.4 Explain it back

1. In section 3, why does the push of 50 copy 16 bytes and not 4?
   <details><summary>Answer</summary>The copy moves all `len = 4` existing elements of `elem = 4` bytes: `len * elem = 16`. Copying `len` bytes (4) would move only the first element correctly and leave 20, 30, 40 as garbage in the new block.</details>
2. What would go wrong if `vec_regrow` freed the old block before calling `tl_alloc`?
   <details><summary>Answer</summary>If the allocation then failed, the contents would already be gone and `v->data` would point to freed memory. Allocating first means the only failure point comes before any change, so `TL_ENOMEM` leaves the vec exactly as it was.</details>
3. Why is the overflow check `cap > SIZE_MAX / elem` and not `cap * elem > SIZE_MAX`?
   <details><summary>Answer</summary>`cap * elem` is computed in `size_t` and wraps around modulo $2^{64}$, so it can never be greater than `SIZE_MAX`; the wrapped value may be small, giving a tiny allocation for a huge request. Dividing first asks the same question without overflowing.</details>
4. Why must `reserve` never shrink, even when asked for less?
   <details><summary>Answer</summary>Shrinking moves the block, which invalidates any pointer the caller holds into it (from `tl_vec_at` or `data`), and it could drop elements if asked for less than `len`. Reserve is a promise about future pushes, not a request to reallocate.</details>
5. How many times does the block move during 100000 pushes, and why is that at most 18?
   <details><summary>Answer</summary>The capacities are $4, 8, 16, \dots, 4 \cdot 2^{k}$; the block moves once to allocate 4 and then once per doubling until the capacity reaches at least 100000. $4 \cdot 2^{15} = 131072 \ge 100000$, so there are 16 moves (the first allocation plus 15 doublings), within the bound of 18.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. growing by a constant | quadratic copying; `--kv-stats` shows allocator spikes | `push` doubles: `2 * v->cap` |
| 2. copying `len` bytes instead of `len * elem` | every element after the first is garbage after a move | `memcpy(p, v->data, v->len * v->elem)` |
| 3. touching the vec before the allocation succeeded, or freeing the old block first | a failed push corrupts the table | `vec_regrow` allocates first, frees after the copy, and updates `data` and `cap` last |
| 4. `i > len` in the bounds check | a pointer into uninitialized spare capacity | `i >= v->len` |
| 5. writing at `cap - 1` instead of `len` | elements overwrite each other, ASan reports overflow on growth | the new element goes to `data + len * elem` |
| 6. reserve shrinking the block | a caller's pointer dangles | `cap_min <= v->cap` returns without moving |
| 7. not freeing the old block | the counting allocator reports a leak on every growth | `tl_free(v->data)` after the copy |
| 8. free leaving `len` and `cap` behind | a reused vec pushes into a NULL block | `tl_vec_free` zeroes `data`, `len`, and `cap` |
| 9. no overflow check on `cap * elem` | a huge reservation allocates a few bytes | `cap > SIZE_MAX / v->elem` returns `TL_EINVAL` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | allocation hooks and error reporting for standalone C modules |
| Back | `lang.03` | the C this needs: pointers, `sizeof`, `memcpy` |
| Forward | `rt.04` | each sequence's block table is a `tl_vec` of `uint32_t` block ids |
| Forward | `ds.02` | the Swiss table grows by the same allocate, copy, free order and the same failure rule |
| Forward | `ds.04` | heap top-k keeps its working set in the same kind of block |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_vec` | Rust `Vec<T>` | typed, with `realloc` in place when the allocator can extend the block; growth factor 2 | `library/alloc/src/raw_vec.rs` |
| `tl_vec_push` | C++ `std::vector::push_back` | strong exception guarantee, the C++ name for failure atomicity; factor 1.5 in MSVC, 2 in libstdc++ | `bits/vector.tcc` (`_M_realloc_insert`) |
| geometric growth | Facebook `folly::fbvector` | factor 1.5 so freed blocks can be reused by later growth, `jemalloc` size classes | `folly/FBVector.h` |
| block tables | vLLM `BlockTable` | preallocated per-request tables on the GPU, appended block by block | `vllm/v1/worker/block_table.py` |
