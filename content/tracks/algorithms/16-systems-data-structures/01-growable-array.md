<!-- ol:module ds.01 -->
# Growable array tl_vec (type-erased, optional C)

## Overview

| | |
|---|---|
| **Module** | `ds.01` · side · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/ds/vec.c`: `tl_vec_init`, `tl_vec_reserve`, `tl_vec_push`, `tl_vec_at`, `tl_vec_free` |
| **Contract** | [`tinyllm/ds.h`](../../../../course/contracts/c/include/tinyllm/ds.h) (the ds.01 section) · the rules every unit follows: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/ds.01/test_vec.c`, built with ASan, UBSan, and the counting allocator (what they check: section 4) · your own tests in `c/tests/ds01_vec_test.c`, rung R4 (properties with `ol_prop.h`), graded by mutation (threshold 0.80, every pitfall mutant required) |
| **Needs** | `rt.02` (C allocation and error support) · reading: `lang.03` the C primer (pointers, `sizeof`, `memcpy`) |
| **Used by** | `rt.04` keeps one `tl_vec` of block ids per sequence (its block table) · `ds.02` and `ds.04` build on this chapter's growth and failure rules |
| **Milestone** | `MS-L9`, the optional standalone C module group |
| **Optional depth** | Cormen et al., *Introduction to Algorithms* (3rd ed.), section 17.4 (dynamic tables, amortized analysis); Sedgewick and Wayne, *Algorithms* (4th ed.), section 1.3 (resizing arrays); the warm-up drill `practice/` C `01` (dynamic array) from `lang.03` |

## Key Takeaways

- A growable array is a block of `cap` slots of `elem` bytes, `len` of them in use; appending to a full block allocates one at least twice as large, copies `len * elem` bytes, and frees the old one (`hand_example`).
- Doubling makes $n$ appends cost fewer than $2n$ element copies in total, $O(1)$ amortized; growing by a constant makes them cost $O(n^2)$ (`growth_is_geometric`).
- Every byte comes through the allocator hook, so a test can make the next allocation fail; a failed growth must leave the vec exactly as it was, with nothing leaked (`alloc_failure_leaves_vec_unchanged`).
- Type erasure means the vec only knows sizes: every offset is `i * elem`, every copy `n * elem` bytes, and an element size of 3 catches code that silently assumed 4 or 8 (`random_ops_match_a_model`).

## How to work this chapter

```bash
ol start ds.01              # stubs vec.c into your repo; ds.h is in contracts/
ol tests ds.01              # read the test catalog first
ol check ds.01              # sanitized C tests, then your tests graded by mutation
ol mutate ds.01             # the full mutation grade of your tests
ol check ds.01 --ref-deps   # only if you skipped rt.01
ol diff  ds.01              # after passing: your code against the reference
```

---

## 1. Why now

The paged KV pool you build next (`rt.04`) gives each sequence a **block table**: the list of block ids that hold its keys and values, one per 16 tokens. A sequence's table grows by one id every 16 generated tokens, and nobody knows in advance how long a request will run. In C there is no `list.append`: you either preallocate the worst case for every sequence (wasting memory on short requests) or grow on demand. Growing on demand has two classic failures. Grow by a fixed amount and a long generation reallocates and copies its table over and over, a "realloc storm" that `--kv-stats` shows as a spike in allocator calls. Grow carelessly and an allocation failure under memory pressure leaves the table half updated, which corrupts a running request. This module writes the array once, generic over the element size, with the growth rule that makes appends cheap and the failure rule that makes them safe.

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

Every allocation goes through `tl_alloc`, which calls the installed allocator hook (`rt.01`); the course tests install a counting allocator that can fail on demand. A growth is three steps: allocate the new block, copy, free the old one. Doing them in that order makes a failure harmless: if the allocation fails, nothing has been touched yet, so `push` returns `TL_ENOMEM` with `data`, `len`, `cap`, and the contents unchanged. Any other order loses data: freeing first destroys the contents, and updating `cap` or `len` first leaves the vec claiming room or elements it does not have. The byte count `cap * elem` can overflow `size_t`; checking `cap > SIZE_MAX / elem` before multiplying turns a would-be tiny allocation into `TL_EINVAL`.

### 2.5 Invariants

After every operation: `len <= cap`; `data == NULL` exactly when `cap == 0`; bytes `[0, len * elem)` hold the pushed elements in order; `tl_vec_at(v, i)` is non-`NULL` exactly for `i < len`; and every block the vec allocated is either its current `data` or freed. The property test checks all of them after each of a million random operations against a plain array, for an 8-byte and a 3-byte element.

## 3. Worked example by hand

Push the `uint32_t` values 10, 20, 30, 40, 50 into a fresh vec (`elem = 4`).

| push | before: `len`, `cap` | what happens | after: `len`, `cap` |
|---|---|---|---|
| 10 | 0, 0 | `len == cap`: allocate $c_0 = 4$ slots (16 bytes); nothing to copy | 1, 4 |
| 20 | 1, 4 | room: copy 4 bytes to `data + 1 * 4` | 2, 4 |
| 30 | 2, 4 | room | 3, 4 |
| 40 | 3, 4 | room | 4, 4 |
| 50 | 4, 4 | full: allocate 8 slots (32 bytes), copy `4 * 4 = 16` bytes, free the old block, then write at `data + 4 * 4` | 5, 8 |

Total copies: 4 elements for 5 pushes. `tl_vec_at(v, 4)` points at 50; `tl_vec_at(v, 5)` is `NULL`. If the allocation of the 8-slot block had failed, `push` would return `TL_ENOMEM` with `len = 4`, `cap = 4`, the same `data`, and 10 to 40 still in place. This is the first case in section 4: `hand_example`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit | section 3: capacities 4 then 8, contents through `tl_vec_at` and `data` | you and the test agree on the growth rule |
| `growth_is_geometric` | unit | 100000 pushes move the block at most 18 times; `cap` within $[len, 2\,len]$ | no realloc storm in block tables |
| `reserve_grows_exactly_and_never_shrinks` | unit | exact reservation, no move within it, no allocation for a smaller request | `rt.04` reserves a request's table up front |
| `at_is_bounds_checked` | boundary | `NULL` at `len` and beyond, error slot set | no reads of spare capacity |
| `alloc_failure_leaves_vec_unchanged` | fault | `TL_ENOMEM`, same `data`, `len`, `cap`, contents; no leak; next push works | a failed request does not corrupt a sequence |
| `free_resets_and_vec_is_reusable` | unit | init state after free, `elem` kept, double free harmless | sequences reuse their table |
| `bad_arguments` | boundary | `NULL`, `elem` 0, overflowing reservation give `TL_EINVAL` | no crash, no tiny allocation |
| `random_ops_match_a_model` | property | a million random operations against a plain array, 8- and 3-byte elements | every invariant of section 2.5 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. growing by a constant | quadratic copying; `--kv-stats` shows allocator spikes | `growth_is_geometric` (mutant `s01`) |
| 2. copying `len` bytes instead of `len * elem` | every element after the first is garbage after a move | `hand_example` (mutant `s02`) |
| 3. touching the vec before the allocation succeeded, or freeing the old block first | a failed push corrupts the table | `alloc_failure_leaves_vec_unchanged` (mutants `s03`, `s05`, `s06`) |
| 4. `i > len` in the bounds check | a pointer into uninitialized spare capacity | `at_is_bounds_checked` (mutant `s04`) |
| 5. writing at `cap - 1` instead of `len` | elements overwrite each other, ASan reports overflow on growth | `hand_example` (mutant `s07`) |
| 6. reserve shrinking the block | a caller's pointer dangles | `reserve_grows_exactly_and_never_shrinks` (mutant `s08`) |
| 7. not freeing the old block | the counting allocator reports a leak on every growth | `growth_is_geometric` (mutant `s09`) |
| 8. free leaving `len` and `cap` behind | a reused vec pushes into a NULL block | `free_resets_and_vec_is_reusable` (mutant `s10`) |
| 9. no overflow check on `cap * elem` | a huge reservation allocates a few bytes | `bad_arguments` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | allocation hooks and error reporting for standalone C modules |
| Back | `lang.03` | the C you need: pointers, `sizeof`, `memcpy` |
| Forward | `rt.04` | each sequence's block table is a `tl_vec` of `uint32_t` block ids |
| Forward | `ds.02` | the Swiss table grows by the same allocate, copy, free order and the same failure rule |
| Forward | `ds.04` | heap top-k keeps its working set in the same kind of block |

If you skip this module, `ol check rt.04` stops with `rt.04 needs ds.01`; `--ref-deps` substitutes the reference.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_vec` | Rust `Vec<T>` | typed, with `realloc` in place when the allocator can extend the block; growth factor 2 | `library/alloc/src/raw_vec.rs` |
| `tl_vec_push` | C++ `std::vector::push_back` | strong exception guarantee, the C++ name for failure atomicity; factor 1.5 in MSVC, 2 in libstdc++ | `bits/vector.tcc` (`_M_realloc_insert`) |
| geometric growth | Facebook `folly::fbvector` | factor 1.5 so freed blocks can be reused by later growth, `jemalloc` size classes | `folly/FBVector.h` |
| block tables | vLLM `BlockTable` | preallocated per-request tables on the GPU, appended block by block | `vllm/v1/worker/block_table.py` |
