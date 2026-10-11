<!-- ol:module rt.02 -->
# C runtime support and arena allocator with marks (optional)

## Overview

| | |
|---|---|
| **Module** | `rt.02` · side · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/runtime/abi.c`: status, error, and allocator helpers; `arena.c`: `tl_arena_create`, `tl_arena_alloc`, `tl_arena_mark_get`, `tl_arena_reset_to`, `tl_arena_stats_get`, `tl_arena_destroy` |
| **Contract** | [`tinyllm/abi.h`](../../../course/contracts/c/include/tinyllm/abi.h) · [`tinyllm/arena.h`](../../../course/contracts/c/include/tinyllm/arena.h) · rules: [`c/ABI.md`](../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/rt.02/test_abi.c` and `test_arena.c`, standalone C under sanitizers |
| **Needs** | `lang.03` for C pointers, `size_t`, and compiling a standalone test binary |
| **Used by** | `L9.1`, `L9.2`, `L9.3`, `L9.4`, `L9.5`, `ds.01`, `ds.02`, `ds.04`, `rt.03`, `rt.04`: optional C exercises share runtime helpers through standalone C test binaries |
| **Milestone** | `MS-L9`, the optional C module group |
| **Optional depth** | Hanson, *Fast allocation and deallocation of memory based on object lifetimes* (1990); Bonwick, *The Slab Allocator* (USENIX 1994); the `tcmalloc` and `jemalloc` design notes |

## Key Takeaways

- **An arena allocates by bumping an offset** and frees everything after a point at once by rewinding it: allocation is an add and a compare, and nothing is freed piece by piece (`hand_example`).
- **Alignment is padding computed from the address**, not from the offset: each pointer is a multiple of its requested power of two, up to 4096 (`every_power_of_two_alignment`).
- **Marks nest like a stack**: rewinding releases exactly what came after the mark, across blocks, and a mark from the future is refused (`nested_marks`, `a_later_mark_is_ignored`).
- **Blocks are kept, not freed, on rewind**, so after warmup a mark, allocate, rewind step never calls the allocator hook: it cannot fail and costs no `malloc` (`zero_hook_calls_after_warmup`).
- **Every hook failure leaves the arena unchanged and leak-free**, checked by failing the n-th allocation for every n (`hook_failure_leaves_the_arena_usable`).

## How to work this chapter

```bash
ol start rt.02              # stubs abi.c and arena.c into your repo
ol tests rt.02              # read the test catalog first
ol check rt.02              # exit code is the verdict
ol check rt.02 --ref-deps   # only if you skipped a listed prerequisite
ol diff  rt.02              # after passing: your code against the reference
```

The counting allocator of `ol_test.h` is installed through `tl_set_allocator`: a test that ends with a block still live fails as a leak, and `ol_alloc_fail_after(n)` makes the n-th allocation fail. The helper API stays inside C test processes; Rust and Python use process and file protocols.

---

## 1. Why now

Part 9's optional C kernels need scratch memory that is not an input or an output: FlashAttention (`L9.3`) keeps tiles of scores and running maxima, and online softmax (`L9.2`) keeps partial sums. `malloc` and `free` for every tile would add allocator overhead and another failure point in each kernel call. This module begins with a small C support layer: fixed status values, a thread-local error message, and replaceable allocation hooks. The arena then uses those hooks to reserve blocks once, bump an offset for each request, and rewind when a kernel finishes. None of these C interfaces is loaded by Python or linked into the Rust engine.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $B$ | the block size given to `tl_arena_create` (0 means 1 MiB) | `size_t` |
| $\mathrm{base}_b$ | the address of block $b$ | pointer |
| $(b, o)$ | the position: the current block's index and the offset of its first free byte | `size_t`, `size_t` |
| $n$ | the size of a request, in bytes, $n > 0$ | `size_t` |
| $a$ | the requested alignment, a power of two $\le 4096$ (0 means 64) | `size_t` |
| $\mathrm{pad}(p, a)$ | bytes from address $p$ up to the next multiple of $a$: $(a - (p \bmod a)) \bmod a$ | `size_t` |
| $U$ | `bytes_used`: allocated bytes including padding | `size_t` |

The shared C support functions have simple contracts too. `tl_status_str` returns a stable name for known values and `TL_UNKNOWN` for any other integer. `tl_set_last_error` copies and truncates a message into a thread-local slot. `tl_alloc` rejects zero sizes and non-power-of-two alignments before calling the installed hook; `tl_free(NULL)` does nothing. These helpers make standalone C failures observable and make allocator behavior countable in tests.

### 2.1 Bump allocation

An arena owns a list of blocks obtained from the hook. Within the current block, an allocation of $n$ bytes aligned to $a$ is

$$p = \mathrm{base}_b + o + \mathrm{pad}(\mathrm{base}_b + o,\ a), \qquad o \leftarrow o + \mathrm{pad} + n,$$

provided $o + \mathrm{pad} + n$ is at most the block's size. That is the whole fast path: one remainder (a mask, since $a$ is a power of two: $p \bmod a = p \mathbin{\&} (a - 1)$), one add, one compare. When the request does not fit, the arena moves to the next block (2.4). A request larger than $B$ gets a block of its own, rounded up to a multiple of 64.

### 2.2 Alignment comes from the address

SIMD loads want their address to be a multiple of 16, 32, or 64 bytes, and a cache line is 64. The padding must therefore be computed from the **address** $\mathrm{base}_b + o$, not from the offset $o$: an offset of 0 is a multiple of everything, but the block itself might start at an address that is only a multiple of 16. Blocks are requested from the hook with alignment $\max(64, a)$, so a new block's first byte is aligned for any request that caused it.

### 2.3 Marks and rewinding

A **mark** is a copy of the position $(b, o)$. `tl_arena_reset_to(mark)` sets the position back to it: every allocation made after the mark is released at once, and the ones before it stay valid. Positions are ordered lexicographically (first by block, then by offset), and marks must nest like a stack: a kernel takes a mark, calls a helper that takes its own mark and rewinds to it, then rewinds to its own. Rewinding to a mark that is **later** than the current position would declare memory free that was handed out after the mark was taken and later rewound past; the contract ignores such a mark and sets the error slot.

`bytes_used` must return exactly to its value at the mark. Track $U$ as "bytes used in all blocks before the current one" plus $o$, and record, for each block the arena moves past, the offset it reached there. On rewind to $(b', o')$ recompute the first term as the sum over blocks before $b'$.

### 2.4 Keep the blocks

On rewind the blocks after the mark stay in the list: they are free, and the next allocations reuse them. When the current block is full, the arena looks among the later blocks for one large enough (they are all free, because marks nest, so their order can be changed: the one found is swapped into the next position), and only if none fits does it ask the hook for a new one, inserted at the next position. After the first step of an engine has grown the list to its working size, every later step allocates without touching the hook. `high_water` records the largest $U$ ever reached, which is how the engine sizes its arena; `bytes_reserved` is the sum of block sizes.

### 2.5 Failure leaves no trace

Two things can fail: the hook returning NULL for a new block, and the hook returning NULL for a larger block list. Check both **before** changing any field, so a failed `tl_arena_alloc` returns NULL with the error slot set and leaves the arena exactly as it was. `tl_arena_destroy` frees every block, the list, and the arena through the hook.

## 3. Worked example by hand

A 256-byte block whose first byte is at an address that is a multiple of 64 (the hook was asked for that).

| Call | Padding | Returned offset | Position after | $U$ |
|---|---|---|---|---|
| `alloc(10, 0)` (0 means 64) | 0 | 0 | (0, 10) | 10 |
| `alloc(4, 64)` | $\mathrm{pad}(10, 64) = 54$ | 64 | (0, 68) | 68 |
| `mark_get` | | | mark = (0, 68) | |
| `alloc(100, 8)` | $\mathrm{pad}(68, 8) = 4$ | 72 | (0, 172) | 172 |
| `reset_to(mark)` | | | (0, 68) | 68 |
| `alloc(100, 8)` | 4 | 72, the same address as before | (0, 172) | 172 |

`high_water` is 172 after the rewind too, `n_blocks` is 1 and `bytes_reserved` 256. A fifth allocation of 100 bytes would need offset 172 + 100 = 272 > 256, so it would go to a second block. `hand_example` makes exactly these calls and checks the offsets relative to the first pointer and the differences in $U$ (58, then 104), so it does not depend on where the hook put the block. `exact_fit_uses_the_last_byte` checks the boundary of the fit test: two 64-byte requests fill a 128-byte block exactly, with no second block.

## 4. The interface

```c
typedef struct tl_arena tl_arena;
typedef struct { size_t offset; size_t block; } tl_arena_mark;
typedef struct { size_t bytes_used, bytes_reserved, high_water, n_blocks; } tl_arena_stats;
tl_status     tl_arena_create(size_t block_bytes, tl_arena **out);   /* no block yet; TL_EINVAL, TL_ENOMEM */
void         *tl_arena_alloc(tl_arena *a, size_t n, size_t align);   /* NULL + error slot: n == 0, bad align, hook failed */
tl_arena_mark tl_arena_mark_get(const tl_arena *a);
void          tl_arena_reset_to(tl_arena *a, tl_arena_mark m);      /* a later mark: ignored, error slot set */
void          tl_arena_stats_get(const tl_arena *a, tl_arena_stats *out);
void          tl_arena_destroy(tl_arena *a);                         /* NULL is a no-op */
```

The arena is not thread-safe: one arena per thread (L9.3 gives each worker of the rt.03 pool its own).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example` | unit, smoke | section 3: offsets 0, 64, 72; $U$ +58 and +104; rewind restores $U$; same address again; one 256-byte block | you and the test agree on the definition |
| `every_power_of_two_alignment` | unit | pointers for alignments 1 to 4096 after odd-sized requests; 0 means 64 | SIMD loads in L9 kernels |
| `bad_requests_change_nothing` | boundary | $n = 0$, alignment 3, alignment 8192: NULL with a message, arena unchanged; `create` with NULL out | defined behavior at the boundary |
| `nested_marks` | unit | inner and outer rewinds across three blocks restore the same addresses | a kernel calling a helper that uses scratch too |
| `a_later_mark_is_ignored` | boundary | rewinding to a mark from after the current position changes nothing and sets the slot | no double hand-out of the same bytes |
| `blocks_are_kept_and_reused` | unit | after a rewind two 100-byte requests reuse the same two blocks: no new hook call | the steady state of an engine step |
| `exact_fit_uses_the_last_byte` | boundary | 64 + 64 bytes fill a 128-byte block | no wasted blocks |
| `oversized_request_gets_its_own_block` | boundary | 1000 bytes in an arena of 128-byte blocks: aligned, writable (ASan), separate | a long prompt's attention tile |
| `random_operations_against_a_model` | property | 300 seeded runs of allocations, marks, rewinds: aligned, disjoint, unchanged contents; $U$ exact at marks; `high_water` the running maximum | everything above, combined |
| `zero_hook_calls_after_warmup` | fault | after 3 warmup steps every hook call fails; 1000 more steps still succeed | the "zero malloc per step" claim, counted |
| `hook_failure_leaves_the_arena_usable` | fault | the n-th hook call fails, for n = 0 to 39: TL_ENOMEM or NULL, unchanged stats, no leak | every constructor's failure path |
| `abi_version_matches_header` | unit | the support version equals the header constant | each C test binary compiles against the same interface |
| `status_str_names_every_code`, `status_str_unknown_codes` | unit, boundary | all declared names and negative/future values | logs remain safe for invalid enum values |
| `error_slot_copies_and_truncates` | boundary | caller storage can disappear; long messages stop at the buffer bound | errors do not retain invalid pointers or overflow |
| `alloc_goes_through_the_hook`, `alloc_alignment_by_hand` | unit | custom hooks count calls; returned addresses satisfy requested alignment | arenas and containers use the configured allocator |
| `set_allocator_null_restores_the_default`, `set_allocator_rejects_half_a_hook`, `set_allocator_copies_the_struct` | boundary | reset, invalid hooks, and stack-local hook structs | no stale function pointers or partial hooks |
| `alloc_rejects_zero_and_bad_alignment`, `alloc_failure_sets_the_error`, `free_null_is_a_no_op` | boundary, fault | invalid inputs, failed hook, and null free | defined failure paths for every C module |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. padding left out of `bytes_used` | $U$ too small, `high_water` under-sizes the engine's arena | `hand_example` (mutant `s01`) |
| 2. aligning the offset, not the address | misaligned pointers whenever a block starts off a 64-byte boundary | `every_power_of_two_alignment` (mutants `s02`, `s03`) |
| 3. rewinding only the offset, not the block | allocations after a cross-block rewind land in the wrong block | `nested_marks` (mutant `s05`) |
| 4. accepting a later mark | bytes handed out twice | `a_later_mark_is_ignored` (mutant `s07`) |
| 5. freeing blocks on rewind, or never reusing them | a hook call every step: `malloc` cost and a failure point per kernel call | `zero_hook_calls_after_warmup` (mutants `s08`, `s09`) |
| 6. an oversized request squeezed into a normal block | a heap overflow (ASan) | `oversized_request_gets_its_own_block` (mutant `s10`) |
| 7. changing fields before the hook call succeeds | a failed alloc corrupts the arena or leaks the block list | `hook_failure_leaves_the_arena_usable` (mutants `s12`, `s13`) |
| 8. an off-by-one in the fit test | a new block for a request that fits exactly | `exact_fit_uses_the_last_byte` (mutant `m001`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.03` | C types, pointers, and compilation fundamentals |
| Forward | `L9.3` | `tl_flash_attn_fwd_f32` takes a `tl_arena *scratch`: query and key tiles, running maxima, rewound per call |
| Forward | `L9.1`, `L9.2`, `L9.4`, `L9.5` | Optional C exercises use the shared error and allocator helpers in standalone C test binaries |
| Forward | `ds.01`, `ds.02`, `ds.04`, `rt.03`, `rt.04` | Optional C exercises reuse the standalone runtime helpers; none are linked into Python or Rust |

If you skip this module, `ol check L9.3` stops with `BLOCKED ... needs rt.02`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| arena with marks | ggml's `ggml_context` and `ggml_gallocr` | a graph allocator that plans tensor lifetimes ahead and reuses memory between tensors that are never live together | `ggml/src/ggml-alloc.c` |
| per-thread arenas | jemalloc arenas, tcmalloc per-CPU caches | many arenas chosen by thread or CPU, size classes, returning memory to the OS | jemalloc `doc/jemalloc.xml.in`; tcmalloc `docs/design.md` |
| rewind-per-step | PyTorch's CUDA caching allocator | keeps freed blocks per stream and size bucket for reuse across iterations | `c10/cuda/CUDACachingAllocator.cpp` |
| `high_water` | vLLM's memory profiling before KV allocation | runs a dummy forward pass, measures the peak, and gives the rest to the KV cache | vLLM `worker/` (`determine_num_available_blocks`) |
