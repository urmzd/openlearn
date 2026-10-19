# C runtime support and arena allocator with marks (optional)

## Overview

| | |
|---|---|
| **Module** | `rt.02` · side · C · Pass 6 |
| **Solution** | [`abi.c`](../../../../course/solutions/c/src/runtime/abi.c): status, error, and allocator helpers; [`arena.c`](../../../../course/solutions/c/src/runtime/arena.c): `tl_arena_create`, `tl_arena_alloc`, `tl_arena_mark_get`, `tl_arena_reset_to`, `tl_arena_stats_get`, `tl_arena_destroy` |
| **Contract** | [`tinyllm/abi.h`](../../../../course/contracts/c/include/tinyllm/abi.h) · [`tinyllm/arena.h`](../../../../course/contracts/c/include/tinyllm/arena.h) · rules: [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Needs** | `lang.03` for C pointers, `size_t`, and compiling a standalone test binary |
| **Used by** | `L9.1`, `L9.2`, `L9.3`, `L9.4`, `L9.5`, `ds.01`, `ds.02`, `ds.04`, `rt.03`, `rt.04`: optional C exercises share runtime helpers through standalone C test binaries; together they form the optional C module group of the kernels pass |
| **Optional depth** | Hanson, *Fast allocation and deallocation of memory based on object lifetimes* (1990); Bonwick, *The Slab Allocator* (USENIX 1994); the `tcmalloc` and `jemalloc` design notes |

## Key Takeaways

- **An arena allocates by bumping an offset** and frees everything after a point at once by rewinding it: allocation is an add and a compare, and nothing is freed piece by piece.
- **Alignment is padding computed from the address**, not from the offset: each pointer is a multiple of its requested power of two, up to 4096.
- **Marks nest like a stack**: rewinding releases exactly what came after the mark, across blocks, and a mark from the future is refused.
- **Blocks are kept, not freed, on rewind**, so after warmup a mark, allocate, rewind step never calls the allocator hook: it cannot fail and costs no `malloc`.
- **Every hook failure leaves the arena unchanged and leak-free**, whichever allocation is the one that fails.

## How to read this chapter

Sections 1 to 3 build the support layer and the arena from first principles and work one block's allocations, a mark, and a rewind by hand. Section 4 fixes the interface. Section 5 walks through the solution, the support layer first and then the arena's three allocation cases, and explains why each choice is the right one. The allocator hook is what makes the arena testable: installing a counting hook with `tl_set_allocator` shows exactly when the arena calls it and lets any single call be made to fail. The helper API stays inside C processes; Rust and Python use process and file protocols.

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

`high_water` is 172 after the rewind too, `n_blocks` is 1 and `bytes_reserved` 256. A fifth allocation of 100 bytes would need offset 172 + 100 = 272 > 256, so it would go to a second block. Checking the offsets relative to the first pointer and the differences in $U$ (58, then 104) makes the example independent of where the hook put the block. The boundary of the fit test: two 64-byte requests fill a 128-byte block exactly, with no second block.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit, smoke | section 3: offsets 0, 64, 72; $U$ +58 and +104; rewind restores $U$; same address again; one 256-byte block | the definition is pinned by numbers you can check by hand |
| Every power-of-two alignment | unit | pointers for alignments 1 to 4096 after odd-sized requests; 0 means 64 | SIMD loads in L9 kernels |
| Bad requests change nothing | boundary | $n = 0$, alignment 3, alignment 8192: NULL with a message, arena unchanged; `create` with NULL out | defined behavior at the boundary |
| Nested marks | unit | inner and outer rewinds across three blocks restore the same addresses | a kernel calling a helper that uses scratch too |
| A later mark is ignored | boundary | rewinding to a mark from after the current position changes nothing and sets the slot | no double hand-out of the same bytes |
| Blocks are kept and reused | unit | after a rewind two 100-byte requests reuse the same two blocks: no new hook call | the steady state of an engine step |
| An exact fit uses the last byte | boundary | 64 + 64 bytes fill a 128-byte block | no wasted blocks |
| An oversized request gets its own block | boundary | 1000 bytes in an arena of 128-byte blocks: aligned, writable (ASan), separate | a long prompt's attention tile |
| Random operations against a model | property | 300 seeded runs of allocations, marks, rewinds: aligned, disjoint, unchanged contents; $U$ exact at marks; `high_water` the running maximum | everything above, combined |
| Zero hook calls after warmup | fault | after 3 warmup steps every hook call fails; 1000 more steps still succeed | the "zero malloc per step" claim, counted |
| Hook failure leaves the arena usable | fault | the n-th hook call fails, for every n from 0 to 39: TL_ENOMEM or NULL, unchanged stats, no leak | every constructor's failure path |
| The ABI version matches the header | unit | the support version equals the header constant | each C test binary compiles against the same interface |
| Status names, known and unknown | unit, boundary | all declared names and negative/future values | logs remain safe for invalid enum values |
| The error slot copies and truncates | boundary | caller storage can disappear; long messages stop at the buffer bound | errors do not retain invalid pointers or overflow |
| Allocation goes through the hook, aligned | unit | custom hooks count calls; returned addresses satisfy requested alignment | arenas and containers use the configured allocator |
| Installing a hook: reset, half a hook, a stack-local struct | boundary | reset, invalid hooks, and stack-local hook structs | no stale function pointers or partial hooks |
| Bad sizes and alignments, hook failure, `tl_free(NULL)` | boundary, fault | invalid inputs, failed hook, and null free | defined failure paths for every C module |

## 5. The solution, step by step

### From first principles

1. **Kernel scratch has a stack lifetime.** A kernel's tiles live exactly as long as the call; a helper's tiles live inside that. So freeing individually is wasted work: everything after a point is released together. That is a bump pointer plus a saved position (a mark).
2. **The position must name a block, not just an offset**, because scratch can outgrow one block. Hence $(b, o)$, ordered lexicographically, and a mark is a copy of it.
3. **Alignment must be computed from the address** (section 2.2). Blocks are requested at alignment at least 64 so that a fresh block satisfies any request that triggered it; in the middle of a block the padding is $\mathrm{pad}(\mathrm{base}_b + o, a)$.
4. **The steady state must make no hook call.** So rewinding keeps blocks; a full block moves to the next one already in the list when it fits; only when none fits is a new block created. Because marks nest, blocks after the current one are all free, so the one that fits can be swapped into place.
5. **`bytes_used` must be exact at every mark**, which needs the offset each earlier block reached, not its size. Keep `used` per block and `used_before` for the blocks before the current one.
6. **Every failure must leave the arena unchanged**: do every fallible step (growing the list, the new block) before writing any field.
7. **The support layer is the contract every C unit reports through**: a stable status name for any integer, a thread-local copied error message, and an allocator hook that is copied when installed and validated on every call.

### The code, annotated

The support layer first. The error slot is thread-local and copies with truncation, because the caller's message buffer may die right after the call:

```c
static _Thread_local char err_slot[TL_LAST_ERROR_CAP];

void tl_set_last_error(const char *msg) {
    if (msg == NULL) msg = "";
    size_t n = strlen(msg);
    if (n > TL_LAST_ERROR_CAP - 1) n = TL_LAST_ERROR_CAP - 1;  /* never overflow */
    memcpy(err_slot, msg, n); /* copy: the caller's buffer may die right after */
    err_slot[n] = '\0';
}
```

`tl_status_str` is a `switch` with a `default` of `"TL_UNKNOWN"`, not an array indexed by the status: the value comes from the caller and may be negative or from a newer ABI. The hook is a struct copied on install, so a stack-local `tl_allocator` is safe, and a half-filled hook is refused:

```c
static tl_allocator hook = {default_alloc, default_free, NULL};

tl_status tl_set_allocator(const tl_allocator *a) {
    if (a == NULL) {                        /* NULL restores the default */
        hook.alloc = default_alloc; hook.free = default_free; hook.user = NULL;
        return TL_OK;
    }
    if (a->alloc == NULL || a->free == NULL) {
        tl_set_last_error("tl_set_allocator: alloc and free must both be set");
        return TL_EINVAL;
    }
    hook = *a; /* copy the struct: the caller's may be a stack local */
    return TL_OK;
}

void *tl_alloc(size_t n, size_t align) {
    if (n == 0) { tl_set_last_error("tl_alloc: size 0"); return NULL; }
    if (align == 0 || (align & (align - 1)) != 0) {     /* powers of two only */
        tl_set_last_error("tl_alloc: align must be a power of two");
        return NULL;
    }
    void *p = hook.alloc(hook.user, n, align);
    if (p == NULL) tl_set_last_error("tl_alloc: the allocator hook returned NULL");
    return p;
}
```

The default hook rounds the alignment up to `sizeof(void *)` and the size up to a multiple of the alignment, as C11's `aligned_alloc` requires.

The arena's state is exactly the quantities of section 2:

```c
typedef struct {
    unsigned char *base; /* from tl_alloc, aligned to at least 64 */
    size_t size;         /* bytes in the block */
    size_t used;         /* the offset reached when the arena moved past it */
} tl_arena_block;

struct tl_arena {
    size_t block_bytes;      /* default block size */
    tl_arena_block *blocks;  /* blocks[0 .. n_blocks), owned */
    size_t n_blocks, cap;    /* used and allocated entries of `blocks` */
    size_t cur, off;         /* the position; cur == n_blocks before the first block */
    size_t used_before;      /* sum of blocks[i].used for i < cur */
    size_t high_water;
};

/* Bytes needed to move address p up to a multiple of align (a power of two). */
static size_t pad_for(const unsigned char *p, size_t align) {
    uintptr_t a = (uintptr_t)p;
    return (size_t)((align - (a & (align - 1))) & (align - 1));   /* the mask is p mod align */
}
```

`tl_arena_create` allocates only the struct; the first block comes with the first allocation, so an unused arena costs nothing.

`tl_arena_alloc` in its three cases, after validating $n > 0$ and the alignment (0 becomes 64; a non-power of two or anything above 4096 is refused):

```c
/* 1. The current block. */
if (a->cur < a->n_blocks) {
    tl_arena_block *b = &a->blocks[a->cur];
    size_t pad = pad_for(b->base + a->off, align);       /* from the address */
    /* Two subtractions instead of off + pad + n <= size: no overflow, and
     * an exact fit (== size) is accepted. */
    if (pad <= b->size - a->off && n <= b->size - a->off - pad) {
        void *p = b->base + a->off + pad;
        a->off += pad + n;                               /* padding counts as used */
        size_t used = a->used_before + a->off;
        if (used > a->high_water) a->high_water = used;
        return p;
    }
}

/* 2. A later, free block that fits: swap it into position next. Every
 *    block after cur is free (marks nest), so their order is ours. */
size_t next = a->cur < a->n_blocks ? a->cur + 1 : 0;
size_t found = a->n_blocks;
for (size_t i = next; i < a->n_blocks; i++) {
    if (pad_for(a->blocks[i].base, align) + n <= a->blocks[i].size) { found = i; break; }
}
if (found == a->n_blocks) {
    /* 3. A new block from the hook, inserted at position next. An oversized
     *    request gets its own block, rounded up to a multiple of 64. */
    size_t size = n > a->block_bytes ? (n + 63) & ~(size_t)63 : a->block_bytes;
    size_t balign = align > TL_ARENA_DEFAULT_ALIGN ? align : TL_ARENA_DEFAULT_ALIGN;
    if (!grow_list(a)) { /* ... error slot ... */ return NULL; }   /* nothing changed yet */
    unsigned char *base = tl_alloc(size, balign);
    if (base == NULL) { /* ... error slot ... */ return NULL; }    /* still nothing changed */
    a->blocks[a->n_blocks] = (tl_arena_block){base, size, 0};
    found = a->n_blocks++;
}
if (found != next) {                                  /* bring it to position next */
    tl_arena_block t = a->blocks[next];
    a->blocks[next] = a->blocks[found];
    a->blocks[found] = t;
}

/* Leave the current block: what it holds now counts as used. */
if (a->cur < a->n_blocks && next == a->cur + 1) {
    a->blocks[a->cur].used = a->off;
    a->used_before += a->off;
}
a->cur = next;
tl_arena_block *b = &a->blocks[next];
size_t pad = pad_for(b->base, align);
a->off = pad + n;
// ... update high_water ...
return b->base + pad;
```

`grow_list` doubles the block list (starting at 8) into a fresh allocation and only then frees the old one, so a failure there leaves the list as it was. On the worked example, `alloc(4, 64)` at offset 10 of a 64-aligned block pads by $\mathrm{pad}(10, 64) = 54$ and returns offset 64; `bytes_used` grows by $54 + 4 = 58$.

Rewinding checks that the mark is not later than the position, then restores the position and recomputes `used_before` from the recorded per-block offsets:

```c
void tl_arena_reset_to(tl_arena *a, tl_arena_mark m) {
    if (a == NULL) return;
    size_t mb = m.block, mo = m.offset;
    if (a->cur >= a->n_blocks) {
        /* No block yet: only the empty mark (0, 0) is valid, and it changes nothing. */
        if (mb != 0 || mo != 0) tl_set_last_error("tl_arena_reset_to: mark is later than the position");
        return;
    }
    if (mb > a->cur || (mb == a->cur && mo > a->off)) {   /* lexicographic: later */
        tl_set_last_error("tl_arena_reset_to: mark is later than the position");
        return;
    }
    size_t before = 0;
    for (size_t i = 0; i < mb; i++) before += a->blocks[i].used;
    a->used_before = before;
    a->cur = mb;                 /* the block, not just the offset */
    a->off = mo;
    /* No block is freed: they stay in the list for reuse. */
}
```

`tl_arena_stats_get` reports `used_before + off`, the sum of block sizes, `high_water`, and the block count; `tl_arena_destroy` frees every block, the list, and the struct through the hook.

Full solution: [`arena.c`](../../../../course/solutions/c/src/runtime/arena.c) and [`abi.c`](../../../../course/solutions/c/src/runtime/abi.c).

### Why this solution

- **Cost.** The fast path is a mask, an add, and two compares. Moving to a reused block scans the later blocks, $O(\text{blocks})$, which is a handful. A rewind is $O(\text{blocks before the mark})$ to recompute `used_before`; storing a running prefix sum would make it $O(1)$ at the cost of more bookkeeping, and the block count is small.
- **Against `malloc` per tile.** It costs a lock or a thread cache lookup per call, can fail in the middle of a kernel, and fragments. The arena has one possible failure point per growth and none after warmup.
- **Against one huge preallocated block.** It needs the peak known in advance. The arena grows to the peak once and `high_water` reports it, so the engine can size a single block next time.
- **Against freeing blocks on rewind.** It returns memory sooner, but every step would pay `malloc` and could fail; keeping them is what makes "zero hook calls after warmup" true.
- **Tradeoff accepted.** Memory is held at the high-water mark until destroy, and the arena is single-threaded (one per worker).

### Explain it back

1. Why does `pad_for` take the address `base + off` instead of `off`?
   <details><summary>Answer</summary>Alignment is a property of the address the caller loads from. An offset of 0 is a multiple of every alignment, but a block's base may be only 64-aligned; a 4096-aligned request at offset 0 of such a block would be misaligned.</details>
2. Why are blocks after `cur` safe to reorder?
   <details><summary>Answer</summary>Marks nest like a stack, so no live allocation exists past the current position; every block after `cur` holds nothing. Swapping one that fits into position `next` does not move any live data.</details>
3. Why does the fit test use `pad <= size - off && n <= size - off - pad` rather than `off + pad + n <= size`?
   <details><summary>Answer</summary>`off + pad + n` can overflow `size_t` for a huge $n$ and wrap to a small number that passes. The subtractions cannot underflow because `off <= size` and the first test guards the second, and the `<=` accepts an exact fit.</details>
4. What would go wrong if `tl_arena_alloc` set `a->cur = next` before calling the hook?
   <details><summary>Answer</summary>If the hook then failed, the position would point at a block that does not exist (or the wrong one), so later allocations or rewinds would corrupt memory. All fallible calls happen first so a failure changes nothing.</details>
5. Why does `tl_set_allocator` copy the struct instead of storing the pointer?
   <details><summary>Answer</summary>Callers often build the hook as a stack local in a test or setup function. Storing its address would leave a dangling pointer once that function returns.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. padding left out of `bytes_used` | $U$ too small, `high_water` under-sizes the engine's arena | `off += pad + n` |
| 2. aligning the offset, not the address | misaligned pointers whenever a block starts off a 64-byte boundary | `pad_for(b->base + a->off, align)` |
| 3. rewinding only the offset, not the block | allocations after a cross-block rewind land in the wrong block | restores `cur` and recomputes `used_before` |
| 4. accepting a later mark | bytes handed out twice | compares the mark lexicographically with the position and ignores a later one |
| 5. freeing blocks on rewind, or never reusing them | a hook call every step: `malloc` cost and a failure point per kernel call | rewind keeps every block; case 2 reuses a later block that fits |
| 6. an oversized request squeezed into a normal block | a heap overflow (ASan) | a request above `block_bytes` gets a block of its own size, rounded to 64 |
| 7. changing fields before the hook call succeeds | a failed alloc corrupts the arena or leaks the block list | `grow_list` and the block allocation both happen before any field is written |
| 8. an off-by-one in the fit test | a new block for a request that fits exactly | `n <= size - off - pad` accepts an exact fit |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.03` | C types, pointers, and compilation fundamentals |
| Forward | `L9.3` | `tl_flash_attn_fwd_f32` takes a `tl_arena *scratch`: query and key tiles, running maxima, rewound per call |
| Forward | `L9.1`, `L9.2`, `L9.4`, `L9.5` | Optional C exercises use the shared error and allocator helpers in standalone C test binaries |
| Forward | `ds.01`, `ds.02`, `ds.04`, `rt.03`, `rt.04` | Optional C exercises reuse the standalone runtime helpers; none are linked into Python or Rust |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| arena with marks | ggml's `ggml_context` and `ggml_gallocr` | a graph allocator that plans tensor lifetimes ahead and reuses memory between tensors that are never live together | `ggml/src/ggml-alloc.c` |
| per-thread arenas | jemalloc arenas, tcmalloc per-CPU caches | many arenas chosen by thread or CPU, size classes, returning memory to the OS | jemalloc `doc/jemalloc.xml.in`; tcmalloc `docs/design.md` |
| rewind-per-step | PyTorch's CUDA caching allocator | keeps freed blocks per stream and size bucket for reuse across iterations | `c10/cuda/CUDACachingAllocator.cpp` |
| `high_water` | vLLM's memory profiling before KV allocation | runs a dummy forward pass, measures the peak, and gives the rest to the KV cache | vLLM `worker/` (`determine_num_available_blocks`) |
