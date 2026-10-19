# C runtime support and an arena allocator with marks

The kernels in this chapter need memory that is neither an input nor an output. FlashAttention keeps tiles of scores and running maxima; online softmax keeps partial sums; a quantized matmul may unpack a group of weights before using it. Calling `malloc` and `free` for every tile would add allocator overhead to each kernel call and, worse, a new place where the call can fail halfway through. This section builds the small piece of infrastructure that removes both problems: a C support layer with fixed status values, a thread-local error message, and replaceable allocation hooks, and on top of it an **arena** that reserves blocks once, bumps an offset for each request, and rewinds when a kernel finishes.

Everything in this chapter is optional. None of these C interfaces is loaded by Python or linked into the Rust engine; each C unit compiles into its own standalone test program, and Rust and Python talk to C only through checked-in files and process protocols. The rules every C unit follows are in [`c/ABI.md`](../../../../course/contracts/c/ABI.md), and the headers you implement against are [`tinyllm/abi.h`](../../../../course/contracts/c/include/tinyllm/abi.h) and [`tinyllm/arena.h`](../../../../course/contracts/c/include/tinyllm/arena.h). The reference solutions are [`abi.c`](../../../../course/solutions/c/src/runtime/abi.c) (status, error, and allocator helpers) and [`arena.c`](../../../../course/solutions/c/src/runtime/arena.c) (`tl_arena_create`, `tl_arena_alloc`, `tl_arena_mark_get`, `tl_arena_reset_to`, `tl_arena_stats_get`, `tl_arena_destroy`).

This section assumes you are comfortable with C pointers, `size_t`, and compiling a standalone test binary, as covered in [C: memory, pointers, structs, headers, and the C11 toolchain](../../programming-languages/02-language-and-tool-primers/03-c.md) in *Programming Languages*. The allocator hook is what makes the arena testable: installing a counting hook with `tl_set_allocator` shows exactly when the arena calls it and lets any single call be made to fail, and you will lean on that throughout.

> **Learning objectives**
> - Implement a bump allocator whose fast path is one mask, one add, and two compares.
> - Compute alignment padding from the address rather than the offset, for any power of two up to 4096.
> - Implement nested marks that rewind across blocks, keep `bytes_used` exact, and refuse a mark from the future.
> - Explain why keeping blocks on rewind makes a warmed-up step call the allocator zero times.
> - Order the fallible steps of an allocation so that any hook failure leaves the arena unchanged and leak-free.

## The support layer

Before the arena there is a thin layer every C unit in this book reports through, and its contracts are deliberately simple. `tl_status_str` returns a stable name for each known status value and `TL_UNKNOWN` for any other integer. `tl_set_last_error` copies a message, truncating it if needed, into a thread-local slot. `tl_alloc` rejects a zero size and any alignment that is not a power of two before it calls the installed hook, and `tl_free(NULL)` does nothing. Together these make standalone C failures observable (a test can read the message) and make allocator behavior countable (a test can install a hook that counts or fails).

## Bump allocation

The arena itself needs a few quantities, which the rest of the section refers to by these names:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $B$ | the block size given to `tl_arena_create` (0 means 1 MiB) | `size_t` |
| $\mathrm{base}_b$ | the address of block $b$ | pointer |
| $(b, o)$ | the position: the current block's index and the offset of its first free byte | `size_t`, `size_t` |
| $n$ | the size of a request, in bytes, $n > 0$ | `size_t` |
| $a$ | the requested alignment, a power of two $\le 4096$ (0 means 64) | `size_t` |
| $\mathrm{pad}(p, a)$ | bytes from address $p$ up to the next multiple of $a$: $(a - (p \bmod a)) \bmod a$ | `size_t` |
| $U$ | `bytes_used`: allocated bytes including padding | `size_t` |

An arena owns a list of blocks obtained from the hook. Within the current block, an allocation of $n$ bytes aligned to $a$ is

$$p = \mathrm{base}_b + o + \mathrm{pad}(\mathrm{base}_b + o,\ a), \qquad o \leftarrow o + \mathrm{pad} + n,$$

provided $o + \mathrm{pad} + n$ is at most the block's size. That is the whole fast path: one remainder (a mask, since $a$ is a power of two: $p \bmod a = p \mathbin{\&} (a - 1)$), one add, one compare. When the request does not fit, the arena moves to the next block, as described below. A request larger than $B$ gets a block of its own, rounded up to a multiple of 64.

## Alignment comes from the address

SIMD loads want their address to be a multiple of 16, 32, or 64 bytes, and a cache line is 64. The padding must therefore be computed from the **address** $\mathrm{base}_b + o$, not from the offset $o$. An offset of 0 is a multiple of everything, but the block itself might start at an address that is only a multiple of 16. To make a fresh block always usable, the arena requests blocks from the hook with alignment $\max(64, a)$, so a new block's first byte is aligned for whatever request caused it to be created.

## Marks and rewinding

Allocation is only half of the arena; the other half is releasing memory in bulk. A **mark** is a copy of the position $(b, o)$. `tl_arena_reset_to(mark)` sets the position back to it: every allocation made after the mark is released at once, and the ones before it stay valid.

Positions are ordered lexicographically, first by block and then by offset, and marks must nest like a stack: a kernel takes a mark, calls a helper that takes its own mark and rewinds to it, then rewinds to its own. Rewinding to a mark that is **later** than the current position would declare free some memory that was handed out after the mark was taken and then rewound past. That is how the same bytes get handed out twice, so the contract ignores such a mark and sets the error slot.

`bytes_used` must return exactly to its value at the mark. The way to get that is to track $U$ as "bytes used in all blocks before the current one" plus $o$, and to record, for each block the arena moves past, the offset it had reached there. On a rewind to $(b', o')$ you recompute the first term as the sum over blocks before $b'$.

## Keep the blocks

On rewind the blocks after the mark stay in the list. They are free, and the next allocations reuse them. When the current block is full, the arena looks among the later blocks for one large enough. Because marks nest, all of those blocks are free, so their order can be changed: the one found is swapped into the next position. Only if none fits does the arena ask the hook for a new block, inserted at the next position.

The payoff is the steady state. After the first step of an engine has grown the list to its working size, every later step allocates without touching the hook. Two statistics make this measurable: `high_water` records the largest $U$ ever reached, which is how the engine sizes its arena, and `bytes_reserved` is the sum of block sizes.

## Failure leaves no trace

Two things can fail during an allocation: the hook returning NULL for a new block, and the hook returning NULL for a larger block list. The arena checks both **before** changing any field, so a failed `tl_arena_alloc` returns NULL with the error slot set and leaves the arena exactly as it was. `tl_arena_destroy` frees every block, the list, and the arena itself through the hook.

## Working it by hand

The rules are easier to trust once you have pushed numbers through them. Take a 256-byte block whose first byte is at an address that is a multiple of 64 (the hook was asked for that), and run this sequence:

| Call | Padding | Returned offset | Position after | $U$ |
|---|---|---|---|---|
| `alloc(10, 0)` (0 means 64) | 0 | 0 | (0, 10) | 10 |
| `alloc(4, 64)` | $\mathrm{pad}(10, 64) = 54$ | 64 | (0, 68) | 68 |
| `mark_get` | | | mark = (0, 68) | |
| `alloc(100, 8)` | $\mathrm{pad}(68, 8) = 4$ | 72 | (0, 172) | 172 |
| `reset_to(mark)` | | | (0, 68) | 68 |
| `alloc(100, 8)` | 4 | 72, the same address as before | (0, 172) | 172 |

After the rewind `high_water` is still 172, `n_blocks` is 1, and `bytes_reserved` is 256. A fifth allocation of 100 bytes would need offset 172 + 100 = 272 > 256, so it would go to a second block.

Notice that the example checks offsets relative to the first pointer and differences in $U$ (58, then 104), which makes it independent of where the hook actually put the block. The boundary of the fit test is worth one more check: two 64-byte requests fill a 128-byte block exactly, with no second block.

## The interface

With the behavior pinned down, the header is short:

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

The arena is not thread-safe. The rule is one arena per thread: [FlashAttention forward in C](06-flash-attention-forward.md) gives each worker of the [thread pool](03-thread-pool-and-parallel-for.md) its own.

## Building it step by step

Start from the lifetime of the memory, because everything else follows from it. A kernel's tiles live exactly as long as the call, and a helper's tiles live inside that. Freeing individually is therefore wasted work: everything after a point is released together. That is a bump pointer plus a saved position, which is to say a mark.

The position must name a block, not just an offset, because scratch can outgrow one block. Hence $(b, o)$, ordered lexicographically, with a mark being a copy of it. Alignment must be computed from the address, as you saw above, and blocks are requested at alignment at least 64 so that a fresh block satisfies any request that triggered it; in the middle of a block the padding is $\mathrm{pad}(\mathrm{base}_b + o, a)$.

The steady state must make no hook call. So rewinding keeps blocks; a full block moves on to the next one already in the list when it fits; and only when none fits is a new block created. Because marks nest, blocks after the current one are all free, so the one that fits can be swapped into place. `bytes_used` must be exact at every mark, which needs the offset each earlier block reached, not its size, so the arena keeps `used` per block and `used_before` for the blocks before the current one. And every failure must leave the arena unchanged, which means doing every fallible step (growing the list, getting the new block) before writing any field.

Finally, the support layer is the contract every C unit reports through: a stable status name for any integer, a thread-local copied error message, and an allocator hook that is copied when installed and validated on every call. The code starts there.

### The support layer in code

The error slot is thread-local and copies with truncation, because the caller's message buffer may die right after the call:

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

`tl_status_str` is a `switch` with a `default` of `"TL_UNKNOWN"`, not an array indexed by the status, because the value comes from the caller and may be negative or from a newer ABI. The hook is a struct copied on install, so a stack-local `tl_allocator` is safe, and a half-filled hook is refused:

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

### The arena's state

The arena's state is exactly the quantities introduced earlier:

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

`tl_arena_create` allocates only the struct. The first block comes with the first allocation, so an unused arena costs nothing.

### Allocation in three cases

`tl_arena_alloc` first validates $n > 0$ and the alignment (0 becomes 64; a non-power of two or anything above 4096 is refused). Then it tries three cases in order: the current block, a later free block that fits, and a new block from the hook.

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

`grow_list` doubles the block list (starting at 8) into a fresh allocation and only then frees the old one, so a failure there leaves the list as it was. You can replay the worked example against this code: `alloc(4, 64)` at offset 10 of a 64-aligned block pads by $\mathrm{pad}(10, 64) = 54$ and returns offset 64, and `bytes_used` grows by $54 + 4 = 58$.

### Rewinding

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

`tl_arena_stats_get` reports `used_before + off`, the sum of block sizes, `high_water`, and the block count; `tl_arena_destroy` frees every block, the list, and the struct through the hook. The full solution is in [`arena.c`](../../../../course/solutions/c/src/runtime/arena.c) and [`abi.c`](../../../../course/solutions/c/src/runtime/abi.c).

### Why this design

The fast path costs a mask, an add, and two compares. Moving to a reused block scans the later blocks, which is $O(\text{blocks})$ but only a handful. A rewind is $O(\text{blocks before the mark})$ to recompute `used_before`; storing a running prefix sum would make it $O(1)$ at the cost of more bookkeeping, and with so few blocks it is not worth it.

Compare the alternatives. A `malloc` per tile costs a lock or a thread-cache lookup per call, can fail in the middle of a kernel, and fragments the heap; the arena has one possible failure point per growth and none after warmup. One huge preallocated block needs the peak known in advance, whereas the arena grows to the peak once and `high_water` reports it, so the engine can size a single block next time. Freeing blocks on rewind returns memory sooner, but every step would pay `malloc` and could fail; keeping them is precisely what makes "zero hook calls after warmup" true. The tradeoff you accept is that memory is held at the high-water mark until destroy, and that the arena is single-threaded, one per worker.

## What can go wrong

A correct implementation has to survive a fairly long list of cases, from the hand example through fault injection. Each one exists because something downstream depends on it:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit, smoke | offsets 0, 64, 72; $U$ +58 and +104; rewind restores $U$; same address again; one 256-byte block | the definition is pinned by numbers you can check by hand |
| Every power-of-two alignment | unit | pointers for alignments 1 to 4096 after odd-sized requests; 0 means 64 | SIMD loads in the kernels of this chapter |
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
| Bad sizes and alignments, hook failure, `tl_free(NULL)` | boundary, fault | invalid inputs, failed hook, and null free | defined failure paths for every C unit |

The mistakes people actually make cluster around a few ideas, and each has a one-line defense in the solution.

Leaving padding out of `bytes_used` makes $U$ too small, so `high_water` under-sizes the engine's arena; the solution writes `off += pad + n`. Aligning the offset instead of the address produces misaligned pointers whenever a block starts off a 64-byte boundary; the solution calls `pad_for(b->base + a->off, align)`. Rewinding only the offset, not the block, sends allocations after a cross-block rewind into the wrong block; the solution restores `cur` and recomputes `used_before`.

> **Warning:** Accepting a mark that is later than the current position hands the same bytes out twice. Compare the mark lexicographically with the position and ignore a later one.

Freeing blocks on rewind, or never reusing them, puts a hook call in every step: `malloc` cost and a failure point per kernel call. Rewind keeps every block, and the second allocation case reuses a later block that fits. Squeezing an oversized request into a normal block is a heap overflow that ASan will report; a request above `block_bytes` gets a block of its own size, rounded to 64. Changing fields before the hook call succeeds means a failed allocation corrupts the arena or leaks the block list; `grow_list` and the block allocation both happen before any field is written. And an off-by-one in the fit test allocates a new block for a request that fits exactly; `n <= size - off - pad` accepts the exact fit.

## Where this leads

Every other optional C exercise uses this support layer. [FlashAttention forward in C](06-flash-attention-forward.md) is the most direct user: `tl_flash_attn_fwd_f32` takes a `tl_arena *scratch` for its query and key tiles and running maxima, rewound per call. [Cache-blocked matmul](04-tiled-batch-invariant-matmul.md), [softmax](05-softmax-three-pass-and-online.md), [paged attention](07-paged-attention-decode.md), and [fused quantized matmul](08-fused-quantized-matmul.md) use the shared error and allocator helpers in their standalone test binaries. Outside this chapter, the [thread pool](03-thread-pool-and-parallel-for.md), the [paged KV block pool](../p08-inference/08-paged-kv-block-pool.md), and the C data structures in *Algorithm Mastery* ([growable array](../../algorithms/16-systems-data-structures/01-growable-array.md), [Swiss table](../../algorithms/16-systems-data-structures/02-swiss-table.md), [binary heap top-k](../../algorithms/16-systems-data-structures/04-binary-heap-top-k.md)) reuse the same helpers. None of them is linked into Python or Rust.

Production systems take each piece of this design further:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| arena with marks | ggml's `ggml_context` and `ggml_gallocr` | a graph allocator that plans tensor lifetimes ahead and reuses memory between tensors that are never live together | `ggml/src/ggml-alloc.c` |
| per-thread arenas | jemalloc arenas, tcmalloc per-CPU caches | many arenas chosen by thread or CPU, size classes, returning memory to the OS | jemalloc `doc/jemalloc.xml.in`; tcmalloc `docs/design.md` |
| rewind-per-step | PyTorch's CUDA caching allocator | keeps freed blocks per stream and size bucket for reuse across iterations | `c10/cuda/CUDACachingAllocator.cpp` |
| `high_water` | vLLM's memory profiling before KV allocation | runs a dummy forward pass, measures the peak, and gives the rest to the KV cache | vLLM `worker/` (`determine_num_available_blocks`) |

### Further reading

- Hanson, *Fast allocation and deallocation of memory based on object lifetimes* (1990).
- Bonwick, *The Slab Allocator* (USENIX 1994).
- The `tcmalloc` and `jemalloc` design notes.

## Key terms

- **arena**: an allocator that hands out memory by bumping an offset through owned blocks and releases everything after a point at once.
- **bump allocation**: allocating by advancing an offset past padding and the request, with no per-allocation bookkeeping.
- **mark**: a saved arena position $(b, o)$ that a later rewind returns to.
- **alignment padding**: the bytes $\mathrm{pad}(p, a)$ needed to move address $p$ up to a multiple of $a$.
- **high-water mark**: the largest `bytes_used` the arena has ever reached, used to size it.
- **allocator hook**: the replaceable `alloc`/`free` pair through which all C memory requests go, which tests can count or make fail.

## Summary

- An arena allocates by bumping an offset and frees everything after a point at once by rewinding it: allocation is an add and a compare, and nothing is freed piece by piece.
- Alignment is padding computed from the address, not from the offset: each pointer is a multiple of its requested power of two, up to 4096.
- Marks nest like a stack: rewinding releases exactly what came after the mark, across blocks, and a mark from the future is refused.
- Blocks are kept, not freed, on rewind, so after warmup a mark, allocate, rewind step never calls the allocator hook: it cannot fail and costs no `malloc`.
- Every hook failure leaves the arena unchanged and leak-free, whichever allocation is the one that fails.

## Review questions

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
6. In the worked example, why does the second `alloc(100, 8)` after the rewind return the same address as the first?
   <details><summary>Answer</summary>The rewind restored the position to (0, 68), the same position the first `alloc(100, 8)` started from, and the block was kept. The padding $\mathrm{pad}(68, 8) = 4$ is the same, so the returned offset is 72 again and no hook call is made.</details>
