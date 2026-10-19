# Arena allocator

`malloc` is a general-purpose allocator, and general purpose has a cost: every allocation needs a header, every `free` has to find its block again, and a long-running program fragments its heap. When many objects share one lifetime (everything built while compiling one function, rendering one frame, or serving one request) you can do far better. An arena allocator hands out memory by moving a pointer forward and frees everything at once. It has no per-allocation `free`, no headers, and no free list.

This section builds one with block chaining, arbitrary power-of-two alignment, marks you can rewind to, and a reset that keeps memory for the next cycle. The interesting parts are all about arithmetic: aligning an address rather than an offset, rounding with a mask, and writing a bounds check that cannot overflow. The reference implementation was itself first written with the alignment bug described below, and its tests caught it. A version extended for an inference runtime is [C runtime support and an arena allocator with marks](../../../../../content/courses/build-an-llm/p09-kernels/02-arena-allocator-with-marks.md) in *Build Your Own LLM System*.

> **Learning objectives**
> - Implement bump allocation over a chain of blocks, using a flexible array member for header and storage in one allocation.
> - Align an allocation's address to any power of two with `(n + align - 1) & ~(align - 1)`, and explain why it fails for other values.
> - Write bounds checks on `size_t` that cannot wrap.
> - Implement marks and resets, and explain why reset keeps memory while free returns it.

## The interface

[`arena.h`](arena.h) declares the interface, [`arena.c`](arena.c) is the solution, and [`main.c`](main.c) is the test suite.

| Function | Does |
|----------|------|
| `arena_alloc_aligned(a, size, align)` | Bump, honouring any power-of-two alignment |
| `arena_alloc(a, size)` | Aligned for any scalar type |
| `arena_strdup(a, s)` | Copy a string in |
| `arena_mark(a)` / `arena_reset_to(a, m)` | Save a position, rewind to it |
| `arena_reset(a)` | Rewind everything, keep the memory |
| `arena_free(a)` | Return every block to the system |
| `arena_block_count(a)` | Blocks held, so the tests can see reuse |

Three constraints apply. `arena_alloc_aligned` returns `NULL` for an alignment that is not a power of two. A zero-size request returns a valid, aligned, non-`NULL` pointer. And an allocation larger than the default block must still work.

To build and run the tests from this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c arena.c -o arena && ./arena
# ok  c/08-arena-allocator  11899 checks passed
```

Add `-g -fsanitize=address,undefined` to check that no allocation strays outside its block.

## The trade an arena makes

No `free` per allocation is a feature, not a limitation. Deallocation is one pointer write regardless of how many objects are live, there is no per-object header, and fragmentation cannot happen. The cost is that individual lifetimes are gone: everything in an arena dies together. That trade is why compilers, game engines, and request-scoped server code use arenas heavily, and why nothing with unpredictable object lifetimes does.

The difference between `reset` and `free` is the entire use case. `reset` keeps the memory and `free` returns it. The pattern an arena is built for is per-frame or per-request: allocate freely, reset at the boundary, repeat. If `reset` returned memory to the system, every cycle would pay for fresh `malloc` calls and you would have gained nothing over the allocator you were avoiding. The test asserts that the second round hands back *the same addresses* as the first.

## Alignment arithmetic

Align the address, not the offset within the block. This is the bug the reference implementation was first written with. Offset arithmetic works only while the requested alignment is no stricter than the block's own: `data[]` starts at `max_align_t`, typically 16 bytes, so aligning the offset silently returns misaligned memory the moment someone asks for 32 or 64. Both are ordinary requests, from SIMD types and cache-line padding respectively. The alignment test loops alignments from 1 to 64 for exactly this reason.

The rounding formula `(n + align - 1) & ~(align - 1)` only works for powers of two. Adding `align - 1` pushes past the boundary and the mask snaps back down to it, which depends entirely on `align` having a single bit set. That is why every alignment API in C, including `aligned_alloc` and `_Alignas`, insists on a power of two, and why this one returns `NULL` rather than misbehaving when handed 3.

For the bounds check, test the subtraction, not the sum. `off + size > cap` looks like the obvious test and can wrap on a hostile size, producing a small sum that passes the check and a write far outside the block. `size > cap - off` cannot wrap, because `off <= cap` is already established.

## Deriving the design

Six observations build the allocator.

If everything dies together, allocation is a pointer bump. No allocation needs to be found again, so none needs a header or a free-list entry. Keep a block and an offset `used`; an allocation returns `data + used` and advances `used`.

Alignment is a property of the address. A `T` must sit at an address that is a multiple of its alignment. So round the *address* `data + used` up, then convert back to an offset, using the power-of-two rounding above.

The block header and storage should be one allocation. A flexible array member `data[]` after the header does that, and `_Alignas(max_align_t)` on it makes the storage start at the strictest scalar alignment.

When a block is full, chain a new one. Blocks link backwards from the newest (`prev`). A new block is the default size or, for an oversized request, the request plus `align - 1` bytes of worst-case padding, so the retry is guaranteed to fit.

A position is a (block, offset) pair. That pair is a mark. Rewinding frees every block newer than the marked one and restores its offset.

Reset keeps one block. The common pattern is reset-then-refill, so keeping the oldest block means the next cycle usually needs no `malloc`.

## Building it step by step

The block layout and the rounding identity:

```c
/* Header and storage in one allocation, via the flexible array member. */
struct Block {
  Block *prev;      /* blocks chain backwards from the newest */
  size_t cap;
  size_t used;
  /* Storage starts at the strictest scalar alignment. */
  _Alignas(max_align_t) unsigned char data[];
};

/* Round n up to a multiple of align (a power of two). In uintptr_t, because
 * it is applied to an address. */
static uintptr_t align_up_uptr(uintptr_t n, size_t align) {
  return (n + (uintptr_t)align - 1) & ~((uintptr_t)align - 1);
}

static int is_power_of_two(size_t x) { return x != 0 && (x & (x - 1)) == 0; }
```

`x & (x - 1)` clears the lowest set bit, so it is zero exactly when `x` has a
single bit set.

The allocator itself:

```c
void *arena_alloc_aligned(Arena *a, size_t size, size_t align) {
  if (!is_power_of_two(align)) return NULL;

  /* At most two attempts: the current block, then a fresh one sized to fit. */
  for (int attempt = 0; attempt < 2; attempt++) {
    Block *b = a->head;
    if (b) {
      /* Align the ADDRESS, then convert back to an offset. Aligning the
       * offset would be wrong for any align stricter than data[]'s own. */
      uintptr_t base = (uintptr_t)(b->data + b->used);
      uintptr_t aligned = align_up_uptr(base, align);
      size_t off = (size_t)(aligned - (uintptr_t)b->data);

      /* off + size could wrap on a hostile size; cap - off cannot, once
       * off <= cap is known. */
      if (off <= b->cap && size <= b->cap - off) {
        b->used = off + size;
        a->used_total += size;
        return b->data + off;
      }
    }

    if (attempt == 0) {
      /* Worst-case padding in a fresh block is align - 1, so ask for it now
       * rather than discovering the shortfall on the retry. */
      if (size > SIZE_MAX - align) return NULL;
      if (arena_grow(a, size + align - 1) != 0) return NULL;
    }
  }
  return NULL;
}
```

A zero-size request takes the same path: it returns an aligned pointer into the
block and advances nothing past it. `arena_grow` uses the larger of the default
block size and the request, and checks `cap > SIZE_MAX - sizeof(Block)` before
`malloc(sizeof(Block) + cap)` so the header addition cannot wrap.

Strings need no alignment, so `arena_strdup` asks for 1 rather than wasting up
to 15 bytes per string:

```c
char *arena_strdup(Arena *a, const char *s) {
  size_t n = strlen(s) + 1;          /* include the terminator */
  char *p = arena_alloc_aligned(a, n, 1);
  if (!p) return NULL;
  memcpy(p, s, n);
  return p;
}
```

Marks and the two resets:

```c
ArenaMark arena_mark(const Arena *a) {
  ArenaMark m;
  m.block = a->head;                       /* NULL on a fresh arena */
  m.used = a->head ? a->head->used : 0;
  return m;
}

void arena_reset_to(Arena *a, ArenaMark mark) {
  /* Free every block newer than the marked one; keeping them would need a
   * free list. A NULL mark (taken before any block existed) frees all. */
  while (a->head && a->head != mark.block) {
    Block *dead = a->head;
    a->head = dead->prev;
    free(dead);
  }
  if (a->head) a->head->used = mark.used;
}

void arena_reset(Arena *a) {
  /* Keep the oldest block and drop the rest, so a refill usually needs
   * no malloc. */
  while (a->head && a->head->prev) {
    Block *dead = a->head;
    a->head = dead->prev;
    free(dead);
  }
  if (a->head) a->head->used = 0;
  a->used_total = 0;
}
```

Note that `arena_reset_to` does not adjust `used_total`; that counter is only
reset by `arena_reset` and `arena_free`.

The full solution is in [`arena.c`](arena.c).

## Why this design

Allocation is O(1): a few arithmetic operations, plus one `malloc` when a block fills. Freeing is O(blocks). Per-allocation overhead is only alignment padding, never a header.

Against `malloc` and `free`, the arena wins by doing less. A general allocator must find a fitting free block, track sizes, and fight fragmentation. An arena does none of that, which is why it is several times faster for many small, same-lifetime objects.

Against a pool, such as the C++ [memory pool allocator](../../cpp/10-pool-allocator/README.md), the choice depends on lifetimes. A pool frees individual fixed-size blocks; an arena frees nothing individually but serves any size. Pick by whether lifetimes are individual or grouped.

On rewind, newer blocks are freed rather than kept. Keeping them for reuse would save `malloc` calls in a rewind-heavy workload but needs a list of spare blocks; the reference chose the smaller design.

The accepted trade is up to `align - 1` bytes of padding per allocation, and the tail of each full block is wasted when a larger request forces a new one.

## What can go wrong

The tests write to every byte of every allocation, then read them all back. Checking that pointers look plausible is not enough: a wrong bump produces overlapping regions where each individual pointer is fine and the second allocation quietly corrupts the first. Filling and verifying is the only way to see it.

| Mistake | Consequence |
|---------|-------------|
| Aligning the offset instead of the address | Misaligned memory for any alignment above `max_align_t` |
| Accepting a non-power-of-two alignment | The mask produces a wrong, possibly unaligned result |
| `off + size <= cap` as the bounds check | Wraps on a huge `size` and writes far outside the block |
| Sizing a fresh block to exactly `size` | Alignment padding makes the retry fail |
| `malloc(sizeof(Block) + cap)` without an overflow check | The header addition wraps for a huge request |
| `reset` that frees every block | Every cycle pays for fresh `malloc` calls |

> **Note:** `arena_reset_to` does not adjust `used_total`. That counter is reset only by `arena_reset` and `arena_free`.

## Where this leads

Two extensions are worth trying. Add `arena_realloc` that grows in place when the block is the most recent allocation and copies otherwise, which is the trick that makes a growable array inside an arena cheap. Then make `arena_reset` poison the reclaimed bytes under a debug flag, so use-after-reset shows up immediately instead of reading plausible stale data. *Build Your Own LLM System* uses an arena with marks for per-step scratch memory in [C runtime support and an arena allocator with marks](../../../../../content/courses/build-an-llm/p09-kernels/02-arena-allocator-with-marks.md), and the [string interning](../09-string-interning/README.md) section that follows shows where an arena is the natural home for data that is never freed individually.

## Key terms

- **Arena (bump) allocator**: an allocator that serves requests by advancing an offset and frees all allocations at once.
- **Alignment**: the requirement that an object's address be a multiple of some power of two.
- **Flexible array member**: a trailing `data[]` array in a struct, letting header and variable-size storage share one allocation.
- **Mark**: a saved (block, offset) position that the arena can later rewind to.
- **`max_align_t`**: the standard type whose alignment is at least as strict as every scalar type's.

## Summary

- Allocation bumps an offset within the newest block; a full block causes a new one to be chained in front.
- Alignment is applied to the address `data + used`, using a mask that is valid only for powers of two.
- Bounds are checked as `size <= cap - off` so the arithmetic cannot wrap; oversized blocks reserve `align - 1` bytes of padding.
- Marks rewind by freeing newer blocks; `reset` keeps the oldest block so the next cycle needs no `malloc`.
- Arenas trade individual lifetimes for O(1) allocation, no headers, and no fragmentation.

## Review questions

1. Why must the solution align the address `data + used` rather than the offset
   `used`?

   <details><summary>Answer</summary>

   `data` is guaranteed to be aligned only to `max_align_t` (usually 16). An
   offset that is a multiple of 64 lands on a 64-byte boundary only if `data`
   itself is on one, which it need not be. Aligning the actual address is correct
   for every power-of-two alignment.

   </details>

2. Why does a fresh block for an oversized request get `size + align - 1` bytes?

   <details><summary>Answer</summary>

   The new block's `data` may need up to `align - 1` bytes of padding before
   the first suitably aligned address. Reserving that up front guarantees the
   second attempt fits, which is why the loop needs only two iterations.

   </details>

3. Why is `size <= b->cap - off` safe where `off + size <= b->cap` is not?

   <details><summary>Answer</summary>

   With a huge `size`, `off + size` wraps around to a small number and passes the
   check, so the allocation overruns the block. After `off <= b->cap` has been
   checked, `b->cap - off` cannot underflow, and the comparison is exact.

   </details>

4. What does `arena_reset_to` do with a mark taken on a fresh arena?

   <details><summary>Answer</summary>

   The mark's block is `NULL`, so the loop runs until `a->head` is `NULL`,
   freeing every block. The arena returns to its initial empty state, which is
   exactly "rewind to before anything was allocated".

   </details>

5. Why is `max_align_t` the alignment for `arena_alloc`?

   <details><summary>Answer</summary>

   `arena_alloc` does not know the type being stored, so it must be suitable for
   any scalar type, as `malloc` is. `_Alignof(max_align_t)` is by definition at
   least as strict as every scalar type's alignment.

   </details>
