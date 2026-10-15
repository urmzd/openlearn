/* tinyllm/arena.h (rt.02): a bump allocator with marks, for kernel scratch
 * (attention tiles, online-softmax rows). Rules in c/ABI.md.
 *
 * An arena is a list of blocks obtained through the allocator hook
 * (tl_alloc). tl_arena_alloc bumps an offset inside the current block and
 * starts a new block when the request does not fit. tl_arena_reset_to
 * rewinds to a mark but keeps every block for reuse, so a steady-state
 * engine step makes zero calls to the hook after warmup (the rt.02 bench
 * counts them).
 *
 * Not thread-safe: the caller serializes every call on one arena.
 *
 * module: rt.02 (c/src/runtime/arena.c)
 * chapter: content/courses/build-an-llm/p09-kernels/02-arena-allocator-with-marks.md */
#ifndef TINYLLM_ARENA_H
#define TINYLLM_ARENA_H

#include <stddef.h>
#include <stdint.h>

#include "tinyllm/abi.h"


typedef struct tl_arena tl_arena;

/* A position in the arena: the index of the current block and the offset
 * inside it. Only tl_arena_mark_get makes one. */
typedef struct {
    size_t offset;
    size_t block;
} tl_arena_mark;

typedef struct {
    size_t bytes_used;     /* sum of live allocations, including alignment padding */
    size_t bytes_reserved; /* sum of block sizes obtained from the hook */
    size_t high_water;     /* largest bytes_used since create */
    size_t n_blocks;
} tl_arena_stats;

/* Creates an empty arena whose blocks are block_bytes long (0 means
 * 1 MiB). No block is allocated until the first tl_arena_alloc.
 * TL_EINVAL for out == NULL; TL_ENOMEM when the hook fails. */
tl_status tl_arena_create(size_t block_bytes, tl_arena **out);

/* n bytes aligned to align (a power of two, at most 4096; 0 means 64).
 * A request larger than block_bytes gets a block of its own, rounded up.
 * Returns NULL and sets the error slot for n == 0, a bad align, or when the
 * hook fails; the arena is unchanged in every failure. Memory is not
 * zeroed. */
void *tl_arena_alloc(tl_arena *a, size_t n, size_t align);

/* The current position. */
tl_arena_mark tl_arena_mark_get(const tl_arena *a);

/* Rewinds to m: every allocation made after m was taken is released at
 * once, earlier ones stay valid. Blocks are kept for reuse. m must come
 * from this arena and must not be later than the current position (marks
 * nest like a stack); a later mark is ignored and sets the error slot. */
void tl_arena_reset_to(tl_arena *a, tl_arena_mark m);

void tl_arena_stats_get(const tl_arena *a, tl_arena_stats *out);

/* Frees every block through the hook. NULL does nothing. */
void tl_arena_destroy(tl_arena *a);


#endif /* TINYLLM_ARENA_H */
