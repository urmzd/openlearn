/* arena.h - a bump allocator.
 *
 * The interface. arena.c implements it and main.c tests it.
 *
 * An arena hands out memory by moving a pointer forward and frees everything
 * at once. There is no per-allocation free, which is the entire point: you
 * give up individual lifetimes and get back an allocation that costs a few
 * instructions and a deallocation that costs one.
 */
#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>

typedef struct Block Block;

typedef struct {
  Block *head;      /* most recent block; blocks chain backwards */
  size_t block_size; /* default size for new blocks */
  size_t used_total; /* bytes handed out, across every block */
} Arena;

typedef struct {
  Block *block;
  size_t used;
} ArenaMark;

/* Prepare an arena. `block_size` is the default chunk requested from malloc;
 * pass 0 for a sensible default. Allocates nothing yet. */
void arena_init(Arena *a, size_t block_size);

/* Allocate `size` bytes aligned to `align`, which must be a power of two.
 * Returns NULL on allocation failure or invalid alignment.
 *
 * A zero-size request returns a non-NULL, correctly aligned pointer that must
 * not be dereferenced, matching what callers expect from malloc(0) in
 * practice. */
void *arena_alloc_aligned(Arena *a, size_t size, size_t align);

/* Allocate with alignment suitable for any scalar type. */
void *arena_alloc(Arena *a, size_t size);

/* Copy a NUL-terminated string into the arena. NULL on failure. */
char *arena_strdup(Arena *a, const char *s);

/* Record the current position, so it can be rewound to later. */
ArenaMark arena_mark(const Arena *a);

/* Rewind to a previous mark, making everything allocated since reusable.
 * Blocks are kept for reuse rather than returned to the system. */
void arena_reset_to(Arena *a, ArenaMark mark);

/* Rewind everything, keeping the memory for reuse. */
void arena_reset(Arena *a);

/* Return every block to the system. */
void arena_free(Arena *a);

/* How many blocks are currently held. Exposed so the tests can prove that
 * reuse actually reuses rather than quietly allocating more. */
size_t arena_block_count(const Arena *a);

#endif /* ARENA_H */
