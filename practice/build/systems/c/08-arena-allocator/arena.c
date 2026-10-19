/* arena.c - the reference implementation of the arena. */
#include "arena.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_DEFAULT_BLOCK 4096

/* Blocks chain backwards from the newest. `data` is a flexible array member,
 * so the header and the storage are one allocation rather than two. */
struct Block {
  Block *prev;
  size_t cap;
  size_t used;
  /* Force the storage to start at the strictest scalar alignment, so that a
   * pointer into data[] can be aligned by arithmetic alone. */
  _Alignas(max_align_t) unsigned char data[];
};

/* Round `n` up to the next multiple of `align`, which must be a power of two.
 *
 * The identity is (n + align - 1) & ~(align - 1): adding align-1 pushes past
 * the boundary, and masking off the low bits snaps back down to it. Only a
 * power of two has the single-bit pattern that makes the mask work, which is
 * why every alignment API in C insists on one. */
static uintptr_t align_up_uptr(uintptr_t n, size_t align) {
  return (n + (uintptr_t)align - 1) & ~((uintptr_t)align - 1);
}

static int is_power_of_two(size_t x) { return x != 0 && (x & (x - 1)) == 0; }

/* Attach a fresh block big enough for `need` bytes. */
static int arena_grow(Arena *a, size_t need) {
  size_t cap = a->block_size;
  if (cap < need) cap = need;

  if (cap > SIZE_MAX - sizeof(Block)) return -1;
  Block *b = malloc(sizeof(Block) + cap);
  if (!b) return -1;

  b->prev = a->head;
  b->cap = cap;
  b->used = 0;
  a->head = b;
  return 0;
}

void arena_init(Arena *a, size_t block_size) {
  a->head = NULL;
  a->block_size = block_size ? block_size : ARENA_DEFAULT_BLOCK;
  a->used_total = 0;
}

void *arena_alloc_aligned(Arena *a, size_t size, size_t align) {
  if (!is_power_of_two(align)) return NULL;

  /* Try the current block, then a fresh one. The loop runs at most twice:
   * either the live block has room, or the block just attached does, because
   * arena_grow sizes it to fit. */
  for (int attempt = 0; attempt < 2; attempt++) {
    Block *b = a->head;
    if (b) {
      /* Align the ADDRESS, not the offset within the block.
       *
       * Aligning the offset is the tempting shortcut and it is only correct
       * while the requested alignment is no stricter than the block's own.
       * data[] starts at max_align_t (typically 16), so an offset-based
       * version silently returns misaligned memory the moment someone asks
       * for a 32- or 64-byte boundary, which SIMD types and cache-line
       * padding both do. */
      uintptr_t base = (uintptr_t)(b->data + b->used);
      uintptr_t aligned = align_up_uptr(base, align);
      size_t off = (size_t)(aligned - (uintptr_t)b->data);

      /* off + size can wrap on a hostile size, and comparing the wrapped sum
       * against cap would pass. Check the subtraction instead, which cannot
       * overflow because off <= cap here. */
      if (off <= b->cap && size <= b->cap - off) {
        b->used = off + size;
        a->used_total += size;
        return b->data + off;
      }
    }

    if (attempt == 0) {
      /* Worst case the new block wastes align-1 bytes on padding, so ask for
       * that much extra rather than discovering the shortfall on retry. */
      if (size > SIZE_MAX - align) return NULL;
      if (arena_grow(a, size + align - 1) != 0) return NULL;
    }
  }
  return NULL;
}

void *arena_alloc(Arena *a, size_t size) {
  return arena_alloc_aligned(a, size, _Alignof(max_align_t));
}

char *arena_strdup(Arena *a, const char *s) {
  size_t n = strlen(s) + 1;
  /* Strings need no alignment beyond 1, and asking for more would waste up to
   * 15 bytes per string in an arena full of them. */
  char *p = arena_alloc_aligned(a, n, 1);
  if (!p) return NULL;
  memcpy(p, s, n);
  return p;
}

ArenaMark arena_mark(const Arena *a) {
  ArenaMark m;
  m.block = a->head;
  m.used = a->head ? a->head->used : 0;
  return m;
}

void arena_reset_to(Arena *a, ArenaMark mark) {
  /* Drop every block newer than the marked one. They are freed rather than
   * kept, because keeping them would need a free list and the whole appeal of
   * this allocator is that it has almost no bookkeeping. */
  while (a->head && a->head != mark.block) {
    Block *dead = a->head;
    a->head = dead->prev;
    free(dead);
  }
  if (a->head) a->head->used = mark.used;
}

void arena_reset(Arena *a) {
  /* Keep the first block and drop the rest: the common pattern is reset-then-
   * refill, and holding one block means the refill usually needs no malloc.
   * Freeing everything would give the caller a fresh malloc on every cycle. */
  while (a->head && a->head->prev) {
    Block *dead = a->head;
    a->head = dead->prev;
    free(dead);
  }
  if (a->head) a->head->used = 0;
  a->used_total = 0;
}

void arena_free(Arena *a) {
  while (a->head) {
    Block *dead = a->head;
    a->head = dead->prev;
    free(dead);
  }
  a->used_total = 0;
}

size_t arena_block_count(const Arena *a) {
  size_t n = 0;
  for (const Block *b = a->head; b; b = b->prev) n++;
  return n;
}
