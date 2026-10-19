/* c/src/runtime/arena.c (rt.02): a bump allocator with marks.
 * Contract: tinyllm/arena.h. Rules: c/ABI.md. Callers: the L9 kernels'
 * scratch (FlashAttention tiles, online-softmax rows), one arena per engine
 * step, rewound with a mark instead of freed piece by piece.
 *
 * An arena is a list of blocks from the allocator hook (tl_alloc). The
 * position is (cur, off): the index of the current block and the first free
 * byte in it. tl_arena_alloc pads off up to the requested alignment and
 * bumps it by n. When the request does not fit, the arena moves to the next
 * block: one already in the list (blocks after cur are free, because marks
 * nest like a stack) when one is large enough, else a new block from the
 * hook. A mark is a copy of (cur, off); rewinding to it releases everything
 * allocated after it in O(1) and keeps every block for reuse, so a steady
 * state of "mark, allocate, reset" makes no hook call at all.
 *
 * The struct below is fixed; the functions that follow maintain it.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm/arena.h"

#define TL_ARENA_DEFAULT_BLOCK ((size_t)1 << 20) /* 1 MiB */
#define TL_ARENA_DEFAULT_ALIGN 64                /* a cache line, and AVX-512 */
#define TL_ARENA_MAX_ALIGN 4096                  /* a page */

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
    return (size_t)((align - (a & (align - 1))) & (align - 1));
}

tl_status tl_arena_create(size_t block_bytes, tl_arena **out) {
    if (out == NULL) {
        tl_set_last_error("tl_arena_create: out is NULL");
        return TL_EINVAL;
    }
    *out = NULL;
    tl_arena *a = tl_alloc(sizeof *a, _Alignof(tl_arena));
    if (a == NULL) {
        tl_set_last_error("tl_arena_create: the allocator hook returned NULL");
        return TL_ENOMEM;
    }
    memset(a, 0, sizeof *a);
    a->block_bytes = block_bytes ? block_bytes : TL_ARENA_DEFAULT_BLOCK;
    *out = a;
    return TL_OK;
}

/* Makes sure blocks[] has room for one more entry. On failure nothing
 * changes and the error slot is set. */
static int grow_list(tl_arena *a) {
    if (a->n_blocks < a->cap) return 1;
    size_t cap = a->cap ? 2 * a->cap : 8;
    tl_arena_block *b = tl_alloc(cap * sizeof *b, _Alignof(tl_arena_block));
    if (b == NULL) return 0;
    if (a->n_blocks) memcpy(b, a->blocks, a->n_blocks * sizeof *b);
    tl_free(a->blocks);
    a->blocks = b;
    a->cap = cap;
    return 1;
}

void *tl_arena_alloc(tl_arena *a, size_t n, size_t align) {
    if (a == NULL || n == 0) {
        tl_set_last_error(a == NULL ? "tl_arena_alloc: arena is NULL" : "tl_arena_alloc: size 0");
        return NULL;
    }
    if (align == 0) align = TL_ARENA_DEFAULT_ALIGN;
    if ((align & (align - 1)) != 0 || align > TL_ARENA_MAX_ALIGN) {
        tl_set_last_error("tl_arena_alloc: align must be a power of two <= 4096");
        return NULL;
    }
    if (n > SIZE_MAX / 2) {
        tl_set_last_error("tl_arena_alloc: size too large");
        return NULL;
    }

    /* 1. The current block. */
    if (a->cur < a->n_blocks) {
        tl_arena_block *b = &a->blocks[a->cur];
        size_t pad = pad_for(b->base + a->off, align);
        if (pad <= b->size - a->off && n <= b->size - a->off - pad) {
            void *p = b->base + a->off + pad;
            a->off += pad + n;
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
        if (pad_for(a->blocks[i].base, align) + n <= a->blocks[i].size) {
            found = i;
            break;
        }
    }
    if (found == a->n_blocks) {
        /* 3. A new block from the hook, inserted at position next. */
        size_t size = n > a->block_bytes ? (n + 63) & ~(size_t)63 : a->block_bytes;
        size_t balign = align > TL_ARENA_DEFAULT_ALIGN ? align : TL_ARENA_DEFAULT_ALIGN;
        if (!grow_list(a)) {
            tl_set_last_error("tl_arena_alloc: the allocator hook returned NULL");
            return NULL;
        }
        unsigned char *base = tl_alloc(size, balign);
        if (base == NULL) {
            tl_set_last_error("tl_arena_alloc: the allocator hook returned NULL");
            return NULL;
        }
        a->blocks[a->n_blocks] = (tl_arena_block){base, size, 0};
        found = a->n_blocks++;
    }
    if (found != next) {
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
    size_t used = a->used_before + a->off;
    if (used > a->high_water) a->high_water = used;
    return b->base + pad;
}

tl_arena_mark tl_arena_mark_get(const tl_arena *a) {
    tl_arena_mark m = {0, 0};
    if (a == NULL) return m;
    m.block = a->cur;
    m.offset = a->off;
    return m;
}

void tl_arena_reset_to(tl_arena *a, tl_arena_mark m) {
    if (a == NULL) return;
    /* Marks are ordered like positions: by block, then by offset. A mark
     * taken before the first block is (0, 0), the start of block 0. */
    size_t mb = m.block, mo = m.offset;
    if (a->cur >= a->n_blocks) {
        /* No block yet: only that empty mark is valid, and it changes nothing. */
        if (mb != 0 || mo != 0) tl_set_last_error("tl_arena_reset_to: mark is later than the position");
        return;
    }
    if (mb > a->cur || (mb == a->cur && mo > a->off)) {
        tl_set_last_error("tl_arena_reset_to: mark is later than the position");
        return;
    }
    size_t before = 0;
    for (size_t i = 0; i < mb; i++) before += a->blocks[i].used;
    a->used_before = before;
    a->cur = mb;
    a->off = mo;
}

void tl_arena_stats_get(const tl_arena *a, tl_arena_stats *out) {
    if (out == NULL) return;
    memset(out, 0, sizeof *out);
    if (a == NULL) return;
    size_t reserved = 0;
    for (size_t i = 0; i < a->n_blocks; i++) reserved += a->blocks[i].size;
    out->bytes_used = a->cur < a->n_blocks ? a->used_before + a->off : 0;
    out->bytes_reserved = reserved;
    out->high_water = a->high_water;
    out->n_blocks = a->n_blocks;
}

void tl_arena_destroy(tl_arena *a) {
    if (a == NULL) return;
    for (size_t i = 0; i < a->n_blocks; i++) tl_free(a->blocks[i].base);
    tl_free(a->blocks);
    tl_free(a);
}
