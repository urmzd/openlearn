/* tinyllm/ds.h (ds.01 to ds.03): the data structures the C runtime uses.
 * chapter: content/courses/algorithms/13-systems-data-structures/01-growable-array.md
 * chapter: content/courses/algorithms/13-systems-data-structures/02-swiss-table.md
 * chapter: content/courses/algorithms/13-systems-data-structures/03-intrusive-list-and-lru.md
 * Rules in c/ABI.md.
 *
 *   tl_vec  ds.01  c/src/ds/vec.c     type-erased growable array (rt.04 block tables)
 *   tl_map  ds.02  c/src/ds/swiss.c   Swiss table, u64 to u64 (rt.04 prefix index)
 *   tl_lru  ds.03  c/src/ds/list.c, lru.c   intrusive list and LRU (rt.04 cached blocks)
 *
 * Every allocation goes through the allocator hook (tl_alloc). Not
 * thread-safe: the caller serializes every call on one object. */
#ifndef TINYLLM_DS_H
#define TINYLLM_DS_H

#include <stddef.h>
#include <stdint.h>

#include "tinyllm/abi.h"


/* -- ds.01: growable array ------------------------------------------------ */

/* elem bytes per element. A zeroed tl_vec is not valid: call tl_vec_init.
 * The caller may read data[0 .. len * elem) directly. */
typedef struct {
    void *data;
    size_t len, cap, elem;
} tl_vec; /* 4 pointer-sized fields */

/* len = cap = 0, data = NULL. TL_EINVAL for v == NULL or elem == 0. */
tl_status tl_vec_init(tl_vec *v, size_t elem);

/* Ensures cap >= cap_min; never shrinks. On failure the vec is unchanged:
 * TL_ENOMEM when the hook fails, TL_EINVAL when cap_min * elem overflows. */
tl_status tl_vec_reserve(tl_vec *v, size_t cap_min);

/* Appends elem bytes copied from x. Growth at least doubles cap (amortized
 * O(1)). On failure the vec is unchanged (same data, len, and cap). */
tl_status tl_vec_push(tl_vec *v, const void *x);

/* Pointer to element i, or NULL (error slot set) for i >= len. */
void *tl_vec_at(const tl_vec *v, size_t i);

/* Frees data through the hook and resets to the tl_vec_init state (elem
 * kept), so the vec can be reused. */
void tl_vec_free(tl_vec *v);

/* -- ds.02: Swiss table ---------------------------------------------------- */

/* Open addressing in groups of 16 control bytes (SWAR group match), maximum
 * load 7/8, tombstones on delete. Every u64 is a valid key, 0 and
 * UINT64_MAX included. */
typedef struct tl_map tl_map;

/* hint: expected number of keys (0 is fine). TL_ENOMEM, TL_EINVAL. */
tl_status tl_map_create(size_t hint, tl_map **out);

/* Inserts or overwrites. A growth that fails leaves the old table valid and
 * unchanged, and returns TL_ENOMEM. */
tl_status tl_map_put(tl_map *m, uint64_t k, uint64_t v);

/* 1 and *v set when found (v may be NULL), 0 when not. */
int tl_map_get(const tl_map *m, uint64_t k, uint64_t *v);

/* 1 when k was present and is now removed, 0 when absent. Never allocates. */
int tl_map_del(tl_map *m, uint64_t k);

size_t tl_map_len(const tl_map *m);

void tl_map_destroy(tl_map *m); /* NULL does nothing */

/* -- ds.03: intrusive list and LRU ------------------------------------------ */

/* Embed a tl_list_node in your own struct and recover the struct with
 * TL_CONTAINER_OF. A node that is in no list has prev == next == NULL;
 * zero-initialize nodes before first use. */
typedef struct tl_list_node {
    struct tl_list_node *prev, *next;
} tl_list_node;

#define TL_CONTAINER_OF(ptr, type, member) \
    ((type *)(void *)((char *)(ptr)-offsetof(type, member)))

/* A circular doubly linked list with a sentinel head: head.next is the
 * least recently used node, head.prev the most recent. */
typedef struct {
    tl_list_node head;
    size_t len;
} tl_lru;

void tl_lru_init(tl_lru *l);

/* Inserts n as the most recent, or moves it there if it is already in l. */
void tl_lru_touch(tl_lru *l, tl_list_node *n);

/* Unlinks n and sets its prev and next to NULL. A node in no list is left
 * alone. */
void tl_lru_remove(tl_lru *l, tl_list_node *n);

/* Unlinks and returns the least recently used node, or NULL when empty. */
tl_list_node *tl_lru_pop_oldest(tl_lru *l);


#endif /* TINYLLM_DS_H */
