/* c/src/ds/vec.c (ds.01): tl_vec, the type-erased growable array.
 * Contract: tinyllm/ds.h (ds.01 section). Rules: c/ABI.md.
 *
 * A tl_vec is {data, len, cap, elem}: len elements of elem bytes each are in
 * use, room for cap. Appending to a full vec allocates a new block with at
 * least twice the capacity, copies the len * elem bytes over, and frees the
 * old block, so n pushes cost O(n) copies in total (amortized O(1)). Every
 * allocation goes through the shared C test-support hooks (tl_alloc / tl_free), so
 * tests can make the k-th allocation fail; a failed growth must leave the
 * vec exactly as it was. rt.04 keeps one tl_vec of block ids per sequence.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm/ds.h"

#define TL_VEC_MIN_CAP 4u
/* Enough for any scalar element type: the vec does not know what it holds. */
#define TL_VEC_ALIGN 16u

/* Replace the block with one of exactly `cap` elements (cap >= len). On
 * failure nothing changes. */
static tl_status vec_regrow(tl_vec *v, size_t cap) {
    if (cap > SIZE_MAX / v->elem) {
        tl_set_last_error("tl_vec: capacity * elem overflows size_t");
        return TL_EINVAL;
    }
    void *p = tl_alloc(cap * v->elem, TL_VEC_ALIGN);
    if (p == NULL) return TL_ENOMEM; /* tl_alloc set the error slot */
    if (v->len) memcpy(p, v->data, v->len * v->elem);
    tl_free(v->data); /* only after the copy succeeded */
    v->data = p;
    v->cap = cap;
    return TL_OK;
}

tl_status tl_vec_init(tl_vec *v, size_t elem) {
    if (v == NULL || elem == 0) {
        tl_set_last_error("tl_vec_init: v is NULL or elem is 0");
        return TL_EINVAL;
    }
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
    v->elem = elem;
    return TL_OK;
}

tl_status tl_vec_reserve(tl_vec *v, size_t cap_min) {
    if (v == NULL || v->elem == 0) {
        tl_set_last_error("tl_vec_reserve: v is NULL or not initialized");
        return TL_EINVAL;
    }
    if (cap_min <= v->cap) return TL_OK; /* never shrinks */
    return vec_regrow(v, cap_min);
}

tl_status tl_vec_push(tl_vec *v, const void *x) {
    if (v == NULL || x == NULL || v->elem == 0) {
        tl_set_last_error("tl_vec_push: v or x is NULL, or v is not initialized");
        return TL_EINVAL;
    }
    if (v->len == v->cap) {
        if (v->cap > SIZE_MAX / 2) {
            tl_set_last_error("tl_vec_push: capacity overflow");
            return TL_ENOMEM;
        }
        size_t cap = v->cap ? 2 * v->cap : TL_VEC_MIN_CAP;
        tl_status s = vec_regrow(v, cap);
        if (s != TL_OK) return s;
    }
    memcpy((char *)v->data + v->len * v->elem, x, v->elem);
    v->len++;
    return TL_OK;
}

void *tl_vec_at(const tl_vec *v, size_t i) {
    if (v == NULL || i >= v->len) {
        tl_set_last_error("tl_vec_at: index out of range");
        return NULL;
    }
    return (char *)v->data + i * v->elem;
}

void tl_vec_free(tl_vec *v) {
    if (v == NULL) return;
    tl_free(v->data);
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
}
