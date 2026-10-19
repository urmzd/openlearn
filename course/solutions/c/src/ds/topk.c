/* c/src/ds/topk.c (ds.04): top-k by a binary min-heap of size k.
 * Contract: tinyllm/topk.h. This optional C unit has a standalone test
 * binary; the Rust sampler has its own implementation.
 *
 * The heap lives in the caller's idx and val arrays: no allocation. Its
 * root is the WORST kept entry, so one comparison decides whether a new
 * value enters. "Worse" is a total order: a smaller value, or an equal
 * value with a larger index. That one rule gives both promises of the
 * contract: among equal values the lower index wins the last place, and
 * the sorted output lists equal values by ascending index.
 */
#include <math.h>
#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/topk.h"

/* 1 when entry (va, ia) ranks below entry (vb, ib). */
static int worse(float va, int32_t ia, float vb, int32_t ib) {
    return va < vb || (va == vb && ia > ib);
}

/* Restores the heap property below position i in a heap of `size`
 * entries: the parent is never better than its children, so the root is
 * the worst entry. */
static void sift_down(int32_t *idx, float *val, int64_t size, int64_t i) {
    for (;;) {
        int64_t l = 2 * i + 1, r = l + 1, w = i;
        if (l < size && worse(val[l], idx[l], val[w], idx[w])) w = l;
        if (r < size && worse(val[r], idx[r], val[w], idx[w])) w = r;
        if (w == i) return;
        float tv = val[i];
        int32_t ti = idx[i];
        val[i] = val[w], idx[i] = idx[w];
        val[w] = tv, idx[w] = ti;
        i = w;
    }
}

/* Moves the entry at position i up while it is worse than its parent. */
static void sift_up(int32_t *idx, float *val, int64_t i) {
    while (i > 0) {
        int64_t p = (i - 1) / 2;
        if (!worse(val[i], idx[i], val[p], idx[p])) return;
        float tv = val[i];
        int32_t ti = idx[i];
        val[i] = val[p], idx[i] = idx[p];
        val[p] = tv, idx[p] = ti;
        i = p;
    }
}

tl_status tl_topk_f32(const float *x, int64_t n, int64_t k, int32_t *idx, float *val) {
    /* 1. Validate everything before writing anything. */
    if (n < 0 || k < 0 || k > n) {
        tl_set_last_error("tl_topk_f32: need 0 <= k <= n");
        return TL_EINVAL;
    }
    if (n > INT32_MAX) {
        tl_set_last_error("tl_topk_f32: n does not fit an int32 index");
        return TL_EINVAL;
    }
    if (n > 0 && x == NULL) {
        tl_set_last_error("tl_topk_f32: x is NULL");
        return TL_EINVAL;
    }
    if (k > 0 && (idx == NULL || val == NULL)) {
        tl_set_last_error("tl_topk_f32: idx or val is NULL");
        return TL_EINVAL;
    }
    for (int64_t i = 0; i < n; i++) {
        if (isnan(x[i])) {
            tl_set_last_error("tl_topk_f32: x contains NaN");
            return TL_EINVAL;
        }
    }
    if (k == 0) return TL_OK;

    /* 2. Fill the heap with the first k entries, then let each later entry
     *    replace the root when it ranks above it. x is scanned in index
     *    order, so a later entry equal to the root ranks BELOW it (larger
     *    index) and stays out: ties go to the lower index. */
    int64_t size = 0;
    for (int64_t i = 0; i < n; i++) {
        if (size < k) {
            val[size] = x[i], idx[size] = (int32_t)i;
            sift_up(idx, val, size);
            size++;
        } else if (worse(val[0], idx[0], x[i], (int32_t)i)) {
            val[0] = x[i], idx[0] = (int32_t)i;
            sift_down(idx, val, k, 0);
        }
    }

    /* 3. Heap sort in place: move the worst entry to the end, shrink, and
     *    repeat. What is left is best first: descending value, and ascending
     *    index among equal values. */
    for (int64_t end = k - 1; end > 0; end--) {
        float tv = val[0];
        int32_t ti = idx[0];
        val[0] = val[end], idx[0] = idx[end];
        val[end] = tv, idx[end] = ti;
        sift_down(idx, val, end, 0);
    }
    return TL_OK;
}
