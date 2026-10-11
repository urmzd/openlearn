/* Course tests for ds.04 (C side): c/src/ds/topk.c against tinyllm/topk.h,
 * under ASan and UBSan with the counting allocator, so a read one past x or
 * a write one past idx stops the run with a report, and any allocation
 * that is not freed fails the case (the contract says it allocates
 * nothing). Standalone vectors check the same ordering contract as L8.1.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

/* The chapter's worked example (section 3), the logits of the sampling
 * spec's own worked example: x = [1, 3, 2, 3, -1], k = 3. The two 3s tie;
 * the lower index (1) comes first. */
OL_TEST(hand_example) {
    /* WHY: you and the test agree on the definition before anything harder:
     *      the k largest values, best first, equal values by ascending
     *      index. These are the logits of spec/sampling.md's worked example.
     * KIND: unit, smoke
     * CATCHES: s05, s06, m01
     * CHAPTER: ds.04 section 3 */
    const float x[5] = {1, 3, 2, 3, -1};
    int32_t idx[3] = {-7, -7, -7};
    float val[3] = {0};
    OL_EQ(tl_topk_f32(x, 5, 3, idx, val), TL_OK);
    const int32_t want_i[3] = {1, 3, 2};
    const float want_v[3] = {3, 3, 2};
    for (int i = 0; i < 3; i++) {
        OL_EQ(idx[i], want_i[i]);
        OL_EQ(val[i], want_v[i]);
    }
}

OL_TEST(tie_at_the_cut_goes_to_the_lower_index) {
    /* WHY: when the k-th place is contested by equal values, the contract
     *      (and spec/sampling.md step 6) keeps the lower index. A heap that
     *      replaces its root on >= instead of > keeps the LAST of the tied
     *      entries, and the Rust sampler would then disagree with your
     *      Python sampler on which token survives top-k.
     * KIND: boundary
     * CATCHES: s01, m02
     * CHAPTER: ds.04 section 5, Pitfalls */
    const float x[6] = {5, 1, 1, 1, 1, 0};
    int32_t idx[2];
    float val[2];
    OL_EQ(tl_topk_f32(x, 6, 2, idx, val), TL_OK);
    OL_EQ(idx[0], 0);
    OL_EQ(idx[1], 1);
    OL_EQ(val[1], 1.0f);
}

OL_TEST(equal_values_are_listed_by_ascending_index) {
    /* WHY: the output order is part of the contract, not only the kept set:
     *      the sampler walks the kept ids in (value desc, id asc) order for
     *      top-p (step 7). All-equal input exercises every comparison in the
     *      heap on the index alone.
     * KIND: boundary
     * CATCHES: s02, s05, m03
     * CHAPTER: ds.04 section 2 */
    float x[9];
    for (int i = 0; i < 9; i++) x[i] = 0.5f;
    int32_t idx[9];
    float val[9];
    OL_EQ(tl_topk_f32(x, 9, 9, idx, val), TL_OK);
    for (int i = 0; i < 9; i++) OL_EQ(idx[i], i);
    OL_EQ(tl_topk_f32(x, 9, 4, idx, val), TL_OK);
    for (int i = 0; i < 4; i++) OL_EQ(idx[i], i);
}

OL_TEST(k_zero_and_k_equal_n) {
    /* WHY: k == 0 is "top-k off" in a request that still calls the kernel,
     *      and must write nothing; k == n keeps everything and is a full
     *      sort. Both edges of the valid range.
     * KIND: boundary
     * CATCHES: s07, m02
     * CHAPTER: ds.04 section 4 */
    const float x[4] = {0.25f, -2.0f, 7.0f, 0.25f};
    int32_t idx[4] = {-9, -9, -9, -9};
    float val[4] = {-9, -9, -9, -9};
    OL_EQ(tl_topk_f32(x, 4, 0, idx, val), TL_OK);
    OL_EQ(idx[0], -9);
    OL_EQ(val[0], -9.0f);
    OL_EQ(tl_topk_f32(x, 4, 4, idx, val), TL_OK);
    const int32_t want[4] = {2, 0, 3, 1};
    for (int i = 0; i < 4; i++) OL_EQ(idx[i], want[i]);
    OL_EQ(tl_topk_f32(NULL, 0, 0, NULL, NULL), TL_OK); /* an empty input */
}

OL_TEST(minus_infinity_is_an_ordinary_value) {
    /* WHY: a masked logit (L8.7 constrained decoding) is -inf. It is a
     *      value, not an error: it ranks below every finite value and ties
     *      with other -inf by index. Rejecting it with an isfinite() check
     *      would break every masked request.
     * KIND: boundary
     * CATCHES: s09
     * CHAPTER: ds.04 section 2 */
    const float x[5] = {-INFINITY, 2.0f, -INFINITY, -1.0f, -INFINITY};
    int32_t idx[4];
    float val[4];
    OL_EQ(tl_topk_f32(x, 5, 4, idx, val), TL_OK);
    OL_EQ(idx[0], 1);
    OL_EQ(idx[1], 3);
    OL_EQ(idx[2], 0);
    OL_EQ(idx[3], 2);
    OL_TRUE(isinf(val[3]) && val[3] < 0);
}

OL_TEST(nan_and_bad_k_are_einval_and_write_nothing) {
    /* WHY: NaN has no place in an order (every comparison with it is
     *      false), so the contract rejects it; k > n or k < 0 would read or
     *      write past a buffer. A rejected call must leave idx and val as
     *      they were: the check comes before the first write.
     * KIND: boundary
     * CATCHES: s03, s04, s08
     * CHAPTER: ds.04 section 4 */
    const float x[4] = {1, 2, NAN, 4};
    const float y[3] = {1, 2, 3};
    int32_t idx[4] = {-5, -5, -5, -5};
    float val[4] = {-5, -5, -5, -5};
    OL_EQ(tl_topk_f32(x, 4, 2, idx, val), TL_EINVAL);
    OL_TRUE(strstr(tl_last_error(), "tl_topk_f32") != NULL);
    OL_EQ(tl_topk_f32(y, 3, 4, idx, val), TL_EINVAL);
    OL_EQ(tl_topk_f32(y, 3, -1, idx, val), TL_EINVAL);
    OL_EQ(tl_topk_f32(y, 3, 2, NULL, val), TL_EINVAL);
    OL_EQ(tl_topk_f32(NULL, 3, 2, idx, val), TL_EINVAL);
    for (int i = 0; i < 4; i++) {
        OL_EQ(idx[i], -5);
        OL_EQ(val[i], -5.0f);
    }
}

/* Property: for random inputs with many ties, the output equals the first
 * k entries of a full sort by (value desc, index asc). */
static int cmp_desc(const void *a, const void *b, const float *x) {
    int32_t i = *(const int32_t *)a, j = *(const int32_t *)b;
    if (x[i] != x[j]) return x[i] > x[j] ? -1 : 1;
    return i < j ? -1 : 1;
}
static const float *g_x;
static int cmp_ctx(const void *a, const void *b) { return cmp_desc(a, b, g_x); }

static int matches_full_sort(ol_gen *g, int size) {
    int64_t n = 1 + size;
    float x[300];
    int32_t order[300], idx[300];
    float val[300];
    for (int64_t i = 0; i < n; i++) x[i] = (float)ol_gen_int(g, -4, 4) * 0.5f; /* 17 values: many ties */
    for (int64_t i = 0; i < n; i++) order[i] = (int32_t)i;
    g_x = x;
    qsort(order, (size_t)n, sizeof order[0], cmp_ctx);
    int64_t k = ol_gen_int(g, 0, n);
    if (tl_topk_f32(x, n, k, idx, val) != TL_OK) return 0;
    for (int64_t i = 0; i < k; i++)
        if (idx[i] != order[i] || val[i] != x[order[i]]) return 0;
    return 1;
}

OL_TEST(matches_a_full_sort) {
    /* WHY: the heap must agree with the obvious O(n log n) answer, sort
     *      everything and take the first k, for every n and k, including
     *      the heap shapes (one child, a full last level) that only random
     *      sizes reach. Values are drawn from 17 levels so ties are common.
     * KIND: property, differential
     * CATCHES: s02, s06, m01, m02, m03
     * CHAPTER: ds.04 section 4 */
    OL_CHECK_PROP(matches_full_sort, 300, 250);
}

int main(void) { return OL_RUN_ALL(); }
