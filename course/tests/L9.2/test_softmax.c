/* Course tests for L9.2 (C side): c/src/kernels/softmax.c against
 * tinyllm/softmax.h, under ASan and UBSan. Every case runs both versions,
 * the three-pass tl_softmax_f32 and the online tl_softmax_online_f32,
 * through one table of kernels, because they promise the same function.
 * Standalone cases cover row maxima, normalization, and non-finite inputs.
 */
#include <math.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

typedef tl_status (*softmax_fn)(const float *, float *, int64_t, int64_t);
static const softmax_fn KERNELS[2] = {tl_softmax_f32, tl_softmax_online_f32};

/* Float32 unit roundoff 2^-24: a sum of n terms rounds n times. */
#define U32 5.9604645e-8

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example: x = [1, 2, 3] gives m = 3,
     *      s = e^-2 + e^-1 + 1 = 1.5032147, y = [0.0900306, 0.2447285,
     *      0.6652410]. Both versions, so you and the tests agree on the
     *      definition before any edge case.
     * KIND: unit, smoke
     * CATCHES: s01, s02, m01
     * CHAPTER: L9.2 section 3 */
    const float x[3] = {1, 2, 3};
    const double want[3] = {0.09003057317038046, 0.24472847105479764, 0.6652409557748219};
    for (int f = 0; f < 2; f++) {
        float y[3] = {0};
        OL_EQ(KERNELS[f](x, y, 1, 3), TL_OK);
        for (int j = 0; j < 3; j++) OL_CLOSE(y[j], want[j], 1e-6, 1e-7);
    }
}

OL_TEST(online_rescales_when_the_max_grows) {
    /* WHY: the online pass sees an increasing row: every new element raises
     *      the maximum, so the running sum must be rescaled by e^(m_old -
     *      m_new) each time. Forgetting the rescale leaves s = 1 + 1 + ... and
     *      gives a uniform-looking answer. A decreasing row never rescales,
     *      so it guards the other branch.
     * KIND: unit
     * CATCHES: s03, s04
     * CHAPTER: L9.2 section 2 */
    const float up[4] = {-1, 0, 1, 2}, down[4] = {2, 1, 0, -1};
    const double e[4] = {0.03205860328008499, 0.08714431874203257, 0.23688281808991013, 0.6439142598879722};
    float y[4];
    OL_EQ(tl_softmax_online_f32(up, y, 1, 4), TL_OK);
    for (int j = 0; j < 4; j++) OL_CLOSE(y[j], e[j], 1e-6, 1e-7);
    OL_EQ(tl_softmax_online_f32(down, y, 1, 4), TL_OK);
    for (int j = 0; j < 4; j++) OL_CLOSE(y[j], e[3 - j], 1e-6, 1e-7);
}

OL_TEST(large_logits_stay_finite) {
    /* WHY: e^10000 overflows float32 (the largest finite e^x is at x =
     *      88.7). Subtracting the maximum first makes every exponent <= 0,
     *      so rows of +-1e4 are exact: [1e4, 1e4] is [0.5, 0.5] and
     *      [-1e4, 0] is [0, 1].
     * KIND: boundary
     * CATCHES: s05, s06
     * CHAPTER: L9.2 section 5, Pitfalls */
    const float x[2][2] = {{1e4f, 1e4f}, {-1e4f, 0.0f}};
    for (int f = 0; f < 2; f++) {
        float y[2][2];
        OL_EQ(KERNELS[f](&x[0][0], &y[0][0], 2, 2), TL_OK);
        OL_EQ(y[0][0], 0.5f);
        OL_EQ(y[0][1], 0.5f);
        OL_EQ(y[1][0], 0.0f);
        OL_EQ(y[1][1], 1.0f);
    }
}

OL_TEST(masked_entries_and_fully_masked_rows) {
    /* WHY: attention masks a key by giving it a score of -inf. A masked
     *      entry must get weight exactly 0, and a row whose every key is
     *      masked (a padded query) must give zeros, not 0/0 = NaN that then
     *      poisons the whole batch. In the online pass, -inf - -inf is NaN,
     *      so a -inf before the first finite value needs care.
     * KIND: boundary
     * CATCHES: s07, s08, s09
     * CHAPTER: L9.2 section 5, Pitfalls */
    const float x[3][3] = {{-INFINITY, 0.0f, 0.0f}, {-INFINITY, -INFINITY, -INFINITY}, {0.0f, -INFINITY, 0.0f}};
    for (int f = 0; f < 2; f++) {
        float y[3][3];
        memset(y, 0x7f, sizeof y); /* garbage that is not zero */
        OL_EQ(KERNELS[f](&x[0][0], &y[0][0], 3, 3), TL_OK);
        OL_EQ(y[0][0], 0.0f);
        OL_EQ(y[0][1], 0.5f);
        OL_EQ(y[0][2], 0.5f);
        for (int j = 0; j < 3; j++) OL_EQ(y[1][j], 0.0f);
        OL_EQ(y[2][0], 0.5f);
        OL_EQ(y[2][1], 0.0f);
        OL_EQ(y[2][2], 0.5f);
    }
}

OL_TEST(nan_propagates_to_its_row_only) {
    /* WHY: a NaN score is a bug upstream, and the contract makes it
     *      visible: its row becomes NaN. Rows are independent, so the next
     *      row is exact. Skipping NaN silently would hide the bug.
     * KIND: boundary
     * CATCHES: s10
     * CHAPTER: L9.2 section 4 */
    const float x[3][2] = {{0.0f, NAN}, {NAN, NAN}, {0.0f, 0.0f}};
    for (int f = 0; f < 2; f++) {
        float y[3][2];
        OL_EQ(KERNELS[f](&x[0][0], &y[0][0], 3, 2), TL_OK);
        OL_TRUE(isnan(y[0][0]) && isnan(y[0][1]));
        OL_TRUE(isnan(y[1][0]) && isnan(y[1][1]));
        OL_EQ(y[2][0], 0.5f);
        OL_EQ(y[2][1], 0.5f);
    }
}

OL_TEST(in_place) {
    /* WHY: the attention kernels and the Python backend normalize a score
     *      buffer in place (y == x). The contract allows it, so every pass
     *      must read x[j] before it writes y[j].
     * KIND: unit
     * CATCHES: s11
     * CHAPTER: L9.2 section 4 */
    for (int f = 0; f < 2; f++) {
        float x[3] = {1, 2, 3};
        OL_EQ(KERNELS[f](x, x, 1, 3), TL_OK);
        OL_CLOSE(x[0], 0.09003057317038046, 1e-6, 1e-7);
        OL_CLOSE(x[2], 0.6652409557748219, 1e-6, 1e-7);
    }
}

OL_TEST(bad_arguments_are_einval) {
    /* WHY: a negative dimension or a NULL buffer with work to do is a caller
     *      bug: TL_EINVAL with the slot naming the function. rows == 0 or
     *      cols == 0 is no work at all, and NULL is then fine.
     * KIND: boundary
     * CATCHES: s12
     * CHAPTER: L9.2 section 4 */
    float x[2] = {0, 0}, y[2];
    for (int f = 0; f < 2; f++) {
        OL_EQ(KERNELS[f](x, y, -1, 2), TL_EINVAL);
        OL_TRUE(strstr(tl_last_error(), "softmax") != NULL);
        OL_EQ(KERNELS[f](x, y, 1, -2), TL_EINVAL);
        OL_EQ(KERNELS[f](NULL, y, 1, 2), TL_EINVAL);
        OL_EQ(KERNELS[f](x, NULL, 1, 2), TL_EINVAL);
        OL_EQ(KERNELS[f](NULL, NULL, 0, 2), TL_OK);
        OL_EQ(KERNELS[f](NULL, NULL, 3, 0), TL_OK);
    }
}

/* Property: random rows with mixed magnitudes and some masked entries. */
static int rows_sum_to_one(ol_gen *g, int size) {
    int64_t cols = 1 + size;
    float x[512], y3[512], yo[512];
    double scale = ol_gen_f64(g, 0.1, 30.0);
    for (int64_t j = 0; j < cols; j++) {
        x[j] = (float)(scale * ol_gen_f64(g, -1, 1));
        if (ol_gen_int(g, 0, 9) == 0) x[j] = -INFINITY;
    }
    x[ol_gen_int(g, 0, cols - 1)] = (float)ol_gen_f64(g, -2, 2); /* at least one finite entry */
    if (tl_softmax_f32(x, y3, 1, cols) != TL_OK || tl_softmax_online_f32(x, yo, 1, cols) != TL_OK) return 0;
    double s3 = 0, so = 0, bound = 2.0 * (double)cols * U32 + 1e-7;
    for (int64_t j = 0; j < cols; j++) {
        if (!(y3[j] >= 0 && yo[j] >= 0)) return 0;
        if (x[j] == -INFINITY && (y3[j] != 0 || yo[j] != 0)) return 0;
        if (fabs((double)y3[j] - yo[j]) > 1e-5 * y3[j] + 1e-7) return 0; /* the two versions agree */
        s3 += y3[j], so += yo[j];
    }
    return fabs(s3 - 1.0) <= bound && fabs(so - 1.0) <= bound;
}

OL_TEST(rows_sum_to_one_and_versions_agree) {
    /* WHY: the defining property: every row is a probability distribution,
     *      non-negative and summing to 1 within the rounding of a cols-term
     *      sum (about cols * 2^-24), with exact zeros on masked entries; and
     *      the online version agrees with the three-pass one, element by
     *      element, to float32 tolerance.
     * KIND: property, differential
     * CATCHES: s01, s02, s03, s04, m01, m02
     * CHAPTER: L9.2 section 2 */
    OL_CHECK_PROP(rows_sum_to_one, 200, 500);
}

int main(void) { return OL_RUN_ALL(); }
