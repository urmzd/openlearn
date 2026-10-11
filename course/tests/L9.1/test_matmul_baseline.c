/* Baseline contract tests for the optional standalone L9.1 C kernel.
 * The Python core matmul is M03.1. These cases exercise row-major
 * alpha/beta behavior before the tiled L9.1 edge cases.
 */
#include <math.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_test.h"

/* The chapter's worked example (section 3):
 *   A = [[1, 2, 3],      B = [[ 7,  8],      A @ B = [[ 58,  64],
 *        [4, 5, 6]]           [ 9, 10],               [139, 154]]
 *                             [11, 12]]
 * Every product and sum is a small integer, so float32 holds it exactly. */
static const float HA[6] = {1, 2, 3, 4, 5, 6};
static const float HB[6] = {7, 8, 9, 10, 11, 12};
static const float HBT[6] = {7, 9, 11, 8, 10, 12}; /* B stored transposed: N x K */
static const float HC[4] = {58, 64, 139, 154};

OL_TEST(baseline_row_major) {
    /* WHY: the chapter's worked example, row-major with no padding:
     *      C[i][j] = sum over k of A[i*3 + k] * B[k*2 + j]. You and the test
     *      agree on the definition before anything harder.
     * KIND: unit, smoke
     * CATCHES: s02, m01, m02, m03
     * CHAPTER: L9.1 section 3 */
    float C[4] = {0};
    OL_EQ(tl_matmul_f32(HA, HB, C, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    for (int i = 0; i < 4; i++) OL_EQ(C[i], HC[i]);
}

OL_TEST(trans_b_reads_rows_of_b) {
    /* WHY: a Linear layer stores its weight as [out, in] = N x K, and L0.0
     *      standalone C callers may store their weights as [out, in].
     *      trans_b = 1 must give the same product with B stored transposed.
     * KIND: unit
     * CATCHES: s03
     * CHAPTER: L9.1 section 2 */
    float C[4] = {0};
    OL_EQ(tl_matmul_f32(HA, HBT, C, 2, 2, 3, 3, 3, 2, 1.0f, 0.0f, 1, NULL), TL_OK);
    for (int i = 0; i < 4; i++) OL_EQ(C[i], HC[i]);
}

OL_TEST(beta_zero_ignores_garbage_in_c) {
    /* WHY: callers hand in an uninitialized output buffer with beta = 0. Any
     *      code that computes beta * C[i][j] turns a NaN already in C into NaN
     *      in the result, because 0 * NaN is NaN.
     * KIND: boundary
     * CATCHES: s07
     * CHAPTER: L9.1 section 5, Pitfalls */
    float C[4] = {NAN, INFINITY, -NAN, NAN};
    OL_EQ(tl_matmul_f32(HA, HB, C, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    for (int i = 0; i < 4; i++) OL_EQ(C[i], HC[i]);
}

OL_TEST(alpha_and_beta_scale_and_accumulate) {
    /* WHY: C = alpha * A @ B + beta * C is the GEMM convention every BLAS
     *      uses; beta = 1 accumulates into C (a residual add), alpha scales.
     *      Values are chosen so every step is exact in float32.
     * KIND: unit
     * CATCHES: m01, m03
     * CHAPTER: L9.1 section 2 */
    float C[4] = {2, 4, 6, 8};
    OL_EQ(tl_matmul_f32(HA, HB, C, 2, 2, 3, 3, 2, 2, 2.0f, 0.5f, 0, NULL), TL_OK);
    const float want[4] = {2 * 58 + 1, 2 * 64 + 2, 2 * 139 + 3, 2 * 154 + 4};
    for (int i = 0; i < 4; i++) OL_EQ(C[i], want[i]);
}

OL_TEST(leading_dimensions_skip_padding) {
    /* WHY: a leading dimension is the distance between two rows, which is
     *      larger than the row length when a matrix is a view into a wider
     *      buffer (one head of a fused QKV projection). Using N or K instead
     *      of ldc or lda reads the padding and writes over it.
     * KIND: unit
     * CATCHES: s04, s06
     * CHAPTER: L9.1 section 3 */
    const float P = 1e30f; /* padding sentinel: a product with it is obviously wrong */
    const float A[2 * 5] = {1, 2, 3, P, P, 4, 5, 6, P, P};                /* lda = 5 */
    const float B[3 * 4] = {7, 8, P, P, 9, 10, P, P, 11, 12, P, P};       /* ldb = 4 */
    float C[2 * 3] = {-1, -1, -1, -1, -1, -1};                            /* ldc = 3 */
    OL_EQ(tl_matmul_f32(A, B, C, 2, 2, 3, 5, 4, 3, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_EQ(C[0], 58.0f);
    OL_EQ(C[1], 64.0f);
    OL_EQ(C[2], -1.0f); /* padding column of C is untouched */
    OL_EQ(C[3], 139.0f);
    OL_EQ(C[4], 154.0f);
    OL_EQ(C[5], -1.0f);
}

OL_TEST(empty_dimensions) {
    /* WHY: an empty batch (M = 0) or an empty output (N = 0) is a no-op that
     *      must not touch C; K = 0 is an empty sum, so C becomes beta * C.
     *      A prompt of length 0 and a model with no heads hit these.
     * KIND: boundary
     * CATCHES: s08
     * CHAPTER: L9.1 section 4 */
    float C[4] = {1, 2, 3, 4};
    OL_EQ(tl_matmul_f32(NULL, NULL, C, 0, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_EQ(tl_matmul_f32(NULL, NULL, C, 2, 0, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_EQ(C[0], 1.0f);
    OL_EQ(C[3], 4.0f);
    OL_EQ(tl_matmul_f32(NULL, NULL, C, 2, 2, 0, 0, 2, 2, 1.0f, 2.0f, 0, NULL), TL_OK);
    OL_EQ(C[0], 2.0f);
    OL_EQ(C[3], 8.0f);
    OL_EQ(tl_matmul_f32(NULL, NULL, C, 2, 2, 0, 0, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_EQ(C[0], 0.0f);
    OL_EQ(C[3], 0.0f);
}

OL_TEST(bad_arguments_are_einval) {
    /* WHY: a negative dimension, a leading dimension shorter than its row, or
     *      a NULL pointer that would be read is a caller bug. The contract
     *      says TL_EINVAL with the error slot naming the function, and C left
     *      exactly as it was, instead of a crash three layers later.
     * KIND: boundary
     * CATCHES: s09
     * CHAPTER: L9.1 section 4 */
    float C[4] = {1, 2, 3, 4};
    OL_EQ(tl_matmul_f32(HA, HB, C, -1, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL);
    OL_TRUE(strstr(tl_last_error(), "tl_matmul_f32") != NULL);
    OL_EQ(tl_matmul_f32(HA, HB, C, 2, 2, -3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL);
    OL_EQ(tl_matmul_f32(HA, HB, C, 2, 2, 3, 2, 2, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL);  /* lda < K */
    OL_EQ(tl_matmul_f32(HA, HB, C, 2, 2, 3, 3, 1, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL);  /* ldb < N */
    OL_EQ(tl_matmul_f32(HA, HBT, C, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 1, NULL), TL_EINVAL); /* ldb < K, trans_b */
    OL_EQ(tl_matmul_f32(HA, HB, C, 2, 2, 3, 3, 2, 1, 1.0f, 0.0f, 0, NULL), TL_EINVAL);  /* ldc < N */
    OL_EQ(tl_matmul_f32(NULL, HB, C, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL);
    OL_EQ(tl_matmul_f32(HA, NULL, C, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL);
    OL_EQ(tl_matmul_f32(HA, HB, NULL, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL);
    const float before[4] = {1, 2, 3, 4};
    OL_TRUE(memcmp(C, before, sizeof C) == 0);
}
