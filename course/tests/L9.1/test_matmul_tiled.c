/* Course tests for L9.1 (C side): the tiled, packed, batch-invariant
 * c/src/kernels/matmul.c against tinyllm/matmul.h, under ASan and UBSan
 * (and ThreadSanitizer for the pool case). M03.1's contract still holds
 * (its tests are this module's regression); these tests add the shapes
 * that cross tile edges, the k-blocking, batch invariance, and threads.
 * Parity with the Python oracle is represented by checked-in fixture files.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_invariance.h"
#include "ol_prop.h"
#include "ol_test.h"

/* float64 reference: alpha * A @ op(B) + beta * C0, and the frozen bound
 * |c - c64| <= sqrt(K) * (1e-6 + 1e-5 |c64|) of close.py. */
static int close_to_reference(const float *A, const float *B, const float *C0, const float *C, int64_t M,
                              int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc, float alpha,
                              float beta, int trans_b) {
    double s = sqrt((double)(K > 0 ? K : 1));
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double acc = 0;
            for (int64_t k = 0; k < K; k++)
                acc += (double)A[i * lda + k] * (double)(trans_b ? B[j * ldb + k] : B[k * ldb + j]);
            double want = alpha * acc + (beta == 0 ? 0.0 : beta * (double)C0[i * ldc + j]);
            double got = C[i * ldc + j];
            if (!(fabs(got - want) <= s * (1e-6 + 1e-5 * fabs(want)))) {
                printf("    C[%lld][%lld] = %.9g, want %.9g (M,N,K = %lld,%lld,%lld)\n", (long long)i,
                       (long long)j, got, want, (long long)M, (long long)N, (long long)K);
                return 0;
            }
        }
    return 1;
}

static void fill(ol_gen *g, float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = ol_gen_f32(g, -1, 1);
}

OL_TEST(hand_example) {
    /* WHY: M03.1's worked example still holds after the rewrite:
     *      [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]],
     *      a 2 x 2 x 3 product that fits inside one zero-padded 4 x 16
     *      micro-tile, so it exercises the padding and the edge store.
     * KIND: unit, smoke
     * CATCHES: s02, m01, m02
     * CHAPTER: L9.1 section 3 */
    const float A[6] = {1, 2, 3, 4, 5, 6}, B[6] = {7, 8, 9, 10, 11, 12};
    float C[4] = {0};
    OL_EQ(tl_matmul_f32(A, B, C, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_EQ(C[0], 58.0f);
    OL_EQ(C[1], 64.0f);
    OL_EQ(C[2], 139.0f);
    OL_EQ(C[3], 154.0f);
}

OL_TEST(shapes_across_tile_edges) {
    /* WHY: a tiled kernel has edge cases a triple loop does not: M, N, K
     *      just below, at, and just above the micro-tile (4 x 16), the
     *      packed blocks (64 rows, 128 columns), and the k-block (128).
     *      Both trans_b settings, against a float64 reference within the
     *      frozen sqrt(K) bound.
     * KIND: differential
     * CATCHES: s01, s03, s05, m03
     * CHAPTER: L9.1 section 4 */
    static const int64_t D[][3] = {{1, 1, 1},   {3, 15, 127}, {4, 16, 128}, {5, 17, 129},
                                   {65, 1, 7},  {1, 129, 3},  {7, 33, 257}, {64, 128, 1}};
    ol_gen g = {11};
    for (size_t t = 0; t < sizeof D / sizeof D[0]; t++)
        for (int tb = 0; tb < 2; tb++) {
            int64_t M = D[t][0], N = D[t][1], K = D[t][2];
            float *A = malloc(sizeof(float) * (size_t)(M * K)), *B = malloc(sizeof(float) * (size_t)(K * N));
            float *C = malloc(sizeof(float) * (size_t)(M * N));
            fill(&g, A, M * K), fill(&g, B, K * N);
            int ok = tl_matmul_f32(A, B, C, M, N, K, K, tb ? K : N, N, 1.0f, 0.0f, tb, NULL) == TL_OK &&
                     close_to_reference(A, B, C, C, M, N, K, K, tb ? K : N, N, 1.0f, 0.0f, tb);
            free(A), free(B), free(C);
            OL_TRUE(ok);
        }
}

OL_TEST(beta_applies_once_across_k_blocks) {
    /* WHY: with K = 300 the sum runs over three k-blocks (128, 128, 44).
     *      beta scales the old C exactly once, in the first block; later
     *      blocks add. Applying it per block gives beta^3 * C, and
     *      overwriting per block keeps only the last 44 products. alpha = 2
     *      and beta = 0.5 are exact in binary.
     * KIND: unit
     * CATCHES: s01, s05
     * CHAPTER: L9.1 section 5, Pitfalls */
    enum { M = 6, N = 20, K = 300 };
    static float A[M * K], B[K * N], C0[M * N], C[M * N];
    ol_gen g = {12};
    fill(&g, A, M * K), fill(&g, B, K * N), fill(&g, C0, M * N);
    memcpy(C, C0, sizeof C);
    OL_EQ(tl_matmul_f32(A, B, C, M, N, K, K, N, N, 2.0f, 0.5f, 0, NULL), TL_OK);
    OL_TRUE(close_to_reference(A, B, C0, C, M, N, K, K, N, N, 2.0f, 0.5f, 0));
    for (int i = 0; i < M * N; i++) C[i] = NAN; /* beta == 0: C is never read */
    OL_EQ(tl_matmul_f32(A, B, C, M, N, K, K, N, N, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_TRUE(close_to_reference(A, B, C0, C, M, N, K, K, N, N, 1.0f, 0.0f, 0));
}

OL_TEST(views_and_padding_untouched) {
    /* WHY: packing reads A and B through lda and ldb, and the edge store
     *      writes only the real corner of a micro-tile through ldc. A 5 x 3
     *      view of C inside a 5 x 20 buffer must leave the other 17 columns
     *      exactly as they were, even though the micro-tile is 16 wide.
     * KIND: unit
     * CATCHES: s02, s04, s06
     * CHAPTER: L9.1 section 2 */
    enum { M = 5, N = 3, K = 9, LDA = 12, LDB = 7, LDC = 20 };
    static float A[M * LDA], B[K * LDB], C0[M * LDC], C[M * LDC];
    ol_gen g = {13};
    fill(&g, A, M * LDA), fill(&g, B, K * LDB), fill(&g, C0, M * LDC);
    memcpy(C, C0, sizeof C);
    OL_EQ(tl_matmul_f32(A, B, C, M, N, K, LDA, LDB, LDC, 1.0f, 1.0f, 0, NULL), TL_OK);
    OL_TRUE(close_to_reference(A, B, C0, C, M, N, K, LDA, LDB, LDC, 1.0f, 1.0f, 0));
    for (int i = 0; i < M; i++)
        for (int j = N; j < LDC; j++) OL_EQ(C[i * LDC + j], C0[i * LDC + j]);
}

OL_TEST(empty_and_invalid_like_v0) {
    /* WHY: the M03.1 contract carries over unchanged: M or N = 0 is a no-op,
     *      K = 0 gives beta * C (zeros when beta = 0, even over NaN), and a
     *      short leading dimension is TL_EINVAL with C untouched.
     * KIND: boundary
     * CATCHES: s07, s08, s09
     * CHAPTER: L9.1 section 4 */
    float C[4] = {1, 2, NAN, 4};
    OL_EQ(tl_matmul_f32(NULL, NULL, C, 0, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_EQ(C[0], 1.0f);
    OL_EQ(tl_matmul_f32(NULL, NULL, C, 2, 2, 0, 0, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK);
    for (int i = 0; i < 4; i++) OL_EQ(C[i], 0.0f);
    float D[4] = {1, 2, 3, 4};
    OL_EQ(tl_matmul_f32(NULL, NULL, D, 2, 2, 0, 0, 2, 2, 1.0f, 2.0f, 0, NULL), TL_OK);
    OL_EQ(D[3], 8.0f);
    const float A[6] = {1, 2, 3, 4, 5, 6}, B[6] = {7, 8, 9, 10, 11, 12};
    OL_EQ(tl_matmul_f32(A, B, D, 2, 2, 3, 2, 2, 2, 1.0f, 0.0f, 0, NULL), TL_EINVAL); /* lda < K */
    OL_TRUE(strstr(tl_last_error(), "tl_matmul_f32") != NULL);
    OL_EQ(D[3], 8.0f);
    float E[4] = {NAN, NAN, NAN, NAN};
    OL_EQ(tl_matmul_f32(A, B, E, 2, 2, 3, 3, 2, 2, 1.0f, 0.0f, 0, NULL), TL_OK); /* beta 0 over NaN */
    OL_EQ(E[0], 58.0f);
}

/* rows -> rows of x @ W^T with W [N, K] as a Linear layer stores it. */
enum { BK = 300, BN = 40 };
static float g_w[BN * BK];
static void linear_rows(void *ctx, const float *x, int64_t m, float *out) {
    (void)ctx;
    tl_matmul_f32(x, g_w, out, m, BN, BK, BK, BK, BN, 1.0f, 0.0f, 1, NULL);
}

OL_TEST(batch_invariant_rows) {
    /* WHY: the engine batches sequences (L10.2) and splits prompts into
     *      chunks (L10.3); greedy output must not change with the batch. So
     *      each row of a product must have the same bits computed alone or
     *      inside any batch at any position (c/ABI.md rule 10). K = 300
     *      spans three k-blocks; 16 rows cover every position in the 4-row
     *      micro-tile and both sides of a block edge.
     * KIND: property
     * CATCHES: s10
     * CHAPTER: L9.1 section 2 */
    static float x[16 * BK];
    ol_gen g = {14};
    fill(&g, g_w, BN * BK), fill(&g, x, 16 * BK);
    float probe[BN];
    OL_EQ(tl_matmul_f32(x, g_w, probe, 1, BN, BK, BK, BK, BN, 1.0f, 0.0f, 1, NULL), TL_OK);
    OL_BATCH_INVARIANT(linear_rows, NULL, x, 16, BK, BN);
}

OL_TEST(batch_invariant_at_1_7_37) {
    /* WHY: the catalog's three batch sizes: row i of a 37-row product (the
     *      rows span packed blocks of 64 and micro-tiles of 4) equals that
     *      row computed with M = 1 and inside a 7-row batch, bit for bit.
     * KIND: property
     * CATCHES: s10
     * CHAPTER: L9.1 section 4 */
    static float x[37 * BK], all[37 * BN], seven[7 * BN], one[BN];
    ol_gen g = {15};
    fill(&g, g_w, BN * BK), fill(&g, x, 37 * BK);
    OL_EQ(tl_matmul_f32(x, g_w, all, 37, BN, BK, BK, BK, BN, 1.0f, 0.0f, 1, NULL), TL_OK);
    for (int i = 0; i < 37; i++) {
        linear_rows(NULL, x + i * BK, 1, one);
        OL_BITS_EQ_F32(one, all + i * BN, BN);
    }
    linear_rows(NULL, x + 30 * BK, 7, seven);
    OL_BITS_EQ_F32(seven, all + 30 * BN, 7 * BN);
}

OL_TEST(pool_result_equals_serial_bitwise) {
    /* WHY: with an rt.03 pool, column strips run on different threads; each
     *      strip is computed by one worker with the same arithmetic, so 1
     *      and 4 threads give the same bits. N = 300 makes three strips,
     *      the last one partial. Runs under ThreadSanitizer too.
     * KIND: property
     * CATCHES: s11
     * CHAPTER: L9.1 section 2 */
    enum { M = 9, N = 300, K = 140 };
    static float A[M * K], B[K * N], C1[M * N], C4[M * N];
    ol_gen g = {16};
    fill(&g, A, M * K), fill(&g, B, K * N);
    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(4, &p), TL_OK);
    tl_status st = tl_matmul_f32(A, B, C4, M, N, K, K, N, N, 1.0f, 0.0f, 0, p);
    tl_pool_destroy(p);
    OL_EQ(st, TL_OK);
    OL_EQ(tl_matmul_f32(A, B, C1, M, N, K, K, N, N, 1.0f, 0.0f, 0, NULL), TL_OK);
    OL_BITS_EQ_F32(C4, C1, M * N);
}

int main(void) { return OL_RUN_ALL(); }
