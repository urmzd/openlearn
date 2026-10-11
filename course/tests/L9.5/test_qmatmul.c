/* Course tests for L9.5 (C side): c/src/kernels/qmatmul.c against
 * tinyllm/qmatmul.h and the int4 layout of formats/safetensors.md, under
 * ASan and UBSan. The differential tests dequantize the weight to float32
 * and multiply with your L9.1 tl_matmul_f32, so the fused kernel is held to
 * "dequantize, then the ordinary matmul". Deterministic fixture vectors
 * keep this standalone C implementation comparable with the Python oracle.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_invariance.h"
#include "ol_prop.h"
#include "ol_test.h"

/* The chapter's worked example (section 3): W is 2 x 4, group 2.
 *   row 0: q = [1, -2 | 3, -8], scales [0.5 | 2]
 *   row 1: q = [7,  0 | -1, 4], scales [1 | 0.25]
 * Bytes, low nibble first: row 0 = {0xE1, 0x83}, row 1 = {0x07, 0x4F}.
 * f16 scales: 0.5 = 0x3800, 2 = 0x4000, 1 = 0x3C00, 0.25 = 0x3400. */
static const uint8_t HQ[4] = {0xE1, 0x83, 0x07, 0x4F};
static const uint16_t HS[4] = {0x3800, 0x4000, 0x3C00, 0x3400};
static const float HX[4] = {1, 2, 3, 4};

OL_TEST(hand_example) {
    /* WHY: the worked example: y0 = 0.5 (1 - 4) + 2 (9 - 32) = -47.5 and
     *      y1 = 1 (7 + 0) + 0.25 (-3 + 16) = 10.25. Every value is exact in
     *      float32, so the test is exact: you and the tests agree on the
     *      nibble order, the sign, and which scale covers which columns.
     * KIND: unit, smoke
     * CATCHES: s01, s02, s03, s04
     * CHAPTER: L9.5 section 3 */
    float y[2] = {0};
    OL_EQ(tl_matmul_q4_f32(HX, HQ, HS, y, 1, 2, 4, 2, NULL), TL_OK);
    OL_EQ(y[0], -47.5f);
    OL_EQ(y[1], 10.25f);
}

OL_TEST(q8_hand_example) {
    /* WHY: the same weights as int8 with one scale per output row:
     *      0.5 (1 - 4 + 9 - 32) = -13 and 0.25 (7 + 0 - 3 + 16) = 5.
     * KIND: unit
     * CATCHES: s08
     * CHAPTER: L9.5 section 3 */
    const int8_t q[8] = {1, -2, 3, -8, 7, 0, -1, 4};
    const float s[2] = {0.5f, 0.25f};
    float y[2] = {0};
    OL_EQ(tl_matmul_q8_f32(HX, q, s, y, 1, 2, 4, NULL), TL_OK);
    OL_EQ(y[0], -13.0f);
    OL_EQ(y[1], 5.0f);
}

OL_TEST(every_nibble_code) {
    /* WHY: all 16 codes decode to their two's complement values: 0x0..0x7
     *      are 0..7 and 0x8..0xF are -8..-1, in the low and in the high
     *      nibble. A one-hot x reads one weight at a time. An unsigned
     *      nibble (0..15) or a nibble order swap passes many random tests
     *      only "approximately wrong"; this one is exact.
     * KIND: boundary
     * CATCHES: s01, s02
     * CHAPTER: L9.5 section 2 */
    uint8_t q[8];
    for (int b = 0; b < 8; b++) q[b] = (uint8_t)((2 * b) | ((2 * b + 1) << 4)); /* codes 0..15 in order */
    const uint16_t one = 0x3C00;
    for (int k = 0; k < 16; k++) {
        float x[16] = {0}, y = 99;
        x[k] = 1.0f;
        OL_EQ(tl_matmul_q4_f32(x, q, &one, &y, 1, 1, 16, 16, NULL), TL_OK);
        OL_EQ(y, (float)(k < 8 ? k : k - 16));
    }
}

OL_TEST(f16_scales_decode_exactly) {
    /* WHY: the scales are IEEE half precision, not bfloat16: the bits
     *      0x0001 are the smallest f16 subnormal, 2^-24, and 0x3555 is
     *      0.33325195. Reading them as the top half of a float32 gives
     *      wrong magnitudes. One weight of 7 against x = 1.
     * KIND: boundary
     * CATCHES: s05
     * CHAPTER: L9.5 section 2 */
    const uint8_t q[1] = {0x07};
    const uint16_t s[2] = {0x0001, 0x3555};
    const float x[2] = {1, 0};
    float y = 0;
    OL_EQ(tl_matmul_q4_f32(x, q, s, &y, 1, 1, 2, 2, NULL), TL_OK);
    OL_EQ(y, 7.0f * 0x1p-24f);
    OL_EQ(tl_matmul_q4_f32(x, q, s + 1, &y, 1, 1, 2, 2, NULL), TL_OK);
    OL_CLOSE(y, 7.0 * 0.333251953125, 1e-7, 0);
}

/* Dequantize an int4 weight [N, K] to float32 the slow, obvious way. */
static void dequant_q4(const uint8_t *q, const uint16_t *s, int64_t N, int64_t K, int64_t G, float *w) {
    for (int64_t n = 0; n < N; n++)
        for (int64_t k = 0; k < K; k++) {
            int nib = (q[n * (K / 2) + k / 2] >> (4 * (k % 2))) & 0xF;
            w[n * K + k] = (float)(nib < 8 ? nib : nib - 16) * tl_f16_to_f32(s[n * (K / G) + k / G]);
        }
}

static void random_q4(ol_gen *g, uint8_t *q, uint16_t *s, int64_t N, int64_t K, int64_t G) {
    for (int64_t i = 0; i < N * K / 2; i++) q[i] = (uint8_t)ol_gen_int(g, 0, 255);
    for (int64_t i = 0; i < N * (K / G); i++) s[i] = tl_f32_to_f16(ol_gen_f32(g, 0.001f, 0.1f));
}

OL_TEST(matches_dequantize_then_matmul) {
    /* WHY: the fused kernel must equal "dequantize, then the ordinary
     *      matmul" (your L9.1 tl_matmul_f32 with trans_b) within the frozen
     *      sqrt(K) bound, for groups of 32 and 64 (the common ones), 2
     *      (the smallest), and one group spanning all of K = 384 (longer
     *      than one unpack chunk), with M = 3 rows of activations.
     * KIND: differential
     * CATCHES: s03, s04, s06, m01, m02
     * CHAPTER: L9.5 section 4 */
    enum { M = 3, N = 37, K = 384 };
    static uint8_t q[N * K / 2];
    static uint16_t s[N * K / 2];
    static float x[M * K], w[N * K], want[M * N], got[M * N];
    static const int64_t groups[4] = {32, 64, 2, K};
    ol_gen g = {21};
    for (int i = 0; i < M * K; i++) x[i] = ol_gen_f32(&g, -1, 1);
    for (int t = 0; t < 4; t++) {
        random_q4(&g, q, s, N, K, groups[t]);
        dequant_q4(q, s, N, K, groups[t], w);
        OL_EQ(tl_matmul_f32(x, w, want, M, N, K, K, K, N, 1.0f, 0.0f, 1, NULL), TL_OK);
        OL_EQ(tl_matmul_q4_f32(x, q, s, got, M, N, K, groups[t], NULL), TL_OK);
        for (int i = 0; i < M * N; i++) OL_CLOSE(got[i], want[i], 1e-5 * sqrt(K), 1e-6 * sqrt(K));
    }
}

OL_TEST(q8_matches_dequantize_then_matmul) {
    /* WHY: the int8 kernel against the same recipe, K = 200 (not a multiple
     *      of the 64-wide unpack chunk or the 8 lanes).
     * KIND: differential
     * CATCHES: s07, s08
     * CHAPTER: L9.5 section 4 */
    enum { M = 2, N = 19, K = 200 };
    static int8_t q[N * K];
    static float s[N], x[M * K], w[N * K], want[M * N], got[M * N];
    ol_gen g = {22};
    for (int i = 0; i < N * K; i++) q[i] = (int8_t)ol_gen_int(&g, -127, 127);
    for (int i = 0; i < N; i++) s[i] = ol_gen_f32(&g, 0.001f, 0.02f);
    for (int i = 0; i < M * K; i++) x[i] = ol_gen_f32(&g, -1, 1);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) w[n * K + k] = (float)q[n * K + k] * s[n];
    OL_EQ(tl_matmul_f32(x, w, want, M, N, K, K, K, N, 1.0f, 0.0f, 1, NULL), TL_OK);
    OL_EQ(tl_matmul_q8_f32(x, q, s, got, M, N, K, NULL), TL_OK);
    for (int i = 0; i < M * N; i++) OL_CLOSE(got[i], want[i], 1e-5 * sqrt(K), 1e-6 * sqrt(K));
}

enum { BN = 24, BK = 256, BG = 32 };
static uint8_t g_q[BN * BK / 2];
static uint16_t g_s[BN * BK / BG];
static void q4_rows_fn(void *ctx, const float *x, int64_t m, float *out) {
    (void)ctx;
    tl_matmul_q4_f32(x, g_q, g_s, out, m, BN, BK, BG, NULL);
}

OL_TEST(batch_invariant_rows) {
    /* WHY: decode (M = 1) and a batch of requests (L10.2) must give each
     *      row the same bits (c/ABI.md rule 10): the lanes and the group
     *      order depend only on the row, never on M.
     * KIND: property
     * CATCHES: s09
     * CHAPTER: L9.5 section 2 */
    static float x[12 * BK];
    ol_gen g = {23};
    random_q4(&g, g_q, g_s, BN, BK, BG);
    for (int i = 0; i < 12 * BK; i++) x[i] = ol_gen_f32(&g, -1, 1);
    float probe[BN];
    OL_EQ(tl_matmul_q4_f32(x, g_q, g_s, probe, 1, BN, BK, BG, NULL), TL_OK);
    OL_BATCH_INVARIANT(q4_rows_fn, NULL, x, 12, BK, BN);
}

OL_TEST(shape_and_argument_errors) {
    /* WHY: K odd cannot be packed two per byte, an odd or zero group
     *      cannot hold whole bytes, and K must split into whole groups:
     *      TL_ESHAPE. Negative dims and NULL pointers are TL_EINVAL. y is
     *      untouched by every rejected call; M or N = 0 is a no-op, and
     *      K = 0 is the empty sum (zeros).
     * KIND: boundary
     * CATCHES: s10, s11
     * CHAPTER: L9.5 section 4 */
    float y[2] = {7, 7};
    OL_EQ(tl_matmul_q4_f32(HX, HQ, HS, y, 1, 2, 3, 2, NULL), TL_ESHAPE);
    OL_TRUE(strstr(tl_last_error(), "tl_matmul_q4_f32") != NULL);
    OL_EQ(tl_matmul_q4_f32(HX, HQ, HS, y, 1, 2, 4, 3, NULL), TL_ESHAPE);
    OL_EQ(tl_matmul_q4_f32(HX, HQ, HS, y, 1, 2, 4, 0, NULL), TL_ESHAPE);
    OL_EQ(tl_matmul_q4_f32(HX, HQ, HS, y, 1, 2, 6, 4, NULL), TL_ESHAPE);
    OL_EQ(tl_matmul_q4_f32(HX, HQ, HS, y, -1, 2, 4, 2, NULL), TL_EINVAL);
    OL_EQ(tl_matmul_q4_f32(HX, NULL, HS, y, 1, 2, 4, 2, NULL), TL_EINVAL);
    OL_EQ(tl_matmul_q8_f32(HX, NULL, (const float *)HS, y, 1, 2, 4, NULL), TL_EINVAL);
    OL_EQ(y[0], 7.0f);
    OL_EQ(y[1], 7.0f);
    OL_EQ(tl_matmul_q4_f32(NULL, NULL, NULL, NULL, 0, 2, 4, 2, NULL), TL_OK);
    OL_EQ(tl_matmul_q4_f32(NULL, NULL, NULL, y, 1, 2, 0, 2, NULL), TL_OK);
    OL_EQ(y[0], 0.0f);
}

OL_TEST(pool_result_equals_serial_bitwise) {
    /* WHY: with an rt.03 pool the output rows split across threads; each
     *      element's arithmetic is the same on any worker, so the bits match
     *      the serial call. N = 37 is not a multiple of the grain of 8.
     * KIND: property
     * CATCHES: s12
     * CHAPTER: L9.5 section 2 */
    enum { M = 2, N = 37, K = 128, G = 32 };
    static uint8_t q[N * K / 2];
    static uint16_t s[N * K / G];
    static float x[M * K], y1[M * N], y4[M * N];
    ol_gen g = {24};
    random_q4(&g, q, s, N, K, G);
    for (int i = 0; i < M * K; i++) x[i] = ol_gen_f32(&g, -1, 1);
    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(4, &p), TL_OK);
    tl_status st = tl_matmul_q4_f32(x, q, s, y4, M, N, K, G, p);
    tl_pool_destroy(p);
    OL_EQ(st, TL_OK);
    OL_EQ(tl_matmul_q4_f32(x, q, s, y1, M, N, K, G, NULL), TL_OK);
    OL_BITS_EQ_F32(y4, y1, M * N);
}

int main(void) { return OL_RUN_ALL(); }
