/* Course tests for M09.6 (C side): c/src/numerics/expf.c against
 * tinyllm/numerics.h, built with ASan and UBSan.
 *
 * The oracle is the C library's exp in double precision, within 1e-16 of
 * the real value: far below the float32 ulp the contract is stated in. The
 * Python-generated vectors compare this kernel with M02.1's
 * exp_range_reduced, the same algorithm in float64.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

#define X_OVER 0x1.62e42ep+6f   /* 88.7228317: the largest x with e^x finite */
#define X_NORMAL -0x1.5d589ep+6f /* -87.3365402: below this e^x is subnormal */

static uint32_t to_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

static float from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

static double ulp_f32(double r) {
    r = fabs(r);
    if (r < 0x1p-126) return 0x1p-149;
    int e;
    frexp(r, &e);
    return ldexp(1.0, e - 1 - 23);
}

/* |y - e^x| in float32 ulps of e^x. NaN when y is NaN. */
static double ulps(float x, float y) {
    double r = exp((double)x);
    return fabs((double)y - r) / ulp_f32(r);
}

/* Tracks the worst error over a sweep and fails the test with the place. */
typedef struct {
    double worst;
    float at;
} worst_t;

static void see(worst_t *w, float x) {
    double e = ulps(x, tl_expf(x));
    if (!(e <= w->worst)) {
        w->worst = isnan(e) ? INFINITY : e;
        w->at = x;
    }
}

#define OL_WORST_AT_MOST(w, bound)                                                                  \
    do {                                                                                            \
        if ((w).worst > (bound)) {                                                                  \
            char msg_[128];                                                                         \
            snprintf(msg_, sizeof msg_, "worst error %.3f ulp at x = %.9g (bound %g)", (w).worst,  \
                     (double)(w).at, (double)(bound));                                              \
            OL_FAIL(msg_);                                                                          \
        }                                                                                           \
    } while (0)

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example. x = 1: k = round(1.442695) = 1,
     *      r = 1 - 0.693359375 + 2.1219444e-4 = 0.30685282, the degree-7
     *      polynomial gives e^r = 1.3591409, and 2^1 times that is
     *      2.7182817, the float32 nearest to e. The others follow the same
     *      steps with k = -1, 14, 0.
     * KIND: unit, smoke
     * CATCHES: m001, m002, m003
     * CHAPTER: M09.6 section 3 */
    OL_EQ(to_bits(tl_expf(1.0f)), to_bits(2.7182817f));
    OL_EQ(to_bits(tl_expf(-1.0f)), to_bits(0.36787945f));
    OL_EQ(to_bits(tl_expf(10.0f)), to_bits(22026.465f));
    OL_EQ(tl_expf(0.0f), 1.0f);
    OL_EQ(tl_expf(-0.0f), 1.0f);
}

static float sample_x(ol_gen *g) { return ol_gen_f32(g, X_NORMAL, X_OVER); }

OL_TEST(million_samples_within_2_ulp) {
    /* WHY: the contract's accuracy over the whole normal range: 10^6 seeded
     *      x in [-87.34, 88.72], every result within 2 ulp of e^x (the
     *      reference's worst is 1.21). Stopping the polynomial at r^6 gives
     *      3 ulp; a reduction done in one product, or by truncating x / ln 2,
     *      gives tens.
     * KIND: differential
     * CATCHES: s01, s02, s03, s04
     * CHAPTER: M09.6 section 2.3 */
    ol_gen g = {ol_prop_base_seed() + 61};
    worst_t w = {0.0, 0.0f};
    for (int i = 0; i < 1000000; i++) see(&w, sample_x(&g));
    OL_WORST_AT_MOST(w, 2.0);
}

OL_TEST(sweep_of_minus_one_to_one) {
    /* WHY: near 0 almost every input has k = 0 or +-1, and tiny x checks
     *      the low-order Horner steps (e^x = 1 + x + ... must not lose x).
     *      Every 512th float of [-1, 1] is tried, about 4 million inputs.
     * KIND: property
     * CATCHES: s01, s03, m001
     * CHAPTER: M09.6 section 2.3 */
    worst_t w = {0.0, 0.0f};
    for (uint32_t u = 0; u <= to_bits(1.0f); u += 512) {
        see(&w, from_bits(u));
        see(&w, -from_bits(u));
    }
    OL_WORST_AT_MOST(w, 2.0);
}

OL_TEST(reduction_boundaries) {
    /* WHY: x = (j + 1/2) ln 2 is where round(x / ln 2) switches from j to
     *      j + 1 and |r| reaches its largest value ln(2) / 2, the worst case
     *      for the polynomial. The 64 floats on each side of every such
     *      point for j in [-126, 127] must all be within 2 ulp.
     * KIND: boundary
     * CATCHES: s01, s02, s04
     * CHAPTER: M09.6 section 2.2 */
    worst_t w = {0.0, 0.0f};
    for (int j = -126; j <= 127; j++) {
        float c = (float)((j + 0.5) * 0.6931471805599453);
        if (!(c > X_NORMAL && c < X_OVER)) continue;
        uint32_t u = to_bits(c);
        for (uint32_t d = 0; d < 64; d++) {
            see(&w, from_bits(u + d));
            see(&w, from_bits(u - d));
        }
    }
    OL_WORST_AT_MOST(w, 2.0);
}

OL_TEST(overflow_edge) {
    /* WHY: 88.7228317 is the largest float whose e^x is finite: there
     *      k = 128, one past the largest float exponent, so 2^k has to be
     *      applied in two halves. The next float up overflows to +inf, as
     *      do FLT_MAX and +inf.
     * KIND: boundary
     * CATCHES: s04, s05
     * CHAPTER: M09.6 section 5, Pitfalls */
    float y = tl_expf(X_OVER);
    OL_TRUE(isfinite(y));
    OL_TRUE(ulps(X_OVER, y) <= 2.0);
    OL_TRUE(isinf(tl_expf(nextafterf(X_OVER, INFINITY))) && tl_expf(nextafterf(X_OVER, INFINITY)) > 0);
    OL_TRUE(isinf(tl_expf(FLT_MAX)));
    OL_TRUE(isinf(tl_expf(INFINITY)) && tl_expf(INFINITY) > 0);
}

OL_TEST(underflow_edge) {
    /* WHY: below -87.34 the result is subnormal and below -103.97 it rounds
     *      to 0. The contract lets an implementation flush subnormal results
     *      to +0, but never return a negative, a NaN, or a value above the
     *      normal range, and e^-inf is exactly 0. -87.3 is still normal and
     *      within 2 ulp.
     * KIND: boundary
     * CATCHES: s02, s04, s07
     * CHAPTER: M09.6 section 5, Pitfalls */
    OL_TRUE(ulps(-87.3f, tl_expf(-87.3f)) <= 2.0);
    OL_EQ(tl_expf(-INFINITY), 0.0f);
    OL_TRUE(!signbit(tl_expf(-INFINITY)));
    OL_EQ(tl_expf(-104.0f), 0.0f);
    OL_EQ(tl_expf(-1000.0f), 0.0f);
    OL_EQ(tl_expf(-FLT_MAX), 0.0f);
    for (float x = -103.9f; x < X_NORMAL; x += 0.0625f) {
        float y = tl_expf(x);
        OL_TRUE(y >= 0.0f && y <= FLT_MIN);
        /* Flushed to 0, or within one subnormal step (2^-149) of e^x. */
        OL_TRUE(y == 0.0f || fabs((double)y - exp((double)x)) <= 0x1p-149);
    }
}

OL_TEST(nan_in_nan_out) {
    /* WHY: a NaN logit (a bug upstream) must stay visible as NaN in the
     *      softmax, not turn into a plausible number. Converting NaN / ln 2
     *      to an int is undefined behavior in C, so the check comes first.
     * KIND: boundary
     * CATCHES: s06
     * CHAPTER: M09.6 section 4 */
    OL_TRUE(isnan(tl_expf(NAN)));
    OL_TRUE(isnan(tl_expf(-NAN)));
}

OL_TEST(array_matches_scalar_and_aliases) {
    /* WHY: the softmax kernel calls the array form on a whole row, often in
     *      place. Each index gets the scalar's bits, exactly n are written,
     *      and n = 0 touches nothing.
     * KIND: unit
     * CATCHES: s08, s09
     * CHAPTER: M09.6 section 4 */
    enum { N = 1000 };
    float x[N + 1], y[N + 1], z[N + 1];
    ol_gen g = {ol_prop_base_seed() + 62};
    for (int i = 0; i < N; i++) x[i] = ol_gen_f32(&g, -20.0f, 20.0f);
    x[N] = 1.0f;
    for (int i = 0; i <= N; i++) y[i] = -1.0f;
    tl_exp_f32(x, y, N);
    OL_EQ(y[N], -1.0f);
    for (int i = 0; i < N; i++) OL_EQ(to_bits(y[i]), to_bits(tl_expf(x[i])));
    memcpy(z, x, sizeof z);
    tl_exp_f32(z, z, N);
    for (int i = 0; i < N; i++) OL_EQ(to_bits(z[i]), to_bits(y[i]));
    y[0] = -1.0f;
    tl_exp_f32(x, y, 0);
    OL_EQ(y[0], -1.0f);
}

int main(void) { return OL_RUN_ALL(); }
