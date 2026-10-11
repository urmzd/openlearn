/* Course tests for M09.5 (C side): c/src/numerics/rsqrt.c against
 * tinyllm/numerics.h, built with ASan and UBSan.
 *
 * The oracle is 1 / sqrt(x) in double precision, which is within 1e-16 of
 * the real value, far below the float32 ulp the contract is stated in. The
 * Python-generated vectors hold the C results against M01.2's rsqrt_newton
 * bit for bit.
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

static float from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

static uint32_t to_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

/* One float32 ulp at the magnitude of r (a double): 2^(floor(log2 r) - 23)
 * for normal r, 2^-149 below the normal range. */
static double ulp_f32(double r) {
    r = fabs(r);
    if (r < 0x1p-126) return 0x1p-149;
    int e;
    frexp(r, &e); /* r = m 2^e, m in [0.5, 1): floor(log2 r) = e - 1 */
    return ldexp(1.0, e - 1 - 23);
}

/* |y - r| in ulps of r, with r = 1 / sqrt(x) in double. */
static double ulps(float x, float y) {
    double r = 1.0 / sqrt((double)x);
    return fabs((double)y - r) / ulp_f32(r);
}

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example. x = 4: the guess 0x3EF759DF is
     *      0.48310754 (3.4% low), two float32 Newton steps give 0.49915358
     *      and 0.49999782, and the float64 step lands on 0.5 exactly. x = 2
     *      gives the float32 nearest to 1 / sqrt(2).
     * KIND: unit, smoke
     * CATCHES: s01, s02, s04, m001, m002
     * CHAPTER: M09.5 section 3 */
    OL_EQ(tl_rsqrtf(4.0f), 0.5f);
    OL_EQ(tl_rsqrtf(0.25f), 2.0f);
    OL_EQ(tl_rsqrtf(1.0f), 1.0f);
    OL_EQ(to_bits(tl_rsqrtf(2.0f)), to_bits(0.70710677f));
    OL_EQ(to_bits(tl_rsqrtf(0.15625f)), to_bits(2.529822f));
}

OL_TEST(every_input_of_two_binades) {
    /* WHY: rsqrt(4x) = rsqrt(x) / 2 and the algorithm commutes with that
     *      scaling, so the 2^24 floats of [1, 4) decide every normal input.
     *      The contract's 2 ulp is checked on all of them; the reference's
     *      worst case is 0.5004 ulp. Two float32 Newton steps alone reach 73
     *      ulp, and a third float32 step 2.18 ulp.
     * KIND: property
     * CATCHES: s01, s02, s04, m002
     * CHAPTER: M09.5 section 2.4 */
    double worst = 0.0;
    float at = 0.0f;
    for (uint32_t u = to_bits(1.0f); u < to_bits(4.0f); u++) {
        float x = from_bits(u);
        double e = ulps(x, tl_rsqrtf(x));
        if (!(e <= worst)) { /* also catches NaN */
            worst = isnan(e) ? INFINITY : e;
            at = x;
        }
    }
    if (worst > 2.0) {
        char msg[128];
        snprintf(msg, sizeof msg, "worst error %.3f ulp at x = %.9g (bound 2)", worst, (double)at);
        OL_FAIL(msg);
    }
}

static int scaling_by_four(ol_gen *g, int size) {
    (void)size;
    /* A normal x with 4x and x / 4 normal too. */
    float x = ldexpf(ol_gen_f32(g, 1.0f, 2.0f), (int)ol_gen_int(g, -120, 120));
    float y = tl_rsqrtf(x);
    return to_bits(tl_rsqrtf(4.0f * x)) == to_bits(0.5f * y) && to_bits(tl_rsqrtf(0.25f * x)) == to_bits(2.0f * y);
}

OL_TEST(scaling_by_four_halves_the_result) {
    /* WHY: the property that makes the two-binade check complete: for normal
     *      x, rsqrt(4x) is exactly rsqrt(x) / 2, bit for bit, because the
     *      guess subtracts exactly 2^23 from the bit pattern and every Newton
     *      product scales by a power of two.
     * KIND: property
     * CATCHES: m001
     * CHAPTER: M09.5 section 2.4 */
    OL_CHECK_PROP(scaling_by_four, 2000, 1);
}

OL_TEST(subnormal_inputs) {
    /* WHY: a subnormal has exponent field 0, so the bit-pattern guess is
     *      wildly wrong and three Newton steps do not recover it. Scaling by
     *      2^24 first (exact) and the result by 2^12 after fixes it. Every
     *      2^23 - 1 positive subnormal is tried.
     * KIND: boundary
     * CATCHES: s03, m003
     * CHAPTER: M09.5 section 5, Pitfalls */
    double worst = 0.0;
    float at = 0.0f;
    for (uint32_t u = 1; u < 0x00800000u; u++) {
        float x = from_bits(u);
        double e = ulps(x, tl_rsqrtf(x));
        if (!(e <= worst)) {
            worst = isnan(e) ? INFINITY : e;
            at = x;
        }
    }
    if (worst > 2.0) {
        char msg[128];
        snprintf(msg, sizeof msg, "worst error %.3f ulp at subnormal x = %.9g (bound 2)", worst, (double)at);
        OL_FAIL(msg);
    }
}

OL_TEST(special_values) {
    /* WHY: the contract's edges. 1/sqrt(+0) = +inf and 1/sqrt(-0) = -inf
     *      (the IEEE rule for 1/x at signed zero); +inf gives +0; negative
     *      numbers and NaN give NaN. RMSNorm never feeds these on purpose,
     *      but a zero row with eps = 0 does.
     * KIND: boundary
     * CATCHES: s05, s06, s07
     * CHAPTER: M09.5 section 4 */
    float pz = tl_rsqrtf(0.0f), nz = tl_rsqrtf(-0.0f);
    OL_TRUE(isinf(pz) && pz > 0);
    OL_TRUE(isinf(nz) && nz < 0);
    float pi = tl_rsqrtf(INFINITY);
    OL_TRUE(pi == 0.0f && !signbit(pi));
    OL_TRUE(isnan(tl_rsqrtf(-1.0f)));
    OL_TRUE(isnan(tl_rsqrtf(-FLT_MIN)));
    OL_TRUE(isnan(tl_rsqrtf(-INFINITY)));
    OL_TRUE(isnan(tl_rsqrtf(NAN)));
    OL_TRUE(ulps(FLT_MAX, tl_rsqrtf(FLT_MAX)) <= 2.0);
    OL_TRUE(ulps(FLT_TRUE_MIN, tl_rsqrtf(FLT_TRUE_MIN)) <= 2.0);
}

OL_TEST(array_matches_scalar_and_aliases) {
    /* WHY: the kernels call the array form on a whole row; it must give the
     *      scalar's bits at every index, write exactly n outputs, and work in
     *      place (y == x), which L9.6 uses for its scratch row.
     * KIND: unit
     * CATCHES: s08, s09
     * CHAPTER: M09.5 section 4 */
    enum { N = 1000 };
    float x[N + 1], y[N + 1], z[N + 1];
    ol_gen g = {ol_prop_base_seed() + 11};
    for (int i = 0; i < N; i++) x[i] = ldexpf(ol_gen_f32(&g, 1.0f, 2.0f), (int)ol_gen_int(&g, -60, 60));
    x[N] = 4.0f;
    for (int i = 0; i <= N; i++) y[i] = 123.0f; /* index N is past the end */
    tl_rsqrt_f32(x, y, N);
    OL_EQ(y[N], 123.0f);
    for (int i = 0; i < N; i++) OL_EQ(to_bits(y[i]), to_bits(tl_rsqrtf(x[i])));
    memcpy(z, x, sizeof z);
    tl_rsqrt_f32(z, z, N);
    for (int i = 0; i < N; i++) OL_EQ(to_bits(z[i]), to_bits(y[i]));
    y[0] = -1.0f;
    tl_rsqrt_f32(x, y, 0); /* n = 0 touches nothing */
    OL_EQ(y[0], -1.0f);
}

int main(void) { return OL_RUN_ALL(); }
