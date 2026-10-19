/* c/src/numerics/expf.c (M09.6): e^x by range reduction and a polynomial.
 * Contract: tinyllm/numerics.h (the M09.6 section). Callers: the softmax
 * kernels of L9.2, the attention kernels of L9.3, and SiLU in L9.6.
 *
 * The algorithm:
 *
 *   1. specials   NaN -> NaN; x > 88.72283 (0x1.62e42ep+6, the largest
 *                 float whose e^x is below FLT_MAX) -> +inf;
 *                 x < -103.97208 (0x1.9fe368p+6 negated, where e^x is below
 *                 half the smallest subnormal 2^-149) -> +0. -inf lands in
 *                 the second rule.
 *   2. reduce     k = round(x / ln 2), r = x - k ln 2, so |r| <= ln(2) / 2
 *                 and e^x = 2^k e^r. The product k ln 2 is computed in two
 *                 parts (Cody and Waite): LN2_HI has only 9 significant
 *                 bits, so k * LN2_HI is exact for |k| <= 150 and x - k LN2_HI
 *                 loses nothing; LN2_LO = ln 2 - LN2_HI carries the rest.
 *   3. polynomial e^r ~ 1 + r + r^2/2! + ... + r^7/7! (Taylor at 0, Horner
 *                 form). On |r| <= 0.3466 the first omitted term is at most
 *                 0.3466^8 / 8! ~ 5.2e-9, below half a float32 ulp of 1, so
 *                 the error is rounding, about 1.2 ulp at worst. Stopping at
 *                 r^6 leaves 1.2e-7 (about 3 ulp): not enough.
 *   4. scale      e^x = p * 2^k, with 2^k built from its bit pattern. k runs
 *                 from -150 to 128, past both ends of the normal exponent
 *                 range, so it is applied in two halves, each a normal power
 *                 of two: p * 2^(k/2) is exact, and the second product rounds
 *                 once, also when the result is subnormal.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm/numerics.h"

#pragma STDC FP_CONTRACT OFF

#define TL_EXP_HI 0x1.62e42ep+6f   /* 88.7228317: largest x with e^x finite */
#define TL_EXP_LO -0x1.9fe368p+6f  /* -103.972076: below this e^x rounds to 0 */
#define TL_LOG2E 0x1.715476p+0f    /* 1 / ln 2 */
#define TL_LN2_HI 0.693359375f     /* 355 / 512: 9 significant bits */
#define TL_LN2_LO -2.12194440e-4f  /* ln 2 - TL_LN2_HI */

/* 2^k for -126 <= k <= 127: exponent field k + 127, mantissa 0. */
static float pow2i(int k) {
    uint32_t u = (uint32_t)(k + 127) << 23;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

float tl_expf(float x) {
    if (isnan(x)) return x + x;
    if (x > TL_EXP_HI) return INFINITY;
    if (x < TL_EXP_LO) return 0.0f; /* also -inf */

    float kf = rintf(x * TL_LOG2E); /* nearest integer, ties to even */
    int k = (int)kf;
    float r = x - kf * TL_LN2_HI;
    r = r - kf * TL_LN2_LO;

    float p = 1.0f / 5040.0f;
    p = p * r + 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    int k1 = k / 2, k2 = k - k1; /* both in [-75, 64] */
    return p * pow2i(k1) * pow2i(k2);
}

void tl_exp_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = tl_expf(x[i]);
}
