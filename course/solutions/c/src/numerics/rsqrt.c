/* c/src/numerics/rsqrt.c (M09.5): 1 / sqrt(x) as a fixed-point iteration.
 * Contract: tinyllm/numerics.h (the M09.5 section). Caller: L9.6's
 * tl_rmsnorm_f32, which scales each row by 1 / sqrt(mean(x^2) + eps).
 *
 * The algorithm, step for step (the Python twin is M01.2's rsqrt_newton,
 * and the course test compares the two bit for bit):
 *
 *   1. guess   y0 = bits(0x5F3759DF - (bits(x) >> 1))
 *              Halving the bit pattern halves the exponent field, which is
 *              almost -log2(x) / 2; the constant puts the bias back and
 *              tunes the mantissa so |y0 - r| / r <= 0.0344 for r = 1/sqrt(x).
 *   2. two Newton steps in float32, each  y <- y * (1.5 - (0.5 x) * y * y)
 *              with the products in that order. The relative error goes
 *              0.0344 -> 1.75e-3 -> 4.7e-6: e_{n+1} ~ 1.5 e_n^2 (quadratic).
 *   3. one Newton step in float64 on the float32 result, rounded to float32.
 *              4.7e-6 squared is far below float32 resolution, and float64
 *              arithmetic adds no float32-sized rounding error, so the
 *              result is within 0.5005 ulp (checked on all 2^24 inputs of
 *              [1, 4), which covers every normal input: see below).
 *
 * Why [1, 4) is enough: rsqrt(4x) = rsqrt(x) / 2, and every step above
 * commutes with that scaling exactly (the guess subtracts 2^23 from the
 * pattern; every product scales by a power of two). So two binades decide
 * all normal inputs. Subnormal inputs break the guess (their exponent field
 * is 0), so they are scaled up by 2^24 first and the result by 2^12 after.
 *
 * Each arithmetic step is its own statement: C may fuse a * b + c inside one
 * expression into one fused multiply-add (one rounding instead of two),
 * which would break bit equality with Python. The pragma says so too.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm/numerics.h"

#pragma STDC FP_CONTRACT OFF

#define TL_RSQRT_MAGIC 0x5F3759DFu

static uint32_t f32_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u); /* the defined way to view a float's bits */
    return u;
}

static float f32_from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

/* Steps 1 to 3 for a positive normal x. */
static float rsqrt_normal(float x) {
    float hx = 0.5f * x;
    float y = f32_from_bits(TL_RSQRT_MAGIC - (f32_bits(x) >> 1));
    for (int i = 0; i < 2; i++) {
        float t = hx * y;
        t = t * y;
        t = 1.5f - t;
        y = y * t;
    }
    double xd = (double)x, yd = (double)y;
    double hd = 0.5 * xd;
    double t = hd * yd;
    t = t * yd;
    t = 1.5 - t;
    yd = yd * t;
    return (float)yd;
}

float tl_rsqrtf(float x) {
    if (isnan(x)) return x + x; /* quiet NaN in, NaN out */
    if (x == 0.0f) return signbit(x) ? -INFINITY : INFINITY;
    if (x < 0.0f) return NAN;
    if (isinf(x)) return 0.0f;
    if (x < 0x1p-126f) {
        /* Subnormal: 2^24 x is normal and exact; rsqrt(2^24 x) = rsqrt(x) / 2^12. */
        return rsqrt_normal(x * 0x1p24f) * 0x1p12f;
    }
    return rsqrt_normal(x);
}

void tl_rsqrt_f32(const float *x, float *y, int64_t n) {
    /* One read and one write per index, in order, so y == x (exact
     * aliasing) is safe. */
    for (int64_t i = 0; i < n; i++) y[i] = tl_rsqrtf(x[i]);
}
