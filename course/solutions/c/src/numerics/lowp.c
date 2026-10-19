/* c/src/numerics/lowp.c (M09.7): float32 to and from fp8 (E4M3, E5M2),
 * bfloat16, and IEEE binary16, round to nearest with ties to even.
 * Contract: tinyllm/numerics.h (M09.7 owns the standalone C exercise).
 * The Python twin is python/tinyllm/num/lowp.py (fp8) and M09.1's
 * python/tinyllm/num/fp.py (bf16, f16); the C and Python results must agree
 * on every code and every input the course tests try.
 *
 * One routine rounds for all four formats. A finite float32 magnitude is
 * s * 2^e2 with an integer significand s < 2^24. The target format has M
 * fraction bits and exponent bias B, so its smallest normal exponent is
 * emin = 1 - B, and within the binade of floor(log2 v) (or the subnormal
 * range below 2^emin) its values are the multiples of the quantum 2^qe,
 * qe = max(floor(log2 v), emin) - M. Rounding v to the format is rounding
 * s * 2^(e2 - qe) to an integer n: a right shift by qe - e2 with the
 * round-half-even rule on the bits shifted out. The code is then
 * ((qe + M + B - 1) << M) + n: in the subnormal range that is n itself, and
 * a carry out of the fraction (n == 2^(M+1)) moves into the exponent field
 * by plain addition. Overflow shows up as a code at or above the format's
 * first non-finite code, and each format decides what to do with it.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm/numerics.h"

static uint32_t f32_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u); /* the defined way to view a float's bits */
    return u;
}

/* Rounded magnitude code of the finite, non-negative float32 with bits
 * `abs` in a format with M fraction bits and bias B. Never overflows a
 * uint32 for M <= 10 and B <= 127. */
static uint32_t rne_code(uint32_t abs, int M, int B) {
    uint32_t exp = abs >> 23, frac = abs & 0x7FFFFFu;
    uint32_t s;
    int e2;
    if (exp == 0) {
        if (frac == 0) return 0; /* +0 */
        s = frac;                /* subnormal float32: 0.frac * 2^-126 */
        e2 = -149;
    } else {
        s = frac | 0x800000u; /* normal: 1.frac * 2^(exp - 127) */
        e2 = (int)exp - 150;
    }
    int bits = 0; /* bit length of s */
    for (uint32_t t = s; t != 0; t >>= 1) bits++;
    int floorlog2 = e2 + bits - 1;
    int emin = 1 - B;
    int qe = (floorlog2 > emin ? floorlog2 : emin) - M;
    int shift = qe - e2; /* > 0: the target has fewer fraction bits */
    uint32_t n;
    if (shift > 24) {
        n = 0; /* v < 2^(qe - 1), half a quantum: rounds to 0 */
    } else {
        uint32_t half = 1u << (shift - 1);
        uint32_t rem = s & ((half << 1) - 1u);
        n = s >> shift;
        if (rem > half || (rem == half && (n & 1u))) n++; /* ties to even */
    }
    if (n == 0) return 0;
    return ((uint32_t)(qe + M + B - 1) << M) + n;
}

/* Exact value of a magnitude code (no specials): fraction bits mant and
 * exponent field ef, subnormal when ef == 0. */
static float decode_mag(uint32_t code, int M, int B) {
    uint32_t ef = code >> M, mant = code & ((1u << M) - 1u);
    if (ef == 0) return ldexpf((float)mant, 1 - B - M);
    return ldexpf((float)((1u << M) + mant), (int)ef - B - M);
}

uint8_t tl_f32_to_e4m3(float x) {
    uint32_t u = f32_bits(x), sign = (u >> 24) & 0x80u, abs = u & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return 0x7F; /* NaN */
    if (abs == 0x7F800000u) return (uint8_t)(sign | 0x7Eu); /* inf saturates */
    uint32_t c = rne_code(abs, 3, 7);
    if (c > 0x7Eu) c = 0x7Eu; /* above 448: saturate (0x7F is NaN) */
    return (uint8_t)(sign | c);
}

float tl_e4m3_to_f32(uint8_t b) {
    if ((b & 0x7Fu) == 0x7Fu) return NAN;
    float v = decode_mag(b & 0x7Fu, 3, 7);
    return (b & 0x80u) ? -v : v;
}

uint8_t tl_f32_to_e5m2(float x) {
    uint32_t u = f32_bits(x), sign = (u >> 24) & 0x80u, abs = u & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return 0x7F; /* NaN */
    if (abs == 0x7F800000u) return (uint8_t)(sign | 0x7Cu); /* inf stays inf */
    uint32_t c = rne_code(abs, 2, 15);
    if (c > 0x7Bu) c = 0x7Bu; /* finite overflow saturates to 57344 */
    return (uint8_t)(sign | c);
}

float tl_e5m2_to_f32(uint8_t b) {
    uint32_t m = b & 0x7Fu;
    if (m >= 0x7Cu) return m == 0x7Cu ? ((b & 0x80u) ? -INFINITY : INFINITY) : NAN;
    float v = decode_mag(m, 2, 15);
    return (b & 0x80u) ? -v : v;
}

uint16_t tl_f32_to_bf16(float x) {
    uint32_t u = f32_bits(x), sign = (u >> 16) & 0x8000u, abs = u & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return 0x7FC0; /* NaN: quiet, sign dropped */
    if (abs == 0x7F800000u) return (uint16_t)(sign | 0x7F80u);
    uint32_t c = rne_code(abs, 7, 127);
    if (c > 0x7F80u) c = 0x7F80u; /* overflow to inf */
    return (uint16_t)(sign | c);
}

float tl_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; /* bf16 is the top half of a float32 */
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

uint16_t tl_f32_to_f16(float x) {
    uint32_t u = f32_bits(x), sign = (u >> 16) & 0x8000u, abs = u & 0x7FFFFFFFu;
    if (abs > 0x7F800000u) return 0x7E00; /* NaN */
    if (abs == 0x7F800000u) return (uint16_t)(sign | 0x7C00u);
    uint32_t c = rne_code(abs, 10, 15);
    if (c > 0x7C00u) c = 0x7C00u; /* overflow to inf */
    return (uint16_t)(sign | c);
}

float tl_f16_to_f32(uint16_t b) {
    uint32_t m = b & 0x7FFFu;
    if (m >= 0x7C00u) return m == 0x7C00u ? ((b & 0x8000u) ? -INFINITY : INFINITY) : NAN;
    float v = decode_mag(m, 10, 15);
    return (b & 0x8000u) ? -v : v;
}
