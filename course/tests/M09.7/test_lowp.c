/* Optional C low-precision tests for M09.7: c/src/numerics/lowp.c against
 * tinyllm/numerics.h, built with ASan and UBSan. M09.4 is the Python oracle;
 * this standalone test checks fixed input/output vectors without FFI.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_test.h"

static uint32_t bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

static float from_bits(uint32_t u) {
    float x;
    memcpy(&x, &u, sizeof x);
    return x;
}

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example in C: 0.3 is 0x2A in e4m3 (0.3125)
     *      and 0x35 in e5m2; the tie 1.0625 goes to the even 1.0 (0x38) and
     *      1.1875 to 1.25 (0x3A).
     * KIND: unit, smoke
     * CATCHES: s19, s21
     * CHAPTER: M09.7 section 3 */
    OL_EQ(tl_f32_to_e4m3(0.3f), 0x2A);
    OL_EQ(tl_e4m3_to_f32(0x2A), 0.3125f);
    OL_EQ(tl_f32_to_e5m2(0.3f), 0x35);
    OL_EQ(tl_e5m2_to_f32(0x35), 0.3125f);
    OL_EQ(tl_f32_to_e4m3(1.0625f), 0x38);
    OL_EQ(tl_f32_to_e4m3(1.1875f), 0x3A);
    OL_EQ(tl_f32_to_e4m3(-0.3f), 0xAA);
}

OL_TEST(fp8_saturation_and_specials) {
    /* WHY: the edge rules of numerics.h: e4m3 saturates past 448 and at
     *      +-inf (it has no infinity), e5m2 saturates finite overflow (61440
     *      is a tie that would round up) but keeps +-inf, NaN gives 0x7F.
     * KIND: boundary
     * CATCHES: s22, s23
     * CHAPTER: M09.7 section 2 */
    OL_EQ(tl_f32_to_e4m3(448.0f), 0x7E);
    OL_EQ(tl_f32_to_e4m3(464.0f), 0x7E);
    OL_EQ(tl_f32_to_e4m3(1e30f), 0x7E);
    OL_EQ(tl_f32_to_e4m3(INFINITY), 0x7E);
    OL_EQ(tl_f32_to_e4m3(-INFINITY), 0xFE);
    OL_EQ(tl_f32_to_e4m3(NAN), 0x7F);
    OL_EQ(tl_f32_to_e4m3(-NAN), 0x7F);
    OL_EQ(tl_f32_to_e5m2(61440.0f), 0x7B);
    OL_EQ(tl_f32_to_e5m2(-3e38f), 0xFB);
    OL_EQ(tl_f32_to_e5m2(INFINITY), 0x7C);
    OL_EQ(tl_f32_to_e5m2(-INFINITY), 0xFC);
    OL_EQ(tl_f32_to_e5m2(NAN), 0x7F);
    OL_TRUE(isnan(tl_e4m3_to_f32(0x7F)) && isnan(tl_e4m3_to_f32(0xFF)));
    OL_TRUE(isinf(tl_e5m2_to_f32(0x7C)) && tl_e5m2_to_f32(0xFC) < 0);
    OL_TRUE(isnan(tl_e5m2_to_f32(0x7D)) && isnan(tl_e5m2_to_f32(0xFF)));
}

OL_TEST(fp8_tiny_and_signed_zero) {
    /* WHY: 2^-10 is half the smallest e4m3 subnormal: a tie that goes to
     *      the even code 0, keeping the sign; the next float32 above it
     *      rounds up to 2^-9. Float32 subnormal inputs round to zero.
     * KIND: boundary
     * CATCHES: s19, s20, s21, s24, s28
     * CHAPTER: M09.7 section 2 */
    OL_EQ(tl_f32_to_e4m3(0x1p-10f), 0x00);
    OL_EQ(tl_f32_to_e4m3(-0x1p-10f), 0x80);
    OL_EQ(tl_f32_to_e4m3(nextafterf(0x1p-10f, 1.0f)), 0x01);
    OL_EQ(tl_f32_to_e4m3(-0.0f), 0x80);
    OL_EQ(tl_f32_to_e4m3(from_bits(1u)), 0x00);
    OL_EQ(tl_f32_to_e5m2(0x1p-16f), 0x01);
    OL_EQ(tl_e4m3_to_f32(0x01), 0x1p-9f);
    OL_EQ(tl_e4m3_to_f32(0x08), 0x1p-6f);
    OL_TRUE(signbit(tl_e4m3_to_f32(0x80)));
}

OL_TEST(fp8_every_code_is_a_fixed_point) {
    /* WHY: decode then encode gives back every finite code of both formats:
     *      a wrong bias, a lost implicit bit, or a carry that does not reach
     *      the exponent field fails whole binades.
     * KIND: property
     * CATCHES: s19, s20, s24
     * CHAPTER: M09.7 section 2 */
    for (int c = 0; c < 256; c++) {
        float v = tl_e4m3_to_f32((uint8_t)c);
        if (!isnan(v)) OL_EQ(tl_f32_to_e4m3(v), c);
        float w = tl_e5m2_to_f32((uint8_t)c);
        if (!isnan(w) && !isinf(w)) OL_EQ(tl_f32_to_e5m2(w), c);
    }
}

OL_TEST(fp8_ties_to_even) {
    /* WHY: the midpoint between any two neighbouring codes rounds to the
     *      code with an even last bit; one float32 step above it rounds up,
     *      one below rounds down (the sticky bits count).
     * KIND: boundary
     * CATCHES: s19, s20, s21, s24, s28
     * CHAPTER: M09.7 section 2 */
    for (int c = 0; c < 0x7E; c++) {
        float lo = tl_e4m3_to_f32((uint8_t)c), hi = tl_e4m3_to_f32((uint8_t)(c + 1));
        float mid = (lo + hi) / 2; /* exact: 4 significant bits plus one */
        OL_EQ(tl_f32_to_e4m3(mid) % 2, 0);
        OL_EQ(tl_f32_to_e4m3(nextafterf(mid, INFINITY)), c + 1);
        OL_EQ(tl_f32_to_e4m3(nextafterf(mid, -INFINITY)), c);
    }
}

OL_TEST(bf16_hand_values) {
    /* WHY: bf16 is the top half of a float32 rounded to nearest even:
     *      1 + 2^-8 is a tie that stays at 1.0 (0x3F80), 1 + 3*2^-8 goes up
     *      to 0x3F82; the largest float32 overflows to +inf; NaN is the
     *      quiet 0x7FC0 with the sign dropped.
     * KIND: unit
     * CATCHES: s19, s20, s21, s25, s26
     * CHAPTER: M09.7 section 2 */
    OL_EQ(tl_f32_to_bf16(1.0f), 0x3F80);
    OL_EQ(tl_f32_to_bf16(1.0f + 0x1p-8f), 0x3F80);
    OL_EQ(tl_f32_to_bf16(1.0f + 3 * 0x1p-8f), 0x3F82);
    OL_EQ(tl_f32_to_bf16(-2.0f), 0xC000);
    OL_EQ(tl_f32_to_bf16(from_bits(0x7F7FFFFFu)), 0x7F80);
    OL_EQ(tl_f32_to_bf16(-INFINITY), 0xFF80);
    OL_EQ(tl_f32_to_bf16(-NAN), 0x7FC0);
    OL_EQ(tl_f32_to_bf16(from_bits(1u)), 0x0000); /* far below bf16's 2^-133 */
    OL_EQ(tl_f32_to_bf16(from_bits(0x00010000u)), 0x0001);
    OL_EQ(bits(tl_bf16_to_f32(0x3F82)), 0x3F820000u);
}

OL_TEST(bf16_every_code_round_trips) {
    /* WHY: all 65536 bf16 codes decode exactly and re-encode to themselves
     *      (NaN codes excepted), subnormals and the sign of zero included.
     * KIND: property
     * CATCHES: s19, s20
     * CHAPTER: M09.7 section 2 */
    for (uint32_t c = 0; c < 65536; c++) {
        float v = tl_bf16_to_f32((uint16_t)c);
        OL_EQ(bits(v), c << 16);
        if (!isnan(v)) OL_EQ(tl_f32_to_bf16(v), c);
    }
}

OL_TEST(f16_hand_values) {
    /* WHY: binary16 has 10 fraction bits, bias 15, subnormals down to 2^-24:
     *      65504 is the largest finite value, 65520 is the tie above it and
     *      rounds to even, which is 65536, so +inf; 2^-25 is a tie with zero
     *      and goes to 0; NaN is 0x7E00.
     * KIND: unit
     * CATCHES: s19, s20, s21, s24, s27, s28
     * CHAPTER: M09.7 section 2 */
    OL_EQ(tl_f32_to_f16(1.0f), 0x3C00);
    OL_EQ(tl_f32_to_f16(65504.0f), 0x7BFF);
    OL_EQ(tl_f32_to_f16(65519.0f), 0x7BFF);
    OL_EQ(tl_f32_to_f16(65520.0f), 0x7C00);
    OL_EQ(tl_f32_to_f16(-1e10f), 0xFC00);
    OL_EQ(tl_f32_to_f16(0x1p-24f), 0x0001);
    OL_EQ(tl_f32_to_f16(0x1p-25f), 0x0000);
    OL_EQ(tl_f32_to_f16(0x1.8p-25f), 0x0001); /* past half a step: up */
    OL_EQ(tl_f32_to_f16(0x1.8p-24f), 0x0002);
    OL_EQ(tl_f32_to_f16(NAN), 0x7E00);
    OL_EQ(tl_f16_to_f32(0x0001), 0x1p-24f);
    OL_EQ(tl_f16_to_f32(0x3555), 0x1.554p-2f);
    OL_TRUE(isinf(tl_f16_to_f32(0xFC00)) && tl_f16_to_f32(0xFC00) < 0);
    OL_TRUE(isnan(tl_f16_to_f32(0x7E00)));
}

OL_TEST(f16_every_code_round_trips) {
    /* WHY: every finite binary16 code is a fixed point of decode then
     *      encode; the KV cache of format v1 (rt.04, L9.4) stores f16 and
     *      must read back exactly what was written.
     * KIND: property
     * CATCHES: s19, s20, s24
     * CHAPTER: M09.7 section 2 */
    for (uint32_t c = 0; c < 65536; c++) {
        float v = tl_f16_to_f32((uint16_t)c);
        if (!isnan(v)) OL_EQ(tl_f32_to_f16(v), c);
    }
}

int main(void) { return OL_RUN_ALL(); }
