/* Course tests for L9.6 (C side): c/src/kernels/elementwise.c against
 * tinyllm/elementwise.h, under ASan and UBSan. These kernels return void
 * (the caller guarantees the preconditions), so every test checks values,
 * and a stub (which aborts) fails the whole binary.
 * Deterministic vectors exercise normalization and position encoding.
 */
#include <math.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_invariance.h"
#include "ol_prop.h"
#include "ol_test.h"


OL_TEST(hand_example) {
    /* WHY: the chapter's worked example, one line per kernel:
     *      RMSNorm of [3, 4] with w = [1, 0.5] is [0.8485281, 0.5656854];
     *      RoPE (half layout) of [1, 2, 3, 4] at position 1 with
     *      inv_freq = [1, 0.01]; SiLU-mul of gate [0, 1, -1] and up
     *      [5, 2, 3]; argmax of [2, 7, 7, -1] is 1 (the first 7).
     * KIND: unit, smoke
     * CATCHES: s03, s05, s09, s12
     * CHAPTER: L9.6 section 3 */
    const float x[2] = {3, 4}, w[2] = {1, 0.5f};
    float y[2];
    tl_rmsnorm_f32(x, w, y, 1, 2, 0.0f);
    OL_CLOSE(y[0], 0.848528137423857, 1e-6, 1e-7);
    OL_CLOSE(y[1], 0.565685424949238, 1e-6, 1e-7);

    float q[4] = {1, 2, 3, 4};
    const int32_t pos[1] = {1};
    const float inv_freq[2] = {1.0f, 0.01f};
    tl_rope_f32(q, pos, 1, 1, 4, 4, inv_freq, 1.0f, 0);
    const double want_q[4] = {-1.9841106485555495, 1.959900667496664, 2.4623779024123156, 4.019799668334994};
    for (int i = 0; i < 4; i++) OL_CLOSE(q[i], want_q[i], 1e-6, 1e-7);

    const float gate[3] = {0, 1, -1}, up[3] = {5, 2, 3};
    float s[3];
    tl_silu_mul_f32(gate, up, s, 3);
    OL_EQ(s[0], 0.0f);
    OL_CLOSE(s[1], 1.4621171572600098, 1e-6, 1e-7);
    OL_CLOSE(s[2], -0.8068242641099853, 1e-6, 1e-7);

    const float logits[4] = {2, 7, 7, -1};
    OL_EQ(tl_argmax_f32(logits, 4), 1);
}

OL_TEST(rmsnorm_eps_is_inside_the_root) {
    /* WHY: Llama and the contract put eps inside the square root:
     *      x / sqrt(mean(x^2) + eps). For x = [1e-3, 1e-3] and eps = 1e-6
     *      that is 0.7071, while x / (sqrt(mean) + eps) is 0.999. The two
     *      agree on ordinary activations, so only a tiny-norm row tells
     *      them apart, and a model trained with one form drifts with the
     *      other.
     * KIND: boundary
     * CATCHES: s01, s02
     * CHAPTER: L9.6 section 5, Pitfalls */
    const float x[2] = {1e-3f, 1e-3f}, w[2] = {1, 1};
    float y[2];
    tl_rmsnorm_f32(x, w, y, 1, 2, 1e-6f);
    OL_CLOSE(y[0], 0.7071067811865476, 1e-5, 1e-6);
    const float z[3] = {0, 0, 0}, w3[3] = {1, 1, 1};
    float yz[3] = {7, 7, 7};
    tl_rmsnorm_f32(z, w3, yz, 1, 3, 1e-6f); /* a zero row stays zero, not NaN */
    for (int i = 0; i < 3; i++) OL_EQ(yz[i], 0.0f);
}

/* rows -> rows of RMSNorm, for the batch-invariance helper. */
static float g_w[96];
static void rms_rows(void *ctx, const float *x, int64_t m, float *out) {
    (void)ctx;
    tl_rmsnorm_f32(x, g_w, out, m, 96, 1e-5f);
}

OL_TEST(rmsnorm_rows_are_independent_and_in_place) {
    /* WHY: the Rust engine normalizes a whole batch in one call (L10.2); a
     *      row's bits must not depend on its neighbours (batch invariance,
     *      c/ABI.md rule 10), and the forward normalizes the residual stream
     *      in place (y == x).
     * KIND: property
     * CATCHES: s15
     * CHAPTER: L9.6 section 4 */
    float x[12 * 96];
    ol_gen g = {7};
    for (int i = 0; i < 96; i++) g_w[i] = ol_gen_f32(&g, 0.5f, 1.5f);
    for (int i = 0; i < 12 * 96; i++) x[i] = ol_gen_f32(&g, -3, 3) * (float)(1 + i / 96);
    OL_BATCH_INVARIANT(rms_rows, NULL, x, 12, 96, 96);
    float out[96], inplace[96];
    memcpy(inplace, x, sizeof inplace);
    tl_rmsnorm_f32(x, g_w, out, 1, 96, 1e-5f);
    tl_rmsnorm_f32(inplace, g_w, inplace, 1, 96, 1e-5f);
    OL_BITS_EQ_F32(out, inplace, 96);
}

OL_TEST(rope_interleaved_layout_and_position_zero) {
    /* WHY: the two layouts pair different entries: half pairs (x[i],
     *      x[i + d_rot/2]) like HF Llama and SmolLM2, interleaved pairs
     *      (x[2i], x[2i + 1]) like Meta's code. Loading HF weights with the
     *      wrong layout gives plausible but wrong attention. Position 0 is
     *      angle 0: the identity, exactly.
     * KIND: unit
     * CATCHES: s04, s05
     * CHAPTER: L9.6 section 2 */
    float q[4] = {1, 2, 3, 4};
    const int32_t pos[2] = {1, 0};
    const float inv_freq[2] = {1.0f, 0.01f};
    tl_rope_f32(q, pos, 1, 1, 4, 4, inv_freq, 1.0f, 1);
    const double want[4] = {-1.1426396637476532, 1.922075596544176, 2.9598506679133294, 4.029799501669161};
    for (int i = 0; i < 4; i++) OL_CLOSE(q[i], want[i], 1e-6, 1e-7);
    float k[4] = {1, 2, 3, 4};
    tl_rope_f32(k, pos + 1, 1, 1, 4, 4, inv_freq, 1.0f, 0);
    for (int i = 0; i < 4; i++) OL_EQ(k[i], (float)(i + 1));
}

OL_TEST(rope_partial_rotary_and_scaling) {
    /* WHY: partial rotary (GPT-NeoX, Phi) rotates only the first d_rot
     *      entries of each head; the rest must pass through untouched.
     *      attn_scaling (YaRN's mscale, L7.4) multiplies cos and sin, so a
     *      scaling of 2 doubles a rotated vector's length. Two heads and two
     *      tokens check the [T, H, D] indexing.
     * KIND: unit
     * CATCHES: s07, s08, m02
     * CHAPTER: L9.6 section 4 */
    float x[2 * 2 * 6];
    for (int i = 0; i < 24; i++) x[i] = (float)(i % 6 + 1);
    const int32_t pos[2] = {0, 3};
    const float inv_freq[2] = {0.5f, 0.25f};
    tl_rope_f32(x, pos, 2, 2, 6, 4, inv_freq, 2.0f, 0);
    for (int t = 0; t < 2; t++)
        for (int h = 0; h < 2; h++) {
            const float *v = x + (t * 2 + h) * 6;
            OL_EQ(v[4], 5.0f); /* the pass-through tail */
            OL_EQ(v[5], 6.0f);
            /* pair 0 is (v[0], v[2]) = (1, 3) rotated and scaled by 2 */
            double ang = t == 0 ? 0.0 : 3 * 0.5;
            OL_CLOSE(v[0], 2 * (1 * cos(ang) - 3 * sin(ang)), 1e-5, 1e-6);
            OL_CLOSE(v[2], 2 * (1 * sin(ang) + 3 * cos(ang)), 1e-5, 1e-6);
        }
}

static int rope_keeps_pair_lengths(ol_gen *g, int size) {
    int64_t D = 2 * (1 + size % 32), T = 1 + size % 5, H = 1 + size % 3;
    float x[5 * 3 * 64], x0[5 * 3 * 64], inv_freq[32];
    int32_t pos[5];
    for (int64_t i = 0; i < T * H * D; i++) x0[i] = x[i] = ol_gen_f32(g, -2, 2);
    for (int64_t i = 0; i < D / 2; i++) inv_freq[i] = (float)pow(10000.0, -2.0 * (double)i / (double)D);
    for (int64_t t = 0; t < T; t++) pos[t] = (int32_t)ol_gen_int(g, 0, 4096);
    int layout = (int)ol_gen_int(g, 0, 1);
    tl_rope_f32(x, pos, T, H, D, D, inv_freq, 1.0f, layout);
    for (int64_t r = 0; r < T * H; r++)
        for (int64_t i = 0; i < D / 2; i++) {
            int64_t a = layout ? r * D + 2 * i : r * D + i, b = layout ? a + 1 : a + D / 2;
            double before = (double)x0[a] * x0[a] + (double)x0[b] * x0[b];
            double after = (double)x[a] * x[a] + (double)x[b] * x[b];
            if (fabs(after - before) > 1e-5 * before + 1e-6) return 0;
        }
    return 1;
}

OL_TEST(rope_is_a_rotation) {
    /* WHY: a rotation keeps every pair's length; any mistake that reads a
     *      half-updated value (a' then b from a') or uses the wrong partner
     *      breaks it. Random D, T, H, positions up to 4096, both layouts.
     * KIND: property
     * CATCHES: s04, s06
     * CHAPTER: L9.6 section 2 */
    OL_CHECK_PROP(rope_keeps_pair_lengths, 150, 150);
}

OL_TEST(silu_mul_saturates_without_nan) {
    /* WHY: at gate = -1e4, e^-g overflows to +inf and g / inf is -0: the
     *      right limit. A formulation like g * e^g / (1 + e^g) gives
     *      inf / inf = NaN at gate = +100. Large gates are rare but real in
     *      trained MLPs, and one NaN poisons the residual stream.
     * KIND: boundary
     * CATCHES: s09, s10
     * CHAPTER: L9.6 section 5, Pitfalls */
    const float gate[4] = {-1e4f, -100.0f, 100.0f, 30.0f}, up[4] = {2, 2, 2, 1};
    float y[4];
    tl_silu_mul_f32(gate, up, y, 4);
    OL_TRUE(fabsf(y[0]) == 0.0f);
    OL_TRUE(fabsf(y[1]) < 1e-30f);
    OL_CLOSE(y[2], 200.0, 1e-6, 0);
    OL_CLOSE(y[3], 30.0, 1e-6, 0);
}

OL_TEST(embedding_gathers_rows) {
    /* WHY: the first op of the forward: out[t] is row ids[t] of the table,
     *      d floats each; repeated ids copy the same row. Indexing the table
     *      by ids[t] instead of ids[t] * d reads the wrong numbers.
     * KIND: unit
     * CATCHES: s11
     * CHAPTER: L9.6 section 4 */
    const float table[3 * 2] = {10, 11, 20, 21, 30, 31};
    const int32_t ids[3] = {2, 0, 2};
    float out[6] = {0};
    tl_embedding_f32(table, ids, out, 3, 2);
    const float want[6] = {30, 31, 10, 11, 30, 31};
    for (int i = 0; i < 6; i++) OL_EQ(out[i], want[i]);
}

OL_TEST(add_in_place) {
    /* WHY: the residual add of every block, y = a + b, done in place into
     *      the residual stream (y == a).
     * KIND: unit
     * CATCHES: m03
     * CHAPTER: L9.6 section 4 */
    float a[3] = {1, 2, 3};
    const float b[3] = {0.5f, -2, 10};
    tl_add_f32(a, b, a, 3);
    OL_EQ(a[0], 1.5f);
    OL_EQ(a[1], 0.0f);
    OL_EQ(a[2], 13.0f);
}

OL_TEST(argmax_ties_nan_and_empty) {
    /* WHY: greedy decoding (temperature 0) is argmax, and the spec breaks
     *      ties to the lowest id, so the Python, C, and Rust engines emit
     *      the same token. NaN entries are skipped; a row with no number at
     *      all, or no entries, gives -1 for the caller to report.
     * KIND: boundary
     * CATCHES: s12, s13, s14
     * CHAPTER: L9.6 section 5, Pitfalls */
    const float tie[4] = {1, 5, 5, 5};
    OL_EQ(tl_argmax_f32(tie, 4), 1);
    const float with_nan[4] = {NAN, 1, NAN, 3};
    OL_EQ(tl_argmax_f32(with_nan, 4), 3);
    const float all_nan[2] = {NAN, NAN};
    OL_EQ(tl_argmax_f32(all_nan, 2), -1);
    OL_EQ(tl_argmax_f32(tie, 0), -1);
    const float masked[3] = {-INFINITY, -INFINITY, -INFINITY};
    OL_EQ(tl_argmax_f32(masked, 3), 0);
}

int main(void) { return OL_RUN_ALL(); }
