/* c/src/kernels/elementwise.c (L9.6): the small kernels of a Llama forward.
 * Contract: tinyllm/elementwise.h. RMSNorm uses M09.5's tl_rsqrtf, SiLU
 * uses M09.6's tl_expf.
 *
 * Each of these reads and writes every element once, so it is bound by
 * memory bandwidth, not arithmetic: the work is to get the definition and
 * the edge cases exactly right, in one pass, with outputs allowed to alias
 * their first input.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "tinyllm/abi.h"
#include "tinyllm/elementwise.h"
#include "tinyllm/numerics.h"

void tl_rmsnorm_f32(const float *x, const float *w, float *y, int64_t rows, int64_t d, float eps) {
    if (rows <= 0 || d <= 0) return;
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * d;
        float *yr = y + r * d;
        /* The mean of squares, accumulated in float32 in increasing i (the
         * contract fixes the order, so a row's result does not depend on
         * how many rows came with it). */
        float ss = 0.0f;
        for (int64_t i = 0; i < d; i++) ss += xr[i] * xr[i];
        float inv = tl_rsqrtf(ss / (float)d + eps); /* eps INSIDE the root */
        for (int64_t i = 0; i < d; i++) yr[i] = xr[i] * inv * w[i];
    }
}

void tl_rope_f32(float *x, const int32_t *pos, int64_t T, int64_t H, int64_t D, int64_t d_rot,
                 const float *inv_freq, float attn_scaling, int layout) {
    if (T <= 0 || H <= 0 || D <= 0 || d_rot <= 0) return;
    const int64_t half = d_rot / 2;
    for (int64_t t = 0; t < T; t++) {
        for (int64_t i = 0; i < half; i++) {
            /* One angle per (token, pair), shared by every head. The product
             * is formed in double and rounded once to float, which equals
             * float32(pos) * float32(inv_freq) for every pos below 2^24. */
            float angle = (float)((double)pos[t] * (double)inv_freq[i]);
            float c = (float)cos((double)angle) * attn_scaling;
            float s = (float)sin((double)angle) * attn_scaling;
            for (int64_t h = 0; h < H; h++) {
                float *xh = x + (t * H + h) * D;
                int64_t ia = layout ? 2 * i : i;
                int64_t ib = layout ? 2 * i + 1 : i + half;
                float a = xh[ia], b = xh[ib]; /* both read before either is written */
                xh[ia] = a * c - b * s;
                xh[ib] = a * s + b * c;
            }
        }
    }
}

void tl_silu_mul_f32(const float *gate, const float *up, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float g = gate[i];
        /* silu(g) = g * sigmoid(g) = g / (1 + e^-g). For g -> -inf, e^-g
         * overflows to +inf and g / inf is -0: the right limit, no NaN. */
        y[i] = g / (1.0f + tl_expf(-g)) * up[i];
    }
}

void tl_embedding_f32(const float *table, const int32_t *ids, float *out, int64_t n, int64_t d) {
    if (n <= 0 || d <= 0) return;
    for (int64_t t = 0; t < n; t++)
        memcpy(out + t * d, table + (int64_t)ids[t] * d, (size_t)d * sizeof(float));
}

void tl_add_f32(const float *a, const float *b, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = a[i] + b[i];
}

int32_t tl_argmax_f32(const float *x, int64_t n) {
    int32_t best = -1;
    for (int64_t i = 0; i < n; i++) {
        if (isnan(x[i])) continue;
        /* Strictly greater: an equal value later in the row never takes
         * over, so ties go to the lowest index (greedy decoding). */
        if (best < 0 || x[i] > x[best]) best = (int32_t)i;
    }
    return best;
}
