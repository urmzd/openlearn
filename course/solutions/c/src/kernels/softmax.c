/* c/src/kernels/softmax.c (L9.2): row softmax, three-pass and online two-pass.
 * Contract: tinyllm/softmax.h. Exponentials come from M09.6's tl_expf.
 *
 * Both versions subtract the row maximum m before exponentiating, so the
 * largest exponent is e^0 = 1 and nothing overflows. The three-pass version
 * finds m first; the online version keeps a running maximum and rescales the
 * running sum whenever the maximum grows, which is the step FlashAttention
 * (L9.3) and paged attention (L9.4) apply to tiles of keys.
 */
#include <math.h>
#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/numerics.h"
#include "tinyllm/softmax.h"

/* Shared argument check; returns TL_OK or TL_EINVAL with the slot set. */
static tl_status check_args(const char *fn, const float *x, const float *y, int64_t rows, int64_t cols) {
    if (rows < 0 || cols < 0) {
        tl_set_last_error(fn);
        return TL_EINVAL;
    }
    if (rows > 0 && cols > 0 && (x == NULL || y == NULL)) {
        tl_set_last_error(fn);
        return TL_EINVAL;
    }
    return TL_OK;
}

tl_status tl_softmax_f32(const float *x, float *y, int64_t rows, int64_t cols) {
    tl_status st = check_args("tl_softmax_f32: negative dimension or NULL pointer", x, y, rows, cols);
    if (st != TL_OK) return st;
    if (rows == 0 || cols == 0) return TL_OK; /* no work; and NULL + 0 is undefined behavior */
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * cols;
        float *yr = y + r * cols;
        /* Pass 1: the maximum. NaN never wins a > comparison, so it is
         * skipped here and propagates through the sum below (or through the
         * masked-row branch when no entry is finite). */
        float m = -INFINITY, fill = 0.0f;
        for (int64_t j = 0; j < cols; j++) {
            if (xr[j] > m) m = xr[j];
            else if (isnan(xr[j])) fill = xr[j];
        }
        if (m == -INFINITY) { /* every key masked: zeros, not 0/0 (or NaN if one was NaN) */
            for (int64_t j = 0; j < cols; j++) yr[j] = fill;
            continue;
        }
        /* Pass 2: exponentiate and sum. Reading xr[j] before writing yr[j]
         * keeps y == x (in place) correct. */
        float s = 0.0f;
        for (int64_t j = 0; j < cols; j++) {
            float e = tl_expf(xr[j] - m);
            yr[j] = e;
            s += e;
        }
        /* Pass 3: normalize. */
        float inv = 1.0f / s;
        for (int64_t j = 0; j < cols; j++) yr[j] *= inv;
    }
    return TL_OK;
}

tl_status tl_softmax_online_f32(const float *x, float *y, int64_t rows, int64_t cols) {
    tl_status st = check_args("tl_softmax_online_f32: negative dimension or NULL pointer", x, y, rows, cols);
    if (st != TL_OK) return st;
    if (rows == 0 || cols == 0) return TL_OK; /* no work; and NULL + 0 is undefined behavior */
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = x + r * cols;
        float *yr = y + r * cols;
        /* Pass 1, one read of x. Invariant after element j:
         *   m = max(x_0..x_j),  s = sum_{i <= j} e^(x_i - m).
         * When x_j raises the maximum, every earlier term was computed
         * against the old maximum, so the sum is rescaled by e^(m_old - m). */
        float m = -INFINITY, s = 0.0f;
        for (int64_t j = 0; j < cols; j++) {
            float v = xr[j];
            if (v == -INFINITY) continue; /* adds e^-inf = 0; and -inf - -inf would be NaN */
            if (v > m) {
                s = s * tl_expf(m - v) + 1.0f;
                m = v;
            } else {
                s += tl_expf(v - m); /* also the path a NaN takes: s becomes NaN */
            }
        }
        if (m == -INFINITY && s == 0.0f) {
            for (int64_t j = 0; j < cols; j++) yr[j] = 0.0f;
            continue;
        }
        /* Pass 2, the second read of x. */
        float inv = 1.0f / s;
        for (int64_t j = 0; j < cols; j++) yr[j] = tl_expf(xr[j] - m) * inv;
    }
    return TL_OK;
}
