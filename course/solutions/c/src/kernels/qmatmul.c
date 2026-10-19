/* c/src/kernels/qmatmul.c (L9.5): fused dequantize-and-multiply for
 * quantized Linear weights, y = x @ W^T, W never materialized in float32.
 * Contract: tinyllm/qmatmul.h; int4 layout in formats/safetensors.md; f16
 * scales decoded by M09.4's tl_f16_to_f32; threads from rt.03.
 *
 * Decode (M = 1) is memory-bound: every weight is read once per token.
 * int4 moves 4.5 bits per weight (with an f16 scale per 32) instead of 32,
 * so the kernel only has to unpack fast enough to keep up with memory.
 * It unpacks a chunk of up to 64 weights into floats, then accumulates the
 * chunk's products in 8 interleaved lanes (lane l takes k = l, l + 8, ...
 * of the group), which the compiler turns into vector registers. The lanes
 * are combined in a fixed tree at the end of each group, then scaled and
 * added to the output in increasing group order. Every step depends only
 * on (m, n), never on M or on the thread: batch invariance (c/ABI.md
 * rule 10).
 */
#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/numerics.h"
#include "tinyllm/pool.h"
#include "tinyllm/qmatmul.h"

/* Keep every product and sum a separately rounded operation, on every code
 * path (see L9.1). */
#pragma STDC FP_CONTRACT OFF

#define LANES 8
#define CHUNK 64 /* weights unpacked at once (even, a multiple of LANES) */

typedef struct {
    const float *x;
    const void *wq;
    const void *scales;
    float *y;
    int64_t M, N, K, group;
} qm_args;

/* lane[l] += x[k] * q[k] for k = l, l + LANES, ... < n (the chunk's own
 * positions; a chunk always starts at a multiple of LANES in its group). */
static void lanes_dot(const float *restrict x, const float *restrict q, int64_t n, float *restrict lane) {
    float acc[LANES]; /* a local copy the compiler can keep in registers */
    for (int l = 0; l < LANES; l++) acc[l] = lane[l];
    int64_t k = 0;
    for (; k + LANES <= n; k += LANES)
        for (int l = 0; l < LANES; l++) acc[l] += x[k + l] * q[k + l];
    for (int l = 0; k + l < n; l++) acc[l] += x[k + l] * q[k + l];
    for (int l = 0; l < LANES; l++) lane[l] = acc[l];
}

/* The fixed combination of the lanes: ((0+1)+(2+3)) + ((4+5)+(6+7)). */
static float lanes_sum(const float lane[LANES]) {
    return ((lane[0] + lane[1]) + (lane[2] + lane[3])) + ((lane[4] + lane[5]) + (lane[6] + lane[7]));
}

/* Signed 4-bit two's complement: 0..7 stay, 8..15 become -8..-1. */
static void unpack_q4(const uint8_t *bytes, int64_t n_bytes, float *q) {
    for (int64_t b = 0; b < n_bytes; b++) {
        int lo = bytes[b] & 0x0F, hi = bytes[b] >> 4;
        q[2 * b] = (float)(lo - ((lo & 8) << 1));     /* low nibble: column 2b */
        q[2 * b + 1] = (float)(hi - ((hi & 8) << 1)); /* high nibble: column 2b + 1 */
    }
}

static void q4_rows(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)worker;
    const qm_args *a = ctx;
    const uint8_t *wq = a->wq;
    const uint16_t *sc = a->scales;
    const int64_t ngroups = a->K / a->group;
    float q[CHUNK];
    for (int64_t n = lo; n < hi; n++) {
        const uint8_t *wrow = wq + n * (a->K / 2);
        for (int64_t m = 0; m < a->M; m++) {
            const float *xrow = a->x + m * a->K;
            float acc = 0.0f;
            for (int64_t g = 0; g < ngroups; g++) {
                float lane[LANES] = {0};
                for (int64_t c = 0; c < a->group; c += CHUNK) {
                    int64_t len = a->group - c < CHUNK ? a->group - c : CHUNK;
                    int64_t k0 = g * a->group + c;
                    unpack_q4(wrow + k0 / 2, len / 2, q);
                    lanes_dot(xrow + k0, q, len, lane);
                }
                acc += tl_f16_to_f32(sc[n * ngroups + g]) * lanes_sum(lane);
            }
            a->y[m * a->N + n] = acc;
        }
    }
}

static void q8_rows(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)worker;
    const qm_args *a = ctx;
    const int8_t *wq = a->wq;
    const float *sc = a->scales;
    float q[CHUNK];
    for (int64_t n = lo; n < hi; n++) {
        const int8_t *wrow = wq + n * a->K;
        for (int64_t m = 0; m < a->M; m++) {
            const float *xrow = a->x + m * a->K;
            float lane[LANES] = {0};
            for (int64_t c = 0; c < a->K; c += CHUNK) {
                int64_t len = a->K - c < CHUNK ? a->K - c : CHUNK;
                for (int64_t i = 0; i < len; i++) q[i] = (float)wrow[c + i];
                lanes_dot(xrow + c, q, len, lane);
            }
            a->y[m * a->N + n] = sc[n] * lanes_sum(lane);
        }
    }
}

/* Rows of W (outputs n) in ranges of 8; a range's arithmetic does not
 * depend on the worker, so threads never change the bits. */
static tl_status run(tl_pool *tp, tl_range_fn fn, qm_args *a) {
    if (tp == NULL) {
        fn(a, 0, a->N, 0);
        return TL_OK;
    }
    return tl_parallel_for(tp, a->N, 8, fn, a);
}

tl_status tl_matmul_q4_f32(const float *x, const uint8_t *wq, const uint16_t *scales_f16,
                           float *y, int64_t M, int64_t N, int64_t K, int64_t group,
                           tl_pool *tp) {
    if (M < 0 || N < 0 || K < 0) {
        tl_set_last_error("tl_matmul_q4_f32: negative dimension");
        return TL_EINVAL;
    }
    if (group <= 0 || group % 2 != 0 || K % 2 != 0 || K % group != 0) {
        tl_set_last_error("tl_matmul_q4_f32: need K even, group even and > 0, K % group == 0");
        return TL_ESHAPE;
    }
    if (M == 0 || N == 0) return TL_OK;
    if (y == NULL || (K > 0 && (x == NULL || wq == NULL || scales_f16 == NULL))) {
        tl_set_last_error("tl_matmul_q4_f32: NULL pointer");
        return TL_EINVAL;
    }
    if (K == 0) { /* the empty sum; and x, wq may be NULL (no pointer arithmetic on them) */
        for (int64_t i = 0; i < M * N; i++) y[i] = 0.0f;
        return TL_OK;
    }
    qm_args a = {x, wq, scales_f16, y, M, N, K, group};
    return run(tp, q4_rows, &a);
}

tl_status tl_matmul_q8_f32(const float *x, const int8_t *wq, const float *scales,
                           float *y, int64_t M, int64_t N, int64_t K, tl_pool *tp) {
    if (M < 0 || N < 0 || K < 0) {
        tl_set_last_error("tl_matmul_q8_f32: negative dimension");
        return TL_EINVAL;
    }
    if (M == 0 || N == 0) return TL_OK;
    if (y == NULL || scales == NULL || (K > 0 && (x == NULL || wq == NULL))) {
        tl_set_last_error("tl_matmul_q8_f32: NULL pointer");
        return TL_EINVAL;
    }
    if (K == 0) { /* the empty sum; and x, wq may be NULL (no pointer arithmetic on them) */
        for (int64_t i = 0; i < M * N; i++) y[i] = 0.0f;
        return TL_OK;
    }
    qm_args a = {x, wq, scales, y, M, N, K, 0};
    return run(tp, q8_rows, &a);
}
