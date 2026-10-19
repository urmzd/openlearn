/* tinyllm/elementwise.h (L9.6): the small kernels of a Llama forward.
 * Rules in c/ABI.md.
 *
 * These return void: the caller guarantees the preconditions (valid
 * pointers, non-negative sizes), and a stub unit does nothing. Outputs may
 * alias their first input exactly, never partly.
 *
 * module: L9.6 (c/src/kernels/elementwise.c)
 * chapter: content/courses/build-an-llm/10-kernels/08-elementwise-kernels.md
 */
#ifndef TINYLLM_ELEMENTWISE_H
#define TINYLLM_ELEMENTWISE_H

#include <stdint.h>

#include "tinyllm/abi.h"


/* RMSNorm over the last axis of x [rows, d]:
 *   y[r, i] = x[r, i] / sqrt(mean_i(x[r, i]^2) + eps) * w[i]
 * The mean is accumulated in f32 in increasing i. */
void tl_rmsnorm_f32(const float *x, const float *w, float *y, int64_t rows, int64_t d, float eps);

/* Rotary position embedding, in place on x [T, H, D] (row-major), rotating
 * the first d_rot dimensions of each head (d_rot even, d_rot <= D; the rest
 * pass through). Token t is at position pos[t]. For pair index i in
 * [0, d_rot / 2): angle = pos[t] * inv_freq[i], c = cos(angle) *
 * attn_scaling, s = sin(angle) * attn_scaling (attn_scaling is the YaRN
 * mscale, 1.0 otherwise), and the pair (a, b) becomes
 * (a * c - b * s, a * s + b * c), where
 *   layout 0 (half, HF Llama):    a = x[i],     b = x[i + d_rot / 2]
 *   layout 1 (interleaved, GPT-J): a = x[2i],    b = x[2i + 1]
 * The angle is computed in double precision, then rounded to float. */
void tl_rope_f32(float *x, const int32_t *pos, int64_t T, int64_t H, int64_t D, int64_t d_rot,
                 const float *inv_freq, float attn_scaling, int layout);

/* y[i] = silu(gate[i]) * up[i], silu(g) = g / (1 + exp(-g)). */
void tl_silu_mul_f32(const float *gate, const float *up, float *y, int64_t n);

/* out[t, :] = table[ids[t], :] for t in [0, n), rows of length d. The
 * caller checks every id against the table's row count. */
void tl_embedding_f32(const float *table, const int32_t *ids, float *out, int64_t n, int64_t d);

/* y[i] = a[i] + b[i]. */
void tl_add_f32(const float *a, const float *b, float *y, int64_t n);

/* The index of the largest x[i]; ties go to the lowest index; NaN entries
 * are skipped; -1 when n <= 0 or every entry is NaN. Greedy decoding
 * (spec/sampling.md, temperature 0). */
int32_t tl_argmax_f32(const float *x, int64_t n);


#endif /* TINYLLM_ELEMENTWISE_H */
