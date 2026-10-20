/* tinyllm/qmatmul.h (L9.5): fused dequantize-and-multiply for quantized
 * linear weights (W4A32 and W8A32), the decode GEMV of a quantized model.
 * Rules in c/ABI.md; the int4 byte layout in formats/safetensors.md.
 *
 * Both compute y = x @ W^T for a weight W of shape [N, K] (a Linear weight
 * [out, in] as stored), x [M, K] and y [M, N], row-major and contiguous.
 * W is never materialized in f32. Each y element sums in one fixed order:
 * 8 interleaved lanes per group (lane l takes k = l, l + 8, ...) combined in
 * a fixed tree, groups in increasing order (batch invariance, c/ABI.md rule 10).
 *
 * module: L9.5 (c/src/kernels/qmatmul.c)
 * chapter: content/courses/ai-engineering/22-kernels/07-fused-quantized-matmul.md
 */
#ifndef TINYLLM_QMATMUL_H
#define TINYLLM_QMATMUL_H

#include <stdint.h>

#include "tinyllm/abi.h"


typedef struct tl_pool tl_pool;

/* int4, symmetric, grouped. wq is [N, K / 2] bytes: byte b of row n holds
 * columns 2b (low nibble) and 2b + 1 (high nibble), each a signed 4-bit
 * two's complement value q in [-8, 7]. scales_f16 is [N, K / group] IEEE
 * f16 bit patterns. W[n, k] = q(n, k) * f16(scales_f16[n * (K / group) +
 * k / group]). Needs K even, group even, and K % group == 0: else
 * TL_ESHAPE. TL_EINVAL for negative dims or NULL pointers. tp may be NULL. */
tl_status tl_matmul_q4_f32(const float *x, const uint8_t *wq, const uint16_t *scales_f16,
                           float *y, int64_t M, int64_t N, int64_t K, int64_t group,
                           tl_pool *tp);

/* int8, symmetric, per output channel. wq is [N, K] int8 and scales is [N]
 * f32: W[n, k] = wq[n * K + k] * scales[n]. */
tl_status tl_matmul_q8_f32(const float *x, const int8_t *wq, const float *scales,
                           float *y, int64_t M, int64_t N, int64_t K, tl_pool *tp);


#endif /* TINYLLM_QMATMUL_H */
