/* tinyllm/softmax.h (L9.2): row softmax, three-pass and online two-pass.
 * Rules in c/ABI.md.
 *
 * x and y are rows x cols, row-major and contiguous (row stride = cols);
 * y may equal x (in place) but must not partly overlap it. For each row:
 *   m = max_j x_j,  s = sum_j exp(x_j - m),  y_j = exp(x_j - m) / s
 * The three-pass version reads the row for m, then for s, then writes y.
 * The online version keeps a running max and rescales the running sum
 * (s = s * exp(m_old - m_new) + exp(x_j - m_new)), so it reads x twice.
 * Both are finite for inputs as large as +-1e4 and each row sums to 1
 * within cols * 2^-24. A row whose entries are all -inf gives all zeros
 * (an attention row with every key masked). NaN propagates.
 *
 * module: L9.2 (c/src/kernels/softmax.c)
 * chapter: content/courses/ai-engineering/22-kernels/04-softmax-three-pass-and-online.md
 */
#ifndef TINYLLM_SOFTMAX_H
#define TINYLLM_SOFTMAX_H

#include <stdint.h>

#include "tinyllm/abi.h"


/* TL_EINVAL for negative dims or a NULL pointer with rows * cols > 0. */
tl_status tl_softmax_f32(const float *x, float *y, int64_t rows, int64_t cols);
tl_status tl_softmax_online_f32(const float *x, float *y, int64_t rows, int64_t cols);


#endif /* TINYLLM_SOFTMAX_H */
