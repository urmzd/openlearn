/* tinyllm/matmul.h (M03.1 naive v0; L9.1 takes it over with a tiled,
 * packed, batch-invariant version under the same symbol and signature).
 * Rules in c/ABI.md.
 *
 * chapter: math/03-linear-algebra/01-vectors-matrices-and-matmul-in-c.md
 * chapter: content/courses/build-an-llm/10-kernels/03-tiled-batch-invariant-matmul.md
 */
#ifndef TINYLLM_MATMUL_H
#define TINYLLM_MATMUL_H

#include <stdint.h>

#include "tinyllm/abi.h"


/* The thread pool of rt.03 (tinyllm/pool.h). Declared here as an
 * incomplete type so this header stands alone; C11 allows the identical
 * typedef again in pool.h. */
typedef struct tl_pool tl_pool;

/* C = alpha * A @ op(B) + beta * C, single precision, row-major.
 *
 *   A  M x K, element (i, k) at A[i * lda + k], lda >= K
 *   B  trans_b == 0: K x N, element (k, j) at B[k * ldb + j], ldb >= N
 *      trans_b != 0: N x K, element (j, k) at B[j * ldb + k], ldb >= K
 *      (op(B) is then its transpose, so a Linear weight [out, in] is used
 *      as stored)
 *   C  M x N, element (i, j) at C[i * ldc + j], ldc >= N
 *
 * beta == 0 writes C without reading it, so NaN or garbage already in C
 * does not propagate. M == 0 or N == 0 is a no-op; K == 0 gives
 * C = beta * C. A pointer may be NULL only when no element is read through
 * it. C must not overlap A or B.
 *
 * Returns TL_OK, or TL_EINVAL (error slot set, C untouched) for a negative
 * dimension, a leading dimension below its minimum, or a NULL pointer that
 * would be read or written.
 *
 * tp may be NULL (serial). v0 ignores tp and may sum over k in any order;
 * course tests compare against numpy within the frozen dot-product bound
 * of close.py. From L9.1 the k order is fixed and independent of M and of
 * the row's position (batch invariance, c/ABI.md). */
tl_status tl_matmul_f32(const float *A, const float *B, float *C,
                        int64_t M, int64_t N, int64_t K,
                        int64_t lda, int64_t ldb, int64_t ldc,
                        float alpha, float beta, int trans_b, tl_pool *tp);


#endif /* TINYLLM_MATMUL_H */
