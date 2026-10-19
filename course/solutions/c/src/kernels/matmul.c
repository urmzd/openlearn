/* c/src/kernels/matmul.c (L9.1, taking over M03.1's naive v0):
 * C = alpha * A @ op(B) + beta * C, cache-blocked, packed, and
 * batch-invariant. Contract: tinyllm/matmul.h; threads from rt.03.
 *
 * The loop nest is Goto's (Goto and van de Geijn 2008):
 *
 *   for each column strip of NC columns          (one rt.03 range each)
 *     for each block of KC values of k           (pack op(B): KC x NC)
 *       for each block of MC rows                (pack A: MC x KC)
 *         for each NR-wide, MR-tall micro-tile   (MR x NR accumulators)
 *
 * Packing copies a block into the order the micro-kernel reads it, padded
 * with zeros to whole micro-tiles, so every element of C goes through the
 * same arithmetic whatever M is and wherever its row sits in a tile. Each
 * output element is the sum over k-blocks, in increasing order, of a
 * partial sum that starts at 0 and adds its KC products in increasing k.
 * KC is a constant: nothing about the order depends on M, on the row's
 * position, or on the thread count (c/ABI.md rule 10).
 *
 * Packing buffers live on the stack (96 KiB): c/ABI.md rule 1 forbids
 * kernels to allocate, and the signature carries no scratch arena.
 */
#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/matmul.h"
#include "tinyllm/pool.h"

/* A fused multiply-add rounds once where a multiply then an add rounds
 * twice. If the compiler fused on one code path and not on another, an
 * element's bits would depend on where it sits in a tile; keep every
 * product and sum a separately rounded operation. */
#pragma STDC FP_CONTRACT OFF

#define MR 4   /* rows of a micro-tile */
#define NR 16  /* columns of a micro-tile */
#define KC 128 /* k values per block: the fixed reduction step */
#define MC 64  /* rows per packed block of A (a multiple of MR) */
#define NC 128 /* columns per strip (a multiple of NR) */

typedef struct {
    const float *A, *B;
    float *C;
    int64_t M, N, K, lda, ldb, ldc;
    float alpha, beta;
    int trans_b;
} mm_args;

/* op(B) for k in [k0, k0 + kc) and columns [j0, j0 + nc), as NR-wide
 * panels: panel p holds bp[p * kc * NR + k * NR + c] = op(B)[k0 + k][j0 + p * NR + c],
 * with zeros past column nc. */
static void pack_b(const mm_args *a, int64_t k0, int64_t kc, int64_t j0, int64_t nc, float *bp) {
    for (int64_t jp = 0; jp < nc; jp += NR) {
        for (int64_t k = 0; k < kc; k++) {
            for (int64_t c = 0; c < NR; c++) {
                int64_t j = j0 + jp + c;
                float v = 0.0f;
                if (jp + c < nc)
                    v = a->trans_b ? a->B[j * a->ldb + (k0 + k)] : a->B[(k0 + k) * a->ldb + j];
                *bp++ = v;
            }
        }
    }
}

/* A for rows [i0, i0 + mc) and k in [k0, k0 + kc), as MR-tall panels:
 * panel p holds ap[p * kc * MR + k * MR + r] = A[i0 + p * MR + r][k0 + k],
 * with zeros past row mc. */
static void pack_a(const mm_args *a, int64_t i0, int64_t mc, int64_t k0, int64_t kc, float *ap) {
    for (int64_t ip = 0; ip < mc; ip += MR) {
        for (int64_t k = 0; k < kc; k++) {
            for (int64_t r = 0; r < MR; r++) {
                *ap++ = ip + r < mc ? a->A[(i0 + ip + r) * a->lda + (k0 + k)] : 0.0f;
            }
        }
    }
}

/* acc = the MR x NR partial sums of one k-block, each from 0 in increasing k. */
static void micro_kernel(int64_t kc, const float *ap, const float *bp, float acc[MR][NR]) {
    for (int r = 0; r < MR; r++)
        for (int c = 0; c < NR; c++) acc[r][c] = 0.0f;
    for (int64_t k = 0; k < kc; k++) {
        const float *av = ap + k * MR, *bv = bp + k * NR;
        for (int r = 0; r < MR; r++)
            for (int c = 0; c < NR; c++) acc[r][c] += av[r] * bv[c];
    }
}

/* Writes the real mr x nr corner of a micro-tile into C. The first k-block
 * applies beta (and never reads C when beta == 0); later blocks add. */
static void store_tile(const mm_args *a, int64_t i, int64_t j, int64_t mr, int64_t nr, float acc[MR][NR],
                       int first) {
    for (int64_t r = 0; r < mr; r++) {
        float *c = a->C + (i + r) * a->ldc + j;
        for (int64_t q = 0; q < nr; q++) {
            float v = a->alpha * acc[r][q];
            if (!first)
                c[q] = c[q] + v;
            else
                c[q] = a->beta == 0.0f ? v : v + a->beta * c[q];
        }
    }
}

/* All of C's columns [j0, j0 + nc), every row, every k. */
static void column_strip(const mm_args *a, int64_t j0, int64_t nc) {
    float bp[KC * NC], ap[MC * KC];
    for (int64_t k0 = 0; k0 < a->K; k0 += KC) {
        int64_t kc = a->K - k0 < KC ? a->K - k0 : KC;
        pack_b(a, k0, kc, j0, nc, bp);
        for (int64_t i0 = 0; i0 < a->M; i0 += MC) {
            int64_t mc = a->M - i0 < MC ? a->M - i0 : MC;
            pack_a(a, i0, mc, k0, kc, ap);
            for (int64_t jr = 0; jr < nc; jr += NR) {
                for (int64_t ir = 0; ir < mc; ir += MR) {
                    float acc[MR][NR];
                    micro_kernel(kc, ap + ir * kc, bp + jr * kc, acc);
                    store_tile(a, i0 + ir, j0 + jr, mc - ir < MR ? mc - ir : MR, nc - jr < NR ? nc - jr : NR, acc,
                               k0 == 0);
                }
            }
        }
    }
}

/* rt.03 range function: strips [lo, hi). */
static void strips(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)worker;
    const mm_args *a = ctx;
    for (int64_t s = lo; s < hi; s++) {
        int64_t j0 = s * NC;
        column_strip(a, j0, a->N - j0 < NC ? a->N - j0 : NC);
    }
}

tl_status tl_matmul_f32(const float *A, const float *B, float *C,
                        int64_t M, int64_t N, int64_t K,
                        int64_t lda, int64_t ldb, int64_t ldc,
                        float alpha, float beta, int trans_b, tl_pool *tp) {
    /* 1. Validate everything before touching C (the M03.1 contract). */
    if (M < 0 || N < 0 || K < 0) {
        tl_set_last_error("tl_matmul_f32: negative dimension");
        return TL_EINVAL;
    }
    if (M == 0 || N == 0) return TL_OK;
    if (ldc < N) {
        tl_set_last_error("tl_matmul_f32: ldc < N");
        return TL_EINVAL;
    }
    if (C == NULL) {
        tl_set_last_error("tl_matmul_f32: C is NULL");
        return TL_EINVAL;
    }
    if (K > 0) {
        if (lda < K) {
            tl_set_last_error("tl_matmul_f32: lda < K");
            return TL_EINVAL;
        }
        if (trans_b ? ldb < K : ldb < N) {
            tl_set_last_error(trans_b ? "tl_matmul_f32: ldb < K (trans_b)" : "tl_matmul_f32: ldb < N");
            return TL_EINVAL;
        }
        if (A == NULL || B == NULL) {
            tl_set_last_error("tl_matmul_f32: A or B is NULL");
            return TL_EINVAL;
        }
    }

    /* 2. K == 0: no k-block runs, and the empty sum is 0, so C = beta * C. */
    if (K == 0) {
        for (int64_t i = 0; i < M; i++)
            for (int64_t j = 0; j < N; j++) C[i * ldc + j] = beta == 0.0f ? 0.0f : beta * C[i * ldc + j];
        return TL_OK;
    }

    /* 3. One range per column strip. A strip is computed by exactly one
     *    worker and its arithmetic does not depend on which, so the result
     *    is the same bits with any number of threads. */
    mm_args a = {A, B, C, M, N, K, lda, ldb, ldc, alpha, beta, trans_b};
    int64_t n_strips = (N + NC - 1) / NC;
    if (tp == NULL) {
        strips(&a, 0, n_strips, 0);
        return TL_OK;
    }
    return tl_parallel_for(tp, n_strips, 1, strips, &a);
}
