/* primers/craft.06/kernels.h: the craft.06 kata (given; do not edit).
 *
 * Four small kernels whose speed depends on how they are written, not on
 * what they compute. Each has a plain version that is correct and slow; the
 * kata asks for the fast version, and your benchmark and gate must notice
 * when a change makes one slow again.
 *
 *   kata_matmul       C = A @ B, n x n row-major float32. Cache-friendly loop
 *                     order (i, k, j: the inner loop walks rows of B and C),
 *                     tiled over k and j so a block of B stays in cache.
 *   kata_sum          the sum of x[0..n), with eight independent partial
 *                     sums (the compiler may not reorder float additions, so
 *                     one accumulator is one long dependency chain).
 *   kata_rmsnorm      y[r, i] = x[r, i] / sqrt(mean_i x[r, i]^2 + eps) * w[i]
 *                     for rows x d; the sum of squares with eight partial
 *                     sums, one division per row.
 *   kata_count_below  how many entries of sorted[0..n) (ascending) are < t:
 *                     a binary search, O(log n), touching no other memory.
 *
 * Results: matmul and rmsnorm within the float32 rounding of the plain
 * version (course/tests/craft.06/kata_check.c), sum within n * 2^-24 of
 * the exact sum relative to sum |x|, count_below exact. */
#ifndef CRAFT06_KERNELS_H
#define CRAFT06_KERNELS_H

void kata_matmul(const float *A, const float *B, float *C, int n);
float kata_sum(const float *x, int n);
void kata_rmsnorm(const float *x, const float *w, float *y, int rows, int d, float eps);
int kata_count_below(const float *sorted, int n, float t);

#endif
