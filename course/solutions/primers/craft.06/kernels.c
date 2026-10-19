/* primers/craft.06/kernels.c: the craft.06 kata (spec in kernels.h). */
#include <math.h>

#include "kernels.h"

/* Tile sizes for kata_matmul: a BK x BJ block of B (64 x 256 floats, 64 KiB)
 * is reused by every row i before the next block is loaded. */
enum { BK = 64, BJ = 256 };

static int min_i(int a, int b) { return a < b ? a : b; }

/* c[0..BJ) += a * b[0..BJ): a fixed trip count and restrict (no alias check
 * needed) are what let gcc vectorize at -O2, as clang already does. */
static void axpy_tile(float *restrict c, const float *restrict b, float a) {
    for (int j = 0; j < BJ; j++) c[j] += a * b[j];
}

void kata_matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = 0.0f;
    for (int k0 = 0; k0 < n; k0 += BK)
        for (int j0 = 0; j0 < n; j0 += BJ)
            for (int i = 0; i < n; i++)
                for (int k = k0; k < min_i(k0 + BK, n); k++) {
                    float a = A[i * n + k];
                    if (j0 + BJ <= n) { /* a whole tile: unit stride over j, vectorized */
                        axpy_tile(C + i * n + j0, B + k * n + j0, a);
                        continue;
                    }
                    for (int j = j0; j < n; j++) C[i * n + j] += a * B[k * n + j];
                }
}

float kata_sum(const float *x, int n) {
    float s[8] = {0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int l = 0; l < 8; l++) s[l] += x[i + l];
    for (; i < n; i++) s[0] += x[i];
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

void kata_rmsnorm(const float *x, const float *w, float *y, int rows, int d, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (long)r * d;
        float *yr = y + (long)r * d;
        float s[8] = {0};
        int i = 0;
        for (; i + 8 <= d; i += 8)
            for (int l = 0; l < 8; l++) s[l] += xr[i + l] * xr[i + l];
        for (; i < d; i++) s[0] += xr[i] * xr[i];
        float ss = ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
        float inv = 1.0f / sqrtf(ss / (float)d + eps);
        for (int k = 0; k < d; k++) yr[k] = xr[k] * inv * w[k];
    }
}

int kata_count_below(const float *sorted, int n, float t) {
    int lo = 0, hi = n; /* invariant: sorted[0..lo) < t <= sorted[hi..n) */
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (sorted[mid] < t)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}
