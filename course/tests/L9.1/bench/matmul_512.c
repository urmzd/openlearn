/* Bench for L9.1 (a B test: `ol bench L9.1 [--assert]`, never `ol check`).
 * Times your tiled tl_matmul_f32 at 512 x 512 x 512 (serial, tp = NULL)
 * against M03.1's triple loop compiled here, and reports
 *   speedup_vs_naive   the budget in course/modules/L9.1.toml (>= 10)
 *   gflops, naive_gflops
 * Both run at -O2 on the same float32 inputs. */
#include "ol_bench.h" /* first: it sets the POSIX feature macros clock_gettime needs */
#include <stdlib.h>

#include "tinyllm.h"

enum { S = 512 };
static float A[S * S], B[S * S], C[S * S];

/* M03.1's v0: one dot product per output, k innermost, B walked down a
 * column (a new cache line on almost every step). */
static void naive(void *ctx) {
    (void)ctx;
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            float acc = 0.0f;
            for (int k = 0; k < S; k++) acc += A[i * S + k] * B[k * S + j];
            C[i * S + j] = acc;
        }
}

static void tiled(void *ctx) {
    (void)ctx;
    if (tl_matmul_f32(A, B, C, S, S, S, S, S, S, 1.0f, 0.0f, 0, NULL) != TL_OK) abort();
}

int main(void) {
    uint64_t s = 1;
    for (int i = 0; i < S * S; i++) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        A[i] = (float)((s >> 40) & 0xffff) / 65536.0f - 0.5f;
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        B[i] = (float)((s >> 40) & 0xffff) / 65536.0f - 0.5f;
    }
    double flops = 2.0 * S * S * S;
    double t_naive = ol_bench_best(naive, NULL, 2, 0.2);
    double t_tiled = ol_bench_best(tiled, NULL, 3, 0.2);
    ol_bench_metric("naive_gflops", flops / t_naive / 1e9);
    ol_bench_metric("gflops", flops / t_tiled / 1e9);
    ol_bench_metric("speedup_vs_naive", t_naive / t_tiled);
    return ol_bench_done();
}
