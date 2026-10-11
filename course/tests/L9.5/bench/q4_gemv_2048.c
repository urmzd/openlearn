/* Bench for L9.5 (a B test: `ol bench L9.5 [--assert]`, never `ol check`).
 * One decode step of a 2048 x 2048 Linear layer (M = 1): your fused int4
 * kernel (group 32) against your L9.1 tl_matmul_f32 on the float32 weight,
 * both serial. Reports
 *   q4_speedup_vs_f32   the budget in course/modules/L9.5.toml (>= 2)
 *   q8_speedup_vs_f32, f32_ms, q4_ms, q8_ms */
#include "ol_bench.h" /* first: it sets the POSIX feature macros clock_gettime needs */
#include <stdlib.h>

#include "tinyllm.h"

enum { N = 2048, K = 2048, G = 32 };
static float x[K], w[N * K], y[N];
static uint8_t q4[N * K / 2];
static uint16_t s4[N * (K / G)];
static int8_t q8[N * K];
static float s8[N];

static void f32(void *c) {
    (void)c;
    if (tl_matmul_f32(x, w, y, 1, N, K, K, K, N, 1.0f, 0.0f, 1, NULL) != TL_OK) abort();
}
static void fq4(void *c) {
    (void)c;
    if (tl_matmul_q4_f32(x, q4, s4, y, 1, N, K, G, NULL) != TL_OK) abort();
}
static void fq8(void *c) {
    (void)c;
    if (tl_matmul_q8_f32(x, q8, s8, y, 1, N, K, NULL) != TL_OK) abort();
}

int main(void) {
    for (int i = 0; i < K; i++) x[i] = (float)(i % 7) - 3.0f;
    for (int i = 0; i < N * K; i++) {
        w[i] = (float)(i % 13) / 13.0f - 0.5f;
        q8[i] = (int8_t)(i % 255 - 127);
    }
    for (int i = 0; i < N * K / 2; i++) q4[i] = (uint8_t)(i * 37);
    for (int i = 0; i < N * (K / G); i++) s4[i] = tl_f32_to_f16(0.01f);
    for (int i = 0; i < N; i++) s8[i] = 0.01f;
    double t32 = ol_bench_best(f32, NULL, 3, 0.2);
    double t4 = ol_bench_best(fq4, NULL, 3, 0.2);
    double t8 = ol_bench_best(fq8, NULL, 3, 0.2);
    ol_bench_metric("f32_ms", t32 * 1e3);
    ol_bench_metric("q4_ms", t4 * 1e3);
    ol_bench_metric("q8_ms", t8 * 1e3);
    ol_bench_metric("q4_speedup_vs_f32", t32 / t4);
    ol_bench_metric("q8_speedup_vs_f32", t32 / t8);
    return ol_bench_done();
}
