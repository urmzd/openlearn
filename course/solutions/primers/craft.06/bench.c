/* primers/craft.06/bench.c: the benchmark of the kata.
 *
 * One program, built against whichever kernels.c the gate hands it. It
 * times each kernel with ol_bench_best (best of 5 repetitions, each at least
 * 20 ms of calls, after one warm-up call) and prints nanoseconds per call as
 * the last stdout line: {"matmul_ns": ..., "sum_ns": ..., "rmsnorm_ns": ...,
 * "count_below_ns": ...}. Inputs are fixed formulas (no RNG), sized so the
 * fast kernels run in microseconds to a millisecond and a slow one still
 * finishes in well under a second. Every result goes to a volatile sink so
 * the compiler cannot delete the work. */
/* ol_bench.h comes first: it defines _POSIX_C_SOURCE for clock_gettime, and
 * glibc ignores that once any other system header is in (under -std=c11 a
 * later define leaves CLOCK_MONOTONIC undeclared on Linux). */
#include "ol_bench.h"

#include <stdlib.h>

#include "kernels.h"

#define NM 256        /* matmul: 256^3 = 16.8M multiply-adds */
#define NS (1 << 20)  /* sum: 4 MiB of floats */
#define RR 64         /* rmsnorm: 64 rows of 1024 */
#define RD 1024
#define NC (1 << 18)  /* count_below: 1 MiB sorted, 256 queries per call */
#define QC 256

static float *A, *B, *C, *X, *Y, *W, *SORTED;
static volatile float sink_f;
static volatile int sink_i;

static void run_matmul(void *ctx) { (void)ctx; kata_matmul(A, B, C, NM); sink_f = C[NM * NM - 1]; }
static void run_sum(void *ctx) { (void)ctx; sink_f = kata_sum(X, NS); }
static void run_rmsnorm(void *ctx) { (void)ctx; kata_rmsnorm(X, W, Y, RR, RD, 1e-5f); sink_f = Y[RR * RD - 1]; }
static void run_count(void *ctx) {
    (void)ctx;
    int acc = 0;
    for (int q = 0; q < QC; q++) acc += kata_count_below(SORTED, NC, (float)((q * 7919) % NC) + 0.5f);
    sink_i = acc;
}

int main(void) {
    A = malloc(sizeof(float) * NM * NM);
    B = malloc(sizeof(float) * NM * NM);
    C = malloc(sizeof(float) * NM * NM);
    X = malloc(sizeof(float) * NS);
    Y = malloc(sizeof(float) * RR * RD);
    W = malloc(sizeof(float) * RD);
    SORTED = malloc(sizeof(float) * NC);
    if (!A || !B || !C || !X || !Y || !W || !SORTED) return 2;
    for (int i = 0; i < NM * NM; i++) {
        A[i] = (float)(i % 7) - 3.0f;
        B[i] = (float)(i % 5) - 2.0f;
    }
    for (int i = 0; i < NS; i++) X[i] = (float)(i % 13) * 0.25f - 1.5f;
    for (int i = 0; i < RD; i++) W[i] = 1.0f + (float)(i % 3) * 0.5f;
    for (int i = 0; i < NC; i++) SORTED[i] = (float)i;
    ol_bench_metric("matmul_ns", 1e9 * ol_bench_best(run_matmul, NULL, 5, 0.02));
    ol_bench_metric("sum_ns", 1e9 * ol_bench_best(run_sum, NULL, 5, 0.02));
    ol_bench_metric("rmsnorm_ns", 1e9 * ol_bench_best(run_rmsnorm, NULL, 5, 0.02));
    ol_bench_metric("count_below_ns", 1e9 * ol_bench_best(run_count, NULL, 5, 0.02));
    return ol_bench_done();
}
