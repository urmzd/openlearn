/* ol_bench.h: the single-header C benchmark kit (course/DESIGN.md 5.9, 5.11).
 *
 * Frozen helper. A course benchmark is a program under
 * course/tests/<ID>/bench/<name>.c that `ol bench <ID>` builds with -O2
 * against the overlay's objects and runs. It times the code under test with
 * ol_bench_best and reports named metrics; the last stdout line is one JSON
 * object, which `ol bench` compares with the module's budget relative to the
 * machine calibration (`ol bench --calibrate`).
 *
 *   #include "tinyllm.h"
 *   #include "ol_bench.h"
 *
 *   static float a[1 << 16], b[1 << 16], c[1 << 16];
 *   static void step(void *ctx) { (void)ctx; tl_matmul_f32(a, b, c, ...); }
 *
 *   int main(void) {
 *       double s = ol_bench_best(step, NULL, 3, 0.2);   // best seconds per call
 *       ol_bench_metric("gflops", 2.0 * M * N * K / s / 1e9);
 *       return ol_bench_done();                          // prints {"gflops": ...}
 *   }
 */
#ifndef OL_BENCH_H
#define OL_BENCH_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef OL_BENCH_MAX_METRICS
#define OL_BENCH_MAX_METRICS 16
#endif

static inline double ol_bench_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Best time per call, in seconds: `reps` repetitions, each calling fn enough
 * times to take at least min_s seconds (one warm-up call first). */
static inline double ol_bench_best(void (*fn)(void *), void *ctx, int reps, double min_s) {
    fn(ctx);
    double best = 1e300;
    for (int r = 0; r < reps; r++) {
        long n = 0;
        double t0 = ol_bench_now(), dt;
        do {
            fn(ctx);
            n++;
            dt = ol_bench_now() - t0;
        } while (dt < min_s);
        if (dt / (double)n < best) best = dt / (double)n;
    }
    return best;
}

static const char *ol_bench__names[OL_BENCH_MAX_METRICS];
static double ol_bench__values[OL_BENCH_MAX_METRICS];
static int ol_bench__n;

static inline void ol_bench_metric(const char *name, double value) {
    for (int i = 0; i < ol_bench__n; i++)
        if (strcmp(ol_bench__names[i], name) == 0) {
            ol_bench__values[i] = value;
            return;
        }
    if (ol_bench__n < OL_BENCH_MAX_METRICS) {
        ol_bench__names[ol_bench__n] = name;
        ol_bench__values[ol_bench__n++] = value;
    }
}

/* Prints the metrics as the last stdout line; returns 0 (an exit code). */
static inline int ol_bench_done(void) {
    putchar('{');
    for (int i = 0; i < ol_bench__n; i++)
        printf("%s\"%s\": %.6g", i ? ", " : "", ol_bench__names[i], ol_bench__values[i]);
    puts("}");
    fflush(stdout);
    return 0;
}

#endif /* OL_BENCH_H */
