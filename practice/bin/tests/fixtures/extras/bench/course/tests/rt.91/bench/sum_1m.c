/* Course benchmark for rt.91: tl_demo_sum_f32 over one million floats. */
#include <stdlib.h>

#include "tinyllm.h"
#include "ol_bench.h"

#define N (1 << 20)
static float x[N];

static void step(void *ctx) {
    float y = 0.0f;
    tl_demo_sum_f32(x, N, &y);
    *(volatile float *)ctx = y;
}

int main(void) {
    for (int i = 0; i < N; i++) x[i] = (float)(i % 7);
    volatile float sink = 0.0f;
    double s = ol_bench_best(step, (void *)&sink, 3, 0.05);
    ol_bench_metric("gelem_per_s", N / s / 1e9);
    return ol_bench_done();
}
