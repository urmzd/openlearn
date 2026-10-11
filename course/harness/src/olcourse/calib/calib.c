/* ol bench --calibrate (course/DESIGN.md 5.11): how fast is this machine?
 *
 * Two kernels stand in for the course's perf-critical paths, so every
 * course perf budget can be written relative to them:
 *   matmul_gflops  a cache-tiled 256 x 256 x 256 float32 matmul (compute bound)
 *   decode_tok_s   one "decode step" = a 2048 x 512 float32 GEMV, a softmax
 *                  over the 2048 outputs, and an argmax (memory bound)
 * Each is timed as the best of several repetitions. Output: one JSON line.
 * Built with -O2 and no -march flags, so the number means the same thing on
 * the host and inside the in-cluster Job. */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t rng_state = 2463534242u;
static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float)(rng_state & 0xffffff) / 16777216.0f - 0.5f;
}

#define N 256
#define T 32

static void matmul_tiled(const float *a, const float *b, float *c) {
    for (int i = 0; i < N * N; i++) c[i] = 0.0f;
    for (int ii = 0; ii < N; ii += T)
        for (int kk = 0; kk < N; kk += T)
            for (int jj = 0; jj < N; jj += T)
                for (int i = ii; i < ii + T; i++)
                    for (int k = kk; k < kk + T; k++) {
                        float aik = a[i * N + k];
                        for (int j = jj; j < jj + T; j++) c[i * N + j] += aik * b[k * N + j];
                    }
}

#define V 2048
#define D 512

static int decode_step(const float *w, const float *x, float *logits) {
    for (int v = 0; v < V; v++) {
        float s = 0.0f;
        for (int d = 0; d < D; d++) s += w[v * D + d] * x[d];
        logits[v] = s;
    }
    float m = logits[0];
    for (int v = 1; v < V; v++) m = logits[v] > m ? logits[v] : m;
    double z = 0.0;
    for (int v = 0; v < V; v++) z += exp((double)(logits[v] - m));
    int best = 0;
    for (int v = 0; v < V; v++) {
        logits[v] = (float)(exp((double)(logits[v] - m)) / z);
        if (logits[v] > logits[best]) best = v;
    }
    return best;
}

int main(void) {
    float *a = malloc(sizeof(float) * N * N), *b = malloc(sizeof(float) * N * N), *c = malloc(sizeof(float) * N * N);
    float *w = malloc(sizeof(float) * V * D), *x = malloc(sizeof(float) * D), *lg = malloc(sizeof(float) * V);
    if (!a || !b || !c || !w || !x || !lg) return 1;
    for (int i = 0; i < N * N; i++) a[i] = frand(), b[i] = frand();
    for (int i = 0; i < V * D; i++) w[i] = frand();
    for (int i = 0; i < D; i++) x[i] = frand();

    double best_mm = 1e30;
    for (int r = 0; r < 7; r++) {
        double t0 = now();
        matmul_tiled(a, b, c);
        double dt = now() - t0;
        if (dt < best_mm) best_mm = dt;
    }
    double best_dec = 1e30;
    volatile int sink = 0;
    for (int r = 0; r < 7; r++) {
        double t0 = now();
        for (int s = 0; s < 8; s++) {
            sink += decode_step(w, x, lg);
            x[s % D] += 1e-3f;  /* the next step reads a new input */
        }
        double dt = (now() - t0) / 8.0;
        if (dt < best_dec) best_dec = dt;
    }
    printf("{\"matmul_gflops\": %.4f, \"decode_tok_s\": %.2f, \"checksum\": %.6f}\n",
           2.0 * N * N * N / best_mm / 1e9, 1.0 / best_dec, (double)c[N + 1] + (double)sink * 0.0);
    free(a), free(b), free(c), free(w), free(x), free(lg);
    return 0;
}
