/* turboquant.c — the math of TurboQuant (arXiv:2504.19874), distilled to C.
 *
 * TurboQuant's idea (README section 5.8): a random rotation makes ANY vector
 * look sphere-uniform, whose coordinate marginals are KNOWN, so you can apply a
 * fixed near-optimal scalar quantizer per coordinate with NO calibration data.
 * For inner products (attention needs ⟨q,k⟩) a pure MSE quantizer is biased, so
 * a 1-bit QJL sign sketch on the residual restores an UNBIASED estimate.
 *
 * This file demonstrates the three load-bearing facts:
 *   (1) the randomized Hadamard rotation R = (1/√n) H D is orthonormal, so it
 *       preserves inner products and norms  ->  ⟨Rx, Rk⟩ = ⟨x, k⟩;
 *   (2) per-coordinate quantization on the rotated vector has low MSE, but
 *       MSE-optimal != inner-product-optimal: quantizing both operands leaves a
 *       biased <q,k> (the paper's motivation for the stage-2 residual sketch);
 *   (3) a 1-bit QJL/SimHash sketch  sign(⟨g, u⟩)  recovers the angle (hence the
 *       inner product) UNBIASEDLY from one bit per random projection.
 *
 * Build & run:  cc -O2 -o turbo turboquant.c -lm && ./turbo
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define N 1024            /* must be a power of two for the FWHT */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* deterministic LCG -> uniform in [0,1) so the demo is reproducible */
static unsigned long g_state = 88172645463325252UL;
static double urand(void) {
    g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17;
    return ((g_state >> 11) & 0xFFFFFFFFFFFFFUL) / (double)0x10000000000000UL;
}
static double gauss(void) {                       /* Box–Muller */
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* In-place Fast Walsh–Hadamard Transform with orthonormal 1/√n scaling: O(n log n). */
static void fwht(float* a, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float u = a[j], v = a[j + len];
                a[j] = u + v; a[j + len] = u - v;
            }
    float inv = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) a[i] *= inv;
}

/* Forward rotation R x = (1/√n) H (D x): random sign flip D, then FWHT.
 * Inverse R^T y = (1/√n) D H y = D .* fwht(y), since H,D are symmetric and D²=I. */
static void rotate(const float* x, const int8_t* sign, float* y, int n) {
    for (int i = 0; i < n; i++) y[i] = sign[i] * x[i];
    fwht(y, n);
}
static void unrotate(const float* y, const int8_t* sign, float* x, int n) {
    for (int i = 0; i < n; i++) x[i] = y[i];
    fwht(x, n);                                  /* (1/√n) H y */
    for (int i = 0; i < n; i++) x[i] *= sign[i]; /* D·(1/√n)H y = Rᵀy; round-trip RᵀR = D²(1/n)H² = I, exact */
}

static double dot(const float* a, const float* b, int n) {
    double s = 0.0; for (int i = 0; i < n; i++) s += (double)a[i] * b[i]; return s;
}
static double norm(const float* a, int n) { return sqrt(dot(a, a, n)); }

int main(void) {
    float x[N], k[N], rx[N], rk[N], xq[N], rec[N];
    int8_t sign[N];

    for (int i = 0; i < N; i++) {            /* two correlated vectors + a few outliers */
        double base = gauss();
        x[i] = (float)(base + 0.3 * gauss());
        k[i] = (float)(base + 0.3 * gauss());
        sign[i] = (urand() < 0.5) ? -1 : 1;  /* the random ±1 diagonal D */
    }
    x[3] = 18.0f; x[300] = -15.0f;           /* outlier coordinates the rotation will spread out */

    /* (1) rotation is orthonormal -> preserves inner products and norms ------ */
    rotate(x, sign, rx, N);
    rotate(k, sign, rk, N);
    printf("(1) rotation R = (1/sqrt(n)) H D is orthonormal:\n");
    printf("    <x,k>      = %.4f      <Rx,Rk> = %.4f   (preserved)\n", dot(x,k,N), dot(rx,rk,N));
    printf("    ||x||      = %.4f      ||Rx||  = %.4f   (preserved)\n", norm(x,N), norm(rx,N));
    float maxbefore = 0, maxafter = 0;
    for (int i = 0; i < N; i++) { float ax=fabsf(x[i]),ar=fabsf(rx[i]);
        if (ax>maxbefore) maxbefore=ax; if (ar>maxafter) maxafter=ar; }
    printf("    max|coord| : before=%.2f  after=%.2f  -> outliers spread out by rotation\n\n",
           maxbefore, maxafter);

    /* (2) per-coordinate quantization on the rotated vector. After rotation the
     *     coords are ~N(0, ||x||^2/n) with a KNOWN marginal, so a fixed quantizer
     *     (here uniform b-bit over [-c,c]; Lloyd–Max levels would be optimal) needs
     *     no calibration. MSE is low, but inner products are BIASED (shrunk).    */
    int bits = 4;
    int levels = (1 << bits) - 1;
    float c = 3.0f * (float)(norm(rx, N) / sqrt((double)N));  /* ~3 sigma clip */
    float step = 2.0f * c / levels;
    for (int i = 0; i < N; i++) {
        float v = rx[i] < -c ? -c : (rx[i] > c ? c : rx[i]);
        int q = (int)lrintf((v + c) / step);
        xq[i] = q * step - c;                                  /* dequantized rotated coord */
    }
    unrotate(xq, sign, rec, N);                                /* back to original space */
    double e = 0; for (int i = 0; i < N; i++){ double d=x[i]-rec[i]; e+=d*d; }
    printf("(2) %d-bit per-coordinate quantization (data-oblivious, no calibration):\n", bits);
    printf("    reconstruction MSE      = %.4e\n", e / N);
    printf("    true      <x,k>         = %.4f\n", dot(x, k, N));
    printf("    quantized <x̂,k>         = %.4f   <- MSE-optimal != inner-product-optimal\n\n",
           dot(rec, k, N));

    /* (3) QJL / SimHash: 1 bit per random projection recovers the ANGLE
     *     unbiasedly, since  E[ sign(<g,u>) == sign(<g,v>) ] = 1 - angle/pi.
     *     This is TurboQuant's stage-2 residual sketch for unbiased inner products. */
    int R = 4096;                                  /* number of 1-bit projections */
    long agree = 0;
    float g[N];
    for (int r = 0; r < R; r++) {
        for (int i = 0; i < N; i++) g[i] = (float)gauss();
        int su = dot(g, x, N) >= 0 ? 1 : -1;
        int sv = dot(g, k, N) >= 0 ? 1 : -1;
        agree += (su == sv);
    }
    double p_agree   = (double)agree / R;
    double angle_hat = M_PI * (1.0 - p_agree);     /* unbiased angle estimate */
    double ip_hat    = norm(x, N) * norm(k, N) * cos(angle_hat);
    double angle_true= acos(dot(x,k,N) / (norm(x,N)*norm(k,N)));
    printf("(3) QJL 1-bit sign sketch (%d projections, %d bits total):\n", R, R);
    printf("    true  angle = %.4f rad     est angle = %.4f rad\n", angle_true, angle_hat);
    printf("    true  <x,k> = %.4f         est <x,k> = %.4f   <- UNBIASED from 1-bit sketches\n",
           dot(x, k, N), ip_hat);
    printf("\nTakeaway: rotate (free, O(n log n)) -> known marginal -> fixed near-optimal\n"
           "quantizer + QJL residual. No calibration data, ideal for streaming KV cache.\n");
    return 0;
}
