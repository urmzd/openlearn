/* nf4_quant.c — NF4 (NormalFloat-4, QLoRA) and a GGUF Q4_K-style dequant in C.
 *
 * Two block-quantization schemes from README section 5.4 and 5.5:
 *   - NF4: a fixed 16-level *quantile* codebook for N(0,1) weights, blockwise
 *          absmax scale, plus "double quantization" of the block scales.
 *   - Q4_K: hierarchical block scales — a 256-weight super-block split into
 *           eight 32-blocks, each with its own 6-bit scale and min, the eight
 *           of which share two fp16 super-scales.
 *
 * Build & run:  cc -O2 -o nf4 nf4_quant.c -lm && ./nf4
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ---- NF4: the 16 NormalFloat levels (quantiles of N(0,1), from QLoRA) ----- */
static const float NF4[16] = {
    -1.0f,               -0.6961928f, -0.5250731f, -0.3949175f,
    -0.2844414f,         -0.1847734f, -0.0910500f,  0.0f,
     0.0795803f,          0.1609302f,  0.2461123f,  0.3379152f,
     0.4407098f,          0.5626170f,  0.7229568f,  1.0f
};

/* nearest codebook index for a value already normalized into [-1, 1] */
static uint8_t nf4_nearest(float v) {
    uint8_t best = 0; float bd = INFINITY;
    for (uint8_t i = 0; i < 16; i++) {
        float d = fabsf(v - NF4[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

/* Quantize one block of `bn` weights: scale = absmax, q = argmin |w/scale - NF4[i]|.
 * Returns the block scale; fills q[] with 4-bit codes. */
static float nf4_quantize_block(const float* w, int bn, uint8_t* q) {
    float amax = 0.0f;
    for (int i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(w[i]));
    float scale = amax > 0 ? amax : 1.0f;            /* maps weights into [-1, 1] */
    for (int i = 0; i < bn; i++) q[i] = nf4_nearest(w[i] / scale);
    return scale;
}

static void nf4_dequantize_block(const uint8_t* q, int bn, float scale, float* out) {
    for (int i = 0; i < bn; i++) out[i] = scale * NF4[q[i]];   /* ŵ = s · codebook[q] */
}

/* ---- GGUF Q4_K-style dequant: hierarchical block scales -------------------
 * Real ggml packs the eight 6-bit scales + eight 6-bit mins into 12 bytes and
 * the 256 nibbles into 128 bytes. We keep them in plain arrays for clarity;
 * the *math* (ŵ = d_super·scale6·q − dmin_super·min6) is identical.            */
#define QK_K     256        /* weights per super-block */
#define SUB      32         /* weights per sub-block   */
#define NSUB     (QK_K/SUB) /* 8 sub-blocks            */

typedef struct {
    float   d_super;          /* fp16 in real GGUF: super-scale for the scales */
    float   dmin_super;       /* fp16 in real GGUF: super-scale for the mins   */
    uint8_t scale6[NSUB];     /* 6-bit per-sub-block scale (0..63)             */
    uint8_t min6[NSUB];       /* 6-bit per-sub-block min   (0..63)             */
    uint8_t q4[QK_K];         /* 4-bit weight codes (0..15)                    */
} block_q4k;

static void dequant_q4k_block(const block_q4k* b, float* out) {
    for (int s = 0; s < NSUB; s++) {
        float d = b->d_super    * (float)b->scale6[s];   /* d_sub = d_super · scale6 */
        float m = b->dmin_super * (float)b->min6[s];     /* m_sub = dmin_super · min6 */
        for (int i = 0; i < SUB; i++) {
            uint8_t q = b->q4[s * SUB + i];
            out[s * SUB + i] = d * (float)q - m;          /* ŵ = d_sub·q − m_sub */
        }
    }
}

/* mean-squared error helper */
static double mse(const float* a, const float* b, int n) {
    double e = 0.0; for (int i = 0; i < n; i++) { double d = a[i]-b[i]; e += d*d; }
    return e / n;
}

int main(void) {
    /* ---- NF4 demo on Gaussian-ish weights (its design assumption) ---------- */
    enum { N = 256, BN = 64 };               /* NF4 default block = 64 */
    float w[N], rec[N];
    uint8_t codes[N];
    float block_scale[N / BN];

    /* deterministic pseudo-Gaussian (Box–Muller on a fixed LCG) */
    unsigned long st = 12345;
    for (int i = 0; i < N; i += 2) {
        st = st * 6364136223846793005UL + 1; double u1 = ((st >> 11) & 0xFFFFFFF) / (double)0xFFFFFFF;
        st = st * 6364136223846793005UL + 1; double u2 = ((st >> 11) & 0xFFFFFFF) / (double)0xFFFFFFF;
        if (u1 < 1e-9) u1 = 1e-9;
        double r = sqrt(-2.0 * log(u1));
        w[i]   = (float)(r * cos(6.2831853 * u2)) * 0.1f;
        w[i+1] = (float)(r * sin(6.2831853 * u2)) * 0.1f;
    }

    for (int blk = 0; blk < N / BN; blk++) {
        block_scale[blk] = nf4_quantize_block(w + blk * BN, BN, codes + blk * BN);
        nf4_dequantize_block(codes + blk * BN, BN, block_scale[blk], rec + blk * BN);
    }
    printf("NF4 (4-bit, block=%d, Gaussian weights):\n", BN);
    printf("  per-weight bits   = 4 + %.3f (block scales) = %.3f\n",
           32.0 / BN, 4.0 + 32.0 / BN);

    /* Double quantization: quantize the FP32 block scales to int8 (block 256). */
    float smax = 0.0f; for (int i = 0; i < N / BN; i++) smax = fmaxf(smax, block_scale[i]);
    float scale_of_scales = smax / 127.0f;
    printf("  with double-quant : 4 + %.3f = %.3f bits/weight\n",
           8.0 / BN + 32.0 / N, 4.0 + 8.0 / BN + 32.0 / N);
    printf("  NF4 reconstruction MSE = %.3e\n\n", mse(w, rec, N));
    (void)scale_of_scales;

    /* ---- Q4_K demo: build a super-block, round-trip it --------------------- */
    block_q4k b;
    float orig[QK_K], deq[QK_K];
    b.d_super = 0.0008f; b.dmin_super = 0.0006f;
    for (int s = 0; s < NSUB; s++) {
        b.scale6[s] = 20 + s;                 /* arbitrary 6-bit scales/mins */
        b.min6[s]   = 5  + (s % 4);
        for (int i = 0; i < SUB; i++) {
            uint8_t q = (uint8_t)((i * 7 + s * 3) & 0x0F);   /* arbitrary nibbles */
            b.q4[s * SUB + i] = q;
            orig[s*SUB+i] = b.d_super*(float)b.scale6[s]*(float)q
                          - b.dmin_super*(float)b.min6[s];   /* ground truth */
        }
    }
    dequant_q4k_block(&b, deq);
    printf("Q4_K (super-block=%d, %d sub-blocks of %d, 6-bit scales+mins):\n",
           QK_K, NSUB, SUB);
    printf("  effective bits/weight = 4 + (8*6 + 8*6 + 2*16)/%d = %.3f\n",
           QK_K, 4.0 + (8.0*6 + 8.0*6 + 2.0*16) / QK_K);
    printf("  dequant matches construction MSE = %.3e (round-trip exact)\n",
           mse(orig, deq, QK_K));
    return 0;
}
