/* Course tests for L9.3 (C side): c/src/kernels/flash_attn.c against
 * tinyllm/attention.h, under ASan and UBSan (and ThreadSanitizer for the
 * pool case). The oracle is the naive attention: the whole score row with
 * masks as -inf and the sink as one extra column, normalized by your L9.2
 * tl_softmax_f32, then the weighted sum of values. Fixed input/output
 * vectors are exchanged with the Python oracle as files.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_invariance.h"
#include "ol_prop.h"
#include "ol_test.h"

typedef struct {
    int64_t B, H, Hkv, Tq, Tk, D, q_offset, window;
    int causal;
    float scale;
    const float *sinks; /* [H] or NULL */
} shape;

static void fill(ol_gen *g, float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = ol_gen_f32(g, -1, 1);
}

/* The naive oracle; o [B,H,Tq,D], lse [B,H,Tq] in double. */
static int naive(const shape *s, const float *q, const float *k, const float *v, float *o, double *lse) {
    float *row = malloc(sizeof(float) * (size_t)(s->Tk + 1)), *p = malloc(sizeof(float) * (size_t)(s->Tk + 1));
    if (!row || !p) return 0;
    for (int64_t b = 0; b < s->B; b++)
        for (int64_t h = 0; h < s->H; h++) {
            int64_t kvh = h / (s->H / s->Hkv);
            for (int64_t i = 0; i < s->Tq; i++) {
                int64_t pos = s->q_offset + i, n = 0;
                double mx = -INFINITY;
                for (int64_t j = 0; j < s->Tk; j++) {
                    int vis = (!s->causal || j <= pos) && (s->window <= 0 || j > pos - s->window);
                    double dot = 0;
                    for (int64_t d = 0; d < s->D; d++)
                        dot += (double)q[((b * s->H + h) * s->Tq + i) * s->D + d] *
                               k[((b * s->Hkv + kvh) * s->Tk + j) * s->D + d];
                    row[n++] = vis ? (float)(s->scale * dot) : -INFINITY;
                    if (vis && s->scale * dot > mx) mx = s->scale * dot;
                }
                if (s->sinks) {
                    row[n++] = s->sinks[h];
                    if (s->sinks[h] > mx) mx = s->sinks[h];
                }
                if (tl_softmax_f32(row, p, 1, n) != TL_OK) return 0;
                double z = 0;
                for (int64_t j = 0; j < n; j++)
                    if (row[j] != -INFINITY) z += exp((double)row[j] - mx);
                lse[(b * s->H + h) * s->Tq + i] = z > 0 ? mx + log(z) : -INFINITY;
                for (int64_t d = 0; d < s->D; d++) {
                    double acc = 0;
                    for (int64_t j = 0; j < s->Tk; j++) acc += (double)p[j] * v[((b * s->Hkv + kvh) * s->Tk + j) * s->D + d];
                    o[((b * s->H + h) * s->Tq + i) * s->D + d] = (float)acc;
                }
            }
        }
    free(row), free(p);
    return 1;
}

static tl_status flash(const shape *s, const float *q, const float *k, const float *v, float *o, float *lse,
                       int64_t Br, int64_t Bc, tl_arena *ar, tl_pool *tp) {
    return tl_flash_attn_fwd_f32(q, k, v, o, lse, s->B, s->H, s->Hkv, s->Tq, s->Tk, s->D, s->scale, s->q_offset,
                                 s->causal, s->window, s->sinks, Br, Bc, ar, tp);
}

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example: one query q = [1, 0], three keys
     *      [1, 0], [0, 1], [1, 1] (scores 1, 0, 1), values [1, 2], [3, 4],
     *      [5, 6], no mask. o = (v0 + e^-1 v1 + v2) / (2 + e^-1) = [3, 4] and
     *      lse = 1 + ln(2 + e^-1) = 1.8619948. Bc = 2 splits the keys into
     *      two tiles, so the online update runs once with a rescale of 1.
     * KIND: unit, smoke
     * CATCHES: s08, m02
     * CHAPTER: L9.3 section 3 */
    const float q[2] = {1, 0}, k[6] = {1, 0, 0, 1, 1, 1}, v[6] = {1, 2, 3, 4, 5, 6};
    float o[2] = {0}, lse = 0;
    OL_EQ(tl_flash_attn_fwd_f32(q, k, v, o, &lse, 1, 1, 1, 1, 3, 2, 1.0f, 0, 0, 0, NULL, 1, 2, NULL, NULL), TL_OK);
    OL_CLOSE(o[0], 3.0, 1e-6, 1e-6);
    OL_CLOSE(o[1], 4.0, 1e-6, 1e-6);
    OL_CLOSE(lse, 1.8619948, 1e-6, 1e-6);
}

static int matches_naive(const shape *s, int64_t Br, int64_t Bc, uint64_t seed) {
    int64_t nq = s->B * s->H * s->Tq * s->D, nk = s->B * s->Hkv * s->Tk * s->D, nl = s->B * s->H * s->Tq;
    float *q = malloc(sizeof(float) * (size_t)nq), *k = malloc(sizeof(float) * (size_t)(nk + 1));
    float *v = malloc(sizeof(float) * (size_t)(nk + 1)), *o = malloc(sizeof(float) * (size_t)nq);
    float *o2 = malloc(sizeof(float) * (size_t)nq), *lse = malloc(sizeof(float) * (size_t)nl);
    double *lse2 = malloc(sizeof(double) * (size_t)nl);
    ol_gen g = {seed};
    fill(&g, q, nq), fill(&g, k, nk), fill(&g, v, nk);
    int ok = flash(s, q, k, v, o, lse, Br, Bc, NULL, NULL) == TL_OK && naive(s, q, k, v, o2, lse2);
    for (int64_t i = 0; ok && i < nq; i++)
        if (!(fabs((double)o[i] - o2[i]) <= 1e-5 + 1e-4 * fabs(o2[i]))) {
            printf("    o[%lld] = %.9g, naive %.9g\n", (long long)i, o[i], o2[i]);
            ok = 0;
        }
    for (int64_t i = 0; ok && i < nl; i++)
        if (!(lse2[i] == -INFINITY ? lse[i] == -INFINITY : fabs(lse[i] - lse2[i]) <= 1e-5 + 1e-5 * fabs(lse2[i]))) {
            printf("    lse[%lld] = %.9g, naive %.9g\n", (long long)i, lse[i], lse2[i]);
            ok = 0;
        }
    free(q), free(k), free(v), free(o), free(o2), free(lse), free(lse2);
    return ok;
}

OL_TEST(gqa_causal_window_match_naive) {
    /* WHY: the kernel against the naive oracle on the cases a Llama-family
     *      model hits: GQA (6 query heads on 2 KV heads, SmolLM2 has 9 on
     *      3), causal prefill, a chunk at q_offset into a longer cache, a
     *      sliding window (Mistral), and tile sizes that divide nothing.
     * KIND: differential
     * CATCHES: s01, s02, s03, s04, s05, s06, m01
     * CHAPTER: L9.3 section 4 */
    const shape cases[] = {
        {2, 6, 2, 13, 13, 24, 0, 0, 1, 0.2f, NULL},  /* causal prefill */
        {1, 4, 1, 5, 29, 16, 24, 0, 1, 0.25f, NULL}, /* MQA, a chunk at q_offset 24 */
        {1, 3, 3, 17, 17, 8, 0, 6, 1, 0.35f, NULL},  /* sliding window 6 */
        {1, 2, 1, 7, 31, 8, 10, 5, 0, 0.3f, NULL},   /* window without causal */
        {1, 2, 2, 9, 11, 12, 0, 0, 0, 0.3f, NULL},   /* plain, non-causal */
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        OL_TRUE(matches_naive(&cases[c], 4, 5, 100 + c));
        OL_TRUE(matches_naive(&cases[c], 0, 0, 200 + c)); /* the default 64 x 64 tiles */
    }
}

OL_TEST(sinks_join_the_denominator) {
    /* WHY: a learned sink (gpt-oss) is one more score per head that takes
     *      weight and gives no value, so with every value equal to 1 the
     *      output is exactly 1 - p_sink, and lse includes the sink. Checked
     *      against the naive oracle too, with a window, and for a head whose
     *      sink dominates.
     * KIND: unit
     * CATCHES: s07, s08
     * CHAPTER: L9.3 section 2 */
    const float sinks[2] = {0.5f, 6.0f};
    const shape s = {1, 2, 1, 6, 9, 8, 3, 4, 1, 0.3f, sinks};
    OL_TRUE(matches_naive(&s, 2, 3, 31));
    const float q[2] = {0, 0}, k[4] = {0, 0, 0, 0}, v[4] = {1, 1, 1, 1};
    float o[2], lse;
    const float sink = 0.0f;
    /* two keys with score 0 and a sink 0: p_sink = 1/3, o = 2/3, lse = ln 3 */
    OL_EQ(tl_flash_attn_fwd_f32(q, k, v, o, &lse, 1, 1, 1, 1, 2, 2, 1.0f, 0, 0, 0, &sink, 0, 0, NULL, NULL), TL_OK);
    OL_CLOSE(o[0], 2.0 / 3.0, 1e-6, 1e-7);
    OL_CLOSE(lse, log(3.0), 1e-6, 1e-7);
}

OL_TEST(no_visible_key_gives_zeros_and_minus_inf) {
    /* WHY: a query past the end of the cache with a window that excludes
     *      every key (position 5, window 2, keys 0 and 1) sees nothing. The
     *      answer is zeros and lse = -inf, never 0/0 = NaN; with a sink the
     *      output is still zeros and lse is the sink logit.
     * KIND: boundary
     * CATCHES: s09
     * CHAPTER: L9.3 section 5, Pitfalls */
    const float q[2] = {1, 1}, k[4] = {1, 0, 0, 1}, v[4] = {1, 2, 3, 4};
    float o[2] = {7, 7}, lse = 7;
    OL_EQ(tl_flash_attn_fwd_f32(q, k, v, o, &lse, 1, 1, 1, 1, 2, 2, 1.0f, 5, 1, 2, NULL, 0, 0, NULL, NULL), TL_OK);
    OL_EQ(o[0], 0.0f);
    OL_EQ(o[1], 0.0f);
    OL_TRUE(isinf(lse) && lse < 0);
    const float sink = 1.5f;
    OL_EQ(tl_flash_attn_fwd_f32(q, k, v, o, &lse, 1, 1, 1, 1, 2, 2, 1.0f, 5, 1, 2, &sink, 0, 0, NULL, NULL), TL_OK);
    OL_EQ(o[0], 0.0f);
    OL_EQ(lse, 1.5f);
}

/* Chunked prefill: Tq = 37 query rows against Tk = 37 keys, causal and
 * windowed, computed in calls of `chunk` rows with q_offset = the chunk's
 * first row. Every chunk size must give the bits of one call. */
enum { CT = 37, CD = 16, CH = 4, CHKV = 2 };
static float c_q[CH * CT * CD], c_k[CHKV * CT * CD], c_v[CHKV * CT * CD];
static void chunked(void *ctx, int64_t chunk, float *out) {
    (void)ctx;
    static float part[CH * CT * CD];
    for (int64_t i0 = 0; i0 < CT; i0 += chunk) {
        int64_t n = CT - i0 < chunk ? CT - i0 : chunk;
        static float qc[CH * CT * CD];
        for (int64_t h = 0; h < CH; h++)
            memcpy(qc + h * n * CD, c_q + (h * CT + i0) * CD, sizeof(float) * (size_t)(n * CD));
        if (tl_flash_attn_fwd_f32(qc, c_k, c_v, part, NULL, 1, CH, CHKV, n, CT, CD, 0.25f, i0, 1, 11, NULL, 4, 8,
                                  NULL, NULL) != TL_OK)
            abort();
        for (int64_t h = 0; h < CH; h++)
            memcpy(out + (h * CT + i0) * CD, part + h * n * CD, sizeof(float) * (size_t)(n * CD));
    }
}

OL_TEST(chunk_invariant_with_q_offset) {
    /* WHY: chunked prefill (L10.3) computes a prompt's queries in pieces,
     *      each with q_offset = its first position, against the cache so
     *      far. Because key tiles are aligned to absolute positions, a row
     *      sees the same tiles in the same order in every split: chunks of
     *      1, 3, 5, 16, and 37 rows give identical bits.
     * KIND: property
     * CATCHES: s06, s10, s14
     * CHAPTER: L9.3 section 2 */
    ol_gen g = {41};
    fill(&g, c_q, CH * CT * CD), fill(&g, c_k, CHKV * CT * CD), fill(&g, c_v, CHKV * CT * CD);
    const int64_t chunks[5] = {CT, 1, 3, 5, 16};
    OL_CHUNK_INVARIANT(chunked, NULL, chunks, 5, CH * CT * CD);
}

OL_TEST(query_tile_and_batch_invariant) {
    /* WHY: the query tile size Br is a tuning knob, not part of the math:
     *      Br = 1, 3, and 64 give the same bits. And a sequence in a batch
     *      of 3 gets the same bits as alone (L10.2 batches requests).
     * KIND: property
     * CATCHES: s10, s14
     * CHAPTER: L9.3 section 2 */
    enum { B = 3, H = 2, T = 19, D = 8 };
    static float q[B * H * T * D], k[B * H * T * D], v[B * H * T * D], o1[B * H * T * D], o3[B * H * T * D],
        o64[B * H * T * D], alone[H * T * D];
    ol_gen g = {42};
    fill(&g, q, B * H * T * D), fill(&g, k, B * H * T * D), fill(&g, v, B * H * T * D);
    const shape s = {B, H, H, T, T, D, 0, 7, 1, 0.3f, NULL};
    OL_EQ(flash(&s, q, k, v, o1, NULL, 1, 4, NULL, NULL), TL_OK);
    OL_EQ(flash(&s, q, k, v, o3, NULL, 3, 4, NULL, NULL), TL_OK);
    OL_EQ(flash(&s, q, k, v, o64, NULL, 64, 4, NULL, NULL), TL_OK);
    OL_BITS_EQ_F32(o3, o1, B * H * T * D);
    OL_BITS_EQ_F32(o64, o1, B * H * T * D);
    const shape s1 = {1, H, H, T, T, D, 0, 7, 1, 0.3f, NULL};
    OL_EQ(flash(&s1, q + H * T * D, k + H * T * D, v + H * T * D, alone, NULL, 3, 4, NULL, NULL), TL_OK);
    OL_BITS_EQ_F32(alone, o3 + H * T * D, H * T * D);
}

OL_TEST(arena_rewound_and_scratch_independent_of_tk) {
    /* WHY: the point of FlashAttention is memory: scratch is Br rows of
     *      state and one tile of scores, whatever the context length. The
     *      arena's high-water mark is the same for Tk = 16 and Tk = 1000,
     *      and the kernel leaves the caller's arena at the mark it found.
     * KIND: property
     * CATCHES: s11, s12
     * CHAPTER: L9.3 section 2 */
    enum { T = 1000, D = 8 };
    static float q[4 * D], k[T * D], v[T * D], o[4 * D];
    ol_gen g = {43};
    fill(&g, q, 4 * D), fill(&g, k, T * D), fill(&g, v, T * D);
    tl_arena *ar = NULL;
    OL_EQ(tl_arena_create(1 << 16, &ar), TL_OK);
    tl_arena_mark before = tl_arena_mark_get(ar);
    tl_arena_stats a, b;
    tl_status st = tl_flash_attn_fwd_f32(q, k, v, o, NULL, 1, 1, 1, 4, 16, D, 0.3f, 12, 1, 0, NULL, 4, 16, ar, NULL);
    tl_arena_stats_get(ar, &a);
    tl_arena_mark after = tl_arena_mark_get(ar);
    tl_status st2 = tl_flash_attn_fwd_f32(q, k, v, o, NULL, 1, 1, 1, 4, T, D, 0.3f, T - 4, 1, 0, NULL, 4, 16, ar, NULL);
    tl_arena_stats_get(ar, &b);
    tl_arena_destroy(ar);
    OL_EQ(st, TL_OK);
    OL_EQ(st2, TL_OK);
    OL_EQ(after.block, before.block);
    OL_EQ(after.offset, before.offset);
    OL_EQ(a.bytes_used, 0);
    OL_EQ(b.high_water, a.high_water);
}

OL_TEST(pool_result_equals_serial_bitwise) {
    /* WHY: work items (sequence, head, query tile) run on rt.03 workers,
     *      each with its own scratch slice; a row's arithmetic does not
     *      depend on the worker, so 4 threads give the serial bits.
     * KIND: property
     * CATCHES: s13
     * CHAPTER: L9.3 section 2 */
    enum { B = 2, H = 4, T = 21, D = 8 };
    static float q[B * H * T * D], k[B * 2 * T * D], v[B * 2 * T * D], o1[B * H * T * D], o4[B * H * T * D];
    ol_gen g = {44};
    fill(&g, q, B * H * T * D), fill(&g, k, B * 2 * T * D), fill(&g, v, B * 2 * T * D);
    const shape s = {B, H, 2, T, T, D, 0, 0, 1, 0.3f, NULL};
    tl_pool *p = NULL;
    OL_EQ(tl_pool_create(4, &p), TL_OK);
    tl_status st = flash(&s, q, k, v, o4, NULL, 5, 4, NULL, p);
    tl_pool_destroy(p);
    OL_EQ(st, TL_OK);
    OL_EQ(flash(&s, q, k, v, o1, NULL, 5, 4, NULL, NULL), TL_OK);
    OL_BITS_EQ_F32(o4, o1, B * H * T * D);
}

OL_TEST(shape_and_argument_errors) {
    /* WHY: H must be a multiple of Hkv (TL_ESHAPE, as for 6 heads on 4 KV
     *      heads); negative sizes, a negative q_offset, and NULL tensors are
     *      TL_EINVAL, with o untouched. Tk = 0 is legal: no key, zeros.
     * KIND: boundary
     * CATCHES: s15
     * CHAPTER: L9.3 section 4 */
    float q[6 * 2] = {0}, k[4 * 2] = {0}, v[4 * 2] = {0}, o[6 * 2], lse[6];
    for (int i = 0; i < 12; i++) o[i] = 9;
    OL_EQ(tl_flash_attn_fwd_f32(q, k, v, o, lse, 1, 6, 4, 1, 1, 2, 1.0f, 0, 0, 0, NULL, 0, 0, NULL, NULL), TL_ESHAPE);
    OL_TRUE(strstr(tl_last_error(), "tl_flash_attn_fwd_f32") != NULL);
    OL_EQ(tl_flash_attn_fwd_f32(q, k, v, o, lse, 1, 2, 4, 1, 1, 2, 1.0f, 0, 0, 0, NULL, 0, 0, NULL, NULL), TL_ESHAPE);
    OL_EQ(tl_flash_attn_fwd_f32(q, k, v, o, lse, 1, 2, 2, 1, 1, 2, 1.0f, -1, 0, 0, NULL, 0, 0, NULL, NULL), TL_EINVAL);
    OL_EQ(tl_flash_attn_fwd_f32(q, NULL, v, o, lse, 1, 2, 2, 1, 1, 2, 1.0f, 0, 0, 0, NULL, 0, 0, NULL, NULL), TL_EINVAL);
    for (int i = 0; i < 12; i++) OL_EQ(o[i], 9.0f);
    OL_EQ(tl_flash_attn_fwd_f32(q, NULL, NULL, o, lse, 1, 2, 2, 1, 0, 2, 1.0f, 0, 1, 0, NULL, 0, 0, NULL, NULL), TL_OK);
    OL_EQ(o[0], 0.0f);
    OL_TRUE(isinf(lse[0]) && lse[0] < 0);
}

int main(void) { return OL_RUN_ALL(); }
