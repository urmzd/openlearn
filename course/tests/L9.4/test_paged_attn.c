/* Course tests for L9.4 (C side): c/src/kernels/paged_attn.c against
 * tinyllm/attention.h, reading K and V from an rt.04 block pool, under ASan
 * and UBSan (and ThreadSanitizer for the pool case). Two oracles: the naive
 * attention over the gathered cache built on your L9.2 tl_softmax_f32, and
 * your L9.3 FlashAttention on the same gathered values, which a paged
 * decode must equal bit for bit. Paged KV inputs and outputs are recorded
 * in the shared file fixture, independent of the Python cache process.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_invariance.h"
#include "ol_prop.h"
#include "ol_test.h"

static void fill(ol_gen *g, float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) x[i] = ol_gen_f32(g, -1, 1);
}

static tl_kv_pool *make_pool(uint32_t n_blocks, uint32_t bt, uint32_t layers, uint32_t hkv, uint32_t d) {
    tl_kv_cfg cfg = {n_blocks, bt, layers, hkv, d, TL_F16, TL_KV_FORMAT_V1};
    tl_kv_pool *p = NULL;
    return tl_kv_pool_create(&cfg, &p) == TL_OK ? p : NULL;
}

/* Writes K and V of position j (all KV heads, [hkv][D] each) of a sequence
 * with block table `table` into layer `layer`, rounded to f16. */
static void put(tl_kv_pool *p, const uint32_t *table, int64_t j, uint32_t layer, const float *k, const float *v) {
    tl_kv_cfg c;
    tl_kv_pool_cfg(p, &c);
    uint16_t *kb = tl_kv_block_ptr(p, table[j / c.block_tokens], layer, 0);
    uint16_t *vb = tl_kv_block_ptr(p, table[j / c.block_tokens], layer, 1);
    for (uint32_t h = 0; h < c.n_kv_heads; h++)
        for (uint32_t d = 0; d < c.head_dim; d++) {
            size_t at = ((size_t)h * c.block_tokens + (size_t)(j % c.block_tokens)) * c.head_dim + d;
            kb[at] = tl_f32_to_f16(k[h * c.head_dim + d]);
            vb[at] = tl_f32_to_f16(v[h * c.head_dim + d]);
        }
}

/* The gathered cache of one sequence, f32: k, v [hkv][ctx][D]. */
static void gather(tl_kv_pool *p, const uint32_t *table, int64_t ctx, uint32_t layer, float *k, float *v) {
    tl_kv_cfg c;
    tl_kv_pool_cfg(p, &c);
    for (int64_t j = 0; j < ctx; j++) {
        const uint16_t *kb = tl_kv_block_ptr(p, table[j / c.block_tokens], layer, 0);
        const uint16_t *vb = tl_kv_block_ptr(p, table[j / c.block_tokens], layer, 1);
        for (uint32_t h = 0; h < c.n_kv_heads; h++)
            for (uint32_t d = 0; d < c.head_dim; d++) {
                size_t at = ((size_t)h * c.block_tokens + (size_t)(j % c.block_tokens)) * c.head_dim + d;
                k[((int64_t)h * ctx + j) * c.head_dim + d] = tl_f16_to_f32(kb[at]);
                v[((int64_t)h * ctx + j) * c.head_dim + d] = tl_f16_to_f32(vb[at]);
            }
    }
}

OL_TEST(hand_example) {
    /* WHY: the chapter's worked example, L9.3's numbers in a paged cache:
     *      keys [1,0], [0,1], [1,1] and values [1,2], [3,4], [5,6] at
     *      positions 0, 1, 2, two positions per block, in blocks 3 and 1 of
     *      a 4-block pool (not contiguous, not in order). The query [1, 0]
     *      is position 2 and sees all three: o = [3, 4]. Every value is
     *      exact in f16.
     * KIND: unit, smoke
     * CATCHES: s03, s07, s08
     * CHAPTER: L9.4 section 3 */
    tl_kv_pool *p = make_pool(4, 2, 1, 1, 2);
    OL_TRUE(p != NULL);
    const uint32_t table[2] = {3, 1};
    const float k[3][2] = {{1, 0}, {0, 1}, {1, 1}}, v[3][2] = {{1, 2}, {3, 4}, {5, 6}};
    for (int j = 0; j < 3; j++) put(p, table, j, 0, k[j], v[j]);
    const float q[2] = {1, 0};
    const int32_t ctx = 3;
    float o[2] = {0};
    tl_status st = tl_paged_attn_decode_f32(q, p, 0, table, 2, &ctx, o, 1, 1, 1, 2, 1.0f, 0, NULL);
    tl_kv_pool_destroy(p);
    OL_EQ(st, TL_OK);
    OL_CLOSE(o[0], 3.0, 1e-6, 1e-6);
    OL_CLOSE(o[1], 4.0, 1e-6, 1e-6);
}

enum { NB = 64, BT = 16, L = 2, HKV = 2, H = 6, D = 32, MAXB = 8 };

/* A pool with 3 sequences: lengths 1, 37, and 70, in shuffled blocks. */
typedef struct {
    tl_kv_pool *p;
    uint32_t tables[3 * MAXB];
    int32_t ctx[3];
    float q[3 * H * D];
} world;

static int build(world *w, uint64_t seed) {
    ol_gen g = {seed};
    w->p = make_pool(NB, BT, L, HKV, D);
    if (!w->p) return 0;
    uint32_t ids[NB];
    if (tl_kv_alloc(w->p, 3 * MAXB, ids) != TL_OK) return 0;
    for (int i = 3 * MAXB - 1; i > 0; i--) { /* shuffle: tables are not in id order */
        int r = (int)ol_gen_int(&g, 0, i);
        uint32_t t = ids[i];
        ids[i] = ids[r], ids[r] = t;
    }
    memcpy(w->tables, ids, sizeof w->tables);
    const int32_t lens[3] = {1, 37, 70};
    float kv[2][HKV * D];
    for (int b = 0; b < 3; b++) {
        w->ctx[b] = lens[b];
        for (int64_t j = 0; j < lens[b]; j++)
            for (uint32_t layer = 0; layer < L; layer++) {
                fill(&g, kv[0], HKV * D), fill(&g, kv[1], HKV * D);
                put(w->p, w->tables + b * MAXB, j, layer, kv[0], kv[1]);
            }
    }
    fill(&g, w->q, 3 * H * D);
    return 1;
}

/* Naive attention for one (b, h) over the gathered f32 cache. */
static int naive_row(const float *q, const float *k, const float *v, int64_t ctx, int64_t window, float scale,
                     float *o) {
    float row[128], pr[128];
    int64_t p = ctx - 1;
    for (int64_t j = 0; j < ctx; j++) {
        double dot = 0;
        for (int d = 0; d < D; d++) dot += (double)q[d] * k[j * D + d];
        row[j] = (window <= 0 || j > p - window) ? (float)(scale * dot) : -INFINITY;
    }
    if (tl_softmax_f32(row, pr, 1, ctx) != TL_OK) return 0;
    for (int d = 0; d < D; d++) {
        double acc = 0;
        for (int64_t j = 0; j < ctx; j++) acc += (double)pr[j] * v[j * D + d];
        o[d] = (float)acc;
    }
    return 1;
}

OL_TEST(matches_naive_over_the_gathered_cache) {
    /* WHY: the kernel against plain attention over the same f16 values,
     *      gathered in position order: GQA (6 query heads on 2 KV heads),
     *      both layers, lengths 1 (one key), 37 (a partial last block), and
     *      70, with and without a sliding window of 20.
     * KIND: differential
     * CATCHES: s01, s02, s04, s05, s06, s07, s08, m01
     * CHAPTER: L9.4 section 4 */
    world w;
    OL_TRUE(build(&w, 51));
    static float k[HKV * 70 * D], v[HKV * 70 * D], o[3 * H * D], want[D];
    int ok = 1;
    for (uint32_t layer = 0; layer < L && ok; layer++)
        for (int window = 0; window <= 20 && ok; window += 20) {
            ok = tl_paged_attn_decode_f32(w.q, w.p, layer, w.tables, MAXB, w.ctx, o, 3, H, HKV, D, 0.18f, window,
                                          NULL) == TL_OK;
            for (int b = 0; b < 3 && ok; b++) {
                gather(w.p, w.tables + b * MAXB, w.ctx[b], layer, k, v);
                for (int h = 0; h < H && ok; h++) {
                    int kvh = h / (H / HKV);
                    ok = naive_row(w.q + (b * H + h) * D, k + kvh * w.ctx[b] * D, v + kvh * w.ctx[b] * D, w.ctx[b],
                                   window, 0.18f, want);
                    for (int d = 0; d < D && ok; d++)
                        if (!(fabs((double)o[(b * H + h) * D + d] - want[d]) <= 1e-5 + 1e-4 * fabs(want[d]))) {
                            printf("    b %d h %d d %d layer %u window %d: %.9g vs %.9g\n", b, h, d, layer, window,
                                   o[(b * H + h) * D + d], want[d]);
                            ok = 0;
                        }
                }
            }
        }
    tl_kv_pool_destroy(w.p);
    OL_TRUE(ok);
}

OL_TEST(equals_flash_attention_bitwise) {
    /* WHY: a decode step is FlashAttention with one query at position
     *      ctx - 1 (causal) and key tiles of block_tokens: your L9.3 kernel
     *      on the gathered values with Bc = 16 must give the same bits. That
     *      is what lets the engine prefill with L9.3 and decode with L9.4
     *      and get the tokens a single kernel would.
     * KIND: differential, property
     * CATCHES: s09, s10
     * CHAPTER: L9.4 section 2 */
    world w;
    OL_TRUE(build(&w, 52));
    static float k[HKV * 70 * D], v[HKV * 70 * D], o[3 * H * D], f[H * D];
    int ok = tl_paged_attn_decode_f32(w.q, w.p, 1, w.tables, MAXB, w.ctx, o, 3, H, HKV, D, 0.18f, 21, NULL) == TL_OK;
    for (int b = 0; b < 3 && ok; b++) {
        gather(w.p, w.tables + b * MAXB, w.ctx[b], 1, k, v);
        ok = tl_flash_attn_fwd_f32(w.q + b * H * D, k, v, f, NULL, 1, H, HKV, 1, w.ctx[b], D, 0.18f, w.ctx[b] - 1, 1,
                                   21, NULL, 1, BT, NULL, NULL) == TL_OK &&
             ol_bits_diff_f32(f, o + b * H * D, H * D) < 0;
    }
    tl_kv_pool_destroy(w.p);
    OL_TRUE(ok);
}

OL_TEST(shared_blocks_and_batch_invariance) {
    /* WHY: after a fork (L8.3) two sequences hold the same prefix blocks;
     *      each reads them independently. And a sequence's output must have
     *      the same bits alone or in a batch of 16 (L10.2 batches decode
     *      steps): 16 sequences share the 37-token prefix and differ in a
     *      last block of their own.
     * KIND: property
     * CATCHES: s11
     * CHAPTER: L9.4 section 2 */
    enum { S = 16, CTX = 40 };
    tl_kv_pool *p = make_pool(NB, BT, 1, HKV, D);
    OL_TRUE(p != NULL);
    uint32_t shared[2], tails[S], tables[S * 3];
    OL_EQ(tl_kv_alloc(p, 2, shared), TL_OK);
    OL_EQ(tl_kv_alloc(p, S, tails), TL_OK);
    ol_gen g = {53};
    float kv[2][HKV * D];
    static float q[S * H * D], all[S * H * D], one[H * D];
    int32_t ctx[S];
    for (int b = 0; b < S; b++) {
        tables[b * 3] = shared[0], tables[b * 3 + 1] = shared[1], tables[b * 3 + 2] = tails[b];
        ctx[b] = CTX;
    }
    for (int j = 0; j < 32; j++) {
        fill(&g, kv[0], HKV * D), fill(&g, kv[1], HKV * D);
        put(p, tables, j, 0, kv[0], kv[1]);
    }
    for (int b = 0; b < S; b++)
        for (int j = 32; j < CTX; j++) {
            fill(&g, kv[0], HKV * D), fill(&g, kv[1], HKV * D);
            put(p, tables + b * 3, j, 0, kv[0], kv[1]);
        }
    fill(&g, q, S * H * D);
    tl_status st = tl_paged_attn_decode_f32(q, p, 0, tables, 3, ctx, all, S, H, HKV, D, 0.2f, 0, NULL);
    int ok = st == TL_OK;
    for (int b = 0; b < S && ok; b++) {
        ok = tl_paged_attn_decode_f32(q + b * H * D, p, 0, tables + b * 3, 3, ctx + b, one, 1, H, HKV, D, 0.2f, 0,
                                      NULL) == TL_OK &&
             ol_bits_diff_f32(one, all + b * H * D, H * D) < 0;
    }
    tl_kv_pool_destroy(p);
    OL_TRUE(ok);
}

OL_TEST(pool_result_equals_serial_bitwise) {
    /* WHY: (sequence, head) rows run on rt.03 workers; each row's
     *      arithmetic is the same on any worker.
     * KIND: property
     * CATCHES: s12
     * CHAPTER: L9.4 section 2 */
    world w;
    OL_TRUE(build(&w, 54));
    static float o1[3 * H * D], o4[3 * H * D];
    tl_pool *tp = NULL;
    OL_EQ(tl_pool_create(4, &tp), TL_OK);
    tl_status st = tl_paged_attn_decode_f32(w.q, w.p, 0, w.tables, MAXB, w.ctx, o4, 3, H, HKV, D, 0.18f, 0, tp);
    tl_pool_destroy(tp);
    tl_status st1 = tl_paged_attn_decode_f32(w.q, w.p, 0, w.tables, MAXB, w.ctx, o1, 3, H, HKV, D, 0.18f, 0, NULL);
    tl_kv_pool_destroy(w.p);
    OL_EQ(st, TL_OK);
    OL_EQ(st1, TL_OK);
    OL_TRUE(ol_bits_diff_f32(o4, o1, 3 * H * D) < 0);
}

OL_TEST(bad_tables_and_shapes) {
    /* WHY: the engine passes block tables it built itself; a bug there must
     *      come back as TL_EINVAL (an empty context, a table too short for
     *      the context, a block id or a layer outside the pool) or
     *      TL_ESHAPE (heads or width that do not match the pool), with out
     *      untouched, not as a read of another sequence's memory.
     * KIND: boundary
     * CATCHES: s13, s14, s15
     * CHAPTER: L9.4 section 4 */
    tl_kv_pool *p = make_pool(4, 2, 1, 1, 2);
    OL_TRUE(p != NULL);
    const uint32_t table[2] = {0, 1}, bad[2] = {0, 9};
    const float q[2] = {1, 0};
    float o[2] = {5, 5};
    const int32_t zero = 0, three = 3, five = 5;
    tl_status s1 = tl_paged_attn_decode_f32(q, p, 0, table, 2, &zero, o, 1, 1, 1, 2, 1.0f, 0, NULL);
    tl_status s2 = tl_paged_attn_decode_f32(q, p, 0, table, 2, &five, o, 1, 1, 1, 2, 1.0f, 0, NULL);
    tl_status s3 = tl_paged_attn_decode_f32(q, p, 0, bad, 2, &three, o, 1, 1, 1, 2, 1.0f, 0, NULL);
    tl_status s4 = tl_paged_attn_decode_f32(q, p, 1, table, 2, &three, o, 1, 1, 1, 2, 1.0f, 0, NULL);
    tl_status s5 = tl_paged_attn_decode_f32(q, p, 0, table, 2, &three, o, 1, 2, 2, 2, 1.0f, 0, NULL);
    tl_status s6 = tl_paged_attn_decode_f32(q, p, 0, table, 2, &three, o, 1, 1, 1, 4, 1.0f, 0, NULL);
    tl_kv_pool_destroy(p);
    OL_EQ(s1, TL_EINVAL);
    OL_EQ(s2, TL_EINVAL);
    OL_EQ(s3, TL_EINVAL);
    OL_EQ(s4, TL_EINVAL);
    OL_EQ(s5, TL_ESHAPE);
    OL_EQ(s6, TL_ESHAPE);
    OL_EQ(o[0], 5.0f);
    OL_EQ(o[1], 5.0f);
}

int main(void) { return OL_RUN_ALL(); }
