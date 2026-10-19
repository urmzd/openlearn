/* c/src/kernels/paged_attn.c (L9.4): attention for one decode step over a
 * paged KV cache. Contract: tinyllm/attention.h. Blocks from rt.04's pool,
 * f16 decoded by M09.4's tl_f16_to_f32, exponentials from M09.6's tl_expf,
 * threads from rt.03.
 *
 * Sequence b's tokens live in the blocks of its block table, not in one
 * contiguous buffer: position j is slot j % block_tokens of block
 * table[j / block_tokens]. The kernel walks the table in position order and
 * runs L9.3's online softmax over each block (in pieces of at most TILE
 * positions, aligned to absolute positions), with the running output kept
 * in the caller's `out` row. Nothing is allocated: the contract gives this
 * kernel no arena, and a decode step has one query row per (b, h).
 *
 * With block_tokens <= TILE, every (b, h) row performs exactly the
 * operations L9.3 performs with Bc = block_tokens on the same values, so a
 * paged decode equals FlashAttention over the gathered cache, bit for bit.
 */
#include <math.h>
#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/attention.h"
#include "tinyllm/kv_pool.h"
#include "tinyllm/numerics.h"
#include "tinyllm/pool.h"

#pragma STDC FP_CONTRACT OFF

#define TILE 256 /* scores held at once: a block is processed in aligned pieces of at most this */

typedef struct {
    const float *q;
    const tl_kv_pool *kv;
    uint32_t layer;
    const uint32_t *tables;
    int32_t max_blocks;
    const int32_t *ctx_lens;
    float *out;
    int64_t H, Hkv, D, window, bt;
    int64_t tile; /* positions scored at once: TILE, whatever the batch */
    float scale;
} pa_args;

/* Row (b, h): the online softmax over positions [lo, p] of the block table. */
static void decode_row(const pa_args *a, int64_t b, int64_t h) {
    const int64_t D = a->D, bt = a->bt;
    const int64_t kvh = h / (a->H / a->Hkv);
    const int64_t p = a->ctx_lens[b] - 1; /* the query is the newest token */
    const int64_t lo = a->window > 0 && p - a->window + 1 > 0 ? p - a->window + 1 : 0;
    const float *qr = a->q + (b * a->H + h) * D;
    float *acc = a->out + (b * a->H + h) * D;
    const uint32_t *table = a->tables + b * (int64_t)a->max_blocks;
    float s[TILE];
    float m = -INFINITY, l = 0.0f;
    for (int64_t d = 0; d < D; d++) acc[d] = 0.0f;

    for (int64_t blk = lo / bt; blk <= p / bt; blk++) {
        const uint16_t *kb = tl_kv_block_ptr(a->kv, table[blk], a->layer, 0);
        const uint16_t *vb = tl_kv_block_ptr(a->kv, table[blk], a->layer, 1);
        kb += kvh * bt * D, vb += kvh * bt * D; /* slab [n_kv_heads][block_tokens][head_dim] */
        for (int64_t j0 = blk * bt; j0 < (blk + 1) * bt && j0 <= p; j0 += a->tile) {
            const int64_t j1 = j0 + a->tile < (blk + 1) * bt ? j0 + a->tile : (blk + 1) * bt;
            const int64_t ja = lo > j0 ? lo : j0, jb = p < j1 - 1 ? p : j1 - 1;
            if (ja > jb) continue;
            float mt = -INFINITY;
            for (int64_t j = ja; j <= jb; j++) {
                const uint16_t *kj = kb + (j - blk * bt) * D;
                float dot = 0.0f;
                for (int64_t d = 0; d < D; d++) dot += qr[d] * tl_f16_to_f32(kj[d]);
                s[j - ja] = a->scale * dot;
                if (s[j - ja] > mt) mt = s[j - ja];
            }
            const float mn = mt > m ? mt : m;
            const float alpha = tl_expf(m - mn);
            float lt = 0.0f;
            for (int64_t d = 0; d < D; d++) acc[d] *= alpha;
            for (int64_t j = ja; j <= jb; j++) {
                const uint16_t *vj = vb + (j - blk * bt) * D;
                float w = tl_expf(s[j - ja] - mn);
                lt += w;
                for (int64_t d = 0; d < D; d++) acc[d] += w * tl_f16_to_f32(vj[d]);
            }
            l = l * alpha + lt;
            m = mn;
        }
    }
    /* l > 0 always here: the query sees at least itself. */
    const float inv = 1.0f / l;
    for (int64_t d = 0; d < D; d++) acc[d] *= inv;
}

static void rows(void *ctx, int64_t lo, int64_t hi, int worker) {
    (void)worker;
    const pa_args *a = ctx;
    for (int64_t it = lo; it < hi; it++) decode_row(a, it / a->H, it % a->H);
}

tl_status tl_paged_attn_decode_f32(const float *q, const tl_kv_pool *kv, uint32_t layer,
                                   const uint32_t *block_tables, int32_t max_blocks,
                                   const int32_t *ctx_lens, float *out,
                                   int64_t B, int64_t H, int64_t Hkv, int64_t D,
                                   float scale, int64_t window, tl_pool *tp) {
    /* 1. Validate everything, every sequence's table included, before the
     *    first write to out. */
    if (B < 0 || H < 0 || Hkv < 0 || D < 0 || max_blocks < 0) {
        tl_set_last_error("tl_paged_attn_decode_f32: negative dimension");
        return TL_EINVAL;
    }
    if (B == 0 || H == 0) return TL_OK;
    if (q == NULL || kv == NULL || block_tables == NULL || ctx_lens == NULL || out == NULL) {
        tl_set_last_error("tl_paged_attn_decode_f32: NULL pointer");
        return TL_EINVAL;
    }
    tl_kv_cfg cfg;
    tl_kv_pool_cfg(kv, &cfg);
    if (Hkv == 0 || Hkv != (int64_t)cfg.n_kv_heads || D != (int64_t)cfg.head_dim || H % Hkv != 0) {
        tl_set_last_error("tl_paged_attn_decode_f32: Hkv and D must match the pool, and H % Hkv == 0");
        return TL_ESHAPE;
    }
    if (cfg.dtype != TL_F16) {
        tl_set_last_error("tl_paged_attn_decode_f32: only KV format 1 (f16) is supported");
        return TL_EUNSUPPORTED;
    }
    if (layer >= cfg.n_layers) {
        tl_set_last_error("tl_paged_attn_decode_f32: layer out of range");
        return TL_EINVAL;
    }
    const int64_t bt = cfg.block_tokens;
    for (int64_t b = 0; b < B; b++) {
        if (ctx_lens[b] < 1) {
            tl_set_last_error("tl_paged_attn_decode_f32: ctx_lens[b] < 1");
            return TL_EINVAL;
        }
        int64_t need = (ctx_lens[b] + bt - 1) / bt;
        if (need > max_blocks) {
            tl_set_last_error("tl_paged_attn_decode_f32: a sequence needs more than max_blocks blocks");
            return TL_EINVAL;
        }
        for (int64_t i = 0; i < need; i++)
            if (block_tables[b * (int64_t)max_blocks + i] >= cfg.n_blocks) {
                tl_set_last_error("tl_paged_attn_decode_f32: block id out of range");
                return TL_EINVAL;
            }
    }

    /* 2. One work item per (b, h). */
    pa_args a = {q, kv, layer, block_tables, max_blocks, ctx_lens, out, H, Hkv, D, window, bt, TILE, scale};
    if (tp == NULL) {
        rows(&a, 0, B * H, 0);
        return TL_OK;
    }
    return tl_parallel_for(tp, B * H, 1, rows, &a);
}
