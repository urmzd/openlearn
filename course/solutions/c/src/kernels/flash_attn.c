/* c/src/kernels/flash_attn.c (L9.3): FlashAttention forward, in float32 on
 * the CPU. Contract: tinyllm/attention.h. Scratch from rt.02's arena,
 * threads from rt.03, exponentials from M09.6's tl_expf.
 *
 * Attention for one query row is softmax(scale * q K^T) V over the keys the
 * row may see. The naive kernel writes the whole score row (Tk floats per
 * query) and reads it back. FlashAttention never stores it: keys are read
 * in tiles of Bc, and each row keeps only its running maximum m, its
 * running denominator l, and its running output acc[D] (L9.2's online
 * update, with acc rescaled by the same factor as l). Scratch is therefore
 * Br rows of state plus one tile of scores per worker, whatever Tk is.
 *
 * Invariance (c/ABI.md rule 10): key tile t always covers the absolute key
 * positions [t * Bc, (t + 1) * Bc). A row visits, in increasing t, exactly
 * the tiles that hold a key it may see, and inside a tile adds its keys in
 * increasing j. That sequence depends on the row alone: not on the batch,
 * not on Br, not on how the queries were split into calls with q_offset.
 */
#include <math.h>
#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/arena.h"
#include "tinyllm/attention.h"
#include "tinyllm/numerics.h"
#include "tinyllm/pool.h"

#pragma STDC FP_CONTRACT OFF

typedef struct {
    const float *q, *k, *v, *sinks;
    float *o, *lse;
    int64_t B, H, Hkv, Tq, Tk, D, q_offset, window, Br, Bc, n_qtiles;
    float scale;
    int causal;
    float *scratch;   /* n_workers slices of slice_floats */
    int64_t slice_floats;
} fa_args;

/* The keys row p may see: [*lo, *hi] (empty when *lo > *hi). */
static void visible_range(const fa_args *a, int64_t p, int64_t *lo, int64_t *hi) {
    *hi = a->Tk - 1;
    if (a->causal && p < *hi) *hi = p;
    *lo = 0;
    if (a->window > 0 && p - a->window + 1 > *lo) *lo = p - a->window + 1;
}

/* One work item: query tile `qt` of head h of sequence b. */
static void query_tile(const fa_args *a, int64_t b, int64_t h, int64_t qt, float *state, float *s) {
    const int64_t D = a->D, Bc = a->Bc;
    const int64_t kvh = h / (a->H / a->Hkv); /* GQA: H / Hkv query heads per KV head */
    const int64_t i0 = qt * a->Br;
    const int64_t p0 = a->q_offset + i0; /* absolute position of the tile's first query */
    const int64_t nr = a->Tq - i0 < a->Br ? a->Tq - i0 : a->Br;
    const float *qh = a->q + ((b * a->H + h) * a->Tq) * D;
    /* No pointer arithmetic on k and v when there are no keys: they may be
     * NULL then, and NULL + 0 is undefined behavior. */
    const float *kh = a->Tk ? a->k + ((b * a->Hkv + kvh) * a->Tk) * D : NULL;
    const float *vh = a->Tk ? a->v + ((b * a->Hkv + kvh) * a->Tk) * D : NULL;
    float *m = state, *l = state + a->Br, *acc = state + 2 * a->Br; /* acc: [Br][D] */

    /* 1. Initial state per row. A sink is one extra score that joins the
     *    denominator and adds no value: start from m = sink, l = e^0 = 1. */
    int64_t lo_min = INT64_MAX, hi_max = -1; /* the keys any row of the tile sees */
    for (int64_t r = 0; r < nr; r++) {
        m[r] = a->sinks ? a->sinks[h] : -INFINITY;
        l[r] = a->sinks ? 1.0f : 0.0f;
        for (int64_t d = 0; d < D; d++) acc[r * D + d] = 0.0f;
        int64_t lo, hi;
        visible_range(a, p0 + r, &lo, &hi);
        if (lo <= hi) {
            if (lo < lo_min) lo_min = lo;
            if (hi > hi_max) hi_max = hi;
        }
    }

    /* 2. Key tiles aligned to absolute positions (j0 a multiple of Bc), in
     *    increasing order. */
    for (int64_t j0 = hi_max < 0 ? 0 : lo_min / Bc * Bc; j0 <= hi_max; j0 += Bc) {
        const int64_t j1 = j0 + Bc < a->Tk ? j0 + Bc : a->Tk;
        for (int64_t r = 0; r < nr; r++) {
            int64_t lo, hi;
            visible_range(a, p0 + r, &lo, &hi);
            const int64_t ja = lo > j0 ? lo : j0, jb = hi < j1 - 1 ? hi : j1 - 1;
            if (ja > jb) continue; /* nothing visible here: this row's state is untouched */
            const float *qr = qh + (i0 + r) * D;
            float mt = -INFINITY;
            for (int64_t j = ja; j <= jb; j++) {
                float dot = 0.0f;
                for (int64_t d = 0; d < D; d++) dot += qr[d] * kh[j * D + d];
                s[j - ja] = a->scale * dot;
                if (s[j - ja] > mt) mt = s[j - ja];
            }
            /* The online update of L9.2, on a whole tile: rescale what was
             * accumulated against the old maximum, then add the tile. */
            const float mn = mt > m[r] ? mt : m[r];
            const float alpha = tl_expf(m[r] - mn); /* m[r] = -inf gives 0: nothing to rescale */
            float *ar = acc + r * D;
            float lt = 0.0f;
            for (int64_t d = 0; d < D; d++) ar[d] *= alpha;
            for (int64_t j = ja; j <= jb; j++) {
                float p = tl_expf(s[j - ja] - mn);
                lt += p;
                for (int64_t d = 0; d < D; d++) ar[d] += p * vh[j * D + d];
            }
            l[r] = l[r] * alpha + lt;
            m[r] = mn;
        }
    }

    /* 3. Normalize and write. No visible key and no sink: zeros and -inf. */
    for (int64_t r = 0; r < nr; r++) {
        float *orow = a->o + ((b * a->H + h) * a->Tq + i0 + r) * D;
        float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        for (int64_t d = 0; d < D; d++) orow[d] = acc[r * D + d] * inv;
        if (a->lse) a->lse[(b * a->H + h) * a->Tq + i0 + r] = l[r] > 0.0f ? m[r] + logf(l[r]) : -INFINITY;
    }
}

/* rt.03 range function over work items (b, h, query tile). */
static void items(void *ctx, int64_t lo, int64_t hi, int worker) {
    const fa_args *a = ctx;
    float *slice = a->scratch + (int64_t)worker * a->slice_floats;
    for (int64_t it = lo; it < hi; it++) {
        int64_t qt = it % a->n_qtiles, bh = it / a->n_qtiles;
        query_tile(a, bh / a->H, bh % a->H, qt, slice, slice + a->Br * (a->D + 2));
    }
}

tl_status tl_flash_attn_fwd_f32(const float *q, const float *k, const float *v, float *o, float *lse,
                                int64_t B, int64_t H, int64_t Hkv, int64_t Tq, int64_t Tk, int64_t D,
                                float scale, int64_t q_offset, int causal, int64_t window,
                                const float *sink_logits, int64_t Br, int64_t Bc,
                                tl_arena *scratch, tl_pool *tp) {
    /* 1. Validate before writing anything. */
    if (B < 0 || H < 0 || Hkv < 0 || Tq < 0 || Tk < 0 || D < 0 || q_offset < 0 || Br < 0 || Bc < 0) {
        tl_set_last_error("tl_flash_attn_fwd_f32: negative dimension, q_offset, or tile size");
        return TL_EINVAL;
    }
    if (B == 0 || H == 0 || Tq == 0) return TL_OK; /* no output row */
    if (Hkv == 0 || Hkv > H || H % Hkv != 0) {
        tl_set_last_error("tl_flash_attn_fwd_f32: need H % Hkv == 0 and 0 < Hkv <= H");
        return TL_ESHAPE;
    }
    if (q == NULL || o == NULL || (Tk > 0 && (k == NULL || v == NULL))) {
        tl_set_last_error("tl_flash_attn_fwd_f32: NULL tensor");
        return TL_EINVAL;
    }
    if (Br == 0) Br = 64;
    if (Bc == 0) Bc = 64;

    /* 2. Scratch: per worker, Br rows of (m, l, acc[D]) and one tile of
     *    scores. Nothing here depends on Tk. */
    tl_arena *own = NULL;
    if (scratch == NULL) {
        tl_status st = tl_arena_create(0, &own);
        if (st != TL_OK) return st;
        scratch = own;
    }
    tl_arena_mark mark = tl_arena_mark_get(scratch);
    const int workers = tl_pool_threads(tp);
    const int64_t slice = Br * (D + 2) + Bc;
    float *buf = tl_arena_alloc(scratch, sizeof(float) * (size_t)(slice * workers), 64);
    if (buf == NULL) {
        tl_arena_reset_to(scratch, mark);
        tl_arena_destroy(own);
        tl_set_last_error("tl_flash_attn_fwd_f32: scratch arena could not grow");
        return TL_ENOMEM;
    }

    /* 3. One work item per (b, h, query tile). */
    fa_args a = {q, k, v, sink_logits, o, lse, B, H, Hkv, Tq, Tk, D, q_offset, window, Br, Bc,
                 (Tq + Br - 1) / Br, scale, causal, buf, slice};
    const int64_t n_items = B * H * a.n_qtiles;
    tl_status st = TL_OK;
    if (tp == NULL)
        items(&a, 0, n_items, 0);
    else
        st = tl_parallel_for(tp, n_items, 1, items, &a);

    tl_arena_reset_to(scratch, mark); /* the caller's arena is exactly as we found it */
    tl_arena_destroy(own);
    return st;
}
