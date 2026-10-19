/* tinyllm/attention.h (L9.3, L9.4): FlashAttention forward for prefill and
 * paged attention for decode. Rules in c/ABI.md.
 *
 * Shared definitions. H query heads share Hkv key/value heads (GQA; MHA is
 * Hkv == H, MQA is Hkv == 1); H % Hkv == 0 and query head h reads KV head
 * h / (H / Hkv). Scores are scale * dot(q, k) in f32. A query at absolute
 * position p may see key position j when
 *   causal == 0 or j <= p,  and  window <= 0 or j > p - window.
 * A sink logit, when given, is one extra score per head that joins the
 * softmax denominator and contributes no value (attention sinks); a query
 * with no visible key and no sink outputs zeros and lse = -inf.
 *
 * Both kernels are batch- and chunk-invariant (c/ABI.md rule 10): each
 * output row's reduction over keys runs in tiles aligned to absolute key
 * positions (tile t covers keys [t * Bc, (t + 1) * Bc)), with one fixed
 * order, whatever the batch size, the row's position in the batch, or how
 * the queries were split into calls with q_offset.
 *
 * modules: L9.3 (c/src/kernels/flash_attn.c), L9.4 (c/src/kernels/paged_attn.c)
 * chapter: content/courses/build-an-llm/10-kernels/05-flash-attention-forward.md
 * chapter: content/courses/build-an-llm/10-kernels/06-paged-attention-decode.md
 */
#ifndef TINYLLM_ATTENTION_H
#define TINYLLM_ATTENTION_H

#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/arena.h"
#include "tinyllm/kv_pool.h"


typedef struct tl_pool tl_pool;

/* FlashAttention forward (L9.3). Row-major, contiguous:
 *   q, o  [B, H,   Tq, D]      query i is at absolute position q_offset + i
 *   k, v  [B, Hkv, Tk, D]      key j is at absolute position j
 *   lse   [B, H,   Tq]         log-sum-exp of each row's scores (sink included);
 *                              may be NULL
 *   sink_logits [H]            may be NULL (no sinks)
 * Br and Bc are the query and key tile sizes (0 picks 64). window <= 0
 * means no window. scratch holds the tiles: the kernel rewinds it to the
 * mark it found on entry before returning, and its high-water use does not
 * depend on Tk; NULL makes the kernel use a private arena. tp may be NULL.
 * TL_EINVAL for negative dims, NULL tensors, or q_offset < 0; TL_ESHAPE when
 * H % Hkv != 0 or Hkv > H; TL_ENOMEM when scratch cannot grow. */
tl_status tl_flash_attn_fwd_f32(const float *q, const float *k, const float *v, float *o, float *lse,
                                int64_t B, int64_t H, int64_t Hkv, int64_t Tq, int64_t Tk, int64_t D,
                                float scale, int64_t q_offset, int causal, int64_t window,
                                const float *sink_logits, int64_t Br, int64_t Bc,
                                tl_arena *scratch, tl_pool *tp);

/* Paged attention for one decode step (L9.4). Sequence b has ctx_lens[b]
 * tokens of KV in the pool (its newest token's K and V already written);
 * its query is at position ctx_lens[b] - 1 and is causal by construction.
 * Its blocks are block_tables[b * max_blocks + i] for
 * i < ceil(ctx_lens[b] / block_tokens); token position j lives in block
 * j / block_tokens at slot j % block_tokens. K and V are read from layer
 * `layer` of the pool and converted from the pool's dtype to f32 (f16 in
 * format 1; e4m3 times the scale in format 2).
 *   q, out [B, H, D]   row-major, contiguous; D == the pool's head_dim
 * window <= 0 means no window. tp may be NULL. TL_EINVAL for NULL pointers,
 * a block id or layer out of range, ctx_lens[b] < 1, or a sequence needing
 * more than max_blocks blocks; TL_ESHAPE when Hkv or D differ from the
 * pool's or H % Hkv != 0. */
tl_status tl_paged_attn_decode_f32(const float *q, const tl_kv_pool *kv, uint32_t layer,
                                   const uint32_t *block_tables, int32_t max_blocks,
                                   const int32_t *ctx_lens, float *out,
                                   int64_t B, int64_t H, int64_t Hkv, int64_t D,
                                   float scale, int64_t window, tl_pool *tp);


#endif /* TINYLLM_ATTENTION_H */
