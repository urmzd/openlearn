# Paged attention for decode in C

[A paged KV cache in Python](../p08-inference/03-paged-kv-cache.md) defined how a paged cache behaves: each sequence owns a block table, prefix blocks are shared after a fork, and a block is copied only when someone writes into a shared one. This section implements the attention side of that design in C for one decode step: a batch of sequences, each with one new query, whose keys and values live in blocks of the [paged KV block pool](../p08-inference/08-paged-kv-block-pool.md), reached through block tables, stored as f16, with GQA and a sliding window, and with no allocation at all.

The pleasant surprise is how little new arithmetic there is. Decode is [FlashAttention](06-flash-attention-forward.md) with one query at the newest position, and the cache's blocks are already tiles aligned to absolute positions, which is exactly the alignment FlashAttention uses for invariance. So the paged kernel can be made to produce the same bits as FlashAttention on the gathered cache: you can prefill with one kernel and decode with the other, and the tokens agree. The work of this section is in the indirection (finding position $j$ through the table) and in validating every table before trusting it.

This kernel is optional and stands alone: its correctness is defined against naive attention and the C prefill kernel, so it can be validated without binding C into Python or Rust, and the Rust engine implements decode independently with its own block manager. It uses the [thread pool](03-thread-pool-and-parallel-for.md), the block pool's [`kv_pool.h`](../../../../course/contracts/c/include/tinyllm/kv_pool.h), `tl_f16_to_f32` from [Low-precision conversions in standalone C](../../mathematics-for-computing/09-numerical-methods-and-floating-point/07-low-precision-c-conversions.md) and `tl_expf` from [Polynomial approximation and range reduction](../../mathematics-for-computing/09-numerical-methods-and-floating-point/06-polynomial-approximation-and-expf.md), both in *Mathematics for Computing*, and `tl_softmax_f32` from [Softmax in C](05-softmax-three-pass-and-online.md) as the naive reference's softmax. The specification is `windowed_attention` from [Sliding window and sinks](../p07-modern-block/07-sliding-window-and-sinks.md), and [MQA/GQA attention](../p07-modern-block/05-gqa-attention.md) is useful background. The contract is [`attention.h`](../../../../course/contracts/c/include/tinyllm/attention.h) under [`c/ABI.md`](../../../../course/contracts/c/ABI.md), and the reference solution is [`paged_attn.c`](../../../../course/solutions/c/src/kernels/paged_attn.c): `tl_paged_attn_decode_f32` with the helpers `decode_row` and `rows`. Try to predict each step before you read it.

> **Learning objectives**
> - Locate the key and value of any position through a block table and the pool's slab layout.
> - Apply FlashAttention's tile update to blocks for a single decode query, keeping all state in locals and the output row.
> - Explain why this kernel equals FlashAttention with $B_c$ equal to the block size, bit for bit.
> - Explain why shared prefix blocks and batching do not change a sequence's output.
> - Validate shapes, layers, context lengths, and every block id before writing anything.

## Where position $j$ lives

The section uses this notation:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $b$ | `block_tokens`: positions per block (16 in the engine) | `uint32_t` |
| $\mathit{table}_s$ | sequence $s$'s block ids in position order, row $s$ of `block_tables` | `uint32_t[max_blocks]` |
| $n_s$ | `ctx_lens[s]`: positions in the cache, the newest included | `int32_t` |
| $p_s = n_s - 1$ | the query's position (the newest token) | |
| $q_{s,h} \in \mathbb{R}^D$ | the query of sequence $s$, head $h$ | `float[D]` |
| $k_j, v_j$ | key and value at position $j$, stored as f16 | `uint16_t[D]` |
| $w$ | the window ($w \le 0$: none) | `int64_t` |
| $m, \ell, a$ | running maximum, denominator, output (as in FlashAttention) | `float`, `float`, `float[D]` |

`kv_pool.h` lays out one block as, for each layer, a K slab and a V slab, each `[n_kv_heads][block_tokens][head_dim]` elements of the pool's dtype (f16 in format 1). So the key of KV head $g$ at position $j$ of sequence $s$ is at

$$\texttt{tl\_kv\_block\_ptr}(\mathit{kv}, \mathit{table}_s[\lfloor j/b \rfloor], \mathit{layer}, 0) + \big(g \cdot b + (j \bmod b)\big) \cdot D ,$$

and the value is at the same offset with `is_v = 1`. The table maps logical blocks (positions $0..b-1$, $b..2b-1$, ...) to physical blocks anywhere in the pool. That indirection is the whole point of paging: memory is allocated a block at a time, no sequence needs a contiguous region, and two sequences can name the same physical block.

## One decode step

For each sequence $s$ and head $h$ (KV head $g = \lfloor h / (H/H_{kv}) \rfloor$), the query is the newest token, so it is causal by construction and sees positions

$$V = \{\, j : \max(0, p_s - w + 1) \le j \le p_s \,\} \quad (\text{all of } 0..p_s \text{ when } w \le 0).$$

The kernel walks the logical blocks that intersect $V$ in increasing order and applies FlashAttention's update to each block's visible positions. It computes scores $\sigma\, q \cdot k_j$ (the f16 key decoded by `tl_f16_to_f32`), takes the block's maximum, and updates $m' = \max(m, \cdot)$, $\alpha = e^{m - m'}$, $\ell \leftarrow \alpha \ell + \sum e^{s_j - m'}$, $a \leftarrow \alpha a + \sum e^{s_j - m'} v_j$. At the end $o = a \cdot (1/\ell)$. The query always sees at least itself, so $\ell > 0$.

## No scratch

The contract gives this kernel no arena: a decode step is small and frequent, and every byte of state fits somewhere already. The running output $a$ lives in the caller's `out` row, which is overwritten anyway; $m$ and $\ell$ are two locals; and the scores of a block go into a fixed stack buffer of 256 floats (a block larger than that is scored in aligned pieces of 256). Nothing depends on $n_s$.

## Equal to FlashAttention, bit for bit

FlashAttention with one query row at `q_offset` $= p_s$, causal, the same window, and $B_c = b$ visits key tiles $[tb, (t+1)b)$ in increasing $t$ and performs, for each, exactly the operations above on the same float32 values (the decoded f16 values). The two kernels therefore produce the same bits. Anything that changes the grouping of the sums keeps the answer close but breaks the bits: a piece size that depends on the batch, a different shift, or dividing by $\ell$ instead of multiplying by $1/\ell$. Breaking the bits breaks the engine's guarantee that prefill and decode agree.

## Batch invariance and threads

Each $(s, h)$ row reads only its own query, its own table, and the pool, and writes only its own output. A sequence's result is therefore the same alone or in a batch of 16, and the same whichever pool worker computes it. Shared prefix blocks need no special handling: after a fork two tables name the same blocks, and each row reads them independently.

## Validate the tables first

The engine builds the tables, and a bug there would make the kernel read another sequence's memory or past the end of the pool. So before the first write the kernel checks the shapes against the pool (`TL_ESHAPE` when $H_{kv}$ or $D$ differ, or $H \bmod H_{kv} \ne 0$), the layer, every context length (at least 1), that each table holds $\lceil n_s / b \rceil \le$ `max_blocks` entries, and that every block id it will read is below `n_blocks` (`TL_EINVAL` for the last four). Pools in format 2 (fp8, see [Interface migration: KV format v1 to v2](../../software-craftsmanship/07-maintenance/01-kv-format-v1-to-v2.md) in *Software Craftsmanship*) return `TL_EUNSUPPORTED` until that migration.

## Working it by hand

Put FlashAttention's worked example into a paged cache: $D = 2$, $b = 2$, one sequence of $n = 3$ positions in a pool of 4 blocks, with table $[3, 1]$. The positions land like this:

| Position $j$ | block $\lfloor j/2 \rfloor$ → id | slot | $k_j$ | $v_j$ |
|---|---|---|---|---|
| 0 | 0 → 3 | 0 | $[1, 0]$ | $[1, 2]$ |
| 1 | 0 → 3 | 1 | $[0, 1]$ | $[3, 4]$ |
| 2 | 1 → 1 | 0 | $[1, 1]$ | $[5, 6]$ |

All values are exact in f16. The query $q = [1, 0]$ is at position $p = 2$ and sees positions 0, 1, and 2.

Block id 3 holds positions 0 and 1, with scores 1 and 0. Then $m' = 1$, $\alpha = e^{-\infty} = 0$, $\ell = 1 + e^{-1} = 1.3678794$, and $a = [1, 2] + 0.3678794\,[3, 4] = [2.1036383, 3.4715177]$.

Block id 1 holds position 2; its slot 1 is empty and not visible. The score is 1, so $m' = 1$, $\alpha = 1$, $\ell = 2.3678794$, and $a = [7.1036383, 9.4715177]$.

Finally $o = a / \ell = [3, 4]$, the same answer as the contiguous example. A correct kernel computes this exact result through the C interface.

## The interface

```c
/* tinyllm/attention.h */
tl_status tl_paged_attn_decode_f32(const float *q, const tl_kv_pool *kv, uint32_t layer,
                                   const uint32_t *block_tables, int32_t max_blocks,
                                   const int32_t *ctx_lens, float *out,
                                   int64_t B, int64_t H, int64_t Hkv, int64_t D,
                                   float scale, int64_t window, tl_pool *tp);
/* q, out [B, H, D]. Sequence b's blocks: block_tables[b * max_blocks + i].
   TL_EINVAL: NULL pointers, a block id or layer out of range, ctx_lens[b] < 1,
   a sequence needing more than max_blocks blocks. TL_ESHAPE: Hkv or D differ
   from the pool, or H % Hkv != 0. */
```

From Python, `PagedKVCache.pool` is the `tl_kv_pool *` and `block_table(seq)` is the row of ids.

## Building it step by step

Decode has one query per (sequence, head), at position $p = n - 1$. It is causal by construction, since every cached position is at most $p$, so the only mask left is the window, which gives $V = [\max(0, p - w + 1), p]$. Position $j$ is found through the table, never by arithmetic on a base pointer: logical block $\lfloor j / b \rfloor$ maps to a physical id, and within that block the KV head's slab offset is $g \cdot b \cdot D$, then the slot $(j \bmod b) \cdot D$. In the worked example, position 2 is logical block 1, physical id 1, slot 0.

The arithmetic is FlashAttention's tile update with tiles equal to blocks. Blocks already cover the absolute positions $[tb, (t+1)b)$, which is exactly the tile alignment FlashAttention uses for invariance, so using the same update, in the same order, with the same final `* (1/l)`, makes the two kernels produce the same bits. No allocation is needed: one row of state is $m$, $\ell$, and $a[D]$, and $a$ can live in the output row the caller will receive anyway. Scores of one piece go into a fixed stack buffer, and a piece size that never depends on the batch keeps a sequence's bits the same alone or batched. Finally, every index the kernel will follow must be checked before it follows any. The tables come from another component, and checking all of them up front means a bad table returns an error with `out` untouched instead of reading another sequence's memory.

### One row

`decode_row` computes one (sequence, head). The setup derives the position, the window's lower bound, and the table row; the running output is the caller's `out` row:

```c
static void decode_row(const pa_args *a, int64_t b, int64_t h) {
    const int64_t D = a->D, bt = a->bt;
    const int64_t kvh = h / (a->H / a->Hkv);          /* GQA: group of H/Hkv heads */
    const int64_t p = a->ctx_lens[b] - 1;             /* the query is the newest token */
    const int64_t lo = a->window > 0 && p - a->window + 1 > 0 ? p - a->window + 1 : 0;
    const float *qr = a->q + (b * a->H + h) * D;
    float *acc = a->out + (b * a->H + h) * D;         /* running output lives in out: no scratch */
    const uint32_t *table = a->tables + b * (int64_t)a->max_blocks;
    float s[TILE];                                    /* fixed: 256 scores, independent of n */
    float m = -INFINITY, l = 0.0f;
    for (int64_t d = 0; d < D; d++) acc[d] = 0.0f;
    // ...
```

### The block walk

The walk visits only the logical blocks that intersect $[lo, p]$, in increasing order, and resolves each through the table. Inside a block, pieces of `a->tile` positions start at the block start (aligned), and each piece is clipped to the visible range:

```c
    for (int64_t blk = lo / bt; blk <= p / bt; blk++) {
        const uint16_t *kb = tl_kv_block_ptr(a->kv, table[blk], a->layer, 0); /* physical id */
        const uint16_t *vb = tl_kv_block_ptr(a->kv, table[blk], a->layer, 1); /* the V slab */
        kb += kvh * bt * D, vb += kvh * bt * D; /* slab [n_kv_heads][block_tokens][head_dim] */
        for (int64_t j0 = blk * bt; j0 < (blk + 1) * bt && j0 <= p; j0 += a->tile) {
            const int64_t j1 = j0 + a->tile < (blk + 1) * bt ? j0 + a->tile : (blk + 1) * bt;
            const int64_t ja = lo > j0 ? lo : j0, jb = p < j1 - 1 ? p : j1 - 1; /* visible part */
            if (ja > jb) continue;
            float mt = -INFINITY;               /* max over VISIBLE scores only */
            for (int64_t j = ja; j <= jb; j++) {
                const uint16_t *kj = kb + (j - blk * bt) * D;   /* slot j mod b */
                float dot = 0.0f;
                for (int64_t d = 0; d < D; d++) dot += qr[d] * tl_f16_to_f32(kj[d]);
                s[j - ja] = a->scale * dot;
                if (s[j - ja] > mt) mt = s[j - ja];
            }
            const float mn = mt > m ? mt : m;
            const float alpha = tl_expf(m - mn);
            float lt = 0.0f;
            for (int64_t d = 0; d < D; d++) acc[d] *= alpha;   /* rescale output ... */
            for (int64_t j = ja; j <= jb; j++) {
                const uint16_t *vj = vb + (j - blk * bt) * D;
                float w = tl_expf(s[j - ja] - mn);
                lt += w;
                for (int64_t d = 0; d < D; d++) acc[d] += w * tl_f16_to_f32(vj[d]);
            }
            l = l * alpha + lt;                                 /* ... and denominator */
            m = mn;
        }
    }
    const float inv = 1.0f / l;           /* l > 0: the query sees at least itself */
    for (int64_t d = 0; d < D; d++) acc[d] *= inv;   /* multiply by 1/l, exactly as L9.3 does */
}
```

The statement order matches `flash_attn.c` line for line: compute scores, take the piece maximum, rescale `acc`, add weighted values, update `l`, then multiply by `1/l` at the end. That correspondence is the proof of bitwise equality.

### Validation and dispatch

The entry point validates first. Note the loop over every sequence's table: it computes how many blocks the sequence needs and checks each id against the pool before any row is computed.

```c
    tl_kv_cfg cfg;
    tl_kv_pool_cfg(kv, &cfg);
    if (Hkv == 0 || Hkv != (int64_t)cfg.n_kv_heads || D != (int64_t)cfg.head_dim || H % Hkv != 0) {
        tl_set_last_error("tl_paged_attn_decode_f32: Hkv and D must match the pool, and H % Hkv == 0");
        return TL_ESHAPE;
    }
    if (cfg.dtype != TL_F16) { /* ... */ return TL_EUNSUPPORTED; }   /* fp8 pools: not yet */
    if (layer >= cfg.n_layers) { /* ... */ return TL_EINVAL; }
    const int64_t bt = cfg.block_tokens;
    for (int64_t b = 0; b < B; b++) {
        if (ctx_lens[b] < 1) { /* ... */ return TL_EINVAL; }   /* else 1/l = inf */
        int64_t need = (ctx_lens[b] + bt - 1) / bt;             /* ceil(n / b) blocks */
        if (need > max_blocks) { /* ... */ return TL_EINVAL; }  /* never trust max_blocks */
        for (int64_t i = 0; i < need; i++)
            if (block_tables[b * (int64_t)max_blocks + i] >= cfg.n_blocks) { /* ... */ return TL_EINVAL; }
    }
    pa_args a = {q, kv, layer, block_tables, max_blocks, ctx_lens, out, H, Hkv, D, window, bt, TILE, scale};
    if (tp == NULL) {
        rows(&a, 0, B * H, 0);
        return TL_OK;
    }
    return tl_parallel_for(tp, B * H, 1, rows, &a);   /* one item per (b, h) */
```

The full solution is [`paged_attn.c`](../../../../course/solutions/c/src/kernels/paged_attn.c).

### Why this design

Each (sequence, head) reads its $\min(n, w)$ visible keys and values once: $O(\min(n, w) D)$ time, $O(D)$ state, and zero heap allocation. Decode is memory-bound, so reading each f16 element once is the cost that matters.

Why not gather the blocks into a contiguous buffer and call FlashAttention? It would be correct and bitwise equal, but it copies the whole context every step and needs scratch proportional to $n$, which is what paging is meant to avoid. Why not split a long context across workers, as Flash-Decoding does? That gives parallelism at batch size 1, but the merge order would depend on the split and the bits would no longer match prefill; with many sequences and heads per step, (sequence, head) already gives enough parallel work. Why validate every table up front rather than check each id as it is read? Checking lazily would leave `out` partially written when an error is found; up-front validation keeps "error means nothing was written" true, at the cost of one pass over the table indices. The tradeoff accepted is that blocks larger than 256 positions are scored in pieces of 256, so equality with FlashAttention holds when $b \le 256$ (the engine uses 16).

## What can go wrong

A correct kernel matches both naive attention and FlashAttention, and fails loudly on a bad table:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | the worked example through table $[3, 1]$ | the indirection and the slot arithmetic |
| naive attention over the gathered cache | differential | GQA 6:2, $D = 32$, two layers, lengths 1, 37, 70 in shuffled blocks, window 0 and 20 | every flag against plain attention |
| equal to FlashAttention | differential, property | FlashAttention on the gathered values with $B_c = 16$, bitwise | prefill and decode agree |
| shared blocks and batch invariance | property | 16 sequences sharing two prefix blocks; each alone vs in the batch, bitwise | forked prefix blocks and sequence isolation |
| pooled equals serial | property | 4 threads vs serial, bitwise and race-free | threads never change bits |
| bad tables and shapes | boundary | empty context, short table, id 9 in a 4-block pool, layer out of range (`TL_EINVAL`); head or width mismatch (`TL_ESHAPE`); `out` untouched | a table bug fails loudly |
| the hand example written through `PagedKVCache` | unit | the worked example with the blocks laid out by the Python cache | the cache and the kernel share one layout |

The rescaling bugs are the same as in FlashAttention. Not rescaling the running output overweights blocks before a larger score; do `acc[d] *= alpha` before each piece. Not rescaling the denominator means outputs no longer average the values; do `l = l * alpha + lt`.

The indirection adds new ways to read the wrong memory. Using the logical block index as the block id reads whatever block happens to have that id; call `tl_kv_block_ptr(kv, table[blk], ...)`. Forgetting the KV head's offset inside the slab makes every head read KV head 0; add `kb += kvh * bt * D`. Mapping heads round-robin ($h \bmod H_{kv}$) makes GQA models attend with the wrong keys; use `kvh = h / (H / Hkv)`. Reading V from the K slab produces averages of keys; take `vb` from `tl_kv_block_ptr(..., 1)`. Placing the query at $n$ instead of $n - 1$ reads an unwritten slot; set `p = ctx_lens[b] - 1`. A window of $w + 1$ positions drifts from the model's training; use `lo = p - w + 1`.

> **Warning:** Several bugs give answers that are close but not FlashAttention's bits, which means prefill and decode disagree on near-ties. Shifting by a maximum that includes positions outside the window (take `mt` over `[ja, jb]` only), dividing by $\ell$ instead of multiplying by $1/\ell$ (compute `inv = 1.0f / l`, then `acc[d] *= inv`), and using a smaller piece size for a lone sequence (keep `tile = TILE` always) all fall in this class.

A pooled partition that drops a row is wrong only with threads; use one flat index over `B * H`, decoded in `rows`. And the validation rules each prevent a crash or a NaN. Accepting an empty context gives $1/\ell = \infty$ and NaN, so `ctx_lens[b] < 1` is `TL_EINVAL`. Trusting `max_blocks` reads past the table, so `need > max_blocks` is `TL_EINVAL`. Trusting the block ids makes `tl_kv_block_ptr` return NULL and the kernel crash, so every id is checked below `n_blocks` first.

## Where this leads

This kernel pulls together most of the chapter: the error slot and allocator support from [the runtime support layer](02-arena-allocator-with-marks.md), `tl_parallel_for` over (sequence, head) from [the thread pool](03-thread-pool-and-parallel-for.md), `tl_kv_pool_cfg` and `tl_kv_block_ptr` from the [paged KV block pool](../p08-inference/08-paged-kv-block-pool.md), `tl_f16_to_f32` and `tl_expf` from *Mathematics for Computing*, the online update from [Softmax in C](05-softmax-three-pass-and-online.md) (whose `tl_softmax_f32` is the naive reference's softmax), [FlashAttention](06-flash-attention-forward.md) as the bitwise reference, `PagedKVCache` from [A paged KV cache in Python](../p08-inference/03-paged-kv-cache.md) for the logical layout, and `windowed_attention` from [Sliding window and sinks](../p07-modern-block/07-sliding-window-and-sinks.md) as the specification.

Nothing later in the book calls it: the production Rust engine uses its own implementation, and this optional C kernel has no production caller. Together with FlashAttention, the [fused quantized matmul](08-fused-quantized-matmul.md), and the [elementwise kernels](09-elementwise-kernels.md), it completes the C kernel set.

Production systems extend paged attention in several directions:

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one query per (sequence, head) | vLLM PagedAttention v1 and v2 | one thread block per (sequence, head); v2 splits long contexts across blocks and merges partial (m, l, a) | vLLM `csrc/attention/` |
| a serial walk over a long table | Flash-Decoding | split the context across workers for one query and merge with the same rescale: parallel at batch size 1, at the price of a fixed split order | Dao et al. (2023) |
| f16 KV | fp8 KV caches | half the bytes per token; scales per head or per block (the KV format v1 to v2 migration) | vLLM `kv_cache_dtype="fp8"` |
| block tables from the engine | SGLang RadixAttention | the tables come from a radix tree over token ids, so shared prefixes are found automatically | SGLang `radix_cache.py` |

### Further reading

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023).
- Dao et al., "Flash-Decoding" (2023).

## Key terms

- **paged KV cache**: a cache stored as fixed-size blocks anywhere in a pool, with a per-sequence table mapping logical blocks to physical ones.
- **block table**: a sequence's list of physical block ids in position order.
- **slab**: the `[n_kv_heads][block_tokens][head_dim]` region of a block holding K or V for one layer.
- **decode step**: one forward pass with a single new query per sequence, at the newest position.
- **shared prefix block**: a physical block named by several sequences' tables after a fork.

## Summary

- A sequence's cache is a list of blocks, not one buffer: position $j$ lives in slot $j \bmod b$ of block $\mathit{table}[\lfloor j / b \rfloor]$, so the kernel reads keys in position order through the table, whatever the block ids.
- Decode is FlashAttention with one query at the newest position: the same online softmax over the blocks, with the running output kept in the caller's `out` row, so the kernel allocates nothing.
- Blocks are tiles aligned to absolute positions, so a paged decode gives the same bits as FlashAttention on the gathered cache with $B_c$ equal to the block size: prefill with one kernel and decode with the other, and the tokens agree.
- Shared prefix blocks just work: after a fork two tables name the same blocks, each read independently, and a sequence's output has the same bits alone or in a batch of 16.
- The whole table is validated before the first write: an empty context, a table too short, a block id outside the pool, or a layer out of range is `TL_EINVAL`.

## Review questions

1. Why is decode causal without a causal flag?
   <details><summary>Answer</summary>The query is the newest token at $p = n - 1$, and the cache holds only positions $0..n-1$. No key is after the query, so causal masking would remove nothing.</details>
2. Why does the kernel equal FlashAttention with $B_c = b$, bit for bit?
   <details><summary>Answer</summary>Blocks cover absolute positions $[tb, (t+1)b)$, the same tiles FlashAttention uses, and within each tile both kernels clip to the visible range, take the visible maximum, rescale, add in increasing $j$, and finish with `* (1/l)`. Same float32 inputs (f16 decoded identically), same operations, same order.</details>
3. What would happen if the piece size shrank for a batch of one?
   <details><summary>Answer</summary>The grouping of scores into pieces would change, so the running maximum changes at different points and the rescales round differently. The sequence's output would differ in the last bits between running alone and in a batch.</details>
4. Why are the running outputs written straight into `out`?
   <details><summary>Answer</summary>The contract gives the kernel no arena, and `out` is overwritten anyway, so it is free storage for $a[D]$. $m$ and $\ell$ are locals and the scores use a fixed stack buffer, so nothing depends on $n$.</details>
5. With $b = 16$, a context of $n = 70$ positions, and a window $w = 20$, which logical blocks does the walk visit?
   <details><summary>Answer</summary>$p = 69$ and $lo = 69 - 20 + 1 = 50$, so the walk covers logical blocks $\lfloor 50/16 \rfloor = 3$ through $\lfloor 69/16 \rfloor = 4$. Block 3 contributes positions 50 to 63 and block 4 positions 64 to 69. Validation still checks all $\lceil 70/16 \rceil = 5$ table entries.</details>
