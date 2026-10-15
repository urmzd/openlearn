/* tinyllm/kv_pool.h (rt.04): the paged KV block pool, KV format v1.
 * chapter: content/courses/build-an-llm/p08-inference/08-paged-kv-block-pool.md
 * Rules in c/ABI.md; the hash and the export envelope in formats/kv-block.md.
 *
 * The pool owns n_blocks fixed-size blocks. A block holds K and V for
 * block_tokens token positions of every layer and KV head. Each block is in
 * exactly one state, and free + used + cached == n_blocks always holds:
 *
 *   free    refcount 0, not registered: available to tl_kv_alloc
 *   used    refcount >= 1
 *   cached  refcount 0 and registered in the prefix index: reusable by
 *           tl_kv_lookup, evicted least recently used first when tl_kv_alloc
 *           runs out of free blocks
 *
 * The prefix index maps a block hash to a block id (the ds.02 Swiss table)
 * and the cached blocks form an LRU list (ds.03). Only full blocks are
 * registered: a block's fill (tl_kv_set_fill) must equal block_tokens.
 *
 * Not thread-safe: the caller serializes every call on one pool (the Rust
 * engine holds the pool behind one Mutex shared by the step loop and the KV
 * transfer tasks). KV format v2 (fp8 e4m3 with scales) is
 * tinyllm/kv_pool_v2.h, which the craft.13 migration adds to tinyllm.h.
 *
 * module: rt.04 (c/src/runtime/kv_pool.c; craft.13 upgrades it) */
#ifndef TINYLLM_KV_POOL_H
#define TINYLLM_KV_POOL_H

#include <stddef.h>
#include <stdint.h>

#include "tinyllm/abi.h"


#define TL_KV_FORMAT_V1 1u
#define TL_KV_MAGIC "TLKV"        /* the first 4 bytes of an export envelope */
#define TL_KV_ENVELOPE_HEADER 28u /* bytes before the first block record */
/* The chained block hash is FNV-1a 64 (formats/kv-block.md, spec/pcg32.md). */
#define TL_FNV1A64_OFFSET 0xCBF29CE484222325ull
#define TL_FNV1A64_PRIME 0x00000100000001B3ull

typedef struct tl_kv_pool tl_kv_pool;

typedef struct {
    uint32_t n_blocks;     /* >= 1 */
    uint32_t block_tokens; /* token positions per block, >= 1 (the engine uses 16) */
    uint32_t n_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    int32_t dtype;   /* tl_dtype: TL_F16 in format 1 (TL_F8_E4M3 in format 2) */
    uint32_t format; /* TL_KV_FORMAT_V1; any other value is TL_EUNSUPPORTED until craft.13 */
} tl_kv_cfg;         /* 28 bytes, no padding */

typedef struct {
    uint32_t free, used, cached;
    uint32_t evictions; /* cached blocks reclaimed by tl_kv_alloc since create */
} tl_kv_stats;

/* TL_EINVAL for NULL arguments or a zero dimension; TL_EUNSUPPORTED for a
 * format or dtype this build does not implement; TL_ENOMEM when the hook
 * fails. Every block starts free with fill 0. */
tl_status tl_kv_pool_create(const tl_kv_cfg *cfg, tl_kv_pool **out);
void tl_kv_pool_destroy(tl_kv_pool *p); /* NULL does nothing */

/* Takes n blocks, each with refcount 1 and fill 0, and writes their ids.
 * When fewer than n blocks are free, evicts cached blocks, least recently
 * used first (each eviction unregisters the block and counts in
 * evictions). All or nothing: TL_EFULL, with nothing taken, when free +
 * cached < n. */
tl_status tl_kv_alloc(tl_kv_pool *p, uint32_t n, uint32_t *ids);

/* refcount++. id must be used (refcount >= 1); otherwise sets the error
 * slot and does nothing (use tl_kv_lookup to revive a cached block). */
void tl_kv_ref(tl_kv_pool *p, uint32_t id);

/* refcount--. At 0 the block becomes cached if registered, else free.
 * TL_EINVAL, and nothing changes, for an id out of range or a block whose
 * refcount is already 0 (double unref). */
tl_status tl_kv_unref(tl_kv_pool *p, uint32_t id);

/* Copy on write. refcount == 1: *out = id. refcount > 1: allocates a new
 * block (TL_EFULL as tl_kv_alloc), copies K, V, and the fill, drops one
 * reference from id, and writes the new id. The copy is not registered.
 * Registered (full) blocks are immutable by convention: the engine never
 * writes into them. */
tl_status tl_kv_cow(tl_kv_pool *p, uint32_t id, uint32_t *out);

/* Records how many token positions of the block hold data, 0..block_tokens.
 * The writer calls it after appending; export writes it as n_tokens.
 * TL_EINVAL for a fill above block_tokens or a registered block. */
tl_status tl_kv_set_fill(tl_kv_pool *p, uint32_t id, uint32_t n_tokens);
uint32_t tl_kv_fill(const tl_kv_pool *p, uint32_t id);

/* Chained FNV-1a 64 (formats/kv-block.md): the FNV-1a 64 of parent as 8
 * little-endian bytes followed by each token id as 4 little-endian bytes;
 * a result of 0 is replaced by 1, because 0 marks the unhashed tail block.
 * The parent of a sequence's first block is 0. Callers hash full blocks
 * only (n == block_tokens). Pure: no pool needed. */
uint64_t tl_kv_block_hash(uint64_t parent, const uint32_t *toks, uint32_t n);

/* Registers a used, full block under hash (hash != 0). TL_EINVAL for a
 * partial block, hash 0, or a block that is not used; TL_EBUSY, and the
 * block stays unregistered, when another block already holds this hash
 * (the caller keeps its own copy or switches to the cached one). Registering
 * a block again under the same hash is TL_OK. */
tl_status tl_kv_register(tl_kv_pool *p, uint32_t id, uint64_t hash);

/* Hit: writes the id, takes a reference (a cached block becomes used), and
 * touches its LRU position; TL_OK. Miss: TL_ENOTFOUND, error slot not set
 * (a miss is not an error). */
tl_status tl_kv_lookup(tl_kv_pool *p, uint64_t hash, uint32_t *id);

/* The K (is_v == 0) or V (is_v != 0) slab of one layer of one block:
 * n_kv_heads * block_tokens * head_dim elements of cfg.dtype, row-major
 * [n_kv_heads][block_tokens][head_dim]. Valid until the pool is destroyed.
 * NULL, with the error slot set, for an id or layer out of range. The pool
 * pointer is const so read-only kernels (L9.4) can use it; writing through
 * the result is allowed only for a block the caller holds a reference to. */
void *tl_kv_block_ptr(const tl_kv_pool *p, uint32_t id, uint32_t layer, int is_v);

/* The configuration the pool was created with. */
void tl_kv_pool_cfg(const tl_kv_pool *p, tl_kv_cfg *out);

/* Payload bytes of one block: n_layers * 2 * n_kv_heads * block_tokens *
 * head_dim * sizeof(dtype). */
size_t tl_kv_block_bytes(const tl_kv_pool *p);

/* Envelope bytes for n blocks:
 * TL_KV_ENVELOPE_HEADER + n * (12 + tl_kv_block_bytes(p)) + 4. */
size_t tl_kv_export_bytes(const tl_kv_pool *p, uint32_t n);

/* Writes the export envelope of formats/kv-block.md for the blocks ids[0..n)
 * in that order: each block's registered hash (0 if unregistered), its fill
 * as n_tokens, and its payload with every position at or past the fill
 * written as zero bytes; then the CRC-32C. *written is the envelope size.
 * TL_EFULL, with *written set to the size needed and buf untouched, when cap
 * is too small; TL_EINVAL for an id that is not used. */
tl_status tl_kv_export(const tl_kv_pool *p, const uint32_t *ids, uint32_t n,
                       void *buf, size_t cap, size_t *written);

/* Reads an envelope: allocates n_blocks blocks (all or nothing, TL_EFULL as
 * tl_kv_alloc), copies payloads and fills, and registers each block whose
 * hash is non-zero and whose fill is full, unless that hash is already
 * registered. ids_out has room for n_blocks, the little-endian u32 at byte
 * offset 8 of the envelope. TL_EFORMAT, with nothing allocated, for a bad
 * magic, a version this pool cannot read (a format 1 pool reads version 1
 * only; kv_pool_v2.h), a dtype that does not match the version, a length
 * that does not match the header, or a CRC mismatch;
 * TL_ESHAPE when block_tokens, n_layers, n_kv_heads, or head_dim differ
 * from the pool's. */
tl_status tl_kv_import(tl_kv_pool *p, const void *buf, size_t len, uint32_t *ids_out);

/* Invariant: free + used + cached == n_blocks. */
void tl_kv_stats_get(const tl_kv_pool *p, tl_kv_stats *out);

/* CRC-32C (Castagnoli, reflected polynomial 0x82F63B78), the envelope and
 * KvChunk checksum. crc is the running value: pass 0 for a fresh checksum,
 * or the previous result to continue over more bytes.
 * tl_crc32c("123456789", 9, 0) == 0xE3069283. */
uint32_t tl_crc32c(const void *data, size_t n, uint32_t crc);


#endif /* TINYLLM_KV_POOL_H */
