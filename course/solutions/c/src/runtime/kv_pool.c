/* c/src/runtime/kv_pool.c (rt.04): the paged KV block pool, format v1.
 * Contract: tinyllm/kv_pool.h; the hash and the export envelope in
 * formats/kv-block.md. craft.13 later takes this file over for format v2.
 *
 * Memory. One slab of n_blocks * block_bytes holds every block. Inside a
 * block the layout is the export payload order, [n_layers][2][n_kv_heads]
 * [block_tokens][head_dim] of f16 bit patterns (index 1 of the second axis:
 * 0 = K, 1 = V), so tl_kv_block_ptr is plain arithmetic and export copies
 * whole rows.
 *
 * State. Every block is in exactly one of three states:
 *
 *   free    ref == 0, not registered: on the free list
 *   used    ref >= 1 (registered or not)
 *   cached  ref == 0, registered: in the prefix index and on the LRU list
 *
 * The free list is a tl_vec of block ids (ds.01) used as a stack: reserved
 * for n_blocks ids at create, so pushing a freed id never reallocates, and
 * popping reads the last id and shortens len (the struct's fields are
 * public). The prefix index is a tl_map (ds.02) from block hash to block id, created
 * for n_blocks keys so it never grows while the pool lives. The cached
 * blocks are linked through a tl_list_node embedded in each block's record
 * (ds.03), so moving a block to the most recent end or evicting the oldest
 * is O(1). The chained hash uses the FNV-1a 64-bit byte update below.
 *
 * Not thread-safe (kv_pool.h). Every allocation goes through tl_alloc.
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm/abi.h"
#include "tinyllm/ds.h"
#include "tinyllm/kv_pool.h"

typedef struct {
    uint32_t ref;        /* holders; 0 for free and cached blocks */
    uint32_t fill;       /* token positions holding data, 0..block_tokens */
    uint64_t hash;       /* registered hash; 0 when not registered */
    tl_list_node lru;    /* linked into pool->lru while cached */
    uint32_t id;
} kv_block;

struct tl_kv_pool {
    tl_kv_cfg cfg;
    size_t elem;        /* bytes per element (2 for f16) */
    size_t row_bytes;   /* one token position of one head: head_dim * elem */
    size_t slab_bytes;  /* one (layer, K or V): n_kv_heads * block_tokens * row_bytes */
    size_t block_bytes; /* n_layers * 2 * slab_bytes */
    unsigned char *data;
    kv_block *blocks;
    tl_vec free_ids; /* ids of free blocks (uint32_t); the last is taken first */
    uint32_t n_used, n_cached, evictions;
    tl_map *index; /* hash -> id of every registered block */
    tl_lru lru;    /* cached blocks, oldest first */
};

static void put_u16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)(v >> 8);
}

static void put_u32(unsigned char *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i));
}

static void put_u64(unsigned char *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (8 * i));
}

static uint16_t get_u16(const unsigned char *p) {
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t get_u32(const unsigned char *p) {
    uint32_t v = 0;
    for (int i = 3; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static uint64_t get_u64(const unsigned char *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

uint32_t tl_crc32c(const void *data, size_t n, uint32_t crc) {
    const unsigned char *p = (const unsigned char *)data;
    uint32_t c = ~crc; /* a fresh checksum starts from 0xFFFFFFFF */
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u)));
    }
    return ~c;
}

static uint64_t kv_fnv1a64(const void *data, size_t n, uint64_t h) {
    const unsigned char *p = data;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(0x100000001B3);
    }
    return h;
}

uint64_t tl_kv_block_hash(uint64_t parent, const uint32_t *toks, uint32_t n) {
    if (toks == NULL && n > 0) {
        tl_set_last_error("tl_kv_block_hash: toks is NULL");
        return 0;
    }
    unsigned char b[8];
    put_u64(b, parent);
    uint64_t h = kv_fnv1a64(b, 8, UINT64_C(0xCBF29CE484222325));
    for (uint32_t i = 0; i < n; i++) {
        put_u32(b, toks[i]);
        h = kv_fnv1a64(b, 4, h);
    }
    return h == 0 ? 1 : h; /* 0 means "no hash" (the partial tail block) */
}

/* Bytes of one block's payload for a configuration, or 0 on overflow. */
static size_t payload_bytes(uint32_t L, uint32_t H, uint32_t B, uint32_t D) {
    uint64_t v = 2u * 2u; /* K and V, 2 bytes per f16 */
    const uint32_t f[4] = {L, H, B, D};
    for (int i = 0; i < 4; i++) {
        if (f[i] != 0 && v > (uint64_t)SIZE_MAX / f[i]) return 0;
        v *= f[i];
    }
    return (size_t)v;
}

tl_status tl_kv_pool_create(const tl_kv_cfg *cfg, tl_kv_pool **out) {
    if (out == NULL || cfg == NULL) {
        tl_set_last_error("tl_kv_pool_create: NULL argument");
        return TL_EINVAL;
    }
    *out = NULL;
    if (cfg->n_blocks == 0 || cfg->block_tokens == 0 || cfg->n_layers == 0 ||
        cfg->n_kv_heads == 0 || cfg->head_dim == 0) {
        tl_set_last_error("tl_kv_pool_create: every dimension must be at least 1");
        return TL_EINVAL;
    }
    if (cfg->format != TL_KV_FORMAT_V1 || cfg->dtype != TL_F16) {
        tl_set_last_error("tl_kv_pool_create: only format 1 with dtype TL_F16 (format 2 arrives with craft.13)");
        return TL_EUNSUPPORTED;
    }
    size_t bb = payload_bytes(cfg->n_layers, cfg->n_kv_heads, cfg->block_tokens, cfg->head_dim);
    if (bb == 0 || bb > SIZE_MAX / cfg->n_blocks) {
        tl_set_last_error("tl_kv_pool_create: pool size overflows");
        return TL_EINVAL;
    }
    tl_kv_pool *p = tl_alloc(sizeof *p, _Alignof(tl_kv_pool));
    if (p == NULL) {
        tl_set_last_error("tl_kv_pool_create: out of memory");
        return TL_ENOMEM;
    }
    memset(p, 0, sizeof *p);
    p->cfg = *cfg;
    p->elem = 2;
    p->row_bytes = (size_t)cfg->head_dim * p->elem;
    p->slab_bytes = (size_t)cfg->n_kv_heads * cfg->block_tokens * p->row_bytes;
    p->block_bytes = bb;
    p->data = tl_alloc(bb * cfg->n_blocks, 64);
    p->blocks = tl_alloc(sizeof(kv_block) * cfg->n_blocks, _Alignof(kv_block));
    tl_vec_init(&p->free_ids, sizeof(uint32_t));
    tl_status st = TL_ENOMEM;
    if (p->data && p->blocks && tl_vec_reserve(&p->free_ids, cfg->n_blocks) == TL_OK)
        st = tl_map_create(cfg->n_blocks, &p->index);
    if (st != TL_OK) {
        tl_free(p->data);
        tl_free(p->blocks);
        tl_vec_free(&p->free_ids);
        tl_free(p);
        tl_set_last_error("tl_kv_pool_create: out of memory");
        return TL_ENOMEM;
    }
    tl_lru_init(&p->lru);
    for (uint32_t i = 0; i < cfg->n_blocks; i++) {
        p->blocks[i] = (kv_block){0, 0, 0, {NULL, NULL}, i};
        uint32_t id = cfg->n_blocks - 1 - i; /* id 0 last: ids come out in order */
        tl_vec_push(&p->free_ids, &id);      /* within the reserved capacity */
    }
    *out = p;
    return TL_OK;
}

void tl_kv_pool_destroy(tl_kv_pool *p) {
    if (p == NULL) return;
    tl_map_destroy(p->index);
    tl_free(p->data);
    tl_free(p->blocks);
    tl_vec_free(&p->free_ids);
    tl_free(p);
}

/* Takes one block out of the cached state: unlinks it (a no-op when
 * tl_lru_pop_oldest already did) and unregisters it. */
static void uncache(tl_kv_pool *p, kv_block *b) {
    tl_lru_remove(&p->lru, &b->lru);
    tl_map_del(p->index, b->hash);
    b->hash = 0;
    p->n_cached--;
}

tl_status tl_kv_alloc(tl_kv_pool *p, uint32_t n, uint32_t *ids) {
    if (p == NULL || (ids == NULL && n > 0)) {
        tl_set_last_error("tl_kv_alloc: NULL argument");
        return TL_EINVAL;
    }
    if ((uint64_t)p->free_ids.len + p->n_cached < n) { /* all or nothing */
        tl_set_last_error("tl_kv_alloc: not enough free or cached blocks");
        return TL_EFULL;
    }
    for (uint32_t i = 0; i < n; i++) {
        kv_block *b;
        if (p->free_ids.len > 0) { /* pop: read the last id, shorten len */
            b = &p->blocks[((const uint32_t *)p->free_ids.data)[p->free_ids.len - 1]];
            p->free_ids.len--;
        } else { /* reclaim the least recently used cached block */
            tl_list_node *old = tl_lru_pop_oldest(&p->lru);
            b = TL_CONTAINER_OF(old, kv_block, lru);
            uncache(p, b); /* already unlinked; drop it from the index */
            p->evictions++;
        }
        b->ref = 1;
        b->fill = 0;
        p->n_used++;
        ids[i] = b->id;
    }
    return TL_OK;
}

static kv_block *used_block(const tl_kv_pool *p, uint32_t id, const char *fn) {
    if (p == NULL || id >= p->cfg.n_blocks || p->blocks[id].ref == 0) {
        tl_set_last_error(fn);
        return NULL;
    }
    return &p->blocks[id];
}

void tl_kv_ref(tl_kv_pool *p, uint32_t id) {
    kv_block *b = used_block(p, id, "tl_kv_ref: block is not in use (lookup revives a cached block)");
    if (b != NULL) b->ref++;
}

tl_status tl_kv_unref(tl_kv_pool *p, uint32_t id) {
    kv_block *b = used_block(p, id, "tl_kv_unref: id out of range or refcount already 0 (double unref)");
    if (b == NULL) return TL_EINVAL;
    if (--b->ref > 0) return TL_OK;
    p->n_used--;
    if (b->hash != 0) { /* registered: keep the contents for a later lookup */
        tl_lru_touch(&p->lru, &b->lru);
        p->n_cached++;
    } else {
        b->fill = 0;
        tl_vec_push(&p->free_ids, &id); /* never grows: cap is n_blocks */
    }
    return TL_OK;
}

tl_status tl_kv_cow(tl_kv_pool *p, uint32_t id, uint32_t *out) {
    kv_block *b = used_block(p, id, "tl_kv_cow: block is not in use");
    if (b == NULL || out == NULL) {
        if (out == NULL) tl_set_last_error("tl_kv_cow: out is NULL");
        return TL_EINVAL;
    }
    if (b->ref == 1) { /* the only holder may write in place */
        *out = id;
        return TL_OK;
    }
    uint32_t nid;
    tl_status st = tl_kv_alloc(p, 1, &nid);
    if (st != TL_OK) return st;
    memcpy(p->data + (size_t)nid * p->block_bytes, p->data + (size_t)id * p->block_bytes, p->block_bytes);
    p->blocks[nid].fill = b->fill;
    b->ref--; /* still >= 1: the other holders keep the original */
    *out = nid;
    return TL_OK;
}

tl_status tl_kv_set_fill(tl_kv_pool *p, uint32_t id, uint32_t n_tokens) {
    kv_block *b = used_block(p, id, "tl_kv_set_fill: block is not in use");
    if (b == NULL) return TL_EINVAL;
    if (n_tokens > p->cfg.block_tokens || b->hash != 0) {
        tl_set_last_error("tl_kv_set_fill: fill above block_tokens, or a registered (immutable) block");
        return TL_EINVAL;
    }
    b->fill = n_tokens;
    return TL_OK;
}

uint32_t tl_kv_fill(const tl_kv_pool *p, uint32_t id) {
    if (p == NULL || id >= p->cfg.n_blocks) return 0;
    return p->blocks[id].fill;
}

tl_status tl_kv_register(tl_kv_pool *p, uint32_t id, uint64_t hash) {
    kv_block *b = used_block(p, id, "tl_kv_register: block is not in use");
    if (b == NULL) return TL_EINVAL;
    if (hash == 0 || b->fill != p->cfg.block_tokens) {
        tl_set_last_error("tl_kv_register: only a full block with a non-zero hash is registered");
        return TL_EINVAL;
    }
    if (b->hash == hash) return TL_OK; /* already registered under this hash */
    if (b->hash != 0) {
        tl_set_last_error("tl_kv_register: block is registered under another hash");
        return TL_EINVAL;
    }
    if (tl_map_get(p->index, hash, NULL)) {
        tl_set_last_error("tl_kv_register: another block holds this hash");
        return TL_EBUSY;
    }
    tl_status st = tl_map_put(p->index, hash, id);
    if (st != TL_OK) return st;
    b->hash = hash;
    return TL_OK;
}

tl_status tl_kv_lookup(tl_kv_pool *p, uint64_t hash, uint32_t *id) {
    if (p == NULL || id == NULL) {
        tl_set_last_error("tl_kv_lookup: NULL argument");
        return TL_EINVAL;
    }
    uint64_t v;
    if (hash == 0 || !tl_map_get(p->index, hash, &v)) return TL_ENOTFOUND; /* a miss is not an error */
    kv_block *b = &p->blocks[v];
    if (b->ref == 0) { /* cached: back to used */
        tl_lru_remove(&p->lru, &b->lru);
        p->n_cached--;
        p->n_used++;
    }
    b->ref++;
    *id = (uint32_t)v;
    return TL_OK;
}

void *tl_kv_block_ptr(const tl_kv_pool *p, uint32_t id, uint32_t layer, int is_v) {
    if (p == NULL || id >= p->cfg.n_blocks || layer >= p->cfg.n_layers) {
        tl_set_last_error("tl_kv_block_ptr: id or layer out of range");
        return NULL;
    }
    size_t off = (size_t)id * p->block_bytes + ((size_t)layer * 2 + (is_v ? 1 : 0)) * p->slab_bytes;
    return p->data + off;
}

void tl_kv_pool_cfg(const tl_kv_pool *p, tl_kv_cfg *out) {
    if (p != NULL && out != NULL) *out = p->cfg;
}

size_t tl_kv_block_bytes(const tl_kv_pool *p) {
    return p == NULL ? 0 : p->block_bytes;
}

size_t tl_kv_export_bytes(const tl_kv_pool *p, uint32_t n) {
    if (p == NULL) return 0;
    return TL_KV_ENVELOPE_HEADER + (size_t)n * (12 + p->block_bytes) + 4;
}

tl_status tl_kv_export(const tl_kv_pool *p, const uint32_t *ids, uint32_t n,
                       void *buf, size_t cap, size_t *written) {
    if (p == NULL || written == NULL || (ids == NULL && n > 0)) {
        tl_set_last_error("tl_kv_export: NULL argument");
        return TL_EINVAL;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (used_block(p, ids[i], "tl_kv_export: block is not in use") == NULL) return TL_EINVAL;
        if (p->blocks[ids[i]].fill == 0) {
            tl_set_last_error("tl_kv_export: an empty block (fill 0) cannot be exported");
            return TL_EINVAL;
        }
    }
    size_t need = tl_kv_export_bytes(p, n);
    *written = need;
    if (buf == NULL || cap < need) {
        tl_set_last_error("tl_kv_export: buffer too small (*written is the size needed)");
        return TL_EFULL;
    }
    unsigned char *o = (unsigned char *)buf;
    memcpy(o, TL_KV_MAGIC, 4);
    put_u16(o + 4, (uint16_t)TL_KV_FORMAT_V1);
    put_u16(o + 6, (uint16_t)p->cfg.dtype);
    put_u32(o + 8, n);
    put_u32(o + 12, p->cfg.block_tokens);
    put_u32(o + 16, p->cfg.n_layers);
    put_u32(o + 20, p->cfg.n_kv_heads);
    put_u32(o + 24, p->cfg.head_dim);
    size_t at = TL_KV_ENVELOPE_HEADER;
    const uint32_t B = p->cfg.block_tokens, D = p->cfg.head_dim;
    for (uint32_t i = 0; i < n; i++) {
        const kv_block *b = &p->blocks[ids[i]];
        put_u64(o + at, b->hash);
        put_u32(o + at + 8, b->fill);
        at += 12;
        const unsigned char *src = p->data + (size_t)ids[i] * p->block_bytes;
        size_t rows = (size_t)p->cfg.n_layers * 2 * p->cfg.n_kv_heads; /* (layer, K/V, head) */
        for (size_t r = 0; r < rows; r++) {
            for (uint32_t t = 0; t < B; t++) {
                const unsigned char *s = src + (r * B + t) * p->row_bytes;
                for (uint32_t d = 0; d < D; d++) {
                    uint16_t v;
                    memcpy(&v, s + (size_t)d * 2, 2); /* in memory: host-order f16 bits */
                    put_u16(o + at, t < b->fill ? v : 0); /* past the fill: zero bytes */
                    at += 2;
                }
            }
        }
    }
    put_u32(o + at, tl_crc32c(o, at, 0));
    return TL_OK;
}

tl_status tl_kv_import(tl_kv_pool *p, const void *buf, size_t len, uint32_t *ids_out) {
    if (p == NULL || (buf == NULL && len > 0)) {
        tl_set_last_error("tl_kv_import: NULL argument");
        return TL_EINVAL;
    }
    const unsigned char *in = (const unsigned char *)buf;
    if (len < TL_KV_ENVELOPE_HEADER + 4 || memcmp(in, TL_KV_MAGIC, 4) != 0) {
        tl_set_last_error("tl_kv_import: too short, or bad magic");
        return TL_EFORMAT;
    }
    if (get_u16(in + 4) != TL_KV_FORMAT_V1 || get_u16(in + 6) != TL_F16) {
        tl_set_last_error("tl_kv_import: this pool reads version 1 with dtype TL_F16 only");
        return TL_EFORMAT;
    }
    uint32_t n = get_u32(in + 8), B = get_u32(in + 12), L = get_u32(in + 16);
    uint32_t H = get_u32(in + 20), D = get_u32(in + 24);
    size_t P = payload_bytes(L, H, B, D);
    uint64_t rec = 12 + (uint64_t)P;
    if (P == 0 || (n != 0 && rec > (UINT64_MAX - 32) / n) ||
        (uint64_t)len != 32 + (uint64_t)n * rec) {
        tl_set_last_error("tl_kv_import: length does not match the header");
        return TL_EFORMAT;
    }
    if (get_u32(in + len - 4) != tl_crc32c(in, len - 4, 0)) {
        tl_set_last_error("tl_kv_import: CRC-32C mismatch");
        return TL_EFORMAT;
    }
    if (B != p->cfg.block_tokens || L != p->cfg.n_layers || H != p->cfg.n_kv_heads || D != p->cfg.head_dim) {
        tl_set_last_error("tl_kv_import: block dimensions differ from the pool's");
        return TL_ESHAPE;
    }
    for (uint32_t i = 0; i < n; i++) { /* reader rule 5, before anything is allocated */
        const unsigned char *r = in + TL_KV_ENVELOPE_HEADER + (size_t)i * rec;
        uint64_t h = get_u64(r);
        uint32_t fill = get_u32(r + 8);
        if (fill < 1 || fill > B || (h != 0 && fill != B)) {
            tl_set_last_error("tl_kv_import: a block's n_tokens is out of range, or a hashed block is partial");
            return TL_EFORMAT;
        }
    }
    if (n > 0 && ids_out == NULL) {
        tl_set_last_error("tl_kv_import: ids_out is NULL");
        return TL_EINVAL;
    }
    tl_status st = tl_kv_alloc(p, n, ids_out);
    if (st != TL_OK) return st;
    for (uint32_t i = 0; i < n; i++) {
        const unsigned char *r = in + TL_KV_ENVELOPE_HEADER + (size_t)i * rec;
        uint64_t h = get_u64(r);
        uint32_t fill = get_u32(r + 8);
        const unsigned char *src = r + 12;
        unsigned char *dst = p->data + (size_t)ids_out[i] * p->block_bytes;
        for (size_t e = 0; e < P / 2; e++) {
            uint16_t v = get_u16(src + 2 * e);
            memcpy(dst + 2 * e, &v, 2);
        }
        p->blocks[ids_out[i]].fill = fill;
        if (h != 0 && !tl_map_get(p->index, h, NULL)) st = tl_kv_register(p, ids_out[i], h);
        if (st != TL_OK) { /* the index could not grow: give every block back */
            for (uint32_t j = 0; j < n; j++) tl_kv_unref(p, ids_out[j]);
            return st;
        }
    }
    return TL_OK;
}

void tl_kv_stats_get(const tl_kv_pool *p, tl_kv_stats *out) {
    if (p == NULL || out == NULL) return;
    out->free = (uint32_t)p->free_ids.len;
    out->used = p->n_used;
    out->cached = p->n_cached;
    out->evictions = p->evictions;
}
