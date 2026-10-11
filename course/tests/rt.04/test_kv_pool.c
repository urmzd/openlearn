/* Course tests for rt.04: the paged KV block pool of tinyllm/kv_pool.h
 * (c/src/runtime/kv_pool.c), built with ASan and UBSan and the counting
 * allocator, so a block read past its slab, a use after destroy, or a leak
 * stops the run.
 *
 * Values are f16 bit patterns (the pool stores TL_F16): 0x3C00 is 1.0,
 * 0x4000 2.0, 0x4200 3.0, 0x4400 4.0, 0x3800 0.5, 0xBC00 -1.0, 0x3400 0.25.
 * The byte-exact cases are the worked examples of formats/kv-block.md.
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

static tl_kv_cfg cfg_of(uint32_t n, uint32_t B, uint32_t L, uint32_t H, uint32_t D) {
    tl_kv_cfg c = {n, B, L, H, D, TL_F16, TL_KV_FORMAT_V1};
    return c;
}

static int stats_are(const tl_kv_pool *p, uint32_t f, uint32_t u, uint32_t c) {
    tl_kv_stats s;
    tl_kv_stats_get(p, &s);
    return s.free == f && s.used == u && s.cached == c;
}

static uint32_t evictions(const tl_kv_pool *p) {
    tl_kv_stats s;
    tl_kv_stats_get(p, &s);
    return s.evictions;
}

/* Fill every (layer, K/V, head, position < fill, d) of block id with the
 * f16 pattern tag ^ index, and set its fill. */
static void write_block(tl_kv_pool *p, uint32_t id, uint32_t fill, uint16_t tag) {
    tl_kv_cfg c;
    tl_kv_pool_cfg(p, &c);
    for (uint32_t l = 0; l < c.n_layers; l++)
        for (int v = 0; v < 2; v++) {
            uint16_t *s = tl_kv_block_ptr(p, id, l, v);
            for (uint32_t h = 0; h < c.n_kv_heads; h++)
                for (uint32_t t = 0; t < fill; t++)
                    for (uint32_t d = 0; d < c.head_dim; d++) {
                        size_t i = ((size_t)h * c.block_tokens + t) * c.head_dim + d;
                        s[i] = (uint16_t)(tag ^ (uint16_t)(i + 131u * l + 977u * (unsigned)v));
                    }
        }
    tl_kv_set_fill(p, id, fill);
}

static int block_has(const tl_kv_pool *p, uint32_t id, uint32_t fill, uint16_t tag) {
    tl_kv_cfg c;
    tl_kv_pool_cfg(p, &c);
    if (tl_kv_fill(p, id) != fill) return 0;
    for (uint32_t l = 0; l < c.n_layers; l++)
        for (int v = 0; v < 2; v++) {
            const uint16_t *s = tl_kv_block_ptr(p, id, l, v);
            for (uint32_t h = 0; h < c.n_kv_heads; h++)
                for (uint32_t t = 0; t < fill; t++)
                    for (uint32_t d = 0; d < c.head_dim; d++) {
                        size_t i = ((size_t)h * c.block_tokens + t) * c.head_dim + d;
                        if (s[i] != (uint16_t)(tag ^ (uint16_t)(i + 131u * l + 977u * (unsigned)v))) return 0;
                    }
        }
    return 1;
}

/* The formats/kv-block.md worked example: B = 2, L = 1, Hkv = 1, D = 2,
 * one full block holding tokens [1, 2]. */
static const uint8_t WORKED[60] = {
    0x54, 0x4c, 0x4b, 0x56, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x66, 0x93, 0x7b, 0x02, 0xc1, 0xb0, 0x1a, 0xa9, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x3c, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44,
    0x00, 0x38, 0x00, 0xbc, 0x00, 0x00, 0x00, 0x34,
    0x1c, 0x54, 0x0b, 0x4f};

static uint32_t worked_block(tl_kv_pool *p) {
    uint32_t id = 99;
    if (tl_kv_alloc(p, 1, &id) != TL_OK) return 99;
    uint16_t *k = tl_kv_block_ptr(p, id, 0, 0), *v = tl_kv_block_ptr(p, id, 0, 1);
    const uint16_t K[4] = {0x3C00, 0x4000, 0x4200, 0x4400}, V[4] = {0x3800, 0xBC00, 0x0000, 0x3400};
    memcpy(k, K, sizeof K);
    memcpy(v, V, sizeof V);
    tl_kv_set_fill(p, id, 2);
    const uint32_t toks[2] = {1, 2};
    tl_kv_register(p, id, tl_kv_block_hash(0, toks, 2));
    return id;
}

OL_TEST(hand_example_block_hash_chain) {
    /* WHY: formats/kv-block.md's worked example: block [1, 2] hashes the 8
     *      zero bytes of its parent and then 01 00 00 00 02 00 00 00 with
     *      FNV-1a 64; block [3, 4] chains that hash in as its parent. A
     *      Rust engine (L10.4) and a decode replica (L10.6) find each
     *      other's blocks only if every implementation gets these values.
     * KIND: unit, golden, smoke
     * CATCHES: s01, s02, m01
     * CHAPTER: rt.04 section 3 */
    const uint32_t b0[2] = {1, 2}, b1[2] = {3, 4};
    uint64_t h0 = tl_kv_block_hash(0, b0, 2);
    OL_EQ(h0, 0xA91AB0C1027B9366ull);
    OL_EQ(tl_kv_block_hash(h0, b1, 2), 0x03DF829571605C60ull);
    OL_TRUE(tl_kv_block_hash(0, b1, 2) != tl_kv_block_hash(h0, b1, 2)); /* same tokens, other prefix */
    const uint32_t swapped[2] = {2, 1};
    OL_TRUE(tl_kv_block_hash(0, swapped, 2) != h0);
}

OL_TEST(hand_example_export_envelope) {
    /* WHY: the 60-byte envelope of formats/kv-block.md, byte for byte: the
     *      28-byte header, the block record (hash, n_tokens 2), the K and V
     *      f16 payload in [layer][K/V][head][token][dim] order, and the
     *      CRC-32C 0x4F0B541C. This is the PushKv payload of L10.6 and the
     *      golden blob of parity/kv.wire.v1.
     * KIND: unit, golden, smoke
     * CATCHES: s03, s04, s05, s06, s19, m02
     * CHAPTER: rt.04 section 3 */
    tl_kv_cfg c = cfg_of(2, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t id = worked_block(p);
    OL_TRUE(id < 2);
    OL_EQ(tl_kv_block_bytes(p), 16);
    OL_EQ(tl_kv_export_bytes(p, 1), 60);
    uint8_t buf[64];
    memset(buf, 0xEE, sizeof buf);
    size_t n = 0;
    OL_EQ(tl_kv_export(p, &id, 1, buf, sizeof buf, &n), TL_OK);
    OL_EQ(n, 60);
    for (int i = 0; i < 60; i++) OL_EQ(buf[i], WORKED[i]);
    OL_EQ(buf[60], 0xEE); /* nothing written past the envelope */
    tl_kv_pool_destroy(p);
}

OL_TEST(crc32c_check_value_and_continuation) {
    /* WHY: CRC-32C (Castagnoli, reflected 0x82F63B78) has the published
     *      check value 0xE3069283 for "123456789", and passing a previous
     *      result continues it: the gRPC KvChunk checksum and the envelope
     *      trailer are both this function.
     * KIND: unit
     * CATCHES: s06
     * CHAPTER: rt.04 section 2 */
    OL_EQ(tl_crc32c("123456789", 9, 0), 0xE3069283u);
    OL_EQ(tl_crc32c("", 0, 0), 0u);
    OL_EQ(tl_crc32c("6789", 4, tl_crc32c("12345", 5, 0)), 0xE3069283u);
}

OL_TEST(alloc_is_all_or_nothing) {
    /* WHY: the scheduler (L10.2) admits a request only if all its blocks
     *      exist. Asking for more blocks than free + cached must fail with
     *      TL_EFULL and take nothing; a partial grab would leak blocks the
     *      caller never learns about.
     * KIND: unit, boundary
     * CATCHES: s07
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(4, 4, 2, 2, 4);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    OL_TRUE(stats_are(p, 4, 0, 0));
    uint32_t ids[5];
    OL_EQ(tl_kv_alloc(p, 3, ids), TL_OK);
    OL_TRUE(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
    for (int i = 0; i < 3; i++) OL_EQ(tl_kv_fill(p, ids[i]), 0);
    OL_TRUE(stats_are(p, 1, 3, 0));
    OL_EQ(tl_kv_alloc(p, 2, ids + 3), TL_EFULL);
    OL_TRUE(stats_are(p, 1, 3, 0));
    OL_EQ(tl_kv_alloc(p, 0, NULL), TL_OK);
    OL_EQ(tl_kv_alloc(p, 1, ids + 3), TL_OK);
    OL_TRUE(stats_are(p, 0, 4, 0));
    for (int i = 0; i < 4; i++) OL_EQ(tl_kv_unref(p, ids[i]), TL_OK);
    OL_TRUE(stats_are(p, 4, 0, 0));
    tl_kv_pool_destroy(p);
}

OL_TEST(unref_frees_or_caches) {
    /* WHY: at refcount 0 a block's next state depends on whether its
     *      contents are findable: a registered (full, hashed) block stays
     *      cached for a later prompt with the same prefix; anything else is
     *      free at once. free + used + cached == n_blocks throughout.
     * KIND: unit
     * CATCHES: s08
     * CHAPTER: rt.04 section 2 */
    tl_kv_cfg c = cfg_of(3, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t a = worked_block(p), b;
    OL_EQ(tl_kv_alloc(p, 1, &b), TL_OK);
    OL_TRUE(stats_are(p, 1, 2, 0));
    tl_kv_ref(p, a);
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_TRUE(stats_are(p, 1, 2, 0)); /* one holder left */
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_TRUE(stats_are(p, 1, 1, 1));
    OL_EQ(tl_kv_unref(p, b), TL_OK);
    OL_TRUE(stats_are(p, 2, 0, 1));
    tl_kv_pool_destroy(p);
}

OL_TEST(double_unref_is_einval_and_changes_nothing) {
    /* WHY: a refcount below zero would put one block on the free stack
     *      twice, so two sequences later share memory without knowing.
     *      The contract turns the second unref (and an id out of range) into
     *      TL_EINVAL with the pool unchanged; tl_kv_ref on a block nobody
     *      holds sets the error slot and does nothing.
     * KIND: fault
     * CATCHES: s09, s10
     * CHAPTER: rt.04 section 5, Pitfalls */
    tl_kv_cfg c = cfg_of(2, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t a;
    OL_EQ(tl_kv_alloc(p, 1, &a), TL_OK);
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_EQ(tl_kv_unref(p, a), TL_EINVAL);
    OL_TRUE(strstr(tl_last_error(), "tl_kv_unref") != NULL);
    OL_EQ(tl_kv_unref(p, 7), TL_EINVAL);
    OL_TRUE(stats_are(p, 2, 0, 0));
    tl_set_last_error("");
    tl_kv_ref(p, a);
    OL_TRUE(tl_last_error()[0] != '\0');
    OL_TRUE(stats_are(p, 2, 0, 0));
    uint32_t ids[2];
    OL_EQ(tl_kv_alloc(p, 2, ids), TL_OK); /* each block handed out once */
    OL_TRUE(ids[0] != ids[1]);
    tl_kv_unref(p, ids[0]);
    tl_kv_unref(p, ids[1]);
    tl_kv_pool_destroy(p);
}

OL_TEST(lookup_revives_a_cached_block_and_misses_quietly) {
    /* WHY: a hit on a cached block takes a reference and moves it back to
     *      used, so eviction can no longer take it. A miss is the normal
     *      answer for a new prompt: TL_ENOTFOUND without touching the
     *      error slot, so a caller's earlier error message survives.
     * KIND: unit
     * CATCHES: s08, s12
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(2, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t a = worked_block(p);
    const uint32_t toks[2] = {1, 2};
    uint64_t h = tl_kv_block_hash(0, toks, 2);
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_TRUE(stats_are(p, 1, 0, 1));
    uint32_t got = 99;
    OL_EQ(tl_kv_lookup(p, h, &got), TL_OK);
    OL_EQ(got, a);
    OL_TRUE(stats_are(p, 1, 1, 0));
    OL_EQ(tl_kv_lookup(p, h, &got), TL_OK); /* a used block: one more holder */
    OL_TRUE(stats_are(p, 1, 1, 0));
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_TRUE(stats_are(p, 1, 1, 0));
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_TRUE(stats_are(p, 1, 0, 1));
    tl_set_last_error("earlier message");
    OL_EQ(tl_kv_lookup(p, h ^ 1, &got), TL_ENOTFOUND);
    OL_EQ(tl_kv_lookup(p, 0, &got), TL_ENOTFOUND);
    OL_TRUE(strcmp(tl_last_error(), "earlier message") == 0);
    size_t n;
    uint8_t buf[64];
    OL_EQ(tl_kv_lookup(p, h, &got), TL_OK);
    OL_EQ(tl_kv_export(p, &got, 1, buf, sizeof buf, &n), TL_OK);
    OL_TRUE(memcmp(buf, WORKED, 60) == 0);
    tl_kv_unref(p, got);
    tl_kv_pool_destroy(p);
}

OL_TEST(eviction_takes_the_least_recently_used) {
    /* WHY: when free blocks run out, tl_kv_alloc reclaims cached blocks
     *      oldest first. Three cached blocks a, b, c (cached in that order);
     *      a hit on b makes it the most recent, so the next two allocations
     *      evict a and then c, and b's prefix still hits. evictions counts
     *      them for /metrics.
     * KIND: unit
     * CATCHES: s13, s14
     * CHAPTER: rt.04 section 3 */
    tl_kv_cfg c = cfg_of(3, 1, 1, 1, 1);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t ids[3];
    uint64_t hs[3];
    OL_EQ(tl_kv_alloc(p, 3, ids), TL_OK);
    for (uint32_t i = 0; i < 3; i++) {
        const uint32_t t = 10 + i;
        hs[i] = tl_kv_block_hash(0, &t, 1);
        OL_EQ(tl_kv_set_fill(p, ids[i], 1), TL_OK);
        OL_EQ(tl_kv_register(p, ids[i], hs[i]), TL_OK);
    }
    for (int i = 0; i < 3; i++) OL_EQ(tl_kv_unref(p, ids[i]), TL_OK); /* cached: a, b, c */
    OL_TRUE(stats_are(p, 0, 0, 3));
    uint32_t got;
    OL_EQ(tl_kv_lookup(p, hs[1], &got), TL_OK);
    OL_EQ(tl_kv_unref(p, got), TL_OK); /* b is now the most recent */
    uint32_t x;
    OL_EQ(tl_kv_alloc(p, 1, &x), TL_OK);
    OL_EQ(x, ids[0]);
    OL_EQ(evictions(p), 1);
    OL_EQ(tl_kv_lookup(p, hs[0], &got), TL_ENOTFOUND); /* a's hash left the index */
    uint32_t y;
    OL_EQ(tl_kv_alloc(p, 1, &y), TL_OK);
    OL_EQ(y, ids[2]);
    OL_EQ(evictions(p), 2);
    OL_EQ(tl_kv_lookup(p, hs[1], &got), TL_OK);
    OL_EQ(got, ids[1]);
    OL_TRUE(stats_are(p, 0, 3, 0));
    OL_EQ(tl_kv_alloc(p, 1, &y), TL_EFULL); /* nothing free, nothing cached */
    tl_kv_pool_destroy(p);
}

OL_TEST(a_revived_block_is_never_evicted) {
    /* WHY: a lookup hit turns a cached block into a used one, so it must
     *      leave the eviction list at once. Blocks a and b are cached (a
     *      older); a request looks a up and holds it; the next allocation
     *      must reclaim b, the only cached block, never the held a, whose
     *      prefix another sequence is reading.
     * KIND: unit
     * CATCHES: s11
     * CHAPTER: rt.04 section 5, Pitfalls */
    tl_kv_cfg c = cfg_of(2, 1, 1, 1, 1);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t ids[2];
    uint64_t hs[2];
    OL_EQ(tl_kv_alloc(p, 2, ids), TL_OK);
    for (uint32_t i = 0; i < 2; i++) {
        const uint32_t t = 20 + i;
        hs[i] = tl_kv_block_hash(0, &t, 1);
        write_block(p, ids[i], 1, (uint16_t)(0x700 + i));
        OL_EQ(tl_kv_register(p, ids[i], hs[i]), TL_OK);
    }
    OL_EQ(tl_kv_unref(p, ids[0]), TL_OK);
    OL_EQ(tl_kv_unref(p, ids[1]), TL_OK);
    uint32_t held, x;
    OL_EQ(tl_kv_lookup(p, hs[0], &held), TL_OK);
    OL_EQ(held, ids[0]);
    OL_EQ(tl_kv_alloc(p, 1, &x), TL_OK);
    OL_EQ(x, ids[1]);
    OL_TRUE(block_has(p, held, 1, 0x700));
    OL_EQ(tl_kv_alloc(p, 1, &x), TL_EFULL);
    tl_kv_unref(p, held);
    tl_kv_unref(p, ids[1]);
    tl_kv_pool_destroy(p);
}

OL_TEST(cow_copies_only_when_shared) {
    /* WHY: a forked sequence (beam search, parallel sampling) shares its
     *      prefix blocks. Before writing into a block, the writer calls
     *      tl_kv_cow: the only holder writes in place; a shared block is
     *      copied (contents and fill) and the writer drops its reference to
     *      the original, which the other holder keeps unchanged.
     * KIND: unit
     * CATCHES: s15, s16
     * CHAPTER: rt.04 section 3 */
    tl_kv_cfg c = cfg_of(4, 4, 2, 2, 3);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t a, out = 99;
    OL_EQ(tl_kv_alloc(p, 1, &a), TL_OK);
    write_block(p, a, 3, 0x1234);
    OL_EQ(tl_kv_cow(p, a, &out), TL_OK);
    OL_EQ(out, a);
    OL_TRUE(stats_are(p, 3, 1, 0));
    tl_kv_ref(p, a); /* fork: two holders */
    OL_EQ(tl_kv_cow(p, a, &out), TL_OK);
    OL_TRUE(out != a);
    OL_TRUE(stats_are(p, 2, 2, 0));
    OL_TRUE(block_has(p, out, 3, 0x1234));
    write_block(p, out, 4, 0x0F0F); /* the writer's copy diverges */
    OL_TRUE(block_has(p, a, 3, 0x1234));
    OL_EQ(tl_kv_unref(p, a), TL_OK); /* the original had exactly one holder left */
    OL_EQ(tl_kv_unref(p, a), TL_EINVAL);
    OL_EQ(tl_kv_unref(p, out), TL_OK);
    OL_TRUE(stats_are(p, 4, 0, 0));
    tl_kv_pool_destroy(p);
}

OL_TEST(cow_in_a_full_pool_is_efull) {
    /* WHY: copy on write allocates, so it can fail like tl_kv_alloc; it
     *      must then change nothing: both holders keep the shared block.
     * KIND: boundary
     * CATCHES: s16
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(1, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t a, out = 99;
    OL_EQ(tl_kv_alloc(p, 1, &a), TL_OK);
    tl_kv_ref(p, a);
    OL_EQ(tl_kv_cow(p, a, &out), TL_EFULL);
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_EQ(tl_kv_unref(p, a), TL_OK); /* still two holders after the failed copy */
    OL_TRUE(stats_are(p, 1, 0, 0));
    tl_kv_pool_destroy(p);
}

OL_TEST(only_full_blocks_are_registered) {
    /* WHY: a partial block's contents will still change, so it must never
     *      be findable by hash: register refuses a partial block, hash 0
     *      (the "no hash" value), and a block nobody holds. A second block
     *      with the same hash gets TL_EBUSY and stays unregistered; once
     *      registered, a block is immutable (set_fill refuses).
     * KIND: unit, boundary
     * CATCHES: s17, s18
     * CHAPTER: rt.04 section 5, Pitfalls */
    tl_kv_cfg c = cfg_of(4, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t ids[3];
    OL_EQ(tl_kv_alloc(p, 3, ids), TL_OK);
    const uint32_t toks[2] = {5, 6};
    uint64_t h = tl_kv_block_hash(0, toks, 2);
    OL_EQ(tl_kv_set_fill(p, ids[0], 1), TL_OK);
    OL_EQ(tl_kv_register(p, ids[0], h), TL_EINVAL);
    OL_EQ(tl_kv_set_fill(p, ids[0], 3), TL_EINVAL); /* above block_tokens */
    OL_EQ(tl_kv_set_fill(p, ids[0], 2), TL_OK);
    OL_EQ(tl_kv_register(p, ids[0], 0), TL_EINVAL);
    OL_EQ(tl_kv_register(p, ids[0], h), TL_OK);
    OL_EQ(tl_kv_register(p, ids[0], h), TL_OK); /* again, same hash */
    OL_EQ(tl_kv_set_fill(p, ids[0], 1), TL_EINVAL);
    OL_EQ(tl_kv_set_fill(p, ids[1], 2), TL_OK);
    OL_EQ(tl_kv_register(p, ids[1], h), TL_EBUSY);
    OL_EQ(tl_kv_unref(p, ids[1]), TL_OK);
    OL_TRUE(stats_are(p, 2, 2, 0)); /* ids[1] was not registered: free, not cached */
    OL_EQ(tl_kv_register(p, ids[1], h ^ 1), TL_EINVAL); /* not held */
    uint32_t got;
    OL_EQ(tl_kv_lookup(p, h, &got), TL_OK);
    OL_EQ(got, ids[0]);
    tl_kv_unref(p, got);
    tl_kv_unref(p, ids[0]);
    tl_kv_unref(p, ids[2]);
    tl_kv_pool_destroy(p);
}

OL_TEST(block_ptr_layout) {
    /* WHY: the paged attention kernel (L9.4) and the Python cache (L8.3)
     *      reach K and V through tl_kv_block_ptr and index the slab as
     *      [n_kv_heads][block_tokens][head_dim]. Every (block, layer, K/V)
     *      slab must be distinct, exactly Hkv * B * D f16 values long, and
     *      out-of-range ids or layers must give NULL, not a wild pointer.
     * KIND: unit, boundary
     * CATCHES: s19
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(3, 4, 2, 3, 5);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    const size_t slab = 3 * 4 * 5 * 2;
    OL_EQ(tl_kv_block_bytes(p), 2 * 2 * slab);
    tl_kv_cfg back;
    tl_kv_pool_cfg(p, &back);
    OL_TRUE(memcmp(&back, &c, sizeof c) == 0);
    uint8_t *ptrs[12];
    int k = 0;
    for (uint32_t id = 0; id < 3; id++)
        for (uint32_t l = 0; l < 2; l++)
            for (int v = 0; v < 2; v++) {
                ptrs[k] = tl_kv_block_ptr(p, id, l, v);
                OL_TRUE(ptrs[k] != NULL);
                memset(ptrs[k], k + 1, slab); /* ASan: the whole slab is ours */
                k++;
            }
    for (int i = 0; i < 12; i++) {
        OL_EQ(ptrs[i][0], i + 1);
        OL_EQ(ptrs[i][slab - 1], i + 1);
    }
    OL_TRUE(tl_kv_block_ptr(p, 3, 0, 0) == NULL);
    OL_TRUE(tl_kv_block_ptr(p, 0, 2, 1) == NULL);
    tl_kv_pool_destroy(p);
}

OL_TEST(export_zeroes_past_the_fill_and_reports_its_size) {
    /* WHY: an envelope must be a pure function of the block's contents, so
     *      positions at or past the fill are written as zero bytes even when
     *      the memory there holds old data (golden blobs compare byte for
     *      byte). A buffer too small gets TL_EFULL with *written set to the
     *      size needed and the buffer untouched; an unheld block is EINVAL.
     * KIND: unit, boundary
     * CATCHES: s20, s21
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(2, 4, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t a;
    OL_EQ(tl_kv_alloc(p, 1, &a), TL_OK);
    write_block(p, a, 4, 0x5555); /* dirty all four positions */
    OL_EQ(tl_kv_set_fill(p, a, 1), TL_OK);
    size_t need = 0;
    uint8_t small[8];
    memset(small, 0xAB, sizeof small);
    OL_EQ(tl_kv_export(p, &a, 1, small, sizeof small, &need), TL_EFULL);
    OL_EQ(need, 28 + 12 + 32 + 4);
    for (int i = 0; i < 8; i++) OL_EQ(small[i], 0xAB);
    uint8_t buf[76];
    size_t n = 0;
    OL_EQ(tl_kv_export(p, &a, 1, buf, sizeof buf, &n), TL_OK);
    OL_EQ(n, 76);
    OL_EQ(buf[28 + 8], 1); /* n_tokens */
    const uint8_t *k = buf + 40, *v = buf + 40 + 16;
    for (int i = 4; i < 16; i++) OL_EQ(k[i], 0); /* positions 1..3 of K */
    for (int i = 4; i < 16; i++) OL_EQ(v[i], 0);
    OL_TRUE(k[0] != 0 || k[1] != 0);
    uint32_t bad = 1; /* never allocated */
    OL_EQ(tl_kv_export(p, &bad, 1, buf, sizeof buf, &n), TL_EINVAL);
    tl_kv_unref(p, a);
    tl_kv_pool_destroy(p);
}

OL_TEST(export_import_roundtrip) {
    /* WHY: disaggregated serving (L10.6) moves a prompt's blocks from a
     *      prefill engine to a decode engine as one envelope. Importing into
     *      another pool and exporting again must give the same bytes; the
     *      full hashed blocks arrive registered (findable by hash), and the
     *      partial tail arrives with its fill but unregistered.
     * KIND: unit
     * CATCHES: s04, s22, s23
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(6, 2, 2, 2, 3);
    tl_kv_pool *a = NULL, *b = NULL;
    OL_EQ(tl_kv_pool_create(&c, &a), TL_OK);
    OL_EQ(tl_kv_pool_create(&c, &b), TL_OK);
    const uint32_t toks[5] = {7, 8, 9, 10, 11};
    uint32_t ids[3];
    OL_EQ(tl_kv_alloc(a, 3, ids), TL_OK);
    uint64_t h0 = tl_kv_block_hash(0, toks, 2), h1 = tl_kv_block_hash(h0, toks + 2, 2);
    write_block(a, ids[0], 2, 0x1111);
    write_block(a, ids[1], 2, 0x2222);
    write_block(a, ids[2], 1, 0x3333);
    OL_EQ(tl_kv_register(a, ids[0], h0), TL_OK);
    OL_EQ(tl_kv_register(a, ids[1], h1), TL_OK);
    size_t need = tl_kv_export_bytes(a, 3), n = 0, n2 = 0;
    uint8_t env[512], env2[512];
    OL_TRUE(need <= sizeof env);
    OL_EQ(tl_kv_export(a, ids, 3, env, sizeof env, &n), TL_OK);
    OL_EQ(n, need);
    uint32_t got[3];
    OL_EQ(tl_kv_import(b, env, n, got), TL_OK);
    OL_TRUE(stats_are(b, 3, 3, 0));
    OL_TRUE(block_has(b, got[0], 2, 0x1111));
    OL_TRUE(block_has(b, got[1], 2, 0x2222));
    OL_TRUE(block_has(b, got[2], 1, 0x3333));
    OL_EQ(tl_kv_export(b, got, 3, env2, sizeof env2, &n2), TL_OK);
    OL_EQ(n2, n);
    OL_TRUE(memcmp(env, env2, n) == 0);
    uint32_t hit;
    OL_EQ(tl_kv_lookup(b, h1, &hit), TL_OK);
    OL_EQ(hit, got[1]);
    tl_kv_unref(b, hit);
    OL_EQ(tl_kv_set_fill(b, got[2], 2), TL_OK); /* the tail is not registered: still writable */
    for (int i = 0; i < 3; i++) {
        tl_kv_unref(a, ids[i]);
        tl_kv_unref(b, got[i]);
    }
    OL_TRUE(stats_are(b, 4, 0, 2));
    tl_kv_pool_destroy(a);
    tl_kv_pool_destroy(b);
}

OL_TEST(import_keeps_the_first_holder_of_a_hash) {
    /* WHY: a decode engine may receive a block whose hash it already
     *      holds. The imported copy is still handed to the caller, but the
     *      index keeps pointing at the first block (registering the copy
     *      would be TL_EBUSY); the KV-transfer dedup of L10.6 relies on it.
     * KIND: unit
     * CATCHES: s24
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(4, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint32_t first = worked_block(p), copy, hit;
    OL_EQ(tl_kv_import(p, WORKED, 60, &copy), TL_OK);
    OL_TRUE(copy != first);
    OL_TRUE(stats_are(p, 2, 2, 0));
    const uint32_t toks[2] = {1, 2};
    OL_EQ(tl_kv_lookup(p, tl_kv_block_hash(0, toks, 2), &hit), TL_OK);
    OL_EQ(hit, first);
    OL_EQ(tl_kv_unref(p, copy), TL_OK);
    OL_TRUE(stats_are(p, 3, 1, 0)); /* the copy was never registered: freed */
    tl_kv_unref(p, hit);
    tl_kv_unref(p, first);
    OL_TRUE(stats_are(p, 3, 0, 1));
    tl_kv_pool_destroy(p);
}

OL_TEST(every_bit_flip_is_eformat) {
    /* WHY: a KV transfer that flips one bit must be refused, or a decode
     *      engine continues a prompt from corrupted keys and values. Each of
     *      the 480 single-bit flips of the worked envelope gives TL_EFORMAT
     *      (CRC, magic, version, dtype, or length) and allocates nothing.
     * KIND: fault
     * CATCHES: s25
     * CHAPTER: rt.04 section 5, Pitfalls */
    tl_kv_cfg c = cfg_of(2, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint8_t e[60];
    uint32_t ids[2];
    int bad = 0;
    for (int bit = 0; bit < 480; bit++) {
        memcpy(e, WORKED, 60);
        e[bit / 8] ^= (uint8_t)(1u << (bit % 8));
        if (tl_kv_import(p, e, 60, ids) != TL_EFORMAT) bad++;
    }
    OL_EQ(bad, 0);
    OL_TRUE(stats_are(p, 2, 0, 0));
    OL_EQ(tl_kv_import(p, WORKED, 59, ids), TL_EFORMAT); /* truncated */
    OL_EQ(tl_kv_import(p, WORKED, 10, ids), TL_EFORMAT);
    OL_EQ(tl_kv_import(p, WORKED, 60, ids), TL_OK);
    tl_kv_unref(p, ids[0]);
    tl_kv_pool_destroy(p);
}

/* WORKED with fields changed and the CRC recomputed, so only the rule
 * under test can reject it. */
static void reseal(uint8_t *e, size_t n) {
    uint32_t crc = tl_crc32c(e, n - 4, 0);
    for (int i = 0; i < 4; i++) e[n - 4 + (size_t)i] = (uint8_t)(crc >> (8 * i));
}

OL_TEST(import_reader_rules) {
    /* WHY: the reader rules of formats/kv-block.md with a valid CRC: a v1
     *      pool refuses version 2 (the craft.13 migration adds it) and a
     *      dtype that is not the version's; n_tokens must be 1..B and a
     *      hashed block must be full; dimensions that differ from the
     *      pool's are TL_ESHAPE. Nothing is allocated on any refusal.
     * KIND: boundary
     * CATCHES: s27, s28
     * CHAPTER: rt.04 section 4 */
    tl_kv_cfg c = cfg_of(2, 2, 1, 1, 2);
    tl_kv_pool *p = NULL;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    uint8_t e[60];
    uint32_t ids[2];
    memcpy(e, WORKED, 60);
    e[4] = 2; /* version 2 */
    reseal(e, 60);
    OL_EQ(tl_kv_import(p, e, 60, ids), TL_EFORMAT);
    memcpy(e, WORKED, 60);
    e[6] = 0; /* dtype TL_F32 */
    reseal(e, 60);
    OL_EQ(tl_kv_import(p, e, 60, ids), TL_EFORMAT);
    memcpy(e, WORKED, 60);
    e[36] = 0; /* n_tokens 0 */
    reseal(e, 60);
    OL_EQ(tl_kv_import(p, e, 60, ids), TL_EFORMAT);
    memcpy(e, WORKED, 60);
    e[36] = 1; /* a hashed block that is partial */
    reseal(e, 60);
    OL_EQ(tl_kv_import(p, e, 60, ids), TL_EFORMAT);
    memcpy(e, WORKED, 60);
    e[36] = 3; /* n_tokens above B */
    reseal(e, 60);
    OL_EQ(tl_kv_import(p, e, 60, ids), TL_EFORMAT);
    OL_TRUE(stats_are(p, 2, 0, 0));
    tl_kv_pool_destroy(p);
    c = cfg_of(2, 2, 1, 2, 1); /* same payload size (2 heads of dim 1), other shape */
    OL_EQ(tl_kv_pool_create(&c, &p), TL_OK);
    OL_EQ(tl_kv_import(p, WORKED, 60, ids), TL_ESHAPE);
    OL_TRUE(stats_are(p, 2, 0, 0));
    tl_kv_pool_destroy(p);
}

OL_TEST(create_rules_and_alloc_failure) {
    /* WHY: format 2 and other dtypes are TL_EUNSUPPORTED until craft.13;
     *      a zero dimension or a NULL is TL_EINVAL. Failing each allocation
     *      of create in turn must give TL_ENOMEM and leak nothing (the
     *      counting allocator checks), including the prefix index's.
     * KIND: boundary, fault
     * CATCHES: s29, s30
     * CHAPTER: rt.04 section 4 */
    tl_kv_pool *p = NULL;
    tl_kv_cfg c = cfg_of(4, 16, 2, 2, 8);
    c.format = 2;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_EUNSUPPORTED);
    c = cfg_of(4, 16, 2, 2, 8);
    c.dtype = TL_F32;
    OL_EQ(tl_kv_pool_create(&c, &p), TL_EUNSUPPORTED);
    c = cfg_of(4, 0, 2, 2, 8);
    OL_EQ(tl_kv_pool_create(&c, &p), TL_EINVAL);
    OL_EQ(tl_kv_pool_create(NULL, &p), TL_EINVAL);
    c = cfg_of(4, 16, 2, 2, 8);
    OL_EQ(tl_kv_pool_create(&c, NULL), TL_EINVAL);
    int saw_ok = 0;
    for (long fail = 0; fail < 16 && !saw_ok; fail++) {
        ol_alloc_fail_after(fail);
        tl_status st = tl_kv_pool_create(&c, &p);
        ol_alloc_fail_after(-1);
        if (st == TL_OK) {
            saw_ok = 1;
            tl_kv_pool_destroy(p);
        } else {
            OL_EQ(st, TL_ENOMEM);
        }
    }
    OL_TRUE(saw_ok);
    tl_kv_pool_destroy(NULL);
}

/* Model-based random operations (10^4 per case). The test holds a list of
 * references; each held block carries a pattern that must survive every
 * other operation, so a block reclaimed while held is detected. */
#define MAXH 256
typedef struct {
    uint32_t id;
    uint16_t tag;
    uint32_t fill;
} held_ref;

static int pool_random_ops(ol_gen *g, int size) {
    tl_kv_cfg c = cfg_of(16, 2, 2, 1, 2);
    tl_kv_pool *p = NULL;
    if (tl_kv_pool_create(&c, &p) != TL_OK) return 0;
    held_ref h[MAXH];
    int nh = 0, ok = 1;
    uint32_t refs[16] = {0}; /* the model's refcount per block id */
    uint16_t next_tag = 1;
    int steps = 200 + 100 * size;
    for (int s = 0; s < steps && ok; s++) {
        int op = (int)ol_gen_int(g, 0, 9);
        if (op <= 2 && nh < MAXH - 4) { /* alloc 1..3 and write a pattern */
            uint32_t n = (uint32_t)ol_gen_int(g, 1, 3), ids[3];
            tl_kv_stats st;
            tl_kv_stats_get(p, &st);
            tl_status r = tl_kv_alloc(p, n, ids);
            if (st.free + st.cached < n) {
                ok &= r == TL_EFULL;
            } else {
                ok &= r == TL_OK;
                for (uint32_t i = 0; i < n && ok; i++) {
                    ok &= refs[ids[i]] == 0; /* never a block someone holds */
                    refs[ids[i]] = 1;
                    uint32_t fill = (uint32_t)ol_gen_int(g, 1, 2);
                    h[nh] = (held_ref){ids[i], next_tag++, fill};
                    write_block(p, ids[i], fill, h[nh].tag);
                    if (fill == 2 && ol_gen_int(g, 0, 1)) {
                        const uint32_t t[2] = {h[nh].tag, h[nh].id};
                        tl_status rr = tl_kv_register(p, ids[i], tl_kv_block_hash(0, t, 2));
                        ok &= rr == TL_OK || rr == TL_EBUSY;
                    }
                    nh++;
                }
            }
        } else if (op <= 5 && nh > 0) { /* drop a reference */
            int k = (int)ol_gen_int(g, 0, nh - 1);
            ok &= tl_kv_unref(p, h[k].id) == TL_OK;
            refs[h[k].id]--;
            h[k] = h[--nh];
        } else if (op == 6 && nh > 0 && nh < MAXH) { /* share a reference */
            int k = (int)ol_gen_int(g, 0, nh - 1);
            tl_kv_ref(p, h[k].id);
            refs[h[k].id]++;
            h[nh++] = h[k];
        } else if (op == 7 && nh > 0) { /* copy on write, then diverge */
            int k = (int)ol_gen_int(g, 0, nh - 1);
            uint32_t out;
            uint32_t before = refs[h[k].id];
            tl_status r = tl_kv_cow(p, h[k].id, &out);
            if (r == TL_OK) {
                if (before == 1) {
                    ok &= out == h[k].id;
                } else {
                    ok &= out != h[k].id && refs[out] == 0;
                    refs[h[k].id]--;
                    refs[out] = 1;
                    ok &= block_has(p, out, h[k].fill, h[k].tag);
                    h[k].id = out;
                    h[k].tag = next_tag++;
                    write_block(p, out, h[k].fill, h[k].tag);
                }
            } else {
                ok &= r == TL_EFULL && before > 1;
            }
        } else if (op == 8 && nh > 0) { /* look a held, registered block up */
            int k = (int)ol_gen_int(g, 0, nh - 1);
            const uint32_t t[2] = {h[k].tag, h[k].id};
            uint32_t got;
            if (h[k].fill == 2 && tl_kv_lookup(p, tl_kv_block_hash(0, t, 2), &got) == TL_OK) {
                ok &= got == h[k].id; /* a held block is never evicted */
                tl_kv_unref(p, got);
            }
        } else { /* export and re-import one held block */
            if (nh == 0) continue;
            int k = (int)ol_gen_int(g, 0, nh - 1);
            uint8_t env[128];
            size_t n;
            ok &= tl_kv_export(p, &h[k].id, 1, env, sizeof env, &n) == TL_OK;
            uint32_t got;
            tl_status r = tl_kv_import(p, env, n, &got);
            if (r == TL_OK) {
                ok &= refs[got] == 0 && block_has(p, got, h[k].fill, h[k].tag);
                ok &= tl_kv_unref(p, got) == TL_OK;
            } else {
                ok &= r == TL_EFULL;
            }
        }
        tl_kv_stats st;
        tl_kv_stats_get(p, &st);
        uint32_t used = 0;
        for (int i = 0; i < 16; i++) used += refs[i] > 0;
        ok &= st.free + st.used + st.cached == 16 && st.used == used;
        for (int i = 0; i < nh && ok; i++) ok &= block_has(p, h[i].id, h[i].fill, h[i].tag);
    }
    for (int i = 0; i < nh; i++) ok &= tl_kv_unref(p, h[i].id) == TL_OK;
    tl_kv_stats st;
    tl_kv_stats_get(p, &st);
    ok &= st.used == 0 && st.free + st.cached == 16;
    tl_kv_pool_destroy(p);
    return ok;
}

OL_TEST(random_ops_keep_the_invariants) {
    /* WHY: the pool's three promises under any interleaving: free + used +
     *      cached == n_blocks after every call, refcounts match a model
     *      (never below zero, never a held block handed out again), and a
     *      referenced block is never evicted or overwritten. 40 seeded runs
     *      of alloc, ref, unref, cow, register, lookup, export, and import.
     * KIND: property
     * CATCHES: s15, s16, s22, s24, m03
     * CHAPTER: rt.04 section 4 */
    OL_CHECK_PROP(pool_random_ops, 40, 98);
}

int main(void) { return OL_RUN_ALL(); }
