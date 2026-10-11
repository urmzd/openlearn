/* My tests for rt.04 (rung R3: written first, red against the stub). They
 * include only the contract headers and the test kit. */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_test.h"

static tl_kv_pool *pool(uint32_t n, uint32_t B, uint32_t L, uint32_t H, uint32_t D) {
    tl_kv_cfg c = {n, B, L, H, D, TL_F16, TL_KV_FORMAT_V1};
    tl_kv_pool *p = NULL;
    return tl_kv_pool_create(&c, &p) == TL_OK ? p : NULL;
}

static void counts(const tl_kv_pool *p, uint32_t *f, uint32_t *u, uint32_t *c) {
    tl_kv_stats s;
    tl_kv_stats_get(p, &s);
    *f = s.free;
    *u = s.used;
    *c = s.cached;
}

OL_TEST(hash_chain_by_hand) {
    const uint32_t a[2] = {1, 2}, b[2] = {3, 4};
    uint64_t h0 = tl_kv_block_hash(0, a, 2);
    OL_EQ(h0, 0xA91AB0C1027B9366ull);
    OL_EQ(tl_kv_block_hash(h0, b, 2), 0x03DF829571605C60ull);
    OL_EQ(tl_crc32c("123456789", 9, 0), 0xE3069283u);
}

OL_TEST(alloc_all_or_nothing) {
    tl_kv_pool *p = pool(3, 2, 1, 1, 2);
    OL_TRUE(p != NULL);
    uint32_t ids[4], f, u, c;
    OL_EQ(tl_kv_alloc(p, 2, ids), TL_OK);
    OL_EQ(tl_kv_alloc(p, 2, ids + 2), TL_EFULL);
    counts(p, &f, &u, &c);
    OL_EQ(f, 1);
    OL_EQ(u, 2);
    OL_EQ(tl_kv_unref(p, ids[0]), TL_OK);
    OL_EQ(tl_kv_unref(p, ids[1]), TL_OK);
    counts(p, &f, &u, &c);
    OL_EQ(f, 3);
    OL_EQ(tl_kv_alloc(p, 3, ids), TL_OK); /* every id came back */
    for (int i = 0; i < 3; i++) tl_kv_unref(p, ids[i]);
    tl_kv_pool_destroy(p);
}

OL_TEST(unref_caches_registered_blocks) {
    tl_kv_pool *p = pool(2, 1, 1, 1, 1);
    uint32_t ids[2], f, u, c;
    OL_EQ(tl_kv_alloc(p, 2, ids), TL_OK);
    const uint32_t t = 7;
    tl_kv_set_fill(p, ids[0], 1);
    OL_EQ(tl_kv_register(p, ids[0], tl_kv_block_hash(0, &t, 1)), TL_OK);
    tl_kv_unref(p, ids[0]);
    tl_kv_unref(p, ids[1]);
    counts(p, &f, &u, &c);
    OL_EQ(f, 1);
    OL_EQ(c, 1);
    tl_kv_pool_destroy(p);
}

OL_TEST(double_unref_refused) {
    tl_kv_pool *p = pool(2, 1, 1, 1, 1);
    uint32_t id, f, u, c;
    OL_EQ(tl_kv_alloc(p, 1, &id), TL_OK);
    OL_EQ(tl_kv_unref(p, id), TL_OK);
    OL_EQ(tl_kv_unref(p, id), TL_EINVAL);
    tl_kv_ref(p, id); /* nobody holds it: refused */
    counts(p, &f, &u, &c);
    OL_EQ(f, 2);
    OL_EQ(u, 0);
    tl_kv_pool_destroy(p);
}

OL_TEST(lookup_hit_is_not_evicted) {
    tl_kv_pool *p = pool(3, 1, 1, 1, 1);
    uint32_t ids[3], hit, x;
    uint64_t h[3];
    OL_EQ(tl_kv_alloc(p, 3, ids), TL_OK);
    for (uint32_t i = 0; i < 3; i++) {
        const uint32_t t = 40 + i;
        h[i] = tl_kv_block_hash(0, &t, 1);
        tl_kv_set_fill(p, ids[i], 1);
        tl_kv_register(p, ids[i], h[i]);
    }
    for (int i = 0; i < 3; i++) tl_kv_unref(p, ids[i]); /* cached oldest first: 0, 1, 2 */
    OL_EQ(tl_kv_lookup(p, h[0], &hit), TL_OK);         /* 0 is held again */
    OL_EQ(hit, ids[0]);
    tl_set_last_error("mine");
    OL_EQ(tl_kv_lookup(p, h[0] ^ 1, &x), TL_ENOTFOUND);
    OL_TRUE(strcmp(tl_last_error(), "mine") == 0);
    OL_EQ(tl_kv_alloc(p, 1, &x), TL_OK); /* evicts 1, the oldest cached */
    OL_EQ(x, ids[1]);
    tl_kv_stats s;
    tl_kv_stats_get(p, &s);
    OL_EQ(s.evictions, 1);
    OL_EQ(tl_kv_lookup(p, h[2], &x), TL_OK);
    OL_EQ(tl_kv_lookup(p, h[1], &x), TL_ENOTFOUND);
    tl_kv_unref(p, ids[2]);
    tl_kv_unref(p, ids[1]);
    tl_kv_unref(p, hit);
    tl_kv_pool_destroy(p);
}

OL_TEST(cow_copies_shared_blocks) {
    tl_kv_pool *p = pool(2, 2, 1, 1, 2);
    uint32_t a, out;
    OL_EQ(tl_kv_alloc(p, 1, &a), TL_OK);
    uint16_t *k = tl_kv_block_ptr(p, a, 0, 0), *v = tl_kv_block_ptr(p, a, 0, 1);
    OL_TRUE(k != NULL && v != NULL && k != v);
    k[0] = 0x3C00;
    v[0] = 0x4000;
    tl_kv_set_fill(p, a, 1);
    OL_EQ(tl_kv_cow(p, a, &out), TL_OK);
    OL_EQ(out, a);
    tl_kv_ref(p, a);
    OL_EQ(tl_kv_cow(p, a, &out), TL_OK);
    OL_TRUE(out != a);
    uint16_t *k2 = tl_kv_block_ptr(p, out, 0, 0);
    OL_EQ(k2[0], 0x3C00);
    OL_EQ(tl_kv_fill(p, out), 1);
    k2[0] = 0;
    OL_EQ(k[0], 0x3C00);
    OL_EQ(tl_kv_unref(p, a), TL_OK);
    OL_EQ(tl_kv_unref(p, a), TL_EINVAL); /* the writer's reference moved to the copy */
    tl_kv_unref(p, out);
    tl_kv_pool_destroy(p);
}

OL_TEST(partial_blocks_not_registered) {
    tl_kv_pool *p = pool(3, 2, 1, 1, 1);
    uint32_t ids[2];
    OL_EQ(tl_kv_alloc(p, 2, ids), TL_OK);
    const uint32_t t[2] = {1, 2};
    uint64_t h = tl_kv_block_hash(0, t, 2);
    tl_kv_set_fill(p, ids[0], 1);
    OL_EQ(tl_kv_register(p, ids[0], h), TL_EINVAL);
    tl_kv_set_fill(p, ids[0], 2);
    OL_EQ(tl_kv_register(p, ids[0], h), TL_OK);
    tl_kv_set_fill(p, ids[1], 2);
    OL_EQ(tl_kv_register(p, ids[1], h), TL_EBUSY);
    tl_kv_unref(p, ids[0]);
    tl_kv_unref(p, ids[1]);
    tl_kv_pool_destroy(p);
}

OL_TEST(roundtrip_and_bit_flip) {
    tl_kv_pool *a = pool(4, 2, 1, 2, 2), *b = pool(4, 2, 1, 2, 2), *other = pool(4, 2, 1, 1, 4);
    OL_TRUE(a && b && other);
    uint32_t ids[2], got[2], o[2];
    OL_EQ(tl_kv_alloc(a, 2, ids), TL_OK);
    for (uint32_t i = 0; i < 2; i++)
        for (int kv = 0; kv < 2; kv++) {
            uint16_t *s = tl_kv_block_ptr(a, ids[i], 0, kv);
            for (int j = 0; j < 8; j++) s[j] = (uint16_t)(0x3C00 + 64 * j + 7 * kv + 300 * i);
        }
    const uint32_t t[2] = {5, 6};
    tl_kv_set_fill(a, ids[0], 2);
    tl_kv_set_fill(a, ids[1], 1);
    uint64_t h = tl_kv_block_hash(0, t, 2);
    tl_kv_register(a, ids[0], h);
    size_t need = 0, n = 0, n2 = 0;
    unsigned char small[4], e[200], e2[200];
    OL_EQ(tl_kv_export(a, ids, 2, small, sizeof small, &need), TL_EFULL);
    OL_EQ(need, tl_kv_export_bytes(a, 2));
    OL_EQ(tl_kv_export(a, ids, 2, e, sizeof e, &n), TL_OK);
    OL_EQ(n, 28 + 2 * (12 + 32) + 4);
    OL_TRUE(memcmp(e, "TLKV\x01\x00\x01\x00\x02\x00\x00\x00\x02\x00\x00\x00", 16) == 0);
    OL_EQ(e[28 + 44 + 8], 1); /* the tail's n_tokens */
    OL_EQ(e[28 + 44 + 12 + 4], 0); /* position 1 of the tail is zero */
    OL_EQ(tl_kv_import(b, e, n, got), TL_OK);
    OL_EQ(tl_kv_export(b, got, 2, e2, sizeof e2, &n2), TL_OK);
    OL_TRUE(n2 == n && memcmp(e, e2, n) == 0);
    uint32_t hit;
    OL_EQ(tl_kv_lookup(b, h, &hit), TL_OK);
    OL_EQ(hit, got[0]);
    OL_EQ(tl_kv_import(b, e, n, o), TL_OK); /* the hash is held: a private copy */
    OL_EQ(tl_kv_lookup(b, h, &hit), TL_OK);
    OL_EQ(hit, got[0]);
    OL_EQ(tl_kv_import(other, e, n, o + 0), TL_ESHAPE);
    e[40] ^= 0x10;
    OL_EQ(tl_kv_import(b, e, n, o), TL_EFORMAT);
    tl_kv_stats s;
    tl_kv_stats_get(b, &s);
    OL_EQ(s.used, 4);
    tl_kv_pool_destroy(a);
    tl_kv_pool_destroy(b);
    tl_kv_pool_destroy(other);
    tl_kv_cfg c2 = {4, 2, 1, 1, 1, TL_F16, 2};
    tl_kv_pool *x = NULL;
    OL_EQ(tl_kv_pool_create(&c2, &x), TL_EUNSUPPORTED);
}

int main(void) { return OL_RUN_ALL(); }
