/* Course tests for ds.02: the Swiss table tl_map of tinyllm/ds.h
 * (c/src/ds/swiss.c), built with ASan and UBSan and the counting allocator.
 *
 * Several tests pick keys by the chapter's hash, the SplitMix64 finalizer
 * mix64: H2 = mix64(k) & 0x7F is the control byte and H1 = mix64(k) >> 7
 * picks the home group. Those keys force the hard cases (16 keys sharing
 * one control byte, a home group that overflows); the assertions are plain
 * map semantics, so they hold for any hash.
 *
 * Allocation is observed through the hook: ol_alloc_fail_after(0) makes
 * every allocation fail, so "this put succeeds" proves it did not grow the
 * table, and a sizing hook records the largest block ever requested.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"
#include "_baseline_lp.h"

static uint64_t mix64(uint64_t x) { return lp_mix64(x); }
static unsigned h2_of(uint64_t k) { return (unsigned)(mix64(k) & 0x7F); }
static size_t home_of(uint64_t k, size_t groups) { return (size_t)(mix64(k) >> 7) & (groups - 1); }

/* The next key at or after *from whose hash satisfies the predicate. */
static uint64_t next_key(uint64_t *from, int (*pred)(uint64_t, uint64_t), uint64_t arg) {
    while (!pred(*from, arg)) (*from)++;
    return (*from)++;
}
static int h2_is(uint64_t k, uint64_t want) { return h2_of(k) == want; }
static int home2_is(uint64_t k, uint64_t want) { return home_of(k, 2) == want; }

static void restore_harness_hook(void) {
#if OL__COUNTING
    tl_allocator a = {ol__alloc, ol__free, NULL};
    tl_set_allocator(&a);
#else
    tl_set_allocator(NULL);
#endif
}

static size_t biggest_request;
static void *sizing_alloc(void *user, size_t n, size_t align) {
    if (n > biggest_request) biggest_request = n;
#if OL__COUNTING
    return ol__alloc(user, n, align);
#else
    (void)user;
    size_t size = (n + align - 1) / align * align;
    return aligned_alloc(align < sizeof(void *) ? sizeof(void *) : align, size ? size : align);
#endif
}
static void sizing_free(void *user, void *p) {
#if OL__COUNTING
    ol__free(user, p);
#else
    (void)user;
    free(p);
#endif
}

OL_TEST(hand_example_fourteen_fit_the_fifteenth_grows) {
    /* WHY: the chapter's worked load example. One group is 16 slots and the
     *      maximum load is 7/8, so create(14) gives 16 slots and 14 keys go
     *      in with every allocation failing; the 15th key needs a bigger
     *      table and gets TL_ENOMEM. create(15) needs 32 slots: 28 keys fit.
     * KIND: unit, boundary, smoke
     * CATCHES: s04, s05, m01
     * CHAPTER: ds.02 section 3 */
    tl_map *m = NULL;
    OL_EQ(tl_map_create(14, &m), TL_OK);
    ol_alloc_fail_after(0);
    for (uint64_t k = 1; k <= 14; k++) OL_EQ(tl_map_put(m, k, 100 + k), TL_OK);
    OL_EQ(tl_map_put(m, 15, 115), TL_ENOMEM);
    ol_alloc_fail_after(-1);
    OL_EQ(tl_map_len(m), 14);
    uint64_t v = 0;
    OL_EQ(tl_map_get(m, 15, &v), 0);
    OL_EQ(tl_map_put(m, 15, 115), TL_OK);
    OL_EQ(tl_map_len(m), 15);
    tl_map_destroy(m);

    OL_EQ(tl_map_create(15, &m), TL_OK);
    ol_alloc_fail_after(0);
    for (uint64_t k = 1; k <= 28; k++) OL_EQ(tl_map_put(m, k, k), TL_OK);
    OL_EQ(tl_map_put(m, 29, 29), TL_ENOMEM);
    ol_alloc_fail_after(-1);
    tl_map_destroy(m);
}

OL_TEST(put_get_overwrite_delete) {
    /* WHY: the map semantics before any probing detail: put inserts or
     *      overwrites (len counts keys, not puts), get reports presence and
     *      may skip the value, del reports whether the key was there.
     * KIND: unit
     * CATCHES: s09, m02, m03
     * CHAPTER: ds.02 section 4 */
    tl_map *m = NULL;
    OL_EQ(tl_map_create(0, &m), TL_OK);
    OL_EQ(tl_map_len(m), 0);
    uint64_t v = 7;
    OL_EQ(tl_map_get(m, 42, &v), 0);
    OL_EQ(v, 7); /* untouched on a miss */
    OL_EQ(tl_map_put(m, 42, 1), TL_OK);
    OL_EQ(tl_map_put(m, 42, 2), TL_OK);
    OL_EQ(tl_map_len(m), 1);
    OL_EQ(tl_map_get(m, 42, &v), 1);
    OL_EQ(v, 2);
    OL_EQ(tl_map_get(m, 42, NULL), 1);
    OL_EQ(tl_map_del(m, 42), 1);
    OL_EQ(tl_map_del(m, 42), 0);
    OL_EQ(tl_map_len(m), 0);
    OL_EQ(tl_map_get(m, 42, &v), 0);
    tl_map_destroy(m);
}

OL_TEST(every_u64_is_a_key) {
    /* WHY: rt.04 keys the index by block hash, any u64 but 0; KV transfer
     *      dedup and other callers may use 0 and UINT64_MAX. A table that
     *      reserves a key value as "empty" (as practice c/02's NULL string
     *      did) loses that key; control bytes make every key legal.
     * KIND: boundary, regression
     * CHAPTER: ds.02 section 2 */
    tl_map *m = NULL;
    OL_EQ(tl_map_create(4, &m), TL_OK);
    const uint64_t ks[4] = {0, UINT64_MAX, 0x80, 0xFE};
    for (int i = 0; i < 4; i++) OL_EQ(tl_map_put(m, ks[i], (uint64_t)i + 1), TL_OK);
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        OL_EQ(tl_map_get(m, ks[i], &v), 1);
        OL_EQ(v, (uint64_t)i + 1);
    }
    OL_EQ(tl_map_len(m), 4);
    tl_map_destroy(m);
}

OL_TEST(keys_sharing_a_control_byte_are_told_apart) {
    /* WHY: the SWAR match only says "this control byte equals H2". With 14
     *      keys that all have the same H2 in one 16-slot group, every match
     *      word reports 14 candidates; only comparing the stored key finds
     *      the right one, and a 15th key with the same H2 is still absent.
     * KIND: unit
     * CATCHES: s02, s10
     * CHAPTER: ds.02 section 2 */
    tl_map *m = NULL;
    OL_EQ(tl_map_create(14, &m), TL_OK);
    uint64_t from = 1000, keys[15];
    for (int i = 0; i < 15; i++) keys[i] = next_key(&from, h2_is, 0x2A);
    for (int i = 0; i < 14; i++) OL_EQ(tl_map_put(m, keys[i], keys[i] * 3), TL_OK);
    for (int i = 0; i < 14; i++) {
        uint64_t v = 0;
        OL_EQ(tl_map_get(m, keys[i], &v), 1);
        OL_EQ(v, keys[i] * 3);
    }
    OL_EQ(tl_map_get(m, keys[14], NULL), 0);
    OL_EQ(tl_map_del(m, keys[14]), 0);
    OL_EQ(tl_map_del(m, keys[5]), 1);
    OL_EQ(tl_map_get(m, keys[5], NULL), 0);
    OL_EQ(tl_map_get(m, keys[6], NULL), 1);
    tl_map_destroy(m);
}

/* 20 keys whose home is group 0 of a 32-slot table (16 fill group 0, four
 * spill to group 1), and 4 whose home is group 1. */
static void spill_keys(uint64_t *home0, uint64_t *home1) {
    uint64_t from = 1;
    for (int i = 0; i < 20; i++) home0[i] = next_key(&from, home2_is, 0);
    from = 1;
    for (int i = 0; i < 4; i++) home1[i] = next_key(&from, home2_is, 1);
}

OL_TEST(probe_continues_past_a_full_group) {
    /* WHY: a key whose home group is full lives further along its probe.
     *      Deleting from the full group must leave a tombstone: an EMPTY
     *      there would end the probe before the spilled keys, and a lookup
     *      that treats a tombstone as the end loses them the same way.
     * KIND: unit
     * CATCHES: s01, s03
     * CHAPTER: ds.02 section 2 */
    uint64_t a[20], b[4];
    spill_keys(a, b);
    tl_map *m = NULL;
    OL_EQ(tl_map_create(28, &m), TL_OK);
    for (int i = 0; i < 20; i++) OL_EQ(tl_map_put(m, a[i], (uint64_t)i), TL_OK);
    for (int i = 0; i < 4; i++) OL_EQ(tl_map_put(m, b[i], 100 + (uint64_t)i), TL_OK);
    for (int i = 0; i < 16; i += 3) OL_EQ(tl_map_del(m, a[i]), 1); /* the first 16 filled group 0 */
    for (int i = 0; i < 20; i++) {
        uint64_t v = 0;
        OL_EQ(tl_map_get(m, a[i], &v), i % 3 == 0 && i < 16 ? 0 : 1);
        if (!(i % 3 == 0 && i < 16)) OL_EQ(v, (uint64_t)i);
    }
    for (int i = 0; i < 4; i++) OL_EQ(tl_map_get(m, b[i], NULL), 1);
    OL_EQ(tl_map_len(m), 24 - 6);
    tl_map_destroy(m);
}

OL_TEST(reput_after_a_tombstone_overwrites) {
    /* WHY: an insert may reuse the first tombstone on its probe only after
     *      checking the whole probe for the key. A spilled key put again
     *      after a delete in its full home group must overwrite its old
     *      slot, not land a second copy in the tombstone, or a later delete
     *      leaves the stale copy behind.
     * KIND: unit
     * CATCHES: s08
     * CHAPTER: ds.02 section 5, Pitfalls */
    uint64_t a[20], b[4];
    spill_keys(a, b);
    tl_map *m = NULL;
    OL_EQ(tl_map_create(28, &m), TL_OK);
    for (int i = 0; i < 20; i++) OL_EQ(tl_map_put(m, a[i], 1), TL_OK);
    OL_EQ(tl_map_del(m, a[0]), 1);         /* tombstone in group 0 */
    OL_EQ(tl_map_put(m, a[19], 2), TL_OK); /* a[19] lives in group 1 */
    OL_EQ(tl_map_len(m), 19);
    uint64_t v = 0;
    OL_EQ(tl_map_get(m, a[19], &v), 1);
    OL_EQ(v, 2);
    OL_EQ(tl_map_del(m, a[19]), 1);
    OL_EQ(tl_map_get(m, a[19], NULL), 0);
    OL_EQ(tl_map_len(m), 18);
    tl_map_destroy(m);
}

OL_TEST(churn_in_a_group_with_room_never_allocates) {
    /* WHY: rt.04 registers and evicts block hashes for the life of the
     *      engine. In a group that still has an EMPTY slot no probe ever
     *      passed, so a delete there can write EMPTY instead of a
     *      tombstone; then 10,000 delete-and-insert rounds at 13 live keys
     *      in 16 slots never fill the table and never allocate.
     * KIND: fault
     * CATCHES: m01
     * CHAPTER: ds.02 section 2 */
    tl_map *m = NULL;
    OL_EQ(tl_map_create(14, &m), TL_OK);
    for (uint64_t k = 0; k < 13; k++) OL_EQ(tl_map_put(m, k, k), TL_OK);
    ol_alloc_fail_after(0);
    int ok = 1;
    for (uint64_t r = 0; r < 10000 && ok; r++) {
        ok &= tl_map_del(m, r) == 1;
        ok &= tl_map_put(m, r + 13, r + 13) == TL_OK;
    }
    ol_alloc_fail_after(-1);
    OL_TRUE(ok);
    OL_EQ(tl_map_len(m), 13);
    for (uint64_t k = 10000; k < 10013; k++) OL_EQ(tl_map_get(m, k, NULL), 1);
    tl_map_destroy(m);
}

OL_TEST(tombstone_churn_rehashes_without_growing) {
    /* WHY: deletes in full groups leave tombstones, and tombstones count
     *      toward the 7/8 load. When they fill the table it must be rebuilt
     *      at the SAME size (dropping them), not doubled: with 28 live keys
     *      in 32 slots (both groups nearly full, so deletes leave
     *      tombstones) the largest block the table ever asks for stays what
     *      its first build asked for, over 20,000 delete-and-insert rounds.
     * KIND: property
     * CATCHES: s13
     * CHAPTER: ds.02 section 2 */
    tl_allocator sizing = {sizing_alloc, sizing_free, NULL};
    biggest_request = 0;
    OL_EQ(tl_set_allocator(&sizing), TL_OK);
    tl_map *m = NULL;
    int ok = tl_map_create(28, &m) == TL_OK;
    size_t first = biggest_request;
    ol_gen g = {7};
    uint64_t live[28];
    for (int i = 0; i < 28; i++) {
        live[i] = (uint64_t)i;
        ok &= tl_map_put(m, live[i], 1) == TL_OK;
    }
    uint64_t next = 1000;
    for (int r = 0; r < 20000 && ok; r++) {
        int i = (int)ol_gen_int(&g, 0, 27);
        ok &= tl_map_del(m, live[i]) == 1;
        live[i] = next++;
        ok &= tl_map_put(m, live[i], 1) == TL_OK;
    }
    for (int i = 0; i < 28; i++) ok &= tl_map_get(m, live[i], NULL) == 1;
    size_t len = tl_map_len(m), biggest = biggest_request;
    tl_map_destroy(m);
    restore_harness_hook();
    OL_TRUE(ok);
    OL_EQ(len, 28);
    OL_TRUE(first > 0);
    OL_EQ(biggest, first);
}

OL_TEST(failed_growth_keeps_the_old_table) {
    /* WHY: the contract says a growth that fails leaves the old table valid
     *      and unchanged. Failing the rebuild's first allocation, then its
     *      second, and so on until one succeeds, must each time give
     *      TL_ENOMEM with every key still there, nothing leaked, and a
     *      working table afterwards. Freeing the old arrays before the new
     *      ones exist would lose every key.
     * KIND: fault
     * CATCHES: s07
     * CHAPTER: ds.02 section 5, Pitfalls */
    tl_map *m = NULL;
    OL_EQ(tl_map_create(14, &m), TL_OK);
    for (uint64_t k = 0; k < 14; k++) OL_EQ(tl_map_put(m, k * 7919, k), TL_OK);
    int failures = 0;
    for (long fail = 0; fail < 8; fail++) {
        ol_alloc_fail_after(fail);
        tl_status st = tl_map_put(m, 999999, 1);
        ol_alloc_fail_after(-1);
        if (st == TL_OK) break;
        OL_EQ(st, TL_ENOMEM);
        failures++;
        OL_EQ(tl_map_len(m), 14);
        OL_EQ(tl_map_get(m, 999999, NULL), 0);
        for (uint64_t k = 0; k < 14; k++) {
            uint64_t v = 99;
            OL_EQ(tl_map_get(m, k * 7919, &v), 1);
            OL_EQ(v, k);
        }
    }
    OL_TRUE(failures >= 1);
    OL_EQ(tl_map_len(m), 15);
    OL_EQ(tl_map_get(m, 999999, NULL), 1);
    tl_map_destroy(m);
}

OL_TEST(create_and_null_arguments) {
    /* WHY: the ABI rules: a NULL out pointer is TL_EINVAL, a failing hook
     *      is TL_ENOMEM with nothing leaked, and the read-only calls treat a
     *      NULL map as empty (destroy(NULL) does nothing), so cleanup paths
     *      in rt.04 need no special cases.
     * KIND: boundary, fault
     * CATCHES: s14
     * CHAPTER: ds.02 section 4 */
    OL_EQ(tl_map_create(0, NULL), TL_EINVAL);
    tl_map *m = NULL;
    for (long fail = 0; fail < 4; fail++) {
        ol_alloc_fail_after(fail);
        tl_status st = tl_map_create(100, &m);
        ol_alloc_fail_after(-1);
        if (st == TL_OK) {
            OL_EQ(tl_map_put(m, 1, 1), TL_OK);
            tl_map_destroy(m);
        } else {
            OL_EQ(st, TL_ENOMEM);
        }
    }
    OL_EQ(tl_map_put(NULL, 1, 1), TL_EINVAL);
    OL_EQ(tl_map_get(NULL, 1, NULL), 0);
    OL_EQ(tl_map_del(NULL, 1), 0);
    OL_EQ(tl_map_len(NULL), 0);
    tl_map_destroy(NULL);
}

/* 10^6 random operations against the baseline linear-probing map. */
static int matches_baseline(ol_gen *g, int size) {
    tl_map *m = NULL;
    if (tl_map_create(0, &m) != TL_OK) return 0;
    lp_map b;
    lp_init(&b);
    int ok = 1;
    long ops = 200000L * (size > 0 ? size : 1);
    uint64_t span = 64 + (uint64_t)size * 4000; /* a small key range keeps hits frequent */
    for (long i = 0; i < ops && ok; i++) {
        uint64_t k = (uint64_t)ol_gen_int(g, 0, (int64_t)span) * 0x9E3779B97F4A7C15ull;
        int op = (int)ol_gen_int(g, 0, 9);
        if (op < 4) {
            uint64_t v = ol_gen_u64(g);
            ok &= tl_map_put(m, k, v) == TL_OK;
            ok &= lp_put(&b, k, v) == 0;
        } else if (op < 7) {
            uint64_t v1 = 0, v2 = 0;
            int f1 = tl_map_get(m, k, &v1), f2 = lp_get(&b, k, &v2);
            ok &= f1 == f2 && (!f1 || v1 == v2);
        } else {
            ok &= tl_map_del(m, k) == lp_del(&b, k);
        }
        ok &= tl_map_len(m) == b.len;
    }
    tl_map_destroy(m);
    lp_free(&b);
    return ok;
}

OL_TEST(differential_against_linear_probing) {
    /* WHY: the Swiss table must be a map: 10^6 seeded puts, gets, and
     *      deletes (five sizes of key range, so the table grows, churns
     *      tombstones, and rehashes) agree with the practice c/02
     *      linear-probing baseline on every result and every length.
     * KIND: differential
     * CATCHES: s01, s03, s06, s08, s10
     * CHAPTER: ds.02 section 4 */
    OL_CHECK_PROP(matches_baseline, 5, 1);
}

int main(void) { return OL_RUN_ALL(); }
