/* Course tests for ds.01: c/src/ds/vec.c against tinyllm/ds.h (ds.01),
 * built with ASan and UBSan and the counting allocator of ol_test.h, which
 * fails any case that ends with a block still live (the leak check) and lets
 * a case make the next allocation fail (ol_alloc_fail_after).
 *
 * The worked example (chapter section 3) pushes the u32 values 10, 20, ...,
 * 50 into a fresh vec: capacity 0, then 4 at the first push, then 8 at the
 * fifth, with the four old elements copied across.
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

typedef struct {
    unsigned char b[3];
} three; /* an element size that is not a power of two */

OL_TEST(hand_example) {
    /* WHY: section 3 by hand: 5 pushes of u32 grow cap 0 -> 4 -> 8; the
     *      fifth push moves the block (old elements copied, then the old
     *      block freed), and every element reads back at data + i * elem.
     * KIND: unit, smoke
     * CATCHES: s02, s06, s07, s09, m01
     * CHAPTER: ds.01 section 3 */
    tl_vec v;
    OL_EQ(tl_vec_init(&v, sizeof(uint32_t)), TL_OK);
    OL_TRUE(v.data == NULL);
    OL_EQ(v.len, 0);
    OL_EQ(v.cap, 0);
    OL_EQ(v.elem, 4);
    uint32_t caps[5];
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t x = 10 * (i + 1);
        OL_EQ(tl_vec_push(&v, &x), TL_OK);
        caps[i] = (uint32_t)v.cap;
    }
    OL_EQ(caps[0], 4);
    OL_EQ(caps[3], 4);
    OL_EQ(caps[4], 8);
    OL_EQ(v.len, 5);
    for (uint32_t i = 0; i < 5; i++) {
        OL_EQ(*(uint32_t *)tl_vec_at(&v, i), 10 * (i + 1));
        OL_EQ(((uint32_t *)v.data)[i], 10 * (i + 1));
    }
    tl_vec_free(&v);
}

OL_TEST(growth_is_geometric) {
    /* WHY: amortized O(1) push needs the capacity to at least double: 100000
     *      pushes may move the block at most about log2(100000) + 1 = 18
     *      times, and cap stays within [len, 2 len] once it is past the first
     *      block. Growing by a constant makes n pushes cost O(n^2) copies (a
     *      realloc storm in rt.04's block tables).
     * KIND: unit
     * CATCHES: s01, s02, s05, s06, s07, s09
     * CHAPTER: ds.01 section 2.2 */
    tl_vec v;
    OL_EQ(tl_vec_init(&v, sizeof(uint64_t)), TL_OK);
    int moves = 0;
    size_t last_cap = 0;
    for (uint64_t i = 0; i < 100000; i++) {
        OL_EQ(tl_vec_push(&v, &i), TL_OK);
        if (v.cap != last_cap) {
            moves++;
            OL_TRUE(v.cap >= 2 * last_cap);
            last_cap = v.cap;
        }
        OL_TRUE(v.cap >= v.len && (v.len < 4 || v.cap <= 2 * v.len));
    }
    OL_TRUE(moves <= 18);
    for (uint64_t i = 0; i < 100000; i += 997) OL_EQ(*(uint64_t *)tl_vec_at(&v, i), i);
    tl_vec_free(&v);
}

OL_TEST(reserve_grows_exactly_and_never_shrinks) {
    /* WHY: reserve(n) makes room for n elements in one allocation (rt.04
     *      reserves a sequence's whole block table up front), keeps the
     *      elements, never shrinks, and reserve below the capacity is a no-op
     *      that does not allocate.
     * KIND: unit
     * CATCHES: s02, s06, s07, s08, s09
     * CHAPTER: ds.01 section 2.3 */
    tl_vec v;
    OL_EQ(tl_vec_init(&v, sizeof(three)), TL_OK);
    three t = {{1, 2, 3}};
    OL_EQ(tl_vec_push(&v, &t), TL_OK);
    OL_EQ(tl_vec_reserve(&v, 1000), TL_OK);
    OL_EQ(v.cap, 1000);
    OL_EQ(v.len, 1);
    OL_TRUE(memcmp(tl_vec_at(&v, 0), &t, 3) == 0);
    void *data = v.data;
    long live = ol_live_allocs();
    OL_EQ(tl_vec_reserve(&v, 10), TL_OK);
    OL_EQ(tl_vec_reserve(&v, 1000), TL_OK);
    OL_EQ(v.cap, 1000);
    OL_TRUE(v.data == data);
    OL_EQ(ol_live_allocs(), live);
    for (int i = 0; i < 999; i++) OL_EQ(tl_vec_push(&v, &t), TL_OK);
    OL_TRUE(v.data == data); /* no move while within the reservation */
    tl_vec_free(&v);
}

OL_TEST(at_is_bounds_checked) {
    /* WHY: tl_vec_at(v, len) is one past the end: NULL with the error slot
     *      set, never a pointer into the spare capacity (which holds garbage)
     *      or past the block.
     * KIND: boundary
     * CATCHES: s04, s06, m03
     * CHAPTER: ds.01 section 4 */
    tl_vec v;
    OL_EQ(tl_vec_init(&v, sizeof(uint16_t)), TL_OK);
    OL_TRUE(tl_vec_at(&v, 0) == NULL);
    uint16_t x = 7;
    OL_EQ(tl_vec_push(&v, &x), TL_OK);
    OL_TRUE(tl_vec_at(&v, 0) != NULL);
    tl_set_last_error("");
    OL_TRUE(tl_vec_at(&v, 1) == NULL);
    OL_TRUE(strlen(tl_last_error()) > 0);
    OL_TRUE(tl_vec_at(&v, SIZE_MAX) == NULL);
    tl_vec_free(&v);
}

OL_TEST(alloc_failure_leaves_vec_unchanged) {
    /* WHY: when the allocator hook fails during growth, push returns
     *      TL_ENOMEM and the vec is exactly as before (same data, len, cap,
     *      and contents) and nothing leaks; the next push, with memory back,
     *      succeeds. rt.04 relies on this to fail a request without
     *      corrupting a sequence's block table.
     * KIND: fault
     * CATCHES: s02, s03, s05, s06, s07, s09
     * CHAPTER: ds.01 section 2.4 */
    tl_vec v;
    OL_EQ(tl_vec_init(&v, sizeof(uint32_t)), TL_OK);
    for (uint32_t i = 0; i < 8; i++) OL_EQ(tl_vec_push(&v, &i), TL_OK);
    OL_EQ(v.len, v.cap); /* full: the next push must grow */
    tl_vec before = v;
    long live = ol_live_allocs();
    uint32_t x = 99;
    ol_alloc_fail_after(0);
    tl_set_last_error("");
    OL_EQ(tl_vec_push(&v, &x), TL_ENOMEM);
    OL_TRUE(strlen(tl_last_error()) > 0);
    OL_EQ(tl_vec_reserve(&v, 100), TL_ENOMEM);
    ol_alloc_fail_after(-1);
    OL_TRUE(v.data == before.data);
    OL_EQ(v.len, before.len);
    OL_EQ(v.cap, before.cap);
    OL_EQ(ol_live_allocs(), live);
    for (uint32_t i = 0; i < 8; i++) OL_EQ(*(uint32_t *)tl_vec_at(&v, i), i);
    OL_EQ(tl_vec_push(&v, &x), TL_OK);
    OL_EQ(*(uint32_t *)tl_vec_at(&v, 8), 99);
    tl_vec_free(&v);
}

OL_TEST(free_resets_and_vec_is_reusable) {
    /* WHY: tl_vec_free returns the block through the hook and puts the vec
     *      back in its init state (elem kept), so it can be reused; freeing an
     *      empty or already freed vec, or NULL, is harmless.
     * KIND: unit
     * CATCHES: s06, s07, s10
     * CHAPTER: ds.01 section 4 */
    long live = ol_live_allocs();
    tl_vec v;
    OL_EQ(tl_vec_init(&v, 8), TL_OK);
    tl_vec_free(&v);
    uint64_t x = 5;
    OL_EQ(tl_vec_push(&v, &x), TL_OK);
    OL_TRUE(ol_live_allocs() == live + 1);
    tl_vec_free(&v);
    OL_TRUE(v.data == NULL);
    OL_EQ(v.len, 0);
    OL_EQ(v.cap, 0);
    OL_EQ(v.elem, 8);
    OL_EQ(ol_live_allocs(), live);
    tl_vec_free(&v);
    tl_vec_free(NULL);
    OL_EQ(tl_vec_push(&v, &x), TL_OK);
    OL_EQ(*(uint64_t *)tl_vec_at(&v, 0), 5);
    tl_vec_free(&v);
}

OL_TEST(bad_arguments) {
    /* WHY: NULL pointers, elem 0, and a reservation whose byte size
     *      overflows size_t are TL_EINVAL with the error slot set, never a
     *      crash or a tiny allocation that later overflows.
     * KIND: boundary
     * CATCHES: s11, s12, s13, m02
     * CHAPTER: ds.01 section 4 */
    tl_vec v;
    OL_EQ(tl_vec_init(NULL, 4), TL_EINVAL);
    OL_EQ(tl_vec_init(&v, 0), TL_EINVAL);
    OL_EQ(tl_vec_init(&v, 4), TL_OK);
    uint32_t x = 1;
    OL_EQ(tl_vec_push(NULL, &x), TL_EINVAL);
    OL_EQ(tl_vec_push(&v, NULL), TL_EINVAL);
    OL_EQ(tl_vec_reserve(NULL, 4), TL_EINVAL);
    tl_set_last_error("");
    OL_EQ(tl_vec_reserve(&v, SIZE_MAX / 2), TL_EINVAL);
    OL_TRUE(strlen(tl_last_error()) > 0);
    OL_EQ(v.cap, 0);
    OL_TRUE(v.data == NULL);
    OL_EQ(tl_vec_reserve(&v, 0), TL_OK);
    tl_vec_free(&v);
}

/* The model: a plain array of what the vec should hold. */
static unsigned char model[65536 * 8];

static int random_ops_match_model(ol_gen *g, int size, size_t elem) {
    tl_vec v;
    if (tl_vec_init(&v, elem) != TL_OK) return 0;
    size_t n = 0;
    int ok = 1;
    for (int op = 0; op < size && ok; op++) {
        int64_t r = ol_gen_int(g, 0, 99);
        if (r < 70 && n < 65536) { /* push */
            unsigned char x[8];
            for (size_t b = 0; b < elem; b++) x[b] = (unsigned char)ol_gen_u32(g);
            ok = tl_vec_push(&v, x) == TL_OK;
            memcpy(model + n * elem, x, elem);
            n++;
        } else if (r < 90) { /* read */
            size_t i = (size_t)ol_gen_int(g, 0, (int64_t)n);
            unsigned char *p = tl_vec_at(&v, i);
            ok = i == n ? p == NULL : (p != NULL && memcmp(p, model + i * elem, elem) == 0);
        } else if (r < 99) { /* reserve */
            size_t want = (size_t)ol_gen_int(g, 0, (int64_t)n + 64);
            size_t cap = v.cap;
            ok = tl_vec_reserve(&v, want) == TL_OK && v.cap == (want > cap ? want : cap);
        } else { /* free and start over */
            tl_vec_free(&v);
            n = 0;
        }
        ok = ok && v.len == n && v.cap >= v.len && v.elem == elem;
    }
    for (size_t i = 0; i < n && ok; i++) ok = memcmp(tl_vec_at(&v, i), model + i * elem, elem) == 0;
    tl_vec_free(&v);
    return ok;
}

static int ops_u64(ol_gen *g, int size) { return random_ops_match_model(g, size, 8); }
static int ops_three(ol_gen *g, int size) { return random_ops_match_model(g, size, 3); }

OL_TEST(random_ops_match_a_model) {
    /* WHY: a million random pushes, reads (including one past the end),
     *      reservations, and frees, checked against a plain array after every
     *      step, for 8-byte and 3-byte elements: the invariants len == model
     *      length, cap >= len, and every element equal hold throughout, under
     *      ASan, with no leak.
     * KIND: property
     * CATCHES: s02, s04, s05, s06, s07, s08, s09, s10
     * CHAPTER: ds.01 section 2.5 */
    OL_CHECK_PROP(ops_u64, 20, 60000);
    OL_CHECK_PROP(ops_three, 20, 40000);
}

int main(void) { return OL_RUN_ALL(); }
