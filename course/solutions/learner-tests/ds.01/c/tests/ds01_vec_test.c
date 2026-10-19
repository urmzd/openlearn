/* My tests for ds.01 (rung R4): properties of tl_vec checked with ol_prop.h
 * against a plain array model, plus the boundary and fault cases the
 * properties cannot reach. Only the contract headers are included. */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

static unsigned char model[40000 * 5];

static int matches_model(ol_gen *g, int size) {
    size_t elem = (size_t)ol_gen_int(g, 1, 5);
    tl_vec v;
    if (tl_vec_init(&v, elem) != TL_OK) return 0;
    size_t n = 0, moves = 0, cap = 0;
    for (int op = 0; op < size; op++) {
        int64_t r = ol_gen_int(g, 0, 9);
        if (r < 7) {
            unsigned char x[5];
            for (size_t b = 0; b < elem; b++) x[b] = (unsigned char)ol_gen_u32(g);
            if (tl_vec_push(&v, x) != TL_OK) return 0;
            memcpy(model + n++ * elem, x, elem);
        } else if (r < 9) {
            size_t want = (size_t)ol_gen_int(g, 0, (int64_t)n + 9);
            size_t before = v.cap;
            if (tl_vec_reserve(&v, want) != TL_OK || v.cap != (want > before ? want : before)) return 0;
        } else if (tl_vec_at(&v, n) != NULL) {
            return 0;
        }
        if (v.cap != cap) {
            moves++;
            cap = v.cap;
        }
        if (v.len != n || v.cap < v.len) return 0;
    }
    for (size_t i = 0; i < n; i++)
        if (memcmp(tl_vec_at(&v, i), model + i * elem, elem) != 0) return 0;
    tl_vec_free(&v);
    return v.len == 0 && v.cap == 0 && v.data == NULL && moves <= 64;
}

OL_TEST(vec_matches_array_model) { OL_CHECK_PROP(matches_model, 30, 40000); }

OL_TEST(pushes_double_capacity) {
    tl_vec v;
    OL_EQ(tl_vec_init(&v, 4), TL_OK);
    size_t prev = 0;
    for (uint32_t i = 0; i < 5000; i++) {
        OL_EQ(tl_vec_push(&v, &i), TL_OK);
        if (v.cap != prev) {
            OL_TRUE(prev == 0 ? v.cap >= 4 : v.cap >= 2 * prev);
            prev = v.cap;
        }
    }
    uint32_t first = 0;
    OL_EQ(*(uint32_t *)tl_vec_at(&v, 4999), 4999);
    tl_vec_free(&v);
    OL_EQ(tl_vec_push(&v, &first), TL_OK);
    OL_EQ(v.cap, 4);
    tl_vec_free(&v);
}

OL_TEST(failed_growth_changes_nothing) {
    tl_vec v;
    OL_EQ(tl_vec_init(&v, 8), TL_OK);
    for (uint64_t i = 0; i < 4; i++) OL_EQ(tl_vec_push(&v, &i), TL_OK);
    tl_vec saved = v;
    long live = ol_live_allocs();
    uint64_t x = 9;
    ol_alloc_fail_after(0);
    OL_EQ(tl_vec_push(&v, &x), TL_ENOMEM);
    ol_alloc_fail_after(-1);
    OL_TRUE(v.data == saved.data && v.len == saved.len && v.cap == saved.cap);
    OL_EQ(ol_live_allocs(), live);
    for (uint64_t i = 0; i < 4; i++) OL_EQ(*(uint64_t *)tl_vec_at(&v, i), i);
    tl_vec_free(&v);
}

OL_TEST(rejects_bad_input) {
    tl_vec v;
    uint32_t x = 3;
    OL_EQ(tl_vec_init(&v, 0), TL_EINVAL);
    OL_EQ(tl_vec_init(&v, 4), TL_OK);
    OL_EQ(tl_vec_push(&v, NULL), TL_EINVAL);
    OL_EQ(tl_vec_reserve(NULL, 1), TL_EINVAL);
    OL_EQ(tl_vec_reserve(&v, SIZE_MAX / 3), TL_EINVAL);
    OL_EQ(tl_vec_push(&v, &x), TL_OK);
    tl_set_last_error("");
    OL_TRUE(tl_vec_at(&v, 1) == NULL && tl_last_error()[0] != '\0');
    tl_vec_free(&v);
}

int main(void) { return OL_RUN_ALL(); }
