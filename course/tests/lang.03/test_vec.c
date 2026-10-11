/* Course tests for the lang.03 dynamic array: YOUR primers/lang.03/vec.c,
 * built with -std=c11 -pedantic -Werror under ASan and UBSan.
 *
 * vec.c's malloc, calloc, realloc, and free are routed through the counting
 * wrappers below (count_alloc.h), so every test also checks that nothing
 * leaked and can make one allocation fail on purpose.
 */
#include <stdint.h>
#include <stdlib.h>

#include "ol_test.h"
#include "vec.h"

static long live_blocks;   /* blocks handed out and not yet freed */
static long reallocs;      /* calls to realloc that returned memory */
static long fail_next = 0; /* 1: the next malloc/calloc/realloc returns NULL */

static int should_fail(void) {
    if (fail_next) {
        fail_next = 0;
        return 1;
    }
    return 0;
}

void *lang03_malloc(size_t n) {
    if (should_fail()) return NULL;
    void *p = malloc(n);
    if (p) live_blocks++;
    return p;
}

void *lang03_calloc(size_t n, size_t size) {
    if (should_fail()) return NULL;
    void *p = calloc(n, size);
    if (p) live_blocks++;
    return p;
}

void *lang03_realloc(void *p, size_t n) {
    if (should_fail()) return NULL;
    void *q = realloc(p, n);
    if (q) {
        reallocs++;
        if (p == NULL) live_blocks++;
    }
    return q;
}

void lang03_free(void *p) {
    if (p) live_blocks--;
    free(p);
}

/* Every test starts from zero and must end with nothing live. */
static void reset_counts(void) {
    live_blocks = 0;
    reallocs = 0;
    fail_next = 0;
}

OL_TEST(init_allocates_nothing) {
    /* WHY: a vector you never push to must never call malloc: an engine
     *      creates thousands of empty per-request buffers, and each
     *      allocation costs time and memory.
     * KIND: unit */
    reset_counts();
    Vec v;
    vec_init(&v);
    OL_EQ(v.len, 0);
    OL_EQ(v.cap, 0);
    OL_TRUE(v.data == NULL);
    OL_EQ(live_blocks, 0);
    vec_free(&v);
}

OL_TEST(push_and_get_by_hand) {
    /* WHY: the chapter's worked example: three pushes grow the capacity once,
     *      from 0 to 4, and the elements read back in order.
     * KIND: unit */
    reset_counts();
    Vec v;
    vec_init(&v);
    OL_EQ(vec_push(&v, 10), 0);
    OL_EQ(vec_push(&v, 20), 0);
    OL_EQ(vec_push(&v, 30), 0);
    OL_EQ(v.len, 3);
    OL_TRUE(v.cap >= 3);
    OL_EQ(vec_get(&v, 0), 10);
    OL_EQ(vec_get(&v, 1), 20);
    OL_EQ(vec_get(&v, 2), 30);
    vec_free(&v);
    OL_EQ(live_blocks, 0);
}

OL_TEST(pop_is_lifo) {
    /* WHY: pop returns the last element pushed and refuses an empty vector
     *      with -1 instead of reading index -1 (which, as size_t, is huge).
     * KIND: boundary */
    reset_counts();
    Vec v;
    int out = 0;
    vec_init(&v);
    OL_EQ(vec_pop(&v, &out), -1);
    vec_push(&v, 1);
    vec_push(&v, 2);
    OL_EQ(vec_pop(&v, &out), 0);
    OL_EQ(out, 2);
    OL_EQ(vec_pop(&v, &out), 0);
    OL_EQ(out, 1);
    OL_EQ(vec_pop(&v, &out), -1);
    OL_EQ(v.len, 0);
    vec_free(&v);
    OL_EQ(live_blocks, 0);
}

OL_TEST(reserve_does_not_change_len) {
    /* WHY: reserve allocates ahead without adding elements, and never
     *      shrinks: asking for less than the capacity is a no-op.
     * KIND: unit */
    reset_counts();
    Vec v;
    vec_init(&v);
    OL_EQ(vec_reserve(&v, 100), 0);
    OL_TRUE(v.cap >= 100);
    OL_EQ(v.len, 0);
    OL_EQ(vec_reserve(&v, 10), 0);
    OL_TRUE(v.cap >= 100);
    vec_free(&v);
    OL_EQ(live_blocks, 0);
}

OL_TEST(growth_preserves_every_element) {
    /* WHY: realloc may move the buffer to a new address and copies the old
     *      bytes there. Tracking len or cap wrong loses or corrupts elements
     *      only once the buffer has moved several times; ASan reports any
     *      write past the end.
     * KIND: property */
    reset_counts();
    enum { N = 10000 };
    Vec v;
    vec_init(&v);
    for (int i = 0; i < N; i++) OL_EQ(vec_push(&v, i * 3), 0);
    OL_EQ(v.len, N);
    OL_TRUE(v.cap >= v.len);
    for (int i = 0; i < N; i++) {
        if (vec_get(&v, (size_t)i) != i * 3) OL_FAIL("an element did not survive reallocation");
    }
    vec_free(&v);
    OL_EQ(live_blocks, 0);
}

OL_TEST(growth_is_geometric) {
    /* WHY: growing by a constant makes n pushes cost O(n^2) copying, which
     *      you only feel at a million elements. Doubling from 4 reaches 4096
     *      in 11 reallocations; 40 allows any growth factor of 1.25 or more.
     * KIND: property */
    reset_counts();
    enum { N = 4096 };
    Vec v;
    vec_init(&v);
    for (int i = 0; i < N; i++) vec_push(&v, i);
    OL_TRUE(v.cap < (size_t)N * 4);
    OL_TRUE(reallocs <= 40);
    vec_free(&v);
    OL_EQ(live_blocks, 0);
}

OL_TEST(failed_growth_leaves_the_vec_unchanged) {
    /* WHY: `v->data = realloc(v->data, ...)` is the classic bug: when realloc
     *      returns NULL, the only pointer to the old buffer is overwritten, so
     *      the elements are lost and the buffer leaks. On failure push must
     *      return -1 and leave data, len, and cap exactly as they were.
     * KIND: fault */
    reset_counts();
    Vec v;
    vec_init(&v);
    for (int i = 0; i < 64 && (v.len < 4 || v.len < v.cap); i++) vec_push(&v, (int)v.len); /* fill to capacity */
    OL_TRUE(v.len >= 4 && v.len == v.cap);
    int *data = v.data;
    size_t len = v.len, cap = v.cap;
    fail_next = 1;
    OL_EQ(vec_push(&v, 99), -1);
    OL_TRUE(v.data == data);
    OL_EQ(v.len, len);
    OL_EQ(v.cap, cap);
    for (size_t i = 0; i < len; i++) OL_EQ(vec_get(&v, i), (int)i);
    OL_EQ(vec_push(&v, 99), 0); /* the allocator works again */
    vec_free(&v);
    OL_EQ(live_blocks, 0);
}

OL_TEST(free_is_idempotent) {
    /* WHY: cleanup code often runs twice (an error path, then the normal
     *      exit). vec_free must leave the zero state so a second call frees
     *      nothing, and the vector can be reused after it.
     * KIND: boundary */
    reset_counts();
    Vec v;
    vec_init(&v);
    vec_push(&v, 1);
    vec_free(&v);
    OL_TRUE(v.data == NULL);
    OL_EQ(v.len, 0);
    OL_EQ(v.cap, 0);
    vec_free(&v); /* ASan: no double free */
    OL_EQ(vec_push(&v, 7), 0);
    OL_EQ(vec_get(&v, 0), 7);
    vec_free(&v);
    OL_EQ(live_blocks, 0);
}

int main(void) { return OL_RUN_ALL(); }
