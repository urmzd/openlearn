/* Course tests for ds.03: the intrusive list and LRU of tinyllm/ds.h
 * (c/src/ds/list.c, c/src/ds/lru.c), built with ASan and UBSan, so a
 * pointer left dangling by a bad unlink stops the run with a report.
 *
 * Elements here are a test struct with an embedded tl_list_node, exactly
 * how rt.04 embeds one in each KV block record. The model for the
 * property test is an array of ids in LRU order (oldest first).
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_prop.h"
#include "ol_test.h"

typedef struct {
    int id;
    tl_list_node by_age; /* one list */
    tl_list_node by_size; /* a second list the same element sits on */
} item;

static int id_of(tl_list_node *n) { return TL_CONTAINER_OF(n, item, by_age)->id; }

/* The ids of l, oldest first, walked with the sentinel; returns the count. */
static int order(const tl_lru *l, int *out, int cap) {
    int k = 0;
    for (const tl_list_node *n = l->head.next; n != &l->head && k < cap; n = n->next)
        out[k++] = TL_CONTAINER_OF(n, item, by_age)->id;
    return k;
}

/* Both directions agree: walking prev from the sentinel visits the reverse. */
static int links_consistent(const tl_lru *l) {
    size_t fwd = 0, back = 0;
    for (const tl_list_node *n = l->head.next; n != &l->head; n = n->next) {
        if (n->next->prev != n) return 0;
        if (++fwd > l->len) return 0;
    }
    for (const tl_list_node *n = l->head.prev; n != &l->head; n = n->prev)
        if (++back > l->len) return 0;
    return fwd == l->len && back == l->len;
}

OL_TEST(hand_example_touch_order) {
    /* WHY: the chapter's worked example. Touch A, B, C (in that order),
     *      then touch A again: A moves to the most recent end, so the
     *      oldest-first order is B, C, A and pops come out B, C, A. This
     *      is the whole LRU contract on four operations.
     * KIND: unit, smoke
     * CATCHES: s01, s02, s04
     * CHAPTER: ds.03 section 3 */
    item it[3] = {{0}, {0}, {0}};
    for (int i = 0; i < 3; i++) it[i].id = 'A' + i;
    tl_lru l;
    tl_lru_init(&l);
    OL_EQ(l.len, 0);
    for (int i = 0; i < 3; i++) tl_lru_touch(&l, &it[i].by_age);
    tl_lru_touch(&l, &it[0].by_age);
    int got[4];
    OL_EQ(order(&l, got, 4), 3);
    OL_EQ(got[0], 'B');
    OL_EQ(got[1], 'C');
    OL_EQ(got[2], 'A');
    OL_EQ(l.len, 3);
    OL_TRUE(links_consistent(&l));
    OL_EQ(id_of(tl_lru_pop_oldest(&l)), 'B');
    OL_EQ(id_of(tl_lru_pop_oldest(&l)), 'C');
    OL_EQ(id_of(tl_lru_pop_oldest(&l)), 'A');
    OL_TRUE(tl_lru_pop_oldest(&l) == NULL);
    OL_EQ(l.len, 0);
}

OL_TEST(empty_list_pops_null) {
    /* WHY: rt.04 asks for the oldest cached block when its free stack runs
     *      out; an empty list must answer NULL, never the sentinel (which
     *      is not inside any element, so container_of on it points into
     *      the tl_lru itself).
     * KIND: boundary
     * CATCHES: s03, m02
     * CHAPTER: ds.03 section 5, Pitfalls */
    tl_lru l;
    tl_lru_init(&l);
    OL_TRUE(l.head.next == &l.head);
    OL_TRUE(l.head.prev == &l.head);
    OL_TRUE(tl_lru_pop_oldest(&l) == NULL);
    OL_TRUE(tl_lru_pop_oldest(&l) == NULL);
    OL_EQ(l.len, 0);
}

OL_TEST(touch_twice_keeps_one_link) {
    /* WHY: touching a node that is already in the list must move it, not
     *      link it a second time. Linking twice makes the list a cycle that
     *      skips the sentinel or loses the node's old neighbours, and len
     *      counts it twice.
     * KIND: unit
     * CATCHES: s01, s05, m01
     * CHAPTER: ds.03 section 2 */
    item a = {0}, b = {0};
    a.id = 1;
    b.id = 2;
    tl_lru l;
    tl_lru_init(&l);
    tl_lru_touch(&l, &a.by_age);
    tl_lru_touch(&l, &a.by_age);
    OL_EQ(l.len, 1);
    tl_lru_touch(&l, &b.by_age);
    tl_lru_touch(&l, &a.by_age);
    tl_lru_touch(&l, &a.by_age);
    OL_EQ(l.len, 2);
    OL_TRUE(links_consistent(&l));
    int got[3];
    OL_EQ(order(&l, got, 3), 2);
    OL_EQ(got[0], 2);
    OL_EQ(got[1], 1);
}

OL_TEST(remove_unlinks_and_marks_the_node) {
    /* WHY: removing a node leaves prev == next == NULL, the contract's mark
     *      for "in no list". A second remove is then a no-op, and a later
     *      touch links it fresh. rt.04 removes a block when a lookup revives
     *      it, and may never touch it again.
     * KIND: unit
     * CATCHES: s06, s07
     * CHAPTER: ds.03 section 4 */
    item it[3] = {{0}, {0}, {0}};
    tl_lru l;
    tl_lru_init(&l);
    for (int i = 0; i < 3; i++) {
        it[i].id = i;
        tl_lru_touch(&l, &it[i].by_age);
    }
    tl_lru_remove(&l, &it[1].by_age);
    OL_TRUE(it[1].by_age.prev == NULL);
    OL_TRUE(it[1].by_age.next == NULL);
    OL_EQ(l.len, 2);
    tl_lru_remove(&l, &it[1].by_age); /* not in the list: nothing happens */
    OL_EQ(l.len, 2);
    OL_TRUE(links_consistent(&l));
    int got[3];
    OL_EQ(order(&l, got, 3), 2);
    OL_EQ(got[0], 0);
    OL_EQ(got[1], 2);
    tl_lru_touch(&l, &it[1].by_age);
    OL_EQ(order(&l, got, 3), 3);
    OL_EQ(got[2], 1);
}

OL_TEST(remove_the_only_node_and_the_ends) {
    /* WHY: the sentinel makes the first, last, and only node ordinary: the
     *      same four writes unlink each. Code that special-cases the ends
     *      (or forgets the sentinel's own prev) breaks exactly here.
     * KIND: boundary
     * CATCHES: s06, s07
     * CHAPTER: ds.03 section 2 */
    item it[3] = {{0}, {0}, {0}};
    tl_lru l;
    tl_lru_init(&l);
    it[0].id = 7;
    tl_lru_touch(&l, &it[0].by_age);
    tl_lru_remove(&l, &it[0].by_age);
    OL_EQ(l.len, 0);
    OL_TRUE(l.head.next == &l.head && l.head.prev == &l.head);
    for (int i = 0; i < 3; i++) {
        it[i].id = i;
        tl_lru_touch(&l, &it[i].by_age);
    }
    tl_lru_remove(&l, &it[2].by_age); /* the most recent end */
    tl_lru_remove(&l, &it[0].by_age); /* the oldest end */
    OL_EQ(l.len, 1);
    OL_TRUE(links_consistent(&l));
    OL_TRUE(l.head.next == &it[1].by_age && l.head.prev == &it[1].by_age);
}

OL_TEST(safe_iteration_removes_while_walking) {
    /* WHY: rt.04 and the chapter's eviction sweep walk the list and unlink
     *      as they go. Remove clears the node's links, so the walk must save
     *      next BEFORE removing; this test walks with the saved successor
     *      and removes every even id, and the odd ones stay in order.
     * KIND: unit
     * CATCHES: s06
     * CHAPTER: ds.03 section 2 */
    item it[8];
    memset(it, 0, sizeof it);
    tl_lru l;
    tl_lru_init(&l);
    for (int i = 0; i < 8; i++) {
        it[i].id = i;
        tl_lru_touch(&l, &it[i].by_age);
    }
    for (tl_list_node *n = l.head.next, *next = n->next; n != &l.head; n = next, next = n->next)
        if (id_of(n) % 2 == 0) tl_lru_remove(&l, n);
    int got[8];
    OL_EQ(order(&l, got, 8), 4);
    for (int i = 0; i < 4; i++) OL_EQ(got[i], 2 * i + 1);
    OL_EQ(l.len, 4);
    OL_TRUE(links_consistent(&l));
}

OL_TEST(container_of_recovers_the_element_on_two_lists) {
    /* WHY: "intrusive" means the node lives inside your struct, so one
     *      element can sit on two lists through two nodes, and
     *      TL_CONTAINER_OF must subtract the offset of the right member.
     *      The two lists hold the same elements in different orders.
     * KIND: unit
     * CATCHES: s02
     * CHAPTER: ds.03 section 2 */
    item it[3] = {{0}, {0}, {0}};
    tl_lru age, size;
    tl_lru_init(&age);
    tl_lru_init(&size);
    for (int i = 0; i < 3; i++) {
        it[i].id = 10 + i;
        tl_lru_touch(&age, &it[i].by_age);
        tl_lru_touch(&size, &it[2 - i].by_size);
    }
    tl_list_node *a = tl_lru_pop_oldest(&age);
    tl_list_node *s = tl_lru_pop_oldest(&size);
    OL_EQ(TL_CONTAINER_OF(a, item, by_age)->id, 10);
    OL_EQ(TL_CONTAINER_OF(s, item, by_size)->id, 12);
    OL_EQ(age.len, 2);
    OL_EQ(size.len, 2);
    OL_TRUE(links_consistent(&age));
}

/* Model: ids in LRU order, oldest first; in[i] says whether id i is linked. */
static int lru_matches_model(ol_gen *g, int size) {
    enum { N = 32 };
    item it[N];
    memset(it, 0, sizeof it);
    for (int i = 0; i < N; i++) it[i].id = i;
    int model[N], m = 0;
    tl_lru l;
    tl_lru_init(&l);
    int steps = 50 + size * 20;
    for (int s = 0; s < steps; s++) {
        int op = (int)ol_gen_int(g, 0, 9), id = (int)ol_gen_int(g, 0, N - 1);
        if (op < 5) { /* touch: to the most recent end */
            int k = 0;
            for (int j = 0; j < m; j++)
                if (model[j] != id) model[k++] = model[j];
            model[k++] = id;
            m = k;
            tl_lru_touch(&l, &it[id].by_age);
        } else if (op < 8) { /* remove */
            int k = 0;
            for (int j = 0; j < m; j++)
                if (model[j] != id) model[k++] = model[j];
            m = k;
            tl_lru_remove(&l, &it[id].by_age);
        } else { /* pop the oldest */
            tl_list_node *n = tl_lru_pop_oldest(&l);
            if (m == 0) {
                if (n != NULL) return 0;
            } else {
                if (n == NULL || id_of(n) != model[0]) return 0;
                memmove(model, model + 1, sizeof(int) * (size_t)(m - 1));
                m--;
            }
        }
        int got[N];
        if (order(&l, got, N) != m || l.len != (size_t)m || !links_consistent(&l)) return 0;
        for (int j = 0; j < m; j++)
            if (got[j] != model[j]) return 0;
    }
    return 1;
}

OL_TEST(random_ops_match_a_model) {
    /* WHY: 200 seeded runs of random touch, remove, and pop over 32
     *      elements, compared after every step with an array kept in LRU
     *      order. Any slip in the four-pointer dance shows up as a wrong
     *      order, a wrong len, or a prev/next mismatch.
     * KIND: property
     * CATCHES: s01, s04, s05, s07, m01
     * CHAPTER: ds.03 section 4 */
    OL_CHECK_PROP(lru_matches_model, 200, 40);
}

int main(void) { return OL_RUN_ALL(); }
