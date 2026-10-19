/* c/src/ds/lru.c (ds.03): least-recently-used order over intrusive nodes.
 * Contract: tinyllm/ds.h (ds.03 section).
 *
 * The policy is one rule: every use of an element moves its node to the
 * most recent end (head.prev); eviction takes the node at the oldest end
 * (head.next, tl_lru_pop_oldest in list.c). Both ends are O(1) because the
 * node is inside the element: finding it costs nothing, and moving it is
 * an unlink plus a link. rt.04 keeps its cached KV blocks on one tl_lru.
 */
#include <stddef.h>

#include "tinyllm/ds.h"

void tl_lru_touch(tl_lru *l, tl_list_node *n) {
    if (n->prev != NULL && n->next != NULL) tl_lru_remove(l, n); /* already in: unlink first */
    tl_list_node *last = l->head.prev; /* the current most recent (the sentinel when empty) */
    n->prev = last;
    n->next = &l->head;
    last->next = n;
    l->head.prev = n;
    l->len++;
}
