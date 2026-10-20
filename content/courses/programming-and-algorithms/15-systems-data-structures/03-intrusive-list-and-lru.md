# An intrusive list and an LRU

The [Paged KV block pool](../../ai-engineering/21-inference/08-paged-kv-block-pool.md) in *AI Engineering* keeps blocks that no running request holds but whose contents may be reused: "cached" blocks. When the pool runs out of free blocks it must reclaim one, and the one least likely to be reused is the one used longest ago. A request that hits a cached block takes it back out of the reclaim order; a request that finishes puts its blocks in as the most recent. Those three events happen on every engine step, for thousands of blocks, so each must be $O(1)$ and must never allocate, because an allocation can fail in the middle of freeing memory.

A list whose links live inside each block's own record does all three in a few pointer writes. This section builds that list, the classic **intrusive list** with a sentinel head, and puts a least-recently-used (LRU) policy on top of it. There is nothing to call here, only pointers; the C you need is in [the C primer](../01-language-primers/03-c.md), and it helps to have read [the Swiss table](02-swiss-table.md) just before. The [practice C drill 03](../../../../practice/build/systems/c/03-intrusive-list/) builds the same list with a splice operation.

The interface is the `tl_list_node`, `TL_CONTAINER_OF`, and `tl_lru` part of [`tinyllm/ds.h`](../../../../course/contracts/c/include/tinyllm/ds.h). The solution is split in two: the primitives in [`list.c`](../../../../course/solutions/c/src/ds/list.c) and the policy in [`lru.c`](../../../../course/solutions/c/src/ds/lru.c). As you read, draw the four nodes of the worked example on paper with their `prev` and `next` arrows; every bug this section guards against is an arrow drawn in the wrong order.

> **Learning objectives**
> - Embed list links in an element and recover the element from its link with `TL_CONTAINER_OF`.
> - Use a sentinel head so that link and unlink are the same pointer writes at every position.
> - Implement an LRU as "touch to the newest end, pop the oldest end," both $O(1)$ and allocation-free.
> - Make remove idempotent and touch safe on a linked node by marking unlinked nodes with NULL.
> - Order pointer writes so that no neighbour ever receives a NULL link.

## Links inside the element

The vocabulary is small:

| Symbol | Meaning | Type |
|---|---|---|
| node | a `tl_list_node {prev, next}` embedded in the element's struct | `tl_list_node` |
| head | the sentinel node inside `tl_lru`; holds no element | `tl_list_node` |
| `len` | number of linked nodes, the sentinel excluded | `size_t` |

A textbook list allocates a wrapper per element that points at the element. An intrusive list puts the node in the element (`struct block { ...; tl_list_node lru; }`) and links the nodes directly. Linking cannot fail, and one element can be on several lists through several nodes. The price is that the list hands you a `tl_list_node *`, and you get your struct back with `TL_CONTAINER_OF(ptr, type, member)`, which subtracts `offsetof(type, member)` from the node's address. Nothing extra is stored to make this work; the offset is a compile-time constant.

## A sentinel removes the special cases

Ends and the empty list are where linked-list code breaks. `tl_lru` holds a node `head` that is part of the circular list, and an empty list is `head.next == head.prev == &head`. With the sentinel, the oldest node is `head.next`, the most recent is `head.prev`, and there is no NULL to check at either end. Linking `n` after `a` (whose next is `b`) is

```text
n.prev = a; n.next = b; a.next = n; b.prev = n
```

and unlinking `n` is `n.prev.next = n.next; n.next.prev = n.prev`, wherever `n` is: first, last, or only node.

### Unlinked means NULL

The sentinel handles positions; the next problem is state. `touch` must be able to move a node rather than link it twice, and `remove` should be harmless on a node that is not in the list. Both need to answer "is this node linked?" without a walk. The node's own pointers carry the answer: after an unlink, `tl_lru_remove` sets the node's `prev` and `next` to NULL. Removing an unlinked node then does nothing, and touching a linked node unlinks it first. Elements must start zeroed so that a fresh node reads as unlinked.

### The order of writes

Read `n.prev` and `n.next` before overwriting them. Clearing `n`'s links first and then relinking its neighbours writes NULL into the list. Walking a list while removing from it needs the same care: save `next` before removing the current node, because remove clears it.

## The LRU policy

With the list in place, the policy is two rules. `tl_lru_touch(l, n)` makes `n` the most recent, unlinking it first if it is linked. `tl_lru_pop_oldest(l)` unlinks and returns `head.next`, or NULL for an empty list. `list.c` holds the primitives (`init`, `remove`, `pop_oldest`) and `lru.c` the policy (`touch`).

## Working it by hand

Touch A, then B, then C, then A again, and then pop until empty. The list reads from `head.next`, oldest first.

| Step | Order, oldest first | `len` | Writes |
|---|---|---|---|
| init | (empty) | 0 | `head.prev = head.next = &head` |
| touch A | A | 1 | A linked between `head.prev` (the sentinel) and the sentinel |
| touch B | A B | 2 | B linked after A |
| touch C | A B C | 3 | C linked after B |
| touch A | B C A | 3 | A unlinked (C stays after B, B is now `head.next`), then linked after C |
| pop | C A, returns B | 2 | `head.next` becomes C |
| pop, pop | (empty), returns C then A | 0 | |
| pop | returns NULL | 0 | the sentinel is never returned |

## The interface

```c
/* tinyllm/ds.h (ds.03) */
typedef struct tl_list_node { struct tl_list_node *prev, *next; } tl_list_node;
#define TL_CONTAINER_OF(ptr, type, member) ((type *)(void *)((char *)(ptr) - offsetof(type, member)))
typedef struct { tl_list_node head; size_t len; } tl_lru;
void          tl_lru_init(tl_lru *l);
void          tl_lru_touch(tl_lru *l, tl_list_node *n);    /* insert, or move to most recent */
void          tl_lru_remove(tl_lru *l, tl_list_node *n);   /* unlink; prev = next = NULL; no-op if unlinked */
tl_list_node *tl_lru_pop_oldest(tl_lru *l);                /* NULL when empty */
```

The header has no list operations besides these four (no splice), and `tl_lru` is not thread-safe: the pool serializes every call.

## Building it step by step

Five observations produce the whole design.

1. **Each operation must be $O(1)$ and must not allocate.** $O(1)$ removal of an arbitrary element needs a doubly linked list (a singly linked one needs the predecessor, which costs a walk). No allocation means the links cannot live in a separately allocated wrapper: they go inside the element. That is the intrusive list.
2. **The list hands back a node, the caller wants the element.** The node is at a fixed offset inside the element, so subtracting `offsetof(type, member)` from the node's address gives the element: `TL_CONTAINER_OF`.
3. **Ends and the empty list are where linked-list code breaks.** A sentinel node that is always in the circle removes them: every real node has a real `prev` and `next`, so link and unlink are the same pointer writes for the first, last, and only node. Empty is "the sentinel points at itself".
4. **"Is this node linked?" must be answerable without a walk,** so that `touch` can move rather than double-link and `remove` can be idempotent. NULL means unlinked. That forces two rules: `remove` writes NULL after unlinking, and elements start zeroed.
5. **The LRU policy then needs one choice of direction.** The solution uses `head.next` as oldest and `head.prev` as newest. `touch` links at `head.prev`; `pop_oldest` unlinks `head.next`. In the worked example, touching A again unlinks it from between the sentinel and B and relinks it after C, so B becomes `head.next` and is the first pop.

### The primitives

`tl_lru_remove` reads `n->prev` and `n->next` while they are still valid, and only then marks `n` unlinked; the opposite order would write NULL into a neighbour.

```c
void tl_lru_init(tl_lru *l) {
    l->head.prev = &l->head; /* empty: the sentinel points at itself */
    l->head.next = &l->head;
    l->len = 0;
}

void tl_lru_remove(tl_lru *l, tl_list_node *n) {
    if (n == NULL || n->prev == NULL || n->next == NULL) return; /* in no list: a no-op */
    n->prev->next = n->next; /* the neighbours now skip n (works at either end: */
    n->next->prev = n->prev; /*  one neighbour may be the sentinel)              */
    n->prev = NULL;          /* mark n as unlinked, so a second remove is harmless */
    n->next = NULL;
    l->len--;                /* only reached for a linked node, so len stays exact */
}

tl_list_node *tl_lru_pop_oldest(tl_lru *l) {
    if (l->head.next == &l->head) return NULL; /* only the sentinel: empty; never return it */
    tl_list_node *n = l->head.next;            /* the oldest end */
    tl_lru_remove(l, n);                       /* reuse remove: one place updates len and marks n */
    return n;
}
```

### The policy

`lru.c` is one function. Unlinking through `tl_lru_remove` (not by hand) keeps `len` right and leaves `n` in the clean unlinked state that the four link writes below assume.

```c
void tl_lru_touch(tl_lru *l, tl_list_node *n) {
    if (n->prev != NULL && n->next != NULL) tl_lru_remove(l, n); /* already in: unlink first */
    tl_list_node *last = l->head.prev; /* the current most recent (the sentinel when empty) */
    n->prev = last;                    /* n's own links first: they point into the list */
    n->next = &l->head;
    last->next = n;                    /* then the two neighbours point at n */
    l->head.prev = n;
    l->len++;
}
```

The full solution is [`list.c`](../../../../course/solutions/c/src/ds/list.c) and [`lru.c`](../../../../course/solutions/c/src/ds/lru.c).

### Why this design

Every operation is $O(1)$ time and touches at most three nodes; memory is two pointers per element, already part of the element, plus one sentinel.

The alternatives each lose on something the pool cares about. A wrapper-allocating list would allocate on every touch, so an LRU update could fail with `TL_ENOMEM` while the pool is trying to free memory, and it would cost an extra pointer chase per access. An array or heap of timestamps would make eviction $O(\log n)$ (heap) or $O(n)$ (scan), and every touch would have to update a priority; the list keeps recency implicitly by position. NULL-terminated ends instead of a sentinel would need branches for "first", "last", and "only" node in each of link, unlink, and pop, which is exactly where pointer bugs hide.

The design accepts two obligations on the caller: elements must be zeroed before first use, and one node must not be on two lists at once (one element on two lists uses two nodes). The list is not thread-safe; the KV pool serializes every call.

## Testing it and what can go wrong

A correct solution handles the following cases, each tied to something the pool does.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example's touch order | unit | order B C A, pops B, C, A, then NULL | the policy is exactly "touch to the newest end, pop the oldest" |
| An empty list pops NULL | boundary | a fresh list points at itself and pops NULL twice | the pool asks for a victim when nothing is cached |
| Touching twice keeps one link | unit | re-touching moves a node; `len` counts it once | a block hit twice in a row |
| Remove unlinks and marks the node | unit | remove clears the links; a second remove is a no-op; a later touch relinks | a revived cached block leaves the list |
| Removing the only node and the ends | boundary | unlinking the only node, the newest, and the oldest | the sentinel removes every special case |
| Removing while walking | unit | removing every even id while walking with a saved successor | sweeping a list while evicting from it |
| `TL_CONTAINER_OF` on two lists | unit | one struct on two lists through two nodes, recovered from each | the element is found from its node |
| Random operations match a model | property | 200 seeded runs of touch, remove, pop over 32 elements against an array model, with link consistency after every step | every interleaving the pool can produce |

A model test suite (touch order by hand, pop of an empty list, re-touch moves, double remove is harmless, walk and remove) is in [`test_lru.c`](../../../../course/solutions/learner-tests/ds.03/c/tests/ds03-lru/test_lru.c).

The pitfalls are all arrows in the wrong place or the wrong order:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Linking a node that is already linked | its old neighbours keep pointing at it; the walk loops or skips the sentinel | `tl_lru_touch` calls `tl_lru_remove` first when the node is linked |
| Linking at the wrong end | the most recently used block is evicted first | `touch` links at `head.prev`, `pop_oldest` takes `head.next` |
| Returning the sentinel from an empty list | `TL_CONTAINER_OF` turns the `tl_lru` itself into a "block" | `pop_oldest` returns NULL when `head.next == &head` |
| Popping `head.prev` | evicts the newest block | `pop_oldest` removes `head.next` |
| Unlinking by hand without updating `len` | `len` grows on every re-touch | `touch` unlinks through `tl_lru_remove`, the one place that decrements `len` |
| Clearing the node's links before relinking its neighbours | NULL written into the list; the next walk crashes | `remove` rewires the neighbours first, then writes NULL |
| Not clearing the links on remove | a second remove decrements `len` again and corrupts the list | `remove` writes NULL and returns early for an unlinked node |

> **Warning:** When you walk the list and remove as you go, save `next` before calling `tl_lru_remove`. Remove sets the current node's links to NULL, so reading `cur->next` afterwards ends the walk with a crash.

## Where this leads

In the [Paged KV block pool](../../ai-engineering/21-inference/08-paged-kv-block-pool.md), every cached block's record embeds a `tl_list_node`: unref of a registered block touches it, a lookup hit removes it, and allocation under pressure pops the oldest. The [radix tree](07-radix-tree.md) later in this chapter keeps an LRU list of leaves on the same idea in safe Rust, with indices instead of pointers.

Production systems use the same structure, often with additions:

- Linux `list_head` adds `list_for_each_entry_safe`, splice, and RCU-safe variants; see [`include/linux/list.h`](https://github.com/torvalds/linux/blob/master/include/linux/list.h).
- vLLM's `FreeKVCacheBlockQueue` is a doubly linked free queue of KV blocks where cached blocks are reclaimed oldest first; see [`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py).
- The Linux page cache keeps two lists (active and inactive) so a page read once does not evict a page read often; see `mm/vmscan.c`.

### Further reading

- Linux kernel, [`include/linux/list.h`](https://github.com/torvalds/linux/blob/master/include/linux/list.h).
- Cormen et al., *Introduction to Algorithms*, section 10.2 (linked lists with sentinels).

## Key terms

- **Intrusive list**: a linked list whose link fields live inside the elements, so linking needs no allocation.
- **`TL_CONTAINER_OF`**: a macro that recovers the enclosing struct from a pointer to one of its members by subtracting the member's offset.
- **Sentinel**: a permanent dummy node in a circular list that makes every real node have real neighbours.
- **LRU (least recently used)**: an eviction policy that reclaims the element whose last use is oldest.
- **Touch**: moving a node to the most recent end, linking it if it was not linked.

## Summary

- An intrusive list stores its links inside the element's own struct, so linking costs no allocation and cannot fail, and `TL_CONTAINER_OF` recovers the struct from the link.
- A sentinel head makes the first, last, and only node ordinary: every link and unlink is the same pointer writes.
- An LRU is two rules: a use moves the node to the most recent end, and eviction takes the oldest end; both are $O(1)$.
- An unlinked node has `prev == next == NULL`, so removing twice is harmless and touching an already-linked node moves it instead of linking it twice.
- Rewire the neighbours before clearing a node's own links.

## Review questions

1. Why does `tl_lru_touch` check whether `n` is linked before linking it?
   <details><summary>Answer</summary>Linking an already linked node would leave its old neighbours pointing at it while it sits somewhere else: the circle now has a shortcut or a loop, a walk can skip the sentinel or never end, and `len` counts the node twice. Unlinking first makes "touch" mean "move to the newest end" whether or not the node was in the list.</details>
2. What goes wrong if `tl_lru_remove` sets `n->prev = n->next = NULL` before updating the neighbours?
   <details><summary>Answer</summary>`n->prev->next = n->next` then dereferences NULL (a crash), or, if written through saved copies in the wrong order, writes NULL into a neighbour's link, so the next walk from the sentinel falls off the list.</details>
3. Why must `tl_lru_pop_oldest` test for an empty list instead of simply removing `head.next`?
   <details><summary>Answer</summary>When the list is empty, `head.next` is the sentinel itself. Returning it would make `TL_CONTAINER_OF` compute a "block" from the address of the `tl_lru`, and removing it would break the circle. NULL is the only correct answer to "no victim".</details>
4. How does one element sit on two lists at the same time?
   <details><summary>Answer</summary>It embeds two `tl_list_node` members, one per list. Each list links only its own member, and `TL_CONTAINER_OF(ptr, type, member)` with the matching member name recovers the same element from either node.</details>
5. Why would a list that allocates a wrapper per element be dangerous in the KV pool?
   <details><summary>Answer</summary>Every touch would allocate, so updating the LRU could fail with `TL_ENOMEM` exactly when the pool is under memory pressure and trying to reclaim blocks, and every access would chase an extra pointer.</details>
