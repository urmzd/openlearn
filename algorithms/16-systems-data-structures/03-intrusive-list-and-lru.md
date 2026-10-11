<!-- ol:module ds.03 -->
# Intrusive list + LRU (optional C)

## Overview

| | |
|---|---|
| **Module** | `ds.03` · side · C · Pass 6 · 2 to 3 h |
| **You build** | `c/src/ds/list.c`: `tl_lru_init`, `tl_lru_remove`, `tl_lru_pop_oldest` · `c/src/ds/lru.c`: `tl_lru_touch` |
| **Contract** | the ds.03 section of [`tinyllm/ds.h`](../../course/contracts/c/include/tinyllm/ds.h) (`tl_list_node`, `TL_CONTAINER_OF`, `tl_lru`) |
| **Tests** | `course/tests/ds.03/test_lru.c`, 8 tests under ASan and UBSan (what they check: section 4) · your own tests in `c/tests/ds03-lru/`, rung R2, graded by mutation (threshold 0.60) |
| **Needs** | nothing to call: pointers only · reading: `ds.02` (the chapter before), the [practice C drill 03](../../practice/build/systems/c/03-intrusive-list/) (the same list with splice) |
| **Used by** | `rt.04` keeps its cached KV blocks on one `tl_lru`, oldest first |
| **Milestone** | `MS-L9`, the optional standalone C module group |
| **Optional depth** | Linux kernel, [`include/linux/list.h`](https://github.com/torvalds/linux/blob/master/include/linux/list.h); Cormen et al., *Introduction to Algorithms*, section 10.2 (linked lists with sentinels) |

## Key Takeaways

- An intrusive list stores its links inside your own struct, so linking costs no allocation and cannot fail, and `TL_CONTAINER_OF` recovers the struct from the link (`container_of_recovers_the_element_on_two_lists`).
- A sentinel head makes the first, last, and only node ordinary: every link and unlink is the same four pointer writes (`remove_the_only_node_and_the_ends`).
- An LRU is two rules: a use moves the node to the most recent end, and eviction takes the oldest end; both are O(1) (`hand_example_touch_order`).
- An unlinked node has `prev == next == NULL`, so removing twice is harmless and touching an already-linked node moves it instead of linking it twice (`remove_unlinks_and_marks_the_node`, `touch_twice_keeps_one_link`).

## How to work this chapter

```bash
ol start ds.03              # stubs c/src/ds/list.c and c/src/ds/lru.c
ol tests ds.03              # read the test catalog first
ol check ds.03              # exit code is the verdict; then grades your tests by mutation
ol diff  ds.03              # after passing: your code against the reference
```

Draw the four nodes of section 3 on paper with their `prev` and `next` arrows before writing a line; every bug in this module is an arrow drawn in the wrong order.

---

## 1. Why now

The KV block pool (`rt.04`) keeps blocks that no running request holds but whose contents may be reused: "cached" blocks. When the pool runs out of free blocks it must reclaim one, and the one least likely to be reused is the one used longest ago. A request that hits a cached block takes it back out of the reclaim order; a request that finishes puts its blocks in as the most recent. Those three events happen on every engine step, for thousands of blocks, so each must be O(1) and must never allocate (an allocation can fail in the middle of freeing memory). A list whose links live inside each block's own record does all three in a few pointer writes.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| node | a `tl_list_node {prev, next}` embedded in your struct | `tl_list_node` |
| head | the sentinel node inside `tl_lru`; holds no element | `tl_list_node` |
| `len` | number of linked nodes, the sentinel excluded | `size_t` |

**Intrusive.** A textbook list allocates a wrapper per element that points at the element. An intrusive list puts the node in the element (`struct block { ...; tl_list_node lru; }`) and links the nodes directly. Linking cannot fail, and one element can be on several lists through several nodes. The price: the list hands you a `tl_list_node *`, and you get your struct back with `TL_CONTAINER_OF(ptr, type, member)`, which subtracts `offsetof(type, member)` from the node's address.

**Sentinel.** `tl_lru` holds a node `head` that is part of the circular list. An empty list is `head.next == head.prev == &head`. With the sentinel, the oldest node is `head.next`, the most recent is `head.prev`, and there is no NULL to check at either end: linking `n` after `a` (whose next is `b`) is `n.prev = a; n.next = b; a.next = n; b.prev = n`, and unlinking `n` is `n.prev.next = n.next; n.next.prev = n.prev`, wherever `n` is.

**Unlinked means NULL.** After an unlink, `tl_lru_remove` sets the node's `prev` and `next` to NULL. That mark lets remove and touch tell a linked node from an unlinked one: removing an unlinked node does nothing, and touching a linked node unlinks it first. Elements must start zeroed so a fresh node reads as unlinked.

**The order of writes matters.** Read `n.prev` and `n.next` before overwriting them: clearing `n`'s links first and then relinking its neighbours writes NULL into the list. Walking a list while removing from it needs the same care: save `next` before removing the current node, because remove clears it.

**LRU policy.** `tl_lru_touch(l, n)` makes `n` the most recent (unlinking it first if it is linked); `tl_lru_pop_oldest(l)` unlinks and returns `head.next`, or NULL for an empty list. `list.c` holds the primitives (`init`, `remove`, `pop_oldest`), `lru.c` the policy (`touch`).

## 3. Worked example by hand

Touch A, then B, then C, then A again (`.` is the sentinel; the list reads from `head.next`):

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

This is `hand_example_touch_order`.

## 4. The interface

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_touch_order` | unit | section 3: order B C A, pops B, C, A, then NULL | you and the tests agree on the policy |
| `empty_list_pops_null` | boundary | a fresh list points at itself and pops NULL twice | rt.04 asks for a victim when nothing is cached |
| `touch_twice_keeps_one_link` | unit | re-touching moves a node; `len` counts it once | a block hit twice in a row |
| `remove_unlinks_and_marks_the_node` | unit | remove clears the links; a second remove is a no-op; a later touch relinks | a revived cached block leaves the list |
| `remove_the_only_node_and_the_ends` | boundary | unlinking the only node, the newest, and the oldest | the sentinel removes every special case |
| `safe_iteration_removes_while_walking` | unit | removing every even id while walking with a saved successor | sweeping a list while evicting from it |
| `container_of_recovers_the_element_on_two_lists` | unit | one struct on two lists through two nodes, recovered from each | the element is found from its node |
| `random_ops_match_a_model` | property | 200 seeded runs of touch, remove, pop over 32 elements against an array model, with link consistency after every step | every interleaving rt.04 can produce |

**Your tests (rung R2).** Write C files under `c/tests/ds03-lru/` with these tests, bodies yours: `touch_order_by_hand`, `pop_empty_is_null`, `retouch_moves`, `double_remove_is_harmless`, `walk_and_remove`. At least 60% of the planted bugs must make one of them fail.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Linking a node that is already linked | its old neighbours keep pointing at it; the walk loops or skips the sentinel | `touch_twice_keeps_one_link` (mutant `s01`) |
| Linking at the wrong end | the most recently used block is evicted first | `hand_example_touch_order`, `container_of_recovers_the_element_on_two_lists` (mutant `s02`) |
| Returning the sentinel from an empty list | `TL_CONTAINER_OF` turns the `tl_lru` itself into a "block" | `empty_list_pops_null` (mutant `s03`) |
| Popping `head.prev` | evicts the newest block | `hand_example_touch_order` (mutant `s04`) |
| Unlinking by hand without updating `len` | `len` grows on every re-touch | `touch_twice_keeps_one_link` (mutant `s05`) |
| Clearing the node's links before relinking its neighbours | NULL written into the list; the next walk crashes | `remove_the_only_node_and_the_ends`, `safe_iteration_removes_while_walking` (mutant `s06`) |
| Not clearing the links on remove | a second remove decrements `len` again and corrupts the list | `remove_unlinks_and_marks_the_node` (mutant `s07`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `rt.04` | every cached block's record embeds a `tl_list_node`; unref of a registered block touches it, a lookup hit removes it, and allocation under pressure pops the oldest |
| Forward | `ds.07` | the radix tree's LRU list of leaves is the same idea in safe Rust, with indices instead of pointers |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_lru` | Linux `list_head` | `list_for_each_entry_safe`, splice, RCU-safe variants | [`include/linux/list.h`](https://github.com/torvalds/linux/blob/master/include/linux/list.h) |
| LRU of cached blocks | vLLM's `FreeKVCacheBlockQueue` | a doubly linked free queue of KV blocks where cached blocks are reclaimed oldest first | [`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py) |
| one LRU list | Linux page cache (active and inactive lists) | two lists so a page read once does not evict a page read often | `mm/vmscan.c` |
