# Intrusive linked list

Most linked lists you have seen allocate a node that points at your data. The Linux kernel does it the other way round: the node lives *inside* your struct, and the list links those embedded nodes together. This section builds that list, a circular doubly linked list with a sentinel head, together with the `container_of` macro that recovers your struct from a pointer to its node.

The payoff is a list that never allocates, so insertion cannot fail, and an object that can sit on several lists at once by carrying several nodes. Along the way you will meet two techniques that are useful far beyond lists: a sentinel element that deletes every edge case, and byte-level pointer arithmetic with a compile-time type check. The same structure, extended into an LRU cache, is [An intrusive list and an LRU](../../../../../content/courses/algorithms/16-systems-data-structures/03-intrusive-list-and-lru.md) in *Algorithms*. You need nothing beyond comfortable C pointers to follow this one.

> **Learning objectives**
> - Implement insert, remove, and splice on a circular sentinel list with no branches for empty, first, or last.
> - Explain how `container_of` computes an enclosing struct's address and why the arithmetic must be in bytes.
> - Iterate a list while deleting from it without reading a freed or unlinked node.
> - Weigh an intrusive list against a wrapper-node list and a `NULL`-terminated one.

## The interface

[`list.h`](list.h) declares the interface and macros, [`list.c`](list.c) is the solution, and [`main.c`](main.c) is the test suite. `container_of` and the iteration macros are in the header; the list operations are in `list.c`.

| Function | Does |
|----------|------|
| `list_init(head)` | Point the sentinel at itself |
| `list_empty(head)` | True when only the sentinel remains |
| `list_insert_after(n, fresh)` | Splice `fresh` in after `n` |
| `list_insert_before(n, fresh)` | Splice `fresh` in before `n` |
| `list_push_back(head, fresh)` | Append. One line over `insert_before` |
| `list_push_front(head, fresh)` | Prepend. One line over `insert_after` |
| `list_remove(n)` | Unlink `n`, leaving it a valid empty node |
| `list_is_linked(n)` | Whether `n` is currently on a list |
| `list_length(head)` | Count, O(n) |
| `list_splice(dst, src)` | Move all of `src` onto `dst`, O(1) |

The one constraint is that `list.c` allocates nothing. The list never owns a node and never knows what type encloses it.

To build and run the tests from this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c list.c -o list && ./list
# ok  c/03-intrusive-list  10057 checks passed
```

Add `-g -fsanitize=address,undefined` to catch any stray write through a stale pointer.

## Why intrusive at all

Because the list allocates nothing, an insert cannot fail, which matters enormously in a kernel or an allocator where "handle malloc failing here" is not a comfortable sentence. One object can also belong to several lists at once by carrying several nodes, which `test_one_object_on_two_lists` demonstrates: a task in the "all tasks" list and the "ready" list simultaneously, with no extra allocation and no second lookup.

The cost is that the list is untyped and every access needs `container_of`, and that a node cannot be in the same list twice. That trade is why you find this pattern in kernels and allocators and almost nowhere else.

## The sentinel and the tail you do not need

The sentinel head deletes every edge case. Because the list is circular and the head is a real node, `head->next` always exists, so insert and remove are four unconditional pointer writes with no branch for empty, first, or last. Write the same list with `NULL` terminators and you find yourself checking for `NULL` in six places and getting one of them wrong. This is a general technique: a fake element that can never be the answer often removes all the special cases.

It also explains something that surprises people: `push_back` needs no tail pointer. The sentinel sits between the tail and the head, so "append" is just "insert before the sentinel". A singly linked list without a sentinel needs an explicit tail pointer and has to maintain it on every operation; here it falls out of the shape for free.

## Getting back to your struct

`container_of` does pointer arithmetic that must be done in bytes. The cast to `char *` is not cosmetic: subtracting `offsetof` from a `Task *` would scale by `sizeof(Task)` and land far outside the object. The `1 ? (ptr) : ...` in the macro is a compile-time type check with no run-time cost. It makes the compiler reject a pointer to the wrong member, which converts a silent wrong-offset memory bug into a build error.

The test suite guards this from the other side. Its node is deliberately not the first member of `Task`. If it were, `offsetof` would be zero and a broken `container_of` that ignored the offset entirely would pass every test, so the test's `Task` puts `id` first.

## Deriving the design

Six observations build the whole list.

The node must be embedded, so the list cannot allocate. Linking an object means writing two pointers that already exist inside it. That is why insert cannot fail, and why one object can be on several lists at once: it carries one `ListNode` per list.

Recovering the object from its node is pure arithmetic. The node sits at a fixed offset inside the struct, `offsetof(type, member)`. Subtracting that many *bytes* from the node's address gives the struct's address. That is all `container_of` does.

A circular list with a sentinel has no ends. Make the head a real node that holds no element, and link the last element back to it. Then every node, including the head, always has a valid `prev` and `next`. Empty is `head->next == head`.

With no ends, insertion is four writes. To put `fresh` after `node`, point `fresh` at its two neighbours, then point the two neighbours at `fresh`. There is nothing to check. Every other insert is this one in disguise: before `n` is after `n->prev`; push-back is before the head; push-front is after the head.

Removal is two writes, plus a choice about the removed node. Unlinking needs only the neighbours. Resetting the removed node to point at itself (the `list_init` state) costs two more writes and buys idempotent removal and a way to ask "am I linked?".

Moving a whole list is O(1). Its first and last elements are `src->next` and `src->prev`; stitch them between `dst`'s tail and `dst`. No element is visited.

## Building it step by step

The core operation. Every insert in the file reduces to it:

```c
void list_insert_after(ListNode *node, ListNode *fresh) {
  /* No NULL checks and no empty-list case: the sentinel means node->next
   * always exists, even when it is node itself. */
  fresh->prev = node;
  fresh->next = node->next;
  node->next->prev = fresh;   /* must read node->next before overwriting it */
  node->next = fresh;
}

void list_insert_before(ListNode *node, ListNode *fresh) {
  list_insert_after(node->prev, fresh);
}

void list_push_back(ListNode *head, ListNode *fresh) {
  /* The sentinel is the element after the tail, so inserting before it
   * appends. That is why push_back needs no tail pointer. */
  list_insert_before(head, fresh);
}
```

The order of the last two writes matters: `node->next->prev = fresh` uses the
*old* `node->next`. Swap them and `fresh` becomes its own predecessor.

Removal and the two queries built on what it leaves behind:

```c
void list_remove(ListNode *node) {
  node->prev->next = node->next;
  node->next->prev = node->prev;

  /* Reset to the empty-node state rather than leaving stale pointers into a
   * list this node is no longer part of. Removing twice is then harmless:
   * the node's neighbours are itself, so the two writes above touch only
   * the node. */
  list_init(node);
}

int list_is_linked(const ListNode *node) {
  return node->next != node;   /* "not on a list" is exactly "points at itself" */
}

int list_empty(const ListNode *head) {
  return head->next == head;
}
```

`list_is_linked` and `list_empty` are the same test. That is not an accident: an
unlinked node and an empty list's sentinel are in the same state.

The O(1) splice:

```c
void list_splice(ListNode *dst, ListNode *src) {
  if (list_empty(src)) return;   /* otherwise first == last == src itself */

  ListNode *first = src->next;
  ListNode *last = src->prev;
  ListNode *tail = dst->prev;

  /* Four pointer writes, whatever the length. The elements in between are
   * never touched. */
  tail->next = first;
  first->prev = tail;
  last->next = dst;
  dst->prev = last;

  list_init(src);   /* src's sentinel no longer owns those elements */
}
```

And the macro that makes the whole design usable, from the header:

```c
#define container_of(ptr, type, member)                                        \
  ((type *)((char *)(1 ? (ptr) : &(((type *)0)->member)) - offsetof(type, member)))
```

`(char *)` makes the subtraction count bytes. The conditional never evaluates
its third operand at runtime, but both operands must have compatible types, so
passing a pointer to the wrong member fails to compile.

The full solution is in [`list.c`](list.c) and [`list.h`](list.h).

## Why this design

Every operation is O(1) except `list_length`, which walks the list. Space overhead is two pointers per membership, inside the object.

There is no size field, and that is forced rather than chosen. `list_remove(node)` takes only the node; it does not know which head it belongs to, so it could not update a count. Keeping `list_length` O(n) is the price of a remove that needs no head.

Against a wrapper-node list (`struct Node { void *item; Node *next; }`), the intrusive design wins on every operation that matters here: in the wrapper version every insert allocates and can fail, items need a cast back from `void *`, and removing an item you hold needs a search to find its node. The intrusive list removes in O(1) from the item itself.

Against `NULL`-terminated lists, the sentinel costs one node and saves a branch for empty, head, and tail in insert and remove, which is where the bugs live.

The accepted trade is that the list is untyped (every access goes through `container_of`), and a node can be on only one list at a time per embedded `ListNode`.

## What can go wrong

The most dangerous mistake is deleting during iteration. The plain `list_for_each` reads `pos->next` at the end of each step, so if the body unlinks `pos`, the next read comes from a node that is no longer in the list. `list_for_each_safe` reads the successor *before* running the body, which is why it needs the extra variable.

> **Warning:** In `list_insert_after`, the order of the last two writes matters. `node->next->prev = fresh` must use the *old* `node->next`; swap the two lines and `fresh` becomes its own predecessor.

The remaining hazards are summarised below.

| Mistake | Consequence |
|---------|-------------|
| Subtracting `offsetof` from a typed pointer | The offset is scaled by the pointee size and lands outside the object |
| Putting the node first in the test struct | A `container_of` that ignores the offset still passes |
| Removing `pos` inside `list_for_each` | The loop reads from an unlinked (self-pointing) or freed node |
| Splicing an empty `src` without the check | The source sentinel is linked into `dst` as if it were an element |
| Leaving a removed node's pointers stale | A second remove corrupts the list it used to be on |

## Where this leads

An intrusive list is the backbone of an LRU cache: a hash map finds the entry, and the embedded node moves it to the front in O(1). [An intrusive list and an LRU](../../../../../content/courses/algorithms/16-systems-data-structures/03-intrusive-list-and-lru.md) in *Algorithms* builds exactly that. Kernel run queues, allocator free lists, and timer wheels all use the same pattern.

## Key terms

- **Intrusive list**: a list whose link fields are embedded in the elements themselves rather than in separately allocated nodes.
- **Sentinel**: a dummy node that is never an element, used so that every real node always has neighbours.
- **`container_of`**: a macro that computes the address of an enclosing struct from the address of one of its members.
- **`offsetof`**: the standard macro giving a member's byte offset within its struct.
- **Splice**: moving a whole run of nodes from one list to another by rewriting only the boundary pointers.

## Summary

- The node lives inside the object, so linking never allocates and an object can be on several lists at once.
- `container_of` subtracts the member's byte offset, through a `char *`, with a conditional that type-checks the member at compile time.
- A circular list with a sentinel has no ends, so insert is four unconditional writes and push-back needs no tail pointer.
- Removal resets the node to point at itself, which makes removal idempotent and gives `list_is_linked` for free.
- Splice is O(1); length is O(n) because remove does not know its head.

## Review questions

1. Why does `container_of` cast to `char *` before subtracting?

   <details><summary>Answer</summary>

   Pointer subtraction scales by the size of the pointee. `offsetof` is a byte
   count, so the arithmetic must be done on a byte pointer. On a `ListNode *` it
   would back up `offset * sizeof(ListNode)` bytes and land outside the object.

   </details>

2. What makes calling `list_remove` twice on the same node safe?

   <details><summary>Answer</summary>

   After the first call the node points at itself. The second call then runs
   `node->prev->next = node->next`, which is `node->next = node`, and the same
   for `prev`: it rewrites the node's own pointers with the values they already
   have and touches nothing else.

   </details>

3. Why does `list_splice` return early when `src` is empty?

   <details><summary>Answer</summary>

   For an empty list, `src->next` and `src->prev` are `src` itself. Without the
   check, the splice would link the source sentinel into `dst` as if it were an
   element, and `list_init(src)` would then leave `dst` pointing at a node that
   no longer points back.

   </details>

4. Why does `list_for_each_safe` need a second variable?

   <details><summary>Answer</summary>

   The plain loop advances with `pos = pos->next` after the body. If the body
   removed `pos`, that read happens on an unlinked node, which (after
   `list_remove`) points at itself, so the loop would spin forever, or with a
   freed node read freed memory. The safe version stores `pos->next` in `tmp`
   before the body runs and advances to `tmp`.

   </details>
