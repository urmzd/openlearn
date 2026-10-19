/* list.h - an intrusive doubly-linked list, in the style the Linux kernel uses.
 *
 * The interface. list.c implements it and main.c tests it.
 *
 * "Intrusive" means the list node lives INSIDE your struct, rather than the
 * list allocating a wrapper node that points at your struct:
 *
 *     typedef struct { int id; ListNode link; } Task;
 *
 * So linking a Task costs no allocation and cannot fail, and one struct can
 * sit on several lists at once by embedding several nodes. The price is that
 * you get back a ListNode * and have to recover the Task * yourself, which is
 * what container_of is for.
 */
#ifndef LIST_H
#define LIST_H

#include <stddef.h>

typedef struct ListNode {
  struct ListNode *prev;
  struct ListNode *next;
} ListNode;

/* Recover the enclosing struct from a pointer to a member.
 *
 * The subtraction is the whole trick: offsetof says how far `member` sits into
 * `type`, so backing up that far from the member lands on the struct. The
 * (char *) cast makes the arithmetic bytewise; without it the compiler scales
 * by the pointee's size and you land somewhere arbitrary.
 *
 * The `1 ? (ptr) : &(((type *)0)->member)` is a compile-time type check that
 * costs nothing at runtime: both arms of the conditional must have compatible
 * types, so passing a pointer to the wrong member fails to build instead of
 * silently computing a wrong offset. */
#define container_of(ptr, type, member)                                        \
  ((type *)((char *)(1 ? (ptr) : &(((type *)0)->member)) - offsetof(type, member)))

/* Iterate over a list. `pos` is declared by the macro and scoped to the loop. */
#define list_for_each(pos, head)                                               \
  for (ListNode *pos = (head)->next; pos != (head); pos = pos->next)

/* Iterate with the successor saved first, so the body may unlink the current
 * node. Walking a list in order to remove things from it is the usual reason
 * to walk a list at all, and the plain macro above cannot survive it. */
#define list_for_each_safe(pos, tmp, head)                                     \
  for (ListNode *pos = (head)->next, *tmp = pos->next; pos != (head);          \
       pos = tmp, tmp = pos->next)

/* A list is a sentinel node whose next and prev point at itself when empty.
 * The sentinel holds no element: it exists so that insertion and removal never
 * have to special-case the ends, which is where hand-rolled lists get their
 * bugs. */
void list_init(ListNode *head);

/* Insert immediately after / before an existing node. Both are O(1) and
 * neither can fail, because nothing is allocated. */
void list_insert_after(ListNode *node, ListNode *fresh);
void list_insert_before(ListNode *node, ListNode *fresh);

/* Append to the tail / prepend to the head. Both are one-liners over the two
 * above, because the sentinel makes "the end" an ordinary position. */
void list_push_back(ListNode *head, ListNode *fresh);
void list_push_front(ListNode *head, ListNode *fresh);

/* Unlink a node and leave it as a valid empty node, so that removing twice is
 * harmless and the node can be relinked afterwards. The list it came from is
 * never consulted: an intrusive node does not know which list it is on. */
void list_remove(ListNode *node);

/* Nonzero when the node is currently on some list. Meaningful for any node
 * that has been through list_init or list_remove; a node that has never been
 * either holds uninitialised pointers and cannot be asked. */
int list_is_linked(const ListNode *node);

/* Nonzero when the list holds no elements. */
int list_empty(const ListNode *head);

/* Walk the list and count. O(n), deliberately: a size field would have to be
 * updated by every operation, and the operations do not know which head they
 * belong to. */
size_t list_length(const ListNode *head);

/* Splice every element of `src` onto the end of `dst`, leaving `src` empty.
 * O(1): a list of a million elements moves in four pointer writes. */
void list_splice(ListNode *dst, ListNode *src);

#endif /* LIST_H */
