/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "list.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks = 0;
#define CHECK(cond, what)                                                      \
  do {                                                                         \
    checks++;                                                                  \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/* Two embedded nodes, so one Task can be on two lists at once: that is the
 * capability an intrusive list has and a wrapper-node list does not.
 *
 * Note that neither node is the first member. If `link` sat at offset zero, a
 * container_of that ignored the offset entirely would pass every test here. */
typedef struct {
  int id;
  ListNode link;  /* position on the main list */
  char name[16];
  ListNode ready; /* position on the ready list */
} Task;

static Task *task_of(ListNode *n) { return container_of(n, Task, link); }
static Task *task_of_ready(ListNode *n) { return container_of(n, Task, ready); }

/* Collect ids in list order, forwards. Returns how many it wrote. */
static size_t ids_forward(ListNode *head, int *out, size_t cap) {
  size_t n = 0;
  for (ListNode *p = head->next; p != head && n < cap; p = p->next) {
    out[n++] = task_of(p)->id;
  }
  return n;
}

/* Same, backwards. A doubly-linked list that is only correct in one direction
 * is a common half-finished state, so every structural test checks both. */
static size_t ids_backward(ListNode *head, int *out, size_t cap) {
  size_t n = 0;
  for (ListNode *p = head->prev; p != head && n < cap; p = p->prev) {
    out[n++] = task_of(p)->id;
  }
  return n;
}

static int same(const int *a, const int *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) return 0;
  }
  return 1;
}

static void test_empty_list(void) {
  ListNode head;
  list_init(&head);
  CHECK(list_empty(&head), "a fresh list is empty");
  CHECK(list_length(&head) == 0, "a fresh list has length 0");
  CHECK(head.next == &head, "sentinel points forward at itself");
  CHECK(head.prev == &head, "sentinel points backward at itself");
}

static void test_container_of_recovers_the_struct(void) {
  Task t = {.id = 77};
  snprintf(t.name, sizeof t.name, "%s", "seventy-seven");

  ListNode *n = &t.link;
  CHECK(container_of(n, Task, link) == &t, "container_of recovers the enclosing struct");
  CHECK(container_of(n, Task, link)->id == 77, "and the recovered struct is intact");
  CHECK(strcmp(container_of(n, Task, link)->name, "seventy-seven") == 0,
        "fields after the node are intact");

  /* The second embedded node sits at a different offset, so recovering
   * through the wrong member would land somewhere else entirely. */
  ListNode *r = &t.ready;
  CHECK(container_of(r, Task, ready) == &t, "container_of works for any member");
  CHECK(container_of(r, Task, ready) == container_of(n, Task, link),
        "both members recover the same object");
}

static void test_push_back_appends(void) {
  ListNode head;
  Task a = {.id = 1}, b = {.id = 2}, c = {.id = 3};
  int got[3], want_f[3] = {1, 2, 3}, want_b[3] = {3, 2, 1};
  list_init(&head);
  list_push_back(&head, &a.link);
  list_push_back(&head, &b.link);
  list_push_back(&head, &c.link);

  CHECK(!list_empty(&head), "a list with elements is not empty");
  CHECK(list_length(&head) == 3, "three pushes give length 3");
  CHECK(ids_forward(&head, got, 3) == 3 && same(got, want_f, 3), "push_back appends in order");
  CHECK(ids_backward(&head, got, 3) == 3 && same(got, want_b, 3), "and the reverse links agree");
}

static void test_push_front_prepends(void) {
  ListNode head;
  Task a = {.id = 1}, b = {.id = 2}, c = {.id = 3};
  int got[3], want[3] = {3, 2, 1};
  list_init(&head);
  list_push_front(&head, &a.link);
  list_push_front(&head, &b.link);
  list_push_front(&head, &c.link);
  CHECK(ids_forward(&head, got, 3) == 3 && same(got, want, 3), "push_front prepends");
  CHECK(list_length(&head) == 3, "length counts them all");
}

static void test_insert_before_and_after(void) {
  ListNode head;
  Task a = {.id = 10}, b = {.id = 20}, mid = {.id = 15}, pre = {.id = 5};
  int got[4], want[4] = {5, 10, 15, 20};
  list_init(&head);
  list_push_back(&head, &a.link);
  list_push_back(&head, &b.link);
  list_insert_after(&a.link, &mid.link);
  list_insert_before(&a.link, &pre.link);
  CHECK(ids_forward(&head, got, 4) == 4 && same(got, want, 4), "inserts land in the right places");
  CHECK(ids_backward(&head, got, 4) == 4, "reverse walk visits the same count");
}

/* Removing from the middle, the head, and the tail are three different pointer
 * situations in a naive implementation. With a sentinel they are the same two
 * writes, which is the reason the sentinel exists. */
static void test_remove_from_every_position(void) {
  ListNode head;
  Task a = {.id = 1}, b = {.id = 2}, c = {.id = 3};
  int got[3];
  list_init(&head);
  list_push_back(&head, &a.link);
  list_push_back(&head, &b.link);
  list_push_back(&head, &c.link);

  list_remove(&b.link); /* middle */
  CHECK(list_length(&head) == 2, "removing the middle leaves two");
  {
    int want[2] = {1, 3};
    CHECK(ids_forward(&head, got, 2) == 2 && same(got, want, 2), "middle removal relinks neighbours");
    CHECK(ids_backward(&head, got, 2) == 2, "and the reverse links survive it");
  }

  list_remove(&a.link); /* head */
  CHECK(list_length(&head) == 1, "removing the head leaves one");
  CHECK(ids_forward(&head, got, 1) == 1 && got[0] == 3, "the survivor is the last one");

  list_remove(&c.link); /* tail, and the only element */
  CHECK(list_empty(&head), "removing the last element empties the list");
  CHECK(list_length(&head) == 0, "and the length agrees");
}

static void test_is_linked_tracks_membership(void) {
  ListNode head;
  Task a = {.id = 1};
  list_init(&head);
  list_init(&a.link);

  CHECK(!list_is_linked(&a.link), "a freshly initialised node is not on a list");
  list_push_back(&head, &a.link);
  CHECK(list_is_linked(&a.link), "a pushed node reports linked");
  list_remove(&a.link);
  CHECK(!list_is_linked(&a.link), "a removed node reports unlinked");
}

/* A removed node must be left as a valid empty node, not as two pointers into
 * a list it is no longer part of. Otherwise removing twice corrupts whatever
 * its old neighbours have become. */
static void test_remove_twice_is_harmless(void) {
  ListNode head;
  Task a = {.id = 1}, b = {.id = 2};
  list_init(&head);
  list_push_back(&head, &a.link);
  list_push_back(&head, &b.link);

  list_remove(&a.link);
  list_remove(&a.link); /* must not touch the list a was on */
  CHECK(list_length(&head) == 1, "the list is unaffected by the second removal");
  CHECK(task_of(head.next)->id == 2, "and still holds the right element");

  /* An unlinked node can be relinked. */
  list_push_back(&head, &a.link);
  CHECK(list_length(&head) == 2, "a removed node can be reinserted");
  CHECK(list_is_linked(&a.link), "and reports linked again");
}

static void test_iteration_may_unlink_as_it_goes(void) {
  ListNode head;
  Task t[6];
  size_t removed = 0;
  list_init(&head);
  for (int i = 0; i < 6; i++) {
    t[i].id = i;
    list_push_back(&head, &t[i].link);
  }

  /* Drop the even ids while walking. Without the saved successor, the first
   * unlink would leave the loop reading a node that has been reset. */
  list_for_each_safe(pos, tmp, &head) {
    if (task_of(pos)->id % 2 == 0) {
      list_remove(pos);
      removed++;
    }
  }
  CHECK(removed == 3, "three even ids were removed");
  CHECK(list_length(&head) == 3, "three odd ids remain");
  {
    int got[3], want[3] = {1, 3, 5};
    CHECK(ids_forward(&head, got, 3) == 3 && same(got, want, 3), "the survivors are the odd ids, in order");
  }

  /* The plain macro is fine when the body does not unlink. */
  int sum = 0;
  list_for_each(pos, &head) sum += task_of(pos)->id;
  CHECK(sum == 9, "list_for_each visits every survivor");
}

/* One object on two lists at once. A wrapper-node list would need a second
 * allocation per membership; here it is a second embedded field. */
static void test_one_object_on_two_lists(void) {
  ListNode all, ready;
  Task a = {.id = 1}, b = {.id = 2}, c = {.id = 3};
  list_init(&all);
  list_init(&ready);

  list_push_back(&all, &a.link);
  list_push_back(&all, &b.link);
  list_push_back(&all, &c.link);
  list_push_back(&ready, &b.ready); /* only b and c are ready */
  list_push_back(&ready, &c.ready);

  CHECK(list_length(&all) == 3, "all holds three");
  CHECK(list_length(&ready) == 2, "ready holds two");
  CHECK(task_of_ready(ready.next)->id == 2, "ready starts at b");

  int sum = 0;
  list_for_each(pos, &ready) sum += task_of_ready(pos)->id;
  CHECK(sum == 5, "the ready list sees the same objects");

  /* Leaving the ready list must not disturb membership of the main list. */
  list_remove(&b.ready);
  CHECK(list_length(&ready) == 1, "b left the ready list");
  CHECK(list_length(&all) == 3, "and is still on the main list");
  CHECK(task_of(all.next->next)->id == 2, "in its original position");
  CHECK(list_is_linked(&b.link), "still linked by its main-list node");
  CHECK(!list_is_linked(&b.ready), "and unlinked by its ready node");
}

static void test_splice_moves_everything(void) {
  ListNode dst, src;
  Task a = {.id = 1}, b = {.id = 2}, c = {.id = 3}, d = {.id = 4};
  int got[4], want[4] = {1, 2, 3, 4};
  list_init(&dst);
  list_init(&src);
  list_push_back(&dst, &a.link);
  list_push_back(&dst, &b.link);
  list_push_back(&src, &c.link);
  list_push_back(&src, &d.link);

  list_splice(&dst, &src);
  CHECK(list_length(&dst) == 4, "dst absorbed both elements");
  CHECK(list_empty(&src), "src is left empty");
  CHECK(ids_forward(&dst, got, 4) == 4 && same(got, want, 4), "order is preserved across the splice");
  CHECK(ids_backward(&dst, got, 4) == 4, "and the reverse links are intact");

  /* Splicing an empty list must be a no-op rather than corrupting dst. */
  list_splice(&dst, &src);
  CHECK(list_length(&dst) == 4, "splicing an empty list changes nothing");
  CHECK(ids_backward(&dst, got, 4) == 4, "and leaves the links alone");

  /* Splicing into an empty destination is the other boundary. */
  ListNode empty;
  list_init(&empty);
  list_splice(&empty, &dst);
  CHECK(list_length(&empty) == 4, "an empty destination absorbs everything");
  CHECK(list_empty(&dst), "and the source is emptied");
  CHECK(ids_forward(&empty, got, 4) == 4 && same(got, want, 4), "order survives that too");
}

static void test_many_elements(void) {
  enum { N = 5000 };
  ListNode head;
  Task *t = malloc(N * sizeof *t);
  CHECK(t != NULL, "test allocation succeeded");
  list_init(&head);
  for (int i = 0; i < N; i++) {
    t[i].id = i;
    list_push_back(&head, &t[i].link);
  }
  CHECK(list_length(&head) == N, "every element is on the list");

  size_t seen = 0;
  int expect = 0;
  for (ListNode *p = head.next; p != &head; p = p->next) {
    CHECK(task_of(p)->id == expect++, "order held across many pushes");
    seen++;
  }
  CHECK(seen == N, "forward walk saw them all");

  seen = 0;
  expect = N - 1;
  for (ListNode *p = head.prev; p != &head; p = p->prev) {
    CHECK(task_of(p)->id == expect--, "reverse order is exactly the mirror");
    seen++;
  }
  CHECK(seen == N, "backward walk saw them all");
  free(t);
}

int main(void) {
  test_empty_list();
  test_container_of_recovers_the_struct();
  test_push_back_appends();
  test_push_front_prepends();
  test_insert_before_and_after();
  test_remove_from_every_position();
  test_is_linked_tracks_membership();
  test_remove_twice_is_harmless();
  test_iteration_may_unlink_as_it_goes();
  test_one_object_on_two_lists();
  test_splice_moves_everything();
  test_many_elements();
  printf("ok  c/03-intrusive-list  %d checks passed\n", checks);
  return 0;
}
