/* main.c - the test suite for the reference implementation. Exits 0 when every
 * check holds. */
#include "intern.h"

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

static void test_empty_table(void) {
  InternTable *t = intern_new();
  CHECK(t != NULL, "table created");
  CHECK(intern_count(t) == 0, "a fresh table holds nothing");
  CHECK(intern_find(t, "absent") == NULL, "finding in an empty table returns NULL");
  intern_free(t);
}

/* The defining property: same text in, same pointer out. */
static void test_same_text_same_pointer(void) {
  InternTable *t = intern_new();
  char buf[32];

  snprintf(buf, sizeof buf, "%s", "hello");
  const Interned *a = intern(t, buf);
  CHECK(a != NULL, "intern succeeds");

  /* A different buffer holding the same text must return the same pointer. */
  snprintf(buf, sizeof buf, "%s", "hello");
  const Interned *b = intern(t, buf);
  CHECK(b == a, "interning the same text twice returns the identical pointer");
  CHECK(intern_count(t) == 1, "and stores only one copy");

  const Interned *c = intern(t, "world");
  CHECK(c != a, "different text gets a different pointer");
  CHECK(intern_count(t) == 2, "two distinct strings");

  CHECK(strcmp(a->str, "hello") == 0, "the stored text is right");
  CHECK(a->len == 5, "the cached length is right");
  intern_free(t);
}

/* The table must own its copy. The caller's buffer being reused or freed
 * afterwards cannot be allowed to change what the table holds. */
static void test_table_owns_its_copies(void) {
  InternTable *t = intern_new();
  char *heap = malloc(16);
  strcpy(heap, "transient");

  const Interned *a = intern(t, heap);
  strcpy(heap, "clobbered");
  CHECK(strcmp(a->str, "transient") == 0, "the table kept its own copy");
  CHECK(a->str != heap, "and it is not aliasing the caller's buffer");

  free(heap);
  CHECK(strcmp(a->str, "transient") == 0, "which survives the caller freeing theirs");

  CHECK(intern_find(t, "transient") == a, "and is still findable");
  CHECK(intern_find(t, "clobbered") == NULL, "the clobbered text was never interned");
  intern_free(t);
}

/* The hard requirement. Pointers handed out before the table grew must remain
 * valid and must still compare equal to what interning returns afterwards.
 * An implementation that reallocates its string storage on growth fails here,
 * and would fail in production as a use-after-free. */
static void test_pointers_survive_growth(void) {
  enum { N = 5000 };
  InternTable *t = intern_new();
  const Interned *saved[N];
  char buf[32];

  for (int i = 0; i < N; i++) {
    snprintf(buf, sizeof buf, "symbol-%d", i);
    saved[i] = intern(t, buf);
    CHECK(saved[i] != NULL, "intern succeeds during growth");
  }
  CHECK(intern_count(t) == N, "every distinct string was stored once");

  /* Every early pointer must still hold its text after many rehashes. */
  for (int i = 0; i < N; i++) {
    snprintf(buf, sizeof buf, "symbol-%d", i);
    CHECK(strcmp(saved[i]->str, buf) == 0, "an early pointer still holds its text");
    CHECK(saved[i]->len == strlen(buf), "and its cached length");
    CHECK(intern(t, buf) == saved[i], "and re-interning still returns it");
  }

  /* Re-interning everything must not have added anything. */
  CHECK(intern_count(t) == N, "re-interning added no duplicates");
  intern_free(t);
}

/* Pointer equality replacing strcmp is the whole payoff, so prove it holds
 * over strings that a weak comparison would confuse: shared prefixes, shared
 * suffixes, and differing lengths. */
static void test_pointer_equality_is_string_equality(void) {
  InternTable *t = intern_new();
  const char *words[] = {"a",  "ab",  "abc", "abcd", "b",   "ba",
                         "cab", "cba", "",    "abcde", "abc"};
  const size_t n = sizeof words / sizeof *words;
  const Interned *got[11];

  for (size_t i = 0; i < n; i++) got[i] = intern(t, words[i]);

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      int same_text = strcmp(words[i], words[j]) == 0;
      int same_ptr = got[i] == got[j];
      CHECK(same_text == same_ptr, "pointer equality matches string equality exactly");
    }
  }

  /* "abc" appears twice in the list, so one distinct string is a duplicate. */
  CHECK(intern_count(t) == n - 1, "the duplicate was not stored twice");
  intern_free(t);
}

static void test_empty_string(void) {
  InternTable *t = intern_new();
  const Interned *e1 = intern(t, "");
  const Interned *e2 = intern(t, "");
  CHECK(e1 != NULL, "the empty string can be interned");
  CHECK(e1 == e2, "and interns to a single copy");
  CHECK(e1->len == 0, "with length zero");
  CHECK(e1->str[0] == '\0', "and a valid terminator");
  CHECK(intern_count(t) == 1, "counted once");
  intern_free(t);
}

/* Embedded NULs distinguish a length-aware table from one that leans on
 * strlen internally. "a\0b" and "a" have the same strlen and different text. */
static void test_embedded_nuls(void) {
  InternTable *t = intern_new();
  const char with_nul[] = {'a', '\0', 'b'};

  const Interned *a = intern_n(t, with_nul, 3);
  const Interned *b = intern(t, "a");
  CHECK(a != NULL && b != NULL, "both intern");
  CHECK(a != b, "a string with an embedded NUL is not the same as its prefix");
  CHECK(a->len == 3, "the full length is kept");
  CHECK(b->len == 1, "and the short one is unaffected");
  CHECK(memcmp(a->str, with_nul, 3) == 0, "the bytes round-trip");
  CHECK(a->str[3] == '\0', "and the table still terminates it for C consumers");

  const Interned *again = intern_n(t, with_nul, 3);
  CHECK(again == a, "re-interning the same bytes returns the same pointer");
  CHECK(intern_count(t) == 2, "two distinct strings");
  intern_free(t);
}

/* A non-terminated buffer: intern_n must read exactly len bytes and no more.
 * Reading one further would pick up the sentinel and change the result. */
static void test_unterminated_input(void) {
  InternTable *t = intern_new();
  char raw[4] = {'x', 'y', 'z', 'Q'}; /* no NUL anywhere */

  const Interned *a = intern_n(t, raw, 3);
  CHECK(a->len == 3, "only the requested bytes were taken");
  CHECK(strcmp(a->str, "xyz") == 0, "and the copy is terminated");
  CHECK(intern(t, "xyz") == a, "which matches the same text interned normally");
  intern_free(t);
}

static void test_find_does_not_intern(void) {
  InternTable *t = intern_new();
  intern(t, "present");
  CHECK(intern_count(t) == 1, "one string interned");

  CHECK(intern_find(t, "missing") == NULL, "find reports an absent string");
  CHECK(intern_count(t) == 1, "and find did not add it");
  CHECK(intern_find(t, "present") != NULL, "find locates a present string");
  CHECK(intern_find(t, "present") == intern(t, "present"), "and agrees with intern");
  intern_free(t);
}

int main(void) {
  test_empty_table();
  test_same_text_same_pointer();
  test_table_owns_its_copies();
  test_pointers_survive_growth();
  test_pointer_equality_is_string_equality();
  test_empty_string();
  test_embedded_nuls();
  test_unterminated_input();
  test_find_does_not_intern();
  printf("ok  c/09-string-interning  %d checks passed\n", checks);
  return 0;
}
