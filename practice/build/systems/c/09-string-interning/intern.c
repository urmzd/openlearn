/* intern.c - the reference implementation of the interning table. */
#include "intern.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Each entry is one allocation holding the Interned header, then the bytes,
 * then a NUL. One allocation per string rather than two means the string data
 * cannot be separated from its header, and halves the allocator traffic.
 *
 * `hash` is cached so that growing the table never rehashes the text, and so
 * that a lookup can reject a colliding entry without touching its bytes. */
typedef struct Entry {
  Interned pub; /* must be first: the public pointer is &entry->pub */
  size_t hash;
  char data[];
} Entry;

struct InternTable {
  Entry **slots; /* open addressing, NULL where empty */
  size_t cap;    /* always a power of two, or 0 */
  size_t len;
};

static size_t hash_bytes(const char *s, size_t n) {
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n; i++) {
    h ^= (unsigned char)s[i];
    h *= 1099511628211ULL;
  }
  return (size_t)h;
}

static int entry_matches(const Entry *e, size_t hash, const char *s, size_t len) {
  /* Compare the cheap things first. The hash rejects almost every collision
   * without a memcmp, and the length rejects the rest before touching bytes. */
  return e->hash == hash && e->pub.len == len && memcmp(e->pub.str, s, len) == 0;
}

/* Grow and rehash. */
static int table_grow(InternTable *t) {
  size_t cap = t->cap ? t->cap * 2 : 16;
  Entry **slots = calloc(cap, sizeof(Entry *));
  if (!slots) return -1;

  /* Reinsert the POINTERS, never the strings. Every Interned handed out so far
   * must keep its address: reallocating or copying the string data here would
   * dangle every pointer a caller is holding, which destroys the one guarantee
   * interning exists to provide. */
  for (size_t i = 0; i < t->cap; i++) {
    Entry *e = t->slots[i];
    if (!e) continue;
    size_t j = e->hash & (cap - 1);
    while (slots[j]) j = (j + 1) & (cap - 1);
    slots[j] = e;
  }

  free(t->slots);
  t->slots = slots;
  t->cap = cap;
  return 0;
}

/* Index of a matching entry, or of the first free slot for it. Returns 1 when
 * found. Assumes cap > 0. */
static int table_probe(const InternTable *t, size_t hash, const char *s,
                       size_t len, size_t *idx) {
  size_t mask = t->cap - 1;
  size_t i = hash & mask;
  while (t->slots[i]) {
    if (entry_matches(t->slots[i], hash, s, len)) {
      *idx = i;
      return 1;
    }
    i = (i + 1) & mask;
  }
  *idx = i;
  return 0;
}

InternTable *intern_new(void) {
  InternTable *t = calloc(1, sizeof *t);
  return t; /* slots stay NULL until the first intern */
}

const Interned *intern_n(InternTable *t, const char *s, size_t len) {
  if (t->cap == 0 && table_grow(t) != 0) return NULL;

  size_t hash = hash_bytes(s, len);
  size_t i;
  if (table_probe(t, hash, s, len, &i)) return &t->slots[i]->pub;

  /* Grow at 3/4 load and re-probe, since the index just computed refers to
   * the old table. There are no tombstones here because interning never
   * removes anything, which is what makes this table simpler than a general
   * hash map. */
  if ((t->len + 1) * 4 >= t->cap * 3) {
    if (table_grow(t) != 0) return NULL;
    table_probe(t, hash, s, len, &i);
  }

  /* Header, bytes and terminator in one allocation. */
  if (len > SIZE_MAX - sizeof(Entry) - 1) return NULL;
  Entry *e = malloc(sizeof(Entry) + len + 1);
  if (!e) return NULL;

  memcpy(e->data, s, len);
  e->data[len] = '\0'; /* terminate even when the input was not */
  e->pub.str = e->data;
  e->pub.len = len;
  e->hash = hash;

  t->slots[i] = e;
  t->len++;
  return &e->pub;
}

const Interned *intern(InternTable *t, const char *s) {
  return intern_n(t, s, strlen(s));
}

const Interned *intern_find(const InternTable *t, const char *s) {
  if (t->cap == 0) return NULL;
  size_t len = strlen(s);
  size_t i;
  if (!table_probe(t, hash_bytes(s, len), s, len, &i)) return NULL;
  return &t->slots[i]->pub;
}

size_t intern_count(const InternTable *t) {
  return t->len;
}

void intern_free(InternTable *t) {
  if (!t) return;
  for (size_t i = 0; i < t->cap; i++) free(t->slots[i]);
  free(t->slots);
  free(t);
}
