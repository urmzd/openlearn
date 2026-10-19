/* map.c - the part you write. */
#include "map.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Given: a decent general-purpose string hash. Not cryptographic, and not
 * collision-resistant against an adversary who knows the seed, which is
 * exactly how hash-flooding denial of service works. */
size_t map_hash(const char *key) {
  uint64_t h = 1469598103934665603ULL; /* FNV offset basis */
  for (const unsigned char *p = (const unsigned char *)key; *p; p++) {
    h ^= (uint64_t)*p;
    h *= 1099511628211ULL; /* FNV prime */
  }
  return (size_t)h;
}

/* Rebuild the table with room for at least `want` entries. A power-of-two
 * capacity lets the bucket index be a mask instead of a modulo, which is the
 * difference between one instruction and a division.
 *
 * Note that this sizes from `want` alone rather than doubling the current
 * capacity, so it can also rehash at the same size or smaller. That is
 * deliberate: a table whose slots are mostly tombstones needs its tombstones
 * purged, not a bigger array. */
static int map_resize(Map *m, size_t want) {
  size_t cap = 8;
  while (cap < want) {
    if (cap > SIZE_MAX / 2) return -1;
    cap *= 2;
  }

  Slot *slots = calloc(cap, sizeof(Slot));
  if (!slots) return -1;

  /* Reinsert every live entry into the new table. Tombstones are dropped here,
   * which is the only thing that ever reclaims them. */
  for (size_t i = 0; i < m->cap; i++) {
    if (m->slots[i].state != SLOT_FULL) continue;
    size_t j = map_hash(m->slots[i].key) & (cap - 1);
    while (slots[j].state == SLOT_FULL) j = (j + 1) & (cap - 1);
    slots[j] = m->slots[i]; /* moves ownership of the key, no copy */
  }

  free(m->slots);
  m->slots = slots;
  m->cap = cap;
  m->used = m->len; /* tombstones are gone */
  return 0;
}

/* Index of `key` if present, or of the first slot it could be inserted into.
 * Returns 1 when found, 0 when not, and writes the index either way. */
static int map_probe(const Map *m, const char *key, size_t *idx) {
  size_t mask = m->cap - 1;
  size_t i = map_hash(key) & mask;
  size_t first_tomb = SIZE_MAX;

  for (size_t step = 0; step < m->cap; step++) {
    Slot *s = &m->slots[i];
    if (s->state == SLOT_EMPTY) {
      /* A run of probes ends at the first genuinely empty slot. Prefer an
       * earlier tombstone so that repeated insert and delete cycles do not
       * push probe sequences longer and longer. */
      *idx = (first_tomb == SIZE_MAX) ? i : first_tomb;
      return 0;
    }
    if (s->state == SLOT_TOMB) {
      if (first_tomb == SIZE_MAX) first_tomb = i;
    } else if (strcmp(s->key, key) == 0) {
      *idx = i;
      return 1;
    }
    i = (i + 1) & mask;
  }

  /* Table is entirely full or entirely tombstoned. */
  *idx = (first_tomb == SIZE_MAX) ? 0 : first_tomb;
  return 0;
}

void map_init(Map *m) {
  m->slots = NULL;
  m->cap = 0;
  m->len = 0;
  m->used = 0;
}

int map_put(Map *m, const char *key, int value) {
  /* Resize when the table is 3/4 used, counting tombstones. Using `used`
   * rather than `len` is what stops a delete-heavy workload from degrading into
   * a linear scan: tombstones still cost probe steps even though they hold
   * nothing.
   *
   * The target size comes from the live count, so a table that tripped this
   * check purely on tombstones is rebuilt at the same capacity with the
   * tombstones dropped, instead of doubling forever while holding nothing. */
  if (m->cap == 0 || (m->used + 1) * 4 >= m->cap * 3) {
    if (map_resize(m, (m->len + 1) * 2) != 0) return -1;
  }

  size_t i;
  if (map_probe(m, key, &i)) {
    m->slots[i].value = value; /* overwrite keeps the existing key copy */
    return 0;
  }

  char *copy = malloc(strlen(key) + 1);
  if (!copy) return -1;
  memcpy(copy, key, strlen(key) + 1);

  if (m->slots[i].state == SLOT_EMPTY) m->used++; /* a tombstone was already counted */
  m->slots[i].key = copy;
  m->slots[i].value = value;
  m->slots[i].state = SLOT_FULL;
  m->len++;
  return 0;
}

int map_get(const Map *m, const char *key, int *out) {
  if (m->cap == 0) return -1;
  size_t i;
  if (!map_probe(m, key, &i)) return -1;
  *out = m->slots[i].value;
  return 0;
}

int map_del(Map *m, const char *key) {
  if (m->cap == 0) return -1;
  size_t i;
  if (!map_probe(m, key, &i)) return -1;

  /* A tombstone, not an empty slot. Clearing it to EMPTY would truncate every
   * probe sequence that ran through this slot, orphaning entries that are
   * still in the table. */
  free(m->slots[i].key);
  m->slots[i].key = NULL;
  m->slots[i].state = SLOT_TOMB;
  m->len--;
  return 0;
}

size_t map_len(const Map *m) {
  return m->len;
}

void map_free(Map *m) {
  for (size_t i = 0; i < m->cap; i++) free(m->slots[i].key);
  free(m->slots);
  map_init(m);
}
