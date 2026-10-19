/* map.h - a string-keyed hash map with open addressing.
 *
 * The interface. map.c implements it and main.c tests it.
 *
 * The map owns copies of the keys you give it, so callers may free or reuse
 * their own buffers immediately after a put.
 */
#ifndef MAP_H
#define MAP_H

#include <stddef.h>

typedef enum { SLOT_EMPTY = 0, SLOT_FULL, SLOT_TOMB } SlotState;

typedef struct {
  char *key;       /* owned by the map, NULL unless state is SLOT_FULL */
  int value;
  SlotState state;
} Slot;

typedef struct {
  Slot *slots;
  size_t cap;   /* number of slots, always a power of two or 0 */
  size_t len;   /* live entries */
  size_t used;  /* live entries plus tombstones: this is what triggers growth */
} Map;

/* Zero state. Allocates nothing. */
void map_init(Map *m);

/* Insert or overwrite. 0 on success, -1 on allocation failure. */
int map_put(Map *m, const char *key, int value);

/* 0 and *out set when present, -1 when absent. */
int map_get(const Map *m, const char *key, int *out);

/* 0 when it was there and is now gone, -1 when it was never there. */
int map_del(Map *m, const char *key);

/* Live entries. Tombstones do not count. */
size_t map_len(const Map *m);

/* Free the slots and every key the map owns, and return to the zero state. */
void map_free(Map *m);

/* FNV-1a over a NUL-terminated string. Public so that the tests can
 * reason about collisions deliberately rather than accidentally. */
size_t map_hash(const char *key);

#endif /* MAP_H */
