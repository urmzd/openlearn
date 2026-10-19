# C 02: Hash map (open addressing)

**Concepts:** hash functions, linear probing, tombstones, load factor
**Difficulty:** ⭐⭐⭐

A string-keyed hash map that stores entries in one flat array and resolves
collisions by walking to the next free slot, rather than by chaining to a linked
list. `map_hash` is public so that the tests can construct collisions on
purpose.

## The problem

[`map.h`](map.h) declares the interface, [`map.c`](map.c) is the solution, and
[`main.c`](main.c) is the test suite.

| Function | Does |
|----------|------|
| `map_init(m)` | Zero state, no allocation |
| `map_put(m, k, v)` | Insert or overwrite. Copies `k` |
| `map_get(m, k, &v)` | 0 and writes `v` if present, -1 if absent |
| `map_del(m, k)` | 0 if it was there, -1 if it never was |
| `map_len(m)` | Live entries, not counting tombstones |
| `map_free(m)` | Release the table and every key it owns |

The map owns its keys. Callers must be free to reuse or free their own buffer
the instant `map_put` returns. The header fixes the representation: each `Slot`
is `EMPTY`, `FULL`, or `TOMB`, and the `Map` tracks `cap` (a power of two),
`len` (live entries), and `used` (live entries plus tombstones).

## Build and run

From this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c map.c -o map && ./map
# ok  c/02-hash-map  50031 checks passed
```

Add `-g -fsanitize=address,undefined` to check for leaks and out-of-bounds
access as well.

## What to notice

**Deletion is the hard part, and it is not obvious why.** In a chained map you
unlink a node and you are done. In an open-addressed map, entries that collided
with the deleted one were placed *after* it in the probe run, and a lookup stops
at the first empty slot. Clear a slot to empty and you have not deleted one
entry, you have made every entry behind it unreachable while it still sits in
the table taking space. So deletion writes a tombstone: a slot that says "keep
probing, but you may insert here."

**Which means the load factor has to count tombstones.** A tombstone costs a
probe step forever but holds nothing, so a table that resizes on live-entry
count alone will fill with tombstones under a delete-heavy workload and decay
to a linear scan while reporting a low load factor. That is what the churn test
(`test_churn_does_not_explode`) measures. Growing on live-plus-tombstones is
what reclaims them, because rehashing simply drops them.

**Power-of-two capacity buys a mask instead of a modulo.** `h & (cap - 1)`
replaces `h % cap`, which matters because integer division is one of the
slowest instructions you can put in a hot loop. The cost is that you now depend
entirely on the hash's low bits being well mixed, which is why a weak hash hurts
a power-of-two table far more than a prime-modulo one.

**The hash is not adversary-resistant, and that is a vulnerability.** FNV-1a is
fine against accidental collisions and useless against a caller who chooses
keys to collide on purpose. Feed attacker-controlled strings into this map and
every insert lands in one bucket, turning an O(1) endpoint into O(n) per
request. This is hash-flooding, it was a real CVE class across most web
frameworks in 2011, and the fix is a per-process random seed (SipHash), not a
better mixing constant.

## The solution, step by step

### From first principles

1. **One flat array, so a lookup touches contiguous memory.** Each key hashes to
   a home slot `h & (cap - 1)`. If that slot holds another key, step to the next
   slot (linear probing) until the key or an empty slot appears.
2. **A lookup may stop at an empty slot because insertion never skips one.** An
   insert places a key in the first free slot of its probe run, so a key can
   never sit behind an empty slot on its own run. That invariant is what makes
   "stop at empty" correct.
3. **Deletion must preserve that invariant.** Turning a deleted slot back into
   `EMPTY` would create a gap in the middle of other keys' runs. So deletion
   writes `TOMB`: lookups treat it as occupied (keep probing), inserts treat it
   as free (reuse it).
4. **Tombstones cost probe steps, so they count toward the load.** The table
   tracks `used = live + tombstones` and resizes when `used` reaches 3/4 of the
   capacity. With linear probing, expected probe lengths stay short below that
   load and explode as it approaches 1.
5. **Resizing is a rebuild, sized from the live count.** Rehashing reinserts only
   `FULL` slots, which drops every tombstone. Sizing from `len` rather than
   doubling `cap` means a table full of tombstones is rebuilt at the same size
   instead of growing forever.
6. **The map owns its keys.** `map_put` copies the key on first insert; an
   overwrite keeps the existing copy; delete and free release it. Rehashing moves
   the `char *` between slots without copying, so each key is allocated exactly
   once.

### The code, annotated

The probe is shared by `put`, `get`, and `del`. It returns whether the key was
found, and either way writes the slot to use:

```c
static int map_probe(const Map *m, const char *key, size_t *idx) {
  size_t mask = m->cap - 1;           /* cap is a power of two */
  size_t i = map_hash(key) & mask;
  size_t first_tomb = SIZE_MAX;       /* "none seen yet" */

  for (size_t step = 0; step < m->cap; step++) {
    Slot *s = &m->slots[i];
    if (s->state == SLOT_EMPTY) {
      /* The run ends here, so the key is absent. Prefer an earlier tombstone
       * for the insert, which keeps runs from growing under churn. */
      *idx = (first_tomb == SIZE_MAX) ? i : first_tomb;
      return 0;
    }
    if (s->state == SLOT_TOMB) {
      if (first_tomb == SIZE_MAX) first_tomb = i;   /* keep probing past it */
    } else if (strcmp(s->key, key) == 0) {
      *idx = i;
      return 1;
    }
    i = (i + 1) & mask;               /* wrap around without a modulo */
  }

  /* Table is entirely full or entirely tombstoned. */
  *idx = (first_tomb == SIZE_MAX) ? 0 : first_tomb;
  return 0;
}
```

The `step < m->cap` bound guarantees termination even if no slot is empty. The
resize policy in `map_put` keeps at least a quarter of the slots `EMPTY`, so in
practice the loop always ends at an empty slot or a match.

Insert decides on resizing *before* probing, because a resize invalidates any
index computed against the old table:

```c
int map_put(Map *m, const char *key, int value) {
  /* Resize at 3/4 counting tombstones. The new size comes from the live
   * count, so a table that tripped on tombstones alone is rebuilt at the same
   * capacity with the tombstones dropped. */
  if (m->cap == 0 || (m->used + 1) * 4 >= m->cap * 3) {
    if (map_resize(m, (m->len + 1) * 2) != 0) return -1;
  }

  size_t i;
  if (map_probe(m, key, &i)) {
    m->slots[i].value = value;   /* overwrite keeps the existing key copy */
    return 0;
  }

  char *copy = malloc(strlen(key) + 1);   /* the map owns its keys */
  if (!copy) return -1;
  memcpy(copy, key, strlen(key) + 1);

  /* Reusing a tombstone does not change `used`: it was already counted. */
  if (m->slots[i].state == SLOT_EMPTY) m->used++;
  m->slots[i].key = copy;
  m->slots[i].value = value;
  m->slots[i].state = SLOT_FULL;
  m->len++;
  return 0;
}
```

The resize reinserts live entries only, moving key ownership rather than
copying:

```c
static int map_resize(Map *m, size_t want) {
  size_t cap = 8;
  while (cap < want) {
    if (cap > SIZE_MAX / 2) return -1;
    cap *= 2;
  }

  Slot *slots = calloc(cap, sizeof(Slot));   /* calloc: every slot EMPTY (0) */
  if (!slots) return -1;

  for (size_t i = 0; i < m->cap; i++) {
    if (m->slots[i].state != SLOT_FULL) continue;   /* tombstones are dropped */
    size_t j = map_hash(m->slots[i].key) & (cap - 1);
    while (slots[j].state == SLOT_FULL) j = (j + 1) & (cap - 1);
    slots[j] = m->slots[i];                  /* moves the key pointer */
  }

  free(m->slots);
  m->slots = slots;
  m->cap = cap;
  m->used = m->len;                          /* no tombstones survive */
  return 0;
}
```

`calloc` is load-bearing: `SLOT_EMPTY` is `0` in the enum, so zeroed memory is a
table of empty slots with `NULL` keys. If the allocation fails, the old table is
untouched and the map is still valid.

Deletion is the tombstone write:

```c
int map_del(Map *m, const char *key) {
  if (m->cap == 0) return -1;
  size_t i;
  if (!map_probe(m, key, &i)) return -1;

  /* TOMB, not EMPTY: an EMPTY here would cut every probe run passing through
   * this slot and orphan entries that are still in the table. */
  free(m->slots[i].key);
  m->slots[i].key = NULL;
  m->slots[i].state = SLOT_TOMB;
  m->len--;          /* `used` is unchanged: the slot still costs a probe */
  return 0;
}
```

Full solution: [`map.c`](map.c).

### Why this solution

- **Cost.** `put`, `get`, `del`: O(1) expected while the load (counting
  tombstones) stays below 3/4, O(n) worst case when keys collide. A resize is
  O(cap) and happens after a number of inserts proportional to `cap`, so it is
  O(1) amortised. Space is `cap` slots plus one allocation per key.
- **Versus chaining.** Chaining makes deletion trivial but costs a heap node per
  entry and a pointer chase per probe. Open addressing keeps probes in one cache
  line or two, which is why it wins in practice despite the tombstone logic.
- **Linear probing versus quadratic or double hashing.** Linear probing clusters,
  but its probes are sequential memory reads, which is cheaper than the extra
  cache misses that more scattered probing incurs. With a good hash and a 3/4
  load cap, clustering stays tolerable.
- **Tombstones versus backward-shift deletion.** Linear probing allows deleting
  without tombstones by shifting later entries of the run back. That avoids
  tombstone build-up but makes deletion O(run length) and harder to get right.
  Tombstones plus "count them in the load" are simpler and correct.
- **Accepted trade.** FNV-1a has no seed, so the map is open to hash flooding.
  A production map exposed to untrusted keys needs a keyed hash such as SipHash.

### Explain it back

1. Why can't `map_del` set the slot to `SLOT_EMPTY`?

   <details><summary>Answer</summary>

   Lookups stop at the first empty slot. Any key that collided with the deleted
   one and was placed later in the same probe run would become unreachable,
   while still occupying a slot and owning its key copy.

   </details>

2. Why does the resize check use `used` rather than `len`?

   <details><summary>Answer</summary>

   Tombstones are skipped by lookups but still cost a probe step. Under
   insert-and-delete churn, `len` stays small while `used` climbs toward `cap`,
   and with no `EMPTY` slots left every miss scans the whole table. Counting
   `used` triggers a rebuild that drops the tombstones.

   </details>

3. Why is the new capacity computed from `(m->len + 1) * 2` instead of
   `m->cap * 2`?

   <details><summary>Answer</summary>

   A table can hit the threshold with few live entries and many tombstones.
   Doubling `cap` would grow memory without bound while holding nothing. Sizing
   from the live count rebuilds at the same size (or smaller) and only grows
   when there are genuinely more entries.

   </details>

4. When `map_put` reuses a tombstone, why does it not increment `used`?

   <details><summary>Answer</summary>

   `used` counts slots that are not `EMPTY`. A tombstone was already counted
   when it was first filled; turning it back into `FULL` changes which kind of
   non-empty slot it is, not how many there are.

   </details>

5. Why must the resize decision happen before `map_probe`, not after?

   <details><summary>Answer</summary>

   The probe returns an index into the current slot array. Resizing allocates a
   new array with a different mask, so an index computed before the resize would
   point at the wrong slot, or past the end of a smaller table.

   </details>

## Extending it

Try Robin Hood probing, where an insert that has travelled further than the
entry it finds steals the slot and displaces it. It flattens the variance of
probe lengths dramatically, which is the property you actually care about: the
worst lookup, not the average one. Rust's `std::collections::HashMap` used it
for years before switching to SwissTable. The course module `ds.02` builds a
Swiss table on top of this exercise (it is the worked baseline); its chapter
lives in [Systems Data Structures](../../../../../content/courses/algorithms/16-systems-data-structures/).
