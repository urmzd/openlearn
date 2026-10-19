# String interning

Compilers, interpreters, and JSON parsers compare the same few strings over and over: identifiers, keywords, field names. Each `strcmp` walks the bytes again. String interning removes that cost by keeping exactly one copy of each distinct string in a table and returning the same pointer every time you ask for it. Once strings are interned, `a == b` replaces `strcmp(a, b) == 0` everywhere downstream.

The design is a hash set keyed by content, and it is shaped almost entirely by one promise: every handle the table hands out stays valid until the table is freed, including across growth. That promise decides how entries are allocated, what the slots hold, and why the table can be simpler than a general map. This section builds on the [hash map with open addressing](../02-hash-map/README.md), whose probing scheme it reuses, and on the flexible-array-member technique from the [arena allocator](../08-arena-allocator/README.md).

> **Learning objectives**
> - Implement find-or-insert over an open-addressing table of entry pointers so that handles survive growth.
> - Lay out an entry with a flexible array member so header and bytes share one allocation.
> - Treat strings as (length, bytes) so embedded NULs and unterminated input are handled correctly.
> - Explain why dropping deletion removes tombstones, and when interning costs more than it saves.

## The interface

[`intern.h`](intern.h) declares the interface, [`intern.c`](intern.c) is the solution, and [`main.c`](main.c) is the test suite.

| Function | Does |
|----------|------|
| `intern_new()` | Empty table, `NULL` on allocation failure |
| `intern(t, s)` | Intern a C string, returning a stable handle |
| `intern_n(t, s, len)` | Intern arbitrary bytes, including embedded NULs |
| `intern_find(t, s)` | Look up without inserting |
| `intern_count(t)` | Distinct strings held |
| `intern_free(t)` | Release the table and every string in it |

A handle is `const Interned *`, where `Interned` is `{ const char *str; size_t len; }`. The hard constraint is that every handle stays valid until `intern_free`, including across growth of the table.

To build and run the tests from this directory:

```bash
cc -std=c11 -Wall -Wextra -O2 main.c intern.c -o intern && ./intern
# ok  c/09-string-interning  20160 checks passed
```

## Stable handles shape everything

Every pointer handed out must stay valid for the life of the table, and that constrains the whole design. Growth rehashes the table, so if the strings live in one big buffer that gets reallocated, every handle a caller is holding dangles. The fix is that entries are individually allocated and the table stores *pointers*: rehashing moves pointers between slots and never touches the string data. `test_pointers_survive_growth` interns 5,000 strings, forcing many rehashes, then checks that the very first handle is still correct.

Each entry is one allocation, not two. The header and the bytes go in a single `malloc` using a flexible array member, so the string cannot be separated from its metadata and the allocator sees half the traffic. This is the same technique as the arena allocator's blocks and worth having in reflex.

The entry also caches its hash. It costs one `size_t` and buys two things: growth never rehashes the text, and a probe can reject a colliding entry by comparing integers instead of calling `memcmp`. With long strings that share a prefix, which is exactly what identifier tables are full of, that is most of the lookup cost.

## What interning leaves out

There are no tombstones, because interning never deletes. The open-addressing hash map needed tombstones and a load factor that counted them, and all of that complexity exists to support removal. Drop removal from the interface and the table gets substantially simpler. It is worth noticing which of your data structure's complications are paying for a feature you are not using.

`strlen` cannot be the internal length. The table stores bytes, and `intern_n` must copy exactly the length it was given: `"a\0b"` and `"a"` have the same `strlen` and are different strings. It still writes a terminator after the bytes, so C consumers can treat `str` normally, but the length is what defines identity. `test_embedded_nuls` and `test_unterminated_input` pin both halves.

## Deriving the design

Six observations build the table.

Pointer equality needs exactly one copy per distinct string. So interning is "find or insert": look the bytes up, return the existing copy if present, otherwise make one. That is a hash set keyed by content.

Handles must not move, so entries must not move. Each entry is its own heap allocation, and the hash table's slots hold `Entry *`. Growing the table reallocates the slot array and moves pointers, never the entries.

The public handle lives inside the entry. Put `Interned pub` first in `Entry`, return `&entry->pub`, and the handle is as stable as the entry. The string bytes follow in a flexible array member, so `pub.str` points into the same allocation.

Identity is (length, bytes), not a C string. Hash and compare exactly `len` bytes. Store a terminator anyway, so `pub.str` is usable as a C string when there are no embedded NULs.

No deletion means no tombstones. A slot is either `NULL` (empty) or an entry, and a probe stops at the first `NULL`. The table uses linear probing with a power-of-two capacity and a 3/4 load limit, as in the hash map.

Finally, cache the hash. Compare hashes before lengths before bytes, and reuse the hash when rehashing.

## Building it step by step

The entry layout:

```c
typedef struct Entry {
  Interned pub;   /* must be first: the public pointer is &entry->pub */
  size_t hash;    /* cached: growth never rehashes the text */
  char data[];    /* the bytes, then a NUL, in the same allocation */
} Entry;

struct InternTable {
  Entry **slots;  /* open addressing, NULL where empty */
  size_t cap;     /* always a power of two, or 0 */
  size_t len;
};
```

The cheap-first comparison, and the probe built on it:

```c
static int entry_matches(const Entry *e, size_t hash, const char *s, size_t len) {
  /* Hash rejects nearly every collision, length most of the rest, and only
   * then are bytes compared. memcmp, not strcmp: embedded NULs count. */
  return e->hash == hash && e->pub.len == len && memcmp(e->pub.str, s, len) == 0;
}

static int table_probe(const InternTable *t, size_t hash, const char *s,
                       size_t len, size_t *idx) {
  size_t mask = t->cap - 1;
  size_t i = hash & mask;
  while (t->slots[i]) {              /* no tombstones: NULL ends the run */
    if (entry_matches(t->slots[i], hash, s, len)) {
      *idx = i;
      return 1;
    }
    i = (i + 1) & mask;
  }
  *idx = i;                          /* the free slot for an insert */
  return 0;
}
```

The loop has no step bound because the load limit guarantees a `NULL` slot
exists.

Growth moves pointers only:

```c
static int table_grow(InternTable *t) {
  size_t cap = t->cap ? t->cap * 2 : 16;
  Entry **slots = calloc(cap, sizeof(Entry *));
  if (!slots) return -1;

  /* Reinsert the POINTERS, never the strings: every handle handed out so far
   * keeps its address. The cached hash means no text is reread. */
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
```

Find-or-insert:

```c
const Interned *intern_n(InternTable *t, const char *s, size_t len) {
  if (t->cap == 0 && table_grow(t) != 0) return NULL;   /* first use */

  size_t hash = hash_bytes(s, len);
  size_t i;
  if (table_probe(t, hash, s, len, &i)) return &t->slots[i]->pub;  /* found */

  /* Grow at 3/4 load, then re-probe: i refers to the old slot array. */
  if ((t->len + 1) * 4 >= t->cap * 3) {
    if (table_grow(t) != 0) return NULL;
    table_probe(t, hash, s, len, &i);
  }

  /* Header, bytes and terminator in one allocation, size checked first. */
  if (len > SIZE_MAX - sizeof(Entry) - 1) return NULL;
  Entry *e = malloc(sizeof(Entry) + len + 1);
  if (!e) return NULL;

  memcpy(e->data, s, len);     /* exactly len bytes, never strlen */
  e->data[len] = '\0';         /* terminate even when the input was not */
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
```

Lookup first, grow second: a string that is already present never triggers a
resize. `intern_find` is the same probe without the insert, and `intern_free`
frees each entry, then the slot array, then the table.

The full solution is in [`intern.c`](intern.c).

## Why this design

`intern` is O(len) expected (hashing plus at most one `memcmp` in the common case), with O(1) amortised table work. Comparing two interned strings afterwards is O(1). Space is one allocation per distinct string plus the slot array.

One big string buffer would mean fewer allocations and better locality, but growing the buffer moves every string and breaks every handle. An [arena](../08-arena-allocator/README.md) that chains blocks instead of reallocating would give both stability and locality, and is what many compilers use.

Storing `Interned` by value in the slots would make handles point into the slot array, which moves on growth. Storing pointers is what makes the stability guarantee possible.

A general map like the [hash map with open addressing](../02-hash-map/README.md) supports deletion and pays for it with tombstones. Interning never deletes, so the simpler table is the right one.

The accepted trade is that nothing is ever freed before `intern_free`, so the table grows with every distinct input. That is fine for identifiers and dangerous for untrusted, unbounded input.

## What can go wrong

Interning is a space/time trade that can lose. You pay a hash and a comparison once per string to make every subsequent comparison free. That wins overwhelmingly for compiler identifiers, JSON keys, and log field names, where a small set of strings is compared constantly. It loses for strings you compare once, and it leaks by construction for unbounded input, since nothing is ever removed.

> **Warning:** A table interning attacker-supplied strings is an unbounded memory sink.

The implementation hazards are these.

| Mistake | Consequence |
|---------|-------------|
| Storing strings in one reallocated buffer | Every outstanding handle dangles after growth |
| Storing `Interned` by value in the slots | Handles point into the slot array, which moves |
| Using `strlen` or `strcmp` internally | `"a\0b"` and `"a"` collide; unterminated input is over-read |
| Writing to the first probe's index after growing | The index belongs to the old array and mask |
| `malloc(sizeof(Entry) + len + 1)` without an overflow check | The size wraps and `memcpy` writes past a tiny block |

## Where this leads

Try adding a per-entry reference count and an `intern_release`, and watch how much complexity re-enters: tombstones come back, pointer stability gets harder, and you have to decide what happens to a handle whose count hit zero. That is the usual reason production interners simply never free, or tie their lifetime to an [arena](../08-arena-allocator/README.md) that is reset wholesale.

## Key terms

- **String interning**: keeping one canonical copy of each distinct string so equality becomes pointer comparison.
- **Handle**: the stable pointer (`const Interned *`) the table returns for a string.
- **Find-or-insert**: the operation that returns an existing entry for a key or creates one.
- **Pointer stability**: the guarantee that an address handed out remains valid as the container grows.
- **Cached hash**: storing an entry's hash beside it to skip rehashing on growth and to reject mismatches cheaply.

## Summary

- Each distinct string is one heap entry (header, bytes, terminator) and the slots hold entry pointers, so growth never moves a handle.
- Identity is (length, bytes); `memcmp` and explicit lengths handle embedded NULs and unterminated input.
- Matches are tested hash first, then length, then bytes, and the cached hash makes rehashing free of text reads.
- With no deletion there are no tombstones, so a probe simply stops at the first `NULL`.
- Interning wins when the same strings are compared many times and is an unbounded leak for untrusted input.

## Review questions

1. Why does `table_grow` move `Entry *` values and never copy entries?

   <details><summary>Answer</summary>

   Callers hold `&entry->pub`, a pointer into the entry. If growth copied or
   reallocated entries, those pointers would dangle, and the "same text, same
   pointer" guarantee would break. Moving only the slot array's pointers leaves
   every entry where it is.

   </details>

2. Why is the allocation guarded by `len > SIZE_MAX - sizeof(Entry) - 1`?

   <details><summary>Answer</summary>

   The size passed to `malloc` is `sizeof(Entry) + len + 1`. For a huge `len`
   that sum wraps to a small number, `malloc` succeeds, and `memcpy` then writes
   `len` bytes past a tiny block. Rearranging the check so it subtracts from
   `SIZE_MAX` keeps every term in range.

   </details>

3. After growing, why does `intern_n` call `table_probe` again?

   <details><summary>Answer</summary>

   The index `i` from the first probe is a position in the old slot array with
   the old mask. The new array has a different size, so the key's free slot is
   somewhere else, and writing to the old `i` would be wrong or out of bounds.

   </details>

4. Why can the probe loop in this table run until it finds `NULL` with no step
   limit, when the [open-addressing hash map](../02-hash-map/README.md)'s probe needed one?

   <details><summary>Answer</summary>

   Here every non-`NULL` slot is a live entry and the table grows before it is
   3/4 full, so at least a quarter of the slots are always `NULL`. That hash map
   can contain tombstones, which a lookup must step over, so its code bounds the
   loop to be safe even if no `EMPTY` slot remains.

   </details>

5. `intern_n(t, "a\0b", 3)` and `intern(t, "a")`: same handle or different, and
   which lines decide it?

   <details><summary>Answer</summary>

   Different. `intern` passes `strlen("a") == 1`, the other passes 3. Their
   hashes differ (FNV-1a over 1 versus 3 bytes), and even on a collision
   `entry_matches` compares `pub.len` before `memcmp`, so the lengths alone keep
   them apart.

   </details>
