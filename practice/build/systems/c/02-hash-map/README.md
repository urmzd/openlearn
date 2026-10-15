# C 02: Hash map (open addressing)

**Concepts:** hash functions, probing, load factor
**Difficulty:** ⭐⭐⭐

A string-keyed hash map that stores entries in one flat array and resolves
collisions by walking to the next free slot, rather than by chaining to a linked
list. `map_hash` is given to you so that the tests can construct collisions on
purpose.

## The contract

`map.h` declares it, `main.c` tests it, you write `map.c`.

| Function | Does |
|----------|------|
| `map_init(m)` | Zero state, no allocation |
| `map_put(m, k, v)` | Insert or overwrite. Copies `k` |
| `map_get(m, k, &v)` | 0 and writes `v` if present, -1 if absent |
| `map_del(m, k)` | 0 if it was there, -1 if it never was |
| `map_len(m)` | Live entries, not counting tombstones |
| `map_free(m)` | Release the table and every key it owns |

The map owns its keys. Callers must be free to reuse or free their own buffer
the instant `map_put` returns.

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
to a linear scan while reporting a low load factor. That is what
`test_churn_does_not_explode` measures. Growing on live-plus-tombstones is what
reclaims them, because rehashing simply drops them.

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

## Extending it

Try Robin Hood probing, where an insert that has travelled further than the
entry it finds steals the slot and displaces it. It flattens the variance of
probe lengths dramatically, which is the property you actually care about: the
worst lookup, not the average one. Rust's `std::collections::HashMap` used it
for years before switching to SwissTable. The course module `ds.02` builds a
Swiss table on top of this exercise (it is the worked baseline); its chapter
lives in [Systems Data Structures](../../../../../content/tracks/algorithms/16-systems-data-structures/).
