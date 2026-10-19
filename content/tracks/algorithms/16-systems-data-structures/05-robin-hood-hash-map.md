# Robin Hood hash map with backward-shift deletion

## Overview

| | |
|---|---|
| **Module** | `ds.05` · build · Rust · Pass 3 · 2 to 3 h reading |
| **Solution** | [`rust/crates/tl-ds/src/robin.rs`](../../../../course/solutions/rust/crates/tl-ds/src/robin.rs): `FxHasher`, `FxBuild`, `slots_for`, `RobinHoodMap<K, V, S>` (`insert`, `get`, `get_mut`, `contains_key`, `remove`, `remove_entry`, `entry`, `reserve`, `iter`, `iter_mut`, `keys`, `values`, `len`, `capacity`, `slot_count`, `max_probe_len`, `layout`, `clear`; private `home`, `place`, `find`, `take_at`) and the `Entry` API · the crate root [`rust/crates/tl-ds/src/lib.rs`](../../../../course/solutions/rust/crates/tl-ds/src/lib.rs) (`pub mod bloom; pub mod heap; pub mod radix; pub mod robin;`) · model tests: [`ds05_robin.rs`](../../../../course/solutions/learner-tests/ds.05/rust/crates/tl-ds/tests/ds05_robin.rs) |
| **Contract** | the interface in section 4 (there is no separate contract file; the public API of `robin.rs` is the interface) |
| **Needs** | reading: `lang.04` the [Rust primer](../../software-craftsmanship/12-language-and-tool-primers/04-rust.md) (ownership, traits, generics) · `S-M06a` [load factor and expected probe length](../../math/06-discrete-math-2/90-problem-set-a.md) |
| **Used by** | `L1.5` keeps its vocabulary (`bytes -> id`) and merge table (`(left, right) -> (rank, id)`) in `RobinHoodMap` (the Rust tokenizer encodes through this map) · later: `ds.02` (the C Swiss table) and `ds.07` (the radix tree) build on this chapter |
| **Optional depth** | Celis, *Robin Hood Hashing* (PhD thesis, 1986); Knuth, *The Art of Computer Programming*, vol. 3, section 6.4 (linear probing analysis); Cormen et al., *Introduction to Algorithms*, chapter 11 |

## Key Takeaways

- A hash table is an array of slots plus a function from keys to slots; open addressing stores every entry in the array itself and walks forward on a collision.
- Robin Hood insertion lets an entry that has travelled further from its home take the slot of one that has travelled less, which keeps the longest probe short at load 7/8 where plain linear probing builds runs of hundreds.
- The same rule lets a lookup stop early: once the slot's displacement is smaller than the distance walked, the key is not in the table.
- Deletion shifts the following run back one slot instead of leaving a tombstone, so the table never slows down with churn.
- Fx multiplies by an odd constant, so its low bits are weak: the home slot comes from the top bits of the hash.

## How to read this chapter

Sections 1 to 3 build open addressing, the Fx hash, and the Robin Hood and backward-shift rules from first principles, and work four keys by hand. Section 4 fixes the interface and lists what a correct map must handle. Section 5 walks through [`robin.rs`](../../../../course/solutions/rust/crates/tl-ds/src/robin.rs) in the order it was designed: `slots_for` and `home`, then `place` (insert), `find` (lookup), and `take_at` (delete), then the entry API. The crate `tl-ds` is std only (`proptest` is a dev-dependency for tests); its root [`lib.rs`](../../../../course/solutions/rust/crates/tl-ds/src/lib.rs) declares the `bloom`, `heap`, `radix`, and `robin` modules. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L1.2` built a byte-level BPE in Python that matches GPT-2 and SmolLM2 id for id. It is too slow to serve: the engine (`L10.1`) and the corpus pipeline (`data.07`) tokenize millions of strings, so the next module, `L1.5`, ports it to Rust. That tokenizer does two lookups for every symbol of every pre-token: "which id is this byte string?" (a 50,000-entry vocabulary) and "do these two adjacent ids merge, and at what rank?" (a 50,000-entry merge table). Python's `dict` answered both; in Rust the course builds the table itself. This is the course's first hash table, so this chapter starts from the array.

## 2. Principles

### 2.1 Keys, slots, and collisions

| Symbol | Meaning | Type |
|---|---|---|
| $n$ | number of entries | `usize` |
| $m$ | number of slots, a power of two | `usize` |
| $\alpha = n / m$ | load factor | real in $[0, 7/8]$ |
| $h(k)$ | the 64-bit hash of key $k$ | `u64` |
| $\text{home}(k)$ | the slot $h(k)$ picks: its top $\log_2 m$ bits | `usize` in $[0, m)$ |
| $d$ | displacement: how many slots past its home an entry sits (wrapping at $m$) | `usize` |

A **hash table** stores entries in an array of $m$ slots and computes, from each key, the slot to look in first: its **home**. Two different keys can have the same home (a **collision**); with $n$ keys and $m$ slots this is certain once $n > m$ and likely long before (the birthday bound of `S-M06a`). **Open addressing** resolves a collision inside the array: if the home is taken, try the next slot, then the next (**linear probing**), wrapping from the last slot to slot 0. The number of slots read to find an entry is its **probe length**, $d + 1$.

### 2.2 Hashing: Fx and the top bits

The hash must be fast (the tokenizer hashes every pair of every word) and must spread the keys the program actually uses. Keys here are token bytes and pairs of small integers that no attacker chooses, so the course uses **Fx**, the hash inside `rustc`: start from $h = 0$ and fold in each 64-bit word $w$ of the key with

$$h \leftarrow (\text{rotl}_5(h) \oplus w) \cdot \text{SEED} \bmod 2^{64}, \qquad \text{SEED} = \texttt{0x517cc1b727220a95}.$$

A byte string is folded as little-endian 8-byte words, then a 4-, 2-, and 1-byte tail, so every byte reaches the hash. Multiplication by an odd constant moves information only **upward**: bit $j$ of the product depends on bits $0..j$ of the input. The low bits of $h$ are therefore weak (keys that are multiples of $2^{20}$ all have $h \equiv 0 \pmod{2^{20}}$), and the top bits are strong. So the home slot is the **top** $\log_2 m$ bits: `hash >> (64 - log2(m))`. Rust's `Hash` trait feeds a value to a `Hasher` through `write_u32`, `write_u64`, `write(&[u8])`; a `BuildHasher` makes a fresh `Hasher` per key. `FxBuild` is the default `S` of the map, so two runs place the same keys in the same slots (std's `RandomState` reseeds every process).

### 2.3 Load factor and growth

Linear probing's expected probe lengths (Knuth, `S-M06a`) are about $\frac12\left(1 + \frac{1}{1-\alpha}\right)$ for a hit and $\frac12\left(1 + \frac{1}{(1-\alpha)^2}\right)$ for a miss: 4.5 and 32.5 slots at $\alpha = 7/8$. They blow up as $\alpha \to 1$, and a completely full table makes an insert of a new key probe forever. The map therefore grows **before** the load passes 7/8: `slots_for(n)` is the smallest power of two $m \ge 8$ with $8n \le 7m$, and an insert that would exceed it first doubles the table and re-inserts every entry at its new home.

### 2.4 Robin Hood: equal suffering

Average probe lengths are fine at 7/8; the **longest** run is not. With plain linear probing, an entry that arrives behind a long run walks the whole run, and runs merge into longer runs. Robin Hood insertion (Celis, 1986) carries the new entry from its home with displacement $d = 0$ and, at each occupied slot holding an entry with displacement $d'$:

- if $d' < d$, the resident is "richer" (closer to home): the carried entry takes the slot and the resident is picked up and carried on, keeping its own displacement;
- otherwise move on, $d \leftarrow d + 1$.

The carried entry lands in the first empty slot. Displacements stay even, so the longest probe at load 7/8 stays a few dozen slots (25 to 35 over typical random seeds) where linear probing's reaches hundreds. The rule keeps an **invariant**: along a run of occupied slots, displacement grows by at most 1 per slot, and an entry right after an empty slot sits at home. That gives lookups an early exit: walking from the home with distance $d$, if the slot holds an entry with displacement $< d$, the key would have taken that slot on insertion, so it is absent.

### 2.5 Backward-shift deletion

Emptying a slot would break the invariant (a lookup for an entry behind the hole would stop at the hole). The classic fix is a **tombstone** marker, which lookups skip; tombstones accumulate and slow everything until a rehash. Robin Hood allows better: after removing the entry, move each following entry with $d > 0$ back one slot, decrementing its displacement, until an empty slot or an entry already at home. The table then looks exactly as if the removed key had never been inserted.

### 2.6 The entry API

`map.entry(k)` hashes and probes once and returns either an occupied entry (a slot index) or a vacant one (the hash and the key). `or_insert`, `or_insert_with`, `or_default`, and `and_modify` then read or write without a second probe: counting words is `*m.entry(w).or_insert(0) += 1`. `entry` reserves room for one more key **before** it returns a vacant entry, so the insert never grows the table under the reference it hands out, and a vacant insert must return a reference to the slot where the **new** entry ends, which is not where the Robin Hood carry finishes when the new entry displaced someone.

## 3. Worked example by hand

Use the tests' `Id` hasher, whose hash is the key itself, and 8 slots, so the home of a key is its top 3 bits. Insert four keys: A (home 2), B (home 2), C (home 3), D (home 2).

| Step | Carried entry and displacement | Slots 2, 3, 4, 5 after the step |
|---|---|---|
| insert A | A at slot 2, $d = 0$, empty: lands | A(0), -, -, - |
| insert B | slot 2 holds A(0); $0 < 0$ is false, move on; slot 3 empty: B lands with $d = 1$ | A(0), B(1), -, - |
| insert C | home 3 holds B(1); $1 < 0$ false; slot 4 empty: C lands with $d = 1$ | A(0), B(1), C(1), - |
| insert D | slot 2: A(0), move on ($d = 1$); slot 3: B(1), $1 < 1$ false, move on ($d = 2$); slot 4: C(1), $1 < 2$: **D takes slot 4** with $d = 2$, C is carried on with $d = 1$; slot 5 is empty, C lands with $d = 2$ | A(0), B(1), D(2), C(2) |

The longest probe reads 3 slots (D and C, $d + 1 = 3$). Plain linear probing would have left C at slot 4 ($d = 1$) and D at slot 5 ($d = 3$): a probe of 4.

**Lookup of D.** Start at slot 2 with distance 0: A(0) is not D, $0 < 0$ is false; slot 3, distance 1: B(1), $1 < 1$ false; slot 4, distance 2: D. Found. A lookup for an absent key E with home 3 reads slot 3 (B, $1 < 0$ false), slot 4 (D, $2 < 1$ false), slot 5 (C, $2 < 2$ false), slot 6: empty, absent.

**Remove B.** Slot 3 is emptied; slot 4 holds D(2) with $d > 0$: it moves back to slot 3 as D(1); slot 5 holds C(2): it moves to slot 4 as C(1); slot 6 is empty, stop. Slots 2, 3, 4: A(0), D(1), C(1). Removing A next shifts the run again, to D(0), C(0) at slots 2 and 3.

**Fx by hand.** One u64 word $w = 1$ from $h = 0$: $\text{rotl}_5(0) \oplus 1 = 1$, times SEED is SEED. The bytes `ab` are the 2-byte tail word `0x6261` (little-endian: `a` = 0x61 is the low byte), so `hash("ab")` $= \texttt{0x6261} \cdot \text{SEED}$; `abc` folds `0x6261` and then the byte `0x63`.

## 4. The interface

```rust
// rust/crates/tl-ds/src/robin.rs
pub const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
pub const MAX_LOAD_NUM: usize = 7;  pub const MAX_LOAD_DEN: usize = 8;  pub const MIN_SLOTS: usize = 8;
#[derive(Clone, Copy, Debug, Default)] pub struct FxHasher { /* hash: u64 */ }   // impl Hasher
#[derive(Clone, Copy, Debug, Default)] pub struct FxBuild;                         // impl BuildHasher
pub fn slots_for(n: usize) -> usize;                     // 0 for 0, else the smallest power of two >= 8 with 8n <= 7m

pub struct RobinHoodMap<K, V, S = FxBuild> { /* slots: Vec<Option<Bucket>>, len, hasher */ }
impl<K, V> RobinHoodMap<K, V> { pub fn new() -> Self; pub fn with_capacity(n: usize) -> Self; }
impl<K, V, S> RobinHoodMap<K, V, S> {
    pub fn with_hasher(hasher: S) -> Self;
    pub fn with_capacity_and_hasher(n: usize, hasher: S) -> Self;
    pub fn len(&self) -> usize;  pub fn is_empty(&self) -> bool;
    pub fn capacity(&self) -> usize;          // 7/8 of the slots
    pub fn slot_count(&self) -> usize;        // 0 or a power of two
    pub fn max_probe_len(&self) -> usize;     // largest displacement + 1
    pub fn layout(&self) -> Vec<Option<(&K, usize)>>;   // per slot: key and displacement
    pub fn clear(&mut self);
    pub fn iter(&self) -> Iter<'_, K, V>;  pub fn iter_mut(&mut self) -> IterMut<'_, K, V>;
    pub fn keys(&self) -> Keys<'_, K, V>;  pub fn values(&self) -> Values<'_, K, V>;
}
impl<K: Hash + Eq, V, S: BuildHasher> RobinHoodMap<K, V, S> {
    pub fn reserve(&mut self, additional: usize);
    pub fn insert(&mut self, key: K, value: V) -> Option<V>;
    pub fn get<Q>(&self, key: &Q) -> Option<&V> where K: Borrow<Q>, Q: Hash + Eq + ?Sized;
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V> /* same bounds */;
    pub fn contains_key<Q>(&self, key: &Q) -> bool /* same bounds */;
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V> /* same bounds */;
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)> /* same bounds */;
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S>;
}
pub enum Entry<'a, K, V, S> { Occupied(OccupiedEntry<'a, K, V, S>), Vacant(VacantEntry<'a, K, V, S>) }
// Entry: or_insert, or_insert_with, or_default, and_modify, key
// OccupiedEntry: key, get, get_mut, into_mut, insert, remove;  VacantEntry: key, insert
// Iter is an ExactSizeIterator; the map is FromIterator, Extend, Debug, Clone.
```

`Borrow` lets a `RobinHoodMap<Vec<u8>, u32>` be queried with a `&[u8]` slice and a `RobinHoodMap<String, _>` with a `&str`, hashed identically, without allocating a key for the lookup. `layout` exists for testing and for the diagrams: production callers never need it.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example's Robin Hood insert | unit | the section 3 table: A(0), B(1), D(2), C(2) in slots 2 to 5; longest probe 3 | the rule is pinned before any random input |
| The hand example's backward shift | unit | removing B gives A(0), D(1), C(1); removing A shifts the run again | no tombstones: lookups stay fast under the tokenizer's rebuilds |
| Lookup finds every displaced key | boundary | every key of the worked table is found; absent keys of every home are not | early exit on the right comparison |
| Inserting an existing key replaces the value only | unit | updating D returns the old value; layout and `len` unchanged | merge tables loaded twice keep one entry |
| Removing an absent key, and removing twice | boundary | absent and repeated removes are no-ops; a new map allocates nothing | the empty table edge |
| Growth before the load passes 7/8 | boundary | `slots_for` at 0, 1, 7, 8, 14, 15, 896, 897; load never above 7/8 over 1,000 inserts | a full table would loop forever |
| `with_capacity` never grows | unit | `with_capacity(7)` is 8 slots, `with_capacity(8)` 16; 1,000 inserts into `with_capacity(1000)` never rehash | `L1.5` sizes its tables once from the file |
| Fx hash values by hand | unit | `write_u64(1)` is SEED; `ab` and `abc` by hand; strings of 0 to 12 equal bytes all differ | every byte of a token reaches the hash |
| The home slot takes the top bits | boundary | 2,000 keys that are multiples of $2^{20}$ keep the longest probe at most 16 | Fx's weak low bits never pick the slot |
| Probe length stays short at high load | property | at exactly 7/8 load over 65,536 slots, longest probe at most 64 and mean below 6 | the Robin Hood bound itself |
| The entry API counts words | unit | word counts; `or_default`, `and_modify`; a vacant insert that displaces C writes D, not C | the reference written through is the entry inserted |
| Borrowed keys look up without allocating | unit | `Vec<u8>` keys looked up by `&[u8]`, `String` keys by `&str`; `get_mut`, `remove_entry` | `L1.5` looks tokens up by slice |
| Iteration visits every entry once | unit | `iter`, `keys`, `values`, `iter_mut` each see 100 entries once; `clear` keeps the slots | building id-to-bytes tables from the map |
| Any `BuildHasher` works | unit | the map with std's `RandomState`, 500 inserts, 250 removes | callers may need a keyed hash |
| Agrees with `std::collections::HashMap` | differential | 30,000 random inserts, removes, updates, and gets agree after every step | the map is a map |
| The Robin Hood invariant holds after churn | property | every 100 operations: displacement matches the home, grows by at most 1 per slot, is 0 after an empty slot | the invariant the early exit relies on |
| `FxBuild` is the default | regression | `new()` uses Fx: two maps from the same keys iterate alike | a tokenizer built twice is identical |

A model integration test suite (`use tl_ds::robin::...` only: insert steals from the rich, remove shifts back, every key is found, grows past seven eighths, the hasher reads every byte, entry counts, matches `std::collections::HashMap`) is in [`ds05_robin.rs`](../../../../course/solutions/learner-tests/ds.05/rust/crates/tl-ds/tests/ds05_robin.rs).

## 5. The solution, step by step

### 5.1 From first principles

1. **Entries live in the slot array itself** (open addressing), so a lookup is a short forward scan over contiguous memory. That needs a free slot to exist and probes to stay short, hence a load bound: grow before $8n > 7m$. Powers of two make "next slot" a mask and "home" a shift.
2. **Fx's information flows upward** (section 2.2), so the slot index must come from the top bits: `hash >> (64 - log2 m)`. Taking `hash & mask` would put every multiple of $2^{20}$ in slot 0.
3. **Each entry must know its displacement** to apply the Robin Hood rule, so a bucket stores `dist`. It also stores the full `hash`: comparing hashes before keys skips most key compares, and a rehash can recompute homes without rehashing keys.
4. **Robin Hood insertion keeps displacements nondecreasing by at most 1 along a run.** The carried entry swaps with any resident that has travelled less (`cur.dist < b.dist`). In the worked example, D reaches slot 4 with $d = 2$ against C's 1 and takes the slot; C is carried on.
5. **That invariant gives the lookup's early exit.** Walking from home at distance $d$, a resident with `dist < d` means the key, had it been present, would have swapped in here. So `find` stops at an empty slot or at `b.dist < dist`, never at `<=` (D sits behind B at equal displacement).
6. **Deletion must leave the table as if the key were never inserted,** or the early exit would lie. Shifting each following entry with `dist > 0` back one slot, with its `dist` decremented, does exactly that, and stops at an empty slot or an entry at home (which could not move closer).
7. **The entry API must not probe twice and must not hand out a reference that a growth would invalidate,** so `entry` reserves before returning a vacant entry, and `place` reports where the *new* entry landed, not where the carry ended.

### 5.2 The code, annotated

Sizing and the home slot.

```rust
pub fn slots_for(n: usize) -> usize {
    if n == 0 {
        return 0;                         // a new map allocates nothing
    }
    let mut slots = MIN_SLOTS;
    while n * MAX_LOAD_DEN > slots * MAX_LOAD_NUM {   // strictly above 7/8: 7 keys fit in 8 slots
        slots *= 2;
    }
    slots
}

/// The home slot of `hash`: its top log2(slots) bits (Fx's strong bits).
fn home(&self, hash: u64) -> usize {
    let bits = self.slots.len().trailing_zeros();
    (hash >> (64 - bits)) as usize
}
```

The bucket and the hasher. Fx folds 8-byte words and then a 4-, 2-, 1-byte tail so that no byte of a short token is dropped.

```rust
struct Bucket<K, V> {
    hash: u64,   // kept: cheap pre-compare, and rehash without rehashing keys
    dist: usize, // displacement from home
    key: K,
    value: V,
}

impl FxHasher {
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}
// Hasher::write: 8-byte LE words, then a u32, u16, and u8 tail, each through add().
```

Insertion. `place` is the Robin Hood carry. It is used by `insert`, by `reserve` (rehash), and by the vacant entry, so there is one implementation of the rule. `landed` records the first swap: that slot holds the entry the caller asked to insert.

```rust
/// Puts `b` into the table from slot `idx`; the key must be absent and a
/// free slot must exist. Returns the slot where `b` itself ends.
fn place(&mut self, mut idx: usize, mut b: Bucket<K, V>) -> usize {
    let mask = self.mask();
    let mut landed = None;
    loop {
        match &mut self.slots[idx] {
            slot @ None => {
                *slot = Some(b);              // the carried entry (maybe a displaced one) lands
                self.len += 1;
                return landed.unwrap_or(idx); // the new entry's slot, not the carry's end
            }
            Some(cur) => {
                if cur.dist < b.dist {
                    // The resident is richer (closer to home): it moves on.
                    std::mem::swap(cur, &mut b);
                    landed.get_or_insert(idx); // only the first swap places the new entry
                }
            }
        }
        idx = (idx + 1) & mask;
        b.dist += 1;                          // whatever we carry is one slot further from home
    }
}

pub fn insert(&mut self, key: K, value: V) -> Option<V> {
    let hash = self.hash_of(&key);
    if let Some(idx) = self.find(hash, &key) {
        let b = self.slots[idx].as_mut().unwrap();
        return Some(std::mem::replace(&mut b.value, value)); // stored key kept, layout unchanged
    }
    self.reserve(1);                          // grow before placing: a free slot must exist
    let home = self.home(hash);
    self.place(home, Bucket { hash, dist: 0, key, value });
    None
}
```

Lookup. Two exits, both justified by the invariant: an empty slot, or a resident that has travelled less than the distance walked.

```rust
fn find<Q>(&self, hash: u64, key: &Q) -> Option<usize>
where K: Borrow<Q>, Q: Eq + ?Sized,
{
    if self.len == 0 {
        return None;                          // also avoids home() on an empty table
    }
    let mask = self.mask();
    let mut idx = self.home(hash);
    let mut dist = 0;
    loop {
        match &self.slots[idx] {
            None => return None,
            Some(b) => {
                if b.dist < dist {
                    return None;              // the key would have taken this slot
                }
                if b.hash == hash && b.key.borrow() == key {  // hash first: cheap reject
                    return Some(idx);
                }
            }
        }
        idx = (idx + 1) & mask;
        dist += 1;
    }
}
```

Deletion by backward shift. The loop takes the next slot's content out; if it can move closer to home (`dist > 0`) it goes into the hole, otherwise it is put back and the shift ends.

```rust
fn take_at(&mut self, idx: usize) -> Bucket<K, V> {
    let mask = self.mask();
    let out = self.slots[idx].take().expect("take_at: empty slot");
    self.len -= 1;
    let mut hole = idx;
    loop {
        let next = (hole + 1) & mask;
        match self.slots[next].take() {
            Some(mut b) if b.dist > 0 => {
                b.dist -= 1;                  // one slot closer to home: keep dist truthful
                self.slots[hole] = Some(b);
                hole = next;
            }
            other => {                        // empty, or an entry already at home
                self.slots[next] = other;
                break;
            }
        }
    }
    out
}
```

Growth. Rehashing resets every `dist` to 0 and re-places from the new home, using the stored hash.

```rust
pub fn reserve(&mut self, additional: usize) {
    let need = self.len + additional;
    if need * MAX_LOAD_DEN <= self.slots.len() * MAX_LOAD_NUM {
        return;                               // already fits at load <= 7/8
    }
    let new_slots = slots_for(need).max(self.slots.len() * 2);  // at least double: amortised O(1)
    let mut fresh = Vec::new();
    fresh.resize_with(new_slots, || None);
    let old = std::mem::replace(&mut self.slots, fresh);
    self.len = 0;                             // place() counts them back in
    for b in old.into_iter().flatten() {
        let home = self.home(b.hash);
        self.place(home, Bucket { dist: 0, ..b });
    }
}
```

The entry API. `entry` probes once; a vacant entry has already reserved, and its insert returns the slot `place` reports.

```rust
pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
    let hash = self.hash_of(&key);
    match self.find(hash, &key) {
        Some(idx) => Entry::Occupied(OccupiedEntry { map: self, idx }),
        None => {
            self.reserve(1);                  // grow now, before any reference is handed out
            Entry::Vacant(VacantEntry { map: self, hash, key })
        }
    }
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S> {
    pub fn insert(self, value: V) -> &'a mut V {
        let VacantEntry { map, hash, key } = self;
        let home = map.home(hash);
        let idx = map.place(home, Bucket { hash, dist: 0, key, value });
        &mut map.slots[idx].as_mut().unwrap().value   // the new entry, even if it displaced others
    }
}
```

Full solution: [`robin.rs`](../../../../course/solutions/rust/crates/tl-ds/src/robin.rs) (iterators, `Debug`, `Clone`, `FromIterator`, and `Extend` are in the same file).

### 5.3 Why this solution

- **Cost.** `get`, `insert`, and `remove` are expected $O(1)$ at load at most 7/8; the expected probe lengths are those of linear probing (section 2.3), and Robin Hood shrinks the variance, so the longest probe stays a few dozen slots instead of hundreds. A rehash is $O(m)$, amortised $O(1)$ per insert. Space: $m \le 8/7 \cdot 2n$ buckets, each a key, a value, a hash, and a displacement.
- **Against plain linear probing.** Same average, much worse tail, no early exit for misses, and deletion needs tombstones.
- **Against tombstone deletion.** Tombstones keep probes long until a rehash; backward shift costs a short run move per delete and leaves the table exactly as if the key had never existed.
- **Against chaining (`Vec<Vec<(K, V)>>` or linked buckets).** A heap allocation per bucket and a pointer chase per lookup; the tokenizer's millions of lookups want contiguous memory.
- **Against std's `HashMap` (SipHash, randomized).** It would work, but SipHash is slower than Fx on short keys, and its per-process seed makes iteration order differ between runs. The keys here are not attacker-chosen, so flood resistance buys nothing. The map still accepts any `BuildHasher` for callers that need it.
- **Tradeoffs accepted.** Storing the 64-bit hash costs 8 bytes per entry. Insert can move many entries in a long run. Iteration order is slot order: deterministic, but not insertion order.

### 5.4 Explain it back

**Why does `find` stop at `b.dist < dist` and not `b.dist <= dist`?**

<details><summary>Answer</summary>

An entry with the same displacement as the distance walked can legitimately sit before the key: in the worked table, B(1) is at slot 3 and D is behind it at slot 4. Insertion swaps only when the resident is strictly richer, so equal displacement does not prove absence. Only a strictly smaller one does: the key would have taken that slot.

</details>

**Why does `place` return `landed` (the first swap) instead of the slot where the loop ends?**

<details><summary>Answer</summary>

After the first swap the loop carries a different entry, the displaced resident. The slot it finally lands in holds that resident, not the new key. A vacant-entry insert must return a reference to the value just inserted, so `*m.entry(w).or_insert(0) += 1` increments `w`'s count and not someone else's.

</details>

**Why does backward shift stop at an entry with `dist == 0`?**

<details><summary>Answer</summary>

That entry is already at its home. Moving it back would put it before its home, where no probe for it starts, so it would become unreachable. An empty slot ends the run for the same reason: nothing after it was displaced through it.

</details>

**Why does `entry` call `reserve(1)` before returning a vacant entry rather than inside `VacantEntry::insert`?**

<details><summary>Answer</summary>

Either works for the insert itself, but reserving up front means `entry` has already done every mutation that can move entries before the caller holds the view, and the insert then only runs the Robin Hood carry. It also keeps "an insert never grows under a live reference" true by construction.

</details>

**What would happen with `home = hash & mask` and keys that are multiples of $2^{20}$?**

<details><summary>Answer</summary>

Fx is a multiply by an odd constant (after a rotate and xor), so the low 20 bits of the hash are 0 for every such key in a one-word hash. All of them would have home 0, forming one run as long as the table; probes would be $O(n)$. The top bits depend on every input bit, so the solution takes those.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Inverting the Robin Hood comparison (a richer entry steals from a poorer one) | lookups stop early and miss keys; displacements grow without bound | `place` swaps only when `cur.dist < b.dist` |
| Never swapping (plain linear probing with the Robin Hood lookup) | the early exit misses displaced keys; at 7/8 load the longest probe reaches hundreds | `place` swaps; `find`'s early exit relies on it |
| Emptying the slot on delete and shifting nothing | keys behind the hole become unreachable | `take_at` shifts the run back |
| Shifting entries back without decrementing their displacement | the invariant breaks; later lookups exit too early | `take_at` does `b.dist -= 1` on each moved entry |
| Exiting the lookup at displacement `<=` distance instead of `<` | D behind B in the worked table is "absent" | `find` returns `None` only for `b.dist < dist` |
| Taking the home slot from the low bits (`hash & mask`) | keys that are multiples of a power of two pile into slot 0 | `home` shifts the top bits down |
| Growing at load equal to 7/8 instead of above it | tables are twice the size the contract says; `with_capacity(7)` gives 16 slots | `slots_for` and `reserve` grow only when `8n > 7m` |
| Returning the slot where the carry ended from a vacant insert | `*entry(k).or_insert(0) += 1` increments a different key | `place` returns `landed`, the first swap's slot |
| Dropping the last odd byte in the hasher | `ab` and `abc` collide; every token that differs in its last byte shares a home | `write` folds a 4-, 2-, and 1-byte tail |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.04` | generics with trait bounds, `Option`, ownership of the moved entries |
| Back | `S-M06a` | the load factor and expected probe lengths of section 2.3 |
| Forward | `L1.5` | the vocabulary `RobinHoodMap<Vec<u8>, u32>` looked up by slice, the merge table `RobinHoodMap<(u32, u32), Merge>`, and the piece cache of `encode_batch` |
| Forward | `ds.02` | the Swiss table in C keeps open addressing and the 7/8 bound, and matches 8 slots at a time with control bytes |
| Forward | `ds.07` | the radix tree over token ids keys its children by id |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `RobinHoodMap` | [hashbrown](https://github.com/rust-lang/hashbrown) (std's `HashMap`) | SwissTable groups: 16 control bytes compared at once with SIMD, tombstones instead of shifts | `src/raw/mod.rs`; the C version is `ds.02` |
| `FxHasher` | [rustc-hash](https://github.com/rust-lang/rustc-hash) | the same hash, plus a newer variant with better low bits | `src/lib.rs` |
| backward-shift deletion | [Robin Hood hashing in Rust before 1.36](https://github.com/rust-lang/rust/blob/1.35.0/src/libstd/collections/hash/table.rs) | std's own map used exactly this scheme until hashbrown replaced it | `src/libstd/collections/hash/map.rs` (1.35) |
| per-process hashing | std's `RandomState` (SipHash 1-3) | resists hash flooding by attackers who choose keys | `std::collections::hash_map::RandomState` |
