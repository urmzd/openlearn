<!-- ol:module ds.05 -->
# Robin Hood hash map with backward-shift deletion

## Overview

| | |
|---|---|
| **Module** | `ds.05` · build · Rust · Pass 3 · 5 to 7 h |
| **You build** | `rust/crates/tl-ds/src/robin.rs`: `FxHasher`, `FxBuild`, `slots_for`, `RobinHoodMap<K, V, S>` (`insert`, `get`, `get_mut`, `contains_key`, `remove`, `remove_entry`, `entry`, `reserve`, `iter`, `iter_mut`, `keys`, `values`, `len`, `capacity`, `slot_count`, `max_probe_len`, `layout`, `clear`) and the `Entry` API · the crate root `rust/crates/tl-ds/src/lib.rs` (`pub mod robin; pub mod heap; pub mod bloom;`) |
| **Contract** | the interface in section 4 (no Rust trait file yet; the tests pin it) |
| **Tests** | `course/tests/rust/ds_05.rs`, 17 tests (what they check: section 4) · your own tests in `rust/crates/tl-ds/tests/ds05_robin.rs`, rung R2, graded by mutation (threshold 0.60) |
| **Needs** | reading: `lang.04` the [Rust primer](../../software-craftsmanship/12-language-and-tool-primers/04-rust.md) (ownership, traits, generics) · `S-M06a` [load factor and expected probe length](../../math/06-discrete-math-2/90-problem-set-a.md) |
| **Used by** | `L1.5` keeps its vocabulary (`bytes -> id`) and merge table (`(left, right) -> (rank, id)`) in `RobinHoodMap` · later: `ds.02` (the C Swiss table) and `ds.07` (the radix tree) build on this chapter |
| **Milestone** | `MS-L1` (the Rust tokenizer encodes through your map) |
| **Optional depth** | Celis, *Robin Hood Hashing* (PhD thesis, 1986); Knuth, *The Art of Computer Programming*, vol. 3, section 6.4 (linear probing analysis); Cormen et al., *Introduction to Algorithms*, chapter 11 |

## Key Takeaways

- A hash table is an array of slots plus a function from keys to slots; open addressing stores every entry in the array itself and walks forward on a collision (`hand_example_robin_hood_insert`).
- Robin Hood insertion lets an entry that has travelled further from its home take the slot of one that has travelled less, which keeps the longest probe short at load 7/8 where plain linear probing builds runs of hundreds (`probe_length_stays_short_at_high_load`).
- The same rule lets a lookup stop early: once the slot's displacement is smaller than the distance walked, the key is not in the table (`lookup_finds_every_displaced_key`).
- Deletion shifts the following run back one slot instead of leaving a tombstone, so the table never slows down with churn (`backward_shift_hand_example`, `differential_against_std_hashmap`).
- Fx multiplies by an odd constant, so its low bits are weak: the home slot comes from the top bits of the hash (`home_slot_takes_the_top_bits`).

## How to work this chapter

```bash
ol start ds.05              # stubs robin.rs and the tl-ds crate root; writes the crate manifest if absent
ol tests ds.05              # read the test catalog first
ol check ds.05              # exit code is the verdict; then grades your tests by mutation
ol diff  ds.05              # after passing: your code against the reference
```

`ol start` writes `rust/crates/tl-ds/Cargo.toml` (std only; `proptest` as a dev-dependency for your own tests) and adds `crates/tl-ds` to your workspace manifest if you have none. The crate root declares all three modules of the crate (`robin`, `heap`, `bloom`); the ones you have not started yet are stubs that compile. Start with `slots_for`, `home`, and `insert` on the worked example with the `Id` hasher of the tests, then `find`, then `remove`.

---

## 1. Why now

`L1.2` gave you a byte-level BPE in Python that matches GPT-2 and SmolLM2 id for id. It is too slow to serve: the engine (`L10.1`) and the corpus pipeline (`data.07`) tokenize millions of strings, so the next module, `L1.5`, ports it to Rust. That tokenizer does two lookups for every symbol of every pre-token: "which id is this byte string?" (a 50,000-entry vocabulary) and "do these two adjacent ids merge, and at what rank?" (a 50,000-entry merge table). Python's `dict` answered both; in Rust you build the table yourself. This is the course's first hash table, so this chapter starts from the array.

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

The carried entry lands in the first empty slot. Displacements stay even, so the longest probe at load 7/8 stays a few dozen slots (25 to 35 over the seeds the tests run) where linear probing's reaches hundreds. The rule keeps an **invariant**: along a run of occupied slots, displacement grows by at most 1 per slot, and an entry right after an empty slot sits at home. That gives lookups an early exit: walking from the home with distance $d$, if the slot holds an entry with displacement $< d$, the key would have taken that slot on insertion, so it is absent.

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

The longest probe reads 3 slots (D and C, $d + 1 = 3$). Plain linear probing would have left C at slot 4 ($d = 1$) and D at slot 5 ($d = 3$): a probe of 4. This is `hand_example_robin_hood_insert`.

**Lookup of D.** Start at slot 2 with distance 0: A(0) is not D, $0 < 0$ is false; slot 3, distance 1: B(1), $1 < 1$ false; slot 4, distance 2: D. Found. A lookup for an absent key E with home 3 reads slot 3 (B, $1 < 0$ false), slot 4 (D, $2 < 1$ false), slot 5 (C, $2 < 2$ false), slot 6: empty, absent.

**Remove B.** Slot 3 is emptied; slot 4 holds D(2) with $d > 0$: it moves back to slot 3 as D(1); slot 5 holds C(2): it moves to slot 4 as C(1); slot 6 is empty, stop. Slots 2, 3, 4: A(0), D(1), C(1). This is `backward_shift_hand_example`; it then removes A, and the run shifts again to D(0), C(0) at slots 2 and 3.

**Fx by hand.** One u64 word $w = 1$ from $h = 0$: $\text{rotl}_5(0) \oplus 1 = 1$, times SEED is SEED. The bytes `ab` are the 2-byte tail word `0x6261` (little-endian: `a` = 0x61 is the low byte), so `hash("ab")` $= \texttt{0x6261} \cdot \text{SEED}$; `abc` folds `0x6261` and then the byte `0x63`. These are the first cases of `fx_hash_hand_values`.

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

`Borrow` lets a `RobinHoodMap<Vec<u8>, u32>` be queried with a `&[u8]` slice and a `RobinHoodMap<String, _>` with a `&str`, hashed identically, without allocating a key for the lookup. `layout` exists for the tests and the diagrams: production callers never need it.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_robin_hood_insert` | unit | the section 3 table: A(0), B(1), D(2), C(2) in slots 2 to 5; longest probe 3 | you and the tests agree on the rule before any random input |
| `backward_shift_hand_example` | unit | removing B gives A(0), D(1), C(1); removing A shifts the run again | no tombstones: lookups stay fast under the tokenizer's rebuilds |
| `lookup_finds_every_displaced_key` | boundary | every key of the worked table is found; absent keys of every home are not | early exit on the right comparison |
| `insert_existing_key_replaces_value_only` | unit | updating D returns the old value; layout and `len` unchanged | merge tables loaded twice keep one entry |
| `remove_absent_and_twice` | boundary | absent and repeated removes are no-ops; a new map allocates nothing | the empty table edge |
| `grows_before_load_passes_seven_eighths` | boundary | `slots_for` at 0, 1, 7, 8, 14, 15, 896, 897; load never above 7/8 over 1,000 inserts | a full table would loop forever |
| `with_capacity_never_grows` | unit | `with_capacity(7)` is 8 slots, `with_capacity(8)` 16; 1,000 inserts into `with_capacity(1000)` never rehash | `L1.5` sizes its tables once from the file |
| `fx_hash_hand_values` | unit | `write_u64(1)` is SEED; `ab` and `abc` by hand; strings of 0 to 12 equal bytes all differ | every byte of a token reaches the hash |
| `home_slot_takes_the_top_bits` | boundary | 2,000 keys that are multiples of $2^{20}$ keep the longest probe at most 16 | Fx's weak low bits never pick the slot |
| `probe_length_stays_short_at_high_load` | property | at exactly 7/8 load over 65,536 slots, longest probe at most 64 and mean below 6 | the Robin Hood bound itself |
| `entry_counts_words` | unit | word counts; `or_default`, `and_modify`; a vacant insert that displaces C writes D, not C | the reference you write through is the entry you inserted |
| `borrowed_keys_look_up_without_allocating` | unit | `Vec<u8>` keys looked up by `&[u8]`, `String` keys by `&str`; `get_mut`, `remove_entry` | `L1.5` looks tokens up by slice |
| `iteration_visits_every_entry_once` | unit | `iter`, `keys`, `values`, `iter_mut` each see 100 entries once; `clear` keeps the slots | building id-to-bytes tables from the map |
| `any_build_hasher_works` | unit | the map with std's `RandomState`, 500 inserts, 250 removes | callers may need a keyed hash |
| `differential_against_std_hashmap` | differential | 30,000 random inserts, removes, updates, and gets agree with `std::collections::HashMap` after every step | the map is a map |
| `robin_hood_invariant_holds_after_churn` | property | every 100 operations: displacement matches the home, grows by at most 1 per slot, is 0 after an empty slot | the invariant the early exit relies on |
| `fx_build_is_the_default` | regression | `new()` uses Fx: two maps from the same keys iterate alike | a tokenizer built twice is identical |

**Your tests (rung R2).** Write `rust/crates/tl-ds/tests/ds05_robin.rs` as an integration test (`use tl_ds::robin::...` only) with these tests, bodies yours: `insert_steals_from_the_rich`, `remove_shifts_back`, `every_key_is_found`, `grows_past_seven_eighths`, `hasher_reads_every_byte`, `entry_counts`, `matches_std_hashmap`. `ol check ds.05` runs them against the reference with one planted bug at a time; at least 60% of the planted bugs must make one of them fail.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Inverting the Robin Hood comparison (a richer entry steals from a poorer one) | lookups stop early and miss keys; displacements grow without bound | `hand_example_robin_hood_insert`, `robin_hood_invariant_holds_after_churn` (mutant `s01`) |
| Never swapping (plain linear probing with the Robin Hood lookup) | the early exit misses displaced keys; at 7/8 load the longest probe reaches hundreds | `probe_length_stays_short_at_high_load`, `hand_example_robin_hood_insert` (mutant `s02`) |
| Emptying the slot on delete and shifting nothing | keys behind the hole become unreachable | `backward_shift_hand_example`, `differential_against_std_hashmap` (mutant `s03`) |
| Shifting entries back without decrementing their displacement | the invariant breaks; later lookups exit too early | `backward_shift_hand_example`, `robin_hood_invariant_holds_after_churn` (mutant `s04`) |
| Exiting the lookup at displacement `<=` distance instead of `<` | D behind B in the worked table is "absent" | `lookup_finds_every_displaced_key` (mutant `s05`) |
| Taking the home slot from the low bits (`hash & mask`) | keys that are multiples of a power of two pile into slot 0 | `home_slot_takes_the_top_bits` (mutant `s06`) |
| Growing at load equal to 7/8 instead of above it | tables are twice the size the contract says; `with_capacity(7)` gives 16 slots | `grows_before_load_passes_seven_eighths`, `with_capacity_never_grows` (mutant `s07`) |
| Returning the slot where the carry ended from a vacant insert | `*entry(k).or_insert(0) += 1` increments a different key | `entry_counts_words` (mutant `s08`) |
| Dropping the last odd byte in the hasher | `ab` and `abc` collide; every token that differs in its last byte shares a home | `fx_hash_hand_values` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.04` | generics with trait bounds, `Option`, ownership of the moved entries |
| Back | `S-M06a` | the load factor and expected probe lengths of section 2.3 |
| Forward | `L1.5` | the vocabulary `RobinHoodMap<Vec<u8>, u32>` looked up by slice, the merge table `RobinHoodMap<(u32, u32), Merge>`, and the piece cache of `encode_batch` |
| Forward | `ds.02` | the Swiss table in C keeps open addressing and the 7/8 bound, and matches 8 slots at a time with control bytes |
| Forward | `ds.07` | the radix tree over token ids keys its children by id |

If you skip this module, `ol check L1.5` stops with `needs ds.05: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `RobinHoodMap` | [hashbrown](https://github.com/rust-lang/hashbrown) (std's `HashMap`) | SwissTable groups: 16 control bytes compared at once with SIMD, tombstones instead of shifts | `src/raw/mod.rs`; the C version is `ds.02` |
| `FxHasher` | [rustc-hash](https://github.com/rust-lang/rustc-hash) | the same hash, plus a newer variant with better low bits | `src/lib.rs` |
| backward-shift deletion | [Robin Hood hashing in Rust before 1.36](https://github.com/rust-lang/rust/blob/1.35.0/src/libstd/collections/hash/table.rs) | std's own map used exactly this scheme until hashbrown replaced it | `src/libstd/collections/hash/map.rs` (1.35) |
| per-process hashing | std's `RandomState` (SipHash 1-3) | resists hash flooding by attackers who choose keys | `std::collections::hash_map::RandomState` |
