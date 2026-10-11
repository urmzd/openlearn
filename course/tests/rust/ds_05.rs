//! ds.05 course tests: the Robin Hood hash map (tl-ds `robin`).
//!
//! Annotated exemplars (DESIGN 5.12). The hand examples use `Id`, a hasher
//! whose hash IS the u64 key, so a test picks every key's home slot: with 8
//! slots the home of key `h` is its top 3 bits, `h >> 61`. Random operations
//! come from the PCG32 transcribed below from contracts/spec/pcg32.md (the
//! frozen generator, D35), never from a crate.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};

use tl_ds::robin::{slots_for, FxBuild, FxHasher, RobinHoodMap, SEED};

// ---------------------------------------------------------------------------
// helpers

/// The frozen PCG32 (spec/pcg32.md): `pcg32_srandom_r(seed, seq)`.
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut g = Pcg32 { state: 0, inc: (seq << 1) | 1 };
        g.next_u32();
        g.state = g.state.wrapping_add(seed);
        g.next_u32();
        g
    }
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        xs.rotate_right((old >> 59) as u32)
    }
    fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | self.next_u32() as u64
    }
    fn below(&mut self, n: u32) -> u32 {
        let t = n.wrapping_neg() % n;
        loop {
            let r = self.next_u32();
            if r >= t {
                return r % n;
            }
        }
    }
}

fn ol_seed() -> u64 {
    std::env::var("OL_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

/// Hash = the key itself: `Id` hashes a u64 key to exactly that u64.
#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn write(&mut self, _: &[u8]) {
        panic!("Id hashes u64 keys only");
    }
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Default)]
struct Id;

impl BuildHasher for Id {
    type Hasher = IdHasher;
    fn build_hasher(&self) -> IdHasher {
        IdHasher(0)
    }
}

/// A key whose home slot in an 8-slot table is `home`; `tag` tells keys apart.
fn key(home: u64, tag: u64) -> u64 {
    (home << 61) | tag
}

const A: u64 = (2 << 61) | 1;
const B: u64 = (2 << 61) | 2;
const C: u64 = (3 << 61) | 3;
const D: u64 = (2 << 61) | 4;

/// The chapter's worked table: A, B, C, D inserted in that order.
fn worked() -> RobinHoodMap<u64, &'static str, Id> {
    let mut m = RobinHoodMap::with_capacity_and_hasher(4, Id);
    for (k, v) in [(A, "a"), (B, "b"), (C, "c"), (D, "d")] {
        assert_eq!(m.insert(k, v), None);
    }
    m
}

/// (slot, key, displacement) of every occupied slot.
fn occupied<V, S>(m: &RobinHoodMap<u64, V, S>) -> Vec<(usize, u64, usize)> {
    m.layout()
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|(k, d)| (i, *k, d)))
        .collect()
}

fn fx<T: Hash + ?Sized>(x: &T) -> u64 {
    let mut h = FxHasher::default();
    x.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// the worked example

#[test]
fn hand_example_robin_hood_insert() {
    // WHY: section 3 by hand. A and B share home 2, C's home is 3, D's is 2.
    //      D reaches slot 4 having travelled 2, where C has travelled only 1:
    //      D takes the slot and C moves on to 5. Linear probing would leave
    //      D at slot 5 with displacement 3.
    // KIND: unit
    // CATCHES: s01, s02
    // CHAPTER: ds.05 section 3, Worked example by hand
    let m = worked();
    assert_eq!(m.slot_count(), 8, "4 entries fit in 8 slots at load 1/2");
    assert_eq!(occupied(&m), vec![(2, A, 0), (3, B, 1), (4, D, 2), (5, C, 2)]);
    assert_eq!(m.max_probe_len(), 3, "the longest successful probe reads 3 slots");
    assert_eq!(m.len(), 4);
}

#[test]
fn backward_shift_hand_example() {
    // WHY: removing B leaves a hole at slot 3. Backward shift moves D and C
    //      back one slot each (displacements 1 and 1) and stops at the empty
    //      slot 6. No tombstone is left, so a later lookup of C still stops
    //      where the Robin Hood rule says it must.
    // KIND: unit
    // CATCHES: s03, s04
    // CHAPTER: ds.05 section 3, Worked example by hand
    let mut m = worked();
    assert_eq!(m.remove(&B), Some("b"));
    assert_eq!(occupied(&m), vec![(2, A, 0), (3, D, 1), (4, C, 1)]);
    assert_eq!(m.get(&C), Some(&"c"));
    assert_eq!(m.get(&D), Some(&"d"));
    assert_eq!(m.get(&B), None);
    assert_eq!(m.len(), 3);
    // Removing the entry at its home slot shifts the whole run back.
    assert_eq!(m.remove(&A), Some("a"));
    assert_eq!(occupied(&m), vec![(2, D, 0), (3, C, 0)]);
}

#[test]
fn lookup_finds_every_displaced_key() {
    // WHY: a lookup may stop early only at a slot whose displacement is
    //      SMALLER than the distance walked. Stopping at "smaller or equal"
    //      misses keys that sit behind an equally displaced neighbour, like
    //      D behind B in the worked table.
    // KIND: boundary
    // CATCHES: s05, m04
    // CHAPTER: ds.05 section 2.4
    let m = worked();
    for (k, v) in [(A, "a"), (B, "b"), (C, "c"), (D, "d")] {
        assert_eq!(m.get(&k), Some(&v), "key with home {}", k >> 61);
        assert!(m.contains_key(&k));
    }
    // Absent keys with every home slot.
    for home in 0..8 {
        assert_eq!(m.get(&key(home, 99)), None);
    }
}

#[test]
fn insert_existing_key_replaces_value_only() {
    // WHY: insert on a present key returns the old value and changes nothing
    //      else: len stays, the slot stays, the stored key is kept.
    // KIND: unit
    // CATCHES: m01
    // CHAPTER: ds.05 section 4
    let mut m = worked();
    assert_eq!(m.insert(D, "D"), Some("d"));
    assert_eq!(m.len(), 4);
    assert_eq!(occupied(&m), vec![(2, A, 0), (3, B, 1), (4, D, 2), (5, C, 2)]);
    assert_eq!(m.get(&D), Some(&"D"));
}

#[test]
fn remove_absent_and_twice() {
    // WHY: removing a key that is not there returns None and leaves the table
    //      untouched; the second remove of the same key is a no-op.
    // KIND: boundary
    // CATCHES: m02
    // CHAPTER: ds.05 section 4
    let mut m = worked();
    assert_eq!(m.remove(&key(6, 7)), None);
    assert_eq!(m.len(), 4);
    assert_eq!(m.remove(&C), Some("c"));
    assert_eq!(m.remove(&C), None);
    assert_eq!(m.len(), 3);
    let mut e: RobinHoodMap<u64, u8> = RobinHoodMap::new();
    assert_eq!(e.remove(&1), None);
    assert_eq!(e.get(&1), None);
    assert!(e.is_empty());
    assert_eq!(e.slot_count(), 0, "a new map allocates nothing");
}

// ---------------------------------------------------------------------------
// load factor and growth

#[test]
fn grows_before_load_passes_seven_eighths() {
    // WHY: the table holds at most 7 entries per 8 slots. slots_for(7) is 8
    //      and slots_for(8) is 16: the eighth insert into 8 slots doubles the
    //      table first. A full table would make an insert of a new key probe
    //      forever.
    // KIND: boundary
    // CATCHES: s07
    // CHAPTER: ds.05 section 2.3
    assert_eq!((slots_for(0), slots_for(1), slots_for(7), slots_for(8)), (0, 8, 8, 16));
    assert_eq!((slots_for(14), slots_for(15), slots_for(896), slots_for(897)), (16, 32, 1024, 2048));
    let mut m: RobinHoodMap<u32, u32> = RobinHoodMap::new();
    for i in 0..7 {
        m.insert(i, i);
    }
    assert_eq!((m.slot_count(), m.capacity()), (8, 7));
    m.insert(7, 7);
    assert_eq!((m.slot_count(), m.capacity()), (16, 14));
    for i in 8..1000 {
        m.insert(i, i);
        assert!(m.len() * 8 <= m.slot_count() * 7, "load above 7/8 at len {}", m.len());
    }
    for i in 0..1000 {
        assert_eq!(m.get(&i), Some(&i));
    }
}

#[test]
fn with_capacity_never_grows() {
    // WHY: L1.5 sizes its vocabulary map once from the file's token count;
    //      with_capacity(n) must hold n entries with no rehash.
    // KIND: unit
    // CATCHES: s07
    // CHAPTER: ds.05 section 4
    assert_eq!(RobinHoodMap::<u8, u8>::with_capacity(7).slot_count(), 8);
    assert_eq!(RobinHoodMap::<u8, u8>::with_capacity(8).slot_count(), 16);
    let mut m: RobinHoodMap<u64, u64> = RobinHoodMap::with_capacity(1000);
    let slots = m.slot_count();
    assert_eq!(slots, 2048);
    for i in 0..1000u64 {
        m.insert(i * 7919, i);
    }
    assert_eq!(m.slot_count(), slots);
    m.reserve(0);
    assert_eq!(m.slot_count(), slots);
}

// ---------------------------------------------------------------------------
// hashing

#[test]
fn fx_hash_hand_values() {
    // WHY: Fx folds each word in with h = (rotl(h, 5) ^ word) * SEED. From
    //      h = 0, one u64 word w gives w * SEED; the bytes "ab" are one u16
    //      word 0x6261 (little-endian); "abc" is that word, then the byte c.
    //      Every byte must reach the hash, or "ab" and "abc" collide.
    // KIND: unit
    // CATCHES: s09
    // CHAPTER: ds.05 section 2.2
    let mut h = FxHasher::default();
    h.write_u64(1);
    assert_eq!(h.finish(), SEED);
    let mut h = FxHasher::default();
    h.write(b"ab");
    assert_eq!(h.finish(), 0x6261u64.wrapping_mul(SEED));
    let mut h = FxHasher::default();
    h.write(b"abc");
    let first = 0x6261u64.wrapping_mul(SEED);
    assert_eq!(h.finish(), (first.rotate_left(5) ^ 0x63).wrapping_mul(SEED));
    // Every byte reaches the hash: all strings of length 0 to 12 over a
    // single byte value hash differently.
    let mut seen = std::collections::HashSet::new();
    for n in 0..=12 {
        let mut h = FxHasher::default();
        h.write(&vec![0x41u8; n]);
        assert!(seen.insert(h.finish()), "length {n} collides");
    }
}

#[test]
fn home_slot_takes_the_top_bits() {
    // WHY: Fx multiplies by an odd constant, so the LOW bits of the hash
    //      depend only on the low bits of the key. Keys that are multiples of
    //      2^20 (byte strings packed into words often are) then share their
    //      low 20 bits, and a table indexed by `hash & mask` piles every one
    //      of them into slot 0. The top bits mix every input bit.
    // KIND: boundary
    // CATCHES: s06
    // CHAPTER: ds.05 section 5, Pitfalls
    let mut m: RobinHoodMap<u64, u64> = RobinHoodMap::new();
    for i in 0..2000u64 {
        m.insert(i << 20, i);
    }
    assert!(m.max_probe_len() <= 16, "max probe length {} with top-bit homes", m.max_probe_len());
    assert_eq!(fx(&(5u64 << 20)) >> 61, ((5u64 << 20).wrapping_mul(SEED)) >> 61);
}

#[test]
fn probe_length_stays_short_at_high_load() {
    // WHY: the point of Robin Hood: at load 7/8 the mean probe is the same
    //      as linear probing's (about 4.5 slots), but the LONGEST probe stays
    //      short (25 to 35 slots over seeds 0 to 3), where linear probing's
    //      longest run reaches hundreds. 57,344 random keys fill 65,536 slots
    //      to exactly 7/8.
    // KIND: property
    // CATCHES: s02
    // CHAPTER: ds.05 section 2.4
    let mut g = Pcg32::new(ol_seed(), 5);
    let mut m: RobinHoodMap<u64, ()> = RobinHoodMap::with_capacity(57_344);
    assert_eq!(m.slot_count(), 65_536);
    while m.len() < 57_344 {
        m.insert(g.next_u64(), ());
    }
    assert_eq!(m.slot_count(), 65_536);
    let worst = m.max_probe_len();
    assert!(worst <= 64, "max probe length {worst} at load 7/8");
    let total: usize = m.layout().iter().flatten().map(|(_, d)| d + 1).sum();
    let mean = total as f64 / m.len() as f64;
    assert!(mean < 6.0, "mean probe length {mean:.2} at load 7/8");
}

// ---------------------------------------------------------------------------
// entry API, iteration, generic keys

#[test]
fn entry_counts_words() {
    // WHY: the entry API does one hash and one probe for read-modify-write,
    //      which is how L1.5 counts and how the chapter counts words. The
    //      reference a vacant insert returns must point at the slot the new
    //      entry ENDS in, even when the Robin Hood rule displaced another
    //      entry to put it there.
    // KIND: unit
    // CATCHES: s08, m03
    // CHAPTER: ds.05 section 4
    let mut m: RobinHoodMap<&str, u32> = RobinHoodMap::new();
    for w in "the cat saw the dog and the cat ran".split(' ') {
        *m.entry(w).or_insert(0) += 1;
    }
    assert_eq!(m.get("the"), Some(&3));
    assert_eq!(m.get("cat"), Some(&2));
    assert_eq!(m.get("ran"), Some(&1));
    assert_eq!(m.len(), 6);
    let mut n: RobinHoodMap<&str, Vec<u8>> = RobinHoodMap::new();
    n.entry("x").or_default().push(1);
    n.entry("x").or_default().push(2);
    n.entry("y").and_modify(|v| v.push(9)).or_insert_with(Vec::new);
    n.entry("x").and_modify(|v| v.push(3)).or_insert_with(Vec::new);
    assert_eq!(n.get("x"), Some(&vec![1, 2, 3]));
    assert_eq!(n.get("y"), Some(&vec![]));
    // A vacant insert that steals a slot: D displaces C in the worked table.
    let mut w: RobinHoodMap<u64, u32, Id> = RobinHoodMap::with_capacity_and_hasher(4, Id);
    for k in [A, B, C] {
        w.insert(k, 0);
    }
    *w.entry(D).or_insert(10) += 5;
    assert_eq!(w.get(&D), Some(&15));
    assert_eq!(w.get(&C), Some(&0), "C was moved, not written");
    match w.entry(C) {
        tl_ds::robin::Entry::Occupied(o) => assert_eq!(o.remove(), 0),
        tl_ds::robin::Entry::Vacant(_) => panic!("C is present"),
    }
    assert_eq!(occupied(&w), vec![(2, A, 0), (3, B, 1), (4, D, 2)]);
}

#[test]
fn borrowed_keys_look_up_without_allocating() {
    // WHY: L1.5's vocabulary maps Vec<u8> to ids and looks up &[u8] slices;
    //      Borrow lets get take the borrowed form, hashed identically.
    // KIND: unit
    // CATCHES: s05
    // CHAPTER: ds.05 section 4
    let mut m: RobinHoodMap<Vec<u8>, u32> = RobinHoodMap::new();
    m.insert(b"hello".to_vec(), 1);
    m.insert(b"\xc4\xa0the".to_vec(), 2);
    assert_eq!(m.get(&b"hello"[..]), Some(&1));
    assert_eq!(m.get(&b"\xc4\xa0the"[..]), Some(&2));
    assert_eq!(m.get(&b"hell"[..]), None);
    let mut s: RobinHoodMap<String, u32> = RobinHoodMap::new();
    s.insert("a".to_string(), 7);
    assert_eq!(s.get("a"), Some(&7));
    *s.get_mut("a").unwrap() += 1;
    assert_eq!(s.remove_entry("a"), Some(("a".to_string(), 8)));
}

#[test]
fn iteration_visits_every_entry_once() {
    // WHY: iter, keys, values, and iter_mut walk the slots in order and
    //      yield each entry exactly once; clear empties the map and keeps
    //      the table.
    // KIND: unit
    // CATCHES: m05, m06
    // CHAPTER: ds.05 section 4
    let mut m: RobinHoodMap<u32, u32> = (0..100u32).map(|i| (i, i * i)).collect();
    assert_eq!(m.iter().len(), 100);
    let mut keys: Vec<u32> = m.keys().copied().collect();
    keys.sort_unstable();
    assert_eq!(keys, (0..100).collect::<Vec<_>>());
    assert_eq!(m.values().map(|&v| v as u64).sum::<u64>(), (0..100u64).map(|i| i * i).sum());
    for (_, v) in m.iter_mut() {
        *v += 1;
    }
    assert_eq!(m.get(&9), Some(&82));
    let slots = m.slot_count();
    m.clear();
    assert!(m.is_empty() && m.iter().next().is_none());
    assert_eq!(m.slot_count(), slots);
    m.insert(3, 3);
    assert_eq!(m.len(), 1);
}

#[test]
fn any_build_hasher_works() {
    // WHY: S is a parameter: the map works with std's RandomState (keys an
    //      attacker chooses) as well as with Fx.
    // KIND: unit
    // CATCHES: s03, m02
    // CHAPTER: ds.05 section 4
    let mut m: RobinHoodMap<String, usize, std::collections::hash_map::RandomState> =
        RobinHoodMap::with_hasher(Default::default());
    for i in 0..500 {
        m.insert(format!("k{i}"), i);
    }
    for i in (0..500).step_by(2) {
        assert_eq!(m.remove(&format!("k{i}")), Some(i));
    }
    assert_eq!(m.len(), 250);
    assert_eq!(m.get("k7"), Some(&7));
    assert_eq!(m.get("k8"), None);
}

// ---------------------------------------------------------------------------
// the model

#[test]
fn differential_against_std_hashmap() {
    // WHY: 30,000 random inserts, removes, updates, and lookups over a small
    //      key space (so keys collide, return, and leave) agree with
    //      std::collections::HashMap after every operation; the final
    //      contents agree as sets. Interleaved removes exercise backward
    //      shift in every position of a run.
    // KIND: differential
    // CATCHES: s01, s03, s04, s05, m01, m02
    // CHAPTER: ds.05 section 4
    let mut g = Pcg32::new(ol_seed(), 5);
    let mut ours: RobinHoodMap<u32, u32> = RobinHoodMap::new();
    let mut std_: HashMap<u32, u32> = HashMap::new();
    for step in 0..30_000u32 {
        let k = g.below(600);
        match g.below(8) {
            0..=3 => assert_eq!(ours.insert(k, step), std_.insert(k, step), "insert {k} at step {step}"),
            4..=5 => assert_eq!(ours.remove(&k), std_.remove(&k), "remove {k} at step {step}"),
            6 => {
                if let Some(v) = ours.get_mut(&k) {
                    *v ^= 1;
                }
                if let Some(v) = std_.get_mut(&k) {
                    *v ^= 1;
                }
            }
            _ => assert_eq!(ours.get(&k), std_.get(&k), "get {k} at step {step}"),
        }
        assert_eq!(ours.len(), std_.len(), "len at step {step}");
    }
    let mut a: Vec<(u32, u32)> = ours.iter().map(|(k, v)| (*k, *v)).collect();
    let mut b: Vec<(u32, u32)> = std_.into_iter().collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b);
}

#[test]
fn robin_hood_invariant_holds_after_churn() {
    // WHY: the invariant behind early termination: along a run of occupied
    //      slots, a displacement grows by at most 1 per slot, and an entry
    //      right after an empty slot is at home. Checked on the layout after
    //      every 100 operations of random churn with the Id hasher (so homes
    //      are chosen, and runs wrap around the end of the table).
    // KIND: property
    // CATCHES: s01, s03, s04
    // CHAPTER: ds.05 section 2.4
    let mut g = Pcg32::new(ol_seed(), 5);
    let mut m: RobinHoodMap<u64, (), Id> = RobinHoodMap::with_hasher(Id);
    for step in 0..6000 {
        let k = (g.next_u64() & !0xFFFF) | g.below(64) as u64; // 64 keys, random homes
        let k = k & 0xF800_0000_0000_003F; // 32 home buckets of the top 5 bits
        if g.below(3) == 0 {
            m.remove(&k);
        } else {
            m.insert(k, ());
        }
        if step % 100 == 0 {
            let lay = m.layout();
            let n = lay.len();
            for i in 0..n {
                let prev = lay[(i + n - 1) % n];
                if let Some((k, d)) = lay[i] {
                    let bits = n.trailing_zeros();
                    let home = (*k >> (64 - bits)) as usize;
                    assert_eq!((home + d) % n, i, "slot {i}: displacement {d} does not match home {home}");
                    match prev {
                        None => assert_eq!(d, 0, "slot {i} follows an empty slot but has displacement {d}"),
                        Some((_, pd)) => assert!(d <= pd + 1, "slot {i}: displacement {d} after {pd}"),
                    }
                }
            }
        }
    }
}

#[test]
fn fx_build_is_the_default() {
    // WHY: RobinHoodMap::new() uses FxBuild: the same keys land in the same
    //      slots on every run and every machine, so a tokenizer built twice
    //      iterates its vocabulary in the same order.
    // KIND: regression
    // CHAPTER: ds.05 section 2.2
    let a: RobinHoodMap<u32, ()> = (0..50).map(|i| (i, ())).collect();
    let b: RobinHoodMap<u32, (), FxBuild> = (0..50).map(|i| (i, ())).collect();
    let ka: Vec<u32> = a.keys().copied().collect();
    let kb: Vec<u32> = b.keys().copied().collect();
    assert_eq!(ka, kb);
    assert_eq!(fx(&7u32), 7u64.wrapping_mul(SEED));
}
