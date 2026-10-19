//! Reference learner tests for ds.05 (rung R2: the chapter names the tests,
//! the learner writes the bodies). They prove the 0.60 mutation bar is
//! reachable with the public API alone (black-box rule: an integration test).

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use tl_ds::robin::{slots_for, FxHasher, RobinHoodMap, SEED};

#[derive(Default)]
struct IdHasher(u64);
impl Hasher for IdHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!()
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

fn k(home: u64, tag: u64) -> u64 {
    (home << 61) | tag
}

fn table() -> RobinHoodMap<u64, u32, Id> {
    let mut m = RobinHoodMap::with_capacity_and_hasher(4, Id);
    for (i, key) in [k(2, 1), k(2, 2), k(3, 3), k(2, 4)].into_iter().enumerate() {
        m.insert(key, i as u32);
    }
    m
}

fn layout(m: &RobinHoodMap<u64, u32, Id>) -> Vec<Option<(u64, usize)>> {
    m.layout().into_iter().map(|s| s.map(|(k, d)| (*k, d))).collect()
}

#[test]
fn insert_steals_from_the_rich() {
    let m = table();
    let l = layout(&m);
    assert_eq!(l[2], Some((k(2, 1), 0)));
    assert_eq!(l[3], Some((k(2, 2), 1)));
    assert_eq!(l[4], Some((k(2, 4), 2)));
    assert_eq!(l[5], Some((k(3, 3), 2)));
}

#[test]
fn remove_shifts_back() {
    let mut m = table();
    assert_eq!(m.remove(&k(2, 2)), Some(1));
    let l = layout(&m);
    assert_eq!(l[3], Some((k(2, 4), 1)));
    assert_eq!(l[4], Some((k(3, 3), 1)));
    assert_eq!(l[5], None);
    assert_eq!(m.get(&k(3, 3)), Some(&2));
    assert_eq!(m.len(), 3);
}

#[test]
fn every_key_is_found() {
    let m = table();
    for (i, key) in [k(2, 1), k(2, 2), k(3, 3), k(2, 4)].into_iter().enumerate() {
        assert_eq!(m.get(&key), Some(&(i as u32)));
    }
    assert_eq!(m.get(&k(4, 9)), None);
}

#[test]
fn grows_past_seven_eighths() {
    assert_eq!((slots_for(7), slots_for(8)), (8, 16));
    let mut m: RobinHoodMap<u32, u32> = RobinHoodMap::new();
    for i in 0..7 {
        m.insert(i, i);
    }
    assert_eq!(m.slot_count(), 8);
}

#[test]
fn hasher_reads_every_byte() {
    let mut a = FxHasher::default();
    a.write(b"ab");
    let mut b = FxHasher::default();
    b.write(b"abc");
    assert_ne!(a.finish(), b.finish());
    let mut c = FxHasher::default();
    c.write_u64(3);
    assert_eq!(c.finish(), 3u64.wrapping_mul(SEED));
}

#[test]
fn entry_counts() {
    let mut m: RobinHoodMap<&str, u32> = RobinHoodMap::new();
    for w in ["a", "b", "a"] {
        *m.entry(w).or_insert(0) += 1;
    }
    assert_eq!((m.get("a"), m.get("b")), (Some(&2), Some(&1)));
}

#[test]
fn matches_std_hashmap() {
    let mut x: u64 = 12345;
    let mut ours: RobinHoodMap<u64, u64> = RobinHoodMap::new();
    let mut std_: HashMap<u64, u64> = HashMap::new();
    for step in 0..20_000 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let key = (x >> 33) % 500;
        if (x >> 20) % 3 == 0 {
            assert_eq!(ours.remove(&key), std_.remove(&key));
        } else {
            assert_eq!(ours.insert(key, step), std_.insert(key, step));
        }
        assert_eq!(ours.len(), std_.len());
    }
    for (key, v) in &std_ {
        assert_eq!(ours.get(key), Some(v));
    }
}
