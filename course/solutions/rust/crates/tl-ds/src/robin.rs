//! Robin Hood hash map with backward-shift deletion (ds.05).
//!
//! Open addressing over a power-of-two array of slots. Every stored entry
//! remembers its *displacement*: how many slots past its home slot (the
//! slot its hash picks) it sits. Two rules keep every probe short:
//!
//! * **Insert (Robin Hood).** Walk from the home slot. When the entry you
//!   carry has travelled further than the entry in the slot (`dist > d`),
//!   swap them and carry the evicted entry on. Displacements stay even, so
//!   the longest probe stays short at a high load factor.
//! * **Delete (backward shift).** Remove the entry, then move each following
//!   entry with displacement > 0 back one slot, until an empty slot or an
//!   entry already home. No tombstones, so lookups never slow down with
//!   churn.
//!
//! Lookup stops at an empty slot or at a slot whose displacement is smaller
//! than the distance walked: the Robin Hood rule would have placed the key
//! there. The table grows (doubling) before the load factor passes 7/8.
//!
//! Callers: `tl-tok` (L1.5) keeps its vocabulary `bytes -> id` and its merge
//! ranks `(left, right) -> (rank, id)` in this map. Chapter:
//! content/courses/algorithms/16-systems-data-structures/05-robin-hood-hash-map.md

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash, Hasher};

/// The multiplier of the Fx hash (rustc's `FxHasher`): an odd 64-bit constant
/// close to 2^64 / phi, so `x * SEED` mixes every input bit into the top bits.
pub const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// The largest number of entries per slot before the table grows: 7/8.
pub const MAX_LOAD_NUM: usize = 7;
pub const MAX_LOAD_DEN: usize = 8;

/// The smallest non-empty table, in slots.
pub const MIN_SLOTS: usize = 8;

/// The Fx hash: `h = (rotl(h, 5) ^ word) * SEED` for every 64-bit word of
/// input. Fast and good enough for keys an attacker does not choose (token
/// bytes and id pairs). Its low bits are weak, so the map takes the slot
/// index from the TOP bits of the hash.
#[derive(Clone, Copy, Debug, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    /// Bytes in little-endian 8-byte words, then a 4-, 2-, and 1-byte tail:
    /// every byte reaches the hash.
    fn write(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        while rest.len() >= 8 {
            self.add(u64::from_le_bytes(rest[..8].try_into().unwrap()));
            rest = &rest[8..];
        }
        if rest.len() >= 4 {
            self.add(u32::from_le_bytes(rest[..4].try_into().unwrap()) as u64);
            rest = &rest[4..];
        }
        if rest.len() >= 2 {
            self.add(u16::from_le_bytes(rest[..2].try_into().unwrap()) as u64);
            rest = &rest[2..];
        }
        if let Some(&b) = rest.first() {
            self.add(b as u64);
        }
    }

    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

/// Builds [`FxHasher`]s: the default `S` of [`RobinHoodMap`].
#[derive(Clone, Copy, Debug, Default)]
pub struct FxBuild;

impl BuildHasher for FxBuild {
    type Hasher = FxHasher;
    fn build_hasher(&self) -> FxHasher {
        FxHasher::default()
    }
}

/// One occupied slot. `dist` is the displacement: the entry sits `dist`
/// slots after its home slot (wrapping around the end of the array).
#[derive(Clone)]
struct Bucket<K, V> {
    hash: u64,
    dist: usize,
    key: K,
    value: V,
}

/// A hash map with Robin Hood insertion and backward-shift deletion.
///
/// `K: Hash + Eq` as for `std::collections::HashMap`; `S` builds the hasher
/// (default [`FxBuild`]). Iteration order is slot order: deterministic for a
/// given sequence of operations, but not insertion order.
#[derive(Clone)]
pub struct RobinHoodMap<K, V, S = FxBuild> {
    slots: Vec<Option<Bucket<K, V>>>, // empty, or a power of two >= MIN_SLOTS
    len: usize,
    hasher: S,
}

impl<K, V> RobinHoodMap<K, V, FxBuild> {
    /// An empty map; allocates nothing until the first insert.
    pub fn new() -> Self {
        Self::with_hasher(FxBuild)
    }

    /// An empty map that holds `n` entries without growing.
    pub fn with_capacity(n: usize) -> Self {
        Self::with_capacity_and_hasher(n, FxBuild)
    }
}

impl<K, V, S> Default for RobinHoodMap<K, V, S>
where
    S: Default,
{
    fn default() -> Self {
        RobinHoodMap { slots: Vec::new(), len: 0, hasher: S::default() }
    }
}

/// The number of slots a table needs to hold `n` entries at load <= 7/8:
/// 0 for 0, else the smallest power of two >= MIN_SLOTS with
/// `n * 8 <= slots * 7`.
pub fn slots_for(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut slots = MIN_SLOTS;
    while n * MAX_LOAD_DEN > slots * MAX_LOAD_NUM {
        slots *= 2;
    }
    slots
}

impl<K, V, S> RobinHoodMap<K, V, S> {
    /// An empty map using `hasher`.
    pub fn with_hasher(hasher: S) -> Self {
        RobinHoodMap { slots: Vec::new(), len: 0, hasher }
    }

    /// An empty map using `hasher` that holds `n` entries without growing.
    pub fn with_capacity_and_hasher(n: usize, hasher: S) -> Self {
        let mut slots = Vec::new();
        slots.resize_with(slots_for(n), || None);
        RobinHoodMap { slots, len: 0, hasher }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Entries the map holds before its next growth: 7/8 of the slots.
    pub fn capacity(&self) -> usize {
        self.slots.len() * MAX_LOAD_NUM / MAX_LOAD_DEN
    }

    /// Number of slots in the table (0 or a power of two).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// The longest successful probe: the largest displacement plus one
    /// (slots read to find that entry), or 0 for an empty map.
    pub fn max_probe_len(&self) -> usize {
        self.slots.iter().flatten().map(|b| b.dist + 1).max().unwrap_or(0)
    }

    /// The table as it sits, slot by slot: `None` for an empty slot, else the
    /// key and its displacement. For tests and the chapter's diagrams.
    pub fn layout(&self) -> Vec<Option<(&K, usize)>> {
        self.slots.iter().map(|s| s.as_ref().map(|b| (&b.key, b.dist))).collect()
    }

    /// Removes every entry and keeps the table.
    pub fn clear(&mut self) {
        for s in self.slots.iter_mut() {
            *s = None;
        }
        self.len = 0;
    }

    /// `(&key, &value)` in slot order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter { slots: self.slots.iter(), left: self.len }
    }

    /// `(&key, &mut value)` in slot order.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut { slots: self.slots.iter_mut(), left: self.len }
    }

    /// The keys, in slot order.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// The values, in slot order.
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// The home slot of `hash`: its top log2(slots) bits.
    #[inline]
    fn home(&self, hash: u64) -> usize {
        let bits = self.slots.len().trailing_zeros();
        (hash >> (64 - bits)) as usize
    }

    /// Puts `b` into the table starting at slot `idx` with the Robin Hood
    /// rule; `b.dist` is its displacement at `idx`. The key must be absent
    /// and there must be a free slot. Returns the slot where `b` itself ends.
    fn place(&mut self, mut idx: usize, mut b: Bucket<K, V>) -> usize {
        let mask = self.mask();
        let mut landed = None;
        loop {
            match &mut self.slots[idx] {
                slot @ None => {
                    *slot = Some(b);
                    self.len += 1;
                    return landed.unwrap_or(idx);
                }
                Some(cur) => {
                    if cur.dist < b.dist {
                        // The resident is richer (closer to home): it moves on.
                        std::mem::swap(cur, &mut b);
                        landed.get_or_insert(idx);
                    }
                }
            }
            idx = (idx + 1) & mask;
            b.dist += 1;
        }
    }

    /// Removes the entry at `idx` and shifts the run after it back by one.
    fn take_at(&mut self, idx: usize) -> Bucket<K, V> {
        let mask = self.mask();
        let out = self.slots[idx].take().expect("take_at: empty slot");
        self.len -= 1;
        let mut hole = idx;
        loop {
            let next = (hole + 1) & mask;
            match self.slots[next].take() {
                Some(mut b) if b.dist > 0 => {
                    b.dist -= 1;
                    self.slots[hole] = Some(b);
                    hole = next;
                }
                other => {
                    self.slots[next] = other;
                    break;
                }
            }
        }
        out
    }
}

impl<K, V, S> RobinHoodMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn hash_of<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        let mut h = self.hasher.build_hasher();
        key.hash(&mut h);
        h.finish()
    }

    /// The slot holding `key`, if any.
    fn find<Q>(&self, hash: u64, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        if self.len == 0 {
            return None;
        }
        let mask = self.mask();
        let mut idx = self.home(hash);
        let mut dist = 0;
        loop {
            match &self.slots[idx] {
                None => return None,
                Some(b) => {
                    if b.dist < dist {
                        return None; // the key would have taken this slot
                    }
                    if b.hash == hash && b.key.borrow() == key {
                        return Some(idx);
                    }
                }
            }
            idx = (idx + 1) & mask;
            dist += 1;
        }
    }

    /// Grows (doubling, rehashing every entry) until `additional` more
    /// entries fit at load <= 7/8.
    pub fn reserve(&mut self, additional: usize) {
        let need = self.len + additional;
        if need * MAX_LOAD_DEN <= self.slots.len() * MAX_LOAD_NUM {
            return;
        }
        let new_slots = slots_for(need).max(self.slots.len() * 2);
        let mut fresh = Vec::new();
        fresh.resize_with(new_slots, || None);
        let old = std::mem::replace(&mut self.slots, fresh);
        self.len = 0;
        for b in old.into_iter().flatten() {
            let home = self.home(b.hash);
            self.place(home, Bucket { dist: 0, ..b });
        }
    }

    /// Inserts `key -> value`. Returns the old value when the key was
    /// present (the stored key is kept), else `None`.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let hash = self.hash_of(&key);
        if let Some(idx) = self.find(hash, &key) {
            let b = self.slots[idx].as_mut().unwrap();
            return Some(std::mem::replace(&mut b.value, value));
        }
        self.reserve(1);
        let home = self.home(hash);
        self.place(home, Bucket { hash, dist: 0, key, value });
        None
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.find(self.hash_of(key), key)?;
        self.slots[idx].as_ref().map(|b| &b.value)
    }

    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.find(self.hash_of(key), key)?;
        self.slots[idx].as_mut().map(|b| &mut b.value)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.find(self.hash_of(key), key).is_some()
    }

    /// Removes `key`; returns its value.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes `key`; returns the stored key and its value.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = self.find(self.hash_of(key), key)?;
        let b = self.take_at(idx);
        Some((b.key, b.value))
    }

    /// The entry for `key`, for in-place update or insertion. Reserves room
    /// for one more entry first, so a vacant insert never grows the table.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        let hash = self.hash_of(&key);
        match self.find(hash, &key) {
            Some(idx) => Entry::Occupied(OccupiedEntry { map: self, idx }),
            None => {
                self.reserve(1);
                Entry::Vacant(VacantEntry { map: self, hash, key })
            }
        }
    }
}

/// A view into one key's slot: present ([`OccupiedEntry`]) or absent
/// ([`VacantEntry`]).
pub enum Entry<'a, K, V, S> {
    Occupied(OccupiedEntry<'a, K, V, S>),
    Vacant(VacantEntry<'a, K, V, S>),
}

pub struct OccupiedEntry<'a, K, V, S> {
    map: &'a mut RobinHoodMap<K, V, S>,
    idx: usize,
}

pub struct VacantEntry<'a, K, V, S> {
    map: &'a mut RobinHoodMap<K, V, S>,
    hash: u64,
    key: K,
}

impl<'a, K, V, S> Entry<'a, K, V, S> {
    /// The value, inserting `default` first if the key was absent.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(default),
        }
    }

    /// The value, inserting `f()` first if the key was absent; `f` runs only then.
    pub fn or_insert_with<F: FnOnce() -> V>(self, f: F) -> &'a mut V {
        match self {
            Entry::Occupied(o) => o.into_mut(),
            Entry::Vacant(v) => v.insert(f()),
        }
    }

    /// Runs `f` on the value if the key is present; returns the entry.
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            Entry::Occupied(mut o) => {
                f(o.get_mut());
                Entry::Occupied(o)
            }
            e @ Entry::Vacant(_) => e,
        }
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(o) => o.key(),
            Entry::Vacant(v) => &v.key,
        }
    }
}

impl<'a, K, V: Default, S> Entry<'a, K, V, S> {
    pub fn or_default(self) -> &'a mut V {
        self.or_insert_with(V::default)
    }
}

impl<'a, K, V, S> OccupiedEntry<'a, K, V, S> {
    pub fn key(&self) -> &K {
        &self.map.slots[self.idx].as_ref().unwrap().key
    }

    pub fn get(&self) -> &V {
        &self.map.slots[self.idx].as_ref().unwrap().value
    }

    pub fn get_mut(&mut self) -> &mut V {
        &mut self.map.slots[self.idx].as_mut().unwrap().value
    }

    pub fn into_mut(self) -> &'a mut V {
        &mut self.map.slots[self.idx].as_mut().unwrap().value
    }

    /// Replaces the value; returns the old one.
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }

    /// Removes the entry (backward shift); returns its value.
    pub fn remove(self) -> V {
        self.map.take_at(self.idx).value
    }
}

impl<'a, K, V, S> VacantEntry<'a, K, V, S> {
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Inserts `value` (Robin Hood placement); returns a reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        let VacantEntry { map, hash, key } = self;
        let home = map.home(hash);
        let idx = map.place(home, Bucket { hash, dist: 0, key, value });
        &mut map.slots[idx].as_mut().unwrap().value
    }
}

/// Iterator over `(&K, &V)` in slot order.
pub struct Iter<'a, K, V> {
    slots: std::slice::Iter<'a, Option<Bucket<K, V>>>,
    left: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        for s in self.slots.by_ref() {
            if let Some(b) = s {
                self.left -= 1;
                return Some((&b.key, &b.value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}

/// Iterator over `(&K, &mut V)` in slot order.
pub struct IterMut<'a, K, V> {
    slots: std::slice::IterMut<'a, Option<Bucket<K, V>>>,
    left: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        for s in self.slots.by_ref() {
            if let Some(b) = s {
                self.left -= 1;
                return Some((&b.key, &mut b.value));
            }
        }
        None
    }
}

pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<&'a K> {
        self.inner.next().map(|(k, _)| k)
    }
}

pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<&'a V> {
        self.inner.next().map(|(_, v)| v)
    }
}

impl<'a, K, V, S> IntoIterator for &'a RobinHoodMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher + Default> FromIterator<(K, V)> for RobinHoodMap<K, V, S> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut m = Self::default();
        m.extend(iter);
        m
    }
}

impl<K: Hash + Eq, V, S: BuildHasher> Extend<(K, V)> for RobinHoodMap<K, V, S> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: fmt::Debug, V: fmt::Debug, S> fmt::Debug for RobinHoodMap<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}
