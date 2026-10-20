//! Bloom filter (ds.08): sizing, double hashing, and the byte layout of
//! contracts/formats/bloom.md, so a filter serialized here answers every
//! `contains` identically in the Python and Rust implementations of that
//! page.
//!
//! A Bloom filter is `m` bits and `k` hash functions. Insert sets the `k`
//! bits an item hashes to; `contains` answers yes when all `k` are set. An
//! inserted item is never missed (no false negatives); an item never
//! inserted is reported present with probability about
//! `(1 - e^(-k n / m))^k` after `n` inserts.
//!
//! Caller: `data.03` exact dedup screens paragraph hashes with it and
//! confirms every positive. Chapter:
//! content/courses/programming-and-algorithms/15-systems-data-structures/08-bloom-filter.md

use std::fmt;

/// FNV-1a 64 offset basis and prime (formats/kv-block.md).
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The serialized header: magic, version, m, k, reserved, n_inserted.
pub const MAGIC: [u8; 4] = *b"TLBF";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 32;

/// The largest filter this implementation builds: 2^40 bits (128 GiB).
pub const MAX_BITS: u64 = 1 << 40;

#[derive(Debug, Clone, PartialEq)]
pub enum BloomError {
    /// `n` or `p` (or `m`, `k`) out of range; the message names which.
    BadParams(String),
    /// `union` of filters whose `m` or `k` differ.
    Mismatch(String),
    /// `from_bytes` input that is not a version-1 filter; the message says why.
    Format(String),
}

impl fmt::Display for BloomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BloomError::BadParams(s) => write!(f, "bad parameters: {s}"),
            BloomError::Mismatch(s) => write!(f, "filters differ: {s}"),
            BloomError::Format(s) => write!(f, "not a bloom filter: {s}"),
        }
    }
}

impl std::error::Error for BloomError {}

/// FNV-1a 64 over `bytes`: for each byte, xor it in, then multiply.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// The SplitMix64 finalizer: a bijection on u64 that mixes every bit.
pub fn mix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Bits for `n` items at false-positive rate `p`: ceil(-n ln p / (ln 2)^2).
/// Assumes `n >= 1` and `0 < p < 1` (checked by [`Bloom::with_rate`]).
pub fn optimal_m(n: u64, p: f64) -> u64 {
    let ln2 = std::f64::consts::LN_2;
    (-(n as f64) * p.ln() / (ln2 * ln2)).ceil() as u64
}

/// Hash functions for `m` bits and `n` items: max(1, round(m / n * ln 2)),
/// rounding half away from zero.
pub fn optimal_k(m: u64, n: u64) -> u32 {
    let k = (m as f64 / n as f64 * std::f64::consts::LN_2).round();
    if k < 1.0 {
        1
    } else {
        k as u32
    }
}

/// The predicted false-positive rate after `n` inserts: (1 - e^(-k n / m))^k.
pub fn predicted_fp_rate(m: u64, k: u32, n: u64) -> f64 {
    (1.0 - (-(k as f64) * n as f64 / m as f64).exp()).powi(k as i32)
}

/// A Bloom filter over byte strings (formats/bloom.md).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bloom {
    m: u64,
    k: u32,
    n_inserted: u64,
    bits: Vec<u8>, // ceil(m / 8) bytes, bit g at byte g / 8, position g % 8 (LSB first)
}

impl Bloom {
    /// An empty filter sized for `n` items at false-positive rate `p`.
    /// `BadParams` unless `n >= 1` and `0 < p < 1` (NaN is rejected).
    pub fn with_rate(n: u64, p: f64) -> Result<Bloom, BloomError> {
        if n < 1 {
            return Err(BloomError::BadParams(format!("n must be >= 1, got {n}")));
        }
        if !(p > 0.0 && p < 1.0) {
            return Err(BloomError::BadParams(format!("p must be in (0, 1), got {p}")));
        }
        let m = optimal_m(n, p);
        Bloom::new(m, optimal_k(m, n))
    }

    /// An empty filter of exactly `m` bits and `k` hash functions.
    pub fn new(m: u64, k: u32) -> Result<Bloom, BloomError> {
        if m == 0 || k == 0 {
            return Err(BloomError::BadParams(format!("m and k must be >= 1, got m={m} k={k}")));
        }
        if m > MAX_BITS {
            return Err(BloomError::BadParams(format!("m = {m} bits is over the 2^40 limit")));
        }
        Ok(Bloom { m, k, n_inserted: 0, bits: vec![0; m.div_ceil(8) as usize] })
    }

    pub fn m(&self) -> u64 {
        self.m
    }

    pub fn k(&self) -> u32 {
        self.k
    }

    /// `insert` calls so far (a union adds the other filter's count).
    pub fn n_inserted(&self) -> u64 {
        self.n_inserted
    }

    /// The `k` bit positions of `item`: g_i = (h1 + i * h2) mod 2^64 mod m,
    /// h1 = fnv1a64(item), h2 = mix64(h1) | 1.
    pub fn bit_positions(&self, item: &[u8]) -> Vec<u64> {
        let h1 = fnv1a64(item);
        let h2 = mix64(h1) | 1;
        (0..self.k as u64).map(|i| h1.wrapping_add(i.wrapping_mul(h2)) % self.m).collect()
    }

    pub fn insert(&mut self, item: &[u8]) {
        for g in self.bit_positions(item) {
            self.bits[(g / 8) as usize] |= 1 << (g % 8);
        }
        self.n_inserted += 1;
    }

    /// True for every inserted item; true for others at the FP rate.
    pub fn contains(&self, item: &[u8]) -> bool {
        self.bit_positions(item)
            .into_iter()
            .all(|g| self.bits[(g / 8) as usize] & (1 << (g % 8)) != 0)
    }

    /// In place: the bitwise OR of both bit arrays; the counts add.
    /// `Mismatch` when `m` or `k` differ.
    pub fn union(&mut self, other: &Bloom) -> Result<(), BloomError> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomError::Mismatch(format!(
                "m={} k={} vs m={} k={}",
                self.m, self.k, other.m, other.k
            )));
        }
        for (a, b) in self.bits.iter_mut().zip(&other.bits) {
            *a |= *b;
        }
        self.n_inserted += other.n_inserted;
        Ok(())
    }

    /// Bits set, out of `m`.
    pub fn count_ones(&self) -> u64 {
        self.bits.iter().map(|b| b.count_ones() as u64).sum()
    }

    /// The formats/bloom.md serialization: a 32-byte little-endian header,
    /// then the bit array.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.bits.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.m.to_le_bytes());
        out.extend_from_slice(&self.k.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&self.n_inserted.to_le_bytes());
        out.extend_from_slice(&self.bits);
        out
    }

    /// Inverse of `to_bytes`. `Format` for a bad magic or version, `m == 0`
    /// or `k == 0`, a non-zero reserved field, or a length other than
    /// `32 + ceil(m / 8)`.
    pub fn from_bytes(b: &[u8]) -> Result<Bloom, BloomError> {
        let fail = |why: String| Err(BloomError::Format(why));
        if b.len() < HEADER_LEN {
            return fail(format!("{} bytes is shorter than the 32-byte header", b.len()));
        }
        if b[0..4] != MAGIC {
            return fail(format!("magic {:02x?} is not \"TLBF\"", &b[0..4]));
        }
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        let (version, m, k, reserved, n) = (u32_at(4), u64_at(8), u32_at(16), u32_at(20), u64_at(24));
        if version != VERSION {
            return fail(format!("version {version}, expected 1"));
        }
        if m == 0 || k == 0 {
            return fail(format!("m={m} k={k}: both must be >= 1"));
        }
        if m > MAX_BITS {
            return fail(format!("m = {m} bits is over the 2^40 limit"));
        }
        if reserved != 0 {
            return fail(format!("reserved field is {reserved}, must be 0"));
        }
        let want = HEADER_LEN as u64 + m.div_ceil(8);
        if b.len() as u64 != want {
            return fail(format!("length {} but m = {m} needs {want}", b.len()));
        }
        Ok(Bloom { m, k, n_inserted: n, bits: b[HEADER_LEN..].to_vec() })
    }
}
