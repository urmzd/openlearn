# Bloom filter

## Overview

| | |
|---|---|
| **Module** | `ds.08` · side · Rust · Pass 3 · 2 h reading (optional) |
| **Solution** | [`rust/crates/tl-ds/src/bloom.rs`](../../../../course/solutions/rust/crates/tl-ds/src/bloom.rs): `fnv1a64`, `mix64`, `optimal_m`, `optimal_k`, `predicted_fp_rate`, `Bloom` (`with_rate`, `new`, `bit_positions`, `insert`, `contains`, `union`, `count_ones`, `to_bytes`, `from_bytes`) · model tests: [`ds08_bloom.rs`](../../../../course/solutions/learner-tests/ds.08/rust/crates/tl-ds/tests/ds08_bloom.rs) · the independent Python screen `_Bloom` in [`python/corpus/dedup.py`](../../../../course/solutions/python/corpus/dedup.py) |
| **Contract** | byte layout and hashing: [`formats/bloom.md`](../../../../course/contracts/formats/bloom.md) |
| **Needs** | reading: `S-M06b` the [false-positive rate and the optimal k](../../mathematics-for-computing/06-discrete-math-2/91-problem-set-b.md), `M06.3` [FNV-1a and the SplitMix finalizer](../../mathematics-for-computing/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) |
| **Used by** | No runtime module. This optional Rust module stands on its own; `data.03` writes an independent Python Bloom screen to the same byte format, and for the same items both set the same bits. |
| **Optional depth** | Bloom, *Space/Time Trade-offs in Hash Coding with Allowable Errors* (1970); Kirsch and Mitzenmacher, *Less Hashing, Same Performance: Building a Better Bloom Filter* (2006); Broder and Mitzenmacher, *Network Applications of Bloom Filters: A Survey* (2004) |

## Key Takeaways

- A Bloom filter is $m$ bits and $k$ hash functions: insert sets $k$ bits, `contains` checks them, so a member is never missed and a non-member is reported present with probability about $(1 - e^{-kn/m})^k$.
- For $n$ items at rate $p$, $m = \lceil -n \ln p / (\ln 2)^2 \rceil$ bits and $k = \mathrm{round}(\frac{m}{n} \ln 2)$ hashes: 4 items at 10% need 20 bits and 3 hashes.
- Two hashes are enough: $g_i = h_1 + i\,h_2 \bmod m$ (double hashing) with $h_1$ = FNV-1a 64 and $h_2$ = the SplitMix finalizer of $h_1$, forced odd.
- Union is bitwise OR of two filters with the same $m$ and $k$, which is exactly the filter of both sets.
- A fixed byte layout makes the filter portable: every implementation of the format produces the same 35 bytes for the worked example.

## How to read this chapter

Sections 1 to 3 derive the sizing formulas and the double-hashing scheme from first principles and work a 20-bit filter by hand, down to its 35 serialized bytes. Section 4 fixes the interface and lists what a correct filter must handle. Section 5 walks through [`bloom.rs`](../../../../course/solutions/rust/crates/tl-ds/src/bloom.rs): hashing and sizing first, then the bit array, then bytes. No other module calls this code: languages meet only over processes and files, and the one Bloom filter on the serving path, the Python screen of `data.03`, is written in Python. The byte format of [`formats/bloom.md`](../../../../course/contracts/formats/bloom.md) is what the two share. Try to predict each step before reading it.

---

## 1. Why now

The corpus pipeline (`data.01` to `data.08`) must drop exact duplicate paragraphs from millions of documents. A `set` of paragraph hashes in Python is 8 bytes of hash plus about 60 bytes of overhead per entry; a Bloom filter answers "have I seen this?" in about 10 bits per paragraph, with no false negatives and a false-positive rate you choose. The dedup stage then confirms every "seen" answer exactly (a sort-merge on the full hashes), so a false positive costs time, never data. The serialized format is useful across processes and implementations; `formats/bloom.md` fixes that layout.

## 2. Principles

### 2.1 Bits and hash functions

| Symbol | Meaning | Type |
|---|---|---|
| $n$ | items inserted (or expected) | `u64` |
| $m$ | bits in the filter | `u64`, $1 \le m \le 2^{40}$ |
| $k$ | hash functions (bits per item) | `u32`, $k \ge 1$ |
| $p$ | target false-positive rate | real in $(0, 1)$ |
| $x$ | an item: a byte string | `&[u8]` |
| $g_0, \dots, g_{k-1}$ | the bit positions of $x$ | `u64` in $[0, m)$ |
| $f$ | the fraction of bits set | real in $[0, 1]$ |

**Insert** sets bits $g_0(x), \dots, g_{k-1}(x)$; **contains** answers yes when all $k$ are set. An inserted item's bits stay set forever (nothing clears bits), so **there are no false negatives**. A non-member is reported present only if all $k$ of its bits were set by other items.

### 2.2 The false-positive rate and the optimal k

Model each of the $kn$ bit choices as uniform and independent. One bit stays 0 after all of them with probability $(1 - 1/m)^{kn} \approx e^{-kn/m}$, so the fraction of set bits is about $f = 1 - e^{-kn/m}$, and a fresh item's $k$ probes all hit set bits with probability

$$q(k) = f^k \approx \left(1 - e^{-kn/m}\right)^k.$$

More hash functions mean more checks per query but more bits set. Minimizing $\ln q = k \ln(1 - e^{-kn/m})$ over $k$ (`S-M06b`) gives $k^\ast = \frac{m}{n}\ln 2$, where exactly half the bits are set, and $q = 2^{-k^\ast}$. Solving $p = 2^{-\frac{m}{n}\ln 2}$ for $m$:

$$m = \left\lceil \frac{-n \ln p}{(\ln 2)^2} \right\rceil, \qquad k = \max\left(1, \mathrm{round}\left(\tfrac{m}{n} \ln 2\right)\right),$$

rounding half away from zero. About $1.44 \log_2(1/p)$ bits per item: 9.6 bits at 1%, 14.4 at 0.1%.

### 2.3 Two hashes make k

Computing $k$ independent hashes per item is slow. Kirsch and Mitzenmacher showed that $g_i = h_1 + i \cdot h_2$ (mod $m$) loses nothing asymptotically. The format fixes:

$$h_1 = \text{fnv1a64}(x), \qquad h_2 = \text{mix64}(h_1) \lor 1, \qquad g_i = \big((h_1 + i\,h_2) \bmod 2^{64}\big) \bmod m.$$

**FNV-1a 64** (the same as the KV block hash of `formats/kv-block.md`): start from the offset basis `0xcbf29ce484222325`; for each byte, XOR it in, **then** multiply by the prime `0x100000001b3` (mod $2^{64}$). Multiplying first and XORing after is FNV-1, a different hash. **mix64** is the SplitMix64 finalizer, a bijection that spreads every input bit over every output bit (`M06.3`). Forcing $h_2$ odd keeps the $k$ probes distinct when $m$ is a power of two. Bit $g$ lives in byte $\lfloor g/8 \rfloor$ at bit position $g \bmod 8$, least significant bit first.

### 2.4 What can be measured

The approximation of 2.2 assumes independent bits. Given an actual filter, the exact probability that a random fresh item is a false positive is $f^k$ with $f$ the measured fraction of set bits (each probe lands on a uniform bit). Over $N$ fresh items the count of false positives is then Binomial$(N, f^k)$, with mean $N f^k$ and standard deviation $\sqrt{N f^k (1 - f^k)}$: a statistical check can ask the count to lie within 3 standard deviations, which a correct filter fails about 0.3% of the time.

### 2.5 Union and bytes

Two filters with the same $m$ and $k$ (and so the same hash positions) merge by bitwise OR: the result has exactly the bits that inserting both sets would set. `n_inserted` counts `insert` calls and a union adds the counts. `to_bytes` writes the 32-byte little-endian header (magic `TLBF`, version 1, $m$, $k$, a reserved zero, `n_inserted`) and then the $\lceil m/8 \rceil$ bytes of bits; `from_bytes` rejects anything else.

The Python corpus pipeline has a separate implementation, `_Bloom` in [`python/corpus/dedup.py`](../../../../course/solutions/python/corpus/dedup.py), with the same sizing and probes (it keeps the bits in memory and does not serialize). This module focuses on the Rust filter and its serialized format.

## 3. Worked example by hand

`with_rate(4, 0.1)`: $-4 \ln 0.1 / (\ln 2)^2 = 9.2103 / 0.48045 = 19.17$, so $m = 20$; $\frac{20}{4} \ln 2 = 3.466$, so $k = 3$. The bit array is $\lceil 20/8 \rceil = 3$ bytes.

`cat`: $h_1 = \text{fnv1a64}(\texttt{cat}) = \texttt{f5e307190ce4a327}$ and $h_2 = \text{mix64}(h_1) \lor 1 = \texttt{abc55d317f9d6793}$, which give $g = 11, 14, 17$ (the format page lists every item's positions). `dog` gives $13, 0, 3$. After inserting both, the set bits are 0 and 3 (byte 0: $2^0 + 2^3 = \texttt{09}$), 11, 13, 14 (byte 1: bits 3, 5, 6 of that byte, $8 + 32 + 64 = \texttt{68}$), and 17 (byte 2: bit 1, $\texttt{02}$): `09 68 02`. `bird` probes $14, 7, 16$: bit 7 is clear, so `contains(b"bird")` is false. Six of 20 bits are set, $f = 0.3$, and a fresh item's chance of a false positive is $0.3^3 = 2.7\%$.

`to_bytes` is the 35 bytes

```text
54 4c 42 46  01 00 00 00  14 00 00 00 00 00 00 00   "TLBF", version 1, m = 20
03 00 00 00  00 00 00 00  02 00 00 00 00 00 00 00   k = 3, reserved 0, n_inserted = 2
09 68 02                                            the bits
```

## 4. The interface

```rust
// rust/crates/tl-ds/src/bloom.rs (formats/bloom.md)
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;  pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
pub const MAGIC: [u8; 4] = *b"TLBF";  pub const VERSION: u32 = 1;  pub const HEADER_LEN: usize = 32;
pub const MAX_BITS: u64 = 1 << 40;
pub enum BloomError { BadParams(String), Mismatch(String), Format(String) }   // Display + Error
pub fn fnv1a64(bytes: &[u8]) -> u64;
pub fn mix64(z: u64) -> u64;
pub fn optimal_m(n: u64, p: f64) -> u64;
pub fn optimal_k(m: u64, n: u64) -> u32;
pub fn predicted_fp_rate(m: u64, k: u32, n: u64) -> f64;
#[derive(Clone, Debug, PartialEq, Eq)] pub struct Bloom { /* m, k, n_inserted, bits */ }
impl Bloom {
    pub fn with_rate(n: u64, p: f64) -> Result<Bloom, BloomError>;     // BadParams unless n >= 1, 0 < p < 1
    pub fn new(m: u64, k: u32) -> Result<Bloom, BloomError>;          // 1 <= m <= 2^40, k >= 1
    pub fn m(&self) -> u64;  pub fn k(&self) -> u32;  pub fn n_inserted(&self) -> u64;
    pub fn bit_positions(&self, item: &[u8]) -> Vec<u64>;
    pub fn insert(&mut self, item: &[u8]);  pub fn contains(&self, item: &[u8]) -> bool;
    pub fn union(&mut self, other: &Bloom) -> Result<(), BloomError>; // Mismatch when m or k differ
    pub fn count_ones(&self) -> u64;
    pub fn to_bytes(&self) -> Vec<u8>;  pub fn from_bytes(b: &[u8]) -> Result<Bloom, BloomError>;
}
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example's sizing and bits | unit | section 3: $m = 20$, $k = 3$, the positions of cat, dog, bird, the 35 bytes | the code, the format page, and the worked example agree |
| Hash function values by hand | golden | FNV-1a of "", "a", "cat"; mix64 of 0 and 1 | every implementation sets the same bits |
| Sizing formulas | unit | $m$ and $k$ for five $(n, p)$, including the rounding of $k$; byte length | the screen is as small as the target allows |
| Bad parameters are errors | boundary | $n = 0$, $p \notin (0, 1)$, NaN, $k = 0$, $m > 2^{40}$ are `BadParams` | a bad config fails at start, not with a 128 GiB allocation |
| No false negatives | property | 5,000 random items found in their filter, after a byte round trip, and in the union | dedup never keeps a duplicate it has seen |
| The false-positive rate is within three sigma | statistical | 100,000 fresh items: false positives within 3 sd of $N f^k$, and below 2% at a 1% design | the filter is as good as its sizing promises |
| Union is the filter of both sets | unit | OR of two filters equals the filter of both; counts add; mismatched $m$ or $k$ refused | `data.03` merges per-shard screens |
| Golden bytes match the format | golden | 10 reference cases (empty items, odd $m$, $p = 10^{-6}$): bytes and probe answers | every implementation of the format writes the same bytes |
| `from_bytes` rejects malformed input | boundary | bad magic, version, $m = 0$, $k = 0$, reserved, short, long, inconsistent length | a corrupt file is refused, not misread |

A model property-test suite (proptest with a fixed seed and no failure files: the worked example bytes, the sizing table, "every inserted item is present", "bytes round-trip", "union equals one filter of both sets", a mismatched union, and malformed bytes) is in [`ds08_bloom.rs`](../../../../course/solutions/learner-tests/ds.08/rust/crates/tl-ds/tests/ds08_bloom.rs).

## 5. The solution, step by step

### 5.1 From first principles

1. **"Have I seen this?" with no false negatives** means a set bit can never be cleared: the structure only ever ORs bits in. A miss is certain; a hit is probabilistic.
2. **The false-positive rate is $f^k$,** the chance that $k$ independent probes all land on set bits. Minimizing it for a given $m/n$ gives $k = \frac{m}{n}\ln 2$ (half the bits set), and inverting gives $m = \lceil -n \ln p / (\ln 2)^2 \rceil$. The ceiling matters: flooring makes the filter short of its target. For the worked example, 19.17 becomes 20 bits and 3.466 becomes 3 hashes.
3. **$k$ hashes per item are expensive; two are enough.** $g_i = h_1 + i\,h_2$ gives $k$ probes from two hashes (Kirsch and Mitzenmacher). $h_2$ must be odd so that, when $m$ is a power of two, the $k$ probes are distinct (an odd step generates every residue).
4. **$h_2$ must not be a simple function of $h_1$'s low bits,** or the two probes correlate. Running $h_1$ through the SplitMix64 finalizer, a bijection that mixes every bit, makes $h_2$ look independent at the cost of a few multiplies.
5. **Filters built in different processes and languages must be interchangeable,** so everything that decides a bit is fixed by the format: FNV-1a (xor, then multiply), mix64, the wrap-around arithmetic mod $2^{64}$ before mod $m$, and LSB-first bit numbering. The header records $m$, $k$, and the count so a filter can be validated and merged.
6. **Two filters with the same $m$ and $k$ probe the same positions for every item,** so OR is exactly the filter of the union of the sets; different $m$ or $k$ have no meaningful union and must be refused.
7. **Bytes from a file are untrusted.** `from_bytes` checks every header field and that the length is exactly $32 + \lceil m/8 \rceil$; otherwise a later `contains` could index past the bit array.

### 5.2 The code, annotated

Hashing and sizing. Each line is one formula from section 2; the casts and `wrapping_mul` make the arithmetic mod $2^{64}$ explicit, which is what other implementations must reproduce.

```rust
/// FNV-1a 64 over `bytes`: for each byte, xor it in, then multiply.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;                 // xor FIRST: this is what makes it FNV-1a, not FNV-1
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

/// ceil(-n ln p / (ln 2)^2): round UP, or the filter misses its target rate.
pub fn optimal_m(n: u64, p: f64) -> u64 {
    let ln2 = std::f64::consts::LN_2;
    (-(n as f64) * p.ln() / (ln2 * ln2)).ceil() as u64
}

/// max(1, round(m / n * ln 2)); f64::round rounds half away from zero.
pub fn optimal_k(m: u64, n: u64) -> u32 {
    let k = (m as f64 / n as f64 * std::f64::consts::LN_2).round();  // round, not truncate: 6.64 -> 7
    if k < 1.0 { 1 } else { k as u32 }
}
```

Construction validates before allocating. `!(p > 0.0 && p < 1.0)` is written that way so that NaN, for which every comparison is false, is rejected too.

```rust
pub fn with_rate(n: u64, p: f64) -> Result<Bloom, BloomError> {
    if n < 1 {
        return Err(BloomError::BadParams(format!("n must be >= 1, got {n}")));
    }
    if !(p > 0.0 && p < 1.0) {         // also catches NaN
        return Err(BloomError::BadParams(format!("p must be in (0, 1), got {p}")));
    }
    let m = optimal_m(n, p);
    Bloom::new(m, optimal_k(m, n))     // new() enforces 1 <= m <= 2^40, k >= 1
}

pub fn new(m: u64, k: u32) -> Result<Bloom, BloomError> {
    if m == 0 || k == 0 { /* ... */ }
    if m > MAX_BITS { /* ... */ }      // refuse a 128 GiB+ allocation from a bad config
    Ok(Bloom { m, k, n_inserted: 0, bits: vec![0; m.div_ceil(8) as usize] })
}
```

Probes, insert, contains. `bit_positions` is the single definition of where an item's bits are; `insert` and `contains` both use it, so they cannot disagree.

```rust
/// g_i = (h1 + i * h2) mod 2^64 mod m, h1 = fnv1a64(item), h2 = mix64(h1) | 1.
pub fn bit_positions(&self, item: &[u8]) -> Vec<u64> {
    let h1 = fnv1a64(item);
    let h2 = mix64(h1) | 1;            // odd: distinct probes when m is a power of two
    (0..self.k as u64).map(|i| h1.wrapping_add(i.wrapping_mul(h2)) % self.m).collect()
}

pub fn insert(&mut self, item: &[u8]) {
    for g in self.bit_positions(item) {
        self.bits[(g / 8) as usize] |= 1 << (g % 8);   // byte g/8, bit g%8 from the LSB
    }
    self.n_inserted += 1;
}

pub fn contains(&self, item: &[u8]) -> bool {
    self.bit_positions(item)
        .into_iter()
        .all(|g| self.bits[(g / 8) as usize] & (1 << (g % 8)) != 0)  // every probe must be set
}

pub fn union(&mut self, other: &Bloom) -> Result<(), BloomError> {
    if self.m != other.m || self.k != other.k {
        return Err(BloomError::Mismatch(/* ... */));   // different probes: no meaningful union
    }
    for (a, b) in self.bits.iter_mut().zip(&other.bits) {
        *a |= *b;                      // OR: a member of either set stays present
    }
    self.n_inserted += other.n_inserted;
    Ok(())
}
```

Serialization. The header is written field by field in little-endian, so the layout does not depend on struct padding or host byte order. `from_bytes` checks in the order that makes each read safe: length before any field, then magic, then the rest.

```rust
pub fn to_bytes(&self) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + self.bits.len());
    out.extend_from_slice(&MAGIC);                     // "TLBF"
    out.extend_from_slice(&VERSION.to_le_bytes());     // u32
    out.extend_from_slice(&self.m.to_le_bytes());      // u64
    out.extend_from_slice(&self.k.to_le_bytes());      // u32
    out.extend_from_slice(&0u32.to_le_bytes());        // reserved
    out.extend_from_slice(&self.n_inserted.to_le_bytes()); // u64: 32 bytes so far
    out.extend_from_slice(&self.bits);
    out
}

pub fn from_bytes(b: &[u8]) -> Result<Bloom, BloomError> {
    let fail = |why: String| Err(BloomError::Format(why));
    if b.len() < HEADER_LEN { return fail(/* ... */); }  // before reading any field
    if b[0..4] != MAGIC { return fail(/* ... */); }
    // ... read version, m, k, reserved, n (little-endian) ...
    if version != VERSION { return fail(/* ... */); }
    if m == 0 || k == 0 { return fail(/* ... */); }
    if m > MAX_BITS { return fail(/* ... */); }
    if reserved != 0 { return fail(/* ... */); }
    let want = HEADER_LEN as u64 + m.div_ceil(8);
    if b.len() as u64 != want { return fail(/* ... */); } // exact: never trust m alone
    Ok(Bloom { m, k, n_inserted: n, bits: b[HEADER_LEN..].to_vec() })
}
```

Full solution: [`bloom.rs`](../../../../course/solutions/rust/crates/tl-ds/src/bloom.rs).

### 5.3 Why this solution

- **Cost.** `insert` and `contains` are $O(k + |x|)$: one pass of FNV over the item, one `mix64`, and $k$ bit operations. Space is $m$ bits, about $1.44 \log_2(1/p)$ per item; `union` is $O(m/8)$.
- **Against a hash set of paragraph hashes.** Exact and simple, but about 68 bytes per entry in Python against about 1.2 bytes per entry for a 1% filter. Because dedup confirms every positive exactly anyway, a filter's false positives cost a little time and never correctness.
- **Against $k$ independent hash functions.** Same false-positive rate asymptotically, $k$ times the hashing work, and $k$ algorithms every other implementation must reproduce bit for bit.
- **Against a counting Bloom filter or a cuckoo filter.** They support deletion, which dedup never needs; a counting filter costs 4 or more times the space.
- **Against `Vec<bool>` or a `u64` word array.** `Vec<bool>` is 8 times larger. A `u64` array is equally compact in memory but makes the on-disk layout depend on word order; bytes with LSB-first bits are the layout the format fixes.
- **Tradeoffs accepted.** `bit_positions` allocates a small `Vec` per call (simple and testable; a hot loop could iterate instead). The filter has fixed capacity: inserting far more than $n$ items raises the rate as $(1 - e^{-kn/m})^k$, and the header's `n_inserted` lets a reader see that it happened.

### 5.4 Explain it back

**Why is $h_2$ forced odd?**

<details><summary>Answer</summary>

When $m$ is a power of two, the probes $h_1 + i\,h_2 \bmod m$ cycle with period $m / \gcd(h_2, m)$. An even $h_2$ shares a factor of 2 with $m$, so probes can repeat within $k$ steps (in the extreme $h_2 \equiv 0$, all $k$ probes hit one bit and the rate is $f$, not $f^k$). An odd $h_2$ is coprime to $m$, so the $k$ probes are distinct.

</details>

**Why can a Bloom filter never give a false negative, and what single change to the code would break that?**

<details><summary>Answer</summary>

`insert` sets exactly the bits `contains` checks (both call `bit_positions`), and nothing ever clears a bit. Any change that makes the two disagree breaks it: setting fewer than $k$ bits in `insert`, or AND instead of OR in `union` (which clears bits set by only one side).

</details>

**Why does `optimal_m` take the ceiling and `optimal_k` round?**

<details><summary>Answer</summary>

$m$ is a resource: fewer bits than $-n \ln p / (\ln 2)^2$ means the rate exceeds the target, so it must round up. $k$ is an optimum of a smooth function, so the nearest integer is best; truncation (6.64 to 6) lands further from the optimum and raises the rate. Both choices are also fixed by the format so every implementation builds the same filter.

</details>

**Why does `from_bytes` require the length to equal $32 + \lceil m/8 \rceil$ exactly, rather than at least that?**

<details><summary>Answer</summary>

Shorter would leave `bits` too small, and a probe near $m$ would index past it and panic later, far from the cause. Longer means the file is not what its header says (wrong $m$, a concatenation, corruption); accepting it would misread data silently. Exact equality refuses both at load time.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Flooring $m$ instead of taking the ceiling | the filter is a bit short: the rate misses its target and bytes differ from every other implementation | `optimal_m` uses `.ceil()` |
| Using `mix64(h1)` without forcing it odd | probes coincide when $m$ is a power of two; bits differ from the format | `bit_positions` uses `mix64(h1) \| 1` |
| Numbering bits from the most significant end | bytes read back wrong in every other implementation | bit `g` is `1 << (g % 8)` in byte `g / 8` |
| Truncating $k$ instead of rounding | 6.64 becomes 6 hashes; more false positives | `optimal_k` uses `.round()` |
| Forgetting to add the counts in `union` | the header's `n_inserted` is wrong after a merge | `union` adds `other.n_inserted` |
| Setting fewer than $k$ bits on insert | false negatives: duplicates slip through dedup | `insert` sets every position `bit_positions` returns |
| Dropping the $i \cdot h_2$ term | all $k$ probes hit one bit: the rate is $f$, not $f^k$ | each probe is `h1 + i * h2` |
| AND instead of OR in `union` | the union forgets members of both sets | `union` ORs the bit arrays |
| Trusting the length in `from_bytes` | a truncated file loads and later panics on an out-of-range bit | `from_bytes` requires the exact length |
| FNV-1 (multiply, then XOR) instead of FNV-1a | every position differs from the format and from `kv-block.md` hashes | `fnv1a64` xors before it multiplies |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M06b` | the false-positive rate, the optimal $k$, and the bits-per-item bound of section 2.2 |
| Back | `M06.3` | FNV-1a and the SplitMix64 finalizer |
| Forward | `data.03` | reading, not a call: its Python `exact_dedup` sizes its own screen at about 10 bits per paragraph with these formulas, inserts each paragraph hash, and confirms every positive exactly; for the same items both screens set the same bits |

This module is optional: no other module depends on it, since `data.03` never calls Rust.

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Bloom` | [datatrove](https://github.com/huggingface/datatrove) exact dedup | sharded hashing with sort-merge and no filter at all, trading disk for certainty | `src/datatrove/pipeline/dedup/exact_substrings.py`, `sentence_dedup.py` |
| fixed $k$ double hashing | Redis Bloom filters (RedisBloom) | scalable filters that add layers as $n$ grows past the design | `src/sb.c` |
| bits per item at a target rate | cuckoo filters, xor filters | deletion, and about 20% fewer bits at low false-positive rates | Fan et al. (2014); Graf and Lemire (2020) |
| `to_bytes` | Parquet and ORC Bloom filters | split-block filters stored per column chunk, read before a scan | the Apache Parquet `BloomFilter.md` spec |
