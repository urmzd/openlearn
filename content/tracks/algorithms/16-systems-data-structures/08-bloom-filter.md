<!-- ol:module ds.08 -->
# Bloom filter

## Overview

| | |
|---|---|
| **Module** | `ds.08` · side · Rust · Pass 3 · 4 to 5 h (optional) |
| **You build** | `rust/crates/tl-ds/src/bloom.rs`: `fnv1a64`, `mix64`, `optimal_m`, `optimal_k`, `predicted_fp_rate`, `Bloom` (`with_rate`, `new`, `bit_positions`, `insert`, `contains`, `union`, `count_ones`, `to_bytes`, `from_bytes`) |
| **Contract** | byte layout and hashing: [`formats/bloom.md`](../../../../course/contracts/formats/bloom.md) |
| **Tests** | `course/tests/rust/ds_08.rs`; what they check: section 4 · parity suite `bloom` · your own tests in `rust/crates/tl-ds/tests/ds08_bloom.rs`, rung R4 (proptest), graded by mutation (threshold 0.80, `s01` to `s10` required) |
| **Needs** | reading: `S-M06b` the [false-positive rate and the optimal k](../../math/06-discrete-math-2/91-problem-set-b.md), `M06.3` [FNV-1a and the SplitMix finalizer](../../math/06-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) |
| **Used by** | No runtime module. This optional Rust module is checked on its own and against the shared golden bit arrays (`ol parity bloom`); `data.03` writes an independent Python Bloom screen to the same format. |
| **Milestone** | none (optional module: MS-corpus runs the Python screen of data.03; `ol parity bloom` compares the two) |
| **Optional depth** | Bloom, *Space/Time Trade-offs in Hash Coding with Allowable Errors* (1970); Kirsch and Mitzenmacher, *Less Hashing, Same Performance: Building a Better Bloom Filter* (2006); Broder and Mitzenmacher, *Network Applications of Bloom Filters: A Survey* (2004) |

## Key Takeaways

- A Bloom filter is $m$ bits and $k$ hash functions: insert sets $k$ bits, `contains` checks them, so a member is never missed and a non-member is reported present with probability about $(1 - e^{-kn/m})^k$ (`no_false_negatives`, `fp_rate_within_three_sigma`).
- For $n$ items at rate $p$, $m = \lceil -n \ln p / (\ln 2)^2 \rceil$ bits and $k = \mathrm{round}(\frac{m}{n} \ln 2)$ hashes: 4 items at 10% need 20 bits and 3 hashes (`hand_example_sizing_and_bits`, `sizing_formulas`).
- Two hashes are enough: $g_i = h_1 + i\,h_2 \bmod m$ (double hashing) with $h_1$ = FNV-1a 64 and $h_2$ = the SplitMix finalizer of $h_1$, forced odd (`hash_functions_hand_values`).
- Union is bitwise OR of two filters with the same $m$ and $k$, which is exactly the filter of both sets (`union_is_the_filter_of_both_sets`).
- A fixed byte layout makes the filter portable: Rust and the oracle produce the same 35 bytes for the worked example (`golden_bytes_match_the_format`).

## How to work this chapter

```bash
ol start ds.08              # stubs bloom.rs in tl-ds
ol tests ds.08              # read the test catalog first
ol check ds.08              # Rust tests, then grades your tests
ol parity bloom             # compare the Rust filter with the golden bit arrays
```

No other module calls this code: languages meet only over processes and files, and the one Bloom filter on the serving path, the Python screen of `data.03`, is written in Python. The byte format is what the two share, and `ol parity bloom` holds both to it. Write the hashing and sizing first and check them on the worked example, then bytes and round trips.

---

## 1. Why now

Your corpus pipeline (`data.01` to `data.08`, batch B5) must drop exact duplicate paragraphs from millions of documents. A `set` of paragraph hashes in Python is 8 bytes of hash plus about 60 bytes of overhead per entry; a Bloom filter answers "have I seen this?" in about 10 bits per paragraph, with no false negatives and a false-positive rate you choose. The dedup stage then confirms every "seen" answer exactly (a sort-merge on the full hashes), so a false positive costs time, never data. The serialized format is useful across processes and implementations; `formats/bloom.md` fixes that layout.

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

### 2.4 What the tests can measure

The approximation of 2.2 assumes independent bits. Given an actual filter, the exact probability that a random fresh item is a false positive is $f^k$ with $f$ the measured fraction of set bits (each probe lands on a uniform bit). Over $N$ fresh items the count of false positives is then Binomial$(N, f^k)$, with mean $N f^k$ and standard deviation $\sqrt{N f^k (1 - f^k)}$: the statistical test asks the count to lie within 3 standard deviations, which a correct filter fails about 0.3% of the time (it passes seeds 0 to 30).

### 2.5 Union and bytes

Two filters with the same $m$ and $k$ (and so the same hash positions) merge by bitwise OR: the result has exactly the bits that inserting both sets would set. `n_inserted` counts `insert` calls and a union adds the counts. `to_bytes` writes the 32-byte little-endian header (magic `TLBF`, version 1, $m$, $k$, a reserved zero, `n_inserted`) and then the $\lceil m/8 \rceil$ bytes of bits; `from_bytes` rejects anything else.

The Python corpus pipeline has a separate implementation in `python/corpus/dedup.py`. This module focuses on the Rust filter and its serialized format.

## 3. Worked example by hand

`with_rate(4, 0.1)`: $-4 \ln 0.1 / (\ln 2)^2 = 9.2103 / 0.48045 = 19.17$, so $m = 20$; $\frac{20}{4} \ln 2 = 3.466$, so $k = 3$. The bit array is $\lceil 20/8 \rceil = 3$ bytes.

`cat`: $h_1 = \text{fnv1a64}(\texttt{cat}) = \texttt{f5e307190ce4a327}$ and $h_2 = \text{mix64}(h_1) \lor 1 = \texttt{abc55d317f9d6793}$, which give $g = 11, 14, 17$ (the format page lists every item's positions). `dog` gives $13, 0, 3$. After inserting both, the set bits are 0 and 3 (byte 0: $2^0 + 2^3 = \texttt{09}$), 11, 13, 14 (byte 1: bits 3, 5, 6 of that byte, $8 + 32 + 64 = \texttt{68}$), and 17 (byte 2: bit 1, $\texttt{02}$): `09 68 02`. `bird` probes $14, 7, 16$: bit 7 is clear, so `contains(b"bird")` is false. Six of 20 bits are set, $f = 0.3$, and a fresh item's chance of a false positive is $0.3^3 = 2.7\%$.

`to_bytes` is the 35 bytes

```text
54 4c 42 46  01 00 00 00  14 00 00 00 00 00 00 00   "TLBF", version 1, m = 20
03 00 00 00  00 00 00 00  02 00 00 00 00 00 00 00   k = 3, reserved 0, n_inserted = 2
09 68 02                                            the bits
```

This is `hand_example_sizing_and_bits` in Rust.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_sizing_and_bits` | unit | section 3: $m = 20$, $k = 3$, the positions of cat, dog, bird, the 35 bytes | you, the format page, and the tests agree |
| `hash_functions_hand_values` | golden | FNV-1a of "", "a", "cat"; mix64 of 0 and 1 | every implementation sets the same bits |
| `sizing_formulas` | unit | $m$ and $k$ for five $(n, p)$, including the rounding of $k$; byte length | the screen is as small as the target allows |
| `bad_parameters_are_errors` | boundary | $n = 0$, $p \notin (0, 1)$, NaN, $k = 0$, $m > 2^{40}$ are `BadParams` | a bad config fails at start, not with a 128 GiB allocation |
| `no_false_negatives` | property | 5,000 random items found in their filter, after a byte round trip, and in the union | dedup never keeps a duplicate it has seen |
| `fp_rate_within_three_sigma` | statistical | 100,000 fresh items: false positives within 3 sd of $N f^k$, and below 2% at a 1% design | the filter is as good as its sizing promises |
| `union_is_the_filter_of_both_sets` | unit | OR of two filters equals the filter of both; counts add; mismatched $m$ or $k$ refused | `data.03` merges per-shard screens |
| `golden_bytes_match_the_format` | golden | 10 oracle cases (empty items, odd $m$, $p = 10^{-6}$): bytes and probe answers | the parity suite `bloom` |
| `from_bytes_rejects_malformed_input` | boundary | bad magic, version, $m = 0$, $k = 0$, reserved, short, long, inconsistent length | a corrupt file is refused, not misread |
**Your tests (rung R4).** Write `rust/crates/tl-ds/tests/ds08_bloom.rs` with proptest (craft.04 teaches it): the worked example bytes, the sizing table, and the properties "every inserted item is present", "bytes round-trip", "union equals one filter of both sets", plus a check of a mismatched union and of malformed bytes. Use a fixed proptest seed and no failure files (`Config { rng_seed: RngSeed::Fixed(..), failure_persistence: None, .. }`). At least 80% of the planted bugs, and every planted bug in `bloom.rs` (`s01` to `s10`), must make one fail.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Flooring $m$ instead of taking the ceiling | the filter is a bit short: the rate misses its target and bytes differ from every other implementation | `sizing_formulas`, `hand_example_sizing_and_bits` (mutant `s01`) |
| Using `mix64(h1)` without forcing it odd | probes coincide when $m$ is a power of two; bits differ from the format | `hand_example_sizing_and_bits`, `golden_bytes_match_the_format` (mutant `s02`) |
| Numbering bits from the most significant end | your bytes read back wrong in every other implementation | `hand_example_sizing_and_bits` (mutant `s03`) |
| Truncating $k$ instead of rounding | 6.64 becomes 6 hashes; more false positives | `sizing_formulas` (mutant `s04`) |
| Forgetting to add the counts in `union` | the header's `n_inserted` is wrong after a merge | `union_is_the_filter_of_both_sets` (mutant `s05`) |
| Setting fewer than $k$ bits on insert | false negatives: duplicates slip through dedup | `no_false_negatives` (mutant `s06`) |
| Dropping the $i \cdot h_2$ term | all $k$ probes hit one bit: the rate is $f$, not $f^k$ | `fp_rate_within_three_sigma` (mutant `s07`) |
| AND instead of OR in `union` | the union forgets members of both sets | `no_false_negatives`, `union_is_the_filter_of_both_sets` (mutant `s08`) |
| Trusting the length in `from_bytes` | a truncated file loads and later panics on an out-of-range bit | `from_bytes_rejects_malformed_input` (mutant `s09`) |
| FNV-1 (multiply, then XOR) instead of FNV-1a | every position differs from the format and from `kv-block.md` hashes | `hash_functions_hand_values` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M06b` | the false-positive rate, the optimal $k$, and the bits-per-item bound of section 2.2 |
| Back | `M06.3` | FNV-1a and the SplitMix64 finalizer |
| Forward | `data.03` | reading, not a call: its Python `exact_dedup` sizes its own screen at about 10 bits per paragraph with these formulas, inserts each paragraph hash, and confirms every positive exactly; `ol parity bloom` checks that both screens write the same bytes |

This module is optional: skipping it blocks no other module, since `data.03` never calls Rust.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Bloom` | [datatrove](https://github.com/huggingface/datatrove) exact dedup | sharded hashing with sort-merge and no filter at all, trading disk for certainty | `src/datatrove/pipeline/dedup/exact_substrings.py`, `sentence_dedup.py` |
| fixed $k$ double hashing | Redis Bloom filters (RedisBloom) | scalable filters that add layers as $n$ grows past the design | `src/sb.c` |
| bits per item at a target rate | cuckoo filters, xor filters | deletion, and about 20% fewer bits at low false-positive rates | Fan et al. (2014); Graf and Lemire (2020) |
| `to_bytes` | Parquet and ORC Bloom filters | split-block filters stored per column chunk, read before a scan | the Apache Parquet `BloomFilter.md` spec |
