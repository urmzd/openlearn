# Bloom filter: sizing, hashing, bytes
<!-- chapter: content/courses/algorithms/16-systems-data-structures/08-bloom-filter.md -->

<!-- modules: ds.08 (rust/crates/tl-ds/src/bloom.rs), data.03 (exact dedup)
     conformance: parity/bloom -->

The Rust `tl_ds::bloom::Bloom` implementation and Python dedup implementation follow this page, so serialized filters preserve the same membership behavior. All integers are little-endian.

## Sizing

For `n` expected items and a target false-positive rate `p` (`n >= 1`, `0 < p < 1`):

```
m = ceil( -n * ln(p) / (ln 2)^2 )      bits
k = max(1, round( m / n * ln 2 ))      hash functions (round half away from zero)
```

These are the minimizers of the false-positive rate `(1 - e^(-k n / m))^k` (S-M06b). Example: `n = 4, p = 0.1` gives `m = ceil(19.17) = 20`, `k = round(3.47) = 3`.

## Hashing

Item `x` (a byte string) sets or tests bits `g_0 .. g_{k-1}`:

```
h1  = fnv1a64(x)                       FNV-1a 64, as in formats/kv-block.md
h2  = mix64(h1) | 1                    odd, so the k probes differ when m is a power of two
g_i = ((h1 + i * h2) mod 2^64) mod m   for i = 0 .. k-1   (Kirsch-Mitzenmacher double hashing)

mix64(z):  z = (z XOR (z >> 30)) * 0xBF58476D1CE4E5B9 mod 2^64
           z = (z XOR (z >> 27)) * 0x94D049BB133111EB mod 2^64
           return z XOR (z >> 31)                       (the SplitMix64 finalizer)
```

Bit `g` lives in byte `g / 8` at bit position `g % 8` (least significant bit first).

## Bytes (`to_bytes`)

```
offset  size          field
0       4             magic "TLBF" (54 4c 42 46)
4       4             u32 version = 1
8       8             u64 m
16      4             u32 k
20      4             u32 reserved = 0
24      8             u64 n_inserted   insert calls so far (union adds the counts)
32      ceil(m / 8)   the bit array; unused high bits of the last byte are 0
```

`from_bytes` rejects (Python `ValueError`) a wrong magic or version, `m == 0` or `k == 0`, a non-zero reserved field, or a length other than `32 + ceil(m / 8)`.

## Worked example

`with_rate(4, 0.1)`: `m = 20`, `k = 3`, a 3-byte bit array.

| Item | Bits `g_0, g_1, g_2` |
|---|---|
| `cat` | 11, 14, 17 |
| `dog` | 13, 0, 3 |
| `bird` (not inserted) | 14, 7, 16: bit 7 is clear, so `contains` is false |

After inserting `cat` and `dog` the bit array is `09 68 02` (bits 0, 3 in byte 0; 11, 13, 14 in byte 1; 17 in byte 2), and `to_bytes()` is 35 bytes:

```
54 4c 42 46 01 00 00 00 14 00 00 00 00 00 00 00
03 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00
09 68 02
```
