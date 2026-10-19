# Interface migration: KV format v1 to v2

## Overview

| | |
|---|---|
| **Module** | `craft.13` · build · rust · Pass 11 · 1 h reading |
| **Solution** | [`lib.rs`](../../../../course/solutions/rust/crates/tl-kv-format/src/lib.rs): `decode`, `encode`, `upgrade_v1`, `Envelope`, `Block`, `Shape`, `Error`; the offline converter `kv-migrate` in [`main.rs`](../../../../course/solutions/rust/crates/tl-kv-format/src/main.rs); crate manifest [`Cargo.toml`](../../../../course/solutions/rust/crates/tl-kv-format/Cargo.toml) |
| **Contract** | [`course/contracts/formats/kv-block.md`](../../../../course/contracts/formats/kv-block.md), sections "Export envelope", "Payload, format v2", and "Reader rules" |
| **Needs** | [`craft.12`](../09-releases-and-deprecation/02-deprecation-policy.md) (deprecation policy) |
| **Used by** | [`ops.04`](../../systems/05-incident-response-and-chaos/04-kv-v2-migration.md), the KV v2 migration drill |

## Key Takeaways

- The v2 envelope keeps the 28-byte header and block records; dtype 3 means fp8 e4m3.
- Each (layer, K or V, head) slab carries one f32 scale, max absolute value divided by 448, or 1.0 for all-zero data.
- The offline converter validates v1 CRC and dimensions before atomically writing v2 output.

## How to read this chapter

Sections 1 to 3 explain why a format change needs a staged rollout and work one block through the conversion by hand. Section 4 fixes the byte layout and the cases a correct converter must handle. Section 5 walks through the solution: validation first, then the two payload encodings, then the one-way upgrade and its atomic file write. Try to predict each step of the solution before reading it.

---

## 1. Why now

KV v2 changes the representation consumed by both the C pool and Rust transfer path. A coordinated migration is necessary because mixed versions can otherwise corrupt blocks or reject valid transfers.

## 2. Principles

Document the wire envelope and negotiation rules first. Readers become compatible before writers emit v2. Keep v1 readable through the rollback window; reject unsupported versions before allocating or mutating state.

## 3. Worked example

Ship a reader that accepts v1 and v2, then deploy writers that emit v2 only after every reader advertises support. Test mixed pairs, fp8 scales, content hash, malformed lengths, and the downgrade path. Track format version by peer.

Take the contract's worked v1 envelope: `B=2, L=1, Hkv=1, D=2`, one full block with hash `0xA91AB0C1027B9366`, K = `[[1, 2], [3, 4]]` and V = `[[0.5, -1], [0, 0.25]]`. Its v1 payload is `1 × 2 × 1 × 2 × 2 × 2 = 16` bytes of f16. The v2 payload is `1 × 2 × 1 × (2 × 2 + 4) = 16` bytes: eight fp8 values plus two f32 scales. The two sizes happen to match for this tiny shape; the interpretation differs.

Convert by hand. The K slab is `1 2 3 4`, so `amax = 4` and `scale = 4/448`. Dividing gives `112 224 336 448`. In e4m3 (bias 7, three mantissa bits) `112 = 1.75 × 2^6` is code `0x6e`, `224 = 1.75 × 2^7` is `0x76`, and `448 = 1.75 × 2^8` is `0x7e`, the largest finite value. `336` falls exactly between the neighbours `320 = 1.25 × 2^8` (`0x7a`) and `352 = 1.375 × 2^8` (`0x7b`); the tie goes to the even code `0x7a`, so the third K value decodes to `320 × 4/448 ≈ 2.857`, not 3. The V slab has `amax = 1` and `scale = 1/448`: `0.5 → 224 → 0x76`, `-1 → -448 → 0xfe`, `0 → 0x00`, `0.25 → 112 → 0x6e`. The converted 60-byte envelope, as the solution produces it:

```
54 4c 4b 56 02 00 03 00 01 00 00 00 02 00 00 00     magic, version 2, dtype 3 (e4m3), n_blocks 1, block_tokens 2
01 00 00 00 01 00 00 00 02 00 00 00                 n_layers 1, n_kv_heads 1, head_dim 2
66 93 7b 02 c1 b0 1a a9 02 00 00 00                 block_hash (unchanged), n_tokens 2
6e 76 7a 7e  76 fe 00 6e                            K codes, V codes
25 49 12 3c  25 49 12 3b                            f32 scales 4/448, 1/448
dd 13 89 2e                                         crc32c of everything before it
```

The hash and fill are copied untouched: they identify the block's tokens, not its numbers, so prefix-cache routing still works after the migration. The decoded values are `1, 2, 2.857, 4, 0.5, -1, 0, 0.25`, which is the loss fp8 costs.

## 4. Interface and cases

The layout is the contract's: a 28-byte header (`"TLKV"`, u16 version, u16 dtype, u32 `n_blocks`, then `B, L, Hkv, D`), `n` records of `u64 hash, u32 n_tokens, payload`, and a trailing CRC-32C. v1 payloads are `f16 [L][2][Hkv][B][D]`; v2 payloads are `e4m3 [L][2][Hkv][B][D]` followed by `f32 [L][2][Hkv]` scales.

```rust
pub fn decode(bytes: &[u8], expected: Shape) -> Result<Envelope, Error>;   // reads v1 and v2
pub fn encode(env: &Envelope, version: u16) -> Result<Vec<u8>, Error>;     // writes v1 or v2
pub fn upgrade_v1(bytes: &[u8], shape: Shape) -> Result<Vec<u8>, Error>;   // v1 in, v2 out, never v2 in
```

The binary is `kv-migrate <input> <output> <block_tokens> <layers> <kv_heads> <head_dim>`; it exits 2 with a message on any error.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example converts to the golden bytes | unit | Converts the section 3 envelope and compares the exact bytes, then checks hash, fill, and values within 0.2. | Establishes the migration result by hand before broader cases. |
| both formats round-trip with their own dtype | unit | Reads and writes v1 and v2 with the correct dtype for each. | Lets readers roll forward while retaining rollback support. |
| corruption is rejected before any block is returned | boundary | Rejects bad CRC before exposing payload data. | Prevents corrupted KV blocks from reaching decode. |
| shape mismatch and unknown version are rejected | boundary | Rejects incompatible dimensions and unsupported format versions. | Keeps peers from misreading valid-looking bytes. |
| a partial tail is zero-filled and never claims a full hash | boundary | Makes padding deterministic and avoids a false full-block hash. | Prevents dedupe from treating incomplete blocks as complete. |
| an all-zero slab uses unit scale | boundary | Gives an all-zero slab finite unit scale. | Avoids invalid quantization metadata. |
| the upgrader refuses a v2 input | boundary | Refuses a second conversion. | Prevents silent repeated quantization. |
| the process converter writes its output atomically | integration | Exercises the subprocess converter and atomic output path. | Keeps Python and Rust separated by the documented file boundary. |

## 5. The solution, step by step

### 5.1 From first principles

1. A reader that trusts any byte before checking the whole envelope can hand a corrupt block to decode. So `decode` checks magic, version, dtype, shape, exact length, and CRC, in that order, and only then builds a single block.
2. Lengths come from untrusted header fields multiplied together. `L × 2 × Hkv × B × D` can overflow `usize` on a hostile header, and a wrapped product could match a short buffer. So every size is computed with checked arithmetic, and an overflow is its own error.
3. The version decides the dtype: v1 must say 1 (f16), v2 must say 3 (e4m3). A mismatched pair is ambiguous, so it is rejected rather than guessed.
4. fp8 e4m3 has a largest finite value of 448 and about two significant decimal digits. Values must be scaled into range, and one scale per slab (layer, K or V, head) is the granularity at which magnitudes differ in attention. Choosing `scale = amax / 448` maps the largest value onto 448 exactly, so nothing saturates. An all-zero slab would give scale 0, which cannot be divided by and which the reader rejects, so it gets 1.0.
5. Positions past `n_tokens` carry no data. Writing them as zero makes the bytes a pure function of the block's real contents, which is what golden comparisons and content hashing need. A non-zero hash promises a full block, so `n_tokens == B` is enforced when the hash is set.
6. Quantization loses information. Converting v2 again would re-quantize already rounded values without any warning, so the upgrader accepts v1 only.
7. A converter that crashes halfway must not leave a truncated file under the output name, where a later reader might pick it up. Write to `<output>.tmp` and rename: a rename on one filesystem is atomic, so readers see the old file or the complete new one.

### 5.2 The code, annotated

The solution file packs each function onto a few long lines. The quotes below are reformatted one statement per line, with comments added; the logic is unchanged.

**Sizes, with overflow as an error.**

```rust
fn product(s: Shape) -> Result<usize, Error> {
    // elements per block: L * 2 (K and V) * Hkv * B * D, every multiply checked
    [s.layers as usize, 2, s.kv_heads as usize, s.block_tokens as usize, s.head_dim as usize]
        .into_iter()
        .try_fold(1usize, |a, b| a.checked_mul(b).ok_or(Error::Overflow))
}
fn slab_count(s: Shape) -> Result<usize, Error> {
    // one scale per (layer, K or V, head)
    (s.layers as usize).checked_mul(2)
        .and_then(|x| x.checked_mul(s.kv_heads as usize))
        .ok_or(Error::Overflow)
}
```

**CRC-32C**, bit by bit with the reflected Castagnoli polynomial. It is slow per byte but needs no table or dependency, and the converter is offline.

```rust
fn crc32c(bytes: &[u8]) -> u32 {
    let mut c = !0u32;                                   // initial value 0xFFFFFFFF
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0x82f63b78 } else { c >> 1 };
        }
    }
    !c                                                   // final XOR 0xFFFFFFFF
}
```

**e4m3.** Decoding follows the format directly: 1 sign bit, 4 exponent bits with bias 7, 3 mantissa bits, subnormals at exponent 0, and only `S.1111.111` as NaN (there is no infinity). Encoding does not do bit tricks; it searches all 127 non-negative finite codes for the nearest, breaking ties toward an even code.

```rust
fn e4_decode(b: u8) -> f32 {
    let sign = if b & 0x80 == 0 { 1.0 } else { -1.0 };
    let e = (b >> 3) & 15;
    let m = b & 7;
    if e == 15 && m == 7 { return f32::NAN }
    if e == 0 { sign * (m as f32 / 8.0) * 2f32.powi(-6) }        // subnormal
    else { sign * (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7) }
}
fn e4_encode(x: f32) -> Result<u8, Error> {
    if !x.is_finite() { return Err(Error::NonFinite) }
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = x.abs().min(448.0);                          // clamp: the scale already keeps |x| <= 448
    let mut best = 0u8;
    let mut err = f32::INFINITY;
    for b in 0..=0x7e {                                  // 0x7f is NaN, never a candidate
        let v = e4_decode(b);
        let d = (v - a).abs();
        if d < err || (d == err && b & 1 == 0) {         // round to nearest, ties to even
            best = b;
            err = d;
        }
    }
    Ok(sign | best)
}
```

Because the codes are visited in increasing order and a tie only replaces `best` with an even code, `336` becomes `0x7a` (even), as in section 3. The f16 helpers `read_f16` and `write_f16` play the same role for v1 and reject non-finite values and overflow past f16's range.

**Decode: validate everything, then build.**

```rust
pub fn decode(bytes: &[u8], expected: Shape) -> Result<Envelope, Error> {
    if bytes.len() < HEADER + 4 { return Err(Error::Truncated) }    // header plus CRC
    if &bytes[..4] != b"TLKV" { return Err(Error::Magic) }
    let version = u16at(bytes, 4);
    let dtype = u16at(bytes, 6);
    if version != V1 && version != V2 { return Err(Error::Version) }
    if dtype != if version == V1 { 1 } else { 3 } { return Err(Error::Dtype) }   // step 3
    let n = u32at(bytes, 8) as usize;
    let shape = Shape { block_tokens: u32at(bytes, 12), layers: u32at(bytes, 16),
                        kv_heads: u32at(bytes, 20), head_dim: u32at(bytes, 24) };
    if shape != expected { return Err(Error::Shape) }               // the receiving pool's shape
    let elems = product(shape)?;
    let slabs = slab_count(shape)?;
    let payload = if version == V1 {
        elems.checked_mul(2).ok_or(Error::Overflow)?                 // two bytes per f16
    } else {
        elems.checked_add(slabs.checked_mul(4).ok_or(Error::Overflow)?).ok_or(Error::Overflow)?
    };
    let rec = 12usize.checked_add(payload).ok_or(Error::Overflow)?; // hash + fill + payload
    let len = HEADER.checked_add(n.checked_mul(rec).ok_or(Error::Overflow)?)
        .and_then(|x| x.checked_add(4)).ok_or(Error::Overflow)?;
    if len != bytes.len() { return Err(Error::Length) }             // exact, not "at least"
    if crc32c(&bytes[..len - 4]) != u32at(bytes, len - 4) { return Err(Error::Checksum) }
    // ... only now walk the records
```

The length check comes before the CRC so the CRC never reads out of bounds, and both come before any block is built, so a corrupt envelope returns an error and nothing else.

```rust
    let slab_elems = elems / slabs;                      // B * D values share one scale
    let mut blocks = Vec::with_capacity(n);
    let mut at = HEADER;
    for _ in 0..n {
        let hash = u64at(bytes, at);
        let fill = u32at(bytes, at + 8);
        // fill in 1..=B, and a registered (hashed) block must be full: step 5
        if fill == 0 || fill > shape.block_tokens || (hash != 0 && fill != shape.block_tokens) {
            return Err(Error::Tokens)
        }
        let p = &bytes[at + 12..at + 12 + payload];
        let mut values = Vec::with_capacity(elems);
        if version == V1 {
            for x in p.chunks_exact(2) { values.push(read_f16(u16::from_le_bytes([x[0], x[1]]))) }
        } else {
            for (i, &b) in p[..elems].iter().enumerate() {
                // the scales follow all the codes; element i belongs to slab i / slab_elems
                let s = elems + (i / slab_elems) * 4;
                let scale = f32::from_le_bytes(p[s..s + 4].try_into().unwrap());
                if !scale.is_finite() || scale <= 0.0 { return Err(Error::Length) }
                values.push(e4_decode(b) * scale)
            }
        }
        blocks.push(Block { hash, n_tokens: fill, values });
        at += rec;
    }
    Ok(Envelope { version, shape, blocks })
}
```

**Encode: the scale rule and zero padding.**

```rust
pub fn encode(env: &Envelope, version: u16) -> Result<Vec<u8>, Error> {
    // ... version check, sizes as in decode, header written little-endian with the version's dtype
    for block in &env.blocks {
        // the same token rules as the reader, so the writer cannot emit what a reader rejects
        if block.values.len() != elems || block.n_tokens == 0 || block.n_tokens > s.block_tokens
            || (block.hash != 0 && block.n_tokens != s.block_tokens) { return Err(Error::Tokens) }
        out.extend_from_slice(&block.hash.to_le_bytes());
        out.extend_from_slice(&block.n_tokens.to_le_bytes());
        if version == V1 {
            for (i, &x) in block.values.iter().enumerate() {
                let pos = (i % se) / (s.head_dim as usize);   // token position inside the slab
                let v = if pos >= block.n_tokens as usize { 0.0 } else { x };   // zero the tail
                out.extend_from_slice(&write_f16(v)?.to_le_bytes())
            }
        } else {
            let mut scales = Vec::with_capacity(slabs);
            for slab in 0..slabs {
                let x = &block.values[slab * se..(slab + 1) * se];
                let max = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                scales.push(if max == 0.0 { 1.0 } else { max / 448.0 });   // step 4
            }
            for (i, &x) in block.values.iter().enumerate() {
                let pos = (i % se) / (s.head_dim as usize);
                let v = if pos >= block.n_tokens as usize { 0.0 } else { x };
                out.push(e4_encode(v / scales[i / se])?)
            }
            for scale in scales { out.extend_from_slice(&scale.to_le_bytes()) }   // after all codes
        }
    }
    let crc = crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    Ok(out)
}
```

Here `se` is the slab size `B × D`. Inside a slab the layout is `[B][D]`, so `(i % se) / D` is the token position, which is how the tail is found without a nested loop. The scale is taken over the whole slab; in the converter path the slab comes from a decoded v1 block whose tail is already zero, so this equals the contract's "amax over filled positions".

**The upgrade and the process boundary.**

```rust
pub fn upgrade_v1(bytes: &[u8], shape: Shape) -> Result<Vec<u8>, Error> {
    let env = decode(bytes, shape)?;                     // full validation, including CRC
    if env.version != V1 { return Err(Error::Version) } // step 6: never quantize twice
    encode(&env, V2)
}
```

```rust
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 { return Err("usage: kv-migrate <input> <output> <block_tokens> <layers> <kv_heads> <head_dim>".into()); }
    let n = |i: usize| args[i].parse::<u32>().map_err(|e| format!("invalid dimension {}: {e}", args[i]));
    let shape = Shape { block_tokens: n(3)?, layers: n(4)?, kv_heads: n(5)?, head_dim: n(6)? };
    let input = fs::read(&args[1])?;
    let output = upgrade_v1(&input, shape)?;             // all work in memory first
    let tmp = args[2].to_owned() + ".tmp";
    fs::write(&tmp, output)?;
    fs::rename(tmp, &args[2])?;                          // step 7: atomic replace
    Ok(())
}
```

The expected shape is a command-line argument, not read from the input, so the operator states the receiving pool's dimensions and a file for another model is refused with `Shape`.

Full solution: [`lib.rs`](../../../../course/solutions/rust/crates/tl-kv-format/src/lib.rs), [`main.rs`](../../../../course/solutions/rust/crates/tl-kv-format/src/main.rs).

### 5.3 Why this solution

- **Cost.** Decoding and encoding are linear in the envelope size. The brute-force e4m3 encoder does 127 comparisons per element, which is fine for an offline converter and easy to check against the format definition; the contract's reference for e4m3 conversion in the C pool is [`tinyllm/numerics.h`](../../../../course/contracts/c/include/tinyllm/numerics.h).
- **Validate-then-build** rather than streaming blocks out as they are parsed: a streaming reader would have handed earlier blocks to the caller before discovering a bad CRC at the end.
- **Per-slab scales** rather than one scale per block or per element: a single block scale lets one large head crush every other head into a few codes; per-element scales cost 4 bytes per byte of data and defeat the purpose of fp8.
- **A separate offline binary** rather than an in-process migration: Python and the engine meet at a documented file boundary, the conversion can be rerun on original inputs, and a rollback is just keeping the v1 files.
- **Tradeoff accepted:** v2 is lossy (the section 3 value 3 becomes 2.857). The migration is safe only because readers learn v2 before writers emit it, and v1 stays readable through the rollback window.

### 5.4 Explain it back

**Why is the length compared with `!=` and not `<`?**

<details><summary>Answer</summary>

Trailing or missing bytes mean the header's `n_blocks` or shape does not describe the buffer. If extra bytes were tolerated, the CRC would sit at `len - 4` computed from the header while the buffer ended elsewhere, and two readers could disagree about which bytes are covered. An exact length makes the CRC's position, and therefore what it protects, unambiguous.

</details>

**Why does an all-zero slab get scale 1.0 instead of 0?**

<details><summary>Answer</summary>

Encoding divides by the scale, so 0 would give NaN codes, and the reader rejects a scale that is not finite and positive. Any positive scale decodes zero codes to zero; 1.0 is the simplest finite choice.

</details>

**What goes wrong if `upgrade_v1` accepted a v2 input?**

<details><summary>Answer</summary>

It would decode the fp8 values, compute new scales, and quantize again. Each pass can move values to different codes, so data degrades silently with every rerun of the migration. Refusing with `Version` makes the operation idempotent in effect: rerunning on the original v1 files gives the same output.

</details>

**Why write to `<output>.tmp` and rename instead of writing `<output>` directly?**

<details><summary>Answer</summary>

A crash or full disk during `fs::write` would leave a truncated file under the real name, and a reader might load it. A rename within one filesystem is atomic, so the output name always holds either nothing (or the previous file) or a complete, CRC-valid envelope.

</details>

**Why does a block with a non-zero hash have to be full?**

<details><summary>Answer</summary>

The hash is the prefix-cache identity of exactly `B` tokens. A partial block has no stable identity yet; if it claimed one, the prefix cache could match a later request against a block missing some of its tokens.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Emitting v2 before all readers advertise support | A mixed-version peer rejects or misreads a block. | `decode` reads both versions, so readers can be upgraded first; `encode` writes either version for the rollback window. |
| 2 | Returning data before validating the envelope CRC | Corrupt blocks can reach the scheduler as plausible KV data. | `decode` checks exact length and CRC before the record loop builds any `Block`; see 5.2. |
| 3 | Re-converting an already quantized v2 block | A second conversion adds loss without an explicit error. | `upgrade_v1` returns `Error::Version` for any input that is not v1. |
| 4 | Unchecked size arithmetic on header fields | A crafted header wraps the size and passes the length check. | `product`, `slab_count`, and every size use `checked_mul`/`checked_add` and fail with `Error::Overflow`. |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | [`craft.12`](../09-releases-and-deprecation/02-deprecation-policy.md) | The deprecation policy sets the rollback window during which v1 must stay readable. |
| Back | [`craft.02`](../06-documentation-writing/01-architecture-decision-records.md) | An ADR records why the format changes and the rollout order (readers first). |
| Back | [`course/contracts/formats/kv-block.md`](../../../../course/contracts/formats/kv-block.md) | The byte layout, scale rule, and reader rules this converter implements. |
| Back | [`L10.6`](../../../courses/build-an-llm/p10-serving/06-disaggregated-prefill-decode.md) | The KV transfer path that carries these envelopes between engines. |
| Forward | [`ops.04`](../../systems/05-incident-response-and-chaos/04-kv-v2-migration.md) | The migration drill: a v1-only reader meets a v2 writer, and the rollout order decides whether that is a clean refusal or an outage. |

## Going further

| Resource | What to inspect |
|---|---|
| [`kv-block.md`](../../../../course/contracts/formats/kv-block.md), worked example (v1) | Decode the v1 bytes by hand, then compare them with the v2 bytes in section 3 above. |
| [fp8 and microscaling formats](../../math/09-numerical-methods-and-floating-point/04-fp8-and-microscaling-formats.md) | Why e4m3 tops out at 448 and how block scaling generalises the per-slab scale. |
