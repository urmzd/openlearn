# KV blocks: hashing and the export envelope
<!-- chapter: content/courses/build-an-llm/11-serving/07-disaggregated-prefill-decode.md -->

<!-- modules: rt.04 (C pool, export and import), L10.4 (prefix cache), L10.6 (transfer), craft.13 (format v2)
     conformance: parity/kv.wire.v1, parity/kv.wire.v2 -->

A KV block holds the attention keys and values of `block_tokens` consecutive token positions for every layer and KV head. The in-memory layout inside the pool is private to `rt.04` ([`tinyllm/kv_pool.h`](../c/include/tinyllm/kv_pool.h)). Two things are shared by every implementation and every version: the **block hash**, which names a block's content so a prefix cache and a KV transfer can deduplicate it, and the **export envelope**, the byte layout of `tl_kv_export`, `tl_kv_import`, and every `KvChunk.payload` of [`tl.kv.v1`](../proto/tl/kv/v1/kv.proto).

All integers are little-endian. Symbols:

| Symbol | Meaning |
|---|---|
| `B` | `block_tokens`, token positions per block (the engine uses 16) |
| `L` | `n_layers` |
| `Hkv` | `n_kv_heads` |
| `D` | `head_dim` |
| `n` | number of blocks in an envelope |

## Block hash

A sequence's tokens are cut into blocks of `B`. Block `i` covers tokens `[i*B, (i+1)*B)`. Only a **full** block (all `B` positions filled) has a hash; the last block of a sequence is usually partial, and a partial block is never hashed, registered, or deduplicated.

```
hash(block 0) = H(0,              tokens[0 .. B))
hash(block i) = H(hash(block i-1), tokens[i*B .. (i+1)*B))

H(parent, toks) = fnv1a64( le_u64(parent) ‖ le_u32(toks[0]) ‖ ... ‖ le_u32(toks[B-1]) ),
                  with a result of 0 replaced by 1
```

`fnv1a64` starts at `h = 0xCBF29CE484222325` and, for each byte, sets `h = (h XOR byte) * 0x100000001B3 mod 2^64`. Chaining the parent in makes a block's hash name the whole prefix up to and including it, so equal hashes mean equal prefixes (up to a 2^-64 collision), not just equal blocks. The value 0 is reserved for "no hash" (the partial tail block), which is why a computed 0 becomes 1.

A hash names token ids only, not the model: a prefix index and a KV transfer are only ever shared between engines serving the same model.

Worked example, `B = 2`, tokens `[1, 2, 3, 4, 5]`:

| Block | Tokens | Bytes hashed | Hash |
|---|---|---|---|
| 0 | `1 2` | `00 00 00 00 00 00 00 00` `01 00 00 00` `02 00 00 00` | `0xA91AB0C1027B9366` |
| 1 | `3 4` | `66 93 7b 02 c1 b0 1a a9` `03 00 00 00` `04 00 00 00` | `0x03DF829571605C60` |
| 2 | `5` | (partial: not hashed) | `0` |

C: `tl_kv_block_hash(parent, toks, B)`. Test vector for the byte function alone: `fnv1a64("a") = 0xAF63DC4C8601EC8C`.

## Export envelope

```
offset  size  field
0       4     magic "TLKV" (54 4c 4b 56)
4       2     u16 version          1 (format v1) or 2 (format v2, craft.13)
6       2     u16 dtype            a tl_dtype: 1 = TL_F16 (v1), 3 = TL_F8_E4M3 (v2)
8       4     u32 n_blocks         n
12      4     u32 block_tokens     B
16      4     u32 n_layers         L
20      4     u32 n_kv_heads       Hkv
24      4     u32 head_dim         D
28            n block records, each:
                8   u64 block_hash   0 for an unregistered (partial) block
                4   u32 n_tokens     the block's fill, 1..B
                P   payload          P bytes, below
end - 4 4     u32 crc32c           CRC-32C of every byte before it
```

The header is 28 bytes, each block record `12 + P` bytes, so an envelope of `n` blocks is `28 + n * (12 + P) + 4` bytes (`tl_kv_export_bytes`). CRC-32C is the Castagnoli CRC (reflected polynomial `0x82F63B78`, initial value and final XOR `0xFFFFFFFF`); its check value is `crc32c("123456789") = 0xE3069283`.

### Payload, format v1

```
f16  [L][2][Hkv][B][D]        index 1 of the second axis: 0 = K, 1 = V
P = L * 2 * Hkv * B * D * 2 bytes
```

Positions at or past `n_tokens` are written as zero bytes, so an envelope is a pure function of the block's contents (the golden blobs of `parity/kv.wire.v1` are compared byte for byte).

### Payload, format v2 (craft.13)

```
e4m3 [L][2][Hkv][B][D]        the same order, one byte per element
f32  [L][2][Hkv]              the scale of each (layer, K or V, head) slab
P = L * 2 * Hkv * (B * D + 4) bytes
```

Element `x` decodes to `e4m3_to_f32(byte) * scale` for its slab. A writer picks `scale = amax / 448` over the slab's filled positions (1.0 when `amax` is 0), so no value saturates; e4m3 conversion follows [`tinyllm/numerics.h`](../c/include/tinyllm/numerics.h) (M09.4).

### Reader rules

A reader (`tl_kv_import`, the decode side of `PushKv`) rejects the envelope, with nothing allocated, when any of these fail; C returns `TL_EFORMAT` (dimension mismatches `TL_ESHAPE`), gRPC answers `DATA_LOSS` (`FAILED_PRECONDITION` for a version it does not read):

1. The length is at least 32 and equals `28 + n * (12 + P) + 4` for the header's values.
2. The magic is `TLKV` and the CRC matches.
3. The version is one this reader supports, and the dtype is the version's (`1` with v1, `3` with v2). A v1-only reader refuses version 2 cleanly; a v2 reader reads both (drill ops.04).
4. `block_tokens`, `n_layers`, `n_kv_heads`, `head_dim` equal the receiving pool's.
5. Every `n_tokens` is in `1..B`, and a non-zero `block_hash` has `n_tokens == B`.

### Worked example (v1)

`B = 2, L = 1, Hkv = 1, D = 2`, one full block holding tokens `[1, 2]`, with K = `[[1, 2], [3, 4]]` and V = `[[0.5, -1], [0, 0.25]]` (rows are token positions). `P = 1 * 2 * 1 * 2 * 2 * 2 = 16`, so the envelope is `28 + 28 + 4 = 60` bytes:

```
54 4c 4b 56 01 00 01 00 01 00 00 00 02 00 00 00     magic, version 1, dtype 1 (f16), n_blocks 1, block_tokens 2
01 00 00 00 01 00 00 00 02 00 00 00                 n_layers 1, n_kv_heads 1, head_dim 2
66 93 7b 02 c1 b0 1a a9 02 00 00 00                 block_hash 0xA91AB0C1027B9366, n_tokens 2
00 3c 00 40 00 42 00 44                             K: f16 1.0 2.0 3.0 4.0
00 38 00 bc 00 00 00 34                             V: f16 0.5 -1.0 0.0 0.25
1c 54 0b 4f                                         crc32c 0x4F0B541C
```

f16 `1.0` is `0x3C00` (sign 0, exponent 15 = bias, mantissa 0), stored little-endian as `00 3c`.

## Transfer (tl.kv.v1)

Each `KvChunk.payload` is a whole envelope with `n_blocks = 1`, at most 4 MiB, and `KvChunk.crc32c` is the CRC-32C of the payload bytes. The prefill side first asks `HasBlocks` for the prompt's full-block hashes; for each block the decode side already holds it sends the chunk with an empty payload (the decode side takes a reference on its cached copy), and for every other block, including the partial tail, the payload. Bytes moved are therefore `(missing full blocks + 1 tail) * envelope size`. A chunk whose `kv_format` the decode side does not read fails the stream with `FAILED_PRECONDITION`.
