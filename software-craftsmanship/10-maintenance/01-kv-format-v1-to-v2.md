<!-- ol:module craft.13 -->
# Interface migration: KV format v1 to v2

## Overview

| | |
|---|---|
| **Module** | `craft.13` · build · rust · Pass 11 · 1 to 2 h |
| **You build** | rust/crates/tl-kv-format/Cargo.toml, rust/crates/tl-kv-format/src/lib.rs, rust/crates/tl-kv-format/src/main.rs |
| **Tests** | `course/tests/rust/craft_13.rs` (named below, with why each exists) |
| **Needs** | craft.12 |
| **Used by** | ops.04 |
| **Milestone** | `MS-P11` |

## Key Takeaways

- The v2 envelope keeps the 28-byte header and block records; dtype 3 means fp8 e4m3.
- Each (layer, K or V, head) slab carries one f32 scale, max absolute value divided by 448, or 1.0 for all-zero data.
- The offline converter validates v1 CRC and dimensions before atomically writing v2 output.

## How to work this chapter

```bash
ol start craft.13
ol tests craft.13
ol check craft.13
```

---

## 1. Why now

KV v2 changes the representation consumed by both the C pool and Rust transfer path. A coordinated migration is necessary because mixed versions can otherwise corrupt blocks or reject valid transfers.

## 2. Principles

Document the wire envelope and negotiation rules first. Readers become compatible before writers emit v2. Keep v1 readable through the rollback window; reject unsupported versions before allocating or mutating state.

## 3. Worked example

Ship a reader that accepts v1 and v2, then deploy writers that emit v2 only after every reader advertises support. Test mixed pairs, fp8 scales, content hash, malformed lengths, and the downgrade path. Track format version by peer.

## 4. Interface and tests


For `B=2, L=1, Hkv=1, D=2`, a v1 block payload is `1 × 2 × 1 × 2 × 2 × 2 = 16` bytes. The v2 block payload is `1 × 2 × 1 × (2 × 2 + 4) = 16` bytes: eight fp8 values plus two f32 scales. The v1 and v2 payload sizes happen to match for this tiny shape; the interpretation differs.

Worked example: the `hand_example_golden_migration_matches_fixture` test uses the small v1 payload above and compares the converted file byte for byte before checking the quantized block values.

The course tests exercise these cases:

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_golden_migration_matches_fixture` | unit | Converts the section 3 fixture and compares the exact golden bytes. | Establishes the migration result by hand before broader cases. |
| `both_formats_roundtrip_with_version_specific_dtype` | unit | Reads and writes v1 and v2 with the correct dtype for each. | Lets readers roll forward while retaining rollback support. |
| `rejects_corruption_before_returning_a_partial_envelope` | boundary | Rejects bad CRC before exposing payload data. | Prevents corrupted KV blocks from reaching decode. |
| `rejects_shape_mismatch_and_unknown_version` | boundary | Rejects incompatible dimensions and unsupported format versions. | Keeps peers from misreading valid-looking bytes. |
| `partial_tail_is_zero_filled_and_never_claims_a_full_hash` | boundary | Makes padding deterministic and avoids a false full-block hash. | Prevents dedupe from treating incomplete blocks as complete. |
| `zero_slab_uses_unit_scale` | boundary | Gives an all-zero slab finite unit scale. | Avoids invalid quantization metadata. |
| `v2_upgrader_refuses_a_v2_input_instead_of_double_converting` | boundary | Refuses a second conversion. | Prevents silent repeated quantization. |
| `process_converter_reads_and_atomically_writes_file_payloads` | integration | Exercises the subprocess converter and atomic output path. | Keeps Python and Rust separated by the documented file boundary. |

Run `ol tests craft.13` before changing the implementation, then `ol check craft.13` after the work.

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Emitting v2 before all readers advertise support | A mixed-version peer rejects or misreads a block. | `both_formats_roundtrip_with_version_specific_dtype` |
| 2 | Returning data before validating the envelope CRC | Corrupt blocks can reach the scheduler as plausible KV data. | `rejects_corruption_before_returning_a_partial_envelope` |
| 3 | Re-converting an already quantized v2 block | A second conversion adds loss without an explicit error. | `v2_upgrader_refuses_a_v2_input_instead_of_double_converting` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.12` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.02` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `course/contracts/formats/kv-block.md` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `L10.6` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Forward | `ops.04` | This declared call site consumes the artifact; verify its expectations before finalizing. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
