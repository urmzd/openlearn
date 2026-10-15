<!-- ol:module lang.10 -->
# Protocol Buffers and gRPC: schema evolution, unary and streaming RPCs, stubs

## Overview

| | |
|---|---|
| **Module** | `lang.10` · practice · Go and Rust · Pass 7 · 7 to 10 h |
| **You build** | `primers/lang.10/go/`: a toy `tl.kv.v1.KvTransferService` server (`kvstore`) over the vendored Go stubs, and `evolve`, which decodes and re-encodes a `KvChunk` · `primers/lang.10/rust/`: the `kvpush` client over the vendored Rust stubs (tonic), pushing blocks with deduplication |
| **Contract** | the service and messages of [`proto/tl/kv/v1/kv.proto`](../../../../course/contracts/proto/tl/kv/v1/kv.proto); the generated code in `contracts/go/gen/tl/kv/v1` and [`contracts/rust/tl-proto`](../../../../course/contracts/rust/tl-proto/src/lib.rs) |
| **Tests** | `course/tests/lang.10/check` builds both halves offline, runs your `go test` and `cargo test`, feeds hand-written wire bytes to `evolve`, and drives your Go server with your Rust client (what each test checks: section 4) |
| **Needs** | reading: `lang.06` Go ([primer](06-go.md)), `lang.04` Rust ([primer](04-rust.md)), `lang.09` async Rust (tonic runs on tokio) ([primer](09-async-rust-and-tokio.md)) |
| **Used by** | no call site (a primer): `L10.6` applies it next (the real `KvTransferService`, in Rust, over the KV pool), then `dur.01` and `dur.04` (the durable engine's gRPC services, in Go) |
| **Milestone** | `MS-P7` |
| **Optional depth** | [Protocol Buffers, Encoding](https://protobuf.dev/programming-guides/encoding/) (free); [Proto3 language guide, updating a message type](https://protobuf.dev/programming-guides/proto3/#updating) (free); [gRPC core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) (free); [gRPC status codes](https://grpc.github.io/grpc/core/md_doc_statuscodes.html) (free); [tonic examples](https://github.com/hyperium/tonic/tree/master/examples) (free) |

## Key Takeaways

- A protobuf message on the wire is a list of (tag, value) pairs; the tag is `field_number << 3 | wire_type`, and a field at its default takes no bytes: `KvChunk{handle_id: "h", block_index: 2}` is `0a 01 68 18 02` (`test_hand_example_wire_bytes`).
- Because every value says its wire type, a reader can skip a field it does not know, and generated code keeps it: new writers and old readers coexist, which is what schema evolution rests on (`test_unknown_fields_survive_a_round_trip`).
- Field numbers are the schema's identity: never change or reuse one; add new fields with new numbers (`test_varints_and_bytes_decode`).
- gRPC calls are unary or streaming; a client stream sends many messages and gets one reply at the end, which is how `PushKv` moves a prompt's blocks (`test_push_then_dedup`).
- Errors are status codes with meanings the contract fixes: `FAILED_PRECONDITION` for a format the reader cannot read, `DATA_LOSS` for a checksum mismatch (`test_crc_mismatch_is_data_loss`).

## How to work this chapter

```bash
ol start lang.10          # records that you started; the exercise lives in primers/lang.10/
ol tests lang.10
ol check lang.10          # first run writes the starter files (stubs), then checks them
cd primers/lang.10/go && go test ./... && go run ./cmd/kvstore --port 50052 &
cd primers/lang.10/rust && cargo run -- push --addr http://127.0.0.1:50052 --handle a --blocks 3
echo 0a01681802 | (cd primers/lang.10/go && go run ./cmd/evolve)
```

The first `ol check lang.10` writes the starter files that are missing: `go.mod`, `go.sum`, and `Cargo.toml` as given, `kvstore_test.go` as given, and every `.go` and `.rs` file with each function body replaced by a stub. Both halves build offline: the Go module depends on the vendored `contracts/go` through a `replace`, the Rust package on `contracts/rust/tl-proto` by path. You never run `protoc` (DESIGN 2.7): the course commits the generated code.

---

## 1. Why now

Pass 7 splits the engine across processes: a prefill worker hands a prompt's KV blocks to a decode worker (`L10.6`), workers report to the gateway's registry (`gw.05`), and Pass 8's durable engine (`dur.01` to `dur.04`) talks to its workers. These are internal calls between services written in Rust and Go, where JSON over HTTP would cost bytes, parsing time, and type safety. The course uses **gRPC** with **Protocol Buffers** for all of them (D5, D6), from contracts that already exist in `contracts/proto/`. Before you implement real services, this primer takes one of them apart: the bytes on the wire, how a schema changes without breaking readers, and the four kinds of call.

## 2. Principles

### 2.1 A schema with numbered fields

```protobuf
message KvChunk {
  string handle_id = 1;
  uint64 block_hash = 2;        // 0 for the partial tail block
  uint32 block_index = 3;
  uint32 n_blocks_total = 4;
  uint32 kv_format = 5;
  bytes payload = 6;
  uint32 crc32c = 7;
}
service KvTransferService {
  rpc HasBlocks(HasBlocksRequest) returns (HasBlocksResponse);   // unary
  rpc PushKv(stream KvChunk) returns (KvAck);                    // client streaming
  rpc Release(ReleaseRequest) returns (ReleaseResponse);         // unary
}
```

Each field has a name (for code), a type, and a **number** (for the wire). In proto3 every field has a default (0, `""`, empty) and a field at its default is not written at all; a reader cannot tell "absent" from "default". `repeated` is a list; `oneof` is "at most one of these"; an `enum`'s first value must be 0 (the default), conventionally `..._UNSPECIFIED`.

### 2.2 The wire format

| Symbol | Meaning | Type |
|---|---|---|
| $f$ | field number | integer, 1 to $2^{29} - 1$ |
| $w$ | wire type: 0 VARINT, 1 I64, 2 LEN, 5 I32 | integer |
| $v$ | a value | bytes |

A message is a sequence of records, each a **tag** then a **value**: the tag is the varint of $(f \ll 3) \mid w$. A **varint** writes an unsigned integer 7 bits at a time, least significant group first, with the high bit of each byte set when more bytes follow: 2 is `02`, 300 = $10\,0101100_2$ is `ac 02` (`0101100` with the continuation bit, then `0000010`). `uint32`, `uint64`, `int32`, `bool`, and enums are VARINT; `fixed64` and `double` are I64; `string`, `bytes`, nested messages, and packed repeated numbers are LEN: a varint length, then that many bytes.

A reader decodes records in any order; for a field it knows, it checks the wire type and stores the value; for a field it does not know, the wire type alone says how many bytes to skip, and generated code keeps those bytes as **unknown fields** and writes them back when it re-encodes. Go's encoder writes known fields in field-number order, then the unknown ones.

### 2.3 Evolving a schema

Readers and writers are upgraded at different times, so every change must keep old and new compatible in both directions:

- **Add** a field with a new number: old readers skip and keep it, new readers see the default in old messages.
- **Never change** a field's number or its wire type, and **never reuse** a number (or a name, for JSON): mark removed ones `reserved 8; reserved "old_name";`.
- **Rename** freely: names are not on the wire.
- **Enums** keep 0 as "unspecified", so an old reader of a new value falls back safely.
- A change readers cannot handle (a new KV byte format) is not a field change: it is a new version (`kv_format = 2`) that readers refuse with `FAILED_PRECONDITION` until they are migrated (`craft.13`).

### 2.4 Generated code

`protoc` (or `buf`) reads `.proto` files and plugins write code: `protoc-gen-go` and `protoc-gen-go-grpc` for Go (message structs with getters, a client type, a server interface, `Register...Server`), `prost` and `tonic` for Rust (structs, a `...Client<Channel>`, a server trait). The course runs them once (`course/oracle/contracts/gen-proto.sh`) and commits the output in `contracts/`, so your build needs neither tool. In Go, a server **embeds** `kvv1.UnimplementedKvTransferServiceServer`: methods you do not write answer `UNIMPLEMENTED`, and adding an RPC to the proto never breaks your build.

### 2.5 gRPC

gRPC carries protobuf messages over HTTP/2: a call is a request on the path `/tl.kv.v1.KvTransferService/PushKv`, each message framed as a 1-byte flag, a 4-byte length, and the encoded message, and the result is a **status** in the trailers. Four kinds of call:

| Kind | Request | Response | In the course |
|---|---|---|---|
| unary | 1 message | 1 message | `HasBlocks`, `Release`, `EngineControl.Info` |
| server streaming | 1 | many | `WorkflowService.GetHistory` (dur.02) |
| client streaming | many | 1, after the client closes its side | `PushKv`: one chunk per block |
| bidirectional | many | many | none yet |

A status code says what kind of failure happened, so a caller can decide what to do: `INVALID_ARGUMENT` (fix the request), `NOT_FOUND`, `FAILED_PRECONDITION` (the system is not in a state to do this; do not retry blindly), `RESOURCE_EXHAUSTED` (back off), `DATA_LOSS` (corruption), `UNAVAILABLE` (retry later). Every message is capped at 4 MiB in the course (DESIGN 2.7). A Go server stops with `GracefulStop` (finish in-flight calls) on SIGTERM.

### 2.6 Checksums

Each `KvChunk` payload carries its **CRC-32C** (Castagnoli polynomial, reflected `0x82F63B78`): a 32-bit check computed by both sides; any flipped bit changes it, and the receiver answers `DATA_LOSS`. Bit by bit: start from $c = \texttt{0xFFFFFFFF}$; for each byte, $c \mathrel{\oplus}= b$, then eight times $c = (c \gg 1) \oplus (\texttt{0x82F63B78}$ if the low bit was 1$)$; finally $\lnot c$. The check value is $\mathrm{crc32c}(\texttt{"123456789"}) = \texttt{0xE3069283}$. Go has it in `hash/crc32` (`crc32.MakeTable(crc32.Castagnoli)`).

## 3. Worked example by hand

**Encoding** `KvChunk{handle_id: "h", block_index: 2}` (`test_hand_example_wire_bytes`):

| Field | $f$ | $w$ | Tag $(f \ll 3) \mid w$ | Value | Bytes |
|---|---|---|---|---|---|
| `handle_id` | 1 | 2 (LEN) | 10 = `0a` | length 1, then `h` = `68` | `0a 01 68` |
| `block_index` | 3 | 0 (VARINT) | 24 = `18` | 2 | `18 02` |

Every other field is at its default: no bytes. The message is `0a 01 68 18 02`, five bytes; the same in JSON, `{"handle_id":"h","block_index":2}`, is 33.

**An unknown field.** A newer writer adds field 99 as a varint with value 7: tag $(99 \ll 3) \mid 0 = 792 = 110\,0011000_2$, varint `98 06`, value `07`. Appended, the message is `0a 01 68 18 02 98 06 07`. Your `evolve`, built from today's schema, decodes `h` and 2, keeps `98 06 07`, and writes the same eight bytes back (`test_unknown_fields_survive_a_round_trip`).

**A push with deduplication** (`test_push_then_dedup`). The client has three blocks with hashes $h_0, h_1, h_2$. `HasBlocks([h0, h1, h2], kv_format 1)` answers `[false, false, false]`; `PushKv` sends three chunks with payloads and CRCs; the ack is `received 3, deduped 0`. Pushing five blocks of the same prompt next, `HasBlocks` answers `[true, true, true, false, false]`: three chunks go with empty payloads (the receiver uses its copy) and two with payloads: `received 2, deduped 3`. `L10.6` saves exactly this bandwidth when a prompt shares its prefix with an earlier one.

## 4. The artifact and its check

```text
primers/lang.10/
  go/   go.mod (module lang10; replace openlearn.urmzd.com/tl/contracts => ../../../contracts/go), go.sum
        kvstore/kvstore.go       type Store; New(); HasBlocks; PushKv; Release; Handles()
        kvstore/kvstore_test.go  given
        cmd/kvstore/main.go      kvstore --port <n>: prints `listening on 127.0.0.1:<port>`; SIGTERM: GracefulStop, exit 0
        cmd/evolve/main.go       hex KvChunk on stdin -> one JSON line of its fields and "reencoded" hex
  rust/ Cargo.toml (tl-proto by path, tonic, tokio, tokio-stream)
        src/lib.rs               crc32c, block_hash, payload, push, release, code_name
        src/main.rs              kvpush push|release --addr <url> --handle <id> [--blocks n] [--seed s] [--kv-format f] [--corrupt]
```

The toy store's rules, from `kv.proto`: `HasBlocks` and every chunk need `kv_format` 1, else `FAILED_PRECONDITION`; chunks arrive in `block_index` order with one `handle_id` and `n_blocks_total`, else `INVALID_ARGUMENT`; a payload whose CRC-32C differs from `crc32c` is `DATA_LOSS`; an empty payload names a block the store must already hold, else `NOT_FOUND`; the ack counts payloads received and empty chunks deduplicated; nothing of a failed stream is kept; `Release` of any handle, known or not, succeeds. `kvpush` prints `{"present":[...],"received":n,"deduped":n}`, `{"released":true}`, or `{"error":"<CODE>","message":"..."}` with exit 1.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_go_builds_and_your_go_tests_pass` | conformance | the Go module builds offline over the vendored stubs; your `go test` passes | dur.01 and dur.04 build the same way |
| `test_rust_builds_and_your_cargo_tests_pass` | conformance | the Rust package builds offline over tl-proto; your `cargo test` passes | L10.6 builds the same way |
| `test_hand_example_wire_bytes` | unit | section 3: `0a 01 68 18 02`, defaults read as 0 | the wire format by hand |
| `test_unknown_fields_survive_a_round_trip` | unit | fields 99 (varint) and 8 (bytes) kept and re-encoded | forward compatibility |
| `test_varints_and_bytes_decode` | unit | 300 as `ac 02`, a bytes field, fields out of order, re-encoded in field order | decoding is order-independent |
| `test_push_then_dedup` | conformance | HasBlocks then PushKv: 3 received; then 3 deduplicated and 2 received | L10.6's dedup |
| `test_crc_mismatch_is_data_loss` | fault | a flipped bit is `DATA_LOSS` | corruption is detected, not served |
| `test_kv_format_mismatch_is_failed_precondition` | fault | `kv_format` 2 is refused before any block | the craft.13 migration |
| `test_release_is_idempotent` | unit | Release twice and of an unknown handle succeeds | cleanup paths may repeat |
| `test_sigterm_stops_gracefully` | fault | SIGTERM: exit 0 | servers in Kubernetes |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Reusing a removed field's number for a new field | old messages decode into the new field with garbage meaning | section 2.3: mark removed numbers reserved |
| Treating "absent" and "default" as different in proto3 | logic that waits for a field that is never sent because it is 0 | `test_hand_example_wire_bytes` |
| Dropping unknown fields when re-encoding (hand-rolled codecs) | a proxy silently strips new fields | `test_unknown_fields_survive_a_round_trip` |
| Answering a format mismatch with `INTERNAL` or `INVALID_ARGUMENT` | callers retry what can never succeed | `test_kv_format_mismatch_is_failed_precondition` |
| Computing the CRC after changing the payload, or not at all | corrupted KV served as valid | `test_crc_mismatch_is_data_loss` |
| Replying before the client closes its stream | the client's later chunks are lost | `test_push_then_dedup` |
| Keeping blocks of a failed stream | a half-transferred handle leaks memory | your own Go tests of the PushKv error paths |
| Network access in a build | the check fails offline | `test_go_builds_and_your_go_tests_pass` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | Go: interfaces, embedding, goroutines (gRPC serves each call on one) |
| Back | `lang.09` | tokio: tonic's client and server are async |
| Forward | `L10.6` | the real `KvTransferService` in Rust: HasBlocks against the pool's prefix index, PushKv into `tl_kv_import`, Release |
| Forward | `dur.01`, `dur.04` | the durable engine's services and worker protocol, in Go |
| Forward | `gw.05` | the gateway's registry (`tl.control.v1`) and `Release` on aborts |
| Forward | `craft.13` | KV format 2: a version readers refuse until migrated |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| committed generated code | [buf](https://buf.build/docs/) with breaking-change checks | CI refuses a `.proto` change that breaks wire compatibility | `buf breaking`, `course/oracle/contracts/gen-proto.sh` |
| status codes | rich error details (`google.rpc.Status`) | structured error payloads | the gRPC error model |
| one client stream per handle | flow control and windowing in HTTP/2 | backpressure across the network | the HTTP/2 spec, section 5.2 |
| a toy store | Mooncake, NIXL | KV transfer over RDMA between machines | [Mooncake](https://arxiv.org/abs/2407.00079) |
