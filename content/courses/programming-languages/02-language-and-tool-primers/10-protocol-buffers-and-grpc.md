# Protocol Buffers and gRPC: schema evolution, unary and streaming RPCs, stubs

Once the engine is split across processes, its parts have to talk. A prefill worker hands a prompt's KV blocks to a decode worker, workers report to the gateway's registry, and the durable engine talks to its workers. These are internal calls between services written in Rust and Go, where JSON over HTTP would cost bytes, parsing time, and type safety. These books use **gRPC** with **Protocol Buffers** for all of them, from contracts that already exist in `course/contracts/proto/`.

Before you meet the real services, this section takes one of them apart: the bytes on the wire, how a schema changes without breaking readers, and the four kinds of call. You build both ends of a toy `tl.kv.v1.KvTransferService`. In Go, a server ([`kvstore.go`](../../../../course/solutions/primers/lang.10/go/kvstore/kvstore.go): `Store`, `HasBlocks`, `PushKv`, `Release`; the binary [`cmd/kvstore`](../../../../course/solutions/primers/lang.10/go/cmd/kvstore/main.go)) over the vendored Go stubs, plus [`cmd/evolve`](../../../../course/solutions/primers/lang.10/go/cmd/evolve/main.go) (`Evolve`), which decodes and re-encodes a `KvChunk`. In Rust, the `kvpush` client ([`lib.rs`](../../../../course/solutions/primers/lang.10/rust/src/lib.rs): `crc32c`, `block_hash`, `push`, `release`) over the vendored tonic stubs, pushing blocks with deduplication. The real service in [Disaggregated prefill/decode, KV transfer, heartbeat client](../../build-an-llm/p10-serving/06-disaggregated-prefill-decode.md) in *Build Your Own LLM System* is this one grown up.

This section builds on [Go: packages, interfaces, goroutines, channels, context, net/http, testing](06-go.md), [Rust: ownership, traits, Result, cargo workspaces, std TCP](04-rust.md), and [Async Rust and tokio](09-async-rust-and-tokio.md), since tonic runs on tokio. The contract is the service and messages of [`proto/tl/kv/v1/kv.proto`](../../../../course/contracts/proto/tl/kv/v1/kv.proto), with generated code in [`contracts/go/gen/tl/kv/v1`](../../../../course/contracts/go/gen/tl/kv/v1/kv.pb.go) and [`contracts/rust/tl-proto`](../../../../course/contracts/rust/tl-proto/src/lib.rs). The solution runs like this:

```bash
cd primers/lang.10/go && go test ./... && go run ./cmd/kvstore --port 50052 &
cd primers/lang.10/rust && cargo run -- push --addr http://127.0.0.1:50052 --handle a --blocks 3
echo 0a01681802 | (cd primers/lang.10/go && go run ./cmd/evolve)
```

Both halves build offline: the Go module depends on the vendored `contracts/go` through a `replace`, the Rust package on `contracts/rust/tl-proto` by path. Neither runs `protoc`, because the generated code is committed.

> **Learning objectives**
> - Encode a protobuf message to bytes by hand: tags, varints, length-delimited values, and omitted defaults.
> - Explain why unknown fields survive a decode and re-encode, and state the rules for evolving a schema safely.
> - Distinguish the four kinds of gRPC call and implement a client-streaming RPC on both sides.
> - Choose the status code that leads the caller to the right action, and validate a stream before committing any of it.
> - Compute and check a CRC-32C over a payload.

## A schema with numbered fields

Start with the schema, because everything on the wire comes from it. Here is the message and service you will implement:

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

Each field has a name (for code), a type, and a **number** (for the wire). In proto3 every field has a default (0, `""`, empty), and a field at its default is not written at all, so a reader cannot tell "absent" from "default". `repeated` is a list; `oneof` is "at most one of these"; an `enum`'s first value must be 0 (the default), conventionally `..._UNSPECIFIED`.

## The wire format

The encoding uses three quantities:

| Symbol | Meaning | Type |
|---|---|---|
| $f$ | field number | integer, 1 to $2^{29} - 1$ |
| $w$ | wire type: 0 VARINT, 1 I64, 2 LEN, 5 I32 | integer |
| $v$ | a value | bytes |

A message is a sequence of records, each a **tag** then a **value**, where the tag is the varint of $(f \ll 3) \mid w$. A **varint** writes an unsigned integer 7 bits at a time, least significant group first, with the high bit of each byte set when more bytes follow. So 2 is `02`, and 300 = $10\,0101100_2$ is `ac 02` (`0101100` with the continuation bit, then `0000010`). `uint32`, `uint64`, `int32`, `bool`, and enums are VARINT; `fixed64` and `double` are I64; `string`, `bytes`, nested messages, and packed repeated numbers are LEN: a varint length, then that many bytes.

A reader decodes records in any order. For a field it knows, it checks the wire type and stores the value. For a field it does not know, the wire type alone says how many bytes to skip, and generated code keeps those bytes as **unknown fields** and writes them back when it re-encodes. Go's encoder writes known fields in field-number order, then the unknown ones. Because every value says its wire type, new writers and old readers coexist, and that is what schema evolution rests on.

## Evolving a schema

Readers and writers are upgraded at different times, so every change must keep old and new compatible in both directions. The rules are few and strict. **Add** a field with a new number: old readers skip and keep it, new readers see the default in old messages. **Never change** a field's number or its wire type, and **never reuse** a number (or a name, for JSON); mark removed ones `reserved 8; reserved "old_name";`. **Rename** freely, since names are not on the wire. **Enums** keep 0 as "unspecified", so an old reader of a new value falls back safely. In short, field numbers are the schema's identity.

Some changes are not field changes at all. A change readers cannot handle, such as a new KV byte format, becomes a new version (`kv_format = 2`) that readers refuse with `FAILED_PRECONDITION` until they are migrated, as [Interface migration: KV format v1 to v2](../../software-craftsmanship/07-maintenance/01-kv-format-v1-to-v2.md) in *Software Craftsmanship* does.

## Generated code

You do not write the encoder. `protoc` (or `buf`) reads `.proto` files, and plugins write code: `protoc-gen-go` and `protoc-gen-go-grpc` for Go (message structs with getters, a client type, a server interface, `Register...Server`), and `prost` and `tonic` for Rust (structs, a `...Client<Channel>`, a server trait). The repository runs them once and commits the output in `contracts/`, so a build needs neither tool. In Go, a server **embeds** `kvv1.UnimplementedKvTransferServiceServer`: methods it does not define answer `UNIMPLEMENTED`, and adding an RPC to the proto never breaks the build.

## gRPC

Messages need a transport. gRPC carries protobuf messages over HTTP/2: a call is a request on the path `/tl.kv.v1.KvTransferService/PushKv`, each message framed as a 1-byte flag, a 4-byte length, and the encoded message, and the result is a **status** in the trailers. There are four kinds of call:

| Kind | Request | Response | In these books |
|---|---|---|---|
| unary | 1 message | 1 message | `HasBlocks`, `Release`, `EngineControl.Info` |
| server streaming | 1 | many | `WorkflowService.GetHistory` ([Workflow service](../../ai-platform-engineering/05-durable-orchestration-and-workers/02-workflow-service.md)) |
| client streaming | many | 1, after the client closes its side | `PushKv`: one chunk per block |
| bidirectional | many | many | none yet |

A client stream sends many messages and gets one reply at the end, which is how `PushKv` moves a prompt's blocks.

Errors are status codes, and a code says what kind of failure happened so a caller can decide what to do: `INVALID_ARGUMENT` (fix the request), `NOT_FOUND`, `FAILED_PRECONDITION` (the system is not in a state to do this; do not retry blindly), `RESOURCE_EXHAUSTED` (back off), `DATA_LOSS` (corruption), `UNAVAILABLE` (retry later). Every message is capped at 4 MiB in these books. A Go server stops with `GracefulStop` (finish in-flight calls) on SIGTERM.

## Checksums

The last ingredient protects the payload. Each `KvChunk` payload carries its **CRC-32C** (Castagnoli polynomial, reflected `0x82F63B78`): a 32-bit check computed by both sides. Any flipped bit changes it, and the receiver answers `DATA_LOSS`. Bit by bit, start from $c = \texttt{0xFFFFFFFF}$; for each byte, $c \mathrel{\oplus}= b$, then eight times $c = (c \gg 1) \oplus (\texttt{0x82F63B78}$ if the low bit was 1$)$; finally $\lnot c$. The check value is $\mathrm{crc32c}(\texttt{"123456789"}) = \texttt{0xE3069283}$. Go has it in `hash/crc32` (`crc32.MakeTable(crc32.Castagnoli)`).

## Working it by hand

### Encoding a message

Encode `KvChunk{handle_id: "h", block_index: 2}`, one record per non-default field:

| Field | $f$ | $w$ | Tag $(f \ll 3) \mid w$ | Value | Bytes |
|---|---|---|---|---|---|
| `handle_id` | 1 | 2 (LEN) | 10 = `0a` | length 1, then `h` = `68` | `0a 01 68` |
| `block_index` | 3 | 0 (VARINT) | 24 = `18` | 2 | `18 02` |

Every other field is at its default, so it takes no bytes. The message is `0a 01 68 18 02`, five bytes; the same in JSON, `{"handle_id":"h","block_index":2}`, is 33.

### An unknown field

Now a newer writer adds field 99 as a varint with value 7. The tag is $(99 \ll 3) \mid 0 = 792 = 110\,0011000_2$, varint `98 06`, and the value is `07`. Appended, the message is `0a 01 68 18 02 98 06 07`. `evolve`, built from today's schema, decodes `h` and 2, keeps `98 06 07`, and writes the same eight bytes back.

### A push with deduplication

Finally, a push. The client has three blocks with hashes $h_0, h_1, h_2$. `HasBlocks([h0, h1, h2], kv_format 1)` answers `[false, false, false]`; `PushKv` sends three chunks with payloads and CRCs; the ack is `received 3, deduped 0`. Pushing five blocks of the same prompt next, `HasBlocks` answers `[true, true, true, false, false]`: three chunks go with empty payloads (the receiver uses its copy) and two with payloads, giving `received 2, deduped 3`. The real KV transfer saves exactly this bandwidth when a prompt shares its prefix with an earlier one.

## Designing the layout and the store's rules

The exercise is laid out like this:

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

The toy store's rules come from `kv.proto`. `HasBlocks` and every chunk need `kv_format` 1, else `FAILED_PRECONDITION`. Chunks arrive in `block_index` order with one `handle_id` and `n_blocks_total`, else `INVALID_ARGUMENT`. A payload whose CRC-32C differs from `crc32c` is `DATA_LOSS`. An empty payload names a block the store must already hold, else `NOT_FOUND`. The ack counts payloads received and empty chunks deduplicated. Nothing of a failed stream is kept. `Release` of any handle, known or not, succeeds. On the client side, `kvpush` prints `{"present":[...],"received":n,"deduped":n}`, `{"released":true}`, or `{"error":"<CODE>","message":"..."}` with exit 1.

## Building it step by step

### Reasoning from first principles

Try to predict each check in `PushKv` from this chain before reading the code.

1. **Do not hand-roll the codec.** The wire format is simple, but the generated code already decodes, validates wire types, and keeps unknown fields. `evolve` is therefore two library calls, `proto.Unmarshal` then `proto.Marshal`, and the round trip you worked by hand is a property of the generated type, not of code you write.
2. **Refuse what can never succeed, first.** A reader that cannot read `kv_format` 2 must say so before doing any work, with a code that tells the caller not to retry: `FAILED_PRECONDITION`. So the format is the first check in `HasBlocks` and in every chunk of `PushKv`.
3. **A client stream is a loop until `io.EOF`.** The server reads chunks until the client closes its side, and only then sends the one reply (`SendAndClose`). Replying earlier would cut the client's remaining chunks.
4. **Validate each chunk against the stream's own header.** The first chunk fixes `handle_id` and `n_blocks_total`; every chunk must match them and arrive at the next `block_index`. After EOF, the count must equal the total, or the stream was truncated. These are request errors: `INVALID_ARGUMENT`.
5. **Trust a payload only if its checksum matches; trust an empty payload only if the store holds the block.** CRC mismatch is corruption (`DATA_LOSS`); an empty chunk for an unknown hash is `NOT_FOUND`, which tells the client to resend with the payload.
6. **Nothing of a failed stream is kept.** New blocks are staged in a local map and merged into the store only after the whole stream validated. Any early `return` simply drops the staging map.
7. **Shared state lives behind one mutex.** gRPC runs each call on its own goroutine, so the maps are guarded; the lock is held only for map operations, never across `stream.Recv()`.
8. **The client mirrors the server.** It asks `HasBlocks` first (unary), then sends every block in order on one stream, with an empty payload and CRC 0 where the server said `true`, and reads the single ack.

### The store and its state

Embedding the `Unimplemented...` server makes the type satisfy the generated interface even as the proto grows:

```go
const Format = 1                                      // the only KV format this store reads
var castagnoli = crc32.MakeTable(crc32.Castagnoli)    // CRC-32C, as kv.proto specifies

type Store struct {
	kvv1.UnimplementedKvTransferServiceServer          // unknown RPCs answer UNIMPLEMENTED

	mu      sync.Mutex
	blocks  map[uint64][]byte   // block hash -> payload (shared across handles: dedup)
	handles map[string][]uint64 // handle id -> hashes it references
}
```

Blocks are keyed by hash, not by handle, so two prompts that share a prefix share the stored bytes. That is what makes `HasBlocks` answer `true` for another handle's blocks.

### `HasBlocks`: unary, format first

```go
func (s *Store) HasBlocks(ctx context.Context, req *kvv1.HasBlocksRequest) (*kvv1.HasBlocksResponse, error) {
	if req.GetKvFormat() != Format {   // refuse before looking: the caller must migrate, not retry
		return nil, status.Errorf(codes.FailedPrecondition, "kv_format %d: this store reads format %d", req.GetKvFormat(), Format)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	present := make([]bool, len(req.GetBlockHashes()))   // one answer per hash, same order
	for i, h := range req.GetBlockHashes() {
		_, present[i] = s.blocks[h]
	}
	return &kvv1.HasBlocksResponse{Present: present}, nil
}
```

### `PushKv`: the client stream

Every one of the store's rules is one `if` in this loop:

```go
func (s *Store) PushKv(stream kvv1.KvTransferService_PushKvServer) error {
	var ack kvv1.KvAck
	var hashes []uint64
	stored := map[uint64][]byte{}   // staged: merged only if the whole stream is valid
	next := uint32(0)
	total := uint32(0)
	for {
		c, err := stream.Recv()
		if errors.Is(err, io.EOF) {   // the client closed its side: now, and only now, reply
			break
		}
		if err != nil {
			return err                // the client went away or the transport failed
		}
		if c.GetKvFormat() != Format {
			return status.Errorf(codes.FailedPrecondition, "kv_format %d: this store reads format %d", c.GetKvFormat(), Format)
		}
		if next == 0 {                // the first chunk fixes the stream's handle and length
			ack.HandleId, total = c.GetHandleId(), c.GetNBlocksTotal()
		}
		if c.GetHandleId() != ack.HandleId || c.GetNBlocksTotal() != total || c.GetBlockIndex() != next {
			return status.Errorf(codes.InvalidArgument, "chunk %d of %q out of order", c.GetBlockIndex(), c.GetHandleId())
		}
		next++
		if len(c.GetPayload()) == 0 {  // dedup: the sender believes we already hold it
			s.mu.Lock()
			_, ok := s.blocks[c.GetBlockHash()]
			s.mu.Unlock()               // never hold the lock across Recv
			if !ok {
				return status.Errorf(codes.NotFound, "block %016x is not cached here: resend it with its payload", c.GetBlockHash())
			}
			ack.BlocksDeduped++
		} else {
			if got := crc32.Checksum(c.GetPayload(), castagnoli); got != c.GetCrc32C() {
				return status.Errorf(codes.DataLoss, "block %d: crc32c %08x, chunk says %08x", c.GetBlockIndex(), got, c.GetCrc32C())
			}
			ack.BlocksReceived++
			if c.GetBlockHash() != 0 {  // hash 0 marks the partial tail block: never shared
				stored[c.GetBlockHash()] = c.GetPayload()
			}
		}
		hashes = append(hashes, c.GetBlockHash())
	}
	if next != total {                // truncated stream: some chunks never came
		return status.Errorf(codes.InvalidArgument, "stream ended after %d of %d chunks", next, total)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for h, p := range stored {        // commit: the only place the store changes
		s.blocks[h] = p
	}
	s.handles[ack.HandleId] = hashes
	return stream.SendAndClose(&ack)  // the one reply of a client stream
}
```

### `Release`: idempotent by construction

`delete` of a missing key is a no-op in Go, so releasing twice, or releasing an unknown handle, succeeds with no special case:

```go
func (s *Store) Release(ctx context.Context, req *kvv1.ReleaseRequest) (*kvv1.ReleaseResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.handles, req.GetHandleId())
	return &kvv1.ReleaseResponse{}, nil
}
```

### The server binary

The binary sets the 4 MiB cap in both directions, installs the signal handler before announcing readiness, and calls `GracefulStop` on SIGTERM:

```go
	srv := grpc.NewServer(grpc.MaxRecvMsgSize(4<<20), grpc.MaxSendMsgSize(4<<20))
	kvv1.RegisterKvTransferServiceServer(srv, kvstore.New())
	// Install the handler before announcing readiness: a SIGTERM that lands
	// between the two would otherwise kill the process outright.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, os.Interrupt)
	fmt.Printf("listening on %s\n", lis.Addr())
	go func() {
		<-sig
		srv.GracefulStop()   // stop accepting, finish in-flight calls; Serve then returns nil
	}()
	if err := srv.Serve(lis); err != nil {
		// ...exit 1
	}
	// main returns: exit 0
```

### `evolve`: schema evolution is the library's job

```go
func Evolve(wire []byte) (Fields, error) {
	var c kvv1.KvChunk
	if err := proto.Unmarshal(wire, &c); err != nil {   // unknown fields kept in c's internal state
		return Fields{}, err
	}
	out, err := proto.Marshal(&c)                        // known fields in number order, then unknown
	if err != nil {
		return Fields{}, err
	}
	return Fields{HandleID: c.GetHandleId(), /* ...every known field via its getter... */
		Reencoded: hex.EncodeToString(out)}, nil
}
```

The getters return the default for an absent field, so `{"handle_id":"h","block_index":2}` prints `block_hash: 0`, `kv_format: 0`, and so on: proto3 cannot tell absent from default, and neither does `evolve`.

### The Rust client

The client computes CRC-32C by the bit-by-bit definition, so it can be checked against `0xE3069283`:

```rust
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;                     // start from 0xFFFFFFFF
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {                      // reflected: shift right, xor the reversed polynomial
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0x82F6_3B78 } else { crc >> 1 };
        }
    }
    !crc                                     // final inversion
}
```

The push asks first, then streams every block, empty where the server already holds it:

```rust
pub async fn push(addr: &str, handle: &str, payloads: &[Vec<u8>], kv_format: u32, corrupt: bool) -> Result<PushReport, tonic::Status> {
    let mut c = client(addr).await?;
    let hashes: Vec<u64> = payloads.iter().map(|p| block_hash(p)).collect();  // FNV-1a, never 0
    let present = c.has_blocks(HasBlocksRequest { block_hashes: hashes.clone(), kv_format }).await?.into_inner().present;
    if present.len() != hashes.len() {
        return Err(tonic::Status::internal("HasBlocks answered for a different number of blocks"));
    }
    let total = payloads.len() as u32;
    let chunks: Vec<KvChunk> = payloads
        .iter()
        .enumerate()
        .map(|(i, p)| {
            // Present: empty payload, CRC 0 (the default, so neither costs a byte on the wire).
            let (mut body, crc) = if present[i] { (Vec::new(), 0) } else { (p.clone(), crc32c(p)) };
            if corrupt && i == 0 && !body.is_empty() {
                body[0] ^= 1;  // after the CRC was computed: the server must answer DATA_LOSS
            }
            KvChunk { handle_id: handle.to_string(), block_hash: hashes[i], block_index: i as u32, n_blocks_total: total, kv_format, payload: body, crc32c: crc }
        })
        .collect();
    // One client stream; the future resolves with the single ack after the stream ends.
    let ack = c.push_kv(tokio_stream::iter(chunks)).await?.into_inner();
    Ok(PushReport { present, received: ack.blocks_received, deduped: ack.blocks_deduped })
}
```

On an error status, `main` prints `{"error":"<CODE>","message":"..."}` using `code_name`, which maps every `tonic::Code` to its canonical upper-case name, and exits 1.

The full solution is in [`kvstore.go`](../../../../course/solutions/primers/lang.10/go/kvstore/kvstore.go), [`cmd/kvstore/main.go`](../../../../course/solutions/primers/lang.10/go/cmd/kvstore/main.go), [`cmd/evolve/main.go`](../../../../course/solutions/primers/lang.10/go/cmd/evolve/main.go), [`rust/src/lib.rs`](../../../../course/solutions/primers/lang.10/rust/src/lib.rs), and [`rust/src/main.rs`](../../../../course/solutions/primers/lang.10/rust/src/main.rs).

### Why this design

`HasBlocks` is $O(k)$ map lookups for $k$ hashes. `PushKv` is $O(\text{bytes})$ for the CRCs plus $O(n)$ map work, and holds the lock only for single lookups and the final commit, so concurrent pushes do not serialize on each other's network reads.

Writing each block into `s.blocks` immediately would be simpler than staging, but a stream that fails at chunk 4 of 5 would leave three orphan blocks referenced by no handle. Staging costs a second map for the life of one call.

Status codes are the API. `INVALID_ARGUMENT` (fix the request), `FAILED_PRECONDITION` (migrate, do not retry), `NOT_FOUND` (resend with the payload), and `DATA_LOSS` (corruption) each lead the caller to a different action. Collapsing them into `INTERNAL` would make every failure look retryable.

Asking first, then pushing, costs one extra round trip and saves every byte of every shared block, which for long shared prefixes is most of the transfer (3 of 5 blocks deduplicated in the hand example).

Two simplifications are accepted: `Release` forgets the handle but leaves its blocks (a real pool reference-counts them), and the toy hash is FNV-1a, not a cryptographic hash.

## What can go wrong

A correct solution passes these checks:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Go builds, Go tests pass | conformance | the Go module builds offline over the vendored stubs; `go test` passes | the durable engine's services build the same way |
| Rust builds, Rust tests pass | conformance | the Rust package builds offline over tl-proto; `cargo test` passes | the real KV transfer builds the same way |
| The hand example's wire bytes | unit | `0a 01 68 18 02`, defaults read as 0 | the wire format by hand |
| Unknown fields survive a round trip | unit | fields 99 (varint) and 8 (bytes) kept and re-encoded | forward compatibility |
| Varints and bytes decode | unit | 300 as `ac 02`, a bytes field, fields out of order, re-encoded in field order | decoding is order-independent |
| Push, then dedup | conformance | HasBlocks then PushKv: 3 received; then 3 deduplicated and 2 received | the real KV transfer's dedup |
| A CRC mismatch is `DATA_LOSS` | fault | a flipped bit is `DATA_LOSS` | corruption is detected, not served |
| A `kv_format` mismatch is `FAILED_PRECONDITION` | fault | `kv_format` 2 is refused before any block | the KV format v2 migration |
| Release is idempotent | unit | Release twice and of an unknown handle succeeds | cleanup paths may repeat |
| SIGTERM stops gracefully | fault | SIGTERM: exit 0 | servers in Kubernetes |

The pitfalls below are the usual ways those cases fail, each with the line of the solution that prevents it:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Reusing a removed field's number for a new field | old messages decode into the new field with garbage meaning | mark removed numbers reserved |
| Treating "absent" and "default" as different in proto3 | logic that waits for a field that is never sent because it is 0 | `evolve` reads every field through its getter and prints defaults as values |
| Dropping unknown fields when re-encoding (hand-rolled codecs) | a proxy silently strips new fields | the generated `proto.Unmarshal`/`proto.Marshal`, which keep them |
| Answering a format mismatch with `INTERNAL` or `INVALID_ARGUMENT` | callers retry what can never succeed | `FAILED_PRECONDITION` in `HasBlocks` and on every chunk, before any other check |
| Computing the CRC after changing the payload, or not at all | corrupted KV served as valid | the server recomputes CRC-32C for every non-empty payload; `DATA_LOSS` on mismatch |
| Replying before the client closes its stream | the client's later chunks are lost | `SendAndClose` only after `Recv` returns `io.EOF` |
| Keeping blocks of a failed stream | a half-transferred handle leaks memory | blocks staged in `stored`, committed only after the whole stream validated |
| Network access in a build | the build fails offline | `replace` to the vendored `contracts/go`; `tl-proto` by path; committed generated code |

> **Warning:** Reusing a field number is the one schema mistake no test of the new code will catch. The new code is self-consistent; the damage appears only when it reads a message an old writer produced. Mark every removed number `reserved`.

## Where this leads

This section relied on [Go](06-go.md) for interfaces, embedding, and goroutines (gRPC serves each call on one), and on [Async Rust and tokio](09-async-rust-and-tokio.md), because tonic's client and server are async.

In *Build Your Own LLM System*, [Disaggregated prefill/decode, KV transfer, heartbeat client](../../build-an-llm/p10-serving/06-disaggregated-prefill-decode.md) implements the real `KvTransferService` in Rust: `HasBlocks` against the pool's prefix index, `PushKv` into `tl_kv_import`, and `Release`. In *AI Platform Engineering*, [Append-only segmented event log](../../ai-platform-engineering/05-durable-orchestration-and-workers/01-event-log.md) and [Task gRPC protocol, Go worker SDK, worker pool](../../ai-platform-engineering/05-durable-orchestration-and-workers/04-task-protocol-and-workers.md) build the durable engine's services and worker protocol in Go, and [Worker registry, routing, cascades, failover](../../ai-platform-engineering/12-gateway/05-routing.md) uses the gateway's registry (`tl.control.v1`) and calls `Release` on aborts. [Interface migration: KV format v1 to v2](../../software-craftsmanship/07-maintenance/01-kv-format-v1-to-v2.md) in *Software Craftsmanship* introduces KV format 2, a version readers refuse until migrated.

The toy pieces have production equivalents:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| committed generated code | [buf](https://buf.build/docs/) with breaking-change checks | CI refuses a `.proto` change that breaks wire compatibility | `buf breaking` |
| status codes | rich error details (`google.rpc.Status`) | structured error payloads | the gRPC error model |
| one client stream per handle | flow control and windowing in HTTP/2 | backpressure across the network | the HTTP/2 spec, section 5.2 |
| a toy store | Mooncake, NIXL | KV transfer over RDMA between machines | [Mooncake](https://arxiv.org/abs/2407.00079) |

### Further reading

- [Protocol Buffers, Encoding](https://protobuf.dev/programming-guides/encoding/) (free)
- [Proto3 language guide, updating a message type](https://protobuf.dev/programming-guides/proto3/#updating) (free)
- [gRPC core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) (free)
- [gRPC status codes](https://grpc.github.io/grpc/core/md_doc_statuscodes.html) (free)
- [tonic examples](https://github.com/hyperium/tonic/tree/master/examples) (free)

## Key terms

- **field number**: the integer that identifies a field on the wire; it must never change or be reused.
- **tag**: the varint $(f \ll 3) \mid w$ that starts each record, combining field number and wire type.
- **varint**: an unsigned integer written 7 bits per byte, least significant group first, with a continuation bit.
- **wire type**: the 3-bit code (VARINT, I64, LEN, I32) that tells a reader how many bytes a value occupies.
- **unknown fields**: records a reader does not recognize, kept by generated code and written back on re-encode.
- **client streaming RPC**: a call in which the client sends many messages and receives one reply after closing its side.
- **status code**: the gRPC result in the trailers, telling the caller what kind of failure happened.
- **CRC-32C**: a 32-bit Castagnoli checksum over a payload, used to detect corruption.

## Summary

- A protobuf message on the wire is a list of (tag, value) pairs; the tag is `field_number << 3 | wire_type`, and a field at its default takes no bytes: `KvChunk{handle_id: "h", block_index: 2}` is `0a 01 68 18 02`.
- Because every value says its wire type, a reader can skip a field it does not know, and generated code keeps it: new writers and old readers coexist, which is what schema evolution rests on.
- Field numbers are the schema's identity: never change or reuse one; add new fields with new numbers.
- gRPC calls are unary or streaming; a client stream sends many messages and gets one reply at the end, which is how `PushKv` moves a prompt's blocks, validated in full before anything is committed.
- Errors are status codes with meanings the contract fixes: `FAILED_PRECONDITION` for a format the reader cannot read, `DATA_LOSS` for a checksum mismatch.

## Review questions

1. Why does `evolve` keep field 99 when today's schema has no field 99, and what would a hand-written decoder that only reads known fields do?
   <details><summary>Answer</summary>
   The record's tag carries its wire type (0, varint), which tells the generated decoder exactly how many bytes to skip; the generated message stores those bytes as unknown fields and `proto.Marshal` writes them back after the known ones. A hand-written decoder that skipped them without storing them would re-encode only the known fields, silently stripping the newer writer's data.
   </details>
2. Why is a `kv_format` mismatch `FAILED_PRECONDITION` and not `INVALID_ARGUMENT`?
   <details><summary>Answer</summary>
   The request is well formed; the receiver is in a state where it cannot do this (it reads only format 1). `FAILED_PRECONDITION` tells the caller that retrying the same thing will not help until the system changes (a migration, as in the KV format v2 migration), whereas `INVALID_ARGUMENT` says "fix your request", which the caller cannot do.
   </details>
3. If a client sends chunks 0, 1, 2 of a 5-block handle and closes its stream, what does the store keep and what does the client see?
   <details><summary>Answer</summary>
   The loop sees `io.EOF` after three chunks, `next` (3) differs from `total` (5), and `PushKv` returns `INVALID_ARGUMENT` "stream ended after 3 of 5 chunks". The staged blocks are in the local map `stored`, which is dropped: nothing is kept, and `handles` is unchanged.
   </details>
4. Why are deduplicated chunks sent with `crc32c` 0 and an empty payload, and why must the server check that it holds the block?
   <details><summary>Answer</summary>
   Both are proto3 defaults, so they cost no bytes on the wire; the chunk still carries `block_hash` and `block_index` so the stream stays in order. The server checks because `HasBlocks` and `PushKv` are separate calls: another client's `Release` or a restart could intervene, and accepting an empty chunk for a block it does not have would silently lose data. `NOT_FOUND` tells the client to resend that block with its payload.
   </details>
5. Why does `cmd/kvstore` call `signal.Notify` before printing `listening on ...`?
   <details><summary>Answer</summary>
   The parent (a test harness, Kubernetes) may send SIGTERM as soon as it sees the readiness line. If the handler were installed after the print, a SIGTERM in between would get Go's default behaviour and kill the process without `GracefulStop`, with a non-zero exit.
   </details>
6. Encode the integer 300 as a varint, and the tag for field 99 with wire type 0.
   <details><summary>Answer</summary>
   300 is $10\,0101100_2$: the low 7 bits `0101100` with the continuation bit set give `ac`, then `0000010` gives `02`, so `ac 02`. The tag is $(99 \ll 3) \mid 0 = 792 = 110\,0011000_2$, which encodes as `98 06`.
   </details>
