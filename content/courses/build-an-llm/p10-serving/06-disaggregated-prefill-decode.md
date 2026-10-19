# Disaggregated prefill/decode, KV transfer, heartbeat client

## Overview

| | |
|---|---|
| **Module** | `L10.6` · build · Rust · Pass 7 · 5 to 7 h reading |
| **Solution** | [`tl-engine/src/kv_transfer.rs`](../../../../course/solutions/rust/crates/tl-engine/src/kv_transfer.rs): `block_hash`, `prompt_hashes`, `crc32c`, `write_envelope` and `read_envelope`, `resume_rng`, the decode side of `tl.kv.v1` (`KvReceiver`, `KvService`), the prefill side's deduplicated push (`build_chunks`, `transfer`) · [`tl-engine/src/heartbeat.rs`](../../../../course/solutions/rust/crates/tl-engine/src/heartbeat.rs): `worker_status`, `beat_once`, `Heartbeat` · [`tl-serve/src/control.rs`](../../../../course/solutions/rust/crates/tl-serve/src/control.rs): `ControlService` for `tl.engine.v1.EngineControl` (Prefill, Info, Cancel, Drain) over a `PrefillBackend` · the serve loop's `EngineBackend` and `resume_generate` in [`tl-serve/src/server.rs`](../../../../course/solutions/rust/crates/tl-serve/src/server.rs) |
| **Contract** | gRPC: [`proto/tl/engine/v1/engine.proto`](../../../../course/contracts/proto/tl/engine/v1/engine.proto), [`proto/tl/kv/v1/kv.proto`](../../../../course/contracts/proto/tl/kv/v1/kv.proto), [`proto/tl/control/v1/control.proto`](../../../../course/contracts/proto/tl/control/v1/control.proto), generated code in [`rust/tl-proto`](../../../../course/contracts/rust/tl-proto/src/lib.rs) · bytes: [`formats/kv-block.md`](../../../../course/contracts/formats/kv-block.md) · conformance: `parity/kv.wire.v1` |
| **Needs** | `L10.1` `KvPool` and the sampler's `stream` ([chapter](01-model-runner-and-sampler.md)) · `L10.1` the Rust pool: export, import, register, lookup ([chapter](../p08-inference/08-paged-kv-block-pool.md)) · reading: `L10.2`, `L10.4`, `L10.5` the serve loop this plugs into, `lang.09` async Rust ([primer](../../../tracks/software-craftsmanship/12-language-and-tool-primers/09-async-rust-and-tokio.md)), `lang.10` gRPC ([primer](../../../tracks/software-craftsmanship/12-language-and-tool-primers/10-protocol-buffers-and-grpc.md)) |
| **Used by** | `gw.05` calls Prefill and Release over gRPC and serves the registry the heartbeat reaches; `craft.13` takes `kv_transfer.rs` over for KV format v2; `craft.14` preserves registration and control in the upgraded server. The production deployment runs these engines as 1 prefill + 2 decode |
| **Optional depth** | [DistServe](https://arxiv.org/abs/2401.09670) (free); [Splitwise](https://arxiv.org/abs/2311.18677) (free); [Mooncake](https://arxiv.org/abs/2407.00079) (free); [gRPC core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) (free) |

## Key Takeaways

- A block's hash names its whole prefix (`H(parent, tokens)`), so "D already holds this block" is one table lookup, and only the missing full blocks plus the partial tail cross the network.
- The decode side reads bytes from the network: every reader rule of `kv-block.md` refuses bad input before anything is allocated, with DATA_LOSS for corruption and FAILED_PRECONDITION for a format it does not read.
- Every error path frees what it took: a refused push, a stream cut mid-transfer, an abandoned client, a released handle; neither engine's pool may leak.
- Prefill on one engine and decode on another give the same tokens as one engine doing both, greedy and seeded, because the decode side's generator ends up advanced by exactly the draws the prefill side used.
- The heartbeat never blocks the engine and never stops: each beat is bounded by a timeout, misses are counted, and a drain request is remembered.

## How to read this chapter

Sections 1 to 3 explain why prefill and decode are split, how blocks are named and shipped, and work one envelope and one transfer by hand. Section 4 fixes the interface. Section 5 walks through the solution: hashing, the envelope reader, the decode side's all-or-nothing push, the prefill side's transfer, the RNG hand-off, and the heartbeat. Try to predict each step before reading it.

The solution's `tl-engine` depends on `tl-proto` (the generated code in [`contracts/rust/tl-proto`](../../../../course/contracts/rust/tl-proto/src/lib.rs)), `tokio` (full), `tonic = "0.14"`, and `prost = "0.14"` (all in [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml)), and declares `pub mod kv_transfer; pub mod heartbeat;` in `tl-engine/src/lib.rs` and `pub mod control;` in `tl-serve/src/lib.rs`. The server starts `kv_transfer::serve_kv` on `[engine].kv_listen` for a decode engine, `control::serve_control` on `[engine].grpc_listen` for every engine, and `Heartbeat::spawn` toward `[engine].gateway_registry`.

---

## 1. Why now

After `L10.5` one engine runs both phases of every request: the compute-bound prefill of the prompt and the memory-bound decode of each new token. They compete for the same batch. A long prompt arriving while thirty streams decode makes every one of them stall for its prefill (TPOT spikes), and a burst of decodes makes a new prompt wait (TTFT spikes). Production systems split the two onto separate workers: prefill engines that only build KV caches, decode engines that only extend them. That needs three things the engine does not have yet: a way for the gateway to ask a prefill engine to run one prompt (`EngineControl.Prefill`), a way to move the prompt's KV blocks to the decode engine without moving the ones it already holds (`tl.kv.v1`), and a way for the gateway to know which engines exist and how loaded they are (the heartbeat to `tl.control.v1`). This module writes all three on top of the block pool and the sampler of `L10.1`.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | tokens per block (`block_tokens`; the engine uses 16, the small examples 4) | `u32` |
| $L, H_{kv}, D$ | layers, KV heads, head dimension | `u32` |
| $P = 2 L H_{kv} B D \cdot 2$ | payload bytes of one block in format v1 (K and V, f16) | `usize` |
| $E = 28 + n(12 + P) + 4$ | bytes of an envelope of $n$ blocks | `usize` |
| $h_i$ | hash of full block $i$; $h_{-1} = 0$ | `u64` |
| $k$ | full blocks the decode engine already caches | count |
| $F$, $t$ | full blocks of the prompt; $t = 1$ when a partial tail exists, else 0 | count |

### 2.1 The flow

```text
gateway                 prefill engine P                         decode engine D
   | Prefill(prompt, seed, decode_target=D) |                              |
   |--------------------------------------->| prefill, sample first token  |
   |                                        |-- HasBlocks(h_0..h_{F-1}) -->| lookup each hash (no reference kept)
   |                                        |<-- present[] ----------------|
   |                                        |== PushKv(stream of chunks) =>| import missing blocks, reference present ones,
   |                                        |<-- KvAck --------------------|   hold them under the handle id
   |<-- KvHandle{first_token, hashes, draws}| drop P's own references       |
   |-- POST /v1/chat/completions, X-TL-KV-Handle ------------------------->| claim the handle's blocks, resume
```

The gateway emits `first_token` to the client at once and then streams the rest from D. On abort it calls `Release` on D.

### 2.2 Naming blocks by content

`L10.1` hashes a full block as $h_i = \mathrm{FNV1a64}(\mathrm{le64}(h_{i-1}) \,\Vert\, \mathrm{le32}(x_{iB}) \cdots \mathrm{le32}(x_{iB+B-1}))$, with a result of 0 replaced by 1. Chaining the parent makes $h_i$ name the whole prefix up to block $i$: two prompts that share their first $k$ blocks share $h_0 \ldots h_{k-1}$ and nothing after. The decode engine's prefix index (the Swiss table of `ds.02` inside `L10.1`) answers "do I hold $h_i$?" in O(1). A partial block has no hash: its contents will still change.

### 2.3 What crosses the wire

Each `KvChunk` carries one block as a one-block envelope of `kv-block.md`: the 28-byte header, the record (`u64` hash, `u32` fill, $P$ payload bytes, zeros past the fill), and a CRC-32C over everything before it. The chunk repeats the hash and carries its own CRC-32C of the payload. A block D reported present travels as an empty payload with its hash set: D takes a reference on its cached copy instead of importing. So the payload bytes of one transfer are

$$ \text{bytes} = (F - k + t)\,(28 + 12 + P + 4). $$

### 2.4 The reader's rules, and what each error means

D refuses an envelope, with nothing allocated, when its length is not $E$, its magic is not `TLKV`, its CRC fails, its version or dtype is not one D reads, its dimensions differ from D's pool, a fill is outside $1..B$, or a hashed block is not full. Corruption is `DATA_LOSS` (the gateway may retry elsewhere); a version D does not read is `FAILED_PRECONDITION` (it will never work on this worker: route elsewhere). A full block that P never registered (another block held its hash) travels with hash 0 in its envelope; the chunk's hash names it and D registers it.

### 2.5 Resuming the generator

The request's tokens are drawn from `stream(seed, sample)` (`spec/pcg32.md`). P drew the first token with one `uniform_f64` (0 draws when greedy) and reports `rng_draws_consumed`. D seeds the same stream and advances it by that many uniforms before its first draw, so a seeded disaggregated request emits exactly the tokens a unified engine would, given the same logits.

### 2.6 Freeing on every path

The KV pool is the engine's scarcest resource, and every path that takes a block must give it back:

| Path | Who frees |
|---|---|
| push succeeds | P drops its references (full blocks stay cached on P); D holds them under the handle until the resume claims them |
| D refuses a chunk | D releases every block taken for this push before answering |
| the stream breaks (peer gone, cancel) | D's stream read fails; same as a refusal |
| the push fails on P's side | P drops its references, then answers the error |
| the client aborts after Prefill | the gateway calls `Release(handle)` on D |

### 2.7 The heartbeat

Every 2 s each engine sends `WorkerStatus` (id, three addresses, role, model, KV format, queue depth, running sequences, free and total blocks, draining). The gateway evicts a worker after 3 missed beats. Two rules follow: a beat must be bounded by a timeout (a hung registry must not hang the engine), and failures must not stop the loop (the registry restarting is not the engine's problem). Only a decode engine advertises `kv_address`. An ack with `drain = true` is remembered until the engine has drained.

## 3. Worked example by hand

**One block, by hand** (the example of `kv-block.md`). $B = 2$, one layer, one head, $D = 2$, the full block of tokens `[1, 2]` with K = `[[1, 2], [3, 4]]` and V = `[[0.5, -1], [0, 0.25]]`.

1. Hash: the bytes are `00`x8 (parent 0), `01 00 00 00`, `02 00 00 00`; FNV-1a 64 gives `0xA91AB0C1027B9366`.
2. $P = 2 \cdot 1 \cdot 1 \cdot 2 \cdot 2 \cdot 2 = 16$, so $E = 28 + 28 + 4 = 60$ bytes.
3. f16: 1.0 is `0x3C00` (exponent 15 = the bias, mantissa 0), stored `00 3c`; 0.5 is `0x3800`; -1.0 is `0xBC00`; 0.25 is `0x3400`.
4. The CRC-32C of the first 56 bytes is `0x4F0B541C`, appended little-endian: `1c 54 0b 4f`.

**One transfer, by hand** (small shapes: $B = 4$, $L = 2$, $H_{kv} = 1$, $D = 4$, so $P = 2 \cdot 2 \cdot 1 \cdot 4 \cdot 4 \cdot 2 = 128$ and a one-block envelope is $28 + 12 + 128 + 4 = 172$ bytes). A 9-token prompt has $F = 2$ full blocks and a tail of 1 token ($t = 1$). D already caches block 0 ($k = 1$):

| Block | Hash | Present on D | Chunk |
|---|---|---|---|
| 0 | $h_0$ | yes | empty payload, hash $h_0$: D takes a reference |
| 1 | $h_1$ | no | 172-byte envelope |
| 2 (tail, 1 token) | 0 | never asked | 172-byte envelope, fill 1, zeros after |

Bytes moved: $(2 - 1 + 1) \cdot 172 = 344$. The ack says `blocks_received = 2`, `blocks_deduped = 1`.

**The hand-off, by hand.** Seed 77 at temperature 0.9: P draws $u_1$ for the first token and reports `rng_draws_consumed = 1`; D's generator starts at `stream(77, sample)` advanced by one `uniform_f64`, so its first draw is $u_2$, the draw a unified engine would use for token 2.

## 4. The interface

```rust
// rust/crates/tl-engine/src/kv_transfer.rs
pub use {tl_proto, tokio, tonic};                     // one version of each for the engine and the tests
pub type SharedKv = Arc<Mutex<KvPool>>;
pub fn fnv1a64(data: &[u8]) -> u64;
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64;              // 0 becomes 1
pub fn prompt_hashes(tokens: &[u32], block_tokens: usize) -> Vec<u64>;   // full blocks only
pub fn crc32c(data: &[u8]) -> u32;
pub struct EnvHeader { pub version: u16, pub dtype: u16, pub n_blocks: u32, pub block_tokens: u32, pub n_layers: u32, pub n_kv_heads: u32, pub head_dim: u32 }
pub struct BlockRecord { pub hash: u64, pub n_tokens: u32, pub payload: Vec<u8> }
pub enum EnvError { Format(String), Version(String), Shape(String) }
pub fn write_envelope(h: &EnvHeader, blocks: &[BlockRecord]) -> Vec<u8>;
pub fn read_envelope(buf: &[u8], expect: Option<&KvCfg>) -> Result<(EnvHeader, Vec<BlockRecord>), EnvError>;
pub fn resume_rng(seed: u64, draws: u32) -> Pcg32;
pub struct KvReceiver { /* pool, handle -> block ids */ }
impl KvReceiver {
    pub fn new(pool: SharedKv) -> KvReceiver;
    pub fn has_blocks(&self, hashes: &[u64], kv_format: u32) -> Result<Vec<bool>, Status>;
    pub fn accept_chunk(&self, c: &KvChunk, expect_index: u32, taken: &mut Vec<u32>) -> Result<bool, Status>;
    pub fn accept_all(&self, chunks: impl IntoIterator<Item = Result<KvChunk, Status>>) -> Result<KvAck, Status>;
    pub fn claim(&self, handle_id: &str) -> Option<Vec<u32>>;     // the resumed request takes the blocks
    pub fn release(&self, handle_id: &str);
    pub fn pending(&self) -> usize;
}
pub struct KvService(pub Arc<KvReceiver>);                       // impl KvTransferService (has_blocks, push_kv, release)
pub fn spawn_router(listener: TcpListener, router: tonic::transport::server::Router) -> io::Result<GrpcServer>;
pub fn serve_kv(listener: TcpListener, recv: Arc<KvReceiver>) -> io::Result<GrpcServer>;
#[tonic::async_trait] pub trait KvTransport: Send + Sync {
    async fn has_blocks(&self, target: &str, hashes: &[u64], fmt: u32) -> Result<Vec<bool>, TransferError>;
    async fn push(&self, target: &str, chunks: Vec<KvChunk>) -> Result<KvAck, TransferError>;
}
pub struct GrpcTransport { pub timeout: Duration }              // impl KvTransport
pub fn build_chunks(pool: &SharedKv, handle_id: &str, ids: &[u32], hashes: &[u64], present: &[bool]) -> Result<Vec<KvChunk>, TransferError>;
pub async fn transfer(t: &dyn KvTransport, target: &str, pool: &SharedKv, handle_id: &str, ids: &[u32], hashes: &[u64]) -> Result<TransferStats, TransferError>;

// rust/crates/tl-engine/src/heartbeat.rs
#[tonic::async_trait] pub trait Registry: Send + Sync { async fn heartbeat(&self, s: WorkerStatus) -> Result<HeartbeatAck, String>; }
pub struct GrpcRegistry { pub target: String, pub timeout: Duration }
pub fn worker_status(id: &WorkerIdentity, load: &Load) -> WorkerStatus;
pub async fn beat_once(reg: &dyn Registry, status: WorkerStatus, timeout: Duration, st: &BeatState) -> bool;
impl Heartbeat { pub fn spawn(reg, id, load: Arc<dyn Fn() -> Load + Send + Sync>, interval, timeout) -> io::Result<Heartbeat>; pub fn stop(self); }

// rust/crates/tl-serve/src/control.rs
#[tonic::async_trait] pub trait PrefillBackend: Send + Sync + 'static {
    async fn prefill(&self, job: PrefillJob) -> Result<PrefillOutcome, Status>;   // first token, block ids, hashes, draws
    fn pool(&self) -> SharedKv; fn info(&self) -> InfoResponse;
    async fn cancel(&self, request_id: &str) -> bool; async fn drain(&self, deadline_ms: i64) -> i32;
}
pub struct ControlService<B: PrefillBackend> { pub backend: Arc<B>, pub transport: Arc<dyn KvTransport>, pub roles: Vec<String> }
pub fn sampling_from_proto(p: Option<&pb::SamplingParams>) -> Result<(SamplingParams, u64, usize), Status>;
pub fn serve_control<B: PrefillBackend>(listener: TcpListener, svc: ControlService<B>) -> io::Result<GrpcServer>;
```

The serve loop implements `PrefillBackend` (`EngineBackend` in `server.rs`: prefill the prompt through the block manager, sample the first token, hand back the block table) and, on a decode engine, resumes a request from `claim(handle)` with its generator advanced as `resume_rng(seed, draws)` would (see 5.7).

### What a correct solution must handle

The cases run a fake model that writes a fixed f16 pattern per position into real `L10.1` pools and whose logits hash every cached KV value, so any lost or stale block changes the next token. Prefill and decode engines run the real servers on `127.0.0.1:0`.

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand envelope | section 3's 60 bytes, the two chained hashes, FNV-1a and CRC-32C check values | the bytes every engine and `craft.13` agree on |
| Reader rules | each reader rule refuses with the right error kind | corrupt or foreign bytes never reach the pool |
| The golden blobs | Rust writer and `parity/kv.wire.v1` agree byte for byte; the reader takes them back | parity with the file contract |
| RNG hand-off | `resume_rng` equals the stream advanced by 0, 1, 2 draws | seeded streams survive the hand-off |
| HasBlocks takes no reference | a hit leaves used and cached counts unchanged; format 2 refused | asking is free |
| Push accepts, dedups, and refuses | import, dedup, and every refusal leaving D exactly as before; an unregistered full block is registered on arrival | no leak on a bad push |
| Release | Release frees, is idempotent, accepts unknown handles; P holds nothing after a push | the gateway's abort path |
| Disaggregated equals colocated | the same 12 tokens through P and D as on one engine, greedy and two seeds, three prompts | the whole point |
| Bytes moved | sent, deduped, and payload bytes equal section 2.3's formula for 3 cache states | dedup really saves the transfer |
| A reset mid-push | a proxy cuts the push: the RPC fails, both pools return to baseline, both engines keep serving | a decode pod dying mid-transfer |
| An abandoned push | a push abandoned after 5 of 10 chunks leaves D clean | a cancelled stream is not a leak |
| An evicted block | NOT_FOUND on a deduped chunk makes P resend every payload once | a cache eviction between the two calls |
| Prefill validation and Info | role, empty fields, ranges refused before work; Info, Cancel, Drain | the gateway's routing mistakes are caught |
| Worker status fields | only decode advertises `kv_address`; format 1; live load | the registry routes KV correctly |
| Beat failures and drain | failures counted and reset, a hung registry bounded by the timeout, drain sticky | the engine outlives a registry outage |
| Heartbeat schedule | beats at once and every interval with fresh load; stop is prompt | the gateway's view is current |
| gRPC registry round trip | the gRPC client against a real `WorkerRegistry` server; an unreachable one errors in time | the production path |

## 5. The solution, step by step

### 5.1 From first principles

1. **Two engines must agree on what a block is without comparing contents.** Sending a block to ask "do you have this?" defeats the purpose. So each full block needs a short name that depends only on its tokens and everything before them, since a block's K and V depend on the whole prefix. That forces a chained hash, $h_i = H(h_{i-1}, \text{tokens}_i)$ (section 2.2).
2. **A partial block cannot be named.** Its contents change as tokens are appended, so it has hash 0, is never deduplicated, and is always sent. That gives the byte count $(F - k + t)$ envelopes of section 2.3.
3. **Bytes from the network are untrusted.** The reader must check length, magic, CRC, version, shape, and fills before allocating anything, and must classify the failure: corruption (DATA_LOSS, retry elsewhere) versus a format this worker will never read (FAILED_PRECONDITION).
4. **A push is a transaction.** D takes a reference per chunk as chunks arrive; if anything fails, the push must leave D exactly as before. So every taken block id goes into one `taken` vector, and the error path releases all of it. Only a complete push is stored under the handle.
5. **Asking must be free.** `HasBlocks` uses the pool's `lookup`, which takes a reference; the answer must release it at once, otherwise every routing question pins blocks.
6. **Between asking and pushing, D can evict.** A deduped chunk for a block that has since been evicted gets NOT_FOUND; the only safe answer is to resend every payload once.
7. **P owns the prompt's blocks until the push ends, either way.** After success, D has its own copies; after failure, nobody needs them. So P releases its references unconditionally, after `transfer` returns and before looking at its result.
8. **Seeded sampling must not notice the split.** Both engines use `stream(seed, sample)`. P used `rng_draws_consumed` uniforms for the first token, so D's generator must stand that many draws in (section 2.5).
9. **Liveness reporting must not depend on the registry being alive.** Each beat is wrapped in a timeout, failures are counted and never end the loop, and the heartbeat runs on its own thread and runtime, so a hung registry cannot touch the engine.

### 5.2 Hashes and checksums

```rust
/// The chained hash of one full block: fnv1a64(le_u64(parent) || le_u32(tokens)...), 0 replaced by 1.
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut buf = Vec::with_capacity(8 + 4 * tokens.len());
    buf.extend_from_slice(&parent.to_le_bytes());   // the parent makes h_i name the whole prefix
    for t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());     // little-endian: the same bytes on every engine
    }
    match fnv1a64(&buf) {
        0 => 1,     // 0 is reserved for "no hash" (a partial or unregistered block)
        h => h,
    }
}

pub fn prompt_hashes(tokens: &[u32], block_tokens: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(tokens.len() / block_tokens);
    let mut parent = 0;
    for chunk in tokens.chunks_exact(block_tokens) {   // chunks_exact drops the partial tail
        parent = block_hash(parent, chunk);
        out.push(parent);
    }
    out
}
```

`crc32c` is the bitwise reflected CRC with polynomial `0x82F63B78`, init and final xor all ones: eight shift-and-xor steps per byte. It is slow next to a table or the SSE4.2 instruction, but a 172-byte block is nothing next to the forward pass that produced it, and the bit loop is easy to check against `kv-block.md`'s check value.

### 5.3 The envelope reader

The reader's checks are ordered so that each one can trust what the previous ones established:

```rust
pub fn read_envelope(buf: &[u8], expect: Option<&KvCfg>) -> Result<(EnvHeader, Vec<BlockRecord>), EnvError> {
    if buf.len() < HEADER_BYTES + 4 { /* Format */ }      // enough bytes to read a header at all
    if &buf[..4] != b"TLKV" { /* Format */ }
    let h = EnvHeader { /* version, dtype, n_blocks, block_tokens, n_layers, n_kv_heads, head_dim */ };
    // u128: a hostile n_blocks cannot overflow the expected length
    let want_len = (h.n_blocks as u128) * (12 + h.payload_bytes() as u128) + HEADER_BYTES as u128 + 4;
    if buf.len() as u128 != want_len { /* Format */ }
    let body = &buf[..buf.len() - 4];
    if crc32c(body) != u32_at(buf, buf.len() - 4) {
        return Err(EnvError::Format("CRC mismatch".into()));          // -> DATA_LOSS
    }
    // only after the CRC: a version field we trust was really sent
    if h.version != 1 || h.dtype != DTYPE_F16 {
        return Err(EnvError::Version(/* ... */));                     // -> FAILED_PRECONDITION
    }
    if let Some(c) = expect {
        // B, L, Hkv, D must equal the receiving pool's, or the bytes mean something else
        // ... EnvError::Shape
    }
    // ... per block: n_tokens in 1..=B, and a non-zero hash only on a full block
}
```

`status_of` maps `EnvError::Version` to FAILED_PRECONDITION and everything else to DATA_LOSS. The CRC check comes before the version check deliberately: a flipped bit in the version field should read as corruption, not as "a format I do not read".

### 5.4 The decode side: accept one chunk

```rust
pub fn accept_chunk(&self, c: &KvChunk, expect_index: u32, taken: &mut Vec<u32>) -> Result<bool, Status> {
    if c.kv_format != KV_FORMAT { return Err(Status::failed_precondition(/* ... */)); }   // mixed-version rollout
    if c.block_index != expect_index { return Err(Status::invalid_argument(/* ... */)); }  // order is the block table
    let mut pool = lock(&self.pool);
    if c.payload.is_empty() {
        // a deduped chunk: take a reference on D's cached copy
        if c.block_hash == 0 { return Err(Status::invalid_argument("an empty payload needs the block's hash")); }
        let id = pool.lookup(c.block_hash).ok_or_else(|| Status::not_found(/* evicted since HasBlocks */))?;
        taken.push(id);
        return Ok(true);
    }
    if crc32c(&c.payload) != c.crc32c { return Err(Status::data_loss(/* ... */)); }
    let cfg = pool.cfg();
    let (_, recs) = read_envelope(&c.payload, Some(&cfg)).map_err(|e| status_of(&e))?;
    if recs.len() != 1 { return Err(Status::data_loss(/* ... */)); }
    // A full block P never registered travels with hash 0 in its envelope;
    // the chunk's hash names it. A registered one must agree.
    if recs[0].hash != 0 && recs[0].hash != c.block_hash { return Err(Status::data_loss(/* ... */)); }
    if c.block_hash != 0 && recs[0].n_tokens != cfg.block_tokens { return Err(Status::data_loss(/* ... */)); }
    let ids = pool.import(&c.payload).map_err(/* EFULL/ENOMEM -> RESOURCE_EXHAUSTED, else DATA_LOSS */)?;
    let id = ids[0];
    taken.push(id);                       // pushed before register: if register fails, the error path frees it
    if c.block_hash != 0 {
        pool.register(id, c.block_hash).map_err(|e| Status::internal(e.to_string()))?;   // the next prompt can dedup it
    }
    Ok(false)
}
```

Every refusal happens before `import`, so a refused chunk allocates nothing; what earlier chunks took is already in `taken`.

### 5.5 The decode side: the whole push is all or nothing

The gRPC handler reads the stream and keeps a single `taken` list:

```rust
async fn push_kv(&self, request: Request<Streaming<KvChunk>>) -> Result<Response<KvAck>, Status> {
    let mut s = request.into_inner();
    let mut taken: Vec<u32> = Vec::new();
    // ... ack, total, i
    let res: Result<(), Status> = async {
        // s.message() fails when the peer goes away mid-transfer: same path as a refusal
        while let Some(c) = s.message().await? {
            // ... the first chunk fixes handle_id and n_blocks_total; later chunks must match
            if self.0.accept_chunk(&c, i, &mut taken)? { ack.blocks_deduped += 1; } else { ack.blocks_received += 1; }
            i += 1;
        }
        if taken.len() as u32 != total || taken.is_empty() {
            return Err(Status::invalid_argument(/* stream ended early */));   // a clean early end is still incomplete
        }
        Ok(())
    }
    .await;
    match res {
        Ok(()) => { self.0.store(&ack.handle_id, taken); Ok(Response::new(ack)) }   // held until claim or Release
        Err(st) => { self.0.release_ids(&taken); Err(st) }                         // D exactly as before
    }
}
```

The async block plus one `match` is the Rust shape of "try, and on any error roll back": every `?` inside lands in `Err(st)`, so no exit path can skip `release_ids`. `store` releases what an older handle with the same id held, so a retried push does not leak the first attempt. `release` is `claim` then `release_ids`, so it is idempotent and accepts unknown handles.

`has_blocks` releases the reference `lookup` took:

```rust
for &h in hashes {
    match pool.lookup(h) {
        Some(id) => {
            pool.release(id).map_err(|e| Status::internal(e.to_string()))?;   // asking takes nothing
            out.push(true);
        }
        None => out.push(false),
    }
}
```

### 5.6 The prefill side

`build_chunks` turns the block table into chunks; `transfer` asks, pushes, and retries once on eviction:

```rust
pub fn build_chunks(pool: &SharedKv, handle_id: &str, ids: &[u32], hashes: &[u64], present: &[bool]) -> Result<Vec<KvChunk>, TransferError> {
    let pool = lock(pool);
    let mut out = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let hash = hashes.get(i).copied().unwrap_or(0);          // past the full blocks: the tail, hash 0
        let dedup = hash != 0 && present.get(i).copied().unwrap_or(false);
        let payload = if dedup { Vec::new() } else { pool.export(&[id]).map_err(/* ... */)? };   // a one-block envelope
        let crc = if payload.is_empty() { 0 } else { crc32c(&payload) };
        out.push(KvChunk { handle_id: handle_id.to_string(), block_hash: hash, block_index: i as u32,
                           n_blocks_total: ids.len() as u32, kv_format: KV_FORMAT, payload, crc32c: crc });
    }
    Ok(out)
}

pub async fn transfer(t: &dyn KvTransport, target: &str, pool: &SharedKv, handle_id: &str, ids: &[u32], hashes: &[u64]) -> Result<TransferStats, TransferError> {
    let present = t.has_blocks(target, hashes, KV_FORMAT).await?;
    // ... a short answer is an error
    let chunks = build_chunks(pool, handle_id, ids, hashes, &present)?;
    match t.push(target, chunks.clone()).await {
        Ok(ack) => Ok(stats(&chunks, &ack)),
        Err(e) if e.code == tonic::Code::NotFound => {
            // a cached block was evicted between the two calls: resend everything once
            let all = build_chunks(pool, handle_id, ids, hashes, &vec![false; hashes.len()])?;
            let ack = t.push(target, all.clone()).await?;
            Ok(stats(&all, &ack))
        }
        Err(e) => Err(e),
    }
}
```

In the section 3 transfer, `present = [true, false]`, so chunk 0 is empty with hash $h_0$, chunk 1 is a 172-byte envelope, and chunk 2 (the tail, `hashes.get(2)` is `None`) is a 172-byte envelope with hash 0: 344 payload bytes.

`ControlService::run_prefill` in [`control.rs`](../../../../course/solutions/rust/crates/tl-serve/src/control.rs) validates before any work (the engine must serve `prefill`; prompt and target must be non-empty; the deadline must be in the future; sampling ranges are checked by `sampling_from_proto`), runs the backend's prefill, and then:

```rust
let res = kv_transfer::transfer(self.transport.as_ref(), &r.decode_target, &pool, &handle_id, &out.block_ids, &out.block_hashes).await;
// P's references go either way: D holds its own copies now, and a
// registered full block stays cached here for the next prompt.
self.release(&out.block_ids);
res.map_err(|e| transfer_status(&e))?;   // an unreachable D becomes UNAVAILABLE: the gateway picks another
// ... KvHandle { handle_id, n_tokens, block_hashes, kv_format, first_token, prompt_tokens, rng_draws_consumed }
```

Releasing before `?` is the line that keeps a failed push from leaking the prompt's blocks on P.

### 5.7 The RNG hand-off

```rust
pub fn resume_rng(seed: u64, draws: u32) -> Pcg32 {
    let mut r = stream(seed, PURPOSE_SAMPLE);   // the same stream P seeded
    for _ in 0..draws {
        r.uniform_f64();                        // one uniform per sampled token; 0 when greedy
    }
    r
}
```

The serve loop's decode side ([`resume_generate`](../../../../course/solutions/rust/crates/tl-serve/src/server.rs)) reaches the same state another way: the resume request does not carry P's first token, so D claims the handle's blocks, recomputes the last prompt position (rewriting the same KV), and samples it from a fresh `stream(seed, sample)`. Given the same logits that draws the same first token P drew, and leaves the generator exactly where `resume_rng(seed, rng_draws_consumed)` would. Either way, token 2 uses the draw $u_2$ of section 3's hand-off example.

### 5.8 The heartbeat

```rust
pub async fn beat_once(reg: &dyn Registry, status: WorkerStatus, timeout: Duration, st: &BeatState) -> bool {
    st.sent.fetch_add(1, Ordering::SeqCst);
    match tokio::time::timeout(timeout, reg.heartbeat(status)).await {   // a hung registry costs at most `timeout`
        Ok(Ok(ack)) => {
            st.consecutive_failures.store(0, Ordering::SeqCst);         // a success resets the miss count
            if ack.drain {
                st.drain.store(true, Ordering::SeqCst);                 // sticky: never cleared by a later ack
            }
            st.route_epoch.store(ack.route_epoch, Ordering::SeqCst);
            true
        }
        _ => {                                                           // an error and a timeout count the same
            st.failed.fetch_add(1, Ordering::SeqCst);
            st.consecutive_failures.fetch_add(1, Ordering::SeqCst);
            false
        }
    }
}
```

`Heartbeat::spawn` runs a current-thread runtime on its own OS thread, ticks with `MissedTickBehavior::Delay` (a slow beat delays the next one instead of bursting), reads `load()` fresh before each beat, and `select!`s on a oneshot so `stop` returns promptly even mid-beat. `worker_status` advertises `kv_address` only when the role is `decode`, since only a decode engine receives pushes.

Full solution: [`kv_transfer.rs`](../../../../course/solutions/rust/crates/tl-engine/src/kv_transfer.rs), [`heartbeat.rs`](../../../../course/solutions/rust/crates/tl-engine/src/heartbeat.rs), [`control.rs`](../../../../course/solutions/rust/crates/tl-serve/src/control.rs), and the backend in [`server.rs`](../../../../course/solutions/rust/crates/tl-serve/src/server.rs).

### 5.9 Why this solution

- **Cost.** Hashing is $O(n)$ in prompt tokens; `HasBlocks` is $O(F)$ lookups; a transfer moves $(F - k + t)(44 + P)$ bytes. For a shared 2,000-token system prompt at $B = 16$, every block after the first request is deduped, and the push carries only the user's tail.
- **Why one chunk per block, not one big envelope?** A per-block chunk can be deduped individually, checked individually (a corrupt block names its index), and streamed without holding the whole prompt's KV in one message, which would hit gRPC's message size limit for long prompts.
- **Why a chained hash, not a hash of the block's own tokens?** Two prompts can share block 3's tokens and differ before it; their KV for block 3 differs because attention saw different prefixes. Only the chain makes equal hashes mean equal KV.
- **Why a separate prefill RPC rather than prefill over HTTP?** The gateway needs the `KvHandle` (hashes, draws, first token) as typed data, and gRPC streams fit the push; HTTP stays the client-facing API.
- **Tradeoffs accepted.** The eviction retry resends every payload rather than only the evicted ones: simpler, and evictions between two calls milliseconds apart are rare. `Cancel` and `Drain` on the engine backend are minimal (a prefill runs to completion; draining is SIGTERM's job).

### 5.10 Explain it back

**Why must `HasBlocks` release the reference that `lookup` takes?**

<details><summary>Answer</summary>

`lookup` returns a block with its refcount raised, which is right for a user of the block but wrong for a question. If `HasBlocks` kept it, every routing decision would pin the prompt's cached blocks, and the decode pool would shrink with every request the gateway merely considered. The push takes its own reference later, in `accept_chunk`.

</details>

**A proxy cuts the push after 5 of 10 chunks. Walk through what each side frees.**

<details><summary>Answer</summary>

On D, `s.message().await?` fails, the async block returns `Err`, and `release_ids(&taken)` drops the five references; the handle is never stored. On P, `transfer` returns an error, and `run_prefill` calls `self.release(&out.block_ids)` before propagating it, so P's references go too. Both pools return to baseline, and registered full blocks on P stay cached (with no request reference) for the next prompt.

</details>

**Why does the envelope reader check the CRC before the version field?**

<details><summary>Answer</summary>

Error kinds drive routing: FAILED_PRECONDITION means "never send this format here", DATA_LOSS means "retry, maybe elsewhere". If the version were read before the CRC, a single flipped bit in the version would be misreported as a format mismatch and the gateway would stop routing to a healthy worker. After the CRC passes, the version is the one the writer meant.

</details>

**Seed 77, temperature 0.9. What breaks if D starts from `stream(77, sample)` and samples token 2 directly?**

<details><summary>Answer</summary>

Token 2 would use $u_1$, the draw P already used for token 1, so every later token is shifted by one draw relative to a unified engine and the seeded stream diverges after the first token. D must stand one draw in: `resume_rng(77, 1)`, or, as the serve loop does, resample the first token from the fresh stream to reach the same state.

</details>

**Why is `drain` sticky in `BeatState`?**

<details><summary>Answer</summary>

The registry asks a worker to drain once; a later ack may not repeat the flag (the registry may restart, or the beat after the request may time out). If one ack without `drain` could clear it, the engine would resume taking traffic the gateway has already moved away. Draining is a one-way transition until the process restarts.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Hashing a block without its parent | two prompts that differ early dedup each other's later blocks: wrong KV | `block_hash` writes `le64(parent)` before the tokens; see 5.2 |
| Not checking the envelope CRC | a flipped bit on the wire becomes a wrong attention value | `read_envelope` checks the envelope CRC, and `accept_chunk` the chunk CRC, before importing; see 5.3 and 5.4 |
| A v1 reader accepting version 2 | fp8 bytes read as f16 during the `craft.13` migration | `read_envelope` refuses any version other than 1 or dtype other than f16 with FAILED_PRECONDITION |
| The writer dropping the block hash | D can never dedup what this engine sends | `build_chunks` sets `block_hash` on every full block's chunk, and `accept_chunk` registers it |
| Advancing the generator one draw too few | seeded disaggregated streams differ after the first token | `resume_rng` advances by `rng_draws_consumed`; see 5.7 |
| HasBlocks keeping its lookup's reference | blocks pinned by questions; the pool shrinks with every routing decision | `has_blocks` releases each hit at once; see 5.5 |
| A refused push keeping what it took | every bad request leaks blocks on D | one `taken` list and `release_ids(&taken)` on every error; see 5.5 |
| Not checking a chunk's `kv_format` | mixed-version rollouts corrupt KV instead of failing cleanly | `accept_chunk` checks `kv_format` first |
| Skipping a deduped block that is gone | a handle with a hole: decode reads an unowned block | a deduped chunk's `lookup` miss is NOT_FOUND, and `transfer` resends every payload once; see 5.6 |
| P never dropping its references | the prefill pool fills after a few hundred requests | `run_prefill` releases `out.block_ids` after every transfer; see 5.6 |
| Sending blocks D already holds | shared system prompts cost a full transfer every time | `HasBlocks` first, then empty payloads for present blocks |
| Returning the transfer error before releasing | each failed push leaks the prompt's blocks on P | `self.release(...)` runs before `res.map_err(...)?` |
| A decode-only engine running Prefill | a misrouted request builds KV nobody will read | `run_prefill` refuses with FAILED_PRECONDITION unless the roles include `prefill` |
| A success not resetting the miss count | the gateway's eviction logic sees phantom misses | `beat_once` stores 0 in `consecutive_failures` on every success; see 5.8 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.1` | `KvPool` (export, import, register, lookup, slabs) and `stream` for the hand-off |
| Back | `L10.1` | the Rust pool behind both engines: the export envelope, the prefix index, refcounts |
| Forward | `gw.05` | picks P and D, calls Prefill and Release, serves the registry the heartbeat reaches |
| Forward | `craft.14` | integrates heartbeat registration and registry-driven draining in the upgraded serve loop |
| Forward | `craft.13` | takes `kv_transfer.rs` over: format v2 (fp8), mixed-version negotiation |

The production deployment runs these engines as 1 prefill + 2 decode on kind, and the `ops.01` incident kills a decode pod mid-load. `obs.01` traces `kv.transfer` as a CLIENT span on P and a SERVER span on D.

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one gRPC stream per handle | NIXL, Mooncake transfer engine | RDMA and GPU-direct transfers, no copy through host memory | [Mooncake](https://github.com/kvcache-ai/Mooncake), [NIXL](https://github.com/ai-dynamo/nixl) |
| dedup by chained block hash | vLLM disaggregated prefill, LMCache | a shared KV store across many engines, not just the decode target | [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill.html), [LMCache](https://github.com/LMCache/LMCache) |
| prefill and decode as fixed roles | DistServe, Splitwise | role ratios tuned to the workload, and role changes at run time | the DistServe and Splitwise papers |
| a 2 s heartbeat | KV-aware routers (llm-d, Dynamo) | routing on each worker's cached prefixes, not just its load | [llm-d](https://github.com/llm-d/llm-d), [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) |
