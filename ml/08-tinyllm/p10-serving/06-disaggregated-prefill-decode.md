<!-- ol:module L10.6 -->
# Disaggregated prefill/decode, KV transfer, heartbeat client

## Overview

| | |
|---|---|
| **Module** | `L10.6` · build · Rust · Pass 7 · 10 to 14 h |
| **You build** | `rust/crates/tl-engine/src/kv_transfer.rs`: block hashes, CRC-32C, the export envelope in Rust, the decode side of `tl.kv.v1` (`KvReceiver`, `KvService`), the prefill side's deduplicated push (`transfer`), the RNG hand-off · `rust/crates/tl-engine/src/heartbeat.rs`: the heartbeat client to the gateway's worker registry · `rust/crates/tl-serve/src/control.rs`: `tl.engine.v1.EngineControl` (Prefill, Info, Cancel, Drain) over a `PrefillBackend` |
| **Contract** | gRPC: [`proto/tl/engine/v1/engine.proto`](../../../course/contracts/proto/tl/engine/v1/engine.proto), [`proto/tl/kv/v1/kv.proto`](../../../course/contracts/proto/tl/kv/v1/kv.proto), [`proto/tl/control/v1/control.proto`](../../../course/contracts/proto/tl/control/v1/control.proto), generated code in [`rust/tl-proto`](../../../course/contracts/rust/tl-proto/src/lib.rs) · bytes: [`formats/kv-block.md`](../../../course/contracts/formats/kv-block.md) · conformance: `parity/kv.wire.v1` |
| **Tests** | `course/tests/rust/l10_6.rs`, 17 tests (what they check: section 4) |
| **Needs** | `L10.1` `KvPool` and the sampler's `stream` ([chapter](01-model-runner-and-sampler.md)) · `L10.1` the Rust pool: export, import, register, lookup ([chapter](../p08-inference/08-paged-kv-block-pool.md)) · reading: `L10.2`, `L10.4`, `L10.5` the serve loop this plugs into, `lang.09` async Rust ([primer](../../../software-craftsmanship/12-language-and-tool-primers/09-async-rust-and-tokio.md)), `lang.10` gRPC ([primer](../../../software-craftsmanship/12-language-and-tool-primers/10-protocol-buffers-and-grpc.md)) · or `--ref-deps` |
| **Used by** | `gw.05` calls Prefill and Release over gRPC and serves the registry your heartbeat reaches; `craft.13` takes `kv_transfer.rs` over for KV format v2 |
| **Milestone** | `MS-L10` (and `MS-prod`, where the engines run as 1 prefill + 2 decode) |
| **Optional depth** | [DistServe](https://arxiv.org/abs/2401.09670) (free); [Splitwise](https://arxiv.org/abs/2311.18677) (free); [Mooncake](https://arxiv.org/abs/2407.00079) (free); [gRPC core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/) (free) |

## Key Takeaways

- A block's hash names its whole prefix (`H(parent, tokens)`), so "D already holds this block" is one table lookup, and only the missing full blocks plus the partial tail cross the network (`hand_example_envelope`, `bytes_moved_are_missing_blocks_plus_tail`).
- The decode side reads bytes from the network: every reader rule of `kv-block.md` refuses bad input before anything is allocated, with DATA_LOSS for corruption and FAILED_PRECONDITION for a format it does not read (`envelope_reader_rules`, `push_accepts_dedups_and_refuses`).
- Every error path frees what it took: a refused push, a stream cut mid-transfer, an abandoned client, a released handle; neither engine's pool may leak (`transfer_reset_frees_both_sides`, `abandoned_push_frees_decode_blocks`, `release_frees_the_handle`).
- Prefill on one engine and decode on another give the same tokens as one engine doing both, greedy and seeded, because the decode side advances its generator by exactly the draws the prefill side used (`disaggregated_equals_colocated`, `resume_rng_hands_off_the_draws`).
- The heartbeat never blocks the engine and never stops: each beat is bounded by a timeout, misses are counted, and a drain request is remembered (`beat_once_counts_failures_and_drain`).

## How to work this chapter

```bash
ol start L10.6               # stubs kv_transfer.rs, heartbeat.rs, control.rs
ol tests L10.6               # the test catalog
ol check L10.6               # exit code is the verdict
ol check L10.6 --ref-deps    # only if L10.1 or L10.1 is not passing yet
ol diff  L10.6               # after passing: your code against the reference
ol parity kv.wire.v1         # your Rust writer against the golden blobs
```

Add to your `tl-engine` manifest the dependencies this module needs: `tl-proto = { path = "../../../../contracts/rust/tl-proto" }`, `tokio` (full), `tonic = "0.14"`, and `prost = "0.14"` (all in `contracts/allowed-deps.toml`), and declare `pub mod kv_transfer; pub mod heartbeat;` in `tl-engine/src/lib.rs` and `pub mod control;` in `tl-serve/src/lib.rs`. Your `main.rs` starts the servers: `kv_transfer::serve_kv` on `[engine].kv_listen` for a decode engine, `control::serve_control` on `[engine].grpc_listen` for every engine, and `Heartbeat::spawn` toward `[engine].gateway_registry`.

---

## 1. Why now

After `L10.5` one engine runs both phases of every request: the compute-bound prefill of the prompt and the memory-bound decode of each new token. They compete for the same batch. A long prompt arriving while thirty streams decode makes every one of them stall for its prefill (TPOT spikes), and a burst of decodes makes a new prompt wait (TTFT spikes). Production systems split the two onto separate workers: prefill engines that only build KV caches, decode engines that only extend them. That needs three things your engine does not have yet: a way for the gateway to ask a prefill engine to run one prompt (`EngineControl.Prefill`), a way to move the prompt's KV blocks to the decode engine without moving the ones it already holds (`tl.kv.v1`), and a way for the gateway to know which engines exist and how loaded they are (the heartbeat to `tl.control.v1`). This module writes all three on top of the block pool `L10.1` gave you and the sampler `L10.1` gave you.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | tokens per block (`block_tokens`; the engine uses 16, the tests 4) | `u32` |
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

**One block, by hand** (the example of `kv-block.md`, test `hand_example_envelope`). $B = 2$, one layer, one head, $D = 2$, the full block of tokens `[1, 2]` with K = `[[1, 2], [3, 4]]` and V = `[[0.5, -1], [0, 0.25]]`.

1. Hash: the bytes are `00`x8 (parent 0), `01 00 00 00`, `02 00 00 00`; FNV-1a 64 gives `0xA91AB0C1027B9366`.
2. $P = 2 \cdot 1 \cdot 1 \cdot 2 \cdot 2 \cdot 2 = 16$, so $E = 28 + 28 + 4 = 60$ bytes.
3. f16: 1.0 is `0x3C00` (exponent 15 = the bias, mantissa 0), stored `00 3c`; 0.5 is `0x3800`; -1.0 is `0xBC00`; 0.25 is `0x3400`.
4. The CRC-32C of the first 56 bytes is `0x4F0B541C`, appended little-endian: `1c 54 0b 4f`.

**One transfer, by hand** (test shapes: $B = 4$, $L = 2$, $H_{kv} = 1$, $D = 4$, so $P = 2 \cdot 2 \cdot 1 \cdot 4 \cdot 4 \cdot 2 = 128$ and a one-block envelope is $28 + 12 + 128 + 4 = 172$ bytes). A 9-token prompt has $F = 2$ full blocks and a tail of 1 token ($t = 1$). D already caches block 0 ($k = 1$):

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

Your engine's step loop implements `PrefillBackend` (prefill the prompt through the block manager, sample the first token, hand back the block table) and, on a decode engine, starts a resumed request from `claim(handle)` with `resume_rng(seed, draws)` and `first_token` as its first input.

### What the tests check

The tests run a fake model that writes a fixed f16 pattern per position into real `L10.1` pools and whose logits hash every cached KV value, so any lost or stale block changes the next token. Prefill and decode engines run your servers on `127.0.0.1:0`.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_envelope` | unit | section 3's 60 bytes, the two chained hashes, FNV-1a and CRC-32C check values | the bytes every engine and `craft.13` agree on |
| `envelope_reader_rules` | boundary | each reader rule refuses with the right error kind | corrupt or foreign bytes never reach the pool |
| `rust_writer_matches_the_golden_blobs` | differential | Rust writer and `parity/kv.wire.v1` agree byte for byte; the reader takes them back | parity with the file contract |
| `resume_rng_hands_off_the_draws` | unit | `resume_rng` equals the stream advanced by 0, 1, 2 draws | seeded streams survive the hand-off |
| `has_blocks_takes_no_reference` | unit | a hit leaves used and cached counts unchanged; format 2 refused | asking is free |
| `push_accepts_dedups_and_refuses` | fault | import, dedup, and every refusal leaving D exactly as before; an unregistered full block is registered on arrival | no leak on a bad push |
| `release_frees_the_handle` | unit | Release frees, is idempotent, accepts unknown handles; P holds nothing after a push | the gateway's abort path |
| `disaggregated_equals_colocated` | differential | the same 12 tokens through P and D as on one engine, greedy and two seeds, three prompts | the whole point |
| `bytes_moved_are_missing_blocks_plus_tail` | property | sent, deduped, and payload bytes equal section 2.3's formula for 3 cache states | dedup really saves the transfer |
| `transfer_reset_frees_both_sides` | fault | a proxy cuts the push: the RPC fails, both pools return to baseline, both engines keep serving | a decode pod dying mid-transfer |
| `abandoned_push_frees_decode_blocks` | fault | a push abandoned after 5 of 10 chunks leaves D clean | a cancelled stream is not a leak |
| `evicted_block_is_resent` | fault | NOT_FOUND on a deduped chunk makes P resend every payload once | a cache eviction between the two calls |
| `prefill_rpc_validation_and_info` | boundary | role, empty fields, ranges refused before work; Info, Cancel, Drain | the gateway's routing mistakes are caught |
| `worker_status_fields` | unit | only decode advertises `kv_address`; format 1; live load | the registry routes KV correctly |
| `beat_once_counts_failures_and_drain` | fault | failures counted and reset, a hung registry bounded by the timeout, drain sticky | the engine outlives a registry outage |
| `heartbeat_beats_on_schedule` | unit | beats at once and every interval with fresh load; stop is prompt | the gateway's view is current |
| `grpc_registry_roundtrip` | conformance | the gRPC client against a real `WorkerRegistry` server; an unreachable one errors in time | the production path |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Hashing a block without its parent | two prompts that differ early dedup each other's later blocks: wrong KV | `hand_example_envelope` (mutant `s01`) |
| Not checking the envelope CRC | a flipped bit on the wire becomes a wrong attention value | `envelope_reader_rules` (mutant `s02`) |
| A v1 reader accepting version 2 | fp8 bytes read as f16 during the `craft.13` migration | `envelope_reader_rules` (mutant `s03`) |
| The writer dropping the block hash | D can never dedup what this engine sends | `envelope_reader_rules` (mutant `s04`) |
| Advancing the generator one draw too few | seeded disaggregated streams differ after the first token | `resume_rng_hands_off_the_draws` (mutant `s05`) |
| HasBlocks keeping its lookup's reference | blocks pinned by questions; the pool shrinks with every routing decision | `has_blocks_takes_no_reference` (mutant `s06`) |
| A refused push keeping what it took | every bad request leaks blocks on D | `push_accepts_dedups_and_refuses` (mutant `s07`) |
| Not checking a chunk's `kv_format` | mixed-version rollouts corrupt KV instead of failing cleanly | `push_accepts_dedups_and_refuses` (mutant `s08`) |
| Skipping a deduped block that is gone | a handle with a hole: decode reads an unowned block | `push_accepts_dedups_and_refuses` (mutant `s09`) |
| P never dropping its references | the prefill pool fills after a few hundred requests | `disaggregated_equals_colocated` (mutant `s10`) |
| Sending blocks D already holds | shared system prompts cost a full transfer every time | `bytes_moved_are_missing_blocks_plus_tail` (mutant `s11`) |
| Returning the transfer error before releasing | each failed push leaks the prompt's blocks on P | `transfer_reset_frees_both_sides` (mutant `s12`) |
| A decode-only engine running Prefill | a misrouted request builds KV nobody will read | `prefill_rpc_validation_and_info` (mutant `s13`) |
| A success not resetting the miss count | the gateway's eviction logic sees phantom misses | `beat_once_counts_failures_and_drain` (mutant `s14`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.1` | `KvPool` (export, import, register, lookup, slabs) and `stream` for the hand-off |
| Back | `L10.1` | the Rust pool behind both engines: the export envelope, the prefix index, refcounts |
| Forward | `gw.05` | picks P and D, calls Prefill and Release, serves the registry your heartbeat reaches |
| Forward | `craft.13` | takes `kv_transfer.rs` over: format v2 (fp8), mixed-version negotiation |

`MS-prod` runs your engines as 1 prefill + 2 decode on kind, and drill `ops.01` kills a decode pod mid-load. `obs.01` traces `kv.transfer` as a CLIENT span on P and a SERVER span on D.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one gRPC stream per handle | NIXL, Mooncake transfer engine | RDMA and GPU-direct transfers, no copy through host memory | [Mooncake](https://github.com/kvcache-ai/Mooncake), [NIXL](https://github.com/ai-dynamo/nixl) |
| dedup by chained block hash | vLLM disaggregated prefill, LMCache | a shared KV store across many engines, not just the decode target | [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill.html), [LMCache](https://github.com/LMCache/LMCache) |
| prefill and decode as fixed roles | DistServe, Splitwise | role ratios tuned to the workload, and role changes at run time | the DistServe and Splitwise papers |
| a 2 s heartbeat | KV-aware routers (llm-d, Dynamo) | routing on each worker's cached prefixes, not just its load | [llm-d](https://github.com/llm-d/llm-d), [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) |
