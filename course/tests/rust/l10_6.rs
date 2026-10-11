//! L10.6 course tests: disaggregated prefill/decode, KV transfer over
//! tl.kv.v1, the EngineControl Prefill RPC, and the heartbeat client.
//!
//! Annotated exemplars (DESIGN 5.12). The model here is a fake written in
//! this file: it stores a deterministic f16 pattern per (token, position,
//! layer, K/V, head, dim) in real Rust KV pools, and its logits
//! are a hash of EVERY cached KV value, so a block that is lost, stale, or
//! corrupted in transfer changes the next token. Prefill and decode run on
//! two pools, connected by your EngineControl and tl.kv.v1 servers on
//! 127.0.0.1:0; the colocated run uses one pool. The fault test puts a
//! byte-counting TCP proxy between them that cuts the connection mid-push.
//! The byte layouts are checked against formats/kv-block.md's worked example
//! and the parity/kv.wire.v1 golden blobs.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tl_engine::heartbeat::{self, BeatState, GrpcRegistry, Heartbeat, Load, Registry, WorkerIdentity};
use tl_engine::kv_transfer::tl_proto::tl::control::v1::worker_registry_server::{WorkerRegistry, WorkerRegistryServer};
use tl_engine::kv_transfer::tl_proto::tl::control::v1::{HeartbeatAck, WorkerStatus};
use tl_engine::kv_transfer::tl_proto::tl::engine::v1 as pb;
use tl_engine::kv_transfer::tl_proto::tl::engine::v1::engine_control_client::EngineControlClient;
use tl_engine::kv_transfer::tl_proto::tl::kv::v1::kv_transfer_service_client::KvTransferServiceClient;
use tl_engine::kv_transfer::tl_proto::tl::kv::v1::{KvAck, KvChunk, ReleaseRequest};
use tl_engine::kv_transfer::tonic::{self, Status};
use tl_engine::kv_transfer::{self as kt, tokio, BlockRecord, EnvError, EnvHeader, GrpcTransport, KvReceiver, KvTransport, SharedKv, TransferError};
use tl_engine::sample::{sample, stream, Pcg32, SamplingParams, PURPOSE_SAMPLE};
use tl_serve::control::{self, ControlService, PrefillBackend, PrefillJob, PrefillOutcome};
use tl_engine::kv::{KvCfg, KvPool, TL_F16};

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol"))
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap()
}

fn listener() -> TcpListener {
    TcpListener::bind("127.0.0.1:0").unwrap()
}

/// Polls `f` until it holds or 5 s pass.
fn eventually(what: &str, f: impl Fn() -> bool) {
    let t0 = Instant::now();
    while !f() {
        assert!(t0.elapsed() < Duration::from_secs(5), "timed out waiting for {what}");
        thread::sleep(Duration::from_millis(5));
    }
}

// ---------------------------------------------------------------------------
// the fake model

const B: u32 = 4;
const L: u32 = 2;
const H: u32 = 1;
const D: u32 = 4;
const V: usize = 32;

fn cfg(n_blocks: u32) -> KvCfg {
    KvCfg { n_blocks, block_tokens: B, n_layers: L, n_kv_heads: H, head_dim: D, dtype: TL_F16, format: 1 }
}

fn shared(n_blocks: u32) -> SharedKv {
    Arc::new(Mutex::new(KvPool::new(cfg(n_blocks)).unwrap()))
}

/// The f16 bit pattern the fake model stores: finite, in [1, 2).
fn kv_value(token: u32, pos: usize, layer: u32, v: u32, h: u32, d: u32) -> u16 {
    0x3C00 + ((token as u64 * 31 + pos as u64 * 7 + layer as u64 * 3 + v as u64 * 5 + h as u64 * 11 + d as u64) % 0x0400) as u16
}

/// Writes the KV of `token` at `pos` into the sequence's blocks, allocating
/// a block when pos starts one, and keeps the fill current.
fn write_pos(pool: &mut KvPool, table: &mut Vec<u32>, token: u32, pos: usize) {
    if pos % B as usize == 0 {
        table.push(pool.alloc(1).expect("a free block")[0]);
    }
    let id = table[pos / B as usize];
    let slot = pos % B as usize;
    for l in 0..L {
        for v in 0..2u32 {
            let slab = pool.slab_mut(id, l, v == 1).unwrap();
            for h in 0..H {
                for d in 0..D {
                    slab[((h * B) as usize + slot) * D as usize + d as usize] = kv_value(token, pos, l, v, h, d);
                }
            }
        }
    }
    pool.set_fill(id, slot as u32 + 1).unwrap();
}

/// Logits after `len` cached positions: a hash of every cached K and V
/// value, spread over the vocabulary.
fn logits(pool: &KvPool, table: &[u32], len: usize) -> Vec<f32> {
    let mut acc: u64 = 0xCBF29CE484222325;
    for pos in 0..len {
        let id = table[pos / B as usize];
        let slot = pos % B as usize;
        for l in 0..L {
            for v in [false, true] {
                let slab = pool.slab(id, l, v).unwrap();
                for h in 0..H as usize {
                    for d in 0..D as usize {
                        acc = (acc ^ slab[(h * B as usize + slot) * D as usize + d] as u64).wrapping_mul(0x100000001B3);
                    }
                }
            }
        }
    }
    (0..V).map(|i| ((acc.rotate_left(i as u32 * 7) ^ (i as u64 * 0x9E37)) % 1000) as f32 / 250.0).collect()
}

/// Prefill on `pool` the way an engine with a prefix cache does it: a full
/// block whose hash is cached is reused (a reference taken), every other
/// position computed, new full blocks registered under their chained
/// hashes; then the first token is sampled from the last position.
fn prefill(pool: &SharedKv, prompt: &[u32], p: &SamplingParams, seed: u64) -> (PrefillOutcome, Pcg32) {
    let mut g = kt::lock(pool);
    let mut table = Vec::new();
    let hashes = kt::prompt_hashes(prompt, B as usize);
    let mut cached = 0;
    let mut reused = vec![false; hashes.len()];
    for (pos, &t) in prompt.iter().enumerate() {
        let blk = pos / B as usize;
        if pos % B as usize == 0 && blk < hashes.len() {
            if let Some(id) = g.lookup(hashes[blk]) {
                table.push(id);
                reused[blk] = true;
                cached += B;
                continue;
            }
        }
        if blk < reused.len() && reused[blk] {
            continue; // inside a reused block
        }
        write_pos(&mut g, &mut table, t, pos);
        if (pos + 1) % B as usize == 0 {
            g.register(table[blk], hashes[blk]).unwrap();
        }
    }
    let mut rng = stream(seed, PURPOSE_SAMPLE);
    let (tok, lp) = sample(&logits(&g, &table, prompt.len()), p, prompt, &[], &mut rng);
    let draws = if p.temperature == 0.0 { 0 } else { 1 };
    (PrefillOutcome { first_token: tok, first_logprob: lp as f32, block_ids: table, block_hashes: hashes, cached_prompt_tokens: cached, rng_draws: draws }, rng)
}

/// Decode after a prefill: feed `first` and continue to n tokens in all.
fn decode(pool: &SharedKv, mut table: Vec<u32>, prompt: &[u32], first: u32, mut rng: Pcg32, p: &SamplingParams, n: usize) -> (Vec<u32>, Vec<u32>) {
    let mut g = kt::lock(pool);
    let mut out = vec![first];
    while out.len() < n {
        let pos = prompt.len() + out.len() - 1;
        write_pos(&mut g, &mut table, *out.last().unwrap(), pos);
        let (tok, _) = sample(&logits(&g, &table, pos + 1), p, prompt, &out, &mut rng);
        out.push(tok);
    }
    (out, table)
}

fn release_all(pool: &SharedKv, ids: &[u32]) {
    let mut g = kt::lock(pool);
    for &id in ids {
        g.release(id).unwrap();
    }
}

fn used(pool: &SharedKv) -> u32 {
    kt::lock(pool).stats().used
}

/// The PrefillBackend of the fake model.
struct FakeBackend {
    pool: SharedKv,
    calls: AtomicUsize,
}

#[tonic::async_trait]
impl PrefillBackend for FakeBackend {
    async fn prefill(&self, job: PrefillJob) -> Result<PrefillOutcome, Status> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(prefill(&self.pool, &job.prompt, &job.params, job.seed).0)
    }
    fn pool(&self) -> SharedKv {
        Arc::clone(&self.pool)
    }
    fn info(&self) -> pb::InfoResponse {
        pb::InfoResponse { model: "fake".into(), max_context: 64, block_size: B as i32, abi: 1, ..Default::default() }
    }
    async fn cancel(&self, request_id: &str) -> bool {
        request_id == "known"
    }
    async fn drain(&self, deadline_ms: i64) -> i32 {
        deadline_ms as i32 / 1000
    }
}

/// A prefill engine (EngineControl over the fake model and gRPC KV
/// transport) and a decode engine (tl.kv.v1 receiver), both serving.
struct Pair {
    p_pool: SharedKv,
    d_pool: SharedKv,
    recv: Arc<KvReceiver>,
    control_addr: std::net::SocketAddr,
    kv_addr: std::net::SocketAddr,
    _servers: Vec<kt::GrpcServer>,
}

fn pair(roles: &[&str]) -> Pair {
    let p_pool = shared(64);
    let d_pool = shared(64);
    let recv = Arc::new(KvReceiver::new(Arc::clone(&d_pool)));
    let kv = kt::serve_kv(listener(), Arc::clone(&recv)).unwrap();
    let svc = ControlService {
        backend: Arc::new(FakeBackend { pool: Arc::clone(&p_pool), calls: AtomicUsize::new(0) }),
        transport: Arc::new(GrpcTransport { timeout: Duration::from_secs(5) }),
        roles: roles.iter().map(|s| s.to_string()).collect(),
    };
    let ctl = control::serve_control(listener(), svc).unwrap();
    Pair { p_pool, d_pool, recv, control_addr: ctl.addr, kv_addr: kv.addr, _servers: vec![kv, ctl] }
}

fn prefill_request(id: &str, prompt: &[u32], target: &str, temperature: f32, seed: u64, max_new: i32) -> pb::PrefillRequest {
    pb::PrefillRequest {
        request_id: id.into(),
        model: "fake".into(),
        prompt_ids: prompt.to_vec(),
        sampling: Some(pb::SamplingParams { temperature, top_k: if temperature > 0.0 { 8 } else { 0 }, seed, max_new_tokens: max_new, ..Default::default() }),
        decode_target: target.into(),
        deadline_unix_ms: 0,
        priority: 0,
    }
}

async fn call_prefill(addr: std::net::SocketAddr, r: pb::PrefillRequest) -> Result<pb::PrefillResponse, Status> {
    let mut c = EngineControlClient::connect(format!("http://{addr}")).await.map_err(|e| Status::unavailable(e.to_string()))?;
    c.prefill(r).await.map(|r| r.into_inner())
}

fn params(temperature: f32) -> SamplingParams {
    let mut p = SamplingParams::default();
    p.temperature = temperature as f64;
    if temperature > 0.0 {
        p.top_k = 8;
    }
    p
}

// ---------------------------------------------------------------------------
// bytes

fn le_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn hand_example_envelope() {
    // WHY: formats/kv-block.md's worked example, byte for byte: B = 2, one
    //      layer, one head, D = 2; the full block [1 2] with K = [[1 2] [3
    //      4]] and V = [[0.5 -1] [0 0.25]] (f16 3C00 4000 4200 4400 / 3800
    //      BC00 0000 3400); hash fnv1a64(le64(0) le32(1) le32(2)) =
    //      0xA91AB0C1027B9366; a 60-byte envelope ending in CRC-32C
    //      0x4F0B541C. The check values of FNV-1a and CRC-32C too.
    // KIND: unit
    // CATCHES: s01, s04, m02
    // CHAPTER: L10.6 section 3
    assert_eq!(kt::fnv1a64(b"a"), 0xAF63DC4C8601EC8C);
    assert_eq!(kt::crc32c(b"123456789"), 0xE3069283);
    assert_eq!(kt::block_hash(0, &[1, 2]), 0xA91AB0C1027B9366);
    assert_eq!(kt::prompt_hashes(&[1, 2, 3, 4, 5], 2), vec![0xA91AB0C1027B9366, 0x03DF829571605C60]);
    let mut payload = Vec::new();
    for x in [0x3C00u16, 0x4000, 0x4200, 0x4400, 0x3800, 0xBC00, 0x0000, 0x3400] {
        payload.extend_from_slice(&x.to_le_bytes());
    }
    let h = EnvHeader { version: 1, dtype: 1, n_blocks: 1, block_tokens: 2, n_layers: 1, n_kv_heads: 1, head_dim: 2 };
    assert_eq!((h.payload_bytes(), h.envelope_bytes()), (16, 60));
    let env = kt::write_envelope(&h, &[BlockRecord { hash: 0xA91AB0C1027B9366, n_tokens: 2, payload: payload.clone() }]);
    let want = "544c4b5601000100010000000200000001000000010000000200000066937b02c1b01aa902000000003c004000420044003800bc000000341c540b4f";
    assert_eq!(le_hex(&env), want);
    let (h2, recs) = kt::read_envelope(&env, None).unwrap();
    assert_eq!((h2, recs[0].hash, recs[0].n_tokens, recs[0].payload.clone()), (h, 0xA91AB0C1027B9366, 2, payload));
}

#[test]
fn envelope_reader_rules() {
    // WHY: the decode side imports bytes from the network: every reader rule
    //      of kv-block.md refuses, with the right kind of error, before
    //      anything is allocated: short, bad magic, a flipped bit (CRC), a
    //      length that does not match the header, version 2 (a v1 reader
    //      refuses it cleanly: FAILED_PRECONDITION, not DATA_LOSS), another
    //      pool shape, n_tokens 0 or > B, a hash on a partial block.
    // KIND: boundary
    // CATCHES: s02, s03, s04, m03, m04
    // CHAPTER: L10.6 section 5, Pitfalls
    let c = cfg(4);
    let h = EnvHeader::v1(&c, 1);
    let rec = |hash: u64, n: u32| BlockRecord { hash, n_tokens: n, payload: vec![7u8; h.payload_bytes()] };
    let good = kt::write_envelope(&h, &[rec(5, B)]);
    assert!(kt::read_envelope(&good, Some(&c)).is_ok());
    assert!(matches!(kt::read_envelope(&good[..20], None), Err(EnvError::Format(_))));
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(matches!(kt::read_envelope(&magic, None), Err(EnvError::Format(_))));
    let mut flip = good.clone();
    flip[40] ^= 0x04;
    assert!(matches!(kt::read_envelope(&flip, None), Err(EnvError::Format(m)) if m.contains("CRC")));
    let mut short = good.clone();
    short.truncate(good.len() - 6);
    assert!(matches!(kt::read_envelope(&short, None), Err(EnvError::Format(_))));
    let v2 = kt::write_envelope(&EnvHeader { version: 2, dtype: 3, ..h }, &[rec(5, B)]);
    assert!(matches!(kt::read_envelope(&v2, None), Err(EnvError::Version(_))));
    let other = KvCfg { head_dim: 8, ..c };
    assert!(matches!(kt::read_envelope(&good, Some(&other)), Err(EnvError::Shape(_))));
    assert!(matches!(kt::read_envelope(&kt::write_envelope(&h, &[rec(0, 0)]), None), Err(EnvError::Format(_))));
    assert!(matches!(kt::read_envelope(&kt::write_envelope(&h, &[rec(0, B + 1)]), None), Err(EnvError::Format(_))));
    assert!(matches!(kt::read_envelope(&kt::write_envelope(&h, &[rec(9, B - 1)]), None), Err(EnvError::Format(_))), "a hashed block must be full");
    assert!(kt::read_envelope(&kt::write_envelope(&h, &[rec(0, B - 1)]), None).is_ok(), "an unhashed partial tail is fine");
}

#[test]
fn rust_writer_matches_the_golden_blobs() {
    // WHY: parity/kv.wire.v1: the Rust writer and the golden blobs agree byte
    //      for byte on six sequences, including partial tails (zeros past the fill) and
    //      multi-layer, multi-head shapes; and the Rust reader takes all of
    //      them back.
    // KIND: differential
    // CATCHES: s01, m02
    // CHAPTER: L10.6 section 2
    let fx = j::parse(&std::fs::read_to_string(fixtures().join("parity/kv_wire_v1.json")).unwrap());
    for c in fx.get("cases").arr() {
        let name = c.get("name").str();
        let inp = c.get("input");
        let (b, l, h, d) = (inp.get("B").u64() as u32, inp.get("L").u64() as u32, inp.get("H").u64() as u32, inp.get("D").u64() as u32);
        let toks: Vec<u32> = inp.get("tokens").arr().iter().map(|x| x.u64() as u32).collect();
        let vals: Vec<u16> = inp.get("values").arr().iter().map(|x| x.u64() as u16).collect();
        let n = toks.len();
        let nb = n.div_ceil(b as usize);
        let hashes = kt::prompt_hashes(&toks, b as usize);
        let kcfg = KvCfg { n_blocks: nb as u32, block_tokens: b, n_layers: l, n_kv_heads: h, head_dim: d, dtype: TL_F16, format: 1 };
        // Rust writer: records straight from the values.
        let mut recs = Vec::new();
        for blk in 0..nb {
            let fill = (n - blk * b as usize).min(b as usize);
            let mut payload = Vec::new();
            for li in 0..l as usize {
                for kv in 0..2 {
                    for hi in 0..h as usize {
                        for slot in 0..b as usize {
                            for di in 0..d as usize {
                                let pos = blk * b as usize + slot;
                                let x = if slot < fill { vals[(((pos * l as usize + li) * 2 + kv) * h as usize + hi) * d as usize + di] } else { 0 };
                                payload.extend_from_slice(&x.to_le_bytes());
                            }
                        }
                    }
                }
            }
            recs.push(BlockRecord { hash: hashes.get(blk).copied().unwrap_or(0), n_tokens: fill as u32, payload });
        }
        let rust = le_hex(&kt::write_envelope(&EnvHeader::v1(&kcfg, nb as u32), &recs));
        assert_eq!(rust, c.get("output").get("hex").str(), "{name}: Rust writer");
        // Rust KV pool exporter: the same blocks written into a pool.
        let mut pool = KvPool::new(kcfg).unwrap();
        let ids = pool.alloc(nb).unwrap();
        for pos in 0..n {
            for li in 0..l {
                for kv in 0..2usize {
                    let slab = pool.slab_mut(ids[pos / b as usize], li, kv == 1).unwrap();
                    for hi in 0..h as usize {
                        for di in 0..d as usize {
                            slab[(hi * b as usize + pos % b as usize) * d as usize + di] = vals[(((pos * l as usize + li as usize) * 2 + kv) * h as usize + hi) * d as usize + di];
                        }
                    }
                }
            }
        }
        for (blk, &id) in ids.iter().enumerate() {
            pool.set_fill(id, (n - blk * b as usize).min(b as usize) as u32).unwrap();
            if let Some(&hh) = hashes.get(blk) {
                pool.register(id, hh).unwrap();
            }
        }
        let c_bytes = pool.export(&ids).unwrap();
        assert_eq!(le_hex(&c_bytes), c.get("output").get("hex").str(), "{name}: C export");
        let (_, back) = kt::read_envelope(&c_bytes, Some(&kcfg)).unwrap();
        assert_eq!(back, recs, "{name}: Rust reader");
    }
}

#[test]
fn resume_rng_hands_off_the_draws() {
    // WHY: P sampled the first token with one uniform draw of the request's
    //      generator; D must continue from the next draw, or every seeded
    //      token after the first changes. Greedy takes no draw.
    // KIND: unit
    // CATCHES: s05
    // CHAPTER: L10.6 section 2
    let mut a = stream(77, PURPOSE_SAMPLE);
    a.uniform_f64();
    assert_eq!(kt::resume_rng(77, 1), a);
    assert_eq!(kt::resume_rng(77, 0), stream(77, PURPOSE_SAMPLE));
    let mut c = stream(77, PURPOSE_SAMPLE);
    c.uniform_f64();
    c.uniform_f64();
    assert_eq!(kt::resume_rng(77, 2), c);
}

// ---------------------------------------------------------------------------
// the decode side, in process

fn chunks_for(pool: &SharedKv, prompt: &[u32], handle: &str, present: &[bool]) -> (Vec<KvChunk>, Vec<u32>) {
    let (out, _) = prefill(pool, prompt, &SamplingParams::greedy(), 0);
    let ch = kt::build_chunks(pool, handle, &out.block_ids, &out.block_hashes, present).unwrap();
    (ch, out.block_ids)
}

#[test]
fn has_blocks_takes_no_reference() {
    // WHY: HasBlocks only answers; it must not pin blocks (a gateway that
    //      asks and then picks another worker would leak them). After a hit
    //      the pool's used and cached counts are what they were. Any
    //      kv_format but 1 is FAILED_PRECONDITION.
    // KIND: unit
    // CATCHES: s06, m06
    // CHAPTER: L10.6 section 4
    let pool = shared(16);
    let recv = KvReceiver::new(Arc::clone(&pool));
    let (out, _) = prefill(&pool, &[1, 2, 3, 4, 5, 6, 7, 8, 9], &SamplingParams::greedy(), 0);
    release_all(&pool, &out.block_ids); // full blocks stay cached, the tail is freed
    let before = kt::lock(&pool).stats();
    assert_eq!((before.used, before.cached), (0, 2));
    let got = recv.has_blocks(&[out.block_hashes[0], 12345, out.block_hashes[1]], 1).unwrap();
    assert_eq!(got, vec![true, false, true]);
    assert_eq!(kt::lock(&pool).stats(), before, "HasBlocks changed the pool");
    assert_eq!(recv.has_blocks(&[1], 2).unwrap_err().code(), tonic::Code::FailedPrecondition);
}

#[test]
fn push_accepts_dedups_and_refuses() {
    // WHY: the decode side of PushKv, chunk by chunk: a payload block is
    //      imported (and a full one registered), an empty one takes a
    //      reference on the cached copy; and every refusal (a flipped bit
    //      with the old CRC, kv_format 2, a dedup of a block no longer
    //      cached, a stream that ends early) frees EVERY block taken for the
    //      push, so D's pool is exactly as before.
    // KIND: fault
    // CATCHES: s07, s08, s09, m07, m08, m09
    // CHAPTER: L10.6 section 5, Pitfalls
    let p_pool = shared(32);
    let d_pool = shared(32);
    let recv = KvReceiver::new(Arc::clone(&d_pool));
    let prompt: Vec<u32> = (10..19).collect(); // 2 full blocks + a tail of 1
    let (ch, p_ids) = chunks_for(&p_pool, &prompt, "h1", &[false, false]);
    let ack = recv.accept_all(ch.clone().into_iter().map(Ok)).unwrap();
    assert_eq!((ack.handle_id.as_str(), ack.blocks_received, ack.blocks_deduped), ("h1", 3, 0));
    assert_eq!((used(&d_pool), recv.pending()), (3, 1));
    let ids = recv.claim("h1").unwrap();
    assert_eq!(ids.len(), 3);
    release_all(&d_pool, &ids);
    assert_eq!(kt::lock(&d_pool).stats().cached, 2, "full blocks stay cached on D");
    // The same prompt again, deduped: the two full blocks are references.
    let (ch2, _) = chunks_for(&p_pool, &prompt, "h2", &[true, true]);
    assert!(ch2[0].payload.is_empty() && ch2[1].payload.is_empty() && !ch2[2].payload.is_empty());
    let ack = recv.accept_all(ch2.into_iter().map(Ok)).unwrap();
    assert_eq!((ack.blocks_received, ack.blocks_deduped), (1, 2));
    recv.release("h2");
    let baseline = kt::lock(&d_pool).stats();
    assert_eq!(baseline.used, 0);
    // Refusals leave the pool as it was.
    let refuse = |chunks: Vec<KvChunk>, code: tonic::Code| {
        let e = recv.accept_all(chunks.into_iter().map(Ok)).unwrap_err();
        assert_eq!(e.code(), code, "{}", e.message());
        assert_eq!(kt::lock(&d_pool).stats(), baseline, "a refused push leaked blocks ({code:?})");
        assert_eq!(recv.pending(), 0);
    };
    let mut bad = ch.clone();
    bad[2].payload[40] ^= 1;
    refuse(bad, tonic::Code::DataLoss);
    let mut v2 = ch.clone();
    v2[1].kv_format = 2;
    refuse(v2, tonic::Code::FailedPrecondition);
    let mut gone = ch.clone();
    gone[1].payload.clear();
    gone[1].block_hash = 0xDEAD;
    gone[1].crc32c = 0;
    refuse(gone, tonic::Code::NotFound);
    refuse(ch[..2].to_vec(), tonic::Code::InvalidArgument);
    release_all(&p_pool, &p_ids);
    // A full block P never registered travels with hash 0 in its envelope;
    // D registers it under the chunk's hash, so the next prompt dedups.
    let mut tab = Vec::new();
    {
        let mut g = kt::lock(&p_pool);
        for (pos, t) in [50u32, 51, 52, 53].into_iter().enumerate() {
            write_pos(&mut g, &mut tab, t, pos);
        }
    }
    let h = kt::prompt_hashes(&[50, 51, 52, 53], B as usize);
    let ch3 = kt::build_chunks(&p_pool, "h3", &tab, &h, &[false]).unwrap();
    recv.accept_all(ch3.into_iter().map(Ok)).unwrap();
    recv.release("h3");
    assert_eq!(recv.has_blocks(&h, 1).unwrap(), vec![true], "an unregistered full block is registered on arrival");
    release_all(&p_pool, &tab);
}

#[test]
fn release_frees_the_handle() {
    // WHY: the gateway's abort path: Release frees what a handle holds (the
    //      tail is freed, full blocks fall back to the cache), an unknown
    //      handle is fine, and releasing twice is harmless.
    // KIND: unit
    // CATCHES: s10, m09, m10
    // CHAPTER: L10.6 section 4
    let pair = pair(&["prefill"]);
    let r = rt();
    let resp = r.block_on(call_prefill(pair.control_addr, prefill_request("abort-me", &(0..10).collect::<Vec<u32>>(), &pair.kv_addr.to_string(), 0.0, 1, 4))).unwrap();
    assert_eq!(resp.handle.unwrap().handle_id, "abort-me");
    assert_eq!(used(&pair.d_pool), 3);
    r.block_on(async {
        let mut c = KvTransferServiceClient::connect(format!("http://{}", pair.kv_addr)).await.unwrap();
        c.release(ReleaseRequest { handle_id: "abort-me".into() }).await.unwrap();
        c.release(ReleaseRequest { handle_id: "abort-me".into() }).await.unwrap();
        c.release(ReleaseRequest { handle_id: "never-was".into() }).await.unwrap();
    });
    assert_eq!(used(&pair.d_pool), 0);
    assert_eq!(pair.recv.pending(), 0);
    assert_eq!(used(&pair.p_pool), 0, "P keeps no references after the push");
}

// ---------------------------------------------------------------------------
// end to end

#[test]
fn disaggregated_equals_colocated() {
    // WHY: the whole point: prefill on P, KV over gRPC, decode on D gives
    //      the same tokens as one engine doing both, greedy and seeded. The
    //      fake model's logits hash every cached KV value, so a lost or
    //      stale block, a wrong tail, or a skipped RNG draw shows up as a
    //      different token. Prompts with and without a partial tail block.
    // KIND: differential
    // CATCHES: s05, s06, s10
    // CHAPTER: L10.6 section 2
    let pair = pair(&["prefill"]);
    let r = rt();
    let prompts: [Vec<u32>; 3] = [(1..9).collect(), (3..14).collect(), vec![5, 5, 5, 9, 2]];
    for (pi, prompt) in prompts.iter().enumerate() {
        for (temp, seed) in [(0.0f32, 0u64), (0.9, 11), (1.3, 1 << 40)] {
            let p = params(temp);
            // colocated: one pool
            let solo = shared(64);
            let (out, rng) = prefill(&solo, prompt, &p, seed);
            let (want, table) = decode(&solo, out.block_ids.clone(), prompt, out.first_token, rng, &p, 12);
            release_all(&solo, &table);
            // disaggregated
            let id = format!("req-{pi}-{seed}");
            let resp = r.block_on(call_prefill(pair.control_addr, prefill_request(&id, prompt, &pair.kv_addr.to_string(), temp, seed, 12))).unwrap();
            let h = resp.handle.clone().unwrap();
            assert_eq!((h.first_token, h.n_tokens, h.prompt_tokens, h.kv_format), (resp.first_token, prompt.len() as u32, prompt.len() as u32, 1));
            assert_eq!(h.rng_draws_consumed, if temp == 0.0 { 0 } else { 1 });
            assert_eq!(h.block_hashes, kt::prompt_hashes(prompt, B as usize));
            let ids = pair.recv.claim(&h.handle_id).expect("the handle's blocks are on D");
            let (got, table) = decode(&pair.d_pool, ids, prompt, h.first_token, kt::resume_rng(seed, h.rng_draws_consumed), &p, 12);
            assert_eq!(got, want, "prompt {pi} T {temp} seed {seed}");
            release_all(&pair.d_pool, &table);
        }
    }
    assert_eq!((used(&pair.p_pool), used(&pair.d_pool)), (0, 0), "no block leaked on either side");
}

#[test]
fn bytes_moved_are_missing_blocks_plus_tail() {
    // WHY: dedup by hash is what makes a shared system prompt cheap to move:
    //      when D already caches the first k full blocks, P sends only the
    //      other full blocks and the partial tail, and the bytes on the wire
    //      are exactly (missing full blocks + 1) envelopes of one block.
    // KIND: property
    // CATCHES: s11, m01, m05, m11
    // CHAPTER: L10.6 section 2
    let pair = pair(&["prefill"]);
    let r = rt();
    let one = EnvHeader::v1(&cfg(1), 1).envelope_bytes() as u64;
    let system: Vec<u32> = (100..112).collect(); // 3 full blocks
    // First request: D holds nothing; afterwards its full blocks are cached on D.
    let mut first = system.clone();
    first.extend([7, 8]);
    let r1 = r.block_on(call_prefill(pair.control_addr, prefill_request("a", &first, &pair.kv_addr.to_string(), 0.0, 0, 2))).unwrap();
    pair.recv.release(&r1.handle.unwrap().handle_id);
    for (k, extra) in [(3usize, vec![1, 2, 3, 4, 5]), (3, vec![9]), (2, vec![])] {
        let mut prompt: Vec<u32> = system[..k * B as usize].to_vec();
        prompt.extend(&extra);
        let full = prompt.len() / B as usize;
        let tail = prompt.len() % B as usize != 0;
        let (out, _) = prefill(&pair.p_pool, &prompt, &SamplingParams::greedy(), 0);
        let present = pair.recv.has_blocks(&out.block_hashes, 1).unwrap();
        let cached = present.iter().filter(|&&x| x).count();
        assert_eq!(cached, k.min(full));
        let t = GrpcTransport { timeout: Duration::from_secs(5) };
        let stats = r.block_on(kt::transfer(&t, &pair.kv_addr.to_string(), &pair.p_pool, "b", &out.block_ids, &out.block_hashes)).unwrap();
        let missing = full - cached + tail as usize;
        assert_eq!((stats.sent as usize, stats.deduped as usize, stats.chunks as usize), (missing, cached, out.block_ids.len()));
        assert_eq!(stats.payload_bytes, missing as u64 * one, "bytes on the wire for {k} cached blocks of {full} (+tail {tail})");
        release_all(&pair.p_pool, &out.block_ids);
        pair.recv.release("b");
    }
}

/// A TCP proxy that forwards both ways and cuts the connection once
/// `limit` bytes went from client to server.
fn cutting_proxy(upstream: std::net::SocketAddr, limit: usize, cut: Arc<AtomicBool>) -> std::net::SocketAddr {
    let l = listener();
    let addr = l.local_addr().unwrap();
    thread::spawn(move || {
        for conn in l.incoming() {
            let Ok(c) = conn else { return };
            let Ok(s) = TcpStream::connect(upstream) else { return };
            let (c2, s2) = (c.try_clone().unwrap(), s.try_clone().unwrap());
            let cut2 = Arc::clone(&cut);
            thread::spawn(move || {
                let (mut c, mut s) = (c, s);
                let mut buf = [0u8; 4096];
                let mut sent = 0;
                loop {
                    let n = match c.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    let take = n.min(limit.saturating_sub(sent));
                    if s.write_all(&buf[..take]).is_err() {
                        break;
                    }
                    sent += take;
                    if sent >= limit {
                        cut2.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                let _ = c.shutdown(Shutdown::Both);
                let _ = s.shutdown(Shutdown::Both);
            });
            thread::spawn(move || {
                let (mut s, mut c) = (s2, c2);
                let mut buf = [0u8; 4096];
                while let Ok(n) = s.read(&mut buf) {
                    if n == 0 || c.write_all(&buf[..n]).is_err() {
                        break;
                    }
                }
                let _ = c.shutdown(Shutdown::Both);
            });
        }
    });
    addr
}

#[test]
fn transfer_reset_frees_both_sides() {
    // WHY: a decode worker's connection dies mid-transfer (a pod killed, a
    //      network fault): the Prefill RPC fails (the gateway retries
    //      elsewhere), P drops its references, and D frees every block it
    //      had taken for the half-received handle. Nothing leaks on either
    //      side, and both engines keep serving.
    // KIND: fault
    // CATCHES: s12, m10
    // CHAPTER: L10.6 section 5, Pitfalls
    let pair = pair(&["prefill"]);
    let r = rt();
    // 16 blocks of 172-byte envelopes: about 2.8 KB in the PushKv stream.
    // Each connection through the proxy may carry 1200 bytes upstream: the
    // HasBlocks call fits, the push is cut part way.
    let prompt: Vec<u32> = (0..64).collect();
    let cut = Arc::new(AtomicBool::new(false));
    let proxy = cutting_proxy(pair.kv_addr, 1200, Arc::clone(&cut));
    let err = r.block_on(call_prefill(pair.control_addr, prefill_request("cut", &prompt, &proxy.to_string(), 0.0, 0, 2))).unwrap_err();
    assert!(cut.load(Ordering::SeqCst), "the proxy never cut the stream");
    assert!(matches!(err.code(), tonic::Code::Unavailable | tonic::Code::Internal | tonic::Code::Unknown | tonic::Code::Cancelled), "{:?}: {}", err.code(), err.message());
    assert_eq!(used(&pair.p_pool), 0, "P kept references after a failed push");
    eventually("D to free the half-received blocks", || used(&pair.d_pool) == 0);
    assert_eq!(pair.recv.pending(), 0);
    // Both still serve.
    let ok = r.block_on(call_prefill(pair.control_addr, prefill_request("after", &prompt, &pair.kv_addr.to_string(), 0.0, 0, 2))).unwrap();
    assert_eq!(ok.handle.unwrap().handle_id, "after");
    pair.recv.release("after");
}

#[test]
fn abandoned_push_frees_decode_blocks() {
    // WHY: the decode side's own cleanup: a PushKv whose client vanishes
    //      after some chunks arrived (the stream is cancelled, not ended)
    //      leaves no block behind on D. The push is held open after 5 of 10
    //      chunks until D holds them, then abandoned.
    // KIND: fault
    // CATCHES: m01, m05, m12
    // CHAPTER: L10.6 section 5, Pitfalls
    use tl_engine::kv_transfer::tonic::codegen::tokio_stream::{self, StreamExt};
    let p_pool = shared(32);
    let d_pool = shared(32);
    let recv = Arc::new(KvReceiver::new(Arc::clone(&d_pool)));
    let srv = kt::serve_kv(listener(), Arc::clone(&recv)).unwrap();
    let prompt: Vec<u32> = (0..40).collect();
    let (out, _) = prefill(&p_pool, &prompt, &SamplingParams::greedy(), 0);
    let chunks = kt::build_chunks(&p_pool, "half", &out.block_ids, &out.block_hashes, &vec![false; out.block_hashes.len()]).unwrap();
    let first: Vec<KvChunk> = chunks[..5].to_vec();
    let r = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let addr = srv.addr;
    let task = r.spawn(async move {
        let mut c = KvTransferServiceClient::connect(format!("http://{addr}")).await.unwrap();
        c.push_kv(tokio_stream::iter(first).chain(tokio_stream::pending())).await
    });
    eventually("D to hold the first 5 blocks", || used(&d_pool) == 5);
    task.abort();
    eventually("D to free them after the stream is cancelled", || used(&d_pool) == 0);
    assert_eq!(recv.pending(), 0);
    release_all(&p_pool, &out.block_ids);
}

/// A transport that lies in HasBlocks (says every block is cached) and
/// hands pushes straight to an in-process receiver.
struct Lying {
    recv: Arc<KvReceiver>,
    pushes: AtomicUsize,
}

#[tonic::async_trait]
impl KvTransport for Lying {
    async fn has_blocks(&self, _t: &str, hashes: &[u64], _f: u32) -> Result<Vec<bool>, TransferError> {
        Ok(vec![true; hashes.len()])
    }
    async fn push(&self, _t: &str, chunks: Vec<KvChunk>) -> Result<KvAck, TransferError> {
        self.pushes.fetch_add(1, Ordering::SeqCst);
        self.recv.accept_all(chunks.into_iter().map(Ok)).map_err(TransferError::from)
    }
}

#[test]
fn evicted_block_is_resent() {
    // WHY: between HasBlocks and PushKv a cached block on D can be evicted;
    //      D answers NOT_FOUND and P resends the whole handle with every
    //      payload, once. Here HasBlocks claims blocks D never had.
    // KIND: fault
    // CATCHES: s09, s11, m13
    // CHAPTER: L10.6 section 4
    let p_pool = shared(16);
    let d_pool = shared(16);
    let recv = Arc::new(KvReceiver::new(Arc::clone(&d_pool)));
    let t = Lying { recv: Arc::clone(&recv), pushes: AtomicUsize::new(0) };
    let (out, _) = prefill(&p_pool, &(0..9).collect::<Vec<u32>>(), &SamplingParams::greedy(), 0);
    let stats = rt().block_on(kt::transfer(&t, "d", &p_pool, "h", &out.block_ids, &out.block_hashes)).unwrap();
    assert_eq!(t.pushes.load(Ordering::SeqCst), 2, "one refused push, one full resend");
    assert_eq!((stats.sent, stats.deduped), (3, 0));
    assert_eq!(recv.claim("h").map(|v| v.len()), Some(3));
}

#[test]
fn prefill_rpc_validation_and_info() {
    // WHY: a decode-only engine refuses Prefill (FAILED_PRECONDITION: the
    //      gateway routed wrongly); a request without prompt or target, with
    //      an out-of-range temperature, or with max_new_tokens 0 is
    //      INVALID_ARGUMENT before any work; Info reports roles, KV format 1,
    //      and the formats it reads; Cancel and Drain reach the engine.
    // KIND: boundary
    // CATCHES: s13, m14
    // CHAPTER: L10.6 section 4
    let dec = pair(&["decode"]);
    let r = rt();
    let target = dec.kv_addr.to_string();
    let e = r.block_on(call_prefill(dec.control_addr, prefill_request("x", &[1, 2], &target, 0.0, 0, 2))).unwrap_err();
    assert_eq!(e.code(), tonic::Code::FailedPrecondition);
    let pre = pair(&["prefill"]);
    let t2 = pre.kv_addr.to_string();
    for (req, what) in [
        (prefill_request("x", &[], &t2, 0.0, 0, 2), "empty prompt"),
        (prefill_request("x", &[1], "", 0.0, 0, 2), "empty target"),
        (prefill_request("x", &[1], &t2, -1.0, 0, 2), "temperature -1"),
        (prefill_request("x", &[1], &t2, 0.0, 0, 0), "max_new_tokens 0"),
    ] {
        let e = r.block_on(call_prefill(pre.control_addr, req)).unwrap_err();
        assert_eq!(e.code(), tonic::Code::InvalidArgument, "{what}: {}", e.message());
    }
    assert_eq!(used(&pre.p_pool), 0, "a refused request did no work");
    r.block_on(async {
        let mut c = EngineControlClient::connect(format!("http://{}", pre.control_addr)).await.unwrap();
        let i = c.info(pb::InfoRequest {}).await.unwrap().into_inner();
        assert_eq!((i.roles.clone(), i.kv_format, i.kv_formats_read.clone(), i.block_size), (vec!["prefill".to_string()], 1, vec![1], B as i32));
        assert!(c.cancel(pb::CancelRequest { request_id: "known".into() }).await.unwrap().into_inner().found);
        assert!(!c.cancel(pb::CancelRequest { request_id: "other".into() }).await.unwrap().into_inner().found);
        assert_eq!(c.drain(pb::DrainRequest { deadline_ms: 3000 }).await.unwrap().into_inner().aborted, 3);
    });
}

// ---------------------------------------------------------------------------
// heartbeat

fn ident(role: &str) -> WorkerIdentity {
    WorkerIdentity { worker_id: "w1".into(), http_address: "127.0.0.1:8000".into(), grpc_address: "127.0.0.1:50051".into(), kv_address: "127.0.0.1:50052".into(), role: role.into(), model: "smol".into() }
}

#[test]
fn worker_status_fields() {
    // WHY: the registry routes KV to kv_address, so only a decode worker may
    //      advertise one; every status carries the engine's KV format (1)
    //      and its live load.
    // KIND: unit
    // CATCHES: m15
    // CHAPTER: L10.6 section 4
    let load = Load { queue_depth: 3, running: 2, kv_free_blocks: 10, kv_total_blocks: 64, draining: true };
    let d = heartbeat::worker_status(&ident("decode"), &load);
    assert_eq!((d.kv_address.as_str(), d.role.as_str(), d.kv_format, d.queue_depth, d.running, d.kv_free_blocks, d.kv_total_blocks, d.draining), ("127.0.0.1:50052", "decode", 1, 3, 2, 10, 64, true));
    for role in ["prefill", "unified"] {
        assert_eq!(heartbeat::worker_status(&ident(role), &load).kv_address, "", "{role} must not advertise a KV address");
    }
}

/// A registry that records statuses, can fail or hang, and asks to drain
/// on demand.
#[derive(Default)]
struct FakeRegistry {
    seen: Mutex<Vec<WorkerStatus>>,
    fail: AtomicBool,
    hang: AtomicBool,
    drain: AtomicBool,
}

#[tonic::async_trait]
impl Registry for FakeRegistry {
    async fn heartbeat(&self, s: WorkerStatus) -> Result<HeartbeatAck, String> {
        if self.hang.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
        if self.fail.load(Ordering::SeqCst) {
            return Err("registry down".into());
        }
        self.seen.lock().unwrap().push(s);
        Ok(HeartbeatAck { drain: self.drain.load(Ordering::SeqCst), route_epoch: 7 })
    }
}

#[test]
fn beat_once_counts_failures_and_drain() {
    // WHY: a registry outage must never stop the engine's beats, and a slow
    //      registry must not block one past its timeout; consecutive
    //      failures are counted (3 misses get a worker evicted) and reset by
    //      a success; a drain request is remembered.
    // KIND: fault
    // CATCHES: s14, m16
    // CHAPTER: L10.6 section 5, Pitfalls
    let reg = FakeRegistry::default();
    let st = BeatState::default();
    let r = rt();
    let status = || heartbeat::worker_status(&ident("decode"), &Load::default());
    assert!(r.block_on(heartbeat::beat_once(&reg, status(), Duration::from_secs(1), &st)));
    reg.fail.store(true, Ordering::SeqCst);
    for _ in 0..2 {
        assert!(!r.block_on(heartbeat::beat_once(&reg, status(), Duration::from_secs(1), &st)));
    }
    assert_eq!((st.sent.load(Ordering::SeqCst), st.failed.load(Ordering::SeqCst), st.consecutive_failures.load(Ordering::SeqCst)), (3, 2, 2));
    reg.fail.store(false, Ordering::SeqCst);
    reg.hang.store(true, Ordering::SeqCst);
    let t0 = Instant::now();
    assert!(!r.block_on(heartbeat::beat_once(&reg, status(), Duration::from_millis(100), &st)));
    assert!(t0.elapsed() < Duration::from_secs(2), "a hanging registry blocked the beat for {:?}", t0.elapsed());
    reg.hang.store(false, Ordering::SeqCst);
    reg.drain.store(true, Ordering::SeqCst);
    assert!(r.block_on(heartbeat::beat_once(&reg, status(), Duration::from_secs(1), &st)));
    assert_eq!(st.consecutive_failures.load(Ordering::SeqCst), 0);
    assert!(st.drain_requested());
    assert_eq!(st.route_epoch.load(Ordering::SeqCst), 7);
    reg.drain.store(false, Ordering::SeqCst);
    r.block_on(heartbeat::beat_once(&reg, status(), Duration::from_secs(1), &st));
    assert!(st.drain_requested(), "a drain request is not forgotten on the next ack");
}

#[test]
fn heartbeat_beats_on_schedule() {
    // WHY: the background beat: one at once, then every interval, each with
    //      the load read at that moment (not the load at start), and stop()
    //      ends it promptly.
    // KIND: unit
    // CATCHES: m17
    // CHAPTER: L10.6 section 4
    let reg = Arc::new(FakeRegistry::default());
    let q = Arc::new(AtomicI32::new(0));
    let q2 = Arc::clone(&q);
    let load: Arc<dyn Fn() -> Load + Send + Sync> = Arc::new(move || Load { queue_depth: q2.fetch_add(1, Ordering::SeqCst), ..Default::default() });
    let hb = Heartbeat::spawn(reg.clone(), ident("unified"), load, Duration::from_millis(20), Duration::from_secs(1)).unwrap();
    eventually("4 beats", || reg.seen.lock().unwrap().len() >= 4);
    let t0 = Instant::now();
    let st = Arc::clone(&hb.state);
    hb.stop();
    assert!(t0.elapsed() < Duration::from_secs(1));
    let seen = reg.seen.lock().unwrap().clone();
    let depths: Vec<i32> = seen.iter().map(|s| s.queue_depth).collect();
    assert!(depths.windows(2).all(|w| w[1] > w[0]), "load must be read per beat: {depths:?}");
    let n = seen.len();
    thread::sleep(Duration::from_millis(80));
    assert_eq!(reg.seen.lock().unwrap().len(), n, "beats continued after stop");
    assert_eq!(st.sent.load(Ordering::SeqCst) as usize, n);
}

/// The gateway side, served for the gRPC client test.
struct CountingRegistry {
    beats: AtomicU64,
}

#[tonic::async_trait]
impl WorkerRegistry for CountingRegistry {
    async fn heartbeat(&self, r: tonic::Request<WorkerStatus>) -> Result<tonic::Response<HeartbeatAck>, Status> {
        let n = self.beats.fetch_add(1, Ordering::SeqCst) + 1;
        let s = r.into_inner();
        if s.worker_id.is_empty() {
            return Err(Status::invalid_argument("no worker id"));
        }
        Ok(tonic::Response::new(HeartbeatAck { drain: false, route_epoch: n }))
    }
}

#[test]
fn grpc_registry_roundtrip() {
    // WHY: the production Registry speaks tl.control.v1 over gRPC: an ack
    //      comes back from a real server, and an unreachable registry is an
    //      error within the timeout, never a hang.
    // KIND: conformance
    // CATCHES: m18
    // CHAPTER: L10.6 section 4
    let svc = Arc::new(CountingRegistry { beats: AtomicU64::new(0) });
    let srv = kt::spawn_router(listener(), tonic::transport::Server::builder().add_service(WorkerRegistryServer::from_arc(Arc::clone(&svc)))).unwrap();
    let reg = GrpcRegistry { target: srv.addr.to_string(), timeout: Duration::from_secs(5) };
    let r = rt();
    let a = r.block_on(reg.heartbeat(heartbeat::worker_status(&ident("decode"), &Load::default()))).unwrap();
    let b = r.block_on(reg.heartbeat(heartbeat::worker_status(&ident("decode"), &Load::default()))).unwrap();
    assert_eq!((a.route_epoch, b.route_epoch), (1, 2));
    let dead = listener().local_addr().unwrap();
    let down = GrpcRegistry { target: dead.to_string(), timeout: Duration::from_millis(500) };
    let t0 = Instant::now();
    assert!(r.block_on(down.heartbeat(WorkerStatus::default())).is_err());
    assert!(t0.elapsed() < Duration::from_secs(3));
    srv.shutdown();
}

/// A tiny JSON reader for the fixtures, independent of the code under test.
mod j {
    #[derive(Debug, Clone, PartialEq)]
    pub enum V {
        Null,
        Bool(bool),
        Num(String),
        Str(String),
        Arr(Vec<V>),
        Obj(Vec<(String, V)>),
    }

    impl V {
        pub fn get(&self, k: &str) -> &V {
            match self {
                V::Obj(kv) => kv.iter().find(|(a, _)| a == k).map(|(_, v)| v).unwrap_or_else(|| panic!("no key {k:?}")),
                _ => panic!("not an object"),
            }
        }
        pub fn str(&self) -> &str {
            match self {
                V::Str(s) => s,
                _ => panic!("not a string: {self:?}"),
            }
        }
        pub fn u64(&self) -> u64 {
            match self {
                V::Num(n) => n.parse().unwrap_or_else(|_| panic!("not an unsigned integer: {n}")),
                _ => panic!("not a number: {self:?}"),
            }
        }
        pub fn arr(&self) -> &Vec<V> {
            match self {
                V::Arr(a) => a,
                _ => panic!("not an array: {self:?}"),
            }
        }
    }

    pub fn parse(s: &str) -> V {
        let b = s.as_bytes();
        let mut i = 0;
        let v = val(b, &mut i);
        v
    }

    fn ws(b: &[u8], i: &mut usize) {
        while *i < b.len() && b[*i].is_ascii_whitespace() {
            *i += 1;
        }
    }

    fn val(b: &[u8], i: &mut usize) -> V {
        ws(b, i);
        match b[*i] {
            b'{' => {
                *i += 1;
                let mut kv = Vec::new();
                loop {
                    ws(b, i);
                    match b[*i] {
                        b'}' => {
                            *i += 1;
                            return V::Obj(kv);
                        }
                        b',' => *i += 1,
                        _ => {
                            let V::Str(k) = val(b, i) else { panic!("key") };
                            ws(b, i);
                            *i += 1;
                            kv.push((k, val(b, i)));
                        }
                    }
                }
            }
            b'[' => {
                *i += 1;
                let mut a = Vec::new();
                loop {
                    ws(b, i);
                    match b[*i] {
                        b']' => {
                            *i += 1;
                            return V::Arr(a);
                        }
                        b',' => *i += 1,
                        _ => a.push(val(b, i)),
                    }
                }
            }
            b'"' => {
                *i += 1;
                let st = *i;
                while b[*i] != b'"' {
                    *i += 1;
                }
                *i += 1;
                V::Str(String::from_utf8(b[st..*i - 1].to_vec()).unwrap())
            }
            b't' => {
                *i += 4;
                V::Bool(true)
            }
            b'f' => {
                *i += 5;
                V::Bool(false)
            }
            b'n' => {
                *i += 4;
                V::Null
            }
            _ => {
                let st = *i;
                while *i < b.len() && matches!(b[*i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                    *i += 1;
                }
                V::Num(std::str::from_utf8(&b[st..*i]).unwrap().to_string())
            }
        }
    }
}
