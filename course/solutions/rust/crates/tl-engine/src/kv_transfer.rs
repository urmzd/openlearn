//! KV transfer between engines (L10.6): the export envelope in Rust, the
//! decode side of `tl.kv.v1` (HasBlocks, PushKv, Release), and the prefill
//! side's deduplicated push.
//!
//! Disaggregated serving runs a prompt's prefill on one engine (P) and its
//! decode on another (D). P computes the prompt's KV into blocks of its pool,
//! asks D which full blocks it already holds (by content hash, the prefix
//! cache's chained FNV-1a), and streams only the missing ones, plus the
//! partial tail block, each as a one-block envelope of
//! course/contracts/formats/kv-block.md. D imports them into its pool and
//! keeps the block references under the handle id until the request resumes
//! there ([`KvReceiver::claim`]) or the gateway releases it.
//!
//! Format v1 only: any other `kv_format` is refused with FAILED_PRECONDITION
//! (craft.13 adds v2 by taking this unit over). Every error frees what it
//! took: a failed push leaves D's pool exactly as it was.
//!
//! The generated gRPC code (`tl_proto`), and the `tonic` and `tokio` it runs
//! on, are re-exported so the engine binary and the tests name one version.
//!
//! Chapter: content/courses/build-an-llm/p10-serving/06-disaggregated-prefill-decode.md.

use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

pub use tl_proto;
pub use tokio;
pub use tonic;

use tl_proto::tl::kv::v1::kv_transfer_service_client::KvTransferServiceClient;
use tl_proto::tl::kv::v1::kv_transfer_service_server::{KvTransferService, KvTransferServiceServer};
use tl_proto::tl::kv::v1::{HasBlocksRequest, HasBlocksResponse, KvAck, KvChunk, ReleaseRequest, ReleaseResponse};
use crate::kv::{KvCfg, KvPool, TL_EFULL, TL_ENOMEM};
use tonic::{Request, Response, Status, Streaming};

use crate::sample::{stream, Pcg32, PURPOSE_SAMPLE};

/// The KV format this engine writes and reads.
pub const KV_FORMAT: u32 = 1;
/// tl_dtype of a v1 payload (TL_F16).
pub const DTYPE_F16: u16 = 1;
/// Envelope header bytes.
pub const HEADER_BYTES: usize = 28;

/// A pool shared by the step loop and the transfer tasks. One Mutex protects
/// the pool while multiple async tasks use it.
pub type SharedKv = Arc<Mutex<KvPool>>;

/// Locks the pool, recovering it if a holder panicked (the pool's own state
/// is consistent between calls).
pub fn lock(p: &SharedKv) -> MutexGuard<'_, KvPool> {
    p.lock().unwrap_or_else(|e| e.into_inner())
}

// -- block hashes ---------------------------------------------------------------

/// FNV-1a 64 over bytes.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    h
}

/// The chained hash of one full block: fnv1a64(le_u64(parent) ‖
/// le_u32(tokens)...), 0 replaced by 1.
pub fn block_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut buf = Vec::with_capacity(8 + 4 * tokens.len());
    buf.extend_from_slice(&parent.to_le_bytes());
    for t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    match fnv1a64(&buf) {
        0 => 1,
        h => h,
    }
}

/// The hashes of a prompt's FULL blocks of `block_tokens` (the partial tail
/// has none).
pub fn prompt_hashes(tokens: &[u32], block_tokens: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(tokens.len() / block_tokens);
    let mut parent = 0;
    for chunk in tokens.chunks_exact(block_tokens) {
        parent = block_hash(parent, chunk);
        out.push(parent);
    }
    out
}

/// CRC-32C (Castagnoli, reflected 0x82F63B78, init and final xor all ones).
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0x82F63B78 } else { crc >> 1 };
        }
    }
    !crc
}

// -- the envelope ----------------------------------------------------------------

/// The envelope header's fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnvHeader {
    pub version: u16,
    pub dtype: u16,
    pub n_blocks: u32,
    pub block_tokens: u32,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

impl EnvHeader {
    /// The header of a v1 envelope of `n_blocks` for a pool's shape.
    pub fn v1(cfg: &KvCfg, n_blocks: u32) -> EnvHeader {
        EnvHeader { version: 1, dtype: DTYPE_F16, n_blocks, block_tokens: cfg.block_tokens, n_layers: cfg.n_layers, n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim }
    }

    /// Payload bytes of one v1 block: f16 [L][2][Hkv][B][D].
    pub fn payload_bytes(&self) -> usize {
        self.n_layers as usize * 2 * self.n_kv_heads as usize * self.block_tokens as usize * self.head_dim as usize * 2
    }

    /// Bytes of a whole envelope: 28 + n (12 + P) + 4.
    pub fn envelope_bytes(&self) -> usize {
        HEADER_BYTES + self.n_blocks as usize * (12 + self.payload_bytes()) + 4
    }
}

/// One block record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRecord {
    /// 0 for the partial tail block.
    pub hash: u64,
    pub n_tokens: u32,
    pub payload: Vec<u8>,
}

/// Why an envelope or a chunk was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EnvError {
    /// Bad length, magic, CRC, or record (gRPC DATA_LOSS).
    Format(String),
    /// A version or dtype this reader does not read (FAILED_PRECONDITION).
    Version(String),
    /// Dimensions that differ from the receiving pool's (DATA_LOSS).
    Shape(String),
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EnvError::Format(m) => write!(f, "kv envelope: {m}"),
            EnvError::Version(m) => write!(f, "kv envelope version: {m}"),
            EnvError::Shape(m) => write!(f, "kv envelope shape: {m}"),
        }
    }
}

/// Writes an envelope (formats/kv-block.md): header, records, CRC-32C of
/// everything before it.
pub fn write_envelope(h: &EnvHeader, blocks: &[BlockRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES + blocks.iter().map(|b| 12 + b.payload.len()).sum::<usize>() + 4);
    out.extend_from_slice(b"TLKV");
    out.extend_from_slice(&h.version.to_le_bytes());
    out.extend_from_slice(&h.dtype.to_le_bytes());
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for v in [h.block_tokens, h.n_layers, h.n_kv_heads, h.head_dim] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for b in blocks {
        out.extend_from_slice(&b.hash.to_le_bytes());
        out.extend_from_slice(&b.n_tokens.to_le_bytes());
        out.extend_from_slice(&b.payload);
    }
    let crc = crc32c(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fn u16_at(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Reads an envelope with the reader rules of kv-block.md: length
/// 28 + n * (12 + P) + 4, magic, CRC, version 1 with dtype 1, the receiving pool's
/// dimensions (when `expect` is given), every n_tokens in 1..=B, and a
/// non-zero hash only on a full block.
pub fn read_envelope(buf: &[u8], expect: Option<&KvCfg>) -> Result<(EnvHeader, Vec<BlockRecord>), EnvError> {
    if buf.len() < HEADER_BYTES + 4 {
        return Err(EnvError::Format(format!("{} bytes is shorter than a header and a CRC", buf.len())));
    }
    if &buf[..4] != b"TLKV" {
        return Err(EnvError::Format("bad magic".into()));
    }
    let h = EnvHeader {
        version: u16_at(buf, 4),
        dtype: u16_at(buf, 6),
        n_blocks: u32_at(buf, 8),
        block_tokens: u32_at(buf, 12),
        n_layers: u32_at(buf, 16),
        n_kv_heads: u32_at(buf, 20),
        head_dim: u32_at(buf, 24),
    };
    let want_len = (h.n_blocks as u128) * (12 + h.payload_bytes() as u128) + HEADER_BYTES as u128 + 4;
    if buf.len() as u128 != want_len {
        return Err(EnvError::Format(format!("length {} but the header says {want_len}", buf.len())));
    }
    let body = &buf[..buf.len() - 4];
    if crc32c(body) != u32_at(buf, buf.len() - 4) {
        return Err(EnvError::Format("CRC mismatch".into()));
    }
    if h.version != 1 || h.dtype != DTYPE_F16 {
        return Err(EnvError::Version(format!("version {} dtype {}; this reader reads version 1 (f16)", h.version, h.dtype)));
    }
    if let Some(c) = expect {
        if (h.block_tokens, h.n_layers, h.n_kv_heads, h.head_dim) != (c.block_tokens, c.n_layers, c.n_kv_heads, c.head_dim) {
            return Err(EnvError::Shape(format!(
                "envelope B={} L={} Hkv={} D={}, pool B={} L={} Hkv={} D={}",
                h.block_tokens, h.n_layers, h.n_kv_heads, h.head_dim, c.block_tokens, c.n_layers, c.n_kv_heads, c.head_dim
            )));
        }
    }
    let p = h.payload_bytes();
    let mut blocks = Vec::with_capacity(h.n_blocks as usize);
    let mut o = HEADER_BYTES;
    for i in 0..h.n_blocks {
        let hash = u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
        let n_tokens = u32_at(buf, o + 8);
        if n_tokens == 0 || n_tokens > h.block_tokens {
            return Err(EnvError::Format(format!("block {i}: n_tokens {n_tokens} outside 1..={}", h.block_tokens)));
        }
        if hash != 0 && n_tokens != h.block_tokens {
            return Err(EnvError::Format(format!("block {i}: a hashed block must be full ({n_tokens} of {})", h.block_tokens)));
        }
        blocks.push(BlockRecord { hash, n_tokens, payload: buf[o + 12..o + 12 + p].to_vec() });
        o += 12 + p;
    }
    Ok((h, blocks))
}

/// The RNG hand-off of a disaggregated request (DESIGN 2.7): D seeds the
/// request's generator as P did, `stream(seed, sample)`, and advances it by
/// the `rng_draws_consumed` uniform draws P used for the first token, so a
/// seeded disaggregated stream equals the unified one.
pub fn resume_rng(seed: u64, draws: u32) -> Pcg32 {
    let mut r = stream(seed, PURPOSE_SAMPLE);
    for _ in 0..draws {
        r.uniform_f64();
    }
    r
}

// -- decode side ---------------------------------------------------------------------

fn status_of(e: &EnvError) -> Status {
    match e {
        EnvError::Version(m) => Status::failed_precondition(m.clone()),
        other => Status::data_loss(other.to_string()),
    }
}

/// The decode side of tl.kv.v1 over one pool: blocks pushed for a handle
/// are held (one reference each) until the request resumes ([`claim`]) or
/// the handle is released.
///
/// [`claim`]: KvReceiver::claim
pub struct KvReceiver {
    pool: SharedKv,
    handles: Mutex<HashMap<String, Vec<u32>>>,
}

impl KvReceiver {
    pub fn new(pool: SharedKv) -> KvReceiver {
        KvReceiver { pool, handles: Mutex::new(HashMap::new()) }
    }

    /// The pool the blocks land in.
    pub fn pool(&self) -> SharedKv {
        Arc::clone(&self.pool)
    }

    /// Handles holding blocks.
    pub fn pending(&self) -> usize {
        self.handles.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// For each hash, whether a full block with it is cached here. Looking
    /// takes no reference: a hit's reference is dropped at once.
    pub fn has_blocks(&self, hashes: &[u64], kv_format: u32) -> Result<Vec<bool>, Status> {
        if kv_format != KV_FORMAT {
            return Err(Status::failed_precondition(format!("kv_format {kv_format}: this engine reads format {KV_FORMAT}")));
        }
        let mut pool = lock(&self.pool);
        let mut out = Vec::with_capacity(hashes.len());
        for &h in hashes {
            match pool.lookup(h) {
                Some(id) => {
                    pool.release(id).map_err(|e| Status::internal(e.to_string()))?;
                    out.push(true);
                }
                None => out.push(false),
            }
        }
        Ok(out)
    }

    /// Takes one chunk's block into `taken` (with a reference): a deduped
    /// chunk (empty payload) looks its hash up; a payload chunk is checked
    /// (format, CRC, envelope of exactly one block with this pool's shape,
    /// its hash) and imported, and a full block is registered under its
    /// hash. Returns whether the chunk was deduped.
    pub fn accept_chunk(&self, c: &KvChunk, expect_index: u32, taken: &mut Vec<u32>) -> Result<bool, Status> {
        if c.kv_format != KV_FORMAT {
            return Err(Status::failed_precondition(format!("kv_format {}: this engine reads format {KV_FORMAT}", c.kv_format)));
        }
        if c.block_index != expect_index {
            return Err(Status::invalid_argument(format!("block_index {} where {expect_index} was due", c.block_index)));
        }
        let mut pool = lock(&self.pool);
        if c.payload.is_empty() {
            if c.block_hash == 0 {
                return Err(Status::invalid_argument("an empty payload needs the block's hash"));
            }
            let id = pool.lookup(c.block_hash).ok_or_else(|| Status::not_found(format!("block {:#018x} is no longer cached", c.block_hash)))?;
            taken.push(id);
            return Ok(true);
        }
        if crc32c(&c.payload) != c.crc32c {
            return Err(Status::data_loss(format!("block {}: crc32c mismatch", c.block_index)));
        }
        let cfg = pool.cfg();
        let (_, recs) = read_envelope(&c.payload, Some(&cfg)).map_err(|e| status_of(&e))?;
        if recs.len() != 1 {
            return Err(Status::data_loss(format!("a chunk carries one block, got {}", recs.len())));
        }
        // A full block P never registered travels with hash 0 in its
        // envelope; the chunk's hash names it. A registered one must agree.
        if recs[0].hash != 0 && recs[0].hash != c.block_hash {
            return Err(Status::data_loss("the chunk's block_hash differs from the envelope's"));
        }
        if c.block_hash != 0 && recs[0].n_tokens != cfg.block_tokens {
            return Err(Status::data_loss("a hashed chunk must carry a full block"));
        }
        let ids = pool.import(&c.payload).map_err(|e| {
            if e.status == TL_EFULL || e.status == TL_ENOMEM {
                Status::resource_exhausted(e.to_string())
            } else {
                Status::data_loss(e.to_string())
            }
        })?;
        let id = ids[0];
        taken.push(id);
        if c.block_hash != 0 {
            pool.register(id, c.block_hash).map_err(|e| Status::internal(e.to_string()))?;
        }
        Ok(false)
    }

    /// Drops one reference on each block (the error path and Release).
    fn release_ids(&self, ids: &[u32]) {
        let mut pool = lock(&self.pool);
        for &id in ids {
            let _ = pool.release(id);
        }
    }

    /// Records the blocks of a completed push under `handle_id` (a repeated
    /// handle id first releases what the old one held).
    fn store(&self, handle_id: &str, ids: Vec<u32>) {
        let old = self.handles.lock().unwrap_or_else(|e| e.into_inner()).insert(handle_id.to_string(), ids);
        if let Some(old) = old {
            self.release_ids(&old);
        }
    }

    /// The resumed request takes the handle's block references (in prompt
    /// order); None for an unknown handle.
    pub fn claim(&self, handle_id: &str) -> Option<Vec<u32>> {
        self.handles.lock().unwrap_or_else(|e| e.into_inner()).remove(handle_id)
    }

    /// Frees the blocks held for a handle; an unknown handle is fine.
    pub fn release(&self, handle_id: &str) {
        if let Some(ids) = self.claim(handle_id) {
            self.release_ids(&ids);
        }
    }

    /// A whole push from an iterator of chunks (the gRPC stream adapts to
    /// this): every chunk accepted in order, then the handle stored. On any
    /// error every block taken for this push is released.
    pub fn accept_all(&self, chunks: impl IntoIterator<Item = Result<KvChunk, Status>>) -> Result<KvAck, Status> {
        let mut taken = Vec::new();
        let mut ack = KvAck::default();
        let mut total: Option<u32> = None;
        let res = (|| {
            for (i, c) in chunks.into_iter().enumerate() {
                let c = c?;
                if ack.handle_id.is_empty() {
                    ack.handle_id = c.handle_id.clone();
                } else if c.handle_id != ack.handle_id {
                    return Err(Status::invalid_argument("one stream carries one handle"));
                }
                match total {
                    None => total = Some(c.n_blocks_total),
                    Some(t) if t != c.n_blocks_total => return Err(Status::invalid_argument("n_blocks_total changed mid-stream")),
                    _ => {}
                }
                if self.accept_chunk(&c, i as u32, &mut taken)? {
                    ack.blocks_deduped += 1;
                } else {
                    ack.blocks_received += 1;
                }
            }
            if taken.len() as u32 != total.unwrap_or(0) || taken.is_empty() {
                return Err(Status::invalid_argument(format!("stream ended after {} of {} blocks", taken.len(), total.unwrap_or(0))));
            }
            Ok(())
        })();
        match res {
            Ok(()) => {
                self.store(&ack.handle_id, taken);
                Ok(ack)
            }
            Err(s) => {
                self.release_ids(&taken);
                Err(s)
            }
        }
    }
}

/// The gRPC face of a [`KvReceiver`].
pub struct KvService(pub Arc<KvReceiver>);

#[tonic::async_trait]
impl KvTransferService for KvService {
    async fn has_blocks(&self, request: Request<HasBlocksRequest>) -> Result<Response<HasBlocksResponse>, Status> {
        let r = request.into_inner();
        Ok(Response::new(HasBlocksResponse { present: self.0.has_blocks(&r.block_hashes, r.kv_format)? }))
    }

    async fn push_kv(&self, request: Request<Streaming<KvChunk>>) -> Result<Response<KvAck>, Status> {
        let mut s = request.into_inner();
        // Chunks are accepted as they arrive; a broken stream (the peer gone
        // mid-transfer) is an error that frees everything taken so far.
        let mut taken: Vec<u32> = Vec::new();
        let mut ack = KvAck::default();
        let mut total = 0u32;
        let mut i = 0u32;
        let res: Result<(), Status> = async {
            while let Some(c) = s.message().await? {
                if i == 0 {
                    ack.handle_id = c.handle_id.clone();
                    total = c.n_blocks_total;
                } else if c.handle_id != ack.handle_id || c.n_blocks_total != total {
                    return Err(Status::invalid_argument("one stream carries one handle and one n_blocks_total"));
                }
                if self.0.accept_chunk(&c, i, &mut taken)? {
                    ack.blocks_deduped += 1;
                } else {
                    ack.blocks_received += 1;
                }
                i += 1;
            }
            if taken.len() as u32 != total || taken.is_empty() {
                return Err(Status::invalid_argument(format!("stream ended after {} of {total} blocks", taken.len())));
            }
            Ok(())
        }
        .await;
        match res {
            Ok(()) => {
                self.0.store(&ack.handle_id, taken);
                Ok(Response::new(ack))
            }
            Err(st) => {
                self.0.release_ids(&taken);
                Err(st)
            }
        }
    }

    async fn release(&self, request: Request<ReleaseRequest>) -> Result<Response<ReleaseResponse>, Status> {
        self.0.release(&request.into_inner().handle_id);
        Ok(Response::new(ReleaseResponse {}))
    }
}

// -- serving and calling -------------------------------------------------------------

/// A gRPC server running on its own thread and runtime; dropped or
/// [`shutdown`](GrpcServer::shutdown), it stops accepting and joins.
pub struct GrpcServer {
    pub addr: SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl GrpcServer {
    pub fn shutdown(mut self) {
        self.stop_now();
    }

    fn stop_now(&mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for GrpcServer {
    fn drop(&mut self) {
        self.stop_now();
    }
}

/// How long a stopping server waits for open connections to finish.
pub const SHUTDOWN_GRACE: Duration = Duration::from_millis(500);

/// Serves a tonic router on an already bound listener (ports are allocated
/// by binding 127.0.0.1:0 first), on a background thread with its own
/// runtime. Messages are capped at 4 MiB by the services themselves.
pub fn spawn_router(listener: std::net::TcpListener, router: tonic::transport::server::Router) -> std::io::Result<GrpcServer> {
    let addr = listener.local_addr()?;
    listener.set_nonblocking(true)?;
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<std::io::Result<()>>();
    let thread = std::thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build() {
            Ok(rt) => rt,
            Err(e) => {
                let _ = ready_tx.send(Err(e));
                return;
            }
        };
        rt.block_on(async move {
            let l = match tokio::net::TcpListener::from_std(listener) {
                Ok(l) => l,
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            let incoming = tonic::transport::server::TcpIncoming::from(l);
            // Graceful shutdown waits for open connections; a client that
            // never closes its connection must not hold the engine hostage,
            // so the grace period is bounded and the rest is cancelled.
            let (grace_tx, grace_rx) = tokio::sync::oneshot::channel::<()>();
            let serve = router.serve_with_incoming_shutdown(incoming, async {
                let _ = grace_rx.await;
            });
            tokio::pin!(serve);
            tokio::select! {
                _ = &mut serve => {}
                _ = rx => {
                    let _ = grace_tx.send(());
                    let _ = tokio::time::timeout(SHUTDOWN_GRACE, &mut serve).await;
                }
            }
        });
    });
    ready_rx.recv().map_err(|_| std::io::Error::other("gRPC server thread died"))??;
    Ok(GrpcServer { addr, stop: Some(tx), thread: Some(thread) })
}

/// Serves `tl.kv.v1` for a receiver.
pub fn serve_kv(listener: std::net::TcpListener, recv: Arc<KvReceiver>) -> std::io::Result<GrpcServer> {
    let svc = KvTransferServiceServer::new(KvService(recv))
        .max_decoding_message_size(tl_proto::MAX_MESSAGE_BYTES)
        .max_encoding_message_size(tl_proto::MAX_MESSAGE_BYTES);
    spawn_router(listener, tonic::transport::Server::builder().add_service(svc))
}

/// Why a push failed, from the prefill side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransferError {
    pub code: tonic::Code,
    pub message: String,
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kv transfer: {:?}: {}", self.code, self.message)
    }
}

impl From<Status> for TransferError {
    fn from(s: Status) -> Self {
        TransferError { code: s.code(), message: s.message().to_string() }
    }
}

/// The two calls the prefill side makes (DESIGN 4.3 `KvTransport`): the
/// gRPC client in production, an in-process fake in tests.
#[tonic::async_trait]
pub trait KvTransport: Send + Sync {
    async fn has_blocks(&self, target: &str, hashes: &[u64], fmt: u32) -> Result<Vec<bool>, TransferError>;
    async fn push(&self, target: &str, chunks: Vec<KvChunk>) -> Result<KvAck, TransferError>;
}

/// tl.kv.v1 over gRPC; one connection per call keeps it simple (a pool of
/// channels is "Going further"). `timeout` bounds each call.
pub struct GrpcTransport {
    pub timeout: Duration,
}

impl GrpcTransport {
    async fn client(&self, target: &str) -> Result<KvTransferServiceClient<tonic::transport::Channel>, TransferError> {
        let uri = if target.contains("://") { target.to_string() } else { format!("http://{target}") };
        let ep = tonic::transport::Endpoint::from_shared(uri).map_err(|e| TransferError { code: tonic::Code::InvalidArgument, message: e.to_string() })?;
        let ch = ep.connect_timeout(self.timeout).timeout(self.timeout).connect().await.map_err(|e| TransferError { code: tonic::Code::Unavailable, message: e.to_string() })?;
        Ok(KvTransferServiceClient::new(ch).max_decoding_message_size(tl_proto::MAX_MESSAGE_BYTES).max_encoding_message_size(tl_proto::MAX_MESSAGE_BYTES))
    }
}

#[tonic::async_trait]
impl KvTransport for GrpcTransport {
    async fn has_blocks(&self, target: &str, hashes: &[u64], fmt: u32) -> Result<Vec<bool>, TransferError> {
        let mut c = self.client(target).await?;
        let r = c.has_blocks(HasBlocksRequest { block_hashes: hashes.to_vec(), kv_format: fmt }).await?;
        Ok(r.into_inner().present)
    }

    async fn push(&self, target: &str, chunks: Vec<KvChunk>) -> Result<KvAck, TransferError> {
        let mut c = self.client(target).await?;
        Ok(c.push_kv(tonic::codegen::tokio_stream::iter(chunks)).await?.into_inner())
    }
}

/// What one transfer moved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TransferStats {
    pub chunks: u32,
    /// Chunks that carried a payload (missing full blocks and the tail).
    pub sent: u32,
    pub deduped: u32,
    /// Payload bytes on the wire.
    pub payload_bytes: u64,
}

/// The chunks of one handle: for block i (prompt order) a deduped chunk
/// (empty payload, hash set, crc 0) when `present[i]`, else the block's
/// one-block envelope from the pool with its CRC-32C. `hashes` covers the
/// full blocks only; blocks past it (the partial tail) have hash 0 and are
/// always sent.
pub fn build_chunks(pool: &SharedKv, handle_id: &str, ids: &[u32], hashes: &[u64], present: &[bool]) -> Result<Vec<KvChunk>, TransferError> {
    let pool = lock(pool);
    let mut out = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        let hash = hashes.get(i).copied().unwrap_or(0);
        let dedup = hash != 0 && present.get(i).copied().unwrap_or(false);
        let payload = if dedup {
            Vec::new()
        } else {
            pool.export(&[id]).map_err(|e| TransferError { code: tonic::Code::Internal, message: e.to_string() })?
        };
        let crc = if payload.is_empty() { 0 } else { crc32c(&payload) };
        out.push(KvChunk {
            handle_id: handle_id.to_string(),
            block_hash: hash,
            block_index: i as u32,
            n_blocks_total: ids.len() as u32,
            kv_format: KV_FORMAT,
            payload,
            crc32c: crc,
        });
    }
    Ok(out)
}

/// The prefill side of a transfer: HasBlocks for the prompt's full-block
/// hashes, then one PushKv of every block (deduped where present). When D
/// answers NOT_FOUND (a cached block was evicted between the two calls) the
/// push is retried once with every payload. The caller still owns `ids` and
/// releases them after (success or not).
pub async fn transfer(t: &dyn KvTransport, target: &str, pool: &SharedKv, handle_id: &str, ids: &[u32], hashes: &[u64]) -> Result<TransferStats, TransferError> {
    let present = t.has_blocks(target, hashes, KV_FORMAT).await?;
    if present.len() != hashes.len() {
        return Err(TransferError { code: tonic::Code::Internal, message: format!("HasBlocks answered {} of {}", present.len(), hashes.len()) });
    }
    let stats = |chunks: &[KvChunk], ack: &KvAck| TransferStats {
        chunks: chunks.len() as u32,
        sent: ack.blocks_received,
        deduped: ack.blocks_deduped,
        payload_bytes: chunks.iter().map(|c| c.payload.len() as u64).sum(),
    };
    let chunks = build_chunks(pool, handle_id, ids, hashes, &present)?;
    match t.push(target, chunks.clone()).await {
        Ok(ack) => Ok(stats(&chunks, &ack)),
        Err(e) if e.code == tonic::Code::NotFound => {
            let all = build_chunks(pool, handle_id, ids, hashes, &vec![false; hashes.len()])?;
            let ack = t.push(target, all.clone()).await?;
            Ok(stats(&all, &ack))
        }
        Err(e) => Err(e),
    }
}
