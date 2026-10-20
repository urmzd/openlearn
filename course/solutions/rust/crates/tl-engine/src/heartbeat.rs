//! The engine's heartbeat to the gateway's worker registry (L10.6,
//! `tl.control.v1.WorkerRegistry`).
//!
//! Every 2 s an engine reports who it is (worker id, its three addresses,
//! role, model, KV format) and how busy it is (queue depth, running
//! sequences, free and total KV blocks, draining). The gateway routes on
//! these numbers and evicts a worker after 3 missed beats
//! (`[gateway].heartbeat_miss_limit`), so a beat must never block the engine
//! and a registry outage must never stop the beats: each call is bounded by
//! a timeout, failures are counted, and the next tick tries again. An ack
//! with `drain = true` (the admin API's model drain) is remembered until the
//! engine has drained.
//!
//! Chapter: content/courses/ai-engineering/23-model-serving/07-disaggregated-prefill-decode.md.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tl_proto::tl::control::v1::worker_registry_client::WorkerRegistryClient;
use tl_proto::tl::control::v1::{HeartbeatAck, WorkerStatus};

use crate::kv_transfer::KV_FORMAT;

/// The beat interval of the contract (control.proto).
pub const INTERVAL: Duration = Duration::from_secs(2);

/// Where beats go: the gRPC registry in production, a fake in tests.
#[tonic::async_trait]
pub trait Registry: Send + Sync {
    async fn heartbeat(&self, status: WorkerStatus) -> Result<HeartbeatAck, String>;
}

/// `tl.control.v1.WorkerRegistry` over gRPC at `target` (host:port).
pub struct GrpcRegistry {
    pub target: String,
    pub timeout: Duration,
}

#[tonic::async_trait]
impl Registry for GrpcRegistry {
    async fn heartbeat(&self, status: WorkerStatus) -> Result<HeartbeatAck, String> {
        let uri = if self.target.contains("://") { self.target.clone() } else { format!("http://{}", self.target) };
        let ep = tonic::transport::Endpoint::from_shared(uri).map_err(|e| e.to_string())?;
        let ch = ep.connect_timeout(self.timeout).timeout(self.timeout).connect().await.map_err(|e| e.to_string())?;
        let mut c = WorkerRegistryClient::new(ch);
        c.heartbeat(status).await.map(|r| r.into_inner()).map_err(|s| format!("{:?}: {}", s.code(), s.message()))
    }
}

/// The fixed part of a worker's status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerIdentity {
    pub worker_id: String,
    pub http_address: String,
    pub grpc_address: String,
    /// "" unless the role is decode (only decode workers receive KV).
    pub kv_address: String,
    /// unified | prefill | decode
    pub role: String,
    pub model: String,
}

/// The changing part, read from the engine at every beat.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Load {
    pub queue_depth: i32,
    pub running: i32,
    pub kv_free_blocks: i32,
    pub kv_total_blocks: i32,
    pub draining: bool,
}

/// One WorkerStatus message. Only a decode worker advertises its KV
/// address; the KV format is this engine's (1).
pub fn worker_status(id: &WorkerIdentity, load: &Load) -> WorkerStatus {
    WorkerStatus {
        worker_id: id.worker_id.clone(),
        http_address: id.http_address.clone(),
        grpc_address: id.grpc_address.clone(),
        kv_address: if id.role == "decode" { id.kv_address.clone() } else { String::new() },
        role: id.role.clone(),
        model: id.model.clone(),
        kv_format: KV_FORMAT,
        queue_depth: load.queue_depth,
        running: load.running,
        kv_free_blocks: load.kv_free_blocks,
        kv_total_blocks: load.kv_total_blocks,
        draining: load.draining,
    }
}

/// Counters a beat updates; shared with whoever asks (readiness, metrics).
#[derive(Debug, Default)]
pub struct BeatState {
    pub sent: AtomicU64,
    pub failed: AtomicU64,
    /// Failures since the last success (the gateway evicts at 3 misses).
    pub consecutive_failures: AtomicU64,
    pub drain: AtomicBool,
    pub route_epoch: AtomicU64,
}

impl BeatState {
    pub fn drain_requested(&self) -> bool {
        self.drain.load(Ordering::SeqCst)
    }
}

/// One beat: the current status, one bounded call, the counters updated.
/// Returns whether the registry answered.
pub async fn beat_once(reg: &dyn Registry, status: WorkerStatus, timeout: Duration, st: &BeatState) -> bool {
    st.sent.fetch_add(1, Ordering::SeqCst);
    match tokio::time::timeout(timeout, reg.heartbeat(status)).await {
        Ok(Ok(ack)) => {
            st.consecutive_failures.store(0, Ordering::SeqCst);
            if ack.drain {
                st.drain.store(true, Ordering::SeqCst);
            }
            st.route_epoch.store(ack.route_epoch, Ordering::SeqCst);
            true
        }
        _ => {
            st.failed.fetch_add(1, Ordering::SeqCst);
            st.consecutive_failures.fetch_add(1, Ordering::SeqCst);
            false
        }
    }
}

/// A running heartbeat: a background thread with its own runtime that beats
/// at once and then every `interval`, each call bounded by `timeout`.
pub struct Heartbeat {
    pub state: Arc<BeatState>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Heartbeat {
    /// Starts beating. `load` is read fresh before every beat.
    pub fn spawn(reg: Arc<dyn Registry>, id: WorkerIdentity, load: Arc<dyn Fn() -> Load + Send + Sync>, interval: Duration, timeout: Duration) -> std::io::Result<Heartbeat> {
        let state = Arc::new(BeatState::default());
        let st = Arc::clone(&state);
        let (tx, mut rx) = tokio::sync::oneshot::channel::<()>();
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        let thread = std::thread::spawn(move || {
            rt.block_on(async move {
                let mut tick = tokio::time::interval(interval);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    tokio::select! {
                        _ = &mut rx => break,
                        _ = tick.tick() => {
                            let status = worker_status(&id, &load());
                            tokio::select! {
                                _ = &mut rx => break,
                                _ = beat_once(reg.as_ref(), status, timeout, &st) => {}
                            }
                        }
                    }
                }
            });
        });
        Ok(Heartbeat { state, stop: Some(tx), thread: Some(thread) })
    }

    /// Stops beating and waits for the thread.
    pub fn stop(mut self) {
        self.stop_now();
    }

    fn stop_now(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        self.stop_now();
    }
}
