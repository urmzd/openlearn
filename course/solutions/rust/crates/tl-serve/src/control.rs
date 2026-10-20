//! `tl.engine.v1.EngineControl` (L10.6): the gRPC face the gateway uses to
//! run a disaggregated request's prefill, and to ask an engine what it is,
//! cancel a request, or drain.
//!
//! `Prefill` on a prefill engine (P): validate, run the prompt through the
//! engine (the [`PrefillBackend`]: prefill plus the first token, sampled with
//! the request's seeded generator), push the prompt's KV blocks to the
//! decode worker named in `decode_target` (tl_engine::kv_transfer, deduped
//! by hash), drop P's own references to them, and answer with the
//! `KvHandle` the decode worker resumes from: the first token, the prompt
//! length, the full-block hashes, and how many uniform draws P took (the RNG
//! hand-off).
//!
//! Chapter: content/courses/ai-engineering/23-model-serving/07-disaggregated-prefill-decode.md.

use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tl_engine::kv_transfer::tl_proto::tl::engine::v1 as pb;
use tl_engine::kv_transfer::tl_proto::tl::engine::v1::engine_control_server::{EngineControl, EngineControlServer};
use tl_engine::kv_transfer::tonic::{self, Request, Response, Status};
use tl_engine::kv_transfer::{self, lock, GrpcServer, KvTransport, SharedKv, TransferError, KV_FORMAT};
use tl_engine::sample::SamplingParams;

/// One prefill as the engine sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillJob {
    pub request_id: String,
    pub model: String,
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub seed: u64,
    pub max_new_tokens: usize,
    pub priority: i32,
}

/// What the engine's prefill produced.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillOutcome {
    pub first_token: u32,
    pub first_logprob: f32,
    /// The prompt's KV blocks in prompt order (full blocks, then the partial
    /// tail if any), each holding one reference owned by this outcome.
    pub block_ids: Vec<u32>,
    /// The chained hashes of the full blocks (formats/kv-block.md).
    pub block_hashes: Vec<u64>,
    /// Prompt tokens served from the prefix cache.
    pub cached_prompt_tokens: u32,
    /// Uniform draws used for first_token: 1 when sampled, 0 when greedy.
    pub rng_draws: u32,
}

/// The engine behind EngineControl: the step loop in production, a fake
/// model in tests.
#[tonic::async_trait]
pub trait PrefillBackend: Send + Sync + 'static {
    async fn prefill(&self, job: PrefillJob) -> Result<PrefillOutcome, Status>;
    /// The pool holding the outcome's blocks.
    fn pool(&self) -> SharedKv;
    fn info(&self) -> pb::InfoResponse;
    async fn cancel(&self, request_id: &str) -> bool;
    /// Stops admitting work, lets in-flight requests run up to
    /// `deadline_ms`, and returns how many it aborted at the deadline.
    async fn drain(&self, deadline_ms: i64) -> i32;
}

/// Engine sampling params from the proto: the ranges of L10.1's
/// `SamplingParams::validate` (INVALID_ARGUMENT otherwise), the seed, and
/// max_new_tokens (at least 1: it includes the first token).
pub fn sampling_from_proto(p: Option<&pb::SamplingParams>) -> Result<(SamplingParams, u64, usize), Status> {
    let d = pb::SamplingParams::default();
    let p = p.unwrap_or(&d);
    let s = SamplingParams {
        temperature: p.temperature as f64,
        top_k: usize::try_from(p.top_k).map_err(|_| Status::invalid_argument(format!("top_k must be >= 0, got {}", p.top_k)))?,
        top_p: if p.top_p == 0.0 { 1.0 } else { p.top_p as f64 },
        min_p: p.min_p as f64,
        repetition_penalty: if p.repetition_penalty == 0.0 { 1.0 } else { p.repetition_penalty as f64 },
        presence_penalty: p.presence_penalty as f64,
        frequency_penalty: p.frequency_penalty as f64,
    };
    s.validate().map_err(Status::invalid_argument)?;
    if p.max_new_tokens < 1 {
        return Err(Status::invalid_argument(format!("max_new_tokens must be >= 1 (it includes the first token), got {}", p.max_new_tokens)));
    }
    Ok((s, p.seed, p.max_new_tokens as usize))
}

/// The status a failed transfer becomes: a decode worker that cannot be
/// reached is UNAVAILABLE (the gateway picks another); the decode side's
/// own refusals keep their code.
pub fn transfer_status(e: &TransferError) -> Status {
    match e.code {
        tonic::Code::Unknown | tonic::Code::Cancelled | tonic::Code::DeadlineExceeded => Status::unavailable(format!("kv transfer: {}", e.message)),
        c => Status::new(c, format!("kv transfer: {}", e.message)),
    }
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// EngineControl over a backend and a KV transport.
pub struct ControlService<B: PrefillBackend> {
    pub backend: Arc<B>,
    pub transport: Arc<dyn KvTransport>,
    /// The roles this engine serves (InfoResponse.roles); Prefill needs
    /// "prefill".
    pub roles: Vec<String>,
}

impl<B: PrefillBackend> ControlService<B> {
    /// Drops the outcome's references on the prefill side.
    fn release(&self, ids: &[u32]) {
        let pool = self.backend.pool();
        let mut p = lock(&pool);
        for &id in ids {
            let _ = p.release(id);
        }
    }

    /// The Prefill RPC without the gRPC wrapper.
    pub async fn run_prefill(&self, r: pb::PrefillRequest) -> Result<pb::PrefillResponse, Status> {
        if !self.roles.iter().any(|x| x == "prefill") {
            return Err(Status::failed_precondition(format!("this engine serves {:?}, not prefill", self.roles)));
        }
        if r.prompt_ids.is_empty() {
            return Err(Status::invalid_argument("prompt_ids is empty"));
        }
        if r.decode_target.is_empty() {
            return Err(Status::invalid_argument("decode_target is empty"));
        }
        if r.deadline_unix_ms > 0 && r.deadline_unix_ms < now_ms() {
            return Err(Status::deadline_exceeded("the request's deadline has passed"));
        }
        let (params, seed, max_new) = sampling_from_proto(r.sampling.as_ref())?;
        let t0 = Instant::now();
        let job = PrefillJob { request_id: r.request_id.clone(), model: r.model.clone(), prompt: r.prompt_ids.clone(), params, seed, max_new_tokens: max_new, priority: r.priority };
        let out = self.backend.prefill(job).await?;
        let ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
        let handle_id = if r.request_id.is_empty() { format!("h{seed:x}") } else { r.request_id.clone() };
        let pool = self.backend.pool();
        let res = kv_transfer::transfer(self.transport.as_ref(), &r.decode_target, &pool, &handle_id, &out.block_ids, &out.block_hashes).await;
        // P's references go either way: D holds its own copies now, and a
        // registered full block stays cached here for the next prompt.
        self.release(&out.block_ids);
        res.map_err(|e| transfer_status(&e))?;
        let n = r.prompt_ids.len() as u32;
        let handle = pb::KvHandle {
            handle_id,
            decode_target: r.decode_target.clone(),
            n_tokens: n,
            block_hashes: out.block_hashes.clone(),
            kv_format: KV_FORMAT,
            first_token: out.first_token,
            prompt_tokens: n,
            rng_draws_consumed: out.rng_draws,
        };
        Ok(pb::PrefillResponse {
            handle: Some(handle),
            first_token: out.first_token,
            first_logprob: out.first_logprob,
            prompt_tokens: n as i32,
            cached_prompt_tokens: out.cached_prompt_tokens as i32,
            ttft_ms,
        })
    }
}

#[tonic::async_trait]
impl<B: PrefillBackend> EngineControl for ControlService<B> {
    async fn prefill(&self, request: Request<pb::PrefillRequest>) -> Result<Response<pb::PrefillResponse>, Status> {
        self.run_prefill(request.into_inner()).await.map(Response::new)
    }

    async fn info(&self, _request: Request<pb::InfoRequest>) -> Result<Response<pb::InfoResponse>, Status> {
        let mut i = self.backend.info();
        i.roles = self.roles.clone();
        i.kv_format = KV_FORMAT;
        if i.kv_formats_read.is_empty() {
            i.kv_formats_read = vec![KV_FORMAT];
        }
        Ok(Response::new(i))
    }

    async fn cancel(&self, request: Request<pb::CancelRequest>) -> Result<Response<pb::CancelResponse>, Status> {
        let found = self.backend.cancel(&request.into_inner().request_id).await;
        Ok(Response::new(pb::CancelResponse { found }))
    }

    async fn drain(&self, request: Request<pb::DrainRequest>) -> Result<Response<pb::DrainResponse>, Status> {
        let aborted = self.backend.drain(request.into_inner().deadline_ms).await;
        Ok(Response::new(pb::DrainResponse { aborted }))
    }
}

/// Serves EngineControl on a bound listener (background thread).
pub fn serve_control<B: PrefillBackend>(listener: std::net::TcpListener, svc: ControlService<B>) -> std::io::Result<GrpcServer> {
    let s = EngineControlServer::new(svc)
        .max_decoding_message_size(kv_transfer::tl_proto::MAX_MESSAGE_BYTES)
        .max_encoding_message_size(kv_transfer::tl_proto::MAX_MESSAGE_BYTES);
    kv_transfer::spawn_router(listener, tonic::transport::Server::builder().add_service(s))
}
