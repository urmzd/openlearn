//! The engine loop (L10.5): one `step` runs the scheduler (L10.2), builds
//! the mixed batch (L10.3), runs the model (L10.1) over the block manager's
//! blocks (L10.4), and samples one token for every sequence that reached
//! its end. It works on token ids only: text, chat templates, stop strings,
//! and HTTP live in tl-serve.
//!
//! Each request carries its own sampling parameters and its own generator,
//! `stream(seed, sample)` (spec/pcg32.md), so a request's tokens do not
//! depend on what else is in the batch. A preempted request keeps its
//! generator and its tokens; recomputing its KV draws nothing.

use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::block_manager::BlockManager;
use crate::chunk::{plan, Chunked};
use crate::model::ModelConfig;
use crate::runner::{lock, EngineConfig, ModelRunner, PrefixCache};
use crate::sample::{sample, stream, Pcg32, SamplingParams, PURPOSE_SAMPLE};
use crate::sched::{AdmitError, BlockSpace, FinishReason, Request, RequestEvent, RequestId, Scheduler, SchedulerConfig, StepOutput};

/// A generation request in token ids.
#[derive(Clone, Debug, PartialEq)]
pub struct GenRequest {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    pub seed: u64,
    pub max_tokens: usize,
    /// Ids that end generation with `Stop` (EOS); the id itself is reported.
    pub stop_ids: Vec<u32>,
    pub priority: i32,
}

/// One sampled token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenOut {
    pub id: RequestId,
    pub token: u32,
    pub logprob: f64,
    pub finish: Option<FinishReason>,
}

/// Counters for /metrics and readiness.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngineStats {
    pub steps: u64,
    pub waiting: usize,
    pub running: usize,
    pub preemptions: u64,
    pub batch_tokens: usize,
    pub kv_total: u32,
    pub kv_free: u32,
    pub kv_used: u32,
    pub kv_cached: u32,
    pub kv_evictions: u64,
    pub prefix_hit_rate: f64,
}

/// Per-request sampling state.
struct ReqState {
    params: SamplingParams,
    rng: Pcg32,
    stop_ids: Vec<u32>,
}

/// Runner + scheduler + block manager + per-request samplers.
pub struct Engine {
    runner: ModelRunner,
    sched: Scheduler<BlockManager>,
    reqs: HashMap<RequestId, ReqState>,
    max_model_len: usize,
    batch_tokens: usize,
}

impl Engine {
    /// An engine over `runner` configured by `cfg`: the prefix-cache mode,
    /// the token budget, the sequence cap, chunked prefill when
    /// `prefill_chunk > 0`, and the waiting-queue capacity `max_waiting`.
    /// The context is the smaller of the model's and the KV pool's.
    pub fn new(runner: ModelRunner, cfg: &EngineConfig, max_waiting: usize) -> Engine {
        let pool_tokens = cfg.kv.blocks * cfg.kv.block_size;
        let max_model_len = runner.config().max_position_embeddings.min(pool_tokens);
        let scfg = SchedulerConfig {
            max_seqs: cfg.max_seqs,
            max_batch_tokens: cfg.max_batch_tokens,
            max_waiting,
            policy: cfg.policy,
            aging_steps: 64,
            max_model_len,
        };
        let bm = BlockManager::new(runner.pool(), cfg.prefix_cache);
        let mut sched = Scheduler::new(scfg, bm);
        if cfg.prefill_chunk > 0 {
            sched = sched.with_prefill_policy(Box::new(Chunked { chunk: cfg.prefill_chunk }));
        }
        Engine { runner, sched, reqs: HashMap::new(), max_model_len, batch_tokens: 0 }
    }

    /// The model's config.json.
    pub fn model_config(&self) -> &ModelConfig {
        self.runner.config()
    }

    /// Prompt plus generated tokens never exceed this.
    pub fn max_model_len(&self) -> usize {
        self.max_model_len
    }

    /// The runner (embeddings run outside the step loop, between steps).
    pub fn runner_mut(&mut self) -> &mut ModelRunner {
        &mut self.runner
    }

    /// Queues a request; its generator is `stream(seed, sample)`.
    pub fn add_request(&mut self, r: GenRequest) -> Result<RequestId, AdmitError> {
        let id = self.sched.add(Request { prompt: r.prompt, max_tokens: r.max_tokens, priority: r.priority })?;
        self.reqs.insert(id, ReqState { params: r.params, rng: stream(r.seed, PURPOSE_SAMPLE), stop_ids: r.stop_ids });
        Ok(id)
    }

    /// Drops a request at once; its blocks go back to the pool.
    pub fn abort(&mut self, id: RequestId) {
        self.sched.abort(id);
        self.reqs.remove(&id);
    }

    /// Drops every request (a failed step, a shutdown); every block goes back.
    pub fn abort_all(&mut self) {
        for id in self.sched.unfinished() {
            self.sched.abort(id);
        }
        self.reqs.clear();
    }

    /// Anything waiting or running.
    pub fn has_work(&self) -> bool {
        self.sched.has_unfinished()
    }

    /// One step: schedule, forward the mixed batch, sample the rows that
    /// end a sequence, and report each token (with `finish` on the last).
    pub fn step(&mut self) -> Result<Vec<TokenOut>> {
        let out = self.sched.schedule();
        self.batch_tokens = out.num_tokens();
        if out.is_empty() {
            return Ok(Vec::new());
        }
        let mut sampled = Vec::new();
        {
            let p = plan(&self.sched, &out);
            let logits = self.runner.forward(&p.batch)?;
            for &(row, id) in &p.sample {
                let st = self.reqs.get_mut(&id).ok_or_else(|| anyhow!("no sampler for request {id}"))?;
                let toks = self.sched.tokens(id).ok_or_else(|| anyhow!("unknown request {id}"))?;
                let plen = self.sched.prompt_len(id).unwrap_or(toks.len());
                let (token, logprob) = sample(logits.row(row), &st.params, &toks[..plen], &toks[plen..], &mut st.rng);
                sampled.push((id, token, logprob, st.stop_ids.contains(&token)));
            }
        }
        let outputs: Vec<StepOutput> = sampled.iter().map(|&(id, token, _, stop)| StepOutput { id, token, stop }).collect();
        let mut finished: HashMap<RequestId, FinishReason> = HashMap::new();
        for ev in self.sched.on_step(&outputs) {
            if let RequestEvent::Finished { id, reason } = ev {
                finished.insert(id, reason);
                self.reqs.remove(&id);
            }
        }
        Ok(sampled.into_iter().map(|(id, token, logprob, _)| TokenOut { id, token, logprob, finish: finished.get(&id).copied() }).collect())
    }

    /// Counters, read between steps.
    pub fn stats(&self) -> EngineStats {
        let s = self.sched.stats();
        let kv = lock(&self.runner.pool()).map(|p| p.stats()).unwrap_or_default();
        let bm = self.sched.blocks();
        // The radix cache holds a pool reference on each block it keeps, so
        // the pool counts them used; with no request on them they are cached.
        let held = if bm.mode() == PrefixCache::Radix { (bm.cached_blocks() as u32).min(kv.used) } else { 0 };
        EngineStats {
            steps: s.step,
            waiting: s.waiting,
            running: s.running,
            preemptions: s.preemptions,
            batch_tokens: self.batch_tokens,
            kv_total: bm.total_blocks() as u32,
            kv_free: kv.free,
            kv_used: kv.used - held,
            kv_cached: kv.cached + held,
            kv_evictions: bm.stats().evictions,
            prefix_hit_rate: bm.hit_rate(),
        }
    }
}
