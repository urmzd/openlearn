//! Continuous batching (L10.2): which requests run in the next engine step.
//!
//! A step is one forward pass over a batch of sequences. Static batching
//! would run a fixed group of requests until the longest one finishes;
//! continuous (iteration-level) batching instead re-forms the batch at every
//! step: finished requests leave, waiting ones join, so the GPU (here, the
//! CPU kernels) never idles on padding or stragglers.
//!
//! Each step the scheduler, under a token budget (`max_batch_tokens`) and a
//! sequence cap (`max_seqs`):
//!
//! 1. gives every running request in decode one slot, highest rank first;
//!    when the KV pool has no block for a slot it **preempts** the
//!    lowest-ranked running request (frees its blocks, puts it back in the
//!    waiting queue with its generated tokens, to be **recomputed** later),
//!    or the request itself when nothing ranks lower;
//! 2. continues requests whose prefill is not finished (chunked prefill,
//!    L10.3, through [`PrefillPolicy`]);
//! 3. admits waiting requests in rank order while blocks for the whole
//!    sequence can be allocated (admission by free blocks); the first one
//!    that does not fit stops admission, so a big request is not starved by
//!    small ones behind it.
//!
//! Rank: `Fcfs` orders by arrival. `Priority` orders by
//! `priority + (now - arrival) / aging_steps`, which ages a waiting request
//! one priority level per `aging_steps` steps. Because every waiting request
//! ages at the same rate, that order equals the order of the static key
//! `priority * aging_steps - arrival`, so the waiting queue is a heap with
//! keys that never change (tl-ds `LazyHeap`, ds.06; its handles make abort
//! O(1)).
//!
//! The scheduler owns no KV memory: blocks are handed out by a
//! [`BlockSpace`], the free list below or the block manager of L10.4.

use std::cmp::Ordering;
use std::collections::HashMap;

use tl_ds::heap::{Handle, LazyHeap};

use crate::runner::SchedPolicy;

/// A request's id, unique per scheduler.
pub type RequestId = u64;

/// The pool could not hand out the blocks asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoCapacity;

/// What `BlockSpace::allocate` gave a new request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Allocation {
    /// Leading positions whose KV was already cached (prefix cache, L10.4);
    /// prefill starts there. A multiple of the block size.
    pub cached_tokens: usize,
}

/// Where KV blocks come from. The scheduler calls it; L10.4's
/// `BlockManager` implements it over the Rust pool with a prefix cache.
pub trait BlockSpace {
    /// Token positions per block.
    fn block_tokens(&self) -> usize;
    /// Blocks the pool holds in all.
    fn total_blocks(&self) -> usize;
    /// Blocks `allocate` and `append_slot` could take right now.
    fn free_blocks(&self) -> usize;
    /// Blocks for positions `0..tokens.len()` of a new request, all or
    /// nothing; may reuse cached blocks of a matching prefix.
    fn allocate(&mut self, id: RequestId, tokens: &[u32]) -> Result<Allocation, NoCapacity>;
    /// Makes the request's blocks cover `total` positions.
    fn append_slot(&mut self, id: RequestId, total: usize) -> Result<(), NoCapacity>;
    /// Gives back every block of `id`. `computed` are the tokens whose KV
    /// the blocks hold (a prefix cache may keep their full blocks).
    fn release(&mut self, id: RequestId, computed: &[u32]);
    /// The request's block table (empty for an unknown id).
    fn block_table(&self, id: RequestId) -> &[u32];
}

/// The simplest `BlockSpace`: a free list over block ids the owner already
/// holds (for example every block of a KV pool, allocated once). No prefix
/// cache.
#[derive(Clone, Debug)]
pub struct FreeListBlocks {
    block_tokens: usize,
    total: usize,
    free: Vec<u32>,
    tables: HashMap<RequestId, Vec<u32>>,
}

impl FreeListBlocks {
    /// A free list over `ids`, blocks of `block_tokens` positions.
    pub fn new(block_tokens: usize, ids: Vec<u32>) -> FreeListBlocks {
        assert!(block_tokens > 0, "block_tokens must be at least 1");
        let mut free = ids;
        free.reverse(); // pop() hands out the lowest ids first
        FreeListBlocks { block_tokens, total: free.len(), free, tables: HashMap::new() }
    }

    /// Takes blocks until `id` covers `total` positions.
    fn grow(&mut self, id: RequestId, total: usize) -> Result<(), NoCapacity> {
        let have = self.tables.get(&id).map_or(0, Vec::len);
        let need = total.div_ceil(self.block_tokens).saturating_sub(have);
        if need > self.free.len() {
            return Err(NoCapacity);
        }
        let start = self.free.len() - need;
        let mut taken: Vec<u32> = self.free.drain(start..).collect();
        taken.reverse();
        self.tables.entry(id).or_default().extend(taken);
        Ok(())
    }
}

impl BlockSpace for FreeListBlocks {
    fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    fn total_blocks(&self) -> usize {
        self.total
    }

    fn free_blocks(&self) -> usize {
        self.free.len()
    }

    fn allocate(&mut self, id: RequestId, tokens: &[u32]) -> Result<Allocation, NoCapacity> {
        if self.tables.contains_key(&id) {
            return Err(NoCapacity);
        }
        self.grow(id, tokens.len())?;
        Ok(Allocation { cached_tokens: 0 })
    }

    fn append_slot(&mut self, id: RequestId, total: usize) -> Result<(), NoCapacity> {
        self.grow(id, total)
    }

    fn release(&mut self, id: RequestId, _computed: &[u32]) {
        if let Some(t) = self.tables.remove(&id) {
            self.free.extend(t.into_iter().rev());
        }
    }

    fn block_table(&self, id: RequestId) -> &[u32] {
        self.tables.get(&id).map_or(&[], Vec::as_slice)
    }
}

/// How many prompt tokens of one request a step processes.
pub trait PrefillPolicy {
    /// Tokens to prefill now, given `remaining` unprocessed tokens and
    /// `budget` tokens left in this step; `alone` when nothing else is
    /// scheduled yet. 0 means not this step.
    fn chunk_len(&self, remaining: usize, budget: usize, alone: bool) -> usize;
}

/// L10.2's policy: a prompt is prefilled whole, in one step. A prompt
/// larger than the budget runs only in a step of its own.
#[derive(Clone, Copy, Debug, Default)]
pub struct WholePrompt;

impl PrefillPolicy for WholePrompt {
    fn chunk_len(&self, remaining: usize, budget: usize, alone: bool) -> usize {
        if remaining <= budget || alone {
            remaining
        } else {
            0
        }
    }
}

/// The scheduler's limits.
#[derive(Clone, Debug, PartialEq)]
pub struct SchedulerConfig {
    pub max_seqs: usize,
    pub max_batch_tokens: usize,
    /// Waiting requests beyond this are refused (the HTTP layer answers 429).
    pub max_waiting: usize,
    pub policy: SchedPolicy,
    /// Steps per priority level of aging; 0 turns aging off.
    pub aging_steps: u64,
    /// The model's context: prompt plus generated tokens never exceed it.
    pub max_model_len: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            max_seqs: 64,
            max_batch_tokens: 2048,
            max_waiting: 1024,
            policy: SchedPolicy::Fcfs,
            aging_steps: 64,
            max_model_len: 2048,
        }
    }
}

/// A new request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    pub prompt: Vec<u32>,
    /// Tokens to generate at most (>= 1).
    pub max_tokens: usize,
    /// Higher runs first under `Priority` (`X-TL-Priority`, -100 to 100).
    pub priority: i32,
}

/// Why `add` refused a request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmitError {
    /// The waiting queue is at `max_waiting`.
    QueueFull,
    /// An empty prompt or `max_tokens == 0`.
    Empty,
    /// The prompt plus one generated token exceed the context.
    TooLong { tokens: usize, limit: usize },
    /// The prompt plus `max_tokens` (capped at the context) need more blocks
    /// than the whole pool holds.
    NeverFits { blocks: usize, total: usize },
}

/// A run of prefill tokens: positions `start..start + len` of the request's
/// sequence (prompt plus any tokens generated before a preemption).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub id: RequestId,
    pub start: usize,
    pub len: usize,
    /// The chunk reaches the end of the sequence: its last logits are sampled.
    pub last: bool,
}

/// One step's plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScheduleOutput {
    pub prefill: Vec<Chunk>,
    /// Requests that feed their newest token this step.
    pub decode: Vec<RequestId>,
    /// Requests preempted while planning this step (they hold no blocks).
    pub preempted: Vec<RequestId>,
}

impl ScheduleOutput {
    /// Tokens this step processes: every chunk's length plus one per decode.
    pub fn num_tokens(&self) -> usize {
        self.prefill.iter().map(|c| c.len).sum::<usize>() + self.decode.len()
    }

    /// Nothing to run.
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}

/// A sampled token, reported back after the forward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepOutput {
    pub id: RequestId,
    pub token: u32,
    /// The engine saw an end condition (an EOS id, a stop string).
    pub stop: bool,
}

/// Why a request finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// What happened to a request in a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestEvent {
    Token { id: RequestId, token: u32 },
    Finished { id: RequestId, reason: FinishReason },
}

/// Counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedStats {
    pub step: u64,
    pub waiting: usize,
    pub running: usize,
    pub preemptions: u64,
    pub finished: u64,
}

/// The waiting queue's key: higher rank first, then earlier arrival.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WaitKey {
    rank: i64,
    arrival: u64,
    id: RequestId,
}

fn wait_order(a: &WaitKey, b: &WaitKey) -> Ordering {
    b.rank.cmp(&a.rank).then(a.arrival.cmp(&b.arrival))
}

/// Per-request state.
#[derive(Clone, Debug)]
struct Seq {
    tokens: Vec<u32>,
    prompt_len: usize,
    max_tokens: usize,
    priority: i32,
    arrival: u64,
    /// Positions whose K and V are in the request's blocks.
    computed: usize,
    /// `Some` while waiting.
    handle: Option<Handle>,
}

/// The continuous-batching scheduler over a block space `B`.
pub struct Scheduler<B: BlockSpace> {
    cfg: SchedulerConfig,
    blocks: B,
    policy: Box<dyn PrefillPolicy + Send>,
    seqs: HashMap<RequestId, Seq>,
    waiting: LazyHeap<WaitKey, fn(&WaitKey, &WaitKey) -> Ordering>,
    running: Vec<RequestId>,
    next_id: RequestId,
    step: u64,
    last: ScheduleOutput,
    stats: SchedStats,
}

impl<B: BlockSpace> Scheduler<B> {
    /// A scheduler that prefills whole prompts (`WholePrompt`).
    pub fn new(cfg: SchedulerConfig, blocks: B) -> Self {
        Scheduler {
            cfg,
            blocks,
            policy: Box::new(WholePrompt),
            seqs: HashMap::new(),
            waiting: LazyHeap::new(wait_order as fn(&WaitKey, &WaitKey) -> Ordering),
            running: Vec::new(),
            next_id: 1,
            step: 0,
            last: ScheduleOutput::default(),
            stats: SchedStats::default(),
        }
    }

    /// Replaces the prefill policy (L10.3's `Chunked`).
    pub fn with_prefill_policy(mut self, p: Box<dyn PrefillPolicy + Send>) -> Self {
        self.policy = p;
        self
    }

    /// The limits.
    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    /// The block space.
    pub fn blocks(&self) -> &B {
        &self.blocks
    }

    /// The block space, mutably (the engine reads stats through it).
    pub fn blocks_mut(&mut self) -> &mut B {
        &mut self.blocks
    }

    /// The rank of a request under the policy (higher runs first).
    fn rank(&self, priority: i32, arrival: u64) -> i64 {
        match self.cfg.policy {
            SchedPolicy::Fcfs => 0,
            SchedPolicy::Priority if self.cfg.aging_steps == 0 => priority as i64,
            SchedPolicy::Priority => priority as i64 * self.cfg.aging_steps as i64 - arrival as i64,
        }
    }

    /// Queues a request; refuses what can never run or a full queue.
    pub fn add(&mut self, req: Request) -> Result<RequestId, AdmitError> {
        if req.prompt.is_empty() || req.max_tokens == 0 {
            return Err(AdmitError::Empty);
        }
        if req.prompt.len() + 1 > self.cfg.max_model_len {
            return Err(AdmitError::TooLong { tokens: req.prompt.len(), limit: self.cfg.max_model_len });
        }
        // the whole sequence must fit in the pool on its own, or a request
        // preempted near its end could never be admitted again
        let longest = (req.prompt.len() + req.max_tokens).min(self.cfg.max_model_len);
        let need = longest.div_ceil(self.blocks.block_tokens());
        if need > self.blocks.total_blocks() {
            return Err(AdmitError::NeverFits { blocks: need, total: self.blocks.total_blocks() });
        }
        if self.waiting.len() >= self.cfg.max_waiting {
            return Err(AdmitError::QueueFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        let prompt_len = req.prompt.len();
        let seq = Seq {
            tokens: req.prompt,
            prompt_len,
            max_tokens: req.max_tokens,
            priority: req.priority,
            arrival: self.step,
            computed: 0,
            handle: None,
        };
        self.seqs.insert(id, seq);
        self.enqueue(id);
        Ok(id)
    }

    /// Puts a known request into the waiting queue.
    fn enqueue(&mut self, id: RequestId) {
        let (priority, arrival) = {
            let s = &self.seqs[&id];
            (s.priority, s.arrival)
        };
        let key = WaitKey { rank: self.rank(priority, arrival), arrival, id };
        let h = self.waiting.push(key);
        self.seqs.get_mut(&id).expect("known id").handle = Some(h);
    }

    /// Drops a request wherever it is; its blocks go back. Unknown ids
    /// (finished, never added) are ignored.
    pub fn abort(&mut self, id: RequestId) {
        let Some(s) = self.seqs.remove(&id) else { return };
        match s.handle {
            Some(h) => {
                self.waiting.remove(h);
            }
            None => {
                self.running.retain(|&r| r != id);
                self.blocks.release(id, &s.tokens[..s.computed]);
            }
        }
    }

    /// Moves a running request back to waiting: its blocks are freed and its
    /// KV will be recomputed from its tokens (prompt plus generated).
    fn preempt(&mut self, id: RequestId, out: &mut ScheduleOutput) {
        self.running.retain(|&r| r != id);
        let s = self.seqs.get_mut(&id).expect("running id");
        let computed = std::mem::take(&mut s.computed);
        let toks = s.tokens[..computed].to_vec();
        self.blocks.release(id, &toks);
        self.enqueue(id);
        out.preempted.push(id);
        self.stats.preemptions += 1;
    }

    /// Plans the next step (see the module doc for the three phases).
    pub fn schedule(&mut self) -> ScheduleOutput {
        self.step += 1;
        let mut out = ScheduleOutput::default();
        let mut budget = self.cfg.max_batch_tokens;
        // Running requests, highest rank first (the last is the first victim).
        let mut order: Vec<RequestId> = self.running.clone();
        order.sort_by(|a, b| {
            let (sa, sb) = (&self.seqs[a], &self.seqs[b]);
            wait_order(
                &WaitKey { rank: self.rank(sa.priority, sa.arrival), arrival: sa.arrival, id: *a },
                &WaitKey { rank: self.rank(sb.priority, sb.arrival), arrival: sb.arrival, id: *b },
            )
        });
        // 1. decodes
        let mut i = 0;
        while i < order.len() {
            let id = order[i];
            let s = &self.seqs[&id];
            if s.tokens.len() - s.computed != 1 {
                i += 1;
                continue; // still prefilling: phase 2
            }
            if budget == 0 {
                break;
            }
            let total = s.tokens.len();
            loop {
                if self.blocks.append_slot(id, total).is_ok() {
                    out.decode.push(id);
                    budget -= 1;
                    break;
                }
                // no block for the slot: preempt the lowest-ranked request not
                // yet scheduled, or this one
                match order.len() - 1 > i {
                    true => {
                        let victim = order.pop().expect("a lower-ranked request");
                        self.preempt(victim, &mut out);
                    }
                    false => {
                        self.preempt(id, &mut out);
                        order.remove(i);
                        break;
                    }
                }
            }
            if out.decode.last() == Some(&id) {
                i += 1;
            }
        }
        // 2. prefills in progress
        for &id in &order {
            let s = &self.seqs[&id];
            let remaining = s.tokens.len() - s.computed;
            if remaining <= 1 || !self.running.contains(&id) {
                continue;
            }
            let n = self.policy.chunk_len(remaining, budget, out.is_empty());
            if n > 0 {
                out.prefill.push(Chunk { id, start: s.computed, len: n, last: n == remaining });
                budget -= n.min(budget);
            }
        }
        // 3. admissions
        while self.running.len() < self.cfg.max_seqs && budget > 0 {
            let Some(&key) = self.waiting.peek() else { break };
            let id = key.id;
            let tokens = self.seqs[&id].tokens.clone();
            let Ok(a) = self.blocks.allocate(id, &tokens) else { break };
            self.waiting.pop();
            let s = self.seqs.get_mut(&id).expect("waiting id");
            s.handle = None;
            s.computed = a.cached_tokens.min(tokens.len() - 1);
            self.running.push(id);
            let remaining = tokens.len() - s.computed;
            let n = self.policy.chunk_len(remaining, budget, out.is_empty());
            if n == 0 {
                break; // admitted, holds its blocks, prefills next step
            }
            out.prefill.push(Chunk { id, start: s.computed, len: n, last: n == remaining });
            budget -= n.min(budget);
        }
        self.last = out.clone();
        self.stats.step = self.step;
        out
    }

    /// Applies the step's results: advances every scheduled request's
    /// computed positions, appends each sampled token, and finishes
    /// requests that stopped or reached their limit (their blocks go back).
    pub fn on_step(&mut self, outputs: &[StepOutput]) -> Vec<RequestEvent> {
        let last = std::mem::take(&mut self.last);
        for c in &last.prefill {
            if let Some(s) = self.seqs.get_mut(&c.id) {
                s.computed = (c.start + c.len).max(s.computed);
            }
        }
        for id in &last.decode {
            if let Some(s) = self.seqs.get_mut(id) {
                s.computed = s.tokens.len();
            }
        }
        let mut events = Vec::new();
        for o in outputs {
            let Some(s) = self.seqs.get_mut(&o.id) else { continue };
            if s.handle.is_some() || s.computed != s.tokens.len() {
                continue; // not sampled this step (preempted, or mid-prefill)
            }
            s.tokens.push(o.token);
            events.push(RequestEvent::Token { id: o.id, token: o.token });
            let generated = s.tokens.len() - s.prompt_len;
            let reason = if o.stop {
                Some(FinishReason::Stop)
            } else if generated >= s.max_tokens || s.tokens.len() >= self.cfg.max_model_len {
                Some(FinishReason::Length)
            } else {
                None
            };
            if let Some(reason) = reason {
                let s = self.seqs.remove(&o.id).expect("known id");
                self.running.retain(|&r| r != o.id);
                self.blocks.release(o.id, &s.tokens[..s.computed]);
                self.stats.finished += 1;
                events.push(RequestEvent::Finished { id: o.id, reason });
            }
        }
        events
    }

    /// The tokens of a request (prompt plus generated), if known.
    pub fn tokens(&self, id: RequestId) -> Option<&[u32]> {
        self.seqs.get(&id).map(|s| s.tokens.as_slice())
    }

    /// The prompt length of a request, if known.
    pub fn prompt_len(&self, id: RequestId) -> Option<usize> {
        self.seqs.get(&id).map(|s| s.prompt_len)
    }

    /// Positions of a request whose KV is computed.
    pub fn computed(&self, id: RequestId) -> Option<usize> {
        self.seqs.get(&id).map(|s| s.computed)
    }

    /// Running request ids, in admission order.
    pub fn running(&self) -> &[RequestId] {
        &self.running
    }

    /// Every request not finished (waiting or running), in id order.
    pub fn unfinished(&self) -> Vec<RequestId> {
        let mut ids: Vec<RequestId> = self.seqs.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Requests not finished (waiting or running).
    pub fn has_unfinished(&self) -> bool {
        !self.seqs.is_empty()
    }

    /// Counters.
    pub fn stats(&self) -> SchedStats {
        SchedStats { waiting: self.waiting.len(), running: self.running.len(), ..self.stats }
    }
}
