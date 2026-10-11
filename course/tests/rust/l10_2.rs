//! L10.2 course tests: the continuous-batching scheduler (tl-engine sched.rs).
//!
//! Annotated exemplars (DESIGN 5.12). Most tests drive your `Scheduler` with
//! a fake model: the next token is a fixed function of the sequence so far
//! (`fake_next`), so every request's output is known in advance and the
//! scheduler alone decides what runs when. The step counter is the clock.
//! Two tests run the real model (L10.1's runner over the tiny Llama of
//! course/fixtures/L7.9/tiny-llama-2l) to show that batching and
//! preemption leave greedy outputs unchanged. Random workloads come from the
//! frozen PCG32 in `mod frozen`, never from yours.

use std::collections::HashMap;
use std::path::PathBuf;

use tl_engine::forward::{ForwardBatch, ForwardSeq};
use tl_engine::runner::{lock, EngineConfig, KvConfig, ModelRunner, SchedPolicy};
use tl_engine::sample::{self, Pcg32, SamplingParams};
use tl_engine::sched::{
    AdmitError, BlockSpace, Chunk, FinishReason, FreeListBlocks, Request, RequestEvent, RequestId, ScheduleOutput, Scheduler, SchedulerConfig, StepOutput,
};

// ---------------------------------------------------------------------------
// helpers

mod frozen {
    pub struct Pcg32 {
        state: u64,
        inc: u64,
    }
    impl Pcg32 {
        pub fn new(seed: u64) -> Pcg32 {
            let mut g = Pcg32 { state: 0, inc: (54 << 1) | 1 };
            g.next_u32();
            g.state = g.state.wrapping_add(seed);
            g.next_u32();
            g
        }
        pub fn next_u32(&mut self) -> u32 {
            let old = self.state;
            self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
            let xs = (((old >> 18) ^ old) >> 27) as u32;
            xs.rotate_right((old >> 59) as u32)
        }
        pub fn below(&mut self, n: u32) -> u32 {
            self.next_u32() % n
        }
    }
}

fn ol_seed() -> u64 {
    std::env::var("OL_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

/// The fake model: the next token is a function of the sequence so far.
fn fake_next(tokens: &[u32]) -> u32 {
    ((*tokens.last().unwrap() as usize * 7 + tokens.len()) % 50) as u32
}

/// What a request should generate, unrolled without any scheduler.
fn fake_expected(prompt: &[u32], n: usize) -> Vec<u32> {
    let mut t = prompt.to_vec();
    let mut out = Vec::new();
    for _ in 0..n {
        let x = fake_next(&t);
        t.push(x);
        out.push(x);
    }
    out
}

fn cfg(max_seqs: usize, max_batch_tokens: usize) -> SchedulerConfig {
    SchedulerConfig { max_seqs, max_batch_tokens, max_waiting: 1000, policy: SchedPolicy::Fcfs, aging_steps: 0, max_model_len: 4096 }
}

fn free_list(block_tokens: usize, blocks: u32) -> FreeListBlocks {
    FreeListBlocks::new(block_tokens, (0..blocks).collect())
}

fn req(prompt: &[u32], max_tokens: usize, priority: i32) -> Request {
    Request { prompt: prompt.to_vec(), max_tokens, priority }
}

/// Rows of a step that are sampled: every decode, and each chunk that ends
/// its sequence.
fn sampled(out: &ScheduleOutput) -> Vec<RequestId> {
    let mut v = out.decode.clone();
    v.extend(out.prefill.iter().filter(|c| c.last).map(|c| c.id));
    v
}

#[derive(Default)]
struct Run {
    generated: HashMap<RequestId, Vec<u32>>,
    finished: HashMap<RequestId, FinishReason>,
    first_token_step: HashMap<RequestId, usize>,
    steps: Vec<ScheduleOutput>,
}

/// Runs the scheduler with the fake model until nothing is left after step
/// `quiet` (or `limit` steps). `arrivals(step)` adds requests before each
/// step.
fn run_fake<B: BlockSpace>(s: &mut Scheduler<B>, quiet: usize, limit: usize, mut arrivals: impl FnMut(usize, &mut Scheduler<B>)) -> Run {
    let mut r = Run::default();
    for step in 0..limit {
        arrivals(step, s);
        let out = s.schedule();
        if out.is_empty() && !s.has_unfinished() && step >= quiet {
            break;
        }
        let outputs: Vec<StepOutput> =
            sampled(&out).into_iter().map(|id| StepOutput { id, token: fake_next(s.tokens(id).unwrap()), stop: false }).collect();
        for ev in s.on_step(&outputs) {
            match ev {
                RequestEvent::Token { id, token } => {
                    r.first_token_step.entry(id).or_insert(step);
                    r.generated.entry(id).or_default().push(token)
                }
                RequestEvent::Finished { id, reason } => {
                    r.finished.insert(id, reason);
                }
            }
        }
        r.steps.push(out);
    }
    r
}

fn tiny_llama() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol")).join("L7.9/tiny-llama-2l")
}

/// The real model with every block of its pool handed to a free list.
fn real(blocks: usize, block_tokens: usize) -> (ModelRunner, FreeListBlocks) {
    let r = ModelRunner::load(&tiny_llama(), &EngineConfig { kv: KvConfig { blocks, block_size: block_tokens }, ..EngineConfig::default() }).unwrap();
    let ids = lock(&r.pool()).unwrap().alloc(blocks).unwrap();
    (r, FreeListBlocks::new(block_tokens, ids))
}

/// Greedy generation for each prompt through the scheduler and the real
/// model, all requests added at step 0.
fn run_real(r: &mut ModelRunner, s: &mut Scheduler<FreeListBlocks>, prompts: &[Vec<u32>], n: usize) -> Vec<Vec<u32>> {
    let ids: Vec<RequestId> = prompts.iter().map(|p| s.add(req(p, n, 0)).unwrap()).collect();
    let mut gen: HashMap<RequestId, Vec<u32>> = HashMap::new();
    for _ in 0..10_000 {
        let out = s.schedule();
        if out.is_empty() && !s.has_unfinished() {
            break;
        }
        let mut seqs = Vec::new();
        let mut rows = Vec::new();
        for &id in &out.decode {
            let t = s.tokens(id).unwrap();
            rows.push((seqs.len(), id));
            seqs.push(ForwardSeq { tokens: &t[t.len() - 1..], start: t.len() - 1, blocks: s.blocks().block_table(id) });
        }
        for c in &out.prefill {
            let t = s.tokens(c.id).unwrap();
            if c.last {
                rows.push((seqs.len(), c.id));
            }
            seqs.push(ForwardSeq { tokens: &t[c.start..c.start + c.len], start: c.start, blocks: s.blocks().block_table(c.id) });
        }
        let logits = r.forward(&ForwardBatch { seqs }).unwrap();
        let outputs: Vec<StepOutput> = rows
            .into_iter()
            .map(|(row, id)| {
                let (t, _) = sample::sample(logits.row(row), &SamplingParams::greedy(), &[], &[], &mut Pcg32::new(0, 54));
                StepOutput { id, token: t, stop: false }
            })
            .collect();
        for ev in s.on_step(&outputs) {
            if let RequestEvent::Token { id, token } = ev {
                gen.entry(id).or_default().push(token);
            }
        }
    }
    ids.iter().map(|id| gen.remove(id).unwrap_or_default()).collect()
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

// ---------------------------------------------------------------------------
// tests

#[test]
fn hand_example_schedule() {
    // WHY: the chapter's worked example, step by step: blocks of 4, budget 8,
    //      2 sequences at most. Step 1 admits A (5 tokens, blocks 0 and 1)
    //      and B (3 tokens, block 2) and fills the budget; step 2 decodes
    //      both and B finishes; step 3 decodes A and admits C into the block
    //      B gave back.
    // KIND: unit
    // CATCHES: s05, s09, s10
    // CHAPTER: L10.2 section 3
    let mut s = Scheduler::new(cfg(2, 8), free_list(4, 6));
    let a = s.add(req(&[1, 2, 3, 4, 5], 3, 0)).unwrap();
    let b = s.add(req(&[6, 7, 8], 2, 0)).unwrap();
    let c = s.add(req(&[9, 10], 1, 0)).unwrap();
    let o1 = s.schedule();
    assert_eq!(o1.prefill, [Chunk { id: a, start: 0, len: 5, last: true }, Chunk { id: b, start: 0, len: 3, last: true }]);
    assert!(o1.decode.is_empty());
    assert_eq!(s.blocks().block_table(a), [0, 1]);
    assert_eq!(s.blocks().block_table(b), [2]);
    s.on_step(&[StepOutput { id: a, token: 40, stop: false }, StepOutput { id: b, token: 41, stop: false }]);
    let o2 = s.schedule();
    assert_eq!(o2.decode, [a, b]);
    assert!(o2.prefill.is_empty(), "C waits: two sequences already run");
    let ev = s.on_step(&[StepOutput { id: a, token: 42, stop: false }, StepOutput { id: b, token: 43, stop: false }]);
    assert!(ev.contains(&RequestEvent::Finished { id: b, reason: FinishReason::Length }));
    assert_eq!(s.blocks().free_blocks(), 4, "B's block went back");
    let o3 = s.schedule();
    assert_eq!(o3.decode, [a]);
    assert_eq!(o3.prefill, [Chunk { id: c, start: 0, len: 2, last: true }]);
    assert_eq!(s.blocks().block_table(c), [2]);
    let ev = s.on_step(&[StepOutput { id: a, token: 44, stop: false }, StepOutput { id: c, token: 45, stop: false }]);
    assert!(ev.contains(&RequestEvent::Finished { id: a, reason: FinishReason::Length }));
    assert!(ev.contains(&RequestEvent::Finished { id: c, reason: FinishReason::Length }));
    assert!(s.schedule().is_empty());
    assert_eq!(s.blocks().free_blocks(), 6);
}

#[test]
fn free_list_is_all_or_nothing() {
    // WHY: the block space hands out the lowest ids first and never a
    //      partial allocation: a request that does not fit takes nothing.
    // KIND: unit
    // CATCHES: s10
    // CHAPTER: L10.2 section 2
    let mut f = free_list(4, 3);
    assert_eq!(f.allocate(1, &[0; 9]).unwrap().cached_tokens, 0);
    assert_eq!(f.block_table(1), [0, 1, 2]);
    assert!(f.allocate(2, &[0; 1]).is_err());
    assert_eq!(f.free_blocks(), 0);
    f.release(1, &[]);
    assert!(f.allocate(2, &[0; 13]).is_err(), "13 tokens need 4 blocks");
    assert_eq!(f.free_blocks(), 3);
    f.allocate(2, &[0; 4]).unwrap();
    f.append_slot(2, 5).unwrap();
    assert_eq!(f.block_table(2), [0, 1]);
}

#[test]
fn every_request_finishes_and_no_block_leaks() {
    // WHY: under random arrivals and lengths, with a pool small enough to
    //      force preemptions, every request finishes with exactly its
    //      max_tokens tokens, each output equals the request run alone, each
    //      step stays within max_seqs and the token budget (a prompt may
    //      exceed it only in a step of its own), and every block is free at
    //      the end.
    // KIND: property
    // CATCHES: s03, s04, s05, s09
    // CHAPTER: L10.2 section 2
    let mut g = frozen::Pcg32::new(ol_seed() + 101);
    let mut s = Scheduler::new(cfg(4, 24), free_list(4, 12));
    let mut want: HashMap<RequestId, Vec<u32>> = HashMap::new();
    let run = run_fake(&mut s, 60, 5000, |step, s| {
        if step < 60 && g.below(3) == 0 {
            let len = 1 + g.below(14) as usize;
            let prompt: Vec<u32> = (0..len).map(|_| g.below(50)).collect();
            let n = 1 + g.below(12) as usize;
            let id = s.add(req(&prompt, n, 0)).unwrap();
            want.insert(id, fake_expected(&prompt, n));
        }
    });
    assert!(!want.is_empty());
    for (id, w) in &want {
        assert_eq!(run.generated.get(id), Some(w), "request {id}");
        assert_eq!(run.finished.get(id), Some(&FinishReason::Length));
    }
    for out in &run.steps {
        let ids: Vec<RequestId> = out.decode.iter().copied().chain(out.prefill.iter().map(|c| c.id)).collect();
        assert!(ids.len() <= 4, "max_seqs");
        assert!(out.num_tokens() <= 24 || ids.len() == 1, "budget exceeded with others in the step: {out:?}");
    }
    assert!(s.stats().preemptions > 0, "the pool was meant to force preemption");
    assert_eq!(s.blocks().free_blocks(), 12, "every block back");
    assert!(!s.has_unfinished());
}

#[test]
fn priority_runs_first() {
    // WHY: under Priority, the waiting request with the higher priority is
    //      admitted first whatever its arrival; under Fcfs, arrival order.
    // KIND: unit
    // CATCHES: s01, s05
    // CHAPTER: L10.2 section 2
    for (policy, first) in [(SchedPolicy::Priority, 2usize), (SchedPolicy::Fcfs, 0usize)] {
        let mut s = Scheduler::new(SchedulerConfig { policy, aging_steps: 10, ..cfg(1, 64) }, free_list(4, 16));
        let ids = [s.add(req(&[1, 2], 1, 0)).unwrap(), s.add(req(&[3, 4], 1, 5)).unwrap(), s.add(req(&[5, 6], 1, 50)).unwrap()];
        let out = s.schedule();
        assert_eq!(out.prefill.len(), 1);
        assert_eq!(out.prefill[0].id, ids[first], "{policy:?}");
    }
}

#[test]
fn aging_bounds_the_wait() {
    // WHY: a steady stream of priority-10 requests must not starve one
    //      priority-0 request forever: with aging_steps = 4, it gains a level
    //      every 4 steps, so it overtakes newer priority-10 arrivals within
    //      about 10 x 4 steps; with aging off it waits out the whole stream.
    // KIND: property
    // CATCHES: s01, s02, s05, s09
    // CHAPTER: L10.2 section 2
    let wait = |aging: u64| {
        let mut s = Scheduler::new(SchedulerConfig { policy: SchedPolicy::Priority, aging_steps: aging, ..cfg(1, 64) }, free_list(4, 64));
        let mut low = 0;
        let run = run_fake(&mut s, 200, 400, |step, s| {
            if step == 0 {
                s.add(req(&[1], 1, 10)).unwrap();
                low = s.add(req(&[2], 1, 0)).unwrap();
            }
            if step < 200 {
                s.add(req(&[3], 1, 10)).unwrap();
            }
        });
        *run.first_token_step.get(&low).unwrap_or(&usize::MAX)
    };
    let aged = wait(4);
    assert!(aged <= 50, "with aging the low-priority request waited {aged} steps");
    let starved = wait(0);
    assert!(starved >= 200, "without aging it should wait for the whole stream, waited {starved}");
}

#[test]
fn preemption_recomputes_and_keeps_outputs() {
    // WHY: when the pool runs out mid-decode, the lowest-ranked running
    //      request (the newest at equal rank) loses its blocks and is
    //      recomputed later from prompt plus generated tokens; with the real
    //      model its greedy output is the same as without memory pressure.
    // KIND: fault, differential
    // CATCHES: s03, s04
    // CHAPTER: L10.2 section 5, Pitfalls
    let prompts = vec![bytes("Once upon a time"), bytes("The cat sat on the"), bytes("In the beginning")];
    let (mut r1, f1) = real(64, 4);
    let mut roomy = Scheduler::new(cfg(8, 256), f1);
    let want = run_real(&mut r1, &mut roomy, &prompts, 12);
    assert_eq!(roomy.stats().preemptions, 0);
    // 14 blocks of 4 = 56 positions; the three need 28 + 30 + 28
    let (mut r2, f2) = real(14, 4);
    let mut tight = Scheduler::new(cfg(8, 256), f2);
    let got = run_real(&mut r2, &mut tight, &prompts, 12);
    assert!(tight.stats().preemptions > 0, "14 blocks must force a preemption");
    assert_eq!(got, want);
    assert_eq!(tight.blocks().free_blocks(), 14);
}

#[test]
fn batched_greedy_equals_single_request() {
    // WHY: four requests decoded together in one continuous batch produce
    //      the same greedy tokens as each request alone, because every
    //      kernel is batch-invariant (c/ABI.md rule 10) and positions,
    //      blocks, and logits rows never mix between sequences.
    // KIND: differential
    // CHAPTER: L10.2 section 2
    let prompts = vec![bytes("Once upon a time"), bytes("The cat"), bytes("Hello, world"), bytes("In the beginning there was")];
    let (mut r, f) = real(64, 16);
    let mut s = Scheduler::new(cfg(8, 512), f);
    let together = run_real(&mut r, &mut s, &prompts, 16);
    for (p, t) in prompts.iter().zip(&together) {
        let (mut r1, f1) = real(64, 16);
        let mut alone = Scheduler::new(cfg(1, 512), f1);
        assert_eq!(&run_real(&mut r1, &mut alone, std::slice::from_ref(p), 16)[0], t);
    }
}

#[test]
fn abort_frees_blocks_and_drops_waiting() {
    // WHY: an aborted request (client gone, L10.5) gives its blocks back at
    //      once whether it was running or waiting, and never produces
    //      another token.
    // KIND: unit
    // CATCHES: s05, s06
    // CHAPTER: L10.2 section 4
    let mut s = Scheduler::new(cfg(1, 64), free_list(4, 8));
    let a = s.add(req(&[1, 2, 3, 4, 5], 10, 0)).unwrap();
    let b = s.add(req(&[6], 10, 0)).unwrap();
    s.schedule();
    assert_eq!(s.blocks().free_blocks(), 6);
    s.abort(a);
    s.abort(b);
    s.abort(999);
    assert_eq!(s.blocks().free_blocks(), 8);
    assert!(s.on_step(&[StepOutput { id: a, token: 1, stop: false }]).is_empty());
    assert!(s.schedule().is_empty());
    assert!(!s.has_unfinished());
}

#[test]
fn decodes_count_against_the_budget() {
    // WHY: every decode is one token of the step's budget: with three
    //      requests decoding and a budget of 8, a 6-token prompt does not fit
    //      (3 + 6 > 8) and waits, holding its blocks, for the next step.
    // KIND: boundary
    // CATCHES: s08
    // CHAPTER: L10.2 section 2
    let mut s = Scheduler::new(cfg(4, 8), free_list(4, 16));
    let ids: Vec<RequestId> = (0..3).map(|i| s.add(req(&[i], 5, 0)).unwrap()).collect();
    s.schedule();
    s.on_step(&ids.iter().map(|&id| StepOutput { id, token: 1, stop: false }).collect::<Vec<_>>());
    s.add(req(&[9; 6], 1, 0)).unwrap();
    let out = s.schedule();
    assert_eq!(out.decode.len(), 3);
    assert!(out.prefill.is_empty(), "the prompt does not fit beside 3 decodes: {out:?}");
    assert_eq!(out.num_tokens(), 3);
}

#[test]
fn stop_finishes_with_stop() {
    // WHY: when the engine reports an end condition (EOS, a stop string),
    //      the request finishes with reason Stop at once, before max_tokens.
    // KIND: unit
    // CHAPTER: L10.2 section 4
    let mut s = Scheduler::new(cfg(1, 64), free_list(4, 8));
    let a = s.add(req(&[1, 2], 10, 0)).unwrap();
    s.schedule();
    let ev = s.on_step(&[StepOutput { id: a, token: 7, stop: true }]);
    assert_eq!(ev, [RequestEvent::Token { id: a, token: 7 }, RequestEvent::Finished { id: a, reason: FinishReason::Stop }]);
    assert_eq!(s.blocks().free_blocks(), 8);
}

#[test]
fn admission_refuses_what_can_never_run() {
    // WHY: requests that could never complete are refused at add, not left
    //      to block the queue: an empty prompt, a prompt past the context,
    //      a sequence larger than the whole pool, and a full waiting queue
    //      (which the HTTP layer turns into 429).
    // KIND: boundary
    // CATCHES: s07
    // CHAPTER: L10.2 section 5, Pitfalls
    let mut s = Scheduler::new(SchedulerConfig { max_waiting: 2, max_model_len: 32, ..cfg(1, 64) }, free_list(4, 4));
    assert_eq!(s.add(req(&[], 1, 0)), Err(AdmitError::Empty));
    assert_eq!(s.add(req(&[1], 0, 0)), Err(AdmitError::Empty));
    assert!(matches!(s.add(req(&[1; 32], 1, 0)), Err(AdmitError::TooLong { .. })));
    assert!(matches!(s.add(req(&[1; 10], 10, 0)), Err(AdmitError::NeverFits { blocks: 5, total: 4 })));
    s.add(req(&[1], 1, 0)).unwrap();
    s.add(req(&[1], 1, 0)).unwrap();
    assert_eq!(s.add(req(&[1], 1, 0)), Err(AdmitError::QueueFull));
}
