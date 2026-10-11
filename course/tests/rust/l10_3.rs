//! L10.3 course tests: chunked prefill and the mixed batch (tl-engine
//! chunk.rs), driven through your L10.2 scheduler.
//!
//! Annotated exemplars (DESIGN 5.12). The fake model (`fake_next`) makes
//! every output known in advance; the real model is L10.1's runner over
//! course/fixtures/L7.9/tiny-llama-2l, used to show that chunking changes
//! when tokens are computed, never what they are. Random workloads come
//! from the frozen PCG32 in `mod frozen`.

use std::collections::HashMap;
use std::path::PathBuf;

use tl_engine::chunk::{plan, split, Chunked};
use tl_engine::forward::{ForwardBatch, ForwardSeq};
use tl_engine::runner::{lock, EngineConfig, KvConfig, ModelRunner, SchedPolicy};
use tl_engine::sample::{self, Pcg32, SamplingParams};
use tl_engine::sched::{
    BlockSpace, Chunk, FreeListBlocks, PrefillPolicy, Request, RequestEvent, RequestId, ScheduleOutput, Scheduler, SchedulerConfig, StepOutput, WholePrompt,
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

fn fake_next(tokens: &[u32]) -> u32 {
    ((*tokens.last().unwrap() as usize * 7 + tokens.len()) % 50) as u32
}

fn fake_expected(prompt: &[u32], n: usize) -> Vec<u32> {
    let mut t = prompt.to_vec();
    (0..n)
        .map(|_| {
            let x = fake_next(&t);
            t.push(x);
            x
        })
        .collect()
}

fn cfg(max_seqs: usize, max_batch_tokens: usize) -> SchedulerConfig {
    SchedulerConfig { max_seqs, max_batch_tokens, max_waiting: 1000, policy: SchedPolicy::Fcfs, aging_steps: 0, max_model_len: 4096 }
}

fn chunked(c: SchedulerConfig, blocks: FreeListBlocks, chunk: usize) -> Scheduler<FreeListBlocks> {
    Scheduler::new(c, blocks).with_prefill_policy(Box::new(Chunked { chunk }))
}

fn free_list(block_tokens: usize, blocks: u32) -> FreeListBlocks {
    FreeListBlocks::new(block_tokens, (0..blocks).collect())
}

fn req(prompt: &[u32], max_tokens: usize) -> Request {
    Request { prompt: prompt.to_vec(), max_tokens, priority: 0 }
}

#[derive(Default)]
struct Run {
    generated: HashMap<RequestId, Vec<u32>>,
    first_token_step: HashMap<RequestId, usize>,
    steps: Vec<ScheduleOutput>,
}

/// The fake model through the scheduler, sampling the rows `plan` marks.
fn run_fake<B: BlockSpace>(s: &mut Scheduler<B>, quiet: usize, limit: usize, mut arrivals: impl FnMut(usize, &mut Scheduler<B>)) -> Run {
    let mut r = Run::default();
    for step in 0..limit {
        arrivals(step, s);
        let out = s.schedule();
        if out.is_empty() && !s.has_unfinished() && step >= quiet {
            break;
        }
        let p = plan(s, &out);
        let outputs: Vec<StepOutput> = p.sample.iter().map(|&(_, id)| StepOutput { id, token: fake_next(s.tokens(id).unwrap()), stop: false }).collect();
        drop(p);
        for ev in s.on_step(&outputs) {
            if let RequestEvent::Token { id, token } = ev {
                r.first_token_step.entry(id).or_insert(step);
                r.generated.entry(id).or_default().push(token);
            }
        }
        r.steps.push(out);
    }
    r
}

fn tiny_llama() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol")).join("L7.9/tiny-llama-2l")
}

fn real(blocks: usize, block_tokens: usize) -> (ModelRunner, FreeListBlocks) {
    let r = ModelRunner::load(&tiny_llama(), &EngineConfig { kv: KvConfig { blocks, block_size: block_tokens }, ..EngineConfig::default() }).unwrap();
    let ids = lock(&r.pool()).unwrap().alloc(blocks).unwrap();
    (r, FreeListBlocks::new(block_tokens, ids))
}

/// Greedy generation through the scheduler, `plan`, and the real model;
/// also returns the logits row of each request's first sampled token.
fn run_real(r: &mut ModelRunner, s: &mut Scheduler<FreeListBlocks>, prompts: &[Vec<u32>], n: usize) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let ids: Vec<RequestId> = prompts.iter().map(|p| s.add(req(p, n)).unwrap()).collect();
    let mut gen: HashMap<RequestId, Vec<u32>> = HashMap::new();
    let mut first: HashMap<RequestId, Vec<f32>> = HashMap::new();
    for _ in 0..10_000 {
        let out = s.schedule();
        if out.is_empty() && !s.has_unfinished() {
            break;
        }
        let p = plan(s, &out);
        let logits = r.forward(&p.batch).unwrap();
        let outputs: Vec<StepOutput> = p
            .sample
            .iter()
            .map(|&(row, id)| {
                first.entry(id).or_insert_with(|| logits.row(row).to_vec());
                let (t, _) = sample::sample(logits.row(row), &SamplingParams::greedy(), &[], &[], &mut Pcg32::new(0, 54));
                StepOutput { id, token: t, stop: false }
            })
            .collect();
        drop(p);
        for ev in s.on_step(&outputs) {
            if let RequestEvent::Token { id, token } = ev {
                gen.entry(id).or_default().push(token);
            }
        }
    }
    (ids.iter().map(|id| gen.remove(id).unwrap_or_default()).collect(), ids.iter().map(|id| first.remove(id).unwrap_or_default()).collect())
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

// ---------------------------------------------------------------------------
// tests

#[test]
fn hand_example_chunk_plan() {
    // WHY: the chapter's worked example: budget 8, chunks of 4. Step 1
    //      prefills A (2 tokens) and the first 4 of B's 10; step 2 decodes A
    //      and prefills 4 more of B; step 3 decodes A and finishes B's prompt
    //      (2 tokens), so B's first token comes at step 3 and no step
    //      exceeds 8 tokens.
    // KIND: unit
    // CATCHES: s02
    // CHAPTER: L10.3 section 3
    let mut s = chunked(cfg(4, 8), free_list(4, 16), 4);
    let a = s.add(req(&[1, 2], 3)).unwrap();
    let b = s.add(req(&[3; 10], 1)).unwrap();
    let o1 = s.schedule();
    assert_eq!(o1.prefill, [Chunk { id: a, start: 0, len: 2, last: true }, Chunk { id: b, start: 0, len: 4, last: false }]);
    s.on_step(&[StepOutput { id: a, token: 9, stop: false }]);
    let o2 = s.schedule();
    assert_eq!(o2.decode, [a]);
    assert_eq!(o2.prefill, [Chunk { id: b, start: 4, len: 4, last: false }]);
    s.on_step(&[StepOutput { id: a, token: 9, stop: false }]);
    let o3 = s.schedule();
    assert_eq!(o3.decode, [a]);
    assert_eq!(o3.prefill, [Chunk { id: b, start: 8, len: 2, last: true }]);
    for o in [&o1, &o2, &o3] {
        assert!(o.num_tokens() <= 8);
    }
}

#[test]
fn whole_prompt_waits_for_a_step_of_its_own() {
    // WHY: the contrast of section 3: without chunks, B (10 tokens > the
    //      budget of 8) cannot share a step with A's decodes, so it waits
    //      until A finishes and then runs alone, a 10-token step.
    // KIND: unit
    // CHAPTER: L10.3 section 3
    let mut s = Scheduler::new(cfg(4, 8), free_list(4, 16)).with_prefill_policy(Box::new(WholePrompt));
    s.add(req(&[1, 2], 3)).unwrap();
    let b = s.add(req(&[3; 10], 1)).unwrap();
    let run = run_fake(&mut s, 0, 100, |_, _| {});
    assert_eq!(run.first_token_step[&b], 3);
    assert_eq!(run.steps[3].num_tokens(), 10);
}

#[test]
fn split_hand_example() {
    // WHY: the chunk boundaries of a 10-token prompt in chunks of 4 are
    //      (0, 4), (4, 4), (8, 2): every token once, in order.
    // KIND: unit
    // CATCHES: s05
    // CHAPTER: L10.3 section 3
    assert_eq!(split(10, 4), [(0, 4), (4, 4), (8, 2)]);
    assert_eq!(split(8, 4), [(0, 4), (4, 4)]);
    assert_eq!(split(3, 0), [(0, 1), (1, 1), (2, 1)], "a chunk of 0 means 1");
    assert!(split(0, 4).is_empty());
}

#[test]
fn plan_samples_only_sequence_ends() {
    // WHY: the mixed batch holds one sequence per decode (its newest token,
    //      at position n - 1) and one per chunk (its positions); only decodes
    //      and chunks that end the sequence are sampled: a middle chunk's
    //      logits predict a token the prompt already has.
    // KIND: unit
    // CATCHES: s02, s03, s04
    // CHAPTER: L10.3 section 4
    let mut s = chunked(cfg(4, 6), free_list(4, 16), 3);
    let a = s.add(req(&[1, 2], 5)).unwrap();
    s.schedule();
    s.on_step(&[StepOutput { id: a, token: 7, stop: false }]);
    let b = s.add(req(&[4, 5, 6, 7, 8], 1)).unwrap();
    let out = s.schedule();
    let p = plan(&s, &out);
    assert_eq!(p.batch.seqs.len(), 2);
    assert_eq!(p.batch.seqs[0], ForwardSeq { tokens: &[7], start: 2, blocks: s.blocks().block_table(a) });
    assert_eq!(p.batch.seqs[1], ForwardSeq { tokens: &[4, 5, 6], start: 0, blocks: s.blocks().block_table(b) });
    assert_eq!(p.sample, [(0, a)], "B's first chunk is not sampled");
}

#[test]
fn token_budget_never_exceeded() {
    // WHY: with chunked prefill every step processes at most
    //      max_batch_tokens tokens, even with prompts four times the budget,
    //      and every request still finishes with its expected output.
    // KIND: property
    // CATCHES: s01
    // CHAPTER: L10.3 section 2
    let mut g = frozen::Pcg32::new(ol_seed() + 303);
    let mut s = chunked(cfg(6, 16), free_list(4, 64), 8);
    let mut want: HashMap<RequestId, Vec<u32>> = HashMap::new();
    let run = run_fake(&mut s, 80, 5000, |step, s| {
        if step < 80 && g.below(2) == 0 {
            let prompt: Vec<u32> = (0..1 + g.below(64)).map(|_| g.below(50)).collect();
            let n = 1 + g.below(10) as usize;
            let id = s.add(req(&prompt, n)).unwrap();
            want.insert(id, fake_expected(&prompt, n));
        }
    });
    assert!(want.len() > 10);
    for out in &run.steps {
        assert!(out.num_tokens() <= 16, "step over budget: {out:?}");
    }
    for (id, w) in &want {
        assert_eq!(run.generated.get(id), Some(w), "request {id}");
    }
    assert_eq!(s.blocks().free_blocks(), 64);
}

#[test]
fn long_prompt_progresses_beside_decodes() {
    // WHY: chunking is what lets a long prompt start while others decode:
    //      with three requests decoding for 40 steps, a 40-token prompt gets
    //      its first token within a few steps under Chunked, but waits for
    //      the decodes to end under WholePrompt (budget 16).
    // KIND: property
    // CHAPTER: L10.3 section 2
    let ttft = |policy: Box<dyn PrefillPolicy + Send>| {
        let mut s = Scheduler::new(cfg(8, 16), free_list(4, 64)).with_prefill_policy(policy);
        let mut long = 0;
        let run = run_fake(&mut s, 0, 1000, |step, s| {
            if step == 0 {
                for i in 0..3 {
                    s.add(req(&[i + 1], 40)).unwrap();
                }
            }
            if step == 2 {
                long = s.add(req(&[5; 40], 2)).unwrap();
            }
        });
        run.first_token_step[&long] - 2
    };
    let c = ttft(Box::new(Chunked { chunk: 8 }));
    let w = ttft(Box::new(WholePrompt));
    assert!(c <= 4, "chunked TTFT {c} steps");
    assert!(w >= 35, "whole-prompt TTFT {w} steps");
}

#[test]
fn chunked_prefill_matches_whole_within_tolerance() {
    // WHY: absolute positions and stable cached values keep chunked logits
    //      within floating-point tolerance of a whole prefill.
    // KIND: differential
    // CATCHES: s03, s04, s05
    // CHAPTER: L10.3 section 2
    let prompts = vec![bytes("Once upon a time, in a land far away, there lived"), bytes("The cat sat on the mat and")];
    let (mut r0, f0) = real(64, 4);
    let mut whole = Scheduler::new(cfg(4, 1024), f0);
    let (want, want_first) = run_real(&mut r0, &mut whole, &prompts, 12);
    for chunk in [1usize, 3, 7, 16] {
        let (mut r, f) = real(64, 4);
        let mut s = chunked(cfg(4, 1024), f, chunk);
        let (got, got_first) = run_real(&mut r, &mut s, &prompts, 12);
        for (a, b) in got_first.iter().zip(&want_first) {
            let diff = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(diff < 2e-2, "chunk {chunk}: first-token logit diff {diff}");
        }
        assert_eq!(got, want, "chunk {chunk}: greedy continuation changed");
    }
    // and directly through the runner: chunks of 5 vs one call
    let mut r = ModelRunner::load(&tiny_llama(), &EngineConfig { kv: KvConfig { blocks: 64, block_size: 4 }, ..EngineConfig::default() }).unwrap();
    let toks = &prompts[0];
    let (_, whole_logits) = r.run_once(toks).unwrap();
    let pool = r.pool();
    let bl = lock(&pool).unwrap().alloc(toks.len().div_ceil(4)).unwrap();
    let mut last = Vec::new();
    for (st, n) in split(toks.len(), 5) {
        last = r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &toks[st..st + n], start: st, blocks: &bl }] }).unwrap().data;
    }
    for b in bl {
        lock(&pool).unwrap().release(b).unwrap();
    }
    let diff = last.iter().zip(&whole_logits).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(diff < 2e-2, "whole vs chunked prefill max logit diff {diff}");
}
