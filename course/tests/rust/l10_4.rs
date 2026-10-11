//! L10.4 course tests: the block manager with prefix caching (tl-engine
//! block_manager.rs) over the Rust KV pool and the radix cache (L8.4).
//!
//! Annotated exemplars (DESIGN 5.12). Unit tests drive `BlockManager`
//! directly with token lists written out in each test and blocks of 4
//! tokens. The differential test runs the real model (L10.1's runner over
//! course/fixtures/L7.9/tiny-llama-2l) through your L10.2 scheduler in all
//! three modes. Random workloads come from the frozen PCG32 in `mod frozen`.
//! `bench_shared_prefix_ttft` is a B test: `ol bench L10.4 --assert` runs
//! it, `ol check` never does.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tl_engine::block_manager::BlockManager;
use tl_engine::forward::{ForwardBatch, ForwardSeq};
use tl_engine::runner::{lock, EngineConfig, KvConfig, ModelRunner, PrefixCache, SchedPolicy, SharedPool};
use tl_engine::sample::{self, Pcg32, SamplingParams};
use tl_engine::sched::{BlockSpace, Request, RequestEvent, RequestId, Scheduler, SchedulerConfig, StepOutput};
use tl_engine::kv::{KvCfg, KvPool, TL_F16, TL_KV_FORMAT_V1};

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

/// A bare pool (no model): blocks of 4 tokens.
fn pool(blocks: u32) -> SharedPool {
    let cfg = KvCfg { n_blocks: blocks, block_tokens: 4, n_layers: 1, n_kv_heads: 1, head_dim: 2, dtype: TL_F16, format: TL_KV_FORMAT_V1 };
    Arc::new(Mutex::new(KvPool::new(cfg).unwrap()))
}


fn fake_next(tokens: &[u32]) -> u32 {
    ((*tokens.last().unwrap() as usize * 7 + tokens.len()) % 50) as u32
}

fn sched_cfg(max_seqs: usize, budget: usize) -> SchedulerConfig {
    SchedulerConfig { max_seqs, max_batch_tokens: budget, max_waiting: 1000, policy: SchedPolicy::Fcfs, aging_steps: 0, max_model_len: 4096 }
}

fn tiny_llama() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol")).join("L7.9/tiny-llama-2l")
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

/// One engine step loop: scheduler + runner + greedy sampling, until every
/// request finishes. Returns the generated tokens per request and the
/// number of prompt tokens actually prefilled.
fn drain(r: &mut ModelRunner, s: &mut Scheduler<BlockManager>, ids: &[RequestId]) -> (Vec<Vec<u32>>, usize) {
    let mut gen: HashMap<RequestId, Vec<u32>> = HashMap::new();
    let mut prefilled = 0;
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
            prefilled += c.len;
            let t = s.tokens(c.id).unwrap();
            if c.last {
                rows.push((seqs.len(), c.id));
            }
            seqs.push(ForwardSeq { tokens: &t[c.start..c.start + c.len], start: c.start, blocks: s.blocks().block_table(c.id) });
        }
        let logits = r.forward(&ForwardBatch { seqs }).unwrap();
        let outs: Vec<StepOutput> = rows
            .into_iter()
            .map(|(row, id)| StepOutput { id, token: sample::sample(logits.row(row), &SamplingParams::greedy(), &[], &[], &mut Pcg32::new(0, 54)).0, stop: false })
            .collect();
        for ev in s.on_step(&outs) {
            if let RequestEvent::Token { id, token } = ev {
                gen.entry(id).or_default().push(token);
            }
        }
    }
    (ids.iter().map(|id| gen.remove(id).unwrap_or_default()).collect(), prefilled)
}

/// The real model and a scheduler over a block manager in `mode`.
fn engine(mode: PrefixCache, blocks: usize) -> (ModelRunner, Scheduler<BlockManager>) {
    let r = ModelRunner::load(&tiny_llama(), &EngineConfig { kv: KvConfig { blocks, block_size: 4 }, ..EngineConfig::default() }).unwrap();
    let bm = BlockManager::new(r.pool(), mode);
    (r, Scheduler::new(sched_cfg(8, 512), bm))
}

// ---------------------------------------------------------------------------
// tests

#[test]
fn hand_example_prefix_hit() {
    // WHY: the chapter's worked example, for both caches: A = tokens 1 to 10
    //      leaves its two full blocks cached; B = 1..8 then 99 98 97 reuses
    //      both (8 cached tokens) and needs one fresh block; A's partial
    //      third block was never shared.
    // KIND: unit
    // CATCHES: s01, s06, s07
    // CHAPTER: L10.4 section 3
    for mode in [PrefixCache::Hash, PrefixCache::Radix] {
        let p = pool(8);
        let mut bm = BlockManager::new(p.clone(), mode);
        let a: Vec<u32> = (1..=10).collect();
        assert_eq!(bm.allocate(1, &a).unwrap().cached_tokens, 0);
        let ta = bm.block_table(1).to_vec();
        assert_eq!(ta.len(), 3);
        bm.release(1, &a);
        let b = [1, 2, 3, 4, 5, 6, 7, 8, 99, 98, 97];
        assert_eq!(bm.allocate(2, &b).unwrap().cached_tokens, 8, "{mode:?}");
        let tb = bm.block_table(2).to_vec();
        assert_eq!(&tb[..2], &ta[..2], "{mode:?}: B starts with A's first two blocks");
        assert_eq!(tb.len(), 3);
        assert_eq!(bm.stats().hit_tokens, 8);
        assert!((bm.hit_rate() - 8.0 / 21.0).abs() < 1e-12, "{mode:?}: 8 hits over 10 + 11 tokens looked up");
        bm.release(2, &b);
    }
}

#[test]
fn none_mode_never_shares() {
    // WHY: with --prefix-cache=none every request gets fresh blocks and
    //      every block is free again when it ends.
    // KIND: unit
    // CHAPTER: L10.4 section 2
    let p = pool(8);
    let mut bm = BlockManager::new(p.clone(), PrefixCache::None);
    let a: Vec<u32> = (1..=10).collect();
    bm.allocate(1, &a).unwrap();
    bm.release(1, &a);
    assert_eq!(bm.allocate(2, &a).unwrap().cached_tokens, 0);
    bm.release(2, &a);
    let s = lock(&p).unwrap().stats();
    assert_eq!((s.free, s.used, s.cached), (8, 0, 0));
}

#[test]
fn last_token_is_always_computed() {
    // WHY: a prompt equal to a cached sequence still reuses at most len - 1
    //      tokens rounded down to a block: the model must run on the last
    //      prompt token to produce the logits of the first new one.
    // KIND: boundary
    // CATCHES: s02, s07
    // CHAPTER: L10.4 section 5, Pitfalls
    for mode in [PrefixCache::Hash, PrefixCache::Radix] {
        let p = pool(8);
        let mut bm = BlockManager::new(p, mode);
        let a: Vec<u32> = (1..=8).collect();
        bm.allocate(1, &a).unwrap();
        bm.release(1, &a);
        assert_eq!(bm.allocate(2, &a).unwrap().cached_tokens, 4, "{mode:?}: 7 usable tokens hold one whole block");
    }
}

#[test]
fn same_tokens_at_another_position_do_not_hit() {
    // WHY: a block's K and V depend on every token before it, so a block is
    //      named by its tokens AND its prefix (the chained hash, the tree
    //      path): tokens 5 to 8 cached at positions 4..8 must not serve a
    //      prompt that starts with 5 6 7 8.
    // KIND: boundary
    // CHAPTER: L10.4 section 5, Pitfalls
    for mode in [PrefixCache::Hash, PrefixCache::Radix] {
        let p = pool(8);
        let mut bm = BlockManager::new(p, mode);
        let a: Vec<u32> = (1..=9).collect();
        bm.allocate(1, &a).unwrap();
        bm.release(1, &a);
        let d = [5, 6, 7, 8, 1, 2, 3, 4, 0];
        assert_eq!(bm.allocate(2, &d).unwrap().cached_tokens, 0, "{mode:?}");
    }
}

#[test]
fn refcounts_balance_under_random_workload() {
    // WHY: requests drawn from three shared prefixes come and go through the
    //      scheduler with a small pool (forcing evictions); at the end no
    //      block is held by a request: in hash mode nothing is used, in radix
    //      mode exactly the cache's blocks are, and free + used + cached is
    //      the pool.
    // KIND: property
    // CATCHES: s05, s06, s07
    // CHAPTER: L10.4 section 2
    for mode in [PrefixCache::None, PrefixCache::Hash, PrefixCache::Radix] {
        let mut g = frozen::Pcg32::new(ol_seed() + 404);
        let p = pool(24);
        let bm = BlockManager::new(p.clone(), mode);
        let mut s = Scheduler::new(sched_cfg(4, 64), bm);
        let prefixes: [Vec<u32>; 3] = [(1..=12).collect(), (20..=27).collect(), vec![9; 16]];
        let mut want: HashMap<RequestId, usize> = HashMap::new();
        let mut got: HashMap<RequestId, usize> = HashMap::new();
        for step in 0..3000 {
            if step < 120 && g.below(2) == 0 {
                let mut prompt = prefixes[g.below(3) as usize].clone();
                prompt.extend((0..g.below(6)).map(|_| g.below(50)));
                let n = 1 + g.below(8) as usize;
                want.insert(s.add(Request { prompt, max_tokens: n, priority: 0 }).unwrap(), n);
            }
            let out = s.schedule();
            if out.is_empty() && !s.has_unfinished() && step >= 120 {
                break;
            }
            let mut ids = out.decode.clone();
            ids.extend(out.prefill.iter().filter(|c| c.last).map(|c| c.id));
            let outs: Vec<StepOutput> = ids.into_iter().map(|id| StepOutput { id, token: fake_next(s.tokens(id).unwrap()), stop: false }).collect();
            for ev in s.on_step(&outs) {
                if let RequestEvent::Token { id, .. } = ev {
                    *got.entry(id).or_default() += 1;
                }
            }
        }
        assert_eq!(got, want, "{mode:?}: every request finished with its tokens");
        let st = lock(&p).unwrap().stats();
        assert_eq!(st.free + st.used + st.cached, 24);
        match mode {
            PrefixCache::Radix => assert_eq!(st.used as usize, s.blocks().cached_blocks(), "radix: only the cache holds blocks"),
            _ => assert_eq!(st.used, 0, "{mode:?}: no block is held"),
        }
        if mode != PrefixCache::None {
            assert!(s.blocks().stats().hit_tokens > 0, "{mode:?}: shared prefixes must hit");
        }
    }
}

#[test]
fn eviction_reclaims_cached_blocks() {
    // WHY: cached blocks are spare capacity, not a leak: when the pool is
    //      full of cached prefixes, a new unrelated request evicts the least
    //      recently used ones and is admitted.
    // KIND: fault
    // CATCHES: s05, s07
    // CHAPTER: L10.4 section 2
    for mode in [PrefixCache::Hash, PrefixCache::Radix] {
        let p = pool(6);
        let mut bm = BlockManager::new(p.clone(), mode);
        for i in 0..6u32 {
            let t: Vec<u32> = (0..9).map(|k| 100 * i + k).collect();
            bm.allocate(i as u64, &t).expect("evicts what earlier requests cached");
            bm.release(i as u64, &t);
        }
        assert!(bm.stats().evictions > 0, "{mode:?}");
        let st = lock(&p).unwrap().stats();
        assert_eq!(st.free + st.used + st.cached, 6);
    }
}

#[test]
fn shared_blocks_return_to_free_after_eviction() {
    // WHY: a request that reused cached blocks must give its reference back
    //      when it ends; otherwise, once the cache evicts those blocks they
    //      stay "used" forever and the pool shrinks. After A caches two
    //      blocks and B reuses them, a request needing all 8 blocks must fit.
    // KIND: property
    // CATCHES: s01, s03, s05, s07
    // CHAPTER: L10.4 section 5, Pitfalls
    for mode in [PrefixCache::Hash, PrefixCache::Radix] {
        let p = pool(8);
        let mut bm = BlockManager::new(p.clone(), mode);
        let a: Vec<u32> = (1..=9).collect();
        bm.allocate(1, &a).unwrap();
        bm.release(1, &a);
        let b = [1, 2, 3, 4, 5, 6, 7, 8, 50, 51];
        assert_eq!(bm.allocate(2, &b).unwrap().cached_tokens, 8);
        bm.release(2, &b);
        let big: Vec<u32> = (100..132).collect();
        assert!(bm.allocate(3, &big).is_ok(), "{mode:?}: 32 tokens need every block, cached ones included");
        bm.release(3, &[]);
        let st = lock(&p).unwrap().stats();
        assert_eq!(st.free + st.used + st.cached, 8);
    }
}

#[test]
fn failed_allocation_returns_its_cache_hits() {
    // WHY: allocate is all or nothing: when the fresh blocks a request needs
    //      are not there, the references its cache hits took must be dropped
    //      too, or those blocks can never be evicted again.
    // KIND: fault
    // CATCHES: s04, s05
    // CHAPTER: L10.4 section 5, Pitfalls
    for mode in [PrefixCache::Hash, PrefixCache::Radix] {
        let p = pool(8);
        let mut bm = BlockManager::new(p.clone(), mode);
        let a: Vec<u32> = (1..=9).collect();
        bm.allocate(1, &a).unwrap();
        bm.release(1, &a);
        let mut b: Vec<u32> = (1..=8).collect();
        b.extend(200..230);
        assert!(bm.allocate(2, &b).is_err(), "{mode:?}: 38 tokens need 10 blocks of 8");
        let big: Vec<u32> = (100..132).collect();
        assert!(bm.allocate(3, &big).is_ok(), "{mode:?}: the failed request must not pin the cached blocks");
    }
}

#[test]
fn prefix_hit_skips_prefill_tokens() {
    // WHY: the scheduler starts a request's prefill at its cached length:
    //      the second request's first chunk begins at position 8, so only
    //      its new tokens run through the model.
    // KIND: unit
    // CHAPTER: L10.4 section 4
    let p = pool(16);
    let mut s = Scheduler::new(sched_cfg(4, 64), BlockManager::new(p, PrefixCache::Radix));
    let a = s.add(Request { prompt: (1..=10).collect(), max_tokens: 1, priority: 0 }).unwrap();
    s.schedule();
    s.on_step(&[StepOutput { id: a, token: 3, stop: false }]);
    let b = s.add(Request { prompt: vec![1, 2, 3, 4, 5, 6, 7, 8, 42, 43], max_tokens: 1, priority: 0 }).unwrap();
    let out = s.schedule();
    assert_eq!(out.prefill.len(), 1);
    assert_eq!((out.prefill[0].id, out.prefill[0].start, out.prefill[0].len), (b, 8, 2));
}

#[test]
fn identical_outputs_with_and_without_cache() {
    // WHY: a prefix cache changes how much is computed, never what: with a
    //      24-byte system prompt shared by four requests, greedy outputs are
    //      the same in none, hash, and radix modes, and the caches prefill
    //      fewer tokens.
    // KIND: differential
    // CATCHES: s01, s08
    // CHAPTER: L10.4 section 2
    let system = "You are a helpful bot. ";
    let users = ["Hi there", "Tell me a story", "What is 2+2?", "Bye"];
    let mut results = Vec::new();
    for mode in [PrefixCache::None, PrefixCache::Hash, PrefixCache::Radix] {
        let (mut r, mut s) = engine(mode, 64);
        // the first request alone fills the cache; the rest arrive after it
        let first = s.add(Request { prompt: bytes(&format!("{system}{}", users[0])), max_tokens: 8, priority: 0 }).unwrap();
        let (mut out, mut pre) = drain(&mut r, &mut s, &[first]);
        let ids: Vec<RequestId> =
            users[1..].iter().map(|u| s.add(Request { prompt: bytes(&format!("{system}{u}")), max_tokens: 8, priority: 0 }).unwrap()).collect();
        let (rest, p2) = drain(&mut r, &mut s, &ids);
        out.extend(rest);
        pre += p2;
        let st = lock(&r.pool()).unwrap().stats();
        assert_eq!(st.free + st.used + st.cached, 64);
        results.push((mode, out, pre));
    }
    let (_, base, base_pre) = &results[0];
    for (mode, out, pre) in &results[1..] {
        assert_eq!(out, base, "{mode:?}");
        assert!(pre + 3 * 20 <= *base_pre, "{mode:?} prefilled {pre} tokens, none {base_pre}");
    }
}

#[test]
#[ignore = "bench: ol bench L10.4 --assert"]
fn bench_shared_prefix_ttft() {
    // WHY: the point of the cache: with a long shared system prompt, the
    //      time to first token of a follow-up request falls by at least 2x
    //      (budget in course/modules/L10.4.toml); hash and radix are both
    //      measured. Prints one JSON line of metrics.
    // KIND: bench
    // CHAPTER: L10.4 section 2
    let system: String = "You are a careful assistant. Answer briefly and kindly. ".repeat(4);
    let mut metrics = Vec::new();
    let mut ttft = HashMap::new();
    for mode in [PrefixCache::None, PrefixCache::Hash, PrefixCache::Radix] {
        let (mut r, mut s) = engine(mode, 256);
        let warm = s.add(Request { prompt: bytes(&format!("{system}first")), max_tokens: 1, priority: 0 }).unwrap();
        drain(&mut r, &mut s, &[warm]);
        let mut best = f64::MAX;
        for k in 0..5 {
            let id = s.add(Request { prompt: bytes(&format!("{system}question {k}")), max_tokens: 1, priority: 0 }).unwrap();
            let t0 = Instant::now();
            drain(&mut r, &mut s, &[id]);
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        }
        ttft.insert(format!("{mode:?}"), best);
        metrics.push(format!("\"ttft_{}_ms\":{best:.3}", format!("{mode:?}").to_lowercase()));
    }
    let sh = ttft["None"] / ttft["Hash"];
    let sr = ttft["None"] / ttft["Radix"];
    println!("{{{},\"speedup_hash\":{sh:.3},\"speedup_radix\":{sr:.3}}}", metrics.join(","));
}
