//! L10.8 course tests: speculative decoding in the engine (tl-engine spec).
//!
//! Annotated exemplars (DESIGN 5.12). The verifier is held to your Python
//! L8.6 on shared inputs: course/fixtures/L10.8/spec_golden.json records
//! what the Python reference's `verify_draft` and `PromptLookupDraft` answer
//! for 120 and 150 seeded cases (course/oracle/L10.8/spec_golden.py). The
//! generation tests run against a fake target model written below: its
//! logits are a fixed function of the context, so plain greedy decoding is
//! computed here independently and compared token for token. The KV tests
//! use a real Rust KV pool, and one test runs RunnerTarget over L7.9's tiny
//! llama (course/fixtures/L7.9/tiny-llama-2l). JSON is read with `mod j`,
//! never with yours.

use std::path::PathBuf;

use tl_engine::sample::{child_seed, stream, Pcg32, SamplingParams, PURPOSE_SAMPLE};
use tl_engine::runner::{EngineConfig, KvConfig, ModelRunner};
use tl_engine::spec::{self, Draft, Finish, RunnerTarget, SpecConfig, SpecError, Target};
use tl_engine::kv::{KvCfg, KvPool, TL_F16};

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ss"))
}

fn golden() -> j::V {
    j::parse(&std::fs::read_to_string(fixtures().join("L10.8/spec_golden.json")).expect("spec_golden.json"))
}

fn u32s(v: &j::V) -> Vec<u32> {
    v.arr().iter().map(|x| x.u64() as u32).collect()
}

fn params(v: &j::V) -> SamplingParams {
    let mut p = SamplingParams::default();
    for (k, x) in v.obj() {
        match k.as_str() {
            "temperature" => p.temperature = x.f64(),
            "top_k" => p.top_k = x.u64() as usize,
            "top_p" => p.top_p = x.f64(),
            "min_p" => p.min_p = x.f64(),
            "repetition_penalty" => p.repetition_penalty = x.f64(),
            "presence_penalty" => p.presence_penalty = x.f64(),
            "frequency_penalty" => p.frequency_penalty = x.f64(),
            other => panic!("unknown param {other}"),
        }
    }
    p
}

// ---------------------------------------------------------------------------
// a fake target model

const V: usize = 16;

/// Logits after a context: the successor (7 * last + 3) mod 16 of the last
/// token gets 4.0, the token after it 2.0, every other id a small value
/// that depends on the last two tokens (so rows differ and nothing ties).
fn fake_row(ctx: &[u32]) -> Vec<f32> {
    let last = *ctx.last().unwrap() as usize;
    let prev = if ctx.len() > 1 { ctx[ctx.len() - 2] as usize } else { 0 };
    let succ = (7 * last + 3) % V;
    (0..V)
        .map(|v| {
            if v == succ {
                4.0
            } else if v == (succ + 1) % V {
                2.0
            } else {
                ((v * 5 + last * 3 + prev) % 11) as f32 * 0.1
            }
        })
        .collect()
}

/// Plain greedy decoding over the fake model, written independently.
fn greedy_reference(prompt: &[u32], n: usize) -> Vec<u32> {
    let mut ctx = prompt.to_vec();
    let mut out = Vec::new();
    for _ in 0..n {
        let row = fake_row(&ctx);
        let mut best = 0;
        for i in 1..V {
            if row[i] > row[best] {
                best = i;
            }
        }
        out.push(best as u32);
        ctx.push(best as u32);
    }
    out
}

/// The fake model with a cache of token ids (no KV payload).
struct Fake {
    ids: Vec<u32>,
    calls: usize,
}

impl Target for Fake {
    fn cached(&self) -> usize {
        self.ids.len()
    }
    fn extend(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, SpecError> {
        self.calls += 1;
        let mut rows = Vec::new();
        for &t in ids {
            self.ids.push(t);
            rows.push(fake_row(&self.ids));
        }
        Ok(rows)
    }
    fn truncate(&mut self, len: usize) -> Result<(), SpecError> {
        assert!(len <= self.ids.len(), "truncate({len}) beyond {} cached positions", self.ids.len());
        self.ids.truncate(len);
        Ok(())
    }
}

/// The fake model whose cache also holds KV blocks in a real Rust KV pool:
/// extend allocates a block whenever a position starts a new one and keeps
/// fills current; truncate is spec::rollback.
struct KvFake {
    fake: Fake,
    pool: KvPool,
    table: Vec<u32>,
}

const B: u32 = 4;
const N_BLOCKS: u32 = 64;

fn pool() -> KvPool {
    KvPool::new(KvCfg { n_blocks: N_BLOCKS, block_tokens: B, n_layers: 1, n_kv_heads: 1, head_dim: 2, dtype: TL_F16, format: 1 }).expect("pool")
}

impl Target for KvFake {
    fn cached(&self) -> usize {
        self.fake.cached()
    }
    fn extend(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, SpecError> {
        // Allocate blocks for all new positions, then compute the rows.
        let start = self.fake.cached();
        let end = start + ids.len();
        while spec::blocks_for(end, B as usize) > self.table.len() {
            let id = self.pool.alloc(1).map_err(|e| SpecError::Kv(e.to_string()))?[0];
            self.table.push(id);
        }
        for (k, &id) in self.table.iter().enumerate() {
            let lo = k * B as usize;
            let fill = end.saturating_sub(lo).min(B as usize);
            if fill > 0 {
                self.pool.set_fill(id, fill as u32).map_err(|e| SpecError::Kv(e.to_string()))?;
            }
        }
        self.fake.extend(ids)
    }
    fn truncate(&mut self, len: usize) -> Result<(), SpecError> {
        spec::rollback(&mut self.pool, &mut self.table, len)?;
        self.fake.truncate(len)
    }
}

// ---------------------------------------------------------------------------
// drafts

#[test]
fn hand_example_prompt_lookup() {
    // WHY: the chapter's worked example. ctx = [5 6 7 8 5 6], k = 3, n from
    //      3 down to 1: the 3-suffix [8 5 6] never occurred earlier; the
    //      2-suffix [5 6] occurred at position 0, so the draft copies what
    //      followed it: [7 8 5].
    // KIND: unit
    // CATCHES: s02, m02
    // CHAPTER: L10.8 section 3
    assert_eq!(spec::prompt_lookup(&[5, 6, 7, 8, 5, 6], 3, 3, 1), vec![7, 8, 5]);
    // The continuation may run into the suffix itself: in [1 2 1 2 1 2] the
    // suffix [2 1 2] last occurred at position 1, and what followed it is
    // [1 2], the suffix's own tail.
    assert_eq!(spec::prompt_lookup(&[1, 2, 1, 2, 1, 2], 4, 3, 1), vec![1, 2]);
    // Nothing repeats: no draft. k = 0: no draft.
    assert_eq!(spec::prompt_lookup(&[1, 2, 3], 4, 3, 1), Vec::<u32>::new());
    assert_eq!(spec::prompt_lookup(&[5, 6, 7, 8, 5, 6], 0, 3, 1), Vec::<u32>::new());
}

#[test]
fn prompt_lookup_matches_python_l8_6() {
    // WHY: the Rust draft is held to PromptLookupDraft on 150 seeded
    //      contexts: the most RECENT earlier occurrence wins (not the first),
    //      longer suffixes are tried first, and min_ngram is respected.
    // KIND: differential
    // CATCHES: s01, s02, m01, m02
    // CHAPTER: L10.8 section 2
    let g = golden();
    for (i, c) in g.get("prompt_lookup").arr().iter().enumerate() {
        let ctx = u32s(c.get("ctx"));
        let want = u32s(c.get("draft"));
        let got = spec::prompt_lookup(&ctx, c.get("k").u64() as usize, c.get("max_ngram").u64() as usize, c.get("min_ngram").u64() as usize);
        assert_eq!(got, want, "case {i}: ctx {ctx:?}");
    }
}

#[test]
fn ngram_draft_hand_example() {
    // WHY: the n-gram draft counts what followed the last n - 1 tokens
    //      anywhere earlier in the context. ctx = [1 2 3 1 2 4 1 2], n = 3:
    //      [1 2] was followed by 3 once and 4 once, a tie that goes to the
    //      lowest id, 3; then [2 3] was followed by 1. With no match at any
    //      order the draft stops.
    // KIND: unit
    // CATCHES: s03, m03
    // CHAPTER: L10.8 section 2
    assert_eq!(spec::ngram_draft(&[1, 2, 3, 1, 2, 4, 1, 2], 2, 3), vec![3, 1]);
    // The longest context wins: [1 2] was followed by 9; the shorter [2]
    // alone was followed by 9 and 4, whose tie would pick 4.
    assert_eq!(spec::ngram_draft(&[1, 2, 9, 3, 2, 4, 1, 2], 1, 3), vec![9]);
    // Back-off: [9 2] never occurred, but [2] was followed by 5 twice.
    assert_eq!(spec::ngram_draft(&[2, 5, 2, 5, 9, 2], 1, 3), vec![5]);
    assert_eq!(spec::ngram_draft(&[1, 2, 3], 3, 3), Vec::<u32>::new());
    let cfg = SpecConfig { draft: Draft::Ngram, k: 4, max_ngram: 3, min_ngram: 1 };
    assert_eq!(spec::propose(&cfg, &[1, 2, 3, 1, 2, 4, 1, 2], 2), vec![3, 1]);
    let none = SpecConfig { draft: Draft::None, ..cfg };
    assert!(spec::propose(&none, &[1, 1, 1, 1], 4).is_empty());
}

#[test]
fn config_parses_runtime_values() {
    // WHY: [engine].speculative = { draft, k } comes from runtime.toml; a
    //      typo or k = 0 must be refused at startup, not ignored.
    // KIND: boundary
    // CATCHES: m04
    // CHAPTER: L10.8 section 4
    assert_eq!(SpecConfig::parse("prompt_lookup", 4).unwrap().draft, Draft::PromptLookup);
    assert_eq!(SpecConfig::parse("ngram", 16).unwrap().k, 16);
    assert_eq!(SpecConfig::parse("none", 1).unwrap().draft, Draft::None);
    for (d, k) in [("lookup", 4), ("ngram", 0), ("ngram", 17)] {
        assert!(SpecConfig::parse(d, k).is_err(), "parse({d:?}, {k}) must fail");
    }
}

// ---------------------------------------------------------------------------
// verification

#[test]
fn hand_example_greedy_verify() {
    // WHY: the chapter's greedy round. Row 0's argmax is 2 and the draft's
    //      first token is 2: accepted. Row 1's argmax is 3 but the draft says
    //      1: rejected, and the target's own 3 is emitted instead. Emitted
    //      [2 3], one draft accepted, and no random draw at temperature 0.
    // KIND: unit
    // CATCHES: s08, m05
    // CHAPTER: L10.8 section 3
    let rows = vec![vec![0.0, 1.0, 3.0, 0.5], vec![0.0, 1.0, 0.5, 2.0], vec![5.0, 0.0, 0.0, 0.0]];
    let mut rng = Pcg32::new(1, 54);
    let before = rng.clone();
    let v = spec::verify_draft(&rows, &[2, 1], None, &SamplingParams::greedy(), &[], &[], &mut rng).unwrap();
    assert_eq!((v.tokens.clone(), v.n_accepted), (vec![2, 3], 1));
    assert_eq!(rng, before, "greedy verification takes no draws");
    // logprob of 2 under row 0: 3 - ln(e^0 + e^1 + e^3 + e^0.5)
    let lse = (1.0f64 + 1f64.exp() + 3f64.exp() + 0.5f64.exp()).ln();
    assert!((v.logprobs[0] - (3.0 - lse)).abs() < 1e-12, "logprob {}", v.logprobs[0]);
    // A fully accepted draft gets the bonus token from the last row.
    let v = spec::verify_draft(&rows, &[2, 3], None, &SamplingParams::greedy(), &[], &[], &mut rng).unwrap();
    assert_eq!((v.tokens, v.n_accepted), (vec![2, 3, 0], 2));
}

#[test]
fn hand_example_sampled_step() {
    // WHY: M07.6 by hand. p = [0.2 0.5 0.3], the draft proposed x = 0 from q
    //      = [0.6 0.2 0.2]: accept with probability p0 / q0 = 1/3. u_accept
    //      = 0.3 accepts; 0.5 rejects, and the replacement comes from the
    //      residual max(p - q, 0) = [0 0.3 0.1] / 0.4 = [0 0.75 0.25]:
    //      u_resample = 0.8 lands on id 2 (0.75 <= 0.8 < 1.0).
    // KIND: unit
    // CATCHES: s05, s06, s12, m07
    // CHAPTER: L10.8 section 3
    let p = [0.2, 0.5, 0.3];
    let q = [0.6, 0.2, 0.2];
    assert_eq!(spec::speculative_step(&p, &q, 0, 0.3, 0.8).unwrap(), (0, true));
    assert_eq!(spec::speculative_step(&p, &q, 0, 0.5, 0.8).unwrap(), (2, false));
    assert_eq!(spec::speculative_step(&p, &q, 0, 0.5, 0.7).unwrap(), (1, false));
    let r = spec::residual(&p, &q);
    assert!((r[0] - 0.0).abs() < 1e-15 && (r[1] - 0.75).abs() < 1e-15 && (r[2] - 0.25).abs() < 1e-15, "residual {r:?}");
    // p == q leaves no residual: p itself.
    assert_eq!(spec::residual(&q, &q), q.to_vec());
    // A token q never proposed is a shape error, not a division by zero.
    assert!(spec::speculative_step(&p, &[1.0, 0.0, 0.0], 1, 0.5, 0.5).is_err());
}

#[test]
fn verify_matches_python_l8_6() {
    // WHY: the port's whole promise: on the same target logits, draft,
    //      draft distributions, params, history, and seed, your Rust verifier
    //      emits the same tokens, accepts the same number, and leaves the
    //      generator at the same state (u_accept then u_resample per position,
    //      both always; one draw for the bonus token; none when greedy) as
    //      the Python L8.6 reference. 120 seeded cases from spec_golden.json.
    // KIND: differential
    // CATCHES: s04, s05, s07, s09, m06
    // CHAPTER: L10.8 section 2
    let g = golden();
    for c in g.get("verify").arr() {
        let name = c.get("name").str();
        let rows: Vec<Vec<f32>> = c.get("logits").arr().iter().map(|r| r.arr().iter().map(|x| x.f64() as f32).collect()).collect();
        let draft = u32s(c.get("draft"));
        let probs: Option<Vec<Vec<f64>>> = match c.get("draft_probs") {
            j::V::Null => None,
            v => Some(v.arr().iter().map(|r| r.arr().iter().map(|x| x.f64()).collect()).collect()),
        };
        let p = params(c.get("params"));
        let mut rng = stream(c.get("seed").u64(), PURPOSE_SAMPLE);
        let v = spec::verify_draft(&rows, &draft, probs.as_deref(), &p, &u32s(c.get("prompt")), &u32s(c.get("output")), &mut rng)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(v.tokens, u32s(c.get("emitted")), "{name}: emitted");
        assert_eq!(v.n_accepted as u64, c.get("n_accepted").u64(), "{name}: n_accepted");
        assert_eq!(v.tokens.len(), v.n_accepted + 1, "{name}: n + 1 tokens");
        assert_eq!(rng.next_u32() as u64, c.get("next_u32").u64(), "{name}: the generator consumed a different number of draws");
    }
}

#[test]
fn verify_rejects_bad_shapes() {
    // WHY: m drafts need exactly m + 1 rows (the last scores the bonus
    //      token); a short slice must be an error, never an index panic in
    //      the step loop; a draft distribution must cover the vocabulary.
    // KIND: boundary
    // CATCHES: m08
    // CHAPTER: L10.8 section 4
    let rows = vec![vec![0.0f32; 4]; 2];
    let mut rng = Pcg32::new(0, 54);
    let p = SamplingParams::default();
    assert!(spec::verify_draft(&rows, &[1, 2], None, &p, &[], &[], &mut rng).is_err());
    assert!(spec::verify_draft(&rows, &[1], Some(&[vec![1.0, 0.0, 0.0]]), &p, &[], &[], &mut rng).is_err());
    assert!(spec::verify_draft(&rows, &[9], None, &p, &[], &[], &mut rng).is_err());
    assert!(spec::verify_draft(&rows[..1], &[], None, &p, &[], &[], &mut rng).is_ok(), "no draft: one row, one token");
}

// ---------------------------------------------------------------------------
// generation

#[test]
fn greedy_spec_equals_greedy_without_spec() {
    // WHY: at temperature 0 speculation may change only the speed, never a
    //      token: for every draft kind and several prompts the output equals
    //      plain greedy decoding, computed independently. On the repetitive
    //      fake model prompt lookup also needs fewer target passes than
    //      tokens (that is the point of it).
    // KIND: differential
    // CATCHES: s10, s11, m09
    // CHAPTER: L10.8 section 2
    let prompts: [&[u32]; 3] = [&[1], &[3, 9, 4, 1, 3], &[0, 0, 0, 5]];
    for prompt in prompts {
        let want = greedy_reference(prompt, 40);
        for (d, k) in [(Draft::None, 4), (Draft::PromptLookup, 4), (Draft::Ngram, 3), (Draft::PromptLookup, 1)] {
            let cfg = SpecConfig { draft: d, k, max_ngram: 3, min_ngram: 1 };
            let mut t = Fake { ids: Vec::new(), calls: 0 };
            let g = spec::generate(&mut t, &cfg, prompt, &SamplingParams::greedy(), 0, 40, &[]).unwrap();
            assert_eq!(g.tokens, want, "draft {d:?} k {k} prompt {prompt:?}");
            assert_eq!(g.finish, Finish::Length);
            assert_eq!(g.logprobs.len(), 40);
            assert_eq!(g.stats.target_calls, t.calls);
            // The cache holds the prompt and every emitted token but the last
            // (it is the next round's input): no rejected draft, nothing past
            // max_new.
            assert_eq!(t.cached(), prompt.len() + 40 - 1, "draft {d:?} k {k}: cached positions");
            if d == Draft::None {
                assert_eq!(g.stats.target_calls, 40, "no draft: one pass per token");
            }
            if d == Draft::PromptLookup && k == 4 {
                assert!(g.stats.target_calls < 25, "prompt lookup on a periodic output: {} passes for 40 tokens", g.stats.target_calls);
                assert!(g.stats.accept_rate() > 0.5, "accept rate {}", g.stats.accept_rate());
            }
        }
    }
}

#[test]
fn eos_and_max_tokens_end_generation() {
    // WHY: a round can emit up to k + 1 tokens at once; the request must
    //      still stop exactly at max_new, and an EOS id inside an accepted
    //      run ends it there (finish Stop, EOS not emitted).
    // KIND: boundary
    // CATCHES: s08, m05, m10
    // CHAPTER: L10.8 section 5, Pitfalls
    let cfg = SpecConfig { draft: Draft::PromptLookup, k: 8, max_ngram: 3, min_ngram: 1 };
    let want = greedy_reference(&[3, 9, 4, 1, 3], 30);
    for n in [1, 2, 7, 13] {
        let mut t = Fake { ids: Vec::new(), calls: 0 };
        let g = spec::generate(&mut t, &cfg, &[3, 9, 4, 1, 3], &SamplingParams::greedy(), 0, n, &[]).unwrap();
        assert_eq!(g.tokens, want[..n].to_vec(), "max_new {n}");
    }
    let stop = want[6];
    let first = want.iter().position(|&x| x == stop).unwrap();
    let mut t = Fake { ids: Vec::new(), calls: 0 };
    let g = spec::generate(&mut t, &cfg, &[3, 9, 4, 1, 3], &SamplingParams::greedy(), 0, 30, &[stop]).unwrap();
    assert_eq!((g.tokens, g.finish), (want[..first].to_vec(), Finish::Stop));
}

#[test]
fn sampled_spec_keeps_the_target_distribution() {
    // WHY: with sampling, accept-or-resample (M07.6) makes each emitted
    //      token follow the target's distribution exactly, whatever the draft
    //      proposes. Over 4000 seeds the first token after the prompt [5 5
    //      5] (the prompt-lookup draft proposes 5, which the target rarely
    //      picks, so most rounds resample) must match the target's softmax
    //      at temperature 1: chi-square over the ids at p > 1e-3.
    // KIND: statistical
    // CATCHES: s06, s12, m06, m09
    // CHAPTER: L10.8 section 2
    let cfg = SpecConfig { draft: Draft::PromptLookup, k: 2, max_ngram: 3, min_ngram: 1 };
    let row = fake_row(&[5, 5, 5]);
    let m = row.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let z: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
    let probs: Vec<f64> = row.iter().map(|&x| (x as f64 - m).exp() / z).collect();
    let n = 4000;
    let mut counts = [0usize; V];
    for s in 0..n {
        let mut t = Fake { ids: Vec::new(), calls: 0 };
        let g = spec::generate(&mut t, &cfg, &[5, 5, 5], &SamplingParams::default(), child_seed(s, 9), 2, &[]).unwrap();
        assert_eq!(g.stats.drafted, 1, "seed {s}: the first round drafts one token");
        counts[g.tokens[0] as usize] += 1;
    }
    // Pool ids with small expected counts into one cell (expected >= 5 each).
    let mut chi = 0.0;
    let mut cells = 0;
    let (mut o_rest, mut e_rest) = (0.0, 0.0);
    for i in 0..V {
        let e = probs[i] * n as f64;
        if e >= 5.0 {
            chi += (counts[i] as f64 - e).powi(2) / e;
            cells += 1;
        } else {
            o_rest += counts[i] as f64;
            e_rest += e;
        }
    }
    if e_rest > 0.0 {
        chi += (o_rest - e_rest).powi(2) / e_rest.max(1e-9);
        cells += 1;
    }
    // chi-square critical values at p = 1e-3 for df = cells - 1
    let crit = [10.83, 13.82, 16.27, 18.47, 20.52, 22.46, 24.32, 26.12, 27.88, 29.59, 31.26, 32.91, 34.53, 36.12, 37.70];
    assert!(chi < crit[cells - 2], "chi-square {chi:.2} with {} df: counts {counts:?}, probs {probs:?}", cells - 1);
}

// ---------------------------------------------------------------------------
// KV rollback

#[test]
fn rollback_hand_example() {
    // WHY: B = 4 and 10 cached positions fill three blocks (4, 4, 2). Keeping
    //      5 positions needs ceil(5 / 4) = 2 blocks: the third is released
    //      and the second's fill drops to 1. Keeping 8 frees nothing more and
    //      sets the fill back to a full 4; keeping 0 releases everything.
    // KIND: unit
    // CATCHES: s13, m11, m12, m14
    // CHAPTER: L10.8 section 3
    let mut p = pool();
    let mut table = p.alloc(3).unwrap();
    for (id, f) in table.clone().into_iter().zip([4, 4, 2]) {
        p.set_fill(id, f).unwrap();
    }
    assert_eq!(spec::rollback(&mut p, &mut table, 5).unwrap(), 1);
    assert_eq!(table.len(), 2);
    assert_eq!(p.fill(table[1]), 1);
    assert_eq!(p.stats().used, 2);
    assert_eq!(spec::rollback(&mut p, &mut table, 8).unwrap(), 0);
    assert_eq!(p.fill(table[1]), 4);
    assert!(spec::rollback(&mut p, &mut table, 9).is_err(), "keep beyond the table is an error");
    assert_eq!(spec::rollback(&mut p, &mut table, 0).unwrap(), 2);
    assert!(table.is_empty());
    assert_eq!(p.stats().used, 0);
    assert_eq!(spec::blocks_for(0, 4), 0);
    assert_eq!(spec::blocks_for(4, 4), 1);
    assert_eq!(spec::blocks_for(5, 4), 2);
}

#[test]
fn rollback_drops_one_reference_only() {
    // WHY: a block can be shared through the prefix cache; rollback drops
    //      THIS sequence's reference, and the other owner keeps the block.
    //      Forcing it free would hand the other sequence's KV to the next
    //      allocation.
    // KIND: fault
    // CATCHES: s13, m12, m13
    // CHAPTER: L10.8 section 5, Pitfalls
    let mut p = pool();
    let mut table = p.alloc(2).unwrap();
    p.retain(table[1]).unwrap(); // another sequence shares block 1
    let shared = table[1];
    spec::rollback(&mut p, &mut table, 3).unwrap();
    assert_eq!(table.len(), 1);
    let s = p.stats();
    assert_eq!((s.used, s.free + s.used + s.cached), (2, N_BLOCKS), "the shared block stays used: {s:?}");
    p.release(shared).unwrap();
    p.release(table[0]).unwrap();
    assert_eq!(p.stats().used, 0);
}

#[test]
fn blocks_conserved_across_rollbacks() {
    // WHY: every rejected draft writes KV that must be given back. Over 200
    //      seeded requests whose drafts are often wrong (sampling at
    //      temperature 1.5 with the n-gram draft), after every round the
    //      sequence holds exactly ceil(cached / B) blocks, the pool's free +
    //      used + cached stays 64, and when each request ends and its table
    //      is released the pool is back to 0 used blocks.
    // KIND: property
    // CATCHES: s10, s11, m14
    // CHAPTER: L10.8 section 2
    let cfg = SpecConfig { draft: Draft::Ngram, k: 4, max_ngram: 3, min_ngram: 1 };
    let mut p = SamplingParams::default();
    p.temperature = 1.5;
    let mut kv = KvFake { fake: Fake { ids: Vec::new(), calls: 0 }, pool: pool(), table: Vec::new() };
    let mut rejected = 0;
    for s in 0..200u64 {
        let prompt: Vec<u32> = (0..(3 + s % 5)).map(|i| ((s * 7 + i * 3) % V as u64) as u32).collect();
        let g = spec::generate(&mut kv, &cfg, &prompt, &p, s, 1 + (s % 23) as usize, &[]).unwrap();
        rejected += g.stats.drafted - g.stats.accepted;
        assert_eq!(kv.fake.cached(), prompt.len() + g.tokens.len() - 1, "seed {s}: rejected drafts' positions are still cached");
        let st = kv.pool.stats();
        assert_eq!(st.free + st.used + st.cached, N_BLOCKS);
        assert_eq!(kv.table.len(), spec::blocks_for(kv.fake.cached(), B as usize), "seed {s}: table vs cached positions");
        assert_eq!(st.used as usize, kv.table.len(), "seed {s}: a rejected draft's block leaked");
        spec::rollback(&mut kv.pool, &mut kv.table, 0).unwrap();
        kv.fake.ids.clear();
        assert_eq!(kv.pool.stats().used, 0);
    }
    assert!(rejected > 100, "the scenario must reject drafts ({rejected} rejected)");
}

#[test]
fn generate_stream_hands_over_each_token_and_stops_when_told() {
    // WHY: the serve loop streams a speculative answer and frees it when the
    //      client leaves, so every emitted token reaches the listener as its
    //      round is verified (the same tokens and logprobs, in order), and a
    //      listener that answers false ends generation at once, with no
    //      further target pass.
    // KIND: conformance
    // CATCHES: m15
    // CHAPTER: L10.8 section 5
    let cfg = SpecConfig { draft: Draft::PromptLookup, k: 4, max_ngram: 3, min_ngram: 1 };
    let prompt = [3, 9, 4, 1, 3];
    let mut seen: Vec<(u32, f64)> = Vec::new();
    let mut t = Fake { ids: Vec::new(), calls: 0 };
    let g = spec::generate_stream(&mut t, &cfg, &prompt, &SamplingParams::greedy(), 0, 30, &[], &mut |tok, lp| {
        seen.push((tok, lp));
        true
    })
    .unwrap();
    assert_eq!(seen, g.tokens.iter().copied().zip(g.logprobs.iter().copied()).collect::<Vec<_>>());
    assert_eq!(g.tokens, greedy_reference(&prompt, 30));
    for stop_after in [1, 3, 7] {
        let mut n = 0;
        let mut t = Fake { ids: Vec::new(), calls: 0 };
        let g = spec::generate_stream(&mut t, &cfg, &prompt, &SamplingParams::greedy(), 0, 30, &[], &mut |_, _| {
            n += 1;
            n < stop_after
        })
        .unwrap();
        assert_eq!((n, g.tokens.len()), (stop_after, stop_after), "the listener said stop after {stop_after} tokens");
        assert_eq!(g.stats.target_calls, t.calls);
    }
}

fn tiny_llama() -> PathBuf {
    fixtures().join("L7.9/tiny-llama-2l")
}

#[test]
fn runner_target_scores_every_fed_position() {
    // WHY: verification needs the target's logits after EVERY fed id, but the
    //      runner's forward returns one row per sequence. On the tiny llama,
    //      RunnerTarget::extend over 9 ids gives 9 rows, each the logits a
    //      fresh run of that prefix gives (within 1e-4), and prompt-lookup
    //      speculation over the runner (several ids per pass) emits the same
    //      greedy tokens as no draft (one id per pass).
    // KIND: differential
    // CHAPTER: L10.8 section 2
    let ecfg = EngineConfig { kv: KvConfig { blocks: 32, block_size: 4 }, ..EngineConfig::default() };
    let ids: [u32; 9] = [72, 101, 108, 108, 111, 32, 72, 101, 108];
    let mut r = ModelRunner::load(&tiny_llama(), &ecfg).unwrap();
    let rows = RunnerTarget::new(&mut r).extend(&ids).unwrap();
    assert_eq!(rows.len(), ids.len(), "one row per fed id");
    for (j, row) in rows.iter().enumerate() {
        let (_, want) = r.run_once(&ids[..=j]).unwrap();
        assert_eq!(row.len(), want.len());
        let worst = row.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst <= 1e-4, "row {j}: max |diff| {worst} against a fresh run of the prefix");
    }
    assert_eq!(tl_engine::runner::lock(&r.pool()).unwrap().stats().used, 0, "the target's blocks go back on drop");
    let prompt: Vec<u32> = "abcabcabcab".bytes().map(u32::from).collect();
    let plain = SpecConfig { draft: Draft::None, k: 4, max_ngram: 3, min_ngram: 1 };
    let lookup = SpecConfig { draft: Draft::PromptLookup, ..plain.clone() };
    let want = spec::generate_with_runner(&mut r, &plain, &prompt, &SamplingParams::greedy(), 0, 24, &[]).unwrap();
    let got = spec::generate_with_runner(&mut r, &lookup, &prompt, &SamplingParams::greedy(), 0, 24, &[]).unwrap();
    assert_eq!(got.tokens, want.tokens);
    assert!(got.stats.drafted > 0, "prompt lookup proposed nothing on a periodic prompt");
}

/// A tiny JSON reader for the fixtures, independent of the code under test.
/// Numbers keep their text so 64-bit seeds stay exact.
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
            self.obj().iter().find(|(a, _)| a == k).map(|(_, v)| v).unwrap_or_else(|| panic!("no key {k:?}"))
        }
        pub fn obj(&self) -> &Vec<(String, V)> {
            match self {
                V::Obj(kv) => kv,
                _ => panic!("not an object: {self:?}"),
            }
        }
        pub fn str(&self) -> &str {
            match self {
                V::Str(s) => s,
                _ => panic!("not a string: {self:?}"),
            }
        }
        pub fn f64(&self) -> f64 {
            match self {
                V::Num(n) => n.parse().unwrap(),
                _ => panic!("not a number: {self:?}"),
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
        ws(b, &mut i);
        assert_eq!(i, b.len(), "trailing characters in JSON");
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
                            let V::Str(k) = val(b, i) else { panic!("object key is not a string") };
                            ws(b, i);
                            assert_eq!(b[*i], b':');
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
                let mut s = Vec::new();
                while b[*i] != b'"' {
                    if b[*i] == b'\\' {
                        *i += 1;
                        match b[*i] {
                            b'n' => s.push(b'\n'),
                            b't' => s.push(b'\t'),
                            b'u' => {
                                let h = std::str::from_utf8(&b[*i + 1..*i + 5]).unwrap();
                                let c = char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap();
                                let mut buf = [0u8; 4];
                                s.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                                *i += 4;
                            }
                            c => s.push(c),
                        }
                    } else {
                        s.push(b[*i]);
                    }
                    *i += 1;
                }
                *i += 1;
                V::Str(String::from_utf8(s).unwrap())
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
