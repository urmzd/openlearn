//! Speculative decoding in the engine (L10.8): a port of L8.6.
//!
//! A cheap draft proposes up to k tokens from the request's own context
//! (prompt-lookup, or an n-gram table over the context); the target model
//! scores the pending token and every draft in ONE forward pass; the
//! verifier keeps the longest acceptable prefix plus one token of the
//! target's own; the KV cache is cut back to what was kept. Greedy
//! acceptance (temperature 0) is "the draft equals the target's argmax", so
//! the output is token for token the plain greedy output. Sampled acceptance
//! is M07.6's rejection step, so every emitted token follows the target's
//! distribution exactly.
//!
//! The random draws follow L8.6 (contracts/py/tinyllm/infer/spec.pyi): one
//! request generator, `stream(seed, sample)`; per verified draft position
//! two uniforms, u_accept then u_resample, both always; the bonus token after
//! a fully accepted draft is L10.1's `sample` (one uniform); greedy draws
//! nothing. Every sum is a left-to-right f64 loop in ascending id order, so
//! the accepted tokens equal your Python L8.6 on the same logits and seed.
//!
//! Chapter: content/courses/build-an-llm/p10-serving/08-speculative-decoding-in-the-engine.md.
//! Configuration: `[engine].speculative = { draft = "none" | "ngram" |
//! "prompt_lookup", k = 4 }` (config/runtime.schema.json); the accept rate is
//! the gauge `tl.engine.spec_accept_rate` (otel/metrics.yaml).

use std::collections::HashMap;
use std::fmt;

use crate::kv::KvPool;
use crate::forward::{ForwardBatch, ForwardSeq};
use crate::runner::{lock, ModelRunner};

use crate::sample::{apply_penalties, argmax, distribution, logprob_of, sample, stream, Pcg32, SamplingParams, PURPOSE_SAMPLE};

/// Which draft proposes tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Draft {
    /// No speculation: one token per target pass.
    None,
    /// Most frequent continuation of the last n - 1 tokens in the context.
    Ngram,
    /// Copy what followed the most recent earlier occurrence of the suffix.
    PromptLookup,
}

/// `[engine].speculative`, plus the n-gram sizes the drafts use.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecConfig {
    pub draft: Draft,
    /// Draft tokens per round (at most).
    pub k: usize,
    /// Prompt lookup tries suffixes of max_ngram down to min_ngram tokens;
    /// the n-gram draft uses contexts of max_ngram - 1 tokens down to 1.
    pub max_ngram: usize,
    pub min_ngram: usize,
}

/// What went wrong in a speculative step.
#[derive(Clone, Debug, PartialEq)]
pub enum SpecError {
    /// A configuration value outside its range.
    Config(String),
    /// Logits or draft distributions of the wrong shape, or a draft
    /// distribution that is not one.
    Shape(String),
    /// The KV pool refused (its status text).
    Kv(String),
    /// The target model failed.
    Target(String),
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecError::Config(m) => write!(f, "speculative config: {m}"),
            SpecError::Shape(m) => write!(f, "speculative shape: {m}"),
            SpecError::Kv(m) => write!(f, "speculative kv: {m}"),
            SpecError::Target(m) => write!(f, "speculative target: {m}"),
        }
    }
}

impl std::error::Error for SpecError {}

impl SpecConfig {
    /// The runtime.toml values: draft "none" | "ngram" | "prompt_lookup",
    /// 1 <= k <= 16; the n-gram sizes default to 3 and 1 (L8.6's
    /// PromptLookupDraft defaults).
    pub fn parse(draft: &str, k: i64) -> Result<SpecConfig, SpecError> {
        let d = match draft {
            "none" => Draft::None,
            "ngram" => Draft::Ngram,
            "prompt_lookup" => Draft::PromptLookup,
            other => return Err(SpecError::Config(format!("draft {other:?} is not none, ngram, or prompt_lookup"))),
        };
        if !(1..=16).contains(&k) {
            return Err(SpecError::Config(format!("k must lie in 1..=16, got {k}")));
        }
        Ok(SpecConfig { draft: d, k: k as usize, max_ngram: 3, min_ngram: 1 })
    }
}

/// Prompt-lookup decoding (L8.6 `PromptLookupDraft.propose`): for n =
/// max_ngram down to min_ngram, take the last n ids of ctx and find their
/// most recent earlier occurrence (a start i < len - n with ctx[i..i+n]
/// equal to them); the draft is ctx[i+n .. i+n+k], which may run into the
/// suffix itself. The first n that finds a non-empty continuation wins; none
/// gives an empty draft.
pub fn prompt_lookup(ctx: &[u32], k: usize, max_ngram: usize, min_ngram: usize) -> Vec<u32> {
    let l = ctx.len();
    if k == 0 || l < 2 || min_ngram == 0 {
        return Vec::new();
    }
    let top = max_ngram.min(l - 1);
    if top < min_ngram {
        return Vec::new();
    }
    for n in (min_ngram..=top).rev() {
        let suffix = &ctx[l - n..];
        // most recent earlier occurrence first: i from l - n - 1 down to 0
        for i in (0..l - n).rev() {
            if &ctx[i..i + n] == suffix {
                let end = (i + n + k).min(l);
                if i + n < end {
                    return ctx[i + n..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// The n-gram draft over the request's own context: up to k tokens, each
/// the most frequent token that followed the last m tokens of (ctx + the
/// drafted ids) anywhere earlier in that sequence, for m = n - 1 down to 1
/// (the first m with any match wins); ties go to the lowest id. Stops early
/// when no m matches.
pub fn ngram_draft(ctx: &[u32], k: usize, n: usize) -> Vec<u32> {
    let mut cur = ctx.to_vec();
    let mut out = Vec::new();
    if n < 2 {
        return out;
    }
    'draft: while out.len() < k {
        let l = cur.len();
        for m in (1..n).rev() {
            if l <= m {
                continue;
            }
            let key = &cur[l - m..];
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for i in 0..l - m {
                if &cur[i..i + m] == key {
                    *counts.entry(cur[i + m]).or_insert(0) += 1;
                }
            }
            if let Some((&tok, _)) = counts.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))) {
                out.push(tok);
                cur.push(tok);
                continue 'draft;
            }
        }
        break;
    }
    out
}

/// The draft of `cfg` for this context, at most `k` tokens (the caller
/// lowers k near max_tokens or the end of the cache).
pub fn propose(cfg: &SpecConfig, ctx: &[u32], k: usize) -> Vec<u32> {
    match cfg.draft {
        Draft::None => Vec::new(),
        Draft::PromptLookup => prompt_lookup(ctx, k, cfg.max_ngram, cfg.min_ngram),
        Draft::Ngram => ngram_draft(ctx, k, cfg.max_ngram),
    }
}

/// M07.6's residual distribution: normalize(max(0, p - q)); p itself when
/// p == q leaves nothing (then nothing is ever rejected).
pub fn residual(p: &[f64], q: &[f64]) -> Vec<f64> {
    let r: Vec<f64> = p.iter().zip(q).map(|(&a, &b)| (a - b).max(0.0)).collect();
    let mut z = 0.0f64;
    for &v in &r {
        z += v;
    }
    if z <= 0.0 {
        return p.to_vec();
    }
    r.iter().map(|&v| v / z).collect()
}

/// Inverse CDF over a dense distribution (M07.1's sample_categorical): the
/// first id with u < running sum; if rounding leaves the sum at or below u,
/// the last id with probability > 0.
pub fn sample_dense(p: &[f64], u: f64) -> usize {
    let mut c = 0.0f64;
    for (i, &pi) in p.iter().enumerate() {
        c += pi;
        if u < c {
            return i;
        }
    }
    p.iter().rposition(|&x| x > 0.0).unwrap_or(0)
}

/// M07.6's speculative step: accept the draft token x (drawn from q) with
/// probability min(1, p[x] / q[x]), i.e. when u_accept < p[x] / q[x];
/// otherwise draw the replacement from the residual with u_resample.
/// Returns (token, accepted). Shape error when q[x] is not > 0 (q never
/// proposed x).
pub fn speculative_step(p: &[f64], q: &[f64], x: usize, u_accept: f64, u_resample: f64) -> Result<(usize, bool), SpecError> {
    if p.len() != q.len() || x >= p.len() {
        return Err(SpecError::Shape(format!("p has {} ids, q {}, draft token {x}", p.len(), q.len())));
    }
    if q[x].is_nan() || q[x] <= 0.0 {
        return Err(SpecError::Shape(format!("q[{x}] = {} but the draft proposed {x}", q[x])));
    }
    if u_accept < p[x] / q[x] {
        return Ok((x, true));
    }
    Ok((sample_dense(&residual(p, q), u_resample), false))
}

/// L8.1's sampling_distribution as a dense vector: the penalized logits'
/// kept-and-softmaxed distribution (one-hot on the greedy id at T = 0).
pub fn dense_distribution(logits: &[f32], p: &SamplingParams, prompt: &[u32], output: &[u32]) -> Vec<f64> {
    let l = apply_penalties(logits, p, prompt, output);
    let mut d = vec![0.0f64; l.len()];
    if p.temperature == 0.0 {
        d[argmax(&l)] = 1.0;
        return d;
    }
    for (i, q) in distribution(&l, p) {
        d[i] = q;
    }
    d
}

/// One verification step's result.
#[derive(Clone, Debug, PartialEq)]
pub struct Verified {
    /// draft[..n_accepted] then one token of the target's own.
    pub tokens: Vec<u32>,
    /// L10.1's logprob of each emitted token under the row that produced it
    /// (log-softmax after penalties, before temperature).
    pub logprobs: Vec<f64>,
    pub n_accepted: usize,
}

/// L8.6's `verify_draft`. `rows[i]` holds the target's logits after
/// prompt + output + draft[..i] (so rows.len() == draft.len() + 1, row m
/// after the whole draft); `draft_probs[i]` is the draft's distribution for
/// draft[i] (None: a deterministic draft, one-hot on its token). Penalties
/// see `output` plus the tokens emitted so far in this step.
///
/// temperature 0: accept while draft[i] equals the greedy id of row i; the
/// first mismatch emits that greedy id instead; after m acceptances the
/// extra token is the greedy id of row m. No draws.
///
/// temperature > 0: per position P = dense_distribution(row i), Q =
/// draft_probs[i] or one-hot; u_accept then u_resample from rng (both
/// always); speculative_step decides. After m acceptances the extra token is
/// L10.1's `sample` of row m (one uniform).
pub fn verify_draft(
    rows: &[Vec<f32>],
    draft: &[u32],
    draft_probs: Option<&[Vec<f64>]>,
    p: &SamplingParams,
    prompt: &[u32],
    output: &[u32],
    rng: &mut Pcg32,
) -> Result<Verified, SpecError> {
    let m = draft.len();
    if rows.len() != m + 1 {
        return Err(SpecError::Shape(format!("need {} rows of logits for a draft of {m}, got {}", m + 1, rows.len())));
    }
    let v = rows[0].len();
    if rows.iter().any(|r| r.len() != v) {
        return Err(SpecError::Shape("rows of logits differ in length".to_string()));
    }
    if let Some(q) = draft_probs {
        if q.len() != m || q.iter().any(|r| r.len() != v) {
            return Err(SpecError::Shape(format!("draft_probs must be [{m}][{v}]")));
        }
    }
    let mut hist = output.to_vec();
    let mut out = Verified { tokens: Vec::new(), logprobs: Vec::new(), n_accepted: 0 };
    let emit = |out: &mut Verified, hist: &mut Vec<u32>, row: &[f32], tok: u32| {
        let lp = logprob_of(&apply_penalties(row, p, prompt, hist), tok as usize);
        out.tokens.push(tok);
        out.logprobs.push(lp);
        hist.push(tok);
    };
    for (i, &x) in draft.iter().enumerate() {
        if (x as usize) >= v {
            return Err(SpecError::Shape(format!("draft token {x} outside the vocabulary of {v}")));
        }
        if p.temperature == 0.0 {
            let g = argmax(&apply_penalties(&rows[i], p, prompt, &hist)) as u32;
            emit(&mut out, &mut hist, &rows[i], g);
            if g != x {
                out.n_accepted = i;
                return Ok(out);
            }
            continue;
        }
        let pd = dense_distribution(&rows[i], p, prompt, &hist);
        let qd = match draft_probs {
            Some(q) => q[i].clone(),
            None => {
                let mut d = vec![0.0; v];
                d[x as usize] = 1.0;
                d
            }
        };
        let u_accept = rng.uniform_f64();
        let u_resample = rng.uniform_f64();
        let (y, ok) = speculative_step(&pd, &qd, x as usize, u_accept, u_resample)?;
        emit(&mut out, &mut hist, &rows[i], y as u32);
        if !ok {
            out.n_accepted = i;
            return Ok(out);
        }
    }
    let (bonus, _) = sample(&rows[m], p, prompt, &hist, rng);
    emit(&mut out, &mut hist, &rows[m], bonus);
    out.n_accepted = m;
    Ok(out)
}

/// Blocks of `block_tokens` positions needed for `tokens` positions.
pub fn blocks_for(tokens: usize, block_tokens: usize) -> usize {
    tokens.div_ceil(block_tokens)
}

/// Cuts a sequence's KV back to its first `keep` positions after a
/// rejection: every block past blocks_for(keep) is released (one reference
/// dropped, so a block shared through the prefix cache survives for its
/// other owners) and removed from `table`, and the last kept block's fill
/// becomes the positions it still holds. Returns the number of blocks
/// released. Shape error when `keep` is more than the table covers.
pub fn rollback(pool: &mut KvPool, table: &mut Vec<u32>, keep: usize) -> Result<usize, SpecError> {
    let b = pool.cfg().block_tokens as usize;
    let need = blocks_for(keep, b);
    if need > table.len() {
        return Err(SpecError::Shape(format!("keep {keep} positions needs {need} blocks, the table has {}", table.len())));
    }
    let mut freed = 0;
    while table.len() > need {
        let id = table.pop().expect("len > need >= 0");
        pool.release(id).map_err(|e| SpecError::Kv(e.to_string()))?;
        freed += 1;
    }
    if let Some(&last) = table.last() {
        let fill = keep - (need - 1) * b;
        pool.set_fill(last, fill as u32).map_err(|e| SpecError::Kv(e.to_string()))?;
    }
    Ok(freed)
}

/// What `generate` needs from the target model: a KV cache it extends and
/// truncates. The engine's step loop implements it over the model runner
/// and its block table; course tests implement it over a fake model.
pub trait Target {
    /// Positions currently cached.
    fn cached(&self) -> usize;
    /// Runs the model over `ids` at positions cached()..cached() + len,
    /// appends them to the cache, and returns one row of logits per id
    /// (row j: the logits after ids[j]).
    fn extend(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, SpecError>;
    /// Drops cached positions at and past `len`.
    fn truncate(&mut self, len: usize) -> Result<(), SpecError>;
}

/// A request-local speculative target backed by the production model runner.
/// The temporary blocks belong to this request and are returned on drop.
pub struct RunnerTarget<'a> {
    runner: &'a mut ModelRunner,
    blocks: Vec<u32>,
    cached: usize,
}

impl<'a> RunnerTarget<'a> {
    pub fn new(runner: &'a mut ModelRunner) -> RunnerTarget<'a> {
        RunnerTarget { runner, blocks: Vec::new(), cached: 0 }
    }

    /// A target over blocks the request already holds (one reference each,
    /// returned on drop) whose first `cached` positions are filled: a
    /// disaggregated request resuming from KV another engine pushed (L10.6).
    pub fn resume(runner: &'a mut ModelRunner, blocks: Vec<u32>, cached: usize) -> RunnerTarget<'a> {
        RunnerTarget { runner, blocks, cached }
    }
}

impl Target for RunnerTarget<'_> {
    fn cached(&self) -> usize { self.cached }

    fn extend(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, SpecError> {
        if ids.is_empty() { return Ok(Vec::new()); }
        let bt = self.runner.block_tokens();
        let need = blocks_for(self.cached + ids.len(), bt);
        if need > self.blocks.len() {
            let pool = self.runner.pool();
            let mut pool = lock(&pool).map_err(|e| SpecError::Kv(e.to_string()))?;
            let more = pool.alloc(need - self.blocks.len()).map_err(|e| SpecError::Kv(e.to_string()))?;
            self.blocks.extend(more);
        }
        // The forward contract returns one row per sequence: the logits
        // after that sequence's last token. Verification needs a row after
        // every fed id, so each id is its own one-token sequence over the
        // request's blocks, at its own position, all in ONE forward. Within a
        // layer a sequence writes its K and V before it attends, so the
        // sequence at position p sees the ones before it; every row then
        // runs the same per-row math as a decode step of that position
        // (batch invariance, DESIGN 2.4), which keeps greedy output equal to
        // the plain loop's.
        let seqs = ids
            .iter()
            .enumerate()
            .map(|(offset, token)| ForwardSeq { tokens: std::slice::from_ref(token), start: self.cached + offset, blocks: &self.blocks })
            .collect();
        let logits = self.runner.forward(&ForwardBatch { seqs }).map_err(|e| SpecError::Target(e.to_string()))?;
        if logits.rows() != ids.len() {
            return Err(SpecError::Target(format!("forward returned {} rows for {} one-token sequences", logits.rows(), ids.len())));
        }
        let rows: Vec<Vec<f32>> = (0..ids.len()).map(|i| logits.row(i).to_vec()).collect();
        self.cached += ids.len();
        Ok(rows)
    }

    fn truncate(&mut self, len: usize) -> Result<(), SpecError> {
        if len > self.cached { return Err(SpecError::Shape(format!("cannot extend cache by truncating to {len}"))); }
        let bt = self.runner.block_tokens();
        let keep = blocks_for(len, bt);
        let pool = self.runner.pool();
        let mut pool = lock(&pool).map_err(|e| SpecError::Kv(e.to_string()))?;
        for id in self.blocks.drain(keep..) { pool.release(id).map_err(|e| SpecError::Kv(e.to_string()))?; }
        if len > 0 && len % bt != 0 {
            pool.set_fill(self.blocks[keep - 1], (len % bt) as u32).map_err(|e| SpecError::Kv(e.to_string()))?;
        }
        self.cached = len;
        Ok(())
    }
}

impl Drop for RunnerTarget<'_> {
    fn drop(&mut self) {
        let shared_pool = self.runner.pool();
        if let Ok(mut pool) = lock(&shared_pool) {
            for id in self.blocks.drain(..) { let _ = pool.release(id); }
        };
    }
}

/// Run speculative decoding directly against the production model runner.
pub fn generate_with_runner(runner: &mut ModelRunner, cfg: &SpecConfig, prompt: &[u32], p: &SamplingParams, seed: u64, max_new: usize, eos: &[u32]) -> Result<Generated, SpecError> {
    generate(&mut RunnerTarget::new(runner), cfg, prompt, p, seed, max_new, eos)
}

/// Counters behind `tl.engine.spec_accept_rate`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    pub drafted: usize,
    pub accepted: usize,
    pub target_calls: usize,
}

impl SpecStats {
    /// accepted / drafted; 0 when nothing was drafted.
    pub fn accept_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    Length,
    Stop,
}

/// One request's speculative generation.
#[derive(Clone, Debug, PartialEq)]
pub struct Generated {
    pub tokens: Vec<u32>,
    pub logprobs: Vec<f64>,
    pub stats: SpecStats,
    pub finish: Finish,
}

/// L8.6's `speculative_generate` over a [`Target`], up to `max_new` tokens.
/// Each round: the draft proposes up to min(k, max_new - emitted - 1) ids
/// after prompt + output; the target runs ONE extend over the tokens not yet
/// cached (the prompt in the first round, then the previous round's extra
/// token) followed by the draft; `verify_draft` keeps n_accepted drafts plus
/// one token; the cache is truncated to the kept length (rejected drafts'
/// KV is dropped). A token in `eos` ends the request (finish Stop) and is
/// not emitted. The generator is `stream(seed, sample)`.
pub fn generate(
    target: &mut dyn Target,
    cfg: &SpecConfig,
    prompt: &[u32],
    p: &SamplingParams,
    seed: u64,
    max_new: usize,
    eos: &[u32],
) -> Result<Generated, SpecError> {
    generate_stream(target, cfg, prompt, p, seed, max_new, eos, &mut |_, _| true)
}

/// [`generate`] for a server that streams: every emitted token (and its
/// logprob) goes to `on_token` as soon as its round is verified, before the
/// next round runs. When `on_token` returns false (the client went away),
/// generation stops at once and returns what it has (finish Length).
#[allow(clippy::too_many_arguments)]
pub fn generate_stream(
    target: &mut dyn Target, cfg: &SpecConfig, prompt: &[u32], p: &SamplingParams,
    seed: u64, max_new: usize, eos: &[u32], on_token: &mut dyn FnMut(u32, f64) -> bool,
) -> Result<Generated, SpecError> {
    if prompt.is_empty() {
        return Err(SpecError::Shape("empty prompt".to_string()));
    }
    let mut rng = stream(seed, PURPOSE_SAMPLE);
    let mut g = Generated { tokens: Vec::new(), logprobs: Vec::new(), stats: SpecStats::default(), finish: Finish::Length };
    let mut pending: Vec<u32> = prompt.to_vec();
    while g.tokens.len() < max_new {
        let room = max_new - g.tokens.len() - 1;
        let mut ctx = prompt.to_vec();
        ctx.extend_from_slice(&g.tokens);
        let draft = propose(cfg, &ctx, cfg.k.min(room));
        let base = target.cached();
        let mut feed = pending.clone();
        feed.extend_from_slice(&draft);
        let rows = target.extend(&feed)?;
        if rows.len() != feed.len() {
            return Err(SpecError::Target(format!("extend returned {} rows for {} ids", rows.len(), feed.len())));
        }
        let tail = &rows[pending.len() - 1..];
        let ver = verify_draft(tail, &draft, None, p, prompt, &g.tokens, &mut rng)?;
        g.stats.target_calls += 1;
        g.stats.drafted += draft.len();
        g.stats.accepted += ver.n_accepted;
        target.truncate(base + pending.len() + ver.n_accepted)?;
        for (&t, &lp) in ver.tokens.iter().zip(&ver.logprobs) {
            if eos.contains(&t) {
                g.finish = Finish::Stop;
                return Ok(g);
            }
            g.tokens.push(t);
            g.logprobs.push(lp);
            if !on_token(t, lp) {
                return Ok(g);
            }
            if g.tokens.len() == max_new {
                return Ok(g);
            }
        }
        pending = vec![*ver.tokens.last().expect("verify emits at least one token")];
    }
    Ok(g)
}
