//! L10.1 course tests: Candle-backed model math, Rust KV storage, int4
//! weights, and the Rust sampler and PCG32.
//!
//! Annotated exemplars (DESIGN 5.12). Oracles:
//! - course/fixtures/L8.1/sampler_golden.json: ids and logprobs your Python
//!   sampler's specification (L8.1, course/oracle/L8.1/sampler_golden.py)
//!   gives on fixed logits and seeds; the Rust sampler must match them bit
//!   for bit (D11).
//! - course/fixtures/L7.9/tiny-llama-2l/ with tiny_logits.json and
//!   tiny_greedy_32.json: a random 2-layer Llama with byte vocabulary and the
//!   float32 Hugging Face forward of it (course/oracle/L7.9/llama_hf.py).
//! - spec/pcg32.md vectors, copied inline below.
//! Every other model is written by hand, byte by byte, in the helpers. JSON
//! is read with `mod j` below, never with yours. Random inputs come from the
//! frozen PCG32 in `mod frozen`, never from yours.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use tl_engine::forward::{ForwardBatch, ForwardSeq};
use tl_engine::model::{parse_layout, Linear, ModelConfig, Weights};
use tl_engine::quant::{bf16_to_f32, f16_to_f32, f32_to_f16, pack_int4, unpack_int4, QLinear};
use tl_engine::runner::{lock, EngineConfig, KvConfig, ModelRunner, Quant};
use tl_engine::sample::{self, child_seed, stream, Pcg32, SamplingParams, PURPOSE_SAMPLE};
use tl_engine::kv::{KvCfg, KvPool, TL_EFULL, TL_F16, TL_KV_FORMAT_V1, TL_EINVAL};

// ---------------------------------------------------------------------------
// helpers

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ol-l10_1-{}-{}-{tag}", std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol"))
}

fn tiny_llama() -> PathBuf {
    fixtures().join("L7.9/tiny-llama-2l")
}

/// A safetensors file: u64 LE header length, the header padded with spaces
/// to a multiple of 8, then the data buffer.
fn safetensors(header: &str, data: &[u8]) -> Vec<u8> {
    let mut h = header.as_bytes().to_vec();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(data);
    out
}

/// Writes named tensors (name, dtype, shape, raw bytes) in order.
fn write_tensors(path: &Path, meta: &str, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = format!("{{\"__metadata__\":{meta}");
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let shape: Vec<String> = shape.iter().map(|x| x.to_string()).collect();
        header += &format!(",\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", shape.join(","), data.len(), data.len() + bytes.len());
        data.extend_from_slice(bytes);
    }
    header += "}";
    std::fs::write(path, safetensors(&header, &data)).unwrap();
}

fn le32(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The tracer checkpoint of L0.0 / L10.0: after byte i the logit of byte
/// i + 1 is 2, every other logit 0.
fn succ_bigram() -> PathBuf {
    let d = tmpdir("bigram");
    let mut w = vec![0.0f32; 256 * 256];
    for i in 0..256 {
        w[i * 256 + (i + 1) % 256] = 2.0;
    }
    write_tensors(&d.join("model.safetensors"), r#"{"format":"tinyllm"}"#, &[("bigram.weight".into(), "F32", vec![256, 256], le32(&w))]);
    std::fs::write(d.join("config.json"), r#"{"tl_arch":"bigram","tl_tokenizer":"bytes","vocab_size":256,"tl_format":1}"#).unwrap();
    d
}

fn engine_cfg(blocks: usize, block_size: usize) -> EngineConfig {
    EngineConfig { kv: KvConfig { blocks, block_size }, ..EngineConfig::default() }
}

/// Logits of `tokens` as one fresh sequence (prefill only).
fn prefill_logits(r: &mut ModelRunner, tokens: &[u32]) -> Vec<f32> {
    r.run_once(tokens).expect("run_once").1
}

/// Greedy continuation: prefill `prompt` once, then one token per step,
/// the way the engine decodes (blocks allocated up front, freed after).
fn decode_greedy(r: &mut ModelRunner, prompt: &[u32], n: usize) -> Vec<u32> {
    let bt = r.block_tokens();
    let pool = r.pool();
    let blocks = lock(&pool).unwrap().alloc((prompt.len() + n).div_ceil(bt)).unwrap();
    let mut toks = prompt.to_vec();
    let mut out = Vec::new();
    let mut logits = r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: prompt, start: 0, blocks: &blocks }] }).unwrap();
    for _ in 0..n {
        let (t, _) = sample::sample(logits.row(0), &SamplingParams::greedy(), prompt, &out, &mut Pcg32::new(0, 54));
        out.push(t);
        toks.push(t);
        let last = [t];
        logits = r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &last, start: toks.len() - 1, blocks: &blocks }] }).unwrap();
    }
    let mut p = lock(&pool).unwrap();
    for b in blocks {
        p.release(b).unwrap();
    }
    out
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Random values from the frozen PCG32 (never the engine's).
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
        /// Uniform in [-1, 1).
        pub fn sym(&mut self) -> f32 {
            (self.next_u32() as f64 / 2147483648.0 - 1.0) as f32
        }
    }
}

fn ol_seed() -> u64 {
    std::env::var("OL_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0)
}

/// A tiny JSON reader for the fixtures, independent of the code under test.
mod j {
    #[derive(Debug, Clone, PartialEq)]
    pub enum V {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<V>),
        Obj(Vec<(String, V)>),
    }

    impl V {
        pub fn get(&self, k: &str) -> &V {
            match self {
                V::Obj(kv) => kv.iter().find(|(a, _)| a == k).map(|(_, v)| v).unwrap_or_else(|| panic!("no key {k:?}")),
                _ => panic!("not an object"),
            }
        }
        pub fn str(&self) -> &str {
            match self {
                V::Str(s) => s,
                _ => panic!("not a string: {self:?}"),
            }
        }
        pub fn num(&self) -> f64 {
            match self {
                V::Num(n) => *n,
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
        let b: Vec<char> = s.chars().collect();
        let mut i = 0;
        let v = val(&b, &mut i);
        ws(&b, &mut i);
        assert_eq!(i, b.len(), "trailing characters in JSON");
        v
    }

    fn ws(b: &[char], i: &mut usize) {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    }

    fn val(b: &[char], i: &mut usize) -> V {
        ws(b, i);
        let rest: String = b[*i..].iter().take(5).collect();
        match b[*i] {
            '{' => {
                *i += 1;
                let mut kv = Vec::new();
                loop {
                    ws(b, i);
                    if b[*i] == '}' {
                        *i += 1;
                        return V::Obj(kv);
                    }
                    if b[*i] == ',' {
                        *i += 1;
                        continue;
                    }
                    let V::Str(k) = val(b, i) else { panic!("object key is not a string") };
                    ws(b, i);
                    assert_eq!(b[*i], ':');
                    *i += 1;
                    kv.push((k, val(b, i)));
                }
            }
            '[' => {
                *i += 1;
                let mut a = Vec::new();
                loop {
                    ws(b, i);
                    if b[*i] == ']' {
                        *i += 1;
                        return V::Arr(a);
                    }
                    if b[*i] == ',' {
                        *i += 1;
                        continue;
                    }
                    a.push(val(b, i));
                }
            }
            '"' => {
                *i += 1;
                let mut s = String::new();
                loop {
                    let c = b[*i];
                    *i += 1;
                    match c {
                        '"' => return V::Str(s),
                        '\\' => {
                            let e = b[*i];
                            *i += 1;
                            match e {
                                'n' => s.push('\n'),
                                't' => s.push('\t'),
                                'u' => {
                                    let h: String = b[*i..*i + 4].iter().collect();
                                    s.push(char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap_or('\u{FFFD}'));
                                    *i += 4;
                                }
                                other => s.push(other),
                            }
                        }
                        c => s.push(c),
                    }
                }
            }
            't' if rest.starts_with("true") => {
                *i += 4;
                V::Bool(true)
            }
            'f' if rest.starts_with("false") => {
                *i += 5;
                V::Bool(false)
            }
            'n' if rest.starts_with("null") => {
                *i += 4;
                V::Null
            }
            _ => {
                let start = *i;
                while *i < b.len() && "+-.eE0123456789".contains(b[*i]) {
                    *i += 1;
                }
                let t: String = b[start..*i].iter().collect();
                V::Num(t.parse().unwrap_or_else(|_| panic!("bad JSON at {rest:?}")))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the sampler and PCG32

#[test]
fn hand_example_sampling() {
    // WHY: the worked example of spec/sampling.md and the chapter's section
    //      3: top-k 3 then top-p 0.8 keep ids 1 and 3, stream(0, sample)
    //      draws u = 0.80209..., and the walk lands on id 3 with logprob
    //      -ln(2.52153) = -0.92487.
    // KIND: unit
    // CATCHES: s01, s02, s05
    // CHAPTER: L10.1 section 3
    assert_eq!(child_seed(0, PURPOSE_SAMPLE), 0xF88BB8A8724C81EC);
    let mut g = stream(0, PURPOSE_SAMPLE);
    let u = g.clone().uniform_f64();
    assert!((u - 0.80209).abs() < 1e-5, "first uniform of stream(0, sample) is {u}");
    let p = SamplingParams { top_k: 3, top_p: 0.8, ..SamplingParams::default() };
    let (id, lp) = sample::sample(&[1.0, 3.0, 2.0, 3.0, -1.0], &p, &[], &[], &mut g);
    assert_eq!(id, 3);
    assert!((lp - (-0.92487)).abs() < 1e-5, "logprob {lp}");
}

#[test]
fn pcg32_reference_vectors() {
    // WHY: one generator across Python, C, Rust, and Go (D10): O'Neill's
    //      pcg32(42) demo line, uniform_f64's 53-bit construction, and the
    //      sub-stream derivation, all from spec/pcg32.vectors.json.
    // KIND: golden
    // CATCHES: s01, s02
    // CHAPTER: L10.1 section 2
    let mut r = Pcg32::new(42, 54);
    let got: Vec<u32> = (0..6).map(|_| r.next_u32()).collect();
    assert_eq!(got, [0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]);
    let mut r = Pcg32::new(1 << 63, 54);
    let got: Vec<u32> = (0..4).map(|_| r.next_u32()).collect();
    assert_eq!(got, [2341717954, 1780988301, 3407770941, 286718070]);
    let mut r = Pcg32::new(0, 54);
    let u: Vec<f64> = (0..3).map(|_| r.uniform_f64()).collect();
    assert_eq!(u, [0.2803122753265841, 0.4892241428740438, 0.6123542393923406]);
    assert_eq!(child_seed(1, PURPOSE_SAMPLE), 8196980753821780235);
    let mut s = stream(1 << 63, PURPOSE_SAMPLE);
    let got: Vec<u32> = (0..4).map(|_| s.next_u32()).collect();
    assert_eq!(got, [3049651521, 2189754526, 922169935, 2648089264]);
}

#[test]
fn sampler_matches_l81_golden() {
    // WHY: given the same logits and seed, the Rust sampler returns the same
    //      ids and the same logprobs as the L8.1 specification, bit for bit,
    //      across temperature, top-k, top-p, min-p, the three penalties,
    //      greedy, and -inf masks (spec/sampling.md, D11). Twelve tokens per
    //      case, each fed back as history.
    // KIND: differential, golden
    // CATCHES: s01, s02, s03, s05, s06, s07
    // CHAPTER: L10.1 section 2
    let text = std::fs::read_to_string(fixtures().join("L8.1/sampler_golden.json")).unwrap();
    let doc = j::parse(&text);
    for case in doc.get("cases").arr() {
        let name = case.get("name").str();
        let logits: Vec<f32> = case.get("logits").arr().iter().map(|v| v.str().parse::<f64>().unwrap() as f32).collect();
        let pr = case.get("params");
        let p = SamplingParams {
            temperature: pr.get("temperature").num(),
            top_k: pr.get("top_k").num() as usize,
            top_p: pr.get("top_p").num(),
            min_p: pr.get("min_p").num(),
            repetition_penalty: pr.get("repetition_penalty").num(),
            presence_penalty: pr.get("presence_penalty").num(),
            frequency_penalty: pr.get("frequency_penalty").num(),
        };
        let prompt: Vec<u32> = case.get("prompt").arr().iter().map(|v| v.num() as u32).collect();
        let want_ids: Vec<u32> = case.get("ids").arr().iter().map(|v| v.num() as u32).collect();
        let want_lp: Vec<f64> = case.get("logprobs").arr().iter().map(|v| v.num()).collect();
        let mut rng = stream(case.get("seed").num() as u64, PURPOSE_SAMPLE);
        let mut out: Vec<u32> = Vec::new();
        for (step, (&wid, &wlp)) in want_ids.iter().zip(&want_lp).enumerate() {
            let (id, lp) = sample::sample(&logits, &p, &prompt, &out, &mut rng);
            assert_eq!(id, wid, "case {name}, token {step}: id");
            assert_eq!(lp.to_bits(), wlp.to_bits(), "case {name}, token {step}: logprob {lp} vs {wlp}");
            out.push(id);
        }
    }
}

#[test]
fn greedy_takes_no_draw_and_sampling_takes_one() {
    // WHY: draw accounting (spec/sampling.md): greedy uses no uniform and a
    //      sampled token exactly one, so the generator's position after n
    //      tokens is known; disaggregated serving (L10.6) depends on it.
    // KIND: property
    // CHAPTER: L10.1 section 2
    let logits = [0.5f32, 2.0, 1.0, -1.0];
    let mut g = stream(7, PURPOSE_SAMPLE);
    let before = g.clone();
    let (id, _) = sample::sample(&logits, &SamplingParams::greedy(), &[], &[], &mut g);
    assert_eq!(id, 1);
    assert_eq!(g, before, "greedy must not touch the generator");
    let mut expect = before.clone();
    expect.uniform_f64();
    sample::sample(&logits, &SamplingParams::default(), &[], &[], &mut g);
    assert_eq!(g, expect, "a sampled token takes exactly one uniform_f64");
}

#[test]
fn top_p_keeps_the_crossing_token() {
    // WHY: nucleus sampling keeps the smallest prefix whose mass reaches p
    //      INCLUDING the token that crosses it (>=, not >); with mass
    //      exactly at p the crossing token stays.
    // KIND: boundary
    // CATCHES: s04
    // CHAPTER: L10.1 section 5, Pitfalls
    let l: Vec<f64> = [0.5f64, 0.25, 0.25].iter().map(|p| p.ln()).collect();
    let p = SamplingParams { top_p: 0.75, ..SamplingParams::default() };
    let kept: Vec<usize> = sample::distribution(&l, &p).iter().map(|x| x.0).collect();
    assert_eq!(kept, [0, 1], "0.5 + 0.25 reaches 0.75 at id 1, which stays");
    let p = SamplingParams { top_p: 0.5, ..SamplingParams::default() };
    let kept: Vec<usize> = sample::distribution(&l, &p).iter().map(|x| x.0).collect();
    assert_eq!(kept, [0]);
}

#[test]
fn penalties_follow_hf_and_openai_semantics() {
    // WHY: the repetition penalty divides a positive logit and MULTIPLIES a
    //      negative one (HF), over prompt and output; presence and frequency
    //      subtract only for ids already generated (OpenAI).
    // KIND: unit
    // CATCHES: s06, s07
    // CHAPTER: L10.1 section 2
    let x = [2.0f32, -2.0, 1.0, 1.0];
    let p = SamplingParams { repetition_penalty: 2.0, ..SamplingParams::default() };
    assert_eq!(sample::apply_penalties(&x, &p, &[0, 1], &[]), [1.0, -4.0, 1.0, 1.0]);
    let p = SamplingParams { presence_penalty: 0.5, frequency_penalty: 0.25, ..SamplingParams::default() };
    assert_eq!(sample::apply_penalties(&x, &p, &[2], &[3, 3]), [2.0, -2.0, 1.0, 0.0]);
}

#[test]
fn seeded_sampling_matches_the_distribution() {
    // WHY: 4000 draws at T = 1 from probabilities 0.4, 0.3, 0.2, 0.1 fit
    //      them (chi-square, 3 degrees of freedom, below 16.27, p = 1e-3), and
    //      the same seed gives the same draws.
    // KIND: statistical
    // CATCHES: s01
    // CHAPTER: L10.1 section 2
    let probs = [0.4f64, 0.3, 0.2, 0.1];
    let logits: Vec<f32> = probs.iter().map(|p| p.ln() as f32).collect();
    let run = |seed: u64| {
        let mut g = stream(seed, PURPOSE_SAMPLE);
        (0..4000).map(|_| sample::sample(&logits, &SamplingParams::default(), &[], &[], &mut g).0).collect::<Vec<u32>>()
    };
    let a = run(ol_seed() + 11);
    assert_eq!(a, run(ol_seed() + 11));
    let mut counts = [0f64; 4];
    for &t in &a {
        counts[t as usize] += 1.0;
    }
    let chi: f64 = counts.iter().zip(probs).map(|(c, p)| (c - 4000.0 * p).powi(2) / (4000.0 * p)).sum();
    assert!(chi < 16.27, "chi-square {chi} for counts {counts:?}");
}

// ---------------------------------------------------------------------------
// Rust-owned KV cache

fn kv_cfg(blocks: u32) -> KvCfg {
    KvCfg { n_blocks: blocks, block_tokens: 4, n_layers: 2, n_kv_heads: 2, head_dim: 8, dtype: TL_F16, format: TL_KV_FORMAT_V1 }
}

#[test]
fn kv_pool_bounds_and_cache_lifecycle() {
    // WHY: invalid block ids, double releases, slab bounds, and cached-block
    //      lookup must return errors or preserve the expected lifecycle state.
    // KIND: boundary
    // CHAPTER: L10.1 section 2
    let mut pool = KvPool::new(kv_cfg(4)).unwrap();
    assert_eq!(pool.alloc(5).unwrap_err().status, TL_EFULL);
    assert_eq!(pool.stats().free, 4);
    let ids = pool.alloc(2).unwrap();
    pool.release(ids[0]).unwrap();
    assert_eq!(pool.release(ids[0]).unwrap_err().status, TL_EINVAL);
    pool.retain(ids[1]).unwrap();
    pool.release(ids[1]).unwrap();
    assert!(pool.slab(4, 0, false).is_err());
    assert!(pool.slab(ids[1], 2, true).is_err());
    pool.set_fill(ids[1], 4).unwrap();
    let h = KvPool::block_hash(0, &[1, 2, 3, 4]);
    assert!(pool.register(ids[1], h).unwrap());
    pool.release(ids[1]).unwrap();
    assert_eq!(pool.stats().cached, 1);
    assert_eq!(pool.lookup(h), Some(ids[1]));
    assert_eq!(pool.lookup(h ^ 1), None);
}

#[test]
fn kv_pool_evicts_the_oldest_cached_block() {
    // WHY: when no free block remains, allocation evicts the least recently
    //      released cached block and leaves newer prefixes available.
    // KIND: boundary
    // CATCHES: s18
    // CHAPTER: L10.1 section 2
    let mut pool = KvPool::new(kv_cfg(2)).unwrap();
    let ids = pool.alloc(2).unwrap();
    pool.set_fill(ids[0], 4).unwrap();
    pool.set_fill(ids[1], 4).unwrap();
    let first = KvPool::block_hash(0, &[1, 2, 3, 4]);
    let second = KvPool::block_hash(0, &[5, 6, 7, 8]);
    pool.register(ids[0], first).unwrap();
    pool.register(ids[1], second).unwrap();
    pool.release(ids[0]).unwrap();
    pool.release(ids[1]).unwrap();
    assert_eq!(pool.alloc(1).unwrap(), [ids[0]]);
    assert_eq!(pool.lookup(first), None);
    assert_eq!(pool.lookup(second), Some(ids[1]));
    assert_eq!(pool.stats().evictions, 1);
}

#[test]
fn kv_pool_rejects_zero_capacity() {
    // WHY: a cache with no blocks cannot satisfy a forward and must fail at
    //      configuration time rather than behave like an empty valid cache.
    // KIND: boundary
    // CATCHES: s19
    // CHAPTER: L10.1 section 5, Pitfalls
    assert!(KvPool::new(kv_cfg(0)).is_err());
}

#[test]
fn kv_pool_exports_and_imports_rust_owned_blocks() {
    // WHY: transferring a cache block must preserve its stored f16 bytes.
    // KIND: unit
    // CHAPTER: L10.1 section 2
    let mut src = KvPool::new(kv_cfg(2)).unwrap();
    let id = src.alloc(1).unwrap()[0];
    src.set_fill(id, 4).unwrap();
    src.slab_mut(id, 0, false).unwrap()[0] = 0x3c00;
    let bytes = src.export(&[id]).unwrap();
    let mut dst = KvPool::new(kv_cfg(2)).unwrap();
    let imported = dst.import(&bytes).unwrap();
    assert_eq!(dst.slab(imported[0], 0, false).unwrap()[0], 0x3c00);
}

// ---------------------------------------------------------------------------
// numbers: f16, bf16, int4

#[test]
fn f16_rounds_to_nearest_even() {
    // WHY: K and V are stored as f16 (KV format v1) and checkpoints arrive
    //      as BF16 or F16; f32 -> f16 must round to nearest, ties to even,
    //      as numpy and M09.4 do, or the KV cache drifts from the reference.
    // KIND: boundary
    // CATCHES: s08
    // CHAPTER: L10.1 section 2
    assert_eq!(f32_to_f16(1.0), 0x3C00);
    assert_eq!(f32_to_f16(-2.0), 0xC000);
    assert_eq!(f32_to_f16(65504.0), 0x7BFF);
    assert_eq!(f32_to_f16(65520.0), 0x7C00, "rounds up to inf");
    assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3C00, "a tie goes to the even neighbour");
    assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3C02, "this tie goes up, to the even one");
    assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) + 2f32.powi(-20)), 0x3C01, "past the tie rounds up");
    assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001, "smallest subnormal");
    assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000, "half of it ties to even zero");
    assert_eq!(f32_to_f16(f32::NAN), 0x7E00);
    for h in 0..=u16::MAX {
        let x = f16_to_f32(h);
        if !x.is_nan() {
            assert_eq!(f32_to_f16(x), h, "f16 {h:#06x} -> {x} -> back");
        }
    }
    assert_eq!(bf16_to_f32(0x3F80), 1.0);
    assert_eq!(bf16_to_f32(0xC0A0), -5.0);
}

#[test]
fn int4_hand_example() {
    // WHY: the chapter's worked example: q = [-8, 7, 1, -1] packs to bytes
    //      0x78 0xF1 (even column in the low nibble, two's complement), and
    //      the weights [0.7, -1.4, 0.0, 0.29995] in one group of 4 get the
    //      f16 scale 0.19995117 = f16(1.4 / 7) and q = [4, -7, 0, 2]: the
    //      last one is 2 only with the stored f16 scale (0.29995 / 0.2 is
    //      1.49975, which rounds to 1), L8.5's rule.
    // KIND: unit
    // CATCHES: s09, s10
    // CHAPTER: L10.1 section 3
    assert_eq!(pack_int4(&[-8, 7, 1, -1], 1, 4).unwrap(), [0x78, 0xF1]);
    assert_eq!(unpack_int4(&[0x78, 0xF1]), [-8, 7, 1, -1]);
    assert!(pack_int4(&[8, 0], 1, 2).is_err());
    let q = QLinear::quantize(&[0.7, -1.4, 0.0, 0.29995], 1, 4, 4).unwrap();
    assert_eq!(q.scales, [f32_to_f16(0.2)]);
    assert_eq!(f16_to_f32(q.scales[0]), 0.199951171875);
    assert_eq!(unpack_int4(&q.qweight), [4, -7, 0, 2]);
    let w = q.dequantize();
    assert_eq!(w, [4.0 * 0.199951171875, -7.0 * 0.199951171875, 0.0, 2.0 * 0.199951171875]);
}

#[test]
fn q4_linear_matches_its_dequantized_weights() {
    // WHY: the int4 product equals x @ dequantize(W)^T in f32 up to
    //      summation order (the q4 kernel sums each group, then the groups):
    //      the packed layout and the scales mean what formats/safetensors.md
    //      says.
    // KIND: differential
    // CATCHES: s09
    // CHAPTER: L10.1 section 2
    let mut g = frozen::Pcg32::new(ol_seed() + 5);
    let (m, out, inp, group) = (3usize, 10usize, 64usize, 16usize);
    let w: Vec<f32> = (0..out * inp).map(|_| g.sym()).collect();
    let x: Vec<f32> = (0..m * inp).map(|_| g.sym()).collect();
    let q = QLinear::quantize(&w, out, inp, group).unwrap();
    let deq = q.dequantize();
    for (a, b) in w.iter().zip(&deq) {
        assert!((a - b).abs() <= 1.0 / 7.0 / 2.0 * 1.01, "|w - q*s| <= s/2 (s <= 1/7)");
    }
    let mut y = vec![0.0f32; m * out];
    q.forward(&x, m, &mut y).unwrap();
    let mut want = vec![0.0f32; m * out];
    for row in 0..m {
        for col in 0..out {
            want[row * out + col] = (0..inp).map(|k| x[row * inp + k] * deq[col * inp + k]).sum();
        }
    }
    assert!(max_abs_diff(&y, &want) < 1e-5, "q4 vs dequantized f32: {}", max_abs_diff(&y, &want));
}

// ---------------------------------------------------------------------------
// the model directory

#[test]
fn safetensors_reader_rules() {
    // WHY: data_offsets count from the start of the DATA BUFFER (after the
    //      8-byte length and the header), tensors must tile that buffer
    //      exactly, and F16 and BF16 widen to the same f32 values.
    // KIND: boundary
    // CATCHES: s11, s12
    // CHAPTER: L10.1 section 5, Pitfalls
    let d = tmpdir("st");
    let p = d.join("a.safetensors");
    let f16: Vec<u8> = [1.0f32, -2.5].iter().flat_map(|&x| f32_to_f16(x).to_le_bytes()).collect();
    let bf16: Vec<u8> = [0x3F80u16, 0xC020].iter().flat_map(|x| x.to_le_bytes()).collect();
    write_tensors(&p, &format!("{{\"format\":\"tinyllm\",\"note\":\"{}\"}}", "x".repeat(200)), &[
        ("a".into(), "F16", vec![2], f16),
        ("b".into(), "BF16", vec![2], bf16),
        ("c".into(), "F32", vec![1], le32(&[7.5])),
    ]);
    let st = tl_engine::model::SafeTensors::open(&p).unwrap();
    assert_eq!(st.f32("a", &[2]).unwrap(), [1.0, -2.5]);
    assert_eq!(st.f32("b", &[2]).unwrap(), [1.0, -2.5]);
    assert_eq!(st.f32("c", &[1]).unwrap(), [7.5]);
    assert!(st.f32("c", &[2]).is_err(), "shape mismatch");
    assert_eq!(st.layout.metadata.get("format").map(String::as_str), Some("tinyllm"));
    // a gap between tensors, trailing bytes, and a header longer than the file
    let gap = safetensors(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#, &[0; 8]);
    assert!(parse_layout(&gap).is_err());
    let trailing = safetensors(r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#, &[0; 6]);
    assert!(parse_layout(&trailing).is_err());
    let mut huge = safetensors(r#"{}"#, &[]);
    huge[..8].copy_from_slice(&1000u64.to_le_bytes());
    assert!(parse_layout(&huge).is_err());
}

#[test]
fn config_refuses_what_the_engine_cannot_run() {
    // WHY: a model the engine would run wrongly (scaled RoPE, a sliding
    //      window, an unknown arch, no tokenizer) must fail at load with a
    //      message, never produce garbage mid-request.
    // KIND: boundary
    // CHAPTER: L10.1 section 5, Pitfalls
    let base = std::fs::read_to_string(tiny_llama().join("config.json")).unwrap();
    let ok = ModelConfig::from_json(&base).unwrap();
    assert_eq!((ok.n_heads, ok.n_kv_heads, ok.head_dim, ok.n_layers), (4, 2, 8, 2));
    assert!(ok.tie_word_embeddings);
    let bad = [
        base.replace("\"default\"", "\"llama3\""),
        base.replace("\"use_cache\"", "\"sliding_window\": 8, \"use_cache\""),
        base.replace("\"tl_arch\": \"llama\"", "\"tl_arch\": \"gpt2\""),
        base.replace("\"tl_tokenizer\": \"bytes\",", ""),
    ];
    for b in bad {
        assert!(ModelConfig::from_json(&b).is_err(), "should refuse:\n{b}");
    }
}

// ---------------------------------------------------------------------------
// the runner

#[test]
fn bigram_checkpoint_still_serves() {
    // WHY: the Pass 1 tracer checkpoint (tl_arch bigram, byte tokenizer)
    //      keeps serving through the new runner (D32): after `a` the logits
    //      are row 97 of W, and greedy decoding continues `bcd`.
    // KIND: conformance
    // CATCHES: s11
    // CHAPTER: L10.1 section 4
    let mut r = ModelRunner::load(&succ_bigram(), &engine_cfg(8, 16)).unwrap();
    let mut toks = bytes("a");
    for _ in 0..3 {
        let l = prefill_logits(&mut r, &toks);
        assert_eq!(l.len(), 256);
        let (t, _) = sample::sample(&l, &SamplingParams::greedy(), &[], &[], &mut Pcg32::new(0, 54));
        toks.push(t);
    }
    assert_eq!(toks, bytes("abcd"));
    let l = prefill_logits(&mut r, &bytes("a"));
    assert_eq!(l[98], 2.0);
    assert_eq!(l[97], 0.0);
}

/// The tolerance against HF's float32 forward. K and V pass through f16
/// (KV format v1), whose rounding is relative 2^-11; the logits of this
/// model reach about 20 in magnitude, so the bound is 1e-3 of that (the
/// measured error is about 7e-3; the chapter derives the bound).
const HF_ATOL: f32 = 2e-2;

#[test]
fn tiny_llama_logits_match_hf() {
    // WHY: the Candle-backed Rust forward computes what Hugging Face's
    //      float32 LlamaForCausalLM computes on the same weights (BF16 file,
    //      tied embeddings, GQA 4:2), within the f16-KV bound of section 2.
    // KIND: golden
    // CATCHES: s11, s12, s15, s16, s20, s22
    // CHAPTER: L10.1 section 2
    let mut r = ModelRunner::load(&tiny_llama(), &engine_cfg(32, 16)).unwrap();
    let doc = j::parse(&std::fs::read_to_string(fixtures().join("L7.9/tiny_logits.json")).unwrap());
    for (prompt, want) in doc.get("prompts").arr().iter().zip(doc.get("logits").arr()) {
        let want: Vec<f32> = want.arr().iter().map(|v| v.num() as f32).collect();
        let got = prefill_logits(&mut r, &bytes(prompt.str()));
        let err = max_abs_diff(&got, &want);
        assert!(err < HF_ATOL, "{:?}: max |rust - hf| = {err}", prompt.str());
    }
}

#[test]
fn tiny_llama_greedy_matches_hf() {
    // WHY: 32 greedy tokens per prompt, decoded one step at a time through
    //      the KV pool, equal HF's greedy ids under the near-tie rule
    //      (DESIGN 5.7): a step whose top-2 margin in the reference is below
    //      the tolerance may differ, and the comparison of that prompt stops.
    // KIND: golden
    // CATCHES: s11, s12, s13, s14, s15, s16, s20, s22
    // CHAPTER: L10.1 section 2
    let mut r = ModelRunner::load(&tiny_llama(), &engine_cfg(32, 16)).unwrap();
    let doc = j::parse(&std::fs::read_to_string(fixtures().join("L7.9/tiny_greedy_32.json")).unwrap());
    let margins: Vec<f64> = doc.get("margins").arr().iter().map(|v| v.num()).collect();
    let mut compared = 0;
    for (k, (prompt, want)) in doc.get("prompts").arr().iter().zip(doc.get("per_prompt").arr()).enumerate() {
        let want: Vec<u32> = want.arr().iter().map(|v| v.num() as u32).collect();
        let got = decode_greedy(&mut r, &bytes(prompt.str()), want.len());
        for i in 0..want.len() {
            if got[i] != want[i] {
                let m = margins[k * want.len() + i];
                assert!(m < 2.0 * HF_ATOL as f64, "{:?} step {i}: {} vs {} with margin {m}", prompt.str(), got[i], want[i]);
                break;
            }
            compared += 1;
        }
    }
    assert!(compared >= 64, "only {compared} greedy steps compared before a near tie");
}

#[test]
fn incremental_decode_equals_full_prefill() {
    // WHY: logits after prefill-then-decode-steps equal one prefill of the
    //      whole sequence within floating-point tolerance: positions are
    //      absolute, the cache holds the same f16 K and V either way, and the
    //      kernels are chunk-invariant. Chunked prefill (L10.3) and
    //      preemption by recompute (L10.2) rest on this.
    // KIND: differential
    // CATCHES: s13, s14
    // CHAPTER: L10.1 section 2
    let mut r = ModelRunner::load(&tiny_llama(), &engine_cfg(32, 4)).unwrap();
    let toks = bytes("The quick brown fox jumps");
    let pool = r.pool();
    let blocks = lock(&pool).unwrap().alloc(toks.len().div_ceil(4)).unwrap();
    let split = 9;
    r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &toks[..split], start: 0, blocks: &blocks }] }).unwrap();
    let mut last = Vec::new();
    for i in split..toks.len() {
        last = r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &toks[i..i + 1], start: i, blocks: &blocks }] }).unwrap().data;
    }
    for b in blocks {
        lock(&pool).unwrap().release(b).unwrap();
    }
    let whole = prefill_logits(&mut r, &toks);
    assert!(max_abs_diff(&last, &whole) <= 2e-5, "max diff {}", max_abs_diff(&last, &whole));
}

#[test]
fn batched_forward_equals_single() {
    // WHY: two sequences in one step (one prefill, one decode) get exactly
    //      the logits each gets alone: the projections run over all tokens at
    //      once and the kernels are batch-invariant, which is what makes
    //      continuous batching (L10.2) output-preserving.
    // KIND: differential
    // CATCHES: s17
    // CHAPTER: L10.1 section 2
    let mut r = ModelRunner::load(&tiny_llama(), &engine_cfg(32, 16)).unwrap();
    let a = bytes("Hello, world");
    let b = bytes("Once upon");
    let alone_a = prefill_logits(&mut r, &a);
    let alone_b = prefill_logits(&mut r, &b);
    let pool = r.pool();
    let (ba, bb) = {
        let mut p = lock(&pool).unwrap();
        (p.alloc(1).unwrap(), p.alloc(1).unwrap())
    };
    let out = r
        .forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &b, start: 0, blocks: &bb }, ForwardSeq { tokens: &a, start: 0, blocks: &ba }] })
        .unwrap();
    assert_eq!(out.rows(), 2);
    assert_eq!(out.row(0), &alone_b[..]);
    assert_eq!(out.row(1), &alone_a[..]);
    let mut p = lock(&pool).unwrap();
    for id in ba.into_iter().chain(bb) {
        p.release(id).unwrap();
    }
    assert_eq!(p.stats().used, 0, "every block went back");
}

#[test]
fn run_once_returns_its_blocks() {
    // WHY: a one-off forward (embeddings, tests) takes temporary blocks and
    //      must give every one back, or the pool drains a little per request.
    // KIND: unit
    // CATCHES: s17
    // CHAPTER: L10.1 section 4
    let mut r = ModelRunner::load(&tiny_llama(), &engine_cfg(8, 4)).unwrap();
    for _ in 0..5 {
        let (h, l) = r.run_once(&bytes("abcdefghij")).unwrap();
        assert_eq!(h.len(), 10 * 32);
        assert_eq!(l.len(), 256);
    }
    assert_eq!(lock(&r.pool()).unwrap().stats().free, 8);
}

#[test]
fn int4_runner_matches_its_dequantized_model() {
    // WHY: loading with quant int4 runs every projection through the q4
    //      kernel and gives (up to summation order) the logits of an f32
    //      model holding the dequantized weights; an int4 FILE written in the
    //      formats/safetensors.md layout loads to the same model bit for bit.
    // KIND: differential
    // CATCHES: s09, s11, s12
    // CHAPTER: L10.1 section 2
    let group = 8;
    let cfg = EngineConfig { quant: Some(Quant::Int4 { group }), ..engine_cfg(16, 16) };
    let mut q4 = ModelRunner::load(&tiny_llama(), &cfg).unwrap();
    // the f32 twin: same weights, each projection replaced by its dequantization
    let mcfg = q4.config().clone();
    let deq = match q4.weights() {
        Weights::Llama(w) => {
            let mut w = w.clone();
            for l in &mut w.layers {
                for lin in [&mut l.q, &mut l.k, &mut l.v, &mut l.o, &mut l.gate, &mut l.up, &mut l.down] {
                    let Linear::Q4(q) = lin.clone() else { panic!("quant int4 must load Q4 linears") };
                    *lin = Linear::F32 { w: q.dequantize(), out: q.out, inp: q.inp };
                }
            }
            Weights::Llama(w)
        }
        _ => panic!("tiny-llama is a llama"),
    };
    let mut f32_twin = ModelRunner::from_parts(mcfg.clone(), deq, &engine_cfg(16, 16)).unwrap();
    let prompt = bytes("The cat sat on the");
    let a = prefill_logits(&mut q4, &prompt);
    let b = prefill_logits(&mut f32_twin, &prompt);
    assert!(max_abs_diff(&a, &b) < 1e-3, "q4 vs dequantized f32: {}", max_abs_diff(&a, &b));
    // the same weights as an int4 file
    let d = tmpdir("q4");
    std::fs::copy(tiny_llama().join("config.json"), d.join("config.json")).unwrap();
    let Weights::Llama(w) = q4.weights() else { unreachable!() };
    let mut tensors: Vec<(String, &str, Vec<usize>, Vec<u8>)> = vec![
        ("model.embed_tokens.weight".into(), "F32", vec![256, 32], le32(&w.embed)),
        ("model.norm.weight".into(), "F32", vec![32], le32(&w.norm)),
    ];
    for (i, l) in w.layers.iter().enumerate() {
        tensors.push((format!("model.layers.{i}.input_layernorm.weight"), "F32", vec![32], le32(&l.input_norm)));
        tensors.push((format!("model.layers.{i}.post_attention_layernorm.weight"), "F32", vec![32], le32(&l.post_norm)));
        for (name, lin) in [("self_attn.q_proj", &l.q), ("self_attn.k_proj", &l.k), ("self_attn.v_proj", &l.v), ("self_attn.o_proj", &l.o), ("mlp.gate_proj", &l.gate), ("mlp.up_proj", &l.up), ("mlp.down_proj", &l.down)] {
            let Linear::Q4(q) = lin else { unreachable!() };
            let sc: Vec<u8> = q.scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            tensors.push((format!("model.layers.{i}.{name}.qweight"), "U8", vec![q.out, q.inp / 2], q.qweight.clone()));
            tensors.push((format!("model.layers.{i}.{name}.scales"), "F16", vec![q.out, q.inp / group], sc));
        }
    }
    write_tensors(&d.join("model.safetensors"), &format!("{{\"format\":\"tinyllm\",\"quant\":\"int4-g{group}-sym\"}}"), &tensors);
    let mut from_file = ModelRunner::load(&d, &engine_cfg(16, 16)).unwrap();
    assert_eq!(prefill_logits(&mut from_file, &prompt), a);
    // and it stays near the unquantized model: this model's weights are
    // random (no outliers to protect), and int4 with 8 columns per scale
    // keeps the logits within a relative L2 distance of about 0.2
    let mut full = ModelRunner::load(&tiny_llama(), &engine_cfg(16, 16)).unwrap();
    let c = prefill_logits(&mut full, &prompt);
    let num: f32 = a.iter().zip(&c).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let den: f32 = c.iter().map(|y| y * y).sum::<f32>().sqrt();
    assert!(num / den < 0.5, "int4 logits drifted to relative L2 distance {} from f32", num / den);
}

#[test]
fn forward_refuses_bad_batches() {
    // WHY: a block table too short for the step, or a position past the
    //      model's context, is an Err before any KV is written.
    // KIND: boundary
    // CHAPTER: L10.1 section 5, Pitfalls
    let mut r = ModelRunner::load(&tiny_llama(), &engine_cfg(8, 4)).unwrap();
    let toks = bytes("abcdefgh");
    let pool = r.pool();
    let one = lock(&pool).unwrap().alloc(1).unwrap();
    assert!(r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &toks, start: 0, blocks: &one }] }).is_err());
    assert!(r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &toks[..1], start: 256, blocks: &one }] }).is_err());
    let invalid = [u32::MAX];
    // CATCHES: s21
    assert!(r.forward(&ForwardBatch { seqs: vec![ForwardSeq { tokens: &invalid, start: 0, blocks: &one }] }).is_err());
    lock(&pool).unwrap().release(one[0]).unwrap();
}
