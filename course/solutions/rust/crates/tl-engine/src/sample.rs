//! The Rust sampler and PCG32 (L10.1): a port of L8.1 and M06.3 that must
//! return the same token id and the same logprob as your Python from the
//! same logits and seed (spec/sampling.md, spec/pcg32.md, D10, D11).
//!
//! Bit-identical results need the same operations in the same order with
//! the same arithmetic: widen to f64 first, penalties, logprobs, greedy,
//! temperature, top-k, top-p, min-p, softmax over the kept ids in ascending
//! id order, one uniform, inverse CDF. Every sum is a left-to-right loop in
//! ascending id order; `exp` and `ln` are Rust's f64 functions (the same
//! libm calls Python's `math.exp` and `math.log` make on one machine).

/// PCG-XSH-RR 64/32 (O'Neill 2014), exactly `pcg32_random_r`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

/// The LCG multiplier of PCG32.
const MULT: u64 = 6364136223846793005;

/// Purpose ids of spec/pcg32.md (sub-streams).
pub const PURPOSE_INIT: u64 = 1;
pub const PURPOSE_DROPOUT: u64 = 2;
pub const PURPOSE_SHUFFLE: u64 = 3;
pub const PURPOSE_SAMPLE: u64 = 4;
pub const PURPOSE_MUTATION: u64 = 5;

impl Pcg32 {
    /// `pcg32_srandom_r(seed, seq)`: inc = (seq << 1) | 1, state = 0, one
    /// step, state += seed, one step. The default stream is seq = 54.
    pub fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut r = Pcg32 { state: 0, inc: (seq << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    /// One step; the output is the permuted OLD state.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULT).wrapping_add(self.inc);
        let xs = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xs.rotate_right(rot)
    }

    /// 53 random bits in [0, 1): a = next >> 5, b = next >> 6,
    /// (a * 2^26 + b) * 2^-53. Two `next_u32` per call.
    pub fn uniform_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        ((a << 26) + b) as f64 * (1.0 / 9007199254740992.0)
    }

    /// (state, inc): what a disaggregated hand-off would carry (L10.6).
    pub fn state(&self) -> (u64, u64) {
        (self.state, self.inc)
    }
}

/// SplitMix64's finalizer (spec/pcg32.md `mix64`).
pub fn mix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// `child_seed(seed, p) = mix64(seed + p * 0x9E3779B97F4A7C15)`.
pub fn child_seed(seed: u64, purpose: u64) -> u64 {
    mix64(seed.wrapping_add(purpose.wrapping_mul(0x9E3779B97F4A7C15)))
}

/// `stream(seed, purpose) = pcg32_srandom_r(child_seed(seed, p), p)`. The
/// engine creates `stream(request.seed, PURPOSE_SAMPLE)` once per request.
pub fn stream(seed: u64, purpose: u64) -> Pcg32 {
    Pcg32::new(child_seed(seed, purpose), purpose)
}

/// The knobs of spec/sampling.md; `Default` turns every step off at T = 1.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub min_p: f64,
    pub repetition_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl SamplingParams {
    /// Greedy: temperature 0, nothing else.
    pub fn greedy() -> SamplingParams {
        SamplingParams { temperature: 0.0, ..SamplingParams::default() }
    }

    /// The ranges of openai-subset.v1.yaml and L8.1's `validate`.
    pub fn validate(&self) -> Result<(), String> {
        let finite = [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("min_p", self.min_p),
            ("repetition_penalty", self.repetition_penalty),
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
        ];
        if let Some((name, v)) = finite.iter().find(|(_, v)| !v.is_finite()) {
            return Err(format!("{name} must be a finite number, got {v}"));
        }
        if self.temperature < 0.0 {
            return Err(format!("temperature must be >= 0, got {}", self.temperature));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!("top_p must lie in (0, 1], got {}", self.top_p));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(format!("min_p must lie in [0, 1], got {}", self.min_p));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(format!("repetition_penalty must be > 0, got {}", self.repetition_penalty));
        }
        Ok(())
    }
}

/// Left-to-right f64 sum (the order every sum in the spec uses).
fn seq_sum(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut s = 0.0f64;
    for x in xs {
        s += x;
    }
    s
}

/// Steps 1 to 3: f64 copy, repetition penalty over the distinct ids of
/// prompt and output, presence and frequency over the output only.
pub fn apply_penalties(logits: &[f32], p: &SamplingParams, prompt: &[u32], output: &[u32]) -> Vec<f64> {
    let mut l: Vec<f64> = logits.iter().map(|&x| x as f64).collect();
    let v = l.len();
    if p.repetition_penalty != 1.0 {
        let mut seen = vec![false; v];
        for &t in prompt.iter().chain(output) {
            if (t as usize) < v {
                seen[t as usize] = true;
            }
        }
        for (i, s) in seen.iter().enumerate() {
            if *s {
                l[i] = if l[i] > 0.0 { l[i] / p.repetition_penalty } else { l[i] * p.repetition_penalty };
            }
        }
    }
    if p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 {
        let mut count = vec![0u32; v];
        for &t in output {
            if (t as usize) < v {
                count[t as usize] += 1;
            }
        }
        for (i, &c) in count.iter().enumerate() {
            if c > 0 {
                l[i] = l[i] - p.frequency_penalty * c as f64 - p.presence_penalty;
            }
        }
    }
    l
}

/// `log_softmax(l)[i]` over all ids: (l[i] - M) - ln(sum exp(l - M)).
pub fn logprob_of(l: &[f64], i: usize) -> f64 {
    let m = l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let z = seq_sum(l.iter().map(|&x| (x - m).exp()));
    (l[i] - m) - z.ln()
}

/// The id of the largest value, ties to the lowest id.
pub fn argmax(l: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in l.iter().enumerate() {
        if x > l[best] {
            best = i;
        }
    }
    best
}

/// Step 9 over the ids in `keep`, in ascending id order: exp(t - M) / Z.
fn softmax_over(t: &[f64], keep: &[usize]) -> Vec<(usize, f64)> {
    let mut ids = keep.to_vec();
    ids.sort_unstable();
    let m = ids.iter().map(|&i| t[i]).fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = ids.iter().map(|&i| (t[i] - m).exp()).collect();
    let z = seq_sum(e.iter().copied());
    ids.into_iter().zip(e).map(|(i, x)| (i, x / z)).collect()
}

/// Steps 5 to 9 on penalized logits (T > 0): the kept ids, ascending, with
/// their probabilities.
pub fn distribution(l: &[f64], p: &SamplingParams) -> Vec<(usize, f64)> {
    let t: Vec<f64> = l.iter().map(|&x| x / p.temperature).collect();
    // (t desc, id asc); ids at -inf can never be drawn
    let mut order: Vec<usize> = (0..t.len()).filter(|&i| t[i] != f64::NEG_INFINITY).collect();
    order.sort_by(|&a, &b| t[b].partial_cmp(&t[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    if p.top_k > 0 && p.top_k < order.len() {
        order.truncate(p.top_k);
    }
    // q of the current kept set, indexed by id
    let dense = |q: Vec<(usize, f64)>| {
        let mut d = vec![0.0f64; t.len()];
        for (i, v) in q {
            d[i] = v;
        }
        d
    };
    if p.top_p < 1.0 {
        let q = dense(softmax_over(&t, &order));
        let mut s = 0.0;
        let mut cut = order.len();
        for (n, &i) in order.iter().enumerate() {
            s += q[i];
            if s >= p.top_p {
                cut = n + 1;
                break;
            }
        }
        order.truncate(cut);
    }
    if p.min_p > 0.0 {
        let q = dense(softmax_over(&t, &order));
        let qmax = order.iter().map(|&i| q[i]).fold(0.0f64, f64::max);
        order.retain(|&i| q[i] >= p.min_p * qmax);
    }
    softmax_over(&t, &order)
}

/// Steps 10 and 11: walk the kept ids in ascending order adding q; the
/// first id with u < c. If rounding leaves c at or below u, the largest id
/// with q > 0.
pub fn inverse_cdf(q: &[(usize, f64)], u: f64) -> usize {
    let mut c = 0.0;
    for &(i, qi) in q {
        c += qi;
        if u < c {
            return i;
        }
    }
    q.iter().rev().find(|x| x.1 > 0.0).or(q.last()).map(|x| x.0).unwrap_or(0)
}

/// One token from `logits` (spec/sampling.md): its id and its logprob
/// (log-softmax after penalties, before temperature and filtering). Greedy
/// (T = 0) takes no draw; otherwise exactly one `uniform_f64`.
pub fn sample(logits: &[f32], p: &SamplingParams, prompt: &[u32], output: &[u32], rng: &mut Pcg32) -> (u32, f64) {
    let l = apply_penalties(logits, p, prompt, output);
    if p.temperature == 0.0 {
        let tok = argmax(&l);
        return (tok as u32, logprob_of(&l, tok));
    }
    let q = distribution(&l, p);
    let u = rng.uniform_f64();
    let tok = inverse_cdf(&q, u);
    (tok as u32, logprob_of(&l, tok))
}
