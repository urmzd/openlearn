// Parity driver for `sampler`, implementation `rust` (L10.1): an example of
// the harness crate ol-tests. Reads one case per stdin line
// {"logits": [repr strings], "params": {...}, "prompt": [...], "seed", "n"}
// and prints {"ids": [...], "logprobs": [...]}: n tokens drawn one after
// another with tl_engine::sample::sample from stream(seed, sample), each
// appended to the output history (spec/sampling.md).
include!("../parity/ol_parity.rs");

use tl_engine::sample::{sample, stream, SamplingParams, PURPOSE_SAMPLE};

/// The strings of the JSON array under `key` (no escapes in the logits).
fn strings(line: &str, key: &str) -> Vec<String> {
    let s = ssp_find(line, key).and_then(|s| s.strip_prefix('[')).expect("array");
    let body = &s[..s.find(']').expect("]")];
    body.split(',').map(|x| x.trim().trim_matches('"').to_string()).filter(|x| !x.is_empty()).collect()
}

fn main() {
    for line in ssp_lines() {
        if line.trim().is_empty() {
            continue;
        }
        let logits: Vec<f32> = strings(&line, "logits").iter().map(|s| s.parse::<f64>().expect("a float repr") as f32).collect();
        let p = SamplingParams {
            temperature: ssp_f64(&line, "temperature").unwrap_or(1.0),
            top_k: ssp_f64(&line, "top_k").unwrap_or(0.0) as usize,
            top_p: ssp_f64(&line, "top_p").unwrap_or(1.0),
            min_p: ssp_f64(&line, "min_p").unwrap_or(0.0),
            repetition_penalty: ssp_f64(&line, "repetition_penalty").unwrap_or(1.0),
            presence_penalty: ssp_f64(&line, "presence_penalty").unwrap_or(0.0),
            frequency_penalty: ssp_f64(&line, "frequency_penalty").unwrap_or(0.0),
        };
        let prompt: Vec<u32> = ssp_f64s(&line, "prompt").unwrap_or_default().into_iter().map(|x| x as u32).collect();
        let n = ssp_u64(&line, "n").unwrap_or(1) as usize;
        let mut rng = stream(ssp_u64(&line, "seed").expect("seed"), PURPOSE_SAMPLE);
        let mut ids: Vec<u32> = Vec::with_capacity(n);
        let mut lps: Vec<String> = Vec::with_capacity(n);
        for _ in 0..n {
            let (id, lp) = sample(&logits, &p, &prompt, &ids, &mut rng);
            ids.push(id);
            lps.push(format!("{lp:?}"));
        }
        let ids: Vec<String> = ids.iter().map(|x| x.to_string()).collect();
        println!("{{\"ids\":[{}],\"logprobs\":[{}]}}", ids.join(","), lps.join(","));
    }
}
