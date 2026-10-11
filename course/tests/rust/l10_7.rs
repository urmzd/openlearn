//! L10.7 course tests: serving metrics, SLO histograms, OTel spans and
//! propagation (tl-serve metrics, telemetry).
//!
//! Annotated exemplars (DESIGN 5.12). The exposition is parsed by the small
//! Prometheus text reader in `mod prom` and compared with the engine
//! instruments of course/contracts/otel/metrics.yaml, which
//! course/fixtures/L10.7/engine_metrics.json lists (course/oracle/L10.7/
//! engine_metrics.py). OTLP JSON is read with `mod j`. The export test runs
//! a one-request collector on 127.0.0.1:0 in a thread.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::thread;
use std::time::Duration;

use tl_serve::metrics::{self, EngineGauges, EngineMetrics, Histogram, Kind, LabelSpec, MetricError, Registry, RequestRecord};
use tl_serve::telemetry::{self, AttrValue, IdGen, RequestTiming, Span, SpanKind, TraceContext};

fn fixtures() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol"))
}

fn req(op: &str, model: &str, ttft: Option<f64>, e2e: f64, tokens: usize, err: Option<&str>) -> RequestRecord {
    RequestRecord { operation: op.into(), model: model.into(), role: "unified".into(), ttft_s: ttft, e2e_s: e2e, output_tokens: tokens, error_type: err.map(String::from) }
}

/// A Prometheus text-format reader: families with their TYPE and HELP, and
/// samples (name, labels, value). Independent of the code under test.
mod prom {
    use std::collections::BTreeMap;

    #[derive(Debug, Default)]
    pub struct Family {
        pub kind: String,
        pub help: String,
        pub samples: Vec<(String, BTreeMap<String, String>, f64)>,
    }

    pub fn parse(text: &str) -> BTreeMap<String, Family> {
        let mut fams: BTreeMap<String, Family> = BTreeMap::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            if let Some(r) = line.strip_prefix("# HELP ") {
                let (n, h) = r.split_once(' ').unwrap_or((r, ""));
                fams.entry(n.into()).or_default().help = h.into();
                continue;
            }
            if let Some(r) = line.strip_prefix("# TYPE ") {
                let (n, t) = r.split_once(' ').expect("TYPE line");
                fams.entry(n.into()).or_default().kind = t.into();
                continue;
            }
            assert!(!line.starts_with('#'), "unknown comment {line:?}");
            let (name, labels, rest) = match line.find('{') {
                Some(b) => {
                    let (labels, after) = labels(&line[b + 1..]);
                    (line[..b].to_string(), labels, after.to_string())
                }
                None => {
                    let (n, v) = line.split_once(' ').expect("sample value");
                    (n.to_string(), BTreeMap::new(), v.to_string())
                }
            };
            let v = match rest.trim() {
                "+Inf" => f64::INFINITY,
                "-Inf" => f64::NEG_INFINITY,
                x => x.parse().unwrap_or_else(|_| panic!("bad value in {line:?}")),
            };
            let fam = ["_bucket", "_sum", "_count"].iter().find_map(|s| name.strip_prefix("").and_then(|n| n.strip_suffix(s))).filter(|base| fams.get(*base).is_some_and(|f| f.kind == "histogram")).unwrap_or(&name).to_string();
            fams.get_mut(&fam).unwrap_or_else(|| panic!("sample {name} before its TYPE line")).samples.push((name, labels, v));
        }
        fams
    }

    /// Parses `k="v",k2="v2"} rest` with the exposition escapes.
    fn labels(s: &str) -> (BTreeMap<String, String>, &str) {
        let b = s.as_bytes();
        let mut i = 0;
        let mut out = BTreeMap::new();
        loop {
            if b[i] == b'}' {
                return (out, &s[i + 1..]);
            }
            if b[i] == b',' {
                i += 1;
                continue;
            }
            let eq = s[i..].find('=').unwrap() + i;
            let key = s[i..eq].to_string();
            assert!(key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'), "label name {key:?}");
            assert_eq!(b[eq + 1], b'"');
            let mut v = String::new();
            let mut k = eq + 2;
            loop {
                match b[k] {
                    b'\\' => {
                        v.push(match b[k + 1] {
                            b'n' => '\n',
                            c => c as char,
                        });
                        k += 2;
                    }
                    b'"' => break,
                    _ => {
                        let ch = s[k..].chars().next().unwrap();
                        v.push(ch);
                        k += ch.len_utf8();
                    }
                }
            }
            out.insert(key, v);
            i = k + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// metrics

#[test]
fn hand_example_histogram() {
    // WHY: the chapter's worked histogram. Bounds [0.1, 0.5, 1.0];
    //      observations 0.05, 0.1, 0.3, 2.0. A value equal to a bound belongs
    //      to that bucket (le means <=), so the per-bucket counts are 2, 1,
    //      0, 1 (+Inf), the wire buckets are cumulative 2, 3, 3, 4, the sum
    //      2.45 and the count 4, rendered exactly as below.
    // KIND: unit
    // CATCHES: s01, s05, m01, m05
    // CHAPTER: L10.7 section 3
    let mut h = Histogram::new(&[0.1, 0.5, 1.0]);
    for v in [0.05, 0.1, 0.3, 2.0] {
        h.observe(v);
    }
    assert_eq!(h.counts, vec![2, 1, 0, 1]);
    assert_eq!(h.cumulative(), vec![2, 3, 3, 4]);
    assert_eq!(h.count, 4);
    assert!((h.sum - 2.45).abs() < 1e-12);
    static B: &[f64] = &[0.1, 0.5, 1.0];
    let mut r = Registry::new();
    r.register("x_seconds", "Example.", Kind::Histogram, vec![LabelSpec { key: "op", values: Some(&["chat"]), capped: false }], B);
    for v in [0.05, 0.1, 0.3, 2.0] {
        r.observe("x_seconds", &[("op", "chat")], v).unwrap();
    }
    let want = "# HELP x_seconds Example.\n# TYPE x_seconds histogram\n\
x_seconds_bucket{op=\"chat\",le=\"0.1\"} 2\nx_seconds_bucket{op=\"chat\",le=\"0.5\"} 3\nx_seconds_bucket{op=\"chat\",le=\"1\"} 3\n\
x_seconds_bucket{op=\"chat\",le=\"+Inf\"} 4\nx_seconds_sum{op=\"chat\"} 2.45\nx_seconds_count{op=\"chat\"} 4\n";
    assert_eq!(r.render(), want);
}

#[test]
fn names_and_buckets_match_the_contract() {
    // WHY: dashboards, SLO rules, and drills query these exact names
    //      (otel/metrics.yaml): every engine instrument is served with its
    //      Prometheus name and type, histograms with exactly the contract's
    //      buckets, series only with the contract's label keys, and nothing
    //      the contract does not list.
    // KIND: conformance
    // CATCHES: s02, s03, m02
    // CHAPTER: L10.7 section 4
    let fx = j::parse(&std::fs::read_to_string(fixtures().join("L10.7/engine_metrics.json")).unwrap());
    let mut m = EngineMetrics::new();
    m.record_request(&req("chat", "smol", Some(0.12), 0.9, 8, None)).unwrap();
    m.record_request(&req("text_completion", "smol", None, 0.02, 0, Some("context_length_exceeded"))).unwrap();
    m.record_http("POST", "/v1/chat/completions", 200, 0.9).unwrap();
    m.set_engine(&EngineGauges { kv_free: 10, kv_used: 4, kv_cached: 2, kv_evictions: 3, queue_depth: 1, active_sequences: 2, batch_tokens: 64, prefix_hit_ratio: 0.5, spec_accept_rate: 0.7, preemptions: 1 }).unwrap();
    m.record_kv_transfer(true, 4096, 3, 1).unwrap();
    let fams = prom::parse(&m.render());
    let mut want_names = Vec::new();
    for ins in fx.get("instruments").arr() {
        let name = ins.get("prometheus").str();
        want_names.push(name.to_string());
        let f = fams.get(name).unwrap_or_else(|| panic!("{name} is not served"));
        assert_eq!(f.kind, ins.get("type").str(), "{name}: TYPE");
        assert!(!f.help.is_empty(), "{name}: HELP");
        let keys: Vec<String> = ins.get("labels").obj().iter().map(|(k, _)| k.clone()).collect();
        for (sname, labels, _) in &f.samples {
            for k in labels.keys() {
                assert!(keys.contains(k) || (k == "le" && sname.ends_with("_bucket")), "{sname}: label {k} is not in the contract {keys:?}");
            }
        }
        if ins.get("type").str() == "histogram" {
            let bounds: Vec<f64> = ins.get("buckets").arr().iter().map(|b| b.f64()).collect();
            let series: Vec<&BTreeMap<String, String>> = f.samples.iter().filter(|(n, _, _)| n.ends_with("_count")).map(|(_, l, _)| l).collect();
            assert!(!series.is_empty(), "{name}: no series after a recorded request");
            for s in series {
                let les: Vec<f64> = f.samples.iter().filter(|(n, l, _)| n.ends_with("_bucket") && l.iter().all(|(k, v)| k == "le" || s.get(k) == Some(v)) && l.len() == s.len() + 1)
                    .map(|(_, l, _)| if l["le"] == "+Inf" { f64::INFINITY } else { l["le"].parse().unwrap() }).collect();
                let mut want = bounds.clone();
                want.push(f64::INFINITY);
                assert_eq!(les, want, "{name} {s:?}: bucket bounds");
            }
        }
    }
    let mut served: Vec<String> = fams.keys().cloned().collect();
    served.sort();
    want_names.sort();
    assert_eq!(served, want_names, "served families vs the contract's engine instruments");
}

#[test]
fn tpot_hand_example() {
    // WHY: TPOT = (E2E - TTFT) / (output tokens - 1): a request with TTFT
    //      0.2 s, E2E 1.0 s, and 5 tokens has TPOT 0.8 / 4 = 0.2 s, which
    //      lands in the le="0.2" bucket. One token has no TPOT; no token has
    //      no TTFT; a failed request carries error_type on its duration.
    // KIND: unit
    // CATCHES: s01, s03, s04, m04
    // CHAPTER: L10.7 section 3
    let mut m = EngineMetrics::new();
    m.record_request(&req("chat", "m", Some(0.2), 1.0, 5, None)).unwrap();
    m.record_request(&req("chat", "m", Some(0.1), 0.1, 1, None)).unwrap();
    m.record_request(&req("chat", "m", None, 0.05, 0, Some("no_capacity"))).unwrap();
    let f = prom::parse(&m.render());
    let get = |fam: &str, sample: &str, le: Option<&str>| -> f64 {
        f[fam].samples.iter().find(|(n, l, _)| n == sample && le.is_none_or(|x| l.get("le").map(String::as_str) == Some(x))).map(|s| s.2).unwrap_or(-1.0)
    };
    assert_eq!(get(metrics::TPOT, &format!("{}_count", metrics::TPOT), None), 1.0, "only the 5-token request has a TPOT");
    assert!((get(metrics::TPOT, &format!("{}_sum", metrics::TPOT), None) - 0.2).abs() < 1e-12);
    assert_eq!(get(metrics::TPOT, &format!("{}_bucket", metrics::TPOT), Some("0.2")), 1.0);
    assert_eq!(get(metrics::TPOT, &format!("{}_bucket", metrics::TPOT), Some("0.15")), 0.0);
    assert_eq!(get(metrics::TTFT, &format!("{}_count", metrics::TTFT), None), 2.0, "the failed request had no first token");
    let errs: Vec<_> = f[metrics::REQUEST_DURATION].samples.iter().filter(|(n, l, _)| n.ends_with("_count") && l.get("error_type").map(String::as_str) == Some("no_capacity")).collect();
    assert_eq!(errs.len(), 1, "the failure is its own series with error_type");
}

#[test]
fn histogram_count_equals_requests() {
    // WHY: a histogram's _count is the number of requests it saw, its +Inf
    //      bucket equals _count (TTFTs above 10 s, the top bound, included),
    //      its buckets never decrease, and _sum is the total: 500 seeded
    //      requests, every one checked against the test's own count.
    // KIND: property
    // CATCHES: s05, m01, m05
    // CHAPTER: L10.7 section 2
    let mut m = EngineMetrics::new();
    let mut x: u64 = 0x2545F4914F6CDD1D;
    let (mut n_ttft, mut sum_ttft) = (0u64, 0.0f64);
    for _ in 0..500 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let ttft = (x % 12000) as f64 / 1000.0; // some above the top bound (10 s)
        let tokens = (x >> 20) as usize % 40;
        let has = tokens > 0;
        m.record_request(&req("chat", "m", has.then_some(ttft), ttft + 0.5, tokens, None)).unwrap();
        if has {
            n_ttft += 1;
            sum_ttft += ttft;
        }
    }
    let f = prom::parse(&m.render());
    let fam = &f[metrics::TTFT];
    let count = fam.samples.iter().find(|(n, _, _)| n.ends_with("_count")).unwrap().2;
    let sum = fam.samples.iter().find(|(n, _, _)| n.ends_with("_sum")).unwrap().2;
    assert_eq!(count as u64, n_ttft);
    assert!((sum - sum_ttft).abs() < 1e-6, "sum {sum} vs {sum_ttft}");
    let buckets: Vec<(f64, f64)> = fam.samples.iter().filter(|(n, _, _)| n.ends_with("_bucket")).map(|(_, l, v)| (if l["le"] == "+Inf" { f64::INFINITY } else { l["le"].parse().unwrap() }, *v)).collect();
    assert!(buckets.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1), "buckets ascend in le and never decrease: {buckets:?}");
    assert_eq!(buckets.last().unwrap().1, count, "+Inf equals _count");
    let total = f[metrics::REQUEST_DURATION].samples.iter().find(|(n, _, _)| n.ends_with("_count")).unwrap().2;
    assert_eq!(total, 500.0);
}

#[test]
fn counters_never_go_down() {
    // WHY: Prometheus rate() assumes counters only rise (a drop reads as a
    //      restart). A running total lower than the last one, or a negative
    //      increment, is refused; scraping twice never shows a smaller value.
    // KIND: fault
    // CATCHES: s06, m06
    // CHAPTER: L10.7 section 5, Pitfalls
    let mut m = EngineMetrics::new();
    let mut g = EngineGauges { kv_evictions: 5, preemptions: 2, ..Default::default() };
    m.set_engine(&g).unwrap();
    let first = prom::parse(&m.render());
    g.kv_evictions = 3;
    assert!(matches!(m.set_engine(&g), Err(MetricError::Decrease(_))));
    assert!(m.reg.inc(metrics::KV_TRANSFER_BYTES, &[("direction", "sent")], -1.0).is_err());
    assert!(m.reg.inc(metrics::KV_TRANSFER_BYTES, &[("direction", "sent")], f64::NAN).is_err());
    m.record_kv_transfer(false, 100, 1, 0).unwrap();
    m.record_kv_transfer(false, 50, 2, 1).unwrap();
    let second = prom::parse(&m.render());
    for name in [metrics::KV_EVICTIONS, metrics::PREEMPTIONS] {
        assert!(second[name].samples[0].2 >= first[name].samples[0].2, "{name} went down");
    }
    assert_eq!(m.reg.value(metrics::KV_TRANSFER_BYTES, &[("direction", "received")]), Some(150.0));
    assert_eq!(m.reg.value(metrics::KV_TRANSFER_BLOCKS, &[("outcome", "deduped")]), Some(1.0));
    assert_eq!(m.reg.value(metrics::KV_EVICTIONS, &[]), Some(5.0), "the refused total left the counter unchanged");
}

#[test]
fn counters_and_gauges_start_at_zero() {
    // WHY: a counter that appears only after its first event makes
    //      rate() and absent() alerts misfire on a fresh engine; the
    //      unlabeled ones are served at 0 from the first scrape. The KV
    //      gauges always add up to the pool: free + used + cached.
    // KIND: boundary
    // CATCHES: m07
    // CHAPTER: L10.7 section 4
    let mut m = EngineMetrics::new();
    let f = prom::parse(&m.render());
    for name in [metrics::KV_EVICTIONS, metrics::PREEMPTIONS, metrics::QUEUE_DEPTH, metrics::ACTIVE_SEQUENCES] {
        assert_eq!(f[name].samples.len(), 1, "{name} is served before any event");
        assert_eq!(f[name].samples[0].2, 0.0);
    }
    m.set_engine(&EngineGauges { kv_free: 10, kv_used: 4, kv_cached: 2, ..Default::default() }).unwrap();
    let f = prom::parse(&m.render());
    let kv: f64 = f[metrics::KV_BLOCKS].samples.iter().map(|s| s.2).sum();
    assert_eq!(kv, 16.0);
    let states: Vec<&str> = f[metrics::KV_BLOCKS].samples.iter().map(|s| s.1["state"].as_str()).collect();
    assert_eq!(states, vec!["cached", "free", "used"]);
}

#[test]
fn gauges_from_engine_stats() {
    // WHY: the gauges come from the engine loop's own counters (L10.5):
    //      waiting requests are tl_engine_queue_depth and running ones
    //      tl_engine_active_sequences; swapping them points the queueing
    //      alerts at the wrong number.
    // KIND: unit
    // CATCHES: m19
    // CHAPTER: L10.7 section 4
    let s = tl_engine::engine::EngineStats { steps: 9, waiting: 3, running: 2, preemptions: 4, batch_tokens: 64, kv_total: 16, kv_free: 10, kv_used: 4, kv_cached: 2, kv_evictions: 6, prefix_hit_rate: 0.25 };
    let g = EngineGauges::from_stats(&s, 0.75);
    assert_eq!(g, EngineGauges { kv_free: 10, kv_used: 4, kv_cached: 2, kv_evictions: 6, queue_depth: 3, active_sequences: 2, batch_tokens: 64, prefix_hit_ratio: 0.25, spec_accept_rate: 0.75, preemptions: 4 });
    let mut m = EngineMetrics::new();
    m.set_engine(&g).unwrap();
    assert_eq!((m.reg.value(metrics::QUEUE_DEPTH, &[]), m.reg.value(metrics::ACTIVE_SEQUENCES, &[]), m.reg.value(metrics::SPEC_ACCEPT_RATE, &[])), (Some(3.0), Some(2.0), Some(0.75)));
}

#[test]
fn label_cap_and_allowed_values() {
    // WHY: an unbounded label (a model name per request) makes Prometheus
    //      store a series per value; a capped label keeps 64 values and
    //      folds the rest into _other. A label with a value list refuses
    //      anything else, and every declared label must be given.
    // KIND: boundary
    // CATCHES: s07, m03, m08
    // CHAPTER: L10.7 section 5, Pitfalls
    let mut m = EngineMetrics::new();
    for i in 0..70 {
        m.record_request(&req("chat", &format!("model-{i}"), Some(0.1), 0.2, 2, None)).unwrap();
    }
    let f = prom::parse(&m.render());
    let models: std::collections::BTreeSet<String> = f[metrics::TTFT].samples.iter().map(|s| s.1["gen_ai_request_model"].clone()).collect();
    assert_eq!(models.len(), 65, "64 models plus _other");
    assert!(models.contains("_other") && models.contains("model-63") && !models.contains("model-64"));
    let other = f[metrics::TTFT].samples.iter().find(|s| s.0.ends_with("_count") && s.1["gen_ai_request_model"] == "_other").unwrap().2;
    assert_eq!(other, 6.0, "the six models past the cap share _other");
    assert!(matches!(m.record_request(&req("summarize", "m", Some(0.1), 0.2, 2, None)), Err(MetricError::Value { .. })));
    assert!(matches!(m.reg.set(metrics::KV_BLOCKS, &[], 1.0), Err(MetricError::Labels(_))));
    assert!(matches!(m.reg.observe(metrics::QUEUE_DEPTH, &[], 1.0), Err(MetricError::Kind(_))));
    assert!(matches!(m.reg.set("tl_engine_nope", &[], 1.0), Err(MetricError::UnknownMetric(_))));
}

#[test]
fn label_values_are_escaped() {
    // WHY: a model id is user data: a quote, a backslash, or a newline in it
    //      must be escaped (\" \\ \n), or the whole scrape fails to parse and
    //      every alert goes blind.
    // KIND: boundary
    // CATCHES: s08, m03
    // CHAPTER: L10.7 section 5, Pitfalls
    let mut m = EngineMetrics::new();
    let name = "we\"ird\\model\nx";
    m.record_request(&req("chat", name, Some(0.1), 0.2, 2, None)).unwrap();
    let text = m.render();
    assert!(text.contains(r#"gen_ai_request_model="we\"ird\\model\nx""#), "escaped value missing");
    let f = prom::parse(&text);
    assert!(f[metrics::TTFT].samples.iter().any(|s| s.1["gen_ai_request_model"] == name));
}

// ---------------------------------------------------------------------------
// traces

const W3C: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

#[test]
fn traceparent_hand_example() {
    // WHY: the W3C specification's own example: version 00, trace id
    //      4bf9...4736, parent span 00f0...02b7, flags 01 (sampled). Parsing
    //      and writing it back must be lossless.
    // KIND: unit
    // CATCHES: s09, m09
    // CHAPTER: L10.7 section 3
    let c = TraceContext::parse(W3C).unwrap();
    assert_eq!(c.trace_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(c.span_hex(), "00f067aa0ba902b7");
    assert!(c.sampled());
    assert_eq!(c.header(), W3C);
    let unsampled = TraceContext::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00").unwrap();
    assert!(!unsampled.sampled());
}

#[test]
fn traceparent_rejects_invalid_headers() {
    // WHY: an invalid traceparent must start a NEW trace, never be half
    //      trusted: uppercase hex, an all-zero trace or span id, version ff,
    //      a version-00 header with extra fields, wrong lengths. A future
    //      version with extra fields is accepted (forward compatibility).
    // KIND: boundary
    // CATCHES: s10, m10
    // CHAPTER: L10.7 section 5, Pitfalls
    for bad in [
        "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
        "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba9g2b7-01",
        "",
    ] {
        assert_eq!(TraceContext::parse(bad), None, "{bad:?} must be refused");
    }
    assert!(TraceContext::parse("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-later").is_some());
}

fn timing(tokens: usize) -> RequestTiming {
    let ms = 1_000_000u64;
    RequestTiming {
        route: "/v1/chat/completions".into(),
        operation: "chat".into(),
        model: "smol".into(),
        status: 200,
        error_type: None,
        received_ns: 1000 * ms,
        admitted_ns: 1004 * ms,
        first_token_ns: 1050 * ms,
        end_ns: 1050 * ms + tokens as u64 * 20 * ms,
        priority: 2,
        prompt_tokens: 12,
        prefix_hit_tokens: 8,
        chunks: 1,
        token_times_ns: (0..tokens as u64).map(|i| 1050 * ms + i * 20 * ms).collect(),
        finish_reasons: vec!["length".into()],
        spec_accept_rate: 0.5,
    }
}

fn attr<'a>(s: &'a Span, k: &str) -> &'a AttrValue {
    s.attributes.iter().find(|(a, _)| a == k).map(|(_, v)| v).unwrap_or_else(|| panic!("{}: attribute {k} missing", s.name))
}

#[test]
fn request_span_tree() {
    // WHY: semconv.md's serving trace on the engine: a SERVER span that
    //      continues the caller's trace (the gateway's proxy span is its
    //      parent), and engine.queue, engine.prefill, engine.decode as its
    //      INTERNAL children over [received, admitted], [admitted, first
    //      token], [first token, end], each with its required attributes;
    //      decode has a `token` event at every 32nd token.
    // KIND: conformance
    // CATCHES: s11, s12, m11, m12
    // CHAPTER: L10.7 section 2
    let incoming = TraceContext::parse(W3C).unwrap();
    let mut ids = IdGen::new(1);
    let t = timing(70);
    let spans = telemetry::request_spans(&mut ids, Some(&incoming), &t);
    let names: Vec<&str> = spans.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["POST /v1/chat/completions", "engine.queue", "engine.prefill", "engine.decode"]);
    let server = &spans[0];
    assert_eq!((server.kind, server.trace_id, server.parent_span_id), (SpanKind::Server, incoming.trace_id, Some(incoming.span_id)));
    assert_eq!((server.start_ns, server.end_ns), (t.received_ns, t.end_ns));
    for (k, v) in [("http.request.method", AttrValue::Str("POST".into())), ("http.route", AttrValue::Str("/v1/chat/completions".into())), ("http.response.status_code", AttrValue::Int(200)), ("gen_ai.operation.name", AttrValue::Str("chat".into())), ("gen_ai.request.model", AttrValue::Str("smol".into()))] {
        assert_eq!(attr(server, k), &v, "SERVER {k}");
    }
    for child in &spans[1..] {
        assert_eq!((child.kind, child.trace_id, child.parent_span_id), (SpanKind::Internal, incoming.trace_id, Some(server.span_id)), "{}", child.name);
        assert_ne!(child.span_id, server.span_id);
    }
    let (q, p, d) = (&spans[1], &spans[2], &spans[3]);
    assert_eq!((q.start_ns, q.end_ns, p.start_ns, p.end_ns, d.start_ns, d.end_ns), (t.received_ns, t.admitted_ns, t.admitted_ns, t.first_token_ns, t.first_token_ns, t.end_ns));
    assert_eq!(attr(q, "tl.engine.queue_ms"), &AttrValue::Float(4.0));
    assert_eq!(attr(q, "tl.engine.priority"), &AttrValue::Int(2));
    assert_eq!((attr(p, "gen_ai.usage.input_tokens"), attr(p, "tl.engine.prefix_hit_tokens"), attr(p, "tl.engine.chunks")), (&AttrValue::Int(12), &AttrValue::Int(8), &AttrValue::Int(1)));
    assert_eq!(attr(d, "gen_ai.usage.output_tokens"), &AttrValue::Int(70));
    assert_eq!(attr(d, "gen_ai.response.finish_reasons"), &AttrValue::StrArray(vec!["length".into()]));
    assert_eq!(attr(d, "tl.engine.spec_accept_rate"), &AttrValue::Float(0.5));
    let ev: Vec<(u64, &AttrValue)> = d.events.iter().map(|e| (e.time_ns, &e.attributes[0].1)).collect();
    assert_eq!(ev, vec![(t.token_times_ns[31], &AttrValue::Int(32)), (t.token_times_ns[63], &AttrValue::Int(64))]);
    assert!(server.error.is_none());
    // Without a caller's context the SERVER span starts a new trace.
    let fresh = telemetry::request_spans(&mut ids, None, &t);
    assert_eq!(fresh[0].parent_span_id, None);
    assert_ne!(fresh[0].trace_id, incoming.trace_id);
    assert_eq!(fresh[1].trace_id, fresh[0].trace_id);
}

#[test]
fn failed_request_sets_error_status() {
    // WHY: a 5xx (or any request with an error code) is a failing span:
    //      status ERROR with error.type, so trace search finds the failures
    //      a drill caused.
    // KIND: unit
    // CATCHES: m13
    // CHAPTER: L10.7 section 4
    let mut t = timing(0);
    t.status = 503;
    t.error_type = Some("no_capacity".into());
    let spans = telemetry::request_spans(&mut IdGen::new(2), None, &t);
    assert_eq!(spans[0].error.as_deref(), Some("no_capacity"));
    assert_eq!(attr(&spans[0], "error.type"), &AttrValue::Str("no_capacity".into()));
    let mut ok = timing(3);
    ok.status = 429;
    assert!(telemetry::request_spans(&mut IdGen::new(2), None, &ok)[0].error.is_none(), "a 429 is the client's problem, not an error span");
}

#[test]
fn otlp_json_shape() {
    // WHY: the collector, Jaeger, and Tempo read OTLP/HTTP JSON: ids as
    //      lowercase hex (32 and 16 digits), parentSpanId only on children,
    //      kind as the enum number, times and 64-bit ints as STRINGS,
    //      doubles as numbers, arrays as arrayValue, status code 2 on error,
    //      the resource's service.name.
    // KIND: conformance
    // CATCHES: s11, s12, s13, m14
    // CHAPTER: L10.7 section 4
    let incoming = TraceContext::parse(W3C).unwrap();
    let mut t = timing(33);
    t.status = 500;
    let spans = telemetry::request_spans(&mut IdGen::new(3), Some(&incoming), &t);
    let res = telemetry::engine_resource("forge", "0.4.0", "decode");
    let doc = j::parse(&telemetry::otlp_json(&res, "tl-serve", &spans));
    let rs = &doc.get("resourceSpans").arr()[0];
    let ra = rs.get("resource").get("attributes").arr();
    assert_eq!((ra[0].get("key").str(), ra[0].get("value").get("stringValue").str()), ("service.name", "forge-engine"));
    let ss = &rs.get("scopeSpans").arr()[0];
    assert_eq!(ss.get("scope").get("name").str(), "tl-serve");
    let js = ss.get("spans").arr();
    assert_eq!(js.len(), 4);
    let server = &js[0];
    assert_eq!(server.get("traceId").str(), "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(server.get("parentSpanId").str(), "00f067aa0ba902b7");
    assert_eq!(server.get("spanId").str().len(), 16);
    assert!(server.get("spanId").str().chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)));
    assert_eq!(server.get("kind").num(), 2.0);
    assert_eq!(server.get("startTimeUnixNano").str(), t.received_ns.to_string());
    assert_eq!(server.get("status").get("code").num(), 2.0);
    let code = server.get("attributes").arr().iter().find(|a| a.get("key").str() == "http.response.status_code").unwrap();
    assert_eq!(code.get("value").get("intValue").str(), "500");
    assert_eq!(js[1].get("kind").num(), 1.0);
    assert_eq!(js[1].get("parentSpanId").str(), server.get("spanId").str());
    let decode = &js[3];
    let fr = decode.get("attributes").arr().iter().find(|a| a.get("key").str() == "gen_ai.response.finish_reasons").unwrap();
    assert_eq!(fr.get("value").get("arrayValue").get("values").arr()[0].get("stringValue").str(), "length");
    let rate = decode.get("attributes").arr().iter().find(|a| a.get("key").str() == "tl.engine.spec_accept_rate").unwrap();
    assert_eq!(rate.get("value").get("doubleValue").num(), 0.5);
    assert_eq!(decode.get("events").arr()[0].get("name").str(), "token");
    let root = telemetry::request_spans(&mut IdGen::new(3), None, &timing(1));
    let doc = j::parse(&telemetry::otlp_json(&res, "tl-serve", &root));
    assert!(doc.get("resourceSpans").arr()[0].get("scopeSpans").arr()[0].get("spans").arr()[0].get_opt("parentSpanId").is_none(), "a root span has no parentSpanId");
}

#[test]
fn export_posts_to_a_collector() {
    // WHY: the export is one POST /v1/traces with Content-Type
    //      application/json and the body unchanged; a dead collector is a
    //      bounded error, never a hang inside a request.
    // KIND: fault
    // CATCHES: m15
    // CHAPTER: L10.7 section 4
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (s, _) = l.accept().unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut r = BufReader::new(s);
        let mut line = String::new();
        r.read_line(&mut line).unwrap();
        let mut headers = Vec::new();
        loop {
            let mut h = String::new();
            r.read_line(&mut h).unwrap();
            if h == "\r\n" || h.is_empty() {
                break;
            }
            headers.push(h.trim_end().to_ascii_lowercase());
        }
        let len: usize = headers.iter().find_map(|h| h.strip_prefix("content-length:").map(|v| v.trim().parse().unwrap())).unwrap();
        let mut body = vec![0u8; len];
        r.read_exact(&mut body).unwrap();
        r.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
        (line, headers, String::from_utf8(body).unwrap())
    });
    let body = telemetry::otlp_json(&telemetry::engine_resource("forge", "1", "unified"), "tl-serve", &telemetry::request_spans(&mut IdGen::new(4), None, &timing(2)));
    let status = telemetry::export(&format!("http://{addr}"), &body, Duration::from_secs(5)).unwrap();
    let (line, headers, got) = server.join().unwrap();
    assert_eq!(status, 200);
    assert!(line.starts_with("POST /v1/traces HTTP/1.1"), "request line {line:?}");
    assert!(headers.iter().any(|h| h == "content-type: application/json"), "{headers:?}");
    assert_eq!(got, body);
    let dead = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap(); // bound, then dropped: refused
    let t0 = std::time::Instant::now();
    assert!(telemetry::export(&format!("http://{dead}"), &body, Duration::from_secs(2)).is_err());
    assert!(t0.elapsed() < Duration::from_secs(3));
    assert!(telemetry::export("https://x:1", &body, Duration::from_secs(1)).is_err(), "only http:// endpoints");
}

#[test]
fn propagation_and_log_correlation() {
    // WHY: the context crosses process boundaries under the key
    //      "traceparent" (HTTP headers and gRPC metadata alike), and every
    //      log line inside a span carries trace_id and span_id so a grep of
    //      kubectl logs finds the request (obs.02).
    // KIND: unit
    // CATCHES: s09, m09, m16
    // CHAPTER: L10.7 section 4
    let c = TraceContext::parse(W3C).unwrap();
    let mut out: Vec<(String, String)> = Vec::new();
    telemetry::inject(&c, &mut |k, v| out.push((k.to_string(), v.to_string())));
    assert_eq!(out, vec![("traceparent".to_string(), W3C.to_string())]);
    let back = telemetry::extract(&|k| out.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone()));
    assert_eq!(back, Some(c));
    assert_eq!(telemetry::extract(&|_| None), None);
    let line = j::parse(&telemetry::log_line("2026-10-09T12:00:00Z", "info", "request done", "forge-engine", Some(&c)));
    assert_eq!((line.get("trace_id").str(), line.get("span_id").str(), line.get("service").str()), ("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7", "forge-engine"));
    assert!(j::parse(&telemetry::log_line("t", "warn", "a \"quoted\"\nmessage", "s", None)).get_opt("trace_id").is_none());
    let mut a = IdGen::new(9);
    let mut b = IdGen::new(9);
    for _ in 0..100 {
        let (x, y) = (a.span_id(), b.span_id());
        assert_eq!(x, y, "seeded ids repeat");
        assert_ne!(x, [0; 8]);
    }
    let s = Span::start(&mut a, "x", SpanKind::Client, Some(&c), 5);
    assert_eq!(s.context().header().split('-').nth(1), Some("4bf92f3577b34da6a3ce929d0e0e4736"));
}

// ---------------------------------------------------------------------------
// the serve loop (server.rs, which this module takes over from L10.5)

/// A running v1 server over the succ bigram (after byte i comes byte i + 1)
/// in a directory named `succ`, spoken to with raw HTTP/1.1 and
/// `Connection: close`. Drained when dropped.
mod serve {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpStream};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use tl_engine::runner::{KvConfig, PrefixCache};
    use tl_serve::server::{self, ServeConfig, ServerHandle};

    pub struct Srv(Option<ServerHandle>);

    impl Srv {
        pub fn http(&self) -> SocketAddr {
            self.0.as_ref().unwrap().http_addr
        }
        pub fn health(&self) -> SocketAddr {
            self.0.as_ref().unwrap().health_addr
        }
    }

    impl Drop for Srv {
        fn drop(&mut self) {
            if let Some(h) = self.0.take() {
                h.drain();
                let _ = h.join();
            }
        }
    }

    fn succ_model() -> PathBuf {
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let d = std::env::temp_dir().join(format!("ol-l10_7-{}-{nanos}", std::process::id())).join("succ");
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let mut w = vec![0.0f32; 256 * 256];
        for i in 0..256 {
            w[i * 256 + (i + 1) % 256] = 2.0;
        }
        let data: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        let header = format!(r#"{{"__metadata__":{{"format":"tinyllm"}},"bigram.weight":{{"dtype":"F32","shape":[256,256],"data_offsets":[0,{}]}}}}"#, data.len());
        let mut h = header.into_bytes();
        while h.len() % 8 != 0 {
            h.push(b' ');
        }
        let mut st = (h.len() as u64).to_le_bytes().to_vec();
        st.extend_from_slice(&h);
        st.extend_from_slice(&data);
        std::fs::write(d.join("model.safetensors"), st).unwrap();
        std::fs::write(d.join("config.json"), r#"{"tl_arch":"bigram","tl_tokenizer":"bytes","vocab_size":256,"tl_format":1}"#).unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"chat_template":"{% for m in messages %}{{ m.content }}{% endfor %}"}"#).unwrap();
        d
    }

    pub fn start() -> Srv {
        let mut c = ServeConfig::for_model(&succ_model());
        c.http_listen = "127.0.0.1:0".to_string();
        c.health_listen = "127.0.0.1:0".to_string();
        c.engine.kv = KvConfig { blocks: 64, block_size: 16 };
        c.engine.prefix_cache = PrefixCache::None;
        let s = Srv(Some(server::spawn(c).expect("server::spawn")));
        let t0 = Instant::now();
        while request(s.health(), "GET", "/readyz", "", "").0 != 200 {
            assert!(t0.elapsed() < Duration::from_secs(5), "/readyz never answered 200");
            std::thread::sleep(Duration::from_millis(5));
        }
        s
    }

    /// (status, body); a chunked body is left as it came (only status and
    /// the side effects matter here).
    pub fn request(addr: SocketAddr, method: &str, path: &str, body: &str, extra: &str) -> (u16, String) {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        let head = format!("{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra}\r\n{body}", body.len());
        s.write_all(head.as_bytes()).unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).expect("read until the server closes");
        let text = String::from_utf8_lossy(&out).into_owned();
        let status = text.split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or_else(|| panic!("no status line in {text:?}"));
        let body = text.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
        (status, body)
    }
}

/// The value of the sample `name` whose labels are exactly `labels`.
fn sample(fams: &BTreeMap<String, prom::Family>, family: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    let want: BTreeMap<String, String> = labels.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    fams.get(family)?.samples.iter().find(|(n, l, _)| n == name && *l == want).map(|(_, _, v)| *v)
}

#[test]
fn serve_loop_records_metrics_and_spans() {
    // WHY: the instruments only matter once the serve loop records them:
    //      every finished generation lands in the TTFT, TPOT, and request
    //      duration histograms (a failure with its error_type), every HTTP
    //      request in the HTTP histogram, and /metrics on the health port
    //      serves them (obs.02 scrapes exactly this). Each request becomes
    //      one trace that continues the caller's traceparent: the SERVER
    //      span and its engine.queue, engine.prefill, and engine.decode
    //      children, posted to $OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces.
    // KIND: conformance
    // CHAPTER: L10.7 section 4
    let collector = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", collector.local_addr().unwrap());
    collector.set_nonblocking(true).unwrap();
    let bodies = thread::spawn(move || {
        let mut got: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        while t0.elapsed() < Duration::from_secs(10) && !got.iter().any(|b| b.contains("engine.decode") && b.contains("4bf92f3577b34da6a3ce929d0e0e4736")) {
            let Ok((s, _)) = collector.accept() else {
                thread::sleep(Duration::from_millis(5));
                continue;
            };
            s.set_nonblocking(false).unwrap();
            s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut r = BufReader::new(s);
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h == "\r\n" || h.is_empty() {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
            }
            let mut body = vec![0u8; len];
            r.read_exact(&mut body).unwrap();
            r.get_mut().write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
            got.push(String::from_utf8(body).unwrap());
        }
        got
    });
    std::env::set_var("OTEL_EXPORTER_OTLP_ENDPOINT", &endpoint);
    std::env::set_var("OTEL_SERVICE_NAME", "forge-engine");
    let s = serve::start();
    let chat = |model: &str, extra: &str| format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"a"}}],"max_tokens":5,"temperature":0{extra}}}"#);
    let (st, body) = serve::request(s.http(), "POST", "/v1/chat/completions", &chat("succ", ""), &format!("traceparent: {W3C}\r\n"));
    assert_eq!(st, 200, "{body}");
    let (st, _) = serve::request(s.http(), "POST", "/v1/chat/completions", &chat("succ", r#","stream":true"#), "");
    assert_eq!(st, 200);
    let (st, _) = serve::request(s.http(), "POST", "/v1/chat/completions", &chat("nope", ""), "");
    assert_eq!(st, 404);
    let (st, text) = serve::request(s.health(), "GET", "/metrics", "", "");
    std::env::remove_var("OTEL_EXPORTER_OTLP_ENDPOINT");
    std::env::remove_var("OTEL_SERVICE_NAME");
    assert_eq!(st, 200);
    let fams = prom::parse(&text);
    let gen = [("gen_ai_operation_name", "chat"), ("gen_ai_request_model", "succ"), ("tl_engine_role", "unified")];
    let count = |fam: &str, labels: &[(&str, &str)]| sample(&fams, fam, &format!("{fam}_count"), labels);
    assert_eq!(count(metrics::TTFT, &gen), Some(2.0), "TTFT: one per answered generation, stream or not:\n{text}");
    assert_eq!(count(metrics::TPOT, &gen), Some(2.0), "TPOT: 5 tokens each, so both count");
    assert_eq!(count(metrics::REQUEST_DURATION, &[("gen_ai_operation_name", "chat"), ("gen_ai_request_model", "succ")]), Some(2.0));
    assert_eq!(
        count(metrics::REQUEST_DURATION, &[("gen_ai_operation_name", "chat"), ("gen_ai_request_model", "nope"), ("error_type", "model_not_found")]),
        Some(1.0),
        "a failed request is recorded with its error code"
    );
    let http = |status: &str| count(metrics::HTTP_DURATION, &[("http_request_method", "POST"), ("http_route", "/v1/chat/completions"), ("http_response_status_code", status)]);
    assert_eq!((http("200"), http("404")), (Some(2.0), Some(1.0)));
    let kv: f64 = ["free", "used", "cached"].iter().map(|st| sample(&fams, metrics::KV_BLOCKS, metrics::KV_BLOCKS, &[("state", st)]).expect("KV gauges")).sum();
    assert_eq!(kv, 64.0, "the KV gauges come from the engine's stats");
    let bodies = bodies.join().unwrap();
    let doc = bodies.iter().find(|b| b.contains("engine.decode") && b.contains("4bf92f3577b34da6a3ce929d0e0e4736")).map(|b| j::parse(b)).unwrap_or_else(|| panic!("no exported span tree continues the caller's trace: {bodies:?}"));
    let rs = &doc.get("resourceSpans").arr()[0];
    let service = rs.get("resource").get("attributes").arr().iter().find(|a| a.get("key").str() == "service.name").map(|a| a.get("value").get("stringValue").str().to_string());
    assert_eq!(service.as_deref(), Some("forge-engine"));
    let spans = rs.get("scopeSpans").arr()[0].get("spans").arr();
    let names: Vec<&str> = spans.iter().map(|s| s.get("name").str()).collect();
    assert_eq!(names, ["POST /v1/chat/completions", "engine.queue", "engine.prefill", "engine.decode"]);
    let server = &spans[0];
    assert_eq!(server.get("traceId").str(), "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(server.get("parentSpanId").str(), "00f067aa0ba902b7", "the SERVER span is the caller's child");
    for child in &spans[1..] {
        assert_eq!(child.get("parentSpanId").str(), server.get("spanId").str());
    }
    let out = spans[3].get("attributes").arr().iter().find(|a| a.get("key").str() == "gen_ai.usage.output_tokens").map(|a| a.get("value").get("intValue").str().to_string());
    assert_eq!(out.as_deref(), Some("5"));
}

/// A tiny JSON reader, independent of the code under test.
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
            self.get_opt(k).unwrap_or_else(|| panic!("no key {k:?}"))
        }
        pub fn get_opt(&self, k: &str) -> Option<&V> {
            self.obj().iter().find(|(a, _)| a == k).map(|(_, v)| v)
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
        pub fn num(&self) -> f64 {
            match self {
                V::Num(n) => *n,
                _ => panic!("not a number: {self:?}"),
            }
        }
        pub fn f64(&self) -> f64 {
            self.num()
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
        assert_eq!(i, b.len(), "trailing characters in JSON: {s:?}");
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
                            let V::Str(k) = val(b, i) else { panic!("key is not a string") };
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
                        s.push(match b[*i] {
                            b'n' => b'\n',
                            b't' => b'\t',
                            c => c,
                        });
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
                V::Num(std::str::from_utf8(&b[st..*i]).unwrap().parse().unwrap())
            }
        }
    }
}
