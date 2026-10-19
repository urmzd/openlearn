//! The `tl-tok` role (spec/cli-roles.md, Pass 3; course/milestones/MS-L1.toml
//! fixes the flags and the final lines).
//!
//!     tl-tok encode --tokenizer <tokenizer.json> --in <texts.jsonl>
//!     tl-tok bench  --tokenizer <tokenizer.json> --in <texts.jsonl> [--threads T] [--repeat R]
//!
//! Entry-point territory (D16): this file is yours. It is glue over L1.5
//! (`ByteBpe::from_hf_json`, `encode`, `encode_batch`). Input texts are JSON
//! lines with a "text" key in a .jsonl file, or one text per line otherwise. The last stdout line is one JSON object; exit 2
//! on a usage error, 1 on any other failure.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use serde_json::{json, Value};
use tl_tok::{ByteBpe, Tokenizer};

struct Args {
    verb: String,
    tokenizer: String,
    input: String,
    threads: usize,
    repeat: usize,
}

fn usage(why: &str) -> ExitCode {
    eprintln!("tl-tok: {why}\nusage: tl-tok <encode|bench> --tokenizer <tokenizer.json> --in <texts.jsonl> [--threads T] [--repeat R]");
    ExitCode::from(2)
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut it = argv.iter();
    let verb = it.next().ok_or("missing verb")?.clone();
    if verb != "encode" && verb != "bench" {
        return Err(format!("unknown verb {verb:?}"));
    }
    let (mut tokenizer, mut input, mut threads, mut repeat) = (None, None, 4usize, 3usize);
    while let Some(flag) = it.next() {
        let val = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--tokenizer" => tokenizer = Some(val.clone()),
            "--in" => input = Some(val.clone()),
            "--threads" => threads = val.parse().map_err(|_| format!("--threads {val:?}"))?,
            "--repeat" => repeat = val.parse().map_err(|_| format!("--repeat {val:?}"))?,
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok(Args {
        verb,
        tokenizer: tokenizer.ok_or("--tokenizer is required")?,
        input: input.ok_or("--in is required")?,
        threads,
        repeat: repeat.max(1),
    })
}

/// A .jsonl file: one JSON object with a string "text" per line. Any other
/// file: one text per line.
fn read_texts(path: &str) -> Result<Vec<String>, String> {
    let body = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    if !path.ends_with(".jsonl") {
        return Ok(body.lines().map(str::to_string).collect());
    }
    let mut out = Vec::new();
    for (n, line) in body.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", n + 1))?;
        match v.get("text").and_then(Value::as_str) {
            Some(t) => out.push(t.to_string()),
            None => return Err(format!("{path}:{}: want a JSON object with a string \"text\"", n + 1)),
        }
    }
    Ok(out)
}

fn run(a: &Args) -> Result<Value, String> {
    let tok = ByteBpe::from_hf_json(Path::new(&a.tokenizer)).map_err(|e| e.to_string())?;
    let texts = read_texts(&a.input)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let bytes: usize = refs.iter().map(|t| t.len()).sum();
    if a.verb == "encode" {
        let ids: Vec<u32> = tok.encode_batch(&refs, 0).into_iter().flatten().collect();
        let n = ids.len();
        return Ok(json!({"ids": ids, "texts": texts.len(), "tokens": n,
                         "bytes_per_token": if n > 0 { bytes as f64 / n as f64 } else { 0.0 }}));
    }
    let best = |f: &dyn Fn() -> usize| -> (f64, usize) {
        let mut t = f64::INFINITY;
        let mut n = 0;
        for _ in 0..a.repeat {
            let t0 = Instant::now();
            n = f();
            t = t.min(t0.elapsed().as_secs_f64());
        }
        (t, n)
    };
    let (t1, n) = best(&|| tok.encode_batch(&refs, 1).iter().map(Vec::len).sum());
    let (tb, _) = best(&|| tok.encode_batch(&refs, a.threads).iter().map(Vec::len).sum());
    Ok(json!({"texts": texts.len(), "tokens": n, "threads": a.threads,
              "tokens_per_s": n as f64 / t1, "batch_tokens_per_s": n as f64 / tb,
              "batch_speedup": t1 / tb}))
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let a = match parse(&argv) {
        Ok(a) => a,
        Err(why) => return usage(&why),
    };
    match run(&a) {
        Ok(v) => {
            println!("{v}");
            ExitCode::SUCCESS
        }
        Err(why) => {
            eprintln!("tl-tok {}: {why}", a.verb);
            ExitCode::from(1)
        }
    }
}
