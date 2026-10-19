//! kvpush: push toy KV blocks to a KvTransferService, or release a handle.
//!
//!   kvpush push --addr <url> --handle <id> --blocks <n> [--seed <s>] [--kv-format <f>] [--corrupt]
//!   kvpush release --addr <url> --handle <id>
//!
//! Prints one JSON line: `{"present":[..],"received":n,"deduped":n}` for a
//! push, `{"released":true}` for a release, or `{"error":"<CODE>",
//! "message":"..."}` (exit 1) when the server answers with an error status.

use std::process;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let usage = || -> ! {
        eprintln!("usage: kvpush push|release --addr <url> --handle <id> [--blocks n] [--seed s] [--kv-format f] [--corrupt]");
        process::exit(2)
    };
    let (Some(verb), Some(addr), Some(handle)) = (args.first().cloned(), flag("--addr"), flag("--handle")) else { usage() };
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("runtime");
    let out = rt.block_on(async {
        match verb.as_str() {
            "push" => {
                let n: u32 = flag("--blocks").and_then(|v| v.parse().ok()).unwrap_or(3);
                let seed: u64 = flag("--seed").and_then(|v| v.parse().ok()).unwrap_or(1);
                let fmt: u32 = flag("--kv-format").and_then(|v| v.parse().ok()).unwrap_or(1);
                let blocks: Vec<Vec<u8>> = (0..n).map(|i| kvpush::payload(seed, i, 64)).collect();
                kvpush::push(&addr, &handle, &blocks, fmt, args.iter().any(|a| a == "--corrupt")).await.map(|r| {
                    let p: Vec<&str> = r.present.iter().map(|&b| if b { "true" } else { "false" }).collect();
                    format!("{{\"present\":[{}],\"received\":{},\"deduped\":{}}}", p.join(","), r.received, r.deduped)
                })
            }
            "release" => kvpush::release(&addr, &handle).await.map(|_| "{\"released\":true}".to_string()),
            _ => usage(),
        }
    });
    match out {
        Ok(line) => println!("{line}"),
        Err(s) => {
            println!("{{\"error\":\"{}\",\"message\":{:?}}}", kvpush::code_name(s.code()), s.message());
            process::exit(1);
        }
    }
}
