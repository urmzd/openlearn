//! tl-serve: the engine's entry point (reference, course CI only).
//!
//!   tl-serve --model-dir <dir> --port <n> --health-port <n>     the tracer (L10.0, API v0)
//!   tl-serve --config <runtime.toml>                            API v1 (L10.5)
//!                                                               (spec/cli-roles.md)
//!
//! Entry points are learner territory (DESIGN D16): this file is never
//! overlaid into a learner's checks. `ol verify course --e2e` copies it into
//! the assembled reference learner so the milestones have an engine to start.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process;
use std::sync::Arc;
use std::thread;

use tl_serve::http;

struct Args {
    model_dir: PathBuf,
    port: u16,
    health_port: u16,
}

fn parse_args() -> Result<Args, String> {
    let mut model_dir = None;
    let mut port = None;
    let mut health_port = None;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--model-dir" => model_dir = Some(PathBuf::from(value)),
            "--port" => port = Some(value.parse().map_err(|e| format!("--port {value}: {e}"))?),
            "--health-port" => health_port = Some(value.parse().map_err(|e| format!("--health-port {value}: {e}"))?),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok(Args {
        model_dir: model_dir.ok_or("--model-dir is required")?,
        port: port.ok_or("--port is required")?,
        health_port: health_port.ok_or("--health-port is required")?,
    })
}

fn bind(port: u16) -> TcpListener {
    TcpListener::bind(("0.0.0.0", port)).unwrap_or_else(|e| {
        eprintln!("tl-serve: bind 0.0.0.0:{port}: {e}");
        process::exit(1)
    })
}

/// `--config <runtime.toml>`: the v1 server of L10.5 (tokio, hyper, the
/// continuous-batching engine); it drains on SIGTERM and exits 0.
fn serve_config(path: &str) -> ! {
    let cfg = tl_serve::server::ServeConfig::load(std::path::Path::new(path)).unwrap_or_else(|e| {
        eprintln!("tl-serve: {e}");
        process::exit(2)
    });
    eprintln!("tl-serve: serving {} on {} (health {})", cfg.model_dir.display(), cfg.http_listen, cfg.health_listen);
    match tl_serve::server::run(cfg) {
        Ok(()) => process::exit(0),
        Err(e) => {
            eprintln!("tl-serve: {e}");
            process::exit(1)
        }
    }
}

/// spec/cli-roles.md: a server exits 0 on SIGTERM, which Kubernetes sends
/// before it kills a pod. The tracer has no requests worth draining, so a
/// watcher thread waits for the signal and exits at once.
fn exit_on_sigterm() {
    thread::spawn(|| {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build();
        if let Ok(rt) = rt {
            rt.block_on(async {
                use tokio::signal::unix::{signal, SignalKind};
                if let Ok(mut term) = signal(SignalKind::terminate()) {
                    term.recv().await;
                    process::exit(0);
                }
            });
        }
    });
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("--config") {
        match argv.get(2) {
            Some(path) if argv.len() == 3 => serve_config(path),
            _ => {
                eprintln!("usage: tl-serve --config <runtime.toml>");
                process::exit(2)
            }
        }
    }
    exit_on_sigterm();
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("tl-serve: {e}\nusage: tl-serve --model-dir <dir> --port <n> --health-port <n>");
        process::exit(2)
    });
    let server = Arc::new(http::Server::from_env(&args.model_dir).unwrap_or_else(|e| {
        eprintln!("tl-serve: {e}");
        process::exit(1)
    }));
    let api = bind(args.port);
    let health = bind(args.health_port);
    let for_health = Arc::clone(&server);
    thread::spawn(move || http::serve(health, for_health));
    eprintln!("tl-serve: serving {} on :{} (health :{})", args.model_dir.display(), args.port, args.health_port);
    if let Err(e) = http::serve(api, server) {
        eprintln!("tl-serve: {e}");
        process::exit(1);
    }
}
