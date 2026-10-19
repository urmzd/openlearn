//! ticker: the lang.09 server.
//!
//!   ticker --port <n>     listen on 127.0.0.1:<n> (0: any free port)
//!
//! Prints `listening on 127.0.0.1:<port>` first. SIGTERM (or Ctrl-C) stops
//! accepting, lets open requests finish, and exits 0.
//!
//! Try it: `curl -N 'http://127.0.0.1:<port>/ticks?n=5&interval_ms=500'`.

use std::io::Write;
use std::process;
use std::sync::Arc;

use ticker::{serve, App};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let port: u16 = match args.as_slice() {
        [] => 8080,
        [flag, v] if flag == "--port" => v.parse().unwrap_or_else(|e| {
            eprintln!("ticker: --port {v}: {e}");
            process::exit(2)
        }),
        _ => {
            eprintln!("usage: ticker [--port <n>]");
            process::exit(2)
        }
    };
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap_or_else(|e| {
        eprintln!("ticker: runtime: {e}");
        process::exit(1)
    });
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.unwrap_or_else(|e| {
            eprintln!("ticker: bind: {e}");
            process::exit(1)
        });
        println!("listening on {}", listener.local_addr().expect("bound"));
        let _ = std::io::stdout().flush();
        let shutdown = async {
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("SIGTERM handler");
            tokio::select! {
                _ = term.recv() => {}
                _ = tokio::signal::ctrl_c() => {}
            }
        };
        serve(listener, Arc::new(App::new()), shutdown).await;
    });
}
