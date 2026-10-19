//! wire: the lang.05 server. HTTP/1.1 over std::net, one request per
//! connection, a thread per connection.
//!
//!   wire --port <n>                 listen on 127.0.0.1:<n> (0: any free port)
//!
//!   GET  /health                    200 {"ok":true}
//!   POST /echo   {"text": "..."}    200 {"text": "...", "bytes": <UTF-8 length>}
//!   GET  /count?n=3&delay_ms=0      SSE: data: {"i":0} ... data: [DONE]
//!
//! Try it: `curl -N 'http://127.0.0.1:<port>/count?n=5&delay_ms=500'`.

use std::io::{self, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process;
use std::thread;
use std::time::Duration;

use wire::{json_string, parse_json, read_request, sse_event, write_response, write_sse_head, HttpError, Json, Request};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let port: u16 = match args.as_slice() {
        [] => 8080,
        [flag, v] if flag == "--port" => v.parse().unwrap_or_else(|e| {
            eprintln!("wire: --port {v}: {e}");
            process::exit(2)
        }),
        _ => {
            eprintln!("usage: wire [--port <n>]");
            process::exit(2)
        }
    };
    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| {
        eprintln!("wire: bind: {e}");
        process::exit(1)
    });
    println!("listening on {}", listener.local_addr().expect("bound"));
    let _ = io::stdout().flush();
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                thread::spawn(move || {
                    if let Err(e) = handle(stream) {
                        eprintln!("wire: {e}");
                    }
                });
            }
            Err(e) => eprintln!("wire: accept: {e}"),
        }
    }
}

/// One connection, one request, one response.
fn handle(stream: TcpStream) -> io::Result<()> {
    // Small writes (one SSE event) must leave at once, not wait for more data.
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut out = stream;
    match read_request(&mut reader) {
        Ok(req) => route(&req, &mut out),
        Err(HttpError::Closed) | Err(HttpError::Io(_)) => Ok(()),
        Err(e) => {
            let msg = format!("{e:?}");
            error(&mut out, e.status(), &msg, &[])
        }
    }
}

/// Writes `{"error": "<msg>"}` with the given status.
fn error(out: &mut TcpStream, status: u16, msg: &str, extra: &[(&str, &str)]) -> io::Result<()> {
    let body = format!("{{\"error\":{}}}", json_string(msg));
    write_response(out, status, "application/json", extra, body.as_bytes())
}

fn route(req: &Request, out: &mut TcpStream) -> io::Result<()> {
    match (req.method.as_str(), req.path()) {
        ("GET", "/health") => write_response(out, 200, "application/json", &[], b"{\"ok\":true}"),
        ("POST", "/echo") => echo(req, out),
        ("GET", "/count") => count(req, out),
        (_, "/health") | (_, "/count") => error(out, 405, "method not allowed", &[("Allow", "GET")]),
        (_, "/echo") => error(out, 405, "method not allowed", &[("Allow", "POST")]),
        _ => error(out, 404, &format!("no route for {}", req.path()), &[]),
    }
}

/// POST /echo: the body's "text" back, with its UTF-8 byte length.
fn echo(req: &Request, out: &mut TcpStream) -> io::Result<()> {
    let text = std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(parse_json);
    let text = match text {
        Ok(v) => match v.get("text").and_then(Json::as_str) {
            Some(t) => t.to_string(),
            None => return error(out, 400, "body must be a JSON object with a string \"text\"", &[]),
        },
        Err(e) => return error(out, 400, &format!("bad JSON: {e}"), &[]),
    };
    let body = format!("{{\"text\":{},\"bytes\":{}}}", json_string(&text), text.len());
    write_response(out, 200, "application/json", &[], body.as_bytes())
}

/// GET /count?n=&delay_ms=: n events `{"i":k}`, delay_ms apart, then [DONE].
fn count(req: &Request, out: &mut TcpStream) -> io::Result<()> {
    let n: u32 = match req.query("n").map(str::parse::<u32>) {
        None => 3,
        Some(Ok(n)) if n <= 1000 => n,
        _ => return error(out, 400, "n must be an integer from 0 to 1000", &[]),
    };
    let delay: u64 = match req.query("delay_ms").map(str::parse::<u64>) {
        None => 0,
        Some(Ok(d)) if d <= 5000 => d,
        _ => return error(out, 400, "delay_ms must be an integer from 0 to 5000", &[]),
    };
    write_sse_head(out)?;
    for i in 0..n {
        if i > 0 && delay > 0 {
            thread::sleep(Duration::from_millis(delay));
        }
        out.write_all(sse_event(&format!("{{\"i\":{i}}}")).as_bytes())?;
        out.flush()?; // a TcpStream has no buffer, but a BufWriter would: flush per event
    }
    out.write_all(sse_event("[DONE]").as_bytes())?;
    out.flush()
}
