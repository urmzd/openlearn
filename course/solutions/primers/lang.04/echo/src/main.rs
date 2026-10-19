//! echo: the lang.04 line-protocol server on std::net.
//!
//!   echo --port <n>      listen on 127.0.0.1:<n> (0 lets the OS pick a port)
//!
//! The first line on stdout is `listening on 127.0.0.1:<port>`, so a script
//! that started it with `--port 0` learns the port. Each connection gets its
//! own thread, so one slow client never blocks another.

use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process;
use std::thread;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let port = match parse_port(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("echo: {e}");
            process::exit(2);
        }
    };
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("echo: bind 127.0.0.1:{port}: {e}");
            process::exit(1);
        }
    };
    let addr = listener.local_addr().expect("a bound listener has an address");
    println!("listening on {addr}");
    let _ = io::stdout().flush();

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                // `move` hands this thread ownership of `stream`; the loop
                // keeps no reference to it.
                thread::spawn(move || {
                    if let Err(e) = serve(stream) {
                        eprintln!("echo: connection: {e}");
                    }
                });
            }
            Err(e) => eprintln!("echo: accept: {e}"),
        }
    }
}

/// `--port <n>`; no flag means 7878.
fn parse_port(args: &[String]) -> Result<u16, String> {
    match args {
        [] => Ok(7878),
        [flag, value] if flag == "--port" => value.parse::<u16>().map_err(|e| format!("--port {value}: {e}")),
        _ => Err("usage: echo [--port <n>]".to_string()),
    }
}

/// One connection: read a line, write its reply, until QUIT or EOF.
fn serve(stream: TcpStream) -> io::Result<()> {
    // Two handles on one socket: a buffered reader (a single read() may hold
    // half a line or three lines) and the stream itself for writing.
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;
        if n == 0 || line.last() != Some(&b'\n') {
            return Ok(()); // EOF, or the client left mid-line: nothing to answer
        }
        let (text, close) = lineproto::reply(&line);
        writer.write_all(text.as_bytes())?;
        writer.write_all(b"\n")?;
        if close {
            return Ok(());
        }
    }
}
