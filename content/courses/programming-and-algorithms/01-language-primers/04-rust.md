# Rust: ownership, traits, Result, cargo workspaces, std TCP

By the time a system can compute a bigram's logits and has a checkpoint on disk, nothing can reach it over a network yet. No process listens on a port, so a gateway has nothing to forward to and a cluster has nothing to run. The next step in *AI Engineering*, [Your first endpoint](../../ai-engineering/23-model-serving/01-your-first-endpoint.md), writes that process: an inference engine in **Rust** that accepts TCP connections, reads requests, and runs the model for every token. It uses only Rust's standard library, so every byte on the wire is visible in its own code.

This section builds the same skeleton with a trivial protocol: a server that reads lines from many clients at once and answers each line. When you reach the engine, the only new ideas will be HTTP, the model, and tracing. The solution is a Cargo workspace ([`Cargo.toml`](../../../../course/solutions/primers/lang.04/Cargo.toml)) with the library [`lineproto/src/lib.rs`](../../../../course/solutions/primers/lang.04/lineproto/src/lib.rs) (`Command`, `ProtoError`, `parse`, `byte_len`, `respond`, `reply`) and the binary [`echo/src/main.rs`](../../../../course/solutions/primers/lang.04/echo/src/main.rs) (`main`, `parse_port`, `serve`: a line-protocol server on `std::net`, one thread per connection). The protocol table and signatures below are the whole contract. To run it: `cd primers/lang.04 && cargo test && cargo run -p echo -- --port 7878`.

Rust is introduced here by comparison with the earlier primers: [Python and numpy](01-python-and-numpy.md), [Shell, git, make, and processes](02-shell-git-make.md) for processes, exit codes, and signals, and [C](03-c.md), which is useful background for comparing byte strings and ownership. Every term is defined before it is used.

> **Learning objectives**
> - Organize Rust code as a Cargo workspace with a pure library and a thin binary.
> - Apply ownership, moves, and borrowing, and explain why they rule out use-after-free and data races at compile time.
> - Model failure with `Result`, enums, and `Display`, and propagate it with `?` instead of panicking.
> - Distinguish UTF-8 byte length from Unicode scalar count in Rust strings.
> - Write a TCP line server that frames a byte stream with `BufReader::read_until` and serves each connection on its own thread.

## Cargo: packages, crates, targets, workspaces

**Cargo** is Rust's build tool and package manager. A **package** is a directory with a `Cargo.toml` manifest naming it, its version, its Rust **edition** (the language revision, `2021` here), and its dependencies. A package holds one or more **crates**: a crate is one compilation unit, either a **library** (`src/lib.rs`, used by other crates) or a **binary** (`src/main.rs`, which has a `fn main` and becomes an executable). Each crate is a **target** of its package.

A **workspace** is a directory whose `Cargo.toml` has a `[workspace]` table listing **member** packages. Members share one `Cargo.lock` (the exact versions of every dependency) and one `target/` directory (all build output). One package depends on another by **path**:

```toml
[dependencies]
lineproto = { path = "../lineproto" }
```

These are the commands you run at the workspace root:

| Command (at the workspace root) | What it does |
|---|---|
| `cargo build` | compiles every member into `target/debug/` (`--release` optimizes, into `target/release/`) |
| `cargo test` | compiles every crate with its `#[test]` functions and runs them |
| `cargo run -p echo -- --port 7878` | builds and runs the binary of package `echo`; arguments after `--` go to the program |
| `cargo clippy` | lints for likely mistakes (optional, recommended) |
| `cargo fmt` | formats every file in the one standard style |

`--offline` tells Cargo not to touch the network; this primer has no dependencies outside the standard library (`std`), so it never needs to. The split into a library and a binary is deliberate: the library has no I/O, so its rules are testable with plain function calls, and the binary only moves bytes between a socket and the library. The engine uses the same split (`tl-serve` is a library plus a `main.rs`), and the system's `rust/Cargo.toml` is a workspace with two members.

## Ownership, moves, and borrowing

The build tool settles how code is organized; ownership settles how memory is managed. C makes the programmer free every allocation; Python frees it when nothing refers to it any more, at run time. Rust decides at **compile time**, with three rules:

1. Every value has exactly one **owner**, a variable.
2. When the owner goes out of scope, the value is **dropped**: its memory (and any socket or file it holds) is released.
3. Assigning a value, or passing it to a function, **moves** it: the new variable is the owner and the old one may not be used again.

```rust
let s = String::from("hello");  // s owns a heap buffer holding 5 bytes
let t = s;                      // moved: t owns the buffer now
println!("{s}");                // compile error: borrow of moved value `s`
```

To use a value without taking it, you **borrow** it with a **reference**: `&s` is a shared (read-only) reference, `&mut s` an exclusive (read-write) one. At any moment a value has either any number of shared references or exactly one exclusive reference, never both, and no reference may outlive the value. Those two rules rule out use-after-free and data races before the program runs, which is how Rust frees memory without a garbage collector. The server below relies on this when it moves each socket into its own thread.

Two pairs of types follow from these rules:

| Owned (you can keep it, grow it, move it) | Borrowed view (a pointer and a length) | Holds |
|---|---|---|
| `String` | `&str` | UTF-8 text |
| `Vec<u8>` | `&[u8]` | any bytes |

A function that only reads text takes `&str`; a value that must outlive the line it was read from (the text inside a parsed command) is an owned `String`, made with `.to_string()`. Bytes from a socket are `&[u8]` until you have checked they are UTF-8: `std::str::from_utf8(bytes)` returns `Ok(&str)` or an error, because a Rust `str` is always valid UTF-8.

Types with no heap data, such as integers and `bool`, are **Copy**: assigning duplicates them and the old variable stays usable.

## Structs, enums, and match

A **struct** groups named fields. An **enum** is a value that is exactly one of several **variants**, and a variant may carry data:

```rust
enum Command { Ping, Echo(String), Len(String), Quit }
```

`match` takes a value apart by variant, and the compiler rejects a `match` that forgets a variant (it must be **exhaustive**). The standard library's two most used enums are `Option<T>` (`Some(value)` or `None`, Rust's replacement for null) and `Result<T, E>`.

`#[derive(Debug, Clone, PartialEq, Eq)]` above a type asks the compiler to write the code for printing it with `{:?}`, copying it with `.clone()`, and comparing it with `==`.

## Errors as values: Result and ?

`Result` deserves its own treatment, because it is how Rust replaces both C's return codes and Python's exceptions. C reports failure through a return code that the caller may ignore; Python raises an exception that unwinds the stack. Rust returns it: a function that can fail returns `Result<T, E>`, which is `Ok(value)` or `Err(error)`, and the caller cannot reach the value without deciding what to do with the error.

```rust
fn parse(line: &str) -> Result<Command, ProtoError> {
    // return Ok(command) or Err(error), depending on the request
}
```

A handful of tools make working with `Result` concise:

| Tool | Meaning |
|---|---|
| `expr?` | if `expr` is `Err(e)`, return `Err(e)` from this function now (converting `e` with `From` if needed); otherwise unwrap the `Ok` value |
| `.map_err(f)` | turn one error type into another |
| `.and_then(f)` | if `Ok(v)`, run `f(v)`, which itself returns a `Result` (chains fallible steps) |
| `.unwrap()`, `.expect("why")` | take the `Ok` value or **panic** |

A **panic** stops the current thread with a message. It is for bugs (a state the code proves impossible), never for bad input: a client that sends garbage must get an error reply, and a panic in a server thread drops that client's connection with no answer. `unwrap()` turns an error into a crash; a server answers errors instead.

## Traits

A **trait** names methods that a type promises to have. A type **implements** a trait with an `impl` block:

```rust
impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { ... }
}
```

Now `format!("ERR {e}")` works for a `ProtoError`, because `{}` formats any `Display` value. `impl std::error::Error for ProtoError {}` marks it as an error type other code can handle generically.

Traits are also how one function works on many types. The standard I/O traits are what make a socket, a file, and a byte slice interchangeable:

| Trait | Promise | Implemented by |
|---|---|---|
| `Read` | `read(&mut self, buf: &mut [u8]) -> io::Result<usize>`: fills up to `buf.len()` bytes, returns how many; 0 means end of stream | `TcpStream`, `File`, `&[u8]` |
| `Write` | `write_all(&mut self, bytes: &[u8])`, `flush()` | `TcpStream`, `File`, `Vec<u8>` |
| `BufRead` | `read_until(&mut self, byte: u8, buf: &mut Vec<u8>)`, `read_line`: reads through the next delimiter, using an internal buffer | `BufReader<R>` for any `R: Read`, `&[u8]` |

A function written as `fn f<R: BufRead>(r: &mut R)` is **generic**: it works for any type `R` that implements `BufRead`, a socket in the server and a byte slice in a test.

## Strings, bytes, and UTF-8

A Rust `str` is valid UTF-8. Its `.len()` is the number of bytes in that encoding; `.chars().count()` is the number of Unicode scalar values. These differ when a character needs more than one UTF-8 byte: `"héllo"` has six bytes and five scalar values. A Rust string can also contain a NUL byte; it is ordinary content because the string carries an explicit length rather than ending at a sentinel, as a C string does.

```rust
let text = "héllo";
assert_eq!(text.len(), 6);              // UTF-8 bytes
assert_eq!(text.chars().count(), 5);   // Unicode scalar values
assert_eq!("a\0b".len(), 3);            // NUL is included
```

The protocol's `LEN` command makes this distinction observable over the wire. Since its line parser has already validated UTF-8, `.len()` gives the byte count needed by byte-oriented protocols without unsafe code or a foreign-library call.

## TCP with std::net

With the language pieces in place, the network comes next. **TCP** gives two programs a reliable, ordered, two-way stream of bytes. A **server** opens a **listening socket** on an address and a **port** (a 16-bit number naming the service on that machine); `127.0.0.1` is this machine only, `0.0.0.0` every network interface (what a container needs). Port 0 asks the operating system for any free port, which is how tests avoid collisions; `listener.local_addr()` then tells you which one you got.

```rust
let listener = TcpListener::bind(("127.0.0.1", port))?;
for conn in listener.incoming() {      // blocks until a client connects
    let stream: TcpStream = conn?;     // one connected client
}
```

A `TcpStream` implements `Read` and `Write`. The key fact: **TCP preserves bytes, not writes**. It is a byte stream, not a message stream. A client that writes `"PING\n"` and then `"ECHO a\n"` may have both arrive in one `read`, or `"PI"` in one read and `"NG\n"` in the next. A line protocol must therefore buffer: `BufReader::new(stream)` reads large chunks into memory and `read_until(b'\n', &mut line)` hands back exactly one line, however it arrived. `stream.try_clone()` gives a second handle on the same socket, so one handle can sit inside the `BufReader` while the other writes replies.

`read_until` returns `Ok(0)` when the client closed its side of the connection (end of stream); a line that ends without `\n` means the client left mid-line.

## Threads

A server that handles one connection at a time stops serving everyone while one client is slow. The simplest fix is one **thread** per connection:

```rust
thread::spawn(move || {
    if let Err(e) = serve(stream) { eprintln!("echo: connection: {e}"); }
});
```

`thread::spawn` runs a **closure** (an anonymous function) on a new operating-system thread. `move` makes the closure take ownership of what it uses, here `stream`, so the new thread owns the socket and the accept loop keeps no reference to it: the compiler would reject sharing it by accident. When `serve` returns, `stream` is dropped and the socket closes. Threads cost memory (a stack each), which is fine for a primer and for the tracer engine; [Async Rust and tokio](09-async-rust-and-tokio.md) and the [OpenAI-compatible server on tokio](../../ai-engineering/23-model-serving/06-openai-server-on-tokio.md) replace them with **async** tasks.

## Tests

A function marked `#[test]` is a test; it passes if it returns without panicking. `assert_eq!(a, b)` panics with both values when they differ. Unit tests live next to the code in a module compiled only for testing:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ping_is_pong() { assert_eq!(reply(b"PING\n"), ("PONG".to_string(), false)); }
}
```

`cargo test` runs them.

## Working it by hand

Before reading the protocol formally, replay a session. It exercises ping, echo, `LEN`, pipelining, errors, and `QUIT`. Client bytes are on the left, server bytes on the right; `\n` is the byte `0x0a`.

| Client sends | Server answers | Why |
|---|---|---|
| `PING\n` | `PONG\n` | the command word is `PING` |
| `ECHO  two  spaces\n` | ` two  spaces\n` | the text is everything after the **first** space, so it starts with a space |
| `LEN héllo\n` | `6\n` | see below |
| `PING\nECHO a\nLEN abc\n` (one write) | `PONG\n` `a\n` `3\n` | three lines, three replies, in order |
| `JUMP high\n` | `ERR unknown command JUMP\n` | the connection stays open |
| `QUIT\n` | `BYE\n`, then the server closes | the only request that ends the connection |

Now trace `LEN héllo` by hand, through `reply(line)`:

1. The line's bytes are `4c 45 4e 20 68 c3 a9 6c 6c 6f 0a`. `é` is the code point U+00E9, which UTF-8 writes as two bytes, `c3 a9`.
2. `std::str::from_utf8` accepts them: `Ok("LEN héllo\n")`.
3. `parse` strips the `\n` (and a `\r` if one came before it), splits at the first space into `"LEN"` and `"héllo"`, and returns `Ok(Command::Len("héllo".to_string()))`.
4. `respond` calls `byte_len("héllo")`. Rust's `str.len()` counts its six UTF-8 bytes: `68 c3 a9 6c 6c 6f`.
5. The reply is `"6"`, and the server writes `6\n`.

`"héllo".chars().count()` is 5 Unicode scalar values; `.len()` is 6 bytes. An interior NUL is counted too: `LEN a\0b` replies `3`.

Try it with the solution's server running (`cargo run -p echo -- --port 7878`) and `nc 127.0.0.1 7878` in another terminal; type the lines. `printf 'PING\nECHO a\nLEN abc\n' | nc 127.0.0.1 7878` sends three lines in one write.

## The protocol and the interface

The workspace has this layout:

```text
primers/lang.04/
  Cargo.toml              [workspace] members = ["lineproto", "echo"]
  lineproto/Cargo.toml    package lineproto, a library
  lineproto/src/lib.rs    the protocol: no I/O, unit-tested with cargo test
  echo/Cargo.toml         package echo, depends on lineproto by path
  echo/src/main.rs        the server: std::net, one thread per connection
```

One request is one line ending in `\n`; a `\r` right before the `\n` is part of the line ending. The command word is everything before the first space (the whole line if there is none); the text is everything after it. Commands are case-sensitive. The full protocol is:

| Request | Reply | Connection |
|---|---|---|
| `PING` | `PONG` | stays open |
| `ECHO <text>` | `<text>`, byte for byte | stays open |
| `LEN <text>` | the UTF-8 byte length of `<text>`, computed by `str.len()` | stays open |
| `QUIT` | `BYE` | the server closes it |
| an empty line | `ERR empty line` | stays open |
| bytes that are not UTF-8 | `ERR invalid utf-8` | stays open |
| any other word `W` | `ERR unknown command W` | stays open |
| `LEN a\0b` | `3` (NUL is ordinary string content) | stays open |

The library's interface, in `lineproto/src/lib.rs`, is:

```rust
pub enum Command { Ping, Echo(String), Len(String), Quit }
pub enum ProtoError { Empty, InvalidUtf8, Unknown(String) }
impl fmt::Display for ProtoError { ... }            // the text after "ERR "
impl std::error::Error for ProtoError {}

pub fn parse(line: &str) -> Result<Command, ProtoError>;      // line with or without "\n" / "\r\n"
pub fn byte_len(text: &str) -> usize;                         // UTF-8 bytes, NUL included
pub fn respond(cmd: &Command) -> String;                      // reply text, no "\n"
pub fn reply(line: &[u8]) -> (String, bool);                  // the whole protocol: (reply, close?)
```

The binary, `echo/src/main.rs`, behaves as follows. `echo --port <n>` binds `127.0.0.1:<n>` (no flag: 7878; port 0: any free port) and prints exactly `listening on 127.0.0.1:<port>` as its first stdout line, so a script that started it with port 0 can read the port from it. It then serves every connection on its own thread: read a line, write `reply(line)` and `\n`, close after `QUIT` or when the client leaves. A bad `--port` exits 2 with a message on stderr.

## Building it step by step

Eight observations shape the solution. Try to write each function before reading it.

1. The protocol rules are pure functions of one line's bytes, so they belong in a library with no I/O, testable with plain calls. The binary only moves bytes between a socket and that library.
2. A line from a socket is bytes, and only valid UTF-8 can become a `&str`. So the first step of handling a line is `from_utf8`, and a failure is a protocol error, not a crash.
3. Every way a request can be wrong is a value of one enum, `ProtoError`. Its `Display` impl is the error text, so the reply `ERR <reason>` is one `format!` and cannot drift from the variants.
4. The text of `ECHO` must come back byte for byte, so the line is split once, at the first space, and the rest is not touched.
5. `LEN` counts encoded bytes, which is exactly what `str::len` returns; no counting code is needed.
6. Whether to close after a reply depends on the command, so the top-level function returns the reply and a flag together; the server never inspects the command itself.
7. TCP delivers a byte stream, so the server buffers and reads up to each `\n`. A final fragment without `\n` means the client left mid-line, and there is nothing to answer.
8. One thread per connection keeps a slow client from blocking others; `move` gives each thread sole ownership of its socket, so the compiler rules out two threads sharing one by accident.

### The types

```rust
/// A parsed request. `Echo` and `Len` own their text (a `String`), so a
/// `Command` outlives the buffer the line was read into.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command { Ping, Echo(String), Len(String), Quit }

/// Every way a line can fail. The `Display` impl is the text after `ERR `.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtoError { Empty, InvalidUtf8, Unknown(String) }

impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {                       // exhaustive: a new variant cannot lack a message
            ProtoError::Empty => write!(f, "empty line"),
            ProtoError::InvalidUtf8 => write!(f, "invalid utf-8"),
            ProtoError::Unknown(word) => write!(f, "unknown command {word}"),
        }
    }
}
impl std::error::Error for ProtoError {}
```

`PartialEq` is derived so `reply` can ask `cmd == Command::Quit`, and unit tests can compare results with `assert_eq!`.

### Parsing

```rust
pub fn parse(line: &str) -> Result<Command, ProtoError> {
    let line = line.strip_suffix('\n').unwrap_or(line);   // "\n" first...
    let line = line.strip_suffix('\r').unwrap_or(line);   // ...then a "\r" before it
    if line.is_empty() {
        return Err(ProtoError::Empty);
    }
    // split at the FIRST space only: the text keeps every later space (step 4)
    let (word, text) = line.split_once(' ').unwrap_or((line, ""));
    match word {                           // case-sensitive: "ping" is unknown
        "PING" => Ok(Command::Ping),
        "QUIT" => Ok(Command::Quit),
        "ECHO" => Ok(Command::Echo(text.to_string())),   // owned copy
        "LEN" => Ok(Command::Len(text.to_string())),
        _ => Err(ProtoError::Unknown(word.to_string())),
    }
}
```

The order of the two `strip_suffix` calls is what makes `\r\n` work: stripping `\r` first would find `\n` at the end and do nothing. Each `strip_suffix` returns `Option<&str>`, a view into the same line, so no copying happens until `to_string`. The session's `ECHO  two  spaces` splits into `"ECHO"` and `" two  spaces"`, keeping the leading space. `PING extra` parses as `Ping`: the text is ignored for commands that take none.

### Responding

```rust
pub fn byte_len(text: &str) -> usize {
    text.len()                             // UTF-8 bytes, NUL included (step 5)
}

pub fn respond(cmd: &Command) -> String {
    match cmd {
        Command::Ping => "PONG".to_string(),
        Command::Echo(text) => text.clone(),
        Command::Len(text) => byte_len(text).to_string(),
        Command::Quit => "BYE".to_string(),
    }
}
```

`respond` borrows the command (`&Command`), so it cannot consume it and the caller can still compare it with `Quit` afterwards.

### The whole protocol for one line

```rust
pub fn reply(line: &[u8]) -> (String, bool) {
    let answer = std::str::from_utf8(line)
        .map_err(|_| ProtoError::InvalidUtf8)        // step 2: bytes -> &str or a ProtoError
        .and_then(parse)                             // a second fallible step, chained
        .map(|cmd| (respond(&cmd), cmd == Command::Quit));   // step 6: reply and close flag
    match answer {
        Ok(pair) => pair,
        Err(e) => (format!("ERR {e}"), false),       // step 3; errors never close
    }
}
```

The chain reads as the rule: decode, then parse, then answer, with the first failure skipping the rest. Trace `LEN héllo\n`: `from_utf8` gives `Ok("LEN héllo\n")`, `parse` gives `Ok(Len("héllo"))`, `map` gives `("6", false)`. For `JUMP high\n`, `parse` returns `Err(Unknown("JUMP"))`, `map` is skipped, and the reply is `ERR unknown command JUMP`.

### The server: startup

```rust
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let port = match parse_port(&args) {
        Ok(p) => p,
        Err(e) => { eprintln!("echo: {e}"); process::exit(2); }   // usage error: status 2
    };
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => { eprintln!("echo: bind 127.0.0.1:{port}: {e}"); process::exit(1); }
    };
    let addr = listener.local_addr().expect("a bound listener has an address");
    println!("listening on {addr}");       // after bind: with port 0, this is the real port
    let _ = io::stdout().flush();          // stdout to a pipe is buffered: push the line out now
    // ... accept loop below
}

fn parse_port(args: &[String]) -> Result<u16, String> {
    match args {                           // a slice pattern: exactly zero or two arguments
        [] => Ok(7878),
        [flag, value] if flag == "--port" => value.parse::<u16>().map_err(|e| format!("--port {value}: {e}")),
        _ => Err("usage: echo [--port <n>]".to_string()),
    }
}
```

`u16` is the type of a port, so `--port 70000` fails to parse and exits 2 instead of wrapping. The flush matters because a script reading the first line through a pipe would otherwise wait for a buffer that never fills.

### The server: one thread per connection

```rust
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                // `move` hands this thread ownership of `stream`; the loop
                // keeps no reference to it.
                thread::spawn(move || {
                    if let Err(e) = serve(stream) {
                        eprintln!("echo: connection: {e}");   // one connection's error, logged
                    }
                });
            }
            Err(e) => eprintln!("echo: accept: {e}"),         // a failed accept does not stop the server
        }
    }
```

### The server: reading lines

```rust
fn serve(stream: TcpStream) -> io::Result<()> {
    // Two handles on one socket: a buffered reader (a single read() may hold
    // half a line or three lines) and the stream itself for writing.
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = Vec::new();             // reused for every line: one allocation per connection
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line)?;   // exactly one line, however it arrived
        if n == 0 || line.last() != Some(&b'\n') {
            return Ok(()); // EOF, or the client left mid-line: nothing to answer
        }
        let (text, close) = lineproto::reply(&line);
        writer.write_all(text.as_bytes())?;
        writer.write_all(b"\n")?;
        if close {
            return Ok(());                 // QUIT: returning drops both handles, closing the socket
        }
    }
}
```

`read_until` reads into `Vec<u8>`, not `String`, because the bytes are not yet known to be UTF-8; `reply` decides that. Three pipelined lines in one packet sit in the `BufReader`'s buffer and come out one per call; a line split across three packets is assembled by `read_until` before it returns. A client that resets the connection makes a read or write return `Err`, which `?` passes up to the thread, which logs it and ends; the accept loop never sees it.

The full solution is in [`lib.rs`](../../../../course/solutions/primers/lang.04/lineproto/src/lib.rs) and [`main.rs`](../../../../course/solutions/primers/lang.04/echo/src/main.rs).

### Why this design

In cost, each line is handled in time linear in its length; each connection costs one OS thread (a stack of a few MB of address space, mostly untouched) and one buffer.

A library plus a binary, rather than one `main.rs`, means the rules are tested with `reply(b"...")` in microseconds, and the binary is small enough to read in one sitting; the engine uses the same split. `BufReader::read_until`, rather than one `read` per request, is required because TCP has no message boundaries, so anything else is wrong under pipelining or slow networks. A thread per connection, rather than async, needs no runtime and no dependency, at the cost of scaling to hundreds rather than tens of thousands of connections; [Async Rust and tokio](09-async-rust-and-tokio.md) makes the async trade.

One tradeoff is accepted deliberately: there is no limit on line length. A client that sends gigabytes without a newline grows `line` without bound; a production server caps it (for example with `take` on the reader) and answers an error.

## What can go wrong

A correct solution passes a specific set of checks. The workspace must build a library and an `echo` binary, std only, offline, because the system's `rust/` is a workspace of the same shape, and the library's `#[test]` functions must pass, since CI runs `cargo test` from then on. The first exchange must give `PONG`. `ECHO` must return the text after the first space byte for byte, spaces and non-ASCII kept, because a server never edits the payload it relays. `PING\r\n` must parse as `PING`, because HTTP lines end in `\r\n`, as the next section shows. `LEN héllo` must answer 6 bytes, not 5 scalar values, and `LEN a\0b` must answer 3 with the connection still usable.

Errors must be replies, not crashes: an unknown command, an empty line, invalid UTF-8, and a lower-case command all get an answer, never a dead thread. `QUIT` must say `BYE` and then end the stream. On the network side, three lines in one write must get three replies, one line split across three writes must get one reply, a stalled client must not block another, and a client vanishing mid-line must end only its own connection, because clients die all the time.

The pitfalls that break these checks are these:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Reading with one `stream.read(&mut buf)` and treating the result as one line | three pipelined requests get one reply; a slow client's half line gets `ERR unknown command PI` | `serve` wraps the socket in a `BufReader` and reads with `read_until(b'\n', ...)` |
| Serving connections in the accept loop instead of a thread each | the second client hangs until the first one leaves | `thread::spawn(move \|\| serve(stream))` per accepted connection |
| Leaving the `\r` on the line | `nc` and `telnet` users get `ERR unknown command PING` | `parse` strips `\n`, then `\r` |
| `.unwrap()` on `from_utf8`, `parse`, or a socket read | one bad line kills that client's thread; the client sees the connection drop with no reply | `reply` turns every `ProtoError` into `ERR ...`; `serve` returns errors with `?` to its thread, which logs them |
| Splitting the text on whitespace (`split_whitespace`) | `ECHO  two  spaces` comes back as `two spaces` | `split_once(' ')` splits at the first space only |
| Counting Unicode scalar values for `LEN` (`chars().count()`) | `LEN héllo` answers 5 instead of 6 bytes | `byte_len` is `text.len()` |
| Printing the port before binding, or not printing `listening on ...` first | a script cannot find the server's port | `main` binds first, then prints `listening on {addr}` from `local_addr()` and flushes |

> **Warning:** TCP preserves bytes, not writes. Any line protocol that treats one `read` as one message works on localhost in testing and fails under real network timing.

## Where this leads

The engine in [Your first endpoint](../../ai-engineering/23-model-serving/01-your-first-endpoint.md) is this server grown up: `TcpListener`, a thread per connection, a `BufRead` parser, and candle-based inference. The next section, [HTTP/1.1, JSON, and Server-Sent Events](05-http-and-sse.md), keeps the same server shape but speaks HTTP/1.1, JSON, and SSE instead of lines. In *Algorithms*, the [Robin Hood hash map](../15-systems-data-structures/05-robin-hood-hash-map.md) uses traits and generics (`Hash`, `Eq`, `BuildHasher`), and in *AI Engineering*, the [Rust fast BPE tokenizer](../../ai-engineering/07-tokenizers/05-rust-fast-bpe.md) is a Rust crate whose Python parity is checked with shared fixtures.

### Further reading

- [The Rust Programming Language](https://doc.rust-lang.org/book/) (free): chapters 4 (ownership), 6 (enums), 9 (errors), 10 (traits), 14 (workspaces), 16 (threads), 17 (async), and 21 (a multithreaded web server).
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/) (free).
- The [std::net](https://doc.rust-lang.org/std/net/) documentation (free).
- [tokio](https://tokio.rs/): an async runtime for thousands of connections on a few threads, with cancellation and timeouts.
- [tokio-util codecs](https://docs.rs/tokio-util/latest/tokio_util/codec/), especially `LinesCodec`: reusable framers (lines, length prefixes) over async streams.
- [cargo-nextest](https://nexte.st/) for faster parallel test runs, and [Miri](https://github.com/rust-lang/miri) (nightly `cargo miri test`), an interpreter that catches undefined behavior in `unsafe` code.

## Key terms

- **crate**: one Rust compilation unit, either a library or a binary.
- **workspace**: a set of Cargo packages sharing one `Cargo.lock` and one `target/` directory.
- **owner**: the single variable responsible for a value; when it goes out of scope, the value is dropped.
- **move**: transfer of ownership by assignment or by passing a value, after which the old variable is unusable.
- **borrow**: use of a value through a shared (`&`) or exclusive (`&mut`) reference without taking ownership.
- **`Result`**: the enum `Ok(T)` or `Err(E)` that a fallible function returns.
- **trait**: a named set of methods a type promises to implement.
- **panic**: an abort of the current thread, meant for bugs rather than bad input.

## Summary

- Every value has exactly one owner; assigning or passing it moves it, and references borrow it. The compiler checks these rules, so Rust frees memory without a garbage collector and without use-after-free.
- Failure is a value: a fallible function returns `Result<T, E>`, and `?` passes the error up. A server answers errors instead of unwrapping them.
- Traits are promised method sets: `Display` gives an error its text, and `Read`, `Write`, and `BufRead` make a socket, a file, and a byte slice interchangeable.
- A Rust `str` is valid UTF-8; `.len()` counts bytes, `.chars().count()` counts scalar values, and NUL is ordinary content.
- TCP is a byte stream: one `read` can hold half a line or three lines, so a buffered reader that splits on `\n` is the fix.

## Review questions

1. Why does `reply` take `&[u8]` rather than `&str`?

   <details><summary>Answer</summary>

   The bytes come straight from a socket and may not be UTF-8. Taking bytes lets `reply` own the decision and answer `ERR invalid utf-8` instead of making every caller decode first, or worse, unwrap.

   </details>

2. What does `serve` do with the bytes `PI` followed by a closed connection?

   <details><summary>Answer</summary>

   `read_until` returns 2 with no trailing `\n`, so `serve` returns `Ok(())` without replying. The client left mid-line; there is nobody to answer and no complete request to answer.

   </details>

3. Why is `stream` moved into the thread instead of borrowed?

   <details><summary>Answer</summary>

   The thread may outlive the loop iteration that accepted the connection. A borrow would have to outlive the stack variable it points at, which the compiler rejects. Moving gives the thread ownership, so the socket lives exactly as long as the thread needs it and closes when `serve` returns.

   </details>

4. Why strip `\n` before `\r`?

   <details><summary>Answer</summary>

   The line ends with `\r\n`. The last character is `\n`, so `strip_suffix('\r')` first would find nothing to strip; stripping `\n` exposes the `\r`, which the second call then removes.

   </details>

5. `LEN héllo` answers 6. What would `text.chars().count()` answer, and why is that wrong here?

   <details><summary>Answer</summary>

   5, the number of Unicode scalar values. The protocol defines `LEN` as the UTF-8 byte length, the quantity byte-oriented protocols need (a `Content-Length`, a buffer size), and `é` is two bytes.

   </details>
