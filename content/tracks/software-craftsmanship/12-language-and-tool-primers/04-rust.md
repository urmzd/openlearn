# Rust: ownership, traits, Result, cargo workspaces, std TCP

## Overview

| | |
|---|---|
| **Module** | `lang.04` · practice · Rust · Pass 1 · 2 to 3 h reading |
| **Solution** | A Cargo workspace ([`Cargo.toml`](../../../../course/solutions/primers/lang.04/Cargo.toml)) with the library [`lineproto/src/lib.rs`](../../../../course/solutions/primers/lang.04/lineproto/src/lib.rs) (`Command`, `ProtoError`, `parse`, `byte_len`, `respond`, `reply`) and the binary [`echo/src/main.rs`](../../../../course/solutions/primers/lang.04/echo/src/main.rs) (`main`, `parse_port`, `serve`: a line-protocol server on `std::net`, one thread per connection) |
| **Contract** | none: the protocol table and the signatures in section 4 are the contract |
| **Needs** | `lang.02` (processes, exit codes, signals); `lang.03` is useful background for comparing byte strings |
| **Used by** | no call site (a primer): `L10.0` applies the Rust server and workspace patterns, then `ds.05` and `L1.5` |
| **Optional depth** | [The Rust Programming Language](https://doc.rust-lang.org/book/) (free), ch. 4 (ownership), 6 (enums), 9 (errors), 10 (traits), 14 (workspaces), 16 (threads), 21 (a multithreaded web server); [Rust by Example](https://doc.rust-lang.org/rust-by-example/) (free); [std::net](https://doc.rust-lang.org/std/net/) docs (free) |

## Key Takeaways

- Every value has exactly one **owner**; assigning or passing it **moves** it, and references **borrow** it without taking ownership. The compiler checks these rules, which is how Rust frees memory without a garbage collector and without use-after-free: the server moves each socket into its own thread.
- Failure is a value: a function that can fail returns `Result<T, E>`, and `?` passes the error up. `unwrap()` turns an error into a crash; a server answers errors instead.
- A **trait** is a set of methods a type promises. `Display` gives an error its text, `Read`/`Write`/`BufRead` are what make a socket, a file, and a byte slice interchangeable.
- A Rust `str` is valid UTF-8; `.len()` counts its bytes and `.chars().count()` counts Unicode scalar values. NUL is ordinary string content.
- TCP is a **byte stream**, not a message stream: one `read` can hold half a line or three lines. A buffered reader that splits on `\n` is the fix.

## How to read this chapter

Section 2 introduces Rust by comparison with Python and C: Cargo workspaces, ownership, enums, `Result`, traits, UTF-8 strings, TCP, and threads. Section 3 replays a client session by hand and traces one request through the library. Section 4 fixes the protocol, the signatures, and the cases a correct solution must handle. Section 5 walks through the library and the server and explains why each piece is written the way it is. Try to write each function before reading it. To run the solution: `cd primers/lang.04 && cargo test && cargo run -p echo -- --port 7878`.

---

## 1. Why now

The system can compute a bigram's logits in C and call that C from Python (`rt.01`, `M03.1`, `L0.0`), and it has a checkpoint on disk. Nothing can reach it over a network: no process listens on a port, so the Go gateway (`gw.00`) has nothing to forward to and the cluster (`dep.00`) nothing to run. The next module, `L10.0`, writes that process: an inference engine in **Rust** that accepts TCP connections, reads requests, and calls the C `tl_matmul_f32` for every token. It uses only Rust's standard library, so every byte on the wire is visible in its own code. This primer builds the same skeleton with a trivial protocol: a server that reads lines from many clients at once and answers each line, plus one call into C. In `L10.0` the only new ideas are HTTP, the model, and tracing.

## 2. Principles

The earlier primers covered Python (`lang.01`), shell (`lang.02`), and C (`lang.03`). Rust is introduced by comparison with those. Every term is defined before it is used.

### 2.1 Cargo: packages, crates, targets, workspaces

**Cargo** is Rust's build tool and package manager. A **package** is a directory with a `Cargo.toml` manifest naming it, its version, its Rust **edition** (the language revision, `2021` here), and its dependencies. A package holds one or more **crates**: a crate is one compilation unit, either a **library** (`src/lib.rs`, used by other crates) or a **binary** (`src/main.rs`, which has a `fn main` and becomes an executable). Each crate is a **target** of its package.

A **workspace** is a directory whose `Cargo.toml` has a `[workspace]` table listing **member** packages. Members share one `Cargo.lock` (the exact versions of every dependency) and one `target/` directory (all build output). One package depends on another by **path**:

```toml
[dependencies]
lineproto = { path = "../lineproto" }
```

| Command (at the workspace root) | What it does |
|---|---|
| `cargo build` | compiles every member into `target/debug/` (`--release` optimizes, into `target/release/`) |
| `cargo test` | compiles every crate with its `#[test]` functions and runs them |
| `cargo run -p echo -- --port 7878` | builds and runs the binary of package `echo`; arguments after `--` go to the program |
| `cargo clippy` | lints for likely mistakes (optional, recommended) |
| `cargo fmt` | formats every file in the one standard style |

`--offline` tells Cargo not to touch the network; this primer has no dependencies outside the standard library (`std`), so it never needs to. The split into a library and a binary is deliberate: the library has no I/O, so its rules are testable with plain function calls, and the binary only moves bytes between a socket and the library. `L10.0` uses the same split (`tl-serve` is a library plus a `main.rs`), and the system's `rust/Cargo.toml` is a workspace with two members.

### 2.2 Ownership, moves, and borrowing

C makes the programmer free every allocation (`lang.03`); Python frees it when nothing refers to it any more, at run time. Rust decides at **compile time**, with three rules:

1. Every value has exactly one **owner**, a variable.
2. When the owner goes out of scope, the value is **dropped**: its memory (and any socket or file it holds) is released.
3. Assigning a value, or passing it to a function, **moves** it: the new variable is the owner and the old one may not be used again.

```rust
let s = String::from("hello");  // s owns a heap buffer holding 5 bytes
let t = s;                      // moved: t owns the buffer now
println!("{s}");                // compile error: borrow of moved value `s`
```

To use a value without taking it, you **borrow** it with a **reference**: `&s` is a shared (read-only) reference, `&mut s` an exclusive (read-write) one. At any moment a value has either any number of shared references or exactly one exclusive reference, never both, and no reference may outlive the value. Those two rules rule out use-after-free and data races before the program runs.

Two pairs of types follow from this:

| Owned (you can keep it, grow it, move it) | Borrowed view (a pointer and a length) | Holds |
|---|---|---|
| `String` | `&str` | UTF-8 text |
| `Vec<u8>` | `&[u8]` | any bytes |

A function that only reads text takes `&str`; a value that must outlive the line it was read from (the text inside a parsed command) is an owned `String`, made with `.to_string()`. Bytes from a socket are `&[u8]` until you have checked they are UTF-8: `std::str::from_utf8(bytes)` returns `Ok(&str)` or an error, because a Rust `str` is always valid UTF-8.

Types with no heap data, such as integers and `bool`, are **Copy**: assigning duplicates them and the old variable stays usable.

### 2.3 Structs, enums, and `match`

A **struct** groups named fields. An **enum** is a value that is exactly one of several **variants**, and a variant may carry data:

```rust
enum Command { Ping, Echo(String), Len(String), Quit }
```

`match` takes a value apart by variant, and the compiler rejects a `match` that forgets a variant (it must be **exhaustive**). The standard library's two most used enums are `Option<T>` (`Some(value)` or `None`, Rust's replacement for null) and `Result<T, E>`.

`#[derive(Debug, Clone, PartialEq, Eq)]` above a type asks the compiler to write the code for printing it with `{:?}`, copying it with `.clone()`, and comparing it with `==`.

### 2.4 Errors as values: `Result` and `?`

C reports failure through a return code that the caller may ignore (`lang.03`); Python raises an exception that unwinds the stack. Rust returns it: a function that can fail returns `Result<T, E>`, which is `Ok(value)` or `Err(error)`, and the caller cannot reach the value without deciding what to do with the error.

```rust
fn parse(line: &str) -> Result<Command, ProtoError> {
    // return Ok(command) or Err(error), depending on the request
}
```

| Tool | Meaning |
|---|---|
| `expr?` | if `expr` is `Err(e)`, return `Err(e)` from this function now (converting `e` with `From` if needed); otherwise unwrap the `Ok` value |
| `.map_err(f)` | turn one error type into another |
| `.and_then(f)` | if `Ok(v)`, run `f(v)`, which itself returns a `Result` (chains fallible steps) |
| `.unwrap()`, `.expect("why")` | take the `Ok` value or **panic** |

A **panic** stops the current thread with a message. It is for bugs (a state the code proves impossible), never for bad input: a client that sends garbage must get an error reply, and a panic in a server thread drops that client's connection with no answer.

### 2.5 Traits

A **trait** names methods that a type promises to have. A type **implements** a trait with an `impl` block:

```rust
impl fmt::Display for ProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { ... }
}
```

Now `format!("ERR {e}")` works for a `ProtoError`, because `{}` formats any `Display` value. `impl std::error::Error for ProtoError {}` marks it as an error type other code can handle generically.

Traits are also how one function works on many types. The standard I/O traits:

| Trait | Promise | Implemented by |
|---|---|---|
| `Read` | `read(&mut self, buf: &mut [u8]) -> io::Result<usize>`: fills up to `buf.len()` bytes, returns how many; 0 means end of stream | `TcpStream`, `File`, `&[u8]` |
| `Write` | `write_all(&mut self, bytes: &[u8])`, `flush()` | `TcpStream`, `File`, `Vec<u8>` |
| `BufRead` | `read_until(&mut self, byte: u8, buf: &mut Vec<u8>)`, `read_line`: reads through the next delimiter, using an internal buffer | `BufReader<R>` for any `R: Read`, `&[u8]` |

A function written as `fn f<R: BufRead>(r: &mut R)` is **generic**: it works for any type `R` that implements `BufRead`, a socket in the server and a byte slice in a test.

### 2.6 Strings, bytes, and UTF-8

A Rust `str` is valid UTF-8. Its `.len()` is the number of bytes in that encoding; `.chars().count()` is the number of Unicode scalar values. These differ when a character needs more than one UTF-8 byte: `"héllo"` has six bytes and five scalar values. A Rust string can also contain a NUL byte; it is ordinary content because the string carries an explicit length rather than ending at a sentinel.

```rust
let text = "héllo";
assert_eq!(text.len(), 6);              // UTF-8 bytes
assert_eq!(text.chars().count(), 5);   // Unicode scalar values
assert_eq!("a\0b".len(), 3);            // NUL is included
```

The `LEN` command makes this distinction observable over the wire. Since its line parser has already validated UTF-8, `.len()` gives the byte count needed by byte-oriented protocols without unsafe code or a foreign-library call.

### 2.7 TCP with `std::net`

**TCP** gives two programs a reliable, ordered, two-way stream of bytes. A **server** opens a **listening socket** on an address and a **port** (a 16-bit number naming the service on that machine); `127.0.0.1` is this machine only, `0.0.0.0` every network interface (what a container needs). Port 0 asks the operating system for any free port, which is how tests avoid collisions; `listener.local_addr()` then tells you which one you got.

```rust
let listener = TcpListener::bind(("127.0.0.1", port))?;
for conn in listener.incoming() {      // blocks until a client connects
    let stream: TcpStream = conn?;     // one connected client
}
```

A `TcpStream` implements `Read` and `Write`. The key fact: **TCP preserves bytes, not writes**. A client that writes `"PING\n"` and then `"ECHO a\n"` may have both arrive in one `read`, or `"PI"` in one read and `"NG\n"` in the next. A line protocol must therefore buffer: `BufReader::new(stream)` reads large chunks into memory and `read_until(b'\n', &mut line)` hands back exactly one line, however it arrived. `stream.try_clone()` gives a second handle on the same socket, so one handle can sit inside the `BufReader` while the other writes replies.

`read_until` returns `Ok(0)` when the client closed its side of the connection (end of stream); a line that ends without `\n` means the client left mid-line.

### 2.8 Threads

A server that handles one connection at a time stops serving everyone while one client is slow. The simplest fix is one **thread** per connection:

```rust
thread::spawn(move || {
    if let Err(e) = serve(stream) { eprintln!("echo: connection: {e}"); }
});
```

`thread::spawn` runs a **closure** (an anonymous function) on a new operating-system thread. `move` makes the closure take ownership of what it uses, here `stream`, so the new thread owns the socket and the accept loop keeps no reference to it: the compiler would reject sharing it by accident. When `serve` returns, `stream` is dropped and the socket closes. Threads cost memory (a stack each), which is fine for a primer and for the tracer engine; `lang.09` and `L10.5` replace them with **async** tasks.

### 2.9 Tests

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

## 3. Worked example by hand

The session below exercises ping, echo, `LEN`, pipelining, errors, and `QUIT`. Client bytes are on the left, server bytes on the right; `\n` is the byte `0x0a`.

| Client sends | Server answers | Why |
|---|---|---|
| `PING\n` | `PONG\n` | the command word is `PING` |
| `ECHO  two  spaces\n` | ` two  spaces\n` | the text is everything after the **first** space, so it starts with a space |
| `LEN héllo\n` | `6\n` | see below |
| `PING\nECHO a\nLEN abc\n` (one write) | `PONG\n` `a\n` `3\n` | three lines, three replies, in order |
| `JUMP high\n` | `ERR unknown command JUMP\n` | the connection stays open |
| `QUIT\n` | `BYE\n`, then the server closes | the only request that ends the connection |

`LEN héllo` by hand, through `reply(line)`:

1. The line's bytes are `4c 45 4e 20 68 c3 a9 6c 6c 6f 0a`. `é` is the code point U+00E9, which UTF-8 writes as two bytes, `c3 a9`.
2. `std::str::from_utf8` accepts them: `Ok("LEN héllo\n")`.
3. `parse` strips the `\n` (and a `\r` if one came before it), splits at the first space into `"LEN"` and `"héllo"`, and returns `Ok(Command::Len("héllo".to_string()))`.
4. `respond` calls `byte_len("héllo")`. Rust's `str.len()` counts its six UTF-8 bytes: `68 c3 a9 6c 6c 6f`.
5. The reply is `"6"`, and the server writes `6\n`.

`"héllo".chars().count()` is 5 Unicode scalar values; `.len()` is 6 bytes. An interior NUL is counted too: `LEN a\0b` replies `3`.

Try it with the solution's server running (`cargo run -p echo -- --port 7878`) and `nc 127.0.0.1 7878` in another terminal; type the lines. `printf 'PING\nECHO a\nLEN abc\n' | nc 127.0.0.1 7878` sends three lines in one write.

## 4. The interface

```text
primers/lang.04/
  Cargo.toml              [workspace] members = ["lineproto", "echo"]
  lineproto/Cargo.toml    package lineproto, a library
  lineproto/src/lib.rs    the protocol: no I/O, unit-tested with cargo test
  echo/Cargo.toml         package echo, depends on lineproto by path
  echo/src/main.rs        the server: std::net, one thread per connection
```

The protocol. One request is one line ending in `\n`; a `\r` right before the `\n` is part of the line ending. The command word is everything before the first space (the whole line if there is none); the text is everything after it. Commands are case-sensitive.

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

The library's interface, in `lineproto/src/lib.rs`:

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

The binary, `echo/src/main.rs`: `echo --port <n>` binds `127.0.0.1:<n>` (no flag: 7878; port 0: any free port), prints exactly `listening on 127.0.0.1:<port>` as its first stdout line (a script that started it with port 0 reads the port from it), then serves every connection on its own thread: read a line, write `reply(line)` and `\n`, close after `QUIT` or when the client leaves. A bad `--port` exits 2 with a message on stderr.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the workspace builds | conformance | the workspace builds a library and an `echo` binary, std only, offline | the system's `rust/` is a workspace of the same shape (`L10.0`) |
| the unit tests pass | unit | the library's `#[test]` functions pass | CI runs `cargo test` from Pass 1 on |
| ping, pong | unit | section 3's first exchange | one request in, one reply out |
| echo returns the text byte for byte | unit | text after the first space, spaces and non-ASCII kept | a server never edits the payload it relays |
| CRLF line endings are accepted | boundary | `PING\r\n` is `PING` | HTTP lines end in `\r\n` (`lang.05`) |
| `LEN` counts UTF-8 bytes | unit | `LEN héllo` is 6 bytes, not 5 scalar values | byte protocols count encoded bytes |
| `LEN` counts NUL as a byte | boundary | `LEN a\0b` is 3; the connection remains usable | Rust strings carry NUL as content |
| errors are replies, not crashes | boundary | unknown command, empty line, invalid UTF-8, lower case | bad input gets an answer, never a dead thread |
| `QUIT` says `BYE` and closes | unit | `BYE`, then end of stream | the server, not the client, ends some exchanges |
| several lines in one packet | boundary | three lines in one write get three replies | TCP is a byte stream |
| a line split across packets | boundary | one line in three writes gets one reply | same, the other direction |
| two clients at once | fault | a stalled client does not block another | the engine serves many clients (`L10.0`) |
| a client vanishing mid-line leaves the server up | fault | an abrupt disconnect ends only that connection | clients die all the time |

@@SOLUTION@@

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Reading with one `stream.read(&mut buf)` and treating the result as one line | three pipelined requests get one reply; a slow client's half line gets `ERR unknown command PI` | `serve` wraps the socket in a `BufReader` and reads with `read_until(b'\n', ...)` |
| Serving connections in the accept loop instead of a thread each | the second client hangs until the first one leaves | `thread::spawn(move \|\| serve(stream))` per accepted connection |
| Leaving the `\r` on the line | `nc` and `telnet` users get `ERR unknown command PING` | `parse` strips `\n`, then `\r` |
| `.unwrap()` on `from_utf8`, `parse`, or a socket read | one bad line kills that client's thread; the client sees the connection drop with no reply | `reply` turns every `ProtoError` into `ERR ...`; `serve` returns errors with `?` to its thread, which logs them |
| Splitting the text on whitespace (`split_whitespace`) | `ECHO  two  spaces` comes back as `two spaces` | `split_once(' ')` splits at the first space only |
| Counting Unicode scalar values for `LEN` (`chars().count()`) | `LEN héllo` answers 5 instead of 6 bytes | `byte_len` is `text.len()` |
| Printing the port before binding, or not printing `listening on ...` first | a script cannot find the server's port | `main` binds first, then prints `listening on {addr}` from `local_addr()` and flushes |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.03` | compare C byte buffers with Rust's length-carrying strings |
| Back | `lang.02` | processes, exit codes (`echo` exits 2 on a bad flag) |
| Forward | `L10.0` | the engine is this server grown up: `TcpListener`, a thread per connection, a `BufRead` parser, and candle-based inference |
| Forward | `lang.05` | the same server shape, speaking HTTP/1.1, JSON, and SSE instead of lines |
| Forward | `ds.05` | traits and generics (`Hash`, `Eq`, `BuildHasher`) in the Robin Hood map |
| Forward | `L1.5` | a Rust tokenizer crate whose Python parity is checked with shared fixtures |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a thread per connection | async runtimes ([tokio](https://tokio.rs/)) | thousands of connections on a few threads, cancellation, timeouts | `lang.09`; the Book, ch. 17 |
| `BufReader::read_until` line framing | [tokio-util codecs](https://docs.rs/tokio-util/latest/tokio_util/codec/) | reusable framers (lines, length prefixes) over async streams | `LinesCodec` |
| `cargo test` | [cargo-nextest](https://nexte.st/), [Miri](https://github.com/rust-lang/miri) | faster parallel runs; an interpreter that catches undefined behavior in `unsafe` code | nightly `cargo miri test` |
