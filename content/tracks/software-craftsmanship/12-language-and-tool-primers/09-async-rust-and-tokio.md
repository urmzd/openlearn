<!-- ol:module lang.09 -->
# Async Rust and tokio: futures, tasks, channels, cancellation, hyper

## Overview

| | |
|---|---|
| **Module** | `lang.09` · practice · Rust · Pass 7 · 6 to 9 h |
| **You build** | `primers/lang.09/`: the `ticker` package, a library (an SSE producer task feeding a bounded channel, bounded admission with a semaphore, the routes on hyper, a server loop with graceful shutdown) and a `ticker` binary: `GET /health`, `GET /ticks` (an SSE stream), `POST /jobs` (two slots, else 429), `GET /stats` |
| **Contract** | none: the routes table and the signatures in section 4 are the contract; the crates are [tokio](https://docs.rs/tokio/1) and [hyper 1](https://docs.rs/hyper/1) |
| **Tests** | `course/tests/lang.09/check` builds your package offline, runs your own `cargo test`, then runs `test_lang09_async.py`, which streams, disconnects, bursts, and sends SIGTERM to your server (what each test checks: section 4) |
| **Needs** | reading: `lang.04` Rust (ownership, traits, `Result`, threads) ([primer](04-rust.md)), `lang.05` HTTP/1.1 and SSE by hand ([primer](05-http-and-sse.md)) |
| **Used by** | no call site (a primer): `L10.5` applies it next (the engine server is this package grown up), then `lang.10` (tonic runs on tokio) |
| **Milestone** | `MS-P7` |
| **Optional depth** | [Asynchronous Programming in Rust](https://rust-lang.github.io/async-book/) (free); [Tokio tutorial](https://tokio.rs/tokio/tutorial) (free); [Alice Ryhl, Async: what is blocking?](https://ryhl.io/blog/async-what-is-blocking/) (free); [hyper 1.x guides](https://hyper.rs/guides/1/) (free); [withoutboats, Pin](https://without.boats/blog/pin/) (free) |

## Key Takeaways

- An `async fn` returns a **future**, a value that does nothing until an executor polls it; `.await` is a point where the task may pause and its thread runs another task (`test_events_leave_as_produced`).
- A task never blocks its thread: a sleep is `tokio::time::sleep(..).await`, and CPU-heavy work goes to its own thread (L10.5's engine) (`test_bounded_admission_answers_429`).
- A **bounded** channel gives backpressure: `send().await` waits while the channel is full, so a slow reader slows the producer instead of growing memory (`test_slow_reader_bounds_the_producer`).
- Dropping is cancelling: when the client goes away, hyper drops the body, the receiver goes with it, and the producer sees `send` fail or `closed()` fire (`test_disconnect_cancels_the_producer`).
- Graceful shutdown is three steps: stop accepting, let in-flight work finish, exit 0 (`test_sigterm_drains_and_exits_zero`).

## How to work this chapter

```bash
ol start lang.09          # records that you started; the exercise lives in primers/lang.09/
ol tests lang.09          # read the test catalog first
ol check lang.09          # first run writes the starter package (stubs), then checks it
cd primers/lang.09 && cargo test && cargo run -- --port 8080
curl -N 'http://127.0.0.1:8080/ticks?n=5&interval_ms=500'      # one event every half second
for i in 1 2 3; do curl -s -X POST localhost:8080/jobs -d '{"ms":2000}' & done; wait   # two 200s, one 429
```

The first `ol check lang.09` writes the starter files that are missing into `primers/lang.09/`: `Cargo.toml` as given, and `src/lib.rs` and `src/main.rs` with every function body replaced by `todo!("lang.09")`. It never overwrites a file. Builds are offline: tokio, hyper, hyper-util, http-body-util, and bytes come from your local Cargo cache.

---

## 1. Why now

Your tracer engine (`L10.0`) gives each connection an OS thread. That is simple and fine for a handful of clients, and it does not scale to what `L10.5` must serve: thousands of open streams, most of them idle between tokens, plus a CPU-bound engine that must never be interrupted. Every Rust network service in the course from here on (the v1 engine server, the gRPC services of `L10.6`, the client in `lang.10`) is written on **tokio**, the async runtime, and **hyper**, the HTTP library on top of it. This primer teaches the four ideas they rest on: futures, tasks, channels, and cancellation.

## 2. Principles

### 2.1 Threads or tasks

A thread is the OS's unit of scheduling: each has its own stack (megabytes of reserved address space), and switching between them is a trip through the kernel. Ten thousand threads mostly waiting on sockets is ten thousand stacks and constant switching. An async **task** is a much smaller thing: a state machine in a heap allocation, run by a user-space scheduler on a few threads. While a task waits for its socket, it takes no thread at all.

### 2.2 Futures

A **future** is a value that may produce a result later. In Rust it is any type implementing

```rust
trait Future { type Output; fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>; }
enum Poll<T> { Ready(T), Pending }
```

`poll` either finishes (`Ready`) or says "not yet" (`Pending`) after arranging, through `cx.waker()`, to be woken when progress is possible (a socket became readable, a timer fired, a channel got a message). The compiler turns an `async fn` into such a state machine: each `.await` is a state where the function may pause, keeping its local variables inside the future. **Nothing runs until something polls it**: calling an `async fn` only builds the future. `Pin` promises the future will not move in memory once polled, because its states may point into themselves.

### 2.3 The runtime

tokio provides the two halves that make futures run. The **executor** keeps a queue of tasks and polls them on a pool of worker threads (one per core by default; idle workers steal tasks from busy ones). The **reactor** asks the OS (kqueue on macOS, epoll on Linux) which sockets are ready and wakes the tasks waiting on them; the timer wheel does the same for `sleep`. `#[tokio::main]`, or `tokio::runtime::Builder::new_multi_thread().enable_all().build()` plus `block_on`, starts both.

The cardinal rule: **never block a worker thread**. `std::thread::sleep`, a blocking file read, or a long computation inside a task holds the thread, and every other task on it waits. Sleep with `tokio::time::sleep(d).await`; move CPU-bound work to `tokio::task::spawn_blocking` or to a thread of its own, as `L10.5` does with its engine.

### 2.4 Tasks

`tokio::spawn(future)` hands a future to the executor as an independent task and returns a `JoinHandle`. The future must be `'static` (it owns everything it uses, since it may outlive the caller) and `Send` (it may move between worker threads), which is why shared state travels as `Arc<...>`. A `JoinSet` holds many tasks so a server can wait for all of them at shutdown.

### 2.5 Channels and semaphores

Tasks talk through channels. `tokio::sync::mpsc::channel(n)` is **bounded**: it holds at most `n` messages, and `send(x).await` pauses the sender while it is full. That pause is **backpressure**: a producer can never be more than `n` messages ahead of its consumer, so memory stays bounded however slow the reader is. `unbounded_channel` never pauses (and never bounds memory). `oneshot` carries one reply; `watch` broadcasts the latest value (a shutdown flag). `Sender::closed().await` completes when the receiver is gone.

A **semaphore** with `k` permits bounds concurrency: `try_acquire_owned()` takes a permit or fails at once, and dropping the permit gives it back. "At most two jobs; the third gets 429" is a semaphore with 2 permits and `try_acquire`.

### 2.6 Cancellation is dropping

A future that is not polled does not run, and a future that is **dropped** is cancelled at the `.await` where it last paused, its destructors run, and nothing after that point happens. Two everyday sources of cancellation:

- `tokio::select!` polls several futures and drops every one that did not finish first. `select!{ _ = sleep(d) => {}, _ = tx.closed() => return }` sleeps unless the client leaves first.
- hyper drops a response body when the client disconnects. If the body is the receiving end of a channel, the sender learns of it at its next `send` (an `Err`) or through `closed()`.

### 2.7 hyper: a server in four pieces

```rust
let listener = tokio::net::TcpListener::bind(addr).await?;
loop {
    let (stream, _) = listener.accept().await?;
    tokio::spawn(async move {
        let svc = hyper::service::service_fn(|req| handle(app.clone(), req));    // Request<Incoming> -> Response<Body>
        hyper::server::conn::http1::Builder::new().serve_connection(hyper_util::rt::TokioIo::new(stream), svc).await
    });
}
```

A response body is anything implementing `hyper::body::Body`, whose `poll_frame` returns the next chunk of bytes when it exists. `http_body_util::Full` is a whole body; a streamed body wraps a channel receiver and returns `rx.poll_recv(cx)`. A body of unknown length goes out with chunked transfer encoding, each frame as soon as it is produced.

### 2.8 Graceful shutdown

On SIGTERM (`tokio::signal::unix::signal(SignalKind::terminate())`), a server stops accepting (it drops its listener), lets the connections already open finish (they run in a `JoinSet`; wait with a timeout), and returns from `main`, exit status 0. Kubernetes sends SIGTERM before it kills a pod, so this is what makes a rolling deploy cut nothing.

## 3. Worked example by hand

`GET /ticks?n=3&interval_ms=10` with a channel of capacity 4.

| Time | Producer task | Channel | Body (hyper) | Client sees |
|---|---|---|---|---|
| 0 ms | sends `data: {"i":0}\n\n` | 1 queued | `poll_frame` returns it | event 0 |
| 0 ms | `select!`: sleep 10 ms or `closed()` | empty | `poll_frame` is `Pending` | |
| 10 ms | sends `{"i":1}` | 1 | returns it | event 1 |
| 20 ms | sends `{"i":2}` | 1 | returns it | event 2 |
| 30 ms | sends `data: [DONE]\n\n`, then returns (drops `tx`) | 1, closed | returns it, then `None`: body ends | `[DONE]`, then the terminal chunk |

The body is exactly `data: {"i":0}\n\ndata: {"i":1}\n\ndata: {"i":2}\n\ndata: [DONE]\n\n` (`test_hand_example_ticks`). On the wire each event is one chunk: `f\r\ndata: {"i":0}\n\n\r\n` (0x0f = 15 bytes), and the stream ends with `0\r\n\r\n`.

**Disconnect.** With `n=1000`, the client reads two events and closes. hyper notices the closed socket, drops the body, which drops `rx`. The producer is sleeping in `select!`; `tx.closed()` completes, and it returns `Cancelled`: `/stats` shows `cancelled` up by one and `active` back to 0 (`test_disconnect_cancels_the_producer`).

**Admission.** Three `POST /jobs {"ms": 600}` at once: the first two take the two permits and sleep 600 ms; the third finds no permit and gets `429` with `Retry-After: 1` in microseconds (`test_bounded_admission_answers_429`).

## 4. The artifact and its check

```rust
// primers/lang.09/src/lib.rs
pub const CAPACITY: usize = 4;   pub const JOB_SLOTS: usize = 2;
pub fn sse_event(payload: &str) -> String;                        // "data: <payload>\n\n"
pub fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str>;
pub fn json_int(body: &str, key: &str) -> Option<u64>;            // {"ms": 200} -> 200
pub struct Stats { pub started, active, completed, cancelled: AtomicUsize }   // to_json()
pub enum Outcome { Completed, Cancelled }
pub async fn produce(n: usize, interval: Duration, tx: mpsc::Sender<Bytes>, stats: Arc<Stats>) -> Outcome;
pub struct Admission { /* Arc<Semaphore> */ }  impl Admission { pub fn new(slots: usize) -> Self; pub fn try_admit(&self) -> Option<OwnedSemaphorePermit>; }
pub struct App { pub stats: Arc<Stats>, pub jobs: Admission }
pub struct ChannelBody { /* mpsc::Receiver<Bytes> */ }            // impl hyper::body::Body
pub async fn handle(app: Arc<App>, req: Request<Incoming>) -> Result<Response<Out>, Infallible>;
pub async fn serve(listener: tokio::net::TcpListener, app: Arc<App>, shutdown: impl Future<Output = ()>);
```

| Route | Answer |
|---|---|
| `GET /health` | 200 `{"ok":true}` |
| `GET /ticks?n=5&interval_ms=100` | 200 `text/event-stream`: `data: {"i":k}` for k in 0..n, then `data: [DONE]` |
| `POST /jobs` `{"ms": 200}` | 200 `{"ok":true,"slept_ms":200}` after sleeping, or 429 `Retry-After: 1` when both slots are taken |
| `GET /stats` | 200 `{"started","active","completed","cancelled"}` |

`ticker --port 0` prints `listening on 127.0.0.1:<port>` first; SIGTERM stops accepting, waits for open requests, and exits 0.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_crate_builds` | conformance | the package builds offline with a `ticker` binary | the crates L10.5 uses |
| `test_your_cargo_tests_pass` | unit | your own `#[tokio::test]`s pass | testing async code |
| `test_hand_example_ticks` | unit | section 3's body, byte for byte | SSE over hyper |
| `test_health` | unit | a plain JSON route | the handler shape |
| `test_events_leave_as_produced` | unit | the first event arrives long before the last is produced | streaming, not buffering |
| `test_disconnect_cancels_the_producer` | fault | a disconnect cancels the producer within 3 s | L10.5 aborts requests the same way |
| `test_slow_reader_bounds_the_producer` | unit | a reader that reads nothing stops a 200,000-event producer | backpressure through a bounded channel |
| `test_bounded_admission_answers_429` | fault | two slots: the third concurrent job gets 429 and `Retry-After` at once; slots come back | L10.5's 429 |
| `test_sigterm_drains_and_exits_zero` | fault | SIGTERM during a job: the job finishes, exit 0 | rolling deploys |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| `std::thread::sleep` in a handler | other requests on that worker stall; the 429 arrives late | `test_bounded_admission_answers_429` |
| Collecting the whole stream before responding | the first event arrives with the last | `test_events_leave_as_produced` |
| An unbounded channel for a stream | a slow client makes the producer run to the end, memory growing | `test_slow_reader_bounds_the_producer` |
| Sleeping without watching `closed()` | a gone client's producer keeps ticking until its next send | `test_disconnect_cancels_the_producer` |
| A semaphore `acquire().await` instead of `try_acquire` | excess jobs queue silently instead of 429 | `test_bounded_admission_answers_429` |
| Exiting on SIGTERM without waiting for connections | in-flight requests are cut; the client sees a reset | `test_sigterm_drains_and_exits_zero` |
| Holding a `std::sync::Mutex` guard across `.await` | the future is not `Send`, or a deadlock under load | your own tests |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.04` | ownership, `Arc`, `Send`, threads |
| Back | `lang.05` | HTTP/1.1 framing and SSE by hand, the same bytes |
| Forward | `L10.5` | the v1 server: hyper, one engine thread, an SSE task per stream fed by a channel, bounded admission, abort on disconnect, drain on SIGTERM |
| Forward | `lang.10` | tonic, the gRPC library, is hyper and tokio underneath |
| Forward | `L10.6`, `L10.7` | async KV transfer and telemetry exporters |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hyper by hand | [axum](https://github.com/tokio-rs/axum) | routing, extractors, middleware on the same hyper | `axum::Router` |
| a semaphore per route | [tower](https://github.com/tower-rs/tower) layers | rate limits, timeouts, load shedding as composable services | `tower::limit`, `tower::load_shed` |
| `select!` and `closed()` | `tokio_util::sync::CancellationToken` | one token cancels a tree of tasks | the tokio-util docs |
| a hand-written shutdown | hyper-util's graceful shutdown | tracks connections and their in-flight requests for you | `hyper_util::server::graceful` |
