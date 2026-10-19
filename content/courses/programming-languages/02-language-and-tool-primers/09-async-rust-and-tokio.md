# Async Rust and tokio: futures, tasks, channels, cancellation, hyper

The tracer engine of [Your first endpoint](../../build-an-llm/11-serving/01-your-first-endpoint.md) in *Build Your Own LLM System* gives each connection an OS thread. That is simple, and fine for a handful of clients, but it does not scale to what the production engine server must handle: thousands of open streams, most of them idle between tokens, plus a CPU-bound engine that must never be interrupted. From here on, every Rust network service in these books (the v1 engine server, the gRPC services, the gRPC client of the next section) is written on **tokio**, the async runtime, and **hyper**, the HTTP library on top of it.

This section teaches the four ideas they rest on: futures, tasks, channels, and cancellation. You build a small server called `ticker` with four routes: `GET /health`, `GET /ticks` (a Server-Sent Events stream fed by a producer task through a bounded channel), `POST /jobs` (two slots, else 429, through a semaphore), and `GET /stats`, with a server loop that shuts down gracefully. The engine server of [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../../build-an-llm/11-serving/06-openai-server-on-tokio.md) is this package grown up.

It builds on [Rust: ownership, traits, Result, cargo workspaces, std TCP](04-rust.md) for ownership, traits, `Result`, and threads, and on [HTTP/1.1, JSON, and Server-Sent Events from the wire up](05-http-and-sse.md), whose bytes you will see again here. The solution is in [`course/solutions/primers/lang.09/`](../../../../course/solutions/primers/lang.09/): the library [`src/lib.rs`](../../../../course/solutions/primers/lang.09/src/lib.rs) (`produce`, `Admission`, `ChannelBody`, `handle`, `serve`) and the binary [`src/main.rs`](../../../../course/solutions/primers/lang.09/src/main.rs) (`main`). The package depends on [tokio](https://docs.rs/tokio/1), [hyper 1](https://docs.rs/hyper/1), hyper-util, http-body-util, and bytes ([`Cargo.toml`](../../../../course/solutions/primers/lang.09/Cargo.toml)), and builds offline from a warm Cargo cache. There is no separate contract; the routes and signatures in "Designing the API" are the contract. The solution runs like this:

```bash
cd primers/lang.09 && cargo test && cargo run -- --port 8080
curl -N 'http://127.0.0.1:8080/ticks?n=5&interval_ms=500'      # one event every half second
for i in 1 2 3; do curl -s -X POST localhost:8080/jobs -d '{"ms":2000}' & done; wait   # two 200s, one 429
```

> **Learning objectives**
> - Explain what a future is, why nothing runs until it is polled, and what the executor and reactor of a runtime each do.
> - Keep worker threads unblocked: async sleeps, and CPU-heavy work on its own thread.
> - Use a bounded channel for backpressure and a semaphore for bounded admission that answers 429 instead of queueing.
> - Explain cancellation as dropping, and make a producer notice a departed client even while it sleeps.
> - Write a hyper server with a streamed body and a three-step graceful shutdown.

## Threads or tasks

Start with why threads are the wrong unit here. A thread is the OS's unit of scheduling: each has its own stack (megabytes of reserved address space), and switching between them is a trip through the kernel. Ten thousand threads mostly waiting on sockets is ten thousand stacks and constant switching. An async **task** is a much smaller thing: a state machine in a heap allocation, run by a user-space scheduler on a few threads. While a task waits for its socket, it takes no thread at all.

## Futures

The state machine inside a task is a **future**, a value that may produce a result later. In Rust it is any type implementing

```rust
trait Future { type Output; fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>; }
enum Poll<T> { Ready(T), Pending }
```

`poll` either finishes (`Ready`) or says "not yet" (`Pending`) after arranging, through `cx.waker()`, to be woken when progress is possible: a socket became readable, a timer fired, a channel got a message. The compiler turns an `async fn` into such a state machine. Each `.await` is a state where the function may pause, keeping its local variables inside the future, and a point where the thread may go and run another task.

The most important consequence is easy to forget: **nothing runs until something polls it**. Calling an `async fn` only builds the future. `Pin` promises the future will not move in memory once polled, because its states may point into themselves.

## The runtime

Something has to do the polling. tokio provides the two halves that make futures run. The **executor** keeps a queue of tasks and polls them on a pool of worker threads (one per core by default; idle workers steal tasks from busy ones). The **reactor** asks the OS (kqueue on macOS, epoll on Linux) which sockets are ready and wakes the tasks waiting on them; the timer wheel does the same for `sleep`. `#[tokio::main]`, or `tokio::runtime::Builder::new_multi_thread().enable_all().build()` plus `block_on`, starts both.

That design gives the cardinal rule: **never block a worker thread**. `std::thread::sleep`, a blocking file read, or a long computation inside a task holds the thread, and every other task on it waits. Sleep with `tokio::time::sleep(d).await`; move CPU-bound work to `tokio::task::spawn_blocking` or to a thread of its own, as the engine server does with its engine.

## Tasks

`tokio::spawn(future)` hands a future to the executor as an independent task and returns a `JoinHandle`. The future must be `'static` (it owns everything it uses, since it may outlive the caller) and `Send` (it may move between worker threads), which is why shared state travels as `Arc<...>`. A `JoinSet` holds many tasks, so a server can wait for all of them at shutdown.

## Channels and semaphores

Tasks talk through channels. `tokio::sync::mpsc::channel(n)` is **bounded**: it holds at most `n` messages, and `send(x).await` pauses the sender while it is full. That pause is **backpressure**: a producer can never be more than `n` messages ahead of its consumer, so memory stays bounded however slow the reader is, and a slow reader slows the producer instead of growing memory. `unbounded_channel` never pauses (and never bounds memory). `oneshot` carries one reply; `watch` broadcasts the latest value (a shutdown flag). `Sender::closed().await` completes when the receiver is gone.

A **semaphore** with `k` permits bounds concurrency. `try_acquire_owned()` takes a permit or fails at once, and dropping the permit gives it back. "At most two jobs; the third gets 429" is a semaphore with 2 permits and `try_acquire`.

## Cancellation is dropping

Rust has no cancel call for futures, and it does not need one. A future that is not polled does not run, and a future that is **dropped** is cancelled at the `.await` where it last paused: its destructors run, and nothing after that point happens. Cancellation comes from two everyday sources.

The first is `tokio::select!`, which polls several futures and drops every one that did not finish first. `select!{ _ = sleep(d) => {}, _ = tx.closed() => return }` sleeps unless the client leaves first.

The second is hyper, which drops a response body when the client disconnects. If the body is the receiving end of a channel, the sender learns of it at its next `send` (an `Err`) or through `closed()`.

## hyper: a server in four pieces

With those ideas, a hyper server is short. You bind a listener, accept in a loop, spawn a task per connection, and serve the connection with a service function:

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

## Graceful shutdown

The last piece is stopping well. On SIGTERM (`tokio::signal::unix::signal(SignalKind::terminate())`), a server does three things: it stops accepting (it drops its listener), lets the connections already open finish (they run in a `JoinSet`; wait with a timeout), and returns from `main`, exit status 0. Kubernetes sends SIGTERM before it kills a pod, so this is what makes a rolling deploy cut nothing.

## Working it by hand

Trace `GET /ticks?n=3&interval_ms=10` with a channel of capacity 4. Each row shows the producer task, the channel, the body hyper polls, and what the client sees:

| Time | Producer task | Channel | Body (hyper) | Client sees |
|---|---|---|---|---|
| 0 ms | sends `data: {"i":0}\n\n` | 1 queued | `poll_frame` returns it | event 0 |
| 0 ms | `select!`: sleep 10 ms or `closed()` | empty | `poll_frame` is `Pending` | |
| 10 ms | sends `{"i":1}` | 1 | returns it | event 1 |
| 20 ms | sends `{"i":2}` | 1 | returns it | event 2 |
| 30 ms | sends `data: [DONE]\n\n`, then returns (drops `tx`) | 1, closed | returns it, then `None`: body ends | `[DONE]`, then the terminal chunk |

The body is exactly `data: {"i":0}\n\ndata: {"i":1}\n\ndata: {"i":2}\n\ndata: [DONE]\n\n`. On the wire each event is one chunk, `f\r\ndata: {"i":0}\n\n\r\n` (0x0f = 15 bytes), and the stream ends with `0\r\n\r\n`.

Now a disconnect. With `n=1000`, the client reads two events and closes. hyper notices the closed socket and drops the body, which drops `rx`. The producer is sleeping in `select!`; `tx.closed()` completes, and it returns `Cancelled`. `/stats` shows `cancelled` up by one and `active` back to 0.

Finally, admission. Send three `POST /jobs {"ms": 600}` at once. The first two take the two permits and sleep 600 ms; the third finds no permit and gets `429` with `Retry-After: 1` at once, without waiting for a slot.

## Designing the API

The library's public surface is small:

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

The routes answer as follows:

| Route | Answer |
|---|---|
| `GET /health` | 200 `{"ok":true}` |
| `GET /ticks?n=5&interval_ms=100` | 200 `text/event-stream`: `data: {"i":k}` for k in 0..n, then `data: [DONE]` |
| `POST /jobs` `{"ms": 200}` | 200 `{"ok":true,"slept_ms":200}` after sleeping, or 429 `Retry-After: 1` when both slots are taken |
| `GET /stats` | 200 `{"started","active","completed","cancelled"}` |

`ticker --port 0` prints `listening on 127.0.0.1:<port>` first. SIGTERM stops accepting, waits for open requests, and exits 0.

## Building it step by step

### Reasoning from first principles

Walk through the reasoning before the code, and try to predict each function from it.

1. **A stream is a producer and a consumer at different speeds.** The producer makes events on a timer; the consumer is a socket that drains as fast as the client reads. Putting a bounded channel between them (capacity `CAPACITY = 4`) means the producer can be at most four events ahead: `send().await` pauses it when the client is slow.
2. **The response body must be the channel's receiver.** Then hyper pulls each event out of the channel as soon as it exists (streaming, not buffering), and when the client disconnects, hyper drops the body, which drops the receiver, which is how the producer learns the client is gone.
3. **The producer must notice a gone client even while it sleeps.** A `send` fails only when it is attempted; between events the producer is in `sleep`. So it sleeps inside `select!` against `tx.closed()`, which completes the moment the receiver is dropped.
4. **The producer is its own task.** The handler must return the response (headers first) immediately and let the body fill later, so the producer is `tokio::spawn`ed and owns its `Sender` and an `Arc<Stats>`.
5. **Admission must refuse, not queue.** "At most two jobs, the third gets 429" is a semaphore with two permits and a non-waiting `try_acquire`. The permit is a value: holding it is holding the slot, and dropping it frees the slot on every path out, including a cancelled handler.
6. **Never block a worker.** The job sleeps with `tokio::time::sleep(..).await`, so a sleeping job takes no thread and the 429 for the third job is answered at once.
7. **Shutdown is a `select!` between accepting and a signal.** On the signal, the loop breaks, the listener is dropped (no new connections), and the server waits, with a bound, for the connection tasks it spawned into a `JoinSet`.

### The producer

The producer counts itself in `Stats`, sends `n` events plus `[DONE]`, and stops at the first sign the receiver is gone:

```rust
pub async fn produce(n: usize, interval: Duration, tx: mpsc::Sender<Bytes>, stats: Arc<Stats>) -> Outcome {
    stats.started.fetch_add(1, Ordering::SeqCst);
    stats.active.fetch_add(1, Ordering::SeqCst);
    let mut outcome = Outcome::Completed;
    for i in 0..=n {
        // i == n is the terminal [DONE] event, so n events plus one.
        let ev = if i == n { sse_event("[DONE]") } else { sse_event(&format!("{{\"i\":{i}}}")) };
        // Backpressure: waits while 4 events are queued. Err means the receiver
        // (the response body) was dropped: the client is gone.
        if tx.send(Bytes::from(ev)).await.is_err() {
            outcome = Outcome::Cancelled;
            break;
        }
        if i < n && !interval.is_zero() {  // no sleep after the last event
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                // Completes as soon as the receiver is dropped: cancel mid-sleep.
                _ = tx.closed() => {
                    outcome = Outcome::Cancelled;
                    break;
                }
            }
        }
    }
    // One exit path, so active always goes back down and exactly one of
    // completed/cancelled goes up.
    stats.active.fetch_sub(1, Ordering::SeqCst);
    match outcome {
        Outcome::Completed => stats.completed.fetch_add(1, Ordering::SeqCst),
        Outcome::Cancelled => stats.cancelled.fetch_add(1, Ordering::SeqCst),
    };
    outcome
}
```

Returning drops `tx`. With no sender left, the receiver's next `poll_recv` returns `None`, which is the end of the body: the last row of the hand trace.

### The streamed body

The whole `Body` implementation is one line, because `mpsc::Receiver::poll_recv` already has the right shape: `Ready(Some(x))`, `Ready(None)` at the end, or `Pending` with the waker registered.

```rust
pub struct ChannelBody {
    rx: mpsc::Receiver<Bytes>,
}

impl Body for ChannelBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        // Some(bytes) -> one data frame (one chunk on the wire); None -> end of body.
        self.rx.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b))))
    }
}
```

### Admission

`Admission` is a cloneable wrapper around an `Arc<Semaphore>`. `try_acquire_owned` returns a permit that is not tied to a borrow of the semaphore, so it can be held across `.await`s in the handler.

```rust
impl Admission {
    pub fn new(slots: usize) -> Admission {
        Admission { sem: Arc::new(Semaphore::new(slots)) }
    }

    /// A permit (released when dropped), or `None` when every slot is taken.
    pub fn try_admit(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.sem).try_acquire_owned().ok()  // never waits
    }
}
```

### The routes

The handler is one `match` on method and path. All responses share the boxed body type `Out`, so a `Full` JSON body and a `ChannelBody` stream can come from the same function.

```rust
pub async fn handle(app: Arc<App>, req: Request<Incoming>) -> Result<Response<Out>, Infallible> {
    let query = req.uri().query().unwrap_or("").to_string();
    Ok(match (req.method(), req.uri().path()) {
        (&Method::GET, "/health") => reply(StatusCode::OK, "{\"ok\":true}".to_string()),
        (&Method::GET, "/stats") => reply(StatusCode::OK, app.stats.to_json()),
        (&Method::GET, "/ticks") => {
            let n = query_param(&query, "n").and_then(|v| v.parse().ok()).unwrap_or(5usize);
            let ms = query_param(&query, "interval_ms").and_then(|v| v.parse().ok()).unwrap_or(100u64);
            let (tx, rx) = mpsc::channel(CAPACITY);  // bounded: the backpressure
            tokio::spawn(produce(n, Duration::from_millis(ms), tx, Arc::clone(&app.stats)));
            // Return now: headers go out at once, events follow as produced.
            let mut r = Response::new(ChannelBody { rx }.boxed());
            r.headers_mut().insert("content-type", "text/event-stream".parse().expect("static"));
            r.headers_mut().insert("cache-control", "no-cache".parse().expect("static"));
            r
        }
        (&Method::POST, "/jobs") => {
            // Admission before reading the body: a refused job costs nothing.
            let Some(permit) = app.jobs.try_admit() else {
                let mut r = reply(StatusCode::TOO_MANY_REQUESTS, "{\"error\":\"busy\"}".to_string());
                r.headers_mut().insert("retry-after", "1".parse().expect("static"));
                return Ok(r);
            };
            let body = match req.into_body().collect().await {
                Ok(b) => String::from_utf8_lossy(&b.to_bytes()).into_owned(),
                Err(_) => String::new(),
            };
            let Some(ms) = json_int(&body, "ms") else {
                return Ok(reply(StatusCode::BAD_REQUEST, "{\"error\":\"ms must be a non-negative integer\"}".to_string()));
            };
            tokio::time::sleep(Duration::from_millis(ms)).await;  // never thread::sleep
            drop(permit);  // the slot is free before the reply is written
            reply(StatusCode::OK, format!("{{\"ok\":true,\"slept_ms\":{ms}}}"))
        }
        _ => reply(StatusCode::NOT_FOUND, "{\"error\":\"not found\"}".to_string()),
    })
}
```

The early `return Ok(r)` after a bad body also drops `permit`, so the slot comes back on every path. `json_int` reads only digits after the colon, so `{"ms": -1}` is `None` and answers 400.

### The server loop and shutdown

```rust
pub async fn serve(listener: tokio::net::TcpListener, app: Arc<App>, shutdown: impl Future<Output = ()>) {
    let mut conns = tokio::task::JoinSet::new();  // every connection task, so we can wait for them
    tokio::pin!(shutdown);                        // polled repeatedly by select!, so it must not move
    loop {
        tokio::select! {
            _ = &mut shutdown => break,           // SIGTERM or Ctrl-C
            acc = listener.accept() => {
                let Ok((stream, _)) = acc else { continue };  // one bad accept is not fatal
                let _ = stream.set_nodelay(true);             // send each small SSE chunk now
                let app = Arc::clone(&app);
                conns.spawn(async move {
                    let svc = service_fn(move |req| handle(Arc::clone(&app), req));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(stream), svc).await;
                });
            }
        }
    }
    drop(listener);  // step 1: stop accepting; new connections are refused
    // Step 2: let open connections finish, but never wait forever.
    let _ = tokio::time::timeout(Duration::from_secs(10), async { while conns.join_next().await.is_some() {} }).await;
}   // step 3: return; main returns; exit 0
```

`main` builds a multi-thread runtime, binds `127.0.0.1:<port>`, prints `listening on ...` and flushes stdout (so a parent process reading the port is not stuck behind a buffer), and passes a `shutdown` future that completes on `SIGTERM` or Ctrl-C.

The full solution is in [`src/lib.rs`](../../../../course/solutions/primers/lang.09/src/lib.rs) and [`src/main.rs`](../../../../course/solutions/primers/lang.09/src/main.rs).

### Why this design

Memory per stream is bounded by `CAPACITY` events plus one small task, however slow the client. A thread per stream would cost a stack each, and an unbounded channel would let a stalled client grow the queue to all $n$ events (200,000 in the slow-reader case).

Cancellation is free because it is dropping: there is no flag to check and no cancel message to route. The only extra code is the `closed()` arm, which turns "noticed at the next send" into "noticed now".

The design answers 429 instead of queueing. `acquire().await` would also bound concurrency, but excess jobs would wait invisibly, holding connections, and the client could not tell overload from slowness. `try_acquire` plus `Retry-After` pushes the decision back to the client, which is what the engine server does under load.

A hand-written shutdown costs a `JoinSet` and a `select!`; hyper-util's graceful helper (see the production equivalents below) also tells idle keep-alive connections to close. The 10 s cap trades a possibly cut long stream for a bounded shutdown, inside Kubernetes' 30 s grace period.

Finally, `SeqCst` everywhere on the counters is stronger than needed (`Relaxed` would do for independent counters), but it costs nothing measurable here and removes a question.

## What can go wrong

A correct solution passes these checks:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The crate builds | conformance | the package builds offline with a `ticker` binary | the crates the engine server uses |
| The package's own tests pass | unit | the `#[tokio::test]`s in `lib.rs` pass | testing async code |
| The hand example | unit | the hand-traced body, byte for byte | SSE over hyper |
| Health | unit | a plain JSON route | the handler shape |
| Events leave as produced | unit | the first event arrives long before the last is produced | streaming, not buffering |
| A disconnect cancels the producer | fault | a disconnect cancels the producer within 3 s | the engine server aborts requests the same way |
| A slow reader bounds the producer | unit | a reader that reads nothing stops a 200,000-event producer | backpressure through a bounded channel |
| Bounded admission answers 429 | fault | two slots: the third concurrent job gets 429 and `Retry-After` at once; slots come back | the engine server's 429 |
| SIGTERM drains and exits 0 | fault | SIGTERM during a job: the job finishes, exit 0 | rolling deploys |

The mistakes that most often break these cases are listed here with their symptoms and the line of the solution that prevents each:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `std::thread::sleep` in a handler | other requests on that worker stall; the 429 arrives late | `tokio::time::sleep(..).await` in `/jobs` (step 6) |
| Collecting the whole stream before responding | the first event arrives with the last | the body is a `ChannelBody`; the producer is spawned and the response returned at once |
| An unbounded channel for a stream | a slow client makes the producer run to the end, memory growing | `mpsc::channel(CAPACITY)` with `CAPACITY = 4` |
| Sleeping without watching `closed()` | a gone client's producer keeps ticking until its next send | `select!` between `sleep` and `tx.closed()` |
| A semaphore `acquire().await` instead of `try_acquire` | excess jobs queue silently instead of 429 | `try_acquire_owned` in `Admission::try_admit`, 429 with `Retry-After: 1` |
| Exiting on SIGTERM without waiting for connections | in-flight requests are cut; the client sees a reset | `serve` drops the listener, then joins the `JoinSet` with a 10 s timeout |
| Holding a `std::sync::Mutex` guard across `.await` | the future is not `Send`, or a deadlock under load | no mutex at all: shared state is atomics and a semaphore |

> **Warning:** Awaiting the producer inside the handler instead of spawning it looks harmless and deadlocks. The body is not polled until the handler returns, so the producer fills the 4-slot channel and waits forever on a reader that never starts.

## Where this leads

This section leaned on [Rust: ownership, traits, Result, cargo workspaces, std TCP](04-rust.md) for ownership, `Arc`, `Send`, and threads, and on [HTTP/1.1, JSON, and Server-Sent Events from the wire up](05-http-and-sse.md) for HTTP/1.1 framing and SSE by hand: the same bytes, now produced by hyper.

In *Build Your Own LLM System*, [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../../build-an-llm/11-serving/06-openai-server-on-tokio.md) is the v1 server: hyper, one engine thread, an SSE task per stream fed by a channel, bounded admission, abort on disconnect, and drain on SIGTERM. [Disaggregated prefill and decode](../../build-an-llm/11-serving/07-disaggregated-prefill-decode.md) and [Serving metrics and tracing](../../build-an-llm/11-serving/08-serving-metrics-and-tracing.md) use the same tools for async KV transfer and telemetry exporters. Next in this chapter, [Protocol Buffers and gRPC](10-protocol-buffers-and-grpc.md) uses tonic, the gRPC library, which is hyper and tokio underneath.

Each hand-built piece has a production equivalent:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hyper by hand | [axum](https://github.com/tokio-rs/axum) | routing, extractors, middleware on the same hyper | `axum::Router` |
| a semaphore per route | [tower](https://github.com/tower-rs/tower) layers | rate limits, timeouts, load shedding as composable services | `tower::limit`, `tower::load_shed` |
| `select!` and `closed()` | `tokio_util::sync::CancellationToken` | one token cancels a tree of tasks | the tokio-util docs |
| a hand-written shutdown | hyper-util's graceful shutdown | tracks connections and their in-flight requests for you | `hyper_util::server::graceful` |

### Further reading

- [Asynchronous Programming in Rust](https://rust-lang.github.io/async-book/) (free)
- [Tokio tutorial](https://tokio.rs/tokio/tutorial) (free)
- [Alice Ryhl, Async: what is blocking?](https://ryhl.io/blog/async-what-is-blocking/) (free)
- [hyper 1.x guides](https://hyper.rs/guides/1/) (free)
- [withoutboats, Pin](https://without.boats/blog/pin/) (free)

## Key terms

- **future**: a value implementing `Future`, which does nothing until polled and then returns `Ready` or `Pending`.
- **task**: a future handed to the executor with `tokio::spawn`, scheduled independently on the worker threads.
- **executor**: the part of the runtime that polls ready tasks on a pool of worker threads.
- **reactor**: the part of the runtime that asks the OS which sockets are ready and wakes the waiting tasks.
- **bounded channel**: a channel with a fixed capacity whose `send` waits while it is full.
- **backpressure**: a slow consumer slowing its producer, so memory between them stays bounded.
- **semaphore permit**: a value representing one slot; holding it holds the slot, dropping it frees it.
- **cancellation by drop**: a dropped future stops at its last `.await`, running destructors and nothing after.
- **graceful shutdown**: stop accepting, let in-flight work finish within a bound, then exit 0.

## Summary

- An `async fn` returns a future, a value that does nothing until an executor polls it; `.await` is a point where the task may pause and its thread runs another task.
- A task never blocks its thread: a sleep is `tokio::time::sleep(..).await`, and CPU-heavy work goes to its own thread.
- A bounded channel gives backpressure: `send().await` waits while the channel is full, so a slow reader slows the producer instead of growing memory.
- Dropping is cancelling: when the client goes away, hyper drops the body, the receiver goes with it, and the producer sees `send` fail or `closed()` fire.
- Bounded admission with `try_acquire` refuses with 429 instead of queueing, and the permit frees its slot on every path out.
- Graceful shutdown is three steps: stop accepting, let in-flight work finish, exit 0.

## Review questions

1. Why does the producer need `tx.closed()` in its `select!` if `send` already fails when the client is gone?
   <details><summary>Answer</summary>
   `send` fails only when it is called. With `interval_ms=60000` the producer would sleep a minute after the client left before trying to send and finding out. `closed()` completes when the receiver is dropped, so `select!` cancels the sleep immediately.
   </details>
2. Where exactly is the backpressure, and what would `mpsc::unbounded_channel` change?
   <details><summary>Answer</summary>
   In `tx.send(..).await`: with 4 events queued it returns `Pending` until hyper's `poll_frame` takes one, which happens only as the socket drains. An unbounded channel never pauses, so the producer runs to the end and every unsent event sits in memory.
   </details>
3. Why is the job's slot released on every path out of the `/jobs` handler, including when the client disconnects mid-sleep?
   <details><summary>Answer</summary>
   The slot is the `OwnedSemaphorePermit` value. If the handler returns early, its local `permit` is dropped; if the client disconnects, hyper drops the handler's future at its `.await`, which drops its locals, including the permit. Dropping a permit returns it to the semaphore.
   </details>
4. Why does the solution `tokio::spawn` the producer instead of awaiting it in the handler?
   <details><summary>Answer</summary>
   The handler must return the `Response` so hyper can send headers and start polling the body. If it awaited the producer first, the producer would block on a full channel that nobody reads (the body is not being polled yet): with more than 4 events (counting `[DONE]`) it would deadlock, and with fewer the client would get everything at the end.
   </details>
5. After SIGTERM, what happens to a connection that is in the middle of a 600 ms job, and to a new connection attempt?
   <details><summary>Answer</summary>
   The loop breaks and drops the listener, so the new connection is refused. The running connection task is in the `JoinSet`; the server waits for it (up to 10 s), the job finishes and is answered, then `serve` returns and `main` exits 0.
   </details>
6. What exact bytes does `GET /ticks?n=3&interval_ms=10` produce as the body, and how is the first event framed on the wire?
   <details><summary>Answer</summary>
   The body is `data: {"i":0}\n\ndata: {"i":1}\n\ndata: {"i":2}\n\ndata: [DONE]\n\n`. Each event is one chunk; the first is `f\r\ndata: {"i":0}\n\n\r\n`, where 0x0f = 15 is its byte length, and the stream ends with the terminal chunk `0\r\n\r\n`.
   </details>
