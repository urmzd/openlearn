//! ticker: the lang.09 library. An async HTTP/1.1 server on tokio and
//! hyper with three ideas the engine server (L10.5) is built on:
//!
//! - **A producer task per stream, a bounded channel to the response body.**
//!   The producer awaits `send`, so it can never run more than `CAPACITY`
//!   events ahead of the client (backpressure).
//! - **Cancellation on disconnect.** When the client goes away, hyper drops
//!   the response body, which drops the channel's receiver; the producer's
//!   next `send` fails (or `closed()` fires while it sleeps) and it stops.
//! - **Bounded admission.** At most `JOB_SLOTS` jobs run at once; the next
//!   one is refused with 429 and `Retry-After` instead of queueing forever.
//!
//! Routes:
//!
//! ```text
//! GET  /health                       200 {"ok":true}
//! GET  /ticks?n=5&interval_ms=100    SSE: data: {"i":0} ... data: [DONE]
//! POST /jobs   {"ms": 200}           200 {"ok":true,"slept_ms":200}, or 429
//! GET  /stats                        200 {"started","active","completed","cancelled"}
//! ```

use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::{Body, Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};

/// Events a producer may run ahead of its client.
pub const CAPACITY: usize = 4;

/// Jobs that may run at once.
pub const JOB_SLOTS: usize = 2;

/// One SSE event: `data: <payload>\n\n`.
pub fn sse_event(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

/// The value of `key` in a query string `a=1&b=2`, if present.
pub fn query_param<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
}

/// The integer field `key` of a small flat JSON object such as `{"ms": 200}`.
pub fn json_int(body: &str, key: &str) -> Option<u64> {
    let at = body.find(&format!("\"{key}\""))? + key.len() + 2;
    let rest = body[at..].trim_start().strip_prefix(':')?.trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Counters, shared by every task.
#[derive(Debug, Default)]
pub struct Stats {
    pub started: AtomicUsize,
    pub active: AtomicUsize,
    pub completed: AtomicUsize,
    pub cancelled: AtomicUsize,
}

impl Stats {
    /// `{"started":..,"active":..,"completed":..,"cancelled":..}`.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"started\":{},\"active\":{},\"completed\":{},\"cancelled\":{}}}",
            self.started.load(Ordering::SeqCst),
            self.active.load(Ordering::SeqCst),
            self.completed.load(Ordering::SeqCst),
            self.cancelled.load(Ordering::SeqCst)
        )
    }
}

/// How a producer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Cancelled,
}

/// Sends `n` events `{"i":k}` one `interval` apart, then `[DONE]`, into
/// `tx`. Stops as soon as the receiver is gone, even mid-sleep.
pub async fn produce(n: usize, interval: Duration, tx: mpsc::Sender<Bytes>, stats: Arc<Stats>) -> Outcome {
    stats.started.fetch_add(1, Ordering::SeqCst);
    stats.active.fetch_add(1, Ordering::SeqCst);
    let mut outcome = Outcome::Completed;
    for i in 0..=n {
        let ev = if i == n { sse_event("[DONE]") } else { sse_event(&format!("{{\"i\":{i}}}")) };
        if tx.send(Bytes::from(ev)).await.is_err() {
            outcome = Outcome::Cancelled;
            break;
        }
        if i < n && !interval.is_zero() {
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = tx.closed() => {
                    outcome = Outcome::Cancelled;
                    break;
                }
            }
        }
    }
    stats.active.fetch_sub(1, Ordering::SeqCst);
    match outcome {
        Outcome::Completed => stats.completed.fetch_add(1, Ordering::SeqCst),
        Outcome::Cancelled => stats.cancelled.fetch_add(1, Ordering::SeqCst),
    };
    outcome
}

/// At most `slots` holders at once; `try_admit` never waits.
#[derive(Clone, Debug)]
pub struct Admission {
    sem: Arc<Semaphore>,
}

impl Admission {
    pub fn new(slots: usize) -> Admission {
        Admission { sem: Arc::new(Semaphore::new(slots)) }
    }

    /// A permit (released when dropped), or `None` when every slot is taken.
    pub fn try_admit(&self) -> Option<OwnedSemaphorePermit> {
        Arc::clone(&self.sem).try_acquire_owned().ok()
    }
}

/// Shared server state.
#[derive(Debug)]
pub struct App {
    pub stats: Arc<Stats>,
    pub jobs: Admission,
}

impl App {
    pub fn new() -> App {
        App { stats: Arc::new(Stats::default()), jobs: Admission::new(JOB_SLOTS) }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// The response body type: whole bodies and streams alike.
pub type Out = BoxBody<Bytes, Infallible>;

/// A streamed body fed by a channel.
pub struct ChannelBody {
    rx: mpsc::Receiver<Bytes>,
}

impl Body for ChannelBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        self.rx.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b))))
    }
}

fn reply(status: StatusCode, body: String) -> Response<Out> {
    let mut r = Response::new(Full::new(Bytes::from(body)).boxed());
    *r.status_mut() = status;
    r.headers_mut().insert("content-type", "application/json".parse().expect("static"));
    r
}

/// Routes one request.
pub async fn handle(app: Arc<App>, req: Request<Incoming>) -> Result<Response<Out>, Infallible> {
    let query = req.uri().query().unwrap_or("").to_string();
    Ok(match (req.method(), req.uri().path()) {
        (&Method::GET, "/health") => reply(StatusCode::OK, "{\"ok\":true}".to_string()),
        (&Method::GET, "/stats") => reply(StatusCode::OK, app.stats.to_json()),
        (&Method::GET, "/ticks") => {
            let n = query_param(&query, "n").and_then(|v| v.parse().ok()).unwrap_or(5usize);
            let ms = query_param(&query, "interval_ms").and_then(|v| v.parse().ok()).unwrap_or(100u64);
            let (tx, rx) = mpsc::channel(CAPACITY);
            tokio::spawn(produce(n, Duration::from_millis(ms), tx, Arc::clone(&app.stats)));
            let mut r = Response::new(ChannelBody { rx }.boxed());
            r.headers_mut().insert("content-type", "text/event-stream".parse().expect("static"));
            r.headers_mut().insert("cache-control", "no-cache".parse().expect("static"));
            r
        }
        (&Method::POST, "/jobs") => {
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
            tokio::time::sleep(Duration::from_millis(ms)).await;
            drop(permit);
            reply(StatusCode::OK, format!("{{\"ok\":true,\"slept_ms\":{ms}}}"))
        }
        _ => reply(StatusCode::NOT_FOUND, "{\"error\":\"not found\"}".to_string()),
    })
}

/// Serves `listener` until `shutdown` resolves, then stops accepting and
/// waits (up to 10 s) for the connections already open to finish.
pub async fn serve(listener: tokio::net::TcpListener, app: Arc<App>, shutdown: impl Future<Output = ()>) {
    let mut conns = tokio::task::JoinSet::new();
    tokio::pin!(shutdown);
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            acc = listener.accept() => {
                let Ok((stream, _)) = acc else { continue };
                let _ = stream.set_nodelay(true);
                let app = Arc::clone(&app);
                conns.spawn(async move {
                    let svc = service_fn(move |req| handle(Arc::clone(&app), req));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(stream), svc).await;
                });
            }
        }
    }
    drop(listener);
    let _ = tokio::time::timeout(Duration::from_secs(10), async { while conns.join_next().await.is_some() {} }).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn produce_sends_n_events_then_done() {
        let (tx, mut rx) = mpsc::channel(CAPACITY);
        let stats = Arc::new(Stats::default());
        let out = produce(3, Duration::ZERO, tx, Arc::clone(&stats)).await;
        assert_eq!(out, Outcome::Completed);
        let mut got = Vec::new();
        while let Some(b) = rx.recv().await {
            got.push(String::from_utf8(b.to_vec()).unwrap());
        }
        assert_eq!(got, ["data: {\"i\":0}\n\n", "data: {\"i\":1}\n\n", "data: {\"i\":2}\n\n", "data: [DONE]\n\n"]);
    }

    #[tokio::test]
    async fn produce_stops_when_the_receiver_is_gone() {
        let (tx, rx) = mpsc::channel(CAPACITY);
        drop(rx);
        let stats = Arc::new(Stats::default());
        assert_eq!(produce(100, Duration::from_millis(1), tx, Arc::clone(&stats)).await, Outcome::Cancelled);
        assert_eq!(stats.cancelled.load(Ordering::SeqCst), 1);
        assert_eq!(stats.active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn query_and_json_helpers() {
        assert_eq!(query_param("n=3&interval_ms=50", "interval_ms"), Some("50"));
        assert_eq!(query_param("n=3", "x"), None);
        assert_eq!(json_int("{\"ms\": 250}", "ms"), Some(250));
        assert_eq!(json_int("{\"ms\": -1}", "ms"), None);
    }
}
