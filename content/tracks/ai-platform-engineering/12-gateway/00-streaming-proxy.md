<!-- ol:module gw.00 -->
# Tracer gateway: static API-key check, SSE pass-through without buffering, traceparent and X-Request-Id propagation

## Overview

| | |
|---|---|
| **Module** | `gw.00` · build · Go · Pass 1 · 3 to 4 h |
| **You build** | `go/gateway/proxy/proxy.go`: `NewProxy(cfg Config) http.Handler`, `Config`, `Span`, `ParseTraceparent`; then your own `go/cmd/gateway` entry point (learner territory) |
| **Contract** | gateway tier of [`course/contracts/openapi/openai-subset.v0.yaml`](../../../../course/contracts/openapi/openai-subset.v0.yaml) and the `gateway` role in [`course/contracts/spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) |
| **Tests** | `course/tests/go/gw_00/` (what they check: section 4) |
| **Needs** | [`lang.06` Go](../../software-craftsmanship/12-language-and-tool-primers/06-go.md) (reading), `L10.0` your first endpoint (reading: the engine this gateway fronts, reached over HTTP only; chapter `content/courses/build-an-llm/p10-serving/00-your-first-endpoint.md`) |
| **Used by** | `gw.01` server skeleton (wraps `NewProxy`) · `gw.04` SSE streaming proxy (takes this package over) · `dep.00` gateway image · `obs.00` the `gateway.proxy` span |
| **Milestone** | MS-P1 |
| **Optional depth** | [RFC 9110 HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) sections 7.6 and 15.6 (free), [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free), [HTML Living Standard, server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free) |

## Key Takeaways

- The key check comes **first**, before any byte goes upstream, and it **fails closed**: no configured key means every request is a 401 in the OpenAI error shape.
- Keys are compared in **constant time**. An early-exit comparison leaks how many leading bytes matched, which turns guessing a key from impossible into linear work.
- A gateway forwards **bytes**, not events: every chunk the engine sends is written and **flushed** before the next read, so the stream reaches the client byte for byte and token by token.
- Before the first byte, an engine failure is a **503** `no_capacity`; after it, the status is already sent, so the failure becomes an **SSE error event**. A client that leaves cancels the engine's request.
- The gateway continues the caller's **W3C trace** with its **own span id**, so the engine's span is a child of the gateway's `gateway.proxy` span, and it carries one **X-Request-Id** to the engine and back.

## How to work this chapter

```bash
ol start gw.00          # writes go/gateway/proxy/proxy.go with stub bodies (and go/go.mod if absent)
ol tests gw.00          # read the test catalog first
ol check gw.00          # exit code is the verdict
ol diff  gw.00          # after passing: your code against the reference
```

Then write your entry point `go/cmd/gateway/main.go` (section 4, "Your entry point") and declare it in `system.toml` as `[entry].gateway`. MS-P1 starts it, runs `ol conform openapi:v0:gateway` against it, and streams 48 tokens through it.

---

## 1. Why now

Your Rust engine from `L10.0` serves `POST /v1/completions` on a port with no authentication at all: anyone who can reach the port can spend your CPU, and when the engine is deployed on kind in `dep.00`, "anyone who can reach the port" is everyone who can reach the cluster. The engine also has no idea who asked for a completion or how to tie one request across two processes in a trace. Both problems belong in front of the engine, not inside it: one **gateway** process that every client talks to, which checks a key, passes the request through, and streams the answer back without slowing it down. `gw.00` is the thinnest gateway that does that. Later modules grow it (real keys in `gw.02`, rate limits in `gw.03`, a streaming observer in `gw.04`, routing across engines in `gw.05`), and every one of them keeps the rules you prove here.

## 2. Principles

### 2.1 A reverse proxy is two HTTP exchanges

A **reverse proxy** receives a request from a **client** (the **downstream** side) and makes a new request to an **upstream** server, here the engine, then relays the response. There are two separate HTTP exchanges, so each header has to be classified:

| Kind | Definition (RFC 9110 section 7.6.1) | Examples | The gateway |
|---|---|---|---|
| **end-to-end** | meant for the final recipient | `Content-Type`, `Accept`, `X-Request-Id`, the request body | copies them |
| **hop-by-hop** | describes one connection only | `Connection`, `Keep-Alive`, `Transfer-Encoding`, `TE`, `Trailer`, `Upgrade`, `Proxy-Authorization` | drops them; `net/http` sets its own |
| **the gateway's own** | addressed to the gateway | `Authorization` (the gateway's key) | consumes it and never forwards it: the engine tier has no auth, and a key that travels further ends up in engine logs |

The upstream URL is the configured base (`http://127.0.0.1:8081`) plus the request's path and query. A base written with a trailing slash must not produce `//v1/completions`.

### 2.2 A static API key, compared in constant time

In the tracer there is one key, a secret string the gateway reads from `TL_API_KEY` at start-up (`spec/cli-roles.md`). A client presents it as `Authorization: Bearer <key>`. Anything else (no header, another scheme such as `Basic`, a different string) is rejected with status 401 and the contract's error body, and the request never reaches the engine.

Why compare in constant time? The obvious comparison walks the two strings and returns at the first difference. Its running time grows with the number of leading bytes that match, and an attacker can measure it.

| Symbol | Meaning | Type |
|---|---|---|
| $L$ | key length in bytes | integer, 12 for the key `tl_demo_0123` of section 3 |
| $k$ | number of leading bytes of a guess that match the key | integer, $0 \le k \le L$ |
| $t(k)$ | time an early-exit compare takes | seconds, about $c \cdot (k+1)$ for a per-byte cost $c$ |

With an early exit, the attacker guesses the first byte: the one of 256 values that makes $t$ slightly larger is correct. Then the second byte, and so on, for at most $256 \cdot L$ guesses in total (3,072 for $L = 12$, a few minutes of requests even with noise averaged away) instead of $256^L$ (about $7.9 \times 10^{28}$). `crypto/subtle.ConstantTimeCompare(a, b)` touches every byte whatever the contents, so its time depends only on the two lengths. The length itself still leaks, which is acceptable for a key of fixed length.

**Fail closed.** If the gateway was started without a key, the configured key is the empty string. A naive check (`strings.TrimPrefix(header, "Bearer ") == key`) then **accepts** a request with no `Authorization` header at all, because both sides are empty. The rule is: an empty configured key rejects everything.

### 2.3 Streaming: flush every chunk, never re-frame

The engine answers a streamed completion with **server-sent events** (`lang.05`): `Content-Type: text/event-stream`, then one `data: <json>\n\n` event per token, then `data: [DONE]\n\n`, with no `Content-Length` and `Connection: close`: the stream ends when the engine closes the connection. Each event is about 160 bytes. Go's `http.ResponseWriter` buffers the body before sending it (`lang.06`, 2.6). How long does the first token wait in that buffer?

| Symbol | Meaning | Value here |
|---|---|---|
| $B$ | response buffer size | 4096 bytes |
| $e$ | bytes per SSE event | about 160 |
| $r$ | tokens the engine produces per second | about 20 |
| $T_1$ | delay before the client sees the first token | seconds |

Without a flush, nothing is sent until the buffer fills, which takes $\lceil B / e \rceil$ events, so

$$T_1 = \frac{\lceil B / e \rceil}{r} = \frac{\lceil 4096 / 160 \rceil}{20} = \frac{26}{20} = 1.3\ \text{s}$$

instead of the 50 ms the engine needed for one token: the client sees nothing, then 26 tokens at once. Flushing after every write sends each chunk as soon as the engine's chunk arrives. The gateway also never parses and re-emits events: it copies bytes, so whatever framing the engine produced (including the empty-text chunks the v0 contract sends while a UTF-8 character is incomplete) reaches the client unchanged.

### 2.4 Failures before and after the first byte

HTTP sends the status line before the body, so the gateway's options depend on whether it has started answering:

| When | What happened | What the client gets (`openai-subset.v0.yaml`) |
|---|---|---|
| before the first byte | the engine refused the connection or is down | **503** with `{"error": {"type": "server_error", "code": "no_capacity", ...}}` |
| before the first byte | the engine answered a 400 | that 400, status, headers, and body unchanged |
| after the first byte | the engine's connection broke mid-stream | the bytes already sent, then one event `data: {"error": {...}}\n\n`, then the end of the stream |
| any time | the client disconnected | nothing; the engine's request is cancelled so it stops generating |

A stream that simply ends after a failure looks like a short, valid answer, which is why the contract forbids silent truncation. Cancelling on disconnect is a matter of making the upstream request on the incoming request's context (`r.Context()`, `lang.06`, 2.5).

### 2.5 W3C trace context

A **trace** is the record of one request's path through several processes. It is a tree of **spans**: each span is one timed operation in one process, with a **span id** (8 random bytes), the **trace id** of the whole tree (16 random bytes), and the span id of its **parent** (empty for the root). Processes pass the current position in the tree to each other in the `traceparent` header:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             |  |                                |                |
             |  trace id: 32 lowercase hex       parent id: 16     flags: 2 hex
             version: 00                         lowercase hex     (01 = sampled)
```

A value is **invalid** when the version is not `00` (`ff` is forbidden outright), when any field has the wrong length or a character outside `0-9a-f`, or when the trace id or parent id is all zeros. An invalid header is treated as absent.

The gateway's job, per request:

1. If the caller sent a valid `traceparent`, keep its trace id and flags; its parent id becomes the **parent** of the gateway's span. Otherwise start a new trace: a random non-zero trace id, no parent, flags `01`.
2. Make a new random non-zero span id for its own span, named `gateway.proxy`.
3. Send `traceparent: 00-<trace id>-<gateway span id>-<flags>` to the engine. The engine's server span then has the gateway's span as its parent. Sending the caller's parent id instead would make the engine's span a sibling of the gateway's.
4. Forward `tracestate` unchanged when the incoming `traceparent` was valid; drop it otherwise.
5. Record the span (start, end, ids, status) and hand it to whoever exports spans. In `gw.00` that is a callback, `Config.OnSpan`, which your entry point wires to an exporter in `obs.00`.

### 2.6 One request id

`X-Request-Id` is a string that names one request in every log line it touches. The gateway keeps the caller's id when it is printable ASCII (bytes 0x21 to 0x7E) of at most 128 bytes, and otherwise makes one (16 random bytes as 32 hex digits). It sends that id to the engine and sets it on every response, including a 401, so a client can quote it in a bug report.

## 3. Worked example by hand

The gateway runs with `Config{Upstream: "http://127.0.0.1:8081", APIKey: "tl_demo_0123"}` in front of your tracer engine.

**Request 1: no key.**

```
POST /v1/completions HTTP/1.1
Content-Type: application/json

{"model":"tracer","prompt":"Once","max_tokens":2}
```

`Authorization` is missing, so the check fails before anything else happens. The engine receives nothing. The client receives:

```
HTTP/1.1 401 Unauthorized
Content-Type: application/json
X-Request-Id: 9c1f0e4b2a7d4e6f8a3b5c7d9e1f2a3b      (generated: 32 hex digits)

{"error":{"message":"Incorrect or missing API key: send the header Authorization: Bearer KEY.","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}
```

All four keys of the error object are present; `param` is `null`, not missing.

**Request 2: the right key, a stream, and a trace.**

```
POST /v1/completions HTTP/1.1
Authorization: Bearer tl_demo_0123
Content-Type: application/json
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

{"model":"tracer","prompt":"Once","max_tokens":2,"stream":true}
```

Step by step:

1. The key matches (12 bytes against 12 bytes, every byte compared).
2. No `X-Request-Id` came in, so the gateway makes one, say `5e0c7b1d9a2f4c3e8b6d1a0f2e4c6b8d`.
3. The `traceparent` is valid: trace `4bf92f3577b34da6a3ce929d0e0e4736`, parent `00f067aa0ba902b7`, flags `01`. The gateway draws its own span id, say `b7ad6b7169203331`.
4. The engine receives (no `Authorization`):

```
POST /v1/completions HTTP/1.1
Content-Type: application/json
Traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-b7ad6b7169203331-01
X-Request-Id: 5e0c7b1d9a2f4c3e8b6d1a0f2e4c6b8d

{"model":"tracer","prompt":"Once","max_tokens":2,"stream":true}
```

5. The engine answers `200`, `Content-Type: text/event-stream`, and three events. The gateway copies the status and headers, sets `X-Request-Id: 5e0c7b1d9a2f4c3e8b6d1a0f2e4c6b8d`, and then each event, flushing after each one:

```
data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"u","finish_reason":null}]}

data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"p","finish_reason":"length"}]}

data: [DONE]

```

6. When the response ends, `OnSpan` receives `Span{Name: "gateway.proxy", TraceID: "4bf92f3577b34da6a3ce929d0e0e4736", SpanID: "b7ad6b7169203331", ParentSpanID: "00f067aa0ba902b7", RequestID: "5e0c7b1d...", StatusCode: 200}`. In Jaeger (`obs.00`) this is the tree caller span `00f067aa0ba902b7`, then `gateway.proxy` `b7ad6b7169203331`, then the engine's `POST /v1/completions`.

**Request 3: the engine is down.** Same as request 2, but nothing listens on port 8081. `Do` fails with `connection refused` before any byte was sent, so the client gets `503` with `{"error":{"message":"the upstream engine is unavailable: ...","type":"server_error","param":null,"code":"no_capacity"}}`.

These three requests are the first cases of `TestRejectsMissingKey`, `TestV0StreamPassesThroughByteExact` with `TestTraceparentChildOfCaller` and `TestRequestID`, and `TestUpstreamDownIs503`.

## 4. The interface

```go
package proxy // import "tinyllm/gateway/proxy"

type Config struct {
	Upstream string         // engine base URL without /v1, e.g. "http://127.0.0.1:8081"
	APIKey   string         // the one key accepted as "Authorization: Bearer <APIKey>"; "" rejects everything
	Client   *http.Client   // nil: a client with no overall timeout (streams can be long)
	OnSpan   func(Span)     // optional: receives the gateway.proxy span when the response ends
}

type Span struct {
	Name         string // "gateway.proxy"
	TraceID      string // 32 lowercase hex
	SpanID       string // 16 lowercase hex: the parent of the engine's span
	ParentSpanID string // the caller's span id, "" when the gateway started the trace
	RequestID    string
	StatusCode   int
	Start, End   time.Time
}

const SpanName = "gateway.proxy"

func NewProxy(cfg Config) http.Handler
func ParseTraceparent(h string) (traceID, parentID, flags string, ok bool)
```

`ol start gw.00` writes this file with every body stubbed (`panic("todo: gw.00")`); the unexported helpers in it (`authorized`, `copyFlush`, `copyHeaders`, `requestID`, `writeError`, ...) are a suggested decomposition, and only the exported names above are the interface. Use only the standard library.

**Your entry point** (`go/cmd/gateway/main.go`, not checked by `ol check`, run by MS-P1) takes the tracer flags of `spec/cli-roles.md`: `--port` for the API, `--health-port` for `GET /healthz` (200 while up) and `GET /readyz` (200 once `<upstream>/healthz` answers, else 503), and `--upstream`. It reads the key from `TL_API_KEY`, serves `NewProxy` on the API port, and exits 0 on SIGTERM. Declare it (`-C go` runs `go` inside `go/`, where `go.mod` is; from the repo root `go run ./go/cmd/gateway` finds no module):

```toml
[entry]
gateway = ["go", "run", "-C", "go", "./cmd/gateway", "--port", "{port}", "--health-port", "{health_port}", "--upstream", "http://127.0.0.1:{engine.port}"]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestRejectsMissingKey` | unit | request 1: 401, the error shape with all four keys, `X-Request-Id` set, the engine never called | `openapi:v0` case `v0.auth.401`; `gw.02` keeps the shape |
| `TestRejectsWrongKeys` | boundary | a prefix of the key, the key plus a byte, another case, `Basic`, no scheme: all 401; the right key: 200 | `gw.02` replaces the static key, the rules stay |
| `TestEmptyConfiguredKeyRejectsEverything` | boundary | `APIKey: ""` rejects no header, `Bearer `, and `Bearer x` | a gateway started without `TL_API_KEY` fails closed |
| `TestForwardsRequestWithoutTheKey` | unit | method, path, query, body, end-to-end headers reach the engine; `Authorization` does not; a trailing slash on `Upstream` changes nothing | the engine tier stays key-free (`dep.00`) |
| `TestStreamsWithoutBuffering` | unit | the first event reaches the client while the engine holds the second | tokens stream through `gw.04` and MS-P1 |
| `TestV0StreamPassesThroughByteExact` | conformance | a recorded v0 stream arrives byte for byte, `text/event-stream`, 200 | `v0.stream.framing` through the gateway |
| `TestV0ResponsesPassThroughUnchanged` | conformance | a 200 completion and a 400 error keep status, headers, and body | `v0.err.400` through the gateway |
| `TestUpstreamDownIs503` | boundary | request 3: 503 `server_error`/`no_capacity` in the error shape | `gw.05` retries elsewhere on exactly this |
| `TestUpstreamFailureMidStreamSendsErrorEvent` | fault | the engine's socket closes after one event: that event, then one `data: {"error": ...}` event | `gw.04`'s chaos test builds on it |
| `TestRequestID` | unit | absent, kept (`req-42`, 128 bytes), replaced (a space, 129 bytes); the client and the engine see the same id | `obs.*` joins logs on it |
| `TestTraceparentChildOfCaller` | unit | request 2: same trace and flags, a new span id, `tracestate` kept, `OnSpan` gets the span | `obs.00` asserts this parentage in Jaeger |
| `TestTraceparentStartsTrace` | boundary | absent, garbage, all-zero trace id, upper-case hex: a new sampled trace, a root span | the engine never receives an invalid header |
| `TestParseTraceparent` | unit | the W3C rules of 2.5, one row each | the engine's parser (`L10.0`) applies the same rules |
| `TestClientDisconnectCancelsUpstream` | fault | a client that cancels mid-stream cancels the engine's request within 3 s | `gw.04` tightens this to 100 ms |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. comparing only as many bytes as the client sent (or the key's prefix) | `Bearer tl_test` opens the door | `TestRejectsWrongKeys` (mutant `s01`) |
| 2. no empty-key guard, with `TrimPrefix(header, "Bearer ") == key` | a gateway started without `TL_API_KEY` accepts requests with no header, and a bare key without `Bearer` | `TestEmptyConfiguredKeyRejectsEverything`, `TestRejectsWrongKeys` (mutant `s06`) |
| 3. copying every request header, `Authorization` included | the gateway's key appears in engine logs | `TestForwardsRequestWithoutTheKey` (mutant `s03`) |
| 4. `io.Copy(w, resp.Body)` with no flush | the first token waits 1.3 s, then 26 arrive at once (section 2.3) | `TestStreamsWithoutBuffering`, `TestClientDisconnectCancelsUpstream` (mutant `s02`) |
| 5. building the upstream request on `context.Background()` | a client that hangs up leaves the engine generating | `TestClientDisconnectCancelsUpstream` (mutant `s05`) |
| 6. writing the body without `WriteHeader(resp.StatusCode)` first | every engine error reaches the client as 200 | `TestV0ResponsesPassThroughUnchanged` (mutant `s11`) |
| 7. `http.Error(w, err.Error(), 502)` when the engine is down | a plain-text 502 that OpenAI clients cannot parse | `TestUpstreamDownIs503` (mutant `s08`) |
| 8. returning quietly when the stream breaks after the first byte | the client sees a short, valid-looking answer | `TestUpstreamFailureMidStreamSendsErrorEvent` (mutant `s09`) |
| 9. forwarding the caller's `traceparent` unchanged | the engine's span is a sibling of `gateway.proxy`, not its child | `TestTraceparentChildOfCaller`, `TestTraceparentStartsTrace` (mutant `s04`) |
| 10. setting `X-Request-Id` on the response only | the engine's logs cannot be joined to the client's id | `TestRequestID` (mutant `s07`) |
| 11. `Upstream + path` without trimming a trailing slash | the engine receives `//v1/completions` (a `ServeMux` answers 301) | `TestForwardsRequestWithoutTheKey` (mutant `s10`) |
| 12. accepting an all-zero trace or parent id | an invalid trace reaches the engine and the trace backend drops it | `TestParseTraceparent`, `TestTraceparentStartsTrace` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | goroutines, `context`, `net/http`, and the streaming proxy this module hardens |
| Back | `L10.0` | the engine behind the gateway: API v0, its `/healthz`, its `traceparent` parser |
| Forward | `dep.00` | builds the gateway image and runs it on kind in front of the engine (NodePort 30080) |
| Forward | `obs.00` | exports `Config.OnSpan`'s span from your entry point; the trace shows `gateway.proxy` above the engine span |
| Forward | `gw.01` | the server skeleton wraps `NewProxy` in the middleware chain (`requestid -> otel -> recover -> authn -> ... -> proxy`) |
| Forward | `gw.04` | takes `go/gateway/proxy/` over (`upgrades`) and adds `StreamProxy` with a `StreamObserver` for time to first token and usage |

If you skip this module, MS-P1 cannot start a gateway, and `ol check gw.00` stays `todo`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `NewProxy` | `net/http/httputil.ReverseProxy` | `Rewrite` hooks, `X-Forwarded-*`, buffer pools, `FlushInterval: -1` for streaming responses | [`httputil.ReverseProxy`](https://pkg.go.dev/net/http/httputil#ReverseProxy) (free) |
| the static key | Envoy AI Gateway, LiteLLM proxy | per-tenant keys, provider credentials injected upstream, budgets | [Envoy AI Gateway](https://aigateway.envoyproxy.io/) (free), [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) (free) |
| hand-made `traceparent` | OpenTelemetry `otelhttp` | spans and propagation as middleware and transport wrappers | [`otelhttp`](https://pkg.go.dev/go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp) (free) |
| SSE pass-through | Envoy's HTTP/2 streaming, llm-d's inference gateway | backpressure across protocols, KV-aware routing behind one front door | [llm-d](https://github.com/llm-d/llm-d) (free) |
