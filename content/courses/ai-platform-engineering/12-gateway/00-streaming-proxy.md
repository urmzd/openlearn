# The tracer gateway

The Rust engine from [Your first endpoint](../../build-an-llm/p10-serving/00-your-first-endpoint.md) in *Build Your Own LLM System* serves `POST /v1/completions` on a port with no authentication at all. Anyone who can reach the port can spend its CPU, and once the engine is deployed on kind in [Tracer deploy](../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) (in *Infrastructure*), "anyone who can reach the port" means everyone who can reach the cluster. The engine also has no idea who asked for a completion, or how to tie one request together across two processes in a trace.

Both problems belong in front of the engine, not inside it. The answer is one **gateway** process that every client talks to: it checks a key, passes the request through, and streams the answer back without slowing it down. This section builds the thinnest gateway that does that. Later sections grow it (real keys in [API keys and scopes](../08-authorization-and-access-control/01-api-keys-and-scopes.md), rate limits in [Rate limiting](../08-authorization-and-access-control/02-rate-limiting.md), a streaming observer in [The SSE streaming proxy](04-sse-streaming-proxy.md), routing across engines in [Routing, cascades, and failover](05-routing.md)), and every one of them keeps the rules established here.

You should be able to read Go before starting; [Go: packages, interfaces, goroutines, channels, context, net/http, testing](../../programming-languages/02-language-and-tool-primers/06-go.md) in *Programming Languages* covers what you need. You also need to know the engine this gateway fronts, which it reaches over HTTP only. By the end, the finished entry point, [`entry/go/cmd/gateway/main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go), runs this proxy as `gateway --port N --health-port N --upstream URL` with the key taken from `TL_API_KEY`. With it, one streamed request runs end to end from client through gateway to engine, and 48 tokens stream through without buffering. In the tracer, the first version of the system that runs every layer end to end, this is the gateway layer.

The solution is [`go/gateway/proxy/proxy.go`](../../../../course/solutions/history/gw.00/go/gateway/proxy/proxy.go) as it stood at this point (the current file, taken over by the streaming proxy, is [`go/gateway/proxy/proxy.go`](../../../../course/solutions/go/gateway/proxy/proxy.go)): `NewProxy(cfg Config) http.Handler`, `Config`, `Span`, and `ParseTraceparent`. The behaviour it implements is the gateway tier of [`course/contracts/openapi/openai-subset.v0.yaml`](../../../../course/contracts/openapi/openai-subset.v0.yaml) and the `gateway` role in [`course/contracts/spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md).

> **Learning objectives**
> - Explain why the key check comes first, fails closed, and compares in constant time, and compute how much an early-exit comparison leaks.
> - Classify headers as end-to-end, hop-by-hop, or the gateway's own, and build the upstream request accordingly.
> - Compute the first-token delay that an unflushed response buffer causes, and write a read-write-flush loop that removes it.
> - Choose the right failure signal before and after the first byte of a response.
> - Continue a caller's W3C trace with the gateway's own span, and propagate one `X-Request-Id`.

## A reverse proxy is two HTTP exchanges

A **reverse proxy** receives a request from a **client** (the **downstream** side), makes a new request to an **upstream** server (here the engine), and relays the response. Because there are two separate HTTP exchanges, every header has to be classified by whom it is for. RFC 9110 section 7.6.1 gives the first two categories; the third is specific to a gateway:

| Kind | Definition | Examples | The gateway |
|---|---|---|---|
| **end-to-end** | meant for the final recipient | `Content-Type`, `Accept`, `X-Request-Id`, the request body | copies them |
| **hop-by-hop** | describes one connection only | `Connection`, `Keep-Alive`, `Transfer-Encoding`, `TE`, `Trailer`, `Upgrade`, `Proxy-Authorization` | drops them; `net/http` sets its own |
| **the gateway's own** | addressed to the gateway | `Authorization` (the gateway's key) | consumes it and never forwards it |

The last row deserves a word. The engine tier has no authentication, so it has no use for the key, and a key that travels further ends up in engine logs.

The upstream URL is the configured base (`http://127.0.0.1:8081`) plus the request's path and query. A base written with a trailing slash must not produce `//v1/completions`.

## A static API key, compared in constant time

In the tracer there is one key, a secret string the gateway reads from `TL_API_KEY` at start-up (`spec/cli-roles.md`). A client presents it as `Authorization: Bearer <key>`. Anything else (no header, another scheme such as `Basic`, a different string) is rejected with status 401 and the contract's error body, and the request never reaches the engine.

Why compare in constant time? The obvious comparison walks the two strings and returns at the first difference. Its running time grows with the number of leading bytes that match, and an attacker can measure it. To see how much that leaks, name the quantities:

| Symbol | Meaning | Type |
|---|---|---|
| $L$ | key length in bytes | integer, 12 for the key `tl_demo_0123` used below |
| $k$ | number of leading bytes of a guess that match the key | integer, $0 \le k \le L$ |
| $t(k)$ | time an early-exit compare takes | seconds, about $c \cdot (k+1)$ for a per-byte cost $c$ |

With an early exit, the attacker guesses the first byte: of the 256 possible values, the one that makes $t$ slightly larger is correct. Then the second byte, and so on, for at most $256 \cdot L$ guesses in total. That is 3,072 for $L = 12$, a few minutes of requests even with noise averaged away, instead of $256^L$ (about $7.9 \times 10^{28}$). `crypto/subtle.ConstantTimeCompare(a, b)` touches every byte whatever the contents, so its time depends only on the two lengths. The length itself still leaks, which is acceptable for a key of fixed length.

There is a second trap in the key check, and it is about configuration rather than timing.

> **Warning: fail closed.** If the gateway was started without a key, the configured key is the empty string. A naive check (`strings.TrimPrefix(header, "Bearer ") == key`) then **accepts** a request with no `Authorization` header at all, because both sides are empty. The rule is: an empty configured key rejects everything.

## Streaming: flush every chunk, never re-frame

Once a request is let through, the response has to come back as fast as the engine produces it. The engine answers a streamed completion with **server-sent events** (see [HTTP/1.1, JSON, and Server-Sent Events from the wire up](../../programming-languages/02-language-and-tool-primers/05-http-and-sse.md)): `Content-Type: text/event-stream`, then one `data: <json>\n\n` event per token, then `data: [DONE]\n\n`, with no `Content-Length` and `Connection: close`. The stream ends when the engine closes the connection. Each event is about 160 bytes.

Go's `http.ResponseWriter` buffers the body before sending it (section 2.6 of the Go primer). How long does the first token wait in that buffer? Here are the quantities:

| Symbol | Meaning | Value here |
|---|---|---|
| $B$ | response buffer size | 4096 bytes |
| $e$ | bytes per SSE event | about 160 |
| $r$ | tokens the engine produces per second | about 20 |
| $T_1$ | delay before the client sees the first token | seconds |

Without a flush, nothing is sent until the buffer fills, which takes $\lceil B / e \rceil$ events, so

$$T_1 = \frac{\lceil B / e \rceil}{r} = \frac{\lceil 4096 / 160 \rceil}{20} = \frac{26}{20} = 1.3\ \text{s}$$

instead of the 50 ms the engine needed for one token. The client sees nothing, then 26 tokens at once. Flushing after every write sends each chunk as soon as the engine's chunk arrives.

The gateway also never parses and re-emits events. It copies bytes, so whatever framing the engine produced, including the empty-text chunks the v0 contract sends while a UTF-8 character is incomplete, reaches the client unchanged.

## Failures before and after the first byte

Streaming changes how failures can be reported. HTTP sends the status line before the body, so the gateway's options depend on whether it has started answering. The v0 contract (`openai-subset.v0.yaml`) fixes what the client gets in each case:

| When | What happened | What the client gets |
|---|---|---|
| before the first byte | the engine refused the connection or is down | **503** with `{"error": {"type": "server_error", "code": "no_capacity", ...}}` |
| before the first byte | the engine answered a 400 | that 400, status, headers, and body unchanged |
| after the first byte | the engine's connection broke mid-stream | the bytes already sent, then one event `data: {"error": {...}}\n\n`, then the end of the stream |
| any time | the client disconnected | nothing; the engine's request is cancelled so it stops generating |

A stream that simply ends after a failure looks like a short, valid answer, which is why the contract forbids silent truncation. Cancelling on disconnect is a matter of making the upstream request on the incoming request's context (`r.Context()`, section 2.5 of the Go primer).

## W3C trace context

The gateway's last responsibility is making one request visible as one story across processes. A **trace** is the record of one request's path through several processes. It is a tree of **spans**: each span is one timed operation in one process, with a **span id** (8 random bytes), the **trace id** of the whole tree (16 random bytes), and the span id of its **parent** (empty for the root). Processes pass the current position in the tree to each other in the `traceparent` header:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             |  |                                |                |
             |  trace id: 32 lowercase hex       parent id: 16     flags: 2 hex
             version: 00                         lowercase hex     (01 = sampled)
```

A value is **invalid** when the version is not `00` (`ff` is forbidden outright), when any field has the wrong length or a character outside `0-9a-f`, or when the trace id or parent id is all zeros. An invalid header is treated as absent.

Per request, the gateway does five things:

1. If the caller sent a valid `traceparent`, keep its trace id and flags; its parent id becomes the **parent** of the gateway's span. Otherwise start a new trace: a random non-zero trace id, no parent, flags `01`.
2. Make a new random non-zero span id for its own span, named `gateway.proxy`.
3. Send `traceparent: 00-<trace id>-<gateway span id>-<flags>` to the engine. The engine's server span then has the gateway's span as its parent. Sending the caller's parent id instead would make the engine's span a sibling of the gateway's.
4. Forward `tracestate` unchanged when the incoming `traceparent` was valid; drop it otherwise.
5. Record the span (start, end, ids, status) and hand it to whoever exports spans. Here that is a callback, `Config.OnSpan`, which the entry point wires to an exporter in [One trace: gateway to engine in Jaeger](../../systems/04-observability/00-one-trace.md) (in *Systems and Architecture*).

Logs need a join key of their own. `X-Request-Id` is a string that names one request in every log line it touches. The gateway keeps the caller's id when it is printable ASCII (bytes 0x21 to 0x7E) of at most 128 bytes, and otherwise makes one (16 random bytes as 32 hex digits). It sends that id to the engine and sets it on every response, including a 401, so a client can quote it in a bug report.

## Following three requests by hand

With the rules in place, follow three requests through a gateway running with `Config{Upstream: "http://127.0.0.1:8081", APIKey: "tl_demo_0123"}` in front of the tracer engine.

The first request carries no key:

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

The second request has the right key, asks for a stream, and continues a trace:

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

5. The engine answers `200`, `Content-Type: text/event-stream`, and three events. The gateway copies the status and headers, sets `X-Request-Id: 5e0c7b1d9a2f4c3e8b6d1a0f2e4c6b8d`, and then copies each event, flushing after each one:

```
data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"u","finish_reason":null}]}

data: {"id":"cmpl-7","object":"text_completion","created":1760000000,"model":"tracer","choices":[{"index":0,"text":"p","finish_reason":"length"}]}

data: [DONE]

```

6. When the response ends, `OnSpan` receives `Span{Name: "gateway.proxy", TraceID: "4bf92f3577b34da6a3ce929d0e0e4736", SpanID: "b7ad6b7169203331", ParentSpanID: "00f067aa0ba902b7", RequestID: "5e0c7b1d...", StatusCode: 200}`. In Jaeger this is the tree: caller span `00f067aa0ba902b7`, then `gateway.proxy` `b7ad6b7169203331`, then the engine's `POST /v1/completions`.

The third request is the same as the second, but nothing listens on port 8081. `Do` fails with `connection refused` before any byte was sent, so the client gets `503` with `{"error":{"message":"the upstream engine is unavailable: ...","type":"server_error","param":null,"code":"no_capacity"}}`.

## The interface

Those three traces pin down what the package must expose. It is small:

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

Only the exported names are the interface; the unexported helpers of the solution (`authorized`, `copyFlush`, `copyHeaders`, `requestID`, `writeError`, and so on) are one decomposition of it. The package uses only the standard library.

The entry point, [`entry/go/cmd/gateway/main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go), takes the tracer flags of `spec/cli-roles.md`: `--port` for the API, `--health-port` for `GET /healthz` (200 while up) and `GET /readyz` (200 once `<upstream>/healthz` answers, else 503), and `--upstream`. It reads the key from `TL_API_KEY`, serves `NewProxy` on the API port, and exits 0 on SIGTERM. In the assembled system it sits at `go/cmd/gateway` inside the `go/` module (where `go.mod` is), so from `go/` it runs as `go run ./cmd/gateway --port 8080 --health-port 9464 --upstream http://127.0.0.1:8081`. From the repository root, `go run ./go/cmd/gateway` finds no module.

## Building it step by step

Before reading the code, it helps to derive its shape from the rules, because each rule fixes one piece of it.

1. **Nothing may reach the engine before the key is checked.** The engine has no authentication, so the gateway is the only line of defence. That fixes the order of `serve`: request id first (so even a 401 is traceable), then the key, and only then any upstream work.
2. **The check must be fail-closed and constant-time.** An empty configured key is a misconfiguration, so it rejects everything; the comparison uses `subtle.ConstantTimeCompare` so its time does not depend on how many bytes matched.
3. **The upstream request is a new request, so build it from parts.** Method, path, query, and body are copied; end-to-end headers are copied; hop-by-hop headers and the gateway's own `Authorization` are not. The trace and request id headers are set by the gateway, not copied.
4. **The trace must branch at the gateway.** The engine's parent must be the gateway's span, so the gateway mints a span id and puts it in the outgoing `traceparent`, keeping the caller's trace id and flags.
5. **Latency is the product; buffering is the enemy.** The body is copied in a read-write-flush loop so each chunk leaves as soon as it arrives, and bytes are copied rather than parsed so framing never changes.
6. **The response status is a one-way door.** Before the first byte, any failure can be a proper status (503); after it, the only honest signal is an SSE error event. The client's context is the upstream request's context, so a client that leaves cancels the engine.

The version of the package described here is preserved as it was before the streaming proxy took it over: [`history/gw.00/go/gateway/proxy/proxy.go`](../../../../course/solutions/history/gw.00/go/gateway/proxy/proxy.go). The current [`go/gateway/proxy/proxy.go`](../../../../course/solutions/go/gateway/proxy/proxy.go) keeps every rule below and adds the stream observer.

### Normalizing once in `NewProxy`

The trailing slash is trimmed when the handler is built, not per request, so `http://127.0.0.1:8081/` and `http://127.0.0.1:8081` behave identically. A nil client becomes one with no overall timeout, because a stream may legitimately last minutes.

```go
func NewProxy(cfg Config) http.Handler {
	client := cfg.Client
	if client == nil {
		client = &http.Client{} // no Timeout: the client's disconnect ends a stream, not a clock
	}
	base := strings.TrimRight(cfg.Upstream, "/") // never "//v1/completions"
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serve(w, r, cfg, client, base)
	})
}
```

### Request id, span, and key

`serve` begins by setting up what every exit path needs. The span is reported from a `defer`, so every exit (401, 503, success) produces one, and `statusWriter` records the status for it.

```go
span := Span{Name: SpanName, Start: time.Now(), RequestID: requestID(r.Header.Get("X-Request-Id"))}
sw := &statusWriter{ResponseWriter: w}
defer func() {
	if cfg.OnSpan != nil {
		span.End = time.Now()
		span.StatusCode = sw.code // what the client actually received
		cfg.OnSpan(span)
	}
}()
w.Header().Set("X-Request-Id", span.RequestID) // set before any answer, so a 401 carries it too

// 1. The key, before anything leaves the gateway.
if !authorized(r.Header.Get("Authorization"), cfg.APIKey) {
	writeError(sw, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key",
		"Incorrect or missing API key: send the header Authorization: Bearer KEY.")
	return
}
```

The check itself encodes both the fail-closed guard and the constant-time comparison:

```go
func authorized(header, key string) bool {
	if key == "" {
		return false // fail closed: a gateway started without TL_API_KEY lets nobody in
	}
	const scheme = "bearer "
	if len(header) < len(scheme) || !strings.EqualFold(header[:len(scheme)], scheme) {
		return false // no header, "Basic ...", or a bare key without a scheme
	}
	presented := strings.TrimSpace(header[len(scheme):])
	// Whole-string compare whose time depends only on the lengths; a prefix of
	// the key or the key plus a byte has a different length and fails.
	return subtle.ConstantTimeCompare([]byte(presented), []byte(key)) == 1
}
```

### The trace and the upstream request

With the caller admitted, `serve` resolves the trace position and builds the upstream request from parts:

```go
traceID, parent, flags, ok := ParseTraceparent(r.Header.Get("Traceparent"))
if !ok { // absent or invalid: start a new sampled trace with the gateway as root
	traceID, parent, flags = randomHex(16), "", "01"
}
span.TraceID, span.ParentSpanID, span.SpanID = traceID, parent, randomHex(8)

target := base + r.URL.EscapedPath() // escaped, so encoded characters survive the hop
if r.URL.RawQuery != "" {
	target += "?" + r.URL.RawQuery
}
// r.Context(): if the client hangs up, this request is cancelled and the engine stops.
up, err := http.NewRequestWithContext(r.Context(), r.Method, target, r.Body)
// ... a build error is a 502 no_capacity
up.ContentLength = r.ContentLength
copyHeaders(up.Header, r.Header) // end-to-end headers only (hop-by-hop skipped)
up.Header.Del("Authorization")   // the gateway's key stops here
up.Header.Del("Tracestate")
if ok && r.Header.Get("Tracestate") != "" { // tracestate belongs to a valid traceparent only
	up.Header.Set("Tracestate", r.Header.Get("Tracestate"))
}
up.Header.Set("Traceparent", "00-"+traceID+"-"+span.SpanID+"-"+flags) // gateway span = engine's parent
up.Header.Set("X-Request-Id", span.RequestID)                         // the same id upstream
```

### Before and after the first byte

The last part of `serve` is where the one-way door of the status line shows up in code:

```go
resp, err := client.Do(up)
if err != nil {
	if r.Context().Err() != nil {
		return // the client left; nobody is listening
	}
	// No byte sent yet: we can still choose the status.
	writeError(sw, http.StatusServiceUnavailable, "server_error", "no_capacity",
		"the upstream engine is unavailable: "+err.Error())
	return
}
defer resp.Body.Close()

copyHeaders(w.Header(), resp.Header)
w.Header().Set("X-Request-Id", span.RequestID)
sw.WriteHeader(resp.StatusCode) // the engine's status, 200 or 400, before any body byte
err = copyFlush(sw, resp.Body)
if err != nil && r.Context().Err() == nil &&
	strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
	// After the first byte the status is gone; say so in-band instead of truncating.
	_, _ = io.WriteString(sw, `data: {"error":{"message":"the upstream stream failed","type":"server_error","param":null,"code":null}}`+"\n\n")
	_ = http.NewResponseController(sw).Flush()
}
```

### The streaming loop

A plain `io.Copy` would leave the bytes in the response buffer. `copyFlush` writes and flushes each read instead. `statusWriter.Unwrap` lets `http.NewResponseController` reach the real writer's `Flush` through the wrapper.

```go
func copyFlush(w http.ResponseWriter, body io.Reader) error {
	rc := http.NewResponseController(w)
	buf := make([]byte, 32*1024)
	for {
		n, err := body.Read(buf) // returns as soon as the engine sent anything
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return werr // the client is gone
			}
			if ferr := rc.Flush(); ferr != nil && !errors.Is(ferr, http.ErrNotSupported) {
				return ferr
			}
		}
		if errors.Is(err, io.EOF) {
			return nil // the engine closed the stream normally
		}
		if err != nil {
			return err // a mid-stream failure: the caller writes the error event
		}
	}
}
```

### Validators and the entry point

The remaining helpers enforce the validity rules from earlier. `ParseTraceparent` rejects any version other than `00`, wrong lengths, upper-case or non-hex characters (`lowerHex`), and all-zero ids (`strings.Trim(id, "0") == ""`). `requestID` keeps a caller's id of 1 to 128 bytes in 0x21 to 0x7E and otherwise makes one. `randomHex` loops until the bytes are not all zero, so a generated id is never invalid. `writeError` encodes `param` and `code` as `*string`, so an absent value is `null`, never a missing key.

[`entry/go/cmd/gateway/main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go) is the tracer form of the `gateway` role: `gateway --port N --health-port N --upstream URL`. It reads the key from `TL_API_KEY` (logging that every request will be rejected when it is empty), serves `NewProxy` with `OnSpan` wired to the span exporter on the API port, and serves `/healthz` (always 200) and `/readyz` (200 only when `<upstream>/healthz` answers 200) on the health port. On SIGTERM it shuts both servers down with a 10 s grace period for in-flight streams, then exits 0.

### Why this design

The cost per request is small: one constant-time compare of 12 bytes, one header copy, a few random bytes, and a 32 KiB buffer reused for the whole stream. A flush per chunk costs a write system call per token, which at about 20 tokens per second is nothing next to the latency it saves.

Why not use `httputil.ReverseProxy`? It would do the header classification and can stream with `FlushInterval: -1`, but the before-and-after-first-byte error rules, the 503 body shape, and the span need hooks in several places (`ErrorHandler`, `ModifyResponse`, `Rewrite`). Writing the 60 lines makes each rule visible; the further reading at the end points at the library for production.

Why copy bytes instead of parsing SSE? Parsing and re-emitting would change framing (empty-text chunks, whitespace), add latency, and couple the gateway to every engine's event format. The gateway's job is to be transparent.

Why a static key at all? It is the thinnest thing that closes the open port. It is replaced in [API keys and scopes](../08-authorization-and-access-control/01-api-keys-and-scopes.md), but every rule here (the 401 shape, constant time, fail closed, the key never forwarded) carries over.

One tradeoff is accepted knowingly: the error event after a mid-stream failure has `code: null`, because the cause is unknown at that point. The client knows the stream failed but not why.

## What can go wrong

A correct gateway has to handle a specific list of cases, and each one matters to something built later. The tests check each of these:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| missing key rejected | unit | request 1: 401, the error shape with all four keys, `X-Request-Id` set, the engine never called | the v0 contract's `v0.auth.401`; API keys and scopes keeps the shape |
| wrong keys rejected | boundary | a prefix of the key, the key plus a byte, another case, `Basic`, no scheme: all 401; the right key: 200 | real keys replace the static key, the rules stay |
| an empty configured key rejects everything | boundary | `APIKey: ""` rejects no header, `Bearer `, and `Bearer x` | a gateway started without `TL_API_KEY` fails closed |
| the request is forwarded without the key | unit | method, path, query, body, end-to-end headers reach the engine; `Authorization` does not; a trailing slash on `Upstream` changes nothing | the engine tier stays key-free in the deployment |
| streams without buffering | unit | the first event reaches the client while the engine holds the second | tokens stream through the streaming proxy and the end-to-end tracer |
| a v0 stream passes through byte for byte | conformance | a recorded v0 stream arrives byte for byte, `text/event-stream`, 200 | `v0.stream.framing` through the gateway |
| v0 responses pass through unchanged | conformance | a 200 completion and a 400 error keep status, headers, and body | `v0.err.400` through the gateway |
| upstream down is a 503 | boundary | request 3: 503 `server_error`/`no_capacity` in the error shape | routing retries elsewhere on exactly this |
| a mid-stream failure sends an error event | fault | the engine's socket closes after one event: that event, then one `data: {"error": ...}` event | the streaming proxy's chaos test builds on it |
| request ids | unit | absent, kept (`req-42`, 128 bytes), replaced (a space, 129 bytes); the client and the engine see the same id | observability joins logs on it |
| the gateway span is the caller's child | unit | request 2: same trace and flags, a new span id, `tracestate` kept, `OnSpan` gets the span | the Jaeger trace asserts this parentage |
| starting a trace | boundary | absent, garbage, all-zero trace id, upper-case hex: a new sampled trace, a root span | the engine never receives an invalid header |
| parsing `traceparent` | unit | the W3C rules above, one row each | the engine's parser applies the same rules |
| a client disconnect cancels upstream | fault | a client that cancels mid-stream cancels the engine's request within 3 s | the streaming proxy tightens this to 100 ms |

Most bugs in a first gateway are one of a dozen mistakes. Each has a recognizable symptom, and the solution avoids each with a specific line:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| comparing only as many bytes as the client sent (or the key's prefix) | `Bearer tl_test` opens the door | `ConstantTimeCompare` over the whole strings returns 0 for different lengths |
| no empty-key guard, with `TrimPrefix(header, "Bearer ") == key` | a gateway started without `TL_API_KEY` accepts requests with no header, and a bare key without `Bearer` | `authorized` returns false for an empty key and requires the `Bearer ` scheme |
| copying every request header, `Authorization` included | the gateway's key appears in engine logs | `up.Header.Del("Authorization")` after `copyHeaders` |
| `io.Copy(w, resp.Body)` with no flush | the first token waits 1.3 s, then 26 arrive at once | `copyFlush` flushes after every write |
| building the upstream request on `context.Background()` | a client that hangs up leaves the engine generating | `http.NewRequestWithContext(r.Context(), ...)` |
| writing the body without `WriteHeader(resp.StatusCode)` first | every engine error reaches the client as 200 | `sw.WriteHeader(resp.StatusCode)` before `copyFlush` |
| `http.Error(w, err.Error(), 502)` when the engine is down | a plain-text 502 that OpenAI clients cannot parse | `writeError(..., 503, "server_error", "no_capacity", ...)` in the contract's JSON shape |
| returning quietly when the stream breaks after the first byte | the client sees a short, valid-looking answer | a failed `copyFlush` on an SSE response writes and flushes an error event |
| forwarding the caller's `traceparent` unchanged | the engine's span is a sibling of `gateway.proxy`, not its child | the outgoing `traceparent` carries `span.SpanID`, a fresh id |
| setting `X-Request-Id` on the response only | the engine's logs cannot be joined to the client's id | the id is set on the upstream request and on the response |
| `Upstream + path` without trimming a trailing slash | the engine receives `//v1/completions` (a `ServeMux` answers 301) | `NewProxy` trims the slash once |
| accepting an all-zero trace or parent id | an invalid trace reaches the engine and the trace backend drops it | `ParseTraceparent` rejects ids that are all zeros |

## Where this leads

This gateway leans on two earlier pieces. The Go primer supplied goroutines, `context`, `net/http`, and the streaming proxy this section hardens; [Your first endpoint](../../build-an-llm/p10-serving/00-your-first-endpoint.md) supplied the engine behind it, with API v0, its `/healthz`, and its `traceparent` parser.

Four later pieces build on it. [Tracer deploy](../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) builds the gateway image and runs it on kind in front of the engine (NodePort 30080). [One trace: gateway to engine in Jaeger](../../systems/04-observability/00-one-trace.md) exports `Config.OnSpan`'s span from the entry point, so the trace shows `gateway.proxy` above the engine span. [The server skeleton](01-server-skeleton.md) wraps `NewProxy` in the middleware chain (`requestid -> otel -> recover -> authn -> ... -> proxy`). And [The SSE streaming proxy](04-sse-streaming-proxy.md) takes `go/gateway/proxy/` over and adds `StreamProxy` with a `StreamObserver` for time to first token and usage.

### Further reading

- Production equivalent of `NewProxy`: [`httputil.ReverseProxy`](https://pkg.go.dev/net/http/httputil#ReverseProxy) (free), which adds `Rewrite` hooks, `X-Forwarded-*`, buffer pools, and `FlushInterval: -1` for streaming responses.
- Production equivalents of the static key: [Envoy AI Gateway](https://aigateway.envoyproxy.io/) (free) and [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) (free), with per-tenant keys, provider credentials injected upstream, and budgets.
- Production equivalent of the hand-made `traceparent`: OpenTelemetry [`otelhttp`](https://pkg.go.dev/go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp) (free), which provides spans and propagation as middleware and transport wrappers.
- Production equivalents of SSE pass-through: Envoy's HTTP/2 streaming and [llm-d](https://github.com/llm-d/llm-d)'s inference gateway (free), with backpressure across protocols and KV-aware routing behind one front door.
- [RFC 9110 HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) sections 7.6 and 15.6 (free), [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free), and the [HTML Living Standard on server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free).

## Key terms

- **reverse proxy**: a server that receives a client's request, makes a new request to an upstream server, and relays the response.
- **downstream / upstream**: the client side and the server side of a proxy, respectively.
- **hop-by-hop header**: a header that describes one connection only and is not forwarded across a proxy.
- **end-to-end header**: a header meant for the final recipient, which a proxy copies.
- **constant-time comparison**: a comparison whose running time depends only on the lengths of its inputs, not their contents.
- **fail closed**: rejecting every request when configuration is missing, rather than accepting them.
- **server-sent events (SSE)**: a `text/event-stream` response of `data: ...\n\n` events over one long HTTP response.
- **trace / span**: a tree of timed operations across processes for one request, and one node of that tree.
- **`traceparent`**: the W3C header carrying version, trace id, parent span id, and flags.
- **`X-Request-Id`**: a string that names one request in every log line it touches.

## Summary

- The key check comes first, before any byte goes upstream, and it fails closed: no configured key means every request is a 401 in the OpenAI error shape.
- Keys are compared in constant time. An early-exit comparison leaks how many leading bytes matched, which turns guessing a key from $256^L$ work into $256 \cdot L$.
- A gateway forwards bytes, not events: every chunk the engine sends is written and flushed before the next read, so the stream reaches the client byte for byte and token by token.
- Before the first byte, an engine failure is a 503 `no_capacity`; after it, the status is already sent, so the failure becomes an SSE error event. A client that leaves cancels the engine's request.
- The gateway continues the caller's W3C trace with its own span id, so the engine's span is a child of `gateway.proxy`, and it carries one `X-Request-Id` to the engine and back.

## Review questions

1. Why does `NewProxy` with `APIKey: ""` reject `Authorization: Bearer ` (a scheme with nothing after it)? Which line makes it so?

   <details><summary>Answer</summary>

   `authorized` returns false on `key == ""` before looking at the header. Without that guard, the presented value after trimming would be `""`, and `ConstantTimeCompare("", "")` returns 1: an unconfigured gateway would let anyone in who sends an empty bearer token, or, with a `TrimPrefix` check, no header at all.

   </details>

2. The gateway forwards the caller's `traceparent` unchanged. What does Jaeger show, and why is that wrong?

   <details><summary>Answer</summary>

   The engine's span would name the caller's span as its parent, the same parent as `gateway.proxy`: the gateway and the engine appear as siblings, and the gateway's time cannot be separated from the engine's. Rewriting the parent id to the gateway's own span id makes the engine a child of `gateway.proxy`, which is what actually happened.

   </details>

3. With `io.Copy(w, resp.Body)` and no flush, why does the first token arrive after about 1.3 s instead of 50 ms?

   <details><summary>Answer</summary>

   The response writer buffers 4096 bytes before sending anything. At about 160 bytes per event and 20 events per second, the buffer fills after $\lceil 4096/160 \rceil = 26$ events, which takes 1.3 s; then 26 tokens arrive at once. Flushing after every write sends each event as soon as it is read.

   </details>

4. The engine's socket breaks after one event. Why can't the gateway answer 503, and what does it send instead?

   <details><summary>Answer</summary>

   The status line and the first event have already been sent: HTTP has no way to change a status once the body has started. The gateway writes one `data: {"error": {...}}` event and flushes it, so the client sees an explicit failure rather than a stream that just ends and looks like a short, valid answer.

   </details>

5. Why is the upstream request built with `r.Context()`?

   <details><summary>Answer</summary>

   The incoming request's context is cancelled when the client disconnects. Making the upstream request on it means `client.Do` and the body reads are cancelled too, the connection to the engine closes, and the engine stops generating tokens nobody will read. With `context.Background()` the engine would keep working for a departed client.

   </details>
