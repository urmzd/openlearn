# Server skeleton, composition root, go/config, graceful shutdown

## Overview

| | |
|---|---|
| **Module** | `gw.01` · build · Go · Pass 7 · 4 to 5 h |
| **Solution** | [`go/gateway/server/chain.go`](../../../../course/solutions/go/gateway/server/chain.go): the chain in its contractual order, the `Exchange`, `WriteError`, `ParseRequest` · [`go/gateway/server/server.go`](../../../../course/solutions/go/gateway/server/server.go): `New`, `Deps`, `Config`, the health port, `Shutdown`, `Run` · [`go/config/config.go`](../../../../course/solutions/go/config/config.go): `runtime.toml` with `TL_<SECTION>__<KEY>` overrides · the composition root [`entry/go/cmd/gateway/compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go) |
| **Contract** | [`course/contracts/config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) (the file and the override rule) · the `gateway` role's `--config` form in [`course/contracts/spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) · health and error rules of [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) |
| **Needs** | reading: [`lang.06` Go](../../programming-languages/02-language-and-tool-primers/06-go.md), [`gw.00` the tracer gateway](00-streaming-proxy.md) |
| **Used by** | `gw.02` authn stage · `gw.03` ratelimit stage · `gw.04` proxy stage and its observer · `gw.05` route stage · `gw.06` cache stage · `gw.07` meter stage and admin API: every one is a `server.Middleware` that reads the `Exchange` and answers errors with `WriteError` · later: `gw.08`; together they make up the full gateway |
| **Optional depth** | Mat Ryer, *How I write HTTP services in Go after 13 years* (free, 2024); RFC 9110 section 15 (status codes); the Kubernetes docs on [pod termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination) (free) |

## Key Takeaways

- The chain order is a contract: `requestid -> otel -> recover -> authn -> policy -> ratelimit -> cache -> route -> proxy -> meter`. A request is named, traced, and protected from panics before anything can reject it, authenticated before anything reads its key, and limited before the cache can answer it.
- The server imports **none** of its stages. Each stage is plain middleware in `Deps`; the stage packages import the server for `Middleware`, the `Exchange`, and `WriteError`, and `main` wires them. That is a composition root, and it is why `gw.01` can be built before any stage exists.
- One request-scoped `Exchange` carries what stages tell each other: the request id, the parsed body (read once, put back for the proxy), the status, the usage the proxy saw, the worker the router chose.
- Shutdown is a sequence: `/readyz` turns 503 first, the listener closes, streams in flight run to completion until a deadline, then their contexts are cancelled.
- Configuration has one precedence, environment over file over defaults, one separator (`__`), and no silent typos: unknown keys and bad overrides stop the process at start-up.

## How to read this chapter

Sections 1 to 3 derive the chain order, the shared request record, the drain sequence, and the configuration precedence from first principles and trace one request, one configuration, and one drain by hand. Section 4 fixes the interface. Section 5 walks through the solution function by function and explains why each choice is the right one; try to predict each step before reading it. `go/config` uses `github.com/BurntSushi/toml` (allowed in [`contracts/allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml)); the gateway's composition root, [`entry/go/cmd/gateway/compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go), is the `gateway --config <runtime.toml>` form of the role.

---

## 1. Why now

The tracer gateway of `gw.00` is one handler with everything inside it: the key check, the trace, the forward. Pass 7 adds real keys (`gw.02`), rate limits (`gw.03`), a streaming observer (`gw.04`), routing across engines (`gw.05`), a response cache (`gw.06`), and a usage ledger (`gw.07`). Bolted into one function, each of them would have to know about the others, the order in which they run would be an accident of whoever edited last, and a panic in any of them would take the process down mid-stream. When `dep.03` rolls the gateway on Kubernetes, the pod receives SIGTERM and has 30 s; a gateway that exits at once cuts every user's stream, and one that keeps answering `/readyz` with 200 keeps receiving new requests while it shuts down. This module builds the skeleton the rest plug into: an ordered chain, a shared request record, a health port that tells the truth, a drain that finishes what it started, and one configuration loader for every Go service.

## 2. Principles

### 2.1 The chain and its order

A **middleware** is a function from handler to handler, `func(next http.Handler) http.Handler`: it can act before calling `next`, after it returns, or instead of calling it (rejecting the request).

| Symbol | Meaning | Type |
|---|---|---|
| $h$ | the handler at the end of the chain (the proxy) | `http.Handler` |
| $m_i$ | the $i$-th middleware, outermost first | `server.Middleware` |
| $k$ | the number of middlewares | integer |

Composing middlewares $m_1, \dots, m_k$ around $h$ gives $m_1(m_2(\cdots m_k(h)))$: $m_1$ sees the request first and the response last. This is the decorator pattern (`ai-platform-engineering/06`). The gateway's order, outermost first:

| Stage | Owner | Why here |
|---|---|---|
| `requestid` | `gw.01` | everything after it, including error answers, can quote the id |
| `otel` | `gw.01` (spans from `obs.01`) | the server span covers every later stage, rejections included |
| `recover` | `gw.01` | inside `otel`, so a panic still ends the span with status 500 |
| `authn` | `gw.02` | nothing below may act for an unknown caller |
| `policy` | `gw.08` | a blocked prompt never spends rate limit or reaches the cache |
| `ratelimit` | `gw.03` | before the cache, so a cache hit still counts as a request |
| `cache` | `gw.06` | a hit skips routing and the engine entirely |
| `route` | `gw.05` | picks the worker the proxy sends to |
| `proxy` | `gw.00`, then `gw.04` | the handler at the end: forward and stream |
| `meter` | `gw.07` | wraps the proxy, so it is the first to see the finished exchange |

`proxy -> meter` orders when stages finish their work: the meter wraps the proxy and records after it returns. Every stage is optional except the proxy: a `nil` slot is skipped, and a missing proxy answers 503 `no_capacity`.

**Panics.** Go's `net/http` recovers a handler panic by closing the connection, which a client sees as a network error. The `recover` stage instead answers 500 with `{"error": {..., "type": "server_error", "code": "internal_error"}}` and logs the stack with the request id, so the process keeps serving. If the response already started (a stream is half sent), the status line is gone: the stage writes nothing more. A handler that panics with `http.ErrAbortHandler` is asking for the connection to be dropped, so that one is re-panicked.

### 2.2 Request id and internal headers

The first stage keeps a caller's `X-Request-Id` when it is printable ASCII (0x21 to 0x7E) of at most 128 bytes (the `gw.00` rule), else makes one, and sets it in three places: the response, the request (so the proxy forwards it), and the `Exchange`. It also deletes every `X-TL-*` header the client sent (except the public `X-TL-API-Version`): `X-TL-Priority` and `X-TL-KV-Handle` are set by the gateway toward engines (`gw.02`, `gw.05`); a client that could set them would jump the engine's queue or resume someone else's KV cache (the contract's `priority.internal` rule).

### 2.3 The Exchange

Stages need to tell each other things without importing each other: the limiter needs the usage the proxy saw to settle; the meter needs the status, the worker, and the usage; the trace needs whatever attributes stages recorded. The `Exchange` is that record, created by `requestid` and reached from any stage with `server.ExchangeFrom(r.Context())`. Its mutable fields are behind a lock, because the proxy writes the usage from the goroutine that reads the stream.

The body is read **once**. `ex.Request(r)` reads at most 4 MiB (the same cap as every gRPC message), parses the fields stages need (`model`, `stream`, the token limit with `max_completion_tokens` winning over `max_tokens`, `temperature`, `seed`), caches the result, and puts an identical reader back on `r.Body` so the proxy forwards the bytes unchanged. Unknown fields are ignored (the contract says so); a body that is not one JSON object is an error the calling stage answers with 400.

### 2.4 Health, readiness, and draining

The health port (`:9464` in Kubernetes, `{health_port}` locally) answers three paths:

| Path | Means | Answer |
|---|---|---|
| `/healthz` | the process is alive (restart me if not) | always 200 |
| `/readyz` | send me traffic | 503 while draining or while `Deps.Ready` fails (`gw.05`: no routable worker), else 200 |
| `/metrics` | Prometheus exposition | `Deps.Metrics` (`obs.01`), else 404 |

A Kubernetes rollout sends SIGTERM, removes the pod from its Service only after the readiness probe fails, and kills the pod after `terminationGracePeriodSeconds`. Hence the order of `Shutdown(ctx)`:

1. Mark draining: `/readyz` answers 503 from now on, so the load balancer stops sending new work.
2. `http.Server.Shutdown(ctx)`: close the listener (new connections are refused), close idle connections, and wait for active requests, SSE streams included, to finish.
3. If `ctx` expires first, cancel the base context every request context descends from (so each handler and its upstream call to the engine stops) and close the connections; return `context.DeadlineExceeded`.
4. Close the health server last, so probes see 503 for the whole drain.

`Run(ctx, api, health)` serves until `ctx` is done, then shuts down with `drain_deadline_s`. The entry point builds that `ctx` with `signal.NotifyContext(ctx, syscall.SIGTERM, os.Interrupt)`.

### 2.5 Configuration

Every Go service reads one `runtime.toml` (`config/runtime.schema.json`). Precedence, highest first:

| Source | Example | Wins over |
|---|---|---|
| environment `TL_<SECTION>__<KEY>` | `TL_GATEWAY__CACHE_ENTRIES=50` | file, defaults |
| the file | `[gateway] cache_entries = 100` | defaults |
| schema defaults | `cache_entries` = 4096 | nothing |

Keys contain single underscores (`kv_blocks`, `cache_entries`), so only the **double** underscore separates section from key: `TL_ENGINE__KV_BLOCKS` is `engine.kv_blocks`. A variable without `__` (`TL_API_KEY`, `TL_GATEWAY_PEPPER`) is not an override at all; those are secrets the services read by name. An override is parsed as its key's type (integer, number, boolean, string); tables and arrays (`engine.speculative`, `gateway.routes`) are not overridable. Unknown keys are errors everywhere except under `[ext]`, which is free-form: a typo such as `cache_entires = 5` would otherwise leave the default in force without a word. Validation (enumerations such as `route_policy`, ranges such as `key_cache_ttl_s` in 0 to 5) runs **after** the overrides, because an environment variable can be as wrong as a file.

## 3. Worked example by hand

**The chain.** Put a recording middleware in every `Deps` slot (each logs `enter X`, calls `next`, logs `leave X`), a fake tracer that logs `enter otel` when it starts a span, and a proxy that logs `proxy`. One request produces:

```
enter otel
enter authn, enter policy, enter ratelimit, enter cache, enter route
enter meter
proxy
leave meter
leave route, leave cache, leave ratelimit, leave policy, leave authn
```

`requestid` and `recover` are the server's own and log nothing; `meter` sits directly around the proxy.

**The configuration.** Defaults say `cache_entries = 4096`; the file says

```toml
[gateway]
listen = ":8081"
cache_entries = 100
route_policy = "least_outstanding"
```

and the environment holds `TL_GATEWAY__CACHE_ENTRIES=50`, `TL_API_KEY=tl_x_y`, `HOME=/home/me`. Loading gives `cache_entries = 50` (environment beats file), `listen = ":8081"` (file beats default), `health_listen = ":9464"` and `key_cache_ttl_s = 5` (defaults), and ignores `TL_API_KEY` (no `__`) and `HOME`.

**The drain.** A client is reading an SSE stream; the gateway receives SIGTERM at $t = 0$ with `drain_deadline_s = 30`:

| $t$ | Event | `/readyz` | New connection | The stream |
|---|---|---|---|---|
| 0 s | `Shutdown` starts, draining marked | 503 | refused | running |
| 4 s | the engine sends `[DONE]` | 503 | refused | complete |
| 4 s | no request left: `Shutdown` returns nil | closed | refused | done |

Had the stream still been running at 30 s, its context would have been cancelled (the engine stops generating), its connection closed, and `Shutdown` would have returned `context.DeadlineExceeded`.

## 4. The interface

```go
package server // import "tinyllm/gateway/server"

type Middleware func(next http.Handler) http.Handler
type Clock interface{ Now() time.Time }     // a fake clock satisfies it in tests
type Tracer interface { Start(ctx context.Context, name string) (context.Context, Span) }
type Span interface { SetAttribute(key string, value any); End() }

var Order []Stage // requestid otel recover authn policy ratelimit cache route proxy meter

type Deps struct {
	Keys, Policy, Limiter, Cache, Router Middleware // authn, policy, ratelimit, cache, route
	Proxy  http.Handler                             // required
	Ledger Middleware                               // meter, wraps Proxy
	Clock  Clock; Tracer Tracer
	Ready   func(ctx context.Context) error         // readiness beyond "not draining"
	Metrics http.Handler                            // GET /metrics on the health port
}
type Config struct { Listen, HealthListen string; DrainDeadline time.Duration; MaxBodyBytes int64 }
func FromRuntime(g config.Gateway) Config

func New(cfg Config, d Deps) *Server
func (s *Server) Handler() http.Handler        // the API chain
func (s *Server) HealthHandler() http.Handler  // /healthz, /readyz, /metrics
func (s *Server) Serve(api, health net.Listener) error
func (s *Server) Shutdown(ctx context.Context) error
func (s *Server) Run(ctx context.Context, api, health net.Listener) error
func (s *Server) Draining() bool

type Usage struct { PromptTokens, CompletionTokens, TotalTokens int }
type Request struct { Model string; Stream bool; MaxTokens int; Temperature *float64; Seed *int64; Body []byte }
func ParseRequest(body []byte) (Request, error)
func ExchangeFrom(ctx context.Context) *Exchange
func (e *Exchange) Request(r *http.Request) (*Request, error)
func (e *Exchange) SetUsage(Usage); func (e *Exchange) MarkFirstByte(); func (e *Exchange) SetWorker(string)
func (e *Exchange) SetAttr(key string, v any); func (e *Exchange) Snapshot() ExchangeState
func WriteError(w http.ResponseWriter, status int, typ, code, param, msg string) // "" param or code is null

package config // import "tinyllm/config"
func Default() Config
func Parse(data []byte, env []string) (Config, error)
func Load(path string, env []string) (Config, error)
func ApplyEnv(c *Config, env []string) error
func EnvName(section, key string) string
func (c Config) Validate() error
```

`New` returns a `*Server` holding two `http.Server`s (the API and the health port) rather than the single `*http.Server` of the catalog sketch, because readiness and the drain belong to one object. `Deps` holds middleware rather than the domain interfaces (`Keys`, `Limiter`, ...) of the sketch: the stage packages import this one, so this one cannot name their types.

**The composition root** ([`entry/go/cmd/gateway/compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go)) loads `config.Load(path, os.Environ())`, builds each stage (`auth.Middleware`, `limit.Middleware`, `cache.Middleware`, `route.Middleware` and `route.Proxy`, `ledger.Meter`), passes them in `Deps`, and calls `Run` with a context cancelled on SIGTERM. Before `gw.04`, `Deps.Proxy` can be the `gw.00` handler.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the chain order by hand | unit | section 3: the enter and leave trace, and `server.Order` | every later stage relies on what ran before it |
| nil stages are skipped | boundary | only a proxy: it serves; no proxy: 503 `no_capacity` | stages land one module at a time |
| the request id stage | unit | kept, generated, replaced (129 bytes, a space); the response, the forwarded header, and the Exchange agree | `obs.02` joins logs on it |
| internal headers stripped | unit | client `X-TL-*` headers never reach the proxy | `priority.internal`; `gw.05`'s KV handle |
| a panic becomes a 500 | fault | a panicking stage: 500 `internal_error` with `X-Request-Id`, the span ends with 500, the next request is fine | one bug costs one request |
| a panic after the first byte | fault | a panic after the first byte appends nothing | streams are never corrupted |
| span attributes | unit | span `POST /v1/chat/completions` with method, route, status, and a stage's `tl.ratelimit.decision` | `obs.01` exports this span |
| the error shape | unit | exact bytes, with `param` and `code`, and with both null | every stage's errors parse in OpenAI clients |
| parsing the request through the Exchange | unit | one read, body forwarded unchanged, `max_completion_tokens` wins, cached parse, non-objects and oversized bodies rejected | auth, limits, cache, and routing read the same fields |
| readiness | unit | `/healthz` 200; `/readyz` follows `Deps.Ready`; `/metrics` | scripts and Kubernetes wait on it |
| shutdown flips `/readyz`, then drains | fault | section 3: 503 during the drain, new connections refused, the stream finishes, `Shutdown` returns nil | `dep.03` rolls the gateway without cutting streams |
| shutdown cuts streams at the deadline | fault | at the deadline the handler's context is cancelled and `DeadlineExceeded` returned | a stuck stream cannot block a rollout |
| `Run` shuts down on cancel | unit | `Run` returns nil after its context is cancelled | the entry point's SIGTERM path |
| the configuration hand example | unit | section 3 precedence, routes with an inline cascade, inline tables, `[ext]` | a generated `runtime.toml` for the full gateway |
| configuration defaults | unit | an empty file yields the schema defaults | templates can stay short |
| the override separator | unit | `__` only; `TL_API_KEY` and lower-case names ignored; `=` inside a value | secrets are never misread as overrides |
| override types | unit | numbers parse; a non-integer fails naming the variable | a typo stops start-up |
| override errors | boundary | unknown key or section, a table, an array: errors naming the variable | the same |
| unknown keys rejected | boundary | unknown keys in the file, `[ext]` free-form | the same |
| validation after overrides | boundary | `route_policy = "random"` from the file or the environment, `key_cache_ttl_s = 9` | enumerations hold after overrides |
| `Load` and `FromRuntime` | unit | `Load` reads a file, `FromRuntime` turns `drain_deadline_s` into a duration | the entry point |

## 5. The solution, step by step

### 5.1 From first principles

1. **Stages must not know each other, so the server must not know them either.** If the server imported the auth or cache packages, every stage would depend on every other through it. Inverting the dependency (stages import `server` for `Middleware`, `Exchange`, `WriteError`; `main` wires them into `Deps`) makes the skeleton buildable before any stage exists, and makes each stage replaceable.
2. **Order is a contract, so it lives in one place.** `Handler` composes the slots innermost first, in a fixed sequence, and a `nil` slot is skipped. Nothing else decides the order.
3. **Stages communicate through one request-scoped record.** Context values are the standard carrier; one `*Exchange` per request, created by the first stage, behind a mutex because the proxy writes from the stream goroutine.
4. **The body is a stream that can be read once.** The first stage that needs it reads it (with a cap), parses it, caches the result, and puts identical bytes back so the proxy forwards them unchanged.
5. **Failure handling must sit where it can still answer.** `recover` goes inside `otel` (so the span sees the 500) and outside every stage (so any stage's panic is caught). It can only write a 500 if nothing was written; the `statusWriter` records whether something was.
6. **Shutdown is ordered by what the load balancer sees.** Readiness flips first; then the listener closes; then in-flight requests drain; at the deadline a base context that every request descends from is cancelled. The health server stays up to the end so probes keep seeing 503.
7. **Configuration must fail loudly.** Decode over defaults, reject undecoded keys (except `[ext]`), apply environment overrides by reflection on the same struct tags, then validate. One precedence, checked in one order.

### 5.2 The code, annotated

**Composing the chain.** Built from the inside out: the proxy (or a 503 handler), then meter, route, cache, ratelimit, policy, authn, and finally the server's own three stages.

```go
func (s *Server) Handler() http.Handler {
	var h http.Handler = s.deps.Proxy
	if h == nil { // the only required slot: without it, an honest 503
		h = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			WriteError(w, http.StatusServiceUnavailable, "server_error", "no_capacity", "", "no upstream is configured")
		})
	}
	// Innermost first: meter wraps the proxy, then route, cache, ratelimit.
	for _, m := range []Middleware{s.deps.Ledger, s.deps.Router, s.deps.Cache, s.deps.Limiter} {
		if m != nil { // stages land one module at a time
			h = m(h)
		}
	}
	if s.deps.Policy != nil { // gw.08: after authn, before ratelimit
		h = s.deps.Policy(h)
	}
	if s.deps.Keys != nil {
		h = s.deps.Keys(h)
	}
	h = s.recoverer(h) // inside otel, so a panic's 500 is on the span
	h = s.otel(h)
	return s.requestID(h) // outermost: everything after it can quote the id
}
```

Reading the wrapping order backwards gives the request order of section 2.1. The `gw.01` version of this file wrapped all six slots in one loop; `gw.08` split out `Policy` so its position is explicit (the original is kept at [`history/gw.01/.../chain.go`](../../../../course/solutions/history/gw.01/go/gateway/server/chain.go)).

**The first stage.** Strip internal headers, pick the id, set it on both sides, create the `Exchange`, and wrap the writer so the status is recorded.

```go
func (s *Server) requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for k := range r.Header {
			// X-TL-Priority and X-TL-KV-Handle are set by the gateway toward
			// engines; a client must not set them. API version selection is public.
			if strings.HasPrefix(strings.ToLower(k), "x-tl-") && !strings.EqualFold(k, "X-TL-API-Version") {
				r.Header.Del(k)
			}
		}
		id := r.Header.Get("X-Request-Id")
		if !validRequestID(id) { // printable ASCII, 1 to 128 bytes: the gw.00 rule
			id = newID()
		}
		r.Header.Set("X-Request-Id", id) // forwarded upstream by the proxy
		w.Header().Set("X-Request-Id", id)
		ex := NewExchange(id, s.clock, s.cfg.MaxBodyBytes)
		next.ServeHTTP(&statusWriter{ResponseWriter: w, ex: ex}, r.WithContext(WithExchange(r.Context(), ex)))
	})
}
```

`statusWriter` sets the `Exchange`'s status on the first `WriteHeader`, or 200 on a first `Write` without one (which is what `net/http` sends), and exposes `Unwrap` so `http.NewResponseController` can still flush through it.

**Recover.** The `Exchange`'s status tells it whether the response has started.

```go
func (s *Server) recoverer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			v := recover()
			if v == nil {
				return
			}
			if v == http.ErrAbortHandler {
				panic(v) // the handler asked for the connection to be dropped: honour it
			}
			ex := ExchangeFrom(r.Context())
			log.Printf("gateway: panic in request %s: %v\n%s", ex.RequestID, v, debug.Stack())
			if ex.Snapshot().Status == 0 { // nothing sent yet: we can still answer properly
				WriteError(w, http.StatusInternalServerError, "server_error", "internal_error", "", "internal error; quote X-Request-Id "+ex.RequestID)
			}
			// otherwise a stream is half sent: appending JSON would corrupt it
		}()
		next.ServeHTTP(w, r)
	})
}
```

The `otel` stage (skipped when `Deps.Tracer` is nil) starts a span named `"<METHOD> <path>"`, and in a `defer` copies the final status and every attribute stages recorded with `ex.SetAttr` onto it before ending it.

**Reading the body once.**

```go
func (e *Exchange) Request(r *http.Request) (*Request, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.req != nil || e.reqErr != nil { // parsed already: same answer for every stage
		if e.req != nil {
			r.Body = io.NopCloser(bytes.NewReader(e.req.Body)) // a fresh reader each time
		}
		return e.req, e.reqErr
	}
	var body []byte
	if r.Body != nil {
		// Read one byte past the cap, so "exactly the cap" and "too large" differ.
		b, err := io.ReadAll(io.LimitReader(r.Body, e.maxBody+1))
		// ... a read error is cached and returned
		body = b
	}
	if int64(len(body)) > e.maxBody {
		e.reqErr = fmt.Errorf("the request body is larger than %d bytes", e.maxBody)
		return nil, e.reqErr
	}
	r.Body = io.NopCloser(bytes.NewReader(body)) // the proxy forwards these exact bytes
	req, err := ParseRequest(body)
	// ... cache req or err
}
```

`ParseRequest` requires the trimmed body to start with `{`, decodes only the fields stages need (unknown fields are ignored), and resolves the token limit with `max_completion_tokens` taking precedence over `max_tokens`.

`WriteError` encodes `param` and `code` as `*string`, so an empty argument is `null`: all four keys are always present.

**Server and shutdown.** Two `http.Server`s; the API one gets a `BaseContext` the server can cancel.

```go
s.baseCtx, s.cancelBase = context.WithCancel(context.Background())
s.api = &http.Server{
	Addr:              cfg.Listen,
	Handler:           s.Handler(),
	ReadHeaderTimeout: 10 * time.Second,
	// Every request context descends from baseCtx, so cancelling it at the
	// drain deadline reaches every in-flight handler and its upstream call.
	BaseContext: func(net.Listener) context.Context { return s.baseCtx },
}
```

```go
func (s *Server) Shutdown(ctx context.Context) error {
	s.draining.Store(true)     // 1. /readyz answers 503 from this instant
	err := s.api.Shutdown(ctx) // 2. close the listener, wait for active requests (streams too)
	if err != nil {            // 3. the deadline passed with requests still running
		s.cancelBase()   // their contexts end: handlers and engine calls stop
		_ = s.api.Close() // and their connections close
	}
	s.once.Do(s.cancelBase)
	hctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = s.health.Shutdown(hctx) // 4. last, so probes saw 503 for the whole drain
	return err                  // nil, or context.DeadlineExceeded
}
```

`/readyz` checks `draining` first and `Deps.Ready` second; `/healthz` always answers 200, because a draining process is alive and must not be restarted. `Run` serves until its context is done, then calls `Shutdown` with a fresh context bounded by `DrainDeadline`: the parent context is already cancelled, so it cannot be the drain's deadline.

**Configuration.**

```go
func Parse(data []byte, env []string) (Config, error) {
	c := Default()                           // 3rd: schema defaults
	md, err := toml.Decode(string(data), &c) // 2nd: the file, over the defaults
	// ... return a decode error
	var unknown []string
	for _, k := range md.Undecoded() { // keys the struct has no field for
		if len(k) > 0 && k[0] == "ext" {
			continue // [ext] is free-form by contract
		}
		unknown = append(unknown, k.String())
	}
	// ... any unknown key is an error listing all of them
	if err := ApplyEnv(&c, env); err != nil { // 1st: the environment wins
		return Config{}, err
	}
	if err := c.Validate(); err != nil { // after overrides: env can be wrong too
		return Config{}, err
	}
	return c, nil
}
```

```go
for _, kv := range vars { // sorted, so several bad variables give a deterministic error
	name, val, found := strings.Cut(kv, "=") // first "=": values may contain "="
	if !found || !strings.HasPrefix(name, "TL_") {
		continue
	}
	section, key, ok := strings.Cut(strings.TrimPrefix(name, "TL_"), "__")
	if !ok {
		continue // TL_API_KEY and friends: secrets read by name, not overrides
	}
	// ... find the section and key by their toml tags, upper-cased; unknown: an error naming the variable
	if err := setScalar(f, val); err != nil { // parsed as the field's kind; tables and arrays refused
		return fmt.Errorf("%s: %w", name, err)
	}
}
```

Because the lookup uses the same `toml` tags as the decoder, there is no second list of keys to keep in sync. `Validate` checks the enumerations (`route_policy`, `engine.role`, `prefix_cache`) and ranges (`key_cache_ttl_s` 0 to 5, `affinity_load_factor` at least 1, `trace_sample_ratio` 0 to 1) a typed decode cannot.

**The composition root.** [`entry/go/cmd/gateway/compose.go`](../../../../course/solutions/entry/go/cmd/gateway/compose.go) is the `--config` form: `config.Load(path, os.Environ())`, the stages built from the loaded `[gateway]` section, `server.New(server.FromRuntime(g), deps)`, and `Run` under `signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)`.

Full solution: [`chain.go`](../../../../course/solutions/go/gateway/server/chain.go), [`server.go`](../../../../course/solutions/go/gateway/server/server.go), [`config.go`](../../../../course/solutions/go/config/config.go).

### 5.3 Why this solution

- **Cost.** The chain adds a header scan, one allocation for the `Exchange`, and a few lock operations per request. The body is read at most once, with a 4 MiB cap, instead of once per stage.
- **Why middleware slots and not interfaces like `Keys` or `Limiter`?** The server would have to name the stage packages' types, which import the server: an import cycle. Middleware is a type the server owns, so the dependency points one way.
- **Why a shared `Exchange` and not one context value per stage?** Context values are write-once per derived context; the proxy's usage, recorded deep in the chain, must be visible to the limiter and the meter that wrapped it. A pointer to one mutable record, under a lock, is the simplest way to pass data back up.
- **Why not let `net/http` recover panics?** It closes the connection, so a client cannot tell a crash from a network fault and gets no request id to report. Answering 500 in the contract's shape is retryable and debuggable.
- **Why reflection for environment overrides?** A hand-written switch per key would drift from the struct. Reusing the `toml` tags makes every scalar key overridable automatically and every unknown variable an error.
- **Tradeoff accepted:** the drain deadline is shared by all streams; a long stream started just before SIGTERM is cut at the deadline. `drain_deadline_s` must be shorter than the pod's `terminationGracePeriodSeconds`.

### 5.4 Explain it back

1. Why is `recover` inside `otel` and not outside it?

<details><summary>Answer</summary>

`otel` ends its span in a `defer` with the status from the `Exchange`. If `recover` were outside, the panic would unwind through `otel` before any 500 was written: the span would end with status 0 and the trace would not show the crash. Inside, `recover` writes the 500 first, then `otel`'s `defer` records it.

</details>

2. A stage reads `r.Body` directly instead of calling `ex.Request(r)`. What does the engine receive?

<details><summary>Answer</summary>

An empty body: the stream has been consumed and nothing put it back. `ex.Request` exists so that the body is read once, cached, and replaced by an identical reader, which the proxy then forwards unchanged.

</details>

3. During a rollout, why must `/readyz` turn 503 before the listener closes, and why is the health server shut down last?

<details><summary>Answer</summary>

Kubernetes removes a pod from its Service only after the readiness probe fails. If the listener closed first, the load balancer would keep sending new connections that are refused. Keeping the health server up until the end means every probe during the drain sees 503, so no new traffic is routed while in-flight streams finish.

</details>

4. Why is `TL_API_KEY` ignored by `ApplyEnv`, while `TL_GATEWAY_CACHE_ENTRIES=5` (single underscore) is also ignored, and is that second case a problem?

<details><summary>Answer</summary>

Only a double underscore separates section from key, because keys themselves contain underscores. Neither variable has `__`, so neither is an override: `TL_API_KEY` is a secret read by name, which is intended. The single-underscore typo is silently ignored too; the rule trades that risk for never misreading a secret as configuration. A wrong name with `__` (`TL_GATEWAY__CACHE_ENTIRES`) is an error.

</details>

5. Why does `Run` create a new context for `Shutdown` instead of passing its own `ctx`?

<details><summary>Answer</summary>

`Run`'s `ctx` is the one that was just cancelled by SIGTERM. Passing it to `Shutdown` would make the drain's deadline already expired, so every stream would be cut at once. A fresh context bounded by `DrainDeadline` gives in-flight requests their grace period.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. stages composed in the wrong order (the limiter around authn, the meter around the router) | limits keyed by nobody; usage recorded before routing | `Handler` wraps the slots in one fixed sequence, innermost first |
| 2. `recover` outside `otel` | the span of a crashed request ends with status 0, not 500 | `h = s.recoverer(h)` comes before `h = s.otel(h)` |
| 3. a recover that only logs | the client sees a dropped connection instead of a 500 it can retry | `recoverer` writes a 500 `internal_error` when the status is still 0 |
| 4. forwarding client `X-TL-*` headers | any client can set its own engine priority | `requestID` deletes every `X-TL-*` header except `X-TL-API-Version` |
| 5. setting `X-Request-Id` only on the response | engine logs cannot be joined to the client's id | the id is set on the request and the response |
| 6. writing after the first byte, or `"param": ""` | a JSON error spliced into an SSE stream; clients that test `param == null` misread errors | `recoverer` writes only when the status is 0; `WriteError` encodes empty `param` and `code` as `null` |
| 7. ignoring unknown keys | `cache_entires = 5` silently keeps 4096 | `Parse` turns every undecoded key outside `[ext]` into an error |
| 8. splitting on `_`, ignoring bad overrides, letting the file beat the environment, validating before overriding | `TL_ENGINE__KV_BLOCKS` never applies; a Kubernetes env override has no effect; `TL_GATEWAY__ROUTE_POLICY=random` starts | `ApplyEnv` cuts on `__`, errors on unknown or unparsable overrides, runs after the decode, and `Validate` runs last |
| 9. `/readyz` that ignores draining, or flips after the drain | new requests keep arriving during a rollout and get refused connections | `Shutdown` sets `draining` before closing the listener; `/readyz` checks it first |
| 10. no cancellation at the deadline | a stuck stream holds the pod until SIGKILL, and the engine keeps generating | the API server's `BaseContext` is cancelled and its connections closed when the drain times out |
| 11. reading the body without putting it back, or `max_tokens` over `max_completion_tokens` | the engine receives an empty body; limits reserve the wrong count | `Exchange.Request` replaces `r.Body`; `ParseRequest` checks `max_completion_tokens` first |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | `context`, `net/http`, goroutines, and signal handling |
| Back | `gw.00` | the request id rule and the tracer proxy, usable in the `Proxy` slot until `gw.04` |
| Forward | `gw.02` | `auth.Middleware` is the `Keys` slot; it parses the model with `ex.Request` and answers 401 and 403 with `WriteError` |
| Forward | `gw.03` | `limit.Middleware` is the `Limiter` slot; it settles with the usage the proxy put on the `Exchange` |
| Forward | `gw.04` | `proxy.Handler` is the `Proxy` slot; its observer records the first byte and the usage on the `Exchange` |
| Forward | `gw.05` | `route.Middleware` is the `Router` slot, `route.Proxy` replaces the proxy; `Router.Ready` feeds `/readyz` |
| Forward | `gw.06` | `cache.Middleware` is the `Cache` slot |
| Forward | `gw.07` | `ledger.Meter` is the `Ledger` slot; the admin API answers errors with `WriteError` |
| Forward | `gw.08` | the policy stage is the `Policy` slot, between authn and ratelimit |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the ordered chain | Envoy HTTP filter chain | filters configured per route, async filters, per-filter stats | [Envoy HTTP filters](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/http_filters) (free) |
| `Shutdown` and `/readyz` | Kubernetes `preStop` hooks, Envoy drain listeners | a sleep before draining so endpoints propagate; connection draining with GOAWAY on HTTP/2 | [Kubernetes pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) (free) |
| `go/config` | Viper, koanf | many sources, file watching and reload | [koanf](https://github.com/knadh/koanf) (free) |
| the `Exchange` | OpenTelemetry baggage, Envoy dynamic metadata | request-scoped data across filters and processes | [Envoy dynamic metadata](https://www.envoyproxy.io/docs/envoy/latest/configuration/advanced/well_known_dynamic_metadata) (free) |
