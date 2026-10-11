<!-- ol:module gw.01 -->
# Server skeleton, composition root, go/config, graceful shutdown

## Overview

| | |
|---|---|
| **Module** | `gw.01` · build · Go · Pass 7 · 4 to 5 h |
| **You build** | `go/gateway/server/chain.go`: the chain in its contractual order, the `Exchange`, `WriteError`, `ParseRequest` · `go/gateway/server/server.go`: `New`, `Deps`, `Config`, the health port, `Shutdown`, `Run` · `go/config/config.go`: `runtime.toml` with `TL_<SECTION>__<KEY>` overrides; then your `go/cmd/gateway` composition root (learner territory) |
| **Contract** | [`course/contracts/config/runtime.schema.json`](../../course/contracts/config/runtime.schema.json) (the file and the override rule) · the `gateway` role's `--config` form in [`course/contracts/spec/cli-roles.md`](../../course/contracts/spec/cli-roles.md) · health and error rules of [`course/contracts/openapi/openai-subset.v1.yaml`](../../course/contracts/openapi/openai-subset.v1.yaml) |
| **Tests** | `course/tests/go/gw_01/` (what they check: section 4) |
| **Needs** | reading: [`lang.06` Go](../../software-craftsmanship/12-language-and-tool-primers/06-go.md), [`gw.00` the tracer gateway](00-streaming-proxy.md) |
| **Used by** | `gw.02` authn stage · `gw.03` ratelimit stage · `gw.04` proxy stage and its observer · `gw.05` route stage · `gw.06` cache stage · `gw.07` meter stage and admin API: every one is a `server.Middleware` that reads the `Exchange` and answers errors with `WriteError` · later: `gw.08` |
| **Milestone** | MS-gateway |
| **Optional depth** | Mat Ryer, *How I write HTTP services in Go after 13 years* (free, 2024); RFC 9110 section 15 (status codes); the Kubernetes docs on [pod termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination) (free) |

## Key Takeaways

- The chain order is a contract: `requestid -> otel -> recover -> authn -> policy -> ratelimit -> cache -> route -> proxy -> meter`. A request is named, traced, and protected from panics before anything can reject it, authenticated before anything reads its key, and limited before the cache can answer it (`TestMiddlewareOrderHand`).
- The server imports **none** of its stages. Each stage is plain middleware in `Deps`; the stage packages import the server for `Middleware`, the `Exchange`, and `WriteError`, and your `main` wires them. That is a composition root, and it is why `gw.01` can be built before any stage exists (`TestNilStagesAreSkipped`).
- One request-scoped `Exchange` carries what stages tell each other: the request id, the parsed body (read once, put back for the proxy), the status, the usage the proxy saw, the worker the router chose (`TestExchangeRequestParsing`).
- Shutdown is a sequence: `/readyz` turns 503 first, the listener closes, streams in flight run to completion until a deadline, then their contexts are cancelled (`TestShutdownFlipsReadyzThenDrains`, `TestShutdownCutsStreamsAtDeadline`).
- Configuration has one precedence, environment over file over defaults, one separator (`__`), and no silent typos: unknown keys and bad overrides stop the process at start-up (`TestConfigHandExample`, `TestConfigEnvErrors`).

## How to work this chapter

```bash
ol start gw.01          # writes chain.go, server.go, config.go with stub bodies
ol tests gw.01          # read the test catalog first
ol check gw.01          # exit code is the verdict
ol diff  gw.01          # after passing: your code against the reference
```

`go/config` uses `github.com/BurntSushi/toml` (allowed in `contracts/allowed-deps.toml`). If your `go/go.mod` was written by `gw.00`, add it once: `go -C go get github.com/BurntSushi/toml@v1.6.0`. Then grow your `go/cmd/gateway/main.go` into the composition root of section 4 and declare the `--config` form in `system.toml`.

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

The first stage keeps a caller's `X-Request-Id` when it is printable ASCII (0x21 to 0x7E) of at most 128 bytes (the `gw.00` rule), else makes one, and sets it in three places: the response, the request (so the proxy forwards it), and the `Exchange`. It also deletes every `X-TL-*` header the client sent: `X-TL-Priority` and `X-TL-KV-Handle` are set by the gateway toward engines (`gw.02`, `gw.05`); a client that could set them would jump the engine's queue or resume someone else's KV cache (conformance case `priority.internal`).

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

`Run(ctx, api, health)` serves until `ctx` is done, then shuts down with `drain_deadline_s`. Your `main` builds that `ctx` with `signal.NotifyContext(ctx, syscall.SIGTERM, os.Interrupt)`.

### 2.5 Configuration

Every Go service reads one `runtime.toml` (`config/runtime.schema.json`). Precedence, highest first:

| Source | Example | Wins over |
|---|---|---|
| environment `TL_<SECTION>__<KEY>` | `TL_GATEWAY__CACHE_ENTRIES=50` | file, defaults |
| the file | `[gateway] cache_entries = 100` | defaults |
| schema defaults | `cache_entries` = 4096 | nothing |

Keys contain single underscores (`kv_blocks`, `cache_entries`), so only the **double** underscore separates section from key: `TL_ENGINE__KV_BLOCKS` is `engine.kv_blocks`. A variable without `__` (`TL_API_KEY`, `TL_GATEWAY_PEPPER`) is not an override at all; those are secrets the services read by name. An override is parsed as its key's type (integer, number, boolean, string); tables and arrays (`engine.speculative`, `gateway.routes`) are not overridable. Unknown keys are errors everywhere except under `[ext]`, which is yours: a typo such as `cache_entires = 5` would otherwise leave the default in force without a word. Validation (enumerations such as `route_policy`, ranges such as `key_cache_ttl_s` in 0 to 5) runs **after** the overrides, because an environment variable can be as wrong as a file.

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

`requestid` and `recover` are the server's own and log nothing; `meter` sits directly around the proxy. This is `TestMiddlewareOrderHand`.

**The configuration.** Defaults say `cache_entries = 4096`; the file says

```toml
[gateway]
listen = ":8081"
cache_entries = 100
route_policy = "least_outstanding"
```

and the environment holds `TL_GATEWAY__CACHE_ENTRIES=50`, `TL_API_KEY=tl_x_y`, `HOME=/home/me`. Loading gives `cache_entries = 50` (environment beats file), `listen = ":8081"` (file beats default), `health_listen = ":9464"` and `key_cache_ttl_s = 5` (defaults), and ignores `TL_API_KEY` (no `__`) and `HOME`. This is `TestConfigHandExample`.

**The drain.** A client is reading an SSE stream; the gateway receives SIGTERM at $t = 0$ with `drain_deadline_s = 30`:

| $t$ | Event | `/readyz` | New connection | The stream |
|---|---|---|---|---|
| 0 s | `Shutdown` starts, draining marked | 503 | refused | running |
| 4 s | the engine sends `[DONE]` | 503 | refused | complete |
| 4 s | no request left: `Shutdown` returns nil | closed | refused | done |

Had the stream still been running at 30 s, its context would have been cancelled (the engine stops generating), its connection closed, and `Shutdown` would have returned `context.DeadlineExceeded`. These are `TestShutdownFlipsReadyzThenDrains` and `TestShutdownCutsStreamsAtDeadline`.

## 4. The interface

```go
package server // import "tinyllm/gateway/server"

type Middleware func(next http.Handler) http.Handler
type Clock interface{ Now() time.Time }     // course testkit clock.Fake satisfies it
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

`New` returns a `*Server` holding two `http.Server`s (the API and the health port) rather than the single `*http.Server` of the catalog sketch, because readiness and the drain belong to one object (DEVIATIONS B93-02). `Deps` holds middleware rather than the domain interfaces (`Keys`, `Limiter`, ...) of the sketch: the stage packages import this one, so this one cannot name their types (B93-03).

**Your composition root** (`go/cmd/gateway/main.go`, learner territory, run by MS-gateway) loads `config.Load(path, os.Environ())`, builds each stage as it lands (`auth.Middleware`, `limit.Middleware`, `cache.Middleware`, `route.Middleware` and `route.Proxy`, `ledger.Meter`), passes them in `Deps`, opens the two listeners, and calls `Run` with a context cancelled on SIGTERM. Before `gw.04`, `Deps.Proxy` can be your `gw.00` handler.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestMiddlewareOrderHand` | unit | section 3: the enter and leave trace, and `server.Order` | every later stage relies on what ran before it |
| `TestNilStagesAreSkipped` | boundary | only a proxy: it serves; no proxy: 503 `no_capacity` | stages land one module at a time |
| `TestRequestIDStage` | unit | kept, generated, replaced (129 bytes, a space); the response, the forwarded header, and the Exchange agree | `obs.02` joins logs on it |
| `TestStripsInternalHeaders` | unit | client `X-TL-*` headers never reach the proxy | `priority.internal`; `gw.05`'s KV handle |
| `TestRecoverTurnsPanicInto500` | fault | a panicking stage: 500 `internal_error` with `X-Request-Id`, the span ends with 500, the next request is fine | one bug costs one request |
| `TestRecoverAfterFirstByte` | fault | a panic after the first byte appends nothing | streams are never corrupted |
| `TestOTelSpanAttributes` | unit | span `POST /v1/chat/completions` with method, route, status, and a stage's `tl.ratelimit.decision` | `obs.01` exports this span |
| `TestWriteErrorShape` | unit | exact bytes, with `param` and `code`, and with both null | every stage's errors parse in OpenAI clients |
| `TestExchangeRequestParsing` | unit | one read, body forwarded unchanged, `max_completion_tokens` wins, cached parse, non-objects and oversized bodies rejected | auth, limits, cache, and routing read the same fields |
| `TestReadyz` | unit | `/healthz` 200; `/readyz` follows `Deps.Ready`; `/metrics` | the milestone runner and Kubernetes wait on it |
| `TestShutdownFlipsReadyzThenDrains` | fault | section 3: 503 during the drain, new connections refused, the stream finishes, `Shutdown` returns nil | `dep.03` rolls the gateway without cutting streams |
| `TestShutdownCutsStreamsAtDeadline` | fault | at the deadline the handler's context is cancelled and `DeadlineExceeded` returned | a stuck stream cannot block a rollout |
| `TestRunShutsDownOnCancel` | unit | `Run` returns nil after its context is cancelled | your `main`'s SIGTERM path |
| `TestConfigHandExample` | unit | section 3 precedence, routes with an inline cascade, inline tables, `[ext]` | MS-gateway's generated `runtime.toml` |
| `TestConfigDefaults` | unit | an empty file yields the schema defaults | templates can stay short |
| `TestConfigSeparator` | unit | `__` only; `TL_API_KEY` and lower-case names ignored; `=` inside a value | secrets are never misread as overrides |
| `TestConfigEnvTypes` | unit | numbers parse; a non-integer fails naming the variable | a typo stops start-up |
| `TestConfigEnvErrors` | boundary | unknown key or section, a table, an array: errors naming the variable | the same |
| `TestConfigRejectsUnknownKeys` | boundary | unknown keys in the file, `[ext]` free-form | the same |
| `TestConfigValidation` | boundary | `route_policy = "random"` from the file or the environment, `key_cache_ttl_s = 9` | enumerations hold after overrides |
| `TestConfigLoadAndFromRuntime` | unit | `Load` reads a file, `FromRuntime` turns `drain_deadline_s` into a duration | your `main` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. stages composed in the wrong order (the limiter around authn, the meter around the router) | limits keyed by nobody; usage recorded before routing | `TestMiddlewareOrderHand` (mutants `s01`, `s02`) |
| 2. `recover` outside `otel` | the span of a crashed request ends with status 0, not 500 | `TestRecoverTurnsPanicInto500` (mutant `s03`) |
| 3. a recover that only logs | the client sees a dropped connection instead of a 500 it can retry | `TestRecoverTurnsPanicInto500` (mutant `s06`) |
| 4. forwarding client `X-TL-*` headers | any client can set its own engine priority | `TestStripsInternalHeaders` (mutant `s04`) |
| 5. setting `X-Request-Id` only on the response | engine logs cannot be joined to the client's id | `TestRequestIDStage` (mutant `s05`) |
| 6. writing after the first byte, or `"param": ""` | a JSON error spliced into an SSE stream; clients that test `param == null` misread errors | `TestRecoverAfterFirstByte`, `TestWriteErrorShape`, `TestOTelSpanAttributes` (mutants `s19`, `s18`) |
| 7. ignoring unknown keys | `cache_entires = 5` silently keeps 4096 | `TestConfigRejectsUnknownKeys` (mutant `s14`) |
| 8. splitting on `_`, ignoring bad overrides, letting the file beat the environment, validating before overriding | `TL_ENGINE__KV_BLOCKS` never applies; a Kubernetes env override has no effect; `TL_GATEWAY__ROUTE_POLICY=random` starts | `TestConfigSeparator`, `TestConfigHandExample`, `TestConfigEnvErrors`, `TestConfigValidation` (mutants `s12`, `s13`, `s15`, `s16`, `s17`) |
| 9. `/readyz` that ignores draining, or flips after the drain | new requests keep arriving during a rollout and get refused connections | `TestShutdownFlipsReadyzThenDrains` (mutants `s07`, `s08`) |
| 10. no cancellation at the deadline | a stuck stream holds the pod until SIGKILL, and the engine keeps generating | `TestShutdownCutsStreamsAtDeadline` (mutant `s09`) |
| 11. reading the body without putting it back, or `max_tokens` over `max_completion_tokens` | the engine receives an empty body; limits reserve the wrong count | `TestExchangeRequestParsing` (mutants `s10`, `s11`) |

## 6. Where it's used next
| Forward | `gw.08` | Registered module relationship. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.06` | `context`, `net/http`, goroutines, and signal handling |
| Back | `gw.00` | the request id rule and the proxy your root plugs in until `gw.04` |
| Forward | `gw.02` | `auth.Middleware` is the `Keys` slot; it parses the model with `ex.Request` and answers 401 and 403 with `WriteError` |
| Forward | `gw.03` | `limit.Middleware` is the `Limiter` slot; it settles with the usage the proxy put on the `Exchange` |
| Forward | `gw.04` | `proxy.Handler` is the `Proxy` slot; its observer records the first byte and the usage on the `Exchange` |
| Forward | `gw.05` | `route.Middleware` is the `Router` slot, `route.Proxy` replaces the proxy; `Router.Ready` feeds `/readyz` |
| Forward | `gw.06` | `cache.Middleware` is the `Cache` slot |
| Forward | `gw.07` | `ledger.Meter` is the `Ledger` slot; the admin API answers errors with `WriteError` |

If you skip this module, every `gw.02` to `gw.07` check reports `needs gw.01: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the ordered chain | Envoy HTTP filter chain | filters configured per route, async filters, per-filter stats | [Envoy HTTP filters](https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/http_filters) (free) |
| `Shutdown` and `/readyz` | Kubernetes `preStop` hooks, Envoy drain listeners | a sleep before draining so endpoints propagate; connection draining with GOAWAY on HTTP/2 | [Kubernetes pod lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/) (free) |
| `go/config` | Viper, koanf | many sources, file watching and reload | [koanf](https://github.com/knadh/koanf) (free) |
| the `Exchange` | OpenTelemetry baggage, Envoy dynamic metadata | request-scoped data across filters and processes | [Envoy dynamic metadata](https://www.envoyproxy.io/docs/envoy/latest/configuration/advanced/well_known_dynamic_metadata) (free) |
