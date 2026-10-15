<!-- ol:module obs.01 -->
# Tracing for the serving path

## Overview

| | |
|---|---|
| **Module** | `obs.01` · build · Go · Pass 7 · 4 to 6 h |
| **You build** | `go/otelx/`: `Setup` (the process's TracerProvider), `Inject`, `Extract`, `Traceparent`, `ContextWithTraceparent`, `IDs`, the HTTP `Middleware` and `Transport`, `StartServer` with `Span.SetAttribute`, four gRPC interceptors, and `LogHandler` |
| **Contract** | span names, kinds, attributes, and propagation: [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md); the Go API: section 4 below and the course tests; [W3C Trace Context](https://www.w3.org/TR/trace-context/) |
| **Tests** | `course/tests/go/obs_01/` (what they check: section 4) |
| **Needs** | nothing to call; read [`obs.00`](00-one-trace.md) first (the hand-written tracer export this replaces) |
| **Used by** | `obs.02` wires it into your gateway (log correlation, the SERVER span, the provider); `dur.04` (activity spans from `ActivityTask.trace_context`) and `ag.03` (agent spans) in later passes |
| **Milestone** | MS-prod (one trace from curl through `gateway.route`, `kv.transfer`, and `engine.decode`) |
| **Optional depth** | [OpenTelemetry Go](https://opentelemetry.io/docs/languages/go/) (free), [Sampling](https://opentelemetry.io/docs/concepts/sampling/) (free), *Observability Engineering*, ch. 6 and 7 |

## Key Takeaways

- Every hop **extracts** the caller's `traceparent`, starts **its own** span as the child, and **injects** its own span id onward. The tests prove the tree across HTTP and gRPC in one request.
- An outgoing call is a CLIENT span that lasts until the response body is done; a streamed completion is a long CLIENT span, not a 1 ms one.
- A wrapper `ResponseWriter` must keep `http.Flusher` and `Unwrap`, or the SSE stream it wraps is buffered.
- The sampler is **ParentBased**: a ratio decides only for new traces, so a sampled request never loses its middle hops.
- Spans leave through a **batch** processor; ending a span never waits for the collector. No endpoint means no exporter at all.

## How to work this chapter

```bash
ol start obs.01                # stubs go/otelx/*.go into your repo
ol tests obs.01                # read the test catalog first
cd go && go get go.opentelemetry.io/otel@v1.44.0 go.opentelemetry.io/otel/sdk@v1.44.0 \
  go.opentelemetry.io/otel/trace@v1.44.0 \
  go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc@v1.44.0 google.golang.org/grpc@v1.83.1
ol check obs.01                # exit code is the verdict
ol diff  obs.01                # after passing: your code against the reference
```

---

## 1. Why now

Pass 1 traced one hop: the gateway's `gateway.proxy` span and the engine's server span, exported by hand as OTLP/HTTP JSON (`obs.00`). The serving platform you are building now has more hops and a second protocol: the gateway authenticates, rate-limits, routes, and calls a prefill engine over gRPC (`tl.engine.v1.EngineControl/Prefill`), which pushes KV blocks over gRPC (`tl.kv.v1`) to a decode engine, which streams tokens back over HTTP. When TTFT doubles under load, the question is which of those hops grew, and only a trace that crosses all of them answers it. A hand-written exporter per hop and per protocol does not scale to that; one small kit that every Go service (gateway now, durable engine and agent later) uses the same way does.

## 2. Principles

### 2.1 Span context and the three steps of a hop

| Symbol | Meaning | Type |
|---|---|---|
| $T$ | trace id, shared by every span of one request | 16 bytes, 32 lowercase hex digits |
| $S$ | span id of one span | 8 bytes, 16 hex digits |
| $P(s)$ | the parent span id of span $s$ | 8 bytes, or none for the root |
| $f$ | trace flags; bit 0 is **sampled** | 1 byte, 2 hex digits |
| $r$ | the sample ratio, `[otel].trace_sample_ratio` | real in $[0, 1]$ |

A **span context** is $(T, S, f)$, plus `tracestate` for vendor data. A hop does three things:

1. **Extract**: read `traceparent` from the incoming request (HTTP header or gRPC metadata). If it is valid, the context of the request holds a *remote* span context $(T, S_{caller}, f)$; if not, it holds none.
2. **Start**: a new span $s$ with a fresh random $S_s$. If the context held a span context, $T$ is kept and $P(s) = S_{caller}$; otherwise $T$ is fresh and $s$ is a root.
3. **Inject**: every outgoing call carries `traceparent = 00-T-S_c-f`, where $S_c$ is the span id of the span that **makes the call**. For HTTP and gRPC that is a CLIENT span started for the call.

The test suite checks these per protocol and then all at once (`TestServingTreeAcrossHTTPAndGRPC`).

### 2.2 Span kinds and names

| Kind | Who records it | Name |
|---|---|---|
| SERVER | the side that handled a request | `<METHOD> <route>` for HTTP (`POST /v1/chat/completions`); `<service>/<method>` for gRPC (`tl.engine.v1.EngineControl/Prefill`) |
| CLIENT | the side that made a call | `<METHOD>` for HTTP; the same `<service>/<method>` for gRPC |
| INTERNAL | work inside a process | `gateway.route`, `engine.decode` |

Names must have **low cardinality**: the route *pattern* `/v1/models/{model}`, never the path `/v1/models/smol-135m`. Each distinct name becomes a row in every trace search and a series in every span-derived metric, so a name per model (or per request id) explodes both.

### 2.3 Status: whose error is it

A server span records `http.response.status_code` always. It sets status **Error** (and `error.type` to the code) only for $code \ge 500$: a 4xx is the client's mistake, and marking it as a server error would page someone for a typo in a request. gRPC spans record `rpc.grpc.status_code` (0 is OK) and set Error with `error.type` = the code's name (`NotFound`) for anything else.

### 2.4 Sampling

Recording every span of every request costs CPU, network, and storage. A **sampler** decides at the root. `TraceIDRatioBased(r)` keeps a trace when the low 8 bytes of $T$, read as a big-endian unsigned integer and shifted right once, are below $r \cdot 2^{63}$:

$$\text{keep}(T) \iff \left\lfloor \tfrac{\text{u64}(T[8{:}16])}{2} \right\rfloor < r \cdot 2^{63}$$

Every process makes the **same** decision for the same $T$ without talking to the others. `ParentBased(sampler)` applies the ratio only to roots; a span with a parent copies the parent's sampled flag. Without ParentBased, a gateway at $r = 1$ and an engine at $r = 0.1$ produce traces with 90% of their engine spans missing.

### 2.5 Export without hurting requests

The SDK hands every ended, sampled span to a **span processor**. The **batch** processor queues it and exports in the background every second (or every 512 spans); the **simple** processor exports inline, inside `span.End()`, on the request's goroutine. With a slow collector, simple turns into request latency. `Setup` uses batch, and builds no exporter at all when `Endpoint` is empty: an exporter aimed at the default `localhost:4317` retries a refused connection until its timeout every time the process stops.

### 2.6 Streams and wrappers

To read the status code, a middleware wraps the `http.ResponseWriter`. A wrapper type that embeds `http.ResponseWriter` exposes only that interface's three methods: the `Flush` method of the real writer is hidden, so `w.(http.Flusher)` fails in the handler and SSE events pile up in a buffer until the handler returns. The wrapper must implement `Flush` (forwarding it) and `Unwrap() http.ResponseWriter` (so `http.NewResponseController` reaches the real writer's deadlines). On the client side, the response headers arrive long before the last SSE event, so the CLIENT span ends when the **body** reaches EOF or is closed.

## 3. Worked example by hand

A traced client calls your gateway with the W3C example header:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

**Parse it** (`ContextWithTraceparent`, the case `TestTraceparentHandExample` checks): version `00`; $T$ = `4bf92f3577b34da6a3ce929d0e0e4736`; caller span $S$ = `00f067aa0ba902b7`; $f$ = `01`, so sampled. Writing the context back (`Traceparent`) gives the same 55 characters. With flags `00` the string comes back with `00`: the decision travels.

**The hops.** Span ids below are the first 4 bytes; every span is in trace `4bf92f35`.

| # | Span | Kind | Span id | Parent | Sent onward as `traceparent` |
|---|---|---|---|---|---|
| 1 | `POST /v1/chat/completions` (gateway, `Middleware`) | SERVER | `a1b2c3d4` | `00f067aa` (remote) | |
| 2 | `tl.engine.v1.EngineControl/Prefill` (gateway, client interceptor) | CLIENT | `51e2f3a4` | `a1b2c3d4` | `00-4bf9...-51e2f3a4...-01` in gRPC metadata |
| 3 | `tl.engine.v1.EngineControl/Prefill` (prefill engine) | SERVER | `9c8d7e6f` | `51e2f3a4` (remote) | |
| 4 | `POST` (gateway, `Transport`) | CLIENT | `77aa88bb` | `a1b2c3d4` | `00-4bf9...-77aa88bb...-01` in HTTP headers |
| 5 | `POST /v1/chat/completions` (decode engine) | SERVER | `0d0e0f10` | `77aa88bb` (remote) | |

Check the rule of 2.1 on each row: every child's parent is the span that **made** the call (rows 3 and 5 name the CLIENT spans, not row 1). If the gateway had injected its SERVER span's id in step 4 (`a1b2c3d4`), row 5 would hang directly under row 1, beside row 4 instead of under it, and the network time of the call would be unaccounted for. That is mutant `s02`.

**The sampling decision at the gateway**, for a root request whose fresh trace id happened to be this one, at $r = 0.25$ and $r = 0.75$:

| Step | Value |
|---|---|
| $T[8{:}16]$ | `a3ce929d0e0e4736` |
| as u64 | 11 803 532 876 627 986 230 |
| shifted right once ($x$) | 5 901 766 438 313 993 115 |
| $0.25 \cdot 2^{63}$ | 2 305 843 009 213 693 952, so $x$ is not below it: **dropped** |
| $0.75 \cdot 2^{63}$ | 6 917 529 027 641 081 856, so $x$ is below it: **kept** |

$x / 2^{63} \approx 0.640$: this trace is kept by every ratio above 0.640 and dropped by every ratio below, in every process. Here the request carried `-01`, so ParentBased skips the ratio entirely and keeps it.

## 4. The interface

```go
package otelx // go/otelx: provider.go, propagation.go, http.go, grpc.go, log.go

type Config struct {
	Endpoint      string            // [otel].endpoint, OTLP/gRPC, "http://otel-collector.observability:4317"; "" = no exporter
	ServiceName   string            // service.name, "<system>-gateway" (required)
	Namespace     string            // service.namespace, "<system>"
	Version       string            // service.version, [system].version
	SampleRatio   float64           // [otel].trace_sample_ratio, in [0, 1]
	Attributes    map[string]string // extra resource attributes, e.g. tl.engine.role
	ExportTimeout time.Duration     // one export; 0 = 5 s
}
func Setup(ctx context.Context, cfg Config, opts ...sdktrace.TracerProviderOption) (*sdktrace.TracerProvider, error)

func Inject(ctx context.Context, h http.Header)
func Extract(ctx context.Context, h http.Header) context.Context
func Traceparent(ctx context.Context) string                     // "" without a valid span context
func ContextWithTraceparent(ctx context.Context, s string) context.Context
func IDs(ctx context.Context) (traceID, spanID string, ok bool)

func Middleware(tp trace.TracerProvider, route func(*http.Request) string, next http.Handler) http.Handler
func Transport(tp trace.TracerProvider, base http.RoundTripper) http.RoundTripper
type Span struct{ trace.Span }
func (s Span) SetAttribute(key string, value any)
func StartServer(ctx context.Context, tp trace.TracerProvider, name string) (context.Context, Span)

func UnaryServerInterceptor(tp trace.TracerProvider) grpc.UnaryServerInterceptor
func StreamServerInterceptor(tp trace.TracerProvider) grpc.StreamServerInterceptor
func UnaryClientInterceptor(tp trace.TracerProvider) grpc.UnaryClientInterceptor
func StreamClientInterceptor(tp trace.TracerProvider) grpc.StreamClientInterceptor

func LogHandler(next slog.Handler) slog.Handler                  // adds trace_id and span_id
```

Nothing touches the OpenTelemetry globals: each function takes the provider it uses, and propagation is always W3C. The gateway's chain (`gw.01`) asks for a `server.Tracer`; three lines in your entry point adapt `StartServer` to it:

```go
type tracer struct{ tp trace.TracerProvider }
func (t tracer) Start(ctx context.Context, name string) (context.Context, server.Span) {
	return otelx.StartServer(ctx, t.tp, name)
}
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestTraceparentHandExample` | unit | the section 3 header parsed, written back, flags kept, a child keeps $T$ | every hop starts here |
| `TestInvalidTraceparentIsIgnored` | boundary | no flags, version `ff`, zero ids, upper-case hex, 31 digits: no span context | a garbage header must not merge requests |
| `TestMiddlewareContinuesCallerTrace` | conformance | the SERVER span is the caller's child; the handler sees it; method, route, status | traced clients (the agent, `ol drill` evidence) |
| `TestMiddlewareStartsNewTraceWithoutHeader` | unit | no header gives a root with a valid trace id | the normal case for curl |
| `TestMiddlewareKeepsSSEStreaming` | unit | the first SSE event reaches the client while the handler still runs | TTFT measured through the gateway |
| `TestMiddlewareSupportsResponseController` | unit | `SetWriteDeadline` works through the wrapper | write deadlines for slow clients |
| `TestMiddlewareStatusAndErrors` | boundary | 200, 404, 499 unset; 500, 503 Error with `error.type`; nothing written is 200 | error ratios and red spans mean server faults |
| `TestMiddlewareNamesSpansByRoute` | unit | `GET /v1/models/{model}` twice, `GET` without a route, no raw path | span names stay a small set |
| `TestTransportInjectsItsOwnSpan` | conformance | CLIENT span under the caller; upstream sees the CLIENT span's id; request not modified | the section 3 table, rows 4 and 5 |
| `TestTransportSpanCoversTheStreamedBody` | unit | the CLIENT span is still open after the first event | a stream's duration is its body's |
| `TestTransportRecordsConnectionErrors` | fault | refused connection: Error and `error.type` | the failover attempt (`gw.05`) is visible |
| `TestGRPCUnaryPropagation` | conformance | CLIENT and SERVER `grpc.health.v1.Health/Check`, parentage, `rpc.*`, caller metadata kept | `EngineControl/Prefill` |
| `TestGRPCErrorsMarkSpans` | boundary | NotFound: both spans Error, code 5, `error.type` `NotFound` | a prefill pool outage is red |
| `TestGRPCStreamPropagation` | conformance | `Watch` stream: SERVER under CLIENT | `tl.kv.v1` and heartbeat streams |
| `TestServingTreeAcrossHTTPAndGRPC` | conformance | five spans, one trace, the section 3 tree | MS-prod's trace step |
| `TestStartServerAdapterKeepsAttributeTypes` | unit | int, bool, float, string slice, string, int64 keep their types | the gateway chain's attributes |
| `TestSetupResourceAndParentBasedSampling` | unit | ratio 0 drops roots, keeps children of sampled callers; ratio 1 respects unsampled callers; resource | section 2.4 |
| `TestSetupRejectsBadConfig` | boundary | empty service name, ratio 1.5 or -0.1 rejected | config mistakes fail at startup |
| `TestSetupExportsOverOTLPgRPC` | conformance | a fake collector receives the span with `service.name` | the collector in your cluster |
| `TestSetupWithoutEndpointShutsDownAtOnce` | boundary | no endpoint: spans still recorded, `Shutdown` under 1 s | local runs and tests |
| `TestEndingSpansNeverWaitsForTheCollector` | fault | a collector that never answers adds no latency to 5 requests | a collector outage is not a gateway outage |
| `TestLogHandlerStampsTraceAndSpanIDs` | unit | `trace_id` and `span_id` inside a span, also through `With`, none outside | `obs.02`'s `kubectl logs \| grep` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. The middleware never calls `Extract` | every request is a new trace; a traced client's tree stops at its own span | `TestMiddlewareContinuesCallerTrace`, `TestServingTreeAcrossHTTPAndGRPC` (mutant `s01`) |
| 2. The caller's context injected instead of the CLIENT span's, or the caller's request modified | the downstream server is a sibling of the client span; the caller's `http.Request` grows a header | `TestTransportInjectsItsOwnSpan` (mutants `s02`, `s10`) |
| 3. The wrapper hides `Flush` or has no `Unwrap` | SSE arrives in one lump at the end; `SetWriteDeadline` returns `ErrNotSupported` | `TestMiddlewareKeepsSSEStreaming` (mutant `s03`), `TestMiddlewareSupportsResponseController` (mutant `s14`) |
| 4. Raw paths or `/`-prefixed method names in span names | one span name per model; gRPC names `/tl.engine.v1...` that match nothing in semconv | `TestMiddlewareNamesSpansByRoute` (mutant `s04`), `TestGRPCUnaryPropagation` (mutant `s22`) |
| 5. Error status on the wrong codes, or none | 404s page people, or 503s look healthy; failed RPCs and refused connections are green | `TestMiddlewareStatusAndErrors` (mutants `s05`, `m01`, `m02`), `TestGRPCErrorsMarkSpans` (mutant `s20`), `TestTransportRecordsConnectionErrors` (mutant `s25`) |
| 6. A ratio sampler without ParentBased | traces with random holes in the middle | `TestSetupResourceAndParentBasedSampling` (mutant `s06`) |
| 7. `WithSyncer` (inline export) | every request waits for the collector; a slow collector is a slow gateway | `TestEndingSpansNeverWaitsForTheCollector` (mutant `s07`) |
| 8. gRPC metadata not extracted, not injected, or replaced | prefill spans start new traces; the caller's `x-request-id` disappears | `TestGRPCUnaryPropagation` (mutants `s08`, `s09`, `s12`), `TestGRPCStreamPropagation` (mutant `s13`) |
| 9. Flags always `01`, or upper-case hex accepted | downstream records what the caller dropped; ids that other tools reject | `TestTraceparentHandExample` (mutant `s11`), `TestInvalidTraceparentIsIgnored` (mutant `s26`) |
| 10. Every attribute turned into a string | status codes cannot be compared or aggregated | `TestStartServerAdapterKeepsAttributeTypes` (mutant `s15`) |
| 11. Endpoint ignored, or an exporter without one | no spans in Tempo; or 5 s hangs at every shutdown | `TestSetupExportsOverOTLPgRPC` (mutant `s16`), `TestSetupWithoutEndpointShutsDownAtOnce` (mutant `s17`) |
| 12. `logger.With(...)` drops the wrapper, or the ids use OTel's `traceId` spelling | some log lines lose their ids; the grep in `obs.02` finds nothing | `TestLogHandlerStampsTraceAndSpanIDs` (mutants `s18`, `s23`) |
| 13. The CLIENT span ends when the headers arrive | a 10 s stream shows as 1 ms; the server span sticks out of its parent | `TestTransportSpanCoversTheStreamedBody` (mutant `s19`) |
| 14. A thin resource, or no config validation | every service is `unknown_service`; a 1.5 ratio is accepted silently | `TestSetupResourceAndParentBasedSampling` (mutant `s21`), `TestSetupRejectsBadConfig` (mutant `s24`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.00` | the tracer's hand-written export; the same three steps of a hop, by hand |
| Forward | `obs.02` | your gateway entry point: `Setup`, the SERVER span through `StartServer`, `LogHandler` for JSON logs with ids |
| Forward | `dur.04` | activity spans from `ActivityTask.trace_context` via `ContextWithTraceparent`; `TRACEPARENT` for Python via `Traceparent` (`dur.09`) |
| Forward | `ag.03` | `agent.run`, `agent.llm_call`, and `agent.tool` spans |
| Related | `L10.7` | the Rust half: `tracing-opentelemetry` in the engine joins the same trace from the `traceparent` your `Transport` and interceptors send |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Middleware`, `Transport` | `otelhttp` (opentelemetry-go-contrib) | metrics from the same middleware, span name formatters, filters | `instrumentation/net/http/otelhttp` |
| four gRPC interceptors | `otelgrpc` stats handlers | message events, sizes, one handler per connection instead of per call | `instrumentation/google.golang.org/grpc/otelgrpc` |
| head sampling with `ParentBased` | tail sampling in the Collector | keep every slow or failed trace, sample the fast ones, after the fact | Collector `tailsamplingprocessor` |
| `LogHandler` | the OTel log bridge (`otelslog`) | logs exported as OTLP with trace context, into a log backend | `bridges/otelslog` |
| W3C only | W3C plus baggage | request-scoped key-values (tenant, experiment) on every hop | [W3C Baggage](https://www.w3.org/TR/baggage/) |
