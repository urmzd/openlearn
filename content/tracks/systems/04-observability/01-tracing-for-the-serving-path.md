# Tracing for the serving path

## Overview

| | |
|---|---|
| **Module** | `obs.01` · build · Go · Pass 7 · 4 to 6 h |
| **Solution** | `go/otelx/`: [`provider.go`](../../../../course/solutions/go/otelx/provider.go) (`Config`, `Setup`), [`propagation.go`](../../../../course/solutions/go/otelx/propagation.go) (`Inject`, `Extract`, `Traceparent`, `ContextWithTraceparent`, `IDs`), [`http.go`](../../../../course/solutions/go/otelx/http.go) (`Middleware`, `Transport`, `StartServer`, `Span.SetAttribute`), [`grpc.go`](../../../../course/solutions/go/otelx/grpc.go) (four interceptors), [`log.go`](../../../../course/solutions/go/otelx/log.go) (`LogHandler`) |
| **Contract** | span names, kinds, attributes, and propagation: [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md); the Go API: section 4 below; [W3C Trace Context](https://www.w3.org/TR/trace-context/) |
| **Needs** | nothing to call; read [`obs.00`](00-one-trace.md) first (the hand-written tracer export this replaces) |
| **Used by** | `obs.02` wires it into the gateway (log correlation, the SERVER span, the provider), which gives the production system one trace from curl through `gateway.route`, `kv.transfer`, and `engine.decode`; `dur.04` (activity spans from `ActivityTask.trace_context`) and `ag.03` (agent spans) in later passes |
| **Optional depth** | [OpenTelemetry Go](https://opentelemetry.io/docs/languages/go/) (free), [Sampling](https://opentelemetry.io/docs/concepts/sampling/) (free), *Observability Engineering*, ch. 6 and 7 |

## Key Takeaways

- Every hop **extracts** the caller's `traceparent`, starts **its own** span as the child, and **injects** its own span id onward. The same tree holds across HTTP and gRPC in one request.
- An outgoing call is a CLIENT span that lasts until the response body is done; a streamed completion is a long CLIENT span, not a 1 ms one.
- A wrapper `ResponseWriter` must keep `http.Flusher` and `Unwrap`, or the SSE stream it wraps is buffered.
- The sampler is **ParentBased**: a ratio decides only for new traces, so a sampled request never loses its middle hops.
- Spans leave through a **batch** processor; ending a span never waits for the collector. No endpoint means no exporter at all.

## How to read this chapter

Sections 1 to 3 build tracing from first principles and work one request's span tree and one sampling decision by hand. Section 4 fixes the Go API and the cases a correct kit must handle. Section 5 walks through the solution file by file and explains why each choice is the right one. Try to predict each step before reading it. The solution depends on `go.opentelemetry.io/otel` (API, SDK, and the OTLP/gRPC trace exporter) v1.44.0 and `google.golang.org/grpc` v1.83.1.

---

## 1. Why now

Pass 1 traced one hop: the gateway's `gateway.proxy` span and the engine's server span, exported by hand as OTLP/HTTP JSON (`obs.00`). The serving platform now has more hops and a second protocol: the gateway authenticates, rate-limits, routes, and calls a prefill engine over gRPC (`tl.engine.v1.EngineControl/Prefill`), which pushes KV blocks over gRPC (`tl.kv.v1`) to a decode engine, which streams tokens back over HTTP. When TTFT doubles under load, the question is which of those hops grew, and only a trace that crosses all of them answers it. A hand-written exporter per hop and per protocol does not scale to that; one small kit that every Go service (gateway now, durable engine and agent later) uses the same way does.

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

A correct kit holds these per protocol and then all at once, in one request that crosses HTTP and gRPC.

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

A traced client calls the gateway with the W3C example header:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

**Parse it** (`ContextWithTraceparent`): version `00`; $T$ = `4bf92f3577b34da6a3ce929d0e0e4736`; caller span $S$ = `00f067aa0ba902b7`; $f$ = `01`, so sampled. Writing the context back (`Traceparent`) gives the same 55 characters. With flags `00` the string comes back with `00`: the decision travels.

**The hops.** Span ids below are the first 4 bytes; every span is in trace `4bf92f35`.

| # | Span | Kind | Span id | Parent | Sent onward as `traceparent` |
|---|---|---|---|---|---|
| 1 | `POST /v1/chat/completions` (gateway, `Middleware`) | SERVER | `a1b2c3d4` | `00f067aa` (remote) | |
| 2 | `tl.engine.v1.EngineControl/Prefill` (gateway, client interceptor) | CLIENT | `51e2f3a4` | `a1b2c3d4` | `00-4bf9...-51e2f3a4...-01` in gRPC metadata |
| 3 | `tl.engine.v1.EngineControl/Prefill` (prefill engine) | SERVER | `9c8d7e6f` | `51e2f3a4` (remote) | |
| 4 | `POST` (gateway, `Transport`) | CLIENT | `77aa88bb` | `a1b2c3d4` | `00-4bf9...-77aa88bb...-01` in HTTP headers |
| 5 | `POST /v1/chat/completions` (decode engine) | SERVER | `0d0e0f10` | `77aa88bb` (remote) | |

Check the rule of 2.1 on each row: every child's parent is the span that **made** the call (rows 3 and 5 name the CLIENT spans, not row 1). If the gateway had injected its SERVER span's id in step 4 (`a1b2c3d4`), row 5 would hang directly under row 1, beside row 4 instead of under it, and the network time of the call would be unaccounted for.

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

Nothing touches the OpenTelemetry globals: each function takes the provider it uses, and propagation is always W3C. The gateway's chain (`gw.01`) asks for a `server.Tracer`; three lines in the gateway's entry point adapt `StartServer` to it:

```go
type tracer struct{ tp trace.TracerProvider }
func (t tracer) Start(ctx context.Context, name string) (context.Context, server.Span) {
	return otelx.StartServer(ctx, t.tp, name)
}
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The W3C example header | unit | the section 3 header parsed, written back, flags kept, a child keeps $T$ | every hop starts here |
| Invalid traceparents are ignored | boundary | no flags, version `ff`, zero ids, upper-case hex, 31 digits: no span context | a garbage header must not merge requests |
| The middleware continues the caller's trace | conformance | the SERVER span is the caller's child; the handler sees it; method, route, status | traced clients (the agent, drill evidence) |
| No header starts a new trace | unit | no header gives a root with a valid trace id | the normal case for curl |
| SSE keeps streaming through the middleware | unit | the first SSE event reaches the client while the handler still runs | TTFT measured through the gateway |
| `http.ResponseController` works through the wrapper | unit | `SetWriteDeadline` works through the wrapper | write deadlines for slow clients |
| Status codes and errors | boundary | 200, 404, 499 unset; 500, 503 Error with `error.type`; nothing written is 200 | error ratios and red spans mean server faults |
| Spans named by route | unit | `GET /v1/models/{model}` twice, `GET` without a route, no raw path | span names stay a small set |
| The transport injects its own span | conformance | CLIENT span under the caller; upstream sees the CLIENT span's id; request not modified | the section 3 table, rows 4 and 5 |
| The CLIENT span covers the streamed body | unit | the CLIENT span is still open after the first event | a stream's duration is its body's |
| Connection errors are recorded | fault | refused connection: Error and `error.type` | the failover attempt (`gw.05`) is visible |
| gRPC unary propagation | conformance | CLIENT and SERVER `grpc.health.v1.Health/Check`, parentage, `rpc.*`, caller metadata kept | `EngineControl/Prefill` |
| gRPC errors mark both spans | boundary | NotFound: both spans Error, code 5, `error.type` `NotFound` | a prefill pool outage is red |
| gRPC stream propagation | conformance | `Watch` stream: SERVER under CLIENT | `tl.kv.v1` and heartbeat streams |
| The serving tree across HTTP and gRPC | conformance | five spans, one trace, the section 3 tree | the production system's end-to-end trace |
| `StartServer`'s attributes keep their types | unit | int, bool, float, string slice, string, int64 keep their types | the gateway chain's attributes |
| Resource and ParentBased sampling | unit | ratio 0 drops roots, keeps children of sampled callers; ratio 1 respects unsampled callers; resource | section 2.4 |
| Bad config is rejected | boundary | empty service name, ratio 1.5 or -0.1 rejected | config mistakes fail at startup |
| Export over OTLP/gRPC | conformance | a fake collector receives the span with `service.name` | the collector in the cluster |
| No endpoint shuts down at once | boundary | no endpoint: spans still recorded, `Shutdown` under 1 s | local runs and tests |
| Ending spans never waits for the collector | fault | a collector that never answers adds no latency to 5 requests | a collector outage is not a gateway outage |
| Log lines carry trace and span ids | unit | `trace_id` and `span_id` inside a span, also through `With`, none outside | `obs.02`'s `kubectl logs \| grep` |

## 5. The solution, step by step

### 5.1 From first principles

1. **A trace is a tree, and a tree needs parent pointers.** Section 2.1 says a span's parent is the span that made the call. The only thing that crosses a process boundary is a header, so each hop must read the caller's span id (Extract), use it as the parent of a new span (Start), and write the new span's id onward (Inject). Nothing else is needed to build the section 3 tree; everything else is naming and recording.
2. **The span that calls must be a span of its own.** If the gateway injected its SERVER span's id, rows 4 and 5 of the table would collapse and the network time would vanish. So every outgoing call (HTTP `Transport`, gRPC client interceptors) starts a CLIENT span first and injects *that* span's context.
3. **One propagator, no globals.** Every process must agree on the header format, so the kit fixes one: W3C `TraceContext`. Taking a `TracerProvider` argument instead of reading `otel.GetTracerProvider()` means tests build a provider per test and a process cannot be half-configured by an import side effect.
4. **The same decision everywhere.** Section 2.4 derives that a hash of $T$ (the ratio sampler) gives every process the same answer for roots, and that children must copy their parent's flag. That is exactly `ParentBased(TraceIDRatioBased(r))`; nothing custom is needed.
5. **Tracing must never cost a request.** Section 2.5 forces a batch processor, a lazy (non-dialing) exporter, and no exporter at all without an endpoint.
6. **Wrappers must be transparent.** A recorder that sees the status code must still flush (SSE) and unwrap (deadlines); a CLIENT span must end with the body, not the headers (section 2.6). The gRPC client stream has the same shape: it ends at the first `RecvMsg` error.
7. **Logs join by id.** A log line inside a span gets `trace_id` and `span_id`, spelled as the contract spells them, so a trace id found in Tempo greps the logs.

### 5.2 The code, annotated

**Propagation** is a thin layer over the SDK's `propagation.TraceContext`. The validation section 2.1 needs (version `ff`, zero ids, upper-case hex, wrong length) is the W3C propagator's, so the kit does not re-parse anything.

```go
// w3c is the only propagator the system uses: traceparent plus tracestate.
var w3c = propagation.TraceContext{}

func Inject(ctx context.Context, h http.Header) {
	w3c.Inject(ctx, propagation.HeaderCarrier(h))
}

// An invalid or absent header returns ctx unchanged: the next span is a root.
func Extract(ctx context.Context, h http.Header) context.Context {
	return w3c.Extract(ctx, propagation.HeaderCarrier(h))
}

// The single-string form (ActivityTask.trace_context, TRACEPARENT for Python)
// reuses the same propagator through a one-key map carrier, so the string
// and header forms can never disagree on what is valid.
func Traceparent(ctx context.Context) string {
	carrier := propagation.MapCarrier{}
	w3c.Inject(ctx, carrier)
	return carrier.Get("traceparent") // "" without a valid span context
}

func ContextWithTraceparent(ctx context.Context, s string) context.Context {
	return w3c.Extract(ctx, propagation.MapCarrier{"traceparent": s})
}
```

Because `Traceparent` serialises the span context's real flags, an unsampled caller's `-00` comes back as `-00`: the decision travels, as section 3 requires.

**`Setup`** validates first, so a bad config fails at startup instead of producing silently wrong traces, then builds exactly the resource the config names:

```go
func Setup(ctx context.Context, cfg Config, opts ...sdktrace.TracerProviderOption) (*sdktrace.TracerProvider, error) {
	if cfg.ServiceName == "" {
		return nil, errors.New("otelx: Config.ServiceName is required (service.name, e.g. forge-gateway)")
	}
	if cfg.SampleRatio < 0 || cfg.SampleRatio > 1 {
		return nil, fmt.Errorf("otelx: SampleRatio %v is outside [0, 1]", cfg.SampleRatio)
	}
	attrs := []attribute.KeyValue{attribute.String("service.name", cfg.ServiceName)}
	// ... service.namespace, service.version, then cfg.Attributes in sorted key order
	all := []sdktrace.TracerProviderOption{
		// Schemaless and without resource.Default(): the resource is exactly
		// what the config says, not whatever OTEL_RESOURCE_ATTRIBUTES holds.
		sdktrace.WithResource(resource.NewSchemaless(attrs...)),
		// Section 2.4: the ratio decides only roots; children copy the parent.
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(cfg.SampleRatio))),
	}
	if cfg.Endpoint != "" { // no endpoint: no exporter, so Shutdown never retries a dead socket
		// ... timeout defaults to 5 s; http:// or https:// URLs go through
		// WithEndpointURL, a bare host:port through WithEndpoint + WithInsecure
		// New does not dial: a collector that is down at startup is not a startup failure.
		exp, err := otlptracegrpc.New(ctx, eopts...)
		if err != nil {
			return nil, fmt.Errorf("otelx: OTLP exporter for %s: %w", cfg.Endpoint, err)
		}
		// Batch, never WithSyncer: span.End() only enqueues (section 2.5).
		all = append(all, sdktrace.WithBatcher(exp, sdktrace.WithBatchTimeout(time.Second)))
	}
	return sdktrace.NewTracerProvider(append(all, opts...)...), nil
}
```

`opts` go last so a caller (a test, or an entry point) can add a recording span processor without the kit knowing about it.

**The HTTP server side.** `Middleware` does the three steps of a hop in order: Extract, Start, and (implicitly, through the context handed to `next`) make the new span the parent of anything the handler does.

```go
func Middleware(tp trace.TracerProvider, route func(*http.Request) string, next http.Handler) http.Handler {
	tracer := tp.Tracer(ScopeName)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := Extract(r.Context(), r.Header) // step 1: the caller's span context, if valid
		rt := ""
		if route != nil {
			rt = route(r) // the route *pattern*; r.Pattern is not set yet at this point
		}
		name := r.Method
		if rt != "" {
			name = r.Method + " " + rt // "GET /v1/models/{model}": low cardinality
		}
		// ... attrs: http.request.method, http.route when known
		ctx, span := tracer.Start(ctx, name, trace.WithSpanKind(trace.SpanKindServer), trace.WithAttributes(attrs...))
		defer span.End()
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r.WithContext(ctx)) // the handler's spans are children of this one
		code := rec.code
		if code == 0 {
			code = http.StatusOK // the handler wrote nothing: net/http answers 200
		}
		span.SetAttributes(attribute.Int("http.response.status_code", code))
		if code >= 500 { // section 2.3: only server faults are errors
			span.SetStatus(codes.Error, http.StatusText(code))
			span.SetAttributes(attribute.String("error.type", strconv.Itoa(code)))
		}
	})
}
```

The recorder is where section 2.6 lives. Embedding gives it `Header`; it overrides `WriteHeader` and `Write` to remember the first code, and adds the two methods embedding would hide:

```go
func (s *statusRecorder) Flush() {
	if s.code == 0 {
		s.code = http.StatusOK // a flush commits the headers, so the status is 200
	}
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush() // SSE: each event leaves now, not when the handler returns
	}
}

// Unwrap lets http.NewResponseController reach the real writer (SetWriteDeadline).
func (s *statusRecorder) Unwrap() http.ResponseWriter {
	return s.ResponseWriter
}
```

**The HTTP client side.** `Transport.RoundTrip` starts the CLIENT span, then clones the request before injecting, because a `RoundTripper` must not modify its caller's request:

```go
	ctx, span := t.tracer.Start(r.Context(), r.Method, trace.WithSpanKind(trace.SpanKindClient), trace.WithAttributes(attrs...))
	// RoundTrip must not modify the caller's request: clone, then inject.
	out := r.Clone(ctx)
	Inject(ctx, out.Header) // ctx holds the CLIENT span: the upstream's parent is this span (row 4)
	resp, err := t.base.RoundTrip(out)
	if err != nil { // refused connection, timeout: no response, so error.type names the failure
		span.SetStatus(codes.Error, err.Error())
		span.SetAttributes(attribute.String("error.type", errorType(err)))
		span.End()
		return nil, err
	}
	span.SetAttributes(attribute.Int("http.response.status_code", resp.StatusCode))
	// ... >= 500 marks Error, as on the server side
	// The call lasts until the body is consumed: for a streamed completion
	// that is the last SSE event, not the response headers.
	resp.Body = &spanBody{ReadCloser: resp.Body, span: span}
	return resp, nil
```

`spanBody` ends the span on the first read error (including `io.EOF`) or on `Close`, guarded by a `sync.Once`, because callers commonly do both.

**Typed attributes.** The gateway chain records ints, bools, and slices; `Span.SetAttribute` keeps each type so a backend can compare `http.response.status_code >= 500` numerically:

```go
func (s Span) SetAttribute(key string, value any) {
	var kv attribute.KeyValue
	switch v := value.(type) {
	case string:
		kv = attribute.String(key, v)
	case bool:
		kv = attribute.Bool(key, v)
	case int:
		kv = attribute.Int(key, v)
	// ... int64, float64, []string
	default:
		kv = attribute.String(key, fmt.Sprint(v)) // unknown types still record, as text
	}
	s.Span.SetAttributes(kv)
}
```

**gRPC.** The same three steps, with metadata instead of headers. `mdCarrier` adapts `metadata.MD` to the propagator; the client side copies the outgoing metadata before injecting, for the same reason `Transport` clones the request (the caller's `x-request-id` must survive):

```go
func clientContext(ctx context.Context) context.Context {
	// A copy: the caller's metadata must not change under it.
	md, _ := metadata.FromOutgoingContext(ctx)
	md = md.Copy()
	w3c.Inject(ctx, mdCarrier(md)) // Set, not Append: one traceparent per call
	return metadata.NewOutgoingContext(ctx, md)
}

// "/tl.engine.v1.EngineControl/Prefill" -> span name without the leading "/"
// (semconv's form), plus rpc.system, rpc.service, rpc.method.
func rpcAttrs(fullMethod string) (string, []attribute.KeyValue) {
	name := strings.TrimPrefix(fullMethod, "/")
	service, method, _ := strings.Cut(name, "/")
	// ...
}

func endRPC(span trace.Span, err error) {
	st, _ := status.FromError(err) // nil err -> codes.OK (0)
	span.SetAttributes(attribute.Int("rpc.grpc.status_code", int(st.Code())))
	if err != nil {
		span.SetStatus(codes.Error, st.Message())
		span.SetAttributes(attribute.String("error.type", st.Code().String())) // "NotFound"
	}
	span.End()
}
```

The unary client interceptor is then three lines: start a CLIENT span from `ctx`, invoke with `clientContext(ctx)`, `endRPC`. The server interceptors start a SERVER span from `serverContext(ctx)` (Extract from incoming metadata). For streams, the server wraps the `ServerStream` so `Context()` returns the span's context, and the client wraps the `ClientStream` so its span ends at the first `RecvMsg` error, treating `io.EOF` as OK because that is how a server finishes a stream:

```go
func (c *clientStream) RecvMsg(m any) error {
	err := c.ClientStream.RecvMsg(m)
	if err != nil && !c.ended {
		c.ended = true
		if errors.Is(err, io.EOF) {
			endRPC(c.span, nil) // the server finished the stream: OK
		} else {
			endRPC(c.span, err)
		}
	}
	return err
}
```

**Logs.** `LogHandler` adds the ids when the record's context has a valid span, and must re-wrap derived handlers:

```go
func (h logHandler) Handle(ctx context.Context, r slog.Record) error {
	if tid, sid, ok := IDs(ctx); ok {
		r = r.Clone() // records are shared with other handlers; do not mutate in place
		r.AddAttrs(slog.String("trace_id", tid), slog.String("span_id", sid))
	}
	return h.next.Handle(ctx, r)
}

// WithAttrs and WithGroup keep the wrapper around the derived handler, or
// logger.With(...) would silently drop the ids.
func (h logHandler) WithAttrs(as []slog.Attr) slog.Handler {
	return logHandler{h.next.WithAttrs(as)}
}
```

Full solution: [`provider.go`](../../../../course/solutions/go/otelx/provider.go), [`propagation.go`](../../../../course/solutions/go/otelx/propagation.go), [`http.go`](../../../../course/solutions/go/otelx/http.go), [`grpc.go`](../../../../course/solutions/go/otelx/grpc.go), [`log.go`](../../../../course/solutions/go/otelx/log.go).

### 5.3 Why this solution

- **Cost.** Per request: one span start and end per hop (a few allocations and a random id), a header clone on the client side, and an enqueue into the batch processor. No I/O happens on the request goroutine. Memory is bounded by the batch queue (2048 spans by default), which drops spans rather than block when the collector is down.
- **Why not `otelhttp` and `otelgrpc`?** They are the production choice (Going further), but they hide the three steps behind options and record far more. Here every line maps to a rule in section 2, and the names and attributes are exactly those of `semconv.md`.
- **Why not the OTel globals?** A global provider makes tests share state and lets an import order decide whether spans export. Passing `tp` costs one argument.
- **Why head sampling?** It needs no coordination and keeps whole traces. The accepted cost: a rare slow trace at $r < 1$ may be dropped. Tail sampling in the Collector fixes that later without changing this kit.
- **Why the SDK's propagator instead of a hand parser?** `obs.00` hand-parsed the header to teach it; production code should share the one implementation every other OTel process uses, so validity rules cannot drift.

### 5.4 Explain it back

1. Why does `Transport` inject the CLIENT span's context and not the caller's?
   <details><summary>Answer</summary>The upstream server's parent must be the span that made the call. Injecting the caller's (SERVER) span id makes the upstream a sibling of the CLIENT span, so the network and queueing time of the call is no longer inside any parent-child interval, and the tree no longer matches section 3.</details>
2. Why does the middleware mark 503 as Error but leave 404 unset?
   <details><summary>Answer</summary>Status Error means "this service failed". A 4xx is the client's mistake; marking it would inflate error ratios and page people for malformed requests. A 5xx is the server's failure and must be red.</details>
3. What goes wrong if `statusRecorder` has no `Flush` method?
   <details><summary>Answer</summary>Embedding `http.ResponseWriter` exposes only that interface's methods, so the handler's `w.(http.Flusher)` assertion fails, SSE events sit in the buffer until the handler returns, and TTFT through the gateway equals total latency.</details>
4. Why is the sampler `ParentBased(TraceIDRatioBased(r))` and not just `TraceIDRatioBased(r)`?
   <details><summary>Answer</summary>Different services may run with different ratios. Without ParentBased, each one re-decides, so a trace the gateway kept at $r = 1$ loses the engine spans an engine at $r = 0.1$ drops. ParentBased makes children copy the sampled flag, so a trace is all or nothing.</details>
5. Why does `Setup` build no exporter when `Endpoint` is empty, instead of using the SDK default?
   <details><summary>Answer</summary>The default targets `localhost:4317`; with nothing listening, every shutdown flush retries until the export timeout (5 s). Without an exporter, spans still get ids, so propagation and log correlation keep working, and `Shutdown` returns at once.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. The middleware never calls `Extract` | every request is a new trace; a traced client's tree stops at its own span | `Middleware` starts from `Extract(r.Context(), r.Header)`; see 5.2 |
| 2. The caller's context injected instead of the CLIENT span's, or the caller's request modified | the downstream server is a sibling of the client span; the caller's `http.Request` grows a header | `RoundTrip` starts the CLIENT span, clones the request with its context, and injects into the clone |
| 3. The wrapper hides `Flush` or has no `Unwrap` | SSE arrives in one lump at the end; `SetWriteDeadline` returns `ErrNotSupported` | `statusRecorder` implements both and forwards them |
| 4. Raw paths or `/`-prefixed method names in span names | one span name per model; gRPC names `/tl.engine.v1...` that match nothing in semconv | names come from the `route` callback's pattern; `rpcAttrs` trims the leading `/` |
| 5. Error status on the wrong codes, or none | 404s page people, or 503s look healthy; failed RPCs and refused connections are green | `code >= 500` only on HTTP; `endRPC` for any non-OK gRPC code; `RoundTrip` marks transport errors with `errorType` |
| 6. A ratio sampler without ParentBased | traces with random holes in the middle | `Setup` wraps the ratio sampler in `ParentBased` |
| 7. `WithSyncer` (inline export) | every request waits for the collector; a slow collector is a slow gateway | `Setup` uses `WithBatcher` |
| 8. gRPC metadata not extracted, not injected, or replaced | prefill spans start new traces; the caller's `x-request-id` disappears | `serverContext` extracts; `clientContext` copies the outgoing metadata and injects into the copy |
| 9. Flags always `01`, or upper-case hex accepted | downstream records what the caller dropped; ids that other tools reject | the W3C propagator serialises the real flags and rejects invalid headers |
| 10. Every attribute turned into a string | status codes cannot be compared or aggregated | `Span.SetAttribute` switches on the value's type |
| 11. Endpoint ignored, or an exporter without one | no spans in Tempo; or 5 s hangs at every shutdown | the exporter is built from `Endpoint` and only when it is set |
| 12. `logger.With(...)` drops the wrapper, or the ids use OTel's `traceId` spelling | some log lines lose their ids; the grep in `obs.02` finds nothing | `WithAttrs` and `WithGroup` re-wrap; the keys are `trace_id` and `span_id` |
| 13. The CLIENT span ends when the headers arrive | a 10 s stream shows as 1 ms; the server span sticks out of its parent | `spanBody` ends the span at EOF, a read error, or `Close` |
| 14. A thin resource, or no config validation | every service is `unknown_service`; a 1.5 ratio is accepted silently | `Setup` rejects an empty name and a ratio outside $[0, 1]$, and sets name, namespace, version, and extra attributes |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.00` | the tracer's hand-written export; the same three steps of a hop, by hand |
| Forward | `obs.02` | the gateway entry point: `Setup`, the SERVER span through `StartServer`, `LogHandler` for JSON logs with ids |
| Forward | `dur.04` | activity spans from `ActivityTask.trace_context` via `ContextWithTraceparent`; `TRACEPARENT` for Python via `Traceparent` (`dur.09`) |
| Forward | `ag.03` | `agent.run`, `agent.llm_call`, and `agent.tool` spans |
| Related | `L10.7` | the Rust half: `tracing-opentelemetry` in the engine joins the same trace from the `traceparent` that `Transport` and the interceptors send |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Middleware`, `Transport` | `otelhttp` (opentelemetry-go-contrib) | metrics from the same middleware, span name formatters, filters | `instrumentation/net/http/otelhttp` |
| four gRPC interceptors | `otelgrpc` stats handlers | message events, sizes, one handler per connection instead of per call | `instrumentation/google.golang.org/grpc/otelgrpc` |
| head sampling with `ParentBased` | tail sampling in the Collector | keep every slow or failed trace, sample the fast ones, after the fact | Collector `tailsamplingprocessor` |
| `LogHandler` | the OTel log bridge (`otelslog`) | logs exported as OTLP with trace context, into a log backend | `bridges/otelslog` |
| W3C only | W3C plus baggage | request-scoped key-values (tenant, experiment) on every hop | [W3C Baggage](https://www.w3.org/TR/baggage/) |
