# Tracing for the serving path

The first trace in this chapter crossed one hop: the gateway's `gateway.proxy` span and the engine's server span, exported by hand as OTLP/HTTP JSON in [One trace from gateway to engine](01-one-trace.md). The serving platform now has more hops and a second protocol. The gateway authenticates, rate-limits, routes, and calls a prefill engine over gRPC (`tl.engine.v1.EngineControl/Prefill`), which pushes KV blocks over gRPC (`tl.kv.v1`) to a decode engine, which streams tokens back over HTTP. When time to first token (TTFT) doubles under load, the question is which of those hops grew, and only a trace that crosses all of them can answer it.

A hand-written exporter per hop and per protocol does not scale to that. What does is one small kit that every Go service uses the same way: the gateway now, the durable engine and the agent later. This section builds that kit, `go/otelx`, on top of the OpenTelemetry Go SDK. Nothing has to be running to use it, but read the previous section first, because the hand-written export there is exactly what this replaces.

The solution lives in `go/otelx/`: [`provider.go`](../../../../course/solutions/go/otelx/provider.go) (`Config`, `Setup`), [`propagation.go`](../../../../course/solutions/go/otelx/propagation.go) (`Inject`, `Extract`, `Traceparent`, `ContextWithTraceparent`, `IDs`), [`http.go`](../../../../course/solutions/go/otelx/http.go) (`Middleware`, `Transport`, `StartServer`, `Span.SetAttribute`), [`grpc.go`](../../../../course/solutions/go/otelx/grpc.go) (four interceptors), and [`log.go`](../../../../course/solutions/go/otelx/log.go) (`LogHandler`). Span names, kinds, attributes, and propagation follow the contract in [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) and [W3C Trace Context](https://www.w3.org/TR/trace-context/); the Go API is given below. The solution depends on `go.opentelemetry.io/otel` (API, SDK, and the OTLP/gRPC trace exporter) v1.44.0 and `google.golang.org/grpc` v1.83.1.

> **Learning objectives**
> - State the three steps every hop performs (extract, start, inject) and show why the span that makes a call must be a CLIENT span of its own.
> - Name spans with low cardinality and set error status only for server faults, over both HTTP and gRPC.
> - Compute a `TraceIDRatioBased` sampling decision by hand and explain why it must be wrapped in `ParentBased`.
> - Wrap an `http.ResponseWriter` without breaking SSE flushing or `http.ResponseController`, and end a CLIENT span when the streamed body ends.
> - Export spans through a batch processor so that tracing never adds request latency.

## Span context and the three steps of a hop

A few symbols make the rules precise:

| Symbol | Meaning | Type |
|---|---|---|
| $T$ | trace id, shared by every span of one request | 16 bytes, 32 lowercase hex digits |
| $S$ | span id of one span | 8 bytes, 16 hex digits |
| $P(s)$ | the parent span id of span $s$ | 8 bytes, or none for the root |
| $f$ | trace flags; bit 0 is **sampled** | 1 byte, 2 hex digits |
| $r$ | the sample ratio, `[otel].trace_sample_ratio` | real in $[0, 1]$ |

A **span context** is $(T, S, f)$, plus `tracestate` for vendor data. A hop does three things with it.

First it **extracts**: it reads `traceparent` from the incoming request, as an HTTP header or as gRPC metadata. If the value is valid, the request's context holds a *remote* span context $(T, S_{caller}, f)$; if not, it holds none.

Then it **starts** a new span $s$ with a fresh random $S_s$. If the context held a span context, $T$ is kept and $P(s) = S_{caller}$; otherwise $T$ is fresh and $s$ is a root.

Finally it **injects**: every outgoing call carries `traceparent = 00-T-S_c-f`, where $S_c$ is the span id of the span that **makes the call**. For HTTP and gRPC that is a CLIENT span started for the call.

These rules must hold per protocol, and then all at once in one request that crosses HTTP and gRPC. The same tree then holds across every hop.

## Span kinds and names

The kind of a span says which side of a call it measures, and the kind largely determines the name:

| Kind | Who records it | Name |
|---|---|---|
| SERVER | the side that handled a request | `<METHOD> <route>` for HTTP (`POST /v1/chat/completions`); `<service>/<method>` for gRPC (`tl.engine.v1.EngineControl/Prefill`) |
| CLIENT | the side that made a call | `<METHOD>` for HTTP; the same `<service>/<method>` for gRPC |
| INTERNAL | work inside a process | `gateway.route`, `engine.decode` |

Names must have **low cardinality**: the route *pattern* `/v1/models/{model}`, never the path `/v1/models/smol-135m`. Each distinct name becomes a row in every trace search and a series in every span-derived metric, so a name per model (or per request id) explodes both.

## Status: whose error is it

Names say what a span did; status says whether it failed, and the important question is whose failure it was. A server span always records `http.response.status_code`. It sets status **Error** (and `error.type` to the code) only for $code \ge 500$. A 4xx is the client's mistake, and marking it as a server error would page someone for a typo in a request. gRPC spans record `rpc.grpc.status_code` (0 is OK) and set Error, with `error.type` set to the code's name (`NotFound`), for anything else.

## Sampling

Recording every span of every request costs CPU, network, and storage, so a **sampler** decides at the root whether a trace is kept. `TraceIDRatioBased(r)` keeps a trace when the low 8 bytes of $T$, read as a big-endian unsigned integer and shifted right once, are below $r \cdot 2^{63}$:

$$\text{keep}(T) \iff \left\lfloor \tfrac{\text{u64}(T[8{:}16])}{2} \right\rfloor < r \cdot 2^{63}$$

Because the decision is a function of $T$ alone, every process makes the **same** decision for the same $T$ without talking to the others. That is necessary but not sufficient. `ParentBased(sampler)` applies the ratio only to roots; a span with a parent copies the parent's sampled flag. Without ParentBased, a gateway at $r = 1$ and an engine at $r = 0.1$ produce traces with 90% of their engine spans missing. With it, a sampled request never loses its middle hops.

## Exporting without hurting requests

Once a span is kept, it still has to leave the process. The SDK hands every ended, sampled span to a **span processor**. The **batch** processor queues it and exports in the background every second (or every 512 spans); the **simple** processor exports inline, inside `span.End()`, on the request's goroutine. With a slow collector, simple export turns into request latency, so ending a span must never wait for the collector. `Setup` uses batch.

It also builds no exporter at all when `Endpoint` is empty. An exporter aimed at the default `localhost:4317` retries a refused connection until its timeout every time the process stops.

## Streams and wrappers

The last principle concerns streaming, which is where tracing code most often breaks the thing it observes. To read the status code, a middleware wraps the `http.ResponseWriter`. A wrapper type that embeds `http.ResponseWriter` exposes only that interface's three methods: the `Flush` method of the real writer is hidden, so `w.(http.Flusher)` fails in the handler and SSE events pile up in a buffer until the handler returns. The wrapper must therefore implement `Flush`, forwarding it, and `Unwrap() http.ResponseWriter`, so that `http.NewResponseController` reaches the real writer's deadlines.

On the client side the problem is timing. The response headers arrive long before the last SSE event, so the CLIENT span ends when the **body** reaches EOF or is closed. A streamed completion is a long CLIENT span, not a 1 ms one.

## Working one request by hand

With the rules in place, trace one request through the platform. A traced client calls the gateway with the W3C example header:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
```

Parse it as `ContextWithTraceparent` does: version `00`; $T$ = `4bf92f3577b34da6a3ce929d0e0e4736`; caller span $S$ = `00f067aa0ba902b7`; $f$ = `01`, so sampled. Writing the context back with `Traceparent` gives the same 55 characters. With flags `00` the string comes back with `00`: the decision travels.

Now follow the hops. Span ids below are the first 4 bytes, and every span is in trace `4bf92f35`.

| # | Span | Kind | Span id | Parent | Sent onward as `traceparent` |
|---|---|---|---|---|---|
| 1 | `POST /v1/chat/completions` (gateway, `Middleware`) | SERVER | `a1b2c3d4` | `00f067aa` (remote) | |
| 2 | `tl.engine.v1.EngineControl/Prefill` (gateway, client interceptor) | CLIENT | `51e2f3a4` | `a1b2c3d4` | `00-4bf9...-51e2f3a4...-01` in gRPC metadata |
| 3 | `tl.engine.v1.EngineControl/Prefill` (prefill engine) | SERVER | `9c8d7e6f` | `51e2f3a4` (remote) | |
| 4 | `POST` (gateway, `Transport`) | CLIENT | `77aa88bb` | `a1b2c3d4` | `00-4bf9...-77aa88bb...-01` in HTTP headers |
| 5 | `POST /v1/chat/completions` (decode engine) | SERVER | `0d0e0f10` | `77aa88bb` (remote) | |

Check the inject rule on each row: every child's parent is the span that **made** the call, so rows 3 and 5 name the CLIENT spans, not row 1. If the gateway had injected its SERVER span's id (`a1b2c3d4`) in row 4, row 5 would hang directly under row 1, beside row 4 instead of under it, and the network time of the call would be unaccounted for.

Next, work the sampling decision at the gateway for a root request whose fresh trace id happened to be this one, at $r = 0.25$ and $r = 0.75$:

| Step | Value |
|---|---|
| $T[8{:}16]$ | `a3ce929d0e0e4736` |
| as u64 | 11 803 532 876 627 986 230 |
| shifted right once ($x$) | 5 901 766 438 313 993 115 |
| $0.25 \cdot 2^{63}$ | 2 305 843 009 213 693 952, so $x$ is not below it: **dropped** |
| $0.75 \cdot 2^{63}$ | 6 917 529 027 641 081 856, so $x$ is below it: **kept** |

Since $x / 2^{63} \approx 0.640$, this trace is kept by every ratio above 0.640 and dropped by every ratio below, in every process. In the actual request, though, the header carried `-01`, so ParentBased skips the ratio entirely and keeps it.

## Designing the kit's API

The whole kit fits on one screen:

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

Nothing touches the OpenTelemetry globals: each function takes the provider it uses, and propagation is always W3C. The gateway's middleware chain, from [The server skeleton](../../ai-platform-engineering/12-gateway/02-server-skeleton.md) in *AI Platform Engineering*, asks for a `server.Tracer`; three lines in the gateway's entry point adapt `StartServer` to it:

```go
type tracer struct{ tp trace.TracerProvider }
func (t tracer) Start(ctx context.Context, name string) (context.Context, server.Span) {
	return otelx.StartServer(ctx, t.tp, name)
}
```

## Building the kit step by step

Seven observations fix the design before any code is written.

A trace is a tree, and a tree needs parent pointers. A span's parent is the span that made the call, and the only thing that crosses a process boundary is a header, so each hop must read the caller's span id (Extract), use it as the parent of a new span (Start), and write the new span's id onward (Inject). Nothing else is needed to build the tree in the worked example; everything else is naming and recording.

The span that calls must be a span of its own. If the gateway injected its SERVER span's id, rows 4 and 5 of the table would collapse and the network time would vanish. So every outgoing call (the HTTP `Transport`, the gRPC client interceptors) starts a CLIENT span first and injects *that* span's context.

There is one propagator and no globals. Every process must agree on the header format, so the kit fixes one: W3C `TraceContext`. Taking a `TracerProvider` argument instead of reading `otel.GetTracerProvider()` means tests build a provider per test, and a process cannot be half-configured by an import side effect.

Every process makes the same decision. A hash of $T$ (the ratio sampler) gives every process the same answer for roots, and children must copy their parent's flag. That is exactly `ParentBased(TraceIDRatioBased(r))`; nothing custom is needed.

Tracing must never cost a request. That forces a batch processor, a lazy (non-dialing) exporter, and no exporter at all without an endpoint.

Wrappers must be transparent. A recorder that sees the status code must still flush (SSE) and unwrap (deadlines), and a CLIENT span must end with the body, not the headers. The gRPC client stream has the same shape: it ends at the first `RecvMsg` error.

Logs join by id. A log line inside a span gets `trace_id` and `span_id`, spelled as the contract spells them, so a trace id found in Tempo greps the logs.

### Propagation

Propagation is a thin layer over the SDK's `propagation.TraceContext`. The validation the extract step needs (rejecting version `ff`, zero ids, upper-case hex, wrong length) is the W3C propagator's, so the kit does not re-parse anything.

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

Because `Traceparent` serialises the span context's real flags, an unsampled caller's `-00` comes back as `-00`: the decision travels, as the worked example requires.

### The provider

`Setup` validates first, so a bad config fails at startup instead of producing silently wrong traces, then builds exactly the resource the config names:

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
		// The ratio decides only roots; children copy the parent.
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
		// Batch, never WithSyncer: span.End() only enqueues.
		all = append(all, sdktrace.WithBatcher(exp, sdktrace.WithBatchTimeout(time.Second)))
	}
	return sdktrace.NewTracerProvider(append(all, opts...)...), nil
}
```

The `opts` go last, so a caller (a test, or an entry point) can add a recording span processor without the kit knowing about it.

### The HTTP server side

`Middleware` does the three steps of a hop in order: Extract, Start, and (implicitly, through the context handed to `next`) make the new span the parent of anything the handler does.

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
		if code >= 500 { // only server faults are errors
			span.SetStatus(codes.Error, http.StatusText(code))
			span.SetAttributes(attribute.String("error.type", strconv.Itoa(code)))
		}
	})
}
```

The recorder is where the streaming principle lives. Embedding gives it `Header`; it overrides `WriteHeader` and `Write` to remember the first code, and adds the two methods embedding would hide:

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

### The HTTP client side

`Transport.RoundTrip` starts the CLIENT span, then clones the request before injecting, because a `RoundTripper` must not modify its caller's request:

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

### Typed attributes

The gateway chain records ints, bools, and slices. `Span.SetAttribute` keeps each type, so a backend can compare `http.response.status_code >= 500` numerically:

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

### gRPC

gRPC takes the same three steps, with metadata instead of headers. `mdCarrier` adapts `metadata.MD` to the propagator. The client side copies the outgoing metadata before injecting, for the same reason `Transport` clones the request: the caller's `x-request-id` must survive.

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

The unary client interceptor is then three lines: start a CLIENT span from `ctx`, invoke with `clientContext(ctx)`, and call `endRPC`. The server interceptors start a SERVER span from `serverContext(ctx)`, which extracts from incoming metadata. For streams, the server wraps the `ServerStream` so that `Context()` returns the span's context, and the client wraps the `ClientStream` so that its span ends at the first `RecvMsg` error, treating `io.EOF` as OK because that is how a server finishes a stream:

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

### Logs

`LogHandler` adds the ids when the record's context has a valid span, and it must re-wrap derived handlers:

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

The full solution is in [`provider.go`](../../../../course/solutions/go/otelx/provider.go), [`propagation.go`](../../../../course/solutions/go/otelx/propagation.go), [`http.go`](../../../../course/solutions/go/otelx/http.go), [`grpc.go`](../../../../course/solutions/go/otelx/grpc.go), and [`log.go`](../../../../course/solutions/go/otelx/log.go).

### Why this design

Per request, the cost is one span start and end per hop (a few allocations and a random id), a header clone on the client side, and an enqueue into the batch processor. No I/O happens on the request goroutine. Memory is bounded by the batch queue (2048 spans by default), which drops spans rather than block when the collector is down.

Why not `otelhttp` and `otelgrpc`? They are the production choice, but they hide the three steps behind options and record far more. Here every line maps to a rule stated above, and the names and attributes are exactly those of `semconv.md`.

Why not the OTel globals? A global provider makes tests share state and lets an import order decide whether spans export. Passing `tp` costs one argument.

Why head sampling? It needs no coordination and keeps whole traces. The accepted cost is that a rare slow trace at $r < 1$ may be dropped. Tail sampling in the Collector fixes that later without changing this kit.

Why the SDK's propagator instead of a hand parser? The previous section hand-parsed the header to teach it; production code should share the one implementation every other OTel process uses, so validity rules cannot drift.

## What can go wrong

A correct kit is checked case by case, and each case exists because something downstream depends on it. The table groups them in the order the kit is built:

| Case | Checks | Why it matters downstream |
|---|---|---|
| The W3C example header | the worked-example header parsed, written back, flags kept, a child keeps $T$ | every hop starts here |
| Invalid traceparents are ignored | no flags, version `ff`, zero ids, upper-case hex, 31 digits: no span context | a garbage header must not merge requests |
| The middleware continues the caller's trace | the SERVER span is the caller's child; the handler sees it; method, route, status | traced clients (the agent, drill evidence) |
| No header starts a new trace | no header gives a root with a valid trace id | the normal case for curl |
| SSE keeps streaming through the middleware | the first SSE event reaches the client while the handler still runs | TTFT measured through the gateway |
| `http.ResponseController` works through the wrapper | `SetWriteDeadline` works through the wrapper | write deadlines for slow clients |
| Status codes and errors | 200, 404, 499 unset; 500, 503 Error with `error.type`; nothing written is 200 | error ratios and red spans mean server faults |
| Spans named by route | `GET /v1/models/{model}` twice, `GET` without a route, no raw path | span names stay a small set |
| The transport injects its own span | CLIENT span under the caller; upstream sees the CLIENT span's id; request not modified | rows 4 and 5 of the worked example |
| The CLIENT span covers the streamed body | the CLIENT span is still open after the first event | a stream's duration is its body's |
| Connection errors are recorded | refused connection: Error and `error.type` | the failover attempt in [Routing, cascades, and failover](../../ai-platform-engineering/12-gateway/04-routing.md) is visible |
| gRPC unary propagation | CLIENT and SERVER `grpc.health.v1.Health/Check`, parentage, `rpc.*`, caller metadata kept | `EngineControl/Prefill` |
| gRPC errors mark both spans | NotFound: both spans Error, code 5, `error.type` `NotFound` | a prefill pool outage is red |
| gRPC stream propagation | `Watch` stream: SERVER under CLIENT | `tl.kv.v1` and heartbeat streams |
| The serving tree across HTTP and gRPC | five spans, one trace, the worked-example tree | the production system's end-to-end trace |
| `StartServer`'s attributes keep their types | int, bool, float, string slice, string, int64 keep their types | the gateway chain's attributes |
| Resource and ParentBased sampling | ratio 0 drops roots, keeps children of sampled callers; ratio 1 respects unsampled callers; resource | the sampling rules |
| Bad config is rejected | empty service name, ratio 1.5 or -0.1 rejected | config mistakes fail at startup |
| Export over OTLP/gRPC | a fake collector receives the span with `service.name` | the collector in the cluster |
| No endpoint shuts down at once | no endpoint: spans still recorded, `Shutdown` under 1 s | local runs and tests |
| Ending spans never waits for the collector | a collector that never answers adds no latency to 5 requests | a collector outage is not a gateway outage |
| Log lines carry trace and span ids | `trace_id` and `span_id` inside a span, also through `With`, none outside | the `kubectl logs \| grep` in the next section |

Most of these cases correspond to a specific mistake that is easy to make and has a recognizable symptom.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| The middleware never calls `Extract` | every request is a new trace; a traced client's tree stops at its own span | `Middleware` starts from `Extract(r.Context(), r.Header)` |
| The caller's context injected instead of the CLIENT span's, or the caller's request modified | the downstream server is a sibling of the client span; the caller's `http.Request` grows a header | `RoundTrip` starts the CLIENT span, clones the request with its context, and injects into the clone |
| The wrapper hides `Flush` or has no `Unwrap` | SSE arrives in one lump at the end; `SetWriteDeadline` returns `ErrNotSupported` | `statusRecorder` implements both and forwards them |
| Raw paths or `/`-prefixed method names in span names | one span name per model; gRPC names `/tl.engine.v1...` that match nothing in semconv | names come from the `route` callback's pattern; `rpcAttrs` trims the leading `/` |
| Error status on the wrong codes, or none | 404s page people, or 503s look healthy; failed RPCs and refused connections are green | `code >= 500` only on HTTP; `endRPC` for any non-OK gRPC code; `RoundTrip` marks transport errors with `errorType` |
| A ratio sampler without ParentBased | traces with random holes in the middle | `Setup` wraps the ratio sampler in `ParentBased` |
| `WithSyncer` (inline export) | every request waits for the collector; a slow collector is a slow gateway | `Setup` uses `WithBatcher` |
| gRPC metadata not extracted, not injected, or replaced | prefill spans start new traces; the caller's `x-request-id` disappears | `serverContext` extracts; `clientContext` copies the outgoing metadata and injects into the copy |
| Flags always `01`, or upper-case hex accepted | downstream records what the caller dropped; ids that other tools reject | the W3C propagator serialises the real flags and rejects invalid headers |
| Every attribute turned into a string | status codes cannot be compared or aggregated | `Span.SetAttribute` switches on the value's type |
| Endpoint ignored, or an exporter without one | no spans in Tempo; or 5 s hangs at every shutdown | the exporter is built from `Endpoint` and only when it is set |
| `logger.With(...)` drops the wrapper, or the ids use OTel's `traceId` spelling | some log lines lose their ids; the grep in the next section finds nothing | `WithAttrs` and `WithGroup` re-wrap; the keys are `trace_id` and `span_id` |
| The CLIENT span ends when the headers arrive | a 10 s stream shows as 1 ms; the server span sticks out of its parent | `spanBody` ends the span at EOF, a read error, or `Close` |
| A thin resource, or no config validation | every service is `unknown_service`; a 1.5 ratio is accepted silently | `Setup` rejects an empty name and a ratio outside $[0, 1]$, and sets name, namespace, version, and extra attributes |

## Where this leads

This kit generalizes the hand-written export of [One trace from gateway to engine](01-one-trace.md), which took the same three steps of a hop by hand. The next section, [Metrics and log correlation](03-metrics-and-log-correlation.md), wires it into the gateway's entry point: `Setup`, the SERVER span through `StartServer`, and `LogHandler` for JSON logs with ids. That gives the production system one trace from curl through `gateway.route`, `kv.transfer`, and `engine.decode`.

Later, [The task protocol and workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/04-task-protocol-and-workers.md) in *AI Platform Engineering* starts activity spans from `ActivityTask.trace_context` via `ContextWithTraceparent`, and [Subprocess activities](../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md) passes `TRACEPARENT` to Python via `Traceparent`. [The agent loop](../../ai-platform-engineering/13-agent-sdk/03-agent-loop.md) records `agent.run`, `agent.llm_call`, and `agent.tool` spans. The Rust half lives in [Serving metrics and tracing](../../build-an-llm/11-serving/08-serving-metrics-and-tracing.md) in *Build Your Own LLM System*: `tracing-opentelemetry` in the engine joins the same trace from the `traceparent` that `Transport` and the interceptors send.

Each piece of the kit has a production equivalent. `Middleware` and `Transport` correspond to `otelhttp` in opentelemetry-go-contrib (`instrumentation/net/http/otelhttp`), which adds metrics from the same middleware, span name formatters, and filters. The four gRPC interceptors correspond to the `otelgrpc` stats handlers (`instrumentation/google.golang.org/grpc/otelgrpc`), which add message events and sizes, with one handler per connection instead of per call. Head sampling with `ParentBased` gives way to tail sampling in the Collector (`tailsamplingprocessor`), which keeps every slow or failed trace and samples the fast ones after the fact. `LogHandler` corresponds to the OTel log bridge (`bridges/otelslog`), which exports logs as OTLP with trace context into a log backend. And W3C alone grows into W3C plus baggage, which carries request-scoped key-values (tenant, experiment) on every hop.

### Further reading

- [OpenTelemetry Go](https://opentelemetry.io/docs/languages/go/) (free)
- [Sampling](https://opentelemetry.io/docs/concepts/sampling/) (free)
- [W3C Baggage](https://www.w3.org/TR/baggage/)
- Charity Majors, Liz Fong-Jones, and George Miranda, *Observability Engineering* (O'Reilly), chapters 6 and 7

## Key terms

- **span context**: the triple of trace id, span id, and trace flags (plus `tracestate`) that propagation carries between processes.
- **extract / start / inject**: the three steps of a hop: read the caller's span context, start a child span, and send the calling span's context onward.
- **low cardinality**: a naming rule that keeps the set of distinct span names small, using route patterns instead of raw paths.
- **`TraceIDRatioBased`**: a sampler that keeps a trace when a hash of its trace id falls below the ratio, giving every process the same answer.
- **`ParentBased`**: a sampler wrapper that applies the inner sampler only to roots and makes children copy the parent's sampled flag.
- **batch span processor**: an SDK component that queues ended spans and exports them in the background, so `span.End()` never waits for the collector.
- **status recorder**: a `ResponseWriter` wrapper that captures the status code while forwarding `Flush` and exposing `Unwrap`.

## Summary

- Every hop extracts the caller's `traceparent`, starts its own span as the child, and injects its own span id onward; the same tree holds across HTTP and gRPC in one request.
- An outgoing call is a CLIENT span that lasts until the response body is done; a streamed completion is a long CLIENT span, not a 1 ms one.
- A wrapper `ResponseWriter` must keep `http.Flusher` and `Unwrap`, or the SSE stream it wraps is buffered.
- The sampler is `ParentBased`: a ratio decides only for new traces, so a sampled request never loses its middle hops.
- Spans leave through a batch processor, so ending a span never waits for the collector; no endpoint means no exporter at all.
- Only server faults (HTTP 5xx, non-OK gRPC codes) mark a span as Error.

## Review questions

1. Why does `Transport` inject the CLIENT span's context and not the caller's?
   <details><summary>Answer</summary>The upstream server's parent must be the span that made the call. Injecting the caller's (SERVER) span id makes the upstream a sibling of the CLIENT span, so the network and queueing time of the call is no longer inside any parent-child interval, and the tree no longer matches the worked example.</details>
2. Why does the middleware mark 503 as Error but leave 404 unset?
   <details><summary>Answer</summary>Status Error means "this service failed". A 4xx is the client's mistake; marking it would inflate error ratios and page people for malformed requests. A 5xx is the server's failure and must be red.</details>
3. What goes wrong if `statusRecorder` has no `Flush` method?
   <details><summary>Answer</summary>Embedding `http.ResponseWriter` exposes only that interface's methods, so the handler's `w.(http.Flusher)` assertion fails, SSE events sit in the buffer until the handler returns, and TTFT through the gateway equals total latency.</details>
4. Why is the sampler `ParentBased(TraceIDRatioBased(r))` and not just `TraceIDRatioBased(r)`?
   <details><summary>Answer</summary>Different services may run with different ratios. Without ParentBased, each one re-decides, so a trace the gateway kept at $r = 1$ loses the engine spans an engine at $r = 0.1$ drops. ParentBased makes children copy the sampled flag, so a trace is all or nothing.</details>
5. Why does `Setup` build no exporter when `Endpoint` is empty, instead of using the SDK default?
   <details><summary>Answer</summary>The default targets `localhost:4317`; with nothing listening, every shutdown flush retries until the export timeout (5 s). Without an exporter, spans still get ids, so propagation and log correlation keep working, and `Shutdown` returns at once.</details>
