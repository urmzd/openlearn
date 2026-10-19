# One trace from gateway to engine

The tracer system is now two processes on a cluster: a gateway in front and an inference engine behind it. When a completion is slow or fails, the gateway's log says it proxied a request and the engine's log says it served one, and nothing says they were the same request, or which of the two spent the time. The `X-Request-Id` header that [The tracer gateway](../../ai-platform-engineering/12-gateway/01-streaming-proxy.md) adds lets an operator grep both logs, but only after already suspecting both. A trace answers the question directly: one tree per request, each node a timed piece of work in a named service.

Most of the pieces already exist. The engine records its own span, as built in [Your first endpoint](../../build-an-llm/11-serving/01-your-first-endpoint.md) in *Build Your Own LLM System*. The gateway records a span called `gateway.proxy`, hands it to its entry point, and drops it on the floor. This section ships that span to Jaeger, so that the first incident drill, [First drill: the engine crashloops](../05-incident-response-and-chaos/01-first-drill.md), and everything after it can be debugged from one picture. The deployment and its Jaeger instance come from [Tracer deploy](../../infrastructure/01-containers-kubernetes/01-tracer-deploy.md) in *Infrastructure*.

The solution is the span export in the gateway's entry point: [`go/cmd/gateway/otlp.go`](../../../../course/solutions/entry/go/cmd/gateway/otlp.go) (`newSpanExporter`, the `proxy.Config.OnSpan` hook, and `otlpRequest`), wired in [`main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go). It works together with the engine's own OTLP/HTTP JSON export, both locally and on the kind cluster. The rules it follows are the telemetry rules of [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `traceparent`), [W3C Trace Context](https://www.w3.org/TR/trace-context/), and [OTLP/HTTP](https://opentelemetry.io/docs/specs/otlp/#otlphttp).

> **Learning objectives**
> - Describe a trace as a tree of spans and name the fields that make the tree: trace id, span id, parent span id, kind, and interval.
> - Explain how `traceparent` propagation works at each hop, and diagnose the classic bug of forwarding the incoming header unchanged.
> - Write an OTLP/HTTP JSON export body by hand, with hex ids and decimal-string nanosecond times.
> - Build a span exporter that can never slow a request, even when the collector accepts connections and never answers.
> - Verify a trace in Jaeger by reading its operation names, references, and services.

## Spans and traces

A **span** records one operation. It carries a handful of fields, each with a fixed meaning and size:

| Field | Meaning | Size or type |
|---|---|---|
| trace id | which request tree it belongs to | 16 random bytes, written as 32 hex digits |
| span id | this span | 8 random bytes, 16 hex digits |
| parent span id | the span that caused this one; empty for the **root** | 8 bytes or empty |
| name | what the operation is: `gateway.proxy`, `POST /v1/completions` | string |
| kind | which side of a call: SERVER (handled an incoming request), CLIENT (made an outgoing one), INTERNAL | enum: 1 INTERNAL, 2 SERVER, 3 CLIENT |
| start, end | wall-clock times | Unix nanoseconds |
| attributes | key-value facts: status code, request id | strings, ints |

A **trace** is the set of spans that share one trace id. The parent links make it a tree: every span except the root names a parent in the same trace. One request through gateway and engine is therefore one tree of at least two spans. Two rules follow, and a correct solution satisfies both.

The first is **same trace, named parent**: the engine's span has the gateway's trace id, and the gateway's span id as its parent. The second is that **a child lies inside its parent**: the engine finishes serving before the gateway's proxy call finishes, so the engine span's interval sits inside the gateway span's interval, up to clock error. Here both processes run on one clock, so the error is small.

## Propagation with `traceparent`

Spans live in different processes, so the ids have to travel between them. W3C Trace Context defines a request header for that:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             |  |                                |                |
             |  trace id (32 hex)                parent id (16)   flags: 01 = sampled
             version 00
```

The **parent id** in the header is the span id of the sender's current span. Each hop does three things. First it **extracts**: if the incoming request has a valid `traceparent`, it keeps that trace id and remembers the parent id; otherwise it makes a new random trace id and has no parent. Then it **starts its own span** with a new random span id, whose parent is the extracted parent id. Finally it **injects**: on the outgoing request, it sends `traceparent` with the same trace id and *its own span id* as the parent id.

This is how a gateway continues a caller's trace when the request carries `traceparent`, and starts a new one only when it does not. The gateway's proxy library already does all three steps; the engine does the first two. The tree exists only because each hop sends its own span id onward: the gateway's CLIENT span `gateway.proxy` becomes the parent of the engine's SERVER span `POST /v1/completions`.

> **Warning:** The classic bug is to forward the incoming header unchanged in the inject step. The engine's span then names the caller as its parent, and the gateway's span floats beside the tree instead of inside it.

## Getting spans out with OTLP

Propagation builds the tree in principle; the spans still have to leave each process and land somewhere they can be joined. The **OpenTelemetry Protocol** (OTLP) is how they leave. Over HTTP it is one `POST` to `<endpoint>/v1/traces`, where `<endpoint>` is `OTEL_EXPORTER_OTLP_ENDPOINT`, for example `http://jaeger:4318`. Port 4318 is OTLP/HTTP; 4317 is OTLP/gRPC. The body is an `ExportTraceServiceRequest`, sent either as protobuf (`Content-Type: application/x-protobuf`, which is what the OTel SDKs send) or as JSON (`application/json`, which is what a hand-written exporter sends):

```json
{"resourceSpans": [{
  "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "forge-gateway"}}]},
  "scopeSpans": [{"scope": {"name": "tinyllm/gateway"}, "spans": [{
    "traceId": "4bf92f3577b34da6a3ce929d0e0e4736", "spanId": "b7ad6b7169203331",
    "parentSpanId": "00f067aa0ba902b7", "name": "gateway.proxy", "kind": 3,
    "startTimeUnixNano": "1759926000000000000", "endTimeUnixNano": "1759926000012400000",
    "attributes": [{"key": "http.response.status_code", "value": {"intValue": "200"}}]}]}]}]}
```

Three details of the JSON mapping trip people up. Ids are **lowercase hex** strings, not base64, which is what generic protobuf-to-JSON tools produce. 64-bit integers, both the times and `intValue`, are **decimal strings**. And the **resource**, which says who sent the spans (`service.name`, taken from `OTEL_SERVICE_NAME`), is separate from the spans themselves.

## Exporting without hurting requests

Knowing the wire format, the tempting implementation is to POST each span as soon as it finishes. That would put the collector on the request path, and telemetry must never decide whether a request succeeds or how long it takes. An exporter therefore queues finished spans in a bounded buffer and returns immediately; when the buffer is full it drops the span and counts the drop. It sends from a background goroutine (or thread), with a timeout on every POST. And it does nothing at all when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset. Export is best effort and off the request path, so that a dead collector never slows a request, even a collector that accepts connections and never answers.

A span is exported once it **ends**. The gateway's span ends when the last SSE byte is flushed, so its export never overlaps the stream it measured. Before reading the solution, it is worth predicting the exporter's shape from these three requirements; the code below follows from them almost line by line.

## Reading the trace in Jaeger

**Jaeger** receives OTLP on 4317 and 4318, stores spans (in memory for the all-in-one image, so a restart loses them), and serves a UI and a JSON API on 16686, reached on the cluster at NodePort 30686. `GET /api/traces/<trace id>` returns the trace with each span's `operationName`, its `references` (`refType: CHILD_OF`, with the `spanID` of the parent), and a `processID`, which maps to a `serviceName` in `processes`. Checking a trace on the cluster means reading exactly those fields.

## Following one request by hand

With the mechanics in place, follow one request's ids and times. A traced client calls the gateway with the header shown above. After its first mention, each id is written as its first 8 hex digits, and times are milliseconds after the request arrives.

The two hops send these headers:

| Hop | `traceparent` sent | Who chose the parent id |
|---|---|---|
| client to gateway | `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` | the client's span `00f067aa` |
| gateway to engine | `00-4bf92f3577b34da6a3ce929d0e0e4736-b7ad6b7169203331-01` | the gateway's new span `b7ad6b71` |

The sink then receives these spans:

| Service | Name | Kind | Trace | Span | Parent | Start | End |
|---|---|---|---|---|---|---|---|
| `forge-gateway` | `gateway.proxy` | 3 CLIENT | `4bf92f35` | `b7ad6b71` | `00f067aa` | 0.0 | 12.4 |
| `forge-engine` | `POST /v1/completions` | 2 SERVER | `4bf92f35` | `5fb1c0a2` | `b7ad6b71` | 0.6 | 11.9 |

Check the two rules by hand. The trace id is `4bf92f35` twice, and the engine's parent `b7ad6b71` is the gateway's span id: one tree, rooted (from the system's side) at the client's span `00f067aa`. For nesting, $0.0 \le 0.6$ and $11.9 \le 12.4$, so the engine's interval lies inside the gateway's. The 0.6 ms before is the gateway's key check and its connection to the engine; the 0.5 ms after is the last flush back to the client.

Write the gateway span's start and end as $[s_g, e_g]$ and the engine span's as $[s_e, e_e]$, in milliseconds here and in Unix nanoseconds on the wire. The nesting rule is

$$s_g \le s_e \quad\text{and}\quad e_e \le e_g,$$

and a check should allow some slack for separately taken timestamps; 50 ms on each side is ample.

On the wire, the gateway's export of its row is the JSON body shown earlier, with `startTimeUnixNano` the arrival time and `endTimeUnixNano` 12.4 ms (12 400 000 ns) later: `"1759926000000000000"` and `"1759926000012400000"`.

Now send the same request with no `traceparent`. The gateway makes a new trace id, its span has no parent (it is the root), and the engine's span still names `gateway.proxy` as its parent. Both cases, with and without a caller's header, must produce one tree.

Finally, work the failure. If the gateway forwarded the client's header unchanged, the engine row would read parent `00f067aa`. The trace id still matches, but no span from the system's services has id `00f067aa`, and `gateway.proxy` has no child: Jaeger draws two siblings under a missing parent. Listing the received spans with their ids shows it at once, as `parent=00f067aa` on the engine span.

## The hook the exporter plugs into

The work is all in the gateway's entry point. The proxy library calls `Config.OnSpan` with a `proxy.Span` (name, hex ids, parent, status, start, end) once each response is finished, so the exporter is a function of that shape:

```go
// go/cmd/gateway: one line in main, one new file next to it
api := &http.Server{Handler: proxy.NewProxy(proxy.Config{
    Upstream: *upstream, APIKey: key, OnSpan: newSpanExporter(), // nil when no endpoint is set
})}
```

The rules for that function are short. It reads `OTEL_EXPORTER_OTLP_ENDPOINT` at startup, and an empty or unset value means no export, expressed as a nil hook. It sends `POST <endpoint>/v1/traces` in OTLP/HTTP JSON (protobuf would also be acceptable), with `service.name` from `OTEL_SERVICE_NAME`. The span it sends is named `gateway.proxy`, has kind 3 (CLIENT), carries exactly the ids `proxy.Span` carries, omits the parent when it is empty, and gives start and end in Unix nanoseconds. On the request path the hook never blocks: a bounded queue, a background sender, a timeout per POST, and a drop when full.

The engine side is already done: it exports `POST /v1/completions` (kind 2) as the child of the incoming `traceparent`, as `spec/cli-roles.md` requires. On kind, the charts from [Tracer deploy](../../infrastructure/01-containers-kubernetes/01-tracer-deploy.md) set both variables; you rebuild and `kind load` the gateway image, then `helm upgrade` its release. Once that is done, the solution can be seen working like this:

```bash
curl -sN http://127.0.0.1:30080/v1/completions -H "Authorization: Bearer $TL_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model":"tracer","prompt":"Once","max_tokens":8,"stream":true}'
open http://127.0.0.1:30686           # Jaeger: service <system>-gateway, Find Traces
```

## Building the exporter step by step

The design follows from five observations. First, **the span already exists; only shipping is missing**. The proxy library records `gateway.proxy` with the right ids (it extracts, starts its own span, and injects its own span id) and hands it to `Config.OnSpan` when the response is finished. So the solution is an `OnSpan` function, not a change to propagation.

Second, **no endpoint means no exporter**. If `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, returning a nil hook means the proxy does no telemetry work at all.

Third, **the hook runs on the request path, so it may not wait**. It must hand the span off and return, and the only non-blocking hand-off with bounded memory is a buffered channel with a `select`/`default` that drops when full.

Fourth, **the network belongs to one background goroutine**, which drains the channel and POSTs each span with a client timeout. A collector that never answers then costs one stuck goroutine for at most the timeout, never a request.

Fifth, **the wire format is fixed by OTLP/HTTP JSON**: hex ids as given by `proxy.Span`, Unix-nanosecond times as decimal strings, kind 3, `parentSpanId` only when there is a parent, and `service.name` in the resource.

The exporter reads its configuration once at startup, then sets up a queue and one sender:

```go
const spanKindClient = 3 // the gateway's span covers its call to the engine

var droppedSpans atomic.Int64

func newSpanExporter() func(proxy.Span) {
	endpoint := strings.TrimRight(os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"), "/")
	if endpoint == "" {
		return nil // no endpoint: the proxy skips the hook entirely
	}
	service := os.Getenv("OTEL_SERVICE_NAME")
	if service == "" {
		service = "gateway"
	}
	url := endpoint + "/v1/traces"                    // OTLP/HTTP path, on the 4318 endpoint
	client := &http.Client{Timeout: 2 * time.Second} // a dead collector costs at most 2 s, off the request path
	queue := make(chan proxy.Span, 1024)              // bounded: memory cannot grow during an outage

	go func() { // the only goroutine that touches the network
		for s := range queue {
			body, err := json.Marshal(otlpRequest(service, s))
			if err != nil {
				continue
			}
			resp, err := client.Post(url, "application/json", bytes.NewReader(body))
			if err != nil {
				log.Printf("gateway: span export to %s failed: %v", url, err)
				continue // best effort: no retry, the next span may get through
			}
			resp.Body.Close()
			if resp.StatusCode/100 != 2 {
				log.Printf("gateway: span export to %s answered %d", url, resp.StatusCode) // e.g. 400 on a bad body
			}
		}
	}()

	return func(s proxy.Span) { // runs on the request goroutine
		select {
		case queue <- s: // room: hand off and return at once
		default:
			droppedSpans.Add(1) // telemetry is best effort; requests are not
		}
	}
}
```

`TrimRight(..., "/")` makes `http://jaeger:4318/` and `http://jaeger:4318` produce the same URL. Logging an export failure, rather than returning it, is deliberate: nothing on the request path could act on it.

The body is one `ExportTraceServiceRequest` per span, in the JSON mapping:

```go
func otlpRequest(service string, s proxy.Span) map[string]any {
	span := map[string]any{
		"traceId":           s.TraceID, // already 32 lowercase hex digits: never base64
		"spanId":            s.SpanID,
		"name":              s.Name,    // "gateway.proxy"
		"kind":              spanKindClient,
		"startTimeUnixNano": strconv.FormatInt(s.Start.UnixNano(), 10), // 64-bit ints are strings
		"endTimeUnixNano":   strconv.FormatInt(s.End.UnixNano(), 10),
		"attributes": []map[string]any{
			{"key": "http.response.status_code", "value": map[string]any{"intValue": strconv.Itoa(s.StatusCode)}},
			{"key": "tl.request_id", "value": map[string]any{"stringValue": s.RequestID}}, // joins trace and logs
		},
	}
	if s.ParentSpanID != "" { // omitted for a root span, never sent as ""
		span["parentSpanId"] = s.ParentSpanID
	}
	if s.StatusCode >= 500 {
		span["status"] = map[string]any{"code": 2} // STATUS_CODE_ERROR: failed requests stand out in Jaeger
	}
	return map[string]any{
		"resourceSpans": []map[string]any{{
			"resource": map[string]any{"attributes": []map[string]any{
				{"key": "service.name", "value": map[string]any{"stringValue": service}}, // who sent it
			}},
			"scopeSpans": []map[string]any{{
				"scope": map[string]any{"name": "tinyllm/gateway"},
				"spans": []map[string]any{span},
			}},
		}},
	}
}
```

The wiring is one line in `main`:

```go
Handler: tel.wrap(proxy.NewProxy(proxy.Config{Upstream: *upstream, APIKey: key, OnSpan: newSpanExporter()})),
```

The `proxy.Span` it receives ([`proxy.go`](../../../../course/solutions/history/gw.00/go/gateway/proxy/proxy.go)) carries the name, the 32-digit trace id, the 16-digit span id, the caller's span id (empty when the gateway started the trace), the request id, the status the client received, and the start and end times. The full solution is in [`otlp.go`](../../../../course/solutions/entry/go/cmd/gateway/otlp.go) and [`main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go).

### Why this design

The cost on the request path is one non-blocking channel send: $O(1)$, no allocation beyond the span value, no system call. Memory during an outage is bounded by 1,024 queued spans.

One POST per span is simple and wasteful; the SDK's batch span processor, adopted in the next section, sends many spans per request. At the tracer's traffic, a handful of requests per second, that cost does not matter, and having no third-party module keeps the tracer's build minimal.

The exporter drops rather than blocking or retrying. Blocking would make a collector outage a gateway outage; unbounded buffering would turn it into an out-of-memory crash; retries would add pressure to a collector that is already struggling. Counting drops in `droppedSpans` keeps the loss visible.

Finally, JSON instead of protobuf matches the engine's hand-written exporter, so both halves of the trace are produced the same way and can be read by eye.

## What can go wrong

A correct solution is checked in five ways. Locally, the engine and the gateway start with `OTEL_EXPORTER_OTLP_ENDPOINT` pointing at an OTLP sink that accepts JSON and protobuf. One streamed request must give a `gateway.proxy` span and a `POST /v1/completions` span whose parent is it, in one trace; every trace read later depends on that. The span kinds must be CLIENT and SERVER, the intervals valid, the engine span inside the gateway span, and the ids the right lengths, because trace viewers draw and time the tree from exactly these fields. With the caller's header from the worked example, `gateway.proxy` must land in trace `4bf92f35...` with parent `00f067aa0ba902b7` and the engine span as its child; traced clients, such as the agent built later and the drills' evidence, rely on this. With a collector that accepts and never answers, three requests must each take under 1.5 s while something is still exported, because a collector outage is not a gateway outage. And on the kind cluster, a request through the gateway's NodePort must appear in Jaeger as `gateway.proxy` (`<system>-gateway`) parent of `POST /v1/completions` (`<system>-engine`).

The common mistakes each have a recognizable symptom:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `OnSpan` never set, or set to a function that only logs | Jaeger shows the engine alone; the sink receives one span per request | `main` passes `OnSpan: newSpanExporter()`, which POSTs every span |
| The incoming `traceparent` forwarded unchanged | the engine's parent is the caller's span; `gateway.proxy` has no child | the proxy injects its own span id; the exporter ships ids unchanged |
| A new trace id for every request, ignoring the header | a traced client's trace stops at its own span | the proxy extracts the caller's trace id; `ParentSpanID` carries the caller's span |
| `http.Post` inside the hook, on the request goroutine | with the collector down, every request waits for the POST timeout | the hook only enqueues; one background goroutine POSTs with a 2 s timeout |
| Ids base64-encoded, or times as JSON numbers | the collector rejects the body (400) or stores garbage ids; nothing links | hex ids passed through as strings; times via `strconv.FormatInt` |
| Endpoint `http://jaeger:4317` or a path other than `/v1/traces` | connection errors in the gateway log; no spans | the URL is the endpoint plus `/v1/traces`; the charts point at 4318 |
| Both charts set the same `OTEL_SERVICE_NAME` | Jaeger shows one service with two operations | `service.name` comes from each service's own `OTEL_SERVICE_NAME` (`<system>-gateway`, `<system>-engine`) |

## Where this leads

This section stands on three earlier pieces. [Tracer deploy](../../infrastructure/01-containers-kubernetes/01-tracer-deploy.md) set the OTel variables in the charts and runs Jaeger listening on 4318 and answering on 30686. [The tracer gateway](../../ai-platform-engineering/12-gateway/01-streaming-proxy.md) in *AI Platform Engineering* records `gateway.proxy` and injects `traceparent` with its own span id. [Your first endpoint](../../build-an-llm/11-serving/01-your-first-endpoint.md) gives the engine's SERVER span and its hand-written OTLP/HTTP JSON export.

Going forward, in [First drill: the engine crashloops](../05-incident-response-and-chaos/01-first-drill.md) the trace of a failing request shows the 503 at the gateway with no engine child. [Tracing for the serving path](02-tracing-for-the-serving-path.md) replaces this hand-written export with the OpenTelemetry Go SDK in `go/otelx`, adds HTTP and gRPC middleware, the collector, and Tempo. [Observability for the control plane](06-control-plane-observability.md) carries the same propagation through durable workflows into Python subprocesses with `TRACEPARENT`.

Each simplification here has a production equivalent. The hand-written OTLP/HTTP JSON gives way to the OpenTelemetry SDKs (Go `sdk/trace`, Rust `opentelemetry`), which add batching span processors, samplers, resource detection, and retries with backoff. Exporting straight to Jaeger gives way to the OpenTelemetry Collector: one endpoint for every service, tail sampling, and fan-out to several backends. Jaeger in memory gives way to Tempo or Jaeger on object storage, for retention, scale, and trace-to-metrics. Sampling everything (`flags 01`) gives way to head and tail sampling, which keep slow and failed traces and drop most fast ones. And two bare spans per request grow into the GenAI semantic conventions (`gen_ai.*` attributes), which put model, token usage, and time to first token on the span.

### Further reading

- [OpenTelemetry traces concepts](https://opentelemetry.io/docs/concepts/signals/traces/) (free)
- [OpenTelemetry Go](https://opentelemetry.io/docs/languages/go/)
- [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/)
- [Jaeger documentation](https://www.jaegertracing.io/docs/latest/) (free)
- [Grafana Tempo](https://grafana.com/docs/tempo/latest/)
- [Sampling](https://opentelemetry.io/docs/concepts/sampling/)
- [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- Charity Majors, Liz Fong-Jones, and George Miranda, *Observability Engineering* (O'Reilly), chapters 6 and 7

## Key terms

- **span**: a record of one timed operation, with a trace id, span id, optional parent span id, name, kind, interval, and attributes.
- **trace**: the tree of spans that share one trace id, linked by parent span ids.
- **span kind**: which side of a call a span measures: SERVER, CLIENT, or INTERNAL.
- **`traceparent`**: the W3C Trace Context header carrying version, trace id, the sender's span id, and flags.
- **OTLP**: the OpenTelemetry Protocol for exporting telemetry, here as an HTTP POST to `/v1/traces`.
- **resource**: the part of an OTLP payload that identifies the sender, such as `service.name`.
- **best-effort export**: shipping telemetry through a bounded queue and background sender that drops rather than blocking requests.

## Summary

- A trace is a tree of spans sharing a 16-byte trace id; each span names its parent by an 8-byte span id, and one gateway-plus-engine request is a tree of at least two spans.
- The tree exists only if each hop injects its own span id in `traceparent`; a gateway continues a caller's trace when given a header and starts one otherwise.
- OTLP/HTTP JSON writes ids as lowercase hex, times as decimal strings of Unix nanoseconds, and goes to `<endpoint>/v1/traces`.
- Export is best effort and off the request path: a bounded queue, one background sender, a timeout per POST, and counted drops.
- In Jaeger, checking a trace means reading `operationName`, `references`, and the service each `processID` maps to.

## Review questions

1. Why does the hook use `select` with a `default` instead of a plain `queue <- s`?
   <details><summary>Answer</summary>
   A plain send blocks when the channel is full, which happens when the collector is slow or dead and the sender goroutine is stuck in a POST. The request goroutine would then wait on telemetry. With `default`, a full queue drops the span and the request continues.
   </details>
2. The collector accepts connections but never answers. Why do requests stay fast, and what is the worst the exporter does?
   <details><summary>Answer</summary>
   Requests only enqueue. The sender goroutine waits up to the 2 s client timeout per POST, so it drains at most one span every 2 s; the queue fills to 1,024 and later spans are dropped and counted. Nothing waits on it except that one goroutine.
   </details>
3. Why does the gateway's span have kind CLIENT and the engine's SERVER, for the same request?
   <details><summary>Answer</summary>
   Kind says which side of a call the span measures. `gateway.proxy` measures the gateway's outgoing call to the engine (client side); the engine's span measures handling the incoming request (server side). The pair CLIENT parent, SERVER child is what trace viewers use to draw the network hop.
   </details>
4. Why is `parentSpanId` left out, rather than sent as an empty string, when there is no caller?
   <details><summary>Answer</summary>
   An absent parent is how OTLP marks a root span. An empty string is a malformed id in the JSON mapping (ids are hex of fixed length), which a collector may reject or store as a span whose parent is missing.
   </details>
5. Where does the "same trace, named parent" rule actually get enforced, in this file or elsewhere?
   <details><summary>Answer</summary>
   Elsewhere: the proxy library extracts `traceparent`, starts the span with a new span id, and injects its own span id upstream, and the engine makes its span a child of the header. This file only ships the ids it is given, unchanged, which is why it must not re-encode them.
   </details>
