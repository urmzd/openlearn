# One trace: gateway to engine in Jaeger (tracer)

## Overview

| | |
|---|---|
| **Module** | `obs.00` · practice · ops, go · Pass 1 · 3 to 5 h |
| **Solution** | span export in the gateway's entry point: [`go/cmd/gateway/otlp.go`](../../../../course/solutions/entry/go/cmd/gateway/otlp.go) (`newSpanExporter`, the `proxy.Config.OnSpan` hook, and `otlpRequest`), wired in [`main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go); it works together with the engine's OTLP/HTTP JSON export (`L10.0`), locally and on the `dep.00` cluster |
| **Contract** | the telemetry rules of [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `traceparent`); [W3C Trace Context](https://www.w3.org/TR/trace-context/); [OTLP/HTTP](https://opentelemetry.io/docs/specs/otlp/#otlphttp) |
| **Needs** | `dep.00` the deployment and its Jaeger; `gw.00` the gateway that records `gateway.proxy`; `L10.0` the engine that records `POST /v1/completions` (reading) |
| **Used by** | `obs.01` replaces the hand-written export with the OpenTelemetry SDK in `go/otelx`; with it, the end of Pass 1 shows one request as one trace across gateway and engine |
| **Optional depth** | [Observability](README.md) (this topic), [OpenTelemetry traces concepts](https://opentelemetry.io/docs/concepts/signals/traces/) (free), [Jaeger](https://www.jaegertracing.io/docs/latest/) (free), *Observability Engineering*, ch. 6 and 7 |

## Key Takeaways

- A **trace** is a tree of **spans** that share one 16-byte trace id; each span names its parent by its 8-byte span id. One request through gateway and engine is one tree of (at least) two spans.
- The tree exists only if each hop sends **its own** span id onward in `traceparent`: the gateway's CLIENT span `gateway.proxy` is the parent of the engine's SERVER span `POST /v1/completions`.
- A gateway **continues** a caller's trace when the request carries `traceparent`, and starts a new one only when it does not.
- Export is **best effort and off the request path**: a bounded queue, a background sender, a timeout. A dead collector must never slow a request, even one that accepts connections and never answers.
- OTLP/HTTP JSON writes ids as **lowercase hex**, times as decimal strings of Unix nanoseconds, and goes to `<endpoint>/v1/traces`.

## How to read this chapter

Section 2 builds tracing from spans up: propagation, OTLP, export off the request path, and Jaeger. Section 3 follows one request's ids and times by hand. Section 4 fixes the hook and its rules, and section 5 walks through the solution's exporter and explains why each line is there. Try to predict the exporter's shape from section 2.4 before reading it. On kind (`dep.00`), after rebuilding and loading the gateway image and upgrading its release, the solution is seen working like this:

```bash
curl -sN http://127.0.0.1:30080/v1/completions -H "Authorization: Bearer $TL_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model":"tracer","prompt":"Once","max_tokens":8,"stream":true}'
open http://127.0.0.1:30686           # Jaeger: service <system>-gateway, Find Traces
```

---

## 1. Why now

The tracer is now two processes on a cluster. When a completion is slow or fails, the gateway's log says it proxied a request and the engine's log says it served one, and nothing says they were the same request, or which of the two spent the time. `X-Request-Id` (from `gw.00`) lets an operator grep both logs, but only after already suspecting both. A trace answers directly: one tree per request, each node a timed piece of work in a named service. The engine already records its span (`L10.0`), and the gateway records `gateway.proxy` but hands it to its entry point and drops it. This module ships that span to Jaeger, so `ops.00` and everything after it can be debugged from one picture.

## 2. Principles

### 2.1 Spans and traces

A **span** records one operation:

| Field | Meaning | Size or type |
|---|---|---|
| trace id | which request tree it belongs to | 16 random bytes, written as 32 hex digits |
| span id | this span | 8 random bytes, 16 hex digits |
| parent span id | the span that caused this one; empty for the **root** | 8 bytes or empty |
| name | what the operation is: `gateway.proxy`, `POST /v1/completions` | string |
| kind | which side of a call: SERVER (handled an incoming request), CLIENT (made an outgoing one), INTERNAL | enum: 1 INTERNAL, 2 SERVER, 3 CLIENT |
| start, end | wall-clock times | Unix nanoseconds |
| attributes | key-value facts: status code, request id | strings, ints |

A **trace** is the set of spans with one trace id. The parent links make it a tree: every span except the root names a parent in the same trace. Two rules follow, and a correct solution satisfies both:

1. **Same trace, named parent.** The engine's span has the gateway's trace id and the gateway's span id as its parent.
2. **A child lies inside its parent.** The engine finishes serving before the gateway's proxy call finishes, so the engine span's interval is inside the gateway span's (up to clock error; both run on one clock here).

### 2.2 Propagation: `traceparent`

Spans live in different processes, so the ids travel in a request header defined by W3C Trace Context:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
             |  |                                |                |
             |  trace id (32 hex)                parent id (16)   flags: 01 = sampled
             version 00
```

The **parent id** in the header is the span id of the sender's current span. A hop does three things:

1. **Extract.** If the incoming request has a valid `traceparent`, keep its trace id and remember its parent id; otherwise make a new random trace id and have no parent.
2. **Start its own span** with a new random span id, parent = the extracted parent id.
3. **Inject.** On the outgoing request, send `traceparent` with the same trace id and **its own span id** as the parent id.

The gateway's proxy library already does all three (`gw.00`); the engine does 1 and 2 (`L10.0`). The classic bug is to forward the incoming header unchanged in step 3: the engine's span then names the caller as its parent, and the gateway's span floats beside the tree instead of inside it.

### 2.3 OTLP: getting spans out

The **OpenTelemetry Protocol** (OTLP) is how spans leave a process. Over HTTP it is one `POST` to `<endpoint>/v1/traces`, where `<endpoint>` is `OTEL_EXPORTER_OTLP_ENDPOINT` (for example `http://jaeger:4318`; port 4318 is OTLP/HTTP, 4317 is OTLP/gRPC). The body is an `ExportTraceServiceRequest`, as protobuf (`Content-Type: application/x-protobuf`, what the OTel SDKs send) or as JSON (`application/json`, what a hand-written exporter sends):

```json
{"resourceSpans": [{
  "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "forge-gateway"}}]},
  "scopeSpans": [{"scope": {"name": "tinyllm/gateway"}, "spans": [{
    "traceId": "4bf92f3577b34da6a3ce929d0e0e4736", "spanId": "b7ad6b7169203331",
    "parentSpanId": "00f067aa0ba902b7", "name": "gateway.proxy", "kind": 3,
    "startTimeUnixNano": "1759926000000000000", "endTimeUnixNano": "1759926000012400000",
    "attributes": [{"key": "http.response.status_code", "value": {"intValue": "200"}}]}]}]}]}
```

Three details of the JSON mapping trip people up: ids are **hex** strings (not base64, which generic protobuf-to-JSON tools produce), 64-bit integers (times, `intValue`) are **decimal strings**, and the **resource** (who sent it: `service.name` from `OTEL_SERVICE_NAME`) is separate from the spans.

### 2.4 Export without hurting requests

Telemetry must never decide whether a request succeeds or how long it takes. An exporter therefore:

- **queues** finished spans in a bounded buffer and returns immediately; when the buffer is full it **drops** the span and counts the drop;
- **sends** from a background goroutine (or thread), with a **timeout** on every POST;
- does nothing at all when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset.

A span is exported once it **ends**: the gateway's span ends when the last SSE byte is flushed, so its export never overlaps the stream it measured.

### 2.5 Jaeger

**Jaeger** receives OTLP (4317, 4318), stores spans (in memory for all-in-one: a restart loses them), and serves a UI and a JSON API on 16686, reached at NodePort 30686 (`dep.00`). `GET /api/traces/<trace id>` returns the trace with each span's `operationName`, `references` (`refType: CHILD_OF`, `spanID` of the parent), and `processID`, which maps to a `serviceName` in `processes`. Checking a trace on the cluster means reading exactly those fields.

## 3. Worked example by hand

A traced client calls the gateway with the header from 2.2. Ids are written as their first 8 hex digits after the first mention; times are milliseconds after the request arrives.

**The headers.**

| Hop | `traceparent` sent | Who chose the parent id |
|---|---|---|
| client to gateway | `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` | the client's span `00f067aa` |
| gateway to engine | `00-4bf92f3577b34da6a3ce929d0e0e4736-b7ad6b7169203331-01` | the gateway's new span `b7ad6b71` |

**The spans** the sink receives:

| Service | Name | Kind | Trace | Span | Parent | Start | End |
|---|---|---|---|---|---|---|---|
| `forge-gateway` | `gateway.proxy` | 3 CLIENT | `4bf92f35` | `b7ad6b71` | `00f067aa` | 0.0 | 12.4 |
| `forge-engine` | `POST /v1/completions` | 2 SERVER | `4bf92f35` | `5fb1c0a2` | `b7ad6b71` | 0.6 | 11.9 |

Check the two rules by hand:

1. Same trace (`4bf92f35` twice) and the engine's parent `b7ad6b71` is the gateway's span id: one tree, rooted (from our side) at the client's span `00f067aa`.
2. $0.0 \le 0.6$ and $11.9 \le 12.4$: the engine's interval lies inside the gateway's. The 0.6 ms before is the gateway's key check and connection to the engine; the 0.5 ms after is the last flush back to the client.

| Symbol | Meaning | Type |
|---|---|---|
| $[s_g, e_g]$ | gateway span start and end | ms, then Unix ns on the wire |
| $[s_e, e_e]$ | engine span start and end | same |

The nesting rule is $s_g \le s_e$ and $e_e \le e_g$; a check should allow some slack (50 ms on each side is ample) for separately taken timestamps.

**On the wire.** The gateway's export of its row is the JSON body in 2.3, with `startTimeUnixNano` the arrival time and `endTimeUnixNano` 12.4 ms (12 400 000 ns) later: `"1759926000000000000"` and `"1759926000012400000"`.

**Without a caller.** Send the same request with no `traceparent`: the gateway makes a new trace id and its span has no parent (it is the root), and the engine's span still names `gateway.proxy` as parent. Both cases, with and without a caller's header, must produce one tree.

**The failure, worked.** If the gateway forwarded the client's header unchanged, the engine row would read parent `00f067aa`. Rule 1 still holds for the trace id, but no span in the trace has id `00f067aa` from the system's services, and `gateway.proxy` has no child: Jaeger draws two siblings under a missing parent. Listing the received spans with their ids shows it at once as `parent=00f067aa` on the engine span.

## 4. The interface

The work is in the gateway's entry point. The proxy library (`gw.00`) calls `Config.OnSpan` with a `proxy.Span` (name, hex ids, parent, status, start, end) once each response is finished:

```go
// go/cmd/gateway: one line in main, one new file next to it
api := &http.Server{Handler: proxy.NewProxy(proxy.Config{
    Upstream: *upstream, APIKey: key, OnSpan: newSpanExporter(), // nil when no endpoint is set
})}
```

| Requirement | Detail |
|---|---|
| endpoint | `OTEL_EXPORTER_OTLP_ENDPOINT` read at startup; empty or unset means no export (a nil hook) |
| wire | `POST <endpoint>/v1/traces`, OTLP/HTTP JSON (section 2.3) or protobuf; `service.name` from `OTEL_SERVICE_NAME` |
| span | name `gateway.proxy`, kind 3 (CLIENT), the ids `proxy.Span` carries, parent omitted when empty, start and end in Unix ns |
| request path | the hook never blocks: bounded queue, background sender, a timeout per POST, drop when full |
| engine | already exports `POST /v1/completions` (kind 2) as the child of the incoming `traceparent` (`L10.0`, `spec/cli-roles.md`) |
| on kind | the `dep.00` charts set both variables; rebuild and `kind load` the gateway image, then `helm upgrade` its release |

### What a correct solution must handle

The local cases start the engine and the gateway with `OTEL_EXPORTER_OTLP_ENDPOINT` pointing at an OTLP sink that accepts JSON and protobuf; the last case needs the `dep.00` cluster.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| One request is one trace | conformance | one streamed request gives `gateway.proxy` and a `POST /v1/completions` whose parent is it, in one trace | every trace read later |
| Span kinds and nesting | unit | CLIENT and SERVER kinds, valid intervals, the engine span inside the gateway span, id lengths | trace viewers draw and time the tree from these |
| The caller's trace is continued | conformance | with the section 2.2 header, `gateway.proxy` is in trace `4bf92f35...` with parent `00f067aa0ba902b7`, and the engine span is its child | traced clients: the agent in Pass 10, drill evidence |
| Export never blocks requests | fault | with a collector that accepts and never answers, three requests each take under 1.5 s, and something was exported | a collector outage is not a gateway outage |
| The trace is in Jaeger on kind | conformance | a request through the gateway's NodePort appears in Jaeger as `gateway.proxy` (`<system>-gateway`) parent of `POST /v1/completions` (`<system>-engine`) | the deployed version |

## 5. The solution, step by step

### 5.1 From first principles

1. **The span already exists; only shipping is missing.** The proxy library records `gateway.proxy` with the right ids (it extracts, starts its own span, and injects its own span id, section 2.2) and hands it to `Config.OnSpan` when the response is finished. So the solution is an `OnSpan` function, not a change to propagation.
2. **No endpoint, no exporter.** If `OTEL_EXPORTER_OTLP_ENDPOINT` is unset, returning a nil hook means the proxy does no telemetry work at all.
3. **The hook runs on the request path, so it may not wait.** It must hand the span off and return; the only non-blocking hand-off with bounded memory is a buffered channel with a `select`/`default` that drops when full.
4. **The network belongs to one background goroutine**, which drains the channel and POSTs each span with a client timeout. A collector that never answers then costs one stuck goroutine for at most the timeout, never a request.
5. **The wire format is fixed by OTLP/HTTP JSON** (section 2.3): hex ids as given by `proxy.Span`, Unix-nanosecond times as decimal strings, kind 3, `parentSpanId` only when there is a parent, `service.name` in the resource.

### 5.2 The code, annotated

**The exporter.** Configuration once at startup, then a queue and one sender.

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

**The body.** One `ExportTraceServiceRequest` per span, in the JSON mapping.

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

**The wiring**, one line in `main`:

```go
Handler: tel.wrap(proxy.NewProxy(proxy.Config{Upstream: *upstream, APIKey: key, OnSpan: newSpanExporter()})),
```

The `proxy.Span` it receives ([`proxy.go`](../../../../course/solutions/history/gw.00/go/gateway/proxy/proxy.go)) carries the name, the 32-digit trace id, the 16-digit span id, the caller's span id (empty when the gateway started the trace), the request id, the status the client received, and the start and end times.

Full solution: [`otlp.go`](../../../../course/solutions/entry/go/cmd/gateway/otlp.go), [`main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go).

### 5.3 Why this solution

- **Cost on the request path** is one non-blocking channel send: $O(1)$, no allocation beyond the span value, no system call. Memory during an outage is bounded by 1,024 queued spans.
- **One POST per span** is simple and wasteful; the SDK's batch span processor (Going further, and `obs.01`) sends many spans per request. At Pass 1 traffic, a handful of requests per second, that cost does not matter, and having no third-party module keeps the tracer's build minimal.
- **Drop instead of block or retry.** Blocking would make a collector outage a gateway outage; unbounded buffering would make it an out-of-memory crash; retries would add pressure to a collector that is already struggling. Counting drops in `droppedSpans` keeps the loss visible.
- **JSON instead of protobuf** matches the engine's hand-written exporter (`L10.0`), so both halves of the trace are produced the same way and can be read by eye.

### 5.4 Explain it back

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
   Elsewhere: the proxy library extracts `traceparent`, starts the span with a new span id, and injects its own span id upstream (`gw.00`), and the engine makes its span a child of the header (`L10.0`). This file only ships the ids it is given, unchanged, which is why it must not re-encode them.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `OnSpan` never set, or set to a function that only logs | Jaeger shows the engine alone; the sink receives one span per request | `main` passes `OnSpan: newSpanExporter()`, which POSTs every span |
| 2. The incoming `traceparent` forwarded unchanged | the engine's parent is the caller's span; `gateway.proxy` has no child | the proxy injects its own span id (`gw.00`); the exporter ships ids unchanged |
| 3. A new trace id for every request, ignoring the header | a traced client's trace stops at its own span | the proxy extracts the caller's trace id; `ParentSpanID` carries the caller's span |
| 4. `http.Post` inside the hook, on the request goroutine | with the collector down, every request waits for the POST timeout | the hook only enqueues; one background goroutine POSTs with a 2 s timeout |
| 5. Ids base64-encoded, or times as JSON numbers | the collector rejects the body (400) or stores garbage ids; nothing links | hex ids passed through as strings; times via `strconv.FormatInt` |
| 6. Endpoint `http://jaeger:4317` or a path other than `/v1/traces` | connection errors in the gateway log; no spans | the URL is the endpoint plus `/v1/traces`; the charts point at 4318 |
| 7. Both charts set the same `OTEL_SERVICE_NAME` | Jaeger shows one service with two operations | `service.name` comes from each service's own `OTEL_SERVICE_NAME` (`<system>-gateway`, `<system>-engine`) |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.00` | the charts set the OTel variables; Jaeger listens on 4318 and answers on 30686 |
| Back | `gw.00` | records `gateway.proxy` and injects `traceparent` with its own span id |
| Back | `L10.0` | the engine's SERVER span and its hand-written OTLP/HTTP JSON export |
| Forward | `ops.00` | the trace of a failing request shows the 503 at the gateway with no engine child |
| Forward | `obs.01` | `go/otelx`: the OpenTelemetry Go SDK, HTTP and gRPC middleware, the collector, Tempo |
| Forward | `obs.05` | the same propagation through durable workflows into Python subprocesses (`TRACEPARENT`) |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hand-written OTLP/HTTP JSON | the OpenTelemetry SDKs (Go `sdk/trace`, Rust `opentelemetry`) | batching span processors, samplers, resource detection, retries with backoff | [OTel Go](https://opentelemetry.io/docs/languages/go/) |
| export straight to Jaeger | the OpenTelemetry Collector | one endpoint for every service, tail sampling, fan-out to several backends | [Collector](https://opentelemetry.io/docs/collector/) |
| Jaeger in memory | Tempo or Jaeger on object storage | retention, scale, trace-to-metrics | [Grafana Tempo](https://grafana.com/docs/tempo/latest/) |
| sample everything (`flags 01`) | head and tail sampling | keep slow and failed traces, drop most fast ones | [Sampling](https://opentelemetry.io/docs/concepts/sampling/) |
| two spans per request | GenAI semantic conventions (`gen_ai.*` attributes, token counts) | model, token usage, and time to first token on the span | [GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |
