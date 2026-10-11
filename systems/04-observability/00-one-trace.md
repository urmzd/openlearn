<!-- ol:module obs.00 -->
# One trace: gateway to engine in Jaeger (tracer)

## Overview

| | |
|---|---|
| **Module** | `obs.00` · practice · ops, go · Pass 1 · 3 to 5 h |
| **You build** | span export in your gateway's entry point (`go/cmd/gateway`: hook `proxy.Config.OnSpan`), checked together with your engine's OTLP/HTTP JSON export (`L10.0`), locally and on the `dep.00` cluster |
| **Contract** | the telemetry rules of [`spec/cli-roles.md`](../../course/contracts/spec/cli-roles.md) (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`, `traceparent`); [W3C Trace Context](https://www.w3.org/TR/trace-context/); [OTLP/HTTP](https://opentelemetry.io/docs/specs/otlp/#otlphttp) |
| **Tests** | `course/tests/obs.00/` (`check` runs `artifacts.py` against the `_otlpsink.py` receiver; what each test checks: section 4) |
| **Needs** | `dep.00` the deployment and its Jaeger; `gw.00` the gateway that records `gateway.proxy`; `L10.0` the engine that records `POST /v1/completions` (reading) |
| **Used by** | `obs.01` replaces the hand-written export with the OpenTelemetry SDK in `go/otelx` |
| **Milestone** | MS-P1 (step `one-trace-spans-gateway-and-engine`) |
| **Optional depth** | [Observability](README.md) (this topic), [OpenTelemetry traces concepts](https://opentelemetry.io/docs/concepts/signals/traces/) (free), [Jaeger](https://www.jaegertracing.io/docs/latest/) (free), *Observability Engineering*, ch. 6 and 7 |

## Key Takeaways

- A **trace** is a tree of **spans** that share one 16-byte trace id; each span names its parent by its 8-byte span id. One request through gateway and engine is one tree of (at least) two spans.
- The tree exists only if each hop sends **its own** span id onward in `traceparent`: the gateway's CLIENT span `gateway.proxy` is the parent of the engine's SERVER span `POST /v1/completions`.
- A gateway **continues** a caller's trace when the request carries `traceparent`, and starts a new one only when it does not.
- Export is **best effort and off the request path**: a bounded queue, a background sender, a timeout. A dead collector must never slow a request; the tests prove it with a collector that never answers.
- OTLP/HTTP JSON writes ids as **lowercase hex**, times as decimal strings of Unix nanoseconds, and goes to `<endpoint>/v1/traces`.

## How to work this chapter

```bash
ol start obs.00                       # records the start; there are no stubs
ol tests obs.00                       # read the test catalog first
# wire the export into go/cmd/gateway (section 4), rebuild, then:
ol check obs.00                       # builds, starts both servers, sends requests to a course OTLP sink
OL_SMOKE=1 ol check obs.00            # without a cluster: the local tier only
# on kind (dep.00): rebuild and load the gateway image, upgrade its release, then
curl -sN http://127.0.0.1:30080/v1/completions -H "Authorization: Bearer $TL_API_KEY" \
  -H 'Content-Type: application/json' -d '{"model":"tracer","prompt":"Once","max_tokens":8,"stream":true}'
open http://127.0.0.1:30686           # Jaeger: service <system>-gateway, Find Traces
```

---

## 1. Why now

Your tracer is now two processes on a cluster. When a completion is slow or fails, the gateway's log says it proxied a request and the engine's log says it served one, and nothing says they were the same request, or which of the two spent the time. `X-Request-Id` (from `gw.00`) lets you grep both logs, but only after you already suspect both. A trace answers directly: one tree per request, each node a timed piece of work in a named service. The engine already records its span (`L10.0`), and the gateway records `gateway.proxy` but hands it to its entry point and drops it. This module ships that span to Jaeger, so `ops.00` and everything after it can be debugged from one picture.

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

A **trace** is the set of spans with one trace id. The parent links make it a tree: every span except the root names a parent in the same trace. Two rules follow, and the tests check both:

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

Your gateway's proxy library already does all three (`gw.00`); your engine does 1 and 2 (`L10.0`). The classic bug is to forward the incoming header unchanged in step 3: the engine's span then names the caller as its parent, and the gateway's span floats beside the tree instead of inside it.

### 2.3 OTLP: getting spans out

The **OpenTelemetry Protocol** (OTLP) is how spans leave a process. Over HTTP it is one `POST` to `<endpoint>/v1/traces`, where `<endpoint>` is `OTEL_EXPORTER_OTLP_ENDPOINT` (for example `http://jaeger:4318`; port 4318 is OTLP/HTTP, 4317 is OTLP/gRPC). The body is an `ExportTraceServiceRequest`, as protobuf (`Content-Type: application/x-protobuf`, what the OTel SDKs send) or as JSON (`application/json`, what you write by hand):

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

**Jaeger** receives OTLP (4317, 4318), stores spans (in memory for all-in-one: a restart loses them), and serves a UI and a JSON API on 16686, reached at NodePort 30686 (`dep.00`). `GET /api/traces/<trace id>` returns the trace with each span's `operationName`, `references` (`refType: CHILD_OF`, `spanID` of the parent), and `processID`, which maps to a `serviceName` in `processes`. The MS-P1 trace step and the cluster test read exactly those fields.

## 3. Worked example by hand

A traced client calls your gateway with the header from 2.2. Ids are written as their first 8 hex digits after the first mention; times are milliseconds after the request arrives.

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

The nesting rule is $s_g \le s_e$ and $e_e \le e_g$; the test allows 50 ms of slack on each side for separate timestamps.

**On the wire.** The gateway's export of its row is the JSON body in 2.3, with `startTimeUnixNano` the arrival time and `endTimeUnixNano` 12.4 ms (12 400 000 ns) later: `"1759926000000000000"` and `"1759926000012400000"`.

**Without a caller.** Send the same request with no `traceparent`: the gateway makes a new trace id and its span has no parent (it is the root), and the engine's span still names `gateway.proxy` as parent. That is the case `test_one_request_is_one_trace` checks; the header case is `test_caller_trace_is_continued`.

**The failure, worked.** If the gateway forwarded the client's header unchanged, the engine row would read parent `00f067aa`. Rule 1 still holds for the trace id, but no span in the trace has id `00f067aa` from your services, and `gateway.proxy` has no child: Jaeger draws two siblings under a missing parent. The test reports the spans it received with their ids, so this shows up as `parent=00f067aa` on the engine span.

## 4. The interface

The work is in your gateway's entry point. The proxy library (`gw.00`) calls `Config.OnSpan` with a `proxy.Span` (name, hex ids, parent, status, start, end) once each response is finished:

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

### What the tests check

`ol check obs.00` runs `course/tests/obs.00/check` in your repo. The local tier runs your `[build]` steps, trains a throwaway model with `[entry].tinyllm`, and starts `[services.engine]` and `[services.gateway]` from `system.toml` exactly as `ol milestone` does (free ports, `{model_dir}` filled), with `OTEL_EXPORTER_OTLP_ENDPOINT` set to a course OTLP sink that accepts JSON and protobuf.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_one_request_is_one_trace` | conformance | one streamed request gives `gateway.proxy` and a `POST /v1/completions` whose parent is it, in one trace | the MS-P1 trace step, every trace you read later |
| `test_span_kinds_and_nesting` | unit | CLIENT and SERVER kinds, valid intervals, the engine span inside the gateway span, id lengths | trace viewers draw and time the tree from these |
| `test_caller_trace_is_continued` | conformance | with the section 2.2 header, `gateway.proxy` is in trace `4bf92f35...` with parent `00f067aa0ba902b7`, and the engine span is its child | traced clients: the agent in Pass 10, `ol drill` evidence |
| `test_export_never_blocks_requests` | fault | with a collector that accepts and never answers, three requests each take under 1.5 s, and something was exported | a collector outage is not a gateway outage |
| `test_trace_in_jaeger_on_kind` | conformance | a request through `gateway_url` appears in Jaeger at `traces` as `gateway.proxy` (`<system>-gateway`) parent of `POST /v1/completions` (`<system>-engine`) | the deployed version, read the way MS-P1 reads it |

The last test is the **cluster tier**: without your `kube_context` it fails and says so; `OL_SMOKE=1 ol check obs.00` skips it with the reason.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `OnSpan` never set, or set to a function that only logs | Jaeger shows the engine alone; the sink receives one span per request | `test_one_request_is_one_trace` |
| 2. The incoming `traceparent` forwarded unchanged | the engine's parent is the caller's span; `gateway.proxy` has no child | `test_one_request_is_one_trace`, `test_caller_trace_is_continued` |
| 3. A new trace id for every request, ignoring the header | a traced client's trace stops at its own span | `test_caller_trace_is_continued` |
| 4. `http.Post` inside the hook, on the request goroutine | with the collector down, every request waits for the POST timeout | `test_export_never_blocks_requests` |
| 5. Ids base64-encoded, or times as JSON numbers | the collector rejects the body (400) or stores garbage ids; nothing links | `test_one_request_is_one_trace` (the sink reports the rejection), `test_span_kinds_and_nesting` |
| 6. Endpoint `http://jaeger:4317` or a path other than `/v1/traces` | connection errors in the gateway log; no spans | `test_trace_in_jaeger_on_kind` |
| 7. Both charts set the same `OTEL_SERVICE_NAME` | Jaeger shows one service with two operations, and MS-P1's `services` check fails | `test_trace_in_jaeger_on_kind` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dep.00` | the charts set the OTel variables; Jaeger listens on 4318 and answers on 30686 |
| Back | `gw.00` | records `gateway.proxy` and injects `traceparent` with its own span id |
| Back | `L10.0` | the engine's SERVER span and its hand-written OTLP/HTTP JSON export |
| Forward | `ops.00` | the trace of a failing request shows the 503 at the gateway with no engine child |
| Forward | `obs.01` | `go/otelx`: the OpenTelemetry Go SDK, HTTP and gRPC middleware, the collector, Tempo |
| Forward | `obs.05` | the same propagation through durable workflows into Python subprocesses (`TRACEPARENT`) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hand-written OTLP/HTTP JSON | the OpenTelemetry SDKs (Go `sdk/trace`, Rust `opentelemetry`) | batching span processors, samplers, resource detection, retries with backoff | [OTel Go](https://opentelemetry.io/docs/languages/go/) |
| export straight to Jaeger | the OpenTelemetry Collector | one endpoint for every service, tail sampling, fan-out to several backends | [Collector](https://opentelemetry.io/docs/collector/) |
| Jaeger in memory | Tempo or Jaeger on object storage | retention, scale, trace-to-metrics | [Grafana Tempo](https://grafana.com/docs/tempo/latest/) |
| sample everything (`flags 01`) | head and tail sampling | keep slow and failed traces, drop most fast ones | [Sampling](https://opentelemetry.io/docs/concepts/sampling/) |
| two spans per request | GenAI semantic conventions (`gen_ai.*` attributes, token counts) | model, token usage, and time to first token on the span | [GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |
