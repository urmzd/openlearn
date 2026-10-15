<!-- ol:module gw.04 -->
# SSE streaming proxy (upgrades gw.00)

## Overview

| | |
|---|---|
| **Module** | `gw.04` · build · Go · Pass 7 · 4 h |
| **You build** | `go/gateway/proxy/proxy.go`, which you wrote in `gw.00` and now extend: `StreamProxy`, `StreamObserver`, `ExchangeObserver`, `Observers`, `Forward`, `Outbound`, `NewOutbound`, `UpstreamError`, and `Handler` (the chain's proxy stage), keeping `NewProxy` and everything `gw.00` proved |
| **Contract** | Streaming, Disconnect, and Errors rules of [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) (gateway tier); `NewProxy` still serves [`course/contracts/openapi/openai-subset.v0.yaml`](../../../../course/contracts/openapi/openai-subset.v0.yaml) |
| **Tests** | `course/tests/go/gw_04/` (what they check: section 4), with the `gw.00` suite as smoke regression · the bench `BenchmarkFirstByteOverhead` (`ol bench gw.04 --assert`) |
| **Needs** | [`gw.01` server skeleton](01-server-skeleton.md) (`server.Usage`, the `Exchange`) · reading: [`gw.00` tracer gateway](00-streaming-proxy.md), `L10.5` the engine's v1 streams, [SSE in `ai-platform-engineering/03`](../03-streaming-sse/) |
| **Used by** | `gw.05` forwards every routing attempt through `Forward` and relies on its commitment boundary · inherited from `gw.00`: `dep.00` builds it into the gateway image, `obs.00` traces it |
| **Milestone** | MS-gateway |
| **Optional depth** | WHATWG HTML, [server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free); RFC 9110 section 9.3 on request semantics and retries (free) |

## Key Takeaways

- The proxy still forwards **bytes**: every read from the engine is written and flushed before the next read, so the client's stream is byte-identical to the engine's whatever the read sizes (`TestRecordedStreamsByteExact`, `TestStreamsEventByEvent`).
- While the bytes pass, a **splitter** cuts them into SSE events at blank lines for an observer: the first byte (time to first token), every event, and the usage from the last usage chunk, delivered once at a clean end (`TestHandExample`).
- `Forward` draws the **commitment boundary**: before the engine answers, a failure writes nothing and is reported uncommitted, so routing may try another worker; after the first byte nothing is retried and a broken stream ends with an SSE error event (`TestForwardUncommittedOnRefused`, `TestChaosResetMidStream`).
- The client's context is the upstream request's context: when the client leaves, the engine's request is cancelled (`TestClientDisconnectCancelsUpstream`).
- `Handler` is the chain's proxy stage: no static key (authn already ran), the gateway's `X-Request-Id` and no duplicate from the engine, the first byte and usage recorded on the `Exchange` for the limiter and the meter (`TestHandlerInChain`).

## How to work this chapter

```bash
ol start gw.04          # proxy.go exists (gw.00): prints the contract diff instead of overwriting
ol tests gw.04          # read the test catalog first
ol check gw.04          # gw.00's smoke tests, then gw.04's tests
ol bench gw.04 --assert # local only: first-byte overhead within 5 ms
ol diff  gw.04          # after passing: your code against the reference
```

`gw.04` takes over a file you own (`upgrades`): edit your own `proxy.go` in place. Your `gw.00` verdict is frozen as passed, and its tests now run as `gw.04`'s smoke regression, so the tracer behavior must survive the upgrade. In your composition root, the `Proxy` slot becomes `proxy.Handler(proxy.Config{Upstream: ...})`.

---

## 1. Why now

The `gw.00` proxy copies bytes and knows nothing about them. Pass 7 needs to know three things about every stream as it passes: when its first byte reached the client (time to first token, the SLO of `obs.03`), how many tokens it used (the limiter settles with it in `gw.03`, the ledger records it in `gw.07`), and whether it may still be retried elsewhere (failover in `gw.05`). Parsing and re-emitting events would be the easy way to learn those things and the wrong one: it delays tokens, it can change bytes (whitespace, key order, a character split across chunks), and it couples the gateway to every field of the engine's JSON. This module watches the stream without touching it.

## 2. Principles

### 2.1 Streams and their two sides

An engine's streamed answer (`openai-subset.v1.yaml`) is `Content-Type: text/event-stream` and a sequence of **events**, each `data: <json>\n\n`: the first chunk carries `delta.role`, content chunks follow (some with empty `content` while a multi-byte character is incomplete), the last content chunk carries `finish_reason`, an optional chunk with `choices: []` carries `usage`, and the stream ends with `data: [DONE]\n\n`. Lines starting with `:` are comments, such as `: ping\n\n` every 15 s while idle. A non-streamed answer is one JSON body with a `usage` object.

The proxy relays to the **client** side and reads from the **upstream** side. The bytes it reads arrive in arbitrary pieces: TCP and the engine's buffers decide the boundaries, not the event structure.

### 2.2 Relay first, observe on the side

`StreamProxy(w, r, up, o)` does, in order:

1. Copy the end-to-end headers of `up` to `w` (hop-by-hop headers dropped, as in `gw.00`), keeping the gateway's own `X-Request-Id`: `Header.Add` would append the engine's copy as a second value, and the client would see two ids.
2. Write the status.
3. Loop: read up to 32 KiB; if it is the first body byte, call `o.OnFirstByte()` **before** writing it (so the timestamp is not inflated by the write); write the bytes; flush; then feed the same bytes to the splitter.

The **splitter** appends each piece to a buffer and cuts complete events at `\n\n`. For each event it joins the payloads of its `data:` lines (one space after the colon removed), skips events with no `data:` line (comments), and calls `o.OnChunk(data)`. Because it keeps the remainder between reads, an event split across ten 1-byte reads is still delivered once and whole. If an event's JSON has a non-null `usage`, it becomes the stream's usage (the last one wins). For a JSON body the proxy keeps a copy (up to 4 MiB) and reads `usage` from it at the end. When the engine omits `total_tokens`, it is `prompt_tokens + completion_tokens`.

4. At a clean end (EOF), call `o.OnDone(usage)` once.

`ExchangeObserver(ex)` is the observer the chain uses by default: it records `MarkFirstByte` and `SetUsage` on the request's `Exchange`, which is how `gw.03` settles and `gw.07` meters without importing this package. `Observers{a, b}` fans out to several (`gw.06` adds a recorder for stream replay).

### 2.3 How long the first token waits

| Symbol | Meaning |
|---|---|
| $t_e$ | when the engine writes its first byte |
| $t_c$ | when the client receives it through the gateway |
| $\Delta = t_c - t_e$ | the gateway's added time to first byte |

With a flush per read, $\Delta$ is one socket read, one write, and one flush: well under a millisecond on one host. Without the flush (section 5 of `gw.00`), $\Delta$ is the time for 4 KiB of events to accumulate, about a second at 20 tokens per second. The contract budget is $\Delta \le 5$ ms; `BenchmarkFirstByteOverhead` measures it as the difference between direct and proxied time to first byte, and `ol bench gw.04 --assert` checks it. A latency bound is a bench, never part of `ol check`, because shared CI machines make timing noisy (DESIGN 4.0).

### 2.4 The commitment boundary

HTTP sends the status line before the body, so a proxy's options depend on what it has sent:

| When | What happened | `Forward` returns | Client sees |
|---|---|---|---|
| before any answer | refused, reset, DNS, timeout | `UpstreamError{Committed: false}`, nothing written | whatever the caller decides: `Handler` answers 503 `no_capacity`, `gw.05` tries another worker |
| the engine answered 503 and `Retry503` is set | draining or queue full: no work was done | `UpstreamError{Committed: false, StatusCode: 503}`, nothing written | the same |
| any other answer | 200, 400, 404 ... | relayed; status and body pass through | the engine's answer |
| after the first byte | the stream broke | `UpstreamError{Committed: true}` after one `data: {"error": ...}` event | the bytes so far, then an error event, never `[DONE]` |
| any time | the client left | the upstream request is cancelled (it runs on the client's context) | nothing |

Retrying after the first byte would repeat or splice tokens: the client already printed "The cat", and a second engine starts again from "The". So a committed failure is final. `Outbound` is the request as it will be sent (method, path, query, the `gw.00` header rules, the body as bytes), built once by `NewOutbound` so `gw.05` can send the same request to several workers in turn.

## 3. Worked example by hand

The engine answers with five events, 163, 147, 148, 156, and 14 bytes long (628 in total), and the gateway happens to read 40 bytes at a time (16 reads):

```
data: {"id":"c1",...,"delta":{"role":"assistant","content":""},...}\n\n
data: {"id":"c1",...,"delta":{"content":"Hel"},...}\n\n
data: {"id":"c1",...,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n
data: {"id":"c1",...,"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n
data: [DONE]\n\n
```

| Read | Bytes | What happens |
|---|---|---|
| 1 | 0 to 39 | `OnFirstByte()` (the client has 0 bytes), write, flush; the splitter holds 40 bytes, no `\n\n` yet |
| 5 | 160 to 199 | write, flush; the splitter finds the first `\n\n` (ending at byte 162): `OnChunk` with the role chunk; 37 bytes stay buffered |
| 8 | 280 to 319 | the `"Hel"` event completes (ends at byte 309): `OnChunk` |
| 12 | 440 to 479 | the `"lo"` event completes (ends at byte 457): `OnChunk` |
| 16 | 600 to 627 | the usage event (ends at 613) and `[DONE]` complete: two `OnChunk` calls; the usage event sets usage 5/2/7 |
| EOF | | `OnDone({5, 2, 7})` |

The client received exactly the 628 bytes, `OnFirstByte` was called once, `OnChunk` five times (the last with `[DONE]`), and `OnDone` once. This is `TestHandExample`.

## 4. The interface

```go
package proxy // import "tinyllm/gateway/proxy"

// gw.00, unchanged
func NewProxy(cfg Config) http.Handler
func ParseTraceparent(h string) (traceID, parentID, flags string, ok bool)

// gw.04
type Config struct { Upstream, APIKey string; Client *http.Client; OnSpan func(Span)
	Observer func(r *http.Request) StreamObserver } // new: per-request observer
func Handler(cfg Config) http.Handler // the chain's proxy stage: no key check

type StreamObserver interface { OnFirstByte(); OnChunk(data []byte); OnDone(u server.Usage) }
func ExchangeObserver(ex *server.Exchange) StreamObserver
type Observers []StreamObserver

func StreamProxy(w http.ResponseWriter, r *http.Request, up *http.Response, o StreamObserver) error

type Outbound struct { Method, Path, RawQuery string; Header http.Header; Body []byte; Retry503 bool }
func NewOutbound(r *http.Request, body []byte) (*Outbound, Span)
type UpstreamError struct { Committed bool; StatusCode int; Err error }
func Forward(ctx context.Context, w http.ResponseWriter, base string, out *Outbound, client *http.Client, o StreamObserver) error
```

Use only the standard library and `tinyllm/gateway/server`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExample` | unit | section 3: bytes, one `OnFirstByte` before any write, five events, usage 5/2/7, status and content type | you and the tests agree on what is observed |
| `TestRecordedStreamsByteExact` | conformance | five fixture streams (chat, UTF-8 split, ping without usage, tool-call fragments, text completions) at read sizes 1, 7, 64, 65536: bytes equal, event counts, usage | conformance `chat.stream.framing` through the gateway |
| `TestPingCommentsPassThrough` | unit | `: ping` reaches the client and is not an event | idle streams stay open without fake events |
| `TestNonStreamUsage` | unit | a JSON body's usage, `total_tokens` computed, no events | the limiter and the ledger for non-streamed calls |
| `TestForwardUncommittedOnRefused` | fault | a dead engine: uncommitted error, nothing written | `gw.05` can fail over |
| `TestForwardRetry503` | unit | with `Retry503` a 503 is handed back unwritten; without it, relayed | draining engines are skipped, not shown to clients |
| `TestNewOutboundDropsKey` | unit | no `Authorization` or hop-by-hop headers; request id, path, query, trace kept | the engine tier stays key-free |
| `TestStreamsEventByEvent` | unit | through real sockets, event 1 arrives while the engine holds event 2 | tokens stream |
| `TestHandlerInChain` | unit | in the chain: no key needed, one `X-Request-Id` (the gateway's), first byte, usage, and status on the `Exchange` | `gw.03` settles and `gw.07` meters from it |
| `TestHandlerUpstreamDown503` | boundary | a dead engine before the first byte: 503 `no_capacity` | the `gw.00` answer, kept |
| `TestChaosResetMidStream` | fault | the testkit chaos proxy resets the engine connection after the first event: the client gets an SSE error event and no `[DONE]`; no `OnDone` | no silent truncation; no metering of a broken stream |
| `TestClientDisconnectCancelsUpstream` | fault | a client that leaves cancels the engine's request (contract: 100 ms; checked here within 1 s) | the engine stops generating for nobody |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. no flush after each write | tokens arrive in bursts; with a fake engine that waits, nothing arrives | `TestStreamsEventByEvent` (mutant `s01`) |
| 2. `OnFirstByte` after the write, or on every read | TTFT includes the write; the metric fires dozens of times per stream | `TestHandExample` (mutant `s02`) |
| 3. usage from the wrong place: never from the SSE usage chunk, never from a JSON body, `total_tokens` left 0 | the limiter refunds everything; the ledger records zero tokens | `TestHandExample`, `TestNonStreamUsage` (mutants `s03`, `s11`, `s13`) |
| 4. cutting events per read, or reporting comments as events | events split across reads vanish; pings count as tokens | `TestRecordedStreamsByteExact`, `TestPingCommentsPassThrough` (mutants `s04`, `s05`) |
| 5. a broken stream that ends quietly and reports done | the client shows a short answer as complete; the ledger meters it | `TestChaosResetMidStream` (mutant `s08`) |
| 6. `Forward` that answers the client itself on failure, or ignores `Retry503` | failover is impossible: the 503 is already sent | `TestForwardUncommittedOnRefused`, `TestForwardRetry503` (mutants `s06`, `s07`) |
| 7. the upstream request on `context.Background()` | a client that hangs up leaves the engine generating | `TestClientDisconnectCancelsUpstream` (mutant `s09`) |
| 8. `Header.Add` for every upstream header | the client gets two `X-Request-Id` values | `TestHandlerInChain` (mutant `s10`) |
| 9. the chain's handler still checking the static key | every request is a 401 once real keys replace `TL_API_KEY` | `TestHandlerInChain` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | `server.Usage`, the `Exchange` the default observer records on, the `Proxy` slot |
| Back | `gw.00` | the request path you upgrade; its suite is now your smoke regression |
| Forward | `gw.05` | sends each routing attempt with `Forward` and retries only uncommitted failures |
| Forward | `dep.00` | the gateway image now serves `Handler` in the chain (inherited call site) |
| Forward | `obs.00` | the `gateway.proxy` span is unchanged (inherited call site) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Forward` and the commitment boundary | saige's retry provider | buffer until the first content delta, then never retry (`drainUntilContentOrError`) | saige `provider/retry` |
| `StreamProxy` | `httputil.ReverseProxy` with `FlushInterval: -1` | the standard library's streaming proxy, hooks for rewriting | [`httputil.ReverseProxy`](https://pkg.go.dev/net/http/httputil#ReverseProxy) (free) |
| the observer | Envoy AI Gateway token usage extraction | usage read from provider streams for metering and limits | [Envoy AI Gateway](https://aigateway.envoyproxy.io/) (free) |
| disconnect handling | vLLM's abort on client disconnect | the engine side of the same contract | [vLLM](https://github.com/vllm-project/vllm) (free) |
