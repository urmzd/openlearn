# The SSE streaming proxy

The [tracer gateway](00-streaming-proxy.md) copies bytes and knows nothing about them. A real gateway needs to know three things about every stream as it passes. It needs to know when the first byte reached the client, which is time to first token, the quantity behind the SLO in [SLOs and multi-window burn-rate alerts](../../systems/04-observability/03-slos-and-burn-rate-alerts.md) (in *Systems and Architecture*). It needs to know how many tokens the stream used, because the limiter in [Rate limiting](../08-authorization-and-access-control/02-rate-limiting.md) settles with that number and the ledger in [The usage ledger and admin API](07-usage-ledger-and-admin-api.md) records it. And it needs to know whether the request may still be retried elsewhere, for failover in [Routing, cascades, and failover](05-routing.md).

Parsing and re-emitting events would be the easy way to learn those things, and the wrong one. It delays tokens, it can change bytes (whitespace, key order, a character split across chunks), and it couples the gateway to every field of the engine's JSON. This section builds a proxy that watches the stream without touching it.

The solution extends the tracer's file in place: [`proxy.go`](../../../../course/solutions/go/gateway/proxy/proxy.go) adds `StreamProxy` (with the unexported `sseSplitter` and `usageOf`), `StreamObserver`, `ExchangeObserver`, `Observers`, `Forward`, `Outbound`, `NewOutbound`, `UpstreamError`, and `Handler` (the chain's proxy stage), and keeps `NewProxy` with its original behaviour. Everything the tracer did still holds, and in the gateway's composition root the `Proxy` slot of the chain becomes `proxy.Handler(proxy.Config{Upstream: ...})`. The behaviour is fixed by the Streaming, Disconnect, and Errors rules of [`course/contracts/openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) (gateway tier); `NewProxy` still serves [`course/contracts/openapi/openai-subset.v0.yaml`](../../../../course/contracts/openapi/openai-subset.v0.yaml).

You need [the server skeleton](01-server-skeleton.md) (for `server.Usage` and the `Exchange`) and the tracer gateway. It also helps to know the engine's v1 streams from [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../../build-an-llm/p10-serving/05-openai-server-on-tokio.md) in *Build Your Own LLM System*, and the wire format from the [Streaming and SSE](../03-streaming-sse/README.md) chapter.

> **Learning objectives**
> - Relay a stream byte for byte while observing its events, first byte, and usage on the side.
> - Split SSE events correctly when reads cut events at arbitrary byte boundaries.
> - Estimate the gateway's added time to first byte, with and without a flush per read.
> - Draw the commitment boundary: say which failures may be retried on another worker and which must end in an SSE error event.
> - Plug the proxy into the chain so the limiter and the meter read usage from the `Exchange`.

## Streams and their two sides

Start with what flows through. An engine's streamed answer (`openai-subset.v1.yaml`) is `Content-Type: text/event-stream` and a sequence of **events**, each `data: <json>\n\n`. The first chunk carries `delta.role`; content chunks follow (some with empty `content` while a multi-byte character is incomplete); the last content chunk carries `finish_reason`; an optional chunk with `choices: []` carries `usage`; and the stream ends with `data: [DONE]\n\n`. Lines starting with `:` are comments, such as `: ping\n\n` every 15 s while idle. A non-streamed answer is one JSON body with a `usage` object.

The proxy relays to the **client** side and reads from the **upstream** side. The bytes it reads arrive in arbitrary pieces: TCP and the engine's buffers decide the boundaries, not the event structure. That one fact shapes the whole design.

## Relay first, observe on the side

`StreamProxy(w, r, up, o)` does four things, in order:

1. Copy the end-to-end headers of `up` to `w` (hop-by-hop headers dropped, as in the tracer), keeping the gateway's own `X-Request-Id`. `Header.Add` would append the engine's copy as a second value, and the client would see two ids.
2. Write the status.
3. Loop: read up to 32 KiB; if this is the first body byte, call `o.OnFirstByte()` **before** writing it (so the timestamp is not inflated by the write); write the bytes; flush; then feed the same bytes to the splitter.
4. At a clean end (EOF), call `o.OnDone(usage)` once.

The **splitter** appends each piece to a buffer and cuts complete events at `\n\n`. For each event it joins the payloads of its `data:` lines (with one space after the colon removed), skips events with no `data:` line (comments), and calls `o.OnChunk(data)`. Because it keeps the remainder between reads, an event split across ten 1-byte reads is still delivered once and whole. If an event's JSON has a non-null `usage`, it becomes the stream's usage; the last one wins. For a JSON body the proxy keeps a copy (up to 4 MiB) and reads `usage` from it at the end. When the engine omits `total_tokens`, it is `prompt_tokens + completion_tokens`.

`ExchangeObserver(ex)` is the observer the chain uses by default. It records `MarkFirstByte` and `SetUsage` on the request's `Exchange`, which is how the limiter settles and the ledger meters without importing this package. `Observers{a, b}` fans out to several observers; the response cache adds a recorder for stream replay this way.

## How long the first token waits

Relaying first raises a quantitative question: how much latency does the gateway add? Name the instants:

| Symbol | Meaning |
|---|---|
| $t_e$ | when the engine writes its first byte |
| $t_c$ | when the client receives it through the gateway |
| $\Delta = t_c - t_e$ | the gateway's added time to first byte |

With a flush per read, $\Delta$ is one socket read, one write, and one flush: well under a millisecond on one host. Without the flush (as the tracer gateway showed), $\Delta$ is the time for 4 KiB of events to accumulate, about a second at 20 tokens per second. The contract budget is $\Delta \le 5$ ms, measured as the difference between direct and proxied time to first byte. It is a benchmark on a quiet machine, not a correctness check, because shared CI machines make timing noisy.

## The commitment boundary

Latency is one constraint; failure is the other. HTTP sends the status line before the body, so a proxy's options depend on what it has already sent. `Forward` encodes that in what it returns:

| When | What happened | `Forward` returns | Client sees |
|---|---|---|---|
| before any answer | refused, reset, DNS, timeout | `UpstreamError{Committed: false}`, nothing written | whatever the caller decides: `Handler` answers 503 `no_capacity`, the router tries another worker |
| the engine answered 503 and `Retry503` is set | draining or queue full: no work was done | `UpstreamError{Committed: false, StatusCode: 503}`, nothing written | the same |
| any other answer | 200, 400, 404 ... | relayed; status and body pass through | the engine's answer |
| after the first byte | the stream broke | `UpstreamError{Committed: true}` after one `data: {"error": ...}` event | the bytes so far, then an error event, never `[DONE]` |
| any time | the client left | the upstream request is cancelled (it runs on the client's context) | nothing |

Retrying after the first byte would repeat or splice tokens: the client already printed "The cat", and a second engine starts again from "The". So a committed failure is final.

`Outbound` is the request as it will be sent: method, path, query, the tracer's header rules, and the body as bytes. `NewOutbound` builds it once, so the router can send the same request to several workers in turn.

## Working a stream by hand

Now trace one stream through. The engine answers with five events, 163, 147, 148, 156, and 14 bytes long (628 in total), and the gateway happens to read 40 bytes at a time (16 reads):

```
data: {"id":"c1",...,"delta":{"role":"assistant","content":""},...}\n\n
data: {"id":"c1",...,"delta":{"content":"Hel"},...}\n\n
data: {"id":"c1",...,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n
data: {"id":"c1",...,"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n
data: [DONE]\n\n
```

The reads where something observable happens are these:

| Read | Bytes | What happens |
|---|---|---|
| 1 | 0 to 39 | `OnFirstByte()` (the client has 0 bytes), write, flush; the splitter holds 40 bytes, no `\n\n` yet |
| 5 | 160 to 199 | write, flush; the splitter finds the first `\n\n` (ending at byte 162): `OnChunk` with the role chunk; 37 bytes stay buffered |
| 8 | 280 to 319 | the `"Hel"` event completes (ends at byte 309): `OnChunk` |
| 12 | 440 to 479 | the `"lo"` event completes (ends at byte 457): `OnChunk` |
| 16 | 600 to 627 | the usage event (ends at 613) and `[DONE]` complete: two `OnChunk` calls; the usage event sets usage 5/2/7 |
| EOF | | `OnDone({5, 2, 7})` |

The client received exactly the 628 bytes, `OnFirstByte` was called once, `OnChunk` five times (the last with `[DONE]`), and `OnDone` once. A correct solution must reproduce this table exactly, whatever the read size.

## The interface

The package keeps the tracer's surface and adds the streaming pieces:

```go
package proxy // import "tinyllm/gateway/proxy"

// from the tracer gateway, unchanged
func NewProxy(cfg Config) http.Handler
func ParseTraceparent(h string) (traceID, parentID, flags string, ok bool)

// the streaming proxy
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

The package uses only the standard library and `tinyllm/gateway/server`.

## Building it step by step

Seven observations determine the code:

1. **The client must see the engine's bytes, unchanged and without delay.** Any re-encoding can alter bytes (key order, whitespace, a UTF-8 character split across two chunks), and any buffering delays tokens. So the relay loop is: read whatever arrived, write it, flush, and only then do anything else with it.
2. **Observation must not sit on the relay path.** The metrics (first byte, events, usage) are side effects. They are computed from a copy of the bytes already written, so a slow or wrong observer can never change what the client sees.
3. **Event boundaries do not line up with reads.** In the hand example the first event ends at byte 162, in the middle of read 5. A parser that looks at each read alone would lose every event that straddles a read. So the splitter keeps a buffer across reads and cuts only at a complete `\n\n`.
4. **The first byte is an instant, not a property of every read.** It is the moment the gateway is about to hand the client its first body byte. So it fires once, guarded by a `first` flag, and before the write; timing the write would add the client's socket to the gateway's TTFT.
5. **Usage arrives late and in one of two shapes.** For a stream it is in a chunk near the end (`choices: []`); for a JSON body it is in the body. Either way it is only trustworthy at a clean EOF, so `OnDone` is called once, after the loop, and never on a broken stream.
6. **What the proxy may still do depends on what it has sent.** HTTP sends the status line before the body. Before the status is written, the request can go elsewhere; after, the client owns a partial answer. So `Forward` reports failures as an `UpstreamError` whose `Committed` field tells the caller which side of that line it was on, and writes nothing when uncommitted, leaving the decision (503, or another worker) to the caller.
7. **Cancellation must flow from the client to the engine.** Building the upstream request on the client's context makes a disconnect cancel the engine's request with no extra code.

### The splitter

The splitter is the only stateful parser in the file. It is a buffer plus a loop that emits every complete event currently in it:

```go
type sseSplitter struct {
	buf  []byte            // bytes of a not-yet-complete event, kept across reads
	emit func(data []byte) // called once per complete event that has data
}

func (s *sseSplitter) write(p []byte) {
	s.buf = append(s.buf, p...)
	for {
		i := bytes.Index(s.buf, []byte("\n\n"))
		if i < 0 {
			return // the rest waits for the next read (step 3)
		}
		event := s.buf[:i]
		s.buf = s.buf[i+2:]
		var data [][]byte
		for _, line := range bytes.Split(event, []byte("\n")) {
			line = bytes.TrimSuffix(line, []byte("\r"))
			if bytes.HasPrefix(line, []byte("data:")) {
				// "data: x" and "data:x" both carry "x": one optional space is removed
				data = append(data, bytes.TrimPrefix(bytes.TrimPrefix(line, []byte("data:")), []byte(" ")))
			}
		}
		if data != nil { // a comment (": ping") has no data line and is not an event
			s.emit(bytes.Join(data, []byte("\n")))
		}
	}
}
```

### Reading usage

`usageOf` reads usage from either shape. The cheap `bytes.Contains` check skips JSON decoding for the hundreds of content chunks that carry no usage, and `*server.Usage` distinguishes `"usage": null` (ignored) from a real object:

```go
func usageOf(doc []byte) (server.Usage, bool) {
	if !bytes.Contains(doc, []byte(`"usage"`)) {
		return server.Usage{}, false // fast path: most chunks
	}
	var v struct {
		Usage *server.Usage `json:"usage"`
	}
	if json.Unmarshal(doc, &v) != nil || v.Usage == nil {
		return server.Usage{}, false
	}
	u := *v.Usage
	if u.TotalTokens == 0 { // some engines omit it; the limiter and ledger need it
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
	}
	return u, true
}
```

### The relay loop

`StreamProxy` is steps 1 to 5 in code:

```go
func StreamProxy(w http.ResponseWriter, r *http.Request, up *http.Response, o StreamObserver) error {
	if o == nil {
		o = Observers(nil) // an empty fan-out: no nil checks in the loop
	}
	reqID := w.Header().Get("X-Request-Id")
	copyHeaders(w.Header(), up.Header) // end-to-end headers only (hop-by-hop dropped)
	if reqID != "" {
		w.Header().Set("X-Request-Id", reqID) // Set, not Add: one id, the gateway's
	}
	w.WriteHeader(up.StatusCode)
	sse := strings.HasPrefix(up.Header.Get("Content-Type"), "text/event-stream")
	var usage server.Usage
	var whole bytes.Buffer // a JSON body's copy, for its usage
	split := &sseSplitter{emit: func(d []byte) {
		o.OnChunk(d)
		if u, ok := usageOf(d); ok {
			usage = u // the last usage chunk wins
		}
	}}
	rc := http.NewResponseController(w)
	buf := make([]byte, 32*1024)
	first := true
	for {
		n, err := up.Body.Read(buf)
		if n > 0 { // a Read may return data and an error together: handle data first
			if first {
				o.OnFirstByte() // once, before the write (step 4)
				first = false
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				return werr
			}
			if ferr := rc.Flush(); ferr != nil && !errors.Is(ferr, http.ErrNotSupported) {
				return ferr
			}
			// only now, after the client has the bytes, observe them (step 2)
			if sse {
				split.write(buf[:n])
			} else if whole.Len() < server.DefaultMaxBody {
				whole.Write(buf[:n])
			}
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			// a broken stream: tell an SSE client explicitly, unless the client is the one who left
			if r.Context().Err() == nil && sse {
				_, _ = io.WriteString(w, `data: {"error":{"message":"the upstream stream failed","type":"server_error","param":null,"code":null}}`+"\n\n")
				_ = rc.Flush()
			}
			return fmt.Errorf("reading the upstream body: %w", err) // and no OnDone
		}
	}
	if !sse {
		if u, ok := usageOf(whole.Bytes()); ok {
			usage = u
		}
	}
	o.OnDone(usage) // a clean end only (step 5)
	return nil
}
```

### The commitment boundary in code

`Forward` draws the line of step 6. Every return before `StreamProxy` is uncommitted and has written nothing; an error out of `StreamProxy` is committed, because the status line is already out:

```go
func Forward(ctx context.Context, w http.ResponseWriter, base string, out *Outbound, client *http.Client, o StreamObserver) error {
	// ... default client; target = base + out.Path [+ "?" + out.RawQuery]
	up, err := http.NewRequestWithContext(ctx, out.Method, target, bytes.NewReader(out.Body)) // ctx: the client's (step 7)
	if err != nil {
		return &UpstreamError{Err: err}
	}
	up.ContentLength = int64(len(out.Body))
	for k, vs := range out.Header {
		up.Header[k] = append([]string(nil), vs...) // a copy: out is reused for the next worker
	}
	resp, err := client.Do(up)
	if err != nil {
		return &UpstreamError{Err: err} // refused, reset, DNS, timeout: nothing written
	}
	defer resp.Body.Close()
	if out.Retry503 && resp.StatusCode == http.StatusServiceUnavailable {
		return &UpstreamError{StatusCode: resp.StatusCode, Err: errors.New("the upstream answered 503")}
	}
	if err := StreamProxy(w, resp.Request, resp, o); err != nil {
		return &UpstreamError{Committed: true, Err: err} // never retry this one
	}
	return nil
}
```

`Outbound` holds the body as `[]byte`, not an `io.Reader`, for the same reason the headers are copied: the router may send the same request to three workers, and a reader can be consumed only once. `NewOutbound` applies the tracer's header rules once (drop `Authorization` and hop-by-hop headers, continue the trace with a new span id, set `X-Request-Id`).

### Choosing the observer and the 503

Finally, `serve` (shared by `NewProxy` and `Handler`, which differ only in `checkKey`) chooses the observer and turns an uncommitted failure into the client-facing 503:

```go
	var obs StreamObserver
	if cfg.Observer != nil {
		obs = cfg.Observer(r) // e.g. the response cache's recorder, combined with Observers{...}
	} else if ex := server.ExchangeFrom(r.Context()); ex != nil {
		obs = ExchangeObserver(ex) // in the chain: MarkFirstByte and SetUsage on the Exchange
	}
	err = Forward(r.Context(), sw, base, out, client, obs)
	var ue *UpstreamError
	if errors.As(err, &ue) && !ue.Committed && r.Context().Err() == nil {
		writeError(sw, http.StatusServiceUnavailable, "server_error", "no_capacity",
			"the upstream engine is unavailable: "+ue.Err.Error())
	}
```

`ExchangeObserver` is three one-line methods (`OnChunk` does nothing), and `Observers` loops over its members for each call. The full solution is [`proxy.go`](../../../../course/solutions/go/gateway/proxy/proxy.go).

### Why this design

Per stream, the relay is $O(n)$ in the bytes, with one 32 KiB buffer. The splitter adds $O(n)$ work and holds at most one incomplete event (plus whatever is still to be cut in the current read). `bytes.Index` restarts from the buffer's start on each write, which is quadratic only in the size of a single event, a few hundred bytes. A JSON body keeps at most `server.DefaultMaxBody` bytes for its usage.

The alternatives each fail somewhere. Parse-and-re-emit (decode every chunk, re-encode it) would be simpler to observe, but it can change bytes, delays each token by a decode and encode, and breaks on any field the gateway does not model; watching a copy has none of those problems. `httputil.ReverseProxy` with `FlushInterval: -1` streams correctly, but its failures go to an `ErrorHandler` that by default answers the client itself (502), so handing "uncommitted, nothing written" back to a router that wants to try another worker takes extra plumbing, and it has no hook per SSE event. Retrying after the first byte, to hide a broken stream, would splice two generations; the explicit error event is the honest choice, and the client can retry the whole request.

> **Note:** One tradeoff is accepted. The usage of a JSON body larger than `DefaultMaxBody` is not read, although the body is still relayed in full. Such answers do not occur for chat completions within the contract's limits.

## What can go wrong

The tests check each of the following, and each matters to a later stage:

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | the same bytes, one `OnFirstByte` before any write, five events, usage 5/2/7, status and content type kept | everyone agrees on what is observed |
| recorded streams at read sizes 1, 7, 64, 65536 | conformance | five recorded streams (chat, a UTF-8 character split across chunks, ping without usage, tool-call fragments, text completions): bytes equal, event counts and usage right at every read size | conformance `chat.stream.framing` through the gateway |
| ping comments | unit | `: ping` reaches the client and is not an event | idle streams stay open without fake events |
| a non-streamed answer | unit | a JSON body's usage, `total_tokens` computed, no events | the limiter and the ledger for non-streamed calls |
| a dead engine through `Forward` | fault | an uncommitted error, nothing written | the router can fail over |
| a 503 with and without `Retry503` | unit | with `Retry503` a 503 is handed back unwritten; without it, relayed | draining engines are skipped, not shown to clients |
| the outbound request | unit | no `Authorization` or hop-by-hop headers; request id, path, query, trace kept | the engine tier stays key-free |
| real sockets, an engine that pauses | unit | event 1 arrives while the engine still holds event 2 | tokens stream |
| `Handler` inside the chain | unit | no key needed, one `X-Request-Id` (the gateway's), first byte, usage, and status on the `Exchange` | the limiter settles and the ledger meters from it |
| a dead engine through `Handler` | boundary | before the first byte: 503 `no_capacity` | the tracer's answer, kept |
| a connection reset after the first event | fault | the client gets an SSE error event and no `[DONE]`; no `OnDone` | no silent truncation; no metering of a broken stream |
| a client that disconnects | fault | the engine's request is cancelled (contract: within 100 ms) | the engine stops generating for nobody |

The mistakes that most often break a streaming proxy, and how the solution avoids each:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| no flush after each write | tokens arrive in bursts; with an engine that waits, nothing arrives | `rc.Flush()` after every write, before the next read (step 1) |
| `OnFirstByte` after the write, or on every read | TTFT includes the write; the metric fires dozens of times per stream | called once, guarded by `first`, before `w.Write` (step 4) |
| usage from the wrong place: never from the SSE usage chunk, never from a JSON body, `total_tokens` left 0 | the limiter refunds everything; the ledger records zero tokens | `usageOf` runs on every event and on the JSON body, and fills in `total_tokens` |
| cutting events per read, or reporting comments as events | events split across reads vanish; pings count as tokens | the splitter buffers across reads and emits only events with a `data:` line |
| a broken stream that ends quietly and reports done | the client shows a short answer as complete; the ledger meters it | a read error writes an SSE error event and returns before `OnDone` |
| `Forward` that answers the client itself on failure, or ignores `Retry503` | failover is impossible: the 503 is already sent | `Forward` returns an uncommitted `UpstreamError` and writes nothing; `serve` decides on the 503 |
| the upstream request on `context.Background()` | a client that hangs up leaves the engine generating | `http.NewRequestWithContext(ctx, ...)` with the client's context (step 7) |
| `Header.Add` for every upstream header | the client gets two `X-Request-Id` values | the gateway's id is saved and restored with `Set` after `copyHeaders` |
| the chain's handler still checking the static key | every request is a 401 once real keys replace `TL_API_KEY` | `Handler` calls `serve` with `checkKey` false; only `NewProxy` checks |

## Where this leads

This section stands on [the server skeleton](01-server-skeleton.md), which supplies `server.Usage`, the `Exchange` that the default observer records on, and the `Proxy` slot, and on [the tracer gateway](00-streaming-proxy.md), whose request path it upgrades while keeping its behaviour.

Going forward, [Routing, cascades, and failover](05-routing.md) sends each routing attempt with `Forward`, relies on its commitment boundary, and retries only uncommitted failures. Together with routing and [the usage ledger](07-usage-ledger-and-admin-api.md), this proxy is what makes the gateway a real gateway: streamed, routed, and metered. Two call sites are inherited from the tracer: the gateway image of [Tracer deploy](../../infrastructure/01-containers-kubernetes/00-tracer-deploy.md) now serves `Handler` in the chain, and the `gateway.proxy` span traced in [One trace: gateway to engine in Jaeger](../../systems/04-observability/00-one-trace.md) is unchanged.

### Further reading

- `Forward` and the commitment boundary in production: saige's retry provider (`provider/retry`), which buffers until the first content delta and then never retries (`drainUntilContentOrError`).
- `StreamProxy` in the standard library: [`httputil.ReverseProxy`](https://pkg.go.dev/net/http/httputil#ReverseProxy) with `FlushInterval: -1` (free), with hooks for rewriting.
- The observer in production: [Envoy AI Gateway](https://aigateway.envoyproxy.io/) token usage extraction (free), which reads usage from provider streams for metering and limits.
- Disconnect handling on the engine side: [vLLM](https://github.com/vllm-project/vllm)'s abort on client disconnect (free).
- WHATWG HTML, [server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free); RFC 9110 section 9.3 on request semantics and retries (free).

## Key terms

- **SSE event**: one `data: ...` block terminated by a blank line (`\n\n`) in a `text/event-stream` response.
- **splitter**: a buffered parser that cuts a byte stream into complete SSE events regardless of read boundaries.
- **stream observer**: a side-channel receiver of the first byte, each event, and the final usage, which cannot affect the bytes the client sees.
- **time to first token (TTFT)**: the time until the first body byte of an answer is ready to send to the client.
- **commitment boundary**: the point (the status line) after which a proxy can no longer retry a request elsewhere.
- **uncommitted failure**: an upstream failure before anything was written to the client, which a caller may retry on another worker.

## Summary

- The proxy still forwards bytes: every read from the engine is written and flushed before the next read, so the client's stream is byte-identical to the engine's whatever the read sizes.
- While the bytes pass, a splitter cuts them into SSE events at blank lines for an observer: the first byte (time to first token), every event, and the usage from the last usage chunk, delivered once at a clean end.
- `Forward` draws the commitment boundary: before the engine answers, a failure writes nothing and is reported uncommitted, so routing may try another worker; after the first byte nothing is retried and a broken stream ends with an SSE error event.
- The client's context is the upstream request's context: when the client leaves, the engine's request is cancelled.
- `Handler` is the chain's proxy stage: no static key (authn already ran), the gateway's `X-Request-Id` and no duplicate from the engine, and the first byte and usage recorded on the `Exchange` for the limiter and the meter.

## Review questions

1. Why is `OnFirstByte` called before `w.Write`, and why only once?

   <details><summary>Answer</summary>

   TTFT is the gateway's view of when the first token was ready to send. Calling it after the write would add the time to push bytes into the client's socket, which the gateway does not control. Calling it on every read would fire dozens of times per stream and make the metric meaningless; the `first` flag makes it one instant.

   </details>

2. In the hand example, read 5 contains the end of event 1 and the start of event 2. What does the splitter hold after it, and why can it not simply parse each read?

   <details><summary>Answer</summary>

   It emits event 1 (which ends at byte 162) and keeps the remaining 37 bytes of event 2 in `buf`. Reads are cut by TCP and buffers, not by events, so a per-read parser would see half events; at read size 1 it would see no complete event at all.

   </details>

3. Why does `Forward` return `Committed: true` for any error out of `StreamProxy`, even one that happens on the first read?

   <details><summary>Answer</summary>

   `StreamProxy` writes the status line before its first read. Once the client has a status, a second engine's answer cannot be substituted cleanly, and if any tokens were sent a retry would repeat or splice them. So everything after `WriteHeader` is final.

   </details>

4. What would change if the upstream request used `context.Background()`?

   <details><summary>Answer</summary>

   A client that hangs up would no longer cancel the engine's request: the engine would keep generating (and holding a batch slot) for nobody until it finished. Using `r.Context()` makes the disconnect propagate for free.

   </details>

5. Why does `StreamProxy` re-`Set` the `X-Request-Id` after copying the upstream headers?

   <details><summary>Answer</summary>

   `copyHeaders` uses `Add`, so the engine's echo of the id would become a second value next to the gateway's. Restoring the saved value with `Set` leaves exactly one id, the gateway's, which is the one in its logs and spans.

   </details>
