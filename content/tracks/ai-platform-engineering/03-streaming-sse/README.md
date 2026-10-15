# Streaming & SSE

## Overview

- **Primary references**: [HTML Server-Sent Events spec](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free), [MDN Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (free)
- **Supplementary**: [MDN EventSource](https://developer.mozilla.org/en-US/docs/Web/API/EventSource), [WebSocket (RFC 6455)](https://www.rfc-editor.org/rfc/rfc6455), [HTTP/2 (RFC 9113)](https://www.rfc-editor.org/rfc/rfc9113.html), [OpenAI streaming docs](https://platform.openai.com/docs/api-reference/streaming), [WHATWG Fetch + ReadableStream](https://fetch.spec.whatwg.org/)
- **Prerequisites**: [RPC & Protocols](../02-rpc-and-protocols/), HTTP basics, [LLM Systems & Inference](../../ml/04-llm-systems/) (why generation is token-by-token)
- **Estimated time**: 1-2 weeks at 6-8 hrs/week

## Key Takeaways

- **SSE is the default transport for LLM responses.** One-way server→client, plain text over a single long-lived HTTP response, with the browser's `EventSource` handling reconnection for free.
- **Generation is inherently a stream.** A model emits one token per forward pass; making the user wait for the whole completion throws away the latency win. Stream tokens as they're produced and time-to-first-token becomes the metric that matters.
- **SSE vs WebSockets is a one-way vs two-way decision.** If the server just pushes and the client just listens (tokens, progress, notifications), SSE is simpler, cheaper, and firewall-friendly. If you need true bidirectional, low-latency, binary frames (collab, games), use WebSockets.
- **The wire format is trivial on purpose**: `data:` lines, blank-line-delimited events, optional `event:`/`id:`/`retry:`. That simplicity is why it survives proxies and reconnects cleanly.
- **Backpressure and disconnects are the hard parts.** A client that stops reading, a proxy that buffers, a dropped mobile connection — the platform must handle cancellation (stop the GPU work) and resumption (`Last-Event-ID`).

## How to Study

- Build a `/stream` endpoint that emits one SSE event per word of a sentence; consume it in the browser with `EventSource` and watch them arrive incrementally.
- Wire it to a real model: stream vLLM/OpenAI tokens through your server to the client. Measure TTFT vs total time.
- Kill the connection mid-stream and confirm `EventSource` auto-reconnects; implement `Last-Event-ID` resume.
- Re-implement the same feature over WebSockets and over gRPC server-streaming; list what each gained and lost.

---

# Concepts & Techniques

## Core Insight

A request/response cycle assumes the answer exists all at once. But a generated answer *comes into being* token by token, and a long job *makes progress* step by step — the result is a sequence over time, not a value. Streaming transports keep one connection open and push pieces as they're ready, so the user sees the first token in tens of milliseconds instead of waiting seconds for the last one. SSE is the minimal viable version of this: a normal HTTP response that simply never ends, framed as a list of text events. Most "AI streaming" is exactly this, and its constraints (one-way, text, HTTP) are features, not limitations.

## 1. Why Stream at All

**Key ideas**:
- **Perceived latency**: autoregressive decoding produces ~1 token per forward pass. Buffering the whole completion means the user waits for *all* of it; streaming means they read along with generation. **Time-to-first-token (TTFT)** becomes the experience.
- **Progress and cancellation**: a stream lets the user *stop* a bad generation early — which lets the platform free the GPU immediately instead of finishing wasted work.
- **Memory**: streaming a large result (or a long-running job's logs) avoids buffering it all server-side.
- **This is the last mile** of the inference path: model → serving engine → gateway → **stream to client**.

## 2. Server-Sent Events (SSE)

**A never-ending HTTP response of text events**

**The protocol** (`Content-Type: text/event-stream`):
```
data: Hello

data: world
id: 42
event: token

data: {"token": "!", "done": false}

```
**Key ideas**:
- **Fields**: `data:` (the payload, can repeat for multi-line), `event:` (named event type the client can listen for), `id:` (event id, echoed back as `Last-Event-ID` on reconnect), `retry:` (reconnect delay in ms). A blank line dispatches the event.
- **Client**: `const es = new EventSource("/stream"); es.onmessage = e => append(e.data);` — the browser handles the connection, parsing, and **automatic reconnection**.
- **One-way**: server → client only. The client opens it with a normal GET; to send data it uses a separate request.
- **Plain HTTP**: works through proxies, CDNs, and firewalls that mishandle WebSocket upgrades. Over HTTP/2 it isn't limited by the old 6-connections-per-domain cap.
- **Text only**: SSE carries UTF-8 text. Binary must be base64-encoded (a reason to prefer WebSockets for binary).

## 3. SSE vs WebSockets vs Long-Polling vs gRPC Streaming

| Transport | Direction | Format | Reconnect | Best for |
|-----------|-----------|--------|-----------|----------|
| **SSE** | Server → client | Text | Built-in (`EventSource`) | LLM tokens, notifications, progress, live feeds |
| **WebSockets** | Bidirectional | Text or binary | Manual | Chat, collaboration, games, trading |
| **Long-polling** | Server → client (faked) | Any | Per-request | Legacy fallback when SSE/WS unavailable |
| **gRPC streaming** | Uni/bidi | Binary (Protobuf) | Manual | Service-to-service streams (not browsers) |

**Key distinctions**:
- **SSE vs WebSockets**: SSE is *simpler* (it's just HTTP), one-way, auto-reconnecting, text. WebSockets are bidirectional, binary-capable, lower-overhead per message, but you manage reconnection and they need a protocol upgrade some infra mishandles. **For streaming model output, SSE is almost always the right default.**
- **SSE vs gRPC streaming**: gRPC server-streaming is the same idea for *backend-to-backend* (binary, typed, HTTP/2). Browsers can't speak it directly. The common pattern: **gRPC stream internally → SSE to the browser** at the gateway.
- **Long-polling**: the fallback — client requests, server holds until data, responds, client re-requests. Higher latency and overhead; use only when SSE/WebSockets aren't available.

## 4. Streaming LLM Tokens End-to-End

**The canonical AI-platform streaming path**

**Key ideas**:
- **The flow**: client opens an SSE request → gateway calls the inference engine (vLLM/TGI) which yields tokens via its own streaming API (often gRPC server-stream or an async generator) → gateway re-emits each token as an SSE `data:` event → final `data: [DONE]` sentinel closes it.
- **Framing the payload**: most APIs send JSON deltas (`{"choices":[{"delta":{"content":"Hel"}}]}`) per event, ending with a `[DONE]` marker — the de-facto convention (OpenAI-style).
- **Cancellation is critical**: when the client disconnects (closes the tab, hits stop), propagate the cancellation down to the serving engine so it **evicts the request and frees KV-cache/GPU** immediately (ties to continuous batching in [LLM Systems](../../ml/04-llm-systems/)).
- **Tool calls / structured events**: use named SSE `event:` types (`token`, `tool_call`, `error`, `done`) so the client can branch without sniffing payloads.

## 5. Operational Concerns

**Key ideas**:
- **Buffering proxies**: nginx/load balancers may buffer the response and defeat streaming — disable buffering (`X-Accel-Buffering: no`), set the right timeouts, and use HTTP/1.1 chunked or HTTP/2.
- **Heartbeats / keep-alive**: send a comment line (`: ping`) periodically so idle connections and intermediaries don't time out.
- **Resumption**: on reconnect the browser sends `Last-Event-ID`; the server can replay from that id if it kept a buffer (often unnecessary for fresh generations, essential for durable event feeds).
- **Connection limits & backpressure**: each stream is a held-open connection and (for generation) a pinned GPU slot. Cap concurrency, apply timeouts, and shed load — a slow client shouldn't pin a GPU forever.
- **Scaling**: long-lived connections complicate load balancing and graceful shutdown (drain, don't cut). Sticky routing or stateless re-attach; HTTP/2 multiplexing helps fan-out.

---

## Decision Cheat Sheet

| Situation | Use |
|-----------|-----|
| Stream LLM tokens to a browser | **SSE** |
| Server-push notifications / progress / live feed | **SSE** |
| Bidirectional, low-latency, binary (chat/collab/games) | **WebSockets** |
| Backend-to-backend streaming | **gRPC server-streaming** |
| Infra blocks SSE and WS | **Long-polling** (fallback) |
| Need to cancel a generation | Propagate disconnect → evict from serving engine |

## Patterns Worth Internalizing

- **A result that arrives over time is a stream, not a value** — push the pieces as they're ready instead of buffering the whole answer.
- **One-way vs two-way is the decision** — SSE when the server pushes and the client listens; WebSockets only when you truly need bidirectional/binary.
- **Cancellation must propagate** — a client disconnect should free the upstream resource (the GPU slot), or you leak work.
- **Backpressure is mandatory** — a held-open connection is a held resource; bound concurrency, time out, and shed load.
- **Internal stream → edge stream** — gRPC server-streaming between services, SSE at the last mile; pick the transport per hop.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Token-by-token decode, TTFT, continuous batching | [LLM Systems & Inference](../../ml/04-llm-systems/) | Why output is a stream; cancellation frees GPU |
| gRPC streaming, HTTP/2, protocol choice | [RPC & Protocols](../02-rpc-and-protocols/) | Internal stream → SSE at the edge |
| Load balancing, timeouts, connection handling | [System Design](../../systems/01-system-design/) | Serving long-lived connections |
| Ingress, draining, graceful shutdown | [Cloud Native](../../systems/03-cloud-native/) | SSE through K8s ingress |
| Tail latency, TTFT measurement | [Observability](../../systems/04-observability/) | Instrumenting the stream |
| Event streams, replay, delivery semantics | [Data Engineering](../../data-engineering/03-batch-streaming/) | Durable feeds vs ephemeral generation |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| OpenAI / Anthropic | Server-push token streaming | SSE is the public API surface |
| Any GenAI product team | Stream + cancel for TTFT-driven UX | SSE with stop-generation |
| Vercel / Cloudflare | Edge streaming / AI gateway | SSE at the CDN |
| Slack / Figma | Bidirectional real-time | WebSockets (the contrast case) |
