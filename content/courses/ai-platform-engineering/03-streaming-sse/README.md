# Streaming and SSE

A request/response cycle assumes the answer exists all at once. A generated answer does not: it *comes into being* token by token, and a long job *makes progress* step by step. The result is a sequence over time, not a value. Streaming transports keep one connection open and push the pieces as they are ready, so the user sees the first token in tens of milliseconds instead of waiting seconds for the last one.

Server-Sent Events (SSE) is the minimal version of this idea: a normal HTTP response that simply never ends, framed as a list of text events. Most "AI streaming" is exactly this. SSE is one-way (server to client), plain text, and ordinary HTTP, and those constraints are features rather than limitations. They are why it survives proxies, why the browser can reconnect it for free, and why it is the default transport for LLM responses.

This chapter first makes the case for streaming at all, then specifies the SSE protocol, compares it with WebSockets, long-polling, and gRPC streaming, and traces the canonical end-to-end path for streaming LLM tokens. It closes with the operational concerns that make streaming hard in production: buffering proxies, heartbeats, resumption, backpressure, and scaling long-lived connections.

> **Learning objectives**
> - Explain why streaming generation improves perceived latency, and why time-to-first-token is the metric that matters.
> - Read and write the SSE wire format: `data:`, `event:`, `id:`, and `retry:` fields, and blank-line dispatch.
> - Choose between SSE, WebSockets, long-polling, and gRPC streaming for a given hop.
> - Trace an LLM token stream from the inference engine through the gateway to the browser, including the `[DONE]` sentinel and cancellation.
> - Handle the operational hazards of streaming: proxy buffering, idle timeouts, `Last-Event-ID` resumption, and backpressure.

## Before you begin

This chapter builds on [RPC and Protocols](../02-rpc-and-protocols/), especially gRPC server streaming and HTTP/2, and assumes basic HTTP. It also helps to know why generation is token by token, which [LLM Systems & Inference](../../machine-learning/04-llm-systems/) in *Machine Learning and AI* explains along with continuous batching and the KV cache.

## In this chapter

- [Why stream at all](#why-stream-at-all): perceived latency, cancellation, and memory.
- [Server-Sent Events](#server-sent-events): the wire format and the browser's `EventSource`.
- [SSE, WebSockets, long-polling, and gRPC streaming](#sse-websockets-long-polling-and-grpc-streaming): choosing a transport by direction and format.
- [Streaming LLM tokens end to end](#streaming-llm-tokens-end-to-end): the canonical AI-platform path, framing, and cancellation.
- [Operational concerns](#operational-concerns): buffering, heartbeats, resumption, backpressure, and scaling.

## Why stream at all

The first reason is **perceived latency**. Autoregressive decoding produces about one token per forward pass. If the server buffers the whole completion, the user waits for *all* of it; if it streams, the user reads along with generation. Making the user wait for the full completion throws away the latency win, and **time-to-first-token (TTFT)** becomes the experience you are optimizing.

The second reason is **progress and cancellation**. A stream lets the user *stop* a bad generation early, and that lets the platform free the GPU immediately instead of finishing wasted work. The third is **memory**: streaming a large result, or a long-running job's logs, avoids buffering it all on the server.

Streaming is the last mile of the inference path, which runs from the model to the serving engine to the gateway and then, as a stream, to the client. The rest of this chapter is about that final hop.

## Server-Sent Events

SSE is a never-ending HTTP response of text events, served with `Content-Type: text/event-stream`. Here is a stream of three events:

```
data: Hello

data: world
id: 42
event: token

data: {"token": "!", "done": false}

```

The format has four fields. `data:` carries the payload and can repeat to send multiple lines. `event:` names an event type the client can listen for. `id:` is the event's id, which the browser echoes back as `Last-Event-ID` when it reconnects. `retry:` sets the reconnect delay in milliseconds. A blank line dispatches the event. That is the whole wire format, and it is trivial on purpose: the simplicity is why it survives proxies and reconnects cleanly.

On the client, the browser does the work:

```javascript
const es = new EventSource("/stream");
es.onmessage = e => append(e.data);
```

`EventSource` handles the connection, the parsing, and **automatic reconnection**.

Three constraints define what SSE is good for. It is **one-way**: server to client only. The client opens it with a normal GET, and to send data back it makes a separate request. It is **plain HTTP**, so it works through proxies, CDNs, and firewalls that mishandle WebSocket upgrades, and over HTTP/2 it is not limited by the old six-connections-per-domain cap. And it is **text only**: SSE carries UTF-8, so binary must be base64-encoded, which is one reason to prefer WebSockets for binary data.

## SSE, WebSockets, long-polling, and gRPC streaming

SSE is not the only way to push data, so it helps to place it among the alternatives. The four transports differ in direction, format, and who handles reconnection:

| Transport | Direction | Format | Reconnect | Best for |
|-----------|-----------|--------|-----------|----------|
| **SSE** | Server → client | Text | Built-in (`EventSource`) | LLM tokens, notifications, progress, live feeds |
| **WebSockets** | Bidirectional | Text or binary | Manual | Chat, collaboration, games, trading |
| **Long-polling** | Server → client (faked) | Any | Per-request | Legacy fallback when SSE/WS unavailable |
| **gRPC streaming** | Uni/bidi | Binary (Protobuf) | Manual | Service-to-service streams (not browsers) |

The decision between SSE and WebSockets is a one-way versus two-way decision. SSE is simpler (it is just HTTP), one-way, auto-reconnecting, and text. WebSockets are bidirectional, binary-capable, and lower-overhead per message, but you manage reconnection yourself and they need a protocol upgrade that some infrastructure mishandles. If the server just pushes and the client just listens (tokens, progress, notifications), SSE is simpler, cheaper, and firewall-friendly. Use WebSockets only when you truly need bidirectional, low-latency, binary frames, as in collaborative editing or games. For streaming model output, SSE is almost always the right default.

gRPC server streaming is the same idea as SSE for *backend-to-backend* traffic: binary, typed, and over HTTP/2. Browsers cannot speak it directly, so the common pattern is a gRPC stream internally and SSE to the browser, converted at the gateway. Pick the transport per hop.

Long-polling is the fallback: the client sends a request, the server holds it until data is available, responds, and the client immediately requests again. It has higher latency and overhead, so use it only when neither SSE nor WebSockets is available.

## Streaming LLM tokens end to end

With the transport chosen, you can trace the canonical streaming path of an AI platform. The client opens an SSE request to the gateway. The gateway calls the inference engine (vLLM or TGI), which yields tokens through its own streaming API, often a gRPC server stream or an async generator. The gateway re-emits each token as an SSE `data:` event, and a final `data: [DONE]` sentinel closes the stream.

Framing the payload follows a de facto convention set by OpenAI-style APIs: each event carries a JSON delta such as `{"choices":[{"delta":{"content":"Hel"}}]}`, and the stream ends with the `[DONE]` marker. When the stream carries more than tokens, such as tool calls or errors, use named SSE `event:` types (`token`, `tool_call`, `error`, `done`) so the client can branch on the event type without sniffing payloads.

> **Warning:** Cancellation is the part teams forget. When the client disconnects (closes the tab, presses stop), propagate the cancellation down to the serving engine so it evicts the request and frees its KV cache and GPU slot immediately. If you do not, the GPU keeps generating for nobody, and you leak work. This ties directly to continuous batching in [LLM Systems & Inference](../../machine-learning/04-llm-systems/).

## Operational concerns

The protocol is easy; running it is where the difficulty lies. Backpressure and disconnects are the hard parts: a client that stops reading, a proxy that buffers, a dropped mobile connection.

**Buffering proxies** are the most common way streaming silently breaks. nginx and load balancers may buffer the whole response, which defeats streaming. Disable buffering (for nginx, the `X-Accel-Buffering: no` response header), set appropriate timeouts, and use HTTP/1.1 chunked transfer or HTTP/2.

**Heartbeats** keep idle connections alive. Send an SSE comment line (`: ping`) periodically so that neither the client nor any intermediary times the connection out while the model is thinking.

**Resumption** uses the `id:` field. On reconnect, the browser sends `Last-Event-ID`, and the server can replay from that id if it kept a buffer. That is often unnecessary for a fresh generation but essential for durable event feeds.

**Connection limits and backpressure** follow from the fact that each stream is a held-open connection and, for generation, a pinned GPU slot. Cap concurrency, apply timeouts, and shed load; a slow client should not pin a GPU forever.

**Scaling** long-lived connections complicates load balancing and graceful shutdown: drain connections rather than cutting them. Use sticky routing or a stateless re-attach, and lean on HTTP/2 multiplexing for fan-out.

## Choosing a streaming transport

The chapter's decisions condense into a short lookup:

| Situation | Use |
|-----------|-----|
| Stream LLM tokens to a browser | **SSE** |
| Server-push notifications / progress / live feed | **SSE** |
| Bidirectional, low-latency, binary (chat/collab/games) | **WebSockets** |
| Backend-to-backend streaming | **gRPC server-streaming** |
| Infra blocks SSE and WS | **Long-polling** (fallback) |
| Need to cancel a generation | Propagate disconnect → evict from serving engine |

You can see these choices across the industry. OpenAI and Anthropic expose SSE token streaming as their public API surface, and nearly every generative-AI product team uses SSE with a stop-generation control to drive a TTFT-focused experience. Vercel and Cloudflare stream SSE at the edge and the CDN, in their AI gateways. Slack and Figma are the contrast case: their real-time features are genuinely bidirectional, so they use WebSockets.

## Where this leads

This book builds the streaming path for real in the gateway chapter: [SSE streaming proxy](../12-gateway/04-sse-streaming-proxy.md) implements the gateway's half of the path described here, relaying an engine's token stream to the client. Outside this book, [LLM Systems & Inference](../../machine-learning/04-llm-systems/) explains token-by-token decode, TTFT, and continuous batching, which is why output is a stream and why cancellation frees the GPU. [System Design](../../systems/01-system-design/) covers load balancing, timeouts, and connection handling for long-lived connections; [Cloud Native](../../systems/03-cloud-native/) covers SSE through Kubernetes ingress, draining, and graceful shutdown; [Observability](../../systems/04-observability/) covers measuring TTFT and tail latency; and [Batch & Streaming Processing](../../data-engineering/03-batch-streaming/) in *Data Engineering* contrasts durable event feeds, with replay and delivery semantics, with ephemeral generation streams.

### Further reading

- The [HTML Server-Sent Events spec](https://html.spec.whatwg.org/multipage/server-sent-events.html) and [MDN Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (free): the primary references.
- [MDN EventSource](https://developer.mozilla.org/en-US/docs/Web/API/EventSource).
- [WebSocket (RFC 6455)](https://www.rfc-editor.org/rfc/rfc6455) and [HTTP/2 (RFC 9113)](https://www.rfc-editor.org/rfc/rfc9113.html).
- [OpenAI streaming docs](https://platform.openai.com/docs/api-reference/streaming), for the de facto delta-and-`[DONE]` framing.
- [WHATWG Fetch and ReadableStream](https://fetch.spec.whatwg.org/), for consuming streams without `EventSource`.

## Key terms

- **Server-Sent Events (SSE)**: a one-way, text-only stream of events over a single long-lived HTTP response with `Content-Type: text/event-stream`.
- **`EventSource`**: the browser API that opens an SSE stream, parses events, and reconnects automatically.
- **Time-to-first-token (TTFT)**: the delay between sending a request and receiving the first generated token.
- **`Last-Event-ID`**: the header a reconnecting client sends with the id of the last event it received, enabling replay.
- **`[DONE]` sentinel**: the final `data:` event that marks the end of an OpenAI-style token stream.
- **Heartbeat**: a periodic SSE comment line (`: ping`) that keeps idle connections from timing out.
- **Backpressure**: limiting what a producer sends or holds when a consumer reads slowly.

## Summary

- Generation is inherently a stream; streaming tokens as they are produced makes time-to-first-token the metric that matters.
- SSE is the default transport for LLM responses: one-way, plain text over one long-lived HTTP response, with `EventSource` reconnecting for free.
- SSE versus WebSockets is a one-way versus two-way decision; use gRPC streaming between backend services and SSE at the last mile.
- The wire format is deliberately trivial: `data:` lines, blank-line-delimited events, optional `event:`, `id:`, and `retry:`.
- Cancellation and backpressure are the hard parts: a disconnect must free the upstream GPU slot, and held-open connections must be bounded, timed out, and shed.

## Review questions

1. Why does streaming improve the user's experience even though total generation time is unchanged?
   <details><summary>Answer</summary>The model produces about one token per forward pass, so the first token exists long before the last. Streaming shows it immediately, so the user starts reading at time-to-first-token instead of waiting for the whole completion.</details>

2. What does a blank line do in an SSE stream, and what does `id:` enable?
   <details><summary>Answer</summary>A blank line dispatches the accumulated fields as one event. `id:` sets the event's id, which the browser sends back as `Last-Event-ID` on reconnect so the server can replay from that point.</details>

3. Why is SSE usually a better choice than WebSockets for streaming model output?
   <details><summary>Answer</summary>Model output is one-way text. SSE is plain HTTP, so it passes through proxies and firewalls that mishandle WebSocket upgrades, and the browser reconnects it automatically. WebSockets add bidirectional, binary capability that this use case does not need, at the cost of manual reconnection.</details>

4. A user closes the tab halfway through a long generation. What should the platform do, and what happens if it does not?
   <details><summary>Answer</summary>The gateway should detect the disconnect and propagate cancellation to the serving engine, which evicts the request and frees its KV cache and GPU slot. Otherwise the engine keeps generating tokens nobody will read, wasting GPU capacity that other requests could use.</details>

5. Your SSE endpoint works locally but clients behind your load balancer receive the whole response at once. What is the likely cause and fix?
   <details><summary>Answer</summary>An intermediary (nginx or the load balancer) is buffering the response. Disable buffering, for example with `X-Accel-Buffering: no` for nginx, check its timeouts, and use HTTP/1.1 chunked transfer or HTTP/2.</details>

## Exercises

- Build a `/stream` endpoint that emits one SSE event per word of a sentence, consume it in the browser with `EventSource`, and watch the events arrive incrementally.
- Wire it to a real model: stream vLLM or OpenAI tokens through your server to the client, and measure TTFT against total time.
- Kill the connection mid-stream, confirm that `EventSource` reconnects automatically, and implement `Last-Event-ID` resumption.
- Re-implement the same feature over WebSockets and over gRPC server streaming, and list what each gained and lost.
