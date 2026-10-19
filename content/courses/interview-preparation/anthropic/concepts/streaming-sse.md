# Streaming and Server-Sent Events

A language model produces its answer one token at a time, and users expect to see it the same way. Streaming turns a multi-second generation into something that starts responding in a fraction of a second. But it also turns a simple request-response exchange into a long-lived connection that must survive proxies, slow clients, safety checks, network drops, and server deployments. Handling all of that well is a large part of what makes an inference API feel fast and reliable.

This section is the background for the streaming parts of [the system design round](../03-system-design.md). It begins with the Server-Sent Events protocol and why it, rather than WebSockets, is the natural fit for LLM serving. It then looks at Anthropic's typed event sequence, at backpressure when a client cannot keep up, at the server-side pipeline from GPU to HTTP writer (including detokenization and speculative safety checks), and at connection management: heartbeats, reconnection, and graceful shutdown. It closes with the latency metrics and SLOs for a streaming service and where latency actually hides.

> **Learning objectives**
> - Describe the SSE wire format and explain why SSE suits unidirectional token streaming better than WebSockets.
> - Explain the order and purpose of Anthropic's typed streaming events.
> - Design backpressure for a slow client, and justify dropping the connection over the alternatives.
> - Handle partial UTF-8 tokens in detokenization and place a speculative safety check in the stream.
> - Implement heartbeats and `Last-Event-ID` reconnection, and define TTFT, TPOT, and TBT with sensible SLOs.

## The SSE protocol

Server-Sent Events is a simple HTTP-based protocol for streaming from server to client over a persistent connection. The exchange has four parts. The client sends an ordinary HTTP GET with `Accept: text/event-stream`. The server responds with `Content-Type: text/event-stream` and keeps the connection open. The server then sends events as they occur, each terminated by a blank line. And the connection is unidirectional: data flows from server to client only.

Each event is a few lines of text:

```text
event: <event-type>
data: <payload>
id: <event-id>
retry: <reconnection-time-ms>

```

The `event:` field is an optional type that clients can listen for selectively. The `data:` field is the payload; multiple `data:` lines are joined with newlines. The `id:` field is an optional event ID that supports reconnection. The `retry:` field is an optional hint for the reconnection delay. And a blank line terminates the event.

The obvious alternative is a WebSocket. The two compare as follows:

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server -> Client | Bidirectional |
| Protocol | HTTP/1.1 or HTTP/2 | Custom upgrade |
| Reconnection | Built-in (Last-Event-ID) | Manual |
| Load balancer support | Standard HTTP | Needs special config |
| Simplicity | Very simple | More complex |
| Use case fit | Token streaming (unidirectional) | Chat (bidirectional) |

LLM inference is fundamentally unidirectional: the client sends a prompt, and the server streams tokens back. SSE is the simpler and better-supported choice for that shape, and you get reconnection for free.

## Anthropic's typed events

On top of SSE, Anthropic's API defines a structured sequence of typed events. A message opens with metadata, contains one or more content blocks each delivered as a run of deltas, and closes with final usage and a stop event:

```text
message_start          # Message metadata (id, model, usage estimate)
    |
content_block_start    # Start of a content block (text, tool_use, etc.)
    |
content_block_delta    # Incremental content (repeated, one per token/chunk)
content_block_delta
content_block_delta
    ...
    |
content_block_stop     # End of content block
    |
message_delta          # Final usage statistics update
    |
message_stop           # Message complete
```

On the wire, a short response looks like this:

```text
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc123","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}

event: message_stop
data: {"type":"message_stop"}
```

Typing the events buys four things. Clients get **structured parsing**, knowing exactly what each event means. **Tool use** streams naturally, since a `content_block_start` with `type: "tool_use"` opens a streamed tool call. The `usage` fields in `message_start` and `message_delta` allow **real-time token counting**. And a single message can carry **multiple content blocks**, with text and tool use interleaved, each identified by its `index`.

## Backpressure and flow control

A GPU generates 50 to 200 tokens per second, which can outpace a slow client's ability to consume them. Without backpressure, the server buffers without bound and eventually exhausts its memory.

The first line of defense is already there: **TCP flow control**. The server writes to the socket; if the client is slow, its TCP receive window fills up; the server's `send()` then blocks or returns a short write; and that naturally throttles the server. For SSE over HTTP/1.1 this works automatically, because the server's write call blocks when the client cannot keep up.

TCP backpressure is not always enough, though. A buffering proxy between server and client absorbs data quickly and hides the slow client behind it. For that case you need **application-level backpressure**: a bounded per-stream buffer that reports when it is full.

```python
class StreamBuffer:
    def __init__(self, max_size: int = 1000):
        self.buffer = asyncio.Queue(maxsize=max_size)
        self.dropped = 0

    async def put(self, event: str) -> bool:
        try:
            self.buffer.put_nowait(event)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            return False

    async def get(self) -> str:
        return await self.buffer.get()
```

When the buffer fills, there are three things you can do. You can **drop the connection**, the cleanest option; the client reconnects. You can **drop events**, which is lossy, though some events, such as heartbeats, are safe to drop. Or you can **block the producer**, pausing token generation for this request, which wastes GPU time that other requests could use. The best practice for LLM serving is to drop the connection once the buffer exceeds its threshold, and let the client resume with `Last-Event-ID`.

## The token streaming pipeline

Between the GPU and the client, each token passes through several stages:

```text
[GPU Forward Pass] --> [Token Decoder] --> [Safety Check] --> [SSE Formatter] --> [HTTP Writer]
                           |                    |
                      [Detokenizer]      [Buffer if needed]
```

The **GPU forward pass** produces logits for the next token. The **token decoder** samples a token ID, or picks the most likely one greedily. The **detokenizer** converts token IDs to text, handling multi-byte UTF-8 and partial tokens. The **safety check** makes a quick check on the generated content; it can run asynchronously or in batches. The **SSE formatter** wraps the text in a `content_block_delta` event. And the **HTTP writer** writes it to the client's response stream. Two of these stages hide real subtlety.

### Detokenization

Tokens do not always align with character boundaries. Often each token is valid text on its own, but a single non-ASCII character can be split across several tokens, each carrying one byte of its UTF-8 encoding:

```text
Token IDs: [15339, 318, 257, 1332]
Tokens:    ["Hello", " is", " a", " test"]  # Clean, each is valid text

Token IDs: [9906, 223, 122]
Tokens:    ["<0xE4>", "<0xBD>", "<0xA0>"]   # UTF-8 bytes of a Chinese character
```

You cannot send partial UTF-8 bytes; the client would render garbage or fail to decode. The detokenizer must buffer bytes until they form a complete character and only then emit it.

### Speculative safety checking

Once text is sent, it cannot be unsent, so a safety check must run *before* tokens leave the server. Speculative checking does this in four steps: generate N tokens ahead (using a small buffer, or speculative decoding); run the safety classifier on the buffered tokens; release them to the client if they are safe; and if they are not, truncate the response and send `stop_reason: "safety"`. This adds N tokens of latency to the stream but prevents harmful content from being delivered.

## Connection management

A streaming connection lives for seconds or minutes, which exposes it to failures a short request never meets.

### Heartbeats

Intermediaries such as proxies, load balancers, and CDNs enforce idle timeouts and may silently drop an SSE connection that has gone quiet, for example while a long prompt is being prefilled. The fix is a heartbeat: a periodic SSE comment, a line starting with `:`, which the client ignores but which keeps the TCP connection active.

```python
async def stream_with_heartbeat(response, generator):
    """Send heartbeat comments to keep connection alive."""
    heartbeat_interval = 15  # seconds

    async def heartbeat():
        while True:
            await asyncio.sleep(heartbeat_interval)
            await response.write(": heartbeat\n\n")  # SSE comment

    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        async for token in generator:
            await response.write(f"event: content_block_delta\ndata: {json.dumps(token)}\n\n")
    finally:
        heartbeat_task.cancel()
```

The `finally` block cancels the heartbeat task however the stream ends, so a finished or failed stream does not leave a task writing to a closed response.

### Reconnection

SSE has reconnection built in. The server includes an `id:` field with each event. If the connection drops, the client reconnects automatically and sends a `Last-Event-ID` header with the last ID it received. The server then resumes from that point.

For LLM serving, this means the server must either buffer recent events or be able to reconstruct them, because the tokens were generated once and will not be generated identically again. A per-request event store is the simplest version:

```python
class EventStore:
    def __init__(self, max_events: int = 10000):
        self.events: dict[str, list[dict]] = {}  # request_id -> events

    def store(self, request_id: str, event_id: str, event: dict):
        if request_id not in self.events:
            self.events[request_id] = []
        self.events[request_id].append({"id": event_id, **event})

    def replay_from(self, request_id: str, last_event_id: str) -> list[dict]:
        events = self.events.get(request_id, [])
        for i, event in enumerate(events):
            if event["id"] == last_event_id:
                return events[i + 1:]
        return events  # ID not found, replay all
```

`replay_from` returns everything after the last acknowledged event, or the whole stream if the ID is unknown. As written, `max_events` is accepted but not enforced; a production store would cap each request's history at that size and expire completed requests, or the store itself becomes the unbounded buffer that backpressure was meant to prevent.

### Graceful shutdown

When a server instance must shut down for a deployment or a scale-down, it should drain rather than cut streams mid-sentence. Stop accepting new requests; send `message_stop` events to all active streams; wait, with a timeout, for clients to disconnect; and force-close whatever remains.

## Latency targets and SLOs

Streaming changes what "latency" means, so define the metrics precisely:

| Metric | Definition |
|--------|-----------|
| TTFT (Time to First Token) | Time from request received to first token sent |
| TPOT (Time Per Output Token) | Average time between consecutive tokens |
| TBT (Time Between Tokens) | Time between any two consecutive tokens (includes variance) |
| Total latency | TTFT + (num_tokens * TPOT) |

Typical targets for each are:

| Metric | Target | Rationale |
|--------|--------|-----------|
| TTFT p50 | < 100ms | Feels instant |
| TTFT p99 | < 500ms | Acceptable worst case |
| TPOT | < 20ms | ~50 tokens/sec, faster than reading speed |
| TBT p99 | < 100ms | No perceptible "stutter" in stream |
| Availability | 99.9% | ~8.7 hours downtime per year |
| Error rate | < 0.1% | Excluding rate limits |

### Where latency hides

To improve TTFT you need to know where the time goes. Trace a request from client to first token:

```text
Client request
    |-- Network (5-50ms)
    v
Load balancer
    |-- Routing (1-5ms)
    v
API Gateway
    |-- Auth + validation (5-20ms)
    v
Queue wait
    |-- Depends on load (0ms - seconds)
    v
Batch formation
    |-- Wait for batch (0-10ms)
    v
GPU forward pass (prefill)
    |-- First token compute (20-200ms, scales with input length)
    v
Safety check
    |-- Pre-send verification (5-20ms)
    v
SSE write
    |-- Serialization + network (1-5ms)
    v
Client receives first token
```

The optimization priorities follow from where the largest and most variable times are. First, **queue wait**: scale GPUs to keep the queue shallow, since it ranges from nothing to seconds. Second, **prefill latency**: optimize the attention implementation and share KV cache across common prefixes. Third, **batch formation delay**: use continuous batching with a short flush timeout. Fourth, **network**: keep servers geographically close to users and use HTTP/2 for multiplexing.

### Monitoring

A streaming LLM service needs dashboards for each of these: the TTFT distribution as a histogram with p50, p95, and p99 lines; token throughput per GPU and in aggregate; queue depth by priority; GPU memory utilization per GPU, tracking KV cache usage; the number of active SSE streams; the error rate by type (timeout, safety, OOM, upstream failure); and the rate and type of safety-filter triggers.

## Where this leads

The batching and KV cache behind the pipeline's first stage are in [GPU inference serving](gpu-inference.md), and the asyncio patterns the streaming code relies on are in [Concurrency patterns](concurrency-patterns.md). For a full OpenAI-compatible streaming server built from scratch, see [OpenAI-compatible HTTP and SSE on tokio](../../../build-an-llm/p10-serving/05-openai-server-on-tokio.md) in *Build Your Own LLM System*.

## Key terms

- **Server-Sent Events (SSE)**: a unidirectional HTTP streaming protocol of text events separated by blank lines.
- **`Last-Event-ID`**: the header a reconnecting SSE client sends so the server can resume after the last event received.
- **Backpressure**: limiting how far a producer can run ahead of a slow consumer, here by bounding the per-stream buffer.
- **Heartbeat**: a periodic SSE comment that keeps intermediaries from closing an idle connection.
- **TTFT, TPOT, TBT**: time to first token, average time per output token, and the time between any two tokens including its variance.

## Summary

- SSE fits LLM streaming because the flow is unidirectional, it runs over standard HTTP, and reconnection is built in.
- Anthropic's typed events (message start, content block start, deltas, block stop, message delta, message stop) support structured parsing, tool use, token counting, and multiple blocks.
- TCP flow control throttles a slow client unless a buffering proxy hides it; then bound the buffer and drop the connection, letting the client resume.
- Detokenizers must buffer partial UTF-8; speculative safety checks hold back N tokens so unsafe text is never sent.
- Heartbeats, an event store for `Last-Event-ID`, and draining on shutdown keep long-lived streams reliable.
- TTFT is dominated by queue wait and prefill, so scale to keep queues shallow and share prefix KV cache.

## Review questions

1. Why is SSE usually a better choice than WebSockets for streaming LLM output?
   <details><summary>Answer</summary>The traffic is unidirectional after the prompt is sent. SSE runs over ordinary HTTP, works with standard load balancers, and has reconnection with <code>Last-Event-ID</code> built in, while WebSockets need a protocol upgrade, special proxy configuration, and manual reconnection.</details>
2. A client is behind a buffering proxy and reads slowly. Why does TCP flow control not protect the server, and what should the server do?
   <details><summary>Answer</summary>The proxy reads from the server quickly and buffers on the client's behalf, so the server's socket never fills. The server needs its own bounded per-stream buffer and should drop the connection when it overflows, letting the client reconnect with <code>Last-Event-ID</code>.</details>
3. Why is blocking the producer a poor response to a full stream buffer in LLM serving?
   <details><summary>Answer</summary>Pausing token generation for one request leaves its slot in the GPU batch doing nothing, wasting compute that other requests could use.</details>
4. The model emits three tokens that are the three UTF-8 bytes of one character. What must the detokenizer do?
   <details><summary>Answer</summary>Buffer the bytes and emit nothing until they form a complete UTF-8 character, then send the character as one delta. Sending partial bytes would produce invalid text on the client.</details>
5. What does the `EventStore` need to support reconnection, and what is missing from the version shown?
   <details><summary>Answer</summary>It stores each request's events with IDs and replays everything after a given <code>Last-Event-ID</code>. It does not enforce <code>max_events</code> or expire finished requests, so in production it would grow without bound.</details>
6. TTFT p99 has doubled under load while prefill time is unchanged. Where would you look first?
   <details><summary>Answer</summary>Queue wait, which ranges from nothing to seconds and grows with load. Scaling GPUs to keep the queue shallow is the highest-impact fix, followed by checking batch-formation delay.</details>
