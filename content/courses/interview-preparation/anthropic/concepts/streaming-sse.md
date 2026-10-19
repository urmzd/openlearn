# Streaming & Server-Sent Events: Deep Dive

Reference material for real-time streaming concepts relevant to LLM inference serving.

## SSE Protocol Basics

Server-Sent Events (SSE) is a simple HTTP-based protocol for server-to-client streaming over a persistent connection.

### How It Works

1. Client sends a regular HTTP GET request with `Accept: text/event-stream`
2. Server responds with `Content-Type: text/event-stream` and keeps the connection open
3. Server sends events as they occur, each followed by a blank line
4. Connection is unidirectional: server -> client only

### Event Format

```
event: <event-type>
data: <payload>
id: <event-id>
retry: <reconnection-time-ms>

```

- `event:` -- Optional event type. Client can listen for specific types.
- `data:` -- The payload. Multiple `data:` lines are joined with newlines.
- `id:` -- Optional event ID for reconnection support.
- `retry:` -- Optional reconnection delay hint.
- Blank line terminates each event.

### Why SSE Over WebSocket for LLM Serving

| Feature | SSE | WebSocket |
|---------|-----|-----------|
| Direction | Server -> Client | Bidirectional |
| Protocol | HTTP/1.1 or HTTP/2 | Custom upgrade |
| Reconnection | Built-in (Last-Event-ID) | Manual |
| Load balancer support | Standard HTTP | Needs special config |
| Simplicity | Very simple | More complex |
| Use case fit | Token streaming (unidirectional) | Chat (bidirectional) |

LLM inference is fundamentally unidirectional: client sends a prompt, server streams tokens back. SSE is the simpler, better-supported choice.

## Anthropic's Typed SSE Events

Anthropic's API uses structured, typed events for streaming responses:

### Event Sequence

```
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

### Event Examples

```
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

### Why Typed Events?

- **Structured parsing**: Client knows exactly what each event means
- **Tool use support**: `content_block_start` with `type: "tool_use"` enables streaming tool calls
- **Progress tracking**: `usage` in `message_start` and `message_delta` allows real-time token counting
- **Multi-block support**: A single message can have multiple content blocks (text + tool_use interleaved)

## Backpressure and Flow Control

### The Problem

Token generation on a GPU (50-200 tokens/sec) can outpace a slow client's ability to consume them. Without backpressure, the server buffers unboundedly, exhausting memory.

### TCP-Level Backpressure

TCP already provides flow control:
1. Server writes to socket
2. If client is slow, TCP receive window fills up
3. Server's `send()` blocks (or returns short write)
4. This naturally throttles the server

For SSE over HTTP/1.1, this works automatically. The server's write call blocks when the client can't keep up.

### Application-Level Backpressure

When TCP backpressure isn't sufficient (e.g., with buffering proxies):

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

Strategies when buffer is full:
- **Drop connection**: Cleanest option. Client reconnects.
- **Drop events**: Lossy, but some events (heartbeats) can be dropped.
- **Block producer**: Pauses token generation for this request. Wastes GPU time.

Best practice for LLM serving: **Drop connection** after buffer exceeds threshold. The client's `Last-Event-ID` allows resumption.

## Token-by-Token Streaming Architecture

### Server-Side Pipeline

```
[GPU Forward Pass] --> [Token Decoder] --> [Safety Check] --> [SSE Formatter] --> [HTTP Writer]
                           |                    |
                      [Detokenizer]      [Buffer if needed]
```

1. **GPU Forward Pass**: Produces logits for next token
2. **Token Decoder**: Samples or greedily selects token ID
3. **Detokenizer**: Converts token ID to text (handles multi-byte UTF-8, partial tokens)
4. **Safety Check**: Quick check on generated content (can be async/batched)
5. **SSE Formatter**: Wraps text in `content_block_delta` event
6. **HTTP Writer**: Writes to the client's HTTP response stream

### Detokenization Challenges

Tokens don't always map to clean text boundaries:

```
Token IDs: [15339, 318, 257, 1332]
Tokens:    ["Hello", " is", " a", " test"]  # Clean, each is valid text

Token IDs: [9906, 223, 122]
Tokens:    ["<0xE4>", "<0xBD>", "<0xA0>"]   # UTF-8 bytes of a Chinese character
```

You can't send partial UTF-8 bytes. The detokenizer must buffer until a complete character is formed.

### Speculative Safety Checking

To avoid sending unsafe content:

1. Generate N tokens ahead (speculative decoding or small buffer)
2. Run safety classifier on the buffered tokens
3. If safe, release to client
4. If unsafe, truncate and send `stop_reason: "safety"`

This adds N-token latency to the stream but prevents sending harmful content that can't be un-sent.

## Connection Management

### Keep-Alive and Heartbeats

SSE connections can be silently dropped by intermediaries (proxies, load balancers, CDNs) that have idle timeouts.

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

SSE comments (lines starting with `:`) are ignored by the client but keep the TCP connection active.

### Reconnection

SSE has built-in reconnection support:

1. Server includes `id:` field with each event
2. If connection drops, client automatically reconnects
3. Client sends `Last-Event-ID` header with the last received ID
4. Server resumes from that point

For LLM serving, this means the server must buffer recent events or be able to reconstruct them:

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

### Graceful Shutdown

When a server instance needs to shut down (deployment, scaling):

1. Stop accepting new requests
2. Send `message_stop` events to all active streams
3. Wait for clients to disconnect (with timeout)
4. Force-close remaining connections

## Latency Targets and SLOs

### Definitions

| Metric | Definition |
|--------|-----------|
| TTFT (Time to First Token) | Time from request received to first token sent |
| TPOT (Time Per Output Token) | Average time between consecutive tokens |
| TBT (Time Between Tokens) | Time between any two consecutive tokens (includes variance) |
| Total latency | TTFT + (num_tokens * TPOT) |

### Typical SLOs

| Metric | Target | Rationale |
|--------|--------|-----------|
| TTFT p50 | < 100ms | Feels instant |
| TTFT p99 | < 500ms | Acceptable worst case |
| TPOT | < 20ms | ~50 tokens/sec, faster than reading speed |
| TBT p99 | < 100ms | No perceptible "stutter" in stream |
| Availability | 99.9% | ~8.7 hours downtime per year |
| Error rate | < 0.1% | Excluding rate limits |

### Where Latency Hides

```
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

**Optimization priorities** (biggest impact first):
1. Queue wait time -- Scale GPUs to keep queue shallow
2. Prefill latency -- Optimize attention implementation, use KV cache sharing for common prefixes
3. Batch formation delay -- Use continuous batching with short flush timeout
4. Network -- Keep servers geographically close to users, use HTTP/2 for multiplexing

### Monitoring

Key dashboards for a streaming LLM service:

- **TTFT distribution**: Histogram with p50/p95/p99 lines
- **Token throughput**: Tokens/sec per GPU, aggregate
- **Queue depth**: Current pending requests, by priority
- **GPU memory utilization**: Per-GPU, track KV cache usage
- **Active streams**: Current open SSE connections
- **Error rate**: By type (timeout, safety, OOM, upstream failure)
- **Safety filter triggers**: Rate and type of content filtered
