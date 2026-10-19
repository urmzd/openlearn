# The System Design Round

The system design round is the centerpiece of the Anthropic infrastructure loop, because it maps most directly onto what the infrastructure team does every day. You get 60 minutes on a virtual whiteboard or shared document, and the subject is almost always some combination of LLM inference infrastructure, distributed systems, and real-time streaming. The interviewer is not looking for a memorized diagram. They want to watch you clarify a problem, sketch a whole system, quantify its constraints, and reason about tradeoffs when they push on one corner of it.

This section first lays out a time budget for the hour, then works through the most likely prompt, the infrastructure behind an LLM inference API, in depth: request flow, GPU memory, batching, the KV cache, autoscaling, streaming, and safety. It then covers two other reported prompts, a distributed search system and training infrastructure, and ends with the numeric targets you should be able to quote.

> **Learning objectives**
> - Run a 60-minute design conversation with a deliberate time budget, from requirements to operations.
> - Design an LLM inference API end to end, from the HTTP request to streamed tokens.
> - Size GPU memory for weights and KV cache, and explain static versus continuous batching and PagedAttention.
> - Choose an autoscaling signal for GPU serving and justify warm pools.
> - Place safety filtering in a streaming pipeline without destroying latency.
> - Sketch a sharded search system and a training system, naming the key tradeoffs in each.

## Budgeting the hour

A design round goes wrong most often through pacing, so plan the hour before you start. Spend about five minutes **clarifying requirements**: scale, latency targets, and consistency needs. Spend about ten on the **high-level architecture**: components, data flow, and APIs. The bulk, about thirty minutes, is the **deep dive**, where the interviewer will steer you toward one or two areas. Reserve about ten minutes for **operational concerns** (monitoring, failure modes, scaling) and the last five for your questions.

## Designing an LLM inference API

The most likely prompt is to design the infrastructure behind Anthropic's Claude API, from HTTP request to token output. Typical requirements are a time-to-first-token (TTFT) under 200 ms, 99.9% availability, support for both streaming (SSE) and non-streaming responses, graceful degradation under bursty traffic, and safety filtering that does not destroy latency.

### The architecture and the request flow

Start with the whole picture before diving into any one box.

```text
Client Request
    |
    v
[Load Balancer] --> [API Gateway]
    |                    |
    |              [Auth / Rate Limit]
    |                    |
    v                    v
[Request Router] --> [Priority Queue]
    |                    |
    v                    v
[GPU Cluster]      [Batch Scheduler]
    |                    |
    v                    v
[Model Worker]     [KV Cache Manager]
    |
    v
[Safety Filter] --> [SSE Streaming Layer] --> Client
```

Walk the interviewer through it in request order. The **API gateway** handles authentication, rate limiting, and request validation. The **request router** sends each request to the right model version and GPU pool based on its properties: the model, `max_tokens`, and priority. Requests then wait in a **priority queue**, where paid tiers get higher priority and the queue depth doubles as an autoscaling signal. The **batch scheduler** groups queued requests into batches for efficient GPU utilization. The **model worker** runs inference on the GPU and produces tokens. The **safety filter** applies Constitutional AI checks to the generated content, and it can run in parallel with generation. Finally the **streaming layer** converts token output into SSE events and manages the connection's lifecycle.

### GPU memory

The deep dive usually starts with the GPU, because GPU memory is the scarcest resource in the system. It must hold four things with different lifecycles:

| Component | Memory | Lifecycle |
|-----------|--------|-----------|
| Model weights | Fixed (~70GB for large model) | Permanent while serving |
| KV cache | Per-request, grows with sequence length | Duration of request |
| Activation memory | Per-batch, transient | Per forward pass |
| CUDA overhead | Fixed | Always |

The practical consequence is **admission control by memory planning**. Before admitting a request, estimate its peak KV cache size from `max_tokens`, and reject or queue the request if there is not enough memory. Capacity follows from simple arithmetic:

```text
Available GPU memory = Total - Weights - CUDA overhead
Max concurrent requests = Available / (avg KV cache per request)
```

### Batching

Memory bounds how many requests fit; batching decides how well you use the GPU while they are there.

**Static batching** waits for N requests or T milliseconds, whichever comes first, and runs them together. It is simple, but it wastes capacity: short sequences that finish early leave their slots idle until the longest request in the batch completes.

**Continuous batching** is the preferred approach. As soon as a request completes, a new one is slotted into its place at the next iteration, which maximizes GPU utilization. Requests are also grouped by similar sequence length to minimize padding waste.

```text
Iteration 1: [Req A (token 5), Req B (token 3), Req C (token 1)]
Iteration 2: [Req A (token 6), Req B (token 4), Req D (token 1)]  # C finished, D slotted in
```

Within either scheme there is a **flush-versus-hold** tradeoff. Flushing sends a batch as soon as any request is ready, which lowers latency for that request. Holding waits until the batch is full, which raises throughput at the cost of latency. Production systems use a hybrid: flush after a short timeout, around 5 ms.

### The KV cache

The key-value cache stores the attention keys and values for previously processed tokens, so each new token does not recompute them. Because it grows with sequence length and lives for the duration of a request, managing it is most of GPU memory management.

Three eviction policies are worth discussing. **LRU** evicts the least recently used cache entries when memory is full. **Prefix-aware** caching shares KV entries across requests with common prefixes, such as a shared system prompt. **Preemption** evicts the caches of lower-priority requests when high-priority requests arrive.

The technique to name is **PagedAttention**, from vLLM. It allocates the KV cache in fixed-size blocks (pages) rather than one contiguous region per request. That eliminates fragmentation and allows near-100% memory utilization, and pages can be shared across requests with common prefixes.

### Autoscaling

When the interviewer asks how the system scales, the key insight is the choice of signal. **Queue depth weighted by estimated token count** is a better scaling signal than raw GPU utilization. GPU utilization can read 90% with 2 requests or with 2000; queue depth captures actual demand.

```text
scaling_metric = sum(estimated_tokens(req) for req in queue) / total_gpu_capacity
scale_up_threshold = 0.7
scale_down_threshold = 0.2
```

The other half of scaling is **cold starts**. Loading a large model onto a GPU takes 30 to 60 seconds, far longer than any latency SLO can absorb. So you maintain a **warm pool** of pre-loaded GPU instances, which is expensive but critical for the SLO, and size it from time-of-day traffic patterns. Cold-start instances handle overflow and workloads that are not latency-sensitive.

### Streaming over SSE

Anthropic streams responses as typed Server-Sent Events. A response is a sequence of events that open the message, open a content block, deliver deltas, and close both:

```text
event: message_start
data: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-sonnet-4-20250514", ...}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_stop
data: {"type": "message_stop"}
```

Connection management is where the operational questions land. Send keep-alive heartbeat events so proxies and load balancers do not time out an idle stream. Support client-side reconnection with `Last-Event-ID` for resumability. And apply backpressure: if a client is slow to consume, buffer up to a limit and then drop the connection rather than letting buffers grow without bound. [Streaming and SSE](concepts/streaming-sse.md) covers all of this in depth.

### The safety layer

Safety is not optional in this round; always address where it fits. Constitutional AI constraints must be enforced without destroying latency, and a layered design does that. A **pre-generation filter** checks the input for known harmful patterns; it is fast and rule-based. A **parallel safety classifier** runs a lightweight safety model alongside generation and stops generation if it flags content. A **post-generation filter** makes a final check on the complete response before delivery.

Streaming makes the last layer hard, because tokens are sent as they are generated, before the response is complete. The answer is **speculative checking**: look ahead by a few tokens before sending. That adds a small delay but prevents unsafe content from reaching the client.

## Designing a distributed search system

A second reported prompt asks for a search system over one billion documents at one million queries per second. The shape is scatter-gather over shards:

```text
[Query API] --> [Query Router]
                    |
        +-----------+-----------+
        |           |           |
   [Shard 1]   [Shard 2]  [Shard N]
        |           |           |
        +-----------+-----------+
                    |
              [Merge Layer]
                    |
              [Result API]
```

The first decision is the **sharding strategy**. Hash-based sharding gives an even distribution but forces a scatter-gather across every shard for every query. Content-based sharding co-locates related documents so some queries hit fewer shards. A hybrid uses hashing for even distribution plus extra replicas for popular content.

Even distribution does not prevent **hot shards**, so monitor per-shard latency percentiles, replicate hot shards dynamically, and route each query to the least-loaded replica.

The **cross-shard merge** has each shard return its top-K results with scores and the merge layer perform a K-way merge. The subtle challenge is consistent scoring across shards: TF-IDF depends on corpus statistics, so shards need normalization or shared global statistics for their scores to be comparable.

For the **index structure**, use an inverted index for text search, with posting lists that carry skip pointers for fast intersection, and a tiered layout: a hot tier in memory, a warm tier on SSD, and a cold tier in object storage.

For **observability**, track per-shard latency (p50 and p99), index freshness (time since the last update per shard), query error rate by type, and cache hit rate.

The internals behind these choices, from postings and skip pointers to scatter-gather and its pagination cost, are covered in [Search and Indexing](../../infrastructure/04-search-and-indexing/README.md) in *Infrastructure*.

## Designing training infrastructure

The third reported prompt is infrastructure for training large language models. Begin with the parallelism strategies, because the choice among them drives everything else:

| Strategy | What's Distributed | When to Use |
|----------|-------------------|-------------|
| Data parallelism | Training data | Model fits on one GPU |
| Model/Tensor parallelism | Model layers | Model too large for one GPU |
| Pipeline parallelism | Model stages | Very deep models |
| Expert parallelism | MoE experts | Mixture-of-experts models |

In **data parallelism**, each GPU holds a full copy of the model and processes different data, and gradients are synchronized with AllReduce. Scaling is near-linear until communication becomes the bottleneck. In **model parallelism**, the model is split across GPUs (for example, layers 0 to 11 on GPU 0 and layers 12 to 23 on GPU 1). That requires a high-bandwidth interconnect such as NVLink or InfiniBand, and it introduces pipeline bubbles, periods when some GPUs sit idle while others compute.

At this scale, failures are routine, so **fault tolerance** is a first-class requirement. Checkpoint model state every N steps to distributed storage. Support elastic training, so workers can be added or removed without a restart. Use redundant gradient computation to tolerate spot-instance preemption. And use RDMA (remote direct memory access) to bypass the CPU for GPU-to-GPU communication, which is critical to gradient synchronization performance.

Finally, **gradient compression** reduces the cost of synchronizing full gradients across many GPUs. The techniques are quantization (FP32 to FP16 or BF16), top-K sparsification, and error feedback. The tradeoff is some loss of training quality in exchange for a significant communication speedup.

## Targets to know

Quantify everything in this round, and have these targets ready:

| Metric | Target | Why |
|--------|--------|-----|
| Time-to-first-token (TTFT) | < 200ms | User perception of responsiveness |
| Token throughput | > 50 tokens/sec | Readable streaming speed |
| Availability | 99.9% | Enterprise SLO requirement |
| Safety filter latency | < 50ms overhead | Can't double response time for safety |
| Cold start time | < 60s | GPU warm pool sizing |
| Batch utilization | > 80% | GPU cost efficiency |

## Preparing for the round

Draw the full picture first and resist diving into one component too early. Quantify everything: how much memory, how many GPUs, what the latency budget is. Know the Anthropic API: read its documentation and understand the SSE event types, rate limits, and error responses. Always address how safety filtering fits into your design. And favor tradeoffs over perfection: the interviewers want to see you reason about choices, not recite a perfect architecture.

## Where this leads

The serving concepts here are developed further in [GPU inference serving](concepts/gpu-inference.md), and the streaming layer in [Streaming and SSE](concepts/streaming-sse.md). For a ground-up build of an inference server, see the serving chapter of *Build Your Own LLM System*, starting with [Continuous batching scheduler with priority and preemption](../../build-an-llm/p10-serving/02-continuous-batching.md); for the infrastructure under it, see *Infrastructure*, starting from its [front matter](../../infrastructure/README.md).

## Key terms

- **Time-to-first-token (TTFT)**: the latency from receiving a request to emitting its first output token.
- **KV cache**: per-request storage of attention keys and values for already-processed tokens, so they are not recomputed.
- **Continuous batching**: replacing finished requests in a running batch at each iteration, rather than waiting for the whole batch to finish.
- **PagedAttention**: allocating the KV cache in fixed-size pages to eliminate fragmentation and allow prefix sharing.
- **Warm pool**: GPU instances with the model already loaded, kept ready to absorb load without a cold start.
- **Speculative checking**: holding back a few tokens of a stream so a safety check can run before they are sent.

## Summary

- Budget the hour: requirements, architecture, a 30-minute deep dive, operations, questions.
- An inference API is gateway, router, priority queue, batch scheduler, model worker, safety filter, and SSE layer, in that order.
- GPU memory holds weights, KV cache, activations, and CUDA overhead; admit requests by estimating peak KV cache from `max_tokens`.
- Continuous batching and PagedAttention keep GPUs full; scale on token-weighted queue depth, and use warm pools to hide 30-to-60-second cold starts.
- Layer safety (pre-filter, parallel classifier, post-filter) and use speculative look-ahead when streaming.
- Search is sharded scatter-gather with careful cross-shard scoring; training is a choice of parallelism plus checkpointing and fault tolerance.

## Review questions

1. Why is token-weighted queue depth a better autoscaling signal than GPU utilization for an inference service?
   <details><summary>Answer</summary>GPU utilization can read 90% whether the GPU is serving 2 requests or 2000, so it says little about unmet demand. Queue depth weighted by estimated tokens measures the work actually waiting, which is what you need to provision for.</details>
2. A GPU has 80 GB, the weights take 70 GB, CUDA overhead is 2 GB, and the average request's KV cache peaks at 0.5 GB. How many requests can run concurrently?
   <details><summary>Answer</summary>Available memory is $80 - 70 - 2 = 8$ GB, so about $8 / 0.5 = 16$ concurrent requests. That is why admission control by estimated KV size matters.</details>
3. Why does continuous batching beat static batching for LLM serving?
   <details><summary>Answer</summary>Generated sequences have very different lengths. In a static batch, slots of requests that finish early sit idle until the longest one completes. Continuous batching slots a new request into a freed slot at the next iteration, keeping the GPU busy.</details>
4. How can you apply a safety check to a response that is being streamed token by token?
   <details><summary>Answer</summary>Use speculative checking: buffer a few tokens ahead and run the check before sending them, alongside a parallel safety classifier that can stop generation. This costs a small delay but keeps unsafe content from reaching the client.</details>
5. In a hash-sharded search system, why can scores from different shards be hard to compare?
   <details><summary>Answer</summary>TF-IDF and BM25 depend on corpus statistics such as document frequency and average length. Each shard computes them over its own subset, so the same document could score differently on different shards. Global statistics or normalization make scores comparable before the merge.</details>
6. When would you choose pipeline parallelism over data parallelism, and what cost do you take on?
   <details><summary>Answer</summary>When the model is too large or too deep to fit on one GPU, you split it into stages across GPUs. You pay in pipeline bubbles, where some GPUs wait while others compute, and you need a high-bandwidth interconnect.</details>
