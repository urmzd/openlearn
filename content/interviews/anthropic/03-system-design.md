# Round 3: System Design

## Format

- **Duration**: 60 minutes
- **Setting**: Virtual whiteboard or shared doc
- **Focus**: LLM inference infrastructure, distributed systems, real-time streaming
- **This is the centerpiece round** -- it most directly maps to the day-to-day work of the infra team

## Approach

1. **Clarify requirements** (5 min) -- Ask about scale, latency targets, consistency needs
2. **High-level architecture** (10 min) -- Components, data flow, APIs
3. **Deep dive** (30 min) -- The interviewer will steer you toward 1-2 areas
4. **Operational concerns** (10 min) -- Monitoring, failure modes, scaling
5. **Questions** (5 min)

---

## Topic 1: LLM Inference API (Most Likely)

Design the infrastructure behind Anthropic's Claude API -- from HTTP request to token output.

### Requirements

- Sub-200ms time-to-first-token (TTFT)
- 99.9% availability
- Support streaming (SSE) and non-streaming responses
- Handle bursty traffic with graceful degradation
- Safety filtering without destroying latency

### High-Level Architecture

```
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

### Request Flow

1. **API Gateway**: Authentication, rate limiting, request validation
2. **Request Router**: Routes to appropriate model version / GPU pool based on request properties (model, max_tokens, priority)
3. **Priority Queue**: Requests queued with priority. Paid tiers get higher priority. Queue depth used for autoscaling signals.
4. **Batch Scheduler**: Groups requests into batches for efficient GPU utilization
5. **Model Worker**: Runs inference on GPU, produces tokens
6. **Safety Filter**: Constitutional AI checks on generated content (can run in parallel with generation)
7. **Streaming Layer**: Converts token output to SSE events, manages connection lifecycle

### GPU Memory Management

GPU memory is the scarcest resource. It must hold:

| Component | Memory | Lifecycle |
|-----------|--------|-----------|
| Model weights | Fixed (~70GB for large model) | Permanent while serving |
| KV cache | Per-request, grows with sequence length | Duration of request |
| Activation memory | Per-batch, transient | Per forward pass |
| CUDA overhead | Fixed | Always |

**Memory planning**: Before admitting a request, estimate its peak KV cache size based on `max_tokens`. Reject or queue if insufficient memory.

```
Available GPU memory = Total - Weights - CUDA overhead
Max concurrent requests = Available / (avg KV cache per request)
```

### Batching Strategies

#### Static Batching
- Wait for N requests or T milliseconds, whichever comes first
- Simple but wastes capacity on short sequences that finish early

#### Continuous Batching (Preferred)
- As a request completes, immediately slot in a new request
- Maximizes GPU utilization
- Requests grouped by similar sequence lengths to minimize padding waste

```
Iteration 1: [Req A (token 5), Req B (token 3), Req C (token 1)]
Iteration 2: [Req A (token 6), Req B (token 4), Req D (token 1)]  # C finished, D slotted in
```

#### Flush vs. Hold Trade-off
- **Flush**: Send a batch as soon as any request is ready (lower latency for that request)
- **Hold**: Wait until batch is full (higher throughput, higher latency)
- Production systems use a hybrid: flush after a short timeout (e.g., 5ms)

### KV Cache Management

The key-value cache stores attention computations for previously generated tokens, avoiding recomputation.

**Eviction policies**:
- **LRU**: Evict least recently used cache entries when memory is full
- **Prefix-aware**: Share KV cache across requests with common prefixes (system prompts)
- **Preemption**: Evict lower-priority request caches when high-priority requests arrive

**PagedAttention** (from vLLM):
- Allocate KV cache in fixed-size blocks (pages) rather than contiguous memory
- Eliminates fragmentation, allows near-100% memory utilization
- Pages can be shared across requests with common prefixes

### Autoscaling

**Scaling signal**: Queue depth weighted by estimated token count is better than raw GPU utilization.

Why? GPU utilization can be 90% with 2 requests or 2000 requests. Queue depth captures actual demand.

```
scaling_metric = sum(estimated_tokens(req) for req in queue) / total_gpu_capacity
scale_up_threshold = 0.7
scale_down_threshold = 0.2
```

**Warm pools vs. cold starts**:
- Loading a large model onto a GPU takes 30-60 seconds
- Maintain a warm pool of pre-loaded GPU instances (expensive but critical for SLO)
- Scale warm pool size based on time-of-day traffic patterns
- Cold start instances for overflow / non-latency-sensitive workloads

### Streaming / SSE

Anthropic uses typed Server-Sent Events for streaming responses:

```
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

**Connection management**:
- Keep-alive with heartbeat events to prevent proxy/LB timeouts
- Client-side reconnection with `Last-Event-ID` for resumability
- Backpressure: if the client is slow to consume, buffer up to a limit, then drop connection

### Safety Layer

Constitutional AI constraints must be enforced without destroying latency:

- **Pre-generation filter**: Check input for known harmful patterns (fast, rule-based)
- **Parallel safety classifier**: Run a lightweight safety model alongside generation. If it flags content, stop generation.
- **Post-generation filter**: Final check on complete response before delivery
- **Streaming challenge**: Tokens are sent as generated. Use speculative checking -- look ahead by a few tokens before sending. This adds a small delay but prevents sending unsafe content.

---

## Topic 2: Distributed Search System

Design a search system for 1B documents handling 1M QPS.

### Architecture

```
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

### Key Decisions

**Sharding strategy**:
- Hash-based: Even distribution, but scatter-gather for every query
- Content-based: Co-locate related documents, some queries hit fewer shards
- Hybrid: Hash for even distribution + replicas for popular content

**Hot shard avoidance**:
- Monitor per-shard latency percentiles
- Replicate hot shards dynamically
- Route to least-loaded replica

**Cross-shard merge**:
- Each shard returns top-K results with scores
- Merge layer performs a K-way merge
- Challenge: consistent scoring across shards (TF-IDF normalization, global statistics)

**Index structure**:
- Inverted index for text search
- Posting lists with skip pointers for fast intersection
- Tiered index: hot tier (in-memory), warm tier (SSD), cold tier (object storage)

**Observability**:
- Per-shard latency (p50, p99)
- Index freshness (time since last update per shard)
- Query error rate by type
- Cache hit rate

---

## Topic 3: Training Infrastructure

Design infrastructure for training large language models.

### Parallelism Strategies

| Strategy | What's Distributed | When to Use |
|----------|-------------------|-------------|
| Data parallelism | Training data | Model fits on one GPU |
| Model/Tensor parallelism | Model layers | Model too large for one GPU |
| Pipeline parallelism | Model stages | Very deep models |
| Expert parallelism | MoE experts | Mixture-of-experts models |

### Data Parallelism
- Each GPU has a full model copy, processes different data
- Gradient synchronization via AllReduce
- Scaling: near-linear up to communication bottleneck

### Model Parallelism
- Split model across GPUs (e.g., layers 0-11 on GPU 0, layers 12-23 on GPU 1)
- Requires high-bandwidth interconnect (NVLink, InfiniBand)
- Pipeline bubbles: some GPUs idle while others compute

### Fault Tolerance
- **Checkpointing**: Save model state every N steps to distributed storage
- **Elastic training**: Add/remove workers without restarting
- **Redundant gradient computation**: Spot instance preemption tolerance
- **RDMA (Remote Direct Memory Access)**: Bypass CPU for GPU-to-GPU communication, critical for gradient sync performance

### Gradient Compression
- Full gradients are expensive to synchronize across many GPUs
- Techniques: quantization (FP32 -> FP16/BF16), top-K sparsification, error feedback
- Trade-off: some training quality loss for significant communication speedup

---

## Targets to Know

| Metric | Target | Why |
|--------|--------|-----|
| Time-to-first-token (TTFT) | < 200ms | User perception of responsiveness |
| Token throughput | > 50 tokens/sec | Readable streaming speed |
| Availability | 99.9% | Enterprise SLO requirement |
| Safety filter latency | < 50ms overhead | Can't double response time for safety |
| Cold start time | < 60s | GPU warm pool sizing |
| Batch utilization | > 80% | GPU cost efficiency |

## Preparation Tips

1. **Draw the full picture first** -- Resist diving into one component too early
2. **Quantify everything** -- "How much memory?" "How many GPUs?" "What's the latency budget?"
3. **Know the Anthropic API** -- Read their API docs. Understand the SSE event types, rate limits, and error responses.
4. **Safety is not optional** -- Always address how safety filtering fits into your design
5. **Trade-offs over perfection** -- They want to see you reason about trade-offs, not recite a perfect architecture
