# GPU Inference Serving: Deep Dive

Reference material for understanding GPU-based LLM inference infrastructure.

## Model Serving Architecture

```
[Request Queue] --> [Scheduler] --> [GPU Worker Pool]
                                        |
                                   [GPU 0] [GPU 1] ... [GPU N]
                                        |
                                   [KV Cache Manager]
                                        |
                                   [Output Buffer] --> [Streaming Layer]
```

### Components

1. **Request Queue**: Buffered, prioritized queue of pending inference requests
2. **Scheduler**: Decides which requests to batch together and which GPU to assign
3. **GPU Worker**: Loads model weights, runs forward passes, manages local memory
4. **KV Cache Manager**: Allocates, evicts, and shares KV cache across requests
5. **Output Buffer**: Collects generated tokens for streaming or batched return

## Batch Size vs. Throughput vs. Safety Capacity

Batch size is the primary knob for trading latency against throughput.

```
Throughput (tokens/sec)
    ^
    |           ___________
    |         /
    |       /
    |     /
    |   /
    | /
    +------------------------> Batch Size

Latency (ms/token)
    ^
    |                    /
    |                  /
    |               /
    |          ___/
    |    ____/
    |___/
    +------------------------> Batch Size
```

- **Small batches** (1-4): Low latency, low throughput, GPU underutilized
- **Medium batches** (8-32): Good latency-throughput balance
- **Large batches** (64+): High throughput, but latency climbs and memory pressure increases

**Safety capacity consideration**: Larger batches leave less GPU headroom for parallel safety checks. If running a safety classifier alongside generation, reserve 10-20% of GPU compute.

## Continuous Batching

Traditional (static) batching waits for all requests in a batch to finish before starting a new batch. This wastes GPU cycles when requests have different lengths.

**Continuous batching** (also called iteration-level batching):

```
Time -->
Step 1: [A(tok5), B(tok3), C(tok1)]
Step 2: [A(tok6), B(tok4), D(tok1)]    # C finished, D inserted
Step 3: [A(tok7), B(tok5), D(tok2)]
Step 4: [A(tok8), E(tok1), D(tok3)]    # B finished, E inserted
```

Benefits:
- GPU never idles waiting for the longest request
- Time-to-first-token for new requests is bounded by a single iteration
- Throughput can increase 2-4x over static batching

Implementation requires:
- Per-request KV cache management (not shared batch allocation)
- Scheduler that runs between every iteration
- Ability to add/remove requests from an in-flight batch

## KV Cache Planning and Eviction

### Memory Budget

For a transformer with:
- `L` layers, `H` attention heads, `D` head dimension
- Sequence length `S`, batch size `B`

KV cache per request = `2 * L * H * D * S * sizeof(dtype)`

Example (70B parameter model, FP16):
- L=80, H=64, D=128, S=4096
- Per request: `2 * 80 * 64 * 128 * 4096 * 2 bytes` = ~10.7 GB

This is why KV cache is the dominant memory consumer and must be carefully managed.

### Eviction Policies

| Policy | Description | Pros | Cons |
|--------|-------------|------|------|
| LRU | Evict least recently accessed | Simple, good baseline | May evict long-running requests |
| Priority-based | Evict lowest-priority request | Respects business tiers | Priority inversion risk |
| Preemptive | Pause and evict a running request | Flexible | Recomputation cost on resume |
| Size-aware | Evict largest cache first | Frees most memory | May unfairly penalize long contexts |

### PagedAttention (vLLM)

Traditional KV cache allocation reserves contiguous memory for max_tokens upfront, leading to fragmentation and waste.

PagedAttention:
- Divides KV cache into fixed-size **blocks** (pages), typically 16 tokens each
- Pages allocated on demand as tokens are generated
- Non-contiguous pages mapped via a page table (like OS virtual memory)
- Shared pages for common prefixes (system prompts)

Result: near-zero waste, memory utilization approaches 100%.

## Warm GPU Pools vs. Autoscaling

### The Cold Start Problem

Loading a 70B model onto a GPU:
1. Transfer weights from CPU/storage to GPU memory: 30-60 seconds
2. CUDA kernel compilation/warmup: 5-10 seconds
3. First inference is slower due to cache cold start

Total cold start: **40-90 seconds** -- unacceptable for real-time API serving.

### Warm Pool Strategy

Maintain a pool of pre-loaded GPU instances:

```
Traffic Prediction --> Warm Pool Sizer
                            |
                   [Warm Pool: N instances with model loaded]
                            |
                   [Cold Pool: M instances on standby, no model]
```

- **Warm pool**: Model loaded, ready for inference in < 100ms
- **Cold pool**: Instance running but model not loaded. Can serve in 40-90s.
- **Scale-from-zero**: Instance not running. Minutes to serve.

### Sizing the Warm Pool

```
warm_pool_size = peak_expected_requests / requests_per_gpu + buffer
buffer = max(2, 0.2 * warm_pool_size)  # 20% headroom or at least 2
```

Adjust warm pool size based on:
- Time-of-day traffic patterns
- Day-of-week patterns
- Upcoming events (product launches)
- Cost constraints (warm GPUs are expensive even when idle)

### Elastic Scaling Signals

| Signal | Description | Responsiveness |
|--------|-------------|---------------|
| Queue depth | Number of pending requests | Real-time, best for burst detection |
| GPU utilization | Compute usage percentage | Lagging, can be misleading |
| Estimated token queue | Sum of estimated tokens in queue | Best overall signal |
| TTFT percentile | Time-to-first-token p95 | Directly measures user experience |

Best practice: Use **estimated token queue** as primary signal with **TTFT p95** as a guardrail. If TTFT exceeds SLO, scale up regardless of other signals.

## Model Serving Layer Architecture

### Multi-Model Serving

When serving multiple model versions (e.g., Claude Sonnet, Opus, Haiku):

```
[Request Router]
    |
    +--> [Haiku Pool]    (smaller model, more instances, lower cost)
    +--> [Sonnet Pool]   (medium model, medium instances)
    +--> [Opus Pool]     (largest model, fewest instances, highest cost)
```

Each pool scales independently based on its own traffic patterns.

### Canary Deployments

Rolling out a new model version:

1. Deploy new version to 1-5% of traffic
2. Monitor: latency, error rate, safety metrics, user satisfaction
3. Gradually increase: 5% -> 25% -> 50% -> 100%
4. Rollback criteria: p99 latency increase > 20%, error rate > 0.1%, any safety regression

### Health Checks

GPU workers need specialized health checks:

- **Liveness**: Is the process running? (basic)
- **Readiness**: Is the model loaded and warm? (model-specific)
- **Inference check**: Can the worker produce a valid output for a canary prompt? (end-to-end)
- **Memory check**: Is GPU memory utilization below critical threshold? (prevents OOM)

Run inference checks every 30-60 seconds. Kill and replace workers that fail 3 consecutive checks.

## Key Numbers to Know

| Metric | Typical Value |
|--------|---------------|
| GPU memory (A100) | 80 GB HBM2e |
| GPU memory (H100) | 80 GB HBM3 |
| NVLink bandwidth | 900 GB/s (H100) |
| PCIe bandwidth | 64 GB/s (Gen5) |
| Model load time (70B, FP16) | 30-60 seconds |
| KV cache per token (70B) | ~2.6 MB |
| Tokens per second per GPU | 50-200 (depends on batch size) |
| Cost per GPU-hour (A100) | ~$2-4 (cloud) |
| Cost per GPU-hour (H100) | ~$4-8 (cloud) |
