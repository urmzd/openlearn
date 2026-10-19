# GPU Inference Serving

Serving a large language model is a resource-allocation problem with one dominant constraint: GPU memory. The weights take most of it before the first request arrives, every in-flight request claims more for its KV cache as it generates, and loading a model onto a fresh GPU takes the better part of a minute. Every design decision in an inference stack, from how requests are batched to how many GPUs sit idle and warm, is a way of living within that constraint while meeting a latency target.

This section is the background for the inference questions in [the system design round](../03-system-design.md). It starts with the serving architecture and its components, then examines batch size as the main latency-throughput knob, continuous batching, the arithmetic of KV cache memory and the policies for evicting it, PagedAttention, the cold-start problem and warm pools, scaling signals, and the serving layer around the GPUs: multi-model pools, canary deployments, and health checks. It ends with the numbers you should be able to quote.

> **Learning objectives**
> - Describe the components of an LLM serving stack and how a request moves through them.
> - Explain how batch size trades latency against throughput, and how continuous batching removes the waste of static batching.
> - Compute KV cache size per request and per token from model dimensions, and compare eviction policies.
> - Explain PagedAttention and why it raises memory utilization toward 100%.
> - Size a warm pool, choose scaling signals, and design canary rollouts and health checks for GPU workers.

## The serving architecture

A serving stack is a pipeline from a queue of requests to a stream of tokens:

```text
[Request Queue] --> [Scheduler] --> [GPU Worker Pool]
                                        |
                                   [GPU 0] [GPU 1] ... [GPU N]
                                        |
                                   [KV Cache Manager]
                                        |
                                   [Output Buffer] --> [Streaming Layer]
```

The **request queue** is a buffered, prioritized queue of pending inference requests. The **scheduler** decides which requests to batch together and which GPU to assign them to. Each **GPU worker** loads the model weights, runs forward passes, and manages its local memory. The **KV cache manager** allocates, evicts, and shares KV cache across requests. And the **output buffer** collects generated tokens for streaming or for a batched return. The rest of this section is about the decisions those components make.

## Batch size, throughput, and safety capacity

Batch size is the primary knob for trading latency against throughput. As the batch grows, throughput rises and then plateaus once the GPU is saturated, while per-token latency stays flat for a while and then climbs:

```text
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

In practice that gives three regimes. **Small batches** (1 to 4) have low latency and low throughput and leave the GPU underutilized. **Medium batches** (8 to 32) strike a good balance between latency and throughput. **Large batches** (64 and up) maximize throughput, but latency climbs and memory pressure increases.

There is also a safety consideration. Larger batches leave less GPU headroom for parallel safety checks, so if you run a safety classifier alongside generation, reserve 10 to 20% of GPU compute for it.

## Continuous batching

Static batching waits for every request in a batch to finish before starting the next batch. Because generated sequences have very different lengths, the GPU wastes cycles on slots whose requests are already done.

**Continuous batching**, also called iteration-level batching, fixes this by rescheduling between every decoding step. When a request finishes, a waiting request takes its slot at the next iteration:

```text
Time -->
Step 1: [A(tok5), B(tok3), C(tok1)]
Step 2: [A(tok6), B(tok4), D(tok1)]    # C finished, D inserted
Step 3: [A(tok7), B(tok5), D(tok2)]
Step 4: [A(tok8), E(tok1), D(tok3)]    # B finished, E inserted
```

The benefits are substantial. The GPU never idles waiting for the longest request. A new request's time to first token is bounded by a single iteration rather than by the remaining length of the current batch. And throughput can increase two to four times over static batching. The cost is in the implementation: it requires per-request KV cache management rather than one allocation shared by the batch, a scheduler that runs between every iteration, and the ability to add and remove requests from a batch that is already in flight.

## KV cache planning and eviction

Continuous batching keeps the GPU busy; the KV cache determines how many requests can be in the batch at all.

### The memory budget

For a transformer with $L$ layers, $H$ attention heads, head dimension $D$, and sequence length $S$, the KV cache for one request is

$$
\text{KV bytes per request} = 2 \cdot L \cdot H \cdot D \cdot S \cdot \text{sizeof(dtype)}
$$

The factor of 2 is for keys and values. A batch of $B$ requests needs $B$ times that.

Work it for a 70B-parameter model in FP16, with $L = 80$, $H = 64$, $D = 128$, and $S = 4096$:

$$
2 \cdot 80 \cdot 64 \cdot 128 \cdot 4096 \cdot 2 \text{ bytes} \approx 10.7 \text{ GB per request}
$$

Divide by $S$ and you get about 2.6 MB per token. That is why the KV cache is the dominant memory consumer once the weights are loaded, and why it must be managed carefully.

> **Note:** Many current large models use grouped-query attention, in which $H$ in this formula is the number of *key-value* heads rather than query heads. With 8 KV heads instead of 64, the same model needs about 1.3 GB per 4096-token request. Ask which attention variant the interviewer has in mind before you size memory.

### Eviction policies

When memory runs out, something must be evicted, and each policy has a different failure mode:

| Policy | Description | Pros | Cons |
|--------|-------------|------|------|
| LRU | Evict least recently accessed | Simple, good baseline | May evict long-running requests |
| Priority-based | Evict lowest-priority request | Respects business tiers | Priority inversion risk |
| Preemptive | Pause and evict a running request | Flexible | Recomputation cost on resume |
| Size-aware | Evict largest cache first | Frees most memory | May unfairly penalize long contexts |

### PagedAttention

Traditional KV cache allocation reserves contiguous memory for `max_tokens` up front. Most requests never use all of it, and the variable-sized contiguous regions fragment memory, so a large fraction is wasted.

**PagedAttention**, from vLLM, borrows the idea of virtual memory from operating systems. It divides the KV cache into fixed-size **blocks** (pages), typically 16 tokens each, and allocates pages on demand as tokens are generated. A page table maps each request's logical sequence to non-contiguous physical pages, and requests with a common prefix, such as a shared system prompt, can share the pages for that prefix. The result is near-zero waste, with memory utilization approaching 100%.

## Warm pools and autoscaling

Memory determines how much one GPU can serve. Autoscaling determines how many GPUs you have, and for LLMs it is dominated by the cold-start problem.

### The cold-start problem

Loading a 70B model onto a GPU happens in three stages. Transferring the weights from CPU memory or storage to GPU memory takes 30 to 60 seconds. CUDA kernel compilation and warm-up take another 5 to 10 seconds. And the first inference is slower still, because caches are cold. The total cold start is **40 to 90 seconds**, which is unacceptable for real-time API serving.

### The warm pool

The answer is to keep capacity ready ahead of demand. A traffic predictor drives a warm-pool sizer, which maintains a tier of GPUs with the model loaded and a tier of standby instances without it:

```text
Traffic Prediction --> Warm Pool Sizer
                            |
                   [Warm Pool: N instances with model loaded]
                            |
                   [Cold Pool: M instances on standby, no model]
```

That yields three tiers of readiness. A **warm** instance has the model loaded and can serve in under 100 ms. A **cold** instance is running but has no model loaded, so it can serve in 40 to 90 seconds. And **scale-from-zero**, where the instance is not even running, takes minutes.

Size the warm pool from expected peak demand plus headroom:

```text
warm_pool_size = peak_expected_requests / requests_per_gpu + buffer
buffer = max(2, 0.2 * warm_pool_size)  # 20% headroom or at least 2
```

Then adjust it for time-of-day patterns, day-of-week patterns, upcoming events such as product launches, and cost constraints, since warm GPUs are expensive even when idle.

### Scaling signals

The pool needs a signal to scale on, and the candidates differ in how well they reflect demand:

| Signal | Description | Responsiveness |
|--------|-------------|---------------|
| Queue depth | Number of pending requests | Real-time, best for burst detection |
| GPU utilization | Compute usage percentage | Lagging, can be misleading |
| Estimated token queue | Sum of estimated tokens in queue | Best overall signal |
| TTFT percentile | Time-to-first-token p95 | Directly measures user experience |

The best practice is to use the **estimated token queue** as the primary signal, with **TTFT p95** as a guardrail: if TTFT exceeds the SLO, scale up regardless of what the other signals say.

## The serving layer around the GPUs

The last layer turns a pool of GPU workers into a service you can operate safely.

### Serving several models

When you serve several model sizes, such as Claude Haiku, Sonnet, and Opus, a router sends each request to a dedicated pool:

```text
[Request Router]
    |
    +--> [Haiku Pool]    (smaller model, more instances, lower cost)
    +--> [Sonnet Pool]   (medium model, medium instances)
    +--> [Opus Pool]     (largest model, fewest instances, highest cost)
```

Each pool scales independently, according to its own traffic patterns.

### Canary deployments

A new model version is rolled out gradually. Deploy it to 1 to 5% of traffic first, and monitor latency, error rate, safety metrics, and user satisfaction. Then increase step by step, from 5% to 25% to 50% to 100%. Define rollback criteria in advance: a p99 latency increase above 20%, an error rate above 0.1%, or any safety regression.

### Health checks

GPU workers need more specialized health checks than an ordinary web service. A **liveness** check asks whether the process is running. A **readiness** check asks whether the model is loaded and warm. An **inference check** asks whether the worker can produce a valid output for a canary prompt, end to end. And a **memory check** asks whether GPU memory utilization is below a critical threshold, which prevents out-of-memory failures. Run the inference check every 30 to 60 seconds, and kill and replace any worker that fails three consecutive checks.

## Numbers to know

Quantifying is half of a good design answer. These are the typical values to have ready:

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

## Where this leads

These concepts are applied in [the system design round](../03-system-design.md), and the token streaming that follows the output buffer is covered in [Streaming and SSE](streaming-sse.md). For a ground-up implementation of continuous batching and a paged KV cache, see [Continuous batching scheduler with priority and preemption](../../../build-an-llm/p10-serving/02-continuous-batching.md) and [Block manager with prefix cache](../../../build-an-llm/p10-serving/04-block-manager-and-prefix-cache.md) in *Build Your Own LLM System*.

## Key terms

- **Continuous (iteration-level) batching**: rescheduling the batch between every decoding step so finished requests are replaced immediately.
- **KV cache**: per-request storage of attention keys and values, sized $2 \cdot L \cdot H \cdot D \cdot S \cdot \text{sizeof(dtype)}$.
- **PagedAttention**: allocating the KV cache in fixed-size pages mapped by a page table, with shared pages for common prefixes.
- **Cold start**: the 40-to-90-second delay to load weights and warm kernels on a fresh GPU.
- **Warm pool**: instances kept with the model loaded so they can serve in under 100 ms.
- **Canary deployment**: routing a small, growing fraction of traffic to a new version, with predefined rollback criteria.

## Summary

- GPU memory is the binding constraint: weights first, then a KV cache that grows with every in-flight token.
- Batch size trades latency for throughput; continuous batching removes the idle slots of static batching and can multiply throughput two to four times.
- A 70B FP16 model with 64 heads needs about 10.7 GB of KV cache per 4096-token request (about 2.6 MB per token); PagedAttention eliminates most allocation waste.
- Cold starts take 40 to 90 seconds, so warm pools sized from predicted peak plus headroom are essential.
- Scale on the estimated token queue with TTFT p95 as a guardrail; roll out models by canary and check workers with an end-to-end inference probe.

## Review questions

1. Compute the KV cache per token for a model with $L = 80$, $H = 64$, $D = 128$ in FP16.
   <details><summary>Answer</summary>$2 \cdot 80 \cdot 64 \cdot 128 \cdot 2 = 2{,}621{,}440$ bytes, about 2.6 MB per token. For 4096 tokens that is about 10.7 GB.</details>
2. Why does continuous batching reduce time-to-first-token for new requests?
   <details><summary>Answer</summary>The scheduler runs between every iteration, so a new request can join as soon as any slot frees up. Its wait is bounded by one iteration rather than by the time for the longest request in a static batch to finish.</details>
3. What problem does PagedAttention solve, and what idea does it borrow?
   <details><summary>Answer</summary>Reserving contiguous memory for <code>max_tokens</code> per request wastes and fragments memory. PagedAttention borrows virtual memory: fixed-size pages allocated on demand and mapped by a page table, with prefix pages shared between requests.</details>
4. The predicted peak is 400 concurrent requests and each GPU serves 16. How large should the warm pool be?
   <details><summary>Answer</summary>The base is $400 / 16 = 25$ instances. The buffer is $\max(2, 0.2 \cdot 25) = 5$, so about 30 warm instances.</details>
5. Why is GPU utilization a poor primary scaling signal, and what guardrail would you add to the estimated token queue?
   <details><summary>Answer</summary>Utilization lags and can be high whether the GPU has two requests or two thousand, so it does not measure unmet demand. Use TTFT p95 as a guardrail: if it exceeds the SLO, scale up regardless of the other signals.</details>
6. Why is a liveness check not enough for a GPU worker?
   <details><summary>Answer</summary>The process can be running while the model is not loaded, the GPU is near out-of-memory, or inference produces bad output. Readiness, memory, and end-to-end inference checks catch those states.</details>
