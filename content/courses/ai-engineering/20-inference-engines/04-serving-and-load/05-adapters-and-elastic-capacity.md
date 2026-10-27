# Adapters and elastic capacity

This part of [Serving, Capacity & Load Testing: Math to Code](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Account for cold starts and adapter working sets when planning replicas.

## Multi-LoRA, serverless versus dedicated, cold start, and autoscaling

The remaining decisions are about the fleet rather than one engine: how many models share a GPU, who pays for idle time, and how replicas come and go.

**Multi-LoRA** serving runs one base model with many adapters $\Delta W = BA$ of rank $r$. An adapter's size is $r \times (d_\text{in} + d_\text{out}) \times \text{modules} \times L \times b$. For Llama-3.1-8B with $r = 16$ on all 7 projections, that is about 42M parameters, 84 MB at BF16, versus 16 GB for a full fine-tune. [Punica](https://arxiv.org/abs/2310.18547) introduced **SGMV**, a kernel that batches requests for different adapters in one launch, and [S-LoRA](https://arxiv.org/abs/2311.03285) pages adapters between host and GPU in a unified pool with the KV cache. The cost is extra FLOPs proportional to $r$ and a throughput hit that grows with the number of distinct adapters per batch (`--max-loras` or `--max-loras-per-batch`).

**Serverless** deployments with shared, per-token pricing can multiplex customers on a device. Available model choices, batch controls, cold-start behavior, and isolation depend on the service contract. **Dedicated** capacity priced per GPU-hour can offer more control, while charging for idle time as well as useful work. Compare the complete service and quality contract before using a price-only break-even model.

For an illustrative dedicated cost of USD 2.50/hour and shared price of USD 0.20 per million output tokens, the price-only break-even is $2.50/0.20\times10^6=12.5\times10^6$ output tokens/hour, about 3,470 tokens/sec. This assumes matched output quality and excludes input-token charges and other service costs. Dedicated capacity must actually sustain that rate under the required SLO; below it, the shared service is cheaper under these assumptions.

**Cold start** is $t \approx \text{image pull} + \text{weights}/\text{storage\_BW} + \text{engine init}$, where init covers profiling, CUDA graphs, and `torch.compile`. 140 GB of 70B BF16 weights at 1 GB/s from object storage takes 140 s; at 7 GB/s from local NVMe, 20 s. The mitigations are weights cached on node NVMe, streaming loaders (vLLM `--load-format runai_streamer`), precompiled graph caches, warm pools, and LoRA instead of full fine-tunes so the base model stays resident. [Model Loading: Hub to GPU](../03-model-loading/README.md) works through these terms in detail.

**Autoscaling** should key on queue depth (`vllm:num_requests_waiting`, `sglang:num_queue_reqs`), KV utilization (`vllm:kv_cache_usage_perc`, `sglang:token_usage`), and TTFT p99 against the SLO.

> **Warning:** Do not autoscale on CPU or GPU utilization. GPU utilization reads near 100% at batch 1 and at batch 200 alike.

**Scale-to-zero** saves money only if cold start is inside the customer's tolerance. Pair it with a minimum of one replica during business hours, and scale on queue depth with a lead time larger than the cold start.

<a id="8-load-testing-methodology"></a>

## Summary

Account for cold starts and adapter working sets when planning replicas. State the assumptions before applying the result to a different workload.

## Review questions

1. Why does an idle replica sometimes remain useful?

   <details><summary>Answer</summary>

   It can provide warm capacity before a new replica loads weights and becomes ready; compare that benefit with idle cost.

   </details>

[Chapter contents](README.md) · [Next: Load tests and the deployment runbook](06-load-tests-and-deployment-runbook.md)
