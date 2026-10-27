# Load tests and the deployment runbook

This part of [Serving, Capacity & Load Testing: Math to Code](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Measure goodput with an open-loop load and explain a deployment decision.

## Load-testing methodology

Every number so far has been a bound or a model. The deliverable is a measurement, and measurements are easy to get wrong. A good benchmark is open loop, uses realistic lengths, includes a warmup, and avoids accidental prefix-cache hits.

The first choice is the arrival model. A **closed-loop** test (N concurrent users, each sending after the previous response) is self-throttling: a slow server receives fewer requests, so latency looks fine. It answers "what do 64 concurrent users feel like" (`--max-concurrency`). An **open-loop** test (arrivals at rate $\lambda$ regardless of responses) exposes queueing, which is what production traffic does. Default to open loop.

For the arrivals themselves, **Poisson** inter-arrival gaps are $\sim \text{Exp}(\lambda)$. Real traffic is burstier, which vLLM's `--burstiness` below 1 (gamma inter-arrivals) models; when the customer has a production trace, replay it (vLLM's `timed_trace` dataset, AIPerf's Mooncake trace format).

**Length distributions** drive everything, because input and output lengths set prefill and decode work. Use the customer's histogram; otherwise use lognormal, not fixed, lengths. Force output length with `ignore_eos` so runs are comparable, and say that you did.

Three pitfalls corrupt results. Without a **warmup**, the first requests pay for CUDA graphs, JIT compilation, and allocator pools, so send a few requests first and exclude them. **Prefix-cache contamination** happens when identical prompts are reused across runs or rates, so you measure cache hits rather than prefill; randomize every prompt with a unique nonce at the start, or disable prefix caching, unless caching is what you are testing, in which case test it deliberately with a fixed shared prefix. And **client bottlenecks** appear because a single-process Python client tops out at a few thousand streaming connections; watch client CPU. NVIDIA replaced GenAI-Perf with the multiprocess [AIPerf](https://docs.nvidia.com/aiperf/getting-started/migrating-from-gen-ai-perf) for this reason.

The standard tools are `vllm bench serve` (with `--request-rate`, `--burstiness`, `--max-concurrency`, `--goodput ttft:500 tpot:40`, `--percentile-metrics`, and `--metric-percentiles`), `python -m sglang.bench_serving` (same shape, plus `--random-range-ratio`), AIPerf or GenAI-Perf for engine-agnostic runs, and [`code/loadgen.py`](code/loadgen.py) for understanding what they compute. The heart of `loadgen.py` is the open-loop arrival process. Note that it never waits on completions:

```python
theta = 1.0 / (rate * args.burstiness)
next_t += rng.gammavariate(args.burstiness, theta)   # burstiness=1 is Exp(rate): Poisson
time.sleep(max(0.0, next_t - time.perf_counter()))
threading.Thread(target=one_request, args=(...)).start()  # never wait on completions
```

Run offline against the `--mock` fake engine, in which ITL grows with batch size and at most 64 requests run at once, a three-rate sweep produces this:

```text
$ python code/loadgen.py --mock --sweep 2,8,32 --num-requests 120 --output-len 48 --endpoint chat \
      --slo-ttft-ms 300 --slo-tpot-ms 25
  rate  out tok/s  goodput  ttft p50  ttft p99  tpot p50  tpot p99
     2         87     1.69      22.3     167.4      12.8      14.4
     8        229     4.41      28.3     178.2      14.0      16.4
    32        373     4.43     177.6     391.2      23.0      27.4
```

Throughput keeps rising from rate 8 to rate 32 while goodput stays flat and TTFT p99 crosses the 300 ms SLO: the knee is between them. That table, plotted, is the deliverable. Try it yourself with `--mock --sweep` and watch TTFT p99 explode once offered load passes capacity, then repeat against a real `vllm serve` on one GPU.

### Presenting results

The presentation is part of the job. Show one chart: achieved output tokens per second (or requests per second) on the x-axis, TTFT p99 and TPOT p99 on the y-axis, one point per offered rate, with the SLO lines drawn and the operating point marked. State the setup: model, precision, GPU type and count, engine and version, flags, input and output length distributions, arrival process, prefix-cache state, and warmup. Then give the **cost per 1M output tokens** at the operating point:

$$\$/\text{1M} = \frac{\text{GPU \$/hr} \times N_\text{gpu}}{\text{tok/s} \times 3600} \times 10^6$$

Assuming \$2.50 per H100-hour, three deployments price out as follows:

| Deployment | Throughput at SLO | Formula | $ per 1M output tokens |
|------------|-------------------|---------|------------------------|
| 8B FP8, 1x H100, roofline bound (`capacity.py`, batch 125) | 10,070 tok/s | `2.50 / (10070 × 3600) × 1e6` | $0.069 |
| 8B FP8, 1x H100, measured at TPOT p99 < 40 ms (illustrative) | 4,000 tok/s | `2.50 / (4000 × 3600) × 1e6` | $0.174 |
| 70B FP8, 2x H100, measured (illustrative) | 1,500 tok/s | `5.00 / (1500 × 3600) × 1e6` | $0.926 |

The gap between the first two rows is the honest conversation: the bound assumes 100% MBU, a server that is 100% busy, and no SLO. Quote the measured number and explain the bound as the ceiling. Price input tokens separately: they cost prefill FLOPs, about $2P$ per token, and are usually 3 to 10 times cheaper per token than output.

The same result must be explained to two audiences. This table shows the same four facts told to a researcher and to a customer:

| Question | To a researcher | To a customer |
|----------|-----------------|---------------|
| Why is TPOT 23 ms at peak? | Memory-bound at 70% MBU: 33 GiB of weights plus about 19 GiB of KV read per GPU per step at 3.35 TB/s | About 40 tokens per second per user, several times faster than anyone reads |
| Why not more throughput? | KV pool is full at 59 sequences of 8k; past that we preempt | Beyond ~60 simultaneous long chats per replica, latency breaks your SLO, so we add a replica |
| Why FP8? | Halves weight and KV bytes; accuracy delta measured on your eval set | Same answers on your test set, half the hardware, half the price |
| Why not speculative decoding? | At batch 60 we are near the ridge; verification costs `γ + 1` tokens of compute | It helps quiet hours, hurts peak hours; we enable it on the low-traffic tier |

## A runbook: from customer SLO to deployment config

Everything in this section composes into one procedure. Run [`code/capacity.py`](code/capacity.py) for the model your customer asked about and derive each printed line by hand before you start; then follow these steps.

1. **Collect** the model (and acceptable quantization), the traffic shape (peak and mean req/s, input and output length histograms, shared-prefix fraction), the SLOs (TTFT p99, TPOT p99 or E2E), region and data constraints, and the budget.
2. **Fit** by running `capacity.py` for candidate precisions and TP degrees. Pick the smallest TP whose KV pool holds $L_\text{Little} \times \text{mean context} \times 1.5$ tokens. Prefer FP8 weights on Hopper or Blackwell after an accuracy check on the customer's eval.
3. **Check the TPOT floor**: the single-stream decode bound at 0.7 MBU must beat the TPOT SLO with margin. If not, raise TP, quantize, add speculative decoding (if concurrency is low), or move to H200 or B200.
4. **Check the TTFT floor**: prefill time for the p99 input length, about $2Pn / (\text{peak} \times 0.5)$, must fit inside the TTFT SLO with room for queueing. If not, use TP or CP, prefix caching, or disaggregation.
5. **Configure**: set `--max-model-len` to the real p99.9 context, not the model maximum; `--max-num-seqs` from Little's law; `--max-num-batched-tokens` to 2k to 8k (smaller for strict ITL); prefix caching on; and FP8 KV if KV-bound.
6. **Measure one replica** with an open-loop sweep using the customer's length distribution, and find goodput at the SLO.
7. **Scale out** to $\lceil \text{peak req/s} \div \text{goodput per replica} \div 0.7 \rceil$ replicas for headroom. Add prefix-aware routing if shared prefixes matter. Disaggregate prefill and decode only when long prompts and strict TPOT coexist at enough scale to keep both pools busy.
8. **Autoscale and alert** on queue depth and KV utilization; alert on TTFT and TPOT p99 against the SLO; track preemptions.
9. **Report** the curve, the config, the cost per 1M tokens, and a one-line explanation of which term of the step-time equation binds.

### A worked deployment

Consider a support chatbot on Llama-3.1-70B with a peak of 20 req/s, lognormal inputs with a mean of 2k tokens (1.5k of them a shared system prompt), a mean output of 300 tokens, and SLOs of TTFT p99 under 800 ms and TPOT p99 under 50 ms.

Little's law gives the in-flight load: $\text{E2E} \approx 0.3 + 300 \times 0.04 \approx 12$ s, so $L \approx 240$ requests in flight at peak, and $240 \times 2.3\text{k} \approx 550\text{k}$ KV tokens. FP8 on TP=2 holds about 487k tokens at 80 KiB per token, so plan two replicas at minimum before measuring. The 94 tok/s single-stream bound gives about 15 ms TPOT at 70% MBU, well under 50 ms. The config is

```bash
vllm serve ... -tp 2 --quantization fp8 --kv-cache-dtype fp8 --max-model-len 8192 \
    --max-num-seqs 160 --max-num-batched-tokens 4096 --enable-prefix-caching
```

behind a prefix-aware router, so that the 1.5k-token system prompt is prefilled once per replica. Then measure, and let goodput set the replica count.

## Where this leads

The papers behind this section form a sequence worth reading in order: Orca, then PagedAttention, then Sarathi-Serve, then DistServe, since each fixes the bottleneck the previous one exposed. Within this chapter, [Quantization: Math to Code](../01-quantization/README.md) supplies $b$ and $b_\text{kv}$ in the memory budget, [Inference Frameworks](../02-frameworks/README.md) covers choosing an engine before tuning it, and [LLM Serving Platforms](../05-serving-platforms.md) places the engine within platform layers such as Ray and OpenRouter.

Beyond this book, *AI Engineering* takes up routing between models and cascades for fleet-level cost and latency in [Model Routing & Cascades](../../24-model-routing/README.md), and how TTFT and ITL reach the client in [Streaming and SSE](../../../systems/04-streaming/README.md). In *Systems Engineering*, [Observability](../../../systems/11-observability/README.md) covers the Prometheus metrics, SLOs, and alerting that drive autoscaling, and [Cloud Native](../../../systems/09-cloud-native/README.md) covers KEDA and Knative scaling on queue depth. The Poisson processes, percentiles, and queueing results used here come from [Probability and Statistics](../../../mathematics/13-probability-statistics/README.md).

This is the core skill set of solutions and field-engineering roles at Fireworks AI, Together AI, and Baseten, who size, configure, benchmark, and price dedicated deployments and explain the curves to customers. NVIDIA builds Dynamo, TensorRT-LLM, and NIXL for disaggregated serving at rack scale; Anyscale, Modal, and Replicate compete on serverless cold start, autoscaling, and scale-to-zero economics; the SGLang (LMSYS) and vLLM maintainers work on the scheduler, prefix caching, speculative decoding, and expert parallelism; Moonshot (Mooncake) and DeepSeek pioneered KV-centric disaggregation, MLA, MTP, and large-scale EP; and Anthropic, OpenAI, and Google run fleet capacity planning around goodput-driven SLOs and cost per token.

### Further reading

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) (2309.06180), [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) (OSDI 2022, iteration-level scheduling), [Sarathi-Serve](https://arxiv.org/abs/2403.02310) (2403.02310), [DistServe](https://arxiv.org/abs/2401.09670) (2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677) (2311.18677), [Mooncake](https://arxiv.org/abs/2407.00079) (2407.00079)
- [vLLM docs](https://docs.vllm.ai/en/latest/) and [SGLang docs](https://docs.sglang.io/)
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (rooflines, inference chapter) and [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- Databricks, [LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (MBU)
- [Speculative decoding](https://arxiv.org/abs/2211.17192) (2211.17192), [Medusa](https://arxiv.org/abs/2401.10774) (2401.10774), [EAGLE-3](https://arxiv.org/abs/2503.01840) (2503.01840)
- [Punica](https://arxiv.org/abs/2310.18547) (2310.18547), [S-LoRA](https://arxiv.org/abs/2311.03285) (2311.03285), [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) (2312.07104)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) and [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)

## Key terms

- **step time**: the duration of one engine iteration, the larger of its compute time and memory time plus communication and overhead.
- **arithmetic intensity**: FLOPs per byte moved; for a linear layer at BF16 it equals the number of tokens in the step.
- **ridge point**: the intensity $I^* = \text{peak}/\text{BW}$ that separates memory-bound from compute-bound work.
- **MBU / MFU**: the fraction of peak memory bandwidth or peak FLOPs an engine actually achieves.
- **KV pool**: the GPU memory left for KV cache after weights and activation headroom; it sets maximum concurrency.
- **TTFT, TPOT, ITL**: time to first token, mean time per output token, and each individual inter-token gap.
- **goodput**: the maximum request rate at which a target fraction of requests meet both the TTFT and TPOT SLOs.
- **Little's law**: $L = \lambda W$, mean in-flight requests equal arrival rate times mean time in system.
- **disaggregated serving**: running prefill and decode on separate GPU pools and transferring the KV cache between them.
- **open-loop load test**: a benchmark that sends requests at a fixed arrival rate regardless of responses, exposing queueing.

## Summary

- Decode is a bandwidth problem and prefill is a compute problem; batching amortizes the weight read and is the economic engine of inference.
- Memory decides concurrency: after weights, every byte is KV cache, and $\text{KV}_\text{tok} = 2 \times L \times n_\text{kv} \times d_\text{head} \times b$ turns the pool into a sequence count.
- SLOs are distributions; report goodput at p99 under a stated arrival process, not mean throughput at infinite load.
- Every knob moves a point along one latency-throughput curve, and every symptom maps to the term of the step-time equation that binds.
- Benchmark open loop with realistic lengths, a warmup, and no accidental prefix-cache hits, then present the curve and the cost per 1M tokens at the SLO point.

## Review questions

1. Why is the arithmetic intensity of a linear layer at BF16 equal to the number of tokens in the step, and what does that imply for prefill and decode on an H100?
   <details><summary>Answer</summary>FLOPs are $2Td_\text{in}d_\text{out}$ and weight bytes are $b\,d_\text{in}d_\text{out}$, so $I \approx 2T/b$, which is $T$ when $b = 2$. The H100's BF16 ridge point is about 295 FLOP/B. A 2048-token prefill has $I \approx 2048$, far above the ridge, so it is compute-bound; decode at batch 1 has $I \approx 1$, so it is memory-bound with compute about 300 times underused.</details>
2. Compute the single-stream decode bound for Llama-3.1-8B in BF16 on one H100 from weights alone.
   <details><summary>Answer</summary>$3.35 \times 10^{12}\ \text{B/s} / 16.06 \times 10^9\ \text{B} \approx 209$ tok/s. Adding a 4k-token KV history lowers it to about 202 tok/s, and real engines reach 60 to 85% of this bound.</details>
3. Why does 70B BF16 on TP=2 "fit but is useless", and why does FP8 on TP=2 match BF16 on TP=4?
   <details><summary>Answer</summary>At BF16, 65.75 GiB of weights per GPU leaves only 4.25 GiB of KV pool, about 27,844 tokens, or 3.4 sequences of 8k. FP8 halves the weights to 32.88 GiB per GPU and halves KV per token to 80 KiB, leaving 37.12 GiB of pool and 486,596 tokens, the same as BF16 at TP=4 but with half the GPUs.</details>
4. A deployment receives 10 req/s with a mean E2E latency of 6 s. What must `--max-num-seqs` and the KV pool accommodate?
   <details><summary>Answer</summary>By Little's law, $L = \lambda W = 10 \times 6 = 60$ requests are in flight on average, so `--max-num-seqs` must be at least 60 and the KV pool must hold about $60 \times$ the mean context in tokens. Otherwise the excess queues and TTFT grows.</details>
5. With acceptance rate 0.8 and 4 drafted tokens, what speedup does speculative decoding give at low and at high concurrency?
   <details><summary>Answer</summary>$\mathbb{E}[\text{tokens}] = (1 - 0.8^5)/(1 - 0.8) = 3.36$. At low concurrency, with a drafter cost of $c = 0.05$, $S = 3.36/1.2 = 2.8\times$. At high concurrency the step is compute-bound and verification costs about $\gamma + 1 = 5$ times as much, so $S \approx 3.36/5 = 0.67$, a 33% slowdown.</details>
6. Why should a load test be open loop, and what does the mock sweep show about goodput?
   <details><summary>Answer</summary>A closed-loop test self-throttles: a slow server receives fewer requests, so latency looks fine. An open-loop test sends at a fixed rate regardless of responses and exposes queueing, as production traffic does. In the mock sweep, output throughput rises from 229 to 373 tok/s between rates 8 and 32, but goodput stays flat at about 4.4 and TTFT p99 crosses the 300 ms SLO, so the knee lies between those rates.</details>
7. Why is TP across PCIe-only cards usually capped at 2?
   <details><summary>Answer</summary>For 70B decode at batch 64 with TP=4, each step moves about 252 MB of all-reduce traffic per GPU. At about 450 GB/s on NVLink that is 0.56 ms, about 3% of a 16 ms step; at about 50 GB/s on PCIe Gen5 it is about 5 ms, about 30%, before the latency of 160 small collectives. Data-parallel replicas avoid the communication entirely, so they beat TP on PCIe-only boxes.</details>

[Chapter contents](README.md)
