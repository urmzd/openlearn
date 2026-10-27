# Serving, Capacity & Load Testing: Math to Code

The chapter so far has explained how inference works: prefill and decode, the KV cache, batching, quantization, and parallelism. This section turns that understanding into a professional skill: sizing, configuring, benchmarking, and explaining a real LLM deployment. Picture the engineer who inherits a customer deployment at an inference cloud from an ML or performance engineer and must defend every number in it, both to a researcher who asks *why* and to a buyer who asks *how much*. Everything here is organized as math first, then the code and engine flags that implement it.

The thread that ties it together is a single equation for the time of one engine step. Once you can write it down, every metric a customer cares about, time to first token, time per output token, throughput, and cost, becomes a function of it, and every engine knob becomes a way of changing one of its terms. We derive the roofline that separates prefill from decode, build a memory budget that decides concurrency, define the metrics and SLOs that form the contract, and then work through parallelism, engine knobs, speculative decoding, multi-LoRA serving, and load-testing methodology. The section ends with a runbook that takes a customer's SLO to a deployment config.

This section builds on the chapter opening, [LLM Systems & Inference](../README.md) (prefill, decode, and the KV cache), and on [Quantization: Math to Code](../01-quantization/README.md) for what FP8 and INT4 buy. The queueing and percentile math uses Poisson processes from [Probability and Statistics](../../../mathematics/13-probability-statistics/README.md) in *Mathematics*, and the metrics side connects to [Observability](../../../systems/11-observability/README.md) in *Systems Engineering*. Runnable, stdlib-only tools live in [`code/`](code/README.md): a capacity calculator and an open-loop load generator with an offline mock engine. Engine flags were reviewed against the vLLM and SGLang documentation in October 2026.

> **Learning objectives**
> - Write the step-time equation for a serving engine and use the roofline to explain why prefill is compute-bound and decode is bandwidth-bound.
> - Build a GPU memory budget and derive maximum concurrency from the KV-cache size per token.
> - Define TTFT, TPOT, ITL, throughput, and goodput, and use Little's law and queueing intuition to size for a target load.
> - Choose a parallelism strategy and engine flags for a workload, and map a failing SLO to the knob that fixes it.
> - Design an open-loop load test, find the goodput knee, and present cost per million tokens to both a technical and a business audience.

## In this chapter

1. [Step time and the roofline](01-step-time-and-roofline.md): Identify whether compute, memory traffic, communication, or overhead limits a workload.
2. [Memory budget and serving SLOs](02-memory-budget-and-slos.md): Derive cache capacity and distinguish admitted concurrency from stable throughput.
3. [Parallelism and disaggregation](03-parallelism-and-disaggregation.md): Choose a serving layout using memory needs and interconnect costs.
4. [Engine controls and speculation](04-engine-controls-and-speculation.md): Map a performance symptom to a measurable scheduling or execution control.
5. [Adapters and elastic capacity](05-adapters-and-elastic-capacity.md): Account for cold starts and adapter working sets when planning replicas.
6. [Load tests and the deployment runbook](06-load-tests-and-deployment-runbook.md): Measure goodput with an open-loop load and explain a deployment decision.
