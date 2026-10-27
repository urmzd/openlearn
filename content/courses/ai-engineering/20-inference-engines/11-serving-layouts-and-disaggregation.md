# Serving layouts: one process, replicas, and distributed pools

Serving architecture decides where weights and request state live and which components communicate on every token. Start with a single process and add distribution when memory, load, latency, or recovery requirements justify it. The right layout follows the workload and model, rather than the provider's product names.

> **Learning objectives**
> - Compare in-process inference, a local server, replicas, and model-parallel serving.
> - Account for state transfer in prefill/decode disaggregation.
> - Include admission, routing, and failure handling in capacity design.

## Layouts and their state boundaries

| Layout | Where state lives | When it helps | Cost or failure boundary |
|---|---|---|---|
| In-process model | Application process owns weights and caches | Debugging logits, custom training/evaluation, low-concurrency use | App and model share memory and lifecycle |
| Local HTTP engine | One service process or engine group owns weights and active KV | Shared API, streaming, concurrent callers | Serialization, bounded admission, process restart loses active state |
| Independent replicas | Every replica has a complete model or model-parallel group | More aggregate capacity and failover | Duplicated weights; routing affects prefix-cache locality |
| Tensor parallel | Weight matrices and partial activations span devices | Model/layers do not fit or large-batch work benefits | Frequent collective latency and interconnect pressure |
| Pipeline parallel | Layer ranges on different devices | Distribute large weights with fewer cross-stage tensor exchanges | Pipeline scheduling, bubbles, and request concurrency |
| Expert parallel | Experts across devices | Fit and execute MoE weights | Token dispatch, expert imbalance, and all-to-all traffic |
| Prefill/decode pools | Prompt processing and token generation on different workers | Independently size phases with different bottlenecks | Transfer KV, manage ownership, and preserve request/sampling state |
| Offline batch queue | Jobs and results are durable outside the engine | Throughput-first workloads with relaxed latency | Queue delay and retry/idempotency semantics |

A “replica” may use eight GPUs internally. Eight replicas and TP=8 describe very different memory, request routing, and failure behavior. Quantify both independent replica count and devices per replica.

## Disaggregation must move real state

The prefill worker reads the prompt and creates attention state. The decode worker needs the compatible KV blocks, positions, model/adapter revision, request history, and sampler state. If the first output token was already sampled, specify whether it is transferred or sampled by the receiver. Duplicate sampling changes the RNG stream and can duplicate output.

For an illustrative 2 GiB transfer over an effective 50 GB/s path, the byte-only lower bound is about 43 ms. Allocation, packing, queueing, network contention, synchronization, and any retransmission add to it. If the saved prefill/decode interference is smaller, separating the phases can make latency worse. Reusing compatible cached blocks can reduce transfer, but adds cache identity and ownership logic.

This is the tradeoff studied by [DistServe](https://arxiv.org/abs/2401.09670). It is a workload-dependent design, not an automatic improvement over colocated execution. The [worked disaggregated engine](../23-model-serving/07-disaggregated-prefill-decode.md) shows block transfer and cleanup; [the capacity parts](04-serving-and-load/03-parallelism-and-disaggregation.md) compare bandwidth and synchronization costs.

## Routing and admission are part of performance

Route by enough state to avoid hot spots: queued/running work, prompt/output lengths, available cache blocks, compatible prefix residency, and model/adapter identity where the engine exposes them. Pure request-count balancing treats a 100-token request like a 100,000-token one. Prefix-aware routing can reduce prefill but overload a popular replica. Compare load and locality together.

Bound admission and queue depth. Overload should produce a defined response or deadline outcome rather than unbounded waiting. Propagate cancellations so abandoned requests release KV. Define whether retries restart generation; a process failure normally loses ephemeral KV unless the system explicitly replicates or exports it. Retrying a partially streamed request cannot silently promise the original byte stream.

## Recovery and scale-out

Warm replicas avoid model download, device allocation, compilation, and graph capture delay. Scale on queueing and sustainable goodput with measured startup lead time. GPU utilization alone cannot distinguish a healthy saturated service from one violating every latency target. Include spare capacity for the failure domain the service must survive, such as one worker group or one node.

The [Kubernetes deployment reading](07-gpu-kubernetes-deployment/README.md) connects replicas to pods and failure domains. The [model release chapter](08-train-publish-share.md); record artifact compatibility before routing traffic between revisions.

## Key terms

- **Replica**: an independently schedulable copy of the serving model, possibly spread across devices.
- **Disaggregation**: placing distinct phases or responsibilities in separate worker pools.
- **State ownership**: the rule identifying which worker may mutate or release a request's cache and sampling state.

## Summary

Count device groups, state transfers, and restart boundaries. Distribution improves a service only when its added communication and coordination fit the workload's latency and cost budget.

## Review questions

1. Why can prefix-aware routing improve TTFT and worsen tail latency at the same time?

   <details><summary>Answer</summary>It can avoid repeated prefill while concentrating popular prefixes on a busy replica, increasing queueing.</details>
2. What must a KV handoff preserve beyond tensor bytes?

   <details><summary>Answer</summary>Model and adapter identity, positions, block ownership, committed token history, sampler/RNG state, and the first-token convention.</details>
