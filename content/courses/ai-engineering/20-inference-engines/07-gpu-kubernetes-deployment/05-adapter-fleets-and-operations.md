# Adapter fleets and operations

This part of [Deploying LLMs on Kubernetes: GPUs, Replicas, and Rollouts](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Operate model and adapter fleets using latency, quality, and memory evidence.

## Many fine-tunes on one base

When many teams or customers each have a LoRA fine-tune of the same base model, the deployment is one base Deployment that serves all the adapters, not one Deployment per fine-tune. Start vLLM with `--enable-lora`, `--max-loras` (adapters resident on the GPU per batch), and `--max-lora-rank`, and either list adapters at startup with `--lora-modules` or, with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, load them at runtime through `/v1/load_lora_adapter`. Each adapter appears as its own model name on the OpenAI-compatible API. The economics are in [Serving, Capacity & Load Testing](../04-serving-and-load/05-adapters-and-elastic-capacity.md#multi-lora-serverless-versus-dedicated-cold-start-and-autoscaling): an 84 MB adapter loads in seconds where a 16 GB full fine-tune needs its own GPU. The routing consequence is adapter affinity: an endpoint picker that knows which pods already have an adapter loaded avoids swapping it in on every request.

## Building at scale: many models, many teams

Past one model the hard problem changes from writing a Deployment to sharing a finite, expensive pool of GPUs among models, teams, and kinds of work. Six practices carry most of the weight.

- **One pool per model, one gateway in front.** Each model gets its own Deployment or LeaderWorkerSet, its own `InferencePool`, and its own autoscaler, because its goodput, flags, and traffic differ. One Gateway routes on the request's `model` field to the right pool, and the AI gateway applies keys and quotas per tenant. Popular models keep warm minimums; long-tail models either scale to zero and accept the cold start, or become LoRA adapters on a shared base.
- **Priority, quotas, and gang admission.** Serving, batch evaluation, and fine-tuning compete for the same GPUs. Give serving a higher `PriorityClass` so it preempts batch work during a spike. Use Kueue for per-team GPU quotas that can borrow idle capacity. Kueue also admits a multi-pod workload's whole quota at once (gang admission), and with `waitForPodsReady` it evicts a group whose pods cannot all start in time. Without that, half-scheduled groups, such as a two-node LeaderWorkerSet replica or a training job, hold GPUs and serve nothing.
- **Heterogeneous hardware.** Keep a node pool per GPU type, labeled by GPU Feature Discovery, and put each model on the cheapest GPU that meets its SLO, which the memory budget in [Serving, Capacity & Load Testing](../04-serving-and-load/README.md) decides. Give each hardware type its own Deployment, since goodput, engine flags, and replica counts all change with the GPU.
- **Capacity is not elastic.** Cloud GPU stock runs out, and the cluster autoscaler cannot create an H100 the region does not have. Reserve capacity for the serving baseline and the failure headroom above; burst onto on-demand or spot capacity for batch and evaluation work, whose deadlines tolerate a reclaim.
- **Startup at scale.** A vLLM image is several gigabytes and the weights tens to hundreds more, and a 50-node scale-out multiplies both. Pre-pull images onto GPU nodes with a DaemonSet or use a lazy-pulling snapshotter, keep weights in node-local caches, and stagger restarts so a fleet-wide rollout does not become a storm.
- **The fleet as code, across clusters.** Keep one Helm chart parameterized per model, applied from Git, as in [Helm charts and the observability stack](../../../systems/10-containers-kubernetes/04-helm-charts-and-observability-stack.md) and [Cloud Native](../../../systems/09-cloud-native/README.md) in *Systems Engineering*, and release model revisions through [ModelRelease](../../../systems/07-durable-orchestration/12-model-release.md). One cluster is both a failure domain and a scaling ceiling, so large fleets run several, in more than one region, behind a global gateway that routes by health and latency and fails over between them.

## What to watch

Two metric sources together tell you what a model fleet is doing. The engine's `/metrics` endpoint reports the serving view: `vllm:num_requests_waiting` and `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, preemptions, and histograms of TTFT and inter-token latency. The DCGM exporter reports the hardware view: `DCGM_FI_DEV_FB_USED` (GPU memory), `DCGM_FI_DEV_GPU_UTIL`, temperatures, and XID errors, which are the driver's code for a failing GPU. Alert on the SLO (TTFT and TPOT p99) and on XID errors; dashboard the rest. Tracing a single request through gateway, router, and engine is built in [Tracing for the serving path](../../../systems/11-observability/02-tracing-for-the-serving-path.md).

## Pitfalls

- **No startup probe.** The liveness probe kills the pod during weight loading, in a loop that looks like a crash.
- **Default `/dev/shm`.** TP > 1 hangs or crashes in NCCL initialization.
- **Surge rollout on a full cluster.** The new pod stays `Pending` and the rollout never completes.
- **Downloading from the Hub at scale-out.** Rate limits, outages, and a moving `main` become your incidents.
- **Autoscaling on GPU utilization, or without lead time.** It scales too late or not at all.
- **TP that does not divide the node.** A TP=3 or TP=6 replica strands GPUs on 8-GPU nodes.
- **Time-sliced engines at default memory utilization.** The second engine fails to start, or one OOMs the others.
- **Proxy timeouts or a grace period shorter than the longest stream.** Long answers are cut off by a 15-second route timeout, or mid-stream on every rollout.
- **The default five-minute toleration for an unreachable node.** A dead node's replicas are not even replaced for five minutes.
- **Packing replicas to fill nodes.** One node becomes most of the fleet, and losing it breaches the SLO.
- **A disaggregated pool of one.** Every request now depends on a single pod.
- **Multi-pod groups without gang admission.** Half-scheduled groups hold GPUs and serve nothing.

## Where this leads

The capacity plan this section deploys comes from [Serving, Capacity & Load Testing](../04-serving-and-load/README.md), and the next section, [From Training Run to Shared Model](../08-train-publish-share.md), covers the other side of the handoff: producing a checkpoint that loads cleanly here, publishing it, and the practical deployments that use it. The Kubernetes mechanics are built from scratch in [Containers and Kubernetes](../../../systems/10-containers-kubernetes/README.md), including Helm charts and KEDA. Releasing a new model revision safely is [ModelRelease](../../../systems/07-durable-orchestration/12-model-release.md), and putting keys, rate limits, and usage accounting in front of the fleet is [Gateway](../../25-ai-gateway/README.md) in *AI Engineering*.

### Further reading

- [vLLM: deploying on Kubernetes](https://docs.vllm.ai/en/latest/deployment/k8s.html) and the vLLM production stack
- [NVIDIA GPU Operator documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/), including MIG and time-slicing
- [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws) and [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
- [KEDA Prometheus scaler](https://keda.sh/docs/latest/scalers/prometheus/)
- [llm-d](https://github.com/llm-d/llm-d) and [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo), two open reference stacks for disaggregated, cache-aware serving on Kubernetes

## Key terms

- **device plugin**: the node agent that advertises GPUs to Kubernetes as the integer extended resource `nvidia.com/gpu`.
- **MIG**: hardware partitioning of one GPU into isolated instances, each requested as its own resource.
- **time-slicing**: advertising one GPU as several, with no memory isolation between the pods that share it.
- **startup probe**: a probe that holds off liveness checks until a slow-starting container is up; for model servers it covers the cold start.
- **lead time**: the delay from an autoscaling decision to a replica serving traffic, including node provisioning and cold start.
- **LeaderWorkerSet**: a Kubernetes workload that creates, restarts, and scales a multi-pod replica as one unit.
- **endpoint picker**: a router that chooses a model-server pod per request from engine metrics, prefix affinity, and loaded adapters.
- **failure domain**: the set of replicas lost together in one failure (a node, a zone, a cluster); the fleet is sized so the replicas outside the largest one still meet peak load.
- **commit rule**: a request that fails before any byte reaches the client is retried elsewhere, and one that fails after is ended with an explicit error event.
- **gang admission**: admitting all the pods of a multi-pod workload together or none of them, so partial groups never hold GPUs.

## Summary

- GPUs per replica come from the memory budget (TP × PP); replicas come from measured goodput with headroom; their product is the GPU bill.
- GPUs are integer, non-overcommitted resources that the scheduler counts but does not size, so model pods select their GPU model and GPU nodes are tainted.
- Weights should reach nodes once, from a pinned revision, through a cache close to the GPU; serving pods never contact the Hub.
- A model server needs a startup probe sized to its cold start, memory-backed shared memory, a drain period for streams, and a rollout strategy chosen against spare GPUs.
- Autoscale on queue depth with lead time in mind; use LeaderWorkerSet for multi-node replicas, a metrics-aware router for prefixes and adapters, and separately sized and separately scaled pools for disaggregated prefill and decode.
- Every hop of a streaming request needs timeouts and a drain period longer than the longest generation, and overload is shed at the gateway rather than through readiness.
- Size the fleet so the replicas outside the largest failure domain still meet peak load, shorten the unreachable-node toleration, spread replicas, and let the commit rule protect requests in flight.
- At scale, give each model its own pool behind one gateway, use priority, quotas, and gang admission to share GPUs, reserve capacity for serving, and run more than one cluster.

## Review questions

1. A 70B FP8 model needs TP=2 and a load test finds 6 req/s of goodput per replica. Peak traffic is 20 req/s. How many replicas and GPUs, and how many 8-GPU nodes?
   <details><summary>Answer</summary>$\lceil 20 / (6 \times 0.7) \rceil = 5$ replicas of 2 GPUs each, 10 GPUs. A node fits four TP=2 pods, so the fleet needs two nodes, with six GPUs spare, which can serve as surge capacity for rollouts or host another model.</details>
2. A new vLLM pod with TP=2 restarts every 30 seconds and never becomes ready, while the same image with TP=1 works. Name two likely causes in the manifest.
   <details><summary>Answer</summary>First, no startup probe: the liveness probe fails while the larger model loads and kills the pod, so it restarts forever. Second, the default 64 MiB `/dev/shm`: TP=2 needs shared memory for NCCL and the worker processes, so initialization crashes or hangs. Add a startup probe sized to the cold start and mount a memory-backed `emptyDir` at `/dev/shm`.</details>
3. Five replicas load a 70 GB checkpoint at once from an NFS volume that sustains 1.5 GB/s in aggregate. How long until the last is ready, and what would you change?
   <details><summary>Answer</summary>350 GB at 1.5 GB/s is about 233 s, plus engine initialization. Move the weights closer: a node-local NVMe cache (about 10 s at 7 GB/s) or streaming from a regional object store, and keep warm capacity so that scale-out does not wait on the storm.</details>
4. Why does `maxSurge: 1, maxUnavailable: 0` sometimes stall a rollout forever, and what does the alternative cost?
   <details><summary>Answer</summary>The surge pod needs a free set of matching GPUs; if every GPU is in use, it stays `Pending` and the rollout never progresses. `maxSurge: 0, maxUnavailable: 1` needs no spare GPUs but removes one replica's capacity for each pod's whole cold start, so a 5-replica fleet runs at 80% for about $5 \times t_\text{cold}$.</details>
5. Why is GPU utilization a bad autoscaling signal, and what lead time must the autoscaler plan for when no GPU node is free?
   <details><summary>Answer</summary>GPU utilization measures whether a kernel is running, not how much work is queued; it reads near 100% at batch 1 and at full batch alike. Scale on queue depth and KV utilization. The lead time is node provisioning (3 to 6 minutes), GPU Operator readiness (1 to 2), and the pod's cold start: often 6 to 12 minutes in total.</details>
6. In the disaggregated sizing, the system prompt stops being cacheable. Which pool changes and by how much?
   <details><summary>Answer</summary>Uncached input rises from 500 to 2,000 tokens per request, so prefill demand goes from 10,000 to 40,000 tokens per second, about 5.7 GPUs instead of 1.4: the prefill pool grows about fourfold. Output is unchanged, so the decode pool stays the same size. Independent scaling of the pools is the reason to disaggregate.</details>
7. The worked fleet runs 5 replicas of 6 req/s goodput against a 20 req/s peak, packed four on one node and one on another. Does it survive losing a node, and what layout on three dedicated 8-GPU nodes does?
   <details><summary>Answer</summary>No. Losing the four-replica node leaves one replica, 6 req/s against 20, and even a three-and-two spread leaves 12 req/s. To survive any one node, $\lceil 20 / 6 \rceil = 4$ replicas must remain after the largest node goes. On three nodes that means two TP=2 replicas each: 6 replicas on 12 GPUs, and losing a node leaves $4 \times 6 = 24$ req/s. That is two more GPUs than the packed fleet, and the other 12 GPUs on those nodes can carry work that yields to serving. In general, replicas minus the most on any one node must be at least four.</details>
8. A product allows answers of up to 6,000 tokens. TPOT p99 is 50 ms and TTFT p99 is 800 ms. What should the route timeout and the termination grace period be, given a 15 s preStop sleep?
   <details><summary>Answer</summary>The longest stream lasts about $6000 \times 0.05 + 0.8 \approx 301$ s. Every total timeout (Envoy's route timeout, HTTPRoute `timeouts.request`, the gateway's upstream timeout) goes above that, say 330 s. An idle timeout resets on each streamed token, so it only needs to cover the longest silence, such as the queue wait before the first token. The grace period counts the preStop sleep, so it needs at least $15 + 301 = 316$ s, say 330 s. The Deployment's 120 s would cut every answer longer than about 2,100 tokens on each rollout.</details>
9. A GPU node loses power. Replicas take four minutes to cold-start and there are free GPUs elsewhere. Roughly when does a replacement serve traffic with the default tolerations, and with `tolerationSeconds: 30`?
   <details><summary>Answer</summary>The node controller marks the node unreachable after about 50 s without a heartbeat. With the default toleration, eviction follows 300 s later and then the 4-minute cold start runs: about $50 + 300 + 240 = 590$ s, nearly ten minutes. With 30 s it is about $50 + 30 + 240 = 320$ s, a little over five minutes. Either way, the spare capacity sized for a node loss carries the traffic in the meantime; the shorter toleration only shortens how long the fleet runs without its headroom.</details>

[Contents](README.md)
