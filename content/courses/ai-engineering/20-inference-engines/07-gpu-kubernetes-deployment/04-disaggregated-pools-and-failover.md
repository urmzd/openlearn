# Disaggregated pools and failover

This part of [Deploying LLMs on Kubernetes: GPUs, Replicas, and Rollouts](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Account for KV handoff, ownership, and failure domains across worker pools.

## Disaggregated prefill and decode as two pools

[Disaggregation](../04-serving-and-load/03-parallelism-and-disaggregation.md#disaggregated-prefill-and-decode) becomes, on Kubernetes, two Deployments (or two LeaderWorkerSets) of the same model: a prefill pool started with the engine's prefill role and a decode pool with the decode role, a KV connector between them (NIXL over RDMA), and a router that sends each request to a prefill worker, then a decode worker. The pools scale independently, which is the point: you size each for its own bottleneck.

Size them from first principles, using the worked chatbot (20 req/s, 2,000-token mean input of which 1,500 tokens are a cached system prompt, 300-token output, 70B FP8 on H100s):

- **Prefill demand.** Uncached input is $20 \times 500 = 10{,}000$ tokens per second. Prefill costs about $2P = 140$ GFLOP per token. An H100 does about 990 TFLOPS of dense FP8 at 50% utilization, so one GPU prefills about $990 \times 10^{12} / 140 \times 10^9 \approx 7{,}000$ tokens per second. That needs 1.4 GPUs: one TP=2 prefill instance.
- **Decode demand.** Output is $20 \times 300 = 6{,}000$ tokens per second. On a TP=2 decode instance at batch 160, each step reads 35 GB of weights per GPU plus the batch's KV ($160 \times 2{,}300 \times 80$ KiB $\approx 30$ GB per GPU), about 65 GB at an effective 2.35 TB/s (70% of 3.35 TB/s), so a step takes about 28 ms and the instance produces $160 / 0.028 \approx 5{,}700$ tokens per second. That is just under the demand, so two decode instances, with headroom.
- **The fleet.** By throughput alone, one prefill instance and two decode instances: 6 GPUs, against 10 for the aggregated design. The 28 ms step also stays under the 50 ms TPOT SLO because no prefill ever enters a decode batch. Add the rules from later in this section, though (at least two instances per pool, and survive losing one) and the fleet becomes two prefill and three decode instances: losing a decode instance leaves $2 \times 5{,}700 = 11{,}400$ tokens per second against 6,000. That is 10 GPUs, the same as aggregated. At this size disaggregation buys a steadier TPOT, not fewer GPUs; the GPU saving appears once the pools are large enough that the minimums stop dominating.

These are first-order numbers: they ignore communication time, attention compute, and the KV transfer itself, and they must be confirmed by load-testing each pool. But they show how to set the ratio, and why it moves with the traffic. If the shared prefix stops being cacheable, prefill demand quadruples and the prefill pool must grow while the decode pool stays the same, which an aggregated fleet cannot do without overbuying decode.

### Running the two pools

On Kubernetes the two pools are two workloads (Deployments, or LeaderWorkerSets for multi-node instances) built from the same image and the same weights, started with different engine roles. In vLLM, the `--kv-transfer-config` JSON names a connector such as NIXL and sets `kv_role` to `kv_producer` for prefill or `kv_consumer` for decode; SGLang uses `--disaggregation-mode prefill|decode`. A disaggregation-aware router sits in front and makes the two-hop call, picking a prefill worker and a decode worker per request. llm-d and NVIDIA Dynamo package that router, and [Routing, cascades, and failover](../../25-ai-gateway/04-routing.md#disaggregated-prefill-and-decode) builds one by hand; [Disaggregated prefill and decode](../../23-model-serving/07-disaggregated-prefill-decode.md) builds the engine side, including the KV hand-off.

Three Kubernetes concerns follow from the KV transfer:

- **Placement.** The KV cache crosses RDMA, so the prefill and decode pods must sit in the same network block, with the RDMA devices exposed to their containers, exactly as for a multi-node LeaderWorkerSet. Over TCP the transfer cost in [Serving, Capacity & Load Testing](../04-serving-and-load/03-parallelism-and-disaggregation.md#disaggregated-prefill-and-decode) erases the gain.
- **Separate autoscaling signals.** Scale the prefill pool on its waiting queue or on TTFT, and the decode pool on KV-cache usage and running sequences, which track TPOT. One ScaledObject per pool, each with its own threshold. The ratio between them moves with prompt length and the cache hit rate, as the review question at the end of this section shows; Dynamo's planner rebalances it automatically.
- **Minimum size.** Each pool needs at least two instances on different nodes. A pool of one is a single point of failure for every request.

### When a pool fails

Disaggregation turns one failure domain per request into two, and each failure lands at a different point in the request:

| Failure | Bytes sent to the client? | What the router does |
|---------|---------------------------|----------------------|
| Prefill pod dies before the hand-off | No | Retries on another prefill worker |
| KV transfer fails or times out | No | Releases the decode worker's partial blocks and retries the whole request |
| Decode pod dies mid-stream | Yes | Ends the stream with an SSE `error` event (the commit rule, below) |
| The whole prefill pool is down | No | Every request fails; route to an aggregated fallback Deployment if you keep one |

Disaggregation pays at scale, with long prompts and strict TPOT, on an RDMA fabric. With a handful of replicas, short prompts, or no RDMA, one aggregated Deployment is simpler and has half the ways to fail.

## Failure domains and failover

Every layer of the fleet fails: an engine process, a GPU, a node, a zone, a whole cluster. Kubernetes replaces a lost replica, but only after a full cold start, so the fleet must carry enough spare capacity to ride out the failures you choose to survive, and the router must keep in-flight requests from turning into wrong answers.

### What fails and what Kubernetes does

| Failure | How it shows | What Kubernetes does on its own | What you add |
|---------|--------------|---------------------------------|--------------|
| Engine crash or OOM | Container exits | Restarts the container; the replica is out for one cold start | Headroom; a startup probe sized to the cold start |
| GPU fault (XID error, ECC error, GPU off the bus) | CUDA errors or a hung engine; the process may stay alive | The device plugin marks the GPU unhealthy so nothing new lands on it, but leaves the running pod alone | A liveness probe that exercises the engine; DCGM health checks; automation that cordons and drains a node with a failing GPU |
| Node lost (kernel panic, power, network partition) | Node goes `NotReady` after about 50 s without a heartbeat | Evicts its pods only after their unreachable toleration expires: 300 s by default | Shorter `tolerationSeconds` on serving pods; replicas spread across nodes |
| One pod of a LeaderWorkerSet group | The group can no longer serve | With the default `RecreateGroupOnPodRestart` policy, recreates the whole group | Count the group as one replica; its cold start is multi-node |
| Zone outage | Many nodes `NotReady` at once | The same, at scale; replacements need free GPUs in other zones | Spread across zones; reserved capacity in each |
| Cluster or region outage | Everything | Nothing | A second cluster behind a global gateway, or a fallback route to another provider |

The 300 s default matters most. Kubernetes adds a five-minute toleration for `node.kubernetes.io/unreachable` and `node.kubernetes.io/not-ready` to every pod, The node controller first waits about 50 s without a heartbeat before it marks the node unreachable. So a replica on a dead node is not replaced until nearly six minutes after the node stops, and then it pays its cold start. Shorten the toleration for serving pods, and spread replicas so one node never holds most of them. These fields go in the pod template, `spec.template.spec` of the Deployment:

```yaml
spec:  # the pod template: spec.template.spec in the Deployment
  tolerations:
    - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
    - { key: node.kubernetes.io/unreachable, operator: Exists, effect: NoExecute, tolerationSeconds: 30 }
    - { key: node.kubernetes.io/not-ready, operator: Exists, effect: NoExecute, tolerationSeconds: 30 }
  topologySpreadConstraints:
    - maxSkew: 1
      topologyKey: kubernetes.io/hostname
      whenUnsatisfiable: DoNotSchedule
      labelSelector: { matchLabels: { app: llama-70b } }
    - maxSkew: 1
      topologyKey: topology.kubernetes.io/zone
      whenUnsatisfiable: ScheduleAnyway
      labelSelector: { matchLabels: { app: llama-70b } }
```

A short toleration only starts the replacement sooner; the cold start still follows. It is the spare capacity that carries the traffic in between. The toleration has a cost too: 30 s on `not-ready` also evicts a healthy replica after a brief kubelet or network blip, and that replica then pays a multi-minute cold start. Pick the value against how often your nodes blip.

Pods with fixed names, which includes the StatefulSets inside a LeaderWorkerSet, can stay `Terminating` on a node that never comes back. Until the Node object is deleted, or tainted `node.kubernetes.io/out-of-service`, the group may not be recreated. Automate that step for nodes your cloud reports as gone.

### Sizing for the failure you must survive

The rule is one inequality over the largest failure domain you have decided to survive:

$$(\text{replicas} - \text{replicas in the largest domain}) \times \text{goodput per replica} \;\ge\; \text{peak req/s}$$

Apply it to the worked fleet, 5 replicas at 6 req/s of goodput against a 20 req/s peak:

- **Losing one replica:** $4 \times 6 = 24 \ge 20$. The fleet survives, running hotter until the replacement is ready.
- **Losing a node:** with bin-packing, node 1 holds four of the five replicas, and losing it leaves $1 \times 6 = 6$ req/s. Packing to fill nodes made one node 80% of the fleet. Even the default three-and-two spread leaves $2 \times 6 = 12$ req/s after losing the larger node.
- **Surviving any one node:** four replicas must remain after the largest node goes, since $\lceil 20 / 6 \rceil = 4$. On three dedicated nodes, two replicas each does it: 6 replicas and 12 GPUs, and losing a node leaves $4 \times 6 = 24$ req/s. That is 20% more GPUs than the 10-GPU fleet. Spreading one replica per node across five nodes also works on 10 GPUs, if those nodes are shared with other workloads. Either way, the GPUs left free on the serving nodes can host other models or batch work that yields to serving.

Surviving a zone is the same calculation with zones as the domain. Decide the domain explicitly: one replica is nearly free, one node usually costs a node, and one zone costs a zone's worth of reserved GPUs.

### The request in flight

Spare capacity protects the requests that arrive after a failure. The ones already running on a dead pod follow the **commit rule**, which the gateway chapter calls [the commitment boundary](../../25-ai-gateway/03-sse-streaming-proxy.md#the-commitment-boundary). If no byte has reached the client, the router retries on another replica, and the client sees only a slower first token. Once bytes have streamed, the stream ends with an SSE `error` event, because no other replica holds that sequence's KV cache. [A decode worker dies mid-stream](../../../systems/12-incident-response-and-chaos/02-kill-decode.md) in *Systems Engineering* explains why splicing or silently closing the stream is worse, and runs this failure as a drill.

## Review questions

1. Why does retrying a partially streamed generation require an explicit contract?

   <details><summary>Answer</summary>

   The original KV and RNG state may be gone, so a restart can produce different output and cannot silently promise continuation.

   </details>

[Contents](README.md) · [Next: Adapter fleets and operations](05-adapter-fleets-and-operations.md)
