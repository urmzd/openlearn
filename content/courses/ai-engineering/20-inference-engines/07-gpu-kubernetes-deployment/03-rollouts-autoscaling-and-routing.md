# Rollouts, autoscaling, and routing

This part of [Deploying LLMs on Kubernetes: GPUs, Replicas, and Rollouts](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Budget rollout and scale-out lead time with the required spare resources.

## Rollouts: surge needs spare GPUs

A rolling update replaces pods one at a time, and the two knobs trade spare hardware against capacity. `maxSurge: 1, maxUnavailable: 0`, as above, starts a new pod before stopping an old one, so capacity never drops, but the new pod needs two free GPUs on a node that matches. If every GPU is in use, the surge pod stays `Pending` and the rollout stalls, silently, forever. `maxSurge: 0, maxUnavailable: 1` needs no spare GPUs but removes one replica, a fifth of capacity in our fleet, for the duration of each pod's replacement, which is the whole cold start.

The arithmetic decides. A rollout of $R$ replicas, one at a time, takes about $R \times t_\text{cold}$. With five replicas and a four-minute cold start, that is 20 minutes during which, with `maxUnavailable: 1`, the fleet runs at $4/5$ of capacity. If peak load needs more than 80% of the fleet's goodput, roll out off-peak or keep a spare node. The six leftover GPUs in the worked fleet are exactly that spare.

A model change is a release, not a configuration change: a new revision can be worse in ways that load tests do not catch. Roll it out as a canary, a second Deployment with one replica and a share of traffic, measured against the old one, before scaling it up. The gates, the bake period, and the rollback saga are built in [ModelRelease](../../../systems/07-durable-orchestration/12-model-release.md). Add a `PodDisruptionBudget` with `maxUnavailable: 1` so that node drains during cluster upgrades also take only one replica at a time, and spread replicas across zones with `topologySpreadConstraints` so that one zone outage does not take most of the fleet ([Failure domains and failover](04-disaggregated-pools-and-failover.md#failure-domains-and-failover) sizes this).

## Autoscaling with lead time

Scale on the signals that measure the engine's load: queue depth (`vllm:num_requests_waiting`) and KV-cache utilization (`vllm:kv_cache_usage_perc`). Not CPU, since the CPU is idle while the GPU works, and not GPU utilization, which reads near 100% at batch 1 and at batch 160 alike. The Horizontal Pod Autoscaler cannot read Prometheus by itself; KEDA can:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llama-70b
spec:
  scaleTargetRef: { name: llama-70b }
  minReplicaCount: 3
  maxReplicaCount: 10
  cooldownPeriod: 900
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: sum(vllm:num_requests_waiting{model_name="llama-70b"})
        threshold: "4"
```

KEDA computes replicas as the query's value divided by the threshold, so four or more requests waiting per replica adds pods. The `cooldownPeriod` of 15 minutes keeps it from scaling in on a brief lull and paying a cold start to scale out again.

The decisive number is **lead time**. When no GPU node is free, a new replica waits for the cluster autoscaler or Karpenter to provision a node (3 to 6 minutes on most clouds), for the GPU Operator to make the node schedulable (1 to 2 minutes), and then for the pod's own cold start. That is 6 to 12 minutes from the decision to the first token. Traffic that doubles in five minutes will breach the SLO before help arrives, whatever the autoscaler does. The defenses come from that number: keep a `minReplicaCount` that covers the expected peak during business hours, scale out on a threshold well before saturation, keep a warm spare node, and shrink the cold start with a node-local cache. Scale-to-zero is correct only when the users can tolerate the whole lead time on their first request.

## Multi-node replicas: LeaderWorkerSet

A replica that needs more GPUs than one node has, such as a 405B model in BF16 or a large MoE, is a group of pods that must start, stop, and fail together: if one pod of a pipeline dies, the others are useless. A Deployment cannot express that; **LeaderWorkerSet** (LWS) can. Each LWS replica is a leader pod plus `size - 1` worker pods, created and restarted as a unit, with the leader's address injected into every worker:

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: llama-405b
spec:
  replicas: 2                      # data parallelism: two independent model copies
  leaderWorkerTemplate:
    size: 2                        # pods per replica: here, two 8-GPU nodes
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      spec:
        containers:
          - name: vllm
            image: vllm/vllm-openai:v0.31.0
            command: ["sh", "-c"]
            args:
              - ray start --head --port=6379 &&
                vllm serve /models/llama-3.1-405b-instruct/<sha>
                  --tensor-parallel-size 8 --pipeline-parallel-size 2
                  --distributed-executor-backend ray
            resources:
              limits: { nvidia.com/gpu: 8 }
    workerTemplate:
      spec:
        containers:
          - name: vllm
            image: vllm/vllm-openai:v0.31.0
            command: ["sh", "-c"]
            args: ["ray start --address=$(LWS_LEADER_ADDRESS):6379 --block"]
            resources:
              limits: { nvidia.com/gpu: 8 }
```

The parallelism layout follows the bandwidth rule from the chapter opening: TP=8 inside each node, over NVLink; PP=2 across the two nodes, over InfiniBand or RoCE. The volumes, probes, and shared memory are as in the Deployment and omitted here. Two things matter that the manifest does not show. The nodes of one replica should sit in the same network block, under one top-of-rack switch, which topology-aware scheduling (Kueue's topology-aware scheduling, or the cloud's placement groups) enforces. And the pods need the RDMA network devices, typically through an RDMA device plugin or SR-IOV and NCCL environment variables, or the pipeline traffic silently falls back to TCP.

## Routing: when round-robin is wrong

A plain Service spreads requests evenly, which is wrong for an LLM in two ways. Requests differ in cost by orders of magnitude (a 100-token question and a 30,000-token document), so equal counts are not equal load. And prefix caching only helps if requests that share a prefix land on the replica that holds it. With five replicas and round-robin, a 1,500-token system prompt is prefilled on each replica and hits the cache only one time in five at first, rather than nearly always.

The **Gateway API Inference Extension** adds an `InferencePool` resource, a set of model-server pods behind a Gateway, and an **endpoint picker** that chooses a pod per request using the engines' own metrics: queue depth, KV utilization, which LoRA adapters each pod has loaded, and prefix-cache affinity. Projects such as llm-d and NVIDIA Dynamo build on the same idea. Use one once you have more than a couple of replicas and either a large shared prefix or many adapters; until then, a Service is fine.

## A request's path through the cluster

A chat completion crosses five hops before its first token comes back, and each one can cut a stream short. In order:

1. **The edge.** A cloud load balancer and a Gateway (or Ingress) terminate TLS and pick a route.
2. **The AI gateway.** It checks the API key, applies rate limits and quotas, records usage, and picks a model route, as built in [Gateway](../../25-ai-gateway/README.md).
3. **The endpoint picker or Service.** It chooses one pod, by engine metrics and prefix affinity or by round-robin.
4. **The engine's API server.** vLLM tokenizes the prompt and admits the request to its scheduler's waiting queue.
5. **The engine loop.** Continuous batching runs the request's prefill, then one decode step at a time, as described in [Inside an Inference Engine](../06-inside-an-inference-engine.md). Each new token goes back as a server-sent event through every hop above.

Three settings along that path are wrong by default for LLM streams.

**Timeouts must outlast the longest generation.** Proxy hops have two kinds of timeout, and they need different values. A *total* timeout caps the whole request, however busy the stream: Envoy's route timeout, which defaults to 15 s and sits under many Gateway implementations; the HTTPRoute `timeouts.request` and `timeouts.backendRequest` fields; and some cloud backend timeouts, such as Google Cloud's 30 s default. Each must exceed the longest stream. An output cap of 4,096 tokens (an API `max_tokens` limit) at the 50 ms TPOT SLO streams for up to $4096 \times 0.05 \approx 205$ s, plus the time to first token. An *idle* timeout, such as an AWS load balancer's 60 s default, resets on every byte, so a streaming response renews it with each token. It must exceed the longest silence instead: the queue wait before the first token, or a non-streaming request's whole generation.

**The drain period must outlast it too.** The Deployment above gives a deleted pod 120 s of grace, and the 15 s preStop sleep counts against it. A request admitted at the very end of that sleep has 105 s left, which at 50 ms per token covers about 2,100 tokens. Streams already running keep generating during the sleep, so this is the worst case. Set `terminationGracePeriodSeconds` to the preStop sleep plus the longest-stream number. That works only if the engine finishes in-flight requests on SIGTERM rather than exiting at once. Check your engine version's shutdown behavior, and if it exits immediately, extend the preStop hook to wait for the running-request count to reach zero.

**Shed load at the front, not in readiness.** When traffic outruns capacity, the engine's waiting queue grows and every queued request's TTFT grows with it. Reject excess early, with a 429 or 503 and a `Retry-After` from the AI gateway or the endpoint picker, both of which can read queue depth. Do not fail the readiness probe on load: a busy pod that drops out of the endpoints pushes its share onto the others, which then drop out too, and the fleet flaps.

## Review questions

1. Why can a healthy rollout remain pending indefinitely?

   <details><summary>Answer</summary>

   A surge replica needs allocatable GPUs with the right placement; a full cluster may have none.

   </details>

[Contents](README.md) · [Next: Disaggregated pools and failover](04-disaggregated-pools-and-failover.md)
