# Deploying LLMs on Kubernetes: GPUs, Replicas, and Rollouts

The capacity math in [Serving, Capacity & Load Testing](04-serving-and-load/README.md) ends with a number: so many replicas, each needing so many GPUs, with a given set of engine flags. This section turns that number into a running fleet on Kubernetes. Kubernetes was designed for small, cheap, stateless pods that start in seconds and can be packed anywhere. An LLM replica is the opposite: it asks for whole GPUs that are scarce and expensive, it needs tens or hundreds of gigabytes of weights before it can answer anything, it takes minutes to start, and the way you route requests to it changes its throughput. Almost every default (round-robin Services, CPU autoscaling, surge rollouts, quick liveness probes) is wrong for it, and each wrong default has a cost you can compute.

You build the deployment in the order a platform engineer would: the arithmetic that sets the pod's shape and the replica count, the GPU node pool, getting weights from Hugging Face onto the node, the Deployment itself, rollouts, autoscaling, multi-node replicas, routing and the path a request takes through the cluster, disaggregated prefill and decode as Kubernetes objects, failure and failover, and running many models on one shared GPU cluster. Each step comes with the manifest and the reason for each field.

You should have read the chapter opening, [LLM Systems & Inference](README.md), [Inside an Inference Engine](06-inside-an-inference-engine.md), which explains what each pod below is doing as it starts, and the memory budget and runbook in [Serving, Capacity & Load Testing](04-serving-and-load/README.md). The Kubernetes basics (Deployments, Services, Helm, KEDA) are built step by step in [Containers and Kubernetes](../../systems/10-containers-kubernetes/README.md) in *Systems Engineering*; this section assumes them and adds what changes when the workload is a GPU-bound model server. How weights move from the Hub to GPU memory is covered byte by byte in [Model Loading: Hub to GPU](03-model-loading/README.md).

> **Note:** This layer moves fast. The manifests were written in October 2026 against Kubernetes 1.34, the NVIDIA GPU Operator, KEDA 2.x, LeaderWorkerSet v1, and vLLM 0.31. Pin image tags and chart versions to the ones you tested.

> **Learning objectives**
> - Derive a pod's GPU request, the replica count, and the total GPU count from a model's memory budget and a measured per-replica goodput.
> - Explain how the NVIDIA device plugin exposes GPUs and how taints, labels, MIG, and time-slicing decide where a model pod lands.
> - Choose a strategy for getting Hugging Face weights onto nodes, and compute the cold-start and scale-out cost of each.
> - Write a vLLM Deployment whose probes, shared memory, rollout strategy, and termination grace match how a model server actually behaves.
> - Autoscale on queue depth with enough lead time, run a model across nodes with LeaderWorkerSet, and size separate prefill and decode pools.
> - Trace a request through the gateway, the endpoint picker, and the engine, and set proxy timeouts and drain periods from the longest stream.
> - Size a fleet to survive the loss of a replica, a node, or a zone, and run disaggregated pools and many models on a shared GPU cluster.

## From the capacity plan to a pod shape

Start with the arithmetic, because every manifest field below follows from it. A serving fleet has three numbers:

$$\text{GPUs per replica} = \text{TP} \times \text{PP}, \qquad \text{replicas} = \left\lceil \frac{\text{peak req/s}}{\text{goodput per replica} \times 0.7} \right\rceil, \qquad \text{total GPUs} = \text{replicas} \times \text{TP} \times \text{PP}$$

The first comes from the memory budget: the smallest TP (and PP, if one node is not enough) whose KV pool holds the in-flight tokens Little's law predicts. The second comes from a load test of one replica at the SLO; the 0.7 is headroom so that a traffic spike or the loss of one replica does not breach the SLO. Surviving the loss of a whole node is sized separately, in [Failure domains and failover](#failure-domains-and-failover). Data parallelism, the replica count, is the Kubernetes `replicas` field. Tensor and pipeline parallelism live *inside* a replica and become its GPU request.

Two constraints from the model limit the choice of TP. TP must divide the number of attention heads, since each rank takes whole heads; and once TP exceeds the number of KV heads, each rank holds a replicated copy of the KV cache, so the extra GPUs buy compute but no KV capacity. For Llama-3.1-70B, with 64 query heads and 8 KV heads, the useful TP values are 1, 2, 4, and 8.

### A worked fleet

Take the support chatbot from the runbook: Llama-3.1-70B, peak 20 req/s, TTFT p99 under 800 ms, TPOT p99 under 50 ms. The memory budget picked FP8 weights and FP8 KV on TP=2, which holds about 487,000 KV tokens per replica. Suppose an open-loop load test of one replica, run with the customer's length distribution, finds a goodput of 6 req/s at the SLO. Then

$$\text{replicas} = \lceil 20 / (6 \times 0.7) \rceil = \lceil 4.76 \rceil = 5, \qquad \text{total GPUs} = 5 \times 2 = 10$$

The pod requests `nvidia.com/gpu: 2`. Kubernetes places all of a pod's GPUs on one node, which is exactly what TP needs, since its all-reduces must stay on NVLink. An 8-GPU H100 node fits four such pods, so the fleet needs two nodes, with six GPUs left over. The default scheduler spreads the five pods about three and two; a bin-packing scoring policy, or consolidation by Karpenter, packs them four and one. That leftover is the next design question: run a second, smaller model there, keep them as warm spare capacity for scale-out, or accept the cost. Fragmentation like this is why replica shapes should divide the node: on 8-GPU nodes, prefer TP of 1, 2, 4, or 8.

The GPU is not the only resource the pod needs. Budget, per pod:

| Resource | Why the engine needs it | Rule of thumb |
|----------|-------------------------|---------------|
| `nvidia.com/gpu` | Weights, KV pool, activations | TP × PP (per node for multi-node) |
| CPU | Tokenization, detokenization, the scheduler loop, the HTTP server | 4 to 8 cores per GPU; a starved CPU shows up as high TTFT with idle GPUs |
| Memory (host RAM) | Reading safetensors through the page cache, pinned staging buffers, CPU-resident LoRA adapters | At least the pod's share of the weights plus 16 to 32 GiB |
| `/dev/shm` | NCCL and the engine's worker processes exchange tensors through shared memory | A memory-backed `emptyDir`; 8 to 16 GiB for TP > 1 |
| Ephemeral or persistent storage | Weights, compiled graph caches | Checkpoint size plus about 10% |

The `/dev/shm` line catches nearly everyone once. Container runtimes give a container 64 MiB of shared memory by default. With TP=1 the engine may never notice; with TP=2 it crashes or hangs at startup in NCCL initialization, with an error that does not mention shared memory.

## The GPU node pool

Kubernetes knows nothing about GPUs on its own. The **NVIDIA GPU Operator** installs, as DaemonSets on each GPU node, the driver, the container toolkit that lets containers see the GPUs, the **device plugin** that advertises `nvidia.com/gpu` as an extended resource, **GPU Feature Discovery**, which labels each node with its GPU model and memory, and the **DCGM exporter**, which publishes GPU metrics to Prometheus. Managed clouds offer node images with some of these preinstalled; the roles are the same.

Extended resources have three rules that shape everything else. A GPU request is an integer: there is no `nvidia.com/gpu: 0.5`. Requests and limits must be equal, so there is no overcommit. And the scheduler counts devices, not memory: it will happily place a pod that needs 140 GiB of weights on a node of 24 GiB GPUs.

That last rule is why model pods must name their hardware. GPU Feature Discovery labels nodes with keys such as `nvidia.com/gpu.product` (for example `NVIDIA-H100-80GB-HBM3`) and `nvidia.com/gpu.memory`; select on them. And taint GPU nodes, typically `nvidia.com/gpu=present:NoSchedule`, so that ordinary pods do not land on a node that costs tens of dollars an hour and take the CPU and memory a model pod needs. Model pods then carry the matching toleration.

Follow one model pod from request to running engine to see where each piece acts. The components named here are introduced in [The control plane: who does what](../../systems/10-containers-kubernetes/README.md#the-control-plane-who-does-what) in *Systems Engineering*.

1. **The device plugin advertises.** On each GPU node it reports a count, say `nvidia.com/gpu: 8`, to the kubelet, which publishes it in the node's status as allocatable capacity.
2. **The pod asks.** Its container spec sets `nvidia.com/gpu: 2` in `limits` (and therefore in `requests`).
3. **The scheduler filters and binds.** It drops every node with fewer than two unallocated GPUs, every node whose `nodeSelector` labels do not match, and every node whose taint the pod does not tolerate, scores what is left, and binds the pod to one node. Because a pod is never split across nodes, both GPUs come from the same machine.
4. **The kubelet allocates.** When it starts the pod, the kubelet's device manager picks two free devices and calls the plugin's `Allocate` with their IDs. The plugin answers with what the container runtime must expose (device nodes, mounts, and environment, or CDI device names), so the container sees exactly those two GPUs, numbered 0 and 1, and none of the other six.
5. **The engine runs.** vLLM starts with `--tensor-parallel-size=2`, finds two visible GPUs, and its all-reduces run over the NVLink between them.

When a pod is deleted, the kubelet returns its devices to the node's free pool, and the scheduler can bind the next pod to them.

Two mechanisms share one physical GPU between pods, and they are not interchangeable:

| | **MIG** (Multi-Instance GPU) | **Time-slicing** |
|---|---|---|
| How | The GPU is partitioned in hardware into isolated instances, each with its own memory and compute slices | The device plugin advertises one GPU as N; the pods share it in turn |
| Resource name | `nvidia.com/mig-1g.10gb`, `nvidia.com/mig-3g.40gb`, ... (H100 80 GB: up to seven 1g.10gb instances) | `nvidia.com/gpu`, with a sharing config of `replicas: N` |
| Memory isolation | Yes | None: one pod can OOM all the others |
| Good for | Embedding models, rerankers, small quantized LLMs with predictable footprints | Dev clusters and low-traffic demos |

With time-slicing, every engine on the GPU must be told its share: vLLM's `--gpu-memory-utilization` claims a fraction of the *whole* GPU's memory, so four engines at the default of 0.9 cannot coexist. Set the fractions to sum below 1. Newer clusters can also use **Dynamic Resource Allocation** (DRA, generally available since Kubernetes 1.34), in which a pod makes a `ResourceClaim` describing the device it needs (for example, a GPU with at least 80 GB) instead of counting an opaque integer; the NVIDIA DRA driver implements it. The concepts in this section carry over unchanged.

## Getting weights from Hugging Face onto the node

A replica cannot serve until its weights are in GPU memory, and on Kubernetes the dominant term is usually getting them onto the node. [Model Loading](03-model-loading/README.md) gives cold start as roughly the checkpoint size divided by the slowest link, plus engine initialization. On a cluster there is a second cost: a scale-out event loads the same bytes into several pods at once, through a link they share.

The strategies, from worst to best for production:

| Strategy | How | Cold start (70 GB, FP8 70B) | Fails how |
|----------|-----|-----------------------------|-----------|
| Download at pod start | `hf download` in an init container, into an `emptyDir` | Minutes, bounded by the Hub and egress | Hub rate limits and outages become your outages; every replica pays it |
| Bake into the image | `COPY` weights into the container image | Image pull: minutes for a 70 GB layer, and the registry must store it | Every weight change is a 70 GB image push |
| Shared volume | A Job downloads once into a `ReadWriteMany` PVC (NFS, Filestore, EFS, CephFS); pods mount it read-only | Volume bandwidth, shared by every pod loading at once | A scale-out storm divides the bandwidth |
| Object-store mirror plus streaming | Mirror the pinned revision to S3 or GCS; the engine streams tensors directly (`--load-format runai_streamer`) | 30 to 90 s at several GB/s | Needs a fast bucket in the same region |
| Node-local cache | A DaemonSet or an init container copies to local NVMe once per node; pods mount it with `hostPath` or a local PV | About 10 s at 7 GB/s | Node replacement re-pays the copy |

Work the scale-out storm for the shared volume. Five replicas start at once; the NFS server sustains 1.5 GB/s in aggregate; each must read 70 GB. The total is 350 GB, so the last replica is ready after $350 / 1.5 \approx 233$ s, plus a minute of engine initialization, compared with about 10 s from local NVMe. If the autoscaler fires because the queue is already growing, that difference is four minutes of SLO breach.

Whichever strategy you pick, four rules hold:

1. **Pin the revision.** Download a 40-character commit sha, not `main`. A replica that starts tomorrow must load the bytes you load-tested today.
2. **Never contact the Hub from a serving pod.** Set `HF_HUB_OFFLINE=1` so that a missing file fails loudly at startup instead of triggering a download during a scale-out.
3. **Keep the token in a Secret.** Gated models such as Llama need an `HF_TOKEN` with read access, mounted into the download Job only, not into the serving pods.
4. **Download only what you serve.** A repo may hold the same weights in several formats; `--include` patterns keep `*.safetensors`, the JSON files, and the tokenizer, and skip `original/` and `*.pth`.

Here is the download Job for the shared-volume strategy. It runs once per model revision:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: fetch-llama-3-1-70b-instruct
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: fetch
          image: python:3.12-slim
          command: ["bash", "-c"]
          args:
            - |
              pip install --no-cache-dir "huggingface_hub[hf_xet]" &&
              hf download meta-llama/Llama-3.1-70B-Instruct \
                --revision "$REVISION" \
                --include "*.safetensors" "*.json" "tokenizer*" \
                --local-dir "/models/llama-3.1-70b-instruct/$REVISION"
          env:
            - name: REVISION
              value: "<pinned 40-character commit sha>"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef: { name: hf-read-token, key: token }
          volumeMounts:
            - { name: models, mountPath: /models }
      volumes:
        - name: models
          persistentVolumeClaim: { claimName: model-store }
```

The path includes the revision, so two revisions can coexist on the volume during a rollout and a rollback is a path change, not a re-download.

## The Deployment

With weights on a volume, the Deployment is short, and almost every field is there for a reason specific to model servers:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-70b
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }
  selector: { matchLabels: { app: llama-70b } }
  template:
    metadata:
      labels: { app: llama-70b }
    spec:
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-80GB-HBM3
      tolerations:
        - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
      terminationGracePeriodSeconds: 120
      containers:
        - name: vllm
          image: vllm/vllm-openai:v0.31.0
          args:
            - --model=/models/llama-3.1-70b-instruct/<sha>
            - --served-model-name=llama-70b
            - --tensor-parallel-size=2
            - --quantization=fp8
            - --kv-cache-dtype=fp8
            - --max-model-len=8192
            - --max-num-seqs=160
            - --max-num-batched-tokens=4096
            - --enable-prefix-caching
            - --gpu-memory-utilization=0.90
            - --port=8000
          env:
            - { name: HF_HUB_OFFLINE, value: "1" }
          ports:
            - { name: http, containerPort: 8000 }
          resources:
            requests: { nvidia.com/gpu: 2, cpu: "12", memory: 128Gi }
            limits:   { nvidia.com/gpu: 2, memory: 128Gi }
          startupProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 10
            failureThreshold: 60
          readinessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /health, port: http }
            periodSeconds: 15
            failureThreshold: 4
          lifecycle:
            preStop:
              exec: { command: ["sleep", "15"] }
          volumeMounts:
            - { name: models, mountPath: /models, readOnly: true }
            - { name: dshm, mountPath: /dev/shm }
      volumes:
        - name: models
          persistentVolumeClaim: { claimName: model-store, readOnly: true }
        - name: dshm
          emptyDir: { medium: Memory, sizeLimit: 16Gi }
```

Read it field by field, as the reasons are the lesson.

- **The engine flags are the runbook's output.** Nothing here is chosen in the manifest; TP, precision, `max-model-len`, and `max-num-seqs` all came from the memory budget and Little's law.
- **The startup probe covers the cold start.** Ten-second periods with a failure threshold of 60 give the pod ten minutes to load weights and capture CUDA graphs before Kubernetes kills it. Without a startup probe, the liveness probe would start immediately, fail during loading, and restart the pod forever, never getting past the load. Set the budget from the cold-start math, about twice the measured worst case.
- **Readiness gates traffic.** `/health` returns success only once the engine is serving. Until then the pod receives no requests from the Service.
- **The preStop sleep and the grace period drain streams.** When a pod is deleted, it is removed from the Service's endpoints at roughly the same time it receives SIGTERM. The 15-second sleep lets load balancers stop sending new requests first; the 120-second grace lets in-flight generations finish rather than be cut off mid-stream. That covers outputs of about 2,000 tokens; [A request's path through the cluster](#a-requests-path-through-the-cluster) sizes it for longer ones.
- **`memory` has a limit; `cpu` does not.** A memory limit protects the node. A CPU limit throttles the tokenizer and scheduler under CFS quotas, which shows up as unexplained TTFT spikes.
- **`/dev/shm` is memory-backed.** This is the shared-memory fix from the pod-shape table. Its size counts against the pod's memory.

The Service in front of it is an ordinary `ClusterIP` on port 8000. Its round-robin load balancing is acceptable for one model without a large shared prefix; the routing section below says when it is not.

## Rollouts: surge needs spare GPUs

A rolling update replaces pods one at a time, and the two knobs trade spare hardware against capacity. `maxSurge: 1, maxUnavailable: 0`, as above, starts a new pod before stopping an old one, so capacity never drops, but the new pod needs two free GPUs on a node that matches. If every GPU is in use, the surge pod stays `Pending` and the rollout stalls, silently, forever. `maxSurge: 0, maxUnavailable: 1` needs no spare GPUs but removes one replica, a fifth of capacity in our fleet, for the duration of each pod's replacement, which is the whole cold start.

The arithmetic decides. A rollout of $R$ replicas, one at a time, takes about $R \times t_\text{cold}$. With five replicas and a four-minute cold start, that is 20 minutes during which, with `maxUnavailable: 1`, the fleet runs at $4/5$ of capacity. If peak load needs more than 80% of the fleet's goodput, roll out off-peak or keep a spare node. The six leftover GPUs in the worked fleet are exactly that spare.

A model change is a release, not a configuration change: a new revision can be worse in ways that load tests do not catch. Roll it out as a canary, a second Deployment with one replica and a share of traffic, measured against the old one, before scaling it up. The gates, the bake period, and the rollback saga are built in [ModelRelease](../../systems/07-durable-orchestration/12-model-release.md). Add a `PodDisruptionBudget` with `maxUnavailable: 1` so that node drains during cluster upgrades also take only one replica at a time, and spread replicas across zones with `topologySpreadConstraints` so that one zone outage does not take most of the fleet ([Failure domains and failover](#failure-domains-and-failover) sizes this).

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
2. **The AI gateway.** It checks the API key, applies rate limits and quotas, records usage, and picks a model route, as built in [Gateway](../25-ai-gateway/README.md).
3. **The endpoint picker or Service.** It chooses one pod, by engine metrics and prefix affinity or by round-robin.
4. **The engine's API server.** vLLM tokenizes the prompt and admits the request to its scheduler's waiting queue.
5. **The engine loop.** Continuous batching runs the request's prefill, then one decode step at a time, as described in [Inside an Inference Engine](06-inside-an-inference-engine.md). Each new token goes back as a server-sent event through every hop above.

Three settings along that path are wrong by default for LLM streams.

**Timeouts must outlast the longest generation.** Proxy hops have two kinds of timeout, and they need different values. A *total* timeout caps the whole request, however busy the stream: Envoy's route timeout, which defaults to 15 s and sits under many Gateway implementations; the HTTPRoute `timeouts.request` and `timeouts.backendRequest` fields; and some cloud backend timeouts, such as Google Cloud's 30 s default. Each must exceed the longest stream. An output cap of 4,096 tokens (an API `max_tokens` limit) at the 50 ms TPOT SLO streams for up to $4096 \times 0.05 \approx 205$ s, plus the time to first token. An *idle* timeout, such as an AWS load balancer's 60 s default, resets on every byte, so a streaming response renews it with each token. It must exceed the longest silence instead: the queue wait before the first token, or a non-streaming request's whole generation.

**The drain period must outlast it too.** The Deployment above gives a deleted pod 120 s of grace, and the 15 s preStop sleep counts against it. A request admitted at the very end of that sleep has 105 s left, which at 50 ms per token covers about 2,100 tokens. Streams already running keep generating during the sleep, so this is the worst case. Set `terminationGracePeriodSeconds` to the preStop sleep plus the longest-stream number. That works only if the engine finishes in-flight requests on SIGTERM rather than exiting at once. Check your engine version's shutdown behavior, and if it exits immediately, extend the preStop hook to wait for the running-request count to reach zero.

**Shed load at the front, not in readiness.** When traffic outruns capacity, the engine's waiting queue grows and every queued request's TTFT grows with it. Reject excess early, with a 429 or 503 and a `Retry-After` from the AI gateway or the endpoint picker, both of which can read queue depth. Do not fail the readiness probe on load: a busy pod that drops out of the endpoints pushes its share onto the others, which then drop out too, and the fleet flaps.

## Disaggregated prefill and decode as two pools

[Disaggregation](04-serving-and-load/README.md#disaggregated-prefill-and-decode) becomes, on Kubernetes, two Deployments (or two LeaderWorkerSets) of the same model: a prefill pool started with the engine's prefill role and a decode pool with the decode role, a KV connector between them (NIXL over RDMA), and a router that sends each request to a prefill worker, then a decode worker. The pools scale independently, which is the point: you size each for its own bottleneck.

Size them from first principles, using the worked chatbot (20 req/s, 2,000-token mean input of which 1,500 tokens are a cached system prompt, 300-token output, 70B FP8 on H100s):

- **Prefill demand.** Uncached input is $20 \times 500 = 10{,}000$ tokens per second. Prefill costs about $2P = 140$ GFLOP per token. An H100 does about 990 TFLOPS of dense FP8 at 50% utilization, so one GPU prefills about $990 \times 10^{12} / 140 \times 10^9 \approx 7{,}000$ tokens per second. That needs 1.4 GPUs: one TP=2 prefill instance.
- **Decode demand.** Output is $20 \times 300 = 6{,}000$ tokens per second. On a TP=2 decode instance at batch 160, each step reads 35 GB of weights per GPU plus the batch's KV ($160 \times 2{,}300 \times 80$ KiB $\approx 30$ GB per GPU), about 65 GB at an effective 2.35 TB/s (70% of 3.35 TB/s), so a step takes about 28 ms and the instance produces $160 / 0.028 \approx 5{,}700$ tokens per second. That is just under the demand, so two decode instances, with headroom.
- **The fleet.** By throughput alone, one prefill instance and two decode instances: 6 GPUs, against 10 for the aggregated design. The 28 ms step also stays under the 50 ms TPOT SLO because no prefill ever enters a decode batch. Add the rules from later in this section, though (at least two instances per pool, and survive losing one) and the fleet becomes two prefill and three decode instances: losing a decode instance leaves $2 \times 5{,}700 = 11{,}400$ tokens per second against 6,000. That is 10 GPUs, the same as aggregated. At this size disaggregation buys a steadier TPOT, not fewer GPUs; the GPU saving appears once the pools are large enough that the minimums stop dominating.

These are first-order numbers: they ignore communication time, attention compute, and the KV transfer itself, and they must be confirmed by load-testing each pool. But they show how to set the ratio, and why it moves with the traffic. If the shared prefix stops being cacheable, prefill demand quadruples and the prefill pool must grow while the decode pool stays the same, which an aggregated fleet cannot do without overbuying decode.

### Running the two pools

On Kubernetes the two pools are two workloads (Deployments, or LeaderWorkerSets for multi-node instances) built from the same image and the same weights, started with different engine roles. In vLLM, the `--kv-transfer-config` JSON names a connector such as NIXL and sets `kv_role` to `kv_producer` for prefill or `kv_consumer` for decode; SGLang uses `--disaggregation-mode prefill|decode`. A disaggregation-aware router sits in front and makes the two-hop call, picking a prefill worker and a decode worker per request. llm-d and NVIDIA Dynamo package that router, and [Routing, cascades, and failover](../25-ai-gateway/04-routing.md#disaggregated-prefill-and-decode) builds one by hand; [Disaggregated prefill and decode](../23-model-serving/07-disaggregated-prefill-decode.md) builds the engine side, including the KV hand-off.

Three Kubernetes concerns follow from the KV transfer:

- **Placement.** The KV cache crosses RDMA, so the prefill and decode pods must sit in the same network block, with the RDMA devices exposed to their containers, exactly as for a multi-node LeaderWorkerSet. Over TCP the transfer cost in [Serving, Capacity & Load Testing](04-serving-and-load/README.md#disaggregated-prefill-and-decode) erases the gain.
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

Spare capacity protects the requests that arrive after a failure. The ones already running on a dead pod follow the **commit rule**, which the gateway chapter calls [the commitment boundary](../25-ai-gateway/03-sse-streaming-proxy.md#the-commitment-boundary). If no byte has reached the client, the router retries on another replica, and the client sees only a slower first token. Once bytes have streamed, the stream ends with an SSE `error` event, because no other replica holds that sequence's KV cache. [A decode worker dies mid-stream](../../systems/12-incident-response-and-chaos/02-kill-decode.md) in *Systems Engineering* explains why splicing or silently closing the stream is worse, and runs this failure as a drill.

## Many fine-tunes on one base

When many teams or customers each have a LoRA fine-tune of the same base model, the deployment is one base Deployment that serves all the adapters, not one Deployment per fine-tune. Start vLLM with `--enable-lora`, `--max-loras` (adapters resident on the GPU per batch), and `--max-lora-rank`, and either list adapters at startup with `--lora-modules` or, with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`, load them at runtime through `/v1/load_lora_adapter`. Each adapter appears as its own model name on the OpenAI-compatible API. The economics are in [Serving, Capacity & Load Testing](04-serving-and-load/README.md#multi-lora-serverless-versus-dedicated-cold-start-and-autoscaling): an 84 MB adapter loads in seconds where a 16 GB full fine-tune needs its own GPU. The routing consequence is adapter affinity: an endpoint picker that knows which pods already have an adapter loaded avoids swapping it in on every request.

## Building at scale: many models, many teams

Past one model the hard problem changes from writing a Deployment to sharing a finite, expensive pool of GPUs among models, teams, and kinds of work. Six practices carry most of the weight.

- **One pool per model, one gateway in front.** Each model gets its own Deployment or LeaderWorkerSet, its own `InferencePool`, and its own autoscaler, because its goodput, flags, and traffic differ. One Gateway routes on the request's `model` field to the right pool, and the AI gateway applies keys and quotas per tenant. Popular models keep warm minimums; long-tail models either scale to zero and accept the cold start, or become LoRA adapters on a shared base.
- **Priority, quotas, and gang admission.** Serving, batch evaluation, and fine-tuning compete for the same GPUs. Give serving a higher `PriorityClass` so it preempts batch work during a spike. Use Kueue for per-team GPU quotas that can borrow idle capacity. Kueue also admits a multi-pod workload's whole quota at once (gang admission), and with `waitForPodsReady` it evicts a group whose pods cannot all start in time. Without that, half-scheduled groups, such as a two-node LeaderWorkerSet replica or a training job, hold GPUs and serve nothing.
- **Heterogeneous hardware.** Keep a node pool per GPU type, labeled by GPU Feature Discovery, and put each model on the cheapest GPU that meets its SLO, which the memory budget in [Serving, Capacity & Load Testing](04-serving-and-load/README.md) decides. Give each hardware type its own Deployment, since goodput, engine flags, and replica counts all change with the GPU.
- **Capacity is not elastic.** Cloud GPU stock runs out, and the cluster autoscaler cannot create an H100 the region does not have. Reserve capacity for the serving baseline and the failure headroom above; burst onto on-demand or spot capacity for batch and evaluation work, whose deadlines tolerate a reclaim.
- **Startup at scale.** A vLLM image is several gigabytes and the weights tens to hundreds more, and a 50-node scale-out multiplies both. Pre-pull images onto GPU nodes with a DaemonSet or use a lazy-pulling snapshotter, keep weights in node-local caches, and stagger restarts so a fleet-wide rollout does not become a storm.
- **The fleet as code, across clusters.** Keep one Helm chart parameterized per model, applied from Git, as in [Helm charts and the observability stack](../../systems/10-containers-kubernetes/04-helm-charts-and-observability-stack.md) and [Cloud Native](../../systems/09-cloud-native/README.md) in *Systems Engineering*, and release model revisions through [ModelRelease](../../systems/07-durable-orchestration/12-model-release.md). One cluster is both a failure domain and a scaling ceiling, so large fleets run several, in more than one region, behind a global gateway that routes by health and latency and fails over between them.

## What to watch

Two metric sources together tell you what a model fleet is doing. The engine's `/metrics` endpoint reports the serving view: `vllm:num_requests_waiting` and `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, preemptions, and histograms of TTFT and inter-token latency. The DCGM exporter reports the hardware view: `DCGM_FI_DEV_FB_USED` (GPU memory), `DCGM_FI_DEV_GPU_UTIL`, temperatures, and XID errors, which are the driver's code for a failing GPU. Alert on the SLO (TTFT and TPOT p99) and on XID errors; dashboard the rest. Tracing a single request through gateway, router, and engine is built in [Tracing for the serving path](../../systems/11-observability/02-tracing-for-the-serving-path.md).

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

The capacity plan this section deploys comes from [Serving, Capacity & Load Testing](04-serving-and-load/README.md), and the next section, [From Training Run to Shared Model](08-train-publish-share.md), covers the other side of the handoff: producing a checkpoint that loads cleanly here, publishing it, and the practical deployments that use it. The Kubernetes mechanics are built from scratch in [Containers and Kubernetes](../../systems/10-containers-kubernetes/README.md), including Helm charts and KEDA. Releasing a new model revision safely is [ModelRelease](../../systems/07-durable-orchestration/12-model-release.md), and putting keys, rate limits, and usage accounting in front of the fleet is [Gateway](../25-ai-gateway/README.md) in *AI Engineering*.

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
