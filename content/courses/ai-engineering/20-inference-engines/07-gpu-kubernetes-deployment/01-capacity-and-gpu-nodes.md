# Capacity and GPU nodes

This part of [Deploying LLMs on Kubernetes: GPUs, Replicas, and Rollouts](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Turn a measured per-replica capacity into a pod shape and node pool.

## From the capacity plan to a pod shape

Start with the arithmetic, because every manifest field below follows from it. A serving fleet has three numbers:

$$\text{GPUs per replica} = \text{TP} \times \text{PP}, \qquad \text{replicas} = \left\lceil \frac{\text{peak req/s}}{\text{goodput per replica} \times 0.7} \right\rceil, \qquad \text{total GPUs} = \text{replicas} \times \text{TP} \times \text{PP}$$

The first comes from the memory budget: the smallest TP (and PP, if one node is not enough) whose KV pool holds the in-flight tokens Little's law predicts. The second comes from a load test of one replica at the SLO; the 0.7 is headroom so that a traffic spike or the loss of one replica does not breach the SLO. Surviving the loss of a whole node is sized separately, in [Failure domains and failover](04-disaggregated-pools-and-failover.md#failure-domains-and-failover). Data parallelism, the replica count, is the Kubernetes `replicas` field. Tensor and pipeline parallelism live *inside* a replica and become its GPU request.

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

Follow one model pod from request to running engine to see where each piece acts. The components named here are introduced in [The control plane: who does what](../../../systems/10-containers-kubernetes/README.md#the-control-plane-who-does-what) in *Systems Engineering*.

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

## Review questions

1. Why does GPU count alone fail to determine fleet capacity?

   <details><summary>Answer</summary>

   Replica layout, memory, model loading, request mix, SLOs, and failure reserve all affect usable capacity.

   </details>

[Contents](README.md) · [Next: Weight staging and deployment](02-weight-staging-and-deployment.md)
