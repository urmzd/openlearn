# Deploying LLMs on Kubernetes: GPUs, Replicas, and Rollouts

The capacity math in [Serving, Capacity & Load Testing](../04-serving-and-load/README.md) ends with a number: so many replicas, each needing so many GPUs, with a given set of engine flags. This section turns that number into a running fleet on Kubernetes. Kubernetes was designed for small, cheap, stateless pods that start in seconds and can be packed anywhere. An LLM replica is the opposite: it asks for whole GPUs that are scarce and expensive, it needs tens or hundreds of gigabytes of weights before it can answer anything, it takes minutes to start, and the way you route requests to it changes its throughput. Almost every default (round-robin Services, CPU autoscaling, surge rollouts, quick liveness probes) is wrong for it, and each wrong default has a cost you can compute.

You build the deployment in the order a platform engineer would: the arithmetic that sets the pod's shape and the replica count, the GPU node pool, getting weights from Hugging Face onto the node, the Deployment itself, rollouts, autoscaling, multi-node replicas, routing and the path a request takes through the cluster, disaggregated prefill and decode as Kubernetes objects, failure and failover, and running many models on one shared GPU cluster. Each step comes with the manifest and the reason for each field.

You should have read the chapter opening, [LLM Systems & Inference](../README.md), [Inside an Inference Engine](../06-inside-an-inference-engine.md), which explains what each pod below is doing as it starts, and the memory budget and runbook in [Serving, Capacity & Load Testing](../04-serving-and-load/README.md). The Kubernetes basics (Deployments, Services, Helm, KEDA) are built step by step in [Containers and Kubernetes](../../../systems/10-containers-kubernetes/README.md) in *Systems Engineering*; this section assumes them and adds what changes when the workload is a GPU-bound model server. How weights move from the Hub to GPU memory is covered byte by byte in [Model Loading: Hub to GPU](../03-model-loading/README.md).

> **Note:** This layer moves fast. The manifests were written in October 2026 against Kubernetes 1.34, the NVIDIA GPU Operator, KEDA 2.x, LeaderWorkerSet v1, and vLLM 0.31. Pin image tags and chart versions to the ones you tested.

> **Learning objectives**
> - Derive a pod's GPU request, the replica count, and the total GPU count from a model's memory budget and a measured per-replica goodput.
> - Explain how the NVIDIA device plugin exposes GPUs and how taints, labels, MIG, and time-slicing decide where a model pod lands.
> - Choose a strategy for getting Hugging Face weights onto nodes, and compute the cold-start and scale-out cost of each.
> - Write a vLLM Deployment whose probes, shared memory, rollout strategy, and termination grace match how a model server actually behaves.
> - Autoscale on queue depth with enough lead time, run a model across nodes with LeaderWorkerSet, and size separate prefill and decode pools.
> - Trace a request through the gateway, the endpoint picker, and the engine, and set proxy timeouts and drain periods from the longest stream.
> - Size a fleet to survive the loss of a replica, a node, or a zone, and run disaggregated pools and many models on a shared GPU cluster.

## In this part

1. [Capacity and GPU nodes](01-capacity-and-gpu-nodes.md): Turn a measured per-replica capacity into a pod shape and node pool.
2. [Weight staging and deployment](02-weight-staging-and-deployment.md): Trace immutable model artifacts into a ready serving replica.
3. [Rollouts, autoscaling, and routing](03-rollouts-autoscaling-and-routing.md): Budget rollout and scale-out lead time with the required spare resources.
4. [Disaggregated pools and failover](04-disaggregated-pools-and-failover.md): Account for KV handoff, ownership, and failure domains across worker pools.
5. [Adapter fleets and operations](05-adapter-fleets-and-operations.md): Operate model and adapter fleets using latency, quality, and memory evidence.
