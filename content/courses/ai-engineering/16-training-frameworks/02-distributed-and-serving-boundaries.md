# Distributed and serving boundaries

This part of [Training and Frameworks](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Explain which responsibilities belong to the learner and the serving runtime.

## Serving frameworks: vLLM and the stack

A framework defines a model; it does not serve one well. A naive `model.generate()` loop wastes the GPU, because concurrent, variable-length traffic leaves it idle between requests. Serving engines exist to keep the GPU saturated under that kind of load, and they sit on top of the framework: PyTorch or JAX defines the model, and vLLM, TGI, [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/), or SGLang serves it efficiently. The full treatment of how these products work, when to use each, and their tradeoffs is in [LLM Serving Platforms](../20-inference-engines/05-serving-platforms.md) and the rest of [LLM Systems & Inference](../20-inference-engines/).

**vLLM** is the default for self-hosting LLMs. It is a high-throughput inference engine built around a PagedAttention KV cache, continuous batching, tensor parallelism, and prefix caching, and it exposes an OpenAI-compatible server. **[NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)** takes a different approach: it compiles the model ahead of time into fused, hardware-specific kernels (with FP8 and in-flight batching) for maximum throughput and minimum latency on NVIDIA GPUs, and it is typically served behind [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/). The tradeoff against vLLM is a build step and less flexibility in exchange for peak performance.

Above the engine sits orchestration. **[Ray](https://docs.ray.io/)** is a common choice: **Ray Serve** wraps a serving engine (vLLM or TensorRT-LLM) with autoscaling, multi-model routing, and request batching across a cluster, while Ray Core's actor and task model is the distributed-compute substrate underneath and **Ray Train** handles the training side, discussed next. From the platform's point of view, the engine is one tier. Around it sit a model registry, an autoscaler (KEDA, Ray Serve, or KServe), a router, and observability; see [Cloud Native](../../systems/09-cloud-native/) in *Systems Engineering* for the substrate and [Streaming and SSE](../../systems/04-streaming/) for how tokens reach the client.

## Distributed training

Serving assumes the model already exists. Producing it at scale raises the opposite problem: the model may not fit on one GPU, and even when it does, one GPU is too slow. "Distributed training" is not a single switch. It is a choice of which dimension of the problem to split, and each choice costs a different amount of communication. There are four axes, and real systems combine them; "3D parallelism" is tensor times pipeline times data parallelism.

| Strategy | What it splits | Communication | Use when |
|----------|---------------|---------------|----------|
| **Data parallel (DDP)** | The *batch* (full model replica per GPU) | All-reduce gradients each step | Model fits on one GPU, want more throughput |
| **Tensor parallel (TP)** | Each *weight matrix* across GPUs | All-reduce per layer (heavy) | Layer too big; keep inside one node (NVLink) |
| **Pipeline parallel (PP)** | *Layers* into stages | Activations between stages | Model spans nodes; tolerate pipeline bubbles |
| **Expert parallel (EP)** | *MoE experts* across GPUs | All-to-all token routing | Mixture-of-Experts models |

Name the dimension you need to split, and the tool follows.

### Sharding the optimizer: ZeRO and FSDP

The surprising fact about training memory is that the weights are not the dominant cost. The **optimizer state** is: Adam keeps two moments per parameter, and together with gradients and activations the total often runs four to eight times the size of the weights in fp32. A model whose weights fit comfortably on a GPU may be impossible to train there.

**ZeRO** (in DeepSpeed) and **FSDP** (in PyTorch) attack exactly that cost. They shard parameters, gradients, and optimizer state across the data-parallel ranks, gather a layer's full weights only for its forward and backward pass, and release them afterward. ZeRO's stages 1, 2, and 3 shard progressively more of that state. The result is that you can train a model far larger than any single GPU's memory without resorting to full tensor parallelism. The general lesson is to shard the most expensive thing: find the dominant cost before deciding what to split.

### The tooling and the habits

A small set of tools does most of this work. **DeepSpeed** provides ZeRO and offload to CPU or NVMe. **Megatron-LM** provides tensor, pipeline, and sequence parallelism and is the reference implementation for large pretraining. **PyTorch FSDP** is the native option, and **HuggingFace Accelerate** lets one config drive many backends. **[Ray Train](https://docs.ray.io/en/latest/train/train.html)** orchestrates multi-node training over Ray Core, handling data loading, fault tolerance, and elastic scaling around PyTorch or DeepSpeed. Underneath all of them is **NCCL**, the GPU collective-communication library.

Three habits are standard on any serious run. **Checkpointing** must be distributed, sharded, and asynchronous so a multi-day run survives a node failure; this is the same checkpoint pattern that [Durable Orchestration and Workers](../../systems/07-durable-orchestration/) builds for workflows. **Mixed precision** (bf16 or fp16 compute with fp32 master weights) cuts memory and time. **Gradient (activation) checkpointing** recomputes activations during the backward pass instead of storing them, trading compute for memory.

## Summary

Explain which responsibilities belong to the learner and the serving runtime. State the assumptions before applying the result to a different workload.

## Review questions

1. Why is a training checkpoint not a complete serving artifact?

   <details><summary>Answer</summary>

   Serving also needs the architecture, tokenizer, template, position rules, precision, and generation contract.

   </details>

[Chapter contents](README.md) · [Next: Embeddings, small models, and platforms](03-embeddings-small-models-and-platforms.md)
