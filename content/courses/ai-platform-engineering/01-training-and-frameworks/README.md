# Training & Frameworks

## Overview

- **Primary references**: [PyTorch docs](https://pytorch.org/docs/stable/index.html) (free), [JAX docs](https://jax.readthedocs.io/) (free), [vLLM docs](https://docs.vllm.ai/) (free)
- **Supplementary**: [HuggingFace Transformers](https://huggingface.co/docs/transformers) + [Accelerate](https://huggingface.co/docs/accelerate) + [PEFT](https://huggingface.co/docs/peft) docs (free), [DeepSpeed](https://www.deepspeed.ai/) + [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (free), [Ray Train / Ray Serve](https://docs.ray.io/) (distributed training + serving, free), [NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) + [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/) (free), [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [Sentence-Transformers](https://www.sbert.net/) (free), [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (free)
- **Prerequisites**: [Deep Learning](../../machine-learning/02-deep-learning/) (backprop, optimizers, transformers), [LLM Systems & Inference](../../machine-learning/04-llm-systems/) (serving, parallelism)
- **Estimated time**: 4-6 weeks at 10-12 hrs/week

## Key Takeaways

- **PyTorch is eager-by-default with an opt-in compiler; JAX is functional and compiler-first.** PyTorch optimizes for debuggability and ecosystem; JAX optimizes for `jit`/`vmap`/`pmap` composability and TPU. They converge on XLA.
- **Distributed training is a parallelism choice, not a single switch.** Data, tensor, pipeline, and expert parallelism each cut a different dimension of the problem and cost a different amount of communication. FSDP/ZeRO shard the *optimizer state* to fit big models on small GPUs.
- **Embedding models are their own product.** Creating one (contrastive fine-tuning), hosting one (low-latency batched inference), and indexing its output (a vector store) are three distinct platform jobs.
- **Small language models win on cost.** A fine-tuned 3-8B model with LoRA + quantization often beats a frontier API on a narrow task at a fraction of the price and latency.
- **The platform owns the loop**: data → training (checkpoints) → evaluation → registry → serving (vLLM) → monitoring → back to data.

## How to Study

- Train the *same* small model three ways: a plain PyTorch loop, `torch.compile`, and JAX `jit`. Watch the speedups and the debugging differences.
- Take a model that won't fit on one GPU and make it fit with FSDP, then with DeepSpeed ZeRO-3. Read the GPU memory numbers as you go.
- Fine-tune an embedding model on a domain dataset, serve it, and measure recall@k against the base model on your own retrieval set.
- Distill or LoRA-tune a small model for one task and benchmark it (quality, $/1M tokens, p99 latency) against calling a large model.

---

# Concepts & Techniques

## Core Insight

A framework is a contract between the math and the hardware. You write a model as composable tensor operations; the framework records them (a graph or a trace), computes gradients by reverse-mode autodiff, and hands the graph to a compiler (XLA, Inductor, TensorRT) that fuses kernels and targets the accelerator. Everything in this topic — eager vs compiled, the parallelism zoo, LoRA, quantization — is about getting that contract to fit on the hardware you have and the budget you have. Training is throughput over a fixed dataset; serving is latency over an open stream; the platform engineer's job is to make both economical on the same fleet.

## 1. PyTorch

**The default research-and-production framework**

**Key ideas**:
- **Eager execution + dynamic graphs**: ops run immediately, control flow is plain Python, autograd records a tape per forward pass. This is why PyTorch is easy to debug.
- **`torch.compile`** (TorchDynamo + Inductor): trace the eager program, capture graphs, and JIT-compile fused kernels — eager ergonomics with much of the speed of a static graph.
- **`nn.Module`, optimizers, autograd**: the model is a tree of modules; `loss.backward()` walks the tape; the optimizer (`AdamW`) steps parameters.
- **Distributed primitives**: `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP), and the `torch.distributed` collectives (all-reduce, all-gather, reduce-scatter) underneath.
- **Ecosystem**: TorchData, TorchEval, the HuggingFace stack, `torchao` for quantization. The gravity well of the field.

## 2. JAX

**Functional, composable, compiler-first**

**Key ideas**:
- **Pure functions + transformations**: `jit` (compile), `grad` (autodiff), `vmap` (auto-batch), `pmap`/`shard_map` (parallelize). They *compose* — `jit(vmap(grad(f)))` is one fused, batched, differentiated, compiled function.
- **Functional state**: no in-place mutation; parameters are pytrees passed explicitly. Optimizers live in libraries (Optax), models in Flax/Equinox/Haiku.
- **XLA-native**: built for the XLA compiler, first-class on TPU and strong on GPU. The mental model is "stage a whole computation, then compile it."
- **`pjit` / sharding**: express device meshes and tensor sharding declaratively; the compiler inserts the collectives. This is how large models train on TPU pods.

**PyTorch vs JAX**:

| | PyTorch | JAX |
|---|---------|-----|
| Style | Imperative, eager | Functional, staged |
| Debugging | Native Python, step-through | Trace abstractions; print inside `jit` is tricky |
| Parallelism | DDP/FSDP, explicit | `pmap`/`pjit`/sharding, compiler-inserted |
| Hardware | GPU-first, TPU via XLA | TPU-first, strong GPU |
| Ecosystem | Largest (HF, vendors) | Smaller, research-heavy (Flax, Optax) |

## 3. Serving Frameworks (vLLM and the Stack)

For how the products work, when to use them, and their trade-offs, see [LLM Serving Platforms](../../machine-learning/04-llm-systems/serving-platforms.md).

**Where training output becomes a product**

**Key ideas**:
- **vLLM**: high-throughput inference engine — PagedAttention KV cache, continuous batching, tensor parallelism, prefix caching, an OpenAI-compatible server. The default for self-hosting LLMs.
- **The serving layer sits on top of the framework**: PyTorch/JAX define the model; vLLM/TGI/[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)/SGLang *serve* it efficiently. (Full treatment in [LLM Systems & Inference](../../machine-learning/04-llm-systems/).)
- **[NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)**: ahead-of-time *compiles* the model into fused, hardware-specific kernels (FP8, in-flight batching) for max throughput/latency on NVIDIA GPUs — typically served behind [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/). The trade vs vLLM: a build step and less flexibility for peak performance.
- **Why a separate engine**: a naive `model.generate()` loop wastes the GPU; serving engines exist to keep it saturated under variable, concurrent traffic.
- **[Ray](https://docs.ray.io/) as the orchestration layer**: **Ray Serve** wraps a serving engine (vLLM/TensorRT-LLM) with autoscaling, multi-model routing, and request batching across a cluster — the tier *above* the engine. (Ray Core's actor/task model is the distributed-compute substrate; **Ray Train** does the training side, below.)
- **Platform view**: the engine is one tier; around it sit a model registry, autoscaler (KEDA / Ray Serve / KServe), router, and observability — see [Cloud Native](../../systems/03-cloud-native/) and [Streaming & SSE](../03-streaming-sse/) for token delivery.

## 4. Distributed Training

**Making a model that doesn't fit, fit — and training faster**

The four axes of parallelism (you combine them — "3D parallelism" is TP × PP × DP):

| Strategy | What it splits | Communication | Use when |
|----------|---------------|---------------|----------|
| **Data parallel (DDP)** | The *batch* (full model replica per GPU) | All-reduce gradients each step | Model fits on one GPU, want more throughput |
| **Tensor parallel (TP)** | Each *weight matrix* across GPUs | All-reduce per layer (heavy) | Layer too big; keep inside one node (NVLink) |
| **Pipeline parallel (PP)** | *Layers* into stages | Activations between stages | Model spans nodes; tolerate pipeline bubbles |
| **Expert parallel (EP)** | *MoE experts* across GPUs | All-to-all token routing | Mixture-of-Experts models |

**Sharding the optimizer — ZeRO / FSDP**:
- The hidden memory cost of training isn't the weights, it's the **optimizer state** (Adam keeps two moments per parameter) + gradients + activations — often 4-8× the weights in fp32.
- **ZeRO** (DeepSpeed) and **FSDP** (PyTorch) shard parameters, gradients, and optimizer state across data-parallel ranks, gathering each layer's full weights only for its forward/backward, then releasing them. ZeRO stages 1/2/3 shard progressively more.
- This is what lets you train a model far larger than any single GPU's memory without full tensor parallelism.

**Key tooling**:
- **DeepSpeed** (ZeRO, offload to CPU/NVMe), **Megatron-LM** (TP/PP/sequence parallelism, the reference for large pretraining), **PyTorch FSDP**, **HuggingFace Accelerate** (one config, many backends), **[Ray Train](https://docs.ray.io/en/latest/train/train.html)** (orchestrates multi-node training over Ray Core — data loading, fault tolerance, and elastic scaling around PyTorch/DeepSpeed), **NCCL** (the GPU collective library underneath it all).
- **Checkpointing**: distributed, sharded, asynchronous checkpoints so a multi-day run survives a node failure (the same checkpoint pattern as durable orchestration in [topic 05](../05-durable-orchestration-and-workers/)).
- **Mixed precision** (bf16/fp16 + fp32 master weights) and **gradient/activation checkpointing** (recompute activations to save memory) are standard.

## 5. Creating & Hosting Embedding Models

**Embeddings are the retrieval substrate of RAG and search**

**Creating**:
- An embedding model maps text (or image/audio) to a dense vector so that semantic similarity ≈ cosine similarity. Start from a base encoder (BERT-family, E5, GTE, BGE, Nomic) and fine-tune.
- **Contrastive training**: pull (query, positive) pairs together and push negatives apart. Losses: InfoNCE / multiple-negatives-ranking, triplet. **Hard negatives** (mined, near-miss) matter far more than easy ones.
- **Tooling**: [Sentence-Transformers](https://www.sbert.net/) for fine-tuning; **Matryoshka** representation learning lets one model emit usable shorter vectors (truncate for cheaper storage); evaluate on [MTEB](https://huggingface.co/spaces/mteb/leaderboard).
- **Why fine-tune**: a domain-tuned embedder lifts retrieval recall far more than a bigger LLM does — retrieval quality caps RAG quality.

**Hosting**:
- Embedding inference is **encoder-only and bidirectional** — no KV cache, no autoregressive decode. It's a throughput problem: big batches, short sequences, often fine on CPU or a small GPU.
- Serve via TEI ([Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference)), vLLM's embeddings endpoint, or a plain ONNX/Triton service. Normalize vectors; pin the model version (re-embedding the whole corpus on a model change is expensive).
- **Write to a vector store** — pgvector (Postgres), Qdrant, Milvus, Weaviate, FAISS — with an ANN index (HNSW/IVF). This bridges to [topic 04](../04-distributed-data-and-caching/) (the store is sharded, cached, durable data) and to RAG retrieval ([topic 07](../07-retrieval-and-rag/)).

## 6. Small Language Models & Training

**The production workhorse**

**Key ideas**:
- **Why SLMs (≈1-8B)**: cheap to fine-tune, fast to serve, fit on one GPU (or CPU/edge when quantized), and *good enough* — or better — on narrow tasks. The platform default when you don't need a frontier model.
- **Getting a good SLM**:
  - **Distillation**: train the small model to match a large teacher's outputs (logits or generated data) — transfer capability into a cheaper package.
  - **Fine-tuning**: choose by the signal you have. **SFT** needs input and correct-output pairs. **DPO** needs chosen-versus-rejected pairs. **RFT** (reinforcement fine-tuning) needs a scorer that returns a reward, so it fits verifiable tasks where you have no gold answers; watch for reward hacking. The ladder, with worked examples, is in [Model Routing & Cascades §9](../11-model-routing-and-cascades/).
  - **PEFT / LoRA / QLoRA**: freeze the base model, train small low-rank adapter matrices. QLoRA fine-tunes a 4-bit-quantized base, so a 7B model tunes on a single consumer GPU. Adapters are tiny (MBs) and swappable — serve many tasks from one base.
- **Quantize for serving**: GPTQ/AWQ/bitsandbytes shrink the SLM further (see [Quantization](../../machine-learning/04-llm-systems/quantization/)).
- **The build-vs-buy decision**: a fine-tuned SLM trades a fixed training cost for a much lower per-token serving cost and no vendor dependency — the core platform economics calculation.

## 7. The Platform Loop (MLOps)

**Key ideas**:
- **Lifecycle**: data versioning → training (distributed, checkpointed) → evaluation (held-out + task benchmarks) → **model registry** (versioned, signed weights) → serving (vLLM) → **monitoring** (drift, quality, cost) → back to data.
- **Experiment tracking**: Weights & Biases / MLflow for runs, metrics, artifacts.
- **Orchestration**: training and data pipelines are multi-step, long-running, and failure-prone — which is exactly what **durable orchestration** ([topic 05](../05-durable-orchestration-and-workers/)) is for.
- **Reproducibility**: pin data, seed, framework, and CUDA versions; a model you can't rebuild is a liability.

---

## Decision Cheat Sheet

| Situation | Reach for |
|-----------|-----------|
| Research, max ecosystem, GPU | PyTorch (+ `torch.compile`) |
| TPU, want `vmap`/`pmap` composability | JAX (+ Flax/Optax) |
| Serve an LLM at high throughput | vLLM |
| Max performance on NVIDIA (compiled) | TensorRT-LLM (+ Triton) |
| Scale training or serving across a cluster | Ray (Ray Train / Ray Serve) |
| Model fits on one GPU, want speed | Data parallel (DDP) |
| Model won't fit, single node | FSDP / ZeRO-3 (shard optimizer state) |
| Huge model across nodes | 3D parallelism (TP×PP×DP), Megatron |
| Cheap retrieval for RAG | Fine-tuned embedding model + vector store |
| Cheap, fast, narrow task | SLM + QLoRA + quantization |
| Labelled input and output pairs | SFT |
| Preference pairs (accepted vs rejected) | DPO |
| A verifiable outcome, no gold answers | RFT, with your eval as the reward |

## Patterns Worth Internalizing

These outlive any framework version:

- **"Distributed training" is a parallelism choice along a specific dimension** — split the batch (data), the matrix (tensor), the layers (pipeline), or the experts (expert). Name the dimension and the tool follows.
- **Shard the most expensive thing.** FSDP/ZeRO shard optimizer state because that, not the weights, is what blows up memory — find the dominant cost before splitting.
- **Freeze the base, train a small adapter** (LoRA) — the general pattern of "perturb a frozen pretrained artifact cheaply," reused across fine-tuning, prompt-tuning, and control.
- **Eager for development, compiled for production** (`torch.compile` / `jit`) — the same staged-execution trade-off shows up in every numeric framework.
- **Eval is part of the loop, not the end** — a model you can't measure is a model you can't improve (see [LLM Evaluation](../09-llm-evaluation/)).

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Transformers, backprop, optimizers | [Deep Learning](../../machine-learning/02-deep-learning/) | The models being trained |
| Serving, KV cache, parallelism, quantization | [LLM Systems & Inference](../../machine-learning/04-llm-systems/) | Deploying training output |
| K8s, GPU scheduling, autoscaling | [Cloud Native](../../systems/03-cloud-native/) | The training/serving substrate |
| Vector stores, OLTP/OLAP, sharding, caching | [Distributed Data & Caching](../04-distributed-data-and-caching/) | Where embeddings and data live |
| Durable workflows, checkpointed pipelines | [Orchestration & Workers](../05-durable-orchestration-and-workers/) | Crash-proof training pipelines |
| Embedding retrieval, chunking, RAG | [Retrieval & RAG](../07-retrieval-and-rag/) | What the embeddings are for |
| Token streaming to clients | [Streaming & SSE](../03-streaming-sse/) | Delivering generated output |
| llama.cpp/GGUF, edge, efficiency architectures | [Edge, Realtime & On-Device](../10-edge-realtime-inference/) | Running models off the GPU fleet |

## How Companies Apply These Patterns

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Anthropic / OpenAI | Large-scale distributed pretraining, checkpointing, 3D parallelism | PhD-level |
| Google / DeepMind | JAX + TPU pods, `pjit` sharding, pathways | PhD-level |
| Meta | PyTorch/FSDP (they build it), LLaMA training, quantization | Expert |
| NVIDIA | Megatron-LM, NeMo, TensorRT, NCCL, GPU efficiency | Expert |
| Databricks / Mosaic | Training platform, distributed training as a product | Expert |
| HuggingFace | Transformers/Accelerate/PEFT/TEI, the open stack | Expert |
| Cohere / Voyage | Embedding models as a product (creating + hosting) | Expert |
