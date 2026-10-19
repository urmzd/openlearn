# Training and Frameworks

A framework is a contract between the math and the hardware. You write a model as composable tensor operations; the framework records them (as a graph or a trace), computes gradients by reverse-mode autodiff, and hands the graph to a compiler (XLA, Inductor, TensorRT) that fuses kernels and targets the accelerator. Everything in this chapter, from eager versus compiled execution to the parallelism zoo, LoRA, and quantization, is about getting that contract to fit on the hardware you have and the budget you have.

Training and serving pull in different directions. Training is throughput over a fixed dataset; serving is latency over an open stream. The platform engineer's job is to make both economical on the same fleet, and to own the loop that connects them: data, then training with checkpoints, then evaluation, then a registry, then serving, then monitoring, and back to data. This chapter opens the book because every later chapter, from the gateway to retrieval to evaluation, sits on top of models that someone trained, fine-tuned, and deployed with these tools.

The chapter moves from the bottom of the stack upward. It starts with the two frameworks that define models, PyTorch and JAX, then the serving engines that turn a trained model into a product. It then takes on distributed training, the problem of making a model that does not fit on one GPU fit anyway. The last three parts cover the platform jobs that sit around a model: creating and hosting embedding models, getting a small language model good enough to replace a frontier API, and running the whole lifecycle as a loop.

> **Learning objectives**
> - Explain the difference between PyTorch's eager-by-default execution and JAX's functional, compiler-first design, and choose between them for a given workload.
> - Name the four axes of parallelism (data, tensor, pipeline, expert), what each splits, and what each costs in communication.
> - Explain why ZeRO and FSDP shard optimizer state, and estimate why training memory runs several times the size of the weights.
> - Describe how an embedding model is created (contrastive fine-tuning), hosted (batched encoder inference), and indexed (a vector store).
> - Make the build-versus-buy case for a fine-tuned small language model, and pick SFT, DPO, or RFT from the training signal you have.

## Before you begin

This chapter assumes you know how neural networks are trained: backpropagation, optimizers such as Adam, and the transformer architecture, all covered in [Deep Learning](../../machine-learning/02-deep-learning/) in *Machine Learning & AI*. It also leans on [LLM Systems & Inference](../../machine-learning/04-llm-systems/) from the same book for serving, the KV cache, and the basics of parallelism. You do not need to have trained a large model yourself; the point here is to understand the choices well enough to make them on a platform.

## In this chapter

- [PyTorch](#pytorch): the default research-and-production framework, eager by default with an opt-in compiler.
- [JAX](#jax): a functional, compiler-first framework whose transformations compose, and how it compares with PyTorch.
- [Serving frameworks](#serving-frameworks-vllm-and-the-stack): where training output becomes a product, from vLLM to TensorRT-LLM and Ray Serve.
- [Distributed training](#distributed-training): the four axes of parallelism and the optimizer-state sharding that lets big models fit on small GPUs.
- [Creating and hosting embedding models](#creating-and-hosting-embedding-models): contrastive fine-tuning, encoder serving, and the vector store.
- [Small language models](#small-language-models): distillation, fine-tuning, LoRA, and the economics of a cheap, narrow model.
- [The platform loop](#the-platform-loop): the MLOps lifecycle that connects all of the above.

## PyTorch

PyTorch is the default framework for both research and production, and the reason is debuggability. It uses **eager execution with dynamic graphs**: each operation runs immediately when Python reaches it, control flow is ordinary Python, and autograd records a tape during every forward pass. When something goes wrong you can set a breakpoint and inspect a tensor, which is exactly what you cannot easily do inside a compiled graph.

The model itself is a tree of `nn.Module` objects. Calling `loss.backward()` walks the autograd tape in reverse to compute gradients, and an optimizer such as `AdamW` then steps the parameters. Speed comes from **`torch.compile`** (TorchDynamo plus Inductor), which traces the eager program, captures graphs, and JIT-compiles fused kernels. That gives you eager ergonomics while you develop and much of the speed of a static graph when you run.

For scale, PyTorch provides `DistributedDataParallel` (DDP) and `FullyShardedDataParallel` (FSDP), built on the `torch.distributed` collectives (all-reduce, all-gather, reduce-scatter) that you will meet again under distributed training below. Around the core sits the largest ecosystem in the field: TorchData, TorchEval, the HuggingFace stack, and `torchao` for quantization. That ecosystem is a gravity well; most new models, papers, and vendor integrations arrive in PyTorch first.

## JAX

JAX takes the opposite starting point. It is functional and compiler-first: you write **pure functions** and apply **transformations** to them. `jit` compiles, `grad` differentiates, `vmap` batches automatically, and `pmap` or `shard_map` parallelize. The transformations compose, so `jit(vmap(grad(f)))` is a single fused, batched, differentiated, compiled function. That composability is JAX's central advantage.

The price is **functional state**. There is no in-place mutation; parameters are pytrees passed explicitly into and out of every function. Optimizers live in libraries such as Optax, and models in Flax, Equinox, or Haiku. JAX was built for the XLA compiler, so it is first-class on TPU and strong on GPU, and its mental model is "stage a whole computation, then compile it." For large models, `pjit` and the sharding API let you describe device meshes and tensor sharding declaratively; the compiler inserts the collectives. This is how large models train on TPU pods.

The two frameworks differ along a handful of axes, summarized here:

| | PyTorch | JAX |
|---|---------|-----|
| Style | Imperative, eager | Functional, staged |
| Debugging | Native Python, step-through | Trace abstractions; print inside `jit` is tricky |
| Parallelism | DDP/FSDP, explicit | `pmap`/`pjit`/sharding, compiler-inserted |
| Hardware | GPU-first, TPU via XLA | TPU-first, strong GPU |
| Ecosystem | Largest (HF, vendors) | Smaller, research-heavy (Flax, Optax) |

In practice they are converging: both increasingly lower to XLA-style compilers, and the eager-for-development, compiled-for-production split appears on both sides. Pick PyTorch when ecosystem and debuggability on GPUs matter most; pick JAX when you are on TPUs or want `vmap` and `pmap` composability.

## Serving frameworks: vLLM and the stack

A framework defines a model; it does not serve one well. A naive `model.generate()` loop wastes the GPU, because concurrent, variable-length traffic leaves it idle between requests. Serving engines exist to keep the GPU saturated under that kind of load, and they sit on top of the framework: PyTorch or JAX defines the model, and vLLM, TGI, [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/), or SGLang serves it efficiently. The full treatment of how these products work, when to use each, and their tradeoffs is in [LLM Serving Platforms](../../machine-learning/04-llm-systems/serving-platforms.md) and the rest of [LLM Systems & Inference](../../machine-learning/04-llm-systems/).

**vLLM** is the default for self-hosting LLMs. It is a high-throughput inference engine built around a PagedAttention KV cache, continuous batching, tensor parallelism, and prefix caching, and it exposes an OpenAI-compatible server. **[NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)** takes a different approach: it compiles the model ahead of time into fused, hardware-specific kernels (with FP8 and in-flight batching) for maximum throughput and minimum latency on NVIDIA GPUs, and it is typically served behind [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/). The tradeoff against vLLM is a build step and less flexibility in exchange for peak performance.

Above the engine sits orchestration. **[Ray](https://docs.ray.io/)** is a common choice: **Ray Serve** wraps a serving engine (vLLM or TensorRT-LLM) with autoscaling, multi-model routing, and request batching across a cluster, while Ray Core's actor and task model is the distributed-compute substrate underneath and **Ray Train** handles the training side, discussed next. From the platform's point of view, the engine is one tier. Around it sit a model registry, an autoscaler (KEDA, Ray Serve, or KServe), a router, and observability; see [Cloud Native](../../systems/03-cloud-native/) in *Systems & Architecture* for the substrate and [Streaming and SSE](../03-streaming-sse/) for how tokens reach the client.

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

Three habits are standard on any serious run. **Checkpointing** must be distributed, sharded, and asynchronous so a multi-day run survives a node failure; this is the same checkpoint pattern that [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/) builds for workflows. **Mixed precision** (bf16 or fp16 compute with fp32 master weights) cuts memory and time. **Gradient (activation) checkpointing** recomputes activations during the backward pass instead of storing them, trading compute for memory.

## Creating and hosting embedding models

Not every model on the platform generates text. Embedding models are the retrieval substrate of RAG and search, and they are their own product: creating one, hosting one, and indexing its output are three distinct platform jobs.

### Creating an embedding model

An embedding model maps text (or images or audio) to a dense vector so that semantic similarity becomes cosine similarity. You rarely train one from scratch. Instead you start from a base encoder (the BERT family, E5, GTE, BGE, Nomic) and fine-tune it with **contrastive training**: pull (query, positive) pairs together and push negatives apart, using a loss such as InfoNCE (multiple-negatives ranking) or a triplet loss. The negatives matter more than anything else. **Hard negatives**, mined near-misses that look relevant but are not, teach the model far more than easy ones.

[Sentence-Transformers](https://www.sbert.net/) is the usual fine-tuning tool. **Matryoshka** representation learning lets one model emit vectors that remain useful when truncated, so you can store shorter vectors cheaply. The [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard is the standard evaluation. The reason to bother is that retrieval quality caps RAG quality: a domain-tuned embedder lifts retrieval recall far more than a bigger LLM does.

### Hosting it

Embedding inference is a different workload from generation. The model is **encoder-only and bidirectional**: there is no KV cache and no autoregressive decode. That makes it a throughput problem of big batches over short sequences, and it often runs fine on a CPU or a small GPU. You can serve it with TEI ([Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference)), vLLM's embeddings endpoint, or a plain ONNX or Triton service. Normalize the vectors, and pin the model version: changing the model means re-embedding the whole corpus, which is expensive.

The vectors then go into a **vector store** with an approximate-nearest-neighbor index (HNSW or IVF): pgvector in Postgres, Qdrant, Milvus, Weaviate, or FAISS. That store is sharded, cached, durable data, which is the subject of [Distributed Data and Caching](../04-distributed-data-and-caching/), and it is what [Retrieval and RAG](../07-retrieval-and-rag/) queries.

## Small language models

The same economic logic that favors a fine-tuned embedder favors small generative models. Small language models, roughly 1 to 8 billion parameters, are the production workhorse. They are cheap to fine-tune, fast to serve, fit on one GPU (or on a CPU or edge device when quantized), and are good enough, or better, on narrow tasks. When you do not need a frontier model, an SLM is the platform default. A fine-tuned 3 to 8B model with LoRA and quantization often beats a frontier API on a narrow task at a fraction of the price and latency.

There are three ways to make a small model good at your task. **Distillation** trains the small model to match a large teacher's outputs, either its logits or data it generates, transferring capability into a cheaper package. **Fine-tuning** adapts the model directly, and the method you choose depends on the signal you have. SFT needs input and correct-output pairs. DPO needs chosen-versus-rejected pairs. RFT (reinforcement fine-tuning) needs a scorer that returns a reward, so it fits verifiable tasks where you have no gold answers; watch it for reward hacking. The full ladder, with worked examples, is in [Model Routing and Cascades](../11-model-routing-and-cascades/), under tuning the router.

The third technique makes fine-tuning cheap enough to do routinely. **PEFT methods such as LoRA and QLoRA** freeze the base model and train small low-rank adapter matrices. QLoRA fine-tunes on top of a 4-bit-quantized base, so a 7B model tunes on a single consumer GPU. Adapters are tiny (megabytes) and swappable, so you can serve many tasks from one base model. "Freeze the base, train a small adapter" is a pattern worth recognizing beyond fine-tuning; it is the general move of perturbing a frozen pretrained artifact cheaply, and it reappears in prompt-tuning and control.

For serving, quantization with GPTQ, AWQ, or bitsandbytes shrinks the model further; see [Quantization](../../machine-learning/04-llm-systems/quantization/) in *Machine Learning & AI*. The decision that ties this together is build versus buy: a fine-tuned SLM trades a fixed training cost for a much lower per-token serving cost and no vendor dependency. That calculation is the core of platform economics.

## The platform loop

Each part of this chapter is one stage of a lifecycle, and the platform owns the whole loop. Data is versioned; training runs distributed and checkpointed; evaluation measures held-out data and task benchmarks; a **model registry** stores versioned, signed weights; serving (vLLM) puts the model in front of traffic; **monitoring** watches drift, quality, and cost; and what monitoring finds goes back into the data.

Experiment tracking with Weights & Biases or MLflow records runs, metrics, and artifacts. Orchestration is the hard operational part: training and data pipelines are multi-step, long-running, and failure-prone, which is exactly what [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/) is for. Reproducibility is the discipline that makes the loop trustworthy: pin the data, the seed, the framework, and the CUDA version, because a model you cannot rebuild is a liability. And evaluation is part of the loop, not its end; a model you cannot measure is a model you cannot improve, which is why [LLM Evaluation](../09-llm-evaluation/) gets its own chapter.

## Choosing the right tool

With the whole stack in view, most decisions reduce to a lookup from the situation to the tool. The table collects the choices made throughout the chapter.

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

These patterns show up across the industry at very different levels of difficulty. Anthropic and OpenAI run large-scale distributed pretraining with checkpointing and 3D parallelism; Google and DeepMind train with JAX on TPU pods using `pjit` sharding and Pathways. Meta builds PyTorch and FSDP and uses them for LLaMA training and quantization. NVIDIA builds Megatron-LM, NeMo, TensorRT, and NCCL. Databricks and Mosaic sell distributed training as a product; HuggingFace maintains the open stack of Transformers, Accelerate, PEFT, and TEI; and Cohere and Voyage sell embedding models, both creating and hosting them.

## Where this leads

The models trained here are the ones the rest of the platform serves. The transformers, backpropagation, and optimizers being trained come from [Deep Learning](../../machine-learning/02-deep-learning/), and deploying the output (serving, the KV cache, parallelism, quantization) is the subject of [LLM Systems & Inference](../../machine-learning/04-llm-systems/). The training and serving substrate of Kubernetes, GPU scheduling, and autoscaling is in [Cloud Native](../../systems/03-cloud-native/). Within this book, [Distributed Data and Caching](../04-distributed-data-and-caching/) covers the vector stores and data where embeddings live, [Durable Orchestration and Workers](../05-durable-orchestration-and-workers/) makes training pipelines crash-proof, [Retrieval and RAG](../07-retrieval-and-rag/) puts the embeddings to work, [Streaming and SSE](../03-streaming-sse/) delivers generated output to clients, and [Edge, Realtime, and On-Device Inference](../10-edge-realtime-inference/) runs models off the GPU fleet with llama.cpp, GGUF, and efficiency architectures.

### Further reading

- [PyTorch docs](https://pytorch.org/docs/stable/index.html), [JAX docs](https://jax.readthedocs.io/), and [vLLM docs](https://docs.vllm.ai/) (free): the primary references.
- [HuggingFace Transformers](https://huggingface.co/docs/transformers), [Accelerate](https://huggingface.co/docs/accelerate), and [PEFT](https://huggingface.co/docs/peft) docs (free).
- [DeepSpeed](https://www.deepspeed.ai/) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (free).
- [Ray Train and Ray Serve](https://docs.ray.io/) for distributed training and serving (free).
- [NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/) and [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/) (free).
- [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html).
- [Sentence-Transformers](https://www.sbert.net/) (free).
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (free), a practical guide to training at scale.

## Key terms

- **Eager execution**: running each tensor operation immediately as Python reaches it, with the autograd tape recorded on the fly.
- **`torch.compile`**: PyTorch's tracer and compiler (TorchDynamo plus Inductor) that turns eager code into fused kernels.
- **Transformation (JAX)**: a function such as `jit`, `grad`, `vmap`, or `pmap` that takes a pure function and returns a compiled, differentiated, batched, or parallel version of it.
- **Data, tensor, pipeline, and expert parallelism**: splitting training across devices by the batch, by each weight matrix, by layers, or by MoE experts.
- **ZeRO / FSDP**: sharding parameters, gradients, and optimizer state across data-parallel ranks, gathering each layer only when it is needed.
- **Contrastive training**: fine-tuning an embedder by pulling matching pairs together and pushing negatives apart.
- **Hard negative**: a near-miss example that looks relevant but is not, the most informative kind of negative.
- **LoRA / QLoRA**: training small low-rank adapters on a frozen base model; QLoRA does so on a 4-bit-quantized base.
- **Model registry**: the store of versioned, signed model weights between evaluation and serving.

## Summary

- PyTorch is eager by default with an opt-in compiler and optimizes for debuggability and ecosystem; JAX is functional and compiler-first and optimizes for composable `jit`, `vmap`, and `pmap` and for TPUs. Both converge on XLA-style compilation.
- Distributed training is a choice of what to split: the batch, the matrix, the layers, or the experts. FSDP and ZeRO shard the optimizer state, the dominant memory cost, to fit big models on small GPUs.
- Embedding models are their own product: contrastive fine-tuning creates them, batched encoder inference hosts them, and a vector store indexes their output.
- A fine-tuned small model with LoRA and quantization often beats a frontier API on a narrow task at a fraction of the cost and latency; pick SFT, DPO, or RFT by the signal you have.
- The platform owns the loop from data through training, evaluation, registry, serving, and monitoring back to data, and evaluation is part of the loop.

## Review questions

1. Why is PyTorch easier to debug than JAX, and what does `torch.compile` give back?
   <details><summary>Answer</summary>PyTorch runs operations eagerly as plain Python, so you can step through and inspect tensors; JAX stages computations into traces, which makes printing inside `jit` tricky. `torch.compile` traces the eager program and compiles fused kernels, recovering much of a static graph's speed while keeping eager development.</details>

2. A model's weights fit on your GPU but training runs out of memory. What is consuming the memory, and which technique addresses it?
   <details><summary>Answer</summary>The optimizer state (Adam's two moments per parameter), plus gradients and activations, which together often run four to eight times the weights in fp32. ZeRO or FSDP shards that state across data-parallel ranks; gradient checkpointing and mixed precision reduce the activation and precision costs further.</details>

3. Why is tensor parallelism usually kept inside one node, while pipeline parallelism can span nodes?
   <details><summary>Answer</summary>Tensor parallelism splits each weight matrix and needs an all-reduce per layer, which is heavy communication that wants NVLink bandwidth. Pipeline parallelism passes only activations between stages, which tolerates slower links at the cost of pipeline bubbles.</details>

4. Why does serving an embedding model look so different from serving a generative LLM?
   <details><summary>Answer</summary>An embedder is encoder-only and bidirectional: one forward pass per input, no KV cache, no autoregressive decode. It is a throughput problem of big batches over short sequences, often fine on CPU or a small GPU, whereas generation is a latency problem dominated by the decode loop.</details>

5. You have a task with a programmatic checker but no gold answers. Which fine-tuning method fits, and what should you watch for?
   <details><summary>Answer</summary>RFT, using the checker (your eval) as the reward. Watch for reward hacking, where the model finds ways to score well without actually solving the task.</details>

6. Why pin the embedding model version?
   <details><summary>Answer</summary>Vectors from different model versions are not comparable, so a model change forces re-embedding the whole corpus, which is expensive. Pinning the version keeps the index consistent until you deliberately migrate.</details>

## Exercises

These projects make the chapter's tradeoffs concrete. Each is open-ended; the point is to observe the numbers yourself.

- Train the same small model three ways: a plain PyTorch loop, `torch.compile`, and JAX `jit`. Compare the speedups and the debugging experience.
- Take a model that does not fit on one GPU and make it fit with FSDP, then with DeepSpeed ZeRO-3, reading the GPU memory numbers as you go.
- Fine-tune an embedding model on a domain dataset, serve it, and measure recall@k against the base model on your own retrieval set.
- Distill or LoRA-tune a small model for one task and benchmark it (quality, dollars per million tokens, p99 latency) against calling a large model.
