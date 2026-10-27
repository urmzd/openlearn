# Embeddings, small models, and platforms

This part of [Training and Frameworks](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Connect specialized training to its evaluation and deployment loop.

## Creating and hosting embedding models

Not every model on the platform generates text. Embedding models are the retrieval substrate of RAG and search, and they are their own product: creating one, hosting one, and indexing its output are three distinct platform jobs.

### Creating an embedding model

An embedding model maps text (or images or audio) to a dense vector so that semantic similarity becomes cosine similarity. You rarely train one from scratch. Instead you start from a base encoder (the BERT family, E5, GTE, BGE, Nomic) and fine-tune it with **contrastive training**: pull (query, positive) pairs together and push negatives apart, using a loss such as InfoNCE (multiple-negatives ranking) or a triplet loss. The negatives matter more than anything else. **Hard negatives**, mined near-misses that look relevant but are not, teach the model far more than easy ones.

[Sentence-Transformers](https://www.sbert.net/) is the usual fine-tuning tool. **Matryoshka** representation learning lets one model emit vectors that remain useful when truncated, so you can store shorter vectors cheaply. The [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard is the standard evaluation. The reason to bother is that retrieval quality caps RAG quality: a domain-tuned embedder lifts retrieval recall far more than a bigger LLM does.

### Hosting it

Embedding inference is a different workload from generation. The model is **encoder-only and bidirectional**: there is no KV cache and no autoregressive decode. That makes it a throughput problem of big batches over short sequences, and it often runs fine on a CPU or a small GPU. You can serve it with TEI ([Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference)), vLLM's embeddings endpoint, or a plain ONNX or Triton service. Normalize the vectors, and pin the model version: changing the model means re-embedding the whole corpus, which is expensive.

The vectors then go into a **vector store** with an approximate-nearest-neighbor index (HNSW or IVF): pgvector in Postgres, Qdrant, Milvus, Weaviate, or FAISS. That store is sharded, cached, durable data, which is the subject of [Distributed Data and Caching](../../systems/06-distributed-data-and-caching/), and it is what [Retrieval and RAG](../27-retrieval-and-rag/) queries.

## Small language models

The same economic logic that favors a fine-tuned embedder favors small generative models. Small language models, roughly 1 to 8 billion parameters, are the production workhorse. They are cheap to fine-tune, fast to serve, fit on one GPU (or on a CPU or edge device when quantized), and are good enough, or better, on narrow tasks. When you do not need a frontier model, an SLM is the platform default. A fine-tuned 3 to 8B model with LoRA and quantization often beats a frontier API on a narrow task at a fraction of the price and latency.

There are three ways to make a small model good at your task. **Distillation** trains the small model to match a large teacher's outputs, either its logits or data it generates, transferring capability into a cheaper package. **Fine-tuning** adapts the model directly, and the method you choose depends on the signal you have. SFT needs input and correct-output pairs. DPO needs chosen-versus-rejected pairs. RFT (reinforcement fine-tuning) needs a scorer that returns a reward, so it fits verifiable tasks where you have no gold answers; watch it for reward hacking. The full ladder, with worked examples, is in [Model Routing and Cascades](../24-model-routing/), under tuning the router.

The third technique makes fine-tuning cheap enough to do routinely. **PEFT methods such as LoRA and QLoRA** freeze the base model and train small low-rank adapter matrices. QLoRA fine-tunes on top of a 4-bit-quantized base, so a 7B model tunes on a single consumer GPU. Adapters are tiny (megabytes) and swappable, so you can serve many tasks from one base model. "Freeze the base, train a small adapter" is a pattern worth recognizing beyond fine-tuning; it is the general move of perturbing a frozen pretrained artifact cheaply, and it reappears in prompt-tuning and control.

For serving, quantization with GPTQ, AWQ, or bitsandbytes shrinks the model further; see [Quantization](../20-inference-engines/01-quantization/) in *AI Engineering*. The decision that ties this together is build versus buy: a fine-tuned SLM trades a fixed training cost for a much lower per-token serving cost and no vendor dependency. That calculation is the core of platform economics.

## The platform loop

Each part of this chapter is one stage of a lifecycle, and the platform owns the whole loop. Data is versioned; training runs distributed and checkpointed; evaluation measures held-out data and task benchmarks; a **model registry** stores versioned, signed weights; serving (vLLM) puts the model in front of traffic; **monitoring** watches drift, quality, and cost; and what monitoring finds goes back into the data.

Experiment tracking with Weights & Biases or MLflow records runs, metrics, and artifacts. Orchestration is the hard operational part: training and data pipelines are multi-step, long-running, and failure-prone, which is exactly what [Durable Orchestration and Workers](../../systems/07-durable-orchestration/) is for. Reproducibility is the discipline that makes the loop trustworthy: pin the data, the seed, the framework, and the CUDA version, because a model you cannot rebuild is a liability. And evaluation is part of the loop, not its end; a model you cannot measure is a model you cannot improve, which is why [LLM Evaluation](../29-evaluation/) gets its own chapter.

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

The models trained here are the ones the rest of the platform serves. The transformers, backpropagation, and optimizers being trained come from [Deep Learning](../02-deep-learning/), and deploying the output (serving, the KV cache, parallelism, quantization) is the subject of [LLM Systems & Inference](../20-inference-engines/). The training and serving substrate of Kubernetes, GPU scheduling, and autoscaling is in [Cloud Native](../../systems/09-cloud-native/). Within this book, [Distributed Data and Caching](../../systems/06-distributed-data-and-caching/) covers the vector stores and data where embeddings live, [Durable Orchestration and Workers](../../systems/07-durable-orchestration/) makes training pipelines crash-proof, [Retrieval and RAG](../27-retrieval-and-rag/) puts the embeddings to work, [Streaming and SSE](../../systems/04-streaming/) delivers generated output to clients, and [Edge, Realtime, and On-Device Inference](../26-edge-inference/) runs models off the GPU fleet with llama.cpp, GGUF, and efficiency architectures.

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

[Chapter contents](README.md)
