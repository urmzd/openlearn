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

This chapter assumes you know how neural networks are trained: backpropagation, optimizers such as Adam, and the transformer architecture, all covered in [Deep Learning](../02-deep-learning/) in *AI Engineering*. It also leans on [LLM Systems & Inference](../20-inference-engines/) from the same book for serving, the KV cache, and the basics of parallelism. You do not need to have trained a large model yourself; the point here is to understand the choices well enough to make them on a platform.

## In this chapter

1. [PyTorch and JAX execution](01-pytorch-and-jax.md): Relate eager, compiled, and transformed execution to the training loop.
2. [Distributed and serving boundaries](02-distributed-and-serving-boundaries.md): Explain which responsibilities belong to the learner and the serving runtime.
3. [Embeddings, small models, and platforms](03-embeddings-small-models-and-platforms.md): Connect specialized training to its evaluation and deployment loop.
