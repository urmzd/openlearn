# Model Loading: Hub to GPU

A model you can download is not yet a model you can serve. Between the files sitting in a Hugging Face Hub repository and a process answering requests on a GPU lies a chain of steps: resolving a revision, reading a byte format, matching thousands of tensor names to module parameters, slicing each tensor for the right GPU, and wiring up a tokenizer and chat template that reproduce what the model saw in training. When any link in that chain is wrong, the symptom is rarely a clean error. More often the model is slow to start, or it loads and then produces garbage, or it produces fluent text that is quietly worse than it should be.

This section takes that chain apart. Every "my custom fine-tune won't load, is slow to start, or gives garbage" ticket at an inference provider reduces to one of four layers: **the repo** (files and metadata), **the bytes** (format, dtype, layout), **the loader** (key mapping, sharding, quantization), or **the tokenizer** (template and special tokens). We work through each layer as math first and then as the code that checks it, ending with a triage playbook you can use on call.

You will get the most from this section if you have read the chapter opening, [LLM Systems & Inference](../README.md), especially its discussion of quantization and multi-GPU parallelism, and if transformer blocks and attention from [Deep Learning](../../02-deep-learning/README.md) are familiar. The quantized formats that appear in checkpoint headers are derived in [Quantization: Math to Code](../01-quantization/README.md); the runtimes that do the loading are mapped in [Inference Frameworks](../02-frameworks/README.md) and [LLM Serving Platforms](../05-serving-platforms.md). The runnable companions live in [`code/`](code/README.md).

> **Note:** This layer of the stack moves monthly. The details here were reviewed in October 2026 against transformers 5.19, huggingface_hub 2.1, safetensors 0.8, vLLM 0.31, and hf_xet 1.7. Pin versions in anything you ship.

> **Learning objectives**
> - Name the files in a Hub model repository and the fields in each that decide whether a model loads and behaves correctly.
> - Read a safetensors header by hand, locally or over HTTP, and verify dtypes, shapes, and sizes without loading any weights.
> - Explain what `from_pretrained` and serving-engine loaders do step by step, including key renaming, fused projections, and tensor-parallel slicing.
> - Estimate cold-start time from checkpoint size and the bandwidth of each link, and choose the technique that attacks the dominant term.
> - Verify a chat template by diffing token ids, and diagnose the common load failures from their symptoms.

## In this chapter

1. [Repository and tensor contract](01-repository-and-tensor-contract.md): Match model configuration and tensor metadata before allocating weights.
2. [Downloads, loaders, and sharding](02-downloads-loaders-and-sharding.md): Trace checkpoint bytes through caches, loaders, and device shards.
3. [Cold starts and tokenizer equivalence](03-cold-start-and-tokenizers.md): Estimate loading time and check the exact tokenization contract.
4. [Portability and load triage](04-portability-and-load-triage.md): Separate artifact, loader, execution, and tokenizer failures.
