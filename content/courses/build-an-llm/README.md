# Build Your Own LLM System

This book builds one language-model system end to end: a byte-level model trained in Python on its own autograd, a Rust engine that streams completions over HTTP, a Go gateway in front of it, optional C kernels, and the deployment that keeps it running on Kubernetes. It starts with a counted byte bigram and upgrades one component at a time while the system keeps running. It is written for engineers who want to understand an LLM system from the matrix multiply to the pager, through worked solutions you read until you can explain every decision.

## What you will be able to do

- Derive and implement autograd, gradient checks, and a training loop, and retrain a model with them.
- Build tokenizers (byte-level BPE, WordPiece, Unigram) that match Hugging Face output exactly.
- Implement attention, the 2017 transformer, and a modern Llama-family decoder that loads real weights.
- Make inference fast and exact with KV caching, quantization, and speculative and constrained decoding, then serve it from an OpenAI-compatible Rust engine with continuous batching.
- Train and post-train a small model with mixed precision, SFT, DPO, and GRPO, and release it through a gated workflow.

## Ways through this book

Building the whole system, setup included, takes about 67 weeks at 10 to 12 hours a week.

- **The whole system, built in passes** (about 67 weeks): read [the system the solutions build](paths/course/SYSTEM.md), then follow [the course path](paths/course/README.md) in Appendix B from [Pass 0: Setup](paths/course-p00-setup/README.md) and [Pass 1: The tracer](paths/course-p01-tracer/README.md) onward. Each pass ends with the system running end to end. This is the recommended route.
- **The model, from bigram to Llama** (about 25 weeks): [Chapter 1: Foundations](01-foundations/README.md), [Chapter 2: Tokenizers](02-tokenizers/README.md), then Chapters 3 to 7 in order, ending with [Chapter 8: The Modern Decoder Block](08-modern-block/README.md) and [8.9 a Llama-family model](08-modern-block/09-llama-family-model.md).
- **Inference and serving** (about 15 weeks): [1.1 the byte bigram](01-foundations/01-byte-bigram.md), [Chapter 9: Inference](09-inference/README.md), then [Chapter 11: Serving](11-serving/README.md) from [11.1 your first endpoint](11-serving/01-your-first-endpoint.md). Add [Chapter 10: Kernels in C](10-kernels/README.md) when you want to see the kernels underneath; nothing later depends on it.
- **Training and post-training** (about 8 weeks, after the model route): [Chapter 12: Training at Scale](12-training-at-scale/README.md), [Chapter 13: Post-training](13-post-training/README.md), then the capstones in Appendix A: [A.1 TinyStories, owned end to end](capstones/01-tinystories.md) and the optional [A.2 post-trained model](capstones/02-post-trained.md).

## Before you start

You need working Python with NumPy and enough C to read it; the primers for [Python and NumPy](../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md), [C](../programming-languages/02-language-and-tool-primers/03-c.md), [Rust](../programming-languages/02-language-and-tool-primers/04-rust.md), and [HTTP and SSE](../programming-languages/02-language-and-tool-primers/05-http-and-sse.md) in *Programming Languages* cover what the book uses. The mathematics is taught just in time from [*Mathematics for Computing*](../mathematics-for-computing/README.md), starting with [vectors, matrices, and matmul](../mathematics-for-computing/04-linear-algebra/01-vectors-matrices-and-matmul.md).

Related books own parts of the system: *AI Platform Engineering* owns the [gateway](../ai-platform-engineering/12-gateway/01-streaming-proxy.md) and durable workflows, *Infrastructure* owns the [containers and Kubernetes deployment](../infrastructure/01-containers-kubernetes/01-tracer-deploy.md), *Data Engineering* owns the corpus pipeline, *Algorithms* owns the [systems data structures](../algorithms/13-systems-data-structures/README.md), and [LLM systems and inference](../machine-learning/04-llm-systems/README.md) in *Machine Learning and AI* surveys the production versions.

New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
