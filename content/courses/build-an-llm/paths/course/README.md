# Build Your Own LLM System

This path is the whole course in one reading order: a single LLM system, called `forge` in the reference solutions, built end to end in Python, Rust, Go, and C. It does not follow the chapter order of *Build an LLM*. Instead it interleaves sections from that book with the mathematics, systems, infrastructure, and craft sections from the other books in the library, in the order the system needs them. A section is read when the component it explains is about to be built, so every idea arrives with a reason to care about it.

The system grows in passes. Pass 1 builds the thinnest version of every layer and runs it end to end: a byte bigram fitted by counting, a Rust engine that streams completions from its checkpoint, a Go gateway, a deployment on a local Kubernetes cluster, and one trace across the services. Every later pass deepens one or more layers behind contracts that do not change, so after each pass the system still runs end to end, and a regression anywhere shows up as a broken path you already understood.

## Before you begin

Read the opening of [*Build an LLM*](../../README.md) first; it explains the idea that organizes the book and how its chapters fit together. Then read [The system the solutions build](SYSTEM.md), the map of every component, its language, its solution, and the contracts it meets at. You should be able to name each component and the contract it serves before starting Pass 0.

## The passes

Each pass has an opening page that lists its sections in reading order, with what you should be able to explain after each, and ends with a page describing what the system does once the pass is complete.

1. [Pass 0: Setup](../course-p00-setup/README.md). The toolchain, the Python and numpy, shell, git, and make that every later build uses, and a CI gate that lints every commit message.
2. [Pass 1: The tracer](../course-p01-tracer/README.md). A thin system through every layer: matmul, the byte bigram, the Rust engine, the Go gateway, containers on kind, one trace, a first incident drill, and a first architecture decision record.
3. [Pass 2: Foundations](../course-p02-foundations/README.md). The precalculus, calculus, linear algebra, probability, numerics, and optimization the autograd rests on, then the autograd itself, which retrains the tracer's bigram by gradient descent.
4. [Pass 3: Tokens and data](../course-p03-tokens-and-data/README.md). BPE, WordPiece, and Unigram tokenizers that match Hugging Face exactly, a deterministic and auditable corpus pipeline, and the n-gram and neural language models.
5. [Pass 4: Sequence models](../course-p04-sequence-models/README.md). Recurrent language models trained with truncated BPTT, encoder-decoders, attention, and beam search.
6. [Pass 5: The transformer](../course-p05-transformer/README.md). The 2017 transformer, GPT, BERT, ELECTRA, LoRA, the model zoo, and the modern Llama-family block that loads SmolLM2 and matches Hugging Face.
7. [Pass 6: Inference and kernels](../course-p06-inference-and-kernels/README.md). Sampling, the KV cache, quantization, speculative and constrained decoding in Python, with optional C kernels.
8. [Pass 7: The serving platform](../course-p07-serving-platform/README.md). The Rust engine with continuous batching and disaggregation, the production gateway, and the platform on kind with its observability stack and SLOs.
9. [Pass 8: Durable execution](../course-p08-durable/README.md). The corpus build and training run as workflows on the course's own durable engine.
10. [Pass 9: Capstone training](../course-p09-capstone-training/README.md). Mixed precision and accumulation, then a ten-million-parameter model trained, evaluated, released through a workflow, and served.
11. [Pass 10: Agents](../course-p10-agents/README.md). An agent SDK on the engine's tool calls, retrieval over the system's own docs, evaluation, and usage policy at the gateway.
12. [Pass 11: Operating the system](../course-p11-operate/README.md). The v1.0.0 release, migrations, incident drills, security review, design reviews, and a mock customer engagement.
13. [Pass 12: Multimodal](../course-p12-multimodal/README.md). Signal and image math, vision encoders, Whisper speech recognition, and multimodal evaluation.

The [study plan](../../study-plan.md) suggests how to pace the passes. The stages themselves, with the files each one reads, are in this path's manifest, [`path.tsv`](path.tsv), which includes every pass's manifest in turn.
