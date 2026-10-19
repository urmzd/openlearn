# Build an LLM

This book builds one language-model system end to end, in several languages: a byte-level language model trained in Python on its own numerics, a Rust inference engine that streams completions over HTTP, a Go gateway in front of it, optional standalone kernels in C, and the containers, charts, traces, and runbooks that keep it running on Kubernetes. Every layer is built from first principles, and every layer's solution is here to read. The book is a manual of worked solutions, not a set of exercises: code is cheap to generate now, so its job is to make you understand the optimal design well enough to explain every decision in it.

It is written for engineers who want to understand an LLM system from the matrix multiply to the pager, rather than only call one. By the end you will be able to derive and implement autograd, tokenizers, statistical and recurrent language models, attention and the transformer, a modern Llama-family decoder, fast exact inference with KV caching, quantization, and speculative and constrained decoding, an OpenAI-compatible serving engine with continuous batching, training at scale with mixed precision, and post-training with SFT, DPO, and GRPO. You will also be able to say why each one is built the way it is.

## The idea that organizes the book

A language model is a function from a prefix of token ids to a distribution over the next id. Everything else in the stack is plumbing that makes that function cheap to train, cheap to run, and safe to expose, and each piece of plumbing has a contract you can test in isolation: a checkpoint format between training and serving, file formats between Python and Rust, an HTTP API between the engine and everything in front of it. Understand each side of each contract and nothing in the stack is magic. The [concepts overview](concepts.md) develops this idea in more detail and maps it onto the chapters.

The book therefore starts with the thinnest model that exercises every contract, a byte-level bigram fitted by counting and served by a real engine, and then upgrades one component at a time behind unchanged contracts: an autograd bigram, an RNN, a transformer, the Llama family. The system grows in a spiral, and after each turn it runs end to end. The map of that growth, with the components, their languages, and the contracts between them, is [the system the solutions build](paths/course/SYSTEM.md). Read it before anything else.

## How the book is organized

The chapters are numbered as parts of the model stack, from foundations to serving, then extensions to vision and speech, then the capstones. The best reading order is not strictly numerical, because the system is built in passes that interleave parts of this book with mathematics, systems, and operations chapters from other books. The [course path](paths/course/README.md) gives that interleaved order, and the [study plan](study-plan.md) paces it.

[Foundations](p00-foundations/README.md) is the model side of the first, thinnest system and the autograd engine every later model trains with. It begins with a [byte bigram](p00-foundations/00-byte-bigram.md) fitted by counting and written as a safetensors checkpoint that the Rust engine serves, then builds tensors with broadcasting backward passes, an op library with gradient checking, fused losses, a module system, a training loop that retrains the bigram, and atomic checkpoints with token streams.

[Tokenizers](p01-tokenizers/README.md) moves from bytes to subwords: the tokenizer protocol, byte-level BPE compatible with GPT-2, WordPiece, the Unigram LM with EM and Viterbi, an independent fast BPE in Rust with a streaming UTF-8 decoder, and the metrics that choose a vocabulary size.

[Statistical language models](p02-statistical-lm/README.md) covers the models before deep learning and the first neural one: an n-gram model with interpolated modified Kneser-Ney smoothing, Bengio's neural probabilistic LM, and word2vec with a PPMI-SVD baseline.

[Recurrent networks](p03-recurrent/README.md) builds sequence models with state: a vanilla RNN with backpropagation through time written by hand, the LSTM and GRU in torch's gate order, bidirectional RNNs, ELMo's scalar mix, and an RNN language model trained with stateful truncated BPTT.

[Attention origins](p04-attention-origins/README.md) shows where attention came from: an encoder-decoder trained with teacher forcing, Bahdanau and Luong attention, a generic beam search, and the metrics that grade generated sequences.

[The transformer (2017)](p05-transformer-2017/README.md) assembles attention from its parts: scaled dot-product attention with its backward pass, masks, multi-head attention, positional encodings, and the full encoder-decoder transformer.

[Objectives and adaptation](p06-objectives/README.md) applies one architecture to many objectives: GPT's causal LM, BERT's masked LM, ELECTRA, T5's span corruption, fine-tuning heads, LoRA with PiSSA, and the evaluation harness behind the model-zoo table every architecture reports to.

[The modern decoder block](p07-modern-block/README.md) covers what changed between 2017 and a Llama-family model: pre-LN with RMSNorm, gated MLPs, RoPE and context extension, grouped-query and multi-head latent attention, sliding windows and sinks, mixture of experts, and a Llama-family model that loads Hugging Face weights.

[Inference](p08-inference/README.md) generates text fast and exactly: sampling and logit processors, the KV cache with incremental decode, paged and radix prefix caches, quantization, speculative decoding, and constrained decoding.

[Optional standalone C kernels](p09-kernels/README.md) collects independent C exercises: an arena allocator, a thread pool, a tiled matmul, softmax, flash and paged attention, fused quantized matmul, and elementwise kernels. Python and Rust implementations stay independent of them.

[Serving](p10-serving/README.md) builds the Rust engine, from [your first endpoint](p10-serving/00-your-first-endpoint.md), a dependency-free HTTP/1.1 and SSE server, to a tokio and hyper engine with continuous batching, chunked prefill, a prefix-caching block manager, disaggregated prefill and decode, metrics and tracing, speculative decoding, tool calls, a load generator, and a regression gate.

[Training at scale](p11-training-at-scale/README.md) covers what it takes to train the capstone on a laptop and what changes on a cluster: mixed precision, loss scaling, gradient accumulation, and activation checkpointing, then collectives, DDP, and ZeRO.

[Post-training](p12-post-training/README.md) adapts a small causal model with supervised fine-tuning, direct preference optimization, GRPO with verifiable rewards, and distillation.

[Vision foundations](p13-vision/README.md) and [Speech recognition](p14-speech/README.md) extend the system to images and audio: convolution, 2D RoPE, a vision transformer, dynamic resolution, contrastive image-text training, and image preprocessing; then a log-mel audio frontend, Whisper's encoder and decoder, long-form decoding, and multimodal evaluation metrics.

[Capstones](capstones/README.md) run the whole system on one goal: training, ablating, evaluating, releasing, and serving a TinyStories model, then optionally post-training it into an instruction follower.

## Prerequisites and related books

You need working Python with NumPy and enough C to read it; the primers for [Python and NumPy](../programming-languages/02-language-and-tool-primers/01-python-and-numpy.md), [C](../programming-languages/02-language-and-tool-primers/03-c.md), [Rust](../programming-languages/02-language-and-tool-primers/04-rust.md), and [HTTP and SSE](../programming-languages/02-language-and-tool-primers/05-http-and-sse.md) in *Programming Languages* cover what the book uses. The mathematics is taught just in time from [*Mathematics for Computing*](../mathematics-for-computing/README.md), starting with [vectors, matrices, and matmul](../mathematics-for-computing/03-linear-algebra/01-vectors-matrices-and-matmul.md); each chapter names the sections it needs.

Several books own parts of the system this one builds on. *AI Platform Engineering* owns the [gateway](../ai-platform-engineering/12-gateway/00-streaming-proxy.md) and the durable workflows. *Infrastructure* owns the [containers and Kubernetes deployment](../infrastructure/01-containers-kubernetes/00-tracer-deploy.md). *Data Engineering* owns the corpus pipeline. *Algorithms* owns the [systems data structures](../algorithms/16-systems-data-structures/README.md) the engine relies on. [LLM systems and inference](../machine-learning/04-llm-systems/README.md) in *Machine Learning and AI* surveys the production systems this book rebuilds in miniature.

## Conventions

Each section teaches one idea from first principles: why it is needed, the principles and math, a worked example by hand, the interface, and then a step-by-step walkthrough of the reference solution with annotated code listings, why it beats the alternatives, what can go wrong, and review questions whose answers are folded under a disclosure triangle. Math appears inline as $P(b \mid a)$ or in display blocks, and code listings are tagged with their language.

The full solution code lives in [`course/solutions/`](../../../course/solutions): `python/`, `rust/`, `go/`, `c/`, the entry points and deploy files in `entry/`, and the reference documents in `docs/`. The interfaces and formats every solution implements live in [`course/contracts/`](../../../course/contracts). Sections link the solution files they explain and the contracts they implement. Numbers are only right if they are checked, so every section lists the cases a correct solution must handle and shows how the solution handles each.
