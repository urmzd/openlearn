# tinyllm: Build an LLM from Scratch

## Overview

- **Primary reference**: the [course](../../paths/course/) itself: every chapter here is one module you build and `ol check` grades. Read [the system map](../../paths/course/SYSTEM.md) first.
- **Supplementary**: Karpathy, [*Neural Networks: Zero to Hero*](https://karpathy.ai/zero-to-hero.html) (free); Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (free), chapters 3 (n-gram LMs) and 9 (transformers); Raschka, *Build a Large Language Model (From Scratch)*
- **Prerequisites**: the Pass 0 primers ([Python and numpy](../../software-craftsmanship/12-language-and-tool-primers/01-python-and-numpy.md), [C](../../software-craftsmanship/12-language-and-tool-primers/03-c.md)) and, just in time, the math each part names (course DESIGN 7.5)
- **Estimated time**: the spine of course passes 1 to 11, about 60 weeks part-time; Pass 1 (the tracer) is about 5 weeks

## Key Takeaways

- An LLM system is a stack of contracts: a checkpoint format between training and serving, a C ABI between Python and Rust and the kernels, an HTTP API between the engine and everything in front of it. Build each side of each contract yourself and nothing in the stack is magic.
- Start with the thinnest model that exercises every contract: a **byte-level bigram** fitted by counting, served by a real engine. Every later model (autograd bigram, RNN, transformer, the Llama family) upgrades a component behind an unchanged contract.
- Numbers are only right if they are tested: every module ships course tests, a reference that passes them, and a stub that fails them.

## How to Study

Follow the course path (`practice/bin/ol learn course`), not this directory's order: parts interleave with math, systems, and operations modules. For each chapter run `ol start <ID>`, read beats 1 to 3 before writing code, implement against the interface in beat 4, and `ol check <ID>` until green. `ol tests <ID>` explains what each test checks and why.

---

# Concepts & Techniques

## Core Insight

A language model is a function from a prefix of token ids to a distribution over the next id. Everything else in the stack is plumbing that makes that function cheap to train, cheap to run, and safe to expose, and each piece of plumbing has a contract you can test in isolation.

## 1. The tracer (Pass 1)

**Key ideas**:
- **Byte tokenizer**: 256 ids, no training, no unknown tokens; decoding a stream must handle incomplete UTF-8.
- **Count bigram**: $P(b \mid a) = (c_{ab} + \alpha) / (\sum_x c_{ax} + 256\alpha)$; its negative log-likelihood is the baseline every later model must beat.
- **Logits in Python**: the bigram model computes logits in the reference implementation.
- **Checkpoint**: Python writes `model.safetensors`; Rust Candle reads the same weights. Shared fixture files check cross-language behavior.

## 2. The spine (Passes 2 to 11)

**Key ideas**:
- **Autograd and training** (p00), **tokenizers** (p01), **statistical LMs** (p02), **recurrent networks** (p03), **attention** (p04), **the 2017 transformer** (p05), **objectives** (p06), **the modern decoder block** (p07), **inference** (p08), **C kernels** (p09), **serving in Rust** (p10), **training at scale** (p11), and an optional **post-training** part (p12).

## Parts

| Part | Directory | Chapters so far | Course pass |
|---|---|---|---|
| p00 | [Foundations](p00-foundations/) | L0.0 byte bigram | 1 (tracer), 2 |
| p01 | [Tokenizers](p01-tokenizers/) | none yet (B4) | 3 |
| p02 | [Statistical LMs](p02-statistical-lm/) | none yet (B5) | 3 |
| p03 | [Recurrent Networks](p03-recurrent/) | none yet (B6) | 4 |
| p04 | [Attention Origins](p04-attention-origins/) | none yet (B6) | 4 |
| p05 | [The Transformer (2017)](p05-transformer-2017/) | none yet (B7) | 5 |
| p06 | [Objectives and Adaptation](p06-objectives/) | none yet (B7) | 5 |
| p07 | [The Modern Decoder Block](p07-modern-block/) | none yet (B7) | 5 |
| p08 | [Inference](p08-inference/) | none yet (B8) | 6 |
| p09 | [Kernels in C](p09-kernels/) | rt.01 the C ABI | 1 (tracer), 6 |
| p10 | [Serving](p10-serving/) | L10.0 your first endpoint | 1 (tracer), 7 |
| p11 | [Training at Scale](p11-training-at-scale/) | none yet (B11) | 9 |
| p12 | [Post-Training (optional)](p12-post-training/) | none yet (B12) | 10 |
| p13 | [Vision foundations](p13-vision/) | L13.1 through L13.6 | 12 |
| p14 | [Speech](p14-speech/) | L14.1 through L14.3 and L14.5 | 12 |
| C1, C2 | [Capstones](capstones/) | none yet (B11, B12) | 9, 10 |

Each part README lists its planned modules; chapters arrive with their authoring batches, and the [course path](../../paths/course/) lists what is available.

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Linear Algebra](../../math/03-linear-algebra/) | M03.1, the naive matmul in C every logit goes through |
| [Gateway](../../ai-platform-engineering/12-gateway/) | gw.00, the front door to the engine |
| [Containers and Kubernetes](../../infrastructure/01-containers-kubernetes/) | dep.00, the images and charts the engine runs in |
| [LLM Systems & Inference](../04-llm-systems/) | the production systems this spine rebuilds in miniature |

## Company Relevance

| Company | Practice |
|---|---|
| Hugging Face | safetensors, `transformers` checkpoints, `tokenizers` |
| llama.cpp / ggml | C kernels behind a stable ABI, served over an OpenAI-compatible HTTP API |
| vLLM, SGLang | Python model code over custom kernels, a separate serving engine |
