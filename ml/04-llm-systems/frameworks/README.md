# Inference Frameworks: The Cross-Language Landscape

Where you *run* an LLM. This maps the inference/serving ecosystem across languages -- the **C/C++ foundation** ([llama.cpp](https://github.com/ggml-org/llama.cpp)/[ggml](https://github.com/ggml-org/ggml)), the **Python serving tier** (vLLM, TGI, SGLang, TensorRT-LLM), the **Rust equivalents** (candle, mistral.rs, burn, ratchet, luminal), and the emerging **Zig** stack (ZML, llama.cpp.zig). The throughline: every framework is the same three layers -- a kernel/tensor core, a runtime, and a server -- assembled in a different language for a different deployment target.

> Parent topic: [LLM Systems & Inference](../). Quantization formats these frameworks consume are in [Quantization: Math → Code](../quantization/). Runnable examples: [`code/`](code/).

> For deployment choices and operating costs, see [LLM Serving Platforms](../serving-platforms.md): Anyscale, Ray, vLLM, SGLang, RouteLLM, and OpenRouter.

## Overview

- **Primary references** (all free): [llama.cpp](https://github.com/ggml-org/llama.cpp) + [ggml](https://github.com/ggml-org/ggml), [candle](https://github.com/huggingface/candle) docs, [mistral.rs](https://github.com/EricLBuehler/mistral.rs), [ZML](https://github.com/zml/zml), [vLLM](https://docs.vllm.ai/) / [SGLang](https://github.com/sgl-project/sglang) / [TGI](https://huggingface.co/docs/text-generation-inference) docs
- **Prerequisites**: [LLM Systems & Inference](../) (esp. §9 frameworks, §10 deployment), C/Rust/Zig basics
- **Estimated time**: 1-2 weeks at 8-10 hrs/week

## Key Takeaways

- Every inference framework is **three layers**: (1) a **kernel/tensor core** (ggml, CUTLASS, candle-core, MLIR/XLA), (2) a **runtime** (graph execution + KV cache + sampling), (3) a **server** (batching scheduler + an OpenAI-compatible HTTP API). Frameworks differ in which layers they own and which they borrow
- **GGUF is the lingua franca** of portable inference: a single quantized model file (from llama.cpp's ecosystem) runs under llama.cpp, candle, mistral.rs, and Zig bindings alike
- **Language picks the deployment target**: Python for max-throughput GPU serving, C for "runs anywhere with no runtime", Rust for a single safe static binary at the edge, Zig for minimal-dependency native compilation and effortless C interop
- The **OpenAI-compatible HTTP API** is the universal contract -- swap llama.cpp's `llama-server` for vLLM for mistral.rs for ZML's LLMD and clients don't change

## How to Study

- Run the *same* GGUF model under llama.cpp (`llama-server`), candle, and mistral.rs; compare tokens/sec, memory, and binary/footprint
- For each framework, identify its three layers and which it borrows (e.g. mistral.rs borrows candle's kernels; Ollama borrows llama.cpp's runtime)
- Build one example from [`code/`](code/) per language to compare each language's runtime and file interfaces

---

# Concepts & Techniques

## Core Insight

There is no single "inference framework" -- there is a **stack**, and each project stakes out a slice of it. Python frameworks (vLLM) maximize datacenter GPU throughput. C (llama.cpp) maximizes portability and minimizes dependencies so a model runs on a laptop. Rust (candle, mistral.rs) trades a little ecosystem maturity for memory safety and a single static binary. Zig (ZML) bets on compiling the whole model graph to a standalone native artifact via MLIR. Pick the language and you've largely picked the deployment story.

## 1. Anatomy of an Inference Stack

| Layer | Job | Examples |
|-------|-----|----------|
| **Kernels / tensor core** | matmul, attention, dequant on CPU/GPU | ggml, CUTLASS/cuBLAS, candle-core, Metal/Vulkan shaders, MLIR+XLA |
| **Runtime** | load weights, run the graph, manage KV cache, sample tokens | llama.cpp, candle-transformers, mistral.rs engine, ZML |
| **Scheduler / server** | continuous batching, request queue, OpenAI HTTP API | `llama-server`, vLLM, TGI, SGLang, mistralrs-server, LLMD |

Most "frameworks" are a *vertical slice* of these. Ollama = llama.cpp runtime + a friendly model manager. mistral.rs = candle kernels + its own runtime + an OpenAI server.

## 2. The C/C++ Foundation: llama.cpp + ggml

**The most-deployed inference engine on earth.**

- **[ggml](https://github.com/ggml-org/ggml)**: a dependency-free C tensor library with a static computation graph and a quantization-first design. Backends: CPU (AVX/NEON), Metal, CUDA, Vulkan, SYCL, ROCm
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**: the transformer runtime + `llama-server` (OpenAI-compatible). Defines **GGUF** (the quantized model container) and the **k-quant / i-quant** formats (see [Quantization §5.5](../quantization/))
- **The C API** (`llama.h`) is the integration point for *every other language* -- Python (`llama-cpp-python`), Rust (`llama-cpp-2`), Zig (`@cImport`), Go, etc. all bind to it
- **Family**: `whisper.cpp`, `stable-diffusion.cpp`, `ggml`-based projects share the same core
- **Downstream**: Ollama, LM Studio, Jan, GPT4All, KoboldCpp all wrap llama.cpp

**Why it wins for edge/local**: no Python, no CUDA required, a single small binary, runs a 7B model on a laptop CPU. **Tradeoff**: lower datacenter GPU throughput than vLLM (no PagedAttention-class scheduler historically; continuous batching is newer).

## 3. The Python Serving Tier (recap)

The datacenter-throughput layer, covered in the [parent topic §9](../). In one line each:
- **vLLM** -- PagedAttention + continuous batching, the general-purpose throughput king
- **TensorRT-LLM** -- compiled NVIDIA kernels, FP8, max performance on H100
- **TGI** -- HuggingFace's production server
- **SGLang** -- RadixAttention prefix sharing, fast structured output
- **Ray Serve / KServe** -- orchestration on top of any of the above

These dominate GPU clusters; the C/Rust/Zig tiers below dominate edge, embedded, and dependency-constrained deployments.

## 4. The Rust Tier (the "Rust equivalents")

Rust trades ecosystem breadth for **memory safety + a single static binary + no GIL**. The current stack:

| Framework | Layer(s) | Niche | Notes |
|-----------|----------|-------|-------|
| **[candle](https://github.com/huggingface/candle)** | kernels + runtime | minimalist ML, the Rust "PyTorch-lite" | CPU/CUDA/Metal/**WASM**; loads GGUF + safetensors; HuggingFace-backed |
| **[mistral.rs](https://github.com/EricLBuehler/mistral.rs)** | runtime + server | turnkey fast inference | builds on candle; **ISQ** (in-situ quant), OpenAI server, vision/tools |
| **[burn](https://burn.dev/)** | kernels + runtime | backend-agnostic train **and** infer | WGPU/CUDA/NdArray/LibTorch; own quantization API ([Quant §7](../quantization/)) |
| **[ratchet](https://github.com/huggingface/ratchet)** | kernels + runtime | web/cross-platform GPU | wgpu-based, browser-first inference |
| **[luminal](https://github.com/jafioti/luminal)** | compiler + runtime | search-compiled kernels | tiny graph IR, compiles to fast kernels |
| **[llama-cpp-2](https://github.com/utilityai/llama-cpp-rs)** | Rust interface | reuse llama.cpp from Rust | `llama_cpp` is the higher-level wrapper |

> **Deprecated**: `rustformers/llm` (and the old `llama-rs`) are unmaintained -- the ecosystem consolidated onto **candle** + **mistral.rs**. Use those.

**When to pick Rust**: edge/embedded, WASM/browser, a CLI you ship as one binary, or a service where you want Rust's safety end-to-end. See [`code/candle_generate.rs`](code/candle_generate.rs) (GGUF generation with candle) and [`code/mistralrs_serve.rs`](code/mistralrs_serve.rs) (OpenAI server + ISQ).

## 5. The Zig Tier

Zig's pitch: **trivial C interop** (`@cImport` reads C headers directly -- no binding generator) and **compile the whole thing to a small native artifact**. Two distinct approaches:

| Project | Approach | Notes |
|---------|----------|-------|
| **[ZML](https://github.com/zml/zml)** | compile model graphs via **MLIR + OpenXLA** | Zig + XLA → standalone native binaries; NVIDIA/AMD/TPU/Trainium from one codebase; LLMD OpenAI server in a ~2.4 GB image |
| **[llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)** | bindings + `build.zig` over llama.cpp | reuse the C engine from Zig via `@cImport` |
| **[llm.zig](https://github.com/camconn/llm.zig)** | from scratch, no deps | pedagogical full-stack inference in pure Zig |
| **[llama2.zig](https://github.com/donge/llama2.zig)** | port of Karpathy's `llama2.c` | the cleanest "read the whole inference loop" reference |

**The two idioms:**
1. **Wrap C** -- `@cImport({ @cInclude("llama.h"); })` and call llama.cpp directly. Zig's interop makes this *less* code than the equivalent Rust `bindgen` setup. See [`code/llama_cpp.zig`](code/llama_cpp.zig)
2. **Pure Zig kernels** -- write the hot loops yourself with `@Vector` SIMD and explicit allocators. See [`code/quant_dot.zig`](code/quant_dot.zig) (INT8 symmetric quantize + SIMD dot, the CPU twin of [`quantization/code/symmetric_quant.cu`](../quantization/code/symmetric_quant.cu))

**When to pick Zig**: you want C-level control and a tiny dependency-free binary, you're already calling C libraries (Zig is arguably the best C-interop language), or you want compile-time graph specialization (ZML).

## 6. Choosing Across Languages

| Deployment target | Language | Framework | Why |
|-------------------|----------|-----------|-----|
| GPU cluster, max throughput | Python | vLLM / TensorRT-LLM | PagedAttention, FP8, mature scheduler |
| Laptop / CPU / "just runs" | C | llama.cpp (`llama-server`) | no deps, GGUF, every backend |
| Single safe static binary, edge | Rust | mistral.rs / candle | memory-safe, OpenAI server, ISQ |
| Browser / WASM | Rust | candle / ratchet | near-native in-browser inference |
| Multi-vendor native compile | Zig | ZML | MLIR/XLA → standalone binary, AMD/TPU/Trainium |
| Reuse C engine, minimal binary | Zig | llama.cpp.zig | `@cImport`, no binding generator |
| Train + infer in one stack | Rust | burn | backend-agnostic, own quant API |

---

## Cross-Language Framework Matrix

| | C/C++ | Python | Rust | Zig |
|---|-------|--------|------|-----|
| **Flagship** | llama.cpp | vLLM | candle / mistral.rs | ZML |
| **Kernel core** | ggml, CUTLASS | Triton, CUTLASS | candle-core | MLIR/XLA, `@Vector` |
| **Quant formats** | GGUF k/i-quants | GPTQ/AWQ/FP8 | GGUF, ISQ | GGUF (via C), native |
| **Server** | `llama-server` | vLLM/TGI/SGLang | mistralrs-server | LLMD |
| **Best at** | portability | throughput | safe single binary | native compile + C interop |
| **File exchange** | checkpoint files | safetensors, tokenizer.json | safetensors, tokenizer.json | serialized model formats |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| GGUF k-quants, ISQ, FP8 | [Quantization: Math → Code](../quantization/) | The formats these frameworks load |
| Continuous batching, KV cache, SLOs | [LLM Systems & Inference](../) | What the runtime/server layers implement |
| Polyglot idioms (C, Rust, Zig) | [Polyglot Practice](../../../practice/) | Same algorithm, different language tradeoffs |
| K8s, autoscaling, OpenAI API | [Cloud Native](../../../systems/03-cloud-native/) | Deploying any of these servers |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Meta / ggml-org | llama.cpp, ggml, GGUF -- the C foundation | Expert |
| HuggingFace | candle, ratchet, TGI -- Rust + Python tiers | Expert |
| ZML (Paris) | Zig + MLIR/XLA multi-vendor inference | PhD-level |
| Ollama / LM Studio | productizing llama.cpp for local use | Advanced |
| NVIDIA | TensorRT-LLM, CUTLASS kernels under the runtimes | Expert |
