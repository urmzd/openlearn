# Inference Frameworks: The Cross-Language Landscape

Once you know how a model is served in principle, the practical question is *where you run it*. The answer depends on far more than raw speed. A datacenter serving millions of requests, a laptop running a model offline, a browser tab, and an embedded device each want something different, and the inference ecosystem has split along language lines to serve them: a **C/C++ foundation** ([llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml)), a **Python serving tier** (vLLM, TGI, SGLang, TensorRT-LLM), a set of **Rust equivalents** (candle, mistral.rs, burn, ratchet, luminal), and an emerging **Zig** stack (ZML, llama.cpp.zig).

This section maps that landscape. The throughline is that every framework is the same three layers, a kernel and tensor core, a runtime, and a server, assembled in a different language for a different deployment target. Once you can see those layers, choosing a framework becomes a matter of choosing a deployment story, and the choice of language largely makes it for you.

This section builds on the chapter opening, [LLM Systems & Inference](../README.md), particularly its tour of serving frameworks and of deployment on GPU clusters, and it assumes basic reading knowledge of C, Rust, and Zig. The quantization formats these frameworks consume are derived in [Quantization: Math to Code](../quantization/README.md), and deployment choices and operating costs for hosted platforms (Anyscale, Ray, vLLM, SGLang, RouteLLM, and OpenRouter) are in [LLM Serving Platforms](../serving-platforms.md). Runnable examples are in [`code/`](code/).

> **Learning objectives**
> - Decompose any inference framework into its kernel, runtime, and server layers, and say which layers it owns and which it borrows.
> - Explain why GGUF and the OpenAI-compatible HTTP API act as the universal exchange formats between frameworks.
> - Compare the C, Python, Rust, and Zig tiers by deployment target, and pick a framework for a given target with a reason.
> - Describe the two Zig idioms, wrapping C with `@cImport` and writing pure-Zig SIMD kernels, and when each fits.

## One stack, many slices

There is no single "inference framework". There is a **stack**, and each project stakes out a slice of it. Python frameworks such as vLLM maximize datacenter GPU throughput. C (llama.cpp) maximizes portability and minimizes dependencies, so a model runs on a laptop. Rust (candle, mistral.rs) trades a little ecosystem maturity for memory safety and a single static binary. Zig (ZML) bets on compiling the whole model graph to a standalone native artifact via MLIR. Pick the language and you have largely picked the deployment story.

The stack itself has three layers, and the table below shows what each one does and who implements it.

| Layer | Job | Examples |
|-------|-----|----------|
| **Kernels / tensor core** | matmul, attention, dequant on CPU/GPU | ggml, CUTLASS/cuBLAS, candle-core, Metal/Vulkan shaders, MLIR+XLA |
| **Runtime** | load weights, run the graph, manage KV cache, sample tokens | llama.cpp, candle-transformers, mistral.rs engine, ZML |
| **Scheduler / server** | continuous batching, request queue, OpenAI HTTP API | `llama-server`, vLLM, TGI, SGLang, mistralrs-server, LLMD |

Most "frameworks" are a *vertical slice* through these layers, and they differ in which layers they own and which they borrow. Ollama is the llama.cpp runtime plus a friendly model manager. mistral.rs is candle's kernels plus its own runtime plus an OpenAI-compatible server.

Two artifacts tie the layers together across projects. **GGUF is the lingua franca** of portable inference: a single quantized model file from llama.cpp's ecosystem runs under llama.cpp, candle, mistral.rs, and the Zig bindings alike. And the **OpenAI-compatible HTTP API is the universal contract** at the top: swap llama.cpp's `llama-server` for vLLM, for mistral.rs, or for ZML's LLMD, and clients do not change. With the layers and the two contracts in view, we can walk the tiers one language at a time, starting at the bottom.

## The C/C++ foundation: llama.cpp and ggml

llama.cpp is the most-deployed inference engine on earth, and most of the non-Python ecosystem rests on it. Its core is **[ggml](https://github.com/ggml-org/ggml)**, a dependency-free C tensor library with a static computation graph and a quantization-first design, with backends for CPU (AVX and NEON), Metal, CUDA, Vulkan, SYCL, and ROCm. On top of ggml, **[llama.cpp](https://github.com/ggml-org/llama.cpp)** provides the transformer runtime and `llama-server`, an OpenAI-compatible server. llama.cpp also defines **GGUF**, the quantized model container, and the **k-quant and i-quant** formats, which are explained in [Quantization: Math to Code](../quantization/README.md).

The C API in `llama.h` is the integration point for *every other language*. Python (`llama-cpp-python`), Rust (`llama-cpp-2`), Zig (`@cImport`), Go, and others all bind to it. The same ggml core underlies a family of projects, including `whisper.cpp` and `stable-diffusion.cpp`, and downstream, Ollama, LM Studio, Jan, GPT4All, and KoboldCpp all wrap llama.cpp.

llama.cpp wins for edge and local use because it needs no Python and no CUDA: it is a single small binary that runs a 7B model on a laptop CPU. The tradeoff is lower datacenter GPU throughput than vLLM. Historically it had no PagedAttention-class scheduler, and its continuous batching is newer.

## The Python serving tier

The Python tier is the datacenter-throughput layer, covered in the chapter opening's tour of serving frameworks. In one line each: **vLLM** combines PagedAttention with continuous batching and is the general-purpose throughput leader; **TensorRT-LLM** uses compiled NVIDIA kernels and FP8 for maximum performance on H100; **TGI** is Hugging Face's production server; **SGLang** offers RadixAttention prefix sharing and fast structured output; and **Ray Serve and KServe** provide orchestration on top of any of these. These dominate GPU clusters. The C, Rust, and Zig tiers dominate edge, embedded, and dependency-constrained deployments.

## The Rust tier

Rust trades ecosystem breadth for **memory safety, a single static binary, and no GIL**. The current stack has several projects, each owning a different combination of layers:

| Framework | Layer(s) | Niche | Notes |
|-----------|----------|-------|-------|
| **[candle](https://github.com/huggingface/candle)** | kernels + runtime | minimalist ML, the Rust "PyTorch-lite" | CPU/CUDA/Metal/**WASM**; loads GGUF + safetensors; HuggingFace-backed |
| **[mistral.rs](https://github.com/EricLBuehler/mistral.rs)** | runtime + server | turnkey fast inference | builds on candle; **ISQ** (in-situ quant), OpenAI server, vision/tools |
| **[burn](https://burn.dev/)** | kernels + runtime | backend-agnostic train **and** infer | WGPU/CUDA/NdArray/LibTorch; own quantization API (see [Quantization: Math to Code](../quantization/README.md)) |
| **[ratchet](https://github.com/huggingface/ratchet)** | kernels + runtime | web/cross-platform GPU | wgpu-based, browser-first inference |
| **[luminal](https://github.com/jafioti/luminal)** | compiler + runtime | search-compiled kernels | tiny graph IR, compiles to fast kernels |
| **[llama-cpp-2](https://github.com/utilityai/llama-cpp-rs)** | Rust interface | reuse llama.cpp from Rust | `llama_cpp` is the higher-level wrapper |

> **Warning:** `rustformers/llm` (and the older `llama-rs`) are unmaintained. The ecosystem consolidated onto **candle** and **mistral.rs**; use those.

Pick Rust for edge and embedded targets, for WASM in the browser, for a CLI you ship as one binary, or for a service where you want Rust's safety end to end. Two examples show the shape of the code: [`code/candle_generate.rs`](code/candle_generate.rs) runs GGUF generation with candle, and [`code/mistralrs_serve.rs`](code/mistralrs_serve.rs) stands up an OpenAI-compatible server with in-situ quantization.

## The Zig tier

Zig makes a different pitch. Its interop with C is trivial, since `@cImport` reads C headers directly with no binding generator, and it compiles the whole program to a small native artifact. The Zig projects take two quite distinct approaches:

| Project | Approach | Notes |
|---------|----------|-------|
| **[ZML](https://github.com/zml/zml)** | compile model graphs via **MLIR + OpenXLA** | Zig + XLA → standalone native binaries; NVIDIA/AMD/TPU/Trainium from one codebase; LLMD OpenAI server in a ~2.4 GB image |
| **[llama.cpp.zig](https://github.com/Deins/llama.cpp.zig)** | bindings + `build.zig` over llama.cpp | reuse the C engine from Zig via `@cImport` |
| **[llm.zig](https://github.com/camconn/llm.zig)** | from scratch, no deps | pedagogical full-stack inference in pure Zig |
| **[llama2.zig](https://github.com/donge/llama2.zig)** | port of Karpathy's `llama2.c` | the cleanest "read the whole inference loop" reference |

Beneath the projects are two idioms. The first is to **wrap C**: write `@cImport({ @cInclude("llama.h"); })` and call llama.cpp directly. Zig's interop makes this *less* code than the equivalent Rust `bindgen` setup; [`code/llama_cpp.zig`](code/llama_cpp.zig) shows it. The second is to write **pure Zig kernels**, implementing the hot loops yourself with `@Vector` SIMD and explicit allocators. [`code/quant_dot.zig`](code/quant_dot.zig) does this for INT8 symmetric quantization followed by a SIMD dot product; it is the CPU twin of [`quantization/code/symmetric_quant.cu`](../quantization/code/symmetric_quant.cu).

Pick Zig when you want C-level control and a tiny dependency-free binary, when you are already calling C libraries (Zig is arguably the best C-interop language), or when you want compile-time graph specialization, as ZML provides.

## Choosing across languages

Having walked the four tiers, you can now turn the question around and start from the deployment target. This table gives the framework that fits each target and the reason:

| Deployment target | Language | Framework | Why |
|-------------------|----------|-----------|-----|
| GPU cluster, max throughput | Python | vLLM / TensorRT-LLM | PagedAttention, FP8, mature scheduler |
| Laptop / CPU / "just runs" | C | llama.cpp (`llama-server`) | no deps, GGUF, every backend |
| Single safe static binary, edge | Rust | mistral.rs / candle | memory-safe, OpenAI server, ISQ |
| Browser / WASM | Rust | candle / ratchet | near-native in-browser inference |
| Multi-vendor native compile | Zig | ZML | MLIR/XLA → standalone binary, AMD/TPU/Trainium |
| Reuse C engine, minimal binary | Zig | llama.cpp.zig | `@cImport`, no binding generator |
| Train + infer in one stack | Rust | burn | backend-agnostic, own quant API |

The same information, organized by language rather than by target, gives a compact matrix of each tier's flagship, kernel core, quantization formats, server, and strength:

| | C/C++ | Python | Rust | Zig |
|---|-------|--------|------|-----|
| **Flagship** | llama.cpp | vLLM | candle / mistral.rs | ZML |
| **Kernel core** | ggml, CUTLASS | Triton, CUTLASS | candle-core | MLIR/XLA, `@Vector` |
| **Quant formats** | GGUF k/i-quants | GPTQ/AWQ/FP8 | GGUF, ISQ | GGUF (via C), native |
| **Server** | `llama-server` | vLLM/TGI/SGLang | mistralrs-server | LLMD |
| **Best at** | portability | throughput | safe single binary | native compile + C interop |
| **File exchange** | checkpoint files | safetensors, tokenizer.json | safetensors, tokenizer.json | serialized model formats |

The best way to make the comparison concrete is to run it. Serve the *same* GGUF model under llama.cpp (`llama-server`), candle, and mistral.rs, and compare tokens per second, memory, and binary footprint. For each framework, identify its three layers and which it borrows (mistral.rs borrows candle's kernels; Ollama borrows llama.cpp's runtime). Then build one example from [`code/`](code/) per language to compare how each language handles runtimes and file interfaces.

## Where this leads

The formats these frameworks load, GGUF k-quants, in-situ quantization, and FP8, are derived in [Quantization: Math to Code](../quantization/README.md), and the continuous batching, KV cache management, and SLOs that the runtime and server layers implement are the subject of the chapter opening, [LLM Systems & Inference](../README.md), and of [Serving & Load](../serving-and-load/README.md). Seeing the same algorithm in C, Rust, and Zig is the theme of [*Practice: Predict and Build*](../../../../../practice/README.md), and deploying any of these servers on Kubernetes with autoscaling behind an OpenAI-compatible API is covered in [Cloud Native](../../../systems/03-cloud-native/README.md) in *Systems and Architecture*.

These tiers map onto real organizations. Meta and ggml-org maintain llama.cpp, ggml, and GGUF, the C foundation; Hugging Face builds candle, ratchet, and TGI across the Rust and Python tiers; ZML in Paris pursues Zig plus MLIR/XLA multi-vendor inference; Ollama and LM Studio productize llama.cpp for local use; and NVIDIA supplies TensorRT-LLM and the CUTLASS kernels underneath many runtimes.

### Further reading

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml)
- [candle](https://github.com/huggingface/candle) documentation
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
- [ZML](https://github.com/zml/zml)
- [vLLM](https://docs.vllm.ai/), [SGLang](https://github.com/sgl-project/sglang), and [TGI](https://huggingface.co/docs/text-generation-inference) documentation

## Key terms

- **kernel / tensor core layer**: the code that performs matmul, attention, and dequantization on a specific CPU or GPU backend.
- **runtime**: the layer that loads weights, executes the model graph, manages the KV cache, and samples tokens.
- **server layer**: the scheduler and request queue that batch requests and expose an HTTP API.
- **GGUF**: llama.cpp's single-file quantized model container, readable across llama.cpp, candle, mistral.rs, and Zig bindings.
- **OpenAI-compatible API**: the de facto HTTP contract that lets clients switch between serving frameworks unchanged.
- **ISQ (in-situ quantization)**: mistral.rs's quantization of weights at load time rather than ahead of time.
- **`@cImport`**: Zig's built-in that reads C headers directly, removing the need for a binding generator.

## Summary

- Every inference framework is three layers, a kernel/tensor core, a runtime, and a server, and frameworks differ in which layers they own and which they borrow.
- GGUF is the lingua franca of portable inference, and the OpenAI-compatible HTTP API is the universal contract at the top of the stack.
- Language picks the deployment target: Python for maximum-throughput GPU serving, C for "runs anywhere with no runtime", Rust for a single safe static binary at the edge, and Zig for minimal-dependency native compilation and effortless C interop.
- llama.cpp's C API is the integration point for nearly every non-Python binding, and Ollama, LM Studio, and others wrap its runtime.

## Review questions

1. Describe Ollama and mistral.rs in terms of the three-layer stack.
   <details><summary>Answer</summary>Ollama is the llama.cpp runtime (and ggml kernels underneath it) plus a friendly model manager; it borrows both lower layers. mistral.rs borrows candle's kernels, owns its own runtime, and adds an OpenAI-compatible server.</details>
2. Why can a client switch from vLLM to `llama-server` to mistral.rs without code changes, and what makes the same model file usable across llama.cpp, candle, and mistral.rs?
   <details><summary>Answer</summary>All of them expose the OpenAI-compatible HTTP API, which is the shared contract at the server layer. At the file level, GGUF is the shared container: one quantized GGUF file runs under llama.cpp, candle, mistral.rs, and the Zig bindings.</details>
3. What does llama.cpp give up relative to vLLM, and what does it gain?
   <details><summary>Answer</summary>It gives up datacenter GPU throughput: historically it lacked a PagedAttention-class scheduler, and its continuous batching is newer. It gains portability: no Python, no CUDA requirement, a single small binary, and backends from AVX and NEON to Metal, CUDA, Vulkan, SYCL, and ROCm, enough to run a 7B model on a laptop CPU.</details>
4. You need to ship a model inside a browser application. Which tier and frameworks fit, and why?
   <details><summary>Answer</summary>The Rust tier: candle supports a WASM backend, and ratchet is a wgpu-based, browser-first inference library. Both give near-native in-browser inference from a memory-safe codebase.</details>
5. What are the two Zig idioms for inference, and when would you choose each?
   <details><summary>Answer</summary>Wrapping C, using `@cImport` to call llama.cpp directly, fits when you want to reuse the mature C engine in a minimal Zig binary with less glue than Rust's `bindgen`. Writing pure Zig kernels with `@Vector` SIMD and explicit allocators fits when you want full control of the hot loops, as in `quant_dot.zig`. ZML adds a third path, compiling the model graph through MLIR and OpenXLA for multi-vendor native binaries.</details>
