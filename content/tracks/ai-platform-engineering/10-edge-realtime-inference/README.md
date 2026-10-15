# Edge, Realtime & On-Device Inference

## Overview

- **Primary references**: [llama.cpp](https://github.com/ggml-org/llama.cpp) + [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) (free), [Ollama docs](https://github.com/ollama/ollama/blob/main/docs/README.md) (free)
- **Supplementary**: [Mistral 7B paper](https://arxiv.org/abs/2310.06825) (sliding-window attention, GQA), [Mixtral paper](https://arxiv.org/abs/2401.04088) (sparse MoE), [Mamba paper (Gu & Dao)](https://arxiv.org/abs/2312.00752) (state-space models), [Whisper paper](https://arxiv.org/abs/2212.04356) + [Conformer paper](https://arxiv.org/abs/2005.08100) (streaming speech encoders), [MLX](https://github.com/ml-explore/mlx) / [ExecuTorch](https://pytorch.org/executorch/) / [ONNX Runtime](https://onnxruntime.ai/) (on-device runtimes)
- **Prerequisites**: [Training & Frameworks](../01-training-and-frameworks/) (serving, parallelism), [LLM Systems & Inference](../../ml/04-llm-systems/) (KV cache, quantization, the [frameworks deep-dive](../../ml/04-llm-systems/frameworks/)), [Neural Architectures](../../ml/06-neural-architectures/) (attention, RNNs)
- **Estimated time**: 2-3 weeks at 8-10 hrs/week

## Key Takeaways

- **There is a deployment spectrum, not a binary.** The same model can run on a datacenter GPU (vLLM/TensorRT-LLM), a single workstation (llama.cpp), or a phone (MLX/ExecuTorch). The pattern is identical — convert → quantize → run — only the runtime and the budget change.
- **llama.cpp + GGUF is the end-to-end local path.** One C/C++ engine, one self-contained file format (weights + tokenizer + metadata), CPU-or-GPU, aggressive quantization — it's how a Hugging Face checkpoint becomes a binary you can run anywhere offline.
- **Realtime is an *architecture* constraint, not just an optimization.** A streaming/realtime encoder cannot see the future, so it must be **causal or chunked** — bounded right-context — trading a little accuracy for the ability to emit output as input arrives.
- **Mistral's models are a clinic in efficiency architecture.** Sliding-window attention (linear-in-context cost), grouped-query attention (small KV cache), sparse mixture-of-experts (more params, same per-token FLOPs), and Mamba/state-space variants (constant per-token cost) are the levers that make models fast, long-context, and realtime-capable.
- **State-space models change the realtime game.** A transformer's per-token cost grows with context (the KV cache); an SSM/RNN keeps a *fixed-size* recurrent state — constant time and memory per token, which is exactly what streaming wants.

## How to Study

- Take one Hugging Face model **end to end on your own machine**: convert to GGUF, quantize to 4-bit, run it with `llama-server` (OpenAI-compatible), and serve it through the [SSE](../03-streaming-sse/) path. Measure tokens/sec and memory on CPU vs GPU.
- Run the *same* model under llama.cpp, Ollama, and MLX (if on Apple silicon); compare setup, speed, and footprint.
- Implement a toy **streaming** encoder: take an offline bidirectional encoder and restrict it to causal / chunked attention; watch accuracy vs latency trade.
- Read the Mistral 7B and Mamba papers back-to-back and list *why* each choice helps latency, memory, or context length.

---

# Concepts & Techniques

## Core Insight

Two pressures push inference off the big cloud GPU. **Place**: latency, privacy, cost, and offline use want the model *near the user* — on a laptop, a phone, a car, a microphone. **Time**: interactive and especially audio/voice workloads can't wait for a whole input before responding — they must process a *stream* and emit as they go. Both pressures bottom out in the same two questions: *how small and portable can the model be* (quantization + a runtime like llama.cpp), and *how does the architecture handle partial, unfolding input* (causal/chunked attention, or a recurrent state-space model with constant per-token cost). This topic is those two questions — portability and realtime — and the architectures (largely pioneered in the open by Mistral and the SSM line) that answer them.

## 1. The Deployment Spectrum (End to End)

| Tier | Runtime | Hardware | Wins |
|------|---------|----------|------|
| **Datacenter serving** | vLLM, TensorRT-LLM (+ Triton), SGLang, orchestrated by Ray Serve / KServe | Multi-GPU (H100/A100) | Max throughput, big models |
| **Single-box / self-host** | **llama.cpp**, Ollama, candle/mistral.rs | One GPU or beefy CPU | Simplicity, cost, control |
| **On-device / edge** | **MLX** (Apple), **ExecuTorch** (PyTorch), ONNX Runtime, llama.cpp | Phone, laptop, Jetson, browser (WASM) | Privacy, offline, zero latency to cloud |

**Key idea**: it's the **same pattern at every tier** — load weights, manage a KV cache, decode token-by-token — so what you learned in [LLM Systems](../../ml/04-llm-systems/) transfers down-market. What changes is the memory budget (which forces quantization) and the runtime (which trades features for portability). The cross-language landscape of these engines is mapped in the [Inference Frameworks deep-dive](../../ml/04-llm-systems/frameworks/).

## 2. llama.cpp & GGUF — the Local End-to-End Path

**Key ideas**:
- **llama.cpp**: a dependency-light C/C++ inference engine (built on the `ggml` tensor library) that runs LLaMA-family *and* dozens of other architectures on **CPU and GPU** (Metal, CUDA, Vulkan, ROCm). It made "run a real LLM on your laptop" normal.
- **GGUF**: a single-file format holding **weights + quantization + tokenizer + metadata**. Self-contained and mmap-able → fast load, easy distribution. It's the de-facto interchange format for local models.
- **The end-to-end pipeline**:
  1. **Convert**: `convert_hf_to_gguf.py` turns a Hugging Face checkpoint into a GGUF file.
  2. **Quantize**: `llama-quantize` compresses to 8/5/4/3/2-bit using **k-quants** (mixed per-block precision) or **i-quants** (importance-matrix-guided) — see [Quantization](../../ml/04-llm-systems/quantization/).
  3. **Run**: `llama-cli` (one-shot) or `llama-server` (an **OpenAI-compatible** HTTP server with token streaming → plug straight into the [SSE](../03-streaming-sse/) last mile).
- **Ollama**: a model manager + daemon *on top of* llama.cpp — `ollama run mistral` handles download, GGUF, templating, and an API. The ergonomic front-end to the same engine.
- **Why it matters for a platform**: this is your offline, air-gapped, edge, and dev-loop story — and a cheap way to serve small/quantized models without a GPU fleet.

## 3. On-Device Runtimes

- **MLX** (Apple): array framework tuned for Apple-silicon unified memory; great for Macs/iPhones.
- **ExecuTorch** (PyTorch): export a PyTorch model to a portable, ahead-of-time-compiled runtime for phones and microcontrollers.
- **ONNX Runtime / TFLite**: export to a standard graph, run cross-platform with hardware delegates (NNAPI, CoreML, QNN).
- **llama.cpp / candle / ZML**: also target edge (WASM, static binaries) — see the [frameworks deep-dive](../../ml/04-llm-systems/frameworks/).
- **The constraint**: tiny memory and power budgets force small models (SLMs, [topic 01](../01-training-and-frameworks/)) + heavy quantization + sometimes NPU-specific kernels.

## 4. Realtime & Streaming Encoders

**Why "realtime" is an architecture decision, not a flag**

**Key ideas**:
- **Offline vs streaming encoder**:
  - An **offline** encoder is **bidirectional** — it attends over the *whole* input (every token sees past and future). Best accuracy; needs the complete input first (e.g. Whisper over a full 30s audio window).
  - A **streaming/realtime** encoder must emit output as input *arrives*, so it can't attend to the future. It is **causal** (only past context) or **chunked/look-ahead** (a small bounded right-context window). This is the core trade: **latency vs accuracy** via how much future you're willing to wait for.
- **Speech is the canonical realtime workload** (live transcription, voice agents, captioning):
  - **Whisper** — encoder-decoder transformer, *offline* by design (fixed 30s windows); "streaming Whisper" fakes it with sliding chunks + overlap and re-decoding.
  - **Conformer** — convolution-augmented transformer; the standard *streaming* ASR encoder, run with limited/chunked context for low latency.
  - **CTC** and **RNN-Transducer (RNN-T)** — streaming-native decoders that emit tokens frame-by-frame without waiting for the utterance to end; the backbone of on-device voice.
- **Realtime metrics**: first-response latency, **real-time factor (RTF)** = processing-time / audio-duration (must be < 1 to keep up), and chunk size (the latency knob).
- **KV cache for streaming text**: the same continuous-decode loop you serve text with *is* a stream; the realtime question is whether the **encoder** of an audio/multimodal model can run incrementally.

## 5. Architectures for Efficiency & Realtime (the Mistral clinic)

**Mistral's open models are a tour of the levers that make inference fast, long-context, and streaming-capable** (architecture foundations in [Neural Architectures](../../ml/06-neural-architectures/) and [Foundation Models](../../ml/05-foundation-models/)):

- **Sliding-Window Attention (SWA)** — *Mistral 7B*: each token attends only to the last `W` tokens, not all of them. Attention cost goes from O(n²) toward **linear in context**, and stacking layers still propagates information far (effective receptive field grows with depth). Smaller attention compute and a bounded KV footprint per layer.
- **Grouped-Query Attention (GQA)** — *Mistral 7B*: query heads share a smaller number of key/value heads, **shrinking the KV cache** (the thing that dominates decode memory/bandwidth) with negligible quality loss. The single biggest practical decode-speed lever for transformers.
- **Sparse Mixture-of-Experts (MoE)** — *Mixtral 8×7B*: route each token to the top-2 of 8 expert MLPs. **Parameter count (capacity) rises without raising per-token FLOPs** — more knowledge, same compute per token — at the cost of memory to hold all experts (ties to [expert parallelism](../01-training-and-frameworks/)).
- **State-Space Models / Mamba** — *Codestral Mamba*: replace attention with a **selective state-space** recurrence. Inference is **linear in sequence length with a fixed-size recurrent state** — *constant* time and memory per token, no growing KV cache. That constant per-step cost is precisely what realtime/streaming and very-long-context want.
- **Compact & on-device models** — *Ministral 3B/8B* and small Mistral variants: tuned to run quantized on edge hardware.
- **Multimodal / audio** — *Voxtral*: a speech-understanding model (Whisper-style audio encoder feeding a language model) for transcription and spoken-query understanding — Mistral's entry into the realtime-audio space.

**The through-line**: the quadratic, KV-cache-growing vanilla transformer is the baseline; **SWA, GQA, MoE, and SSMs are four different escapes from its cost curve**, each trading something (context shape, heads, memory, exactness) for speed, footprint, or streaming-friendliness. Recognizing *which* curve a model bends is how you predict its serving behavior.

## 6. Putting It End-to-End

A concrete local, realtime-capable stack — every piece links to its topic:
1. **Pick/shrink the model**: an SLM or an SWA/GQA/SSM model ([topic 01](../01-training-and-frameworks/)), quantized to GGUF ([quantization](../../ml/04-llm-systems/quantization/)).
2. **Run it**: `llama-server` / Ollama / MLX locally, OpenAI-compatible.
3. **Stream it**: tokens out over [SSE](../03-streaming-sse/); for audio, a streaming Conformer/RNN-T encoder feeding the model.
4. **Ground it**: local [RAG](../07-retrieval-and-rag/) over an on-device vector store, [permission-filtered](../08-authorization-and-access-control/) if multi-user.
5. **Measure it**: tokens/sec, RTF, [bits-per-byte/quality](../09-llm-evaluation/) of the quantized vs full model.

---

## Decision Cheat Sheet

| Need | Reach for |
|------|-----------|
| Max throughput, large model, GPU fleet | vLLM / TensorRT-LLM |
| Run a real LLM locally, offline, end-to-end | **llama.cpp + GGUF** (or Ollama) |
| Phone / laptop / embedded | MLX, ExecuTorch, ONNX Runtime, TFLite |
| Shrink the KV cache | GQA (and KV-cache quantization) |
| Long context, cheap attention | Sliding-window attention |
| More capacity, same per-token FLOPs | Sparse MoE |
| Constant per-token cost / very long / streaming | State-space model (Mamba) |
| Stream speech in realtime | Conformer / RNN-T (chunked, causal) |
| Offline best-accuracy transcription | Whisper (full window) |

## Patterns Worth Internalizing

- **Convert → quantize → run is the same pipeline at every tier** — datacenter to phone; only the runtime and budget change.
- **Realtime forbids the future** — streaming encoders are causal or chunked; latency vs accuracy is set by how much right-context you allow.
- **Know which cost curve a model bends** — vanilla attention is O(n²) with a growing KV cache; SWA, GQA, MoE, and SSMs each escape a different part of that cost.
- **Constant per-token state beats a growing cache for streaming** — the SSM/RNN property that makes realtime and long-context cheap.
- **The local server speaks OpenAI** — llama.cpp/Ollama expose the same API and token stream, so the [SSE](../03-streaming-sse/) and app layers don't change between cloud and edge.

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Quantization (k/i-quants, GGUF) | [LLM Systems — Quantization](../../ml/04-llm-systems/quantization/) | Shrinking models for edge |
| Cross-language inference engines | [LLM Systems — Frameworks](../../ml/04-llm-systems/frameworks/) | llama.cpp/candle/ZML landscape |
| Attention variants, RNNs, CNNs | [Neural Architectures](../../ml/06-neural-architectures/) | SWA/GQA/MoE/SSM foundations |
| Transformer/attention math, KV cache | [LLM Systems](../../ml/04-llm-systems/) | Why GQA/SWA help decode |
| SLMs, quantization, adapters | [Training & Frameworks](../01-training-and-frameworks/) | The small models that fit on-device |
| Token/audio streaming, cancellation | [Streaming & SSE](../03-streaming-sse/) | Delivering realtime output |
| Quantized-vs-full quality, RTF | [LLM Evaluation](../09-llm-evaluation/) | Measuring the trade |

## How Companies Apply These Patterns

| Company | The pattern they lean on | Instance |
|---------|--------------------------|----------|
| Mistral | Efficiency architectures, open weights | SWA, GQA, Mixtral MoE, Codestral Mamba, Voxtral |
| ggml-org / Ollama | Portable local inference | llama.cpp + GGUF, Ollama |
| Apple | On-device unified-memory inference | MLX, CoreML, on-device models |
| NVIDIA | Streaming ASR + edge | Riva (Conformer/RNN-T), Jetson |
| OpenAI | Offline robust transcription | Whisper |
| Meta / Google | State-space & efficient attention research | (SSMs, linear attention, on-device Gemma/LLaMA) |
