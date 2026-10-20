# Edge and Realtime Inference

Two pressures push inference off the big cloud GPU. The first is **place**: latency, privacy, cost, and offline use all want the model near the user, on a laptop, a phone, a car, or a microphone. The second is **time**: interactive workloads, and especially audio and voice, cannot wait for a whole input before responding. They must process a stream and emit output as they go. Both pressures come down to the same two questions. How small and portable can the model be? That is answered by quantization and a runtime like llama.cpp. And how does the architecture handle partial, unfolding input? That is answered by causal or chunked attention, or by a recurrent state-space model with constant per-token cost.

This chapter is about those two questions, portability and realtime, and about the architectures, largely pioneered in the open by Mistral and the state-space model line, that answer them. It moves from the deployment spectrum (datacenter, single box, device) to the concrete local path through llama.cpp and GGUF, then to on-device runtimes, then to the architectural side: why realtime encoders must be causal or chunked, and how sliding-window attention, grouped-query attention, mixture-of-experts, and state-space models each bend the cost curve of the vanilla transformer. It closes by assembling a complete local, realtime-capable stack.

> **Learning objectives**
> - Describe the deployment spectrum and explain why convert, quantize, run is the same pipeline at every tier.
> - Take a Hugging Face checkpoint to a quantized GGUF file served by an OpenAI-compatible local server.
> - Explain why a streaming encoder must be causal or chunked, and measure the latency-accuracy trade with real-time factor.
> - Explain which part of the transformer's cost curve sliding-window attention, GQA, sparse MoE, and state-space models each escape.
> - Assemble a local stack that streams, grounds, and measures a quantized model.

## Before you begin

This chapter assumes you know how models are served and parallelized from [Training and Frameworks](../16-training-frameworks/), and the KV cache and quantization from [LLM Systems and Inference](../20-inference-engines/) in *AI Engineering*, including its [inference frameworks deep-dive](../20-inference-engines/02-frameworks/). The architecture discussion builds on attention and RNNs from [Neural Architectures](../03-neural-architectures/). Streaming output uses the path from [Streaming and SSE](../../systems/04-streaming/).

## The deployment spectrum

There is a deployment spectrum, not a binary. The same model can run on a datacenter GPU, a single workstation, or a phone, and the three tiers compare like this:

| Tier | Runtime | Hardware | Wins |
|------|---------|----------|------|
| **Datacenter serving** | vLLM, TensorRT-LLM (with Triton), SGLang, orchestrated by Ray Serve or KServe | Multi-GPU (H100/A100) | Maximum throughput, big models |
| **Single-box / self-host** | **llama.cpp**, Ollama, candle/mistral.rs | One GPU or a strong CPU | Simplicity, cost, control |
| **On-device / edge** | **MLX** (Apple), **ExecuTorch** (PyTorch), ONNX Runtime, llama.cpp | Phone, laptop, Jetson, browser (WASM) | Privacy, offline use, no round trip to the cloud |

The key idea is that it is the **same pattern at every tier**: load weights, manage a KV cache, decode token by token. What you learned in [LLM Systems](../20-inference-engines/) transfers down-market. What changes is the memory budget, which forces quantization, and the runtime, which trades features for portability. The cross-language landscape of these engines is mapped in the [inference frameworks deep-dive](../20-inference-engines/02-frameworks/).

## llama.cpp and GGUF: the local end-to-end path

Of the single-box and edge runtimes, one stands out as the end-to-end local path. **llama.cpp** is a dependency-light C/C++ inference engine, built on the `ggml` tensor library, that runs LLaMA-family models and dozens of other architectures on CPU and GPU (Metal, CUDA, Vulkan, ROCm). It made running a real LLM on a laptop normal. Its file format, **GGUF**, holds weights, quantization, tokenizer, and metadata in a single file. Because the file is self-contained and can be memory-mapped, it loads fast and distributes easily, and it has become the de facto interchange format for local models.

The pipeline from a Hugging Face checkpoint to a running model has three steps:

1. **Convert**: `convert_hf_to_gguf.py` turns a Hugging Face checkpoint into a GGUF file.
2. **Quantize**: `llama-quantize` compresses the weights to 8, 5, 4, 3, or 2 bits using **k-quants** (mixed per-block precision) or **i-quants** (guided by an importance matrix); see [Quantization](../20-inference-engines/01-quantization/).
3. **Run**: `llama-cli` runs one-shot, and `llama-server` is an **OpenAI-compatible** HTTP server with token streaming that plugs straight into the [SSE](../../systems/04-streaming/) last mile.

**Ollama** is a model manager and daemon built on top of llama.cpp: `ollama run mistral` handles the download, the GGUF file, prompt templating, and an API. It is the ergonomic front end to the same engine. For a platform, this path is your offline, air-gapped, edge, and development-loop story, and a cheap way to serve small or quantized models without a GPU fleet.

The best way to internalize it is to take one Hugging Face model end to end on your own machine: convert it to GGUF, quantize it to 4-bit, run it with `llama-server`, serve it through the SSE path, and measure tokens per second and memory on CPU versus GPU. Then run the same model under llama.cpp, Ollama, and MLX (if you are on Apple silicon), and compare setup, speed, and footprint.

## On-device runtimes

Below the single box, a family of runtimes targets phones, laptops, and embedded hardware. **MLX** is Apple's array framework, tuned for the unified memory of Apple silicon and well suited to Macs and iPhones. **ExecuTorch** exports a PyTorch model to a portable, ahead-of-time-compiled runtime for phones and microcontrollers. **ONNX Runtime** and **TFLite** export to a standard graph and run it cross-platform with hardware delegates such as NNAPI, CoreML, and QNN. llama.cpp, candle, and ZML also target the edge through WASM and static binaries, as the [frameworks deep-dive](../20-inference-engines/02-frameworks/) describes. Whatever the runtime, the constraint is the same: tiny memory and power budgets force small language models (see [Training and Frameworks](../16-training-frameworks/)), heavy quantization, and sometimes NPU-specific kernels.

## Realtime and streaming encoders

Portability is only half the story. Realtime is an *architecture* constraint, not just an optimization, and the reason is the encoder.

An **offline** encoder is bidirectional: it attends over the whole input, so every token sees both past and future. That gives the best accuracy but requires the complete input first, as Whisper does over a full 30-second audio window. A **streaming** or realtime encoder must emit output as input arrives, so it cannot attend to the future. It is either **causal**, seeing only past context, or **chunked** with a small bounded look-ahead window. This is the core trade: latency against accuracy, set by how much future you are willing to wait for. A useful exercise is to take an offline bidirectional encoder, restrict it to causal or chunked attention, and watch accuracy trade against latency.

Speech is the canonical realtime workload: live transcription, voice agents, and captioning. **Whisper** is an encoder-decoder transformer that is offline by design, with fixed 30-second windows; "streaming Whisper" fakes streaming with sliding, overlapping chunks and re-decoding. The **Conformer**, a convolution-augmented transformer, is the standard streaming speech recognition encoder, run with limited or chunked context for low latency. **CTC** and the **RNN-Transducer (RNN-T)** are streaming-native decoders that emit tokens frame by frame without waiting for the utterance to end, and they are the backbone of on-device voice.

Realtime systems are measured by first-response latency, by **real-time factor**, $\text{RTF} = \text{processing time} / \text{audio duration}$, which must be below 1 to keep up, and by chunk size, which is the latency knob. For text, note that the continuous decode loop with a KV cache that you already serve text with *is* a stream. The realtime question is whether the **encoder** of an audio or multimodal model can run incrementally.

## Architectures for efficiency and realtime

That raises the question of what architectures make inference fast, long-context, and streaming-capable. Mistral's open models are a tour of the levers. The foundations are in [Neural Architectures](../03-neural-architectures/) and [Foundation Models](../05-foundation-models/).

**Sliding-window attention (SWA)**, used in Mistral 7B, lets each token attend only to the last $W$ tokens rather than all of them. Attention cost goes from $O(n^2)$ toward linear in context, and stacking layers still propagates information far, because the effective receptive field grows with depth. Attention compute shrinks and the KV footprint per layer is bounded.

**Grouped-query attention (GQA)**, also in Mistral 7B, lets query heads share a smaller number of key and value heads. That shrinks the KV cache, which dominates decode memory and bandwidth, with negligible quality loss, and it is the single biggest practical decode-speed lever for transformers.

**Sparse mixture-of-experts (MoE)**, in Mixtral 8x7B, routes each token to the top 2 of 8 expert MLPs. Parameter count, and so capacity, rises without raising per-token FLOPs: more knowledge for the same compute per token, at the cost of the memory needed to hold every expert (which ties to expert parallelism in [Training and Frameworks](../16-training-frameworks/)).

**State-space models such as Mamba**, used in Codestral Mamba, replace attention with a selective state-space recurrence. Inference is linear in sequence length with a fixed-size recurrent state: constant time and memory per token, and no growing KV cache. A transformer's per-token cost grows with context because of the KV cache; an SSM or RNN keeps a fixed-size state, which is exactly what streaming and very long contexts want.

Mistral also ships compact models for the edge, such as Ministral 3B and 8B and other small variants tuned to run quantized on edge hardware, and a multimodal audio model, Voxtral, which feeds a Whisper-style audio encoder into a language model for transcription and spoken-query understanding.

The through-line is this. The quadratic, KV-cache-growing vanilla transformer is the baseline, and SWA, GQA, MoE, and SSMs are four different escapes from its cost curve, each trading something (context shape, heads, memory, exactness) for speed, footprint, or streaming-friendliness. Recognizing which curve a model bends is how you predict its serving behaviour. Reading the Mistral 7B and Mamba papers back to back, and listing why each choice helps latency, memory, or context length, is a good way to make this concrete.

## Putting it together end to end

A concrete local, realtime-capable stack draws on every part of this book:

1. **Pick and shrink the model**: a small model, or an SWA, GQA, or SSM model ([Training and Frameworks](../16-training-frameworks/)), quantized to GGUF ([Quantization](../20-inference-engines/01-quantization/)).
2. **Run it**: `llama-server`, Ollama, or MLX locally, all OpenAI-compatible.
3. **Stream it**: tokens out over [SSE](../../systems/04-streaming/); for audio, a streaming Conformer or RNN-T encoder feeding the model.
4. **Ground it**: local [RAG](../27-retrieval-and-rag/) over an on-device vector store, [permission-filtered](../../software-engineering/10-authorization/) if there are multiple users.
5. **Measure it**: tokens per second, RTF, and the [bits-per-byte and quality](../29-evaluation/) of the quantized model against the full one.

Because the local server speaks the OpenAI API and the same token stream, the SSE and application layers do not change between cloud and edge.

When choosing among these options, this mapping from need to tool summarizes the chapter:

| Need | Reach for |
|------|-----------|
| Maximum throughput, large model, GPU fleet | vLLM / TensorRT-LLM |
| Run a real LLM locally, offline, end to end | **llama.cpp + GGUF** (or Ollama) |
| Phone, laptop, or embedded | MLX, ExecuTorch, ONNX Runtime, TFLite |
| Shrink the KV cache | GQA (and KV-cache quantization) |
| Long context, cheap attention | Sliding-window attention |
| More capacity, same per-token FLOPs | Sparse MoE |
| Constant per-token cost, very long context, or streaming | State-space model (Mamba) |
| Stream speech in realtime | Conformer / RNN-T (chunked, causal) |
| Offline best-accuracy transcription | Whisper (full window) |

## Where these ideas show up in practice

Mistral leans on efficiency architectures and open weights: SWA, GQA, the Mixtral MoE, Codestral Mamba, and Voxtral. ggml-org and Ollama make local inference portable with llama.cpp, GGUF, and Ollama. Apple runs on-device inference over unified memory with MLX, CoreML, and on-device models. NVIDIA combines streaming speech recognition (Riva, with Conformer and RNN-T) with edge hardware (Jetson). OpenAI's Whisper is the reference for robust offline transcription. Meta and Google research state-space models, linear attention, and on-device models such as Gemma and LLaMA.

The chapter connects to several others. Quantization with k-quants, i-quants, and GGUF is covered in [LLM Systems: Quantization](../20-inference-engines/01-quantization/), and the llama.cpp, candle, and ZML landscape in [LLM Systems: Frameworks](../20-inference-engines/02-frameworks/). The attention variants, RNNs, and CNNs behind SWA, GQA, MoE, and SSMs are in [Neural Architectures](../03-neural-architectures/), and the attention math and KV cache that explain why GQA and SWA help decode are in [LLM Systems](../20-inference-engines/). The small models, quantization, and adapters that fit on-device are in [Training and Frameworks](../16-training-frameworks/). Token and audio streaming with cancellation is [Streaming and SSE](../../systems/04-streaming/), and measuring quantized-versus-full quality and RTF is [LLM Evaluation](../29-evaluation/).

## Key takeaways

- Convert, quantize, run is the same pipeline at every tier from datacenter to phone; only the runtime and the budget change.
- llama.cpp and GGUF are the end-to-end local path: one engine, one self-contained file, CPU or GPU, aggressive quantization.
- Realtime forbids the future: streaming encoders are causal or chunked, and latency versus accuracy is set by how much right-context you allow.
- Know which cost curve a model bends: vanilla attention is $O(n^2)$ with a growing KV cache, and SWA, GQA, MoE, and SSMs each escape a different part of that cost.
- A constant per-token state beats a growing cache for streaming, which is why state-space models change the realtime game.
- The local server speaks OpenAI, so the SSE and application layers do not change between cloud and edge.

## Key terms

- **Deployment spectrum**: the range of places a model can run, from datacenter GPU serving through a single self-hosted box to phones and edge devices; every tier loads weights, manages a KV cache, and decodes token by token, and only the runtime and memory budget change.
- **llama.cpp**: a dependency-light C/C++ inference engine built on the `ggml` tensor library that runs quantized LLMs on CPU and GPU; together with GGUF it is the standard end-to-end path for running a model locally.
- **K-quants / i-quants**: llama.cpp's weight quantization schemes for GGUF files; k-quants mix precision per block, and i-quants choose precision guided by an importance matrix measured on sample data.
- **Streaming encoder**: an encoder that emits output as input arrives, so it is causal (past context only) or chunked (a small bounded look-ahead); unlike an offline bidirectional encoder, it trades accuracy for latency.
- **Chunked attention**: attention restricted to fixed-size chunks of input with limited look-ahead, letting an encoder run incrementally on a stream; chunk size is the latency knob.
- **Real-time factor (RTF)**: processing time divided by audio duration; a streaming speech system must keep it below 1 to keep up with live input.
- **Conformer**: a convolution-augmented transformer that is the standard streaming speech recognition encoder, run with limited or chunked context for low latency.
- **Connectionist temporal classification (CTC)**: a streaming-native speech decoding objective that emits a label or blank per audio frame, so tokens appear without waiting for the utterance to end.
- **RNN-Transducer (RNN-T)**: a streaming speech decoder that combines an audio encoder with a prediction network over emitted tokens and emits output frame by frame; the backbone of on-device voice recognition.
- **State-space model (SSM)**: a sequence model, such as Mamba, that replaces attention with a learned linear recurrence over a fixed-size state, giving constant time and memory per token and no growing KV cache.

## Further reading

- [llama.cpp](https://github.com/ggml-org/llama.cpp) and the [GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) (free).
- [Ollama documentation](https://github.com/ollama/ollama/blob/main/docs/README.md) (free).
- The [Mistral 7B paper](https://arxiv.org/abs/2310.06825), for sliding-window attention and GQA.
- The [Mixtral paper](https://arxiv.org/abs/2401.04088), for sparse MoE.
- Gu and Dao, the [Mamba paper](https://arxiv.org/abs/2312.00752), for state-space models.
- The [Whisper paper](https://arxiv.org/abs/2212.04356) and the [Conformer paper](https://arxiv.org/abs/2005.08100), for speech encoders.
- [MLX](https://github.com/ml-explore/mlx), [ExecuTorch](https://pytorch.org/executorch/), and [ONNX Runtime](https://onnxruntime.ai/), for on-device runtimes.
