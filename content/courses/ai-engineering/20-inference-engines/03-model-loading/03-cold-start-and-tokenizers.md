# Cold starts and tokenizer equivalence

This part of [Model Loading: Hub to GPU](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Estimate loading time and check the exact tokenization contract.

## Cold-start math

So far we have asked whether a model loads correctly. The other half of the problem is how fast. Cold start is a bandwidth equation, and writing it out term by term shows where the time goes:

$$T_\text{cold} = T_\text{schedule} + T_\text{fetch} + T_\text{read} + T_\text{h2d} + T_\text{init}$$

Here $T_\text{fetch} \approx S / B_\text{net}$, paid only if weights are not already on the node. Reading from disk and copying host-to-device cost $T_\text{read} + T_\text{h2d} \approx S / \min(B_\text{disk}, B_\text{pcie})$ when the two are streamed and overlapped, and their sum otherwise. $T_\text{init}$ covers the CUDA context, KV-cache profiling, CUDA graph capture, and `torch.compile` if used. The size is $S = \text{params} \times \text{bytes/param}$, or $S/t$ per rank if checkpoints are pre-sharded. Ignoring scheduling and init, the core of it is $T \approx S / \min(B_\text{net}, B_\text{disk}, B_\text{pcie})$.

Plugging in real numbers makes the lesson concrete. The table uses BF16 sizes: the 8B model is 16.06 GB (from the Llama 3.1 index), and a 70B model with 70.55 B parameters is 141.1 GB.

| Path | Effective bandwidth | 8B (16.1 GB) | 70B (141 GB) |
|------|--------------------|--------------|--------------|
| Hub or S3, single HTTP stream | ~0.2 GB/s | 80 s | 12 min |
| Object store, parallel range reads, 25 GbE | ~3 GB/s | 5.4 s | 47 s |
| Object store, parallel, 100 GbE | ~10 GB/s | 1.6 s | 14 s |
| Local NVMe Gen4 (one drive) | ~6.5 GB/s | 2.5 s | 22 s |
| NVMe RAID-0 ×4 or page cache | ~25 GB/s | 0.6 s | 5.6 s |
| PCIe Gen5 x16 host → one GPU | ~50 GB/s | 0.3 s | 2.8 s (÷ t with TP, each GPU has its own link) |

Two lessons fall out. The network and disk terms dominate by 10 to 100 times, so **cache placement beats loader tuning**: production loads from local NVMe or parallel object-store reads, never from the Hub at request time. And once weights are local, $T_\text{init}$ (graph capture and compilation, often 20 to 60 s for large models) becomes the floor, which is why the last techniques below snapshot GPU state instead of reloading.

Each technique for faster loading attacks a specific term of the equation:

| Technique | Attacks | Mechanism |
|-----------|---------|-----------|
| safetensors `mmap` | Copies | Lazy page-in, no deserialization; worst case on network filesystems (random 4 KB faults) |
| [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) (`--load-format fastsafetensors`) | `T_read + T_h2d` | Batched reads and GPUDirect Storage straight into GPU memory |
| [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) (`--load-format runai_streamer`) | `T_fetch` | Concurrent reads from S3/GCS/Azure/local into a CPU buffer, overlapped with H2D; `--model-loader-extra-config '{"concurrency":16}'` |
| [tensorizer](https://github.com/coreweave/tensorizer) (`--load-format tensorizer`) | `T_fetch` | Serialize once, stream from HTTP/S3 at line rate |
| Pre-sharded state (`sharded_state`) | `T_read` per rank | Save each TP rank's slice once; each rank reads `S/t` with no slicing |
| Local NVMe cache / warm pool | `T_fetch` | Daemonset pre-pulls pinned revisions; scale from warm nodes |
| GPU snapshotting ([cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) + CRIU, platform GPU memory snapshots) | `T_init` | Restore a process with weights and CUDA graphs already resident |

[`code/config_explain.py`](code/config_explain.py) prints this estimate for any `config.json`.

<a id="7-tokenizers-and-chat-templates"></a>

## Tokenizers and chat templates

A model can load correctly and start fast and still be wrong, because the last layer, the tokenizer, decides what the model actually sees. Four tokenizer families cover nearly every model you will serve, and each leaves a tell-tale sign in the repo:

| Family | Algorithm | Examples | Tell-tale |
|--------|-----------|----------|-----------|
| **BPE** ([Sennrich et al.](https://arxiv.org/abs/1508.07909)) | Greedy merges of frequent pairs from a merges table | GPT-2 lineage | `merges` in `tokenizer.json` |
| **Byte-level BPE** | BPE over bytes, so no unknown tokens | GPT-2, Llama 3 (128,256 vocab), Qwen | `Ġ` marks a leading space |
| **SentencePiece** ([repo](https://github.com/google/sentencepiece)) | Unigram LM or BPE over raw text, whitespace as `▁` | Llama 2, Gemma, T5 | `▁` prefix; `tokenizer.model` file |
| **tiktoken** ([repo](https://github.com/openai/tiktoken)) | Byte-level BPE with regex pre-split, fast Rust core | OpenAI `cl100k_base`, `o200k_base`; gpt-oss `o200k_harmony` | Ranks file, no merges table |

On top of the tokenizer sits the **chat template**: a Jinja program stored in `tokenizer_config.json` (as `chat_template`) or in `chat_template.jinja`. It turns a list of `{"role", "content"}` messages into the exact string the model was trained on. Two models render the same one-message conversation very differently:

```
SmolLM2:  <|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n
Llama 3 (simplified; 3.1 also injects a dated system header):
          <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

Chat templates fail silently. With the wrong template the model still sees valid tokens, so its output is fluent. It is simply conditioned on an out-of-distribution prefix: system prompts get ignored, tool calls come out malformed, refusals spike, and evals drop a few points. Nothing errors. The only proof of correctness is to diff the token ids the serving path produces against the ones training saw. Do this on every custom-model onboarding:

```python
ids_serving = tok.apply_chat_template(msgs, add_generation_prompt=True)     # what the engine sends
ids_training = tok(training_formatter(msgs), add_special_tokens=False).input_ids  # what SFT saw
assert ids_serving == ids_training, first_divergence(ids_serving, ids_training)
```

The divergences this check catches are a familiar list. The most common is a **double BOS**: the template already emits `<|begin_of_text|>`, then the rendered string is tokenized again with `add_special_tokens=True`, and the post-processor adds a second one. Others are a missing `add_generation_prompt`, a fine-tune that added special tokens (`added_tokens`) the serving tokenizer lacks, trailing whitespace or newline differences, and a template that drops the system role. Keep in mind that vLLM uses the repo's template unless `--chat-template` overrides it, and that `--generation-config auto` pulls sampling defaults from `generation_config.json`, so both files are part of the deployed contract.

## Summary

Estimate loading time and check the exact tokenization contract. State the assumptions before applying the result to a different workload.

## Review questions

1. What should you compare when a fine-tune is fluent but worse after deployment?

   <details><summary>Answer</summary>

   Compare rendered bytes, special tokens, token IDs, position handling, and generation defaults before attributing the problem to weights.

   </details>

[Chapter contents](README.md) · [Next: Portability and load triage](04-portability-and-load-triage.md)
