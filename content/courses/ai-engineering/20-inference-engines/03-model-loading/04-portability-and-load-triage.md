# Portability and load triage

This part of [Model Loading: Hub to GPU](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Separate artifact, loader, execution, and tokenizer failures.

## Beyond PyTorch: JAX, Flax NNX, and Keras 3

Everything so far assumed PyTorch, but transformers v5 dropped its TensorFlow and Flax backends, so the JAX world loads through its own stack. The pattern is the same as the meta-device skeleton, expressed in Flax NNX and Orbax: build an abstract model with shapes only, then restore into it.

```python
# Flax NNX + Orbax: build an abstract model (shapes only), then restore into it
abstract = nnx.eval_shape(lambda: Transformer(cfg, rngs=nnx.Rngs(0)))
graphdef, abstract_state = nnx.split(abstract)
state = ocp.StandardCheckpointer().restore(ckpt_dir / "state", abstract_state)
model = nnx.merge(graphdef, state)
```

`nnx.eval_shape` plays the role of PyTorch's meta device, and Orbax restores each array **directly onto its target sharding** across a device mesh, with no host-side gather; this is how MaxText and TPU serving load. Converting Hugging Face safetensors into JAX means the same key renaming plus transposes, because `nn.Linear` stores `out × in` while Flax `Dense` kernels store `in × out`.

**Keras 3 and KerasHub** presets are backend-agnostic. `keras_hub.models.CausalLM.from_preset("hf://<org>/<repo>", dtype="bfloat16")` converts supported Hugging Face architectures (Llama 3, Gemma, Mistral, Mixtral, Qwen, gpt-oss, and others in [`utils/transformers/`](https://github.com/keras-team/keras-hub/tree/master/keras_hub/src/utils/transformers)) on load and runs them on JAX, PyTorch, or TensorFlow.

## A triage playbook

The four layers give you a diagnostic order. When a customer reports a problem, start from their words, run the first command, and check the most likely causes in order:

| Customer says | First command | Most likely cause |
|---------------|---------------|-------------------|
| "Won't load" | `safetensors_inspect.py <index.json>` and diff keys vs base model | Prefix drift, missing shard, fused or renamed projections, unknown `model_type` |
| "Loads but OOMs" | `config_explain.py config.json` | Weights + KV cache at max context exceed GPU; dtype upcast to FP32 |
| "Slow to start" | Time each term of the cold-start equation separately | Pulling from the Hub per replica; network filesystem with mmap; no warm cache |
| "Gives garbage" | Same prompt in transformers BF16 vs engine | Missing `lm_head`, FP16 overflow, TP slicing of fused tensors, quant config mismatch |
| "Worse than the base model" | Diff templated token ids vs training | Chat template, double BOS, EOS list, sampling defaults |
| "Never stops" | Print `generation_config.eos_token_id` | Turn terminator not in EOS list |

The best way to internalize this table is to break a load on purpose. Run [`code/safetensors_inspect.py`](code/safetensors_inspect.py) with `--url` against three Hub models (one with tied embeddings, one FP8, one MXFP4) and explain every tensor name you see. Then rename keys with a `module.` prefix, drop `lm_head.weight`, delete `rope_scaling`, and swap the chat template, recording the symptom of each. The table you produce is your on-call runbook. Finally, read one real loader end to end: vLLM's [`llama.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py), following `hf_to_vllm_mapper` and `AutoWeightsLoader`.

## Where this leads

Every tensor name in a checkpoint corresponds to a piece of the transformer block from [Deep Learning](../../02-deep-learning/README.md): attention projections, GQA heads, MoE experts, RoPE parameters. The FP8, MXFP4, NVFP4, AWQ, and GPTQ tensors you learned to recognize here are derived in [Quantization: Math to Code](../01-quantization/README.md). LoRA adapters and chat templates come from fine-tuning and SFT data formatting, covered in [Training and Post-Training](../../19-training-methods/README.md). Getting weights onto nodes through image baking, daemonsets, and warm pools is the subject of [Cloud Native](../../../systems/09-cloud-native/README.md) in *Systems Engineering*, and measuring each cold-start term alongside time to first token belongs to [Observability](../../../systems/11-observability/README.md). Once a model loads, [Serving & Load](../04-serving-and-load/README.md) takes up what happens when traffic arrives.

This is daily work at inference providers. At Fireworks, Together, and Baseten, field engineers debug customer checkpoints, LoRA multi-tenancy, and cold-start SLAs; Hugging Face owns the Hub storage, safetensors, and the transformers loader; NVIDIA ships ModelOpt FP8 and NVFP4 checkpoints, TensorRT-LLM engine builds, and GPUDirect Storage; Modal, Replicate, and RunPod compete on scale-to-zero, GPU memory snapshots, and weight caching; Anyscale and Databricks run vLLM-based serving from model registries and object-store streaming; and Google builds on Orbax, MaxText, KerasHub, and TPU serving.

### Further reading

- [safetensors format spec](https://github.com/huggingface/safetensors#format)
- [Transformers: Loading models](https://huggingface.co/docs/transformers/main/en/models) and [Dynamic weight loading](https://huggingface.co/docs/transformers/main/en/weightconverter)
- [huggingface_hub: Download files](https://huggingface.co/docs/huggingface_hub/guides/download) and [Understand caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- [Transformers: Chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [Hub Xet storage](https://huggingface.co/docs/hub/xet/index), [`hf` CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [vLLM Run:ai Model Streamer](https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html), [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors), [CoreWeave tensorizer](https://github.com/coreweave/tensorizer), [NVIDIA cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)
- [PEFT LoRA docs](https://huggingface.co/docs/peft) and the [LoRA paper](https://arxiv.org/abs/2106.09685)
- [FP8 formats paper](https://arxiv.org/abs/2209.05433) and the [OCP Microscaling (MX) spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Flax NNX checkpointing](https://flax.readthedocs.io/en/latest/guides/checkpointing.html) and [KerasHub](https://keras.io/keras_hub/)

## Key terms

- **safetensors**: a weight format consisting of an 8-byte little-endian header length, a JSON header of tensor names, dtypes, shapes, and offsets, and a flat byte buffer.
- **revision**: a commit sha in a Hub repo; pinning one makes a load reproducible, while `main` can change.
- **meta device**: a PyTorch device that records tensor shapes without allocating memory, used to build a model skeleton before filling it from a checkpoint.
- **fused projection**: an engine-side parameter such as `qkv_proj` or `gate_up_proj` that concatenates several checkpoint tensors into one GEMM.
- **column- and row-parallel**: the two tensor-parallel slicing patterns, splitting a weight's output rows (no communication) or its input columns (partial sums followed by an all-reduce).
- **LoRA adapter**: a low-rank update $W' = W + (\alpha/r)BA$ stored separately from the base model.
- **cold start**: the time from scheduling a replica to serving its first request, dominated by fetching and reading weight bytes and then by engine initialization.
- **chat template**: a Jinja program that renders a list of chat messages into the exact token sequence the model was trained on.

## Summary

- A model is config plus weights plus tokenizer plus generation defaults, each in a different file; garbage output is more often a tokenizer or config bug than a weights bug.
- safetensors is a JSON header plus a flat byte buffer, so you can inspect any checkpoint, even remotely, by reading its first `8 + N` bytes.
- `from_pretrained` and engine loaders are key-matching problems; prefix drift, tied weights, fused projections, and tensor-parallel slicing of fused tensors cause most failures.
- Cold start is roughly $S / \min(B_\text{net}, B_\text{disk}, B_\text{pcie})$ plus engine init; cache placement beats loader tuning, and production never loads from the Hub at request time.
- Chat templates fail silently; only diffing templated token ids against training proves correctness.

## Review questions

1. A customer's checkpoint loads with "Some weights were not initialized" for every layer, and the unexpected keys all start with `module.`. What happened, and how do you fix it?
   <details><summary>Answer</summary>The checkpoint was saved from a DDP wrapper (similar prefixes come from `torch.compile`, `_orig_mod.`, and PEFT, `base_model.model.`), so every name carries a prefix the model's parameters lack and no key matches. Diff the header keys against `model.state_dict().keys()` to confirm, then strip the prefix or re-save from the unwrapped model (for PEFT, `merge_and_unload()` first).</details>
2. How can you learn the dtypes and shapes of every tensor in a 100 GB remote checkpoint without downloading it?
   <details><summary>Answer</summary>Read the safetensors header with HTTP `Range` requests: the first 8 bytes give the header length `N` as a little-endian u64, and the next `N` bytes are a JSON object listing each tensor's dtype, shape, and data offsets. Two range requests suffice; `safetensors_inspect.py --url` does exactly this.</details>
3. Hand-derive the parameter count of Llama 3.1 8B from its config (vocabulary 128,256; hidden size 4096; 32 layers; 32 query heads and 8 KV heads of dimension 128; MLP intermediate size 14,336; untied embeddings). Check it against the index file.
   <details><summary>Answer</summary>The embedding and the untied `lm_head` are each $128{,}256 \times 4096 = 525{,}336{,}576$. Per layer, attention has $q$ and $o$ at $4096 \times 4096 = 16{,}777{,}216$ each and $k$ and $v$ at $4096 \times 1024 = 4{,}194{,}304$ each, for 41,943,040; the MLP has three $4096 \times 14{,}336$ matrices, 176,160,768; and two RMSNorm vectors add 8,192. That is 218,112,000 per layer, or 6,979,584,000 for 32 layers. Adding both embedding matrices and the final 4,096-element norm gives 8,030,261,248, and at 2 bytes each in BF16 that is 16,060,522,496 bytes, exactly the index's `metadata.total_size`.</details>
4. Why must a fused `gate_up_proj` be sliced per sub-matrix under tensor parallelism, and why is getting it wrong so dangerous?
   <details><summary>Answer</summary>The fused tensor is $[G; U]$, and each rank needs its slice of both, $[G_r; U_r]$. Taking the contiguous $r$-th chunk of the fused tensor would give rank 0 all of $G$ and the last rank all of $U$. Because the shapes still match, this loads without error and silently produces garbage.</details>
5. Using the cold-start table, explain why "cache placement beats loader tuning", and what becomes the floor once weights are local.
   <details><summary>Answer</summary>A 70B BF16 model takes about 12 minutes over a single HTTP stream but about 22 s from one local NVMe drive and 2.8 s across PCIe Gen5; the network and disk terms dominate the others by 10 to 100 times, so where the bytes sit matters far more than how the loader reads them. Once weights are local, engine initialization (CUDA graph capture and compilation, often 20 to 60 s) is the floor, which GPU state snapshotting attacks.</details>
6. An instruct fine-tune never stops generating and keeps printing `assistant` turns. What do you check?
   <details><summary>Answer</summary>Compare `generation_config.eos_token_id` with the template's turn terminator. In Llama 3 an instruct turn ends with `<|eot_id|>` (128009); if only `<|end_of_text|>` (128001) is listed as EOS, generation runs past each turn. Make `eos_token_id` a list containing every terminator.</details>
7. Why are chat-template mismatches silent, and what is the one reliable test?
   <details><summary>Answer</summary>The model still receives valid tokens, so it produces fluent text; it is merely conditioned on an out-of-distribution prefix, which shows up as ignored system prompts, malformed tool calls, more refusals, and a few points lost on evals. The reliable test is to compare the token ids from `apply_chat_template(..., add_generation_prompt=True)` with the ids of the training formatter's output tokenized with `add_special_tokens=False`, and assert they are identical.</details>

[Chapter contents](README.md)
