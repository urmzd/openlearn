# From Training Run to Shared Model: Packaging, Publishing, and Use Cases

Most of this chapter takes the consumer's side: someone else trained a model, and you load and serve it. This section takes the producer's side. You have trained or fine-tuned a model, and other people (another team, your customers, the open-source community) need to use it. What you hand them decides whether it loads in their engine on the first try, whether it behaves the way it did in your evaluations, and whether they can serve it economically. A checkpoint that only your training code can read is not yet a model anyone else can use.

The section answers four questions in order. How is a model's architecture actually stored, so that an engine you have never met can rebuild it? How does LoRA work during training and at serving time, since it decides what you ship? How do you publish and share, as weights, as adapters, or as an endpoint? And what do practical deployments look like, with the GPUs and replicas each needs?

Read [Model Loading: Hub to GPU](03-model-loading/README.md) first: it describes from the loader's side every file this section asks you to produce. The training side, mixed precision and sharded optimizers, is in [Training and Post-Training](../07-training-and-post-training/README.md), and LoRA is derived and built by hand in [LoRA, PiSSA, and merge](../../build-an-llm/07-objectives/06-lora-pissa-and-merge.md) in *Build Your Own LLM System*. The deployments at the end use the Kubernetes patterns from [Deploying LLMs on Kubernetes](06-gpu-kubernetes-deployment.md).

> **Learning objectives**
> - Explain how a model's architecture is stored as configuration plus code plus named tensors, and export a checkpoint from your own training code into a format standard engines load.
> - Compute the memory of full fine-tuning, LoRA, and QLoRA, and explain how a LoRA adapter runs merged, unmerged, and batched across tenants.
> - Publish a model or adapter with a model card, pinned revisions, and access control, and verify the published artifact the way a stranger will load it.
> - Choose between sharing weights, sharing adapters on a common base, and sharing an endpoint.
> - Size the GPUs, replicas, and engine configuration for common practical deployments.

## How a model's architecture is stored

A trained model is three things, and only one of them is in the weight files.

**The code** is the forward pass: the sequence of matmuls, norms, attention, and activations. It does not travel with the model. It lives in libraries, in `transformers`' `LlamaForCausalLM`, in vLLM's own Llama implementation, in llama.cpp's graph builder, each written independently.

**The configuration** is the set of numbers that instantiate that code: `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `rope_theta`, `rope_scaling`, `vocab_size`, `tie_word_embeddings`, and the norm epsilon. It is `config.json`, and its `model_type` and `architectures` fields name the code to use. Every engine keeps a registry from those names to its implementation.

**The weights** are a dictionary from tensor names to tensors, stored in safetensors. The names are a contract with the code: `model.layers.17.self_attn.k_proj.weight` must exist, with shape `[num_key_value_heads × head_dim, hidden_size]`, because the code will look it up.

So "the architecture" is stored as a *name* (`model_type: llama`) that points to code someone else maintains, plus numbers that size it, plus tensors whose names and shapes match it. This has a large practical consequence for a producer. If your model's architecture matches a family the engines already implement, it runs everywhere for free: vLLM, SGLang, TensorRT-LLM, llama.cpp, and the hosted providers all load a Llama-shaped checkpoint. If it does not, you must ship code, and every option for that is worse. A `modeling_*.py` in the repo with `auto_map` requires every user to pass `trust_remote_code=True` and run your Python. vLLM's Transformers backend (`--model-impl transformers`) can run a model it does not implement natively, usually more slowly. And contributing an implementation upstream to each engine is the real fix and takes weeks. The cheapest architecture decision a producer makes is to stay inside a supported family unless the change is the point of the research.

### Exporting your own checkpoint to a standard format

If you trained with your own code, as in *Build Your Own LLM System*, exporting is a renaming problem: the same bijection [Model Loading](03-model-loading/README.md) describes, run in reverse. Write a `config.json` whose fields describe your model, then map each of your tensor names to the convention the target family uses:

```python
import json, torch
from safetensors.torch import save_file

def export_llama(my_state: dict[str, torch.Tensor], cfg: dict, out_dir: str) -> None:
    rename = {
        "tok_emb.weight": "model.embed_tokens.weight",
        "norm_f.weight": "model.norm.weight",
        "head.weight": "lm_head.weight",
    }
    per_layer = {
        "attn.wq": "self_attn.q_proj", "attn.wk": "self_attn.k_proj",
        "attn.wv": "self_attn.v_proj", "attn.wo": "self_attn.o_proj",
        "mlp.gate": "mlp.gate_proj", "mlp.up": "mlp.up_proj", "mlp.down": "mlp.down_proj",
        "norm_attn": "input_layernorm", "norm_mlp": "post_attention_layernorm",
    }
    out = {}
    for name, t in my_state.items():
        if name in rename:
            out[rename[name]] = t
            continue
        _, i, rest = name.split(".", 2)          # "blocks.17.attn.wq.weight"
        mod, suffix = rest.rsplit(".", 1)
        out[f"model.layers.{i}.{per_layer[mod]}.{suffix}"] = t
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    save_file(out, f"{out_dir}/model.safetensors", metadata={"format": "pt"})
    json.dump({"architectures": ["LlamaForCausalLM"], "model_type": "llama", **cfg},
              open(f"{out_dir}/config.json", "w"), indent=2)
```

The renaming is mechanical. Three conventions are not, and each loads without error and produces garbage if you get it wrong:

1. **RoPE layout.** Hugging Face's Llama applies rotary embeddings to the two *halves* of each head (`rotate_half`), while Meta's original code rotates *interleaved* pairs. If your training code used the interleaved form, the rows of `q_proj` and `k_proj` must be permuted within each head on export; this is exactly what the `permute` step in transformers' Llama conversion script does.
2. **Linear layout.** PyTorch stores `nn.Linear` weights as `[out, in]`. Code that stored `[in, out]` and computed `x @ W` must transpose.
3. **Tied embeddings.** If `lm_head` shares the embedding matrix, set `tie_word_embeddings: true` and omit `lm_head.weight`; if they are separate, store both.

Then ship the rest of what a loader reads: the tokenizer (`tokenizer.json` and `tokenizer_config.json`) from the exact tokenizer you trained with, the **chat template** your instruction data was formatted with, and a `generation_config.json` whose `eos_token_id` lists every token that ends a turn. Most "the model loads but rambles" reports are a missing end-of-turn id here, not a weights problem.

If you trained with `transformers` or PEFT, `save_pretrained(out_dir)` writes all of this for you, in safetensors, sharded at `max_shard_size` (5 GB by default). Never ship pickle `.bin` or `.pt` files: loading them executes arbitrary code.

### Proving the export before anyone else sees it

An export is correct only when an independent loader reproduces your model's outputs. Check three things, in order, each cheaper to debug than the next:

1. **Logit parity.** Feed the same token ids to your training code and to `AutoModelForCausalLM.from_pretrained(out_dir)`. In BF16, the maximum absolute difference of the logits should be small (on the order of $10^{-2}$) and the argmax should agree at nearly every position. A disagreement everywhere is a naming or layout bug; agreement for early positions that decays with position is a RoPE bug.
2. **Template parity.** Format a conversation with `apply_chat_template(..., add_generation_prompt=True)` and with your training formatter, and assert the token ids are identical.
3. **Engine parity.** Serve the export with vLLM and compare greedy completions with `transformers` for a few hundred prompts. Expect high top-1 agreement, not bit-identity: batching changes reduction order, which [Tiled, batch-invariant matmul](../../build-an-llm/10-kernels/03-tiled-batch-invariant-matmul.md) explains. Then run your evaluation suite against the served model, because that is what users get.

## How LoRA runs

Most fine-tunes that get shared today are LoRA adapters, so it pays to know exactly what one is, what it costs to train, and what it costs to serve.

### In training

LoRA freezes the base weight $W$ and learns a low-rank update. For an adapted linear layer,

$$h = Wx + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times d_\text{in}},\ B \in \mathbb{R}^{d_\text{out} \times r}$$

$A$ starts random and $B$ starts at zero, so at step 0 the model is exactly the base model and training moves it smoothly away. Gradients flow through $W$ to reach earlier layers, but only $A$ and $B$ are updated, so only they need gradients stored and optimizer state.

That is where the memory goes. Full fine-tuning with AdamW in mixed precision stores, per parameter, BF16 weights (2 bytes), BF16 gradients (2), an FP32 master copy (4), and Adam's two moments (8): 16 bytes. LoRA stores the frozen base at 2 bytes per parameter and the 16-byte overhead only for the adapter. For Llama-3.1-8B with rank 16 on all seven projections (about 42 M adapter parameters):

| Method | Base weights | Trainable state | Total before activations | Fits on |
|--------|--------------|-----------------|--------------------------|---------|
| Full fine-tune (AdamW, mixed precision) | 8.03 B × 16 B | (included) | ≈ 128 GB | Several GPUs with ZeRO or FSDP |
| LoRA, BF16 base | 8.03 B × 2 B ≈ 16 GB | 42 M × 16 B ≈ 0.7 GB | ≈ 17 GB | One 24 to 48 GB GPU |
| QLoRA, NF4 base | ≈ 0.55 B/param ≈ 4.5 GB | ≈ 0.7 GB | ≈ 5 GB | One consumer GPU |

Activations come on top and scale with batch size and sequence length; gradient checkpointing trades them for recomputation. QLoRA stores the frozen base in 4-bit NF4 and dequantizes each block on the fly in the forward and backward pass, so it trains slower but fits on far less hardware. Note what QLoRA does *not* change: the adapter it produces is an ordinary BF16 LoRA, trained against a quantized base.

### At serving time

An adapter can run three ways, and the choice is a deployment decision, not a training one.

**Merged.** Compute $W' = W + \frac{\alpha}{r} BA$ once and save the result as a full model. It serves at exactly the base model's speed, and it costs a full copy of the weights per fine-tune. Merging into a quantized base needs dequantize, merge, requantize; merging a QLoRA adapter into the BF16 base is fine, but evaluate it, because the adapter was trained against the quantized weights.

**Unmerged, one adapter.** Keep $W$ and compute $Wx$ plus the thin path $B(Ax)$. The extra work is $2r(d_\text{in} + d_\text{out})$ FLOPs per token per adapted matrix: for the 8B example about 84 MFLOPs per token against the base's 16 GFLOPs, roughly half a percent. The real cost is not FLOPs but the extra small kernels per layer.

**Unmerged, many adapters in one batch.** This is multi-LoRA serving, and it is the reason adapters are the dominant way fine-tunes are shared. Requests for different adapters share one batch: the base matmul runs once for everyone, and a batched kernel (Punica's SGMV, or S-LoRA's unified paging) applies each request's own $B_i A_i$. One GPU holding one 16 GB base can serve hundreds of 84 MB adapters, loading the active ones into GPU memory and keeping the rest in host memory. Throughput drops as the number of *distinct* adapters per batch grows, which is why engines cap it (`--max-loras`).

The rule that follows: ship a fine-tune as an adapter unless it will carry enough traffic to fill its own GPUs, and merge only the ones that do.

## Publishing and sharing

### What to publish

A model repository is complete when a stranger can load, template, and serve it with no message to you. Before uploading, check it against this list:

| File | Must contain |
|------|--------------|
| `config.json` | A supported `model_type` and `architectures`, every shape field, `rope_scaling` if used, the stored dtype |
| `*.safetensors` (+ `model.safetensors.index.json` when sharded) | Every tensor the config implies, nothing extra |
| `tokenizer.json`, `tokenizer_config.json` | The exact training tokenizer, special tokens, and the chat template |
| `generation_config.json` | Every end-of-turn id in `eos_token_id`, sensible sampling defaults |
| `README.md` (model card) | YAML front matter (`license`, `base_model`, `base_model_relation`, `datasets`, `language`, `pipeline_tag`, `library_name`), plus intended use, training data summary, evaluation results, known limitations, and the prompt format |
| For an adapter | `adapter_config.json` (with `base_model_name_or_path` and the base's revision) and `adapter_model.safetensors` |

The `base_model` and `base_model_relation` fields (`finetune`, `adapter`, `quantized`, or `merge`) record lineage on the Hub, which is how users and tools find every derivative of a base model. Your license is constrained by the base model's: a fine-tune of a model under a community license inherits its terms, often including naming and attribution requirements, so read them before choosing your own.

### How to publish

From a training script, `model.push_to_hub("my-org/support-8b-lora", private=True)` and `tokenizer.push_to_hub(...)` upload directly. From the command line, upload a finished directory:

```bash
hf upload my-org/support-8b-lora ./out --repo-type model --private
```

Then make it addressable by version, because consumers must pin. Every upload is a commit; tag the one you evaluated:

```python
from huggingface_hub import HfApi
HfApi().create_tag("my-org/support-8b-lora", tag="v1.0", revision="<commit sha>")
```

Access control is part of publishing. A **private** repository is visible only to you or your organization. A **gated** repository is public but requires each user to request access and accept your terms, which is how most open-weight base models are distributed. Use fine-grained tokens: a write token that can push to one repository, held only by the CI job that publishes, and read-only tokens everywhere else. Finally, verify the published artifact the way a stranger will load it: on a clean machine, `hf download` the tag, start vLLM from it, and run the parity and evaluation checks again. Uploads drop files more often than training corrupts them.

### Three ways to share

"Sharing a model" means one of three different things, with different costs for you and for the people you share with:

| Share | What they get | Who pays for GPUs | Good when | Bad when |
|-------|---------------|-------------------|-----------|----------|
| **Weights** (Hub repo) | Full control: any engine, quantization, hardware | They do | Open release; partners with their own platform | The weights are a competitive asset; users lack serving skill |
| **Adapters on a shared base** | Their fine-tune, served from your multi-LoRA fleet | You, amortized across tenants | Many teams or customers, each with a small fine-tune of one base | One tenant needs isolation or very high throughput |
| **An endpoint** (OpenAI-compatible API) | Tokens, not weights | You | Internal platform; a product; keeping weights private | Users need weight-level customization |

An endpoint shared with other people needs what any multi-tenant API needs: authentication with scoped API keys, per-key rate limits measured in tokens rather than requests, usage accounting, and a release process that does not break callers when the model changes. Those are built in [API keys and scopes](../../ai-platform-engineering/08-authorization-and-access-control/01-api-keys-and-scopes.md), [Rate limiting](../../ai-platform-engineering/08-authorization-and-access-control/02-rate-limiting.md), the [Gateway](../../ai-platform-engineering/12-gateway/README.md) chapter, and [ModelRelease](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) in *AI Platform Engineering*. One rule specific to models: give each served revision an immutable name (`support-8b@v1.0`) as well as a moving alias (`support-8b`), so that callers who need reproducibility can pin.

## Practical deployments

The rest of the chapter composes into a handful of deployments that cover most real requests. Each is sized from the memory budget and the formulas in [Deploying LLMs on Kubernetes](06-gpu-kubernetes-deployment.md); the throughput figures marked "measured" are the kind of number a load test produces, and stand in for your own.

**1. An internal assistant for many teams.** Twenty teams each fine-tune Llama-3.1-8B on their own documentation and tickets. Ship each as a rank-16 LoRA adapter (84 MB). Serve one base model on H100s with `--enable-lora --max-loras 8 --max-lora-rank 16`, adapters registered at runtime, each team's adapter exposed as its own model name behind the gateway. The 8B base in BF16 leaves about 55 GiB of KV pool per GPU, room for about 55 concurrent 8k-token sequences. If the peak is 15 req/s and a replica measures 5 req/s of goodput with mixed adapters, $\lceil 15 / (5 \times 0.7) \rceil = 5$ single-GPU replicas, against twenty dedicated deployments if each team had merged its own model.

**2. Batch extraction over a document archive.** Ten million documents, about 1,000 input tokens and 200 output tokens each, must become structured JSON overnight. Latency does not matter; throughput per dollar does. Use an 8B model in FP8, constrained decoding against the JSON schema, and the engine's offline batch mode (`vllm run-batch` over an OpenAI batch file, or the `LLM` class in Python) so that the scheduler always has a full batch. Run it as an indexed Kubernetes Job on spot GPUs, one shard of documents per index, with output written per shard so that a preempted pod reruns only its shard. The total is 12 B tokens; at a measured 20,000 tokens per second per GPU, that is $12 \times 10^9 / 20{,}000 / 3600 \approx 167$ GPU-hours, which 24 GPUs finish in about seven hours.

**3. Retrieval-augmented question answering.** Two models with very different shapes: a small embedding model and reranker, and a generator. The embedder is a few hundred million parameters and fits a MIG slice (`nvidia.com/mig-1g.10gb`), several to one H100. The generator sees long prompts (retrieved passages) and short answers, so it is prefill-heavy: TTFT dominates, prefix caching helps only for the shared instructions, and chunked prefill keeps long prompts from stalling other users' decodes. Size its KV pool for the long contexts, not the short outputs. Ingestion, retrieval, and context assembly are built in [Retrieval and RAG](../../ai-platform-engineering/07-retrieval-and-rag/README.md).

**4. A customer-facing chatbot on a 70B model.** This is the runbook's worked deployment: 70B in FP8 on TP=2, five replicas for 20 req/s at a 800 ms TTFT and 50 ms TPOT, two 8-GPU nodes, a prefix-aware router for the shared system prompt, autoscaling on queue depth, and canary releases. At larger scale and with long prompts, split it into a prefill pool and a decode pool, which the worked disaggregation sizing shows can cut the GPU count from 10 to about 6.

**5. A coding assistant with long context.** Prompts carry whole files and repositories, 30,000 tokens and more, and users edit and resend nearly the same prompt every few seconds. Prefix caching is the dominant optimization, so route by prefix affinity to keep a user on the replica that holds their context. Size the KV pool for the long contexts: at 160 KiB per token for a 70B model with FP8 KV, one 32k-token context is about 5 GB, so a TP=2 replica with about 37 GiB of KV pool per GPU holds about fifteen of them. Speculative decoding helps because code is predictable and concurrency per replica is low.

**6. On a laptop or at the edge.** For a single user with no GPU budget, convert to GGUF and quantize with llama.cpp: an 8B model at Q4_K_M is about 4.9 GB and runs on a laptop's CPU or integrated GPU at interactive speed. The quality cost of 4-bit quantization must be measured on your own evaluation, as [Quantization](01-quantization/README.md) warns. [Inference Frameworks](02-frameworks/README.md) covers the runtimes.

## An end-to-end checklist

From a finished training run to other people using the model:

1. **Export** to a supported architecture family, in safetensors, with config, tokenizer, chat template, and generation config.
2. **Verify** logit, template, and engine parity, and run the evaluation suite against the *served* model.
3. **Decide** how to share it: weights, an adapter on a shared base, or an endpoint.
4. **Publish** with a model card, lineage, a license compatible with the base, a pinned tag, and the right access control; re-verify from a clean download.
5. **Size** the deployment: precision and TP from the memory budget, replicas from a load test, GPUs as their product.
6. **Deploy** with weights cached near the GPUs, a startup probe sized to the cold start, autoscaling on queue depth, and a canary for every new revision.
7. **Operate** against the SLO, with immutable model names so that callers can pin, and a rollback that is a path change, not a re-download.

## Where this leads

The loader's side of every file in this section is [Model Loading: Hub to GPU](03-model-loading/README.md), and the fleet that serves the published model is [Deploying LLMs on Kubernetes](06-gpu-kubernetes-deployment.md). LoRA, PiSSA, and merging are built from scratch in [LoRA, PiSSA, and merge](../../build-an-llm/07-objectives/06-lora-pissa-and-merge.md), and the training that produces checkpoints is in [Training and Post-Training](../07-training-and-post-training/README.md). Evaluating a model before you share it is [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/README.md) in *AI Platform Engineering*.

### Further reading

- Hu et al., [*LoRA: Low-Rank Adaptation of Large Language Models*](https://arxiv.org/abs/2106.09685), and Dettmers et al., [*QLoRA*](https://arxiv.org/abs/2305.14314)
- Chen et al., [*Punica*](https://arxiv.org/abs/2310.18547), and Sheng et al., [*S-LoRA*](https://arxiv.org/abs/2311.03285), for serving many adapters
- Hugging Face Hub documentation: [model cards](https://huggingface.co/docs/hub/model-cards), [gated models](https://huggingface.co/docs/hub/models-gated), and the [`hf` command line](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [PEFT documentation](https://huggingface.co/docs/peft), including `merge_and_unload`
- [vLLM: LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html)

## Key terms

- **model type**: the `config.json` field that names the forward-pass code an engine must use; architecture travels as this name, not as code.
- **export**: renaming and re-laying-out a checkpoint's tensors so that a standard implementation of a model family reproduces its outputs.
- **logit parity**: agreement between the outputs of the training code and an independent loader on the same inputs, the test of a correct export.
- **LoRA adapter**: a pair of low-rank matrices per adapted layer whose product is added to the frozen base weight.
- **QLoRA**: LoRA trained against a base stored in 4-bit NF4, cutting training memory to roughly a quarter.
- **multi-LoRA serving**: one base model serving many adapters in the same batch with a batched low-rank kernel.
- **base model relation**: model-card metadata recording whether a model is a fine-tune, adapter, quantization, or merge of its base.

## Summary

- A model's architecture is stored as a name pointing to code that engines maintain, numbers that size it, and named tensors that fit it; staying inside a supported family is what makes a model portable.
- Exporting is renaming plus three layout conventions (RoPE halves versus interleaved, linear orientation, tied embeddings); prove it with logit, template, and engine parity.
- LoRA trains a small low-rank update against a frozen base, cutting fine-tuning memory from about 16 bytes per parameter to about 2; it serves merged for speed or unmerged for many tenants on one base.
- Publishing means a complete repository, a model card with lineage and license, a pinned tag, access control, and re-verification from a clean download.
- Share weights, adapters, or an endpoint depending on who should pay for GPUs and who needs control, then size the deployment with the same memory budget and replica formula.

## Review questions

1. What, concretely, is stored about a model's architecture, and why does a novel architecture make a model hard to share?
   <details><summary>Answer</summary>Only a name (`model_type` and `architectures`) that points to forward-pass code in each engine, the numbers in `config.json` that size that code, and tensors whose names and shapes match it. A novel architecture has no implementation in the engines, so users must run your custom code with `trust_remote_code`, use a slower generic backend, or wait for an upstream implementation.</details>
2. Your exported model matches your training code's argmax at the first few positions, but agreement falls off as the position grows. What is the likely bug?
   <details><summary>Answer</summary>A RoPE layout mismatch. Position 0 has no rotation, so early positions agree, but the rotation grows with position. If the training code rotated interleaved pairs and the target family rotates halves, the `q_proj` and `k_proj` rows must be permuted within each head on export.</details>
3. Compute the memory before activations for full fine-tuning and for LoRA (rank 16, all seven projections) of an 8B model with AdamW.
   <details><summary>Answer</summary>Full fine-tuning stores about 16 bytes per parameter (BF16 weights and gradients, FP32 master weights, two FP32 Adam moments): $8.03 \times 10^9 \times 16 \approx 128$ GB. LoRA stores the frozen base at 2 bytes per parameter, about 16 GB, plus about 42 M adapter parameters at 16 bytes, about 0.7 GB: roughly 17 GB, which fits on one GPU.</details>
4. When should a fine-tune be merged into a full model instead of served as an adapter?
   <details><summary>Answer</summary>When it carries enough traffic to fill its own GPUs and every bit of TPOT matters: merged, it runs at the base model's speed with no extra kernels. Otherwise serve it unmerged on a shared multi-LoRA base, where it costs megabytes instead of a full copy of the weights and loads in seconds.</details>
5. A partner says your published model "never stops talking" in their engine, though it behaved correctly in your evaluation. What do you check first?
   <details><summary>Answer</summary>`generation_config.json`: its `eos_token_id` must include every end-of-turn token in the chat template, not just the end-of-text token. Your evaluation harness may have stopped on the turn token explicitly while the partner's engine relies on the generation config. Then diff the chat template's token ids against your training formatter.</details>
6. Twenty teams each want a fine-tune of the same 8B base. Compare one merged deployment per team with one multi-LoRA fleet.
   <details><summary>Answer</summary>Twenty merged deployments need at least twenty GPUs, most of them idle, and twenty 16 GB copies of the weights. One multi-LoRA fleet holds one base and serves all twenty 84 MB adapters in mixed batches; it needs only as many replicas as the combined traffic requires, five in the worked example, at a modest throughput cost for mixing adapters.</details>
