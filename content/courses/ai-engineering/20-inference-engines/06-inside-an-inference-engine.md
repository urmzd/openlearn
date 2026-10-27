# Inside an Inference Engine: How vLLM and SGLang Run a Model

From the outside, an inference engine looks simple: point it at a model directory, and it answers OpenAI-style requests. Inside, `vllm serve` or `python -m sglang.launch_server` does a fixed sequence of things before the first request and another fixed sequence on every step after. It reads the architecture from `config.json` and builds the model's code, split for the GPUs. It streams the weights from safetensors, giving each GPU only its slice. It measures how much memory is left and turns it into a pool of KV-cache blocks. It compiles and captures the decode step. Then it runs a loop: schedule a batch, run one forward pass across all GPUs, sample, stream tokens back, repeat.

This section walks through both sequences in order, for vLLM and then for what SGLang does differently. The individual mechanisms each have their own section elsewhere: weight mapping and tensor-parallel slicing in [Model Loading](03-model-loading/README.md), the memory budget in [Serving, Capacity & Load Testing](04-serving-and-load/README.md), and the scheduler, block manager, and server built from scratch in Rust in [Serving](../23-model-serving/README.md) in *AI Engineering*. Here they are assembled into one picture of a running engine, so that when an engine is slow to start, runs out of memory, or produces garbage, you know which stage to look at.

> **Note:** Engine internals change from release to release. This walkthrough follows vLLM's V1 architecture (vLLM 0.31) and SGLang's runtime as of October 2026. The stages are stable; class and flag names drift.

> **Learning objectives**
> - Describe the processes a serving engine starts and what each one owns.
> - Trace engine startup from `config.json` to a ready server: building the model per rank, loading and slicing weights, profiling memory, allocating KV blocks, and capturing CUDA graphs.
> - Compute the number of KV-cache blocks an engine will allocate from the GPU's memory and the model's shape.
> - Trace one engine step: scheduling under a token budget, building the ragged batch, the forward pass over paged KV, sampling, and streaming.
> - Explain what SGLang does differently (radix-tree caching and an overlapped scheduler) and when that matters.

## The processes

An engine is several processes, not one, so that slow Python work on the CPU never sits between two GPU steps. vLLM serving a model with tensor parallelism of 2 looks like this:

```
 client ──HTTP──▶ API server process          (FastAPI; tokenize, apply chat template,
                     │   ▲                      detokenize, stream SSE)
                ZMQ  │   │  token ids out
                     ▼   │
                  EngineCore process          (busy loop: scheduler + KV block manager)
                     │  scheduler output, broadcast through shared memory
            ┌────────┴────────┐
            ▼                 ▼
      Worker, rank 0     Worker, rank 1       (one process per GPU: model shard,
       GPU 0              GPU 1                 KV-cache shard, model runner)
            └──── NCCL all-reduce ────┘        (over NVLink, twice per layer)
```

The **API server** owns everything about text: the chat template, tokenization on the way in, and incremental detokenization and streaming on the way out. The **EngineCore** owns all decisions: which requests run this step, how many tokens each gets, and which KV blocks each holds. It never touches a tensor. The **workers**, one per GPU, own all tensors: each holds its slice of the weights and its slice of the KV cache, and runs the forward pass when told. With pipeline parallelism, there are TP × PP workers; across nodes, they are launched through Ray or a multi-node launcher, but the roles are the same.

## Startup, stage by stage

### 1. Read the architecture

The engine reads `config.json` and looks up `architectures[0]`, say `LlamaForCausalLM`, in its **model registry**, which maps the name to the engine's own implementation of that model. It is not transformers' code: it is a rewrite with fused projections, the engine's attention backend, and parallel layers. From the config it takes the shapes: hidden size, layer count, head counts, vocabulary, RoPE settings. It also reads `quantization_config` to pick the linear-layer method, and checks that TP divides the head count. An unknown architecture fails here, or falls back to the slower Transformers backend.

This is the sense in which "the architecture is loaded first": nothing about the architecture is in the weight files. The code comes from the engine, the numbers from the config.

### 2. Start the workers and the communicators

The executor starts one worker process per GPU. Each worker binds to its device, joins a `torch.distributed` process group, and builds the NCCL communicators it will need: one for its TP group, one for its PP group. A worker learns its **rank**, its position in each group, and everything it builds from now on depends on it. A hang at this stage is almost always the network: NCCL cannot reach its peers, `/dev/shm` is too small, or across nodes the wrong network interface was picked.

### 3. Build the model, already split

Each worker constructs the model with its rank already applied. The linear layers are the engine's parallel versions, so they allocate only this rank's share:

| Layer | Class | This rank allocates | Communication in forward |
|-------|-------|---------------------|--------------------------|
| `qkv_proj` (q, k, v fused) | `QKVParallelLinear` | Its query heads and its KV heads | None |
| `o_proj` | `RowParallelLinear` | Its slice of the input columns | All-reduce of partial sums |
| `gate_up_proj` (fused) | `MergedColumnParallelLinear` | Its slice of gate and of up | None |
| `down_proj` | `RowParallelLinear` | Its slice of the input columns | All-reduce |
| `embed_tokens`, `lm_head` | `VocabParallelEmbedding`, `ParallelLMHead` | Its slice of the (padded) vocabulary | All-reduce or gather of logits |

So with TP=2, each of the two workers creates a model whose tensors are half the size, allocated directly on its GPU and left uninitialized; nothing is ever built at full size. With pipeline parallelism, a worker builds only the layers of its stage, and the others are placeholders. A layer is the Megatron pattern: a column-parallel matmul splits the output with no communication, the row-parallel matmul that follows produces partial sums, and one all-reduce adds them. That is two all-reduces per transformer layer, one after attention and one after the MLP.

### 4. Stream and slice the weights

Now the weights. Each worker iterates over the safetensors shards: memory-mapped, so on one node the files are read from disk once and shared through the page cache. For every checkpoint tensor it:

1. **Renames** it to the engine's parameter: `q_proj`, `k_proj`, and `v_proj` all map to `qkv_proj`, with a shard id saying which part.
2. **Calls the parameter's `weight_loader`**, which cuts out this rank's rows or columns and copies them into the right offset of the (possibly fused) parameter. A column-parallel weight $W \in \mathbb{R}^{\text{out} \times \text{in}}$ keeps rows $[r \cdot \text{out}/t, (r+1) \cdot \text{out}/t)$; a row-parallel one keeps the same range of columns.
3. **Discards** the rest. The other rank keeps the other half of the same tensor.

Afterwards, a post-processing pass prepares quantized weights for their kernels (repacking for Marlin, for example). [Model Loading](03-model-loading/02-downloads-loaders-and-sharding.md#how-serving-engines-load) shows the mapping and slicing code. When this stage is the slow one, the bottleneck is the storage the files come from, which [Deploying LLMs on Kubernetes](07-gpu-kubernetes-deployment/README.md) addresses.

### 5. Profile memory and allocate the KV cache

With weights resident, the engine works out how much memory is left for the KV cache by measuring, not guessing. It runs a **profiling forward pass** with dummy input at the largest batch it will ever schedule (`max_num_batched_tokens`) and records the peak memory: weights, the largest activations, and the logits tensor. Then

$$\text{KV memory per GPU} = \text{util} \times M_\text{gpu} - \text{peak during profiling}$$

where `util` is `--gpu-memory-utilization` (vLLM) or `--mem-fraction-static` (SGLang). It divides that into **blocks** of a fixed number of tokens (16 by default in vLLM):

$$\text{blocks} = \frac{\text{KV memory per GPU}}{\text{block size} \times \text{KV}_\text{tok per GPU}}, \qquad \text{KV}_\text{tok per GPU} = \frac{2 \times L \times n_\text{kv} \times d_\text{head} \times b}{t}$$

For Llama-3.1-8B in BF16 on one 80 GiB H100 at `util = 0.90`, the KV memory is about 55 GiB and each token takes $2 \times 32 \times 8 \times 128 \times 2 = 128$ KiB, so a block of 16 tokens is 2 MiB and the engine allocates about 28,000 blocks, room for about 450,000 tokens across all requests. vLLM prints this at startup as the KV cache size in tokens and the maximum concurrency at `max_model_len`. Every rank allocates the same number of blocks, each holding its own KV heads, so one block id means the same tokens on every GPU, and the scheduler can manage them all with one table.

If this number is too small for even one sequence of `max_model_len`, startup fails here with an error asking you to raise `util` or lower `max_model_len`. That is the memory budget from [Serving, Capacity & Load Testing](04-serving-and-load/02-memory-budget-and-slos.md#the-memory-budget) enforced by the engine.

### 6. Compile and capture

The last stage removes overhead from the decode step. A decode step for a small batch does little GPU work, so the hundreds of kernel launches Python issues per step would dominate it. The engine therefore **captures CUDA graphs**: it runs the forward pass once for each of a list of batch sizes (1, 2, 4, 8, up to a maximum) and records the sequence of kernels, so that a later step of that size replays the whole graph in one launch. A batch of 13 is padded to the next captured size, 16. vLLM also runs `torch.compile` first, which fuses elementwise operations into fewer kernels and caches the compiled result on disk for the next start.

This stage typically takes 20 to 60 seconds and a gigabyte or two of memory, and it is the floor of cold start once the weights are local. `--enforce-eager` skips it, which starts faster and decodes slower: useful for debugging, wrong for production.

Then the API server opens its port and `/health` starts returning success.

### Startup at a glance

| Stage | What happens | Typical failure |
|-------|--------------|-----------------|
| 1. Architecture | Registry lookup, config parsing, quantization method | Unsupported architecture; TP does not divide heads |
| 2. Workers | One process per GPU, NCCL groups | Hang: network, `/dev/shm`, wrong interface |
| 3. Build | Parallel layers sized for this rank | Out of memory: weights do not fit at this TP |
| 4. Load | Rename, slice per rank, copy | Missing or unexpected keys; slow storage |
| 5. Profile and KV | Peak measured, rest becomes blocks | "Not enough KV cache for max_model_len" |
| 6. Compile and capture | `torch.compile`, CUDA graphs per batch size | Slow start; out of memory during capture |

## One step of the loop

Once the engine is running, the EngineCore runs a loop, one iteration per forward pass. Follow one iteration.

**Schedule.** The scheduler has a **token budget** per step (`max_num_batched_tokens`, say 4,096) and a cap on running sequences (`max_num_seqs`). vLLM V1 does not have separate prefill and decode phases in the scheduler: every request simply has a number of tokens computed so far and a number it still needs. Running requests that are decoding take one token each. Then waiting requests are admitted, each given as many prompt tokens as the remaining budget allows; a long prompt that does not fit is split across steps, which is chunked prefill. Before admitting tokens, the scheduler asks the KV block manager for blocks, first matching the prompt's full blocks against the prefix cache by hash, so cached tokens are neither recomputed nor stored twice. If no blocks are free, the lowest-priority running request is **preempted**: its blocks are freed and it is recomputed later.

**Prepare the batch.** The scheduler's decision is broadcast to every worker. Each builds the same inputs: all scheduled tokens from all requests concatenated into one flat array (no padding), their positions, each request's **block table** (which physical blocks hold its earlier tokens), and a **slot mapping** (which slot of which block each new token's K and V will be written to).

**Forward.** Every worker runs its shard of the model on that flat batch. In each layer, attention first writes the new tokens' K and V into the paged cache at their slots, then attends over each request's cached tokens by following its block table. The attention backend (FlashAttention or FlashInfer) handles prompt chunks and single decode tokens in the same kernel launch. After attention and after the MLP, the two all-reduces combine the ranks' partial results. Only the last position of each request produces logits, since only it is sampled.

**Sample.** The logits for each request go through its own logit processors and sampling parameters: penalties, temperature, top-k and top-p, grammar masks for structured output. Then one token is chosen per request.

**Update and stream.** The EngineCore appends each new token, checks stop conditions (an end-of-turn id, a stop string, `max_tokens`), and frees the blocks of finished requests. New token ids go back to the API server, which detokenizes incrementally and sends them out as server-sent events. Then the loop begins the next step.

Put as code, the engine core is short. The difficulty is all inside the calls:

```python
while True:
    batch = scheduler.schedule()          # token budget, admission, blocks, preemption
    if batch.empty():
        wait_for_requests(); continue
    out = executor.execute_model(batch)   # broadcast; every rank runs its shard
    finished = scheduler.update(batch, out.sampled_tokens)   # append, stop checks, free blocks
    send_to_api_server(out.sampled_tokens, finished)         # detokenize and stream elsewhere
```

You build each of these pieces in *AI Engineering*: the [model runner and sampler](../23-model-serving/02-model-runner-and-sampler.md), the [continuous-batching scheduler](../23-model-serving/03-continuous-batching.md), [chunked prefill](../23-model-serving/04-chunked-prefill.md), the [block manager and prefix cache](../23-model-serving/05-block-manager-and-prefix-cache.md), and the [OpenAI server](../23-model-serving/06-openai-server-on-tokio.md).

## What SGLang does differently

SGLang has the same stages and the same loop; its differences are in three places.

**The processes.** SGLang splits text handling from scheduling the same way, into a **tokenizer manager**, one **scheduler** process per GPU rank (each scheduler drives its own model worker, and the schedulers make the same decisions in lockstep), and a **detokenizer manager**. The memory flag is `--mem-fraction-static`, and the KV pool is sized by the same profile-then-fill logic.

**The cache is a radix tree.** Where vLLM hashes full blocks to find shared prefixes, SGLang's **RadixAttention** keeps every cached sequence in a radix tree keyed by token ids, with a page size of one token by default. Any shared prefix of any length is found by walking the tree, and finished requests leave their KV in the tree, evicted least recently used, so the next request with the same history (a multi-turn chat, an agent loop, a few-shot template, many branches of one prompt) starts where the last one left off. The scheduler can also order waiting requests to maximize prefix hits. [Radix prefix cache](../21-inference/04-radix-prefix-cache.md) builds this structure.

**The scheduler overlaps the GPU.** SGLang's default scheduler prepares step $n+1$ on the CPU while the GPU runs step $n$, so that scheduling, block allocation, and input preparation cost no GPU time. This matters most for small models and small batches, where a decode step takes a few milliseconds and CPU overhead would otherwise be a large fraction of it.

Beyond these, SGLang leans on a few features you will see in its flags: grammar-constrained decoding through XGrammar, **DP attention** (`--enable-dp-attention`) for MLA models such as DeepSeek where tensor-parallel attention would replicate the latent cache, and expert parallelism for MoE.

| | vLLM | SGLang |
|---|---|---|
| Processes | API server, EngineCore, one worker per GPU | Tokenizer manager, one scheduler per GPU, detokenizer |
| Prefix cache | Hashes of full blocks (16 tokens) | Radix tree over tokens; LRU eviction |
| Scheduling | Unified token budget; chunked prefill by default | Overlapped CPU scheduling; prefix-aware ordering |
| KV memory flag | `--gpu-memory-utilization` | `--mem-fraction-static` |
| Strongest when | Broad model and hardware coverage; general traffic | Heavy prefix reuse: multi-turn chat, agents, structured generation |

Both are fast for most workloads, and they adopt each other's ideas quickly. Choose by measuring your own traffic with the load-testing method in [Serving, Capacity & Load Testing](04-serving-and-load/README.md).

## Pitfalls

- **Reading GPU memory as a leak.** The engine claims `util` of the GPU at startup and keeps it; `nvidia-smi` showing 90% used on an idle server is the KV pool, by design.
- **Two engines on one GPU at default `util`.** Each claims 90% of the whole GPU; the second fails at stage 5.
- **Setting `max_model_len` to the model maximum.** A 128k-token limit forces the profile and the minimum-KV check to plan for 128k, shrinking or failing the pool; set it to the real need.
- **Benchmarking with `--enforce-eager`.** Without CUDA graphs, small-batch decode is several times slower than production.
- **Expecting bit-identical output across batch sizes.** The same prompt in a different batch can take a different reduction order and a different CUDA graph size; tiny logit differences can flip a sampled token.

## Where this leads

The stages of startup are the cold-start terms in [Model Loading](03-model-loading/03-cold-start-and-tokenizers.md#cold-start-math), and the step loop is the step-time equation in [Serving, Capacity & Load Testing](04-serving-and-load/README.md). Running the engine as a fleet is [Deploying LLMs on Kubernetes](07-gpu-kubernetes-deployment/README.md), and producing a checkpoint that passes stages 1 and 4 cleanly is [From Training Run to Shared Model](08-train-publish-share.md). To build every component yourself, follow [Serving](../23-model-serving/README.md) in *AI Engineering*.

### Further reading

- Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (vLLM)
- Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104) (RadixAttention)
- Shoeybi et al., [*Megatron-LM*](https://arxiv.org/abs/1909.08053), for column- and row-parallel layers
- The [vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview.html) and the [SGLang documentation](https://docs.sglang.ai/)

## Key terms

- **model registry**: the engine's map from an `architectures` name to its own implementation of that model.
- **rank**: a worker's position in its tensor- or pipeline-parallel group, which decides which slice of every weight it holds.
- **column- and row-parallel linear**: splitting a weight by output rows (no communication) or by input columns (partial sums combined by all-reduce).
- **profiling pass**: a dummy forward pass at the largest batch, run at startup to measure peak memory before sizing the KV cache.
- **block table**: a request's list of the physical KV blocks holding its tokens, used by attention to read the paged cache.
- **CUDA graph capture**: recording a forward pass's kernel sequence once per batch size so that each later step replays it with one launch.
- **token budget**: the maximum number of tokens, prompt and generated, the scheduler puts into one step.

## Summary

- An engine is an API server for text, an engine core for decisions, and one worker per GPU for tensors.
- Startup is six stages: read the architecture, start the workers, build the model already split by rank, stream and slice the weights, profile memory and fill the rest with KV blocks, and capture the decode step.
- Each step schedules tokens under a budget, runs one flat batch over the paged cache on every GPU with two all-reduces per layer, samples, and streams.
- SGLang shares the design and differs in its radix-tree prefix cache and overlapped scheduler, which pay off when prefixes are heavily reused and steps are short.

## Review questions

1. Where does an engine get a model's architecture, given that the safetensors files hold only named tensors?
   <details><summary>Answer</summary>From two places: `config.json`'s `architectures` field names the implementation in the engine's model registry (the code), and the config's shape fields instantiate it (the numbers). The weight files only fill in the parameters that this code creates.</details>
2. With TP=2, does each worker build the full model and then discard half?
   <details><summary>Answer</summary>No. Each worker constructs the model from parallel layers that allocate only its rank's share, and during loading each checkpoint tensor is sliced by the parameter's `weight_loader` before the copy, so no GPU ever holds a full-size tensor.</details>
3. An engine for Llama-3.1-8B in BF16 on an 80 GiB GPU at `util = 0.90` has about 55 GiB left after profiling. How many 16-token KV blocks does it allocate, and how many tokens do they hold?
   <details><summary>Answer</summary>A token's KV is $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ bytes, 128 KiB, so a 16-token block is 2 MiB. 55 GiB divided by 2 MiB is about 28,000 blocks, about 450,000 tokens shared among all requests.</details>
4. Why does an engine capture CUDA graphs, and what does `--enforce-eager` cost?
   <details><summary>Answer</summary>A small-batch decode step does little GPU work, so launching hundreds of kernels from Python dominates its time; a captured graph replays the step in one launch. `--enforce-eager` skips the capture, so startup is faster but small-batch decode is several times slower.</details>
5. In one vLLM step, a 6,000-token prompt arrives while 100 requests are decoding, with a 4,096-token budget. What does the scheduler do?
   <details><summary>Answer</summary>The 100 decoding requests take one token each, 100 tokens. The remaining 3,996 tokens of budget go to the first chunk of the new prompt; the remaining 2,004 prompt tokens are scheduled in the next step. That is chunked prefill, which keeps the decoding requests' step time bounded.</details>
6. When would you expect SGLang's radix cache to beat block hashing, and why?
   <details><summary>Answer</summary>When requests share long prefixes that are not aligned to block boundaries, or share history that grows turn by turn: multi-turn chat, agent loops, many samples from one prompt. The radix tree matches any shared prefix to the token and keeps finished requests' KV for reuse, while full-block hashing only matches whole blocks.</details>
