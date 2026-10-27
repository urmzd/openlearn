# Runtime controls and inference flags

A flag selects a mechanism in a particular runtime. Learn the mechanism, its resource tradeoff, and its correctness test first; then map it to the option names in the installed engine. Defaults and support vary by release, hardware, and model, so there is no universal command that optimizes every workload.

> **Learning objectives**
> - Distinguish request-level sampling controls from engine-level resource controls.
> - Map a performance symptom to a bounded experiment.
> - Record resolved configuration and evaluate interacting flags.

## Request controls change the work or the output distribution

| Generic control | What it does | What to measure |
|---|---|---|
| Maximum generated tokens, stop conditions | Bound output work and define completion termination | Truncation, task success, generated-length distribution |
| Temperature, top-k, top-p, min-p | Transform next-token probabilities | Quality/diversity and rollout behavior probabilities |
| Repetition/frequency penalties | Modify logits based on history | Repetition and unintended suppression |
| Number of candidates, best-of-N | Spend more generation and scoring compute per answer | Success per request and total compute, not only selected-answer quality |
| Grammar or schema constraints | Mask invalid token continuations | Schema validity, semantic correctness, constraint overhead |
| Seed and random-stream policy | Control assignment of random draws | Replay behavior under changing batch order |
| Log probabilities and returned alternatives | Expose extra scoring/output data | Memory, transfer, serialization, and scoring semantics |

Reducing output length is often an effective system optimization, but comparing 32 generated tokens with 512 does not establish a faster engine. Freeze the workload when benchmarking execution. A sampler change also changes the behavior policy for RL.

## Scheduling, memory, and caching

| Generic engine control | Mechanism | Benefit and limiting tradeoff | Required check |
|---|---|---|---|
| Active-sequence limit | Bound sequences scheduled concurrently | More weight reuse; longer per-user step time and more KV | Goodput, ITL, preemptions, queue time |
| Per-step token budget | Bound scheduled prefill and decode work | Larger prefill throughput; longer interference with decode | TTFT and ITL together on mixed traffic |
| Chunked prefill and chunk size | Interleave portions of long prompts with decodes | Shorter decode stalls; repeated scheduling and possible prefill slowdown | Long/short prompt fairness and numerical equivalence |
| Context limit | Bound accepted prompt plus output | Lower worst-case memory; less supported context | Explicit input rejection or truncation semantics |
| Cache memory budget | Reserve KV capacity after weights/workspaces | Higher concurrency; less safety margin | Peak memory during warmup, graph capture, and long-context traffic |
| Cache block size | Trade allocation granularity against bookkeeping | Smaller tail waste; more metadata and kernel constraints | Resident blocks, cache hit rate, useful capacity |
| Prefix-cache reuse and eviction | Reuse exact compatible prefix state | Less repeated prefill; cached prefixes occupy space | Namespace correctness, hit/miss behavior, cold versus warm runs |
| Preemption/recompute/swap | Reclaim state when capacity is exhausted | Avoid immediate failure; recompute or transfer cost | Preemption rate, tail latency, forward progress |
| Admission limits and priority | Bound queued/running work and choose ordering | Protect SLOs; reject or defer excess work | Rejection rate, starvation, cancellation cleanup |

An engine's “batch size” may mean sequences, tokens in an iteration, or static tensor slots. These are not interchangeable. At 32 active sequences, a step may contain 32 decode tokens, thousands of prefill tokens, or a mixture. Ask which resource the control actually bounds.

## Execution, representation, and placement

| Generic control | Mechanism | Tradeoff to validate |
|---|---|---|
| Weight/activation dtype | BF16/FP16/FP8 or integer arithmetic/storage | Device support, scaling, numerical error, quality |
| Weight quantization method | Compress weights with a matching kernel/layout | Smaller memory versus dequantization cost and quality |
| KV dtype and scales | Compress retained attention state | Context-dependent error and supported attention backend |
| Attention backend | Choose tiled/fused kernels and layout | Shapes, head sizes, masks, hardware, invariance |
| Compilation and fused kernels | Reduce launches and fuse operations | Compilation time, shape specialization, temporary memory |
| Device graph capture | Replay a prepared launch sequence | Capture/warmup cost, fixed-shape buffers, reserved memory |
| Tensor/pipeline/expert parallel degrees | Split matrices/layers/experts | Fit and throughput versus collective traffic and bubbles |
| Replica count | Serve independent request batches | Aggregate capacity and redundancy versus duplicated weights |
| CPU/NVMe offload | Move weights or state out of device memory | Fit versus per-step transfer latency and host bandwidth |
| Speculative draft and draft length | Verify several proposed tokens per target call | Acceptance, draft cost, verification cost, KV rollback |
| Deterministic/batch-invariant execution | Restrict algorithms to a replay contract | Supported shapes and topologies, memory, and throughput cost |

Quantization, graphs, and kernel selection interact: a faster format may lack a compatible graph or attention path. A larger cache fraction can leave too little graph-capture memory. A speculative draft can compete with the target for memory and bandwidth. Test combinations after isolated measurements reveal which controls matter.

## Work one tuning decision

Suppose short interactive requests meet an ITL target when served alone, but long prompts cause stalls. Start with a mixed prompt distribution at a fixed offered rate. Record queue time, prefill duration, ITL, active tokens, and memory. Sweep the per-step token budget with chunked prefill enabled. A smaller budget may reduce decode stalls while increasing long-prompt TTFT. Choose a point meeting both SLOs, then repeat the [batch-invariance checks](13-batch-invariance-and-reproducibility.md).

If memory is the limiting factor instead, scheduling alone cannot shrink the model. Compute weights, KV, and temporary buffers separately. Compare fewer active sequences, shorter supported context, a supported KV/weight representation, or a different placement. Use quality tests for any representation change. The [capacity parts](04-serving-and-load/README.md) provide the arithmetic.

## A versioned mapping example

This table maps a few portable concepts to **vLLM 0.11.0**, an archived teaching reference. It does not describe every provider or recommend that release for production. Option names are checked against its [serve reference](https://docs.vllm.ai/en/v0.11.0/cli/serve.html); its [tuning guide](https://docs.vllm.ai/en/v0.11.0/configuration/optimization.html) explains the scheduling tradeoffs. Runtime-selected defaults should be read from the resolved configuration.

| Concept | Example option in that release |
|---|---|
| Active sequence limit | `--max-num-seqs` |
| Per-step token budget | `--max-num-batched-tokens` |
| Chunked prefill | `--enable-chunked-prefill` |
| Maximum model context | `--max-model-len` |
| GPU memory fraction | `--gpu-memory-utilization` |
| Prefix reuse | `--enable-prefix-caching` |
| Tensor/pipeline placement | `--tensor-parallel-size`, `--pipeline-parallel-size` |
| Weight/KV representation | `--dtype`, `--quantization`, `--kv-cache-dtype` |
| Disable graph execution for comparison | `--enforce-eager` |
| Draft/verification configuration | `--speculative-config` |

Inspect the installed runtime's version, help, startup logs, and support matrix before translating any row into a command. Provider APIs may expose only context/output limits or a service tier; hidden provider optimizations cannot be inferred from a fast response.

SGLang uses different names and sometimes different budget semantics. These mappings were checked against its rolling [server-argument reference](https://docs.sglang.io/docs/advanced_features/server_arguments) on 2026-10-26; pin the installed release and inspect its help before using them.

| Generic concept | SGLang example | Semantic detail |
|---|---|---|
| Active sequences | `--max-running-requests` | Running requests rather than all queued arrivals |
| Prefill chunk budget | `--chunked-prefill-size` | A chunk limit, not necessarily the same total iteration budget as another runtime |
| Static memory allocation | `--mem-fraction-static` | Includes weights and KV pool; not an interchangeable percentage with every other engine |
| Disable prefix reuse | `--disable-radix-cache` | A disable flag, opposite polarity to an enable-prefix flag |
| Tensor/pipeline layout | `--tp-size`, `--pp-size` | Cooperating ranks rather than independent replicas |
| Graph backend by phase | `--cuda-graph-backend-decode`, `--cuda-graph-backend-prefill` | Rolling docs list `--disable-cuda-graph` as deprecated |
| Batch-invariant operation mode | `--enable-deterministic-inference` | Read the supported backend/model restrictions and run the invariance matrix |

The [SGLang deterministic-inference guide](https://docs.sglang.io/docs/advanced_features/deterministic_inference) scopes its controls. Equivalent concepts across vLLM and SGLang do not imply equal defaults, identical kernels, or bitwise-equal outputs.

## Record what actually ran

Save model and adapter hashes, tokenizer/template revision, runtime/container version, driver and device topology, resolved engine settings, environment variables affecting execution, generation defaults, and workload seed. Configuration files, command-line flags, model generation defaults, and request overrides can have different precedence. Capture the final values used by the request rather than assuming the checked-in configuration won.

## Key terms

- **Token budget**: the amount of prefill/decode work admitted into a scheduling iteration.
- **Graph capture**: recording a device execution graph for later replay with supported shapes and buffers.
- **Resolved configuration**: the values actually selected after defaults and overrides are applied.

## Summary

Every control should connect a bottleneck to a mechanism, a tradeoff, and a correctness check. Provider flag names are mappings onto those concepts.

## Review questions

1. Why can a larger token budget improve aggregate throughput while violating an interactive SLO?

   <details><summary>Answer</summary>It permits more work in one step and amortizes overhead, but can make each step or prefill interruption longer.</details>
2. Why should cache and graph settings be measured together after separate experiments?

   <details><summary>Answer</summary>Both reserve memory. Each can work alone but fail or trigger preemption when combined.</details>
