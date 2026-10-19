# LLM Serving Platforms: How They Work and When to Use Them

Between an application that wants an answer and the GPU that produces a token sit several layers, and the names attached to them, Anyscale, Ray, vLLM, SGLang, RouteLLM, OpenRouter, are easy to blur together. Some are inference engines, some orchestrate clusters, some choose which model answers, and some are organizations rather than software at all. Choosing well starts with knowing which layer each one occupies, because adding a component that solves a problem you do not have only adds cost, latency, and another thing to debug.

This section is a practical guide to those layers. It explains how each platform works, what problem it is good for, and what it costs you, then turns to two questions that cut across all of them: whether results are reproducible, and what public evidence of production use actually establishes. The use cases and trade-off judgments are engineering guidance, not benchmark guarantees; primary sources were reviewed on September 12, 2026.

You should have read the chapter opening, [LLM Systems & Inference](README.md), for the engine-level ideas (continuous batching, the KV cache, prefix caching) that this section assumes. [Inference Frameworks](02-frameworks/README.md) covers the wider engine landscape across C, Rust, and Zig, and [Serving, Capacity & Load Testing](04-serving-and-load/README.md) covers how to size and benchmark an engine once you have chosen one.

> **Learning objectives**
> - Place each of Anyscale, Ray, vLLM, SGLang, RouteLLM, and OpenRouter in the correct layer of the serving stack.
> - Explain how each works, when it is worth using, and its main trade-off.
> - Distinguish provider selection, model fallbacks, and automatic model selection in a gateway, and explain how each affects caching and evaluation.
> - Explain why LLM inference can be nondeterministic and what batch-invariant execution costs.
> - Read production-adoption evidence for what it does and does not establish.

## Who is who?

Start with the cast. The table below names each organization or project, its role, and the primary reference to read:

| Organization or project | Role | Primary reference |
|---|---|---|
| **LMSYS** (Large Model Systems) | Nonprofit incubating open-source AI research and systems; originated in a multi-university collaboration. Known for SGLang, FastChat, Vicuna, and the original Chatbot Arena project, which has graduated. | [About LMSYS](https://www.lmsys.org/about/) |
| **Anyscale** | Commercial platform built on Ray for distributed AI workloads, including training, data processing, and inference. Collaborated on RouteLLM research. | [Anyscale](https://www.anyscale.com/), [RouteLLM collaboration](https://github.com/lm-sys/RouteLLM#motivation) |
| **Ray** | Open-source distributed-computing framework; Ray Serve adds serving and application orchestration. | [Ray documentation](https://docs.ray.io/en/latest/) |
| **RouteLLM** | Framework for learning, serving, and evaluating routing policies between strong and weak models. | [Repository](https://github.com/lm-sys/RouteLLM) |
| **OpenRouter** | Hosted API gateway and model marketplace; selects provider endpoints and offers optional model routing and fallbacks. | [Quickstart](https://openrouter.ai/docs/quickstart) |
| **vLLM** | Inference engine that executes models, batches requests, and manages serving memory. | [Documentation](https://docs.vllm.ai/en/stable/) |
| **SGLang** | Open-source inference framework with RadixAttention prefix reuse, scheduling, and structured generation support. | [Documentation](https://docs.sglang.io/), [paper](https://arxiv.org/abs/2312.07104) |
| **Thinking Machines Lab** | AI research company whose inference study explains batch-dependent numerical variation and demonstrates reproducible execution. | [Inference study](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) |

## Start with the layer you need

With the names sorted out, the more useful way to approach them is by question. Each layer answers a different one:

| Question | Layer | Examples |
|---|---|---|
| Which model should answer? | Model selection | RouteLLM, OpenRouter Auto Router |
| Which endpoint should receive the call? | Gateway and provider selection | OpenRouter |
| How are Python tasks, model replicas, and cluster resources coordinated? | Distributed execution and serving orchestration | Ray Core, Ray Data, Ray Serve |
| Who provisions, operates, and observes those clusters? | Managed platform | Anyscale |
| How do weights and requests become tokens efficiently? | Inference engine | vLLM, SGLang, TensorRT-LLM, llama.cpp |

LMSYS and Thinking Machines are organizations producing research and software, not additional request-processing layers.

To see how the layers compose, consider two request paths. A possible self-hosted deployment is:

```text
Application -> Ray Serve endpoint -> model replica -> vLLM -> GPU -> tokens

Anyscale control plane manages cluster provisioning, scaling, and rollouts.
It is not another model that must inspect each prompt.
```

A hosted API path is:

```text
Application -> OpenRouter -> selected provider endpoint -> provider's engine
```

The provider chooses its own engine; this diagram does not imply OpenRouter's providers all use vLLM. Neither path requires a learned model router. The sections that follow walk the self-hosted path from the managed platform (Anyscale) through orchestration (Ray) to the engines (vLLM, SGLang, and others), and then turn to model selection and gateways.

## Anyscale: managed operations for Ray workloads

**How it works.** You configure a container environment and compute resources, then develop in a workspace or submit a job or service. Anyscale manages Ray clusters and provides its optimized runtime, dashboards, and developer tools. Jobs suit finite work such as batch inference; services expose Ray Serve applications as production endpoints. [Platform overview](https://docs.anyscale.com/get-started/what-is-anyscale)

Its control plane coordinates provisioning and cluster state; the data plane runs workloads in customer cloud accounts or Kubernetes infrastructure. Anyscale's architecture separates workload execution from management, although management metadata, logs, and metrics are part of the platform relationship. You still configure cloud access and your application. [Architecture](https://docs.anyscale.com/get-started/architecture)

**Why use it?** Suppose a team extracts information from millions of documents, creates embeddings, and serves a custom model. Different stages need different CPU/GPU resources. Managed scheduling, observability, and cluster lifecycle reduce the infrastructure the team must operate. Online services add managed rollouts and recovery; open-source KubeRay also supports Ray deployments and rolling upgrades, so those capabilities are not exclusive to Anyscale. [Services and KubeRay comparison](https://docs.anyscale.com/services)

**Trade-off.** You pay platform charges in addition to the resources your workload consumes and depend on platform configuration and operations. This is attractive when the saved engineering effort and utilization gains justify it. A single occasional API call or one stable GPU server rarely needs a distributed platform merely to run inference. Ray applications can also run without Anyscale; moving platform-specific deployment configuration requires separate work.

## Ray: distribute work and operate model replicas

**How it works.** Ray Core turns functions into remote tasks and classes into stateful actors. The scheduler places work according to resource requirements; object references connect dependent computations. A GPU actor can keep a model loaded and process many inputs rather than reloading weights for each call. [Ray concepts](https://docs.ray.io/en/latest/ray-core/key-concepts.html)

Ray Data provides data-processing pipelines, Ray Train coordinates training, and Ray Serve operates serving deployments. These libraries share distributed-computing foundations but solve different parts of the application. [Ray overview](https://docs.ray.io/en/latest/ray-overview/index.html)

**Why use it?** A document pipeline can distribute parsing across CPU workers and embedding generation across GPU workers. A service can scale model replicas while leaving token-level scheduling to the inference engine inside each replica.

**Trade-off.** Serialization, data movement, scheduling, and failure handling introduce overhead. Parallelizing tiny tasks can cost more than it saves. Scaling replicas across machines also differs from splitting one large model across GPUs; an engine's tensor/expert parallelism handles the latter. Choose Ray when distributed work is the problem, not because every model call needs a cluster.

## vLLM: keep model execution and serving memory efficient

**How it works.** vLLM loads model weights, schedules requests, and generates tokens. Paged KV-cache storage avoids reserving a large contiguous allocation for every growing sequence. Continuous batching admits and removes requests as generation proceeds rather than waiting for a fixed batch to finish. [vLLM documentation](https://docs.vllm.ai/en/stable/), [paged attention design](https://docs.vllm.ai/en/stable/design/paged_attention/)

Automatic prefix caching reuses previously computed attention state for matching prefixes. For example, many questions about the same long document can avoid repeating its prefill work. It does not eliminate generation of each answer's new tokens. [Prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)

**Why use it?** You need to run supported open-weight or fine-tuned models under your control, with enough concurrent traffic to benefit from batching and memory management.

**Trade-off.** You own GPU capacity, model compatibility, upgrades, and performance tuning. More concurrency can improve aggregate throughput while increasing queueing delay. Low utilization can make a self-hosted endpoint more expensive than paying for external requests. Benchmark your actual model, hardware, prompt lengths, and latency target.

## SGLang: reuse prefixes across structured generation workloads

**How it works.** SGLang is an inference framework with a runtime and a programming interface. Its original design introduced RadixAttention: a tree-organized KV cache that reuses shared token prefixes across generation calls. The paper also describes structured-output decoding optimizations. Modern deployments can use its model server through compatible APIs without adopting the original programming interface. [SGLang paper](https://arxiv.org/abs/2312.07104), [current documentation](https://docs.sglang.io/)

**Why use it?** Consider many requests that share a long instruction prefix, a branching generation workflow, or a model/hardware combination for which SGLang's runtime performs well. It is a serving-engine candidate alongside vLLM, not a replacement for cluster management.

**Trade-off.** Cache reuse depends on actual shared prefixes, available memory, and scheduling. Different prompts do not become reusable merely because they discuss the same subject. vLLM also supports prefix reuse; RadixAttention is a mechanism to study, not proof that one engine always wins. Compare both under the same quality constraints and workload.

## Other engines you will encounter

vLLM and SGLang are not the only engines. Three others appear often enough that you should know how each works, why you might use it, and its main trade-off:

| Engine | How it works | Why use it? | Main trade-off |
|---|---|---|---|
| [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html) | NVIDIA-oriented optimized kernels and runtimes, with PyTorch and TensorRT execution paths. | Tune supported models for an NVIDIA deployment. | Hardware/backend coupling and version-specific support; not every path requires the same build workflow. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++ inference, GGUF model files, CPU/GPU backends, and an optional server. | Local inference, quantized models, or deployment on varied consumer hardware. | Model fit and performance remain device-dependent; do not assume datacenter throughput from a laptop result. |
| [TGI](https://huggingface.co/docs/text-generation-inference/en/index) | Hugging Face's model-serving toolkit with batching, streaming, and multiple backends. | Understand or maintain an existing deployment. | Official documentation lists maintenance mode at review time; evaluate maintenance and model support before a new deployment. |

The [cross-language framework guide](02-frameworks/README.md) covers the wider C, Rust, and Zig ecosystem.

Everything so far assumes you already know which model should answer. The next two layers make that decision, or route around it.

## RouteLLM: decide when a stronger model is worth a call

**How it works.** A trained router estimates a prompt-dependent preference score. Above a configured threshold it selects the strong model; below it selects the weak model. It predicts which model to call before obtaining the answer, rather than calling both and judging their outputs. You configure the candidate pair and can deploy the routing framework yourself. [Project documentation](https://github.com/lm-sys/RouteLLM)

**Why use it?** A workload contains many requests the cheaper model handles well and a smaller set where the stronger model materially improves results. A measured routing policy may spend less than always using the stronger model.

**Trade-off.** Routing errors, classifier overhead, and distribution shift can erase savings. Preference scores are not correctness guarantees. Include cache misses and retries in cost, and verify complete agent trajectories when early decisions change later inputs.

### Read the evidence and the implementation

Before relying on RouteLLM, read what its sources actually show, and how the implementation behaves on inputs its training did not anticipate:

- [RouteLLM paper](https://arxiv.org/abs/2406.18665): preference-trained routing and evaluation across model pairs.
- [LMSYS research introduction](https://www.lmsys.org/blog/2024-07-01-routellm/): benchmark cost/quality trade-offs and the importance of relevant training data. Arena-only routers performed near randomly on MMLU before augmentation. Historical benchmark savings are not production guarantees.
- [Controller source](https://github.com/lm-sys/RouteLLM/blob/main/routellm/controller.py): the reviewed implementation scores the last message, then forwards the conversation to the selected model. Its comment notes first-turn training. Study what happens when the last message is merely “continue” or a tool result.
- [Matrix-factorization source](https://github.com/lm-sys/RouteLLM/blob/main/routellm/routers/matrix_factorization/model.py): prompt embedding plus learned model representations produces a preference score. The embedding request adds a network dependency.
- [Prompt-caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and [model-switch behavior](https://code.claude.com/docs/en/prompt-caching#switching-models): cache hits, writes, expiry, and model changes affect the actual bill.

The right way to evaluate a router follows from those caveats. Compare a pinned model against a fixed router on complete held-out sessions. Measure task success, cost per successful task, latency, cache usage, and retries. Include the router version, threshold, candidate models, prompts, tools, and fallback policy in the evaluated configuration. Automatic selection is testable; an untracked policy change invalidates the comparison.

## OpenRouter: gateway, provider selection, and model routing

[OpenRouter](https://openrouter.ai/docs/quickstart) provides a common API for accessing models from multiple providers. An application sends its request to OpenRouter, which forwards it to a selected provider endpoint and returns the response. This is a hosted intermediary, whereas RouteLLM is a routing framework you can run yourself and vLLM is an inference engine.

OpenRouter makes three distinct decisions, and it is important not to conflate them:

| Feature | What changes? | Primary reference |
|---|---|---|
| Provider selection | Which endpoint serves the requested model. Default routing considers price and availability; provider preferences can constrain or order endpoints. | [Provider selection](https://openrouter.ai/docs/guides/routing/provider-selection) |
| Model fallbacks | Which model runs after a failure, using an ordered list supplied by the caller. | [Model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks) |
| Auto Router | Which model is selected for the request when the caller uses `openrouter/auto`. Allowed models can be constrained. | [Auto Router](https://openrouter.ai/docs/guides/routing/routers/auto-router) |

Using OpenRouter with an explicit model does not inherently mean using automatic model selection. Conversely, pinning only the model name does not pin its serving provider. Provider controls include an allowlist, explicit order, disabled fallbacks, required parameter support, and quantization filters. Constraining these reduces deployment variation but does not guarantee deterministic kernels or identical outputs.

**Why use it?** You want access to several hosted models through a common integration instead of operating their GPU servers or maintaining separate provider adapters. You can select a fixed model and use provider failover without enabling automatic model selection.

**Caching matters here:** OpenRouter documents [provider stickiness](https://openrouter.ai/docs/guides/best-practices/prompt-caching) to preserve cache reuse, with explicit `session_id` support. Router models can also reuse the resolved model on a best-effort basis while it remains eligible. This is not a hard model pin, and a provider failure can cause fallback. Do not assume that all gateways ignore cache economics or that stickiness guarantees cache hits.

Each benefit of a gateway comes with a corresponding trade-off:

| Benefit | Corresponding trade-off |
|---|---|
| One integration for multiple providers | Another service dependency and request-processing hop |
| Provider fallback for availability | The execution environment and cache location can change |
| Optional model selection | The policy, candidate set, and actual selected model must be included in evaluation |
| Central routing and usage management | Total cost still includes provider usage, applicable platform fees, cache behavior, and retries |

Check the current [FAQ and billing terms](https://openrouter.ai/docs/faq) before comparing costs. For reproducible evaluation, record the resolved model and provider, constrain allowed routes, and compare complete sessions rather than assuming a single model slug describes the entire deployed system.

## Reproducibility: mechanism, availability, and incentives

Gateways and fallbacks raise a question that applies to every layer: if you send the same request twice, do you get the same answer? Often you do not, even at temperature zero, and the reason is numerical rather than random. Three sources describe the mechanism and what engines currently offer:

- [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), September 2025: batch shape can alter reduction order and floating-point results. Batch-invariant kernels produced 1,000 identical completions in the reported test. The workload took 26 seconds by default, 55 seconds with unoptimized deterministic kernels, and 42 seconds with improved attention. These imply approximately 53% and 38% lower throughput for that setup, not universal penalties.
- [vLLM batch invariance](https://docs.vllm.ai/en/stable/features/batch_invariance/): opt-in configuration with documented hardware/model constraints and beta status at review time. Availability does not establish deployment-wide adoption or a customer-facing API guarantee.

- [SGLang deterministic inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference): `--enable-deterministic-inference` is disabled by default. The backend compatibility table matters; support for radix caching differs across attention backends. Seeded non-greedy sampling can still provide reproducible samples.

Separate the mathematical forward pass, deliberate sampling, and numerical execution. A fixed seed does not repair changing logits. Deterministic arithmetic still permits diverse outputs through different random draws.

**Economic interpretation:** a throughput penalty is an immediate operating cost. Reproducibility becomes easier to justify when debugging savings, training stability, or a paid guarantee offset that cost. This is a workload-dependent incentive argument, not evidence that all providers reject determinism.

## Production evidence and adoption limits

Claims that an engine "runs in production everywhere" are common, so it is worth reading the public evidence carefully. Each source below establishes something specific, and each leaves something unestablished:

| Source | What it establishes | What it does not establish |
|---|---|---|
| [LinkedIn engineering, August 2025](https://www.linkedin.com/blog/engineering/ai/how-we-leveraged-vllm-to-power-our-genai-applications) | vLLM supported 50+ GenAI use cases across thousands of hosts. | How many deployments enable deterministic execution. |
| [Roblox engineering, September 2024](https://about.roblox.com/newsroom/2024/09/running-ai-inference-at-scale-in-the-hybrid-cloud) | vLLM was its primary LLM engine; reported volume was approximately 4 billion tokens weekly. | Current traffic or industry-wide market share. |
| [vLLM project report, July 2026](https://vllm.ai/blog/2026-07-16-keeping-vllm-production-quality) | Project-reported 5.6M+ monthly pip installs and 2.5M+ monthly image pulls. | Unique people, organizations, or production deployments. |

The cited RouteLLM sources establish research results and a serving implementation, not comparable named production scale. Lack of such evidence is not proof of non-use.

## Choosing a layer before a vendor

The discipline this section argues for is to choose a layer before choosing a vendor. Try it on a concrete case. Take a support assistant with a shared policy document, a custom model, and an overnight embedding job. Draw its online request path and offline data path separately. Mark where the model is chosen, where replicas are scheduled, where KV state lives, and who pays for idle resources. Then remove any component that does not solve a measured requirement.

Compare a fixed model, a fixed model with provider fallback, and a learned model router on the same sessions. Record task success, total cost, time to first token, per-token latency, retries, and cache hits. Repeat with cold and warm caches. Treat provider-published benchmark results as hypotheses to test on your workload.

## Where this leads

Within this chapter, [Inference Frameworks](02-frameworks/README.md) maps the engine layer across languages, [Model Loading: Hub to GPU](03-model-loading/README.md) explains how engines get weights onto GPUs, and [Serving, Capacity & Load Testing](04-serving-and-load/README.md) shows how to size and benchmark a chosen engine. The routing and evaluation questions raised here are developed in *AI Platform Engineering*: [Model Routing & Cascades](../../ai-platform-engineering/11-model-routing-and-cascades/README.md) covers router families, oracle versus deployed routers, escalation policy, and decision models; [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/README.md) covers evaluating complete sessions; and [Training and Frameworks](../../ai-platform-engineering/01-training-and-frameworks/README.md) covers the training side of the same stack.

## Key terms

- **inference engine**: software that loads weights, schedules requests, manages the KV cache, and generates tokens (vLLM, SGLang, TensorRT-LLM, llama.cpp).
- **serving orchestration**: coordinating model replicas and cluster resources across machines, as Ray Serve does, separate from token-level scheduling inside each engine.
- **managed platform**: a service, such as Anyscale, that provisions, operates, and observes clusters on your behalf.
- **model router**: a policy, such as RouteLLM's, that predicts before generation whether a stronger or weaker model should answer a request.
- **gateway**: a hosted intermediary, such as OpenRouter, that forwards requests to a selected provider endpoint behind one API.
- **provider stickiness**: a gateway's best-effort reuse of the same provider for a session to preserve prompt-cache hits.
- **batch invariance**: the property that a request's numerical results do not depend on what else is in its batch.

## Summary

- Each platform occupies one layer: model selection (RouteLLM, OpenRouter Auto Router), gateway and provider selection (OpenRouter), distributed orchestration (Ray), managed operations (Anyscale), and the inference engine (vLLM, SGLang, and others).
- Every layer has a cost; add one only when it solves a measured requirement, and choose the layer before the vendor.
- Routers and gateways change what actually runs, so evaluations must record the resolved model, provider, policy, and cache behavior and compare complete sessions.
- Batch shape can change floating-point reduction order and therefore outputs; batch-invariant kernels restore reproducibility at a workload-dependent throughput cost.
- Public adoption evidence establishes specific facts, such as install counts or one company's usage, not market share or feature adoption.

## Review questions

1. In the self-hosted path `Application -> Ray Serve endpoint -> model replica -> vLLM -> GPU`, which component does token-level batching, and what does Anyscale contribute?
   <details><summary>Answer</summary>vLLM, inside each replica, does token-level scheduling with continuous batching and paged KV-cache management. Ray Serve scales and routes among replicas. Anyscale's control plane manages cluster provisioning, scaling, and rollouts; it is not another model inspecting each prompt.</details>
2. Does calling OpenRouter with an explicit model slug pin the full deployed system? What should you record for reproducible evaluation?
   <details><summary>Answer</summary>No. Pinning the model name does not pin its serving provider, and provider fallback can change the execution environment and cache location. Using an explicit model also does not mean automatic model selection is in use. Record the resolved model and provider, constrain allowed routes, and compare complete sessions.</details>
3. Why can a learned router fail to save money even when its benchmark results look good?
   <details><summary>Answer</summary>Routing errors, classifier overhead (including the embedding request it may make), and distribution shift can erase the savings, and preference scores are not correctness guarantees. Cache misses caused by switching models, retries, and agent trajectories in which early decisions change later inputs must all be counted. The reviewed RouteLLM controller scores only the last message, which may be just "continue" or a tool result.</details>
4. Why can the same prompt produce different outputs at temperature zero, and what does fixing it cost?
   <details><summary>Answer</summary>Batch shape can change the reduction order of floating-point operations and therefore the logits, so results depend on what else was batched with the request; a fixed seed does not repair changing logits. In the Thinking Machines test, batch-invariant kernels gave 1,000 identical completions, with the workload taking 26 s by default, 55 s with unoptimized deterministic kernels, and 42 s with improved attention, roughly 53% and 38% lower throughput for that setup.</details>
5. What does vLLM's report of 5.6M+ monthly pip installs establish, and what does it not?
   <details><summary>Answer</summary>It establishes project-reported download volume. It does not establish the number of unique people, organizations, or production deployments.</details>
