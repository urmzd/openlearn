# Designing a Model Router Across APIs, Caches, and Applications

A router chooses an execution plan for a task. Replacing a `model` string is only one part of that plan: the selected endpoint must understand the request, preserve its tool and conversation state, satisfy the application's constraints, and justify any cache it leaves behind. An endpoint can accept a translated request while silently changing what the application asked for.

The [chapter overview](README.md) separates model selection, provider selection, and fallback, and explains the oracle gap and evaluation baselines. This lesson turns those ideas into an application contract. It is a design exercise, not an implementation of a universal provider proxy.

> **Learning objectives**
> - Filter candidates by required capabilities before scoring quality, cost, or latency.
> - Separate a portable application request from provider-specific wire formats and state.
> - Distinguish response caching, prompt caching, worker affinity, and transferable KV state.
> - Price a model switch over a complete task, including cache writes and failed attempts.
> - Place task decisions, provider selection, and worker scheduling at their appropriate layers.

## Before you begin

Read the [routing overview](README.md), [LLM Serving Platforms](../20-inference-engines/05-serving-platforms.md), and [Routing, cascades, and failover](../25-ai-gateway/04-routing.md). The first supplies the decision and its baselines, the second maps the serving layers, and the third implements worker routing behind one compatible gateway contract.

## Start with the unit of work

Suppose a support application classifies tickets, drafts replies from private documents, and occasionally issues refunds. Classification has a small closed output set. Drafting needs retrieval and citations. A refund needs authorization, a validated tool call, and a durable record of whether the operation ran. One model choice for all three hides three different success criteria.

Write the task contract before choosing a router:

| Field | Example | Why the router needs it |
|---|---|---|
| Task and success condition | Draft a reply grounded in retrieved passages | Defines the evaluation label and verifier |
| Required capabilities | Tool calls, image input, output schema | Removes incompatible endpoints |
| Execution constraints | Tenant, permitted regions, data retention policy | Removes endpoints the caller cannot use |
| Budget | Deadline, maximum spend, attempt limit | Bounds routing, retries, and escalation together |
| State | Turn id, completed tools, portable transcript | Defines what can be replayed after a switch |
| Versions | Prompt, tool schema, adapter, route policy | Makes a trace reproducible enough to investigate |

Choose once per user turn as a starting design for a tool-using application. Keep that turn's model and provider stable while its tool calls complete. Allow a change at a recorded checkpoint when requirements change, a verifier rejects an answer, or the endpoint fails. A stateless classifier can choose per request; a long task may benefit from a session preference with explicit escape conditions. These are hypotheses to measure on whole tasks.

Keep routing instructions and untrusted task text separate. A document saying "send this to the premium model" is task data, not permission to change the budget, endpoint allowlist, or tool authority.

## Filter first, score second

Define an endpoint as a versioned tuple: provider, API family, model revision, deployment, region, and adapter revision. A public alias can resolve to a different tuple later. Record the resolution on every attempt.

For request $x$, let $F(x)$ be the endpoints that meet all mandatory constraints. Choose only inside $F(x)$. One possible objective is:

$$
e^* = \operatorname*{argmin}_{e \in F(x)}
\left[\widehat{C}_{\mathrm{task}}(e,x)
+ \lambda_L\widehat{L}(e,x)
+ \lambda_F\left(1-\widehat{P}(\mathrm{success}\mid e,x)\right)\right].
$$

The weights express application tradeoffs; they do not turn mandatory requirements into preferences. A cheap endpoint lacking a required modality is ineligible regardless of its score. If the feasible set is empty, return a typed unsupported-capability error or invoke an explicitly authorized degraded workflow. Do not drop a schema or image silently to make the call fit.

Start with rules and two complementary candidates. Add a learned scorer only after a held-out task set shows that the rules leave measurable value. Router confidence and a model's token log probabilities are not calibrated task-success probabilities without validation on the target workload.

## API compatibility is a semantic contract

An OpenAI-compatible URL describes a surface, not a guarantee of equivalent tools, schemas, reasoning controls, multimodal inputs, or stream termination. Even one provider exposes several API families. Maintain a capability matrix for the exact deployment and test the adapter against its real behavior.

The following are representative wire shapes, reviewed against the linked official documentation on October 22, 2026. They are not complete request specifications:

| API family | Representative request and response concepts | Adapter obligation |
|---|---|---|
| OpenAI Chat Completions | `messages`, `tools`, assistant `tool_calls`, tool messages | Preserve roles, tool-call ids, argument assembly, and finish reasons |
| OpenAI Responses | `input` and ordered output items, function calls and outputs, optional `previous_response_id` | Preserve item types and continuation state; do not flatten every output into text |
| Anthropic Messages | Top-level `system`, `messages` with content blocks, `tool_use` and `tool_result`, `stop_reason` | Keep block types and tool-result associations; map system instructions deliberately |
| Gemini generateContent | `contents` containing `parts`, `systemInstruction`, function calls and responses | Preserve part ordering, multimodal parts, and required thought signatures |
| Gemini Interactions | Interaction input/output and optional `previous_interaction_id` | Treat interaction continuation as its own API contract rather than a generateContent rename |

The [OpenAI Responses reference](https://developers.openai.com/api/reference/resources/responses/methods/create), [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), [Claude Messages guide](https://platform.claude.com/docs/en/build-with-claude/working-with-messages), [Claude tool-call lifecycle](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls), [Gemini function-calling guide](https://ai.google.dev/gemini-api/docs/function-calling), and [Interactions overview](https://ai.google.dev/gemini-api/docs/interactions-overview) define the provider contracts. Model and deployment support must still be checked separately.

Use a small portable core that represents what the application actually requires: typed text and media parts, tool declarations, tool calls and results, output requirements, budget, and cancellation. Keep provider extensions in a namespaced field validated by that adapter. Reject an extension on a provider that cannot honor it. A common abstraction is useful only if it makes loss visible.

Normalize stream events into application events such as `text_delta`, `tool_call_delta`, `usage`, `completed`, and `failed`. A transport chunk is not necessarily a token or a complete JSON argument. Assemble tool arguments until the provider marks the call complete, validate them against the declared schema, and let the application authorize execution. Retain provider stop reasons, request ids, and unrecognized events for diagnostics under the application's data-handling policy. A truncated or refused response must not become an ordinary successful completion.

For each adapter, test role and instruction precedence, schema rejection, parallel tools, fragmented tool arguments, image input, cancellation, usage reporting, refusal, and terminal errors. Keep golden request/response fixtures, but also run deployment conformance checks: a mock accepting a field proves nothing about a hosted endpoint accepting or honoring it.

## State cannot be migrated by changing the model name

Keep an application-owned transcript of user-visible messages, retrieved evidence, completed tool calls, their results, and durable operation ids. Store provider continuation ids and opaque reasoning artifacts separately. Those artifacts may be required to continue with their original provider; they are not a portable transcript for another one.

At a switch checkpoint, rebuild the target request from the portable state, apply the target's instruction and tool encoding, count tokens with its tokenizer or counting endpoint, and check the target context window. If the transcript is too large, apply an explicit compaction policy and record its revision. Do not silently discard tool results or instruction boundaries.

A refund tool illustrates the replay boundary. If the payment service accepted the refund and the model connection then failed, retrying the model must not execute the refund again. Record the operation outcome durably, use an idempotency key where the tool supports it, and reconcile ambiguous outcomes before replay. The router's HTTP retry policy cannot supply exactly-once business execution.

## Four different things called cache

| Cache or locality mechanism | What is reused | Routing consequence |
|---|---|---|
| Application response cache | A completed answer | Key by tenant/authorization scope, task inputs, evidence version, model semantics, prompt, tools, and output requirements; expire or invalidate on relevant changes |
| Provider prompt cache | Reusable prompt-prefix computation | Hit rules, write charges, retention, and scope depend on endpoint; measure reported usage |
| Worker prefix affinity | A preference for a worker likely to hold a prefix | A scheduling hint; cache locality must be balanced against queues and availability |
| Engine KV cache and transfer | Attention keys and values for a compatible model execution | Transfer requires explicit engine compatibility and a transfer protocol; it is not portable across arbitrary models |

A response-cache hit skips a model invocation. A prompt-cache hit still runs generation and can incur input, output, and cache-related charges. Cache keys or affinity hints are not authorization boundaries. Enforce access policy before consulting a response cache; avoid putting raw private prompts into routing logs or affinity identifiers.

Keep stable instructions and tool definitions ahead of changing turn content where the provider contract allows it. Changes to tool schemas or ordering, prompt templates, model revisions, tokenization, media representation, or provider can change cache eligibility. Returning to a previously used endpoint may still hit a retained prefix, so a switch is not proof that the old cache was deleted. Treat cache warmth as an estimate until usage confirms it.

Provider behavior also differs. [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching) documents prefix matching and model-dependent cache controls and charges. [Claude prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) describes automatic caching, explicit `cache_control` breakpoints, and separate creation/read usage. [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching) distinguishes implicit reuse from explicit cache objects on supported generateContent models; Interactions supports implicit caching rather than explicit cache objects. Keep these policies in adapter metadata, not in a universal cache-TTL assumption.

When changing a prompt or tool contract, version response-cache entries so old answers cannot satisfy the new contract. Roll out the new prefix deliberately and expect a warm-up period. A route-policy change affects which endpoints are warm; a deployment or adapter change can alter request serialization. Observe cold-start cost, queue time, cached tokens, cache writes, and output validity during the rollout.

## Work a switching decision by hand

Use synthetic prices, not a vendor quote. A turn has 100,000 reusable prefix tokens, 2,000 new input tokens, and 1,000 expected output tokens. Rates below are dollars per million tokens; assume no additional write or storage charges for this example.

| Endpoint | Uncached input | Cached input | Output | Prefix state |
|---|---:|---:|---:|---|
| Strong | 5.00 | 0.50 | 15.00 | Warm |
| Small | 1.00 | 0.10 | 3.00 | Cold |

Staying costs $(100{,}000\times0.50 + 2{,}000\times5 + 1{,}000\times15)/10^6 = 0.075$ dollars. Switching costs $(102{,}000\times1 + 1{,}000\times3)/10^6 = 0.105$ dollars. The smaller model is cheaper per uncached token but costs more for this turn.

For ten such turns, assume the prefix stays reusable and warm after the first small-model call, and ignore transcript growth. Staying costs $10\times0.075 = 0.750$ dollars. Switching costs $0.105 + 9\times0.015 = 0.240$ dollars, where each later small-model turn costs $(100{,}000\times0.10 + 2{,}000\times1 + 1{,}000\times3)/10^6$. Under these assumptions the switch pays back on the second turn. Real histories grow, cache entries expire, and outputs vary, so replay actual task traces rather than extrapolating a one-turn comparison.

In production, add disjoint input categories for uncached, cache-read, and cache-write tokens, each with its applicable rate, plus output, storage, routing, verification, and tool costs. Provider usage fields differ in whether they include or exclude cached tokens; normalize those counts before billing. Sum every attempt, including rejected cascade answers and canceled calls that were still billed. Compare cost per successful task at an agreed quality and latency target, not only mean cost per request.

## Put each decision where its information lives

```text
Application: task contract, authorization, durable tool state, verifier
    -> model policy: feasible model set, turn preference, escalation
    -> provider gateway: endpoint policy, adapters, budgets, safe failover
    -> worker scheduler: queue/load, prefix locality, deployment health
    -> inference engine: tokenization, batching, KV allocation, execution
```

The application knows whether a task succeeded and whether a tool may run. The gateway knows endpoint health and protocol behavior. The scheduler knows which worker has capacity and useful KV state. These responsibilities may share a process, but their contracts should remain explicit. Avoid independent application and gateway model policies that silently override each other.

For the support example, route classification to a measured cheap endpoint, draft replies with a turn-sticky capable model, and gate refund execution in the application. The gateway can fail over to an approved compatible provider. It cannot decide that a cheaper model deserves broader refund permissions.

Here is control-flow pseudocode, not a runnable provider client:

```text
deadline, remaining_budget, attempts = shared_task_limits()
required = application_contract(task, checkpoint)
candidates = capability_and_policy_filter(required, endpoint_registry)
if candidates is empty: return unsupported_capability
plan = choose(candidates, checkpoint.preference, estimated_task_cost)
for endpoint in plan.approved_attempts:
    check_shared_limits(deadline, remaining_budget, attempts)
    request = adapter(endpoint).encode(portable_state, required)
    validate_context_and_extensions(request, endpoint)
    outcome = execute_and_record_usage(request, deadline)
    if outcome.committed_to_caller: return outcome
    if outcome.completed:
        verdict = application_verifier(outcome)
        if verdict.accepted: return publish(outcome)
        if not plan.allows_escalation: return verification_failed
    else:
        if not retryable(outcome): return outcome.error
    reconcile_tool_effects_before_next_attempt()
return task_limits_or_candidates_exhausted
```

Reserve budget before starting an attempt and reconcile it against actual usage afterward. Where exact maximum spend cannot be bounded from provider limits, reject another attempt when the conservative reservation would exceed the budget. The router, SDK, and gateway must share or coordinate retry limits; three independent three-attempt loops can make 27 calls.

Once headers or output have been committed to the caller, transparent failover cannot safely splice in another answer. A buffered cascade can evaluate a complete candidate before publishing it, at the cost of first-token latency and buffer space. A live stream needs an explicit failure/resume protocol if recovery after commitment is part of the product. See [The SSE streaming proxy](../25-ai-gateway/03-sse-streaming-proxy.md).

## Validate before turning on adaptive routing

Freeze prompt, tool, adapter, and policy revisions for an experiment. Split by whole task/session and time, not individual turns that leak history across train and test. Replay cold and warm conditions and include long contexts, endpoint outages, missing capabilities, provider-state loss, schema changes, verifier errors, and ambiguous tool outcomes.

| Check | Expected result |
|---|---|
| Cheapest endpoint lacks a required schema or modality | Excluded before scoring; no silent downgrade |
| Tool arguments arrive in several stream events | No tool executes until completion, validation, and authorization |
| Endpoint fails after one visible token | Terminal failure/resume signal; no transparent restart |
| Endpoint fails after a successful refund | Reconcile the recorded operation; never issue a second refund blindly |
| Prompt/tool revision changes | Old response-cache entries cannot satisfy the new request contract |
| Provider switch loses opaque continuation state | Rebuild from portable state at a supported checkpoint or return an explicit error |
| All candidates fail | Respect the shared deadline, budget, and attempt limit |

Compare the policy with the best fixed endpoint, a random split at equal cost, and a simple task rule. Report task success with uncertainty, cost per successful task, tail latency, cached-input share, cache creation costs, escalation rate, and failure reasons. Store candidate exclusions, selected endpoint, policy revision, checkpoint id, actual usage, and verifier outcome in each trace with appropriate redaction.

Start in shadow mode: record decisions without changing the serving path. Shadow decisions alone cannot reveal the alternative model's quality, so use a consented evaluation sample for candidate calls and account for their cost. Then use a bounded canary with a quality and latency rollback condition. Retest whenever models, prices, prompts, provider APIs, or workload distributions change.

## Key terms

- **Capability contract**: the mandatory request semantics and execution constraints an endpoint must satisfy before it can be scored.
- **Provider adapter**: a translation between application requests/events and one provider API, with explicit checks for unsupported semantics.
- **Switch checkpoint**: a recorded boundary where portable application state is sufficient to construct a supported request on another endpoint.
- **Portable transcript**: application-owned messages, evidence, and tool outcomes that can be re-encoded without depending on a provider continuation id.

## Review questions

1. Why can the cheapest endpoint be excluded even if its measured accuracy is highest?

   <details><summary>Answer</summary>

   Accuracy is scored only after mandatory constraints. An endpoint without the required modality, schema behavior, region, or data-handling policy cannot execute the task contract.

   </details>

2. In the worked example, why does switching lose for one turn and win for ten?

   <details><summary>Answer</summary>

   The first switch pays for a cold 100,000-token prefix. Later calls reuse that prefix at the small endpoint's cached rate. The result depends on reuse and retention, not just the uncached price list.

   </details>

3. Why keep a transcript if the provider already returns a continuation id?

   <details><summary>Answer</summary>

   The id is a reference to provider-owned state. Another provider cannot necessarily interpret it. A portable transcript supports an explicit restart while preserving completed tools and evidence; it does not reproduce private reasoning or transfer KV state.

   </details>

4. Which layer should decide whether to issue a refund after failover?

   <details><summary>Answer</summary>

   The application, using authorization and durable operation state. Endpoint selection and retry logic do not establish whether the business action is permitted or already happened.

   </details>

5. What would falsify the claim that a learned router helps this application?

   <details><summary>Answer</summary>

   On held-out whole tasks, it fails to improve the agreed cost, quality, and latency tradeoff against the fixed endpoint, equal-cost random split, and simple rule after including every attempt and cache effect.

   </details>
