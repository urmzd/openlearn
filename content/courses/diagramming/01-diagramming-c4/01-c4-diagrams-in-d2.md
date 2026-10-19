# C4 diagrams in D2

By the time an LLM platform has a gateway, an inference engine, a durable control plane, workers, training tools, persistent files, and a telemetry collector, the architecture is too large to hold in one view. A new maintainer needs a way in: first the users and external systems, then the processes that run, then the inside of the one process they will actually change. A C4 set gives them exactly that path.

This section draws that set for the platform built in *Build Your Own LLM System*. You will produce three D2 files, a context view, a container view, and a component view of the Rust engine, each rendered to SVG next to its source. The model answers live in [`docs/c4/context.d2`](../../../../course/solutions/docs/c4/context.d2), [`docs/c4/containers.d2`](../../../../course/solutions/docs/c4/containers.d2), and [`docs/c4/components-engine.d2`](../../../../course/solutions/docs/c4/components-engine.d2), rendered as [`context.svg`](../../../../course/solutions/docs/c4/context.svg), [`containers.svg`](../../../../course/solutions/docs/c4/containers.svg), and [`components-engine.svg`](../../../../course/solutions/docs/c4/components-engine.svg). The diagrams are an independent artifact: no code calls them, but every reviewer of the system reads them.

This section builds on the chapter opening, [Diagramming with the C4 Model](README.md), and leans on one decision record, [ADR 0001](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md), which is explained in [Architecture decision records](../02-documentation-writing/01-architecture-decision-records.md). Before you read each view below, sketch it yourself from the request trace in "Tracing one request by hand"; comparing your sketch with the solution is where most of the learning happens.

> **Learning objectives**
> - Decide, for any box, which C4 level it belongs to by asking what unit that level is about.
> - Label every arrow with a protocol, subprocess, or file format, and explain why an unlabeled arrow signals a missing interface or a box at the wrong level.
> - Write context, container, and component views in D2 that compile with `d2` and keep the same system boundary across all three.
> - Justify the design compromises in the solution: three views rather than four, one component view, and a partly open context box.

## What each level is for

Different readers ask different questions. A stakeholder asks "who uses it and what does it depend on?", an operator asks "which processes run and how do they talk?", and an engine maintainer asks "what is inside the engine?" One diagram cannot answer all three legibly, so there are three diagrams, and each one keeps to its own question.

That gives three rules for placing things. Keep the context diagram about people and external dependencies. Show deployable processes at container level. Put internal services and stores in the component view. The test for any box is whether it is the unit its level is about. At container level that unit is "something you deploy, or a store you manage": the gateway, engine, durable service, worker, training tools, files, and collector. A scheduler is not deployed on its own, so it belongs in the component view.

Arrows follow a rule too: every arrow names a protocol or a file boundary. The languages of the platform meet only at process and file boundaries ([ADR 0001](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md)), so every container-level arrow is either a network protocol (HTTPS, HTTP, gRPC, OTLP), a subprocess, or a file format. An arrow without such a label is either a missing interface or a box drawn at the wrong level. Labeling makes the diagram checkable against the contracts.

Finally, the diagrams are code. D2 files are diffable and compile in CI, so a renamed service breaks the build of its diagram instead of silently going stale. Render each one with `d2 docs/c4/context.d2 docs/c4/context.svg` (and likewise for the others); a diagram that no longer compiles is a failing test.

## Tracing one request by hand

The best way to find the boxes is to follow one completion request from the CLI through the gateway, the engine, artifact storage, and telemetry, drawing one view per abstraction level. As you go, check that the same system boundary is preserved and that the engine's subcomponents appear only in the component view.

Here is the trace. The client sends HTTPS to the gateway, a hop that appears in both the context and the container views. The gateway sends HTTP inference to the engine (container view). Inside the engine, the HTTP layer hands the request to the scheduler, which tokenizes, runs prefill and decode, and allocates KV blocks; all of that is component-only. The engine reads safetensors and tokenizer files from storage (container view). The gateway and the engine export OTLP to the collector (container view). Every hop names its protocol or file format, and nothing inside the engine appears above the component view.

## The three files

The deliverable is three D2 files under `docs/c4/`, each rendered to SVG next to it:

- `context.d2`: people, the platform, and external systems.
- `containers.d2`: each deployable process and managed store.
- `components-engine.d2`: the inside of the Rust engine.

Each file's first line states its scope, so no reader has to guess which question a view answers.

## Building the views step by step

### The context view

The context view puts people and external systems around one platform box.

```d2
# System context: people and external systems only.
client: Client application
operator: Platform operator
platform: TinyLLM platform {
  gateway: API gateway
  engine: Inference engine
  durable: Durable control plane
  artifacts: Model and corpus files
  telemetry: Telemetry collector
}
provider: External model or data provider

# the public API
client -> platform.gateway: HTTPS / OpenAI subset
# a second actor, a second API
operator -> platform.gateway: HTTPS / admin API
operator -> platform.durable: gRPC / workflow operations
platform.gateway -> platform.engine: HTTP inference, gRPC control
# ... engine and durable to artifacts (files); gateway, engine, durable to telemetry (OTLP)
# the one outbound dependency
platform.gateway -> provider: HTTPS provider adapter
```

The two actors are distinguished because they use different APIs (`openai-subset` versus `admin`) with different trust, and the provider sits outside the boundary because the platform does not own it. The platform box shows its five major parts rather than a single opaque box. That is a deliberate compromise: the actors' arrows land on the part they actually talk to. The parts carry no technology labels here, because technology belongs to the container view.

### The container view

Zooming in one level, each box becomes a deployable process or a managed external system.

```d2
# Container view: each box is a deployable process or a managed external system.
client: Client process
subgraph tinyllm: TinyLLM platform {
  gateway: Go gateway\nHTTP API, auth, routing
  engine: Rust engine\nHTTP/SSE, scheduler, KV blocks
  durable: Go durable service\nworkflows, queues, history
  worker: Go worker\nactivity subprocesses
  corpus: Python corpus and training tools\nCLI + subprocess
  storage: Persistent files\nweights, tokenizer, corpus, WAL
  collector: OTel collector
}
provider: Model provider

client -> gateway: HTTPS
# two protocols: data plane and control plane
 gateway -> engine: HTTP inference\n gRPC control
worker -> durable: gRPC
corpus -> durable: gRPC workflow client
# the Go-to-Python boundary is a process
worker -> corpus: subprocess + files
# file formats are interfaces too
engine -> storage: safetensors, tokenizer, KV envelopes
corpus -> storage: corpus shards, tokens, checkpoints
 durable -> storage: WAL and snapshots
 gateway -> collector: OTLP
 engine -> collector: OTLP
 durable -> collector: OTLP
 gateway -> provider: HTTPS
```

Each box names its language and responsibilities, because at this level "which process, in which language" is the question. The worker and the training tools, which were folded into one "durable control plane" in the context view, split into the processes that actually run. Every cross-language arrow is a protocol, a subprocess, or a file format, as ADR 0001 requires. Notice the `worker -> corpus` arrow in particular: the worker is Go and the training tools are Python, so the label "subprocess + files" names the boundary where that contract lives. Durable state (the WAL and snapshots) and telemetry also appear here, which makes the recovery paths visible to an operator.

### The component view

Only the engine is opened, because it is the container most work on the platform happens inside.

```d2
# Component view: internal to the Rust engine container.
http: HTTP and SSE API
scheduler: Request scheduler\ncontinuous batching
# ... registry client, tokenizer, prefill, decode, KV block manager, prefix cache, weights loader, telemetry

http -> scheduler: request, cancel, stream
scheduler -> registry: register, heartbeat
scheduler -> tokenizer: encode and decode
scheduler -> prefill: prompt batches
scheduler -> decode: token steps
prefill -> kv: allocate and write blocks
decode -> kv: read and append blocks
scheduler -> prefix: lookup and insert
weights -> prefill: model parameters
weights -> decode: model parameters
http -> telemetry: request and stream outcomes
scheduler -> telemetry: queue, batch, token metrics
```

Inside one process the arrows are function calls, so they are labeled with what flows (batches, blocks, parameters) rather than with a protocol. The scheduler is the hub, and that is the point the view makes: continuous batching means every request, prefill, and decode step passes through it. Prefill and decode both reach the KV manager, but with different verbs (write versus read and append), which is exactly the distinction a KV bug report needs.

The full diagrams are in [`docs/c4/`](../../../../course/solutions/docs/c4/).

### Why this design

Each choice in the solution trades something away, and it is worth being explicit about what.

D2 beats a drawing tool because text diffs in review, compiles in CI, and lays itself out; the cost is less control over placement. The set has three views, not four, because C4's code level is left to the source, and a fourth diagram would go stale fastest. There is only one component view because opening every container would triple the maintenance for readers who mostly work in the engine; the other containers' internals are documented in their own chapters. And the context box is partly open. Purists keep the context system opaque, but showing the parts makes the actors' entry points visible at a glance. The price is that the context view must change whenever a major part is added.

## What can go wrong

A reviewer checks the set against four requirements, each of which protects something downstream.

| Requirement | How it is checked | Why it matters |
|---|---|---|
| the three declared paths exist and compile with `d2` | artifact | a plausible narrative cannot stand in for a reviewable diagram |
| every arrow is labeled with a protocol or file boundary | review | readers can find the interface (OpenAPI, gRPC, safetensors) behind each line |
| the system boundary is the same in all three views | review | a box that moves across the boundary between views is a contradiction |
| engine internals appear only in the component view | review | each level answers one question; mixing levels hides both answers |

Three pitfalls account for most failed reviews of an architecture artifact. The first is naming a decision without the evidence behind it, so reviewers cannot verify the claim against the repository; the solution avoids it because every arrow names a protocol or file format that can be checked against the contracts. The second is leaving out the failure or rollback case, so the happy path looks complete while an operator has no safe response under failure; the solution puts durable state and telemetry on the container view so recovery paths are visible. The third is leaving the scope implicit, so two reviewers reach different conclusions from the same picture; the solution states each file's scope (context, container, engine components) on its first line.

> **Warning:** Do not let engine internals creep into the container view "just to be helpful." A scheduler box next to the gateway makes the container view answer two questions at once, and readers can no longer tell what is deployed.

## Where this leads

The boundary rule every arrow follows comes from [Architecture decision records](../02-documentation-writing/01-architecture-decision-records.md), where ADR 0001 is written and annotated. The diagrams are also an input to [Design review: the whole system](../../systems/01-system-design/03-design-review-whole-system.md) in *Systems and Architecture*, which uses them to verify the system's contracts and evidence.

Further reading:

- [ADR 0001: languages meet at process and file boundaries](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md). Compare the container view's arrows with the boundaries the ADR allows.
- Simon Brown, [The C4 model](https://c4model.com/). The four levels and the notation rules the three views follow.

## Key terms

- **Context view**: the C4 level-1 diagram showing one system, the people who use it, and the external systems it depends on.
- **Container view**: the C4 level-2 diagram in which each box is a separately deployable process or managed store, and each arrow is a protocol or file boundary.
- **Component view**: the C4 level-3 diagram of one container's internals, whose arrows are in-process calls labeled with what flows.
- **System boundary**: the line separating what the platform owns from what it does not; it must be identical in every view.
- **Diagrams as code**: diagrams kept as text source (here D2) that diff in review and compile in CI.

## Summary

- Three readers ask three questions, so the set has three views: context, container, and engine components.
- A box belongs at the level whose unit it is; a scheduler is not deployed alone, so it lives only in the component view.
- Every container-level arrow is a protocol, a subprocess, or a file format, because the platform's languages meet only at those boundaries.
- Component-level arrows are calls, labeled with what flows; the scheduler is the hub because of continuous batching.
- D2 sources compile in CI, so a stale diagram becomes a failing build rather than a silent lie.

## Review questions

1. Why does the scheduler appear only in the component view?
   <details><summary>Answer</summary>It is not deployed on its own; it lives inside the engine process. The container view is about deployable units and their protocols, so the scheduler there would mix levels.</details>
2. Why is `worker -> corpus` labeled "subprocess + files" rather than a function call?
   <details><summary>Answer</summary>The worker is Go and the training tools are Python; languages meet only at process and file boundaries. The label names that boundary, which is where its contract lives.</details>
3. Why does the context view distinguish the client from the operator?
   <details><summary>Answer</summary>They are different actors with different APIs and different trust: the client uses the public OpenAI subset, the operator the admin API and workflow operations.</details>
4. Why do prefill and decode have different labels on their arrows to the KV manager?
   <details><summary>Answer</summary>Prefill allocates and writes blocks, while decode reads and appends them. A KV bug report needs to know which of those operations misbehaved, so the view records the difference.</details>
