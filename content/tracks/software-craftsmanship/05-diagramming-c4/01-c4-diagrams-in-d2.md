# C4 diagrams in D2

## Overview

| | |
|---|---|
| **Module** | `craft.09` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | [`docs/c4/context.d2`](../../../../course/solutions/docs/c4/context.d2), [`docs/c4/containers.d2`](../../../../course/solutions/docs/c4/containers.d2), [`docs/c4/components-engine.d2`](../../../../course/solutions/docs/c4/components-engine.d2) (rendered: [`context.svg`](../../../../course/solutions/docs/c4/context.svg), [`containers.svg`](../../../../course/solutions/docs/c4/containers.svg), [`components-engine.svg`](../../../../course/solutions/docs/c4/components-engine.svg)) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |

## Key Takeaways

- Tie each claim to the exact platform evidence that `C4 diagrams in D2` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 explain what each C4 level is for and trace one request through the levels by hand. Section 4 states what the three diagrams must contain. Section 5 walks through the solution's three D2 files and explains why each box and arrow is at the level it is. Before reading each view, sketch it yourself from the request trace. The `.d2` files render with `d2 docs/c4/context.d2 docs/c4/context.svg` (and likewise for the others); a diagram that no longer compiles is a failing test.

---

## 1. Why now

The architecture is now too large to hold in one view. A C4 set lets a new maintainer move from users and external systems to containers and then to engine components.

## 2. Principles

Keep the context diagram about people and external dependencies. Show deployable processes at container level. Put internal services and stores in the component view. Every arrow names a protocol or file boundary: the languages of the platform meet only at process and file boundaries ([ADR 0001](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md)), so an arrow without one is either a missing interface or a box at the wrong level.

## 3. Worked example

Trace one completion request from CLI through gateway, engine, artifact storage, and telemetry. Draw one view per abstraction level; check that the same system boundary is preserved and the engine subcomponents appear only in the component view.

By hand: the client sends HTTPS to the gateway (context and container); the gateway sends HTTP inference to the engine (container); inside the engine the HTTP layer hands the request to the scheduler, which tokenizes, runs prefill and decode, and allocates KV blocks (component only); the engine reads safetensors and tokenizer files from storage (container); the gateway and engine export OTLP to the collector (container). Every hop names its protocol or file format, and nothing inside the engine appears above the component view.

## 4. The artifact

Three D2 files under `docs/c4/`: `context.d2` (people, the platform, external systems), `containers.d2` (each deployable process and managed store), and `components-engine.d2` (the inside of the Rust engine), each rendered to SVG next to it.

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the three declared paths exist and compile with `d2` | artifact | a plausible narrative cannot stand in for a reviewable diagram |
| every arrow is labeled with a protocol or file boundary | review | readers can find the interface (OpenAPI, gRPC, safetensors) behind each line |
| the system boundary is the same in all three views | review | a box that moves across the boundary between views is a contradiction |
| engine internals appear only in the component view | review | each level answers one question; mixing levels hides both answers |

## 5. The solution, step by step

### 5.1 From first principles

1. **Different readers ask different questions.** A stakeholder asks "who uses it and what does it depend on?", an operator "which processes run and how do they talk?", an engine maintainer "what is inside the engine?". One diagram cannot answer all three legibly, so there are three.
2. **A box at a level must be the unit that level is about.** At container level that unit is "something you deploy or a store you manage": the gateway, engine, durable service, worker, training tools, files, and collector. A scheduler is not deployed alone, so it belongs to the component view.
3. **An arrow is an interface.** Since languages meet at process and file boundaries, every container arrow is either a network protocol (HTTPS, HTTP, gRPC, OTLP), a subprocess, or a file format. Labeling it makes the diagram checkable against the contracts.
4. **Diagrams as code** (D2) are diffable and compile in CI, so a renamed service breaks the build of its diagram instead of silently going stale.

### 5.2 The three views, annotated

**Context.** People and external systems around one platform box.

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

client -> platform.gateway: HTTPS / OpenAI subset      # the public API
operator -> platform.gateway: HTTPS / admin API        # a second actor, a second API
operator -> platform.durable: gRPC / workflow operations
platform.gateway -> platform.engine: HTTP inference, gRPC control
# ... engine and durable to artifacts (files); gateway, engine, durable to telemetry (OTLP)
platform.gateway -> provider: HTTPS provider adapter    # the one outbound dependency
```

The two actors are distinguished because they use different APIs (`openai-subset` versus `admin`), and the provider is outside the boundary because the platform does not own it. The platform box is shown with its five major parts rather than as a single opaque box: a deliberate compromise, so the actors' arrows land on the part they actually talk to. The parts carry no technology labels here; those belong to the container view.

**Containers.** Each box is a deployable process or a managed external system.

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
 gateway -> engine: HTTP inference\n gRPC control          # two protocols: data plane and control plane
worker -> durable: gRPC
corpus -> durable: gRPC workflow client
worker -> corpus: subprocess + files                      # the Go-to-Python boundary is a process
engine -> storage: safetensors, tokenizer, KV envelopes   # file formats are interfaces too
corpus -> storage: corpus shards, tokens, checkpoints
 durable -> storage: WAL and snapshots
 gateway -> collector: OTLP
 engine -> collector: OTLP
 durable -> collector: OTLP
 gateway -> provider: HTTPS
```

Each box names its language and responsibilities, because at this level "which process, in which language" is the question. The worker and the training tools, which were one "durable control plane" in the context view, split into the processes that actually run. Every cross-language arrow is a protocol, a subprocess, or a file format, as ADR 0001 requires.

**Components.** Only the engine is opened, because that is the container this pass works inside.

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

Inside one process the arrows are calls, so they are labeled with what flows (batches, blocks, parameters) rather than a protocol. The scheduler is the hub, which is the point the view makes: continuous batching means every request, prefill, and decode step passes through it. Prefill and decode both reach the KV manager but with different verbs (write versus read and append), which is the distinction a KV bug report needs.

Full diagrams: [`docs/c4/`](../../../../course/solutions/docs/c4/).

### 5.3 Why this solution

- **D2 versus a drawing tool.** Text diffs in review, compiles in CI, and lays itself out; the cost is less control over placement.
- **Three views, not four.** C4's code level is left to the source; a fourth diagram would go stale fastest.
- **One component view.** Opening every container would triple the maintenance for readers who mostly work in the engine; other containers' internals are documented in their own chapters.
- **A partly open context box.** Purists keep the context system opaque. Showing the parts makes the actors' entry points visible at a glance; the price is that the context view must change when a major part is added.

### 5.4 Explain it back

1. Why does the scheduler appear only in the component view?
   <details><summary>Answer</summary>It is not deployed on its own; it lives inside the engine process. The container view is about deployable units and their protocols, so the scheduler there would mix levels.</details>
2. Why is `worker -> corpus` labeled "subprocess + files" rather than a function call?
   <details><summary>Answer</summary>The worker is Go and the training tools are Python; languages meet only at process and file boundaries. The label names that boundary, which is where its contract lives.</details>
3. Why does the context view distinguish the client from the operator?
   <details><summary>Answer</summary>They are different actors with different APIs and different trust: the client uses the public OpenAI subset, the operator the admin API and workflow operations.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The artifact names a decision without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | every arrow names the protocol or file format, checkable against the contracts |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | durable state (WAL, snapshots) and telemetry are on the container view, so recovery paths are visible |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | each file's first line states its scope (context, container, engine components) |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.02` | The ADRs; ADR 0001 is the boundary rule every arrow follows. |
| Back | `review.03` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [ADR 0001: languages meet at process and file boundaries](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md) | Compare the container view's arrows with the boundaries the ADR allows. |
| Simon Brown, [The C4 model](https://c4model.com/) | The four levels and the notation rules the three views follow. |
