# Design review: the whole system

## Overview

| | |
|---|---|
| **Module** | `review.03` · proof · docs · Pass 11 · 1 to 2 h |
| **Solution** | a model review of the whole system, `docs/reviews/whole-system.md`, written out in section 5: the design-review rubric answered line by line with evidence from the solution's code, documents, drills, and postmortems, each "no" a note with an owner and a next step, and a decision |
| **Needs** | `review.02` (the control-plane review this one builds on), `craft.09`, and the system's own documents in [`course/solutions/docs/`](../../../../course/solutions/docs/) |
| **Used by** | no code call site; it is the last design review of the course |

## Key Takeaways

- Tie each claim to the exact evidence that supports it: a contract, an ADR, a drill's numbers, a postmortem.
- A whole-system review follows requests and changes across every layer; most of its findings are at the boundaries between layers, not inside one.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 say what a whole-system review looks for and trace a request and a model update through every layer. Section 4 states the task. Section 5 is the model review, with the reasoning behind each answer. Try answering each rubric line from the traces in section 3 before reading the model answer.

---

## 1. Why now

A complete platform review checks whether the training, serving, gateway, durable, agent, deployment, and operations layers fit together under real constraints.

## 2. Principles

Use the whole-system boundary map. Evaluate the same options against cost, latency, reliability, security, maintainability, and implementation effort. Make tradeoffs explicit and support measurements with reproducible evidence.

## 3. Worked example

Follow a customer request from a cited agent answer through gateway policy, engine inference, retrieval, audit, and telemetry. Follow a model update from licensed data through training, evaluation, release approval, canary, rollback, and serving.

**A cited agent answer.**

1. The agent run calls the model through the gateway's OpenAI-compatible API ([`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml)). The gateway authenticates the key, applies rate limits, and evaluates the ordered usage policy ([`policy.go`](../../../../course/solutions/go/gateway/policy/policy.go), [`policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json)); a deny is written as an audit event with the request id, tenant, model, rule id, and reason.
2. The gateway routes to an engine and streams SSE back; the request is metered into the usage ledger ([`usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql)).
3. The agent's `search_docs` tool retrieves chunks (BM25 and vector, fused) and numbers each chunk it shows the model in a citation ledger that keeps a chunk's number for the whole run ([`cite.go`](../../../../course/solutions/go/agent/rag/assemble/cite.go)); the answer's citations are parsed back against that ledger.
4. Every hop emits spans and metrics under the telemetry contracts ([`otel/`](../../../../course/contracts/otel/)), so one trace runs from the agent through the gateway to the engine.

**A model update** ([`model_release.go`](../../../../course/solutions/go/workflows/model_release.go)), a durable workflow whose every decision comes from its input and recorded results:

| Step | What happens | What stops it |
|---|---|---|
| preflight | the model card and the data ledger are checked | a missing card or a ledger row without its licence fields; non-retryable |
| export | the checkpoint becomes `models/<id>/<version>` | export failure |
| evaluate | EvalSuite runs the release's suites, then each gate (suite, task, metric, `>=` or `<=`, value) is applied | any gate failing |
| approve | waits for a human `approve` signal, with an optional timeout | `reject`, or the timeout |
| canary | snapshots the route, then sends `canary_weight` of the public model's traffic to the new version | |
| bake | a durable timer of `canary_wait_s` | |
| burn check | one PromQL query of the canary's SLO burn rate | burn above `max` (14.4, the 1 h page threshold of a 30-day budget) |
| decide | promote, or restore the snapshotted route | |

A cancel or a failure after the canary runs the saga that restores the old route. A crash anywhere replays the same path and re-executes nothing that completed.

## 4. Claim and rubric

Write `docs/reviews/whole-system.md` as the design review of the whole system. Answer each line of the design-review rubric (the eight lines of [`review.01`](01-design-review-tracer-and-serving-platform.md) section 2.1) with evidence from the completed system. Each "no" becomes a review note with an owner and a next step, and the review ends with a decision.

## 5. The solution, step by step

### 5.1 From first principles

1. **Start from the boundary map.** The system's languages meet only at process and file boundaries ([ADR 0001](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md)), and every boundary has a contract. The review walks those boundaries, because a whole-system failure is almost always two correct components disagreeing at one.
2. **One request and one change cover most of the system.** The request path exercises policy, routing, serving, retrieval, citation, metering, and tracing; the release path exercises data provenance, training, evaluation, approval, canary, and rollback.
3. **Evidence must be reproducible.** A drill's numbers and a postmortem's timeline are evidence; "it works" is not.
4. **The previous reviews are inputs.** review.01's capacity numbers and review.02's open notes are re-checked here, not repeated.

### 5.2 The model review

> **Review: the whole system** (`docs/reviews/whole-system.md`). Scope: training (Python), serving (the Rust engine), the gateway and its policy and ledger (Go), the durable control plane and platform workflows, the agent and its retrieval, the deployment on kind, and operations (alerts, runbooks, drills).

| Rubric line | Answer | Evidence or note |
|---|---|---|
| problem, users, constraints first | yes | the system's context and container diagrams ([`docs/c4/`](../../../../course/solutions/docs/c4/)); users are API clients and tenants, the agent, the release approver, and the operator; one kind node, CPU only |
| two options, same criteria, numbers | yes | the engine topology table of review.01 (unified, two unified, prefill plus two decode, with TTFT, TPOT, and KV-transfer numbers), and the single-node versus Raft log of review.02 |
| what the choice gives up | yes | ADR 0001's consequences (versioned file formats and parity fixtures as the price of independent processes); review.01's "what C gives up" |
| every interface names its contract | yes | the contracts tree ([`course/contracts/`](../../../../course/contracts/)): C ABI headers, OpenAPI, protobuf, file formats, telemetry, the subprocess-activity spec |
| failure modes, detection, degradation | yes | table below, each row backed by a drill and a postmortem |
| capacity and latency budgets, arithmetic | **no** | note W1 |
| rollout and rollback, migrations | yes | ModelRelease (section 3) for models; Helm rollback for code, demonstrated by the engine-crashloop postmortem; readers-first KV format migration from review.01 |
| open questions with an owner | yes | notes W1 to W3, and review.02's N1 and N2 carried forward |

**Failure modes with evidence.**

| Failure | Detection | Degradation | Evidence |
|---|---|---|---|
| a bad engine image crashloops | readiness never passes; the gateway answers `503 no_capacity` | 14 minutes without completions until `helm rollback`; no data lost (the engine is stateless) | [`2026-10-08-engine-crashloop.md`](../../../../course/solutions/docs/postmortems/2026-10-08-engine-crashloop.md) |
| a decode pod is killed under load | `TTFTBudgetBurnFast` 2 minutes after the kill | its 11 in-flight streams end with an SSE `error` event; the survivor carries the load for 2 min 40 s and the alert resolves on its own | [`2026-10-09-kill-decode.md`](../../../../course/solutions/docs/postmortems/2026-10-09-kill-decode.md) |
| the durable server is SIGKILLed | `DurableRedeliveriesSpike` | redeliveries; no acknowledged event lost, no effect twice | [`2026-10-09-durable-kill9.md`](../../../../course/solutions/docs/postmortems/2026-10-09-durable-kill9.md) |
| a poison task | `DurableDeadLetters` | the waiting workflow is stuck until fixed and redriven | [`2026-10-09-poison-task.md`](../../../../course/solutions/docs/postmortems/2026-10-09-poison-task.md) |
| a released model burns the SLO on canary | ModelRelease's burn check | the old route is restored automatically | [`model_release.go`](../../../../course/solutions/go/workflows/model_release.go) |
| a policy-violating request | the policy evaluator | denied, with an audit event naming the rule | [`policy.go`](../../../../course/solutions/go/gateway/policy/policy.go) |

**Notes.**

- **W1, capacity of the whole node (owner: platform, before the next release).** review.01 budgets the serving platform (about 4.8 GiB of 12 GiB), and review.02's N1 asks for a WAL budget, but no document adds the durable server, the workers running training and evaluation subprocesses, the agent, and the retrieval index to the same node. Next step: measure each Deployment's working set during a release (training, evaluation, and canary running at once) and write the node's total with headroom, or the limit that schedules them apart.
- **W2, a gap between two layers (owner: platform).** The canary's burn check reads one PromQL number after `canary_wait_s`. At low traffic the canary share may see too few requests in that time for the burn rate to mean anything. Next step: state the minimum number of canary requests the bake must see, and have the release fail closed (restore the route) below it.
- **W3, assumption to test (owner: agent runtime).** The threat model ([`THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md)) says its controls have not been independently penetration-tested. Next step: run the agent's tool-boundary and policy cases from the threat model as a drill and record the result.

**Decision: approve with changes.** W1 and review.02's N1 and N2 must be closed before the next production release; W2 before the next model release; W3 is an accepted, recorded risk.

### 5.3 Why this review

- **Re-use, then extend.** The two earlier reviews already answered most lines for their layers; this review's value is the cross-layer findings (W1 sums memory across layers no single review owned; W2 sits between the release workflow and the observability stack).
- **Every failure row has a drill behind it.** A failure mode that has been injected and measured is evidence; one that has only been imagined is a hypothesis, and the review says which is which.
- **Decision with gates.** "Approve with changes" names which notes block which kind of change (any release, a model release), so the review is usable by the person deciding whether to ship.

### 5.4 Explain it back

1. Why does a whole-system review find problems that the layer reviews (review.01, review.02) did not?
   <details><summary>Answer</summary>
   Each layer review checks its layer against its contracts. Shared resources (one node's memory) and decisions that depend on another layer's numbers (a burn rate that needs enough canary traffic) belong to no single layer, so they appear only when requests and changes are followed across all of them.
   </details>
2. Why is ModelRelease's preflight non-retryable?
   <details><summary>Answer</summary>
   It checks documents (the model card, the data ledger). A missing card or an unlicensed source will be just as missing on the next attempt; retrying would only delay the failure. It needs a human to fix the input.
   </details>
3. Why is "the old route is restored automatically" safe to rely on after a crash in the middle of the canary?
   <details><summary>Answer</summary>
   The route snapshot and the canary change are recorded activity results in the workflow's durable history, and the restore is registered as a saga compensation. After a crash the workflow replays to the same point and the compensation runs from the recorded snapshot, so the route returns to exactly what it was.
   </details>
4. In the engine-crashloop incident, why did the gateway answer 503 for 14 minutes instead of serving from the old version?
   <details><summary>Answer</summary>
   The old pod was replaced before the new one became ready, so there was no healthy replica left. A rollout strategy with `maxUnavailable: 0` and a readiness gate (review.01's choice for the engines) keeps the old pod until the new one is ready.
   </details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Treating version selection or a document claim as implicit | A reader or client cannot determine which contract or evidence applies. | every rubric answer links its contract, document, or postmortem |
| 2 | Skipping the failure path | The successful example works, but malformed input or rollback breaks the system. | the failure table is built from injected drills, and the release trace includes its rollback |
| 3 | Omitting a named owner and recovery condition | The artifact cannot guide an operator when conditions change. | each note has an owner and a next step; the decision says what each note blocks |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `review.02` | the control-plane review whose notes this one carries forward |
| Back | `craft.09` | read before writing the review; use it to verify the relevant contract or evidence |
