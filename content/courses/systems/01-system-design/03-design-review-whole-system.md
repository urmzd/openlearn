# Design review: the whole system

The two previous reviews each looked at one layer: the serving platform, then the control plane. A complete platform review asks a different question: do the training, serving, gateway, durable, agent, deployment, and operations layers fit together under real constraints? Most of its findings are not inside any one layer. They sit at the boundaries between layers, where two components that each pass their own review disagree, or where a shared resource belongs to nobody.

This is the last design review in the book, and it builds directly on [Design review: the control plane](02-design-review-control-plane.md), whose open notes it carries forward. It also relies on the system's diagrams, drawn as in [C4 diagrams in D2](../../software-engineering/11-diagramming/01-c4-diagrams-in-d2.md) from *Software Engineering*, and on the system's own documents in [`course/solutions/docs/`](../../../../course/solutions/docs/): ADRs, the threat model, runbooks, and postmortems. You will trace a customer request and a model update through every layer, then read a model review, `docs/reviews/whole-system.md`, written out in full below. As before, every rubric line is answered with evidence from the solution's code, documents, drills, and postmortems, each "no" is a note with an owner and a next step, and the review ends in a decision.

Try answering each rubric line from the two traces before reading the model's answer.

> **Learning objectives**
> - Walk the whole-system boundary map and explain why cross-layer failures are found at boundaries.
> - Evaluate options against cost, latency, reliability, security, maintainability, and implementation effort.
> - Back every failure mode with reproducible evidence: a drill's numbers or a postmortem's timeline.
> - Find findings that no single layer review owns, such as shared node memory and canary traffic too thin for a burn rate.
> - Write a decision that names which notes block which kind of change.

## What a whole-system review looks for

Start from the whole-system boundary map. The system's languages meet only at process and file boundaries, as recorded in [ADR 0001](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md), and every boundary has a contract. The review walks those boundaries, because a whole-system failure is almost always two correct components disagreeing at one of them.

Options are evaluated against the same criteria across layers: cost, latency, reliability, security, maintainability, and implementation effort. Tradeoffs are made explicit, and measurements are backed by reproducible evidence. A drill's numbers and a postmortem's timeline are evidence; "it works" is not.

Two more principles keep the review tractable. One request and one change cover most of the system: the request path exercises policy, routing, serving, retrieval, citation, metering, and tracing, and the release path exercises data provenance, training, evaluation, approval, canary, and rollback. And the previous reviews are inputs. The serving-platform review's capacity numbers and the control-plane review's open notes are re-checked here, not repeated.

## Tracing a request and a change

The first trace follows a customer request from a cited agent answer through gateway policy, engine inference, retrieval, audit, and telemetry. The second follows a model update from licensed data through training, evaluation, release approval, canary, rollback, and serving.

### A cited agent answer

1. The agent run calls the model through the gateway's OpenAI-compatible API ([`openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml)). The gateway authenticates the key, applies rate limits, and evaluates the ordered usage policy ([`policy.go`](../../../../course/solutions/go/gateway/policy/policy.go), [`policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json)); a deny is written as an audit event with the request id, tenant, model, rule id, and reason.
2. The gateway routes to an engine and streams SSE back; the request is metered into the usage ledger ([`usage.v1.sql`](../../../../course/contracts/formats/usage.v1.sql)).
3. The agent's `search_docs` tool retrieves chunks (BM25 and vector, fused) and numbers each chunk it shows the model in a citation ledger that keeps a chunk's number for the whole run ([`cite.go`](../../../../course/solutions/go/agent/rag/assemble/cite.go)); the answer's citations are parsed back against that ledger.
4. Every hop emits spans and metrics under the telemetry contracts ([`otel/`](../../../../course/contracts/otel/)), so one trace runs from the agent through the gateway to the engine.

### A model update

The model update is a durable workflow, [`model_release.go`](../../../../course/solutions/go/workflows/model_release.go), whose every decision comes from its input and recorded results. The following table lists its steps in order, with what happens at each and what stops the release.

| Step | What happens | What stops it |
|---|---|---|
| preflight | the model card and the data ledger are checked | an empty model card, or a ledger row that forbids training (no `train` in `allowed_uses`, or revoked); non-retryable |
| export | the checkpoint becomes `models/<id>/<version>` | export failure |
| evaluate | EvalSuite runs the release's suites, then each gate (suite, task, metric, `>=` or `<=`, value) is applied | any gate failing |
| approve | waits for a human `approve` signal, with an optional timeout | `reject`, or the timeout |
| canary | snapshots the route, then sends `canary_weight` of the public model's traffic to the new version | |
| bake | a durable timer of `canary_wait_s` | |
| burn check | one PromQL query of the canary's SLO burn rate | burn above `max` (14.4, the 1 h page threshold of a 30-day budget) |
| decide | promote, or restore the snapshotted route | |

A cancel or a failure after the canary runs the saga that restores the old route. A crash anywhere replays the same path and re-executes nothing that completed.

## The task

The task is to write `docs/reviews/whole-system.md` as the design review of the whole system. Answer each of the eight rubric lines (defined in [Design documents and design reviews](01-design-review-tracer-and-serving-platform.md#design-documents-and-design-reviews)) with evidence from the completed system. Each "no" becomes a review note with an owner and a next step, and the review ends with a decision.

## The model review

The review opens with its scope:

> **Review: the whole system** (`docs/reviews/whole-system.md`). Scope: training (Python), serving (the Rust engine), the gateway and its policy and ledger (Go), the durable control plane and platform workflows, the agent and its retrieval, the deployment on kind, and operations (alerts, runbooks, drills).

Its rubric answers lean heavily on the two earlier reviews, as they should:

| Rubric line | Answer | Evidence or note |
|---|---|---|
| problem, users, constraints first | yes | the system's context and container diagrams ([`docs/c4/`](../../../../course/solutions/docs/c4/)); users are API clients and tenants, the agent, the release approver, and the operator; one kind node, CPU only |
| two options, same criteria, numbers | yes | the engine topology table of the serving-platform review (unified, two unified, prefill plus two decode, with TTFT, TPOT, and KV-transfer numbers), and the single-node versus Raft log of the control-plane review |
| what the choice gives up | yes | ADR 0001's consequences (versioned file formats and parity fixtures as the price of independent processes); the serving-platform review's "what C gives up" |
| every interface names its contract | yes | the contracts tree ([`course/contracts/`](../../../../course/contracts/)): C ABI headers, OpenAPI, protobuf, file formats, telemetry, the subprocess-activity spec |
| failure modes, detection, degradation | yes | table below, each row backed by a drill and a postmortem |
| capacity and latency budgets, arithmetic | **no** | note W1 |
| rollout and rollback, migrations | yes | ModelRelease (traced above) for models; Helm rollback for code, demonstrated by the engine-crashloop postmortem; readers-first KV format migration from the serving-platform review |
| open questions with an owner | yes | notes W1 to W3, and the control-plane review's N1 and N2 carried forward |

### Failure modes with evidence

Every row of the failure table has a drill or a postmortem behind it. That is the difference between evidence and a hypothesis: a failure mode that has been injected and measured is evidence, and one that has only been imagined is not.

| Failure | Detection | Degradation | Evidence |
|---|---|---|---|
| a bad engine image crashloops | readiness never passes; the gateway answers `503 no_capacity` | 14 minutes without completions until `helm rollback`; no data lost (the engine is stateless) | [`2026-10-08-engine-crashloop.md`](../../../../course/solutions/docs/postmortems/2026-10-08-engine-crashloop.md) |
| a decode pod is killed under load | `TTFTBudgetBurnFast` 2 minutes after the kill | its 11 in-flight streams end with an SSE `error` event; the survivor carries the load for 2 min 40 s and the alert resolves on its own | [`2026-10-09-kill-decode.md`](../../../../course/solutions/docs/postmortems/2026-10-09-kill-decode.md) |
| the durable server is SIGKILLed | `DurableRedeliveriesSpike` | redeliveries; no acknowledged event lost, no effect twice | [`2026-10-09-durable-kill9.md`](../../../../course/solutions/docs/postmortems/2026-10-09-durable-kill9.md) |
| a poison task | `DurableDeadLetters` | the waiting workflow is stuck until fixed and redriven | [`2026-10-09-poison-task.md`](../../../../course/solutions/docs/postmortems/2026-10-09-poison-task.md) |
| a released model burns the SLO on canary | ModelRelease's burn check | the old route is restored automatically | [`model_release.go`](../../../../course/solutions/go/workflows/model_release.go) |
| a policy-violating request | the policy evaluator | denied, with an audit event naming the rule | [`policy.go`](../../../../course/solutions/go/gateway/policy/policy.go) |

### Notes

The notes are where this review earns its keep, because each one sits across layers:

- **W1, capacity of the whole node (owner: platform, before the next release).** The serving-platform review budgets the serving platform (about 4.8 GiB of 12 GiB), and the control-plane review's N1 asks for a WAL budget, but no document adds the durable server, the workers running training and evaluation subprocesses, the agent, and the retrieval index to the same node. Next step: measure each Deployment's working set during a release (training, evaluation, and canary running at once) and write the node's total with headroom, or the limit that schedules them apart.
- **W2, a gap between two layers (owner: platform).** The canary's burn check reads one PromQL number after `canary_wait_s`. At low traffic the canary share may see too few requests in that time for the burn rate to mean anything. Next step: state the minimum number of canary requests the bake must see, and have the release fail closed (restore the route) below it.
- **W3, assumption to test (owner: agent runtime).** The threat model ([`THREAT_MODEL.md`](../../../../course/solutions/docs/THREAT_MODEL.md)) says its controls have not been independently penetration-tested. Next step: run the agent's tool-boundary and policy cases from the threat model as a drill and record the result.

### The decision

**Approve with changes.** W1 and the control-plane review's N1 and N2 must be closed before the next production release; W2 before the next model release; W3 is an accepted, recorded risk.

### Why this review

The review reuses, then extends. The two earlier reviews already answered most lines for their layers, so this one's value lies in the cross-layer findings: W1 sums memory across layers that no single review owned, and W2 sits between the release workflow and the observability stack. Every failure row has a drill behind it, and the review says which claims are measured and which are not. And the decision has gates: "approve with changes" names which notes block which kind of change (any release, or a model release), so the review is usable by the person deciding whether to ship.

## What can go wrong

The same three failures threaten this review as the control-plane one, and the model handles each the same way. Leaving version selection or a document claim implicit means a reader cannot determine which contract or evidence applies; every rubric answer here links its contract, document, or postmortem. Skipping the failure path means the successful example works but malformed input or a rollback breaks the system; the failure table is built from injected drills, and the release trace includes its rollback.

> **Warning:** A review without a named owner and recovery condition for each finding cannot guide an operator when conditions change. Each note here has an owner and a next step, and the decision says what each note blocks.

## Where this leads

This review closes the chain that began with [Design review: the tracer and the serving platform](01-design-review-tracer-and-serving-platform.md) and continued through [Design review: the control plane](02-design-review-control-plane.md), whose notes it carries forward. Before writing a review like this one, reread [C4 diagrams in D2](../../software-engineering/11-diagramming/01-c4-diagrams-in-d2.md) and use the diagrams to verify the relevant contract or evidence at each boundary. The threat model it cites is the subject of [Threat modeling with STRIDE](../../software-engineering/09-security/01-threat-model-stride.md) in *Software Engineering*.

## Key terms

- **Boundary map**: the set of process and file boundaries where the system's components meet, each with a contract.
- **Cross-layer finding**: a problem that belongs to no single layer, such as a shared resource or a decision that depends on another layer's numbers.
- **Canary**: a release step that sends a fraction (`canary_weight`) of a model's traffic to the new version before promotion.
- **Burn check**: a single query of the canary's SLO burn rate after the bake, compared against a threshold.
- **Evidence**: a reproducible measurement, such as a drill's numbers or a postmortem's timeline, as opposed to an assertion.

## Summary

- Tie each claim to the exact evidence that supports it: a contract, an ADR, a drill's numbers, a postmortem.
- A whole-system review follows requests and changes across every layer; most of its findings are at the boundaries between layers, not inside one.
- Earlier layer reviews are inputs to re-check and carry forward, not work to repeat.
- State the owner, success condition, and recovery action before declaring the work complete, and say which findings block which kind of change.
- Record unresolved assumptions so the next review can test them.

## Review questions

1. Why does a whole-system review find problems that the layer reviews (of the serving platform and the control plane) did not?
   <details><summary>Answer</summary>
   Each layer review checks its layer against its contracts. Shared resources (one node's memory) and decisions that depend on another layer's numbers (a burn rate that needs enough canary traffic) belong to no single layer, so they appear only when requests and changes are followed across all of them.
   </details>
2. Why is ModelRelease's preflight non-retryable?
   <details><summary>Answer</summary>
   It checks documents (the model card, the data ledger). An empty card or a source not allowed for training will be just as wrong on the next attempt; retrying would only delay the failure. It needs a human to fix the input.
   </details>
3. Why is "the old route is restored automatically" safe to rely on after a crash in the middle of the canary?
   <details><summary>Answer</summary>
   The route snapshot and the canary change are recorded activity results in the workflow's durable history, and the restore is registered as a saga compensation. After a crash the workflow replays to the same point and the compensation runs from the recorded snapshot, so the route returns to exactly what it was.
   </details>
4. In the engine-crashloop incident, why did the gateway answer 503 for 14 minutes instead of serving from the old version?
   <details><summary>Answer</summary>
   The old pod was replaced before the new one became ready, so there was no healthy replica left. A rollout strategy with `maxUnavailable: 0` and a readiness gate (the serving-platform review's choice for the engines) keeps the old pod until the new one is ready.
   </details>
