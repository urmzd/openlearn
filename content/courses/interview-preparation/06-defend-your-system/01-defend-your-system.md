# Defend your system

A technical defense tests whether you understand a system well enough to explain its limits to another engineer. It is the format behind the hardest part of a system design interview, the follow-up questions, and behind every real design review: someone who did not build the system asks why it is the way it is, and you have to answer in a way they can check.

This section treats a defense as a written artifact rather than a performance. You will fix a rubric of eight questions that every defense must answer, work one design question by hand against it, and then read a complete model defense of the reference system, the Python trainer, Rust engine, and Go gateway built across the other books. The model answers come with the evidence they cite, all of it in [`course/solutions/docs/`](../../../../course/solutions/docs/) and [`course/contracts/`](../../../../course/contracts/), and with an explanation of why that evidence is the right kind. Before you read each answer, try to name the file or number you would cite yourself.

There are no code prerequisites. The section is most useful once you have read [Design review: the whole system](../../systems/01-system-design/03-design-review-whole-system.md) in *Systems and Architecture*, since the defense checks the contracts and evidence that review names.

> **Learning objectives**
> - Tie each claim in a defense to the exact evidence that backs it: a contract file, a config value, or a postmortem number.
> - Answer the eight-line design-review rubric in order, and explain why the order matters.
> - Mark a partly supported answer as a review note with an owner, a success condition, and a next step.
> - Record unresolved assumptions so that the next review can test them.

## What a defense is

The principle is simple to state: answer from evidence. A good answer states the constraints, the rejected alternatives, the failure behaviour, and a number you can reproduce. When the data is not available, it says so and names how you would obtain it.

Four ideas follow from that principle, and they shape everything below.

First, a defense is a claim plus the evidence that would falsify it. A reviewer cannot check "the system is robust"; they can check "a SIGKILLed durable server loses no acknowledged event, see the kill-9 postmortem". So every answer in the model defense names a file or a number.

Second, evidence has a hierarchy. A contract file (what the system promises) beats a description of the code; a measured number from a drill or a config value beats an estimate; an estimate beats an adjective. When only an estimate exists, say so and name how to measure it.

Third, the rubric order is the order of a design. Problem comes before options, options before choice, choice before interfaces, interfaces before failures and capacity, and only then rollout. Answering in that order stops the defense from justifying a decision before stating what it had to satisfy.

Fourth, every "no" is useful. The rubric exists to find gaps, and a defense with eight unqualified yeses is less credible than one with a named open question and an owner.

## A question to work by hand

Before reading the full rubric, hold one concrete question in mind, because the model defense answers it along the way. Defend why the engine and gateway use separate boundaries, HTTP for the public API and gRPC for internal control. Then describe how a rolling API migration behaves during a worker restart. Finally, calculate one capacity limit from the design. You will find the first answer under rubric line 4, the second under line 5, and the arithmetic under line 6. Try to sketch your own answers first; the point of the exercise is to notice which parts you can back with a file and which you can only assert.

## The rubric

The defense is a document (in the reference layout, `docs/defense/answers.md`) that answers each line of the design-review rubric with evidence from the completed system. Each line is answered yes or no. A no is not a failure but a review note with an owner and a next step. These are the eight lines:

1. The problem, its users, and the constraints are stated before any solution.
2. At least two options are compared on the same criteria, with numbers where numbers exist.
3. The chosen option names what it gives up, and why that cost is acceptable.
4. Every interface between components names its contract (an `.h`, `.proto`, OpenAPI path, or schema).
5. Failure modes are listed with how each is detected and how the system degrades.
6. Capacity and latency budgets are estimated, with the arithmetic shown.
7. Rollout and rollback are described, including data and contract migrations.
8. Open questions are listed with an owner.

The defense ends with a decision or reflection that names the next action and its owner. With the rubric fixed, you can read a complete answer to it.

## The model defense

What follows is a defense of the reference system, one paragraph per rubric line, each ending with its verdict.

**1. Problem, users, constraints.** The system trains small language models in Python, serves them from a Rust engine, and fronts the engine with a Go gateway that authenticates, limits, routes, caches, and meters requests. Users are API clients speaking an OpenAI-compatible subset, and operators running it on Kubernetes. Constraints: each language owns its own memory and failures; the serving path must stream tokens without buffering; long jobs (corpus builds, training runs, releases) must survive process restarts. Evidence: [ADR-0001](../../../../course/solutions/docs/adr/0001-languages-meet-at-process-and-file-boundaries.md), Context section. **Yes.**

**2. Options compared.** ADR-0001 compares process and file boundaries against in-process bindings and against a C backend for the engine, on the same criterion: who owns memory and failure. Why this is the right evidence: the criterion is stated, and both rejected options fail it for a reason a reviewer can check. What is missing is a number (for example the per-request overhead of the HTTP hop versus an in-process call). **Partly: a review note**, owner the engine owner, next step: measure gateway-added latency at p50 and p99 with the load generator and add it to the ADR.

**3. What the choice gives up.** Process boundaries cost a serialisation step and a network hop per request, and require versioned file formats and parity fixtures (ADR-0001, Consequences). The cost is acceptable because a crash in the engine cannot corrupt the gateway's state, each service builds, tests, and restarts independently, and every boundary is observable (one span per hop in Jaeger). **Yes.**

**4. Every interface names its contract.** Client to gateway: [`openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) and [`v2`](../../../../course/contracts/openapi/openai-subset.v2.yaml). Gateway admin: [`openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml). Gateway to engine control: [`proto/tl/engine/v1/engine.proto`](../../../../course/contracts/proto/tl/engine/v1/engine.proto) (`Prefill`, `Info`, `Cancel`, `Drain`). Python to Rust: [`formats/safetensors.md`](../../../../course/contracts/formats/safetensors.md) and [`formats/tokenizer.md`](../../../../course/contracts/formats/tokenizer.md). Durable server and workers: [`proto/tl/durable/`](../../../../course/contracts/proto/tl/durable/). **Yes.**

This also answers the first part of the question you worked by hand. Public traffic uses HTTP and SSE because the contract is a published, client-facing API that SDKs already speak and that streams naturally. Internal control uses gRPC because it is a typed, versioned schema between services the team owns, with calls (`Cancel`, `Drain`) that have no client-facing equivalent. Two boundaries let each evolve on its own schedule: the public API has a 90-day deprecation window, the internal proto only needs both sides deployed.

**5. Failure modes, detection, degradation.** The postmortems give measured behaviour, not intentions. In [the durable kill-9 drill](../../../../course/solutions/docs/postmortems/2026-10-09-durable-kill9.md) the durable server was SIGKILLed five times; it truncated a torn last record and replayed its log, activities with live leases were redelivered after their visibility timeout, `DurableRedeliveriesSpike` paged 80 s after the first kill, new workflow starts failed with `UNAVAILABLE` for about 8 s per restart, and no acknowledged event was lost. Engine crash loops have [a runbook](../../../../course/solutions/docs/runbooks/engine-crashloop.md) and a [postmortem](../../../../course/solutions/docs/postmortems/2026-10-08-engine-crashloop.md); poison tasks are dead-lettered ([runbook](../../../../course/solutions/docs/runbooks/DurableDeadLetters.md)). **Yes.**

The second part of that question, a rolling API migration during a worker restart: the API version is selected per request by the client (`X-TL-API-Version: 2`), and the gateway, not the worker, interprets it. A worker restart affects durable tasks, which are leased and redelivered; it does not change which API version an in-flight HTTP request uses. So the migration and the restart are independent: the restart delays work by up to a visibility timeout, and the migration window is unaffected.

**6. Capacity with arithmetic.** One limit is set directly in [`gateway.runtime.toml`](../../../../course/solutions/entry/config/gateway.runtime.toml): `env_key_rpm = 60`, so the environment key sustains $60 / 60 = 1$ request per second. A burst of 64 requests in one second therefore exceeds the budget and must meet at least one `429`; that is exactly why the limit sits under 64. For the durable side, the kill-9 drill gives a recovery number: about 8 s of unavailability per restart, so five restarts cost about 40 s of rejected starts, which the CLI's retry with backoff hid. Engine throughput is not stated here with arithmetic. **Partly: a review note**, owner the engine owner, next step: derive the KV-cache tokens that fit in the engine's memory budget and the resulting concurrent-stream ceiling.

**7. Rollout and rollback, including migrations.** [The deprecation policy](../../../../course/solutions/docs/DEPRECATION.md) covers the API v1 to v2 migration (notice headers, at least 90 days, exit when v1 traffic is zero for 30 consecutive days, rollback by routing the affected cohort back to v1) and the KV envelope v1 to v2 migration (readers accept both before any writer emits v2; rollback by disabling v2 writes). Model rollout is a durable release workflow with a canary. **Yes.**

**8. Open questions with owners.** From the answers above: gateway-added latency (engine owner, measure with the load generator); the concurrent-stream ceiling (engine owner, derive from the KV budget); and whether 8 s of durable unavailability per restart is acceptable once more callers depend on it (durable owner, compare against the start-latency SLO). **Yes.**

**Decision.** The design stands. The next action is the gateway-latency measurement, owned by the engine owner, because it is the cheapest missing number and it closes two notes (lines 2 and 6) at once.

### Why this answer works

Three properties make this defense strong. It cites the contract, config, or postmortem for every claim, so a reviewer can verify each line in minutes. It marks two lines as partial instead of padding them: a defense that hides a missing number invites the reviewer to find it, while one that names it, with an owner, shows the system is understood well enough to know its gaps. And it uses the rubric order, so the reasoning cannot run backwards from the decision.

## What can go wrong

Most weak defenses fail in one of three ways, and the model defense is built to avoid each of them.

The first is treating version selection or a document claim as implicit. The symptom is that a reader or client cannot determine which contract or evidence applies. The model defense avoids this by citing a contract or document by path on every rubric line, and the system itself makes the API version an explicit header rather than something inferred.

The second is skipping the failure path. The successful example works, but malformed input or a rollback breaks the system, and the defense never mentions it. Here, line 5 cites measured drill behaviour and line 7 states the rollback for each migration.

The third is omitting a named owner and recovery condition. An artifact without them cannot guide an operator when conditions change. In the model defense, each partial answer and each open question has an owner and a next step.

> **Warning:** An answer that sounds complete but cites nothing is the most common failure in a design interview. If you cannot name the file or the number, say that you would measure it and how.

## Where this leads

This section closes the loop on the whole build. To check the contracts and evidence it cites, reread [Design review: the whole system](../../systems/01-system-design/03-design-review-whole-system.md) in *Systems and Architecture*. For how owners and next actions are written down so that someone else can act on them, reread [Handing off operations](../../field-engineering/08-mock-engagement/07-escalation-and-handoff.md) in *Field Engineering*. The company chapters in this book show how different interviewers will phrase the same questions.

### Further reading

- [Course contracts](../../../../course/contracts/): the current contracts and system boundaries, the primary evidence for any defense.
- [Reference docs](../../../../course/solutions/docs/): the ADRs, runbooks, postmortems, and policies to cite.

## Key terms

- **Defense**: a written answer to a fixed design-review rubric in which every claim cites evidence that could falsify it.
- **Design-review rubric**: the eight questions, from problem statement to open questions, that a design must answer in order.
- **Evidence hierarchy**: the ranking contract file, then measured number or config value, then estimate, then adjective.
- **Review note**: a "no" or "partly" on a rubric line, recorded with an owner and a next step rather than hidden.

## Summary

- A defense is a set of claims, each tied to a contract, config value, or measured number that a reviewer can check.
- Answer the rubric in design order so the reasoning cannot run backwards from the decision.
- Partial answers with owners and next steps are more credible than unqualified yeses.
- Separate boundaries (public HTTP and SSE, internal gRPC) let each interface evolve under its own change rules.
- End with a decision and the cheapest next action, named with its owner.

## Review questions

1. Why is "the system survived kill -9" weaker evidence than "no acknowledged event was lost across five SIGKILLs; the first recovery replayed 412 records"?

   <details><summary>Answer</summary>

   The first is a conclusion; the second is a measurement with a defined property (acknowledged events) and a count a reviewer can reproduce by rerunning the drill. Only the second can be falsified.

   </details>

2. Why do the public API and the engine control plane use different protocols rather than one?

   <details><summary>Answer</summary>

   They have different consumers and different change rules. The public API must match what OpenAI-compatible clients send and must deprecate slowly (90 days). The control plane is between services the team deploys together, needs calls clients never see (`Cancel`, `Drain`), and benefits from a typed schema. One protocol would force one change rule on both.

   </details>

3. Why does the rate limit sit at 60 requests per minute rather than 64 or more?

   <details><summary>Answer</summary>

   60 per minute is 1 per second. A burst of 64 inside that window is over budget, so the limiter must reject some of it with `429`. At 64 or above the burst could pass, and the limit's behaviour would go unobserved.

   </details>

4. Why does a worker restart not interfere with an in-progress API version migration?

   <details><summary>Answer</summary>

   The version is chosen per request by the client header and interpreted at the gateway. Workers only hold leased durable tasks, which are redelivered after the visibility timeout. The two mechanisms share no state.

   </details>
