# Runbooks and Diataxis docs

## Overview

| | |
|---|---|
| **Module** | `craft.10` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | [`docs/README.md`](../../../../course/solutions/docs/README.md) (the Diátaxis index) and [`docs/runbooks/`](../../../../course/solutions/docs/runbooks/): [`TTFTBudgetBurnFast.md`](../../../../course/solutions/docs/runbooks/TTFTBudgetBurnFast.md), [`engine-crashloop.md`](../../../../course/solutions/docs/runbooks/engine-crashloop.md), [`DurableDeadLetters.md`](../../../../course/solutions/docs/runbooks/DurableDeadLetters.md), [`DurableRedeliveriesSpike.md`](../../../../course/solutions/docs/runbooks/DurableRedeliveriesSpike.md), [`RaftPartition.md`](../../../../course/solutions/docs/runbooks/RaftPartition.md) |
| **Contract** | the runbook template [`course/contracts/templates/RUNBOOK.md`](../../../../course/contracts/templates/RUNBOOK.md) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Runbooks and Diataxis docs` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 explain why operating knowledge must be written down and work one alert by hand. Section 4 states what the index and the runbooks must contain. Section 5 walks through the solution's index and one runbook, and explains why each part is there. Before reading the runbook, write down the four diagnostic questions you would ask for that alert.

---

## 1. Why now

Incidents recur when operating knowledge lives only in chat. Runbooks make detection and recovery executable; Diátaxis helps readers find tutorials, task procedures, explanations, and reference.

## 2. Principles

A runbook starts with impact and a trigger, names a safe first action, and separates diagnosis from recovery. Each command has expected output and a rollback condition.

## 3. Worked example

For an availability burn alert, identify the affected SLO, query the 5xx and request-rate series, compare the last deployment, then shift traffic or roll back. Verify the alert clears and record the incident timeline.

## 4. The artifact

`docs/README.md` sorts every document by reader goal into the four Diátaxis types. `docs/runbooks/` holds one runbook per paging alert, named after the alert, each with an owner, what the alert means, symptoms, diagnosis, and mitigation.

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the declared paths (`docs/runbooks`, `docs/README.md`) exist | artifact | a plausible narrative cannot stand in for the reviewable deliverable |
| every paging alert has a runbook with the alert's exact name | review | the alert's `runbook` annotation links straight to it |
| each runbook has an owner, a meaning, symptoms, diagnosis, and mitigation | review | the on-call engineer at 3 a.m. needs all five, in that order |
| each diagnostic command says what to look for in its output | review | a command without an expected result does not diagnose anything |
| each mitigation ends with how to confirm recovery | review | "the alert stopped firing" is the success condition, stated in advance |

## 5. The solution, step by step

### 5.1 From first principles

1. **Readers arrive with a goal, not a document name.** Diátaxis sorts documents by that goal: learning (tutorial), doing a task (how-to), understanding (explanation), looking something up (reference). So the index is a table of goals, each pointing to one document.
2. **A runbook is a how-to under stress.** Its reader is paged, tired, and not the author. So it must be found from the alert (same name, linked from the annotation), start with what the alert means, and give commands rather than prose.
3. **Diagnosis before mitigation.** The same symptom (slow first tokens) has different causes (a lost replica, saturation, a bad rollout), and each has a different fix. So diagnosis is a short decision sequence that ends in a cause, and mitigation is keyed by cause.
4. **Every step must be checkable.** Each command is followed by what its output means; each mitigation by how to confirm it worked. Otherwise the reader cannot tell whether to move on.
5. **Documents rot unless someone owns them.** So each runbook names its owner, and the index states when it must be updated.

### 5.2 The index

```markdown
| Type | Reader goal | Start here |
|---|---|---|
| Tutorial | Build a local serving stack from an empty checkout | [Local quickstart](../../../README.md#quickstart) |
| How-to | Recover a failing engine | [Engine crashloop runbook](runbooks/engine-crashloop.md) |
| How-to | Respond to a TTFT budget burn | [TTFT budget runbook](runbooks/TTFTBudgetBurnFast.md) |
| How-to | Operate durable queues | [Dead letter runbook](runbooks/DurableDeadLetters.md) |
| Explanation | Understand the system boundaries | [C4 context diagram](c4/context.d2) · ... |
| Reference | Check the supported runtime and deployment configuration | [Configuration schemas](../../contracts/config/) |
| Reference | Check model and dataset disclosures | [Model card](MODEL_CARD.md) · [Datasheet](DATASHEET.md) |
| Reference | Check format and API retirement windows | [Deprecation policy](DEPRECATION.md) |
```

The "Reader goal" column is the point: it is phrased as what someone wants to do, so they can find the row without knowing the document exists. The C4 diagrams are explanation (they say why the system is shaped as it is), the contracts and policies are reference. The index ends with a maintenance rule: each service owner updates its runbook when an alert, command, threshold, or recovery step changes, in the same pull request, and every command is checked in a disposable environment before an operational procedure is approved.

### 5.3 One runbook, annotated

`TTFTBudgetBurnFast.md` opens with its owner, where it is linked from, and what the alert means in numbers:

```markdown
Owner: the serving path (`<system>-gateway`, `<system>-engine`). Linked from
the alert's `runbook` annotation (obs.03). Written after drill `ops.01`.
The alert means: over both its long and its short window, more than 14.4
times the allowed share of requests waited longer than the TTFT threshold for
their first token. At that rate the 30-day error budget is gone in about two
days.
```

"Gone in about two days" tells the reader how urgent it is without knowing burn-rate math ($30 / 14.4 \approx 2$). Symptoms then list what the reader will see on the page, the dashboard, and from users, so they can confirm they are in the right runbook.

Diagnosis is four questions, each with a command and how to read its result:

```markdown
1. Which replicas are serving, and did one restart?
   kubectl -n <system> get pods -l app.kubernetes.io/name=<system>-engine -o wide
   ...
   A pod with `AGE` of seconds, `READY 0/1`, or a `Killing` event is the
   replica that went away. Its peers are carrying its share.

2. Is the remaining capacity saturated? ...
   sum by (pod) (tl_engine_queue_depth)
   A growing queue on the survivors with the load unchanged means lost
   capacity, not a slow model.

3. Is the gateway routing around the hole? ...
4. Rule out a release: kubectl -n <system> rollout history deploy/<system>-engine-decode.
   A new revision at the start of the burn points at the rollout instead.
```

The order goes from the most likely cause to the least, and each answer narrows the next step. Mitigation is keyed to those causes: a lost replica (wait for Kubernetes, watch `rollout status`), saturated survivors (scale up now, restore the chart value afterwards with `helm upgrade`), a bad rollout (`rollout undo`). It ends with the confirmation: the burn stat under 1, the alert resolved (the short window clears first), and a smoke test of the gateway API passing.

The other runbooks follow the same shape. Full set: [`docs/runbooks/`](../../../../course/solutions/docs/runbooks/).

### 5.4 Why this solution

- **One runbook per alert, named after it.** The page links one document; there is no search under pressure. The cost is some repetition between runbooks, accepted because each must stand alone.
- **Commands with expected output, not prose.** Prose is faster to write and slower to act on; a command that says what its output means can be followed by someone new to the system.
- **Diátaxis instead of a flat docs folder.** A flat folder makes the reader know the document's name; a goal-sorted index makes them know only their goal.
- **"Written after drill ops.01".** Runbooks written from a real drill contain the commands that actually worked, not the ones the author expected to work.

### 5.5 Explain it back

1. Why does the runbook say "gone in about two days" next to the burn rate?
   <details><summary>Answer</summary>A burn rate of 14.4 over a 30-day budget exhausts it in $30 / 14.4 \approx 2.1$ days. Saying so tells the paged reader the urgency directly, without requiring SLO arithmetic during an incident.</details>
2. Why is checking for a recent release the last diagnostic step, not the first?
   <details><summary>Answer</summary>The runbook was written from the failure this alert usually means (a lost replica, then saturation). A rollout is a rarer cause here; checking it last keeps the common path short, while still ruling it out before mitigating.</details>
3. Why is the C4 diagram listed as explanation and not reference?
   <details><summary>Answer</summary>It answers "why is the system shaped like this?", which is understanding, not lookup. The contracts are the reference for exact interfaces.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The artifact names a decision without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | every diagnostic step is a command plus how to read its output |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | mitigation includes `rollout undo` and a confirmation step |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | each runbook names its owner and the alert's threshold; the index names who maintains what |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.02` | ADRs are explanation; runbooks are how-to. The index keeps the two apart. |
| Back | `ops.00` | The first runbook (engine crashloop) came from this drill. |

## Going further

| Resource | What to inspect |
|---|---|
| [Diátaxis](https://diataxis.fr/) | The four document types and how to tell them apart. |
| *Site Reliability Engineering*, ch. 11 "Being On-Call" (free online) | Why playbooks shorten time to mitigation. |
