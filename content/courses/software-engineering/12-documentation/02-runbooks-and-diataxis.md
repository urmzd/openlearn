# Runbooks and Diataxis docs

Incidents recur when operating knowledge lives only in chat. The engineer who fixed last month's outage remembers which command told them the cause; the engineer paged tonight does not. Runbooks turn that knowledge into something executable: a document that starts from the alert, says what it means, and walks a tired reader from diagnosis to a confirmed recovery. Around them, a Diátaxis index lets every reader find what they came for, whether that is a tutorial, a task procedure, an explanation, or a reference.

This section builds both for the platform from *AI Engineering*. You will produce a documentation index, [`docs/README.md`](../../../../course/solutions/docs/README.md), that sorts every document by reader goal, and a directory [`docs/runbooks/`](../../../../course/solutions/docs/runbooks/) with one runbook per paging alert. The model answers are [`TTFTBudgetBurnFast.md`](../../../../course/solutions/docs/runbooks/TTFTBudgetBurnFast.md), [`engine-crashloop.md`](../../../../course/solutions/docs/runbooks/engine-crashloop.md), [`DurableDeadLetters.md`](../../../../course/solutions/docs/runbooks/DurableDeadLetters.md), [`DurableRedeliveriesSpike.md`](../../../../course/solutions/docs/runbooks/DurableRedeliveriesSpike.md), and [`RaftPartition.md`](../../../../course/solutions/docs/runbooks/RaftPartition.md), all following the template in [`course/contracts/templates/RUNBOOK.md`](../../../../course/contracts/templates/RUNBOOK.md).

The section builds on the Diátaxis framework from the [chapter opening](README.md) and on [Architecture decision records](01-architecture-decision-records.md): ADRs are explanation, runbooks are how-to, and the index keeps the two apart. The alerts themselves come from [SLOs and multi-window burn-rate alerts](../../systems/11-observability/04-slos-and-burn-rate-alerts.md) in *Systems Engineering*. Before you read the annotated runbook below, write down the four diagnostic questions you would ask for its alert.

> **Learning objectives**
> - Build a documentation index that sorts documents by reader goal into the four Diátaxis types.
> - Write a runbook that is found from its alert and gives an owner, meaning, symptoms, diagnosis, and mitigation, in that order.
> - Order diagnostic steps from most to least likely cause, each with a command and how to read its output.
> - Key mitigations to causes and end each one with a stated recovery check.

## How a runbook is shaped

A runbook starts with impact and a trigger, names a safe first action, and separates diagnosis from recovery. Each command has expected output and a rollback condition. Those rules follow from five observations about who reads operating documents and when.

Readers arrive with a goal, not a document name. Diátaxis sorts documents by that goal: learning (tutorial), doing a task (how-to), understanding (explanation), or looking something up (reference). So the index should be a table of goals, each pointing to one document.

A runbook is a how-to guide read under stress. Its reader is paged, tired, and not the author. It must therefore be found from the alert (same name, linked from the alert's annotation), start with what the alert means, and give commands rather than prose.

Diagnosis comes before mitigation. The same symptom, slow first tokens for example, has different causes (a lost replica, saturation, a bad rollout), and each has a different fix. So diagnosis is a short decision sequence that ends in a cause, and mitigation is keyed by cause.

Every step must be checkable. Each command is followed by what its output means, and each mitigation by how to confirm it worked; otherwise the reader cannot tell whether to move on.

Documents rot unless someone owns them. So each runbook names its owner, and the index states when it must be updated.

## Working one alert by hand

Before looking at the solution, work an availability burn alert through by hand. First identify the affected SLO. Then query the 5xx and request-rate series to see the size of the problem. Compare against the last deployment to see whether a release lines up with the start of the burn. Then shift traffic or roll back. Finally, verify that the alert clears, and record the incident timeline. Every runbook in the solution follows this arc: meaning, evidence, cause, action, confirmation.

## What the deliverable contains

`docs/README.md` sorts every document by reader goal into the four Diátaxis types. `docs/runbooks/` holds one runbook per paging alert, named after the alert, each with an owner, what the alert means, symptoms, diagnosis, and mitigation. A reviewer checks the set against five requirements:

| Requirement | How it is checked | Why it matters |
|---|---|---|
| the declared paths (`docs/runbooks`, `docs/README.md`) exist | artifact | a plausible narrative cannot stand in for the reviewable deliverable |
| every paging alert has a runbook with the alert's exact name | review | the alert's `runbook` annotation links straight to it |
| each runbook has an owner, a meaning, symptoms, diagnosis, and mitigation | review | the on-call engineer at 3 a.m. needs all five, in that order |
| each diagnostic command says what to look for in its output | review | a command without an expected result does not diagnose anything |
| each mitigation ends with how to confirm recovery | review | "the alert stopped firing" is the success condition, stated in advance |

## Building the index and a runbook

### The index

The index is a table whose rows are reader goals:

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

The "Reader goal" column is the point. It is phrased as what someone wants to do, so they can find the row without knowing the document exists. The C4 diagrams are explanation, because they say why the system is shaped as it is; the contracts and policies are reference. The index ends with a maintenance rule: each service owner updates its runbook in the same pull request whenever an alert, command, threshold, or recovery step changes, and every command is checked in a disposable environment before an operational procedure is approved.

### One runbook, annotated

`TTFTBudgetBurnFast.md` opens with its owner, where it is linked from, and what the alert means in numbers:

```markdown
Owner: the serving path (`<system>-gateway`, `<system>-engine`). Linked from
the alert's `runbook` annotation (obs.03). Written after drill `ops.01`.
The alert means: over both its long and its short window, more than 14.4
times the allowed share of requests waited longer than the TTFT threshold for
their first token. At that rate the 30-day error budget is gone in about two
days.
```

"Gone in about two days" tells the reader how urgent the page is without asking them to do burn-rate math: $30 / 14.4 \approx 2$. The runbook was written after the drill in [A decode worker dies mid-stream](../../systems/12-incident-response-and-chaos/02-kill-decode.md), so its commands are ones that actually worked. Next come the symptoms: what the reader will see on the page, on the dashboard, and from users, so they can confirm they are in the right runbook.

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

The questions run from the most likely cause to the least, and each answer narrows the next step. Mitigation is keyed to those causes. For a lost replica, wait for Kubernetes to replace it and watch `rollout status`. For saturated survivors, scale up now and restore the chart value afterwards with `helm upgrade`. For a bad rollout, run `rollout undo`. The runbook ends with the confirmation: the burn stat under 1, the alert resolved (the short window clears first), and a smoke test of the gateway API passing.

The other runbooks follow the same shape. The full set is in [`docs/runbooks/`](../../../../course/solutions/docs/runbooks/).

### Why this design

One runbook per alert, named after it, means the page links exactly one document and there is no searching under pressure. The cost is some repetition between runbooks, which is accepted because each must stand alone.

Commands with expected output beat prose. Prose is faster to write and slower to act on; a command that says what its output means can be followed by someone new to the system.

A Diátaxis index beats a flat docs folder. A flat folder forces the reader to know the document's name; a goal-sorted index asks them to know only their goal.

Writing runbooks from a real drill, as the "Written after drill" line records, means they contain the commands that actually worked, not the ones the author expected to work.

## What can go wrong

Three pitfalls recur in operational documents, and the solution guards against each.

The first is a claim without evidence: a step says "check the engine" without saying what to run or what a healthy result looks like, so a reviewer cannot verify it and a reader cannot act on it. In the solution every diagnostic step is a command plus how to read its output.

The second is omitting the failure or rollback case. The happy path reads well while the operator has no safe response when the fix itself goes wrong. The solution's mitigation includes `rollout undo` and ends with a confirmation step.

The third is leaving the owner, threshold, or scope implicit, so two engineers reach different decisions from the same evidence. Each runbook names its owner and the alert's threshold, and the index names who maintains what.

> **Tip:** If a runbook's first diagnostic step is the rarest cause, reorder it. Checking for a recent release last keeps the common path short while still ruling it out before you mitigate.

## Where this leads

The index keeps ADRs (explanation) apart from runbooks (how-to), the distinction set up in [Architecture decision records](01-architecture-decision-records.md). The first runbook in the set, the engine crashloop, came out of [First drill: the engine crashloops](../../systems/12-incident-response-and-chaos/01-first-drill.md), and every later drill in *Systems Engineering* ends with a runbook or postmortem in the same shape.

Further reading:

- [Diátaxis](https://diataxis.fr/). The four document types and how to tell them apart.
- *Site Reliability Engineering*, ch. 11, "Being On-Call" (free online). Why playbooks shorten time to mitigation.

## Key terms

- **Runbook**: a how-to guide for operating or recovering a system, found from its alert and read under stress.
- **Diátaxis index**: a table of documents sorted by reader goal into tutorial, how-to, reference, and explanation.
- **Burn rate**: how many times faster than allowed an SLO's error budget is being spent; 14.4 over a 30-day budget exhausts it in about two days.
- **Recovery check**: the stated condition, decided in advance, that confirms a mitigation worked.

## Summary

- Readers arrive with a goal, so the index is a table of goals, each pointing to one document.
- One runbook per paging alert, named after it and linked from the alert's annotation.
- Each runbook gives owner, meaning, symptoms, diagnosis, and mitigation, in that order.
- Diagnosis runs from most to least likely cause, every command paired with how to read its output; mitigation is keyed by cause and ends with a recovery check.
- Runbooks written from real drills contain commands that actually worked; owners update them in the same pull request as any change.

## Review questions

1. Why does the runbook say "gone in about two days" next to the burn rate?
   <details><summary>Answer</summary>A burn rate of 14.4 over a 30-day budget exhausts it in $30 / 14.4 \approx 2.1$ days. Saying so tells the paged reader the urgency directly, without requiring SLO arithmetic during an incident.</details>
2. Why is checking for a recent release the last diagnostic step, not the first?
   <details><summary>Answer</summary>The runbook was written from the failure this alert usually means (a lost replica, then saturation). A rollout is a rarer cause here; checking it last keeps the common path short, while still ruling it out before mitigating.</details>
3. Why is the C4 diagram listed as explanation and not reference?
   <details><summary>Answer</summary>It answers "why is the system shaped like this?", which is understanding, not lookup. The contracts are the reference for exact interfaces.</details>
4. A growing queue appears on the surviving engine replicas while incoming load is unchanged. What does that tell you, and which mitigation follows?
   <details><summary>Answer</summary>Capacity was lost, not that the model became slow. The survivors are saturated, so scale up now and restore the chart value afterwards with `helm upgrade`.</details>
