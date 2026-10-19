# Mock engagement: POC and evaluation

## Overview

| | |
|---|---|
| **Module** | `field.04` · practice · docs · Pass 11 · about 30 min reading |
| **Solution** | model answer: [`docs/engagement/poc.md`](../../../../course/solutions/docs/engagement/poc.md) (the pre-registered POC evaluation for Northstar Commerce) |
| **Needs** | none |
| **Used by** | no code call site; the migration plan (`field.05`) takes a passed POC as its first precondition |

## Key Takeaways

- Tie each claim to the exact evidence behind it: a frozen dataset, a revision, a scored result.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 set out why a POC needs its criteria fixed before it runs and what to compare. Section 4 lists what a correct POC plan must contain. Section 5 walks through the model answer for Northstar Commerce and explains why each part is there. Try to write the acceptance gates yourself before reading section 5.

---

## 1. Why now

A proof of concept needs a decision at its end. Evaluation criteria chosen after seeing results invite cherry-picking.

## 2. Principles

Set acceptance thresholds, representative tasks, comparison baseline, data handling, and rollback before the first run. Include operational and safety behavior as well as quality.

## 3. Worked example

Choose a small evaluation set with held-out cases. Compare candidate and baseline under a fixed seed and report quality, latency, cost, refusal behavior, and failure examples. State what the sample cannot establish.

## 4. The artifact

### What a correct solution must handle

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The decision | what a pass permits and what it does not | a POC pass is not read as production approval |
| Pre-registration | dataset manifest, prompt template, model revision, rubric, and thresholds frozen before the first run | results cannot be fitted to the criteria |
| A representative, stratified set | case types in stated proportions, including ones that should be refused or escalated | safety behaviour is measured, not assumed |
| A real baseline | the candidate is compared with the current workflow on the same cases | "good" means better than what exists |
| Scoring that resists bias | independent reviewers, a tie-breaker, exact counts and examples | one reviewer's taste does not decide |
| Gates, including blocking safety gates | quality, safety, latency, and reproducibility thresholds, with safety failures blocking regardless of averages | a high mean cannot hide a dangerous answer |
| Limits | what the sample size cannot establish | the result is not over-generalized |

## 5. The solution, step by step

### 5.1 From first principles

1. **The POC exists to make one decision**, so the plan states it (does the bounded support use case merit a production design phase?) and what a pass does not authorize.
2. **Criteria set after the results are not criteria.** Everything that could be adjusted to make a result look better (dataset, template, revision, rubric, thresholds) is frozen first.
3. **The test set must contain the cases that matter most**, not only the common ones. Stratifying (routine lookups, policy questions, ambiguous requests, must-refuse requests) makes each kind visible in the result.
4. **Quality is relative.** Comparing with the current manual workflow on the same cases is the only comparison the customer's decision depends on.
5. **Averages hide failures.** Safety gates are per case and blocking: one critical policy error or one missed refusal fails the POC even if the average score is high.
6. **Small samples bound what can be claimed.** With 20 refusal cases, a perfect score still leaves a non-trivial rare-failure rate unmeasured; the plan says so.

### 5.2 The model answer, annotated

The full plan is [`docs/engagement/poc.md`](../../../../course/solutions/docs/engagement/poc.md).

**Decision to make.** "Decide whether the bounded support-assistant use case merits a production design phase. This POC does not authorize production use or imply that the model can answer every support question safely."

**Pre-registered evaluation.** Freeze the dataset manifest, prompt template, model and config revision, scoring rubric, and thresholds before the first run. The held-out set is 200 redacted questions: 120 routine order lookups, 40 policy questions, 20 ambiguous requests, and 20 that should be refused or escalated. Priya approves every example and the retention policy; without approval, the set is replaced with synthetic examples, and the plan says: "do not treat the results as customer evidence". Two support reviewers score correctness and completeness independently, a third settles disagreements, exact counts and examples are reported, latency and errors use the `field.03` protocol, and refusals and escalations are logged separately from answer quality.

**Proposed acceptance gates**, explicitly "proposed for sponsor approval, not observed results":

- at least 90% of routine cases rated correct and complete by both reviewers;
- no critical policy error in the 40 policy cases;
- all 20 designed refusal/escalation cases handled correctly;
- latency and reliability within the limits Maya signs before the run;
- every result reproducible from the frozen manifest and revision IDs.

> Failure on a safety gate blocks expansion even if the average quality score is high.

Cost per resolved case and reviewer time are reported separately, with no savings claim unless observed in the workflow.

**Limits and decision record.** Two hundred cases are too few for rare-event safety or broad generalization; the sample may miss seasonal traffic, language variation, adversarial inputs, and policy changes. A pass permits a production architecture review only; a fail returns to manual handling with the gap and owner recorded.

### 5.3 Why this answer

- **An open-ended demo** ("try it and see") is easier to schedule and almost guarantees a decision by impression; pre-registered gates make the outcome a fact.
- **A single aggregate score** (mean rubric score) is simpler but lets a few dangerous answers hide behind many good ones; per-gate, blocking safety criteria cannot be averaged away.
- **A larger, unstratified sample** would narrow the confidence interval on routine cases but might contain no must-refuse cases at all. Stratification spends the 200 cases where the decision is sensitive.
- **Tradeoff accepted.** 200 hand-scored cases with two reviewers is expensive in support staff time; the plan reports that cost rather than hiding it.

### 5.4 Explain it back

1. Why must the thresholds be frozen before the first run?
   <details><summary>Answer</summary>Otherwise they can be chosen, consciously or not, to match the results, and the POC no longer tests anything. Freezing them makes the outcome decidable from the plan alone.</details>
2. If all 20 refusal cases pass, why does the plan still say the sample cannot establish rare-event safety?
   <details><summary>Answer</summary>Zero failures in 20 trials is consistent with a true failure rate of several percent (the "rule of three" puts the 95% upper bound near $3/20 = 15\%$). Rare failures need far more cases to rule out.</details>
3. Why compare against the manual workflow and not against another model?
   <details><summary>Answer</summary>The customer's decision is whether to change from what they do today. A model-versus-model comparison could pick the better candidate and still be worse than the current process.</details>
4. Why are refusals logged separately from answer quality?
   <details><summary>Answer</summary>A correct refusal on a must-refuse case and a wrong answer are different outcomes; mixing them into one quality score would let a model that refuses everything, or nothing, look average.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | every result must be reproducible from the frozen manifest and revision IDs |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | gates are labelled "proposed, not observed"; the limits section bounds the claims |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | decision, evaluation, and security owners named; the decision meeting dated; a fail path defined |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.03` | the latency and reliability protocol the POC reuses |
| Back | `L6.7` | the evaluation practice behind the scoring |
| Back | `ag.09` | the evaluation practice behind the refusal and escalation cases |
| Forward | `field.05` | a passed POC is the migration plan's first precondition |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
