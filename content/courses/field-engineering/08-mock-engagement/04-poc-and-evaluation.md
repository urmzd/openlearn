# Pre-registering the POC

A proof of concept needs a decision at its end, and it can only deliver one if its criteria could have failed. Evaluation criteria chosen after seeing the results invite cherry-picking, conscious or not, and a POC judged that way tests nothing. The remedy is the same one clinical trials use: register the plan, the dataset, and the thresholds before the first run.

This section writes the POC plan for Northstar Commerce, following [POC and Evaluation](../04-poc-and-evaluation/README.md) from earlier in this book. It reuses the latency and reliability protocol from [Designing the performance test protocol](03-performance-engagement.md), and its repeatable, frozen-revision scoring rests on the [The evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) in *Build Your Own LLM System* and the [eval runner](../../ai-platform-engineering/09-llm-evaluation/09-eval-runner.md) in *AI Platform Engineering*. The model answer is [`docs/engagement/poc.md`](../../../../course/solutions/docs/engagement/poc.md). Try writing the acceptance gates yourself before reading the walkthrough.

> **Learning objectives**
> - State the single decision a POC makes and what a pass does not authorize.
> - Freeze the dataset, template, revision, rubric, and thresholds before the first run.
> - Build a stratified evaluation set that includes must-refuse cases, and compare against the current workflow.
> - Write blocking safety gates and state what the sample size cannot establish.

## Fixing the criteria before the run

Set acceptance thresholds, representative tasks, a comparison baseline, data handling, and rollback before the first run, and include operational and safety behavior as well as quality. The reasoning behind each piece follows from what a POC is for.

The POC exists to make one decision, so the plan states it (does the bounded support use case merit a production design phase?) and what a pass does not authorize.

Criteria set after the results are not criteria. Everything that could be adjusted to make a result look better (dataset, template, revision, rubric, thresholds) is frozen first.

The test set must contain the cases that matter most, not only the common ones. Stratifying it (routine lookups, policy questions, ambiguous requests, must-refuse requests) makes each kind visible in the result.

Quality is relative. Comparing with the current manual workflow on the same cases is the only comparison the customer's decision depends on.

Averages hide failures. Safety gates are per case and blocking: one critical policy error or one missed refusal fails the POC even if the average score is high.

Small samples bound what can be claimed. With 20 refusal cases, a perfect score still leaves a non-trivial rare-failure rate unmeasured, and the plan says so.

## The shape of the evaluation

The worked procedure is to choose a small evaluation set with held-out cases, compare candidate and baseline under a fixed seed, and report quality, latency, cost, refusal behavior, and failure examples, stating what the sample cannot establish.

## What the plan must contain

A correct POC plan covers these cases:

| Case | What must hold | Why it matters downstream |
|---|---|---|
| The decision | what a pass permits and what it does not | a POC pass is not read as production approval |
| Pre-registration | dataset manifest, prompt template, model revision, rubric, and thresholds frozen before the first run | results cannot be fitted to the criteria |
| A representative, stratified set | case types in stated proportions, including ones that should be refused or escalated | safety behaviour is measured, not assumed |
| A real baseline | the candidate is compared with the current workflow on the same cases | "good" means better than what exists |
| Scoring that resists bias | independent reviewers, a tie-breaker, exact counts and examples | one reviewer's taste does not decide |
| Gates, including blocking safety gates | quality, safety, latency, and reproducibility thresholds, with safety failures blocking regardless of averages | a high mean cannot hide a dangerous answer |
| Limits | what the sample size cannot establish | the result is not over-generalized |

## Reading the model answer

The full plan is [`docs/engagement/poc.md`](../../../../course/solutions/docs/engagement/poc.md).

It opens with the decision to make: "Decide whether the bounded support-assistant use case merits a production design phase. This POC does not authorize production use or imply that the model can answer every support question safely."

The pre-registered evaluation comes next. The dataset manifest, prompt template, model and config revision, scoring rubric, and thresholds are frozen before the first run. The held-out set is 200 redacted questions: 120 routine order lookups, 40 policy questions, 20 ambiguous requests, and 20 that should be refused or escalated. Priya approves every example and the retention policy; without approval, the set is replaced with synthetic examples, and the plan says: "do not treat the results as customer evidence". Two support reviewers score correctness and completeness independently, and a third settles disagreements. Exact counts and examples are reported, latency and errors use the [performance protocol](03-performance-engagement.md), and refusals and escalations are logged separately from answer quality.

The acceptance gates are explicitly "proposed for sponsor approval, not observed results":

- at least 90% of routine cases rated correct and complete by both reviewers;
- no critical policy error in the 40 policy cases;
- all 20 designed refusal/escalation cases handled correctly;
- latency and reliability within the limits Maya signs before the run;
- every result reproducible from the frozen manifest and revision IDs.

> Failure on a safety gate blocks expansion even if the average quality score is high.

Cost per resolved case and reviewer time are reported separately, with no savings claim unless one is observed in the workflow.

The plan closes with its limits and the decision record. Two hundred cases are too few for rare-event safety or broad generalization, and the sample may miss seasonal traffic, language variation, adversarial inputs, and policy changes. A pass permits a production architecture review only; a fail returns to manual handling with the gap and its owner recorded.

## Why this shape and not another

An open-ended demo ("try it and see") is easier to schedule and almost guarantees a decision by impression; pre-registered gates make the outcome a fact. A single aggregate score, such as the mean rubric score, is simpler but lets a few dangerous answers hide behind many good ones, whereas per-gate, blocking safety criteria cannot be averaged away. A larger, unstratified sample would narrow the confidence interval on routine cases but might contain no must-refuse cases at all; stratification spends the 200 cases where the decision is sensitive.

The tradeoff accepted is reviewer time. Two hundred hand-scored cases with two reviewers is expensive in support staff time, and the plan reports that cost rather than hiding it.

## What can go wrong

> **Warning:** A result without its evidence cannot be reproduced. Every result in this plan must be reproducible from the frozen manifest and revision IDs.

> **Warning:** An assumption written as a measured fact overstates what the system has shown. The gates are labelled "proposed, not observed", and the limits section bounds the claims.

> **Warning:** Without owners and a decision gate, work stalls. The plan names decision, evaluation, and security owners, dates the decision meeting, and defines a fail path.

## Where this leads

The POC reuses the protocol from [Designing the performance test protocol](03-performance-engagement.md), and its scoring follows the [The evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) and the [eval runner](../../ai-platform-engineering/09-llm-evaluation/09-eval-runner.md), which executes the frozen set and reports per-case results. A passed POC is the first precondition of [Planning the staged migration](05-migration-plan.md).

Further reading:

- The [chapter opening](README.md), to see how the evidence and open decisions travel into each later customer-facing artifact.

## Key terms

- **Pre-registration**: freezing the dataset, template, revision, rubric, and thresholds before the first run.
- **Stratified set**: an evaluation set built from case types in stated proportions, so each type is visible in the result.
- **Blocking gate**: a criterion whose failure fails the POC regardless of other scores.
- **Baseline**: the current workflow, compared with the candidate on the same cases.
- **Rule of three**: with zero failures in $n$ trials, the 95% upper bound on the failure rate is about $3/n$.

## Summary

- A POC makes one decision; say what a pass does and does not permit.
- Freeze everything that could be tuned to the results before the first run.
- Stratify the set so must-refuse and ambiguous cases are measured, and compare with the current workflow.
- Make safety gates per case and blocking, and log refusals separately from quality.
- State plainly what the sample size cannot establish.

## Review questions

1. Why must the thresholds be frozen before the first run?
   <details><summary>Answer</summary>Otherwise they can be chosen, consciously or not, to match the results, and the POC no longer tests anything. Freezing them makes the outcome decidable from the plan alone.</details>
2. If all 20 refusal cases pass, why does the plan still say the sample cannot establish rare-event safety?
   <details><summary>Answer</summary>Zero failures in 20 trials is consistent with a true failure rate of several percent (the "rule of three" puts the 95% upper bound near $3/20 = 15\%$). Rare failures need far more cases to rule out.</details>
3. Why compare against the manual workflow and not against another model?
   <details><summary>Answer</summary>The customer's decision is whether to change from what they do today. A model-versus-model comparison could pick the better candidate and still be worse than the current process.</details>
4. Why are refusals logged separately from answer quality?
   <details><summary>Answer</summary>A correct refusal on a must-refuse case and a wrong answer are different outcomes; mixing them into one quality score would let a model that refuses everything, or nothing, look average.</details>
