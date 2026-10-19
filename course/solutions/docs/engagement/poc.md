# Proof of concept evaluation: Northstar Commerce

**Decision owner:** Maya Chen, platform lead  
**Evaluation owner:** Alex Rivera, solutions engineer  
**Security approver:** Priya Shah  
**Decision meeting:** 2026-10-23

## Decision to make

Decide whether the bounded support-assistant use case merits a production design phase. This POC does not authorize production use or imply that the model can answer every support question safely.

## Pre-registered evaluation

Before the first run, freeze the dataset manifest, prompt template, model/config revision, scoring rubric, and thresholds. The held-out set contains 200 redacted support questions: 120 routine order lookups, 40 policy questions, 20 ambiguous requests, and 20 requests that should be refused or escalated. Priya must approve every example and the retention policy. If approval is not obtained, replace the set with synthetic examples and do not treat the results as customer evidence.

Compare the candidate to the current manual workflow on the same cases. Two support reviewers score correctness and completeness independently; disagreements go to a third reviewer. Report exact counts and examples, not only an aggregate score. Measure p95 TTFT/TPOT and error rate using the protocol in `performance.md`. Log refusal and escalation outcomes separately from answer quality.

## Proposed acceptance gates

These are proposed for sponsor approval, not observed results:

- At least 90% of routine cases are rated correct and complete by both reviewers.
- No critical policy error occurs in the 40 policy cases.
- All 20 designed refusal/escalation cases are escalated or refused correctly.
- Latency and reliability meet the numeric limits Maya signs before the run.
- Every result is reproducible from the frozen manifest and revision IDs.

Failure on a safety gate blocks expansion even if the average quality score is high. Report cost per resolved case and reviewer time separately; do not claim net savings without observing the operational workflow.

## Limits and decision record

Two hundred cases are too few to establish rare-event safety or broad generalization. The sample may not represent seasonal traffic, language variation, adversarial inputs, or policy changes. A pass permits a production architecture review only. A fail returns the system to manual handling while the team records the gap and owner.

**Next action:** Maya approves or revises the gates by Oct 20; Priya signs the dataset and retention plan. Alex publishes the run manifest and results before the decision meeting. **Decision:** pending evidence; no production approval is implied.
