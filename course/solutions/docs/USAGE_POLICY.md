# Gateway usage policy

**Owner:** Gateway Safety Review (gateway maintainers and the privacy lead)  
**Revision:** `usage-policy-2026-10-09.1`  
**Review cadence:** before each model or classifier change, and at least every 90 days  
**Appeal:** users can request review through the support channel with the request ID; include no prompt text in the ticket unless the user chooses to provide it.

This policy describes which requests the public gateway accepts. It does not certify that a model is safe, infer a user's intent, or replace human review. Operators own the rule set and must review changes with the safety and privacy leads.

## Decision order

The gateway evaluates `deploy/policy.v1.yaml` from top to bottom and stops at the first match. `restricted-model` blocks the named experimental model. `free-token-cap` blocks free-tier requests exceeding 1,024 output tokens. `unsafe-prompt` evaluates a normalized embedding with the versioned usage linear head and blocks the unsafe class at probability 0.90 or higher. A classifier or embedding service error fails closed with a temporary service error; it is never treated as an allow.

Keep targeted denials ahead of broad rules. Each rule needs a stable identifier and a user-safe reason. A change to rule order, model, head, or threshold requires a policy revision bump and a red-team review.

## User outcome and review

The response includes the stable rule ID and a short reason, plus a request ID for support. A user may appeal a mistaken denial. The reviewer checks the policy revision, rule ID, and available aggregate detector evidence, then records an outcome without copying prompt content into ordinary logs or tickets. Emergency policy changes require a named approver and a follow-up review within one business day.

## Privacy-safe audit fields

Record the request ID, policy revision, matched rule ID, decision, timestamp, model ID, and classifier version when applicable. The audit system **must not place prompt text**, completions, credentials, or raw embedding vectors in the policy audit event. Restrict access to audit records to gateway operators and the privacy response team. The retention policy keeps per-request decision metadata for 30 days, then deletes it; retain de-identified aggregate counts for up to 13 months to detect rule drift.

## Limitations and monitoring

Classifier thresholds trade false positives against false negatives and can drift as prompts and models change. The threshold is not a measure of certainty or user intent. Review false-positive and false-negative samples through a privacy-approved process, monitor deny rates by rule and model, and suspend the classifier rule if its serving dependency or calibration is unhealthy. Human reviewers must not use this policy alone to make consequential decisions about a person.
