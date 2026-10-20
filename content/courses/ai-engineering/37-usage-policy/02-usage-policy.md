# Usage policy

This is a compact example of the policy document described in [Usage policy at the gateway](01-usage-policy-at-the-gateway.md). It is written the way a gateway owner would publish it: scope and ownership first, then how decisions are made, what data is kept, how a user gets a decision reviewed, and when the policy itself is reviewed. The fuller reference version lives in [`course/solutions/docs/USAGE_POLICY.md`](../../../../course/solutions/docs/USAGE_POLICY.md).

## Scope and owner

This policy governs requests sent through your TinyLLM gateway. The gateway owner maintains the rules and reviews them at each model or policy revision. It is configuration for this system, not a claim that a classifier can determine intent or replace professional judgment.

## Decisions

Rules are evaluated in file order; the first match decides. Model, tenant, and requested output-token limits may deny a request. The classifier uses the configured engine's `/v1/embeddings` endpoint and the exported usage head. A matching deny returns HTTP 451 with code `usage_policy`, a stable rule id, and a short reason. An embeddings service failure returns 503 and fails closed.

## Data and limits

Prompts may contain personal or confidential data. The gateway must not place prompt text in policy audit logs. The existing PII detectors from [PII scrub with typed placeholders and audit spans](../../data-engineering/05-corpus-pipeline/05-pii-scrub.md) redact any request-derived audit field. The classifier is trained on a small labeled fixture and may produce false positives and false negatives. It is not a safety guarantee.

## User recourse

The response includes the rule id and reason. Operators review a report that includes request id, tenant id, model id, policy revision, decision, and timestamp. Users can contact the system owner with the request id to ask for review. Reviewers must not request that the user resend sensitive prompt text unless necessary and authorized.

## Review cadence

Review rules and held-out classifier metrics before each model or policy release, after a substantiated false positive or false negative, and at least quarterly. Record the policy revision and decision in the release notes. Retire rules that no longer have an owner or evidence for their scope.
