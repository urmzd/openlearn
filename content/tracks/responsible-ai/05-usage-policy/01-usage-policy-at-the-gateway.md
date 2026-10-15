<!-- ol:module ethics.05 -->
# Usage policy at the gateway

## Overview

| | |
|---|---|
| **Module** | `ethics.05` · practice · policy and documentation · Pass 10 · 1 to 2 h |
| **You build** | `docs/USAGE_POLICY.md` and `deploy/policy.v1.yaml` |
| **Contract** | [`policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json), [`linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json) |
| **Tests** | `course/tests/ethics.05/` (why: policy ordering, classifier decisions, red-team coverage, and audit privacy) |
| **Needs** | `gw.08` usage policy enforcement, `L6.5` linear head, `data.05` privacy detectors |
| **Used by** | `MS-agent` checks the policy matrix and red-team prompts |
| **Milestone** | [MS-agent](../../../../course/milestones/MS-agent.toml) |
| **Optional depth** | NIST AI Risk Management Framework, Govern and Measure functions |

## Key takeaways

- Policy rules are ordered and the first matching rule wins.
- Model and tenant restrictions are explicit; classifier rules use the engine embedding endpoint and a versioned linear head.
- A classifier or embeddings failure fails closed with 503.
- Every deny has a stable rule id, a useful reason, and a privacy-safe audit record.
- The policy describes permitted use; it does not claim the classifier can determine intent or replace human review.

## How to work this chapter

```bash
ol start ethics.05
ol tests ethics.05
ol check ethics.05
```

## 1. Why now

Your agent now reaches users through the gateway, and its tool gates cannot prevent every harmful inference request. A policy layer gives operators a place to set model, tenant, token, and classifier restrictions, and gives users a consistent explanation when a request is blocked.

## 2. Principles

The policy document is configuration, not a promise that a model is safe. Keep rules narrow, explain their scope, and review false positives and false negatives. The linear head scores normalized embeddings using `z = W e + b`; the softmax probability for class `c` is `exp(z_c) / sum_j exp(z_j)`. Its training labels, threshold, embedding model, and held-out metrics belong in the model card. The Go gateway evaluates the exported head so Python is not in the request path.

| Symbol | Meaning |
|---|---|
| `e` | normalized embedding returned by engine `/v1/embeddings` |
| `W` | classifier weights, one row per class |
| `b` | one bias per class |
| `z` | class logits |
| `τ` | probability threshold that activates a classifier rule |

## 3. Worked example

Suppose a request has `max_tokens = 2048`, tenant `free`, and matches a rule capped at 1024 tokens. That rule denies before any classifier call. For a separate request, logits `[0, 2]` give unsafe probability `e²/(1+e²) ≈ 0.881`; with threshold `0.9`, that classifier rule allows it. The fixtures exercise both cases, including the embeddings service failing.

## 4. The artifact and its check

Write `USAGE_POLICY.md` with scope, responsible owner, appeal path, review cadence, model and data limitations, and the actions users can expect. Define ordered rules in the schema. Include model restrictions, a token cap, and the fitted classifier head at `models/smol-135m/heads/usage.json`. Explain why each red-team fixture is denied without copying sensitive user content into audit logs.

| Test | Why it exists | Expected result |
|---|---|---|
| `test_policy_schema_and_order` | Rejects ambiguous or misspelled rules | First matching deny wins |
| `test_red_team_prompts_are_denied` | Checks configured refusals at the policy boundary | Every blocked fixture returns 451 with its rule id |
| `test_no_pii_in_audit_examples` | Avoids retaining prompt PII in logs | Rule id and request id only; redact detector matches |

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Putting a broad allow rule before a targeted deny | `test_policy_schema_and_order`; mutant `s01` |
| Treating an embeddings timeout as safe | `test_red_team_prompts_are_denied`; mutant `s02` |
| Logging the prompt in a deny event | `test_no_pii_in_audit_examples`; mutant `s03` |
| Applying the threshold to a raw logit | `test_policy_schema_and_order`; mutant `s04` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.08` | Read the enforcement behavior this policy artifact configures. |
| Back | `L6.5` | Inspect how the exported linear head is fitted and represented. |
| Back | `data.05` | Reuse its privacy detector categories when writing safe audit examples. |
| Forward | `MS-agent` | Checks the configured policy matrix, red-team fixture set, audit log, and gateway behavior. |
| Forward | `field.06` | Uses this policy in a customer security review. |

## Going further

Production policy systems add policy version rollout, appeals, jurisdiction-specific rules, monitoring for drift, and human review for high-impact decisions. Keep each addition traceable to an owner and an observed failure mode.
