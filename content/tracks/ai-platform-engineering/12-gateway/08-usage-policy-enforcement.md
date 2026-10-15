<!-- ol:module gw.08 -->
# Usage policy enforcement

## Overview

| | |
|---|---|
| **Module** | `gw.08` · build · Go · Pass 10 · 2 to 3 h |
| **You build** | `go/gateway/policy/policy.go` |
| **Contract** | [`policy.v1.schema.json`](../../../../course/contracts/formats/policy.v1.schema.json), [`linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json) |
| **Tests** | `course/tests/go/gw_08/` (why: ordered rules, linear-head parity, audit, and fail-closed behavior) |
| **Needs** | [`gw.01`](01-server-skeleton.md) middleware chain, [`gw.02`](../08-authorization-and-access-control/01-api-keys-and-scopes.md) principal, [`gw.05`](05-routing.md) `route.InferenceRequest`, [`ethics.05`](../../responsible-ai/05-usage-policy/01-usage-policy-at-the-gateway.md) policy, [`data.05`](../../data-engineering/05-corpus-pipeline/05-pii-scrub.md) detector list |
| **Used by** | `gw.07` route layer consumes policy decisions; `MS-agent` exercises the policy fixtures |
| **Milestone** | [MS-agent](../../../../course/milestones/MS-agent.toml) |
| **Optional depth** | OPA and Cedar policy evaluation |

## Key takeaways

- Evaluate policy rules in file order and use the first match.
- Model, tenant, and token limits are cheap checks; classifier rules use engine embeddings.
- Stable rule ids travel through audit events and the 451 response.
- Classifier dependencies fail closed with 503.
- Redact request text with the Go PII detector port before logging.

## How to work this chapter

```bash
ol start gw.08
ol tests gw.08
ol check gw.08
```

## 1. Why now

The agent now sends user requests through the gateway, so a usage restriction must be enforced on every inference path. Middleware placement after authentication gives the policy a tenant and model identity while keeping it ahead of rate limiting, cache, routing, and proxying.

## 2. Principles

`Policy.Check(ctx, principal, request)` returns an allow or deny decision. A rule matches only when all stated `match` fields match. `max_tokens: N` matches a request whose requested output cap exceeds `N`. A classifier rule calls the configured engine's `/v1/embeddings`, normalizes the returned vector, evaluates the exported head, and compares the selected class probability with its threshold. A matching deny is audit-logged and returned as 451 `usage_policy`; dependency failure is 503.

## 3. Worked example

For an `unsafe` class logit of 2 and a `safe` logit of 0, `p(unsafe) = exp(2)/(exp(0)+exp(2)) ≈ 0.881`. The decision is allow at threshold 0.9 and deny at threshold 0.85. `TestPolicyOrderingAndModelTenantRules` also walks a hand-built request through ordered rules. A zero embedding produces the softmax of the bias vector, not NaN. The large-vector fixture ensures normalization and dot products remain finite.

## 4. Interface and tests

Implement the middleware consumed by `server.Deps.Policy`. Use the exchange's parsed request and authenticated principal. Do not log the request body. The named checks are `TestPolicyOrderingAndModelTenantRules`, `TestLinearHeadFixtureParity`, `TestZeroAndLargeEmbeddingsAreFinite`, `TestEmbeddingsFailureFailsClosed`, `TestEveryDenyIsAudited`, and `TestRedactPII`.

| Test | Why it exists | Expected result |
|---|---|---|
| `TestPolicyOrderingAndModelTenantRules` | Protects rule semantics | Stable first matching decision |
| `TestPolicyYAMLLoad` | Checks the versioned operator policy artifact | Loads valid policy and rejects malformed YAML |
| `TestLinearHeadFixtureParity` | Protects Python/Go boundary | Difference at most `1e-6` |
| `TestZeroAndLargeEmbeddingsAreFinite` | Keeps normalization finite at boundary values | Finite scores for zero and large vectors |
| `TestEmbeddingsFailureFailsClosed` | Avoids fail-open behavior | 503 and audit event |
| `TestEveryDenyIsAudited` | Makes both rule and default denials durable | One audit event per deny with the matching rule id |
| `TestRedactPII` | Protects audit privacy | Redacted prompt data |

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Matching `max_tokens` in the opposite direction | `TestPolicyOrderingAndModelTenantRules`; mutant `s01` |
| Reordering rules | `TestPolicyOrderingAndModelTenantRules`; mutant `s02` |
| Using raw logits as probabilities | `TestLinearHeadFixtureParity`; mutant `s03` |
| Allowing an embeddings error | `TestEmbeddingsFailureFailsClosed`; mutant `s04` |
| Returning a deny without recording its rule id | `TestEveryDenyIsAudited` |
| Auditing raw prompt text | `TestRedactPII`; mutant `s05` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `gw.01` | Places the policy middleware in the authenticated request chain. |
| Back | `gw.02` | Supplies the principal used for tenant and model matching. |
| Back | `gw.05` | `route.InferenceRequest` is the parsed request the policy matches on. |
| Forward | `gw.07` | Consumes policy decisions in the route layer. |
| Forward | `MS-agent` | Checks usage restrictions through the public gateway interface. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| ordered rules | signed policy revisions and rollout controls | Auditable change history and staged activation | OPA and Cedar policy evaluation |
| 451 decision | measured false-positive review | Operator feedback and correction loops | Envoy AI Gateway policy examples |
