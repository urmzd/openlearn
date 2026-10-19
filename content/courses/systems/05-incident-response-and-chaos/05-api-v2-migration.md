# Drill: API v2 migration

## Overview

| | |
|---|---|
| **Module** | `ops.05` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/ApiV2Migration.md`, written out with its reasoning in section 5 |
| **Needs** | craft.14 |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- During the migration window one deployment serves v1 and v2 on the same paths; the client picks with `X-TL-API-Version`, and every response echoes the version it used.
- v2 breaks some clients in three ways: required `usage.prompt_tokens_details.cached_tokens`, `/v1/completions` deprecated with `Deprecation` and `Sunset` headers, and split error codes.
- Evidence is **per version**: 4xx, 5xx, disconnects, and token accounting in the usage ledger by `api_version`, so a v2 regression is not averaged away by v1 traffic.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

A simultaneous API rollout can break clients even when the new engine is healthy. Version-specific evidence is needed to distinguish protocol errors from service failures.

## 2. Principles

Keep v1 available while validating v2. Route by an explicit version, compare success and latency by version, and define a rollback threshold before increasing the canary.

## 3. Worked example

Send synthetic v1 and v2 streams through the gateway. Verify both OpenAPI suites and usage-ledger totals. Expand the v2 cohort in steps while watching 4xx, 5xx, disconnects, and token accounting.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and preconditions | who owns the change; both OpenAPI suites (v1 and v2) pass through the gateway; the ledger records `api_version` | both versions must work before any client moves |
| Cohorts and routing | which clients send `X-TL-API-Version: 2`, in which order, and the default version during the window | a controlled cohort limits the blast radius |
| Canary gate | per-version 4xx, 5xx, stream disconnects, latency, and token totals compared with v1 | version-specific signals separate protocol errors from service failures |
| Rollback | the trigger and the action: route the cohort back to v1; v1 stays served until the sunset date | the old route must be restorable |
| Sunset | the `Deprecation` and `Sunset` dates, the 410 `api_version_sunset` after it, and the ledger query that shows remaining v1 traffic | nobody is cut off by surprise |

## 5. The solution, step by step

### 5.1 From first principles

1. **What can break?** Not the engine: the contract. v2 adds a required usage field, deprecates `/v1/completions`, and splits error codes (`rate_limit_exceeded` into `requests_limit_exceeded`, `tokens_limit_exceeded`, `queue_full`; `model_draining` from `no_capacity`). A client that parses strictly or switches on error codes breaks while the server is healthy.
2. **So keep both.** The contract serves v1 and v2 on the same paths during the window, selected by `X-TL-API-Version`; any other value is 400 `unsupported_api_version`. Rollback for a cohort is just sending `1` again.
3. **Measure per version.** Every response echoes `X-TL-API-Version` and the usage ledger records `api_version`, so 4xx, 5xx, disconnects, latency, and token totals can be split. A rise in v2 4xx with flat v1 is a client-compatibility problem, not an outage.
4. **Token accounting is part of correctness.** v2 reports `cached_tokens`; the ledger's totals for the same synthetic workload must agree between versions, or billing breaks silently.
5. **Close the window deliberately.** v1 responses carry `Deprecation` and `Sunset`; after the sunset, v1 answers 410 `api_version_sunset`. The ledger shows who still sends v1 before that date.

### 5.2 The model runbook

`docs/runbooks/ApiV2Migration.md`:

````markdown
# Runbook: API v2 migration

Owner: the gateway on-call (`<system>-gateway`). Change ticket: the craft.14 migration.

## Preconditions

- The v1 and v2 OpenAPI suites pass through the gateway NodePort
  (`openai-subset.v1.yaml`, `openai-subset.v2.yaml`).
- Every response carries `X-TL-API-Version`; the usage ledger has `api_version`.
- v1 responses carry `Deprecation` and `Sunset`; the sunset date is announced.

## Canary

1. Send synthetic v1 and v2 streams (`X-TL-API-Version: 1` and `2`) through the gateway;
   compare per-version 4xx, 5xx, disconnects, TTFT, and ledger token totals.
2. Move the first client cohort to v2. Gate (held 1 hour): v2 5xx and disconnect
   rates within the v1 baseline; no unexpected 4xx codes; ledger totals for the
   cohort match its v1 totals within rounding.
3. Expand cohorts in steps, repeating the gate.

## Rollback

Trigger: a gate threshold missed, or a cohort reports a parse failure. Action:
the cohort sends `X-TL-API-Version: 1` (or the gateway default stays v1); v1 is
served until the sunset date. Record the trigger and the client.

## Verification and sunset

Before the sunset date, the ledger query of v1 requests by key shows no remaining
v1 traffic, or each remaining client has acknowledged the date. After it, v1
requests answer 410 `api_version_sunset`; watch for them for a week.
````

### 5.3 Why this solution

- **Version routing by header** gives per-request rollback without a deploy: the same pods serve both versions, so moving a cohort back is instant.
- **Per-version signals** make the decision honest: averaged over all traffic, a 5% failure in a 10% cohort is a 0.5% blip.
- **Why not a hard cutover?** Every client would have to change in the same minute, and any strict parser or error-code switch would break at once with no path back.

### 5.4 Explain it back

1. Why can v2 break a client while the server is perfectly healthy?
   <details><summary>Answer</summary>The breakage is in the contract: a newly required field, new error codes, and deprecated endpoints. A client that parses strictly or branches on `rate_limit_exceeded` fails on a correct v2 response.</details>
2. How does a cohort roll back without a deploy?
   <details><summary>Answer</summary>It sends `X-TL-API-Version: 1`. The same deployment serves both versions during the window, so the version is chosen per request.</details>
3. Why compare ledger token totals between versions?
   <details><summary>Answer</summary>v2 changes the usage object (`cached_tokens`). If accounting differs for the same workload, billing and quotas drift silently; the ledger is where that shows.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Advancing the cohort before the rollback trigger is written | a regression is argued about instead of acted on | the gate thresholds and the trigger are in the runbook before the first cohort moves |
| 2 | Watching total error rate instead of per-version rates | a v2-only regression is diluted by v1 traffic | every signal is split by `X-TL-API-Version` and the ledger's `api_version` |
| 3 | Sunsetting v1 without checking who still uses it | clients get 410 `api_version_sunset` without warning | the ledger query of remaining v1 traffic gates the sunset |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.14` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `ops.04` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
