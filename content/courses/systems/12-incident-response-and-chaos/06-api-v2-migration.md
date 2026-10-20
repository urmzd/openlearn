# The API v2 migration

A simultaneous API rollout can break clients even when the new engine is healthy. The server returns exactly what the v2 contract says, and a client that parses strictly or switches on error codes fails anyway. This drill turns [Interface migration: API v1 to v2](../../software-engineering/08-maintenance/02-api-v1-to-v2.md) in *Software Engineering* into a runbook, and its central lesson is that you need version-specific evidence to tell protocol errors from service failures.

During the migration window one deployment serves v1 and v2 on the same paths; the client picks with `X-TL-API-Version`, and every response echoes the version it used. v2 breaks some clients in three ways: a required `usage.prompt_tokens_details.cached_tokens`, `/v1/completions` deprecated with `Deprecation` and `Sunset` headers, and split error codes. The procedure follows the same discipline as [The KV format v2 migration](05-kv-v2-migration.md): preconditions, a canary with a gate written in advance, and a rollback that is always available.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure, ending with the model runbook `docs/runbooks/ApiV2Migration.md`. Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Name the three ways v2 breaks clients while the server stays healthy.
> - Roll a client cohort back without a deploy, using the version header.
> - Split every signal (4xx, 5xx, disconnects, latency, token totals) by API version so a cohort regression is not averaged away.
> - Plan a deliberate sunset gated on the usage ledger.

## Principles

Keep v1 available while validating v2. Route by an explicit version, compare success and latency by version, and define a rollback threshold before increasing the canary. Evidence is per version: 4xx, 5xx, disconnects, and token accounting in the usage ledger by `api_version`, so a v2 regression is not averaged away by v1 traffic.

## The drill scenario

Send synthetic v1 and v2 streams through the gateway. Verify both OpenAPI suites and usage-ledger totals. Expand the v2 cohort in steps while watching 4xx, 5xx, disconnects, and token accounting.

## What a correct runbook must cover

A runbook for this change is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and preconditions | who owns the change; both OpenAPI suites (v1 and v2) pass through the gateway; the ledger records `api_version` | both versions must work before any client moves |
| Cohorts and routing | which clients send `X-TL-API-Version: 2`, in which order, and the default version during the window | a controlled cohort limits the blast radius |
| Canary gate | per-version 4xx, 5xx, stream disconnects, latency, and token totals compared with v1 | version-specific signals separate protocol errors from service failures |
| Rollback | the trigger and the action: route the cohort back to v1; v1 stays served until the sunset date | the old route must be restorable |
| Sunset | the `Deprecation` and `Sunset` dates, the 410 `api_version_sunset` after it, and the ledger query that shows remaining v1 traffic | nobody is cut off by surprise |

## Deriving the procedure

### Reasoning from first principles

What can break? Not the engine: the contract. v2 adds a required usage field, deprecates `/v1/completions`, and splits error codes (`rate_limit_exceeded` into `requests_limit_exceeded`, `tokens_limit_exceeded`, `queue_full`; `model_draining` from `no_capacity`). A client that parses strictly or switches on error codes breaks while the server is healthy.

So keep both. The contract serves v1 and v2 on the same paths during the window, selected by `X-TL-API-Version`; any other value is 400 `unsupported_api_version`. Rollback for a cohort is just sending `1` again.

Measure per version. Every response echoes `X-TL-API-Version` and the usage ledger records `api_version`, so 4xx, 5xx, disconnects, latency, and token totals can be split. A rise in v2 4xx with flat v1 is a client-compatibility problem, not an outage.

Token accounting is part of correctness. v2 reports `cached_tokens`; the ledger's totals for the same synthetic workload must agree between versions, or billing breaks silently.

Close the window deliberately. v1 responses carry `Deprecation` and `Sunset`; after the sunset, v1 answers 410 `api_version_sunset`. The ledger shows who still sends v1 before that date.

### The model runbook

The procedure written out as `docs/runbooks/ApiV2Migration.md`:

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

### Why this solution

Version routing by header gives per-request rollback without a deploy: the same pods serve both versions, so moving a cohort back is instant. Per-version signals make the decision honest: averaged over all traffic, a 5% failure in a 10% cohort is a 0.5% blip. A hard cutover would throw both advantages away. Every client would have to change in the same minute, and any strict parser or error-code switch would break at once with no path back.

## What can go wrong

Three mistakes turn a controlled migration into an argument or a surprise:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Advancing the cohort before the rollback trigger is written | a regression is argued about instead of acted on | the gate thresholds and the trigger are in the runbook before the first cohort moves |
| Watching total error rate instead of per-version rates | a v2-only regression is diluted by v1 traffic | every signal is split by `X-TL-API-Version` and the ledger's `api_version` |
| Sunsetting v1 without checking who still uses it | clients get 410 `api_version_sunset` without warning | the ledger query of remaining v1 traffic gates the sunset |

## Where this leads

The contract this runbook relies on is [Interface migration: API v1 to v2](../../software-engineering/08-maintenance/02-api-v1-to-v2.md), and the procedure builds on [The KV format v2 migration](05-kv-v2-migration.md). The next drill, [A breaking dependency upgrade](07-dependency-upgrade.md), applies the same evidence-first approach to a change you did not write yourself. To compare this inject with the adjacent drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **Version header**: `X-TL-API-Version`, which selects v1 or v2 per request and is echoed on every response.
- **Cohort**: a controlled group of clients moved to the new version together.
- **Per-version signal**: a metric split by API version so a small cohort's regression stays visible.
- **Sunset**: the announced date after which v1 answers 410 `api_version_sunset`.

## Summary

- During the window one deployment serves v1 and v2 on the same paths, chosen per request by `X-TL-API-Version`, and every response echoes the version used.
- v2 breaks some clients through a required `cached_tokens` field, the deprecation of `/v1/completions`, and split error codes, all while the server is healthy.
- Evidence is per version: 4xx, 5xx, disconnects, latency, and ledger token totals by `api_version`.
- Rollback is a header change, not a deploy; the sunset is gated on the ledger showing no remaining v1 traffic.

## Review questions

1. Why can v2 break a client while the server is perfectly healthy?
   <details><summary>Answer</summary>The breakage is in the contract: a newly required field, new error codes, and deprecated endpoints. A client that parses strictly or branches on `rate_limit_exceeded` fails on a correct v2 response.</details>
2. How does a cohort roll back without a deploy?
   <details><summary>Answer</summary>It sends `X-TL-API-Version: 1`. The same deployment serves both versions during the window, so the version is chosen per request.</details>
3. Why compare ledger token totals between versions?
   <details><summary>Answer</summary>v2 changes the usage object (`cached_tokens`). If accounting differs for the same workload, billing and quotas drift silently; the ledger is where that shows.</details>
4. A 10% cohort on v2 sees a 5% failure rate. What does the overall error rate show, and why does that matter?
   <details><summary>Answer</summary>About 0.5%, a blip that looks like noise. Only the per-version split shows that one in twenty v2 requests is failing, which is why the gate is defined per version.</details>
