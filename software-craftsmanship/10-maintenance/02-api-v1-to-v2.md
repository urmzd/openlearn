<!-- ol:module craft.14 -->
# Interface migration: API v1 to v2

## Overview

| | |
|---|---|
| **Module** | `craft.14` · build · rust, go · Pass 11 · 1 to 2 h |
| **You build** | rust/crates/tl-serve/src/api_version.rs, go/gateway/server/apiversion.go · `rust/crates/tl-serve/src/server.rs`, taken over from `L10.9`: every API request selects its version before dispatch, every response echoes it, v2 answers carry `cached_tokens` and the v2 error codes, and a configured window (`[ext.api]` in runtime.toml) adds `Deprecation` and `Sunset` |
| **Tests** | `course/tests/go/craft_14/` (named below, with why each exists) |
| **Needs** | craft.12 · `L10.9` the serve loop you take over |
| **Used by** | ops.05 |
| **Milestone** | `MS-P11` |

## Key Takeaways

- The request header chooses v1 or v2; absent a header, a configured default applies.
- During the overlap, v1 replies include Deprecation and Sunset and every reply echoes its selected version.
- After sunset v1 returns 410, and per-version ledger grouping shows remaining clients.

## How to work this chapter

```bash
ol start craft.14
ol tests craft.14
ol check craft.14
```

---

## 1. Why now

API v2 changes client-visible behavior while v1 clients remain in service. The gateway and engine must route versions deliberately and meter them separately so operators can see migration progress.

## 2. Principles

Keep v1 and v2 contracts explicit. Validate requests against the selected version, preserve error semantics, and never infer a client version from payload fields that can be ambiguous.

## 3. Worked example

The worked request sends `X-TL-API-Version: 2` while the configured default remains `1`; assert the reply echoes `2` before introducing v2 routes while v1 remains default for existing clients. Send a canary cohort to v2, compare request errors, streaming behavior, and usage totals, then expand. Retain a switch back to v1 until all consumers migrate.

## 4. The interface and tests

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleExplicitV2HeaderIsEchoed` | unit | The worked explicit v2 request gets a response marked v2. | Confirms the header-selected contract end to end. |
| `hand_example_explicit_version_overrides_default` | unit | A request header overrides the configured default. | Allows controlled migration cohorts. |
| `rejects_unknown_version_before_dispatch` | boundary | Unknown versions fail before backend work starts. | Avoids ambiguous execution under an unsupported contract. |
| `response_version_is_echoed_and_v1_is_marked_deprecated` | unit | Responses identify the selected version and mark v1 deprecated. | Clients and operators can observe migration state. |
| `TestV1CarriesSunsetHeadersDuringWindow` | unit | v1 replies carry the deprecation window headers. | Gives consumers a concrete removal date. |
| `TestMeterSeesAPIVersionForGrouping` | integration | The middleware passes the selected API version to the downstream meter grouping map. | Operators can separate v1 and v2 usage during the migration. |
| `v1_sunset_is_a_hard_boundary` | boundary | v1 is rejected after sunset. | Prevents indefinite compatibility promises. |
| `TestSunsetAndUnsupportedVersionReturnOpenAIErrorShape` | boundary | Sunset and unsupported-version errors retain the public error shape. | Existing clients can handle failure consistently. |
| `v2_error_names_are_specific_and_v1_codes_are_stable` | unit | v2 errors are specific while v1 codes remain stable. | Preserves compatibility through the overlap period. |
| `cached_token_usage_is_present_only_in_v2` | unit | Exposes cached-token usage only in v2 responses. | Preserves the versioned response contract. |
| `engine_serves_v1_and_v2_side_by_side` | conformance | The engine echoes the selected version, marks v1 and v2 `/v1/completions` deprecated during the window, adds `cached_tokens` to v2 usage, refuses an unknown version, and answers a sunset v1 with 410. | `ol conform openapi:v2 --target engine` and the ops.05 drill run against this serve loop. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | Inferring a version from request contents when the header is absent | The same payload can dispatch under different contracts. | `rejects_unknown_version_before_dispatch` |
| 2 | Returning v1 success after its sunset | Clients continue depending on a contract the service has withdrawn. | `v1_sunset_is_a_hard_boundary` |
| 3 | Combining usage across v1 and v2 | Operators cannot tell whether old clients remain. | `TestMeterSeesAPIVersionForGrouping` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.12` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.02` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `course/contracts/openapi/openai-subset.v2.yaml` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `gw.07` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `L10.9` | The serve loop you take over: version selection runs before its routes, and its usage and error documents take the selected version. |
| Back | `L10.5` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Forward | `ops.05` | This declared call site consumes the artifact; verify its expectations before finalizing. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
