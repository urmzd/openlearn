<!-- ol:module craft.18 -->
# SBOM and supply chain

## Overview

| | |
|---|---|
| **Module** | `craft.18` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | `docs/sbom/engine.spdx.json` and `docs/sbom/gateway.spdx.json`; use [`course/contracts/allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml) as the dependency policy |
| **Tests** | `course/tests/craft.18` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `SBOM and supply chain` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.18
ol tests craft.18
ol check craft.18
```

---

## 1. Why now

A deployment is built from many direct and transitive packages. An SBOM and dependency policy help responders identify exposure and prove what was shipped.

## 2. Principles

Generate inventories from the exact image digest. Pin base images and dependencies, record licenses and provenance, and define a review path for exceptions. `course/contracts/allowed-deps.toml` is the source policy; it is not a generated learner artifact. Regenerate an SPDX document for each released image and retain the image digest with the CI run.

## 3. Worked example

Build each service image, capture its digest, generate an SPDX or CycloneDX SBOM, and compare every dependency with the allowlist. Investigate unpinned transitive dependencies and known vulnerabilities before signing off.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Parses both SPDX 2.3 JSON files, checks package IDs and dependency relationships, and confirms the engine and gateway policy sections exist. This catches malformed inventories and disconnected package rows. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The artifact names a decision without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | `test_artifact_paths_are_declared` |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | `test_artifact_paths_are_declared` |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.17` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
