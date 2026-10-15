<!-- ol:module craft.19 -->
# Secrets and authorization review

## Overview

| | |
|---|---|
| **Module** | `craft.19` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/security-review.md |
| **Tests** | `course/tests/craft.19` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Secrets and authorization review` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.19
ol tests craft.19
ol check craft.19
```

---

## 1. Why now

API keys and tenant scopes protect model access and usage data. Review the authorization design as a matrix of principals, actions, and resources, then inspect where secrets can leak.

## 2. Principles

Store only keyed hashes of API secrets, compare credentials in constant time, scope every operation, and audit denies without logging presented secrets or prompt content. Rotate and revoke through a documented path.

## 3. Worked example

Check that an infer-only key cannot call admin routes, tenant A cannot retrieve tenant B usage, and revocation takes effect within the documented cache window. Search source, logs, image layers, and CI artifacts for credential patterns.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the submitted paths and required evidence exist, so a plausible narrative cannot pass without the reviewable deliverable. |

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
| Back | `gw.02` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
