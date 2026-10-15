<!-- ol:module craft.17 -->
# Threat model (STRIDE on your DFD)

## Overview

| | |
|---|---|
| **Module** | `craft.17` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/THREAT_MODEL.md |
| **Tests** | `course/tests/craft.17` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Threat model (STRIDE on your DFD)` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.17
ol tests craft.17
ol check craft.17
```

---

## 1. Why now

The system crosses user, service, storage, and tool boundaries. A threat model identifies where an attacker can act and which controls must prevent or detect it.

## 2. Principles

Start from a data-flow diagram. For each boundary, list assets, actors, threats, existing controls, residual risk, and an owner. STRIDE is a prompt for analysis, not a substitute for evidence.

## 3. Worked example

Trace a retrieved web page into the agent tool loop. Consider spoofed source, malicious instructions, disclosure through tool output, and unauthorized state change. Tie mitigations to allowlists, taint-aware gates, audit logs, and tests.

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
| Back | `craft.09` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.19` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
