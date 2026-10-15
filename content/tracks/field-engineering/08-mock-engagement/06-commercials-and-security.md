<!-- ol:module field.06 -->
# Commercials and security review

## Overview

| | |
|---|---|
| **Module** | `field.06` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | docs/engagement/security-questionnaire.md |
| **Tests** | `course/tests/field.06` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Commercials and security review` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start field.06
ol tests field.06
ol check field.06
```

---

## 1. Why now

Commercial and security reviews shape whether a customer can adopt the platform. Answers should be traceable to controls and current artifacts.

## 2. Principles

Answer only what evidence supports. Distinguish implemented controls from planned work. State subprocessors, data flows, retention, access, incident notification, and exceptions plainly.

## 3. Worked example

Use the threat model, image SBOMs, and usage policy to answer how prompts are protected, dependencies tracked, access scoped, and incidents escalated. Link each answer to evidence and owner.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the submitted paths and required evidence exist, so a plausible narrative cannot pass without the reviewable deliverable. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The deliverable records a conclusion without its customer evidence | Another engineer cannot reproduce the sizing or recommendation. | `test_artifact_paths_are_declared` |
| 2 | An assumption is written as a measured fact | The engagement plan can promise capacity or security the system has not demonstrated. | `test_artifact_paths_are_declared` |
| 3 | The next owner and decision gate are missing | Work stalls at handoff even when the technical work is complete. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `field.05` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.17` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.18` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `ethics.05` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [Mock engagement sequence](README.md) | Carry the evidence and open decisions into the next customer-facing artifact in the sequence. |
