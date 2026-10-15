<!-- ol:module craft.11 -->
# Commits, semver, releases (v1.0.0)

## Overview

| | |
|---|---|
| **Module** | `craft.11` · practice · docs · Pass 11 · 1 to 2 h |
| **You build** | `CHANGELOG.md`, `.github/workflows/release.yml`, `docs/releases/v1.0.0.md` for the annotated tag evidence |
| **Tests** | `course/tests/craft.11` (named below, with why each exists) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |
| **Milestone** | `MS-P11` |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Commits, semver, releases (v1.0.0)` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to work this chapter

```bash
ol start craft.11
ol tests craft.11
ol check craft.11
```

---

## 1. Why now

The P11 release has to be reproducible and auditable. Conventional commits make intent searchable; semantic versioning communicates compatibility; a changelog gives operators an upgrade path.

## 2. Principles

A release tag points to an immutable commit. Version changes follow public compatibility: breaking API changes require a major version, additive compatible changes a minor version, and compatible fixes a patch.

## 3. Worked example

Prepare v1.0.0 from a clean tree. Group changes by user effect, validate CI at the tagged commit, and record the migration prerequisites and rollback instructions. Compare the changelog with the commit history. The release evidence records the annotated tag name, target commit, and CI result so an operator can verify which source was released.

## 4. The artifact and its check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_artifact_paths_are_declared` | artifact | Confirms the registry declares the changelog, release workflow, and release evidence paths. | Ensures `ol diff craft.11` compares the actual release record and automation. |

## 5. Pitfalls

| # | Pitfall | Symptom | Caught by |
|---|---|---|---|
| 1 | The tag evidence omits the exact release commit or CI result | A reviewer cannot establish which source tree was released. | `test_artifact_paths_are_declared` |
| 2 | The workflow runs against a branch instead of the pushed version tag | A successful branch build can be mistaken for a published release. | `test_artifact_paths_are_declared` |
| 3 | The changelog version does not match the tag | The release notes and source version diverge. | `test_artifact_paths_are_declared` |

## 6. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.01` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |
| Back | `craft.02` | Read this declared prerequisite before producing the artifact; use it to verify the relevant contract or evidence. |

## Going further

| Resource | What to inspect |
|---|---|
| [C4 model and architecture rules](../../../../course/DESIGN.md), section 2.3 | Compare the submitted views with the course system boundary and its process/file interfaces. |
