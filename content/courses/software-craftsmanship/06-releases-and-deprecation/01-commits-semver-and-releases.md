# Commits, semver, releases (v1.0.0)

## Overview

| | |
|---|---|
| **Module** | `craft.11` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | [`CHANGELOG.md`](../../../../course/solutions/entry/CHANGELOG.md), [`.github/workflows/release.yml`](../../../../course/solutions/entry/.github/workflows/release.yml) (job `verify-tagged-commit`), and [`docs/releases/v1.0.0.md`](../../../../course/solutions/docs/releases/v1.0.0.md) (the annotated tag evidence) |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Commits, semver, releases (v1.0.0)` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 explain what a release must prove and walk through preparing v1.0.0. Section 4 states the three artifacts. Section 5 walks through each of them and explains why every line is there. Before reading the workflow, list what could go wrong between "CI passed" and "this is what we shipped".

---

## 1. Why now

The P11 release has to be reproducible and auditable. Conventional commits make intent searchable; semantic versioning communicates compatibility; a changelog gives operators an upgrade path.

## 2. Principles

A release tag points to an immutable commit. Version changes follow public compatibility: breaking API changes require a major version, additive compatible changes a minor version, and compatible fixes a patch.

## 3. Worked example

Prepare v1.0.0 from a clean tree. Group changes by user effect, validate CI at the tagged commit, and record the migration prerequisites and rollback instructions. Compare the changelog with the commit history. The release evidence records the annotated tag name, target commit, and CI result so an operator can verify which source was released.

## 4. The artifacts

| File | Holds |
|---|---|
| `CHANGELOG.md` | one section per version, `## [X.Y.Z] - date`, grouped by user effect |
| `.github/workflows/release.yml` | a workflow triggered only by version tags that checks the tag and the changelog agree |
| `docs/releases/v1.0.0.md` | the evidence: tag, target commit, CI result, changelog section, rollback, and the procedure |

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the changelog, the release workflow, and the release evidence exist at the declared paths | artifact | the release record and its automation are reviewable files |
| the workflow runs on pushed version tags, not branches | review | a branch build must never pass for a release |
| the changelog has a section for the tagged version | review | release notes and source version cannot diverge |
| the evidence names the exact commit and the CI run | review | anyone can verify which source was released |
| a rollback path that keeps tags immutable | review | a bad release is fixed forward with a patch version |

## 5. The solution, step by step

### 5.1 From first principles

1. **"What did we release?" must have one answer.** A branch moves; a tag can be moved too unless policy forbids it. So the release is an annotated tag, and the evidence records the commit it resolves to (`git rev-parse v1.0.0^{commit}`).
2. **The thing tested must be the thing tagged.** So CI runs on the tag push, and the workflow verifies the checked-out commit is the tag's commit.
3. **Users read the changelog, not the commits.** So the changelog groups changes by effect (Added, Compatibility, Verification) and the workflow refuses a tag without a matching section.
4. **The version number is a promise.** 1.0.0 declares the first stable public API; from here a breaking change needs 2.0.0. So the changelog states which compatibility windows 1.0.0 establishes.
5. **Releases are immutable.** Fixing a bad release by moving its tag would make the evidence lie. So rollback is "deploy the previous tag", and correction is a new patch version.

### 5.2 The workflow

```yaml
on:
  push:
    tags: ["v*.*.*"]          # only version tags trigger a release run

permissions:
  contents: read              # verification needs no write access

jobs:
  verify-tagged-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0      # full history, so the tag can be resolved
      - name: Verify tag points to this commit
        shell: bash
        run: |
          set -euo pipefail
          # ^{commit} peels an annotated tag to the commit it points at
          test "$(git rev-parse HEAD)" = "$(git rev-parse '${{ github.ref }}^{commit}')"
      - name: Require matching changelog section
        shell: bash
        env:
          RELEASE_TAG: ${{ github.ref_name }}
        run: |
          set -euo pipefail
          # v1.0.0 -> "## [1.0.0]"; grep fails the job if the section is missing
          grep -F "## [${RELEASE_TAG#v}]" CHANGELOG.md
```

`${RELEASE_TAG#v}` strips the leading `v`, because tags carry it and changelog headings do not. `grep -F` matches the brackets literally. The tag pattern `v*.*.*` keeps non-version tags from starting a release. Full workflow: [`release.yml`](../../../../course/solutions/entry/.github/workflows/release.yml).

### 5.3 The changelog and the evidence

```markdown
## [1.0.0] - 2026-10-09

First stable release of the TinyLLM learning platform and reference implementation.

### Added
- OpenAI-subset completion and embedding APIs at the gateway and engine.
- ...

### Compatibility
- This release establishes the v1 API and KV envelope compatibility windows described in [the deprecation policy](../docs/DEPRECATION.md).

### Verification
- Release source is the commit carrying the annotated `v1.0.0` tag.
- The tag is created only after the required CI workflow passes on that commit.
```

The Compatibility section is what makes 1.0.0 a semver promise: it points to the policy that says how long v1 is kept. The evidence file ([`v1.0.0.md`](../../../../course/solutions/docs/releases/v1.0.0.md)) is a table (version, tag, target, source state, CI, changelog, rollback) plus a five-step procedure: confirm the changelog, confirm a clean tree and green CI, `git tag -a v1.0.0 -m "Release v1.0.0"` and push, record the resolved commit and the workflow run URL, publish only after the tag workflow succeeds. Its rollback row reads: "Revert the release deployment to the previous tagged commit. Keep the tag immutable; issue a patch version to correct a published release."

### 5.4 Why this solution

- **Verify on the tag, not trust the branch.** A green branch build says nothing about a tag created later from a different commit; the workflow closes that gap.
- **Annotated tags.** They carry a tagger, a date, and a message, so the tag itself is evidence; lightweight tags are just pointers.
- **A grep, not a changelog tool.** One line enforces the one rule that matters (a section per version); a generator from commits would produce commit-level noise instead of user effects.
- **The tradeoff.** The workflow verifies, it does not publish; publication stays a human step after the check, which is slower but leaves a person accountable for each release.

### 5.5 Explain it back

1. Why does the workflow compare `HEAD` with `${{ github.ref }}^{commit}` rather than just using `HEAD`?
   <details><summary>Answer</summary>It asserts that the checkout is exactly the tagged commit. `^{commit}` peels the annotated tag object to its commit; without the check, a misconfigured checkout could test a different tree.</details>
2. A release has a bug. Why issue 1.0.1 instead of moving the `v1.0.0` tag?
   <details><summary>Answer</summary>Anyone who fetched or deployed v1.0.0 would silently have a different source than the tag now names, and the evidence file would be false. Immutable tags keep every record true.</details>
3. What version must a release that removes API v1 carry?
   <details><summary>Answer</summary>A new major version (2.0.0): removing a public interface is a breaking change.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The tag evidence omits the exact release commit or CI result | A reviewer cannot establish which source tree was released. | the evidence records `git rev-parse v1.0.0^{commit}` and the workflow run URL |
| 2 | The workflow runs against a branch instead of the pushed version tag | A successful branch build can be mistaken for a published release. | `on: push: tags: ["v*.*.*"]` and the HEAD-equals-tag check |
| 3 | The changelog version does not match the tag | The release notes and source version diverge. | the `grep -F "## [${RELEASE_TAG#v}]"` step fails the job |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.01` | Conventional commits and the CI gate that must be green on the release commit. |
| Back | `craft.02` | ADRs record decisions; the changelog records their user-visible effect. |
| Forward | `craft.12` | The deprecation policy the changelog's Compatibility section points to. |

## Going further

| Resource | What to inspect |
|---|---|
| [Semantic Versioning 2.0.0](https://semver.org/) | The exact rules for major, minor, and patch. |
| [Keep a Changelog](https://keepachangelog.com/) | The heading format the changelog follows. |
