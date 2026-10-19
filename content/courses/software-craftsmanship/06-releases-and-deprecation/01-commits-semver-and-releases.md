# Commits, semver, and releases

By the time the system is ready for its first stable release, "CI passed" is no longer enough of an answer to "what did we ship?" A release has to be reproducible and auditable: anyone, months later, should be able to name the exact source that was released, see that it was tested, and read what changed in terms of its effect on them. Three tools make that possible. Conventional commits make intent searchable, semantic versioning communicates compatibility, and a changelog gives operators an upgrade path.

This section prepares `v1.0.0` of the system and builds the small amount of automation that keeps the release honest. The work produces three artifacts: [`CHANGELOG.md`](../../../../course/solutions/entry/CHANGELOG.md); a release workflow, [`.github/workflows/release.yml`](../../../../course/solutions/entry/.github/workflows/release.yml), whose single job `verify-tagged-commit` checks that the tag and the changelog agree; and [`docs/releases/v1.0.0.md`](../../../../course/solutions/docs/releases/v1.0.0.md), the evidence that records the annotated tag and what it points at. The section has no hard prerequisites, though it builds directly on the conventional history and CI gate from [The repository and its CI gate](../05-code-review-and-ci/01-your-repo-and-ci-gate.md).

Before you read the workflow, make a list of what could go wrong between "CI passed" and "this is what we shipped". Each line of the solution answers one item on that list.

> **Learning objectives**
> - State the semver rules for major, minor, and patch versions and what 1.0.0 promises.
> - Explain why a release is an annotated, immutable tag and how to record the commit it resolves to.
> - Build a tag-triggered workflow that verifies the checked-out commit is the tagged commit and that the changelog has a section for the version.
> - Write a changelog grouped by user effect, and a rollback procedure that keeps tags immutable.

## Versions as promises

A release tag points to an immutable commit. The version number on it is a promise about compatibility. Semantic versioning, `MAJOR.MINOR.PATCH`, ties each part to the public interface: breaking API changes require a major version, additive compatible changes a minor version, and compatible fixes a patch. For this system, the public interface is the HTTP surface, the C ABI, and the file formats.

Version 1.0.0 is a particular promise. It declares the first stable public API, and from then on a breaking change needs 2.0.0. That is why releasing from history rather than from memory matters: when commit headers say `feat`, `fix`, or `!`, the next version and the changelog are a function of the commits since the last tag.

## Preparing v1.0.0 by hand

Walk through a release before automating any of it. Start from a clean tree. Group the changes since the beginning by their effect on users rather than by commit, and compare that changelog against the commit history to make sure nothing is missing. Validate that CI is green at the commit you intend to tag. Record the migration prerequisites and the rollback instructions. Then create the annotated tag and write down the evidence: the tag name, the commit it targets, and the CI result, so an operator can verify which source was released.

## The artifacts

The release produces three files:

| File | Holds |
|---|---|
| `CHANGELOG.md` | one section per version, `## [X.Y.Z] - date`, grouped by user effect |
| `.github/workflows/release.yml` | a workflow triggered only by version tags that checks the tag and the changelog agree |
| `docs/releases/v1.0.0.md` | the evidence: tag, target commit, CI result, changelog section, rollback, and the procedure |

## Building it step by step

Five observations shape the design.

1. **"What did we release?" must have one answer.** A branch moves, and a tag can be moved too unless policy forbids it. So the release is an annotated tag, and the evidence records the commit it resolves to (`git rev-parse v1.0.0^{commit}`).
2. **The thing tested must be the thing tagged.** So CI runs on the tag push, and the workflow verifies that the checked-out commit is the tag's commit.
3. **Users read the changelog, not the commits.** So the changelog groups changes by effect (Added, Compatibility, Verification), and the workflow refuses a tag without a matching section.
4. **The version number is a promise.** 1.0.0 declares the first stable public API. So the changelog states which compatibility windows 1.0.0 establishes.
5. **Releases are immutable.** Fixing a bad release by moving its tag would make the evidence lie. So rollback is "deploy the previous tag", and correction is a new patch version.

### The workflow

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

`${RELEASE_TAG#v}` strips the leading `v`, because tags carry it and changelog headings do not. `grep -F` matches the brackets literally. The tag pattern `v*.*.*` keeps non-version tags from starting a release. The full workflow is [`release.yml`](../../../../course/solutions/entry/.github/workflows/release.yml).

### The changelog and the evidence

The changelog section for the release looks like this:

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

The Compatibility section is what makes 1.0.0 a semver promise: it points to the policy that says how long v1 is kept, which is the subject of [Deprecation policy](02-deprecation-policy.md).

The evidence file, [`v1.0.0.md`](../../../../course/solutions/docs/releases/v1.0.0.md), is a table (version, tag, target, source state, CI, changelog, rollback) plus a five-step procedure: confirm the changelog; confirm a clean tree and green CI; run `git tag -a v1.0.0 -m "Release v1.0.0"` and push; record the resolved commit and the workflow run URL; and publish only after the tag workflow succeeds. Its rollback row reads: "Revert the release deployment to the previous tagged commit. Keep the tag immutable; issue a patch version to correct a published release."

### Why this design

The workflow **verifies on the tag rather than trusting the branch**. A green branch build says nothing about a tag created later from a different commit, and the HEAD-equals-tag check closes that gap. The release uses **annotated tags**, which carry a tagger, a date, and a message, so the tag itself is evidence; lightweight tags are just pointers. The changelog is enforced by **a grep, not a changelog tool**: one line enforces the one rule that matters, a section per version, whereas a generator working from commits would produce commit-level noise instead of user effects.

There is a deliberate tradeoff. The workflow verifies; it does not publish. Publication stays a human step after the check, which is slower but leaves a person accountable for each release.

## What can go wrong

A correct release handles each of these cases:

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the changelog, the release workflow, and the release evidence exist at the declared paths | artifact | the release record and its automation are reviewable files |
| the workflow runs on pushed version tags, not branches | review | a branch build must never pass for a release |
| the changelog has a section for the tagged version | review | release notes and source version cannot diverge |
| the evidence names the exact commit and the CI run | review | anyone can verify which source was released |
| a rollback path that keeps tags immutable | review | a bad release is fixed forward with a patch version |

Three pitfalls account for most failures. If the tag evidence omits the exact release commit or the CI result, a reviewer cannot establish which source tree was released; the evidence therefore records `git rev-parse v1.0.0^{commit}` and the workflow run URL. If the workflow runs against a branch instead of the pushed version tag, a successful branch build can be mistaken for a published release; the `on: push: tags: ["v*.*.*"]` trigger and the HEAD-equals-tag check prevent it.

> **Warning:** If the changelog version does not match the tag, the release notes and the source version diverge. The `grep -F "## [${RELEASE_TAG#v}]"` step fails the job before that can happen.

## Where this leads

This section rests on [The repository and its CI gate](../05-code-review-and-ci/01-your-repo-and-ci-gate.md), whose conventional commits and CI gate must be green on the release commit, and pairs with [Architecture decision records](../../diagramming/02-documentation-writing/01-architecture-decision-records.md) in *Diagramming and Documentation*: ADRs record decisions, and the changelog records their user-visible effect. Next, [Deprecation policy](02-deprecation-policy.md) writes the policy that the changelog's Compatibility section points to.

### Further reading

- [Semantic Versioning 2.0.0](https://semver.org/): the exact rules for major, minor, and patch.
- [Keep a Changelog](https://keepachangelog.com/): the heading format the changelog follows.

## Key terms

- **Semantic versioning (semver)**: `MAJOR.MINOR.PATCH`, where major signals breaking changes, minor compatible additions, and patch compatible fixes.
- **Annotated tag**: a git tag object with a tagger, date, and message, pointing at a commit.
- **`^{commit}`**: git's peel syntax, resolving a tag object to the commit it points at.
- **Changelog**: a per-version record of changes grouped by their effect on users.
- **Release evidence**: a record of the tag, its target commit, the CI result, and the rollback path for one release.

## Summary

- A version number is a promise about compatibility; 1.0.0 declares the first stable public API.
- Release from history, not from memory: conventional commits make the next version and the changelog a function of the commits since the last tag.
- The thing tested must be the thing tagged; the release workflow runs on the tag and checks HEAD against it.
- The changelog is for users, grouped by effect, and the workflow refuses a tag without a matching section.
- Tags are immutable; a bad release is rolled back to the previous tag and corrected with a patch version.

## Review questions

1. Why does the workflow compare `HEAD` with `${{ github.ref }}^{commit}` rather than just using `HEAD`?
   <details><summary>Answer</summary>It asserts that the checkout is exactly the tagged commit. `^{commit}` peels the annotated tag object to its commit; without the check, a misconfigured checkout could test a different tree.</details>
2. A release has a bug. Why issue 1.0.1 instead of moving the `v1.0.0` tag?
   <details><summary>Answer</summary>Anyone who fetched or deployed v1.0.0 would silently have a different source than the tag now names, and the evidence file would be false. Immutable tags keep every record true.</details>
3. What version must a release that removes API v1 carry?
   <details><summary>Answer</summary>A new major version (2.0.0): removing a public interface is a breaking change.</details>
4. Why does the workflow have only `contents: read` permission?
   <details><summary>Answer</summary>It only verifies the tag and the changelog; it does not publish anything. Publication is a deliberate human step after the check, so the workflow needs no write access.</details>
