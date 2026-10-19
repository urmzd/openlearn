# A breaking dependency upgrade

A major dependency release can change runtime behaviour without changing any source-level contract. Your code still compiles, the API you call still has the same signature, and yet a default, a timeout, a serialization detail, or the performance profile is different. This drill practices containing such a change and keeping a reproducible rollback, turning the practice of [Dependency upgrades](../../software-craftsmanship/07-maintenance/03-dependency-upgrades.md) in *Software Craftsmanship* into an operational runbook.

The discipline is the one from [The API v2 migration](05-api-v2-migration.md): gates written before the change, evidence that measures behaviour, and a rollback that needs no thought. What is new here is isolation. Because the change comes from someone else's code, you control it by changing exactly one thing at a time.

The section states the principles and the drill scenario, lists what a correct runbook must cover, and derives the procedure, ending with the model runbook `docs/runbooks/DependencyUpgrade.md`. Try to write the procedure yourself before you read the solution.

> **Learning objectives**
> - Explain why a passing build is weak evidence for a dependency upgrade, and what evidence replaces it.
> - Isolate a change to one dependency and inspect its resolved graph for transitive bumps.
> - Gate a deploy on contract suites and a like-for-like load comparison against the last known good image.
> - Keep rollback a redeploy of a tagged image rather than a rebuild.

## Principles

Inspect the resolved dependency graph and release notes. Change one dependency, keep lockfiles, run the focused contract suite, and compare the same load profile before deployment. Change one dependency at a time, with the lockfile committed, so any difference has one cause and the rollback is a revert of one diff. Deploy only after the gates pass, and keep the last known good image tagged so rollback is a redeploy, not a rebuild.

## The drill scenario

Upgrade the designated major version in a scratch branch. Build, run native and contract tests, then compare latency and error distributions with the last known good image. Deploy only after the gates pass.

## What a correct runbook must cover

A runbook for this change is complete when it answers five questions:

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and scope | the one dependency, its old and new version, and the release notes' breaking changes | a single cause for any difference |
| Reproducible build | the lockfile diff, a clean build, and the image digest | the same input gives the same artifact |
| Gates | native and contract test suites, then the saved load profile against the last known good image, with thresholds | behaviour, not just compilation |
| Rollback | the last known good image tag and the revert commit | restoring service needs no rebuild |
| Verification | post-deploy SLO burn and error rates compared with before | done means measured |

## Deriving the procedure

### Reasoning from first principles

What could a dependency change break? Anything it touches at runtime: defaults, timeouts, serialisation, performance. Compilation success says almost nothing, so the evidence must be tests that exercise contracts plus a load comparison.

Isolate the cause. Upgrading several dependencies at once makes any regression ambiguous. Change one dependency per change, with its resolved graph inspected (`go mod graph`, `cargo tree`, `uv pip tree`) so transitive bumps are visible.

Make it reproducible. The lockfile pins the whole graph; committing it means the canary, the deploy, and the rollback all use exactly the tested set.

Compare like with like. Run the same load profile (rate, mix, seed, duration) against the last known good image and the candidate, comparing latency and error distributions, not single numbers.

Rollback is a redeploy. Keep the previous image tag: reverting the commit restores the source, and the image tag restores production at once.

### The model runbook

The procedure written out as `docs/runbooks/DependencyUpgrade.md`:

````markdown
# Runbook: dependency upgrade

Owner: the service owner of the component being upgraded.

## Preconditions

- One dependency, old and new versions named; release notes read for breaking changes.
- The resolved graph diff reviewed (`go mod graph`, `cargo tree`, `uv pip tree`).
- The last known good image tag recorded.

## Gates (on a branch)

1. Update the dependency and its lockfile only; build cleanly from scratch.
2. Run the native and contract suites (OpenAPI conformance, ABI and format tests).
3. Run the saved load profile against the last known good image and the
   candidate; compare TTFT, TPOT, end-to-end latency, and error rate
   distributions. Gate: p95 within 5% and error rate not higher.

## Deploy and rollback

Deploy the candidate image. Rollback trigger: a burn-rate alert, or error rate
above the pre-deploy baseline for 10 minutes. Action: redeploy the last known
good tag, then revert the commit.

## Verification

SLO burn below 1 for 24 hours after the deploy; the decision and the load
comparison attached to the change.
````

### Why this solution

One change per deploy keeps attribution and rollback trivial. A load comparison against the last good image catches performance and timeout changes that functional tests miss. And the alternative, waiting for production to tell you, is expensive: a regression found under real traffic has already spent error budget, while the same load profile finds it on a branch.

## What can go wrong

Three shortcuts account for most painful upgrades:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Several dependencies bumped together | a regression with many possible causes; rollback undoes unrelated fixes | one dependency and its lockfile per change |
| Only compile and unit tests run | runtime behaviour or latency changes reach production | contract suites and the saved load profile gate the deploy |
| No recorded last known good image | rollback requires a rebuild under pressure | the tag is a precondition |

## Where this leads

The practice this runbook relies on is [Dependency upgrades](../../software-craftsmanship/07-maintenance/03-dependency-upgrades.md), and it follows [The API v2 migration](05-api-v2-migration.md). When an upgrade or any other change slips through and costs latency, [A performance regression bisect](07-performance-regression.md) finds the commit responsible. To compare this inject with the adjacent drills and reuse the same containment and verification fields, see the [chapter opening](README.md).

## Key terms

- **Lockfile**: the file that pins the whole resolved dependency graph so every build uses the same set.
- **Resolved graph**: the full set of direct and transitive dependencies actually selected, shown by `go mod graph`, `cargo tree`, or `uv pip tree`.
- **Contract suite**: tests that exercise behaviour at an interface (OpenAPI conformance, ABI and format tests).
- **Last known good image**: the tagged artifact that already ran in production, kept so rollback is a redeploy.

## Summary

- A dependency upgrade can change runtime behaviour with no source-level change, so evidence must be behavioural: contract tests and the same load profile.
- Change one dependency at a time, lockfile committed, so a difference has one cause and rollback reverts one diff.
- Compare distributions under a like-for-like load profile against the last known good image, with thresholds set in advance.
- Keep the last known good image tagged so rollback is a redeploy, not a rebuild.

## Review questions

1. Why upgrade one dependency at a time?
   <details><summary>Answer</summary>So any behavioural difference has exactly one cause and the rollback is reverting one diff.</details>
2. Why is a passing build not enough evidence?
   <details><summary>Answer</summary>A dependency can keep its API and change defaults, timeouts, or performance; only tests that exercise behaviour and a load comparison see that.</details>
3. Why keep the last known good image tag?
   <details><summary>Answer</summary>Rollback then means redeploying an artifact that already ran in production, which is faster and safer than rebuilding the old source.</details>
4. Why inspect the resolved graph and not just the one line you changed?
   <details><summary>Answer</summary>Upgrading one direct dependency can pull in new versions of its transitive dependencies. The graph diff makes those visible, so "one change" really is one change.</details>
