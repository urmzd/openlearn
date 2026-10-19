# Drill: breaking dependency upgrade

## Overview

| | |
|---|---|
| **Module** | `ops.06` · drill · ops, docs · Pass 11 · 1 to 2 h |
| **Solution** | the model runbook `docs/runbooks/DependencyUpgrade.md`, written out with its reasoning in section 5 |
| **Needs** | none |
| **Used by** | no code call site: a drill, one of the Pass 11 operations drills |

## Key Takeaways

- A major dependency release can change runtime behaviour without changing any source-level contract, so the evidence must be behavioural: contract tests and the same load profile.
- Change **one** dependency at a time, with the lockfile committed, so any difference has one cause and the rollback is a revert of one diff.
- Deploy only after the gates pass, and keep the last known good image tagged so rollback is a redeploy, not a rebuild.

## How to read this chapter

This chapter is a worked drill. Sections 1 to 3 state the risk, the principles, and the drill scenario. Section 4 lists what a correct runbook must cover. Section 5 derives the procedure from first principles, gives the model runbook, and explains why each step is the right one. Try to write the procedure yourself from sections 2 and 3 before reading section 5.

---

## 1. Why now

A breaking dependency release can change runtime behavior without changing your source-level contract. This drill practices containment and a reproducible rollback.

## 2. Principles

Inspect the resolved dependency graph and release notes. Change one dependency, keep lockfiles, run the focused contract suite, and compare the same load profile before deployment.

## 3. Worked example

Upgrade the designated major version in a scratch branch. Build, run native and contract tests, then compare latency and error distributions with the last known good image. Deploy only after the gates pass.

## 4. What a correct runbook must cover

| Case | What the runbook must contain | Why it matters |
|---|---|---|
| Owner and scope | the one dependency, its old and new version, and the release notes' breaking changes | a single cause for any difference |
| Reproducible build | the lockfile diff, a clean build, and the image digest | the same input gives the same artifact |
| Gates | native and contract test suites, then the saved load profile against the last known good image, with thresholds | behaviour, not just compilation |
| Rollback | the last known good image tag and the revert commit | restoring service needs no rebuild |
| Verification | post-deploy SLO burn and error rates compared with before | done means measured |

## 5. The solution, step by step

### 5.1 From first principles

1. **What could a dependency change break?** Anything it touches at runtime: defaults, timeouts, serialisation, performance. Compilation success says almost nothing, so the evidence must be tests that exercise contracts plus a load comparison.
2. **Isolate the cause.** Upgrading several dependencies at once makes any regression ambiguous. One dependency per change, with its resolved graph inspected (`go mod graph`, `cargo tree`, `uv pip tree`) so transitive bumps are visible.
3. **Make it reproducible.** The lockfile pins the whole graph; committing it means the canary, the deploy, and the rollback all use exactly the tested set.
4. **Compare like with like.** The same load profile (rate, mix, seed, duration) against the last known good image and the candidate, comparing latency and error distributions, not single numbers.
5. **Rollback is a redeploy.** Keep the previous image tag; reverting the commit restores the source, the image tag restores production at once.

### 5.2 The model runbook

`docs/runbooks/DependencyUpgrade.md`:

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

### 5.3 Why this solution

- **One change per deploy** keeps attribution and rollback trivial.
- **Load comparison against the last good image** catches performance and timeout changes that functional tests miss.
- **Why not wait for production to tell you?** A regression found under real traffic has already spent error budget; the same load profile finds it on a branch.

### 5.4 Explain it back

1. Why upgrade one dependency at a time?
   <details><summary>Answer</summary>So any behavioural difference has exactly one cause and the rollback is reverting one diff.</details>
2. Why is a passing build not enough evidence?
   <details><summary>Answer</summary>A dependency can keep its API and change defaults, timeouts, or performance; only tests that exercise behaviour and a load comparison see that.</details>
3. Why keep the last known good image tag?
   <details><summary>Answer</summary>Rollback then means redeploying an artifact that already ran in production, which is faster and safer than rebuilding the old source.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | Several dependencies bumped together | a regression with many possible causes; rollback undoes unrelated fixes | one dependency and its lockfile per change |
| 2 | Only compile and unit tests run | runtime behaviour or latency changes reach production | contract suites and the saved load profile gate the deploy |
| 3 | No recorded last known good image | rollback requires a rebuild under pressure | the tag is a precondition |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.15` | A prerequisite: the contract or evidence this runbook relies on. |
| Back | `ops.05` | A prerequisite: the contract or evidence this runbook relies on. |

## Going further

| Resource | What to inspect |
|---|---|
| [Incident drill catalog](README.md) | Compare this inject with the adjacent failure drills and use the same containment and verification fields. |
