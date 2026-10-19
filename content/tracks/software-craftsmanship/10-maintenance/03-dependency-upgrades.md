# Dependency upgrades

## Overview

| | |
|---|---|
| **Module** | `craft.15` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | Model review: [`dependency-upgrades.md`](../../../../course/solutions/docs/maintenance/dependency-upgrades.md), an upgrade review of the gateway's OpenTelemetry family |
| **Contract** | [`course/contracts/allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml), the version requirements each component may use |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact. The [`ops.06`](../../systems/05-incident-response-and-chaos/06-dependency-upgrade.md) drill exercises the same procedure under pressure. |

## Key Takeaways

- Tie each claim in an upgrade review to evidence from the candidate itself: the resolved graph, test and scan output, and paired benchmark runs.
- State the owner, the merge threshold, and the rollback action before the upgrade ships.
- Record unresolved results as `pending`; a successful compile is not evidence of compatibility.

## How to read this chapter

Sections 1 to 3 explain why an upgrade needs a written review and outline the procedure. Section 4 lists what a complete review must contain. Section 5 walks through the model review section by section and explains why each piece is there. Try to write down what you would check before reading the model answer.

---

## 1. Why now

Dependency changes can alter APIs, performance, and security properties at once. A deliberate upgrade isolates the cause and leaves a reversible path.

## 2. Principles

Read release notes and compatibility ranges before changing a major version. Upgrade one dependency family at a time, pin the resolved graph, and run behavior, race, security, and performance checks relevant to its use.

## 3. Worked example

For a breaking Go or Rust library update, first make a minimal branch that builds against the new API, then run contract suites and compare benchmark distributions to the recorded baseline. Keep the prior lockfile available.

## 4. The artifact

The artifact is `docs/maintenance/dependency-upgrades.md`, one review per upgrade family.

### What a correct review must contain

| Element | What it states | Why it matters downstream |
|---|---|---|
| Change and scope | The exact family, from and to versions, and which components are affected. | A reviewer knows what else could break and what is untouched. |
| Resolved graph | The lockfile diff (`go.mod`/`go.sum`, `Cargo.lock`), with transitive changes reviewed. | A semver range is a requirement, not proof of what was resolved. |
| Compatibility evidence | Test and contract-suite results on the candidate. | Behavior changes show up before users see them. |
| Security evidence | `govulncheck` (or `cargo audit`) output, with exceptions linked and dated. | An upgrade can introduce a vulnerability as easily as fix one. |
| Performance evidence | A fixed workload, paired runs, median and p95 against the same host. | Regressions hide in tails and in noise. |
| Decision rule, owner, rollback | The threshold set in advance, who decides, and the exact recovery step. | Two reviewers reach the same decision from the same numbers. |

## 5. The solution, step by step

### 5.1 From first principles

1. An upgrade is a change to code you did not write, so you cannot review its diff line by line. You can only review its **effects**: what it resolves to, whether the system's own tests and contracts still hold, whether it is known-vulnerable, and whether it is slower.
2. Effects are only attributable if one thing changes at a time. So the unit of review is one **compatibility family**: packages that must move together (an SDK and its exporters) move together, and nothing else moves with them.
3. Version requirements (`"v1"`, `">=1.30"`) admit many resolutions. The thing that ships is the lockfile, so the review records the lockfile diff and reads its transitive changes.
4. A threshold chosen after seeing the numbers can be bent to fit them. So the merge rule (tests pass, no unapproved vulnerability, p95 regression below a bound) is written before the measurements.
5. Something will eventually go wrong after merge. The recovery that is fastest and least surprising is to redeploy the previous immutable artifact, not to edit dependencies in production. The review names that artifact and who acts.
6. A template that looks filled in is dangerous. Anything not yet measured stays visibly `pending`.

### 5.2 The model review, annotated

**Change, owner, scope.**

> **Change:** `go.opentelemetry.io/otel` and the matching SDK/exporter family, from 1.43.x to 1.44.0. The change is reviewed as one compatibility family because mixing SDK and exporter minors can fail at runtime even when Go resolves the graph.
>
> **Owner:** Gateway maintainers. **Reviewer:** release engineering. **Scope:** gateway telemetry only. The engine's Rust dependency graph is unchanged.

Step 2 in practice: the OpenTelemetry Go modules are versioned separately but tested together, so upgrading only `otel` while leaving `sdk` behind can compile and still fail at run time. Stating that the Rust graph is unchanged bounds what reviewers must look at. In [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml) the family appears under `[go.tinyllm]` as `go.opentelemetry.io/otel`, `/sdk`, `/trace`, `/sdk/metric`, and the exporters, all at major `v1` except the Prometheus exporter at `v0`; a minor bump stays inside the policy, which is why this review is about evidence rather than a policy exception.

**Evidence before merge.**

| Evidence | Result recorded for this change |
|---|---|
| Resolved graph | `go list -m all`; save the diff of `go.mod` and `go.sum` with the change. No unreviewed transitive changes. |
| API compatibility | `go test ./...` in `go/` and contracts tests pass. Span names, resource attributes, and exporter shutdown behavior remain stable. |
| Security | `govulncheck ./...` is clean, or each finding has a linked exception and expiry date. |
| Performance | Fixed gateway request fixture, warm-up 200 requests, 5 runs of 2,000 requests. Compare median and p95 against the same host and config. |
| Rollout | Deploy one canary replica; check export failures and gateway p95 for 15 minutes before widening. |

Each row answers one way the upgrade could hurt:

- *Resolved graph* answers "what actually changed" (step 3). `go list -m all` prints the full module graph after resolution, which shows transitive bumps that `go.mod` alone hides.
- *API compatibility* names the observable things a telemetry upgrade can silently change: span names and resource attributes (dashboards and alerts key on them) and exporter shutdown (a changed flush can drop the last spans of a terminating pod).
- *Security* uses `govulncheck`, which reports only vulnerabilities in code paths the program can reach, so a clean result is meaningful. An exception has an expiry so it cannot become permanent by neglect.
- *Performance* fixes the workload and the host, warms up (the first requests pay for connection setup and caches), and repeats: five runs give a distribution, so one noisy run cannot decide.
- *Rollout* limits blast radius: one canary replica receives a fraction of traffic, and the two signals watched are exactly the ones this change could move.

**The policy sentence.**

> This review does not treat a semver range as proof of the resolved version. Attach command output or CI run links for the actual candidate before approval.

This is step 3 stated as a rule. The contract says `"v1"`; the lockfile says which v1.

**Decision and recovery.**

> Merge only when unit/contract tests pass, no unapproved vulnerability remains, and gateway p95 regresses by less than 5% across the five paired runs. [...] If export errors rise or p95 exceeds the threshold, stop rollout and redeploy the prior immutable image digest with its prior `go.mod` and `go.sum`; do not attempt a live downgrade of only one module. Re-run the same fixture after rollback and record the result.

Three conditions, all checkable, set in advance (step 4). Rollback is by image digest (step 5): the digest identifies exactly the bytes that were running before, including the old dependency graph. Downgrading one module in place would create a third, never-tested combination. Re-running the fixture after rollback confirms the regression really came from the upgrade.

**Open evidence.**

> Unknown results remain `pending`; they are not inferred from a successful compile.

Step 6. The review is a record of facts about one candidate; a field without a fact says so.

Full model answer: [`dependency-upgrades.md`](../../../../course/solutions/docs/maintenance/dependency-upgrades.md).

### 5.3 Why this review

- **One family per review** rather than a monthly "bump everything": when the batch regresses, nobody can say which package did it, and the rollback reverts fixes along with the regression.
- **Paired runs on one host** rather than comparing against last month's dashboard: host, kernel, and neighbouring load change, and the comparison would measure them.
- **p95 with a 5% bound** rather than the mean: telemetry overhead tends to show up as occasional slow exports, which move the tail long before the mean.
- **Tradeoff accepted:** the review costs an hour of benchmark time per family, and small, frequent upgrades are slower to land than a batch. In exchange each one is attributable and reversible.

### 5.4 Explain it back

**Why is the SDK upgraded together with `otel`, but the Rust graph is left alone?**

<details><summary>Answer</summary>

The OpenTelemetry Go modules form one compatibility family: mismatched minors can fail at run time. The Rust engine shares no code with them, so changing it in the same review would add a second cause for any regression.

</details>

**`go build` succeeds on the candidate. Which rows of the evidence table does that fill in?**

<details><summary>Answer</summary>

None. A build proves the API still type-checks. It says nothing about behavior (span names, shutdown), known vulnerabilities, or latency, which is why those rows stay `pending` until their own evidence is attached.

</details>

**Why roll back by image digest instead of reverting `go.mod` and rebuilding?**

<details><summary>Answer</summary>

A rebuild resolves and compiles again and may not reproduce the old bytes (a different toolchain or a different transitive resolution). The prior digest is exactly what ran successfully before. It is also faster in an incident.

</details>

**Why does the threshold come before the measurement?**

<details><summary>Answer</summary>

With the numbers in hand, it is tempting to pick a bound just above what was measured. A threshold fixed in advance makes the decision mechanical and lets a second reviewer reach the same conclusion.

</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The review names a decision without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | Every row of the evidence table names a command or a run to attach, and approval requires the attachments. |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | "Decision and recovery" names the trigger (export errors, p95), the action (redeploy prior digest), and the follow-up rerun. |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | Owner, reviewer, and scope are stated at the top; the merge rule is three explicit conditions. |
| 4 | Trusting the version requirement | A transitive bump ships unreviewed. | The resolved graph row and the "semver range is not proof" rule. |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | [`craft.02`](../06-documentation-writing/01-architecture-decision-records.md) | A major upgrade that changes an interface is worth an ADR as well as a review. |
| Forward | [`ops.06`](../../systems/05-incident-response-and-chaos/06-dependency-upgrade.md) | The dependency upgrade drill, where a lockfile leaves the policy and this procedure finds and reverts it. |

## Going further

| Resource | What to inspect |
|---|---|
| [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml) | The requirement each component declares; compare with a lockfile to see what was resolved. |
| Go vulnerability management (`govulncheck` documentation) | How reachability analysis decides which findings apply. |
