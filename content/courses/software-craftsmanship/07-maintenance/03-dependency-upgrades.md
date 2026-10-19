# Dependency upgrades

A dependency upgrade is a change you did not write. It can alter APIs, performance, and security properties all at once, and because you cannot review its diff line by line, the usual tools of review do not apply. What you can do is make the upgrade deliberate: isolate it so any regression has one cause, measure its effects against evidence you agreed on in advance, and keep a reversible path back. That turns "bump and hope" into an ordinary change that rolls out like any other.

This section works through a model upgrade review, [`dependency-upgrades.md`](../../../../course/solutions/docs/maintenance/dependency-upgrades.md), for the gateway's OpenTelemetry family. The version requirements each component may use are fixed by the contract [`course/contracts/allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml). The section has no prerequisites, and the same procedure is exercised under pressure in [A breaking dependency upgrade](../../systems/05-incident-response-and-chaos/07-dependency-upgrade.md) in *Systems and Architecture*. Before reading the model answer, write down what you would check.

> **Learning objectives**
> - Scope an upgrade to one compatibility family and say why.
> - Tie each claim in an upgrade review to evidence from the candidate: the resolved graph, test and scan output, and paired benchmark runs.
> - Write a merge rule, an owner, and a rollback action before any measurement is taken.
> - Explain why a semver range or a successful compile is not evidence of compatibility.

## Upgrading deliberately

The principles come first. Read the release notes and compatibility ranges before changing a major version. Upgrade one dependency family at a time, pin the resolved graph, and run the behavior, race, security, and performance checks relevant to how the dependency is used.

For a breaking Go or Rust library update, that procedure looks like this. First make a minimal branch that builds against the new API. Then run the contract suites, and compare benchmark distributions against the recorded baseline. Throughout, keep the prior lockfile available.

## What a complete review contains

The artifact is `docs/maintenance/dependency-upgrades.md`, with one review per upgrade family. A complete review states each of the following:

| Element | What it states | Why it matters downstream |
|---|---|---|
| Change and scope | The exact family, from and to versions, and which components are affected. | A reviewer knows what else could break and what is untouched. |
| Resolved graph | The lockfile diff (`go.mod`/`go.sum`, `Cargo.lock`), with transitive changes reviewed. | A semver range is a requirement, not proof of what was resolved. |
| Compatibility evidence | Test and contract-suite results on the candidate. | Behavior changes show up before users see them. |
| Security evidence | `govulncheck` (or `cargo audit`) output, with exceptions linked and dated. | An upgrade can introduce a vulnerability as easily as fix one. |
| Performance evidence | A fixed workload, paired runs, median and p95 against the same host. | Regressions hide in tails and in noise. |
| Decision rule, owner, rollback | The threshold set in advance, who decides, and the exact recovery step. | Two reviewers reach the same decision from the same numbers. |

## Building the review step by step

Six observations explain the shape of that table.

1. An upgrade is a change to code you did not write, so you cannot review its diff line by line. You can only review its **effects**: what it resolves to, whether the system's own tests and contracts still hold, whether it is known to be vulnerable, and whether it is slower.
2. Effects are only attributable if one thing changes at a time. So the unit of review is one **compatibility family**: packages that must move together (an SDK and its exporters) move together, and nothing else moves with them.
3. Version requirements (`"v1"`, `">=1.30"`) admit many resolutions. The thing that ships is the lockfile, so the review records the lockfile diff and reads its transitive changes.
4. A threshold chosen after seeing the numbers can be bent to fit them. So the merge rule (tests pass, no unapproved vulnerability, p95 regression below a bound) is written before the measurements.
5. Something will eventually go wrong after merge. The fastest and least surprising recovery is to redeploy the previous immutable artifact, not to edit dependencies in production. So the review names that artifact and who acts.
6. A template that looks filled in is dangerous. Anything not yet measured stays visibly `pending`.

### Change, owner, and scope

The model review opens with the change and its boundaries:

> **Change:** `go.opentelemetry.io/otel` and the matching SDK/exporter family, from 1.43.x to 1.44.0. The change is reviewed as one compatibility family because mixing SDK and exporter minors can fail at runtime even when Go resolves the graph.
>
> **Owner:** Gateway maintainers. **Reviewer:** release engineering. **Scope:** gateway telemetry only. The engine's Rust dependency graph is unchanged.

This is the second observation in practice. The OpenTelemetry Go modules are versioned separately but tested together, so upgrading only `otel` while leaving `sdk` behind can compile and still fail at run time. Stating that the Rust graph is unchanged bounds what reviewers must look at. In [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml) the family appears under `[go.tinyllm]` as `go.opentelemetry.io/otel`, `/sdk`, `/trace`, `/sdk/metric`, and the exporters, all at major `v1` except the Prometheus exporter at `v0`. A minor bump stays inside the policy, which is why this review is about evidence rather than a policy exception.

### Evidence before merge

The heart of the review is the evidence table:

| Evidence | Result recorded for this change |
|---|---|
| Resolved graph | `go list -m all`; save the diff of `go.mod` and `go.sum` with the change. No unreviewed transitive changes. |
| API compatibility | `go test ./...` in `go/` and contracts tests pass. Span names, resource attributes, and exporter shutdown behavior remain stable. |
| Security | `govulncheck ./...` is clean, or each finding has a linked exception and expiry date. |
| Performance | Fixed gateway request fixture, warm-up 200 requests, 5 runs of 2,000 requests. Compare median and p95 against the same host and config. |
| Rollout | Deploy one canary replica; check export failures and gateway p95 for 15 minutes before widening. |

Each row answers one way the upgrade could hurt. The *resolved graph* row answers "what actually changed" (the third observation): `go list -m all` prints the full module graph after resolution, which shows transitive bumps that `go.mod` alone hides. The *API compatibility* row names the observable things a telemetry upgrade can silently change: span names and resource attributes, which dashboards and alerts key on, and exporter shutdown, where a changed flush can drop the last spans of a terminating pod.

The *security* row uses `govulncheck`, which reports only vulnerabilities in code paths the program can reach, so a clean result is meaningful; and an exception has an expiry so it cannot become permanent by neglect. The *performance* row fixes the workload and the host, warms up (the first requests pay for connection setup and caches), and repeats: five runs give a distribution, so one noisy run cannot decide. Finally, the *rollout* row limits the blast radius. One canary replica receives a fraction of the traffic, and the two signals watched are exactly the ones this change could move.

### The policy sentence

> This review does not treat a semver range as proof of the resolved version. Attach command output or CI run links for the actual candidate before approval.

This is the third observation stated as a rule. The contract says `"v1"`; the lockfile says which v1.

### Decision and recovery

> Merge only when unit/contract tests pass, no unapproved vulnerability remains, and gateway p95 regresses by less than 5% across the five paired runs. [...] If export errors rise or p95 exceeds the threshold, stop rollout and redeploy the prior immutable image digest with its prior `go.mod` and `go.sum`; do not attempt a live downgrade of only one module. Re-run the same fixture after rollback and record the result.

There are three conditions, all checkable, set in advance (the fourth observation). Rollback is by image digest (the fifth): the digest identifies exactly the bytes that were running before, including the old dependency graph. Downgrading one module in place would create a third, never-tested combination. Re-running the fixture after rollback confirms that the regression really came from the upgrade.

### Open evidence

> Unknown results remain `pending`; they are not inferred from a successful compile.

This is the sixth observation. The review is a record of facts about one candidate, and a field without a fact says so. The full model answer is [`dependency-upgrades.md`](../../../../course/solutions/docs/maintenance/dependency-upgrades.md).

### Why this design

The review covers **one family at a time** rather than a monthly "bump everything". When a batch regresses, nobody can say which package did it, and the rollback reverts fixes along with the regression. It uses **paired runs on one host** rather than a comparison against last month's dashboard, because host, kernel, and neighbouring load all change, and the comparison would measure them instead of the upgrade. And it judges **p95 with a 5% bound** rather than the mean, because telemetry overhead tends to show up as occasional slow exports, which move the tail long before the mean.

The tradeoff is accepted knowingly. The review costs an hour of benchmark time per family, and small, frequent upgrades are slower to land than a batch. In exchange, each one is attributable and reversible.

## What can go wrong

Four pitfalls recur. A review that names a decision without the evidence that supports it cannot be verified against the repository or the system; here every row of the evidence table names a command or a run to attach, and approval requires the attachments. A review that omits the failure or rollback case passes on the happy path while leaving an operator with no safe response under failure; "Decision and recovery" names the trigger (export errors, p95), the action (redeploy the prior digest), and the follow-up rerun. If the owner, threshold, or scope is implicit, two reviewers can reach different release or risk decisions from the same evidence; the model states owner, reviewer, and scope at the top, and the merge rule is three explicit conditions.

> **Warning:** Trusting the version requirement lets a transitive bump ship unreviewed. The resolved-graph row and the "a semver range is not proof" rule exist to prevent exactly that.

## Where this leads

A major upgrade that changes an interface deserves an [architecture decision record](../../diagramming/02-documentation-writing/01-architecture-decision-records.md), described in *Diagramming and Documentation*, as well as a review. The procedure is put under pressure in [A breaking dependency upgrade](../../systems/05-incident-response-and-chaos/07-dependency-upgrade.md), where a lockfile leaves the policy and this procedure finds and reverts it.

### Further reading

- [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml): the requirement each component declares; compare it with a lockfile to see what was resolved.
- Go vulnerability management (the `govulncheck` documentation): how reachability analysis decides which findings apply.
- *Software Engineering at Google*, ch. 21, [Dependency Management](https://abseil.io/resources/swe-book/html/ch21.html) (free).

## Key terms

- **Compatibility family**: a set of packages that must be upgraded together, such as an SDK and its exporters.
- **Resolved graph**: the exact versions a build uses after resolution, recorded in the lockfile, as opposed to the version requirements.
- **`govulncheck`**: Go's vulnerability scanner, which reports only vulnerabilities reachable from the program's code.
- **Paired runs**: repeated benchmark runs of baseline and candidate on the same host and configuration, compared as distributions.
- **Rollback by digest**: recovering by redeploying the previous immutable image, identified by its content digest.

## Summary

- A dependency upgrade is a change you did not write, so review its effects: resolved graph, tests and contracts, vulnerabilities, and performance.
- Upgrade one compatibility family at a time so any regression has a single cause.
- A semver range is a requirement, not proof; the lockfile is what ships.
- Fix the merge threshold, owner, and rollback before measuring, and roll back by the prior image digest.
- Unmeasured results stay `pending`; a successful compile is not evidence of compatibility.

## Review questions

1. Why is the SDK upgraded together with `otel`, but the Rust graph is left alone?
   <details><summary>Answer</summary>The OpenTelemetry Go modules form one compatibility family: mismatched minors can fail at run time. The Rust engine shares no code with them, so changing it in the same review would add a second cause for any regression.</details>
2. `go build` succeeds on the candidate. Which rows of the evidence table does that fill in?
   <details><summary>Answer</summary>None. A build proves the API still type-checks. It says nothing about behavior (span names, shutdown), known vulnerabilities, or latency, which is why those rows stay `pending` until their own evidence is attached.</details>
3. Why roll back by image digest instead of reverting `go.mod` and rebuilding?
   <details><summary>Answer</summary>A rebuild resolves and compiles again and may not reproduce the old bytes (a different toolchain or a different transitive resolution). The prior digest is exactly what ran successfully before. It is also faster in an incident.</details>
4. Why does the threshold come before the measurement?
   <details><summary>Answer</summary>With the numbers in hand, it is tempting to pick a bound just above what was measured. A threshold fixed in advance makes the decision mechanical and lets a second reviewer reach the same conclusion.</details>
