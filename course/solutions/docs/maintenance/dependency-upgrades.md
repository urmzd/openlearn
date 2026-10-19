# Dependency upgrade review: telemetry SDK

**Change:** `go.opentelemetry.io/otel` and the matching SDK/exporter family, from 1.43.x to 1.44.0. The change is reviewed as one compatibility family because mixing SDK and exporter minors can fail at runtime even when Go resolves the graph.

**Owner:** Gateway maintainers. **Reviewer:** release engineering. **Scope:** gateway telemetry only. The engine's Rust dependency graph is unchanged.

## Evidence before merge

| Evidence | Result recorded for this change |
|---|---|
| Resolved graph | `go list -m all`; save the diff of `go.mod` and `go.sum` with the change. No unreviewed transitive changes. |
| API compatibility | `go test ./...` in `go/` and contracts tests pass. Span names, resource attributes, and exporter shutdown behavior remain stable. |
| Security | `govulncheck ./...` is clean, or each finding has a linked exception and expiry date. |
| Performance | Fixed gateway request fixture, warm-up 200 requests, 5 runs of 2,000 requests. Compare median and p95 against the same host and config. |
| Rollout | Deploy one canary replica; check export failures and gateway p95 for 15 minutes before widening. |

The current repository requirement is declared in `course/contracts/allowed-deps.toml`; exact versions belong in the generated lock state. This review does not treat a semver range as proof of the resolved version. Attach command output or CI run links for the actual candidate before approval.

## Decision and recovery

Merge only when unit/contract tests pass, no unapproved vulnerability remains, and gateway p95 regresses by less than 5% across the five paired runs. The release owner records the CI run, image digest, and reviewer. If export errors rise or p95 exceeds the threshold, stop rollout and redeploy the prior immutable image digest with its prior `go.mod` and `go.sum`; do not attempt a live downgrade of only one module. Re-run the same fixture after rollback and record the result.

## Open evidence

This repository example is a review template, not a claim that an upgrade has been deployed. The learner replaces the candidate range, resolved graph, CI links, image digest, and measurements with evidence from the change under review. Unknown results remain `pending`; they are not inferred from a successful compile.
