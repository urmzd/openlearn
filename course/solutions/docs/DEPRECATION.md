# Deprecation and removal policy

## Purpose

This policy gives API clients and KV-format consumers a measurable migration window. The service owner records each deprecation in the changelog, API responses where applicable, and the runbook for the migration.

## Required notice

Every notice names the old interface and version, replacement, first release carrying the notice, removal date or objective exit condition, compatibility window, owner, consumer action, and support channel. A notice without an owner or a way to measure remaining use cannot start the removal clock.

## API v1 to v2

- **Replacement:** API v2, selected explicitly with `X-API-Version: 2`.
- **Notice:** v1 responses include `Deprecation` and `Sunset` headers during the overlap.
- **Window:** at least 90 days from the first production release that emits both headers.
- **Measure:** the gateway usage meter groups request totals by selected API version. Review v1 volume weekly and notify identified consumers directly.
- **Exit condition:** remove v1 only after the 90-day window has elapsed and the v1 request count is zero for 30 consecutive days.
- **Owner:** gateway service owner. The owner publishes the notice, tracks consumer migration, and records the removal decision.
- **Rollback:** if v2 causes an elevated error rate, route the affected cohort back to v1 while the window remains open. Do not extend the sunset silently; publish a revised date and reason.

## KV envelope v1 to v2

- **Replacement:** v2 envelopes with fp8 payloads and per-slab f32 scale metadata.
- **Compatibility:** readers accept v1 and v2 before any writer emits v2. Keep v1 readable through the full rollback window.
- **Measure:** record negotiated format by peer and count v1 reads and writes. Verify hashes, scale values, and reconstructed blocks on a canary pair.
- **Exit condition:** stop accepting v1 only after every peer advertises v2 support and v1 reads and writes remain zero for 30 consecutive days.
- **Owner:** engine storage owner. The owner coordinates peer readiness, the canary, and the removal review.
- **Rollback:** disable v2 writes and return to v1 while all readers still support it. Stop rollout on negotiation failures, hash mismatches, or request errors above the established SLO.

## Removal review

Before removal, the owner links the notice, usage evidence, conformance results, and rollback record in the release review. The release approver confirms the window and exit condition, then the changelog marks the old version as removed. If an exit condition is not met, keep compatibility or publish a revised window with a reason.
