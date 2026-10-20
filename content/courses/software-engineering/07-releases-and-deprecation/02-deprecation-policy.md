# Deprecation policy

A stable service accumulates interfaces that eventually need replacing. The first stable release promised compatibility; this section says how that promise ends for an interface without breaking the people who depend on it. With enough users, every observable behavior is depended on (Hyrum's Law), so removing anything needs a policy: announce, warn, measure usage, migrate, and only then remove. A deprecation policy gives consumers time and operators evidence before removal.

The section writes the system's policy, [`docs/DEPRECATION.md`](../../../../course/solutions/docs/DEPRECATION.md). It contains a general rule for what every notice must say, one entry for each active deprecation (API v1 to v2, and the KV envelope v1 to v2), and the review that must happen before anything is removed. The two migrations that follow the policy are carried out later in [KV format v1 to v2](../08-maintenance/01-kv-format-v1-to-v2.md) and [API v1 to v2](../08-maintenance/02-api-v1-to-v2.md). There are no hard prerequisites; the changelog of [Commits, semver, and releases](01-commits-semver-and-releases.md) is what links to this policy.

Before you read the solution, write down what you would measure before removing API v1. The answer turns out to be the heart of the policy.

> **Learning objectives**
> - List what every deprecation notice must name, and why a notice without an owner or a meter cannot start the removal clock.
> - Write an exit condition that combines a minimum window with measured zero use.
> - Explain why format deprecations need "readers first, then writers" and when rollback is still possible.
> - Design a removal review that requires recorded evidence before an interface is removed.

## Deprecate before you remove

The principles are short. Deprecate before removal. Name the replacement, the affected versions, the sunset date or condition, the compatibility window, the owner, and the support path. And measure real use before setting a removal date. Deprecation is a feature with a budget: it has an owner, a timeline, and telemetry that shows when the last caller is gone.

Here is what that looks like for KV v1, worked by hand. Announce the v2 envelope. Accept both formats during the migration. Count v1 reads and writes, and alert on any remaining consumers. Remove v1 only after the stated threshold and window have both been met.

## What the policy must contain

The artifact is `docs/DEPRECATION.md`: a general rule for what every notice contains, one entry per active deprecation, and the removal review. A correct policy handles each of the following:

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the policy exists at the declared path | artifact | it is the document the changelog's Compatibility section links |
| each deprecation names replacement, notice, window, measure, exit condition, owner, and rollback | review | consumers know what to do and by when; operators know when removal is safe |
| exit conditions are measurable | review | "when nobody uses it" must be a query, not a feeling |
| readers accept the new format before any writer emits it | review | a mixed fleet during rollout can always read what it is sent |
| removal requires a recorded review | review | nobody removes an interface on their own initiative |

## Building the policy step by step

Five observations shape the policy.

1. **A consumer cannot migrate to what it does not know about.** So every notice names the old interface, the replacement, and what the consumer must do, and it reaches them through channels they read: the changelog, response headers, the migration runbook.
2. **A date alone is a guess.** Removal is safe when use has stopped, which can only be known by measuring it. So each deprecation has a meter and an exit condition stated in terms of it, plus a minimum window so slow consumers have time.
3. **Someone must own the clock.** A notice without an owner, or without a way to measure use, cannot start the removal clock, because nobody would know when it ends.
4. **Formats differ from APIs.** An API has a client choosing a version per request; a stored or exchanged format has readers and writers that upgrade at different times. So formats need "readers first, then writers", and a rollback that is possible only while all readers still accept the old format.
5. **Plans change.** If the exit condition is not met, the honest move is to publish a revised window with a reason, never to extend silently or remove anyway.

### The general rule

```markdown
Every notice names the old interface and version, replacement, first release carrying the notice,
removal date or objective exit condition, compatibility window, owner, consumer action, and support
channel. A notice without an owner or a way to measure remaining use cannot start the removal clock.
```

The second sentence is the enforcement. It turns "we announced it" into "we can prove when it is safe".

### API v1 to v2

```markdown
- **Replacement:** API v2, selected explicitly with `X-API-Version: 2`.
- **Notice:** v1 responses include `Deprecation` and `Sunset` headers during the overlap.
- **Window:** at least 90 days from the first production release that emits both headers.
- **Measure:** the gateway usage meter groups request totals by selected API version. ...
- **Exit condition:** remove v1 only after the 90-day window has elapsed and the v1 request count is zero for 30 consecutive days.
- **Owner:** gateway service owner. ...
- **Rollback:** if v2 causes an elevated error rate, route the affected cohort back to v1 while the window remains open. Do not extend the sunset silently; publish a revised date and reason.
```

Each field earns its place. Opting in by header means no client changes version by accident. The headers put the notice in every v1 response, where a client's logs will show it even if nobody reads the changelog. The window starts at the first release that emits the headers, not at the announcement, because only then can clients see it. And the exit condition is a conjunction: time served *and* zero measured use for 30 days, so neither a forgotten deadline nor one quiet week triggers removal.

### KV envelope v1 to v2

The format deprecation adds what an API deprecation does not need:

```markdown
- **Compatibility:** readers accept v1 and v2 before any writer emits v2. Keep v1 readable through the full rollback window.
- **Measure:** record negotiated format by peer and count v1 reads and writes. Verify hashes, scale values, and reconstructed blocks on a canary pair.
- **Exit condition:** stop accepting v1 only after every peer advertises v2 support and v1 reads and writes remain zero for 30 consecutive days.
- **Rollback:** disable v2 writes and return to v1 while all readers still support it. Stop rollout on negotiation failures, hash mismatches, or request errors above the established SLO.
```

This is the fourth observation in practice: readers first; a canary pair that checks the numerical content (fp8 payloads with f32 scales must reconstruct the blocks); and a rollback that works because v1 stays readable. The stop conditions are named in advance, so a rollout is halted by a rule, not a debate.

### The removal review

The removal review closes the loop. The owner links the notice, the usage evidence, the conformance results, and the rollback record; the release approver confirms the window and the exit condition; and the changelog marks the old version removed. The full policy is [`DEPRECATION.md`](../../../../course/solutions/docs/DEPRECATION.md).

### Why this design

The central choice is a **measured exit rather than a fixed date**. A fixed date is simple and breaks the last consumers; a measured exit protects them, at the cost of needing a meter by version and peer. Both deprecations already have one: the gateway usage meter and format negotiation.

The policy uses **a minimum window plus the measure** because each guards against a different failure. The window protects consumers that are idle for a while, such as a monthly batch job; the 30 days of zero use protect against removal while use continues.

Finally, **per-deprecation entries in one policy** split the general from the specific. The general rule keeps every notice complete; the entries hold the specifics. A new deprecation is a new entry, reviewed like code.

## What can go wrong

Three pitfalls recur in deprecation policies. The first is an artifact that names a decision without the evidence that supports it, so reviewers cannot verify the claim against the repository or the system. The solution states each exit condition in terms of a meter, and the removal review links the usage evidence.

The second is omitting the failure or rollback case. The happy path passes while an operator has no safe response under failure. Each entry here has a rollback and, for KV, named stop conditions.

> **Warning:** If the owner, threshold, or scope is left implicit, two reviewers can reach different release or risk decisions from the same evidence. Name an owner per deprecation and state the thresholds: 90 and 30 days, and zero use.

## Where this leads

A deprecation is a decision, and the migrations record theirs as [Architecture decision records](../12-documentation/01-architecture-decision-records.md), described in *Software Engineering*. The changelog's Compatibility section in [Commits, semver, and releases](01-commits-semver-and-releases.md) links this policy. The next chapter carries out both entries: [KV format v1 to v2](../08-maintenance/01-kv-format-v1-to-v2.md) and [API v1 to v2](../08-maintenance/02-api-v1-to-v2.md).

### Further reading

- RFC 9745 (the `Deprecation` header) and RFC 8594 (the `Sunset` header): the HTTP headers the API notice uses.
- *Software Engineering at Google*, ch. 15, "Deprecation" (free): advisory versus compulsory deprecation, and why measured use matters.

## Key terms

- **Deprecation**: announcing that an interface will be removed, with a replacement and a timeline, before removing it.
- **Hyrum's Law**: with enough users, every observable behavior of an interface is depended on by somebody.
- **`Deprecation` and `Sunset` headers**: HTTP response headers that announce, in band, that an API is deprecated and when it will be removed.
- **Exit condition**: the measurable criterion, here a minimum window plus a period of zero use, that makes removal safe.
- **Readers first, then writers**: the rollout order for a format change, so every message sent can be read by every peer.
- **Removal review**: a recorded check of notice, usage evidence, conformance, and rollback before an interface is removed.

## Summary

- With enough users every behavior is depended on, so removal needs a policy: announce, warn, measure, migrate, remove.
- Every notice names the interface, replacement, window, exit condition, owner, consumer action, and support channel; without an owner and a meter, the clock does not start.
- Exit conditions combine a minimum window with measured zero use, so neither a deadline nor a quiet week alone triggers removal.
- Formats roll out readers first, then writers, and rollback works only while every reader still accepts the old format.
- Plans change honestly: publish a revised window with a reason rather than extending silently or removing anyway.

## Review questions

1. Why must readers accept KV v2 before any writer emits it?
   <details><summary>Answer</summary>During rollout peers upgrade at different times. If a writer emits v2 while some reader accepts only v1, that reader fails. Readers first means every message sent can be read, and rollback (stop writing v2) stays possible.</details>
2. Why does the API window start at the first release that emits both headers, not at the announcement?
   <details><summary>Answer</summary>The headers are how clients actually learn of the deprecation; before they ship, a client has no in-band signal. Counting from the announcement would shorten the effective window for anyone who did not read it.</details>
3. The v1 request count has been zero for 30 days, but only 60 days of the window have passed. Can v1 be removed?
   <details><summary>Answer</summary>No. The exit condition requires both the 90-day window and 30 days of zero use.</details>
4. Why is a fixed removal date alone a poor exit condition?
   <details><summary>Answer</summary>A date is a guess about when use will stop. If consumers remain, removal breaks them. Measuring use by version or peer turns "when nobody uses it" into a query, and the window still protects consumers that are idle for long stretches.</details>
