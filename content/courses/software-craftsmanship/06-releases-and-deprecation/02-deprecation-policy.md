# Deprecation policy

## Overview

| | |
|---|---|
| **Module** | `craft.12` · practice · docs · Pass 11 · 1 h reading |
| **Solution** | [`docs/DEPRECATION.md`](../../../../course/solutions/docs/DEPRECATION.md): the required notice, the API v1 to v2 and KV envelope v1 to v2 deprecations, and the removal review |
| **Needs** | none |
| **Used by** | no code call site; this is an independent artifact. `craft.13` (KV format v1 to v2) and `craft.14` (API v1 to v2) carry out the two deprecations it defines |

## Key Takeaways

- Tie each claim to the exact platform evidence that `Deprecation policy` produces.
- State the owner, success condition, and recovery action before declaring the work complete.
- Record unresolved assumptions so the next review can test them.

## How to read this chapter

Sections 1 to 3 explain why deprecation needs a policy and work the KV v1 case by hand. Section 4 states what the policy must contain. Section 5 walks through the solution's policy entry by entry and explains why each field is there. Before reading it, write down what you would measure before removing API v1.

---

## 1. Why now

A stable service accumulates interfaces that eventually need replacement. A deprecation policy gives consumers time and operators evidence before removal.

## 2. Principles

Deprecate before removal. Name the replacement, affected versions, sunset date or condition, compatibility window, owner, and support path. Measure real use before setting a removal date.

## 3. Worked example

For KV v1, announce the v2 envelope, accept both formats during migration, count v1 reads and writes, alert on remaining consumers, then remove v1 only after the stated threshold and window.

## 4. The artifact

`docs/DEPRECATION.md`: a general rule for what every notice contains, one entry per active deprecation (API v1 to v2, KV envelope v1 to v2), and the review that must happen before anything is removed.

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the policy exists at the declared path | artifact | it is the document the changelog's Compatibility section links |
| each deprecation names replacement, notice, window, measure, exit condition, owner, and rollback | review | consumers know what to do and by when; operators know when removal is safe |
| exit conditions are measurable | review | "when nobody uses it" must be a query, not a feeling |
| readers accept the new format before any writer emits it | review | a mixed fleet during rollout can always read what it is sent |
| removal requires a recorded review | review | nobody removes an interface on their own initiative |

## 5. The solution, step by step

### 5.1 From first principles

1. **A consumer cannot migrate to what it does not know about.** So every notice names the old interface, the replacement, and what the consumer must do, and it reaches them through channels they read: the changelog, response headers, the migration runbook.
2. **A date alone is a guess.** Removal is safe when use has stopped, which can only be known by measuring it. So each deprecation has a meter and an exit condition in terms of it, plus a minimum window so slow consumers have time.
3. **Someone must own the clock.** A notice without an owner, or without a way to measure use, cannot start the removal clock, because nobody would know when it ends.
4. **Formats differ from APIs.** An API has a client choosing a version per request; a stored or exchanged format has readers and writers that upgrade at different times. So formats need "readers first, then writers" and a rollback that is possible only while all readers still accept the old format.
5. **Plans change.** If the exit condition is not met, the honest move is to publish a revised window with a reason, never to extend silently or remove anyway.

### 5.2 The policy, annotated

The general rule:

```markdown
Every notice names the old interface and version, replacement, first release carrying the notice,
removal date or objective exit condition, compatibility window, owner, consumer action, and support
channel. A notice without an owner or a way to measure remaining use cannot start the removal clock.
```

The second sentence is the enforcement: it turns "we announced it" into "we can prove when it is safe".

API v1 to v2:

```markdown
- **Replacement:** API v2, selected explicitly with `X-API-Version: 2`.
- **Notice:** v1 responses include `Deprecation` and `Sunset` headers during the overlap.
- **Window:** at least 90 days from the first production release that emits both headers.
- **Measure:** the gateway usage meter groups request totals by selected API version. ...
- **Exit condition:** remove v1 only after the 90-day window has elapsed and the v1 request count is zero for 30 consecutive days.
- **Owner:** gateway service owner. ...
- **Rollback:** if v2 causes an elevated error rate, route the affected cohort back to v1 while the window remains open. Do not extend the sunset silently; publish a revised date and reason.
```

Opting in by header means no client changes version by accident. The headers put the notice in every v1 response, where a client's logs will show it even if nobody reads the changelog. The window starts at the first release that emits the headers, not at the announcement, because only then can clients see it. The exit condition is a conjunction: time served *and* zero measured use for 30 days, so neither a forgotten deadline nor one quiet week triggers removal.

KV envelope v1 to v2:

```markdown
- **Compatibility:** readers accept v1 and v2 before any writer emits v2. Keep v1 readable through the full rollback window.
- **Measure:** record negotiated format by peer and count v1 reads and writes. Verify hashes, scale values, and reconstructed blocks on a canary pair.
- **Exit condition:** stop accepting v1 only after every peer advertises v2 support and v1 reads and writes remain zero for 30 consecutive days.
- **Rollback:** disable v2 writes and return to v1 while all readers still support it. Stop rollout on negotiation failures, hash mismatches, or request errors above the established SLO.
```

This is step 4 in practice: readers first, a canary pair that checks the numerical content (fp8 payloads with f32 scales must reconstruct the blocks), and a rollback that works because v1 stays readable. The stop conditions are named in advance, so a rollout is halted by a rule, not a debate.

The removal review closes the loop: the owner links notice, usage evidence, conformance results, and rollback record; the release approver confirms the window and exit condition; the changelog marks the old version removed. Full policy: [`DEPRECATION.md`](../../../../course/solutions/docs/DEPRECATION.md).

### 5.3 Why this solution

- **Measured exit versus a fixed date.** A fixed date is simple and breaks the last consumers; a measured exit protects them at the cost of needing a meter by version and peer. Both deprecations already have one (the gateway usage meter, format negotiation).
- **A minimum window plus the measure.** The window protects consumers that are idle for a while (a monthly batch job); the 30-day zero protects against removal while use continues.
- **Per-deprecation entries in one policy.** The general rule keeps every notice complete; the entries hold the specifics. A new deprecation is a new entry, reviewed like code.

### 5.4 Explain it back

1. Why must readers accept KV v2 before any writer emits it?
   <details><summary>Answer</summary>During rollout peers upgrade at different times. If a writer emits v2 while some reader accepts only v1, that reader fails. Readers first means every message sent can be read, and rollback (stop writing v2) stays possible.</details>
2. Why does the API window start at the first release that emits both headers, not at the announcement?
   <details><summary>Answer</summary>The headers are how clients actually learn of the deprecation; before they ship, a client has no in-band signal. Counting from the announcement would shorten the effective window for anyone who did not read it.</details>
3. The v1 request count has been zero for 30 days, but only 60 days of the window have passed. Can v1 be removed?
   <details><summary>Answer</summary>No. The exit condition requires both the 90-day window and 30 days of zero use.</details>

## 6. Pitfalls

| # | Pitfall | Symptom | How the solution avoids it |
|---|---|---|---|
| 1 | The artifact names a decision without the evidence that supports it | Reviewers cannot verify the claim against the repository or system. | each exit condition is stated in terms of a meter; the removal review links the usage evidence |
| 2 | The failure or rollback case is omitted | The happy path passes while an operator has no safe response under failure. | each entry has a rollback and, for KV, named stop conditions |
| 3 | The owner, threshold, or scope is implicit | Two reviewers can reach different release or risk decisions from the same evidence. | an owner per deprecation; thresholds of 90 and 30 days and zero use |

## 7. Where it's used next

| Direction | Module | Connection |
|---|---|---|
| Back | `craft.02` | A deprecation is a decision; the migrations record theirs as ADRs. |
| Back | `craft.11` | The changelog's Compatibility section links this policy. |
| Forward | `craft.13`, `craft.14` | The KV format and API v1 to v2 migrations follow these entries. |

## Going further

| Resource | What to inspect |
|---|---|
| RFC 9745 (the `Deprecation` header) and RFC 8594 (the `Sunset` header) | The HTTP headers the API notice uses. |
| *Software Engineering at Google*, ch. 15 "Deprecation" (free) | Advisory versus compulsory deprecation, and why measured use matters. |
