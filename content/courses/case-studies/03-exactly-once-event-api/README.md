# Exactly-Once Event API

A caller assembles a request incrementally (it creates a container, then uploads documents into it) and then submits it for downstream processing. A double click, a client retry, or a redelivered webhook must never cause the downstream system to be invoked twice. That one requirement, submitting twice never processes twice, is the whole of this case study, and it splits cleanly into two halves worth designing separately: the shape of the API, and the guarantee behind it.

The runnable implementation is [`event_api.py`](event_api.py): a queue, a service, and a route table, with a real multi-threaded race test. It uses the standard library only; run it with `python event_api.py`. The study ends with an honest account of what the guarantee is, because "exactly-once delivery" does not exist, and the useful skill is choosing between at-most-once and at-least-once deliberately.

> **Learning objectives**
> - Design an API that separates mutable build-up from a one-way submission, and justify returning 200 for a duplicate submit.
> - Explain why check-then-act dedup is a race and how a unique constraint closes it.
> - Implement claiming as a single state transition so two workers never take the same job.
> - Distinguish at-most-once, at-least-once, and effectively-once processing, and say which this design builds.
> - Test a concurrency guarantee with barrier-based tests rather than single-threaded ones.

## Before you begin

You need HTTP verbs and status codes, basic SQL, and the idea of a race condition. Before reading the next section, design the endpoints yourself. Notice whether you made submission its own route or a field update, and what that forced onto the client.

## The API shape

Building a request and submitting it have different semantics. Build-up is mutable, repeatable, and cheap; submission is a one-way door with side effects. Collapsing them into one endpoint is what forces "did I already send this?" onto the client. The routes are:

| Route | Returns | Note |
|-------|---------|------|
| `POST /requests` | 201, `{id, documents: []}` | Create the container |
| `POST /requests/{id}/documents` | 200, updated request | 404 if unknown |
| `POST /requests/{id}/submit` | 200, `{queued: true, status: "queued"}` | First call |
| `POST /requests/{id}/submit` | 200, `{queued: false, status: "duplicate"}` | Every later call |
| `POST /events/run` | 200, `{claimed: n}` | Drain the queue |
| `GET /events?status=pending` | 200, event list | Inspection |

Two decisions in that table need defending. The first is that submission is its own endpoint, not a `PATCH` that flips a field. It is a state transition with side effects, so it gets its own verb and a response describing what happened. A field update implies that the state is data; it is not, it is an event.

The second is that the duplicate returns 200, not 409. The client asked for this request to be submitted, and after both calls it is submitted, so the client's intent is satisfied either way. A 409 would make every caller write error handling for a non-error, and clients that retry on network timeouts hit this path constantly. Reserve 409 for when the second call *means* something different from the first. The response is a small object rather than a bare boolean for the same reason: the caller needs to distinguish "I queued it" from "it was already queued," and both are successes.

## Constraint beats check

With the shape settled, the guarantee comes next, and its first layer is deduplicating the enqueue. The naive dedup is a race:

```python
if not queue.get(source):     # check
    queue.insert(event)       # ...act
```

Two callers can both pass the check before either inserts. The window is small and the bug is real, and it is exactly the kind of thing that survives testing and fails in production under load. A check followed by an act is not a guarantee.

The fix is to make the check and the act the same statement by letting the database enforce it:

```sql
CREATE TABLE events (
    source TEXT PRIMARY KEY,   -- the dedup key IS the key
    ...
);
```

```python
try:
    db.execute("INSERT INTO events (source, ...) VALUES (?, ...)", ...)
except sqlite3.IntegrityError:
    return False              # already queued: a no-op, not an error
return True
```

No interleaving defeats this, because uniqueness is enforced at the point of write by the only component that sees all writers. The constraint is the dedup. The dedup key is the request id the client already has; nothing is hashed or invented.

## Claiming is a transition

Deduplicating the enqueue is not enough. Two workers draining the queue can both read the same pending row and both handle it. The fix has the same shape as before, one statement instead of two:

```sql
UPDATE events SET status = 'claimed', claimed_at = datetime('now')
WHERE source IN (SELECT source FROM events WHERE status = 'pending' LIMIT ?)
RETURNING source, request_id, document_ids;
```

That statement moves the rows out of `pending` and reports which rows it moved, and the second worker's `WHERE` clause no longer matches them. On Postgres the same idea is `SELECT ... FOR UPDATE SKIP LOCKED`. The principle is identical, and it is the principle that transfers: **claiming is a state transition, never a read followed by a write.**

## What the guarantee actually is

This design is **at-most-once**. A worker that dies mid-handle leaves the event claimed and unfinished, and nothing retries it. The three guarantees compare like this:

| Guarantee | Property | How to get it |
|-----------|----------|---------------|
| At-most-once | Never double-fires, may drop | Claim and never retry (what this builds) |
| At-least-once | Never drops, may double-fire | Add a lease and a sweeper for expired claims |
| Exactly-once | Not achievable end-to-end | Unless the downstream effect is itself idempotent |

"Exactly-once delivery" is marketing. What is achievable is **effectively-once processing**: at-least-once delivery plus an idempotent downstream effect. The dedup key in this design is precisely the idempotency key that makes that upgrade safe, which is why the at-most-once version is a legitimate stopping point rather than a dead end.

Moving to at-least-once is a small change: add `claimed_until`, and a sweeper that returns expired claims to `pending`. What it costs is a hard requirement that the downstream call be idempotent, because it *will* be retried. Most systems reach for exactly-once when at-least-once plus idempotent effects is both achievable and simpler; decide what your own systems actually need.

## Failure is recorded, not lost

An event whose handler raises is marked `failed`: not dropped, and not silently retried. The lifecycle is `pending -> claimed -> completed | failed`, and every state is queryable.

"It disappeared" is the one outcome an operator cannot debug. A failed event that is visible in `GET /events?status=failed` is an incident with a work list; a swallowed exception is an unexplained gap in the data.

## Layering, and the seam that removes mocks

The implementation has four layers:

```
route table   HTTP only: routes to service calls, domain errors to status codes
service       application logic: requests, submission, draining
queue         persistence: the two atomic dedup points
domain        Event, Status, and the DownstreamSystem protocol
```

The service raises `NotFound`; only the route table knows that this means 404. Push HTTP vocabulary any deeper and the service becomes untestable without a client.

The downstream system is a `Protocol`, not a base class or a patched import. Tests pass a recording fake that counts invocations; production passes the real client. That seam is what makes the central guarantee directly assertable without a mocking library: `len(sink.calls) == 1` is the whole test.

One thing was deliberately left out. The request store is in-memory and does not survive a restart, while the event queue is durable. That asymmetry is the design. The part that must never double-fire is the part that gets durability; build-up state is cheap to reconstruct and not worth the cost. Naming what you did not build, and why, is part of the design rather than an admission.

## The build log

The service was assembled in this order:

1. **Write the spec before the code.** Endpoints, payloads, and the dedup rule stated in prose. This build was explicitly two-phase, spec then implementation, and the spec is where the submit-is-its-own-endpoint and duplicate-is-200 decisions actually got made.
2. **Model the domain**: `Event`, `Status`, the `DownstreamSystem` protocol. No framework, no database, no HTTP.
3. **The queue, with the primary key as the dedup**, and `enqueue_once` returning a boolean rather than raising.
4. **The atomic claim**, written as a transition from the start. Retrofitting this onto a read-then-write is harder than writing it correctly once.
5. **The service layer**: create, upload, submit, drain. Domain errors only.
6. **Single-threaded tests** for the happy path, double submit, double drain, and the failure path.
7. **The concurrency tests last, and they are the real ones.** Eight threads submitting through a barrier; six workers draining fifty events through a barrier. Everything before this passes against a broken implementation.
8. **The route table**, mapping domain errors to status codes, tested without a server.

Step 7 is the step people skip. Single-threaded tests of a concurrency guarantee prove nothing at all: a check-then-act implementation passes every test in step 6.

> **Try it:** Replace `enqueue_once` with a check-then-act version and run the tests. `test_concurrent_submits_enqueue_once` fails. That failure is the entire argument for constraint-based dedup, and it is worth seeing rather than believing.

## Techniques to take away

| Technique | When to apply |
|-----------|---------------|
| Unique constraint as dedup | Any idempotent write endpoint |
| Atomic claim with `RETURNING` | Any queue with more than one consumer |
| Idempotency key from the client's own id | Any retryable submit |
| Duplicate as 200 with explicit status | Any endpoint clients will retry |
| Separate build-up from submission | Any multi-step resource creation |
| Record failure as a state | Any background processing |
| Protocol seam for external systems | Any test that would otherwise need a mock |
| Durable only where it must be | Any system with mixed-criticality state |
| Barrier-based concurrency tests | Any guarantee that only breaks under a race |

## Where this leads

The same patterns at broker scale (idempotency, dead-letter queues, consumer groups) and what at-least-once needs to be safe (durable execution, leases, checkpoints) are developed in [Durable orchestration and workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/README.md). Why the constraint is atomic, in terms of constraints, transactions, and isolation, is covered in [Storage and warehousing](../../data-engineering/02-storage-warehousing/README.md). Contract design under retries, with status-code semantics, belongs to [System design](../../systems/01-system-design/README.md). And the preference for fakes over mocks that the Protocol seam embodies is part of [The testing mentality](../../software-craftsmanship/03-testing-mentality/README.md).

Where the problem shows up: in payments and fintech, double-charge prevention is this exact problem, solved with idempotency keys; any webhook consumer must assume redelivery and build at-least-once plus idempotent effects; any queue-backed backend needs claim semantics and poison-message handling; and any API design round probes retry semantics, status codes, and resource shape.

## Key terms

- **Idempotency key**: an identifier that makes repeating a request have the same effect as making it once.
- **Check-then-act race**: a bug in which two callers both pass a check before either performs the action.
- **Atomic claim**: a single statement that moves work from pending to claimed and reports what it moved.
- **At-most-once**: work is never processed twice but may be dropped.
- **At-least-once**: work is never dropped but may be processed more than once.
- **Effectively-once processing**: at-least-once delivery combined with an idempotent effect.

## Summary

- Separate mutable build-up from a one-way submit endpoint, and return 200 with an explicit status for a duplicate.
- A primary key on the idempotency key makes the check and the act one statement; no interleaving defeats it.
- Claiming is one `UPDATE ... RETURNING` transition, never a read followed by a write.
- This design is at-most-once; adding a lease and sweeper gives at-least-once, which is safe only with an idempotent downstream effect.
- Failures are recorded as a state, and only barrier-based concurrency tests can prove the guarantee.

## Review questions

1. Why should a duplicate submit return 200 rather than 409?
   <details><summary>Answer</summary>The client's intent, that the request be submitted, is satisfied either way. A 409 forces every caller, including every client that retries on timeouts, to handle a non-error. Reserve 409 for a second call that means something different from the first.</details>
2. Why does a primary key on the idempotency key defeat every interleaving that check-then-act does not?
   <details><summary>Answer</summary>Uniqueness is enforced at the point of write by the database, the only component that sees all writers, so the check and the act are a single statement. A duplicate insert raises `IntegrityError`, which the queue treats as an already-queued no-op.</details>
3. What goes wrong if workers claim events with a read followed by a write?
   <details><summary>Answer</summary>Two workers can both read the same pending row and both handle it. A single `UPDATE ... WHERE status = 'pending' ... RETURNING` moves the rows and reports them atomically, so the second worker's `WHERE` no longer matches.</details>
4. What would it take to move this design from at-most-once to at-least-once, and what does that require downstream?
   <details><summary>Answer</summary>Add a `claimed_until` lease and a sweeper that returns expired claims to `pending`. The downstream call must then be idempotent, because it will be retried; the existing dedup key is the idempotency key that makes this safe.</details>
5. Why does a check-then-act implementation pass the single-threaded tests?
   <details><summary>Answer</summary>Without concurrency, the check always completes before the act, so the race never occurs. Only tests that release many threads together through a barrier expose it.</details>
