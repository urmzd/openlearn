# Exactly-Once Event API

## Overview

- **Problem**: design an API where a caller builds up a request and submits it
  for processing, such that submitting twice never processes twice.
- **Runnable**: [`event_api.py`](event_api.py) -- queue, service, and route
  table with a real multi-threaded race test. Standard library only,
  `python event_api.py`.
- **Prerequisites**: HTTP verbs and status codes, basic SQL, and the idea of a
  race condition.
- **Estimated time**: 1-2 days

## Key Takeaways

- **A check followed by an act is not a guarantee.** Two callers can both pass
  the check before either acts. Make the check and the act the same statement.
- **The constraint is the dedup.** A primary key on the idempotency key turns a
  duplicate submit into a caught `IntegrityError` -- no interleaving defeats it.
- **Claiming work is a state transition, never a read-then-write.** One
  `UPDATE ... WHERE status = 'pending' RETURNING` is what stops two workers
  grabbing the same job.
- **A duplicate submit is a success, not an error.** The client's intent is
  satisfied either way. Returning 409 pushes error handling onto every caller
  for a case that is not a failure.
- **Exactly-once delivery does not exist.** Pick at-most-once or at-least-once
  deliberately, and say which you built and why.

## How to Study

- Run the tests, then replace `enqueue_once` with a check-then-act version and
  watch `test_concurrent_submits_enqueue_once` fail. That failure is the entire
  argument for constraint-based dedup, and it is worth seeing rather than
  believing.
- Design the endpoints yourself before reading section 1. Notice whether you
  made submission its own route or a field update, and what that forced onto
  the client.
- Read the guarantee section last, and decide what *your* systems actually need.
  Most reach for exactly-once when at-least-once plus idempotent effects is both
  achievable and simpler.

---

# Concepts & Techniques

## The Problem

A caller assembles a request incrementally -- create a container, upload
documents into it -- and then submits it for downstream processing. A double
click, a client retry, or a redelivered webhook must never cause the downstream
system to be invoked twice.

Two halves, worth designing separately: **the API shape**, and **the guarantee**.

## 1. The API shape

Building a request and submitting it have different semantics. Build-up is
mutable, repeatable, and cheap. Submission is a one-way door with side effects.
Collapsing them into one endpoint is what forces "did I already send this?" onto
the client.

| Route | Returns | Note |
|-------|---------|------|
| `POST /requests` | 201, `{id, documents: []}` | Create the container |
| `POST /requests/{id}/documents` | 200, updated request | 404 if unknown |
| `POST /requests/{id}/submit` | 200, `{queued: true, status: "queued"}` | First call |
| `POST /requests/{id}/submit` | 200, `{queued: false, status: "duplicate"}` | Every later call |
| `POST /events/run` | 200, `{claimed: n}` | Drain the queue |
| `GET /events?status=pending` | 200, event list | Inspection |

Two decisions worth defending:

**Submission is its own endpoint, not a `PATCH` that flips a field.** It is a
state transition with side effects, so it gets its own verb and a response
describing what happened. A field update implies the state is data; it is not,
it is an event.

**The duplicate returns 200, not 409.** The client asked for this request to be
submitted, and after both calls it is submitted -- intent satisfied. A 409 would
make every caller write error handling for a non-error, and clients that retry
on network timeouts hit this path constantly. Reserve 409 for when the second
call *means* something different from the first.

The response is a small object rather than a bare boolean for the same reason:
the caller needs to distinguish "I queued it" from "it was already queued", and
both are successes.

## 2. The guarantee, layer one: constraint beats check

The naive dedup is a race:

```python
if not queue.get(source):     # check
    queue.insert(event)       # ...act
```

Two callers can both pass the check before either inserts. The window is small
and the bug is real, and it is exactly the kind of thing that survives testing
and fails in production under load.

The fix is to make the check and the act the same statement by letting the
database enforce it:

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

No interleaving defeats this, because uniqueness is enforced at the point of
write by the only component that sees all writers. The dedup key is the request
id the client already has -- nothing hashed or invented.

## 3. The guarantee, layer two: claiming is a transition

Deduplicating the enqueue is not enough. Two workers draining the queue can both
read the same pending row and both handle it. The fix has the same shape:

```sql
UPDATE events SET status = 'claimed', claimed_at = datetime('now')
WHERE source IN (SELECT source FROM events WHERE status = 'pending' LIMIT ?)
RETURNING source, request_id, document_ids;
```

One statement moves the rows out of `pending` and reports which rows it moved.
The second worker's `WHERE` clause no longer matches. On Postgres this is
`SELECT ... FOR UPDATE SKIP LOCKED`; the principle is identical, and it is the
principle that transfers: **claiming is a state transition, never a read
followed by a write.**

## 4. What the guarantee actually is

This is **at-most-once**. A worker that dies mid-handle leaves the event claimed
and unfinished, and nothing retries it.

| Guarantee | Property | How to get it |
|-----------|----------|---------------|
| At-most-once | Never double-fires, may drop | Claim and never retry (what this builds) |
| At-least-once | Never drops, may double-fire | Add a lease and a sweeper for expired claims |
| Exactly-once | Not achievable end-to-end | Unless the downstream effect is itself idempotent |

"Exactly-once delivery" is marketing. What is achievable is **effectively-once
processing**: at-least-once delivery plus an idempotent downstream effect. The
dedup key in this design is precisely the idempotency key that makes that
upgrade safe, which is why the at-most-once version is a legitimate stopping
point rather than a dead end.

Moving to at-least-once is a small change: add `claimed_until`, and a sweeper
that returns expired claims to `pending`. What that costs is a hard requirement
that the downstream call be idempotent, because it *will* be retried.

## 5. Failure is recorded, not lost

An event whose handler raises is marked `failed`, not dropped and not retried
silently. The lifecycle is `pending -> claimed -> completed | failed`, and every
state is queryable.

"It disappeared" is the one outcome an operator cannot debug. A failed event
that is visible in `GET /events?status=failed` is an incident with a work list;
a swallowed exception is an unexplained gap in the data.

## 6. Layering, and the seam that removes mocks

```
route table   HTTP only: routes to service calls, domain errors to status codes
service       application logic: requests, submission, draining
queue         persistence: the two atomic dedup points
domain        Event, Status, and the DownstreamSystem protocol
```

The service raises `NotFound`; only the route table knows that means 404. Push
HTTP vocabulary any deeper and the service becomes untestable without a client.

The downstream system is a `Protocol`, not a base class or a patched import.
Tests pass a recording fake that counts invocations; production passes the real
client. That seam is what makes the central guarantee directly assertable --
`len(sink.calls) == 1` is the whole test -- without a mocking library.

**What was deliberately left out**: the request store is in-memory and does not
survive a restart, while the event queue is durable. That asymmetry is the
design. The part that must never double-fire is the part that gets durability;
build-up state is cheap to reconstruct and not worth the cost. Naming what you
did not build, and why, is part of the design rather than an admission.

## Build Log

1. **Write the spec before the code.** Endpoints, payloads, and the dedup rule
   stated in prose. This build was explicitly two-phase -- spec, then
   implementation -- and the spec is where the submit-is-its-own-endpoint and
   duplicate-is-200 decisions actually got made.
2. **Model the domain**: `Event`, `Status`, the `DownstreamSystem` protocol. No
   framework, no database, no HTTP.
3. **The queue, with the primary key as the dedup**, and `enqueue_once`
   returning a boolean rather than raising.
4. **The atomic claim**, written as a transition from the start. Retrofitting
   this onto a read-then-write is harder than writing it correctly once.
5. **The service layer**: create, upload, submit, drain. Domain errors only.
6. **Single-threaded tests** for the happy path, double submit, double drain,
   and the failure path.
7. **The concurrency tests last, and they are the real ones.** Eight threads
   submitting through a barrier; six workers draining fifty events through a
   barrier. Everything before this passes against a broken implementation.
8. **The route table**, mapping domain errors to status codes, tested without a
   server.

Step 7 is the step people skip. Single-threaded tests of a concurrency
guarantee prove nothing at all -- a check-then-act implementation passes every
test in step 6.

## Technique Catalog

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

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Idempotency, DLQs, consumer groups | [Distributed Workers](../../ai-platform-engineering/05-durable-orchestration-and-workers/) | The same patterns at broker scale |
| Durable execution, leases, checkpoints | [Durable Orchestration](../../ai-platform-engineering/05-durable-orchestration-and-workers/) | What at-least-once needs to be safe |
| Constraints, transactions, isolation | [Storage & Warehousing](../../data-engineering/02-storage-warehousing/) | Why the constraint is atomic |
| API design and status-code semantics | [System Design](../../systems/01-system-design/) | Contract design under retries |
| Fakes over mocks, testing behaviour | [The Testing Mentality](../../software-craftsmanship/03-testing-mentality/) | The Protocol seam |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| Payments and fintech | Double-charge prevention is this exact problem | Idempotency keys |
| Any webhook consumer | Redelivery is guaranteed, not exceptional | At-least-once plus idempotent effects |
| Any queue-backed backend | Claim semantics and poison-message handling | Transitions over read-then-write |
| Any API design round | Retry semantics, status codes, resource shape | Contracts that survive clients |
