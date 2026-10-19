# Task queue: visibility timeout, fenced leases, retries to DLQ, long poll

## Overview

| | |
|---|---|
| **Module** | `dur.03` · build · Go · Pass 8 · 4 to 6 h |
| **Solution** | [`go/durable/queue/queue.go`](../../../../course/solutions/go/durable/queue/queue.go): `Open`, `Enqueue`, `Poll`, `Check`, `Heartbeat`, `Complete`, `Fail`, `DLQ`, `Redrive`, `Depth`; model property tests in [`queue_learner_test.go`](../../../../course/solutions/learner-tests/dur.03/go/durable/queue/queue_learner_test.go) |
| **Contract** | the Go API is section 4 of this chapter; the transitions it persists are the `TaskTransition` kinds of [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto) |
| **Needs** | `dur.01` the log; reading: [Messaging and queueing](../../infrastructure/02-messaging-and-queueing/), [case study 03, exactly-once event API](../../case-studies/03-exactly-once-event-api/) |
| **Used by** | `dur.02` enqueues every workflow task · `dur.04` leases every workflow and activity task |
| **Optional depth** | the Amazon SQS visibility timeout and dead-letter queue docs; Kleppmann, *How to do distributed locking* (2016), on fencing tokens |

## Key Takeaways

- A worker does not take a task, it **leases** it until a deadline; a worker that dies simply never answers, and the task becomes visible again at the deadline as the next attempt.
- Every lease carries a **fencing token** larger than all before it; only the newest token, before its deadline, may complete, fail, or heartbeat, so a worker presumed dead cannot overwrite the live one.
- Failures retry after a delay; the attempt that reaches `MaxAttempts` parks the task in the dead-letter queue, from where an operator redrives it.
- A long poll is a wakeup, not a sleep loop: it returns the instant a task is enqueued or a retry comes due.
- Every transition is a log record first: after a crash, `Open` rebuilds every lease, retry, and dead letter, and new tokens keep growing.

## How to read this chapter

Sections 1 to 3 build the queue from first principles and work a full timeline by hand. Section 4 fixes the interface. Section 5 walks through the solution (persist and apply, lazy expiry, `Poll`, fencing, `Fail`), explains why each choice is the right one, and shows the model property tests. The solution works with `Options.Log` nil (memory only) or over the `dur.01` log; the logic is the same. Try to predict each step before reading it.

---

## 1. Why now

The durable server (`dur.02`) records what must happen; something has to hand that work to workers and survive the workers dying. A plain channel or an in-memory list loses the task when the process holding it dies, and "remove it when a worker takes it" loses it when the worker dies. This module is the queue between them: at-least-once delivery that survives both the server and the workers, with the fencing that lets a slow worker be replaced without two workers both writing a result, and a dead-letter queue so one poison task cannot burn every worker forever (drill `ops.03`).

## 2. Principles

### 2.1 Leases and the visibility timeout

| Symbol | Meaning |
|---|---|
| $V$ | visibility timeout: the lease length (per task, else the queue default, 30 s) |
| $t$ | the clock's now |
| $d$ | a lease's deadline, $t_{\text{poll}} + V$, moved by heartbeats |
| $k$ | a lease's fencing token |
| $a$ | the attempt number of a delivery, 1 for the first |
| $M$ | `MaxAttempts`: deliveries before the dead-letter queue (per task, else the default, 5) |

`Poll` picks a visible task, marks it leased with deadline $d = t + V$, and returns a lease. The task stays in the queue, invisible to other pollers, until one of: `Complete` (it leaves), `Fail` (it is retried, dead-lettered, or dropped), or $t \ge d$ (the lease **expires**: the worker is presumed dead). An expired lease turns the task visible again at $d$, and the next delivery is attempt $a + 1$. Nothing needs to watch the clock for this: whoever next looks at the queue (a poll, a depth, a DLQ listing) treats any lease with $t \ge d$ as expired. That is also why an expiry needs no log record: the lease record already says when it ends.

**Heartbeats.** A long task keeps its lease by calling `Heartbeat`, which moves $d$ to $t + V$ and stores small **details** (a checkpoint path). If the worker dies anyway, the next attempt's lease carries those details, so it can resume instead of starting over.

### 2.2 Fencing tokens

Each delivery gets a token $k$ strictly larger than every token handed out before, over all queues, and after a restart too (recovery restores the counter from the lease records). A lease is **live** while it is the task's current lease ($k$ matches) and $t < d$. `Check`, `Heartbeat`, `Complete`, and `Fail` refuse anything else with `ErrLeaseLost` and change nothing. Without fencing, a worker that paused for 40 s (garbage collection, a slow disk) comes back after its task was redelivered and completes it a second time, with a stale result.

### 2.3 Retries and the dead-letter queue

`Fail(lease, TaskError)` decides by the attempt number:

| Case | Disposition | What happens |
|---|---|---|
| `NonRetryable` | `Dropped` | the task leaves the queue now (the server records a terminal failure) |
| $a \ge M$ | `DeadLettered` | parked in the DLQ with $a$ and the last error |
| otherwise | `Retrying` | visible again at $t$ + `RetryAfter` (the backoff of `dur.05`) |

An expired lease of the last attempt ($a \ge M$) dead-letters the task too: a task that crashes its worker every time (a poison task) ends in the DLQ instead of looping. `Redrive` moves dead letters back, visible now with the attempt count reset, after the operator fixed the cause.

### 2.4 Order and the long poll

Among visible tasks, `Poll` takes the one that became visible first, ties in enqueue order, so a retried task waits its turn and old work is not starved. With nothing visible, `Poll` waits up to `wait` for the earliest of: an enqueue (or redrive, or retry) waking it, the next retry's visible time, the next lease deadline, or its own end, then returns `ErrNoTask`. The wakeup is a channel that is closed and replaced on every change (a broadcast), and the timed waits go through the injected `Clock` so tests control time exactly.

### 2.5 Durability and idempotent enqueue

Each transition is appended to stream `queue/<name>` of the log as a small JSON record (`enq`, `lease`, `hb`, `retry`, `dead`, `redrive`, `ack`, `drop`) **before** memory changes; if the append fails, nothing changes. `Open` replays the records to rebuild every queue. `Enqueue` of a task id the queue still holds (visible, leased, or dead) is a no-op: the server re-enqueues every pending task after a crash, and this is what keeps that from doubling work. Once a task is acknowledged its id is free again.

**Exactly-once effects.** The queue delivers at least once; the effect happens once only if it is idempotent: keyed by the task, so a second delivery finds the effect already applied. That is the property the exactly-once-under-chaos case checks, and the job of the idempotency keys of `dur.05`.

## 3. Worked example by hand

Queue `data`, $V = 30$ s, $M = 3$, fake clock at $t = 0$. Tasks `a` and `b` are enqueued.

| $t$ (s) | Call | Result | Why |
|---|---|---|---|
| 0 | `Poll` | `a`, attempt 1, token 1, deadline 30 | visible first |
| 0 | `Poll` | `b`, attempt 1, token 2, deadline 30 | |
| 0 | `Complete(b)` | ok, `b` leaves | live lease |
| 10 | `Fail(a, RetryAfter 5 s)` | `Retrying`, visible at 15 | $a = 1 < 3$ |
| 10 | `Poll` (wait 0) | `ErrNoTask` | `a` hidden until 15 |
| 15 | `Poll` | `a`, attempt 2, token 3, deadline 45 | the retry is due |
| 45 | `Poll` | `a`, attempt 3, token 4, deadline 75 | token 3's lease expired at 45: the worker died |
| 45 | `Complete(a, token 3)` | `ErrLeaseLost` | fenced: token 4 owns the task |
| 75 | `Poll` | `ErrNoTask` | attempt 3 expired and $3 \ge M$: dead-lettered at 75 |
| 75 | `DLQ` | `a`, 3 attempts, at 75 | |
| 75 | `Redrive` then `Poll` | `a`, attempt 1, token 5, deadline 105 | fresh attempt count |

## 4. The interface

```go
package queue // import "tinyllm/durable/queue"

type Clock interface { Now() time.Time; After(d time.Duration) <-chan time.Time }

type Task struct {
	ID          string
	Payload     []byte
	MaxAttempts int           // 0 = Options.MaxAttempts
	Visibility  time.Duration // 0 = Options.Visibility
}
type Lease struct {
	Queue    string
	Task     Task
	Token    uint64
	Attempt  int
	Deadline time.Time
	Details  []byte // last heartbeat details, kept across attempts
}
type TaskError struct { Message string; NonRetryable bool; RetryAfter time.Duration }
type Disposition int // Retrying, DeadLettered, Dropped
type DeadLetter struct { Task Task; Attempts int; LastError string; At time.Time }
var ErrNoTask, ErrLeaseLost, ErrNotFound error

type Options struct {
	Log *log.Log; Clock Clock     // nil log: memory only; nil clock: the wall clock
	Visibility time.Duration       // 0 = 30 s
	MaxAttempts int                // 0 = 5
}

func Open(ctx context.Context, o Options) (*Queue, error)
func (q *Queue) Enqueue(ctx context.Context, name string, t Task) error
func (q *Queue) Poll(ctx context.Context, name, worker string, wait time.Duration) (Lease, error)
func (q *Queue) Check(l Lease) error
func (q *Queue) Heartbeat(ctx context.Context, l Lease, details []byte) (Lease, error)
func (q *Queue) Complete(ctx context.Context, l Lease) error
func (q *Queue) Fail(ctx context.Context, l Lease, e TaskError) (Disposition, error)
func (q *Queue) DLQ(ctx context.Context, name string) ([]DeadLetter, error) // enqueue order
func (q *Queue) Redrive(ctx context.Context, name string, ids ...string) (int, error)
func (q *Queue) Depth(name string) (visible, leased, dead int)
```

A fake clock that implements `Now` and `After` satisfies `Clock`, which is how the timeline above is reproduced exactly. A lease is identified by `Queue`, `Task.ID`, and `Token`; the server rebuilds one from a task token with exactly those three fields.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand timeline | unit | section 3, row for row | the definition the server relies on |
| duplicate enqueue is a no-op | unit, boundary | an id held as visible, leased, or dead is not reset; after an ack it is free | recovery re-enqueues everything pending |
| FIFO among visible | unit | delivered in visible order, ties by enqueue | retries do not jump the line |
| stale token: lease lost | unit, fault | expired or superseded leases are refused by every call; the live one is undisturbed | a slow worker cannot overwrite the live attempt |
| heartbeat extends the lease | unit | the deadline moves to now + V; details reach the next attempt | long activities, resume from checkpoint |
| poison task to DLQ after max attempts | unit, fault | dispositions in order; DLQ with attempts and last error; per-task `MaxAttempts` | drill `ops.03` |
| non-retryable dropped | unit | a non-retryable failure leaves the queue at once | bad input is not retried |
| redrive | unit | one id, all, an unknown id changes nothing, attempts restart | `wf dlq redrive` |
| long poll wakes on enqueue | unit | a waiting poll returns on enqueue without the clock moving | workers react at once |
| long poll wakes on retry due | unit | a waiting poll returns when a retry becomes due | the same, for retries |
| long poll times out | boundary | `ErrNoTask` at the end of the wait | the server's 30 s empty answer |
| crashed worker: task reappears | fault, boundary | hidden until exactly V, attempt 2 at V | a SIGKILLed worker |
| recovery from the log | fault | visible, leased, retrying, and dead tasks rebuilt; a live lease survives; tokens never repeat | a server crash |
| log failure changes nothing | fault | an append that fails (quota) leaves memory unchanged | memory never runs ahead of the log |
| concurrent pollers, one lease each | property | 8 pollers, 200 tasks, 200 distinct leases and tokens | many workers |
| exactly-once effects under chaos | fault, property | a seeded schedule of deaths and late answers; each task acked once, each effect key applied once | at-least-once plus idempotence |

The model property tests state these as properties: under any schedule of polls, failures, expiries, and late completions, a task is acknowledged at most once and only by its newest lease; tokens strictly increase, across a reopen too; nothing is delivered before it is visible; a task with $M$ failures ends in the DLQ with $M$ attempts.

## 5. The solution, step by step

### From first principles

1. **A worker can die at any instant, silently.** So a task cannot be removed when it is handed out; it must stay in the queue, owned for a bounded time. That is a lease with a deadline, and expiry is the failure detector.
2. **Expiry needs no actor.** A lease's deadline is recorded when it is granted, so whether it has expired is a pure function of the clock. Every operation that looks at a queue first applies that function (`expire`); no background sweeper, no extra log record.
3. **A presumed-dead worker may not be dead.** After its lease expires and the task is redelivered, the old worker may still answer. To make that harmless, each delivery gets a strictly increasing token and every mutating call must present the current one before its deadline: a fencing token. Strictly increasing across restarts means recovery must restore the counter.
4. **Failures need a bound.** A task that fails every time would otherwise loop forever. Counting deliveries and parking the task at $M$ attempts gives the dead-letter queue; the same rule applies to an expired last lease, because a poison task usually kills its worker rather than reporting failure.
5. **Waiting must be event-driven.** A long poll that sleeps and rechecks either wastes CPU or adds latency. A per-queue broadcast channel (closed and replaced on every change) plus a timed wait for the next known due time (a retry becoming visible, a lease expiring) gives an immediate wakeup with no spinning.
6. **Memory follows the log.** Every transition is appended first and applied second, through one `apply` used by both live calls and recovery, so a failed append changes nothing and a restart rebuilds the same queues.
7. **Recovery re-enqueues, so enqueue must be idempotent.** An id the queue still holds in any state is a no-op; after an ack the id is free again.

### The code, annotated

**The item and the transition record** ([`queue.go`](../../../../course/solutions/go/durable/queue/queue.go)).

```go
type item struct {
	task      Task
	state     state     // visible, leased, or dead
	attempt   int       // deliveries so far
	seq       uint64    // enqueue order, the FIFO tie breaker
	visibleAt time.Time // a retry is hidden until here
	token     uint64    // the current lease's fencing token
	deadline  time.Time
	details   []byte    // last heartbeat details, kept across attempts
	// ... lastErr, deadAt
}

type named struct {
	items map[string]*item
	wake  chan struct{} // closed and replaced whenever a task may have become pollable
}

type transition struct {
	Kind    string `json:"k"` // enq lease hb retry dead redrive ack drop
	ID      string `json:"id"`
	// ... payload, max attempts, visibility
	Token   uint64 `json:"tok,omitempty"`
	Attempt int    `json:"att,omitempty"`
	UntilMs int64  `json:"until_ms,omitempty"` // lease deadline, or when a retry is visible
	// ... details, error, at
}
```

Every time in a record is an absolute millisecond timestamp, not a duration, so replay at a later wall-clock time reproduces the same deadlines.

**Persist, then apply.**

```go
func (q *Queue) persist(ctx context.Context, name string, tr transition) error {
	if q.o.Log != nil {
		data, err := json.Marshal(tr)
		// ...
		// dlog.Any: queue records need no optimistic check, q.mu already serialises them.
		if _, err := q.o.Log.Append(ctx, streamOf(name), dlog.Any, dlog.Event{Type: tr.Kind, Data: data}); err != nil {
			return err // nothing applied: memory never runs ahead of the log
		}
	}
	q.apply(name, tr)
	return nil
}

func (q *Queue) apply(name string, tr transition) {
	n := q.named(name)
	it := n.items[tr.ID]
	switch tr.Kind {
	case "enq":
		q.seq++
		n.items[tr.ID] = &item{ /* task fields from tr */ seq: q.seq, visibleAt: fromMs(tr.AtUnixMs)}
	case "lease":
		it.state, it.token, it.attempt, it.deadline = leased, tr.Token, tr.Attempt, fromMs(tr.UntilMs)
		if tr.Token > q.tokens { // recovery: never reuse a token after a restart
			q.tokens = tr.Token
		}
	case "hb":
		it.deadline, it.details = fromMs(tr.UntilMs), tr.Details
	case "retry":
		it.state, it.visibleAt, it.lastErr = visible, fromMs(tr.UntilMs), tr.Error
	case "dead":
		it.state, it.lastErr, it.deadAt = dead, tr.Error, fromMs(tr.AtUnixMs)
	case "redrive":
		it.state, it.attempt, it.visibleAt = visible, 0, fromMs(tr.AtUnixMs) // fresh attempt count
	case "ack", "drop":
		delete(n.items, tr.ID) // the id is free again
	}
	q.notify(n) // any change may make a task pollable: wake every waiter
}
```

**Lazy expiry.**

```go
func (q *Queue) expire(n *named, now time.Time) {
	for _, it := range n.items {
		if it.state != leased || now.Before(it.deadline) {
			continue // still live: now < deadline
		}
		if it.attempt >= q.maxAttempts(it) {
			it.state, it.deadAt, it.lastErr = dead, it.deadline, "lease expired" // the poison-task path
		} else {
			it.state, it.visibleAt = visible, it.deadline // visible again exactly at the deadline
		}
	}
}
```

Because expiry is derived from the recorded deadline, the same leases expire the same way before and after a restart, which is why it needs no record.

**Poll: pick, lease, or wait.**

```go
func (q *Queue) Poll(ctx context.Context, name, worker string, wait time.Duration) (Lease, error) {
	end := q.o.Clock.Now().Add(wait)
	for {
		q.mu.Lock()
		now := q.o.Clock.Now()
		n := q.named(name)
		q.expire(n, now)
		var best *item
		next := end // the earliest instant something could change without a notify
		for _, it := range n.items {
			switch it.state {
			case visible:
				if !it.visibleAt.After(now) {
					// Oldest visible first, enqueue order on ties: retries wait their turn.
					if best == nil || it.visibleAt.Before(best.visibleAt) ||
						(it.visibleAt.Equal(best.visibleAt) && it.seq < best.seq) {
						best = it
					}
				} else if it.visibleAt.Before(next) {
					next = it.visibleAt // a retry coming due
				}
			case leased:
				if it.deadline.Before(next) {
					next = it.deadline // a lease about to expire
				}
			}
		}
		if best != nil {
			q.tokens++ // a fresh fencing token, larger than any handed out
			tr := transition{
				Kind: "lease", ID: best.task.ID, Token: q.tokens, Attempt: best.attempt + 1,
				UntilMs: ms(now.Add(q.visibility(best))),
			}
			err := q.persist(ctx, name, tr)
			l := q.leaseOf(name, best)
			q.mu.Unlock()
			if err != nil {
				return Lease{}, err
			}
			return l, nil
		}
		if !now.Before(end) {
			q.mu.Unlock()
			return Lease{}, ErrNoTask // wait 0 returns here at once
		}
		wake := n.wake // grab the channel under the lock, so no notify is missed
		q.mu.Unlock()
		select {
		case <-wake: // an enqueue, retry, redrive, ack...
		case <-q.o.Clock.After(next.Sub(now)): // the next known due time, through the injected clock
		case <-ctx.Done():
			return Lease{}, ctx.Err()
		}
	}
}
```

Taking `wake` before unlocking is the detail that prevents a lost wakeup: a notify after the unlock closes the very channel this poller is about to wait on.

**Fencing.** One predicate guards every mutating call.

```go
func (q *Queue) live(l Lease) (*named, *item, error) {
	n := q.queues[l.Queue]
	if n == nil {
		return nil, nil, ErrLeaseLost
	}
	it := n.items[l.Task.ID]
	// The current lease (same token), still leased, and strictly before its deadline.
	if it == nil || it.state != leased || it.token != l.Token || !q.o.Clock.Now().Before(it.deadline) {
		return nil, nil, ErrLeaseLost
	}
	return n, it, nil
}
```

`Check`, `Heartbeat`, `Complete`, and `Fail` all call `live` first, so an expired or superseded lease changes nothing. At exactly the deadline the lease is already lost, matching `expire`, so there is no instant at which two workers both own a task.

**Fail: drop, dead-letter, or retry.**

```go
switch {
case e.NonRetryable:
	return Dropped, q.persist(ctx, l.Queue, transition{Kind: "drop", ID: it.task.ID, Error: e.Message})
case it.attempt >= q.maxAttempts(it): // this was the M-th delivery
	return DeadLettered, q.persist(ctx, l.Queue, transition{Kind: "dead", ID: it.task.ID, Error: e.Message, AtUnixMs: ms(now)})
default:
	// Hidden until now + RetryAfter (the backoff dur.05 computes), not visible at once.
	return Retrying, q.persist(ctx, l.Queue, transition{Kind: "retry", ID: it.task.ID, Error: e.Message, UntilMs: ms(now.Add(e.RetryAfter))})
}
```

`Heartbeat` persists an `hb` record with `UntilMs = now + V` and a copy of the details; the details survive into the next lease because `leaseOf` copies `it.details`. `Enqueue` returns nil without a record when the id is already present. `Redrive` validates every id before moving any (so an unknown id moves nothing), then persists one `redrive` per id. `DLQ` sorts by `seq` so listing order is enqueue order regardless of map iteration.

Full solution: [`queue.go`](../../../../course/solutions/go/durable/queue/queue.go).

### The model property tests

This module's solution includes a test suite in [`queue_learner_test.go`](../../../../course/solutions/learner-tests/dur.03/go/durable/queue/queue_learner_test.go), on a fake clock. The fencing test shows the shape: let a lease reach its deadline, check that every call with the old token is refused, and that none of them disturbed the live one.

```go
old := poll(t, q, "q")
clk.Advance(10 * time.Second) // deadline reached, not yet redelivered
// Check and Complete with old: ErrLeaseLost
cur := poll(t, q, "q")
if cur.Token <= old.Token || cur.Attempt != 2 { /* tokens must grow, attempts count */ }
// Heartbeat, Fail, Complete with old: ErrLeaseLost
if err := q.Check(cur); err != nil { /* the stale calls disturbed the live lease */ }
if err := q.Complete(ctx, cur); err != nil { /* ... */ }
if err := q.Complete(ctx, cur); !errors.Is(err, queue.ErrLeaseLost) { /* a second ack must fail */ }
```

The suite's chaos test drives a seeded schedule of worker deaths and late answers and asserts each task is acknowledged once and each effect key applied once: at-least-once delivery plus idempotent effects.

### Why this solution

- **Cost.** `Poll` and `expire` scan the queue's items: O(n) per poll. That is simple and correct for course-sized queues; a production queue keeps a min-heap by `visibleAt` and another by deadline to make both O(log n). Every transition is one log append.
- **Lazy expiry over timers.** One timer per lease would need cancelling on every heartbeat and ack and rebuilding after recovery. Deriving expiry from the recorded deadline is the same answer with no moving parts.
- **Global token counter.** One counter over all queues is simpler than per-task counters and still strictly increasing per task, which is all fencing needs.
- **Visibility timeout over explicit liveness.** The queue never needs to know whether a worker is alive; a missing answer is indistinguishable from a dead worker, and the lease treats them the same.
- **Accepted tradeoff.** Delivery is at least once. A task can be executed twice (a worker that finished but died before acknowledging), so effects must be idempotent; fencing only guarantees that the second execution's late answer cannot overwrite the first's.

### Explain it back

1. In the worked example, why is `Complete(a, token 3)` at $t = 45$ refused when the worker holding token 3 may have finished its work?

   <details><summary>Answer</summary>

   At 45 its lease reached its deadline and the task was leased again with token 4. `live` requires the current token and `now < deadline`; token 3 fails both. Accepting it would let two workers act on one task, and the stale one could overwrite the live attempt.

   </details>

2. Why does an expired lease need no log record, while a retry does?

   <details><summary>Answer</summary>

   The lease record already stores its absolute deadline, so expiry is a function of the clock that `expire` evaluates identically before and after a restart. A retry's visibility time depends on `RetryAfter` and the moment `Fail` was called, which nothing else records.

   </details>

3. Why must recovery set `q.tokens` to the largest token seen?

   <details><summary>Answer</summary>

   If the counter restarted at 0, a new lease after the restart could get a token equal to one a pre-crash worker still holds, and that worker's late `Complete` would pass the fencing check.

   </details>

4. What makes a duplicate `Enqueue` a no-op, and why does the server depend on it?

   <details><summary>Answer</summary>

   `Enqueue` returns early when the id exists in any state. The server re-enqueues every pending task on recovery because it cannot know which enqueues reached the log before the crash; idempotence turns that blind redo into exactly one task.

   </details>

5. Why does `Poll` read `n.wake` before releasing the mutex?

   <details><summary>Answer</summary>

   If it released the lock first, an `Enqueue` could notify (close and replace the channel) in between, and the poller would then wait on the new channel and miss that enqueue until its timeout.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a lease still live at its deadline instant | at exactly $V$ the old and the new worker both own the task | `live` and `expire` both use `now.Before(deadline)`: at the deadline the lease is gone |
| 2. no fencing: the token is not compared, an expired lease may still act, every lease gets the same token, or the counter restarts after a crash | a stale worker completes, fails, or heartbeats a task someone else owns | `live` compares the token and the deadline; `Poll` increments `q.tokens`; `apply` restores the maximum on recovery |
| 3. the attempt count: one extra attempt, an expired last attempt redelivered forever, a non-retryable failure retried, a redrive that keeps the old count | poison tasks loop; redriven tasks dead-letter at once | `attempt >= maxAttempts` in both `Fail` and `expire`; `NonRetryable` drops first; `redrive` resets `attempt` to 0 |
| 4. a retry visible at once, ignoring its delay | a failing dependency is hammered | a `retry` record sets `visibleAt = now + RetryAfter` |
| 5. a heartbeat that does not extend the lease, or loses its details | long tasks are redelivered while running; resumes start over | `Heartbeat` persists `UntilMs = now + V` and the details; `leaseOf` carries them |
| 6. order and wakeups: newest first, no broadcast on change, no wake for a retry coming due, spinning at the end of the wait | starvation; workers idle while tasks wait; a CPU at 100% | oldest `visibleAt` then `seq`; `notify` on every `apply`; a timed wait to `next`; `ErrNoTask` once `now >= end` |
| 7. a duplicate enqueue that replaces the task, or an ack that does not remove it | recovery resets a running task; finished tasks come back | `Enqueue` returns early for a held id; `ack` deletes the item |
| 8. changing memory before the record is durable | after a restart the queue and its log disagree | `persist` appends before `apply` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.01` | stream `queue/<name>` holds every transition; `Open` replays it |
| Forward | `dur.02` | each `WorkflowTaskScheduled` enqueues `<run>/wt/<id>` on `wf:<queue>`; recovery re-enqueues everything pending |
| Forward | `dur.04` | `Poll*Task` leases, `Check` fences completions, `Complete` acknowledges |
| Forward | `dur.05` | `Fail` with the jittered backoff; `Heartbeat`; `DLQ` and `Redrive` behind `wf dlq` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one queue per name, in one process | Temporal matching service | task queues partitioned over hosts, with forwarding between partitions | `temporalio/temporal`: `service/matching` |
| every task through the queue | sticky task queues | a workflow's next task goes to the worker that has its state cached, so history is not replayed each time | Temporal "Sticky Execution" |
| unbounded polling | task-queue rate limits | caps tasks per second per queue to protect downstream systems | Temporal `TaskQueueActivitiesPerSecond` |
| poll for every activity | eager activity dispatch | an activity scheduled by a workflow task is returned in the completion response itself | Temporal "Eager Activity Execution" |
