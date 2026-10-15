<!-- ol:module dur.03 -->
# Task queue: visibility timeout, fenced leases, retries to DLQ, long poll

## Overview

| | |
|---|---|
| **Module** | `dur.03` · build · Go · Pass 8 · 4 to 6 h |
| **You build** | `go/durable/queue/queue.go`: `Open`, `Enqueue`, `Poll`, `Check`, `Heartbeat`, `Complete`, `Fail`, `DLQ`, `Redrive`, `Depth` |
| **Contract** | the Go API is section 4 of this chapter, held by the course tests; the transitions it persists are the `TaskTransition` kinds of [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto) |
| **Tests** | `course/tests/go/dur_03/` (what they check: section 4) · your own tests in `go/durable/queue/queue_learner_test.go`, rung R4, graded by mutation (threshold 0.80, every pitfall's mutant required) |
| **Needs** | `dur.01` the log; reading: [Messaging and queueing](../../infrastructure/02-messaging-and-queueing/), [case study 03, exactly-once event API](../../../case-studies/03-exactly-once-event-api/) |
| **Used by** | `dur.02` enqueues every workflow task · `dur.04` leases every workflow and activity task |
| **Milestone** | MS-durable |
| **Optional depth** | the Amazon SQS visibility timeout and dead-letter queue docs; Kleppmann, *How to do distributed locking* (2016), on fencing tokens |

## Key Takeaways

- A worker does not take a task, it **leases** it until a deadline; a worker that dies simply never answers, and the task becomes visible again at the deadline as the next attempt (`TestCrashedWorkerTaskReappears`).
- Every lease carries a **fencing token** larger than all before it; only the newest token, before its deadline, may complete, fail, or heartbeat, so a worker presumed dead cannot overwrite the live one (`TestStaleTokenLeaseLost`).
- Failures retry after a delay; the attempt that reaches `MaxAttempts` parks the task in the dead-letter queue, from where an operator redrives it (`TestPoisonTaskToDLQAfterMaxAttempts`, `TestRedrive`).
- A long poll is a wakeup, not a sleep loop: it returns the instant a task is enqueued or a retry comes due (`TestLongPollWakesOnEnqueue`, `TestLongPollWakesOnRetryDue`).
- Every transition is a log record first: after a crash, `Open` rebuilds every lease, retry, and dead letter, and new tokens keep growing (`TestRecoveryFromLog`).

## How to work this chapter

```bash
ol start dur.03
ol tests dur.03
ol check dur.03          # needs dur.01 (or --ref-deps)
ol diff  dur.03
```

Build it in memory first (`Options.Log` nil) and pass the timeline of section 3, then persist each transition.

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

**Exactly-once effects.** The queue delivers at least once; the effect happens once only if it is idempotent: keyed by the task, so a second delivery finds the effect already applied. That is the contract of `effects.AssertExactlyOnce` in the course tests, and of the idempotency keys of `dur.05`.

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

This is `TestHandTimeline`, row for row.

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

The testkit's `*clock.Fake` satisfies `Clock`. A lease is identified by `Queue`, `Task.ID`, and `Token`; the server rebuilds one from a task token with exactly those three fields.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandTimeline` | unit | section 3, row for row | the definition the server and the tests share |
| `TestEnqueueDuplicateIsNoop` | unit, boundary | an id held as visible, leased, or dead is not reset; after an ack it is free | recovery re-enqueues everything pending |
| `TestFIFOAmongVisible` | unit | delivered in visible order, ties by enqueue | retries do not jump the line |
| `TestStaleTokenLeaseLost` | unit, fault | expired or superseded leases are refused by every call; the live one is undisturbed | a slow worker cannot overwrite the live attempt |
| `TestHeartbeatExtendsLease` | unit | the deadline moves to now + V; details reach the next attempt | long activities, resume from checkpoint |
| `TestPoisonTaskToDLQAfterMaxAttempts` | unit, fault | dispositions in order; DLQ with attempts and last error; per-task `MaxAttempts` | drill `ops.03` |
| `TestNonRetryableDropped` | unit | a non-retryable failure leaves the queue at once | bad input is not retried |
| `TestRedrive` | unit | one id, all, an unknown id changes nothing, attempts restart | `wf dlq redrive` |
| `TestLongPollWakesOnEnqueue` | unit | a waiting poll returns on enqueue without the clock moving | workers react at once |
| `TestLongPollWakesOnRetryDue` | unit | a waiting poll returns when a retry becomes due | the same, for retries |
| `TestLongPollTimesOut` | boundary | `ErrNoTask` at the end of the wait | the server's 30 s empty answer |
| `TestCrashedWorkerTaskReappears` | fault, boundary | hidden until exactly V, attempt 2 at V | a SIGKILLed worker |
| `TestRecoveryFromLog` | fault | visible, leased, retrying, and dead tasks rebuilt; a live lease survives; tokens never repeat | a server crash |
| `TestLogFailureChangesNothing` | fault | an append that fails (quota) leaves memory unchanged | memory never runs ahead of the log |
| `TestConcurrentPollersOneLeaseEach` | property | 8 pollers, 200 tasks, 200 distinct leases and tokens | many workers |
| `TestExactlyOnceEffectsUnderChaos` | fault, property | a seeded schedule of deaths and late answers; each task acked once, each effect key applied once | at-least-once plus idempotence |

Your own tests (rung R4) go in `go/durable/queue/queue_learner_test.go` as `package queue_test`, on the testkit's fake clock (`openlearn.urmzd.com/tl/testkit/clock`), written as properties: under any schedule of polls, failures, expiries, and late completions, a task is acknowledged at most once and only by its newest lease; tokens strictly increase, across a reopen too; nothing is delivered before it is visible; a task with $M$ failures ends in the DLQ with $M$ attempts. `ol check dur.03` grades them against the reference with one planted bug at a time: 80% must be caught, and every required one.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a lease still live at its deadline instant | at exactly $V$ the old and the new worker both own the task | `TestCrashedWorkerTaskReappears`, `TestStaleTokenLeaseLost` (mutant `s03`) |
| 2. no fencing: the token is not compared, an expired lease may still act, every lease gets the same token, or the counter restarts after a crash | a stale worker completes, fails, or heartbeats a task someone else owns | `TestStaleTokenLeaseLost`, `TestHandTimeline`, `TestRecoveryFromLog` (mutants `s04`, `s05`, `s06`, `s16`) |
| 3. the attempt count: one extra attempt, an expired last attempt redelivered forever, a non-retryable failure retried, a redrive that keeps the old count | poison tasks loop; redriven tasks dead-letter at once | `TestPoisonTaskToDLQAfterMaxAttempts`, `TestHandTimeline`, `TestNonRetryableDropped`, `TestRedrive` (mutants `s09`, `s10`, `s11`, `s12`) |
| 4. a retry visible at once, ignoring its delay | a failing dependency is hammered | `TestHandTimeline` (mutant `s01`) |
| 5. a heartbeat that does not extend the lease, or loses its details | long tasks are redelivered while running; resumes start over | `TestHeartbeatExtendsLease` (mutants `s07`, `s08`) |
| 6. order and wakeups: newest first, no broadcast on change, no wake for a retry coming due, spinning at the end of the wait | starvation; workers idle while tasks wait; a CPU at 100% | `TestFIFOAmongVisible`, `TestLongPollWakesOnEnqueue`, `TestLongPollWakesOnRetryDue`, `TestLongPollTimesOut` (mutants `s02`, `s14`, `s15`, `s19`) |
| 7. a duplicate enqueue that replaces the task, or an ack that does not remove it | recovery resets a running task; finished tasks come back | `TestEnqueueDuplicateIsNoop`, `TestExactlyOnceEffectsUnderChaos` (mutants `s13`, `s18`) |
| 8. changing memory before the record is durable | after a restart the queue and its log disagree | `TestLogFailureChangesNothing` (mutant `s17`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.01` | stream `queue/<name>` holds every transition; `Open` replays it |
| Forward | `dur.02` | each `WorkflowTaskScheduled` enqueues `<run>/wt/<id>` on `wf:<queue>`; recovery re-enqueues everything pending |
| Forward | `dur.04` | `Poll*Task` leases, `Check` fences completions, `Complete` acknowledges |
| Forward | `dur.05` | `Fail` with the jittered backoff; `Heartbeat`; `DLQ` and `Redrive` behind `wf dlq` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one queue per name, in one process | Temporal matching service | task queues partitioned over hosts, with forwarding between partitions | `temporalio/temporal`: `service/matching` |
| every task through the queue | sticky task queues | a workflow's next task goes to the worker that has its state cached, so history is not replayed each time | Temporal "Sticky Execution" |
| unbounded polling | task-queue rate limits | caps tasks per second per queue to protect downstream systems | Temporal `TaskQueueActivitiesPerSecond` |
| poll for every activity | eager activity dispatch | an activity scheduled by a workflow task is returned in the completion response itself | Temporal "Eager Activity Execution" |
