# Activities

With the protocol from [the task protocol and workers](04-task-protocol-and-workers.md), a failing activity is retried at once, forever. A downstream service that is down gets hammered by every worker, a bad input burns every attempt, and an activity that was killed after charging a card charges it again on the retry. None of that is acceptable for work that runs for hours and touches real systems.

This section makes failure a first-class part of the engine. You will add retries with exponential backoff and jitter, a dead-letter queue when they run out, non-retryable failures that stop at once, heartbeats that let a long activity keep its lease and resume from its last checkpoint, and the idempotency key that makes "at least once" safe. At the end, `<system> wf start FlakyDemo --fail-rate 0.5` completes, with every effect applied once.

The section builds on the worker and protocol of the previous section. The one piece of probability it needs, the expected value of a uniform draw, is worked in [Probability problem set, part a](../../mathematics/13-probability-statistics/09-problem-set-a.md) in *Mathematics*. The contract is `RetryPolicy`, `ActivityOptions`, `Failure`, and the heartbeat and fail RPCs in [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto); the Go API is given below. The full solution is the activity package [`go/durable/activity/activity.go`](../../../../course/solutions/go/durable/activity/activity.go) (`IdempotencyKey`, `Attempt`, `HeartbeatDetails`, `RecordHeartbeat`, `NewError`, `NewNonRetryable`, `IsNonRetryable`, `Backoff`, `Jitter`) and the server half [`go/durable/server/activities.go`](../../../../course/solutions/go/durable/server/activities.go) (`RecordHeartbeat`, `FailActivityTask`, `ListDeadLetters`, `RedriveDeadLetter`).

> **Learning objectives**
> - Compute an exponential backoff schedule with a cap, and the effect of full jitter on it.
> - Decide a failed attempt's disposition: retry, fail the workflow's activity, or dead-letter.
> - Explain why exhausted retries park the task for an operator rather than failing the workflow.
> - Use heartbeats to keep a long activity's lease and resume from a checkpoint.
> - Design idempotency keys that make at-least-once delivery apply each effect once.

## Exponential backoff

Start with the delay between attempts. The symbols, with their defaults, are:

| Symbol | Meaning | Default |
|---|---|---|
| $n$ | the attempt that just failed, $n \ge 1$ | |
| $b_0$ | `initial_ms` | 1 s |
| $r$ | `backoff`, the growth factor (below 1 counts as unset) | 2.0 |
| $b_{\max}$ | `max_interval_ms`, the cap | 60 s |
| $d_n$ | the backoff before attempt $n+1$ | |
| $U$ | a uniform random number in $[0, 1)$ | |
| $\delta_n$ | the delay actually used | |

The backoff is

$$d_n = \min\left(b_0 \cdot r^{\,n-1},\; b_{\max}\right)$$

The first retry waits $b_0$, each later one $r$ times longer, and none more than $b_{\max}$. A zero field means the default, because a policy left empty must not retry in a hot loop.

## Full jitter

Backoff alone still has a problem. If a dependency fails for 1,000 activities at once, they all retry after exactly $d_1$, fail together again, and so on: synchronized waves. Jitter spreads them out:

$$\delta_n = U \cdot d_n, \qquad U \sim \mathrm{Uniform}[0, 1).$$

Then $0 \le \delta_n < d_n$ and $\mathbb{E}[\delta_n] = d_n / 2$, the computation from the probability problem set. The expected wait halves, and retries land spread uniformly over the window. The server draws $U$ from its own seeded generator (`Options.Seed`); tests can replace the whole step with `Options.Jitter`.

## Retryable, non-retryable, dead-lettered

Knowing how long to wait is half the problem; the other half is whether to retry at all. `FailActivityTask(failure)` on a live lease takes one of three paths:

| Condition | History | Queue |
|---|---|---|
| `failure.non_retryable`, or `failure.type` listed in `RetryPolicy.non_retryable` | `ActivityTaskStarted`, `ActivityTaskFailed` (the workflow is woken and sees an error) | `Dropped` |
| the attempt reaches `max_attempts` (0 = `[durable].dlq_after_attempts`, 5) | `ActivityDeadLettered` (informational: the workflow keeps waiting) | `DeadLettered` |
| otherwise | nothing (the retry lives in the queue; describe shows the last failure) | `Retrying` after $\delta_n$ |

Why does an exhausted activity not fail the workflow? Because exhausting retries usually means a bug or an outage, which an operator fixes and then **redrives** (`RedriveDeadLetter`, which starts again at attempt 1). Failing the workflow would throw away hours of finished steps. `ListDeadLetters` shows each dead letter with its workflow, activity, attempts, and last failure (`<system> wf dlq list data`).

Activities choose their own failure type. `activity.NewNonRetryable("BadSpec", msg)` fails for good; `activity.NewError("Timeout", msg)` is retryable, with a type a policy can name. For example, the runner in [subprocess activities](09-subprocess-activities.md) maps exit code 65 to type `ExitCode65`, which policies list as non-retryable.

## Timeouts and heartbeats

An attempt's lease is its **start-to-close** timeout, or its **heartbeat timeout** $h$ when it has one. A heartbeating activity keeps its lease alive by calling `activity.RecordHeartbeat(ctx, details)` at least every $h$. Each heartbeat moves the deadline to now $+ h$ and stores the details (a checkpoint path, a byte offset). If the worker dies, the next attempt starts with `activity.HeartbeatDetails(ctx)` equal to the last details, so it resumes instead of starting over. A heartbeat with a stale lease is `FAILED_PRECONDITION`, which the worker turns into a canceled context.

## Idempotency keys: effectively once

All of this still delivers at least once: an attempt can apply its effect and die before reporting, so the next attempt applies it again. The fix is not in the engine but in the effect. Give the effect a key that is the same for every attempt of the activity, `activity.IdempotencyKey(ctx)` = `"<workflow_id>/<activity_id>"`, and make the receiver ignore a key it has seen, with a unique constraint, an output directory named by the key, or an effects table keyed by it. Putting the attempt number, worker id, or run id in the key would make every attempt a different effect.

## Working it by hand

Take the policy $b_0 = 1$ s, $r = 2$, $b_{\max} = 10$ s, `max_attempts` 4. The schedule, with full jitter evaluated at $U = 0.5$, is:

| Failed attempt $n$ | $b_0 r^{n-1}$ | $d_n$ | full jitter at $U = 0.5$ |
|---|---|---|---|
| 1 | 1 s | 1 s | 0.5 s |
| 2 | 2 s | 2 s | 1 s |
| 3 | 4 s | 4 s | 2 s |
| 4 | 8 s | 8 s | 4 s |
| 5 | 16 s | 10 s | 5 s |
| 6 | 32 s | 10 s | 5 s |

Now switch jitter off (`Options.Jitter` = identity) and start the clock at 0. Attempt 1 fails at 0, and attempt 2 is pollable at exactly 1 s (not at 999 ms). Attempt 2 fails, and attempt 3 is pollable 2 s later. Attempt 3 fails, and attempt 4 comes 4 s later. Attempt 4 fails, and since $4 \ge$ `max_attempts`, the task is dead-lettered with `ActivityDeadLettered{attempts 4}` and no workflow task. A redrive delivers it as attempt 1.

## Designing the API

The activity package gives activity code its helpers and the pure retry functions; the server adds four RPCs:

```go
package activity // import "tinyllm/durable/activity"

const DefaultInitial, DefaultBackoff, DefaultInterval = time.Second, 2.0, time.Minute

func IdempotencyKey(ctx context.Context) string // "<workflow_id>/<activity_id>"
func Attempt(ctx context.Context) int
func HeartbeatDetails(ctx context.Context) []byte
func RecordHeartbeat(ctx context.Context, details []byte)

type Error struct { Type, Message string; NoRetry bool; Cause error } // FailureType(), NonRetryable()
func NewError(typ, msg string) error
func NewNonRetryable(typ, msg string) error
func IsNonRetryable(err error) bool

func Backoff(p *durablev1.RetryPolicy, n int) time.Duration
func Jitter(d time.Duration, u float64) time.Duration // u in [0, 1)

// go/durable/server/activities.go (methods on *Server)
func (s *Server) RecordHeartbeat(ctx context.Context, req *durablev1.HeartbeatRequest) (*durablev1.HeartbeatResponse, error)
func (s *Server) FailActivityTask(ctx context.Context, req *durablev1.FailActivityRequest) (*durablev1.Empty, error)
func (s *Server) ListDeadLetters(ctx context.Context, req *durablev1.ListDeadLettersRequest) (*durablev1.ListDeadLettersResponse, error)
func (s *Server) RedriveDeadLetter(ctx context.Context, req *durablev1.RedriveDeadLetterRequest) (*durablev1.RedriveDeadLetterResponse, error)
```

## Building it step by step

The implementation follows from six observations. Try to predict each.

1. **A failed attempt has three possible futures: retry, give up and tell the workflow, or park for an operator.** Which one is right depends on whether retrying can help (non-retryable) and whether we have tried enough (max attempts). Those two inputs, and nothing else, decide the disposition.
2. **Retries must slow down.** If the cause is an overloaded dependency, retrying at a constant rate keeps it overloaded. Exponential growth backs off quickly; a cap stops the delay growing without bound; defaults for zero fields stop an empty policy becoming a hot loop.
3. **Retries must not synchronise.** Many tasks failing at the same instant would all retry at the same instant. Multiplying by a uniform $U \in [0, 1)$ spreads them over the window, at the cost of an expected wait of $d/2$ instead of $d$.
4. **Retry state belongs in the queue, not in history.** A retry is a delay before the next delivery, which is exactly what the queue's `Fail` with `RetryAfter` provides. Writing every retry to history would grow it with events the workflow never reacts to. Only outcomes the workflow or operator must see are recorded: `ActivityTaskFailed` (non-retryable) and `ActivityDeadLettered`.
5. **Exhaustion parks rather than fails.** A run that spent hours reaching this activity should not be thrown away because a dependency was down; the operator fixes it and redrives.
6. **Effects must tolerate re-execution.** Delivery is at least once, so the engine hands the activity a key that is identical on every attempt, and the effect's receiver deduplicates on it.

### The backoff formula

The two pure functions in [`activity.go`](../../../../course/solutions/go/durable/activity/activity.go) implement the formulas, with defaults and clamping:

```go
func Backoff(p *durablev1.RetryPolicy, n int) time.Duration {
	if n < 1 {
		n = 1
	}
	initial := time.Duration(p.GetInitialMs()) * time.Millisecond
	if initial <= 0 {
		initial = DefaultInitial // 1 s: a zero field must not mean "retry now"
	}
	mult := p.GetBackoff()
	if mult < 1 {
		mult = DefaultBackoff // below 1 would shrink delays: treat as unset
	}
	max := time.Duration(p.GetMaxIntervalMs()) * time.Millisecond
	if max <= 0 {
		max = DefaultInterval // 60 s
	}
	d := float64(initial) * math.Pow(mult, float64(n-1)) // n-1: the first retry waits exactly initial
	if d >= float64(max) {
		return max // compare in float64 before converting: r^(n-1) can overflow a Duration
	}
	return time.Duration(d)
}

func Jitter(d time.Duration, u float64) time.Duration {
	// Clamp u into [0, 1) so the result is always strictly below d.
	if u < 0 {
		u = 0
	}
	if u >= 1 {
		u = math.Nextafter(1, 0)
	}
	return time.Duration(u * float64(d)) // full jitter: replaces d, not added to it
}
```

Check it against the worked example's policy ($b_0 = 1$ s, $r = 2$, $b_{\max} = 10$ s): $n = 5$ gives $16$ s in float, which is at least the cap, so `Backoff` returns 10 s, and `Jitter(10 s, 0.5)` is 5 s.

### Typed errors

`Error` implements `FailureType()` and `NonRetryable()`, the two methods the worker's `failureOf` looks for with `errors.As`, and `Unwrap()`, so the classification survives `fmt.Errorf("...: %w", err)` wrapping:

```go
type Error struct {
	Type    string
	Message string
	NoRetry bool
	Cause   error
}

func (e *Error) Unwrap() error       { return e.Cause }
func (e *Error) FailureType() string { return e.Type }    // matched against RetryPolicy.non_retryable
func (e *Error) NonRetryable() bool  { return e.NoRetry }

func IsNonRetryable(err error) bool {
	var nr interface{ NonRetryable() bool }
	return errors.As(err, &nr) && nr.NonRetryable() // searches the whole wrap chain
}
```

The context helpers (`IdempotencyKey`, `Attempt`, `HeartbeatDetails`, `RecordHeartbeat`) read the `worker.Info` that the worker stores in the activity context, so activity code never sees the protocol.

### `FailActivityTask`: classify, then act

On the server, in [`activities.go`](../../../../course/solutions/go/durable/server/activities.go), `FailActivityTask` fences, classifies, and then either commits a terminal failure or hands the retry to the queue:

```go
s.mu.Lock()
defer s.mu.Unlock()
r, a, err := s.pendingActivity(tok) // fenced: a stale lease cannot fail the live attempt
if err != nil {
	return nil, err
}
// ...
policy := a.attrs.GetOptions().GetRetry()
// Non-retryable by the activity's flag, or by the policy naming its type (not its message).
if f.GetNonRetryable() || slices.Contains(policy.GetNonRetryable(), f.GetType()) {
	b := &batch{r: r}
	st := s.add(b, /* ActivityTaskStarted{scheduled, identity, attempt, lease token} */)
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActFailed{ActFailed: &durablev1.ActivityTaskFailed{
		ScheduledEventId: a.scheduledID, StartedEventId: st.GetEventId(), Failure: f,
	}}})
	if err := s.commit(ctx, b); err != nil { // act_failed is a trigger: the workflow is woken
		return nil, err
	}
	s.o.Queue.Fail(ctx, tok.lease(), queue.TaskError{Message: f.GetMessage(), NonRetryable: true}) // Dropped
	return &durablev1.Empty{}, nil
}
// Retryable: the queue decides Retrying or DeadLettered from its attempt count.
disp, err := s.o.Queue.Fail(ctx, tok.lease(), queue.TaskError{Message: f.GetMessage(), RetryAfter: s.retryDelay(policy, int(a.attempt))})
// ... ErrLeaseLost -> FAILED_PRECONDITION
a.lastFailure = f // describe shows it while the retry waits
if disp == queue.DeadLettered {
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_DeadLettered{DeadLettered: &durablev1.ActivityDeadLettered{
		ScheduledEventId: a.scheduledID, TaskQueue: tok.Queue, TaskId: tok.TaskID, Attempts: a.attempt, LastFailure: f,
	}}})
	if err := s.commit(ctx, b); err != nil { // not a trigger: the workflow keeps waiting
		return nil, err
	}
}
return &durablev1.Empty{}, nil
```

`retryDelay` computes `Backoff(policy, n)` for the attempt that just failed, then applies `Options.Jitter` if a test set one, and otherwise `Jitter(d, s.rng.Float64())` from the server's seeded generator. The draw happens under `s.mu`, so the stream is deterministic for a given seed and order. The `max_attempts` limit itself was already handed to the queue at enqueue time (by `enqueueActivity` in the workflow service, with `[durable].dlq_after_attempts` for 0), which is why the queue can decide `DeadLettered` on its own.

### Heartbeats

`RecordHeartbeat` checks the details size, fences with `pendingActivity`, then calls `Queue.Heartbeat`, which persists the new deadline and details. It records the details and time on the in-memory activity for describe, and answers with the new deadline in Unix milliseconds and `cancel_requested` (set only when this activity was asked to stop, as described in [signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md)):

```go
l, err := s.o.Queue.Heartbeat(ctx, tok.lease(), req.GetDetails())
if errors.Is(err, queue.ErrLeaseLost) {
	return nil, status.Error(codes.FailedPrecondition, "stale lease token") // the worker cancels its context
}
// ...
a.lastHeartbeat, a.lastHeartbeatMs = req.GetDetails(), s.o.Clock.Now().UnixMilli()
return &durablev1.HeartbeatResponse{
	CancelRequested: a.cancelRequested,
	DeadlineUnixMs:  l.Deadline.UnixMilli(), // milliseconds, as the proto says
}, nil
```

The next attempt gets the details from the queue lease (`l.Details` in `startActivityTask`), which survives a server restart because the queue's `hb` record holds them.

### Dead letters

`ListDeadLetters` reads the queue's DLQ for `act:<task_queue>`, pages it with an offset token, and enriches each entry with its workflow, run, activity id, and type by parsing the task id. `RedriveDeadLetter` redrives the given ids (or all), maps an unknown id to `NOT_FOUND` (not `UNAVAILABLE`, which would invite a retry), and clears `deadLettered` on each activity so describe and recovery treat it as pending again.

The full solution is in [`activity.go`](../../../../course/solutions/go/durable/activity/activity.go) and [`activities.go`](../../../../course/solutions/go/durable/server/activities.go).

### Why this design

- **Cost.** `Backoff` and `Jitter` are O(1). A retryable failure costs one queue append and no history event; only terminal outcomes cost a history commit.
- **Full jitter over equal or decorrelated jitter.** Full jitter is among the best performers for a contended dependency in Brooker's comparison, and it is one line. Its downside, a possible near-zero delay, is acceptable because the expected delay still grows exponentially.
- **Retry in the queue over a timer per retry.** The queue already hides a task until a time and survives restarts; a separate timer would duplicate that and need its own recovery.
- **Park over fail.** The workflow never sees an exhausted activity as an error, which keeps finished work. The cost is that a run can wait indefinitely, which is why dead letters are listed, alerted on (`DurableDeadLetters`), and redrivable.
- **Idempotency in the effect, not the engine.** The engine cannot know whether a dead worker applied its effect; only the receiver can, with the key.

## What a correct implementation must handle

The tests cover the following cases:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example backoff | unit | the worked schedule, the defaults, a non-integer factor, jitter at 0.5 | the formula the server applies |
| jitter bounds and mean | statistical | 20,000 draws in $[0, d)$ with mean $d/2$ within 4 sd | retries spread, as the probability calculation predicts |
| error types | unit | non-retryable and typed errors survive `fmt.Errorf("%w")` | the worker inspects them with `errors.As` |
| retry schedule under a fake clock | unit, fault | retries due at exactly 1, 2, 4 s; the 4th failure dead-letters; describe, `ListDeadLetters`, redrive | `wf dlq list`, the [poison-task drill](../12-incident-response-and-chaos/04-poison-task.md) |
| non-retryable stops | unit | a marked failure and a policy-listed type both end in `ActivityTaskFailed` after one attempt | bad input is reported, not retried |
| worker maps activity errors | unit | an activity returning `NewNonRetryable` reaches history as such, after one attempt | the SDK end to end |
| heartbeat extends the lease and resumes | unit, fault | the deadline moves to now + h; describe shows the heartbeat; attempt 2 sees `ckpt-3`, attempt 2, and the same key | resume from checkpoint in subprocess activities |
| stale lease heartbeat and fail refused | fault | a stale lease cannot heartbeat or fail | a presumed-dead worker cannot steer a retry |
| flaky activity kill loop, exactly once | fault | 20 flaky activities, 8 worker SIGKILLs, server restarts: completed, every key applied once, some delivered twice | the durable engine's exactly-once effects |

## What can go wrong

Retry logic is full of small numeric traps, and each one has a large operational symptom.

The backoff can be off in three ways: the exponent off by one, no cap, or using the backoff of the next attempt instead of the failed one. The first retry then waits too long, or retries wait for days. `Backoff` uses $r^{n-1}$ with a cap, and `retryDelay` passes the failed attempt's number. Using zero policy fields as they are makes a default policy retry in a hot loop; zero or sub-1 fields fall back to `DefaultInitial`, `DefaultBackoff`, and `DefaultInterval`.

> **Warning:** Jitter added on top of the backoff, rather than replacing it, gives delays up to twice the backoff that are still in lockstep at the floor. `Jitter` returns `u * d`, replacing the backoff.

Non-retryable failures get retried when the flag is lost, the type is matched against the message, or the type is dropped; bad input then burns every attempt and lands in the DLQ. `Error` implements `Unwrap`, `FailureType`, and `NonRetryable`, and the server matches `f.GetType()` against the policy list.

Dead letters fail operators when they are not recorded, carry no counts or names, or when a redrive of an unknown task is reported as an outage: nobody can see what is stuck or why. The solution records `ActivityDeadLettered` with attempts and last failure, `ListDeadLetters` adds names, and an unknown redrive is `NOT_FOUND`.

Heartbeats go wrong when details are not kept or not passed on, so a resumed attempt starts from scratch, or when the deadline is reported in the wrong unit, so the SDK misjudges its lease. The queue persists details with the heartbeat, and the response gives the deadline in Unix ms.

An idempotency key that is not stable per activity applies the same effect once per attempt, or collides across workflows. The key is `"<workflow_id>/<activity_id>"`, set by the server and read through `IdempotencyKey`. And failing an attempt without checking its lease lets a stale worker fail the live attempt and trigger a needless retry; `FailActivityTask` and `RecordHeartbeat` call `pendingActivity` first.

## Where this leads

This section completed the protocol of [the task protocol and workers](04-task-protocol-and-workers.md): the worker reports failures and heartbeats with these RPCs, and `Info` carries the key and the details. The only mathematics was the expected value of a uniform draw, $d/2$.

Looking ahead, in [deterministic replay](06-deterministic-replay.md) `ExecuteActivity` futures resolve with `ActivityTaskFailed`, and activity options carry the `RetryPolicy`. [Signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md) builds cancellation on these failure and heartbeat handlers. [Subprocess activities](09-subprocess-activities.md) heartbeats checkpoint paths and maps exit codes to failure types.

Production engines extend these pieces:

- **Typed heartbeat details.** Temporal heartbeat details give resumable activities typed progress, with throttled heartbeats. See Temporal "Activity Heartbeats".
- **Asynchronous completion.** Instead of a worker reporting each attempt, an external system completes the activity later by its task token. See Temporal "Asynchronous Activity Completion".
- **Retry state in visibility.** Instead of retry state living only in the queue, the current attempt, last failure, and next retry time are searchable. See Temporal `PendingActivityInfo`.
- **More timeouts.** Beyond start-to-close and heartbeat leases, schedule-to-close and schedule-to-start timeouts bound the total time including retries, and the time spent waiting in the queue. See Temporal "Activity Timeouts".

### Further reading

- Marc Brooker, *Exponential Backoff and Jitter* (AWS Architecture Blog, 2015).
- Pat Helland, *Idempotence Is Not a Medical Condition* (ACM Queue, 2012).

## Key terms

- **Exponential backoff**: a retry delay that grows by a constant factor per failure, up to a cap.
- **Full jitter**: replacing the backoff $d$ with a uniform draw in $[0, d)$, so retries do not synchronise.
- **Non-retryable failure**: a failure that cannot succeed on retry, marked by the activity or named by the policy; it ends the activity at once.
- **Dead letter**: an activity whose retries are exhausted, parked in the DLQ with `ActivityDeadLettered` until an operator redrives it.
- **Redrive**: returning a dead letter to the queue as attempt 1.
- **Heartbeat details**: small progress data recorded with a heartbeat and handed to the next attempt so it can resume.
- **Idempotency key**: `"<workflow_id>/<activity_id>"`, identical on every attempt, which the effect's receiver deduplicates on.

## Summary

- The delay before attempt $n+1$ is $\min(b_0 \cdot r^{n-1}, b_{\max})$, and the server applies full jitter, a uniform draw in $[0, d)$, so a burst of failures does not retry in lockstep.
- A failure that cannot succeed on retry (bad input) is non-retryable: marked by the activity or named by the policy's `non_retryable` types, it ends the activity at once.
- Exhausted retries park the task in the DLQ and record `ActivityDeadLettered`; the workflow waits for a redrive instead of failing.
- Retry state lives in the queue; only outcomes the workflow or an operator must see go into history.
- The idempotency key `"<workflow_id>/<activity_id>"` is the same on every attempt; with it, a flaky activity under SIGKILLs applies each effect exactly once.

## Review questions

1. Why is the backoff exponent $n - 1$ and not $n$?

   <details><summary>Answer</summary>

   $n$ is the attempt that just failed, starting at 1. The first retry should wait exactly $b_0$, which needs $r^0$. Using $r^n$ would make the first retry wait $2 b_0$ and shift the whole schedule.

   </details>

2. With full jitter, what is the expected delay after the third failure under the worked example's policy?

   <details><summary>Answer</summary>

   $d_3 = \min(1 \cdot 2^2, 10) = 4$ s, and $\mathbb{E}[U \cdot d_3] = d_3 / 2 = 2$ s.

   </details>

3. Why does a dead-lettered activity append `ActivityDeadLettered` but not wake the workflow?

   <details><summary>Answer</summary>

   The event is not a trigger. Exhaustion is an operational problem (a bug, an outage) to be fixed and redriven; waking the workflow would let it fail and discard its finished work. The event exists so history, describe, and operators can see why the run is waiting.

   </details>

4. An activity returns `fmt.Errorf("parse spec: %w", activity.NewNonRetryable("BadSpec", "missing field"))`. Why is it still non-retryable?

   <details><summary>Answer</summary>

   `failureOf` and `IsNonRetryable` use `errors.As`, which walks the `%w` chain to the `*activity.Error` and calls its `NonRetryable()` method. A type assertion on the outer error would miss it.

   </details>

5. Why would putting the attempt number in the idempotency key break exactly-once effects?

   <details><summary>Answer</summary>

   An attempt that applied its effect and died before reporting would be retried with a different key, so the receiver would see a new key and apply the effect again. The key must name the activity, not the attempt.

   </details>
