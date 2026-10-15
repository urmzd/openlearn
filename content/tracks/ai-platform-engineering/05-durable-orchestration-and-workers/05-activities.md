<!-- ol:module dur.05 -->
# Activities: retries, backoff + jitter, timeouts, heartbeats, idempotency keys

## Overview

| | |
|---|---|
| **Module** | `dur.05` · build · Go · Pass 8 · 4 to 5 h |
| **You build** | `go/durable/activity/activity.go`: `IdempotencyKey`, `Attempt`, `HeartbeatDetails`, `RecordHeartbeat`, `NewError`, `NewNonRetryable`, `IsNonRetryable`, `Backoff`, `Jitter`; `go/durable/server/activities.go`: `RecordHeartbeat`, `FailActivityTask`, `ListDeadLetters`, `RedriveDeadLetter` |
| **Contract** | `RetryPolicy`, `ActivityOptions`, `Failure`, and the heartbeat and fail rpcs in [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto); the Go API is section 4 |
| **Tests** | `course/tests/go/dur_05/` (what they check: section 4) |
| **Needs** | `dur.04` the protocol and the worker; reading: [`S-M07a`](../../math/07-probability-statistics/90-problem-set-a.md) (the expected value of a jittered backoff) |
| **Used by** | `dur.06` workflows and `dur.08` cancellation use these activity failure and heartbeat handlers |
| **Milestone** | MS-durable |
| **Optional depth** | Brooker, *Exponential Backoff and Jitter* (AWS Architecture Blog, 2015); Helland, *Idempotence Is Not a Medical Condition* (ACM Queue, 2012) |

## Key Takeaways

- The delay before attempt $n+1$ is $\min(b_0 \cdot r^{n-1}, b_{\max})$ (`TestBackoffHandExample`), and the server applies full jitter, a uniform draw in $[0, d)$, so a burst of failures does not retry in lockstep (`TestJitterBoundsAndMean`).
- A failure that cannot succeed on retry (bad input) is non-retryable: marked by the activity or named by the policy's `non_retryable` types, it ends the activity at once (`TestNonRetryableStops`).
- Exhausted retries park the task in the DLQ and record `ActivityDeadLettered`; the workflow waits for a redrive instead of failing (`TestRetryScheduleUnderFakeClock`).
- The idempotency key `"<workflow_id>/<activity_id>"` is the same on every attempt; with it, a flaky activity under SIGKILLs applies each effect exactly once (`TestFlakyActivityKillLoopExactlyOnce`).

## How to work this chapter

```bash
ol start dur.05
ol tests dur.05
ol check dur.05          # needs dur.04 (or --ref-deps)
ol diff  dur.05
```

`Backoff` and `Jitter` are pure functions: write and test them first, then the server's `FailActivityTask`, then heartbeats.

---

## 1. Why now

With `dur.04` a failing activity is retried at once, forever: a downstream service that is down gets hammered by every worker, a bad input burns every attempt, and an activity that was killed after charging a card charges it again on the retry. This module makes failure a first-class part of the engine: retries with exponential backoff and jitter, a dead-letter queue when they run out, non-retryable failures that stop at once, heartbeats that let a long activity keep its lease and resume from its last checkpoint, and the idempotency key that makes "at least once" safe. `<system> wf start FlakyDemo --fail-rate 0.5` completes, with every effect applied once.

## 2. Principles

### 2.1 Exponential backoff

| Symbol | Meaning | Default |
|---|---|---|
| $n$ | the attempt that just failed, $n \ge 1$ | |
| $b_0$ | `initial_ms` | 1 s |
| $r$ | `backoff`, the growth factor (below 1 counts as unset) | 2.0 |
| $b_{\max}$ | `max_interval_ms`, the cap | 60 s |
| $d_n$ | the backoff before attempt $n+1$ | |
| $U$ | a uniform random number in $[0, 1)$ | |
| $\delta_n$ | the delay actually used | |

$$d_n = \min\left(b_0 \cdot r^{\,n-1},\; b_{\max}\right)$$

The first retry waits $b_0$, each later one $r$ times longer, never more than $b_{\max}$. A zero field means the default: a policy left empty must not retry in a hot loop.

### 2.2 Full jitter

If a dependency fails for 1,000 activities at once, they all retry after exactly $d_1$, fail together again, and so on: synchronized waves. Jitter spreads them:

$$\delta_n = U \cdot d_n, \qquad U \sim \mathrm{Uniform}[0, 1).$$

Then $0 \le \delta_n < d_n$ and $\mathbb{E}[\delta_n] = d_n / 2$ (the computation of `S-M07a`): the expected wait halves, and retries land spread uniformly over the window. The server draws $U$ from its own seeded generator (`Options.Seed`); tests can replace the whole step with `Options.Jitter`.

### 2.3 Retryable, non-retryable, dead-lettered

`FailActivityTask(failure)` on a live lease:

| Condition | History | Queue |
|---|---|---|
| `failure.non_retryable`, or `failure.type` listed in `RetryPolicy.non_retryable` | `ActivityTaskStarted`, `ActivityTaskFailed` (the workflow is woken and sees an error) | `Dropped` |
| the attempt reaches `max_attempts` (0 = `[durable].dlq_after_attempts`, 5) | `ActivityDeadLettered` (informational: the workflow keeps waiting) | `DeadLettered` |
| otherwise | nothing (the retry lives in the queue; describe shows the last failure) | `Retrying` after $\delta_n$ |

Why does an exhausted activity not fail the workflow? Because exhausting retries usually means a bug or an outage, which an operator fixes and then **redrives** (`RedriveDeadLetter`, attempt 1 again); failing the workflow would throw away hours of finished steps. `ListDeadLetters` shows each dead letter with its workflow, activity, attempts, and last failure (`<system> wf dlq list data`).

Activities choose their failure type: `activity.NewNonRetryable("BadSpec", msg)` fails for good; `activity.NewError("Timeout", msg)` is retryable with a type the policy can name. The subprocess runner of `dur.09` maps exit code 65 to type `ExitCode65`, which policies list as non-retryable.

### 2.4 Timeouts and heartbeats

An attempt's lease is its **start-to-close** timeout, or its **heartbeat timeout** $h$ when it has one: a heartbeating activity keeps its lease alive by calling `activity.RecordHeartbeat(ctx, details)` at least every $h$. Each heartbeat moves the deadline to now $+ h$ and stores the details (a checkpoint path, a byte offset). If the worker dies, the next attempt starts with `activity.HeartbeatDetails(ctx)` = the last details: it resumes instead of starting over. A heartbeat with a stale lease is `FAILED_PRECONDITION`, which the worker turns into a canceled context (`dur.04`).

### 2.5 Idempotency keys: effectively once

Delivery is at least once: an attempt can apply its effect and die before reporting, so the next attempt applies it again. The fix is not in the engine but in the effect: give it a key that is the same for every attempt of the activity, `activity.IdempotencyKey(ctx)` = `"<workflow_id>/<activity_id>"`, and make the receiver ignore a key it has seen (a unique constraint, an output directory named by the key, the `effects` sink of the course tests). Attempt number, worker id, or run id in the key would make every attempt a different effect.

## 3. Worked example by hand

Policy: $b_0 = 1$ s, $r = 2$, $b_{\max} = 10$ s, `max_attempts` 4.

| Failed attempt $n$ | $b_0 r^{n-1}$ | $d_n$ | full jitter at $U = 0.5$ |
|---|---|---|---|
| 1 | 1 s | 1 s | 0.5 s |
| 2 | 2 s | 2 s | 1 s |
| 3 | 4 s | 4 s | 2 s |
| 4 | 8 s | 8 s | 4 s |
| 5 | 16 s | 10 s | 5 s |
| 6 | 32 s | 10 s | 5 s |

With jitter switched off (`Options.Jitter` = identity) and the clock at 0: attempt 1 fails at 0 and attempt 2 is pollable at exactly 1 s (not at 999 ms); attempt 2 fails and attempt 3 is pollable 2 s later; attempt 3 fails, 4 s later attempt 4; attempt 4 fails and, $4 \ge$ `max_attempts`, the task is dead-lettered with `ActivityDeadLettered{attempts 4}` and no workflow task. A redrive delivers it as attempt 1. These are `TestBackoffHandExample` and `TestRetryScheduleUnderFakeClock`.

## 4. The interface

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestBackoffHandExample` | unit | section 3's schedule, the defaults, a non-integer factor, jitter at 0.5 | the formula the server applies |
| `TestJitterBoundsAndMean` | statistical | 20,000 draws in $[0, d)$ with mean $d/2$ within 4 sd | retries spread, as `S-M07a` predicts |
| `TestErrorTypes` | unit | non-retryable and typed errors survive `fmt.Errorf("%w")` | the worker inspects them with `errors.As` |
| `TestRetryScheduleUnderFakeClock` | unit, fault | retries due at exactly 1, 2, 4 s; the 4th failure dead-letters; describe, `ListDeadLetters`, redrive | `wf dlq list`, drill `ops.03` |
| `TestNonRetryableStops` | unit | a marked failure and a policy-listed type both end in `ActivityTaskFailed` after one attempt | bad input is reported, not retried |
| `TestWorkerMapsActivityErrors` | unit | an activity returning `NewNonRetryable` reaches history as such, after one attempt | the SDK end to end |
| `TestHeartbeatExtendsLeaseAndResumes` | unit, fault | the deadline moves to now + h; describe shows the heartbeat; attempt 2 sees `ckpt-3`, attempt 2, and the same key | resume from checkpoint (`dur.09`) |
| `TestStaleLeaseHeartbeatAndFailRefused` | fault | a stale lease cannot heartbeat or fail | a presumed-dead worker cannot steer a retry |
| `TestFlakyActivityKillLoopExactlyOnce` | fault | 20 flaky activities, 8 worker SIGKILLs, server restarts: completed, every key applied once, some delivered twice | MS-durable's exactly-once effects |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the exponent off by one, no cap, or the backoff of the next attempt instead of the failed one | the first retry waits too long, or retries wait for days | `TestBackoffHandExample`, `TestRetryScheduleUnderFakeClock` (mutants `s01`, `s02`, `s08`) |
| 2. zero policy fields used as they are | a default policy retries in a hot loop | `TestBackoffHandExample` (mutant `s03`) |
| 3. jitter added on top of the backoff | delays up to twice the backoff, still in lockstep at the floor | `TestJitterBoundsAndMean` (mutant `s04`) |
| 4. non-retryable failures retried (the flag lost, the type matched against the message, the type dropped) | bad input burns every attempt and lands in the DLQ | `TestErrorTypes`, `TestNonRetryableStops`, `TestWorkerMapsActivityErrors` (mutants `s05`, `s09`, `s10`, `s17`) |
| 5. dead letters not recorded, without counts or names, or a redrive of an unknown task reported as an outage | operators cannot see what is stuck or why | `TestRetryScheduleUnderFakeClock` (mutants `s06`, `s07`, `s14`, `s15`) |
| 6. heartbeat details not kept or not passed on; the deadline reported in the wrong unit | a resumed attempt starts from scratch; the SDK misjudges its lease | `TestHeartbeatExtendsLeaseAndResumes` (mutants `s11`, `s13`, `s16`) |
| 7. an idempotency key that is not stable per activity | the same effect is applied once per attempt, or keys collide across workflows | `TestFlakyActivityKillLoopExactlyOnce`, `TestHeartbeatExtendsLeaseAndResumes` (mutant `s12`) |
| 8. failing an attempt without checking its lease | a stale worker fails the live attempt and triggers a needless retry | `TestStaleLeaseHeartbeatAndFailRefused` (mutant `s18`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.04` | the worker reports failures and heartbeats with these rpcs; `Info` carries the key and the details |
| Back | `S-M07a` | the expected value of a uniform draw, $d/2$ |
| Forward | `dur.06` | `ExecuteActivity` futures resolve with `ActivityTaskFailed`; options carry the `RetryPolicy` |
| Forward | `dur.08` | cancellation uses these activity failure and heartbeat handlers |
| Forward | `dur.09` | the subprocess runner heartbeats checkpoint paths and maps exit codes to failure types |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| heartbeat details | Temporal heartbeat details | resumable activities with typed progress, throttled heartbeats | Temporal "Activity Heartbeats" |
| a worker reports each attempt | asynchronous activity completion | an external system completes the activity later by its task token | Temporal "Asynchronous Activity Completion" |
| retry state in the queue | retry state in visibility | the current attempt, last failure, and next retry time are searchable | Temporal `PendingActivityInfo` |
| start-to-close or heartbeat lease | schedule-to-close and schedule-to-start timeouts | bounds on the total time including retries, and on waiting in the queue | Temporal "Activity Timeouts" |
