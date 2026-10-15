<!-- ol:module dur.07 -->
# Durable timers and the test clock

## Overview

| | |
|---|---|
| **Module** | `dur.07` · build · Go · Pass 8 · 3 to 4 h |
| **You build** | `go/durable/timer/timer.go`: the min-heap (`Heap`: `Push`, `Peek`, `Pop`, `Remove`), the timer `Service` (`Schedule`, `Cancel`, `Run`), and the `--test-clock` clock (`OffsetClock`: `Shift`, `Handler`); `go/durable/server/timers.go`: the `start_timer` and `cancel_timer` commands and `FireTimer` |
| **Contract** | `StartTimer`, `CancelTimer`, `TimerStarted`, `TimerFired`, `TimerCanceled` in [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto), and `POST /debug/clock` of DESIGN 2.7; the Go API is section 4 |
| **Tests** | `course/tests/go/dur_07/` (what they check: section 4) |
| **Needs** | `dur.06` the SDK whose `workflow.Sleep` issues `StartTimer` |
| **Used by** | `dur.08` integrates timer-backed `workflow.Sleep` with the runtime; `dur.11`, `dur.12`, and `ag.05` use it for evaluation cadence, canary wait, and approval TTL |
| **Milestone** | MS-durable |
| **Optional depth** | Cormen et al., *Introduction to Algorithms*, ch. 6 (heaps); Varghese and Lauck, *Hashed and Hierarchical Timing Wheels* (SOSP 1987) |

## Key Takeaways

- A timer is durable because `TimerStarted{fire_at}` is in history: the server arms it when the event commits and re-arms every pending one on recovery; the in-memory service only fires what it holds (`TestRestartWithPendingTimers`).
- Exactly once lives in `FireTimer`: it appends `TimerFired` only for a timer still pending, so a duplicate fire appends nothing (`TestFireTimerIsIdempotent`).
- The service is a binary min-heap ordered by deadline, ties in schedule order, kept in an array where the children of $i$ are $2i+1$ and $2i+2$ (`TestHeapHandExample`).
- `--test-clock` moves time forward through `POST /debug/clock`, and every timer that became due fires once, right away (`TestDebugClockShiftsTime`).

## How to work this chapter

```bash
ol start dur.07
ol tests dur.07
ol check dur.07          # needs dur.06 (or --ref-deps)
ol diff  dur.07
```

Write the heap and check it against section 3's trace, then the service, then the server commands.

---

## 1. Why now

`workflow.Sleep(ctx, 24*time.Hour)` already issues a `StartTimer` command (`dur.06`), but nothing ever fires it, so a workflow that waits (`TrainRun` evaluating every hour, `ModelRelease` waiting 30 minutes of canary traffic before promoting, `AgentRun` expiring an approval after a day) sleeps forever. And a timer kept only in memory would vanish with the process, the same failure every other durable piece avoids. This module makes waiting durable: `<system> wf start SleepDemo --for 30s`, kill and restart the server, and the run still wakes once at its deadline.

## 2. Principles

### 2.1 Where durability comes from

The `StartTimer` command appends `TimerStarted{timer_id, duration_ms, fire_at_unix_ms}` with $\text{fire\_at} = \text{now} + \text{duration}$, computed once and stored. From then on the history says the timer is pending until a `TimerFired` or `TimerCanceled` for it appears. The server arms the timer in the timer service when `TimerStarted` commits (the effect, `dur.02`), and on recovery it re-arms every pending timer at its recorded `fire_at`. The service is not durable and does not need to be.

When a timer is due the service calls `FireTimer(run, timer)`: if the timer is still pending, append `TimerFired` (a trigger: the batch also schedules a workflow task) and the workflow's `Sleep` returns on its next task. If not pending (already fired, canceled, the run closed), do nothing. That check is the whole exactly-once argument: the service may fire twice (re-armed by a restart, a race), but only one `TimerFired` can ever be appended. If the append itself fails (disk full), the timer is armed again a second later, so it is never lost.

### 2.2 The min-heap

| Symbol | Meaning |
|---|---|
| $n$ | number of armed timers |
| $A[0..n-1]$ | the heap array |
| $(t_i, q_i)$ | the key of $A[i]$: deadline, then schedule sequence number |
| $\mathrm{parent}(i) = \lfloor (i-1)/2 \rfloor$ | for $i \ge 1$ |
| $\mathrm{children}(i) = 2i+1,\ 2i+2$ | when they exist |

**Invariant:** no entry is smaller than its parent: $(t_{\mathrm{parent}(i)}, q_{\mathrm{parent}(i)}) \le (t_i, q_i)$, comparing deadlines first and sequence numbers on a tie. So $A[0]$ is the next timer to fire. Operations, each $O(\log n)$:

| Operation | How |
|---|---|
| `Push(e)` | append at $A[n]$, then **sift up**: while smaller than its parent, swap with it |
| `Pop()` | move $A[n-1]$ into $A[0]$, shrink, then **sift down**: while a child is smaller, swap with the smaller child |
| `Remove(e)` | move $A[n-1]$ into $e$'s slot, shrink, then sift down and up from that slot (the moved entry may belong either way) |

Each entry remembers its index, so `Remove` (needed by `Cancel` and by rescheduling) finds it in $O(1)$. The sequence number makes equal deadlines fire in the order they were scheduled, so the order is a function of the inputs, not of the heap's shape.

### 2.3 The service

`Schedule(key, at)` arms `key`; scheduling a key already armed **moves** it (recovery re-arms timers the service may still hold; there must never be two). `Cancel(key)` disarms it. `Run(ctx, fire)` loops: pop every entry with $t \le$ now (removing it **before** calling `fire`), call `fire` for each, then wait for the earliest of the next deadline (through the injected clock), a change (`Schedule`, `Cancel`, `Wake` close a broadcast channel), or the end of `ctx`. A timer due at exactly now fires now.

### 2.4 The test clock

`{durable} --test-clock` runs the server on an `OffsetClock`: the wall clock plus an offset that only grows. `POST /debug/clock {"offset_ms": N}` on the health port adds $N$ ms and answers the total; `GET` reads it; a negative or malformed body is 400. After a shift every `OnShift` callback runs; the server registers the timer service's `Wake`, so timers that became due fire immediately instead of at their old wall-clock instant. The `clock-skew` drill injector uses this endpoint. Time never moves backwards, not even in a test: a backwards jump would let a lease expiry or a timer fire "before" it was armed.

## 3. Worked example by hand

Push five timers, sequence numbers 1 to 5 in this order: a at 5 s, b at 3 s, c at 8 s, d at 3 s, e at 1 s.

| Step | Array after | What happened |
|---|---|---|
| push a(5) | [a] | |
| push b(3) | [b a] | b at index 1, parent a(5): 3 < 5, swap |
| push c(8) | [b a c] | c at 2, parent b(3): stays |
| push d(3) | [b d c a] | d at 3, parent a(5): swap to 1; parent b(3, seq 2) vs d(3, seq 4): b is smaller, stop |
| push e(1) | [e b c a d] | e at 4, parent d(3): swap to 1; parent b(3): swap to 0 |
| pop | [b d c a] | e out; d moves to 0; children b(3, seq 2) and c(8): b smaller than d(3, seq 4), swap; d at 1, child a(5): stop |

The remaining pops come out b, d, a, c. Through the service on a fake clock, advancing to exactly 3 s fires e, b, d (in that order) and advancing to 8 s fires a, c. These are `TestHeapHandExample` and `TestServiceHandExample`.

## 4. The interface

```go
package timer // import "tinyllm/durable/timer"

type Clock interface { Now() time.Time; After(d time.Duration) <-chan time.Time }

type Entry[K comparable] struct { Key K; At time.Time; Seq uint64 /* + its index */ }
type Heap[K comparable] struct{ /* the array */ }
func (h *Heap[K]) Len() int
func (h *Heap[K]) Items() []K               // keys in array order
func (h *Heap[K]) Push(e *Entry[K])
func (h *Heap[K]) Peek() *Entry[K]          // nil when empty
func (h *Heap[K]) Pop() *Entry[K]           // nil when empty
func (h *Heap[K]) Remove(e *Entry[K]) *Entry[K]

type Service[K comparable] struct{ /* heap, keys, clock */ }
func New[K comparable](clk Clock) *Service[K]
func (s *Service[K]) Schedule(key K, at time.Time)
func (s *Service[K]) Cancel(key K)
func (s *Service[K]) Len() int
func (s *Service[K]) Wake()
func (s *Service[K]) Run(ctx context.Context, fire func(K)) error

type OffsetClock struct{ /* base + offset */ }
var ErrBackwards error
func NewOffsetClock(base Clock) *OffsetClock
func (c *OffsetClock) Now() time.Time
func (c *OffsetClock) After(d time.Duration) <-chan time.Time
func (c *OffsetClock) Shift(d time.Duration) error
func (c *OffsetClock) OnShift(f func())
func (c *OffsetClock) Offset() time.Duration
func (c *OffsetClock) Handler() http.Handler // /debug/clock

// go/durable/server/timers.go
func (s *Server) FireTimer(k TimerKey)
// registers "start_timer" and "cancel_timer" with the dur.04 command table
```

For the milestone's `SleepDemo`, register a workflow that reads its input as
an integer duration in milliseconds, calls `workflow.Sleep`, and returns
`workflow.Now(ctx).UnixMilli()` as JSON. The milestone passes `1000` as input
and checks that the run completes with a fired time. This exercises the whole
path from the `dur.02` workflow service through the worker and timer service.

Wiring in your server's main: `ts := timer.New[server.TimerKey](clk)`, pass `Timers: ts` to `server.Open`, then `go ts.Run(ctx, srv.FireTimer)`; with `--test-clock`, `clk := timer.NewOffsetClock(wall)`, `clk.OnShift(ts.Wake)`, and mount `clk.Handler()` at `/debug/clock` on the health port.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHeapHandExample` | unit | section 3's arrays and pop order | the structure, step by step |
| `TestServiceHandExample` | unit | the same timers fire e, b, d at exactly 3 s and a, c at 8 s, each once | deadline order, ties in schedule order |
| `TestHeapMatchesSortedModel` | property | 20 seeded rounds of pushes, pops, and removals against a sorted model; the invariant after every operation | `Cancel` and reschedule remove from the middle |
| `TestServiceRescheduleAndCancel` | unit, boundary | rescheduling moves a key; a canceled timer never fires; a past-due one fires at once | recovery re-arms timers |
| `TestDebugClockShiftsTime` | unit, fault | POST shifts fire due timers once, later ones wait; negative and malformed bodies are 400; GET reads the offset | the `clock-skew` drill |
| `TestStartTimerHand` | unit | `TimerStarted{fire_at = now + 30 s}`; `TimerFired` and a workflow task at exactly the deadline | `workflow.Sleep` |
| `TestTimerCommandsValidated` | boundary | zero duration, no id, a duplicate or pending id, canceling an unknown timer; `TimerCanceled` and never fired | `dur.08` cancellation |
| `TestFireTimerIsIdempotent` | unit, fault | duplicate and unknown fires append nothing | exactly once |
| `TestFireTimerRetriesWhenAppendFails` | fault | a fire whose append fails re-arms the timer | a full disk does not lose a wakeup |
| `TestRestartWithPendingTimers` | fault | two restarts: three timers fire once each, in deadline order; fired ones are not re-armed | drill `ops.02` |
| `TestSleepWorkflowSurvivesRestart` | fault | `SleepDemo(30 s)` through your worker, a server restart in the middle, wakes once and completes | the module's demo |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. ties not broken by schedule order | two timers due together fire in an order that depends on the heap's shape | `TestHeapHandExample`, `TestServiceHandExample` (mutant `s01`) |
| 2. the 1-based parent formula $i/2$ in a 0-based array, or sift-down that looks at one child | the heap invariant breaks; timers fire out of order | `TestHeapHandExample`, `TestHeapMatchesSortedModel` (mutants `s02`, `s03`) |
| 3. a removal from the middle that only sifts down | the moved entry belonged higher up; a later timer fires first | `TestHeapMatchesSortedModel` (mutant `s04`) |
| 4. scheduling an armed key twice, or canceling only in the key map | a re-armed timer fires twice; a canceled one still fires | `TestServiceRescheduleAndCancel` (mutants `s05`, `s06`) |
| 5. firing without removing, or `FireTimer` without the pending check | the same timer fires forever, or `TimerFired` is appended twice | `TestServiceHandExample`, `TestFireTimerIsIdempotent` (mutants `s07`, `s14`) |
| 6. a timer at exactly its deadline waiting for the next tick | wakeups arrive late by one wait | `TestServiceHandExample` (mutant `s08`) |
| 7. a clock shift that wakes nobody, or one that may go backwards | `/debug/clock` appears to do nothing; time runs backwards in a drill | `TestDebugClockShiftsTime` (mutants `s09`, `s10`) |
| 8. the deadline in the wrong unit | a 30 s sleep lasts 8 hours | `TestStartTimerHand` (mutant `s11`) |
| 9. two pending timers with one id, or canceling a timer that is not pending | `TimerFired` cannot say which timer it was; a cancel of nothing is recorded | `TestTimerCommandsValidated` (mutants `s12`, `s13`) |
| 10. dropping a timer whose `TimerFired` append failed | the workflow sleeps forever | `TestFireTimerRetriesWhenAppendFails` (mutant `s15`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.06` | `workflow.Sleep` issues `StartTimer` and waits for `TimerFired` |
| Back | `dur.02` | commit arms, recovery re-arms; `Options.Timers` is this service |
| Forward | `dur.08` | the Runtime seam exposes timer-backed sleep and cancellation |
| Forward (call site in the catalog) | `dur.11`, `dur.12`, `ag.05` | periodic evaluation, the canary wait, approval expiry; they call `workflow.Sleep` through the `dur.08` runtime seam |
| Forward | drills | the `clock-skew` injector posts to `/debug/clock` |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one heap per server | timer queues sharded by history shard | each history shard owns its timers and fires them from persisted timer tasks | `temporalio/temporal`: `service/history/queues` |
| `Sleep` | Schedules | cron-like recurring workflows with backfill and overlap policies | Temporal "Schedules" |
| a binary heap | hierarchical timing wheels | $O(1)$ insert and fire for millions of timers | Kafka's `TimingWheel`, Varghese and Lauck (1987) |
| practice `go/04` sorted set | the same idea with a skip list | ordered iteration and range queries | Redis sorted sets |
