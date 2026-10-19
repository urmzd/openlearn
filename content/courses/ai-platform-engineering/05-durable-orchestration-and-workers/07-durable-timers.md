# Durable timers

Since [deterministic replay](06-deterministic-replay.md), `workflow.Sleep(ctx, 24*time.Hour)` issues a `StartTimer` command, but nothing ever fires it. So a workflow that waits sleeps forever: `TrainRun` evaluating every hour, `ModelRelease` waiting through 30 minutes of canary traffic before promoting, `AgentRun` expiring an approval after a day. And a timer kept only in memory would vanish with the process, the same failure every other durable piece of this chapter avoids.

This section makes waiting durable. By the end you can run `<system> wf start SleepDemo --for 30s`, kill and restart the server, and watch the run still wake exactly once at its deadline. The design splits cleanly in two. The durable part is a fact in history, `TimerStarted{fire_at}`; the part that actually wakes things up is an in-memory min-heap that can be rebuilt from history at any time. Exactly-once firing comes not from the heap but from a single check where history is written.

You need the SDK from the previous section, whose `workflow.Sleep` issues `StartTimer`, and the effects-and-recovery machinery of [the workflow service](02-workflow-service.md). The heap itself is the textbook binary heap; Cormen et al. cover it in chapter 6 of *Introduction to Algorithms*. The contract is `StartTimer`, `CancelTimer`, `TimerStarted`, `TimerFired`, and `TimerCanceled` in [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto), plus the `POST /debug/clock` endpoint described below. The full solution is [`go/durable/timer/timer.go`](../../../../course/solutions/go/durable/timer/timer.go) (the min-heap `Heap` with `Push`, `Peek`, `Pop`, `Remove`; the timer `Service` with `Schedule`, `Cancel`, `Run`; and the test clock `OffsetClock` with `Shift` and `Handler`) and [`go/durable/server/timers.go`](../../../../course/solutions/go/durable/server/timers.go) (the `start_timer` and `cancel_timer` commands and `FireTimer`).

> **Learning objectives**
> - Explain where a timer's durability comes from, and why the in-memory timer service need not be durable.
> - Argue that `FireTimer` appends `TimerFired` exactly once even when the service fires twice.
> - Trace pushes and pops through a binary min-heap with sequence-number tie breaks.
> - Implement removal from the middle of a heap, and say why it must sift both ways.
> - Move time forward in tests with a monotonic offset clock.

## Where durability comes from

The `StartTimer` command appends `TimerStarted{timer_id, duration_ms, fire_at_unix_ms}` with $\text{fire\_at} = \text{now} + \text{duration}$, computed once and stored. From then on, the history says the timer is pending until a `TimerFired` or `TimerCanceled` for it appears. The server arms the timer in the timer service when `TimerStarted` commits (an effect, in the workflow service's sense), and on recovery it re-arms every pending timer at its recorded `fire_at`. The service is not durable and does not need to be.

When a timer is due, the service calls `FireTimer(run, timer)`. If the timer is still pending, `FireTimer` appends `TimerFired`, which is a trigger, so the batch also schedules a workflow task, and the workflow's `Sleep` returns on its next task. If the timer is not pending (already fired, canceled, or the run closed), it does nothing.

That check is the whole exactly-once argument. The service may fire twice (re-armed by a restart, or in a race), but only one `TimerFired` can ever be appended. And if the append itself fails (a full disk), the timer is armed again a second later, so it is never lost.

## The min-heap

The service needs to answer one question quickly: which timer fires next? A binary min-heap answers it. Here is the notation:

| Symbol | Meaning |
|---|---|
| $n$ | number of armed timers |
| $A[0..n-1]$ | the heap array |
| $(t_i, q_i)$ | the key of $A[i]$: deadline, then schedule sequence number |
| $\mathrm{parent}(i) = \lfloor (i-1)/2 \rfloor$ | for $i \ge 1$ |
| $\mathrm{children}(i) = 2i+1,\ 2i+2$ | when they exist |

The heap maintains one **invariant**: no entry is smaller than its parent, $(t_{\mathrm{parent}(i)}, q_{\mathrm{parent}(i)}) \le (t_i, q_i)$, comparing deadlines first and sequence numbers on a tie. So $A[0]$ is always the next timer to fire. Three operations, each $O(\log n)$, preserve it:

| Operation | How |
|---|---|
| `Push(e)` | append at $A[n]$, then **sift up**: while smaller than its parent, swap with it |
| `Pop()` | move $A[n-1]$ into $A[0]$, shrink, then **sift down**: while a child is smaller, swap with the smaller child |
| `Remove(e)` | move $A[n-1]$ into $e$'s slot, shrink, then sift down and up from that slot (the moved entry may belong either way) |

Each entry remembers its index, so `Remove`, which `Cancel` and rescheduling need, finds it in $O(1)$. The sequence number makes equal deadlines fire in the order they were scheduled, so the order is a function of the inputs, not of the heap's shape.

## The service

The service wraps the heap with three calls. `Schedule(key, at)` arms `key`; scheduling a key that is already armed **moves** it, because recovery re-arms timers the service may still hold, and there must never be two. `Cancel(key)` disarms it. `Run(ctx, fire)` loops: it pops every entry with $t \le$ now (removing each **before** calling `fire`), calls `fire` for each, then waits for the earliest of the next deadline (through the injected clock), a change (`Schedule`, `Cancel`, and `Wake` close a broadcast channel), or the end of `ctx`. A timer due at exactly now fires now.

## The test clock

Waiting 24 hours in a test is not an option, so the server can run on a clock you control. `{durable} --test-clock` uses an `OffsetClock`: the wall clock plus an offset that only grows. `POST /debug/clock {"offset_ms": N}` on the health port adds $N$ ms and answers the total; `GET` reads it; a negative or malformed body is 400. After a shift, every `OnShift` callback runs. The server registers the timer service's `Wake`, so timers that became due fire immediately instead of at their old wall-clock instant. The `clock-skew` drill injector uses this endpoint.

Time never moves backwards, not even in a test. A backwards jump would let a lease expiry or a timer fire "before" it was armed.

## Working it by hand

Push five timers, with sequence numbers 1 to 5 in this order: a at 5 s, b at 3 s, c at 8 s, d at 3 s, e at 1 s. The table shows the array after each step:

| Step | Array after | What happened |
|---|---|---|
| push a(5) | [a] | |
| push b(3) | [b a] | b at index 1, parent a(5): 3 < 5, swap |
| push c(8) | [b a c] | c at 2, parent b(3): stays |
| push d(3) | [b d c a] | d at 3, parent a(5): swap to 1; parent b(3, seq 2) vs d(3, seq 4): b is smaller, stop |
| push e(1) | [e b c a d] | e at 4, parent d(3): swap to 1; parent b(3): swap to 0 |
| pop | [b d c a] | e out; d moves to 0; children b(3, seq 2) and c(8): b smaller than d(3, seq 4), swap; d at 1, child a(5): stop |

The remaining pops come out b, d, a, c. Through the service on a fake clock, advancing to exactly 3 s fires e, b, d (in that order), and advancing to 8 s fires a, c.

## Designing the API

The timer package holds the heap, the service, and the test clock; the server adds `FireTimer` and two commands:

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

The demo workflow `SleepDemo` reads its input as an integer duration in milliseconds, calls `workflow.Sleep`, and returns `workflow.Now(ctx).UnixMilli()` as JSON. Started with input `1000`, it completes with its fired time, which exercises the whole path from the workflow service through the worker and the timer service.

The server's entry point wires the pieces together:

```go
ts := timer.New[server.TimerKey](clk)
srv, _ := server.Open(ctx, server.Options{ /* ... */ Timers: ts})
go ts.Run(ctx, srv.FireTimer)
// with --test-clock: clk := timer.NewOffsetClock(wall), clk.OnShift(ts.Wake),
// and clk.Handler() mounted at /debug/clock on the health port
```

## Building it step by step

The design follows from six observations. Try to predict each.

1. **A timer must survive the process, but a timer service in memory cannot.** So the durable part is the fact that a timer exists and when it fires: `TimerStarted{fire_at}` in history. The in-memory service is a cache of pending timers, rebuilt from history on recovery.
2. **`fire_at` is computed once, at commit.** Storing the duration alone would make the deadline depend on when it was read; storing the absolute time makes every re-arm agree.
3. **Firing may happen more than once; recording a fire must not.** Restarts re-arm timers that may already be armed, and the service may race a cancel. Exactly once therefore lives where the history is written: `FireTimer` appends `TimerFired` only if the timer is still pending in run state, under the server lock.
4. **A failed fire must not be lost.** If the append fails, the timer is still pending in history but no longer armed in the service; re-arming it shortly after closes that gap.
5. **The service needs "next deadline" fast and "remove this one" fast.** A binary min-heap gives $O(\log n)$ push and pop with $O(1)$ peek; storing each entry's index gives $O(\log n)$ removal from the middle, which `Cancel` and rescheduling need. A sequence number as tie breaker makes the firing order a function of the inputs.
6. **Tests and drills need to move time.** A clock that is the wall clock plus a non-decreasing offset lets time jump forward without ever going backwards, and a wake callback makes the service notice at once.

### The heap

The heap in [`timer.go`](../../../../course/solutions/go/durable/timer/timer.go) keeps each entry's index up to date on every swap:

```go
func (h *Heap[K]) less(i, j int) bool {
	a, b := h.items[i], h.items[j]
	if !a.At.Equal(b.At) {
		return a.At.Before(b.At)
	}
	return a.Seq < b.Seq // equal deadlines: first scheduled fires first
}

func (h *Heap[K]) swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.items[i].index = i // every move updates the entry's own index
	h.items[j].index = j
}

func (h *Heap[K]) up(i int) {
	for i > 0 {
		p := (i - 1) / 2 // 0-based parent; i/2 is the 1-based formula and is wrong here
		if !h.less(i, p) {
			return
		}
		h.swap(i, p)
		i = p
	}
}

func (h *Heap[K]) down(i int) {
	n := len(h.items)
	for {
		l, small := 2*i+1, i
		if l < n && h.less(l, small) {
			small = l
		}
		if r := l + 1; r < n && h.less(r, small) { // compare both children: swap with the smaller
			small = r
		}
		if small == i {
			return
		}
		h.swap(i, small)
		i = small
	}
}

func (h *Heap[K]) Remove(e *Entry[K]) *Entry[K] {
	i, last := e.index, len(h.items)-1
	if i < 0 || i > last || h.items[i] != e {
		return nil // not in this heap (already removed)
	}
	if i != last {
		h.swap(i, last)
	}
	h.items[last] = nil // let the GC reclaim it
	h.items = h.items[:last]
	if i < last {
		// The entry moved into slot i came from the bottom, but from a different
		// subtree: it may be too big for its new place or too small. Try both.
		h.down(i)
		h.up(i)
	}
	e.index = -1
	return e
}
```

`Push` appends and sifts up, and `Pop` is `Remove(items[0])`. In the worked example, pushing `d(3, seq 4)` stops below `b(3, seq 2)` only because of the sequence tie break.

### The service

`Schedule` moves an existing key rather than duplicating it, and `Run` pops due timers before firing them:

```go
func (s *Service[K]) Schedule(key K, at time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seq++
	if e := s.byKey[key]; e != nil {
		s.heap.Remove(e) // re-arming moves the timer: never two entries for one key
	}
	e := &Entry[K]{Key: key, At: at, Seq: s.seq}
	s.byKey[key] = e
	s.heap.Push(e)
	s.notify() // the earliest deadline may have changed: wake Run
}

func (s *Service[K]) Run(ctx context.Context, fire func(K)) error {
	for ctx.Err() == nil {
		s.mu.Lock()
		now := s.clk.Now()
		var due []K
		// "!After(now)": a timer due at exactly now fires now, not one wait later.
		for e := s.heap.Peek(); e != nil && !e.At.After(now); e = s.heap.Peek() {
			s.heap.Pop()           // removed before fire: it can never fire again from here
			delete(s.byKey, e.Key)
			due = append(due, e.Key)
		}
		var timeout <-chan time.Time
		if e := s.heap.Peek(); e != nil && len(due) == 0 {
			timeout = s.clk.After(e.At.Sub(now))
		}
		wake := s.wake
		s.mu.Unlock()
		for _, k := range due {
			fire(k) // outside the lock: FireTimer takes the server lock and may Schedule
		}
		if len(due) > 0 {
			continue // re-read the clock: firing took time
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-wake: // Schedule, Cancel, or a clock shift
		case <-timeout:
		}
	}
	return ctx.Err()
}
```

Calling `fire` outside the service lock matters: `FireTimer` may call `Schedule` to re-arm after a failed append, which takes the service lock again. `Cancel` removes from both the heap and `byKey`, so a canceled timer cannot fire. With no timers, `timeout` is nil and the select waits only for a wake or the end.

### The test clock

`Shift` refuses to go backwards and wakes its listeners outside the lock:

```go
func (c *OffsetClock) Shift(d time.Duration) error {
	if d < 0 {
		return ErrBackwards // a lease or timer must never appear to fire before it was armed
	}
	c.mu.Lock()
	c.offset += d
	ls := append([]func(){}, c.listeners...) // copy, then call outside the lock
	c.mu.Unlock()
	for _, f := range ls {
		f() // Service.Wake: due timers fire now
	}
	return nil
}
```

`Handler` serves `/debug/clock`. POST decodes `{"offset_ms": N}` into a pointer field, so a missing field is distinguishable from 0 and refused; it shifts and answers the total. GET answers the total. A negative or malformed body is 400.

### The server commands and `FireTimer`

On the server, in [`timers.go`](../../../../course/solutions/go/durable/server/timers.go), `startTimer` validates and records the absolute deadline, and `FireTimer` holds the exactly-once check:

```go
func startTimer(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	st := c.GetStartTimer()
	if st.GetTimerId() == "" || st.GetDurationMs() <= 0 {
		return status.Error(codes.InvalidArgument, "start_timer needs a timer_id and a duration_ms > 0")
	}
	// One pending timer per id, in the run and in this batch: TimerFired names its timer by id.
	if b.r.timers[st.GetTimerId()] != nil { /* INVALID_ARGUMENT: already pending */ }
	// ... and not started twice in the same batch
	fireAt := s.o.Clock.Now().Add(time.Duration(st.GetDurationMs()) * time.Millisecond) // computed once, here
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_TimerStarted{TimerStarted: &durablev1.TimerStarted{
		TimerId: st.GetTimerId(), DurationMs: st.GetDurationMs(), FireAtUnixMs: fireAt.UnixMilli(), // milliseconds
		WorkflowTaskCompletedEventId: completed,
	}}})
	return nil // commit's effect arms it in the service
}

func (s *Server) FireTimer(k TimerKey) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[k.RunID]
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		return // closed or unknown run: nothing to wake
	}
	t := r.timers[k.TimerID]
	if t == nil {
		return // already fired or canceled: the exactly-once check
	}
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_TimerFired{TimerFired: &durablev1.TimerFired{
		TimerId: t.id, StartedEventId: t.startedID,
	}}})
	// timer_fired is a trigger: commit also schedules a workflow task.
	if err := s.commit(context.Background(), b); err != nil && s.o.Timers != nil {
		s.o.Timers.Schedule(k, s.o.Clock.Now().Add(time.Second)) // still pending in history: try again
	}
}
```

`cancelTimer` requires the timer to be pending and appends `TimerCanceled`. The commit effect calls `Timers.Cancel`, and `apply` deletes the timer from run state, so even a fire already in flight finds nothing.

### Wiring

The solution's durable entry point, [`entry/go/cmd/durable/main.go`](../../../../course/solutions/entry/go/cmd/durable/main.go), builds the service with `timer.New` over the server's clock, passes it as `Options.Timers` to `server.Open`, and runs `ts.Run(ctx, srv.FireTimer)`. With `--test-clock` it uses an `OffsetClock` for every component, registers `timers.Wake` with `OnShift`, and mounts the handler at `/debug/clock` on the health port. The worker entry point registers the `SleepDemo` workflow, which reads milliseconds as JSON, calls `workflow.Sleep`, and returns `workflow.Now(ctx).UnixMilli()`.

The full solution is in [`timer.go`](../../../../course/solutions/go/durable/timer/timer.go) and [`timers.go`](../../../../course/solutions/go/durable/server/timers.go).

### Why this design

- **Cost.** Schedule, cancel, and pop are $O(\log n)$; peek is $O(1)$; memory is $O(n)$ for armed timers. Recovery re-arms all pending timers in $O(n \log n)$.
- **Heap over a sorted slice or a timing wheel.** A sorted slice makes insert $O(n)$. A timing wheel gives $O(1)$ insert and fire but needs a tick resolution and more machinery; for thousands of timers a heap is simpler and exact to the millisecond.
- **Exactly once at the history, not the service.** Making the service itself exactly once would require it to be durable and coordinated with history. Putting the check in `FireTimer`, under the same lock as every other run mutation, makes duplicates harmless with no extra state.
- **Hand-written heap over `container/heap`.** The standard package would work; writing it out keeps the index bookkeeping (needed for `Remove`) explicit and generic over the key type.

## What a correct implementation must handle

The tests cover the heap, the service, the clock, and the server path:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example heap | unit | the worked arrays and pop order | the structure, step by step |
| hand example service | unit | the same timers fire e, b, d at exactly 3 s and a, c at 8 s, each once | deadline order, ties in schedule order |
| heap matches a sorted model | property | 20 seeded rounds of pushes, pops, and removals against a sorted model; the invariant after every operation | `Cancel` and reschedule remove from the middle |
| service reschedule and cancel | unit, boundary | rescheduling moves a key; a canceled timer never fires; a past-due one fires at once | recovery re-arms timers |
| debug clock shifts time | unit, fault | POST shifts fire due timers once, later ones wait; negative and malformed bodies are 400; GET reads the offset | the `clock-skew` drill |
| start timer by hand | unit | `TimerStarted{fire_at = now + 30 s}`; `TimerFired` and a workflow task at exactly the deadline | `workflow.Sleep` |
| timer commands validated | boundary | zero duration, no id, a duplicate or pending id, canceling an unknown timer; `TimerCanceled` and never fired | cancellation |
| `FireTimer` is idempotent | unit, fault | duplicate and unknown fires append nothing | exactly once |
| `FireTimer` retries when the append fails | fault | a fire whose append fails re-arms the timer | a full disk does not lose a wakeup |
| restart with pending timers | fault | two restarts: three timers fire once each, in deadline order; fired ones are not re-armed | the [SIGKILL-loop drill](../../systems/05-incident-response-and-chaos/02-durable-kill9.md) |
| sleep workflow survives a restart | fault | `SleepDemo(30 s)` through the worker, a server restart in the middle, wakes once and completes | the section's demo |

## What can go wrong

Heap bugs make timers fire out of order; service and server bugs make them fire twice, never, or late.

On the heap side, ties not broken by schedule order make two timers due together fire in an order that depends on the heap's shape; `less` compares `Seq` on equal deadlines. The 1-based parent formula $i/2$ in a 0-based array, or a sift-down that looks at only one child, breaks the heap invariant so timers fire out of order; `up` uses `(i - 1) / 2`, and `down` compares both children.

> **Warning:** A removal from the middle that only sifts down is wrong whenever the moved entry belonged higher up: a later timer then fires first. `Remove` sifts down and then up.

In the service, scheduling an armed key twice makes a re-armed timer fire twice, and canceling only in the key map lets a canceled one still fire; `Schedule` removes an existing entry first, and `Cancel` removes from heap and map. Firing without removing, or a `FireTimer` without the pending check, makes the same timer fire forever or appends `TimerFired` twice; `Run` pops before `fire`, and `FireTimer` returns when the timer is not pending. A timer at exactly its deadline that waits for the next tick wakes late by one wait; `Run` fires when `!e.At.After(now)`.

The test clock fails if a shift wakes nobody, so `/debug/clock` appears to do nothing, or if it may go backwards, so time runs backwards in a drill. `Shift` runs the `OnShift` listeners and refuses negative shifts.

On the server, a deadline in the wrong unit makes a 30 s sleep last 8 hours; `FireAtUnixMs` is `fireAt.UnixMilli()`. Two pending timers with one id leave `TimerFired` unable to say which timer it was, and canceling a timer that is not pending records a cancel of nothing; `startTimer` refuses a pending or same-batch id, and `cancelTimer` requires a pending timer. Finally, dropping a timer whose `TimerFired` append failed leaves the workflow sleeping forever; `FireTimer` re-schedules the key one second later.

## Where this leads

Looking back, `workflow.Sleep` from [deterministic replay](06-deterministic-replay.md) issues `StartTimer` and waits for `TimerFired`, and [the workflow service](02-workflow-service.md) arms timers at commit and re-arms them on recovery; its `Options.Timers` is this service.

Looking forward, [signals, cancellation, and sagas](08-signals-cancellation-and-sagas.md) exposes timer-backed sleep and cancellation through its runtime seam. Through that seam, [TrainRun and EvalSuite](11-trainrun-and-evalsuite.md) runs periodic evaluation, [ModelRelease](12-model-release.md) waits out the canary, and [durable agent runs](../13-agent-sdk/05-durable-agent-runs.md) expire approvals, all by calling `workflow.Sleep`. In the drills, the `clock-skew` injector posts to `/debug/clock`.

Production timer systems scale the same ideas:

- **Sharded timer queues.** Instead of one heap per server, each history shard owns its timers and fires them from persisted timer tasks. See `temporalio/temporal`: `service/history/queues`.
- **Schedules.** Beyond `Sleep`, cron-like recurring workflows with backfill and overlap policies. See Temporal "Schedules".
- **Hierarchical timing wheels.** Instead of a binary heap, $O(1)$ insert and fire for millions of timers. See Kafka's `TimingWheel`, and Varghese and Lauck (1987).
- **Skip lists.** The same ordered-by-deadline idea as a sorted set backed by a skip list gives ordered iteration and range queries. See Redis sorted sets.

### Further reading

- Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms*, chapter 6 (heaps).
- George Varghese and Tony Lauck, *Hashed and Hierarchical Timing Wheels* (SOSP 1987).

## Key terms

- **Durable timer**: a timer whose existence and deadline are recorded in history as `TimerStarted{fire_at}`, so it survives any restart.
- **Timer service**: the in-memory component that fires armed timers; a cache rebuilt from history, not itself durable.
- **Binary min-heap**: an array-backed tree in which no entry is smaller than its parent, so the minimum is at index 0.
- **Sift up / sift down**: restoring the heap invariant by swapping an entry with its parent, or with its smaller child.
- **Tie break**: the schedule sequence number that orders timers with equal deadlines.
- **Offset clock**: the wall clock plus a non-decreasing offset, used to move time forward in tests and drills.

## Summary

- A timer is durable because `TimerStarted{fire_at}` is in history: the server arms it when the event commits and re-arms every pending one on recovery; the in-memory service only fires what it holds.
- Exactly once lives in `FireTimer`: it appends `TimerFired` only for a timer still pending, so a duplicate fire appends nothing, and a failed append re-arms the timer.
- The service is a binary min-heap ordered by deadline, ties in schedule order, kept in an array where the children of $i$ are $2i+1$ and $2i+2$; removal from the middle sifts both ways.
- `--test-clock` moves time forward through `POST /debug/clock`, and every timer that became due fires once, right away; time never moves backwards.

## Review questions

1. The server restarts with a timer pending whose `fire_at` passed during the downtime. What happens?

   <details><summary>Answer</summary>

   Recovery (`resume`) calls `Timers.Schedule` with the recorded `fire_at`. `Run`'s loop sees `!At.After(now)`, pops it, and calls `FireTimer`, which finds it pending and appends `TimerFired`. The workflow wakes immediately rather than never.

   </details>

2. Why does `Remove` call both `down(i)` and `up(i)`?

   <details><summary>Answer</summary>

   The last entry moved into slot `i` comes from an arbitrary subtree. It may be larger than a child of `i` (needs to go down) or smaller than the parent of `i` (needs to go up). At most one of the two moves anything; calling both covers both cases.

   </details>

3. `FireTimer` is called twice for the same key. Why is only one `TimerFired` appended?

   <details><summary>Answer</summary>

   The first call commits `TimerFired`; `apply` deletes the timer from `r.timers`. The second call, under the same lock, finds `r.timers[id] == nil` and returns without appending.

   </details>

4. Why must the test clock refuse a negative shift?

   <details><summary>Answer</summary>

   Leases, retries, and timers are all expressed as absolute times. Moving backwards could make an event appear to happen before the event that caused it (a lease expiring before it was granted). Every component assumes time is non-decreasing, so a backwards jump would test behaviour that cannot happen in production and hide behaviour that can.

   </details>

5. Why is `fire_at` stored rather than recomputed from `duration_ms` on recovery?

   <details><summary>Answer</summary>

   Recomputing as `now + duration` at recovery time would push every pending timer later by the downtime, and differently on each restart. The stored absolute time is the one deadline everyone agrees on.

   </details>

6. Five timers are pushed in order a(5 s), b(3 s), c(8 s), d(3 s), e(1 s). In what order do they fire?

   <details><summary>Answer</summary>

   e, b, d, a, c. The deadline orders them, and b fires before d because both are due at 3 s and b was scheduled first (sequence 2 versus 4).

   </details>
