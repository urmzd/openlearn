# Durable timers and the test clock

## Overview

| | |
|---|---|
| **Module** | `dur.07` · build · Go · Pass 8 · 3 to 4 h |
| **Solution** | [`go/durable/timer/timer.go`](../../../../course/solutions/go/durable/timer/timer.go): the min-heap (`Heap`: `Push`, `Peek`, `Pop`, `Remove`), the timer `Service` (`Schedule`, `Cancel`, `Run`), and the `--test-clock` clock (`OffsetClock`: `Shift`, `Handler`); [`go/durable/server/timers.go`](../../../../course/solutions/go/durable/server/timers.go): the `start_timer` and `cancel_timer` commands and `FireTimer` |
| **Contract** | `StartTimer`, `CancelTimer`, `TimerStarted`, `TimerFired`, `TimerCanceled` in [`durable.proto`](../../../../course/contracts/proto/tl/durable/v1/durable.proto), and `POST /debug/clock` (section 2.4); the Go API is section 4 |
| **Needs** | `dur.06` the SDK whose `workflow.Sleep` issues `StartTimer` |
| **Used by** | `dur.08` integrates timer-backed `workflow.Sleep` with the runtime; `dur.11`, `dur.12`, and `ag.05` use it for evaluation cadence, canary wait, and approval TTL |
| **Optional depth** | Cormen et al., *Introduction to Algorithms*, ch. 6 (heaps); Varghese and Lauck, *Hashed and Hierarchical Timing Wheels* (SOSP 1987) |

## Key Takeaways

- A timer is durable because `TimerStarted{fire_at}` is in history: the server arms it when the event commits and re-arms every pending one on recovery; the in-memory service only fires what it holds.
- Exactly once lives in `FireTimer`: it appends `TimerFired` only for a timer still pending, so a duplicate fire appends nothing.
- The service is a binary min-heap ordered by deadline, ties in schedule order, kept in an array where the children of $i$ are $2i+1$ and $2i+2$.
- `--test-clock` moves time forward through `POST /debug/clock`, and every timer that became due fires once, right away.

## How to read this chapter

Sections 1 to 3 derive durable timers from first principles and trace the heap by hand. Section 4 fixes the interface. Section 5 walks through the heap, the timer service, the test clock, and the server's timer commands, explaining why each choice is the right one. Try to predict each step before reading it.

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

The remaining pops come out b, d, a, c. Through the service on a fake clock, advancing to exactly 3 s fires e, b, d (in that order) and advancing to 8 s fires a, c.

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

The demo workflow `SleepDemo` reads its input as an integer duration in milliseconds, calls `workflow.Sleep`, and returns `workflow.Now(ctx).UnixMilli()` as JSON. Started with input `1000`, it completes with its fired time, which exercises the whole path from the `dur.02` workflow service through the worker and the timer service.

The server's entry point wires it:

```go
ts := timer.New[server.TimerKey](clk)
srv, _ := server.Open(ctx, server.Options{ /* ... */ Timers: ts})
go ts.Run(ctx, srv.FireTimer)
// with --test-clock: clk := timer.NewOffsetClock(wall), clk.OnShift(ts.Wake),
// and clk.Handler() mounted at /debug/clock on the health port
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example heap | unit | section 3's arrays and pop order | the structure, step by step |
| hand example service | unit | the same timers fire e, b, d at exactly 3 s and a, c at 8 s, each once | deadline order, ties in schedule order |
| heap matches a sorted model | property | 20 seeded rounds of pushes, pops, and removals against a sorted model; the invariant after every operation | `Cancel` and reschedule remove from the middle |
| service reschedule and cancel | unit, boundary | rescheduling moves a key; a canceled timer never fires; a past-due one fires at once | recovery re-arms timers |
| debug clock shifts time | unit, fault | POST shifts fire due timers once, later ones wait; negative and malformed bodies are 400; GET reads the offset | the `clock-skew` drill |
| start timer by hand | unit | `TimerStarted{fire_at = now + 30 s}`; `TimerFired` and a workflow task at exactly the deadline | `workflow.Sleep` |
| timer commands validated | boundary | zero duration, no id, a duplicate or pending id, canceling an unknown timer; `TimerCanceled` and never fired | `dur.08` cancellation |
| `FireTimer` is idempotent | unit, fault | duplicate and unknown fires append nothing | exactly once |
| `FireTimer` retries when the append fails | fault | a fire whose append fails re-arms the timer | a full disk does not lose a wakeup |
| restart with pending timers | fault | two restarts: three timers fire once each, in deadline order; fired ones are not re-armed | drill `ops.02` |
| sleep workflow survives a restart | fault | `SleepDemo(30 s)` through the worker, a server restart in the middle, wakes once and completes | the module's demo |

## 5. The solution, step by step

### From first principles

1. **A timer must survive the process, but a timer service in memory cannot.** So the durable part is the fact that a timer exists and when it fires: `TimerStarted{fire_at}` in history. The in-memory service is a cache of pending timers, rebuilt from history on recovery.
2. **`fire_at` is computed once, at commit.** Storing the duration alone would make the deadline depend on when it was read; storing the absolute time makes every re-arm agree.
3. **Firing may happen more than once; recording a fire must not.** Restarts re-arm timers that may already be armed, and the service may race a cancel. Exactly once therefore lives where the history is written: `FireTimer` appends `TimerFired` only if the timer is still pending in run state, under the server lock.
4. **A failed fire must not be lost.** If the append fails, the timer is still pending in history but no longer armed in the service; re-arming it shortly after closes that gap.
5. **The service needs "next deadline" fast and "remove this one" fast.** A binary min-heap gives $O(\log n)$ push and pop with $O(1)$ peek; storing each entry's index gives $O(\log n)$ removal from the middle, which `Cancel` and rescheduling need. A sequence number as tie breaker makes the firing order a function of the inputs.
6. **Tests and drills need to move time.** A clock that is the wall clock plus a non-decreasing offset lets time jump forward without ever going backwards, and a wake callback makes the service notice at once.

### The code, annotated

**The heap** ([`timer.go`](../../../../course/solutions/go/durable/timer/timer.go)).

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

`Push` appends and sifts up; `Pop` is `Remove(items[0])`. In the worked example, pushing `d(3, seq 4)` stops below `b(3, seq 2)` only because of the sequence tie break.

**The service.**

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

**The test clock.**

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

`Handler` serves `/debug/clock`: POST decodes `{"offset_ms": N}` into a pointer field (so a missing field is distinguishable from 0 and refused), shifts, and answers the total; GET answers the total; a negative or malformed body is 400.

**The server commands and `FireTimer`** ([`timers.go`](../../../../course/solutions/go/durable/server/timers.go)).

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

`cancelTimer` requires the timer to be pending and appends `TimerCanceled`; the commit effect calls `Timers.Cancel`, and `apply` deletes the timer from run state, so even a fire already in flight finds nothing.

**Wiring.** The solution's durable entry point ([`entry/go/cmd/durable/main.go`](../../../../course/solutions/entry/go/cmd/durable/main.go)) builds the service with `timer.New` over the server's clock, passes it as `Options.Timers` to `server.Open`, runs `ts.Run(ctx, srv.FireTimer)`, and with `--test-clock` uses an `OffsetClock` for every component, registers `timers.Wake` with `OnShift`, and mounts the handler at `/debug/clock` on the health port. The worker entry point registers a `SleepDemo` workflow that reads milliseconds as JSON, calls `workflow.Sleep`, and returns `workflow.Now(ctx).UnixMilli()`.

Full solution: [`timer.go`](../../../../course/solutions/go/durable/timer/timer.go), [`timers.go`](../../../../course/solutions/go/durable/server/timers.go).

### Why this solution

- **Cost.** Schedule, cancel, and pop are $O(\log n)$; peek is $O(1)$; memory is $O(n)$ for armed timers. Recovery re-arms all pending timers in $O(n \log n)$.
- **Heap over a sorted slice or a timing wheel.** A sorted slice makes insert $O(n)$. A timing wheel gives $O(1)$ insert and fire but needs a tick resolution and more machinery; for thousands of timers a heap is simpler and exact to the millisecond.
- **Exactly once at the history, not the service.** Making the service itself exactly once would require it to be durable and coordinated with history. Putting the check in `FireTimer`, under the same lock as every other run mutation, makes duplicates harmless with no extra state.
- **Hand-written heap over `container/heap`.** The standard package would work; writing it out keeps the index bookkeeping (needed for `Remove`) explicit and generic over the key type.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. ties not broken by schedule order | two timers due together fire in an order that depends on the heap's shape | `less` compares `Seq` on equal deadlines |
| 2. the 1-based parent formula $i/2$ in a 0-based array, or sift-down that looks at one child | the heap invariant breaks; timers fire out of order | `up` uses `(i - 1) / 2`; `down` compares both children |
| 3. a removal from the middle that only sifts down | the moved entry belonged higher up; a later timer fires first | `Remove` sifts down and then up |
| 4. scheduling an armed key twice, or canceling only in the key map | a re-armed timer fires twice; a canceled one still fires | `Schedule` removes an existing entry first; `Cancel` removes from heap and map |
| 5. firing without removing, or `FireTimer` without the pending check | the same timer fires forever, or `TimerFired` is appended twice | `Run` pops before `fire`; `FireTimer` returns when the timer is not pending |
| 6. a timer at exactly its deadline waiting for the next tick | wakeups arrive late by one wait | `Run` fires when `!e.At.After(now)` |
| 7. a clock shift that wakes nobody, or one that may go backwards | `/debug/clock` appears to do nothing; time runs backwards in a drill | `Shift` runs the `OnShift` listeners and refuses negative shifts |
| 8. the deadline in the wrong unit | a 30 s sleep lasts 8 hours | `FireAtUnixMs` is `fireAt.UnixMilli()` |
| 9. two pending timers with one id, or canceling a timer that is not pending | `TimerFired` cannot say which timer it was; a cancel of nothing is recorded | `startTimer` refuses a pending or same-batch id; `cancelTimer` requires a pending timer |
| 10. dropping a timer whose `TimerFired` append failed | the workflow sleeps forever | `FireTimer` re-schedules the key one second later |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.06` | `workflow.Sleep` issues `StartTimer` and waits for `TimerFired` |
| Back | `dur.02` | commit arms, recovery re-arms; `Options.Timers` is this service |
| Forward | `dur.08` | the Runtime seam exposes timer-backed sleep and cancellation |
| Forward (call site in the catalog) | `dur.11`, `dur.12`, `ag.05` | periodic evaluation, the canary wait, approval expiry; they call `workflow.Sleep` through the `dur.08` runtime seam |
| Forward | drills | the `clock-skew` injector posts to `/debug/clock` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one heap per server | timer queues sharded by history shard | each history shard owns its timers and fires them from persisted timer tasks | `temporalio/temporal`: `service/history/queues` |
| `Sleep` | Schedules | cron-like recurring workflows with backfill and overlap policies | Temporal "Schedules" |
| a binary heap | hierarchical timing wheels | $O(1)$ insert and fire for millions of timers | Kafka's `TimingWheel`, Varghese and Lauck (1987) |
| practice `go/04` sorted set | the same idea with a skip list | ordered iteration and range queries | Redis sorted sets |
