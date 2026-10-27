# Resilience tests

The durable engine of *AI Engineering* makes three promises that no unit test has ever checked. An acknowledged event survives a crash ([The append-only event log](../../systems/07-durable-orchestration/01-event-log.md)). A task has one owner at a time ([Task queue: visibility timeout, fenced leases](../../systems/07-durable-orchestration/03-task-queue.md)). And an activity's effect happens once however often it is retried ([Activities: retries, backoff, idempotency keys](../../systems/07-durable-orchestration/05-activities.md) and [Subprocess activity runner](../../systems/07-durable-orchestration/09-subprocess-activities/README.md)). Unit tests run the happy path in one process that never dies, so a log that forgets to fsync, a worker that forgets to renew its lease, or an activity keyed by its attempt number passes all of them. These bugs appear only when a machine dies at the wrong instant, which in production is "eventually, at 3 a.m., during the biggest training run."

The [durable orchestration chapter](../../systems/07-durable-orchestration/README.md) ends by killing the real server and worker 500 times. This section shows the small, deterministic version of that test first, on a kata small enough to hold in your head, judged by whether it catches the faults that matter. It states the promises a durable queue makes, replays one crash by hand, then walks through the queue's write path, replay, leases, and worker, and finally the tests that kill it. Before each test, predict which planted fault it is meant to catch.

You should know how a test suite is judged by planted faults ([TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md)), properties ([Property-based tests](02-property-based-tests.md)), and reading survivors ([Mutation testing in depth](05-mutation-testing-in-depth.md)), plus Go's `testing` package from [Go: packages, interfaces, goroutines](../../programming-and-algorithms/01-language-primers/06-go.md). The kata models the engine pieces linked above: the event log, fenced leases, and idempotent activities.

The solution is [`taskq.go`](../../../../course/solutions/primers/craft.21/taskq.go), a kata that packs a durable task queue and its worker (the log, leases, and idempotent effects of the durable engine) into one file (`Open`, `Queue.write`, `Lease`, `Renew`, `Complete`, `Worker.RunOne`), and [`resilience_test.go`](../../../../course/solutions/primers/craft.21/resilience_test.go), the **resilience tests** in package `craft21_test`, including the seeded kill loop `TestKillLoopExactlyOnce`. There is no separate contract: the kata's API is in its file, and the tests use a small fault kit, package `craft21/faults` (`CrashFS`, `Clock`, `Effects`). The tests run with `go test ./...` in a module (`craft21`) that holds the kata, the tests, and the fault kit.

> **Learning objectives**
> - State durability, exclusivity, exactly-once effect, and idempotent submission as checkable invariants.
> - Explain why a SIGKILL-only test cannot catch a missing fsync, and model a power loss instead.
> - Use leases with renewal and fencing tokens to keep one owner per task, and test them with a manual clock.
> - Write a seeded kill loop that explores crash points deterministically and checks every promise at the end.

## Promises, not return values

A **resilience test** kills the system at its worst moment and checks a **promise**, not a return value. It is a loop: put the system in a state, inject a fault, let it recover, check an invariant. The invariants are the system's promises, stated so a test can check them after the fact:

| Promise | Checked as | The kata's mechanism |
|---|---|---|
| **durability**: a call that returned nil is on disk | after `Crash()` and `Open`, every acknowledged enqueue, lease, and completion replays | append one JSON line, `Sync`, only then update memory |
| **exclusivity**: one owner per task at a time | while a lease is live, `Lease` gives `ErrNoTask`; a stale token's `Renew` and `Complete` get `ErrLeaseLost` | leases with a deadline and a **fencing token** that grows on every delivery |
| **exactly-once effect** | the external system's applied keys equal the task ids, with deliveries counted separately | the effect is keyed by the task id, the same on every attempt |
| **idempotent submission** | enqueueing an id twice is one task | `Enqueue` of an existing id is a no-op |

Delivery is **at least once**: a worker can die after applying an effect and before completing, and the retry applies it again. That is unavoidable (the Two Generals problem), so the effect must be **idempotent**: the external system deduplicates by key. Exactly once is therefore "at least once delivery" plus an idempotency key that is the same on every attempt. The fault kit's `Effects` deduplicates exactly that way, and counts deliveries, so a test sees both the state (`Keys`) and the retries (`Deliveries`).

## Crashes: what survives and what does not

Not every crash loses the same things, and the difference decides what a test must simulate:

| Fault | What is lost | What a test needs |
|---|---|---|
| SIGKILL of the process | memory; nothing a `write` returned (the page cache belongs to the kernel) | a child process, or abandoning the goroutine |
| power loss, kernel panic | memory **and** every byte written but not synced | a disk model: `CrashFS.Crash()` |
| a crash during a write | the end of that write: a **torn** record | `CrashFS.CrashTorn(n)` keeps `n` bytes of the unsynced tail |
| a full disk | nothing yet; the write fails | `CrashFS.FailWrites(err)` |

A SIGKILL loses nothing a `write` returned, so to catch a missing **fsync** the test must lose the page cache, which is what `CrashFS.Crash()` does by keeping only synced bytes.

Two rules follow. First, **sync before acknowledging**, and update memory only after the sync succeeded: a queue that changes memory first serves a completion that vanishes on restart. Second, **replay must survive a torn tail**: the last line may be half a record. Its call never returned, so it is safe to ignore; but the next append lands right after it, so `Open` must **seal** the fragment with a newline, or the first record acknowledged after the restart is glued to garbage and lost on the next replay.

## Leases, renewals, and fencing tokens

A **lease** is ownership with a deadline: worker `w` may work on task `t` until `now + TTL`. If `w` dies, the task comes back after the deadline. Two things go wrong with that simple scheme.

A task that takes longer than the TTL loses its lease mid-run and a second worker starts it too. The worker must therefore **renew** (heartbeat) while it works, every `TTL/3` in the kata, so two missed renewals still leave time.

A worker that was paused (by GC, a stopped VM, or a partition) wakes up after its lease expired, when someone else owns the task. It must not finish. Every lease carries a **fencing token**, a number that grows with every delivery, and `Renew` and `Complete` with a stale token return `ErrLeaseLost`. When a renewal fails, the worker cancels its work and does not apply the effect.

## Deterministic fault injection

Time is an input. The kata takes its disk (`FS`) and its time (`Clock`) as interfaces, so a test owns both. With the fault kit's `Clock`, time moves only on `Advance`, and `BlockUntil(n)` waits until the code under test is actually waiting, so a test never sleeps and never races. A test can hold a lease for 90 s, pause a worker past its lease, and expire it, in microseconds and the same way every run. With `CrashFS`, a "crash" is one call at an exact point.

Because the test is the same on every run, a failure is a bug report instead of a flake. A **kill loop** then explores many crash points with a seeded generator. Each round picks where to die (before the effect, after it, after `Complete`), crashes the disk, reopens the queue, expires the leases, and continues until no task is pending. At the end, every promise must hold.

## How a resilience suite is judged

The tests run against a correct kata (they must pass, twice, so they are not flaky), then against 12 copies of it, each with one planted fault. A fault the tests make fail is **killed**. A good suite scores at least 0.90 and kills every **required** fault. Three of them are the resilience faults:

| Fault | What it breaks | A test that catches it |
|---|---|---|
| idempotency key ignored | the effect's key changes per attempt, so a retry applies it twice | crash after the effect, retry, `AssertExactlyOnce` |
| lease not renewed | a long task is handed to a second worker mid-run | hold a task for 3 x TTL, try to lease it from another worker |
| fsync skipped | acknowledged records live only in the page cache | `Crash()`, reopen, check acknowledged state |

The other nine are the pitfalls listed in [What can go wrong](#what-can-go-wrong).

## Working one crash by hand

Start with the clock at $t_0$ = 1760000000 s (Unix), a TTL of 30 s, and an empty disk, and follow the log through one power loss:

| Step | Action | Durable log after it |
|---|---|---|
| 1 | `Enqueue("t1","a")`, `Enqueue("t2","b")` | two lines: `{"op":"enq","id":"t1","payload":"a"}`, `{"op":"enq","id":"t2","payload":"b"}` |
| 2 | `w1` leases `t1`: token 1, attempt 1, deadline $t_0$ + 30 s | `{"op":"lease","id":"t1","worker":"w1","token":1,"deadline":1760000030000000000}` |
| 3 | `w1` applies the effect `t1 = done:a` (deliveries of `t1`: 1) | unchanged: the effect is outside the log |
| 4 | **power loss** before `Complete` | unchanged: everything above was synced |
| 5 | restart: `Open` replays three records; `t1` is leased to `w1` until $t_0$ + 30 s | |
| 6 | clock moves to $t_0$ + 31 s; `w2` leases `t1`: token 2, attempt 2 | `lease t1 w2 token 2` |
| 7 | `w2` applies `t1` again: the external system has the key, so it keeps `done:a` (deliveries 2, applied keys `{t1}`) | |
| 8 | `w2` completes `t1` with token 2, then runs `t2` | `done t1`, `lease t2`, `done t2` |
| 9 | `w1` wakes up and calls `Complete` with token 1 | `ErrLeaseLost`: fenced |

Every promise holds: applied keys are exactly `{t1, t2}`, `t1` was delivered twice and applied once, and the zombie's completion was refused. Change the idempotency key to `t1#<token>`, and step 7 applies a second key `t1#2`, so `AssertExactlyOnce` fails. That is the first resilience fault, and the kill loop shown later is this table run 200 times with seeded crash points.

## The kata and the fault kit

`primers/craft.21/resilience_test.go` is in package `craft21_test`, so it is black box: it uses only the kata's exported API and `craft21/faults`. It crashes the disk, moves time past a lease, restarts the queue with `craft21.Open`, and asserts on effects. It covers each promise with at least one test and has one kill loop. The APIs it works with:

```go
// primers/craft.21/taskq.go (the kata)
func Open(fs FS, clock Clock) (*Queue, error)
func (q *Queue) Enqueue(id, payload string) error
func (q *Queue) Lease(worker string, ttl time.Duration) (Lease, error)   // ErrNoTask
func (q *Queue) Renew(l Lease, ttl time.Duration) (Lease, error)         // ErrLeaseLost
func (q *Queue) Complete(l Lease, result string) error                   // ErrLeaseLost
func (q *Queue) Result(id string) (string, bool)
func (q *Queue) Pending() []string
type Worker struct { Q *Queue; ID string; TTL time.Duration; Clock Clock; Work func(context.Context, string) (string, error); Apply Effect }
func (w *Worker) RunOne(ctx context.Context) (bool, error)

// craft21/faults (the fault kit)
fs := faults.NewCrashFS()   // Crash(), CrashTorn(n), FailWrites(err), Durable(name)
clk := faults.NewClock(t0)  // Now, After, Advance(d), BlockUntil(n)
fx := faults.NewEffects()   // Apply(key, v), Deliveries(key), Keys(), FailNext(key, n), AssertExactlyOnce(t, keys)
```

The kata itself must pass the following cases; the resilience tests check the same promises in their own words.

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the hand-worked crash, line by line: crash before `Complete`, redelivery, a fenced zombie | fault | the whole contract in one scenario |
| a duplicate enqueue is one task, even after it is done | unit | retried submissions from the gateway must not rerun work |
| acknowledged enqueues and completions replay after `Crash()` | fault | durability: the fsync rule |
| a torn record is ignored and sealed; the next record survives the next crash | fault | a crash mid-write must not brick or corrupt the log |
| a failed write leaves memory as it was | fault | memory never runs ahead of the disk |
| no second lease until the deadline, then a higher token and attempt 2 | boundary | exclusivity, and the exact expiry instant |
| a zombie's `Renew` and `Complete` are `ErrLeaseLost` | fault | fencing |
| a 90 s task keeps its 30 s lease by renewing | fault | long activities are not stolen |
| a failed renewal cancels the work and skips the effect | fault | a worker that lost ownership does nothing visible |
| crash between effect and completion: delivered twice, applied once | fault | exactly-once effects |
| the tests pass on a correct kata twice in a row, and fail on at least 90% of 12 planted faults, every resilience fault included | fault | a resilience test that passes on a log that loses writes protects nothing |

## Building the queue step by step

Seven facts force the design.

1. **"Acknowledged" must mean "on disk".** A caller that got `nil` will act on it, so the record must be synced before the call returns. And memory must change only after the sync succeeded, or a failed write leaves memory claiming something the disk does not have. This forces one function, `write`, that marshals, appends, syncs, and only then calls `apply`.
2. **State is a fold over the log.** If every change is a record and `apply` is the only thing that changes state, then `Open` is just "apply every record," and replay after a crash gives exactly the acknowledged state. This forces `apply` to be the same code for live writes and replay.
3. **A crash can cut the last write.** Its call never returned, so dropping it is correct. But the file position is after the fragment, so the next append would be glued to it; `Open` must write a newline to seal it. Skipping undecodable lines in the middle (sealed fragments from earlier crashes) keeps later records.
4. **Ownership needs a deadline and a version.** The deadline returns a dead worker's task; the version (fencing token) refuses a worker that only *looks* dead. One global counter, `q.token`, incremented per delivery and recovered from the log by `apply`, gives a strictly increasing token across restarts.
5. **A long task must prove it is alive.** Renewing every `TTL/3` means two missed renewals still leave a third of the TTL. A failed renewal is the signal to stop: cancel the work's context and return without applying.
6. **Delivery is at least once; the effect must not be.** The only key that is the same on every attempt is the task id; any attempt-specific key (token, attempt, worker) turns retries into duplicates.
7. **Tests must own time and the disk.** A `Clock` interface and an `FS` interface make every fault one deterministic call; nothing in the tests sleeps except a safety timeout.

### The write path and replay

```go
// write appends r to the log and syncs it; only then is r applied.
func (q *Queue) write(r record) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if _, err := q.f.Write(append(b, '\n')); err != nil {   // one record = one line
		return err
	}
	if err := q.f.Sync(); err != nil {                        // the durability point
		return err
	}
	q.apply(r)                                                // memory follows the disk, never leads
	return nil
}
```

Every public mutation (`Enqueue`, `Lease`, `Renew`, `Complete`) goes through `write` under `q.mu`, so there is one place where the fsync rule and the "memory after disk" rule hold. Skipping `Sync` is the third resilience fault: nothing visible changes until `Crash()`. Calling `apply` before `Write` is pitfall 5.

Replay reads the log line by line:

```go
func Open(fs FS, clock Clock) (*Queue, error) {
	q := &Queue{clock: clock, tasks: map[string]*task{}}
	b, err := fs.ReadFile(LogName)
	if err != nil && !errors.Is(err, os.ErrNotExist) {   // a missing log is an empty queue
		return nil, err
	}
	torn := false
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		if i < 0 {
			torn = true // a write cut by the crash: its call never returned
			break
		}
		var r record
		if err := json.Unmarshal(b[:i], &r); err == nil {
			q.apply(r)
		} // else: a torn write sealed by an earlier Open; skip it, keep going
		b = b[i+1:]
	}
	if q.f, err = fs.Append(LogName); err != nil {
		return nil, err
	}
	if torn {
		// Seal the fragment with a newline, or the next record would be
		// glued to it and lost on the next replay.
		if _, err := q.f.Write([]byte("\n")); err != nil {
			return nil, err
		}
		if err := q.f.Sync(); err != nil {
			return nil, err
		}
	}
	return q, nil
}
```

There are two different "bad line" cases, and the code treats them differently on purpose. A final fragment with no newline (this crash's torn write) is dropped and sealed; a complete line that does not decode (an earlier crash's fragment, already sealed) is skipped. Refusing to open on either would turn every power cut into an outage.

`apply` is the fold that both paths share:

```go
func (q *Queue) apply(r record) {
	t := q.tasks[r.ID]
	switch r.Op {
	case "enq":
		if t == nil {                     // idempotent submission, also on replay
			q.tasks[r.ID] = &task{payload: r.Payload}
			q.order = append(q.order, r.ID)
		}
	case "lease", "renew":
		if t != nil {
			if r.Op == "lease" {
				t.attempts++              // a renewal is not a new delivery
			}
			t.worker, t.token, t.deadline = r.Worker, r.Token, time.Unix(0, r.Deadline)
		}
	case "done":
		if t != nil {
			t.done, t.result = true, r.Result
		}
	}
	if r.Token > q.token {
		q.token = r.Token                 // the token counter survives restarts
	}
}
```

Deadlines are stored as absolute Unix nanoseconds, not "TTL remaining," so a restart does not hand a dead worker a fresh 30 s.

### Leases, fencing, and the worker

```go
func (q *Queue) Lease(worker string, ttl time.Duration) (Lease, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	now := q.clock.Now()
	for _, id := range q.order {
		t := q.tasks[id]
		if t.done || (t.token != 0 && now.Before(t.deadline)) {   // live until the deadline, exclusive at it
			continue
		}
		r := record{Op: "lease", ID: id, Worker: worker, Token: q.token + 1, Deadline: now.Add(ttl).UnixNano()}
		if err := q.write(r); err != nil {
			return Lease{}, err
		}
		// After write, apply has set t.token, t.deadline, t.attempts from r.
		return Lease{Task: id, Payload: t.payload, Worker: worker, Token: t.token, Deadline: t.deadline, Attempt: t.attempts}, nil
	}
	return Lease{}, ErrNoTask
}

func (q *Queue) Renew(l Lease, ttl time.Duration) (Lease, error) {
	// ... lock, look up t (ErrUnknown if missing)
	now := q.clock.Now()
	if t.done || t.token != l.Token || !now.Before(t.deadline) {   // fenced, finished, or already expired
		return Lease{}, ErrLeaseLost
	}
	// ... write a "renew" record with the same token and now + ttl
}

func (q *Queue) Complete(l Lease, result string) error {
	// ... lock, look up t
	if t.done || t.token != l.Token {      // the fencing check: a stale token cannot finish
		return ErrLeaseLost
	}
	return q.write(record{Op: "done", ID: l.Task, Token: l.Token, Result: result})
}
```

`now.Before(t.deadline)` makes the lease live on $[\text{start}, \text{deadline})$: at the deadline exactly, another worker may lease, and the old owner may no longer renew. Both sides use the same comparison, so there is no instant when two workers both hold it. `Complete` does not check the deadline: a worker whose lease expired but which nobody has re-leased still holds the current token, and letting it finish is harmless and saves a redelivery.

The worker runs the work under a heartbeat:

```go
func (w *Worker) RunOne(ctx context.Context) (bool, error) {
	l, err := w.Q.Lease(w.ID, w.TTL)
	// ... (false, nil) on ErrNoTask; other errors returned
	wctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var mu sync.Mutex
	lost := false
	stop := make(chan struct{})
	renewed := make(chan struct{})
	go func() {                                       // the heartbeat
		defer close(renewed)
		for {
			select {
			case <-stop:
				return
			case <-wctx.Done():
				return
			case <-w.Clock.After(w.TTL / 3):          // the test's clock: no real sleeping
				nl, err := w.Q.Renew(l, w.TTL)
				mu.Lock()
				if err != nil {
					lost = true
					mu.Unlock()
					cancel()                          // tell the work to stop
					return
				}
				l = nl
				mu.Unlock()
			}
		}
	}()
	res, werr := w.Work(wctx, l.Payload)
	close(stop)
	<-renewed                                         // the heartbeat has exited: lost and l are final
	mu.Lock()
	gone := lost
	mu.Unlock()
	if gone {
		return true, ErrLeaseLost                     // never apply after losing ownership
	}
	if werr != nil {
		return true, werr
	}
	if err := w.Apply(l.Task, res); err != nil {      // the idempotency key is the task id
		return true, err
	}
	return true, w.Q.Complete(l, res)
}
```

The ordering is the design: work, then check ownership, then apply, then complete. Apply before complete is what makes a crash between them a duplicate *delivery* rather than a lost effect, and the task-id key is what makes that duplicate harmless. Waiting on `renewed` before reading `lost` closes the race where a renewal fails just as the work returns. The full kata is [`taskq.go`](../../../../course/solutions/primers/craft.21/taskq.go).

## Building the tests

The kill loop is the strongest test: it is the hand-worked crash explored over 200 seeded rounds.

```go
// lcg is a tiny seeded generator: the kill points are the same on every run.
type lcg uint64

func (g *lcg) next(n int) int {
	*g = *g*6364136223846793005 + 1442695040888963407
	return int(uint64(*g>>33) % uint64(n))
}

func TestKillLoopExactlyOnce(t *testing.T) {
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(start), faults.NewEffects()
	q := reopen(t, fs, clk)
	// ... enqueue task-00 .. task-19
	g := lcg(7)
	for round := 0; round < 200 && len(q.Pending()) > 0; round++ {
		point := g.next(4) // 0 before apply, 1 after apply, 2 after complete, 3 no kill
		w := &craft21.Worker{Q: q, ID: fmt.Sprintf("w%d", round), TTL: lease, Clock: clk, Work: upper,
			Apply: func(k, v string) error {
				if point == 0 {
					return errors.New("killed before applying")
				}
				fx.Apply(k, v)
				if point == 1 {
					return errors.New("killed after applying")   // the worst moment: effect done, not acknowledged
				}
				return nil
			}}
		w.RunOne(context.Background())
		if point != 3 {
			fs.Crash()                 // lose the page cache, not just the process
			q = reopen(t, fs, clk)
			clk.Advance(lease)         // let the dead worker's lease expire
		}
	}
	if p := q.Pending(); len(p) > 0 {
		t.Fatalf("tasks never finished: %v", p)
	}
	fx.AssertExactlyOnce(t, ids)       // applied keys == task ids, whatever the deliveries
	fs.Crash()
	q = reopen(t, fs, clk)
	// ... every Result(id) is "R(" + id + ")": acknowledged completions survived
}
```

This one test kills all three resilience faults. A per-attempt key fails `AssertExactlyOnce`. A skipped fsync loses enqueues, leases, or completions on the first `Crash()`, so tasks vanish before their effect is applied (and `AssertExactlyOnce` misses their keys) or results vanish. And the final crash-and-reopen checks completions specifically.

The focused tests each pin one promise at its boundary. Exclusivity comes first:

```go
func TestOneOwnerAtATime(t *testing.T) {
	// ...
	first, _ := q.Lease("w1", lease)
	clk.Advance(lease - time.Nanosecond)              // one nanosecond before the deadline
	if _, err := q.Lease("w2", lease); !errors.Is(err, craft21.ErrNoTask) {
		t.Fatalf("leased twice while the first lease was live: %v", err)
	}
	clk.Advance(time.Nanosecond)                       // exactly at the deadline
	second, err := q.Lease("w2", lease)
	if err != nil || second.Token <= first.Token {
		t.Fatalf("an expired lease must be redelivered with a higher token: %+v %v", second, err)
	}
	// ... the zombie's Renew and Complete must both be ErrLeaseLost
}
```

The two nanosecond steps pin `Before` against `<=` in both directions, which is how a deadline off by one fails.

Renewal is tested with a task that outlives its lease:

```go
func TestLongTaskIsNotStolen(t *testing.T) {
	// ... a worker whose Work blocks on a gate, started in a goroutine
	<-g.started
	for i := 0; i < 6; i++ {
		if err := clk.BlockUntil(1); err != nil {      // wait until the heartbeat is waiting on After
			t.Fatal(err)
		}
		clk.Advance(lease / 3)                         // fire exactly one renewal
		if _, err := q.Lease("thief", lease); !errors.Is(err, craft21.ErrNoTask) {
			t.Fatalf("stolen after %d renewals: %v", i+1, err)
		}
	}
	close(g.release)
	// ... RunOne returns nil
}
```

Six thirds of a TTL is 60 s of work under a 30 s lease: without renewal, the thief gets it on the third step. `BlockUntil(1)` is what removes the race, because advancing the clock before the heartbeat has called `After` would skip a renewal.

The remaining tests are shorter. `TestPausedWorkerStops` advances `3 * lease` in one step: the heartbeat's renewal now fails, the work is cancelled, and `RunOne` must return `ErrLeaseLost` with zero deliveries. `TestTornWriteKeepsTheRest` writes an unsynced record, keeps 7 bytes of it with `CrashTorn(7)`, then checks that the record enqueued *after* reopening survives another crash (the sealing newline). `TestDiskFullLeavesStateAlone` makes `Complete` fail with `ENOSPC` and checks `Result` is still not done. `TestEnqueueTwiceIsOneTask` re-enqueues a done task and checks it is not delivered again. The full tests are in [`resilience_test.go`](../../../../course/solutions/primers/craft.21/resilience_test.go).

## Why this design

The costs are modest. Each mutation is one fsync; a lease is $O(n)$ in the number of tasks (a scan in enqueue order), fine for a kata and replaced by an index in the real engine. Replay is $O(\text{log length})$, and the real engine snapshots to bound it.

A log of records beats rewriting a state file. Rewriting a whole state file per change needs write-temp, fsync, rename, fsync-directory to be crash-safe; appending one line and syncing is one durable step, and a torn tail is easy to detect.

Fencing tokens beat deadlines alone. Deadlines alone assume clocks and pauses are bounded, and a paused process breaks that. The token makes correctness independent of timing: a stale owner is refused however late it wakes.

A fake disk and clock beat real processes and sleeps. A real SIGKILL cannot lose the page cache, so it cannot catch the fsync fault, and real sleeps make the suite slow and flaky. The fakes make every crash point exact and every run identical, at the cost of trusting that the fake models the real file system's guarantees (ALICE and CrashMonkey test that assumption).

The task id as idempotency key beats a stored "applied" flag. A flag in the queue's log would need a write between the effect and the flag, which is the same crash window again. Deduplicating in the external system is the only place where the effect and the check are atomic.

## What can go wrong

Each pitfall below is either a planted fault or a testing mistake that lets one survive:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. no fencing on `Complete` or `Renew` | a paused worker overwrites the current owner's result | both compare `t.token != l.Token` first |
| 2. lease deadlines checked wrong | an expired task is never redelivered, or a live one is delivered twice | one comparison, `now.Before(t.deadline)`, in `Lease` and `Renew`; pinned to the nanosecond by `TestOneOwnerAtATime` |
| 3. torn tails | the queue refuses to start, or loses the first record after a restart | `Open` drops and seals the fragment, and skips undecodable lines |
| 4. enqueue not idempotent | a retried submission runs a done task again | `Enqueue` returns early for a known id; `apply` ignores a second `enq` |
| 5. memory updated before the disk | a completion served from memory vanishes on restart | `write` calls `apply` only after `Sync` succeeds |
| 6. a lost lease does not stop the work | two workers apply the same task's effect | a failed renewal cancels `wctx`, and `RunOne` returns before `Apply` |
| 7. idempotency key per attempt (resilience fault) | a retry applies the effect twice | `w.Apply(l.Task, res)`: the task id, the same on every attempt |
| 8. lease not renewed (resilience fault) | long tasks are stolen mid-run | the heartbeat renews every `TTL/3` |
| 9. fsync skipped (resilience fault) | acknowledged work disappears after a power cut | `Sync` in `write` before returning |
| 10. testing crashes with SIGKILL alone | the fsync fault survives the suite: the page cache outlives the process | the tests use `CrashFS.Crash()`, which keeps only synced bytes |
| 11. real sleeps in resilience tests | flaky timing, slow suites, a check that fails "only sometimes" | a manual `Clock` and `BlockUntil`; the only real timer is a 5 s safety timeout |

> **Warning:** The test that proves exactly-once effects must crash between the effect and the acknowledgement. A suite that only crashes before the effect or after the completion will pass with a per-attempt idempotency key.

## Where this leads

This section reused survivor reading from [Mutation testing in depth](05-mutation-testing-in-depth.md): a surviving resilience fault names the promise the suite never checked. Its worker has the shape of a [subprocess activity](../../systems/07-durable-orchestration/09-subprocess-activities/README.md): lease, heartbeat, idempotent outputs. Going forward, the [durable orchestration chapter](../../systems/07-durable-orchestration/README.md) checks the same promises with a kill loop over the real durable server and worker with 500 activities, and the drills [The durable server is SIGKILLed in a loop](../../systems/12-incident-response-and-chaos/03-durable-kill9.md) and [A poison task lands in the dead-letter queue](../../systems/12-incident-response-and-chaos/04-poison-task.md) in *Systems Engineering* check them on a kind cluster. The [incident-response chapter](../../systems/12-incident-response-and-chaos/README.md) uses resilience tests throughout.

Each piece has a production equivalent:

- `CrashFS` corresponds to ALICE and CrashMonkey, which record every syscall and replay every crash state the file system allows, reordering included (Pillai et al., OSDI 2014).
- The kill loop corresponds to deterministic simulation (FoundationDB, TigerBeetle VOPR): the whole cluster in one thread with a seeded scheduler, so every interleaving is reproducible from a seed (FoundationDB "Simulation" docs).
- The `Effects` checks correspond to Jepsen and Elle, a history checker that proves linearizability or serializability over a recorded run (jepsen.io, `elle`).
- Fencing tokens correspond to Chubby sequencers and etcd lease revisions, where the storage service checks the token, not the client (Burrows, "The Chubby lock service").

### Further reading

- Kyle Kingsbury, [Jepsen analyses](https://jepsen.io/analyses).
- Pillai et al., ["All File Systems Are Not Created Equal"](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/pillai) (OSDI 2014).
- Martin Kleppmann, ["How to do distributed locking"](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html) (fencing tokens).
- FoundationDB, ["Testing Distributed Systems w/ Deterministic Simulation"](https://www.youtube.com/watch?v=4fFDFbi3toc).

## Key terms

- **Resilience test**: a test that injects a fault at a chosen moment and checks a promise of the system afterwards.
- **Durability**: a call that returned success is on disk and survives a crash.
- **Torn write**: a record cut short by a crash in the middle of writing it.
- **Lease**: ownership of a task with a deadline.
- **Fencing token**: a number that grows with every delivery, used to refuse stale owners.
- **Idempotency key**: a key, the same on every attempt, by which the external system deduplicates an effect.
- **Kill loop**: a seeded loop that crashes the system at many points and checks every promise at the end.

## Summary

- A resilience test kills the system at its worst moment and checks a promise, not a return value: effects exactly once, nothing acknowledged lost, one owner per task.
- A SIGKILL loses nothing a `write` returned; to catch a missing fsync the test must lose the page cache, which `CrashFS.Crash()` does by keeping only synced bytes.
- Time is an input: with a manual clock a test can hold a lease for 90 s, pause a worker past its lease, and expire it, in microseconds and the same way every run.
- Exactly once is at-least-once delivery plus an idempotency key that is the same on every attempt; the proving test must crash between the effect and the acknowledgement.
- The suite fails on at least 90% of 12 planted faults, and on all three resilience faults: the idempotency key ignored, the lease not renewed, fsync skipped.

## Review questions

1. Why does `write` call `apply` after `Sync`, and which test fails if it is before?

   <details><summary>Answer</summary>If memory changes first and the write or sync fails, the queue serves state the disk does not have; after a restart it disappears. `TestDiskFullLeavesStateAlone` (a failed `Complete` must leave `Result` not done) catches it, and the kill loop can too.</details>

2. Why must `Open` write a newline after a torn tail, rather than just ignoring it?

   <details><summary>Answer</summary>The file position is after the fragment. Without a newline, the next record is appended to the same line, the combined line fails to decode on the next replay, and an acknowledged record is lost. `TestTornWriteKeepsTheRest` checks exactly that record.</details>

3. Why would a SIGKILL-only suite miss the fsync fault?

   <details><summary>Answer</summary>The page cache belongs to the kernel and survives the process. Bytes written but not synced are still read back after a SIGKILL, so the queue looks durable. Only losing unsynced bytes (`CrashFS.Crash()`, or power loss) exposes it.</details>

4. What goes wrong if the effect's key is `fmt.Sprintf("%s#%d", l.Task, l.Token)`?

   <details><summary>Answer</summary>Every redelivery has a new token, so a crash after the effect and before `Complete` applies the effect again under a new key. The external system cannot deduplicate it; `AssertExactlyOnce` sees `t1#1` and `t1#2`.</details>

5. Why does `RunOne` wait for the heartbeat goroutine to exit before checking `lost`?

   <details><summary>Answer</summary>A renewal could fail just as the work returns. Without the wait, `RunOne` might read `lost == false`, apply the effect, and then the heartbeat marks the lease lost: a worker that no longer owns the task applied its effect. Closing `stop` and receiving on `renewed` makes `lost` final before the decision.</details>
