<!-- ol:module craft.21 -->
# Resilience tests (R10)

## Overview

| | |
|---|---|
| **Module** | `craft.21` · practice · Go · Pass 8 · 4 to 6 h |
| **You build** | `primers/craft.21/taskq.go` (a kata written by `ol start`: a durable task queue and its worker, the log, leases, and idempotent effects of your durable engine in one file) and `primers/craft.21/resilience_test.go` (your **resilience tests**, package `craft21_test`) |
| **Contract** | none of its own: the kata's API is in its file, and the course's fault kit is `course/tests/craft.21/faults/faults.go` (copied to `primers/craft.21/faults/` by the first check) |
| **Tests** | `course/tests/craft.21/check`: the course's suite (`course/tests/go/craft_21/`) on your kata, then the grade of your tests by 12 planted faults (section 4) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how tests are graded · [`craft.04`](02-property-based-tests.md) properties · [`craft.07`](05-mutation-testing-in-depth.md) reading survivors · [`lang.06`](../12-language-and-tool-primers/06-go.md) Go `testing` · [`dur.09`](../../ai-platform-engineering/05-durable-orchestration-and-workers/09-subprocess-activities.md) subprocess activities; the engine pieces the kata models: the event log (`dur.01`), fenced leases (`dur.03`), idempotent activities (`dur.05`) |
| **Used by** | no call site (a practice): rung R10 grades resilience tests in the durable engine (`dur.*`) and the drills (`ops.*`) from here on |
| **Milestone** | MS-durable (its kill loop is the system-scale version of your tests) |
| **Optional depth** | Kyle Kingsbury, [Jepsen analyses](https://jepsen.io/analyses) (free); Pillai et al., ["All File Systems Are Not Created Equal"](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/pillai) (OSDI 2014, free); Martin Kleppmann, ["How to do distributed locking"](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html) (fencing tokens, free); FoundationDB, ["Testing Distributed Systems w/ Deterministic Simulation"](https://www.youtube.com/watch?v=4fFDFbi3toc) (free) |

## Key Takeaways

- A **resilience test** kills the system at its worst moment and checks a **promise**, not a return value: effects exactly once, nothing acknowledged lost, one owner per task.
- A SIGKILL loses nothing a `write` returned. To catch a missing **fsync** the test must lose the page cache: the fault kit's `CrashFS.Crash()` keeps only synced bytes.
- Time is an input. With a **manual clock** a test can hold a lease for 90 s, pause a worker past its lease, and expire it, in microseconds and the same way every run.
- **Exactly once** is "at least once delivery" plus an **idempotency key that is the same on every attempt**. The test that proves it must crash between the effect and the acknowledgement.
- The grade is rung R10: your tests must fail on at least 90% of 12 planted faults, and on all three resilience faults: the idempotency key ignored, the lease not renewed, fsync skipped.

## How to work this chapter

```bash
ol start craft.21                       # writes primers/craft.21/taskq.go with stub bodies
ol tests craft.21                       # read the test catalog first
ol check craft.21                       # first run: writes go.mod and faults/faults.go, then fails (no tests yet)
# write resilience_test.go (section 4), against the stub first: every test must fail
(cd primers/craft.21 && go test ./...)
# then make the kata pass your tests and the course's
ol check craft.21                       # grades your tests by planted faults
```

`ol check` never shows a planted fault's code. A surviving resilience fault prints its one-line name (they are public: section 2.5); any other survivor prints only "a planted pitfall".

---

## 1. Why now

Your durable engine now makes three promises that no unit test has ever checked: an acknowledged event survives a crash (`dur.01`), a task has one owner at a time (`dur.03`), and an activity's effect happens once however often it is retried (`dur.05`, `dur.09`). Unit tests run the happy path in one process that never dies, so a log that forgets to fsync, a worker that forgets to renew its lease, or an activity keyed by its attempt number passes all of them. These bugs appear only when a machine dies at the wrong instant, which in production is "eventually, at 3 a.m., during the biggest training run". MS-durable will kill your real server and worker 500 times; this module teaches you to write the small, deterministic version of that test first, on a kata small enough to hold in your head, and grades it by whether it catches the faults that matter.

## 2. Principles

### 2.1 Promises, not return values

A resilience test is a loop: put the system in a state, inject a fault, let it recover, check an invariant. The invariants are the system's promises, stated so a test can check them after the fact:

| Promise | Checked as | The kata's mechanism |
|---|---|---|
| **durability**: a call that returned nil is on disk | after `Crash()` and `Open`, every acknowledged enqueue, lease, and completion replays | append one JSON line, `Sync`, only then update memory |
| **exclusivity**: one owner per task at a time | while a lease is live, `Lease` gives `ErrNoTask`; a stale token's `Renew` and `Complete` get `ErrLeaseLost` | leases with a deadline and a **fencing token** that grows on every delivery |
| **exactly-once effect** | the external system's applied keys equal the task ids, with deliveries counted separately | the effect is keyed by the task id, the same on every attempt |
| **idempotent submission** | enqueueing an id twice is one task | `Enqueue` of an existing id is a no-op |

Delivery is **at least once**: a worker can die after applying an effect and before completing, and the retry applies it again. That is unavoidable (the Two Generals problem), so the effect must be **idempotent**: the external system deduplicates by key. The fault kit's `Effects` does exactly that, and counts deliveries, so a test sees both the state (`Keys`) and the retries (`Deliveries`).

### 2.2 Crashes: what survives and what does not

| Fault | What is lost | What a test needs |
|---|---|---|
| SIGKILL of the process | memory; nothing a `write` returned (the page cache belongs to the kernel) | a child process, or abandoning the goroutine |
| power loss, kernel panic | memory **and** every byte written but not synced | a disk model: `CrashFS.Crash()` |
| a crash during a write | the end of that write: a **torn** record | `CrashFS.CrashTorn(n)` keeps `n` bytes of the unsynced tail |
| a full disk | nothing yet; the write fails | `CrashFS.FailWrites(err)` |

Two rules follow. First, **sync before acknowledging**, and update memory only after the sync succeeded: a queue that changes memory first serves a completion that vanishes on restart. Second, **replay must survive a torn tail**: the last line may be half a record. Its call never returned, so it is safe to ignore; but the next append lands right after it, so `Open` must **seal** the fragment with a newline, or the first record acknowledged after the restart is glued to garbage and lost on the next replay.

### 2.3 Leases, renewals, and fencing tokens

A **lease** is ownership with a deadline: worker `w` may work on task `t` until `now + TTL`. If `w` dies, the task comes back after the deadline. Two things go wrong:

- A task that takes longer than the TTL loses its lease mid-run and a second worker starts it too. The worker must **renew** (heartbeat) while it works, every `TTL/3` in the kata, so two missed renewals still leave time.
- A worker that was paused (GC, a stopped VM, a partition) wakes up after its lease expired and someone else owns the task. It must not finish: every lease carries a **fencing token**, a number that grows with every delivery, and `Renew` and `Complete` with a stale token are `ErrLeaseLost`. When a renewal fails, the worker cancels its work and does not apply the effect.

### 2.4 Deterministic fault injection

The kata takes its disk (`FS`) and its time (`Clock`) as interfaces, so a test owns both. With the fault kit's `Clock`, time moves only on `Advance`, and `BlockUntil(n)` waits until the code under test is actually waiting, so a test never sleeps and never races. With `CrashFS`, a "crash" is one call at an exact point. The test is the same on every run, which is what makes a failure a bug report instead of a flake. A **kill loop** then explores many crash points with a seeded generator: each round picks where to die (before the effect, after it, after `Complete`), crashes the disk, reopens the queue, expires the leases, and continues until no task is pending; at the end the promises of 2.1 must hold.

### 2.5 How R10 grades you

Your tests run against the course's correct kata (they must pass, twice), then against 12 copies of it, each with one planted fault. A fault your tests make fail is **killed**. Rung R10 needs a score of at least 0.90 and every **required** fault killed. Three of them are the resilience faults of the testing ladder, and their names are public:

| Fault | What it breaks | A test that catches it |
|---|---|---|
| idempotency key ignored | the effect's key changes per attempt, so a retry applies it twice | crash after the effect, retry, `AssertExactlyOnce` |
| lease not renewed | a long task is handed to a second worker mid-run | hold a task for 3 x TTL, try to lease it from another worker |
| fsync skipped | acknowledged records live only in the page cache | `Crash()`, reopen, check acknowledged state |

The other nine are pitfalls (section 5) and stay hidden until you pass.

## 3. Worked example by hand

Clock at $t_0$ = 1760000000 s (Unix), TTL 30 s, empty disk.

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

Every promise holds: applied keys are exactly `{t1, t2}`, `t1` was delivered twice and applied once, the zombie's completion was refused. This is the course's `TestHandExampleCrashBeforeComplete`; with the idempotency key changed to `t1#<token>`, step 7 applies a second key `t1#2` and `AssertExactlyOnce` fails.

## 4. The artifact and its check

Write `primers/craft.21/resilience_test.go` in package `craft21_test` (black box: only the kata's exported API and `craft21/faults`). It must at least crash the disk, move time past a lease, restart the queue with `craft21.Open`, and assert on effects; the check rejects a suite that never does. Cover each promise of 2.1 with at least one test, and write one kill loop (2.4).

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

// primers/craft.21/faults/faults.go (the course's kit)
fs := faults.NewCrashFS()   // Crash(), CrashTorn(n), FailWrites(err), Durable(name)
clk := faults.NewClock(t0)  // Now, After, Advance(d), BlockUntil(n)
fx := faults.NewEffects()   // Apply(key, v), Deliveries(key), Keys(), FailNext(key, n), AssertExactlyOnce(t, keys)
```

### What the tests check

The check (`course/tests/craft.21/artifacts.py`):

| Test | KIND | Checks |
|---|---|---|
| `test_files_present` | unit | the kata and your tests exist; writes `go.mod` and `faults/faults.go` once |
| `test_your_tests_inject_faults` | unit | black-box package; uses a crash, a clock advance, a restart, and an effect assertion |
| `test_your_kata_passes_the_course_suite` | fault | the course's suite passes on your kata |
| `test_your_tests_pass_on_your_kata` | unit | your tests pass on your kata |
| `test_your_tests_pass_on_the_course_kata` | unit | your tests pass on the course's kata, twice (baseline A, and no flakes) |
| `test_your_tests_catch_the_planted_faults` | fault | score >= 0.90 over 12 faults, every required one killed |

The course's suite (`course/tests/go/craft_21/`), run on your kata:

| Test | KIND | Checks |
|---|---|---|
| `TestHandExampleCrashBeforeComplete` | fault | section 3, line by line |
| `TestEnqueueIsIdempotent` | unit | a duplicate enqueue is one task, even after it is done |
| `TestAcknowledgedSurvivesCrash` | fault | acknowledged enqueues and completions replay after `Crash()` |
| `TestTornTailIsIgnored` | fault | a torn record is ignored and sealed; the next record survives the next crash |
| `TestFailedWriteChangesNothing` | fault | a failed write leaves memory as it was |
| `TestLeaseIsExclusiveUntilExpiry` | boundary | no second lease until the deadline, then a higher token and attempt 2 |
| `TestStaleTokenIsFenced` | fault | a zombie's `Renew` and `Complete` are `ErrLeaseLost` |
| `TestLongTaskKeepsItsLease` | fault | a 90 s task keeps its 30 s lease by renewing |
| `TestLostLeaseCancelsWork` | fault | a failed renewal cancels the work and skips the effect |
| `TestCrashAfterApplyIsExactlyOnce` | fault | crash between effect and completion: delivered twice, applied once |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. no fencing on `Complete` or `Renew` | a paused worker overwrites the current owner's result | `TestStaleTokenIsFenced` (mutants `s01`, `s06`) |
| 2. lease deadlines checked wrong | an expired task is never redelivered, or a live one is delivered twice | `TestLeaseIsExclusiveUntilExpiry` (mutants `s02`, `s03`) |
| 3. torn tails | the queue refuses to start, or loses the first record after a restart | `TestTornTailIsIgnored` (mutants `s04`, `s09`) |
| 4. enqueue not idempotent | a retried submission runs a done task again | `TestEnqueueIsIdempotent` (mutant `s05`) |
| 5. memory updated before the disk | a completion served from memory vanishes on restart | `TestFailedWriteChangesNothing` (mutant `s07`) |
| 6. a lost lease does not stop the work | two workers apply the same task's effect | `TestLostLeaseCancelsWork` (mutant `s08`) |
| 7. idempotency key per attempt (resilience fault) | a retry applies the effect twice | `TestCrashAfterApplyIsExactlyOnce` (mutant `s10`) |
| 8. lease not renewed (resilience fault) | long tasks are stolen mid-run | `TestLongTaskKeepsItsLease` (mutant `s11`) |
| 9. fsync skipped (resilience fault) | acknowledged work disappears after a power cut | `TestAcknowledgedSurvivesCrash` (mutant `s12`) |
| 10. testing crashes with SIGKILL alone | the fsync fault survives your suite: the page cache outlives the process | your grade: `s12` survives |
| 11. real sleeps in resilience tests | flaky timing, slow suites, a check that fails "only sometimes" | `test_your_tests_pass_on_the_course_kata` (two runs) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.07` | reading survivors: a surviving resilience fault names the promise your suite never checked |
| Back | `dur.09` | the kata's worker is the shape of a subprocess activity: lease, heartbeat, idempotent outputs |
| Forward | MS-durable | the same promises checked by a kill loop over your real `{durable}` and `{worker}` with 500 activities |
| Forward | `ops.02`, `ops.03` | drills `durable-kill9` and `poison-task` check these promises on kind |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `CrashFS` | ALICE, CrashMonkey | record every syscall and replay every crash state the file system allows, reordering included | Pillai et al., OSDI 2014 |
| the kill loop | deterministic simulation (FoundationDB, TigerBeetle VOPR) | the whole cluster in one thread with a seeded scheduler: every interleaving is reproducible from a seed | FoundationDB "Simulation" docs |
| `Effects` checks | Jepsen and Elle | a history checker that proves linearizability or serializability over a recorded run | jepsen.io, `elle` |
| fencing tokens | Chubby sequencers, etcd lease revisions | the storage service checks the token, not the client | Burrows, "The Chubby lock service" |
