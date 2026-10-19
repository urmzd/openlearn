// Model test suite for dur.03: properties of the queue,
// black box (package queue_test), on the testkit's fake clock.
//
// Time is the testkit's fake clock: it moves only when a test calls Advance,
// so "the lease expires at 30 s" is an exact statement, not a race. Waiting
// for a long poll uses a real 5 s guard only to turn a hang into a failure.
package queue_test

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	dlog "tinyllm/durable/log"
	"tinyllm/durable/queue"

	"openlearn.urmzd.com/tl/testkit/clock"
	"openlearn.urmzd.com/tl/testkit/effects"
)

var ctx = context.Background()

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func fastSync(f *os.File) error { return syscall.Fsync(int(f.Fd())) }

func newQueue(t *testing.T, o queue.Options) (*queue.Queue, *clock.Fake) {
	t.Helper()
	clk := clock.NewFake(t0)
	if o.Clock == nil {
		o.Clock = clk
	}
	q, err := queue.Open(ctx, o)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return q, clk
}

func enqueue(t *testing.T, q *queue.Queue, name string, ids ...string) {
	t.Helper()
	for _, id := range ids {
		if err := q.Enqueue(ctx, name, queue.Task{ID: id, Payload: []byte("p-" + id)}); err != nil {
			t.Fatalf("Enqueue(%s): %v", id, err)
		}
	}
}

// pollCtx bounds every poll by 5 s of real time, so a poll that spins or
// waits forever (a planted bug) fails the test instead of hanging it.
func pollCtx(t *testing.T) context.Context {
	c, cancel := context.WithTimeout(ctx, 5*time.Second)
	t.Cleanup(cancel)
	return c
}

func poll(t *testing.T, q *queue.Queue, name string) queue.Lease {
	t.Helper()
	l, err := q.Poll(pollCtx(t), name, "w", 0)
	if err != nil {
		t.Fatalf("Poll(%s): %v", name, err)
	}
	return l
}

func noTask(t *testing.T, q *queue.Queue, name string) {
	t.Helper()
	if l, err := q.Poll(pollCtx(t), name, "w", 0); !errors.Is(err, queue.ErrNoTask) {
		t.Fatalf("Poll(%s) = %s attempt %d, %v; want ErrNoTask", name, l.Task.ID, l.Attempt, err)
	}
}

func want(t *testing.T, l queue.Lease, id string, attempt int, token uint64, deadline time.Duration) {
	t.Helper()
	if l.Task.ID != id || l.Attempt != attempt || l.Token != token || !l.Deadline.Equal(t0.Add(deadline)) {
		t.Fatalf("lease %s attempt %d token %d deadline +%v; want %s attempt %d token %d deadline +%v",
			l.Task.ID, l.Attempt, l.Token, l.Deadline.Sub(t0), id, attempt, token, deadline)
	}
}

func TestHandTimeline(t *testing.T) {
	q, clk := newQueue(t, queue.Options{Visibility: 30 * time.Second, MaxAttempts: 3})
	enqueue(t, q, "data", "a", "b")
	a1 := poll(t, q, "data")
	want(t, a1, "a", 1, 1, 30*time.Second)
	b1 := poll(t, q, "data")
	want(t, b1, "b", 1, 2, 30*time.Second)
	if err := q.Complete(ctx, b1); err != nil {
		t.Fatalf("Complete(b): %v", err)
	}
	clk.Advance(10 * time.Second) // t = 10
	if d, err := q.Fail(ctx, a1, queue.TaskError{Message: "flaky", RetryAfter: 5 * time.Second}); err != nil || d != queue.Retrying {
		t.Fatalf("Fail(a) = %v, %v; want Retrying", d, err)
	}
	noTask(t, q, "data")         // a is hidden until t = 15
	clk.Advance(5 * time.Second) // t = 15
	a2 := poll(t, q, "data")
	want(t, a2, "a", 2, 3, 45*time.Second)
	clk.Advance(30 * time.Second) // t = 45: a2's deadline, the worker died
	a3 := poll(t, q, "data")
	want(t, a3, "a", 3, 4, 75*time.Second)
	if err := q.Complete(ctx, a2); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("late Complete with token 3: %v, want ErrLeaseLost", err)
	}
	clk.Advance(30 * time.Second) // t = 75: the last attempt's lease expires
	noTask(t, q, "data")
	dlq, _ := q.DLQ(ctx, "data")
	if len(dlq) != 1 || dlq[0].Task.ID != "a" || dlq[0].Attempts != 3 || !dlq[0].At.Equal(t0.Add(75*time.Second)) {
		t.Fatalf("DLQ = %+v; want a after 3 attempts at +75s", dlq)
	}
	if n, err := q.Redrive(ctx, "data"); n != 1 || err != nil {
		t.Fatalf("Redrive = %d, %v", n, err)
	}
	a4 := poll(t, q, "data")
	want(t, a4, "a", 1, 5, 105*time.Second)
	if string(a4.Task.Payload) != "p-a" {
		t.Fatalf("payload %q survived redrive as %q", "p-a", a4.Task.Payload)
	}
}

func TestEnqueueDuplicateIsNoop(t *testing.T) {
	q, _ := newQueue(t, queue.Options{MaxAttempts: 1})
	enqueue(t, q, "q", "x", "y")
	lx := poll(t, q, "q")
	if err := q.Enqueue(ctx, "q", queue.Task{ID: "x", Payload: []byte("other")}); err != nil {
		t.Fatal(err)
	}
	if err := q.Enqueue(ctx, "q", queue.Task{ID: "y", Payload: []byte("other")}); err != nil {
		t.Fatal(err)
	}
	if v, l, d := q.Depth("q"); v != 1 || l != 1 || d != 0 {
		t.Fatalf("depth after duplicate enqueues: visible %d leased %d dead %d; want 1 1 0", v, l, d)
	}
	if err := q.Check(lx); err != nil {
		t.Fatalf("a duplicate enqueue broke the live lease of x: %v", err)
	}
	ly := poll(t, q, "q")
	if string(ly.Task.Payload) != "p-y" {
		t.Fatalf("duplicate enqueue replaced y's payload with %q", ly.Task.Payload)
	}
	q.Fail(ctx, ly, queue.TaskError{Message: "boom"}) // MaxAttempts 1: dead
	enqueue(t, q, "q", "y")
	if v, l, d := q.Depth("q"); v != 0 || l != 1 || d != 1 {
		t.Fatalf("enqueue of a dead-lettered id changed depth: %d %d %d", v, l, d)
	}
	q.Complete(ctx, lx)
	enqueue(t, q, "q", "x")
	if l := poll(t, q, "q"); l.Task.ID != "x" || l.Attempt != 1 {
		t.Fatalf("after its ack, x enqueued again should be a new task: %+v", l)
	}
}

func TestFIFOAmongVisible(t *testing.T) {
	q, clk := newQueue(t, queue.Options{})
	enqueue(t, q, "q", "1", "2")
	l1 := poll(t, q, "q")
	q.Fail(ctx, l1, queue.TaskError{RetryAfter: 2 * time.Second}) // "1" visible at +2s
	clk.Advance(time.Second)
	enqueue(t, q, "q", "3") // visible at +1s
	clk.Advance(5 * time.Second)
	var got []string
	for i := 0; i < 3; i++ {
		got = append(got, poll(t, q, "q").Task.ID)
	}
	if strings.Join(got, ",") != "2,3,1" {
		t.Fatalf("delivery order %v; want 2 (t=0), 3 (t=1), 1 (retry at t=2)", got)
	}
}

func TestStaleTokenLeaseLost(t *testing.T) {
	q, clk := newQueue(t, queue.Options{Visibility: 10 * time.Second})
	enqueue(t, q, "q", "t")
	old := poll(t, q, "q")
	clk.Advance(10 * time.Second) // deadline reached, not yet redelivered
	if err := q.Check(old); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("Check of an expired lease: %v, want ErrLeaseLost", err)
	}
	if err := q.Complete(ctx, old); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("Complete of an expired lease: %v, want ErrLeaseLost", err)
	}
	cur := poll(t, q, "q")
	if cur.Token <= old.Token || cur.Attempt != 2 {
		t.Fatalf("redelivery token %d (old %d) attempt %d", cur.Token, old.Token, cur.Attempt)
	}
	if _, err := q.Heartbeat(ctx, old, []byte("x")); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("Heartbeat with the stale token: %v", err)
	}
	if _, err := q.Fail(ctx, old, queue.TaskError{NonRetryable: true}); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("Fail with the stale token: %v", err)
	}
	if err := q.Complete(ctx, old); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("Complete with the stale token: %v", err)
	}
	if err := q.Check(cur); err != nil {
		t.Fatalf("the stale calls disturbed the live lease: %v", err)
	}
	if err := q.Complete(ctx, cur); err != nil {
		t.Fatalf("Complete with the live token: %v", err)
	}
	if err := q.Complete(ctx, cur); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("a second Complete of the same lease: %v, want ErrLeaseLost", err)
	}
}

func TestHeartbeatExtendsLease(t *testing.T) {
	q, clk := newQueue(t, queue.Options{Visibility: 30 * time.Second})
	enqueue(t, q, "q", "long")
	l := poll(t, q, "q")
	clk.Advance(20 * time.Second)
	l, err := q.Heartbeat(ctx, l, []byte("ckpt-1"))
	if err != nil || !l.Deadline.Equal(t0.Add(50*time.Second)) {
		t.Fatalf("Heartbeat at +20s: deadline +%v, %v; want +50s", l.Deadline.Sub(t0), err)
	}
	clk.Advance(25 * time.Second) // +45s: past the original deadline, before the new one
	noTask(t, q, "q")
	if err := q.Check(l); err != nil {
		t.Fatalf("lease lost at +45s despite the heartbeat: %v", err)
	}
	clk.Advance(5 * time.Second) // +50s: the worker stopped heartbeating
	l2 := poll(t, q, "q")
	if l2.Attempt != 2 || string(l2.Details) != "ckpt-1" {
		t.Fatalf("redelivery attempt %d details %q; want 2 and ckpt-1", l2.Attempt, l2.Details)
	}
}

func TestPoisonTaskToDLQAfterMaxAttempts(t *testing.T) {
	q, _ := newQueue(t, queue.Options{MaxAttempts: 3})
	enqueue(t, q, "q", "poison")
	q.Enqueue(ctx, "q", queue.Task{ID: "fragile", MaxAttempts: 1})
	var ds []queue.Disposition
	for i := 0; i < 4; i++ {
		l, err := q.Poll(pollCtx(t), "q", "w", 0)
		if err != nil {
			break
		}
		d, err := q.Fail(ctx, l, queue.TaskError{Message: fmt.Sprintf("panic %d (%s)", l.Attempt, l.Task.ID)})
		if err != nil {
			t.Fatal(err)
		}
		ds = append(ds, d)
	}
	// A retry with no delay is visible at once and keeps its place (it became
	// visible at t = 0, like fragile, and was enqueued first).
	if fmt.Sprint(ds) != fmt.Sprint([]queue.Disposition{queue.Retrying, queue.Retrying, queue.DeadLettered, queue.DeadLettered}) {
		t.Fatalf("dispositions %v; want retry(poison 1), retry(poison 2), dead(poison 3), dead(fragile 1)", ds)
	}
	noTask(t, q, "q")
	dlq, _ := q.DLQ(ctx, "q")
	if len(dlq) != 2 || dlq[0].Task.ID != "poison" || dlq[0].Attempts != 3 || dlq[0].LastError != "panic 3 (poison)" ||
		dlq[1].Task.ID != "fragile" || dlq[1].Attempts != 1 {
		t.Fatalf("DLQ = %+v", dlq)
	}
}

func TestNonRetryableDropped(t *testing.T) {
	q, _ := newQueue(t, queue.Options{MaxAttempts: 5})
	enqueue(t, q, "q", "bad")
	d, err := q.Fail(ctx, poll(t, q, "q"), queue.TaskError{Message: "invalid spec", NonRetryable: true})
	if err != nil || d != queue.Dropped {
		t.Fatalf("Fail(non-retryable) = %v, %v; want Dropped", d, err)
	}
	if v, l, dd := q.Depth("q"); v+l+dd != 0 {
		t.Fatalf("a dropped task is still in the queue: %d %d %d", v, l, dd)
	}
}

func TestRedrive(t *testing.T) {
	q, _ := newQueue(t, queue.Options{MaxAttempts: 2})
	enqueue(t, q, "q", "a", "b", "c")
	for i := 0; i < 6; i++ {
		l := poll(t, q, "q")
		q.Fail(ctx, l, queue.TaskError{Message: "x"})
	}
	if _, _, d := q.Depth("q"); d != 3 {
		t.Fatalf("setup: %d dead letters, want 3", d)
	}
	if n, err := q.Redrive(ctx, "q", "b", "nope"); !errors.Is(err, queue.ErrNotFound) || n != 0 {
		t.Fatalf("Redrive with an unknown id = %d, %v; want 0, ErrNotFound", n, err)
	}
	if _, _, d := q.Depth("q"); d != 3 {
		t.Fatal("a failed redrive moved something")
	}
	if n, err := q.Redrive(ctx, "q", "b"); n != 1 || err != nil {
		t.Fatalf("Redrive(b) = %d, %v", n, err)
	}
	lb := poll(t, q, "q")
	if lb.Task.ID != "b" || lb.Attempt != 1 {
		t.Fatalf("redriven b: %s attempt %d", lb.Task.ID, lb.Attempt)
	}
	if d, _ := q.Fail(ctx, lb, queue.TaskError{}); d != queue.Retrying {
		t.Fatalf("first failure after a redrive: %v, want Retrying (the attempt count restarted)", d)
	}
	if n, _ := q.Redrive(ctx, "q"); n != 2 {
		t.Fatalf("Redrive(all) moved %d, want 2", n)
	}
}

// blockUntil waits until n fake-clock timers are pending, turning a poll
// that never waits (a planted bug) into a failure instead of a hang.
func blockUntil(t *testing.T, clk *clock.Fake, n int) {
	t.Helper()
	done := make(chan struct{})
	go func() { clk.BlockUntil(n); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatalf("the poll never started waiting on the clock (5 s real-time guard)")
	}
}

func pollAsync(q *queue.Queue, name string, wait time.Duration) <-chan queue.Lease {
	ch := make(chan queue.Lease, 1)
	go func() {
		l, err := q.Poll(ctx, name, "w", wait)
		if err != nil {
			l.Task.ID = "error: " + err.Error()
		}
		ch <- l
	}()
	return ch
}

func await(t *testing.T, ch <-chan queue.Lease) queue.Lease {
	t.Helper()
	select {
	case l := <-ch:
		return l
	case <-time.After(5 * time.Second):
		t.Fatal("the long poll did not return (5 s real-time guard)")
		return queue.Lease{}
	}
}

func TestLongPollWakesOnEnqueue(t *testing.T) {
	q, clk := newQueue(t, queue.Options{})
	ch := pollAsync(q, "q", 30*time.Second)
	blockUntil(t, clk, 1)
	enqueue(t, q, "q", "now")
	if l := await(t, ch); l.Task.ID != "now" {
		t.Fatalf("long poll returned %q", l.Task.ID)
	}
}

func TestLongPollWakesOnRetryDue(t *testing.T) {
	q, clk := newQueue(t, queue.Options{})
	enqueue(t, q, "q", "r")
	q.Fail(ctx, poll(t, q, "q"), queue.TaskError{RetryAfter: 5 * time.Second})
	ch := pollAsync(q, "q", 30*time.Second)
	blockUntil(t, clk, 1)
	clk.Advance(5 * time.Second)
	if l := await(t, ch); l.Task.ID != "r" || l.Attempt != 2 {
		t.Fatalf("long poll returned %q attempt %d", l.Task.ID, l.Attempt)
	}
}

func TestLongPollTimesOut(t *testing.T) {
	q, clk := newQueue(t, queue.Options{})
	ch := make(chan error, 1)
	go func() { _, err := q.Poll(ctx, "q", "w", 30*time.Second); ch <- err }()
	blockUntil(t, clk, 1)
	clk.Advance(30 * time.Second)
	select {
	case err := <-ch:
		if !errors.Is(err, queue.ErrNoTask) {
			t.Fatalf("Poll after its wait: %v, want ErrNoTask", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Poll did not return after its wait")
	}
}

func TestCrashedWorkerTaskReappears(t *testing.T) {
	q, clk := newQueue(t, queue.Options{Visibility: 30 * time.Second})
	enqueue(t, q, "q", "job")
	poll(t, q, "q") // the worker that dies
	clk.Advance(30*time.Second - time.Millisecond)
	noTask(t, q, "q")
	clk.Advance(time.Millisecond)
	if l := poll(t, q, "q"); l.Task.ID != "job" || l.Attempt != 2 {
		t.Fatalf("after the timeout: %s attempt %d", l.Task.ID, l.Attempt)
	}
}

func TestRecoveryFromLog(t *testing.T) {
	dir := t.TempDir()
	lg, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
	if err != nil {
		t.Fatal(err)
	}
	clk := clock.NewFake(t0)
	q, _ := newQueue(t, queue.Options{Log: lg, Clock: clk, Visibility: 30 * time.Second, MaxAttempts: 2})
	enqueue(t, q, "q", "a", "b", "c", "d")
	la := poll(t, q, "q")                                                  // a: token 1, deadline +30s
	q.Fail(ctx, poll(t, q, "q"), queue.TaskError{RetryAfter: time.Minute}) // b: token 2, visible at +60s
	q.Fail(ctx, poll(t, q, "q"), queue.TaskError{Message: "x"})            // c: token 3, visible now
	clk.Advance(time.Minute)                                               // +60s: a's lease has expired
	lc := poll(t, q, "q")                                                  // c again: attempt 2, token 4
	if lc.Task.ID != "c" || lc.Attempt != 2 {
		t.Fatalf("setup: polled %s attempt %d, want c attempt 2", lc.Task.ID, lc.Attempt)
	}
	q.Fail(ctx, lc, queue.TaskError{Message: "boom"}) // MaxAttempts 2: c is dead
	ld := poll(t, q, "q")                             // d: token 5, live until +90s
	lg.Close()                                        // the server crashes

	lg2, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
	if err != nil {
		t.Fatal(err)
	}
	defer lg2.Close()
	q2, err := queue.Open(ctx, queue.Options{Log: lg2, Clock: clk, Visibility: 30 * time.Second, MaxAttempts: 2})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if v, l, d := q2.Depth("q"); v != 2 || l != 1 || d != 1 {
		t.Fatalf("after reopen: visible %d leased %d dead %d; want 2 (a, b) 1 (d) 1 (c)", v, l, d)
	}
	if dlq, _ := q2.DLQ(ctx, "q"); len(dlq) != 1 || dlq[0].Task.ID != "c" || dlq[0].Attempts != 2 || dlq[0].LastError != "boom" {
		t.Fatalf("DLQ after reopen = %+v", dlq)
	}
	if err := q2.Complete(ctx, la); !errors.Is(err, queue.ErrLeaseLost) {
		t.Fatalf("a's expired pre-crash lease completed after reopen: %v", err)
	}
	var ids []string
	last := uint64(5)
	for i := 0; i < 2; i++ {
		l := poll(t, q2, "q")
		ids = append(ids, l.Task.ID+strconv.Itoa(l.Attempt))
		if l.Token <= last {
			t.Fatalf("token %d after reopen, but tokens up to %d were already handed out", l.Token, last)
		}
		last = l.Token
	}
	if strings.Join(ids, " ") != "a2 b2" {
		t.Fatalf("after reopen delivered %v; want a2 b2", ids)
	}
	if err := q2.Complete(ctx, ld); err != nil {
		t.Fatalf("d's lease was live across the restart; Complete: %v", err)
	}
}

func TestLogFailureChangesNothing(t *testing.T) {
	lg, err := dlog.Open(t.TempDir(), dlog.Options{Sync: fastSync, MaxBytes: 400})
	if err != nil {
		t.Fatal(err)
	}
	defer lg.Close()
	q, _ := newQueue(t, queue.Options{Log: lg})
	enqueue(t, q, "q", "a")
	var failed error
	for i := 0; i < 20 && failed == nil; i++ {
		failed = q.Enqueue(ctx, "q", queue.Task{ID: fmt.Sprintf("t%02d", i), Payload: []byte("0123456789")})
	}
	if !errors.Is(failed, dlog.ErrQuota) {
		t.Fatalf("expected the log quota to stop enqueues, got %v", failed)
	}
	v, _, _ := q.Depth("q")
	if v != int(lg.Version("queue/q")) {
		t.Fatalf("%d visible tasks but %d enqueue records: memory ran ahead of the log", v, lg.Version("queue/q"))
	}
}

func TestConcurrentPollersOneLeaseEach(t *testing.T) {
	q, _ := newQueue(t, queue.Options{})
	for i := 0; i < 200; i++ {
		enqueue(t, q, "q", strconv.Itoa(i))
	}
	var mu sync.Mutex
	got := map[string]int{}
	tokens := map[uint64]bool{}
	var wg sync.WaitGroup
	pc := pollCtx(t)
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				l, err := q.Poll(pc, "q", "w", 0)
				if err != nil {
					return
				}
				mu.Lock()
				got[l.Task.ID]++
				tokens[l.Token] = true
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	if len(got) != 200 || len(tokens) != 200 {
		t.Fatalf("%d distinct tasks and %d distinct tokens leased; want 200 and 200", len(got), len(tokens))
	}
	for id, n := range got {
		if n != 1 {
			t.Fatalf("task %s leased %d times", id, n)
		}
	}
}

func TestExactlyOnceEffectsUnderChaos(t *testing.T) {
	sink, err := effects.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()
	seed, _ := strconv.ParseUint(os.Getenv("OL_SEED"), 10, 64)
	rng := rand.New(rand.NewPCG(seed, 3))
	q, clk := newQueue(t, queue.Options{Visibility: 10 * time.Second, MaxAttempts: 1000})
	var keys []string
	for i := 0; i < 60; i++ {
		id := fmt.Sprintf("task-%02d", i)
		keys = append(keys, id)
		enqueue(t, q, "q", id)
	}
	acks := map[string]int{}
	var stale []queue.Lease
	pc := pollCtx(t)
	for steps := 0; len(acks) < 60 && steps < 5000; steps++ {
		l, err := q.Poll(pc, "q", "w", 0)
		if errors.Is(err, queue.ErrNoTask) {
			clk.Advance(10 * time.Second)
			continue
		}
		if err != nil {
			t.Fatalf("Poll: %v", err)
		}
		switch r := rng.IntN(10); {
		case r < 2: // dies before the effect
		case r < 4: // applies the effect, then dies before Complete
			post(t, sink.URL(), l.Task.ID)
			stale = append(stale, l)
		case r < 5: // fails and is retried
			q.Fail(ctx, l, queue.TaskError{Message: "flaky"})
		default:
			post(t, sink.URL(), l.Task.ID)
			if err := q.Complete(ctx, l); err == nil {
				acks[l.Task.ID]++
			}
		}
		if len(stale) > 0 && rng.IntN(4) == 0 { // a slow worker finally answers
			s := stale[0]
			stale = stale[1:]
			if err := q.Complete(ctx, s); err == nil {
				acks[s.Task.ID]++
			}
		}
	}
	for _, k := range keys {
		if acks[k] != 1 {
			t.Fatalf("%s acknowledged %d times; want 1", k, acks[k])
		}
	}
	sink.AssertExactlyOnce(t, keys)
}

func post(t *testing.T, url, key string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(fmt.Sprintf(`{"key":%q,"value":"v"}`, key)))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
}
