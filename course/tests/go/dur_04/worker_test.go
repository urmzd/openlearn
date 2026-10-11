package dur_04

// Course tests for dur.04, part 2: the worker SDK and pool (go/durable/worker)
// against the reference server. The workflow side is a scripted handler
// written here (the real SDK is dur.06): it schedules N activities in its
// first task and completes with their results once all are back.

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials/insecure"

	"tinyllm/durable/worker"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
	"openlearn.urmzd.com/tl/testkit/clock"
)

// scripted is a WorkflowHandler: input "N" schedules activities "0".."N-1"
// of type typ in the first task, then completes with their results joined.
type scripted struct {
	typ   string
	hbMs  int64
	tasks atomic.Int32
}

func (s *scripted) HandleWorkflowTask(_ context.Context, task *durablev1.WorkflowTask, h []*durablev1.HistoryEvent) ([]*durablev1.Command, error) {
	s.tasks.Add(1)
	n, _ := strconv.Atoi(string(h[0].GetStarted().GetInput()))
	results := map[int64]string{}
	var order []int64
	for _, ev := range h {
		if ev.GetActScheduled() != nil {
			order = append(order, ev.EventId)
		}
		if c := ev.GetActCompleted(); c != nil {
			results[c.ScheduledEventId] = string(c.Result)
		}
	}
	if len(order) == 0 {
		var cmds []*durablev1.Command
		for i := 0; i < n; i++ {
			c := schedule(strconv.Itoa(i), s.typ, []byte(strconv.Itoa(i)))
			c.GetScheduleActivity().Options.HeartbeatTimeoutMs = s.hbMs
			cmds = append(cmds, c)
		}
		return cmds, nil
	}
	if len(results) < len(order) {
		return nil, nil
	}
	var out []string
	for _, id := range order {
		out = append(out, results[id])
	}
	return []*durablev1.Command{completeCmd(strings.Join(out, ","))}, nil
}

func dial(t *testing.T, addr string) *grpc.ClientConn {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithConnectParams(grpc.ConnectParams{Backoff: backoff.Config{BaseDelay: 20 * time.Millisecond, Multiplier: 1.6, MaxDelay: 200 * time.Millisecond}}))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

// runWorker starts w.Run and returns a stop function that cancels it and
// waits (5 s guard) for Run to return.
func runWorker(t *testing.T, w *worker.Worker) (stop func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(bg)
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	var once sync.Once
	stop = func() {
		once.Do(func() {
			cancel()
			select {
			case err := <-done:
				if err != nil {
					t.Errorf("Run returned %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Errorf("Run did not return within 5 s of cancel")
			}
		})
	}
	t.Cleanup(stop)
	return stop
}

// eventually polls cond every 5 ms of real time for up to 5 s.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	deadline := time.After(5 * time.Second)
	for !cond() {
		select {
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func (e *env) status(id string) durablev1.WorkflowStatus {
	inf, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: id})
	if err != nil {
		return durablev1.WorkflowStatus_WORKFLOW_STATUS_UNSPECIFIED
	}
	return inf.Status
}

func TestReconnectDelaySchedule(t *testing.T) {
	// WHY: a restarting server must not be hammered: the delay doubles from
	//      the initial value per consecutive failure and stops at the cap.
	// KIND: unit
	// CATCHES: s15
	// CHAPTER: dur.04 section 2, reconnect backoff
	ms := time.Millisecond
	for n, want := range map[int]time.Duration{0: 0, 1: 100 * ms, 2: 200 * ms, 3: 400 * ms, 6: 3200 * ms, 7: 5000 * ms, 40: 5000 * ms} {
		if got := worker.ReconnectDelay(n, 100*ms, 5*time.Second); got != want {
			t.Fatalf("ReconnectDelay(%d) = %v; want %v", n, got, want)
		}
	}
}

func TestWorkerRunsWorkflowEndToEnd(t *testing.T) {
	// WHY: the whole protocol through your worker: workflow tasks go to the
	//      handler with their full history, activities get an Info whose
	//      idempotency key is "<workflow_id>/<activity_id>", results go back,
	//      and the run completes with them.
	// KIND: unit
	// CATCHES: s16, s17
	// CHAPTER: dur.04 section 4
	e := newEnv(t).start()
	h := &scripted{typ: "Shout"}
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default", Identity: "w1", Workflows: h})
	var mu sync.Mutex
	keys := map[string]int{}
	w.RegisterActivity("Shout", func(ctx context.Context, in []byte) ([]byte, error) {
		inf, ok := worker.InfoFrom(ctx)
		if !ok {
			return nil, errors.New("no Info in the activity context")
		}
		mu.Lock()
		keys[inf.IdempotencyKey] = inf.Attempt
		mu.Unlock()
		return []byte(string(in) + "!"), nil
	})
	runWorker(t, w)
	e.startWF("shout-1", "Echo", "3")
	eventually(t, "the run to complete", func() bool { return e.status("shout-1") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	if inf := e.describe("shout-1"); string(inf.Result) != "0!,1!,2!" {
		t.Fatalf("result %q; want 0!,1!,2!", inf.Result)
	}
	if fmt.Sprint(keys) != "map[shout-1/0:1 shout-1/1:1 shout-1/2:1]" {
		t.Fatalf("idempotency keys and attempts seen: %v", keys)
	}
	if h := e.mustHistory("shout-1"); h[len(h)-2].GetWtCompleted().GetIdentity() != "w1" || h[len(h)-3].GetWtStarted().GetIdentity() != "w1" {
		t.Fatalf("the worker's identity is not recorded: %v", h[len(h)-2])
	}
}

func TestWorkerFetchesEveryHistoryPage(t *testing.T) {
	// WHY: a long history arrives in pages; the handler must see all of it
	//      (the worker follows next_page_token), or replay would stop early.
	// KIND: unit
	// CATCHES: s18
	// CHAPTER: dur.04 section 2, history pages
	e := newEnv(t).start()
	h := &scripted{typ: "Big"}
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default", Workflows: h})
	w.RegisterActivity("Big", func(ctx context.Context, in []byte) ([]byte, error) {
		return []byte(strings.Repeat("y", 700<<10)), nil
	})
	runWorker(t, w)
	e.startWF("big", "Echo", "2") // two 700 KiB results: the last task's history is over 1 MiB
	eventually(t, "the run to complete", func() bool { return e.status("big") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
}

func TestWorkerBoundedConcurrency(t *testing.T) {
	// WHY: MaxActivities bounds the pool, and the worker takes a slot BEFORE
	//      it polls: with 2 slots and 6 ready tasks exactly 2 are leased. A
	//      worker that polls first would hold leases it cannot run, and they
	//      would expire into needless retries.
	// KIND: unit, property
	// CATCHES: s19, s20
	// CHAPTER: dur.04 section 2, bounded pool
	e := newEnv(t).start()
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default", MaxActivities: 2, Workflows: &scripted{typ: "Block"}})
	release := make(chan struct{})
	var running, peak atomic.Int32
	w.RegisterActivity("Block", func(ctx context.Context, in []byte) ([]byte, error) {
		n := running.Add(1)
		for p := peak.Load(); n > p && !peak.CompareAndSwap(p, n); p = peak.Load() {
		}
		defer running.Add(-1)
		<-release
		return in, nil
	})
	runWorker(t, w)
	e.startWF("pool", "Echo", "6")
	eventually(t, "two activities running", func() bool { return running.Load() == 2 })
	eventually(t, "the third poll to be waiting", func() bool { v, l, _ := e.q.Depth("act:default"); return v == 4 && l == 2 })
	// Hold the state for a while: a pool one slot too big reaches 2 running
	// on its way to 3, so a single observation of 2 proves nothing.
	for deadline := time.Now().Add(300 * time.Millisecond); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
		if v, l, _ := e.q.Depth("act:default"); v != 4 || l != 2 || running.Load() != 2 {
			t.Fatalf("with 2 slots busy: visible %d leased %d running %d; want 4, 2, 2", v, l, running.Load())
		}
	}
	close(release)
	eventually(t, "the run to complete", func() bool { return e.status("pool") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	if peak.Load() != 2 {
		t.Fatalf("peak concurrency %d; want 2", peak.Load())
	}
}

func TestWorkerDrainsOnCancel(t *testing.T) {
	// WHY: SIGTERM (Run's ctx ending) stops polling at once but lets the
	//      running activity finish and report, so a rolling deploy loses no
	//      work and redoes none; Run returns only after that report.
	// KIND: fault
	// CATCHES: s21, s22
	// CHAPTER: dur.04 section 2, drain
	e := newEnv(t).start()
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default", Workflows: &scripted{typ: "Slow"}})
	started, release := make(chan struct{}, 1), make(chan struct{})
	var sawCancel atomic.Bool
	w.RegisterActivity("Slow", func(ctx context.Context, in []byte) ([]byte, error) {
		started <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
			sawCancel.Store(true)
		}
		return []byte("done"), nil
	})
	stop := runWorker(t, w)
	e.startWF("drain", "Echo", "1")
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the activity never started")
	}
	stopped := make(chan struct{})
	go func() { stop(); close(stopped) }()
	eventually(t, "every poll to end", func() bool { return e.polls.Load() == 0 })
	e.startWF("after", "Echo", "1")
	select {
	case <-stopped:
		t.Fatal("Run returned while an activity was still running")
	case <-time.After(50 * time.Millisecond):
	}
	if v, _, _ := e.q.Depth("wf:default"); v != 1 || e.polls.Load() != 0 {
		t.Fatalf("a draining worker polled again (visible %d, polls %d)", v, e.polls.Load())
	}
	close(release)
	<-stopped
	if sawCancel.Load() {
		t.Fatal("the in-flight activity's context was canceled by the drain")
	}
	h := e.mustHistory("drain")
	if !strings.Contains(kinds(h), "act_completed") {
		t.Fatalf("the drained activity's result was not reported: %q", kinds(h))
	}
	if e.status("after") != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || len(e.mustHistory("after")) != 2 {
		t.Fatal("a stopped worker took a new workflow task")
	}
}

func TestMissedHeartbeatCancelsActivity(t *testing.T) {
	// WHY: an activity that stops heartbeating has lost its lease: the server
	//      will hand it to another worker after the heartbeat timeout. The
	//      worker must cancel its context with ErrHeartbeatTimeout and report
	//      nothing, so two copies do not both write.
	// KIND: fault
	// CATCHES: s23, s24
	// CHAPTER: dur.04 section 5, Pitfalls
	e := newEnv(t).start()
	wclk := clock.NewFake(t0)
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default", Clock: wclk, Workflows: &scripted{typ: "Stuck", hbMs: 2000}})
	cause := make(chan error, 1)
	started := make(chan struct{}, 1)
	w.RegisterActivity("Stuck", func(ctx context.Context, in []byte) ([]byte, error) {
		started <- struct{}{}
		<-ctx.Done()
		cause <- context.Cause(ctx)
		return []byte("too late"), nil
	})
	stop := runWorker(t, w)
	e.startWF("hb", "Echo", "1")
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the activity never started")
	}
	wait := make(chan struct{})
	go func() { wclk.BlockUntil(1); close(wait) }()
	select {
	case <-wait:
	case <-time.After(5 * time.Second):
		t.Fatal("no heartbeat watchdog is waiting on the worker's clock")
	}
	wclk.Advance(1999 * time.Millisecond)
	select {
	case c := <-cause:
		t.Fatalf("canceled before the heartbeat timeout: %v", c)
	case <-time.After(20 * time.Millisecond):
	}
	wclk.Advance(time.Millisecond)
	select {
	case c := <-cause:
		if !errors.Is(c, worker.ErrHeartbeatTimeout) {
			t.Fatalf("cancel cause %v; want ErrHeartbeatTimeout", c)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the activity context was not canceled after the heartbeat timeout")
	}
	stop() // Run returns after the activity goroutine is done: any report has been sent
	if h := e.mustHistory("hb"); strings.Contains(kinds(h), "act_completed") || strings.Contains(kinds(h), "act_failed") {
		t.Fatalf("the worker reported an attempt whose lease it lost: %q", kinds(h))
	}
}

func TestWorkerReconnectsAfterServerRestart(t *testing.T) {
	// WHY: the durable server restarts (deploy, crash, drill ops.02); workers
	//      see UNAVAILABLE, back off, and carry on against the new process
	//      without being restarted themselves.
	// KIND: fault
	// CATCHES: s25
	// CHAPTER: dur.04 section 2, reconnect backoff
	e := newEnv(t).start()
	w := worker.New(dial(t, e.addr), worker.Options{
		TaskQueue: "default", Workflows: &scripted{typ: "Echo"},
		ReconnectInitial: 10 * time.Millisecond, ReconnectMax: 50 * time.Millisecond,
	})
	w.RegisterActivity("Echo", func(ctx context.Context, in []byte) ([]byte, error) { return in, nil })
	runWorker(t, w)
	e.startWF("before", "Echo", "1")
	eventually(t, "the first run", func() bool { return e.status("before") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	e.crash()
	<-time.After(200 * time.Millisecond) // the worker fails its polls and backs off
	e.start()
	e.startWF("after", "Echo", "2")
	eventually(t, "a run after the restart", func() bool { return e.status("after") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	if got := string(e.describe("after").Result); got != "0,1" {
		t.Fatalf("result after restart %q", got)
	}
}
