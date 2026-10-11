// Course tests for dur.08, end to end: a real worker runs Runtime workflows
// through the dur.06 SDK and FromContext (go/workflows/runtime.go) against
// the server, so the signal hooks, the cancel path, and CancelRun are
// exercised as the platform runs them.
package dur_08

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials/insecure"

	dact "tinyllm/durable/activity"
	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"
	"tinyllm/workflows"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

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

// runWorker starts w.Run until the test ends.
func runWorker(t *testing.T, w *worker.Worker) {
	t.Helper()
	ctx, cancel := context.WithCancel(bg)
	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Errorf("worker.Run did not return after its context ended")
		}
	})
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func (e *env) status(id string) durablev1.WorkflowStatus {
	inf, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: id})
	if err != nil {
		return durablev1.WorkflowStatus_WORKFLOW_STATUS_UNSPECIFIED
	}
	return inf.Status
}

// gate is an activity that blocks until released or cancelled.
type gate struct {
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func newGate() *gate { return &gate{started: make(chan struct{}), release: make(chan struct{})} }

func (g *gate) activity(ctx context.Context, in []byte) ([]byte, error) {
	g.once.Do(func() { close(g.started) })
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-g.release:
			return []byte(`"prepared"`), nil
		case <-ctx.Done():
			return nil, dact.NewNonRetryable("Canceled", "the activity was cancelled")
		case <-tick.C:
			worker.Heartbeat(ctx, nil) // a cancel arrives as the heartbeat's answer
		}
	}
}

type approvalResult struct {
	Decision string `json:"decision"`
	By       string `json:"by"`
}

func approval(rt workflows.Runtime, in string) (approvalResult, error) {
	if err := rt.ExecuteActivity("prepare", in, workflows.StepOptions{StartToClose: time.Minute, HeartbeatTimeout: 10 * time.Second}, nil); err != nil {
		return approvalResult{}, err
	}
	name, payload, err := rt.AwaitSignal([]string{"approve", "reject"}, 30*time.Second)
	if err != nil {
		return approvalResult{}, err
	}
	if name == "" {
		return approvalResult{Decision: "expired"}, nil
	}
	return approvalResult{Decision: name, By: string(payload)}, nil
}

func newWorker(t *testing.T, e *env, reg *workflow.Registry) *worker.Worker {
	return worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default", Identity: "w1", Workflows: reg, MaxActivities: 4})
}

func TestEndToEndSignalBeforeWait(t *testing.T) {
	// WHY: the catalog's promise, with a real worker: the approver signals
	//      while the workflow is still busy in its prepare activity, long
	//      before it waits on the channel. The signal is buffered in history
	//      and the workflow, when it reaches AwaitSignal, gets it at once.
	// KIND: fault
	// CATCHES: s23, s24
	// CHAPTER: dur.08 section 2.1
	e := newEnv(t).start()
	reg := workflow.NewRegistry()
	reg.Register("Approval", workflows.Workflow(approval))
	w := newWorker(t, e, reg)
	g := newGate()
	w.RegisterActivity("prepare", g.activity)
	runWorker(t, w)
	e.startWF("rel-1", "Approval", `"v1"`)
	<-g.started
	if err := e.signal("rel-1", "approve", "ana", "r1"); err != nil {
		t.Fatal(err)
	}
	close(g.release)
	eventually(t, "the run to complete", func() bool { return e.status("rel-1") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	if res := string(e.describe("rel-1").GetResult()); res != `{"decision":"approve","by":"ana"}` {
		t.Fatalf("result %s", res)
	}
}

func TestEndToEndSignalTimeout(t *testing.T) {
	// WHY: AwaitSignal with a timeout is a durable timer raced against the
	//      channel: with no signal, the timer fires (the fake clock passes 30
	//      s) and the workflow goes on; the timer is a real StartTimer the
	//      server recorded, so a restart in between would not restart it.
	// KIND: unit
	// CATCHES: s25
	// CHAPTER: dur.08 section 2.4
	e := newEnv(t).start()
	reg := workflow.NewRegistry()
	reg.Register("Approval", workflows.Workflow(approval))
	w := newWorker(t, e, reg)
	g := newGate()
	close(g.release)
	w.RegisterActivity("prepare", g.activity)
	runWorker(t, w)
	e.startWF("rel-2", "Approval", `"v1"`)
	eventually(t, "the approval timer", func() bool { return e.ts.Len() == 1 })
	e.clk.Advance(31 * time.Second)
	e.ts.Wake()
	eventually(t, "the run to complete", func() bool { return e.status("rel-2") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	if res := string(e.describe("rel-2").GetResult()); res != `{"decision":"expired","by":""}` {
		t.Fatalf("result %s", res)
	}
}

func book(rt workflows.Runtime, in string) (string, error) {
	var saga workflows.Saga
	saga.Add("release the GPU", "gpu.release", in, workflows.StepOptions{StartToClose: time.Minute})
	if err := rt.ExecuteActivity("prepare", in, workflows.StepOptions{StartToClose: time.Minute}, nil); err != nil {
		return "", saga.Fail(rt, err)
	}
	return "booked", nil
}

// stuck is an activity that never heartbeats and ignores cancellation (a
// C kernel that does not check a flag): the workflow must stop waiting for
// it on its own.
func (g *gate) stuck(ctx context.Context, in []byte) ([]byte, error) {
	g.once.Do(func() { close(g.started) })
	<-g.release
	return []byte(`"late"`), nil
}

func TestEndToEndCancelMidActivityCompensatesOnce(t *testing.T) {
	// WHY: the catalog's promise: cancel while an activity runs. Here the
	//      activity never heartbeats, so it never hears the cancel: the
	//      workflow must stop waiting for it on its own (the cancel request is
	//      in its history), run its compensation once on the detached Runtime,
	//      and end the run CANCELED through CancelWorkflowExecution, not
	//      FAILED.
	// KIND: fault
	// CATCHES: s26, s27, s28
	// CHAPTER: dur.08 section 3, worked example
	e := newEnv(t).start()
	reg := workflow.NewRegistry()
	reg.Register("Book", workflows.Workflow(book))
	w := newWorker(t, e, reg)
	g := newGate()
	defer close(g.release)
	w.RegisterActivity("prepare", g.stuck)
	var mu sync.Mutex
	released := map[string]int{}
	w.RegisterActivity("gpu.release", func(ctx context.Context, in []byte) ([]byte, error) {
		inf, _ := worker.InfoFrom(ctx)
		mu.Lock()
		released[inf.IdempotencyKey]++
		mu.Unlock()
		return nil, nil
	})
	runWorker(t, w)
	e.startWF("book-1", "Book", `"job-9"`)
	<-g.started
	if err := e.cancel("book-1"); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the run to end canceled", func() bool { return e.status("book-1") == durablev1.WorkflowStatus_WORKFLOW_STATUS_CANCELED })
	mu.Lock()
	defer mu.Unlock()
	if len(released) != 1 || released["book-1/compensate-1"] != 1 {
		t.Fatalf("compensations: %v; want book-1/compensate-1 once", released)
	}
	h := e.mustHistory("book-1")
	if kindOf(h[len(h)-1]) != "canceled" || !strings.Contains(string(h[len(h)-1].GetCanceled().GetDetails()), "canceled") {
		t.Fatalf("the run must end with WorkflowExecutionCanceled: %s", kinds(h))
	}
	if count(h, "act_cancel_requested") != 1 {
		t.Fatalf("the running activity is asked to stop once: %s", kinds(h))
	}
}

func TestEndToEndSignalsInOrder(t *testing.T) {
	// WHY: signals of one name are a queue: two approvals sent before the
	//      workflow waits are received one per AwaitSignal, oldest first, and
	//      each exactly once.
	// KIND: unit
	// CATCHES: s24
	// CHAPTER: dur.08 section 2.1
	e := newEnv(t).start()
	reg := workflow.NewRegistry()
	reg.Register("Two", workflows.Workflow(func(rt workflows.Runtime, in string) ([]string, error) {
		if err := rt.ExecuteActivity("prepare", in, workflows.StepOptions{StartToClose: time.Minute}, nil); err != nil {
			return nil, err
		}
		var got []string
		for i := 0; i < 2; i++ {
			_, p, err := rt.AwaitSignal([]string{"approve"}, 0)
			if err != nil {
				return nil, err
			}
			got = append(got, string(p))
		}
		return got, nil
	}))
	w := newWorker(t, e, reg)
	g := newGate()
	w.RegisterActivity("prepare", g.activity)
	runWorker(t, w)
	e.startWF("two", "Two", `""`)
	<-g.started
	e.signal("two", "approve", "first", "")
	e.signal("two", "approve", "second", "")
	close(g.release)
	eventually(t, "the run to complete", func() bool { return e.status("two") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	if res := string(e.describe("two").GetResult()); res != `["first","second"]` {
		t.Fatalf("received %s", res)
	}
}

func TestExecuteActivityIDKeys(t *testing.T) {
	// WHY: a named activity id is the idempotency key's second half
	//      ("<workflow id>/<id>"): CorpusBuild keys its stages by name so a
	//      code change that adds a step cannot shift every later key. Ids
	//      must be unique and not numbers (those are ExecuteActivity's).
	// KIND: unit
	// CATCHES: s29
	// CHAPTER: dur.08 section 4, The interface
	e := newEnv(t).start()
	reg := workflow.NewRegistry()
	reg.Register("Named", workflows.Workflow(func(rt workflows.Runtime, in string) ([]string, error) {
		var out []string
		for _, id := range []string{"fetch", "shard"} {
			var key string
			if err := rt.ExecuteActivity("key", nil, workflows.StepOptions{ID: id, StartToClose: time.Minute}, &key); err != nil {
				return nil, err
			}
			out = append(out, key)
		}
		err := rt.ExecuteActivity("key", nil, workflows.StepOptions{ID: "shard", StartToClose: time.Minute}, nil)
		var sf *workflows.StepFailure
		if !errors.As(err, &sf) || sf.Type != "BadActivityID" {
			return nil, errors.New("a repeated activity id must fail at once")
		}
		return out, nil
	}))
	w := newWorker(t, e, reg)
	w.RegisterActivity("key", func(ctx context.Context, in []byte) ([]byte, error) {
		inf, _ := worker.InfoFrom(ctx)
		return []byte(`"` + inf.IdempotencyKey + `"`), nil
	})
	runWorker(t, w)
	e.startWF("cb-1", "Named", `""`)
	eventually(t, "the run to complete", func() bool { return e.status("cb-1") == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED })
	if res := string(e.describe("cb-1").GetResult()); res != `["cb-1/fetch","cb-1/shard"]` {
		t.Fatalf("keys %s", res)
	}
}
