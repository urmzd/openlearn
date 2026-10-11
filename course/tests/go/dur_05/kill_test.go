package dur_05

// The flaky-activity kill loop: the test binary re-executes itself as a
// worker process (TestMain), the parent SIGKILLs it at seeded offsets and
// restarts the in-process server every third kill. Each activity posts to the
// effects sink with its idempotency key, and fails before or after its effect
// on a fixed fraction of attempts. However the kills land, every key must be
// applied exactly once.

import (
	"context"
	"fmt"
	"hash/fnv"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials/insecure"

	"tinyllm/durable/activity"
	"tinyllm/durable/worker"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
	"openlearn.urmzd.com/tl/testkit/effects"
	"openlearn.urmzd.com/tl/testkit/proc"
)

func TestMain(m *testing.M) {
	if addr := os.Getenv("DUR05_ADDR"); addr != "" {
		os.Exit(childWorker(addr, os.Getenv("DUR05_SINK"), os.Getenv("DUR05_WF")))
	}
	os.Exit(m.Run())
}

// batch is the workflow side: input N schedules N "tl.test.Append"
// activities with a fast retry policy, then completes with their results.
type batch struct{}

func (batch) HandleWorkflowTask(_ context.Context, _ *durablev1.WorkflowTask, h []*durablev1.HistoryEvent) ([]*durablev1.Command, error) {
	n, _ := strconv.Atoi(string(h[0].GetStarted().GetInput()))
	done := map[int64]string{}
	var order []int64
	for _, ev := range h {
		if ev.GetActScheduled() != nil {
			order = append(order, ev.EventId)
		}
		if c := ev.GetActCompleted(); c != nil {
			done[c.ScheduledEventId] = string(c.Result)
		}
	}
	if len(order) == 0 {
		var cmds []*durablev1.Command
		for i := 0; i < n; i++ {
			cmds = append(cmds, &durablev1.Command{Cmd: &durablev1.Command_ScheduleActivity{ScheduleActivity: &durablev1.ScheduleActivity{
				ActivityId: strconv.Itoa(i), ActivityType: "tl.test.Append", Input: []byte(strconv.Itoa(i)),
				Options: &durablev1.ActivityOptions{StartToCloseMs: 400,
					Retry: &durablev1.RetryPolicy{InitialMs: 10, Backoff: 2, MaxIntervalMs: 50, MaxAttempts: 100}},
			}}})
		}
		return cmds, nil
	}
	if len(done) < len(order) {
		return nil, nil
	}
	var out []string
	for _, id := range order {
		out = append(out, done[id])
	}
	return []*durablev1.Command{{Cmd: &durablev1.Command_Complete{Complete: &durablev1.CompleteWorkflow{Result: []byte(strings.Join(out, ","))}}}}, nil
}

// fate decides, from the key and the attempt alone, what this attempt does:
// 0 fail before the effect, 1 apply the effect and then fail, else succeed.
func fate(key string, attempt int) uint32 {
	h := fnv.New32a()
	fmt.Fprintf(h, "%s#%d", key, attempt)
	return h.Sum32() % 4
}

func appendActivity(sink string) worker.ActivityFunc {
	return func(ctx context.Context, in []byte) ([]byte, error) {
		key := activity.IdempotencyKey(ctx)
		f := fate(key, activity.Attempt(ctx))
		if f == 0 {
			return nil, activity.NewError("Flaky", "before the effect")
		}
		resp, err := http.Post(sink, "application/json", strings.NewReader(fmt.Sprintf(`{"key":%q,"value":%q}`, key, in)))
		if err != nil {
			return nil, err
		}
		resp.Body.Close()
		if f == 1 {
			return nil, activity.NewError("Flaky", "after the effect")
		}
		return in, nil
	}
}

func childWorker(addr, sink, wf string) int {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithConnectParams(grpc.ConnectParams{Backoff: backoff.Config{BaseDelay: 20 * time.Millisecond, Multiplier: 1.6, MaxDelay: 200 * time.Millisecond}}))
	if err != nil {
		fmt.Println("dial:", err)
		return 3
	}
	w := worker.New(conn, worker.Options{TaskQueue: "default", Workflows: batch{}, MaxActivities: 4,
		ReconnectInitial: 10 * time.Millisecond, ReconnectMax: 100 * time.Millisecond, DrainTimeout: time.Second})
	w.RegisterActivity("tl.test.Append", appendActivity(sink))
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		wfs := durablev1.NewWorkflowServiceClient(conn)
		for {
			inf, err := wfs.DescribeWorkflow(ctx, &durablev1.DescribeWorkflowRequest{WorkflowId: wf})
			if err == nil && inf.Status == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED {
				cancel()
				return
			}
			select {
			case <-time.After(20 * time.Millisecond):
			case <-ctx.Done():
				return
			}
		}
	}()
	w.Run(ctx)
	return 0
}

func dial(t *testing.T, addr string) *grpc.ClientConn {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

func runWorker(t *testing.T, w *worker.Worker) {
	ctx, cancel := context.WithCancel(bg)
	done := make(chan struct{})
	go func() { w.Run(ctx); close(done) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Errorf("worker did not stop")
		}
	})
}

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

func TestFlakyActivityKillLoopExactlyOnce(t *testing.T) {
	// WHY: the module's promise under real crashes: 20 flaky activities, the
	//      worker SIGKILLed 8 times at seeded instants and the server restarted
	//      every third kill. The run completes with every result, and the
	//      effects sink saw each "<workflow_id>/<activity_id>" key applied
	//      exactly once, though many were delivered more than once.
	// KIND: fault
	// CATCHES: s12
	// CHAPTER: dur.05 section 4, kill loop
	sink, err := effects.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer sink.Close()
	e := newEnv(t)
	e.real, e.pollWait, e.wtLease = true, time.Second, 300*time.Millisecond
	e.start()
	e.startWF("flaky", "Batch", "20")
	var mu sync.Mutex
	seed, _ := strconv.ParseUint(os.Getenv("OL_SEED"), 10, 64)
	res, err := proc.KillLoop(func() *exec.Cmd {
		cmd := exec.Command(os.Args[0], "-test.run=^$")
		cmd.Env = append(os.Environ(), "DUR05_ADDR="+e.addr, "DUR05_SINK="+sink.URL(), "DUR05_WF=flaky")
		return cmd
	}, proc.Options{Kills: 8, MinUp: 150 * time.Millisecond, Jitter: 250 * time.Millisecond, Seed: seed, Finish: 30 * time.Second,
		OnRestart: func(i int) {
			if i%3 == 0 {
				mu.Lock()
				e.restart()
				mu.Unlock()
			}
		}}, nil)
	if err != nil {
		t.Fatalf("kill loop: %v\n%s", err, res.Output)
	}
	inf := e.describe("flaky")
	if inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED {
		t.Fatalf("run status %v", inf.Status)
	}
	var keys, want []string
	for i := 0; i < 20; i++ {
		keys = append(keys, fmt.Sprintf("flaky/%d", i))
		want = append(want, strconv.Itoa(i))
	}
	if string(inf.Result) != strings.Join(want, ",") {
		t.Fatalf("result %q", inf.Result)
	}
	sink.AssertExactlyOnce(t, keys)
	redelivered := 0
	for _, r := range sink.Records() {
		if r.Deliveries > 1 {
			redelivered++
		}
	}
	if redelivered == 0 {
		t.Fatal("no effect was ever delivered twice: the loop did not exercise retries")
	}
}
