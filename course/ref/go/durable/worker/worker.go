// Package worker is the Go worker SDK of the durable engine (dur.04): it
// long-polls the server's TaskService, runs activities in a bounded pool,
// hands workflow tasks to a WorkflowHandler (dur.06's workflow.Registry),
// drains on shutdown, cancels an activity whose lease it can no longer keep,
// and reconnects with exponential backoff when the server goes away.
package worker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// ActivityFunc runs one attempt of an activity. Payloads are opaque bytes
// (the workflow SDK uses JSON).
type ActivityFunc func(ctx context.Context, input []byte) ([]byte, error)

// WorkflowHandler turns a workflow task (its complete history, every page
// fetched) into commands. An error means "do not complete": the task times
// out and is retried, which keeps a nondeterministic run visibly stuck.
type WorkflowHandler interface {
	HandleWorkflowTask(ctx context.Context, task *durablev1.WorkflowTask, history []*durablev1.HistoryEvent) ([]*durablev1.Command, error)
}

// Clock is the time seam; the testkit's *clock.Fake satisfies it.
type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

type wall struct{}

func (wall) Now() time.Time                         { return time.Now() }
func (wall) After(d time.Duration) <-chan time.Time { return time.After(d) }

// Options configures a Worker.
type Options struct {
	TaskQueue        string          // required
	Identity         string          // "" = hostname:pid
	MaxActivities    int             // concurrent activities; 0 = 4
	Workflows        WorkflowHandler // nil: run activities only
	Clock            Clock           // heartbeat watchdog and backoff; nil = wall clock
	ReconnectInitial time.Duration   // first backoff after a failed poll; 0 = 100 ms
	ReconnectMax     time.Duration   // backoff cap; 0 = 5 s
	DrainTimeout     time.Duration   // after Run's ctx ends, how long in-flight activities may finish; 0 = 30 s
	Logf             func(format string, args ...any)
	// Failpoint, when set, is evaluated as "dur/activity/poison/<activity id>"
	// just before each activity runs, inside the panic recovery: the poison-task
	// drill (ops.03) makes one activity panic with TL_FAILPOINTS.
	Failpoint func(name string) error
}

var (
	// ErrHeartbeatTimeout is the cause of an activity context that missed
	// its heartbeat timeout: the server has given the task to someone else.
	ErrHeartbeatTimeout = errors.New("worker: heartbeat timeout")
	// ErrLeaseLost: the server refused a heartbeat with a stale lease.
	ErrLeaseLost = errors.New("worker: activity lease lost")
	// ErrCancelRequested: a heartbeat came back with cancel_requested.
	ErrCancelRequested = errors.New("worker: activity cancel requested")
	// ErrNotActivity: Heartbeat outside an activity.
	ErrNotActivity = errors.New("worker: not running inside an activity")
)

// Info describes the activity attempt a context belongs to.
type Info struct {
	WorkflowID, RunID, ActivityID, ActivityType string
	IdempotencyKey                              string // "<workflow_id>/<activity_id>", the same on every attempt
	Attempt                                     int
	HeartbeatDetails                            []byte // the last details recorded by an earlier attempt
	TraceContext                                map[string]string
	Deadline                                    time.Time
	heartbeat                                   func([]byte) error
}

type infoKey struct{}

// InfoFrom returns the activity Info stored in ctx.
func InfoFrom(ctx context.Context) (Info, bool) {
	inf, ok := ctx.Value(infoKey{}).(*Info)
	if !ok {
		return Info{}, false
	}
	return *inf, true
}

// Heartbeat records progress for the running activity and extends its lease.
func Heartbeat(ctx context.Context, details []byte) error {
	// SOLUTION-BEGIN dur.04
	inf, ok := ctx.Value(infoKey{}).(*Info)
	if !ok || inf.heartbeat == nil {
		return ErrNotActivity
	}
	return inf.heartbeat(details)
	// SOLUTION-END
}

// ReconnectDelay is the backoff before poll attempt n+1 after n consecutive
// failures: min(initial * 2^(n-1), max).
func ReconnectDelay(n int, initial, max time.Duration) time.Duration {
	// SOLUTION-BEGIN dur.04
	if n < 1 {
		return 0
	}
	d := initial
	for i := 1; i < n && d < max; i++ {
		d *= 2
	}
	if d > max {
		d = max
	}
	return d
	// SOLUTION-END
}

// Worker polls one task queue. Register activities before Run.
type Worker struct {
	tasks durablev1.TaskServiceClient
	wfs   durablev1.WorkflowServiceClient
	o     Options

	mu   sync.Mutex
	acts map[string]ActivityFunc
}

// New makes a worker over a connection to the durable server.
func New(conn grpc.ClientConnInterface, o Options) *Worker {
	// SOLUTION-BEGIN dur.04
	if o.Identity == "" {
		host, _ := os.Hostname()
		o.Identity = fmt.Sprintf("%s:%d", host, os.Getpid())
	}
	if o.MaxActivities <= 0 {
		o.MaxActivities = 4
	}
	if o.Clock == nil {
		o.Clock = wall{}
	}
	if o.ReconnectInitial <= 0 {
		o.ReconnectInitial = 100 * time.Millisecond
	}
	if o.ReconnectMax <= 0 {
		o.ReconnectMax = 5 * time.Second
	}
	if o.DrainTimeout <= 0 {
		o.DrainTimeout = 30 * time.Second
	}
	if o.Logf == nil {
		o.Logf = func(string, ...any) {}
	}
	return &Worker{
		tasks: durablev1.NewTaskServiceClient(conn), wfs: durablev1.NewWorkflowServiceClient(conn),
		o: o, acts: map[string]ActivityFunc{},
	}
	// SOLUTION-END
}

// RegisterActivity makes fn the implementation of activity type name.
func (w *Worker) RegisterActivity(name string, fn ActivityFunc) {
	// SOLUTION-BEGIN dur.04
	w.mu.Lock()
	defer w.mu.Unlock()
	w.acts[name] = fn
	// SOLUTION-END
}

// Run polls until ctx ends (SIGTERM in the entry point). Then it stops
// polling at once, lets in-flight activities finish for up to DrainTimeout
// (their contexts are not canceled by ctx), cancels the rest, and returns nil.
func (w *Worker) Run(ctx context.Context) error {
	// SOLUTION-BEGIN dur.04
	if w.o.TaskQueue == "" {
		return errors.New("worker: Options.TaskQueue is required")
	}
	work, cancelWork := context.WithCancel(context.WithoutCancel(ctx))
	defer cancelWork()
	var pollers, inflight sync.WaitGroup
	slots := make(chan struct{}, w.o.MaxActivities)
	pollers.Add(1)
	go func() {
		defer pollers.Done()
		w.pollActivities(ctx, work, slots, &inflight)
	}()
	if w.o.Workflows != nil {
		pollers.Add(1)
		go func() {
			defer pollers.Done()
			w.pollWorkflows(ctx, work)
		}()
	}
	<-ctx.Done()
	pollers.Wait()
	done := make(chan struct{})
	go func() { inflight.Wait(); close(done) }()
	select {
	case <-done:
	case <-w.o.Clock.After(w.o.DrainTimeout):
		cancelWork()
		<-done
	}
	return nil
	// SOLUTION-END
}

// backoff waits ReconnectDelay(n) or until ctx ends; false when ctx ended.
func (w *Worker) backoff(ctx context.Context, n int) bool {
	// SOLUTION-BEGIN dur.04
	select {
	case <-w.o.Clock.After(ReconnectDelay(n, w.o.ReconnectInitial, w.o.ReconnectMax)):
		return true
	case <-ctx.Done():
		return false
	}
	// SOLUTION-END
}

// pollActivities takes a pool slot before every poll, so the worker never
// leases a task it has no capacity to run.
func (w *Worker) pollActivities(ctx, work context.Context, slots chan struct{}, inflight *sync.WaitGroup) {
	// SOLUTION-BEGIN dur.04
	failures := 0
	for {
		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
			return
		}
		t, err := w.tasks.PollActivityTask(ctx, &durablev1.PollRequest{TaskQueue: w.o.TaskQueue, Identity: w.o.Identity})
		if err != nil {
			<-slots
			if ctx.Err() != nil {
				return
			}
			failures++
			w.o.Logf("poll activity: %v (retry %d)", err, failures)
			if !w.backoff(ctx, failures) {
				return
			}
			continue
		}
		failures = 0
		if len(t.GetTaskToken()) == 0 {
			<-slots
			continue
		}
		inflight.Add(1)
		go func() {
			defer inflight.Done()
			defer func() { <-slots }()
			w.runActivity(work, t)
		}()
	}
	// SOLUTION-END
}

// runActivity runs one attempt and reports it, unless the lease was lost.
func (w *Worker) runActivity(work context.Context, t *durablev1.ActivityTask) {
	// SOLUTION-BEGIN dur.04
	w.mu.Lock()
	fn := w.acts[t.GetActivityType()]
	w.mu.Unlock()
	if fn == nil {
		w.fail(work, t, &durablev1.Failure{Message: "activity type " + t.GetActivityType() + " is not registered on this worker", Type: "ActivityNotRegistered"})
		return
	}
	ctx, cancel := context.WithCancelCause(work)
	defer cancel(nil)
	beat := make(chan struct{}, 1)
	inf := &Info{
		WorkflowID: t.GetWorkflowId(), RunID: t.GetRunId(), ActivityID: t.GetActivityId(), ActivityType: t.GetActivityType(),
		IdempotencyKey: t.GetIdempotencyKey(), Attempt: int(t.GetAttempt()), HeartbeatDetails: t.GetLastHeartbeatDetails(),
		TraceContext: t.GetTraceContext(), Deadline: time.UnixMilli(t.GetDeadlineUnixMs()),
	}
	inf.heartbeat = func(details []byte) error {
		resp, err := w.tasks.RecordHeartbeat(ctx, &durablev1.HeartbeatRequest{TaskToken: t.GetTaskToken(), LeaseToken: t.GetLeaseToken(), Details: details})
		if status.Code(err) == codes.FailedPrecondition {
			cancel(ErrLeaseLost)
			return ErrLeaseLost
		}
		if err != nil {
			return err
		}
		if resp.GetCancelRequested() {
			cancel(ErrCancelRequested)
		}
		select {
		case beat <- struct{}{}:
		default:
		}
		return nil
	}
	if hb := time.Duration(t.GetHeartbeatTimeoutMs()) * time.Millisecond; hb > 0 {
		go w.watchdog(ctx, cancel, beat, hb)
	}
	res, err := call(context.WithValue(ctx, infoKey{}, inf), w.poisoned(t.GetActivityId(), fn), t.GetInput())
	if cause := context.Cause(ctx); errors.Is(cause, ErrHeartbeatTimeout) || errors.Is(cause, ErrLeaseLost) {
		w.o.Logf("activity %s/%s: %v; not reporting", t.GetWorkflowId(), t.GetActivityId(), cause)
		return
	}
	if err != nil {
		w.fail(work, t, failureOf(err))
		return
	}
	w.retry(work, func(ctx context.Context) error {
		_, err := w.tasks.CompleteActivityTask(ctx, &durablev1.CompleteActivityRequest{
			TaskToken: t.GetTaskToken(), LeaseToken: t.GetLeaseToken(), Result: res, Identity: w.o.Identity,
		})
		return err
	})
	// SOLUTION-END
}

// watchdog cancels the activity when hb passes without a heartbeat.
func (w *Worker) watchdog(ctx context.Context, cancel context.CancelCauseFunc, beat <-chan struct{}, hb time.Duration) {
	// SOLUTION-BEGIN dur.04
	for {
		select {
		case <-beat:
		case <-w.o.Clock.After(hb):
			cancel(ErrHeartbeatTimeout)
			return
		case <-ctx.Done():
			return
		}
	}
	// SOLUTION-END
}

// poisoned wraps fn with the activity failpoint, when one is configured.
func (w *Worker) poisoned(activityID string, fn ActivityFunc) ActivityFunc {
	if w.o.Failpoint == nil {
		return fn
	}
	return func(ctx context.Context, in []byte) ([]byte, error) {
		if err := w.o.Failpoint("dur/activity/poison/" + activityID); err != nil {
			return nil, err
		}
		return fn(ctx, in)
	}
}

// call runs fn, turning a panic into an error of type "Panic".
func call(ctx context.Context, fn ActivityFunc, in []byte) (out []byte, err error) {
	// SOLUTION-BEGIN dur.04
	defer func() {
		if r := recover(); r != nil {
			err = &panicError{msg: fmt.Sprint(r), stack: string(debug.Stack())}
		}
	}()
	return fn(ctx, in)
	// SOLUTION-END
}

type panicError struct{ msg, stack string }

func (e *panicError) Error() string       { return "panic: " + e.msg }
func (e *panicError) FailureType() string { return "Panic" }

// failureOf maps an activity error to a Failure. An error (or one it wraps)
// with a NonRetryable() bool method that returns true fails for good; a
// FailureType() string method names the type matched against
// RetryPolicy.non_retryable (dur.05 builds such errors).
func failureOf(err error) *durablev1.Failure {
	// SOLUTION-BEGIN dur.04
	f := &durablev1.Failure{Message: err.Error(), Type: "Error"}
	var typed interface{ FailureType() string }
	if errors.As(err, &typed) {
		f.Type = typed.FailureType()
	}
	var nr interface{ NonRetryable() bool }
	if errors.As(err, &nr) {
		f.NonRetryable = nr.NonRetryable()
	}
	var pe *panicError
	if errors.As(err, &pe) {
		f.Stack = pe.stack
	}
	return f
	// SOLUTION-END
}

func (w *Worker) fail(work context.Context, t *durablev1.ActivityTask, f *durablev1.Failure) {
	// SOLUTION-BEGIN dur.04
	w.retry(work, func(ctx context.Context) error {
		_, err := w.tasks.FailActivityTask(ctx, &durablev1.FailActivityRequest{
			TaskToken: t.GetTaskToken(), LeaseToken: t.GetLeaseToken(), Failure: f, Identity: w.o.Identity,
		})
		return err
	})
	// SOLUTION-END
}

// retry repeats a report while the server is unreachable (it keeps the
// lease across a restart); any other answer, success or refusal, is final.
func (w *Worker) retry(work context.Context, report func(context.Context) error) {
	// SOLUTION-BEGIN dur.04
	for n := 1; ; n++ {
		err := report(work)
		if status.Code(err) != codes.Unavailable {
			if err != nil {
				w.o.Logf("report: %v", err)
			}
			return
		}
		if n >= 20 || !w.backoff(work, n) {
			return
		}
	}
	// SOLUTION-END
}

// pollWorkflows handles one workflow task at a time; a task in progress is
// finished (and completed) even when ctx ends.
func (w *Worker) pollWorkflows(ctx, work context.Context) {
	// SOLUTION-BEGIN dur.04
	failures := 0
	for ctx.Err() == nil {
		t, err := w.tasks.PollWorkflowTask(ctx, &durablev1.PollRequest{TaskQueue: w.o.TaskQueue, Identity: w.o.Identity})
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			failures++
			w.o.Logf("poll workflow: %v (retry %d)", err, failures)
			if !w.backoff(ctx, failures) {
				return
			}
			continue
		}
		failures = 0
		if len(t.GetTaskToken()) == 0 {
			continue
		}
		history, err := w.fullHistory(work, t)
		if err != nil {
			w.o.Logf("workflow %s: fetch history: %v", t.GetWorkflowId(), err)
			continue
		}
		cmds, err := w.o.Workflows.HandleWorkflowTask(work, t, history)
		if err != nil {
			w.o.Logf("workflow %s: %v; leaving the task to time out", t.GetWorkflowId(), err)
			continue
		}
		w.retry(work, func(ctx context.Context) error {
			_, err := w.tasks.CompleteWorkflowTask(ctx, &durablev1.CompleteWorkflowTaskRequest{TaskToken: t.GetTaskToken(), Commands: cmds, Identity: w.o.Identity})
			return err
		})
	}
	// SOLUTION-END
}

// fullHistory is the task's first page plus every later page from GetHistory.
func (w *Worker) fullHistory(ctx context.Context, t *durablev1.WorkflowTask) ([]*durablev1.HistoryEvent, error) {
	// SOLUTION-BEGIN dur.04
	h := append([]*durablev1.HistoryEvent(nil), t.GetHistory()...)
	if len(t.GetNextPageToken()) == 0 {
		return h, nil
	}
	st, err := w.wfs.GetHistory(ctx, &durablev1.GetHistoryRequest{WorkflowId: t.GetWorkflowId(), RunId: t.GetRunId(), NextPageToken: t.GetNextPageToken()})
	if err != nil {
		return nil, err
	}
	for {
		ev, err := st.Recv()
		if errors.Is(err, io.EOF) {
			return h, nil
		}
		if err != nil {
			return nil, err
		}
		h = append(h, ev)
	}
	// SOLUTION-END
}
