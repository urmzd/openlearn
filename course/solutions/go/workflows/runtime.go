package workflows

// The workflow runtime seam (dur.08). Platform workflows (CorpusBuild,
// TrainRun, EvalSuite, ModelRelease) are plain Go functions of a Runtime:
// the slice of the durable SDK (ExecuteActivity, Sleep, Now,
// signal channels, cancellation) they need. The worker's composition root
// adapts a workflow.Context to Runtime when it registers them, and the
// course tests drive them with a replaying simulator, so the workflow logic
// is tested without a server.
//
// FromContext is that adapter over the dur.06 SDK, and Workflow wraps a
// Runtime workflow as a workflow.Workflow for a Registry:
//
//	reg.Register("CorpusBuild", workflows.Workflow(workflows.CorpusBuild))
//
// The contract every Runtime keeps, and every workflow relies on:
//
//   - every call is recorded in the run's history the first time and
//     answered from it on replay: a workflow that runs again after a crash
//     reaches the same point without executing anything twice
//   - once the run has a cancel request, every blocking call of the Runtime
//     returns an error that wraps ErrCanceled, the one waiting at that moment
//     included; calls of Detached() are never interrupted, which is how
//     compensations run after a cancel

import (
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"tinyllm/durable/workflow"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// ErrCanceled ends a blocking Runtime call once the run has a cancel
// request. A workflow returns it (wrapped or not) to end the run CANCELED.
var ErrCanceled = errors.New("workflows: the run was canceled")

// StepOptions are the per-call options of an activity
// (tl.durable.v1.ActivityOptions; the SDK supplies retry backoff).
type StepOptions struct {
	// ID is the activity id, which makes the idempotency key
	// "<workflow_id>/<ID>". "" lets the Runtime number calls in order.
	ID               string
	StartToClose     time.Duration
	HeartbeatTimeout time.Duration
	MaxAttempts      int // 0: the server's default (dlq_after_attempts)
}

// Runtime is the durable SDK seen from one workflow run.
type Runtime interface {
	// ExecuteActivity runs activity `name` with `in` (JSON-encoded) and
	// decodes its JSON result into out (nil: discard). It returns a
	// *StepFailure once the activity failed for good, or an error wrapping
	// ErrCanceled.
	ExecuteActivity(name string, in any, opts StepOptions, out any) error
	// Sleep blocks on a durable timer.
	Sleep(d time.Duration) error
	// Now is the run's deterministic clock, never the wall clock.
	Now() time.Time
	// AwaitSignal blocks until one of the named signals arrives or timeout
	// passes (0: no timeout; then name is ""). Signals sent before the
	// workflow waits are buffered, in arrival order.
	AwaitSignal(names []string, timeout time.Duration) (name string, payload []byte, err error)
	// Detached is the same run with cancellation ignored: compensations
	// run on it after ErrCanceled.
	Detached() Runtime
}

// StepFailure is an activity that failed for good.
type StepFailure struct {
	Activity     string
	Type         string
	Message      string
	NonRetryable bool
}

func (f *StepFailure) Error() string {
	return fmt.Sprintf("activity %s failed (%s): %s", f.Activity, f.Type, f.Message)
}

// IsCanceled reports whether err ends the run as canceled.
func IsCanceled(err error) bool {
	return errors.Is(err, ErrCanceled)
}

// FromContext adapts a workflow.Context (dur.06) to Runtime.
func FromContext(ctx workflow.Context) Runtime {
	return &sdk{ctx: ctx}
}

type sdk struct {
	ctx      workflow.Context
	detached bool
}

// canceled: the run has a cancel request and this Runtime is not detached.
func (s *sdk) canceled() bool {
	return !s.detached && workflow.CancelRequested(s.ctx)
}

func (s *sdk) ExecuteActivity(name string, in any, o StepOptions, out any) error {
	if s.canceled() {
		return fmt.Errorf("activity %s: %w", name, ErrCanceled)
	}
	opts := workflow.ActivityOptions{StartToClose: o.StartToClose, HeartbeatTimeout: o.HeartbeatTimeout}
	if o.MaxAttempts > 0 {
		opts.Retry = &durablev1.RetryPolicy{MaxAttempts: int32(o.MaxAttempts)}
	}
	var f workflow.Future[json.RawMessage]
	if o.ID != "" {
		f = workflow.ExecuteActivityID[json.RawMessage](s.ctx, o.ID, name, in, opts)
	} else {
		f = workflow.ExecuteActivity[json.RawMessage](s.ctx, name, in, opts)
	}
	// Wait for the result, or for a cancel: the activity hears the cancel
	// from its heartbeat; the workflow stops waiting for it now.
	workflow.Await(s.ctx, func() bool { return f.Ready() || s.canceled() })
	if !f.Ready() {
		return fmt.Errorf("activity %s: %w", name, ErrCanceled)
	}
	raw, err := f.Get(s.ctx)
	if err != nil {
		var ae *workflow.ActivityError
		if errors.As(err, &ae) {
			sf := &StepFailure{Activity: name, Type: ae.Failure.GetType(), Message: ae.Failure.GetMessage(), NonRetryable: ae.Failure.GetNonRetryable()}
			if sf.Type == "Canceled" && !s.detached {
				return fmt.Errorf("%w: %w", sf, ErrCanceled)
			}
			return sf
		}
		return err
	}
	if out != nil && len(raw) > 0 {
		return json.Unmarshal(raw, out)
	}
	return nil
}

func (s *sdk) Sleep(d time.Duration) error {
	if s.canceled() {
		return ErrCanceled
	}
	t := workflow.NewTimer(s.ctx, d)
	workflow.Await(s.ctx, func() bool { return t.Fired() || s.canceled() })
	if !t.Fired() {
		t.Cancel(s.ctx)
		return ErrCanceled
	}
	return nil
}

func (s *sdk) Now() time.Time { return workflow.Now(s.ctx) }

func (s *sdk) AwaitSignal(names []string, timeout time.Duration) (string, []byte, error) {
	if s.canceled() {
		return "", nil, ErrCanceled
	}
	chans := make([]workflow.ReceiveChannel, len(names))
	for i, n := range names {
		chans[i] = workflow.GetSignalChannel(s.ctx, n)
	}
	ready := func() int {
		for i, c := range chans {
			if c.Len() > 0 {
				return i
			}
		}
		return -1
	}
	var t *workflow.Timer
	if ready() < 0 && timeout > 0 {
		t = workflow.NewTimer(s.ctx, timeout)
	}
	workflow.Await(s.ctx, func() bool { return ready() >= 0 || (t != nil && t.Fired()) || s.canceled() })
	i := ready()
	if t != nil && i >= 0 {
		t.Cancel(s.ctx)
	}
	switch {
	case i >= 0:
		b, _ := chans[i].ReceiveAsync()
		return names[i], b, nil
	case s.canceled():
		if t != nil {
			t.Cancel(s.ctx)
		}
		return "", nil, ErrCanceled
	default:
		return "", nil, nil // the timeout fired
	}
}

func (s *sdk) Detached() Runtime { return &sdk{ctx: s.ctx, detached: true} }

// Workflow wraps f as a workflow.Workflow: the input and the result travel
// as JSON, and an error matching ErrCanceled ends the run CANCELED
// (CancelWorkflowExecution) instead of FAILED.
func Workflow[I, O any](f func(Runtime, I) (O, error)) workflow.Workflow {
	return func(ctx workflow.Context, input []byte) ([]byte, error) { return runWorkflow(f, ctx, input) }
}

func runWorkflow[I, O any](f func(Runtime, I) (O, error), ctx workflow.Context, input []byte) ([]byte, error) {
	var in I
	if len(input) > 0 {
		if err := json.Unmarshal(input, &in); err != nil {
			return nil, fmt.Errorf("workflow input: %w", err)
		}
	}
	out, err := f(FromContext(ctx), in)
	if IsCanceled(err) {
		workflow.CancelRun(ctx, []byte(err.Error()))
	}
	if err != nil {
		return nil, err
	}
	return json.Marshal(out)
}
