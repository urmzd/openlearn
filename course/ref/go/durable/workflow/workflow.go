// Package workflow is the deterministic workflow SDK (dur.06). A workflow is
// ordinary Go code:
//
//	func Echo(ctx workflow.Context, in []byte) ([]byte, error) {
//		return workflow.ExecuteActivity[[]byte](ctx, "Upper", in, opts).Get(ctx)
//	}
//
// It never runs once from start to finish. Every workflow task replays it
// from the beginning against the run's history: each call that would have a
// side effect (an activity, a timer, a side effect, a version check) is a
// command, and during replay each command must match the event the history
// recorded for it, or the task fails with ErrNondeterminism. Results come
// from the history, so a replayed workflow reaches exactly the state it had
// and continues from there. That is why workflow code must be deterministic:
// no time.Now (use Now), no math/rand (use SideEffect), no goroutines, no
// I/O, no map iteration order.
//
// This file is the API the workflow calls; replay.go is the engine.
package workflow

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// Workflow is the signature of workflow code.
type Workflow func(ctx Context, input []byte) ([]byte, error)

// Context is the handle a workflow passes to every SDK call. It is not a
// context.Context: workflow code does not block on channels or I/O.
type Context struct{ e *env }

// ErrNondeterminism: replaying the workflow produced a command the history
// does not have at that position (DESIGN 2.7).
var ErrNondeterminism = errors.New("durable: command does not match history")

// ActivityOptions are the per-call options of ExecuteActivity.
type ActivityOptions struct {
	TaskQueue        string // "" = the workflow's task queue
	StartToClose     time.Duration
	ScheduleToClose  time.Duration
	HeartbeatTimeout time.Duration
	Retry            *durablev1.RetryPolicy
}

func (o ActivityOptions) proto() *durablev1.ActivityOptions {
	return &durablev1.ActivityOptions{
		TaskQueue: o.TaskQueue, StartToCloseMs: o.StartToClose.Milliseconds(),
		ScheduleToCloseMs: o.ScheduleToClose.Milliseconds(), HeartbeatTimeoutMs: o.HeartbeatTimeout.Milliseconds(),
		Retry: o.Retry,
	}
}

// ActivityError is what Future.Get returns for an activity that failed for
// good (non-retryable), timed out, or was canceled.
type ActivityError struct {
	ActivityID, ActivityType string
	Failure                  *durablev1.Failure
}

func (e *ActivityError) Error() string {
	return fmt.Sprintf("activity %s (%s) failed: %s", e.ActivityID, e.ActivityType, e.Failure.GetMessage())
}

// Future is the eventual result of an activity.
type Future[O any] interface {
	Get(ctx Context) (O, error) // blocks the workflow (not a thread) until the result is in history
	Ready() bool
}

type future struct {
	ready bool
	value []byte
	err   error
}

func (f *future) resolve(v []byte, err error) {
	f.ready, f.value, f.err = true, v, err
}

func (f *future) get(ctx Context) ([]byte, error) {
	// SOLUTION-BEGIN dur.06
	for !f.ready {
		ctx.e.block()
	}
	return f.value, f.err
	// SOLUTION-END
}

type typedFuture[O any] struct{ f *future }

func (t typedFuture[O]) Ready() bool { return t.f.ready }

func (t typedFuture[O]) Get(ctx Context) (O, error) {
	// SOLUTION-BEGIN dur.06
	var out O
	b, err := t.f.get(ctx)
	if err != nil {
		return out, err
	}
	return out, decode(b, &out)
	// SOLUTION-END
}

// encode is the payload of a value: []byte as is, anything else as JSON.
func encode(v any) ([]byte, error) {
	// SOLUTION-BEGIN dur.06
	if b, ok := v.([]byte); ok {
		return b, nil
	}
	return json.Marshal(v)
	// SOLUTION-END
}

func decode(b []byte, out any) error {
	// SOLUTION-BEGIN dur.06
	if p, ok := out.(*[]byte); ok {
		*p = b
		return nil
	}
	if len(b) == 0 {
		return nil
	}
	return json.Unmarshal(b, out)
	// SOLUTION-END
}

// ExecuteActivity schedules activity name with input in (JSON, or []byte as
// is). Its id is the command's sequence number, so the same code always
// schedules the same ids in the same order.
func ExecuteActivity[O any](ctx Context, name string, in any, o ActivityOptions) Future[O] {
	// SOLUTION-BEGIN dur.06
	e := ctx.e
	f := &future{}
	input, err := encode(in)
	if err != nil {
		f.resolve(nil, err)
		return typedFuture[O]{f}
	}
	e.seq++
	id := strconv.Itoa(e.seq)
	e.futures["a:"+id] = f
	e.activityTypes[id] = name
	e.emit(&durablev1.Command{Cmd: &durablev1.Command_ScheduleActivity{ScheduleActivity: &durablev1.ScheduleActivity{
		ActivityId: id, ActivityType: name, Input: input, Options: o.proto(),
	}}})
	return typedFuture[O]{f}
	// SOLUTION-END
}

// Sleep blocks the workflow for d with a durable timer (dur.07 fires it).
func Sleep(ctx Context, d time.Duration) error {
	// SOLUTION-BEGIN dur.06
	if d <= 0 {
		return nil
	}
	e := ctx.e
	e.seq++
	id := strconv.Itoa(e.seq)
	f := &future{}
	e.futures["t:"+id] = f
	e.emit(&durablev1.Command{Cmd: &durablev1.Command_StartTimer{StartTimer: &durablev1.StartTimer{TimerId: id, DurationMs: d.Milliseconds()}}})
	_, err := f.get(ctx)
	return err
	// SOLUTION-END
}

// Now is the time of the current workflow task (its WorkflowTaskStarted
// event), the same on every replay. Never call time.Now in a workflow.
func Now(ctx Context) time.Time {
	// SOLUTION-BEGIN dur.06
	return ctx.e.now
	// SOLUTION-END
}

// SideEffect runs f once and records its result as a marker; replays return
// the recorded value without calling f. Use it for random ids and the like.
func SideEffect[T any](ctx Context, f func() T) T {
	// SOLUTION-BEGIN dur.06
	e := ctx.e
	var out T
	if e.replaying {
		ev := e.consume(&durablev1.Command{Cmd: &durablev1.Command_RecordMarker{RecordMarker: &durablev1.RecordMarker{MarkerName: "SideEffect"}}})
		if err := json.Unmarshal(ev.GetMarker().GetDetails(), &out); err != nil {
			panic(nondeterminism("SideEffect marker at event %d does not decode: %v", ev.GetEventId(), err))
		}
		return out
	}
	out = f()
	details, err := json.Marshal(out)
	if err != nil {
		panic(fmt.Errorf("workflow: SideEffect value does not encode as JSON: %w", err))
	}
	e.emit(&durablev1.Command{Cmd: &durablev1.Command_RecordMarker{RecordMarker: &durablev1.RecordMarker{MarkerName: "SideEffect", Details: details}}})
	return out
	// SOLUTION-END
}

type versionMarker struct {
	ChangeID string `json:"change_id"`
	Version  int    `json:"version"`
}

// GetVersion makes a code change safe for runs already in flight. Code
// written as
//
//	if workflow.GetVersion(ctx, "use-b", 0, 1) == 0 { old path } else { new path }
//
// returns max for a new run (and records a "Version" marker), the recorded
// version when replaying a run that has the marker, and min when replaying
// a run whose history predates the change (no marker at this point). The
// same changeID returns the same version for the rest of the run.
func GetVersion(ctx Context, changeID string, min, max int) int {
	// SOLUTION-BEGIN dur.06
	e := ctx.e
	if v, ok := e.versions[changeID]; ok {
		return v
	}
	v := max
	if e.replaying {
		v = min
		if ev := e.peek(); ev != nil && ev.GetMarker().GetMarkerName() == "Version" {
			var vm versionMarker
			if json.Unmarshal(ev.GetMarker().GetDetails(), &vm) == nil && vm.ChangeID == changeID {
				e.expIdx++
				v = vm.Version
			}
		}
	} else {
		details, _ := json.Marshal(versionMarker{ChangeID: changeID, Version: max})
		e.emit(&durablev1.Command{Cmd: &durablev1.Command_RecordMarker{RecordMarker: &durablev1.RecordMarker{MarkerName: "Version", Details: details}}})
	}
	if v < min || v > max {
		panic(nondeterminism("GetVersion(%q) = %d is outside the supported range [%d, %d]", changeID, v, min, max))
	}
	e.versions[changeID] = v
	return v
	// SOLUTION-END
}

type continueAsNewError struct{ input []byte }

func (c *continueAsNewError) Error() string { return "workflow: continue as new" }

// ContinueAsNew ends this run; a new run of the same workflow id and type
// starts with input. Use it as the workflow's return value:
//
//	return nil, workflow.ContinueAsNew(ctx, next)
func ContinueAsNew(ctx Context, input []byte) error {
	// SOLUTION-BEGIN dur.06
	return &continueAsNewError{input: input}
	// SOLUTION-END
}

// ContinueAsNewSuggested reports whether the history, as of the current
// workflow task, has passed 10,000 events or 32 MiB: the server's rule,
// recomputed from the history so it is the same on every replay.
func ContinueAsNewSuggested(ctx Context) bool {
	// SOLUTION-BEGIN dur.06
	return ctx.e.suggested
	// SOLUTION-END
}

// Info about the run, from its first event.
func WorkflowID(ctx Context) string { return ctx.e.workflowID }
func RunID(ctx Context) string      { return ctx.e.runID }

func nondeterminism(format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{ErrNondeterminism}, args...)...)
}
