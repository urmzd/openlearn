package workflow

// Signals, cancellation, and named activities (dur.08): the SDK half. The
// replay engine (replay.go, dur.06) hands every history event it does not
// resolve itself to eventHooks; this file registers the hooks for
// SignalReceived and WorkflowExecutionCancelRequested and keeps their state
// where the engine keeps all per-run state, in the run's futures, so a
// replay rebuilds it from history like everything else:
//
//	"s:<name>:<n>"   the n-th signal named <name> (1-based), its payload as value
//	"sc:<name>"      how many signals named <name> the workflow consumed
//	"c:"             resolved once the run has a cancel request
//
// A signal that arrived before the workflow waits for it is already in its
// future when the workflow asks: buffered, never lost, delivered in arrival
// order per name.

import (
	"strconv"
	"time"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

func init() {
	eventHooks["signal"] = deliverSignal
	eventHooks["cancel_requested"] = deliverCancel
}

func sigKey(name string, n int) string { return "s:" + name + ":" + strconv.Itoa(n) }

// signalCount is how many signals named name have been delivered so far.
func signalCount(e *env, name string) int {
	// SOLUTION-BEGIN dur.08
	n := 0
	for e.futures[sigKey(name, n+1)] != nil {
		n++
	}
	return n
	// SOLUTION-END
}

func deliverSignal(e *env, ev *durablev1.HistoryEvent) {
	// SOLUTION-BEGIN dur.08
	s := ev.GetSignal()
	f := &future{}
	f.resolve(append([]byte(nil), s.GetInput()...), nil)
	e.futures[sigKey(s.GetSignalName(), signalCount(e, s.GetSignalName())+1)] = f
	// SOLUTION-END
}

func deliverCancel(e *env, ev *durablev1.HistoryEvent) {
	// SOLUTION-BEGIN dur.08
	f := &future{}
	f.resolve(nil, nil)
	e.futures["c:"] = f
	// SOLUTION-END
}

// CancelRequested reports whether the run has a cancel request as of the
// events delivered so far.
func CancelRequested(ctx Context) bool {
	// SOLUTION-BEGIN dur.08
	f := ctx.e.futures["c:"]
	return f != nil && f.ready
	// SOLUTION-END
}

// Await blocks the workflow until cond is true. cond is checked now and
// again after every batch of history events the engine delivers, so it may
// only read workflow state (futures, signal channels, CancelRequested).
func Await(ctx Context, cond func() bool) {
	// SOLUTION-BEGIN dur.08
	for !cond() {
		ctx.e.block()
	}
	// SOLUTION-END
}

// ReceiveChannel is the buffer of one signal name.
type ReceiveChannel struct {
	e    *env
	name string
}

// GetSignalChannel returns the channel of signals named name.
func GetSignalChannel(ctx Context, name string) ReceiveChannel {
	return ReceiveChannel{e: ctx.e, name: name}
}

func (c ReceiveChannel) consumed() int {
	// SOLUTION-BEGIN dur.08
	if f := c.e.futures["sc:"+c.name]; f != nil {
		n, _ := strconv.Atoi(string(f.value))
		return n
	}
	return 0
	// SOLUTION-END
}

// Len is the number of signals received and not yet consumed.
func (c ReceiveChannel) Len() int {
	// SOLUTION-BEGIN dur.08
	return signalCount(c.e, c.name) - c.consumed()
	// SOLUTION-END
}

// ReceiveAsync takes the oldest buffered signal without blocking.
func (c ReceiveChannel) ReceiveAsync() ([]byte, bool) {
	// SOLUTION-BEGIN dur.08
	n := c.consumed()
	f := c.e.futures[sigKey(c.name, n+1)]
	if f == nil {
		return nil, false
	}
	done := &future{}
	done.resolve([]byte(strconv.Itoa(n+1)), nil)
	c.e.futures["sc:"+c.name] = done
	return f.value, true
	// SOLUTION-END
}

// Receive blocks until a signal is buffered and takes it.
func (c ReceiveChannel) Receive(ctx Context) []byte {
	// SOLUTION-BEGIN dur.08
	Await(ctx, func() bool { return c.Len() > 0 })
	b, _ := c.ReceiveAsync()
	return b
	// SOLUTION-END
}

// Timer is a durable timer the workflow can wait on alongside other
// conditions (a signal with a timeout) and cancel.
type Timer struct {
	id string
	f  *future
}

// NewTimer starts a durable timer of d (StartTimer).
func NewTimer(ctx Context, d time.Duration) *Timer {
	// SOLUTION-BEGIN dur.08
	e := ctx.e
	e.seq++
	id := strconv.Itoa(e.seq)
	f := &future{}
	if d <= 0 {
		f.resolve(nil, nil)
		return &Timer{id: "", f: f}
	}
	e.futures["t:"+id] = f
	e.emit(&durablev1.Command{Cmd: &durablev1.Command_StartTimer{StartTimer: &durablev1.StartTimer{TimerId: id, DurationMs: d.Milliseconds()}}})
	return &Timer{id: id, f: f}
	// SOLUTION-END
}

// Fired reports whether the timer has fired.
func (t *Timer) Fired() bool { return t.f.ready }

// Cancel stops a timer that has not fired (CancelTimer); a fired or already
// cancelled timer is left alone.
func (t *Timer) Cancel(ctx Context) {
	// SOLUTION-BEGIN dur.08
	if t.id == "" || t.f.ready {
		return
	}
	t.f.resolve(nil, nil)
	ctx.e.emit(&durablev1.Command{Cmd: &durablev1.Command_CancelTimer{CancelTimer: &durablev1.CancelTimer{TimerId: t.id}}})
	// SOLUTION-END
}

// ExecuteActivityID is ExecuteActivity with a chosen activity id, so the
// idempotency key is "<workflow_id>/<id>" (CorpusBuild keys stages by name).
// Ids must be unique within the run and must not be numbers (those are the
// ids ExecuteActivity numbers in order).
func ExecuteActivityID[O any](ctx Context, id, name string, in any, o ActivityOptions) Future[O] {
	// SOLUTION-BEGIN dur.08
	e := ctx.e
	f := &future{}
	if _, err := strconv.Atoi(id); err == nil || id == "" || e.futures["a:"+id] != nil {
		f.resolve(nil, &ActivityError{ActivityID: id, ActivityType: name, Failure: &durablev1.Failure{
			Message: "activity id must be unique and not a number", Type: "BadActivityID", NonRetryable: true}})
		return typedFuture[O]{f}
	}
	input, err := encode(in)
	if err != nil {
		f.resolve(nil, err)
		return typedFuture[O]{f}
	}
	e.futures["a:"+id] = f
	e.activityTypes[id] = name
	e.emit(&durablev1.Command{Cmd: &durablev1.Command_ScheduleActivity{ScheduleActivity: &durablev1.ScheduleActivity{
		ActivityId: id, ActivityType: name, Input: input, Options: o.proto(),
	}}})
	return typedFuture[O]{f}
	// SOLUTION-END
}

// CancelRun ends the run as CANCELED (CancelWorkflowExecution with details):
// the workflow's answer to a cancel request, after its compensations. It
// does not return; the run is over once the task completes.
func CancelRun(ctx Context, details []byte) {
	// SOLUTION-BEGIN dur.08
	ctx.e.emit(&durablev1.Command{Cmd: &durablev1.Command_CancelWorkflow{CancelWorkflow: &durablev1.CancelWorkflowExecution{Details: details}}})
	for {
		ctx.e.block()
	}
	// SOLUTION-END
}
