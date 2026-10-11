package workflow

// The replay engine. A history is a sequence of workflow tasks: each
// WorkflowTaskCompleted names the WorkflowTaskStarted it answers and is
// followed (in the same atomic append) by the events of its commands. To
// handle a task the engine runs the workflow as a coroutine from the start:
//
//	for each past task, in order:
//	    deliver the events before its WorkflowTaskStarted (results resolve futures)
//	    run the workflow until it blocks; every command it issues must match
//	    the next recorded command event of that task (ErrNondeterminism if not)
//	deliver the events before the current WorkflowTaskStarted
//	run the workflow until it blocks; the commands it issues now are new
//
// Only one goroutine runs at a time (the coroutine hands control back and
// forth over channels), so a replay is a deterministic function of the code
// and the history.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// Thresholds of ContinueAsNewSuggested (the server's, durable.proto).
const (
	suggestEvents = 10000
	suggestBytes  = 32 << 20
)

type env struct {
	workflowID, runID string
	input             []byte
	now               time.Time
	suggested         bool
	seq               int
	commands          []*durablev1.Command
	replaying         bool
	expected          []*durablev1.HistoryEvent
	expIdx            int
	futures           map[string]*future
	activityTypes     map[string]string
	actByEvent        map[int64]string
	versions          map[string]int

	resume, yield, killed chan struct{}
	done, finished, dead  bool
	result                []byte
	err                   error
	failure               error
}

// eventHooks lets later modules (dur.08: signals, cancellation) react to
// history events this engine does not resolve itself.
var eventHooks = map[string]func(e *env, ev *durablev1.HistoryEvent){}

func eventKind(ev *durablev1.HistoryEvent) string {
	m := ev.ProtoReflect()
	if fd := m.WhichOneof(m.Descriptor().Oneofs().ByName("attrs")); fd != nil {
		return string(fd.Name())
	}
	return ""
}

func commandKind(c *durablev1.Command) string {
	m := c.ProtoReflect()
	if fd := m.WhichOneof(m.Descriptor().Oneofs().ByName("cmd")); fd != nil {
		return string(fd.Name())
	}
	return ""
}

// isCommandEvent: the events a command produces.
func isCommandEvent(ev *durablev1.HistoryEvent) bool {
	switch eventKind(ev) {
	case "act_scheduled", "timer_started", "marker", "completed", "failed", "continued",
		"canceled", "timer_canceled", "act_cancel_requested":
		return true
	}
	return false
}

// matches reports whether command c is the one that produced event ev.
func matches(c *durablev1.Command, ev *durablev1.HistoryEvent) bool {
	// SOLUTION-BEGIN dur.06
	switch x := c.GetCmd().(type) {
	case *durablev1.Command_ScheduleActivity:
		a := ev.GetActScheduled()
		return a != nil && a.GetActivityId() == x.ScheduleActivity.GetActivityId() && a.GetActivityType() == x.ScheduleActivity.GetActivityType()
	case *durablev1.Command_StartTimer:
		return ev.GetTimerStarted().GetTimerId() == x.StartTimer.GetTimerId() && ev.GetTimerStarted() != nil
	case *durablev1.Command_RecordMarker:
		return ev.GetMarker() != nil && ev.GetMarker().GetMarkerName() == x.RecordMarker.GetMarkerName()
	case *durablev1.Command_Complete:
		return ev.GetCompleted() != nil
	case *durablev1.Command_Fail:
		return ev.GetFailed() != nil
	case *durablev1.Command_ContinueAsNew:
		return ev.GetContinued() != nil
	case *durablev1.Command_CancelTimer:
		return ev.GetTimerCanceled() != nil && ev.GetTimerCanceled().GetTimerId() == x.CancelTimer.GetTimerId()
	case *durablev1.Command_CancelActivity:
		return ev.GetActCancelRequested() != nil
	case *durablev1.Command_CancelWorkflow:
		return ev.GetCanceled() != nil
	}
	return false
	// SOLUTION-END
}

func newEnv(workflowID, runID string) *env {
	return &env{
		workflowID: workflowID, runID: runID,
		futures: map[string]*future{}, activityTypes: map[string]string{},
		actByEvent: map[int64]string{}, versions: map[string]int{},
	}
}

// emit records a command: new in the current task, matched against history
// while replaying a past one.
func (e *env) emit(c *durablev1.Command) {
	// SOLUTION-BEGIN dur.06
	if e.replaying {
		e.consume(c)
		return
	}
	e.commands = append(e.commands, c)
	// SOLUTION-END
}

func (e *env) peek() *durablev1.HistoryEvent {
	if e.expIdx < len(e.expected) {
		return e.expected[e.expIdx]
	}
	return nil
}

// consume matches c against the next recorded command event of the task
// being replayed and returns that event; a mismatch panics with
// ErrNondeterminism (recovered by the coroutine and returned by the task).
func (e *env) consume(c *durablev1.Command) *durablev1.HistoryEvent {
	// SOLUTION-BEGIN dur.06
	ev := e.peek()
	if ev == nil {
		panic(nondeterminism("the workflow issued %s, but the history has no further command for this task", commandKind(c)))
	}
	if !matches(c, ev) {
		panic(nondeterminism("the workflow issued %s, but the history has %s at event %d", commandKind(c), eventKind(ev), ev.GetEventId()))
	}
	e.expIdx++
	return ev
	// SOLUTION-END
}

// deliver applies one history event: results resolve their futures.
func (e *env) deliver(ev *durablev1.HistoryEvent) {
	// SOLUTION-BEGIN dur.06
	act := func(sched int64) (*future, string) {
		id := e.actByEvent[sched]
		return e.futures["a:"+id], id
	}
	fail := func(sched int64, f *durablev1.Failure) {
		if fu, id := act(sched); fu != nil {
			fu.resolve(nil, &ActivityError{ActivityID: id, ActivityType: e.activityTypes[id], Failure: f})
		}
	}
	switch a := ev.GetAttrs().(type) {
	case *durablev1.HistoryEvent_ActScheduled:
		e.actByEvent[ev.GetEventId()] = a.ActScheduled.GetActivityId()
	case *durablev1.HistoryEvent_ActCompleted:
		if fu, _ := act(a.ActCompleted.GetScheduledEventId()); fu != nil {
			fu.resolve(a.ActCompleted.GetResult(), nil)
		}
	case *durablev1.HistoryEvent_ActFailed:
		fail(a.ActFailed.GetScheduledEventId(), a.ActFailed.GetFailure())
	case *durablev1.HistoryEvent_ActTimedOut:
		fail(a.ActTimedOut.GetScheduledEventId(), &durablev1.Failure{Message: "activity timed out", Type: "Timeout"})
	case *durablev1.HistoryEvent_ActCanceled:
		fail(a.ActCanceled.GetScheduledEventId(), &durablev1.Failure{Message: "activity canceled", Type: "Canceled"})
	case *durablev1.HistoryEvent_TimerFired:
		if fu := e.futures["t:"+a.TimerFired.GetTimerId()]; fu != nil {
			fu.resolve(nil, nil)
		}
	default:
		if h := eventHooks[eventKind(ev)]; h != nil {
			h(e, ev)
		}
	}
	// SOLUTION-END
}

// start runs wf as a coroutine that waits for the first step.
func (e *env) start(wf Workflow) {
	// SOLUTION-BEGIN dur.06
	e.resume, e.yield, e.killed = make(chan struct{}), make(chan struct{}), make(chan struct{})
	go func() {
		defer func() {
			r := recover()
			if e.dead {
				return // killed: nobody waits for this goroutine
			}
			if r != nil {
				if err, ok := r.(error); ok {
					e.failure = err
				} else {
					e.failure = fmt.Errorf("workflow panic: %v", r)
				}
			}
			e.done = true
			e.yield <- struct{}{}
		}()
		select {
		case <-e.resume:
		case <-e.killed:
			e.dead = true
			return
		}
		e.result, e.err = wf(Context{e}, e.input)
	}()
	// SOLUTION-END
}

// block hands control back to the engine until the next step.
func (e *env) block() {
	// SOLUTION-BEGIN dur.06
	e.yield <- struct{}{}
	select {
	case <-e.resume:
	case <-e.killed:
		e.dead = true
		runtime.Goexit()
	}
	// SOLUTION-END
}

// step runs the workflow until it blocks or returns; once it has returned,
// its final command (complete, fail, or continue as new) is issued.
func (e *env) step() (err error) {
	// SOLUTION-BEGIN dur.06
	if e.finished {
		return nil
	}
	if !e.done {
		e.resume <- struct{}{}
		<-e.yield
	}
	if e.failure != nil {
		return e.failure
	}
	if !e.done {
		return nil
	}
	defer func() {
		if r := recover(); r != nil {
			err = r.(error)
		}
	}()
	e.finished = true
	var can *continueAsNewError
	switch {
	case errors.As(e.err, &can):
		e.emit(&durablev1.Command{Cmd: &durablev1.Command_ContinueAsNew{ContinueAsNew: &durablev1.ContinueAsNew{Input: can.input}}})
	case e.err != nil:
		e.emit(&durablev1.Command{Cmd: &durablev1.Command_Fail{Fail: &durablev1.FailWorkflow{Failure: &durablev1.Failure{Message: e.err.Error(), Type: "WorkflowError"}}}})
	default:
		e.emit(&durablev1.Command{Cmd: &durablev1.Command_Complete{Complete: &durablev1.CompleteWorkflow{Result: e.result}}})
	}
	return nil
	// SOLUTION-END
}

func (e *env) kill() {
	if e.killed != nil {
		close(e.killed)
	}
}

// replay runs wf over h. current is the event id of the current task's
// WorkflowTaskStarted; 0 replays a closed history, whose last task must end
// the run.
func (e *env) replay(wf Workflow, h []*durablev1.HistoryEvent, current int64) ([]*durablev1.Command, error) {
	// SOLUTION-BEGIN dur.06
	if len(h) == 0 || h[0].GetStarted() == nil {
		return nil, errors.New("workflow: history does not start with WorkflowExecutionStarted")
	}
	e.input = h[0].GetStarted().GetInput()
	sizes := make([]int64, len(h)+1) // sizes[i] = bytes of h[:i]
	for i, ev := range h {
		sizes[i+1] = sizes[i] + int64(proto.Size(ev))
	}
	at := func(started int64) {
		ev := h[started-1]
		e.now = time.UnixMilli(ev.GetTsUnixMs()).UTC()
		e.suggested = started > suggestEvents || sizes[started] > suggestBytes
	}
	e.start(wf)
	defer e.kill()
	next := 0
	deliverBefore := func(id int64) {
		for next < len(h) && h[next].GetEventId() < id {
			e.deliver(h[next])
			next++
		}
	}
	for i, ev := range h {
		c := ev.GetWtCompleted()
		if c == nil {
			continue
		}
		deliverBefore(c.GetStartedEventId())
		at(c.GetStartedEventId())
		e.replaying, e.expIdx, e.expected = true, 0, nil
		for j := i + 1; j < len(h) && isCommandEvent(h[j]); j++ {
			e.expected = append(e.expected, h[j])
		}
		if err := e.step(); err != nil {
			return nil, err
		}
		if e.expIdx != len(e.expected) {
			ev := e.expected[e.expIdx]
			return nil, nondeterminism("the history has %s at event %d, which the workflow no longer issues", eventKind(ev), ev.GetEventId())
		}
	}
	e.replaying, e.expected = false, nil
	if current == 0 {
		if !e.finished {
			return nil, errors.New("workflow: the history ends but the workflow has not")
		}
		return nil, nil
	}
	deliverBefore(current)
	at(current)
	if err := e.step(); err != nil {
		return nil, err
	}
	return e.commands, nil
	// SOLUTION-END
}

// Registry maps workflow type names to code; it is the WorkflowHandler a
// dur.04 worker runs workflow tasks with.
type Registry struct {
	mu sync.RWMutex
	m  map[string]Workflow
}

func NewRegistry() *Registry { return &Registry{m: map[string]Workflow{}} }

// Register makes wf the code of workflow type name.
func (r *Registry) Register(name string, wf Workflow) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.m[name] = wf
}

func (r *Registry) get(name string) Workflow {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.m[name]
}

// HandleWorkflowTask replays the task's history and returns the new
// commands. An error (ErrNondeterminism, a panic, an unknown type) means the
// task must not be completed.
func (r *Registry) HandleWorkflowTask(ctx context.Context, task *durablev1.WorkflowTask, h []*durablev1.HistoryEvent) ([]*durablev1.Command, error) {
	// SOLUTION-BEGIN dur.06
	wf := r.get(task.GetWorkflowType())
	if wf == nil {
		return nil, fmt.Errorf("workflow type %q is not registered on this worker", task.GetWorkflowType())
	}
	var current int64
	for _, ev := range h {
		if ev.GetWtStarted() != nil {
			current = ev.GetEventId()
		}
	}
	if current == 0 {
		return nil, errors.New("workflow: the task's history has no WorkflowTaskStarted")
	}
	return newEnv(task.GetWorkflowId(), task.GetRunId()).replay(wf, h, current)
	// SOLUTION-END
}

// ReplayHistory replays a closed history of workflowType: every command the
// code issues must match the history, the final one included. It returns
// what the workflow returned (ErrNondeterminism wrapped on a mismatch).
func ReplayHistory(r *Registry, workflowType string, h []*durablev1.HistoryEvent) ([]byte, error) {
	// SOLUTION-BEGIN dur.06
	wf := r.get(workflowType)
	if wf == nil {
		return nil, fmt.Errorf("workflow type %q is not registered", workflowType)
	}
	e := newEnv("", "")
	if _, err := e.replay(wf, h, 0); err != nil {
		return nil, err
	}
	return e.result, e.err
	// SOLUTION-END
}

// HistoryFile is the JSON of a recorded run (fixtures/dur/histories/*.json):
// its ids, type, and events in protojson.
type HistoryFile struct {
	WorkflowID   string            `json:"workflow_id"`
	RunID        string            `json:"run_id"`
	WorkflowType string            `json:"workflow_type"`
	Events       []json.RawMessage `json:"events"`
}

// LoadHistory reads a recorded run.
func LoadHistory(path string) (*HistoryFile, []*durablev1.HistoryEvent, error) {
	// SOLUTION-BEGIN dur.06
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var f HistoryFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	h := make([]*durablev1.HistoryEvent, len(f.Events))
	for i, m := range f.Events {
		h[i] = &durablev1.HistoryEvent{}
		if err := protojson.Unmarshal(m, h[i]); err != nil {
			return nil, nil, fmt.Errorf("%s event %d: %w", path, i+1, err)
		}
	}
	return &f, h, nil
	// SOLUTION-END
}

// ReplayHistoryFromFile replays a recorded run with the code registered
// under its workflow type.
func ReplayHistoryFromFile(r *Registry, path string) ([]byte, error) {
	// SOLUTION-BEGIN dur.06
	f, h, err := LoadHistory(path)
	if err != nil {
		return nil, err
	}
	return ReplayHistory(r, f.WorkflowType, h)
	// SOLUTION-END
}
