package server

// The TaskService rpcs workers call (dur.04): long-poll a workflow task,
// complete it with commands, long-poll an activity task, complete it. Every
// delivery is a dur.03 lease; its fencing token travels inside the opaque
// task_token, so a late or duplicate completion is refused.
//
// Commands are dispatched through a table. This file registers the commands
// it owns (schedule_activity, complete, fail); continue.go (dur.06) and
// timers.go (dur.07) register theirs the same way, and so can dur.08.

import (
	"context"
	"encoding/json"
	"errors"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"tinyllm/durable/queue"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// taskToken is the content of a task_token.
type taskToken struct {
	Queue   string `json:"q"`
	TaskID  string `json:"t"`
	Lease   uint64 `json:"l"`
	RunID   string `json:"r"`
	Sched   int64  `json:"s"`
	Started int64  `json:"st,omitempty"` // workflow tasks: the WorkflowTaskStarted event id
}

func (t taskToken) lease() queue.Lease {
	return queue.Lease{Queue: t.Queue, Task: queue.Task{ID: t.TaskID}, Token: t.Lease}
}

func encodeToken(t taskToken) []byte {
	b, _ := json.Marshal(t)
	return b
}

func decodeToken(b []byte) (taskToken, error) {
	var t taskToken
	if err := json.Unmarshal(b, &t); err != nil || t.RunID == "" || t.TaskID == "" {
		return t, status.Error(codes.InvalidArgument, "malformed task_token")
	}
	return t, nil
}

// commandHandler appends the events of one command to b. completed is the
// event id of this batch's WorkflowTaskCompleted.
type commandHandler func(s *Server, b *batch, completed int64, c *durablev1.Command) error

var commandHandlers = map[string]commandHandler{}

func registerCommand(kind string, h commandHandler) { commandHandlers[kind] = h }

func init() {
	registerCommand("schedule_activity", scheduleActivity)
	registerCommand("complete", completeWorkflow)
	registerCommand("fail", failWorkflow)
}

// CommandKind is the name of a command's oneof field ("schedule_activity", ...).
func CommandKind(c *durablev1.Command) string {
	m := c.ProtoReflect()
	fd := m.WhichOneof(m.Descriptor().Oneofs().ByName("cmd"))
	if fd == nil {
		return ""
	}
	return string(fd.Name())
}

// PollWorkflowTask long-polls the run's workflow tasks. A delivered task
// whose WorkflowTaskScheduled is no longer pending is stale (history says
// it already ran): it is acknowledged and the poll continues. Otherwise the
// server appends WorkflowTaskStarted and returns the history through it,
// first page only (at most 1 MiB), with continue_as_new_suggested set once
// the history passes 10,000 events or 32 MiB.
func (s *Server) PollWorkflowTask(ctx context.Context, req *durablev1.PollRequest) (*durablev1.WorkflowTask, error) {
	if req.GetTaskQueue() == "" {
		return nil, status.Error(codes.InvalidArgument, "task_queue is required")
	}
	end := s.o.Clock.Now().Add(s.o.PollWait)
	for {
		wait := end.Sub(s.o.Clock.Now())
		if wait < 0 {
			wait = 0
		}
		l, err := s.o.Queue.Poll(ctx, workflowQueue(req.GetTaskQueue()), req.GetIdentity(), wait)
		if errors.Is(err, queue.ErrNoTask) {
			return &durablev1.WorkflowTask{}, nil
		}
		if err != nil {
			return nil, logErr(err)
		}
		t, err := s.startWorkflowTask(ctx, l, req.GetIdentity())
		if err != nil || t != nil {
			return t, err
		}
	}
}

func (s *Server) startWorkflowTask(ctx context.Context, l queue.Lease, identity string) (*durablev1.WorkflowTask, error) {
	runID, _, sched, err := parseTaskID(l.Task.ID)
	if err != nil {
		s.o.Queue.Complete(ctx, l)
		return nil, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[runID]
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || r.wtScheduled != sched {
		s.o.Queue.Complete(ctx, l)
		return nil, nil
	}
	b := &batch{r: r}
	st := s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_WtStarted{
		WtStarted: &durablev1.WorkflowTaskStarted{ScheduledEventId: sched, Identity: identity},
	}})
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	page, next := firstPage(r.history)
	return &durablev1.WorkflowTask{
		TaskToken:  encodeToken(taskToken{Queue: l.Queue, TaskID: l.Task.ID, Lease: l.Token, RunID: r.id, Sched: sched, Started: st.GetEventId()}),
		WorkflowId: r.workflowID, RunId: r.id, WorkflowType: r.wfType,
		History: page, NextPageToken: next,
		ContinueAsNewSuggested: len(r.history) > ContinueAsNewEvents || r.bytes > ContinueAsNewBytes,
	}, nil
}

// firstPage is the longest prefix of h within FirstPageBytes (at least one
// event), and the page token of the rest (nil when nothing is left).
func firstPage(h []*durablev1.HistoryEvent) ([]*durablev1.HistoryEvent, []byte) {
	size := 0
	for i, e := range h {
		size += proto.Size(e)
		if size > FirstPageBytes && i > 0 {
			return h[:i:i], PageToken(int64(i + 1))
		}
	}
	return h[:len(h):len(h)], nil
}

// CompleteWorkflowTask appends WorkflowTaskCompleted and the events of the
// commands, atomically. The token must name the run's current workflow task
// (its pending WorkflowTaskScheduled and latest WorkflowTaskStarted) and a
// live lease; anything else is FAILED_PRECONDITION and appends nothing.
func (s *Server) CompleteWorkflowTask(ctx context.Context, req *durablev1.CompleteWorkflowTaskRequest) (*durablev1.Empty, error) {
	tok, err := decodeToken(req.GetTaskToken())
	if err != nil {
		return nil, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[tok.RunID]
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || r.wtScheduled != tok.Sched || r.wtStarted != tok.Started {
		return nil, status.Error(codes.FailedPrecondition, "not the run's current workflow task")
	}
	if err := s.o.Queue.Check(tok.lease()); err != nil {
		return nil, status.Error(codes.FailedPrecondition, "workflow task lease lost")
	}
	b := &batch{r: r}
	done := s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_WtCompleted{
		WtCompleted: &durablev1.WorkflowTaskCompleted{ScheduledEventId: tok.Sched, StartedEventId: tok.Started, Identity: req.GetIdentity()},
	}})
	for _, c := range req.GetCommands() {
		if closing(b.events[len(b.events)-1]) {
			return nil, status.Error(codes.InvalidArgument, "a command after the one that ends the run")
		}
		h := commandHandlers[CommandKind(c)]
		if h == nil {
			return nil, status.Errorf(codes.Unimplemented, "command %q is not supported", CommandKind(c))
		}
		if err := h(s, b, done.GetEventId(), c); err != nil {
			return nil, err
		}
	}
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	s.o.Queue.Complete(ctx, tok.lease())
	return &durablev1.Empty{}, nil
}

func scheduleActivity(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	sa := c.GetScheduleActivity()
	switch {
	case sa.GetActivityId() == "" || sa.GetActivityType() == "":
		return status.Error(codes.InvalidArgument, "activity_id and activity_type are required")
	case len(sa.GetInput()) > MaxPayloadBytes:
		return status.Errorf(codes.InvalidArgument, "activity input is %d bytes; the limit is %d", len(sa.GetInput()), MaxPayloadBytes)
	case sa.GetOptions().GetStartToCloseMs() <= 0:
		return status.Error(codes.InvalidArgument, "start_to_close_ms is required")
	}
	for _, a := range b.r.activities {
		if a.attrs.GetActivityId() == sa.GetActivityId() {
			return status.Errorf(codes.InvalidArgument, "activity id %q is already pending", sa.GetActivityId())
		}
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActScheduled{ActScheduled: &durablev1.ActivityTaskScheduled{
		ActivityId: sa.GetActivityId(), ActivityType: sa.GetActivityType(), Input: sa.GetInput(),
		Options: sa.GetOptions(), WorkflowTaskCompletedEventId: completed,
	}}})
	return nil
}

func completeWorkflow(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	res := c.GetComplete().GetResult()
	if len(res) > MaxPayloadBytes {
		return status.Errorf(codes.InvalidArgument, "workflow result is %d bytes; the limit is %d", len(res), MaxPayloadBytes)
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Completed{Completed: &durablev1.WorkflowExecutionCompleted{Result: res}}})
	return nil
}

func failWorkflow(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Failed{Failed: &durablev1.WorkflowExecutionFailed{Failure: c.GetFail().GetFailure()}}})
	return nil
}

// PollActivityTask long-polls activity tasks. A delivery whose activity is no
// longer pending (completed, failed, canceled, run closed) is acknowledged
// and skipped. The idempotency key "<workflow_id>/<activity_id>" is the same
// for every attempt; lease_token fences this attempt.
func (s *Server) PollActivityTask(ctx context.Context, req *durablev1.PollRequest) (*durablev1.ActivityTask, error) {
	if req.GetTaskQueue() == "" {
		return nil, status.Error(codes.InvalidArgument, "task_queue is required")
	}
	end := s.o.Clock.Now().Add(s.o.PollWait)
	for {
		wait := end.Sub(s.o.Clock.Now())
		if wait < 0 {
			wait = 0
		}
		l, err := s.o.Queue.Poll(ctx, activityQueue(req.GetTaskQueue()), req.GetIdentity(), wait)
		if errors.Is(err, queue.ErrNoTask) {
			return &durablev1.ActivityTask{}, nil
		}
		if err != nil {
			return nil, logErr(err)
		}
		if t := s.startActivityTask(ctx, l); t != nil {
			return t, nil
		}
	}
}

func (s *Server) startActivityTask(ctx context.Context, l queue.Lease) *durablev1.ActivityTask {
	runID, _, sched, err := parseTaskID(l.Task.ID)
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[runID]
	var a *activity
	if err == nil && r != nil && r.status == durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		a = r.activities[sched]
	}
	if a == nil {
		s.o.Queue.Complete(ctx, l)
		return nil
	}
	a.attempt = int32(l.Attempt)
	details := l.Details
	if details == nil {
		details = a.lastHeartbeat
	}
	return &durablev1.ActivityTask{
		TaskToken:  encodeToken(taskToken{Queue: l.Queue, TaskID: l.Task.ID, Lease: l.Token, RunID: r.id, Sched: sched}),
		WorkflowId: r.workflowID, RunId: r.id, ActivityId: a.attrs.GetActivityId(), ActivityType: a.attrs.GetActivityType(),
		Input: a.attrs.GetInput(), Attempt: int32(l.Attempt),
		IdempotencyKey:       r.workflowID + "/" + a.attrs.GetActivityId(),
		LastHeartbeatDetails: details, TraceContext: r.traceContext, LeaseToken: l.Token,
		DeadlineUnixMs: l.Deadline.UnixMilli(), HeartbeatTimeoutMs: a.attrs.GetOptions().GetHeartbeatTimeoutMs(),
	}
}

// pendingActivity resolves an activity task token to its run and pending
// activity, and checks that its lease is still live. The caller holds s.mu.
func (s *Server) pendingActivity(tok taskToken) (*run, *activity, error) {
	r := s.runs[tok.RunID]
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || r.activities[tok.Sched] == nil {
		return nil, nil, status.Error(codes.FailedPrecondition, "the activity is no longer pending")
	}
	if err := s.o.Queue.Check(tok.lease()); err != nil {
		return nil, nil, status.Error(codes.FailedPrecondition, "stale lease token: the activity was redelivered or timed out")
	}
	return r, r.activities[tok.Sched], nil
}

// CompleteActivityTask appends ActivityTaskStarted and ActivityTaskCompleted
// (and a WorkflowTaskScheduled when needed), then acknowledges the lease. A
// stale lease token is FAILED_PRECONDITION and the result is dropped.
func (s *Server) CompleteActivityTask(ctx context.Context, req *durablev1.CompleteActivityRequest) (*durablev1.Empty, error) {
	tok, err := decodeToken(req.GetTaskToken())
	if err != nil {
		return nil, err
	}
	if lt := req.GetLeaseToken(); lt != 0 {
		tok.Lease = lt
	}
	if len(req.GetResult()) > MaxPayloadBytes {
		return nil, status.Errorf(codes.InvalidArgument, "activity result is %d bytes; the limit is %d", len(req.GetResult()), MaxPayloadBytes)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r, a, err := s.pendingActivity(tok)
	if err != nil {
		return nil, err
	}
	b := &batch{r: r}
	st := s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActStarted{ActStarted: &durablev1.ActivityTaskStarted{
		ScheduledEventId: a.scheduledID, Identity: req.GetIdentity(), Attempt: a.attempt, LeaseToken: tok.Lease,
	}}})
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActCompleted{ActCompleted: &durablev1.ActivityTaskCompleted{
		ScheduledEventId: a.scheduledID, StartedEventId: st.GetEventId(), Result: req.GetResult(), Identity: req.GetIdentity(),
	}}})
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	s.o.Queue.Complete(ctx, tok.lease())
	return &durablev1.Empty{}, nil
}
