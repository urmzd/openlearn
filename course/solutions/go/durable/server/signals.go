package server

// Signals and cancellation on the server (dur.08): the SignalWorkflow and
// CancelWorkflow rpcs, and the commands a workflow answers a cancellation
// with (RequestCancelActivity, CancelWorkflowExecution).
//
// Both rpcs only append history. A signal is a SignalReceived event; a
// cancel is a WorkflowExecutionCancelRequested event. Either is news the
// workflow must react to, so commit schedules a workflow task, or marks the
// one in flight to be followed by another (needsWT): a signal that arrives
// while the workflow is busy, or before it ever waits on the channel, sits in
// the history until a workflow task delivers it. Nothing is ever pushed to a
// worker, so nothing can be lost between the server and the worker.

import (
	"context"
	"sort"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

func init() {
	registerCommand("cancel_activity", cancelActivity)
	registerCommand("cancel_workflow", cancelWorkflow)
}

// SignalWorkflow appends SignalReceived to the run (the current run when
// run_id is empty). A non-empty request_id makes it idempotent: a second
// signal with the same id is acknowledged and appends nothing, across
// restarts too (the ids are part of the run's state, rebuilt by apply).
// NOT_FOUND for an unknown or closed run; INVALID_ARGUMENT for an empty
// signal name or an input over 2 MiB.
func (s *Server) SignalWorkflow(ctx context.Context, req *durablev1.SignalWorkflowRequest) (*durablev1.SignalWorkflowResponse, error) {
	if req.GetSignalName() == "" {
		return nil, status.Error(codes.InvalidArgument, "signal_name is required")
	}
	if len(req.GetInput()) > MaxPayloadBytes {
		return nil, status.Errorf(codes.InvalidArgument, "signal input is %d bytes; the limit is %d", len(req.GetInput()), MaxPayloadBytes)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.lookup(req.GetWorkflowId(), req.GetRunId())
	if err != nil {
		return nil, err
	}
	if r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		return nil, status.Errorf(codes.NotFound, "workflow %q run %q is closed", r.workflowID, r.id)
	}
	if id := req.GetRequestId(); id != "" && r.signals[id] {
		return &durablev1.SignalWorkflowResponse{}, nil
	}
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Signal{Signal: &durablev1.SignalReceived{
		SignalName: req.GetSignalName(), Input: req.GetInput(), RequestId: req.GetRequestId(),
	}}})
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	return &durablev1.SignalWorkflowResponse{}, nil
}

// CancelWorkflow asks a run to cancel: it appends
// WorkflowExecutionCancelRequested once (a repeated request, or one for a
// run that already ended canceled, appends nothing). Running activities
// learn it from their next heartbeat (cancel_requested); the workflow learns
// it from its next workflow task, runs its compensations, and ends with
// CancelWorkflowExecution. NOT_FOUND for an unknown run;
// FAILED_PRECONDITION for a run that completed, failed, or continued.
func (s *Server) CancelWorkflow(ctx context.Context, req *durablev1.CancelWorkflowRequest) (*durablev1.CancelWorkflowResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.lookup(req.GetWorkflowId(), req.GetRunId())
	if err != nil {
		return nil, err
	}
	switch r.status {
	case durablev1.WorkflowStatus_WORKFLOW_STATUS_CANCELED:
		return &durablev1.CancelWorkflowResponse{}, nil
	case durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING:
	default:
		return nil, status.Errorf(codes.FailedPrecondition, "workflow %q run %q already closed (%s)", r.workflowID, r.id, r.status)
	}
	if r.cancelRequested {
		return &durablev1.CancelWorkflowResponse{}, nil
	}
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_CancelRequested{CancelRequested: &durablev1.WorkflowExecutionCancelRequested{
		Reason: req.GetReason(),
	}}})
	// Every activity pending now is asked to stop, by name, in the same
	// batch. Activities the workflow schedules afterwards (its compensations)
	// are not: they must run to completion.
	ids := make([]int64, 0, len(r.activities))
	for id := range r.activities {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		requestCancel(s, b, r.activities[id])
	}
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	return &durablev1.CancelWorkflowResponse{}, nil
}

// cancelActivity handles RequestCancelActivity. It appends
// ActivityTaskCancelRequested for a pending activity, which the attempt in
// flight sees as cancel_requested on its next heartbeat. A dead-lettered
// activity has no attempt in flight to tell, so it is canceled at once
// (ActivityTaskCanceled). An activity that is no longer pending (it finished
// in a batch this workflow task has not seen) or was already asked is left
// alone: cancellation races completion, and completion may win.
func cancelActivity(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	id := c.GetCancelActivity().GetActivityId()
	if id == "" {
		return status.Error(codes.InvalidArgument, "activity_id is required")
	}
	for _, a := range b.r.activities {
		if a.attrs.GetActivityId() != id {
			continue
		}
		requestCancel(s, b, a)
		return nil
	}
	return nil
}

// requestCancel appends ActivityTaskCancelRequested for a, once, and for a
// dead-lettered activity (no attempt in flight to tell) ActivityTaskCanceled
// right after it.
func requestCancel(s *Server, b *batch, a *activity) {
	if a.cancelRequested || pendingIn(b, a.scheduledID) {
		return
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActCancelRequested{ActCancelRequested: &durablev1.ActivityTaskCancelRequested{
		ScheduledEventId: a.scheduledID,
	}}})
	if a.deadLettered {
		s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActCanceled{ActCanceled: &durablev1.ActivityTaskCanceled{
			ScheduledEventId: a.scheduledID,
		}}})
	}
}

// pendingIn reports whether b already asks to cancel the activity scheduled
// at sched (a workflow that sends the command twice in one task).
func pendingIn(b *batch, sched int64) bool {
	for _, e := range b.events {
		if e.GetActCancelRequested().GetScheduledEventId() == sched {
			return true
		}
	}
	return false
}

// cancelWorkflow handles CancelWorkflowExecution, which ends the run as
// CANCELED. It is valid only after a cancel was requested: a workflow cannot
// cancel itself (it fails or completes instead).
func cancelWorkflow(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	if !b.r.cancelRequested {
		return status.Error(codes.FailedPrecondition, "CancelWorkflowExecution without a cancel request: fail or complete the workflow instead")
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Canceled{Canceled: &durablev1.WorkflowExecutionCanceled{
		Details: c.GetCancelWorkflow().GetDetails(),
	}}})
	return nil
}
