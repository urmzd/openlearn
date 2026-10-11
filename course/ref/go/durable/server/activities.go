package server

// Activity failure handling on the server (dur.05): heartbeats, retries with
// exponential backoff and full jitter, non-retryable failures, the
// dead-letter queue, and redrive.

import (
	"context"
	"errors"
	"slices"
	"strconv"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	dact "tinyllm/durable/activity"
	"tinyllm/durable/queue"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// RecordHeartbeat extends a live attempt's lease by its heartbeat timeout and
// stores details, which a later attempt receives as last_heartbeat_details.
// A stale lease token is FAILED_PRECONDITION. cancel_requested is true once
// this activity was asked to stop (ActivityTaskCancelRequested, dur.08), not
// merely because its run has a cancel request: compensations scheduled after
// the cancel must run to the end.
func (s *Server) RecordHeartbeat(ctx context.Context, req *durablev1.HeartbeatRequest) (*durablev1.HeartbeatResponse, error) {
	// SOLUTION-BEGIN dur.05
	tok, err := decodeToken(req.GetTaskToken())
	if err != nil {
		return nil, err
	}
	if lt := req.GetLeaseToken(); lt != 0 {
		tok.Lease = lt
	}
	if len(req.GetDetails()) > MaxPayloadBytes {
		return nil, status.Error(codes.InvalidArgument, "heartbeat details over 2 MiB")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_, a, err := s.pendingActivity(tok)
	if err != nil {
		return nil, err
	}
	l, err := s.o.Queue.Heartbeat(ctx, tok.lease(), req.GetDetails())
	if errors.Is(err, queue.ErrLeaseLost) {
		return nil, status.Error(codes.FailedPrecondition, "stale lease token")
	}
	if err != nil {
		return nil, logErr(err)
	}
	a.lastHeartbeat, a.lastHeartbeatMs = req.GetDetails(), s.o.Clock.Now().UnixMilli()
	return &durablev1.HeartbeatResponse{
		CancelRequested: a.cancelRequested,
		DeadlineUnixMs:  l.Deadline.UnixMilli(),
	}, nil
	// SOLUTION-END
}

// retryDelay is the jittered backoff after attempt n failed.
func (s *Server) retryDelay(p *durablev1.RetryPolicy, n int) time.Duration {
	// SOLUTION-BEGIN dur.05
	d := dact.Backoff(p, n)
	if s.o.Jitter != nil {
		return s.o.Jitter(d)
	}
	return dact.Jitter(d, s.rng.Float64())
	// SOLUTION-END
}

// FailActivityTask ends an attempt with a failure. A failure marked
// non-retryable, or whose type the policy lists in non_retryable, appends
// ActivityTaskStarted and ActivityTaskFailed (the workflow is told). Any
// other is retried after the jittered backoff; the attempt that reaches
// max_attempts parks the task in the DLQ and appends ActivityDeadLettered,
// and the workflow keeps waiting for a redrive.
func (s *Server) FailActivityTask(ctx context.Context, req *durablev1.FailActivityRequest) (*durablev1.Empty, error) {
	// SOLUTION-BEGIN dur.05
	tok, err := decodeToken(req.GetTaskToken())
	if err != nil {
		return nil, err
	}
	if lt := req.GetLeaseToken(); lt != 0 {
		tok.Lease = lt
	}
	f := req.GetFailure()
	if f == nil {
		f = &durablev1.Failure{Message: "unknown failure", Type: "Error"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	r, a, err := s.pendingActivity(tok)
	if err != nil {
		return nil, err
	}
	if d := req.GetLastHeartbeatDetails(); len(d) > 0 {
		a.lastHeartbeat = d
	}
	policy := a.attrs.GetOptions().GetRetry()
	if f.GetNonRetryable() || slices.Contains(policy.GetNonRetryable(), f.GetType()) {
		b := &batch{r: r}
		st := s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActStarted{ActStarted: &durablev1.ActivityTaskStarted{
			ScheduledEventId: a.scheduledID, Identity: req.GetIdentity(), Attempt: a.attempt, LeaseToken: tok.Lease,
		}}})
		s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_ActFailed{ActFailed: &durablev1.ActivityTaskFailed{
			ScheduledEventId: a.scheduledID, StartedEventId: st.GetEventId(), Failure: f,
		}}})
		if err := s.commit(ctx, b); err != nil {
			return nil, err
		}
		s.o.Queue.Fail(ctx, tok.lease(), queue.TaskError{Message: f.GetMessage(), NonRetryable: true})
		return &durablev1.Empty{}, nil
	}
	disp, err := s.o.Queue.Fail(ctx, tok.lease(), queue.TaskError{Message: f.GetMessage(), RetryAfter: s.retryDelay(policy, int(a.attempt))})
	if errors.Is(err, queue.ErrLeaseLost) {
		return nil, status.Error(codes.FailedPrecondition, "stale lease token")
	}
	if err != nil {
		return nil, logErr(err)
	}
	a.lastFailure = f
	if disp == queue.DeadLettered {
		b := &batch{r: r}
		s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_DeadLettered{DeadLettered: &durablev1.ActivityDeadLettered{
			ScheduledEventId: a.scheduledID, TaskQueue: tok.Queue, TaskId: tok.TaskID, Attempts: a.attempt, LastFailure: f,
		}}})
		if err := s.commit(ctx, b); err != nil {
			return nil, err
		}
	}
	return &durablev1.Empty{}, nil
	// SOLUTION-END
}

// ListDeadLetters lists the dead-lettered activity tasks of a task queue.
func (s *Server) ListDeadLetters(ctx context.Context, req *durablev1.ListDeadLettersRequest) (*durablev1.ListDeadLettersResponse, error) {
	// SOLUTION-BEGIN dur.05
	dl, err := s.o.Queue.DLQ(ctx, activityQueue(req.GetTaskQueue()))
	if err != nil {
		return nil, logErr(err)
	}
	off := 0
	if tok := req.GetPageToken(); len(tok) > 0 {
		if off, err = strconv.Atoi(string(tok)); err != nil || off < 0 {
			return nil, status.Error(codes.InvalidArgument, "bad page token")
		}
	}
	size := int(req.GetPageSize())
	if size <= 0 {
		size = 100
	}
	resp := &durablev1.ListDeadLettersResponse{}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := off; i < len(dl) && i < off+size; i++ {
		d := dl[i]
		out := &durablev1.DeadLetter{
			TaskId: d.Task.ID, Attempts: int32(d.Attempts), DeadLetteredUnixMs: d.At.UnixMilli(),
			LastFailure: &durablev1.Failure{Message: d.LastError},
		}
		if runID, _, sched, err := parseTaskID(d.Task.ID); err == nil {
			if r := s.runs[runID]; r != nil {
				out.WorkflowId, out.RunId = r.workflowID, r.id
				if a := r.activities[sched]; a != nil {
					out.ActivityId, out.ActivityType = a.attrs.GetActivityId(), a.attrs.GetActivityType()
					if a.lastFailure != nil {
						out.LastFailure = a.lastFailure
					}
				}
			}
		}
		resp.DeadLetters = append(resp.DeadLetters, out)
	}
	if off+size < len(dl) {
		resp.NextPageToken = []byte(strconv.Itoa(off + size))
	}
	return resp, nil
	// SOLUTION-END
}

// RedriveDeadLetter moves dead-lettered tasks (all of the queue's when
// task_ids is empty) back to their queue with attempt 1.
func (s *Server) RedriveDeadLetter(ctx context.Context, req *durablev1.RedriveDeadLetterRequest) (*durablev1.RedriveDeadLetterResponse, error) {
	// SOLUTION-BEGIN dur.05
	ids := req.GetTaskIds()
	if len(ids) == 0 {
		dl, err := s.o.Queue.DLQ(ctx, activityQueue(req.GetTaskQueue()))
		if err != nil {
			return nil, logErr(err)
		}
		for _, d := range dl {
			ids = append(ids, d.Task.ID)
		}
	}
	n, err := s.o.Queue.Redrive(ctx, activityQueue(req.GetTaskQueue()), ids...)
	if errors.Is(err, queue.ErrNotFound) {
		return nil, status.Error(codes.NotFound, err.Error())
	}
	if err != nil {
		return nil, logErr(err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		if runID, _, sched, err := parseTaskID(id); err == nil {
			if r := s.runs[runID]; r != nil && r.activities[sched] != nil {
				r.activities[sched].deadLettered = false
			}
		}
	}
	return &durablev1.RedriveDeadLetterResponse{Redriven: int32(n)}, nil
	// SOLUTION-END
}
