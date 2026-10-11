package server

// The WorkflowService rpcs the CLI calls (dur.02): StartWorkflow (idempotent
// on workflow_id), DescribeWorkflow, GetHistory (paged), ListWorkflows.

import (
	"bytes"
	"context"
	"sort"
	"strconv"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// StartWorkflow creates a run, or returns the existing one: while any run
// with this workflow_id exists, the same type and input give its newest
// run_id with started = false, and anything else is FAILED_PRECONDITION.
func (s *Server) StartWorkflow(ctx context.Context, req *durablev1.StartWorkflowRequest) (*durablev1.StartWorkflowResponse, error) {
	// SOLUTION-BEGIN dur.02
	switch {
	case req.GetWorkflowId() == "":
		return nil, status.Error(codes.InvalidArgument, "workflow_id is required")
	case req.GetWorkflowType() == "":
		return nil, status.Error(codes.InvalidArgument, "workflow_type is required")
	case req.GetTaskQueue() == "":
		return nil, status.Error(codes.InvalidArgument, "task_queue is required")
	case len(req.GetInput()) > MaxPayloadBytes:
		return nil, status.Errorf(codes.InvalidArgument, "input is %d bytes; the limit is %d (pass large data by path)", len(req.GetInput()), MaxPayloadBytes)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if first, ok := s.origin[req.GetWorkflowId()]; ok {
		o := s.runs[first]
		if o.wfType != req.GetWorkflowType() || !bytes.Equal(o.input, req.GetInput()) {
			return nil, status.Errorf(codes.FailedPrecondition,
				"workflow %q already exists with a different type or input", req.GetWorkflowId())
		}
		return &durablev1.StartWorkflowResponse{RunId: s.current[req.GetWorkflowId()], Started: false}, nil
	}
	r, err := s.startRun(ctx, req.GetWorkflowId(), s.o.NewRunID(), &durablev1.WorkflowExecutionStarted{
		WorkflowType: req.GetWorkflowType(), TaskQueue: req.GetTaskQueue(), Input: req.GetInput(),
		TraceContext: req.GetTraceContext(), Identity: req.GetIdentity(),
	})
	if err != nil {
		return nil, err
	}
	s.origin[r.workflowID] = r.id
	return &durablev1.StartWorkflowResponse{RunId: r.id, Started: true}, nil
	// SOLUTION-END
}

// startRun commits a new run's first events (WorkflowExecutionStarted, then
// the first WorkflowTaskScheduled) and registers it as the workflow's newest
// run. The caller holds s.mu. ContinueAsNew (dur.06) and recovery use it too.
func (s *Server) startRun(ctx context.Context, workflowID, runID string, started *durablev1.WorkflowExecutionStarted) (*run, error) {
	// SOLUTION-BEGIN dur.02
	r := &run{id: runID, workflowID: workflowID}
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Started{Started: started}})
	if err := s.commit(ctx, b); err != nil {
		return nil, err
	}
	s.runs[r.id] = r
	s.current[workflowID] = r.id
	return r, nil
	// SOLUTION-END
}

// DescribeWorkflow reports a run (the newest when run_id is empty).
func (s *Server) DescribeWorkflow(ctx context.Context, req *durablev1.DescribeWorkflowRequest) (*durablev1.WorkflowInfo, error) {
	// SOLUTION-BEGIN dur.02
	s.mu.Lock()
	defer s.mu.Unlock()
	r, err := s.lookup(req.GetWorkflowId(), req.GetRunId())
	if err != nil {
		return nil, err
	}
	return s.info(r), nil
	// SOLUTION-END
}

func (s *Server) info(r *run) *durablev1.WorkflowInfo {
	// SOLUTION-BEGIN dur.02
	inf := &durablev1.WorkflowInfo{
		WorkflowId: r.workflowID, RunId: r.id, WorkflowType: r.wfType, TaskQueue: r.taskQueue,
		Status: r.status, StartUnixMs: r.startMs, CloseUnixMs: r.closeMs,
		HistoryLength: int64(len(r.history)), HistoryBytes: r.bytes,
		ContinuedFromRunId: r.continuedFrom, ContinuedAsRunId: r.continuedAs,
		Result: r.result, Failure: r.failure,
	}
	ids := make([]int64, 0, len(r.activities))
	for id := range r.activities {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		a := r.activities[id]
		inf.PendingActivities = append(inf.PendingActivities, &durablev1.PendingActivity{
			ActivityId: a.attrs.GetActivityId(), ActivityType: a.attrs.GetActivityType(), Attempt: a.attempt,
			DeadLettered: a.deadLettered, LastHeartbeatDetails: a.lastHeartbeat,
			LastHeartbeatUnixMs: a.lastHeartbeatMs, LastFailure: a.lastFailure,
		})
	}
	return inf
	// SOLUTION-END
}

// GetHistory streams a run's events in event_id order, from the event the
// page token names (the first event when it is empty).
func (s *Server) GetHistory(req *durablev1.GetHistoryRequest, stream grpc.ServerStreamingServer[durablev1.HistoryEvent]) error {
	// SOLUTION-BEGIN dur.02
	next, err := ParsePageToken(req.GetNextPageToken())
	if err != nil {
		return err
	}
	s.mu.Lock()
	r, err := s.lookup(req.GetWorkflowId(), req.GetRunId())
	var evs []*durablev1.HistoryEvent
	if err == nil && next <= int64(len(r.history)) {
		evs = append(evs, r.history[next-1:]...)
	}
	s.mu.Unlock()
	if err != nil {
		return err
	}
	for _, e := range evs {
		if err := stream.Send(e); err != nil {
			return err
		}
	}
	return nil
	// SOLUTION-END
}

// ListWorkflows lists runs newest start first (ties by workflow id, then run
// id), filtered by status and type, page_size at a time (0 = 100).
func (s *Server) ListWorkflows(ctx context.Context, req *durablev1.ListWorkflowsRequest) (*durablev1.ListWorkflowsResponse, error) {
	// SOLUTION-BEGIN dur.02
	off := 0
	if tok := req.GetPageToken(); len(tok) > 0 {
		n, err := strconv.Atoi(string(tok))
		if err != nil || n < 0 {
			return nil, status.Error(codes.InvalidArgument, "bad page token")
		}
		off = n
	}
	size := int(req.GetPageSize())
	if size <= 0 {
		size = 100
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	var rs []*run
	for _, r := range s.runs {
		if st := req.GetStatus(); st != durablev1.WorkflowStatus_WORKFLOW_STATUS_UNSPECIFIED && r.status != st {
			continue
		}
		if t := req.GetWorkflowType(); t != "" && r.wfType != t {
			continue
		}
		rs = append(rs, r)
	}
	sort.Slice(rs, func(i, j int) bool {
		if rs[i].startMs != rs[j].startMs {
			return rs[i].startMs > rs[j].startMs
		}
		if rs[i].workflowID != rs[j].workflowID {
			return rs[i].workflowID < rs[j].workflowID
		}
		return rs[i].id < rs[j].id
	})
	resp := &durablev1.ListWorkflowsResponse{}
	if off >= len(rs) {
		return resp, nil
	}
	end := off + size
	if end < len(rs) {
		resp.NextPageToken = []byte(strconv.Itoa(end))
	} else {
		end = len(rs)
	}
	for _, r := range rs[off:end] {
		resp.Workflows = append(resp.Workflows, s.info(r))
	}
	return resp, nil
	// SOLUTION-END
}
