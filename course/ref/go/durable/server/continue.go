package server

// The server side of the workflow SDK (dur.06): markers (SideEffect and
// GetVersion record their results as MarkerRecorded) and ContinueAsNew,
// which closes a run and starts the next one under the same workflow id, so
// a long-lived workflow's history stays bounded.

import (
	"context"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

func init() {
	registerCommand("record_marker", recordMarker)
	registerCommand("continue_as_new", continueAsNew)
}

func recordMarker(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	// SOLUTION-BEGIN dur.06
	m := c.GetRecordMarker()
	if m.GetMarkerName() == "" {
		return status.Error(codes.InvalidArgument, "marker_name is required")
	}
	if len(m.GetDetails()) > MaxPayloadBytes {
		return status.Errorf(codes.InvalidArgument, "marker details are %d bytes; the limit is %d", len(m.GetDetails()), MaxPayloadBytes)
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Marker{Marker: &durablev1.MarkerRecorded{
		MarkerName: m.GetMarkerName(), Details: m.GetDetails(), WorkflowTaskCompletedEventId: completed,
	}}})
	return nil
	// SOLUTION-END
}

// continueAsNew closes the run with WorkflowExecutionContinuedAsNew naming a
// fresh run id; once that is durable, the new run starts under the same
// workflow id (recovery starts it if the server dies in between).
func continueAsNew(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	// SOLUTION-BEGIN dur.06
	can := c.GetContinueAsNew()
	if len(can.GetInput()) > MaxPayloadBytes {
		return status.Errorf(codes.InvalidArgument, "continue-as-new input is %d bytes; the limit is %d", len(can.GetInput()), MaxPayloadBytes)
	}
	r := b.r
	ev := &durablev1.WorkflowExecutionContinuedAsNew{
		NewRunId: s.o.NewRunID(), Input: can.GetInput(),
		WorkflowType: can.GetWorkflowType(), TaskQueue: can.GetTaskQueue(),
	}
	if ev.WorkflowType == "" {
		ev.WorkflowType = r.wfType
	}
	if ev.TaskQueue == "" {
		ev.TaskQueue = r.taskQueue
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_Continued{Continued: ev}})
	b.after = append(b.after, func(ctx context.Context) error {
		_, err := s.startRun(ctx, r.workflowID, ev.GetNewRunId(), continuedStart(r, ev))
		return err
	})
	return nil
	// SOLUTION-END
}
