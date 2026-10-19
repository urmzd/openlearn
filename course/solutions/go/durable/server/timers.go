package server

// Durable timers on the server (dur.07): StartTimer and CancelTimer commands,
// and FireTimer, which the timer service calls when a timer is due. The timer
// is durable because TimerStarted is in history: commit arms it, recovery
// re-arms every pending one, and FireTimer appends TimerFired only for a
// timer that is still pending, so each fires exactly once.
//
// Wiring (in your server's main):
//
//	ts := timer.New[server.TimerKey](clk)
//	srv, _ := server.Open(ctx, server.Options{..., Timers: ts})
//	go ts.Run(ctx, srv.FireTimer)

import (
	"context"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

func init() {
	registerCommand("start_timer", startTimer)
	registerCommand("cancel_timer", cancelTimer)
}

func startTimer(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	st := c.GetStartTimer()
	if st.GetTimerId() == "" || st.GetDurationMs() <= 0 {
		return status.Error(codes.InvalidArgument, "start_timer needs a timer_id and a duration_ms > 0")
	}
	if b.r.timers[st.GetTimerId()] != nil {
		return status.Errorf(codes.InvalidArgument, "timer %q is already pending", st.GetTimerId())
	}
	for _, e := range b.events {
		if e.GetTimerStarted().GetTimerId() == st.GetTimerId() {
			return status.Errorf(codes.InvalidArgument, "timer %q started twice", st.GetTimerId())
		}
	}
	fireAt := s.o.Clock.Now().Add(time.Duration(st.GetDurationMs()) * time.Millisecond)
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_TimerStarted{TimerStarted: &durablev1.TimerStarted{
		TimerId: st.GetTimerId(), DurationMs: st.GetDurationMs(), FireAtUnixMs: fireAt.UnixMilli(),
		WorkflowTaskCompletedEventId: completed,
	}}})
	return nil
}

func cancelTimer(s *Server, b *batch, completed int64, c *durablev1.Command) error {
	id := c.GetCancelTimer().GetTimerId()
	t := b.r.timers[id]
	if t == nil {
		return status.Errorf(codes.InvalidArgument, "timer %q is not pending", id)
	}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_TimerCanceled{TimerCanceled: &durablev1.TimerCanceled{
		TimerId: id, StartedEventId: t.startedID,
	}}})
	return nil
}

// FireTimer appends TimerFired (and the workflow task it triggers) for a
// timer that is still pending. A fired, canceled, or unknown timer, or a
// closed run, is a no-op. If the append fails the timer is re-armed a
// second later, so it is never lost.
func (s *Server) FireTimer(k TimerKey) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := s.runs[k.RunID]
	if r == nil || r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		return
	}
	t := r.timers[k.TimerID]
	if t == nil {
		return
	}
	b := &batch{r: r}
	s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_TimerFired{TimerFired: &durablev1.TimerFired{
		TimerId: t.id, StartedEventId: t.startedID,
	}}})
	if err := s.commit(context.Background(), b); err != nil && s.o.Timers != nil {
		s.o.Timers.Schedule(k, s.o.Clock.Now().Add(time.Second))
	}
}
