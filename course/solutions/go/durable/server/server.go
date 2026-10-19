// Package server is the durable execution server: the gRPC WorkflowService
// and TaskService of contracts/proto/tl/durable/v1/durable.proto over the
// dur.01 log and the dur.03 queue.
//
// The design is event sourcing. A run's history (stream "run/<run_id>" in the
// log, one WalRecord per event) is the only source of truth. Everything else
// is derived from it by two functions in this file:
//
//	apply  folds one event into the run's state (status, pending workflow
//	       task, pending activities, timers)
//	effect turns one event into its side effect (enqueue a workflow task or
//	       an activity task, arm or cancel a timer)
//
// An rpc builds a batch of events, commits it (one atomic log append), then
// applies and runs the effects. Recovery replays every history through
// apply and re-runs the effects of whatever is still pending; the queue
// ignores a task it already holds, so nothing is doubled.
//
// Files and owners: server.go and workflows.go (dur.02), tasks.go (dur.04),
// activities.go (dur.05), continue.go (dur.06), timers.go (dur.07).
package server

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	mrand "math/rand/v2"
	"sort"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	dlog "tinyllm/durable/log"
	"tinyllm/durable/queue"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// Limits from durable.proto.
const (
	MaxPayloadBytes          = 2 << 20  // activity input and result, workflow input and result
	ContinueAsNewEvents      = 10000    // past this the workflow task suggests ContinueAsNew
	ContinueAsNewBytes       = 32 << 20 // or past this many history bytes
	MaxHistoryEvents         = 20000    // a run's history never grows past this
	FirstPageBytes           = 1 << 20  // a WorkflowTask carries at most this much history
	DefaultPollWait          = 30 * time.Second
	DefaultWorkflowTaskLease = 10 * time.Second
)

// Clock is the time seam; the testkit's *clock.Fake satisfies it.
type Clock = queue.Clock

// TimerKey names one durable timer of one run.
type TimerKey struct{ RunID, TimerID string }

// Timers is the durable timer service (dur.07). The server arms a timer for
// every pending TimerStarted event (at commit and on recovery) and cancels
// it on TimerCanceled; the service calls Server.FireTimer when one is due.
type Timers interface {
	Schedule(key TimerKey, at time.Time)
	Cancel(key TimerKey)
}

// Options configures a Server. Log and Queue are required; the queue must be
// opened over the same log.
type Options struct {
	Log                 *dlog.Log
	Queue               *queue.Queue
	Clock               Clock
	Timers              Timers        // nil: timers are recorded but never fire
	NewRunID            func() string // nil: 32 random hex digits
	PollWait            time.Duration // long-poll limit of Poll*Task; 0 = 30 s
	WorkflowTaskTimeout time.Duration // lease of a workflow task; 0 = 10 s
	DLQAfterAttempts    int           // [durable].dlq_after_attempts for RetryPolicy.max_attempts 0; 0 = 5
	Seed                uint64        // seeds the retry jitter (dur.05)
	// Jitter maps a backoff to the delay actually used (dur.05); nil = full
	// jitter, a uniform draw in [0, d] from the Seed stream.
	Jitter func(d time.Duration) time.Duration
}

type wall struct{}

func (wall) Now() time.Time                         { return time.Now() }
func (wall) After(d time.Duration) <-chan time.Time { return time.After(d) }

// Server implements durablev1.WorkflowServiceServer and durablev1.TaskServiceServer.
type Server struct {
	durablev1.UnimplementedWorkflowServiceServer
	durablev1.UnimplementedTaskServiceServer

	mu      sync.Mutex
	o       Options
	runs    map[string]*run   // by run id
	current map[string]string // workflow id -> the run id of its newest run
	origin  map[string]string // workflow id -> the run id of its first run
	rng     *mrand.Rand       // retry jitter, drawn under mu
}

type run struct {
	id, workflowID   string
	wfType           string
	taskQueue        string
	input            []byte
	identity         string
	traceContext     map[string]string
	history          []*durablev1.HistoryEvent
	bytes            int64
	status           durablev1.WorkflowStatus
	startMs, closeMs int64
	result           []byte
	failure          *durablev1.Failure
	continuedFrom    string
	continuedAs      string

	wtScheduled int64 // event id of the pending WorkflowTaskScheduled; 0 = none
	wtStarted   int64 // event id of its latest WorkflowTaskStarted; 0 = not started
	needsWT     bool  // a trigger event arrived while a workflow task was in flight

	activities      map[int64]*activity // pending, by ActivityTaskScheduled event id
	timers          map[string]*timer   // pending, by timer id
	cancelRequested bool
	signals         map[string]bool // request ids already received (dur.08)
}

type activity struct {
	scheduledID     int64
	attrs           *durablev1.ActivityTaskScheduled
	attempt         int32
	deadLettered    bool
	cancelRequested bool
	lastFailure     *durablev1.Failure
	lastHeartbeat   []byte
	lastHeartbeatMs int64
}

type timer struct {
	id        string
	startedID int64
	fireAtMs  int64
}

// Open recovers every run from the log, re-enqueues pending workflow and
// activity tasks, and re-arms pending timers.
func Open(ctx context.Context, o Options) (*Server, error) {
	if o.Log == nil || o.Queue == nil {
		return nil, errors.New("server: Options.Log and Options.Queue are required")
	}
	if o.Clock == nil {
		o.Clock = wall{}
	}
	if o.NewRunID == nil {
		o.NewRunID = randomID
	}
	if o.PollWait <= 0 {
		o.PollWait = DefaultPollWait
	}
	if o.WorkflowTaskTimeout <= 0 {
		o.WorkflowTaskTimeout = DefaultWorkflowTaskLease
	}
	if o.DLQAfterAttempts <= 0 {
		o.DLQAfterAttempts = 5
	}
	s := &Server{
		o: o, runs: map[string]*run{}, current: map[string]string{}, origin: map[string]string{},
		rng: mrand.New(mrand.NewPCG(o.Seed, 0x6475722e3035)),
	}
	for _, stream := range o.Log.Streams("run/") {
		evs, err := o.Log.Read(ctx, stream, 1, 0)
		if err != nil {
			return nil, err
		}
		r := &run{id: strings.TrimPrefix(stream, "run/")}
		for _, e := range evs {
			var rec durablev1.WalRecord
			if err := proto.Unmarshal(e.Data, &rec); err != nil {
				return nil, fmt.Errorf("server: %s event %d: %w", stream, e.Version, err)
			}
			r.workflowID = rec.GetWorkflowId()
			apply(r, rec.GetEvent())
		}
		s.runs[r.id] = r
	}
	for _, r := range s.runs {
		if r.continuedFrom == "" {
			s.origin[r.workflowID] = r.id
		}
		if r.continuedAs == "" {
			if cur, ok := s.current[r.workflowID]; !ok || s.runs[cur].startMs <= r.startMs {
				s.current[r.workflowID] = r.id
			}
		}
	}
	for _, r := range s.runs {
		if err := s.resume(ctx, r); err != nil {
			return nil, err
		}
	}
	// A crash between a ContinuedAsNew append and the next run's first
	// append leaves a run that names a successor that does not exist.
	for _, r := range s.runs {
		if c := r.history[len(r.history)-1].GetContinued(); c != nil && s.runs[c.GetNewRunId()] == nil {
			if _, err := s.startRun(ctx, r.workflowID, c.GetNewRunId(), continuedStart(r, c)); err != nil {
				return nil, err
			}
		}
	}
	return s, nil
}

// continuedStart is the WorkflowExecutionStarted of the run that r
// continues as: the same workflow id, the new input.
func continuedStart(r *run, c *durablev1.WorkflowExecutionContinuedAsNew) *durablev1.WorkflowExecutionStarted {
	return &durablev1.WorkflowExecutionStarted{
		WorkflowType: c.GetWorkflowType(), TaskQueue: c.GetTaskQueue(), Input: c.GetInput(),
		TraceContext: r.traceContext, ContinuedFromRunId: r.id, Identity: r.identity,
	}
}

// resume re-runs the side effects of whatever r still has pending.
func (s *Server) resume(ctx context.Context, r *run) error {
	if r.status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		return nil
	}
	if r.wtScheduled != 0 {
		if err := s.enqueueWorkflowTask(ctx, r, r.wtScheduled); err != nil {
			return err
		}
	}
	ids := make([]int64, 0, len(r.activities))
	for id := range r.activities {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	for _, id := range ids {
		if a := r.activities[id]; !a.deadLettered {
			if err := s.enqueueActivity(ctx, r, a); err != nil {
				return err
			}
		}
	}
	if s.o.Timers != nil {
		for _, t := range r.timers {
			s.o.Timers.Schedule(TimerKey{r.id, t.id}, time.UnixMilli(t.fireAtMs))
		}
	}
	return nil
}

func randomID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(b[:])
}

// Kind is the name of an event's oneof field ("started", "wt_scheduled", ...).
func Kind(e *durablev1.HistoryEvent) string {
	m := e.ProtoReflect()
	fd := m.WhichOneof(m.Descriptor().Oneofs().ByName("attrs"))
	if fd == nil {
		return ""
	}
	return string(fd.Name())
}

// trigger reports whether e is news the workflow must react to, so a
// workflow task must follow it.
func trigger(e *durablev1.HistoryEvent) bool {
	switch Kind(e) {
	case "started", "act_completed", "act_failed", "act_timed_out", "act_canceled",
		"timer_fired", "signal", "cancel_requested":
		return true
	}
	return false
}

// closing reports whether e ends the run.
func closing(e *durablev1.HistoryEvent) bool {
	switch Kind(e) {
	case "completed", "failed", "canceled", "continued":
		return true
	}
	return false
}

// apply folds one event into r. It is the only code that changes run state,
// at commit time and during recovery alike.
func apply(r *run, e *durablev1.HistoryEvent) {
	r.history = append(r.history, e)
	r.bytes += int64(proto.Size(e))
	if trigger(e) && r.wtScheduled != 0 && r.wtStarted != 0 {
		r.needsWT = true
	}
	switch a := e.GetAttrs().(type) {
	case *durablev1.HistoryEvent_Started:
		r.wfType, r.taskQueue, r.input = a.Started.GetWorkflowType(), a.Started.GetTaskQueue(), a.Started.GetInput()
		r.traceContext, r.continuedFrom, r.identity = a.Started.GetTraceContext(), a.Started.GetContinuedFromRunId(), a.Started.GetIdentity()
		r.status, r.startMs = durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING, e.GetTsUnixMs()
		r.activities, r.timers, r.signals = map[int64]*activity{}, map[string]*timer{}, map[string]bool{}
	case *durablev1.HistoryEvent_WtScheduled:
		r.wtScheduled, r.wtStarted, r.needsWT = e.GetEventId(), 0, false
	case *durablev1.HistoryEvent_WtStarted:
		r.wtStarted = e.GetEventId()
	case *durablev1.HistoryEvent_WtCompleted:
		r.wtScheduled, r.wtStarted = 0, 0
	case *durablev1.HistoryEvent_ActScheduled:
		r.activities[e.GetEventId()] = &activity{scheduledID: e.GetEventId(), attrs: a.ActScheduled}
	case *durablev1.HistoryEvent_ActStarted:
		if act := r.activities[a.ActStarted.GetScheduledEventId()]; act != nil {
			act.attempt = a.ActStarted.GetAttempt()
		}
	case *durablev1.HistoryEvent_ActCompleted:
		delete(r.activities, a.ActCompleted.GetScheduledEventId())
	case *durablev1.HistoryEvent_ActFailed:
		delete(r.activities, a.ActFailed.GetScheduledEventId())
	case *durablev1.HistoryEvent_ActTimedOut:
		delete(r.activities, a.ActTimedOut.GetScheduledEventId())
	case *durablev1.HistoryEvent_ActCanceled:
		delete(r.activities, a.ActCanceled.GetScheduledEventId())
	case *durablev1.HistoryEvent_DeadLettered:
		if act := r.activities[a.DeadLettered.GetScheduledEventId()]; act != nil {
			act.deadLettered, act.lastFailure = true, a.DeadLettered.GetLastFailure()
		}
	case *durablev1.HistoryEvent_ActCancelRequested:
		if act := r.activities[a.ActCancelRequested.GetScheduledEventId()]; act != nil {
			act.cancelRequested = true
		}
	case *durablev1.HistoryEvent_TimerStarted:
		r.timers[a.TimerStarted.GetTimerId()] = &timer{id: a.TimerStarted.GetTimerId(), startedID: e.GetEventId(), fireAtMs: a.TimerStarted.GetFireAtUnixMs()}
	case *durablev1.HistoryEvent_TimerFired:
		delete(r.timers, a.TimerFired.GetTimerId())
	case *durablev1.HistoryEvent_TimerCanceled:
		delete(r.timers, a.TimerCanceled.GetTimerId())
	case *durablev1.HistoryEvent_Signal:
		if id := a.Signal.GetRequestId(); id != "" {
			r.signals[id] = true
		}
	case *durablev1.HistoryEvent_CancelRequested:
		r.cancelRequested = true
	case *durablev1.HistoryEvent_Completed:
		r.close(e, durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED)
		r.result = a.Completed.GetResult()
	case *durablev1.HistoryEvent_Failed:
		r.close(e, durablev1.WorkflowStatus_WORKFLOW_STATUS_FAILED)
		r.failure = a.Failed.GetFailure()
	case *durablev1.HistoryEvent_Canceled:
		r.close(e, durablev1.WorkflowStatus_WORKFLOW_STATUS_CANCELED)
	case *durablev1.HistoryEvent_Continued:
		r.close(e, durablev1.WorkflowStatus_WORKFLOW_STATUS_CONTINUED_AS_NEW)
		r.continuedAs = a.Continued.GetNewRunId()
	}
}

func (r *run) close(e *durablev1.HistoryEvent, st durablev1.WorkflowStatus) {
	r.status, r.closeMs = st, e.GetTsUnixMs()
	r.wtScheduled, r.wtStarted, r.needsWT = 0, 0, false
}

// effect is the side effect of one committed event.
func (s *Server) effect(ctx context.Context, r *run, e *durablev1.HistoryEvent) error {
	switch a := e.GetAttrs().(type) {
	case *durablev1.HistoryEvent_WtScheduled:
		return s.enqueueWorkflowTask(ctx, r, e.GetEventId())
	case *durablev1.HistoryEvent_ActScheduled:
		if act := r.activities[e.GetEventId()]; act != nil {
			return s.enqueueActivity(ctx, r, act)
		}
	case *durablev1.HistoryEvent_TimerStarted:
		if s.o.Timers != nil {
			s.o.Timers.Schedule(TimerKey{r.id, a.TimerStarted.GetTimerId()}, time.UnixMilli(a.TimerStarted.GetFireAtUnixMs()))
		}
	case *durablev1.HistoryEvent_TimerCanceled:
		if s.o.Timers != nil {
			s.o.Timers.Cancel(TimerKey{r.id, a.TimerCanceled.GetTimerId()})
		}
	}
	return nil
}

// Queue names: workflow and activity tasks of one task queue live apart.
func workflowQueue(taskQueue string) string { return "wf:" + taskQueue }
func activityQueue(taskQueue string) string { return "act:" + taskQueue }

// Task ids: "<run_id>/wt/<scheduled event id>" and "<run_id>/a/<scheduled event id>".
func taskID(runID, kind string, scheduled int64) string {
	return fmt.Sprintf("%s/%s/%d", runID, kind, scheduled)
}

func parseTaskID(id string) (runID, kind string, scheduled int64, err error) {
	parts := strings.Split(id, "/")
	if len(parts) != 3 {
		return "", "", 0, fmt.Errorf("server: bad task id %q", id)
	}
	if _, err := fmt.Sscan(parts[2], &scheduled); err != nil {
		return "", "", 0, fmt.Errorf("server: bad task id %q", id)
	}
	return parts[0], parts[1], scheduled, nil
}

func (s *Server) enqueueWorkflowTask(ctx context.Context, r *run, scheduled int64) error {
	return s.o.Queue.Enqueue(ctx, workflowQueue(r.taskQueue), queue.Task{
		ID: taskID(r.id, "wt", scheduled), MaxAttempts: 1 << 30, Visibility: s.o.WorkflowTaskTimeout,
	})
}

// enqueueActivity queues one delivery stream of a scheduled activity: its
// lease is the heartbeat timeout when there is one, else start-to-close, and
// it dead-letters after the retry policy's max_attempts.
func (s *Server) enqueueActivity(ctx context.Context, r *run, a *activity) error {
	o := a.attrs.GetOptions()
	lease := time.Duration(o.GetStartToCloseMs()) * time.Millisecond
	if hb := o.GetHeartbeatTimeoutMs(); hb > 0 {
		lease = time.Duration(hb) * time.Millisecond
	}
	max := int(o.GetRetry().GetMaxAttempts())
	if max <= 0 {
		max = s.o.DLQAfterAttempts
	}
	tq := o.GetTaskQueue()
	if tq == "" {
		tq = r.taskQueue
	}
	return s.o.Queue.Enqueue(ctx, activityQueue(tq), queue.Task{
		ID: taskID(r.id, "a", a.scheduledID), MaxAttempts: max, Visibility: lease,
	})
}

// batch is the events one rpc appends to one run, atomically, plus actions
// to run once they are durable (ContinueAsNew starts the next run there).
type batch struct {
	r      *run
	events []*durablev1.HistoryEvent
	after  []func(ctx context.Context) error
}

// add stamps e with the next event id and the current time.
func (s *Server) add(b *batch, e *durablev1.HistoryEvent) *durablev1.HistoryEvent {
	e.EventId = int64(len(b.r.history) + len(b.events) + 1)
	e.TsUnixMs = s.o.Clock.Now().UnixMilli()
	b.events = append(b.events, e)
	return e
}

// commit appends b to the run's stream (expected version = the history
// length, so two commits can never interleave), then applies every event and
// runs its effects (enqueue, arm a timer). Before appending it adds a WorkflowTaskScheduled when the
// run stays open, something needs the workflow's attention, and no workflow
// task will be pending to see it.
func (s *Server) commit(ctx context.Context, b *batch) error {
	r := b.r
	var trig, completes, closes, schedules bool
	for _, e := range b.events {
		trig = trig || trigger(e)
		closes = closes || closing(e)
		switch Kind(e) {
		case "wt_completed":
			completes = true
		case "wt_scheduled":
			schedules = true
		}
	}
	pendingAfter := (r.wtScheduled != 0 && !completes) || schedules
	open := !closes && (r.status == durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || Kind(b.events[0]) == "started")
	if open && !pendingAfter && (trig || (completes && r.needsWT)) {
		s.add(b, &durablev1.HistoryEvent{Attrs: &durablev1.HistoryEvent_WtScheduled{
			WtScheduled: &durablev1.WorkflowTaskScheduled{TaskQueue: r.taskQueueOr(b), Attempt: 1},
		}})
	}
	if n := len(r.history) + len(b.events); n > MaxHistoryEvents {
		return status.Errorf(codes.FailedPrecondition, "run %s would reach %d events (limit %d): continue as new", r.id, n, MaxHistoryEvents)
	}
	evs := make([]dlog.Event, len(b.events))
	for i, e := range b.events {
		data, err := proto.Marshal(&durablev1.WalRecord{WorkflowId: r.workflowID, RunId: r.id, Record: &durablev1.WalRecord_Event{Event: e}})
		if err != nil {
			return status.Errorf(codes.Internal, "encode event: %v", err)
		}
		evs[i] = dlog.Event{Type: Kind(e), Data: data}
	}
	if _, err := s.o.Log.Append(ctx, "run/"+r.id, int64(len(r.history)), evs...); err != nil {
		return logErr(err)
	}
	for _, e := range b.events {
		apply(r, e)
	}
	for _, e := range b.events {
		// The events are durable, so the rpc has succeeded whatever happens
		// here; an effect that fails (the queue's append hit the quota) is
		// redone by resume after the next restart.
		_ = s.effect(ctx, r, e)
	}
	for _, f := range b.after {
		_ = f(ctx) // the same: recovery repairs what did not happen
	}
	return nil
}

// taskQueueOr is the run's task queue, or the one its first event names
// (a batch that starts the run is not applied yet).
func (r *run) taskQueueOr(b *batch) string {
	if r.taskQueue != "" {
		return r.taskQueue
	}
	return b.events[0].GetStarted().GetTaskQueue()
}

// logErr maps storage errors to gRPC codes: the WAL quota is RESOURCE_EXHAUSTED.
func logErr(err error) error {
	if _, ok := status.FromError(err); ok && status.Code(err) != codes.Unknown {
		return err
	}
	if errors.Is(err, dlog.ErrQuota) {
		return status.Error(codes.ResourceExhausted, err.Error())
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return status.FromContextError(err).Err()
	}
	return status.Error(codes.Unavailable, err.Error())
}

// PageToken is the token of the page that starts at event id next.
func PageToken(next int64) []byte {
	return binary.BigEndian.AppendUint64(nil, uint64(next))
}

// ParsePageToken is the inverse of PageToken; an empty token is event 1.
func ParsePageToken(tok []byte) (int64, error) {
	if len(tok) == 0 {
		return 1, nil
	}
	if len(tok) != 8 {
		return 0, status.Error(codes.InvalidArgument, "bad page token")
	}
	next := int64(binary.BigEndian.Uint64(tok))
	if next < 1 {
		return 0, status.Error(codes.InvalidArgument, "bad page token")
	}
	return next, nil
}

// lookup finds a run by workflow id (and run id; "" = the newest run).
func (s *Server) lookup(workflowID, runID string) (*run, error) {
	if runID == "" {
		runID = s.current[workflowID]
	}
	r := s.runs[runID]
	if r == nil || r.workflowID != workflowID {
		return nil, status.Errorf(codes.NotFound, "workflow %q run %q not found", workflowID, runID)
	}
	return r, nil
}
