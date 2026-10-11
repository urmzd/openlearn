package dur_08

// The test environment (the one dur.07's tests use): a real dur.01 log in a
// temp dir, a dur.03 queue over it, the server on a loopback gRPC port with
// the dur.07 timer service, and a fake clock. crash() drops the server and
// closes the log without any shutdown logic running; start() recovers from
// the same directory. The server tests play the worker by hand over the
// TaskService; the end-to-end tests run a real worker and the workflow SDK.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	dlog "tinyllm/durable/log"
	"tinyllm/durable/queue"
	"tinyllm/durable/server"
	"tinyllm/durable/timer"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
	"openlearn.urmzd.com/tl/testkit/clock"
)

var bg = context.Background()

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func fastSync(f *os.File) error { return syscall.Fsync(int(f.Fd())) }

type env struct {
	t        *testing.T
	dir      string
	clk      *clock.Fake
	maxBytes int64
	ids      int
	log      *dlog.Log
	q        *queue.Queue
	srv      *server.Server
	gs       *grpc.Server
	conn     *grpc.ClientConn
	wf       durablev1.WorkflowServiceClient
	tasks    durablev1.TaskServiceClient
	addr     string // fixed across restarts, so workers can reconnect
	pollWait time.Duration
	polls    atomic.Int32 // Poll*Task rpcs in progress
	real     bool         // wall clock instead of the fake one (kill loops)
	wtLease  time.Duration
	jitter   func(time.Duration) time.Duration

	ts      *timer.Service[server.TimerKey] // the timer service of the current process
	stopTS  context.CancelFunc
	tsDone  chan struct{}
	firedMu sync.Mutex
	fired   []server.TimerKey // every FireTimer call, across restarts
}

func newEnv(t *testing.T) *env {
	e := &env{t: t, dir: t.TempDir(), clk: clock.NewFake(t0), pollWait: 30 * time.Second}
	t.Cleanup(e.crash)
	return e
}

// recoverPanics turns a panic in a handler (a stubbed function) into an
// Internal error instead of killing the test binary.
func recoverPanics(ctx context.Context, req any, _ *grpc.UnaryServerInfo, h grpc.UnaryHandler) (resp any, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = status.Errorf(codes.Internal, "panic: %v", r)
		}
	}()
	return h(ctx, req)
}

func recoverStream(srv any, ss grpc.ServerStream, _ *grpc.StreamServerInfo, h grpc.StreamHandler) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = status.Errorf(codes.Internal, "panic: %v", r)
		}
	}()
	return h(srv, ss)
}

func (e *env) start() *env {
	t := e.t
	t.Helper()
	var err error
	if e.log, err = dlog.Open(e.dir, dlog.Options{Sync: fastSync, MaxBytes: e.maxBytes}); err != nil {
		t.Fatalf("log.Open: %v", err)
	}
	var clk queue.Clock = e.clk
	if e.real {
		clk = nil
	}
	if e.q, err = queue.Open(bg, queue.Options{Log: e.log, Clock: clk}); err != nil {
		t.Fatalf("queue.Open: %v", err)
	}
	var tclk timer.Clock = e.clk
	if e.real {
		tclk = wallClock{}
	}
	e.ts = timer.New[server.TimerKey](tclk)
	e.srv, err = server.Open(bg, server.Options{
		Log: e.log, Queue: e.q, Clock: clk, PollWait: e.pollWait, WorkflowTaskTimeout: e.wtLease, Jitter: e.jitter, Timers: e.ts,
		NewRunID: func() string { e.ids++; return fmt.Sprintf("run-%d", e.ids) },
	})
	if err != nil {
		t.Fatalf("server.Open: %v", err)
	}
	addr := e.addr
	if addr == "" {
		addr = "127.0.0.1:0"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatal(err)
	}
	e.addr = ln.Addr().String()
	countPolls := func(ctx context.Context, req any, info *grpc.UnaryServerInfo, h grpc.UnaryHandler) (any, error) {
		if strings.Contains(info.FullMethod, "/Poll") {
			e.polls.Add(1)
			defer e.polls.Add(-1)
		}
		return recoverPanics(ctx, req, info, h)
	}
	e.gs = grpc.NewServer(grpc.UnaryInterceptor(countPolls), grpc.StreamInterceptor(recoverStream))
	durablev1.RegisterWorkflowServiceServer(e.gs, e.srv)
	durablev1.RegisterTaskServiceServer(e.gs, e.srv)
	go e.gs.Serve(ln)
	tctx, cancel := context.WithCancel(bg)
	e.stopTS, e.tsDone = cancel, make(chan struct{})
	srv, done, ts := e.srv, e.tsDone, e.ts
	go func() {
		defer close(done)
		ts.Run(tctx, func(k server.TimerKey) {
			e.firedMu.Lock()
			e.fired = append(e.fired, k)
			e.firedMu.Unlock()
			srv.FireTimer(k)
		})
	}()
	if e.conn, err = grpc.NewClient(ln.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials())); err != nil {
		t.Fatal(err)
	}
	e.wf = durablev1.NewWorkflowServiceClient(e.conn)
	e.tasks = durablev1.NewTaskServiceClient(e.conn)
	return e
}

func (e *env) crash() {
	if e.stopTS != nil {
		e.stopTS()
		<-e.tsDone
		e.stopTS = nil
	}
	if e.conn != nil {
		e.conn.Close()
		e.conn = nil
	}
	if e.gs != nil {
		e.gs.Stop()
		e.gs = nil
	}
	if e.log != nil {
		e.log.Close()
		e.log = nil
	}
	e.srv, e.q = nil, nil
}

func (e *env) restart() *env { e.crash(); return e.start() }

type wallClock struct{}

func (wallClock) Now() time.Time                         { return time.Now() }
func (wallClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

func (e *env) firedKeys() []server.TimerKey {
	e.firedMu.Lock()
	defer e.firedMu.Unlock()
	return append([]server.TimerKey(nil), e.fired...)
}

func startReq(id, typ, input string) *durablev1.StartWorkflowRequest {
	return &durablev1.StartWorkflowRequest{WorkflowId: id, WorkflowType: typ, TaskQueue: "default", Input: []byte(input), Identity: "test"}
}

func (e *env) startWF(id, typ, input string) *durablev1.StartWorkflowResponse {
	e.t.Helper()
	resp, err := e.wf.StartWorkflow(bg, startReq(id, typ, input))
	if err != nil {
		e.t.Fatalf("StartWorkflow(%s): %v", id, err)
	}
	return resp
}

func (e *env) history(id, runID string, token []byte) ([]*durablev1.HistoryEvent, error) {
	st, err := e.wf.GetHistory(bg, &durablev1.GetHistoryRequest{WorkflowId: id, RunId: runID, NextPageToken: token})
	if err != nil {
		return nil, err
	}
	var out []*durablev1.HistoryEvent
	for {
		ev, err := st.Recv()
		if errors.Is(err, io.EOF) {
			return out, nil
		}
		if err != nil {
			return out, err
		}
		out = append(out, ev)
	}
}

func (e *env) mustHistory(id string) []*durablev1.HistoryEvent {
	e.t.Helper()
	h, err := e.history(id, "", nil)
	if err != nil {
		e.t.Fatalf("GetHistory(%s): %v", id, err)
	}
	return h
}

func (e *env) describe(id string) *durablev1.WorkflowInfo {
	e.t.Helper()
	inf, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: id})
	if err != nil {
		e.t.Fatalf("DescribeWorkflow(%s): %v", id, err)
	}
	return inf
}

// kindOf is the event's oneof field name, computed here so a wrong
// server.Kind cannot agree with itself.
func kindOf(ev *durablev1.HistoryEvent) string {
	m := ev.ProtoReflect()
	if fd := m.WhichOneof(m.Descriptor().Oneofs().ByName("attrs")); fd != nil {
		return string(fd.Name())
	}
	return ""
}

func kinds(h []*durablev1.HistoryEvent) string {
	var out []string
	for _, ev := range h {
		out = append(out, kindOf(ev))
	}
	return strings.Join(out, " ")
}

func code(err error) codes.Code { return status.Code(err) }

// guard bounds an rpc that long-polls the fake clock.
func guard(t *testing.T) context.Context {
	c, cancel := context.WithTimeout(bg, 5*time.Second)
	t.Cleanup(cancel)
	return c
}

func (e *env) pollWT() *durablev1.WorkflowTask {
	e.t.Helper()
	wt, err := e.tasks.PollWorkflowTask(guard(e.t), &durablev1.PollRequest{TaskQueue: "default", Identity: "wk"})
	if err != nil || len(wt.TaskToken) == 0 {
		e.t.Fatalf("PollWorkflowTask: %v (token %q)", err, wt.GetTaskToken())
	}
	return wt
}

func (e *env) pollAT() *durablev1.ActivityTask {
	e.t.Helper()
	at, err := e.tasks.PollActivityTask(guard(e.t), &durablev1.PollRequest{TaskQueue: "default", Identity: "wk"})
	if err != nil || len(at.TaskToken) == 0 {
		e.t.Fatalf("PollActivityTask: %v (token %q)", err, at.GetTaskToken())
	}
	return at
}

func (e *env) completeWT(wt *durablev1.WorkflowTask, cmds ...*durablev1.Command) error {
	_, err := e.tasks.CompleteWorkflowTask(bg, &durablev1.CompleteWorkflowTaskRequest{TaskToken: wt.TaskToken, Commands: cmds, Identity: "wk"})
	return err
}

func (e *env) mustCompleteWT(wt *durablev1.WorkflowTask, cmds ...*durablev1.Command) {
	e.t.Helper()
	if err := e.completeWT(wt, cmds...); err != nil {
		e.t.Fatalf("CompleteWorkflowTask: %v", err)
	}
}

func (e *env) heartbeat(at *durablev1.ActivityTask) (*durablev1.HeartbeatResponse, error) {
	return e.tasks.RecordHeartbeat(bg, &durablev1.HeartbeatRequest{TaskToken: at.TaskToken, LeaseToken: at.LeaseToken})
}

func (e *env) signal(id, name, input, reqID string) error {
	_, err := e.wf.SignalWorkflow(bg, &durablev1.SignalWorkflowRequest{WorkflowId: id, SignalName: name, Input: []byte(input), RequestId: reqID})
	return err
}

func (e *env) cancel(id string) error {
	_, err := e.wf.CancelWorkflow(bg, &durablev1.CancelWorkflowRequest{WorkflowId: id, Reason: "test"})
	return err
}

func schedule(id, typ string, input []byte) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_ScheduleActivity{ScheduleActivity: &durablev1.ScheduleActivity{
		ActivityId: id, ActivityType: typ, Input: input,
		Options: &durablev1.ActivityOptions{StartToCloseMs: 10000, HeartbeatTimeoutMs: 5000},
	}}}
}

func cancelActivityCmd(id string) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_CancelActivity{CancelActivity: &durablev1.RequestCancelActivity{ActivityId: id}}}
}

func cancelWorkflowCmd(details string) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_CancelWorkflow{CancelWorkflow: &durablev1.CancelWorkflowExecution{Details: []byte(details)}}}
}

func completeCmd(result string) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_Complete{Complete: &durablev1.CompleteWorkflow{Result: []byte(result)}}}
}

// count is how many events of kind k the history holds.
func count(h []*durablev1.HistoryEvent, k string) int {
	n := 0
	for _, ev := range h {
		if kindOf(ev) == k {
			n++
		}
	}
	return n
}
