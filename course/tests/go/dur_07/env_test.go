package dur_07

// The test environment: a real dur.01 log in a temp dir, a dur.03 queue over
// it, your server on a loopback gRPC port, and a fake clock. crash() drops
// the server and closes the log without any shutdown logic of yours running;
// start() recovers from the same directory.

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
	tclk    *armClock         // the timer service's view of clk
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
	e.tclk = &armClock{Fake: e.clk, armed: map[int64]bool{}}
	var tclk timer.Clock = e.tclk
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

// armClock is the timer service's view of the fake clock. It records every
// deadline the service waits on, so a test advances time only once the
// service waits on the deadline it expects. BlockUntil cannot tell: the shared
// fake clock also holds the waiters of crashed instances and of finished
// polls, and a service that read Now just before an Advance and calls After
// just after it would wait d past the new time.
type armClock struct {
	*clock.Fake
	mu    sync.Mutex
	armed map[int64]bool // absolute deadlines, UnixNano
}

func (c *armClock) After(d time.Duration) <-chan time.Time {
	at := c.Fake.Now().Add(d)
	ch := c.Fake.After(d)
	c.mu.Lock()
	c.armed[at.UnixNano()] = true
	c.mu.Unlock()
	return ch
}

func (c *armClock) has(at time.Time) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.armed[at.UnixNano()]
}

// waitArmed blocks until the current timer service waits on the clock for
// exactly at.
func (e *env) waitArmed(at time.Time) {
	e.t.Helper()
	e.eventually(fmt.Sprintf("the timer service to wait for +%v", at.Sub(t0)), func() bool { return e.tclk.has(at) })
}

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
