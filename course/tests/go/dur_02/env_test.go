package dur_02

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
	timers   server.Timers
}

func newEnv(t *testing.T) *env {
	e := &env{t: t, dir: t.TempDir(), clk: clock.NewFake(t0)}
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
	if e.q, err = queue.Open(bg, queue.Options{Log: e.log, Clock: e.clk}); err != nil {
		t.Fatalf("queue.Open: %v", err)
	}
	e.srv, err = server.Open(bg, server.Options{
		Log: e.log, Queue: e.q, Clock: e.clk, PollWait: time.Second, Timers: e.timers,
		NewRunID: func() string { e.ids++; return fmt.Sprintf("run-%d", e.ids) },
	})
	if err != nil {
		t.Fatalf("server.Open: %v", err)
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	e.gs = grpc.NewServer(grpc.UnaryInterceptor(recoverPanics), grpc.StreamInterceptor(recoverStream))
	durablev1.RegisterWorkflowServiceServer(e.gs, e.srv)
	durablev1.RegisterTaskServiceServer(e.gs, e.srv)
	go e.gs.Serve(ln)
	if e.conn, err = grpc.NewClient(ln.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials())); err != nil {
		t.Fatal(err)
	}
	e.wf = durablev1.NewWorkflowServiceClient(e.conn)
	return e
}

func (e *env) crash() {
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
