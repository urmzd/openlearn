package dur_06

// Record mode (maintainers only): DUR06_RECORD=<dir> go test ./dur_06/
// runs every course test workflow against the server in this process with a
// fake clock, driving workflow and activity tasks one at a time so each
// history is a deterministic function of the code, and writes one JSON file
// per run (workflow.HistoryFile). course/oracle/dur/record-histories.sh
// runs it against the reference.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/encoding/protojson"

	dlog "tinyllm/durable/log"
	"tinyllm/durable/queue"
	"tinyllm/durable/server"
	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
	"openlearn.urmzd.com/tl/testkit/clock"
)

func TestMain(m *testing.M) {
	if dir := os.Getenv("DUR06_RECORD"); dir != "" {
		if err := record(dir); err != nil {
			fmt.Fprintln(os.Stderr, "record:", err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// driver runs tasks one at a time: a pending workflow task first, else one
// activity task, until nothing is pending.
type driver struct {
	tasks durablev1.TaskServiceClient
	wfs   durablev1.WorkflowServiceClient
	q     *queue.Queue
	clk   *clock.Fake
	reg   *workflow.Registry
	acts  map[string]worker.ActivityFunc
}

func (d *driver) run(ctx context.Context) error {
	for {
		d.clk.Advance(time.Second)
		if v, _, _ := d.q.Depth("wf:default"); v > 0 {
			wt, err := d.tasks.PollWorkflowTask(ctx, &durablev1.PollRequest{TaskQueue: "default", Identity: "recorder"})
			if err != nil {
				return err
			}
			h := wt.History
			if len(wt.NextPageToken) > 0 {
				st, err := d.wfs.GetHistory(ctx, &durablev1.GetHistoryRequest{WorkflowId: wt.WorkflowId, RunId: wt.RunId, NextPageToken: wt.NextPageToken})
				if err != nil {
					return err
				}
				for {
					ev, err := st.Recv()
					if errors.Is(err, io.EOF) {
						break
					}
					if err != nil {
						return err
					}
					h = append(h, ev)
				}
			}
			cmds, err := d.reg.HandleWorkflowTask(ctx, wt, h)
			if err != nil {
				return fmt.Errorf("%s: %w", wt.WorkflowId, err)
			}
			if _, err := d.tasks.CompleteWorkflowTask(ctx, &durablev1.CompleteWorkflowTaskRequest{TaskToken: wt.TaskToken, Commands: cmds, Identity: "recorder"}); err != nil {
				return err
			}
			continue
		}
		if v, _, _ := d.q.Depth("act:default"); v > 0 {
			at, err := d.tasks.PollActivityTask(ctx, &durablev1.PollRequest{TaskQueue: "default", Identity: "recorder"})
			if err != nil {
				return err
			}
			res, err := d.acts[at.ActivityType](ctx, at.Input)
			if err != nil {
				return err
			}
			if _, err := d.tasks.CompleteActivityTask(ctx, &durablev1.CompleteActivityRequest{TaskToken: at.TaskToken, LeaseToken: at.LeaseToken, Result: res, Identity: "recorder"}); err != nil {
				return err
			}
			continue
		}
		return nil
	}
}

func record(dir string) error {
	ctx := context.Background()
	tmp, err := os.MkdirTemp("", "dur06-record")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)
	clk := clock.NewFake(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC))
	lg, err := dlog.Open(tmp, dlog.Options{Sync: fastSync})
	if err != nil {
		return err
	}
	q, err := queue.Open(ctx, queue.Options{Log: lg, Clock: clk})
	if err != nil {
		return err
	}
	ids := 0
	srv, err := server.Open(ctx, server.Options{Log: lg, Queue: q, Clock: clk,
		NewRunID: func() string { ids++; return fmt.Sprintf("fx-run-%d", ids) }})
	if err != nil {
		return err
	}
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return err
	}
	gs := grpc.NewServer()
	durablev1.RegisterWorkflowServiceServer(gs, srv)
	durablev1.RegisterTaskServiceServer(gs, srv)
	go gs.Serve(ln)
	defer gs.Stop()
	conn, err := grpc.NewClient(ln.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	d := &driver{tasks: durablev1.NewTaskServiceClient(conn), wfs: durablev1.NewWorkflowServiceClient(conn), q: q, clk: clk, acts: activities}
	before := registry()
	before.Register("Versioned", VersionedBefore)
	for _, sc := range []struct {
		file, id, typ, input string
		reg                  *workflow.Registry
	}{
		{"sequence", "seq-3", "Sequence", "3", registry()},
		{"parallel", "par-1", "Parallel", "null", registry()},
		{"side-effects", "side-1", "SideEffects", "null", registry()},
		{"versioned-new", "versioned-new", "Versioned", "null", registry()},
		{"versioned-before", "versioned-before", "Versioned", "null", before},
		{"reorder", "reorder-1", "Reorder", "null", registry()},
		{"clock", "clock-1", "Clock", "null", registry()},
		{"counter", "counter", "Counter", "0", registry()},
	} {
		if _, err := d.wfs.StartWorkflow(ctx, &durablev1.StartWorkflowRequest{WorkflowId: sc.id, WorkflowType: sc.typ, TaskQueue: "default", Input: []byte(sc.input)}); err != nil {
			return err
		}
		d.reg = sc.reg
		if err := d.run(ctx); err != nil {
			return err
		}
		runs, err := d.wfs.ListWorkflows(ctx, &durablev1.ListWorkflowsRequest{})
		if err != nil {
			return err
		}
		var mine []*durablev1.WorkflowInfo
		for _, w := range runs.Workflows {
			if w.WorkflowId == sc.id {
				mine = append([]*durablev1.WorkflowInfo{w}, mine...) // oldest first
			}
		}
		for i, w := range mine {
			name := sc.file
			if len(mine) > 1 {
				name = fmt.Sprintf("%s-%d", sc.file, i+1)
			}
			if err := writeRun(ctx, d.wfs, filepath.Join(dir, name+".json"), w); err != nil {
				return err
			}
		}
	}
	return nil
}

func writeRun(ctx context.Context, wfs durablev1.WorkflowServiceClient, path string, w *durablev1.WorkflowInfo) error {
	st, err := wfs.GetHistory(ctx, &durablev1.GetHistoryRequest{WorkflowId: w.WorkflowId, RunId: w.RunId})
	if err != nil {
		return err
	}
	f := workflow.HistoryFile{WorkflowID: w.WorkflowId, RunID: w.RunId, WorkflowType: w.WorkflowType}
	for {
		ev, err := st.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
		raw, err := protojson.Marshal(ev)
		if err != nil {
			return err
		}
		var v any // protojson whitespace is unstable on purpose; normalize it
		if err := json.Unmarshal(raw, &v); err != nil {
			return err
		}
		norm, _ := json.Marshal(v)
		f.Events = append(f.Events, norm)
	}
	out, err := json.MarshalIndent(f, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(out, '\n'), 0o644)
}
