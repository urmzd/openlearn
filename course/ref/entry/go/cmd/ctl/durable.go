package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	durablev1 "supersource.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

const (
	rpcTimeout   = 30 * time.Second
	pollInterval = 250 * time.Millisecond
	// unavailableGrace is how long --wait polls through a server that
	// answers Unavailable (down, restarting) before it gives up.
	unavailableGrace = 2 * time.Minute
	identity         = "ctl"
)

// durable is a tl.durable.v1 WorkflowService client.
type durable struct {
	conn *grpc.ClientConn
	wf   durablev1.WorkflowServiceClient
}

func dial(addr string) (*durable, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, usagef("--durable %s: %v", addr, err)
	}
	return &durable{conn: conn, wf: durablev1.NewWorkflowServiceClient(conn)}, nil
}

func (d *durable) Close() { _ = d.conn.Close() }

// rpcErr is an RPC failure as the role rules print it: the gRPC code, then
// the server's message.
func rpcErr(op string, err error) error {
	st := status.Convert(err)
	return fmt.Errorf("%s: %s: %s", op, st.Code(), st.Message())
}

// statusName drops the proto prefix: RUNNING, COMPLETED, FAILED, ...
func statusName(s durablev1.WorkflowStatus) string {
	return strings.TrimPrefix(s.String(), "WORKFLOW_STATUS_")
}

// start calls StartWorkflow with a JSON input.
func (d *durable) start(wfType, id, queue string, input []byte) (*durablev1.StartWorkflowResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), rpcTimeout)
	defer cancel()
	resp, err := d.wf.StartWorkflow(ctx, &durablev1.StartWorkflowRequest{
		WorkflowId: id, WorkflowType: wfType, TaskQueue: queue, Input: input, Identity: identity,
	})
	if err != nil {
		return nil, rpcErr("StartWorkflow", err)
	}
	return resp, nil
}

// describe calls DescribeWorkflow on the current run of id.
func (d *durable) describe(id string) (*durablev1.WorkflowInfo, error) {
	info, err := d.describeRPC(id)
	if err != nil {
		return nil, rpcErr("DescribeWorkflow", err)
	}
	return info, nil
}

// describeRPC is DescribeWorkflow with the gRPC error as it came.
func (d *durable) describeRPC(id string) (*durablev1.WorkflowInfo, error) {
	ctx, cancel := context.WithTimeout(context.Background(), rpcTimeout)
	defer cancel()
	return d.wf.DescribeWorkflow(ctx, &durablev1.DescribeWorkflowRequest{WorkflowId: id})
}

// wait describes the current run of id until it is no longer RUNNING. The
// current run follows ContinueAsNew, so a continued workflow is waited for
// to its end. A server that is down while ctl waits (killed and
// restarting) answers Unavailable; wait keeps polling through that for up
// to unavailableGrace in a row, since the run lives on in the server's log.
func (d *durable) wait(id string) (*durablev1.WorkflowInfo, error) {
	var downSince time.Time
	for {
		info, err := d.describeRPC(id)
		if err != nil {
			if status.Code(err) != codes.Unavailable {
				return nil, rpcErr("DescribeWorkflow", err)
			}
			if downSince.IsZero() {
				downSince = time.Now()
			} else if time.Since(downSince) > unavailableGrace {
				return nil, rpcErr("DescribeWorkflow", err)
			}
			time.Sleep(pollInterval)
			continue
		}
		downSince = time.Time{}
		if info.GetStatus() != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
			return info, nil
		}
		time.Sleep(pollInterval)
	}
}

// run starts a workflow and, with wait, waits for its end. It returns the
// start response and the run as last described.
func (d *durable) run(wfType, id, queue string, input []byte, wait bool) (*durablev1.StartWorkflowResponse, *durablev1.WorkflowInfo, error) {
	resp, err := d.start(wfType, id, queue, input)
	if err != nil {
		return nil, nil, err
	}
	var info *durablev1.WorkflowInfo
	if wait {
		info, err = d.wait(id)
	} else {
		info, err = d.describe(id)
	}
	return resp, info, err
}

// resultValue is a run's result as a JSON value: null until COMPLETED, the
// bytes as JSON when they parse, else as a string.
func resultValue(info *durablev1.WorkflowInfo) any {
	b := info.GetResult()
	if info.GetStatus() != durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED || len(b) == 0 {
		return nil
	}
	if json.Valid(b) {
		return json.RawMessage(b)
	}
	return string(b)
}

// notCompleted is the error of a waited run that did not complete: its
// status and failure message.
func notCompleted(info *durablev1.WorkflowInfo) error {
	if info.GetStatus() == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED {
		return nil
	}
	msg := info.GetFailure().GetMessage()
	if msg == "" {
		msg = "no failure recorded"
	}
	return fmt.Errorf("workflow %s is %s: %s", info.GetWorkflowId(), statusName(info.GetStatus()), msg)
}

// readArtifact returns a file under the worker's artifacts root. Activity
// results are paths (spec/subprocess-activity.md), and only the worker sees
// the files, so ctl runs the worker's ReadArtifact workflow on the queue the
// parent run used. Its id is derived from the parent run and the path, so a
// repeated call returns the first read.
func (d *durable) readArtifact(parent *durablev1.WorkflowInfo, path string) ([]byte, error) {
	sum := sha256.Sum256([]byte(path))
	id := fmt.Sprintf("%s/read/%s/%s", parent.GetWorkflowId(), parent.GetRunId(), hex.EncodeToString(sum[:4]))
	input, err := json.Marshal(path)
	if err != nil {
		return nil, err
	}
	if _, err := d.start("ReadArtifact", id, parent.GetTaskQueue(), input); err != nil {
		return nil, err
	}
	info, err := d.wait(id)
	if err != nil {
		return nil, err
	}
	if err := notCompleted(info); err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	return info.GetResult(), nil
}
