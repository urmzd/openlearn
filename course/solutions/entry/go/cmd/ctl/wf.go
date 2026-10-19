package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// The wf verbs: workflows and dead letters on tl.durable.v1.
func init() {
	dlq := map[string]verb{"list": wfDLQList, "redrive": wfDLQRedrive}
	wf := map[string]verb{
		"start":    wfStart,
		"describe": wfDescribe,
		"history":  wfHistory,
		"dlq":      func(args []string) error { return subverbs("wf dlq", dlq, args) },
	}
	register("wf", func(args []string) error { return subverbs("wf", wf, args) })
}

// wfStart starts a workflow of any registered type; with --wait it waits
// for the run's end. Final line: {workflow_id, run_id, started, status,
// result}, result null until COMPLETED.
func wfStart(args []string) error {
	fs := newFlags("wf start")
	id := fs.String("id", "", "workflow id")
	input := fs.String("input-json", "null", "workflow input, JSON")
	queue := fs.String("queue", "default", "task queue")
	wait := fs.Bool("wait", false, "wait until the run is not RUNNING")
	addr := durableFlag(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("wf start", pos, 1, "one workflow type"); err != nil {
		return err
	}
	if *id == "" {
		return usagef("wf start: --id is required")
	}
	if !json.Valid([]byte(*input)) {
		return usagef("wf start: --input-json is not JSON: %s", *input)
	}
	d, err := dial(*addr)
	if err != nil {
		return err
	}
	defer d.Close()
	resp, info, err := d.run(pos[0], *id, *queue, []byte(*input), *wait)
	if err != nil {
		return err
	}
	if err := emit(map[string]any{
		"workflow_id": *id, "run_id": resp.GetRunId(), "started": resp.GetStarted(),
		"status": statusName(info.GetStatus()), "result": resultValue(info),
	}); err != nil {
		return err
	}
	if *wait {
		return notCompleted(info)
	}
	return nil
}

// wfDescribe prints a run's state. Final line: {workflow_id, run_id,
// status, history_length, pending_activities, dead_lettered} (counts).
func wfDescribe(args []string) error {
	fs := newFlags("wf describe")
	addr := durableFlag(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("wf describe", pos, 1, "one workflow id"); err != nil {
		return err
	}
	d, err := dial(*addr)
	if err != nil {
		return err
	}
	defer d.Close()
	info, err := d.describe(pos[0])
	if err != nil {
		return err
	}
	dead := 0
	for _, p := range info.GetPendingActivities() {
		if p.GetDeadLettered() {
			dead++
		}
	}
	return emit(map[string]any{
		"workflow_id": info.GetWorkflowId(), "run_id": info.GetRunId(),
		"status": statusName(info.GetStatus()), "history_length": info.GetHistoryLength(),
		"pending_activities": len(info.GetPendingActivities()), "dead_lettered": dead,
	})
}

// wfHistory prints every event of the current run, one {event_id, type}
// line each, then {events, first, last}.
func wfHistory(args []string) error {
	fs := newFlags("wf history")
	addr := durableFlag(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("wf history", pos, 1, "one workflow id"); err != nil {
		return err
	}
	d, err := dial(*addr)
	if err != nil {
		return err
	}
	defer d.Close()
	ctx, cancel := context.WithTimeout(context.Background(), rpcTimeout)
	defer cancel()
	stream, err := d.wf.GetHistory(ctx, &durablev1.GetHistoryRequest{WorkflowId: pos[0]})
	if err != nil {
		return rpcErr("GetHistory", err)
	}
	n, first, last := 0, "", ""
	for {
		ev, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return rpcErr("GetHistory", err)
		}
		t := eventType(ev)
		if n == 0 {
			first = t
		}
		n, last = n+1, t
		if err := emit(map[string]any{"event_id": ev.GetEventId(), "type": t}); err != nil {
			return err
		}
	}
	return emit(map[string]any{"events": n, "first": first, "last": last})
}

// eventType is the name of the event's attribute message, for example
// "WorkflowExecutionStarted".
func eventType(ev *durablev1.HistoryEvent) string {
	m := ev.ProtoReflect()
	fd := m.WhichOneof(m.Descriptor().Oneofs().ByName("attrs"))
	if fd == nil {
		return ""
	}
	return string(fd.Message().Name())
}

// wfDLQList prints every dead letter of a task queue, one {task_id,
// workflow_id, activity_type, attempts} line each, then {queue, count}.
func wfDLQList(args []string) error {
	fs := newFlags("wf dlq list")
	addr := durableFlag(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if err := exactly("wf dlq list", pos, 1, "one task queue"); err != nil {
		return err
	}
	d, err := dial(*addr)
	if err != nil {
		return err
	}
	defer d.Close()
	count := 0
	var token []byte
	for {
		page, err := d.deadLetters(pos[0], token)
		if err != nil {
			return err
		}
		for _, dl := range page.GetDeadLetters() {
			count++
			if err := emit(map[string]any{
				"task_id": dl.GetTaskId(), "workflow_id": dl.GetWorkflowId(),
				"activity_type": dl.GetActivityType(), "attempts": dl.GetAttempts(),
			}); err != nil {
				return err
			}
		}
		if token = page.GetNextPageToken(); len(token) == 0 {
			break
		}
	}
	return emit(map[string]any{"queue": pos[0], "count": count})
}

// deadLetters fetches one page of ListDeadLetters.
func (d *durable) deadLetters(queue string, token []byte) (*durablev1.ListDeadLettersResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), rpcTimeout)
	defer cancel()
	page, err := d.wf.ListDeadLetters(ctx, &durablev1.ListDeadLettersRequest{TaskQueue: queue, PageToken: token})
	if err != nil {
		return nil, rpcErr("ListDeadLetters", err)
	}
	return page, nil
}

// wfDLQRedrive puts dead letters back on their queue: every one with --all,
// else the named task ids. Final line: {queue, redriven}.
func wfDLQRedrive(args []string) error {
	fs := newFlags("wf dlq redrive")
	all := fs.Bool("all", false, "redrive every dead letter of the queue")
	addr := durableFlag(fs)
	jsonFlag(fs)
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(pos) == 0 {
		return usagef("wf dlq redrive: want a task queue")
	}
	queue, ids := pos[0], pos[1:]
	if *all == (len(ids) > 0) {
		return usagef("wf dlq redrive: give --all or task ids, not both or neither")
	}
	d, err := dial(*addr)
	if err != nil {
		return err
	}
	defer d.Close()
	ctx, cancel := context.WithTimeout(context.Background(), rpcTimeout)
	defer cancel()
	resp, err := d.wf.RedriveDeadLetter(ctx, &durablev1.RedriveDeadLetterRequest{TaskQueue: queue, TaskIds: ids})
	if err != nil {
		return rpcErr("RedriveDeadLetter", err)
	}
	return emit(map[string]any{"queue": queue, "redriven": resp.GetRedriven()})
}
