// Course tests for dur.04, part 1: the server half of the task protocol
// (go/durable/server/tasks.go), driven over gRPC with no worker in between.
package dur_04

import (
	"context"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/proto"

	"tinyllm/durable/queue"
	"tinyllm/durable/server"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

// guard bounds an rpc that long-polls the fake clock: a poll with nothing to
// deliver would otherwise wait for a clock nobody advances.
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

func (e *env) completeAT(at *durablev1.ActivityTask, result string) error {
	_, err := e.tasks.CompleteActivityTask(bg, &durablev1.CompleteActivityRequest{TaskToken: at.TaskToken, LeaseToken: at.LeaseToken, Result: []byte(result), Identity: "wk"})
	return err
}

func schedule(id, typ string, input []byte) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_ScheduleActivity{ScheduleActivity: &durablev1.ScheduleActivity{
		ActivityId: id, ActivityType: typ, Input: input,
		Options: &durablev1.ActivityOptions{StartToCloseMs: 10000},
	}}}
}

func completeCmd(result string) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_Complete{Complete: &durablev1.CompleteWorkflow{Result: []byte(result)}}}
}

func TestTaskProtocolHandExample(t *testing.T) {
	// WHY: the chapter's worked example, event by event: Echo("hi") with one
	//      activity Upper. Poll appends WorkflowTaskStarted (3); completing
	//      with ScheduleActivity appends 4 and 5; the activity task carries
	//      attempt 1, the idempotency key "demo-1/1", and lease token 2;
	//      completing it appends 6, 7, and a new WorkflowTaskScheduled (8);
	//      the second workflow task (9) completes the run (10, 11).
	// KIND: unit
	// CATCHES: s01, s02, s03, s05
	// CHAPTER: dur.04 section 3, worked example
	e := newEnv(t).start()
	e.startWF("demo-1", "Echo", "hi")
	wt := e.pollWT()
	if kinds(wt.History) != "started wt_scheduled wt_started" || wt.WorkflowType != "Echo" || wt.RunId != "run-1" ||
		wt.History[2].GetWtStarted().ScheduledEventId != 2 || len(wt.NextPageToken) != 0 || wt.ContinueAsNewSuggested {
		t.Fatalf("first workflow task: %q %v", kinds(wt.History), wt)
	}
	if err := e.completeWT(wt, schedule("1", "Upper", []byte("hi"))); err != nil {
		t.Fatalf("CompleteWorkflowTask: %v", err)
	}
	h := e.mustHistory("demo-1")
	if kinds(h[3:]) != "wt_completed act_scheduled" || h[4].GetActScheduled().WorkflowTaskCompletedEventId != 4 ||
		h[3].GetWtCompleted().StartedEventId != 3 {
		t.Fatalf("after the first completion: %q", kinds(h))
	}
	at := e.pollAT()
	if at.ActivityId != "1" || at.ActivityType != "Upper" || string(at.Input) != "hi" || at.Attempt != 1 ||
		at.IdempotencyKey != "demo-1/1" || at.LeaseToken != 2 || at.WorkflowId != "demo-1" || at.RunId != "run-1" ||
		at.DeadlineUnixMs != t0.Add(10*time.Second).UnixMilli() {
		t.Fatalf("activity task = %v", at)
	}
	if err := e.completeAT(at, "HI"); err != nil {
		t.Fatalf("CompleteActivityTask: %v", err)
	}
	h = e.mustHistory("demo-1")
	if kinds(h[5:]) != "act_started act_completed wt_scheduled" || h[6].GetActCompleted().ScheduledEventId != 5 ||
		string(h[6].GetActCompleted().Result) != "HI" || h[5].GetActStarted().Attempt != 1 || h[5].GetActStarted().LeaseToken != 2 {
		t.Fatalf("after the activity: %q", kinds(h))
	}
	wt2 := e.pollWT()
	if len(wt2.History) != 9 || kindOf(wt2.History[8]) != "wt_started" || wt2.History[8].GetWtStarted().ScheduledEventId != 8 {
		t.Fatalf("second workflow task history: %q", kinds(wt2.History))
	}
	if err := e.completeWT(wt2, completeCmd("HI")); err != nil {
		t.Fatalf("complete the run: %v", err)
	}
	inf := e.describe("demo-1")
	if inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED || string(inf.Result) != "HI" || inf.HistoryLength != 11 {
		t.Fatalf("describe = %v", inf)
	}
	if v, l, d := e.q.Depth("wf:default"); v+l+d != 0 {
		t.Fatalf("workflow tasks left in the queue: %d %d %d", v, l, d)
	}
}

func TestCompleteWorkflowTaskFenced(t *testing.T) {
	// WHY: only the run's current workflow task may complete. A second
	//      completion with the same token, or one whose lease expired and was
	//      redelivered (a new WorkflowTaskStarted), is FAILED_PRECONDITION and
	//      appends nothing; the new holder's token works.
	// KIND: unit, fault
	// CATCHES: s04, s05
	// CHAPTER: dur.04 section 2, fencing
	e := newEnv(t).start()
	e.startWF("w", "Echo", "")
	old := e.pollWT()
	e.clk.Advance(server.DefaultWorkflowTaskLease) // the worker died: the lease expires
	cur := e.pollWT()
	if kindOf(cur.History[len(cur.History)-1]) != "wt_started" || len(cur.History) != 4 {
		t.Fatalf("redelivery should append a second WorkflowTaskStarted: %q", kinds(cur.History))
	}
	if err := e.completeWT(old, completeCmd("late")); code(err) != codes.FailedPrecondition {
		t.Fatalf("completion with the expired task's token: %v; want FAILED_PRECONDITION", err)
	}
	if err := e.completeWT(cur, schedule("1", "A", nil)); err != nil {
		t.Fatalf("completion with the current token: %v", err)
	}
	if err := e.completeWT(cur, completeCmd("twice")); code(err) != codes.FailedPrecondition {
		t.Fatalf("second completion of the same task: %v; want FAILED_PRECONDITION", err)
	}
	if h := e.mustHistory("w"); kinds(h) != "started wt_scheduled wt_started wt_started wt_completed act_scheduled" {
		t.Fatalf("history %q", kinds(h))
	}
}

func TestStaleActivityLeaseDropsResult(t *testing.T) {
	// WHY: a worker that was presumed dead (lease expired, task redelivered
	//      as attempt 2) must not record its late result: FAILED_PRECONDITION,
	//      history unchanged. The current attempt's result is the one recorded.
	// KIND: fault
	// CATCHES: s06, s07
	// CHAPTER: dur.04 section 2, fencing
	e := newEnv(t).start()
	e.startWF("w", "Echo", "")
	e.completeWT(e.pollWT(), schedule("1", "Slow", nil))
	slow := e.pollAT()
	e.clk.Advance(10 * time.Second)
	fresh := e.pollAT()
	if fresh.Attempt != 2 || fresh.LeaseToken <= slow.LeaseToken || fresh.IdempotencyKey != slow.IdempotencyKey {
		t.Fatalf("redelivery: attempt %d token %d key %q (first: token %d key %q)", fresh.Attempt, fresh.LeaseToken, fresh.IdempotencyKey, slow.LeaseToken, slow.IdempotencyKey)
	}
	if err := e.completeAT(slow, "late"); code(err) != codes.FailedPrecondition {
		t.Fatalf("late completion: %v; want FAILED_PRECONDITION", err)
	}
	if n := len(e.mustHistory("w")); n != 5 {
		t.Fatalf("the refused completion appended events (%d)", n)
	}
	if err := e.completeAT(fresh, "fresh"); err != nil {
		t.Fatal(err)
	}
	h := e.mustHistory("w")
	if got := string(h[6].GetActCompleted().GetResult()); got != "fresh" || h[5].GetActStarted().Attempt != 2 {
		t.Fatalf("recorded result %q attempt %d; want fresh, 2", got, h[5].GetActStarted().Attempt)
	}
}

func TestStaleQueueTaskSkipped(t *testing.T) {
	// WHY: history is the truth and queue tasks are hints. A delivered task
	//      whose WorkflowTaskScheduled already completed (left over by a crash
	//      or a duplicate) is acknowledged and skipped, never run twice.
	// KIND: fault
	// CATCHES: s09
	// CHAPTER: dur.04 section 5, Pitfalls
	e := newEnv(t).start()
	e.startWF("a", "Echo", "")
	e.completeWT(e.pollWT(), schedule("1", "A", nil))           // a keeps running, no workflow task pending
	e.q.Enqueue(bg, "wf:default", queue.Task{ID: "run-1/wt/2"}) // a stale duplicate of a's first task
	e.startWF("b", "Echo", "")
	wt := e.pollWT()
	if wt.WorkflowId != "b" {
		t.Fatalf("got a workflow task for %q; the stale one for a should have been skipped", wt.WorkflowId)
	}
	if v, l, d := e.q.Depth("wf:default"); v != 0 || l != 1 || d != 0 {
		t.Fatalf("queue after skipping: visible %d leased %d dead %d; want 0 1 0", v, l, d)
	}
	if h := e.mustHistory("a"); len(h) != 5 {
		t.Fatalf("the stale task touched a's history: %q", kinds(h))
	}
}

func TestTriggerWhileWorkflowTaskInFlight(t *testing.T) {
	// WHY: one workflow task at a time per run. An activity that completes
	//      while a workflow task is running must not schedule a second one;
	//      it is remembered, and the completion of the running task schedules
	//      the next one in the same append, so the news is never lost.
	// KIND: unit
	// CATCHES: s10
	// CHAPTER: dur.04 section 2, one workflow task at a time
	e := newEnv(t).start()
	e.startWF("w", "Echo", "")
	e.completeWT(e.pollWT(), schedule("a", "A", nil), schedule("b", "B", nil)) // events 4, 5, 6
	first, second := e.pollAT(), e.pollAT()
	e.completeAT(first, "ra")  // 7, 8, wt_scheduled 9
	wt := e.pollWT()           // wt_started 10: in flight
	e.completeAT(second, "rb") // 11, 12: no new workflow task yet
	h := e.mustHistory("w")
	if got := kinds(h[6:]); got != "act_started act_completed wt_scheduled wt_started act_started act_completed" {
		t.Fatalf("while in flight: %q", got)
	}
	if err := e.completeWT(wt); err != nil {
		t.Fatal(err)
	}
	h = e.mustHistory("w")
	if got := kinds(h[12:]); got != "wt_completed wt_scheduled" {
		t.Fatalf("completing the in-flight task: %q; want wt_completed then a new wt_scheduled", got)
	}
	if next := e.pollWT(); len(next.History) != 15 {
		t.Fatalf("the next workflow task sees %d events; want 15", len(next.History))
	}
}

func TestFirstPageWithinOneMiB(t *testing.T) {
	// WHY: a WorkflowTask carries at most 1 MiB of history (messages are
	//      capped at 4 MiB); the rest is fetched with GetHistory from
	//      next_page_token. The two parts must join into the full history.
	// KIND: boundary
	// CATCHES: s12
	// CHAPTER: dur.04 section 4
	e := newEnv(t).start()
	e.startWF("big", "Echo", "")
	in := []byte(strings.Repeat("x", 900<<10))
	e.completeWT(e.pollWT(), schedule("1", "A", in), schedule("2", "A", in), schedule("3", "A", in))
	e.completeAT(e.pollAT(), "ok")
	wt := e.pollWT()
	if len(wt.NextPageToken) == 0 {
		t.Fatalf("a %d-event history over 2 MiB came in one page", len(wt.History))
	}
	size := 0
	for _, ev := range wt.History {
		size += proto.Size(ev)
	}
	if size > server.FirstPageBytes || len(wt.History) != 5 {
		t.Fatalf("first page: %d events, %d bytes; want the 5 that fit in 1 MiB", len(wt.History), size)
	}
	rest, err := e.history("big", "", wt.NextPageToken)
	if err != nil {
		t.Fatal(err)
	}
	full := append(append([]*durablev1.HistoryEvent(nil), wt.History...), rest...)
	if kinds(full) != kinds(e.mustHistory("big")) || full[len(full)-1].GetWtStarted() == nil {
		t.Fatalf("first page + rest = %q", kinds(full))
	}
}

func TestActivityPayloadLimits(t *testing.T) {
	// WHY: payloads above 2 MiB travel by path, never through the server:
	//      an oversized activity input, activity result, or workflow result is
	//      INVALID_ARGUMENT and appends nothing; start_to_close is required.
	// KIND: boundary
	// CATCHES: s13, s14
	// CHAPTER: dur.04 section 4
	e := newEnv(t).start()
	e.startWF("w", "Echo", "")
	wt := e.pollWT()
	big := make([]byte, server.MaxPayloadBytes+1)
	noTimeout := schedule("1", "A", nil)
	noTimeout.GetScheduleActivity().Options = &durablev1.ActivityOptions{}
	for name, cmd := range map[string]*durablev1.Command{"input": schedule("1", "A", big), "no timeout": noTimeout, "result": completeCmd(string(big))} {
		if err := e.completeWT(wt, cmd); code(err) != codes.InvalidArgument {
			t.Fatalf("%s: %v; want INVALID_ARGUMENT", name, err)
		}
	}
	if err := e.completeWT(wt, schedule("1", "A", big[:server.MaxPayloadBytes])); err != nil {
		t.Fatalf("an input of exactly 2 MiB: %v", err)
	}
	at := e.pollAT()
	if err := e.completeAT(at, string(big)); code(err) != codes.InvalidArgument {
		t.Fatalf("oversized activity result: %v", err)
	}
	if err := e.completeAT(at, "small"); err != nil {
		t.Fatalf("a valid result after the refused one: %v", err)
	}
}
