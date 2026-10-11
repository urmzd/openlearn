// Course tests for dur.08, the server half (go/durable/server/signals.go):
// SignalWorkflow, CancelWorkflow, and the RequestCancelActivity and
// CancelWorkflowExecution commands, driven over gRPC with the test playing
// the worker by hand.
package dur_08

import (
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

func TestHandExampleSignalWhileBusy(t *testing.T) {
	// WHY: the chapter's worked example (section 3). The approver signals
	//      "approve" (request r1) while the workflow's first task is still in
	//      flight, and again with the same request id. Nothing is pushed to
	//      the worker: the signal is event 4 of the history, the duplicate is
	//      acknowledged and appends nothing, and completing the busy task
	//      schedules another (event 7) whose history carries the signal.
	// KIND: unit
	// CATCHES: s01, s02, s03
	// CHAPTER: dur.08 section 3, worked example
	e := newEnv(t).start()
	e.startWF("release-7", "Approval", `{"model":"tinystories-10m"}`)
	wt1 := e.pollWT()
	if err := e.signal("release-7", "approve", `{"by":"ana"}`, "r1"); err != nil {
		t.Fatalf("SignalWorkflow: %v", err)
	}
	if err := e.signal("release-7", "approve", `{"by":"ana"}`, "r1"); err != nil {
		t.Fatalf("a duplicate signal is acknowledged, not refused: %v", err)
	}
	e.mustCompleteWT(wt1, schedule("1", "prepare", nil))
	h := e.mustHistory("release-7")
	if got, want := kinds(h), "started wt_scheduled wt_started signal wt_completed act_scheduled wt_scheduled"; got != want {
		t.Fatalf("history:\n got %s\nwant %s", got, want)
	}
	sig := h[3].GetSignal()
	if sig.GetSignalName() != "approve" || string(sig.GetInput()) != `{"by":"ana"}` || sig.GetRequestId() != "r1" {
		t.Fatalf("SignalReceived carries the name, input, and request id: %v", sig)
	}
	wt2 := e.pollWT()
	if count(wt2.GetHistory(), "signal") != 1 {
		t.Fatalf("the next workflow task's history must carry the signal: %s", kinds(wt2.GetHistory()))
	}
}

func TestSignalBeforeFirstTaskIsInItsHistory(t *testing.T) {
	// WHY: "a signal sent before the workflow waits is not lost": a signal
	//      that arrives before any worker has polled the run is already in
	//      the first workflow task's history, and adds no second pending task.
	// KIND: unit
	// CHAPTER: dur.08 section 2.1
	e := newEnv(t).start()
	e.startWF("early", "Approval", "")
	if err := e.signal("early", "approve", "yes", ""); err != nil {
		t.Fatal(err)
	}
	wt := e.pollWT()
	if got := kinds(wt.GetHistory()); got != "started wt_scheduled signal wt_started" {
		t.Fatalf("first task history %q: the signal must sit before WorkflowTaskStarted, with one task scheduled", got)
	}
}

func TestSignalDedupSurvivesRestart(t *testing.T) {
	// WHY: a client retries SignalWorkflow after a timeout; the request id
	//      makes the retry harmless, also after the server restarted between
	//      the two calls (the ids are rebuilt from history, not kept in a map
	//      that dies with the process). Signals without a request id are
	//      never deduplicated: two approvals are two events.
	// KIND: fault
	// CATCHES: s02, s03
	// CHAPTER: dur.08 section 2.1
	e := newEnv(t).start()
	e.startWF("dedup", "Approval", "")
	if err := e.signal("dedup", "approve", "a", "req-9"); err != nil {
		t.Fatal(err)
	}
	e.restart()
	if err := e.signal("dedup", "approve", "a", "req-9"); err != nil {
		t.Fatal(err)
	}
	e.signal("dedup", "note", "x", "")
	e.signal("dedup", "note", "x", "")
	h := e.mustHistory("dedup")
	if n := count(h, "signal"); n != 3 {
		t.Fatalf("%d signal events, want 3 (req-9 once, two notes without ids): %s", n, kinds(h))
	}
}

func TestSignalRefusals(t *testing.T) {
	// WHY: a signal must name a live run: an unknown workflow and a closed
	//      run are NOT_FOUND (a signal to a finished release would otherwise
	//      vanish), an empty name is INVALID_ARGUMENT, and so is an input
	//      over the 2 MiB payload limit (DESIGN 2.7).
	// KIND: boundary
	// CATCHES: s04, s05
	// CHAPTER: dur.08 section 2.1
	e := newEnv(t).start()
	if err := e.signal("nobody", "approve", "", ""); code(err) != codes.NotFound {
		t.Fatalf("unknown workflow: %v", err)
	}
	e.startWF("short", "Approval", "")
	if err := e.signal("short", "", "", ""); code(err) != codes.InvalidArgument {
		t.Fatalf("empty signal name: %v", err)
	}
	if err := e.signal("short", "approve", strings.Repeat("x", 2<<20+1), ""); code(err) != codes.InvalidArgument {
		t.Fatalf("input over 2 MiB: %v", err)
	}
	e.mustCompleteWT(e.pollWT(), completeCmd("done"))
	if err := e.signal("short", "approve", "", ""); code(err) != codes.NotFound {
		t.Fatalf("closed run: %v, want NOT_FOUND", err)
	}
}

func TestCancelIsIdempotent(t *testing.T) {
	// WHY: cancellation is requested by people and scripts that retry. The
	//      first request appends WorkflowExecutionCancelRequested and a
	//      workflow task; repeats append nothing; a run that already ended
	//      canceled acknowledges; one that completed refuses
	//      (FAILED_PRECONDITION), and an unknown one is NOT_FOUND.
	// KIND: unit
	// CATCHES: s06, s07
	// CHAPTER: dur.08 section 2.2
	e := newEnv(t).start()
	if err := e.cancel("ghost"); code(err) != codes.NotFound {
		t.Fatalf("unknown workflow: %v", err)
	}
	e.startWF("c1", "Train", "")
	e.mustCompleteWT(e.pollWT())
	for i := 0; i < 3; i++ {
		if err := e.cancel("c1"); err != nil {
			t.Fatalf("cancel %d: %v", i+1, err)
		}
	}
	h := e.mustHistory("c1")
	if count(h, "cancel_requested") != 1 || kindOf(h[len(h)-1]) != "wt_scheduled" {
		t.Fatalf("three cancels: %s", kinds(h))
	}
	e.mustCompleteWT(e.pollWT(), cancelWorkflowCmd("compensated"))
	if err := e.cancel("c1"); err != nil {
		t.Fatalf("cancel after the run ended canceled: %v", err)
	}
	e.startWF("c2", "Train", "")
	e.mustCompleteWT(e.pollWT(), completeCmd("ok"))
	if err := e.cancel("c2"); code(err) != codes.FailedPrecondition {
		t.Fatalf("cancel of a completed run: %v", err)
	}
}

func TestCancelReachesRunningActivities(t *testing.T) {
	// WHY: a running activity learns about a workflow cancel only from its
	//      heartbeat answer. Every activity pending when the cancel arrives is
	//      asked to stop by name (ActivityTaskCancelRequested), so its next
	//      heartbeat says cancel_requested.
	// KIND: fault
	// CATCHES: s08
	// CHAPTER: dur.08 section 2.2
	e := newEnv(t).start()
	e.startWF("train-1", "Train", "")
	e.mustCompleteWT(e.pollWT(), schedule("1", "train", nil))
	at := e.pollAT()
	if hb, err := e.heartbeat(at); err != nil || hb.GetCancelRequested() {
		t.Fatalf("before the cancel: %v, %v", hb, err)
	}
	if err := e.cancel("train-1"); err != nil {
		t.Fatal(err)
	}
	h := e.mustHistory("train-1")
	if count(h, "act_cancel_requested") != 1 {
		t.Fatalf("the pending activity must be asked to stop by name: %s", kinds(h))
	}
	hb, err := e.heartbeat(at)
	if err != nil || !hb.GetCancelRequested() {
		t.Fatalf("after the cancel the heartbeat must say cancel_requested: %v, %v", hb, err)
	}
}

func TestCompensationRunsAfterCancel(t *testing.T) {
	// WHY: after a cancel the workflow schedules its compensations, and they
	//      must run to the end: an activity scheduled after the cancel request
	//      is not asked to stop (its heartbeat says nothing), and the run stays
	//      open until the workflow ends it with CancelWorkflowExecution.
	// KIND: regression
	// CHAPTER: dur.08 section 2.3
	e := newEnv(t).start()
	e.startWF("cb", "CorpusBuild", "")
	e.mustCompleteWT(e.pollWT(), schedule("1", "shard", nil))
	shard := e.pollAT()
	if err := e.cancel("cb"); err != nil {
		t.Fatal(err)
	}
	e.mustCompleteWT(e.pollWT(), schedule("2", "delete-partial-shards", nil))
	comp := e.pollAT()
	if comp.GetActivityType() != "delete-partial-shards" {
		comp = shard // the queue may deliver in either order; find the compensation
	}
	if comp.GetActivityType() != "delete-partial-shards" {
		t.Fatalf("the compensation was not delivered")
	}
	hb, err := e.heartbeat(comp)
	if err != nil || hb.GetCancelRequested() {
		t.Fatalf("a compensation scheduled after the cancel must not be told to stop: %v, %v", hb, err)
	}
	if st := e.describe("cb").GetStatus(); st != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		t.Fatalf("the run is %s before the workflow ends it", st)
	}
}

func TestRequestCancelActivity(t *testing.T) {
	// WHY: a workflow can cancel one activity (a race between two, a timeout
	//      it enforces itself): RequestCancelActivity appends
	//      ActivityTaskCancelRequested once, the attempt in flight hears it on
	//      its next heartbeat, and an id that is not pending (it already
	//      finished) is ignored rather than failing the workflow task.
	// KIND: unit
	// CATCHES: s09, s10
	// CHAPTER: dur.08 section 2.2
	e := newEnv(t).start()
	e.startWF("race", "Race", "")
	e.mustCompleteWT(e.pollWT(), schedule("a", "slow", nil), schedule("b", "fast", nil))
	at := e.pollAT()
	if err := e.signal("race", "go", "", ""); err != nil { // makes the next workflow task
		t.Fatal(err)
	}
	e.mustCompleteWT(e.pollWT(), cancelActivityCmd(at.GetActivityId()), cancelActivityCmd(at.GetActivityId()), cancelActivityCmd("no-such"))
	h := e.mustHistory("race")
	if n := count(h, "act_cancel_requested"); n != 1 {
		t.Fatalf("%d ActivityTaskCancelRequested events, want 1: %s", n, kinds(h))
	}
	if hb, err := e.heartbeat(at); err != nil || !hb.GetCancelRequested() {
		t.Fatalf("the cancelled activity's heartbeat: %v, %v", hb, err)
	}
	if st := e.describe("race").GetStatus(); st != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		t.Fatalf("cancelling an activity does not end the run: %s", st)
	}
}

func TestCancelDeadLetteredActivity(t *testing.T) {
	// WHY: a dead-lettered activity has no attempt in flight to hear a
	//      heartbeat answer; waiting for its acknowledgement would wait for
	//      ever. Cancelling it appends ActivityTaskCanceled at once, which
	//      wakes the workflow.
	// KIND: boundary
	// CATCHES: s11
	// CHAPTER: dur.08 section 5, Pitfalls
	e := newEnv(t).start()
	e.startWF("poison", "Corpus", "")
	cmd := schedule("1", "pii", nil)
	cmd.GetScheduleActivity().Options.Retry = &durablev1.RetryPolicy{MaxAttempts: 1}
	e.mustCompleteWT(e.pollWT(), cmd)
	at := e.pollAT()
	if _, err := e.tasks.FailActivityTask(bg, &durablev1.FailActivityRequest{TaskToken: at.TaskToken, LeaseToken: at.LeaseToken,
		Failure: &durablev1.Failure{Type: "ExitCode1", Message: "boom"}}); err != nil {
		t.Fatal(err)
	}
	if count(e.mustHistory("poison"), "dead_lettered") != 1 {
		t.Fatalf("setup: the activity did not dead-letter: %s", kinds(e.mustHistory("poison")))
	}
	if err := e.cancel("poison"); err != nil {
		t.Fatal(err)
	}
	h := e.mustHistory("poison")
	if count(h, "act_canceled") != 1 {
		t.Fatalf("a dead-lettered activity is canceled at once: %s", kinds(h))
	}
}

func TestCancelWorkflowExecutionNeedsARequest(t *testing.T) {
	// WHY: CancelWorkflowExecution is the workflow's answer to a cancel
	//      request. Sent without one, it is refused (the workflow task fails
	//      and nothing is appended): a workflow ends itself by completing or
	//      failing. With one, the run ends CANCELED with the details, and a
	//      late activity completion is refused.
	// KIND: boundary
	// CATCHES: s12, s13
	// CHAPTER: dur.08 section 2.2
	e := newEnv(t).start()
	e.startWF("self", "Train", "")
	e.mustCompleteWT(e.pollWT(), schedule("1", "train", nil))
	at := e.pollAT()
	e.signal("self", "poke", "", "")
	wt := e.pollWT()
	if err := e.completeWT(wt, cancelWorkflowCmd("")); code(err) != codes.FailedPrecondition {
		t.Fatalf("CancelWorkflowExecution without a request: %v", err)
	}
	if count(e.mustHistory("self"), "canceled") != 0 {
		t.Fatal("a refused command appended an event")
	}
	if err := e.cancel("self"); err != nil {
		t.Fatal(err)
	}
	e.clk.Advance(20 * time.Second) // the refused task's lease runs out
	wt = e.pollWT()
	e.mustCompleteWT(wt, cancelWorkflowCmd("rolled back"))
	inf := e.describe("self")
	if inf.GetStatus() != durablev1.WorkflowStatus_WORKFLOW_STATUS_CANCELED {
		t.Fatalf("status %s, want CANCELED", inf.GetStatus())
	}
	h := e.mustHistory("self")
	if last := h[len(h)-1]; kindOf(last) != "canceled" || string(last.GetCanceled().GetDetails()) != "rolled back" {
		t.Fatalf("last event %s %v", kindOf(last), last)
	}
	if _, err := e.tasks.CompleteActivityTask(bg, &durablev1.CompleteActivityRequest{TaskToken: at.TaskToken, LeaseToken: at.LeaseToken, Result: []byte("late")}); code(err) != codes.FailedPrecondition {
		t.Fatalf("an activity of a canceled run completing late: %v", err)
	}
}
