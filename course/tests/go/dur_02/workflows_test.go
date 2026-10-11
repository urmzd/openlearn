// Course tests for dur.02, the workflow service and its event-sourced core
// (go/durable/server: server.go and workflows.go).
package dur_02

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	dlog "tinyllm/durable/log"
	"tinyllm/durable/server"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

func TestStartHandExample(t *testing.T) {
	// WHY: the chapter's worked example: `wf start Echo --id demo-1` twice.
	//      The first start commits WorkflowExecutionStarted (event 1) and
	//      WorkflowTaskScheduled (event 2) in one append; the second returns
	//      the same run with started = false and appends nothing.
	// KIND: unit
	// CATCHES: s01, s02, s03
	// CHAPTER: dur.02 section 3, worked example
	e := newEnv(t).start()
	first := e.startWF("demo-1", "Echo", `"hi"`)
	if first.RunId != "run-1" || !first.Started {
		t.Fatalf("first start = %+v; want run-1, started", first)
	}
	again := e.startWF("demo-1", "Echo", `"hi"`)
	if again.RunId != "run-1" || again.Started {
		t.Fatalf("second start = %+v; want run-1, not started", again)
	}
	h := e.mustHistory("demo-1")
	if kinds(h) != "started wt_scheduled" || h[0].EventId != 1 || h[1].EventId != 2 {
		t.Fatalf("history %q ids %d,%d; want started (1), wt_scheduled (2)", kinds(h), h[0].GetEventId(), h[len(h)-1].GetEventId())
	}
	st := h[0].GetStarted()
	if st.WorkflowType != "Echo" || st.TaskQueue != "default" || string(st.Input) != `"hi"` || st.Identity != "test" ||
		h[0].TsUnixMs != t0.UnixMilli() || h[1].GetWtScheduled().TaskQueue != "default" || h[1].GetWtScheduled().Attempt != 1 {
		t.Fatalf("events: %v / %v", h[0], h[1])
	}
	inf := e.describe("demo-1")
	if inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING || inf.RunId != "run-1" || inf.WorkflowType != "Echo" ||
		inf.HistoryLength != 2 || inf.StartUnixMs != t0.UnixMilli() || inf.CloseUnixMs != 0 {
		t.Fatalf("describe = %v", inf)
	}
	if e.log.Version("run/run-1") != 2 || e.log.LastSeq() != 2 {
		// one record for the start batch, one for the workflow task's enqueue
		t.Fatalf("log: run stream at %d, %d records; want 2 events in one record plus one queue record", e.log.Version("run/run-1"), e.log.LastSeq())
	}
}

func TestStartIdempotentUnder64Concurrent(t *testing.T) {
	// WHY: 64 clients start the same workflow at once (a retried CLI, two
	//      operators): exactly one WorkflowExecutionStarted, one run id, and
	//      every caller learns that id; 63 of them see started = false.
	// KIND: fault, property
	// CATCHES: s01, s04
	// CHAPTER: dur.02 section 2, idempotent start
	e := newEnv(t).start()
	var wg sync.WaitGroup
	var mu sync.Mutex
	runs, started := map[string]int{}, 0
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			resp, err := e.wf.StartWorkflow(bg, startReq("corpus/tinystories/v1", "CorpusBuild", `{"shards":4}`))
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				t.Errorf("StartWorkflow: %v", err)
				return
			}
			runs[resp.RunId]++
			if resp.Started {
				started++
			}
		}()
	}
	wg.Wait()
	if len(runs) != 1 || started != 1 {
		t.Fatalf("%d run ids %v, %d started; want 1 and 1", len(runs), runs, started)
	}
	h := e.mustHistory("corpus/tinystories/v1")
	if strings.Count(kinds(h), "started") != 1 || len(h) != 2 {
		t.Fatalf("history %q; want one started and one wt_scheduled", kinds(h))
	}
	if v, _, _ := e.q.Depth("wf:default"); v != 1 {
		t.Fatalf("%d workflow tasks queued; want 1", v)
	}
}

func TestStartConflict(t *testing.T) {
	// WHY: an id names one piece of work. The same id with a different type
	//      or input is a mistake the caller must see (FAILED_PRECONDITION),
	//      never a silent second run and never the old run disguised as theirs.
	// KIND: unit, boundary
	// CATCHES: s01, s05
	// CHAPTER: dur.02 section 2, idempotent start
	e := newEnv(t).start()
	e.startWF("w", "Echo", "a")
	for _, req := range []*durablev1.StartWorkflowRequest{startReq("w", "Echo", "b"), startReq("w", "Other", "a"), startReq("w", "Echo", "")} {
		if _, err := e.wf.StartWorkflow(bg, req); code(err) != codes.FailedPrecondition {
			t.Fatalf("start %s/%q over Echo/\"a\": %v; want FAILED_PRECONDITION", req.WorkflowType, req.Input, err)
		}
	}
	if resp := e.startWF("w", "Echo", "a"); resp.Started || resp.RunId != "run-1" {
		t.Fatalf("the matching start after conflicts = %+v", resp)
	}
	if n := len(e.mustHistory("w")); n != 2 {
		t.Fatalf("conflicting starts appended events: %d", n)
	}
}

func TestStartValidation(t *testing.T) {
	// WHY: the contract's input checks: workflow_id, workflow_type, and
	//      task_queue are required, and input is at most 2 MiB (larger data
	//      travels by path); exactly 2 MiB is accepted.
	// KIND: boundary
	// CATCHES: s06
	// CHAPTER: dur.02 section 4
	e := newEnv(t).start()
	for name, req := range map[string]*durablev1.StartWorkflowRequest{
		"no id":    {WorkflowType: "T", TaskQueue: "q"},
		"no type":  {WorkflowId: "w", TaskQueue: "q"},
		"no queue": {WorkflowId: "w", WorkflowType: "T"},
		"too big":  {WorkflowId: "w", WorkflowType: "T", TaskQueue: "q", Input: make([]byte, server.MaxPayloadBytes+1)},
	} {
		if _, err := e.wf.StartWorkflow(bg, req); code(err) != codes.InvalidArgument {
			t.Fatalf("%s: %v; want INVALID_ARGUMENT", name, err)
		}
	}
	big := &durablev1.StartWorkflowRequest{WorkflowId: "w", WorkflowType: "T", TaskQueue: "q", Input: make([]byte, server.MaxPayloadBytes)}
	if _, err := e.wf.StartWorkflow(bg, big); err != nil {
		t.Fatalf("exactly 2 MiB: %v", err)
	}
}

func TestStartEnqueuesOneWorkflowTask(t *testing.T) {
	// WHY: the side effect of WorkflowTaskScheduled is one task on the run's
	//      queue, "wf:<task_queue>", named "<run_id>/wt/<event id>", so a
	//      worker polling that queue gets the new run's first workflow task.
	// KIND: unit
	// CATCHES: s07
	// CHAPTER: dur.02 section 2, derived side effects
	e := newEnv(t).start()
	e.wf.StartWorkflow(bg, &durablev1.StartWorkflowRequest{WorkflowId: "w", WorkflowType: "T", TaskQueue: "gpu"})
	l, err := e.q.Poll(bg, "wf:gpu", "test", 0)
	if err != nil || l.Task.ID != "run-1/wt/2" {
		t.Fatalf("poll wf:gpu = %q, %v; want run-1/wt/2", l.Task.ID, err)
	}
	if v, _, _ := e.q.Depth("wf:default"); v != 0 {
		t.Fatal("the task went to the wrong queue")
	}
}

func TestHistoryIsWalRecords(t *testing.T) {
	// WHY: formats/wal.md: each history event is stored as a tl.durable.v1
	//      WalRecord carrying the workflow id, the run id, and the event, on
	//      stream "run/<run_id>" whose version equals the event id. Raft
	//      (dur.10) and every recovery depend on exactly these bytes.
	// KIND: conformance
	// CATCHES: s08
	// CHAPTER: dur.02 section 2, storage
	e := newEnv(t).start()
	e.startWF("w", "Echo", "x")
	evs, err := e.log.Read(bg, "run/run-1", 1, 0)
	if err != nil || len(evs) != 2 {
		t.Fatalf("log stream run/run-1: %d events, %v", len(evs), err)
	}
	h := e.mustHistory("w")
	for i, ev := range evs {
		var rec durablev1.WalRecord
		if err := proto.Unmarshal(ev.Data, &rec); err != nil {
			t.Fatalf("event %d data is not a WalRecord: %v", i+1, err)
		}
		if rec.WorkflowId != "w" || rec.RunId != "run-1" || !proto.Equal(rec.GetEvent(), h[i]) || ev.Version != h[i].EventId {
			t.Fatalf("log event %d = %v (version %d); want the WalRecord of %v", i+1, &rec, ev.Version, h[i])
		}
	}
}

// fixture is a recorded run: course/fixtures/dur/histories/<name>.json.
type fixture struct {
	WorkflowID string            `json:"workflow_id"`
	RunID      string            `json:"run_id"`
	Type       string            `json:"workflow_type"`
	Events     []json.RawMessage `json:"events"`
	events     []*durablev1.HistoryEvent
	name       string
}

func fixtures(t *testing.T) []*fixture {
	t.Helper()
	dir := filepath.Join(os.Getenv("TINYLLM_FIXTURES"), "dur", "histories")
	paths, _ := filepath.Glob(filepath.Join(dir, "*.json"))
	if len(paths) == 0 {
		t.Fatalf("no recorded histories in %s (is TINYLLM_FIXTURES set?)", dir)
	}
	var out []*fixture
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		f := &fixture{name: filepath.Base(p)}
		if err := json.Unmarshal(raw, f); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		for _, m := range f.Events {
			ev := &durablev1.HistoryEvent{}
			if err := protojson.Unmarshal(m, ev); err != nil {
				t.Fatalf("%s: %v", p, err)
			}
			f.events = append(f.events, ev)
		}
		out = append(out, f)
	}
	return out
}

// seed writes a recorded run into the log the way the server stores it.
func seed(t *testing.T, lg *dlog.Log, f *fixture) {
	t.Helper()
	var evs []dlog.Event
	for _, ev := range f.events {
		data, _ := proto.Marshal(&durablev1.WalRecord{WorkflowId: f.WorkflowID, RunId: f.RunID, Record: &durablev1.WalRecord_Event{Event: ev}})
		evs = append(evs, dlog.Event{Type: kindOf(ev), Data: data})
	}
	if _, err := lg.Append(bg, "run/"+f.RunID, 0, evs...); err != nil {
		t.Fatal(err)
	}
}

func TestRecordedHistoriesRecover(t *testing.T) {
	// WHY: recovery is a fold over history. Each recorded run of the course
	//      test workflows (fixtures/dur/histories) is written into an empty
	//      log; the server must then report it exactly: the same events from
	//      GetHistory (also from a page token in the middle), and a
	//      DescribeWorkflow and ListWorkflows derived from them alone.
	// KIND: conformance, golden
	// CATCHES: s02, s10, s11
	// CHAPTER: dur.02 section 2, recovery
	fs := fixtures(t)
	e := newEnv(t)
	lg, err := dlog.Open(e.dir, dlog.Options{Sync: fastSync})
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range fs {
		seed(t, lg, f)
	}
	lg.Close()
	e.start()
	listed, err := e.wf.ListWorkflows(bg, &durablev1.ListWorkflowsRequest{})
	if err != nil || len(listed.Workflows) != len(fs) {
		t.Fatalf("ListWorkflows: %d runs, %v; want %d", len(listed.GetWorkflows()), err, len(fs))
	}
	for _, f := range fs {
		h, err := e.history(f.WorkflowID, f.RunID, nil)
		if err != nil || len(h) != len(f.events) {
			t.Fatalf("%s: GetHistory %d events, %v; want %d", f.name, len(h), err, len(f.events))
		}
		var size int64
		for i := range h {
			if !proto.Equal(h[i], f.events[i]) {
				t.Fatalf("%s: event %d = %v; want %v", f.name, i+1, h[i], f.events[i])
			}
			size += int64(proto.Size(f.events[i]))
		}
		mid := int64(len(h)/2 + 1)
		page, err := e.history(f.WorkflowID, f.RunID, server.PageToken(mid))
		if err != nil || len(page) != len(h)-int(mid)+1 || !proto.Equal(page[0], h[mid-1]) {
			t.Fatalf("%s: GetHistory from event %d: %d events, %v", f.name, mid, len(page), err)
		}
		last := f.events[len(f.events)-1]
		inf, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: f.WorkflowID, RunId: f.RunID})
		if err != nil {
			t.Fatalf("%s: describe: %v", f.name, err)
		}
		want := &durablev1.WorkflowInfo{
			WorkflowId: f.WorkflowID, RunId: f.RunID, WorkflowType: f.Type, TaskQueue: f.events[0].GetStarted().GetTaskQueue(),
			StartUnixMs: f.events[0].TsUnixMs, HistoryLength: int64(len(h)), HistoryBytes: size,
			ContinuedFromRunId: f.events[0].GetStarted().GetContinuedFromRunId(),
		}
		switch {
		case last.GetCompleted() != nil:
			want.Status, want.CloseUnixMs, want.Result = durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED, last.TsUnixMs, last.GetCompleted().Result
		case last.GetContinued() != nil:
			want.Status, want.CloseUnixMs = durablev1.WorkflowStatus_WORKFLOW_STATUS_CONTINUED_AS_NEW, last.TsUnixMs
			want.ContinuedAsRunId = last.GetContinued().NewRunId
		case last.GetFailed() != nil:
			want.Status, want.CloseUnixMs, want.Failure = durablev1.WorkflowStatus_WORKFLOW_STATUS_FAILED, last.TsUnixMs, last.GetFailed().Failure
		default:
			t.Fatalf("%s: the fixture should end its run", f.name)
		}
		if !proto.Equal(inf, want) {
			t.Fatalf("%s: describe =\n%v\nwant\n%v", f.name, inf, want)
		}
	}
	if v, l, d := e.q.Depth("wf:default"); v+l+d != 0 {
		t.Fatalf("closed runs re-enqueued workflow tasks: %d %d %d", v, l, d)
	}
}

func TestRecoveryRebuildsEverything(t *testing.T) {
	// WHY: a crash loses nothing acknowledged and doubles nothing: after a
	//      restart, describe, list, and history are identical, a repeated
	//      start is still idempotent, each run still has exactly one queued
	//      workflow task, and new runs continue normally.
	// KIND: fault
	// CATCHES: s09, s22
	// CHAPTER: dur.02 section 2, recovery
	e := newEnv(t).start()
	for i := 0; i < 3; i++ {
		e.startWF(fmt.Sprintf("w%d", i), "Echo", fmt.Sprint(i))
		e.clk.Advance(time.Second)
	}
	before, _ := e.wf.ListWorkflows(bg, &durablev1.ListWorkflowsRequest{})
	hist := e.mustHistory("w1")
	e.restart()
	after, err := e.wf.ListWorkflows(bg, &durablev1.ListWorkflowsRequest{})
	if err != nil || !proto.Equal(before, after) {
		t.Fatalf("list differs after restart:\n%v\nvs\n%v (%v)", before, after, err)
	}
	if h2 := e.mustHistory("w1"); len(h2) != len(hist) || !proto.Equal(h2[0], hist[0]) {
		t.Fatalf("history of w1 differs after restart")
	}
	if resp := e.startWF("w2", "Echo", "2"); resp.Started || resp.RunId != "run-3" {
		t.Fatalf("start after restart = %+v; want run-3, not started", resp)
	}
	if v, _, _ := e.q.Depth("wf:default"); v != 3 {
		t.Fatalf("%d workflow tasks queued after restart; want 3 (one per run, none doubled)", v)
	}
	if resp := e.startWF("w9", "Echo", "9"); !resp.Started || resp.RunId != "run-4" {
		t.Fatalf("new start after restart = %+v", resp)
	}
}

func TestRecoveryEnqueuesLostTask(t *testing.T) {
	// WHY: a crash can land after the history append and before the queue
	//      enqueue. History wins: recovery sees the pending
	//      WorkflowTaskScheduled and enqueues its task.
	// KIND: fault
	// CATCHES: s12
	// CHAPTER: dur.02 section 5, Pitfalls
	e := newEnv(t)
	lg, _ := dlog.Open(e.dir, dlog.Options{Sync: fastSync})
	seed(t, lg, &fixture{WorkflowID: "lost", RunID: "r-lost", events: []*durablev1.HistoryEvent{
		{EventId: 1, TsUnixMs: 1, Attrs: &durablev1.HistoryEvent_Started{Started: &durablev1.WorkflowExecutionStarted{WorkflowType: "T", TaskQueue: "default"}}},
		{EventId: 2, TsUnixMs: 1, Attrs: &durablev1.HistoryEvent_WtScheduled{WtScheduled: &durablev1.WorkflowTaskScheduled{TaskQueue: "default", Attempt: 1}}},
	}})
	lg.Close()
	e.start()
	l, err := e.q.Poll(bg, "wf:default", "t", 0)
	if err != nil || l.Task.ID != "r-lost/wt/2" {
		t.Fatalf("after recovery the queue gave %q, %v; want r-lost/wt/2", l.Task.ID, err)
	}
	if e.describe("lost").Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		t.Fatal("recovered run is not running")
	}
}

func TestListWorkflowsNewestFirstPaged(t *testing.T) {
	// WHY: `wf list` shows the newest runs first, filters by status and type,
	//      and pages with an opaque token so a long list never arrives in one
	//      message.
	// KIND: unit
	// CATCHES: s13, s14
	// CHAPTER: dur.02 section 4
	e := newEnv(t).start()
	for i, typ := range []string{"Echo", "Train", "Echo", "Echo", "Train"} {
		e.startWF(fmt.Sprintf("w%d", i), typ, "")
		e.clk.Advance(time.Minute)
	}
	var ids []string
	var tok []byte
	pages := 0
	for {
		resp, err := e.wf.ListWorkflows(bg, &durablev1.ListWorkflowsRequest{PageSize: 2, PageToken: tok})
		if err != nil {
			t.Fatal(err)
		}
		pages++
		for _, w := range resp.Workflows {
			ids = append(ids, w.WorkflowId)
		}
		if len(resp.NextPageToken) == 0 {
			break
		}
		tok = resp.NextPageToken
	}
	if strings.Join(ids, " ") != "w4 w3 w2 w1 w0" || pages != 3 {
		t.Fatalf("listed %v in %d pages; want w4..w0 in 3", ids, pages)
	}
	trains, _ := e.wf.ListWorkflows(bg, &durablev1.ListWorkflowsRequest{WorkflowType: "Train"})
	if len(trains.Workflows) != 2 || trains.Workflows[0].WorkflowId != "w4" {
		t.Fatalf("type filter: %v", trains)
	}
	done, _ := e.wf.ListWorkflows(bg, &durablev1.ListWorkflowsRequest{Status: durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED})
	if len(done.Workflows) != 0 {
		t.Fatalf("status filter: %d completed runs; want 0", len(done.Workflows))
	}
}

func TestNotFoundAndBadTokens(t *testing.T) {
	// WHY: an unknown workflow or run is NOT_FOUND (the CLI prints it), and a
	//      malformed page token is INVALID_ARGUMENT rather than a panic.
	// KIND: boundary
	// CATCHES: s23
	// CHAPTER: dur.02 section 4
	e := newEnv(t).start()
	e.startWF("w", "Echo", "")
	if _, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: "nope"}); code(err) != codes.NotFound {
		t.Fatalf("describe unknown: %v", err)
	}
	if _, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: "w", RunId: "run-9"}); code(err) != codes.NotFound {
		t.Fatalf("describe unknown run: %v", err)
	}
	if _, err := e.history("nope", "", nil); code(err) != codes.NotFound {
		t.Fatalf("history unknown: %v", err)
	}
	if _, err := e.history("w", "", []byte{1, 2}); code(err) != codes.InvalidArgument {
		t.Fatalf("bad page token: %v", err)
	}
	if h, err := e.history("w", "", server.PageToken(3)); err != nil || len(h) != 0 {
		t.Fatalf("page past the end: %d events, %v", len(h), err)
	}
	if _, err := e.wf.ListWorkflows(bg, &durablev1.ListWorkflowsRequest{PageToken: []byte("x")}); code(err) != codes.InvalidArgument {
		t.Fatalf("bad list token: %v", err)
	}
}

func TestPageTokenRoundTrip(t *testing.T) {
	// WHY: a page token names the next event id; workers and the CLI pass it
	//      back unchanged, so it must survive the round trip, and an empty
	//      token means "from event 1".
	// KIND: unit
	// CATCHES: s15
	// CHAPTER: dur.02 section 4
	for _, n := range []int64{1, 2, 300, 1 << 40} {
		if got, err := server.ParsePageToken(server.PageToken(n)); err != nil || got != n {
			t.Fatalf("ParsePageToken(PageToken(%d)) = %d, %v", n, got, err)
		}
	}
	if got, err := server.ParsePageToken(nil); err != nil || got != 1 {
		t.Fatalf("empty token = %d, %v; want 1", got, err)
	}
	if _, err := server.ParsePageToken(server.PageToken(0)); err == nil {
		t.Fatal("token for event 0 accepted")
	}
}

func TestQuotaResourceExhausted(t *testing.T) {
	// WHY: when the WAL quota is reached a start must fail cleanly with
	//      RESOURCE_EXHAUSTED and leave no half-registered run; after the
	//      operator raises the quota and restarts, starts work again
	//      (drill ops.11).
	// KIND: fault
	// CATCHES: s16, s17
	// CHAPTER: dur.02 section 5, Pitfalls
	e := newEnv(t)
	e.maxBytes = 600
	e.start()
	var err error
	n := 0
	for ; n < 20 && err == nil; n++ {
		_, err = e.wf.StartWorkflow(bg, startReq(fmt.Sprintf("w%d", n), "Echo", "payload-payload"))
	}
	if code(err) != codes.ResourceExhausted {
		t.Fatalf("start at the quota: %v; want RESOURCE_EXHAUSTED", err)
	}
	failed := fmt.Sprintf("w%d", n-1)
	if _, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: failed}); code(err) != codes.NotFound {
		t.Fatalf("the refused start left a run behind: %v", err)
	}
	ok := e.describe("w0")
	e.maxBytes = 0
	e.restart()
	if got := e.describe("w0"); !proto.Equal(got, ok) {
		t.Fatalf("acknowledged run changed across the quota: %v", got)
	}
	if resp := e.startWF(failed, "Echo", "payload-payload"); !resp.Started {
		t.Fatalf("start after raising the quota: %+v", resp)
	}
}

// armed records what the server asks the timer service to do.
type armed struct{ log []string }

func (a *armed) Schedule(k server.TimerKey, at time.Time) {
	a.log = append(a.log, fmt.Sprintf("schedule %s/%s %d", k.RunID, k.TimerID, at.UnixMilli()))
}
func (a *armed) Cancel(k server.TimerKey) { a.log = append(a.log, "cancel "+k.RunID+"/"+k.TimerID) }

func ev(id int64, attrs any) *durablev1.HistoryEvent {
	e := &durablev1.HistoryEvent{EventId: id, TsUnixMs: 1000}
	switch a := attrs.(type) {
	case *durablev1.WorkflowExecutionStarted:
		e.Attrs = &durablev1.HistoryEvent_Started{Started: a}
	case *durablev1.WorkflowTaskScheduled:
		e.Attrs = &durablev1.HistoryEvent_WtScheduled{WtScheduled: a}
	case *durablev1.WorkflowTaskStarted:
		e.Attrs = &durablev1.HistoryEvent_WtStarted{WtStarted: a}
	case *durablev1.WorkflowTaskCompleted:
		e.Attrs = &durablev1.HistoryEvent_WtCompleted{WtCompleted: a}
	case *durablev1.TimerStarted:
		e.Attrs = &durablev1.HistoryEvent_TimerStarted{TimerStarted: a}
	case *durablev1.TimerFired:
		e.Attrs = &durablev1.HistoryEvent_TimerFired{TimerFired: a}
	case *durablev1.WorkflowExecutionContinuedAsNew:
		e.Attrs = &durablev1.HistoryEvent_Continued{Continued: a}
	}
	return e
}

func prefix(typ string) []*durablev1.HistoryEvent {
	return []*durablev1.HistoryEvent{
		ev(1, &durablev1.WorkflowExecutionStarted{WorkflowType: typ, TaskQueue: "default"}),
		ev(2, &durablev1.WorkflowTaskScheduled{TaskQueue: "default", Attempt: 1}),
		ev(3, &durablev1.WorkflowTaskStarted{ScheduledEventId: 2}),
		ev(4, &durablev1.WorkflowTaskCompleted{ScheduledEventId: 2, StartedEventId: 3}),
	}
}

func TestRecoveryRearmsTimers(t *testing.T) {
	// WHY: a timer is durable because recovery re-arms every TimerStarted
	//      that has not fired or been canceled, at its recorded fire_at
	//      (dur.07's service then fires it); a fired timer is not re-armed.
	// KIND: fault
	// CATCHES: s24
	// CHAPTER: dur.02 section 2, recovery
	e := newEnv(t)
	lg, _ := dlog.Open(e.dir, dlog.Options{Sync: fastSync})
	seed(t, lg, &fixture{WorkflowID: "a", RunID: "ra", events: append(prefix("Sleep"),
		ev(5, &durablev1.TimerStarted{TimerId: "1", DurationMs: 5000, FireAtUnixMs: 6000, WorkflowTaskCompletedEventId: 4}))})
	seed(t, lg, &fixture{WorkflowID: "b", RunID: "rb", events: append(prefix("Sleep"),
		ev(5, &durablev1.TimerStarted{TimerId: "1", DurationMs: 1, FireAtUnixMs: 1001, WorkflowTaskCompletedEventId: 4}),
		ev(6, &durablev1.TimerFired{TimerId: "1", StartedEventId: 5}))})
	lg.Close()
	a := &armed{}
	e.timers = a
	e.start()
	if strings.Join(a.log, "; ") != "schedule ra/1 6000" {
		t.Fatalf("recovery armed %v; want only ra/1 at 6000", a.log)
	}
}

func TestRecoveryStartsMissingContinuedRun(t *testing.T) {
	// WHY: ContinueAsNew is two appends on two streams: the old run's
	//      WorkflowExecutionContinuedAsNew, then the new run's start. A crash
	//      in between leaves a run naming a successor that does not exist;
	//      recovery must start it, with the recorded id, type, and input.
	// KIND: fault
	// CATCHES: s25
	// CHAPTER: dur.02 section 5, Pitfalls
	e := newEnv(t)
	lg, _ := dlog.Open(e.dir, dlog.Options{Sync: fastSync})
	seed(t, lg, &fixture{WorkflowID: "chain", RunID: "old", events: append(prefix("Loop"),
		ev(5, &durablev1.WorkflowExecutionContinuedAsNew{NewRunId: "next", Input: []byte("2"), WorkflowType: "Loop", TaskQueue: "default"}))})
	lg.Close()
	e.start()
	inf := e.describe("chain")
	if inf.RunId != "next" || inf.ContinuedFromRunId != "old" || inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
		t.Fatalf("after recovery the newest run is %v", inf)
	}
	h := e.mustHistory("chain")
	if kinds(h) != "started wt_scheduled" || string(h[0].GetStarted().Input) != "2" || h[0].GetStarted().WorkflowType != "Loop" {
		t.Fatalf("the new run: %q %v", kinds(h), h[0])
	}
	if l, err := e.q.Poll(bg, "wf:default", "t", 0); err != nil || l.Task.ID != "next/wt/2" {
		t.Fatalf("its first workflow task: %q %v", l.Task.ID, err)
	}
}
