// Course tests for dur.06, deterministic replay (go/durable/workflow) and
// its server half (go/durable/server/continue.go).
package dur_06

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	"tinyllm/durable/server"
	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
	"openlearn.urmzd.com/tl/testkit/failpoint"
)

func fixture(t *testing.T, name string) string {
	t.Helper()
	p := filepath.Join(os.Getenv("TINYLLM_FIXTURES"), "dur", "histories", name)
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("fixture %s: %v (is TINYLLM_FIXTURES set?)", name, err)
	}
	return p
}

func load(t *testing.T, name string) (*workflow.HistoryFile, []*durablev1.HistoryEvent) {
	t.Helper()
	f, h, err := workflow.LoadHistory(fixture(t, name))
	if err != nil {
		t.Fatal(err)
	}
	return f, h
}

func lastResult(h []*durablev1.HistoryEvent) []byte { return h[len(h)-1].GetCompleted().GetResult() }

// hand builds the chapter's worked history of Sequence(2), event by event.
func hand() []*durablev1.HistoryEvent {
	ev := func(id int64, a any) *durablev1.HistoryEvent {
		e := &durablev1.HistoryEvent{EventId: id, TsUnixMs: 1000 * id}
		switch x := a.(type) {
		case *durablev1.WorkflowExecutionStarted:
			e.Attrs = &durablev1.HistoryEvent_Started{Started: x}
		case *durablev1.WorkflowTaskScheduled:
			e.Attrs = &durablev1.HistoryEvent_WtScheduled{WtScheduled: x}
		case *durablev1.WorkflowTaskStarted:
			e.Attrs = &durablev1.HistoryEvent_WtStarted{WtStarted: x}
		case *durablev1.WorkflowTaskCompleted:
			e.Attrs = &durablev1.HistoryEvent_WtCompleted{WtCompleted: x}
		case *durablev1.ActivityTaskScheduled:
			e.Attrs = &durablev1.HistoryEvent_ActScheduled{ActScheduled: x}
		case *durablev1.ActivityTaskStarted:
			e.Attrs = &durablev1.HistoryEvent_ActStarted{ActStarted: x}
		case *durablev1.ActivityTaskCompleted:
			e.Attrs = &durablev1.HistoryEvent_ActCompleted{ActCompleted: x}
		}
		return e
	}
	return []*durablev1.HistoryEvent{
		ev(1, &durablev1.WorkflowExecutionStarted{WorkflowType: "Sequence", TaskQueue: "default", Input: []byte("2")}),
		ev(2, &durablev1.WorkflowTaskScheduled{TaskQueue: "default", Attempt: 1}),
		ev(3, &durablev1.WorkflowTaskStarted{ScheduledEventId: 2}),
		ev(4, &durablev1.WorkflowTaskCompleted{ScheduledEventId: 2, StartedEventId: 3}),
		ev(5, &durablev1.ActivityTaskScheduled{ActivityId: "1", ActivityType: "Upper", Input: []byte(`"s1"`), WorkflowTaskCompletedEventId: 4}),
		ev(6, &durablev1.ActivityTaskStarted{ScheduledEventId: 5, Attempt: 1}),
		ev(7, &durablev1.ActivityTaskCompleted{ScheduledEventId: 5, StartedEventId: 6, Result: []byte(`"S1"`)}),
		ev(8, &durablev1.WorkflowTaskScheduled{TaskQueue: "default", Attempt: 1}),
		ev(9, &durablev1.WorkflowTaskStarted{ScheduledEventId: 8}),
		ev(10, &durablev1.WorkflowTaskCompleted{ScheduledEventId: 8, StartedEventId: 9}),
		ev(11, &durablev1.ActivityTaskScheduled{ActivityId: "2", ActivityType: "Upper", Input: []byte(`"s2"`), WorkflowTaskCompletedEventId: 10}),
		ev(12, &durablev1.ActivityTaskStarted{ScheduledEventId: 11, Attempt: 1}),
		ev(13, &durablev1.ActivityTaskCompleted{ScheduledEventId: 11, StartedEventId: 12, Result: []byte(`"S2"`)}),
		ev(14, &durablev1.WorkflowTaskScheduled{TaskQueue: "default", Attempt: 1}),
		ev(15, &durablev1.WorkflowTaskStarted{ScheduledEventId: 14}),
	}
}

func handle(t *testing.T, reg *workflow.Registry, h []*durablev1.HistoryEvent) ([]*durablev1.Command, error) {
	t.Helper()
	return reg.HandleWorkflowTask(bg, &durablev1.WorkflowTask{WorkflowId: "w", RunId: "r", WorkflowType: h[0].GetStarted().WorkflowType}, h)
}

func cmdKinds(cs []*durablev1.Command) string {
	var out []string
	for _, c := range cs {
		m := c.ProtoReflect()
		out = append(out, string(m.WhichOneof(m.Descriptor().Oneofs().ByName("cmd")).Name()))
	}
	return strings.Join(out, " ")
}

func TestReplayHandExample(t *testing.T) {
	// WHY: the chapter's worked replay of Sequence(2). Task 1 (history up to
	//      event 3) issues ScheduleActivity "1"; task 2 (up to event 9)
	//      replays task 1, matching "1" against event 5, gets "S1" from event
	//      7, and issues only "2"; task 3 replays both and completes with
	//      ["S1","S2"]. Each call is a fresh replay from event 1.
	// KIND: unit
	// CATCHES: s01, s02, s04
	// CHAPTER: dur.06 section 3, worked example
	h, reg := hand(), registry()
	for _, step := range []struct {
		upto int
		want string
		arg  string
	}{{3, "schedule_activity", `"s1"`}, {9, "schedule_activity", `"s2"`}, {15, "complete", `["S1","S2"]`}} {
		cmds, err := handle(t, reg, h[:step.upto])
		if err != nil || cmdKinds(cmds) != step.want {
			t.Fatalf("task ending at event %d: %q, %v; want %s", step.upto, cmdKinds(cmds), err, step.want)
		}
		var got string
		if sa := cmds[0].GetScheduleActivity(); sa != nil {
			got = string(sa.Input)
			if want := fmt.Sprint(map[int]string{3: "1", 9: "2"}[step.upto]); sa.ActivityId != want || sa.ActivityType != "Upper" || sa.Options.StartToCloseMs != 10000 {
				t.Fatalf("task %d scheduled %v; want activity id %s", step.upto, sa, want)
			}
		} else {
			got = string(cmds[0].GetComplete().Result)
		}
		if got != step.arg {
			t.Fatalf("task ending at event %d: payload %s; want %s", step.upto, got, step.arg)
		}
	}
}

func TestRecordedHistoriesReplay(t *testing.T) {
	// WHY: the conformance suite of the SDK: every recorded run of the course
	//      test workflows replays with your engine without ErrNondeterminism
	//      and returns exactly the result the history recorded.
	// KIND: conformance, golden
	// CATCHES: s02, s06, s10
	// CHAPTER: dur.06 section 4
	paths, _ := filepath.Glob(filepath.Join(os.Getenv("TINYLLM_FIXTURES"), "dur", "histories", "*.json"))
	if len(paths) < 8 {
		t.Fatalf("expected the recorded histories under TINYLLM_FIXTURES/dur/histories, found %d", len(paths))
	}
	for _, p := range paths {
		f, h, err := workflow.LoadHistory(p)
		if err != nil {
			t.Fatal(err)
		}
		res, err := workflow.ReplayHistoryFromFile(registry(), p)
		if h[len(h)-1].GetContinued() != nil {
			if err == nil || !strings.Contains(err.Error(), "continue as new") {
				t.Fatalf("%s: replay of a continued run returned %q, %v; want the ContinueAsNew error", filepath.Base(p), res, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s (%s): %v", filepath.Base(p), f.WorkflowType, err)
		}
		if string(res) != string(lastResult(h)) {
			t.Fatalf("%s: replay returned %s; history recorded %s", filepath.Base(p), res, lastResult(h))
		}
	}
}

func TestNondeterminismDetected(t *testing.T) {
	// WHY: a code change that alters the command sequence must be caught,
	//      not silently continued: a different activity type, a missing
	//      command, and an extra command each give ErrNondeterminism.
	// KIND: unit, fault
	// CATCHES: s09
	// CHAPTER: dur.06 section 2, determinism
	_, h := load(t, "sequence.json")
	cases := map[string]workflow.Workflow{
		"other type": func(ctx workflow.Context, in []byte) ([]byte, error) {
			workflow.ExecuteActivity[string](ctx, "Lower", "s1", opts).Get(ctx)
			return nil, nil
		},
		"fewer commands": func(ctx workflow.Context, in []byte) ([]byte, error) {
			s, err := workflow.ExecuteActivity[string](ctx, "Upper", "s1", opts).Get(ctx)
			b, _ := json.Marshal([]string{s})
			return b, err
		},
		"extra command": func(ctx workflow.Context, in []byte) ([]byte, error) {
			workflow.ExecuteActivity[string](ctx, "Upper", "s0", opts)
			return Sequence(ctx, in)
		},
	}
	// A live task: the history up to the third WorkflowTaskStarted, so two
	// past tasks are replayed before the current one.
	third, seen := 0, 0
	for i, ev := range h {
		if ev.GetWtStarted() != nil {
			if seen++; seen == 3 {
				third = i + 1
				break
			}
		}
	}
	for name, wf := range cases {
		reg := workflow.NewRegistry()
		reg.Register("Sequence", wf)
		if _, err := workflow.ReplayHistory(reg, "Sequence", h); !errors.Is(err, workflow.ErrNondeterminism) {
			t.Fatalf("%s: replay error %v; want ErrNondeterminism", name, err)
		}
		if _, err := handle(t, reg, h[:third]); !errors.Is(err, workflow.ErrNondeterminism) {
			t.Fatalf("%s: a live workflow task over the same history: %v; want ErrNondeterminism", name, err)
		}
	}
}

func TestReorderFailpointNondeterministic(t *testing.T) {
	// WHY: TL_FAILPOINTS="dur/workflow/reorder=..." flips Reorder's command
	//      order. Replaying its recorded history must then fail with
	//      ErrNondeterminism at the same event every time, and pass without it.
	// KIND: fault
	// CATCHES: s03
	// CHAPTER: dur.06 section 5, Pitfalls
	if _, err := workflow.ReplayHistoryFromFile(registry(), fixture(t, "reorder.json")); err != nil {
		t.Fatalf("without the failpoint: %v", err)
	}
	if err := failpoint.Load("dur/workflow/reorder=error(reorder)"); err != nil {
		t.Fatal(err)
	}
	defer failpoint.Load("")
	var msgs []string
	for i := 0; i < 3; i++ {
		_, err := workflow.ReplayHistoryFromFile(registry(), fixture(t, "reorder.json"))
		if !errors.Is(err, workflow.ErrNondeterminism) {
			t.Fatalf("with the failpoint: %v; want ErrNondeterminism", err)
		}
		msgs = append(msgs, err.Error())
	}
	if msgs[0] != msgs[1] || msgs[1] != msgs[2] {
		t.Fatalf("the error differs between replays: %q", msgs)
	}
}

func TestSideEffectNotReexecuted(t *testing.T) {
	// WHY: a side effect runs once, ever: replays return the recorded value
	//      and never call the function again (a new random id on every replay
	//      would fork the workflow's state).
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: dur.06 section 2, side effects
	_, h := load(t, "side-effects.json")
	before := sideCalls.Load()
	res, err := workflow.ReplayHistory(registry(), "SideEffects", h)
	if err != nil || string(res) != string(lastResult(h)) {
		t.Fatalf("replay = %s, %v; want %s", res, err, lastResult(h))
	}
	if sideCalls.Load() != before {
		t.Fatalf("replay called the side-effect function %d times", sideCalls.Load()-before)
	}
	cmds, err := handle(t, registry(), h[:3])
	if err != nil || cmdKinds(cmds) != "record_marker schedule_activity" || cmds[0].GetRecordMarker().MarkerName != "SideEffect" {
		t.Fatalf("first task issues %q, %v", cmdKinds(cmds), err)
	}
	if sideCalls.Load() != before+1 {
		t.Fatal("a first execution must call the function exactly once")
	}
}

func TestGetVersionMigration(t *testing.T) {
	// WHY: a deploy must not break runs already in flight. With GetVersion,
	//      a history recorded before the change replays on the old path
	//      (version 0, activity A, no marker), a history recorded after it
	//      replays on the new path (its "Version" marker says 1), and a new
	//      run records the marker and takes B.
	// KIND: unit, regression
	// CATCHES: s06, s11, s12
	// CHAPTER: dur.06 section 2, versioning
	for name, want := range map[string]string{"versioned-before.json": `[0,"A:v"]`, "versioned-new.json": `[1,"B:v"]`} {
		res, err := workflow.ReplayHistoryFromFile(registry(), fixture(t, name))
		if err != nil || string(res) != want {
			t.Fatalf("%s: %s, %v; want %s", name, res, err, want)
		}
	}
	_, h := load(t, "versioned-before.json")
	if strings.Contains(kinds(h), "marker") {
		t.Fatal("fixture: the old history must have no marker")
	}
	cmds, err := handle(t, registry(), h[:3])
	if err != nil || cmdKinds(cmds) != "record_marker schedule_activity" || cmds[1].GetScheduleActivity().ActivityType != "B" {
		t.Fatalf("a new run's first task: %q, %v", cmdKinds(cmds), err)
	}
	var vm map[string]any
	json.Unmarshal(cmds[0].GetRecordMarker().Details, &vm)
	if cmds[0].GetRecordMarker().MarkerName != "Version" || vm["change_id"] != "use-b" || vm["version"] != float64(1) {
		t.Fatalf("version marker %s %s", cmds[0].GetRecordMarker().MarkerName, cmds[0].GetRecordMarker().Details)
	}
}

func TestNowComesFromHistory(t *testing.T) {
	// WHY: workflow.Now is the WorkflowTaskStarted time of the task being
	//      replayed, so a replay a day later sees the same times; reading the
	//      wall clock would change the workflow's state on every replay.
	// KIND: unit
	// CATCHES: s13
	// CHAPTER: dur.06 section 2, time
	_, h := load(t, "clock.json")
	var starts []int64
	for _, ev := range h {
		if ev.GetWtStarted() != nil {
			starts = append(starts, ev.TsUnixMs)
		}
	}
	want, _ := json.Marshal([]int64{starts[0], starts[1]})
	res, err := workflow.ReplayHistory(registry(), "Clock", h)
	if err != nil || string(res) != string(want) || string(res) != string(lastResult(h)) {
		t.Fatalf("Clock replay = %s, %v; want %s (the first two WorkflowTaskStarted times)", res, err, want)
	}
}

// --- through the real server and worker -------------------------------------

func dial(t *testing.T, addr string) *grpc.ClientConn {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close() })
	return conn
}

func runWorker(t *testing.T, addr string, reg *workflow.Registry) {
	t.Helper()
	w := worker.New(dial(t, addr), worker.Options{TaskQueue: "default", Workflows: reg,
		ReconnectInitial: 10 * time.Millisecond, ReconnectMax: 50 * time.Millisecond})
	for name, fn := range activities {
		w.RegisterActivity(name, fn)
	}
	ctx, cancel := context.WithCancel(bg)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); w.Run(ctx) }()
	t.Cleanup(func() { cancel(); wg.Wait() })
}

func (e *env) waitClosed(id string, within time.Duration) *durablev1.WorkflowInfo {
	e.t.Helper()
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	deadline := time.After(within)
	for {
		if inf, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: id}); err == nil &&
			inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_RUNNING {
			return inf
		}
		select {
		case <-tick.C:
		case <-deadline:
			e.t.Fatalf("%s did not finish within %v", id, within)
		}
	}
}

func TestRecordThenReplayOwnRun(t *testing.T) {
	// WHY: your own workflows are tested the same way: run once through the
	//      server and your worker, fetch the history, replay it, and require
	//      the same result with no nondeterminism.
	// KIND: unit
	// CATCHES: s02, s07, s14
	// CHAPTER: dur.06 section 4
	e := newEnv(t)
	e.real = true
	e.start()
	runWorker(t, e.addr, registry())
	e.startWF("own", "Sequence", "3")
	inf := e.waitClosed("own", 5*time.Second)
	if inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED || string(inf.Result) != `["S1","S2","S3"]` {
		t.Fatalf("run: %v %s", inf.Status, inf.Result)
	}
	res, err := workflow.ReplayHistory(registry(), "Sequence", e.mustHistory("own"))
	if err != nil || string(res) != string(inf.Result) {
		t.Fatalf("replay of the recorded run: %s, %v", res, err)
	}
}

func TestKillServerMidWorkflowSameResult(t *testing.T) {
	// WHY: the server dies in the middle of a run and comes back: the worker
	//      reconnects, replays, and the run ends with exactly the result of an
	//      uninterrupted run, its history still replaying cleanly.
	// KIND: fault
	// CATCHES: s07, s14
	// CHAPTER: dur.06 section 4
	e := newEnv(t)
	e.real, e.wtLease = true, 300*time.Millisecond
	e.start()
	blocked := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	acts := map[string]worker.ActivityFunc{}
	for k, v := range activities {
		acts[k] = v
	}
	upper := activities["Upper"]
	acts["Upper"] = func(ctx context.Context, in []byte) ([]byte, error) {
		if string(in) == `"s2"` {
			once.Do(func() { close(blocked); <-release })
		}
		return upper(ctx, in)
	}
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default", Workflows: registry(),
		ReconnectInitial: 10 * time.Millisecond, ReconnectMax: 50 * time.Millisecond})
	for k, v := range acts {
		w.RegisterActivity(k, v)
	}
	ctx, cancel := context.WithCancel(bg)
	defer cancel()
	go w.Run(ctx)
	e.startWF("kill", "Sequence", "3")
	select {
	case <-blocked:
	case <-time.After(5 * time.Second):
		t.Fatal("never reached the second activity")
	}
	e.crash() // the server dies while activity 2 runs
	e.start()
	close(release)
	inf := e.waitClosed("kill", 10*time.Second)
	if inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED || string(inf.Result) != `["S1","S2","S3"]` {
		t.Fatalf("after a server crash: %v %s", inf.Status, inf.Result)
	}
	if _, err := workflow.ReplayHistory(registry(), "Sequence", e.mustHistory("kill")); err != nil {
		t.Fatalf("the recovered history does not replay: %v", err)
	}
}

// LongRunning emits 200-byte markers in batches of 2,000, one Tick activity
// between batches, and continues as new when the server suggests it.
func LongRunning(ctx workflow.Context, in []byte) ([]byte, error) {
	var st struct{ Done, Total int }
	json.Unmarshal(in, &st)
	pad := strings.Repeat("p", 200)
	for st.Done < st.Total {
		if workflow.ContinueAsNewSuggested(ctx) {
			next, _ := json.Marshal(st)
			return nil, workflow.ContinueAsNew(ctx, next)
		}
		for i := 0; i < 2000 && st.Done < st.Total; i++ {
			workflow.SideEffect(ctx, func() string { return pad })
			st.Done++
		}
		if _, err := workflow.ExecuteActivity[int](ctx, "Tick", st.Done, opts).Get(ctx); err != nil {
			return nil, err
		}
	}
	return json.Marshal(st.Done)
}

func TestContinueAsNewKeepsHistoryBounded(t *testing.T) {
	// WHY: a workflow with 12,000 steps must not grow one history without
	//      bound: past 10,000 events the server suggests ContinueAsNew, the
	//      workflow continues in a new run of the same workflow id, every run
	//      stays under the 20,000-event cap, and the last run completes. The
	//      histories pass 1 MiB, so the worker also fetches later pages.
	// KIND: fault, boundary
	// CATCHES: s10, s15
	// CHAPTER: dur.06 section 2, ContinueAsNew
	e := newEnv(t)
	e.real = true
	e.start()
	reg := registry()
	reg.Register("LongRunning", LongRunning)
	runWorker(t, e.addr, reg)
	e.startWF("long", "LongRunning", `{"Total":12000}`)
	inf := e.waitClosed("long", 15*time.Second)
	if inf.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED || string(inf.Result) != "12000" {
		t.Fatalf("final run: %v %s", inf.Status, inf.Result)
	}
	if inf.ContinuedFromRunId == "" {
		t.Fatal("the workflow never continued as new")
	}
	first, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: "long", RunId: inf.ContinuedFromRunId})
	if err != nil {
		t.Fatal(err)
	}
	if first.Status != durablev1.WorkflowStatus_WORKFLOW_STATUS_CONTINUED_AS_NEW || first.ContinuedAsRunId != inf.RunId ||
		first.HistoryLength <= server.ContinueAsNewEvents || first.HistoryLength >= server.MaxHistoryEvents {
		t.Fatalf("first run: status %v, %d events, continued as %q", first.Status, first.HistoryLength, first.ContinuedAsRunId)
	}
	h, _ := e.history("long", first.RunId, nil)
	if c := h[len(h)-1].GetContinued(); c == nil || c.WorkflowType != "LongRunning" || c.TaskQueue != "default" {
		t.Fatalf("first run ends with %v", h[len(h)-1])
	}
	var st struct{ Done, Total int }
	json.Unmarshal(h[len(h)-1].GetContinued().Input, &st)
	if st.Total != 12000 || st.Done < 8000 {
		t.Fatalf("carried state %+v", st)
	}
	second, _ := e.history("long", inf.RunId, nil)
	if got := second[0].GetStarted(); got.ContinuedFromRunId != first.RunId || !proto.Equal(&durablev1.WorkflowExecutionStarted{WorkflowType: "LongRunning", TaskQueue: "default", Input: h[len(h)-1].GetContinued().Input, ContinuedFromRunId: first.RunId, Identity: "test"}, got) {
		t.Fatalf("second run starts with %v", got)
	}
}

func TestServerCommandLimits(t *testing.T) {
	// WHY: markers and continue-as-new inputs above 2 MiB are refused with
	//      INVALID_ARGUMENT and append nothing (large data travels by path);
	//      a marker needs a name.
	// KIND: boundary
	// CATCHES: s16, s17, s18
	// CHAPTER: dur.06 section 4
	e := newEnv(t).start()
	e.startWF("lim", "Sequence", "1")
	wt, err := e.tasks.PollWorkflowTask(bg, &durablev1.PollRequest{TaskQueue: "default"})
	if err != nil {
		t.Fatal(err)
	}
	big := make([]byte, server.MaxPayloadBytes+1)
	for name, c := range map[string]*durablev1.Command{
		"marker details": {Cmd: &durablev1.Command_RecordMarker{RecordMarker: &durablev1.RecordMarker{MarkerName: "SideEffect", Details: big}}},
		"marker name":    {Cmd: &durablev1.Command_RecordMarker{RecordMarker: &durablev1.RecordMarker{}}},
		"can input":      {Cmd: &durablev1.Command_ContinueAsNew{ContinueAsNew: &durablev1.ContinueAsNew{Input: big}}},
	} {
		if _, err := e.tasks.CompleteWorkflowTask(bg, &durablev1.CompleteWorkflowTaskRequest{TaskToken: wt.TaskToken, Commands: []*durablev1.Command{c}}); code(err) != codes.InvalidArgument {
			t.Fatalf("%s: %v; want INVALID_ARGUMENT", name, err)
		}
	}
	if n := len(e.mustHistory("lim")); n != 3 {
		t.Fatalf("refused commands appended events: %d", n)
	}
	ok := &durablev1.Command{Cmd: &durablev1.Command_ContinueAsNew{ContinueAsNew: &durablev1.ContinueAsNew{Input: []byte("2")}}}
	if _, err := e.tasks.CompleteWorkflowTask(bg, &durablev1.CompleteWorkflowTaskRequest{TaskToken: wt.TaskToken, Commands: []*durablev1.Command{ok}}); err != nil {
		t.Fatal(err)
	}
	inf := e.describe("lim")
	if inf.RunId == "run-1" || inf.ContinuedFromRunId != "run-1" || inf.HistoryLength != 2 ||
		inf.WorkflowType != "Sequence" || inf.TaskQueue != "default" {
		t.Fatalf("after ContinueAsNew the newest run is %v", inf)
	}
}
