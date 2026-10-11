// Course tests for dur.07, durable timers: the min-heap and the timer
// service (go/durable/timer), and the server's timer commands and FireTimer
// (go/durable/server/timers.go).
package dur_07

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"tinyllm/durable/server"
	"tinyllm/durable/timer"
	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
	"openlearn.urmzd.com/tl/testkit/clock"
)

func at(s int) time.Time { return t0.Add(time.Duration(s) * time.Second) }

func TestHeapHandExample(t *testing.T) {
	// WHY: the chapter's trace: pushing a@5, b@3, c@8, d@3, e@1 (Seq 1 to 5)
	//      leaves the array [e b c a d]; the first pop returns e and leaves
	//      [b d c a]; the rest come out b, d, a, c (b before d: equal
	//      deadlines go in schedule order).
	// KIND: unit
	// CATCHES: s01, s02
	// CHAPTER: dur.07 section 3, worked example
	var h timer.Heap[string]
	for i, p := range []struct {
		k string
		s int
	}{{"a", 5}, {"b", 3}, {"c", 8}, {"d", 3}, {"e", 1}} {
		h.Push(&timer.Entry[string]{Key: p.k, At: at(p.s), Seq: uint64(i + 1)})
	}
	if got := strings.Join(h.Items(), " "); got != "e b c a d" {
		t.Fatalf("array after five pushes: [%s]; want [e b c a d]", got)
	}
	if p := h.Peek(); p.Key != "e" {
		t.Fatalf("Peek = %s", p.Key)
	}
	first := h.Pop()
	if first.Key != "e" || strings.Join(h.Items(), " ") != "b d c a" {
		t.Fatalf("Pop = %s leaving [%s]; want e leaving [b d c a]", first.Key, strings.Join(h.Items(), " "))
	}
	var rest []string
	for h.Len() > 0 {
		rest = append(rest, h.Pop().Key)
	}
	if strings.Join(rest, " ") != "b d a c" || h.Pop() != nil || h.Peek() != nil {
		t.Fatalf("remaining pops %v; want b d a c, then nil", rest)
	}
}

func TestServiceHandExample(t *testing.T) {
	// WHY: the same five timers through the service on a fake clock: at
	//      +3 s exactly, e, b, d fire in that order (b and d tie at 3 s and
	//      fire in schedule order); at +8 s a then c; nothing fires twice.
	// KIND: unit
	// CATCHES: s01, s07, s08
	// CHAPTER: dur.07 section 3, worked example
	clk := clock.NewFake(t0)
	s := timer.New[string](clk)
	for _, p := range []struct {
		k string
		s int
	}{{"a", 5}, {"b", 3}, {"c", 8}, {"d", 3}, {"e", 1}} {
		s.Schedule(p.k, at(p.s))
	}
	fired := make(chan string, 16)
	ctx, cancel := context.WithCancel(bg)
	done := make(chan struct{})
	go func() { s.Run(ctx, func(k string) { send(fired, k) }); close(done) }()
	defer func() { cancel(); <-done }()
	waitTimer(t, clk)
	clk.Advance(3 * time.Second)
	if got := take(t, fired, 3); got != "e b d" {
		t.Fatalf("fired at +3s: %s; want e b d", got)
	}
	waitTimer(t, clk)
	clk.Advance(5 * time.Second)
	if got := take(t, fired, 2); got != "a c" {
		t.Fatalf("fired at +8s: %s; want a c", got)
	}
	select {
	case k := <-fired:
		t.Fatalf("%s fired twice", k)
	case <-time.After(30 * time.Millisecond):
	}
	if s.Len() != 0 {
		t.Fatalf("%d timers left", s.Len())
	}
}

// send never blocks the service: a planted bug that fires forever must fail
// the test, not hang it.
func send(ch chan<- string, k string) {
	select {
	case ch <- k:
	default:
	}
}

func waitTimer(t *testing.T, clk *clock.Fake) {
	t.Helper()
	ok := make(chan struct{})
	go func() { clk.BlockUntil(1); close(ok) }()
	select {
	case <-ok:
	case <-time.After(5 * time.Second):
		t.Fatal("the service never waited on the clock")
	}
}

func take(t *testing.T, ch <-chan string, n int) string {
	t.Helper()
	var out []string
	for len(out) < n {
		select {
		case k := <-ch:
			out = append(out, k)
		case <-time.After(5 * time.Second):
			t.Fatalf("only %v fired; want %d timers", out, n)
		}
	}
	return strings.Join(out, " ")
}

func TestHeapMatchesSortedModel(t *testing.T) {
	// WHY: the heap invariant (no child earlier than its parent) and the
	//      pop order must hold under any mix of pushes, pops, and removals
	//      from the middle; a sorted slice is the model.
	// KIND: property
	// CATCHES: s02, s03, s04
	// CHAPTER: dur.07 section 2, the heap
	seed := uint64(1)
	for round := 0; round < 20; round++ {
		rng := newPCG32(seed, uint64(round))
		var h timer.Heap[int]
		var model []*timer.Entry[int]
		next := 0
		for op := 0; op < 300; op++ {
			switch r := rng.intN(10); {
			case r < 5 || len(model) == 0:
				e := &timer.Entry[int]{Key: next, At: at(rng.intN(20))}
				next++
				h.Push(e)
				model = append(model, e)
			case r < 8:
				sort.SliceStable(model, func(i, j int) bool { return model[i].At.Before(model[j].At) })
				got := h.Pop()
				if got == nil || !got.At.Equal(model[0].At) {
					t.Fatalf("round %d op %d: Pop gave %v, the earliest deadline is %v", round, op, got, model[0].At)
				}
				for i, e := range model {
					if e == got {
						model = append(model[:i], model[i+1:]...)
						break
					}
				}
			default:
				i := rng.intN(len(model))
				if h.Remove(model[i]) != model[i] {
					t.Fatalf("round %d op %d: Remove of a member failed", round, op)
				}
				model = append(model[:i], model[i+1:]...)
			}
			checkHeap(t, &h, model)
		}
	}
}

func checkHeap(t *testing.T, h *timer.Heap[int], model []*timer.Entry[int]) {
	t.Helper()
	if h.Len() != len(model) {
		t.Fatalf("heap has %d entries, model %d", h.Len(), len(model))
	}
	atOf := map[int]time.Time{}
	for _, e := range model {
		atOf[e.Key] = e.At
	}
	keys := h.Items()
	for i := 1; i < len(keys); i++ {
		if atOf[keys[i]].Before(atOf[keys[(i-1)/2]]) {
			t.Fatalf("heap order broken at %d: child %v before parent %v", i, atOf[keys[i]], atOf[keys[(i-1)/2]])
		}
	}
}

func TestServiceRescheduleAndCancel(t *testing.T) {
	// WHY: recovery re-arms timers the service may already hold: scheduling
	//      a key again moves it, never doubles it. A canceled timer never
	//      fires, and a timer already past due fires as soon as Run starts.
	// KIND: unit, boundary
	// CATCHES: s05, s06
	// CHAPTER: dur.07 section 2, the service
	clk := clock.NewFake(t0)
	s := timer.New[string](clk)
	s.Schedule("late", t0.Add(-time.Minute))
	s.Schedule("x", at(10))
	s.Schedule("x", at(2)) // moved earlier
	s.Schedule("y", at(4))
	s.Schedule("gone", at(1))
	s.Cancel("gone")
	s.Cancel("never-armed")
	if s.Len() != 3 {
		t.Fatalf("Len = %d; want 3 (late, x, y)", s.Len())
	}
	fired := make(chan string, 16)
	ctx, cancel := context.WithCancel(bg)
	done := make(chan struct{})
	go func() { s.Run(ctx, func(k string) { send(fired, k) }); close(done) }()
	defer func() { cancel(); <-done }()
	if got := take(t, fired, 1); got != "late" {
		t.Fatalf("first fire %s; want the past-due timer", got)
	}
	waitTimer(t, clk)
	clk.Advance(10 * time.Second)
	if got := take(t, fired, 2); got != "x y" {
		t.Fatalf("fired %s; want x y", got)
	}
	select {
	case k := <-fired:
		t.Fatalf("%s fired; want nothing more", k)
	case <-time.After(30 * time.Millisecond):
	}
}

func TestDebugClockShiftsTime(t *testing.T) {
	// WHY: `--test-clock` serves POST /debug/clock {"offset_ms": N}: time
	//      jumps forward, every timer that became due fires once right away
	//      (no waiting for the old wall-clock instant), later ones wait, and
	//      time never moves backwards (the clock-skew drill relies on it).
	// KIND: unit, fault
	// CATCHES: s09, s10
	// CHAPTER: dur.07 section 4
	base := clock.NewFake(t0)
	oc := timer.NewOffsetClock(base)
	s := timer.New[string](oc)
	oc.OnShift(s.Wake)
	s.Schedule("1h", t0.Add(time.Hour))
	s.Schedule("2h", t0.Add(2*time.Hour))
	s.Schedule("3h", t0.Add(3*time.Hour))
	fired := make(chan string, 16)
	ctx, cancel := context.WithCancel(bg)
	done := make(chan struct{})
	go func() { s.Run(ctx, func(k string) { send(fired, k) }); close(done) }()
	defer func() { cancel(); <-done }()
	srv := httptest.NewServer(oc.Handler())
	defer srv.Close()
	post := func(body string) (int, string) {
		resp, err := http.Post(srv.URL, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		json.NewDecoder(resp.Body).Decode(&out)
		return resp.StatusCode, fmt.Sprint(out["offset_ms"])
	}
	if code, off := post(`{"offset_ms": 7200000}`); code != 200 || off != "7.2e+06" {
		t.Fatalf("POST 2h: %d %s", code, off)
	}
	if got := take(t, fired, 2); got != "1h 2h" {
		t.Fatalf("after +2h fired %s; want 1h 2h", got)
	}
	select {
	case k := <-fired:
		t.Fatalf("%s fired early", k)
	case <-time.After(30 * time.Millisecond):
	}
	if code, _ := post(`{"offset_ms": -1}`); code != 400 {
		t.Fatalf("a negative offset: %d; want 400", code)
	}
	if code, _ := post(`nonsense`); code != 400 {
		t.Fatalf("a malformed body: %d; want 400", code)
	}
	post(`{"offset_ms": 3600000}`)
	if got := take(t, fired, 1); got != "3h" {
		t.Fatalf("after +3h fired %s", got)
	}
	if !oc.Now().Equal(t0.Add(3*time.Hour)) || oc.Offset() != 3*time.Hour {
		t.Fatalf("clock reads %v (offset %v)", oc.Now(), oc.Offset())
	}
	resp, _ := http.Get(srv.URL)
	var out map[string]int64
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if out["offset_ms"] != 3*3600000 {
		t.Fatalf("GET offset %v", out)
	}
}

// --- through the server -------------------------------------------------------

func (e *env) pollWT() *durablev1.WorkflowTask {
	e.t.Helper()
	c, cancel := context.WithTimeout(bg, 5*time.Second)
	defer cancel()
	wt, err := e.tasks.PollWorkflowTask(c, &durablev1.PollRequest{TaskQueue: "default", Identity: "wk"})
	if err != nil || len(wt.TaskToken) == 0 {
		e.t.Fatalf("PollWorkflowTask: %v", err)
	}
	return wt
}

func (e *env) complete(wt *durablev1.WorkflowTask, cmds ...*durablev1.Command) error {
	_, err := e.tasks.CompleteWorkflowTask(bg, &durablev1.CompleteWorkflowTaskRequest{TaskToken: wt.TaskToken, Commands: cmds})
	return err
}

func startTimerCmd(id string, d time.Duration) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_StartTimer{StartTimer: &durablev1.StartTimer{TimerId: id, DurationMs: d.Milliseconds()}}}
}

func cancelTimerCmd(id string) *durablev1.Command {
	return &durablev1.Command{Cmd: &durablev1.Command_CancelTimer{CancelTimer: &durablev1.CancelTimer{TimerId: id}}}
}

func (e *env) eventually(what string, cond func() bool) {
	e.t.Helper()
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	deadline := time.After(5 * time.Second)
	for !cond() {
		select {
		case <-tick.C:
		case <-deadline:
			e.t.Fatalf("timed out waiting for %s", what)
		}
	}
}

func TestStartTimerHand(t *testing.T) {
	// WHY: StartTimer records TimerStarted with fire_at = now + duration (the
	//      deadline survives restarts because it is in history); at exactly
	//      fire_at the server appends TimerFired and schedules a workflow task.
	// KIND: unit
	// CATCHES: s11
	// CHAPTER: dur.07 section 3, worked example
	e := newEnv(t).start()
	e.startWF("sleep", "SleepDemo", "")
	if err := e.complete(e.pollWT(), startTimerCmd("1", 30*time.Second)); err != nil {
		t.Fatal(err)
	}
	h := e.mustHistory("sleep")
	ts := h[4].GetTimerStarted()
	if ts == nil || ts.TimerId != "1" || ts.DurationMs != 30000 || ts.FireAtUnixMs != t0.Add(30*time.Second).UnixMilli() || ts.WorkflowTaskCompletedEventId != 4 {
		t.Fatalf("event 5: %v", h[4])
	}
	e.waitArmed(t0.Add(30 * time.Second))
	e.clk.Advance(30*time.Second - time.Millisecond)
	if n := len(e.firedKeys()); n != 0 {
		t.Fatalf("fired %d timers before the deadline", n)
	}
	e.clk.Advance(time.Millisecond)
	e.eventually("TimerFired", func() bool { return len(e.mustHistory("sleep")) == 7 })
	h = e.mustHistory("sleep")
	if kinds(h[5:]) != "timer_fired wt_scheduled" || h[5].GetTimerFired().StartedEventId != 5 || h[5].TsUnixMs != t0.Add(30*time.Second).UnixMilli() {
		t.Fatalf("after the deadline: %q", kinds(h))
	}
}

func TestTimerCommandsValidated(t *testing.T) {
	// WHY: a timer needs an id and a positive duration, one pending timer per
	//      id; CancelTimer of a pending timer records TimerCanceled and it
	//      never fires; canceling an unknown timer is INVALID_ARGUMENT.
	// KIND: boundary
	// CATCHES: s12, s13, s06
	// CHAPTER: dur.07 section 4
	e := newEnv(t).start()
	e.startWF("t", "SleepDemo", "")
	wt := e.pollWT()
	for name, cmds := range map[string][]*durablev1.Command{
		"zero duration":  {startTimerCmd("1", 0)},
		"no id":          {startTimerCmd("", time.Second)},
		"same id twice":  {startTimerCmd("1", time.Second), startTimerCmd("1", time.Second)},
		"cancel unknown": {cancelTimerCmd("9")},
	} {
		if err := e.complete(wt, cmds...); code(err).String() != "InvalidArgument" {
			t.Fatalf("%s: %v; want INVALID_ARGUMENT", name, err)
		}
	}
	if err := e.complete(wt, startTimerCmd("1", 10*time.Second), startTimerCmd("2", 60*time.Second)); err != nil {
		t.Fatal(err)
	}
	e.waitArmed(t0.Add(10 * time.Second))
	e.clk.Advance(10 * time.Second) // timer 1 fires and wakes the workflow
	e.eventually("timer 1", func() bool { return strings.Contains(kinds(e.mustHistory("t")), "timer_fired") })
	wt2 := e.pollWT()
	if err := e.complete(wt2, startTimerCmd("2", time.Second)); code(err).String() != "InvalidArgument" {
		t.Fatalf("restarting pending timer 2: %v", err)
	}
	if err := e.complete(wt2, cancelTimerCmd("2")); err != nil {
		t.Fatal(err)
	}
	h := e.mustHistory("t")
	if c := h[len(h)-1].GetTimerCanceled(); c == nil || c.TimerId != "2" || c.StartedEventId != 6 {
		t.Fatalf("history ends %q", kinds(h[len(h)-2:]))
	}
	e.clk.Advance(time.Hour)
	if e.ts.Len() != 0 || strings.Count(kinds(e.mustHistory("t")), "timer_fired") != 1 {
		t.Fatalf("the canceled timer is armed (%d) or fired: %q", e.ts.Len(), kinds(e.mustHistory("t")))
	}
}

func TestFireTimerIsIdempotent(t *testing.T) {
	// WHY: exactly once lives in FireTimer: a duplicate call (two services,
	//      a re-arm racing a fire) or a call for an unknown run or timer must
	//      append nothing.
	// KIND: unit, fault
	// CATCHES: s14
	// CHAPTER: dur.07 section 5, Pitfalls
	e := newEnv(t).start()
	e.startWF("idem", "SleepDemo", "")
	e.complete(e.pollWT(), startTimerCmd("1", time.Hour))
	k := server.TimerKey{RunID: "run-1", TimerID: "1"}
	e.srv.FireTimer(k)
	e.srv.FireTimer(k)
	e.srv.FireTimer(server.TimerKey{RunID: "run-1", TimerID: "7"})
	e.srv.FireTimer(server.TimerKey{RunID: "nope", TimerID: "1"})
	if got := kinds(e.mustHistory("idem")[5:]); got != "timer_fired wt_scheduled" {
		t.Fatalf("after duplicate fires: %q", got)
	}
}

func TestFireTimerRetriesWhenAppendFails(t *testing.T) {
	// WHY: a timer is popped from the service before FireTimer appends its
	//      TimerFired; when that append fails (disk full, log closed) the
	//      server must arm the timer again, or the workflow sleeps forever.
	// KIND: fault
	// CATCHES: s15
	// CHAPTER: dur.07 section 5, Pitfalls
	e := newEnv(t).start()
	e.startWF("lost", "SleepDemo", "")
	e.complete(e.pollWT(), startTimerCmd("1", time.Minute))
	e.log.Close() // every append fails from now on
	e.waitArmed(t0.Add(time.Minute))
	e.clk.Advance(time.Minute)
	e.eventually("the fire attempt", func() bool { return len(e.firedKeys()) == 1 })
	e.eventually("the timer to be armed again", func() bool { return e.ts.Len() == 1 })
}

func TestRestartWithPendingTimers(t *testing.T) {
	// WHY: timers outlive the process: three runs sleep 10, 20, and 30 s;
	//      the server restarts (recovery re-arms all three), the first fires,
	//      the server restarts again (only two are re-armed), and the rest
	//      fire. Each fires exactly once, in deadline order.
	// KIND: fault
	// CATCHES: s11
	// CHAPTER: dur.07 section 4, restart
	e := newEnv(t).start()
	for i, d := range []time.Duration{20 * time.Second, 10 * time.Second, 30 * time.Second} {
		id := fmt.Sprintf("w%d", i)
		e.startWF(id, "SleepDemo", "")
		if err := e.complete(e.pollWT(), startTimerCmd("1", d)); err != nil {
			t.Fatal(err)
		}
	}
	e.restart()
	if e.ts.Len() != 3 {
		t.Fatalf("after restart %d timers armed; want 3", e.ts.Len())
	}
	e.waitArmed(t0.Add(10 * time.Second))
	e.clk.Advance(15 * time.Second)
	e.eventually("the 10 s timer", func() bool { return len(e.firedKeys()) == 1 })
	e.restart()
	if e.ts.Len() != 2 {
		t.Fatalf("after the second restart %d timers armed; want 2 (the fired one stays fired)", e.ts.Len())
	}
	e.waitArmed(t0.Add(20 * time.Second))
	e.clk.Advance(20 * time.Second)
	e.eventually("the rest", func() bool { return len(e.firedKeys()) == 3 })
	var order []string
	for _, k := range e.firedKeys() {
		order = append(order, k.RunID)
	}
	if strings.Join(order, " ") != "run-2 run-1 run-3" {
		t.Fatalf("fire order %v; want run-2 (10 s), run-1 (20 s), run-3 (30 s)", order)
	}
	for i := 0; i < 3; i++ {
		if n := strings.Count(kinds(e.mustHistory(fmt.Sprintf("w%d", i))), "timer_fired"); n != 1 {
			t.Fatalf("w%d has %d TimerFired events", i, n)
		}
	}
}

// SleepDemo is `wf start SleepDemo --for 30s`: sleep, then report the time.
func SleepDemo(ctx workflow.Context, in []byte) ([]byte, error) {
	var ms int64
	json.Unmarshal(in, &ms)
	if err := workflow.Sleep(ctx, time.Duration(ms)*time.Millisecond); err != nil {
		return nil, err
	}
	return json.Marshal(workflow.Now(ctx).UnixMilli())
}

func TestSleepWorkflowSurvivesRestart(t *testing.T) {
	// WHY: the milestone's demo: `wf start SleepDemo --for 30s`, kill and
	//      restart the server, and the run still wakes once at its deadline
	//      and completes, through the SDK's workflow.Sleep and your worker.
	// KIND: fault
	// CATCHES: s11
	// CHAPTER: dur.07 section 1
	e := newEnv(t).start()
	reg := workflow.NewRegistry()
	reg.Register("SleepDemo", SleepDemo)
	conn, err := grpc.NewClient(e.addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	w := worker.New(conn, worker.Options{TaskQueue: "default", Workflows: reg, ReconnectInitial: 10 * time.Millisecond, ReconnectMax: 50 * time.Millisecond})
	ctx, cancel := context.WithCancel(bg)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); w.Run(ctx) }()
	defer func() { cancel(); wg.Wait() }()
	e.startWF("demo", "SleepDemo", "30000")
	e.eventually("the timer to start", func() bool { return strings.Contains(kinds(e.mustHistory("demo")), "timer_started") })
	e.restart()
	e.waitArmed(t0.Add(30 * time.Second))
	e.clk.Advance(30 * time.Second)
	e.eventually("the run to complete", func() bool {
		inf, err := e.wf.DescribeWorkflow(bg, &durablev1.DescribeWorkflowRequest{WorkflowId: "demo"})
		return err == nil && inf.Status == durablev1.WorkflowStatus_WORKFLOW_STATUS_COMPLETED
	})
	h := e.mustHistory("demo")
	if strings.Count(kinds(h), "timer_fired") != 1 {
		t.Fatalf("history %q", kinds(h))
	}
	var woke int64
	json.Unmarshal(e.describe("demo").Result, &woke)
	if woke < t0.Add(30*time.Second).UnixMilli() {
		t.Fatalf("woke at %v, before the deadline", time.UnixMilli(woke).Sub(t0))
	}
}

// -- frozen PCG32 ---------------------------------------------------------------------

// pcg32 transcribes course/tests/_lib/pcg32.py (spec/pcg32.md): PCG-XSH-RR
// 64/32, seeded as pcg32_srandom_r(seed, seq). Course tests never import
// math/rand (D35).
type pcg32 struct{ state, inc uint64 }

func newPCG32(seed, seq uint64) *pcg32 {
	p := &pcg32{inc: seq<<1 | 1}
	p.next()
	p.state += seed
	p.next()
	return p
}

func (p *pcg32) next() uint32 {
	old := p.state
	p.state = old*6364136223846793005 + p.inc
	xs := uint32(((old >> 18) ^ old) >> 27)
	rot := uint32(old >> 59)
	return xs>>rot | xs<<((-rot)&31)
}

// below is the unbiased draw of spec/pcg32.md: uniform in [0, n).
func (p *pcg32) below(n uint32) uint32 {
	t := -n % n
	for {
		if r := p.next(); r >= t {
			return r % n
		}
	}
}

// intN is uniform in [0, n).
func (p *pcg32) intN(n int) int { return int(p.below(uint32(n))) }
