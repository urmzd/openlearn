// Course tests for dur.09, the Go half of a subprocess activity
// (go/activities/subprocess.go; contracts/spec/subprocess-activity.md).
//
// The Python side is played by a scripted fake child,
// fixtures/dur.09/fake_child.py, run with python3: each test writes the
// script into the spec and checks what the runner did with it (the argv and
// environment the child saw, the heartbeats, the result or the Failure). No
// test sleeps for a fixed time: waits poll a condition with a deadline, so a
// correct runner passes in well under a second per test and a broken one
// fails with a message instead of hanging.
package dur_09

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"tinyllm/activities"
)

const patience = 10 * time.Second

func fakeChild(t *testing.T) string {
	t.Helper()
	p := filepath.Join(os.Getenv("TINYLLM_FIXTURES"), "dur.09", "fake_child.py")
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("fixture missing: %v (TINYLLM_FIXTURES=%q)", err, os.Getenv("TINYLLM_FIXTURES"))
	}
	return p
}

func python(t *testing.T) string {
	t.Helper()
	p, err := exec.LookPath("python3")
	if err != nil {
		t.Fatal("python3 is not on PATH")
	}
	return p
}

// runner is a Subprocess over the fake child with short timings.
func runner(t *testing.T) (activities.Subprocess, string) {
	t.Helper()
	art := t.TempDir()
	return activities.Subprocess{
		Entry:     []string{python(t), fakeChild(t)},
		Artifacts: art,
		Grace:     2 * time.Second,
		Poll:      20 * time.Millisecond,
		Env:       []string{"PATH=" + os.Getenv("PATH"), "HOME=" + os.Getenv("HOME")},
	}, art
}

// script is the fake child's spec for every attempt.
func script(onTerm string, actions ...map[string]any) []byte {
	b, _ := json.Marshal(map[string]any{"on_term": onTerm, "attempts": map[string]any{"*": actions}})
	return b
}

func act(k string, v any) map[string]any { return map[string]any{k: v} }

func emit(fields map[string]any) map[string]any { return act("emit", fields) }

// beats records every heartbeat and answers cancel once cancelIf says so.
type beats struct {
	mu       sync.Mutex
	details  []string
	cancelIf func() bool // nil: never cancel
}

func (b *beats) hb(ctx context.Context, d []byte) (bool, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.details = append(b.details, string(d))
	return b.cancelIf != nil && b.cancelIf(), nil
}

// started reports whether the child has written its first progress line,
// which it does after installing its SIGTERM handler: a signal sent earlier
// would test Python's default action, not the runner.
func started(art, key string) func() bool {
	p := filepath.Join(art, "activities", filepath.FromSlash(key), "progress.jsonl")
	return func() bool {
		b, err := os.ReadFile(p)
		return err == nil && strings.Contains(string(b), "\n")
	}
}

func (b *beats) all() []string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]string(nil), b.details...)
}

func (b *beats) last() string {
	all := b.all()
	if len(all) == 0 {
		return ""
	}
	return all[len(all)-1]
}

type record struct {
	Argv []string          `json:"argv"`
	Env  map[string]string `json:"env"`
	Pid  int               `json:"pid"`
	Pgid int               `json:"pgid"`
}

func readRecord(t *testing.T, dir, name string) record {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name+".json"))
	if err != nil {
		t.Fatalf("the child never recorded %s: %v", name, err)
	}
	var r record
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func failure(t *testing.T, err error) *activities.Failure {
	t.Helper()
	var f *activities.Failure
	if !errors.As(err, &f) {
		t.Fatalf("want a *activities.Failure, got %T: %v", err, err)
	}
	return f
}

func runCtx(t *testing.T) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), patience)
	t.Cleanup(cancel)
	return ctx
}

func TestHandExampleResumeFlow(t *testing.T) {
	// WHY: the chapter's worked example (section 3). Attempt 1 of activity
	//      train-1/3 checkpoints at step 500 and dies (exit 75); the runner
	//      heartbeats the checkpoint path. Attempt 2 is started with that
	//      path as --resume, attempt 2 in TL_ATTEMPT, the same work dir and
	//      key, finishes, and its DONE.json is the activity's result.
	// KIND: unit
	// CATCHES: s01, s03
	// CHAPTER: dur.09 section 3, worked example
	r, art := runner(t)
	ckpt := "runs/train-1/ckpt/step-000500"
	spec, _ := json.Marshal(map[string]any{"attempts": map[string]any{
		"1": []map[string]any{
			act("record", "attempt1"),
			emit(map[string]any{"kind": "step", "step": 500, "loss": 2.31, "lr": 0.0009, "tokens": 8192000}),
			emit(map[string]any{"kind": "ckpt", "step": 500, "ckpt": ckpt}),
			act("sleep", 0.3),
			act("stderr", "CUDA out of memory (pretend)\n"),
			act("exit", 75),
		},
		"2": []map[string]any{
			act("record", "attempt2"),
			act("done", []string{"runs/train-1/ckpt/step-001000"}),
		},
	}})
	b1 := &beats{}
	task := activities.Task{IdempotencyKey: "train-1/3", Attempt: 1, HeartbeatTimeout: 150 * time.Millisecond}
	_, err := r.Run(runCtx(t), task, []string{"train"}, spec, b1.hb)
	f := failure(t, err)
	if f.Type != "ExitCode75" || f.NonRetryable() || f.ExitCode != 75 {
		t.Fatalf("attempt 1: want a retryable ExitCode75, got %+v", f)
	}
	if !strings.Contains(f.Message, "CUDA out of memory") {
		t.Fatalf("Failure.message must carry the child's stderr tail, got %q", f.Message)
	}
	if b1.last() != ckpt {
		t.Fatalf("the last heartbeat must carry the checkpoint path %q, got %q (all: %q)", ckpt, b1.last(), b1.all())
	}
	dir := filepath.Join(art, "activities", "train-1", "3")
	rec1 := readRecord(t, dir, "attempt1")
	if want := []string{"train", "--spec", filepath.Join(dir, "spec.json"), "--progress", filepath.Join(dir, "progress.jsonl")}; strings.Join(rec1.Argv[1:], " ") != strings.Join(want, " ") {
		t.Fatalf("attempt 1 argv after the entry:\n got %q\nwant %q", rec1.Argv[1:], want)
	}

	task.Attempt, task.LastHeartbeat = 2, []byte(b1.last())
	res, err := r.Run(runCtx(t), task, []string{"train"}, spec, (&beats{}).hb)
	if err != nil {
		t.Fatalf("attempt 2: %v", err)
	}
	if string(res) != `{"outputs": ["runs/train-1/ckpt/step-001000"]}` {
		t.Fatalf("the result is DONE.json's bytes, got %q", res)
	}
	rec2 := readRecord(t, dir, "attempt2")
	tailArgs := rec2.Argv[len(rec2.Argv)-2:]
	if tailArgs[0] != "--resume" || tailArgs[1] != ckpt {
		t.Fatalf("attempt 2 must end its argv with --resume %s, got %q", ckpt, rec2.Argv)
	}
	if rec2.Env["TL_ATTEMPT"] != "2" || rec2.Env["TL_IDEMPOTENCY_KEY"] != "train-1/3" {
		t.Fatalf("attempt 2 env: TL_ATTEMPT=%q TL_IDEMPOTENCY_KEY=%q", rec2.Env["TL_ATTEMPT"], rec2.Env["TL_IDEMPOTENCY_KEY"])
	}
}

func TestWorkDirFromKey(t *testing.T) {
	// WHY: every attempt of one activity must land in the same directory,
	//      named by the idempotency key, and a key must never escape the
	//      activities directory (a workflow id is caller-chosen text).
	// KIND: boundary
	// CATCHES: s02
	// CHAPTER: dur.09 section 2.2
	got, err := activities.WorkDir("/artifacts", "corpus/tinystories/v1/7")
	if err != nil || got != "/artifacts/activities/corpus/tinystories/v1/7" {
		t.Fatalf("WorkDir: %q, %v", got, err)
	}
	for _, bad := range []string{"", "a//b", "../etc", "a/../../b", "a/./b", "a b/c", "a/b/", "/abs"} {
		if p, err := activities.WorkDir("/artifacts", bad); err == nil {
			t.Fatalf("key %q must be rejected, got %q", bad, p)
		}
	}
	r, _ := runner(t)
	_, err = r.Run(runCtx(t), activities.Task{IdempotencyKey: "../escape", Attempt: 1}, nil, script("exit130"), (&beats{}).hb)
	if f := failure(t, err); !f.NonRetryable() {
		t.Fatalf("a bad key can never succeed: it must be non-retryable, got %+v", f)
	}
}

func TestEnvironmentContract(t *testing.T) {
	// WHY: the child learns who it is only from its environment: the key
	//      (where outputs go), the attempt, the artifact root, and the
	//      activity span's traceparent. Values inherited from the worker's
	//      own environment must be replaced, never leaked: a stale
	//      TRACEPARENT puts Python spans in the wrong trace.
	// KIND: unit
	// CATCHES: s04
	// CHAPTER: dur.09 section 4, The interface
	r, art := runner(t)
	r.Env = append(r.Env, "TRACEPARENT=00-ffffffffffffffffffffffffffffffff-ffffffffffffffff-01",
		"TL_IDEMPOTENCY_KEY=stale", "TL_ATTEMPT=9", "KEEP_ME=1")
	r.OTLPEndpoint, r.ServiceName = "http://collector:4318", "forge-python"
	tp := "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	task := activities.Task{IdempotencyKey: "wf-env/1", Attempt: 3, TraceContext: map[string]string{"traceparent": tp}}
	if _, err := r.Run(runCtx(t), task, []string{"eval"}, script("exit130", act("record", "env"), act("done", []string{})), (&beats{}).hb); err != nil {
		t.Fatal(err)
	}
	env := readRecord(t, filepath.Join(art, "activities", "wf-env", "1"), "env").Env
	want := map[string]string{
		"TL_IDEMPOTENCY_KEY": "wf-env/1", "TL_ATTEMPT": "3", "TL_ARTIFACTS": art, "TRACEPARENT": tp,
		"OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318", "OTEL_SERVICE_NAME": "forge-python", "KEEP_ME": "1",
	}
	for k, v := range want {
		if env[k] != v {
			t.Errorf("%s = %q, want %q", k, env[k], v)
		}
	}

	// A task with no trace context: the worker's own TRACEPARENT must not
	// leak into the child.
	task = activities.Task{IdempotencyKey: "wf-env/2", Attempt: 1}
	if _, err := r.Run(runCtx(t), task, []string{"eval"}, script("exit130", act("record", "env"), act("done", []string{})), (&beats{}).hb); err != nil {
		t.Fatal(err)
	}
	env = readRecord(t, filepath.Join(art, "activities", "wf-env", "2"), "env").Env
	if v, ok := env["TRACEPARENT"]; ok {
		t.Fatalf("TRACEPARENT leaked from the worker's environment: %q", v)
	}
}

func TestExitCodeTable(t *testing.T) {
	// WHY: the exit code is the whole verdict (contract table): 65 is a
	//      wrong input and must never be retried, 75 and crashes are
	//      retryable, 130 is a cancellation, a signal death is a crash.
	//      Getting one row wrong either retries a poison input forever or
	//      gives up on a transient failure.
	// KIND: unit
	// CATCHES: s05, s06
	// CHAPTER: dur.09 section 2.4
	cases := []struct {
		actions      []map[string]any
		typ          string
		nonRetryable bool
		code         int
	}{
		{[]map[string]any{act("exit", 65)}, "ExitCode65", true, 65},
		{[]map[string]any{act("exit", 75)}, "ExitCode75", false, 75},
		{[]map[string]any{act("exit", 1)}, "ExitCode1", false, 1},
		{[]map[string]any{act("exit", 2)}, "ExitCode2", false, 2},
		{[]map[string]any{act("exit", 130)}, "Canceled", true, 130},
		{[]map[string]any{act("kill_self", int(syscall.SIGSEGV))}, "Signal11", false, -1},
		{[]map[string]any{act("kill_self", int(syscall.SIGKILL))}, "Signal9", false, -1},
	}
	r, _ := runner(t)
	for i, c := range cases {
		_, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "codes/" + strconv.Itoa(i), Attempt: 1}, nil, script("exit130", c.actions...), (&beats{}).hb)
		f := failure(t, err)
		if f.Type != c.typ || f.NonRetryable() != c.nonRetryable || f.ExitCode != c.code {
			t.Errorf("case %d %v: got {Type:%s NonRetryable:%v ExitCode:%d}, want {%s %v %d}", i, c.actions, f.Type, f.NonRetryable(), f.ExitCode, c.typ, c.nonRetryable, c.code)
		}
	}
}

func TestSuccessNeedsDoneFile(t *testing.T) {
	// WHY: exit 0 alone is not success: the contract's last act is
	//      DONE.json, and its bytes are the result. A child that exits 0
	//      without it (a broken entry) must be retried, and a DONE.json over
	//      2 MiB cannot travel as a result at all.
	// KIND: boundary
	// CATCHES: s07
	// CHAPTER: dur.09 section 2.4
	r, _ := runner(t)
	_, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "nodone/1", Attempt: 1}, nil, script("exit130"), (&beats{}).hb)
	if f := failure(t, err); f.Type != "MissingDone" || f.NonRetryable() {
		t.Fatalf("exit 0 without DONE.json: want a retryable MissingDone, got %+v", f)
	}
	_, err = r.Run(runCtx(t), activities.Task{IdempotencyKey: "nodone/2", Attempt: 1}, nil, script("exit130", act("big_done", activities.MaxResult+1)), (&beats{}).hb)
	if f := failure(t, err); !f.NonRetryable() {
		t.Fatalf("a DONE.json over 2 MiB: want a non-retryable failure, got %+v", f)
	}
	res, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "nodone/3", Attempt: 1}, nil, script("exit130", act("big_done", activities.MaxResult)), (&beats{}).hb)
	if err != nil || len(res) != activities.MaxResult {
		t.Fatalf("a DONE.json of exactly 2 MiB is a result: len %d, %v", len(res), err)
	}
}

func TestTailKeepsPartialLines(t *testing.T) {
	// WHY: the runner reads a file the child is still writing. A line
	//      without its newline is half an event: parsed early it is either
	//      dropped forever or misread. Only complete lines count.
	// KIND: unit
	// CATCHES: s08
	// CHAPTER: dur.09 section 2.3
	p := filepath.Join(t.TempDir(), "progress.jsonl")
	tl := &activities.Tail{Path: p}
	if evs, err := tl.Read(); err != nil || len(evs) != 0 {
		t.Fatalf("a missing file is no events yet: %v, %v", evs, err)
	}
	f, _ := os.Create(p)
	defer f.Close()
	f.WriteString(`{"ts":1,"kind":"step","step":1}` + "\n" + `{"ts":2,"kind":"ck`)
	evs, _ := tl.Read()
	if len(evs) != 1 || evs[0].Step != 1 {
		t.Fatalf("after one complete line and half of another: want 1 event, got %+v", evs)
	}
	f.WriteString(`pt","step":2,"ckpt":"runs/a/ckpt/step-000002"}` + "\n" + "not json\n")
	evs, _ = tl.Read()
	if len(evs) != 1 || evs[0].Kind != "ckpt" || evs[0].Ckpt != "runs/a/ckpt/step-000002" {
		t.Fatalf("the completed line must parse whole, garbage skipped: got %+v", evs)
	}
	if evs, _ = tl.Read(); len(evs) != 0 {
		t.Fatalf("nothing new: got %+v", evs)
	}
}

func TestHeartbeatCarriesNewCheckpoints(t *testing.T) {
	// WHY: a heartbeat's details are the resume point the server hands the
	//      next attempt. Each new checkpoint must reach a heartbeat while the
	//      child is still running (a worker SIGKILLed one second later must
	//      not lose it), including a ckpt line the child finished writing in
	//      two pieces.
	// KIND: unit
	// CATCHES: s01, s08
	// CHAPTER: dur.09 section 2.3
	r, _ := runner(t)
	b := &beats{}
	spec := script("exit130",
		emit(map[string]any{"kind": "ckpt", "step": 100, "ckpt": "runs/r/ckpt/step-000100"}),
		act("sleep", 0.3),
		act("partial", `{"ts":1,"kind":"ckpt","step":200,`),
		act("sleep", 0.2),
		act("partial", `"ckpt":"runs/r/ckpt/step-000200"}`),
		act("finish_line", true),
		act("sleep", 0.3),
		act("done", []string{}),
	)
	if _, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "hb/1", Attempt: 1, HeartbeatTimeout: 30 * time.Second}, nil, spec, b.hb); err != nil {
		t.Fatal(err)
	}
	all := strings.Join(b.all(), " ")
	if !strings.Contains(all, "runs/r/ckpt/step-000100") || !strings.Contains(all, "runs/r/ckpt/step-000200") {
		t.Fatalf("a heartbeat must carry each new checkpoint promptly (heartbeat timeout 30 s, run under 2 s): got %q", b.all())
	}
	for _, d := range b.all() {
		if strings.Contains(d, "step\":200") || strings.HasPrefix(d, "{") {
			t.Fatalf("a half-written line reached a heartbeat: %q", d)
		}
	}
}

func TestResumePointSurvivesUntilANewCheckpoint(t *testing.T) {
	// WHY: attempt 2 starts with the server's last heartbeat details. If its
	//      first heartbeats sent empty details (no checkpoint of its own
	//      yet), they would overwrite that resume point, and a kill before
	//      attempt 2's first checkpoint would restart training from step 0.
	// KIND: fault
	// CATCHES: s09
	// CHAPTER: dur.09 section 5, Pitfalls
	r, _ := runner(t)
	b := &beats{}
	resume := "runs/x/ckpt/step-000500"
	spec := script("exit130", act("sleep", 0.4), act("done", []string{}))
	_, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "keep/1", Attempt: 2, LastHeartbeat: []byte(resume), HeartbeatTimeout: 90 * time.Millisecond}, nil, spec, b.hb)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.all()) < 2 {
		t.Fatalf("with a 90 ms heartbeat timeout a 400 ms run heartbeats several times, got %d", len(b.all()))
	}
	for i, d := range b.all() {
		if d != resume {
			t.Fatalf("heartbeat %d sent %q: before a new checkpoint every heartbeat must keep %q", i+1, d, resume)
		}
	}
}

func TestHeartbeatEveryThirdOfTimeout(t *testing.T) {
	// WHY: the server fences an activity whose heartbeats stop for
	//      heartbeat_timeout. Heartbeating only on checkpoints (every few
	//      minutes in training) loses the lease mid-run; the contract says at
	//      least every timeout/3.
	// KIND: unit
	// CATCHES: s10
	// CHAPTER: dur.09 section 2.3
	r, _ := runner(t)
	b := &beats{}
	start := time.Now()
	_, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "rate/1", Attempt: 1, HeartbeatTimeout: 300 * time.Millisecond}, nil,
		script("exit130", act("sleep", 0.9), act("done", []string{})), b.hb)
	if err != nil {
		t.Fatal(err)
	}
	el := time.Since(start)
	// 0.9 s at one heartbeat per 100 ms: at least 5, allowing for start-up.
	if n := len(b.all()); n < 5 {
		t.Fatalf("%d heartbeats in %v with a 300 ms timeout; want one at least every 100 ms", n, el)
	}
}

func TestCancelRequestedStopsTheChild(t *testing.T) {
	// WHY: a workflow cancel reaches the activity as cancel_requested in a
	//      heartbeat answer. The runner must SIGTERM the child's group (the
	//      child checkpoints and exits 130), and the attempt ends as a
	//      non-retryable Canceled, with the checkpoint it wrote on the way out
	//      heartbeated.
	// KIND: fault
	// CATCHES: s11
	// CHAPTER: dur.09 section 2.5
	r, art := runner(t)
	b := &beats{cancelIf: started(art, "cancel/1")}
	start := time.Now()
	_, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "cancel/1", Attempt: 1, HeartbeatTimeout: 150 * time.Millisecond}, nil,
		script("ckpt_then_130", emit(map[string]any{"kind": "step", "step": 1}), act("wait_term", 8.0), act("exit", 0)), b.hb)
	f := failure(t, err)
	if f.Type != "Canceled" || !f.NonRetryable() {
		t.Fatalf("want a non-retryable Canceled, got %+v", f)
	}
	if el := time.Since(start); el > 3*time.Second {
		t.Fatalf("the child waits for SIGTERM; the run took %v, so SIGTERM never came", el)
	}
	if b.last() != "runs/fake/ckpt/on-term" {
		t.Fatalf("the checkpoint written on SIGTERM must be heartbeated before the attempt ends, got %q", b.all())
	}
}

func TestGraceThenSIGKILL(t *testing.T) {
	// WHY: a child that ignores SIGTERM (stuck in a C kernel, or a bug) must
	//      not hold a worker slot forever: after the grace period the whole
	//      group is SIGKILLed, including any grandchild it started.
	// KIND: fault
	// CATCHES: s12, s13
	// CHAPTER: dur.09 section 2.5
	r, art := runner(t)
	r.Grace = 300 * time.Millisecond
	b := &beats{cancelIf: started(art, "stubborn/1")}
	start := time.Now()
	_, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "stubborn/1", Attempt: 1, HeartbeatTimeout: 150 * time.Millisecond}, nil,
		script("ignore", act("spawn_sleeper", "grandchild"), emit(map[string]any{"kind": "step", "step": 1}), act("sleep", 6)), b.hb)
	f := failure(t, err)
	if el := time.Since(start); el > 4*time.Second {
		t.Fatalf("grace 300 ms: the run took %v; the child was never SIGKILLed", el)
	}
	if f.Type != "Canceled" || f.Signal != int(syscall.SIGKILL) {
		t.Fatalf("want Canceled after SIGKILL (Signal 9), got %+v", f)
	}
	pidb, err := os.ReadFile(filepath.Join(art, "activities", "stubborn", "1", "grandchild.pid"))
	if err != nil {
		t.Fatal(err)
	}
	pid, _ := strconv.Atoi(string(pidb))
	deadline := time.Now().Add(patience)
	for syscall.Kill(pid, 0) == nil {
		// A zombie still answers kill(pid, 0) until reaped; ask ps for its state.
		out, _ := exec.Command("ps", "-o", "stat=", "-p", strconv.Itoa(pid)).Output()
		if s := strings.TrimSpace(string(out)); s == "" || strings.HasPrefix(s, "Z") {
			break
		}
		if time.Now().After(deadline) {
			syscall.Kill(pid, syscall.SIGKILL)
			t.Fatalf("grandchild %d survived the runner: signal the group (-pgid), not the pid", pid)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

func TestOwnProcessGroup(t *testing.T) {
	// WHY: the child must lead its own process group, so a SIGTERM to the
	//      group reaches its grandchildren, and a ^C or a group signal meant
	//      for the worker does not reach the child behind the runner's back.
	// KIND: unit
	// CATCHES: s12
	// CHAPTER: dur.09 section 2.5
	r, art := runner(t)
	if _, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "pg/1", Attempt: 1}, nil, script("exit130", act("record", "pg"), act("done", []string{})), (&beats{}).hb); err != nil {
		t.Fatal(err)
	}
	rec := readRecord(t, filepath.Join(art, "activities", "pg", "1"), "pg")
	if rec.Pgid != rec.Pid || rec.Pgid == syscall.Getpgrp() {
		t.Fatalf("child pid %d pgid %d, worker pgid %d: the child must lead a new group", rec.Pid, rec.Pgid, syscall.Getpgrp())
	}
}

func TestWorkerShutdownIsRetryable(t *testing.T) {
	// WHY: a worker that is draining (SIGTERM from a deploy) stops its
	//      children the same way, but the activity is not cancelled: another
	//      worker must resume it from the checkpoint. Reporting the child's
	//      exit 130 as a non-retryable Canceled would end the training run
	//      for good on every rollout.
	// KIND: fault
	// CATCHES: s14
	// CHAPTER: dur.09 section 5, Pitfalls
	r, art := runner(t)
	ctx, cancel := context.WithCancel(runCtx(t))
	b := &beats{}
	ready := started(art, "drain/1")
	go func() {
		deadline := time.Now().Add(patience)
		for !ready() && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}
		cancel()
	}()
	_, err := r.Run(ctx, activities.Task{IdempotencyKey: "drain/1", Attempt: 1, HeartbeatTimeout: 150 * time.Millisecond}, nil,
		script("ckpt_then_130", emit(map[string]any{"kind": "step", "step": 1}), act("wait_term", 8.0), act("exit", 0)), b.hb)
	f := failure(t, err)
	if f.NonRetryable() || f.Type != "WorkerShutdown" {
		t.Fatalf("a drain must leave the activity retryable (WorkerShutdown), got %+v", f)
	}
	if string(f.Details) != "runs/fake/ckpt/on-term" || b.last() != "runs/fake/ckpt/on-term" {
		t.Fatalf("the checkpoint written on the way out is the next attempt's resume point: Failure.Details %q, last heartbeat %q", f.Details, b.last())
	}
}

func TestCancelCauseFromContext(t *testing.T) {
	// WHY: the worker SDK delivers a cancel_requested answer by cancelling the
	//      activity's context with its own cause (ErrCancelRequested), not
	//      through the Heartbeat return value. With Subprocess.Canceled set,
	//      that cause must end the attempt as a non-retryable Canceled, while
	//      any other cause (heartbeat timeout, lease lost, drain) stays a
	//      retryable WorkerShutdown.
	// KIND: fault
	// CATCHES: s40
	// CHAPTER: dur.09 section 2.5
	errCancel := errors.New("cancel requested")
	for _, c := range []struct {
		cause error
		typ   string
	}{{errCancel, "Canceled"}, {errors.New("lease lost"), "WorkerShutdown"}} {
		r, art := runner(t)
		r.Canceled = func(cause error) bool { return errors.Is(cause, errCancel) }
		key := "cause/" + strings.ReplaceAll(c.typ, " ", "")
		ctx, cancel := context.WithCancelCause(runCtx(t))
		ready := started(art, key)
		go func() {
			deadline := time.Now().Add(patience)
			for !ready() && time.Now().Before(deadline) {
				time.Sleep(10 * time.Millisecond)
			}
			cancel(c.cause)
		}()
		_, err := r.Run(ctx, activities.Task{IdempotencyKey: key, Attempt: 1}, nil,
			script("exit130", emit(map[string]any{"kind": "step", "step": 1}), act("wait_term", 8.0), act("exit", 0)), (&beats{}).hb)
		f := failure(t, err)
		if f.Type != c.typ || f.NonRetryable() != (c.typ == "Canceled") {
			t.Fatalf("ctx ended with cause %q: got %+v, want Type %s", c.cause, f, c.typ)
		}
	}
}

func TestSpecWrittenAtomically(t *testing.T) {
	// WHY: the child reads spec.json at start; the runner writes it before
	//      exec through a temp file and a rename, so a retry that rewrites it
	//      never lets a reader see half a spec. Here an earlier spec.json is
	//      a hard link to another file: a rename replaces the directory entry
	//      and leaves the other file alone, while writing in place truncates
	//      the shared inode and changes it.
	// KIND: unit
	// CATCHES: s15
	// CHAPTER: dur.09 section 2.2
	r, art := runner(t)
	dir := filepath.Join(art, "activities", "spec", "1")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	other := filepath.Join(art, "reader-holds-this.json")
	if err := os.WriteFile(other, []byte(`{"old": true}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(other, filepath.Join(dir, "spec.json")); err != nil {
		t.Fatal(err)
	}
	spec := script("exit130", act("done", []string{}))
	if _, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "spec/1", Attempt: 1}, nil, spec, (&beats{}).hb); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(filepath.Join(dir, "spec.json"))
	if string(got) != string(spec) {
		t.Fatalf("spec.json differs from the spec given")
	}
	if old, _ := os.ReadFile(other); string(old) != `{"old": true}` {
		t.Fatalf("the old spec.json's inode was rewritten in place (%q): write spec.json.tmp, then rename", old)
	}
	if _, err := os.Stat(filepath.Join(dir, "spec.json.tmp")); err == nil {
		t.Fatal("spec.json.tmp left behind")
	}
}

func TestStderrTailIsBounded(t *testing.T) {
	// WHY: Failure.message travels in a gRPC call and the workflow history;
	//      a child that prints a million lines must not make it unbounded.
	//      The last 4 KiB (where the traceback is) are kept.
	// KIND: boundary
	// CATCHES: s16
	// CHAPTER: dur.09 section 2.4
	r, _ := runner(t)
	big := strings.Repeat("x", 9000) + "\nTraceback: the real error\n"
	_, err := r.Run(runCtx(t), activities.Task{IdempotencyKey: "stderr/1", Attempt: 1}, nil, script("exit130", act("stderr", big), act("exit", 1)), (&beats{}).hb)
	f := failure(t, err)
	if f.FailureType() != "ExitCode1" {
		t.Fatalf("FailureType() is what the worker SDK matches against non_retryable: got %q", f.FailureType())
	}
	if len(f.Message) > 4096 || !strings.HasSuffix(f.Message, "Traceback: the real error\n") {
		t.Fatalf("want the last <= 4096 bytes of stderr ending in the traceback, got %d bytes ending %q", len(f.Message), f.Message[max(0, len(f.Message)-40):])
	}
}

func TestShutdownFlushesUnacknowledgedCheckpoint(t *testing.T) {
	// WHY: a checkpoint consumed before shutdown must still reach the
	// server if its heartbeat failed; the final tail read can be empty.
	// KIND: fault
	// CATCHES: s41
	// CHAPTER: dur.09 section 5, Pitfalls
	r, _ := runner(t)
	ctx, cancel := context.WithCancel(runCtx(t))
	defer cancel()
	const checkpoint = "runs/fake/ckpt/unacknowledged"
	b := &beats{}
	failed := false
	hb := func(hctx context.Context, details []byte) (bool, error) {
		if err := hctx.Err(); err != nil {
			return false, err
		}
		if string(details) == checkpoint && !failed {
			failed = true
			cancel() // deterministic shutdown after the checkpoint was consumed
			return false, errors.New("temporary heartbeat transport failure")
		}
		return b.hb(hctx, details)
	}
	_, err := r.Run(ctx, activities.Task{IdempotencyKey: "flush/1", Attempt: 1, HeartbeatTimeout: 150 * time.Millisecond}, nil,
		script("exit130", emit(map[string]any{"kind": "ckpt", "step": 1, "ckpt": checkpoint}), act("wait_term", 8.0)), hb)
	f := failure(t, err)
	if !failed || f.Type != "WorkerShutdown" || f.NonRetryable() {
		t.Fatalf("checkpoint heartbeat must fail once, then drain retryably: failed=%v, failure=%+v", failed, f)
	}
	if string(f.Details) != checkpoint || b.last() != checkpoint {
		t.Fatalf("checkpoint was consumed but not acknowledged: details=%q heartbeat=%q", f.Details, b.last())
	}
}
