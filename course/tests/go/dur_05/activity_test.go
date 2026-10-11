// Course tests for dur.05, activities: the backoff formula and jitter
// (go/durable/activity), and retries, heartbeats, non-retryable failures, the
// dead-letter queue, and redrive on the server (go/durable/server/activities.go).
package dur_05

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"

	"tinyllm/durable/activity"
	"tinyllm/durable/worker"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

func guard(t *testing.T, d time.Duration) context.Context {
	c, cancel := context.WithTimeout(bg, d)
	t.Cleanup(cancel)
	return c
}

func (e *env) pollWT() *durablev1.WorkflowTask {
	e.t.Helper()
	wt, err := e.tasks.PollWorkflowTask(guard(e.t, 5*time.Second), &durablev1.PollRequest{TaskQueue: "default", Identity: "wk"})
	if err != nil || len(wt.TaskToken) == 0 {
		e.t.Fatalf("PollWorkflowTask: %v", err)
	}
	return wt
}

func (e *env) pollAT() *durablev1.ActivityTask {
	e.t.Helper()
	at, err := e.tasks.PollActivityTask(guard(e.t, 5*time.Second), &durablev1.PollRequest{TaskQueue: "default", Identity: "wk"})
	if err != nil || len(at.TaskToken) == 0 {
		e.t.Fatalf("PollActivityTask: %v", err)
	}
	return at
}

// noActivityTask: nothing is pollable now (the fake clock does not move, so
// a waiting poll is cut by a 50 ms real-time deadline). The client gives up
// at the deadline, but the server's handler sees its cancel a moment later;
// a clock Advance in that window would wake the abandoned poll, which would
// lease the task for a caller that is gone. So wait for the handler to end.
func (e *env) noActivityTask() {
	e.t.Helper()
	at, err := e.tasks.PollActivityTask(guard(e.t, 50*time.Millisecond), &durablev1.PollRequest{TaskQueue: "default", Identity: "wk"})
	if err == nil && len(at.TaskToken) > 0 {
		e.t.Fatalf("an activity task was pollable: attempt %d", at.Attempt)
	}
	eventually(e.t, "the abandoned poll to end", func() bool { return e.polls.Load() == 0 })
}

func (e *env) failAT(at *durablev1.ActivityTask, f *durablev1.Failure) error {
	_, err := e.tasks.FailActivityTask(bg, &durablev1.FailActivityRequest{TaskToken: at.TaskToken, LeaseToken: at.LeaseToken, Failure: f, Identity: "wk"})
	return err
}

func (e *env) heartbeat(at *durablev1.ActivityTask, details string) (*durablev1.HeartbeatResponse, error) {
	return e.tasks.RecordHeartbeat(bg, &durablev1.HeartbeatRequest{TaskToken: at.TaskToken, LeaseToken: at.LeaseToken, Details: []byte(details)})
}

// startWithActivity starts a run and completes its first workflow task with
// one ScheduleActivity of the given options.
func (e *env) startWithActivity(id string, o *durablev1.ActivityOptions) {
	e.t.Helper()
	e.startWF(id, "Batch", "")
	cmd := &durablev1.Command{Cmd: &durablev1.Command_ScheduleActivity{ScheduleActivity: &durablev1.ScheduleActivity{
		ActivityId: "1", ActivityType: "Flaky", Options: o,
	}}}
	if _, err := e.tasks.CompleteWorkflowTask(bg, &durablev1.CompleteWorkflowTaskRequest{TaskToken: e.pollWT().TaskToken, Commands: []*durablev1.Command{cmd}}); err != nil {
		e.t.Fatal(err)
	}
}

func TestBackoffHandExample(t *testing.T) {
	// WHY: the chapter's worked schedule: initial 1 s, backoff 2, cap 10 s
	//      gives 1, 2, 4, 8, 10, 10 s before attempts 2 to 7; an empty policy
	//      uses the defaults (1 s, x2, 60 s); full jitter at u = 0.5 halves it.
	// KIND: unit
	// CATCHES: s01, s02, s03
	// CHAPTER: dur.05 section 3, worked example
	p := &durablev1.RetryPolicy{InitialMs: 1000, Backoff: 2, MaxIntervalMs: 10000}
	var got []string
	for n := 1; n <= 6; n++ {
		got = append(got, activity.Backoff(p, n).String())
	}
	if strings.Join(got, " ") != "1s 2s 4s 8s 10s 10s" {
		t.Fatalf("schedule %v; want 1s 2s 4s 8s 10s 10s", got)
	}
	var def []string
	for _, n := range []int{1, 2, 6, 7, 30} {
		def = append(def, activity.Backoff(&durablev1.RetryPolicy{}, n).String())
	}
	if strings.Join(def, " ") != "1s 2s 32s 1m0s 1m0s" {
		t.Fatalf("defaults %v; want 1s 2s 32s 1m0s 1m0s", def)
	}
	if activity.Backoff(&durablev1.RetryPolicy{InitialMs: 500, Backoff: 1.5, MaxIntervalMs: 60000}, 3) != 1125*time.Millisecond {
		t.Fatal("Backoff(0.5 s, x1.5, n=3) != 1.125 s")
	}
	if j := activity.Jitter(8*time.Second, 0.5); j != 4*time.Second {
		t.Fatalf("Jitter(8s, 0.5) = %v; want 4s", j)
	}
}

func TestJitterBoundsAndMean(t *testing.T) {
	// WHY: full jitter draws the delay uniformly from [0, d): never longer
	//      than the backoff, and d/2 on average (the expected jittered backoff
	//      of S-M07a), so a burst of failures spreads out instead of retrying
	//      in lockstep.
	// KIND: statistical
	// CATCHES: s04
	// CHAPTER: dur.05 section 2, jitter
	rng := newPCG32(7, 7)
	d := 10 * time.Second
	var sum float64
	const n = 20000
	for i := 0; i < n; i++ {
		u := rng.unit()
		j := activity.Jitter(d, u)
		if j < 0 || j >= d {
			t.Fatalf("Jitter(%v, %.4f) = %v, outside [0, d)", d, u, j)
		}
		sum += float64(j)
	}
	// The mean of n uniform draws on [0, d) has sd d / sqrt(12 n); allow 4 sd.
	mean, sd := sum/n, float64(d)/(3.4641016*141.42136)
	if diff := mean - float64(d)/2; diff > 4*sd || diff < -4*sd {
		t.Fatalf("mean jittered delay %v; want %v within %v", time.Duration(mean), d/2, time.Duration(4*sd))
	}
	if activity.Jitter(d, 0) != 0 || activity.Jitter(d, 1) >= d {
		t.Fatal("u = 0 must give 0 and u = 1 must stay below d")
	}
}

func TestErrorTypes(t *testing.T) {
	// WHY: activities say "do not retry" with NewNonRetryable and name their
	//      failure type for RetryPolicy.non_retryable; both survive wrapping,
	//      because the worker inspects them with errors.As.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: dur.05 section 4
	nr := fmt.Errorf("parse spec: %w", activity.NewNonRetryable("BadSpec", "missing field"))
	if !activity.IsNonRetryable(nr) || activity.IsNonRetryable(activity.NewError("Timeout", "slow")) || activity.IsNonRetryable(errors.New("x")) {
		t.Fatal("IsNonRetryable")
	}
	var typed interface{ FailureType() string }
	if !errors.As(nr, &typed) || typed.FailureType() != "BadSpec" {
		t.Fatalf("FailureType through a wrap: %v", typed)
	}
}

func TestRetryScheduleUnderFakeClock(t *testing.T) {
	// WHY: the server applies the formula: with jitter off, attempt 2 becomes
	//      pollable exactly 1 s after attempt 1 fails, attempt 3 after 2 s,
	//      attempt 4 after 4 s; attempt 4 failing (max_attempts 4) parks the
	//      task in the DLQ, appends ActivityDeadLettered, and the workflow
	//      keeps waiting. A redrive delivers it again as attempt 1.
	// KIND: unit, fault
	// CATCHES: s06, s07, s08, s14, s15
	// CHAPTER: dur.05 section 3, worked example
	e := newEnv(t)
	e.jitter = func(d time.Duration) time.Duration { return d }
	e.start()
	e.startWithActivity("retry", &durablev1.ActivityOptions{StartToCloseMs: 30000,
		Retry: &durablev1.RetryPolicy{InitialMs: 1000, Backoff: 2, MaxIntervalMs: 10000, MaxAttempts: 4}})
	for n, wait := range []time.Duration{time.Second, 2 * time.Second, 4 * time.Second} {
		at := e.pollAT()
		if at.Attempt != int32(n+1) {
			t.Fatalf("attempt %d delivered as %d", n+1, at.Attempt)
		}
		if err := e.failAT(at, &durablev1.Failure{Message: fmt.Sprintf("flaky %d", n+1), Type: "Flaky"}); err != nil {
			t.Fatal(err)
		}
		e.clk.Advance(wait - time.Millisecond)
		e.noActivityTask()
		e.clk.Advance(time.Millisecond)
	}
	last := e.pollAT()
	if last.Attempt != 4 {
		t.Fatalf("fourth delivery is attempt %d", last.Attempt)
	}
	e.failAT(last, &durablev1.Failure{Message: "flaky 4", Type: "Flaky"})
	h := e.mustHistory("retry")
	dl := h[len(h)-1].GetDeadLettered()
	if dl == nil || dl.Attempts != 4 || dl.LastFailure.GetMessage() != "flaky 4" || dl.ScheduledEventId != 5 {
		t.Fatalf("history ends %q; want act_dead_lettered after 4 attempts", kinds(h[len(h)-3:]))
	}
	if strings.Contains(kinds(h), "act_failed") || strings.Count(kinds(h), "wt_scheduled") != 1 {
		t.Fatalf("a dead letter must not wake the workflow: %q", kinds(h))
	}
	pend := e.describe("retry").PendingActivities
	if len(pend) != 1 || !pend[0].DeadLettered || pend[0].Attempt != 4 {
		t.Fatalf("pending activities %v", pend)
	}
	list, err := e.wf.ListDeadLetters(bg, &durablev1.ListDeadLettersRequest{TaskQueue: "default"})
	if err != nil || len(list.DeadLetters) != 1 {
		t.Fatalf("ListDeadLetters: %v %v", list, err)
	}
	if d := list.DeadLetters[0]; d.WorkflowId != "retry" || d.ActivityId != "1" || d.ActivityType != "Flaky" || d.Attempts != 4 || d.LastFailure.GetMessage() != "flaky 4" {
		t.Fatalf("dead letter %v", d)
	}
	if _, err := e.wf.RedriveDeadLetter(bg, &durablev1.RedriveDeadLetterRequest{TaskQueue: "default", TaskIds: []string{"nope"}}); code(err) != codes.NotFound {
		t.Fatalf("redrive of an unknown task: %v; want NOT_FOUND", err)
	}
	if r, err := e.wf.RedriveDeadLetter(bg, &durablev1.RedriveDeadLetterRequest{TaskQueue: "default"}); err != nil || r.Redriven != 1 {
		t.Fatalf("redrive: %v %v", r, err)
	}
	again := e.pollAT()
	if again.Attempt != 1 {
		t.Fatalf("after redrive: attempt %d; want 1", again.Attempt)
	}
	e.tasks.CompleteActivityTask(bg, &durablev1.CompleteActivityRequest{TaskToken: again.TaskToken, LeaseToken: again.LeaseToken, Result: []byte("ok")})
	if h := e.mustHistory("retry"); kinds(h[len(h)-3:]) != "act_started act_completed wt_scheduled" {
		t.Fatalf("after the redriven attempt: %q", kinds(h))
	}
}

func TestNonRetryableStops(t *testing.T) {
	// WHY: retrying a failure that cannot succeed wastes attempts and hides
	//      it. A failure marked non_retryable, or whose type the policy lists
	//      (the subprocess runner's "ExitCode65"), ends the activity at once
	//      with ActivityTaskFailed and wakes the workflow.
	// KIND: unit
	// CATCHES: s09, s10, s17
	// CHAPTER: dur.05 section 2, non-retryable failures
	e := newEnv(t).start()
	for i, f := range []*durablev1.Failure{
		{Message: "missing field", Type: "BadSpec", NonRetryable: true},
		{Message: "exit 65", Type: "ExitCode65"},
	} {
		id := fmt.Sprintf("nr-%d", i)
		e.startWithActivity(id, &durablev1.ActivityOptions{StartToCloseMs: 30000,
			Retry: &durablev1.RetryPolicy{MaxAttempts: 10, NonRetryable: []string{"ExitCode65"}}})
		if err := e.failAT(e.pollAT(), f); err != nil {
			t.Fatal(err)
		}
		h := e.mustHistory(id)
		if got := kinds(h[5:]); got != "act_started act_failed wt_scheduled" {
			t.Fatalf("%s: %q; want act_started act_failed wt_scheduled", f.Type, got)
		}
		if af := h[6].GetActFailed(); af.Failure.GetType() != f.Type || h[5].GetActStarted().Attempt != 1 {
			t.Fatalf("%s: recorded %v", f.Type, af)
		}
		// the woken workflow gives up, so the next case starts from a quiet queue
		e.tasks.CompleteWorkflowTask(bg, &durablev1.CompleteWorkflowTaskRequest{TaskToken: e.pollWT().TaskToken, Commands: []*durablev1.Command{
			{Cmd: &durablev1.Command_Fail{Fail: &durablev1.FailWorkflow{Failure: h[6].GetActFailed().GetFailure()}}}}})
	}
	if v, l, d := e.q.Depth("act:default"); v+l+d != 0 {
		t.Fatalf("failed activities still queued: %d %d %d", v, l, d)
	}
}

func TestWorkerMapsActivityErrors(t *testing.T) {
	// WHY: end to end through the worker: an activity returning
	//      activity.NewNonRetryable reaches the history as a non-retryable
	//      failure with its type, after exactly one attempt.
	// KIND: unit
	// CATCHES: s05
	// CHAPTER: dur.05 section 4
	e := newEnv(t).start()
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default"})
	attempts := 0
	w.RegisterActivity("Flaky", func(ctx context.Context, in []byte) ([]byte, error) {
		attempts++
		return nil, activity.NewNonRetryable("BadSpec", "missing field")
	})
	runWorker(t, w)
	e.startWithActivity("map", &durablev1.ActivityOptions{StartToCloseMs: 30000, Retry: &durablev1.RetryPolicy{MaxAttempts: 5}})
	eventually(t, "the failure to be recorded", func() bool { return strings.Contains(kinds(e.mustHistory("map")), "act_failed") })
	h := e.mustHistory("map")
	if f := h[6].GetActFailed().GetFailure(); f.Type != "BadSpec" || !f.NonRetryable || attempts != 1 {
		t.Fatalf("failure %v after %d attempts", f, attempts)
	}
}

func TestHeartbeatExtendsLeaseAndResumes(t *testing.T) {
	// WHY: a heartbeat moves the attempt's deadline to now + heartbeat
	//      timeout and stores its details; when the worker dies anyway, the
	//      next attempt gets those details (resume from the checkpoint) and
	//      the same idempotency key. Describe shows the last heartbeat.
	// KIND: unit, fault
	// CATCHES: s11, s12, s13, s16
	// CHAPTER: dur.05 section 2, heartbeats
	e := newEnv(t).start()
	e.startWithActivity("hb", &durablev1.ActivityOptions{StartToCloseMs: 60000, HeartbeatTimeoutMs: 2000})
	first := e.pollAT()
	if first.DeadlineUnixMs != t0.Add(2*time.Second).UnixMilli() || first.HeartbeatTimeoutMs != 2000 {
		t.Fatalf("first lease deadline +%dms", first.DeadlineUnixMs-t0.UnixMilli())
	}
	e.clk.Advance(1500 * time.Millisecond)
	resp, err := e.heartbeat(first, "ckpt-3")
	if err != nil || resp.DeadlineUnixMs != t0.Add(3500*time.Millisecond).UnixMilli() || resp.CancelRequested {
		t.Fatalf("heartbeat at +1.5s: %v %v; want deadline +3.5s", resp, err)
	}
	if p := e.describe("hb").PendingActivities[0]; string(p.LastHeartbeatDetails) != "ckpt-3" || p.LastHeartbeatUnixMs != t0.Add(1500*time.Millisecond).UnixMilli() {
		t.Fatalf("describe pending: %v", p)
	}
	e.clk.Advance(1999 * time.Millisecond) // +3.499 s: still leased
	e.noActivityTask()
	e.clk.Advance(time.Millisecond) // +3.5 s: the worker is presumed dead
	w := worker.New(dial(t, e.addr), worker.Options{TaskQueue: "default"})
	w.RegisterActivity("Flaky", func(ctx context.Context, in []byte) ([]byte, error) {
		return []byte(fmt.Sprintf("%s|%d|%s", activity.HeartbeatDetails(ctx), activity.Attempt(ctx), activity.IdempotencyKey(ctx))), nil
	})
	runWorker(t, w)
	eventually(t, "the second attempt to complete", func() bool { return strings.Contains(kinds(e.mustHistory("hb")), "act_completed") })
	h := e.mustHistory("hb")
	if got := string(h[6].GetActCompleted().GetResult()); got != "ckpt-3|2|hb/1" {
		t.Fatalf("second attempt saw %q; want ckpt-3|2|hb/1", got)
	}
}

func TestStaleLeaseHeartbeatAndFailRefused(t *testing.T) {
	// WHY: a worker that lost its lease must hear so on its next heartbeat
	//      (FAILED_PRECONDITION, so the SDK cancels the activity) and must not
	//      be able to fail or retry a task someone else now owns.
	// KIND: fault
	// CATCHES: s18
	// CHAPTER: dur.05 section 5, Pitfalls
	e := newEnv(t).start()
	e.startWithActivity("st", &durablev1.ActivityOptions{StartToCloseMs: 5000})
	old := e.pollAT()
	e.clk.Advance(5 * time.Second)
	cur := e.pollAT()
	if _, err := e.heartbeat(old, "x"); code(err) != codes.FailedPrecondition {
		t.Fatalf("heartbeat with a stale lease: %v", err)
	}
	if err := e.failAT(old, &durablev1.Failure{Message: "late", NonRetryable: true}); code(err) != codes.FailedPrecondition {
		t.Fatalf("fail with a stale lease: %v", err)
	}
	if err := e.failAT(old, &durablev1.Failure{Message: "late"}); code(err) != codes.FailedPrecondition {
		t.Fatalf("retryable fail with a stale lease: %v", err)
	}
	if _, err := e.heartbeat(cur, "y"); err != nil {
		t.Fatalf("the live attempt's heartbeat: %v", err)
	}
	if h := e.mustHistory("st"); len(h) != 5 {
		t.Fatalf("stale calls appended events: %q", kinds(h))
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

// unit is uniform in [0, 1) on a 2^-32 grid.
func (p *pcg32) unit() float64 { return float64(p.next()) / (1 << 32) }
