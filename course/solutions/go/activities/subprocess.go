// Package activities holds the worker's activity implementations. dur.09
// owns subprocess.go: the Go half of a subprocess activity
// (contracts/spec/subprocess-activity.md). Python never speaks gRPC to the
// platform (D9); a Go worker runs a Python entry ({tinyllm} train, eval,
// export; {corpus} run --stage) as a child process, tails its progress file,
// turns checkpoints into heartbeats, and turns its exit code into the
// activity's result or failure.
//
// The runner does not import the worker SDK. The worker's composition root
// adapts a delivery (dur.04, dur.05) to Task and Heartbeat:
//
//	r := activities.Subprocess{Entry: tinyllmArgv, Artifacts: "/artifacts"}
//	w.RegisterActivity("train", func(ctx context.Context, spec []byte) ([]byte, error) {
//		info := activity.GetInfo(ctx)
//		return r.Run(ctx, activities.Task{IdempotencyKey: activity.IdempotencyKey(ctx), ...},
//			[]string{"train"}, spec, heartbeatAdapter)
//	})
package activities

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Exit codes of the subprocess contract.
const (
	ExitOK       = 0
	ExitDataErr  = 65  // EX_DATAERR: the input is wrong, never retried
	ExitTempFail = 75  // EX_TEMPFAIL: retryable
	ExitCanceled = 130 // cancelled: SIGTERM honored
)

const (
	// DefaultGrace is how long a child has between SIGTERM and SIGKILL.
	DefaultGrace = 30 * time.Second
	// DefaultHeartbeat is the heartbeat interval when the activity has no
	// heartbeat timeout: the heartbeat answer is also how a cancel arrives.
	DefaultHeartbeat = 10 * time.Second
	// DefaultPoll is how often the progress file is read.
	DefaultPoll = 200 * time.Millisecond
	// MaxResult is the largest DONE.json that can be an activity result.
	MaxResult = 2 << 20
	// stderrTail is how much of the child's stderr a Failure keeps.
	stderrTail = 4 << 10
)

// Task is what the runner needs to know about one delivery of an activity
// (from tl.durable.v1.ActivityTask).
type Task struct {
	IdempotencyKey   string            // "<workflow_id>/<activity_id>", stable across attempts
	Attempt          int               // 1-based
	LastHeartbeat    []byte            // last_heartbeat_details: the last checkpoint path, or empty
	TraceContext     map[string]string // the activity span's W3C context: "traceparent" (and "tracestate")
	HeartbeatTimeout time.Duration     // 0: none required (the runner still heartbeats every DefaultHeartbeat)
}

// Heartbeat records a heartbeat with details (the last checkpoint path).
// cancelRequested is the server asking for the activity to be cancelled
// (a workflow cancel, or a lease this worker no longer holds). An error is
// treated as transient: the runner keeps the child running and heartbeats
// again on the next tick.
type Heartbeat func(ctx context.Context, details []byte) (cancelRequested bool, err error)

// Subprocess runs one Python entry under the subprocess contract.
type Subprocess struct {
	Entry        []string      // the entry argv, from the runtime config (for example ["python", "python/tinyllm/__main__.py"])
	Artifacts    string        // TL_ARTIFACTS: the shared artifact root
	OTLPEndpoint string        // OTEL_EXPORTER_OTLP_ENDPOINT for the child; "" leaves it unset
	ServiceName  string        // OTEL_SERVICE_NAME for the child; "" leaves it unset
	Grace        time.Duration // SIGTERM to SIGKILL; 0 = DefaultGrace
	Poll         time.Duration // progress file poll interval; 0 = DefaultPoll
	Env          []string      // the child's base environment; nil = os.Environ()
	Dir          string        // the child's working directory; "" = the worker's
	// Canceled reports whether the cause that ended ctx is a cancellation of
	// the activity (the worker SDK cancels with its ErrCancelRequested) rather
	// than the worker going away. nil: every end of ctx is a shutdown.
	Canceled func(cause error) bool
}

// Failure is the error Run returns for every attempt that did not succeed.
// It maps onto tl.durable.v1.Failure (type, message, non_retryable): its
// FailureType and NonRetryable methods are what the worker SDK reads.
type Failure struct {
	Type     string // "ExitCode<n>", "Signal<n>", "Canceled", "WorkerShutdown", "MissingDone", ...
	Message  string // the last 4 KiB of the child's stderr, or what went wrong
	NoRetry  bool   // non-retryable
	ExitCode int    // the child's exit code; -1 when it was killed by a signal or never started
	Signal   int    // the signal that killed it, or 0
	Details  []byte // the last checkpoint path, for FailActivityRequest.last_heartbeat_details
}

// FailureType is Failure.type, matched against RetryPolicy.non_retryable.
func (f *Failure) FailureType() string { return f.Type }

// NonRetryable is Failure.non_retryable.
func (f *Failure) NonRetryable() bool { return f.NoRetry }

func (f *Failure) Error() string {
	retry := "retryable"
	if f.NoRetry {
		retry = "non-retryable"
	}
	msg := strings.TrimSpace(f.Message)
	if i := strings.LastIndexByte(msg, '\n'); i >= 0 {
		msg = msg[i+1:]
	}
	return fmt.Sprintf("subprocess activity: %s (%s): %s", f.Type, retry, msg)
}

var keySegment = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// WorkDir is <artifacts>/activities/<key>, with each "/"-separated segment
// of the idempotency key a directory. Every attempt of one activity shares
// it, which is what lets a retry resume instead of restart. A segment that
// is empty, ".", "..", or has other characters is an error: the key would
// escape the activities directory.
func WorkDir(artifacts, key string) (string, error) {
	if artifacts == "" {
		return "", errors.New("activities: no artifacts root")
	}
	segs := strings.Split(key, "/")
	for _, s := range segs {
		if s == "." || s == ".." || !keySegment.MatchString(s) {
			return "", fmt.Errorf("activities: idempotency key %q has a bad segment %q", key, s)
		}
	}
	return filepath.Join(append([]string{artifacts, "activities"}, segs...)...), nil
}

// Progress is one line of progress.jsonl (formats/progress.schema.json).
type Progress struct {
	TS      float64  `json:"ts"`
	Kind    string   `json:"kind"`
	Step    int64    `json:"step,omitempty"`
	Loss    float64  `json:"loss,omitempty"`
	Ckpt    string   `json:"ckpt,omitempty"`
	Outputs []string `json:"outputs,omitempty"`
}

// Tail reads complete lines from a growing file. A line still being written
// (no newline yet) is kept back until it is complete; a line that is not a
// JSON object is skipped.
type Tail struct {
	Path    string
	offset  int64
	partial []byte
}

// Read returns the events appended since the last call.
func (t *Tail) Read() ([]Progress, error) {
	f, err := os.Open(t.Path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if _, err := f.Seek(t.offset, io.SeekStart); err != nil {
		return nil, err
	}
	b, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}
	t.offset += int64(len(b))
	buf := append(t.partial, b...)
	var out []Progress
	for {
		i := bytes.IndexByte(buf, '\n')
		if i < 0 {
			break
		}
		line := bytes.TrimSpace(buf[:i])
		buf = buf[i+1:]
		var p Progress
		if len(line) > 0 && json.Unmarshal(line, &p) == nil && p.Kind != "" {
			out = append(out, p)
		}
	}
	t.partial = append([]byte(nil), buf...)
	return out, nil
}

// Classify turns a finished child into nil (exit 0) or a Failure, using the
// contract's exit-code table. stderr is the tail of the child's stderr.
func Classify(state *os.ProcessState, stderr []byte) *Failure {
	msg := string(stderr)
	if ws, ok := state.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
		sig := int(ws.Signal())
		return &Failure{Type: "Signal" + strconv.Itoa(sig), Message: msg, ExitCode: -1, Signal: sig}
	}
	code := state.ExitCode()
	switch code {
	case ExitOK:
		return nil
	case ExitDataErr:
		return &Failure{Type: "ExitCode65", Message: msg, NoRetry: true, ExitCode: code}
	case ExitCanceled:
		return &Failure{Type: "Canceled", Message: msg, NoRetry: true, ExitCode: code}
	default: // 75 and every crash: retryable
		return &Failure{Type: "ExitCode" + strconv.Itoa(code), Message: msg, ExitCode: code}
	}
}

// childEnv is the base environment without the variables the contract sets,
// plus the contract's values for this delivery. Inherited copies are removed
// first: a worker started under a TRACEPARENT (or a test harness's TL_*)
// must not leak its own values into an activity that has none.
func (s Subprocess) childEnv(t Task) []string {
	base := s.Env
	if base == nil {
		base = os.Environ()
	}
	set := map[string]string{
		"TL_IDEMPOTENCY_KEY": t.IdempotencyKey,
		"TL_ATTEMPT":         strconv.Itoa(max(t.Attempt, 1)),
		"TL_ARTIFACTS":       s.Artifacts,
	}
	if tp := t.TraceContext["traceparent"]; tp != "" {
		set["TRACEPARENT"] = tp
		if ts := t.TraceContext["tracestate"]; ts != "" {
			set["TRACESTATE"] = ts
		}
	}
	if s.OTLPEndpoint != "" {
		set["OTEL_EXPORTER_OTLP_ENDPOINT"] = s.OTLPEndpoint
	}
	if s.ServiceName != "" {
		set["OTEL_SERVICE_NAME"] = s.ServiceName
	}
	drop := map[string]bool{"TRACEPARENT": true, "TRACESTATE": true, "OTEL_EXPORTER_OTLP_ENDPOINT": true}
	for k := range set {
		drop[k] = true
	}
	env := make([]string, 0, len(base)+len(set))
	for _, kv := range base {
		k, _, _ := strings.Cut(kv, "=")
		if !drop[k] {
			env = append(env, kv)
		}
	}
	for _, k := range []string{"TL_IDEMPOTENCY_KEY", "TL_ATTEMPT", "TL_ARTIFACTS", "TRACEPARENT", "TRACESTATE", "OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_SERVICE_NAME"} {
		if v, ok := set[k]; ok {
			env = append(env, k+"="+v)
		}
	}
	return env
}

// writeAtomic writes path through path.tmp and a rename.
func writeAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ring keeps the last n bytes written to it.
type ring struct {
	mu  sync.Mutex
	n   int
	buf []byte
}

func (r *ring) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.buf = append(r.buf, p...)
	if len(r.buf) > r.n {
		r.buf = append([]byte(nil), r.buf[len(r.buf)-r.n:]...)
	}
	return len(p), nil
}

func (r *ring) Bytes() []byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]byte(nil), r.buf...)
}

// Run executes one attempt:
//
//	<Entry> <args...> --spec <dir>/spec.json --progress <dir>/progress.jsonl [--resume <ckpt>]
//
// in its own process group, with <dir> = WorkDir(Artifacts, key). It writes
// spec atomically, passes --resume when the task carries heartbeat details,
// tails the progress file, and heartbeats the last checkpoint path at least
// every HeartbeatTimeout/3 (and at once when a new checkpoint appears). It
// returns the bytes of <dir>/DONE.json on exit 0, or a *Failure.
//
// Two ways to stop the child, both SIGTERM to the group and SIGKILL after
// Grace: a heartbeat answer with cancelRequested (the activity is cancelled:
// Failure "Canceled", non-retryable), and ctx ending (the worker is draining
// or lost the lease: Failure "WorkerShutdown", retryable, so another worker
// resumes from the checkpoint).
func (s Subprocess) Run(ctx context.Context, t Task, args []string, spec []byte, hb Heartbeat) ([]byte, error) {
	if len(s.Entry) == 0 {
		return nil, &Failure{Type: "BadEntry", Message: "no entry argv", NoRetry: true, ExitCode: -1}
	}
	dir, err := WorkDir(s.Artifacts, t.IdempotencyKey)
	if err != nil {
		return nil, &Failure{Type: "BadIdempotencyKey", Message: err.Error(), NoRetry: true, ExitCode: -1}
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, &Failure{Type: "WorkDir", Message: err.Error(), ExitCode: -1}
	}
	specPath := filepath.Join(dir, "spec.json")
	progPath := filepath.Join(dir, "progress.jsonl")
	if err := writeAtomic(specPath, spec); err != nil {
		return nil, &Failure{Type: "WorkDir", Message: err.Error(), ExitCode: -1}
	}

	argv := append(append(append([]string{}, s.Entry...), args...), "--spec", specPath, "--progress", progPath)
	if len(t.LastHeartbeat) > 0 {
		argv = append(argv, "--resume", string(t.LastHeartbeat))
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = s.childEnv(t)
	cmd.Dir = s.Dir
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	logf, err := os.OpenFile(filepath.Join(dir, fmt.Sprintf("attempt-%d.log", max(t.Attempt, 1))), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, &Failure{Type: "WorkDir", Message: err.Error(), ExitCode: -1}
	}
	defer logf.Close()
	tail := &ring{n: stderrTail}
	cmd.Stdout = logf
	cmd.Stderr = io.MultiWriter(logf, tail)

	// Start tailing at the current end: earlier attempts' events are history.
	tl := &Tail{Path: progPath}
	if fi, err := os.Stat(progPath); err == nil {
		tl.offset = fi.Size()
	}
	if err := cmd.Start(); err != nil {
		return nil, &Failure{Type: "StartFailed", Message: err.Error(), ExitCode: -1}
	}
	pgid := cmd.Process.Pid
	exited := make(chan error, 1)
	go func() { exited <- cmd.Wait() }()

	poll := s.Poll
	if poll <= 0 {
		poll = DefaultPoll
	}
	every := DefaultHeartbeat
	if t.HeartbeatTimeout > 0 {
		every = t.HeartbeatTimeout / 3
	}
	grace := s.Grace
	if grace <= 0 {
		grace = DefaultGrace
	}
	ticker := time.NewTicker(poll)
	defer ticker.Stop()

	// The resume point survives until the child reports a newer one: a
	// heartbeat with empty details would erase it on the server.
	last := append([]byte(nil), t.LastHeartbeat...)
	lastBeat := time.Time{}
	var stopReason string // "" running, "Canceled", "WorkerShutdown"
	var kill <-chan time.Time
	acknowledged := string(last) // checkpoint already recorded by the durable server

	stop := func(reason string) {
		if stopReason != "" {
			return
		}
		stopReason = reason
		_ = syscall.Kill(-pgid, syscall.SIGTERM)
		kill = time.After(grace)
	}
	beat := func() {
		// A heartbeat call must not hang the runner past the activity's own
		// end: it gets the attempt's context.
		cancel, err := hb(ctx, last)
		lastBeat = time.Now()
		if err == nil && cancel {
			stop("Canceled")
		}
		if err == nil {
			acknowledged = string(last)
		}
	}
	drain := func() bool {
		evs, _ := tl.Read()
		fresh := false
		for _, e := range evs {
			if e.Kind == "ckpt" && e.Ckpt != "" && e.Ckpt != string(last) {
				last = []byte(e.Ckpt)
				fresh = true
			}
		}
		return fresh
	}

	ctxDone := ctx.Done()
	var waitErr error
loop:
	for {
		select {
		case waitErr = <-exited:
			break loop
		case <-ctxDone:
			ctxDone = nil
			if s.Canceled != nil && s.Canceled(context.Cause(ctx)) {
				stop("Canceled")
			} else {
				stop("WorkerShutdown")
			}
		case <-kill:
			kill = nil
			_ = syscall.Kill(-pgid, syscall.SIGKILL)
		case <-ticker.C:
			fresh := drain()
			if stopReason == "" && (fresh || time.Since(lastBeat) >= every) {
				beat()
			}
		}
	}
	// Whatever the child left behind in its group goes too.
	_ = syscall.Kill(-pgid, syscall.SIGKILL)
	fresh := drain()
	if fresh || string(last) != acknowledged {
		// A stopping poll can consume a checkpoint without heartbeating it.
		// Also retry a checkpoint whose earlier heartbeat failed. Flush the
		// latest unacknowledged resume point even when the final read is empty.
		fctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		_, _ = hb(fctx, last)
		cancel()
	}

	var f *Failure
	if waitErr != nil && cmd.ProcessState == nil {
		f = &Failure{Type: "WaitFailed", Message: waitErr.Error(), ExitCode: -1}
	} else {
		f = Classify(cmd.ProcessState, tail.Bytes())
	}
	if f != nil {
		f.Details = last
	}
	switch {
	case f == nil:
		// exit 0: the work finished, even if a stop raced with it.
	case stopReason == "WorkerShutdown":
		// Not a cancel: the work is unfinished and another worker resumes it.
		return nil, &Failure{Type: "WorkerShutdown", Message: "the worker stopped the activity: " + string(tail.Bytes()), ExitCode: f.ExitCode, Signal: f.Signal, Details: last}
	case stopReason == "Canceled":
		f.Type, f.NoRetry = "Canceled", true
		return nil, f
	default:
		return nil, f
	}
	done, err := os.ReadFile(filepath.Join(dir, "DONE.json"))
	if err != nil {
		return nil, &Failure{Type: "MissingDone", Message: "exit 0 without DONE.json: " + err.Error(), ExitCode: 0}
	}
	if len(done) > MaxResult {
		return nil, &Failure{Type: "ResultTooLarge", Message: fmt.Sprintf("DONE.json is %d bytes (max %d)", len(done), MaxResult), NoRetry: true}
	}
	return done, nil
}
