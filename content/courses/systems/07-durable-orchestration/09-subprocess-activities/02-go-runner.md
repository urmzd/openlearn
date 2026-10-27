# The Go subprocess runner

This part of [Subprocess activities: the Go runner and the Python helper](README.md) develops the worked mechanisms below.

> **Learning objectives**
> - Trace process-group startup, monitoring, shutdown, and retry state in the runner.

## Building the Go runner

The principles above translate into code almost line for line. Start with the pieces the runner's loop depends on, then the loop itself.

### The work directory

Each segment of the key is validated before it is joined, so `filepath.Join` can never climb out of `<artifacts>/activities` (Listing 4).

```go
var keySegment = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

func WorkDir(artifacts, key string) (string, error) {
	if artifacts == "" {
		return "", errors.New("activities: no artifacts root")
	}
	segs := strings.Split(key, "/")
	for _, s := range segs {
		// "." and ".." match the character class, so they are rejected by name;
		// an empty segment (a leading or doubled "/") fails the regexp.
		if s == "." || s == ".." || !keySegment.MatchString(s) {
			return "", fmt.Errorf("activities: idempotency key %q has a bad segment %q", key, s)
		}
	}
	// The same key always gives the same directory: that is what makes a retry a resume.
	return filepath.Join(append([]string{artifacts, "activities"}, segs...)...), nil
}
```

### Tailing a growing file

`Tail` remembers a byte offset and the unfinished tail of the last read. Only text up to a newline is parsed (Listing 5).

```go
func (t *Tail) Read() ([]Progress, error) {
	f, err := os.Open(t.Path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil // the child has not written anything yet
	}
	// ...
	if _, err := f.Seek(t.offset, io.SeekStart); err != nil {
		return nil, err
	}
	b, err := io.ReadAll(f)
	// ...
	t.offset += int64(len(b))
	buf := append(t.partial, b...) // the half line from last time comes first
	var out []Progress
	for {
		i := bytes.IndexByte(buf, '\n')
		if i < 0 {
			break // no newline: the rest is still being written
		}
		line := bytes.TrimSpace(buf[:i])
		buf = buf[i+1:]
		var p Progress
		// A line that is not a JSON object with a kind is skipped, never fatal.
		if len(line) > 0 && json.Unmarshal(line, &p) == nil && p.Kind != "" {
			out = append(out, p)
		}
	}
	t.partial = append([]byte(nil), buf...) // copy: buf aliases the read buffer
	return out, nil
}
```

### The exit-code table as code

`Classify` is the exit-code table translated directly. A signal is checked first, because a signalled process has no meaningful exit code (Listing 6).

```go
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
	case ExitDataErr: // the input is wrong: retrying cannot help
		return &Failure{Type: "ExitCode65", Message: msg, NoRetry: true, ExitCode: code}
	case ExitCanceled:
		return &Failure{Type: "Canceled", Message: msg, NoRetry: true, ExitCode: code}
	default: // 75 and every crash: retryable
		return &Failure{Type: "ExitCode" + strconv.Itoa(code), Message: msg, ExitCode: code}
	}
}
```

### The child's environment

`childEnv` removes inherited copies of every contract variable before setting this delivery's values. Without the `drop` set, a worker started under its own `TRACEPARENT` would leak it into an activity that has none, and the Python spans would join an unrelated trace (Listing 7).

```go
func (s Subprocess) childEnv(t Task) []string {
	// ...
	set := map[string]string{
		"TL_IDEMPOTENCY_KEY": t.IdempotencyKey,
		"TL_ATTEMPT":         strconv.Itoa(max(t.Attempt, 1)),
		"TL_ARTIFACTS":       s.Artifacts,
	}
	if tp := t.TraceContext["traceparent"]; tp != "" {
		set["TRACEPARENT"] = tp
		// ... TRACESTATE, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME likewise
	}
	// Trace variables are always dropped, even when this task sets none.
	drop := map[string]bool{"TRACEPARENT": true, "TRACESTATE": true, "OTEL_EXPORTER_OTLP_ENDPOINT": true}
	for k := range set {
		drop[k] = true
	}
	// ... copy base minus drop, then append set in a fixed order
}
```

### Starting the attempt

`Run` publishes the spec by rename, builds the argv the contract names, and starts the child as the leader of a new process group. stderr goes both to the attempt log and to a 4 KiB ring, so a chatty child cannot inflate the failure message (Listing 8).

```go
	specPath := filepath.Join(dir, "spec.json")
	progPath := filepath.Join(dir, "progress.jsonl")
	if err := writeAtomic(specPath, spec); err != nil { // spec.json.tmp, fsync, rename
		return nil, &Failure{Type: "WorkDir", Message: err.Error(), ExitCode: -1}
	}

	argv := append(append(append([]string{}, s.Entry...), args...), "--spec", specPath, "--progress", progPath)
	if len(t.LastHeartbeat) > 0 {
		argv = append(argv, "--resume", string(t.LastHeartbeat)) // the server's last checkpoint
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = s.childEnv(t)
	cmd.Dir = s.Dir
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // pgid == child's pid
	// ... open attempt-<n>.log
	tail := &ring{n: stderrTail}
	cmd.Stdout = logf
	cmd.Stderr = io.MultiWriter(logf, tail)

	// Start tailing at the current end: earlier attempts' events are history.
	tl := &Tail{Path: progPath}
	if fi, err := os.Stat(progPath); err == nil {
		tl.offset = fi.Size()
	}
```

Starting the tail at the file's current size means each attempt reads only its own events. Attempt 1's lines (step 500 in the worked example) are still in `progress.jsonl` when attempt 2 starts, but they are history, and the resume point they produced already arrived through `LastHeartbeat`.

### The state of one attempt

Three pieces of state drive the loop. `last` is the resume point, seeded from the task so a retry never heartbeats empty details; `acknowledged` is the last value the server accepted; and `stopReason` records why the runner sent SIGTERM (Listing 9).

```go
	last := append([]byte(nil), t.LastHeartbeat...)
	lastBeat := time.Time{}
	var stopReason string // "" running, "Canceled", "WorkerShutdown"
	var kill <-chan time.Time
	acknowledged := string(last) // checkpoint already recorded by the durable server

	stop := func(reason string) {
		if stopReason != "" {
			return // the first reason wins; one SIGTERM, one grace timer
		}
		stopReason = reason
		_ = syscall.Kill(-pgid, syscall.SIGTERM) // negative pid: the whole group
		kill = time.After(grace)
	}
	beat := func() {
		cancel, err := hb(ctx, last)
		lastBeat = time.Now()
		if err == nil && cancel {
			stop("Canceled")
		}
		if err == nil {
			acknowledged = string(last) // a failed heartbeat is not an acknowledgement
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
```

### The loop

One `select` serves four events. A fresh checkpoint triggers a heartbeat at once; otherwise the heartbeat fires every $T_{hb}/3$ (or `DefaultHeartbeat`), which leaves two missed beats of slack before the server fences (Listing 10).

```go
	ctxDone := ctx.Done()
	var waitErr error
loop:
	for {
		select {
		case waitErr = <-exited:
			break loop
		case <-ctxDone:
			ctxDone = nil // a closed channel would fire forever
			if s.Canceled != nil && s.Canceled(context.Cause(ctx)) {
				stop("Canceled")
			} else {
				stop("WorkerShutdown")
			}
		case <-kill:
			kill = nil
			_ = syscall.Kill(-pgid, syscall.SIGKILL) // the grace is over
		case <-ticker.C:
			fresh := drain()
			if stopReason == "" && (fresh || time.Since(lastBeat) >= every) {
				beat()
			}
		}
	}
```

### After the exit

When the child exits, the group is SIGKILLed once more (a grandchild may outlive its parent), the file is drained one last time, and any checkpoint the server has not acknowledged is flushed on a context that survives the attempt's own cancellation. Then the stop reason overrides the child's verdict (Listing 11).

```go
	_ = syscall.Kill(-pgid, syscall.SIGKILL)
	fresh := drain()
	if fresh || string(last) != acknowledged {
		fctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		_, _ = hb(fctx, last)
		cancel()
	}
	// ... f = Classify(cmd.ProcessState, tail.Bytes()); f.Details = last
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
	if len(done) > MaxResult { // 2 MiB: the result must fit a history event
		return nil, &Failure{Type: "ResultTooLarge", /* ... */ NoRetry: true}
	}
	return done, nil
```

The full runner is [`subprocess.go`](../../../../../course/solutions/go/activities/subprocess.go).

## Review questions

1. Why send termination to a process group rather than only the immediate child?

   <details><summary>Answer</summary>

   The child may spawn descendants that otherwise survive and retain files, devices, or background work.

   </details>

[Contents](README.md) · [Next: Python helper and exporter](03-python-helper-and-exporter.md)
