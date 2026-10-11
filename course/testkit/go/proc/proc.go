// Package proc gives durable tests KillLoop (DESIGN 4.4, MS-durable): start
// a learner binary, SIGKILL it at seeded random offsets, restart it, and
// after the last restart let it finish and check the invariants.
package proc

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// Options of one kill loop.
type Options struct {
	Kills     int           // how many SIGKILLs before the final run
	MinUp     time.Duration // earliest kill after a start
	Jitter    time.Duration // kill at MinUp + uniform[0, Jitter)
	Seed      uint64        // the offsets are a PCG stream of this seed (OL_SEED)
	Finish    time.Duration // how long the final run may take before it is killed as hung
	Restart   time.Duration // pause between a kill and the restart
	OnRestart func(i int)   // called before each restart (i = 1..Kills)
}

// Result of a loop.
type Result struct {
	Kills    int             // SIGKILLs delivered while the process ran
	Offsets  []time.Duration // when each kill landed after its start
	Exited   []int           // exit codes of runs that ended before their kill
	FinalErr error           // the final run's error (nil on exit 0)
	Output   string          // combined output of every run, separated by markers
}

// KillLoop runs newCmd() repeatedly: Kills times it is SIGKILLed after a
// seeded offset (a run that exits first is just restarted), then the last run
// goes to completion. check (may be nil) runs after the final exit and its
// error is returned: that is where exactly-once and replay invariants live.
func KillLoop(newCmd func() *exec.Cmd, o Options, check func() error) (Result, error) {
	rng := rand.New(rand.NewPCG(o.Seed, 0x6b696c6c))
	var res Result
	var out bytes.Buffer
	var mu sync.Mutex
	if o.Finish == 0 {
		o.Finish = time.Minute
	}
	for i := 0; i <= o.Kills; i++ {
		if i > 0 {
			if o.Restart > 0 {
				time.Sleep(o.Restart)
			}
			if o.OnRestart != nil {
				o.OnRestart(i)
			}
		}
		cmd := newCmd()
		w := &lockedWriter{mu: &mu, b: &out}
		cmd.Stdout, cmd.Stderr = w, w
		fmt.Fprintf(w, "--- run %d ---\n", i+1)
		if err := cmd.Start(); err != nil {
			return res, fmt.Errorf("start run %d: %w", i+1, err)
		}
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		if i < o.Kills {
			off := o.MinUp
			if o.Jitter > 0 {
				off += time.Duration(rng.Int64N(int64(o.Jitter)))
			}
			select {
			case err := <-done:
				res.Exited = append(res.Exited, exitCode(err))
			case <-time.After(off):
				_ = cmd.Process.Signal(syscall.SIGKILL)
				<-done
				res.Kills++
				res.Offsets = append(res.Offsets, off)
			}
			continue
		}
		select {
		case err := <-done:
			res.FinalErr = err
		case <-time.After(o.Finish):
			_ = cmd.Process.Signal(syscall.SIGKILL)
			<-done
			res.FinalErr = fmt.Errorf("the final run did not finish within %v", o.Finish)
		}
	}
	res.Output = out.String()
	if res.FinalErr != nil {
		return res, res.FinalErr
	}
	if check != nil {
		return res, check()
	}
	return res, nil
}

func exitCode(err error) int {
	if err == nil {
		return 0
	}
	if ee, ok := err.(*exec.ExitError); ok {
		return ee.ExitCode()
	}
	return -1
}

type lockedWriter struct {
	mu *sync.Mutex
	b  *bytes.Buffer
}

func (w *lockedWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.b.Write(p)
}
