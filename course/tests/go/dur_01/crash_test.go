package dur_01

// Crash tests: the test binary re-executes itself as a child that appends to
// a log in a loop and prints "acked N" after each acknowledged append; the
// parent SIGKILLs it at seeded offsets (testkit proc.KillLoop) or lets a
// failpoint crash it between write and fsync, then checks the log.

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
	"time"

	dlog "tinyllm/durable/log"

	"openlearn.urmzd.com/tl/testkit/failpoint"
	"openlearn.urmzd.com/tl/testkit/proc"
)

const childEnv = "DUR01_CHILD_DIR"

func TestMain(m *testing.M) {
	if dir := os.Getenv(childEnv); dir != "" {
		os.Exit(appender(dir))
	}
	os.Exit(m.Run())
}

// appender is the child: append to stream "k" up to DUR01_TARGET, the data of
// version v being v itself, printing every acknowledged version.
func appender(dir string) int {
	target, _ := strconv.ParseInt(os.Getenv("DUR01_TARGET"), 10, 64)
	l, err := dlog.Open(dir, dlog.Options{Sync: fastSync, Failpoint: failpoint.Inject})
	if err != nil {
		fmt.Println("open:", err)
		return 3
	}
	out := bufio.NewWriter(os.Stdout)
	// One append per millisecond, so a kill lands mid-run at the same rate
	// whether or not the binary was built with -race.
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for v := l.Version("k"); v < target; {
		<-tick.C
		nv, err := l.Append(ctx, "k", v, ev("n", strconv.FormatInt(v+1, 10)))
		if err != nil {
			fmt.Println("append:", err)
			return 4
		}
		fmt.Fprintf(out, "acked %d\n", nv)
		out.Flush()
		v = nv
	}
	return 0
}

func child(dir string, target int, failpoints string) *exec.Cmd {
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append(os.Environ(), childEnv+"="+dir, "DUR01_TARGET="+strconv.Itoa(target), "TL_FAILPOINTS="+failpoints)
	return cmd
}

func ackedVersions(out string) []int64 {
	var acked []int64
	for _, line := range strings.Split(out, "\n") {
		if v, ok := strings.CutPrefix(line, "acked "); ok {
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err == nil {
				acked = append(acked, n)
			}
		}
	}
	return acked
}

// checkDense opens dir and requires stream "k" to hold versions 1..n with
// data equal to the version: no gap, no duplicate, no foreign record.
func checkDense(dir string) (int64, error) {
	l, err := dlog.Open(dir, dlog.Options{Sync: fastSync})
	if err != nil {
		return 0, fmt.Errorf("reopen: %w", err)
	}
	defer l.Close()
	evs, err := l.Read(ctx, "k", 1, 0)
	if err != nil {
		return 0, err
	}
	for i, e := range evs {
		if e.Version != int64(i+1) || string(e.Data) != strconv.Itoa(i+1) {
			return 0, fmt.Errorf("event %d is version %d with data %q", i, e.Version, e.Data)
		}
	}
	if l.LastSeq() != uint64(len(evs)) {
		return 0, fmt.Errorf("seq %d but %d records", l.LastSeq(), len(evs))
	}
	return int64(len(evs)), nil
}

func seed() uint64 {
	s, _ := strconv.ParseUint(os.Getenv("OL_SEED"), 10, 64)
	return s
}

func TestKillLoopAckedSurvive(t *testing.T) {
	// WHY: the log's whole promise: SIGKILL the writer at 30 seeded instants
	//      and every version it printed as acknowledged is still there, in
	//      order, with its own data; each restart continues at last+1.
	// KIND: fault
	// CATCHES: s05, s08, s09
	// CHAPTER: dur.01 section 4, kill loop
	dir := t.TempDir()
	const target = 1000
	res, err := proc.KillLoop(func() *exec.Cmd { return child(dir, target, "") },
		proc.Options{Kills: 30, MinUp: 20 * time.Millisecond, Jitter: 40 * time.Millisecond, Seed: seed(), Finish: 30 * time.Second},
		func() error {
			n, err := checkDense(dir)
			if err != nil {
				return err
			}
			if n != target {
				return fmt.Errorf("final run ended at version %d, want %d", n, target)
			}
			return nil
		})
	if err != nil {
		t.Fatalf("kill loop: %v\n%s", err, tail(res.Output))
	}
	acked := ackedVersions(res.Output)
	for i := 1; i < len(acked); i++ {
		// A restart resumes after the last durable version, so the printed
		// versions strictly increase across every kill.
		if acked[i] <= acked[i-1] {
			t.Fatalf("acked %d after %d: a version was acknowledged twice", acked[i], acked[i-1])
		}
	}
	if res.Kills < 10 {
		t.Fatalf("only %d of 30 kills landed while the child ran; the loop tested nothing", res.Kills)
	}
}

func TestFailpointCrashAfterWrite(t *testing.T) {
	// WHY: TL_FAILPOINTS="dur/log/after-write-before-fsync=6*crash" kills the
	//      writer between write and fsync on its 6th append: the five acked
	//      versions survive, the unacked record is whole or absent (never
	//      half), and a restart continues densely to the target.
	// KIND: fault
	// CATCHES: s08, s09
	// CHAPTER: dur.01 section 4, failpoint
	dir := t.TempDir()
	out, err := child(dir, 50, dlog.FailpointAfterWrite+"=6*crash").CombinedOutput()
	if code := exitCode(err); code != 137 {
		t.Fatalf("child exit %d, want 137 from the crash failpoint\n%s", code, out)
	}
	acked := ackedVersions(string(out))
	if len(acked) != 5 || acked[4] != 5 {
		t.Fatalf("acked %v before the crash, want 1..5", acked)
	}
	n, err := checkDense(dir)
	if err != nil || n < 5 || n > 6 {
		t.Fatalf("after the crash: %d events, %v (want 5 or 6, dense)", n, err)
	}
	if out, err := child(dir, 50, "").CombinedOutput(); err != nil {
		t.Fatalf("restart: %v\n%s", err, out)
	}
	if n, err := checkDense(dir); err != nil || n != 50 {
		t.Fatalf("after the restart: %d events, %v (want 50)", n, err)
	}
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

func tail(s string) string {
	if len(s) > 2000 {
		return "..." + s[len(s)-2000:]
	}
	return s
}
