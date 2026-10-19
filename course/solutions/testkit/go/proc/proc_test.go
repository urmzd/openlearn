package proc_test

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/proc"
)

// A worker that appends 1..N to a journal, one line at a time, resuming from
// the last line it finds: the invariant is exactly N lines, each once.
const worker = `
f=$1; n=$2
last=$(tail -n 1 "$f" 2>/dev/null || echo 0)
i=$((last + 1))
while [ "$i" -le "$n" ]; do
  echo "$i" >> "$f"
  i=$((i + 1))
  sleep 0.002
done
`

func TestKillLoopResumesAndChecks(t *testing.T) {
	dir := t.TempDir()
	journal := filepath.Join(dir, "journal")
	script := filepath.Join(dir, "w.sh")
	if err := os.WriteFile(script, []byte(worker), 0o755); err != nil {
		t.Fatal(err)
	}
	restarts := 0
	res, err := proc.KillLoop(
		func() *exec.Cmd { return exec.Command("sh", script, journal, "200") },
		proc.Options{Kills: 4, MinUp: 20 * time.Millisecond, Jitter: 40 * time.Millisecond, Seed: 7, OnRestart: func(int) { restarts++ }},
		func() error {
			b, _ := os.ReadFile(journal)
			lines := strings.Fields(string(b))
			for i, l := range lines {
				if l != strconv.Itoa(i+1) {
					return errors.New("line " + strconv.Itoa(i+1) + " is " + l)
				}
			}
			if len(lines) != 200 {
				return errors.New("journal has " + strconv.Itoa(len(lines)) + " lines")
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("%v\n%s", err, res.Output)
	}
	if res.Kills+len(res.Exited) != 4 || restarts != 4 {
		t.Fatalf("kills %d, early exits %v, restarts %d", res.Kills, res.Exited, restarts)
	}
	if res.Kills == 0 {
		t.Fatal("no kill landed: the loop proves nothing")
	}
}

func TestOffsetsAreSeeded(t *testing.T) {
	run := func() []time.Duration {
		res, _ := proc.KillLoop(func() *exec.Cmd { return exec.Command("sleep", "5") },
			proc.Options{Kills: 3, MinUp: time.Millisecond, Jitter: 30 * time.Millisecond, Seed: 42, Finish: time.Millisecond},
			nil)
		return res.Offsets
	}
	a, b := run(), run()
	if len(a) != 3 || a[0] != b[0] || a[1] != b[1] || a[2] != b[2] {
		t.Fatalf("offsets differ: %v vs %v", a, b)
	}
}
