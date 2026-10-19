// The reference answer to craft.21: resilience tests for the kata, graded
// by the planted faults in course/mutants/craft.21 (rung R10). Each test
// kills, partitions, or starves the queue at the worst moment and checks one
// promise: effects exactly once, nothing acknowledged lost, one owner per
// task at a time.
package craft21_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"craft21"
	"craft21/faults"
)

var start = time.Unix(1760000000, 0)

const lease = 30 * time.Second

func reopen(t *testing.T, fs *faults.CrashFS, clk *faults.Clock) *craft21.Queue {
	t.Helper()
	q, err := craft21.Open(fs, clk)
	if err != nil {
		t.Fatalf("Open after a crash: %v", err)
	}
	return q
}

func upper(ctx context.Context, p string) (string, error) { return "R(" + p + ")", nil }

// lcg is a tiny seeded generator: the kill points are the same on every run.
type lcg uint64

func (g *lcg) next(n int) int {
	*g = *g*6364136223846793005 + 1442695040888963407
	return int(uint64(*g>>33) % uint64(n))
}

// TestKillLoopExactlyOnce kills the worker at a seeded point of every
// attempt (before the effect, after it, or after Complete), crashes the
// disk, restarts, and lets the lease expire, until every task is done.
func TestKillLoopExactlyOnce(t *testing.T) {
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(start), faults.NewEffects()
	q := reopen(t, fs, clk)
	var ids []string
	for i := 0; i < 20; i++ {
		id := fmt.Sprintf("task-%02d", i)
		ids = append(ids, id)
		if err := q.Enqueue(id, id); err != nil {
			t.Fatal(err)
		}
	}
	g := lcg(7)
	for round := 0; round < 200 && len(q.Pending()) > 0; round++ {
		point := g.next(4) // 0 before apply, 1 after apply, 2 after complete, 3 no kill
		w := &craft21.Worker{Q: q, ID: fmt.Sprintf("w%d", round), TTL: lease, Clock: clk, Work: upper,
			Apply: func(k, v string) error {
				if point == 0 {
					return errors.New("killed before applying")
				}
				fx.Apply(k, v)
				if point == 1 {
					return errors.New("killed after applying")
				}
				return nil
			}}
		w.RunOne(context.Background())
		if point != 3 {
			fs.Crash()
			q = reopen(t, fs, clk)
			clk.Advance(lease)
		}
	}
	if p := q.Pending(); len(p) > 0 {
		t.Fatalf("tasks never finished: %v", p)
	}
	fx.AssertExactlyOnce(t, ids)
	fs.Crash()
	q = reopen(t, fs, clk)
	for _, id := range ids {
		if r, ok := q.Result(id); !ok || r != "R("+id+")" {
			t.Fatalf("%s: acknowledged result lost or wrong after a crash: %q %v", id, r, ok)
		}
	}
}

func TestEnqueueTwiceIsOneTask(t *testing.T) {
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(start), faults.NewEffects()
	q := reopen(t, fs, clk)
	q.Enqueue("a", "x")
	w := &craft21.Worker{Q: q, ID: "w", TTL: lease, Clock: clk, Work: upper, Apply: fx.Apply}
	w.RunOne(context.Background())
	q.Enqueue("a", "y")
	if ok, _ := w.RunOne(context.Background()); ok || fx.Deliveries("a") != 1 {
		t.Fatalf("a duplicate Enqueue made a done task run again (deliveries %d)", fx.Deliveries("a"))
	}
}

func TestTornWriteKeepsTheRest(t *testing.T) {
	fs, clk := faults.NewCrashFS(), faults.NewClock(start)
	q := reopen(t, fs, clk)
	q.Enqueue("a", "x")
	f, _ := fs.Append(craft21.LogName)
	f.Write([]byte(`{"op":"enq","id":"b","payload":"y"}` + "\n"))
	fs.CrashTorn(7)
	q = reopen(t, fs, clk)
	if p := q.Pending(); len(p) != 1 || p[0] != "a" {
		t.Fatalf("pending after a torn write: %v", p)
	}
	q.Enqueue("c", "z")
	fs.Crash()
	q = reopen(t, fs, clk)
	if p := q.Pending(); len(p) != 2 || p[1] != "c" {
		t.Fatalf("a record acknowledged after the torn crash was lost: %v", p)
	}
}

func TestDiskFullLeavesStateAlone(t *testing.T) {
	fs, clk := faults.NewCrashFS(), faults.NewClock(start)
	q := reopen(t, fs, clk)
	q.Enqueue("a", "x")
	l, _ := q.Lease("w", lease)
	fs.FailWrites(errors.New("ENOSPC"))
	if err := q.Complete(l, "r"); err == nil {
		t.Fatal("Complete succeeded on a full disk")
	}
	if _, ok := q.Result("a"); ok {
		t.Fatal("memory says done, the disk does not")
	}
}

func TestOneOwnerAtATime(t *testing.T) {
	fs, clk := faults.NewCrashFS(), faults.NewClock(start)
	q := reopen(t, fs, clk)
	q.Enqueue("a", "x")
	first, _ := q.Lease("w1", lease)
	clk.Advance(lease - time.Nanosecond)
	if _, err := q.Lease("w2", lease); !errors.Is(err, craft21.ErrNoTask) {
		t.Fatalf("leased twice while the first lease was live: %v", err)
	}
	clk.Advance(time.Nanosecond)
	second, err := q.Lease("w2", lease)
	if err != nil || second.Token <= first.Token {
		t.Fatalf("an expired lease must be redelivered with a higher token: %+v %v", second, err)
	}
	if _, err := q.Renew(first, lease); !errors.Is(err, craft21.ErrLeaseLost) {
		t.Fatalf("the zombie renewed: %v", err)
	}
	if err := q.Complete(first, "zombie"); !errors.Is(err, craft21.ErrLeaseLost) {
		t.Fatalf("the zombie completed: %v", err)
	}
}

type gate struct{ started, release chan struct{} }

func (g gate) work(ctx context.Context, p string) (string, error) {
	close(g.started)
	select {
	case <-g.release:
		return "R(" + p + ")", nil
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func TestLongTaskIsNotStolen(t *testing.T) {
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(start), faults.NewEffects()
	q := reopen(t, fs, clk)
	q.Enqueue("a", "x")
	g := gate{make(chan struct{}), make(chan struct{})}
	w := &craft21.Worker{Q: q, ID: "w1", TTL: lease, Clock: clk, Work: g.work, Apply: fx.Apply}
	done := make(chan error, 1)
	go func() { _, err := w.RunOne(context.Background()); done <- err }()
	<-g.started
	for i := 0; i < 6; i++ {
		if err := clk.BlockUntil(1); err != nil {
			t.Fatal(err)
		}
		clk.Advance(lease / 3)
		if _, err := q.Lease("thief", lease); !errors.Is(err, craft21.ErrNoTask) {
			t.Fatalf("stolen after %d renewals: %v", i+1, err)
		}
	}
	close(g.release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestPausedWorkerStops(t *testing.T) {
	fs, clk, fx := faults.NewCrashFS(), faults.NewClock(start), faults.NewEffects()
	q := reopen(t, fs, clk)
	q.Enqueue("a", "x")
	g := gate{make(chan struct{}), make(chan struct{})}
	w := &craft21.Worker{Q: q, ID: "w1", TTL: lease, Clock: clk, Work: g.work, Apply: fx.Apply}
	done := make(chan error, 1)
	go func() { _, err := w.RunOne(context.Background()); done <- err }()
	<-g.started
	if err := clk.BlockUntil(1); err != nil {
		t.Fatal(err)
	}
	clk.Advance(3 * lease)
	select {
	case err := <-done:
		if !errors.Is(err, craft21.ErrLeaseLost) || fx.Deliveries("a") != 0 {
			t.Fatalf("after losing its lease: err %v, deliveries %d", err, fx.Deliveries("a"))
		}
	case <-time.After(5 * time.Second):
		close(g.release)
		t.Fatal("work kept running after the lease was lost")
	}
}
