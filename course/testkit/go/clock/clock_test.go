package clock_test

import (
	"testing"
	"time"

	"openlearn.urmzd.com/tl/testkit/clock"
)

var t0 = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

func TestTimersFireInDeadlineOrder(t *testing.T) {
	f := clock.NewFake(t0)
	a, b := f.After(2*time.Second), f.After(time.Second)
	f.Advance(1500 * time.Millisecond)
	select {
	case got := <-b:
		if !got.Equal(t0.Add(time.Second)) {
			t.Fatalf("b fired at %v", got)
		}
	default:
		t.Fatal("the 1 s timer did not fire at 1.5 s")
	}
	select {
	case <-a:
		t.Fatal("the 2 s timer fired early")
	default:
	}
	f.Advance(time.Second)
	if got := <-a; !got.Equal(t0.Add(2 * time.Second)) {
		t.Fatalf("a fired at %v", got)
	}
	if !f.Now().Equal(t0.Add(2500 * time.Millisecond)) {
		t.Fatalf("now = %v", f.Now())
	}
}

func TestSleepWithBlockUntil(t *testing.T) {
	f := clock.NewFake(t0)
	done := make(chan struct{})
	go func() { f.Sleep(time.Minute); close(done) }()
	f.BlockUntil(1)
	f.Advance(time.Minute)
	<-done
}

func TestStopAndReset(t *testing.T) {
	f := clock.NewFake(t0)
	tm := f.NewTimer(time.Second)
	if !tm.Stop() {
		t.Fatal("Stop of an active timer returns true")
	}
	f.Advance(2 * time.Second)
	select {
	case <-tm.C():
		t.Fatal("a stopped timer fired")
	default:
	}
	tm.Reset(time.Second)
	f.Advance(time.Second)
	<-tm.C()
	if f.Waiters() != 0 {
		t.Fatalf("%d waiters left", f.Waiters())
	}
}

var _ clock.Clock = clock.Real{}
var _ clock.Clock = (*clock.Fake)(nil)
