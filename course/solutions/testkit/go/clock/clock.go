// Package clock is the time seam of the durable engine, gateway, and
// loadgen contracts (no time.Sleep in tests). Production code
// takes a Clock; tests pass a *Fake and move time with Advance.
package clock

import (
	"sort"
	"sync"
	"time"
)

// Clock is the contract every time-dependent component takes.
type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
	NewTimer(d time.Duration) Timer
	Sleep(d time.Duration)
}

// Timer is the subset of *time.Timer the contracts use.
type Timer interface {
	C() <-chan time.Time
	Stop() bool
	Reset(d time.Duration) bool
}

// Real is the wall clock.
type Real struct{}

func (Real) Now() time.Time                         { return time.Now() }
func (Real) After(d time.Duration) <-chan time.Time { return time.After(d) }
func (Real) Sleep(d time.Duration)                  { time.Sleep(d) }
func (Real) NewTimer(d time.Duration) Timer         { return realTimer{time.NewTimer(d)} }

type realTimer struct{ t *time.Timer }

func (r realTimer) C() <-chan time.Time        { return r.t.C }
func (r realTimer) Stop() bool                 { return r.t.Stop() }
func (r realTimer) Reset(d time.Duration) bool { return r.t.Reset(d) }

// Fake is a manual clock: time moves only on Advance, and timers fire in
// deadline order (ties in creation order) as it passes them.
type Fake struct {
	mu      sync.Mutex
	now     time.Time
	seq     int
	waiters []*fakeTimer
	changed *sync.Cond
}

// NewFake starts a fake clock at t.
func NewFake(t time.Time) *Fake {
	f := &Fake{now: t}
	f.changed = sync.NewCond(&f.mu)
	return f
}

func (f *Fake) Now() time.Time {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.now
}

func (f *Fake) After(d time.Duration) <-chan time.Time { return f.NewTimer(d).C() }

// Sleep blocks until another goroutine advances the clock past now+d.
func (f *Fake) Sleep(d time.Duration) { <-f.After(d) }

func (f *Fake) NewTimer(d time.Duration) Timer {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.seq++
	t := &fakeTimer{f: f, ch: make(chan time.Time, 1), at: f.now.Add(d), seq: f.seq, active: true}
	if d <= 0 {
		t.active = false
		t.ch <- f.now
	} else {
		f.waiters = append(f.waiters, t)
	}
	f.changed.Broadcast()
	return t
}

// Advance moves time forward by d, firing every timer whose deadline it passes.
func (f *Fake) Advance(d time.Duration) {
	f.mu.Lock()
	target := f.now.Add(d)
	for {
		sort.SliceStable(f.waiters, func(i, j int) bool {
			if f.waiters[i].at.Equal(f.waiters[j].at) {
				return f.waiters[i].seq < f.waiters[j].seq
			}
			return f.waiters[i].at.Before(f.waiters[j].at)
		})
		if len(f.waiters) == 0 || f.waiters[0].at.After(target) {
			break
		}
		t := f.waiters[0]
		f.waiters = f.waiters[1:]
		f.now = t.at
		t.active = false
		select {
		case t.ch <- t.at:
		default:
		}
	}
	f.now = target
	f.mu.Unlock()
}

// Waiters is the number of pending timers (Sleep and After included).
func (f *Fake) Waiters() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.waiters)
}

// BlockUntil waits until n timers are pending, so a test can advance only
// after the code under test has started waiting (no sleeps, no races).
func (f *Fake) BlockUntil(n int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for len(f.waiters) < n {
		f.changed.Wait()
	}
}

type fakeTimer struct {
	f      *Fake
	ch     chan time.Time
	at     time.Time
	seq    int
	active bool
}

func (t *fakeTimer) C() <-chan time.Time { return t.ch }

func (t *fakeTimer) Stop() bool {
	t.f.mu.Lock()
	defer t.f.mu.Unlock()
	was := t.active
	t.active = false
	for i, w := range t.f.waiters {
		if w == t {
			t.f.waiters = append(t.f.waiters[:i], t.f.waiters[i+1:]...)
			break
		}
	}
	return was
}

func (t *fakeTimer) Reset(d time.Duration) bool {
	was := t.Stop()
	t.f.mu.Lock()
	defer t.f.mu.Unlock()
	t.f.seq++
	t.at, t.seq, t.active = t.f.now.Add(d), t.f.seq, true
	select {
	case <-t.ch:
	default:
	}
	t.f.waiters = append(t.f.waiters, t)
	t.f.changed.Broadcast()
	return was
}
