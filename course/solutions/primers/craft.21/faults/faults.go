// Package faults is the course's fault kit for the craft.21 kata (rung R10,
// DESIGN 5.12: "fault injectors as a library"). It is course code: the
// check copies it into primers/craft.21/faults/ the first time and always
// uses its own copy when grading. Your resilience tests import it.
//
//	CrashFS   an in-memory FS where only synced bytes survive Crash(), a
//	          torn tail can be kept, and writes can be made to fail
//	Clock     a manual clock: time moves only on Advance; After channels fire
//	          in deadline order; BlockUntil waits for sleepers (no time.Sleep)
//	Effects   an external system that deduplicates by idempotency key and
//	          counts every delivery, so a test sees state and retries
//
// The same three ideas, at system scale, are course/testkit/go
// (failpoint, clock, effects) and ol drill.
package faults

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"sync"
	"testing"
	"time"

	"craft21"
)

// ErrCrashed is returned by a file handle opened before a Crash.
var ErrCrashed = errors.New("faults: the process crashed; this handle is dead")

// CrashFS models a disk under a process that can die: Write puts bytes in
// the page cache, Sync makes them durable, and Crash (power loss or kernel
// panic) throws away everything not synced. A plain SIGKILL loses nothing
// a write returned, which is why a kill alone cannot catch a missing fsync.
type CrashFS struct {
	mu      sync.Mutex
	durable map[string][]byte
	pending map[string][]byte
	gen     int
	failErr error
}

func NewCrashFS() *CrashFS {
	return &CrashFS{durable: map[string][]byte{}, pending: map[string][]byte{}}
}

type file struct {
	fs   *CrashFS
	name string
	gen  int
}

// Append opens name for appending (creating it).
func (fs *CrashFS) Append(name string) (craft21.File, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if _, ok := fs.durable[name]; !ok {
		fs.durable[name] = nil
	}
	return &file{fs: fs, name: name, gen: fs.gen}, nil
}

// ReadFile is what a running process sees: durable plus pending bytes.
func (fs *CrashFS) ReadFile(name string) ([]byte, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	d, ok := fs.durable[name]
	if !ok {
		return nil, fmt.Errorf("%s: %w", name, os.ErrNotExist)
	}
	return append(append([]byte(nil), d...), fs.pending[name]...), nil
}

func (f *file) Write(p []byte) (int, error) {
	f.fs.mu.Lock()
	defer f.fs.mu.Unlock()
	if f.gen != f.fs.gen {
		return 0, ErrCrashed
	}
	if f.fs.failErr != nil {
		return 0, f.fs.failErr
	}
	f.fs.pending[f.name] = append(f.fs.pending[f.name], p...)
	return len(p), nil
}

func (f *file) Sync() error {
	f.fs.mu.Lock()
	defer f.fs.mu.Unlock()
	if f.gen != f.fs.gen {
		return ErrCrashed
	}
	if f.fs.failErr != nil {
		return f.fs.failErr
	}
	f.fs.durable[f.name] = append(f.fs.durable[f.name], f.fs.pending[f.name]...)
	delete(f.fs.pending, f.name)
	return nil
}

func (f *file) Close() error { return nil }

// Crash loses every unsynced byte and kills every open handle.
func (fs *CrashFS) Crash() { fs.CrashTorn(0) }

// CrashTorn is Crash, except that the first keep bytes of each file's
// unsynced tail reach the disk: a write torn by the power cut.
func (fs *CrashFS) CrashTorn(keep int) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	for name, p := range fs.pending {
		if keep > len(p) {
			keep = len(p)
		}
		fs.durable[name] = append(fs.durable[name], p[:keep]...)
	}
	fs.pending = map[string][]byte{}
	fs.gen++
	fs.failErr = nil
}

// FailWrites makes every later Write and Sync return err (a full disk);
// nil heals it.
func (fs *CrashFS) FailWrites(err error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	fs.failErr = err
}

// Durable is what would survive a crash now.
func (fs *CrashFS) Durable(name string) []byte {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return append([]byte(nil), fs.durable[name]...)
}

// Clock is a manual clock.
type Clock struct {
	mu      sync.Mutex
	now     time.Time
	waiters []*waiter
	seq     int
}

type waiter struct {
	at  time.Time
	seq int
	ch  chan time.Time
}

func NewClock(t time.Time) *Clock { return &Clock{now: t} }

func (c *Clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

// After fires once Advance moves the clock to now + d or past it.
func (c *Clock) After(d time.Duration) <-chan time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch := make(chan time.Time, 1)
	if d <= 0 {
		ch <- c.now
		return ch
	}
	c.seq++
	c.waiters = append(c.waiters, &waiter{at: c.now.Add(d), seq: c.seq, ch: ch})
	return ch
}

// Advance moves time by d, firing due waiters in deadline order.
func (c *Clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
	sort.Slice(c.waiters, func(i, j int) bool {
		if c.waiters[i].at.Equal(c.waiters[j].at) {
			return c.waiters[i].seq < c.waiters[j].seq
		}
		return c.waiters[i].at.Before(c.waiters[j].at)
	})
	keep := c.waiters[:0]
	for _, w := range c.waiters {
		if !w.at.After(c.now) {
			w.ch <- c.now
		} else {
			keep = append(keep, w)
		}
	}
	c.waiters = keep
}

// Waiters is the number of pending After channels.
func (c *Clock) Waiters() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.waiters)
}

// BlockUntil waits (up to 10 s of real time) until n After channels are
// pending, so a test advances time only once the code is waiting.
func (c *Clock) BlockUntil(n int) error {
	deadline := time.Now().Add(10 * time.Second)
	for c.Waiters() < n {
		if time.Now().After(deadline) {
			return fmt.Errorf("faults: %d waiters after 10 s, want %d", c.Waiters(), n)
		}
		time.Sleep(time.Millisecond)
	}
	return nil
}

// Effects is an external system that deduplicates by idempotency key.
type Effects struct {
	mu         sync.Mutex
	value      map[string]string
	deliveries map[string]int
	fail       map[string]int // key -> remaining failures to inject
}

func NewEffects() *Effects {
	return &Effects{value: map[string]string{}, deliveries: map[string]int{}, fail: map[string]int{}}
}

// Apply records one delivery; the first one per key sets the value.
func (e *Effects) Apply(key, value string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if n := e.fail[key]; n > 0 {
		e.fail[key] = n - 1
		return fmt.Errorf("faults: injected failure applying %s", key)
	}
	e.deliveries[key]++
	if _, ok := e.value[key]; !ok {
		e.value[key] = value
	}
	return nil
}

// FailNext makes the next n deliveries of key fail before applying.
func (e *Effects) FailNext(key string, n int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.fail[key] = n
}

// Keys are the distinct keys applied, sorted.
func (e *Effects) Keys() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]string, 0, len(e.value))
	for k := range e.value {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Deliveries is how many times key was delivered (retries included).
func (e *Effects) Deliveries(key string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.deliveries[key]
}

// Value is the applied value of key.
func (e *Effects) Value(key string) (string, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	v, ok := e.value[key]
	return v, ok
}

// AssertExactlyOnce fails t unless the applied keys are exactly keys.
func (e *Effects) AssertExactlyOnce(t testing.TB, keys []string) {
	t.Helper()
	want := append([]string(nil), keys...)
	sort.Strings(want)
	got := e.Keys()
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("effects applied for keys %v, want exactly %v", got, want)
	}
}
