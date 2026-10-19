// Package timer is the durable server's timer service (dur.07): a binary
// min-heap of deadlines written by hand (it replaces practice go/04's sorted
// set), a service that fires each due timer exactly once in deadline order,
// and the --test-clock clock that POST /debug/clock moves forward.
//
// Durability is not here: a timer is durable because the server records
// TimerStarted in history and re-arms every pending timer on recovery. This
// service only has to fire what it holds, once, in order.
package timer

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"time"
)

// Clock is the time seam; the testkit's *clock.Fake satisfies it.
type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

// Entry is one scheduled timer.
type Entry[K comparable] struct {
	Key   K
	At    time.Time
	Seq   uint64 // schedule order: ties at the same instant fire first-scheduled first
	index int    // position in the heap array, kept by the heap for O(log n) removal
}

// Heap is a binary min-heap of entries ordered by (At, seq), stored in an
// array: the children of i are 2i+1 and 2i+2, its parent (i-1)/2.
type Heap[K comparable] struct {
	items []*Entry[K]
}

func (h *Heap[K]) Len() int { return len(h.items) }

// Items lists the keys in array order (for tests and the chapter's trace).
func (h *Heap[K]) Items() []K {
	out := make([]K, len(h.items))
	for i, e := range h.items {
		out[i] = e.Key
	}
	return out
}

func (h *Heap[K]) less(i, j int) bool {
	a, b := h.items[i], h.items[j]
	if !a.At.Equal(b.At) {
		return a.At.Before(b.At)
	}
	return a.Seq < b.Seq
}

func (h *Heap[K]) swap(i, j int) {
	h.items[i], h.items[j] = h.items[j], h.items[i]
	h.items[i].index = i
	h.items[j].index = j
}

// up moves item i toward the root while it is smaller than its parent.
func (h *Heap[K]) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !h.less(i, p) {
			return
		}
		h.swap(i, p)
		i = p
	}
}

// down moves item i toward the leaves while a child is smaller.
func (h *Heap[K]) down(i int) {
	n := len(h.items)
	for {
		l, small := 2*i+1, i
		if l < n && h.less(l, small) {
			small = l
		}
		if r := l + 1; r < n && h.less(r, small) {
			small = r
		}
		if small == i {
			return
		}
		h.swap(i, small)
		i = small
	}
}

// Push adds e.
func (h *Heap[K]) Push(e *Entry[K]) {
	e.index = len(h.items)
	h.items = append(h.items, e)
	h.up(e.index)
}

// Peek is the smallest entry (nil when empty).
func (h *Heap[K]) Peek() *Entry[K] {
	if len(h.items) == 0 {
		return nil
	}
	return h.items[0]
}

// Pop removes and returns the smallest entry (nil when empty).
func (h *Heap[K]) Pop() *Entry[K] {
	if len(h.items) == 0 {
		return nil
	}
	return h.Remove(h.items[0])
}

// Remove takes e out of the heap: move the last item into its slot, then
// restore the order up or down from there.
func (h *Heap[K]) Remove(e *Entry[K]) *Entry[K] {
	i, last := e.index, len(h.items)-1
	if i < 0 || i > last || h.items[i] != e {
		return nil
	}
	if i != last {
		h.swap(i, last)
	}
	h.items[last] = nil
	h.items = h.items[:last]
	if i < last {
		h.down(i)
		h.up(i)
	}
	e.index = -1
	return e
}

// Service holds the armed timers and fires them.
type Service[K comparable] struct {
	mu    sync.Mutex
	clk   Clock
	heap  Heap[K]
	byKey map[K]*Entry[K]
	seq   uint64
	wake  chan struct{}
}

// New makes an empty service on clk.
func New[K comparable](clk Clock) *Service[K] {
	return &Service[K]{clk: clk, byKey: map[K]*Entry[K]{}, wake: make(chan struct{})}
}

func (s *Service[K]) notify() {
	close(s.wake)
	s.wake = make(chan struct{})
}

// Schedule arms key to fire at at. Scheduling a key that is already armed
// moves it (recovery re-arms timers it may already hold: never two).
func (s *Service[K]) Schedule(key K, at time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.seq++
	if e := s.byKey[key]; e != nil {
		s.heap.Remove(e)
	}
	e := &Entry[K]{Key: key, At: at, Seq: s.seq}
	s.byKey[key] = e
	s.heap.Push(e)
	s.notify()
}

// Cancel disarms key; a key that is not armed is a no-op.
func (s *Service[K]) Cancel(key K) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.byKey[key]; e != nil {
		s.heap.Remove(e)
		delete(s.byKey, key)
		s.notify()
	}
}

// Len is the number of armed timers.
func (s *Service[K]) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.heap.Len()
}

// Wake makes Run re-read the clock now (after the clock jumps).
func (s *Service[K]) Wake() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.notify()
}

// Run fires every due timer, earliest first (ties in schedule order), each
// exactly once: it is removed before fire is called. Then it sleeps until the
// next deadline, a Schedule or Cancel, a Wake, or ctx's end.
func (s *Service[K]) Run(ctx context.Context, fire func(K)) error {
	for ctx.Err() == nil {
		s.mu.Lock()
		now := s.clk.Now()
		var due []K
		for e := s.heap.Peek(); e != nil && !e.At.After(now); e = s.heap.Peek() {
			s.heap.Pop()
			delete(s.byKey, e.Key)
			due = append(due, e.Key)
		}
		var timeout <-chan time.Time
		if e := s.heap.Peek(); e != nil && len(due) == 0 {
			timeout = s.clk.After(e.At.Sub(now))
		}
		wake := s.wake
		s.mu.Unlock()
		for _, k := range due {
			fire(k)
		}
		if len(due) > 0 {
			continue
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-wake:
		case <-timeout:
		}
	}
	return ctx.Err()
}

// OffsetClock is the clock of `{durable} --test-clock`: base time plus an
// offset that only grows. POST /debug/clock {"offset_ms": N} adds N ms (the
// clock-skew drill uses it); every OnShift callback runs after a shift.
type OffsetClock struct {
	mu        sync.Mutex
	base      Clock
	offset    time.Duration
	listeners []func()
}

// NewOffsetClock wraps base (the wall clock in the server).
func NewOffsetClock(base Clock) *OffsetClock { return &OffsetClock{base: base} }

func (c *OffsetClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.base.Now().Add(c.offset)
}

func (c *OffsetClock) After(d time.Duration) <-chan time.Time { return c.base.After(d) }

// OnShift registers f to run after every shift (Service.Wake, so timers that
// became due fire now instead of at their old wall-clock instant).
func (c *OffsetClock) OnShift(f func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.listeners = append(c.listeners, f)
}

// ErrBackwards: time never moves backwards, not even in a test.
var ErrBackwards = errors.New("timer: the test clock only moves forward")

// Shift moves the clock forward by d and notifies the listeners.
func (c *OffsetClock) Shift(d time.Duration) error {
	if d < 0 {
		return ErrBackwards
	}
	c.mu.Lock()
	c.offset += d
	ls := append([]func(){}, c.listeners...)
	c.mu.Unlock()
	for _, f := range ls {
		f()
	}
	return nil
}

// Offset is the total shift so far.
func (c *OffsetClock) Offset() time.Duration {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.offset
}

// Handler serves /debug/clock: POST {"offset_ms": N} shifts by N ms and
// answers {"offset_ms": total}; GET answers the total; a negative or
// malformed body is 400; other methods 405.
func (c *OffsetClock) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
		case http.MethodPost:
			var in struct {
				OffsetMs *int64 `json:"offset_ms"`
			}
			if err := json.NewDecoder(r.Body).Decode(&in); err != nil || in.OffsetMs == nil {
				http.Error(w, `{"error":"want {\"offset_ms\": N}"}`, http.StatusBadRequest)
				return
			}
			if err := c.Shift(time.Duration(*in.OffsetMs) * time.Millisecond); err != nil {
				http.Error(w, `{"error":"the test clock only moves forward"}`, http.StatusBadRequest)
				return
			}
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]int64{"offset_ms": c.Offset().Milliseconds()})
	})
}
