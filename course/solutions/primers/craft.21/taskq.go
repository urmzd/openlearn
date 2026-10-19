// Package craft21 is the craft.21 kata: a small durable task queue and the
// worker that drains it, the same three mechanisms as the system's durable engine
// (dur.01 log, dur.03 fenced leases, dur.05 idempotent activities) in one
// file a test can kill and replay.
//
// Every state change is one JSON line appended to a log on an FS and synced
// before the call returns; Open replays the log (a torn last line is
// ignored). A task is leased with a token and a deadline; the worker renews
// its lease while working and applies its effect under the task id as the
// idempotency key, then completes the task with its token. A stale token is
// ErrLeaseLost.
//
// Your resilience tests (resilience_test.go) kill this code at its worst
// moments with the course's fault kit (faults/faults.go: CrashFS, Clock,
// Effects) and assert that every task's effect is applied exactly once and
// that nothing acknowledged is lost.
package craft21

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"sync"
	"time"
)

// File is an append-only file: bytes written are not durable until Sync.
type File interface {
	Write(p []byte) (int, error)
	Sync() error
	Close() error
}

// FS opens files for appending and reads them back.
type FS interface {
	Append(name string) (File, error)     // create if missing
	ReadFile(name string) ([]byte, error) // os.ErrNotExist when missing
}

// Clock is the time seam: the queue reads Now, the worker waits on After.
type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

// SystemClock is the wall clock.
type SystemClock struct{}

func (SystemClock) Now() time.Time                         { return time.Now() }
func (SystemClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// LogName is the queue's log file.
const LogName = "queue.log"

var (
	ErrNoTask    = errors.New("craft21: no task available")
	ErrLeaseLost = errors.New("craft21: lease lost")
	ErrUnknown   = errors.New("craft21: unknown task")
)

// Lease is one delivery of a task to a worker.
type Lease struct {
	Task     string
	Payload  string
	Worker   string
	Token    uint64
	Deadline time.Time
	Attempt  int
}

type record struct {
	Op       string `json:"op"` // enq | lease | renew | done
	ID       string `json:"id"`
	Payload  string `json:"payload,omitempty"`
	Worker   string `json:"worker,omitempty"`
	Token    uint64 `json:"token,omitempty"`
	Deadline int64  `json:"deadline,omitempty"` // Unix nanoseconds
	Result   string `json:"result,omitempty"`
}

type task struct {
	payload  string
	worker   string
	token    uint64
	deadline time.Time
	attempts int
	done     bool
	result   string
}

// Queue is the durable task queue.
type Queue struct {
	mu    sync.Mutex
	f     File
	clock Clock
	tasks map[string]*task
	order []string // enqueue order
	token uint64   // the highest token handed out
}

// Open replays the log on fs and opens it for appending.
func Open(fs FS, clock Clock) (*Queue, error) {
	q := &Queue{clock: clock, tasks: map[string]*task{}}
	b, err := fs.ReadFile(LogName)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	torn := false
	for len(b) > 0 {
		i := bytes.IndexByte(b, '\n')
		if i < 0 {
			torn = true // a write cut by the crash: its call never returned
			break
		}
		var r record
		if err := json.Unmarshal(b[:i], &r); err == nil {
			q.apply(r)
		} // else: a torn write sealed by an earlier Open; skip it, keep going
		b = b[i+1:]
	}
	if q.f, err = fs.Append(LogName); err != nil {
		return nil, err
	}
	if torn {
		// Seal the fragment with a newline, or the next record would be
		// glued to it and lost on the next replay.
		if _, err := q.f.Write([]byte("\n")); err != nil {
			return nil, err
		}
		if err := q.f.Sync(); err != nil {
			return nil, err
		}
	}
	return q, nil
}

// apply changes the in-memory state for one record.
func (q *Queue) apply(r record) {
	t := q.tasks[r.ID]
	switch r.Op {
	case "enq":
		if t == nil {
			q.tasks[r.ID] = &task{payload: r.Payload}
			q.order = append(q.order, r.ID)
		}
	case "lease", "renew":
		if t != nil {
			if r.Op == "lease" {
				t.attempts++
			}
			t.worker, t.token, t.deadline = r.Worker, r.Token, time.Unix(0, r.Deadline)
		}
	case "done":
		if t != nil {
			t.done, t.result = true, r.Result
		}
	}
	if r.Token > q.token {
		q.token = r.Token
	}
}

// write appends r to the log and syncs it; only then is r applied.
func (q *Queue) write(r record) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if _, err := q.f.Write(append(b, '\n')); err != nil {
		return err
	}
	if err := q.f.Sync(); err != nil {
		return err
	}
	q.apply(r)
	return nil
}

// Enqueue adds a task. Enqueueing an id that exists (in any state) is a no-op.
func (q *Queue) Enqueue(id, payload string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if _, ok := q.tasks[id]; ok {
		return nil
	}
	return q.write(record{Op: "enq", ID: id, Payload: payload})
}

// Lease hands the first task in enqueue order that is not done and whose
// previous lease (if any) has expired to worker, with a fresh token and a
// deadline of now + ttl. ErrNoTask when there is none.
func (q *Queue) Lease(worker string, ttl time.Duration) (Lease, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	now := q.clock.Now()
	for _, id := range q.order {
		t := q.tasks[id]
		if t.done || (t.token != 0 && now.Before(t.deadline)) {
			continue
		}
		r := record{Op: "lease", ID: id, Worker: worker, Token: q.token + 1, Deadline: now.Add(ttl).UnixNano()}
		if err := q.write(r); err != nil {
			return Lease{}, err
		}
		return Lease{Task: id, Payload: t.payload, Worker: worker, Token: t.token, Deadline: t.deadline, Attempt: t.attempts}, nil
	}
	return Lease{}, ErrNoTask
}

// Renew extends a lease to now + ttl. ErrLeaseLost when the token is not
// the task's current one, the task is done, or the lease already expired.
func (q *Queue) Renew(l Lease, ttl time.Duration) (Lease, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	t := q.tasks[l.Task]
	if t == nil {
		return Lease{}, ErrUnknown
	}
	now := q.clock.Now()
	if t.done || t.token != l.Token || !now.Before(t.deadline) {
		return Lease{}, ErrLeaseLost
	}
	if err := q.write(record{Op: "renew", ID: l.Task, Worker: l.Worker, Token: l.Token, Deadline: now.Add(ttl).UnixNano()}); err != nil {
		return Lease{}, err
	}
	l.Deadline = t.deadline
	return l, nil
}

// Complete records the result. ErrLeaseLost when the token is not the
// task's current one (another worker has it, or it is done).
func (q *Queue) Complete(l Lease, result string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	t := q.tasks[l.Task]
	if t == nil {
		return ErrUnknown
	}
	if t.done || t.token != l.Token {
		return ErrLeaseLost
	}
	return q.write(record{Op: "done", ID: l.Task, Token: l.Token, Result: result})
}

// Result is a done task's result.
func (q *Queue) Result(id string) (string, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	t := q.tasks[id]
	if t == nil || !t.done {
		return "", false
	}
	return t.result, true
}

// Pending lists the ids of tasks not done, in enqueue order.
func (q *Queue) Pending() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	var out []string
	for _, id := range q.order {
		if !q.tasks[id].done {
			out = append(out, id)
		}
	}
	return out
}

// Close closes the log.
func (q *Queue) Close() error {
	return q.f.Close()
}

// Effect applies a task's external side effect. The external system
// deduplicates by key, so applying the same key twice is harmless.
type Effect func(key, value string) error

// Worker drains a queue.
type Worker struct {
	Q     *Queue
	ID    string
	TTL   time.Duration // lease time; renewed every TTL/3 while working
	Clock Clock
	Work  func(ctx context.Context, payload string) (string, error)
	Apply Effect
}

// RunOne leases one task, works on it while renewing the lease, applies its
// effect with the task id as the idempotency key, and completes it.
// (false, nil) when there is no task. When a renewal fails the work's
// context is cancelled and RunOne returns ErrLeaseLost without applying.
func (w *Worker) RunOne(ctx context.Context) (bool, error) {
	l, err := w.Q.Lease(w.ID, w.TTL)
	if errors.Is(err, ErrNoTask) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	wctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var mu sync.Mutex
	lost := false
	stop := make(chan struct{})
	renewed := make(chan struct{})
	go func() {
		defer close(renewed)
		for {
			select {
			case <-stop:
				return
			case <-wctx.Done():
				return
			case <-w.Clock.After(w.TTL / 3):
				nl, err := w.Q.Renew(l, w.TTL)
				mu.Lock()
				if err != nil {
					lost = true
					mu.Unlock()
					cancel()
					return
				}
				l = nl
				mu.Unlock()
			}
		}
	}()
	res, werr := w.Work(wctx, l.Payload)
	close(stop)
	<-renewed
	mu.Lock()
	gone := lost
	mu.Unlock()
	if gone {
		return true, ErrLeaseLost
	}
	if werr != nil {
		return true, werr
	}
	if err := w.Apply(l.Task, res); err != nil {
		return true, err
	}
	return true, w.Q.Complete(l, res)
}
