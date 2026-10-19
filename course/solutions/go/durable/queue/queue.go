// Package queue is the durable server's task queue (dur.03): named queues
// of tasks that workers lease with a visibility timeout, fenced lease tokens
// so a late worker cannot complete a task someone else now owns, retries
// with a delay, a dead-letter queue after MaxAttempts, redrive, and a long
// poll that wakes as soon as a task becomes visible.
//
// Every transition (enqueue, lease, heartbeat, retry, dead letter, redrive,
// ack, drop) is appended to the dur.01 log on stream "queue/<name>" before
// it takes effect, so Open rebuilds the queues exactly after a crash. A lease
// that expires needs no record: an expired lease is visible again (or dead,
// when it was the last attempt) by definition, before and after a restart.
package queue

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	dlog "tinyllm/durable/log"
)

// Clock is the time seam (DESIGN 5.11). The testkit's *clock.Fake satisfies it.
type Clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

type wallClock struct{}

func (wallClock) Now() time.Time                         { return time.Now() }
func (wallClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// Task is the unit of work. ID is unique within its queue.
type Task struct {
	ID          string
	Payload     []byte
	MaxAttempts int           // deliveries before the dead-letter queue; 0 = Options.MaxAttempts
	Visibility  time.Duration // lease length; 0 = Options.Visibility
}

// Lease is one delivery of a task. Token fences it: it grows with every
// delivery, and only the current lease, before its Deadline, may heartbeat,
// complete, or fail the task.
type Lease struct {
	Queue    string
	Task     Task
	Token    uint64
	Attempt  int       // 1 for the first delivery
	Deadline time.Time // the task becomes visible again at this instant
	Details  []byte    // the last heartbeat details, kept across attempts
}

// TaskError reports a failed attempt.
type TaskError struct {
	Message      string
	NonRetryable bool          // the task leaves the queue now (Dropped)
	RetryAfter   time.Duration // a retry becomes visible after this delay
}

// Disposition is what Fail did with the task.
type Disposition int

const (
	Retrying     Disposition = iota + 1 // visible again after RetryAfter
	DeadLettered                        // attempts exhausted: parked in the DLQ
	Dropped                             // non-retryable: removed
)

// DeadLetter is a task parked in a queue's dead-letter queue.
type DeadLetter struct {
	Task      Task
	Attempts  int
	LastError string
	At        time.Time
}

var (
	ErrNoTask    = errors.New("queue: no task within the wait")
	ErrLeaseLost = errors.New("queue: lease lost (stale token or past its deadline)")
	ErrNotFound  = errors.New("queue: no such dead-lettered task")
)

// Options configures a Queue.
type Options struct {
	Log         *dlog.Log     // nil keeps everything in memory
	Clock       Clock         // nil = the wall clock
	Visibility  time.Duration // default lease length; 0 = 30 s
	MaxAttempts int           // default deliveries before the DLQ; 0 = 5
}

type state int

const (
	visible state = iota
	leased
	dead
)

type item struct {
	task      Task
	state     state
	attempt   int // deliveries so far
	seq       uint64
	visibleAt time.Time
	token     uint64
	deadline  time.Time
	details   []byte
	lastErr   string
	deadAt    time.Time
}

type named struct {
	items map[string]*item
	wake  chan struct{} // closed and replaced whenever a task may have become pollable
}

// Queue is safe for concurrent use.
type Queue struct {
	mu     sync.Mutex
	o      Options
	queues map[string]*named
	tokens uint64 // the last lease token handed out, over every queue
	seq    uint64 // enqueue order, for FIFO among equally visible tasks
}

// transition is one persisted record (stream "queue/<name>", JSON data).
type transition struct {
	Kind     string `json:"k"` // enq lease hb retry dead redrive ack drop
	ID       string `json:"id"`
	Payload  []byte `json:"payload,omitempty"`
	Max      int    `json:"max,omitempty"`
	VisMs    int64  `json:"vis_ms,omitempty"`
	Token    uint64 `json:"tok,omitempty"`
	Attempt  int    `json:"att,omitempty"`
	UntilMs  int64  `json:"until_ms,omitempty"` // lease deadline, or when a retry is visible
	Details  []byte `json:"details,omitempty"`
	Error    string `json:"err,omitempty"`
	AtUnixMs int64  `json:"at_ms,omitempty"`
}

func ms(t time.Time) int64        { return t.UnixMilli() }
func fromMs(v int64) time.Time    { return time.UnixMilli(v) }
func streamOf(name string) string { return "queue/" + name }

// Open rebuilds every queue from the log's "queue/" streams.
func Open(ctx context.Context, o Options) (*Queue, error) {
	if o.Clock == nil {
		o.Clock = wallClock{}
	}
	if o.Visibility <= 0 {
		o.Visibility = 30 * time.Second
	}
	if o.MaxAttempts <= 0 {
		o.MaxAttempts = 5
	}
	q := &Queue{o: o, queues: map[string]*named{}}
	if o.Log == nil {
		return q, nil
	}
	for _, s := range o.Log.Streams("queue/") {
		evs, err := o.Log.Read(ctx, s, 1, 0)
		if err != nil {
			return nil, err
		}
		name := strings.TrimPrefix(s, "queue/")
		for _, e := range evs {
			var tr transition
			if err := json.Unmarshal(e.Data, &tr); err != nil {
				return nil, fmt.Errorf("queue %s event %d: %w", name, e.Version, err)
			}
			q.apply(name, tr)
		}
	}
	return q, nil
}

// apply changes memory to match one transition; persisting comes first.
func (q *Queue) apply(name string, tr transition) {
	n := q.named(name)
	it := n.items[tr.ID]
	switch tr.Kind {
	case "enq":
		q.seq++
		n.items[tr.ID] = &item{
			task:      Task{ID: tr.ID, Payload: tr.Payload, MaxAttempts: tr.Max, Visibility: time.Duration(tr.VisMs) * time.Millisecond},
			seq:       q.seq,
			visibleAt: fromMs(tr.AtUnixMs),
		}
	case "lease":
		it.state, it.token, it.attempt, it.deadline = leased, tr.Token, tr.Attempt, fromMs(tr.UntilMs)
		if tr.Token > q.tokens { // recovery: never reuse a token after a restart
			q.tokens = tr.Token
		}
	case "hb":
		it.deadline, it.details = fromMs(tr.UntilMs), tr.Details
	case "retry":
		it.state, it.visibleAt, it.lastErr = visible, fromMs(tr.UntilMs), tr.Error
	case "dead":
		it.state, it.lastErr, it.deadAt = dead, tr.Error, fromMs(tr.AtUnixMs)
	case "redrive":
		it.state, it.attempt, it.visibleAt = visible, 0, fromMs(tr.AtUnixMs)
	case "ack", "drop":
		delete(n.items, tr.ID)
	}
	q.notify(n)
}

func (q *Queue) named(name string) *named {
	n := q.queues[name]
	if n == nil {
		n = &named{items: map[string]*item{}, wake: make(chan struct{})}
		q.queues[name] = n
	}
	return n
}

func (q *Queue) notify(n *named) {
	close(n.wake)
	n.wake = make(chan struct{})
}

// persist appends tr to the log (when there is one), then applies it.
func (q *Queue) persist(ctx context.Context, name string, tr transition) error {
	if q.o.Log != nil {
		data, err := json.Marshal(tr)
		if err != nil {
			return err
		}
		if _, err := q.o.Log.Append(ctx, streamOf(name), dlog.Any, dlog.Event{Type: tr.Kind, Data: data}); err != nil {
			return err
		}
	}
	q.apply(name, tr)
	return nil
}

func (q *Queue) maxAttempts(it *item) int {
	if it.task.MaxAttempts > 0 {
		return it.task.MaxAttempts
	}
	return q.o.MaxAttempts
}

func (q *Queue) visibility(it *item) time.Duration {
	if it.task.Visibility > 0 {
		return it.task.Visibility
	}
	return q.o.Visibility
}

// expire turns every lease whose deadline has passed back into a visible
// task, or into a dead letter when that lease was the last allowed attempt.
// It needs no log record: the lease record already says when it ends.
func (q *Queue) expire(n *named, now time.Time) {
	for _, it := range n.items {
		if it.state != leased || now.Before(it.deadline) {
			continue
		}
		if it.attempt >= q.maxAttempts(it) {
			it.state, it.deadAt, it.lastErr = dead, it.deadline, "lease expired"
		} else {
			it.state, it.visibleAt = visible, it.deadline
		}
	}
}

// Enqueue adds t to queue name, visible now. An ID already in the queue
// (visible, leased, or dead-lettered) is a no-op, so a caller may repeat an
// enqueue after a crash without creating a duplicate.
func (q *Queue) Enqueue(ctx context.Context, name string, t Task) error {
	if t.ID == "" {
		return errors.New("queue: empty task id")
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if _, ok := q.named(name).items[t.ID]; ok {
		return nil
	}
	return q.persist(ctx, name, transition{
		Kind: "enq", ID: t.ID, Payload: t.Payload, Max: t.MaxAttempts,
		VisMs: t.Visibility.Milliseconds(), AtUnixMs: ms(q.o.Clock.Now()),
	})
}

// Poll leases the visible task that became visible first (ties in enqueue
// order). With nothing visible it waits up to wait for an enqueue, a retry
// coming due, or an expiring lease; then ErrNoTask. wait 0 does not block.
func (q *Queue) Poll(ctx context.Context, name, worker string, wait time.Duration) (Lease, error) {
	end := q.o.Clock.Now().Add(wait)
	for {
		q.mu.Lock()
		now := q.o.Clock.Now()
		n := q.named(name)
		q.expire(n, now)
		var best *item
		next := end
		for _, it := range n.items {
			switch it.state {
			case visible:
				if !it.visibleAt.After(now) {
					if best == nil || it.visibleAt.Before(best.visibleAt) ||
						(it.visibleAt.Equal(best.visibleAt) && it.seq < best.seq) {
						best = it
					}
				} else if it.visibleAt.Before(next) {
					next = it.visibleAt
				}
			case leased:
				if it.deadline.Before(next) {
					next = it.deadline
				}
			}
		}
		if best != nil {
			q.tokens++ // a fresh fencing token, larger than any handed out
			tr := transition{
				Kind: "lease", ID: best.task.ID, Token: q.tokens, Attempt: best.attempt + 1,
				UntilMs: ms(now.Add(q.visibility(best))),
			}
			err := q.persist(ctx, name, tr)
			l := q.leaseOf(name, best)
			q.mu.Unlock()
			if err != nil {
				return Lease{}, err
			}
			return l, nil
		}
		if !now.Before(end) {
			q.mu.Unlock()
			return Lease{}, ErrNoTask
		}
		wake := n.wake
		q.mu.Unlock()
		select {
		case <-wake:
		case <-q.o.Clock.After(next.Sub(now)):
		case <-ctx.Done():
			return Lease{}, ctx.Err()
		}
	}
}

func (q *Queue) leaseOf(name string, it *item) Lease {
	return Lease{
		Queue: name, Task: it.task, Token: it.token, Attempt: it.attempt,
		Deadline: it.deadline, Details: it.details,
	}
}

// live returns l's item when l is still the task's lease: same token, and
// the clock is before its deadline.
func (q *Queue) live(l Lease) (*named, *item, error) {
	n := q.queues[l.Queue]
	if n == nil {
		return nil, nil, ErrLeaseLost
	}
	it := n.items[l.Task.ID]
	if it == nil || it.state != leased || it.token != l.Token || !q.o.Clock.Now().Before(it.deadline) {
		return nil, nil, ErrLeaseLost
	}
	return n, it, nil
}

// Check reports whether l is still the task's live lease (nil) or not
// (ErrLeaseLost), without changing anything.
func (q *Queue) Check(l Lease) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	_, _, err := q.live(l)
	return err
}

// Heartbeat extends a live lease by the task's visibility and stores
// details; it returns the extended lease.
func (q *Queue) Heartbeat(ctx context.Context, l Lease, details []byte) (Lease, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	_, it, err := q.live(l)
	if err != nil {
		return Lease{}, err
	}
	until := q.o.Clock.Now().Add(q.visibility(it))
	if err := q.persist(ctx, l.Queue, transition{Kind: "hb", ID: it.task.ID, Token: it.token, UntilMs: ms(until), Details: append([]byte(nil), details...)}); err != nil {
		return Lease{}, err
	}
	return q.leaseOf(l.Queue, it), nil
}

// Complete acknowledges a live lease: the task leaves the queue. A stale or
// expired lease is ErrLeaseLost and changes nothing.
func (q *Queue) Complete(ctx context.Context, l Lease) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if _, _, err := q.live(l); err != nil {
		return err
	}
	return q.persist(ctx, l.Queue, transition{Kind: "ack", ID: l.Task.ID, Token: l.Token})
}

// Fail ends a live lease with an error: a non-retryable error drops the
// task; the attempt that reaches MaxAttempts dead-letters it; any other is
// retried, visible again after e.RetryAfter.
func (q *Queue) Fail(ctx context.Context, l Lease, e TaskError) (Disposition, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	_, it, err := q.live(l)
	if err != nil {
		return 0, err
	}
	now := q.o.Clock.Now()
	switch {
	case e.NonRetryable:
		return Dropped, q.persist(ctx, l.Queue, transition{Kind: "drop", ID: it.task.ID, Error: e.Message})
	case it.attempt >= q.maxAttempts(it):
		return DeadLettered, q.persist(ctx, l.Queue, transition{Kind: "dead", ID: it.task.ID, Error: e.Message, AtUnixMs: ms(now)})
	default:
		return Retrying, q.persist(ctx, l.Queue, transition{Kind: "retry", ID: it.task.ID, Error: e.Message, UntilMs: ms(now.Add(e.RetryAfter))})
	}
}

// DLQ lists queue name's dead letters in the order they were enqueued.
func (q *Queue) DLQ(ctx context.Context, name string) ([]DeadLetter, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	n := q.named(name)
	q.expire(n, q.o.Clock.Now())
	var out []DeadLetter
	var seqs []uint64
	for _, it := range n.items {
		if it.state == dead {
			out = append(out, DeadLetter{Task: it.task, Attempts: it.attempt, LastError: it.lastErr, At: it.deadAt})
			seqs = append(seqs, it.seq)
		}
	}
	for i := 1; i < len(out); i++ { // insertion sort by enqueue order
		for j := i; j > 0 && seqs[j] < seqs[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
			seqs[j], seqs[j-1] = seqs[j-1], seqs[j]
		}
	}
	return out, nil
}

// Redrive moves dead letters back to the queue, visible now with a fresh
// attempt count; no ids means every dead letter of the queue. An id that is
// not dead-lettered is ErrNotFound and nothing moves. It returns how many moved.
func (q *Queue) Redrive(ctx context.Context, name string, ids ...string) (int, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	n := q.named(name)
	now := q.o.Clock.Now()
	q.expire(n, now)
	if len(ids) == 0 {
		for id, it := range n.items {
			if it.state == dead {
				ids = append(ids, id)
			}
		}
	}
	for _, id := range ids {
		if it := n.items[id]; it == nil || it.state != dead {
			return 0, fmt.Errorf("%w: %s/%s", ErrNotFound, name, id)
		}
	}
	for _, id := range ids {
		if err := q.persist(ctx, name, transition{Kind: "redrive", ID: id, AtUnixMs: ms(now)}); err != nil {
			return 0, err
		}
	}
	return len(ids), nil
}

// Depth counts queue name's tasks by state (expired leases count as visible).
func (q *Queue) Depth(name string) (visibleN, leasedN, deadN int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	n := q.named(name)
	q.expire(n, q.o.Clock.Now())
	for _, it := range n.items {
		switch it.state {
		case visible:
			visibleN++
		case leased:
			leasedN++
		case dead:
			deadN++
		}
	}
	return
}
