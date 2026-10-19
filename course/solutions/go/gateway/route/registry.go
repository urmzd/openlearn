// The worker registry (gw.05): engines report themselves with tl.control.v1
// heartbeats every 2 s; a worker that misses heartbeat_miss_limit of them in
// a row is evicted from routing.

package route

import (
	"context"
	"sort"
	"sync"
	"time"

	controlv1 "openlearn.urmzd.com/tl/contracts/gen/tl/control/v1"
	"tinyllm/gateway/server"
)

// HeartbeatInterval is how often engines report (control.proto).
const HeartbeatInterval = 2 * time.Second

// Worker is one engine as its last heartbeat described it.
type Worker struct {
	ID            string
	HTTPAddress   string // host:port of the OpenAI HTTP surface
	GRPCAddress   string // host:port of tl.engine.v1.EngineControl
	KVAddress     string // host:port of tl.kv.v1.KvTransferService (decode role)
	Role          string // unified | prefill | decode
	Model         string
	KVFormat      uint32
	QueueDepth    int
	Running       int
	KVFreeBlocks  int
	KVTotalBlocks int
	Draining      bool
	LastHeartbeat time.Time
}

// Registry is the gateway's view of its workers; it serves
// tl.control.v1.WorkerRegistry. Safe for concurrent use.
type Registry struct {
	controlv1.UnimplementedWorkerRegistryServer

	clock     server.Clock
	missLimit int
	interval  time.Duration

	mu      sync.Mutex
	workers map[string]*Worker
	drain   map[string]bool // models asked to drain (admin :drain)
	epoch   uint64
}

// NewRegistry evicts a worker once missLimit heartbeat intervals pass
// without one ([gateway].heartbeat_miss_limit, default 3).
func NewRegistry(clock server.Clock, missLimit int) *Registry {
	if clock == nil {
		clock = server.WallClock
	}
	if missLimit < 1 {
		missLimit = 3
	}
	return &Registry{clock: clock, missLimit: missLimit, interval: HeartbeatInterval,
		workers: map[string]*Worker{}, drain: map[string]bool{}}
}

// Heartbeat records a worker's status (the gRPC method). The ack asks the
// worker to drain when its model is draining.
func (r *Registry) Heartbeat(ctx context.Context, s *controlv1.WorkerStatus) (*controlv1.HeartbeatAck, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.workers[s.GetWorkerId()] = &Worker{
		ID: s.GetWorkerId(), HTTPAddress: s.GetHttpAddress(), GRPCAddress: s.GetGrpcAddress(),
		KVAddress: s.GetKvAddress(), Role: s.GetRole(), Model: s.GetModel(), KVFormat: s.GetKvFormat(),
		QueueDepth: int(s.GetQueueDepth()), Running: int(s.GetRunning()), KVFreeBlocks: int(s.GetKvFreeBlocks()),
		KVTotalBlocks: int(s.GetKvTotalBlocks()), Draining: s.GetDraining(), LastHeartbeat: r.clock.Now(),
	}
	return &controlv1.HeartbeatAck{Drain: r.drain[s.GetModel()], RouteEpoch: r.epoch}, nil
}

// live reports whether w heartbeated within missLimit intervals of now.
func (r *Registry) live(w *Worker, now time.Time) bool {
	return now.Sub(w.LastHeartbeat) < time.Duration(r.missLimit)*r.interval
}

// Snapshot evicts the workers that missed missLimit heartbeats and returns
// the rest, sorted by id (the admin /admin/v1/workers view).
func (r *Registry) Snapshot() []Worker {
	now := r.clock.Now()
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]Worker, 0, len(r.workers))
	for id, w := range r.workers {
		if !r.live(w, now) {
			delete(r.workers, id)
			continue
		}
		out = append(out, *w)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Routable is the live workers that may take new work: not draining, and
// not serving a model the gateway is draining.
func (r *Registry) Routable() []Worker {
	all := r.Snapshot()
	r.mu.Lock()
	defer r.mu.Unlock()
	out := all[:0]
	for _, w := range all {
		if !w.Draining && !r.drain[w.Model] {
			out = append(out, w)
		}
	}
	return out
}

// Drain stops routing new work to model's workers and asks them, in their
// next heartbeat ack, to drain.
func (r *Registry) Drain(model string) {
	r.mu.Lock()
	r.drain[model] = true
	r.mu.Unlock()
}

// SetEpoch records the route table's epoch for heartbeat acks.
func (r *Registry) SetEpoch(e uint64) {
	r.mu.Lock()
	r.epoch = e
	r.mu.Unlock()
}
