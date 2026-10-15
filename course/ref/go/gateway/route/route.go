// Package route is the gateway's routing layer (gw.05): the worker registry
// (registry.go), the route table and the routing policies (this file), and
// the chain stages that route, fail over before the first byte, run
// cascades, and orchestrate disaggregated prefill and decode (handler.go).
//
// Contract: proto/tl/control/v1/control.proto (served),
// proto/tl/engine/v1/engine.proto and proto/tl/kv/v1/kv.proto (called),
// openapi/admin.v1.yaml (Route, Worker), config/runtime.schema.json
// ([gateway] route_policy, affinity_load_factor, heartbeat_miss_limit,
// routes). Chapter: content/tracks/ai-platform-engineering/12-gateway/05-routing.md.
package route

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"

	"tinyllm/config"
	"tinyllm/ds/ring"
)

// Policies of [gateway].route_policy.
const (
	PolicyWeighted         = "weighted"
	PolicyLeastOutstanding = "least_outstanding"
	PolicyAffinity         = "affinity"
)

// Errors of Route.
var (
	ErrModelNotFound = errors.New("model not found")              // 404 model_not_found
	ErrNoCapacity    = errors.New("no routable worker for model") // 503 no_capacity
)

// InferenceRequest is what routing needs from a request.
type InferenceRequest struct {
	Model     string // the public model id the client sent
	RequestID string // picks the canary backend deterministically
	PrefixKey []byte // affinity key: the start of the prompt (PrefixKey)
	Stream    bool
}

// Exclusions are worker ids a retry must avoid.
type Exclusions map[string]bool

// Target is where one attempt goes.
type Target struct {
	Worker      Worker  // the unified or decode worker
	Prefill     *Worker // the prefill worker of a disaggregated route
	ServedModel string  // the model id the workers report (sent upstream)
	Reason      string  // affinity | load | canary | cascade
	CascadeStep int     // index into the route's cascade, -1 without one
}

// Router picks workers. Safe for concurrent use.
type Router struct {
	reg    *Registry
	policy string
	c      float64

	mu       sync.Mutex
	table    map[string]config.Route // by model and alias
	routes   []config.Route
	epoch    uint64
	inflight map[string]int
	ring     *ring.Ring
	ringSet  string         // the members the ring was built from
	smooth   map[string]int // smooth weighted round robin state
}

// NewRouter routes over reg's workers with policy and the bounded-load
// factor c (affinity_load_factor, at least 1).
func NewRouter(reg *Registry, policy string, c float64, routes []config.Route) *Router {
	// SOLUTION-BEGIN gw.05
	if c < 1 {
		c = 1
	}
	rt := &Router{reg: reg, policy: policy, c: c, inflight: map[string]int{}, smooth: map[string]int{}}
	rt.SetRoutes(routes)
	return rt
	// SOLUTION-END
}

// SetRoutes replaces the route table (PUT /admin/v1/routes) and returns its
// new epoch.
func (rt *Router) SetRoutes(routes []config.Route) uint64 {
	// SOLUTION-BEGIN gw.05
	rt.mu.Lock()
	defer rt.mu.Unlock()
	rt.routes = append([]config.Route(nil), routes...)
	rt.table = map[string]config.Route{}
	for _, r := range routes {
		rt.table[r.Model] = r
		for _, a := range r.Aliases {
			rt.table[a] = r
		}
	}
	rt.epoch++
	if rt.reg != nil {
		rt.reg.SetEpoch(rt.epoch)
	}
	return rt.epoch
	// SOLUTION-END
}

// Routes returns the table and its epoch.
func (rt *Router) Routes() ([]config.Route, uint64) {
	// SOLUTION-BEGIN gw.05
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return append([]config.Route(nil), rt.routes...), rt.epoch
	// SOLUTION-END
}

// Lookup resolves a public model id (or alias) to its route. A model with
// no route is its own route (one backend, the same id), so a gateway without
// a table routes to whatever the workers serve.
func (rt *Router) Lookup(model string) config.Route {
	// SOLUTION-BEGIN gw.05
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if r, ok := rt.table[model]; ok {
		return r
	}
	return config.Route{Model: model}
	// SOLUTION-END
}

// Acquire and Release count a worker's in-flight requests (the "outstanding"
// of least_outstanding and the load of affinity).
func (rt *Router) Acquire(id string) {
	// SOLUTION-BEGIN gw.05
	rt.mu.Lock()
	rt.inflight[id]++
	rt.mu.Unlock()
	// SOLUTION-END
}

func (rt *Router) Release(id string) {
	// SOLUTION-BEGIN gw.05
	rt.mu.Lock()
	if rt.inflight[id] > 0 {
		rt.inflight[id]--
	}
	rt.mu.Unlock()
	// SOLUTION-END
}

// Inflight is a worker's in-flight count.
func (rt *Router) Inflight(id string) int {
	// SOLUTION-BEGIN gw.05
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.inflight[id]
	// SOLUTION-END
}

// Unit maps s to [0, 1) with the ring's finalized hash (ds.09 Hash64, the
// top 53 bits): the deterministic draw that picks a canary backend, so one
// request id always lands on the same backend.
func Unit(s string) float64 {
	// SOLUTION-BEGIN gw.05
	return float64(ring.Hash64([]byte(s))>>11) / float64(uint64(1)<<53)
	// SOLUTION-END
}

// PickBackend chooses the served model for a request: backends split by
// weight (a canary), drawn with Unit(requestID); no backends means the
// route's own model.
func PickBackend(r config.Route, requestID string) string {
	// SOLUTION-BEGIN gw.05
	if len(r.Backends) == 0 {
		return r.Model
	}
	total := 0.0
	for _, b := range r.Backends {
		total += b.Weight
	}
	u := Unit(requestID) * total
	acc := 0.0
	for _, b := range r.Backends {
		acc += b.Weight
		if u < acc {
			return b.ServedModel
		}
	}
	return r.Backends[len(r.Backends)-1].ServedModel
	// SOLUTION-END
}

// PrefixKeyBytes is how much of the prompt the affinity key covers: about
// one KV block (16 tokens of about 4 bytes), the block the engine's prefix
// cache hashes first, so requests that share it share cached KV.
const PrefixKeyBytes = 64

// PrefixKey is the affinity key of a request body: the first PrefixKeyBytes
// of the completions prompt, or of the chat messages rendered as
// "role\ncontent\n" in order.
func PrefixKey(body []byte) []byte {
	// SOLUTION-BEGIN gw.05
	var f struct {
		Prompt   *string `json:"prompt"`
		Messages []struct {
			Role    string `json:"role"`
			Content any    `json:"content"`
		} `json:"messages"`
	}
	_ = json.Unmarshal(body, &f)
	var sb strings.Builder
	if f.Prompt != nil {
		sb.WriteString(*f.Prompt)
	}
	for _, m := range f.Messages {
		sb.WriteString(m.Role)
		sb.WriteByte('\n')
		if s, ok := m.Content.(string); ok {
			sb.WriteString(s)
		}
		sb.WriteByte('\n')
		if sb.Len() >= PrefixKeyBytes {
			break
		}
	}
	b := []byte(sb.String())
	if len(b) > PrefixKeyBytes {
		b = b[:PrefixKeyBytes]
	}
	return b
	// SOLUTION-END
}

// load is a worker's routing load: gateway in-flight plus engine queue.
func (rt *Router) load(w Worker) int {
	// SOLUTION-BEGIN gw.05
	return rt.inflight[w.ID] + w.QueueDepth
	// SOLUTION-END
}

// pick applies the policy to candidates (non-empty, sorted by id). Callers
// hold rt.mu.
func (rt *Router) pick(cands []Worker, key []byte) (Worker, string) {
	// SOLUTION-BEGIN gw.05
	switch rt.policy {
	case PolicyAffinity:
		ids := make([]string, len(cands))
		byID := map[string]Worker{}
		for i, w := range cands {
			ids[i] = w.ID
			byID[w.ID] = w
		}
		set := strings.Join(ids, ",")
		if rt.ring == nil || rt.ringSet != set {
			rt.ring = ring.New(160, nil)
			for _, id := range ids {
				rt.ring.Add(id)
			}
			rt.ringSet = set
		}
		owner, _ := rt.ring.Get(key)
		id, _ := rt.ring.GetBounded(key, func(n string) int { return rt.load(byID[n]) }, rt.c)
		if id != owner {
			return byID[id], "load"
		}
		return byID[id], "affinity"
	case PolicyWeighted:
		// Smooth weighted round robin (nginx): weight = KV capacity.
		total, best := 0, -1
		for i, w := range cands {
			wt := max(w.KVTotalBlocks, 1)
			rt.smooth[w.ID] += wt
			total += wt
			if best < 0 || rt.smooth[w.ID] > rt.smooth[cands[best].ID] {
				best = i
			}
		}
		rt.smooth[cands[best].ID] -= total
		return cands[best], "load"
	default: // least_outstanding
		best := 0
		for i, w := range cands {
			if rt.load(w) < rt.load(cands[best]) {
				best = i
			}
		}
		return cands[best], "load"
	}
	// SOLUTION-END
}

// candidates are the routable workers serving model in role, minus ex,
// sorted by id.
func candidates(ws []Worker, model, role string, ex Exclusions) []Worker {
	// SOLUTION-BEGIN gw.05
	var out []Worker
	for _, w := range ws {
		if w.Model == model && w.Role == role && !ex[w.ID] {
			out = append(out, w)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
	// SOLUTION-END
}

// Route picks the target for one attempt of req, avoiding ex. It returns
// ErrModelNotFound when no route names the model and no worker serves it,
// and ErrNoCapacity when the model exists but no routable worker is left.
// A disaggregated route ("targets": ["disaggregated"]) picks a prefill
// worker (least loaded) and a decode worker (by the policy). step selects
// the cascade step (0 without a cascade).
func (rt *Router) Route(ctx context.Context, req *InferenceRequest, ex Exclusions, step int) (Target, error) {
	// SOLUTION-BEGIN gw.05
	r := rt.Lookup(req.Model)
	served := PickBackend(r, req.RequestID)
	reason := ""
	if len(r.Backends) > 1 {
		reason = "canary"
	}
	cstep := -1
	if len(r.Cascade) > 0 {
		step = min(max(step, 0), len(r.Cascade)-1)
		served, reason, cstep = r.Cascade[step].Model, "cascade", step
	}
	ws := rt.reg.Routable()
	known := false
	for _, w := range ws {
		if w.Model == served {
			known = true
		}
	}
	rt.mu.Lock()
	_, routed := rt.table[req.Model]
	rt.mu.Unlock()
	if !known && !routed {
		return Target{}, fmt.Errorf("%w: %s", ErrModelNotFound, req.Model)
	}
	disagg := len(r.Targets) > 0 && r.Targets[0] == "disaggregated"
	role := "unified"
	if disagg {
		role = "decode"
	}
	cands := candidates(ws, served, role, ex)
	if len(cands) == 0 {
		return Target{}, fmt.Errorf("%w %s (%s)", ErrNoCapacity, served, role)
	}
	rt.mu.Lock()
	defer rt.mu.Unlock()
	w, why := rt.pick(cands, req.PrefixKey)
	if reason == "" {
		reason = why
	}
	t := Target{Worker: w, ServedModel: served, Reason: reason, CascadeStep: cstep}
	if disagg {
		ps := candidates(ws, served, "prefill", ex)
		if len(ps) == 0 {
			return Target{}, fmt.Errorf("%w %s (prefill)", ErrNoCapacity, served)
		}
		best := 0
		for i, p := range ps {
			if rt.load(p) < rt.load(ps[best]) {
				best = i
			}
		}
		p := ps[best]
		t.Prefill = &p
	}
	return t, nil
	// SOLUTION-END
}

// Ready is the gateway's readiness: at least one routable worker.
func (rt *Router) Ready(ctx context.Context) error {
	// SOLUTION-BEGIN gw.05
	if len(rt.reg.Routable()) == 0 {
		return errors.New("no routable worker")
	}
	return nil
	// SOLUTION-END
}

// Accept is a cascade step's acceptance test over the step's token
// logprobs.
type Accept func(logprobs []float64) bool

// ParseAcceptIf parses `<metric> <op> <number>` (runtime.schema.json
// cascadeStep.accept_if): metric mean_logprob or min_logprob, op one of
// > >= < <=. An answer with no tokens is never accepted.
func ParseAcceptIf(s string) (Accept, error) {
	// SOLUTION-BEGIN gw.05
	f := strings.Fields(s)
	if len(f) != 3 {
		return nil, fmt.Errorf("accept_if %q: want `<metric> <op> <number>`", s)
	}
	tau, err := strconv.ParseFloat(f[2], 64)
	if err != nil {
		return nil, fmt.Errorf("accept_if %q: %v", s, err)
	}
	var metric func([]float64) float64
	switch f[0] {
	case "mean_logprob":
		metric = func(lp []float64) float64 {
			s := 0.0
			for _, x := range lp {
				s += x
			}
			return s / float64(len(lp))
		}
	case "min_logprob":
		metric = func(lp []float64) float64 {
			m := math.Inf(1)
			for _, x := range lp {
				m = math.Min(m, x)
			}
			return m
		}
	default:
		return nil, fmt.Errorf("accept_if %q: metric must be mean_logprob or min_logprob", s)
	}
	var cmp func(a, b float64) bool
	switch f[1] {
	case ">":
		cmp = func(a, b float64) bool { return a > b }
	case ">=":
		cmp = func(a, b float64) bool { return a >= b }
	case "<":
		cmp = func(a, b float64) bool { return a < b }
	case "<=":
		cmp = func(a, b float64) bool { return a <= b }
	default:
		return nil, fmt.Errorf("accept_if %q: op must be >, >=, <, or <=", s)
	}
	return func(lp []float64) bool {
		return len(lp) > 0 && cmp(metric(lp), tau)
	}, nil
	// SOLUTION-END
}

// LogprobsOf returns the token logprobs of a non-streamed response: chat
// choices[0].logprobs.content[].logprob, or completions
// choices[0].logprobs.token_logprobs.
func LogprobsOf(body []byte) ([]float64, error) {
	// SOLUTION-BEGIN gw.05
	var v struct {
		Choices []struct {
			Logprobs *struct {
				Content []struct {
					Logprob float64 `json:"logprob"`
				} `json:"content"`
				TokenLogprobs []float64 `json:"token_logprobs"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &v); err != nil {
		return nil, err
	}
	if len(v.Choices) == 0 || v.Choices[0].Logprobs == nil {
		return nil, errors.New("the response has no logprobs")
	}
	lp := v.Choices[0].Logprobs
	if lp.Content != nil {
		out := make([]float64, len(lp.Content))
		for i, c := range lp.Content {
			out[i] = c.Logprob
		}
		return out, nil
	}
	return lp.TokenLogprobs, nil
	// SOLUTION-END
}

// CascadeSample is one evaluation prompt of a two-step cascade: the small
// model's token logprobs and whether each model answered correctly.
type CascadeSample struct {
	SmallLogprobs []float64 `json:"small_logprobs"`
	SmallCorrect  bool      `json:"small_correct"`
	LargeCorrect  bool      `json:"large_correct"`
}

// CurvePoint is the cascade's outcome at one threshold: the fraction of
// prompts escalated to the large model, the cost relative to always using
// the large model (small costs smallCost, large 1, an escalated prompt
// pays both), and the accuracy.
type CurvePoint struct {
	Tau       float64 `json:"tau"`
	Escalated float64 `json:"escalated"`
	Cost      float64 `json:"cost"`
	Accuracy  float64 `json:"accuracy"`
}

// CascadeCurve sweeps the threshold of `mean_logprob >= tau` over taus.
func CascadeCurve(samples []CascadeSample, taus []float64, smallCost float64) []CurvePoint {
	// SOLUTION-BEGIN gw.05
	out := make([]CurvePoint, 0, len(taus))
	n := float64(len(samples))
	for _, tau := range taus {
		accept, _ := ParseAcceptIf("mean_logprob >= " + strconv.FormatFloat(tau, 'f', -1, 64))
		esc, cost, ok := 0.0, 0.0, 0.0
		for _, s := range samples {
			cost += smallCost
			if accept(s.SmallLogprobs) {
				if s.SmallCorrect {
					ok++
				}
				continue
			}
			esc++
			cost++
			if s.LargeCorrect {
				ok++
			}
		}
		out = append(out, CurvePoint{Tau: tau, Escalated: esc / n, Cost: cost / n, Accuracy: ok / n})
	}
	return out
	// SOLUTION-END
}
