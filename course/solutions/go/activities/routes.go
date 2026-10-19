// routes.go (dur.12): the route-table activities of ModelRelease, a client
// of the gateway's admin API (contracts/openapi/admin.v1.yaml, served by
// gw.05 and gw.07).
//
// The route table is replaced whole by PUT /admin/v1/routes, guarded by
// optimistic concurrency: GET returns an ETag, PUT sends it back in
// If-Match, a stale ETag is 412 etag_mismatch. Every change here is a
// read-modify-write of ONE route: read the table, change that route's
// `backends`, leave every other route byte for byte as read, PUT with the
// ETag, and on 412 read again and redo the change.
//
// Every activity is idempotent by desired state: it computes the table it
// wants and does nothing when the gateway already has it. A worker that dies
// after its PUT but before the server recorded the activity's completion is
// retried, finds the change in place, and does not apply it a second time.
package activities

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
)

// Routes is a client of the gateway's admin API.
type Routes struct {
	Base         string       // the gateway base URL without /admin (for example http://127.0.0.1:30080)
	Key          string       // an API key with the admin scope
	Client       *http.Client // nil: http.DefaultClient
	MaxConflicts int          // 412 retries before giving up (the activity is then retried); 0: 5
}

// Backend is one weighted target of a route (admin.v1.yaml Backend).
type Backend struct {
	ServedModel string  `json:"served_model"`
	Weight      float64 `json:"weight"`
}

// Route activities' inputs and results.
type (
	SnapshotInput struct {
		Route string `json:"route"` // the public model id clients send
	}
	RouteSnapshot struct {
		Route json.RawMessage `json:"route"` // the route object exactly as the gateway returned it
		ETag  string          `json:"etag"`
	}
	CanaryInput struct {
		Route  string  `json:"route"`
		Served string  `json:"served"` // the model id the new engines report (WorkerStatus.model)
		Weight float64 `json:"weight"` // the canary's share, 0 < weight < 1
	}
	PromoteInput struct {
		Route  string `json:"route"`
		Served string `json:"served"`
	}
	RestoreInput struct {
		Route    string          `json:"route"`
		Previous json.RawMessage `json:"previous"` // a RouteSnapshot.Route taken before the canary
	}
	RouteResult struct {
		Changed  bool      `json:"changed"` // false: the table already said this (a retried activity)
		Backends []Backend `json:"backends"`
		ETag     string    `json:"etag"`
	}
)

var (
	// ErrStale is a 412 etag_mismatch: someone changed the table since it was read.
	ErrStale = errors.New("routes: stale ETag")
	// ErrNoRoute: the table has no route for the model (Permanent).
	ErrNoRoute = errors.New("routes: no such route")
	// ErrNoWorkers: no live worker reports the served model yet; routing
	// traffic to it would fail every request (retryable: engines come up).
	ErrNoWorkers = errors.New("routes: no ready worker serves the model")
)

// BackendsOf is a route's backends; a route without `backends` sends
// everything to one backend whose served_model is the route's model
// (admin.v1.yaml).
func BackendsOf(route map[string]any) ([]Backend, error) {
	raw, ok := route["backends"]
	if !ok || raw == nil {
		m, _ := route["model"].(string)
		return []Backend{{ServedModel: m, Weight: 1}}, nil
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}
	var out []Backend
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("routes: backends: %w", err)
	}
	return out, nil
}

// CanaryBackends gives `served` the share w and scales every other backend
// by 1 - w, so the weights still sum to 1. A backend already named `served`
// is replaced, not added twice.
func CanaryBackends(current []Backend, served string, w float64) []Backend {
	out := make([]Backend, 0, len(current)+1)
	rest := 0.0
	for _, b := range current {
		if b.ServedModel != served {
			rest += b.Weight
		}
	}
	for _, b := range current {
		if b.ServedModel == served {
			continue
		}
		share := 0.0
		if rest > 0 {
			share = b.Weight / rest * (1 - w)
		}
		out = append(out, Backend{ServedModel: b.ServedModel, Weight: round9(share)})
	}
	return append(out, Backend{ServedModel: served, Weight: round9(w)})
}

// SameBackends compares two backend lists as sets, weights to 1e-9.
func SameBackends(a, b []Backend) bool {
	if len(a) != len(b) {
		return false
	}
	want := map[string]float64{}
	for _, x := range a {
		want[x.ServedModel] = x.Weight
	}
	for _, y := range b {
		w, ok := want[y.ServedModel]
		if !ok || math.Abs(w-y.Weight) > 1e-9 {
			return false
		}
	}
	return true
}

// marshalPlain is json.Marshal without HTML escaping: encoding/json turns
// > into \u003e even inside a json.RawMessage, which would rewrite every
// route holding "accept_if": "mean_logprob >= -1.5" on each PUT.
func marshalPlain(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// round9 rounds a weight to 1e-9, so 0.9 * 1 stays 0.9 in the table.
func round9(x float64) float64 {
	return math.Round(x*1e9) / 1e9
}

// do sends one admin request with the admin key and returns the response
// and its body (at most 4 MiB).
func (r Routes) do(ctx context.Context, method, path string, body []byte, header map[string]string) (*http.Response, []byte, error) {
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, r.Base+path, rd)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+r.Key)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, v := range header {
		req.Header.Set(k, v)
	}
	c := r.Client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	return resp, b, err
}

// Get reads the route table: each route as the raw JSON object the gateway
// sent (so a PUT can return the routes it did not change unaltered) and the
// table's ETag.
func (r Routes) Get(ctx context.Context) ([]json.RawMessage, string, error) {
	resp, body, err := r.do(ctx, http.MethodGet, "/admin/v1/routes", nil, nil)
	if err != nil {
		return nil, "", err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("routes: GET: HTTP %d: %s", resp.StatusCode, promTrunc(body, 200))
	}
	var t struct {
		Routes []json.RawMessage `json:"routes"`
	}
	if err := json.Unmarshal(body, &t); err != nil {
		return nil, "", fmt.Errorf("routes: GET: %w", err)
	}
	etag := resp.Header.Get("ETag")
	if etag == "" {
		return nil, "", errors.New("routes: GET answered without an ETag")
	}
	return t.Routes, etag, nil
}

// Put replaces the table with If-Match: etag and returns the new ETag. A 412
// is ErrStale; a 400 (the gateway rejected the table) is Permanent.
func (r Routes) Put(ctx context.Context, routes []json.RawMessage, etag string) (string, error) {
	if routes == nil {
		routes = []json.RawMessage{}
	}
	body, err := marshalPlain(map[string]any{"routes": routes})
	if err != nil {
		return "", err
	}
	resp, rb, err := r.do(ctx, http.MethodPut, "/admin/v1/routes", body, map[string]string{"If-Match": etag})
	if err != nil {
		return "", err
	}
	switch {
	case resp.StatusCode == http.StatusOK:
		return resp.Header.Get("ETag"), nil
	case resp.StatusCode == http.StatusPreconditionFailed:
		return "", ErrStale
	case resp.StatusCode == http.StatusBadRequest:
		return "", Permanent{fmt.Errorf("routes: PUT rejected: %s", promTrunc(rb, 300))}
	default:
		return "", fmt.Errorf("routes: PUT: HTTP %d: %s", resp.StatusCode, promTrunc(rb, 200))
	}
}

// update is the read-modify-write loop: fn changes the one route named
// model in place and says whether it changed anything. Nothing changed: no
// PUT. A 412 reads the table again and reapplies fn, at most MaxConflicts
// times.
func (r Routes) update(ctx context.Context, model string, fn func(route map[string]any) (bool, error)) (RouteResult, error) {
	tries := r.MaxConflicts
	if tries <= 0 {
		tries = 5
	}
	for attempt := 0; ; attempt++ {
		routes, etag, err := r.Get(ctx)
		if err != nil {
			return RouteResult{}, err
		}
		idx := -1
		var route map[string]any
		for i, raw := range routes {
			var m map[string]any
			if err := json.Unmarshal(raw, &m); err != nil {
				return RouteResult{}, fmt.Errorf("routes: route %d: %w", i, err)
			}
			if m["model"] == model {
				idx, route = i, m
				break
			}
		}
		if idx < 0 {
			return RouteResult{}, Permanent{fmt.Errorf("%w: %q", ErrNoRoute, model)}
		}
		changed, err := fn(route)
		if err != nil {
			return RouteResult{}, err
		}
		backends, err := BackendsOf(route)
		if err != nil {
			return RouteResult{}, err
		}
		if !changed {
			return RouteResult{Changed: false, Backends: backends, ETag: etag}, nil
		}
		b, err := marshalPlain(route)
		if err != nil {
			return RouteResult{}, err
		}
		next := append([]json.RawMessage(nil), routes...)
		next[idx] = b
		tag, err := r.Put(ctx, next, etag)
		if errors.Is(err, ErrStale) && attempt+1 < tries {
			continue
		}
		if err != nil {
			return RouteResult{}, err
		}
		return RouteResult{Changed: true, Backends: backends, ETag: tag}, nil
	}
}

// Snapshot returns one route as the gateway holds it now: what Restore puts
// back if the canary fails.
func (r Routes) Snapshot(ctx context.Context, in SnapshotInput) (RouteSnapshot, error) {
	routes, etag, err := r.Get(ctx)
	if err != nil {
		return RouteSnapshot{}, err
	}
	for _, raw := range routes {
		var m struct {
			Model string `json:"model"`
		}
		if json.Unmarshal(raw, &m) == nil && m.Model == in.Route {
			return RouteSnapshot{Route: raw, ETag: etag}, nil
		}
	}
	return RouteSnapshot{}, Permanent{fmt.Errorf("%w: %q", ErrNoRoute, in.Route)}
}

// Ready says whether a live (not draining) worker reports `served`
// (GET /admin/v1/workers).
func (r Routes) Ready(ctx context.Context, served string) (bool, error) {
	resp, body, err := r.do(ctx, http.MethodGet, "/admin/v1/workers", nil, nil)
	if err != nil {
		return false, err
	}
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("routes: GET workers: HTTP %d: %s", resp.StatusCode, promTrunc(body, 200))
	}
	var w struct {
		Data []struct {
			Model    string `json:"model"`
			Draining bool   `json:"draining"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &w); err != nil {
		return false, fmt.Errorf("routes: GET workers: %w", err)
	}
	for _, x := range w.Data {
		if x.Model == served && !x.Draining {
			return true, nil
		}
	}
	return false, nil
}

// Canary sends the share in.Weight of in.Route's traffic to in.Served (the
// "routes.canary" activity). It refuses (ErrNoWorkers, retryable) while no
// worker serves the new model, and does nothing when the canary is already
// in place.
func (r Routes) Canary(ctx context.Context, in CanaryInput) (RouteResult, error) {
	if !(in.Weight > 0 && in.Weight < 1) {
		return RouteResult{}, Permanent{fmt.Errorf("routes: canary weight %v is not in (0, 1)", in.Weight)}
	}
	ok, err := r.Ready(ctx, in.Served)
	if err != nil {
		return RouteResult{}, err
	}
	if !ok {
		return RouteResult{}, fmt.Errorf("%w: %q", ErrNoWorkers, in.Served)
	}
	return r.update(ctx, in.Route, func(route map[string]any) (bool, error) {
		cur, err := BackendsOf(route)
		if err != nil {
			return false, err
		}
		for _, b := range cur {
			if b.ServedModel == in.Served && math.Abs(b.Weight-in.Weight) <= 1e-9 {
				return false, nil // already applied by an earlier attempt
			}
		}
		route["backends"] = CanaryBackends(cur, in.Served, in.Weight)
		return true, nil
	})
}

// Promote sends all of in.Route's traffic to in.Served ("routes.promote").
func (r Routes) Promote(ctx context.Context, in PromoteInput) (RouteResult, error) {
	want := []Backend{{ServedModel: in.Served, Weight: 1}}
	return r.update(ctx, in.Route, func(route map[string]any) (bool, error) {
		cur, err := BackendsOf(route)
		if err != nil {
			return false, err
		}
		if SameBackends(cur, want) {
			return false, nil
		}
		route["backends"] = want
		return true, nil
	})
}

// Restore puts in.Previous back as in.Route's entry ("routes.restore"): the
// rollback. Every other route keeps what it has now.
func (r Routes) Restore(ctx context.Context, in RestoreInput) (RouteResult, error) {
	var prev map[string]any
	if err := json.Unmarshal(in.Previous, &prev); err != nil || prev["model"] != in.Route {
		return RouteResult{}, Permanent{fmt.Errorf("routes: the previous route is not a route for %q", in.Route)}
	}
	return r.update(ctx, in.Route, func(route map[string]any) (bool, error) {
		a, _ := json.Marshal(route)
		b, _ := json.Marshal(prev)
		if bytes.Equal(a, b) {
			return false, nil
		}
		for k := range route {
			delete(route, k)
		}
		for k, v := range prev {
			route[k] = v
		}
		return true, nil
	})
}
