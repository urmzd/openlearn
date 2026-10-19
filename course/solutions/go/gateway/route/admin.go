// The routing part of the admin API (gw.05): GET /admin/v1/workers,
// GET and PUT /admin/v1/routes (ETag and If-Match), and
// POST /admin/v1/models/{model}:drain (openapi/admin.v1.yaml). gw.07 mounts
// these handlers behind its admin scope check.

package route

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"tinyllm/config"
	"tinyllm/gateway/server"
)

type workerJSON struct {
	WorkerID      string `json:"worker_id"`
	Role          string `json:"role"`
	Model         string `json:"model"`
	HTTPAddress   string `json:"http_address"`
	GRPCAddress   string `json:"grpc_address"`
	KVAddress     string `json:"kv_address"`
	KVFormat      uint32 `json:"kv_format"`
	QueueDepth    int    `json:"queue_depth"`
	Running       int    `json:"running"`
	KVFreeBlocks  int    `json:"kv_free_blocks"`
	KVTotalBlocks int    `json:"kv_total_blocks"`
	Draining      bool   `json:"draining"`
	LastHeartbeat int64  `json:"last_heartbeat_unix_ms"`
}

// WorkersHandler serves GET /admin/v1/workers.
func WorkersHandler(reg *Registry, rt *Router) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := []workerJSON{}
		for _, wk := range reg.Snapshot() {
			data = append(data, workerJSON{wk.ID, wk.Role, wk.Model, wk.HTTPAddress, wk.GRPCAddress, wk.KVAddress,
				max(wk.KVFormat, 1), wk.QueueDepth, wk.Running, wk.KVFreeBlocks, wk.KVTotalBlocks, wk.Draining,
				wk.LastHeartbeat.UnixMilli()})
		}
		_, epoch := rt.Routes()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data, "route_epoch": epoch})
	})
}

// ETag is the route table's strong ETag: its epoch, quoted.
func ETag(epoch uint64) string {
	return `"routes-` + strconv.FormatUint(epoch, 10) + `"`
}

// ValidateRoutes applies the [[gateway.routes]] rules to a table.
func ValidateRoutes(routes []config.Route) error {
	c := config.Default()
	c.Gateway.Routes = routes
	if err := c.Validate(); err != nil {
		return err
	}
	for _, r := range routes {
		for i, st := range r.Cascade {
			last := i == len(r.Cascade)-1
			if (st.AcceptIf == "") != last {
				return fmt.Errorf("route %s: every cascade step but the last needs accept_if, the last has none", r.Model)
			}
			if !last {
				if _, err := ParseAcceptIf(st.AcceptIf); err != nil {
					return err
				}
			}
		}
		sum := 0.0
		for _, b := range r.Backends {
			sum += b.Weight
		}
		if len(r.Backends) > 0 && (sum < 0.999 || sum > 1.001) {
			return fmt.Errorf("route %s: backend weights must sum to 1", r.Model)
		}
	}
	return nil
}

// RoutesHandler serves GET and PUT /admin/v1/routes with optimistic
// concurrency: PUT needs If-Match equal to the current ETag (412
// etag_mismatch when stale, 428 if_match_required when missing).
func RoutesHandler(rt *Router) http.Handler {
	write := func(w http.ResponseWriter, routes []config.Route, epoch uint64) {
		if routes == nil {
			routes = []config.Route{}
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("ETag", ETag(epoch))
		_ = json.NewEncoder(w).Encode(map[string]any{"route_epoch": epoch, "routes": routes})
	}
	var mu sync.Mutex // one PUT at a time: check the ETag and replace atomically
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			routes, epoch := rt.Routes()
			write(w, routes, epoch)
		case http.MethodPut:
			mu.Lock()
			defer mu.Unlock()
			_, epoch := rt.Routes()
			im := r.Header.Get("If-Match")
			if im == "" {
				server.WriteError(w, http.StatusPreconditionRequired, "invalid_request_error", "if_match_required", "", "send If-Match with the ETag of GET /admin/v1/routes")
				return
			}
			if im != ETag(epoch) {
				server.WriteError(w, http.StatusPreconditionFailed, "invalid_request_error", "etag_mismatch", "", "the route table changed since "+im)
				return
			}
			var in struct {
				Routes []config.Route `json:"routes"`
			}
			if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
				server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", "malformed JSON: "+err.Error())
				return
			}
			if err := ValidateRoutes(in.Routes); err != nil {
				server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "routes", err.Error())
				return
			}
			epoch = rt.SetRoutes(in.Routes)
			write(w, in.Routes, epoch)
		default:
			server.WriteError(w, http.StatusMethodNotAllowed, "invalid_request_error", "", "", "use GET or PUT")
		}
	})
}

// DrainHandler serves POST /admin/v1/models/{model}:drain: 202 with the
// number of workers told to drain, 404 when no worker serves the model.
func DrainHandler(reg *Registry) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rest := strings.TrimPrefix(r.URL.Path, "/admin/v1/models/")
		model, ok := strings.CutSuffix(rest, ":drain")
		if !ok || model == "" || r.Method != http.MethodPost {
			server.WriteError(w, http.StatusNotFound, "invalid_request_error", "not_found", "", "POST /admin/v1/models/{model}:drain")
			return
		}
		var in struct {
			DeadlineS *int `json:"deadline_s"`
		}
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil || in.DeadlineS == nil || *in.DeadlineS < 0 || *in.DeadlineS > 3600 {
			server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "deadline_s", "deadline_s (0 to 3600) is required")
			return
		}
		n := 0
		for _, wk := range reg.Snapshot() {
			if wk.Model == model {
				n++
			}
		}
		if n == 0 {
			server.WriteError(w, http.StatusNotFound, "invalid_request_error", "not_found", "", "no worker serves "+model)
			return
		}
		reg.Drain(model)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]any{"model": model, "workers": n})
	})
}
