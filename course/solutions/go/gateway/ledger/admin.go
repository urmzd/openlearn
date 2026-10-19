// The admin API (gw.07 owns the server side of openapi/admin.v1.yaml): one
// handler for every /admin/v1 path. Usage is served here from the ledger;
// keys (gw.02), workers, routes, and drains (gw.05), the cache purge (gw.06),
// and the policy (gw.08) are handlers their modules build, mounted here.

package ledger

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/server"
)

// AdminDeps are the handlers behind each admin path. A nil handler answers
// 404 not_found ("not served by this gateway"), so a gateway that has not
// built a stage yet still serves a well-formed admin API.
type AdminDeps struct {
	Usage   Ledger       // GET /admin/v1/usage (this module)
	Keys    http.Handler // /admin/v1/keys and /admin/v1/keys/{key_id} (gw.02)
	Workers http.Handler // GET /admin/v1/workers (gw.05)
	Routes  http.Handler // GET, PUT /admin/v1/routes (gw.05)
	Drain   http.Handler // POST /admin/v1/models/{model}:drain (gw.05)
	Policy  http.Handler // GET, PUT /admin/v1/policy (gw.08)
	Cache   http.Handler // POST /admin/v1/cache:purge (gw.06)
}

// Prefix is where the admin API lives on the gateway's API port.
const Prefix = "/admin/v1/"

func notFound(w http.ResponseWriter, what string) {
	server.WriteError(w, http.StatusNotFound, "invalid_request_error", "not_found", "", what)
}

func methodNotAllowed(w http.ResponseWriter, allow string) {
	w.Header().Set("Allow", allow)
	server.WriteError(w, http.StatusMethodNotAllowed, "invalid_request_error", "", "", "method not allowed; use "+allow)
}

// Admin serves every /admin/v1 path. It fails closed: a request without an
// authenticated Principal is a 401 and one without the admin scope a 403,
// even if the handler was mounted outside the authn stage by mistake.
func Admin(d AdminDeps) http.Handler {
	usage := UsageHandler(d.Usage)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p, ok := auth.PrincipalFrom(r.Context())
		if !ok {
			server.WriteError(w, http.StatusUnauthorized, "invalid_request_error", "invalid_api_key", "",
				"the admin API needs an API key with the admin scope")
			return
		}
		if !p.HasScope(auth.ScopeAdmin) {
			server.WriteError(w, http.StatusForbidden, "permission_error", "insufficient_scope", "",
				"key "+p.KeyID+" lacks the admin scope")
			return
		}
		path := r.URL.Path
		var h http.Handler
		switch {
		case path == Prefix+"usage":
			if d.Usage != nil {
				h = usage
			}
		case path == Prefix+"keys" || strings.HasPrefix(path, Prefix+"keys/"):
			h = d.Keys
		case path == Prefix+"workers":
			h = d.Workers
		case path == Prefix+"routes":
			h = d.Routes
		case strings.HasPrefix(path, Prefix+"models/") && strings.HasSuffix(path, ":drain"):
			h = d.Drain
		case path == Prefix+"policy":
			h = d.Policy
		case path == Prefix+"cache:purge":
			h = d.Cache
		default:
			notFound(w, "no admin endpoint "+path)
			return
		}
		if h == nil {
			notFound(w, path+" is not served by this gateway")
			return
		}
		h.ServeHTTP(w, r)
	})
}

// usageResponse is the 200 body of GET /admin/v1/usage.
type usageResponse struct {
	Since *string    `json:"since"`
	Until *string    `json:"until"`
	Data  []UsageRow `json:"data"`
}

func parseTime(q string) (time.Time, *string, bool) {
	if q == "" {
		return time.Time{}, nil, true
	}
	t, err := time.Parse(time.RFC3339Nano, q)
	if err != nil {
		return time.Time{}, nil, false
	}
	s := t.UTC().Format(time.RFC3339Nano)
	return t, &s, true
}

// UsageHandler serves GET /admin/v1/usage: query parameters tenant, key_id,
// since and until (RFC 3339; until exclusive), group_by (none, model,
// tenant, key_id, api_version). A bad parameter is a 400 whose `param`
// names it.
func UsageHandler(l Ledger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			methodNotAllowed(w, http.MethodGet)
			return
		}
		q := r.URL.Query()
		f := UsageFilter{Tenant: q.Get("tenant"), KeyID: q.Get("key_id"), GroupBy: q.Get("group_by")}
		if f.GroupBy == "" {
			f.GroupBy = "none"
		}
		if _, ok := groupColumns[f.GroupBy]; !ok && f.GroupBy != "none" {
			server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "group_by", ErrBadGroupBy.Error())
			return
		}
		var resp usageResponse
		var ok bool
		if f.Since, resp.Since, ok = parseTime(q.Get("since")); !ok {
			server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "since", "since must be an RFC 3339 time")
			return
		}
		if f.Until, resp.Until, ok = parseTime(q.Get("until")); !ok {
			server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "until", "until must be an RFC 3339 time")
			return
		}
		if !f.Since.IsZero() && !f.Until.IsZero() && !f.Since.Before(f.Until) {
			server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "until", "until must be later than since")
			return
		}
		rows, err := l.Query(r.Context(), f)
		if err != nil {
			server.WriteError(w, http.StatusInternalServerError, "server_error", "internal_error", "", "usage query failed")
			return
		}
		resp.Data = rows
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
}
