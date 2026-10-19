// The keys part of the admin API (gw.02): POST, GET /admin/v1/keys and
// DELETE /admin/v1/keys/{key_id} (openapi/admin.v1.yaml). gw.07 mounts this
// handler behind its admin scope check.

package auth

import (
	"encoding/json"
	"errors"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"time"

	"tinyllm/gateway/server"
)

// KeyInfo is a record as the admin API shows it: no HMAC, no secret.
type KeyInfo struct {
	KeyID     string     `json:"key_id"`
	Tenant    string     `json:"tenant"`
	Name      string     `json:"name"`
	Scopes    []string   `json:"scopes"`
	RPM       int        `json:"rpm"`
	TPM       int        `json:"tpm"`
	Priority  int        `json:"priority"`
	Models    []string   `json:"models"`
	CreatedAt time.Time  `json:"created_at"`
	ExpiresAt *time.Time `json:"expires_at"`
	Revoked   bool       `json:"revoked"`
}

// Info is r without its HMAC.
func (r Record) Info() KeyInfo {
	models := r.Models
	if models == nil {
		models = []string{}
	}
	return KeyInfo{KeyID: r.KeyID, Tenant: r.Tenant, Name: r.Name, Scopes: r.Scopes, RPM: r.RPM, TPM: r.TPM,
		Priority: r.Priority, Models: models, CreatedAt: r.CreatedAt, ExpiresAt: r.ExpiresAt, Revoked: r.Revoked}
}

var tenantRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,62}$`)

// KeysHandler serves the keys admin paths over s.
func KeysHandler(s *Store) http.Handler {
	bad := func(w http.ResponseWriter, msg string) {
		server.WriteError(w, http.StatusBadRequest, "invalid_request_error", "", "", msg)
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id, one := strings.CutPrefix(r.URL.Path, "/admin/v1/keys/")
		switch {
		case !one && r.Method == http.MethodPost:
			var in struct {
				Tenant    string     `json:"tenant"`
				Name      string     `json:"name"`
				Scopes    []string   `json:"scopes"`
				RPM       *int       `json:"rpm"`
				TPM       *int       `json:"tpm"`
				Priority  int        `json:"priority"`
				Models    []string   `json:"models"`
				ExpiresAt *time.Time `json:"expires_at"`
			}
			if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
				bad(w, "malformed JSON: "+err.Error())
				return
			}
			if !tenantRE.MatchString(in.Tenant) || in.Name == "" || len(in.Name) > 128 || len(in.Scopes) == 0 {
				bad(w, "tenant, name, and at least one scope are required (tenant: ^[a-z0-9][a-z0-9-]{0,62}$)")
				return
			}
			for _, sc := range in.Scopes {
				if sc != ScopeInfer && sc != ScopeEmbed && sc != ScopeAdmin && sc != ScopeDebug {
					bad(w, "unknown scope "+sc)
					return
				}
			}
			spec := KeySpec{Tenant: in.Tenant, Name: in.Name, Scopes: in.Scopes, RPM: 60, TPM: 100000,
				Priority: in.Priority, Models: in.Models, ExpiresAt: in.ExpiresAt}
			if in.RPM != nil {
				spec.RPM = *in.RPM
			}
			if in.TPM != nil {
				spec.TPM = *in.TPM
			}
			if spec.RPM < 0 || spec.TPM < 0 || spec.Priority < -100 || spec.Priority > 100 {
				bad(w, "rpm and tpm are at least 0; priority is -100 to 100")
				return
			}
			plain, rec, err := s.Create(r.Context(), spec)
			if err != nil {
				server.WriteError(w, http.StatusInternalServerError, "server_error", "internal_error", "", err.Error())
				return
			}
			out := struct {
				KeyInfo
				Key string `json:"key"`
			}{rec.Info(), plain}
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(out)
		case !one && r.Method == http.MethodGet:
			recs, err := s.List(r.Context())
			if err != nil {
				server.WriteError(w, http.StatusInternalServerError, "server_error", "internal_error", "", err.Error())
				return
			}
			tenant := r.URL.Query().Get("tenant")
			data := []KeyInfo{}
			for _, rec := range recs {
				if tenant == "" || rec.Tenant == tenant {
					data = append(data, rec.Info())
				}
			}
			sort.SliceStable(data, func(i, j int) bool { return data[i].CreatedAt.After(data[j].CreatedAt) })
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
		case one && r.Method == http.MethodDelete:
			if err := s.Revoke(r.Context(), id); errors.Is(err, ErrNotFound) {
				server.WriteError(w, http.StatusNotFound, "invalid_request_error", "not_found", "", "no key "+id)
				return
			} else if err != nil {
				server.WriteError(w, http.StatusInternalServerError, "server_error", "internal_error", "", err.Error())
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			server.WriteError(w, http.StatusMethodNotAllowed, "invalid_request_error", "", "", "method not allowed")
		}
	})
}
