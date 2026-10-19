// Package gateway assembles the gateway from its stages (gw.07). Your
// go/cmd/gateway reads the config, opens the files, builds one value per
// stage, and listens; everything between "the stages exist" and "a
// handler serves them" is here, where a test can call it: Deps is the
// inference chain's server.Deps (gw.01 orders it), Admin is the admin
// API's AdminDeps, every path served by the handler its module builds.
//
// One wiring decision lives only here: the cache keys answers by the route
// table's epoch, so a PUT /admin/v1/routes makes every answer cached under
// the old table unreachable.
//
// Chapter: content/tracks/ai-platform-engineering/12-gateway/07-usage-ledger-and-admin-api.md.
package gateway

import (
	"log/slog"
	"net/http"

	"tinyllm/gateway/auth"
	"tinyllm/gateway/cache"
	"tinyllm/gateway/ledger"
	"tinyllm/gateway/limit"
	"tinyllm/gateway/route"
	"tinyllm/gateway/server"
)

// Stages are the built stages. A nil stage is left out of the chain, and
// its admin paths answer 404 not_found.
type Stages struct {
	Keys        *auth.Store       // gw.02: authn and /admin/v1/keys
	Logger      *slog.Logger      // authn's log; nil = slog.Default()
	Policy      server.Middleware // gw.08: the usage policy stage
	PolicyAdmin http.Handler      // gw.08: /admin/v1/policy
	Limiter     *limit.Limiter    // gw.03: RPM and TPM per key
	Counter     limit.Counter     // gw.03: exact prompt tokens; nil = bytes / 4
	Cache       *cache.LRU        // gw.06: the response cache and /admin/v1/cache:purge
	// CacheOptions configure the cache stage. A nil Rev with a Router is
	// the route table's ETag.
	CacheOptions cache.Options
	Registry     *route.Registry // gw.05: /admin/v1/workers and the drains
	Router       *route.Router   // gw.05: the route stage and /admin/v1/routes
	Proxy        http.Handler    // the terminal stage (route.Proxy, or gw.04's proxy)
	Ledger       ledger.Ledger   // gw.07: the meter and /admin/v1/usage
	Meter        ledger.MeterOptions
}

// Deps is the inference chain: the stages as server.Deps middleware.
func Deps(s Stages) server.Deps {
	d := server.Deps{Policy: s.Policy, Proxy: s.Proxy}
	if s.Keys != nil {
		d.Keys = auth.Middleware(s.Keys, s.Logger)
	}
	if s.Limiter != nil {
		d.Limiter = limit.Middleware(s.Limiter, s.Counter)
	}
	if s.Cache != nil {
		o := s.CacheOptions
		if o.Rev == nil && s.Router != nil {
			rt := s.Router
			o.Rev = func() string {
				_, epoch := rt.Routes()
				return route.ETag(epoch)
			}
		}
		d.Cache = cache.Middleware(s.Cache, o)
	}
	if s.Router != nil {
		d.Router = route.Middleware(s.Router)
	}
	if s.Ledger != nil {
		d.Ledger = ledger.Meter(s.Ledger, s.Meter)
	}
	return d
}

// Admin is the admin API's handlers, built by the modules that own them;
// serve it with ledger.Admin behind authn.
func Admin(s Stages) ledger.AdminDeps {
	a := ledger.AdminDeps{Usage: s.Ledger, Policy: s.PolicyAdmin}
	if s.Keys != nil {
		a.Keys = auth.KeysHandler(s.Keys)
	}
	if s.Registry != nil {
		a.Drain = route.DrainHandler(s.Registry)
		if s.Router != nil {
			a.Workers = route.WorkersHandler(s.Registry, s.Router)
		}
	}
	if s.Router != nil {
		a.Routes = route.RoutesHandler(s.Router)
	}
	if s.Cache != nil {
		a.Cache = cache.PurgeHandler(s.Cache)
	}
	return a
}
