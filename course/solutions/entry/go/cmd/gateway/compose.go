// The --config form of the gateway (gw.01 to gw.07, spec/cli-roles.md): the
// composition root that reads runtime.toml, builds one value per stage, and
// hands them to go/gateway (Stages, Deps, Admin), which orders the chain.
//
//	gateway --config <runtime.toml> [--api-key-env TL_API_KEY]
//
// What this file adds around the library:
//   - keys: the key in $TL_API_KEY ([endpoints].api_key_env) is accepted with
//     the scopes infer and admin, next to the keys of [gateway].keys_file
//     (a gw.02 store, opened when the variable named by pepper_env is set).
//     Its limits come from [ext] env_key_rpm and env_key_tpm (0 = none).
//   - workers: engines that heartbeat to [gateway].registry_listen join the
//     registry (gw.05); [gateway].upstream, when set, is also heartbeated on
//     the engine's behalf from its GET /v1/models, so a local engine that
//     does not know the gateway's registry address is still routable.
//   - the RPM and TPM limits (gw.03) apply to the inference routes only.
//   - the meter wraps the cache stage instead of the proxy alone, so a cache
//     hit is a ledger row (cache_hit = 1) with the usage it replayed: the
//     ledger then matches what every client was answered (MS-gateway).
//   - the admin API (/admin/v1/...) shares the API port behind its own authn.
//   - the usage policy (gw.08, policy.go) runs after authn when
//     [gateway].policy_file is set.
//
// Telemetry is obs.02's (telemetry.go): the SERVER span, /metrics on the
// health port, JSON logs. SIGTERM turns /readyz to 503, drains in-flight
// requests for [gateway].drain_deadline_s, and exits 0.
package main

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"google.golang.org/grpc"

	controlv1 "openlearn.urmzd.com/tl/contracts/gen/tl/control/v1"
	"tinyllm/config"
	"tinyllm/gateway"
	"tinyllm/gateway/auth"
	"tinyllm/gateway/cache"
	"tinyllm/gateway/ledger"
	"tinyllm/gateway/limit"
	"tinyllm/gateway/proxy"
	"tinyllm/gateway/route"
	"tinyllm/gateway/server"
)

// envKeys accepts the environment's key, then the key store's keys.
type envKeys struct {
	key   string
	p     auth.Principal
	store *auth.Store // nil without keys_file or pepper
}

func (k envKeys) Lookup(ctx context.Context, presented string) (auth.Principal, error) {
	if k.key != "" && subtle.ConstantTimeCompare([]byte(presented), []byte(k.key)) == 1 {
		return k.p, nil
	}
	if k.store != nil {
		return k.store.Lookup(ctx, presented)
	}
	return auth.Principal{}, auth.ErrInvalidKey
}

// extInt reads an integer from [ext]; absent means def.
func extInt(c config.Config, key string, def int) (int, error) {
	v, ok := c.Ext[key]
	if !ok {
		return def, nil
	}
	n, ok := v.(int64)
	if !ok || n < 0 {
		return 0, fmt.Errorf("[ext].%s = %v, want a non-negative integer", key, v)
	}
	return int(n), nil
}

// runConfig is the --config form; it returns the exit code.
func runConfig(path, keyEnv string) int {
	cfg, err := config.Load(path, os.Environ())
	if err != nil {
		log.Printf("gateway: %v", err)
		return 1
	}
	g := cfg.Gateway
	apiConfig, err := apiVersionConfig(cfg)
	if err != nil {
		log.Printf("gateway: %v", err)
		return 1
	}
	rpm, err := extInt(cfg, "env_key_rpm", 0)
	if err != nil {
		log.Printf("gateway: %v", err)
		return 1
	}
	tpm, err := extInt(cfg, "env_key_tpm", 0)
	if err != nil {
		log.Printf("gateway: %v", err)
		return 1
	}

	tel, err := newObsTelemetry()
	if err != nil {
		log.Printf("gateway: telemetry: %v", err)
		return 1
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		tel.shutdown(ctx)
	}()

	keys := envKeys{key: os.Getenv(keyEnv), p: auth.Principal{
		KeyID: "env", Tenant: "default", Name: keyEnv,
		Scopes: []string{auth.ScopeInfer, auth.ScopeAdmin}, RPM: rpm, TPM: tpm,
	}}
	if keys.key == "" {
		log.Printf("gateway: %s is empty, so only keys_file keys are accepted", keyEnv)
	}
	if g.KeysFile != "" {
		if pepper := os.Getenv(g.PepperEnv); pepper == "" {
			log.Printf("gateway: %s is empty, so keys_file %s is not opened", g.PepperEnv, g.KeysFile)
		} else {
			if err := os.MkdirAll(filepath.Dir(g.KeysFile), 0o755); err != nil {
				log.Printf("gateway: %v", err)
				return 1
			}
			keys.store, err = auth.NewStore(g.KeysFile, []byte(pepper),
				auth.Options{CacheTTL: time.Duration(g.KeyCacheTTLS) * time.Second})
			if err != nil {
				log.Printf("gateway: %v", err)
				return 1
			}
		}
	}

	reg := route.NewRegistry(nil, g.HeartbeatMissLimit)
	rt := route.NewRouter(reg, g.RoutePolicy, g.AffinityLoadFactor, g.Routes)
	st := gateway.Stages{
		Keys:         keys.store,
		Limiter:      limit.New(nil),
		CacheOptions: cache.Options{TTL: time.Duration(g.CacheTTLS) * time.Second},
		Registry:     reg,
		Router:       rt,
		Proxy:        route.Proxy(rt, route.Options{OnSpan: routeSpanExporter()}),
	}
	if g.CacheEntries > 0 {
		st.Cache = cache.New(g.CacheEntries, nil)
	}
	if st.Policy, err = policyStage(cfg); err != nil {
		log.Printf("gateway: policy_file: %v", err)
		return 1
	}
	if g.UsageDB != "" {
		if err := os.MkdirAll(filepath.Dir(g.UsageDB), 0o755); err != nil {
			log.Printf("gateway: %v", err)
			return 1
		}
		db, err := ledger.Open(g.UsageDB)
		if err != nil {
			log.Printf("gateway: usage_db: %v", err)
			return 1
		}
		defer db.Close()
		st.Ledger = db
	}

	// The inference chain: go/gateway's Deps with the env-aware authn, and
	// the meter moved out from around the proxy to around the cache.
	chain := st
	chain.Ledger = nil
	deps := gateway.Deps(chain)
	authn := auth.Middleware(keys, slog.Default())
	deps.Keys = authn
	// Select the version after authentication but before rate limits, the
	// response cache, and metering. Admin routes retain their own contract.
	policy := deps.Policy
	deps.Policy = func(next http.Handler) http.Handler {
		if policy != nil {
			next = policy(next)
		}
		return server.VersionedAPI(apiConfig, next)
	}
	if lim := deps.Limiter; lim != nil {
		// Budgets are spent by inference calls only: a GET /v1/models (or a
		// stray path) neither costs a request nor waits behind a 429.
		deps.Limiter = func(next http.Handler) http.Handler {
			limited := lim(next)
			return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost && ledger.MeteredRoutes[r.URL.Path] {
					limited.ServeHTTP(w, r)
					return
				}
				next.ServeHTTP(w, r)
			})
		}
	}
	if st.Ledger != nil {
		meter := ledger.Meter(st.Ledger, st.Meter)
		inner := deps.Cache
		deps.Cache = func(next http.Handler) http.Handler {
			if inner != nil {
				next = inner(next)
			}
			return meter(next)
		}
	}
	deps.Ready = rt.Ready
	deps.Metrics = tel.withMetrics(http.NotFoundHandler(), g.Upstream)
	srv := server.New(server.FromRuntime(g), deps)
	admin := server.New(server.Config{}, server.Deps{Keys: authn, Proxy: ledger.Admin(gateway.Admin(st))}).Handler()

	mux := http.NewServeMux()
	mux.Handle("/admin/", admin)
	mux.Handle("/", srv.Handler())
	api := &http.Server{Addr: g.Listen, Handler: tel.wrap(mux), ReadHeaderTimeout: 10 * time.Second}
	health := &http.Server{Addr: g.HealthListen, Handler: srv.HealthHandler(), ReadHeaderTimeout: 5 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()
	errs := make(chan error, 3)
	for _, s := range []*http.Server{api, health} {
		go func(s *http.Server) {
			if err := s.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				errs <- err
			}
		}(s)
	}
	var gs *grpc.Server
	if g.RegistryListen != "" {
		lis, err := net.Listen("tcp", g.RegistryListen)
		if err != nil {
			log.Printf("gateway: registry_listen: %v", err)
			return 1
		}
		gs = grpc.NewServer()
		controlv1.RegisterWorkerRegistryServer(gs, reg)
		go func() {
			if err := gs.Serve(lis); err != nil {
				errs <- err
			}
		}()
	}
	if g.Upstream != "" {
		go heartbeatUpstream(ctx, reg, g.Upstream)
	}
	log.Printf("gateway: API on %s, health on %s, registry on %s, upstream %q", g.Listen, g.HealthListen, g.RegistryListen, g.Upstream)

	code := 0
	select {
	case err := <-errs:
		log.Printf("gateway: %v", err)
		code = 1
	case <-ctx.Done():
	}
	// /readyz turns 503 first, then in-flight requests (streams included)
	// get the drain deadline.
	dctx, cancel := context.WithTimeout(context.Background(), time.Duration(g.DrainDeadlineS)*time.Second)
	defer cancel()
	_ = srv.Shutdown(dctx)
	if err := api.Shutdown(dctx); err != nil {
		_ = api.Close()
	}
	if gs != nil {
		gs.Stop()
	}
	hctx, hcancel := context.WithTimeout(context.Background(), time.Second)
	defer hcancel()
	_ = health.Shutdown(hctx)
	return code
}

// heartbeatUpstream reports the engine at base (its URL without /v1) to the
// registry every heartbeat interval, one unified worker per model its
// GET /v1/models lists. An engine that stops answering stops being reported
// and is evicted after heartbeat_miss_limit intervals.
func heartbeatUpstream(ctx context.Context, reg *route.Registry, base string) {
	u, err := url.Parse(base)
	if err != nil || u.Host == "" {
		log.Printf("gateway: upstream %q is not a URL; not heartbeating it", base)
		return
	}
	client := &http.Client{Timeout: time.Second}
	beat := func() {
		resp, err := client.Get(base + "/v1/models")
		if err != nil {
			return
		}
		defer resp.Body.Close()
		var list struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&list) != nil {
			return
		}
		for _, m := range list.Data {
			_, _ = reg.Heartbeat(ctx, &controlv1.WorkerStatus{
				WorkerId: "upstream/" + m.ID, Role: "unified", Model: m.ID, HttpAddress: u.Host,
				KvFormat: 1, KvTotalBlocks: 1, KvFreeBlocks: 1,
			})
		}
	}
	beat()
	tick := time.NewTicker(route.HeartbeatInterval)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
			beat()
		}
	}
}

// routeSpanExporter adapts newSpanExporter (otlp.go, obs.00) to the route
// stage's span hook, so the --config form exports the same gateway.proxy
// span as the tracer form. nil when no OTLP endpoint is configured.
func routeSpanExporter() func(route.SpanRecord) {
	export := newSpanExporter()
	if export == nil {
		return nil
	}
	return func(r route.SpanRecord) {
		export(proxy.Span{
			Name: proxy.SpanName, TraceID: r.TraceID, SpanID: r.SpanID, ParentSpanID: r.ParentSpanID,
			RequestID: r.RequestID, StatusCode: r.Status, Start: r.Start, End: r.End,
		})
	}
}
