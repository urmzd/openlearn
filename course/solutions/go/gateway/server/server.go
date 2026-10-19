// The server lifecycle (gw.01): two listeners (the API and the health port),
// readiness, and a graceful drain on shutdown. Chain and Exchange: chain.go.

package server

import (
	"context"
	"errors"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"tinyllm/config"
)

// Config is the server's part of [gateway] in runtime.toml.
type Config struct {
	Listen        string        // API, ":8080" in Kubernetes
	HealthListen  string        // /healthz, /readyz, /metrics; ":9464"
	DrainDeadline time.Duration // how long in-flight requests may run after shutdown starts
	MaxBodyBytes  int64         // 0 = DefaultMaxBody
}

// FromRuntime takes the server's settings from a loaded runtime.toml.
func FromRuntime(g config.Gateway) Config {
	return Config{
		Listen:        g.Listen,
		HealthListen:  g.HealthListen,
		DrainDeadline: time.Duration(g.DrainDeadlineS) * time.Second,
	}
}

// Deps are the stages and seams the composition root (your go/cmd/gateway)
// builds and hands in. Every stage is optional except Proxy.
type Deps struct {
	Keys    Middleware   // authn: API keys and scopes (gw.02)
	Policy  Middleware   // usage policy (gw.08)
	Limiter Middleware   // ratelimit: RPM and TPM (gw.03)
	Cache   Middleware   // response cache (gw.06)
	Router  Middleware   // route: pick a worker (gw.05)
	Proxy   http.Handler // proxy: forward and stream (gw.00, gw.04)
	Ledger  Middleware   // meter: record usage (gw.07); wraps Proxy
	Clock   Clock        // nil = WallClock
	Tracer  Tracer       // nil = no spans
	// Ready reports whether the gateway can take traffic (gw.05: at least one
	// routable worker). nil = always ready.
	Ready func(ctx context.Context) error
	// Metrics serves GET /metrics on the health port (obs.01); nil = 404.
	Metrics http.Handler
}

// Server is the gateway process: the API server and the health server.
type Server struct {
	cfg    Config
	deps   Deps
	clock  Clock
	api    *http.Server
	health *http.Server

	draining   atomic.Bool
	baseCtx    context.Context
	cancelBase context.CancelFunc
	serveErr   chan error
	once       sync.Once
}

// New builds the server; nothing listens until Serve or Run.
func New(cfg Config, d Deps) *Server {
	if cfg.MaxBodyBytes <= 0 {
		cfg.MaxBodyBytes = DefaultMaxBody
	}
	s := &Server{cfg: cfg, deps: d, clock: d.Clock, serveErr: make(chan error, 2)}
	if s.clock == nil {
		s.clock = WallClock
	}
	s.baseCtx, s.cancelBase = context.WithCancel(context.Background())
	s.api = &http.Server{
		Addr:              cfg.Listen,
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		// Every request context descends from baseCtx, so cancelling it at the
		// drain deadline reaches every in-flight handler and its upstream call.
		BaseContext: func(net.Listener) context.Context { return s.baseCtx },
	}
	s.health = &http.Server{Addr: cfg.HealthListen, Handler: s.HealthHandler(), ReadHeaderTimeout: 10 * time.Second}
	return s
}

// HealthHandler serves the health port: /healthz is 200 while the process is
// up; /readyz is 503 from the moment shutdown starts (and while Deps.Ready
// fails), else 200; /metrics is Deps.Metrics.
func (s *Server) HealthHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		if s.draining.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("draining\n"))
			return
		}
		if s.deps.Ready != nil {
			if err := s.deps.Ready(r.Context()); err != nil {
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte("not ready: " + err.Error() + "\n"))
				return
			}
		}
		_, _ = w.Write([]byte("ready\n"))
	})
	if s.deps.Metrics != nil {
		mux.Handle("/metrics", s.deps.Metrics)
	}
	return mux
}

// Draining reports whether shutdown has started.
func (s *Server) Draining() bool {
	return s.draining.Load()
}

// Serve answers on the two listeners until Shutdown; it returns nil after a
// clean shutdown and the first serving error otherwise.
func (s *Server) Serve(api, health net.Listener) error {
	go func() { s.serveErr <- s.health.Serve(health) }()
	go func() { s.serveErr <- s.api.Serve(api) }()
	var first error
	for i := 0; i < 2; i++ {
		if err := <-s.serveErr; err != nil && !errors.Is(err, http.ErrServerClosed) && first == nil {
			first = err
		}
	}
	return first
}

// Shutdown drains: /readyz turns 503 first (so a load balancer stops sending
// new work), the API listener closes, and in-flight requests (streams
// included) run until they finish or ctx expires. At the deadline every
// remaining request context is cancelled and its connection closed, and
// Shutdown returns ctx.Err(). The health server closes last.
func (s *Server) Shutdown(ctx context.Context) error {
	s.draining.Store(true)
	err := s.api.Shutdown(ctx)
	if err != nil {
		s.cancelBase()
		_ = s.api.Close()
	}
	s.once.Do(s.cancelBase)
	hctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = s.health.Shutdown(hctx)
	return err
}

// Run serves until ctx is done (your main cancels it on SIGTERM or SIGINT),
// then shuts down with the configured drain deadline.
func (s *Server) Run(ctx context.Context, api, health net.Listener) error {
	done := make(chan error, 1)
	go func() { done <- s.Serve(api, health) }()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
	}
	dctx, cancel := context.WithTimeout(context.Background(), s.cfg.DrainDeadline)
	defer cancel()
	err := s.Shutdown(dctx)
	if serr := <-done; serr != nil && err == nil {
		err = serr
	}
	return err
}
