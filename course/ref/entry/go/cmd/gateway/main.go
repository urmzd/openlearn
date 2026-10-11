// Command gateway is the reference entry point for the `gateway` role
// (course/contracts/spec/cli-roles.md), used only by course CI. Entry points
// are learner territory (DESIGN D16): the learner writes their own
// go/cmd/gateway; this one shows what the milestone runner expects.
//
//	gateway --port <n> --health-port <n> --upstream <engine base URL>   (tracer form, gw.00)
//	gateway --config <runtime.toml>                                     (gw.01 on; compose.go)
//
// The key is read from the environment variable named by --api-key-env
// (default TL_API_KEY, the `[endpoints].api_key_env` default). Spans go to
// OTEL_EXPORTER_OTLP_ENDPOINT through proxy.Config.OnSpan (otlp.go, obs.00);
// telemetry.go adds the SERVER span, /metrics on the health port, and JSON
// logs with trace_id and span_id (obs.02).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"tinyllm/gateway/proxy"
)

func main() {
	port := flag.Int("port", 8080, "API port")
	healthPort := flag.Int("health-port", 9464, "health port: /healthz and /readyz")
	upstream := flag.String("upstream", "", "engine base URL without /v1, e.g. http://127.0.0.1:8081")
	keyEnv := flag.String("api-key-env", "TL_API_KEY", "environment variable holding the API key")
	configPath := flag.String("config", "", "runtime.toml: the --config form (compose.go)")
	flag.Parse()
	if *configPath != "" && *upstream == "" && flag.NArg() == 0 {
		os.Exit(runConfig(*configPath, *keyEnv))
	}
	if *upstream == "" || *configPath != "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: gateway --port N --health-port N --upstream URL | gateway --config FILE")
		os.Exit(2)
	}
	key := os.Getenv(*keyEnv)
	if key == "" {
		log.Printf("gateway: %s is empty, so every request is rejected with 401", *keyEnv)
	}
	// obs.02: SERVER span, metrics.yaml instruments on /metrics, JSON logs
	// with trace_id and span_id. Spans leave through a batch processor, so a
	// hanging collector never blocks a request.
	tel, err := newObsTelemetry()
	if err != nil {
		log.Fatalf("gateway: telemetry: %v", err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		tel.shutdown(ctx)
	}()

	api := &http.Server{
		Addr:              fmt.Sprintf(":%d", *port),
		Handler:           tel.wrap(proxy.NewProxy(proxy.Config{Upstream: *upstream, APIKey: key, OnSpan: newSpanExporter()})),
		ReadHeaderTimeout: 10 * time.Second,
	}
	health := &http.Server{
		Addr:              fmt.Sprintf(":%d", *healthPort),
		Handler:           tel.withMetrics(healthMux(strings.TrimRight(*upstream, "/")), *upstream),
		ReadHeaderTimeout: 5 * time.Second,
	}

	errs := make(chan error, 2)
	for _, s := range []*http.Server{api, health} {
		go func(s *http.Server) {
			if err := s.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				errs <- err
			}
		}(s)
	}
	log.Printf("gateway: API on :%d, health on :%d, upstream %s", *port, *healthPort, *upstream)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, os.Interrupt)
	select {
	case err := <-errs:
		log.Fatalf("gateway: %v", err)
	case <-stop:
	}
	// Exit 0 on SIGTERM (cli-roles.md): let in-flight streams finish briefly.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_ = api.Shutdown(ctx)
	_ = health.Shutdown(ctx)
}

func healthMux(upstream string) http.Handler {
	client := &http.Client{Timeout: time.Second}
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		resp, err := client.Get(upstream + "/healthz")
		if err != nil {
			http.Error(w, "upstream unreachable: "+err.Error(), http.StatusServiceUnavailable)
			return
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			http.Error(w, fmt.Sprintf("upstream /healthz answered %d", resp.StatusCode), http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintln(w, "ready")
	})
	return mux
}
