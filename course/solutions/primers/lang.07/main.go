// The lang.07 primer server: one JSON greeting and a health endpoint.
//
// GET /healthz  -> 200 "ok"
// GET /         -> 200 {"greeting": $GREETING, "pod": <hostname>}
//
// It exits 0 on SIGTERM after draining in-flight requests, which is what
// `docker stop` and a Kubernetes rollout send first.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	greeting := os.Getenv("GREETING")
	if greeting == "" {
		greeting = "hello"
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	pod, _ := os.Hostname()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"greeting": greeting, "pod": pod})
	})

	srv := &http.Server{Addr: ":" + port, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()

	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	log.Printf("listening on :%s", port)

	select {
	case err := <-errc:
		log.Fatal(err) // could not bind
	case <-ctx.Done():
	}
	shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdown); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
