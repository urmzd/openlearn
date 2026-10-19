// Command durable is the reference entry point for the `durable` role
// (course/contracts/spec/cli-roles.md): the durable execution server of Pass
// 8, the composition root of the dur.01 log, the dur.03 queue, the dur.07
// timers, and the dur.02 server. It is wiring only (flags, config, signals);
// the logic lives in the library packages. Its flags are what the dep.06
// chart passes.
//
//	durable --data <dir> --port <n> [--health-port <n>] [--test-clock]
//	durable --config <runtime.toml>
//
// --config reads [durable] (contracts/config/runtime.schema.json) with the
// TL_DURABLE__<KEY> overrides; the flags, when given, win over it. The WAL
// lives in <data>/wal (or [durable].wal_dir). The health port serves
// /healthz, /readyz, /metrics (tl_durable_task_queue_depth, which KEDA scales
// the workers on), and, with --test-clock, /debug/clock. --replicas 3 (Raft,
// dur.10) is not wired in this reference: any value but 1 exits 2.
//
// SIGTERM stops accepting rpcs, lets in-flight ones (long polls included)
// finish for up to 10 s, closes the log, and exits 0.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"tinyllm/config"
	dlog "tinyllm/durable/log"
	"tinyllm/durable/queue"
	"tinyllm/durable/server"
	"tinyllm/durable/timer"

	durablev1 "openlearn.urmzd.com/tl/contracts/gen/tl/durable/v1"
)

type wall struct{}

func (wall) Now() time.Time                         { return time.Now() }
func (wall) After(d time.Duration) <-chan time.Time { return time.After(d) }

func usage(msg string) {
	fmt.Fprintf(os.Stderr, "durable: %s\nusage: durable --data DIR --port N [--health-port N] [--test-clock] | durable --config FILE\n", msg)
	os.Exit(2)
}

func main() {
	cfgPath := flag.String("config", "", "runtime.toml; [durable] is read")
	data := flag.String("data", "", "data directory; the WAL is <data>/wal")
	port := flag.Int("port", 0, "gRPC port (tl.durable.v1)")
	healthPort := flag.Int("health-port", 0, "health port: /healthz, /readyz, /metrics")
	testClock := flag.Bool("test-clock", false, "serve /debug/clock on the health port (tests only)")
	replicas := flag.Int("replicas", 1, "Raft replicas (dur.10); this reference runs one")
	flag.Parse()
	if flag.NArg() != 0 {
		usage("unexpected arguments: " + strings.Join(flag.Args(), " "))
	}
	if *replicas != 1 {
		usage(fmt.Sprintf("--replicas %d: this reference server runs a single replica", *replicas))
	}

	c := config.Default().Durable
	if *cfgPath != "" {
		full, err := config.Load(*cfgPath, os.Environ())
		if err != nil {
			fmt.Fprintf(os.Stderr, "durable: %v\n", err)
			os.Exit(1)
		}
		c = full.Durable
	}
	if *data != "" {
		c.WALDir = filepath.Join(*data, "wal")
	}
	if *port != 0 {
		c.GRPCListen = fmt.Sprintf(":%d", *port)
	}
	if *healthPort != 0 {
		c.HealthListen = fmt.Sprintf(":%d", *healthPort)
	}
	if c.WALDir == "" {
		usage("no WAL directory: give --data or [durable].wal_dir")
	}
	if c.Raft.Enabled {
		usage("[durable.raft].enabled: this reference server runs a single replica")
	}

	var clk server.Clock = wall{}
	var offset *timer.OffsetClock
	if *testClock {
		offset = timer.NewOffsetClock(wall{})
		clk = offset
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()

	wal, err := dlog.Open(c.WALDir, dlog.Options{MaxBytes: c.WALMaxBytes, Now: clk.Now})
	if err != nil {
		log.Fatalf("durable: open the log in %s: %v", c.WALDir, err)
	}
	q, err := queue.Open(ctx, queue.Options{
		Log: wal, Clock: clk,
		Visibility:  time.Duration(c.VisibilityTimeoutMS) * time.Millisecond,
		MaxAttempts: c.DLQAfterAttempts,
	})
	if err != nil {
		log.Fatalf("durable: open the queues: %v", err)
	}
	timers := timer.New[server.TimerKey](clk)
	if offset != nil {
		offset.OnShift(timers.Wake)
	}
	srv, err := server.Open(ctx, server.Options{
		Log: wal, Queue: q, Clock: clk, Timers: timers,
		PollWait:         time.Duration(c.LongPollMS) * time.Millisecond,
		DLQAfterAttempts: c.DLQAfterAttempts,
		Seed:             uint64(time.Now().UnixNano()),
	})
	if err != nil {
		log.Fatalf("durable: recover: %v", err)
	}
	timerCtx, stopTimers := context.WithCancel(context.Background())
	timersDone := make(chan struct{})
	go func() {
		defer close(timersDone)
		_ = timers.Run(timerCtx, srv.FireTimer)
	}()

	g := grpc.NewServer()
	durablev1.RegisterWorkflowServiceServer(g, srv)
	durablev1.RegisterTaskServiceServer(g, srv)
	ln, err := net.Listen("tcp", c.GRPCListen)
	if err != nil {
		log.Fatalf("durable: listen %s: %v", c.GRPCListen, err)
	}

	var ready atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprintln(w, "ok") })
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		if !ready.Load() {
			http.Error(w, "draining", http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintln(w, "ready")
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) { writeMetrics(w, wal, q) })
	if offset != nil {
		mux.Handle("/debug/clock", offset.Handler())
	}
	health := &http.Server{Addr: c.HealthListen, Handler: mux, ReadHeaderTimeout: 5 * time.Second}

	errs := make(chan error, 2)
	go func() { errs <- g.Serve(ln) }()
	go func() {
		if err := health.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errs <- err
		}
	}()
	ready.Store(true)
	log.Printf("durable: gRPC on %s, health on %s, WAL %s (%d bytes)", c.GRPCListen, c.HealthListen, c.WALDir, wal.Size())

	code := 0
	select {
	case <-ctx.Done():
	case err := <-errs:
		log.Printf("durable: %v", err)
		code = 1
	}
	ready.Store(false)
	drained := make(chan struct{})
	go func() { g.GracefulStop(); close(drained) }()
	select {
	case <-drained:
	case <-time.After(10 * time.Second):
		g.Stop()
	}
	stopTimers()
	<-timersDone
	sctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = health.Shutdown(sctx)
	if err := wal.Close(); err != nil {
		log.Printf("durable: close the log: %v", err)
		code = 1
	}
	log.Printf("durable: stopped")
	os.Exit(code)
}

// writeMetrics is the Prometheus text form of the queue gauges
// (contracts/otel/metrics.yaml): pollable tasks per task queue and kind, and
// the dead letters per task queue. Server queues are "wf:<q>" and "act:<q>".
func writeMetrics(w http.ResponseWriter, wal *dlog.Log, q *queue.Queue) {
	names := []string{}
	for _, s := range wal.Streams("queue/") {
		names = append(names, strings.TrimPrefix(s, "queue/"))
	}
	sort.Strings(names)
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintln(w, "# HELP tl_durable_task_queue_depth Pollable tasks per queue.")
	fmt.Fprintln(w, "# TYPE tl_durable_task_queue_depth gauge")
	dead := map[string]int{}
	for _, n := range names {
		kind, tq, ok := strings.Cut(n, ":")
		if !ok {
			continue
		}
		switch kind {
		case "wf":
			kind = "workflow"
		case "act":
			kind = "activity"
		default:
			continue
		}
		visible, _, d := q.Depth(n)
		dead[tq] += d
		fmt.Fprintf(w, "tl_durable_task_queue_depth{queue=%q,kind=%q} %d\n", tq, kind, visible)
	}
	fmt.Fprintln(w, "# HELP tl_durable_dlq_size Dead-lettered tasks per queue.")
	fmt.Fprintln(w, "# TYPE tl_durable_dlq_size gauge")
	tqs := make([]string, 0, len(dead))
	for tq := range dead {
		tqs = append(tqs, tq)
	}
	sort.Strings(tqs)
	for _, tq := range tqs {
		fmt.Fprintf(w, "tl_durable_dlq_size{queue=%q} %d\n", tq, dead[tq])
	}
}
