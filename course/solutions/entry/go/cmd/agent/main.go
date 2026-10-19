// Command agent is the reference entry point for the `agent` role
// (course/contracts/spec/cli-roles.md, ag.05, dep.07): the agent worker. It
// polls the agent task queue of the durable server and runs the AgentRun
// workflow, whose one activity, tl.agent.Run, is a durable agent run
// (agentkit.Kit.Ask over durableagent): the run's journal lives under
// <artifacts>/agent-runs, so a retried activity (the worker was killed)
// replays the model and tool steps that completed instead of calling them
// again. `ctl agent ask --durable <addr>` starts AgentRun and waits.
//
//	agent --config <runtime.toml> [--queue <q>] [--durable-address <host:port>] [--health-port <n>]
//
// --config reads [agent] (durable, task_queue, health_listen, base_url,
// model, api_key_env, rag_index, max_iterations), [paths].artifacts, and
// [ext].faketool_rules; flags win over the file. PROVIDER_BASE_URL,
// PROVIDER_MODEL, and PROVIDER_API_KEY (the dep.07 chart's environment)
// win over [agent]. With no base URL the provider is the faketool rules
// file (PR CI); a base URL (the gateway) selects the OpenAI provider.
//
// AgentRun input: {run_id, query, suite}; suite (default: [agent].rag_index)
// is resolved by agentkit.ResolveIndex. Result: agentkit.Answer.
//
// The health port serves /healthz and /readyz (503 while draining). SIGTERM
// stops polling, lets an in-flight run finish for up to 60 s, and exits 0.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"tinyllm/agent/durableagent"
	"tinyllm/cmd/internal/agentkit"
	"tinyllm/config"
	"tinyllm/durable/activity"
	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"
)

const (
	actRun       = "tl.agent.Run"
	drainTimeout = 60 * time.Second
)

// runInput is AgentRun's input.
type runInput struct {
	RunID string `json:"run_id"`
	Query string `json:"query"`
	Suite string `json:"suite,omitempty"`
}

func usage(msg string) {
	fmt.Fprintf(os.Stderr, "agent: %s\nusage: agent --config FILE [--queue Q] [--durable-address HOST:PORT] [--health-port N]\n", msg)
	os.Exit(2)
}

func main() {
	cfgPath := flag.String("config", "", "runtime.toml; [agent], [paths], and [ext] are read")
	queue := flag.String("queue", "", "task queue (default [agent].task_queue)")
	durable := flag.String("durable-address", "", "durable server host:port (default [agent].durable, else $TL_DURABLE_ADDR)")
	flag.StringVar(durable, "durable", "", "alias of --durable-address")
	healthPort := flag.Int("health-port", 0, "health port: /healthz and /readyz")
	flag.Parse()
	if flag.NArg() != 0 {
		usage("unexpected arguments: " + strings.Join(flag.Args(), " "))
	}
	full := config.Default()
	if *cfgPath != "" {
		var err error
		if full, err = config.Load(*cfgPath, os.Environ()); err != nil {
			fmt.Fprintf(os.Stderr, "agent: %v\n", err)
			os.Exit(1)
		}
	}
	c := full.Agent
	if *queue != "" {
		c.TaskQueue = *queue
	}
	if *durable != "" {
		c.Durable = *durable
	}
	if c.Durable == "" {
		c.Durable = os.Getenv("TL_DURABLE_ADDR")
	}
	if *healthPort != 0 {
		c.HealthListen = fmt.Sprintf(":%d", *healthPort)
	}
	if v := os.Getenv("PROVIDER_BASE_URL"); v != "" {
		c.BaseURL = v
	}
	if v := os.Getenv("PROVIDER_MODEL"); v != "" {
		c.Model = v
	}
	apiKey := os.Getenv(c.APIKeyEnv)
	if v := os.Getenv("PROVIDER_API_KEY"); v != "" {
		apiKey = v
	}
	if c.Durable == "" {
		usage("no durable address: give --durable-address, [agent].durable, or TL_DURABLE_ADDR")
	}
	if c.RAGIndex == "" {
		c.RAGIndex = "course-docs"
	}
	rules, _ := full.Ext["faketool_rules"].(string)
	kit, err := agentkit.New(agentkit.Options{Rules: rules, BaseURL: c.BaseURL, APIKey: apiKey, Model: c.Model, MaxIter: c.MaxIterations})
	if err != nil {
		usage(err.Error())
	}
	defer kit.Close()
	artifacts := full.Paths.Artifacts
	if v := os.Getenv("TL_ARTIFACTS"); v != "" {
		artifacts = v
	}
	store, err := durableagent.OpenFileStore(filepath.Join(artifacts, "agent-runs"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: %v\n", err)
		os.Exit(1)
	}

	conn, err := grpc.NewClient(c.Durable, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		usage(fmt.Sprintf("--durable-address %s: %v", c.Durable, err))
	}
	defer conn.Close()

	reg := workflow.NewRegistry()
	reg.Register("AgentRun", func(ctx workflow.Context, in []byte) ([]byte, error) {
		return workflow.ExecuteActivity[[]byte](ctx, actRun, in, workflow.ActivityOptions{StartToClose: 10 * time.Minute}).Get(ctx)
	})
	run := func(ctx context.Context, in []byte) ([]byte, error) {
		var r runInput
		if err := json.Unmarshal(in, &r); err != nil || r.Query == "" || !durableagent.ValidRunID(r.RunID) {
			return nil, activity.NewNonRetryable("BadInput", "AgentRun input is {run_id, query, suite}")
		}
		suite := r.Suite
		if suite == "" {
			suite = c.RAGIndex
		}
		dir, err := agentkit.ResolveIndex(suite)
		if err != nil {
			return nil, activity.NewNonRetryable("BadInput", err.Error())
		}
		a, err := kit.Ask(ctx, store, r.RunID, dir, r.Query)
		if err != nil {
			return nil, err
		}
		return json.Marshal(a)
	}

	var draining atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprintln(w, "ok") })
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		if draining.Load() {
			http.Error(w, "draining", http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintln(w, "ready")
	})
	health := &http.Server{Addr: c.HealthListen, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	healthErr := make(chan error, 1)
	go func() {
		if err := health.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			healthErr <- err
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()
	runCtx, cancelRun := context.WithCancel(ctx)
	defer cancelRun()
	w := worker.New(conn, worker.Options{TaskQueue: c.TaskQueue, Workflows: reg, DrainTimeout: drainTimeout, Logf: log.Printf})
	w.RegisterActivity(actRun, run)
	done := make(chan struct{})
	go func() {
		defer close(done)
		if err := w.Run(runCtx); err != nil {
			log.Printf("agent: queue %s: %v", c.TaskQueue, err)
		}
	}()
	log.Printf("agent: queue %s on %s, health on %s, index %s", c.TaskQueue, c.Durable, c.HealthListen, c.RAGIndex)

	code := 0
	select {
	case <-ctx.Done():
	case err := <-healthErr:
		log.Printf("agent: health: %v", err)
		code = 1
	}
	draining.Store(true)
	cancelRun()
	<-done
	sctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = health.Shutdown(sctx)
	os.Exit(code)
}
