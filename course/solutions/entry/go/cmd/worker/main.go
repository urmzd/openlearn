// Command worker is the reference entry point for the `worker` role
// (course/contracts/spec/cli-roles.md): the composition root of the dur.04
// worker SDK, the dur.06 workflow registry, the platform workflows
// (CorpusBuild, TrainRun, EvalSuite), and the dur.09 subprocess activities.
// Entry points are learner territory (DESIGN D16): the learner writes their
// own go/cmd/worker; this one shows what the milestone runner and the dep.06
// chart expect.
//
//	worker --queue <q>[,<q>...] --durable <host:port> [--health-port <n>] [--test-activities]
//	worker --config <runtime.toml>
//
// --config reads [worker] and [paths].artifacts; the flags, when given, win.
// --durable defaults to $TL_DURABLE_ADDR. Python entries: [worker].python
// (else $TL_TINYLLM_ENTRY, else `python python/tinyllm/__main__.py`) and
// $TL_CORPUS_ENTRY (else `python python/corpus/__main__.py`), each split on
// spaces. The artifact root is $TL_ARTIFACTS, else [paths].artifacts, else
// ./artifacts.
//
// --test-activities registers the course test activity tl.test.Append (POSTs
// its idempotency key to $TL_TEST_EFFECTS_URL, the testkit effects sink, and
// records nothing when that is unset) and the test workflows Echo (one
// tl.test.Append of its input, returns the input), SleepDemo (input:
// milliseconds; result: the fired time in Unix milliseconds), and
// tl.test.Batch (input N: N tl.test.Append activities with ids 1 to N, so
// the sink's keys are "<workflow id>/1" to "<workflow id>/N"; the kill
// loop of MS-durable).
//
// ReadArtifact (artifact.go) is always registered: it returns a file under
// the artifacts root, which is how ctl reads what an activity wrote.
//
// The health port serves /healthz and /readyz (503 while draining). SIGTERM
// stops polling, lets in-flight activities finish for up to 60 s (the chart's
// grace period is longer), and exits 0.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"tinyllm/activities"
	"tinyllm/config"
	"tinyllm/durable/activity"
	"tinyllm/durable/worker"
	"tinyllm/durable/workflow"
	"tinyllm/workflows"
)

const drainTimeout = 60 * time.Second

func usage(msg string) {
	fmt.Fprintf(os.Stderr, "worker: %s\nusage: worker --queue Q[,Q...] --durable HOST:PORT [--health-port N] [--test-activities] | worker --config FILE\n", msg)
	os.Exit(2)
}

func main() {
	cfgPath := flag.String("config", "", "runtime.toml; [worker] and [paths] are read")
	queues := flag.String("queue", "", "task queue(s) to poll, comma-separated")
	durable := flag.String("durable", "", "durable server host:port (default $TL_DURABLE_ADDR)")
	healthPort := flag.Int("health-port", 0, "health port: /healthz and /readyz")
	testActs := flag.Bool("test-activities", false, "register tl.test.Append, Echo, SleepDemo, and tl.test.Batch (tests only)")
	flag.Parse()
	if flag.NArg() != 0 {
		usage("unexpected arguments: " + strings.Join(flag.Args(), " "))
	}

	full := config.Default()
	if *cfgPath != "" {
		var err error
		if full, err = config.Load(*cfgPath, os.Environ()); err != nil {
			fmt.Fprintf(os.Stderr, "worker: %v\n", err)
			os.Exit(1)
		}
	}
	c := full.Worker
	if *queues != "" {
		c.TaskQueues = nil
		for _, q := range strings.Split(*queues, ",") {
			if q = strings.TrimSpace(q); q != "" {
				c.TaskQueues = append(c.TaskQueues, q)
			}
		}
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
	if len(c.TaskQueues) == 0 {
		usage("no task queue: give --queue or [worker].task_queues")
	}
	if c.Durable == "" {
		usage("no durable address: give --durable, [worker].durable, or TL_DURABLE_ADDR")
	}
	artifacts := firstOf(os.Getenv("TL_ARTIFACTS"), full.Paths.Artifacts, "artifacts")
	tinyllm := strings.Fields(firstOf(c.Python, os.Getenv("TL_TINYLLM_ENTRY"), "python python/tinyllm/__main__.py"))
	corpus := strings.Fields(firstOf(os.Getenv("TL_CORPUS_ENTRY"), "python python/corpus/__main__.py"))

	conn, err := grpc.NewClient(c.Durable, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		usage(fmt.Sprintf("--durable %s: %v", c.Durable, err))
	}
	defer conn.Close()

	reg := workflow.NewRegistry()
	reg.Register("CorpusBuild", workflows.Workflow(workflows.CorpusBuild))
	reg.Register("TrainRun", workflows.Workflow(workflows.TrainRun))
	reg.Register("EvalSuite", workflows.Workflow(workflows.EvalSuite))
	reg.Register("ReadArtifact", readArtifact)
	if *testActs {
		reg.Register("Echo", echo)
		reg.Register("SleepDemo", sleepDemo)
		reg.Register("tl.test.Batch", batch)
	}

	canceled := func(cause error) bool { return errors.Is(cause, worker.ErrCancelRequested) }
	py := activities.Subprocess{Entry: tinyllm, Artifacts: artifacts, OTLPEndpoint: full.OTel.Endpoint, ServiceName: os.Getenv("OTEL_SERVICE_NAME"), Canceled: canceled}
	cp := activities.Subprocess{Entry: corpus, Artifacts: artifacts, OTLPEndpoint: full.OTel.Endpoint, ServiceName: os.Getenv("OTEL_SERVICE_NAME"), Canceled: canceled}
	acts := map[string]worker.ActivityFunc{
		workflows.ActivityTrain: subprocess(py, func([]byte) []string { return []string{"train"} }),
		workflows.ActivityEval:  subprocess(py, func([]byte) []string { return []string{"eval"} }),
		workflows.ActExport:     subprocess(py, func([]byte) []string { return []string{"export"} }),
		workflows.ActCorpusStage: func(ctx context.Context, in []byte) ([]byte, error) {
			var st workflows.StageInput
			if err := json.Unmarshal(in, &st); err != nil {
				return nil, activity.NewNonRetryable("BadInput", err.Error())
			}
			return subprocess(cp, func([]byte) []string { return []string{"run", "--stage", st.Stage} })(ctx, st.Config)
		},
		workflows.ActCorpusCleanup: subprocess(cp, func([]byte) []string { return []string{"cleanup"} }),
		actReadArtifact:            artifactReader(artifacts),
	}
	if *testActs {
		acts["tl.test.Append"] = appendEffect
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
	var wg sync.WaitGroup
	for _, q := range c.TaskQueues {
		w := worker.New(conn, worker.Options{
			TaskQueue: q, Identity: c.Identity, MaxActivities: c.MaxConcurrentActivities,
			Workflows: reg, DrainTimeout: drainTimeout, Logf: log.Printf,
		})
		for name, fn := range acts {
			w.RegisterActivity(name, fn)
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := w.Run(runCtx); err != nil {
				log.Printf("worker: queue %s: %v", q, err)
			}
		}()
	}
	log.Printf("worker: queues %s on %s, health on %s, artifacts %s", strings.Join(c.TaskQueues, ","), c.Durable, c.HealthListen, artifacts)

	code := 0
	select {
	case <-ctx.Done():
	case err := <-healthErr:
		log.Printf("worker: health: %v", err)
		code = 1
	}
	draining.Store(true)
	cancelRun()
	wg.Wait()
	sctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = health.Shutdown(sctx)
	log.Printf("worker: drained")
	os.Exit(code)
}

func firstOf(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// subprocess adapts a dur.09 runner to an activity: the delivery's Info is
// the runner's Task, and its heartbeats go through the worker SDK (a cancel
// arrives as the context's cause).
func subprocess(r activities.Subprocess, args func(in []byte) []string) worker.ActivityFunc {
	return func(ctx context.Context, in []byte) ([]byte, error) {
		inf, _ := worker.InfoFrom(ctx)
		t := activities.Task{
			IdempotencyKey: inf.IdempotencyKey, Attempt: inf.Attempt,
			LastHeartbeat: inf.HeartbeatDetails, TraceContext: inf.TraceContext,
		}
		hb := func(ctx context.Context, details []byte) (bool, error) {
			err := worker.Heartbeat(ctx, details)
			return errors.Is(context.Cause(ctx), worker.ErrCancelRequested), err
		}
		return r.Run(ctx, t, args(in), in, hb)
	}
}

// appendEffect is tl.test.Append: one POST of {key, value} to the effects
// sink, which dedupes by key, so a retried attempt is visible but harmless.
// Without a sink (a run that checks no effects, such as the MS-durable
// smoke) it records nothing and returns its input. An input object with
// "hold_ms" (tl.test.Batch's) holds that long after the effect before the
// activity completes: the window in which a kill leaves an applied effect
// unreported, so the redelivery is what the exactly-once check sees.
func appendEffect(ctx context.Context, in []byte) ([]byte, error) {
	if err := postEffect(ctx, in); err != nil {
		return nil, err
	}
	var opt struct {
		HoldMs int `json:"hold_ms"`
	}
	if json.Unmarshal(in, &opt) == nil && opt.HoldMs > 0 {
		select {
		case <-time.After(time.Duration(opt.HoldMs) * time.Millisecond):
		case <-ctx.Done():
			return nil, context.Cause(ctx)
		}
	}
	return in, nil
}

// postEffect records the activity's idempotency key in the effects sink.
func postEffect(ctx context.Context, in []byte) error {
	url := os.Getenv("TL_TEST_EFFECTS_URL")
	if url == "" {
		return nil
	}
	body, _ := json.Marshal(map[string]string{"key": activity.IdempotencyKey(ctx), "value": string(in)})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("effects sink answered %d", resp.StatusCode)
	}
	return nil
}

var testActivity = workflow.ActivityOptions{StartToClose: 30 * time.Second}

// echo runs one tl.test.Append of its input and returns the input.
func echo(ctx workflow.Context, in []byte) ([]byte, error) {
	return workflow.ExecuteActivity[[]byte](ctx, "tl.test.Append", in, testActivity).Get(ctx)
}

// sleepDemo sleeps on a durable timer for its input (JSON milliseconds) and
// returns the workflow time it woke at, in Unix milliseconds (a JSON integer).
func sleepDemo(ctx workflow.Context, in []byte) ([]byte, error) {
	var ms int64
	if err := json.Unmarshal(in, &ms); err != nil {
		return nil, fmt.Errorf("SleepDemo: input is milliseconds as JSON: %w", err)
	}
	if err := workflow.Sleep(ctx, time.Duration(ms)*time.Millisecond); err != nil {
		return nil, err
	}
	return json.Marshal(workflow.Now(ctx).UnixMilli())
}

// batchHoldMs is how long each of tl.test.Batch's appends holds after its
// effect (appendEffect).
const batchHoldMs = 100

// batch is tl.test.Batch: input N (JSON), N tl.test.Append activities
// scheduled at once, the i-th (0-based) with input {"index": i, "hold_ms":
// batchHoldMs}. The SDK numbers activity ids in schedule order from 1, so
// the i-th has id i+1 and idempotency key "<workflow id>/<i+1>": the keys
// are "<workflow id>/1" to "<workflow id>/N" on every replay. The result is
// N once every append completed.
func batch(ctx workflow.Context, in []byte) ([]byte, error) {
	var n int
	if err := json.Unmarshal(in, &n); err != nil || n < 0 {
		return nil, fmt.Errorf("tl.test.Batch: input is a non-negative integer as JSON, got %q", in)
	}
	fs := make([]workflow.Future[[]byte], n)
	for i := range fs {
		in := map[string]int{"index": i, "hold_ms": batchHoldMs}
		fs[i] = workflow.ExecuteActivity[[]byte](ctx, "tl.test.Append", in, testActivity)
	}
	for _, f := range fs {
		if _, err := f.Get(ctx); err != nil {
			return nil, err
		}
	}
	return json.Marshal(n)
}
