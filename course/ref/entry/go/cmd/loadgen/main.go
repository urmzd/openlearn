// Command loadgen is the reference entry point for the `loadgen` role
// (course/contracts/spec/cli-roles.md): the main of load.01's open-loop
// generator and, as `loadgen compare`, of load.02's regression gate. Entry
// points are learner territory (DESIGN D16): the learner writes their own
// go/cmd/loadgen; this one shows what MS-L10, MS-prod, the dep.05 perf gate,
// and the drills call.
//
//	loadgen --target <base URL without /v1> [--model m] [--prompt p] [--max-tokens n]
//	        [--mode poisson|constant|burst] [--rate rps] [--burst n] [--duration d]
//	        [--seed s] [--out report.json]
//	loadgen compare <base.json> <head.json> --metric ttft_p95 --max-regress 5%
//	        [--alpha 0.05] [--permutations 1000] [--seed s]
//
// A run sends streamed chat requests with the key in $TL_API_KEY (the
// [endpoints].api_key_env default), writes the report
// (contracts/formats/loadgen-report.schema.json) to --out, and prints it as
// the final line of stdout. --mode burst sends --burst requests at once,
// then again every --burst / --rate seconds (once per --duration without
// --rate). `compare` prints its verdict as the final JSON line and exits 1
// when the head regressed.
//
// Exit codes: 0 done, 1 a regression (compare), 2 a usage error, 3 any
// other failure (an unreadable report, a target that is not a URL).
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"tinyllm/ds/rng"
	"tinyllm/loadgen"
	"tinyllm/loadgen/compare"
)

const (
	exitRegressed = 1
	exitUsage     = 2
	exitFailure   = 3
)

func main() {
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "compare" {
		os.Exit(runCompare(args[1:], os.Stdout, os.Stderr))
	}
	os.Exit(runLoad(args, os.Stdout, os.Stderr))
}

// runLoad is one load run; it returns the exit code.
func runLoad(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("loadgen", flag.ContinueOnError)
	fs.SetOutput(stderr)
	target := fs.String("target", "", "base URL without /v1, e.g. http://127.0.0.1:30080")
	model := fs.String("model", "tracer", "model id sent in every request")
	prompt := fs.String("prompt", "Once upon a time", "user message of every request")
	maxTokens := fs.Int("max-tokens", 16, "max_tokens of every request")
	mode := fs.String("mode", "poisson", "arrival schedule: poisson, constant, or burst")
	rate := fs.Float64("rate", 0, "offered rate in requests per second (burst: a burst every --burst / --rate seconds)")
	burst := fs.Int("burst", 1, "requests per burst (--mode burst)")
	duration := fs.Duration("duration", 60*time.Second, "how long arrivals are scheduled")
	seed := fs.Uint64("seed", 0, "seed of the poisson arrival stream")
	out := fs.String("out", "", "also write the report to this file")
	keyEnv := fs.String("api-key-env", "TL_API_KEY", "environment variable holding the API key")
	sloTTFT := fs.Float64("slo-ttft-ms", 500, "TTFT bound of goodput, ms (DESIGN 2.11)")
	sloTPOT := fs.Float64("slo-tpot-ms", 60, "TPOT bound of goodput, ms (DESIGN 2.11)")
	if err := fs.Parse(args); err != nil {
		return exitUsage
	}
	usage := func(msg string) int {
		fmt.Fprintf(stderr, "loadgen: %s\n", msg)
		fs.Usage()
		return exitUsage
	}
	if fs.NArg() != 0 {
		return usage(fmt.Sprintf("unexpected argument %q", fs.Arg(0)))
	}
	if *target == "" {
		return usage("--target is required")
	}
	if u, err := url.Parse(*target); err != nil || u.Scheme == "" || u.Host == "" {
		return usage(fmt.Sprintf("--target %q is not a base URL", *target))
	}
	if *duration <= 0 || *maxTokens < 1 || *burst < 1 || *rate < 0 || math.IsInf(*rate, 0) || math.IsNaN(*rate) {
		return usage("--duration, --max-tokens, and --burst must be positive, --rate non-negative")
	}

	var sched loadgen.Schedule
	switch *mode {
	case "poisson", "constant":
		if *rate <= 0 {
			return usage(fmt.Sprintf("--mode %s needs --rate > 0", *mode))
		}
		if *mode == "poisson" {
			sched = loadgen.Poisson(*rate, rng.Stream(*seed, rng.PurposeSample))
		} else {
			sched = loadgen.Constant(*rate)
		}
	case "burst":
		every := *duration // one burst per run unless --rate says otherwise
		if *rate > 0 {
			every = time.Duration(float64(*burst) / *rate * float64(time.Second))
		}
		if every <= 0 {
			return usage("--rate is too high for --burst")
		}
		sched = loadgen.Burst(*burst, every)
	default:
		return usage(fmt.Sprintf("--mode %q: want poisson, constant, or burst", *mode))
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	statuses := &statusCount{next: http.DefaultTransport, n: map[string]int{}}
	rep, err := loadgen.Run(ctx, loadgen.Config{
		Client:    &http.Client{Transport: statuses},
		Target:    strings.TrimRight(*target, "/"),
		Model:     *model,
		Prompts:   []string{*prompt},
		MaxTokens: *maxTokens,
		APIKey:    os.Getenv(*keyEnv),
		Schedule:  sched,
		Duration:  *duration,
		SLO:       map[string]float64{"ttft_ms_p95": *sloTTFT, "tpot_ms_p95": *sloTPOT},
		RunID:     "run-" + time.Now().UTC().Format("20060102T150405Z"),
		Seed:      *seed,
	})
	if err != nil {
		fmt.Fprintf(stderr, "loadgen: %v\n", err)
		return exitFailure
	}
	line, err := json.Marshal(rep)
	if err != nil {
		fmt.Fprintf(stderr, "loadgen: report: %v\n", err)
		return exitFailure
	}
	if *out != "" {
		if err := os.WriteFile(*out, append(line, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "loadgen: %v\n", err)
			return exitFailure
		}
	}
	if rep.Errors > 0 {
		fmt.Fprintf(stderr, "loadgen: %d of %d requests failed; responses: %s\n", rep.Errors, rep.Requests, statuses.String())
	}
	fmt.Fprintf(stdout, "%d requests, %d errors, TTFT p50 %.1f ms, TPOT p50 %.1f ms, %.1f tokens/s\n",
		rep.Requests, rep.Errors, rep.TTFTms.P50, rep.TPOTms.P50, rep.TokensPerS)
	fmt.Fprintln(stdout, string(line))
	return 0
}

// statusCount counts the responses (or transport errors) of a run, so a
// run whose requests failed says how on stderr; the report holds counts only.
type statusCount struct {
	next http.RoundTripper
	mu   sync.Mutex
	n    map[string]int
}

func (s *statusCount) RoundTrip(r *http.Request) (*http.Response, error) {
	resp, err := s.next.RoundTrip(r)
	k := "transport error"
	if err == nil {
		k = "HTTP " + strconv.Itoa(resp.StatusCode)
	}
	s.mu.Lock()
	s.n[k]++
	s.mu.Unlock()
	return resp, err
}

func (s *statusCount) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	keys := make([]string, 0, len(s.n))
	for k := range s.n {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = fmt.Sprintf("%s x%d", k, s.n[k])
	}
	return strings.Join(parts, ", ")
}

// verdictJSON is compare's final line. delta is null when the base is 0 and
// the head is not (an infinite relative change).
type verdictJSON struct {
	Metric     string   `json:"metric"`
	Base       float64  `json:"base"`
	Head       float64  `json:"head"`
	Delta      *float64 `json:"delta"`
	PValue     float64  `json:"p_value"`
	MaxRegress float64  `json:"max_regress"`
	Regressed  bool     `json:"regressed"`
}

// runCompare is `loadgen compare`; it returns the exit code.
func runCompare(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("loadgen compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	metric := fs.String("metric", "", "<metric>_<stat> (ttft_p95, tpot_p50, e2e_mean, ...) or error_rate")
	maxRegress := fs.String("max-regress", "", "the budget: 5% or 0.05 lets the head be 5% worse")
	alpha := fs.Float64("alpha", 0.05, "significance level of the permutation test")
	perms := fs.Int("permutations", 1000, "random relabelings of the permutation test")
	seed := fs.Uint64("seed", 0, "seed of the permutation test")
	usage := func(msg string) int {
		fmt.Fprintf(stderr, "loadgen compare: %s\nusage: loadgen compare <base.json> <head.json> --metric M --max-regress PCT [--alpha A] [--permutations N] [--seed S]\n", msg)
		return exitUsage
	}
	// Flags may come before, between, or after the two report paths.
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return exitUsage
		}
		if fs.NArg() == 0 {
			break
		}
		pos = append(pos, fs.Arg(0))
		args = fs.Args()[1:]
	}
	if len(pos) != 2 {
		return usage(fmt.Sprintf("want 2 report paths, got %d", len(pos)))
	}
	if *metric == "" || *maxRegress == "" {
		return usage("--metric and --max-regress are required")
	}
	budget, err := compare.ParseRegress(*maxRegress)
	if err != nil {
		return usage(err.Error())
	}
	if _, _, err := compare.ParseMetric(*metric); err != nil {
		return usage(err.Error())
	}
	if !(*alpha > 0 && *alpha < 1) || *perms < 1 {
		return usage("--alpha must be in (0, 1) and --permutations at least 1")
	}
	reports := make([]*loadgen.Report, 2)
	for i, p := range pos {
		b, err := os.ReadFile(p)
		if err != nil {
			fmt.Fprintf(stderr, "loadgen compare: %v\n", err)
			return exitFailure
		}
		var r loadgen.Report
		if err := json.Unmarshal(lastJSONLine(b), &r); err != nil {
			fmt.Fprintf(stderr, "loadgen compare: %s is not a loadgen report: %v\n", p, err)
			return exitFailure
		}
		reports[i] = &r
	}
	v, err := compare.Compare(reports[0], reports[1], compare.Options{
		Metric: *metric, MaxRegress: budget, Alpha: *alpha, Permutations: *perms, Seed: *seed,
	})
	if err != nil {
		fmt.Fprintf(stderr, "loadgen compare: %v\n", err)
		return exitFailure
	}
	out := verdictJSON{Metric: *metric, Base: v.Base, Head: v.Head, PValue: v.PValue, MaxRegress: budget, Regressed: v.Regressed}
	if !math.IsInf(v.Delta, 0) && !math.IsNaN(v.Delta) {
		d := v.Delta
		out.Delta = &d
	}
	word := "ok"
	if v.Regressed {
		word = "REGRESSED"
	}
	fmt.Fprintf(stdout, "%s: base %.3f, head %.3f, change %+.1f%% (budget %.1f%%), p = %.4f: %s\n",
		*metric, v.Base, v.Head, 100*v.Delta, 100*budget, v.PValue, word)
	line, _ := json.Marshal(out)
	fmt.Fprintln(stdout, string(line))
	if v.Regressed {
		return exitRegressed
	}
	return 0
}

// lastJSONLine accepts a report file or a saved stdout whose final line is
// the report.
func lastJSONLine(b []byte) []byte {
	lines := strings.Split(strings.TrimSpace(string(b)), "\n")
	last := strings.TrimSpace(lines[len(lines)-1])
	if strings.HasPrefix(last, "{") {
		return []byte(last)
	}
	return b
}
