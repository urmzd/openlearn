// Course tests for ag.09: the eval runner. Scorers and subjects here are
// fakes with known outputs (fakes_test.go), the bootstrap is compared with an
// independent Python oracle that uses the frozen PCG32
// (course/oracle/ag.09/make_fixtures.py), and the report is checked against
// the rules of contracts/formats/eval-result.schema.json.
package ag_09

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	"tinyllm/agent/durableagent"
	"tinyllm/agent/eval"
	"tinyllm/agent/loop"
	"tinyllm/agent/tool"
	"tinyllm/agent/types"
)

func fixture(t *testing.T, name string) string {
	t.Helper()
	dir := os.Getenv("TINYLLM_FIXTURES")
	if dir == "" {
		t.Fatal("TINYLLM_FIXTURES is not set (ol check sets it)")
	}
	return filepath.Join(dir, "ag.09", name)
}

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol*math.Max(1, math.Abs(b)) }

// handObs is the chapter's worked example: four answers already scored by
// a value scorer that cannot score case c.
func handObs() []eval.Observation {
	return []eval.Observation{
		{ID: "a", Input: str("q1"), Output: num(1)},
		{ID: "b", Input: str("q2"), Output: num(0)},
		{ID: "c", Input: str("q3"), Output: num(0)},
		{ID: "d", Input: str("q4"), Output: num(1)},
	}
}

func TestHandExample(t *testing.T) {
	// WHY: the chapter's worked example (section 3): scores 1, 0, (error),
	//      1. The mean is over the three values that exist, 2/3, with n = 3
	//      and errored = 1. Averaging the failure in as 0 would report 0.5
	//      and blame the model for a broken scorer.
	// KIND: unit
	// CATCHES: s01
	// CHAPTER: ag.09 section 3, worked example
	res, err := eval.Run(context.Background(), "hand", handObs(), []eval.Scorer{valueScorer{name: "exact", fail: map[string]bool{"c": true}}})
	if err != nil {
		t.Fatal(err)
	}
	st := res.Metrics["exact"]
	if !near(st.Mean, 2.0/3, 1e-15) || st.N != 3 || st.Errored != 1 {
		t.Fatalf("exact = %+v, want mean 2/3, n 3, errored 1", st)
	}
	if !res.Rows[2].Errored(0) || res.Rows[0].Errored(0) {
		t.Fatalf("row c must be errored, row a not: %+v", res.Rows)
	}
	if st.CILow > st.Mean || st.CIHigh < st.Mean {
		t.Fatalf("CI [%v, %v] does not contain the mean %v", st.CILow, st.CIHigh, st.Mean)
	}
}

func TestScorerFailuresExcluded(t *testing.T) {
	// WHY: a scorer fails in four ways: an error, a panic, a NaN or
	//      infinite value, an Error field. Each is excluded and counted,
	//      the run goes on, and the other scorers are unaffected. A panic
	//      that escaped would end the whole eval at case 1.
	// KIND: fault
	// CATCHES: s02, s03, s04
	// CHAPTER: ag.09 section 2.2
	scorers := []eval.Scorer{
		trickScorer{name: "panics", mode: "panic"},
		trickScorer{name: "nan", mode: "nan"},
		trickScorer{name: "inf", mode: "inf"},
		trickScorer{name: "flagged", mode: "error-field"},
		trickScorer{name: "fine", mode: ""},
	}
	res, err := eval.Run(context.Background(), "tricks", handObs(), scorers)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"panics", "nan", "inf", "flagged"} {
		if st := res.Metrics[n]; st.N != 0 || st.Errored != 4 {
			t.Errorf("%s = %+v, want n 0, errored 4", n, st)
		}
	}
	if st := res.Metrics["fine"]; st.Mean != 1 || st.N != 4 || st.Errored != 0 {
		t.Errorf("fine = %+v, want mean 1 over 4", st)
	}
	for _, r := range res.Rows {
		for i, s := range r.Scores {
			if (i < 4) != (s.Error != "") || s.Name != res.Scorers[i] {
				t.Fatalf("row %s score %d = %+v", r.ID, i, s)
			}
		}
	}
}

func TestDeterministicOrderUnderConcurrency(t *testing.T) {
	// WHY: with 8 workers, cases finish in any order; rows must still come
	//      out in case order, then sample order, and every number must equal
	//      a run with one worker. The subject also writes an annotation:
	//      observations must not share one map across workers (-race).
	// KIND: property
	// CATCHES: s05, s06
	// CHAPTER: ag.09 section 2.3
	var obs []eval.Observation
	for i := 0; i < 40; i++ {
		obs = append(obs, eval.Observation{ID: fmt.Sprintf("case-%02d", i), Input: str(fmt.Sprint(i)),
			Annotations: map[string]json.RawMessage{"tags": json.RawMessage(`["x"]`)}})
	}
	subject := func(_ context.Context, o *eval.Observation) error {
		var i int
		json.Unmarshal(o.Input, &i)
		time.Sleep(time.Duration((40-i)%7) * time.Millisecond)
		o.Annotations["seen"] = str(o.ID)
		o.Output = num(float64((i*7+o.Sample)%5) / 4)
		return nil
	}
	run := func(c int) *eval.SuiteResult {
		res, err := eval.Run(context.Background(), "order", obs, []eval.Scorer{valueScorer{name: "v"}},
			eval.WithSubject("s", subject), eval.WithConcurrency(c), eval.WithSamples(2), eval.WithSeed(3))
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	one, eight := run(1), run(8)
	for i, r := range eight.Rows {
		if r.ID != obs[i/2].ID || r.Sample != i%2 {
			t.Fatalf("row %d is %s sample %d, want %s sample %d", i, r.ID, r.Sample, obs[i/2].ID, i%2)
		}
		if string(r.Annotations["seen"]) != string(str(r.ID)) {
			t.Fatalf("row %s carries the annotation of %s", r.ID, r.Annotations["seen"])
		}
		if !reflect.DeepEqual(r.Scores, one.Rows[i].Scores) {
			t.Fatalf("row %d scores differ between 1 and 8 workers", i)
		}
	}
	if !reflect.DeepEqual(one.Metrics, eight.Metrics) {
		t.Fatalf("metrics differ: %+v vs %+v", one.Metrics, eight.Metrics)
	}
	if _, ok := obs[0].Annotations["seen"]; ok {
		t.Fatal("Run wrote into the caller's observation")
	}
}

type bootCase struct {
	Name   string
	Groups [][]float64
	NBoot  int `json:"n_boot"`
	Alpha  float64
	Seed   uint64
	Mean   float64
	Lo, Hi float64
}

func bootCases(t *testing.T) []bootCase {
	b, err := os.ReadFile(fixture(t, "bootstrap.json"))
	if err != nil {
		t.Fatal(err)
	}
	var f struct{ Cases []bootCase }
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatal(err)
	}
	return f.Cases
}

func TestBootstrapMatchesOracle(t *testing.T) {
	// WHY: the interval is specified to the draw (section 2.4), so the
	//      same seed gives the same interval in Go and Python: binary
	//      scores, grouped samples, and a 90% interval, each equal to the
	//      oracle's bounds.
	// KIND: conformance
	// CATCHES: s07, s08, s09
	// CHAPTER: ag.09 section 2.4
	for _, c := range bootCases(t) {
		lo, hi := eval.BootstrapCI(c.Groups, c.NBoot, c.Alpha, eval.BootstrapRNG(c.Seed))
		if !near(lo, c.Lo, 1e-12) || !near(hi, c.Hi, 1e-12) {
			t.Fatalf("%s: CI = [%.17g, %.17g], want [%.17g, %.17g]", c.Name, lo, hi, c.Lo, c.Hi)
		}
	}
}

func TestPercentileBounds(t *testing.T) {
	// WHY: the bounds of a percentile interval are two indices into the
	//      sorted resample means; off by one at either end changes every
	//      interval. 2000 resamples at alpha 0.05 read indices 50 and 1949.
	// KIND: boundary
	// CATCHES: s08
	// CHAPTER: ag.09 section 2.4
	for _, c := range []struct {
		n      int
		alpha  float64
		lo, hi int
	}{{2000, 0.05, 50, 1949}, {1000, 0.05, 25, 974}, {500, 0.1, 25, 474}, {10, 0.05, 0, 9}, {1, 0.05, 0, 0}} {
		lo, hi := eval.PercentileBounds(c.n, c.alpha)
		if lo != c.lo || hi != c.hi {
			t.Errorf("PercentileBounds(%d, %v) = %d, %d; want %d, %d", c.n, c.alpha, lo, hi, c.lo, c.hi)
		}
	}
}

func TestRunResamplesCases(t *testing.T) {
	// WHY: three samples of one case are not three independent cases.
	//      Run's interval resamples cases with all their samples, so with
	//      the grouped fixture as outputs (10 cases x 3 samples) its metric
	//      equals the oracle's cluster bootstrap. Resampling 30 rows would
	//      give a narrower, overconfident interval.
	// KIND: conformance
	// CATCHES: s07
	// CHAPTER: ag.09 section 2.4
	var c bootCase
	for _, x := range bootCases(t) {
		if x.Name == "grouped-10x3" {
			c = x
		}
	}
	var obs []eval.Observation
	for i := range c.Groups {
		obs = append(obs, eval.Observation{ID: fmt.Sprint(i), Input: num(float64(i))})
	}
	subject := func(_ context.Context, o *eval.Observation) error {
		var i int
		json.Unmarshal(o.Input, &i)
		o.Output = num(c.Groups[i][o.Sample])
		return nil
	}
	res, err := eval.Run(context.Background(), "grouped", obs, []eval.Scorer{valueScorer{name: "v"}},
		eval.WithSubject("s", subject), eval.WithSamples(3), eval.WithSeed(c.Seed), eval.WithBootstrap(c.NBoot, c.Alpha))
	if err != nil {
		t.Fatal(err)
	}
	st := res.Metrics["v"]
	if !near(st.Mean, c.Mean, 1e-12) || !near(st.CILow, c.Lo, 1e-12) || !near(st.CIHigh, c.Hi, 1e-12) || st.N != 30 {
		t.Fatalf("v = %+v, want mean %v CI [%v, %v] n 30", st, c.Mean, c.Lo, c.Hi)
	}
}

func TestSubjectErrorFailsRow(t *testing.T) {
	// WHY: when the agent itself fails (a dead provider, a run stuck
	//      awaiting approval), its row has no output to score. Every score
	//      of the row is an error naming the subject's failure, the row is
	//      counted in SubjectErrors, and no scorer sees an empty output.
	// KIND: fault
	// CATCHES: s10
	// CHAPTER: ag.09 section 2.2
	subject := func(_ context.Context, o *eval.Observation) error {
		if o.ID == "b" {
			return errors.New("provider unreachable")
		}
		o.Output = num(1)
		return nil
	}
	res, err := eval.Run(context.Background(), "subj", handObs(), []eval.Scorer{valueScorer{name: "v"}, trickScorer{name: "one"}}, eval.WithSubject("s", subject))
	if err != nil {
		t.Fatal(err)
	}
	r := res.Rows[1]
	if r.SubjectError == "" || !r.Errored(0) || !r.Errored(1) || !strings.Contains(r.Scores[1].Error, "provider unreachable") {
		t.Fatalf("row b = %+v; want both scores errored with the subject's error", r)
	}
	if res.SubjectErrors != 1 || res.Metrics["one"].N != 3 || res.Metrics["one"].Errored != 1 {
		t.Fatalf("SubjectErrors %d, one = %+v", res.SubjectErrors, res.Metrics["one"])
	}
}

func TestRejectsBadSuites(t *testing.T) {
	// WHY: two cases with one id would overwrite each other in every
	//      report and A/B pairing; two scorers with one name would hide one
	//      metric. Both are errors before anything runs.
	// KIND: boundary
	// CATCHES: s19
	// CHAPTER: ag.09 section 4
	dup := append(handObs(), eval.Observation{ID: "a", Input: str("again")})
	if _, err := eval.Run(context.Background(), "x", dup, nil); !errors.Is(err, eval.ErrSuite) {
		t.Fatalf("duplicate case id: %v", err)
	}
	if _, err := eval.Run(context.Background(), "x", []eval.Observation{{Input: str("q")}}, nil); !errors.Is(err, eval.ErrSuite) {
		t.Fatalf("empty case id: %v", err)
	}
	if _, err := eval.Run(context.Background(), "x", handObs(), []eval.Scorer{valueScorer{name: "v"}, trickScorer{name: "v"}}); !errors.Is(err, eval.ErrSuite) {
		t.Fatalf("duplicate scorer name: %v", err)
	}
}

func TestLoadSuite(t *testing.T) {
	// WHY: suites are files in formats/eval-case.schema.json, written by
	//      hand and easy to get wrong: the fixture loads with every field
	//      in place, and a missing required key, an unknown key, or a
	//      repeated id is an error naming its line.
	// KIND: unit
	// CATCHES: s11, s12
	// CHAPTER: ag.09 section 2.1
	f, err := os.Open(fixture(t, "suite.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	obs, err := eval.LoadSuite(f)
	if err != nil {
		t.Fatal(err)
	}
	if len(obs) != 5 || obs[2].ID != "chat-003" || !strings.Contains(string(obs[2].Input), `"messages"`) {
		t.Fatalf("loaded %d cases: %+v", len(obs), obs)
	}
	if string(obs[3].Annotations["scorer_args"]) != `{"tool_called": {"name": "query_usage"}}` || string(obs[0].Annotations["tags"]) != `["rag"]` {
		t.Fatalf("annotations = %s / %s", obs[3].Annotations["scorer_args"], obs[0].Annotations["tags"])
	}
	if obs[4].GroundTruth != nil {
		t.Fatalf("a case without ground_truth has %s", obs[4].GroundTruth)
	}
	for _, bad := range []struct{ text, line string }{
		{`{"input": "q", "tags": [], "scorer_args": {}}`, "line 1"},
		{"\n" + `{"case_id": "a", "input": "q", "tags": [], "scorer_args": {}, "extra": 1}`, "line 2"},
		{`{"case_id": "a", "input": "q", "scorer_args": {}}`, "line 1"},
		{`{"case_id": "a", "input": "q", "tags": [], "scorer_args": {}}` + "\n" + `{"case_id": "a", "input": "r", "tags": [], "scorer_args": {}}`, "line 2"},
		{`not json`, "line 1"},
	} {
		_, err := eval.LoadSuite(strings.NewReader(bad.text))
		if !errors.Is(err, eval.ErrSuite) || !strings.Contains(err.Error(), bad.line) {
			t.Errorf("LoadSuite(%q) err = %v, want ErrSuite naming %s", bad.text, err, bad.line)
		}
	}
}

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

func TestReportSchema(t *testing.T) {
	// WHY: the release gate (dur.12) and the A/B runner read these files:
	//      results.jsonl rows and summary.json must keep
	//      formats/eval-result.schema.json. Checked here: the allowed and
	//      required keys, input_sha as the SHA-256 of the input bytes, an
	//      errored score as null with its message in errors, ttft_ms and
	//      trace_id null when unknown, and the summary's statistics.
	// KIND: conformance
	// CATCHES: s13, s14, s15
	// CHAPTER: ag.09 section 2.5
	obs := handObs()
	obs[0].TraceID = "0af7651916cd43dd8448eb211c80319c"
	obs[0].Timing = eval.Timing{TTFT: 95 * time.Millisecond, Total: 812500 * time.Microsecond}
	obs[0].Tokens = 9
	res, err := eval.Run(context.Background(), "docsqa", obs, []eval.Scorer{valueScorer{name: "exact", fail: map[string]bool{"c": true}}}, eval.WithSeed(5))
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := eval.WriteResults(dir, res.ResultRows(), res.Summary("run-1")); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(dir, "results.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	allowed := map[string]bool{"suite": true, "case_id": true, "subject": true, "sample": true, "input_sha": true, "output": true,
		"scores": true, "errors": true, "latency_ms": true, "ttft_ms": true, "tokens": true, "trace_id": true}
	required := []string{"suite", "case_id", "subject", "input_sha", "output", "scores", "latency_ms", "ttft_ms", "tokens", "trace_id"}
	sc := bufio.NewScanner(f)
	var rows []map[string]json.RawMessage
	for sc.Scan() {
		var m map[string]json.RawMessage
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("results.jsonl line is not JSON: %v", err)
		}
		for k := range m {
			if !allowed[k] {
				t.Fatalf("row key %q is not in the schema", k)
			}
		}
		for _, k := range required {
			if _, ok := m[k]; !ok {
				t.Fatalf("row lacks required %q", k)
			}
		}
		rows = append(rows, m)
	}
	if len(rows) != 4 {
		t.Fatalf("%d rows, want 4", len(rows))
	}
	var sha string
	json.Unmarshal(rows[0]["input_sha"], &sha)
	want := sha256.Sum256(obs[0].Input)
	if sha != hex.EncodeToString(want[:]) || !hex64.MatchString(sha) {
		t.Fatalf("input_sha %q, want sha256 of the input bytes %s", sha, hex.EncodeToString(want[:]))
	}
	if string(rows[0]["ttft_ms"]) != "95" || string(rows[0]["latency_ms"]) != "812.5" || string(rows[0]["trace_id"]) != `"0af7651916cd43dd8448eb211c80319c"` || string(rows[0]["tokens"]) != "9" {
		t.Fatalf("row 0 timing/trace = %s %s %s %s", rows[0]["ttft_ms"], rows[0]["latency_ms"], rows[0]["trace_id"], rows[0]["tokens"])
	}
	if string(rows[1]["ttft_ms"]) != "null" || string(rows[1]["trace_id"]) != "null" {
		t.Fatalf("row 1 unknown ttft/trace must be null: %s %s", rows[1]["ttft_ms"], rows[1]["trace_id"])
	}
	var scores map[string]*float64
	var errs map[string]string
	json.Unmarshal(rows[2]["scores"], &scores)
	json.Unmarshal(rows[2]["errors"], &errs)
	if v, ok := scores["exact"]; !ok || v != nil || errs["exact"] == "" {
		t.Fatalf("errored score: scores %s errors %s; want null and a message", rows[2]["scores"], rows[2]["errors"])
	}
	b, _ := os.ReadFile(filepath.Join(dir, "summary.json"))
	var sum map[string]json.RawMessage
	json.Unmarshal(b, &sum)
	var keys []string
	for k := range sum {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if !reflect.DeepEqual(keys, []string{"metrics", "n_boot", "run_id", "seed", "subjects", "suite"}) {
		t.Fatalf("summary keys %v", keys)
	}
	var metrics map[string]map[string]map[string]float64
	json.Unmarshal(sum["metrics"], &metrics)
	m := metrics[res.Subject]["exact"]
	if len(m) != 5 || m["n"] != 3 || m["errored"] != 1 || !near(m["mean"], 2.0/3, 1e-12) {
		t.Fatalf("summary metrics = %s", sum["metrics"])
	}
	if _, err := os.Stat(filepath.Join(dir, "results.jsonl.tmp")); err == nil {
		t.Fatal("a temporary file was left behind")
	}
}

func TestProviderSubjectTiming(t *testing.T) {
	// WHY: TTFT and token times are what the latency scorers (ag.10) read.
	//      With a clock that advances 10 ms per reading: start at 10, text
	//      pieces at 20 and 30 (TTFT 10 ms, token times 10 and 20 ms), end
	//      of stream at 40 (latency 30 ms); the usage's completion tokens
	//      and the joined text are recorded.
	// KIND: unit
	// CATCHES: s16
	// CHAPTER: ag.09 section 2.6
	p := &script{replies: []reply{{text: []string{"Hel", "lo"}, usage: types.Usage{In: 5, Out: 2}}}}
	clk := &fakeClock{t: time.Unix(0, 0), step: 10 * time.Millisecond}
	res, err := eval.Run(context.Background(), "p", []eval.Observation{{ID: "x", Input: str("Say hello")}}, nil,
		eval.WithSubject("smol", eval.ProviderSubject(p, clk.now)), eval.WithConcurrency(1))
	if err != nil {
		t.Fatal(err)
	}
	o := res.Rows[0].Observation
	if string(o.Output) != `"Hello"` || o.Tokens != 2 || o.Timing.TTFT != 10*time.Millisecond || o.Timing.Total != 30*time.Millisecond ||
		!reflect.DeepEqual(o.Timing.TokenTimes, []time.Duration{10 * time.Millisecond, 20 * time.Millisecond}) {
		t.Fatalf("observation = %s, %d tokens, %+v", o.Output, o.Tokens, o.Timing)
	}
	p = &script{replies: []reply{{text: []string{"Hel"}, fail: errors.New("connection reset")}}}
	res, _ = eval.Run(context.Background(), "p", []eval.Observation{{ID: "x", Input: str("Say hello")}}, nil, eval.WithSubject("smol", eval.ProviderSubject(p, nil)))
	if !strings.Contains(res.Rows[0].SubjectError, "connection reset") {
		t.Fatalf("a stream that ends in an error must fail the case: %+v", res.Rows[0])
	}
}

func TestAgentSubjectRecordsToolCalls(t *testing.T) {
	// WHY: tool-use scorers (ag.10) need to see which tools the agent
	//      called and whether each call failed. The agent here calls lookup
	//      twice (one call has invalid arguments), then answers; the
	//      tool_calls annotation lists both calls in order with their
	//      verdicts and error flags, and tokens sum over both model calls.
	// KIND: unit
	// CATCHES: s17
	// CHAPTER: ag.09 section 2.6
	reg := tool.NewRegistry()
	reg.MustRegister(tool.New("lookup", "look up", `{"type":"object","properties":{"q":{"type":"string"}},"required":["q"],"additionalProperties":false}`,
		func(_ context.Context, args json.RawMessage) (string, error) { return "found", nil }))
	p := &script{replies: []reply{
		{calls: []types.ToolCall{{ID: "c1", Name: "lookup", Args: json.RawMessage(`{"q":"kv"}`)}, {ID: "c2", Name: "lookup", Args: json.RawMessage(`{"x":1}`)}}, usage: types.Usage{In: 10, Out: 4}},
		{text: []string{"done"}, usage: types.Usage{In: 30, Out: 1}},
	}}
	a := loop.New(loop.Config{Provider: p, Tools: reg})
	res, err := eval.Run(context.Background(), "agent", []eval.Observation{{ID: "x", Input: str("find kv")}}, nil, eval.WithSubject("agent", eval.AgentSubject(a, nil)))
	if err != nil {
		t.Fatal(err)
	}
	o := res.Rows[0]
	var calls []eval.ToolCallRecord
	if err := json.Unmarshal(o.Annotations["tool_calls"], &calls); err != nil {
		t.Fatal(err)
	}
	if o.SubjectError != "" || string(o.Output) != `"done"` || o.Tokens != 5 || len(calls) != 2 ||
		calls[0].Name != "lookup" || calls[0].IsError || calls[0].Verdict != "allow" || calls[0].Result != "found" || !calls[1].IsError {
		t.Fatalf("agent row = %+v, calls %+v", o, calls)
	}
}

func TestDurableSubjectReplays(t *testing.T) {
	// WHY: an eval of 500 agent cases that dies at case 300 must not pay
	//      for 300 cases again. Each case and sample is its own durable run
	//      (ag.05): the second run of the suite calls the model zero times
	//      and gets the same outputs, and two samples of one case are two
	//      runs, not one replayed twice.
	// KIND: fault
	// CATCHES: s18
	// CHAPTER: ag.09 section 2.6
	store, err := durableagent.OpenFileStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	p := &script{replies: []reply{{text: []string{"answer"}}}}
	cfg := durableagent.Config{Agent: loop.Config{Provider: p}, Store: store}
	obs := []eval.Observation{{ID: "q/1", Input: str("one")}, {ID: "q 2", Input: str("two")}, {ID: "q3", Input: str("three")}}
	run := func() *eval.SuiteResult {
		res, err := eval.Run(context.Background(), "durable", obs, nil, eval.WithSubject("agent", eval.DurableSubject(cfg, "ev1", nil)), eval.WithSamples(2), eval.WithConcurrency(1))
		if err != nil {
			t.Fatal(err)
		}
		return res
	}
	first := run()
	if p.calls() != 6 {
		t.Fatalf("first run made %d model calls, want 6 (3 cases x 2 samples)", p.calls())
	}
	second := run()
	if p.calls() != 6 {
		t.Fatalf("second run made %d more model calls, want 0 (replayed)", p.calls()-6)
	}
	for i := range first.Rows {
		if first.Rows[i].SubjectError != "" || string(first.Rows[i].Output) != string(second.Rows[i].Output) {
			t.Fatalf("row %d: %+v vs %+v", i, first.Rows[i], second.Rows[i])
		}
	}
}

func TestRunID(t *testing.T) {
	// WHY: a durable run id is a file name: separators and spaces in case
	//      ids are replaced, very long ids are hashed, and every id is one
	//      ag.05 accepts; the same case and sample always map to the same
	//      run, different samples to different runs.
	// KIND: boundary
	// CATCHES: s21
	// CHAPTER: ag.09 section 2.6
	long := strings.Repeat("x", 300)
	ids := map[string]bool{}
	for _, c := range []struct {
		cs string
		s  int
	}{{"q/1", 0}, {"q/1", 1}, {"a b", 0}, {"../etc", 0}, {long, 0}, {long + "y", 0}} {
		id := eval.RunID("ev1", c.cs, c.s)
		if !durableagent.ValidRunID(id) || len(id) > 128 || ids[id] {
			t.Fatalf("RunID(%q, %d) = %q: invalid, too long, or a collision", c.cs, c.s, id)
		}
		ids[id] = true
		if eval.RunID("ev1", c.cs, c.s) != id {
			t.Fatal("RunID is not deterministic")
		}
	}
}

func TestMessages(t *testing.T) {
	// WHY: a case input is a prompt string or an OpenAI-shaped chat; both
	//      become the messages the subject sends, and anything else is
	//      ErrInput rather than an empty prompt.
	// KIND: unit
	// CATCHES: s22
	// CHAPTER: ag.09 section 2.1
	m, err := eval.Messages(str("hi"))
	if err != nil || len(m) != 1 || m[0].Role != types.RoleUser || m[0].Content != "hi" {
		t.Fatalf("string input: %+v %v", m, err)
	}
	m, err = eval.Messages(json.RawMessage(`{"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]}`))
	if err != nil || len(m) != 2 || m[0].Role != types.RoleSystem {
		t.Fatalf("chat input: %+v %v", m, err)
	}
	for _, bad := range []string{`42`, `{"messages": []}`, `{"messages": [{"role": "tool", "content": "x"}]}`, `{"prompt": "x"}`} {
		if _, err := eval.Messages(json.RawMessage(bad)); !errors.Is(err, eval.ErrInput) {
			t.Fatalf("Messages(%s) = %v, want ErrInput", bad, err)
		}
	}
}

func TestCancelStopsRun(t *testing.T) {
	// WHY: cancelling an eval (Ctrl-C, a workflow cancel) must stop it and
	//      say so; returning a result with half the cases missing would
	//      look like a complete, smaller suite.
	// KIND: fault
	// CATCHES: s20
	// CHAPTER: ag.09 section 4
	for _, c := range []struct{ n, workers int }{{50, 2}, {4, 4}} {
		ctx, cancel := context.WithCancel(context.Background())
		var obs []eval.Observation
		for i := 0; i < c.n; i++ {
			obs = append(obs, eval.Observation{ID: fmt.Sprint(i), Input: str("x")})
		}
		subject := func(ctx context.Context, o *eval.Observation) error {
			if o.ID == "3" {
				cancel()
			}
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(5 * time.Millisecond):
			}
			o.Output = num(1)
			return nil
		}
		res, err := eval.Run(ctx, "cancel", obs, []eval.Scorer{valueScorer{name: "v"}}, eval.WithSubject("s", subject), eval.WithConcurrency(c.workers))
		if !errors.Is(err, context.Canceled) || res != nil {
			t.Fatalf("%d cases, %d workers: Run after cancel = %v, %v; want nil, context.Canceled", c.n, c.workers, res, err)
		}
		cancel()
	}
}
