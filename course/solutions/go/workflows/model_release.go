// Package workflows holds the platform's durable workflows (Go only:
// workflows are deterministic code, Python implements activities, DESIGN
// 2.7). dur.12 owns model_release.go: ModelRelease, the gated path from a
// checkpoint to traffic.
//
//	preflight   model card and data ledger gates      activity release.preflight (non-retryable)
//	export      checkpoint -> models/<id>/<version>   activity export ({tinyllm} export --spec)
//	evaluate    EvalSuite on the exported model       dur.11 (activity eval per suite), then
//	            read its reports, then gate           activity release.results
//	approve     wait for the signal approve (reject)  durable signal wait with a timeout
//	canary      route a share of traffic to it        activities routes.snapshot, routes.canary
//	bake        a durable timer                       Sleep(canary_wait_s)
//	burn check  one PromQL query of the SLO burn      activity promql.query
//	decide      promote, or restore the old route     routes.promote | routes.restore
//
// The workflow is a function of dur.08's Runtime (runtime.go): the slice of
// the durable SDK it needs. Its evaluation step is dur.11's EvalSuite, run
// inline as a child step; the canary is undone by a dur.08 Saga when the run
// is canceled or a later step fails for good. The worker's composition root
// adapts a workflow.Context to Runtime and registers the activities under
// the Act* names:
//
//	w.RegisterWorkflow("ModelRelease", func(ctx workflow.Context, in []byte) ([]byte, error) {
//		var spec workflows.ReleaseSpec
//		if err := json.Unmarshal(in, &spec); err != nil { return nil, err }
//		res, err := workflows.ModelRelease(sdkRuntime{ctx}, spec)
//		if err != nil { return nil, err }
//		return json.Marshal(res)
//	})
//	w.RegisterActivity(workflows.ActPreflight, workflows.Preflight)
//	w.RegisterActivity(workflows.ActResults, workflows.ReadResults)
//	w.RegisterActivity(workflows.ActCanary, routes.Canary)   // routes := activities.Routes{...}
//
// Everything ModelRelease decides comes from its input and from recorded
// activity results, timers, and signals, so a replay after a crash takes the
// same path and re-executes nothing that completed.
package workflows

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"tinyllm/activities"
)

// Activity and signal names. The worker registers the activities under
// these names (the eval activity is dur.11's ActivityEval); `<system> wf
// signal <id> approve` sends the signal.
const (
	ActPreflight  = "release.preflight"
	ActExport     = "export"
	ActResults    = "release.results"
	ActSnapshot   = "routes.snapshot"
	ActCanary     = "routes.canary"
	ActPromote    = "routes.promote"
	ActRestore    = "routes.restore"
	ActQuery      = "promql.query"
	SignalApprove = "approve"
	SignalReject  = "reject"
)

// Outcomes of a release that ran to its end.
const (
	OutcomePromoted   = "promoted"
	OutcomeRolledBack = "rolled_back"
	OutcomeRejected   = "rejected"
	OutcomeExpired    = "expired" // no approve or reject before approve_timeout_s
)

// GateError fails the release before any traffic moves: a gate said no.
type GateError struct {
	Gate   string // spec, model_card, ledger, preflight, export, eval
	Reason string
}

func (e *GateError) Error() string      { return "release gate " + e.Gate + ": " + e.Reason }
func (e *GateError) NonRetryable() bool { return true }

// ReleaseSpec is ModelRelease's input (`{ctl} release --spec`).
type ReleaseSpec struct {
	ModelID         string    `json:"model_id"`          // tinystories-10m
	Version         string    `json:"version"`           // v1
	From            string    `json:"from"`              // checkpoint dir under /artifacts (ckpt/: LATEST)
	Dtype           string    `json:"dtype,omitempty"`   // export-spec dtype
	Quant           string    `json:"quant,omitempty"`   // export-spec quant
	ModelCard       string    `json:"model_card"`        // MODEL_CARD.md, relative to /artifacts or absolute
	Ledger          string    `json:"ledger"`            // corpus/LEDGER.jsonl
	Suites          []string  `json:"suites"`            // the eval suites EvalSuite runs on the exported model
	EvalSeed        int64     `json:"eval_seed"`         // their seed
	Gates           []Gate    `json:"gates"`             // thresholds on the eval rows
	Route           string    `json:"route"`             // the public model id whose traffic moves
	CanaryWeight    float64   `json:"canary_weight"`     // 0 < w < 1
	CanaryWaitS     int       `json:"canary_wait_s"`     // the bake time before the burn check
	ApproveTimeoutS int       `json:"approve_timeout_s"` // 0: wait for a human forever
	Burn            BurnCheck `json:"burn"`
}

// Gate is one threshold on an eval row of the released model.
type Gate struct {
	Suite  string  `json:"suite"`  // quality, zoo, safety, bias, ...
	Task   string  `json:"task"`   // the row's task
	Metric string  `json:"metric"` // the row's metric
	Op     string  `json:"op"`     // ">=" or "<="
	Value  float64 `json:"value"`
}

// BurnCheck is the SLO burn query of the canary and its ceiling.
type BurnCheck struct {
	Query string  `json:"query"` // aggregates to one number, the burn rate
	Max   float64 `json:"max"`   // roll back above it (14.4 is the 1 h page threshold for a 30-day budget)
}

// Activity payloads owned by this workflow.
type (
	PreflightInput struct {
		ModelCard string `json:"model_card"`
		Ledger    string `json:"ledger"`
		Root      string `json:"root,omitempty"` // relative paths resolve here (the worker's TL_ARTIFACTS)
	}
	PreflightResult struct {
		Sources int `json:"sources"` // ledger rows checked
	}
	// ExportResult is the export activity's DONE.json (spec/subprocess-activity.md).
	ExportResult struct {
		Outputs []string `json:"outputs"` // outputs[0]: the released model directory
	}
	// ResultsInput names the files EvalSuite's activities produced.
	ResultsInput struct {
		Outputs []string `json:"outputs"`
		Root    string   `json:"root,omitempty"` // relative paths resolve here (the worker's TL_ARTIFACTS)
	}
	EvalRow struct {
		Model  string   `json:"model"`
		Task   string   `json:"task"`
		Metric string   `json:"metric"`
		Value  *float64 `json:"value"`
		Status string   `json:"status"`
	}
	EvalReport struct {
		Suite string    `json:"suite"`
		Rows  []EvalRow `json:"rows"`
	}
	// EvalResult is release.results' answer: every formats/eval-results
	// report among the outputs.
	EvalResult struct {
		Reports []EvalReport `json:"reports"`
	}
	GateResult struct {
		Gate   Gate     `json:"gate"`
		Value  *float64 `json:"value"`
		OK     bool     `json:"ok"`
		Reason string   `json:"reason,omitempty"`
	}
)

// ReleaseResult is what a release that ran to its end returns.
type ReleaseResult struct {
	Outcome    string          `json:"outcome"`
	Served     string          `json:"served"`
	ModelDir   string          `json:"model_dir,omitempty"`
	Gates      []GateResult    `json:"gates,omitempty"`
	Approval   json.RawMessage `json:"approval,omitempty"`
	Burn       *float64        `json:"burn,omitempty"`
	BurnNoData bool            `json:"burn_no_data,omitempty"`
	Reason     string          `json:"reason,omitempty"`
}

// releaseOpts are the options of each step; the ID makes the activity's
// idempotency key "<workflow_id>/<ID>".
func releaseOpts(id string) StepOptions {
	switch id {
	case "release-export":
		return StepOptions{ID: id, StartToClose: 2 * time.Hour, HeartbeatTimeout: time.Minute, MaxAttempts: 5}
	case "release-preflight", "release-results":
		return StepOptions{ID: id, StartToClose: time.Minute, MaxAttempts: 3}
	case "promql-burn":
		return StepOptions{ID: id, StartToClose: 30 * time.Second, MaxAttempts: 5}
	default: // the route activities wait out ErrNoWorkers through their retries
		return StepOptions{ID: id, StartToClose: 30 * time.Second, MaxAttempts: 30}
	}
}

var idPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)

// Served is the model id the new engines report: <model_id>-<version>.
func (s ReleaseSpec) Served() string { return s.ModelID + "-" + s.Version }

// Validate checks the spec before anything runs.
func (s ReleaseSpec) Validate() error {
	var errs []string
	if !idPattern.MatchString(s.ModelID) || !idPattern.MatchString(s.Version) {
		errs = append(errs, "model_id and version must match ^[a-z0-9][a-z0-9._-]*$")
	}
	if s.From == "" || s.ModelCard == "" || s.Ledger == "" || s.Route == "" {
		errs = append(errs, "from, model_card, ledger, and route are required")
	}
	if !(s.CanaryWeight > 0 && s.CanaryWeight < 1) {
		errs = append(errs, fmt.Sprintf("canary_weight %v is not in (0, 1)", s.CanaryWeight))
	}
	if s.CanaryWaitS <= 0 {
		errs = append(errs, "canary_wait_s must be positive")
	}
	if s.ApproveTimeoutS < 0 {
		errs = append(errs, "approve_timeout_s must not be negative")
	}
	if strings.TrimSpace(s.Burn.Query) == "" || !(s.Burn.Max > 0) {
		errs = append(errs, "burn needs a query and a positive max")
	}
	if len(s.Suites) == 0 {
		errs = append(errs, "suites is required: the release is evaluated before it ships")
	}
	for i, g := range s.Gates {
		if g.Op != ">=" && g.Op != "<=" {
			errs = append(errs, fmt.Sprintf("gates[%d].op %q: want >= or <=", i, g.Op))
		}
		if g.Suite == "" || g.Task == "" || g.Metric == "" {
			errs = append(errs, fmt.Sprintf("gates[%d] needs suite, task, and metric", i))
		}
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

// ExportSpec is the export activity's input (formats/export-spec.schema.json).
func (s ReleaseSpec) ExportSpec() map[string]any {
	e := map[string]any{"from": s.From, "model_id": s.ModelID, "version": s.Version,
		"model_card": s.ModelCard, "ledger": s.Ledger}
	if s.Dtype != "" {
		e["dtype"] = s.Dtype
	}
	if s.Quant != "" {
		e["quant"] = s.Quant
	}
	return e
}

// EvalInput is EvalSuite's input: the spec's suites with the exported
// model as the only subject (id model_id).
func (s ReleaseSpec) EvalInput(modelDir string) EvalSuiteInput {
	return EvalSuiteInput{Name: "release-" + s.ModelID + "-" + s.Version, Tag: "release", Suites: s.Suites,
		Subjects: []EvalSubject{{ID: s.ModelID, Model: modelDir}}, Seed: s.EvalSeed}
}

// CheckGates evaluates every gate against the rows of the subject in the
// reports. A gate with no matching row, or whose row is not ok or has no
// value, fails: a suite that did not run proves nothing.
func CheckGates(gates []Gate, reports []EvalReport, subject string) []GateResult {
	out := make([]GateResult, 0, len(gates))
	for _, g := range gates {
		r := GateResult{Gate: g, Reason: "no row for this gate"}
		for _, rep := range reports {
			if rep.Suite != g.Suite {
				continue
			}
			for _, row := range rep.Rows {
				if row.Model != subject || row.Task != g.Task || row.Metric != g.Metric {
					continue
				}
				if row.Status != "ok" || row.Value == nil {
					r.Reason = "row status " + row.Status
					continue
				}
				v := *row.Value
				r.Value = &v
				switch g.Op {
				case ">=":
					r.OK = v >= g.Value
				case "<=":
					r.OK = v <= g.Value
				}
				r.Reason = ""
				if !r.OK {
					r.Reason = fmt.Sprintf("%s %s %s = %g, want %s %g", g.Suite, g.Task, g.Metric, v, g.Op, g.Value)
				}
			}
		}
		out = append(out, r)
	}
	return out
}

// modelCardHeadings are the sections of contracts/templates/MODEL_CARD.md.
var modelCardHeadings = []string{"## Model details", "## Intended use", "## Evaluation",
	"## Bias, risks, and limitations", "## Data"}

var placeholder = regexp.MustCompile(`<[A-Za-z][^<>\n]*>`)
var htmlComment = regexp.MustCompile(`(?s)<!--.*?-->`)

// ModelCardProblems lists what keeps a model card from gating a release:
// a missing template section, or a template placeholder (`<model_id>`,
// `<you>`) left in the text outside HTML comments.
func ModelCardProblems(text string) []string {
	var out []string
	if strings.TrimSpace(text) == "" {
		return []string{"the model card is empty"}
	}
	lines := map[string]bool{}
	for _, l := range strings.Split(text, "\n") {
		lines[strings.TrimSpace(l)] = true
	}
	for _, h := range modelCardHeadings {
		if !lines[h] {
			out = append(out, "missing section "+h)
		}
	}
	body := htmlComment.ReplaceAllString(text, "")
	if ph := placeholder.FindAllString(body, 3); len(ph) > 0 {
		out = append(out, "template placeholders left: "+strings.Join(ph, ", "))
	}
	return out
}

// LedgerProblems lists the ledger lines (formats/ledger.schema.json) that
// forbid training on their source: allowed_uses without "train", a revoked
// source, or a line that is not a ledger row. An empty ledger is a problem:
// a model trained on undocumented data cannot ship.
func LedgerProblems(ledger []byte) []string {
	var out []string
	n := 0
	for i, line := range strings.Split(string(ledger), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		n++
		var row struct {
			SourceID    *string  `json:"source_id"`
			License     string   `json:"license_spdx"`
			AllowedUses []string `json:"allowed_uses"`
			Revoked     bool     `json:"revoked"`
		}
		if err := json.Unmarshal([]byte(line), &row); err != nil || row.SourceID == nil {
			out = append(out, fmt.Sprintf("line %d: not a ledger row", i+1))
			continue
		}
		train := false
		for _, u := range row.AllowedUses {
			train = train || u == "train"
		}
		if !train {
			out = append(out, fmt.Sprintf("line %d: source %s (%s) does not allow train", i+1, *row.SourceID, row.License))
		}
		if row.Revoked {
			out = append(out, fmt.Sprintf("line %d: source %s is revoked", i+1, *row.SourceID))
		}
	}
	if n == 0 {
		out = append(out, "the ledger has no sources")
	}
	return out
}

// Preflight is the release.preflight activity: the model card and ledger
// gates. A problem is a *GateError (non-retryable: rereading the same file
// gives the same answer); a file that cannot be read is retryable.
func Preflight(ctx context.Context, in PreflightInput) (PreflightResult, error) {
	resolve := func(p string) string {
		if filepath.IsAbs(p) || in.Root == "" {
			return p
		}
		return filepath.Join(in.Root, p)
	}
	card, err := os.ReadFile(resolve(in.ModelCard))
	if errors.Is(err, os.ErrNotExist) {
		return PreflightResult{}, &GateError{Gate: "model_card", Reason: "no model card at " + in.ModelCard}
	}
	if err != nil {
		return PreflightResult{}, err
	}
	if p := ModelCardProblems(string(card)); len(p) > 0 {
		return PreflightResult{}, &GateError{Gate: "model_card", Reason: strings.Join(p, "; ")}
	}
	ledger, err := os.ReadFile(resolve(in.Ledger))
	if errors.Is(err, os.ErrNotExist) {
		return PreflightResult{}, &GateError{Gate: "ledger", Reason: "no data ledger at " + in.Ledger}
	}
	if err != nil {
		return PreflightResult{}, err
	}
	if p := LedgerProblems(ledger); len(p) > 0 {
		return PreflightResult{}, &GateError{Gate: "ledger", Reason: strings.Join(p, "; ")}
	}
	n := 0
	for _, l := range strings.Split(string(ledger), "\n") {
		if strings.TrimSpace(l) != "" {
			n++
		}
	}
	return PreflightResult{Sources: n}, nil
}

// ReadResults is the release.results activity: every output of EvalSuite
// that is a formats/eval-results report (a .json file whose format is
// tl.eval-results.v1), in the order given. Other outputs (summaries,
// results.jsonl) are skipped. A file that cannot be read is retryable.
func ReadResults(ctx context.Context, in ResultsInput) (EvalResult, error) {
	var out EvalResult
	for _, p := range in.Outputs {
		if !strings.HasSuffix(p, ".json") {
			continue
		}
		if !filepath.IsAbs(p) && in.Root != "" {
			p = filepath.Join(in.Root, p)
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return EvalResult{}, err
		}
		var rep struct {
			Format string    `json:"format"`
			Suite  string    `json:"suite"`
			Rows   []EvalRow `json:"rows"`
		}
		if json.Unmarshal(b, &rep) != nil || rep.Format != "tl.eval-results.v1" {
			continue
		}
		out.Reports = append(out.Reports, EvalReport{Suite: rep.Suite, Rows: rep.Rows})
	}
	return out, nil
}

// gateFrom turns a preflight that failed for good into a gate refusal; the
// activity's message names the gate ("release gate ledger: ...").
func gateFrom(err error) error {
	var f *StepFailure
	if errors.As(err, &f) && f.NonRetryable {
		return &GateError{Gate: "preflight", Reason: f.Message}
	}
	return err
}

// ModelRelease runs one release to its end. It returns an error when a
// gate refuses the release (*GateError), when an activity it cannot do
// without fails for good, or when the run is canceled (ErrCanceled); once
// traffic has moved, every one of those first puts the old route back.
// Otherwise it returns a result whose Outcome says what happened.
func ModelRelease(rt Runtime, spec ReleaseSpec) (ReleaseResult, error) {
	res := ReleaseResult{Served: spec.Served()}
	if err := spec.Validate(); err != nil {
		return res, &GateError{Gate: "spec", Reason: err.Error()}
	}

	// 1. Gates that need no model: before any expensive work.
	var pre PreflightResult
	if err := rt.ExecuteActivity(ActPreflight, PreflightInput{ModelCard: spec.ModelCard, Ledger: spec.Ledger},
		releaseOpts("release-preflight"), &pre); err != nil {
		return res, gateFrom(err)
	}

	// 2. Export, 3. evaluate the exported model (EvalSuite), then gate on its rows.
	var ex ExportResult
	if err := rt.ExecuteActivity(ActExport, spec.ExportSpec(), releaseOpts("release-export"), &ex); err != nil {
		return res, err
	}
	if len(ex.Outputs) == 0 {
		return res, &GateError{Gate: "export", Reason: "export produced no model directory"}
	}
	res.ModelDir = ex.Outputs[0]
	suites, err := EvalSuite(rt, spec.EvalInput(res.ModelDir))
	if err != nil {
		return res, err
	}
	var outputs []string
	for _, so := range suites.Suites {
		outputs = append(outputs, so.Outputs...)
	}
	var ev EvalResult
	if err := rt.ExecuteActivity(ActResults, ResultsInput{Outputs: outputs}, releaseOpts("release-results"), &ev); err != nil {
		return res, err
	}
	res.Gates = CheckGates(spec.Gates, ev.Reports, spec.ModelID)
	var failed []string
	for _, g := range res.Gates {
		if !g.OK {
			failed = append(failed, g.Gate.Suite+"/"+g.Gate.Task+"/"+g.Gate.Metric+": "+g.Reason)
		}
	}
	if len(failed) > 0 {
		return res, &GateError{Gate: "eval", Reason: strings.Join(failed, "; ")}
	}

	// 4. A human approves (or rejects) before any traffic moves.
	name, payload, err := rt.AwaitSignal([]string{SignalApprove, SignalReject},
		time.Duration(spec.ApproveTimeoutS)*time.Second)
	if err != nil {
		return res, err
	}
	switch name {
	case SignalApprove:
		res.Approval = payload
	case SignalReject:
		res.Outcome, res.Approval = OutcomeRejected, payload
		return res, nil
	default:
		res.Outcome, res.Reason = OutcomeExpired, "no approve or reject signal before approve_timeout_s"
		return res, nil
	}

	// 5. Canary: remember the route as it is, register its restore, then
	// move a share of traffic. From here every way out but a promotion puts
	// the snapshot back.
	var snap activities.RouteSnapshot
	if err := rt.ExecuteActivity(ActSnapshot, activities.SnapshotInput{Route: spec.Route}, releaseOpts("routes-snapshot"), &snap); err != nil {
		return res, err
	}
	var saga Saga
	saga.Add("restore the route", ActRestore, activities.RestoreInput{Route: spec.Route, Previous: snap.Route},
		releaseOpts("routes-restore"))
	var canary activities.RouteResult
	if err := rt.ExecuteActivity(ActCanary, activities.CanaryInput{Route: spec.Route, Served: spec.Served(),
		Weight: spec.CanaryWeight}, releaseOpts("routes-canary"), &canary); err != nil {
		return res, saga.Fail(rt, err)
	}

	// 6. Bake on a durable timer, 7. one burn-rate query at the end of it.
	if err := rt.Sleep(time.Duration(spec.CanaryWaitS) * time.Second); err != nil {
		return res, saga.Fail(rt, err)
	}
	var s activities.Sample
	qerr := rt.ExecuteActivity(ActQuery, activities.QueryInput{Expr: spec.Burn.Query, At: rt.Now()}, releaseOpts("promql-burn"), &s)
	if IsCanceled(qerr) {
		return res, saga.Fail(rt, qerr)
	}
	healthy := false
	switch {
	case qerr != nil:
		res.Reason = "burn query failed: " + qerr.Error()
	case s.NoData:
		res.BurnNoData, res.Reason = true, "the burn query returned no data"
	default:
		v := s.Value
		res.Burn = &v
		healthy = v <= spec.Burn.Max
		if !healthy {
			res.Reason = fmt.Sprintf("burn rate %g above %g", v, spec.Burn.Max)
		}
	}

	// 8. Promote, or put the old route back.
	if healthy {
		var done activities.RouteResult
		if err := rt.ExecuteActivity(ActPromote, activities.PromoteInput{Route: spec.Route, Served: spec.Served()},
			releaseOpts("routes-promote"), &done); err != nil {
			return res, saga.Fail(rt, err)
		}
		res.Outcome = OutcomePromoted
		return res, nil
	}
	if err := saga.Compensate(rt); err != nil {
		return res, err
	}
	res.Outcome = OutcomeRolledBack
	return res, nil
}
