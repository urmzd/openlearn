package workflows

// dur.11 owns eval_suite.go: EvalSuite, model evaluations as a durable
// workflow. Each suite is one subprocess activity, `{tinyllm} eval --spec`
// (formats/eval-spec.schema.json) with that one suite, so a crash in the
// middle of the zoo suite does not rerun the finished ppl suite, and every
// suite's results land in evals/<suite>/<run>/ exactly once. TrainRun calls
// it after each training segment; ModelRelease (dur.12) calls it as its
// evaluation step.

import (
	"encoding/json"
	"fmt"
	"time"
)

// EvalSubject is one model under evaluation (an eval-spec subject).
type EvalSubject struct {
	ID    string `json:"id"`
	Model string `json:"model,omitempty"` // a model or checkpoint directory under /artifacts
}

// EvalSuiteInput is EvalSuite's input.
type EvalSuiteInput struct {
	Name       string             `json:"name,omitempty"` // eval-spec name; default "evalsuite"
	Tag        string             `json:"tag,omitempty"`  // part of every activity id: TrainRun uses the step
	Suites     []string           `json:"suites"`
	Subjects   []EvalSubject      `json:"subjects"`
	Seed       int64              `json:"seed"`
	Scorers    []string           `json:"scorers,omitempty"`
	Judge      *EvalJudge         `json:"judge,omitempty"`
	AB         *EvalAB            `json:"ab,omitempty"`
	Thresholds map[string]float64 `json:"thresholds,omitempty"`
}

type EvalJudge struct {
	BaseURL  string `json:"base_url"`
	Model    string `json:"model"`
	Rubric   string `json:"rubric"`
	Samples  int    `json:"samples,omitempty"`
	Pairwise bool   `json:"pairwise,omitempty"`
}

type EvalAB struct {
	Base  string  `json:"base"`
	Exp   string  `json:"exp"`
	NBoot int     `json:"n_boot,omitempty"`
	Alpha float64 `json:"alpha,omitempty"`
}

// EvalSuiteResult maps each suite to its eval activity's outputs
// (evals/<suite>/<run>/summary.json first).
type EvalSuiteResult struct {
	Suites []SuiteOutputs `json:"suites"`
}

// SuiteOutputs is one suite's outputs.
type SuiteOutputs struct {
	Suite   string   `json:"suite"`
	Outputs []string `json:"outputs"`
}

// evalSpec is the spec.json one eval activity gets.
type evalSpec struct {
	Name       string             `json:"name"`
	Suites     []string           `json:"suites"`
	Subjects   []EvalSubject      `json:"subjects"`
	Seed       int64              `json:"seed"`
	Scorers    []string           `json:"scorers,omitempty"`
	Judge      *EvalJudge         `json:"judge,omitempty"`
	AB         *EvalAB            `json:"ab,omitempty"`
	Thresholds map[string]float64 `json:"thresholds,omitempty"`
}

// EvalOptions are one suite's activity options: "eval-<tag>-<suite>" as
// activity id ("eval-<suite>" without a tag).
func EvalOptions(tag, suite string) StepOptions {
	id := "eval-" + suite
	if tag != "" {
		id = fmt.Sprintf("eval-%s-%s", tag, suite)
	}
	return StepOptions{ID: id, StartToClose: 6 * time.Hour, HeartbeatTimeout: 2 * time.Minute, MaxAttempts: 3}
}

// EvalSuite runs the suites one activity each, in the order given, and
// fails with the first suite that fails for good.
func EvalSuite(rt Runtime, in EvalSuiteInput) (EvalSuiteResult, error) {
	if len(in.Suites) == 0 || len(in.Subjects) == 0 {
		return EvalSuiteResult{}, &SpecError{Reason: "an eval suite needs suites and subjects"}
	}
	name := in.Name
	if name == "" {
		name = "evalsuite"
	}
	var res EvalSuiteResult
	for _, suite := range in.Suites {
		spec := evalSpec{Name: name, Suites: []string{suite}, Subjects: in.Subjects, Seed: in.Seed}
		applyAdvancedSpecOptions(&spec, in)
		var done StageDone
		if err := rt.ExecuteActivity(ActivityEval, spec, EvalOptions(in.Tag, suite), &done); err != nil {
			return res, fmt.Errorf("suite %s: %w", suite, err)
		}
		res.Suites = append(res.Suites, SuiteOutputs{Suite: suite, Outputs: done.Outputs})
	}
	return res, nil
}

// MarshalSpec is the eval-spec JSON EvalSuite gives one suite's activity
// (exported for the worker's tests and the CLI's dry run).
func MarshalSpec(in EvalSuiteInput, suite string) ([]byte, error) {
	name := in.Name
	if name == "" {
		name = "evalsuite"
	}
	spec := evalSpec{Name: name, Suites: []string{suite}, Subjects: in.Subjects, Seed: in.Seed}
	applyAdvancedSpecOptions(&spec, in)
	return json.Marshal(spec)
}

// applyAdvancedSpecOptions adds the judge and A/B options to a durable
// suite's per-suite subprocess request.
func applyAdvancedSpecOptions(spec *evalSpec, in EvalSuiteInput) {
	spec.Scorers, spec.Judge, spec.AB, spec.Thresholds = in.Scorers, in.Judge, in.AB, in.Thresholds
}
