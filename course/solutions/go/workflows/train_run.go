package workflows

// dur.11 owns train_run.go: TrainRun, a training run as a durable
// workflow. Training is a subprocess activity (dur.09): `{tinyllm} train
// --spec` checkpoints atomically (L0.6), its runner heartbeats each
// checkpoint, and a retry resumes from it. The workflow cuts the run into
// segments of eval_every steps; after each segment it evaluates that
// segment's checkpoint (EvalSuite), so evaluation happens at the same steps
// on every run and each evaluation runs once.
//
//	[corpus]   CorpusBuild first, when the input carries a corpus config (data.09)
//	train      steps 0 .. e          activity train-000500   -> runs/<name>/ckpt/step-000500
//	eval       that checkpoint       activities eval-000500-<suite>
//	train      steps e .. 2e         activity train-001000   (resumes from runs/<name>/ckpt/LATEST)
//	...
//
// A kill of the worker, the server, or both, at any point, resumes the run:
// a finished segment replays from history; the segment in flight is
// redelivered with its last checkpoint as --resume.

import (
	"encoding/json"
	"fmt"
	"time"
)

// Activity names the worker registers for TrainRun and EvalSuite.
const (
	ActivityTrain = "train" // {tinyllm} train --spec <dir>/spec.json --progress ... [--resume <ckpt>]
	ActivityEval  = "eval"  // {tinyllm} eval --spec <dir>/spec.json --progress ...
)

// TrainRunInput is TrainRun's input.
type TrainRunInput struct {
	Spec       json.RawMessage `json:"spec"`                  // formats/train-spec.schema.json
	Corpus     json.RawMessage `json:"corpus,omitempty"`      // a corpus config to build first (optional)
	EvalSuites []string        `json:"eval_suites,omitempty"` // suites run on every segment's checkpoint
	EvalSeed   int64           `json:"eval_seed,omitempty"`
}

// TrainRunResult is what a finished run returns.
type TrainRunResult struct {
	Name       string             `json:"name"`
	Steps      int                `json:"steps"`
	Checkpoint string             `json:"checkpoint"` // the final step directory
	Segments   []Segment          `json:"segments"`
	Corpus     *CorpusBuildResult `json:"corpus,omitempty"`
}

// Segment is one train activity and the evaluations of its checkpoint.
type Segment struct {
	Until      int              `json:"until"`
	Checkpoint string           `json:"checkpoint"`
	Evals      *EvalSuiteResult `json:"evals,omitempty"`
}

type trainSpec struct {
	Name                string `json:"name"`
	Steps               int    `json:"steps"`
	EvalEvery           int    `json:"eval_every"`
	EvalIntervalSeconds int    `json:"eval_interval_seconds"`
}

// SpecError is a spec the workflow refuses before any activity runs.
type SpecError struct{ Reason string }

func (e *SpecError) Error() string      { return "train run: " + e.Reason }
func (e *SpecError) NonRetryable() bool { return true }

// Segments are the step counts each train activity runs to: eval_every,
// 2 * eval_every, ..., ending exactly at steps (the last one may be
// shorter). eval_every 0 means one segment.
func Segments(steps, evalEvery int) []int {
	if evalEvery <= 0 || evalEvery >= steps {
		return []int{steps}
	}
	var out []int
	for s := evalEvery; s < steps; s += evalEvery {
		out = append(out, s)
	}
	return append(out, steps)
}

// TrainOptions are a train segment's activity options: the segment's end
// step in the activity id ("train-000500"), a heartbeat timeout (the runner
// heartbeats every checkpoint and every third of it), a day to finish.
func TrainOptions(until int) StepOptions {
	return StepOptions{ID: fmt.Sprintf("train-%06d", until), StartToClose: 24 * time.Hour, HeartbeatTimeout: time.Minute, MaxAttempts: 5}
}

// withField returns obj (a JSON object) with key set to v.
func withField(obj json.RawMessage, key string, v any) (json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(obj, &m); err != nil {
		return nil, err
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	m[key] = b
	return json.Marshal(m)
}

// TrainRun builds the corpus when asked, then trains segment by segment and
// evaluates each segment's checkpoint. A cancel ends it with an error
// matching ErrCanceled: the train activity checkpoints on its way out
// (dur.09) and nothing is deleted.
func TrainRun(rt Runtime, in TrainRunInput) (TrainRunResult, error) {
	var spec trainSpec
	if err := json.Unmarshal(in.Spec, &spec); err != nil {
		return TrainRunResult{}, &SpecError{Reason: "spec is not a JSON object: " + err.Error()}
	}
	if spec.Name == "" || spec.Steps < 1 || spec.EvalEvery < 0 || spec.EvalIntervalSeconds < 0 {
		return TrainRunResult{}, &SpecError{Reason: fmt.Sprintf("spec needs a name, steps >= 1, and non-negative evaluation intervals (name %q, steps %d, eval_every %d, eval_interval_seconds %d)", spec.Name, spec.Steps, spec.EvalEvery, spec.EvalIntervalSeconds)}
	}
	res := TrainRunResult{Name: spec.Name, Steps: spec.Steps}
	if len(in.Corpus) > 0 {
		cb, err := CorpusBuild(rt, CorpusBuildInput{Config: in.Corpus})
		if err != nil {
			return res, fmt.Errorf("corpus: %w", err)
		}
		res.Corpus = &cb
	}
	segments := Segments(spec.Steps, spec.EvalEvery)
	for i, until := range segments {
		seg, err := withField(in.Spec, "steps", until)
		if err != nil {
			return res, &SpecError{Reason: err.Error()}
		}
		var done StageDone
		if err := rt.ExecuteActivity(ActivityTrain, json.RawMessage(seg), TrainOptions(until), &done); err != nil {
			return res, fmt.Errorf("train to step %d: %w", until, err)
		}
		if len(done.Outputs) == 0 {
			return res, fmt.Errorf("train to step %d: no checkpoint in the result", until)
		}
		s := Segment{Until: until, Checkpoint: done.Outputs[0]}
		if len(in.EvalSuites) > 0 {
			ev, err := EvalSuite(rt, EvalSuiteInput{
				Tag: fmt.Sprintf("%06d", until), Suites: in.EvalSuites, Seed: in.EvalSeed,
				Subjects: []EvalSubject{{ID: fmt.Sprintf("%s@%d", spec.Name, until), Model: s.Checkpoint}},
			})
			if err != nil {
				return res, fmt.Errorf("eval at step %d: %w", until, err)
			}
			s.Evals = &ev
			if spec.EvalIntervalSeconds > 0 && i+1 < len(segments) {
				if err := rt.Sleep(time.Duration(spec.EvalIntervalSeconds) * time.Second); err != nil {
					return res, fmt.Errorf("eval cadence after step %d: %w", until, err)
				}
			}
		}
		res.Segments = append(res.Segments, s)
		res.Checkpoint = s.Checkpoint
	}
	return res, nil
}
