package workflows

// data.09 owns corpus_build.go: CorpusBuild, the corpus pipeline of data.01
// to data.08 as a durable workflow. Each stage is a subprocess activity
// (dur.09) that runs `{corpus} run --stage <s>`; the workflow only orders
// them, keys them, and undoes a build that is cancelled or fails.
//
//	fetch     raw documents and ledger rows        corpus/raw/, corpus/LEDGER.jsonl
//	shard     filter, dedup, PII, Parquet shards   corpus/<dataset>/<version>/
//	tokenize  .bin token streams                   tokens/<tokenizer>/<dataset>/
//
// The stages are the pipeline's on-disk boundaries: the streaming stages in
// between (filters, dedup, PII) hand documents to each other in memory and
// are not worth a round trip through the durable server each.
//
// Idempotency: the workflow id names the dataset, the version, and the
// config's hash (CorpusWorkflowID), and each stage's activity id is the
// stage name, so the key "<workflow id>/<stage>" is the same on every
// attempt and different for every config. A changed config is a different
// build, never a silent rerun of an old one.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// Activity names the worker registers for CorpusBuild.
const (
	ActCorpusStage   = "corpus.stage"   // {corpus} run --stage <s> --spec ... --progress ...
	ActCorpusCleanup = "corpus.cleanup" // {corpus} cleanup --spec ...: the compensation
)

// CorpusStages are the activities of a build, in order.
var CorpusStages = []string{"fetch", "shard", "tokenize"}

// CorpusBuildInput is the workflow's input: the corpus config
// (formats/corpus-config.schema.json) as JSON, the way the runner hands it
// to Python.
type CorpusBuildInput struct {
	Config json.RawMessage `json:"config"`
}

// StageInput is one stage activity's input (the spec.json the runner writes).
type StageInput struct {
	Stage        string          `json:"stage"`
	ConfigSHA256 string          `json:"config_sha256"`
	Config       json.RawMessage `json:"config"`
}

// CleanupInput names what a cancelled or failed build leaves behind.
type CleanupInput struct {
	Dataset      string `json:"dataset"`
	Version      string `json:"version"`
	TokenizerID  string `json:"tokenizer_id"`
	ConfigSHA256 string `json:"config_sha256"`
}

// StageDone is a stage activity's result: the bytes of its DONE.json.
type StageDone struct {
	Outputs []string `json:"outputs"`
}

// CorpusBuildResult is what a finished build returns.
type CorpusBuildResult struct {
	Dataset      string              `json:"dataset"`
	Version      string              `json:"version"`
	ConfigSHA256 string              `json:"config_sha256"`
	Manifest     string              `json:"manifest"` // corpus/<dataset>/<version>/_MANIFEST.json
	Tokens       []string            `json:"tokens"`   // tokens/<tokenizer>/<dataset>/...
	Stages       map[string][]string `json:"stages"`   // every stage's outputs
}

type corpusConfig struct {
	Dataset   string `json:"dataset"`
	Version   string `json:"version"`
	Tokenizer struct {
		ID string `json:"id"`
	} `json:"tokenizer"`
}

// ConfigError is a config the workflow refuses before any activity runs.
type ConfigError struct{ Reason string }

func (e *ConfigError) Error() string      { return "corpus build: " + e.Reason }
func (e *ConfigError) NonRetryable() bool { return true }

// ConfigSHA256 hashes a config by its content, not its spelling: the JSON
// is decoded and re-encoded (object keys sorted, no whitespace), so the same
// config written two ways is one build.
func ConfigSHA256(cfg json.RawMessage) (string, error) {
	var v any
	if err := json.Unmarshal(cfg, &v); err != nil {
		return "", &ConfigError{Reason: "config is not JSON: " + err.Error()}
	}
	canon, err := json.Marshal(v) // encoding/json sorts map keys
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canon)
	return hex.EncodeToString(sum[:]), nil
}

func parseCorpusConfig(cfg json.RawMessage) (corpusConfig, error) {
	var c corpusConfig
	if err := json.Unmarshal(cfg, &c); err != nil {
		return c, &ConfigError{Reason: "config is not a JSON object: " + err.Error()}
	}
	// A slice, not a map: workflow code never ranges over a map (the order,
	// and so this error, would change between a run and its replay).
	for _, f := range [][2]string{{"dataset", c.Dataset}, {"version", c.Version}, {"tokenizer.id", c.Tokenizer.ID}} {
		if f[1] == "" {
			return c, &ConfigError{Reason: "config has no " + f[0]}
		}
	}
	return c, nil
}

// CorpusWorkflowID is the workflow id of a build: corpus/<dataset>/<version>/
// followed by the first 12 hex digits of the config hash. Starting the same
// config twice finds the same run (StartWorkflow is idempotent on the id).
func CorpusWorkflowID(cfg json.RawMessage) (string, error) {
	c, err := parseCorpusConfig(cfg)
	if err != nil {
		return "", err
	}
	sha, err := ConfigSHA256(cfg)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("corpus/%s/%s/%s", c.Dataset, c.Version, sha[:12]), nil
}

// StageOptions are a stage's activity options: the stage name as activity
// id, a heartbeat timeout (the runner heartbeats every third of it), and a
// start-to-close long enough for the real corpus.
func StageOptions(stage string) StepOptions {
	o := StepOptions{ID: stage, HeartbeatTimeout: 2 * time.Minute, StartToClose: 2 * time.Hour, MaxAttempts: 5}
	if stage == "fetch" {
		o.MaxAttempts = 8 // flaky networks: more retries, each cheap (data.01 resumes)
	}
	return o
}

// CorpusBuild runs the stages in order. Before the first stage that writes
// the build's own outputs (shard), it registers the compensation that
// deletes them, so a cancel or a failure from then on leaves no partial
// shards or token files behind (fetched raw documents are a shared cache
// and stay). A cancel returns an error matching ErrCanceled after the
// compensation ran; any other failure returns the stage's error.
func CorpusBuild(rt Runtime, in CorpusBuildInput) (CorpusBuildResult, error) {
	c, err := parseCorpusConfig(in.Config)
	if err != nil {
		return CorpusBuildResult{}, err
	}
	sha, err := ConfigSHA256(in.Config)
	if err != nil {
		return CorpusBuildResult{}, err
	}
	res := CorpusBuildResult{Dataset: c.Dataset, Version: c.Version, ConfigSHA256: sha, Stages: map[string][]string{}}
	var saga Saga
	for _, stage := range CorpusStages {
		if stage == "shard" {
			saga.Add("delete the build's partial outputs", ActCorpusCleanup,
				CleanupInput{Dataset: c.Dataset, Version: c.Version, TokenizerID: c.Tokenizer.ID, ConfigSHA256: sha},
				StepOptions{ID: "cleanup", StartToClose: 10 * time.Minute, MaxAttempts: 10})
		}
		var done StageDone
		if err := rt.ExecuteActivity(ActCorpusStage, StageInput{Stage: stage, ConfigSHA256: sha, Config: in.Config}, StageOptions(stage), &done); err != nil {
			return res, saga.Fail(rt, fmt.Errorf("stage %s: %w", stage, err))
		}
		res.Stages[stage] = done.Outputs
	}
	if out := res.Stages["shard"]; len(out) > 0 {
		res.Manifest = out[0]
	}
	res.Tokens = res.Stages["tokenize"]
	if res.Manifest == "" {
		return res, errors.New("corpus build: the shard stage reported no manifest")
	}
	return res, nil
}
