# Pipeline as a durable workflow CorpusBuild

## Overview

| | |
|---|---|
| **Module** | `data.09` · build · Go · Pass 8 · 3 to 4 h |
| **Solution** | [`go/workflows/corpus_build.go`](../../../../course/solutions/go/workflows/corpus_build.go): `CorpusBuild(rt, in)`, `ConfigSHA256`, `CorpusWorkflowID`, `StageOptions`, the activity names `corpus.stage` and `corpus.cleanup`; wired in the worker ([`entry/go/cmd/worker/main.go`](../../../../course/solutions/entry/go/cmd/worker/main.go)) and `ctl data build` ([`entry/go/cmd/ctl/data.go`](../../../../course/solutions/entry/go/cmd/ctl/data.go)) |
| **Contract** | [`course/contracts/formats/corpus-config.schema.json`](../../../../course/contracts/formats/corpus-config.schema.json) (the stage spec) and [`course/contracts/spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) (how a stage runs) |
| **Needs** | `dur.08` (the `Runtime` seam and `Saga`); reading: `data.08` (the pipeline it wraps, `data.01` to `data.08`), `dur.09` (the subprocess runner the stages run through) |
| **Used by** | `dur.11` (TrainRun builds its corpus first); later `C1`, which trains on a corpus that `ctl data build` produced and that survives worker kills with the same manifest hash |
| **Optional depth** | [datatrove: pipelines and executors](https://github.com/huggingface/datatrove) (free); [Temporal: idempotency of activities](https://docs.temporal.io/activity-definition#idempotency) (free) |

## Key Takeaways

- The workflow **orders, keys, and undoes**; the stages do the work. Each stage is one subprocess activity at an **on-disk boundary** of the pipeline (`fetch`, `shard`, `tokenize`).
- The **workflow id carries the config's content hash** and each stage's activity id is its name, so the idempotency key `corpus/<dataset>/<version>/<hash12>/<stage>` is stable across attempts and different for every config.
- The hash is of the config's **content**, not its spelling: decoded and re-encoded with sorted keys.
- A kill at any point gives the **same result and files** as an uninterrupted build; a finished stage is replayed from history, a stage in flight reruns as a no-op or a whole rewrite.
- **Cancel or failure deletes the build's partial outputs** (a compensation registered before the shard stage), never the shared raw-document cache.

## How to read this chapter

Sections 1 to 3 derive where the pipeline is cut into activities, how the keys come from the config's content, and what kills and cancels do, and work the keys of one build by hand. Section 4 fixes the interface and lists the cases a correct solution must handle. Section 5 walks through `corpus_build.go` and its wiring in the worker and `ctl`, and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The workflow is wired into the solution's entry points. The worker ([`entry/go/cmd/worker/main.go`](../../../../course/solutions/entry/go/cmd/worker/main.go)) registers `CorpusBuild` and two subprocess activities over the corpus CLI: `corpus.stage` runs `run --stage <in.Stage>` with `in.Config` as the spec, and `corpus.cleanup` runs `cleanup` with its input as the spec. `ctl data build --config <toml> --id <id> [--wait]` ([`entry/go/cmd/ctl/data.go`](../../../../course/solutions/entry/go/cmd/ctl/data.go)) parses the TOML and starts `CorpusBuild`; the id to pass is `CorpusWorkflowID(config)`, so that the same config always names the same run.

---

## 1. Why now

The corpus pipeline (`data.01` to `data.08`) is one command: `{corpus} run --config small.toml`. On the real TinyStories corpus it runs for an hour, downloads gigabytes, and dies with the laptop's battery or a flaky mirror; a rerun starts from scratch, and a run stopped halfway leaves shards that the next training run reads as a complete corpus. The project now has a durable engine (`dur.01` to `dur.08`) and a way to run Python as activities (`dur.09`). `CorpusBuild` puts them together: each stage becomes an activity with a stable idempotency key, a kill anywhere resumes, and a cancel cleans up after itself.

## 2. Principles

### 2.1 Stages at on-disk boundaries

A durable workflow pays a round trip through the server (history events, a task, a poll) for every activity, and only work that writes its result somewhere durable can be resumed. The pipeline's stages pass documents to each other in memory (`compose` of generator stages, `data.02`), so CorpusBuild cuts it where the outputs are files:

| Activity `corpus.stage` | Runs | Writes |
|---|---|---|
| `fetch` | `run --stage fetch` | `corpus/raw/`, `corpus/LEDGER.jsonl` (a cache shared by every build) |
| `shard` | `run --stage shard` (filters, dedup, PII, shards, ledger check) | `corpus/<dataset>/<version>/` with `_MANIFEST.json` |
| `tokenize` | `run --stage tokenize` (from the manifest) | `tokens/<tokenizer>/<dataset>/` |

### 2.2 Keys from content

| Symbol | Meaning |
|---|---|
| $c$ | the corpus config (JSON) |
| $\text{canon}(c)$ | $c$ decoded and re-encoded: object keys sorted, no whitespace |
| $h = \text{sha256}(\text{canon}(c))$ | the config hash, 64 hex digits |
| $w$ = `corpus/<dataset>/<version>/` + $h_{0..11}$ | the workflow id |
| $k_s = w$ + `/` + $s$ | stage $s$'s idempotency key |

`StartWorkflow` is idempotent on the workflow id (`dur.02`): starting the same config again finds the run that exists, and a changed config, even one changed value, is a different run with different keys, so it can never be mistaken for a rerun of the old one. Hashing the canonical form means a config with other whitespace or key order is the same build.

### 2.3 Kills: replay or rerun

A stage that finished is in the run's history: a replay answers it without running anything. A stage that was running when the worker died is redelivered with the same key and runs again in the same work directory, where `DONE.json` (`dur.09`) makes it a no-op, or, without one, the stage rewrites its outputs whole (every file is published by rename). Either way the build's result, and every file it wrote, equals an uninterrupted build's.

### 2.4 Cancel and failure: compensate

The compensation `corpus.cleanup` deletes `corpus/<dataset>/<version>/` and `tokens/<tokenizer>/<dataset>/`. It is registered with the saga **before** the shard stage (a shard stage interrupted halfway may have left files) and not before fetch (a fetch that failed left nothing of this build, and deleting would destroy a previous good build of the same version). It runs once, on the detached runtime, whether the build was cancelled (`ErrCanceled`) or a stage failed for good (an unlicensed source exits 65).

## 3. Worked example by hand

Config `{"version": "v1", "dataset": "tinystories", "tokenizer": {"id": "bytes"}}`. Its canonical form is the 67 bytes

```
{"dataset":"tinystories","tokenizer":{"id":"bytes"},"version":"v1"}
```

whose sha256 is `a03711ec959b2a8fb6cf51d9445bee90fcb69a103cfa033483b52f836269c0a6`. So the workflow id is `corpus/tinystories/v1/a03711ec959b`, and the three activities run in order with keys `.../a03711ec959b/fetch`, `/shard`, `/tokenize`, each with the stage name as activity id, a 2-minute heartbeat timeout, and (for fetch) more attempts. The result names the manifest `corpus/tinystories/v1/_MANIFEST.json` and the tokens `tokens/bytes/tinystories`. A cancel that arrives while `tokenize` waits runs `corpus.cleanup` with key `.../cleanup`, deletes the shards and tokens, keeps `corpus/raw/`, and ends the run canceled.

## 4. The interface

```go
const (
	ActCorpusStage   = "corpus.stage"
	ActCorpusCleanup = "corpus.cleanup"
)
var CorpusStages = []string{"fetch", "shard", "tokenize"}
type CorpusBuildInput struct{ Config json.RawMessage }
type StageInput struct{ Stage, ConfigSHA256 string; Config json.RawMessage }
type CleanupInput struct{ Dataset, Version, TokenizerID, ConfigSHA256 string }
type StageDone struct{ Outputs []string }
type CorpusBuildResult struct {
	Dataset, Version, ConfigSHA256, Manifest string
	Tokens []string
	Stages map[string][]string
}
func ConfigSHA256(cfg json.RawMessage) (string, error)
func CorpusWorkflowID(cfg json.RawMessage) (string, error)
func StageOptions(stage string) StepOptions
func CorpusBuild(rt Runtime, in CorpusBuildInput) (CorpusBuildResult, error)
```

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | section 3: the hash, the workflow id, the keys, the stage order, the result | builds are compared by these keys |
| Content, not spelling | property | whitespace and key order do not change the id; a changed value does | one config, one build |
| A bad config | boundary | missing dataset, version, or tokenizer fails non-retryable, before any activity | no gigabytes fetched for nothing |
| A kill at every position | fault | a crash during or between stages: same result, same files, each written once, no cleanup | `C1` trains on a corpus that survived kills |
| Cancel during shard or tokenize | fault | one cleanup, no shards or tokens left, raw cache kept, run ends canceled | a cancelled build is never read as a corpus |
| A failed stage | fault | a non-retryable stage failure cleans up and reports the stage's failure | an unlicensed source leaves nothing behind |
| A failed fetch | boundary | runs no cleanup and keeps an earlier build | the previous good corpus survives |
| Stage options | unit | ids are stage names, heartbeat timeouts set, fetch retries more | dead workers are noticed in minutes |

These cases run `CorpusBuild` through the replaying simulator over fake stages that keep the subprocess contract.

## 5. The solution, step by step

### 5.1 From first principles

1. **A workflow step must be resumable, so only work that leaves a durable result deserves an activity.** The pipeline's on-disk boundaries are after fetch (raw cache and ledger), after shard (Parquet plus manifest), and after tokenize (`.bin` files). Everything between them streams in memory, so those three are the activities.
2. **An idempotency key must be stable across attempts and distinct across inputs.** Naming each activity by its stage (not by call order) makes the key survive code changes; putting the config hash in the workflow id makes every config its own build.
3. **"The same config" means the same content.** Decoding and re-encoding with sorted keys and no whitespace gives a canonical byte string; its sha256 is the identity.
4. **A config that cannot run must fail before anything is fetched**, and fail non-retryably, because no retry fixes a config.
5. **Kills need no special code in the workflow.** A finished stage replays from history; a stage in flight is redelivered with the same key and is either a no-op (`DONE.json`) or a whole rewrite (files published by rename). The workflow just has to be deterministic.
6. **Undo exactly what this build could have left, and nothing shared.** The cleanup deletes this version's shards and tokens, never `corpus/raw/`. It is registered just before shard: before that, nothing of this build exists, and deleting would destroy a previous good build of the same version.
7. **Cancel and failure are the same path.** Any error from a stage (including `ErrCanceled`) goes through `saga.Fail`, which compensates once on the detached runtime and returns the original cause.

### 5.2 The code, annotated

The content hash and the id ([`corpus_build.go`](../../../../course/solutions/go/workflows/corpus_build.go)):

```go
func ConfigSHA256(cfg json.RawMessage) (string, error) {
	var v any
	if err := json.Unmarshal(cfg, &v); err != nil {
		return "", &ConfigError{Reason: "config is not JSON: " + err.Error()}
	}
	// Decoding into `any` turns objects into map[string]any; Marshal writes
	// map keys sorted and no whitespace, so this is the canonical form.
	canon, err := json.Marshal(v)
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

func CorpusWorkflowID(cfg json.RawMessage) (string, error) {
	// ... parse and hash ...
	return fmt.Sprintf("corpus/%s/%s/%s", c.Dataset, c.Version, sha[:12]), nil   // readable and content-addressed
}
```

`ConfigError` implements `NonRetryable() bool { return true }`, which is how the durable engine knows not to retry it.

Activity options per stage:

```go
func StageOptions(stage string) StepOptions {
	// ID = stage name: the key is "<workflow id>/<stage>", whatever the code order.
	// Heartbeat 2 min: a dead worker is noticed in minutes, not after the 2 h start-to-close.
	o := StepOptions{ID: stage, HeartbeatTimeout: 2 * time.Minute, StartToClose: 2 * time.Hour, MaxAttempts: 5}
	if stage == "fetch" {
		o.MaxAttempts = 8 // flaky networks: more retries, each cheap (data.01 resumes)
	}
	return o
}
```

The workflow:

```go
func CorpusBuild(rt Runtime, in CorpusBuildInput) (CorpusBuildResult, error) {
	c, err := parseCorpusConfig(in.Config)        // fails before any activity runs
	if err != nil {
		return CorpusBuildResult{}, err
	}
	sha, err := ConfigSHA256(in.Config)
	if err != nil {
		return CorpusBuildResult{}, err
	}
	res := CorpusBuildResult{Dataset: c.Dataset, Version: c.Version, ConfigSHA256: sha, Stages: map[string][]string{}}
	var saga Saga
	for _, stage := range CorpusStages {          // fetch, shard, tokenize: a slice, so a fixed order
		if stage == "shard" {
			// Registered BEFORE shard (a half-run shard may have left files), not before fetch
			// (a failed fetch left nothing of this build; deleting would destroy a previous one).
			saga.Add("delete the build's partial outputs", ActCorpusCleanup,
				CleanupInput{Dataset: c.Dataset, Version: c.Version, TokenizerID: c.Tokenizer.ID, ConfigSHA256: sha},
				StepOptions{ID: "cleanup", StartToClose: 10 * time.Minute, MaxAttempts: 10})
		}
		var done StageDone
		if err := rt.ExecuteActivity(ActCorpusStage, StageInput{Stage: stage, ConfigSHA256: sha, Config: in.Config}, StageOptions(stage), &done); err != nil {
			// one path for cancel and failure: compensate once, return the cause (still matches ErrCanceled)
			return res, saga.Fail(rt, fmt.Errorf("stage %s: %w", stage, err))
		}
		res.Stages[stage] = done.Outputs          // the outputs listed in the stage's DONE.json
	}
	if out := res.Stages["shard"]; len(out) > 0 {
		res.Manifest = out[0]                     // corpus/<dataset>/<version>/_MANIFEST.json
	}
	res.Tokens = res.Stages["tokenize"]
	if res.Manifest == "" {
		return res, errors.New("corpus build: the shard stage reported no manifest")
	}
	return res, nil
}
```

`Saga.Fail` (from `dur.08`, [`saga.go`](../../../../course/solutions/go/workflows/saga.go)) runs every registered compensation newest first on `rt.Detached()`, so the cleanup still runs after the workflow's own context was cancelled, and a second call does nothing, so the cleanup runs once.

Full solution: [`corpus_build.go`](../../../../course/solutions/go/workflows/corpus_build.go).

### 5.3 Why this solution

- **Cost.** Three activities per build: a handful of history events and round trips, negligible next to an hour of fetching and tokenizing. The workflow holds only stage outputs (lists of paths).
- **An activity per streaming stage** (filters, dedup, PII each separately) would make every boundary durable, but each would have to write and reread the whole corpus, and the round trips would buy nothing: those stages are fast and deterministic, so rerunning the shard stage from the raw cache is cheaper than checkpointing between them.
- **Per-shard fan-out** (Going further) would parallelize the shard stage, but dedup needs the whole corpus at once, so the fan-out would need its own reduce step. One shard activity keeps the semantics of `data.03` and `data.04` intact.
- **A random or timestamped workflow id** would make every start a new build, so a retried `ctl data build` would fetch and shard twice; a fixed id per dataset/version would make a changed config silently find the old run. The content hash is the only choice that gives both "same config, same run" and "new config, new run".
- **Accepted tradeoff.** A shard stage killed halfway reruns from the start of that stage; finer resumption would need per-shard activities.

### 5.4 Explain it back

1. Why is the activity id the stage name rather than left to the runtime's call counter?
   <details><summary>Answer</summary>The idempotency key is `<workflow id>/<activity id>`. A call-order id would shift every key when a stage is added or reordered in code, so a replay or redelivery could match the wrong stage's result. The stage name keeps each key naming its stage.</details>
2. Why is the cleanup registered just before `shard` and not before `fetch`?
   <details><summary>Answer</summary>A failed fetch has written nothing that belongs to this build (only the shared raw cache), and the cleanup deletes `corpus/<dataset>/<version>/`, which may hold a previous good build of the same version. From shard on, a partial output of this build may exist, so the cleanup must be armed.</details>
3. Why does `parseCorpusConfig` range over a slice of pairs and not a map of required fields?
   <details><summary>Answer</summary>Go randomizes map iteration order. With a map, a config missing two fields could report a different missing field in a run and in its replay, which is non-determinism in workflow code. A slice fixes the order.</details>
4. What happens when the worker is killed while `tokenize` is running?
   <details><summary>Answer</summary>On replay, `fetch` and `shard` are answered from history without running. `tokenize` is redelivered with the same key and the same work directory; it either finds its `DONE.json` and returns it, or rewrites its outputs whole (they are published by rename). The result and files equal an uninterrupted build's, and no cleanup runs.</details>
5. Why does a cancel still delete the partial shards even though the workflow's context is cancelled?
   <details><summary>Answer</summary>`saga.Fail` runs the compensation on `rt.Detached()`, a runtime not bound to the cancelled context, so `corpus.cleanup` is scheduled and runs; then the original `ErrCanceled`-matching error is returned and the run ends canceled.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. stages out of order | tokenize reads a manifest that does not exist yet | iterates the fixed slice `CorpusStages` |
| 2. activity ids by call order | the key no longer names the stage; a code change shifts every key | `StageOptions` sets `ID: stage` |
| 3. hashing the config's bytes, or leaving the hash out of the id | the same config twice is two builds; a changed config reuses the old run | `ConfigSHA256` hashes the canonical form; `CorpusWorkflowID` includes its first 12 hex digits |
| 4. no config validation | a broken config fetches before it fails | `parseCorpusConfig` runs first and returns a non-retryable `ConfigError` |
| 5. cleanup registered after the shard stage | a cancel during shard leaves partial shards | `saga.Add` runs just before the shard activity |
| 6. compensating only on cancel, or only on failure | half a corpus left behind by the other path | every stage error goes through `saga.Fail` |
| 7. cleanup registered before fetch | a failed download deletes yesterday's good build | the saga is empty while fetch runs |
| 8. no heartbeat timeout | a dead worker is noticed after the 2 h start-to-close | `HeartbeatTimeout: 2 * time.Minute` on every stage |

## 7. Where it's used next


| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.08` | `Runtime`, `Saga`, `ErrCanceled` |
| Back | `data.08` | the pipeline the stages run, ledger checks included |
| Back | `dur.09` | the subprocess runner the stages run through |
| Forward | `dur.11` | TrainRun builds its corpus with CorpusBuild before training |
| Forward | `C1` | the capstone corpus is a CorpusBuild |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| three stage activities | datatrove executors (local, Slurm) | many tasks per stage, each over a slice of the input, with completion markers per task | `datatrove/executor/base.py` |
| one shard stage | per-shard fan-out | each shard its own activity, run in parallel; the manifest written by a final reduce | Temporal "batch processing" samples |
| content-hashed workflow ids | Dolma's versioned configs, DVC pipelines | outputs addressed by the hash of their inputs and code | `dvc.lock` |
