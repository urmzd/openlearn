<!-- ol:module data.09 -->
# Pipeline as a durable workflow CorpusBuild

## Overview

| | |
|---|---|
| **Module** | `data.09` · build · Go · Pass 8 · 3 to 4 h |
| **You build** | `go/workflows/corpus_build.go`: `CorpusBuild(rt, in)`, `ConfigSHA256`, `CorpusWorkflowID`, `StageOptions`, the activity names `corpus.stage` and `corpus.cleanup`; then the activity form of your `{corpus}` CLI (`run --stage <s>` and `cleanup`, entry-point territory) |
| **Contract** | [`course/contracts/formats/corpus-config.schema.json`](../../../../course/contracts/formats/corpus-config.schema.json) (the stage spec) and [`course/contracts/spec/subprocess-activity.md`](../../../../course/contracts/spec/subprocess-activity.md) (how a stage runs) |
| **Tests** | `course/tests/go/data_09/` (CorpusBuild through the replaying simulator, over fake stages that keep the subprocess contract; section 4) |
| **Needs** | `dur.08` (the `Runtime` seam and `Saga`); reading: `data.08` (the pipeline you wrap, `data.01` to `data.08`), `dur.09` (the subprocess runner the stages run through) |
| **Used by** | `dur.11` (TrainRun builds its corpus first); later `C1` |
| **Milestone** | MS-durable (`{ctl} data build` survives kills with the same manifest hash) |
| **Optional depth** | [datatrove: pipelines and executors](https://github.com/huggingface/datatrove) (free); [Temporal: idempotency of activities](https://docs.temporal.io/activity-definition#idempotency) (free) |

## Key Takeaways

- The workflow **orders, keys, and undoes**; the stages do the work. Each stage is one subprocess activity at an **on-disk boundary** of the pipeline (`fetch`, `shard`, `tokenize`).
- The **workflow id carries the config's content hash** and each stage's activity id is its name, so the idempotency key `corpus/<dataset>/<version>/<hash12>/<stage>` is stable across attempts and different for every config.
- The hash is of the config's **content**, not its spelling: decoded and re-encoded with sorted keys.
- A kill at any point gives the **same result and files** as an uninterrupted build; a finished stage is replayed from history, a stage in flight reruns as a no-op or a whole rewrite.
- **Cancel or failure deletes the build's partial outputs** (a compensation registered before the shard stage), never the shared raw-document cache.

## How to work this chapter

```bash
ol start data.09         # stubs go/workflows/corpus_build.go
ol tests data.09         # read the test catalog first
ol check data.09
ol diff  data.09
```

Then, in your worker's composition root, register the workflow (adapting `workflow.Context` to `Runtime`) and two subprocess activities over `{corpus}`: `corpus.stage` runs `run --stage <in.Stage>` with `in.Config` as the spec, `corpus.cleanup` runs `cleanup` with its input as the spec. Add `{ctl} data build --config <toml>`: parse the TOML, start `CorpusBuild` with workflow id `CorpusWorkflowID(config)`.

---

## 1. Why now

Your corpus pipeline (`data.01` to `data.08`) is one command: `{corpus} run --config small.toml`. On the real TinyStories corpus it runs for an hour, downloads gigabytes, and dies with the laptop's battery or a flaky mirror; a rerun starts from scratch, and a run you stop halfway leaves shards that the next training run reads as a complete corpus. You now own a durable engine (`dur.01` to `dur.08`) and a way to run Python as activities (`dur.09`). `CorpusBuild` puts them together: each stage becomes an activity with a stable idempotency key, a kill anywhere resumes, and a cancel cleans up after itself.

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

This is `TestHandExampleKeysAndOrder`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleKeysAndOrder` | unit | section 3: the hash, the workflow id, the keys, the stage order, the result | MS-durable compares manifests by these keys |
| `TestConfigHashIsContentNotSpelling` | property | whitespace and key order do not change the id; a changed value does | one config, one build |
| `TestBadConfigFailsBeforeAnyActivity` | boundary | missing dataset, version, or tokenizer fails non-retryable, no activity | no gigabytes fetched for nothing |
| `TestKillLoopSameResult` | fault | a crash at every position, during or between stages: same result, same files, each written once, no cleanup | `C1` trains on a corpus that survived kills |
| `TestCancelDeletesPartialShards` | fault | cancel during shard or tokenize: one cleanup, no shards or tokens left, raw cache kept, canceled | a cancelled build is never read as a corpus |
| `TestFailedStageCompensates` | fault | a non-retryable stage failure cleans up and reports the stage's failure | an unlicensed source leaves nothing behind |
| `TestFetchFailureNeedsNoCleanup` | boundary | a failed fetch runs no cleanup and keeps an earlier build | the previous good corpus survives |
| `TestStageOptions` | unit | ids are stage names, heartbeat timeouts set, fetch retries more | dead workers are noticed in minutes |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. stages out of order | tokenize reads a manifest that does not exist yet | `TestHandExampleKeysAndOrder` (mutant `s01`) |
| 2. activity ids by call order | the key no longer names the stage; a code change shifts every key | `TestHandExampleKeysAndOrder` (mutant `s02`) |
| 3. hashing the config's bytes, or leaving the hash out of the id | the same config twice is two builds; a changed config reuses the old run | `TestHandExampleKeysAndOrder`, `TestConfigHashIsContentNotSpelling` (mutants `s03`, `s04`) |
| 4. no config validation | a broken config fetches before it fails | `TestBadConfigFailsBeforeAnyActivity` (mutant `s05`) |
| 5. cleanup registered after the shard stage | a cancel during shard leaves partial shards | `TestCancelDeletesPartialShards` (mutant `s07`) |
| 6. compensating only on cancel, or only on failure | half a corpus left behind by the other path | `TestCancelDeletesPartialShards`, `TestFailedStageCompensates` (mutants `s08`, `s09`) |
| 7. cleanup registered before fetch | a failed download deletes yesterday's good build | `TestFetchFailureNeedsNoCleanup` (mutant `s10`) |
| 8. no heartbeat timeout | a dead worker is noticed after the 2 h start-to-close | `TestStageOptions` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.08` | `Runtime`, `Saga`, `ErrCanceled` |
| Back | `data.08` | the pipeline the stages run, ledger checks included |
| Back | `dur.09` | the subprocess runner the stages run through |
| Forward | `dur.11` | TrainRun builds its corpus with CorpusBuild before training |
| Forward | `C1` | the capstone corpus is a CorpusBuild |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| three stage activities | datatrove executors (local, Slurm) | many tasks per stage, each over a slice of the input, with completion markers per task | `datatrove/executor/base.py` |
| one shard stage | per-shard fan-out | each shard its own activity, run in parallel; the manifest written by a final reduce | Temporal "batch processing" samples |
| content-hashed workflow ids | Dolma's versioned configs, DVC pipelines | outputs addressed by the hash of their inputs and code | `dvc.lock` |
