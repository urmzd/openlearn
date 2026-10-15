<!-- ol:module dur.12 -->
# ModelRelease: gates, canary, PromQL burn check, rollback

## Overview

| | |
|---|---|
| **Module** | `dur.12` · build · Go · Pass 9 · 5 to 7 h |
| **You build** | `go/workflows/model_release.go`: `ModelRelease`, `ReleaseSpec` (`Validate`, `ExportSpec`, `EvalInput`), `CheckGates`, `ModelCardProblems`, `LedgerProblems`, and two activities, `Preflight` (`release.preflight`) and `ReadResults` (`release.results`); `go/activities/routes.go`: the admin route client `Routes` (`Get`, `Put`, `Snapshot`, `Ready`, `Canary`, `Promote`, `Restore`, `BackendsOf`, `CanaryBackends`, `SameBackends`); `go/activities/promql.go`: `Prom.Query`, `Decode`, `ParseSample` |
| **Contract** | the Go API of section 4 (fixed by this chapter and its tests, DEVIATIONS B112-01), speaking [`openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml) (routes and workers), [`formats/export-spec.schema.json`](../../../../course/contracts/formats/export-spec.schema.json), [`formats/eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json), [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json), and [`templates/MODEL_CARD.md`](../../../../course/contracts/templates/MODEL_CARD.md) |
| **Tests** | `course/tests/go/dur_12/`: the workflow under a replaying fake of the durable SDK, the route client against a fake gateway, the query against a fake Prometheus; what they check: section 4 |
| **Needs** | [`dur.08`](08-signals-cancellation-and-sagas.md) the workflow `Runtime` seam and `Saga` · [`dur.11`](11-trainrun-and-evalsuite.md) `EvalSuite`, run as the evaluation step (or `--ref-deps`); reading: [`dur.09`](09-subprocess-activities.md) (export and eval run as subprocess activities) · `data.08` the data ledger · `gw.05` and `gw.07` the admin route API · `obs.03` burn-rate alerts |
| **Used by** | `C1` (the capstone ships through your `ModelRelease`; its check runs your preflight on your release); later `ops.08` (the retrain decision after a data incident) |
| **Milestone** | MS-C1 (`{ctl} release --spec specs/c1/release.json` promotes or rolls back your capstone model) |
| **Optional depth** | Beyer et al., [*The Site Reliability Workbook*](https://sre.google/workbook/alerting-on-slos/), ch. 5 "Alerting on SLOs" (burn rates, free); [Prometheus HTTP API](https://prometheus.io/docs/prometheus/latest/querying/api/) (free); [Argo Rollouts analysis](https://argo-rollouts.readthedocs.io/en/stable/features/analysis/) and [Flagger](https://docs.flagger.app/) (canary analysis, free); RFC 9110 section 13.1.1 (`If-Match`) |

## Key Takeaways

- A release is a **durable workflow**, not a script: every step is an activity or a timer recorded in history, so a worker that dies mid-canary resumes where it was, and the 10-minute bake does not start over (`TestWorkerKilledMidCanary`).
- **Gates fail closed and early.** The model card and the data ledger are checked before the export; a missing eval row fails its gate; a burn query that returns nothing, NaN, or an error rolls back (`TestMissingEvalRowFailsTheGate`, `TestNoDataFailsClosed`).
- **No traffic moves before a human approves**, and a release nobody approves expires on the durable clock instead of holding a worker forever. Once traffic has moved, every way out but a promotion (a burn, a cancel, a failed step) **puts the snapshot back** through a saga compensation (`TestCancelRestoresTheRoute`).
- Route changes are **read-modify-write with `If-Match`** on one route: on 412 read again and redo the change, and send every other route back byte for byte.
- Every activity is **idempotent by desired state**: a retried canary finds its weight in place and does nothing, because activities run at least once.

## How to work this chapter

```bash
ol start dur.12         # stubs go/workflows/model_release.go, go/activities/{routes,promql}.go
ol tests dur.12         # read the test catalog first
ol check dur.12         # the workflow (replayed), the route client, the PromQL query
ol diff  dur.12         # after passing: your code against the reference
```

Then wire it in your worker's composition root (`go/cmd/worker`, learner territory): register the workflow as `ModelRelease` through the same `workflow.Context`-to-`Runtime` adapter as `TrainRun` (`dur.08`), and register the activities under the `Act*` names: `Preflight`, `ReadResults`, the subprocess runner of `dur.09` for `export` and `eval`, and the methods of one `activities.Routes` and one `activities.Prom`. Your umbrella CLI gets `{ctl} release --spec <file>` (start) and `<system> wf signal <id> approve`.

---

## 1. Why now

Your capstone model exists as a checkpoint under `runs/`, and your gateway serves whatever the route table says. Between the two there is only you, typing: copy the files, hope the model card is current, edit the route by hand, watch a dashboard, and undo it if something looks wrong. Every one of those steps fails in a known way. A model trained on a source whose license forbids training ships because nobody re-read the ledger. A route edit erases a colleague's cascade because you `PUT` a table you read an hour ago. The canary runs for ten minutes, your laptop sleeps, and nobody checks the error budget. Your durable engine (Pass 8) already survives crashes and keeps timers and signals; this module turns the release itself into a workflow on it: gated, approved, canaried, measured, and reversible.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $o$ | the SLO objective, the fraction of requests that must succeed (0.99) | `float64` |
| $b = 1 - o$ | the error budget as a fraction of requests (0.01) | `float64` |
| $e_W$ | the canary's error ratio over a window $W$: 5xx responses / all responses | `float64` |
| $B_W = e_W / b$ | the burn rate: how many times faster than allowed the budget is spent | `float64` |
| $B_{\max}$ | the ceiling (`burn.max`): roll back when $B_W > B_{\max}$ | `float64` |
| $w$ | the canary's share of the route's traffic, $0 < w < 1$ | `float64` |
| $T$ | the bake time (`canary_wait_s`) | duration |

### 2.1 The release is a workflow

`ModelRelease` is deterministic Go that decides; activities do (DESIGN 2.7). The workflow may branch only on its input and on what its `Env` recorded: activity results, timer firings, signals, and `Env.Now()`. It never reads a file, the wall clock, or the network itself, because on a restart the durable server replays the history through the same code, and any other input could take a different path (`ErrNondeterminism`). That is why the model card check is an activity (`release.preflight`) and not an `os.ReadFile` in the workflow.

The workflow is a function of `dur.08`'s `Runtime`, the SDK calls it needs (`ExecuteActivity`, `Sleep`, `Now`, `AwaitSignal`, `Detached`), like `TrainRun` and `EvalSuite`. Your worker adapts its `workflow.Context` to it once for all of them, and the course tests drive it with a fake that records a history, crashes and cancels on cue, and replays.

### 2.2 The steps

| # | Step | How | Fails as |
|---|---|---|---|
| 1 | validate the spec | pure | `GateError{spec}` before any activity |
| 2 | preflight: model card, ledger | activity `release.preflight` | `GateError` (non-retryable) |
| 3 | export the checkpoint | activity `export` (`{tinyllm} export --spec`) | activity failure |
| 4 | evaluate the **exported** model, then gate | `EvalSuite` (one `eval` activity per suite), `release.results` reads the reports, `CheckGates` | `GateError{eval}` |
| 5 | wait for `approve` or `reject` | `AwaitSignal`, timeout `approve_timeout_s` | outcome `rejected` or `expired` |
| 6 | snapshot the route, then canary | `routes.snapshot`, `routes.canary` | activity failure |
| 7 | bake | `Sleep(T)` | (durable: survives restarts) |
| 8 | burn check | `promql.query` at `Now()` | no data or error: roll back |
| 9 | promote or restore | `routes.promote`, or the saga's `routes.restore` | outcome `promoted` or `rolled_back` |

Every step carries a fixed activity id (`release-preflight`, `release-export`, `eval-release-<suite>`, `release-results`, `routes-snapshot`, `routes-canary`, `promql-burn`, `routes-promote`, `routes-restore`), so its idempotency key `<workflow_id>/<id>` is the same on every replay and retry.

Gates first because they are cheap and permanent: a forbidden license does not become allowed on retry, so a gate failure is non-retryable and the workflow fails before exporting gigabytes. The eval runs on the exported directory, not on the checkpoint: the export may quantize (`quant: int4-g32`), and the artifact you serve is the one you must measure.

### 2.3 The gates

- **Model card.** `MODEL_CARD.md` exists, has the five sections of `templates/MODEL_CARD.md`, and holds no template placeholder (`<model_id>`, `<you>`) outside HTML comments. A `<` followed by a digit or a space (`p < 0.05`) is prose, not a placeholder.
- **Ledger.** Every line of `LEDGER.jsonl` is a ledger row whose `allowed_uses` includes `train` and that is not `revoked`; an empty ledger fails (a model trained on undocumented data cannot ship).
- **Eval rows.** Each gate names `(suite, task, metric, op, value)`. It looks for the released model's row in that suite's report and passes when `value op threshold` holds, with `op` either `>=` or `<=` (inclusive). No row, a row whose status is not `ok`, or a row with a null value fails: a suite that did not run proves nothing.

### 2.4 Approval

A signal can arrive before the workflow waits for it (your operator approves while the eval still runs); the SDK buffers it (`dur.08`). The wait has a timeout on the durable clock; when it passes, the release ends as `expired` with no traffic moved.

### 2.5 The canary and optimistic concurrency

A route either has no `backends` (all traffic to the model of its own name) or a list of `{served_model, weight}` summing to 1. The canary gives the new model the share $w$ and scales every other backend by $1 - w$. The gateway replaces the whole table on `PUT` and guards it with an ETag: `GET` returns the table and `ETag: "routes-7"`; `PUT` with `If-Match: "routes-7"` succeeds and returns `"routes-8"`; a `PUT` with a stale ETag is `412`. So every change is:

1. `GET` the table; keep every route as the raw JSON the gateway sent.
2. Change one route's `backends` (or replace it, for a restore).
3. `PUT` the table with `If-Match`. On `412`, start again at 1 (at most `MaxConflicts` times).

Before the canary, `Ready` checks that a live, non-draining worker reports the new served model (`<model_id>-<version>`): routing a share to a model nobody serves fails that share of requests. Until one appears the canary returns a retryable `ErrNoWorkers`, and the activity's retries wait.

### 2.6 Cancellation and the saga

Right after the snapshot, the workflow registers one compensation with a `dur.08` `Saga`: "restore the route to the snapshot". If the run is canceled during the bake, or the canary or promotion fails for good, `saga.Fail` runs the compensation on `rt.Detached()` (so the cancel cannot interrupt the undo) and returns the original error, which still matches `ErrCanceled`. A burn above the ceiling runs the same compensation and ends as `rolled_back`. A cancel before the snapshot (during the approval wait) has nothing to undo.

### 2.7 Idempotent activities

An activity runs **at least once**: the worker can die after the `PUT` and before the server records the completion, and then the activity runs again. Each route activity therefore compares the table with the state it wants and does nothing when it is already there. Restoring needs the state from **before** the canary, so the workflow snapshots the route as its own recorded activity first; the snapshot is in history, and a replay restores exactly that.

### 2.8 The burn check

The error budget over a 30-day SLO period of 720 h is $b$ of all requests. A burn rate $B = 1$ spends it exactly in 30 days; $B = 14.4$ for one hour spends $14.4 \cdot 1 / 720 = 2\%$ of it, the page threshold of the SRE Workbook. The release asks Prometheus once, at the end of the bake, for $B_W$ of the **canary's** traffic (an expression that aggregates to one number), evaluated at `Env.Now()` so that a retried query asks the same question. It promotes when $B_W \le B_{\max}$ and rolls back otherwise. An empty vector (no canary request matched) and `NaN` ($0/0$) are **no data**; no data, and a query that failed for good, roll back: the canary did not prove itself.

## 3. Worked example by hand

The route `tinystories` serves `tinystories-10m-v1` with weight 1. You release `v2` with $w = 0.1$, $T = 600$ s, $o = 0.99$, $B_{\max} = 14.4$; the approval arrives; the workflow clock reads $t_0$ = 2026-10-09 12:00:00 UTC (Unix 1791547200).

**Canary weights.** Other backends total $r = 1$. `v1` gets $1 / r \cdot (1 - 0.1) = 0.9$, `v2` gets $0.1$; sum $1$. (With an even split `a` 0.5, `b` 0.5: each gets $0.5 / 1 \cdot 0.9 = 0.45$, and `v2` 0.1.) The gateway had ETag `"routes-7"`; the canary `PUT` sends `If-Match: "routes-7"` and the table becomes `"routes-8"`.

**Bake.** The timer fires at $t_0 + 600$ s, so the query carries `time=1791547800.000`.

**Burn, healthy.** In the last 5 minutes the canary answered 1,000 requests with 8 errors: $e = 8 / 1000 = 0.008$, $B = 0.008 / 0.01 = 0.8 \le 14.4$. Promote: `backends = [{tinystories-10m-v2, 1}]`, ETag `"routes-9"`. Two `PUT`s in all. This is `TestHandExampleCanaryPromotes` (and the weights are `TestHandExampleCanaryWeights`, the query `TestHandExampleBurnRate`).

**Burn, too fast.** 1,000 requests with 205 errors: $e = 0.205$, $B = 20.5 > 14.4$. Restore the snapshot: the route is again exactly `{"model":"tinystories","aliases":["ts"],"backends":[{"served_model":"tinystories-10m-v1","weight":1}],"x-owner":"ml-team"}`, and the cascade route `smart` never changed. This is `TestFastBurnRollsBackToTheSnapshot`.

**A crash.** The worker dies right after the canary `PUT`; another picks the run up 2 minutes later, replays steps 1 to 6 from history, and runs the canary activity again: the table already says `v2` 0.1, so no `PUT`. The timer starts at $t_0 + 2$ min. A second worker dies with the timer pending and comes back 2 minutes later; the timer keeps its deadline $t_0 + 2$ min $+ 600$ s, so the query is at Unix 1791547920. This is `TestWorkerKilledMidCanary`.

## 4. The interface

```go
// go/workflows/model_release.go (Runtime, StepOptions, StepFailure, Saga: dur.08; EvalSuite: dur.11)
func ModelRelease(rt Runtime, spec ReleaseSpec) (ReleaseResult, error) // *GateError, a *StepFailure, or ErrCanceled
func (s ReleaseSpec) Validate() error
func (s ReleaseSpec) ExportSpec() map[string]any                       // formats/export-spec.schema.json
func (s ReleaseSpec) EvalInput(modelDir string) EvalSuiteInput         // the release's EvalSuite
func CheckGates(gates []Gate, reports []EvalReport, subject string) []GateResult
func ModelCardProblems(text string) []string
func LedgerProblems(ledger []byte) []string
func Preflight(ctx context.Context, in PreflightInput) (PreflightResult, error) // activity release.preflight
func ReadResults(ctx context.Context, in ResultsInput) (EvalResult, error)     // activity release.results

// go/activities/routes.go
func (r Routes) Get(ctx) ([]json.RawMessage, string, error)            // the routes as sent, and the ETag
func (r Routes) Put(ctx, routes []json.RawMessage, etag string) (string, error) // ErrStale on 412
func (r Routes) Snapshot(ctx, SnapshotInput) (RouteSnapshot, error)    // activity routes.snapshot
func (r Routes) Ready(ctx, served string) (bool, error)                // GET /admin/v1/workers
func (r Routes) Canary(ctx, CanaryInput) (RouteResult, error)          // activity routes.canary
func (r Routes) Promote(ctx, PromoteInput) (RouteResult, error)        // activity routes.promote
func (r Routes) Restore(ctx, RestoreInput) (RouteResult, error)        // activity routes.restore
func BackendsOf(route map[string]any) ([]Backend, error)
func CanaryBackends(current []Backend, served string, w float64) []Backend
func SameBackends(a, b []Backend) bool

// go/activities/promql.go
func (p Prom) Query(ctx, QueryInput) (Sample, error)                   // activity promql.query
func Decode(body []byte) (Sample, error)
func ParseSample(pair []json.RawMessage) (float64, bool, error)       // (value, isNaN, err)
```

An error that a retry cannot fix implements `NonRetryable() bool` returning true: `*GateError` and `activities.Permanent` (a route that does not exist, a weight outside $(0, 1)$, a table the gateway rejects with 400, a query Prometheus rejects). Your composition root maps it to `Failure.non_retryable`. Transport errors, `ErrStale` after `MaxConflicts`, `ErrNoWorkers`, and a 5xx from Prometheus are retryable.

A release spec (`specs/c1/release.json` in the capstone):

```json
{"model_id": "tinystories-10m", "version": "v2", "from": "runs/c1/ckpt",
 "model_card": "docs/MODEL_CARD.md", "ledger": "corpus/LEDGER.jsonl",
 "suites": ["quality", "safety"], "eval_seed": 0,
 "gates": [{"suite": "quality", "task": "ts-val", "metric": "bpb", "op": "<=", "value": 1.30},
           {"suite": "safety", "task": "refusal", "metric": "score", "op": ">=", "value": 0.9}],
 "route": "tinystories", "canary_weight": 0.1, "canary_wait_s": 600, "approve_timeout_s": 86400,
 "burn": {"query": "sum(rate(...{model=\"tinystories-10m-v2\",http_response_status_code=~\"5..\"}[5m])) / sum(rate(...{model=\"tinystories-10m-v2\"}[5m])) / (1 - 0.99)", "max": 14.4}}
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `TestHandExampleCanaryPromotes` | unit | section 3: the step order and activity ids, 0.9/0.1 during the bake, the query at $t_0 + 600$ s, promotion, two `PUT`s, the eval runs on the exported directory | you and the test agree on the release |
| `TestFastBurnRollsBackToTheSnapshot` | fault | burn 20.5: the route is restored to its snapshot, the other route untouched | a bad model leaves no trace in routing |
| `TestBurnAtTheCeilingPromotes` | boundary | burn equal to the ceiling promotes | the ceiling is inclusive |
| `TestNoDataFailsClosed` | fault | an empty vector and NaN roll back | a canary with no traffic proves nothing |
| `TestFailedBurnQueryFailsClosed` | fault | a rejected query rolls back after one try | a broken check is not a pass |
| `TestUnlicensedSourceFailsBeforeExport` | fault | an eval-only source fails the gate once, before export | licenses are checked before any work |
| `TestRevokedSourceFails` | fault | a revoked source fails the gate | the `ops.08` data incident blocks releases |
| `TestModelCardGate` | fault | missing card, missing section, leftover placeholder; `p < 0.05` is fine | ethics.03's card is a real gate |
| `TestMissingEvalRowFailsTheGate` | fault | no safety row: the eval gate refuses; no signal consumed, no `PUT` | ethics.04's rows must exist to pass |
| `TestGateDirections` | boundary | `<=` and `>=` inclusive; another model's row ignored | lower-is-better and higher-is-better metrics |
| `TestRejectStopsBeforeTraffic` | unit | `reject` ends the release with no route access | humans can stop a release |
| `TestApprovalTimesOut` | unit | no answer: `expired` after exactly `approve_timeout_s` on the durable clock | nothing waits forever |
| `TestWorkerKilledMidCanary` | fault | crash after the canary `PUT` and during the timer: one canary `PUT`, the timer keeps its deadline | releases survive worker loss |
| `TestCancelRestoresTheRoute` | fault | a cancel in the approval wait touches nothing; a cancel in the bake restores the snapshot and never queries | an operator can stop a release safely |
| `TestReplayIsDeterministic` | regression | replaying a finished history with no live activities gives the same result | the workflow obeys the replay rules |
| `TestSpecIsValidatedFirst` | boundary | weight 1, an empty query, op `>`: refused before any activity | a bad spec moves nothing |
| `TestHandExampleCanaryWeights` | unit | 1 to 0.9/0.1; 0.5/0.5 to 0.45/0.45/0.1; re-weighting a canary | the gateway's sum-to-1 rule |
| `TestImplicitBackendIsTheRouteModel` | boundary | a route without `backends` keeps 0.75 for its own model | the old model keeps its traffic |
| `TestPutCarriesTheETag` | conformance | one `PUT` with `If-Match: "routes-7"` | admin.v1 optimistic concurrency |
| `TestStaleETagRereadsAndKeepsTheOtherChange` | fault | a concurrent edit: 412, reread, both changes survive | operators and releases can overlap |
| `TestOtherRoutesAreUntouched` | regression | the cascade route is byte-identical; unknown fields kept | a release cannot erase routing it did not own |
| `TestCanaryIsIdempotent` | property | a second canary changes nothing | at-least-once activities |
| `TestCanaryWaitsForAServingWorker` | fault | no live worker (or a draining one): retryable `ErrNoWorkers`, no `PUT` | no traffic to a model nobody serves |
| `TestUnknownRouteAndBadInputArePermanent` | boundary | unknown route, weight 1, a 400 table: non-retryable | fail fast on what retries cannot fix |
| `TestPromoteAndRestore` | unit | promote to weight 1; restore exactly and idempotently; another route's snapshot refused | the two ends of a release |
| `TestHandExampleBurnRate` | unit | section 3's query: value 20.5, `query` and `time` sent | the measurement the decision rests on |
| `TestScalarAndInfinity` | unit | a scalar result; `+Inf` is a value, not no data | `scalar(...)` expressions; all-error canaries |
| `TestNoSeriesAndNaNAreNoData` | boundary | empty vector and NaN are no data | fail closed (pitfall 11) |
| `TestSeveralSeriesAreRefused` | boundary | two series or a matrix: permanent `ErrQuery` | an unaggregated query checks a random pod |
| `TestErrorStatusIsPermanentAndServerErrorsRetry` | fault | `bad_data` is permanent; a 503 is retryable | one Prometheus blip must not roll back |

The workflow tests run `ModelRelease` under a fake `Runtime` that keeps a history, crashes and cancels on cue, and replays from the top after every crash, re-executing only activities whose completion was not recorded. The clock is virtual: no test sleeps. Its `eval` activity writes one `tl.eval-results.v1` report per suite, which your `ReadResults` reads.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. gate results computed but not enforced | a model that failed its safety suite takes traffic | `TestMissingEvalRowFailsTheGate` (mutant `s01`) |
| 2. a gate with no row passes | the safety suite crashed and the release went out | `TestMissingEvalRowFailsTheGate` (mutant `s02`) |
| 3. `>=` written as `>` | a model exactly at its threshold is refused | `TestGateDirections` (mutant `s03`) |
| 4. the ledger check reads `eval` as enough, or skips `revoked` | a non-commercial or revoked source ships | `TestUnlicensedSourceFailsBeforeExport`, `TestRevokedSourceFails` (mutants `s04`, `s05`) |
| 5. a template copied but never filled passes | `<model_id>` in a published model card | `TestModelCardGate` (mutant `s06`) |
| 6. a gate failure returned as a plain error | the gate is retried three times, then reported as an outage | `TestUnlicensedSourceFailsBeforeExport` (mutant `s07`) |
| 7. evaluating the checkpoint, not the exported model | the int4 export ships unmeasured | `TestHandExampleCanaryPromotes` (mutant `s08`) |
| 8. reject, timeout, or the wait itself missing | a rejected model ships; a release waits forever; traffic moves unapproved | `TestRejectStopsBeforeTraffic`, `TestApprovalTimesOut` (mutants `s09`, `s10`, `s11`) |
| 9. the snapshot taken after the canary, a rollback branch that never restores, or a restore that merges | rollback keeps the canary's backends | `TestFastBurnRollsBackToTheSnapshot`, `TestPromoteAndRestore` (mutants `s12`, `s27`, `s37`) |
| 10. no bake, or a query at no fixed time | the burn is measured before the canary served anything; a retry measures another window | `TestHandExampleCanaryPromotes`, `TestBurnAtTheCeilingPromotes`, `TestWorkerKilledMidCanary` (mutants `s13`, `s33`) |
| 11. no data or a failed query read as healthy | a canary that served nothing, or that Prometheus could not see, is promoted | `TestNoDataFailsClosed`, `TestFailedBurnQueryFailsClosed` (mutants `s15`, `s16`, `s29`) |
| 12. a strict ceiling | a canary exactly at the budget line rolls back | `TestBurnAtTheCeilingPromotes` (mutant `s17`) |
| 13. `PUT` without `If-Match`, or giving up on 412 | 428 on every change; a concurrent edit kills the release | `TestPutCarriesTheETag`, `TestStaleETagRereadsAndKeepsTheOtherChange` (mutants `s18`, `s19`) |
| 14. the table rebuilt instead of returned as read | `>=` comes back as `>=`; other routes vanish | `TestOtherRoutesAreUntouched` (mutants `s20`, `s21`) |
| 15. a canary that is not idempotent | a retried activity makes 0.9/0.1 into 0.81/0.09/0.1 | `TestCanaryIsIdempotent`, `TestWorkerKilledMidCanary` (mutant `s22`) |
| 16. weights not rescaled, or a missing `backends` read as none | the gateway rejects the table (sum not 1); the old model loses its share | `TestHandExampleCanaryWeights`, `TestImplicitBackendIsTheRouteModel` (mutants `s23`, `s24`) |
| 17. no check that a worker serves the new model | 10% of requests fail with no backend | `TestCanaryWaitsForAServingWorker` (mutant `s25`) |
| 18. permanent errors returned as retryable | an unknown route or a bad expression retries for minutes | `TestUnknownRouteAndBadInputArePermanent`, `TestFailedBurnQueryFailsClosed` (mutants `s26`, `s34`) |
| 19. the first of several series used | the check reads one random pod | `TestSeveralSeriesAreRefused` (mutant `s30`) |
| 20. scalars refused, or a 503 treated as permanent | `scalar(...)` queries fail; one overload rolls back a good model | `TestScalarAndInfinity`, `TestErrorStatusIsPermanentAndServerErrorsRetry` (mutants `s31`, `s32`) |
| 21. a canary weight of 1 accepted | the "canary" is a full cutover with no way back but a rollback | `TestSpecIsValidatedFirst` (mutant `s35`) |
| 22. a cancel that does not compensate | a canceled release leaves 10% of traffic on an unapproved model | `TestCancelRestoresTheRoute` (mutant `s36`) |
| 23. activity ids left to the runtime's counter | an added step renumbers every later idempotency key | `TestHandExampleCanaryPromotes`, `TestCancelRestoresTheRoute` (mutant `s39`) |
| 24. `time.Now()`, `rand`, or map order in the workflow | replay diverges after a restart (`ErrNondeterminism`) | `TestReplayIsDeterministic` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.08` | the `Runtime` the workflow runs on, and the `Saga` that undoes the canary |
| Back | `dur.11` | `EvalSuite` is the evaluation step |
| Back | `dur.09` | `export` and `eval` are subprocess activities run by your runner |
| Back | `data.08` | the ledger the preflight gate reads |
| Back | `gw.05`, `gw.07` | the admin route and worker API this client speaks |
| Back | `obs.03` | the burn-rate expressions and thresholds of your SLO rules |
| Forward | `C1` | your capstone model is released through this workflow; C1's check runs your `Validate` and `Preflight` on its release |
| Forward | `ops.08` | the data-incident drill revokes a source and expects the next release to fail its ledger gate |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ModelRelease` | Argo Rollouts, Flagger | progressive steps (1%, 5%, 25%, ...), several metric analyses per step, automatic abort, traffic mirroring | `argo-rollouts/rollout/canary.go`, Flagger `pkg/controller/scheduler.go` |
| the burn check | Kayenta (Spinnaker) | statistical canary analysis: the canary compared with a baseline started at the same time, Mann-Whitney per metric | `kayenta-judge` |
| `Runtime` | Temporal Go SDK | `workflow.Context`, `workflow.NewSelector` over signals and timers, versioning with `GetVersion` | `go.temporal.io/sdk/workflow` |
| gates | MLflow model registry, Vertex AI Model Registry | stage transitions with approval, lineage to data and runs | MLflow `model_registry` docs |
| `Routes` | Envoy AI Gateway, Gateway API `HTTPRoute` weights | weighted backends as Kubernetes objects; the API server's `resourceVersion` is the ETag | Gateway API `backendRefs[].weight` |
