# ModelRelease: gates, canary, PromQL burn check, rollback

## Overview

| | |
|---|---|
| **Module** | `dur.12` · build · Go · Pass 9 · reading time 2 to 3 h |
| **Solution** | [`go/workflows/model_release.go`](../../../../course/solutions/go/workflows/model_release.go): `ModelRelease`, `ReleaseSpec` (`Validate`, `ExportSpec`, `EvalInput`), `CheckGates`, `ModelCardProblems`, `LedgerProblems`, and two activities, `Preflight` (`release.preflight`) and `ReadResults` (`release.results`); [`go/activities/routes.go`](../../../../course/solutions/go/activities/routes.go): the admin route client `Routes` (`Get`, `Put`, `Snapshot`, `Ready`, `Canary`, `Promote`, `Restore`, `BackendsOf`, `CanaryBackends`, `SameBackends`); [`go/activities/promql.go`](../../../../course/solutions/go/activities/promql.go): `Prom.Query`, `Decode`, `ParseSample` |
| **Contract** | the Go API of section 4, speaking [`openapi/admin.v1.yaml`](../../../../course/contracts/openapi/admin.v1.yaml) (routes and workers), [`formats/export-spec.schema.json`](../../../../course/contracts/formats/export-spec.schema.json), [`formats/eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json), [`formats/ledger.schema.json`](../../../../course/contracts/formats/ledger.schema.json), and [`templates/MODEL_CARD.md`](../../../../course/contracts/templates/MODEL_CARD.md) |
| **Needs** | [`dur.08`](08-signals-cancellation-and-sagas.md) the workflow `Runtime` seam and `Saga` · [`dur.11`](11-trainrun-and-evalsuite.md) `EvalSuite`, run as the evaluation step; reading: [`dur.09`](09-subprocess-activities.md) (export and eval run as subprocess activities) · `data.08` the data ledger · `gw.05` and `gw.07` the admin route API · `obs.03` burn-rate alerts |
| **Used by** | `C1` (the capstone ships through `ModelRelease`: a release of [`specs/c1/release.json`](../../../../course/solutions/entry/specs/c1/release.json) promotes or rolls back the capstone model, and C1 runs `Validate` and `Preflight` on it); later `ops.08` (the retrain decision after a data incident) |
| **Optional depth** | Beyer et al., [*The Site Reliability Workbook*](https://sre.google/workbook/alerting-on-slos/), ch. 5 "Alerting on SLOs" (burn rates, free); [Prometheus HTTP API](https://prometheus.io/docs/prometheus/latest/querying/api/) (free); [Argo Rollouts analysis](https://argo-rollouts.readthedocs.io/en/stable/features/analysis/) and [Flagger](https://docs.flagger.app/) (canary analysis, free); RFC 9110 section 13.1.1 (`If-Match`) |

## Key Takeaways

- A release is a **durable workflow**, not a script: every step is an activity or a timer recorded in history, so a worker that dies mid-canary resumes where it was, and the 10-minute bake does not start over.
- **Gates fail closed and early.** The model card and the data ledger are checked before the export; a missing eval row fails its gate; a burn query that returns nothing, NaN, or an error rolls back.
- **No traffic moves before a human approves**, and a release nobody approves expires on the durable clock instead of holding a worker forever. Once traffic has moved, every way out but a promotion (a burn, a cancel, a failed step) **puts the snapshot back** through a saga compensation.
- Route changes are **read-modify-write with `If-Match`** on one route: on 412 read again and redo the change, and send every other route back byte for byte.
- Every activity is **idempotent by desired state**: a retried canary finds its weight in place and does nothing, because activities run at least once.

## How to read this chapter

Sections 1 to 3 derive the release steps, the gates, the canary arithmetic, and the burn check, and work one release by hand. Section 4 fixes the Go API. Section 5 walks through the workflow, the route client, and the PromQL query, and explains why each choice is the right one. Try to predict each step of the solution before reading it.

The worker's composition root registers the workflow as `ModelRelease` through the same `workflow.Context`-to-`Runtime` adapter as `TrainRun` (`dur.08`), and the activities under the `Act*` names: `Preflight`, `ReadResults`, the subprocess runner of `dur.09` for `export` and `eval`, and the methods of one `activities.Routes` and one `activities.Prom` (the package comment of `model_release.go` shows the registration). The system's CLI starts a release with `{ctl} release --spec <file>` and approves it with `<system> wf signal <id> approve`.

---

## 1. Why now

Your capstone model exists as a checkpoint under `runs/`, and your gateway serves whatever the route table says. Between the two there is only you, typing: copy the files, hope the model card is current, edit the route by hand, watch a dashboard, and undo it if something looks wrong. Every one of those steps fails in a known way. A model trained on a source whose license forbids training ships because nobody re-read the ledger. A route edit erases a colleague's cascade because you `PUT` a table you read an hour ago. The canary runs for ten minutes, your laptop sleeps, and nobody checks the error budget. The durable engine (Pass 8) already survives crashes and keeps timers and signals; this module turns the release itself into a workflow on it: gated, approved, canaried, measured, and reversible.

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

`ModelRelease` is deterministic Go that decides; activities do. The workflow may branch only on its input and on what its `Env` recorded: activity results, timer firings, signals, and `Env.Now()`. It never reads a file, the wall clock, or the network itself, because on a restart the durable server replays the history through the same code, and any other input could take a different path (`ErrNondeterminism`). That is why the model card check is an activity (`release.preflight`) and not an `os.ReadFile` in the workflow.

The workflow is a function of `dur.08`'s `Runtime`, the SDK calls it needs (`ExecuteActivity`, `Sleep`, `Now`, `AwaitSignal`, `Detached`), like `TrainRun` and `EvalSuite`. The worker adapts its `workflow.Context` to it once for all of them, and a fake `Runtime` that records a history, crashes and cancels on cue, and replays is enough to exercise every path.

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

**Burn, healthy.** In the last 5 minutes the canary answered 1,000 requests with 8 errors: $e = 8 / 1000 = 0.008$, $B = 0.008 / 0.01 = 0.8 \le 14.4$. Promote: `backends = [{tinystories-10m-v2, 1}]`, ETag `"routes-9"`. Two `PUT`s in all.

**Burn, too fast.** 1,000 requests with 205 errors: $e = 0.205$, $B = 20.5 > 14.4$. Restore the snapshot: the route is again exactly `{"model":"tinystories","aliases":["ts"],"backends":[{"served_model":"tinystories-10m-v1","weight":1}],"x-owner":"ml-team"}`, and the cascade route `smart` never changed.

**A crash.** The worker dies right after the canary `PUT`; another picks the run up 2 minutes later, replays steps 1 to 6 from history, and runs the canary activity again: the table already says `v2` 0.1, so no `PUT`. The timer starts at $t_0 + 2$ min. A second worker dies with the timer pending and comes back 2 minutes later; the timer keeps its deadline $t_0 + 2$ min $+ 600$ s, so the query is at Unix 1791547920.

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

An error that a retry cannot fix implements `NonRetryable() bool` returning true: `*GateError` and `activities.Permanent` (a route that does not exist, a weight outside $(0, 1)$, a table the gateway rejects with 400, a query Prometheus rejects). The worker's composition root maps it to `Failure.non_retryable`. Transport errors, `ErrStale` after `MaxConflicts`, `ErrNoWorkers`, and a 5xx from Prometheus are retryable.

A release spec ([`specs/c1/release.json`](../../../../course/solutions/entry/specs/c1/release.json) in the capstone):

```json
{"model_id": "tinystories-10m", "version": "v2", "from": "runs/c1/ckpt",
 "model_card": "docs/MODEL_CARD.md", "ledger": "corpus/LEDGER.jsonl",
 "suites": ["quality", "safety"], "eval_seed": 0,
 "gates": [{"suite": "quality", "task": "ts-val", "metric": "bpb", "op": "<=", "value": 1.30},
           {"suite": "safety", "task": "refusal", "metric": "score", "op": ">=", "value": 0.9}],
 "route": "tinystories", "canary_weight": 0.1, "canary_wait_s": 600, "approve_timeout_s": 86400,
 "burn": {"query": "sum(rate(...{model=\"tinystories-10m-v2\",http_response_status_code=~\"5..\"}[5m])) / sum(rate(...{model=\"tinystories-10m-v2\"}[5m])) / (1 - 0.99)", "max": 14.4}}
```

### What a correct solution must handle

The workflow cases run `ModelRelease` under a fake `Runtime` that keeps a history, crashes and cancels on cue, and replays from the top after every crash, re-executing only activities whose completion was not recorded. The clock is virtual. Its `eval` activity writes one `tl.eval-results.v1` report per suite, which `ReadResults` reads.

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| the hand example promotes | unit | section 3: the step order and activity ids, 0.9/0.1 during the bake, the query at $t_0 + 600$ s, promotion, two `PUT`s, the eval runs on the exported directory | the reference behaviour of a release |
| a fast burn rolls back to the snapshot | fault | burn 20.5: the route is restored to its snapshot, the other route untouched | a bad model leaves no trace in routing |
| a burn at the ceiling promotes | boundary | burn equal to the ceiling promotes | the ceiling is inclusive |
| no data fails closed | fault | an empty vector and NaN roll back | a canary with no traffic proves nothing |
| a failed burn query fails closed | fault | a rejected query rolls back after one try | a broken check is not a pass |
| an unlicensed source fails before export | fault | an eval-only source fails the gate once, before export | licenses are checked before any work |
| a revoked source fails | fault | a revoked source fails the gate | the `ops.08` data incident blocks releases |
| the model card gate | fault | missing card, missing section, leftover placeholder; `p < 0.05` is fine | ethics.03's card is a real gate |
| a missing eval row fails the gate | fault | no safety row: the eval gate refuses; no signal consumed, no `PUT` | ethics.04's rows must exist to pass |
| gate directions | boundary | `<=` and `>=` inclusive; another model's row ignored | lower-is-better and higher-is-better metrics |
| reject stops before traffic | unit | `reject` ends the release with no route access | humans can stop a release |
| the approval times out | unit | no answer: `expired` after exactly `approve_timeout_s` on the durable clock | nothing waits forever |
| a worker killed mid-canary | fault | crash after the canary `PUT` and during the timer: one canary `PUT`, the timer keeps its deadline | releases survive worker loss |
| cancel restores the route | fault | a cancel in the approval wait touches nothing; a cancel in the bake restores the snapshot and never queries | an operator can stop a release safely |
| replay is deterministic | regression | replaying a finished history with no live activities gives the same result | the workflow obeys the replay rules |
| the spec is validated first | boundary | weight 1, an empty query, op `>`: refused before any activity | a bad spec moves nothing |
| the hand example's canary weights | unit | 1 to 0.9/0.1; 0.5/0.5 to 0.45/0.45/0.1; re-weighting a canary | the gateway's sum-to-1 rule |
| an implicit backend is the route's model | boundary | a route without `backends` keeps 0.75 for its own model | the old model keeps its traffic |
| `PUT` carries the ETag | conformance | one `PUT` with `If-Match: "routes-7"` | admin.v1 optimistic concurrency |
| a stale ETag rereads and keeps the other change | fault | a concurrent edit: 412, reread, both changes survive | operators and releases can overlap |
| other routes are untouched | regression | the cascade route is byte-identical; unknown fields kept | a release cannot erase routing it did not own |
| the canary is idempotent | property | a second canary changes nothing | at-least-once activities |
| the canary waits for a serving worker | fault | no live worker (or a draining one): retryable `ErrNoWorkers`, no `PUT` | no traffic to a model nobody serves |
| unknown route and bad input are permanent | boundary | unknown route, weight 1, a 400 table: non-retryable | fail fast on what retries cannot fix |
| promote and restore | unit | promote to weight 1; restore exactly and idempotently; another route's snapshot refused | the two ends of a release |
| the hand example's burn rate | unit | section 3's query: value 20.5, `query` and `time` sent | the measurement the decision rests on |
| scalar and infinity | unit | a scalar result; `+Inf` is a value, not no data | `scalar(...)` expressions; all-error canaries |
| no series and NaN are no data | boundary | empty vector and NaN are no data | fail closed (pitfall 11) |
| several series are refused | boundary | two series or a matrix: permanent `ErrQuery` | an unaggregated query checks a random pod |
| error status is permanent, server errors retry | fault | `bad_data` is permanent; a 503 is retryable | one Prometheus blip must not roll back |

## 5. The solution, step by step

### 5.1 From first principles

1. **A release spans minutes to days and must survive any crash**, so it is a workflow: every effect is an activity, every wait a durable timer or signal. Replay then demands determinism: the workflow may branch only on its input and recorded results, so even reading the model card is an activity.
2. **Cheap, permanent checks go first.** A forbidden license or an unfilled model card does not change on retry, so they are checked before the export, as non-retryable `GateError`s. Validation of the spec itself happens before any activity.
3. **Measure what ships.** The export may quantize, so the evaluation subject is the exported directory, and gates read rows for that subject only. A missing row proves nothing, so it fails the gate.
4. **A human decides before traffic moves, and silence is a "no".** `AwaitSignal` with a timeout on the durable clock gives `approve`, `reject`, or `expired`.
5. **Every way out after traffic moves must undo it.** That needs the state from before the change, recorded in history (`routes.snapshot`), and an undo that cannot itself be cancelled (`saga.Fail` on `rt.Detached()`). The compensation is registered before the canary runs, so a canary that fails half-way is undone too.
6. **The route table is shared with humans and other releases.** A whole-table `PUT` is only safe as an optimistic read-modify-write on one route with `If-Match`, returning every other route exactly as read.
7. **Activities run at least once.** Each route activity computes its desired state and does nothing when the gateway already has it, so a retry after an unrecorded `PUT` is harmless.
8. **The burn check must fail closed and be reproducible.** One instant query at `rt.Now()` (recorded, so a retry asks about the same window); no data, NaN, or a failed query means roll back.

### 5.2 The code, annotated

**The workflow, gates first.** Each step uses a fixed activity id from `releaseOpts`, so idempotency keys do not shift when a step is added:

```go
func ModelRelease(rt Runtime, spec ReleaseSpec) (ReleaseResult, error) {
	res := ReleaseResult{Served: spec.Served()} // "<model_id>-<version>"
	if err := spec.Validate(); err != nil {
		return res, &GateError{Gate: "spec", Reason: err.Error()} // before any activity
	}

	// 1. Gates that need no model: before any expensive work.
	var pre PreflightResult
	if err := rt.ExecuteActivity(ActPreflight, PreflightInput{ModelCard: spec.ModelCard, Ledger: spec.Ledger},
		releaseOpts("release-preflight"), &pre); err != nil {
		return res, gateFrom(err) // a non-retryable failure becomes a GateError
	}

	// 2. Export, 3. evaluate the exported model (EvalSuite), then gate on its rows.
	var ex ExportResult
	if err := rt.ExecuteActivity(ActExport, spec.ExportSpec(), releaseOpts("release-export"), &ex); err != nil {
		return res, err
	}
	// ...
	res.ModelDir = ex.Outputs[0]
	suites, err := EvalSuite(rt, spec.EvalInput(res.ModelDir)) // the exported dir is the subject
	// ... collect outputs, ReadResults (release.results)
	res.Gates = CheckGates(spec.Gates, ev.Reports, spec.ModelID)
	// ... any !g.OK: return &GateError{Gate: "eval", ...}
```

**The eval gate.** A gate starts as failed ("no row") and only a matching `ok` row with a value can make it pass. Both comparisons are inclusive:

```go
func CheckGates(gates []Gate, reports []EvalReport, subject string) []GateResult {
	out := make([]GateResult, 0, len(gates))
	for _, g := range gates {
		r := GateResult{Gate: g, Reason: "no row for this gate"} // fail closed by default
		for _, rep := range reports {
			if rep.Suite != g.Suite {
				continue
			}
			for _, row := range rep.Rows {
				if row.Model != subject || row.Task != g.Task || row.Metric != g.Metric {
					continue // another model's row says nothing about this one
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
				// ... reason when !r.OK
			}
		}
		out = append(out, r)
	}
	return out
}
```

**The preflight gates.** The placeholder pattern requires a letter after `<`, which is what lets `p < 0.05` through; HTML comments are stripped first because the template keeps its instructions in them. The ledger requires `train` in `allowed_uses`, no `revoked`, and at least one row:

```go
var placeholder = regexp.MustCompile(`<[A-Za-z][^<>\n]*>`) // "<model_id>", not "< 0.05"
var htmlComment = regexp.MustCompile(`(?s)<!--.*?-->`)

func LedgerProblems(ledger []byte) []string {
	// ... for each non-empty line:
		if err := json.Unmarshal([]byte(line), &row); err != nil || row.SourceID == nil {
			out = append(out, fmt.Sprintf("line %d: not a ledger row", i+1))
			continue
		}
		train := false
		for _, u := range row.AllowedUses {
			train = train || u == "train" // "eval" alone is not enough
		}
		if !train { /* ... does not allow train */ }
		if row.Revoked { /* ... is revoked */ }
	// ...
	if n == 0 {
		out = append(out, "the ledger has no sources") // undocumented data cannot ship
	}
	return out
}
```

`Preflight` turns a problem into `*GateError` (non-retryable: the same file gives the same answer) but returns a read error as is (retryable: a mount may come back).

**Approval, then the canary under a saga.** The snapshot is its own recorded activity, the restore is registered before the canary runs, and every failure from here goes through `saga.Fail`:

```go
	// 4. A human approves (or rejects) before any traffic moves.
	name, payload, err := rt.AwaitSignal([]string{SignalApprove, SignalReject},
		time.Duration(spec.ApproveTimeoutS)*time.Second)
	if err != nil {
		return res, err // a cancel here: nothing to undo
	}
	switch name {
	case SignalApprove:
		res.Approval = payload
	case SignalReject:
		res.Outcome, res.Approval = OutcomeRejected, payload
		return res, nil
	default: // "" : the timeout fired
		res.Outcome, res.Reason = OutcomeExpired, "no approve or reject signal before approve_timeout_s"
		return res, nil
	}

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
		return res, saga.Fail(rt, err) // undo on Detached(), return the original error
	}
```

**Bake, measure, decide.** The query time is `rt.Now()`, recorded in history. A cancel during the query still compensates; every other non-healthy result rolls back:

```go
	if err := rt.Sleep(time.Duration(spec.CanaryWaitS) * time.Second); err != nil {
		return res, saga.Fail(rt, err) // canceled during the bake: restore, never query
	}
	var s activities.Sample
	qerr := rt.ExecuteActivity(ActQuery, activities.QueryInput{Expr: spec.Burn.Query, At: rt.Now()}, releaseOpts("promql-burn"), &s)
	if IsCanceled(qerr) {
		return res, saga.Fail(rt, qerr)
	}
	healthy := false
	switch {
	case qerr != nil:
		res.Reason = "burn query failed: " + qerr.Error() // fail closed
	case s.NoData:
		res.BurnNoData, res.Reason = true, "the burn query returned no data" // fail closed
	default:
		v := s.Value
		res.Burn = &v
		healthy = v <= spec.Burn.Max // inclusive ceiling
		// ...
	}
	if healthy {
		// ... routes.promote; on failure saga.Fail
		res.Outcome = OutcomePromoted
		return res, nil
	}
	if err := saga.Compensate(rt); err != nil {
		return res, err
	}
	res.Outcome = OutcomeRolledBack
	return res, nil
```

Full solution: [`model_release.go`](../../../../course/solutions/go/workflows/model_release.go).

**Canary weights.** The new model gets $w$; every other backend keeps its proportion of the rest, scaled by $1 - w$. Rounding to $10^{-9}$ keeps `0.9` from printing as `0.9000000000000001`:

```go
func CanaryBackends(current []Backend, served string, w float64) []Backend {
	out := make([]Backend, 0, len(current)+1)
	rest := 0.0
	for _, b := range current {
		if b.ServedModel != served { // re-weighting an existing canary replaces it
			rest += b.Weight
		}
	}
	for _, b := range current {
		if b.ServedModel == served {
			continue
		}
		share := 0.0
		if rest > 0 {
			share = b.Weight / rest * (1 - w) // 0.5 / 1 * 0.9 = 0.45
		}
		out = append(out, Backend{ServedModel: b.ServedModel, Weight: round9(share)})
	}
	return append(out, Backend{ServedModel: served, Weight: round9(w)})
}
```

`BackendsOf` reads a route without `backends` as one backend named by the route's own `model` with weight 1, so the old model keeps its $1 - w$.

**The read-modify-write loop.** The table is kept as raw JSON per route; only the target route is decoded, changed, and re-encoded:

```go
func (r Routes) update(ctx context.Context, model string, fn func(route map[string]any) (bool, error)) (RouteResult, error) {
	// ... tries = MaxConflicts or 5
	for attempt := 0; ; attempt++ {
		routes, etag, err := r.Get(ctx) // []json.RawMessage, exactly as sent
		// ... find idx and decode route; missing route: Permanent{ErrNoRoute}
		changed, err := fn(route)
		// ...
		if !changed {
			return RouteResult{Changed: false, Backends: backends, ETag: etag}, nil // desired state already there: no PUT
		}
		b, err := marshalPlain(route) // no HTML escaping: ">=" stays ">="
		// ...
		next := append([]json.RawMessage(nil), routes...)
		next[idx] = b // every other route byte for byte as read
		tag, err := r.Put(ctx, next, etag) // If-Match: etag
		if errors.Is(err, ErrStale) && attempt+1 < tries {
			continue // 412: someone else changed the table; reread and redo fn
		}
		// ...
		return RouteResult{Changed: true, Backends: backends, ETag: tag}, nil
	}
}
```

`marshalPlain` exists because `encoding/json` escapes `>` as `>` even inside a `json.RawMessage`, which would rewrite a cascade route's `"accept_if": "mean_logprob >= -1.5"` on every `PUT`.

**The idempotent canary.** Weight is validated first (permanent), readiness second (retryable), and the closure returns "unchanged" when the canary is already in place:

```go
func (r Routes) Canary(ctx context.Context, in CanaryInput) (RouteResult, error) {
	if !(in.Weight > 0 && in.Weight < 1) {
		return RouteResult{}, Permanent{fmt.Errorf("routes: canary weight %v is not in (0, 1)", in.Weight)}
	}
	ok, err := r.Ready(ctx, in.Served) // a live, non-draining worker reports the model
	// ...
	if !ok {
		return RouteResult{}, fmt.Errorf("%w: %q", ErrNoWorkers, in.Served) // retryable
	}
	return r.update(ctx, in.Route, func(route map[string]any) (bool, error) {
		cur, err := BackendsOf(route)
		// ...
		for _, b := range cur {
			if b.ServedModel == in.Served && math.Abs(b.Weight-in.Weight) <= 1e-9 {
				return false, nil // already applied by an earlier attempt
			}
		}
		route["backends"] = CanaryBackends(cur, in.Served, in.Weight)
		return true, nil
	})
}
```

`Restore` replaces every key of the route with the snapshot's (not a merge), and refuses a snapshot of another route. Full solution: [`routes.go`](../../../../course/solutions/go/activities/routes.go).

**The PromQL answer.** Prometheus encodes sample values as strings so that `NaN` and `+Inf` survive JSON. NaN is no data; `+Inf` is a real (very bad) value, never no data:

```go
func ParseSample(pair []json.RawMessage) (float64, bool, error) {
	// ... pair must be [time, "value"]
	v, err := strconv.ParseFloat(s, 64)
	// ...
	if math.IsNaN(v) {
		return 0, true, nil // 0/0: the canary saw no traffic
	}
	return v, false, nil
}

func Decode(body []byte) (Sample, error) {
	// ... status must be "success", else ErrQuery
	switch d.ResultType {
	case "scalar":
		// ... one pair
	case "vector":
		// ...
		switch len(series) {
		case 0:
			return Sample{NoData: true}, nil // nothing matched the canary's labels
		case 1:
			// ... ParseSample(series[0].Value)
		default:
			return Sample{}, fmt.Errorf("%w: %d series; the expression must aggregate to one (wrap it in sum(...))", ErrQuery, len(series))
		}
	default: // matrix, string
		return Sample{}, fmt.Errorf("%w: result type %q; want an instant vector or a scalar", ErrQuery, d.ResultType)
	}
}
```

`Query` sends `time` as Unix seconds with millisecond precision, returns a 5xx as a plain (retryable) error, and wraps any `Decode` error in `Permanent`: asking again gives the same answer. Full solution: [`promql.go`](../../../../course/solutions/go/activities/promql.go).

### 5.3 Why this solution

- **Cost.** A release is a constant number of activities plus one per suite; the route update is $O(R)$ in the number of routes per attempt, with at most `MaxConflicts` rereads. The bake costs nothing while it waits: it is a timer in history, not a held worker.
- **A workflow over a script or a CI job.** A script loses its place when the machine sleeps, and a CI job cannot wait a day for approval or restore a route after being killed. The workflow gets both from the engine.
- **One burn query at the end over continuous analysis.** Argo Rollouts and Flagger sample metrics repeatedly and abort early; one query is simpler to reason about and replay, at the price of exposing $w$ of traffic for the full bake even when the canary is failing from the first minute.
- **Optimistic concurrency over a lock.** A lock on the route table would need its own lease and expiry, and a crashed release would block operators. `If-Match` costs nothing until two writers actually collide.
- **Accepted tradeoffs.** A single canary step (no 1%, 5%, 25% ramp); the burn check compares against an absolute ceiling, not against a baseline running at the same time (see Kayenta in Going further).

### 5.4 Explain it back

1. Why is the compensation registered *before* the canary activity runs, not after it succeeds?
   <details><summary>Answer</summary>The canary activity may `PUT` the new table and then fail (or the worker may die before the completion is recorded and the retries run out). Traffic has then moved even though the activity "failed". With the restore already registered, `saga.Fail` puts the snapshot back; restoring an unchanged route is a no-op, so registering early costs nothing.</details>
2. Why does the burn query carry `At: rt.Now()` instead of letting Prometheus use the current time?
   <details><summary>Answer</summary>The activity can be retried, possibly minutes later. With a fixed evaluation time every attempt measures the same 5-minute window right after the bake; without it a retry would measure a different window and the decision would depend on when the retry happened.</details>
3. Why do no data and NaN roll back instead of promoting?
   <details><summary>Answer</summary>They mean the canary served no matched requests (or the labels are wrong), so there is no evidence it works. A check that passes on no evidence is not a check; failing closed costs one more release attempt, failing open could ship a model that fails every request.</details>
4. In the crash example, why does the second canary attempt not turn 0.9/0.1 into 0.81/0.09/0.1?
   <details><summary>Answer</summary>`Canary` reads the table, sees `tinystories-10m-v2` already at weight 0.1, and returns `Changed: false` without a `PUT`. And even if the weight differed, `CanaryBackends` excludes the served model from `rest` and replaces it, so re-applying never compounds.</details>
5. Why is a gate failure a `*GateError` that implements `NonRetryable()`, rather than a plain error?
   <details><summary>Answer</summary>A plain error would be retried by the SDK, and a forbidden license or a failing eval row gives the same answer every time. The release would burn its attempts and then be reported as an outage instead of as a refusal.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. gate results computed but not enforced | a model that failed its safety suite takes traffic | any `!g.OK` returns a `GateError{eval}` before the approval wait |
| 2. a gate with no row passes | the safety suite crashed and the release went out | each `GateResult` starts as failed with "no row for this gate" |
| 3. `>=` written as `>` | a model exactly at its threshold is refused | `CheckGates` compares with `>=` and `<=` |
| 4. the ledger check reads `eval` as enough, or skips `revoked` | a non-commercial or revoked source ships | `LedgerProblems` requires `train` in `allowed_uses` and flags `revoked` |
| 5. a template copied but never filled passes | `<model_id>` in a published model card | `ModelCardProblems` finds `<letter...>` placeholders outside HTML comments |
| 6. a gate failure returned as a plain error | the gate is retried three times, then reported as an outage | `Preflight` returns `*GateError`; `gateFrom` keeps it a gate refusal |
| 7. evaluating the checkpoint, not the exported model | the int4 export ships unmeasured | `EvalInput(res.ModelDir)` names the exported directory |
| 8. reject, timeout, or the wait itself missing | a rejected model ships; a release waits forever; traffic moves unapproved | `AwaitSignal` with a timeout before the snapshot; `reject` and `""` end the run |
| 9. the snapshot taken after the canary, a rollback branch that never restores, or a restore that merges | rollback keeps the canary's backends | snapshot first, restore registered in the saga, `Restore` replaces every key |
| 10. no bake, or a query at no fixed time | the burn is measured before the canary served anything; a retry measures another window | `rt.Sleep(canary_wait_s)`, then `QueryInput{At: rt.Now()}` |
| 11. no data or a failed query read as healthy | a canary that served nothing, or that Prometheus could not see, is promoted | `healthy` starts false and only a real value at or below the ceiling sets it |
| 12. a strict ceiling | a canary exactly at the budget line rolls back | `v <= spec.Burn.Max` |
| 13. `PUT` without `If-Match`, or giving up on 412 | 428 on every change; a concurrent edit kills the release | `Put` always sends `If-Match`; `update` rereads on `ErrStale` |
| 14. the table rebuilt instead of returned as read | `>=` comes back as `>=`; other routes vanish | routes kept as `json.RawMessage`; only the target is re-encoded, with `marshalPlain` |
| 15. a canary that is not idempotent | a retried activity makes 0.9/0.1 into 0.81/0.09/0.1 | `Canary` returns unchanged when the weight is already in place |
| 16. weights not rescaled, or a missing `backends` read as none | the gateway rejects the table (sum not 1); the old model loses its share | `CanaryBackends` scales by $1 - w$; `BackendsOf` defaults to the route's model |
| 17. no check that a worker serves the new model | 10% of requests fail with no backend | `Canary` calls `Ready` and returns retryable `ErrNoWorkers` |
| 18. permanent errors returned as retryable | an unknown route or a bad expression retries for minutes | `Permanent{...}` for unknown routes, bad weights, 400 tables, and decode errors |
| 19. the first of several series used | the check reads one random pod | `Decode` refuses more than one series |
| 20. scalars refused, or a 503 treated as permanent | `scalar(...)` queries fail; one overload rolls back a good model | `Decode` accepts `scalar`; `Query` returns a 5xx as retryable |
| 21. a canary weight of 1 accepted | the "canary" is a full cutover with no way back but a rollback | `Validate` and `Canary` require $0 < w < 1$ |
| 22. a cancel that does not compensate | a canceled release leaves 10% of traffic on an unapproved model | `Sleep` and query cancels go through `saga.Fail` |
| 23. activity ids left to the runtime's counter | an added step renumbers every later idempotency key | `releaseOpts` gives every step a fixed id |
| 24. `time.Now()`, `rand`, or map order in the workflow | replay diverges after a restart (`ErrNondeterminism`) | the workflow reads only `rt.Now()`, recorded results, and its input |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `dur.08` | the `Runtime` the workflow runs on, and the `Saga` that undoes the canary |
| Back | `dur.11` | `EvalSuite` is the evaluation step |
| Back | `dur.09` | `export` and `eval` are subprocess activities run by the `dur.09` runner |
| Back | `data.08` | the ledger the preflight gate reads |
| Back | `gw.05`, `gw.07` | the admin route and worker API this client speaks |
| Back | `obs.03` | the burn-rate expressions and thresholds of the SLO rules |
| Forward | `C1` | the capstone model is released through this workflow; C1 runs `Validate` and `Preflight` on its release |
| Forward | `ops.08` | the data-incident drill revokes a source and expects the next release to fail its ledger gate |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ModelRelease` | Argo Rollouts, Flagger | progressive steps (1%, 5%, 25%, ...), several metric analyses per step, automatic abort, traffic mirroring | `argo-rollouts/rollout/canary.go`, Flagger `pkg/controller/scheduler.go` |
| the burn check | Kayenta (Spinnaker) | statistical canary analysis: the canary compared with a baseline started at the same time, Mann-Whitney per metric | `kayenta-judge` |
| `Runtime` | Temporal Go SDK | `workflow.Context`, `workflow.NewSelector` over signals and timers, versioning with `GetVersion` | `go.temporal.io/sdk/workflow` |
| gates | MLflow model registry, Vertex AI Model Registry | stage transitions with approval, lineage to data and runs | MLflow `model_registry` docs |
| `Routes` | Envoy AI Gateway, Gateway API `HTTPRoute` weights | weighted backends as Kubernetes objects; the API server's `resourceVersion` is the ETag | Gateway API `backendRefs[].weight` |
