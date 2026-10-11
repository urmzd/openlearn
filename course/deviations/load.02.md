# load.02 deviations

## load.02: optional (`kind = "side"`), since no module calls the gate

`load.02` failed verify check 9 (no call site): every user of the gate goes
through the learner's `{loadgen} compare` entry point (D16), which is not a
call site. It is now a side module: optional, checked on its own, gating no
milestone.

- `dep.05`'s `perf-gate` job runs `go/bin/loadgen compare <base> head.json
  --metric ttft_p95 --max-regress 5%`. Its check reads the workflow
  statically and calls no load.02 code, so load.02 stays in `dep.05`'s
  `reading` (B95-04). The job is now optional: `test_required_jobs` requires
  lint, unit, images, and kind-e2e, and
  `test_perf_gate_runs_on_main_against_the_last_report` checks the perf gate
  only when the workflow has one. A present perf gate is held to the same
  rules as before (main only, `compare` with metric and budget, report
  uploaded).
- `ops.07` (drill) and `craft.16` (practice, docs) run the same command under
  `git bisect run`; ops.07 also accepts `ol bench course --assert`.
- No library module needs an unpaired two-sample gate over load reports.
  `ag.12`'s A/B experiments are paired and use a sign-permutation test over
  eval scores. `dur.12`'s canary gate reads PromQL burn rates, not load
  reports.
- `load.01.used_by` drops `load.02`; `ag.07`, `ag.09`, and `ag.12` remain its
  call sites. `MS-L10` and `MS-prod` no longer require load.02, and MS-L10's
  `compare-a-run-with-itself` step is gone (load.02's course tests cover the
  identical-runs case statistically, `TestIdenticalRunsPassAndShiftsAreCaught`).

It returns to core if a later Go module gains a library unit that takes two
`loadgen.Report`s and acts on the verdict, for example a `dur.12` canary
activity that calls `compare.Compare`.
