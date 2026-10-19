# Performance regression bisect record

## Reproduction envelope

| Field | Fixed value for this example |
|---|---|
| Metric | time to first token (TTFT), milliseconds |
| Workload | `fixtures/perf/ttft-64-prompts.jsonl`, 64 prompt cases, 128-token output cap |
| Host | one pinned CI runner class; no concurrent benchmark jobs |
| Configuration | same model digest, tokenizer digest, batch limit, and runtime config at every revision |
| Sampling | 2 warm-up rounds, then 7 runs; compare medians and p95, retain raw JSON |
| Regression rule | candidate median is at least 15% slower and the increase exceeds 3 ms |

The example report uses hypothetical measurements to show the decision method. Replace them with raw output from the same host before opening a performance incident.

| Revision | Median TTFT | p95 TTFT | Raw result |
|---|---:|---:|---|
| `base` | 41 ms | 55 ms | attach `base.json` |
| `head` | 50 ms | 67 ms | attach `head.json` |

The median increase is 22.0%, or 9 ms, so the candidate crosses both predeclared regression limits. The p95 moves in the same direction. The check is not accepted if the baseline itself fails to reproduce within 5% over three repetitions.

## Bisect procedure

1. Pin the workload, model, host class, and runner configuration before inspecting commits.
2. Verify `base` is good and `head` is bad using the regression rule above.
3. Run `git bisect run ./scripts/bench-ttft-gate.sh`; the script returns 0 for good, 1 for bad, and a distinct infrastructure error for an invalid run.
4. Repeat the first bad revision and both neighbors with seven runs. A single noisy midpoint is not a finding.
5. Attach raw results, the bisect log, and the identified commit to the issue. The owner proposes a fix or documents why the change is intentional.

**Rollback:** revert the identified change or pin the prior image digest if the regression is already deployed. Keep the workload and raw results with the incident. **Owner:** serving performance on-call. **Approval threshold:** reviewer confirms endpoint reproducibility and the final rerun is below both limits.
