<!-- ol:module craft.23 -->
# Agent evals as tests (R9)

## Overview

| | |
|---|---|
| **Module** | `craft.23` · practice · Go · Pass 10 · 2 to 3 h |
| **You build** | a repeatable R9 eval suite and learner tests for agent behavior |
| **Contract** | [`eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json) |
| **Tests** | `course/tests/craft.23/` (why: repeatability, scorer failures, and semantic mutant detection) |
| **Needs** | `ag.09` runner, `ag.10` scorers, `ag.11` judge, `ag.12` experiments |
| **Used by** | `MS-agent` gates citation quality, tool safety, and A/B behavior |
| **Milestone** | [MS-agent](../../../../course/milestones/MS-agent.toml) |
| **Optional depth** | Online shadow evaluation and human preference review |

## Key takeaways

- Each eval observation carries a stable id, sample, timing, and ground truth.
- Scorers measure distinct properties and return errors explicitly.
- Confidence intervals describe uncertainty; they do not prove a universal quality claim.
- Semantic mutants check that the suite catches plausible regressions.

## How to work this chapter

```bash
ol start craft.23
ol tests craft.23
ol check craft.23
```

## 1. Why now

Agent output depends on model sampling, retrieval, and tool execution. A unit test for one canned answer misses regressions in citations, tool success, latency, and state changes. R9 uses a fixed suite to check those behaviors as a release gate.

## 2. Principles

The evaluator runs a subject on each observation and applies named scorers. For an estimated score difference, report a 95% paired interval. A/B comparisons use the same inputs in base and experiment. Judge scorers are sampled and calibrated against human labels; parse errors count as scorer errors, never zero scores.

| Symbol | Meaning |
|---|---|
| `n` | scored observations |
| `d_i` | paired base-to-experiment difference |
| `CI` | confidence interval for mean `d_i` |

## 3. Worked example

Three paired tool-success differences `[0, 1, 0]` have mean `1/3`. The sample is too small to establish a stable improvement. The suite stores all three outcomes and the interval, rather than reporting only the fraction as a definitive result.

## 4. The artifact and its check

Build `go/tests/agent-evals/suite.json` with a fixed seed, disjoint train, validation, and held-out example IDs, plus distinct citation, tool outcome, state-diff, prompt-injection, and latency scorers. Each observation names its split, scenario, expected tool, and expected state. `test_eval_run_is_reproducible` checks that every ID belongs to exactly one declared split. `test_agent_mutants_have_separate_ci_coverage` checks for dedicated citation, tool-success, and state-change scorers. Then run semantic agent mutants one at a time and compare paired outcomes with a 95% interval; never tune against held-out cases.

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Aggregate unrelated scores into one average | `test_eval_run_is_reproducible`; mutant `s01` |
| Drop failed judge calls silently | `test_eval_run_is_reproducible`; mutant `s02` |
| Tune on the held-out eval suite | `test_agent_mutants_have_separate_ci_coverage`; mutant `s03` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.09` | Supplies repeatable execution and observation capture. |
| Back | `ag.10` | Supplies named scorers with explicit errors. |
| Back | `ag.11` | Supplies calibrated judge scoring. |
| Back | `ag.12` | Supplies paired experiment analysis. |
| Forward | `MS-agent` | Runs the learner's docs suite, prompt-injection fixtures, and paired experiment. |
| Forward | `MS-C2` | Reuses paired analysis for post-training outcomes. |

## Going further

Online evaluations need consent, privacy controls, and rollback thresholds. Keep offline fixtures as a stable regression baseline even after adding production telemetry.
