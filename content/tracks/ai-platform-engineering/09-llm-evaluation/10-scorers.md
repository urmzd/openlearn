<!-- ol:module ag.10 -->
# Agent evaluation scorers

## Overview

| | |
|---|---|
| **Module** | `ag.10` · build · Go · Pass 10 · 3 to 4 h |
| **You build** | deterministic scorers for text, JSON schema, stream timing, tool use, retrieval, citations, and state changes |
| **Tests** | `course/tests/go/ag_10/`: `TestTextAndSchemaScorers`, `TestTimingMetrics`, `TestRankingHandExample`, `TestStateChangeCriteria` |
| **Needs** | `ag.09`, `ag.02`, `ag.08`; reading: `L4.5` and case study 05 |
| **Used by** | `ag.12` experiment runner, `craft.23` |
| **Milestone** | MS-agent |

## 1. Why now

The runner from ag.09 can execute cases and report measurements, but it does not define what a good answer means. This module turns common agent outcomes into repeatable measurements. Each scorer is small and deterministic, so the same observation always produces the same score.

## 2. Principles

Keep the measurement narrow. Exact match checks normalized answer text. Ranking metrics inspect ordered chunk ids. A state grader checks the resulting state, not the agent's explanation. A missing input for a measurement is an error, because a missing ground truth is not evidence of failure.

For retrieval, let relevance grade at rank `i` be `g_i`. Then `DCG@k = sum((2^g_i - 1) / log2(i + 1))`, for ranks starting at one. `nDCG@k` divides by the ideal ranking's DCG. `MRR` is the reciprocal rank of the first relevant result; `hit@k` is one if any relevant result occurs in the first k.

For state changes, compute `delta = after - before`. The task passes only when required changes are a subset of delta and delta is a subset of allowed changes. This catches both missing work and unrequested work.

## 3. Worked example

Suppose relevant chunks are `{a: 2, c: 1}` and retrieval returns `[b, a, a, c]`. Repeated ids count at their first rank only. `hit@2 = 1`, `MRR = 1/2`, and DCG uses grades `[0, 2, 1]` at ranks one through three. The ideal grades are `[2, 1]`. Keeping this example hand-computable makes off-by-one rank bugs visible.

## 4. The interface

Implement the constructors in `go/agent/eval/scorers`: `Exact`, `Regex`, `Schema`, `TTFT`, `TTLT`, `ITL`, `ToolSuccess`, `ToolCalled`, `HitAtK`, `MRR`, `NDCG`, citation scorers, and state grading. `TestTextAndSchemaScorers` checks normalization and schema failures; `TestTimingMetrics` checks first-token and inter-token calculations; `TestRankingHandExample` checks duplicate removal and rank discounts; `TestStateChangeCriteria` checks required and forbidden changes.

| Test | What it checks |
|---|---|
| `TestRankingHandExample` | hand-computed duplicate-free ranking metrics |
| `TestTextAndSchemaScorers` | normalized exact match and schema rejection |
| `TestTimingMetrics` | first token and inter-token timing |
| `TestStateChangeCriteria` | required and forbidden state changes |

```bash
ol start ag.10
ol tests ag.10
ol check ag.10
```

## 5. Pitfalls

- An answer with no citation has undefined precision; return an error. Citation recall is zero.
- A missing token stream is not zero latency. Return a scorer error.
- Deduplicate retrieved chunk ids before ranking.
- Distinguish an absent field from a JSON `null` value in state diffs.
- Exclude bookkeeping fields such as `updated_at` from user-visible changes.

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.09` | provides the observation, scorer, and runner interfaces |
| Back | `ag.02` | provides shared evaluation observations and score types |
| Back | `ag.08` | provides citation parsing and resolution |
| Forward | `ag.12` | computes paired deltas for every scorer |
| Forward | `craft.23` | uses agent evaluations as test evidence |

The helpdesk-world fixture exercises both text and state-change outcomes, and MS-agent reports the scorer results with confidence intervals.

## Going further

| Your piece | Production equivalent | What it adds |
|---|---|---|
| deterministic scorers | saige eval package | composable and versioned measurements |
| retrieval scores | Ragas | faithfulness and context precision/recall |
| state grader | application-specific oracle | explicit before/after invariants |
