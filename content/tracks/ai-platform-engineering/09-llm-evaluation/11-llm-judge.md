<!-- ol:module ag.11 -->
# LLM judge with position control

## Overview

| | |
|---|---|
| **Module** | `ag.11` · build · Go · Pass 10 · 3 h |
| **You build** | rubric-based score parsing, pairwise comparisons with position swaps, and sampled scoring |
| **Tests** | `course/tests/go/ag_11/`: `TestJudgeParsesEvidence`, `TestJudgeRejectsMalformedOutput`, `TestPairwisePositionSwap`, `TestSampledTolerance`, `TestKappaAgainstLabels` |
| **Needs** | `ag.01` provider and `ag.09` runner; reading: `M07.4` |
| **Used by** | `ag.12` experiment runner, `craft.23` |
| **Milestone** | MS-agent |

## 1. Why now

Code scorers in ag.10 are reliable when the target is explicit, such as an exact answer or a state delta. They cannot judge whether an explanation is useful or whether a response follows a nuanced rubric. A language model can help, but its output must be treated as a noisy measurement.

## 2. Principles

Ask for a constrained JSON response with a numeric score and short evidence. Reject malformed output, out-of-range values, and missing evidence. Never turn a judge failure into score zero. For pairwise comparisons, ask both A/B and B/A, map the second answer back to the original labels, and resolve disagreement using a fixed rule.

When repeated human labels are available, Cohen's kappa is `κ = (p_o - p_e)/(1 - p_e)`, where `p_o` is observed agreement and `p_e` is agreement expected from the label marginals. Sampling reduces judge cost; report both the number sampled and tolerance used.

## 3. Worked example

For a pair of responses, the first prompt says A then B and returns `A`. The swapped prompt says B then A and returns `A`, which maps to original B. The judge disagrees across positions, so the comparison is marked position-sensitive and cannot silently count as a clean win. `TestPairwisePositionSwap` catches a judge that prefers the first option regardless of quality.

## 4. The interface

Build `NewJudgeScorer`, pairwise scoring, kappa reporting, and `Sampled`. The generator interface should accept a prompt and return raw JSON text, which lets tests provide deterministic responses without network access. `TestJudgeParsesEvidence` and `TestJudgeRejectsMalformedOutput` check response validation; `TestPairwisePositionSwap` catches position bias; `TestSampledTolerance` checks sample accounting; `TestKappaAgainstLabels` checks agreement against frozen labels.

| Test | What it checks |
|---|---|
| `TestPairwisePositionSwap` | hand-worked position swap and winner mapping |
| `TestJudgeParsesEvidence` | valid rubric response parsing |
| `TestJudgeRejectsMalformedOutput` | malformed and out-of-range responses |
| `TestSampledTolerance` | deterministic sample selection |
| `TestKappaAgainstLabels` | agreement against frozen labels |

```bash
ol start ag.11
ol tests ag.11
ol check ag.11
```

## 5. Pitfalls

- Parse and validate the full response; do not extract the first digit from prose.
- Swap candidate positions and invert the second result before combining.
- Keep judge errors distinct from low scores.
- A sampled scorer's uncertainty is not the same as its inner scorer's value spread.

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `ag.01` | provides the provider used by the judge generator |
| Back | `ag.09` | stores judge results and scorer failures |
| Forward | `ag.12` | compares versions with code and judge scorers |
| Forward | `craft.23` | uses judge scores as evaluation evidence |

MS-agent reports the judge's sample count, tolerance, and agreement with the frozen labels fixture.

## Going further

| Your piece | Production equivalent | What it adds |
|---|---|---|
| pairwise position swap | LMSYS Chatbot Arena | many human comparisons aggregated into model rankings |
| sampled judge | eval budget policy | cost controls with a measured quality tolerance |
| rubric | human annotation guide | calibrated labels for judge validation |
