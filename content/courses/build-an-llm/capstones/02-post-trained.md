<!-- ol:module C2 -->
# Capstone 2: post-trained TinyStories model

## Overview

| | |
|---|---|
| **Module** | `C2` · optional practice · Python and Go · Pass 10 · 1 to 2 days |
| **You build** | an SFT and preference or verifiable-reward run, a model artifact, and an updated model card |
| **Contract** | `formats/checkpoint.md`, `formats/eval-results.schema.json`, and the `L12.*` contracts |
| **Tests** | `course/tests/C2/` (why: records run provenance and prevents unsupported model-card claims) |
| **Needs** | `C1`, `L12.1`, `L12.2`, `L12.3`, `L12.4` |
| **Used by** | `MS-C2` checks held-out reward improvement |
| **Milestone** | [MS-C2](../../../../course/milestones/MS-C2.toml) |
| **Optional depth** | `L12.4` distillation, or compare all three adaptation methods |

## Key Takeaways

- Post-training starts from the C1 checkpoint and keeps a reproducible base reference.
- The held-out suite is fixed before training and reports uncertainty.
- The model card distinguishes measured outcomes from intended behavior.

## How to work this chapter

```bash
ol start C2
ol milestone MS-C2 --smoke
```

## 1. Why now

C1 proves the model can be trained, evaluated, released, and served. This capstone asks whether demonstrations and preference or verifiable-reward data improve a specific behavior while preserving language-model quality.

## 2. Principles

Use the same held-out examples before and after adaptation. For a score difference `d_i = score_after_i - score_before_i`, report the mean difference and a paired confidence interval; the comparison preserves prompt difficulty. Do not claim a general safety improvement from a small task-specific dataset. Keep a frozen copy of C1 and record dataset revision, tokenizer, template, seed, optimizer, and objective.

| Symbol | Meaning |
|---|---|
| `d_i` | per-example paired score change |
| `n` | number of held-out examples |
| `CI` | uncertainty interval for the mean paired change |

## 3. Worked example

If three held-out reward differences are `[0.2, 0.1, 0.3]`, the mean gain is `0.2`. This tiny sample is not enough to claim a reliable improvement; the capstone uses a larger fixture and requires a paired interval excluding zero with `p < 0.05`.

## 4. Interface and tests

Train from C1 using SFT, then DPO or GRPO; distillation is a documented alternate. Evaluate reward and language-model regression, compare with paired bootstrap or permutation, and publish an artifact manifest. The model card records intended uses, limitations, data provenance, evaluation setup, and third-party teacher/model licenses. Smoke checks verify paths and schema; the full optional milestone checks statistical gain.

## 5. Pitfalls

| Pitfall | Caught by |
|---|---|
| Evaluate on training pairs | Held-out split check; mutant `s01` |
| Report only the best seed | Run manifest check; mutant `s02` |
| Omit base-model provenance | Model-card artifact check; mutant `s03` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `C1` | Provides the pretrained checkpoint and baseline evaluation. |
| Back | `L12.1` | Converts demonstrations into an SFT checkpoint. |
| Back | `L12.2` | Applies preference-pair optimization. |
| Back | `L12.3` | Applies verifiable reward optimization. |
| Back | `L12.4` | Provides distillation as an alternate recipe. |
| Forward | `MS-C2` | Evaluates the post-trained artifact. |
| Forward | `MS-P10` | Remains the agent pass gate; the capstone is optional. |

## Going further

Larger alignment programs add red-team review, multiple annotator cohorts, data governance, and staged deployment. Those processes are required before treating an offline score as a product benefit.
