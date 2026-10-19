# Capstone 2: post-trained TinyStories model

## Overview

| | |
|---|---|
| **Module** | `C2` · optional practice · Python and Go · Pass 10 · about 1 h reading; a full run takes 1 to 2 days |
| **Solution** | an SFT and preference or verifiable-reward run recorded as a run manifest, a model artifact, and an updated model card. Reference (provenance smoke tier): [`artifacts/c2/run.json`](../../../../course/solutions/entry/artifacts/c2/run.json) and [`docs/capstone/c2/MODEL_CARD.md`](../../../../course/solutions/entry/docs/capstone/c2/MODEL_CARD.md); the paired statistics are `bootstrap_ci` in [`prob/stats.py`](../../../../course/solutions/python/tinyllm/prob/stats.py) and `paired_permutation_test` in [`prob/tests.py`](../../../../course/solutions/python/tinyllm/prob/tests.py) |
| **Contract** | [`formats/checkpoint.md`](../../../../course/contracts/formats/checkpoint.md), [`formats/eval-results.schema.json`](../../../../course/contracts/formats/eval-results.schema.json), and the `L12.*` contracts |
| **Needs** | `C1`, `L12.1`, `L12.2`, `L12.3`, `L12.4` |
| **Used by** | nothing downstream: the capstone is optional, and its end-to-end result is a held-out reward improvement over the C1 base |
| **Optional depth** | `L12.4` distillation, or compare all three adaptation methods |

## Key Takeaways

- Post-training starts from the C1 checkpoint and keeps a reproducible base reference.
- The held-out suite is fixed before training and reports uncertainty.
- The model card distinguishes measured outcomes from intended behavior.

## How to read this chapter

Sections 1 to 3 state what the capstone must show and work a paired comparison by hand. Section 4 fixes what the run produces. Section 5 walks through the reference run manifest and model card, field by field, and the paired statistics that decide whether an improvement is real.

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

If three held-out reward differences are `[0.2, 0.1, 0.3]`, the mean gain is `0.2`. This tiny sample is not enough to claim a reliable improvement; the capstone uses a larger held-out set and requires a paired interval excluding zero with `p < 0.05`.

The paired permutation test shows why exactly. Under the null hypothesis (adaptation changed nothing), each pair's two scores could have come from either model, so each $d_i$ keeps or flips its sign with probability 1/2. With $n = 3$ there are $2^3 = 8$ equally likely sign patterns, and the observed statistic is $|\sum d_i| = 0.6$:

| Signs | $\sum s_i d_i$ | $\ge 0.6$ in absolute value? |
|---|---|---|
| $+ + +$ | 0.6 | yes |
| $- - -$ | $-0.6$ | yes |
| $- + +$ | 0.2 | no |
| $+ - +$ | 0.4 | no |
| $+ + -$ | 0.0 | no |
| $- - +$ | 0.0 | no |
| $- + -$ | $-0.4$ | no |
| $+ - -$ | $-0.2$ | no |

The exact two-sided p-value is $2/8 = 0.25$. And no three differences can ever do better: the two all-same-sign patterns always reach the observed value, so $p \ge 2/8$ for any data. A sample this small cannot pass the bar, whatever the model does.

## 4. What the run produces

Train from C1 using SFT, then DPO or GRPO; distillation is a documented alternate. Evaluate reward and language-model regression, compare with paired bootstrap or permutation, and publish an artifact manifest. The model card records intended uses, limitations, data provenance, evaluation setup, and third-party teacher/model licenses. The smoke tier checks that the paths and the manifest's fields are present and consistent; the full run must show a statistical gain.

### What a correct solution must handle

| Case | Why it matters |
|---|---|
| a run manifest with every provenance field | the base checkpoint, dataset revision, tokenizer, template, seed, optimizer, and objective reproduce the run |
| a held-out split disjoint from training | a reward measured on training pairs measures memorization |
| claims that match the evidence | the card and manifest claim an improvement only when the paired interval excludes zero |
| every seed reported | reporting only the best seed turns noise into a result |

## 5. The solution, step by step

### From first principles

1. **The comparison must be paired.** Prompts differ in difficulty far more than the two models differ, so the unit of comparison is one prompt scored by both the C1 base and the candidate; $d_i$ removes the prompt's difficulty.
2. **The held-out set is fixed before training,** or it becomes another thing that was tuned. So the manifest names its source and split, and names the base it is paired with.
3. **Two statistics answer two questions.** The bootstrap CI of $\bar d$ says how large the gain plausibly is; the paired permutation test says whether a gain of this size could come from chance. A claim needs the CI to exclude zero and $p < 0.05$.
4. **Provenance makes the run reproducible.** The base checkpoint, data revision, tokenizer, chat template, seed, optimizer, precision, and objective are each something that, if changed silently, changes the result.
5. **Claims are data, not prose.** Recording `measured_heldout_improvement` and `general_safety_improvement` as booleans lets a check compare them with the evaluation instead of parsing a sentence.

### The reference run, annotated

**The run manifest.** [`run.json`](../../../../course/solutions/entry/artifacts/c2/run.json) is a provenance smoke artifact: it has every field a full run records, and it truthfully claims nothing.

```json
{
  "format": "tinyllm-post-training-run-v1",
  "run_id": "c2-reference-smoke",
  "base_checkpoint": "artifacts/c1/model",
  "candidate_model": "artifacts/c2/model",
  "dataset_revision": "post-training-fixtures-v1",
  "training_data": {"revision": "post-training-fixtures-v1", "source": "...distill.jsonl", "split": "train"},
  "heldout": {"source": "...heldout.jsonl", "split": "heldout", "paired_with": "artifacts/c1/model"},
  "objective": "sft",
  "seed": 0,
  "tokenizer": "artifacts/c1/model/tokenizer.json",
  "chat_template": "tinystories-chat-v1",
  "optimizer": "adamw",
  "precision": "f32",
  "evaluation": {
    "metric": "paired_reward_delta",
    "confidence_interval": "paired bootstrap, 95%",
    "significance_test": "paired permutation, two-sided"
  },
  "claims": {"measured_heldout_improvement": false, "general_safety_improvement": false},
  "limitations": ["...records provenance only...", "...too small to support general claims..."]
}
```

Read it against the principles. `base_checkpoint` and `heldout.paired_with` name the same C1 model, so the comparison is paired against the real base, not a different snapshot. `training_data.split` is `"train"` and `heldout.split` is `"heldout"`, two different files: the held-out reward is never measured on training pairs. The tokenizer is C1's own `tokenizer.json`, so the candidate encodes text exactly as the base did. The evaluation block fixes the method before any number exists. And because this smoke artifact has no trained weights, both claims are `false`: the manifest says what was measured, not what was hoped.

**The model card.** [`MODEL_CARD.md`](../../../../course/solutions/entry/docs/capstone/c2/MODEL_CARD.md) repeats the same discipline in prose. It names the base checkpoint and the objective, points at the manifest for the rest of the provenance, and says in its evaluation section that the paired bootstrap interval and the permutation test are the plan and that this artifact "records the evaluation plan, not measured results". Its limitations section states that the fixture cannot establish a reliable improvement. A full run replaces that section with the measured $\bar d$, its 95% CI, the p-value, the language-model regression (held-out bpb before and after), and every seed's result.

**The statistics.** The CI is the course's percentile bootstrap applied to the differences with `stat = mean` (the same function C1 uses, quoted in C1 section 5). The significance test is the sign-flip permutation test from section 3:

```python
def paired_permutation_test(a, b, n_perm, rng) -> float:
    x, y = _vector(a, "a"), _vector(b, "b")
    if x.shape != y.shape:                 # pairs must line up one to one
        raise ValueError(f"a and b must pair up, got lengths {x.size} and {y.size}")
    n_perm = _n_perm(n_perm)
    d = x - y
    t_obs = abs(float(np.sum(d)))          # two-sided: the size of the total change
    bar = t_obs * (1.0 - TIE_RTOL)         # a hair below t_obs, so rounding does not hide ties
    count = 0
    for _ in range(n_perm):
        # Under H0 the two scores of a pair could have come from either model:
        # each difference keeps or flips its sign with probability 1/2.
        s = np.array([-1.0 if rng.uniform() < 0.5 else 1.0 for _ in range(d.size)])
        if abs(float(np.sum(s * d))) >= bar:
            count += 1
    return (1 + count) / (1 + n_perm)      # +1: the observed labelling is one permutation
```

Sampling `n_perm` random sign patterns approximates the exact enumeration of section 3 when $2^n$ is too many to list. The `(1 + count) / (1 + n_perm)` form never returns 0, because the observed data is itself one of the possible labellings; a Monte Carlo p-value of exactly 0 would claim more certainty than `n_perm` samples can give.

### Why this design

- **Paired over unpaired tests.** An unpaired comparison of the two models' mean rewards adds the between-prompt variance to the noise and needs many more prompts to see the same gain.
- **Permutation over a t-test.** Reward differences on a small held-out set are bounded and often skewed; the permutation test needs no normality assumption and is exact in the limit of all sign patterns.
- **Booleans for claims.** A card that says "improves story quality" is easy to write and impossible to check; a boolean next to the evaluation it depends on is checked mechanically.
- **Tradeoffs accepted.** The reference is provenance-only: it proves the shape of the run and the honesty of its claims, not a gain. A real gain needs the full run, more held-out prompts, and several seeds.

### Explain it back

1. Why can three held-out prompts never support a claim at $p < 0.05$ with the paired permutation test?

   <details><summary>Answer</summary>

   With $n$ pairs there are $2^n$ sign patterns, and the all-positive and all-negative patterns always reach $|\sum d_i|$. So the smallest possible two-sided p-value is $2 / 2^n$, which is $0.25$ for $n = 3$. It drops below 0.05 only from $n = 6$ ($2/64 \approx 0.031$).

   </details>

2. Why does the manifest record `heldout.paired_with`?

   <details><summary>Answer</summary>

   A paired difference is only meaningful against the exact base the candidate started from. Recording it makes the pairing checkable: if the base was re-trained or swapped for another snapshot, the differences would mix the adaptation's effect with the base change.

   </details>

3. The reference sets both claims to `false`. Under what evidence would a full run set `measured_heldout_improvement` to `true`?

   <details><summary>Answer</summary>

   When the 95% paired bootstrap CI of $\bar d$ on the held-out set excludes zero on the positive side and the two-sided paired permutation test gives $p < 0.05$, with every seed reported. `general_safety_improvement` stays `false` regardless: a small task-specific set cannot support a general safety claim.

   </details>

## 6. Pitfalls

| Pitfall | How the solution avoids it |
|---|---|
| Evaluate on training pairs | the manifest separates `training_data` (split `train`) from `heldout` (split `heldout`), two different files |
| Report only the best seed | the manifest records the seed, and a full run reports every seed's result |
| Omit base-model provenance | `base_checkpoint`, `heldout.paired_with`, and the tokenizer all point at the C1 model, and the card names the base |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `C1` | Provides the pretrained checkpoint and baseline evaluation. |
| Back | `L12.1` | Converts demonstrations into an SFT checkpoint. |
| Back | `L12.2` | Applies preference-pair optimization. |
| Back | `L12.3` | Applies verifiable reward optimization. |
| Back | `L12.4` | Provides distillation as an alternate recipe. |
| Forward | Pass 10 | The agent pass is complete without this capstone; it is optional. |

## Going further

Larger alignment programs add red-team review, multiple annotator cohorts, data governance, and staged deployment. Those processes are required before treating an offline score as a product benefit.
