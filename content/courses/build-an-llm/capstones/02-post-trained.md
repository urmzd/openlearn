# Capstone 2: post-trained TinyStories model

The [first capstone](01-tinystories.md) proves that the model can be trained, evaluated, released, and served. This optional second capstone asks a narrower and harder question: do demonstrations, and then preference or verifiable-reward data, improve a specific behavior of that model while preserving its language-model quality? Answering it honestly takes more discipline than training does. The held-out comparison has to be fixed before the run, the base model has to be kept as a reproducible reference, and the model card has to keep measured outcomes apart from intended behavior.

The run starts from the first capstone's released checkpoint and uses the methods of the [post-training](../13-post-training/README.md) part of this book: [supervised fine-tuning](../13-post-training/01-supervised-fine-tuning.md) to turn demonstrations into an SFT checkpoint, then either [direct preference optimization](../13-post-training/02-direct-preference-optimization.md) on preference pairs or [GRPO with verifiable rewards](../13-post-training/03-grpo-with-verifiable-rewards.md), with [distillation](../13-post-training/04-distillation.md) as a documented alternate recipe. It produces three things: a run manifest, a model artifact, and an updated model card, following the [checkpoint format](../../../../course/contracts/formats/checkpoint.md), the [eval results schema](../../../../course/contracts/formats/eval-results.schema.json), and the post-training contracts. Reading the section takes about an hour; a full run takes one to two days.

> **Learning objectives**
> - Design a paired before-and-after evaluation on a held-out set fixed before training.
> - Compute an exact paired permutation p-value by hand and explain the minimum sample size it implies.
> - Read a post-training run manifest for provenance and for claims that match the evidence.
> - Explain why claims are recorded as data, and when a full run may set them to `true`.

## Comparing before and after

The core of the capstone is one comparison: the same held-out examples scored by the base model from the first capstone and by the post-trained candidate. For each example $i$, the score difference is $d_i = \text{score}_{\text{after},i} - \text{score}_{\text{before},i}$. You report the mean difference $\bar d$ over the $n$ held-out examples together with a paired confidence interval. Pairing preserves prompt difficulty: a hard prompt is hard for both models, and the difference cancels it.

Two further rules keep the comparison honest. Do not claim a general safety improvement from a small, task-specific dataset; the data can support at most a claim about the behavior it measured. And keep a frozen copy of the base model, recording the dataset revision, tokenizer, template, seed, optimizer, and objective, so the comparison can be rerun against the exact base.

## Working a paired test by hand

Suppose three held-out reward differences are $[0.2, 0.1, 0.3]$. The mean gain is $0.2$, which looks promising. It is not enough to claim a reliable improvement: the capstone uses a larger held-out set and requires a paired interval excluding zero with $p < 0.05$.

The paired permutation test shows exactly why three examples cannot do it. Under the null hypothesis that adaptation changed nothing, each pair's two scores could have come from either model, so each $d_i$ keeps or flips its sign with probability 1/2. With $n = 3$ there are $2^3 = 8$ equally likely sign patterns, and the observed statistic is $|\sum d_i| = 0.6$. The table enumerates them all.

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

The exact two-sided p-value is $2/8 = 0.25$. No three differences can ever do better: the two all-same-sign patterns always reach the observed value, so $p \ge 2/8$ for any data. A sample this small cannot pass the bar, whatever the model does.

## What the run produces

The recipe is fixed: train from the first capstone's checkpoint using SFT, then DPO or GRPO, with distillation as a documented alternate. Evaluate both the reward and any language-model regression, compare with a paired bootstrap or permutation test, and publish an artifact manifest. The model card records intended uses, limitations, data provenance, the evaluation setup, and the licenses of any third-party teacher or model. At the smoke tier the check is only that the paths and the manifest's fields are present and consistent; the full run must show a statistical gain.

A correct solution must handle four cases, each for a reason worth stating.

| Case | Why it matters |
|---|---|
| a run manifest with every provenance field | the base checkpoint, dataset revision, tokenizer, template, seed, optimizer, and objective reproduce the run |
| a held-out split disjoint from training | a reward measured on training pairs measures memorization |
| claims that match the evidence | the card and manifest claim an improvement only when the paired interval excludes zero |
| every seed reported | reporting only the best seed turns noise into a result |

## The reference run, step by step

The reference is a provenance smoke tier: [`artifacts/c2/run.json`](../../../../course/solutions/entry/artifacts/c2/run.json) and [`docs/capstone/c2/MODEL_CARD.md`](../../../../course/solutions/entry/docs/capstone/c2/MODEL_CARD.md), with the paired statistics implemented as `bootstrap_ci` in [`prob/stats.py`](../../../../course/solutions/python/tinyllm/prob/stats.py) and `paired_permutation_test` in [`prob/tests.py`](../../../../course/solutions/python/tinyllm/prob/tests.py). Before looking at it, follow the reasoning that shapes it.

### The reasoning from first principles

1. **The comparison must be paired.** Prompts differ in difficulty far more than the two models differ, so the unit of comparison is one prompt scored by both the base and the candidate; $d_i$ removes the prompt's difficulty.
2. **The held-out set is fixed before training,** or it becomes another thing that was tuned. So the manifest names its source and split, and names the base it is paired with.
3. **Two statistics answer two questions.** The bootstrap CI of $\bar d$ says how large the gain plausibly is; the paired permutation test says whether a gain of this size could come from chance. A claim needs the CI to exclude zero and $p < 0.05$.
4. **Provenance makes the run reproducible.** The base checkpoint, data revision, tokenizer, chat template, seed, optimizer, precision, and objective are each something that, if changed silently, changes the result.
5. **Claims are data, not prose.** Recording `measured_heldout_improvement` and `general_safety_improvement` as booleans lets a check compare them with the evaluation instead of parsing a sentence.

### The run manifest

[`run.json`](../../../../course/solutions/entry/artifacts/c2/run.json) is a provenance smoke artifact: it has every field a full run records, and it truthfully claims nothing.

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

Read it against the principles. `base_checkpoint` and `heldout.paired_with` name the same first-capstone model (`artifacts/c1/model`), so the comparison is paired against the real base, not a different snapshot. `training_data.split` is `"train"` and `heldout.split` is `"heldout"`, two different files, so the held-out reward is never measured on training pairs. The tokenizer is the base model's own `tokenizer.json`, so the candidate encodes text exactly as the base did. The evaluation block fixes the method before any number exists. And because this smoke artifact has no trained weights, both claims are `false`: the manifest says what was measured, not what was hoped.

### The model card

[`MODEL_CARD.md`](../../../../course/solutions/entry/docs/capstone/c2/MODEL_CARD.md) repeats the same discipline in prose. It names the base checkpoint and the objective, points at the manifest for the rest of the provenance, and says in its evaluation section that the paired bootstrap interval and the permutation test are the plan and that this artifact "records the evaluation plan, not measured results". Its limitations section states that the fixture cannot establish a reliable improvement. A full run replaces that section with the measured $\bar d$, its 95% CI, the p-value, the language-model regression (held-out bpb before and after), and every seed's result.

### The statistics

The CI is the course's percentile bootstrap applied to the differences with `stat = mean`, the same `bootstrap_ci` the [first capstone](01-tinystories.md) quotes and annotates in its ablation walkthrough. The significance test is the sign-flip permutation test you worked by hand above:

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

Sampling `n_perm` random sign patterns approximates the exact enumeration when $2^n$ is too many to list. The `(1 + count) / (1 + n_perm)` form never returns 0, because the observed data is itself one of the possible labellings; a Monte Carlo p-value of exactly 0 would claim more certainty than `n_perm` samples can give.

### Why this design

Paired tests beat unpaired ones here. An unpaired comparison of the two models' mean rewards adds the between-prompt variance to the noise and needs many more prompts to see the same gain.

The permutation test beats a t-test for the same data. Reward differences on a small held-out set are bounded and often skewed; the permutation test needs no normality assumption and is exact in the limit of all sign patterns.

Booleans beat prose for claims. A card that says "improves story quality" is easy to write and impossible to check; a boolean next to the evaluation it depends on is checked mechanically.

The tradeoff is accepted openly: the reference is provenance-only. It proves the shape of the run and the honesty of its claims, not a gain. A real gain needs the full run, more held-out prompts, and several seeds.

## What can go wrong

Three mistakes account for most post-training results that do not hold up, and the manifest is shaped to prevent each.

Evaluating on training pairs measures memorization, not improvement. The manifest separates `training_data` (split `train`) from `heldout` (split `heldout`), two different files.

Reporting only the best seed turns noise into a result. The manifest records the seed, and a full run reports every seed's result.

Omitting base-model provenance makes the paired difference meaningless. `base_checkpoint`, `heldout.paired_with`, and the tokenizer all point at the first capstone's model, and the card names the base.

> **Warning:** A gain on a small task-specific held-out set is not a general safety improvement. Keep `general_safety_improvement` set to `false` unless the evidence was designed to test exactly that claim.

## Where this leads

This capstone sits at the end of a chain: the [first capstone](01-tinystories.md) provides the pretrained checkpoint and baseline evaluation, [supervised fine-tuning](../13-post-training/01-supervised-fine-tuning.md) converts demonstrations into an SFT checkpoint, [DPO](../13-post-training/02-direct-preference-optimization.md) applies preference-pair optimization, [GRPO](../13-post-training/03-grpo-with-verifiable-rewards.md) applies verifiable-reward optimization, and [distillation](../13-post-training/04-distillation.md) provides an alternate recipe. Nothing downstream depends on it. It belongs alongside the [agents stage](../paths/course-p10-agents/milestone.md) of the course path, which is complete without it, and its end-to-end result is a held-out reward improvement over the first capstone's base model. For extra depth, try the distillation recipe, or compare all three adaptation methods on the same held-out set.

Larger alignment programs add red-team review, multiple annotator cohorts, data governance, and staged deployment. Those processes are required before treating an offline score as a product benefit.

## Key terms

- **Paired difference**: $d_i$, the change in one held-out example's score from the base to the candidate.
- **Paired permutation test**: a significance test that flips the sign of each $d_i$ at random under the null hypothesis and asks how often the total is as large as observed.
- **Run manifest**: a machine-readable record of a run's provenance, evaluation method, and claims.
- **Provenance**: the base checkpoint, data revision, tokenizer, template, seed, optimizer, precision, and objective that together reproduce a run.
- **Claim boolean**: a manifest field such as `measured_heldout_improvement` that states a conclusion as data a check can verify.

## Summary

- Post-training starts from the first capstone's checkpoint and keeps a frozen, reproducible base reference.
- The held-out suite is fixed before training, paired with the base, and reports uncertainty with a bootstrap CI and a permutation p-value.
- With $n$ pairs the smallest two-sided permutation p-value is $2/2^n$, so three examples can never reach $p < 0.05$; six is the minimum.
- The model card distinguishes measured outcomes from intended behavior, and the manifest records claims as booleans that must match the evidence.
- The reference artifact is provenance-only and truthfully claims nothing; a full run must show the gain across every seed.

## Review questions

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

4. Why does `paired_permutation_test` return `(1 + count) / (1 + n_perm)` rather than `count / n_perm`?

   <details><summary>Answer</summary>

   The observed labelling is itself one of the possible sign patterns, so it is counted once in both numerator and denominator. This also means the function never returns 0: a Monte Carlo p-value of exactly 0 would claim more certainty than `n_perm` random samples can support.

   </details>
