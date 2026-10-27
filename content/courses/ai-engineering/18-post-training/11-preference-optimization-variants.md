# Preference optimization: DPO variants and related objectives

Preference optimization is a family of objectives and data pipelines. Start by asking whether labels are pairs or individual judgments, whether the objective uses a frozen reference, how sequence scores are normalized, and whether new candidates are generated during training. An algorithm name alone does not specify those choices.

> **Learning objectives**
> - Compare DPO, IPO, noisy-label variants, KTO, ORPO, SimPO, and online preference learning.
> - Distinguish objective changes from loss weighting, reference updates, and data refresh.
> - Explain why a reference model or finite margin target does not enforce a global KL bound.

## Core objectives and their evidence

| Method | Required feedback | Reference | Defining change and tradeoff |
|---|---|---|---|
| DPO | Chosen/rejected pair for a prompt | Commonly frozen SFT policy | Logistic loss on reference-adjusted sequence log-probability differences; coverage is limited by the supplied pairs |
| IPO | Preference pairs | Yes | Squared loss toward a finite relative-score target; score normalization must match the intended recipe |
| Label-smoothed/conservative DPO | Preference pairs with uncertainty assumptions | Yes | Mix both label directions; reduces the drive toward an infinite fitted margin |
| Robust DPO | Noisy pairs under a specified noise model | Yes | Correct the estimated loss for label flips; incorrect noise assumptions can undermine the correction |
| KTO | Individually desirable/undesirable outputs | Yes in the original formulation | Prospect-theoretic utility with a reference baseline; does not require paired labels for the same prompt |
| ORPO | Chosen/rejected pairs | No separate reference | Combine chosen-answer likelihood with an odds-ratio preference penalty; the supervised term is part of the objective |
| SimPO | Chosen/rejected pairs | No separate reference | Use length-averaged log probability as implicit reward and introduce a target reward margin |
| Online/iterative DPO | Newly generated, judged pairs | Recipe-dependent, commonly fixed or explicitly refreshed | Refresh candidate distribution while learning; generation and judge costs enter the budget |

The primary sources are [DPO](https://arxiv.org/abs/2305.18290), [IPO](https://arxiv.org/abs/2310.12036), [Robust DPO](https://arxiv.org/abs/2403.00409), [KTO](https://arxiv.org/abs/2402.01306), [ORPO](https://arxiv.org/abs/2403.07691), [SimPO](https://arxiv.org/abs/2405.14734), and [online AI-feedback alignment](https://arxiv.org/abs/2402.04792). Each paper evaluates particular models, feedback, and budgets; their reported rankings are not universal method rankings.

## Change the loss, not just its label

For DPO, write the reference-adjusted score difference as $z$ and the scaled margin as $m=\beta z$. The ordinary loss is $\operatorname{softplus}(-m)$. Label smoothing with $0<s<0.5$ gives the illustrative binary cross-entropy mixture

$$
L_s=(1-s)\operatorname{softplus}(-m)+s\operatorname{softplus}(m).
$$

For an isolated freely fitted margin its optimum is $m^*=\log((1-s)/s)$. At $s=0.1$, that is $\log9\approx2.1972$. Shared model parameters and conflicting examples mean an actual policy need not attain that target for every pair. A noise-corrected robust loss is a different estimator, not merely another name for this positive mixture. Check the installed trainer's exact formula and assumptions.

The [DPO/IPO implementation](02-direct-preference-optimization.md) computes IPO on supplied completion scores. It illustrates $(z-1/(2\beta))^2$, but supplies sequence log-probability sums for its small exercise. Original-style IPO implementations can use completion-averaged log probabilities; the [TRL 0.23.1 reference](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) calls out this distinction. Changing the reduction changes $z$ and the effective regularization scale. Do not transfer a tuned `beta` between incompatible conventions and call it the same experiment.

## Work a length-normalization example

Suppose two candidate sequences have token-log-probability sums -4 and -6, with lengths 2 and 6. The first has a larger sequence probability because $-4>-6$, while average log probabilities are -2 and -1, ranking the second higher. Average scores therefore change the preference objective rather than only its numerical implementation.

SimPO's reference-free implicit score uses an average log probability, with a separate target margin inside its pairwise objective. ORPO uses an odds ratio combined with a chosen-response likelihood term. Neither is reproduced merely by deleting the reference arrays from the DPO helper. Evaluate output length and content quality independently; length normalization is not a guarantee that length bias disappears.

## A reference score is not a hard trust region

Consider three possible outputs. The reference assigns probabilities $[0.1,0.1,0.8]$ and a candidate policy assigns $[0.49,0.49,0.02]$. For a pair comprising the first two outputs, both policies assign equal probability within the pair, so the reference-adjusted margin is zero and DPO loss is $\log2$. Nevertheless,

$$
D_{KL}(\pi\|\pi_{\mathrm{ref}})=0.98\log4.9+0.02\log0.025\approx1.484.
$$

The observed pair loss alone cannot detect how probability moved between that pair and the third output. IPO's finite target constrains the fitted pair score, not every unobserved output or the full-policy KL. Monitor reference drift and held-out behavior rather than treating the presence of a reference or `beta` as a hard global bound.

## Other variants fit a few recurring mechanisms

The versioned [TRL DPO trainer](https://huggingface.co/docs/trl/v0.23.1/dpo_trainer) exposes additional research objectives. Treat this as implementation vocabulary to inspect, not a single portable list of defaults.

| Family | Examples | What changes and what to validate |
|---|---|---|
| Margin/classification shape | SLiC hinge, IPO, DiscoPOP | Gradient shape and saturation; compare stability and held-out quality |
| Divergence/distribution matching | EXO, AOT | Direction or distribution-level objective; check grouping, pairing, and sampling assumptions |
| Absolute or anchored scores | NCA, BCO, SPPO, APO | How chosen and rejected likelihoods move individually, beyond their difference |
| Data weighting | WPO, length-decomposed DPO | Which samples/tokens receive weight; inspect concentration and length-conditioned outcomes |
| Reference refresh | TR-DPO | When and how the reference changes; invalidate cached reference scores when necessary |
| Combined objectives | RPO, mixed preference optimization | Relative weights of preference, chosen-answer likelihood, and other losses; report the full combination |

A trainer option can activate a substantive objective change. Log the loss type, formula/reduction, weights, reference revision, label smoothing/noise model, and dataset construction. Multi-loss configurations should be compared with their individual terms before attributing a gain to a method name.

## Online preference learning and reward-based alternatives

Online DPO generates new candidates, labels them, and fits preference pairs. That online collection step does not turn its classification loss into PPO. Pair freshness, judge quality, prompt coverage, and reference-refresh policy are separate controls. Record both generation and labeling cost and evaluate judge independence.

PPO and GRPO instead optimize from sampled outcomes using policy-gradient estimators. **RLOO**, developed in [Back to Basics](https://arxiv.org/abs/2402.14740), uses a REINFORCE-style sequence update with a leave-one-out baseline from other completions of the same prompt. For $G>1$, that baseline is $b_i=(\sum_{j\ne i}R_j)/(G-1)$ under the independent-sample construction. It removes the particular self-inclusion of an inclusive group-mean baseline; it does not remove rollout cost, stale-policy issues, or reward errors. Compare it with [PPO/GAE](06-ppo-with-gae.md) and [GRPO variants](08-rollout-systems-and-grpo-variants.md) using the same reward and total compute budget.

## A common comparison protocol

Hold base checkpoint, data split, tokenizer/template, evaluation, output budget, and hardware budget fixed. State whether candidates and teacher/reference scores were precomputed and count that cost. Log chosen and rejected log probabilities separately, their length-normalized counterparts, reference KL where estimable, task success, judge agreement, generated length, and retained-capability regressions.

For unpaired labels, check class balance and weighting rather than inventing arbitrary pairs across unrelated prompts. For noisy preferences, test disagreement and the assumed label-flip rate. For reference-free methods, still evaluate drift against the baseline even though the training loss does not load a reference. Re-evaluate after adapter merge, export, and quantization using the [local serving lab](../20-inference-engines/14-local-serving-and-validation.md).

## Key terms

- **Reference-free objective**: an objective that does not require a separate reference policy in its training formula.
- **Label smoothing**: replacing a hard classification target with a specified mixture of label directions.
- **Online preference collection**: collecting and labeling new candidate outputs during learning.

## Summary

Compare feedback, objective, normalization, reference handling, and sampling separately. Method choice requires measured task quality and cost rather than a universal ranking.

## Review questions

1. Does a DPO loss of $\log2$ prove the policy still equals the reference?

   <details><summary>Answer</summary>No. It establishes zero relative margin for the observed pair. The three-output counterexample has that loss while its full distributions differ substantially.</details>
2. Can KTO consume an isolated desirable output without a rejected answer to the same prompt?

   <details><summary>Answer</summary>Yes. That unpaired desirable/undesirable feedback is a defining difference from pairwise DPO.</details>
3. Is SimPO the same as DPO with cached reference probabilities omitted?

   <details><summary>Answer</summary>No. Its length-averaged implicit reward and target margin change the objective.</details>

## Checks

[Learning checks](../19-training-methods/code/learning_checks.py) verifies the smoothing optimum, the sequence/average ranking reversal, the reference-margin counterexample, and the leave-one-out example. These verify the worked mathematics, not a benchmark ranking or a reproduced training result for every listed objective.
