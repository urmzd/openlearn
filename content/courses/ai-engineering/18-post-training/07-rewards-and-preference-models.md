# Rewards, preference models, and independent evaluation

A policy can optimize exactly the reward you wrote while becoming less useful. Reward design therefore starts with a task outcome and a test that can expose shortcuts. PPO and GRPO are optimizers for the supplied signal; neither repairs a flawed signal.

> **Learning objectives**
> - Derive a pairwise reward-model loss.
> - Distinguish outcome rewards, process rewards, and format checks.
> - Design a reward audit using held-out cases and adversarial outputs.

## Learn a preference reward

Given a prompt $x$, a preferred answer $y_w$, and a rejected answer $y_l$, a scalar reward model $r_\phi$ can use the Bradley–Terry likelihood

$$P(y_w\succ y_l|x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)).$$

Minimize $-\log\sigma(\Delta r)$ using a stable softplus, $\log(1+\exp(-\Delta r))$. At equal rewards the loss is $\log2\approx0.6931$; at a margin of 2 it is $0.1269$. The likelihood depends on differences, so adding a common constant to rewards does not change it. Reward scale and calibration still matter to a downstream RL update.

This is the basic comparison-model setup used in [InstructGPT, section 3.2](https://arxiv.org/abs/2203.02155). Human feedback can express preferences without defining a perfect utility function. Report agreement, label instructions, and prompt coverage. A learned judge's confidence is not an independent guarantee of correctness.

Split comparisons by prompt and, when appropriate, source, author, customer, or time. Randomly splitting near-duplicate answers to the same prompt can leak the task across train and test. Include counterexamples where verbosity, a confident tone, or a specific formatting cue conflicts with actual answer quality. Evaluate the reward model on outputs from the evolving actor, since the actor changes the distribution on which the judge operates.

## Verifiable does not mean complete

| Signal | What it can establish | Failure to test |
|---|---|---|
| Exact numerical answer | Agreement with an expected value and parser | Ambiguous units, parser exploits, answer leakage |
| Unit-test pass | Behavior on those tests | Hardcoded visible cases, missing edge cases, nondeterministic tests |
| Schema validity | Membership in the schema | Valid JSON containing wrong facts |
| Tool or simulator outcome | Outcome under the environment's rules | Simulator mismatch, failed tool calls, hidden state |
| Process reward | A score for intermediate reasoning steps | Locally plausible steps that lead to a false result |
| Preference reward | Agreement with the labeling rubric on covered examples | Style shortcuts, label bias, distribution shift |

Run generated code or tool actions in the exercise's defined environment and capture exit status, timeouts, resource limits, and output. A timeout is a distinct outcome, not automatically evidence that the reasoning was wrong. Keep checker errors separate from task failures. A checker can itself have bugs or exploitable omissions.

## Combine components explicitly

For an illustrative math task, write $R=R_{\mathrm{answer}}+0.1R_{\mathrm{format}}$. If a policy improves formatting while answer correctness remains flat, aggregate reward can rise without solving more tasks. Log both components, output length, truncation rate, and per-difficulty results. Hard requirements can be gates instead of small additive bonuses when the task contract demands them.

Length penalties change the optimization problem. So does treating a capped completion as a failure or excluding it from training. Preserve the count and denominator of every excluded sample. Otherwise the training dashboard can improve simply by dropping hard prompts. For repeated generations, report pass@1 separately from success after many attempts and include generation cost.

## Audit before scaling rollouts

Freeze a small prompt set with expected outcomes and adversarial answers. Test valid alternatives, wrong answers in perfect format, correct answers in imperfect format, empty outputs, repeated answers, long outputs, and checker exceptions. Then evaluate a checkpoint on an independent task suite that does not reuse the training reward as its only judge. Keep a fixed audit set for regression detection and collect fresh audits to detect exploitation outside it.

Preference optimization and rejection sampling offer other uses for the same evidence. [DPO](02-direct-preference-optimization.md) learns directly from labeled pairs. Rejection sampling generates candidates, keeps those passing a criterion, and trains on the selected answers with SFT. The latter avoids a policy-gradient loop but spends sampling compute and can narrow the data distribution. Compare quality per total training-and-generation cost.

## Key terms

- **Reward exploitation**: improving a proxy score by behavior that misses the intended outcome.
- **Outcome reward**: feedback on a final answer or result.
- **Process reward**: feedback on intermediate steps or actions.

## Summary

Specify the task, measure reward components separately, and evaluate policy behavior outside the training reward. Scaling a biased checker scales its bias.

## Review questions

1. A model passes every visible unit test. What remains unproven?

   <details><summary>Answer</summary>Correctness on unseen cases, absence of hardcoding, safe tool behavior, and transfer to the real environment.</details>
2. Can a higher preference-model accuracy justify changing the RL reward scale without retuning?

   <details><summary>Answer</summary>No. Ranking accuracy and numerical scale are different. A scale change affects advantage variance and the balance against reference regularization.</details>
