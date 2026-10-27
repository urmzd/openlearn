# PPO with a critic and generalized advantage estimation

PPO alternates between collecting rollouts and taking a limited number of gradient updates on them. The actor chooses actions; a critic predicts future return so the actor can compare an outcome with what was expected. This part derives the quantities needed for a small language-model RLHF loop. Read [on-policy and off-policy learning](05-on-policy-and-off-policy.md) first.

> **Learning objectives**
> - Compute token rewards, temporal-difference residuals, and GAE.
> - Evaluate both signs of PPO's clipped surrogate.
> - Separate critic fitting, actor clipping, KL regularization, and stopping criteria.

## Four roles, not necessarily four full models

The actor is trainable. A frozen old snapshot supplies rollout probabilities. A frozen reference anchors behavior. A reward model or checker scores outcomes, and a trainable value head or critic estimates returns. Implementations can share a backbone, cache scores, or swap weights, so count resident tensors and optimizer states instead of multiplying “model size” by a fixed number. Shared actor/critic parameters also couple their gradients.

For generated token $y_t$ and history $s_t=(x,y_{<t})$, an illustrative RLHF reward is

$$
r_t=-\beta[\log\pi_{\mathrm{old}}(y_t|s_t)-\log\pi_{\mathrm{ref}}(y_t|s_t)]
+\mathbf1_{t=T}R(x,y).
$$

Here the first term is a sampled reference-KL shaping reward, fixed with this rollout batch. Individual log ratios may be negative; their expectation under the old policy is its KL to the reference. Other recipes place an explicit KL penalty in the actor loss. Define which formulation you use so the penalty is not accidentally counted twice. A length cap needs an explicit termination convention; it does not automatically mean the task reached a terminal state.

## Derive the advantage

The critic $V(s_t)$ predicts discounted return. With discount $\gamma$, define

$$
\delta_t=r_t+\gamma b_tV(s_{t+1})-V(s_t),\qquad
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

The **bootstrap mask** $b_t$ is zero at a true terminal and one when a next-state value is legitimate. The **trace mask** $c_t$ is zero where a collected trajectory ends, so advantages cannot propagate into another episode or packed example. At an artificial truncation one may bootstrap ($b_t=1$) while stopping the trace ($c_t=0$). A recipe that treats length-capped text as a terminal uses $b_t=0$ there instead. State and test the choice.

This recurrence is generalized advantage estimation. $\lambda$ trades shorter bootstrapped estimates against longer return estimates. $\lambda=0$ gives a one-step residual; $\lambda=1$ with appropriate terminal handling yields return minus value. LLM recipes often use $\gamma=1$, but discounting is a task decision. See [GAE, section 3](https://arxiv.org/abs/1506.02438).

**Work it by hand.** Rewards are $[0,1]$, old values $[0.2,0.4]$, the final next value is zero, and the last action truly terminates. With $\gamma=1$, $\lambda=0.5$, residuals are $[0.2,0.6]$. Backward recursion gives advantages $[0.5,0.6]$. Fixed critic targets $\hat A+V_{\mathrm{old}}$ are $[0.7,1.0]$. These are detached targets, not numbers recomputed through the changing critic on every minibatch.

## Derive the clipped actor objective

At each valid action token, the ratio is

$$\rho_t=\exp(\log\pi_\theta(y_t|s_t)-\log\pi_{\mathrm{old}}(y_t|s_t)).$$

Maximize the mean of

$$\min\left(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right).$$

With $\epsilon=0.2$, the four cases below expose the sign handling:

| Advantage | Ratio | Raw term | Clipped term | Minimum |
|---|---:|---:|---:|---:|
| $+1$ | 1.5 | 1.5 | 1.2 | 1.2 |
| $+1$ | 0.5 | 0.5 | 0.8 | 0.5 |
| $-1$ | 1.5 | -1.5 | -1.2 | -1.5 |
| $-1$ | 0.5 | -0.5 | -0.8 | -0.8 |

The clip removes the incentive for excessive movement in the favorable direction; it retains penalties for harmful movement. It is not a hard bound on parameters, probability ratios for unseen actions, or full-policy KL. The clipped formulation and combined policy/value/entropy objective come from [PPO, equations 7 and 9](https://arxiv.org/abs/1707.06347).

A simple minimized loss is

$$L=-\mathbb E[\text{surrogate}]+c_v\mathbb E[(V_\phi-\hat R)^2]/2-c_H\mathbb E[H(\pi_\theta)].$$

Use completion masks for actor, critic, and entropy reductions; mask padding and prompt-only positions. Some implementations clip the value update as well; this is an extra design choice. Advantage normalization changes optimization scale. Record its scope and do not divide by the standard deviation of a singleton without an epsilon and a defined convention.

## One complete update round

```text
freeze rollout revision and old token log probabilities
collect prompts → generate completions → score rewards
compute frozen reference scores and old values
build terminal/truncation masks → compute GAE and return targets
for a bounded number of minibatch epochs:
    score recorded action tokens under the current actor
    compute masked clipped actor, critic, and optional entropy loss
    backward → finite-gradient check → clip norm → optimizer step
    measure old/current KL and stop early if the drift budget is exceeded
discard or archive the batch; collect fresh rollouts
```

At initialization, $\rho=1$ for matching behavior and learner computations. Compare the rollout and teacher-forced scoring paths before training; a systematic mismatch can come from filtering, chat templates, dropout, quantization, or [batch-dependent kernels](../20-inference-engines/13-batch-invariance-and-reproducibility.md). Never replace the actual behavior denominator simply to make ratios look like one.

## Controls and failure evidence

| Control | Mechanism | Evidence to inspect |
|---|---|---|
| Clip range, update epochs, learning rate | Limit how strongly one batch changes the actor | Ratio tails, old/current KL, clip fraction, held-out return |
| Reference KL coefficient | Penalize drift from the anchor | Reference KL and behavior changes, separately from old/current KL |
| Value-loss weight and critic learning rate | Fit return predictions | Value error and explained variance on nonconstant targets |
| $\gamma$, $\lambda$, reward scaling | Change credit assignment and gradient scale | Length-conditioned reward, advantage variance, terminal correctness |
| Rollout budget and lag limit | Control sample cost and policy mismatch | Accepted fresh tokens/sec, stale fraction, reward per compute budget |

If reward rises while independent task success falls, investigate reward exploitation. If KL spikes immediately, inspect behavior probabilities and update scale. If the critic predicts a constant, inspect masks, terminal rewards, and target construction before adding capacity. A loss scalar alone does not diagnose these failures.

## Key terms

- **Critic**: a learned estimate of value or return.
- **Advantage**: estimated improvement over the state's baseline value.
- **GAE**: an exponentially weighted combination of temporal-difference residuals.
- **Clip fraction**: a defined count of ratios outside the clipping interval; it is not the fraction of all gradients that are zero.

## Summary

PPO needs valid rollout probabilities, fixed advantages and targets, correct episode boundaries, and controlled reuse. Critic fitting and policy improvement must both be checked against independent outcomes.

## Review questions

1. Why does the negative-advantage, ratio-1.5 row use -1.5?

   <details><summary>Answer</summary>Increasing the probability of a bad action is harmful. The minimum retains that penalty rather than hiding it with the clipped -1.2 term.</details>
2. Why are truncation and terminal masks separate?

   <details><summary>Answer</summary>A rollout can stop collecting while the underlying task remains nonterminal. Bootstrap from that final state if the task definition permits it, but stop the trace at the collection boundary.</details>

## Checks

[Learning checks](../19-training-methods/code/learning_checks.py) verifies the four clipping cases, the GAE example, and terminal versus truncated bootstrapping. It checks arithmetic and boundary conventions, not convergence of a trained policy. Continue with [reward design](07-rewards-and-preference-models.md), then compare the critic-free [GRPO objective](03-grpo-with-verifiable-rewards.md).
