# Policy gradients and actor-critic

This part of [Reinforcement Learning](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Explain policy-gradient baselines, PPO clipping, and continuous control.

## Policy gradient methods

*Reading: Sutton and Barto Chapter 13.*

Value-based methods learn values and derive a policy from them. Policy gradient methods instead parameterize the policy directly and climb the gradient of expected return $J(\theta)$. The **policy gradient theorem** gives that gradient:

$$\nabla J(\theta) = E\Big[\sum_t \nabla_\theta \log \pi(a_t \mid s_t; \theta)\, G_t\Big],$$

where $G_t$ is the return from step $t$. This is why policy gradients work at all: the theorem provides an exact expression for the gradient even though the environment's dynamics are unknown, because the dynamics do not depend on $\theta$ and drop out of the derivative.

**REINFORCE** estimates this expectation with Monte Carlo returns, which is unbiased but has high variance. Subtracting a **baseline** $b(s)$ from $G_t$ reduces the variance without introducing bias, and the natural choice is the value function $V(s)$. **Actor-critic** methods make that systematic: an actor (the policy) is trained with a critic (a learned value function) that provides the low-variance baseline.

## How backpropagation uses the reward

The reward is never differentiated. A checker, a test suite, or a judge model produces it, and none of these needs to be differentiable. The reward enters the loss as a constant that weights the gradient of the policy's own log probabilities. For a language model, one update runs like this:

1. The sampler generates a **rollout**: the prompt, every sampled token, any tool calls, and the final answer. It records the log probability of each sampled token.
2. A reward function scores the rollout.
3. The trainer replays the same tokens in one teacher-forced forward pass and computes the log probability of each sampled token under the current weights.
4. The loss weights those log probabilities by the advantage, which is the reward minus a baseline:

$$L = -\,(R - b)\sum_t \log \pi_\theta(a_t \mid s_t).$$

5. Backpropagation runs through that forward pass exactly as in supervised fine-tuning. Rollouts that scored above the baseline become more likely, and the others become less likely.

A single reward carries little information. A score of $0.7$ becomes a signal only beside other attempts at the same prompt that scored $0.4$, which is why rollouts are sampled in groups and compared with a baseline.

Compared with supervised fine-tuning, only the loss differs: reward-weighted log probabilities replace cross-entropy against a label. The same holds when the trainable parameters are [LoRA adapters](../19-training-methods/05-parameter-efficient-fine-tuning.md).

## Trust region methods: TRPO and PPO

*Reading: Spinning Up; Schulman et al., 2015 and 2017.*

Policy gradient steps have a dangerous failure mode: one overly large update can wreck the policy, and since the policy generates its own data, it may never recover. **TRPO** prevents such catastrophic updates by constraining each update to stay within a trust region measured by KL divergence between the old and new policies. **PPO** achieves much the same effect more simply. Its clipped objective takes the probability ratio $r(\theta) = \pi_{\text{new}}/\pi_{\text{old}}$ and clips it to $[1 - \epsilon, 1 + \epsilon]$, removing the incentive for large updates without an explicit KL constraint.

PPO is a widely studied on-policy method and was used in the documented InstructGPT RLHF recipe. Other tasks and model-training systems use different objectives. The worked [PPO and GAE part](../18-post-training/06-ppo-with-gae.md) derives a language-model update and its failure checks.

## Continuous control: DDPG, TD3, and SAC

*Reading: Spinning Up.*

When actions are continuous, such as joint torques for a robot, taking a max over actions is no longer simple, and a family of off-policy actor-critic methods takes over. **DDPG** (Deep Deterministic Policy Gradient) learns a deterministic actor alongside a Q-function critic. **TD3** (Twin Delayed DDPG) fixes DDPG's overestimation with three changes: two critics whose minimum is used as the target, delayed policy updates, and target policy smoothing. **SAC** (Soft Actor-Critic) maximizes reward plus policy entropy, encouraging exploration in continuous control; compare its performance under the task and compute budget.

## Summary

Explain policy-gradient baselines, PPO clipping, and continuous control. State the assumptions before applying the result to a different workload.

## Review questions

1. Does clipping guarantee a bound on the full-policy KL divergence?

   <details><summary>Answer</summary>

   No. It changes the sampled surrogate incentive; monitor KL and stop or reduce updates when drift is excessive.

   </details>

2. The reward comes from a test suite that is not differentiable. How does backpropagation still produce an update?

   <details><summary>Answer</summary>

   The gradient is taken through the model's log probabilities of the tokens it sampled, not through the reward. The reward, minus a baseline, is a constant weight on that gradient.

   </details>

3. Why does a policy-gradient step sample several rollouts for each prompt?

   <details><summary>Answer</summary>

   A reward is informative only relative to a baseline. Several attempts at one prompt give a comparison, so the update can raise the probability of the better attempts and lower the rest.

   </details>

[Chapter contents](README.md) · [Next: World models, multiple agents, and RLHF](04-models-multiple-agents-and-rlhf.md)
