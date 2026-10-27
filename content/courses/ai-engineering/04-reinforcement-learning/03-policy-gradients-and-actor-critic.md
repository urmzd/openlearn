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

[Chapter contents](README.md) · [Next: World models, multiple agents, and RLHF](04-models-multiple-agents-and-rlhf.md)
