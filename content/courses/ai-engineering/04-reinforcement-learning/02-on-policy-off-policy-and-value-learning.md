# On-policy, off-policy, and value learning

This part of [Reinforcement Learning](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Compare Monte Carlo, TD, SARSA, and Q-learning data assumptions.

## Monte Carlo methods

*Reading: Sutton and Barto Chapter 5.*

Monte Carlo methods need no model of the environment; they learn from complete episodes. **First-visit MC** estimates $V(s)$ as the average of the returns that follow the first visit to $s$ in each episode, and **every-visit MC** averages the returns after every visit. **Off-policy MC** learns about one policy while following another, correcting for the mismatch with importance sampling. The cost of needing no model is having to wait until an episode ends before learning anything.

## Temporal-difference learning

*Reading: Sutton and Barto Chapter 6.*

Temporal-difference learning combines the benefits of the previous two approaches: like Monte Carlo it needs no model, and like dynamic programming it bootstraps from its current estimates, so it need not wait for an episode to end. **TD(0)** updates after every step toward a one-step target:

$$V(s) \leftarrow V(s) + \alpha\big[R + \gamma V(s') - V(s)\big].$$

The same idea applied to action values gives two control algorithms. **SARSA** is on-policy, using the action $a'$ the policy actually takes next:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\big[R + \gamma Q(s', a') - Q(s, a)\big].$$

**Q-learning** is off-policy, using the best next action regardless of what the agent does:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\big[R + \gamma \max_{a'} Q(s', a') - Q(s, a)\big].$$

Taking a max over noisy estimates is biased upward, so **double Q-learning** keeps two Q-functions, using one to choose the action and the other to evaluate it, which reduces this maximization bias.

## Function approximation and DQN

*Reading: Sutton and Barto Chapters 9-10; Mnih et al., 2015.*

Tables of values do not scale to large or continuous state spaces, so the next step is to approximate $Q$ with a parameterized function. **Linear function approximation**, $Q(s, a) = w^\top \phi(s, a)$, converges with TD under certain conditions. A neural network $Q(s, a; \theta)$ is far more expressive but can diverge; the combination of function approximation, bootstrapping, and off-policy learning is known as the **deadly triad**.

**DQN** (deep Q-network) made neural Q-learning work with two stabilizers. Experience replay stores transitions and trains on random samples of them, breaking the correlation between consecutive updates. A target network, a periodically updated copy of the Q-network, computes the bootstrap targets so they do not shift with every gradient step. **Double DQN** carries the double Q-learning idea over: the online network selects the next action and the target network evaluates it, reducing overestimation.

## Summary

Compare Monte Carlo, TD, SARSA, and Q-learning data assumptions. State the assumptions before applying the result to a different workload.

## Review questions

1. Why does replay require attention to the behavior policy?

   <details><summary>Answer</summary>

   The data distribution may differ from the target policy, changing which update estimators are valid.

   </details>

[Chapter contents](README.md) · [Next: Policy gradients and actor-critic](03-policy-gradients-and-actor-critic.md)
