# Bandits, MDPs, and planning

This part of [Reinforcement Learning](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Distinguish immediate reward optimization from sequential credit assignment.

## Multi-armed bandits

*Reading: Sutton and Barto Chapter 2.*

A bandit is RL with the state transitions removed: you repeatedly choose one of several actions (arms), each with an unknown reward distribution, and try to maximize total reward. That makes it the cleanest place to study exploration. Should you exploit the arm with the best estimated value, or explore another arm that might be better?

The simplest strategy is **epsilon-greedy**: with probability $\varepsilon$ pick an arm uniformly at random, and otherwise pick the best-known one. **Upper Confidence Bound** (UCB) replaces randomness with optimism in the face of uncertainty, choosing the action that maximizes

$$Q(a) + c\sqrt{\frac{\ln t}{N(a)}},$$

where $N(a)$ is how often arm $a$ has been tried and $t$ is the total number of steps; rarely tried arms get a large bonus. **Thompson sampling** keeps a posterior distribution over each arm's value, samples from each, and plays the arm whose sample is highest, which balances exploration naturally because uncertain arms sometimes produce high samples.

## Markov decision processes

*Reading: Sutton and Barto Chapter 3.*

Adding state turns a bandit into a **Markov decision process** (MDP), the tuple $(S, A, P, R, \gamma)$: states, actions, transition dynamics $P(s' \mid s, a)$, a reward function, and a discount factor $\gamma$. The agent's behavior is a **policy** $\pi(a \mid s)$, a mapping from states to action probabilities.

To compare policies, define the **value function**, the expected discounted return from state $s$ when following $\pi$:

$$V^\pi(s) = E\Big[\sum_t \gamma^t R_t \,\Big|\, s_0 = s, \pi\Big],$$

and the **Q-function**, which also fixes the first action:

$$Q^\pi(s, a) = E\Big[\sum_t \gamma^t R_t \,\Big|\, s_0 = s, a_0 = a, \pi\Big].$$

The central result is the **Bellman equation**, which says that the value of a state is the immediate reward plus the discounted value of the next state, averaged over the policy and the dynamics:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\big[R(s, a, s') + \gamma V^\pi(s')\big].$$

Every method in the rest of the chapter is, in one way or another, a way of solving or approximating this equation.

## Dynamic programming

*Reading: Sutton and Barto Chapter 4.*

If you know the MDP completely, you can solve the Bellman equation by iteration. **Policy evaluation** applies the Bellman equation repeatedly as an update until $V^\pi$ converges. **Policy improvement** makes the policy greedy with respect to the current value function. **Policy iteration** alternates the two until the policy stops changing, and **value iteration** fuses them into a single update that converges to the optimal value function $V^*$.

The limitation is the premise: dynamic programming requires complete knowledge of the transition dynamics, and real-world problems rarely provide it. The next two methods learn from experience instead.

## Summary

Distinguish immediate reward optimization from sequential credit assignment. State the assumptions before applying the result to a different workload.

## Review questions

1. Why can the greedy action have a lower return?

   <details><summary>Answer</summary>

   It can sacrifice future rewards or prevent useful exploration.

   </details>

[Chapter contents](README.md) · [Next: On-policy, off-policy, and value learning](02-on-policy-off-policy-and-value-learning.md)
