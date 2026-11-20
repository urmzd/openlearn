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

## What makes a method on-policy or off-policy

A **policy** is a rule that maps a state to a probability distribution over actions, written $\pi(a \mid s)$. A Q-table is not a policy. It stores values, and a policy is a rule applied on top of those values, such as "take the largest" (greedy) or "take the largest most of the time and a random action otherwise" (epsilon-greedy). One table can therefore supply two different policies.

Two policies matter in every update:

- The **behavior policy** picks the actions and so produces the experience.
- The **target policy** is the one the update learns about.

A method is **on-policy** when the two are the same and **off-policy** when they differ. The number of tables or networks does not decide it. Q-learning has one table and is off-policy, because it acts with epsilon-greedy and its target assumes the greedy action comes next. The $\max$ in the update is the sign.

A worked target shows the difference. Suppose $R = 0$, $\gamma = 1$, and the next state has values $Q(s', \cdot) = [1, 3]$. The agent explores and takes the first action. SARSA's target is $0 + 1 = 1$, the value of the action it took. Q-learning's target is $0 + 3 = 3$, the value of an action it did not take.

| Method | Behaves with | Learns about | Type |
|---|---|---|---|
| SARSA | Epsilon-greedy | Epsilon-greedy | On-policy |
| Q-learning | Epsilon-greedy | Greedy | Off-policy |
| DQN with a replay buffer | Older versions of itself | Current greedy | Off-policy |
| Policy gradient on a language model | The current model | The current model | On-policy |

For a language model the state is every token so far, the action is the next token, and the model's output distribution is the policy. [On-policy, off-policy, offline, and online learning](../18-post-training/05-on-policy-and-off-policy.md) carries this distinction into post-training.

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

2. Q-learning reads one Q-table both to choose actions and to compute its update. Does that make it on-policy?

   <details><summary>Answer</summary>

   No. The table stores values, and two different rules read it. Actions come from epsilon-greedy, while the update's $\max$ assumes the greedy action. The behavior and target policies differ, so Q-learning is off-policy.

   </details>

3. Does an off-policy method need two tables or two networks?

   <details><summary>Answer</summary>

   No. Off-policy describes a mismatch between the policy that generated the data and the policy being learned. Q-learning is off-policy with one table. Double Q-learning and target networks add a second estimator for stability, which is a separate concern.

   </details>

4. With $R = 0$, $\gamma = 1$, and $Q(s', \cdot) = [1, 3]$, the agent takes the first action in $s'$. What target does each of SARSA and Q-learning use?

   <details><summary>Answer</summary>

   SARSA uses $1$, the value of the action taken. Q-learning uses $3$, the largest value in $s'$.

   </details>

[Chapter contents](README.md) · [Next: Policy gradients and actor-critic](03-policy-gradients-and-actor-critic.md)
