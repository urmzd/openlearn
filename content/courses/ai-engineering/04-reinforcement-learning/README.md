# Reinforcement Learning

Reinforcement learning is learning from interaction. An agent observes the state of an environment, takes an action, receives a reward, and finds itself in a new state; over many such steps it must learn a policy that maximizes its cumulative reward. Two things make this harder than supervised learning. Nobody tells the agent the right action, only how well things went, often many steps later. And the agent's own choices determine what data it sees, so it must explore to discover which actions are good while also exploiting what it already knows. That exploration-exploitation tradeoff has no counterpart in supervised learning.

RL matters in *AI Engineering* for more than games and robots. Policy gradient methods, PPO above all, are the backbone of reinforcement learning from human feedback (RLHF) and therefore of modern AI alignment, which is where this chapter ends.

The chapter follows Sutton and Barto's *Reinforcement Learning: An Introduction* in its natural order. It starts with bandits, the simplest setting for studying exploration, then formalizes sequential decisions as Markov decision processes and the Bellman equations. Three families of solution methods follow: dynamic programming, which needs a full model; Monte Carlo methods, which learn from complete episodes; and temporal-difference learning, which combines the best of both. Function approximation and DQN scale those ideas to large state spaces. The chapter then switches from learning values to learning policies directly, with policy gradients, actor-critic methods, TRPO and PPO, and continuous-control algorithms. It closes with model-based RL, which is more sample-efficient but requires learning an accurate world model, multi-agent RL, and RLHF.

> **Learning objectives**
> - Formalize a sequential decision problem as an MDP and write the Bellman equation for its value function.
> - Compare epsilon-greedy, UCB, and Thompson sampling as solutions to the exploration-exploitation tradeoff.
> - Derive and contrast the DP, Monte Carlo, and TD updates, including SARSA and Q-learning.
> - Explain why deep Q-learning can diverge and how experience replay and target networks stabilize it.
> - State the policy gradient theorem and explain how baselines, actor-critic methods, and PPO's clipping make policy optimization practical, including in RLHF.

## Before you begin

This chapter assumes [Deep Learning](../02-deep-learning/README.md), the previous chapter, for neural function approximation, and [Probability and Statistics](../../mathematics/13-probability-statistics/README.md) from *Mathematics* for expectations and Markov chains. Familiarity with dynamic programming from the [Algorithms](../../programming-and-algorithms/09-dynamic-programming/README.md) book helps, since policy and value iteration are dynamic programming.

For theory, read [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html) (2nd edition, free online) in this order: Chapters 1-6, then 9-10, then 13. For implementation, use [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/), starting with vanilla policy gradient and then PPO; its key papers list is the place to go for research depth. David Silver's UCL lectures, free on YouTube, are an excellent companion. Before reaching for any library, implement Q-learning on a grid world from scratch.

## In this chapter

- [Multi-armed bandits](#multi-armed-bandits): exploration and exploitation in the simplest setting.
- [Markov decision processes](#markov-decision-processes): states, actions, policies, value functions, and the Bellman equation.
- [Dynamic programming](#dynamic-programming), [Monte Carlo methods](#monte-carlo-methods), and [Temporal-difference learning](#temporal-difference-learning): three ways to compute values, from full models to bootstrapping.
- [Function approximation and DQN](#function-approximation-and-dqn): scaling value methods with neural networks, and the deadly triad.
- [Policy gradient methods](#policy-gradient-methods) and [Trust region methods: TRPO and PPO](#trust-region-methods-trpo-and-ppo): optimizing the policy directly, safely.
- [Continuous control](#continuous-control-ddpg-td3-and-sac): DDPG, TD3, and SAC.
- [Model-based RL](#model-based-rl) and [Multi-agent RL](#multi-agent-rl): planning with learned models and learning among other agents.
- [RLHF and alignment](#rlhf-and-alignment): RL applied to language models.

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

## Policy gradient methods

*Reading: Sutton and Barto Chapter 13.*

Value-based methods learn values and derive a policy from them. Policy gradient methods instead parameterize the policy directly and climb the gradient of expected return $J(\theta)$. The **policy gradient theorem** gives that gradient:

$$\nabla J(\theta) = E\Big[\sum_t \nabla_\theta \log \pi(a_t \mid s_t; \theta)\, G_t\Big],$$

where $G_t$ is the return from step $t$. This is why policy gradients work at all: the theorem provides an exact expression for the gradient even though the environment's dynamics are unknown, because the dynamics do not depend on $\theta$ and drop out of the derivative.

**REINFORCE** estimates this expectation with Monte Carlo returns, which is unbiased but has high variance. Subtracting a **baseline** $b(s)$ from $G_t$ reduces the variance without introducing bias, and the natural choice is the value function $V(s)$. **Actor-critic** methods make that systematic: an actor (the policy) is trained with a critic (a learned value function) that provides the low-variance baseline.

## Trust region methods: TRPO and PPO

*Reading: Spinning Up; Schulman et al., 2015 and 2017.*

Policy gradient steps have a dangerous failure mode: one overly large update can wreck the policy, and since the policy generates its own data, it may never recover. **TRPO** prevents such catastrophic updates by constraining each update to stay within a trust region measured by KL divergence between the old and new policies. **PPO** achieves much the same effect more simply. Its clipped objective takes the probability ratio $r(\theta) = \pi_{\text{new}}/\pi_{\text{old}}$ and clips it to $[1 - \epsilon, 1 + \epsilon]$, removing the incentive for large updates without an explicit KL constraint.

PPO is the default for most practical deep RL, including RLHF for language models; it is the algorithm behind ChatGPT's RLHF training. Understanding it is essential for AI alignment work.

## Continuous control: DDPG, TD3, and SAC

*Reading: Spinning Up.*

When actions are continuous, such as joint torques for a robot, taking a max over actions is no longer simple, and a family of off-policy actor-critic methods takes over. **DDPG** (Deep Deterministic Policy Gradient) learns a deterministic actor alongside a Q-function critic. **TD3** (Twin Delayed DDPG) fixes DDPG's overestimation with three changes: two critics whose minimum is used as the target, delayed policy updates, and target policy smoothing. **SAC** (Soft Actor-Critic) maximizes reward plus policy entropy, which encourages exploration and makes it state-of-the-art for continuous control.

## Model-based RL

*Reading: Sutton and Barto Chapter 8, plus research papers.*

All the methods so far are model-free. Model-based RL learns a model of the environment and uses it to plan. **Dyna** learns a model and uses it to generate simulated experience for planning alongside real experience. **World models** learn a latent dynamics model and plan in the latent space (Ha and Schmidhuber, 2018). **MuZero** learns a model that predicts rewards and values without reconstructing observations at all, and it reached superhuman play in Go, chess, and Atari.

The tradeoff is consistent across these methods: model-based RL is more sample-efficient than model-free RL, but errors in the learned model compound over long plans, a problem called model bias.

## Multi-agent RL

*Research frontier.*

With several learning agents, each one's environment includes the others. In **cooperative** settings agents share a common reward, and the challenges are credit assignment and communication. In **competitive**, zero-sum settings, self-play has produced landmark systems such as AlphaGo and OpenAI Five. **Mixed**, general-sum settings use the Nash equilibrium as their solution concept. A striking research thread is **emergent communication**, in which agents develop their own communication protocols through interaction.

## RLHF and alignment

*Research frontier; connects to [Deep Learning](../02-deep-learning/README.md).*

The chapter's last application brings RL to language models. **Reward modeling** trains a model from human preference comparisons ("A is better than B"). The **RLHF pipeline** then runs supervised fine-tuning, trains the reward model, and optimizes the policy against it with PPO. **DPO** (Direct Preference Optimization) reparameterizes the reward in terms of the policy itself, so no separate reward model is needed. **Constitutional AI** has the model critique its own outputs using written principles and then fine-tunes on the revised outputs. And the open problem of **scaling supervision** asks how to provide a training signal once models surpass human ability.

The key papers are InstructGPT (Ouyang et al., 2022), Constitutional AI (Bai et al., 2022), and DPO (Rafailov et al., 2023).

## Comparing the algorithms

The algorithms in this chapter differ in what they learn, whether they learn on-policy or off-policy, and what problems they suit. The catalog below summarizes them:

| Algorithm | Type | On/Off-Policy | Best For |
|-----------|------|---------------|----------|
| Q-learning | Value-based | Off-policy | Discrete actions, tabular |
| DQN | Value-based | Off-policy | Discrete actions, function approx |
| REINFORCE | Policy gradient | On-policy | Simple, educational |
| PPO | Policy gradient | On-policy | General purpose, RLHF |
| DDPG/TD3 | Actor-critic | Off-policy | Continuous control |
| SAC | Actor-critic | Off-policy | Continuous control (with exploration) |
| MuZero | Model-based | -- | Games, planning |

## Where this leads

RL connects to much of the rest of the library. MDPs are stochastic processes, and the Bellman equations rest on the Markov chains of [Probability and Statistics](../../mathematics/13-probability-statistics/README.md). Policy and value iteration *are* dynamic programming, as covered in the [Algorithms](../../programming-and-algorithms/09-dynamic-programming/README.md) book, and the minimax reasoning of multi-agent RL connects to game theory in [Programming and Algorithms](../../programming-and-algorithms/README.md). Neural networks from [Deep Learning](../02-deep-learning/README.md) supply the function approximation for Q-functions and policies, and PPO and reward modeling return to that chapter as the machinery of language-model alignment, developed further in [Training and Post-Training](../19-training-methods/README.md). RL also combines with evolutionary methods; linear genetic programming paired with Q-learning is one example of that hybrid line of research.

RL is a recurring topic in interviews, especially at AI labs and trading firms, as the table below shows:

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Anthropic | RLHF, DPO, constitutional AI, alignment research | PhD-level |
| OpenAI | PPO for ChatGPT, RLHF pipeline, scaling RL | PhD-level |
| DeepMind | AlphaGo/Zero/Fold, MuZero, multi-agent, robotics | PhD-level |
| Robotics companies | Control policies, sim-to-real transfer | Expert |
| Jane Street | Optimal execution, portfolio as MDP | Expert |
| Two Sigma | Trading strategies as sequential decisions | Expert |

### Further reading

- Sutton and Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html), 2nd edition (free online).
- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), including its key papers list.
- David Silver's RL lectures (UCL), free on YouTube.

## Key terms

- **Policy**: $\pi(a \mid s)$, the agent's probability of taking each action in each state.
- **Value function**: the expected discounted return from a state (or state-action pair) under a policy.
- **Bellman equation**: the recursive relation expressing a state's value as immediate reward plus discounted next-state value.
- **Bootstrapping**: updating an estimate toward a target that itself uses current estimates.
- **On-policy / off-policy**: learning about the policy being followed versus learning about a different policy.
- **Deadly triad**: function approximation, bootstrapping, and off-policy learning together, which can make value learning diverge.
- **Policy gradient theorem**: an expression for the gradient of expected return that does not require the environment's dynamics.
- **PPO**: a policy gradient method that clips the new-to-old probability ratio to keep updates small.

## Summary

- RL learns a policy by trial and error in an environment, and the exploration-exploitation tradeoff is unique to it.
- MDPs and the Bellman equation formalize the problem; DP needs a model, Monte Carlo needs complete episodes, and TD bootstraps without either.
- Deep Q-learning needs experience replay and target networks to tame the deadly triad.
- Policy gradients optimize the policy directly; baselines and critics reduce variance, and PPO's clipping keeps updates safe.
- Model-based RL is more sample-efficient but suffers from model bias; PPO-based RLHF is the backbone of modern alignment.

## Review questions

1. How does UCB balance exploration and exploitation without randomness?
   <details><summary>Answer</summary>It adds an uncertainty bonus $c\sqrt{\ln t / N(a)}$ to each arm's estimated value. Rarely tried arms have small $N(a)$ and a large bonus, so they get chosen until their estimates are reliable; as $N(a)$ grows the bonus shrinks and the choice becomes greedy. This is optimism in the face of uncertainty.</details>
2. What is the difference between SARSA and Q-learning?
   <details><summary>Answer</summary>SARSA is on-policy: its target uses $Q(s', a')$ for the action the policy actually takes next, so it learns the value of the behavior policy including its exploration. Q-learning is off-policy: its target uses $\max_{a'} Q(s', a')$, so it learns the value of the greedy policy regardless of how the agent explores.</details>
3. Why does TD learning combine the advantages of Monte Carlo and dynamic programming?
   <details><summary>Answer</summary>Like Monte Carlo it learns from sampled experience and needs no model of the dynamics; like DP it bootstraps from current value estimates, so it can update after every step instead of waiting for the episode to end.</details>
4. What two mechanisms does DQN use to stabilize learning, and what problem does each solve?
   <details><summary>Answer</summary>Experience replay samples past transitions at random, breaking the correlation between consecutive updates. The target network computes bootstrap targets from a periodically updated copy of the Q-network, so targets do not move with every gradient step.</details>
5. Why does subtracting a baseline reduce variance without biasing the policy gradient?
   <details><summary>Answer</summary>The expected value of $\nabla \log \pi(a \mid s)\, b(s)$ is zero, because $b(s)$ does not depend on the action and the gradient of the action probabilities sums to zero. Subtracting it leaves the expectation unchanged while centering the returns, which reduces the variance of the estimate; $V(s)$ is the natural choice.</details>
6. What problem does PPO's clipping solve, and why is PPO important for language models?
   <details><summary>Answer</summary>A large policy update can collapse performance, and because the policy generates its own data it may not recover. Clipping the ratio $\pi_{\text{new}}/\pi_{\text{old}}$ to $[1-\epsilon, 1+\epsilon]$ removes the incentive for large updates without TRPO's explicit KL constraint. PPO is the optimizer in the standard RLHF pipeline used to align language models.</details>
