# World models, multiple agents, and RLHF

This part of [Reinforcement Learning](README.md) develops the mechanisms below. Work the numerical examples or trace the code before checking the answer.

> **Learning objectives**
> - Relate general RL methods to language-model post-training.

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

[Chapter contents](README.md)
