# Reinforcement Learning

## Overview

- **Primary textbook**: *Reinforcement Learning: An Introduction* by Sutton & Barto (2nd ed) -- [free online](http://incompleteideas.net/book/the-book-2nd.html)
- **Supplementary**: [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) (free, hands-on deep RL), David Silver's RL lectures (UCL, free on YouTube)
- **Prerequisites**: [Deep Learning](../02-deep-learning/), [Probability & Statistics](../../math/07-probability-statistics/)
- **Estimated time**: 4-6 weeks at 10-12 hrs/week

## Key Takeaways

- RL is learning from interaction -- an agent learns a policy by trial and error in an environment
- The exploration-exploitation tradeoff is unique to RL; supervised learning doesn't have it
- Policy gradient methods (PPO) are the backbone of RLHF and modern AI alignment
- Model-based RL is more sample-efficient but requires learning an accurate world model

## How to Study

- Sutton & Barto for theory (Chapters 1-6, then 9-10, then 13)
- Spinning Up for implementation (start with VPG, then PPO)
- Implement Q-learning on a grid world from scratch before using any library
- Read the key papers list from Spinning Up for research depth

---

# Concepts & Techniques

## Core Insight

RL formalizes sequential decision-making under uncertainty. An agent observes a state, takes an action, receives a reward, and transitions to a new state. The goal is to find a policy that maximizes cumulative reward. Unlike supervised learning, the agent must explore to discover which actions are good.

## 1. Multi-Armed Bandits

**Sutton & Barto Ch 2**

**Key ideas**:
- **Exploration vs exploitation**: exploit the best-known action, or explore to find better ones?
- **Epsilon-greedy**: explore uniformly with probability epsilon; exploit otherwise
- **UCB**: Upper Confidence Bound -- optimism in the face of uncertainty; choose action maximizing Q(a) + c * sqrt(ln(t) / N(a))
- **Thompson sampling**: sample from posterior distribution of each arm; naturally balances exploration

**Connection**: bandits are RL without state transitions -- the simplest setting to study exploration.

## 2. Markov Decision Processes

**Sutton & Barto Ch 3**

**Key definitions**:
- **MDP**: (S, A, P, R, gamma) -- states, actions, transition dynamics, reward function, discount factor
- **Policy**: pi(a|s) -- mapping from states to action probabilities
- **Value function**: V^pi(s) = E[sum gamma^t * R_t | s_0 = s, pi] -- expected cumulative reward from state s
- **Q-function**: Q^pi(s,a) = E[sum gamma^t * R_t | s_0 = s, a_0 = a, pi]

**Key theorem**:
- **Bellman equation**: V^pi(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V^pi(s')]. *Intuition*: the value of a state is the immediate reward plus the discounted value of the next state.

## 3. Dynamic Programming

**Sutton & Barto Ch 4**

**Key ideas**:
- **Policy evaluation**: iteratively apply Bellman equation until V^pi converges
- **Policy improvement**: make policy greedy with respect to current value function
- **Policy iteration**: alternate evaluation and improvement until convergence
- **Value iteration**: combine evaluation and improvement into one update; converges to optimal V*

**Limitation**: requires complete knowledge of the MDP (transition dynamics). Real-world problems rarely have this.

## 4. Monte Carlo Methods

**Sutton & Barto Ch 5**

**Key ideas**:
- **First-visit MC**: estimate V(s) as average of returns following first visit to s in each episode
- **Every-visit MC**: average returns from every visit to s
- **Off-policy MC**: learn about one policy while following another; importance sampling
- **Advantage**: no model of environment needed; learns from complete episodes

## 5. Temporal-Difference Learning

**Sutton & Barto Ch 6**

**Key ideas**:
- **TD(0)**: V(s) ← V(s) + alpha * [R + gamma * V(s') - V(s)]; bootstraps from current estimate
- **SARSA**: on-policy TD control; Q(s,a) ← Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
- **Q-learning**: off-policy TD control; Q(s,a) ← Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
- **Double Q-learning**: use two Q-functions to reduce maximization bias

**Key insight**: TD combines the benefits of MC (no model needed) and DP (bootstrapping, no need to wait for episode end).

## 6. Function Approximation & DQN

**Sutton & Barto Ch 9-10, Mnih et al. 2015**

**Key ideas**:
- **Linear function approximation**: Q(s,a) = w^T * phi(s,a); convergent with TD under certain conditions
- **Neural function approximation**: Q(s,a; theta); can diverge (deadly triad: function approx + bootstrapping + off-policy)
- **DQN**: deep Q-network; experience replay (break correlation) + target network (stabilize targets)
- **Double DQN**: use online network to select action, target network to evaluate; reduces overestimation

## 7. Policy Gradient Methods

**Sutton & Barto Ch 13**

**Key ideas**:
- **Policy gradient theorem**: nabla J(theta) = E[sum_t nabla log pi(a_t|s_t; theta) * G_t]; gradient of expected return
- **REINFORCE**: Monte Carlo policy gradient; high variance but unbiased
- **Baseline**: subtract b(s) from G_t to reduce variance; natural choice: V(s)
- **Actor-critic**: actor (policy) + critic (value function); critic provides low-variance baseline

**Key theorem**: the policy gradient theorem provides an exact expression for the gradient even though the environment dynamics are unknown. This is why policy gradients work.

## 8. Trust Region Methods: TRPO & PPO

**Spinning Up, Schulman et al. 2015, 2017**

**Key ideas**:
- **TRPO**: constrain policy update to stay within a KL divergence trust region; prevents catastrophic updates
- **PPO (clipped)**: simpler alternative; clip the probability ratio r(theta) = pi_new/pi_old to [1-eps, 1+eps]; prevents large updates without explicit KL constraint
- **PPO is the default**: used for most practical deep RL including RLHF for language models

**Why PPO matters**: it's the algorithm behind ChatGPT's RLHF training. Understanding PPO is essential for AI alignment work.

## 9. Continuous Control: DDPG, TD3, SAC

**Spinning Up**

**Key ideas**:
- **DDPG**: Deep Deterministic Policy Gradient; actor-critic for continuous actions; off-policy
- **TD3**: Twin Delayed DDPG; two critics (take minimum), delayed policy updates, target policy smoothing
- **SAC**: Soft Actor-Critic; maximize reward + entropy; encourages exploration; state-of-the-art for continuous control

## 10. Model-Based RL

**Sutton & Barto Ch 8, research papers**

**Key ideas**:
- **Dyna**: learn a model of the environment; use it to generate simulated experience for planning
- **World models**: learn latent dynamics model; plan in latent space (Ha & Schmidhuber 2018)
- **MuZero**: learn model that predicts rewards and value without reconstructing observations; superhuman in Go, chess, Atari
- **Tradeoff**: more sample-efficient than model-free, but model errors compound (model bias)

## 11. Multi-Agent RL

**Research frontier**

**Key ideas**:
- **Cooperative**: agents share a common reward; challenges: credit assignment, communication
- **Competitive**: zero-sum games; self-play (AlphaGo, OpenAI Five)
- **Mixed**: general-sum games; Nash equilibrium is the solution concept
- **Emergent communication**: agents develop communication protocols through interaction

## 12. RLHF & Alignment

**Research frontier -- connects to [Deep Learning](../02-deep-learning/)**

**Key ideas**:
- **Reward modeling**: train a reward model from human preference comparisons (A > B)
- **RLHF pipeline**: SFT → reward model → PPO optimization against reward model
- **DPO**: Direct Preference Optimization; reparameterize reward in terms of policy; no separate reward model needed
- **Constitutional AI**: model self-critiques using principles, then fine-tunes on revised outputs
- **Scaling supervision**: as models surpass human ability, how do we provide training signal?

**Key papers**: InstructGPT (Ouyang 2022), Constitutional AI (Bai 2022), DPO (Rafailov 2023)

---

## Technique Catalog

| Algorithm | Type | On/Off-Policy | Best For |
|-----------|------|---------------|----------|
| Q-learning | Value-based | Off-policy | Discrete actions, tabular |
| DQN | Value-based | Off-policy | Discrete actions, function approx |
| REINFORCE | Policy gradient | On-policy | Simple, educational |
| PPO | Policy gradient | On-policy | General purpose, RLHF |
| DDPG/TD3 | Actor-critic | Off-policy | Continuous control |
| SAC | Actor-critic | Off-policy | Continuous control (with exploration) |
| MuZero | Model-based | -- | Games, planning |

## Connections to Other Tracks

| Concept | Connected Track | Application |
|---------|-----------------|-------------|
| Markov chains, Bellman equations | [Probability](../../math/07-probability-statistics/) | MDPs are stochastic processes |
| Dynamic programming | [Algorithms (DP)](../../algorithms/07-dynamic-programming/) | Policy/value iteration IS dynamic programming |
| Game theory | [Competitive Programming](../../competitive-programming/) | Multi-agent RL, minimax |
| Neural networks | [Deep Learning](../02-deep-learning/) | Function approximation for Q and policy |
| PPO, reward modeling | [Deep Learning (RLHF)](../02-deep-learning/) | Alignment of language models |
| Honours research | User's LGP + Q-learning project | Linear genetic programming with RL |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Anthropic | RLHF, DPO, constitutional AI, alignment research | PhD-level |
| OpenAI | PPO for ChatGPT, RLHF pipeline, scaling RL | PhD-level |
| DeepMind | AlphaGo/Zero/Fold, MuZero, multi-agent, robotics | PhD-level |
| Robotics companies | Control policies, sim-to-real transfer | Expert |
| Jane Street | Optimal execution, portfolio as MDP | Expert |
| Two Sigma | Trading strategies as sequential decisions | Expert |
