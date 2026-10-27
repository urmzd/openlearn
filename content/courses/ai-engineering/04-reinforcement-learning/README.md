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

1. [Bandits, MDPs, and planning](01-bandits-mdps-and-planning.md): Distinguish immediate reward optimization from sequential credit assignment.
2. [On-policy, off-policy, and value learning](02-on-policy-off-policy-and-value-learning.md): Compare Monte Carlo, TD, SARSA, and Q-learning data assumptions.
3. [Policy gradients and actor-critic](03-policy-gradients-and-actor-critic.md): Explain policy-gradient baselines, PPO clipping, and continuous control.
4. [World models, multiple agents, and RLHF](04-models-multiple-agents-and-rlhf.md): Relate general RL methods to language-model post-training.
