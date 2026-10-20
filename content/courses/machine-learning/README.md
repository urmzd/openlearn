# Machine Learning and AI

This book runs from the statistical foundations of learning to the systems that train and serve today's large language models. It is for engineers and students who can program and have the mathematics in hand, and who want to understand machine learning well enough to reason about it rather than only call it. Each chapter pairs a free primary textbook with the papers and documentation that bring it up to date, and several include small runnable programs that check their own formulas.

## What you will be able to do

- Frame a learning problem statistically, reason about bias and variance, and choose and validate a classical model.
- Derive backpropagation and the main deep-learning architectures (MLPs, CNNs, RNNs and LSTMs, transformers) from their equations, and implement them.
- Formulate a problem as a Markov decision process and apply value-based and policy-gradient methods, through PPO and RLHF.
- Estimate the compute, memory, and data budget of a training or fine-tuning run, and choose a parallelism and post-training method to match.
- Reason about LLM inference: prefill and decode, the KV cache, batching, quantization, and the cost and latency of serving.

## Ways through this book

Reading every chapter takes about 37 weeks at ten to twelve hours a week, so most readers follow one of these routes instead.

- **Foundations, or preparing for ML interviews** (about 16 weeks): [Chapter 1: Statistical Learning](01-statistical-learning/README.md), then [Chapter 2: Deep Learning](02-deep-learning/README.md) with [Chapter 6: Neural Network Architectures](06-neural-architectures/README.md) read alongside its first four weeks. If you know the basics, start at Chapter 2.
- **AI engineering and ML infrastructure** (about 15 weeks): [Transformers and modern architectures](02-deep-learning/README.md#transformers-and-modern-architectures) in Chapter 2, then [Chapter 5: Foundation Models](05-foundation-models/README.md), [Chapter 4: LLM Systems](04-llm-systems/README.md) with its sections [4.1 Quantization](04-llm-systems/01-quantization/README.md) through [4.5 Serving platforms](04-llm-systems/05-serving-platforms.md), and [Chapter 7: Training and Post-Training](07-training-and-post-training/README.md) for scoping fine-tunes.
- **Alignment and AI safety** (about 16 weeks): Chapter 2, then [Chapter 3: Reinforcement Learning](03-reinforcement-learning/README.md) through its [RLHF section](03-reinforcement-learning/README.md#rlhf-and-alignment), then [Post-training](07-training-and-post-training/README.md#post-training) in Chapter 7 for DPO and GRPO.
- **Quantitative and trading roles** (about 11 weeks): Chapter 1, then Chapter 3, with extra time on [bandits](03-reinforcement-learning/README.md#multi-armed-bandits) and [MDPs](03-reinforcement-learning/README.md#markov-decision-processes).

## Before you start

You need the mathematics in *Mathematics for Computing*: [Linear Algebra](../mathematics-for-computing/04-linear-algebra/README.md), calculus from [Calculus 1](../mathematics-for-computing/02-calculus-1/README.md) through [multivariable calculus](../mathematics-for-computing/05-calculus-3/README.md), and [Probability and Statistics](../mathematics-for-computing/08-probability-statistics/README.md). To see the whole stack built end to end, follow the [course path](../build-an-llm/paths/course/README.md) of *[Build Your Own LLM System](../build-an-llm/README.md)*. *[AI Platform Engineering](../ai-platform-engineering/README.md)* covers the frameworks, retrieval, evaluation, and serving infrastructure around models, and infrastructure readers should pair Chapter 4 with [Cloud Native](../systems/03-cloud-native/README.md) in *Systems and Architecture* and with *[Data Engineering](../data-engineering/README.md)*.

New to these books? [How to read these books](../../references/how-to-read.md) explains how sections, review questions, and solutions work.
