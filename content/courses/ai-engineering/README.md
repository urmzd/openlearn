# AI Engineering

This book follows an AI system from its learning assumptions to a model you can build, train, serve, evaluate, and operate responsibly. The learning foundations explain the statistical and neural models; the construction chapters expose numerics, autograd, tokenization, attention, and inference; the production parts build retrieval, agents, evaluation, and serving around them. Read the worked implementations alongside the reasoning, commit to predictions before revealing answers, and validate each claim against a small example or a controlled failure.

## What you will be able to do

- Derive a model or learning objective from its assumptions and check it on a small example.
- Build and explain autograd, tokenizers, language models, attention, and an inference engine.
- Choose training, adaptation, routing, and serving strategies using measured constraints.
- Build retrieval and agent applications with explicit evaluation and failure tests.
- Validate data rights, privacy, safety, and deployment policies alongside model behavior.

## Ways through this book

Read the parts in order for the full progression, or choose a route below and follow its prerequisites. Pause at the review questions before revealing worked answers.

- **Understanding learning:** [Statistical Learning](01-statistical-learning/README.md), then [Deep Learning](02-deep-learning/README.md), then [Reinforcement Learning](04-reinforcement-learning/README.md).
- **Building and training a model:** [Numerics and Autograd](06-numerics-and-autograd/README.md), then [Tokenizers](07-tokenizers/README.md), then [The Modern Decoder Block](13-decoder-block/README.md). Follow the [training-to-inference route](training-to-inference.md) for learning methods, PPO/GRPO, distillation, local serving, batch invariance, and scale.
- **Shipping AI applications:** [Inference Engines and GPU Deployment](20-inference-engines/README.md), then [Retrieval and RAG](27-retrieval-and-rag/README.md), then [Agent SDK](28-agents/README.md), then [LLM Evaluation](29-evaluation/README.md).
- **Responsible deployment:** [Data Licensing](33-data-licensing/README.md), then [Privacy and PII](34-privacy-and-pii/README.md), then [Bias and Safety Evaluations](36-bias-and-safety-evals/README.md).

## Before you start

The mathematical prerequisites are in [Mathematics](../mathematics/README.md), and the programming background is in [Programming and Algorithms](../programming-and-algorithms/README.md). [Data Engineering](../data-engineering/README.md) owns ingestion, corpus pipelines, and data quality. [Systems Engineering](../systems/README.md) owns shared protocols, caching, durable execution, Kubernetes, and observability. [Software Engineering](../software-engineering/README.md) owns testing, authorization, design practices, and documentation. Follow those links when a chapter needs a shared mechanism; the AI-specific behavior stays here.

[How to read these books](../../references/how-to-read.md) explains the shared first-principles, prediction, reasoning, testing, and validation method.
