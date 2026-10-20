# Research Reading Plan

This plan is for readers pursuing research depth in machine learning and AI. It is a reading guide across the library's books and the primary literature, and it owns no chapters of its own. It assumes the mathematics foundations in *Mathematics* and [Statistical learning](../ai-engineering/01-statistical-learning/README.md) are already complete; if they are not, start with the [Technology Foundations study plan](../computer-science/study-plan.md). At about ten hours a week the plan takes twenty weeks or more, and the last stage is open-ended by design.

## Deep learning, in full

Spend the first eight weeks on [Deep learning](../ai-engineering/02-deep-learning/README.md) in *AI Engineering*, reading Goodfellow, Bengio, and Courville in full: Parts I to III, all twenty chapters. This is the longest stage because everything after it assumes it. For a from-scratch view of the same ideas, the autograd sections of [Foundations](../ai-engineering/06-numerics-and-autograd/README.md) in *AI Engineering* build the machinery by hand.

## Information theory

Weeks nine and ten cover entropy, KL divergence, and channel coding with MacKay, alongside [Information Theory](../mathematics/17-information-theory/README.md) in *Mathematics*. These quantities come back in every loss function and evaluation metric you will read about.

## Reinforcement learning

Weeks eleven to fourteen read Sutton and Barto in full, together with the key papers from Spinning Up, guided by [Reinforcement learning](../ai-engineering/04-reinforcement-learning/README.md).

## Transformer architectures

Weeks fifteen to seventeen read the papers that define modern language models: *Attention Is All You Need*, BERT, GPT, and the scaling-laws papers. [Foundation models](../ai-engineering/05-foundation-models/README.md) frames them, and [The transformer (2017)](../ai-engineering/11-transformer/README.md), [Objectives and adaptation](../ai-engineering/12-objectives-and-adaptation/README.md), and [The modern decoder block](../ai-engineering/13-decoder-block/README.md) in *AI Engineering* implement each idea so you can check your reading against working code.

## Alignment and safety

Weeks eighteen and nineteen read the Constitutional AI, RLHF, and DPO papers. [Training and post-training](../ai-engineering/19-training-methods/README.md) in *AI Engineering* and [Post-training](../ai-engineering/18-post-training/README.md) in *AI Engineering* cover the methods, and [AI Engineering](../ai-engineering/README.md) covers the evaluation and governance around them.

## Research exploration

From week twenty on, pick a subfield, read ten or more papers in it, and implement one. Choose the paper you implement for what it would teach you, not for how impressive it would look, and compare your results with the paper's before you trust either.

## Using the plan

Pick your starting point honestly: if you already know a stage's material, skip it. Parallelize where you can, since the mathematics topics in particular are designed to be studied side by side. Do not skip the connections between books, because the cross-references are where real understanding lives. Adjust the pace to your hours. And if you are also preparing for interviews, use [Interview Preparation](../interview-preparation/README.md) last; the guides are most effective after you have built deep understanding.
