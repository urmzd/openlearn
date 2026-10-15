# Bias and Safety Evals

## Overview

- **Primary references**: Parrish et al., [*BBQ: A Hand-Built Bias Benchmark for Question Answering*](https://arxiv.org/abs/2110.08193) (free); Gehman et al., [*RealToxicityPrompts*](https://arxiv.org/abs/2009.11462) (free)
- **Supplementary**: Liang et al., [*Holistic Evaluation of Language Models (HELM)*](https://arxiv.org/abs/2211.09110) (free); Nangia et al., [*CrowS-Pairs*](https://arxiv.org/abs/2010.00133) (free); Weidinger et al., [*Ethical and social risks of harm from Language Models*](https://arxiv.org/abs/2112.04359) (free)
- **Prerequisites**: the model-evaluation harness (`L6.7`), [Probability & Statistics](../../math/07-probability-statistics/) (confidence intervals), [Model and Data Cards](../03-model-and-data-cards/)
- **Estimated time**: 1 week in course Pass 9

## Key Takeaways

- **Bias is measured as a difference**: the same prompt template with only a group term changed, scored on the same metric, with a confidence interval on the gap.
- **Safety evals need a scorer you trust**: a deterministic one (a lexicon, a classifier with known precision) before an LLM judge.
- **Small models fail differently, not less.** A 10M-parameter story model will not produce dangerous instructions, but it will reproduce stereotypes from its corpus; measure what your model can actually do.
- **An eval is a test**: fixed prompts, fixed seeds, a threshold, and a place in the release gate.

## How to Study

Read BBQ and RealToxicityPrompts, then the HELM sections on bias and toxicity metrics. In the course, `ethics.04` is a build module: you write paired-template bias evals and a toxicity scorer in Python, run them through `EvalSuite`, and report the results in the model card.

---

# Concepts & Techniques

## Core Insight

A fairness or safety claim is only as good as its measurement. Paired templates isolate the effect of one attribute; bootstrap intervals separate a real gap from noise; and running the suite on every release candidate turns a one-time audit into a regression test.

## 1. Measuring bias

**Key ideas**:
- **Paired templates**: identical prompts differing in one group term; compare log-likelihoods or completions.
- **Uncertainty**: a paired bootstrap interval on the difference, as in `M07.5`; report the interval, not just the point estimate.

## 2. Measuring safety

**Key ideas**:
- **Toxicity and refusal rates** on fixed prompt sets with seeded sampling.
- **Where it runs**: the `EvalSuite` workflow (`dur.11`) and the release gate (`dur.12`); results land in the model card (`ethics.03`).

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ethics.04` | Bias and safety evals | build | 9 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ethics.04` | [Bias and safety evals](01-bias-and-safety-evals.md) | build | 9 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Responsible AI](../) | the track overview and how the six topics connect |
| [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) | eval runners, judges, and A/B experiments |
| [Probability & Statistics](../../math/07-probability-statistics/) | bootstrap intervals and paired tests |
| [Usage Policy](../05-usage-policy/) | what you do about the failures you find |
