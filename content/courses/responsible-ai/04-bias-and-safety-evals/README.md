# Bias and Safety Evals

A fairness or safety claim is only as good as its measurement. This chapter builds the measurements. Bias is measured as a difference: the same prompt template with only a group term changed, scored on the same metric, with a confidence interval on the gap. Safety evals need a scorer you trust, which means a deterministic one (a lexicon, a classifier with known precision) before an LLM judge. Small models fail differently, not less: a 10M-parameter story model will not produce dangerous instructions, but it will reproduce stereotypes from its corpus, so you measure what your model can actually do. And an eval is a test: fixed prompts, fixed seeds, a threshold, and a place in the release gate.

The chapter's single section, [Bias and safety evals](01-bias-and-safety-evals.md), is a build. You write paired-template bias evals and a toxicity scorer in Python, compute Wilson intervals for every rate and bootstrap intervals for every gap, run them through `EvalSuite`, gate the release on interval bounds, and hand the rows to the model card. Running the suite on every release candidate turns a one-time audit into a regression test.

> **Learning objectives**
> - Measure a bias gap as a paired counterfactual difference with a bootstrap interval, and name the confounds (length, frequency) it cannot separate.
> - Report toxicity, refusal, and over-refusal rates with Wilson intervals from a seeded generator.
> - Build a noisy-OR lexicon scorer and measure its ROC-AUC and average precision on labelled data.
> - Gate a release on interval bounds, failing any missing row.

## Before you begin

You need the model evaluation harness from [LM evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) in *Build Your Own LLM System*, the confidence intervals and paired tests of [Probability & Statistics](../../mathematics-for-computing/07-probability-statistics/) in *Mathematics for Computing*, and the model card from [Model and Data Cards](../03-model-and-data-cards/README.md), where these numbers are reported. Reading BBQ and RealToxicityPrompts, then the HELM sections on bias and toxicity metrics, is good preparation.

## In this chapter

- [Bias and safety evals](01-bias-and-safety-evals.md): paired templates and the bias gap, stereotype preference, Wilson intervals, a noisy-OR toxicity scorer and how to measure it, seeded safety rates, and a release gate that decides on evidence, with an annotated walkthrough of the full solution.

For eval runners, judges, and A/B experiments at platform scale, see [LLM Evaluation](../../ai-platform-engineering/09-llm-evaluation/) in *AI Platform Engineering*. What you do about the failures these evals find is the subject of the next chapter, [Usage Policy](../05-usage-policy/README.md).

## Further reading

- Parrish et al., [*BBQ: A Hand-Built Bias Benchmark for Question Answering*](https://arxiv.org/abs/2110.08193) (free).
- Gehman et al., [*RealToxicityPrompts*](https://arxiv.org/abs/2009.11462) (free).
- Liang et al., [*Holistic Evaluation of Language Models (HELM)*](https://arxiv.org/abs/2211.09110) (free).
- Nangia et al., [*CrowS-Pairs*](https://arxiv.org/abs/2010.00133) (free).
- Weidinger et al., [*Ethical and social risks of harm from Language Models*](https://arxiv.org/abs/2112.04359) (free).
