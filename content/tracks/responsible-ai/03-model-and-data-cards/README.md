# Model and Data Cards

## Overview

- **Primary references**: Mitchell et al., [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993) (free); Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010) (free)
- **Supplementary**: Hugging Face, [model card guide](https://huggingface.co/docs/hub/model-cards) (free); Pushkarna et al., [*Data Cards*](https://arxiv.org/abs/2204.01075) (free); the course ships `MODEL_CARD.md` and `DATASHEET.md` templates with the [course contracts](../../../../course/contracts/) when this module lands
- **Prerequisites**: [Data Licensing](../01-data-licensing/) and [Privacy and PII](../02-privacy-and-pii/); for the course module, the trained capstone model (`C1`)
- **Estimated time**: 3 to 4 h in course Pass 9

## Key Takeaways

- **A model card is the model's interface documentation**: intended use, out-of-scope use, training data, evaluation results with uncertainty, and known limitations.
- **A datasheet answers why the dataset exists**, how it was collected and cleaned, what it contains, and what it should not be used for.
- **Cards are generated where possible and checked always.** The datasheet comes from the ledger (`data.08`); the release workflow refuses a model without a card (`dur.12`).
- **Third-party models are labelled as such**: the agent model you fetch (SmolLM2-135M-Instruct) gets its own provenance line, not yours.

## How to Study

Read both papers and then three real model cards on the Hugging Face Hub, noting what each omits. In the course, `ethics.03` writes the datasheet and model card for your capstone model and wires them into the release gate.

---

# Concepts & Techniques

## Core Insight

Documentation is how a model's limits travel with it. Without a card, every downstream user re-learns the model's failure modes in production; with one, the limits are stated, measured, and versioned next to the weights.

## 1. What a card contains

**Key ideas**:
- **Model card**: details, intended use, factors, metrics, evaluation data, training data, quantitative analyses, ethical considerations, caveats.
- **Datasheet**: motivation, composition, collection, preprocessing, uses, distribution, maintenance.

## 2. Cards in the release path

**Key ideas**:
- **Generated sections**: corpus statistics and license summary from the ledger, eval tables from the `EvalSuite` report.
- **Gate**: `ModelRelease` (`dur.12`) checks the card exists and its eval section matches the release candidate.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ethics.03` | Datasheet and model card | practice | 9 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ethics.03` | [Datasheet and model card](01-datasheet-and-model-card.md) | practice | 9 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Responsible AI](../) | the track overview and how the six topics connect |
| [Bias and Safety Evals](../04-bias-and-safety-evals/) | the numbers the card reports |
| [Documentation Writing](../../software-craftsmanship/06-documentation-writing/) | cards as reference documents |
| [tinyllm Capstones](../../../courses/build-an-llm/capstones/) | the model the card describes |
