# Privacy and PII

## Overview

- **Primary references**: [NIST Privacy Framework 1.0](https://www.nist.gov/privacy-framework) (free); Carlini et al., [*Extracting Training Data from Large Language Models*](https://arxiv.org/abs/2012.07805) (free)
- **Supplementary**: [GDPR](https://gdpr-info.eu/) (free), especially articles 5 (principles) and 17 (erasure); Microsoft, [Presidio](https://microsoft.github.io/presidio/) (free, a reference PII detector); Lukas et al., [*Analyzing Leakage of Personally Identifiable Information in Language Models*](https://arxiv.org/abs/2302.00539) (free)
- **Prerequisites**: none for the reading; `data.04` for the course module
- **Estimated time**: 3 to 5 h in course Pass 3

## Key Takeaways

- **Language models memorize**, and repeated or rare strings (emails, phone numbers, keys) are the easiest to extract. What is in the corpus can come out of the model.
- **Scrub before training, redact before logging.** The corpus pipeline replaces PII with typed placeholders; the gateway applies the same detector list to its logs.
- **A detector is a classifier with a precision and a recall.** Measure both on labelled fixtures, and test the lookalikes (ISBNs, version strings) that cause false positives.
- **Data minimization is the cheapest control**: what you never collect or keep cannot leak or need erasing.

## How to Study

Read Carlini et al. sections 1 to 5, then the NIST Privacy Framework core. In the course, `ethics.02` writes your PII policy (what you detect, replace, log, and keep); `data.05` implements the scrub and is graded on recall and precision against it.

---

# Concepts & Techniques

## Core Insight

Privacy in an ML system is a data-flow property: personal data enters through the corpus and through user requests, and leaves through model outputs and logs. A policy names each flow and its control; the pipeline and the gateway enforce it; tests on labelled fixtures show it works.

## 1. Where personal data enters and leaves

**Key ideas**:
- **In**: crawled text, user prompts, uploaded documents. **Out**: generations, logs, traces, eval reports.
- **Memorization**: duplicated sequences are memorized far more often, which is one more reason deduplication (`data.03`, `data.04`) runs before training.

## 2. Detection and redaction

**Key ideas**:
- **Typed placeholders** (`<EMAIL>`, `<PHONE>`, `<CARD>`) keep text shape for training while removing the value; Luhn checks separate card numbers from other digit runs.
- **Audit spans** record what was replaced and where, without storing the value.
- **Gateway logs** reuse the detector list (`gw.08`), so the same policy covers training data and traffic.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ethics.02` | Privacy and PII policy | practice | 3 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ethics.02` | [Privacy and PII policy](01-privacy-and-pii-policy.md) | practice | 3 |
<!-- /ol:chapters -->

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Responsible AI](../) | the track overview and how the six topics connect |
| [Corpus Pipeline](../../data-engineering/05-corpus-pipeline/) | the PII scrub (`data.05`) |
| [Observability](../../systems/04-observability/) | what traces and logs may contain |
| [Security](../../software-craftsmanship/11-security/) | secrets review and the threat model |

Pass 12: [Consent and biometric evidence in model cards](02-consent-and-biometric-evidence.md) (`ethics.08`).
