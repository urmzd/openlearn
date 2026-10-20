# Privacy and PII

Language models memorize, and repeated or rare strings (emails, phone numbers, keys) are the easiest to extract. What is in the corpus can come out of the model. Privacy in an ML system is therefore a data-flow property: personal data enters through the corpus and through user requests, and leaves through model outputs and logs. A policy names each flow and its control, the pipeline and the gateway enforce it, and tests on labelled fixtures show it works.

This chapter writes that policy and then extends it to media. The first section, on PII, covers what counts as personal data, where it enters and leaves, and why you scrub before training and redact before logging: the corpus pipeline replaces PII with typed placeholders such as `<EMAIL>`, and the gateway applies the same detector list to its logs. It treats a detector as a classifier with a precision and a recall, measured on labelled fixtures that include the lookalikes (ISBNs, version strings, trace ids) that cause false positives. Throughout, data minimization is the cheapest control: what you never collect or keep cannot leak or need erasing. The second section turns to images and voice, where a license to a recording is not consent to use a person's face or voice, and builds the evidence program that keeps multimodal model cards honest.

> **Learning objectives**
> - Map every flow by which personal data enters and leaves an LLM system, and assign each a control.
> - Write a PII policy with typed placeholders, an audit that never stores values, log retention, and an erasure procedure.
> - Measure a detector's precision and recall, and use structural rules such as the Luhn check to keep precision high.
> - Keep license and consent as separate records for biometric media, and bind a model card's numbers to their evidence.

## Before you begin

The reading has no prerequisites. The first section follows the shape of [Data Licensing](../33-data-licensing/README.md), a policy as prose plus decisions a program checks, so reading that chapter first helps. The policy is implemented by the scrub in [PII scrub with typed placeholders and audit spans](../../data-engineering/05-corpus-pipeline/05-pii-scrub.md) in *Data Engineering*, which runs after deduplication. The second section assumes the vision and speech towers of [AI Engineering](../README.md) and the model card from [Model and Data Cards](../35-model-and-data-cards/README.md). Good preparation is Carlini et al., sections 1 to 5, and the core of the NIST Privacy Framework.

## In this chapter

- [Privacy and PII policy](01-privacy-and-pii-policy.md): the five kinds of personal data and secrets the system detects, the flows they travel, detectors as classifiers, the Luhn check, and a walkthrough of the reference policy, its machine-readable half, and twelve review cases.
- [Consent and biometric evidence in model cards](02-consent-and-biometric-evidence.md): consent fields on the ledger, pooled WER per group, digest-bound model cards, and an evidence program that refuses changed or revoked media before inference.

The policy also touches other books. What traces and logs may contain is a concern of [Observability](../../systems/11-observability/) in *Systems Engineering*, and the secrets review and threat model live in [Security](../../software-engineering/09-security/) in *Software Engineering*.

## Further reading

- [NIST Privacy Framework 1.0](https://www.nist.gov/privacy-framework) (free).
- Carlini et al., [*Extracting Training Data from Large Language Models*](https://arxiv.org/abs/2012.07805) (free).
- [GDPR](https://gdpr-info.eu/) (free), especially articles 5 (principles) and 17 (erasure).
- Microsoft, [Presidio](https://microsoft.github.io/presidio/) (free), a reference PII detector.
- Lukas et al., [*Analyzing Leakage of Personally Identifiable Information in Language Models*](https://arxiv.org/abs/2302.00539) (free).
