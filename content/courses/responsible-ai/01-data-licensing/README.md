# Data Licensing

You cannot honor terms you did not record. Every document in a training corpus came from somewhere, under some license or under none, and once those documents are mixed into shards the question "may we train on this and release what we trained?" can no longer be answered by looking. This chapter treats licensing for training data as a provenance problem: each byte of the corpus must trace back to a source, and each source must carry the terms under which it was obtained. Once provenance is data, compliance becomes a check that runs in CI.

The chapter has a single section, [Data licensing and the ledger](01-data-licensing-and-the-ledger.md). It explains why text carries a license at all, how licenses group into families, and how SPDX identifiers give each one a name a program can match. It then builds the two artifacts the rest of the system depends on: a prose licensing policy and a machine-readable allowlist that decides each common license per use, plus the ledger row that binds a reviewed license to the exact bytes fetched. This is engineering, not legal advice: the chapter teaches you to make provenance auditable, and a lawyer decides what a license permits.

> **Learning objectives**
> - Explain why a dataset's license is a property of every source in it, and why "no license" means "no".
> - Classify licenses into families (public domain, permissive, attribution, ShareAlike, non-commercial, no derivatives) by what they permit and ask.
> - Write a licensing policy and an allowlist keyed by SPDX identifiers, with separate decisions for training and evaluation.
> - Describe the ledger row recorded at fetch time and how verification fails a build on a refused or unknown license.

## Before you begin

The reading needs no prerequisites. The policy is applied to the corpus pipeline in *Data Engineering*: license capture happens in [Async fetch with resume, checksums, and license capture](../../data-engineering/05-corpus-pipeline/01-async-fetch-resume-checksums.md), and every shard is verified against the allowlist in [Licensing ledger verification and the datasheet](../../data-engineering/05-corpus-pipeline/08-ledger-and-datasheet.md). Before you start, try looking up the SPDX identifiers of five datasets you have used; the exercise shows quickly how often a dataset card's license line is vague or wrong.

## In this chapter

- [Data licensing and the ledger](01-data-licensing-and-the-ledger.md): why text has a license, license families and SPDX names, the rules that are not judgment calls, and a step-by-step walkthrough of the reference policy and allowlist, decision by decision.

Later chapters build on this one. The datasheet in [Model and Data Cards](../03-model-and-data-cards/README.md) reports what the ledger records, and the policy structure (prose plus decisions a program checks) returns for personal data in [Privacy and PII](../02-privacy-and-pii/README.md).

## Further reading

- Longpre et al., [*The Data Provenance Initiative*](https://arxiv.org/abs/2310.16787) (free), the audit of dataset licensing and attribution.
- The [SPDX License List](https://spdx.org/licenses/) (free).
- [Creative Commons licenses](https://creativecommons.org/licenses/list.en) (free) and [Choose a License](https://choosealicense.com/) (free).
- Gokaslan et al., [*CommonCanvas*](https://arxiv.org/abs/2310.16825) (free), on training on openly licensed data.
- The [Common Crawl terms of use](https://commoncrawl.org/terms-of-use) (free).
