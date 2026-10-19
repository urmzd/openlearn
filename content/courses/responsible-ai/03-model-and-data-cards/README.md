# Model and Data Cards

Documentation is how a model's limits travel with it. Without a card, every downstream user re-learns the model's failure modes in production; with one, the limits are stated, measured, and versioned next to the weights. This chapter writes the two documents every released model needs. A model card is the model's interface documentation: intended use, out-of-scope use, training data, evaluation results with uncertainty, and known limitations. A datasheet answers why the dataset exists, how it was collected and cleaned, what it contains, and what it should not be used for.

The chapter's single section, [Datasheet and model card](01-datasheet-and-model-card.md), builds both for a small reference model and wires them into the release gate. Cards are generated where possible and checked always: the datasheet's counts come from the ledger, the evaluation table from the eval suite, and the release workflow refuses a model without a card. The section also shows how to label a third-party model as such: the agent model the system fetches, SmolLM2-135M-Instruct, gets its own provenance card, not yours.

> **Learning objectives**
> - Write a model card whose out-of-scope list is specific enough for a gateway rule to enforce.
> - Report every evaluation number with its confidence interval and release threshold.
> - Write a datasheet that records every source, license, and preprocessing step, consistent with the licensing allowlist.
> - Write an honest third-party model card with a pinned upstream revision.

## Before you begin

Read [Data Licensing](../01-data-licensing/README.md) and [Privacy and PII](../02-privacy-and-pii/README.md) first: the datasheet reports what the ledger records and what the PII scrub found. The section applies to the trained capstone model, the [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) in *Build Your Own LLM System*, and uses a fast-rebuilding 4-gram as its worked example. A useful warm-up is to read Mitchell et al. and Gebru et al., then three real model cards on the Hugging Face Hub, noting what each omits.

## In this chapter

- [Datasheet and model card](01-datasheet-and-model-card.md): what each document is for, the rules that make them useful (enforceable scope, intervals, measured limitations, consistent licenses, labelled third-party weights, versioning), and an annotated walkthrough of the reference model card, datasheet, and third-party card.

The numbers the card reports come from the next chapter, [Bias and Safety Evals](../04-bias-and-safety-evals/README.md). For cards as reference documents in general, see [Documentation Writing](../../diagramming/02-documentation-writing/) in *Diagramming and Documentation*.

## Further reading

- Mitchell et al., [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993) (free).
- Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010) (free).
- Hugging Face, [model card guide](https://huggingface.co/docs/hub/model-cards) (free).
- Pushkarna et al., [*Data Cards*](https://arxiv.org/abs/2204.01075) (free).
- The `MODEL_CARD.md` and `DATASHEET.md` templates in the [system contracts](../../../../course/contracts/).
