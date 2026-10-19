# Documentation index

This index uses Diátaxis to help readers choose a document by the task they need to do.

| Type | Reader goal | Start here |
|---|---|---|
| Tutorial | Build a local serving stack from an empty checkout | [Local quickstart](../../../README.md#quickstart) |
| How-to | Recover a failing engine | [Engine crashloop runbook](runbooks/engine-crashloop.md) |
| How-to | Respond to a TTFT budget burn | [TTFT budget runbook](runbooks/TTFTBudgetBurnFast.md) |
| How-to | Operate durable queues | [Dead letter runbook](runbooks/DurableDeadLetters.md) |
| Explanation | Understand the system boundaries | [C4 context diagram](c4/context.d2) · [container diagram](c4/containers.d2) · [engine components](c4/components-engine.d2) |
| Reference | Check the supported runtime and deployment configuration | [Configuration schemas](../../contracts/config/) |
| Reference | Check model and dataset disclosures | [Model card](MODEL_CARD.md) · [Datasheet](DATASHEET.md) |
| Reference | Check format and API retirement windows | [Deprecation policy](DEPRECATION.md) |

## Maintenance

The owner of each service updates its runbook when an alert, command, threshold, or recovery step changes. The docs owner updates the tutorial and reference links with the same pull request that changes the corresponding contract. Review links in CI and check every command in a disposable environment before approving an operational procedure.
