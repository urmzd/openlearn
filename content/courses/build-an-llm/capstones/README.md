# Capstones

The capstones run the whole system you built on one goal. C1 trains a roughly 10M-parameter Llama-family model on TinyStories end to end: your corpus pipeline cleans the data as a durable workflow, your tokenizer is trained and ablated (BPE vs Unigram), your trainer runs with mixed precision and resumes bitwise, the model is evaluated against the model zoo, released through your release workflow with a model card and data ledger, and served by your engine behind your gateway. C2 (optional) post-trains it into an instruction follower.

**Course passes**: 9 (C1, gate MS-C1, part of MS-P9); 10, optional (C2, gate MS-C2)

**Before you start**: every pass before it. C1 exercises the data modules, the tokenizers, the modern block, the optimizers, `L11.1`, the durable workflows (`dur.06`, `dur.09`, `dur.11`, `dur.12`), the inference stack, and the serving platform; C2 needs [Part 12](../p12-post-training/).

## Key ideas

- **One config, two sizes**: the full run (about 10.4M parameters, 100M tokens, an estimated 4 to 12 h on an M-series laptop) and a `short` config (2.5M parameters, under 1 h) accepted at a looser threshold.
- **Ablations are experiments**: tokenizer (BPE vs Unigram), attention (MLA vs GQA at equal KV bytes), MLP (MoE vs dense at equal active parameters), each reported with a paired confidence interval and recorded in an ADR.
- **Release is gated**: the model card, the data ledger, and the eval suite must pass before the engine serves the model.

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `C1` | Capstone: TinyStories, owned end to end | capstone | 9 |
| `C2` | Capstone: the post-trained instruction follower (optional) | capstone | 10, optional |

Chapter files: `01-tinystories.md` (C1) and `02-post-trained.md` (C2), per course/DESIGN.md 3.3.

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `C1` | [Capstone: TinyStories, owned end to end](01-tinystories.md) | practice | 9 |
| 2 | `C2` | [Capstone 2: post-trained TinyStories model](02-post-trained.md) | practice | 10 |
<!-- /ol:chapters -->

## Going further

- Eldan and Li, [*TinyStories*](https://arxiv.org/abs/2305.07759); Hoffmann et al., [*Training Compute-Optimal Large Language Models*](https://arxiv.org/abs/2203.15556) (the scaling-law fit).
