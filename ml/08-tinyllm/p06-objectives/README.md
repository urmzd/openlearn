# Part 6: Objectives and Adaptation

One architecture, many objectives. GPT's causal language modeling (with GPT-2 weight loading), BERT's masked LM, ELECTRA's replaced-token detection, and T5's span corruption (optional); fine-tuning heads for sequences, tokens, and rewards, with the linear-head export the gateway's usage policy uses; LoRA with PiSSA initialization; and the evaluation harness behind the model-zoo table every architecture in the course reports to.

**Course passes**: 5 (L6.1 to L6.7, milestone MS-L6, part of gate MS-P5)

**Before you start**: the just-in-time math `M01.4`, `M07.5`, `M07.7` and the solve part `S-M07d`; the transformer of [Part 5](../p05-transformer-2017/).

## Key ideas

- **The objective decides what the model learns**: predicting the next token gives a generator, reconstructing masked tokens gives an encoder, detecting replaced tokens trains on every position.
- **LoRA** freezes $W$ and learns $W + BA$ with rank $r$; PiSSA initializes $A$ and $B$ from the top singular vectors of $W$.
- **Evaluation is a measurement with error bars**: strided perplexity, log-likelihood multiple choice, and paired comparisons with confidence intervals (`L6.7`).

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L6.1` | GPT decoder-only, causal LM loss, GPT-2 weight loading | build | 5 |
| `L6.2` | BERT encoder and MLM masking | build | 5 |
| `L6.3` | ELECTRA replaced-token detection | build | 5 |
| `L6.4` | T5 span corruption, relative position buckets | build | 5, optional |
| `L6.5` | Fine-tuning heads (sequence, token, reward) and the linear-head export (D33) | build | 5 |
| `L6.6` | LoRA with PiSSA init and merge | build | 5 |
| `L6.7` | LM evaluation harness: strided perplexity, multiple choice, tasks with CIs, paired comparison, and the model zoo (D36) | build | 5 |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L6.1` | [GPT decoder-only, causal LM loss, GPT-2 weight loading](01-gpt-decoder-only.md) | build | 5 |
| 2 | `L6.2` | [BERT encoder and MLM masking](02-bert-masked-lm.md) | build | 5 |
| 3 | `L6.3` | [ELECTRA replaced-token detection](03-electra-replaced-token-detection.md) | build | 5 |
| 4 | `L6.4` | [T5 span corruption and relative position buckets](04-t5-span-corruption-and-relative-buckets.md) | side | 5 |
| 5 | `L6.5` | [Fine-tuning heads and the linear-head export](05-fine-tuning-heads-and-the-linear-head-export.md) | build | 5 |
| 6 | `L6.6` | [LoRA with PiSSA init and merge](06-lora-pissa-and-merge.md) | build | 5 |
| 7 | `L6.7` | [LM evaluation harness and the model zoo](07-lm-evaluation-harness-and-the-model-zoo.md) | build | 5 |
<!-- /ol:chapters -->

## Going further

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2); Devlin et al., [BERT](https://arxiv.org/abs/1810.04805); Clark et al., [ELECTRA](https://arxiv.org/abs/2003.10555); Raffel et al., [T5](https://arxiv.org/abs/1910.10683).
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685); Meng et al., [PiSSA](https://arxiv.org/abs/2404.02948); EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
