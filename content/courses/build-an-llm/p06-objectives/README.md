# Objectives and Adaptation

Part 5 built the Transformer from its parts. This chapter shows that one architecture supports many training objectives, and that the objective, more than the architecture, decides what a model learns. Predicting the next token gives a generator. Reconstructing masked tokens gives a bidirectional encoder. Detecting replaced tokens trains that encoder on every position instead of 15% of them. Dropping whole spans and writing them back gives a text-to-text model.

The chapter's arc runs from pretraining to use. It opens with GPT, which this book's later parts extend into a modern model, and loads real GPT-2 weights to prove the implementation matches. BERT and ELECTRA follow as the two encoder objectives, and T5's span corruption and relative position buckets come as an optional section. The second half turns pretrained backbones into tools: fine-tuning heads for sequences, tokens, and rewards, including the linear policy head the gateway evaluates in Go; LoRA with PiSSA initialization, so that a classifier fine-tune trains under 5% of the parameters; and finally the evaluation harness and model zoo, where every architecture in the book reports to one table with error bars.

> **Learning objectives**
> - Implement causal LM, masked LM, replaced-token detection, and span corruption objectives, and explain what each teaches a model.
> - Load Hugging Face GPT-2 and BERT checkpoints and verify them against the reference at the level of values, not shapes.
> - Build sequence, token, and reward heads that are invariant to padding, and export a linear head that another language scores identically.
> - Fine-tune with LoRA, initialize it with PiSSA, and merge it for serving.
> - Evaluate language models with strided perplexity, log-likelihood multiple choice, confidence intervals, and paired comparisons.

## Before you begin

You should have the Transformer of [Part 5](../p05-transformer-2017/README.md) in hand, in particular [masks](../p05-transformer-2017/02-masks.md), [multi-head attention](../p05-transformer-2017/03-multi-head-attention.md), [positional encodings](../p05-transformer-2017/04-positional-encodings.md), and the [encoder-decoder Transformer](../p05-transformer-2017/05-encoder-decoder-transformer.md). From *Mathematics for Computing*, the sections that matter most here are [definite integrals](../../mathematics-for-computing/01-calculus-1/04-definite-integrals-trapezoid-simpson.md), [hypothesis tests](../../mathematics-for-computing/07-probability-statistics/05-hypothesis-tests.md), and [logistic regression, ROC-AUC, and calibration](../../mathematics-for-computing/07-probability-statistics/07-logistic-regression-roc-auc-calibration.md); working through [the problem set on hypothesis tests and regression](../../mathematics-for-computing/07-probability-statistics/93-problem-set-d.md) is good preparation for the evaluation section. The LoRA section also relies on [SVD and low-rank approximation](../../mathematics-for-computing/03-linear-algebra/05-svd-low-rank-and-least-squares.md).

## In this chapter

1. [GPT: the decoder-only language model](01-gpt-decoder-only.md). The Transformer decoder without cross-attention, the causal LM loss with its one shift, and a loader that makes Hugging Face GPT-2 checkpoints produce identical logits.
2. [BERT: the bidirectional encoder and masked language modeling](02-bert-masked-lm.md). A post-LN encoder that reads both ways, trained by the 15% and 80/10/10 masking rule in a fixed, reproducible draw order.
3. [ELECTRA: replaced-token detection](03-electra-replaced-token-detection.md). A small generator fills the masks and a discriminator labels every token original or replaced, so every position trains.
4. [T5: span corruption and relative position buckets](04-t5-span-corruption-and-relative-buckets.md) (optional). Whole spans replaced by sentinels for an encoder-decoder to write back, and a learned bias indexed by logarithmic buckets of relative distance.
5. [Fine-tuning heads and the linear-head export](05-fine-tuning-heads-and-the-linear-head-export.md). Pooling that ignores padding, sequence, token, and reward heads, LoRA fine-tuning, and a logistic-regression policy head exported as JSON for the Go gateway.
6. [LoRA: low-rank adaptation with PiSSA initialization and merging](06-lora-pissa-and-merge.md). A frozen Linear plus a scaled low-rank update, zero or principal-singular initialization, PEFT-compatible adapter files, and merging for free serving.
7. [The evaluation harness and the model zoo](07-lm-evaluation-harness-and-the-model-zoo.md). Strided perplexity, multiple-choice scoring, an interval on every metric, paired comparison, and one report with a row for every model family.

## Further reading

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2); Devlin et al., [BERT](https://arxiv.org/abs/1810.04805); Clark et al., [ELECTRA](https://arxiv.org/abs/2003.10555); Raffel et al., [T5](https://arxiv.org/abs/1910.10683).
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685); Meng et al., [PiSSA](https://arxiv.org/abs/2404.02948); EleutherAI, [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
