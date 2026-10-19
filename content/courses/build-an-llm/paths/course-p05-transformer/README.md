# Pass 5: The Transformer

This is the pass where the course reaches the architecture behind every modern language model. It builds the 2017 transformer from its parts (scaled dot-product attention, masks, multi-head attention, positional encodings) and trains it to add three-digit numbers exactly, then shows why pre-LN trains where post-LN does not. It then applies one architecture to several pretraining objectives: GPT's next-token prediction, BERT's masked tokens, and ELECTRA's replaced-token detection, with LoRA fine-tuning and an evaluation harness that puts every model the course has trained on one report.

The last third of the pass covers what changed between 2017 and a Llama-family model: RMSNorm, gated MLPs, RoPE and context extension, grouped-query and multi-head latent attention, sliding windows and sinks, and mixture of experts. It ends with a Llama-family model that loads SmolLM2-135M exactly as Hugging Face saves it and matches Hugging Face's logits and greedy continuation token for token. From here on, the engine can serve real weights.

> **Learning objectives**
> - Derive scaled dot-product attention's forward and backward passes and explain each mask and positional encoding.
> - Explain why pre-LN trains without warmup where post-LN does not.
> - Explain the GPT, BERT, and ELECTRA objectives and how LoRA adapts a model through under 5% of its parameters.
> - Count a model's parameters, FLOPs, and KV-cache bytes.
> - Explain each component of the modern decoder block and how a published Llama checkpoint is loaded and verified.

## Before you begin

This pass needs [Pass 4](../course-p04-sequence-models/README.md), whose attention, beam search, and sequence metrics the transformer reuses.

## In this pass

The pass opens with the statistics used to compare models, then builds the transformer, its objectives, and the modern block:

1. [Definite Integrals: Trapezoid and Simpson](../../../mathematics-for-computing/02-calculus-1/04-definite-integrals-trapezoid-simpson.md). Afterwards you can explain the error orders of the trapezoid and Simpson rules.
2. [Hypothesis tests: paired and two-sample permutation, McNemar, Holm](../../../mathematics-for-computing/08-probability-statistics/06-hypothesis-tests.md). Afterwards you can explain how paired and two-sample permutation tests, McNemar, and Holm's correction decide significance.
3. [Logistic regression (IRLS), ROC-AUC, calibration (ECE)](../../../mathematics-for-computing/08-probability-statistics/08-logistic-regression-roc-auc-calibration.md). Afterwards you can explain how IRLS fits logistic regression and how ROC-AUC and ECE measure ranking and calibration.
4. [Probability problem set, part d: hypothesis tests, regression, queueing, balls into bins](../../../mathematics-for-computing/08-probability-statistics/12-problem-set-d.md). Afterwards you can work every problem in the set and explain each answer.
5. [Scaled dot-product attention (forward and backward)](../../06-transformer-2017/01-scaled-dot-product-attention.md). Afterwards you can derive scaled dot-product attention's forward and backward and explain the 1/sqrt(d) scale.
6. [Masks: causal, padding, sliding window, additive](../../06-transformer-2017/02-masks.md). Afterwards you can explain how causal, padding, and sliding-window masks combine as additive masks.
7. [Multi-head attention](../../06-transformer-2017/03-multi-head-attention.md). Afterwards you can explain how multi-head attention splits, attends, and merges heads.
8. [Positional encodings: sinusoidal, learned](../../06-transformer-2017/04-positional-encodings.md). Afterwards you can explain how sinusoidal and learned positional encodings inject position.
9. [Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing](../../06-transformer-2017/05-encoder-decoder-transformer.md). Afterwards you can explain why pre-LN trains where post-LN needs warmup, and how the Noam schedule and label smoothing are applied.
10. [GPT: the decoder-only language model](../../07-objectives/01-gpt-decoder-only.md). Afterwards you can explain the decoder-only GPT, its shifted causal LM loss, and how GPT-2 weights are mapped in.
11. [BERT: the bidirectional encoder and masked language modeling](../../07-objectives/02-bert-masked-lm.md). Afterwards you can explain the BERT encoder and the 80/10/10 MLM masking rule.
12. [ELECTRA: replaced-token detection](../../07-objectives/03-electra-replaced-token-detection.md). Afterwards you can explain how ELECTRA's generator and discriminator are trained for replaced-token detection.
13. [LoRA: low-rank adaptation with PiSSA initialization and merging](../../07-objectives/06-lora-pissa-and-merge.md). Afterwards you can explain how LoRA adds a low-rank update, how PiSSA initializes it from the SVD, and how it merges back.
14. [Fine-tuning heads and the linear-head export](../../07-objectives/05-fine-tuning-heads-and-the-linear-head-export.md). Afterwards you can explain how fine-tuning heads attach to the backbone and what the linear-head export writes.
15. [The evaluation harness and the model zoo](../../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md). Afterwards you can explain how the evaluation harness scores the models in the zoo.
16. [Counting: parameters, FLOPs, KV bytes, and memory plans](../../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md). Afterwards you can count a model's parameters, FLOPs, and KV-cache bytes by hand.
17. [Pre-LN and RMSNorm](../../08-modern-block/01-pre-ln-rmsnorm.md). Afterwards you can explain how RMSNorm differs from LayerNorm and why the modern block is pre-LN.
18. [Gated MLPs: SwiGLU and GeGLU](../../08-modern-block/02-gated-mlps.md). Afterwards you can explain how SwiGLU and GeGLU gate the MLP and how the hidden size is chosen.
19. [Rotary Position Embeddings](../../08-modern-block/03-rope.md). Afterwards you can explain how RoPE rotates query and key pairs in the half and interleaved layouts, and how partial rotary works.
20. [Context Extension: PI, NTK, YaRN, Llama 3 Scaling, and ALiBi](../../08-modern-block/04-context-extension.md). Afterwards you can explain how PI, NTK, YaRN, and Llama-3 scaling stretch RoPE to longer contexts.
21. [Grouped-Query Attention with a Cache Hook, Windows, and Sinks](../../08-modern-block/05-gqa-attention.md). Afterwards you can explain how MQA and GQA share key-value heads and how the cache hook, window, and learned sinks plug in.
22. [Multi-Head Latent Attention and Weight Absorption](../../08-modern-block/06-multi-head-latent-attention.md). Afterwards you can explain how multi-head latent attention compresses the KV cache and how weight absorption avoids decompressing it.
23. [Sliding Windows and Attention Sinks](../../08-modern-block/07-sliding-window-and-sinks.md). Afterwards you can explain how a sliding window bounds attention and why StreamingLLM keeps sink tokens.
24. [Mixture of Experts](../../08-modern-block/08-mixture-of-experts.md). Afterwards you can explain how MoE routes tokens to experts, dispatches them in sorted order, and balances load with the Switch aux loss or the aux-free bias.
25. [The Llama-Family Model: Config, Weight Loading, and Download](../../08-modern-block/09-llama-family-model.md). Afterwards you can trace how the Llama-family model reads an HF config, loads weights, and matches Hugging Face logits.
26. [Oracles, golden and differential tests, gradcheck as a test](../../../software-craftsmanship/03-testing-mentality/03-oracles-golden-and-gradcheck.md). Afterwards you can explain how oracles, golden and differential tests, and gradcheck verify numerical code.
27. [What the System Does After Pass 5](milestone.md). Afterwards you can explain why pre-LN trains where post-LN does not, and how the modern decoder loads SmolLM2-135M and matches Hugging Face.

## Going deeper

One optional section covers the third classic pretraining objective:

28. [T5: span corruption and relative position buckets](../../07-objectives/04-t5-span-corruption-and-relative-buckets.md). Afterwards you can explain T5's span corruption and how relative position buckets are computed.

The stages are also in this pass's manifest, [`path.tsv`](path.tsv). [Pass 6](../course-p06-inference-and-kernels/README.md) makes this model fast to run without changing a single output token.
