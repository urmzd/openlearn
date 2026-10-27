# Distillation data, student rollouts, and deployment

Distillation transfers a teacher's behavior into a student with a different cost or deployment profile. The important choices are what the teacher supplies, which prefixes the student sees, what discrepancy is optimized, and how the final student is evaluated. The [KL implementation](04-distillation.md) covers the probability arithmetic; this part covers the recipe and its engineering.

> **Learning objectives**
> - Separate sequence distillation, distribution distillation, and feature matching.
> - Treat prefix distribution and KL direction as independent choices.
> - Measure compression using quality and total serving cost.

## What evidence can the teacher provide?

| Teacher access | Training signal | Engineering consequence |
|---|---|---|
| Generated text only | SFT on accepted teacher answers | Works with text APIs; records one answer rather than the full distribution |
| Token probabilities/logits | Cross entropy or KL on aligned token distributions | Large tensors; vocabulary alignment and exact normalization matter |
| Hidden representations | Feature or attention matching with projections if needed | Requires internal model access and compatible layer/shape mappings |
| Scores on student answers | Reward or preference signals, or sampled distribution objectives | Needs a clearly specified estimator; it is not automatically full-logit distillation |

For **sequence-level distillation**, generate candidate answers, filter or score them, and train on accepted examples. A single sampled target omits alternatives and can teach teacher errors. A deterministic decoding teacher and a stochastic teacher produce different datasets. Freeze prompts, generation parameters, teacher revision, provenance, and filtering decisions.

For **distribution distillation**, freeze the teacher and differentiate through the student. With teacher $p_T$ and student $q_T$, forward KL $D_{KL}(p_T\|q_T)$ weights by teacher mass; reverse KL weights by student mass. The [original distillation paper](https://arxiv.org/abs/1503.02531) introduces temperature-softened targets. Teacher entropy, student capacity, and the chosen discrepancy all affect behavior.

## Whose prefixes are used?

Offline text or teacher trajectories offer reusable targets. Student-generated prefixes expose the teacher to the states the student actually visits, including mistakes. This is **on-policy distillation** in the prefix-distribution sense. It does not prescribe forward or reverse KL; one can evaluate several discrepancies on those prefixes. [On-policy distillation](https://arxiv.org/html/2306.13649v3) explicitly studies this distribution-shift problem.

An illustrative mixed objective is

$$L=\alpha L_{\mathrm{CE}}(y,q)+(1-\alpha)T^2D_{KL}(p_T\|q_T).$$

Specify the prefix source, response mask, reduction denominator, temperature, and mixing coefficient. Both distributions use the same temperature in this formulation. The $T^2$ factor compensates for a high-temperature gradient-scale effect; it does not guarantee one learning rate works for every temperature or student.

When teacher and student vocabularies differ, tokenwise KL over vocabulary indices is undefined without a justified alignment. Use text targets or an explicitly designed cross-tokenizer method. A top-k log-probability API also omits probability mass: renormalizing its top-k entries creates a different distribution. Do not present it as exact full-vocabulary KL.

## Data and memory budgeting

With 100,000 sequences, 512 scored positions, 32,000 vocabulary entries, and two-byte logits, a full cached tensor would occupy about 3.28 TB before metadata and masks. Streaming teacher batches, recomputing logits, restricting positions, or storing sparse summaries trades storage against compute and approximation. Name the approximation and test its effect.

Student rollouts add generation cost, teacher scoring cost, and synchronization just as RL does. A frozen teacher need not reside alongside all learner optimizer state if scoring is a separate stage. Conversely, copying the teacher to every learner rank can dominate memory. Plan the pipeline by live tensors, not parameter counts alone.

## Validate the student after export

Evaluate base student, distilled student, and teacher on the same held-out tasks. Include errors the teacher makes, long contexts, uncommon formats, calibration where relevant, and robustness to student-generated imperfect prefixes. Teacher agreement is a training-related metric; actual task success is a separate outcome.

Then measure time to first token, inter-token latency, goodput, weight/cache memory, output length, and total task cost on the deployment runtime. A student that answers with twice as many tokens or escalates frequently may not deliver the expected saving. Conversion, adapter merge, and quantization each require new artifact and numerical checks.

## Key terms

- **Sequence distillation**: training on teacher-generated target sequences.
- **Distribution distillation**: training against a teacher's probability distribution.
- **Prefix distribution**: the histories on which teacher and student predictions are evaluated.

## Summary

Choose teacher access, prefix source, discrepancy, and deployment target separately. Evaluate task performance and end-to-end cost after every transformation.

## Review questions

1. Does student-generated data force reverse KL?

   <details><summary>Answer</summary>No. Prefix source and divergence direction are independent design choices.</details>
2. Can an API's top-five token probabilities establish exact full-vocabulary KL?

   <details><summary>Answer</summary>No. The omitted mass and its allocation are unknown. Any renormalization or tail approximation must be specified.</details>
