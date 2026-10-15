<!-- ol:module L6.3 -->
# ELECTRA replaced-token detection

## Overview

| | |
|---|---|
| **Module** | `L6.3` · build · Python · Pass 5 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/obj/electra.py`: `ElectraDiscriminatorHead`, `ElectraDiscriminator`, `ELECTRA`, `replace_tokens`, `rtd_loss`, `electra_step`, `rtd_accuracy`, `save_electra`, `load_electra`; and your own oracle tests in `python/tests/l6-3-electra/` |
| **Contract** | [`course/contracts/py/tinyllm/obj/electra.pyi`](../../../../course/contracts/py/tinyllm/obj/electra.pyi) |
| **Tests** | `course/tests/L6.3/test_electra.py` (what they check: section 4); the oracle is Hugging Face's `ElectraForPreTraining`; your tests are graded by mutation, threshold 0.80 with every required pitfall fault killed |
| **Needs** | `L6.2` `BertEncoder`, `BertForMLM`, `mlm_mask` · [`M07.1` `sample_categorical`](../../../tracks/math/07-probability-statistics/01-categorical-sampling.md) · `L0.3` `bce_with_logits` · `L0.1` · `L0.2` · `L0.4` · `L0.6` safetensors · `M06.3` PCG32 · `M07.3` `normal_init` (or `--ref-deps`) |
| **Used by** | `L6.7` the zoo's ELECTRA row (replaced-token detection accuracy) · the discriminator body is the alternative classification backbone of `L6.5` (`tl_arch = "electra"`) |
| **Milestone** | `MS-L6` (`train electra`, then a LoRA classifier over its discriminator) |
| **Optional depth** | Clark, Luong, Le, and Manning, "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (ICLR 2020), sections 2, 3.2, and appendix A; Goodfellow et al., "Generative Adversarial Nets" (2014), for what ELECTRA is not |

## Key Takeaways

- The discriminator learns from **every** real token, not only the 15% BERT masks: each one is labelled original or replaced (`test_electra_step_is_its_pieces`).
- A sampled token equal to the original is labelled **original** (`test_a_lucky_sample_is_original`, `test_hand_example_replaced_token_labels`).
- The sample is data: the generator trains on its MLM loss only, and the total is $L_G + \lambda L_D$ with $\lambda = 50$ (`test_no_gradient_through_the_sample`, `test_lambda_weights_the_discriminator`).
- The discriminator is BERT's encoder plus a dense, GELU, dense head, and its loss averages over real tokens only; both match Hugging Face (`test_discriminator_matches_hf`, `test_rtd_loss_ignores_padding`).
- Generator and discriminator share one set of embeddings (`test_tied_embeddings_are_one_tensor`).

## How to work this chapter

```bash
ol start L6.3              # stubs electra.py; prints your test path and rung (R5)
ol tests L6.3              # the course tests
# write your oracle tests in python/tests/l6-3-electra/, then:
ol check L6.3              # course tests and the mutation grade of your tests
ol mutate L6.3             # the full grade, cached by your test files' hash
ol diff  L6.3              # after passing: your code against the reference
```

---

## 1. Why now

Your BERT (`L6.2`) learns from a masked-LM loss on 15% of the tokens of each batch; the other 85% cost a full forward and backward pass and teach nothing. On the small corpus and compute budget of this course that waste shows: a BERT pretrained for the `MS-L6` step budget is a weak backbone for the sentiment classifier of `L6.5`. ELECTRA turns every position into a training signal with the same encoder: a small generator fills the masked positions with plausible guesses, and the encoder, now a discriminator, says for every token whether it was replaced. The zoo (`L6.7`) puts the two backbones side by side on the same classification task, which is the comparison the paper is about.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | the batch of token ids | `int64[B, T]` |
| $m$ | the real-token mask (`attn_mask`), 1 = real, 0 = padding | `bool[B, T]` |
| $\tilde x$, $\ell$ | the masked inputs and MLM labels from `mlm_mask` (`-100` = not picked) | `int64[B, T]` |
| $G$ | the generator, a small `BertForMLM` | Module |
| $D$ | the discriminator, `BertEncoder` plus a token head | Module |
| $p_G(\cdot \mid \tilde x, t)$ | the generator's distribution at position $t$ | `float64[V]` |
| $\hat x$ | the corrupted ids (`corrupt`) | `int64[B, T]` |
| $y_t = [\hat x_t \ne x_t]$ | the discriminator's label (`is_replaced`) | `float32[B, T]` |
| $L_G$, $L_D$ | MLM cross-entropy, replaced-token BCE | scalars |
| $\lambda$ | the discriminator's loss weight, 50 | `float` |

### 2.1 One step

1. Mask like BERT: $(\tilde x, \ell) = $ `mlm_mask(x, special` $\lor \lnot m$ `, mask_id, V, p, rng)`. Padding counts as special, so it is never picked.
2. Run the generator on $\tilde x$: logits and $L_G$, the mean cross-entropy over the picked positions.
3. At each picked position, in C order, draw one uniform and sample $\hat x_t \sim p_G(\cdot \mid \tilde x, t)$ with `sample_categorical` (`M07.1`, inverse CDF in float64); elsewhere $\hat x_t = x_t$.
4. Label every token: $y_t = 1$ when $\hat x_t \ne x_t$. Run the discriminator on $\hat x$ and take $L_D$, the mean binary cross-entropy over the real tokens.
5. $L = L_G + \lambda L_D$.

The random draws are `mlm_mask`'s, then one uniform per picked position, all from one generator. Sampling is not differentiable, and the paper does not try to make it so (an adversarial generator trained by reinforcement learning did worse): the corrupted ids enter the discriminator as plain data, so $L_D$ sends no gradient into $G$. Each network learns from its own loss, and one backward pass of $L$ does both.

### 2.2 The discriminator's labels

The label asks "is this token different from the original?", not "was this position masked?". A good generator often samples the original token back, especially for function words; those positions are labelled original. Labelling every masked position as replaced teaches the discriminator to flag tokens that never changed, and its accuracy then measures the masking rate, not the text.

### 2.3 The discriminator

The body is `L6.2`'s `BertEncoder`, unchanged, under the name `electra`. The head is Hugging Face's `ElectraDiscriminatorPredictions`: `dense` ($d \to d$), the exact GELU, `dense_prediction` ($d \to 1$), and the last axis dropped, so the output is one logit per token, positive meaning "replaced". The loss is `L0.3`'s `bce_with_logits` over the real tokens only, the mean over them: padding contributes neither to the sum nor to the count. With these names, a Hugging Face ELECTRA checkpoint loads through `L6.2`'s `hf_bert_encoder_sd` after dropping its `electra.` prefix, and the golden test does exactly that.

### 2.4 Sharing embeddings

ELECTRA ties the generator's token, position, and type embeddings to the discriminator's: one Tensor under two names. The generator is otherwise smaller (fewer layers here; the paper also narrows it), which matters: a generator as strong as the discriminator produces replacements too hard to detect. In the `ELECTRA` module the discriminator registers first, so the shared Tensors are listed once, under `discriminator.`, and both losses add their gradients into them.

The model directory holds both networks, the tie, the mask id, and the special ids, so the zoo can corrupt held-out text the same way and report replaced-token detection accuracy (`rtd_accuracy`): over the real tokens, how often the discriminator's call (logit $> 0$) matches $y$.

## 3. Worked example by hand

Four tokens $x = (5, 6, 7, 8)$; `mlm_mask` picked positions 1 and 3, so $\ell = (-100, 6, -100, 8)$. The generator's distributions at the picked positions:

| position | $p_G$ | uniform | cumulative sum crosses it at | $\hat x_t$ | $y_t$ |
|---|---|---|---|---|---|
| 1 | 0.75 on token 6, 0.25 on token 9 | 0.1 | token 6 (0.75) | 6 | 0 (the original) |
| 3 | 0.5 on token 2, 0.5 on token 8 | 0.3 | token 2 (0.5) | 2 | 1 |

So $\hat x = (5, 6, 7, 2)$ and $y = (0, 0, 0, 1)$. Position 1 was masked, but the sample is the original token: not replaced.

With all discriminator logits 0, each token's BCE is $\ln 2$ and $L_D = \ln 2 = 0.693147$. With logits $(-2, -2, -2, 2)$ every call is right and each token pays $\ln(1 + e^{-2}) = 0.126928$, the mean too.

This is `test_hand_example_replaced_token_labels`.

## 4. The interface

```python
class ElectraDiscriminator(Module):            # electra.* (BertEncoder) + discriminator_predictions.*
    def __init__(self, cfg: BertConfig, rng=None) -> None: ...
    def forward(self, ids, token_type_ids=None, attn_mask=None) -> Tensor: ...    # [B, T] logits
class ELECTRA(Module):                         # discriminator.*, generator.* (tied embeddings)
    def __init__(self, gen_cfg: BertConfig, disc_cfg: BertConfig, rng=None, tie_embeddings=True) -> None: ...
def replace_tokens(ids, labels, gen_logits, rng, ignore_index=-100) -> tuple[NDArray, NDArray]: ...
def rtd_loss(disc_logits: Tensor, is_replaced, attn_mask=None) -> Tensor: ...
def electra_step(gen, disc, ids, rng, lam=50.0, *, mask_id, vocab=None, special_mask=None,
                 token_type_ids=None, attn_mask=None, p=0.15) -> dict: ...
def rtd_accuracy(model: ELECTRA, ids, rng, *, mask_id, ...) -> NDArray: ...
def save_electra(model, dir, mask_id, special_ids=(), tokenizer="file") -> None: ...
def load_electra(dir) -> tuple[ELECTRA, dict]: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_replaced_token_labels` | unit | section 3: $\hat x$, $y$, two uniforms used, both loss values | you and the test agree on the labels |
| `test_a_lucky_sample_is_original` | boundary | a certain generator replaces nothing | the label is "different", not "masked" |
| `test_replace_draws_one_uniform_per_masked_position` | unit | three picked positions, three uniforms, C order | the same seed gives the same corruption everywhere |
| `test_discriminator_matches_hf` | golden | logits, loss, and gradients of HF's `ElectraForPreTraining` | pretrained ELECTRA checkpoints load and score alike |
| `test_rtd_loss_ignores_padding` | property | padded logits and labels change nothing | padded batches train the same as unpadded ones |
| `test_electra_step_is_its_pieces` | differential | the step equals mask, generate, sample, discriminate, recomputed from a copied rng | the order of the five steps |
| `test_padding_is_never_masked_or_scored` | property | padding and specials are never picked or replaced | no learning from padding |
| `test_no_gradient_through_the_sample` | property | generator gradients of $L$ equal those of $L_G$ alone | each network learns from its own loss |
| `test_lambda_weights_the_discriminator` | unit | default $\lambda = 50$; $L = L_G + \lambda L_D$ for three values | the paper's balance of the two losses |
| `test_tied_embeddings_are_one_tensor` | property | one Tensor, listed once under `discriminator.`; mismatched configs raise | shared embeddings, saved once |
| `test_save_load_roundtrip` | property | the directory restores both networks, the tie, the mask id, and the specials | the zoo loads it |
| `test_rtd_accuracy_definition` | unit | an "always original" discriminator scores the fraction not replaced | the zoo's ELECTRA metric |

### Your graded tests (rung R5)

Your oracles: hand-made generator logits whose samples you know (a certain token, a uniform row and chosen uniforms), the BCE written out with `math.log1p`, and the step recomputed from its pieces with a deep copy of the generator. Cover the lucky sample, one draw per picked position, the head's formula (dense, exact GELU, dense) recomputed in numpy, the real-token mean, padding never picked, the discriminator reading the corrupted ids, and the generator's gradients coming from $L_G$ only. Import only the contract (`tinyllm.obj.electra`, `tinyllm.obj.bert`). `ol check L6.3` requires a mutation score of at least 0.80 with every required pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. every masked position labelled replaced | the discriminator learns where the masks were; its accuracy tracks the masking rate | `test_a_lucky_sample_is_original`, `test_hand_example_replaced_token_labels` (mutant `s01`) |
| 2. a uniform drawn at every position | a seed gives another corruption than the spec, so Python and a replay disagree | `test_replace_draws_one_uniform_per_masked_position` (mutant `s02`) |
| 3. a head without the GELU | logits differ from HF; pretrained checkpoints score differently | `test_discriminator_matches_hf` (mutant `s03`) |
| 4. padding in the loss or in the masking | short sequences in a batch weigh less; padding gets replaced | `test_rtd_loss_ignores_padding` (mutant `s04`), `test_padding_is_never_masked_or_scored` (mutant `s05`) |
| 5. the discriminator reading the masked inputs | it sees `[MASK]` where the replacement should be: the task becomes "find the masks" | `test_electra_step_is_its_pieces` (mutant `s06`) |
| 6. $\lambda$ on the wrong loss | the generator's MLM loss dominates and its gradients grow 50 times | `test_lambda_weights_the_discriminator`, `test_no_gradient_through_the_sample` (mutant `s07`) |
| 7. untied embeddings | two embedding tables, the generator's trained only by $L_G$ | `test_tied_embeddings_are_one_tensor` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L6.2` | `mlm_mask` picks positions; `BertForMLM` is the generator and `BertEncoder` the discriminator's body |
| Back | `M07.1` | `sample_categorical` draws the replacements |
| Back | `L0.3` | `bce_with_logits` is the replaced-token loss |
| Back | `L0.1` | the Tensor and `no_grad` (the zoo scores without a graph) |
| Back | `L0.2` | `gelu` and `reshape` in the head |
| Back | `L0.4` | the head's `Linear` layers and the `Module` registration order |
| Back | `L0.6` | the model directory's safetensors |
| Back | `M06.3` | PCG32: the default init stream and every draw of the step |
| Back | `M07.3` | `normal_init` for the head's weights |
| Forward | `L6.7` | the zoo loads ELECTRA directories and reports replaced-token detection accuracy next to BERT's masked-token accuracy |

If you skip this module, `ol check L6.7` stops with `needs L6.3`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ElectraDiscriminator` | HF `ElectraForPreTraining` | `embeddings_project` when the embedding size differs from the hidden size | `transformers/models/electra/modeling_electra.py` |
| `electra_step` | the original ELECTRA pretraining code | a generator a third to a quarter of the discriminator's width; gumbel-noise sampling; masking 85% [MASK], 15% kept | `google-research/electra`, `pretrain/pretrain_helpers.py` |
| replaced-token detection | DeBERTaV3 | gradient-disentangled embedding sharing: the generator's gradients stop at the shared embeddings | He, Gao, and Chen, "DeBERTaV3" (2021) |
| ELECTRA as a backbone | ELECTRA-small fine-tuned on GLUE | SST-2 and the other GLUE tasks with a classification head | `L6.5`; the paper's table 1 |
