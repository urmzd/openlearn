<!-- ol:module L6.5 -->
# Fine-tuning heads and the linear-head export

## Overview

| | |
|---|---|
| **Module** | `L6.5` · build · Python · Pass 5 · 4 to 5 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/obj/heads.py`: `hidden_states`, `pool`, `SequenceClassifier`, `TokenClassifier`, `RewardHead`, `pairwise_reward_loss`, `lora_classifier`, `train_classifier`, `predict`, `save_classifier`, `load_classifier`, and the policy head: `fit_linear_head`, `head_probs`, `head_metrics`, `export_linear_head`, `load_linear_head`; and your own oracle tests in `python/tests/l6-5-heads/` |
| **Contract** | [`course/contracts/py/tinyllm/obj/heads.pyi`](../../../course/contracts/py/tinyllm/obj/heads.pyi) |
| **Tests** | `course/tests/L6.5/test_heads.py` (what they check: section 4); one learning test against the reference's calibrated bar; your tests are graded by mutation, threshold 0.80 with every required pitfall fault killed |
| **Needs** | `L6.1` GPT · `L6.2` `BertEncoder` · `L6.6` `inject_lora` · [`M07.7` IRLS, ROC-AUC, ECE](../../../math/07-probability-statistics/07-logistic-regression-roc-auc-calibration.md) · [`M10.3` AdamW](../../../math/10-optimization/03-adam-and-adamw.md) · `L0.1` · `L0.2` · `L0.3` · `L0.4` · `L0.6` · `M06.3` (or `--ref-deps`) |
| **Used by** | `L6.7` the zoo's classifier rows · later: `ethics.05` the usage-policy head, evaluated in Go by `gw.08`; `L12.3` reward model (optional) |
| **Milestone** | `MS-L6` (`finetune classify --lora r=8`: accuracy at the calibrated bar, `trainable_frac < 0.05`) |
| **Optional depth** | Devlin et al., BERT (2019), section 4; Ouyang et al., "Training language models to follow instructions with human feedback" (2022), section 3.5 (the reward model); Bradley and Terry, "Rank Analysis of Incomplete Block Designs" (1952); scikit-learn's `LogisticRegression` docs |

## Key Takeaways

- A head is small: pool one vector per sequence (`cls`, `mean` over **real** tokens, or the **last real** token) and apply one Linear (`test_hand_example_pooling_and_loss`, `test_last_pool_reads_the_last_real_token`).
- Padding must be invisible: the same row with or without padding gets the same logits (`test_bert_classifier_does_not_see_padding`).
- A reward head is trained on pairs: the loss is $-\log\sigma(r_{\text{chosen}} - r_{\text{rejected}})$ (`test_hand_example_reward_loss`, `test_reward_head_ranks_by_the_margin`).
- LoRA fine-tuning trains the adapters **and** the new head (`test_lora_classifier_trains_adapters_and_the_head`).
- The policy head is M07.7's logistic regression fitted on **unit** embeddings and exported as JSON that Go scores identically (`test_fit_linear_head_is_irls_on_unit_vectors`, `test_python_scores_equal_the_shared_fixture`, `test_export_validates_against_the_schema`).

## How to work this chapter

```bash
ol start L6.5              # stubs heads.py; prints your test path and rung (R5)
ol tests L6.5              # the course tests
# write your oracle tests in python/tests/l6-5-heads/, then:
ol check L6.5              # course tests, the learning test, and the mutation grade of your tests
ol mutate L6.5             # the full grade, cached by your test files' hash
ol diff  L6.5              # after passing: your code against the reference
```

---

## 1. Why now

Your backbones (`L6.1` GPT, `L6.2` BERT, `L6.3` ELECTRA's discriminator) turn tokens into vectors, but nothing in the system asks them a question yet. Three call sites need a head. The model zoo (`L6.7`) compares backbones by how well they classify the same sentences. The gateway's usage policy (`gw.08`, Pass 10) must refuse some prompts, and Python never serves HTTP (D9), so the classifier has to be something Go can evaluate: a linear head over the engine's own `/v1/embeddings` (D33). And post-training (`L12.3`, optional) needs a reward model. All three are a pooled vector and one Linear; what goes wrong is the plumbing around it: padding, pooling, which parameters train, and numbers that must survive a trip through JSON into another language.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $h$ | the backbone's hidden states | `float32[B, T, d]` |
| $m$ | the real-token mask (`attn_mask`), right padding | `bool[B, T]` |
| $v_b$ | the pooled vector of row $b$ | `float32[d]` |
| $W_c, b_c$ | the head's Linear (`classifier`) | `[C, d]`, `[C]` |
| $r$ | a reward, one per sequence | `float32[B]` |
| $e$ | an embedding from `/v1/embeddings` | `float64[d]` |
| $u = e / \lVert e \rVert$ | the unit embedding the policy head reads ($u = 0$ when $e = 0$) | `float64[d]` |
| $w, c$ | the logistic regression's weights and intercept (M07.7) | `[d]`, scalar |
| $\sigma(z) = 1 / (1 + e^{-z})$ | the logistic function | |

### 2.1 Pooling

Every head starts from $h$: an encoder is called as `backbone(ids, token_type_ids, attn_mask)` (`L6.2`), a decoder through `backbone.hidden(ids)` (`L6.1`; causal attention never reads the right padding, so it needs no mask). One vector per row:

| pool | $v_b$ | when |
|---|---|---|
| `cls` | $h_{b,0}$ | BERT and ELECTRA: position 0 is `[CLS]`, which attends to everything |
| `mean` | $\sum_t m_{bt} h_{bt} / \sum_t m_{bt}$ | any encoder; robust for short fine-tunes |
| `last` | $h_{b, t^*}$ with $t^*$ the last real position | decoders: only the last token has read the whole row |

The mean is over real tokens in both the sum and the count; `last` reads the last real token, not position $T - 1$, which for a short row is padding. A `SequenceClassifier` is then `classifier(dropout(pool(h)))` and its loss `L0.3`'s cross-entropy against one label per row.

### 2.2 Token and reward heads

A `TokenClassifier` applies the Linear at every position: logits `[B, T, C]`, and a loss over the real positions whose label is not `-100` (padding never counts, whatever label the batch gives it). A `RewardHead` maps the last real token to a scalar, $r = w^\top h_{b,t^*}$ (`score`, no bias: a constant shift would cancel in every comparison). It learns from preference pairs with the Bradley-Terry model, $P(\text{chosen} \succ \text{rejected}) = \sigma(r_c - r_r)$, whose negative log-likelihood is

$$L = -\log \sigma(r_c - r_r) = \mathrm{softplus}(-(r_c - r_r)),$$

computed as `L0.3`'s `bce_with_logits` of the margin against 1. Only the margin matters.

### 2.3 Fine-tuning with LoRA

`lora_classifier` calls `L6.6`'s `inject_lora` on the classifier, which freezes everything and adapts the backbone's attention queries and values (the Linears named `...q_proj` and `...v_proj`). Then it makes the head trainable again: the head is new, there is no pretrained value to protect, and a frozen random head cannot learn anything. This is PEFT's `modules_to_save`. `train_classifier` runs AdamW (`M10.3`) over the parameters that require grad, with batches drawn with replacement by `rng.below(n)`. Before saving, merge the adapters (`L6.6`), so the directory is a plain classifier that `load_classifier` rebuilds from `tl_backbone` (a `BertConfig` or `GPTConfig` as a dict) and `tl_head`.

### 2.4 The linear policy head (D33)

The gateway sees a prompt, asks the engine for its embedding $e$, and must decide "allow" or "refuse" with nothing but arithmetic. The head is

$$p = \mathrm{softmax}(W u + b), \qquad u = e / \lVert e \rVert,$$

stored as `formats/linear-head.schema.json`. Normalizing makes the score independent of the embedding's length, which varies with the prompt; the head must be **fitted** on unit vectors too, or it learns to read length. `fit_linear_head` fits `M07.7`'s `logistic_regression_fit` (IRLS: Newton steps, each a weighted least-squares solve) on the unit rows, giving $w$ and an intercept $c$, and stores two softmax rows with class 0 as the reference:

$$W = \begin{pmatrix} 0 \\ w^\top \end{pmatrix}, \quad b = (0, c), \quad p_1 = \frac{e^{w^\top u + c}}{1 + e^{w^\top u + c}} = \sigma(w^\top u + c),$$

exactly the fitted probability. The decision is $p_1 \ge$ `threshold`. `head_metrics` records accuracy, precision, and recall at that threshold, plus `M07.7`'s threshold-free ROC-AUC and expected calibration error. The export writes every number as the shortest decimal that reads back as the same float64, so Go's `strconv.ParseFloat` gets the same weights, and `course/fixtures/L6.5/linear_head.json` holds probe embeddings and the probabilities both languages must produce within 1e-6.

## 3. Worked example by hand

**Pooling.** One row of hidden states $(1, 2)$, $(3, 4)$, $(5, 6)$ with the third position padding ($m = (1, 1, 0)$): `cls` $= (1, 2)$; `mean` $= ((1 + 3)/2, (2 + 4)/2) = (2, 3)$; `last` $= (3, 4)$, the last real token, not $(5, 6)$. With an identity classifier and no bias, the mean-pooled logits are $(2, 3)$; for label 1 the cross-entropy is $-\log \frac{e^3}{e^2 + e^3} = \log(1 + e^{-1}) = 0.313262$.

**Policy head.** $e = (3, 4)$, so $\lVert e \rVert = 5$ and $u = (0.6, 0.8)$. With $W = \begin{pmatrix} 0 & 0 \\ 1 & 2 \end{pmatrix}$ and $b = (0, -1)$: logits $(0,\ 0.6 + 1.6 - 1) = (0, 1.2)$ and $p_{\text{unsafe}} = \sigma(1.2) = 0.768525$. The embeddings $(0.03, 0.04)$ and $(300, 400)$ have the same $u$, so the same score. The zero vector stays $u = 0$ and scores $\mathrm{softmax}(b) = (0.731059, 0.268941)$.

**Reward pair.** $r_c = 2$, $r_r = 0$: $L = \log(1 + e^{-2}) = 0.126928$; $(5, 3)$ gives the same loss.

These are `test_hand_example_pooling_and_loss`, `test_hand_example_linear_head_probs`, and `test_hand_example_reward_loss`.

## 4. The interface

```python
def pool(h: Tensor, attn_mask, how: str) -> Tensor: ...                       # [B, d]
class SequenceClassifier(Module):
    def __init__(self, backbone, d_model, n_classes, pool="cls", dropout=0.0, rng=None): ...
    def forward(self, ids, token_type_ids=None, attn_mask=None, labels=None) -> tuple[Tensor, Tensor | None]: ...
class TokenClassifier(Module): ...                                             # logits [B, T, C]
class RewardHead(Module): def forward(self, ids, attn_mask=None) -> Tensor: ...  # [B]
def pairwise_reward_loss(r_chosen: Tensor, r_rejected: Tensor) -> Tensor: ...
def lora_classifier(clf, r, alpha, target=None, init="default", rng=None) -> list[str]: ...
def train_classifier(clf, ids, labels, steps, batch_size, lr, rng, attn_mask=None, weight_decay=0.0) -> list[float]: ...
def predict(clf, ids, attn_mask=None, batch_size=64) -> NDArray: ...
def save_classifier(clf, dir, arch, tokenizer="file", labels=()) -> None: ...
def load_classifier(dir) -> tuple[SequenceClassifier, dict]: ...
def fit_linear_head(embeddings, labels, classes, l2, iters=50, threshold=0.5) -> dict: ...
def head_probs(head, embeddings) -> NDArray: ...                               # float64 [n, C]
def head_metrics(head, embeddings, labels) -> dict[str, float]: ...
def export_linear_head(head, embedding_model, path, metrics=None) -> None: ...
def load_linear_head(path) -> dict: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_pooling_and_loss` | unit | section 3: the three pools and the loss 0.313262 | you and the test agree on pooling |
| `test_hand_example_linear_head_probs` | unit | section 3: $p = \sigma(1.2)$ at three scales; the zero vector scores softmax($b$) | the gateway's formula |
| `test_hand_example_reward_loss` | unit | $\log(1 + e^{-2})$ for two pairs with margin 2 | the Bradley-Terry loss |
| `test_last_pool_reads_the_last_real_token` | boundary | a GPT backbone: `last` picks positions 6, 3, 1 of three padded rows | decoder classifiers and reward models |
| `test_mean_pool_ignores_padding` | property | huge values in padding change nothing; an all-padding row and an unknown pool raise | padded batches |
| `test_bert_classifier_does_not_see_padding` | property | a padded row and the same row alone get the same logits, whatever the padding ids | the encoder gets the mask |
| `test_token_classifier_skips_padding_and_ignored_labels` | unit | the loss equals the mean over real, labelled positions | token tasks |
| `test_reward_head_ranks_by_the_margin` | property | $r = w^\top h_{t^*}$; one step on the pair loss widens the margin | reward modelling (L12.3) |
| `test_lora_classifier_trains_adapters_and_the_head` | unit | exactly the q and v adapters plus the head train; a few steps run | the MS-L6 LoRA fine-tune |
| `test_fit_linear_head_is_irls_on_unit_vectors` | differential | $p_1$ equals M07.7's fit on unit rows; scale invariance; 3 classes refused | the head is the fitted regression |
| `test_python_scores_equal_the_shared_fixture` | golden | the probabilities stored for gw.08, to 1e-12 | Python and Go agree within 1e-6 |
| `test_export_validates_against_the_schema` | conformance | `formats/linear-head.schema.json`; weights read back bit for bit; bad heads refused | the gateway loads the file |
| `test_head_metrics_at_the_threshold` | unit | precision, recall, accuracy at the threshold; AUC 3.5/6 | the numbers the policy card reports |
| `test_save_load_classifier_roundtrip` | property | BERT and GPT classifiers reload and predict alike; adapters must be merged first | the zoo loads classifiers |
| `test_classifier_learns_sentiment` | learning | a tiny BERT, 150 AdamW steps on the synthetic SST-2 stand-in: validation accuracy at the reference bar | the whole path learns |
| `test_validation` | boundary | bad class counts, pools, shapes, and label counts raise | caller bugs fail loudly |

The learning test's data, `course/fixtures/small-corpora/sst2-2k.tsv`, is a synthetic stand-in for the SST-2 2k subset of the design (the real SST-2 has no clear license to commit): 2000 template sentences with negation ("is not dull") and contrast ("dull but the ending is wonderful"), so bag-of-words is not enough. Its bar is the reference's mean minus 3 standard deviations over 5 seeds (`course/fixtures/ref-thresholds.tsv`).

### Your graded tests (rung R5)

Your oracles: numpy pools and cross-entropy written out on a constant backbone (a Module that returns fixed hidden states), a padded and unpadded run of the same encoder, M07.7's own `logistic_regression_fit` and `logistic_predict_proba` on the unit rows, and the reward loss from `math.log1p`. Cover the three pools, the encoder mask, the token loss over real positions, the reward sign, the head on unit vectors, and which parameters `lora_classifier` leaves trainable. Import only the contract. `ol check L6.5` requires a mutation score of at least 0.80 with every required pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `last` reads position $T - 1$ | every short row is classified by a padding vector | `test_last_pool_reads_the_last_real_token`, `test_hand_example_pooling_and_loss` (mutant `s01`) |
| 2. padding in the mean, or an encoder called without the mask | predictions change with batch composition | `test_mean_pool_ignores_padding` (mutant `s02`), `test_bert_classifier_does_not_see_padding` (mutant `s03`) |
| 3. padding counted in a token loss | the loss rewards predicting padding labels | `test_token_classifier_skips_padding_and_ignored_labels` (mutant `s04`) |
| 4. the policy head fitted or scored on raw embeddings | long prompts score differently from short ones with the same content; Go and Python disagree | `test_hand_example_linear_head_probs`, `test_fit_linear_head_is_irls_on_unit_vectors` (mutant `s05`) |
| 5. the reward margin with the wrong sign | the reward model learns to prefer the rejected answers | `test_hand_example_reward_loss` (mutant `s06`) |
| 6. the new head frozen under LoRA | accuracy stays at chance whatever the adapters do | `test_lora_classifier_trains_adapters_and_the_head` (mutant `s07`) |
| 7. weights exported at float32 precision | Go reads other doubles; scores drift past 1e-6 | `test_export_validates_against_the_schema` (mutant `s09`) |
| 8. precision and recall at 0.5 instead of the threshold | the policy card reports a classifier the gateway never runs | `test_head_metrics_at_the_threshold` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L6.1` | `GPT.hidden` is the decoder backbone; `GPTConfig` rebuilds it from a directory |
| Back | `L6.2` | `BertEncoder` is the encoder backbone; `BertConfig` rebuilds it |
| Back | `L6.6` | `inject_lora` adapts the backbone; `LoRALinear` marks an unmerged model |
| Back | `M07.7` | `logistic_regression_fit`, `roc_auc`, `ece` |
| Back | `M10.3` | AdamW in `train_classifier` |
| Back | `L0.3` | `cross_entropy` and `bce_with_logits` |
| Back | `L0.4` | `Linear`, `Dropout`, `Module` |
| Back | `L0.1` | the Tensor and `no_grad` in `predict` |
| Back | `L0.2` | the ops of pooling |
| Back | `L0.6` | the classifier directory's safetensors |
| Back | `M06.3` | PCG32: the default init stream and the batch draws |
| Forward | `L6.7` | the zoo loads classifier directories (BERT and ELECTRA backbones) and reports accuracy |
| Forward | `ethics.05` | fits and exports the usage-policy head over the engine's embeddings (Pass 10) |
| Forward | `gw.08` | evaluates the exported head in Go: unit vector, dot products, softmax, threshold |

If you skip this module, `ol check L6.7` stops with `needs L6.5`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SequenceClassifier` | HF `BertForSequenceClassification`, `AutoModelForSequenceClassification` | BERT's pooler (a tanh layer over `[CLS]`), regression and multi-label problem types | `transformers/models/bert/modeling_bert.py` |
| `RewardHead`, `pairwise_reward_loss` | TRL `RewardTrainer` | margins per pair, centering rewards, the reward model behind PPO and best-of-n | `trl/trainer/reward_trainer.py` |
| `lora_classifier` | PEFT `TaskType.SEQ_CLS` | `modules_to_save=["classifier", "score"]` chosen per architecture | `peft/utils/constants.py` |
| the linear policy head | moderation endpoints, Llama Guard | a full classifier model with a taxonomy; here a single dot product over embeddings the engine already computes | OpenAI moderation docs; Inan et al., "Llama Guard" (2023) |
