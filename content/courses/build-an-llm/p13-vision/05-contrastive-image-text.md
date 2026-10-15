<!-- ol:module L13.5 -->
# Contrastive image-text: CLIP and SigLIP

## Overview

| | |
|---|---|
| **Module** | `L13.5` · build · Python · Pass 12 · 5 to 7 h |
| **You build** | `python/tinyllm/vision/contrastive.py`: `clip_loss`, `siglip_loss`, `TextEncoder`, `DualEncoder` (`encode_image`, `encode_text`, `loss`), `zero_shot_classifier`, `zero_shot_predict` |
| **Contract** | [`course/contracts/py/tinyllm/vision/contrastive.pyi`](../../../../course/contracts/py/tinyllm/vision/contrastive.pyi) |
| **Tests** | `course/tests/L13.5/test_contrastive.py` (what they check: section 4), golden values in `course/fixtures/L13.5/contrastive_hf.npz` from `course/oracle/L13.5/contrastive_hf.py` (HF `CLIPModel` and `SiglipModel` losses and autograd gradients on fixture embeddings, transformers 5.19.0, float64) |
| **Needs** | `L13.6`; `L13.3` `ViT`, `VisionBlock`, `ViTConfig` · `L0.4` `Embedding`, `LayerNorm`, `Linear`, `ModuleList`, `Module` · `L0.2` `matmul`, `transpose`, `log_softmax`, `exp`, `sum`, `mean`, `reshape` · `L0.1` `Tensor`, `from_op` · `M06.3` `PCG32` (or `--ref-deps`). Reading: `L5.3` attention, `M11.1` softmax with a temperature, `L0.3` cross-entropy |
| **Used by** | `ethics.08`; later `data.10` (the CLIP-score filter for image-text pairs) · `ethics.07` (the zero-shot disparity probe); neither is registered yet |
| **Milestone** | `MS-L13` step 2 (a tiny dual encoder learns zero-shot classification of 24 color-shape classes) |
| **Optional depth** | Radford et al., "Learning transferable visual models from natural language supervision" (2021), sections 2.3 and 3.1; Zhai et al., "Sigmoid loss for language image pre-training" (2023), sections 3 and 4; van den Oord et al., "Representation learning with contrastive predictive coding" (2018), for InfoNCE as a bound on mutual information |

## Key Takeaways

- CLIP's loss is two cross-entropies over one $B \times B$ matrix of scaled cosines, texts choosing images along the rows and images choosing texts along the columns, so it is symmetric in the two modalities and equals Hugging Face's to $10^{-8}$ (`test_clip_loss_is_symmetric_in_modalities`, `test_clip_loss_matches_hf`).
- SigLIP asks $B^2$ independent yes-or-no questions and divides by $B$; it needs no softmax over the batch, so it is defined, with a gradient, for a single pair (`test_siglip_loss_matches_hf`, `test_siglip_defined_at_batch_one`).
- The temperature is learned as `logit_scale` $= \ln s$ and multiplies the cosines. On a batch the model already separates, a larger $s$ always lowers the loss, so CLIP caps $s$ at 100 (`test_temperature_gradient_sign`, `test_clip_temperature_is_clamped_at_100`).
- $\log \sigma(x)$ must be computed as $\min(x, 0) - \ln(1 + e^{-|x|})$; the formula as written returns $-\infty$ at $x = -2000$ (`test_siglip_loss_is_stable_at_large_logits`).
- A classifier can be written in words: one row per class, the renormalized mean of its prompt embeddings. After 600 steps on captions alone the tiny model labels held-out shapes at about 0.89 accuracy against a chance of $1/24$ (`test_zero_shot_classifier_is_a_prompt_ensemble`, `test_dual_encoder_learns_zero_shot_shapes`).

## How to work this chapter

```bash
ol start L13.5              # stubs python/tinyllm/vision/contrastive.py into your repo
ol tests L13.5              # read the test catalog first: rung R0, you write no tests here
ol check L13.5              # exit code is the verdict
ol check L13.5 --ref-deps   # only if one of your Needs modules is not passing yet
ol diff  L13.5              # after passing: your code against the reference
```

---

## 1. Why now

`L13.3` gave you a vision tower that turns pixels into vectors, but nothing yet says what those vectors should mean. Every classifier so far (`L13.1`'s LeNet, `L6.5`'s heads) needed a fixed label set and a labelled example of each class. Captions are free supervision with an open vocabulary: 400 million image-text pairs trained CLIP, and its descendant SigLIP is the vision tower inside SmolVLM, the model `L13.7` loads. This module builds the objective that makes an image vector and a text vector comparable, and the two things that fall out of it: a similarity score for any image-text pair (the filter `data.10` runs over a scraped corpus) and a classifier for classes named only at test time (what `ethics.07` probes for disparities).

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $B$, $d$ | pairs in the batch, embedding width | int |
| $u_i$, $v_j$ | L2-normalized text $i$ and image $j$ | `float[d]`, length 1 |
| $C_{ij} = u_i \cdot v_j$ | cosine matrix, texts on rows | `float[B, B]`, in $[-1, 1]$ |
| $s = e^{\tau}$ | scale; $\tau$ is `logit_scale` (CLIP) or `t` (SigLIP) | scalar, $s > 0$ |
| $b$ | SigLIP bias | scalar |
| $\ell_{ij}$ | logit of pair $(i, j)$ | `float[B, B]` |
| $z_{ij}$ | $+1$ when $i = j$, else $-1$ | `float[B, B]` |
| $\sigma(x)$ | $1 / (1 + e^{-x})$ | scalar |

### 2.1 One matrix holds every comparison

Row $i$ of the images goes with row $i$ of the texts; every other combination is a pair that does not match. Normalize each row to length 1 and take all dot products at once: $C = U V^\top$. The diagonal holds the $B$ matching pairs and the $B^2 - B$ off-diagonal entries are negatives that cost nothing to collect. Both losses normalize inside, so they score directions and ignore lengths.

### 2.2 CLIP: InfoNCE as two cross-entropies

Treat "which image goes with text $i$?" as classification over the $B$ images in the batch, with logits $\ell_{ij} = s\,C_{ij}$:
$$p(j \mid i) = \frac{e^{s C_{ij}}}{\sum_k e^{s C_{ik}}}, \qquad \mathcal{L}_{t \to v} = -\frac{1}{B} \sum_i \ln p(i \mid i).$$
That is `L0.3`'s cross-entropy of row $i$ against class $i$. The same matrix read by columns asks "which text goes with image $j$?" and gives $\mathcal{L}_{v \to t}$, the cross-entropy of $\ell^\top$ against the diagonal. The loss is the mean of the two,
$$\mathcal{L}_\text{CLIP} = \tfrac{1}{2}\left(\mathcal{L}_{t \to v} + \mathcal{L}_{v \to t}\right),$$
so swapping the modalities transposes $\ell$ and changes nothing. A model that cannot tell the pairs apart has $p = 1/B$ and loss $\ln B$; van den Oord et al. show $\ln B - \mathcal{L}$ is a lower bound on the mutual information between image and text (`M11.1`), which is why larger batches help. Differentiating a softmax cross-entropy gives $\partial \mathcal{L} / \partial \ell = (P^\text{row} + P^\text{col} - 2I) / 2B$: each wrong pair is pushed apart in proportion to the probability it stole. At $B = 1$ both softmaxes have one class, so the loss and every gradient are 0.

### 2.3 SigLIP: a sigmoid per pair

The softmax couples a whole row: no pair can be scored without all the others. SigLIP instead models every pair as its own Bernoulli variable, "is this a match?", with logit $\ell_{ij} = s\,C_{ij} + b$ and $p(\text{match}) = \sigma(\ell_{ij})$. Since $1 - \sigma(x) = \sigma(-x)$, the log-likelihood of the right answer is $\ln \sigma(z_{ij} \ell_{ij})$ for both kinds of pair, and
$$\mathcal{L}_\text{SigLIP} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{j=1}^{B} \ln \sigma(z_{ij}\,\ell_{ij}).$$
The sum is over $B^2$ terms but the divisor is $B$, as in `SiglipModel`. With one pair there is still a question to answer, so the loss is $-\ln \sigma(sC_{11} + b)$ and its gradient is not zero.

The bias exists because the questions are unbalanced: $B$ say yes and $B^2 - B$ say no. At $b = 0$ a fresh model has $\ell \approx 0$, every negative costs $\ln 2$, and the first steps do nothing but push all logits down. Starting at $b = -10$ makes the model's prior $\sigma(-10) = 4.5 \cdot 10^{-5}$, about one match in 22000, the right order for the batch sizes it was trained with; each negative then costs $4.5 \cdot 10^{-5}$ and the positives carry the gradient. The published init is $\tau = \ln 10$, $b = -10$.

Compute $\ln \sigma(x) = \min(x, 0) - \ln(1 + e^{-|x|})$, never $\ln(1 / (1 + e^{-x}))$: the second overflows once $x < -709$ in float64. Its derivative is $\sigma(-x)$, the one backward this module adds to the op library (`from_op`).

### 2.4 Why the temperature is learned, and why it is clamped

Cosines live in $[-1, 1]$. With $s = 1$ even a perfect model (matching pair at $1$, all others at $-1$) has $p(i \mid i) = e / (e + (B - 1)e^{-1})$, so at $B = 32768$ the loss cannot fall below $\ln(1 + 32767 e^{-2}) = 8.4$. The logits need a scale, and how large depends on how separated the embeddings are, so it is a parameter. Learning $\tau = \ln s$ keeps $s$ positive and makes each step a multiplicative change; CLIP starts at $s = 1/0.07$.

By the chain rule $\partial \mathcal{L} / \partial \tau = s \sum_{ij} C_{ij}\, \partial \mathcal{L} / \partial \ell_{ij}$. When every matching pair already has the largest cosine in its row and column, this is negative: sharpening the softmax moves probability onto the right answer. It stays negative however large $s$ gets, because the loss goes to 0 only as $s \to \infty$. Nothing in the objective stops the growth, and a huge $s$ turns the softmax into a hard max that learns only from the single hardest negative and overflows in half precision. So CLIP caps the scale: $s = \min(e^{\tau}, 100)$. In this contract, when $e^{\tau} > 100$ the loss is the loss at exactly 100 and `logit_scale` gets no gradient. Dividing by $e^{\tau}$ where the loss multiplies flips the sign of this gradient: the model then cools itself whenever it is right.

### 2.5 The dual encoder

Two towers that never see each other's input. The image tower is `L13.3`'s `ViT`; its pooled output is the CLS token when the tower has one (CLIP) and the mean over patches otherwise (SigLIP). The text tower stacks the same pre-LN `VisionBlock`s over token plus learned position embeddings and pools by the mean over real tokens: id 0 is padding, hidden as an attention key and left out of the mean, so a caption's embedding does not depend on its batch. One bias-free `Linear` per tower projects into the shared $d$-dimensional space, and both encoders return unit rows.

### 2.6 Zero-shot classification

For class names $c_1, \dots, c_K$ and prompt templates $T_1, \dots, T_M$ such as `"a photo of a {}"`, embed every filled template and average:
$$W_k = \frac{\bar{u}_k}{\lVert \bar{u}_k \rVert}, \qquad \bar{u}_k = \frac{1}{M} \sum_{m} u(T_m(c_k)), \qquad \hat{y}(x) = \arg\max_k\; v(x) \cdot W_k.$$
$W$ is the weight matrix of a linear classifier that the text tower wrote; no image of any class was labelled. The mean of unit vectors is shorter than 1, so it is normalized again to keep the scores cosines.

## 3. Worked example by hand

Two pairs in two dimensions: images $v_1 = (1, 0)$, $v_2 = (0, 1)$ and texts $u_1 = (1, 0)$, $u_2 = (0.6, 0.8)$, all of length 1. Then $C = \begin{pmatrix} 1 & 0 \\ 0.6 & 0.8 \end{pmatrix}$.

CLIP with $\tau = \ln 5$: logits $\begin{pmatrix} 5 & 0 \\ 3 & 4 \end{pmatrix}$. A two-class cross-entropy is $\ln(1 + e^{-\text{margin}})$. Text rows: margins $5 - 0$ and $4 - 3$ give $\ln(1 + e^{-5}) = 0.006715$ and $\ln(1 + e^{-1}) = 0.313262$, mean $0.159988$. Image columns: margins $5 - 3$ and $4 - 0$ give $0.126928$ and $0.018150$, mean $0.072539$. Loss $= (0.159988 + 0.072539)/2 = 0.116264$.

SigLIP with $\tau = \ln 5$, $b = -3$: logits $\begin{pmatrix} 2 & -3 \\ 0 & 1 \end{pmatrix}$; multiplied by the signs $\begin{pmatrix} + & - \\ - & + \end{pmatrix}$ they are $2, 3, 0, 1$. The four costs $\ln(1 + e^{-x})$ are $0.126928$, $0.048587$, $0.693147$, $0.313262$, which sum to $1.181924$; divided by $B = 2$ the loss is $0.590962$. The pair $(u_2, v_1)$ with cosine $0.6$ is the expensive one: a negative the model has not yet pushed away. `test_hand_example_two_pairs` checks both numbers, handing the rows over at lengths 2, 0.5, 3, and 5.

## 4. The interface

```python
def clip_loss(img_emb: Tensor, txt_emb: Tensor, logit_scale: Tensor) -> Tensor: ...   # [B, d], [B, d], [1]
def siglip_loss(img_emb: Tensor, txt_emb: Tensor, t: Tensor, b: Tensor) -> Tensor: ...
class TextEncoder(Module):                       # ids [N, T] -> [N, T, d]; id 0 is padding
    def __init__(self, vocab, d, n_layers, n_heads, d_ff, max_len, rng=None): ...
class DualEncoder(Module):
    def __init__(self, vision: ViT, text: Module, d_embed, loss, t_init, b_init=-10.0, rng=None): ...
    def encode_image(self, pixel_values: Tensor) -> Tensor: ...     # [N, d_embed], unit rows
    def encode_text(self, ids: NDArray) -> Tensor: ...              # [N, d_embed], unit rows
    def loss(self, pixel_values: Tensor, ids: NDArray) -> Tensor: ...
def zero_shot_classifier(model, tok, class_names, templates) -> NDArray: ...   # [n_classes, d_embed]
def zero_shot_predict(model, pixel_values: Tensor, W: NDArray) -> NDArray: ...  # int64 [N]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_two_pairs` | unit, smoke | section 3's two losses, on unnormalized rows | you and the tests agree on both definitions |
| `test_clip_loss_matches_hf` | golden | `CLIPModel`'s loss and three gradients at $B = 4$; zero at $B = 1$ | fine-tuning a published CLIP |
| `test_siglip_loss_matches_hf` | golden | `SiglipModel`'s loss and four gradients at $B = 5, 3, 1$ | SigLIP is SmolVLM's vision tower (`L13.7`) |
| `test_gradcheck_clip_loss` | gradcheck | backward to both embeddings and `logit_scale` | the temperature trains |
| `test_gradcheck_siglip_loss` | gradcheck | backward through $\ln \sigma$ to embeddings, $t$, $b$ | the one new op is right |
| `test_clip_loss_is_symmetric_in_modalities` | property | swapping arguments or reordering pairs changes nothing | both towers get a training signal |
| `test_siglip_defined_at_batch_one` | boundary | $B = 1$: SigLIP has a loss and gradient, CLIP is 0 | small and sharded batches |
| `test_temperature_gradient_sign` | property | $\partial \mathcal{L} / \partial \tau < 0$ on matched pairs, $> 0$ on mismatched | the scale grows when the model is right |
| `test_clip_temperature_is_clamped_at_100` | boundary | past $\ln 100$: the loss at 100, no gradient | stable long runs |
| `test_siglip_loss_is_stable_at_large_logits` | boundary | logits of $\pm 2000$ give $\ln 2 + 2000$, not $\infty$ | late training, half precision |
| `test_dual_encoder_parameters_and_loss` | unit | names, bias-free projections, `logit_bias` only for SigLIP, backward reaches every parameter | checkpoints load by name |
| `test_image_pooling_cls_or_mean` | unit | CLS when the tower has one, else the patch mean | CLIP and SigLIP towers both fit |
| `test_text_padding_is_invisible` | property | extra padding never changes a caption's embedding | `data.10` scores captions in batches |
| `test_zero_shot_classifier_is_a_prompt_ensemble` | unit | rows are renormalized template means; predict is the argmax | `ethics.07`'s probe |
| `test_bad_embedding_shapes_are_refused` | boundary | unequal batches or widths raise `ValueError` | no loss on broadcast noise |
| `test_dual_encoder_learns_zero_shot_shapes` | learning | 600 SigLIP steps on PCG32-drawn shapes: zero-shot accuracy at least the reference threshold (0.80) | the parts train together (`MS-L13` step 2) |

| `test_encoded_images_match_explicit_pixel_pipeline` | Encoded images traverse the complete processor with preserved order; incompatible processors fail. |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. raw dot products in the loss | long vectors win; the scale parameter means nothing | `test_hand_example_two_pairs`, `test_clip_loss_matches_hf` (mutant `s01`) |
| 2. one direction of the CLIP loss | the image tower gets half the signal; not symmetric | `test_clip_loss_is_symmetric_in_modalities` (mutant `s02`) |
| 3. dividing by $e^{\tau}$ | the temperature gradient has the wrong sign | `test_temperature_gradient_sign`, `test_clip_loss_matches_hf` (mutant `s03`) |
| 4. no clamp | the scale grows without bound | `test_clip_temperature_is_clamped_at_100` (mutant `s04`) |
| 5. SigLIP averaged over $B^2$ | gradients $B$ times too small | `test_siglip_loss_matches_hf` (mutant `s05`) |
| 6. labels 1 and 0 for $z$ | negatives are never pushed away; nothing is learned | `test_siglip_loss_matches_hf`, `test_dual_encoder_learns_zero_shot_shapes` (mutant `s06`) |
| 7. $\ln(\sigma(x))$ as written | `inf` loss once a logit is very negative | `test_siglip_loss_is_stable_at_large_logits` (mutant `s07`) |
| 8. bias added before the scale | the init is $-100$, not $-10$: no gradient at the start | `test_siglip_loss_matches_hf`, `test_siglip_defined_at_batch_one` (mutant `s08`) |
| 9. text mean over padding too | a caption's embedding depends on its batch | `test_text_padding_is_invisible` (mutant `s09`) |
| 10. `encode_image` not normalized | zero-shot scores are not cosines | `test_dual_encoder_parameters_and_loss`, `test_image_pooling_cls_or_mean` (mutant `s10`) |
| 11. $t$ used as the scale, not $e^t$ | the init scale is 2.3, and it can go negative | `test_siglip_loss_matches_hf` (mutant `s11`) |
| 12. padding visible to attention | the same as 9, through the keys | `test_text_padding_is_invisible` (mutant `s12`) |
| 13. mean pooling over a CLS tower | a loaded CLIP projection reads the wrong vector | `test_image_pooling_cls_or_mean` (mutant `s13`) |
| 14. prompt mean not renormalized | classes with agreeing prompts get larger scores | `test_zero_shot_classifier_is_a_prompt_ensemble` (mutant `s14`) |
| 15. $\ln \sigma$'s derivative as $\sigma(x)$ | right loss, wrong gradients | `test_gradcheck_siglip_loss` (mutant `s15`) |
| 16. clamp computed in numpy | `logit_scale` is silently frozen | `test_gradcheck_clip_loss`, `test_clip_temperature_is_clamped_at_100` (mutant `s16`) |
| accepting unequal batches | a loss over pairs that do not exist | `test_bad_embedding_shapes_are_refused` (mutant `m01`) |
| accepting an unknown loss name | a typo trains SigLIP when you asked for CLIP | `test_dual_encoder_parameters_and_loss` (mutant `m02`) |
| accepting an all-padding caption | a NaN embedding enters the batch | `test_text_padding_is_invisible` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L13.3` | `ViT` is the image tower; `VisionBlock` and `ViTConfig` build the text tower's blocks |
| Back | `L0.4` | `Embedding`, `LayerNorm`, `Linear`, `ModuleList`, `Module` |
| Back | `L0.2` | `matmul`, `transpose`, `log_softmax`, `exp`, `sum`, `mean`, `reshape` |
| Back | `L0.1` | `Tensor`; `from_op` for the $\ln \sigma$ op and its backward |
| Back | `M06.3` | `PCG32(0).substream("init")` when no rng is given |
| Back | `L5.3`, `M11.1`, `L0.3` | attention in the towers, softmax with a temperature, cross-entropy (reading) |
| Forward | `data.10` | scores each scraped image-caption pair with the cosine of the two encodings and drops the low ones (not registered yet) |
| Forward | `ethics.07` | builds zero-shot classifiers and compares accuracy across image source domains (not registered yet) |

| Back | `L13.6` | `encode_images` applies the image processor before encoding each image, rejecting incompatible geometry. |

| Forward | `ethics.08` | The consent and model-card evidence practice calls this implementation. |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `clip_loss` | OpenCLIP `ClipLoss` | features gathered across GPUs so every device's batch serves as negatives for all | `mlfoundations/open_clip`, `src/open_clip/loss.py` |
| `siglip_loss` | big_vision SigLIP | chunked pairwise loss: devices swap text halves, so no all-gather and batch 32k fits | `google-research/big_vision`, `trainers/proj/image_text/siglip.py` |
| `DualEncoder` | HF `CLIPModel`, `SiglipModel` | pretrained towers, EOS-token and attention pooling, SigLIP 2's NaFlex resolutions | `transformers/models/siglip/modeling_siglip.py` |
| `zero_shot_classifier` | OpenCLIP `build_zero_shot_classifier` | 80 ImageNet templates, batched over classes | `src/open_clip/zero_shot_classifier.py` |
| the similarity score | CLIPScore, LAION filtering | the cosine as a caption metric and as the filter behind LAION-5B | Hessel et al. (2021); Schuhmann et al. (2022) |
