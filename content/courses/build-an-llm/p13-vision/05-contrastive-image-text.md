# Contrastive image-text: CLIP and SigLIP

## Overview

| | |
|---|---|
| **Module** | `L13.5` · build · Python · Pass 12 |
| **Solution** | [`contrastive.py`](../../../../course/solutions/python/tinyllm/vision/contrastive.py): `clip_loss`, `siglip_loss`, `TextEncoder`, `DualEncoder` (`encode_image`, `encode_text`, `encode_images`, `loss`), `zero_shot_classifier`, `zero_shot_predict` |
| **Contract** | [`course/contracts/py/tinyllm/vision/contrastive.pyi`](../../../../course/contracts/py/tinyllm/vision/contrastive.pyi) |
| **Needs** | `L13.6`; `L13.3` `ViT`, `VisionBlock`, `ViTConfig` · `L0.4` `Embedding`, `LayerNorm`, `Linear`, `ModuleList`, `Module` · `L0.2` `matmul`, `transpose`, `log_softmax`, `exp`, `sum`, `mean`, `reshape` · `L0.1` `Tensor`, `from_op` · `M06.3` `PCG32`. Reading: `L5.3` attention, `M11.1` softmax with a temperature, `L0.3` cross-entropy |
| **Used by** | `ethics.08`; later `data.10` (the CLIP-score filter for image-text pairs) · `ethics.07` (the zero-shot disparity probe); neither is written yet. The end-to-end evidence that the parts train together is a tiny dual encoder that learns zero-shot classification of 24 color-shape classes |
| **Optional depth** | Radford et al., "Learning transferable visual models from natural language supervision" (2021), sections 2.3 and 3.1; Zhai et al., "Sigmoid loss for language image pre-training" (2023), sections 3 and 4; van den Oord et al., "Representation learning with contrastive predictive coding" (2018), for InfoNCE as a bound on mutual information |

## Key Takeaways

- CLIP's loss is two cross-entropies over one $B \times B$ matrix of scaled cosines, texts choosing images along the rows and images choosing texts along the columns, so it is symmetric in the two modalities and equals Hugging Face's to $10^{-8}$.
- SigLIP asks $B^2$ independent yes-or-no questions and divides by $B$; it needs no softmax over the batch, so it is defined, with a gradient, for a single pair.
- The temperature is learned as `logit_scale` $= \ln s$ and multiplies the cosines. On a batch the model already separates, a larger $s$ always lowers the loss, so CLIP caps $s$ at 100.
- $\log \sigma(x)$ must be computed as $\min(x, 0) - \ln(1 + e^{-|x|})$; the formula as written returns $-\infty$ at $x = -2000$.
- A classifier can be written in words: one row per class, the renormalized mean of its prompt embeddings. After 600 steps on captions alone the tiny model labels held-out shapes at about 0.89 accuracy against a chance of $1/24$.

## How to read this chapter

Section 1 says why captions are the supervision. Section 2 derives both losses, the temperature, the dual encoder, and zero-shot classification from first principles, and section 3 computes both losses by hand on two pairs. Section 4 fixes the interface. Section 5 walks through `contrastive.py` and explains why each line is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L13.3` built a vision tower that turns pixels into vectors, but nothing yet says what those vectors should mean. Every classifier so far (`L13.1`'s LeNet, `L6.5`'s heads) needed a fixed label set and a labelled example of each class. Captions are free supervision with an open vocabulary: 400 million image-text pairs trained CLIP, and its descendant SigLIP is the vision tower inside SmolVLM, the model `L13.7` loads. This module builds the objective that makes an image vector and a text vector comparable, and the two things that fall out of it: a similarity score for any image-text pair (the filter `data.10` runs over a scraped corpus) and a classifier for classes named only at test time (what `ethics.07` probes for disparities).

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

SigLIP with $\tau = \ln 5$, $b = -3$: logits $\begin{pmatrix} 2 & -3 \\ 0 & 1 \end{pmatrix}$; multiplied by the signs $\begin{pmatrix} + & - \\ - & + \end{pmatrix}$ they are $2, 3, 0, 1$. The four costs $\ln(1 + e^{-x})$ are $0.126928$, $0.048587$, $0.693147$, $0.313262$, which sum to $1.181924$; divided by $B = 2$ the loss is $0.590962$. The pair $(u_2, v_1)$ with cosine $0.6$ is the expensive one: a negative the model has not yet pushed away. A correct solution gives both numbers even when the rows are handed over at lengths 2, 0.5, 3, and 5, because both losses normalize first.

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
    def encode_images(self, images: Sequence[bytes], cfg: ProcessorConfig) -> Tensor: ...   # files -> [N, d_embed]
    def loss(self, pixel_values: Tensor, ids: NDArray) -> Tensor: ...
def zero_shot_classifier(model, tok, class_names, templates) -> NDArray: ...   # [n_classes, d_embed]
def zero_shot_predict(model, pixel_values: Tensor, W: NDArray) -> NDArray: ...  # int64 [N]
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: two pairs | unit, smoke | section 3's two losses, on unnormalized rows | both definitions are pinned |
| CLIP loss matches HF | golden | `CLIPModel`'s loss and three gradients at $B = 4$; zero at $B = 1$ | fine-tuning a published CLIP |
| SigLIP loss matches HF | golden | `SiglipModel`'s loss and four gradients at $B = 5, 3, 1$ | SigLIP is SmolVLM's vision tower (`L13.7`) |
| gradient check, CLIP | gradcheck | backward to both embeddings and `logit_scale` | the temperature trains |
| gradient check, SigLIP | gradcheck | backward through $\ln \sigma$ to embeddings, $t$, $b$ | the one new op is right |
| CLIP is symmetric in the modalities | property | swapping arguments or reordering pairs changes nothing | both towers get a training signal |
| SigLIP is defined at batch one | boundary | $B = 1$: SigLIP has a loss and gradient, CLIP is 0 | small and sharded batches |
| temperature gradient sign | property | $\partial \mathcal{L} / \partial \tau < 0$ on matched pairs, $> 0$ on mismatched | the scale grows when the model is right |
| CLIP temperature clamped at 100 | boundary | past $\ln 100$: the loss at 100, no gradient | stable long runs |
| SigLIP stable at large logits | boundary | logits of $\pm 2000$ give $\ln 2 + 2000$, not $\infty$ | late training, half precision |
| dual encoder parameters and loss | unit | names, bias-free projections, `logit_bias` only for SigLIP, backward reaches every parameter | checkpoints load by name |
| image pooling: CLS or mean | unit | CLS when the tower has one, else the patch mean | CLIP and SigLIP towers both fit |
| text padding is invisible | property | extra padding never changes a caption's embedding | `data.10` scores captions in batches |
| zero-shot classifier is a prompt ensemble | unit | rows are renormalized template means; predict is the argmax | `ethics.07`'s probe |
| bad embedding shapes refused | boundary | unequal batches or widths raise `ValueError` | no loss on broadcast noise |
| dual encoder learns zero-shot shapes | learning | 600 SigLIP steps on PCG32-drawn shapes: zero-shot accuracy at least 0.80 | the parts train together |
| encoded images match the explicit pixel pipeline | unit | `encode_images` equals preprocessing then `encode_image`, in order; a processor with the wrong geometry or splitting raises | `ethics.08` and `data.10` encode image files directly |

## 5. The solution, step by step

### From first principles

1. **Both losses start from one matrix.** Section 2.1: normalize, then $C = U V^\top$ with texts on rows. A shared `_pairs` helper does the shape check and the normalization once, so neither loss can forget either.
2. **CLIP is cross-entropy against the diagonal, twice.** Row $i$'s correct class is $i$, so the loss of a logit matrix is the mean of `log_softmax(logits)[i, i]`, negated. Reading by columns is the same function on the transpose.
3. **The clamp must cut the gradient, not just the value.** Above $\ln 100$ the scale is a constant 100 (no path to `logit_scale`); below it the scale is `exp(t)` in the graph. A numpy clamp everywhere would freeze the temperature; no clamp lets it grow forever.
4. **SigLIP needs one new op.** $\ln \sigma$ must be stable for $|x|$ in the thousands and its derivative is $\sigma(-x)$, so it is a `from_op` node with both computed in their stable forms.
5. **SigLIP's sign trick turns $B^2$ yes-or-no questions into one sum.** With $z = 2I - 1$, every term is $\ln \sigma(z_{ij} \ell_{ij})$; the sum divided by $B$ is the loss.
6. **The encoders must return unit rows, independent of padding.** Image pooling is CLS or the patch mean depending on the tower; text pooling is the mean over real tokens, and padding is masked as a key too. Both project with a bias-free `Linear` and normalize.
7. **A zero-shot classifier is just normalized text embeddings.** Average each class's prompt embeddings, renormalize, stack; prediction is an argmax of cosines.

### The code, annotated

The shared helpers:

```python
def _normalize(x: Tensor) -> Tensor:
    return x * (F.sum(x * x, axis=-1, keepdims=True) ** -0.5)   # unit rows, in the graph


def _pairs(img_emb, txt_emb, who) -> Tensor:
    """The cosine matrix C[i, j] = <txt_i, img_j>, [B, B]."""
    if img_emb.ndim != 2 or txt_emb.ndim != 2 or img_emb.shape != txt_emb.shape:
        raise ValueError(...)                                     # no loss on broadcast noise
    return F.matmul(_normalize(txt_emb), F.transpose(_normalize(img_emb), 0, 1))


def _diag_ce(logits: Tensor) -> Tensor:
    """Mean cross-entropy of row i against class i."""
    b = logits.shape[0]
    lp = F.log_softmax(logits, axis=-1)
    return -F.mean(lp[np.arange(b), np.arange(b)])
```

CLIP, with the clamp as a branch on the value:

```python
def clip_loss(img_emb, txt_emb, logit_scale):
    c = _pairs(img_emb, txt_emb, "clip_loss")
    t = F.reshape(logit_scale, ())
    # Clamp the temperature at 100: past it the scale is a constant.
    scale = F.exp(t) if float(t.data) <= LN100 else Tensor(np.array(100.0), dtype=c.data.dtype)
    logits = c * scale                                        # multiply: a larger s sharpens
    # texts choose images (rows) and images choose texts (columns)
    return (_diag_ce(logits) + _diag_ce(F.transpose(logits, 0, 1))) * 0.5
```

The stable $\ln \sigma$ op:

```python
def _logsigmoid(x: Tensor) -> Tensor:
    """log sigmoid(x) = min(x, 0) - log(1 + exp(-|x|)), stable for any x."""
    a = x.data
    y = np.minimum(a, 0) - np.log1p(np.exp(-np.abs(a)))       # exp only of non-positive numbers
    # d/dx log sigmoid(x) = 1 - sigmoid(x) = sigmoid(-x), also in a form that cannot overflow
    sneg = np.exp(np.minimum(-a, 0)) / (1.0 + np.exp(-np.abs(a)))
    return from_op(y.astype(a.dtype), (x,), lambda g: (g * sneg.astype(a.dtype),), "logsigmoid")
```

SigLIP:

```python
def siglip_loss(img_emb, txt_emb, t, b):
    c = _pairs(img_emb, txt_emb, "siglip_loss")
    n = c.shape[0]
    logits = c * F.exp(F.reshape(t, ())) + F.reshape(b, ())   # scale e^t first, then the bias
    z = 2.0 * np.eye(n) - 1.0  # +1 for the matching pair, -1 for every other
    return -F.sum(_logsigmoid(logits * z)) * (1.0 / n)        # B^2 terms, divided by B
```

The text tower reuses `L13.3`'s `VisionBlock` (a pre-LN block does not care what its tokens came from) over token and position embeddings, with padding masked as keys:

```python
def forward(self, ids):
    a = np.asarray(ids)
    # ... raise unless [N, T <= max_len]
    h = self.token_embedding(a) + self.position_embedding(np.arange(a.shape[1]))
    mask = (a != 0)[:, None, None, :]  # padding is never a key
    for layer in self.layers:
        h = layer(h, mask)
    return self.final_layer_norm(h)
```

`DualEncoder` holds the two towers, two bias-free projections, `logit_scale`, and `logit_bias` only for SigLIP. Its encoders pool, project, and normalize:

```python
def encode_image(self, pixel_values):
    last, _ = self.vision(pixel_values)
    pooled = last[:, 0] if self.vision.cfg.cls_token else F.mean(last, axis=1)   # CLS tower or mean
    return _normalize(self.visual_projection(pooled))

def encode_text(self, ids):
    a = np.asarray(ids)
    keep = (a != 0).astype(np.float32)
    counts = keep.sum(axis=1, keepdims=True)
    if np.any(counts == 0):
        raise ValueError("encode_text: a row holds only padding")   # 0/0 would be NaN
    h = self.text(a)
    pooled = F.sum(h * keep[:, :, None], axis=1) * (1.0 / counts)   # mean over real tokens only
    return _normalize(self.text_projection(pooled))
```

`loss` encodes both sides and dispatches on `loss_kind`. `encode_images` takes image files, runs `L13.6`'s `preprocess` with a single-frame processor config (raising for a splitting processor or pixels that do not match the tower's size), concatenates the frames in input order, and calls `encode_image`.

Zero-shot classification is a few lines of numpy on the encoders' outputs:

```python
def zero_shot_classifier(model, tok, class_names, templates):
    out = []
    for name in class_names:
        e = model.encode_text(_encode_prompts(tok, [t.format(name) for t in templates])).data
        m = e.mean(axis=0)                     # prompt ensemble
        out.append(m / np.linalg.norm(m))      # renormalize: a mean of unit vectors is shorter than 1
    return np.stack(out)


def zero_shot_predict(model, pixel_values, W):
    scores = model.encode_image(pixel_values).data @ np.asarray(W).T   # cosines
    return np.argmax(scores, axis=1).astype(np.int64)
```

`_encode_prompts` tokenizes every prompt and right-pads with id 0, which `encode_text` then ignores.

Full solution: [`contrastive.py`](../../../../course/solutions/python/tinyllm/vision/contrastive.py)

### Why this solution

- **Cost.** Both losses are $O(B^2 d)$ for the matrix plus $O(B^2)$ for the softmaxes or sigmoids; the towers dominate. SigLIP's terms are independent, so the matrix can be computed in chunks across devices without a batch-wide softmax, which is the reason it scales to very large batches.
- **Normalizing inside the loss versus trusting the caller.** The losses are defined on directions; normalizing inside makes them correct for any input and keeps the scale parameter meaningful.
- **A branch for the clamp versus `minimum`.** A smooth or numpy clamp either changes the loss below the cap or kills the gradient everywhere. The branch gives exactly "the loss at 100, no gradient" above the cap and the plain graph below it.
- **One `from_op` for $\ln \sigma$ versus composing ops.** Composing `log(sigmoid(x))` overflows; composing the stable form from `minimum`, `abs`, `exp`, `log1p` would work if all existed as differentiable ops, but a single node with the known derivative $\sigma(-x)$ is shorter and exact.
- **Reusing `VisionBlock` for text.** The block is modality-agnostic; reuse keeps one tested block and one set of parameter names.

### Explain it back

**Why does SigLIP have a nonzero gradient at $B = 1$ while CLIP's loss is exactly 0?**

<details><summary>Answer</summary>

CLIP's softmax over one candidate always gives probability 1, so the cross-entropy is 0 whatever the logit. SigLIP asks whether that one pair matches: $-\ln \sigma(sC + b)$ is positive unless the logit is $+\infty$, and its gradient pushes the cosine up.

</details>

**Why is the temperature multiplied, and what does the clamp change about its gradient?**

<details><summary>Answer</summary>

Multiplying by $s = e^\tau$ means a model that ranks the pairs correctly lowers its loss by growing $s$, so $\partial \mathcal{L}/\partial \tau < 0$ when it is right. Dividing would flip that sign. Above $s = 100$ the solution uses a constant, so the loss is the loss at 100 and `logit_scale` gets no gradient: the growth stops there.

</details>

**Why does SigLIP start the bias at $-10$, and why must the scale be applied before the bias?**

<details><summary>Answer</summary>

Most pairs are negatives, so the prior "no match" should be strong: $\sigma(-10) \approx 4.5 \cdot 10^{-5}$. With the bias added first and then scaled by $s = 10$, the effective bias would be $-100$: every sigmoid saturates and the gradient at the start is essentially zero.

</details>

**What does masking padding as a key add to leaving padding out of the mean?**

<details><summary>Answer</summary>

Leaving it out of the mean removes padded positions' own outputs, but real tokens would still attend to padded keys and change with the amount of padding. Masking the keys makes every real token's output independent of padding, so a caption's embedding is the same in any batch.

</details>

**Why renormalize the mean of the prompt embeddings?**

<details><summary>Answer</summary>

Each prompt embedding is a unit vector; their mean is shorter the more the prompts disagree. Without renormalizing, classes whose prompts agree would get longer weight rows and higher scores for every image, a bias unrelated to the image.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. raw dot products in the loss | long vectors win; the scale parameter means nothing | `_pairs` normalizes both sides before the matmul |
| 2. one direction of the CLIP loss | the image tower gets half the signal; not symmetric | `clip_loss` averages `_diag_ce(logits)` and `_diag_ce` of the transpose |
| 3. dividing by $e^{\tau}$ | the temperature gradient has the wrong sign | `logits = c * scale` |
| 4. no clamp | the scale grows without bound | `clip_loss` uses a constant 100 once `t > ln 100` |
| 5. SigLIP averaged over $B^2$ | gradients $B$ times too small | `siglip_loss` multiplies the sum by `1.0 / n` |
| 6. labels 1 and 0 for $z$ | negatives are never pushed away; nothing is learned | `z = 2.0 * np.eye(n) - 1.0` gives $\pm 1$ |
| 7. $\ln(\sigma(x))$ as written | `inf` loss once a logit is very negative | `_logsigmoid` computes `min(x, 0) - log1p(exp(-|x|))` |
| 8. bias added before the scale | the init is $-100$, not $-10$: no gradient at the start | `c * F.exp(t) + b`: scale first, then bias |
| 9. text mean over padding too | a caption's embedding depends on its batch | `encode_text` weights by `keep` and divides by the real-token count |
| 10. `encode_image` not normalized | zero-shot scores are not cosines | `encode_image` returns `_normalize(...)` |
| 11. $t$ used as the scale, not $e^t$ | the init scale is 2.3, and it can go negative | `siglip_loss` uses `F.exp(t)` |
| 12. padding visible to attention | the same as 9, through the keys | `TextEncoder` masks keys with `(a != 0)` |
| 13. mean pooling over a CLS tower | a loaded CLIP projection reads the wrong vector | `encode_image` takes `last[:, 0]` when `cls_token` |
| 14. prompt mean not renormalized | classes with agreeing prompts get larger scores | `zero_shot_classifier` divides each mean by its norm |
| 15. $\ln \sigma$'s derivative as $\sigma(x)$ | right loss, wrong gradients | `_logsigmoid`'s vjp multiplies by $\sigma(-x)$ |
| 16. clamp computed in numpy | `logit_scale` is silently frozen | below the cap the scale is `F.exp(t)`, a graph op |
| 17. accepting unequal batches | a loss over pairs that do not exist | `_pairs` raises unless both are `[B, d]` of one shape |
| 18. accepting an unknown loss name | a typo trains SigLIP when you asked for CLIP | `DualEncoder.__init__` raises unless `clip` or `siglip` |
| 19. accepting an all-padding caption | a NaN embedding enters the batch | `encode_text` raises when a row's count is 0 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L13.3` | `ViT` is the image tower; `VisionBlock` and `ViTConfig` build the text tower's blocks |
| Back | `L0.4` | `Embedding`, `LayerNorm`, `Linear`, `ModuleList`, `Module` |
| Back | `L0.2` | `matmul`, `transpose`, `log_softmax`, `exp`, `sum`, `mean`, `reshape` |
| Back | `L0.1` | `Tensor`; `from_op` for the $\ln \sigma$ op and its backward |
| Back | `M06.3` | `PCG32(0).substream("init")` when no rng is given |
| Back | `L5.3`, `M11.1`, `L0.3` | attention in the towers, softmax with a temperature, cross-entropy (reading) |
| Forward | `data.10` | scores each scraped image-caption pair with the cosine of the two encodings and drops the low ones (planned) |
| Forward | `ethics.07` | builds zero-shot classifiers and compares accuracy across image source domains (planned) |
| Back | `L13.6` | `encode_images` applies the image processor before encoding each image, rejecting incompatible geometry |
| Forward | `ethics.08` | the consent and model-card evidence practice calls this implementation |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `clip_loss` | OpenCLIP `ClipLoss` | features gathered across GPUs so every device's batch serves as negatives for all | `mlfoundations/open_clip`, `src/open_clip/loss.py` |
| `siglip_loss` | big_vision SigLIP | chunked pairwise loss: devices swap text halves, so no all-gather and batch 32k fits | `google-research/big_vision`, `trainers/proj/image_text/siglip.py` |
| `DualEncoder` | HF `CLIPModel`, `SiglipModel` | pretrained towers, EOS-token and attention pooling, SigLIP 2's NaFlex resolutions | `transformers/models/siglip/modeling_siglip.py` |
| `zero_shot_classifier` | OpenCLIP `build_zero_shot_classifier` | 80 ImageNet templates, batched over classes | `src/open_clip/zero_shot_classifier.py` |
| the similarity score | CLIPScore, LAION filtering | the cosine as a caption metric and as the filter behind LAION-5B | Hessel et al. (2021); Schuhmann et al. (2022) |
