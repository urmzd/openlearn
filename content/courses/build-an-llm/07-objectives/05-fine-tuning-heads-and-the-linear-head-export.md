# Fine-tuning heads and the linear-head export

The backbones of this chapter, [GPT](01-gpt-decoder-only.md), [BERT](02-bert-masked-lm.md), and [ELECTRA's discriminator](03-electra-replaced-token-detection.md), turn tokens into vectors, but nothing in the system asks them a question yet. Three call sites need a head. The [model zoo](07-lm-evaluation-harness-and-the-model-zoo.md) compares backbones by how well they classify the same sentences. The gateway's [usage policy enforcement](../../ai-platform-engineering/12-gateway/06-usage-policy-enforcement.md), in *AI Platform Engineering*, must refuse some prompts, and because Python never serves HTTP in this system, the classifier has to be something Go can evaluate: a linear head over the engine's own `/v1/embeddings`. And post-training, in [group relative policy optimization with verifiable rewards](../13-post-training/03-grpo-with-verifiable-rewards.md) (optional), needs a reward model.

All three are a pooled vector and one Linear. What goes wrong is the plumbing around it: padding, pooling, which parameters train, and numbers that must survive a trip through JSON into another language. This section gets each of those right. By the end of the chapter the system fine-tunes a classifier with LoRA at $r = 8$, training under 5% of the parameters, and *Responsible AI*'s [usage policy at the gateway](../../responsible-ai/05-usage-policy/01-usage-policy-at-the-gateway.md) fits and exports the policy head built here.

Besides the three backbones, this section uses `inject_lora` from [LoRA with PiSSA](06-lora-pissa-and-merge.md), and from *Mathematics for Computing* it uses [logistic regression by IRLS, ROC-AUC, and calibration](../../mathematics-for-computing/08-probability-statistics/08-logistic-regression-roc-auc-calibration.md) and [AdamW](../../mathematics-for-computing/11-optimization/03-adam-and-adamw.md), along with the Part 0 foundations and [PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md). The reference implementation is [`python/tinyllm/obj/heads.py`](../../../../course/solutions/python/tinyllm/obj/heads.py) (`hidden_states`, `pool`, `SequenceClassifier`, `TokenClassifier`, `RewardHead`, `pairwise_reward_loss`, `lora_classifier`, `train_classifier`, `predict`, `save_classifier`, `load_classifier`, and the policy head's `fit_linear_head`, `head_probs`, `head_metrics`, `export_linear_head`, `load_linear_head`), with the contract in [`course/contracts/py/tinyllm/obj/heads.pyi`](../../../../course/contracts/py/tinyllm/obj/heads.pyi).

> **Learning objectives**
> - Pool hidden states with `cls`, `mean`, and `last` so that padding is invisible.
> - Build sequence, token, and reward heads, and compute the Bradley-Terry reward loss by hand.
> - Fine-tune a classifier with LoRA while keeping the new head trainable.
> - Fit a linear policy head on unit embeddings with IRLS and express it as a two-row softmax.
> - Export the head as JSON that Go reads back bit for bit.

## Pooling one vector per row

Here is the notation for this section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $h$ | the backbone's hidden states | `float32[B, T, d]` |
| $m$ | the real-token mask (`attn_mask`), right padding | `bool[B, T]` |
| $v_b$ | the pooled vector of row $b$ | `float32[d]` |
| $W_c, b_c$ | the head's Linear (`classifier`) | `[C, d]`, `[C]` |
| $r$ | a reward, one per sequence | `float32[B]` |
| $e$ | an embedding from `/v1/embeddings` | `float64[d]` |
| $u = e / \lVert e \rVert$ | the unit embedding the policy head reads ($u = 0$ when $e = 0$) | `float64[d]` |
| $w, c$ | the logistic regression's weights and intercept | `[d]`, scalar |
| $\sigma(z) = 1 / (1 + e^{-z})$ | the logistic function | |

Every head starts from $h$. An encoder is called as `backbone(ids, token_type_ids, attn_mask)`; a decoder is called through `backbone.hidden(ids)`, and because causal attention never reads the right padding, it needs no mask. A sequence head then needs one vector per row, and there are three sensible choices.

| pool | $v_b$ | when |
|---|---|---|
| `cls` | $h_{b,0}$ | BERT and ELECTRA: position 0 is `[CLS]`, which attends to everything |
| `mean` | $\sum_t m_{bt} h_{bt} / \sum_t m_{bt}$ | any encoder; robust for short fine-tunes |
| `last` | $h_{b, t^*}$ with $t^*$ the last real position | decoders: only the last token has read the whole row |

The mean is over real tokens in both the sum and the count. `last` reads the last real token, not position $T - 1$, which for a short row is padding. A `SequenceClassifier` is then `classifier(dropout(pool(h)))`, and its loss is cross-entropy against one label per row.

## Token and reward heads

The same idea extends in two directions. A `TokenClassifier` applies the Linear at every position, giving logits `[B, T, C]`, and its loss runs over the real positions whose label is not `-100`; padding never counts, whatever label the batch gives it.

A `RewardHead` maps the last real token to a scalar, $r = w^\top h_{b,t^*}$, through a Linear called `score` with no bias, since a constant shift would cancel in every comparison. It learns from preference pairs with the Bradley-Terry model, $P(\text{chosen} \succ \text{rejected}) = \sigma(r_c - r_r)$, whose negative log-likelihood is

$$L = -\log \sigma(r_c - r_r) = \mathrm{softplus}(-(r_c - r_r)),$$

computed as `bce_with_logits` of the margin against 1. Only the margin matters.

## Fine-tuning with LoRA

Full fine-tuning updates every backbone weight. LoRA is cheaper. `lora_classifier` calls `inject_lora` on the classifier, which freezes everything and adapts the backbone's attention queries and values (the Linears named `...q_proj` and `...v_proj`). Then it makes the head trainable again: the head is new, there is no pretrained value to protect, and a frozen random head cannot learn anything. This is what PEFT calls `modules_to_save`.

`train_classifier` runs AdamW over the parameters that require grad, with batches drawn with replacement by `rng.below(n)`. Before saving, the adapters are merged, so the directory is a plain classifier that `load_classifier` rebuilds from `tl_backbone` (a `BertConfig` or `GPTConfig` as a dict) and `tl_head`.

## The linear policy head

The gateway's problem is different. It sees a prompt, asks the engine for its embedding $e$, and must decide "allow" or "refuse" with nothing but arithmetic. The head is

$$p = \mathrm{softmax}(W u + b), \qquad u = e / \lVert e \rVert,$$

stored in the format [`formats/linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json). Normalizing makes the score independent of the embedding's length, which varies with the prompt. The head must be **fitted** on unit vectors too, or it learns to read length.

`fit_linear_head` fits `logistic_regression_fit` (IRLS: Newton steps, each a weighted least-squares solve) on the unit rows, giving $w$ and an intercept $c$. It stores two softmax rows, with class 0 as the reference:

$$W = \begin{pmatrix} 0 \\ w^\top \end{pmatrix}, \quad b = (0, c), \quad p_1 = \frac{e^{w^\top u + c}}{1 + e^{w^\top u + c}} = \sigma(w^\top u + c),$$

which is exactly the fitted probability. The decision is $p_1 \ge$ `threshold`. `head_metrics` records accuracy, precision, and recall at that threshold, plus the threshold-free ROC-AUC and expected calibration error.

The last step is getting the numbers into Go unchanged. The export writes every number as the shortest decimal that reads back as the same float64, so Go's `strconv.ParseFloat` gets the same weights. A shared set of probe embeddings with their probabilities pins both languages to agree within 1e-6.

## Working it by hand

Start with **pooling**. Take one row of hidden states $(1, 2)$, $(3, 4)$, $(5, 6)$ with the third position padding, $m = (1, 1, 0)$. Then `cls` $= (1, 2)$; `mean` $= ((1 + 3)/2, (2 + 4)/2) = (2, 3)$; and `last` $= (3, 4)$, the last real token, not $(5, 6)$. With an identity classifier and no bias, the mean-pooled logits are $(2, 3)$; for label 1 the cross-entropy is $-\log \frac{e^3}{e^2 + e^3} = \log(1 + e^{-1}) = 0.313262$.

Next, the **policy head**. Take $e = (3, 4)$, so $\lVert e \rVert = 5$ and $u = (0.6, 0.8)$. With $W = \begin{pmatrix} 0 & 0 \\ 1 & 2 \end{pmatrix}$ and $b = (0, -1)$, the logits are $(0,\ 0.6 + 1.6 - 1) = (0, 1.2)$, and $p_{\text{unsafe}} = \sigma(1.2) = 0.768525$. The embeddings $(0.03, 0.04)$ and $(300, 400)$ have the same $u$, so they get the same score. The zero vector stays $u = 0$ and scores $\mathrm{softmax}(b) = (0.731059, 0.268941)$.

Finally, a **reward pair**. With $r_c = 2$ and $r_r = 0$, $L = \log(1 + e^{-2}) = 0.126928$, and the pair $(5, 3)$ gives the same loss.

## Designing the interface

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

The interface has two halves. The neural heads wrap any backbone as a `Module`. The policy head is plain float64 numpy and a dict, because its consumer is a JSON file and a Go program, not an autograd graph.

## Building it step by step

The reasoning runs as follows.

1. A head answers one question per sequence (or per token) from the backbone's hidden states. The backbone already did the hard work; the head should be one Linear so that fine-tuning is cheap and the pretrained representation is not overwhelmed.
2. A sequence head needs one vector per row, and which vector depends on what has read the whole row. In an encoder, `[CLS]` attends to everything; in a decoder, only the last real token has seen every other. A mean over real tokens works for any encoder.
3. Padding must be invisible. That forces three things: the encoder receives the mask, `mean` divides by the number of real tokens, and `last` finds the last real index rather than $T - 1$. In the worked example that is $(3, 4)$, not $(5, 6)$.
4. A reward model is only ever compared, never read absolutely, so a bias is useless (it cancels in $r_c - r_r$), and the loss is the Bradley-Terry likelihood of the margin. $(2, 0)$ and $(5, 3)$ give the same loss.
5. LoRA freezes the backbone except small adapters. The head is new and random, so it must train in full, or nothing can map the adapted features to labels.
6. The gateway (Go, no Python) must compute the policy score with arithmetic alone, so the policy head is a linear map over the engine's embeddings. Embedding norm varies with prompt length, so the head reads unit vectors and must be **fitted** on unit vectors too.
7. A binary logistic regression is a two-row softmax with row 0 fixed at zero. Storing it that way lets Go run one generic softmax and still reproduce $\sigma(w^\top u + c)$ exactly.
8. For two languages to agree to 1e-6, the JSON must carry every float64 exactly: the shortest round-trip decimal.

### Hidden states and pooling

```python
def hidden_states(backbone, ids, token_type_ids=None, attn_mask=None):
    if hasattr(backbone, "hidden"):  # a decoder (GPT): causal, no padding mask needed
        return backbone.hidden(ids)
    return backbone(ids, token_type_ids, attn_mask)  # an encoder (BertEncoder)


def pool(h, attn_mask, how):
    # ... how must be cls, mean or last ...
    B, T = h.shape[0], h.shape[1]
    real = _real(np.zeros((B, T), dtype=np.int64), attn_mask)   # all-True if no mask; refuses all-padding rows
    if how == "cls":
        return h[:, 0]
    if how == "mean":
        w = real.astype(h.data.dtype)[:, :, None]
        return F.sum(h * w, axis=1) / w.sum(axis=1)   # real tokens in the sum AND the count
    # last: the last REAL token of each row (right padding), not position T - 1.
    last = T - 1 - np.argmax(real[:, ::-1], axis=1)
    return h[np.arange(B), last]
```

`argmax` over the reversed mask finds the first True from the right, and subtracting from $T - 1$ turns it back into an index. With $m = (1, 1, 0)$ the reversed mask is $(0, 1, 1)$, `argmax` is 1, and `last` $= 2 - 1 = 1$, the token $(3, 4)$. A causal decoder ignores right padding for real positions, so `hidden` needs no mask, but `pool` still needs it to pick the right position.

### The three heads

```python
class SequenceClassifier(Module):
    # ... backbone, dropout, classifier = Linear(d_model, n_classes) ...
    def forward(self, ids, token_type_ids=None, attn_mask=None, labels=None):
        h = hidden_states(self.backbone, ids, token_type_ids, attn_mask)
        logits = self.classifier(self.dropout(pool(h, attn_mask, self.pool)))
        if labels is None:
            return logits, None
        return logits, cross_entropy(logits, np.asarray(labels).astype(np.int64))


class TokenClassifier(Module):
    def forward(self, ids, token_type_ids=None, attn_mask=None, labels=None):
        x = np.asarray(ids)
        real = _real(x, attn_mask)
        logits = self.classifier(self.dropout(hidden_states(self.backbone, x, token_type_ids, attn_mask)))
        # ...
        lab = np.asarray(labels).astype(np.int64)
        # Padding is never a target, whatever label it carries.
        return logits, cross_entropy(logits, np.where(real, lab, -100), ignore_index=-100)


class RewardHead(Module):
    def __init__(self, backbone, d_model, rng=None):
        # ...
        self.score = Linear(d_model, 1, bias=False, rng=_rng(rng))   # a bias would cancel in every margin

    def forward(self, ids, attn_mask=None):
        h = hidden_states(self.backbone, ids, None, attn_mask)
        r = self.score(pool(h, attn_mask, "last"))  # [B, 1]
        return F.reshape(r, (r.shape[0],))


def pairwise_reward_loss(r_chosen, r_rejected):
    # ... both [B] ...
    # -log sigmoid(r_c - r_r) = BCE with target 1 on the margin.
    return bce_with_logits(r_chosen - r_rejected, np.ones(r_chosen.shape, dtype=np.float32))
```

The attributes are named `classifier` and `score`, as in Hugging Face and PEFT, so `lora_classifier` and the saved keys find them by name.

### LoRA fine-tuning and training

```python
def _attn_value(name, m):
    # Adapt the backbone's attention queries and values only (the LoRA paper's choice).
    return name.startswith("backbone.") and name.endswith((".q_proj", ".v_proj"))


def lora_classifier(clf, r, alpha, target=None, init="default", rng=None):
    names = inject_lora(clf, target or _attn_value, r, alpha, init=init, rng=rng)   # freezes everything
    # The head is new: it trains in full (PEFT's modules_to_save).
    head = clf.classifier if hasattr(clf, "classifier") else clf.score
    for p in head.parameters():
        p.requires_grad = True
    return names
```

```python
    params = [p for p in clf.parameters() if p.requires_grad]   # adapters + head under LoRA
    opt = AdamW(params, lr=lr, weight_decay=weight_decay)
    clf.train()
    for _ in range(steps):
        idx = np.array([rng.below(len(x)) for _ in range(batch_size)])  # with replacement
        opt.zero_grad()
        _, loss = clf(x[idx], None, real[idx], y[idx])
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    clf.eval()
```

`save_classifier` refuses a model that still contains `LoRALinear` modules: the directory must be a plain classifier, so the adapters are merged first. It stores the backbone config as a dict under `tl_backbone` and the head under `tl_head`. `load_classifier` rebuilds a `GPT` for `arch == "gpt"` and a `BertEncoder` otherwise, since ELECTRA's discriminator body is a `BertEncoder`.

### The policy head

The policy head is unit rows, IRLS, and two softmax rows:

```python
def _unit_rows(E):
    X = np.asarray(E, dtype=np.float64)
    # ... finite 2-D check ...
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > 0)  # a zero vector stays zero


def fit_linear_head(embeddings, labels, classes, l2, iters=50, threshold=0.5):
    # ... exactly 2 classes; 0 <= threshold <= 1 ...
    X = _unit_rows(embeddings)  # fit on what the gateway will score: unit vectors
    y = np.asarray(labels)
    w = logistic_regression_fit(X, y, l2, iters)  # IRLS; intercept last
    d = X.shape[1]
    # Two softmax rows: class 0 is the reference (all zeros), so
    # p(class 1) = softmax([0, w.e + c])[1] = sigmoid(w.e + c), exactly the fit.
    W = np.zeros((2, d))
    W[1] = w[:d]
    b = np.array([0.0, w[d]])
    return {"dim": d, "classes": list(classes), "W": W, "b": b, "threshold": float(threshold), "l2": float(l2)}


def head_probs(head, embeddings):
    X = _unit_rows(embeddings)
    W, b = np.asarray(head["W"], dtype=np.float64), np.asarray(head["b"], dtype=np.float64)
    # ... dim check ...
    z = X @ W.T + b
    z -= z.max(axis=1, keepdims=True)    # stable softmax
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)
```

### Metrics at the deployed threshold, and the export

```python
    p1 = head_probs(head, embeddings)[:, -1]
    flag = p1 >= head["threshold"]          # the decision the gateway makes, not p1 >= 0.5
    # ... tp, fp, fn; accuracy, precision, recall; auc and ece ...
```

```python
    doc = {
        "embedding_model": embedding_model,
        "dim": int(head["dim"]),
        "classes": classes,
        # float() of a float64 prints the shortest repr that round-trips: Go's
        # strconv.ParseFloat reads back the exact same double.
        "W": [[float(v) for v in row] for row in W],
        "b": [float(v) for v in b],
        "threshold": float(head["threshold"]),
    }
```

Before writing, `export_linear_head` requires a non-empty `embedding_model` (a head is only valid for the embeddings it was fitted on), checks `W` and `b` against the class count and `dim`, and refuses non-finite weights, which JSON cannot carry. The full solution is in [`heads.py`](../../../../course/solutions/python/tinyllm/obj/heads.py).

### Why this design

A head adds $d \cdot C$ parameters. Under LoRA the trainable fraction is the adapters plus the head, well under 5% for this chapter's model at $r = 8$. Scoring the policy head is one $d$-dimensional dot product per class.

**Pooling is a function, not a subclass per pool**, so there is one code path and the reward head reuses `last`. The policy head uses **IRLS rather than SGD**: the problem is convex and small, so Newton steps converge to the exact regularized optimum in a few iterations, deterministically, which makes the Python and Go numbers reproducible. It is stored as a **softmax over two rows rather than a bare sigmoid**, so the Go side and the schema handle $C$ classes uniformly, and the reference row of zeros makes it exactly the fitted logistic regression. The tradeoff accepted is that a linear head over frozen embeddings is weaker than a fine-tuned classifier; but it costs nothing at serving time, since the engine computes the embedding anyway, and it can be evaluated without Python.

## Testing it

A correct solution has to pass these checks.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand-worked pooling and loss | unit | the three pools and the loss 0.313262 | everyone agrees on pooling |
| hand-worked policy head | unit | $p = \sigma(1.2)$ at three scales; the zero vector scores softmax($b$) | the gateway's formula |
| hand-worked reward loss | unit | $\log(1 + e^{-2})$ for two pairs with margin 2 | the Bradley-Terry loss |
| `last` reads the last real token | boundary | a GPT backbone: `last` picks positions 6, 3, 1 of three padded rows | decoder classifiers and reward models |
| the mean ignores padding | property | huge values in padding change nothing; an all-padding row and an unknown pool raise | padded batches |
| a BERT classifier does not see padding | property | a padded row and the same row alone get the same logits, whatever the padding ids | the encoder gets the mask |
| token loss skips padding and ignored labels | unit | the loss equals the mean over real, labelled positions | token tasks |
| the reward head ranks by the margin | property | $r = w^\top h_{t^*}$; one step on the pair loss widens the margin | reward modelling |
| LoRA trains the adapters and the head | unit | exactly the q and v adapters plus the head train; a few steps run | the chapter's LoRA fine-tune |
| the policy head is IRLS on unit vectors | differential | $p_1$ equals the IRLS fit on unit rows; scale invariance; 3 classes refused | the head is the fitted regression |
| Python scores equal the shared probe set | golden | stored probe probabilities for the gateway, to 1e-12 | Python and Go agree within 1e-6 |
| the export validates against the schema | conformance | [`linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json); weights read back bit for bit; bad heads refused | the gateway loads the file |
| metrics at the threshold | unit | precision, recall, accuracy at the threshold; AUC 3.5/6 | the numbers the policy card reports |
| save and load a classifier | property | BERT and GPT classifiers reload and predict alike; adapters must be merged first | the zoo loads classifiers |
| it learns sentiment | learning | a tiny BERT, 150 AdamW steps on a synthetic SST-2 stand-in reach a reference validation accuracy | the whole path learns |
| bad arguments | boundary | bad class counts, pools, shapes, and label counts raise | caller bugs fail loudly |

The learning case uses a synthetic stand-in for the SST-2 2k subset, because the real SST-2 has no clear license to commit: 2000 template sentences with negation ("is not dull") and contrast ("dull but the ending is wonderful"), so that bag-of-words is not enough. The model tests in [`test_heads.py`](../../../../course/solutions/learner-tests/L6.5/python/tests/l6-5-heads/test_heads.py) use numpy pools on a constant backbone, a padded and an unpadded run of the same encoder, the IRLS fit itself on the unit rows, and the reward loss from `math.log1p`.

## What can go wrong

Every pitfall here is a plumbing bug: the head itself is one Linear, and almost nothing can go wrong inside it.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `last` reads position $T - 1$ | every short row is classified by a padding vector | `last = T - 1 - argmax(real[:, ::-1])` |
| padding in the mean, or an encoder called without the mask | predictions change with batch composition | `mean` weights by the mask in sum and count; `hidden_states` passes `attn_mask` to encoders |
| padding counted in a token loss | the loss rewards predicting padding labels | `np.where(real, lab, -100)` before `cross_entropy` |
| the policy head fitted or scored on raw embeddings | long prompts score differently from short ones with the same content; Go and Python disagree | `_unit_rows` in both `fit_linear_head` and `head_probs` |
| the reward margin with the wrong sign | the reward model learns to prefer the rejected answers | BCE with target 1 on `r_chosen - r_rejected` |
| the new head frozen under LoRA | accuracy stays at chance whatever the adapters do | `lora_classifier` sets `requires_grad = True` on the head |
| weights exported at float32 precision | Go reads other doubles; scores drift past 1e-6 | `float(v)` of float64 values: the shortest round-trip repr |
| precision and recall at 0.5 instead of the threshold | the policy card reports a classifier the gateway never runs | `flag = p1 >= head["threshold"]` |

> **Warning:** A padding bug often passes every test that uses batches of equal-length rows. Always test a padded row against the same row alone; if the logits differ, padding is leaking into the head.

## Where this leads

The [model zoo](07-lm-evaluation-harness-and-the-model-zoo.md) loads classifier directories with BERT and ELECTRA backbones and reports their accuracy. In *Responsible AI*, [usage policy at the gateway](../../responsible-ai/05-usage-policy/01-usage-policy-at-the-gateway.md) fits and exports the usage-policy head over the engine's embeddings, and in *AI Platform Engineering*, [usage policy enforcement](../../ai-platform-engineering/12-gateway/06-usage-policy-enforcement.md) evaluates the exported head in Go: unit vector, dot products, softmax, threshold. The reward head returns in [GRPO with verifiable rewards](../13-post-training/03-grpo-with-verifiable-rewards.md).

The production equivalents carry the same ideas further. Hugging Face's `BertForSequenceClassification` and `AutoModelForSequenceClassification` (`transformers/models/bert/modeling_bert.py`) add BERT's pooler, a tanh layer over `[CLS]`, and regression and multi-label problem types. TRL's `RewardTrainer` (`trl/trainer/reward_trainer.py`) adds per-pair margins and reward centering, and its reward model sits behind PPO and best-of-n. PEFT's `TaskType.SEQ_CLS` chooses `modules_to_save=["classifier", "score"]` per architecture (`peft/utils/constants.py`). And moderation endpoints and Llama Guard use a full classifier model with a taxonomy, where this book uses a single dot product over embeddings the engine already computes.

Further reading:

- Devlin et al., BERT (2019), section 4.
- Ouyang et al., "Training language models to follow instructions with human feedback" (2022), section 3.5 (the reward model).
- Bradley and Terry, "Rank Analysis of Incomplete Block Designs" (1952).
- scikit-learn's `LogisticRegression` documentation.
- OpenAI moderation documentation; Inan et al., "Llama Guard" (2023).

## Key terms

- **Head**: a small layer, usually one Linear, that maps a backbone's hidden states to a task's outputs.
- **Pooling**: reducing a row's hidden states to one vector (`cls`, `mean` over real tokens, or `last` real token).
- **Bradley-Terry loss**: $-\log\sigma(r_c - r_r)$, the negative log-likelihood that the chosen answer beats the rejected one.
- **`modules_to_save`**: modules kept fully trainable alongside LoRA adapters, here the new head.
- **Linear policy head**: a logistic regression over unit embeddings, stored as a two-row softmax for the gateway.

## Summary

- A head is small: pool one vector per sequence (`cls`, `mean` over **real** tokens, or the **last real** token) and apply one Linear.
- Padding must be invisible: the same row with or without padding gets the same logits.
- A reward head is trained on pairs: the loss is $-\log\sigma(r_{\text{chosen}} - r_{\text{rejected}})$.
- LoRA fine-tuning trains the adapters **and** the new head.
- The policy head is a logistic regression fitted on **unit** embeddings and exported as JSON that Go scores identically.

## Review questions

1. Why does a decoder classifier pool with `last` while BERT uses `cls`?
   <details><summary>Answer</summary>Under causal attention, position $t$ has read only tokens $0..t$, so only the last real token summarizes the whole row. In BERT every position reads every other, and `[CLS]` is trained (by fine-tuning convention) to collect the sentence.</details>
2. Why must the policy head be fitted on unit vectors and not just scored on them?
   <details><summary>Answer</summary>Fitting on raw embeddings gives weights that partly read the vector's length. Scoring those weights on unit vectors applies them to inputs of a different scale, so the probabilities no longer match the fit. Fitting on what is scored keeps the model and the deployed formula the same.</details>
3. Why does `RewardHead.score` have no bias?
   <details><summary>Answer</summary>The loss depends only on $r_c - r_r$, so a bias cancels and receives zero gradient. It would be a parameter that never trains.</details>
4. What happens under LoRA if the head is left frozen?
   <details><summary>Answer</summary>The head stays at its random initialization. The adapters can change the features, but a random fixed projection maps them to labels arbitrarily, so accuracy stays near chance.</details>
5. Why are precision and recall computed at `head["threshold"]` and not at 0.5?
   <details><summary>Answer</summary>The gateway refuses when $p_1 \ge$ threshold. Metrics at any other cut describe a classifier that is never deployed; ROC-AUC and ECE are the threshold-free complements.</details>
