# Fine-tuning heads and the linear-head export

## Overview

| | |
|---|---|
| **Module** | `L6.5` · build · Python · Pass 5 · 2 h reading |
| **Solution** | [`python/tinyllm/obj/heads.py`](../../../../course/solutions/python/tinyllm/obj/heads.py): `hidden_states`, `pool`, `SequenceClassifier`, `TokenClassifier`, `RewardHead`, `pairwise_reward_loss`, `lora_classifier`, `train_classifier`, `predict`, `save_classifier`, `load_classifier`, and the policy head: `fit_linear_head`, `head_probs`, `head_metrics`, `export_linear_head`, `load_linear_head`; model oracle tests in [`learner-tests/L6.5/.../test_heads.py`](../../../../course/solutions/learner-tests/L6.5/python/tests/l6-5-heads/test_heads.py) |
| **Contract** | [`course/contracts/py/tinyllm/obj/heads.pyi`](../../../../course/contracts/py/tinyllm/obj/heads.pyi) |
| **Needs** | `L6.1` GPT · `L6.2` `BertEncoder` · `L6.6` `inject_lora` · [`M07.7` IRLS, ROC-AUC, ECE](../../../tracks/math/07-probability-statistics/07-logistic-regression-roc-auc-calibration.md) · [`M10.3` AdamW](../../../tracks/math/10-optimization/03-adam-and-adamw.md) · `L0.1` · `L0.2` · `L0.3` · `L0.4` · `L0.6` · `M06.3` |
| **Used by** | `L6.7` the zoo's classifier rows · at the end of Part 6 the system fine-tunes a classifier with LoRA at $r = 8$, training under 5% of the parameters · later: `ethics.05` the usage-policy head, evaluated in Go by `gw.08`; `L12.3` reward model (optional) |
| **Optional depth** | Devlin et al., BERT (2019), section 4; Ouyang et al., "Training language models to follow instructions with human feedback" (2022), section 3.5 (the reward model); Bradley and Terry, "Rank Analysis of Incomplete Block Designs" (1952); scikit-learn's `LogisticRegression` docs |

## Key Takeaways

- A head is small: pool one vector per sequence (`cls`, `mean` over **real** tokens, or the **last real** token) and apply one Linear.
- Padding must be invisible: the same row with or without padding gets the same logits.
- A reward head is trained on pairs: the loss is $-\log\sigma(r_{\text{chosen}} - r_{\text{rejected}})$.
- LoRA fine-tuning trains the adapters **and** the new head.
- The policy head is M07.7's logistic regression fitted on **unit** embeddings and exported as JSON that Go scores identically.

## How to read this chapter

Sections 1 to 3 derive the three heads and the linear policy head and work pooling, the policy score, and the reward loss by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each choice is right. Try to predict each step before reading it.

---

## 1. Why now

The backbones (`L6.1` GPT, `L6.2` BERT, `L6.3` ELECTRA's discriminator) turn tokens into vectors, but nothing in the system asks them a question yet. Three call sites need a head. The model zoo (`L6.7`) compares backbones by how well they classify the same sentences. The gateway's usage policy (`gw.08`, Pass 10) must refuse some prompts, and Python never serves HTTP (D9), so the classifier has to be something Go can evaluate: a linear head over the engine's own `/v1/embeddings` (D33). And post-training (`L12.3`, optional) needs a reward model. All three are a pooled vector and one Linear; what goes wrong is the plumbing around it: padding, pooling, which parameters train, and numbers that must survive a trip through JSON into another language.

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

stored as [`formats/linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json). Normalizing makes the score independent of the embedding's length, which varies with the prompt; the head must be **fitted** on unit vectors too, or it learns to read length. `fit_linear_head` fits `M07.7`'s `logistic_regression_fit` (IRLS: Newton steps, each a weighted least-squares solve) on the unit rows, giving $w$ and an intercept $c$, and stores two softmax rows with class 0 as the reference:

$$W = \begin{pmatrix} 0 \\ w^\top \end{pmatrix}, \quad b = (0, c), \quad p_1 = \frac{e^{w^\top u + c}}{1 + e^{w^\top u + c}} = \sigma(w^\top u + c),$$

exactly the fitted probability. The decision is $p_1 \ge$ `threshold`. `head_metrics` records accuracy, precision, and recall at that threshold, plus `M07.7`'s threshold-free ROC-AUC and expected calibration error. The export writes every number as the shortest decimal that reads back as the same float64, so Go's `strconv.ParseFloat` gets the same weights, and a shared set of probe embeddings with their probabilities pins both languages to agree within 1e-6.

## 3. Worked example by hand

**Pooling.** One row of hidden states $(1, 2)$, $(3, 4)$, $(5, 6)$ with the third position padding ($m = (1, 1, 0)$): `cls` $= (1, 2)$; `mean` $= ((1 + 3)/2, (2 + 4)/2) = (2, 3)$; `last` $= (3, 4)$, the last real token, not $(5, 6)$. With an identity classifier and no bias, the mean-pooled logits are $(2, 3)$; for label 1 the cross-entropy is $-\log \frac{e^3}{e^2 + e^3} = \log(1 + e^{-1}) = 0.313262$.

**Policy head.** $e = (3, 4)$, so $\lVert e \rVert = 5$ and $u = (0.6, 0.8)$. With $W = \begin{pmatrix} 0 & 0 \\ 1 & 2 \end{pmatrix}$ and $b = (0, -1)$: logits $(0,\ 0.6 + 1.6 - 1) = (0, 1.2)$ and $p_{\text{unsafe}} = \sigma(1.2) = 0.768525$. The embeddings $(0.03, 0.04)$ and $(300, 400)$ have the same $u$, so the same score. The zero vector stays $u = 0$ and scores $\mathrm{softmax}(b) = (0.731059, 0.268941)$.

**Reward pair.** $r_c = 2$, $r_r = 0$: $L = \log(1 + e^{-2}) = 0.126928$; $(5, 3)$ gives the same loss.

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand-worked pooling and loss | unit | section 3: the three pools and the loss 0.313262 | everyone agrees on pooling |
| hand-worked policy head | unit | section 3: $p = \sigma(1.2)$ at three scales; the zero vector scores softmax($b$) | the gateway's formula |
| hand-worked reward loss | unit | $\log(1 + e^{-2})$ for two pairs with margin 2 | the Bradley-Terry loss |
| `last` reads the last real token | boundary | a GPT backbone: `last` picks positions 6, 3, 1 of three padded rows | decoder classifiers and reward models |
| the mean ignores padding | property | huge values in padding change nothing; an all-padding row and an unknown pool raise | padded batches |
| a BERT classifier does not see padding | property | a padded row and the same row alone get the same logits, whatever the padding ids | the encoder gets the mask |
| token loss skips padding and ignored labels | unit | the loss equals the mean over real, labelled positions | token tasks |
| the reward head ranks by the margin | property | $r = w^\top h_{t^*}$; one step on the pair loss widens the margin | reward modelling (`L12.3`) |
| LoRA trains the adapters and the head | unit | exactly the q and v adapters plus the head train; a few steps run | the Part 6 LoRA fine-tune |
| the policy head is IRLS on unit vectors | differential | $p_1$ equals `M07.7`'s fit on unit rows; scale invariance; 3 classes refused | the head is the fitted regression |
| Python scores equal the shared probe set | golden | stored probe probabilities for `gw.08`, to 1e-12 | Python and Go agree within 1e-6 |
| the export validates against the schema | conformance | [`linear-head.schema.json`](../../../../course/contracts/formats/linear-head.schema.json); weights read back bit for bit; bad heads refused | the gateway loads the file |
| metrics at the threshold | unit | precision, recall, accuracy at the threshold; AUC 3.5/6 | the numbers the policy card reports |
| save and load a classifier | property | BERT and GPT classifiers reload and predict alike; adapters must be merged first | the zoo loads classifiers |
| it learns sentiment | learning | a tiny BERT, 150 AdamW steps on a synthetic SST-2 stand-in reach a reference validation accuracy | the whole path learns |
| bad arguments | boundary | bad class counts, pools, shapes, and label counts raise | caller bugs fail loudly |

The learning case uses a synthetic stand-in for the SST-2 2k subset (the real SST-2 has no clear license to commit): 2000 template sentences with negation ("is not dull") and contrast ("dull but the ending is wonderful"), so bag-of-words is not enough. A model test suite (numpy pools on a constant backbone, a padded and unpadded run of the same encoder, `M07.7`'s own fit on the unit rows, the reward loss from `math.log1p`) is in [`test_heads.py`](../../../../course/solutions/learner-tests/L6.5/python/tests/l6-5-heads/test_heads.py).

## 5. The solution, step by step

### 5.1 From first principles

1. A head answers one question per sequence (or per token) from the backbone's hidden states. The backbone already did the hard work; the head should be one Linear so that fine-tuning is cheap and the pretrained representation is not overwhelmed.
2. A sequence head needs one vector per row. Which vector depends on what has read the whole row: in an encoder, `[CLS]` attends to everything; in a decoder, only the last real token has seen every other. A mean over real tokens works for any encoder.
3. Padding must be invisible. That forces three things: the encoder receives the mask; `mean` divides by the number of real tokens; `last` finds the last real index, not $T - 1$. In the worked example $(3, 4)$, not $(5, 6)$.
4. A reward model is only ever compared, never read absolutely, so a bias is useless (it cancels in $r_c - r_r$) and the loss is the Bradley-Terry likelihood of the margin. $(2, 0)$ and $(5, 3)$ give the same loss.
5. LoRA freezes the backbone except small adapters. The head is new and random, so it must train in full, or nothing can map the adapted features to labels.
6. The gateway (Go, no Python) must compute the policy score with arithmetic alone, so the policy head is a linear map over the engine's embeddings. Embedding norm varies with prompt length, so the head reads unit vectors, and must be **fitted** on unit vectors too.
7. A binary logistic regression is a two-row softmax with row 0 fixed at zero; storing it that way lets Go run one generic softmax and still reproduce $\sigma(w^\top u + c)$ exactly.
8. For two languages to agree to 1e-6, the JSON must carry every float64 exactly: the shortest round-trip decimal.

### 5.2 The code, annotated

**Hidden states and pooling.**

```python
def hidden_states(backbone, ids, token_type_ids=None, attn_mask=None):
    if hasattr(backbone, "hidden"):  # a decoder (L6.1's GPT): causal, no padding mask needed
        return backbone.hidden(ids)
    return backbone(ids, token_type_ids, attn_mask)  # an encoder (L6.2's BertEncoder)


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

`argmax` over the reversed mask finds the first True from the right; subtracting from $T - 1$ turns it back into an index. With $m = (1, 1, 0)$ the reversed mask is $(0, 1, 1)$, `argmax` is 1, and `last` $= 2 - 1 = 1$, the token $(3, 4)$. A causal decoder ignores right padding for real positions, so `hidden` needs no mask, but `pool` still needs it to pick the right position.

**The three heads.**

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

**LoRA fine-tuning and training.**

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

`save_classifier` refuses a model that still contains `LoRALinear` modules: the directory must be a plain classifier, so the adapters are merged first (`L6.6`). It stores the backbone config as a dict under `tl_backbone` and the head under `tl_head`; `load_classifier` rebuilds a `GPT` for `arch == "gpt"` and a `BertEncoder` otherwise (ELECTRA's discriminator body is a `BertEncoder`).

**The policy head.** Unit rows, IRLS, two softmax rows:

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
    w = logistic_regression_fit(X, y, l2, iters)  # M07.7 IRLS; intercept last
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

**Metrics at the deployed threshold, and the export.**

```python
    p1 = head_probs(head, embeddings)[:, -1]
    flag = p1 >= head["threshold"]          # the decision the gateway makes, not p1 >= 0.5
    # ... tp, fp, fn; accuracy, precision, recall; auc and ece from M07.7 ...
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

Before writing, `export_linear_head` requires a non-empty `embedding_model` (a head is only valid for the embeddings it was fitted on), checks `W` and `b` against the class count and `dim`, and refuses non-finite weights, which JSON cannot carry.

Full solution: [`heads.py`](../../../../course/solutions/python/tinyllm/obj/heads.py).

### 5.3 Why this solution

- **Cost.** A head adds $d \cdot C$ parameters; under LoRA the trainable fraction is the adapters plus the head, well under 5% for the Part 6 model at $r = 8$. Scoring the policy head is one $d$-dimensional dot product per class.
- **Pooling as a function, not a subclass per pool:** one code path, and the reward head reuses `last`.
- **IRLS rather than SGD for the policy head:** it is convex and small, so Newton steps converge to the exact regularized optimum in a few iterations, deterministically. That makes the Python and Go numbers reproducible.
- **Softmax over two rows rather than a bare sigmoid:** the Go side and the schema handle $C$ classes uniformly, and the reference row of zeros makes it exactly the fitted logistic regression.
- **Tradeoff accepted.** A linear head over frozen embeddings is weaker than a fine-tuned classifier, but it costs nothing at serving time (the engine computes the embedding anyway) and can be evaluated without Python.

### 5.4 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `last` reads position $T - 1$ | every short row is classified by a padding vector | `last = T - 1 - argmax(real[:, ::-1])` |
| 2. padding in the mean, or an encoder called without the mask | predictions change with batch composition | `mean` weights by the mask in sum and count; `hidden_states` passes `attn_mask` to encoders |
| 3. padding counted in a token loss | the loss rewards predicting padding labels | `np.where(real, lab, -100)` before `cross_entropy` |
| 4. the policy head fitted or scored on raw embeddings | long prompts score differently from short ones with the same content; Go and Python disagree | `_unit_rows` in both `fit_linear_head` and `head_probs` |
| 5. the reward margin with the wrong sign | the reward model learns to prefer the rejected answers | BCE with target 1 on `r_chosen - r_rejected` |
| 6. the new head frozen under LoRA | accuracy stays at chance whatever the adapters do | `lora_classifier` sets `requires_grad = True` on the head |
| 7. weights exported at float32 precision | Go reads other doubles; scores drift past 1e-6 | `float(v)` of float64 values: the shortest round-trip repr |
| 8. precision and recall at 0.5 instead of the threshold | the policy card reports a classifier the gateway never runs | `flag = p1 >= head["threshold"]` |

## 7. Where it's used next

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

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `SequenceClassifier` | HF `BertForSequenceClassification`, `AutoModelForSequenceClassification` | BERT's pooler (a tanh layer over `[CLS]`), regression and multi-label problem types | `transformers/models/bert/modeling_bert.py` |
| `RewardHead`, `pairwise_reward_loss` | TRL `RewardTrainer` | margins per pair, centering rewards, the reward model behind PPO and best-of-n | `trl/trainer/reward_trainer.py` |
| `lora_classifier` | PEFT `TaskType.SEQ_CLS` | `modules_to_save=["classifier", "score"]` chosen per architecture | `peft/utils/constants.py` |
| the linear policy head | moderation endpoints, Llama Guard | a full classifier model with a taxonomy; here a single dot product over embeddings the engine already computes | OpenAI moderation docs; Inan et al., "Llama Guard" (2023) |
