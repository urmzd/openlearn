# ELMo: biLM, ScalarMix, linear probes

## Overview

| | |
|---|---|
| **Module** | `L3.5` · side (optional, D31) · Python · Pass 4 · 1.5 to 2 h reading |
| **Solution** | [`python/tinyllm/rnn/elmo.py`](../../../../course/solutions/python/tinyllm/rnn/elmo.py): `BiLM` (`layers`, `forward`), `bilm_loss`, `ScalarMix`, `fit_linear_probe`, `probe_accuracy` |
| **Contract** | [`course/contracts/py/tinyllm/rnn/elmo.pyi`](../../../../course/contracts/py/tinyllm/rnn/elmo.pyi) |
| **Needs** | `L3.2` `LSTM` · `L3.4` `reverse_padded` · `L0.3` `cross_entropy` · `L0.4` `Embedding`, `Linear`, `ModuleList` · `L0.2` ops · `L0.1` `Tensor` · `M06.3` `PCG32` · training: `M10.3` `AdamW` · reading: `L3.6` (an RNN language model), `content/tracks/math/07-probability-statistics` (logistic regression) |
| **Used by** | no module: the side quest `sq.elmo-probes` (probing every zoo checkpoint) builds on it; optional, and not on the main recurrent path |
| **Optional depth** | Peters et al., "Deep contextualized word representations" (NAACL 2018); Tenney et al., "What do you learn from context? Probing for sentence structure in contextualized word representations" (ICLR 2019); Hewitt and Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019) |

## Key Takeaways

- A biLM is two language models over the same sentence: the forward one predicts the next token, the backward one the previous token, and they share the embedding and the softmax.
- Each direction must see only its own side: forward states never depend on later tokens, backward states never on earlier ones, and the backward stack reverses inside each length.
- ScalarMix is a softmax-weighted sum of the layers times $\gamma$, learned per task.
- A linear probe measures what a frozen layer encodes: the embedding layer cannot tell "saw" the noun from "saw" the verb, the first LSTM layer can.

## How to read this chapter

Sections 1 to 3 build the biLM, ScalarMix and the linear probe from first principles and work the mix by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The word vectors so far (`L2.3`, the embedding table of every model) give a word one vector, whatever the sentence: "saw" in "the saw cuts wood" and in "they saw the dog" is the same point. The recurrent language models of `L3.6` have hidden states that **do** depend on the sentence. ELMo (2018) made that the representation: run a forward and a backward LM, keep every layer's states, and let each downstream task learn a mix of them. It was the step between word2vec and BERT (`L6.2`), and the linear probe that shows what it learned is the tool interpretability work still uses. The module is optional because nothing in the final system calls it (D31); `sq.elmo-probes` applies the probe to every checkpoint in the zoo.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_{1:T}$, $\ell$ | token ids of a sentence, its real length | `int[B, T]`, `int[B]` |
| $d$ | embedding and hidden width | `int` |
| $L$ | number of LSTM layers per direction (`n_layers`) | `int` |
| $e_t = E[x_t]$ | token embedding | `float32[d]` |
| $f^k_t$, $b^k_t$ | forward and backward layer-$k$ states at position $t$, in source order | `float32[d]` |
| $R^0_t = [e_t ; e_t]$, $R^k_t = [f^k_t ; b^k_t]$ | the representation of layer $k$ | `float32[2d]` |
| $s$, $\gamma$ | ScalarMix parameters | `float32[L + 1]`, `float32[1]` |
| $X$, $y$ | probe features (one row per token) and labels | `float[N, D]`, `int[N]` |

### 2.1 Context from both sides

A forward LM's state $f_t$ summarizes $x_{1:t}$; a backward LM's state $b_t$ summarizes $x_{t:\ell}$. Concatenated, $[f_t ; b_t]$ describes $x_t$ in its whole sentence. They cannot be one bidirectional network trained as a language model: if position $t$ saw $x_{t+1}$ it would just copy the answer. So the two directions are trained separately, each to predict in its own direction, and only their **states** are joined.

### 2.2 The biLM

Each direction is a stack of $L$ single-direction LSTMs (`L3.2`). The forward stack reads $e$ left to right; the backward stack reads $e$ reversed **inside each sentence's length** with `L3.4`'s `reverse_padded`, so it starts at the last real token, and its outputs are reversed back to source order. The forward LM predicts $x_{t+1}$ from $f^L_t$, the backward LM $x_{t-1}$ from $b^L_t$, both through the same output layer:

$$\mathcal{L} = \tfrac12 \Big( \operatorname{mean}_{t \le \ell - 2} \mathrm{CE}(W f^L_t, x_{t+1}) + \operatorname{mean}_{t \ge 1} \mathrm{CE}(W b^L_t, x_{t-1}) \Big) .$$

### 2.3 ScalarMix

Different layers carry different information: lower ones more about the word, higher ones more about the context. ELMo lets each task choose:

$$\mathrm{ELMo}_t = \gamma \sum_{k=0}^{L} \operatorname{softmax}(s)_k \, R^k_t .$$

The softmax makes the weights positive and sum to 1; $\gamma$ rescales the result to whatever the task's next layer expects. At initialization $s = 0$ (every layer weighted $1/(L+1)$) and $\gamma = 1$, the plain average. Both are trained with the task.

### 2.4 Linear probes

To ask "does layer $k$ encode X?", freeze the layer, take its vectors for many tokens, and fit the simplest classifier: softmax regression. If a linear map from $R^k$ predicts the label, the information is there in an easily usable form; a deep probe could learn the task itself and say little about the representation (Hewitt and Liang). `fit_linear_probe` standardizes the features (each column to mean 0 and standard deviation 1, so one learning rate suits every column), runs full-batch gradient descent from zero on the cross-entropy with a small L2 penalty, and folds the standardization back into $W$ and $b$ so the probe applies to raw features. From a zero start with full batches it is deterministic, so an accuracy is a property of the layer. (The catalog's probe is `M07.7`'s IRLS logistic regression, taught in Pass 5; this module uses its own gradient-descent version.)

## 3. Worked example by hand

ScalarMix over two layers, $s = (0, \ln 3)$, $\gamma = 2$, with $R^0 = (1, 2)$ and $R^1 = (5, 6)$.

1. $\operatorname{softmax}(s) = (e^0, e^{\ln 3}) / (1 + 3) = (1/4, 3/4)$.
2. The mix: $\tfrac14 (1, 2) + \tfrac34 (5, 6) = (0.25 + 3.75, 0.5 + 4.5) = (4, 5)$.
3. Times $\gamma$: $(8, 10)$.

With the raw scalars as weights instead of their softmax ($0 \cdot R^0 + 1.0986 \cdot R^1$) the result would be $(10.99, 13.18)$. A correct `ScalarMix` must give exactly $(8, 10)$ here.

## 4. The interface

```python
class ScalarMix(Module):
    def __init__(self, n_layers: int) -> None: ...
    def weights(self) -> NDArray: ...; def forward(self, layers: Sequence[Tensor]) -> Tensor: ...
class BiLM(Module):
    def __init__(self, vocab: int, d: int, n_layers: int = 2, rng=None) -> None: ...
    def layers(self, ids, lengths=None) -> list[Tensor]: ...       # [R^0 .. R^L], each [B, T, 2d]
    def forward(self, ids, lengths=None) -> tuple[Tensor, Tensor]: ...   # fwd and bwd logits [B, T, V]
def bilm_loss(model, ids, lengths=None) -> Tensor: ...
def fit_linear_probe(features, labels, n_classes, l2=1e-3, steps=200, lr=0.5) -> tuple[NDArray, NDArray]: ...
def probe_accuracy(W, b, features, labels) -> float: ...
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example of the mix | unit | section 3: $(8, 10)$, weights $(1/4, 3/4)$ | the mix is pinned to a case you can compute |
| ScalarMix starts uniform and trains | gradcheck | the average at init; gradients of $s$, $\gamma$, and every layer in float64 | a task can learn its mix |
| matches torch | golden | layers, both logits, the loss, a mix, and every gradient against torch's packed `nn.LSTM`s | the biLM is ELMo's |
| forward layers never see the future | property | changing later tokens leaves $f^k_t$ bitwise equal | the forward LM cannot copy its answer |
| backward layers never see the past | property | changing earlier tokens leaves $b^k_t$ bitwise equal, with lengths 6, 4, 2 | the backward LM reads from each sentence's own end |
| padding never leaks | property | padding content changes no real position; padded $R^k$ are 0; a row alone equals it in a batch | batches of mixed lengths |
| loss targets | unit | the loss recomputed with explicit targets; length-1 rows only raise | both directions are scored on the right token |
| probe separates separable data | unit | accuracy 1 on separable clusters far from the origin; folded weights work on raw features | the probe measures the features |
| probe is deterministic and validates | boundary | identical probes from identical data; bad shapes, labels, one class | probe accuracies are reproducible |
| contextual layers disambiguate | learning | after 150 steps: loss at the reference bar; probe on $R^0$ at the majority rate, on $R^1$ at least 0.95 | the reason contextual vectors exist |

## 5. The solution, step by step

### From first principles

1. **Each direction may see only its own side.** A language model trained to predict $x_{t+1}$ must not have $x_{t+1}$ in its input (section 2.1). So the forward stack is fed only forward states, the backward stack only backward states, and the two are joined only *after* the last layer, for use as features. No layer mixes them.
2. **The backward stack is a forward stack on reversed input.** Rather than a second recurrence that runs right to left, reverse each sentence inside its length (`L3.4`), run an ordinary `LSTM`, and reverse the outputs back so position $t$ in `b_layers` is the state for token $t$. Padding stays at the end the whole time, so the `lengths` mask of `L3.2` still applies.
3. **Targets are shifts, and their edges are ignored.** Forward position $t$ predicts $x_{t+1}$, which exists only for $t + 1 < \ell$; backward position $t$ predicts $x_{t-1}$, which exists only for $1 \le t < \ell$. Everything else gets the target $-100$, which `cross_entropy` skips (`ignore_index`). The two means are averaged, matching the $\tfrac12(\cdot + \cdot)$ in section 2.2.
4. **Layer 0 must have the same width as the others.** $R^k$ for $k \ge 1$ is $[f^k ; b^k]$, width $2d$; the embedding is width $d$. Repeating it, $[e ; e]$, makes every $R^k$ the same shape so ScalarMix can sum them.
5. **ScalarMix weights must be a convex combination.** Softmax of free scalars gives positive weights summing to 1 with no constraint in the optimizer; $\gamma$ restores a free overall scale. With $s = 0$, $\gamma = 1$ the start is the plain average.
6. **A probe must be deterministic to be a measurement.** Standardize features so one learning rate suits every column, start from $W = 0$, use full-batch gradient descent: identical data gives identical weights. Then fold the standardization into $(W, b)$ so the returned probe works on raw features, as the caller expects.

### The code, annotated

The two stacks. Everything is time first (`[T, B, d]`) because that is what `LSTM` takes:

```python
def _stacks(self, ids, lengths):
    a, n = _ids(ids, self.vocab, lengths)   # validated int64 ids [B, T] and lengths [B]
    e = self.emb(a.T)                       # [T, B, d]
    f_layers, b_layers = [], []
    f = e
    # The backward LM reads each sentence from its last REAL token, so the
    # reversal stays inside each length (L3.4), not a flip of the padding.
    b = reverse_padded(e, n)
    for k in range(self.n_layers):
        f, _ = self.fwd[k](f, lengths=n)    # layer k+1 reads only forward layer k
        b, _ = self.bwd[k](b, lengths=n)    # layer k+1 reads only backward layer k (still reversed)
        f_layers.append(f)
        b_layers.append(reverse_padded(b, n))  # back to source order, for output only
    return e, f_layers, b_layers, n
```

Note that the loop keeps `b` in reversed time and only the *copy* appended to `b_layers` is reversed back. The next backward layer must keep reading in reversed order.

Representations and logits:

```python
def layers(self, ids, lengths=None):
    e, fs, bs, _ = self._stacks(ids, lengths)
    # R^0 = [e; e] so every layer is 2d wide; R^k = [f^k; b^k].
    reps = [F.concat([e, e], axis=-1)] + [F.concat([f, b], axis=-1) for f, b in zip(fs, bs)]
    return [F.transpose(r, 0, 1) for r in reps]       # each [B, T, 2d]

def forward(self, ids, lengths=None):
    _, fs, bs, _ = self._stacks(ids, lengths)
    # One shared output layer for both directions, applied to the top states only.
    return self.out(F.transpose(fs[-1], 0, 1)), self.out(F.transpose(bs[-1], 0, 1))
```

The loss builds both target arrays with $-100$ wherever there is nothing to predict:

```python
def bilm_loss(model, ids, lengths=None):
    a, n = _ids(ids, model.vocab, lengths)
    B, T = a.shape
    fl, bl = model(a, n)
    t = np.arange(T)[None, :]
    # Forward: position t predicts t + 1 (inside the length); backward: t - 1.
    fwd_tgt = np.full((B, T), -100, dtype=np.int64)
    fwd_tgt[:, :-1] = np.where(t[:, :-1] + 1 < n[:, None], a[:, 1:], -100)
    bwd_tgt = np.full((B, T), -100, dtype=np.int64)
    bwd_tgt[:, 1:] = np.where(t[:, 1:] < n[:, None], a[:, :-1], -100)
    if (fwd_tgt >= 0).sum() == 0:
        # Every length is 1: no token to predict, and a mean over nothing is NaN.
        raise ValueError("bilm_loss: no row has a token to predict (every length is 1)")
    V = model.vocab
    lf = cross_entropy(F.reshape(fl, (-1, V)), fwd_tgt.reshape(-1))
    lb = cross_entropy(F.reshape(bl, (-1, V)), bwd_tgt.reshape(-1))
    return (lf + lb) * 0.5
```

ScalarMix is a softmax and a weighted sum, all inside autograd so $s$, $\gamma$ and every layer get gradients:

```python
class ScalarMix(Module):
    def __init__(self, n_layers):
        # ...
        self.scalar_parameters = Tensor(np.zeros(n_layers, dtype=np.float32), requires_grad=True)
        self.gamma = Tensor(np.ones(1, dtype=np.float32), requires_grad=True)

    def forward(self, layers):
        # ... (count and shape checks)
        # Softmax of the scalars: the mix is a convex combination, rescaled by gamma.
        w = F.softmax(self.scalar_parameters, axis=0)
        mixed = layers[0] * w[0]
        for k in range(1, self.n_layers):
            mixed = mixed + layers[k] * w[k]
        return mixed * self.gamma
```

The probe is plain NumPy in float64, since nothing upstream needs its gradient:

```python
def fit_linear_probe(features, labels, n_classes, l2=1e-3, steps=200, lr=0.5):
    X = np.asarray(features, dtype=np.float64)
    # ... (shape, label and n_classes checks)
    N, D = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)          # a constant column would divide by 0
    Z = (X - mu) / sd
    Y = np.zeros((N, n_classes)); Y[np.arange(N), y] = 1.0
    W = np.zeros((D, n_classes)); b = np.zeros(n_classes)   # zero start: deterministic
    for _ in range(steps):
        s = Z @ W + b
        s = s - s.max(axis=1, keepdims=True)                # stable softmax
        P = np.exp(s); P /= P.sum(axis=1, keepdims=True)
        G = P - Y                                           # dCE/dlogits
        W -= lr * (Z.T @ G / N + l2 * W)                    # descent, with L2 on W only
        b -= lr * G.sum(axis=0) / N
    # Fold the standardization in: Z W + b = X (W / sd) + (b - (mu / sd) W).
    W_raw = W / sd[:, None]
    return W_raw, b - mu @ W_raw
```

`probe_accuracy` is then `argmax(X @ W + b) == labels`, averaged. Full solution: [`elmo.py`](../../../../course/solutions/python/tinyllm/rnn/elmo.py).

### Why this solution

- **Cost.** Two stacks of $L$ LSTMs: $O(2 L T B d^2)$ for the recurrences, plus $O(2 T B d V)$ for the shared softmax. The probe is $O(\text{steps} \cdot N D C)$, trivial next to the model.
- **Why not one bidirectional LSTM stack (`L3.4` per layer)?** Layer $k+1$ would read $[f^k ; b^k]$, so the forward state at $t$ would depend on $b^k_t$, which has read $x_{t+1}$. The forward LM could then copy its target and the loss would go to 0 while learning nothing (pitfall 1).
- **Why share the embedding and the softmax?** It halves those parameters and forces both directions into one vocabulary space, which is what makes $[f ; b]$ a single coherent vector. Only the LSTMs are direction-specific.
- **Why a linear probe and not an MLP?** A probe that can learn the task itself measures the probe, not the layer (Hewitt and Liang). The linear one answers "is it there in an easily usable form?".
- **Why gradient descent rather than a closed form or IRLS?** Softmax regression has no closed form; IRLS (`M07.7`) converges faster but needs a Hessian solve per step. Fixed-step gradient descent is simple, deterministic and accurate enough to compare layers.

### Explain it back

1. Why is the forward stack never fed backward states, even though BERT-style models mix both directions in every layer?
   <details><summary>Answer</summary>Each ELMo direction is trained as a language model. If forward layer $k+1$ read $b^k_t$, which depends on $x_{t+1}$, the forward LM would see its own target. BERT avoids this by masking the input tokens instead of separating directions.</details>
2. In `_stacks`, why is `reverse_padded(b, n)` appended to `b_layers` instead of being assigned back to `b`?
   <details><summary>Answer</summary>The next backward layer must keep reading in reversed time. Only the copy kept as output is put back into source order.</details>
3. Why are the backward targets `a[:, :-1]` placed at positions `1:`, and why mask with `t < n`?
   <details><summary>Answer</summary>Backward position $t$ predicts the previous token $x_{t-1}$, so the target at $t$ is `a[t-1]`. Position 0 has nothing before it, and positions at or past $\ell$ are padding, so both get $-100$.</details>
4. Why does ScalarMix need $\gamma$ when the softmax weights already sum to 1?
   <details><summary>Answer</summary>The softmax fixes the weights to a convex combination, which keeps the mix's scale tied to the layers' scale. The task's next layer may want something larger or smaller, and $\gamma$ is a single free factor for that.</details>
5. What does the fold at the end of `fit_linear_probe` compute, and what breaks without it?
   <details><summary>Answer</summary>It rewrites $Z W + b$ with $Z = (X - \mu)/\sigma$ as $X (W/\sigma) + (b - \mu (W/\sigma))$. Without it the returned weights expect standardized features, and `probe_accuracy` on raw features gives wrong predictions.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a forward layer that also reads the backward states | the forward LM sees the future and its loss collapses to 0 | the forward stack reads only forward layers; directions are joined only in `layers` (step 1) |
| 2. reversing the padded tensor, or not reversing back, or running the backward LSTMs without lengths | the backward LM reads padding first, or states land at the wrong positions | `reverse_padded(e, n)` in, `reverse_padded(b, n)` out, and every LSTM call passes `lengths=n` |
| 3. scoring a direction on the current token | each LM is trained to predict what it already read | targets are `a[:, 1:]` for forward and `a[:, :-1]` for backward, with $-100$ at the edges |
| 4. returning probe weights for standardized features | predictions on raw features are wrong | the fold `W / sd`, `b - mu @ W_raw` before returning |
| raw scalars as mix weights | negative or unnormalized layer weights | `F.softmax(self.scalar_parameters)` before mixing |
| $\gamma$ never applied | the task cannot rescale the mix | `return mixed * self.gamma` |
| gradient ascent in the probe | accuracy falls as it trains | `W -= lr * (...)`: the update subtracts the gradient |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L3.2` | `LSTM`, one per layer per direction |
| Back | `L3.4` | `reverse_padded` reverses each sentence inside its length |
| Back | `L0.3` | `cross_entropy` with `ignore_index` for both directions |
| Back | `L0.4` | `Embedding`, `Linear`, `ModuleList` |
| Back | `L0.2` | the op library |
| Back | `L0.1` | `Tensor` |
| Back | `M06.3` | `PCG32` for initialization |
| Back | `M10.3` | `AdamW` in the learning test |
| Forward | `sq.elmo-probes` | the side quest probes every zoo checkpoint layer by layer for part of speech and position |
| Forward | `L6.2` | BERT replaces two one-directional LMs with one masked LM that reads both sides at every layer |

No module calls this code (it is a side module, D31). It is still worth reading: the probe is the cheapest way to see what a network has learned, and the biLM's "each side only sees its own side" rule is exactly the causality property `L5.2` tests for transformer masks.

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `BiLM` | AllenNLP's ELMo | a character CNN instead of a token embedding, projections and residual connections between layers, 4096-wide LSTMs | `allennlp/modules/elmo.py`, `elmo_lstm.py` |
| `ScalarMix` | AllenNLP `ScalarMix` | optional layer normalization of each layer before mixing | `allennlp/modules/scalar_mix.py` |
| `fit_linear_probe` | edge probing, control tasks | span-pair probes, selectivity against random control labels | Tenney et al. 2019; Hewitt and Liang 2019 |
| contextual vectors | BERT, sentence-transformers | one bidirectional encoder trained by masked LM, pooled sentence embeddings | `L6.2`; `ag.06`'s embedding endpoint |
