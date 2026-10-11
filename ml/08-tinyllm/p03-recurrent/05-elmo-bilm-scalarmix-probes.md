<!-- ol:module L3.5 -->
# ELMo: biLM, ScalarMix, linear probes

## Overview

| | |
|---|---|
| **Module** | `L3.5` · side (optional, D31) · Python · Pass 4 · 3 to 4 h |
| **You build** | `python/tinyllm/rnn/elmo.py`: `BiLM` (`layers`, `forward`), `bilm_loss`, `ScalarMix`, `fit_linear_probe`, `probe_accuracy` |
| **Contract** | [`course/contracts/py/tinyllm/rnn/elmo.pyi`](../../../course/contracts/py/tinyllm/rnn/elmo.pyi) |
| **Tests** | `course/tests/L3.5/test_elmo.py` (what they check: section 4), golden values from torch 2.14.1 in `course/fixtures/L3.5/elmo_torch.npz` (`course/oracle/L3.5/elmo_torch.py`), a learning bar in `course/fixtures/ref-thresholds.tsv` |
| **Needs** | `L3.2` `LSTM` · `L3.4` `reverse_padded` · `L0.3` `cross_entropy` · `L0.4` `Embedding`, `Linear`, `ModuleList` · `L0.2` ops · `L0.1` `Tensor` · `M06.3` `PCG32` · tests: `M10.3` `AdamW` · reading: `L3.6` (an RNN language model), `math/07-probability-statistics` (logistic regression) (or `--ref-deps`) |
| **Used by** | no module: the side quest `sq.elmo-probes` (probing every zoo checkpoint) builds on it |
| **Milestone** | none (optional; not part of `MS-L3`) |
| **Optional depth** | Peters et al., "Deep contextualized word representations" (NAACL 2018); Tenney et al., "What do you learn from context? Probing for sentence structure in contextualized word representations" (ICLR 2019); Hewitt and Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019) |

## Key Takeaways

- A biLM is two language models over the same sentence: the forward one predicts the next token, the backward one the previous token, and they share the embedding and the softmax (`test_bilm_loss_targets`, `test_golden_torch`).
- Each direction must see only its own side: forward states never depend on later tokens, backward states never on earlier ones, and the backward stack reverses inside each length (`test_forward_layers_never_see_the_future`, `test_backward_layers_never_see_the_past`, `test_padding_never_leaks`).
- ScalarMix is a softmax-weighted sum of the layers times $\gamma$, learned per task (`test_hand_example_scalar_mix`, `test_scalar_mix_starts_uniform_and_trains`).
- A linear probe measures what a frozen layer encodes: the embedding layer cannot tell "saw" the noun from "saw" the verb, the first LSTM layer can (`test_contextual_layers_disambiguate`).

## How to work this chapter

```bash
ol start L3.5              # stubs elmo.py
ol tests L3.5              # the course tests
ol check L3.5              # exit code is the verdict
ol diff  L3.5              # after passing: your code against the reference
```

---

## 1. Why now

Your word vectors so far (`L2.3`, the embedding table of every model) give a word one vector, whatever the sentence: "saw" in "the saw cuts wood" and in "they saw the dog" is the same point. You have also just trained recurrent language models (`L3.6`) whose hidden states **do** depend on the sentence. ELMo (2018) made that the representation: run a forward and a backward LM, keep every layer's states, and let each downstream task learn a mix of them. It was the step between word2vec and BERT (`L6.2`), and the linear probe that shows what it learned is the tool interpretability work still uses. The module is optional because nothing in the final system calls it (D31); `sq.elmo-probes` applies the probe to every checkpoint in the zoo.

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

With the raw scalars as weights instead of their softmax ($0 \cdot R^0 + 1.0986 \cdot R^1$) the result would be $(10.99, 13.18)$. This is `test_hand_example_scalar_mix`.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_scalar_mix` | unit | section 3: $(8, 10)$, weights $(1/4, 3/4)$ | you and the test agree on the mix |
| `test_scalar_mix_starts_uniform_and_trains` | gradcheck | the average at init; gradients of $s$, $\gamma$, and every layer in float64 | a task can learn its mix |
| `test_golden_torch` | golden | layers, both logits, the loss, a mix, and every gradient against torch's packed `nn.LSTM`s | your biLM is ELMo's |
| `test_forward_layers_never_see_the_future` | property | changing later tokens leaves $f^k_t$ bitwise equal | the forward LM cannot copy its answer |
| `test_backward_layers_never_see_the_past` | property | changing earlier tokens leaves $b^k_t$ bitwise equal, with lengths 6, 4, 2 | the backward LM reads from each sentence's own end |
| `test_padding_never_leaks` | property | padding content changes no real position; padded $R^k$ are 0; a row alone equals it in a batch | batches of mixed lengths |
| `test_bilm_loss_targets` | unit | the loss recomputed with explicit targets; length-1 rows only raise | both directions are scored on the right token |
| `test_probe_separates_separable_data` | unit | accuracy 1 on separable clusters far from the origin; folded weights work on raw features | the probe measures the features |
| `test_probe_is_deterministic_and_validates` | boundary | identical probes from identical data; bad shapes, labels, one class | probe accuracies are reproducible |
| `test_contextual_layers_disambiguate` | learning | after 150 steps: loss at the reference bar; probe on $R^0$ at the majority rate, on $R^1$ at least 0.95 | the reason contextual vectors exist |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a forward layer that also reads the backward states | the forward LM sees the future and its loss collapses to 0 | `test_forward_layers_never_see_the_future` (mutant `s03`) |
| 2. reversing the padded tensor, or not reversing back, or running the backward LSTMs without lengths | the backward LM reads padding first, or states land at the wrong positions | `test_padding_never_leaks` (mutant `s04`), `test_backward_layers_never_see_the_past` (mutants `s05`, `s06`) |
| 3. scoring a direction on the current token | each LM is trained to predict what it already read | `test_bilm_loss_targets` (mutants `s07`, `s08`) |
| 4. returning probe weights for standardized features | predictions on raw features are wrong | `test_probe_separates_separable_data` (mutant `s09`) |
| raw scalars as mix weights | negative or unnormalized layer weights | `test_hand_example_scalar_mix` (mutant `s01`) |
| $\gamma$ never applied | the task cannot rescale the mix | `test_hand_example_scalar_mix` (mutant `s02`) |
| gradient ascent in the probe | accuracy falls as it trains | `test_probe_separates_separable_data` (mutant `s10`) |

## 6. Where it's used next

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

No module calls this code (it is a side module, D31). It is still worth doing: the probe is the cheapest way to see what a network has learned, and the biLM's "each side only sees its own side" rule is exactly the causality property `L5.2` tests for transformer masks.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `BiLM` | AllenNLP's ELMo | a character CNN instead of a token embedding, projections and residual connections between layers, 4096-wide LSTMs | `allennlp/modules/elmo.py`, `elmo_lstm.py` |
| `ScalarMix` | AllenNLP `ScalarMix` | optional layer normalization of each layer before mixing | `allennlp/modules/scalar_mix.py` |
| `fit_linear_probe` | edge probing, control tasks | span-pair probes, selectivity against random control labels | Tenney et al. 2019; Hewitt and Liang 2019 |
| contextual vectors | BERT, sentence-transformers | one bidirectional encoder trained by masked LM, pooled sentence embeddings | `L6.2`; `ag.06`'s embedding endpoint |
