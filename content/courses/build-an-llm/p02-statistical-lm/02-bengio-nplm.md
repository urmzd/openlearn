<!-- ol:module L2.2 -->
# Bengio's neural probabilistic language model

## Overview

| | |
|---|---|
| **Module** | `L2.2` · build · Python · Pass 3 · 4 to 5 h, plus your graded tests (rung R3) |
| **You build** | `python/tinyllm/lm/nplm.py`: `windows`, `NPLM(vocab, context, d_emb, d_hidden, direct, rng)` with `forward`, `nll`, `perplexity`, `generate`; `train_nplm`; `save_nplm` and `load_nplm` (the model directory); and your own tests in `python/tests/l2-2-nplm/`, written first |
| **Contract** | [`course/contracts/py/tinyllm/lm/nplm.pyi`](../../../../course/contracts/py/tinyllm/lm/nplm.pyi) |
| **Tests** | `course/tests/L2.2/test_nplm.py` (what they check: section 4); golden values from the same architecture written in torch, `course/oracle/L2.2/nplm_torch.py`; the learning test trains on the MS-L2 corpus; your tests are graded by mutation, threshold 0.70 plus one required fault, with a red-then-green journal |
| **Needs** | `L0.1` `Tensor`, `no_grad` · `L0.2` `F.reshape`, `F.tanh`, `F.embedding` · `L0.3` `cross_entropy` · `L0.4` `Module`, `Embedding`, `Linear` · `L0.5` `DataLoader`, `train_step` · `L0.6` `save_safetensors`, `load_safetensors`, `open_tokens` · `M10.2` `SGD` · `M11.2` `NLLAccumulator` · `M06.3` `PCG32` (or `--ref-deps`). Reading: `L2.1` (the count model this one beats), `L0.0` |
| **Used by** | later `L6.7`: the model zoo loads the NPLM directory and reports its bits per byte (it joins the registry with its batch) |
| **Milestone** | `MS-L2` (`{tinyllm} lm train nplm`, then `{tinyllm} eval`: perplexity at the calibrated bar and below the bigram; `{tinyllm} generate` twice with one seed gives one story) |
| **Optional depth** | Bengio, Ducharme, Vincent, and Jauvin, "A Neural Probabilistic Language Model" (JMLR 2003), sections 1 to 3; Goodfellow, Bengio, Courville, *Deep Learning*, section 12.4; Jurafsky and Martin, *Speech and Language Processing*, 3rd ed., ch. 7 |

## Key Takeaways

- An NPLM replaces the count table with a learned embedding per token and a small network over a fixed window, so a window never seen in training still gets a sensible prediction from the windows it resembles (`test_nplm_learns_the_corpus`).
- The window is concatenated **oldest first** and the embedding lookup is an autograd op: the order is part of the parameter layout, and a lookup that bypasses the graph leaves the table untrained (`test_hand_example_forward`, `test_gradcheck_every_parameter`).
- The whole model is ordinary layers, so the forward is one line of math and backward is free; torch, given the same weights, computes the same logits, loss, and gradients (`test_golden_torch`).
- Training is exactly the L0.5 loop with one SGD across epochs, and every random choice comes from one PCG32 per purpose, so a seed reproduces the run bit for bit (`test_train_nplm_is_the_l05_loop`, `test_init_draws_from_one_generator`).
- The model directory (config plus F32 safetensors) is what your CLI writes, `generate` reads, and the zoo loads (`test_save_load_roundtrip`).

## How to work this chapter

```bash
ol start L2.2              # stubs nplm.py; prints your test path and rung (R3)
ol tests L2.2              # the course tests
# write ONE test in python/tests/l2-2-nplm/ (start with section 4's), then:
ol tdd red L2.2            # must FAIL against your current code: records the red
# make it pass, then:
ol tdd green L2.2          # must PASS with the same test files: records the green
# repeat for each test; then:
ol check L2.2              # course tests, the red-then-green journal, the mutation grade
ol mutate L2.2             # the full grade, cached by your test files' hash
ol check L2.2 --ref-deps   # only if an L0 module is not passing yet
```

Your CLI gains `lm train nplm` and `eval` (and `generate` learns to read an `nplm` directory); `course/milestones/MS-L2.toml` fixes their flags and final lines.

---

## 1. Why now

Your Kneser-Ney model (`L2.1`) is the best a count table can do, and it hits a wall that smoothing cannot move. To a count table, "the cat ran to the park" and "the dog ran to the park" are unrelated strings: everything learned about one tells it nothing about the other, so it needs to see every word in every context. More context makes this exponentially worse: a byte 8-gram has $256^8 \approx 1.8 \times 10^{19}$ possible histories, and almost all the ones you meet at test time were never counted. Bengio's 2003 answer is the idea every later model in this course keeps: give each token a learned vector, and let a network predict from the vectors of the window. Tokens that appear in similar windows get similar vectors, so the model generalizes across them. It is also the first model you train with your own autograd (`L0.1` to `L0.4`) and training loop (`L0.5`) on real text, and on the MS-L2 corpus it beats the 4-gram (perplexity about 1.54 against 1.87) and the count bigram (8.52) by a wide margin.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size; tokens are ids $0 \dots V-1$ (256 bytes in MS-L2) | `int` |
| $n$ | the order: the model sees the $n - 1$ previous tokens | `int` |
| $c = n - 1$ | the window (`context`) | `int` |
| $m$ | embedding size (`d_emb`) | `int` |
| $h$ | hidden size (`d_hidden`) | `int` |
| $B$ | batch size: windows per step | `int` |
| $C$ | embedding table, row $C[w]$ is token $w$'s vector | `float32[V, m]` |
| $x$ | the window's embeddings concatenated, oldest first | `float32[B, c m]` |
| $H, d$ | hidden layer weight and bias | `float32[h, c m]`, `float32[h]` |
| $U, b$ | output weight and bias | `float32[V, h]`, `float32[V]` |
| $W$ | direct path from the embeddings to the logits (optional) | `float32[V, c m]` |
| $y$ | logits | `float32[B, V]` |
| $p = \mathrm{softmax}(y)$ | the predicted distribution of the next token | `float32[B, V]` |
| $\mathcal{L}$ | mean cross-entropy over the batch, in nats | scalar |
| $\eta$, $\mu$ | SGD learning rate and momentum (`M10.2`) | `float` |

### 2.1 From counts to vectors

An n-gram model estimates $P(w_t \mid w_{t-c} \dots w_{t-1})$ with one free number per (history, token) pair: a table with $V^{c} \times V$ entries, nearly all of them estimated from zero data. A neural model shares parameters across histories. Each token gets one vector $C[w] \in \mathbb{R}^m$, used in every window where the token appears, and one function maps a window of vectors to a distribution. The parameter count is now $V m + (\text{network})$, linear in $V$ and in $c$, instead of exponential in $c$. If "cat" and "dog" end up with nearby vectors (because they appear in similar windows), then every window containing one is also evidence about windows containing the other.

### 2.2 The architecture

Bengio et al. write the model as one equation:

$$x = [C[w_{t-c}]; \dots; C[w_{t-1}]], \qquad y = b + W x + U \tanh(d + H x), \qquad p = \mathrm{softmax}(y).$$

Read it from the inside out. Look up the $c$ rows of $C$ for the window and concatenate them, oldest first, into one vector of length $c m$; that is exactly a row-major reshape of the `[B, c, m]` lookup to `[B, c m]`. Apply one hidden layer with a $\tanh$ nonlinearity. Map the hidden vector to $V$ logits. The paper also adds $W x$, a **direct** linear path from the embeddings to the logits: a log-linear model in parallel with the hidden layer, which the paper found speeds up training. In the course's layers: `emb = Embedding(V, m)`, `hidden = Linear(c m, h)`, `out = Linear(h, V)`, and `direct = Linear(c m, V, bias=False)`. Those attribute names, in that order, are the state-dict keys and so the safetensors keys of the model directory.

The order of the concatenation is part of the model. Position $j$ of the window always multiplies the same columns $H[:, jm:(j+1)m]$, so the network learns what the token two back means differently from the token one back. Concatenating newest first is not wrong in itself, but it is a different parameter layout: weights trained one way are garbage read the other way, and a checkpoint is no longer interchangeable with anyone else's.

### 2.3 Training

The loss is the mean cross-entropy (`L0.3`) of each window's logits against the next token. Its gradient at the logits is $p - \mathrm{onehot}(\text{target})$ per row, divided by $B$; everything below the logits is the chain rule through `Linear`, `tanh`, the reshape, and the embedding lookup, which your autograd already does. The one place a gradient can silently vanish is the lookup: `F.embedding` is an op whose backward adds each row's gradient into the rows it read (a token used twice in a window gets both). Reading `emb.weight.data[ids]` with numpy instead gives the same forward numbers and a constant: $C$ never learns.

`train_nplm` is the `L0.5` loop with nothing new. The windows of the whole sequence become arrays `x` (`int64[N, c]`) and `y` (`int64[N]`); a `DataLoader` shuffles them with the given PCG32 (a new permutation each epoch) in batches of $B$, dropping the last partial batch; one `SGD` (`M10.2`) is created before the first step, so its momentum buffers carry across epochs; and each step is `train_step` with the given `clip`. It runs for exactly `steps` steps, wrapping into new epochs as needed.

### 2.4 Scoring

`nll(ids)` scores every token that has a full window: positions $c \dots \text{len} - 1$. Each value is $\mathrm{logsumexp}(y) - y_{\text{target}}$ in float64, from float32 logits, computed under `no_grad` in batches (batching changes nothing but rounding). `perplexity` is $\exp$ of the mean, summed with `M11.2`'s `NLLAccumulator`; with byte tokens, bits per byte is the mean divided by $\ln 2$. The first $c$ tokens of a file are not scored: an n-gram model predicts them from `<s>`, the NPLM has no start symbol, so on the MS-L2 validation file the NPLM scores 14 667 tokens where the 4-gram scores 14 675. Over that many tokens the difference does not move the comparison.

### 2.5 Generation and the model directory

`generate(prefix, n, temperature, seed)` repeats: take the last $c$ ids, compute the logits, choose the next id, append it. At temperature 0 it is the first maximum (ties to the lowest id, D11). Otherwise it draws exactly as `L0.5`'s `BigramLM.sample`: one `PCG32(seed)` (stream 54) per call, one `uniform()` per token, weights $e^{z_j - \max z}$ of $z = y / T$ in float64 summed in id order, and the first id whose running sum exceeds $u \cdot \sum$. A prompt shorter than the window is an error; a longer prompt conditions only on its end.

`save_nplm(model, dir)` writes `config.json` (`tl_arch = "nplm"`, the sizes as `tl_context`, `tl_d_emb`, `hidden_size`, `tl_direct`) and `model.safetensors` (the state dict as F32, metadata `{"format": "tinyllm", "tl_arch": "nplm"}`) with `L0.6`'s writer; `load_nplm` rebuilds the model from the config and loads the tensors strictly.

## 3. Worked example by hand

$V = 3$ tokens `a`, `b`, `c` (ids 0, 1, 2), a window of $c = 2$, embedding size $m = 1$, one hidden unit $h = 1$, with the direct path:

$$C = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}, \quad H = \begin{pmatrix} 0.5 & -0.5 \end{pmatrix}, \quad d = 0, \quad U = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}, \quad b = \begin{pmatrix} 0 \\ 0.5 \\ 0 \end{pmatrix}, \quad W = \begin{pmatrix} 1 & 0 \\ 0 & 0 \\ 0 & 1 \end{pmatrix}.$$

**Windows.** The sequence 5 6 7 8 with $c = 2$ gives the windows (5, 6) with target 7 and (6, 7) with target 8 (`test_windows_hand_example`); the first two tokens are never targets.

**Forward for the window (a, c).** Oldest first: $x = (C[a], C[c]) = (1, -1)$. Hidden: $H x + d = 0.5 \cdot 1 - 0.5 \cdot (-1) = 1$, so the hidden activation is $\tanh 1 = 0.761594$. Direct path: $W x = (1, 0, -1)$. Output path: $U \tanh 1 = (0.761594, 0, -0.761594)$. Logits:

$$y = b + W x + U \tanh 1 = (1.761594,\ 0.5,\ -1.761594).$$

The reversed window (c, a) has $x = (-1, 1)$, hidden pre-activation $-1$, and logits $(-1.761594, 0.5, 1.761594)$: the mirror image. The order matters (`test_hand_example_forward`). Without the direct path only $b + U \tanh 1 = (0.761594, 0.5, -0.761594)$ remains.

**Probabilities and loss.** $e^{y} = (5.821711, 1.648721, 0.171771)$, sum $7.642203$, so $p = (0.761784, 0.215739, 0.022477)$. With target `a` the loss is $-\ln 0.761784 = 0.272092$ nats.

**Gradient at the output bias.** $\partial \mathcal{L} / \partial y = p - \mathrm{onehot}(a) = (-0.238216, 0.215739, 0.022477)$, and since $y = b + \dots$, that is exactly the gradient of $b$ (`test_hand_example_gradient_reaches_the_bias`). Going one layer down, the hidden unit receives $\sum_v (p_v - \mathrm{onehot}_v) U_v = -0.238216 - 0.022477 = -0.260692$, times $1 - \tanh^2 1 = 0.419974$ gives $-0.109484$ at the pre-activation; from there $H$, and through both paths $C[a]$ and $C[c]$, receive their shares. $C[b]$ was not in the window: its gradient is exactly 0.

## 4. The interface

```python
def windows(ids: ArrayLike, context: int) -> tuple[NDArray, NDArray]: ...   # ([N, c], [N]), N = len - c

class NPLM(Module):
    def __init__(self, vocab: int, context: int, d_emb: int, d_hidden: int,
                 direct: bool = True, rng=None) -> None: ...                 # emb, hidden, out, direct
    def forward(self, ctx_ids: ArrayLike) -> Tensor: ...                      # [B, c] -> [B, V] logits
    def nll(self, ids: ArrayLike, batch_size: int = 1024) -> NDArray: ...     # float64 [len - c]
    def perplexity(self, ids: ArrayLike) -> float: ...
    def generate(self, prefix, n, temperature, seed) -> list[int]: ...

def train_nplm(model, ids, steps, batch_size, lr, rng, momentum=0.0,
               weight_decay=0.0, clip=None) -> list[float]: ...               # one loss per step
def save_nplm(model: NPLM, dir: str, tokenizer: str = "bytes") -> None: ...
def load_nplm(dir: str) -> NPLM: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_windows_hand_example` | unit, smoke | (5, 6) -> 7 and (6, 7) -> 8; `context` 1 too | every training example and every scored token comes from here |
| `test_hand_example_forward` | unit, smoke | section 3's logits and the mirrored window; the model without the direct path | the parameter layout every checkpoint shares |
| `test_hand_example_gradient_reaches_the_bias` | unit | loss $0.272092$, $\partial b = p - \mathrm{onehot}$, $C[b]$ untouched | you and the tests agree on the loss before any training |
| `test_golden_torch` | golden | logits, loss, and every parameter's gradient against torch, with and without the direct path; ids repeat in and across windows | torch is the oracle for every later model (`L3`, `L5`) |
| `test_gradcheck_every_parameter` | gradcheck | the frozen central differences in float64 for all six parameters | a table cut out of the graph never learns |
| `test_state_dict_names_and_shapes` | unit | keys and shapes in registration order, F32, `direct.weight` only with the direct path | the zoo (`L6.7`) loads by these names |
| `test_init_draws_from_one_generator` | unit | layers drawn from one rng in order; two default models equal; no repeated draws across layers | a seed fixes the whole model |
| `test_nll_matches_cross_entropy_per_window` | unit | per-token NLL in float64, batch size changes only rounding, mean = the loss, perplexity = exp(mean) | MS-L2's `eval` and the zoo's bits per byte |
| `test_generate_seeded_and_greedy` | unit | the contract's draw rebuilt with the frozen PCG32; same seed, same ids; greedy is the step-by-step argmax | MS-L2 generates twice and compares |
| `test_greedy_ties_go_to_the_lowest_id` | boundary | an all-zero model decodes greedily to id 0 every step | the D11 tie rule every engine shares |
| `test_generate_uses_the_last_window` | boundary | a long prompt conditions on its end | prompts are longer than the window |
| `test_train_nplm_is_the_l05_loop` | differential | 10 steps over 4-batch epochs with momentum and weight decay: equal bit for bit to the loop rebuilt from `L0.5` and `M10.2` | the training is reproducible from a seed |
| `test_train_nplm_runs_exactly_steps` | boundary | data of exactly one batch: 3 steps are 3 epochs | `steps` counts optimizer steps, not epochs |
| `test_train_nplm_passes_the_clip` | unit | with clip 0.5 the run equals the rebuilt loop with the same clip | long runs survive one bad batch |
| `test_nplm_learns_the_corpus` | learning | 300 steps on 50 000 bytes reach the reference's validation perplexity (mean + 3 sd over 5 seeds), below the bigram's 8.52 | the model actually learns |
| `test_save_load_roundtrip` | unit | `config.json` exactly, F32 tensors with the metadata, identical logits after reload, with and without the direct path | `{tinyllm} generate` and the zoo load what `lm train nplm` wrote |
| `test_input_validation` | boundary | sizes below 1, wrong window width, ids outside $0 \dots V-1$, short prompts, too few windows, bad tokenizer, wrong `tl_arch` | caller bugs fail at the call |

### Your graded tests (rung R3)

Rung R3 gives you the interface (above) and one test; you write the rest **before** the code that makes each pass. The given test:

```python
# python/tests/l2-2-nplm/test_nplm.py
import math
import numpy as np
from tinyllm.lm.nplm import NPLM
from tinyllm.num.rng import PCG32

def test_hand_example_logits():
    """Window (a, c): x = (1, -1), h = tanh 1, logits (1 + tanh 1, 0.5, -1 - tanh 1)."""
    m = NPLM(3, 2, 1, 1, rng=PCG32(0))
    m.load_state_dict({"emb.weight": [[1.0], [0.0], [-1.0]], "hidden.weight": [[0.5, -0.5]],
                       "hidden.bias": [0.0], "out.weight": [[1.0], [0.0], [-1.0]],
                       "out.bias": [0.0, 0.5, 0.0], "direct.weight": [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]})
    t = math.tanh(1.0)
    np.testing.assert_allclose(m(np.array([[0, 2]])).data, [[1 + t, 0.5, -1 - t]], rtol=1e-6)
```

Then, one at a time, red then green: the windows of a short sequence; every parameter receiving a gradient after one backward, the table included; a finite-difference check of a few entries of the embedding gradient; the state-dict names and the one-generator rule; `nll` against `cross_entropy` on the same windows; greedy generation from an all-zero model (ties) and a long prompt against its last window; `train_nplm` against the loop rebuilt from `DataLoader`, `train_step`, and `SGD` across an epoch boundary; a loss that drops on a repeating pattern; a save and load roundtrip with and without the direct path. Import only contract names. `ol check L2.2` requires, for every test file, a `ol tdd red` record before its last `ol tdd green`, a mutation score of at least 0.70, and the required fault killed: it is pitfall 1.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the window's rows read from `emb.weight.data` instead of through `F.embedding` | the forward is right, the table gets no gradient and never learns (and numpy reads id -1 as the last row) | `test_gradcheck_every_parameter`, `test_golden_torch` (mutant `s01`) |
| 2. the window concatenated newest first | another parameter layout: torch's weights and every saved checkpoint read wrong | `test_hand_example_forward`, `test_golden_torch` (mutant `s02`) |
| 3. the direct path registered but never added to the logits | the parameter exists, saves, and loads, and has no effect | `test_hand_example_forward`, `test_golden_torch` (mutant `s03`) |
| 4. `rng=None` passed on to every layer | each layer restarts the default stream: correlated initial weights | `test_init_draws_from_one_generator` (mutant `s04`) |
| 5. targets shifted by one (predicting the last window token) | a copy task: training loss collapses, nothing is learned about the next token | `test_windows_hand_example`, `test_nll_matches_cross_entropy_per_window` (mutant `s05`) |
| 6. generation conditioned on the first ids of the prompt | every step sees the same window: the text loops on the prompt's start | `test_generate_uses_the_last_window` (mutant `s06`) |
| a new `SGD` every epoch | the momentum buffers reset at each epoch boundary | `test_train_nplm_is_the_l05_loop` (mutant `s07`) |
| training stops at the end of the first epoch | fewer steps than asked, silently | `test_train_nplm_is_the_l05_loop`, `test_train_nplm_runs_exactly_steps` (mutant `s08`) |
| the direct matrix not saved | the reloaded model cannot load strictly | `test_save_load_roundtrip` (mutant `s09`) |
| greedy ties broken toward the highest id | a different text from every other engine (D11) | `test_greedy_ties_go_to_the_lowest_id` (mutant `s10`) |
| `clip` not passed to `train_step` | long runs blow up on one bad batch | `test_train_nplm_passes_the_clip` (mutant `s11`) |
| perplexity returned as the mean NLL | a number in nats reported as a perplexity | `test_nll_matches_cross_entropy_per_window` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | `Tensor` carries the gradients; `no_grad` for scoring and generation |
| Back | `L0.2` | `F.embedding` (the lookup with a backward), `F.reshape`, `F.tanh` |
| Back | `L0.3` | `cross_entropy` is the training loss |
| Back | `L0.4` | `Module`, `Embedding`, `Linear`: the layers and their state-dict names |
| Back | `L0.5` | `DataLoader` and `train_step` are the whole training loop |
| Back | `L0.6` | `save_safetensors`, `load_safetensors` for the model directory; `open_tokens` reads the corpus |
| Back | `M10.2` | `SGD` with momentum and weight decay |
| Back | `M11.2` | `NLLAccumulator` for perplexity and bits per byte |
| Back | `M06.3` | `PCG32`: the default init stream and the sampler |
| Back | `L2.1` | the count model this one is compared with in MS-L2 (reading) |
| Forward | `L6.7` | the zoo loads the NPLM directory and reports its bits per byte next to KN-4 and the transformers |
| Forward | `L3.6` | the recurrent LM replaces the fixed window with a state; MS-L3 requires its bits per byte to beat this model's on the same file |

If you skip this module, `L6.7` stops with `BLOCKED ... needs L2.2` once it lands: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a full softmax over $V$ | hierarchical softmax (Morin and Bengio 2005), adaptive softmax (Grave et al. 2017) | the output layer costs $O(\log V)$ or a few clusters instead of $O(V)$ per token | `torch.nn.AdaptiveLogSoftmaxWithLoss` (`torch/nn/modules/adaptive.py`) |
| fixed window of $c$ tokens | recurrent and attention models | unbounded context with a state, or every earlier token at once | `L3.6` (RNN LM) and `L5.1` (attention) in this course |
| embedding table trained with the model | word2vec and GloVe (`L2.3`), then tied input and output embeddings (Press and Wolf 2017) | the table pretrained on more text, or shared with the output layer to halve the parameters | `transformers` `tie_word_embeddings` |
| the direct path $W x$ | residual connections (He et al. 2016) | an identity-like shortcut around the nonlinearity, now in every transformer block | `L5.5` in this course |
| CPU numpy, one process | the paper's parallel training on 40 CPUs | data-parallel and parameter-parallel SGD over the output layer | Bengio et al. (2003), section 3 |
