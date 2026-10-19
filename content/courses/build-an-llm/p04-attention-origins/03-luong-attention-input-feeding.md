# Luong attention and input feeding

## Overview

| | |
|---|---|
| **Module** | `L4.3` · build · Python · Pass 4 · 1 to 1.5 h reading |
| **Solution** | [`python/tinyllm/seq2seq/luong.py`](../../../../course/solutions/python/tinyllm/seq2seq/luong.py): `LuongAttention` with the `dot`, `general`, and `concat` scores (`project_keys`, `scores`, `forward`) and `attentional`, the state input feeding passes on; model oracle tests in [`test_luong.py`](../../../../course/solutions/learner-tests/L4.3/python/tests/l4-3-luong/test_luong.py) |
| **Contract** | [`course/contracts/py/tinyllm/seq2seq/luong.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/luong.pyi) |
| **Needs** | `L4.2` `length_mask` (and the masked read built there) · `L0.4` `Linear`, `Module` · `L0.2` ops · `L0.1` `Tensor` · `M06.3` `PCG32` · reading: `M09.2` |
| **Used by** | `L4.1` plugs it into the decoder with input feeding · `L4.4` decodes a Luong seq2seq in its tests · later `L6.7` the zoo's attention rows · on the dates task, a Luong seq2seq reaches an exact match of at least 0.97 |
| **Optional depth** | Luong, Pham, and Manning, "Effective Approaches to Attention-based Neural Machine Translation" (EMNLP 2015), sections 3.1 and 3.3; Britz et al., "Massive Exploration of Neural Machine Translation Architectures" (2017), on additive vs multiplicative scores |

## Key Takeaways

- Luong keeps Bahdanau's masked read (`L4.2`) and changes two things: the score function, and when the decoder asks, **after** its recurrent step.
- `dot` is $h \cdot k_s$, `general` the bilinear $h \cdot (W_a k_s)$ (with $W_a = I$ it is `dot`), `concat` a hidden layer on $[h ; k_s]$.
- The attentional state $\tilde h = \tanh(W_c [c ; h])$, context first, is what the output layer reads and what input feeding passes to the next step.
- The plain dot score is not scaled here; scaling by $1/\sqrt{d}$ is the transformer's change in `L5.1`.

## How to read this chapter

Sections 1 to 3 build Luong's scores and input feeding from first principles and work a `dot` example by hand. Section 4 fixes the interface and what a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Bahdanau's attention (`L4.2`) works, but every decoder step runs a hidden layer over every source position, and its query is the state from **before** the step, so the decoder chooses where to look before it has seen the token it was just given. Luong, Pham, and Manning simplified both: score with a dot product, and attend with the state just computed. They also noticed that the decoder forgets where it looked, and fixed that by feeding the attentional state back into the next step's input. The dot score is the one the transformer keeps, so this module is the last step before `L5.1`.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $d$ | width of both the decoder state and the keys | `int` |
| $h$ | the query: the decoder state after this step | `float32[B, d]` |
| $k_s$ | the encoder output at source position $s$ | `float32[B, S, d]` |
| $W_a$ | `score_proj.weight`: $[d, d]$ for `general`, $[d, 2d]$ for `concat` | `float32` |
| $v$ | `v.weight`, `concat` only | `float32[1, d]` |
| $W_c$ | `combine.weight` | `float32[d, 2d]` |
| $e_s$, $a_s$, $c$ | score, weight, context as in `L4.2` | |
| $\tilde h$ | the attentional state $\tanh(W_c [c ; h])$ | `float32[B, d]` |

### 2.1 Attend after the step

Bahdanau: read with $s_{t-1}$, feed the context into the cell. Luong: run the cell first to get $h_t$, then read with $h_t$, then combine:

$$h_t = \mathrm{cell}([y_{t-1} ; \tilde h_{t-1}], h_{t-1}), \quad c_t = \sum_s a_s k_s, \quad \tilde h_t = \tanh(W_c [c_t ; h_t]), \quad \text{logits}_t = W_o \tilde h_t + b_o .$$

The decoder (`L4.1`) does the cell and the output layer; this module provides the scores, the read, and $\tilde h$. The class attribute `query_from = "current"` tells the decoder which order to use.

### 2.2 Three scores

| Score | $e_s$ | Parameters |
|---|---|---|
| `dot` | $h \cdot k_s$ | none |
| `general` | $h \cdot (W_a k_s)$ | $W_a$ $[d, d]$ |
| `concat` | $v \cdot \tanh(W_a [h ; k_s])$ | $W_a$ $[d, 2d]$, $v$ |

`dot` needs $h$ and $k_s$ in the same space; `general` learns a bilinear form between them (with $W_a = I$ it is `dot`, and $W_a$ and $W_a^\top$ give different scores, so weights load only one way round); `concat` is Bahdanau's network with one weight matrix on the concatenation. Masking, the softmax, and the context are exactly `L4.2`'s: padded scores $-\infty$, exact zero weights, an empty row reads nothing.

### 2.3 Precomputing the keys

For `concat`, $W_a [h ; k_s] = W_h h + W_k k_s$ where $W_h$ is the first $d$ columns of $W_a$ and $W_k$ the last $d$. So $W_k k_s$ is computed once per sentence (`project_keys`), and each step adds $W_h h$. For `general`, `project_keys` is $W_a k_s$; for `dot`, the keys themselves.

### 2.4 Input feeding

Without it, the decoder at step $t$ does not know what it attended to at step $t - 1$; Luong's input feeding concatenates $\tilde h_{t-1}$ to the next input embedding ($\tilde h_0 = 0$). The model can then avoid translating the same source word twice, a coverage signal for free. It also makes the network deeper in time: the gradient of step $t$ reaches the attention of step $t - 1$.

## 3. Worked example by hand

$d = 2$, the `dot` score, $h = (1, 0)$, keys $k_1 = (1, 0)$, $k_2 = (0, 1)$, $k_3 = (2, 0)$, the third padding.

1. Scores: $h \cdot k = (1, 0, 2)$; after the mask $(1, 0, -\infty)$.
2. Weights: $e^1 / (e^1 + 1) = 0.731059$ and $1 / (e + 1) = 0.268941$, then 0.
3. Context: $c = 0.731059 (1, 0) + 0.268941 (0, 1) = (0.731059, 0.268941)$.
4. With $W_c = [I \; I]$ (so $W_c [c ; h] = c + h$): $\tilde h = \tanh(1.731059, 0.268941) = (0.939181, 0.262640)$.
5. With $W_c = [I \; 0]$ instead, $\tilde h = \tanh(c) = (\tanh 0.731059, \tanh 0.268941)$: the context half is the first $d$ columns.

A correct solution reproduces steps 1 to 4 exactly and gives $\tanh(c)$ in step 5.

## 4. The interface

```python
class LuongAttention(Module):
    query_from = "current"
    def __init__(self, d: int, score: Literal["dot", "general", "concat"], rng=None) -> None: ...
    def project_keys(self, keys: Tensor) -> Tensor: ...               # [B, S, d]
    def scores(self, query: Tensor, keys: Tensor, proj=None) -> Tensor: ...   # [B, S], unmasked
    def forward(self, query, keys, mask, proj=None) -> tuple[Tensor, Tensor]: ...  # context, weights
    def attentional(self, query: Tensor, context: Tensor) -> Tensor: ...  # tanh(W_c [c ; h])
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: `dot` | unit | section 3 steps 1 to 4 | pins down the read |
| hand example attentional order | unit | $W_c = [I \; 0]$ gives $\tanh(c)$ | context first, as Luong trained it |
| matches torch | golden | each score: weights, context, $\tilde h$, and every gradient against torch | the decoder trains through this |
| gradcheck of every input and parameter | gradcheck | float64 central differences, each score, a padded row | every parameter learns |
| `general` with identity is `dot` | property | `general` with $W_a = I$ equals `dot` | the relation between the scores |
| `general` is not symmetric | unit | $W_a$, not $W_a^\top$ | checkpoints load the right way round |
| `concat` projection splits the weight | property | `project_keys` and the score equal $v \cdot \tanh(W_a [h ; k])$ | the per-sentence cache is exact |
| masked weights are exactly zero | property | exact zeros, no gradient, an empty row reads nothing | padding is never read |
| `dot` is not scaled | unit | scores exactly $(1, 0, 2)$ | the scaled score belongs to `L5.1` |
| parameter names and shapes | unit | each score's keys and shapes, `combine` last, `query_from` | the safetensors keys of Luong checkpoints |
| validation | boundary | unknown score, wrong widths, wrong mask | wiring bugs fail loudly |

### The oracle tests

As in `L4.2`, the oracle in the model suite [`test_luong.py`](../../../../course/solutions/learner-tests/L4.3/python/tests/l4-3-luong/test_luong.py) is the formula in numpy with the module's own weights from `state_dict()`: all three scores, the masked softmax, the context, and $\tilde h$ in float64. It adds the section 3 numbers, a finite-difference gradient check of the keys, an empty row, and the parameter names, and imports only contract modules.

## 5. The solution, step by step

### From first principles

1. **The read is unchanged.** Weights must still be a masked softmax over positions and the context a weighted average of keys (`L4.2`). So `forward` is `L4.2`'s three lines with a different `scores`.
2. **A dot product is the cheapest score that couples query and key.** $h \cdot k_s$ is large when they point the same way; it needs no parameters but requires $h$ and $k_s$ to live in the same $d$-dimensional space, which is why the module has one width $d$.
3. **`general` adds a learned change of basis.** $h \cdot (W_a k_s)$ lets the model compare in a learned metric. Writing it as "project the keys, then dot" means the projection is per sentence and each step is still one dot per position. Since $h^\top W_a k \ne h^\top W_a^\top k$ in general, which side carries $W_a$ is part of the format.
4. **`concat` is Bahdanau with one matrix, and the matrix splits.** $W_a [h ; k] = W_h h + W_k k$ with $W_h$ the first $d$ columns and $W_k$ the last $d$. The key half is cached; the query half is computed each step.
5. **The output and the next input need what was read.** $\tilde h = \tanh(W_c [c ; h])$ mixes the context into the state; context first is Luong's convention and therefore the checkpoint layout. Feeding $\tilde h$ into the next input is the decoder's job (`L4.1`); this module only provides `attentional`.
6. **No $1/\sqrt{d}$.** Luong's dot score is unscaled. In the worked example the scores are exactly $(1, 0, 2)$; scaling them would change every weight. The scale is the transformer's addition (`L5.1`).

### The code, annotated

Parameters depend on the score; `combine` ($W_c$) is always present and registered last:

```python
class LuongAttention(Module):
    query_from = "current"             # L4.1: step the cell first, then attend with h_t

    def __init__(self, d, score, rng=None):
        # ... (d >= 1, score in SCORES, default rng)
        if score == "general":
            self.score_proj = Linear(d, d, bias=False, rng=r)        # W_a [d, d]
        elif score == "concat":
            self.score_proj = Linear(2 * d, d, bias=False, rng=r)    # W_a [d, 2d]
            self.v = Linear(d, 1, bias=False, rng=r)
        self.combine = Linear(2 * d, d, bias=False, rng=r)           # W_c [d, 2d]
```

The per-sentence part of each score:

```python
def project_keys(self, keys):
    self._check_keys(keys)
    if self.score == "dot":
        return keys                                   # nothing to precompute
    if self.score == "general":
        return self.score_proj(keys)                  # W_a k_s for every position
    # concat: W_a [h ; k] = W_h h + W_k k, and W_k k does not depend on h.
    w_k = self.score_proj.weight[:, self.d :]         # the key half: last d columns
    return F.matmul(keys, F.transpose(w_k, 0, 1))
```

The per-step part. For `dot` and `general` the score is one batched matrix-vector product:

```python
def scores(self, query, keys, proj=None):
    # ... (shape checks; proj = self.project_keys(keys) if not given)
    B, S, d = keys.shape
    if self.score in ("dot", "general"):
        # One dot product per position: [B, S, d] @ [B, d, 1].  No 1/sqrt(d).
        e = F.matmul(proj, F.reshape(query, (B, d, 1)))
        return F.reshape(e, (B, S))
    w_h = self.score_proj.weight[:, :d]               # the query half: first d columns
    hq = F.reshape(F.matmul(query, F.transpose(w_h, 0, 1)), (B, 1, d))
    e = self.v(F.tanh(proj + hq))                     # v . tanh(W_h h + W_k k_s)
    return F.reshape(e, (B, S))
```

Slicing `score_proj.weight` with getitem keeps both halves in the graph, so the one `[d, 2d]` parameter gets its full gradient from both uses.

The read is `L4.2`'s, and the attentional state is one line:

```python
def forward(self, query, keys, mask, proj=None):
    # ... (key and mask checks)
    e = F.masked_fill(self.scores(query, keys, proj), ~m.astype(bool), -np.inf)
    a = F.softmax(e, axis=-1)                         # padded weights exactly 0
    ctx = F.matmul(F.reshape(a, (B, 1, S)), keys)
    return F.reshape(ctx, (B, d)), a

def attentional(self, query, context):
    # ... (both [B, d])
    # Luong's order: the context first, then the decoder state.
    return F.tanh(self.combine(F.concat([context, query], axis=-1)))
```

Full solution: [`luong.py`](../../../../course/solutions/python/tinyllm/seq2seq/luong.py).

### Why this solution

- **Cost per step.** `dot`/`general`: $O(B S d)$ for the scores, with `general`'s $O(B S d^2)$ projection paid once per sentence. `concat`: $O(B d^2)$ for $W_h h$ plus $O(B S d)$ for the $\tanh$ and $v$. Bahdanau's per-step cost without caching would be $O(B S d^2)$.
- **Why one `[d, 2d]` matrix for `concat` and not two `Linear`s?** It matches Luong's formulation and the checkpoint layout (`score_proj.weight`); splitting by slicing gives the same computation and the same cache without changing the key names.
- **Why attend after the step?** With $h_t$ as query, the decoder has already absorbed $y_{t-1}$ when it decides where to look; with $s_{t-1}$ (Bahdanau) it chooses before seeing it. Input feeding compensates for what is lost: the next step still sees where this one looked.
- **Tradeoff.** Unscaled dot scores grow with $d$, so for large $d$ the softmax saturates and gradients vanish. That is acceptable at seq2seq widths and is exactly the problem `L5.1`'s $1/\sqrt{d}$ fixes.

### Explain it back

1. Why does `general` with $W_a = I$ give exactly `dot`, and why does $W_a^\top$ give something different?
   <details><summary>Answer</summary>With $W_a = I$, `project_keys` returns the keys and the score is $h \cdot k_s$. In general $h^\top W_a k \ne h^\top W_a^\top k$ unless $W_a$ is symmetric, so a checkpoint trained with one convention scores differently under the other.</details>
2. In `concat`, why can the key half be precomputed but not the query half?
   <details><summary>Answer</summary>$W_a [h ; k] = W_h h + W_k k$ is linear before the $\tanh$. $W_k k_s$ depends only on the sentence; $W_h h$ changes every step. The $\tanh$ is applied after the sum, so it must stay per step.</details>
3. In step 5 of the worked example, why does $W_c = [I \; 0]$ give $\tanh(c)$ and not $\tanh(h)$?
   <details><summary>Answer</summary>`attentional` concatenates `[context, query]`, so the first $d$ columns of $W_c$ multiply $c$. With $[I \; 0]$ only the context passes through.</details>
4. What would input feeding give at step 1, and where is it set up?
   <details><summary>Answer</summary>$\tilde h_0 = 0$. `L4.1`'s `init_state` creates a zero `feed` when `query_from == "current"`, and each `decode_step` replaces it with the new `attentional(h, ctx)`.</details>
5. Why is the dot score not divided by $\sqrt{d}$ here?
   <details><summary>Answer</summary>Luong's model is unscaled, and weights trained for it expect raw scores; the worked example's weights $(0.731, 0.269)$ come from scores $(1, 0)$. Scaling is a separate design decision made by the transformer (`L5.1`).</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. $W_c [h ; c]$ instead of $W_c [c ; h]$ | Luong-trained weights decode garbage | `attentional` concatenates `[context, query]`, context first |
| 2. `general` with $W_a^\top$ | the bilinear form is transposed | `general` projects the keys with `score_proj(keys)`, i.e. $W_a k_s$, and dots with $h$ |
| 3. masking after the softmax | real weights do not sum to 1 | `masked_fill(..., -np.inf)` on the scores, then the softmax |
| 4. scaling the dot score by $1/\sqrt{d}$ | every weight differs from Luong's | `dot` scores are the raw `matmul` of keys and query |
| 5. `concat` precomputing with the query half of $W_a$ | the cached keys are multiplied by the wrong columns | `project_keys` uses `weight[:, d:]` (the key half); `scores` uses `weight[:, :d]` for the query |
| masking with $-10^9$ | an empty row reads padding uniformly | fills with `-np.inf`, never a finite number |
| $\tilde h$ without the `tanh` | the attentional state is unbounded | `F.tanh(self.combine(...))` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L4.2` | the masked read and `length_mask` |
| Back | `L0.4` | `Linear` holds $W_a$, $v$, $W_c$ |
| Back | `L0.2` | the op library gives the backward |
| Back | `L0.1` | `Tensor` |
| Back | `M06.3` | `PCG32` for the default initialization |
| Forward | `L4.1` | the decoder steps first, attends with the new state, and feeds $\tilde h$ into the next input |
| Forward | `L4.4` | its tests beam-search a Luong seq2seq, whose state carries `feed` |
| Forward | `L5.1` | scaled dot-product attention: Luong's `dot` divided by $\sqrt{d}$, over many heads |
| Forward | `L6.7` | the zoo's Luong rows |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `dot` score | `F.scaled_dot_product_attention` | the $1/\sqrt{d}$ scale, heads, causal masks, fused kernels | `L5.1`; FlashAttention |
| input feeding | the attentional decoder of OpenNMT | the same `input_feed` option, with coverage and copy attention | OpenNMT-py `onmt/decoders/decoder.py` (`InputFeedRNNDecoder`) |
| `general` score | bilinear attention in readers and parsers | the biaffine scorer of dependency parsers | Dozat and Manning, "Deep Biaffine Attention" (2017) |
| local attention | Luong's local-p attention | reads a window around a predicted position: cost per step independent of $S$ | Luong et al. 2015, section 3.2; sliding-window attention in `L7.7` |
