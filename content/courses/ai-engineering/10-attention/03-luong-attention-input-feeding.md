# Luong attention and input feeding

[Bahdanau's attention](02-bahdanau-additive-attention.md) works, but every decoder step runs a hidden layer over every source position, and its query is the state from **before** the step, so the decoder chooses where to look before it has seen the token it was just given. Luong, Pham, and Manning simplified both: score with a dot product, and attend with the state just computed. They also noticed that the decoder forgets where it looked, and fixed that by feeding the attentional state back into the next step's input. The dot score is the one the transformer keeps, so this section is the last step before [scaled dot-product attention](../11-transformer/01-scaled-dot-product-attention.md).

You need the masked read and `length_mask` from the previous section, plus `Linear` and `Module` from [the module system](../06-numerics-and-autograd/05-module-system-and-layers.md), [the op library](../06-numerics-and-autograd/03-op-library-and-gradcheck.md), the `Tensor` of [the tensor section](../06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md), and `PCG32` from [Modular arithmetic, hashing, and PCG32](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics*. [Stable numerics](../../mathematics/15-numerical-methods-and-floating-point/02-stable-numerics.md) in the same book is background for the masked softmax.

The solution is [`python/tinyllm/seq2seq/luong.py`](../../../../course/solutions/python/tinyllm/seq2seq/luong.py): `LuongAttention` with the `dot`, `general`, and `concat` scores (`project_keys`, `scores`, `forward`) and `attentional`, the state that input feeding passes on. It implements the contract [`course/contracts/py/tinyllm/seq2seq/luong.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/luong.pyi), and its oracle tests are in [`test_luong.py`](../../../../course/solutions/learner-tests/L4.3/python/tests/l4-3-luong/test_luong.py). Plugged into the [encoder-decoder](01-encoder-decoder-teacher-forcing.md), a Luong seq2seq reaches an exact match of at least 0.97 on the dates task.

> **Learning objectives**
> - Explain how Luong's step order (attend after the cell) differs from Bahdanau's, and what input feeding restores.
> - Implement the `dot`, `general`, and `concat` scores, and state how they relate.
> - Split `concat`'s weight into a cached key half and a per-step query half.
> - Compute the attentional state $\tilde h = \tanh(W_c [c ; h])$ with the context first, and explain why the dot score is left unscaled.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $d$ | width of both the decoder state and the keys | `int` |
| $h$ | the query: the decoder state after this step | `float32[B, d]` |
| $k_s$ | the encoder output at source position $s$ | `float32[B, S, d]` |
| $W_a$ | `score_proj.weight`: $[d, d]$ for `general`, $[d, 2d]$ for `concat` | `float32` |
| $v$ | `v.weight`, `concat` only | `float32[1, d]` |
| $W_c$ | `combine.weight` | `float32[d, 2d]` |
| $e_s$, $a_s$, $c$ | score, weight, context, as in Bahdanau attention | |
| $\tilde h$ | the attentional state $\tanh(W_c [c ; h])$ | `float32[B, d]` |

## Attend after the step

Luong keeps Bahdanau's masked read and changes two things: the score function, and when the decoder asks. Bahdanau reads with $s_{t-1}$ and feeds the context into the cell. Luong runs the cell first to get $h_t$, then reads with $h_t$, then combines:

$$h_t = \mathrm{cell}([y_{t-1} ; \tilde h_{t-1}], h_{t-1}), \quad c_t = \sum_s a_s k_s, \quad \tilde h_t = \tanh(W_c [c_t ; h_t]), \quad \text{logits}_t = W_o \tilde h_t + b_o .$$

With $h_t$ as the query, the decoder has already absorbed $y_{t-1}$ when it decides where to look. The decoder does the cell and the output layer; this module provides the scores, the read, and $\tilde h$. The class attribute `query_from = "current"` tells the decoder which order to use.

The read itself is unchanged. Weights must still be a masked softmax over positions and the context a weighted average of keys: padded scores $-\infty$, exact zero weights, an empty row reads nothing. So `forward` is Bahdanau's three lines with a different `scores`.

## Three scores

Luong proposes three score functions, which differ in parameters and cost:

| Score | $e_s$ | Parameters |
|---|---|---|
| `dot` | $h \cdot k_s$ | none |
| `general` | $h \cdot (W_a k_s)$ | $W_a$ $[d, d]$ |
| `concat` | $v \cdot \tanh(W_a [h ; k_s])$ | $W_a$ $[d, 2d]$, $v$ |

A dot product is the cheapest score that couples query and key. $h \cdot k_s$ is large when they point the same way, and it needs no parameters, but it requires $h$ and $k_s$ to live in the same $d$-dimensional space, which is why the module has a single width $d$.

`general` adds a learned change of basis: $h \cdot (W_a k_s)$ compares the two in a learned metric. With $W_a = I$ it is `dot`. Since $h^\top W_a k \ne h^\top W_a^\top k$ unless $W_a$ is symmetric, which side carries $W_a$ is part of the checkpoint format, and weights load only one way round. Writing it as "project the keys, then dot" means the projection is per sentence and each step is still one dot per position.

`concat` is Bahdanau's network with one weight matrix on the concatenation.

## Precomputing the keys

Each score has a per-sentence part that can be cached. For `concat`, the matrix splits: $W_a [h ; k_s] = W_h h + W_k k_s$, where $W_h$ is the first $d$ columns of $W_a$ and $W_k$ the last $d$. So $W_k k_s$ is computed once per sentence (`project_keys`), and each step adds $W_h h$. The $\tanh$ is applied after the sum, so it must stay per step. For `general`, `project_keys` is $W_a k_s$; for `dot`, it is the keys themselves.

## Input feeding

There is still something the decoder lacks. Without help, the decoder at step $t$ does not know what it attended to at step $t - 1$. Luong's **input feeding** concatenates $\tilde h_{t-1}$ to the next input embedding, with $\tilde h_0 = 0$. The model can then avoid translating the same source word twice, a coverage signal for free. It also makes the network deeper in time: the gradient of step $t$ reaches the attention of step $t - 1$.

The attentional state $\tilde h = \tanh(W_c [c ; h])$ mixes the context into the state; it is both what the output layer reads and what input feeding passes on. Context first is Luong's convention and therefore the checkpoint layout. Feeding $\tilde h$ into the next input is the decoder's job; this module only provides `attentional`.

One last detail sets Luong apart from the transformer: there is no $1/\sqrt{d}$. Luong's dot score is unscaled, and scaling would change every weight. The scale is the transformer's addition.

## Working it by hand

Take $d = 2$, the `dot` score, $h = (1, 0)$, and keys $k_1 = (1, 0)$, $k_2 = (0, 1)$, $k_3 = (2, 0)$, the third being padding.

1. Scores: $h \cdot k = (1, 0, 2)$; after the mask, $(1, 0, -\infty)$.
2. Weights: $e^1 / (e^1 + 1) = 0.731059$ and $1 / (e + 1) = 0.268941$, then 0.
3. Context: $c = 0.731059 (1, 0) + 0.268941 (0, 1) = (0.731059, 0.268941)$.
4. With $W_c = [I \; I]$ (so $W_c [c ; h] = c + h$): $\tilde h = \tanh(1.731059, 0.268941) = (0.939181, 0.262640)$.
5. With $W_c = [I \; 0]$ instead, $\tilde h = \tanh(c) = (\tanh 0.731059, \tanh 0.268941)$: the context half is the first $d$ columns.

A correct solution reproduces steps 1 to 4 exactly and gives $\tanh(c)$ in step 5. Note that the scores in step 1 are exactly $(1, 0, 2)$; a scaled score would give different weights.

## Designing the API

```python
class LuongAttention(Module):
    query_from = "current"
    def __init__(self, d: int, score: Literal["dot", "general", "concat"], rng=None) -> None: ...
    def project_keys(self, keys: Tensor) -> Tensor: ...               # [B, S, d]
    def scores(self, query: Tensor, keys: Tensor, proj=None) -> Tensor: ...   # [B, S], unmasked
    def forward(self, query, keys, mask, proj=None) -> tuple[Tensor, Tensor]: ...  # context, weights
    def attentional(self, query: Tensor, context: Tensor) -> Tensor: ...  # tanh(W_c [c ; h])
```

The interface mirrors `AdditiveAttention`, so the decoder can hold either; the only additions are the `score` argument and `attentional`.

## Building it step by step

### The parameters

Parameters depend on the score; `combine` ($W_c$) is always present and registered last:

```python
class LuongAttention(Module):
    query_from = "current"             # the decoder steps the cell first, then attends with h_t

    def __init__(self, d, score, rng=None):
        # ... (d >= 1, score in SCORES, default rng)
        if score == "general":
            self.score_proj = Linear(d, d, bias=False, rng=r)        # W_a [d, d]
        elif score == "concat":
            self.score_proj = Linear(2 * d, d, bias=False, rng=r)    # W_a [d, 2d]
            self.v = Linear(d, 1, bias=False, rng=r)
        self.combine = Linear(2 * d, d, bias=False, rng=r)           # W_c [d, 2d]
```

### The per-sentence part

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

### The per-step part

For `dot` and `general` the score is one batched matrix-vector product:

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

### The read and the attentional state

The read is Bahdanau's, and the attentional state is one line:

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

The full solution is [`luong.py`](../../../../course/solutions/python/tinyllm/seq2seq/luong.py).

### Why this design

Per step, `dot` and `general` cost $O(B S d)$ for the scores, with `general`'s $O(B S d^2)$ projection paid once per sentence. `concat` costs $O(B d^2)$ for $W_h h$ plus $O(B S d)$ for the $\tanh$ and $v$. Bahdanau's per-step cost without caching would be $O(B S d^2)$.

Why one `[d, 2d]` matrix for `concat` and not two `Linear`s? It matches Luong's formulation and the checkpoint layout (`score_proj.weight`); splitting by slicing gives the same computation and the same cache without changing the key names.

Why attend after the step? With $h_t$ as the query, the decoder has already absorbed $y_{t-1}$ when it decides where to look; with $s_{t-1}$, as in Bahdanau, it chooses before seeing it. Input feeding compensates for what is lost: the next step still sees where this one looked.

The tradeoff is that unscaled dot scores grow with $d$, so for large $d$ the softmax saturates and gradients vanish. That is acceptable at seq2seq widths, and it is exactly the problem the transformer's $1/\sqrt{d}$ fixes.

## Testing it

As with Bahdanau attention, the oracle in [`test_luong.py`](../../../../course/solutions/learner-tests/L4.3/python/tests/l4-3-luong/test_luong.py) is the formula in numpy with the module's own weights from `state_dict()`: all three scores, the masked softmax, the context, and $\tilde h$ in float64. It adds the hand-worked numbers, a finite-difference gradient check of the keys, an empty row, and the parameter names, and imports only contract modules.

The full list of behaviors a correct solution must have:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: `dot` | unit | steps 1 to 4 of the hand example | pins down the read |
| hand example attentional order | unit | $W_c = [I \; 0]$ gives $\tanh(c)$ | context first, as Luong trained it |
| matches torch | golden | each score: weights, context, $\tilde h$, and every gradient against torch | the decoder trains through this |
| gradcheck of every input and parameter | gradcheck | float64 central differences, each score, a padded row | every parameter learns |
| `general` with identity is `dot` | property | `general` with $W_a = I$ equals `dot` | the relation between the scores |
| `general` is not symmetric | unit | $W_a$, not $W_a^\top$ | checkpoints load the right way round |
| `concat` projection splits the weight | property | `project_keys` and the score equal $v \cdot \tanh(W_a [h ; k])$ | the per-sentence cache is exact |
| masked weights are exactly zero | property | exact zeros, no gradient, an empty row reads nothing | padding is never read |
| `dot` is not scaled | unit | scores exactly $(1, 0, 2)$ | the scaled score belongs to the transformer |
| parameter names and shapes | unit | each score's keys and shapes, `combine` last, `query_from` | the safetensors keys of Luong checkpoints |
| validation | boundary | unknown score, wrong widths, wrong mask | wiring bugs fail loudly |

## What can go wrong

Several Luong bugs give a model that trains fine from scratch but cannot load Luong-trained weights, because the bug is a layout convention rather than a math error:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| $W_c [h ; c]$ instead of $W_c [c ; h]$ | Luong-trained weights decode garbage | `attentional` concatenates `[context, query]`, context first |
| `general` with $W_a^\top$ | the bilinear form is transposed | `general` projects the keys with `score_proj(keys)`, i.e. $W_a k_s$, and dots with $h$ |
| masking after the softmax | real weights do not sum to 1 | `masked_fill(..., -np.inf)` on the scores, then the softmax |
| scaling the dot score by $1/\sqrt{d}$ | every weight differs from Luong's | `dot` scores are the raw `matmul` of keys and query |
| `concat` precomputing with the query half of $W_a$ | the cached keys are multiplied by the wrong columns | `project_keys` uses `weight[:, d:]` (the key half); `scores` uses `weight[:, :d]` for the query |
| masking with $-10^9$ | an empty row reads padding uniformly | fills with `-np.inf`, never a finite number |
| $\tilde h$ without the `tanh` | the attentional state is unbounded | `F.tanh(self.combine(...))` |

## Where this leads

The [encoder-decoder](01-encoder-decoder-teacher-forcing.md) steps first, attends with the new state, and feeds $\tilde h$ into the next input. [Beam search](04-beam-search.md) decodes a Luong seq2seq in its tests, whose state carries `feed`. In the next part, [scaled dot-product attention](../11-transformer/01-scaled-dot-product-attention.md) is Luong's `dot` divided by $\sqrt{d}$, over many heads, and [the model zoo](../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) has Luong rows.

Each piece has a production descendant:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `dot` score | `F.scaled_dot_product_attention` | the $1/\sqrt{d}$ scale, heads, causal masks, fused kernels | [scaled dot-product attention](../11-transformer/01-scaled-dot-product-attention.md); FlashAttention |
| input feeding | the attentional decoder of OpenNMT | the same `input_feed` option, with coverage and copy attention | OpenNMT-py `onmt/decoders/decoder.py` (`InputFeedRNNDecoder`) |
| `general` score | bilinear attention in readers and parsers | the biaffine scorer of dependency parsers | Dozat and Manning, "Deep Biaffine Attention" (2017) |
| local attention | Luong's local-p attention | reads a window around a predicted position: cost per step independent of $S$ | Luong et al. 2015, section 3.2; [sliding-window attention](../13-decoder-block/07-sliding-window-and-sinks.md) |

### Further reading

- Luong, Pham, and Manning, "Effective Approaches to Attention-based Neural Machine Translation" (EMNLP 2015), sections 3.1 and 3.3.
- Britz et al., "Massive Exploration of Neural Machine Translation Architectures" (2017), on additive vs multiplicative scores.

## Key terms

- **dot score**: $h \cdot k_s$, the parameter-free score between a query and a key of the same width.
- **general score**: the bilinear score $h \cdot (W_a k_s)$.
- **concat score**: $v \cdot \tanh(W_a [h ; k_s])$, Bahdanau's network with one matrix on the concatenation.
- **attentional state**: $\tilde h = \tanh(W_c [c ; h])$, what the output layer reads.
- **input feeding**: concatenating $\tilde h_{t-1}$ to the next decoder input so the decoder remembers where it looked.

## Summary

- Luong keeps Bahdanau's masked read and changes two things: the score function, and when the decoder asks, **after** its recurrent step.
- `dot` is $h \cdot k_s$, `general` the bilinear $h \cdot (W_a k_s)$ (with $W_a = I$ it is `dot`), `concat` a hidden layer on $[h ; k_s]$.
- `concat`'s weight splits into a key half, cached per sentence, and a query half, computed per step.
- The attentional state $\tilde h = \tanh(W_c [c ; h])$, context first, is what the output layer reads and what input feeding passes to the next step.
- The plain dot score is not scaled here; scaling by $1/\sqrt{d}$ is the transformer's change.

## Review questions

1. Why does `general` with $W_a = I$ give exactly `dot`, and why does $W_a^\top$ give something different?
   <details><summary>Answer</summary>With $W_a = I$, `project_keys` returns the keys and the score is $h \cdot k_s$. In general $h^\top W_a k \ne h^\top W_a^\top k$ unless $W_a$ is symmetric, so a checkpoint trained with one convention scores differently under the other.</details>
2. In `concat`, why can the key half be precomputed but not the query half?
   <details><summary>Answer</summary>$W_a [h ; k] = W_h h + W_k k$ is linear before the $\tanh$. $W_k k_s$ depends only on the sentence; $W_h h$ changes every step. The $\tanh$ is applied after the sum, so it must stay per step.</details>
3. In step 5 of the worked example, why does $W_c = [I \; 0]$ give $\tanh(c)$ and not $\tanh(h)$?
   <details><summary>Answer</summary>`attentional` concatenates `[context, query]`, so the first $d$ columns of $W_c$ multiply $c$. With $[I \; 0]$ only the context passes through.</details>
4. What would input feeding give at step 1, and where is it set up?
   <details><summary>Answer</summary>$\tilde h_0 = 0$. The encoder-decoder's `init_state` creates a zero `feed` when `query_from == "current"`, and each `decode_step` replaces it with the new `attentional(h, ctx)`.</details>
5. Why is the dot score not divided by $\sqrt{d}$ here?
   <details><summary>Answer</summary>Luong's model is unscaled, and weights trained for it expect raw scores; the worked example's weights $(0.731, 0.269)$ come from scores $(1, 0)$. Scaling is a separate design decision made by the transformer's scaled dot-product attention.</details>
