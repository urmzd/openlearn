# Multi-head attention

[Scaled dot-product attention](01-scaled-dot-product-attention.md) gives you one attention: every query compares itself with every key through one dot product over all $d$ features, and gets one weight vector. Take the addition task the 2017 transformer is trained on in this chapter. To write the tens digit of the sum, the decoder has to look at the tens digit of $a$ **and** the tens digit of $b$, four positions apart. A single softmax can split its mass between the two, but then it averages them into one blurred vector.

Vaswani et al. split the model width into $H$ heads that each run their own attention with their own projections, and concatenate the results: head 0 can find $a$'s digit while head 1 finds $b$'s. Every model in this chapter and the next (the [encoder-decoder Transformer](05-encoder-decoder-transformer.md), [GPT](../07-objectives/01-gpt-decoder-only.md), [BERT](../07-objectives/02-bert-masked-lm.md)) calls this one module, so its layout and parameter names are fixed here once. In this section you will build the module, work a two-head example by hand, and see exactly why the split and merge must each be a reshape *and* a transpose.

This section builds on scaled dot-product attention, on `Linear` and `Module` from [Module system and basic layers](../01-foundations/05-module-system-and-layers.md), on `reshape` and `transpose` from [The op library and gradcheck_all](../01-foundations/03-op-library-and-gradcheck.md), on the `Tensor` of [Tensor, broadcasting backward, and no_grad](../01-foundations/02-tensor-and-broadcasting-backward.md), and on `PCG32` from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*. The full solution is [`python/tinyllm/xfmr/mha.py`](../../../../course/solutions/python/tinyllm/xfmr/mha.py) (`MultiHeadAttention` with `split_heads`, `merge_heads`, `head_mask`, `attend`, `forward`, `load_packed_in_proj`), its contract is [`course/contracts/py/tinyllm/xfmr/mha.pyi`](../../../../course/contracts/py/tinyllm/xfmr/mha.pyi), and the model oracle tests are in [`test_mha.py`](../../../../course/solutions/learner-tests/L5.3/python/tests/l5-3-mha/test_mha.py).

> **Learning objectives**
> - Explain why $H$ heads can read several places at once where one wide head can only average.
> - Split `[B, T, d]` into `[B, H, T, d_head]` and merge it back without mixing positions.
> - Justify the per-head scale $1/\sqrt{d_{\text{head}}}$ and see where it comes from in the code.
> - Use one module for self- and cross-attention, and give a per-sequence mask the head axis it needs.

## The notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $B, T_q, T_k$ | batch, query length, key length | `int` |
| $d$ | model width `d_model` | `int` |
| $H$ | number of heads `n_heads` | `int`, divides $d$ |
| $d_h = d / H$ | width of one head `d_head` | `int` |
| $X_q, X_{kv}$ | the inputs queries and keys/values come from | `[B, Tq, d]`, `[B, Tk, d]` |
| $W_q, W_k, W_v, W_o$ | the four projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) | `float32[d, d]` each, plus biases `[d]` |
| $Q_h, K_h, V_h$ | head $h$'s slice of the projected queries, keys, values | `[B, Tq, dh]`, `[B, Tk, dh]` |
| $M$ | mask, True = may attend | bool, broadcasts to `[B, H, Tq, Tk]` |

## One head

One head is scaled dot-product attention after learned projections:

$$\text{head}(X_q, X_{kv}) = \operatorname{softmax}\!\Big(\frac{(X_q W_q^\top)(X_{kv} W_k^\top)^\top}{\sqrt{d}} + M\Big)\, X_{kv} W_v^\top .$$

With $H = 1$ the module is exactly this followed by $W_o$, which you can check by computing it in numpy from the module's own weights.

## Many heads

With $H$ heads the projections stay $d \times d$, but their output is read in $H$ slices of width $d_h$: head $h$ owns features $h d_h$ to $(h + 1) d_h - 1$. Each head attends in its own slice, scaled by $1/\sqrt{d_h}$ because a dot product of $d_h$ terms with unit-variance entries has variance $d_h$:

$$\text{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K_h^\top}{\sqrt{d_h}} + M\Big) V_h, \qquad \text{out} = [\text{head}_0 ; \dots ; \text{head}_{H-1}]\, W_o^\top .$$

In code all heads run in one batched call. A projected `[B, T, d]` tensor is reshaped to `[B, T, H, d_h]` (the slices become an axis) and transposed to `[B, H, T, d_h]`, so attention sees $H$ more batch rows. Merging reverses it: transpose back to `[B, T, H, d_h]`, then reshape to `[B, T, d]`.

> **Warning:** reshaping `[B, H, T, d_h]` straight to `[B, T, d]` is legal numpy and wrong. It glues together the rows of different positions, and nothing crashes.

The cost is the same as one wide head: four $d \times d$ projections and $H$ attentions of width $d_h$.

## Masks for every head

Masks come from the [previous section](02-masks.md) and mean "True = may attend". A causal mask is `[Tq, Tk]` and applies to every sequence and head; a padding mask is per sequence. The module accepts `[Tq, Tk]`, `[B, Tq, Tk]` (it inserts the head axis at 1), and any shape that already broadcasts to `[B, H, Tq, Tk]`, such as `[B, 1, 1, Tk]` key padding.

The head axis matters because numpy aligns shapes from the right. A `[B, Tq, Tk]` mask without the inserted axis broadcasts against `[B, H, Tq, Tk]` with its batch axis on the head axis. When $B = H$ nothing fails and the masks are simply wrong.

## Self- and cross-attention

`attend(x_q, x_kv, mask)` takes queries from `x_q` and **both** keys and values from `x_kv`. Self-attention passes the same tensor twice; the decoder's cross-attention in the [encoder-decoder Transformer](05-encoder-decoder-transformer.md) passes its own states as `x_q` and the encoder's output as `x_kv`. The output has `x_q`'s length.

Keys are a set: permuting the positions of `x_kv` (with its mask) leaves the output unchanged, and permuting `x_q` permutes the output rows. Position information therefore has to come from outside, which is the job of [positional encodings](04-positional-encodings.md).

## Dropout and torch's packed layout

Two practical details complete the design. Attention dropout zeroes attention weights at random during training (rate `dropout`, drawn from `PCG32(0).substream("dropout")` by scaled dot-product attention) and is off in eval mode. And torch's `nn.MultiheadAttention` keeps $W_q, W_k, W_v$ stacked in one `in_proj_weight` of shape `[3d, d]`, rows in the order q, k, v; `load_packed_in_proj` copies them into the three `Linear`s.

## Working it by hand

Take $d = 4$, $H = 2$, $d_h = 2$, and every projection the identity with zero bias, so $Q = X_q$ and $K = V = X_{kv}$. There is one query and two keys, split by head:

| | features 0, 1 (head 0) | features 2, 3 (head 1) |
|---|---|---|
| $q = (1, 0, 0, 2)$ | $(1, 0)$ | $(0, 2)$ |
| $k_1 = (1, 0, 0, 0)$ | $(1, 0)$ | $(0, 0)$ |
| $k_2 = (0, 1, 0, 2)$ | $(0, 1)$ | $(0, 2)$ |

**Head 0.** The scores are $(1, 0) / \sqrt 2 = (0.707107, 0)$; the weights are $\sigma(0.707107) = 0.669762$ and $0.330238$; the context is $0.669762 (1, 0) + 0.330238 (0, 1) = (0.669762, 0.330238)$.

**Head 1.** The scores are $(0, 4) / \sqrt 2 = (0, 2.828427)$; the weights are $\sigma(-2.828427) = 0.055817$ and $0.944183$; the context is $0.944183 \cdot (0, 2) = (0, 1.888366)$.

**Merge.** Head 0's context fills features 0 and 1, head 1's fills 2 and 3: $(0.669762, 0.330238, 0, 1.888366)$, and $W_o = I$ leaves it unchanged. A correct solution reproduces these weights and this output.

Compare **one** head of width 4 on the same numbers. The scores are $(q \cdot k_1, q \cdot k_2) / 2 = (0.5, 2)$, the weights are $(0.182426, 0.817574)$ for every feature, and the output is $(0.182426, 0.817574, 0, 1.635149)$. Head 0's preference for $k_1$ is outvoted. That is the whole argument for heads in one example.

## Designing the API

```python
class MultiHeadAttention(Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True, rng=None)
    def split_heads(self, x: Tensor) -> Tensor              # [B, T, d] -> [B, H, T, dh]
    def merge_heads(self, x: Tensor) -> Tensor              # [B, H, T, dh] -> [B, T, d]
    def head_mask(self, mask, B, Tq, Tk) -> Optional[NDArray]
    def attend(self, x_q: Tensor, x_kv: Tensor, mask=None) -> tuple[Tensor, Tensor]   # out, weights [B, H, Tq, Tk]
    def forward(self, x_q: Tensor, x_kv: Tensor, mask=None) -> Tensor
    def load_packed_in_proj(self, weight, bias=None) -> None  # torch's [3d, d], rows q | k | v
```

The parameters, in `state_dict` order, are `q_proj.weight`, `q_proj.bias`, `k_proj.*`, `v_proj.*`, `out_proj.*`: four `Linear`s drawn from one `rng`. Those names are the checkpoint keys used through the modern block.

## Building it step by step

The reasoning chain:

1. **Several independent reads need several softmaxes.** One softmax gives one weight vector per query, so it can only average where it looks. $H$ heads give $H$ weight vectors, each over its own feature slice.
2. **Keep the cost of one wide head.** Projections stay $d \times d$; their outputs are *read* as $H$ slices of width $d_h$. Head $h$ owns features $h d_h .. (h+1) d_h - 1$, so splitting is a reshape of the last axis.
3. **Heads must become a batch axis.** Scaled dot-product attention treats leading axes as independent problems, so heads go to axis 1: reshape `[B, T, d] -> [B, T, H, d_h]`, then transpose to `[B, H, T, d_h]`. Merging must undo both in reverse order; a direct reshape would glue rows of different positions.
4. **Scale by the width compared.** Each head's dot product has $d_h$ terms, so its scale is $1/\sqrt{d_h}$. Since the last axis of the split tensors *is* $d_h$, the attention function's default scale is already right; passing nothing is the correct call. In the worked example head 0's scores are $(1, 0)/\sqrt 2$.
5. **Queries and keys may come from different sequences.** Taking `x_q` and `x_kv` separately makes one module serve self-attention (same tensor twice) and cross-attention (decoder states and encoder output).
6. **Masks need a head axis.** A per-sequence `[B, Tq, Tk]` mask must become `[B, 1, Tq, Tk]`; otherwise right-aligned broadcasting puts the batch axis on the head axis.
7. **Heads must be mixed.** The concatenated heads pass through $W_o$, so each output feature can combine what every head read.

### Construction

Four `Linear`s in contract order (their names are the checkpoint keys), and a dedicated dropout stream:

```python
class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, dropout=0.0, bias=True, rng=None):
        super().__init__()
        # ... (positive sizes; d_model % n_heads == 0; 0 <= dropout < 1)
        r = rng if rng is not None else PCG32(0).substream("init")
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.dropout = float(dropout)
        self.q_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.k_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.v_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.out_proj = Linear(d_model, d_model, bias=bias, rng=r)
        self.dropout_rng = PCG32(0).substream("dropout")   # dropout draws never shift init
```

### Split and merge

Each is a reshape and a transpose, in mirror order:

```python
def split_heads(self, x):
    # ... (x must be [B, T, d_model])
    B, T, _ = x.shape
    # Head h owns features h*d_head .. (h+1)*d_head - 1 of every position.
    return F.transpose(F.reshape(x, (B, T, self.n_heads, self.d_head)), 1, 2)   # [B, H, T, dh]

def merge_heads(self, x):
    # ... (x must be [B, H, T, d_head])
    B, _, T, _ = x.shape
    # Back to [B, T, H, d_head] first: positions must not mix.
    return F.reshape(F.transpose(x, 1, 2), (B, T, self.d_model))                # [B, T, d]
```

### The mask

The mask is normalized to something that broadcasts to `[B, H, Tq, Tk]`:

```python
def head_mask(self, mask, B, Tq, Tk):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        raise ValueError(f"mask must be bool (True = may attend), got {m.dtype}")
    if m.ndim == 3:
        # [B, Tq, Tk]: one mask per sequence, the same for every head.
        m = m[:, None, :, :]
    elif m.ndim not in (2, 4):
        raise ValueError(...)
    # ... (must broadcast to (B, H, Tq, Tk), else ValueError)
    return m
```

A `[Tq, Tk]` causal mask and a `[B, 1, 1, Tk]` padding mask (from `combine`) pass through unchanged, since they already broadcast correctly.

### The layer

```python
def attend(self, x_q, x_kv, mask=None):
    # ... (both [B, T, d], same batch)
    B, Tq, Tk = x_q.shape[0], x_q.shape[1], x_kv.shape[1]
    m = self.head_mask(mask, B, Tq, Tk)
    q = self.split_heads(self.q_proj(x_q))      # queries from x_q
    k = self.split_heads(self.k_proj(x_kv))     # keys AND values from x_kv
    v = self.split_heads(self.v_proj(x_kv))
    p = self.dropout if self.training else 0.0  # no dropout in eval mode
    # Default scale 1 / sqrt(d_head): the width of what each head compares.
    ctx, w = scaled_dot_product_attention(q, k, v, mask=m, dropout_p=p, rng=self.dropout_rng)
    return self.out_proj(self.merge_heads(ctx)), w   # w: [B, H, Tq, Tk]
```

`forward` returns only the output. Because the whole attention is a single autograd node between the projections, the backward through all heads is exact and cheap.

### Loading torch weights

Loading torch's packed projection just slices rows in q, k, v order:

```python
def load_packed_in_proj(self, weight, bias=None):
    w = np.asarray(weight, dtype=np.float32)
    d = self.d_model
    # ... (w must be [3d, d])
    sd = {"q_proj.weight": w[:d], "k_proj.weight": w[d : 2 * d], "v_proj.weight": w[2 * d :]}
    if bias is not None:
        # ... (b must be [3d]; the module must have biases)
        sd.update({"q_proj.bias": b[:d], "k_proj.bias": b[d : 2 * d], "v_proj.bias": b[2 * d :]})
    self.load_state_dict(sd, strict=False)      # out_proj is loaded separately
```

The full solution is [`mha.py`](../../../../course/solutions/python/tinyllm/xfmr/mha.py).

### Why this design

The cost is four $d \times d$ projections, $O(B T d^2)$, plus $H$ attentions of width $d_h$, $O(B H T_q T_k d_h) = O(B T_q T_k d)$: the same as one wide head. Heads cost nothing extra in FLOPs, only in the $[B, H, T_q, T_k]$ weights kept for the backward.

Why three `Linear`s and not torch's packed `in_proj`? Separate names match the contract and later models (Llama's `q_proj`, `k_proj`, `v_proj`), and cross-attention needs keys and values projected from a different input than queries anyway. `load_packed_in_proj` bridges to torch's layout.

Why reuse scaled dot-product attention with batched heads instead of a loop over heads? One call runs all $B \cdot H$ problems with NumPy broadcasting and one hand-written backward; a Python loop over heads would be $H$ graph nodes and $H$ times the overhead. And why let that function pick the scale? Its default reads the last axis, which after splitting is $d_h$. Passing $1/\sqrt{d_{\text{model}}}$ explicitly would make the softmax too flat by a factor $\sqrt H$.

## What can go wrong

A correct module must reproduce the two-head hand example (weights per head and the merged output, which pins down the layout and the scale) and the one-head comparison (which shows why heads exist). It must match torch's `nn.MultiheadAttention` on output, per-head weights, and every gradient, for self-attention with causal and padding masks and for cross-attention with padding; the encoder-decoder Transformer, GPT, and BERT all load torch-trained weights through this. A gradcheck covers both inputs and all 8 parameter tensors against frozen central differences, since every model trains through this backward.

The properties: head $h$ owns features $h d_h ..$ and merge inverts split, so positions never mix; changing head 1's query rows leaves head 0's weights bitwise equal; with $H = 1$ the module equals numpy $\operatorname{softmax}(QK^\top/\sqrt d)V$ followed by $W_o$; `[Tq, Tk]`, `[B, Tq, Tk]`, and `[B, H, Tq, Tk]` masks agree even with $B = H$; masked keys weigh exactly 0 and their values cannot change the output; and in cross-attention the output length comes from `x_q` while `x_kv` positions behave as a set. Eval mode must equal no dropout, so evaluation and decoding are deterministic. The eight parameter names must appear in order with seeded init and `bias=False` support; torch's packed rows must load as q, k, v; and heads that do not tile $d$, bad shapes, and non-bool masks must raise.

The model suite [`test_mha.py`](../../../../course/solutions/learner-tests/L5.3/python/tests/l5-3-mha/test_mha.py) uses an oracle: multi-head attention written again in float64 numpy from the module's own `state_dict` (project, reshape, transpose, softmax with the mask, merge, project), compared with `attend`'s output and weights for self- and cross-attention, with a per-sequence `[B, Tq, Tk]` mask where $B = H$. It adds the worked numbers, a split/merge check, eval-mode dropout, the packed load order, and a float64 gradient check with `tinyllm.num.gradcheck` (from [Partial derivatives, gradients, and gradcheck](../../mathematics-for-computing/05-calculus-3/01-gradients-and-gradcheck.md)). It imports only names in `contracts/py`.

Most of the classic bugs are silent, which is why the list is worth memorizing:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| merging (or splitting) heads with a plain reshape | positions mix; torch disagrees; nothing crashes | `split_heads` reshapes then transposes; `merge_heads` transposes then reshapes |
| scaling by $1/\sqrt{d_{\text{model}}}$ | softmax too flat by a factor $\sqrt H$ | no explicit scale: the default $1/\sqrt{d}$ sees `d_head` as the last axis |
| projecting keys from `x_q` | cross-attention reads the decoder, or crashes when lengths differ | `k_proj(x_kv)` and `v_proj(x_kv)` |
| a `[B, Tq, Tk]` mask without a head axis | with $B = H$ head $b$ gets sequence $b$'s mask | `head_mask` inserts the head axis: `m[:, None, :, :]` |
| dropout in eval mode | decoding is random, evaluation noisy | `p = self.dropout if self.training else 0.0` |
| reading torch's packed `in_proj` as k, q, v | a loaded checkpoint attends with swapped roles | rows `[:d]`, `[d:2d]`, `[2d:]` go to q, k, v |
| the output projection skipped | heads are never mixed; shapes still fit | `attend` returns `self.out_proj(self.merge_heads(ctx))` |

## Where this leads

The [encoder-decoder Transformer](05-encoder-decoder-transformer.md) uses this module three ways: encoder self-attention, decoder masked self-attention, and cross-attention. When you train it and translate with a beam of 4, it runs in every layer. [GPT](../07-objectives/01-gpt-decoder-only.md) uses it for causal self-attention (GPT-2's `c_attn` split into q, k, v), and [BERT](../07-objectives/02-bert-masked-lm.md) for bidirectional self-attention. [MQA/GQA attention](../08-modern-block/05-gqa-attention.md) shares $K, V$ heads between query heads. Every block of the [vision transformer](../14-vision/03-vision-transformer.md) and of the [Whisper encoder](../15-speech/02-whisper-encoder.md) is `MultiHeadAttention` with no mask.

The production versions differ mainly in fusion and in how many key-value heads they keep:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `split_heads` / `merge_heads` | PyTorch `F.scaled_dot_product_attention` on `[B, H, T, d]` | fused kernels (FlashAttention, memory-efficient) behind one call | `torch/nn/functional.py`, `aten/src/ATen/native/transformers/` |
| `load_packed_in_proj` | HF `GPT2Attention` c_attn, Llama's separate q/k/v | fused QKV matmul for speed, split views | `transformers/models/gpt2/modeling_gpt2.py` |
| one K, V per head | multi-query and grouped-query attention | fewer K, V heads: a smaller KV cache | Shazeer (2019); Ainslie et al. (2023); [MQA/GQA attention](../08-modern-block/05-gqa-attention.md) |
| attention dropout | most modern LLMs train with 0 | dropout matters for small data, not trillions of tokens | Llama and Mistral configs |

### Further reading

- Vaswani et al., "Attention Is All You Need" (2017), section 3.2.2.
- Michel, Levy, and Neubig, "Are Sixteen Heads Really Better than One?" (NeurIPS 2019).

## Key terms

- **head**: one attention run inside its own $d_{\text{head}} = d_{\text{model}}/H$ slice of the projected features.
- **split heads**: reshape `[B, T, d]` to `[B, T, H, d_h]`, then transpose to `[B, H, T, d_h]`.
- **merge heads**: transpose back to `[B, T, H, d_h]`, then reshape to `[B, T, d]`.
- **cross-attention**: attention with queries from one sequence and keys and values from another.
- **packed in_proj**: torch's `[3d, d]` stack of $W_q, W_k, W_v$, rows in q, k, v order.

## Summary

- A head attends inside its own $d_{\text{head}} = d_{\text{model}} / H$ slice of the features, so one position can read two places at once; one wide head cannot.
- Splitting is a reshape to `[B, T, H, d_head]` **and** a transpose to `[B, H, T, d_head]`; merging undoes both, in that order.
- Each head scales by $1/\sqrt{d_{\text{head}}}$, the width it compares, not $1/\sqrt{d_{\text{model}}}$.
- Self-attention and cross-attention are the same module: queries from `x_q`, keys and values from `x_kv`.
- A per-sequence mask needs a head axis, or sequence $b$'s mask silently lands on head $b$.

## Review questions

1. Why must `merge_heads` transpose before reshaping?
   <details><summary>Answer</summary>The context is `[B, H, T, d_h]` in memory order. Reshaping it directly to `[B, T, d]` would read head 0's rows for positions 0 and 1 as one position's features. Transposing to `[B, T, H, d_h]` first puts each position's $H$ slices next to each other, so the reshape concatenates heads, not positions.</details>
2. In the worked example, why does head 0 prefer $k_1$ while the single wide head prefers $k_2$?
   <details><summary>Answer</summary>Head 0 compares only features 0 and 1, where $q = (1, 0)$ matches $k_1 = (1, 0)$. The wide head sums all four features; features 2 and 3 give $q \cdot k_2$ an extra 4, which outvotes head 0's preference.</details>
3. Where does the $1/\sqrt{d_h}$ come from in the code, given that `attend` never passes a scale?
   <details><summary>Answer</summary>`scaled_dot_product_attention` defaults to $1/\sqrt{d}$ with $d$ the last axis of `q`. After `split_heads` that axis is `d_head`.</details>
4. Why does a `[B, Tq, Tk]` mask need `m[:, None, :, :]`, and why can the bug go unnoticed?
   <details><summary>Answer</summary>NumPy aligns shapes from the right, so `[B, Tq, Tk]` against `[B, H, Tq, Tk]` matches $B$ with $H$. When $B = H$ (or $B = 1$) it broadcasts without error, and each head silently gets another sequence's mask.</details>
5. What would break if keys were projected from `x_q`?
   <details><summary>Answer</summary>Cross-attention would attend over the decoder's own states instead of the encoder output, and when the lengths differ, the keys and values would have different lengths, so it would crash or read the wrong source.</details>
