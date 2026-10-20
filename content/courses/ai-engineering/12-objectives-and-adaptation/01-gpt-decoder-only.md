# GPT: the decoder-only language model

The [encoder-decoder Transformer](../11-transformer/05-encoder-decoder-transformer.md) of the previous chapter maps one sequence to another. The system this book builds serves something simpler and more useful: a language model, a model that continues text. In a language model every token of a corpus is a training target for the prefix before it, and nothing needs to be encoded separately. GPT keeps exactly the half of the encoder-decoder that does this job: the decoder, with its cross-attention removed, because there is no source sequence to attend to.

This section builds that model and follows GPT-2's exact layout, so that a checkpoint trained with Hugging Face loads into it and produces the same logits. The loader is the bridge the rest of the book walks over: the [Llama-family model](../13-decoder-block/09-llama-family-model.md) loads SmolLM2 the same way, and [the KV cache and `generate`](../21-inference/02-kv-cache-and-generate.md) is first built on this model.

You will need the pieces of Part 5: [causal masks](../11-transformer/02-masks.md), [multi-head attention](../11-transformer/03-multi-head-attention.md), and [learned positional encodings](../11-transformer/04-positional-encodings.md); the foundations of Part 0 ([the `Tensor`](../06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md), [the op library](../06-numerics-and-autograd/03-op-library-and-gradcheck.md), [losses](../06-numerics-and-autograd/04-losses-with-fused-backward.md), [layers](../06-numerics-and-autograd/05-module-system-and-layers.md), [the training loop](../06-numerics-and-autograd/06-training-loop-and-the-autograd-bigram.md), and [safetensors and the token stream](../06-numerics-and-autograd/07-safetensors-checkpoints-and-token-streams.md)); and from *Mathematics*, [AdamW](../../mathematics/16-optimization/03-adam-and-adamw.md), [cosine schedules](../../mathematics/16-optimization/04-schedules-and-gradient-clipping.md), [variance propagation and initialization](../../mathematics/13-probability-statistics/04-variance-propagation-and-initialization.md), and [PCG32](../../mathematics/12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) for seeded initialization. The reference implementation is [`python/tinyllm/obj/gpt.py`](../../../../course/solutions/python/tinyllm/obj/gpt.py) (`GPTConfig`, `Block`, `GPT` with `hidden` and `forward`, `clm_loss`, `load_hf_gpt2`, `fit_gpt`, `save_gpt`, `load_gpt`), and its contract is [`course/contracts/py/tinyllm/obj/gpt.pyi`](../../../../course/contracts/py/tinyllm/obj/gpt.pyi).

> **Learning objectives**
> - Explain why one causal self-attention stack trains on every position of a window in a single forward pass.
> - Compute the causal LM loss by hand, including the shift by one and padded targets.
> - Describe GPT-2's block exactly: pre-LN, tanh GELU, final `ln_f`, tied head, and depth-scaled residual init.
> - Load a Hugging Face GPT-2 state dict, transposing every `Conv1D` weight and splitting `c_attn` into q, k, v.
> - Explain why only a value comparison, not a shape check, catches a missing transpose.

## One stack, one mask

Here is the notation used throughout the section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $t_0 \dots t_{T-1}$ | the token ids of one window | `int[T]` |
| $V, d, L, H$ | vocabulary, width, layers, heads | `int` |
| $n_{ctx}$ | the longest window (rows of `wpe`) | `int` |
| $W_{te} \in \mathbb{R}^{V \times d}$ | token table `wte` | `float32[V, d]` |
| $W_{pe} \in \mathbb{R}^{n_{ctx} \times d}$ | learned positions `wpe` | `float32[n_ctx, d]` |
| $h^{(\ell)}_t$ | the residual stream at layer $\ell$, position $t$ | `[B, T, d]` |
| $z_t \in \mathbb{R}^V$ | logits at position $t$ | `[B, T, V]` |
| $\text{GELU}_{\tanh}(u)$ | $\tfrac12 u (1 + \tanh(\sqrt{2/\pi}(u + 0.044715 u^3)))$ | elementwise |

A language model factors the probability of a window by the chain rule:

$$P(t_0 \dots t_{T-1}) = \prod_{t=1}^{T-1} P(t_t \mid t_0 \dots t_{t-1}) .$$

Every factor conditions only on a prefix. With the causal mask, position $t$ attends to positions $0 \dots t$ only, so its output is a function of the prefix and can be trained to predict $t_{t+1}$. One forward pass therefore computes all $T$ predictions at once. Without the mask, position $t$ could read $t_{t+1}$ directly, and the loss would teach the model to copy rather than to predict.

## The causal LM loss

The mask fixes what each position may see. The loss then pairs each position with the token after it:

$$\mathcal{L} = \frac{1}{|K|} \sum_{t \in K} -\log \operatorname{softmax}(z_t)_{t_{t+1}}, \qquad K = \{ t < T - 1 : t_{t+1} \text{ is a real token} \}.$$

In code this is logits `[:, :-1]` against ids `[:, 1:]`, with `-100` wherever the target is padding, which [`cross_entropy`](../06-numerics-and-autograd/04-losses-with-fused-backward.md) ignores. The important discipline is that the shift happens **once**. Either `clm_loss` shifts a full window, or the data already arrives as (inputs, targets) pairs offset by one, as the `TokenStream` does, and `forward(x, y)` uses them as they are. Shifting in both places would train the model to predict two tokens ahead.

## The GPT-2 block

With the objective settled, the architecture follows. Each GPT-2 block is pre-LN (the `norm="pre"` variant of the encoder-decoder chapter):

$$h \leftarrow h + \text{Attn}(\text{LN}_1(h)), \qquad h \leftarrow h + W_{proj}\,\text{GELU}_{\tanh}(W_{fc}\,\text{LN}_2(h)),$$

followed by a final $\text{LN}_f$ after the last block, and $z = \text{LN}_f(h) W_{te}^\top$: the output layer is the token table itself (a tied head). GPT-2 used the tanh approximation of GELU. The exact erf form differs from it by up to about $10^{-3}$ at $|u| \approx 2$, which is enough to fail a golden comparison against Hugging Face.

Initialization follows GPT-2 too. Tables and every `Linear` weight are drawn from $\mathcal{N}(0, 0.02^2)$, biases are 0, and LayerNorms start at $(1, 0)$. One refinement matters at depth. Each block adds two outputs to the residual stream (`attn.out_proj` and `mlp_proj`), so after $L$ blocks the stream holds $2L$ such terms. Their weights start at $0.02/\sqrt{2L}$ (the `scaled_residual_std` of [variance propagation and initialization](../../mathematics/13-probability-statistics/04-variance-propagation-and-initialization.md)), which keeps the stream's variance from growing with depth.

## Loading GPT-2

Matching the computation is only half of loading a real checkpoint; the other half is matching the names and layouts. Hugging Face's `GPT2LMHeadModel` uses the following keys and shapes.

| HF key (after `transformer.`) | Shape | In the solution |
|---|---|---|
| `wte.weight`, `wpe.weight` | `[V, d]`, `[n_ctx, d]` | same names |
| `h.i.ln_1.*`, `h.i.ln_2.*`, `ln_f.*` | `[d]` | same names |
| `h.i.attn.c_attn.weight`, `.bias` | `[d, 3d]`, `[3d]` | `attn.q_proj`, `k_proj`, `v_proj`: columns $0..d$, $d..2d$, $2d..3d$, each **transposed** |
| `h.i.attn.c_proj.weight` | `[d, d]` | `attn.out_proj.weight` **transposed** |
| `h.i.mlp.c_fc.weight` | `[d, 4d]` | `mlp_fc.weight` **transposed** |
| `h.i.mlp.c_proj.weight` | `[4d, d]` | `mlp_proj.weight` **transposed** |

The transposes come from a difference in conventions. `Conv1D` computes $x W + b$ with $W$ stored `[in, out]`; `Linear` computes $x W^\top + b$ with $W$ stored `[out, in]`. The square `c_proj` of attention loads either way without a shape error, which is why only a comparison of values, not shapes, catches a missing transpose. Old checkpoints also carry the buffers `attn.bias` and `attn.masked_bias`, a stored causal mask; they are not weights, and the loader ignores them. Finally, `lm_head.weight` is the same tensor as `wte.weight`.

## Working the loss by hand

Before writing code, it helps to pin down the shift with numbers everyone can agree on. Take $V = 2$, ids $(0, 1, 1)$, and logit rows $z_0 = (0, \ln 3)$, $z_1 = (0, 0)$, $z_2 = (5, -5)$.

- Position 0 predicts $t_1 = 1$: $\operatorname{softmax}(0, \ln 3) = (1/4, 3/4)$, loss $-\ln 0.75 = 0.287682$.
- Position 1 predicts $t_2 = 1$: $(1/2, 1/2)$, loss $\ln 2 = 0.693147$.
- Position 2 has no next token, so it is dropped.

The loss is $\mathcal{L} = (0.287682 + 0.693147)/2 = 0.490415$. Now mark the third token as padding. Position 1's target is padding, so only position 0 counts and $\mathcal{L} = 0.287682$. The gradient on $z_0$ is $\operatorname{softmax}(z_0) - \text{onehot}(1) = (0.25, -0.25)$, and it is zero on the other rows.

## Designing the interface

The public surface is small: a config, the model with a shared trunk and a forward pass, a standalone loss, the loader, a training loop, and save/load.

```python
@dataclass
class GPTConfig: vocab: int; n_ctx: int; d_model: int; n_heads: int; n_layers: int; d_ff: int
                 dropout: float = 0.0; ln_eps: float = 1e-5; tie: bool = True
class GPT(Module):
    def hidden(self, ids) -> Tensor                                       # ln_f output [B, T, d]
    def forward(self, ids, targets=None) -> tuple[Tensor, Optional[Tensor]]
def clm_loss(logits: Tensor, ids, mask=None) -> Tensor
def load_hf_gpt2(model: GPT, sd: Mapping[str, Any]) -> None
def fit_gpt(model, stream, steps, lr, warmup=0, weight_decay=0.1, clip=1.0) -> list[float]
def save_gpt(model, dir, tokenizer="bytes") / load_gpt(dir)
```

`hidden` returns the output of `ln_f`, the trunk that the [fine-tuning heads](05-fine-tuning-heads-and-the-linear-head-export.md) reuse. `forward` takes targets that are already shifted. `clm_loss` takes a full window and does the shift itself.

## Building it step by step

The reasoning runs in a short chain.

1. A language model is $\prod_t P(t_t \mid t_{<t})$. Every factor conditions only on a prefix, so a single causal self-attention stack can compute every factor in one pass: position $t$'s output predicts $t_{t+1}$. That is the encoder-decoder's decoder with cross-attention deleted, because there is no source.
2. The loss must pair logits at $t$ with ids at $t + 1$. The shift is done in exactly one place: `clm_loss` shifts a full window, and `forward(ids, targets)` assumes the caller already shifted, as `TokenStream` does. Doing it in both would predict two tokens ahead.
3. To load real GPT-2 weights, the module must match GPT-2's computation exactly: pre-LN, tanh GELU, learned positions, final `ln_f`, tied head. Each of these is a numeric difference a golden comparison detects.
4. Naming follows Hugging Face where possible (`wte`, `wpe`, `h.i.ln_1`, `ln_f`), so the loader is mostly a rename. Where the layouts differ (Conv1D `[in, out]` against Linear `[out, in]`, and the fused `c_attn`), the loader transposes and splits.
5. Depth makes the residual stream a sum of $2L$ terms. With independent terms the variances add, so each writing projection starts at std $0.02/\sqrt{2L}$ to keep the total near the single-term scale.
6. The tied head is a `matmul` against the transpose of the same `wte` tensor, so the gradient from the output side lands on the same parameter as the gradient from the input side.

### The block

The block is pre-LN with both residuals and the GPT-2 init:

```python
class Block(Module):
    def __init__(self, cfg: GPTConfig, rng: Any) -> None:
        super().__init__()
        d = cfg.d_model
        self.ln_1 = LayerNorm(d, cfg.ln_eps)
        self.attn = MultiHeadAttention(d, cfg.n_heads, cfg.dropout, rng=rng)
        self.ln_2 = LayerNorm(d, cfg.ln_eps)
        self.mlp_fc = Linear(d, cfg.d_ff, rng=rng)
        self.mlp_proj = Linear(cfg.d_ff, d, rng=rng)
        self.drop = Dropout(cfg.dropout)
        resid = scaled_residual_std(INIT_STD, cfg.n_layers)   # 0.02 / sqrt(2 L)
        for lin in (self.attn.q_proj, self.attn.k_proj, self.attn.v_proj, self.mlp_fc):
            _init_linear(lin, INIT_STD, rng)
        # The two projections that write into the residual stream.
        for lin in (self.attn.out_proj, self.mlp_proj):
            _init_linear(lin, resid, rng)

    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        h = self.ln_1(x)
        x = x + self.drop(self.attn(h, h, mask))
        # GPT-2's GELU is the tanh approximation, not the erf form.
        m = self.mlp_proj(F.gelu(self.mlp_fc(self.ln_2(x)), approximate="tanh"))
        return x + self.drop(m)
```

### The model

`hidden` is the shared trunk the heads reuse; `forward` adds the tied output layer:

```python
    def hidden(self, ids: ArrayLike) -> Tensor:
        ids = np.asarray(ids)
        # ... int [B, T] check ...
        T = ids.shape[1]
        if T > self.cfg.n_ctx:   # wpe has no row past n_ctx
            raise ValueError(f"T = {T} exceeds the context n_ctx = {self.cfg.n_ctx}")
        x = self.drop(self.wpe(self.wte(ids)))   # token rows + position rows
        mask = causal_mask(T)  # position t reads tokens 0..t
        for block in self.h:
            x = block(x, mask)
        return self.ln_f(x)

    def forward(self, ids, targets=None):
        x = self.hidden(ids)
        if self.cfg.tie:
            # The output layer IS the embedding table: logits = x E^T.
            logits = F.matmul(x, F.transpose(self.wte.weight, 0, 1))
        else:
            logits = self.lm_head(x)
        # targets are already shifted by the caller: no shift here.
        loss = None if targets is None else cross_entropy(logits, np.asarray(targets).astype(np.int64))
        return logits, loss
```

With `tie=True` there is no `lm_head` attribute at all, so the `state_dict` has no duplicate key and the checkpoint shrinks by $Vd$.

### The causal LM loss on a full window

```python
def clm_loss(logits, ids, mask=None):
    ids = np.asarray(ids).astype(np.int64)
    # ... logits [B, T, V] must match ids [B, T] ...
    if ids.shape[1] < 2:
        raise ValueError("the causal LM loss needs T >= 2 (position t predicts token t + 1)")
    # Position t predicts token t + 1: drop the last logit row and the first id.
    targets = ids[:, 1:].copy()
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        # ...
        # The mask is applied to TARGETS (m[:, 1:]), not inputs.
        targets[~m[:, 1:]] = -100  # a padded target is not a prediction
    return cross_entropy(logits[:, :-1, :], targets, ignore_index=-100)
```

In the worked example, ids $(0, 1, 1)$ give targets $(1, 1)$ for logit rows 0 and 1. With the third token padded, `targets` becomes $(1, -100)$ and only row 0 counts, giving 0.287682.

### The GPT-2 loader

```python
def _hf_key(sd, name):
    # Accept both GPT2LMHeadModel ("transformer.h.0...") and bare GPT2Model ("h.0...").
    for k in ("transformer." + name, name):
        if k in sd:
            return np.asarray(sd[k], dtype=np.float32)
    raise KeyError(f"GPT-2 state dict has no {name!r} (with or without 'transformer.')")


def load_hf_gpt2(model, sd):
    c = model.cfg
    d = c.d_model
    out = {
        "wte.weight": _hf_key(sd, "wte.weight"),
        "wpe.weight": _hf_key(sd, "wpe.weight"),
        "ln_f.weight": _hf_key(sd, "ln_f.weight"),
        "ln_f.bias": _hf_key(sd, "ln_f.bias"),
    }
    for i in range(c.n_layers):
        p = f"h.{i}."
        # HF's Conv1D stores weight as [in, out] (x @ W): transpose to Linear's [out, in].
        w_attn = _hf_key(sd, p + "attn.c_attn.weight")
        b_attn = _hf_key(sd, p + "attn.c_attn.bias")
        # ... shape check: [d, 3d] ...
        # The 3 d output columns are q | k | v, each d wide.
        for j, n in enumerate(("q_proj", "k_proj", "v_proj")):
            out[f"{p}attn.{n}.weight"] = w_attn[:, j * d : (j + 1) * d].T
            out[f"{p}attn.{n}.bias"] = b_attn[j * d : (j + 1) * d]
        out[p + "attn.out_proj.weight"] = _hf_key(sd, p + "attn.c_proj.weight").T
        out[p + "attn.out_proj.bias"] = _hf_key(sd, p + "attn.c_proj.bias")
        out[p + "mlp_fc.weight"] = _hf_key(sd, p + "mlp.c_fc.weight").T
        out[p + "mlp_fc.bias"] = _hf_key(sd, p + "mlp.c_fc.bias")
        out[p + "mlp_proj.weight"] = _hf_key(sd, p + "mlp.c_proj.weight").T
        out[p + "mlp_proj.bias"] = _hf_key(sd, p + "mlp.c_proj.bias")
        for ln in ("ln_1", "ln_2"):
            out[f"{p}{ln}.weight"] = _hf_key(sd, f"{p}{ln}.weight")
            out[f"{p}{ln}.bias"] = _hf_key(sd, f"{p}{ln}.bias")
    if not c.tie:
        out["lm_head.weight"] = np.asarray(sd["lm_head.weight"], dtype=np.float32)
    # Strict load: every one of our keys must be filled, with the right shape.
    model.load_state_dict(out)
```

The loader builds a dict of exactly the keys the model has and reads only what it needs. The old buffers `attn.bias` and `attn.masked_bias` are never touched, and `lm_head.weight` (the same tensor as `wte` in a tied Hugging Face model) is ignored. A missing key raises with its name; a model built with the wrong `d` fails the `c_attn` shape check or the shape check in `load_state_dict`.

### Training and saving

`fit_gpt` is AdamW with GPT-style betas, a cosine schedule down to $0.1 \cdot$ `lr`, and gradient clipping:

```python
    opt = AdamW(list(model.parameters()), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    model.train()

    def loss_fn(m, b):
        return m(b["x"], b["y"])[1]   # TokenStream already shifted y by one

    for step in range(steps):
        x, y = stream.next_batch()
        opt.lr = cosine_with_warmup(step, warmup, steps, lr, 0.1 * lr)
        losses.append(train_step(model, {"x": x, "y": y}, loss_fn, opt, clip)["loss"])
```

`save_gpt` writes Hugging Face-style config names (`n_embd`, `n_layer`, `n_positions`) plus `tl_arch = "gpt"` and the tokenizer name; `load_gpt` reverses it. The full solution is in [`gpt.py`](../../../../course/solutions/python/tinyllm/obj/gpt.py).

### Why this design

A forward pass costs $O(L(T^2 d + T d\, d_{ff}) + TVd)$, and for small models with byte or BPE vocabularies the tied head is the single largest matmul. The parameter count is $Vd + n_{ctx} d + L(4d^2 + 2 d\, d_{ff})$ plus biases and norms, with no second $Vd$ for the head.

The solution keeps **separate q, k, v projections** rather than one fused `[3d, d]` Linear. That way the attention module from Part 5 is reused unchanged, and [LoRA](06-lora-pissa-and-merge.md) can target each projection by name. The price is the split in the loader. Putting **the shift in one place**, only in `clm_loss` and never in `forward`, keeps `TokenStream`'s pre-shifted batches and full-window evaluation consistent. **Strict loading** is preferred over "load what matches", because a silently missing tensor would leave random weights and produce plausible but wrong text. The one tradeoff accepted is that learned positions cap the context at `n_ctx`; Part 7 replaces them with [RoPE](../13-decoder-block/03-rope.md).

## Testing against Hugging Face

A correct solution has to survive the following checks.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked loss | unit | 0.490415, the padded 0.287682, the gradient | everyone agrees on the shift |
| agreement with Hugging Face GPT-2 | golden | a random tiny `GPT2LMHeadModel`: logits, loss, and the gradient of every HF tensor through the loader | real GPT-2 checkpoints load (the Llama-family loader does the same for SmolLM2) |
| the key variants HF ships | boundary | bare `GPT2Model` keys and old buffers load; a missing key is named; a wrong config is a shape error | checkpoints from many sources |
| logits are causal | property | changing tokens after $t$ leaves logits $0..t$ bitwise equal | training on all positions is honest |
| one block against the formula | differential | one block in float64 numpy with large weights | tanh GELU, pre-LN, both residuals |
| `forward` with targets equals `clm_loss` | property | `forward(x[:, :-1], x[:, 1:])` equals `clm_loss(x)` | the shift happens once |
| the head is tied | unit | no `lm_head`; logits $= h W_{te}^\top$; `tie=False` adds one | half the output parameters |
| GPT-2 init | statistical | residual projections at $0.02/\sqrt{2L}$, others 0.02, seeded | training starts stable at depth |
| save and load | unit | the model directory loads back exactly | the model zoo |
| bad arguments | boundary | windows past `n_ctx`, one-token loss, float ids | errors, not silent garbage |
| it learns | learning | 150 AdamW updates of a 2-layer byte GPT reach a reference held-out loss | the model learns language |

The model tests are in [`test_gpt.py`](../../../../course/solutions/learner-tests/L6.1/python/tests/l6-1-gpt/test_gpt.py). Their key idea is worth borrowing: build a GPT-2 state dict in Hugging Face's layout by hand, write the Hugging Face forward pass in float64 numpy from **that dict**, and compare it with the loaded model. The oracle never sees the loader's transposes, so it cannot share their bugs.

## What can go wrong

Most bugs in a GPT are silent: the model still trains, or still loads, and only its numbers are wrong. The most common ones, and how the solution avoids each, are these.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| a Conv1D weight loaded without the transpose | `c_proj` is square: no error, wrong logits | every Conv1D weight is loaded with `.T` |
| `c_attn` split into interleaved columns instead of three blocks | q, k, v mixed | contiguous slices `[:, j*d:(j+1)*d]` for $j = 0, 1, 2$ |
| no causal mask | loss near zero in training, nonsense when generating | `hidden` always passes `causal_mask(T)` to every block |
| no shift (each position predicts its own token), or the padding mask applied to inputs | the model learns to copy; padded targets counted | `clm_loss` pairs `logits[:, :-1]` with `ids[:, 1:]` and masks with `m[:, 1:]` |
| exact GELU | off by about $10^{-3}$: HF parity fails | `F.gelu(..., approximate="tanh")` |
| `ln_f` forgotten | logits too large; HF disagrees | `hidden` returns `self.ln_f(x)` |
| residual projections at the plain 0.02 | the stream's variance grows with depth | `out_proj` and `mlp_proj` use `scaled_residual_std(0.02, L)` |
| position embeddings never added | the model is a bag of tokens | `self.wpe(self.wte(ids))` adds the position rows |
| no gradient through the tied head | `wte` learns only from the input side | the head is a `matmul` on `self.wte.weight` itself, not a copy of its data |

> **Warning:** The missing transpose on attention's `c_proj` is the classic GPT-2 loading bug. The matrix is square, so nothing complains; the model simply produces fluent-looking nonsense. Only a value-level comparison with a reference catches it.

## Where this leads

The rest of this chapter builds on `GPT.hidden`. The [fine-tuning heads](05-fine-tuning-heads-and-the-linear-head-export.md) put sequence, token, and reward heads on it; [LoRA with PiSSA](06-lora-pissa-and-merge.md) attaches adapters to its attention projections; and the [evaluation harness and model zoo](07-lm-evaluation-harness-and-the-model-zoo.md) reports its `gpt` rows in bits per byte. Later, [the KV cache and `generate`](../21-inference/02-kv-cache-and-generate.md) adds the first incremental decoder to this model.

The production equivalents add a good deal around the same core. nanoGPT and Hugging Face's `GPT2LMHeadModel` add flash attention, weight-decay groups, and `torch.compile`. Hugging Face's `from_pretrained` adds sharded safetensors, dtype casting, and key-renaming hooks. Llama-style models replace learned `wpe` with RoPE, which removes the length cap of the table (see [RoPE](../13-decoder-block/03-rope.md)). And training systems such as llm.c and Megatron add mixed precision, gradient accumulation, and data parallelism, which [mixed precision and accumulation](../17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md) covers.

Further reading:

- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019), section 2.3.
- Karpathy, nanoGPT, `model.py`.
- `transformers/models/gpt2/modeling_gpt2.py` and `transformers/modeling_utils.py`.
- `karpathy/llm.c`.

## Key terms

- **Decoder-only model**: a single causal self-attention stack in which every position predicts the next token.
- **Causal LM loss**: mean cross-entropy of logits at position $t$ against the token at $t + 1$, over real targets.
- **Tied head**: an output layer that reuses the token embedding table, $z = h W_{te}^\top$.
- **`Conv1D`**: Hugging Face's GPT-2 projection, storing $W$ as `[in, out]` and computing $xW + b$.
- **Scaled residual init**: starting the residual-writing projections at $0.02/\sqrt{2L}$ so the stream's variance does not grow with depth.

## Summary

- A GPT is the Transformer decoder without cross-attention: one causal stack, so one forward pass trains on every position.
- The causal LM loss shifts by one, exactly once; the last position has no target and padded targets count for nothing.
- GPT-2's block is pre-LN with a tanh-approximated GELU and a final `ln_f`, and the output layer is the token table.
- Hugging Face stores GPT-2 projections as `Conv1D` with weight `[in, out]`: every weight is transposed on load, and `c_attn` is cut into three column blocks q, k, v.
- The two projections that write into the residual stream start at $0.02/\sqrt{2L}$.

## Review questions

1. Why does a missing transpose on `attn.c_proj` give no error, while a missing transpose on `mlp.c_fc` does?
   <details><summary>Answer</summary>`c_proj` is $d \times d$, so its transpose has the same shape and `load_state_dict` accepts it; the logits are simply wrong. `c_fc` is $[d, 4d]$ against `mlp_fc.weight` of $[4d, d]$, so the shape check fails.</details>
2. Why are the residual-writing projections initialized at $0.02/\sqrt{2L}$ and not $0.02/\sqrt L$?
   <details><summary>Answer</summary>Each block adds two terms to the stream (attention output and MLP output), so $L$ blocks add $2L$ terms. Scaling each by $1/\sqrt{2L}$ keeps the variance of their sum at about $0.02^2$ times a constant, independent of depth.</details>
3. `forward(ids, targets)` calls `cross_entropy` without shifting, while `clm_loss` shifts. Why is that not a contradiction?
   <details><summary>Answer</summary>They take different inputs. `forward` receives (inputs, targets) already offset by one (as `TokenStream` produces), so `forward(x[:, :-1], x[:, 1:])` equals `clm_loss` on the full window `x`. Shifting in both would predict two tokens ahead.</details>
4. In `clm_loss`, why is the mask sliced as `m[:, 1:]`?
   <details><summary>Answer</summary>Row $t$ predicts token $t + 1$, so whether row $t$ counts depends on whether token $t + 1$ is real. Masking with `m[:, :-1]` (the input side) would count predictions of padding and drop real targets.</details>
5. How does the tied head make `wte` learn from the output side?
   <details><summary>Answer</summary>The logits are `matmul(x, transpose(wte.weight))`, a differentiable op on the same `Tensor` the embedding reads. Backward accumulates the output-side gradient into `wte.weight.grad` alongside the input-side gradient from the embedding gather.</details>
6. Why does the test oracle build the Hugging Face forward pass from the Hugging Face-layout dict rather than from the loaded model?
   <details><summary>Answer</summary>So that the oracle never passes through the loader's transposes and splits. If it reused the loaded weights, a loader bug would appear identically on both sides and the comparison would pass.</details>
