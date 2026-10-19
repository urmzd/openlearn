# GPT decoder-only, causal LM loss, GPT-2 weight loading

## Overview

| | |
|---|---|
| **Module** | `L6.1` · build · Python · Pass 5 · 1.5 h reading |
| **Solution** | [`python/tinyllm/obj/gpt.py`](../../../../course/solutions/python/tinyllm/obj/gpt.py): `GPTConfig`, `Block`, `GPT` (`hidden`, `forward`), `clm_loss`, `load_hf_gpt2`, `fit_gpt`, `save_gpt`, `load_gpt`; model oracle tests in [`learner-tests/L6.1/.../test_gpt.py`](../../../../course/solutions/learner-tests/L6.1/python/tests/l6-1-gpt/test_gpt.py) |
| **Contract** | [`course/contracts/py/tinyllm/obj/gpt.pyi`](../../../../course/contracts/py/tinyllm/obj/gpt.pyi) |
| **Needs** | `L5.2` `causal_mask` · `L5.3` attention · `L5.4` `LearnedPE` · `L0.1` · `L0.2` · `L0.3` `cross_entropy` · `L0.4` layers · `L0.5` `train_step` · `L0.6` safetensors, `TokenStream` · `M10.3` `AdamW` · `M10.4` `cosine_with_warmup` · `M07.3` `normal_init`, `scaled_residual_std` · `M06.3` `PCG32` |
| **Used by** | `L6.5` sequence, token, and reward heads · `L6.7` the zoo's `gpt` rows · later `L6.6` LoRA targets, `L8.2`'s first KV cache |
| **Optional depth** | Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019), section 2.3; Karpathy, nanoGPT `model.py` |

## Key Takeaways

- A GPT is `L5.5`'s decoder without cross-attention: one causal stack, so every position predicts the next token and one forward pass trains on all of them.
- The causal LM loss shifts by one: logits at position $t$ against the token at $t + 1$; the last position has no target, and a padded target counts for nothing.
- GPT-2's block is pre-LN with a **tanh-approximated** GELU and a final `ln_f`; the output layer is the token table.
- Hugging Face's GPT-2 stores projections as `Conv1D`, weight `[in, out]`: every weight is transposed on load, and `c_attn` is cut into three column blocks q, k, v.
- The two projections that write into the residual stream start at $0.02/\sqrt{2L}$.

## How to read this chapter

Sections 1 to 3 reduce `L5.5` to a decoder-only stack, fix GPT-2's exact layout and key names, and work the causal LM loss by hand. Section 4 fixes the interface. Section 5 walks through the solution, including the weight loader, and explains why each choice is right. Try to predict each step before reading it.

---

## 1. Why now

The encoder-decoder of `L5.5` maps one sequence to another, but the system this course builds serves a language model: a model that continues text. Every token of a corpus is a training target for the prefix before it, and nothing needs to be encoded separately. GPT keeps exactly the half of `L5.5` that does this: the decoder, minus its cross-attention. The solution follows GPT-2's exact layout, so that a checkpoint trained by Hugging Face loads into it and gives the same logits. That loader is the bridge the rest of the course walks over: `L7.9` loads SmolLM2 the same way, and `L8.2` puts the first KV cache on this model.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $t_0 \dots t_{T-1}$ | the token ids of one window | `int[T]` |
| $V, d, L, H$ | vocabulary, width, layers, heads | `int` |
| $n_{ctx}$ | the longest window (rows of `wpe`) | `int` |
| $W_{te} \in \mathbb{R}^{V \times d}$ | token table `wte` | `float32[V, d]` |
| $W_{pe} \in \mathbb{R}^{n_{ctx} \times d}$ | learned positions `wpe` (`L5.4`) | `float32[n_ctx, d]` |
| $h^{(\ell)}_t$ | the residual stream at layer $\ell$, position $t$ | `[B, T, d]` |
| $z_t \in \mathbb{R}^V$ | logits at position $t$ | `[B, T, V]` |
| $\text{GELU}_{\tanh}(u)$ | $\tfrac12 u (1 + \tanh(\sqrt{2/\pi}(u + 0.044715 u^3)))$ | elementwise |

### 2.1 One stack, one mask

$$P(t_0 \dots t_{T-1}) = \prod_{t=1}^{T-1} P(t_t \mid t_0 \dots t_{t-1}) .$$

With the causal mask (`L5.2`), position $t$ attends to positions $0 \dots t$ only, so its output is a function of the prefix and can be trained to predict $t_{t+1}$. One forward pass computes all $T$ predictions at once; without the mask position $t$ could read $t_{t+1}$ and the loss would teach copying.

### 2.2 The causal LM loss

$$\mathcal{L} = \frac{1}{|K|} \sum_{t \in K} -\log \operatorname{softmax}(z_t)_{t_{t+1}}, \qquad K = \{ t < T - 1 : t_{t+1} \text{ is a real token} \}.$$

In code: logits `[:, :-1]` against ids `[:, 1:]`, with `-100` where the target is padding (`L0.3` ignores it). The shift happens **once**: either `clm_loss` shifts a full window, or the data already gives (inputs, targets) pairs shifted by one, as `L0.6`'s `TokenStream` does, and `forward(x, y)` uses them as they are.

### 2.3 The GPT-2 block

$$h \leftarrow h + \text{Attn}(\text{LN}_1(h)), \qquad h \leftarrow h + W_{proj}\,\text{GELU}_{\tanh}(W_{fc}\,\text{LN}_2(h)),$$

pre-LN (`L5.5`'s `norm="pre"`), then $\text{LN}_f$ after the last block and $z = \text{LN}_f(h) W_{te}^\top$: the output layer is the token table (tied). GPT-2 used the tanh approximation of GELU; the exact form differs by up to about $10^{-3}$ at $|u| \approx 2$, enough to fail a golden comparison.

**Init.** $\mathcal{N}(0, 0.02^2)$ for the tables and every `Linear` weight; biases 0; LayerNorms $(1, 0)$. Each block adds two outputs to the residual stream (`attn.out_proj`, `mlp_proj`), so after $L$ blocks the stream holds $2L$ such terms; their weights start at $0.02/\sqrt{2L}$ (`M07.3`'s `scaled_residual_std`) to keep its variance from growing with depth.

### 2.4 Loading GPT-2

Hugging Face's `GPT2LMHeadModel` names and shapes:

| HF key (after `transformer.`) | Shape | In the solution |
|---|---|---|
| `wte.weight`, `wpe.weight` | `[V, d]`, `[n_ctx, d]` | same names |
| `h.i.ln_1.*`, `h.i.ln_2.*`, `ln_f.*` | `[d]` | same names |
| `h.i.attn.c_attn.weight`, `.bias` | `[d, 3d]`, `[3d]` | `attn.q_proj`, `k_proj`, `v_proj`: columns $0..d$, $d..2d$, $2d..3d$, each **transposed** |
| `h.i.attn.c_proj.weight` | `[d, d]` | `attn.out_proj.weight` **transposed** |
| `h.i.mlp.c_fc.weight` | `[d, 4d]` | `mlp_fc.weight` **transposed** |
| `h.i.mlp.c_proj.weight` | `[4d, d]` | `mlp_proj.weight` **transposed** |

`Conv1D` computes $x W + b$ with $W$ `[in, out]`; `Linear` computes $x W^\top + b$ with $W$ `[out, in]`. The square `c_proj` of attention loads either way without a shape error, which is why only a comparison of values, not shapes, catches a missing transpose. Old checkpoints also carry the buffers `attn.bias` and `attn.masked_bias` (a stored causal mask): they are not weights, and the loader ignores them. `lm_head.weight` is the same tensor as `wte.weight`.

## 3. Worked example by hand

$V = 2$, ids $(0, 1, 1)$, logits rows $z_0 = (0, \ln 3)$, $z_1 = (0, 0)$, $z_2 = (5, -5)$.

- Position 0 predicts $t_1 = 1$: $\operatorname{softmax}(0, \ln 3) = (1/4, 3/4)$, loss $-\ln 0.75 = 0.287682$.
- Position 1 predicts $t_2 = 1$: $(1/2, 1/2)$, loss $\ln 2 = 0.693147$.
- Position 2 has no next token: dropped.

$\mathcal{L} = (0.287682 + 0.693147)/2 = 0.490415$. With the third token marked as padding, position 1's target is padding: $\mathcal{L} = 0.287682$, and the gradient on $z_0$ is $\operatorname{softmax}(z_0) - \text{onehot}(1) = (0.25, -0.25)$, zero on the other rows.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked loss | unit | section 3: 0.490415, the padded 0.287682, the gradient | everyone agrees on the shift |
| agreement with Hugging Face GPT-2 | golden | a random tiny `GPT2LMHeadModel`: logits, loss, and the gradient of every HF tensor through the loader | real GPT-2 checkpoints load (`L7.9` does the same for SmolLM2) |
| the key variants HF ships | boundary | bare `GPT2Model` keys and old buffers load; a missing key is named; a wrong config is a shape error | checkpoints from many sources |
| logits are causal | property | changing tokens after $t$ leaves logits $0..t$ bitwise equal | training on all positions is honest |
| one block against the formula | differential | one block in float64 numpy with large weights | tanh GELU, pre-LN, both residuals |
| `forward` with targets equals `clm_loss` | property | `forward(x[:, :-1], x[:, 1:])` equals `clm_loss(x)` | the shift happens once |
| the head is tied | unit | no `lm_head`; logits $= h W_{te}^\top$; `tie=False` adds one | half the output parameters |
| GPT-2 init | statistical | residual projections at $0.02/\sqrt{2L}$, others 0.02, seeded | training starts stable at depth |
| save and load | unit | the model directory loads back exactly | the zoo (`L6.7`) |
| bad arguments | boundary | windows past `n_ctx`, one-token loss, float ids | errors, not silent garbage |
| it learns | learning | 150 AdamW updates of a 2-layer byte GPT reach a reference held-out loss | the model learns language |

A model test suite is in [`test_gpt.py`](../../../../course/solutions/learner-tests/L6.1/python/tests/l6-1-gpt/test_gpt.py). Its key idea: build a GPT-2 state dict in Hugging Face's layout by hand, write the HF forward in float64 numpy from **that dict**, and compare with the loaded model, so the oracle never sees the loader's transposes.

## 5. The solution, step by step

### 5.1 From first principles

1. A language model is $\prod_t P(t_t \mid t_{<t})$. Every factor conditions only on a prefix, so a single causal self-attention stack can compute every factor in one pass: position $t$'s output predicts $t_{t+1}$. That is `L5.5`'s decoder with cross-attention deleted, because there is no source.
2. The loss must pair logits at $t$ with ids at $t + 1$. The shift is done in exactly one place: `clm_loss` shifts a full window; `forward(ids, targets)` assumes the caller already shifted (as `TokenStream` does). Doing it in both would predict two tokens ahead.
3. To load real GPT-2 weights the module must match GPT-2's computation exactly: pre-LN, tanh GELU, learned positions, final `ln_f`, tied head. Each of these is a numeric difference a golden comparison detects.
4. Naming follows HF where possible (`wte`, `wpe`, `h.i.ln_1`, `ln_f`) so the loader is mostly a rename; where the layouts differ (Conv1D `[in, out]` vs Linear `[out, in]`, fused `c_attn`), the loader transposes and splits.
5. Depth makes the residual stream a sum of $2L$ terms. With independent terms the variance adds, so each writing projection starts at std $0.02/\sqrt{2L}$ to keep the total near the single-term scale.
6. The tied head is a `matmul` against the transpose of the same `wte` tensor, so the gradient from the output side lands on the same parameter as the input side.

### 5.2 The code, annotated

**The block.** Pre-LN with both residuals and the GPT-2 init:

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

**The model.** `hidden` is the shared trunk the heads of `L6.5` reuse; `forward` adds the tied output layer:

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

With `tie=True` there is no `lm_head` attribute at all, so the `state_dict` has no duplicate key and checkpoint size drops by $Vd$.

**The causal LM loss on a full window.**

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

In the worked example, ids $(0, 1, 1)$ give targets $(1, 1)$ for logit rows 0 and 1; with the third token padded, `targets` becomes $(1, -100)$ and only row 0 counts, giving 0.287682.

**The GPT-2 loader.**

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

The loader builds a dict of exactly the keys the model has and reads only what it needs, so the old buffers `attn.bias` and `attn.masked_bias` are never touched, and `lm_head.weight` (the same tensor as `wte` in a tied HF model) is ignored. A missing key raises with its name; a model built with the wrong `d` fails the `c_attn` shape check or `load_state_dict`'s shape check.

**Training.** `fit_gpt` is AdamW with GPT-style betas, a cosine schedule to $0.1 \cdot$ `lr`, and clipping:

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

`save_gpt` writes HF-style config names (`n_embd`, `n_layer`, `n_positions`) plus `tl_arch = "gpt"` and the tokenizer name; `load_gpt` reverses it.

Full solution: [`gpt.py`](../../../../course/solutions/python/tinyllm/obj/gpt.py).

### 5.3 Why this solution

- **Cost.** A forward pass is $O(L(T^2 d + T d\, d_{ff}) + TVd)$; the tied head is the single largest matmul for small models with byte or BPE vocabularies. Parameters: $Vd + n_{ctx} d + L(4d^2 + 2 d\, d_{ff})$ plus biases and norms, with no second $Vd$ for the head.
- **Separate q, k, v projections** rather than one fused `[3d, d]` Linear: the attention module of `L5.3` is reused unchanged and LoRA (`L6.6`) can target each projection by name. The price is the split in the loader.
- **One shift, in one place.** Putting the shift only in `clm_loss` and never in `forward` keeps `TokenStream`'s pre-shifted batches and full-window evaluation consistent.
- **Strict loading** rather than "load what matches": a silently missing tensor would leave random weights and produce plausible but wrong text.
- **Tradeoff accepted.** Learned positions cap the context at `n_ctx`; Part 7 replaces them with RoPE.

### 5.4 Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a Conv1D weight loaded without the transpose | `c_proj` is square: no error, wrong logits | every Conv1D weight is loaded with `.T` |
| 2. `c_attn` split into interleaved columns instead of three blocks | q, k, v mixed | contiguous slices `[:, j*d:(j+1)*d]` for $j = 0, 1, 2$ |
| 3. no causal mask | loss near zero in training, nonsense when generating | `hidden` always passes `causal_mask(T)` to every block |
| 4. no shift (each position predicts its own token), or the padding mask applied to inputs | the model learns to copy; padded targets counted | `clm_loss` pairs `logits[:, :-1]` with `ids[:, 1:]` and masks with `m[:, 1:]` |
| 5. exact GELU | off by about $10^{-3}$: HF parity fails | `F.gelu(..., approximate="tanh")` |
| 6. `ln_f` forgotten | logits too large; HF disagrees | `hidden` returns `self.ln_f(x)` |
| 7. residual projections at the plain 0.02 | the stream's variance grows with depth | `out_proj` and `mlp_proj` use `scaled_residual_std(0.02, L)` |
| position embeddings never added | the model is a bag of tokens | `self.wpe(self.wte(ids))` adds the position rows |
| no gradient through the tied head | `wte` learns only from the input side | the head is a `matmul` on `self.wte.weight` itself, not a copy of its data |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.2` | `causal_mask` for every block |
| Back | `L5.3` | the attention of every block |
| Back | `L5.4` | `LearnedPE` is `wpe` |
| Back | `L0.1` | `Tensor` |
| Back | `L0.2` | the tanh GELU, `matmul` for the tied head |
| Back | `L0.3` | `cross_entropy` for both losses |
| Back | `L0.4` | `Linear`, `Embedding`, `LayerNorm` |
| Back | `L0.5` | `train_step` inside `fit_gpt` |
| Back | `L0.6` | safetensors model directory, `TokenStream` batches |
| Back | `M10.3` | `AdamW` |
| Back | `M10.4` | `cosine_with_warmup` |
| Back | `M07.3` | `normal_init`, `scaled_residual_std` |
| Back | `M06.3` | `PCG32` init stream |
| Forward | `L6.5` | sequence, token, and reward heads on `GPT.hidden` |
| Forward | `L6.6` | LoRA adapters on the attention projections |
| Forward | `L6.7` | the zoo's `gpt` rows (bits per byte) |
| Forward | `L8.2` | the first KV cache and `generate` |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `GPT` | nanoGPT, HF `GPT2LMHeadModel` | flash attention, weight decay groups, `torch.compile` | `karpathy/nanoGPT/model.py`; `transformers/models/gpt2/modeling_gpt2.py` |
| `load_hf_gpt2` | HF `from_pretrained` | sharded safetensors, dtype casting, key renaming hooks | `transformers/modeling_utils.py` |
| learned `wpe` | RoPE in Llama-style models | no length cap from the table (`L7.3`) | Part 7 |
| `fit_gpt` | llm.c, Megatron | mixed precision, gradient accumulation, data parallelism | `L11.1`; `karpathy/llm.c` |
